/*******************************************************************************
    Copyright (c) 2016-2024 NVIDIA Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

*******************************************************************************/

#include "uvm_linux.h"
#include "uvm_perf_events.h"
#include "uvm_perf_module.h"
#include "uvm_perf_prefetch.h"
#include "uvm_perf_utils.h"
#include "uvm_kvmalloc.h"
#include "uvm_va_block.h"
#include "uvm_va_range.h"
#include "uvm_test.h"
#include "uvm_bpf_struct_ops.h"

//
// Tunables for prefetch detection/prevention (configurable via module parameters)
//

// Enable/disable prefetch performance heuristics
static unsigned uvm_perf_prefetch_enable = 1;

// TODO: Bug 1778037: [uvm] Use adaptive threshold for page prefetching
#define UVM_PREFETCH_THRESHOLD_DEFAULT 51

// Percentage of children subregions that need to be resident in order to
// trigger prefetching of the remaining subregions
//
// Valid values 1-100
static unsigned uvm_perf_prefetch_threshold  = UVM_PREFETCH_THRESHOLD_DEFAULT;

#define UVM_PREFETCH_MIN_FAULTS_MIN     1
#define UVM_PREFETCH_MIN_FAULTS_DEFAULT 1
#define UVM_PREFETCH_MIN_FAULTS_MAX     20

// Minimum number of faults on a block in order to enable the prefetching
// logic
static unsigned uvm_perf_prefetch_min_faults = UVM_PREFETCH_MIN_FAULTS_DEFAULT;

// Module parameters for the tunables
module_param(uvm_perf_prefetch_enable, uint, S_IRUGO);
module_param(uvm_perf_prefetch_threshold, uint, S_IRUGO);
module_param(uvm_perf_prefetch_min_faults, uint, S_IRUGO);

static bool g_uvm_perf_prefetch_enable;
static unsigned g_uvm_perf_prefetch_threshold;
static unsigned g_uvm_perf_prefetch_min_faults;

void uvm_perf_prefetch_bitmap_tree_iter_init(const uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
                                             uvm_page_index_t page_index,
                                             uvm_perf_prefetch_bitmap_tree_iter_t *iter)
{
    UVM_ASSERT(bitmap_tree->level_count > 0);
    UVM_ASSERT_MSG(page_index < bitmap_tree->leaf_count,
                   "%zd vs %zd",
                   (size_t)page_index,
                   (size_t)bitmap_tree->leaf_count);

    iter->level_idx = bitmap_tree->level_count - 1;
    iter->node_idx  = page_index;
}

uvm_va_block_region_t uvm_perf_prefetch_bitmap_tree_iter_get_range(const uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
                                                                   const uvm_perf_prefetch_bitmap_tree_iter_t *iter)
{
    NvU16 range_leaves = uvm_perf_tree_iter_leaf_range(bitmap_tree, iter);
    NvU16 range_start = uvm_perf_tree_iter_leaf_range_start(bitmap_tree, iter);
    uvm_va_block_region_t subregion = uvm_va_block_region(range_start, range_start + range_leaves);

    UVM_ASSERT(iter->level_idx >= 0);
    UVM_ASSERT(iter->level_idx < bitmap_tree->level_count);

    return subregion;
}

NvU16 uvm_perf_prefetch_bitmap_tree_iter_get_count(const uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
                                                   const uvm_perf_prefetch_bitmap_tree_iter_t *iter)
{
    uvm_va_block_region_t subregion = uvm_perf_prefetch_bitmap_tree_iter_get_range(bitmap_tree, iter);

    return uvm_page_mask_region_weight(&bitmap_tree->pages, subregion);
}

// 核心预取区间计算器：基于当前发生缺页的页面索引，在一棵记录了“访问密度”的位图树中游走，
// 寻找满足预取阈值的最大连续区间。
static uvm_va_block_region_t compute_prefetch_region(uvm_page_index_t page_index, // 当前发生缺页的页索引
                                                     uvm_perf_prefetch_bitmap_tree_t *bitmap_tree, // 预取状态树
                                                     uvm_va_block_region_t max_prefetch_region) // 允许预取的最大绝对物理边界
{
    NvU16 counter; // 记录某个树节点（代表一段区间）内，已经被访问/驻留的页面数量
    uvm_perf_prefetch_bitmap_tree_iter_t iter; // 树遍历迭代器
    uvm_va_block_region_t prefetch_region = uvm_va_block_region(0, 0); // 初始建议区间为空
    enum uvm_bpf_action action;

    // ==========================================================
    // 机制 1：eBPF 终极拦截 (The BPF Hook)
    // ==========================================================
    // 在进行任何复杂计算之前，先呼叫 eBPF。
    // 这允许外部加载的自定义 BPF 脚本直接接管预取逻辑。
    action = uvm_bpf_call_gpu_page_prefetch(page_index, bitmap_tree,
                                            &max_prefetch_region, &prefetch_region);

    if (action == UVM_BPF_ACTION_BYPASS) {
        // 【模式 A：完全接管】
        // BPF 脚本说：“别算了，我已经把算好的区间写进 prefetch_region 里了。”
        // 驱动直接跳过所有底层 C 代码逻辑。
    } else if (action == UVM_BPF_ACTION_ENTER_LOOP) {
        // 【模式 B：步进式监控/修改】
        // BPF 脚本说：“你按你的逻辑遍历树，但每爬一个节点都要叫我一声，我要微调。”
        uvm_perf_prefetch_bitmap_tree_traverse_counters(counter,
                                                        bitmap_tree,
                                                        page_index - max_prefetch_region.first + bitmap_tree->offset,
                                                        &iter) {
            uvm_va_block_region_t subregion = uvm_perf_prefetch_bitmap_tree_iter_get_range(bitmap_tree, &iter);

            // 在树的每一个层级迭代时，都调用一次 BPF 回调。
            // BPF 可以通过 kfunc 动态修改 prefetch_region。
            (void)uvm_bpf_call_gpu_page_prefetch_iter(bitmap_tree,
                                               &max_prefetch_region, &subregion,
                                               counter, &prefetch_region);
        }
    } else {
        // ==========================================================
        // 机制 2：UVM 原生预测逻辑 (The Native Heuristic)
        // ==========================================================
        // 【模式 C：默认算法 (UVM_BPF_ACTION_DEFAULT)】
        // 这是 99.9% 情况下运行的核心算法。
        //
        // 算法原理：自底向上（从叶子到根）遍历这棵位图树。
        // 叶子节点代表 1 个页，上一层代表 2 个页，再上一层代表 4 个页...
        // 迭代器从当前缺页的那个叶子节点开始，不断向它的“父节点”扩张。
        uvm_perf_prefetch_bitmap_tree_traverse_counters(counter,
                                                        bitmap_tree,
                                                        page_index - max_prefetch_region.first + bitmap_tree->offset,
                                                        &iter) {
            // 获取当前树节点代表的区间（subregion）
            uvm_va_block_region_t subregion = uvm_perf_prefetch_bitmap_tree_iter_get_range(bitmap_tree, &iter);
            // 计算这个区间总共有多少页
            NvU16 subregion_pages = uvm_va_block_region_num_pages(subregion);

            UVM_ASSERT(counter <= subregion_pages);
            
            // 【核心数学公式：密度阈值判定】
            // counter 是这棵树记录的：在这个子区间里，已经有多少页是驻留的/被访问过的。
            // g_uvm_perf_prefetch_threshold 是全局设定的密度阈值（通常是个百分比，比如 50）。
            // 
            // 如果 (已访问页数 / 总页数) > 阈值百分比：
            // 即：counter * 100 > subregion_pages * threshold
            // 
            // 这意味着什么？
            // 意味着这片区域的“访问密度”非常高！既然你已经密集访问了这里面的一大半，
            // 那我干脆“顺水推舟”，把这个节点代表的【整个子区间】全都划入预取范围！
            if (counter * 100 > subregion_pages * g_uvm_perf_prefetch_threshold)
                prefetch_region = subregion; 
                // 注意：因为是自底向上遍历，subregion 会越来越大。
                // 只要高层的密度依然满足阈值，预取区间就会像滚雪球一样不断成倍扩张！
                // 一旦某一层密度不够了（比如进入了一大片未访问的荒地），遍历结束，保留上一次成功扩张的区间。
        }
    }

    // ==========================================================
    // 机制 3：物理边界截断 (Clamping)
    // ==========================================================
    // 树的逻辑区间是基于内部偏移量 (offset) 算出来的，可能超出了我们允许的最大物理边界。
    // 最后这一步纯粹是数学运算，把算出来的 prefetch_region“钳制 (Clamp)”在绝对安全的范围内，
    // 防止出现类似 prefetch_region.first < max_prefetch_region.first 这样的越界灾难。
    if (prefetch_region.outer) {
        // 恢复真实页索引坐标系
        prefetch_region.first += max_prefetch_region.first;
        
        // 处理下界越界
        if (prefetch_region.first < bitmap_tree->offset) {
            prefetch_region.first = bitmap_tree->offset;
        } else {
            prefetch_region.first -= bitmap_tree->offset;
            if (prefetch_region.first < max_prefetch_region.first)
                prefetch_region.first = max_prefetch_region.first;
        }

        // 处理上界越界
        prefetch_region.outer += max_prefetch_region.first;
        if (prefetch_region.outer < bitmap_tree->offset) {
            prefetch_region.outer = bitmap_tree->offset;
        } else {
            prefetch_region.outer -= bitmap_tree->offset;
            if (prefetch_region.outer > max_prefetch_region.outer)
                prefetch_region.outer = max_prefetch_region.outer;
        }
    }

    return prefetch_region;
}

static void grow_fault_granularity_if_no_thrashing(uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
                                                   uvm_va_block_region_t region,
                                                   uvm_page_index_t first,
                                                   const uvm_page_mask_t *faulted_pages,
                                                   const uvm_page_mask_t *thrashing_pages)
{
    if (!uvm_page_mask_region_empty(faulted_pages, region) &&
        (!thrashing_pages || uvm_page_mask_region_empty(thrashing_pages, region))) {
        UVM_ASSERT(region.first >= first);
        region.first = region.first - first + bitmap_tree->offset;
        region.outer = region.outer - first + bitmap_tree->offset;
        UVM_ASSERT(region.outer <= bitmap_tree->leaf_count);
        uvm_page_mask_region_fill(&bitmap_tree->pages, region);
    }
}

static void grow_fault_granularity(uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
                                   NvU64 big_page_size,
                                   uvm_va_block_region_t big_pages_region,
                                   uvm_va_block_region_t max_prefetch_region,
                                   const uvm_page_mask_t *faulted_pages,
                                   const uvm_page_mask_t *thrashing_pages)
{
    uvm_page_index_t pages_per_big_page = big_page_size / PAGE_SIZE;
    uvm_page_index_t page_index;

    // Migrate whole block if no big pages and no page in it is thrashing
    if (!big_pages_region.outer) {
        grow_fault_granularity_if_no_thrashing(bitmap_tree,
                                               max_prefetch_region,
                                               max_prefetch_region.first,
                                               faulted_pages,
                                               thrashing_pages);
        return;
    }

    // Migrate whole "prefix" if no page in it is thrashing
    if (big_pages_region.first > max_prefetch_region.first) {
        uvm_va_block_region_t prefix_region = uvm_va_block_region(max_prefetch_region.first, big_pages_region.first);

        grow_fault_granularity_if_no_thrashing(bitmap_tree,
                                               prefix_region,
                                               max_prefetch_region.first,
                                               faulted_pages,
                                               thrashing_pages);
    }

    // Migrate whole big pages if they are not thrashing
    for (page_index = big_pages_region.first;
         page_index < big_pages_region.outer;
         page_index += pages_per_big_page) {
        uvm_va_block_region_t big_region = uvm_va_block_region(page_index,
                                                               page_index + pages_per_big_page);

        grow_fault_granularity_if_no_thrashing(bitmap_tree,
                                               big_region,
                                               max_prefetch_region.first,
                                               faulted_pages,
                                               thrashing_pages);
    }

    // Migrate whole "suffix" if no page in it is thrashing
    if (big_pages_region.outer < max_prefetch_region.outer) {
        uvm_va_block_region_t suffix_region = uvm_va_block_region(big_pages_region.outer,
                                                                  max_prefetch_region.outer);

        grow_fault_granularity_if_no_thrashing(bitmap_tree,
                                               suffix_region,
                                               max_prefetch_region.first,
                                               faulted_pages,
                                               thrashing_pages);
    }
}

static void init_bitmap_tree_from_region(uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
                                         uvm_va_block_region_t max_prefetch_region,
                                         const uvm_page_mask_t *resident_mask,
                                         const uvm_page_mask_t *faulted_pages)
{
    if (resident_mask)
        uvm_page_mask_or(&bitmap_tree->pages, resident_mask, faulted_pages);
    else
        uvm_page_mask_copy(&bitmap_tree->pages, faulted_pages);

    // If we are using a subregion of the va_block, align bitmap_tree
    uvm_page_mask_shift_right(&bitmap_tree->pages, &bitmap_tree->pages, max_prefetch_region.first);

    bitmap_tree->offset = 0;
    bitmap_tree->leaf_count = uvm_va_block_region_num_pages(max_prefetch_region);
    bitmap_tree->level_count = ilog2(roundup_pow_of_two(bitmap_tree->leaf_count)) + 1;
}

static void update_bitmap_tree_from_va_block(uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
                                             uvm_va_block_t *va_block,
                                             uvm_va_block_context_t *va_block_context,
                                             uvm_processor_id_t new_residency,
                                             const uvm_page_mask_t *faulted_pages,
                                             uvm_va_block_region_t max_prefetch_region)

{
    NvU64 big_page_size;
    uvm_va_block_region_t big_pages_region;
    uvm_va_space_t *va_space;
    const uvm_page_mask_t *thrashing_pages;

    UVM_ASSERT(va_block);
    UVM_ASSERT(va_block_context);

    va_space = uvm_va_block_get_va_space(va_block);

    // Get the big page size for the new residency.
    // Assume 64K size if the new residency is the CPU or no GPU va space is
    // registered in the current process for this GPU.
    if (UVM_ID_IS_GPU(new_residency) &&
        uvm_processor_mask_test(&va_space->registered_gpu_va_spaces, new_residency)) {
        uvm_gpu_t *gpu = uvm_gpu_get(new_residency);

        big_page_size = uvm_va_block_gpu_big_page_size(va_block, gpu);
    }
    else {
        big_page_size = UVM_PAGE_SIZE_64K;
    }

    big_pages_region = uvm_va_block_big_page_region_subset(va_block, max_prefetch_region, big_page_size);

    // Adjust the prefetch tree to big page granularity to make sure that we
    // get big page-friendly prefetching hints
    if (big_pages_region.first - max_prefetch_region.first > 0) {
        bitmap_tree->offset = big_page_size / PAGE_SIZE - (big_pages_region.first - max_prefetch_region.first);
        bitmap_tree->leaf_count = uvm_va_block_region_num_pages(max_prefetch_region) + bitmap_tree->offset;

        UVM_ASSERT(bitmap_tree->offset < big_page_size / PAGE_SIZE);
        UVM_ASSERT(bitmap_tree->leaf_count <= PAGES_PER_UVM_VA_BLOCK);

        uvm_page_mask_shift_left(&bitmap_tree->pages, &bitmap_tree->pages, bitmap_tree->offset);

        bitmap_tree->level_count = ilog2(roundup_pow_of_two(bitmap_tree->leaf_count)) + 1;
    }

    thrashing_pages = uvm_perf_thrashing_get_thrashing_pages(va_block);

    // Assume big pages by default. Prefetch the rest of 4KB subregions within
    // the big page region unless there is thrashing.
    grow_fault_granularity(bitmap_tree,
                           big_page_size,
                           big_pages_region,
                           max_prefetch_region,
                           faulted_pages,
                           thrashing_pages);
}

// 核心掩码生成器：遍历当前批次中真正发生缺页的页面，利用“位图树（Bitmap Tree）”算法，
// 计算并拼凑出一个最终的“建议预取掩码”。
static void compute_prefetch_mask(uvm_va_block_region_t faulted_region, // 本批次真实缺页的边界范围
                                  uvm_va_block_region_t max_prefetch_region, // 允许预取的最大绝对边界（通常是整个 2MB 块的剩余部分）
                                  uvm_perf_prefetch_bitmap_tree_t *bitmap_tree, // 包含历史访问密度的预取算法状态树
                                  const uvm_page_mask_t *faulted_pages, // 真实缺页的位图掩码
                                  uvm_page_mask_t *out_prefetch_mask) // 输出参数：最终算出来的预取掩码
{
    uvm_page_index_t page_index;

    // 第一步：清空输出掩码，准备在一张白纸上绘制预测图
    uvm_page_mask_zero(out_prefetch_mask);

    // 第二步：核心循环
    // 遍历 faulted_pages 这个掩码里，所有值为 1 的位（也就是真实发生了缺页的页面）。
    for_each_va_block_page_in_region_mask(page_index, faulted_pages, faulted_region) {
        
        // 【算法核心呼叫】
        // 拿着当前这个发生缺页的页索引 (page_index)，去问底层的那棵树 (bitmap_tree)：
        // “既然用户访问了第 N 页，你觉得我接下来顺便把哪段区域搬过来比较好？”
        // compute_prefetch_region 会在树中进行查找（通常是找相邻的、高密度的未驻留块），
        // 并返回一个建议的连续区间 (region)。
        uvm_va_block_region_t region = compute_prefetch_region(page_index, bitmap_tree, max_prefetch_region);

        // 【绘制掩码】
        // 算法返回了一个区间（比如：建议预取第 N 到第 N+31 页）。
        // 我们用这个区间，把输出掩码 (out_prefetch_mask) 中对应的比特位全部涂黑（置为 1）。
        uvm_page_mask_region_fill(out_prefetch_mask, region);

        // ==========================================================
        // 【神来之笔：提前终止 (Early Out) 优化】
        // ==========================================================
        // 假设我们在一个 2MB 块里，第 5 页和第 6 页同时发生了缺页。
        // 循环跑到第 5 页时，算法一看：“哦，顺序访问！我建议你把从第 5 页到最后第 511 页全搬了！”
        // 此时，region.outer（建议区间的结束位置）就已经等于 max_prefetch_region.outer（最大允许边界）了。
        // 
        // 既然我们已经决定要把剩下的所有内存都搬过来了，
        // 那循环走到第 6 页时，还有必要再问一遍算法“第 6 页附近需要预取啥”吗？
        // 完全没必要！因为我们要预取的掩码已经满了！
        // 此时直接 break 跳出循环，省去了后续无意义的树查询计算，极大地节约了 CPU 时钟周期。
        if (region.outer == max_prefetch_region.outer)
            break;
    }
}

// 核心预取计算函数：通知预取算法即将发生的缺页迁移，并计算出建议的预取页面掩码。
// 注意：正如顶部的 TODO 注释所言，目前的 UVM 架构限制了在一个 2MB 的 VA Block 内，
// 只能向单一的处理器（Processor）进行预取。
static NvU32 uvm_perf_prefetch_prenotify_fault_migrations(uvm_va_block_t *va_block,
                                                          uvm_va_block_context_t *va_block_context,
                                                          uvm_processor_id_t new_residency, // 目标地
                                                          const uvm_page_mask_t *faulted_pages, // 真实缺页
                                                          uvm_va_block_region_t faulted_region, // 缺页范围
                                                          uvm_page_mask_t *prefetch_pages, // 输出参数：建议预取掩码
                                                          uvm_perf_prefetch_bitmap_tree_t *bitmap_tree) // 位图树状态
{
    const uvm_page_mask_t *resident_mask = NULL;
    const uvm_va_policy_t *policy = uvm_va_policy_get_region(va_block, faulted_region);
    uvm_va_block_region_t max_prefetch_region; // 允许预取的最大合法范围
    const uvm_page_mask_t *thrashing_pages = uvm_perf_thrashing_get_thrashing_pages(va_block); // 获取正在抖动的页面名单

    // ==========================================================
    // 机制 1：目标切换检测
    // ==========================================================
    // 如果这次缺页要搬往的处理器，和上次不一样了（比如上次搬给 GPU 0，这次搬给 GPU 1）
    // 预取算法立刻“清零历史信任度”。把连贯性计数器清零。
    if (!uvm_id_equal(va_block->prefetch_info.last_migration_proc_id, new_residency)) {
        va_block->prefetch_info.last_migration_proc_id = new_residency;
        va_block->prefetch_info.fault_migrations_to_last_proc = 0;
    }

    // ==========================================================
    // 机制 2：划定预取最大边界
    // ==========================================================
    // Compute the expanded region that prefetching is allowed from.
    if (uvm_va_block_is_hmm(va_block)) {
        max_prefetch_region = uvm_hmm_get_prefetch_region(va_block,
                                                          va_block_context->hmm.vma,
                                                          policy,
                                                          uvm_va_block_region_start(va_block, faulted_region));
    }
    else {
        max_prefetch_region = uvm_va_block_region_from_block(va_block);
    }

    uvm_page_mask_zero(prefetch_pages); // 清空输出结果，准备计算

    // 获取目标处理器上目前已经驻留的页面掩码
    if (UVM_ID_IS_CPU(new_residency) || va_block->gpus[uvm_id_gpu_index(new_residency)] != NULL)
        resident_mask = uvm_va_block_resident_mask_get(va_block, new_residency, NUMA_NO_NODE);

    // ==========================================================
    // 机制 3：史诗级优化 —— "首次触碰 (First-Touch)" 满载预取
    // ==========================================================
    // 这是一个极其极其重要的性能优化点！
    // 如果这个 2MB 的内存块目前是一片荒芜（没有任何处理器驻留数据，resident 为空），
    // 并且当前缺页要搬去的地方，刚好符合用户策略配置的 preferred_location（首选地）。
    // 驱动认为：“这绝对是程序刚用 cudaMallocManaged 分配完内存，正在做初始化填充！”
    // 此时根本不需要算什么树形结构，直接大笔一挥：【满载全拉！】
    // 把最大允许范围内的所有页面，全部加入预取名单。
    if (uvm_processor_mask_empty(&va_block->resident) &&
        uvm_id_equal(new_residency, policy->preferred_location)) {
        uvm_page_mask_region_fill(prefetch_pages, max_prefetch_region);
    }
    // ==========================================================
    // 机制 4：常规算法预测 (Bitmap Tree)
    // ==========================================================
    else {
        // 1. 初始化预取状态树
        init_bitmap_tree_from_region(bitmap_tree, max_prefetch_region, resident_mask, faulted_pages);

        // 2. 将当前 2MB Block 的历史访问状态（例如页表映射状态）喂给这棵树
        update_bitmap_tree_from_va_block(bitmap_tree, va_block, va_block_context, new_residency, faulted_pages, max_prefetch_region);

        // 3. 剔除毒瘤（抖动页面）：绝不用正在“疯狂搬运(Thrashing)”的页面去作为预测未来的输入依据。
        if (thrashing_pages)
            uvm_page_mask_andnot(&va_block_context->scratch_page_mask, faulted_pages, thrashing_pages);
        else
            uvm_page_mask_copy(&va_block_context->scratch_page_mask, faulted_pages);

        // 4. 执行核心预测算法！结果写入 prefetch_pages
        compute_prefetch_mask(faulted_region, max_prefetch_region, bitmap_tree, &va_block_context->scratch_page_mask, prefetch_pages);
    }

    // ==========================================================
    // 机制 5：结果的二次过滤 (黑名单审查)
    // ==========================================================

    // 黑名单 1：剔除本次真实缺页的页面。
    // （你已经在搬运名单里了，不需要被“预取”）
    uvm_page_mask_andnot(prefetch_pages, prefetch_pages, faulted_pages);

    // 黑名单 2：CPU Remap 陷阱规避
    // TODO: Bug 1765432。如果我们要把数据预取到 CPU，且这页数据目前已经在 CPU 上有读取映射了。
    // 如果强行预取，会触发一次毫无意义的、极其昂贵的操作系统级页表重映射（Remap）！
    // 这种重映射的开销往往比不预取还要大。所以，如果是搬往 CPU，把这些已经有 Read 权限的页面从预取名单里踢掉。
    if (UVM_ID_IS_CPU(new_residency) && !uvm_va_block_is_hmm(va_block)) {
        uvm_page_mask_and(&va_block_context->scratch_page_mask, resident_mask, &va_block->cpu.pte_bits[UVM_PTE_BITS_CPU_READ]);
        uvm_page_mask_andnot(prefetch_pages, prefetch_pages, &va_block_context->scratch_page_mask);
    }

    // 黑名单 3：剔除正在发生“内存抖动”的页面。
    // 抖动说明多方正在争抢，强行预取过去马上又会被抢走，纯属浪费总线带宽。
    if (thrashing_pages)
        uvm_page_mask_andnot(prefetch_pages, prefetch_pages, thrashing_pages);

    // ==========================================================
    // 机制 6：更新历史记录
    // ==========================================================
    // 记录这次向目标处理器真实搬运了多少页面。这个数字就是我们在上一个函数里看到的
    // `fault_migrations_to_last_proc >= g_uvm_perf_prefetch_min_faults` 这个门槛的判断依据。
    va_block->prefetch_info.fault_migrations_to_last_proc += uvm_page_mask_region_weight(faulted_pages, faulted_region);

    // 返回最终敲定的、需要被预取的页面数量
    return uvm_page_mask_weight(prefetch_pages);
}

bool uvm_perf_prefetch_enabled(uvm_va_space_t *va_space)
{
    if (!g_uvm_perf_prefetch_enable)
        return false;

    UVM_ASSERT(va_space);

    return va_space->test.page_prefetch_enabled;
}

void uvm_perf_prefetch_compute_ats(uvm_va_space_t *va_space,
                                   const uvm_page_mask_t *faulted_pages,
                                   uvm_va_block_region_t faulted_region,
                                   uvm_va_block_region_t max_prefetch_region,
                                   const uvm_page_mask_t *residency_mask,
                                   uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
                                   uvm_page_mask_t *out_prefetch_mask)
{
    UVM_ASSERT(faulted_pages);
    UVM_ASSERT(bitmap_tree);
    UVM_ASSERT(out_prefetch_mask);

    uvm_page_mask_zero(out_prefetch_mask);

    if (!uvm_perf_prefetch_enabled(va_space))
        return;

    init_bitmap_tree_from_region(bitmap_tree, max_prefetch_region, residency_mask, faulted_pages);

    compute_prefetch_mask(faulted_region, max_prefetch_region, bitmap_tree, faulted_pages, out_prefetch_mask);
}

// 计算并获取针对当前 VA Block 的预取提示 (Prefetch Hint)。
// 此函数会被缺页处理流程调用。它会分析当前的缺页情况，并结合历史迁移记录，
// 决定是否应该进行预取，以及哪些页面需要被预取。
void uvm_perf_prefetch_get_hint_va_block(uvm_va_block_t *va_block,
                                         uvm_va_block_context_t *va_block_context,
                                         uvm_processor_id_t new_residency, // 这些缺页即将搬往的处理器（如 GPU 0）
                                         const uvm_page_mask_t *faulted_pages, // 当前批次真实发生缺页的页面掩码
                                         uvm_va_block_region_t faulted_region, // 这些缺页覆盖的最小/最大页索引范围
                                         uvm_perf_prefetch_bitmap_tree_t *bitmap_tree, // 预取算法使用的内部状态树
                                         uvm_perf_prefetch_hint_t *out_hint) // 输出结果：预取建议
{
    uvm_va_space_t *va_space = uvm_va_block_get_va_space(va_block);
    uvm_page_mask_t *prefetch_pages = &out_hint->prefetch_pages_mask;
    NvU32 pending_prefetch_pages; // 记录算法建议预取的页面总数

    // 严苛的安全防御：确保外层调用者正确获取了 VA Space 读锁和当前 Block 的互斥锁
    uvm_assert_rwsem_locked(&va_space->lock);
    uvm_assert_mutex_locked(&va_block->lock);
    UVM_ASSERT(uvm_hmm_check_context_vma_is_valid(va_block, va_block_context->hmm.vma, faulted_region));

    // ==========================================================
    // 第一阶段：初始化与功能开关检查
    // ==========================================================
    // 默认不进行预取
    out_hint->residency = UVM_ID_INVALID;
    uvm_page_mask_zero(prefetch_pages);

    // 如果全局未开启预取功能，直接返回
    if (!uvm_perf_prefetch_enabled(va_space))
        return;

    // ==========================================================
    // 第二阶段：呼叫底层核心预测算法
    // ==========================================================
    // 通知底层的预测算法：“嘿，我又发生了缺页，它们要搬到 new_residency 去！”
    // 底层算法（通常基于位图树，类似于访问密度或顺序访问的识别）会更新它的历史记录，
    // 如果它觉得有规律可循，就会通过 prefetch_pages 参数返回一个“它建议预取”的页面掩码。
    pending_prefetch_pages = uvm_perf_prefetch_prenotify_fault_migrations(va_block,
                                                                          va_block_context,
                                                                          new_residency,
                                                                          faulted_pages,
                                                                          faulted_region,
                                                                          prefetch_pages,
                                                                          bitmap_tree);

    // ==========================================================
    // 第三阶段：预取规则审查（保守策略）
    // ==========================================================
    // 预测算法给出了建议，但我们不能盲目听从。
    // 只有当：
    // 1. 最近向这个目标处理器发生的缺页迁移次数，达到了最低门槛（g_uvm_perf_prefetch_min_faults），说明不是偶发访问。
    // 2. 预测算法确实给出了至少 1 页以上的预取建议。
    if (va_block->prefetch_info.fault_migrations_to_last_proc >= g_uvm_perf_prefetch_min_faults &&
        pending_prefetch_pages > 0) {
        
        bool changed = false;
        uvm_range_group_range_t *rgr;

        // 【最核心的边界审查逻辑：基于 Range Group】
        // 虚拟内存空间不仅被切分成 2MB 的 VA Block，还会被切分成由用户/驱动定义的更细粒度的 Range Group。
        // 规则：我们绝不跨越那些“没有发生任何缺页的 Range Group”进行预取。
        
        // 遍历当前 2MB Block 内覆盖的所有 Range Group
        uvm_range_group_for_each_range_in(rgr, va_space, va_block->start, va_block->end) {
            
            // 计算当前 Range Group 在这个 2MB Block 里的交集区域（起始页索引、结束页索引）
            uvm_va_block_region_t region = uvm_va_block_region_from_start_end(va_block,
                                                                              max(rgr->node.start, va_block->start),
                                                                              min(rgr->node.end, va_block->end));

            // 如果当前这块细分区域内，【没有发生任何真实的缺页】（faulted_pages 为空），
            // 但是预测算法却在这个区域内【建议了预取页面】（prefetch_pages 不为空）...
            if (uvm_page_mask_region_empty(faulted_pages, region) &&
                !uvm_page_mask_region_empty(prefetch_pages, region)) {
                
                // 【否决预取建议】
                // 驱动认为算法“预测得太激进”了。越界预取极有可能导致无用功（搬了不用）。
                // 所以，强制将这个区域内的预取建议掩码清零！
                uvm_page_mask_region_clear(prefetch_pages, region);
                changed = true; // 标记我们修改了算法的原始建议
            }
        }

        // ==========================================================
        // 第四阶段：最终裁决输出
        // ==========================================================
        // 如果刚才的审查删除了某些不合理的预取建议，我们需要重新统计剩下的还要预取多少页（计算 1 的个数）
        if (changed)
            pending_prefetch_pages = uvm_page_mask_weight(prefetch_pages);

        // 如果审查完之后，依然有页面需要预取，那就正式下发指令！
        // 把目标处理器的 ID 赋给 out_hint，外层调用者看到有效 ID 就会把这些页面一并加入搬运大队。
        if (pending_prefetch_pages > 0)
            out_hint->residency = va_block->prefetch_info.last_migration_proc_id;
    }
}

NV_STATUS uvm_perf_prefetch_init(void)
{
    g_uvm_perf_prefetch_enable = uvm_perf_prefetch_enable != 0;

    if (!g_uvm_perf_prefetch_enable)
        return NV_OK;

    if (uvm_perf_prefetch_threshold <= 100) {
        g_uvm_perf_prefetch_threshold = uvm_perf_prefetch_threshold;
    }
    else {
        UVM_INFO_PRINT("Invalid value %u for uvm_perf_prefetch_threshold. Using %u instead\n",
                       uvm_perf_prefetch_threshold,
                       UVM_PREFETCH_THRESHOLD_DEFAULT);

        g_uvm_perf_prefetch_threshold = UVM_PREFETCH_THRESHOLD_DEFAULT;
    }

    if (uvm_perf_prefetch_min_faults >= UVM_PREFETCH_MIN_FAULTS_MIN &&
        uvm_perf_prefetch_min_faults <= UVM_PREFETCH_MIN_FAULTS_MAX) {
        g_uvm_perf_prefetch_min_faults = uvm_perf_prefetch_min_faults;
    }
    else {
        UVM_INFO_PRINT("Invalid value %u for uvm_perf_prefetch_min_faults. Using %u instead\n",
                       uvm_perf_prefetch_min_faults,
                       UVM_PREFETCH_MIN_FAULTS_DEFAULT);

        g_uvm_perf_prefetch_min_faults = UVM_PREFETCH_MIN_FAULTS_DEFAULT;
    }

    return NV_OK;
}

NV_STATUS uvm_test_set_page_prefetch_policy(UVM_TEST_SET_PAGE_PREFETCH_POLICY_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);

    if (params->policy >= UVM_TEST_PAGE_PREFETCH_POLICY_MAX)
        return NV_ERR_INVALID_ARGUMENT;

    uvm_va_space_down_write(va_space);

    if (params->policy == UVM_TEST_PAGE_PREFETCH_POLICY_ENABLE)
        va_space->test.page_prefetch_enabled = true;
    else
        va_space->test.page_prefetch_enabled = false;

    uvm_va_space_up_write(va_space);

    return NV_OK;
}

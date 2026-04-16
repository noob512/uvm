/* SPDX-License-Identifier: GPL-2.0 */
/* 必须包含的头文件，vmlinux.h 包含了当前编译内核的所有数据结构，是 eBPF 开发的核心 */
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>     // 提供 eBPF 辅助函数，如 map 查找、打印等
#include <bpf/bpf_tracing.h>     // 提供 BPF_PROG 宏，用于定义 BPF 程序入口
#include <bpf/bpf_core_read.h>   // 提供 CO-RE (Compile Once – Run Everywhere) 的安全内存读取宏
#include "uvm_types.h"           // 项目自定义的头文件，定义了 NVIDIA UVM 的数据结构
#include "bpf_testmod.h"

/* 声明 eBPF 程序的许可证。GPL 是必须的，否则内核中的许多高级辅助函数会拒绝加载此程序 */
char _license[] SEC("license") = "GPL";

/* * 策略说明：
 *
 * 1. 预取比例 (percentage): 0-100
 * - 用户态程序会监控 PCIe 带宽。带宽充足时提高比例，带宽拥挤时降低比例。
 * - 100%: 顶格预取整个最大允许区域。
 * - 0%:   不预取。
 * - 50%:  预取最大允许区域的一半页面。
 *
 * 2. 预取方向 (direction):
 * - 0 (FORWARD) 向前: 预取缺页地址之后的页面。适合数组遍历 (0 -> N)。
 * - 1 (BACKWARD) 向后: 预取缺页地址之前的页面。适合逆向数组遍历 (N -> 0)。
 * - 2 (FORWARD_START): 忽略当前缺页地址，直接从允许区域的头部开始向前预取。
 *
 * 3. 绝对页数 (num_pages):
 * - 如果 >0，则忽略百分比逻辑，直接预取固定数量的页面。
 */

#define PREFETCH_FORWARD       0
#define PREFETCH_BACKWARD      1
#define PREFETCH_FORWARD_START 2

/* =========================================================================
 * BPF Maps 定义
 * 这些 Map 是内核空间和用户空间共享数据的内存区域。
 * 这里的类型是 BPF_MAP_TYPE_ARRAY，大小只有 1 (max_entries=1)，类似全局变量。
 * ========================================================================= */

/* BPF map: 存储预取百分比 (0-100) */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, u32);  // 值: 预取百分比
} prefetch_pct_map SEC(".maps");

/* BPF map: 存储预取方向 (0=forward, 1=backward, 2=forward_start) */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, u32);
} prefetch_direction_map SEC(".maps");

/* BPF map: 存储固定预取页数 (如果为 0 则使用百分比逻辑) */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, u32);
} prefetch_num_pages_map SEC(".maps");

/* =========================================================================
 * 辅助函数 (Helpers)
 * 内联函数，用于安全地从 Map 中读取用户态传递过来的配置
 * ========================================================================= */

static __always_inline unsigned int get_prefetch_percentage(void)
{
    u32 key = 0; // 数组索引 0
    u32 *pct = bpf_map_lookup_elem(&prefetch_pct_map, &key);

    if (!pct)
        return 100;  // 保护机制：如果读取失败，默认激进预取 100%

    return *pct;
}

static __always_inline unsigned int get_prefetch_direction(void)
{
    u32 key = 0;
    u32 *dir = bpf_map_lookup_elem(&prefetch_direction_map, &key);

    if (!dir)
        return PREFETCH_FORWARD;  /* 默认向前预取 */

    return *dir;
}

static __always_inline unsigned int get_prefetch_num_pages(void)
{
    u32 key = 0;
    u32 *num = bpf_map_lookup_elem(&prefetch_num_pages_map, &key);

    if (!num)
        return 0;  /* 默认返回 0，即依赖百分比计算 */

    return *num;
}

/* =========================================================================
 * 核心逻辑：缺页异常发生时的预取区域计算函数
 * ========================================================================= */
SEC("struct_ops/gpu_page_prefetch")
int BPF_PROG(gpu_page_prefetch,
             uvm_page_index_t page_index,                 /* 当前触发缺页异常的内存页索引 */
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,/* GPU 驱动提供的历史访问位图树(此处不使用) */
             uvm_va_block_region_t *max_prefetch_region,  /* GPU 驱动评估出的、当前允许安全预取的最大内存范围 */
             uvm_va_block_region_t *result_region)        /* [输出参数] 我们需要填充的最终决定预取的范围 */
{
    /* 1. 从 BPF Maps 中读取用户态下发的动态配置 */
    u32 pct = get_prefetch_percentage();
    u32 direction = get_prefetch_direction();
    u32 num_pages = get_prefetch_num_pages();

    /* 2. 读取 GPU 驱动提供的最大允许预取范围 [max_first, max_outer) */
    /* 注意：必须使用 BPF_CORE_READ 宏，防止内核版本升级导致结构体偏移量改变 */
    uvm_page_index_t max_first = BPF_CORE_READ(max_prefetch_region, first);
    uvm_page_index_t max_outer = BPF_CORE_READ(max_prefetch_region, outer);
    unsigned int total_pages = max_outer - max_first;

    /* 3. 边界处理：如果不允许预取，或者百分比被设为了 0 */
    if (pct == 0 || total_pages == 0) {
        /* 将预取结果范围设置为 [0, 0)，即不预取任何页面 */
        bpf_gpu_set_prefetch_region(result_region, 0, 0);
        return 1; /* UVM_BPF_ACTION_BYPASS: 告诉驱动我们已经处理完了，直接返回，跳过默认策略 */
    }

    /* 4. 计算到底要预取多少页 (prefetch_pages) */
    unsigned int prefetch_pages;
    if (num_pages > 0) {
        // 场景A: 用户指定了固定页数，直接使用（覆盖百分比逻辑）
        prefetch_pages = num_pages;
    } else if (pct >= 100) {
        // 场景B: 100% 预取，直接拉满
        prefetch_pages = total_pages;
    } else {
        // 场景C: 根据比例计算页数
        prefetch_pages = (total_pages * pct) / 100;
        if (prefetch_pages == 0)
            prefetch_pages = 1;  // 兜底机制：只要比例大于0，至少预取1页
    }

    uvm_page_index_t new_first, new_outer; // 最终要输出的起始页和结束页 [new_first, new_outer)

    /* 5. 根据方向，结合触发缺页的地址 (page_index) 计算预取范围 */
    if (direction == PREFETCH_BACKWARD) {
        /* --- 模式 1: 向后预取 (BACKWARD) --- */
        /* 取 page_index 之前的页面 */
        
        /* 如果触发缺页的地址比允许的最大起始地址还要靠前或相等，说明前面没东西可预取了 */
        if (page_index <= max_first) {
            bpf_gpu_set_prefetch_region(result_region, 0, 0);
            return 1;
        }

        new_outer = page_index; // 结束地址就是当前缺页地址

        /* 计算起始地址（注意防止下溢出减出负数/大整数） */
        if (page_index >= prefetch_pages) {
            new_first = page_index - prefetch_pages;
        } else {
            new_first = 0;
        }

        /* 范围裁剪：确保不要超出了驱动允许的边界 [max_first, max_outer) */
        if (new_first < max_first)
            new_first = max_first;
        if (new_outer > max_outer)
            new_outer = max_outer;

    } else if (direction == PREFETCH_FORWARD_START) {
        /* --- 模式 2: 从头向前预取 (FORWARD_START) --- */
        /* 不管在哪缺页，直接从允许区域的开头往后取 n 页 */
        new_first = max_first;
        new_outer = max_first + prefetch_pages;

        /* 范围裁剪 */
        if (new_outer > max_outer)
            new_outer = max_outer;

    } else {
        /* --- 模式 3: 向前预取 (FORWARD, 默认模式) --- */
        /* 取 page_index 之后的页面 */
        new_first = page_index + 1; // 起始地址是当前缺页地址的下一页

        /* 如果下一页已经到达或超过了允许的最大边界，说明后面没东西可预取了 */
        if (new_first >= max_outer) {
            bpf_gpu_set_prefetch_region(result_region, 0, 0);
            return 1;
        }

        /* 范围裁剪：起始地址不能小于允许的最小地址 */
        if (new_first < max_first)
            new_first = max_first;

        /* 计算结束地址 */
        new_outer = new_first + prefetch_pages;
        
        /* 范围裁剪：结束地址不能大于允许的最大边界 */
        if (new_outer > max_outer)
            new_outer = max_outer;
    }

    /* 6. 将计算好的最终范围写入输出参数 result_region 中 */
    bpf_gpu_set_prefetch_region(result_region, new_first, new_outer);

    /* 返回 BYPASS 标志，告诉驱动：本次缺页异常的预取区域已由本 eBPF 程序接管并计算完毕 */
    return 1; 
}

/* =========================================================================
 * 迭代预取钩子 (本策略不使用)
 * NVIDIA 默认策略由于太复杂，需要不断迭代更新预取区域。我们这里一步到位，所以忽略此钩子。
 * ========================================================================= */
SEC("struct_ops/gpu_page_prefetch_iter")
int BPF_PROG(gpu_page_prefetch_iter,
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
             uvm_va_block_region_t *max_prefetch_region,
             uvm_va_block_region_t *current_region,
             unsigned int counter,
             uvm_va_block_region_t *prefetch_region)
{
    return 0; // 由于前面的主函数返回了 BYPASS，这个迭代函数实际上不会被内核调用。
}

/* 用于测试触发的钩子函数 (占位符) */
SEC("struct_ops/gpu_test_trigger")
int BPF_PROG(gpu_test_trigger, const char *buf, int len)
{
    return 0;
}

/* =========================================================================
 * 结构体操作注册 (Struct Ops Registration)
 * 这是最关键的一步。它将上面写好的 C 函数指针打包成一个结构体 `gpu_mem_ops`。
 * 当用户态程序调用 libbpf 进行 Attach 时，内核就会把这个结构体替换掉 GPU 驱动原有的策略结构体，
 * 从而实现"不改内核源码、不重启系统"即可替换硬件底层调度策略的 eBPF 魔法。
 * ========================================================================= */
SEC(".struct_ops")
struct gpu_mem_ops uvm_ops_adaptive_sequential = {
    .gpu_test_trigger = (void *)gpu_test_trigger,
    .gpu_page_prefetch = (void *)gpu_page_prefetch,             // 替换预取主函数
    .gpu_page_prefetch_iter = (void *)gpu_page_prefetch_iter,   // 替换迭代函数
};
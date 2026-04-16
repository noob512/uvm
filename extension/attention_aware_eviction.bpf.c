/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Attention-Aware Eviction Policy (注意力感知的显存驱逐策略)
 *
 * 这是一套专为大语言模型（LLM）KV Cache 定制的显存调度方案：
 * - 低注意力分数的 KV block → 优先斩立决（move_head，移到 LRU 链表头）
 * - 高注意力分数的 KV block → 赐免死金牌（move_tail，移到 LRU 链表尾）
 * - 非 KV 显存块（如模型权重、CUDA 算子） → 采用传统的 T1 高频访问保护策略
 *
 * 数据面与控制面分离：
 * 分数来源：用户态的 daemon（score_bridge.py）会实时计算分数，
 * 并通过挂载在 /sys/fs/bpf/attention_score_map 的 BPF Map 传递给内核。
 *
 * Phase 1: 使用 StreamingLLM 的启发式规则打分 (首尾 Token 分数高，中间低)。
 * Phase 2: 将替换为 GPU 算子实时累加的真实 Attention Score。
 */

#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "uvm_types.h"
#include "bpf_testmod.h"

char _license[] SEC("license") = "GPL";

/* ========================================================================
 * Constants (常量与调优参数配置)
 * ======================================================================== */

/* 语义分级（Tier）定义 */
#define TIER_TRASH  0   // 垃圾层：中间的废话 Token，显存一满最先踢掉它
#define TIER_COOL   1   // 普通层：顺其自然，让底层 UVM 默认的 LRU 决定生死
#define TIER_HOT    2   // 核心层：Attention Sink 或最近的上下文，死死保住

/* 兜底策略（模型权重）的配置 */
#define T1_FREQ_THRESHOLD 3     // 访问超过 3 次即视为核心（T1）数据
#define COUNTER_SLOTS 16384     // 计数器数组大小（必须是 2 的幂）
#define COUNTER_MASK  (COUNTER_SLOTS - 1) 

/* 显存分页对齐参数 */
#define VA_SHIFT      21        // 虚拟地址右移 21 位。因为底层显存按 2MB (2^21) 大页管理，
                                // 这样可以把庞大的 64 位虚拟地址压缩成一个紧凑的 page_id。

/* Stats counter indices (遥测指标索引，用于监控内核态的策略执行分布) */
#define STAT_ACTIVATE_TOTAL   0  // 总共触发了多少次“寻找替死鬼”的判定
#define STAT_SCORE_HIT        1  // 成功命中 Python 打分表的次数 (说明是 KV Cache)
#define STAT_MOVE_HEAD_TRASH  2  // 成功把垃圾数据推上断头台的次数
#define STAT_MOVE_TAIL_HOT    3  // 成功给核心数据发免死金牌的次数
#define STAT_TIER_COOL        4  // 不做干预，交由默认 LRU 处理的次数
#define STAT_T1_PROTECT       5  // 触发模型权重高频保护的次数
#define STAT_SCORE_MISS       6  // 查不到分数的次数 (说明是非 KV 数据)
#define STAT_NUM_COUNTERS     8  // 计数器总数

/* ========================================================================
 * Data structures (数据结构)
 * ======================================================================== */

/* 注意力分数结构体。设计得极其紧凑（共 4 字节），以节省宝贵的内核内存空间 */
struct block_score {
    u16 attention_score;   /* 归一化到 [0, 65535] 的真实分数 (Phase 2 备用) */
    u8  tier;              /* 分级标识：TIER_TRASH / COOL / HOT */
    u8  flags;             /* 备用标记位，如 bit0 标记是否为 Sink Token */
};

/* ========================================================================
 * Maps (eBPF 内存映射表：打通内核与用户态的桥梁)
 * * eBPF Maps 是驻留在内核空间的数据结构。
 * 它们允许 eBPF 程序在多次调用之间保持状态，并允许用户态程序（如 Python/C++ 守护进程）
 * 与内核态的 eBPF 程序进行双向的数据读写和配置同步。
 * ======================================================================== */

/*
 * score_map: 由用户态 (如 score_bridge.py) 异步填充的控制面数据。
 * * 【设计意图】：这是一个用于维护内存块（通常是 GPU 显存或主存）评分的哈希表。
 * 用户态程序可能会根据各种指标（如访问频率、重要性等）计算出评分，
 * 并写入此 Map。eBPF 程序在内核侧读取此评分，以做出快速的拦截或调度决策。
 *
 * Key: page_id (VA >> 21) 
 * - 键是虚拟地址 (VA) 右移 21 位。
 * - 2^21 = 2MB，这意味着该系统按 2MB 的大页 (HugePage) 粒度来管理内存。
 * Value: 上述的 block_score 结构体 (包含该 2MB 内存块的评分、元数据等)。
 */
struct {
    /* 采用哈希表类型。适合键值范围大但实际存储较稀疏的场景，支持精确查找 */
    __uint(type, BPF_MAP_TYPE_HASH);
    
    /* 最大容量为 65536 个表项。
     * 配合 2MB 的 page 粒度，最多可管理 65536 * 2MB = 128GB 的内存/显存地址空间 */
    __uint(max_entries, 65536); 
    
    /* Key 的类型为 32 位无符号整数，足以存下 128GB 空间内的 page_id */
    __type(key, u32);
    
    /* Value 为自定义的 block_score 结构体，需在代码前文定义 */
    __type(value, struct block_score);
} score_map SEC(".maps"); /* SEC(".maps") 指示 BPF 加载器（如 libbpf）将此结构放入 ELF 文件的 maps 段中创建 */


/*
 * stats_map: 性能无损的遥测 (Telemetry) 看板。
 * * 【设计意图】：用于记录系统的全局统计信息（如各种事件的发生次数、命中率等）。
 * 因为这些统计在 eBPF 程序的每次触发（热路径，Hot path）中都会执行，
 * 所以必须保证极低的开销，坚决避免多核并发时的锁竞争。
 */
struct {
    /* 采用 PERCPU_ARRAY (每 CPU 数组) 类型。
     * 内核会为每个 CPU 核心分配一份独立的数据副本。当 eBPF 程序在某个 CPU 上运行时，
     * 只修改自己专属的那个计数器。绝对无锁 (Lock-free)，对缓存行友好 (Cache-line friendly)，性能极高。
     * 用户态读取时，会自动将所有 CPU 的值汇总。 */
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    
    /* 数组长度为 STAT_NUM_COUNTERS，通常是一个预定义的枚举值，代表需要统计的指标数量 */
    __uint(max_entries, STAT_NUM_COUNTERS);
    
    /* 数组的索引 (Key) 固定为 32 位整数 (0 到 STAT_NUM_COUNTERS-1) */
    __type(key, u32);
    
    /* Value 为 64 位无符号整数，防止高频累加导致计数器溢出 */
    __type(value, u64);
} stats_map SEC(".maps");


/*
 * access_counts: 针对非 KV (Key-Value) 数据的兜底频次计数器。
 * * 【设计意图】：这可能是一个用于实现缓存替换算法（如 LFU / LRU 变种）的热度统计表。
 * 用于追踪某些特定索引或内存槽位的访问频次。
 */
struct {
    /* 同样采用 PERCPU_ARRAY 类型，以保证在热路径中高频更新时的极致性能，避免锁竞争 */
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    
    /* 数组容量为 COUNTER_SLOTS，表示需要追踪的槽位总数 */
    __uint(max_entries, COUNTER_SLOTS);
    
    /* 索引类型为 32 位整数 */
    __type(key, u32);
    
    /* Value 仅为 8 位无符号整数 (u8，最大值 255)。
     * 采用 u8 是为了极致压缩内存占用（当 COUNTER_SLOTS 非常大时）。
     * 这暗示了该计数器可能是一个带衰减机制 (Decay) 的粗粒度热度计数，不需要精确到无限大。*/
    __type(value, u8);
} access_counts SEC(".maps");

/* ========================================================================
 * Helpers (辅助内联函数)
 * ======================================================================== */

/* BPF Verifier 黑魔法：如何安全地对内核指针做哈希运算 */
static __always_inline u32 chunk_hash(uvm_gpu_chunk_t *chunk)
{
    u64 ptr = 0;
    // BPF 验证器禁止直接对指针做算术运算，防越界。
    // 我们用 probe_read 把指针当作纯二进制数据拷贝到 ptr 变量中，骗过验证器。
    bpf_probe_read_kernel(&ptr, sizeof(ptr), &chunk);
    // 高中低位混合异或，最后按位与操作限定在数组大小内，避免昂贵的取模(%)运算。
    return (u32)((ptr >> 6) ^ (ptr >> 18)) & COUNTER_MASK;
}

/* 统计计数器自增的封装函数 */
static __always_inline void stat_inc(u32 idx)
{
    u64 *val = bpf_map_lookup_elem(&stats_map, &idx);
    if (val)
        (*val)++;
}

/* ========================================================================
 * Prefetch: always_max 激进预取策略
 * ======================================================================== */

/* * 发生缺页中断时，不要像挤牙膏一样一页一页拉取。
 * 直接将包含该页的整个 2MB 虚拟地址块全部拉进显存，最大化压榨 PCIe 带宽。
 */
/* ========================================================================
 * GPU Page Prefetcher (显存激进预取器)
 * * 【设计意图】：默认的 GPU 驱动在发生缺页中断时，通常比较“保守”（为了省显存，
 * 往往只按需拉取 4KB 或少量相邻页）。但在大模型/张量计算场景下，数据访问几乎都是
 * 大块连续的。如果小口拉取，会频繁触发 PCIe 延迟，导致计算核心严重停机等待（Stall）。
 * * * 本程序通过 eBPF 劫持驱动逻辑：发生缺页时，直接将包含该页的整个大块
 * （通常是 2MB）全部拉进显存，用极少的显存浪费换取对 PCIe 带宽的极限压榨。
 * ======================================================================== */

/*
 * SEC("struct_ops/...") 允许 eBPF 程序动态替换内核/驱动中结构体的函数指针。
 * 这里我们替换了 UVM 驱动默认的 gpu_page_prefetch 决策函数。
 */
SEC("struct_ops/gpu_page_prefetch")
int BPF_PROG(gpu_page_prefetch,
         uvm_page_index_t page_index,                   // 引发缺页的具体 4KB 页面索引
         uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,  // 预取位图树（记录哪些页已就绪）
         uvm_va_block_region_t *max_prefetch_region,    // 底层系统允许的最大预取物理边界
         uvm_va_block_region_t *result_region)          // [输出参数] 我们需要把最终决定拉取的范围写进这里
{
    /*
     * BPF_CORE_READ: eBPF 的核心黑魔法 CO-RE (Compile Once-Run Everywhere)。
     * 不同版本的 Linux 内核或 GPU 驱动中，uvm_va_block_region_t 结构体的字段偏移量可能会变。
     * 直接读取可能会导致系统崩溃。CORE 会在程序加载时，自动比对内核 BTF 信息，
     * 动态修正偏移量，保证这段代码能在各个版本的驱动上安全运行。
     */
    uvm_page_index_t max_first = BPF_CORE_READ(max_prefetch_region, first); // 获取允许的最大起始边界
    uvm_page_index_t max_outer = BPF_CORE_READ(max_prefetch_region, outer); // 获取允许的最大结束边界
    
    /*
     * 核心暴力美学：无视默认的智能预取算法，直接读取系统允许的最大边界，
     * 然后强行要求驱动把这个最大范围（通常是一个完整的 2MB HugePage 块）一次性全拉进显存。
     */
    bpf_gpu_set_prefetch_region(result_region, max_first, max_outer);
    
    /*
     * 返回 1 在 struct_ops 的语义中通常代表 BYPASS（拦截/覆盖）。
     * 它明确告诉 GPU 驱动：“不用执行你原有的保守预取决策了，直接按我写在 result_region 里的结果去搬数据。”
     */
    return 1; 
}

/*
 * 预取操作通常包含两个阶段：决策阶段（上面那个函数）和迭代执行阶段（本函数）。
 * 这里挂载到预取循环迭代器上的 eBPF 钩子。
 */
SEC("struct_ops/gpu_page_prefetch_iter")
int BPF_PROG(gpu_page_prefetch_iter, ...)
{
    /*
     * 返回 0 代表 DEFAULT（放行）。
     * 因为我们在前一个函数中已经“一锤定音”定死了预取范围，
     * 对于后续具体的页表遍历和硬件 DMA 映射过程，我们不需要再做任何特殊干预，
     * 直接让 GPU 驱动按照原本的底层硬件逻辑去干活即可。
     */
    return 0; 
}

/* ========================================================================
 * Eviction: 显存驱逐核心大脑
 * * 【设计意图】：接管 GPU 原生驱动（如 NVIDIA UVM）的显存 LRU 链表管理。
 * 结合大模型 (LLM) 的上下文语义特征（Python 用户态打分）和底层高频访问特征
 * （eBPF 内核态统计），实现“智能保核心、无情踢废话”的显存调度策略。
 * ======================================================================== */

SEC("struct_ops/gpu_block_activate")
int BPF_PROG(gpu_block_activate,
         uvm_pmm_gpu_t *pmm,       // GPU 物理内存管理结构
         uvm_gpu_chunk_t *chunk,   // 当前被激活的底层物理显存块
         struct list_head *list)   // 原生驱动的 LRU（最近最少使用）链表
{
    stat_inc(STAT_ACTIVATE_TOTAL); // 全局无锁计数器：记录激活事件总数

    // 1. 安全地顺藤摸瓜，拿到物理显存块对应的虚拟起始地址 (VA)
    // 使用 BPF_CORE_READ 保证在不同内核/驱动版本间的兼容性
    uvm_va_block_t *va_block = BPF_CORE_READ(chunk, va_block);
    if (!va_block) return 0; // 如果尚未映射虚拟地址，直接放行，交由原生驱动处理
    
    u64 chunk_va = BPF_CORE_READ(va_block, start);
    if (chunk_va == 0) return 0;

    /* ========== 轨道 1：语义感知驱逐 (针对 KV Cache) ========== */
    
    // 将虚拟地址右移，按 2MB 大页粒度压缩为 page_id，作为哈希表的 Key
    u32 page_id = (u32)(chunk_va >> VA_SHIFT);
    
    // O(1) 查表：从 score_map 中查询用户态 (Python) 异步写入的语义评分
    struct block_score *bs = bpf_map_lookup_elem(&score_map, &page_id);

    if (bs) {
        stat_inc(STAT_SCORE_HIT); // 命中语义评分表

        switch (bs->tier) {
        case TIER_TRASH:
            // 【垃圾数据】：Python 判定这段 KV Cache 已失去上下文价值。
            // 操作：将其移动到 LRU 链表的最前面（Head）。
            // 结果：当系统显存不足触发 OOM 驱逐时，它会第一个被踢出显存。
            bpf_gpu_block_move_head(chunk, list);
            stat_inc(STAT_MOVE_HEAD_TRASH);
            break;
        case TIER_HOT:
            // 【核心数据】：如 System Prompt 或关键实体的 KV Cache。
            // 操作：将其移动到 LRU 链表的最后面（Tail）。
            // 结果：赋予“免死金牌”，极其难以被系统回收，保证核心推理不卡顿。
            bpf_gpu_block_move_tail(chunk, list);
            stat_inc(STAT_MOVE_TAIL_HOT);
            break;
        default: /* TIER_COOL */
            // 【普通数据】：暂不干预，顺其自然。
            stat_inc(STAT_TIER_COOL);
            break;
        }
        // return 1 (BYPASS)：通知底层 UVM 驱动，eBPF 已经完成了排序，
        // 请放弃你原有的、无脑的默认 LRU 驱逐逻辑。
        return 1; 
    }

    /* ========== 轨道 2：频率兜底保护 (针对非 KV 数据) ========== */
    
    // 如果查不到分数，说明这是大模型的静态权重 (Weights) 或者是系统的可执行代码。
    // 这类数据绝对不能被轻易驱逐，否则会引发严重的 PCIe 倒腾延迟。
    stat_inc(STAT_SCORE_MISS);

    // 利用极速位运算算出哈希索引，避免 eBPF 验证器报错
    u32 idx = chunk_hash(chunk);
    
    // 查询极致压缩的 8 位无锁高频计数器
    u8 *count = bpf_map_lookup_elem(&access_counts, &idx);
    if (!count) return 0;

    u8 c = *count;
    // 计数器最多累加到 255 防止溢出，配合后台的 Decay(衰减) 机制体现“近期热度”
    if (c < 255) *count = c + 1;

    // 只要这段无分数的显存块近期被连续访问超过 T1_FREQ_THRESHOLD (如 3 次)，
    // 就认定其为核心权重/代码，强制移动到 LRU 尾部进行重点保护。
    if (c + 1 >= T1_FREQ_THRESHOLD) {
        bpf_gpu_block_move_tail(chunk, list);
        stat_inc(STAT_T1_PROTECT);
        return 1; /* BYPASS：拦截并覆盖原生逻辑 */
    }

    // 轨道 1 和轨道 2 都不满足条件，彻底放权，返回 0 走系统默认的 LRU 管理逻辑。
    return 0; 
}

/* ========================================================================
 * 零干预区域 (保持极简，避免性能损耗)
 * * 【设计意图】：在使用 struct_ops 替换内核结构体指针时，有些回调函数
 * 我们并不想魔改。为了保证内核热路径（Hot Path）的极致性能，必须提供
 * 极简的“桩函数（Stub）”，直接 return 0，产生最小的 CPU 开销。
 * ======================================================================== */

// 显存块被频繁访问时触发（极高频）。
// 我们的热度统计已经在 gpu_block_activate 中完成了，这里不需要再做任何事。
// 返回 0 (DEFAULT) 直接放行。
SEC("struct_ops/gpu_block_access")
int BPF_PROG(gpu_block_access, ...) { return 0; }

// GPU 驱动准备开始驱逐某个显存块前触发。
// 我们的 LRU 链表重排已经在激活阶段做完了，这里不需要在临死前再做干预。
// 返回 0 (DEFAULT) 直接放行。
SEC("struct_ops/gpu_evict_prepare")
int BPF_PROG(gpu_evict_prepare, ...) { return 0; }

// 用于用户态程序测试/触发 eBPF 逻辑的占位钩子，当前环境未启用。
SEC("struct_ops/gpu_test_trigger")
int BPF_PROG(gpu_test_trigger, const char *buf, int len) { return 0; }

/* ========================================================================
 * Struct ops registration (将所有钩子打包，替换内核底层函数表)
 * ======================================================================== */

SEC(".struct_ops")
struct gpu_mem_ops uvm_ops_attention_aware = {
    .gpu_test_trigger     = (void *)gpu_test_trigger,
    .gpu_page_prefetch    = (void *)gpu_page_prefetch,
    .gpu_page_prefetch_iter = (void *)gpu_page_prefetch_iter,
    .gpu_block_activate   = (void *)gpu_block_activate,
    .gpu_block_access     = (void *)gpu_block_access,
    .gpu_evict_prepare    = (void *)gpu_evict_prepare,
};
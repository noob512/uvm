/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Stride-based Prefetch Policy (基于步长的预取策略)
 *
 * 在页面级别检测步长访问模式并据此进行预取。
 * - 追踪每个VA块的访问历史
 * - 检测稳定的步长模式
 * - 根据步长预测并预取下一页或几页
 *
 * 置信度衰减：当步长发生变化时，置信度降低1而不是重置为0，
 * 这使得策略对偶尔出现的不规则访问具有鲁棒性。
 */

#include <vmlinux.h>           // 包含vmlinux.h是编写BPF程序的基础，它定义了内核符号和类型
#include <bpf/bpf_helpers.h>   // 包含BPF程序常用的辅助函数，如map操作、打印等
#include <bpf/bpf_tracing.h>   // 包含用于跟踪的BPF辅助函数，如BPF_KPROBE
#include <bpf/bpf_core_read.h> // 包含BPF CO-RE (Compile Once - Run Everywhere) 读取辅助函数，用于安全访问内核结构体字段
#include "uvm_types.h"         // 包含UVM (Unified Virtual Memory) 相关的类型定义
#include "bpf_testmod.h"       // 包含与内核模块交互的BPF定义，如struct gpu_mem_ops
#include "trace_helper.h"      // 可能包含一些通用的跟踪辅助函数

char _license[] SEC("license") = "GPL"; // 必需的许可证声明，使BPF程序可以访问内核辅助函数

/* 配置键：用于从配置map中查找不同参数 */
#define CONFIG_CONFIDENCE_THRESHOLD  0  /* 触发预取所需的最小置信度 */
#define CONFIG_PREFETCH_PAGES        1  /* 要预取的页面数量 */
#define CONFIG_MAX_STRIDE            2  /* 允许的最大步长 */

/* 配置map：用于动态调整策略参数。这是一个数组map，可以由用户态程序更新 */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY); // Map类型：数组
    __uint(max_entries, 8);           // 最大条目数
    __type(key, u32);                 // 键类型：u32
    __type(value, u64);               // 值类型：u64
} policy_config SEC(".maps");         // 放在maps段，由内核解析

/* 每个VA块的步长状态追踪结构体 */
struct stride_state {
    s32 last_page;       /* 上一次访问的页面索引 (-1 表示未初始化) */
    s32 stride;          /* 检测到的步长 */
    s32 confidence;      /* 置信度级别 (可以为负) */
    u32 total_faults;    /* 总页面错误次数 (统计信息) */
    u32 prefetch_count;  /* 发出的预取次数 (统计信息) */
    u32 stride_hits;     /* 步长匹配的次数 (统计信息) */
};

/* 每个VA块的步长追踪map: 以va_block指针为键，存储其对应的stride_state */
struct {
    __uint(type, BPF_MAP_TYPE_HASH); // Map类型：哈希表
    __uint(max_entries, 4096);       // 最大条目数
    __type(key, u64);                // 键类型：u64 (va_block指针的值)
    __type(value, struct stride_state); // 值类型：stride_state结构体
} stride_map SEC(".maps");

/* Per-CPU缓存：用于临时存储当前正在处理的VA块指针 */
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY); // Map类型：Per-CPU数组
    __uint(max_entries, 1);                  // 只有一个元素
    __type(key, u32);                        // 键类型
    __type(value, u64);                      // 值类型：va_block指针
} va_block_cache SEC(".maps");

/* 全局统计信息结构体 */
struct stride_stats {
    u64 total_faults;      // 总故障次数
    u64 prefetch_issued;   // 发出的预取次数
    u64 stride_detected;   // 检测到步长的次数
    u64 no_prefetch;       // 未执行预取的次数
};

/* 存储全局统计信息的map */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, struct stride_stats);
} global_stats SEC(".maps");

/* 辅助函数：获取配置值，如果未找到则返回默认值 */
static __always_inline u64 get_config(u32 key, u64 default_val)
{
    u64 *val = bpf_map_lookup_elem(&policy_config, &key); // 在配置map中查找
    return val ? *val : default_val; // 如果找到则返回值，否则返回默认值
}

/* 辅助函数：更新全局统计信息 */
static __always_inline void update_stats(bool prefetched)
{
    u32 key = 0;
    struct stride_stats *stats = bpf_map_lookup_elem(&global_stats, &key);
    if (stats) {
        __sync_fetch_and_add(&stats->total_faults, 1); // 原子加法更新计数器
        if (prefetched) {
            __sync_fetch_and_add(&stats->prefetch_issued, 1);
        } else {
            __sync_fetch_and_add(&stats->no_prefetch, 1);
        }
    }
}

/*
 * Hook: uvm_perf_prefetch_get_hint_va_block (通过kprobe挂载)
 * 此函数在内核函数`uvm_perf_prefetch_get_hint_va_block`被调用时触发。
 * 它的主要作用是捕获当前正在处理的`va_block`指针，并将其缓存起来，
 * 以便后续的`gpu_page_prefetch` BPF程序能够知道它正在处理哪个VA块。
 */
SEC("kprobe/uvm_perf_prefetch_get_hint_va_block") // 将此函数挂载为kprobe
int BPF_KPROBE(prefetch_get_hint_va_block,
               uvm_va_block_t *va_block,              // 函数的第一个参数
               void *va_block_context,
               u32 new_residency,
               void *faulted_pages,
               u32 faulted_region_packed,
               uvm_perf_prefetch_bitmap_tree_t *bitmap_tree)
{
    u32 key = 0;
    // 从per-cpu缓存map中获取当前CPU的缓存槽
    u64 *cached = bpf_map_lookup_elem(&va_block_cache, &key);
    if (cached) {
        *cached = (u64)va_block; // 将当前va_block指针存入缓存
    }
    return 0; // kprobe返回0表示正常继续执行原函数
}

/* 辅助函数：获取缓存的va_block指针 */
static __always_inline u64 get_cached_va_block(void)
{
    u32 key = 0;
    u64 *cached = bpf_map_lookup_elem(&va_block_cache, &key);
    return cached ? *cached : 0; // 如果缓存有效则返回指针，否则返回0
}

/* 辅助函数：计算s32类型的绝对值 */
static __always_inline s32 abs_s32(s32 x)
{
    return x < 0 ? -x : x;
}

/**
 * @brief 核心BPF程序：gpu_page_prefetch
 * @details 这是挂载到gpu_mem_ops结构体上`gpu_page_prefetch`钩子的BPF程序。
 * 当内核决定可能需要预取时，会调用这个程序。
 * 程序会分析访问模式，判断是否为步长模式，并设置预取区域。
 */
SEC("struct_ops/gpu_page_prefetch") // 挂载到struct_ops的gpu_page_prefetch钩子
int BPF_PROG(gpu_page_prefetch,
             uvm_page_index_t page_index,                          // 当前触发预取的页面索引
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,         // 用于分析访问模式的位图树
             uvm_va_block_region_t *max_prefetch_region,         // 内核建议的最大预取区域
             uvm_va_block_region_t *result_region)               // [out] BPF程序应填充此结构体来指定预取区域
{
    // 1. 获取当前VA块指针和配置参数
    u64 va_block_ptr = get_cached_va_block(); // 从缓存中获取VA块指针
    s32 confidence_threshold = (s32)get_config(CONFIG_CONFIDENCE_THRESHOLD, 2); // 获取置信度阈值
    u32 prefetch_pages = (u32)get_config(CONFIG_PREFETCH_PAGES, 2);             // 获取要预取的页数
    s32 max_stride = (s32)get_config(CONFIG_MAX_STRIDE, 128);                   // 获取最大步长限制

    /* 使用BPF CO-RE安全地读取max_prefetch_region的边界 */
    uvm_page_index_t max_first = BPF_CORE_READ(max_prefetch_region, first);
    uvm_page_index_t max_outer = BPF_CORE_READ(max_prefetch_region, outer);

    /* 默认行为：不进行预取。清空result_region */
    bpf_gpu_set_prefetch_region(result_region, 0, 0);

    // 2. 检查是否有有效的VA块指针
    if (va_block_ptr == 0) {
        update_stats(false); // 更新统计信息
        return 1; /* BYPASS: 告诉内核我们已经处理完毕，但没有请求预取 */
    }

    // 3. 查找或创建该VA块的步长状态
    struct stride_state *state = bpf_map_lookup_elem(&stride_map, &va_block_ptr);
    struct stride_state new_state = { // 新状态的默认初始化
        .last_page = -1,
        .stride = 0,
        .confidence = 0,
        .total_faults = 0,
        .prefetch_count = 0,
        .stride_hits = 0,
    };

    if (!state) { // 如果是第一次遇到这个VA块
        /* 首次见到这个va_block */
        new_state.last_page = (s32)page_index; // 记录第一个访问的页面
        new_state.total_faults = 1;
        // 将新的状态插入到stride_map中
        bpf_map_update_elem(&stride_map, &va_block_ptr, &new_state, BPF_ANY);
        update_stats(false); // 第一次访问不预取
        return 1; /* BYPASS, 首次访问不进行预取 */
    }

    /* 更新状态中的总故障计数 */
    __sync_fetch_and_add(&state->total_faults, 1);

    /* 检查是否是首次真实访问 (last_page为-1) */
    if (state->last_page < 0) {
        state->last_page = (s32)page_index;
        update_stats(false); // 不预取
        return 1; /* BYPASS */
    }

    // 4. 计算当前步长并更新状态
    s32 current_stride = (s32)page_index - state->last_page; // 计算当前步长

    /* 更新last_page为当前页面 */
    state->last_page = (s32)page_index;

    /* 处理步长为0的情况 (再次访问同一页) */
    if (current_stride == 0) {
        update_stats(false);
        return 1; /* BYPASS */
    }

    /* 检查步长是否在允许范围内 */
    if (abs_s32(current_stride) > max_stride) {
        /* 步长过大，降低置信度 */
        if (state->confidence > 0)
            state->confidence--;
        update_stats(false);
        return 1; /* BYPASS */
    }

    // 5. 更新步长和置信度
    if (current_stride == state->stride) {
        /* 步长匹配，增加置信度 */
        state->confidence++;
        __sync_fetch_and_add(&state->stride_hits, 1);
    } else {
        /* 步长改变，降低置信度并更新步长 */
        if (state->confidence > 0)
            state->confidence--; // 置信度衰减
        state->stride = current_stride; // 更新为新的步长
    }

    // 6. 判断是否应该执行预取
    if (state->confidence >= confidence_threshold) { // 如果置信度足够高
        /* 预测下一次访问的位置 */
        s32 predicted = (s32)page_index + state->stride;

        /* 计算预取区域 */
        s32 pf_first, pf_outer;

        if (state->stride > 0) {
            /* 正向步长 */
            pf_first = predicted; // 从预测位置开始
            pf_outer = predicted + (s32)prefetch_pages; // 结束位置
        } else {
            /* 负向步长 */
            pf_first = predicted - (s32)prefetch_pages + 1; // 开始位置
            pf_outer = predicted + 1; // 结束位置
        }

        /* 将预取区域限制在有效范围内 */
        if (pf_first < (s32)max_first)
            pf_first = (s32)max_first;
        if (pf_outer > (s32)max_outer)
            pf_outer = (s32)max_outer;
        if (pf_first < 0)
            pf_first = 0;

        /* 只有在区域有效时才进行预取 */
        if (pf_first < pf_outer) {
            /* 通过kfunc设置预取区域，这是BPF程序与内核通信的关键 */
            bpf_gpu_set_prefetch_region(result_region,
                                        (uvm_page_index_t)pf_first,
                                        (uvm_page_index_t)pf_outer);
            __sync_fetch_and_add(&state->prefetch_count, 1); // 更新预取计数

            // 打印调试信息
            bpf_printk("stride_prefetch: page=%d, stride=%d, conf=%d, pf=[%d,%d)\n",
                       page_index, state->stride, state->confidence, pf_first, pf_outer);

            update_stats(true); // 更新统计信息，表示进行了预取
            return 1; /* BYPASS: 告知内核我们已完成处理 */
        }
    }

    update_stats(false); // 未达到预取条件
    return 1; /* BYPASS */
}

/* 
 * 未使用的struct_ops钩子。
 * 我们在gpu_page_prefetch中处理了所有逻辑，因此此函数保持为空。
 */
SEC("struct_ops/gpu_page_prefetch_iter")
int BPF_PROG(gpu_page_prefetch_iter,
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
             uvm_va_block_region_t *max_prefetch_region,
             uvm_va_block_region_t *current_region,
             unsigned int counter,
             uvm_va_block_region_t *prefetch_region)
{
    return 0; // 返回0表示继续执行内核原生逻辑
}

/* 用于测试触发器的空实现 */
SEC("struct_ops/gpu_test_trigger")
int BPF_PROG(gpu_test_trigger, const char *buf, int len)
{
    return 0; // 什么都不做
}

/* 
 * 定义struct_ops map的实例。
 * 这是将BPF程序与内核中定义的`struct gpu_mem_ops`结构体关联起来的关键。
 * `.struct_ops`段由内核解析，将函数指针与BPF程序的入口点连接。
 */
SEC(".struct_ops")
struct gpu_mem_ops uvm_ops_stride = {
    .gpu_test_trigger = (void *)gpu_test_trigger,      // 将结构体中的函数指针指向具体的BPF程序
    .gpu_page_prefetch = (void *)gpu_page_prefetch,
    .gpu_page_prefetch_iter = (void *)gpu_page_prefetch_iter,
};
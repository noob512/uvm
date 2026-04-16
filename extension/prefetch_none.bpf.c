// SPDX-License-Identifier: GPL-2.0
// 该代码实现了一个名为 "none" 的eBPF策略，其核心功能是完全禁用GPU内存预取。

#include <vmlinux.h>        // 包含内核头文件
#include <bpf/bpf_helpers.h> // eBPF辅助函数
#include <bpf/bpf_tracing.h> // eBPF追踪功能
#include <bpf/bpf_core_read.h> // CO-RE读取功能
#include "uvm_types.h"      // UVM (Unified Virtual Memory) 类型定义
#include "bpf_testmod.h"    // 测试模块相关定义

// 定义许可证为GPL，这是加载eBPF程序的必要条件
char _license[] SEC("license") = "GPL";

/*
 * 预取禁用策略 (No Prefetch Policy)
 * 此策略通过挂钩预取相关的内核函数，阻止任何预取操作的发生。
 * 它会将预取区域设置为空，从而有效地禁用所有预取功能。
 */

/**
 * @brief 主要的预取决策点
 * 
 * 这是eBPF程序的核心入口，当内核认为需要进行预取时会被调用。
 * 本策略的实现是将预取结果区域设置为空，从而禁用本次预取。
 */
SEC("struct_ops/gpu_page_prefetch")
int BPF_PROG(gpu_page_prefetch,
             uvm_page_index_t page_index,                          // 触发预取的页面索引
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,         // 用于分析访问模式的位图树
             uvm_va_block_region_t *max_prefetch_region,          // 可能的最大预取区域
             uvm_va_block_region_t *result_region)                // 输出：最终确定的预取区域
{
    // 打印调试日志，记录正在为哪个页面禁用预取
    bpf_printk("BPF prefetch_none: Disabling prefetch for page_index=%u\n", page_index);

    /*
     * 将输出的预取区域设置为空。
     * `bpf_gpu_set_prefetch_region` 是一个辅助函数，用于修改 `result_region` 结构。
     * 当起始(first)和结束(outer)索引相等时，表示区域为空。
     */
    bpf_gpu_set_prefetch_region(result_region, 0, 0);

    /*
     * 返回1 (UVM_BPF_ACTION_BYPASS)，告诉内核我们已经处理了此次预取请求，
     * 并且跳过内核原有的默认预取计算逻辑。这确保了我们的空区域设置生效。
     */
    return 1; /* UVM_BPF_ACTION_BYPASS */
}

/**
 * @brief 预取位图树迭代钩子
 * 
 * 此函数在内核遍历预取位图树以决定预取哪些页面时被调用。
 * 在 "none" 策略中，我们不关心迭代过程，因此直接返回默认行为。
 */
SEC("struct_ops/gpu_page_prefetch_iter")
int BPF_PROG(gpu_page_prefetch_iter,
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,         // 位图树
             uvm_va_block_region_t *max_prefetch_region,          // 最大预取区域
             uvm_va_block_region_t *current_region,               // 当前正在检查的区域
             unsigned int counter,                                // 迭代计数器
             uvm_va_block_region_t *prefetch_region)              // 输出：本次迭代确定的预取区域
{
    // 在 "none" 策略中，此钩子不执行任何操作，直接返回默认值。
    return 0; /* UVM_BPF_ACTION_DEFAULT */
}

/**
 * @brief 一个废弃的测试触发器
 * 
 * 这是一个为旧版测试框架保留的桩函数（dummy function）。
 * 它没有实际功能，只是为了满足结构体定义而存在。
 */
SEC("struct_ops/gpu_test_trigger")
int BPF_PROG(gpu_test_trigger, const char *buf, int len)
{
    // 不执行任何操作，直接返回成功
    return 0;
}

/*
 * 定义一个名为uvm_ops_none的结构体实例，并将其放置在名为 ".struct_ops" 的ELF段中。
 * 这个结构体是一个“struct_ops”类型的eBPF程序集合，它会挂载到内核中
 * `gpu_mem_ops` 类型的结构上，从而替换掉内核中原有的预取相关函数实现。
 */
SEC(".struct_ops")
struct gpu_mem_ops uvm_ops_none = {
    .gpu_test_trigger = (void *)gpu_test_trigger,       // 挂载测试触发器函数
    .gpu_page_prefetch = (void *)gpu_page_prefetch,     // 挂载主预取决策函数
    .gpu_page_prefetch_iter = (void *)gpu_page_prefetch_iter, // 挂载预取迭代函数
};
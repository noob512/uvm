#ifndef _UVM_BPF_STRUCT_OPS_H
#define _UVM_BPF_STRUCT_OPS_H

#include "uvm_va_block_types.h"      // 包含虚拟地址块相关的类型定义
#include "uvm_perf_prefetch.h"       // 包含内存预取相关的类型定义
#include "uvm_pmm_gpu.h"             // 包含GPU物理内存管理(PMM)相关的类型定义

/* 定义BPF钩子函数可以返回的动作码，用于控制内核的后续行为 */
enum uvm_bpf_action {
    UVM_BPF_ACTION_DEFAULT = 0,       /* 使用内核的默认行为。 */
    UVM_BPF_ACTION_BYPASS = 1,        /* 跳过内核的默认行为，直接由BPF程序处理。 */
    UVM_BPF_ACTION_ENTER_LOOP = 2,    /* 用于特定场景，指示进入一个带有BPF钩子的迭代循环。 */
};

/* 
 * BPF struct_ops 生命周期管理函数的声明。
 * 这些函数通常在内核模块初始化和卸载时被调用。
 */
int uvm_bpf_struct_ops_init(void);    // 初始化BPF struct_ops框架
void uvm_bpf_struct_ops_exit(void);   // 退出/清理BPF struct_ops框架

/* 
 * 内存预取 (Prefetch) 相关BPF钩子的包装函数声明。
 * 这些函数是内核代码调用BPF程序的入口点，它们封装了对BPF程序的实际调用，
 * 并处理返回的动作码。
 */
enum uvm_bpf_action uvm_bpf_call_gpu_page_prefetch(
    uvm_page_index_t page_index,                             // 触发预取的页面索引
    uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,            // 用于分析访问模式的位图树
    uvm_va_block_region_t *max_prefetch_region,             // 可能的最大预取区域
    uvm_va_block_region_t *result_region);                  // [out] BPF程序输出的最终预取区域

enum uvm_bpf_action uvm_bpf_call_gpu_page_prefetch_iter(
    uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,            // 位图树
    uvm_va_block_region_t *max_prefetch_region,             // 最大预取区域
    uvm_va_block_region_t *current_region,                  // 当前正在检查的区域
    unsigned int counter,                                   // 迭代计数器
    uvm_va_block_region_t *prefetch_region);                // [out] 本次迭代的预取区域

/* 
 * GPU内存块置换 (Eviction) 相关BPF钩子的包装函数声明。
 * 这些函数用于在内存块的状态发生变化时，调用BPF程序来执行自定义的策略。
 */
void uvm_bpf_call_gpu_block_activate(
    uvm_pmm_gpu_t *pmm,                 // GPU物理内存管理器
    uvm_gpu_chunk_t *chunk,             // 被激活的GPU内存块
    struct list_head *list);            // 内存块所在的链表

enum uvm_bpf_action uvm_bpf_call_gpu_block_access(
    uvm_pmm_gpu_t *pmm,                 // GPU物理内存管理器
    uvm_gpu_chunk_t *chunk,             // 被访问的GPU内存块
    struct list_head *list);            // 内存块所在的链表

void uvm_bpf_call_gpu_evict_prepare(
    uvm_pmm_gpu_t *pmm,                 // GPU物理内存管理器
    struct list_head *va_block_used,      // 已使用的虚拟地址块列表
    struct list_head *va_block_unused);   // 未使用的虚拟地址块列表

#endif /* _UVM_BPF_STRUCT_OPS_H */
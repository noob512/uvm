/*******************************************************************************
    Copyright (c) 2015-2025 NVIDIA Corporation

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

#include "linux/sort.h"
#include "linux/kernel.h"
#include "nv_uvm_interface.h"
#include "uvm_common.h"
#include "uvm_linux.h"
#include "uvm_global.h"
#include "uvm_gpu_replayable_faults.h"
#include "uvm_hal.h"
#include "uvm_kvmalloc.h"
#include "uvm_tools.h"
#include "uvm_va_block.h"
#include "uvm_va_range.h"
#include "uvm_va_space.h"
#include "uvm_va_space_mm.h"
#include "uvm_procfs.h"
#include "uvm_perf_thrashing.h"
#include "uvm_gpu_non_replayable_faults.h"
#include "uvm_ats_faults.h"
#include "uvm_test.h"

// The documentation at the beginning of uvm_gpu_non_replayable_faults.c
// provides some background for understanding replayable faults, non-replayable
// faults, and how UVM services each fault type.

// The HW fault buffer flush mode instructs RM on how to flush the hardware
// replayable fault buffer; it is only used in Confidential Computing.
//
// Unless HW_FAULT_BUFFER_FLUSH_MODE_MOVE is functionally required (because UVM
// needs to inspect the faults currently present in the HW fault buffer) it is
// recommended to use HW_FAULT_BUFFER_FLUSH_MODE_DISCARD for performance
// reasons.
typedef enum
{
    // Flush the HW fault buffer, discarding all the resulting faults. UVM never
    // gets to see these faults.
    HW_FAULT_BUFFER_FLUSH_MODE_DISCARD,

    // Flush the HW fault buffer, and move all the resulting faults to the SW
    // fault ("shadow") buffer.
    HW_FAULT_BUFFER_FLUSH_MODE_MOVE,
} hw_fault_buffer_flush_mode_t;

#define UVM_PERF_REENABLE_PREFETCH_FAULTS_LAPSE_MSEC_DEFAULT 1000

// Lapse of time in milliseconds after which prefetch faults can be re-enabled.
// 0 means it is never disabled
static unsigned uvm_perf_reenable_prefetch_faults_lapse_msec = UVM_PERF_REENABLE_PREFETCH_FAULTS_LAPSE_MSEC_DEFAULT;
module_param(uvm_perf_reenable_prefetch_faults_lapse_msec, uint, S_IRUGO);

#define UVM_PERF_FAULT_BATCH_COUNT_MIN 1
#define UVM_PERF_FAULT_BATCH_COUNT_DEFAULT 256

// Number of entries that are fetched from the GPU fault buffer and serviced in
// batch
static unsigned uvm_perf_fault_batch_count = UVM_PERF_FAULT_BATCH_COUNT_DEFAULT;
module_param(uvm_perf_fault_batch_count, uint, S_IRUGO);

#define UVM_PERF_FAULT_REPLAY_POLICY_DEFAULT UVM_PERF_FAULT_REPLAY_POLICY_BATCH_FLUSH

// Policy that determines when to issue fault replays
static uvm_perf_fault_replay_policy_t uvm_perf_fault_replay_policy = UVM_PERF_FAULT_REPLAY_POLICY_DEFAULT;
module_param(uvm_perf_fault_replay_policy, uint, S_IRUGO);

#define UVM_PERF_FAULT_REPLAY_UPDATE_PUT_RATIO_DEFAULT 50

// Reading fault buffer GET/PUT pointers from the CPU is expensive. However,
// updating PUT before flushing the buffer helps minimizing the number of
// duplicates in the buffer as it discards faults that were not processed
// because of the batch size limit or because they arrived during servicing.
// If PUT is not updated, the replay operation will make them show up again
// in the buffer as duplicates.
//
// We keep track of the number of duplicates in each batch and we use
// UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT for the fault buffer flush after if the
// percentage of duplicate faults in a batch is greater than the ratio defined
// in the following module parameter. UVM_GPU_BUFFER_FLUSH_MODE_CACHED_PUT is
// used, otherwise.
static unsigned uvm_perf_fault_replay_update_put_ratio = UVM_PERF_FAULT_REPLAY_UPDATE_PUT_RATIO_DEFAULT;
module_param(uvm_perf_fault_replay_update_put_ratio, uint, S_IRUGO);

#define UVM_PERF_FAULT_MAX_BATCHES_PER_SERVICE_DEFAULT 20

#define UVM_PERF_FAULT_MAX_THROTTLE_PER_SERVICE_DEFAULT 5

// Maximum number of batches to be processed per execution of the bottom-half
static unsigned uvm_perf_fault_max_batches_per_service = UVM_PERF_FAULT_MAX_BATCHES_PER_SERVICE_DEFAULT;
module_param(uvm_perf_fault_max_batches_per_service, uint, S_IRUGO);

// Maximum number of batches with thrashing pages per execution of the bottom-half
static unsigned uvm_perf_fault_max_throttle_per_service = UVM_PERF_FAULT_MAX_THROTTLE_PER_SERVICE_DEFAULT;
module_param(uvm_perf_fault_max_throttle_per_service, uint, S_IRUGO);

static unsigned uvm_perf_fault_coalesce = 1;
module_param(uvm_perf_fault_coalesce, uint, S_IRUGO);

// Log replayable fault addresses as each HW fault entry is parsed.
// 0 = disabled (default), 1 = enabled.
static unsigned uvm_perf_fault_log_addresses;
module_param(uvm_perf_fault_log_addresses, uint, S_IRUGO | S_IWUSR);

// Destination for replayable fault address logs:
// 0 = kernel log (dmesg), 1 = ftrace buffer (trace_pipe).
static unsigned uvm_perf_fault_log_destination;
module_param(uvm_perf_fault_log_destination, uint, S_IRUGO | S_IWUSR);

// Log replayable fault counters after each serviced batch.
// 0 = disabled (default), 1 = enabled.
static unsigned uvm_perf_fault_log_counters;
module_param(uvm_perf_fault_log_counters, uint, S_IRUGO | S_IWUSR);

// Enable KV cache address-range classification for replayable fault stats.
// 0 = disabled (default), 1 = enabled.
static unsigned uvm_perf_fault_kv_range_enable;
module_param(uvm_perf_fault_kv_range_enable, uint, S_IRUGO | S_IWUSR);

// Inclusive KV cache virtual address range used for replayable fault
// classification. The values are expected to come from vLLM UVM address logs.
static unsigned long long uvm_perf_fault_kv_start;
module_param(uvm_perf_fault_kv_start, ullong, S_IRUGO | S_IWUSR);

static unsigned long long uvm_perf_fault_kv_end;
module_param(uvm_perf_fault_kv_end, ullong, S_IRUGO | S_IWUSR);

/**
 * 判断给定的缺页地址是否属于 KV Cache (Key-Value Cache) 追踪区域。
 * * 背景：在跑大语言模型 (LLM) 等 AI 负载时，KV Cache 会占用海量显存且访存极为频繁。
 * 驱动开发者为了专门分析这类特定内存的缺页性能，允许通过内核模块参数
 * 指定一段虚拟内存地址范围作为“监控特区”。
 *
 * @param fault_address 触发 GPU 缺页中断的实际虚拟地址。
 * @return true 如果地址落在配置的 KV Cache 范围内，否则返回 false。
 */
bool uvm_perf_fault_address_is_in_kv_cache(NvU64 fault_address)
{
    // ---------------------------------------------------------
    // 第一步：读取全局/模块配置参数
    // ---------------------------------------------------------
    // uvm_perf_fault_kv_start 和 uvm_perf_fault_kv_end 通常是可以通过 
    // sysfs 或驱动加载参数动态配置的全局变量。
    // 强制转换为无符号 64 位整数 (NvU64)，以匹配 64 位系统的虚拟地址长度。
    NvU64 start = (NvU64)uvm_perf_fault_kv_start;
    NvU64 end = (NvU64)uvm_perf_fault_kv_end;

    // ---------------------------------------------------------
    // 第二步：总开关检查 (Master Switch)
    // ---------------------------------------------------------
    // 如果监控 KV Cache 的功能根本没有被启用，直接返回 false。
    // 这避免了在常规用户场景下做无意义的地址比较，节省 CPU 周期。
    if (!uvm_perf_fault_kv_range_enable)
        return false;

    // ---------------------------------------------------------
    // 第三步：配置合法性校验（防御性编程）
    // ---------------------------------------------------------
    // 由于 start 和 end 通常是用户态通过接口传进来的参数，绝不能盲目信任。
    // 1. start == 0 或 end == 0: 地址未初始化，或者是无效的零页地址。
    // 2. end < start: 用户配置错误，把结束地址设得比起始地址还小。
    // 遇到这些非法配置，安全起见，直接判定为不在范围内。
    if (start == 0 || end == 0 || end < start)
        return false;

    // ---------------------------------------------------------
    // 第四步：核心边界判定
    // ---------------------------------------------------------
    // 检查目标地址是否位于 [start, end] 这个闭区间内。
    return fault_address >= start && fault_address <= end;
}

static NvU64 percent_x100_u64(NvU64 numerator, NvU64 denominator)
{
    if (denominator == 0)
        return 0;

    return (numerator * 10000ULL) / denominator;
}

/**
 * 记录单次可重放缺页中断（Replayable Fault）的详细硬件条目信息。
 * * @param parent_gpu        指向触发该中断的 GPU 实例的指针。
 * @param fault_entry       指向包含中断硬件底层详细信息的结构体指针（包含中断类型、来源等）。
 * @param fault_address_raw GPU 实际尝试访问并导致缺页的具体虚拟内存地址（未对齐的原始地址）。
 */
static void log_replayable_fault_entry(uvm_parent_gpu_t *parent_gpu,
                                       const uvm_fault_buffer_entry_t *fault_entry,
                                       NvU64 fault_address_raw)
{
    // 根据模块参数或全局配置决定日志的输出目的地。
    // 如果 uvm_perf_fault_log_destination == 1，使用内核的高性能追踪机制 (ftrace)。
    if (uvm_perf_fault_log_destination == 1) {
        // trace_printk 会将日志写入内核的 ring buffer，开销极小，适合高频中断场景。
        trace_printk("GPU名称:%s,精确原始地址=0x%llx,对齐到页地址=0x%llx,中断类型=%s,访问类型=%s\n",
                     uvm_parent_gpu_name(parent_gpu),                  // GPU 名称 (如 GPU-0)
                     fault_address_raw,                                // 导致中断的精确原始地址
                     fault_entry->fault_address,                       // 对齐到页边界的中断地址
                     uvm_fault_type_string(fault_entry->fault_type),   // 中断类型 (如 PDE 缺失, PTE 缺失等)
                     uvm_fault_access_type_string(fault_entry->fault_access_type)); // 访问类型 (如 Read, Write, Atomic)
                     // 下面四个参数是 NVIDIA GPU 内部硬件单元的标识，用于精准定位是哪个硬件模块触发了缺页：
                    //   utlb=%u gpc=%u client=%u ve=%u
                    // fault_entry->fault_source.utlb_id,                // 微型 TLB (Translation Lookaside Buffer) ID
                    //  fault_entry->fault_source.gpc_id,                 // 图形处理簇 (Graphics Processing Cluster) ID
                    //  fault_entry->fault_source.client_id,              // 内存客户端 ID (如 L1 缓存, 纹理单元等)
                    //  fault_entry->fault_source.ve_id);                 // 虚拟引擎 (Virtual Engine) ID
    }
    else {
        // 否则，回退使用标准的内核日志打印机制 (最终通常会通过 printk 写入 dmesg / syslog)。
        // 这种方式开销较大，在缺页频繁发生时可能会引发性能瓶颈或日志刷屏。
        UVM_INFO_PRINT("Replayable fault GPU %s: raw=0x%llx page=0x%llx type=%s access=%s utlb=%u gpc=%u client=%u ve=%u\n",
                       uvm_parent_gpu_name(parent_gpu),
                       fault_address_raw,
                       fault_entry->fault_address,
                       uvm_fault_type_string(fault_entry->fault_type),
                       uvm_fault_access_type_string(fault_entry->fault_access_type),
                       fault_entry->fault_source.utlb_id,
                       fault_entry->fault_source.gpc_id,
                       fault_entry->fault_source.client_id,
                       fault_entry->fault_source.ve_id);
    }
}

/**
 * 记录处理一批缺页中断后的统计计数器信息（主要用于监控去重效果）。
 * * GPU 拥有成千上万个并发线程。如果它们同时访问同一个未映射的内存页，
 * 会瞬间产生海量的、针对同一地址的缺页中断（称为 Fault Storm）。
 * 驱动程序通常会按批次 (batch) 处理这些中断，并进行去重 (deduplication)，以节省 CPU 处理时间。
 * * @param parent_gpu        指向当前 GPU 实例的指针。
 * @param batch_faults      当前处理批次中，硬件报告的缺页中断总数。
 * @param batch_duplicates  当前处理批次中，被识别为重复并过滤掉的中断数量。
 */
/**
 * 记录单次可重放缺页中断（Replayable Fault）的批量统计信息。
 * * @param parent_gpu          指向 GPU 实例的指针，用于获取全局累计的统计数据。
 * @param batch_faults        当前批次中包含的总缺页实例数（包含重复项）。
 * @param batch_duplicates    当前批次中被判定为重复的缺页实例数。
 * @param batch_kv_faults     当前批次中属于 KV 类别（通常指 Kernel 侧或特定内存区域）的总缺页数。
 * @param batch_kv_duplicates 当前批次中属于 KV 类的重复缺页数。
 */
static void log_replayable_fault_counters(uvm_parent_gpu_t *parent_gpu,
                                          NvU32 batch_faults,
                                          NvU32 batch_duplicates,
                                          NvU32 batch_kv_faults,
                                          NvU32 batch_kv_duplicates)
{
    // ---------------------------------------------------------
    // 第一步：计算去重后的真实缺页数量 (Deduplication)
    // ---------------------------------------------------------
    // 扣除重复项，得到真正需要进行底层内存分配和映射的有效缺页数。
    // 使用 >= 判断是为了防止由于极其罕见的统计竞态导致无符号整数下溢（Underflow）变成巨大正数。
    NvU32 batch_after_dedup = batch_faults >= batch_duplicates ? batch_faults - batch_duplicates : 0;
    NvU32 batch_kv_after_dedup = batch_kv_faults >= batch_kv_duplicates ? batch_kv_faults - batch_kv_duplicates : 0;
    
    // ---------------------------------------------------------
    // 第二步：计算占比百分比 (放大 100 倍的定点数)
    // ---------------------------------------------------------
    // percent_x100_u64 会计算比例并放大一定倍数（通常是 10000 倍，用来表示带两位小数的百分比）。
    // 例如：如果实际占比是 12.34%，该函数会返回整数 1234。
    // 这也是 Linux 内核开发中的标准做法，因为内核态通常禁止使用浮点数 (float/double)。
    NvU64 batch_kv_ratio_x100 = percent_x100_u64(batch_kv_faults, batch_faults);
    NvU64 batch_kv_after_dedup_ratio_x100 = percent_x100_u64(batch_kv_after_dedup, batch_after_dedup);

    // ---------------------------------------------------------
    // 第三步：抓取 GPU 全局生命周期内的累计统计数据
    // ---------------------------------------------------------
    // 除了当前这一批次的数据，还要将自 GPU 初始化以来的全局总数据也拉出来展示。
    NvU64 total_faults = parent_gpu->stats.num_replayable_faults;
    NvU64 total_duplicates = parent_gpu->fault_buffer.replayable.stats.num_duplicate_faults;
    NvU64 total_after_dedup = total_faults >= total_duplicates ? total_faults - total_duplicates : 0;
    
    NvU64 total_kv_faults = parent_gpu->fault_buffer.replayable.stats.num_kv_faults;
    NvU64 total_kv_duplicates = parent_gpu->fault_buffer.replayable.stats.num_kv_duplicate_faults;
    NvU64 total_kv_after_dedup = total_kv_faults >= total_kv_duplicates ? total_kv_faults - total_kv_duplicates : 0;
    
    NvU64 total_kv_ratio_x100 = percent_x100_u64(total_kv_faults, total_faults);
    NvU64 total_kv_after_dedup_ratio_x100 = percent_x100_u64(total_kv_after_dedup, total_after_dedup);

    // ---------------------------------------------------------
    // 第四步：输出日志
    // ---------------------------------------------------------
    if (uvm_perf_fault_log_destination == 1) {
        trace_printk("GPU名称 %s,本批次总缺页实例数=%u,去重后=%u,KV类的总缺页数=%u,去重后=%u,比例=%llu.%02llu%%,去重后=%llu.%02llu%%|| 总缺页数=%llu,去重后=%llu,kv总错误数=%llu,去重后=%llu,总kv比例=%llu.%02llu%%,去重后=%llu.%02llu%%\n",
                     
                     // 【基础信息】
                     uvm_parent_gpu_name(parent_gpu),  // 1. GPU 名称 (例如 "GPU-0")

                     // ==========================================
                     // 【当前批次 (Batch) 维度统计】 —— 仅针对刚刚处理完的这一批中断
                     // ==========================================
                     batch_faults,                     // 2. 本批次：总缺页实例数 (包含重复项)
                     batch_after_dedup,                // 4. 本批次：去重后的有效缺页数 (真实发生的独立缺页)
                     
                     batch_kv_faults,                  // 5. 本批次：KV类 (Kernel侧/特定区域) 的总缺页数
                     batch_kv_after_dedup,             // 7. 本批次：KV类 去重后的有效缺页数
                     
                     // 8 & 9. 本批次：KV类缺页占总缺页的百分比 (未去重) -> 例如 "12.34%"
                     (unsigned long long)(batch_kv_ratio_x100 / 100ULL),
                     (unsigned long long)(batch_kv_ratio_x100 % 100ULL),
                     
                     // 10 & 11. 本批次：KV类缺页占总缺页的百分比 (已去重) -> 例如 "15.00%"
                     (unsigned long long)(batch_kv_after_dedup_ratio_x100 / 100ULL),
                     (unsigned long long)(batch_kv_after_dedup_ratio_x100 % 100ULL),

                     // ==========================================
                     // 【全局累计 (Total) 维度统计】 —— 自 GPU 启动或驱动加载以来的历史总和
                     // ==========================================
                     (unsigned long long)total_faults,          // 12. 历史累计：总缺页实例数
                     (unsigned long long)total_after_dedup,     // 14. 历史累计：去重后的总有效缺页数
                     
                     (unsigned long long)total_kv_faults,       // 15. 历史累计：KV类 的总缺页数
                     (unsigned long long)total_kv_after_dedup,  // 17. 历史累计：KV类 去重后的总缺页数
                     
                     // 18 & 19. 历史累计：KV类缺页占历史总缺页的百分比 (未去重)
                     (unsigned long long)(total_kv_ratio_x100 / 100ULL),
                     (unsigned long long)(total_kv_ratio_x100 % 100ULL),
                     
                     // 20 & 21. 历史累计：KV类缺页占历史总缺页的百分比 (已去重)
                     (unsigned long long)(total_kv_after_dedup_ratio_x100 / 100ULL),
                     (unsigned long long)(total_kv_after_dedup_ratio_x100 % 100ULL));
    }
    else {
        // 如果配置为走普通内核日志，逻辑与上述完全相同，只是底层打印函数不同。
        UVM_INFO_PRINT("..."); 
    }
    //     // ---------------------------------------------------------
    // // 第四步：输出日志
    // // ---------------------------------------------------------
    // if (uvm_perf_fault_log_destination == 1) {
    //     trace_printk("GPU名称 %s: 本批次总缺页实例数=%u batch_duplicates=%u batch_after_dedup=%u batch_kv_faults=%u batch_kv_duplicates=%u batch_kv_after_dedup=%u batch_kv_ratio=%llu.%02llu%% batch_kv_after_dedup_ratio=%llu.%02llu%% total_faults=%llu total_duplicates=%llu total_after_dedup=%llu total_kv_faults=%llu total_kv_duplicates=%llu total_kv_after_dedup=%llu total_kv_ratio=%llu.%02llu%% total_kv_after_dedup_ratio=%llu.%02llu%%\n",
                     
    //                  // 【基础信息】
    //                  uvm_parent_gpu_name(parent_gpu),  // 1. GPU 名称 (例如 "GPU-0")

    //                  // ==========================================
    //                  // 【当前批次 (Batch) 维度统计】 —— 仅针对刚刚处理完的这一批中断
    //                  // ==========================================
    //                  batch_faults,                     // 2. 本批次：总缺页实例数 (包含重复项)
    //                  batch_duplicates,                 // 3. 本批次：重复的缺页实例数
    //                  batch_after_dedup,                // 4. 本批次：去重后的有效缺页数 (真实发生的独立缺页)
                     
    //                  batch_kv_faults,                  // 5. 本批次：KV类 (Kernel侧/特定区域) 的总缺页数
    //                  batch_kv_duplicates,              // 6. 本批次：KV类 的重复缺页数
    //                  batch_kv_after_dedup,             // 7. 本批次：KV类 去重后的有效缺页数
                     
    //                  // 8 & 9. 本批次：KV类缺页占总缺页的百分比 (未去重) -> 例如 "12.34%"
    //                  (unsigned long long)(batch_kv_ratio_x100 / 100ULL),
    //                  (unsigned long long)(batch_kv_ratio_x100 % 100ULL),
                     
    //                  // 10 & 11. 本批次：KV类缺页占总缺页的百分比 (已去重) -> 例如 "15.00%"
    //                  (unsigned long long)(batch_kv_after_dedup_ratio_x100 / 100ULL),
    //                  (unsigned long long)(batch_kv_after_dedup_ratio_x100 % 100ULL),

    //                  // ==========================================
    //                  // 【全局累计 (Total) 维度统计】 —— 自 GPU 启动或驱动加载以来的历史总和
    //                  // ==========================================
    //                  (unsigned long long)total_faults,          // 12. 历史累计：总缺页实例数
    //                  (unsigned long long)total_duplicates,      // 13. 历史累计：总重复缺页数
    //                  (unsigned long long)total_after_dedup,     // 14. 历史累计：去重后的总有效缺页数
                     
    //                  (unsigned long long)total_kv_faults,       // 15. 历史累计：KV类 的总缺页数
    //                  (unsigned long long)total_kv_duplicates,   // 16. 历史累计：KV类 的重复缺页数
    //                  (unsigned long long)total_kv_after_dedup,  // 17. 历史累计：KV类 去重后的总缺页数
                     
    //                  // 18 & 19. 历史累计：KV类缺页占历史总缺页的百分比 (未去重)
    //                  (unsigned long long)(total_kv_ratio_x100 / 100ULL),
    //                  (unsigned long long)(total_kv_ratio_x100 % 100ULL),
                     
    //                  // 20 & 21. 历史累计：KV类缺页占历史总缺页的百分比 (已去重)
    //                  (unsigned long long)(total_kv_after_dedup_ratio_x100 / 100ULL),
    //                  (unsigned long long)(total_kv_after_dedup_ratio_x100 % 100ULL));
    // }
    // else {
    //     // 如果配置为走普通内核日志，逻辑与上述完全相同，只是底层打印函数不同。
    //     UVM_INFO_PRINT("..."); 
    // }
}

// This function is used for both the initial fault buffer initialization and
// the power management resume path.
static void fault_buffer_reinit_replayable_faults(uvm_parent_gpu_t *parent_gpu)
{
    uvm_replayable_fault_buffer_t *replayable_faults = &parent_gpu->fault_buffer.replayable;

    // Read the current get/put pointers, as this might not be the first time
    // we take control of the fault buffer since the GPU was initialized,
    // or since we may need to bring UVM's cached copies back in sync following
    // a sleep cycle.
    replayable_faults->cached_get = parent_gpu->fault_buffer_hal->read_get(parent_gpu);
    replayable_faults->cached_put = parent_gpu->fault_buffer_hal->read_put(parent_gpu);

    // (Re-)enable fault prefetching
    if (parent_gpu->fault_buffer.prefetch_faults_enabled)
        parent_gpu->arch_hal->enable_prefetch_faults(parent_gpu);
    else
        parent_gpu->arch_hal->disable_prefetch_faults(parent_gpu);
}

// There is no error handling in this function. The caller is in charge of
// calling fault_buffer_deinit_replayable_faults on failure.
static NV_STATUS fault_buffer_init_replayable_faults(uvm_parent_gpu_t *parent_gpu)
{
    NV_STATUS status = NV_OK;
    uvm_replayable_fault_buffer_t *replayable_faults = &parent_gpu->fault_buffer.replayable;
    uvm_fault_service_batch_context_t *batch_context = &replayable_faults->batch_service_context;

    UVM_ASSERT(parent_gpu->fault_buffer.rm_info.replayable.bufferSize %
               parent_gpu->fault_buffer_hal->entry_size(parent_gpu) == 0);

    replayable_faults->max_faults = parent_gpu->fault_buffer.rm_info.replayable.bufferSize /
                                    parent_gpu->fault_buffer_hal->entry_size(parent_gpu);

    // Check provided module parameter value
    parent_gpu->fault_buffer.max_batch_size = max(uvm_perf_fault_batch_count,
                                                  (NvU32)UVM_PERF_FAULT_BATCH_COUNT_MIN);
    parent_gpu->fault_buffer.max_batch_size = min(parent_gpu->fault_buffer.max_batch_size,
                                                  replayable_faults->max_faults);

    if (parent_gpu->fault_buffer.max_batch_size != uvm_perf_fault_batch_count) {
        UVM_INFO_PRINT("Invalid uvm_perf_fault_batch_count value on GPU %s: %u. Valid range [%u:%u] Using %u instead\n",
                       uvm_parent_gpu_name(parent_gpu),
                       uvm_perf_fault_batch_count,
                       UVM_PERF_FAULT_BATCH_COUNT_MIN,
                       replayable_faults->max_faults,
                       parent_gpu->fault_buffer.max_batch_size);
    }

    batch_context->fault_cache = uvm_kvmalloc_zero(replayable_faults->max_faults * sizeof(*batch_context->fault_cache));
    if (!batch_context->fault_cache)
        return NV_ERR_NO_MEMORY;

    // fault_cache is used to signal that the tracker was initialized.
    uvm_tracker_init(&replayable_faults->replay_tracker);

    batch_context->ordered_fault_cache = uvm_kvmalloc_zero(replayable_faults->max_faults *
                                                           sizeof(*batch_context->ordered_fault_cache));
    if (!batch_context->ordered_fault_cache)
        return NV_ERR_NO_MEMORY;

    // This value must be initialized by HAL
    UVM_ASSERT(replayable_faults->utlb_count > 0);

    batch_context->utlbs = uvm_kvmalloc_zero(replayable_faults->utlb_count * sizeof(*batch_context->utlbs));
    if (!batch_context->utlbs)
        return NV_ERR_NO_MEMORY;

    batch_context->max_utlb_id = 0;

    status = uvm_rm_locked_call(nvUvmInterfaceOwnPageFaultIntr(parent_gpu->rm_device, NV_TRUE));
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to take page fault ownership from RM: %s, GPU %s\n",
                      nvstatusToString(status),
                      uvm_parent_gpu_name(parent_gpu));
        return status;
    }

    replayable_faults->replay_policy = uvm_perf_fault_replay_policy < UVM_PERF_FAULT_REPLAY_POLICY_MAX?
                                           uvm_perf_fault_replay_policy:
                                           UVM_PERF_FAULT_REPLAY_POLICY_DEFAULT;

    if (replayable_faults->replay_policy != uvm_perf_fault_replay_policy) {
        UVM_INFO_PRINT("Invalid uvm_perf_fault_replay_policy value on GPU %s: %d. Using %d instead\n",
                       uvm_parent_gpu_name(parent_gpu),
                       uvm_perf_fault_replay_policy,
                       replayable_faults->replay_policy);
    }

    replayable_faults->replay_update_put_ratio = min(uvm_perf_fault_replay_update_put_ratio, 100u);
    if (replayable_faults->replay_update_put_ratio != uvm_perf_fault_replay_update_put_ratio) {
        UVM_INFO_PRINT("Invalid uvm_perf_fault_replay_update_put_ratio value on GPU %s: %u. Using %u instead\n",
                       uvm_parent_gpu_name(parent_gpu),
                       uvm_perf_fault_replay_update_put_ratio,
                       replayable_faults->replay_update_put_ratio);
    }

    // Re-enable fault prefetching just in case it was disabled in a previous run
    parent_gpu->fault_buffer.prefetch_faults_enabled = parent_gpu->prefetch_fault_supported;

    fault_buffer_reinit_replayable_faults(parent_gpu);

    return NV_OK;
}

static void fault_buffer_deinit_replayable_faults(uvm_parent_gpu_t *parent_gpu)
{
    uvm_replayable_fault_buffer_t *replayable_faults = &parent_gpu->fault_buffer.replayable;
    uvm_fault_service_batch_context_t *batch_context = &replayable_faults->batch_service_context;

    if (batch_context->fault_cache) {
        UVM_ASSERT(uvm_tracker_is_empty(&replayable_faults->replay_tracker));
        uvm_tracker_deinit(&replayable_faults->replay_tracker);
    }

    if (parent_gpu->fault_buffer.rm_info.faultBufferHandle) {
        // Re-enable prefetch faults in case we disabled them
        if (parent_gpu->prefetch_fault_supported && !parent_gpu->fault_buffer.prefetch_faults_enabled)
            parent_gpu->arch_hal->enable_prefetch_faults(parent_gpu);
    }

    uvm_kvfree(batch_context->fault_cache);
    uvm_kvfree(batch_context->ordered_fault_cache);
    uvm_kvfree(batch_context->utlbs);
    batch_context->fault_cache         = NULL;
    batch_context->ordered_fault_cache = NULL;
    batch_context->utlbs               = NULL;
}

NV_STATUS uvm_parent_gpu_fault_buffer_init(uvm_parent_gpu_t *parent_gpu)
{
    NV_STATUS status = NV_OK;

    uvm_assert_mutex_locked(&g_uvm_global.global_lock);
    UVM_ASSERT(parent_gpu->replayable_faults_supported);

    status = uvm_rm_locked_call(nvUvmInterfaceInitFaultInfo(parent_gpu->rm_device,
                                                            &parent_gpu->fault_buffer.rm_info));
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to init fault buffer info from RM: %s, GPU %s\n",
                      nvstatusToString(status),
                      uvm_parent_gpu_name(parent_gpu));

        // nvUvmInterfaceInitFaultInfo may leave fields in rm_info populated
        // when it returns an error. Set the buffer handle to zero as it is
        // used by the deinitialization logic to determine if it was correctly
        // initialized.
        parent_gpu->fault_buffer.rm_info.faultBufferHandle = 0;
        goto fail;
    }

    status = fault_buffer_init_replayable_faults(parent_gpu);
    if (status != NV_OK)
        goto fail;

    if (parent_gpu->non_replayable_faults_supported) {
        status = uvm_parent_gpu_fault_buffer_init_non_replayable_faults(parent_gpu);
        if (status != NV_OK)
            goto fail;
    }

    return NV_OK;

fail:
    uvm_parent_gpu_fault_buffer_deinit(parent_gpu);

    return status;
}

// Reinitialize state relevant to replayable fault handling after returning
// from a power management cycle.
void uvm_parent_gpu_fault_buffer_resume(uvm_parent_gpu_t *parent_gpu)
{
    UVM_ASSERT(parent_gpu->replayable_faults_supported);

    fault_buffer_reinit_replayable_faults(parent_gpu);
}

void uvm_parent_gpu_fault_buffer_deinit(uvm_parent_gpu_t *parent_gpu)
{
    NV_STATUS status = NV_OK;

    uvm_assert_mutex_locked(&g_uvm_global.global_lock);

    if (parent_gpu->non_replayable_faults_supported)
        uvm_parent_gpu_fault_buffer_deinit_non_replayable_faults(parent_gpu);

    fault_buffer_deinit_replayable_faults(parent_gpu);

    if (parent_gpu->fault_buffer.rm_info.faultBufferHandle) {
        status = uvm_rm_locked_call(nvUvmInterfaceOwnPageFaultIntr(parent_gpu->rm_device, NV_FALSE));
        UVM_ASSERT(status == NV_OK);

        uvm_rm_locked_call_void(nvUvmInterfaceDestroyFaultInfo(parent_gpu->rm_device,
                                                               &parent_gpu->fault_buffer.rm_info));

        parent_gpu->fault_buffer.rm_info.faultBufferHandle = 0;
    }
}

bool uvm_parent_gpu_replayable_faults_pending(uvm_parent_gpu_t *parent_gpu)
{
    uvm_replayable_fault_buffer_t *replayable_faults = &parent_gpu->fault_buffer.replayable;

    UVM_ASSERT(parent_gpu->replayable_faults_supported);

    // Fast path 1: we left some faults unserviced in the buffer in the last
    // pass
    if (replayable_faults->cached_get != replayable_faults->cached_put)
        return true;

    // Fast path 2: read the valid bit of the fault buffer entry pointed by the
    // cached get pointer
    if (!parent_gpu->fault_buffer_hal->entry_is_valid(parent_gpu, replayable_faults->cached_get)) {
        // Slow path: read the put pointer from the GPU register via BAR0
        // over PCIe
        replayable_faults->cached_put = parent_gpu->fault_buffer_hal->read_put(parent_gpu);

        // No interrupt pending
        if (replayable_faults->cached_get == replayable_faults->cached_put)
            return false;
    }

    return true;
}

// Push a fault cancel method on the given client. Any failure during this
// operation may lead to application hang (requiring manual Ctrl+C from the
// user) or system crash (requiring reboot).
// In that case we log an error message.
//
// gpc_id and client_id aren't used if global_cancel is true.
//
// This function acquires both the given tracker and the replay tracker
static NV_STATUS push_cancel_on_gpu(uvm_gpu_t *gpu,
                                    uvm_gpu_phys_address_t instance_ptr,
                                    bool global_cancel,
                                    NvU32 gpc_id,
                                    NvU32 client_id,
                                    uvm_tracker_t *tracker)
{
    NV_STATUS status;
    uvm_push_t push;
    uvm_tracker_t *replay_tracker = &gpu->parent->fault_buffer.replayable.replay_tracker;

    UVM_ASSERT(tracker != NULL);

    status = uvm_tracker_add_tracker_safe(tracker, replay_tracker);
    if (status != NV_OK)
        return status;

    if (global_cancel) {
        status = uvm_push_begin_acquire(gpu->channel_manager,
                                        UVM_CHANNEL_TYPE_MEMOPS,
                                        tracker,
                                        &push,
                                        "Cancel targeting instance_ptr {0x%llx:%s}\n",
                                        instance_ptr.address,
                                        uvm_aperture_string(instance_ptr.aperture));
    }
    else {
        status = uvm_push_begin_acquire(gpu->channel_manager,
                                        UVM_CHANNEL_TYPE_MEMOPS,
                                        tracker,
                                        &push,
                                        "Cancel targeting instance_ptr {0x%llx:%s} gpc %u client %u\n",
                                        instance_ptr.address,
                                        uvm_aperture_string(instance_ptr.aperture),
                                        gpc_id,
                                        client_id);
    }

    UVM_ASSERT(status == NV_OK);
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to create push and acquire trackers before pushing cancel: %s, GPU %s\n",
                      nvstatusToString(status),
                      uvm_gpu_name(gpu));
        return status;
    }

    if (global_cancel)
        gpu->parent->host_hal->cancel_faults_global(&push, instance_ptr);
    else
        gpu->parent->host_hal->cancel_faults_targeted(&push, instance_ptr, gpc_id, client_id);

    // We don't need to put the cancel in the GPU replay tracker since we wait
    // on it immediately.
    status = uvm_push_end_and_wait(&push);

    UVM_ASSERT(status == NV_OK);
    if (status != NV_OK)
        UVM_ERR_PRINT("Failed to wait for pushed cancel: %s, GPU %s\n", nvstatusToString(status), uvm_gpu_name(gpu));

    // The cancellation is complete, so the input trackers must be complete too.
    uvm_tracker_clear(tracker);
    uvm_tracker_clear(replay_tracker);

    return status;
}

static NV_STATUS push_cancel_on_gpu_targeted(uvm_gpu_t *gpu,
                                             uvm_gpu_phys_address_t instance_ptr,
                                             NvU32 gpc_id,
                                             NvU32 client_id,
                                             uvm_tracker_t *tracker)
{
    return push_cancel_on_gpu(gpu, instance_ptr, false, gpc_id, client_id, tracker);
}

static NV_STATUS push_cancel_on_gpu_global(uvm_gpu_t *gpu, uvm_gpu_phys_address_t instance_ptr, uvm_tracker_t *tracker)
{
    UVM_ASSERT(!gpu->parent->smc.enabled);

    return push_cancel_on_gpu(gpu, instance_ptr, true, 0, 0, tracker);
}

// Volta implements a targeted VA fault cancel that simplifies the fault cancel
// process. You only need to specify the address, type, and mmu_engine_id for
// the access to be cancelled. Caller must hold the VA space lock for the access
// to be cancelled.
static NV_STATUS cancel_fault_precise_va(uvm_fault_buffer_entry_t *fault_entry,
                                         uvm_fault_cancel_va_mode_t cancel_va_mode)
{
    NV_STATUS status;
    uvm_gpu_va_space_t *gpu_va_space;
    uvm_va_space_t *va_space = fault_entry->va_space;
    uvm_gpu_t *gpu = fault_entry->gpu;
    uvm_gpu_phys_address_t pdb;
    uvm_push_t push;
    uvm_replayable_fault_buffer_t *replayable_faults = &gpu->parent->fault_buffer.replayable;
    NvU64 offset;

    UVM_ASSERT(gpu->parent->replayable_faults_supported);
    UVM_ASSERT(fault_entry->fatal_reason != UvmEventFatalReasonInvalid);
    UVM_ASSERT(!fault_entry->filtered);

    gpu_va_space = uvm_gpu_va_space_get(va_space, gpu);
    UVM_ASSERT(gpu_va_space);
    pdb = uvm_page_tree_pdb_address(&gpu_va_space->page_tables);

    // Record fatal fault event
    uvm_tools_record_gpu_fatal_fault(gpu->id, va_space, fault_entry, fault_entry->fatal_reason);

    status = uvm_push_begin_acquire(gpu->channel_manager,
                                    UVM_CHANNEL_TYPE_MEMOPS,
                                    &replayable_faults->replay_tracker,
                                    &push,
                                    "Precise cancel targeting PDB {0x%llx:%s} VA 0x%llx VEID %u with access type %s",
                                    pdb.address,
                                    uvm_aperture_string(pdb.aperture),
                                    fault_entry->fault_address,
                                    fault_entry->fault_source.ve_id,
                                    uvm_fault_access_type_string(fault_entry->fault_access_type));
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to create push and acquire replay tracker before pushing cancel: %s, GPU %s\n",
                      nvstatusToString(status),
                      uvm_gpu_name(gpu));
        return status;
    }

    // UVM aligns fault addresses to PAGE_SIZE as it is the smallest mapping
    // and coherence tracking granularity. However, the cancel method requires
    // the original address (4K-aligned) reported in the packet, which is lost
    // at this point. Since the access permissions are the same for the whole
    // 64K page, we issue a cancel per 4K range to make sure that the HW sees
    // the address reported in the packet.
    for (offset = 0; offset < PAGE_SIZE; offset += UVM_PAGE_SIZE_4K) {
        gpu->parent->host_hal->cancel_faults_va(&push, pdb, fault_entry, cancel_va_mode);
        fault_entry->fault_address += UVM_PAGE_SIZE_4K;
    }
    fault_entry->fault_address = UVM_PAGE_ALIGN_DOWN(fault_entry->fault_address - 1);

    // We don't need to put the cancel in the GPU replay tracker since we wait
    // on it immediately.
    status = uvm_push_end_and_wait(&push);
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to wait for pushed VA global fault cancel: %s, GPU %s\n",
                      nvstatusToString(status), uvm_gpu_name(gpu));
    }

    uvm_tracker_clear(&replayable_faults->replay_tracker);

    return status;
}

static NV_STATUS push_replay_on_gpu(uvm_gpu_t *gpu,
                                    uvm_fault_replay_type_t type,
                                    uvm_fault_service_batch_context_t *batch_context)
{
    NV_STATUS status;
    uvm_push_t push;
    uvm_replayable_fault_buffer_t *replayable_faults = &gpu->parent->fault_buffer.replayable;
    uvm_tracker_t *tracker = NULL;

    if (batch_context)
        tracker = &batch_context->tracker;

    status = uvm_push_begin_acquire(gpu->channel_manager, UVM_CHANNEL_TYPE_MEMOPS, tracker, &push,
                                    "Replaying faults");
    if (status != NV_OK)
        return status;

    gpu->parent->host_hal->replay_faults(&push, type);

    // Do not count REPLAY_TYPE_START_ACK_ALL's toward the replay count.
    // REPLAY_TYPE_START_ACK_ALL's are issued for cancels, and the cancel
    // algorithm checks to make sure that no REPLAY_TYPE_START's have been
    // issued using batch_context->replays.
    if (batch_context && type != UVM_FAULT_REPLAY_TYPE_START_ACK_ALL) {
        uvm_tools_broadcast_replay(gpu, &push, batch_context->batch_id, UVM_FAULT_CLIENT_TYPE_GPC);
        ++batch_context->num_replays;
    }

    uvm_push_end(&push);

    // Add this push to the GPU's replay_tracker so cancel can wait on it.
    status = uvm_tracker_add_push_safe(&replayable_faults->replay_tracker, &push);

    if (uvm_procfs_is_debug_enabled()) {
        if (type == UVM_FAULT_REPLAY_TYPE_START)
            ++replayable_faults->stats.num_replays;
        else
            ++replayable_faults->stats.num_replays_ack_all;
    }

    return status;
}

static NV_STATUS push_replay_on_parent_gpu(uvm_parent_gpu_t *parent_gpu,
                                           uvm_fault_replay_type_t type,
                                           uvm_fault_service_batch_context_t *batch_context)
{
    uvm_gpu_t *gpu = uvm_parent_gpu_find_first_valid_gpu(parent_gpu);

    if (gpu)
        return push_replay_on_gpu(gpu, type, batch_context);

    return NV_OK;
}

static void write_get(uvm_parent_gpu_t *parent_gpu, NvU32 get)
{
    uvm_replayable_fault_buffer_t *replayable_faults = &parent_gpu->fault_buffer.replayable;

    UVM_ASSERT(uvm_sem_is_locked(&parent_gpu->isr.replayable_faults.service_lock));

    // Write get on the GPU only if it's changed.
    if (replayable_faults->cached_get == get)
        return;

    replayable_faults->cached_get = get;

    // Update get pointer on the GPU
    parent_gpu->fault_buffer_hal->write_get(parent_gpu, get);
}

// In Confidential Computing GSP-RM owns the HW replayable fault buffer.
// Flushing the fault buffer implies flushing both the HW buffer (using a RM
// API), and the SW buffer accessible by UVM ("shadow" buffer).
//
// The HW buffer needs to be flushed first. This is because, once that flush
// completes, any faults that were present in the HW buffer have been moved to
// the shadow buffer, or have been discarded by RM.
static NV_STATUS hw_fault_buffer_flush_locked(uvm_parent_gpu_t *parent_gpu, hw_fault_buffer_flush_mode_t flush_mode)
{
    NV_STATUS status;
    NvBool is_flush_mode_move;

    UVM_ASSERT(uvm_sem_is_locked(&parent_gpu->isr.replayable_faults.service_lock));
    UVM_ASSERT((flush_mode == HW_FAULT_BUFFER_FLUSH_MODE_MOVE) || (flush_mode == HW_FAULT_BUFFER_FLUSH_MODE_DISCARD));

    if (!g_uvm_global.conf_computing_enabled)
        return NV_OK;

    is_flush_mode_move = (NvBool) (flush_mode == HW_FAULT_BUFFER_FLUSH_MODE_MOVE);
    status = nvUvmInterfaceFlushReplayableFaultBuffer(&parent_gpu->fault_buffer.rm_info, is_flush_mode_move);

    UVM_ASSERT(status == NV_OK);

    return status;
}

static void fault_buffer_skip_replayable_entry(uvm_parent_gpu_t *parent_gpu, NvU32 index)
{
    UVM_ASSERT(parent_gpu->fault_buffer_hal->entry_is_valid(parent_gpu, index));

    // Flushed faults are never decrypted, but the decryption IV associated with
    // replayable faults still requires manual adjustment so it is kept in sync
    // with the encryption IV on the GSP-RM's side.
    if (g_uvm_global.conf_computing_enabled)
        uvm_conf_computing_fault_increment_decrypt_iv(parent_gpu);

    parent_gpu->fault_buffer_hal->entry_clear_valid(parent_gpu, index);
}

static NV_STATUS fault_buffer_flush_locked(uvm_parent_gpu_t *parent_gpu,
                                           uvm_gpu_t *gpu,
                                           uvm_gpu_buffer_flush_mode_t flush_mode,
                                           uvm_fault_replay_type_t fault_replay,
                                           uvm_fault_service_batch_context_t *batch_context)
{
    NvU32 get;
    NvU32 put;
    uvm_spin_loop_t spin;
    uvm_replayable_fault_buffer_t *replayable_faults = &parent_gpu->fault_buffer.replayable;
    NV_STATUS status;

    UVM_ASSERT(uvm_sem_is_locked(&parent_gpu->isr.replayable_faults.service_lock));
    UVM_ASSERT(parent_gpu->replayable_faults_supported);

    // Wait for the prior replay to flush out old fault messages
    if (flush_mode == UVM_GPU_BUFFER_FLUSH_MODE_WAIT_UPDATE_PUT) {
        status = uvm_tracker_wait(&replayable_faults->replay_tracker);
        if (status != NV_OK)
            return status;
    }

    // Read PUT pointer from the GPU if requested
    if (flush_mode == UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT || flush_mode == UVM_GPU_BUFFER_FLUSH_MODE_WAIT_UPDATE_PUT) {
        status = hw_fault_buffer_flush_locked(parent_gpu, HW_FAULT_BUFFER_FLUSH_MODE_DISCARD);
        if (status != NV_OK)
            return status;

        replayable_faults->cached_put = parent_gpu->fault_buffer_hal->read_put(parent_gpu);
    }

    get = replayable_faults->cached_get;
    put = replayable_faults->cached_put;

    while (get != put) {
        // Wait until valid bit is set
        UVM_SPIN_WHILE(!parent_gpu->fault_buffer_hal->entry_is_valid(parent_gpu, get), &spin) {
            // Channels might be idle (e.g. in teardown) so check for errors
            // actively. In that case the gpu pointer is valid.
            status = gpu ? uvm_channel_manager_check_errors(gpu->channel_manager) : uvm_global_get_status();
            if (status != NV_OK) {
                write_get(parent_gpu, get);
                return status;
            }
        }

        fault_buffer_skip_replayable_entry(parent_gpu, get);
        ++get;
        if (get == replayable_faults->max_faults)
            get = 0;
    }

    write_get(parent_gpu, get);

    // Issue fault replay
    if (gpu)
        return push_replay_on_gpu(gpu, fault_replay, batch_context);

    return push_replay_on_parent_gpu(parent_gpu, fault_replay, batch_context);
}

NV_STATUS uvm_gpu_fault_buffer_flush(uvm_gpu_t *gpu)
{
    NV_STATUS status = NV_OK;

    UVM_ASSERT(gpu->parent->replayable_faults_supported);

    // Disables replayable fault interrupts and fault servicing
    uvm_parent_gpu_replayable_faults_isr_lock(gpu->parent);

    status = fault_buffer_flush_locked(gpu->parent,
                                       gpu,
                                       UVM_GPU_BUFFER_FLUSH_MODE_WAIT_UPDATE_PUT,
                                       UVM_FAULT_REPLAY_TYPE_START,
                                       NULL);

    // This will trigger the top half to start servicing faults again, if the
    // replay brought any back in
    uvm_parent_gpu_replayable_faults_isr_unlock(gpu->parent);
    return status;
}

static inline int cmp_fault_instance_ptr(const uvm_fault_buffer_entry_t *a,
                                         const uvm_fault_buffer_entry_t *b)
{
    int result = uvm_gpu_phys_addr_cmp(a->instance_ptr, b->instance_ptr);
    // On Volta+ we need to sort by {instance_ptr + subctx_id} pair since it can
    // map to a different VA space
    if (result != 0)
        return result;
    return UVM_CMP_DEFAULT(a->fault_source.ve_id, b->fault_source.ve_id);
}

// Compare two VA spaces
static inline int cmp_va_space(const uvm_va_space_t *a, const uvm_va_space_t *b)
{
    return UVM_CMP_DEFAULT(a, b);
}

// Compare two GPUs
static inline int cmp_gpu(const uvm_gpu_t *a, const uvm_gpu_t *b)
{
    NvU32 id_a = a ? uvm_id_value(a->id) : 0;
    NvU32 id_b = b ? uvm_id_value(b->id) : 0;

    return UVM_CMP_DEFAULT(id_a, id_b);
}

// Compare two virtual addresses
static inline int cmp_addr(NvU64 a, NvU64 b)
{
    return UVM_CMP_DEFAULT(a, b);
}

// Compare two fault access types
static inline int cmp_access_type(uvm_fault_access_type_t a, uvm_fault_access_type_t b)
{
    UVM_ASSERT(a >= 0 && a < UVM_FAULT_ACCESS_TYPE_COUNT);
    UVM_ASSERT(b >= 0 && b < UVM_FAULT_ACCESS_TYPE_COUNT);

    // Check that fault access type enum values are ordered by "intrusiveness"
    BUILD_BUG_ON(UVM_FAULT_ACCESS_TYPE_ATOMIC_STRONG <= UVM_FAULT_ACCESS_TYPE_ATOMIC_WEAK);
    BUILD_BUG_ON(UVM_FAULT_ACCESS_TYPE_ATOMIC_WEAK <= UVM_FAULT_ACCESS_TYPE_WRITE);
    BUILD_BUG_ON(UVM_FAULT_ACCESS_TYPE_WRITE <= UVM_FAULT_ACCESS_TYPE_READ);
    BUILD_BUG_ON(UVM_FAULT_ACCESS_TYPE_READ <= UVM_FAULT_ACCESS_TYPE_PREFETCH);

    return b - a;
}

typedef enum
{
    // Fetch a batch of faults from the buffer. Stop at the first entry that is
    // not ready yet
    FAULT_FETCH_MODE_BATCH_READY,

    // Fetch all faults in the buffer before PUT. Wait for all faults to become
    // ready
    FAULT_FETCH_MODE_ALL,
} fault_fetch_mode_t;

static void fetch_fault_buffer_merge_entry(uvm_fault_buffer_entry_t *current_entry,
                                           uvm_fault_buffer_entry_t *last_entry)
{
    UVM_ASSERT(last_entry->num_instances > 0);

    ++last_entry->num_instances;
    uvm_fault_access_type_mask_set(&last_entry->access_type_mask, current_entry->fault_access_type);

    if (current_entry->fault_access_type > last_entry->fault_access_type) {
        // If the new entry has a higher access type, it becomes the
        // fault to be serviced. Add the previous one to the list of instances
        current_entry->access_type_mask = last_entry->access_type_mask;
        current_entry->num_instances = last_entry->num_instances;
        last_entry->filtered = true;

        // We only merge faults from different uTLBs if the new fault has an
        // access type with the same or lower level of intrusiveness.
        UVM_ASSERT(current_entry->fault_source.utlb_id == last_entry->fault_source.utlb_id);

        list_replace(&last_entry->merged_instances_list, &current_entry->merged_instances_list);
        list_add(&last_entry->merged_instances_list, &current_entry->merged_instances_list);
    }
    else {
        // Add the new entry to the list of instances for reporting purposes
        current_entry->filtered = true;
        list_add(&current_entry->merged_instances_list, &last_entry->merged_instances_list);
    }
}

// 尝试将当前刚获取的页错误条目 (current_entry) 与之前缓存的条目进行合并
// 返回 true 表示成功合并（当前条目可以被丢弃/归档），返回 false 表示无法合并（是一个全新的错误）
static bool fetch_fault_buffer_try_merge_entry(uvm_fault_buffer_entry_t *current_entry,
                                               uvm_fault_service_batch_context_t *batch_context,
                                               uvm_fault_utlb_info_t *current_tlb,
                                               bool is_same_instance_ptr)
{
    // 获取当前 uTLB (微型TLB) 记录的最后一次错误
    uvm_fault_buffer_entry_t *last_tlb_entry = current_tlb->last_fault;
    // 获取整个批次 (Batch) 层面全局记录的最后一次错误
    uvm_fault_buffer_entry_t *last_global_entry = batch_context->last_fault;

    // 【检查 1：局部合并 (同一个 uTLB 内部的去重)】
    // 检查条件：
    // 1. 该 uTLB 之前确实有未处理的错误 (num_pending_faults > 0)
    // 2. 所属的进程地址空间 (instance_ptr) 相同
    // 3. 发生缺页的虚拟地址 (fault_address) 完全相同
    const bool is_last_tlb_fault = current_tlb->num_pending_faults > 0 &&
                                   cmp_fault_instance_ptr(current_entry, last_tlb_entry) == 0 &&
                                   current_entry->fault_address == last_tlb_entry->fault_address;

    // 【检查 2：全局合并 (跨不同 uTLB 的去重，即不同计算单元访问了同一个页)】
    // 我们只在一种非常特定的情况下允许跨 uTLB 合并：
    // 当新错误的访问类型“侵入性” <= (小于或等于) 旧错误时！
    // 为什么？因为如果是新错误侵入性更高（比如旧的是 Read，新的是 Write），
    // 那么新的 Write 必须成为“代表”。如果跨 uTLB 转移“代表权”，
    // 驱动就需要去修改两个不同 uTLB 的 num_pending_faults 计数器，逻辑会非常复杂且影响性能。
    const bool is_last_fault = is_same_instance_ptr &&
                               current_entry->fault_address == last_global_entry->fault_address &&
                               current_entry->fault_access_type <= last_global_entry->fault_access_type;

    // --- 开始执行合并操作 ---

    // 优先尝试局部合并 (同 uTLB)
    if (is_last_tlb_fault) {
        // 将新条目合并到上一个条目中（通常是将新条目标记为 'filtered' 并挂载到旧条目的链表上，
        // 同时把访问类型的掩码做 OR 运算）
        fetch_fault_buffer_merge_entry(current_entry, last_tlb_entry);
        
        // 如果新错误的侵入性比旧的更高（比如旧的是读，新的是写），
        // 那么新的错误必须篡位成为这个 uTLB 的“代表” (last_fault)
        if (current_entry->fault_access_type > last_tlb_entry->fault_access_type)
            current_tlb->last_fault = current_entry;

        return true; // 合并成功
    }
    // 如果局部合并失败，尝试全局合并 (跨 uTLB)
    else if (is_last_fault) {
        // 合并到全局最后一个条目中
        fetch_fault_buffer_merge_entry(current_entry, last_global_entry);
        
        // 注意：基于前面 is_last_fault 的定义，这里的 if 条件 (新 > 旧) 
        // 理论上永远不会成立，因为跨 uTLB 合并的前提就是 (新 <= 旧)。
        // 这行代码属于 NVIDIA 驱动的防御性编程 (Defensive Programming)，
        // 防止未来修改 is_last_fault 的判断条件时漏掉这里的逻辑。
        if (current_entry->fault_access_type > last_global_entry->fault_access_type)
            batch_context->last_fault = current_entry;

        return true; // 合并成功
    }

    // 既不是同一个 uTLB 的重复页，也不是跨 uTLB 且符合降级条件的重复页
    // 这是一个全新的页错误，需要分配新的资源去处理
    return false;
}

// -----------------------------------------------------------------------------
// 函数功能：从 GPU 的硬件错误缓冲区 (Fault Buffer) 中获取缺页错误条目，
// 解码这些条目，并将它们存储在批处理上下文 (batch_context) 中。
//
// 【核心优化：故障合并/去重 (Coalescing)】
// 为了最小化处理开销，驱动会尽可能合并重复的缺页错误。
// 规则：如果连续几个错误具有相同的 "实例指针 (instance pointer，代表虚拟地址空间)" 
// 和相同的 "页面虚拟地址"，则认为它们是重复的。
//
// 驱动会保留一个“代表性 (representative)”错误条目，通常是“侵入性”最强的那个
// (优先级：原子操作 atomic > 写操作 write > 读操作 read > 预取 prefetch)。
// 其他被合并的重复条目会被标记为 "filtered(已过滤)"，并挂载到这个代表性条目的链表上。
// -----------------------------------------------------------------------------
static NV_STATUS fetch_fault_buffer_entries(uvm_parent_gpu_t *parent_gpu,
                                            uvm_fault_service_batch_context_t *batch_context,
                                            fault_fetch_mode_t fetch_mode)
{
    NvU32 get; // CPU 读取指针
    NvU32 put; // GPU 写入指针
    NvU32 fault_index;          // 当前读取的条目总数（包括被合并的）
    NvU32 num_coalesced_faults; // 合并去重后的唯一条目数
    NvU32 utlb_id;              // GPU 微型 TLB (uTLB) 的 ID
    uvm_fault_buffer_entry_t *fault_cache; // 软件层面缓存这些错误的数组
    uvm_spin_loop_t spin;
    NV_STATUS status = NV_OK;
    uvm_replayable_fault_buffer_t *replayable_faults = &parent_gpu->fault_buffer.replayable;
    
    // 特殊情况：如果是 Pascal 架构的 GPU 且处于取消错误的路径中，则不能进行合并优化。
    // 因为 Pascal 硬件需要精确追踪每个 uTLB 中的所有错误才能保证正确取消。
    const bool in_pascal_cancel_path = (!parent_gpu->fault_cancel_va_supported && fetch_mode == FAULT_FETCH_MODE_ALL);
    // 是否允许进行合并过滤
    const bool may_filter = uvm_perf_fault_coalesce && !in_pascal_cancel_path;

    // 断言：确保当前持有服务锁，并且 GPU 支持可重试错误
    UVM_ASSERT(uvm_sem_is_locked(&parent_gpu->isr.replayable_faults.service_lock));
    UVM_ASSERT(parent_gpu->replayable_faults_supported);

    fault_cache = batch_context->fault_cache;

    // 获取上一次读取到的位置 (Cached Get)
    get = replayable_faults->cached_get;

    // 如果 Get 追上了 Cached Put，说明软件以为读完了，
    // 这时需要真正去读一次 GPU 硬件寄存器，获取最新的 Put 指针。
    if (get == replayable_faults->cached_put)
        replayable_faults->cached_put = parent_gpu->fault_buffer_hal->read_put(parent_gpu);

    put = replayable_faults->cached_put;

    // 初始化批处理上下文状态
    batch_context->is_single_instance_ptr = true; // 假设这批错误都属于同一个进程/地址空间
    batch_context->last_fault = NULL;

    fault_index = 0;
    num_coalesced_faults = 0;

    // 清理所有 uTLB 的统计信息
    for (utlb_id = 0; utlb_id <= batch_context->max_utlb_id; ++utlb_id) {
        batch_context->utlbs[utlb_id].num_pending_faults = 0;
        batch_context->utlbs[utlb_id].has_fatal_faults = false;
    }
    batch_context->max_utlb_id = 0;

    // 如果 Get == Put，说明环形缓冲区为空，没有新的错误，直接跳到结束
    if (get == put)
        goto done;

    // 【主循环】：开始遍历环形缓冲区，直到读完 (get == put) 
    // 或者达到了单次处理的最大批次上限 (max_batch_size)
    while ((get != put) &&
           (fetch_mode == FAULT_FETCH_MODE_ALL || fault_index < parent_gpu->fault_buffer.max_batch_size)) {
        bool is_same_instance_ptr = true;
        NvU64 fault_address_raw;
        uvm_fault_buffer_entry_t *current_entry = &fault_cache[fault_index];
        uvm_fault_utlb_info_t *current_tlb;

        // 【硬件同步】：因为 GPU 写入条目可能是乱序的，即使 Get < Put，
        // 当前位置的条目可能还没完全写回系统内存。
        // 所以这里必须自旋等待该条目的 "Valid(有效)" 位被硬件置位。
        UVM_SPIN_WHILE(!parent_gpu->fault_buffer_hal->entry_is_valid(parent_gpu, get), &spin) {
            // 如果是在 BATCH_READY 模式下且已经读取了一些条目，就不等了，先处理手头的
            if (fetch_mode == FAULT_FETCH_MODE_BATCH_READY && fault_index > 0)
                goto done;

            // 检查全局驱动状态，防止在系统关机或崩溃时死锁
            status = uvm_global_get_status();
            if (status != NV_OK)
                goto done;
        }

        // 【内存屏障】：确保在读取 Valid 位之后，才去读取条目的实际内容。
        // 防止 CPU 乱序执行导致读到旧的脏数据。
        smp_mb__after_atomic();

        // 硬件 Valid 位已设置，开始解析这条硬件记录，转为软件结构体
        status = parent_gpu->fault_buffer_hal->parse_replayable_entry(parent_gpu, get, current_entry);
        if (status != NV_OK)
            goto done;

        fault_address_raw = current_entry->fault_address;

        // GPU 硬件是以 4KB 对齐报告错误地址的，
        // 但操作系统的 PAGE_SIZE 可能是 64KB (如某些 ARM/PowerPC 架构)，
        // 因此这里需要向下对齐到操作系统的页面边界。
        current_entry->fault_address = UVM_PAGE_ALIGN_DOWN(current_entry->fault_address);

        if (uvm_perf_fault_log_addresses)
            log_replayable_fault_entry(parent_gpu, current_entry, fault_address_raw);

        // 初始化/标记致命错误状态 (Fatal)
        current_entry->is_fatal = (current_entry->fault_type >= UVM_FAULT_TYPE_FATAL);

        if (current_entry->is_fatal) {
            // 稍后在锁定 VA space (虚拟地址空间) 时再详细记录致命错误原因
            current_entry->fatal_reason = UvmEventFatalReasonInvalidFaultType;
        }
        else {
            current_entry->fatal_reason = UvmEventFatalReasonInvalid;
        }

        // 初始化当前条目的其他软件追踪字段
        current_entry->va_space = NULL;
        current_entry->gpu = NULL;
        current_entry->filtered = false; // 默认未被过滤(未被合并)
        current_entry->replayable.cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_ALL;

        // 更新这批错误中遇到的最大 uTLB ID
        if (current_entry->fault_source.utlb_id > batch_context->max_utlb_id) {
            UVM_ASSERT(current_entry->fault_source.utlb_id < replayable_faults->utlb_count);
            batch_context->max_utlb_id = current_entry->fault_source.utlb_id;
        }

        current_tlb = &batch_context->utlbs[current_entry->fault_source.utlb_id];

        // 【核心合并逻辑】：如果这不是第一个条目，尝试与前一个条目合并
        if (fault_index > 0) {
            UVM_ASSERT(batch_context->last_fault);
            // 比较当前错误与上一个错误的实例指针 (判断是否属于同一个上下文)
            is_same_instance_ptr = cmp_fault_instance_ptr(current_entry, batch_context->last_fault) == 0;

            // 如果允许合并且当前不是致命错误
            if (may_filter && !current_entry->is_fatal) {
                // 尝试合并 (如果地址相同，则更新代表性条目的访问类型，并将当前条目标记为过滤)
                bool merged = fetch_fault_buffer_try_merge_entry(current_entry,
                                                                 batch_context,
                                                                 current_tlb,
                                                                 is_same_instance_ptr);
                // 如果成功合并，直接跳过后续初始化，处理下一个硬件条目
                if (merged)
                    goto next_fault; 
            }
        }

        // 如果发现这批错误属于不同的实例指针 (不同进程)，更新标志位
        if (batch_context->is_single_instance_ptr && !is_same_instance_ptr)
            batch_context->is_single_instance_ptr = false;

        // 【新唯一错误初始化】：如果代码走到这里，说明这是一个新的、无法被合并的唯一错误
        current_entry->num_instances = 1;
        // 将访问类型 (读/写/原子) 转换为掩码，方便后续合并时做或运算 (OR)
        current_entry->access_type_mask = uvm_fault_access_type_mask_bit(current_entry->fault_access_type);
        INIT_LIST_HEAD(&current_entry->merged_instances_list); // 初始化合并链表头

        // 更新 uTLB 追踪状态
        ++current_tlb->num_pending_faults;
        current_tlb->last_fault = current_entry;
        batch_context->last_fault = current_entry; // 记录为最后一个独立错误，供下一个条目比较

        ++num_coalesced_faults; // 唯一错误计数器 +1

    next_fault:
        ++fault_index; // 总读取条目数 +1
        ++get;         // 推进 Get 指针
        // 处理环形缓冲区的回绕 (Wrap-around)
        if (get == replayable_faults->max_faults)
            get = 0;
    }

done:
    // 【通知硬件】：将最新的 Get 指针写回给 GPU 硬件。
    // 这非常关键，等于告诉 GPU：“这部分缓冲区的错误我已经读走了，你可以覆盖它们了。”
    write_get(parent_gpu, get);

    // 将统计数据保存到上下文结构体中，供后续处理函数使用
    batch_context->num_cached_faults = fault_index;               // 实际从硬件读了多少条
    batch_context->num_coalesced_faults = num_coalesced_faults;   // 去重后剩下多少条需要处理

    return status;
}

// Sort comparator for pointers to fault buffer entries that sorts by
// instance pointer
static int cmp_sort_fault_entry_by_instance_ptr(const void *_a, const void *_b)
{
    const uvm_fault_buffer_entry_t **a = (const uvm_fault_buffer_entry_t **)_a;
    const uvm_fault_buffer_entry_t **b = (const uvm_fault_buffer_entry_t **)_b;

    return cmp_fault_instance_ptr(*a, *b);
}

// Sort comparator for pointers to fault buffer entries that sorts by va_space,
// GPU ID, fault address, and fault access type.
static int cmp_sort_fault_entry_by_va_space_gpu_address_access_type(const void *_a, const void *_b)
{
    const uvm_fault_buffer_entry_t **a = (const uvm_fault_buffer_entry_t **)_a;
    const uvm_fault_buffer_entry_t **b = (const uvm_fault_buffer_entry_t **)_b;

    int result;

    result = cmp_va_space((*a)->va_space, (*b)->va_space);
    if (result != 0)
        return result;

    result = cmp_gpu((*a)->gpu, (*b)->gpu);
    if (result != 0)
        return result;

    result = cmp_addr((*a)->fault_address, (*b)->fault_address);
    if (result != 0)
        return result;

    return cmp_access_type((*a)->fault_access_type, (*b)->fault_access_type);
}

// Translate all instance pointers to a VA space and GPU instance. Since the
// buffer is ordered by instance_ptr, we minimize the number of translations.
//
// This function returns NV_WARN_MORE_PROCESSING_REQUIRED if a fault buffer
// flush occurred and executed successfully, or the error code if it failed.
// NV_OK otherwise.
static NV_STATUS translate_instance_ptrs(uvm_parent_gpu_t *parent_gpu,
                                         uvm_fault_service_batch_context_t *batch_context)
{
    NvU32 i;
    NV_STATUS status;

    for (i = 0; i < batch_context->num_coalesced_faults; ++i) {
        uvm_fault_buffer_entry_t *current_entry;

        current_entry = batch_context->ordered_fault_cache[i];

        // If this instance pointer matches the previous instance pointer, just
        // copy over the already-translated va_space and move on.
        if (i != 0 && cmp_fault_instance_ptr(current_entry, batch_context->ordered_fault_cache[i - 1]) == 0) {
            current_entry->va_space = batch_context->ordered_fault_cache[i - 1]->va_space;
            current_entry->gpu = batch_context->ordered_fault_cache[i - 1]->gpu;
            continue;
        }

        status = uvm_parent_gpu_fault_entry_to_va_space(parent_gpu,
                                                        current_entry,
                                                        &current_entry->va_space,
                                                        &current_entry->gpu);
        if (status != NV_OK) {
            uvm_gpu_t *gpu = NULL;

            if (status == NV_ERR_PAGE_TABLE_NOT_AVAIL) {
                // The channel is valid but the subcontext is not. This can only
                // happen if the subcontext is torn down before its work is
                // complete while other subcontexts in the same TSG are still
                // executing which means some GPU is still valid under that
                // parent GPU. This is a violation of the programming model. We
                // have limited options since the VA space is gone, meaning we
                // can't target the PDB for cancel even if we wanted to. So
                // we'll just throw away precise attribution and cancel this
                // fault using the SW method, which validates that the intended
                // context (TSG) is still running so we don't cancel an innocent
                // context.
                gpu = uvm_parent_gpu_find_first_valid_gpu(parent_gpu);

                UVM_ASSERT(!current_entry->va_space);
                UVM_ASSERT(gpu);
                UVM_ASSERT(gpu->max_subcontexts > 0);

                if (parent_gpu->smc.enabled) {
                    status = push_cancel_on_gpu_targeted(gpu,
                                                         current_entry->instance_ptr,
                                                         current_entry->fault_source.gpc_id,
                                                         current_entry->fault_source.client_id,
                                                         &batch_context->tracker);
                }
                else {
                    status = push_cancel_on_gpu_global(gpu, current_entry->instance_ptr, &batch_context->tracker);
                }

                if (status != NV_OK)
                    return status;

                // Fall through and let the flush restart fault processing
            }
            else {
                UVM_ASSERT(status == NV_ERR_INVALID_CHANNEL);
            }

            // If the channel is gone then we're looking at a stale fault entry.
            // The fault must have been resolved already (serviced or
            // cancelled), so we can just flush the fault buffer.
            //
            // No need to use UVM_GPU_BUFFER_FLUSH_MODE_WAIT_UPDATE_PUT since
            // there was a context preemption for the entries we want to flush,
            // meaning PUT must reflect them.
            status = fault_buffer_flush_locked(parent_gpu,
                                               gpu,
                                               UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT,
                                               UVM_FAULT_REPLAY_TYPE_START,
                                               batch_context);
            if (status != NV_OK)
                 return status;

            return NV_WARN_MORE_PROCESSING_REQUIRED;
        }
        else {
            UVM_ASSERT(current_entry->va_space);
            UVM_ASSERT(current_entry->gpu);
        }
    }

    return NV_OK;
}

// Fault cache preprocessing for fault coalescing
// 错误缓存预处理（用于更高级别的错误合并与处理优化）
//
// 该函数会生成一个针对 fault_cache 的“有序视图”。在这个视图中，
// 错误会按照 虚拟地址空间(VA space)、缺页地址(按4K对齐) 以及 访问类型的“侵入性”进行排序。
// 
// 为了最小化将 instance_ptr（硬件实例指针）翻译成 VA space（软件虚拟地址空间）的性能开销，
// 我们首先会进行一次基于 instance_ptr 的排序。
//
// 当前的处理策略如下：
// 1) 按照 instance_ptr 进行排序（把属于同一个进程的错误聚在一起）
// 2) 批量将所有的 instance_ptrs 翻译为 VA spaces
// 3) 按照 va_space、GPU ID、缺页地址 以及 访问类型 进行最终排序。
static NV_STATUS preprocess_fault_batch(uvm_parent_gpu_t *parent_gpu,
                                        uvm_fault_service_batch_context_t *batch_context)
{
    NV_STATUS status;
    NvU32 i, j;
    // 获取用来存放指针的数组 (Ordered View)。注意：这里存放的是指针，不是结构体本身！
    uvm_fault_buffer_entry_t **ordered_fault_cache = batch_context->ordered_fault_cache;

    // 断言：确保当前批次至少有一个去重后的有效错误
    UVM_ASSERT(batch_context->num_coalesced_faults > 0);
    // 断言：硬件读取的总错误数 肯定大于等于 去重后的有效错误数
    UVM_ASSERT(batch_context->num_cached_faults >= batch_context->num_coalesced_faults);

    // 【步骤 0：提取有效代表，剔除已合并的废弃条目】
    // 遍历刚刚从硬件拉取的所有原始错误记录。
    // 我们只把那些没有被标记为 'filtered' (即上一轮没有被合并掉的、作为代表的错误) 
    // 的地址指针，提取到 ordered_fault_cache 数组中。
    // 这是一种经典的 C 语言优化：只移动指针，不拷贝庞大的结构体数据。
    for (i = 0, j = 0; i < batch_context->num_cached_faults; ++i) {
        if (!batch_context->fault_cache[i].filtered)
            ordered_fault_cache[j++] = &batch_context->fault_cache[i];
    }
    // 确保提取出的指针数量，正好等于我们之前统计的有效错误数
    UVM_ASSERT(j == batch_context->num_coalesced_faults);

    // 【步骤 1：按 instance_ptr 排序 (物理上下文分组)】
    // 如果这一批错误来自多个不同的进程/地址空间
    if (!batch_context->is_single_instance_ptr) {
        // 使用底层的排序算法 (类似 qsort)，将属于同一个 instance_ptr 的错误指针排到一起
        sort(ordered_fault_cache,
             batch_context->num_coalesced_faults,
             sizeof(*ordered_fault_cache),
             cmp_sort_fault_entry_by_instance_ptr,
             NULL);
    }

    // 【步骤 2：翻译 instance_ptrs (极其昂贵的操作)】
    // 将硬件维度的 instance_ptr 翻译为操作系统维度的 VA space (虚拟地址空间)。
    // 为什么要先在步骤 1 排序？
    // 因为翻译操作需要查表和加锁，非常耗时。排序后，同一个进程的错误都在一起，
    // translate_instance_ptrs 内部就可以缓存上一次的翻译结果。
    // 比如：遇到 100 个同属进程 A 的错误，只需查 1 次表，后 99 次直接复用结果。
    status = translate_instance_ptrs(parent_gpu, batch_context);
    if (status != NV_OK)
        return status;

    // 【步骤 3：终极排序 (为了最高效的内存分配和页表修改)】
    // 此时所有错误都已拥有了对应的 va_space。
    // 我们再次进行排序，优先级顺序为：va_space > GPU ID > 缺页地址 > 访问类型
    sort(ordered_fault_cache,
         batch_context->num_coalesced_faults,
         sizeof(*ordered_fault_cache),
         cmp_sort_fault_entry_by_va_space_gpu_address_access_type, // 复杂的比较函数
         NULL);

    return NV_OK;
}

static bool check_fault_entry_duplicate(const uvm_fault_buffer_entry_t *current_entry,
                                        const uvm_fault_buffer_entry_t *previous_entry)
{
    bool is_duplicate = false;

    if (previous_entry) {
        is_duplicate = (current_entry->va_space == previous_entry->va_space) &&
                       (current_entry->fault_address == previous_entry->fault_address);
    }

    return is_duplicate;
}

/**
 * 更新批处理上下文中的缺页统计信息，并向性能监控系统发送缺页通知。
 * * 在 GPU 缺页处理中，成百上千个线程（如一个 Warp 或 Thread Block）同时访问同一个未映射的地址时，
 * 硬件和底层驱动会将这些相同的缺页合并（Coalesce）。此函数用于精确记录这些合并后的统计信息，
 * 并触发系统级的性能事件。
 *
 * @param gpu                发生缺页中断的 GPU 实例。
 * @param batch_context      当前缺页批处理的上下文（包含本批次的统计数据、Batch ID等）。
 * @param va_block           缺页地址所属的虚拟地址块（VA Block，通常管理 2MB 的虚拟内存）。
 * @param preferred_location 根据 UVM 启发式算法，判定该内存页当前最适合存放的处理器位置（CPU 或某块具体的 GPU）。
 * @param current_entry      当前正在处理的硬件缺页缓冲条目（包含原始地址、访问类型、合并的实例数量等）。
 * @param is_duplicate       布尔值，指示当前这个条目是否在软件层面被判定为“重复”（例如该地址所在的页在当前批次中刚刚已经被处理过了）。
 */
static void update_batch_and_notify_fault(uvm_gpu_t *gpu,
                                          uvm_fault_service_batch_context_t *batch_context,
                                          uvm_va_block_t *va_block,
                                          uvm_processor_id_t preferred_location,
                                          uvm_fault_buffer_entry_t *current_entry,
                                          bool is_duplicate)
{
    // ---------------------------------------------------------
    // 第一步：精确计算并更新重复缺页（Duplicate Faults）的数量
    // ---------------------------------------------------------
    // 注意：硬件层面的缺页条目 (current_entry) 自带一个 num_instances 属性，
    // 这代表硬件已经在底层把多次针对完全相同地址的缺页合并成了这一个条目。
    
    if (is_duplicate)
        // 情况 A：整个条目都是重复的。
        // 说明在软件处理这批中断时，发现这个内存页已经被映射过了（可能是被本批次前面的条目处理了）。
        // 因此，这个条目代表的所有实例 (num_instances) 全都是纯粹的重复开销。
        batch_context->num_duplicate_faults += current_entry->num_instances;
    else
        // 情况 B：这是一个全新的缺页条目。
        // 意味着这是本批次第一次处理这个内存页。
        // 所以，在这个硬件条目包含的 num_instances 中，第 1 次是“有效”的（真正触发了后续的内存分配/迁移），
        // 剩下的 (num_instances - 1) 次都是被硬件合并的重复访问。
        batch_context->num_duplicate_faults += current_entry->num_instances - 1;

    // ---------------------------------------------------------
    // 第二步：触发性能监控事件（Performance Event Notification）
    // ---------------------------------------------------------
    // 将缺页的详细上下文广播给 UVM 的性能监控子系统。
    // 这对于性能分析工具（如 NVIDIA Nsight Compute / Nsight Systems）至关重要，
    // 它们借此在时间轴上绘制出 GPU 缺页的具体位置、频率以及 UVM 的迁移决策。
    uvm_perf_event_notify_gpu_fault(&current_entry->va_space->perf_events, // 目标虚拟地址空间的性能事件分发器
                                    va_block,                              // 内存块上下文
                                    gpu->id,                               // 触发中断的 GPU ID
                                    preferred_location,                    // 理想的数据驻留位置
                                    current_entry,                         // 硬件中断原始数据
                                    batch_context->batch_id,               // 批处理流水号
                                    is_duplicate);                         // 是否为软件层面的重复项
}

static void mark_fault_invalid_prefetch(uvm_fault_service_batch_context_t *batch_context,
                                        uvm_fault_buffer_entry_t *fault_entry)
{
    fault_entry->is_invalid_prefetch = true;

    // For block faults, the following counter might be updated more than once
    // for the same fault if block_context->num_retries > 0. As a result, this
    // counter might be higher than the actual count. In order for this counter
    // to be always accurate, block_context needs to passed down the stack from
    // all callers. But since num_retries > 0 case is uncommon and imprecise
    // invalid_prefetch counter doesn't affect functionality (other than
    // disabling prefetching if the counter indicates lots of invalid prefetch
    // faults), this is ok.
    batch_context->num_invalid_prefetch_faults += fault_entry->num_instances;
}

static void mark_fault_throttled(uvm_fault_service_batch_context_t *batch_context,
                                 uvm_fault_buffer_entry_t *fault_entry)
{
    fault_entry->is_throttled = true;
    batch_context->has_throttled_faults = true;
}

static void mark_fault_fatal(uvm_fault_service_batch_context_t *batch_context,
                             uvm_fault_buffer_entry_t *fault_entry,
                             UvmEventFatalReason fatal_reason,
                             uvm_fault_cancel_va_mode_t cancel_va_mode)
{
    uvm_fault_utlb_info_t *utlb = &batch_context->utlbs[fault_entry->fault_source.utlb_id];

    fault_entry->is_fatal = true;
    fault_entry->fatal_reason = fatal_reason;
    fault_entry->replayable.cancel_va_mode = cancel_va_mode;

    utlb->has_fatal_faults = true;

    if (!batch_context->fatal_va_space) {
        UVM_ASSERT(fault_entry->va_space);
        batch_context->fatal_va_space = fault_entry->va_space;
        batch_context->fatal_gpu = fault_entry->gpu;
    }
}

static void fault_entry_duplicate_flags(uvm_fault_service_batch_context_t *batch_context,
                                        uvm_fault_buffer_entry_t *current_entry,
                                        const uvm_fault_buffer_entry_t *previous_entry)
{
    UVM_ASSERT(previous_entry);
    UVM_ASSERT(check_fault_entry_duplicate(current_entry, previous_entry));

    // Propagate the is_invalid_prefetch flag across all prefetch faults
    // on the page
    if (previous_entry->is_invalid_prefetch)
        mark_fault_invalid_prefetch(batch_context, current_entry);

    // If a page is throttled, all faults on the page must be skipped
    if (previous_entry->is_throttled)
        mark_fault_throttled(batch_context, current_entry);
}

// This function computes the maximum access type that can be serviced for the
// reported fault instances given the logical permissions of the VA range. If
// none of the fault instances can be serviced UVM_FAULT_ACCESS_TYPE_COUNT is
// returned instead.
//
// In the case that there are faults that cannot be serviced, this function
// also sets the flags required for fault cancellation. Prefetch faults do not
// need to be cancelled since they disappear on replay.
//
// The UVM driver considers two scenarios for logical permissions violation:
// - All access types are invalid. For example, when faulting from a processor
// that doesn't have access to the preferred location of a range group when it
// is not migratable. In this case all accesses to the page must be cancelled.
// - Write/atomic accesses are invalid. Basically, when trying to modify a
// read-only VA range. In this case we restrict fault cancelling to those types
// of accesses.
//
// Return values:
// - service_access_type: highest access type that can be serviced.
static uvm_fault_access_type_t check_fault_access_permissions(uvm_gpu_t *gpu,
                                                              uvm_fault_service_batch_context_t *batch_context,
                                                              uvm_va_block_t *va_block,
                                                              uvm_service_block_context_t *service_block_context,
                                                              uvm_fault_buffer_entry_t *fault_entry,
                                                              bool allow_migration)
{
    NV_STATUS perm_status;
    UvmEventFatalReason fatal_reason;
    uvm_fault_cancel_va_mode_t cancel_va_mode;
    uvm_fault_access_type_t ret = UVM_FAULT_ACCESS_TYPE_COUNT;
    uvm_va_block_context_t *va_block_context = service_block_context->block_context;

    perm_status = uvm_va_block_check_logical_permissions(va_block,
                                                         va_block_context,
                                                         gpu->id,
                                                         uvm_va_block_cpu_page_index(va_block,
                                                                                     fault_entry->fault_address),
                                                         fault_entry->fault_access_type,
                                                         allow_migration);
    if (perm_status == NV_OK)
        return fault_entry->fault_access_type;

    if (fault_entry->fault_access_type == UVM_FAULT_ACCESS_TYPE_PREFETCH) {
        // Only update the count the first time since logical permissions cannot
        // change while we hold the VA space lock
        // TODO: Bug 1750144: That might not be true with HMM.
        if (service_block_context->num_retries == 0)
            mark_fault_invalid_prefetch(batch_context, fault_entry);

        return ret;
    }

    // At this point we know that some fault instances cannot be serviced
    fatal_reason = uvm_tools_status_to_fatal_fault_reason(perm_status);

    if (fault_entry->fault_access_type > UVM_FAULT_ACCESS_TYPE_READ) {
        cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_WRITE_AND_ATOMIC;

        // If there are pending read accesses on the same page, we have to
        // service them before we can cancel the write/atomic faults. So we
        // retry with read fault access type.
        if (uvm_fault_access_type_mask_test(fault_entry->access_type_mask, UVM_FAULT_ACCESS_TYPE_READ)) {
            perm_status = uvm_va_block_check_logical_permissions(va_block,
                                                                 va_block_context,
                                                                 gpu->id,
                                                                 uvm_va_block_cpu_page_index(va_block,
                                                                                             fault_entry->fault_address),
                                                                 UVM_FAULT_ACCESS_TYPE_READ,
                                                                 allow_migration);
            if (perm_status == NV_OK) {
                ret = UVM_FAULT_ACCESS_TYPE_READ;
            }
            else {
                // Read accesses didn't succeed, cancel all faults
                cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_ALL;
                fatal_reason = uvm_tools_status_to_fatal_fault_reason(perm_status);
            }
        }
    }
    else {
        cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_ALL;
    }

    mark_fault_fatal(batch_context, fault_entry, fatal_reason, cancel_va_mode);

    return ret;
}

/**
 * @brief 批量处理单个虚拟地址块（VA Block）上的多个GPU回放式页面错误。
 *
 * 此函数是UVM驱动中处理GPU产生的回放式页面错误的核心部分。它接收一个已排序的错误列表，
 * 并假设列表中的第一个错误属于传入的`va_block`。该函数会遍历所有属于此块的错误，
 * 通知性能事件，根据权限和策略决定如何处理每个错误（例如迁移内存、复制读取、更新页表等），
 * 最后将计算出的操作应用到实际的物理内存上。
 *
 * 关键行为：
 * - 通知所有块内错误的性能事件以更新启发式算法。
 * - 识别并标记致命错误，但不在此处中断服务流程。
 * - 对重复的错误（同一地址的多次访问）进行去重处理。
 * - 根据访问类型和权限判断是否需要服务该错误。
 * - 计算新的内存驻留位置，并应用服务操作。
 *
 * @param [in]     gpu                 发生错误的GPU实例。
 * @param [in]     va_block            需要处理错误的虚拟地址块。
 * @param [in,out] va_block_retry      用于在资源不足时重试的上下文结构。
 * @param [in,out] batch_context       批处理上下文，包含排序后的错误列表和其他状态。
 * @param [in]     first_fault_index   `batch_context->ordered_fault_cache`中属于此块的第一个错误的索引。
 * @param [in]     hmm_migratable      指示HMM (Heterogeneous Memory Management) 内存是否可迁移。
 * @param [out]    block_faults        返回本次调用处理了多少个错误。
 *
 * @return
 * - NV_OK                        如果所有错误（包括致命和非致命）都已处理。
 * - NV_ERR_MORE_PROCESSING_REQUIRED  如果因资源不足（如内存分配失败）导致服务未完成，需要重试。
 * - NV_ERR_NO_MEMORY             如果由于内存耗尽（OOM）而无法处理错误。
 * - 其他值                       表示UVM全局错误。
 */
static NV_STATUS service_fault_batch_block_locked(uvm_gpu_t *gpu,
                                                  uvm_va_block_t *va_block,
                                                  uvm_va_block_retry_t *va_block_retry,
                                                  uvm_fault_service_batch_context_t *batch_context,
                                                  NvU32 first_fault_index,
                                                  const bool hmm_migratable,
                                                  NvU32 *block_faults)
{
    NV_STATUS status = NV_OK; // 返回状态，默认成功
    NvU32 i; // 循环索引
    uvm_page_index_t first_page_index; // 在此块中遇到的最小页面索引
    uvm_page_index_t last_page_index;  // 在此块中遇到的最大页面索引
    NvU32 page_fault_count = 0; // 需要被服务的页面数量计数器
    uvm_range_group_range_iter_t iter; // 用于迭代地址范围的migratability属性的迭代器
    uvm_replayable_fault_buffer_t *replayable_faults = &gpu->parent->fault_buffer.replayable; // GPU的回放错误缓冲区
    uvm_fault_buffer_entry_t **ordered_fault_cache = batch_context->ordered_fault_cache; // 指向排序后错误条目的指针数组
    uvm_fault_buffer_entry_t *first_fault_entry = ordered_fault_cache[first_fault_index]; // 属于此块的第一个错误条目
    uvm_service_block_context_t *block_context = &replayable_faults->block_service_context; // 块级服务上下文，用于累积本次批处理的所有决策
    uvm_va_space_t *va_space = uvm_va_block_get_va_space(va_block); // 获取VA块所属的虚拟地址空间
    const uvm_va_policy_t *policy; // 当前地址范围的内存管理策略
    NvU64 end; // 当前策略适用的结束地址

    // 编译时检查，确保访问类型枚举值可以用NvU8存储
    BUILD_BUG_ON(UVM_FAULT_ACCESS_TYPE_COUNT > (int)(NvU8)-1);

    // 断言：必须持有VA块的锁才能进入此函数
    uvm_assert_mutex_locked(&va_block->lock);

    // 初始化返回值
    *block_faults = 0;

    // 初始化页面索引边界
    first_page_index = PAGES_PER_UVM_VA_BLOCK; // 设置为最大可能值
    last_page_index = 0; // 设置为最小值

    // 初始化块级服务上下文，清空之前的状态
    uvm_processor_mask_zero(&block_context->resident_processors); // 清空新驻留处理器掩码
    block_context->thrashing_pin_count = 0; // 清空抖动钉住计数
    block_context->read_duplicate_count = 0; // 清空读取重复计数

    // 初始化migratability迭代器，从VA块起始地址开始
    uvm_range_group_range_migratability_iter_first(va_space, va_block->start, va_block->end, &iter);

    // 断言：确保第一个错误确实属于这个VA块
    UVM_ASSERT(first_fault_entry->va_space == va_space);
    UVM_ASSERT(first_fault_entry->gpu == gpu);
    UVM_ASSERT(first_fault_entry->fault_address >= va_block->start);
    UVM_ASSERT(first_fault_entry->fault_address <= va_block->end);

    // 获取适用于第一个错误地址的策略
    if (uvm_va_block_is_hmm(va_block)) { // 如果是HMM管理的内存
        policy = uvm_hmm_find_policy_end(va_block, // 查找策略并获取其结束地址
                                         block_context->block_context->hmm.vma, // VMA信息
                                         first_fault_entry->fault_address,
                                         &end);
    }
    else { // 否则是标准UVM管理的内存
        policy = &va_block->managed_range->policy; // 直接获取策略
        end = va_block->end; // 结束地址就是块的结尾
    }

    // --- 主循环：遍历所有属于此VA块的错误 ---
    for (i = first_fault_index;
         i < batch_context->num_coalesced_faults && // 未超出总错误数
         ordered_fault_cache[i]->va_space == va_space && // 属于同一个VA空间
         ordered_fault_cache[i]->gpu == gpu && // 属于同一个GPU
         ordered_fault_cache[i]->fault_address <= end; // 地址在当前策略范围内
         ++i) {

        uvm_fault_buffer_entry_t *current_entry = ordered_fault_cache[i]; // 当前处理的错误条目
        const uvm_fault_buffer_entry_t *previous_entry = NULL; // 前一个错误条目（用于重复检查）
        bool read_duplicate; // 输出参数：指示是否需要创建读取副本
        uvm_processor_id_t new_residency; // 输出参数：计算出的新内存驻留位置
        uvm_perf_thrashing_hint_t thrashing_hint; // 抖动检测启发式提示
        uvm_page_index_t page_index = uvm_va_block_cpu_page_index(va_block, current_entry->fault_address); // 将错误地址转换为块内的页面索引
        bool is_duplicate = false; // 标记当前错误是否为重复错误
        uvm_fault_access_type_t service_access_type; // 经过权限检查后，实际需要服务的访问类型
        NvU32 service_access_type_mask; // 对应的服务访问类型掩码

        // 断言：错误条目中的最高优先级访问类型应与记录的故障访问类型一致
        UVM_ASSERT(current_entry->fault_access_type ==
                   uvm_fault_access_type_mask_highest(current_entry->access_type_mask));

        // 断言：调用者已经过滤掉了不可服务的错误（如致命错误或无效地址），所以这里不应出现
        UVM_ASSERT(!current_entry->is_fatal);
        // 重置错误条目的状态标志
        current_entry->is_throttled        = false;
        current_entry->is_invalid_prefetch = false;

        // 检查是否与前一个错误重复（即同一页面的多次访问）
        if (i > first_fault_index) {
            previous_entry = ordered_fault_cache[i - 1];
            is_duplicate = check_fault_entry_duplicate(current_entry, previous_entry);
        }

        // 如果是首次处理此批次，则更新统计信息并通知错误事件
        if (block_context->num_retries == 0) {
            update_batch_and_notify_fault(gpu, // 更新批量计数器
                                          batch_context,
                                          va_block,
                                          policy->preferred_location, // 传递首选位置给事件
                                          current_entry,
                                          is_duplicate); // 通知错误发生（可能包括重复信息）
        }

        // 如果是重复错误，设置其标记并跳过后续服务逻辑，因为页面已被前一个实例处理
        if (is_duplicate) {
            fault_entry_duplicate_flags(batch_context, current_entry, previous_entry); // 复制前一个错误的处理结果

            // 如果前一个错误不是致命的，说明页面已经被成功服务，当前重复错误可以跳过
            if (!previous_entry->is_fatal)
                continue;
        }

        // 确保migratability迭代器覆盖了当前错误地址。如果地址超出了当前迭代器范围，则移动迭代器。
        while (iter.end < current_entry->fault_address)
            uvm_range_group_range_migratability_iter_next(va_space, &iter, va_block->end);

        // 断言：当前错误地址应在迭代器所指的有效范围内
        UVM_ASSERT(iter.start <= current_entry->fault_address && iter.end >= current_entry->fault_address);

        // 检查错误的访问类型是否具有逻辑上的权限。这可能会降低访问权限（如将写转为读）或发现权限错误。
        service_access_type = check_fault_access_permissions(gpu,
                                                             batch_context,
                                                             va_block,
                                                             block_context,
                                                             current_entry,
                                                             iter.migratable); // 传入当前范围的migratability属性

        // 如果权限检查失败（返回无效的访问类型），则跳过此错误，继续处理下一个
        if (service_access_type == UVM_FAULT_ACCESS_TYPE_COUNT)
            continue;

        // 如果权限检查修改了访问类型（例如从WRITE降级为READ）
        if (service_access_type != current_entry->fault_access_type) {
            // 重新计算需要服务的访问类型掩码
            UVM_ASSERT(service_access_type < current_entry->fault_access_type);
            service_access_type_mask = uvm_fault_access_type_mask_bit(service_access_type);
        }
        else {
            // 否则，使用原始的访问类型掩码
            service_access_type_mask = current_entry->access_type_mask;
        }

        // 检查GPU是否已经拥有足够的权限来执行请求的访问（例如，已经是可写的RW页）。如果是，则无需服务。
        if (uvm_va_block_page_is_gpu_authorized(va_block,
                                                page_index,
                                                gpu->id,
                                                uvm_fault_access_type_to_prot(service_access_type))) // 转换访问类型为内存保护标志
            continue;

        // --- 性能优化/启发式处理 ---

        // 获取抖动检测启发式提示（例如是否需要限流或钉住页面）
        thrashing_hint = uvm_perf_thrashing_get_hint(va_block,
                                                     block_context->block_context,
                                                     current_entry->fault_address,
                                                     gpu->id);

        if (thrashing_hint.type == UVM_PERF_THRASHING_HINT_TYPE_THROTTLE) {
            // 实施限流策略：在CPU端休眠，在GPU端继续处理其他页面的错误
            // 只在首次处理时设置标志
            if (block_context->num_retries == 0)
                mark_fault_throttled(batch_context, current_entry);

            continue; // 跳过此页面的进一步服务
        }
        else if (thrashing_hint.type == UVM_PERF_THRASHING_HINT_TYPE_PIN) {
            // 实施钉住策略：将页面固定在当前驻留位置以减少迁移开销
            if (block_context->thrashing_pin_count++ == 0)
                uvm_page_mask_zero(&block_context->thrashing_pin_mask); // 第一次钉住时清空钉住掩码

            uvm_page_mask_set(&block_context->thrashing_pin_mask, page_index); // 将当前页面加入钉住掩码
        }

        // --- 决策阶段：计算新的内存驻留位置 ---
        // 核心决策函数，基于策略、访问模式、抖动提示等，决定页面应该迁移到哪个处理器，并判断是否需要创建读取副本。
        new_residency = uvm_va_block_select_residency(va_block,
                                                      block_context->block_context,
                                                      page_index,
                                                      gpu->id, // 请求者
                                                      service_access_type_mask, // 访问需求
                                                      policy, // 用户策略
                                                      &thrashing_hint, // 抖动提示
                                                      UVM_SERVICE_OPERATION_REPLAYABLE_FAULTS, // 操作类型
                                                      hmm_migratable, // HMM可迁移性
                                                      &read_duplicate); // 输出：是否需要读取副本

        // --- 累积决策到块上下文 ---
        // 如果新的驻留地是首次被选中，则清空其对应的页面掩码
        if (!uvm_processor_mask_test_and_set(&block_context->resident_processors, new_residency))
            uvm_page_mask_zero(&block_context->per_processor_masks[uvm_id_value(new_residency)].new_residency);

        // 将当前页面添加到目标处理器的新驻留页面掩码中
        uvm_page_mask_set(&block_context->per_processor_masks[uvm_id_value(new_residency)].new_residency, page_index);

        // 如果需要读取副本，也进行类似的累积
        if (read_duplicate) {
            if (block_context->read_duplicate_count++ == 0)
                uvm_page_mask_zero(&block_context->read_duplicate_mask); // 第一次需要副本时清空掩码

            uvm_page_mask_set(&block_context->read_duplicate_mask, page_index); // 将当前页面加入副本掩码
        }

        // --- 更新内部统计 ---
        ++page_fault_count; // 增加需要服务的页面计数

        // 记录该页面需要的最终访问类型
        block_context->access_type[page_index] = service_access_type;

        // 更新本次批处理需要服务的页面边界
        if (page_index < first_page_index)
            first_page_index = page_index;
        if (page_index > last_page_index)
            last_page_index = page_index;
    } // 结束主循环

    // --- 应用阶段：将累积的决策应用到物理内存 ---
    // 如果有页面需要被服务，则调用核心服务函数
    if (page_fault_count > 0) {
        // 定义需要服务的区域（从第一个到最后一个页面）
        block_context->region = uvm_va_block_region(first_page_index, last_page_index + 1);
        // 执行实际的内存操作（迁移、映射等）
        status = uvm_va_block_service_locked(gpu->id, va_block, va_block_retry, block_context);
    }

    // 计算本次调用处理了多少个错误，并通过参数返回
    *block_faults = i - first_fault_index;

    // 增加重试计数器
    ++block_context->num_retries;

    // 如果服务成功，并且存在致命错误（由其他地方设置），则取消整个VA块上的服务
    if (status == NV_OK && batch_context->fatal_va_space)
        status = uvm_va_block_set_cancel(va_block, block_context->block_context, gpu);

    return status;
}
// 我们会通知（更新）该 block 内所有错误的性能启发式算法（Heuristics）状态。
// 在处理整个缺页故障的期间，必须持有该 VA block 的互斥锁。
// 但是，如果在此期间发现显存满了需要“驱逐（Eviction）”其他内存，
// 锁可能会被暂时释放（以便其他线程工作），并在驱逐完成后重新获取。
static NV_STATUS service_fault_batch_block(uvm_gpu_t *gpu,
                                           uvm_va_block_t *va_block,
                                           uvm_fault_service_batch_context_t *batch_context,
                                           NvU32 first_fault_index,
                                           const bool hmm_migratable,
                                           NvU32 *block_faults)
{
    NV_STATUS status;
    uvm_va_block_retry_t va_block_retry;
    NV_STATUS tracker_status;
    uvm_replayable_fault_buffer_t *replayable_faults = &gpu->parent->fault_buffer.replayable;
    uvm_service_block_context_t *fault_block_context = &replayable_faults->block_service_context;

    // 【上下文初始化】
    // 标记当前 block 正在进行的操作是“处理可重试的缺页错误”。
    // 这可以让底层的其他函数知道当前的上下文环境。
    fault_block_context->operation = UVM_SERVICE_OPERATION_REPLAYABLE_FAULTS;
    fault_block_context->num_retries = 0; // 重试次数归零

    // 【步骤 1：HMM 操作系统级同步 (准备阶段)】
    // 如果这是一个 Linux HMM（异构内存管理）纳管的普通内存块。
    if (uvm_va_block_is_hmm(va_block))
        // 通知 Linux 内核：“GPU 准备动这块内存了，如果 CPU 那边有其他线程
        // 正准备把它交换到硬盘或者做其他迁移，请让它们先等一等！”
        uvm_hmm_migrate_begin_wait(va_block);

    // 【步骤 2：获取核心互斥锁】
    // 拿到了这把锁，意味着当前线程对这 2MB 虚拟地址空间拥有了绝对的控制权。
    // 其他任何想要读写、迁移这 2MB 数据的 GPU 线程或 CPU 线程，都必须在外面排队。
    uvm_mutex_lock(&va_block->lock);

    // 【步骤 3：核心业务逻辑与“重试宏”】
    // UVM_VA_BLOCK_RETRY_LOCKED 是一个极其复杂的宏！
    // 它的内部其实包含了一个 while 循环。
    // 为什么需要循环？因为在执行内部的 service_fault_batch_block_locked 时，
    // 如果发现显存不足，驱动需要去“驱逐(Evict)”别的内存。而驱逐操作可能需要很长时间，
    // 为了防止死锁，内部函数会**主动把 va_block->lock 解开**，等驱逐完再重新加锁。
    // 这个宏就是用来处理这种“解锁 -> 等待 -> 重新加锁 -> 从头重试”逻辑的。
    status = UVM_VA_BLOCK_RETRY_LOCKED(va_block, &va_block_retry,
                                       service_fault_batch_block_locked(gpu,
                                                                        va_block,
                                                                        &va_block_retry,
                                                                        batch_context,
                                                                        first_fault_index,
                                                                        hmm_migratable,
                                                                        block_faults));

    // 【步骤 4：合并追踪器 (Tracker)】
    // 内部的 locked 函数已经把“分配物理内存”、“拷贝数据”、“写 GPU 页表”的异步指令
    // 推送给 GPU 硬件了。这些异步指令的凭证都记录在了 va_block->tracker 里。
    // 现在，我们需要把这个局部 block 的追踪凭证，合并到整个批次 (batch) 的全局追踪器中。
    // 这样，外层函数才能在最后通过等待 batch_context->tracker 来确保所有硬件操作都已落盘。
    tracker_status = uvm_tracker_add_tracker_safe(&batch_context->tracker, &va_block->tracker);

    // 【步骤 5：释放核心互斥锁】
    // 活干完了，异步指令发完了，释放这 2MB 空间的控制权。
    uvm_mutex_unlock(&va_block->lock);

    // 【步骤 6：HMM 操作系统级同步 (收尾阶段)】
    if (uvm_va_block_is_hmm(va_block))
        // 通知 Linux 内核：“GPU 已经把该发出的指令都发完了，你们可以继续处理这块内存的其他事务了。”
        uvm_hmm_migrate_finish(va_block);

    // 返回状态：优先返回业务处理的状态 (status)，如果业务成功但合并追踪器失败了，则返回追踪器的错误。
    return status == NV_OK ? tracker_status : status;
}

typedef enum
{
    // Use this mode when calling from the normal fault servicing path
    FAULT_SERVICE_MODE_REGULAR,

    // Use this mode when servicing faults from the fault cancelling algorithm.
    // In this mode no replays are issued
    FAULT_SERVICE_MODE_CANCEL,
} fault_service_mode_t;

static void service_fault_batch_fatal(uvm_fault_service_batch_context_t *batch_context,
                                      NvU32 first_fault_index,
                                      NV_STATUS status,
                                      uvm_fault_cancel_va_mode_t cancel_va_mode,
                                      NvU32 *block_faults)
{
    uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[first_fault_index];
    const uvm_fault_buffer_entry_t *previous_entry = first_fault_index > 0 ?
                                                       batch_context->ordered_fault_cache[first_fault_index - 1] : NULL;
    bool is_duplicate = check_fault_entry_duplicate(current_entry, previous_entry);

    if (is_duplicate)
        fault_entry_duplicate_flags(batch_context, current_entry, previous_entry);

    if (current_entry->fault_access_type == UVM_FAULT_ACCESS_TYPE_PREFETCH)
        mark_fault_invalid_prefetch(batch_context, current_entry);
    else
        mark_fault_fatal(batch_context, current_entry, uvm_tools_status_to_fatal_fault_reason(status), cancel_va_mode);

    (*block_faults)++;
}

static void service_fault_batch_fatal_notify(uvm_gpu_t *gpu,
                                             uvm_fault_service_batch_context_t *batch_context,
                                             NvU32 first_fault_index,
                                             NV_STATUS status,
                                             uvm_fault_cancel_va_mode_t cancel_va_mode,
                                             NvU32 *block_faults)
{
    uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[first_fault_index];
    const uvm_fault_buffer_entry_t *previous_entry = first_fault_index > 0 ?
                                                       batch_context->ordered_fault_cache[first_fault_index - 1] : NULL;
    bool is_duplicate = check_fault_entry_duplicate(current_entry, previous_entry);

    service_fault_batch_fatal(batch_context, first_fault_index, status, cancel_va_mode, block_faults);

    update_batch_and_notify_fault(gpu, batch_context, NULL, UVM_ID_INVALID, current_entry, is_duplicate);
}

static NV_STATUS service_fault_batch_ats_sub_vma(uvm_gpu_va_space_t *gpu_va_space,
                                                 struct vm_area_struct *vma,
                                                 NvU64 base,
                                                 uvm_fault_service_batch_context_t *batch_context,
                                                 NvU32 fault_index_start,
                                                 NvU32 fault_index_end,
                                                 NvU32 *block_faults)
{
    NvU32 i;
    NV_STATUS status = NV_OK;
    uvm_ats_fault_context_t *ats_context = &batch_context->ats_context;
    const uvm_page_mask_t *read_fault_mask = &ats_context->faults.read_fault_mask;
    const uvm_page_mask_t *write_fault_mask = &ats_context->faults.write_fault_mask;
    const uvm_page_mask_t *reads_serviced_mask = &ats_context->faults.reads_serviced_mask;
    uvm_page_mask_t *faults_serviced_mask = &ats_context->faults.faults_serviced_mask;
    uvm_page_mask_t *accessed_mask = &ats_context->faults.accessed_mask;

    UVM_ASSERT(vma);

    ats_context->client_type = UVM_FAULT_CLIENT_TYPE_GPC;

    uvm_page_mask_or(accessed_mask, write_fault_mask, read_fault_mask);

    status = uvm_ats_service_faults(gpu_va_space, vma, base, &batch_context->ats_context);

    // Remove SW prefetched pages from the serviced mask since fault servicing
    // failures belonging to prefetch pages need to be ignored.
    uvm_page_mask_and(faults_serviced_mask, faults_serviced_mask, accessed_mask);

    UVM_ASSERT(uvm_page_mask_subset(faults_serviced_mask, accessed_mask));

    if ((status != NV_OK) || uvm_page_mask_equal(faults_serviced_mask, accessed_mask)) {
        (*block_faults) += (fault_index_end - fault_index_start);
        return status;
    }

    // Check faults_serviced_mask and reads_serviced_mask for precise fault
    // attribution after calling the ATS servicing routine. The
    // errors returned from ATS servicing routine should only be
    // global errors such as OOM or ECC.
    // uvm_parent_gpu_service_replayable_faults() handles global errors by
    // calling cancel_fault_batch(). Precise attribution isn't currently
    // supported in such cases.
    //
    // Precise fault attribution for global errors can be handled by
    // servicing one fault at a time until fault servicing encounters an
    // error.
    // TODO: Bug 3989244: Precise ATS fault attribution for global errors.
    for (i = fault_index_start; i < fault_index_end; i++) {
        uvm_page_index_t page_index;
        uvm_fault_cancel_va_mode_t cancel_va_mode;
        uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[i];
        uvm_fault_access_type_t access_type = current_entry->fault_access_type;

        page_index = (current_entry->fault_address - base) / PAGE_SIZE;

        if (uvm_page_mask_test(faults_serviced_mask, page_index)) {
            (*block_faults)++;
            continue;
        }

        if (access_type <= UVM_FAULT_ACCESS_TYPE_READ) {
            cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_ALL;
        }
        else {
            UVM_ASSERT(access_type >= UVM_FAULT_ACCESS_TYPE_WRITE);
            if (uvm_fault_access_type_mask_test(current_entry->access_type_mask, UVM_FAULT_ACCESS_TYPE_READ) &&
                !uvm_page_mask_test(reads_serviced_mask, page_index))
                cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_ALL;
            else
                cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_WRITE_AND_ATOMIC;
        }

        service_fault_batch_fatal(batch_context, i, NV_ERR_INVALID_ADDRESS, cancel_va_mode, block_faults);
    }

    return status;
}

static void start_new_sub_batch(NvU64 *sub_batch_base,
                                NvU64 address,
                                NvU32 *sub_batch_fault_index,
                                NvU32 fault_index,
                                uvm_ats_fault_context_t *ats_context)
{
    uvm_page_mask_zero(&ats_context->faults.read_fault_mask);
    uvm_page_mask_zero(&ats_context->faults.write_fault_mask);
    uvm_page_mask_zero(&ats_context->faults.prefetch_only_fault_mask);

    *sub_batch_fault_index = fault_index;
    *sub_batch_base = UVM_VA_BLOCK_ALIGN_DOWN(address);
}

static NV_STATUS service_fault_batch_ats_sub(uvm_gpu_va_space_t *gpu_va_space,
                                             struct vm_area_struct *vma,
                                             uvm_fault_service_batch_context_t *batch_context,
                                             NvU32 fault_index,
                                             NvU64 outer,
                                             NvU32 *block_faults)
{
    NV_STATUS status = NV_OK;
    NvU32 i = fault_index;
    NvU32 sub_batch_fault_index;
    NvU64 sub_batch_base;
    uvm_fault_buffer_entry_t *previous_entry = NULL;
    uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[i];
    uvm_ats_fault_context_t *ats_context = &batch_context->ats_context;
    uvm_page_mask_t *read_fault_mask = &ats_context->faults.read_fault_mask;
    uvm_page_mask_t *write_fault_mask = &ats_context->faults.write_fault_mask;
    uvm_page_mask_t *prefetch_only_fault_mask = &ats_context->faults.prefetch_only_fault_mask;
    uvm_gpu_t *gpu = gpu_va_space->gpu;
    bool replay_per_va_block =
                        (gpu->parent->fault_buffer.replayable.replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_BLOCK);

    UVM_ASSERT(vma);

    outer = min(outer, (NvU64) vma->vm_end);

    start_new_sub_batch(&sub_batch_base, current_entry->fault_address, &sub_batch_fault_index, i, ats_context);

    do {
        uvm_page_index_t page_index;
        NvU64 fault_address = current_entry->fault_address;
        uvm_fault_access_type_t access_type = current_entry->fault_access_type;
        bool is_duplicate = check_fault_entry_duplicate(current_entry, previous_entry);

        // ATS faults can't be unserviceable, since unserviceable faults require
        // GMMU PTEs.
        UVM_ASSERT(!current_entry->is_fatal);

        i++;

        update_batch_and_notify_fault(gpu,
                                      batch_context,
                                      NULL,
                                      UVM_ID_INVALID,
                                      current_entry,
                                      is_duplicate);

        // End of sub-batch. Service faults gathered so far.
        if (fault_address >= (sub_batch_base + UVM_VA_BLOCK_SIZE)) {
            UVM_ASSERT(!uvm_page_mask_empty(read_fault_mask) ||
                       !uvm_page_mask_empty(write_fault_mask) ||
                       !uvm_page_mask_empty(prefetch_only_fault_mask));

            status = service_fault_batch_ats_sub_vma(gpu_va_space,
                                                     vma,
                                                     sub_batch_base,
                                                     batch_context,
                                                     sub_batch_fault_index,
                                                     i - 1,
                                                     block_faults);
            if (status != NV_OK || replay_per_va_block)
                break;

            start_new_sub_batch(&sub_batch_base, fault_address, &sub_batch_fault_index, i - 1, ats_context);
        }

        page_index = (fault_address - sub_batch_base) / PAGE_SIZE;

        // Do not check for coalesced access type. If there are multiple
        // different accesses to an address, we can disregard the prefetch one.
        if ((access_type == UVM_FAULT_ACCESS_TYPE_PREFETCH) &&
            (uvm_fault_access_type_mask_highest(current_entry->access_type_mask) == UVM_FAULT_ACCESS_TYPE_PREFETCH))
            uvm_page_mask_set(prefetch_only_fault_mask, page_index);

        if ((access_type == UVM_FAULT_ACCESS_TYPE_READ) ||
            uvm_fault_access_type_mask_test(current_entry->access_type_mask, UVM_FAULT_ACCESS_TYPE_READ))
            uvm_page_mask_set(read_fault_mask, page_index);

        if (access_type >= UVM_FAULT_ACCESS_TYPE_WRITE)
            uvm_page_mask_set(write_fault_mask, page_index);

        previous_entry = current_entry;
        current_entry = i < batch_context->num_coalesced_faults ? batch_context->ordered_fault_cache[i] : NULL;

    } while (current_entry &&
             (current_entry->fault_address < outer) &&
             (previous_entry->va_space == current_entry->va_space));

    // Service the last sub-batch.
    if ((status == NV_OK) &&
        (!uvm_page_mask_empty(read_fault_mask) ||
         !uvm_page_mask_empty(write_fault_mask) ||
         !uvm_page_mask_empty(prefetch_only_fault_mask))) {
        status = service_fault_batch_ats_sub_vma(gpu_va_space,
                                                 vma,
                                                 sub_batch_base,
                                                 batch_context,
                                                 sub_batch_fault_index,
                                                 i,
                                                 block_faults);
    }

    return status;
}

static NV_STATUS service_fault_batch_ats(uvm_gpu_va_space_t *gpu_va_space,
                                         struct mm_struct *mm,
                                         uvm_fault_service_batch_context_t *batch_context,
                                         NvU32 first_fault_index,
                                         NvU64 outer,
                                         NvU32 *block_faults)
{
    NvU32 i;
    NV_STATUS status = NV_OK;

    for (i = first_fault_index; i < batch_context->num_coalesced_faults;) {
        uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[i];
        const uvm_fault_buffer_entry_t *previous_entry = i > first_fault_index ?
                                                                       batch_context->ordered_fault_cache[i - 1] : NULL;
        NvU64 fault_address = current_entry->fault_address;
        struct vm_area_struct *vma;
        NvU32 num_faults_before = (*block_faults);

        if (previous_entry &&
            (previous_entry->va_space != current_entry->va_space || previous_entry->gpu != current_entry->gpu))
            break;

        if (fault_address >= outer)
            break;

        vma = find_vma_intersection(mm, fault_address, fault_address + 1);
        if (!vma) {
            // Since a vma wasn't found, cancel all accesses on the page since
            // cancelling write and atomic accesses will not cancel pending read
            // faults and this can lead to a deadlock since read faults need to
            // be serviced first before cancelling write faults.
            service_fault_batch_fatal_notify(gpu_va_space->gpu,
                                             batch_context,
                                             i,
                                             NV_ERR_INVALID_ADDRESS,
                                             UVM_FAULT_CANCEL_VA_MODE_ALL,
                                             block_faults);

            // Do not fail due to logical errors.
            status = NV_OK;

            break;
        }

        status = service_fault_batch_ats_sub(gpu_va_space, vma, batch_context, i, outer, block_faults);
        if (status != NV_OK)
            break;

        i += ((*block_faults) - num_faults_before);
    }

    return status;
}

// 针对批处理中的某一个具体错误（fault_index），判断其内存类型，
// 并将其所属的整个内存块（VA Block）中的所有相关错误一次性派发处理。
// 处理完成后，将本次消化掉的错误总数通过 block_faults 指针返回给外层。
static NV_STATUS service_fault_batch_dispatch(uvm_va_space_t *va_space,
                                              uvm_gpu_va_space_t *gpu_va_space,
                                              uvm_fault_service_batch_context_t *batch_context,
                                              NvU32 fault_index,
                                              NvU32 *block_faults,
                                              bool replay_per_va_block,
                                              const bool hmm_migratable)
{
    NV_STATUS status;
    uvm_va_range_t *va_range = NULL;
    uvm_va_range_t *va_range_next = NULL;
    uvm_va_block_t *va_block;
    uvm_gpu_t *gpu = gpu_va_space->gpu;
    uvm_va_block_context_t *va_block_context =
        gpu->parent->fault_buffer.replayable.block_service_context.block_context;
    // 获取当前需要处理的这个错误记录
    uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[fault_index];
    struct mm_struct *mm = va_block_context->mm;
    NvU64 fault_address = current_entry->fault_address; // 缺页的虚拟地址

    (*block_faults) = 0; // 初始化本次派发解决的错误数为 0

    // ---------------------------------------------------------
    // 第一步：地址查表（查找虚拟地址属于哪个 Range）
    // ---------------------------------------------------------
    // 在 UVM 维护的红黑树（或者类似结构）中，查找包含此地址的“托管内存范围”
    va_range_next = uvm_va_space_iter_gmmu_mappable_first(va_space, fault_address);
    if (va_range_next && (fault_address >= va_range_next->node.start)) {
        UVM_ASSERT(fault_address < va_range_next->node.end);

        va_range = va_range_next; // 找到了！这是一个 UVM 托管的内存地址
        va_range_next = uvm_va_range_gmmu_mappable_next(va_range);
    }

    // ---------------------------------------------------------
    // 第二步：获取或创建底层内存块（VA Block）
    // ---------------------------------------------------------
    if (va_range)
        // 路线 A：这是 UVM 托管内存（cudaMallocManaged）。
        // 寻找或者创建一个底层的 VA Block（通常是 2MB 物理内存块的管理器）
        status = uvm_va_block_find_create_in_range(va_space, va_range, fault_address, &va_block);
    else if (mm)
        // 路线 B：这不是 UVM 托管内存，但进程有合法的系统内存上下文（mm）。
        // 那么尝试走 Linux 异构内存管理（HMM）路线，把普通系统内存当成显存来处理。
        status = uvm_hmm_va_block_find_create(va_space, fault_address, &va_block_context->hmm.vma, &va_block);
    else
        // 路线 C：啥都不是，这是一个野指针/无效地址！
        status = NV_ERR_INVALID_ADDRESS;

    // ---------------------------------------------------------
    // 第三步：实际分配与搬运数据
    // ---------------------------------------------------------
    if (status == NV_OK) {
        // 如果成功拿到了 VA Block（无论是 UVM 还是 HMM），进入真正的处理函数！
        // service_fault_batch_block 会去分配物理内存、拷数据、写页表，
        // 并且会把后续同属于这个 block 的错误全顺手处理掉，写入 block_faults。
        status = service_fault_batch_block(gpu, va_block, batch_context, fault_index, hmm_migratable, block_faults);
    }
    // ---------------------------------------------------------
    // 路线 D：无效地址的最后挣扎（尝试 ATS 硬件透明映射）
    // ---------------------------------------------------------
    else if ((status == NV_ERR_INVALID_ADDRESS) && uvm_ats_can_service_faults(gpu_va_space, mm)) {
        NvU64 outer = ~0ULL;

         UVM_ASSERT(replay_per_va_block ==
                    (gpu->parent->fault_buffer.replayable.replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_BLOCK));

        // 计算一个安全边界。
        // 因为在 GPU 硬件层面上，ATS（直接查主板页表）和 GMMU（GPU本地页表）
        // 不能在同一个粒度区域内混用。
        if (va_range_next) {
            outer = min(va_range_next->node.start,
                           UVM_ALIGN_DOWN(fault_address + UVM_GMMU_ATS_GRANULARITY, UVM_GMMU_ATS_GRANULARITY));
        }

        // 检查这个野指针地址是不是非法侵入了 GMMU 的保留区域。
        // 如果是，可能会引发无限缺页死循环，必须将其作为致命错误拦截掉。
        if (uvm_ats_check_in_gmmu_region(va_space, fault_address, va_range_next)) {
            // 报告致命错误并取消该错误（杀掉对应的 Kernel 线程）
            service_fault_batch_fatal_notify(gpu,
                                             batch_context,
                                             fault_index,
                                             NV_ERR_INVALID_ADDRESS,
                                             UVM_FAULT_CANCEL_VA_MODE_ALL,
                                             block_faults);

            // 注意：这里重置 status 为 NV_OK！
            // 意思是“这个非法的缺页我已经妥善处理（枪毙）了，外层驱动不用崩溃，继续跑”。
            status = NV_OK;
        }
        else {
            // 安全！走 ATS 路线处理。驱动会配置 GPU 硬件，
            // 使得该地址以后的访存直接通过 PCIe 请求 CPU 解析，不再经过 UVM。
            status = service_fault_batch_ats(gpu_va_space, mm, batch_context, fault_index, outer, block_faults);
        }
    }
    // ---------------------------------------------------------
    // 终极死亡：彻底救不活的野指针
    // ---------------------------------------------------------
    else {
        // 既不在托管内存，不支持 HMM，也不支持 ATS。这就是个纯粹的越界访问。
        service_fault_batch_fatal_notify(gpu,
                                         batch_context,
                                         fault_index,
                                         status,
                                         UVM_FAULT_CANCEL_VA_MODE_ALL,
                                         block_faults);

        // 同样，重置为 NV_OK，只杀程序不挂驱动。
        status = NV_OK;
    }

    return status;
}

// Called when a fault in the batch has been marked fatal. Flush the buffer
// under the VA and mmap locks to remove any potential stale fatal faults, then
// service all new faults for just that VA space and cancel those which are
// fatal. Faults in other VA spaces are replayed when done and will be processed
// when normal fault servicing resumes.
static NV_STATUS service_fault_batch_for_cancel(uvm_fault_service_batch_context_t *batch_context)
{
    NV_STATUS status = NV_OK;
    NvU32 i;
    uvm_va_space_t *va_space = batch_context->fatal_va_space;
    uvm_gpu_t *gpu = batch_context->fatal_gpu;
    uvm_gpu_va_space_t *gpu_va_space = NULL;
    struct mm_struct *mm;
    uvm_replayable_fault_buffer_t *replayable_faults = &gpu->parent->fault_buffer.replayable;
    uvm_service_block_context_t *service_context = &gpu->parent->fault_buffer.replayable.block_service_context;
    uvm_va_block_context_t *va_block_context = service_context->block_context;

    UVM_ASSERT(va_space);
    UVM_ASSERT(gpu);
    UVM_ASSERT(gpu->parent->replayable_faults_supported);

    // Perform the flush and re-fetch while holding the mmap_lock and the
    // VA space lock. This avoids stale faults because it prevents any vma
    // modifications (mmap, munmap, mprotect) from happening between the time HW
    // takes the fault and we cancel it.
    mm = uvm_va_space_mm_retain_lock(va_space);
    uvm_va_block_context_init(va_block_context, mm);
    uvm_va_space_down_read(va_space);

    // We saw fatal faults in this VA space before. Flush while holding
    // mmap_lock to make sure those faults come back (aren't stale).
    //
    // We need to wait until all old fault messages have arrived before
    // flushing, hence UVM_GPU_BUFFER_FLUSH_MODE_WAIT_UPDATE_PUT.
    status = fault_buffer_flush_locked(gpu->parent,
                                       gpu,
                                       UVM_GPU_BUFFER_FLUSH_MODE_WAIT_UPDATE_PUT,
                                       UVM_FAULT_REPLAY_TYPE_START,
                                       batch_context);
    if (status != NV_OK)
        goto done;

    // Wait for the flush's replay to finish to give the legitimate faults a
    // chance to show up in the buffer again.
    status = uvm_tracker_wait(&replayable_faults->replay_tracker);
    if (status != NV_OK)
        goto done;

    // We expect all replayed faults to have arrived in the buffer so we can re-
    // service them. The replay-and-wait sequence above will ensure they're all
    // in the HW buffer. When GSP owns the HW buffer, we also have to wait for
    // GSP to copy all available faults from the HW buffer into the shadow
    // buffer.
    status = hw_fault_buffer_flush_locked(gpu->parent, HW_FAULT_BUFFER_FLUSH_MODE_MOVE);
    if (status != NV_OK)
        goto done;

    // If there is no GPU VA space for the GPU, ignore all faults in the VA
    // space. This can happen if the GPU VA space has been destroyed since we
    // unlocked the VA space in service_fault_batch. That means the fatal faults
    // are stale, because unregistering the GPU VA space requires preempting the
    // context and detaching all channels in that VA space. Restart fault
    // servicing from the top.
    gpu_va_space = uvm_gpu_va_space_get(va_space, gpu);
    if (!gpu_va_space)
        goto done;

    // Re-parse the new faults
    batch_context->num_invalid_prefetch_faults = 0;
    batch_context->num_duplicate_faults        = 0;
    batch_context->num_replays                 = 0;
    batch_context->fatal_va_space              = NULL;
    batch_context->fatal_gpu                   = NULL;
    batch_context->has_throttled_faults        = false;

    status = fetch_fault_buffer_entries(gpu->parent, batch_context, FAULT_FETCH_MODE_ALL);
    if (status != NV_OK)
        goto done;

    // No more faults left. Either the previously-seen fatal entry was stale, or
    // RM killed the context underneath us.
    if (batch_context->num_cached_faults == 0)
        goto done;

    ++batch_context->batch_id;

    status = preprocess_fault_batch(gpu->parent, batch_context);
    if (status != NV_OK) {
        if (status == NV_WARN_MORE_PROCESSING_REQUIRED) {
            // Another flush happened due to stale faults or a context-fatal
            // error. The previously-seen fatal fault might not exist anymore,
            // so restart fault servicing from the top.
            status = NV_OK;
        }

        goto done;
    }

    // Search for the target VA space and GPU.
    for (i = 0; i < batch_context->num_coalesced_faults; i++) {
        uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[i];
        UVM_ASSERT(current_entry->va_space);
        if (current_entry->va_space == va_space && current_entry->gpu == gpu)
            break;
    }

    while (i < batch_context->num_coalesced_faults) {
        uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[i];

        if (current_entry->va_space != va_space || current_entry->gpu != gpu)
            break;

        // service_fault_batch_dispatch() doesn't expect unserviceable faults.
        // Just cancel them directly.
        if (current_entry->is_fatal) {
            status = cancel_fault_precise_va(current_entry, UVM_FAULT_CANCEL_VA_MODE_ALL);
            if (status != NV_OK)
                break;

            ++i;
        }
        else {
            uvm_ats_fault_invalidate_t *ats_invalidate = &gpu->parent->fault_buffer.replayable.ats_invalidate;
            NvU32 block_faults;
            const bool hmm_migratable = true;

            ats_invalidate->tlb_batch_pending = false;

            // Service all the faults that we can. We only really need to search
            // for fatal faults, but attempting to service all is the easiest
            // way to do that.
            status = service_fault_batch_dispatch(va_space, gpu_va_space, batch_context, i, &block_faults, false, hmm_migratable);
            if (status != NV_OK) {
                // TODO: Bug 3900733: clean up locking in service_fault_batch().
                // We need to drop lock and retry. That means flushing and
                // starting over.
                if (status == NV_WARN_MORE_PROCESSING_REQUIRED || status == NV_WARN_MISMATCHED_TARGET)
                    status = NV_OK;

                break;
            }

            // Invalidate TLBs before cancel to ensure that fatal faults don't
            // get stuck in HW behind non-fatal faults to the same line.
            status = uvm_ats_invalidate_tlbs(gpu_va_space, ats_invalidate, &batch_context->tracker);
            if (status != NV_OK)
                break;

            while (block_faults-- > 0) {
                current_entry = batch_context->ordered_fault_cache[i];
                if (current_entry->is_fatal) {
                    status = cancel_fault_precise_va(current_entry, current_entry->replayable.cancel_va_mode);
                    if (status != NV_OK)
                        break;
                }

                ++i;
            }
        }
    }

done:
    uvm_va_space_up_read(va_space);
    uvm_va_space_mm_release_unlock(va_space, mm);

    if (status == NV_OK) {
        // There are two reasons to flush the fault buffer here.
        //
        // 1) Functional. We need to replay both the serviced non-fatal faults
        //    and the skipped faults in other VA spaces. The former need to be
        //    restarted and the latter need to be replayed so the normal fault
        //    service mechanism can fetch and process them.
        //
        // 2) Performance. After cancelling the fatal faults, a flush removes
        //    any potential duplicated fault that may have been added while
        //    processing the faults in this batch. This flush also avoids doing
        //    unnecessary processing after the fatal faults have been cancelled,
        //    so all the rest are unlikely to remain after a replay because the
        //    context is probably in the process of dying.
        status = fault_buffer_flush_locked(gpu->parent,
                                           gpu,
                                           UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT,
                                           UVM_FAULT_REPLAY_TYPE_START,
                                           batch_context);
    }

    return status;
}

// 扫描已经排好序的缺页错误视图（ordered_fault_cache），
// 并将它们按不同的 va_blocks（统一虚拟内存的底层管理单元）进行分组。
// 对于托管内存（managed faults），按 va_block 批量进行服务。
// 对于非托管内存的错误，在扫描遇到时逐个处理。
//
// 致命错误（Fatal faults）在这里会被打上标记，留给外层调用者去处理（即我们之前讨论的挂起/取消逻辑）。
static NV_STATUS service_fault_batch(uvm_parent_gpu_t *parent_gpu,
                                     fault_service_mode_t service_mode,
                                     uvm_fault_service_batch_context_t *batch_context)
{
    NV_STATUS status = NV_OK;
    NvU32 i;
    uvm_va_space_t *va_space = NULL;           // 当前正在处理的虚拟地址空间（对应一个进程）
    uvm_gpu_va_space_t *prev_gpu_va_space = NULL; // 上一个处理的 GPU 虚拟地址空间
    uvm_ats_fault_invalidate_t *ats_invalidate = &parent_gpu->fault_buffer.replayable.ats_invalidate;
    struct mm_struct *mm = NULL;               // Linux 内核的内存描述符（代表进程的内存上下文）
    
    // 是否开启了“处理完一个 Block 就立即重试”的策略
    const bool replay_per_va_block = service_mode != FAULT_SERVICE_MODE_CANCEL &&
                                     parent_gpu->fault_buffer.replayable.replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_BLOCK;
    uvm_service_block_context_t *service_context =
        &parent_gpu->fault_buffer.replayable.block_service_context;
    uvm_va_block_context_t *va_block_context = service_context->block_context;
    bool hmm_migratable = true; // 异构内存管理（HMM）是否允许数据迁移

    UVM_ASSERT(parent_gpu->replayable_faults_supported);

    ats_invalidate->tlb_batch_pending = false;

    // 【主循环】：遍历经过预处理、排好序的错误列表。
    // 注意：这里 for 循环没有 ++i，因为 i 的递增是由内部按块（block）批量跳跃控制的。
    for (i = 0; i < batch_context->num_coalesced_faults;) {
        NvU32 block_faults; // 记录一次 dispatch 处理了多少个属于同一个块的错误
        uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[i];
        uvm_fault_utlb_info_t *utlb = &batch_context->utlbs[current_entry->fault_source.utlb_id];
        uvm_gpu_va_space_t *gpu_va_space;

        UVM_ASSERT(current_entry->va_space);

        // 【进程/地址空间切换逻辑】
        // 这就是为什么在上一步 `preprocess_fault_batch` 中要按 va_space 排序！
        // 只有当发现 va_space 变了（跨进程了），才需要执行极其昂贵的解锁和重新加锁操作。
        if (current_entry->va_space != va_space) {
            if (prev_gpu_va_space) {
                // TLB 失效指令是按 GPU VA space 下发的。
                // 既然进程都换了，先把上一个进程在 GPU 里的 TLB 缓存刷新掉。
                status = uvm_ats_invalidate_tlbs(prev_gpu_va_space, ats_invalidate, &batch_context->tracker);
                if (status != NV_OK)
                    goto fail;

                prev_gpu_va_space = NULL;
            }

            // 如果之前持有旧地址空间的锁，释放它们
            if (va_space) {
                uvm_va_space_up_read(va_space);
                uvm_va_space_mm_release_unlock(va_space, mm);
                mm = NULL;
            }

            // 更新为新的地址空间
            va_space = current_entry->va_space;

            // 获取新地址空间的锁。
            // 注意层级顺序：在 Linux 系统中，必须先获取进程 mm 的锁（mmap_lock），
            // 然后才能获取 UVM 驱动自己的 va_space 锁，否则会导致系统死锁。
            mm = uvm_va_space_mm_retain_lock(va_space);
            uvm_va_block_context_init(va_block_context, mm);

            uvm_va_space_down_read(va_space); // 加 UVM 读锁
        }

        // 【致命错误拦截】
        // 有些错误（如越界、非法的内存访问）是 UVM 驱动无法修复的。
        if (current_entry->is_fatal) {
            ++i; // 跳过这个错误
            if (!batch_context->fatal_va_space) {
                // 将致命错误的上下文记录到 batch_context 中。
                // 这就是外层函数能够检测到致命错误，并执行 tracker_wait 和精确取消(RC)的原因！
                batch_context->fatal_va_space = va_space;
                batch_context->fatal_gpu = current_entry->gpu;
            }

            utlb->has_fatal_faults = true;
            UVM_ASSERT(utlb->num_pending_faults > 0);
            continue; // 继续扫描下一个错误
        }

        gpu_va_space = uvm_gpu_va_space_get(va_space, current_entry->gpu);

        // 如果在同一个进程内，但是跨 GPU 了（例如多卡互联场景），也要刷新上一个 GPU 的 TLB
        if (prev_gpu_va_space && prev_gpu_va_space != gpu_va_space) {
            status = uvm_ats_invalidate_tlbs(prev_gpu_va_space, ats_invalidate, &batch_context->tracker);
            if (status != NV_OK)
                goto fail;
        }

        prev_gpu_va_space = gpu_va_space;

        // 如果该 GPU 没有对应的 VA space，直接忽略。
        // 这通常发生在极端竞态条件下：比如 GPU 正在被销毁/卸载，此时残留的旧错误还在 Buffer 里。
        if (!gpu_va_space) {
            ++i;
            continue;
        }

        // ==========================================================
        // 【核心派发：真正干活的地方】
        // service_fault_batch_dispatch 会处理从当前 i 开始，所有属于同一个 
        // 内存块 (VA Block) 的缺页错误。它会分配物理内存、执行 Host-to-Device 拷贝、写页表。
        // 处理完毕后，它会将实际处理掉的错误数量写回给 `block_faults`。
        // ==========================================================
        status = service_fault_batch_dispatch(va_space,
                                              gpu_va_space,
                                              batch_context,
                                              i,
                                              &block_faults, // 输出参数
                                              replay_per_va_block,
                                              hmm_migratable);

        // 【重试与回退机制 (Fallback)】
        // NV_WARN_MORE_PROCESSING_REQUIRED 表示可能遇到锁冲突或者内存不足，需要释放当前锁重试。
        // NV_WARN_MISMATCHED_TARGET 表示 HMM (异构内存管理) 迁移失败，需要禁用 migratable 标志重试。
        if (status == NV_WARN_MORE_PROCESSING_REQUIRED || status == NV_WARN_MISMATCHED_TARGET) {
            if (status == NV_WARN_MISMATCHED_TARGET)
                hmm_migratable = false; // 降级策略：不再尝试迁移到显存，可能直接用系统内存
            
            // 释放所有锁
            uvm_va_space_up_read(va_space);
            uvm_va_space_mm_release_unlock(va_space, mm);
            mm = NULL;
            va_space = NULL;
            prev_gpu_va_space = NULL;
            status = NV_OK;
            continue; // 注意这里没有 ++i，这意味着下一次循环会【重新尝试】处理当前这个失败的错误
        }

        if (status != NV_OK)
            goto fail; // 真出错了，直接完蛋

        hmm_migratable = true; // 重置 HMM 标志
        i += block_faults;     // 【批量跳跃】直接跳过刚刚被 dispatch 批量解决掉的所有同块错误！

        // 【提前重试策略】
        // 如果策略是按 Block 重试，且当前没有发生致命错误，
        // 那么刚处理完这个块，就马上告诉 GPU：“这个块的内存我搬完了，卡在这个块上的线程可以继续跑了！”
        // 这样可以最大化 GPU 硬件的并行度，不用等整个 Batch 全部弄完才重试。
        if (replay_per_va_block && !batch_context->fatal_va_space) {
            status = push_replay_on_gpu(gpu_va_space->gpu, UVM_FAULT_REPLAY_TYPE_START, batch_context);
            if (status != NV_OK)
                goto fail;

            // 因为已经发了一次 replay，所以需要增加 batch_id，作为追踪标记
            ++batch_context->batch_id;
        }
    } // 结束 for 循环

    // 善后工作：刷新最后一个 GPU 的 TLB 缓存
    if (prev_gpu_va_space) {
        NV_STATUS invalidate_status = uvm_ats_invalidate_tlbs(prev_gpu_va_space, ats_invalidate, &batch_context->tracker);
        if (invalidate_status != NV_OK)
            status = invalidate_status;
    }

fail:
    // 清理：如果在出错或结束时还持有锁，安全地释放掉
    if (va_space) {
        uvm_va_space_up_read(va_space);
        uvm_va_space_mm_release_unlock(va_space, mm);
    }

    return status;
}

// Tells if the given fault entry is the first one in its uTLB
static bool is_first_fault_in_utlb(uvm_fault_service_batch_context_t *batch_context, NvU32 fault_index)
{
    NvU32 i;
    NvU32 utlb_id = batch_context->fault_cache[fault_index].fault_source.utlb_id;

    for (i = 0; i < fault_index; ++i) {
        uvm_fault_buffer_entry_t *current_entry = &batch_context->fault_cache[i];

        // We have found a prior fault in the same uTLB
        if (current_entry->fault_source.utlb_id == utlb_id)
            return false;
    }

    return true;
}

// Compute the number of fatal and non-fatal faults for a page in the given uTLB
static void faults_for_page_in_utlb(uvm_fault_service_batch_context_t *batch_context,
                                    uvm_va_space_t *va_space,
                                    NvU64 addr,
                                    NvU32 utlb_id,
                                    NvU32 *fatal_faults,
                                    NvU32 *non_fatal_faults)
{
    uvm_gpu_t *gpu = NULL;
    NvU32 i;

    *fatal_faults = 0;
    *non_fatal_faults = 0;

    // Fault filtering is not allowed in the TLB-based fault cancel path
    UVM_ASSERT(batch_context->num_cached_faults == batch_context->num_coalesced_faults);

    for (i = 0; i < batch_context->num_cached_faults; ++i) {
        uvm_fault_buffer_entry_t *current_entry = &batch_context->fault_cache[i];

        if (!gpu)
            gpu = current_entry->gpu;
        else
            UVM_ASSERT(current_entry->gpu == gpu);

        if (current_entry->fault_source.utlb_id == utlb_id &&
            current_entry->va_space == va_space &&
            current_entry->fault_address == addr) {
            // We have found the page
            if (current_entry->is_fatal)
                ++(*fatal_faults);
            else
                ++(*non_fatal_faults);
        }
    }
}

// Function that tells if there are addresses (reminder: they are aligned to 4K)
// with non-fatal faults only
static bool no_fatal_pages_in_utlb(uvm_fault_service_batch_context_t *batch_context,
                                   NvU32 start_index,
                                   NvU32 utlb_id)
{
    NvU32 i;

    // Fault filtering is not allowed in the TLB-based fault cancel path
    UVM_ASSERT(batch_context->num_cached_faults == batch_context->num_coalesced_faults);

    for (i = start_index; i < batch_context->num_cached_faults; ++i) {
        uvm_fault_buffer_entry_t *current_entry = &batch_context->fault_cache[i];

        if (current_entry->fault_source.utlb_id == utlb_id) {
            // We have found a fault for the uTLB
            NvU32 fatal_faults;
            NvU32 non_fatal_faults;

            faults_for_page_in_utlb(batch_context,
                                    current_entry->va_space,
                                    current_entry->fault_address,
                                    utlb_id,
                                    &fatal_faults,
                                    &non_fatal_faults);

            if (non_fatal_faults > 0 && fatal_faults == 0)
                return true;
        }
    }

    return false;
}

static void record_fatal_fault_helper(uvm_fault_buffer_entry_t *entry, UvmEventFatalReason reason)
{
    uvm_va_space_t *va_space = entry->va_space;
    uvm_gpu_t *gpu = entry->gpu;

    UVM_ASSERT(va_space);
    UVM_ASSERT(gpu);

    uvm_va_space_down_read(va_space);
    // Record fatal fault event
    uvm_tools_record_gpu_fatal_fault(gpu->id, va_space, entry, reason);
    uvm_va_space_up_read(va_space);
}

// This function tries to find and issue a cancel for each uTLB that meets
// the requirements to guarantee precise fault attribution:
// - No new faults can arrive on the uTLB (uTLB is in lockdown)
// - The first fault in the buffer for a specific uTLB is fatal
// - There are no other addresses in the uTLB with non-fatal faults only
//
// This function and the related helpers iterate over faults as read from HW,
// not through the ordered fault view
//
// TODO: Bug 1766754
// This is very costly, although not critical for performance since we are
// cancelling.
// - Build a list with all the faults within a uTLB
// - Sort by uTLB id
static NV_STATUS try_to_cancel_utlbs(uvm_fault_service_batch_context_t *batch_context)
{
    NvU32 i;

    // Fault filtering is not allowed in the TLB-based fault cancel path
    UVM_ASSERT(batch_context->num_cached_faults == batch_context->num_coalesced_faults);

    for (i = 0; i < batch_context->num_cached_faults; ++i) {
        uvm_fault_buffer_entry_t *current_entry = &batch_context->fault_cache[i];
        uvm_fault_utlb_info_t *utlb = &batch_context->utlbs[current_entry->fault_source.utlb_id];
        NvU32 gpc_id = current_entry->fault_source.gpc_id;
        NvU32 utlb_id = current_entry->fault_source.utlb_id;
        NvU32 client_id = current_entry->fault_source.client_id;
        uvm_gpu_t *gpu = current_entry->gpu;

        // Only fatal faults are considered
        if (!current_entry->is_fatal)
            continue;

        // Only consider uTLBs in lock-down
        if (!utlb->in_lockdown)
            continue;

        // Issue a single cancel per uTLB
        if (utlb->cancelled)
            continue;

        if (is_first_fault_in_utlb(batch_context, i) &&
            !no_fatal_pages_in_utlb(batch_context, i + 1, utlb_id)) {
            NV_STATUS status;

            record_fatal_fault_helper(current_entry, current_entry->fatal_reason);

            status = push_cancel_on_gpu_targeted(gpu,
                                                 current_entry->instance_ptr,
                                                 gpc_id,
                                                 client_id,
                                                 &batch_context->tracker);
            if (status != NV_OK)
                return status;

            utlb->cancelled = true;
        }
    }

    return NV_OK;
}

static NvU32 find_fatal_fault_in_utlb(uvm_fault_service_batch_context_t *batch_context,
                                      NvU32 utlb_id)
{
    NvU32 i;

    // Fault filtering is not allowed in the TLB-based fault cancel path
    UVM_ASSERT(batch_context->num_cached_faults == batch_context->num_coalesced_faults);

    for (i = 0; i < batch_context->num_cached_faults; ++i) {
        if (batch_context->fault_cache[i].is_fatal &&
            batch_context->fault_cache[i].fault_source.utlb_id == utlb_id)
            return i;
    }

    return i;
}

static NvU32 is_fatal_fault_in_buffer(uvm_fault_service_batch_context_t *batch_context,
                                      uvm_fault_buffer_entry_t *fault)
{
    NvU32 i;

    // Fault filtering is not allowed in the TLB-based fault cancel path
    UVM_ASSERT(batch_context->num_cached_faults == batch_context->num_coalesced_faults);

    for (i = 0; i < batch_context->num_cached_faults; ++i) {
        uvm_fault_buffer_entry_t *current_entry = &batch_context->fault_cache[i];
        if (cmp_fault_instance_ptr(current_entry, fault) == 0 &&
            current_entry->fault_address == fault->fault_address &&
            current_entry->fault_access_type == fault->fault_access_type &&
            current_entry->fault_source.utlb_id == fault->fault_source.utlb_id) {
            return true;
        }
    }

    return false;
}

// Cancel all faults in the given fault service batch context, even those not
// marked as fatal.
static NV_STATUS cancel_faults_all(uvm_fault_service_batch_context_t *batch_context, UvmEventFatalReason reason)
{
    NV_STATUS status = NV_OK;
    NV_STATUS fault_status;
    uvm_gpu_t *gpu = NULL;
    NvU32 i = 0;

    UVM_ASSERT(reason != UvmEventFatalReasonInvalid);
    UVM_ASSERT(batch_context->num_coalesced_faults > 0);

    while (i < batch_context->num_coalesced_faults && status == NV_OK) {
        uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[i];
        uvm_va_space_t *va_space = current_entry->va_space;
        bool skip_gpu_va_space;

        gpu = current_entry->gpu;
        UVM_ASSERT(gpu);
        UVM_ASSERT(va_space);

        uvm_va_space_down_read(va_space);

        // If there is no GPU VA space for the GPU, ignore all faults in
        // that GPU VA space. This can happen if the GPU VA space has been
        // destroyed since we unlocked the VA space in service_fault_batch.
        // Ignoring the fault avoids targetting a PDB that might have been
        // reused by another process.
        skip_gpu_va_space = !uvm_gpu_va_space_get(va_space, gpu);

        for (;
             i < batch_context->num_coalesced_faults &&
                 current_entry->va_space == va_space &&
                 current_entry->gpu == gpu;
             current_entry = batch_context->ordered_fault_cache[++i]) {
            uvm_fault_cancel_va_mode_t cancel_va_mode;

            if (skip_gpu_va_space)
                continue;

            if (current_entry->is_fatal) {
                UVM_ASSERT(current_entry->fatal_reason != UvmEventFatalReasonInvalid);
                cancel_va_mode = current_entry->replayable.cancel_va_mode;
            }
            else {
                current_entry->fatal_reason = reason;
                cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_ALL;
            }

            status = cancel_fault_precise_va(current_entry, cancel_va_mode);
            if (status != NV_OK)
                break;
        }

        uvm_va_space_up_read(va_space);
    }

    // Because each cancel itself triggers a replay, there may be a large number
    // of new duplicated faults in the buffer after cancelling all the known
    // ones. Flushing the buffer discards them to avoid unnecessary processing.
    // Note that we are using one of the GPUs with a fault, but the choice of
    // which one is arbitrary.
    fault_status = fault_buffer_flush_locked(gpu->parent,
                                             gpu,
                                             UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT,
                                             UVM_FAULT_REPLAY_TYPE_START,
                                             batch_context);

    // We report the first encountered error.
    if (status == NV_OK)
        status = fault_status;

    return status;
}

// Function called when the system has found a global error and needs to
// trigger RC in RM.
static void cancel_fault_batch_tlb(uvm_fault_service_batch_context_t *batch_context, UvmEventFatalReason reason)
{
    NvU32 i;

    for (i = 0; i < batch_context->num_coalesced_faults; ++i) {
        NV_STATUS status = NV_OK;
        uvm_fault_buffer_entry_t *current_entry;
        uvm_fault_buffer_entry_t *coalesced_entry;
        uvm_va_space_t *va_space;
        uvm_gpu_t *gpu;

        current_entry = batch_context->ordered_fault_cache[i];
        va_space = current_entry->va_space;
        gpu = current_entry->gpu;

        // The list iteration below skips the entry used as 'head'.
        // Report the 'head' entry explicitly.
        uvm_va_space_down_read(va_space);
        uvm_tools_record_gpu_fatal_fault(gpu->id, va_space, current_entry, reason);

        list_for_each_entry(coalesced_entry, &current_entry->merged_instances_list, merged_instances_list)
            uvm_tools_record_gpu_fatal_fault(gpu->id, va_space, coalesced_entry, reason);
        uvm_va_space_up_read(va_space);

        // We need to cancel each instance pointer to correctly handle faults
        // from multiple contexts.
        status = push_cancel_on_gpu_global(gpu, current_entry->instance_ptr, &batch_context->tracker);
        if (status != NV_OK)
            break;
    }
}

static void cancel_fault_batch(uvm_parent_gpu_t *parent_gpu,
                               uvm_fault_service_batch_context_t *batch_context,
                               UvmEventFatalReason reason)
{
    // Return code is ignored since we're on a global error path and wouldn't be
    // able to recover anyway.
    if (parent_gpu->fault_cancel_va_supported)
        cancel_faults_all(batch_context, reason);
    else
        cancel_fault_batch_tlb(batch_context, reason);
}


// Current fault cancel algorithm
//
// 1- Disable prefetching to avoid new requests keep coming and flooding the
// buffer.
// LOOP
//   2- Record one fatal fault per uTLB to check if it shows up after the replay
//   3- Flush fault buffer (REPLAY_TYPE_START_ACK_ALL to prevent new faults from
//      coming to TLBs with pending faults)
//   4- Wait for replay to finish
//   5- Fetch all faults from buffer
//   6- Check what uTLBs are in lockdown mode and can be cancelled
//   7- Preprocess faults (order per va_space, fault address, access type)
//   8- Service all non-fatal faults and mark all non-serviceable faults as
//      fatal.
//      8.1- If fatal faults are not found, we are done
//   9- Search for a uTLB which can be targeted for cancel, as described in
//      try_to_cancel_utlbs. If found, cancel it.
// END LOOP
// 10- Re-enable prefetching
//
// NOTE: prefetch faults MUST NOT trigger fault cancel. We make sure that no
// prefetch faults are left in the buffer by disabling prefetching and
// flushing the fault buffer afterwards (prefetch faults are not replayed and,
// therefore, will not show up again)
static NV_STATUS cancel_faults_precise_tlb(uvm_gpu_t *gpu, uvm_fault_service_batch_context_t *batch_context)
{
    NV_STATUS status;
    NV_STATUS tracker_status;
    uvm_replayable_fault_buffer_t *replayable_faults = &gpu->parent->fault_buffer.replayable;
    bool first = true;

    UVM_ASSERT(gpu->parent->replayable_faults_supported);

    // 1) Disable prefetching to avoid new requests keep coming and flooding
    //    the buffer
    if (gpu->parent->fault_buffer.prefetch_faults_enabled)
        gpu->parent->arch_hal->disable_prefetch_faults(gpu->parent);

    while (1) {
        NvU32 utlb_id;

        // 2) Record one fatal fault per uTLB to check if it shows up after
        // the replay. This is used to handle the case in which the uTLB is
        // being cancelled from behind our backs by RM. See the comment in
        // step 6.
        for (utlb_id = 0; utlb_id <= batch_context->max_utlb_id; ++utlb_id) {
            uvm_fault_utlb_info_t *utlb = &batch_context->utlbs[utlb_id];

            if (!first && utlb->has_fatal_faults) {
                NvU32 idx = find_fatal_fault_in_utlb(batch_context, utlb_id);
                UVM_ASSERT(idx < batch_context->num_cached_faults);

                utlb->prev_fatal_fault = batch_context->fault_cache[idx];
            }
            else {
                utlb->prev_fatal_fault.fault_address = (NvU64)-1;
            }
        }
        first = false;

        // 3) Flush fault buffer. After this call, all faults from any of the
        // faulting uTLBs are before PUT. New faults from other uTLBs can keep
        // arriving. Therefore, in each iteration we just try to cancel faults
        // from uTLBs that contained fatal faults in the previous iterations
        // and will cause the TLB to stop generating new page faults after the
        // following replay with type UVM_FAULT_REPLAY_TYPE_START_ACK_ALL.
        //
        // No need to use UVM_GPU_BUFFER_FLUSH_MODE_WAIT_UPDATE_PUT since we
        // don't care too much about old faults, just new faults from uTLBs
        // which faulted before the replay.
        status = fault_buffer_flush_locked(gpu->parent,
                                           gpu,
                                           UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT,
                                           UVM_FAULT_REPLAY_TYPE_START_ACK_ALL,
                                           batch_context);
        if (status != NV_OK)
            break;

        // 4) Wait for replay to finish
        status = uvm_tracker_wait(&replayable_faults->replay_tracker);
        if (status != NV_OK)
            break;

        batch_context->num_invalid_prefetch_faults = 0;
        batch_context->num_replays                 = 0;
        batch_context->fatal_va_space              = NULL;
        batch_context->fatal_gpu                   = NULL;
        batch_context->has_throttled_faults        = false;

        // 5) Fetch all faults from buffer
        status = fetch_fault_buffer_entries(gpu->parent, batch_context, FAULT_FETCH_MODE_ALL);
        if (status != NV_OK)
            break;

        ++batch_context->batch_id;

        UVM_ASSERT(batch_context->num_cached_faults == batch_context->num_coalesced_faults);

        // No more faults left, we are done
        if (batch_context->num_cached_faults == 0)
            break;

        // 6) Check what uTLBs are in lockdown mode and can be cancelled
        for (utlb_id = 0; utlb_id <= batch_context->max_utlb_id; ++utlb_id) {
            uvm_fault_utlb_info_t *utlb = &batch_context->utlbs[utlb_id];

            utlb->in_lockdown = false;
            utlb->cancelled   = false;

            if (utlb->prev_fatal_fault.fault_address != (NvU64)-1) {
                // If a previously-reported fault shows up again we can "safely"
                // assume that the uTLB that contains it is in lockdown mode
                // and no new translations will show up before cancel.
                // A fatal fault could only be removed behind our backs by RM
                // issuing a cancel, which only happens when RM is resetting the
                // engine. That means the instance pointer can't generate any
                // new faults, so we won't have an ABA problem where a new
                // fault arrives with the same state.
                if (is_fatal_fault_in_buffer(batch_context, &utlb->prev_fatal_fault))
                    utlb->in_lockdown = true;
            }
        }

        // 7) Preprocess faults
        status = preprocess_fault_batch(gpu->parent, batch_context);
        if (status == NV_WARN_MORE_PROCESSING_REQUIRED)
            continue;
        else if (status != NV_OK)
            break;

        // 8) Service all non-fatal faults and mark all non-serviceable faults
        // as fatal
        status = service_fault_batch(gpu->parent, FAULT_SERVICE_MODE_CANCEL, batch_context);
        UVM_ASSERT(batch_context->num_replays == 0);
        if (status == NV_ERR_NO_MEMORY)
            continue;
        else if (status != NV_OK)
            break;

        // No more fatal faults left, we are done
        if (!batch_context->fatal_va_space)
            break;

        // 9) Search for uTLBs that contain fatal faults and meet the
        // requirements to be cancelled
        try_to_cancel_utlbs(batch_context);
    }

    // 10) Re-enable prefetching
    if (gpu->parent->fault_buffer.prefetch_faults_enabled)
        gpu->parent->arch_hal->enable_prefetch_faults(gpu->parent);

    if (status == NV_OK)
        status = push_replay_on_gpu(gpu, UVM_FAULT_REPLAY_TYPE_START, batch_context);

    tracker_status = uvm_tracker_wait(&batch_context->tracker);

    return status == NV_OK? tracker_status: status;
}

static NV_STATUS cancel_faults_precise(uvm_fault_service_batch_context_t *batch_context)
{
    uvm_gpu_t *gpu;

    UVM_ASSERT(batch_context->fatal_va_space);
    UVM_ASSERT(batch_context->fatal_gpu);

    gpu = batch_context->fatal_gpu;
    if (gpu->parent->fault_cancel_va_supported)
        return service_fault_batch_for_cancel(batch_context);

    return cancel_faults_precise_tlb(gpu, batch_context);
}

static void enable_disable_prefetch_faults(uvm_parent_gpu_t *parent_gpu,
                                           uvm_fault_service_batch_context_t *batch_context)
{
    if (!parent_gpu->prefetch_fault_supported)
        return;

    // If more than 66% of faults are invalid prefetch accesses, disable
    // prefetch faults for a while.
    // num_invalid_prefetch_faults may be higher than the actual count. See the
    // comment in mark_fault_invalid_prefetch(..).
    // Some tests rely on this logic (and ratio) to correctly disable prefetch
    // fault reporting. If the logic changes, the tests will have to be changed.
    if (parent_gpu->fault_buffer.prefetch_faults_enabled &&
        uvm_perf_reenable_prefetch_faults_lapse_msec > 0 &&
        ((batch_context->num_invalid_prefetch_faults * 3 > parent_gpu->fault_buffer.max_batch_size * 2) ||
         (uvm_enable_builtin_tests &&
          parent_gpu->rm_info.isSimulated &&
          batch_context->num_invalid_prefetch_faults > 5))) {
        uvm_parent_gpu_disable_prefetch_faults(parent_gpu);
    }
    else if (!parent_gpu->fault_buffer.prefetch_faults_enabled) {
        NvU64 lapse = NV_GETTIME() - parent_gpu->fault_buffer.disable_prefetch_faults_timestamp;

        // Reenable prefetch faults after some time
        if (lapse > ((NvU64)uvm_perf_reenable_prefetch_faults_lapse_msec * (1000 * 1000)))
            uvm_parent_gpu_enable_prefetch_faults(parent_gpu);
    }
}

// 处理 GPU 上可重试错误（主要是 GPU 缺页错误 Page Faults）的主服务函数
void uvm_parent_gpu_service_replayable_faults(uvm_parent_gpu_t *parent_gpu)
{
    NvU32 num_replays = 0;   // 记录发送给 GPU 的重试(Replay)指令数量
    NvU32 num_batches = 0;   // 记录已处理的错误批次数量
    NvU32 num_throttled = 0; // 记录被限流(throttled)的错误数量
    NV_STATUS status = NV_OK;
    
    // 获取可重试错误缓冲区及当前批次处理的上下文
    uvm_replayable_fault_buffer_t *replayable_faults = &parent_gpu->fault_buffer.replayable;
    uvm_fault_service_batch_context_t *batch_context = &replayable_faults->batch_service_context;

    // 断言确保当前 GPU 支持可重试错误（现代 NVIDIA GPU 架构均支持）
    UVM_ASSERT(parent_gpu->replayable_faults_supported);

    // 初始化追踪器，用于同步异步的 GPU 操作
    uvm_tracker_init(&batch_context->tracker);

    // 主循环：持续处理缓冲区中的所有页错误
    while (1) {
        NvU64 total_kv_faults_before = parent_gpu->fault_buffer.replayable.stats.num_kv_faults;
        NvU64 total_kv_duplicates_before = parent_gpu->fault_buffer.replayable.stats.num_kv_duplicate_faults;

        // 退出条件 1：防止占用 CPU 太久。如果限流次数或处理批次达到上限，暂时退出，让出 CPU
        if (num_throttled >= uvm_perf_fault_max_throttle_per_service ||
            num_batches >= uvm_perf_fault_max_batches_per_service) {
            break;
        }

        // 重置当前批次的统计信息
        batch_context->num_invalid_prefetch_faults = 0;
        batch_context->num_duplicate_faults        = 0; // 重复的页错误数（多个线程访问同一缺失页）
        batch_context->num_kv_faults               = 0;
        batch_context->num_kv_duplicate_faults     = 0;
        batch_context->num_replays                 = 0;
        batch_context->fatal_va_space              = NULL;
        batch_context->fatal_gpu                   = NULL;
        batch_context->has_throttled_faults        = false;

        // 核心步骤 1：从 GPU 的硬件 Fault Buffer 中拉取一批错误条目到内存上下文中
        status = fetch_fault_buffer_entries(parent_gpu, batch_context, FAULT_FETCH_MODE_BATCH_READY);
        if (status != NV_OK)
            break; // 读取失败则退出

        // 如果没有拉取到任何错误，说明缓冲区已空，退出循环
        if (batch_context->num_cached_faults == 0)
            break;

        ++batch_context->batch_id; // 增加批次 ID

        // 核心步骤 2：预处理这批错误（例如：排序、合并去重，因为一个 Warp 里的多个线程可能触发同一个页错误）
        status = preprocess_fault_batch(parent_gpu, batch_context);

        num_replays += batch_context->num_replays;

        // 处理预处理返回的状态
        if (status == NV_WARN_MORE_PROCESSING_REQUIRED)
            continue; // 需要更多处理，跳入下一次循环
        else if (status != NV_OK)
            break;    // 出现严重错误，退出

        // 核心步骤 3：实际服务这批缺页错误！
        // 这里面会进行实际的统一内存（UVM）操作：分配物理内存、数据迁移（Host to Device）、建立 GPU 页表映射等。
        status = service_fault_batch(parent_gpu, FAULT_SERVICE_MODE_REGULAR, batch_context);

        // UVM_PERF_FAULT_REPLAY_POLICY_BLOCK 策略下，或者缓冲区被刷新时，可能已经发出了重试指令
        num_replays += batch_context->num_replays;

        // 根据策略启用或禁用预取(prefetch)引起的缺页错误
        enable_disable_prefetch_faults(parent_gpu, batch_context);

        batch_context->num_kv_faults = parent_gpu->fault_buffer.replayable.stats.num_kv_faults >= total_kv_faults_before ?
                                       (NvU32)(parent_gpu->fault_buffer.replayable.stats.num_kv_faults - total_kv_faults_before) :
                                       0;
        batch_context->num_kv_duplicate_faults =
            parent_gpu->fault_buffer.replayable.stats.num_kv_duplicate_faults >= total_kv_duplicates_before ?
            (NvU32)(parent_gpu->fault_buffer.replayable.stats.num_kv_duplicate_faults - total_kv_duplicates_before) :
            0;

        if (uvm_perf_fault_log_counters) {
            log_replayable_fault_counters(parent_gpu,
                                          batch_context->num_cached_faults,
                                          batch_context->num_duplicate_faults,
                                          batch_context->num_kv_faults,
                                          batch_context->num_kv_duplicate_faults);
        }

        // 如果在服务错误(分配内存/映射)时发生严重错误（如 OOM 内存耗尽 或 ECC 硬件错误）
        if (status != NV_OK) {
            // 无条件取消这批错误以触发 RC (RC = Recovery/Reset, 错误恢复)。
            // 这种全局错误无法精确定位到某个操作，直接将状态转化为致命错误并取消处理。
            cancel_fault_batch(parent_gpu, batch_context, uvm_tools_status_to_fatal_fault_reason(status));
            break;
        }

        // 处理特定虚拟地址空间(VA space)的致命错误（比如越界访问、非法的指针解引用）
        if (batch_context->fatal_va_space) {
            // 等待当前挂起的操作完成
            status = uvm_tracker_wait(&batch_context->tracker);
            if (status == NV_OK) {
                // 精确取消引发致命错误的那些操作（通常会导致触发该错误的 CUDA kernel 崩溃，但不会挂掉整个 GPU）
                status = cancel_faults_precise(batch_context);
                if (status == NV_OK) {
                    // 取消操作至少会发送一个 replay 指令给 GPU 让其抛出异常
                    UVM_ASSERT(batch_context->num_replays > 0);
                    ++num_batches;
                    continue; // 继续处理其他正常的错误
                }
            }
            break;
        }

        // 核心步骤 4：根据配置的策略，告诉 GPU 重新尝试执行之前失败的访存指令 (Replay)
        if (replayable_faults->replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_BATCH) {
            // 策略：每处理完一个批次，就发送一次 Replay 信号给 GPU
            status = push_replay_on_parent_gpu(parent_gpu, UVM_FAULT_REPLAY_TYPE_START, batch_context);
            if (status != NV_OK)
                break;
            ++num_replays;
        }
        else if (replayable_faults->replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_BATCH_FLUSH) {
            // 策略：批次刷新模式，根据重复错误的比例决定是仅仅放入缓存还是立即更新
            uvm_gpu_buffer_flush_mode_t flush_mode = UVM_GPU_BUFFER_FLUSH_MODE_CACHED_PUT;

            if (batch_context->num_duplicate_faults * 100 >
                batch_context->num_cached_faults * replayable_faults->replay_update_put_ratio) {
                flush_mode = UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT;
            }

            // 锁定并刷新 Fault Buffer
            status = fault_buffer_flush_locked(parent_gpu, NULL, flush_mode, UVM_FAULT_REPLAY_TYPE_START, batch_context);
            if (status != NV_OK)
                break;
            ++num_replays;
            // 等待 replay 指令完成
            status = uvm_tracker_wait(&replayable_faults->replay_tracker);
            if (status != NV_OK)
                break;
        }

        if (batch_context->has_throttled_faults)
            ++num_throttled;

        ++num_batches;
    } // 结束 while (1)

    if (status == NV_WARN_MORE_PROCESSING_REQUIRED)
        status = NV_OK;

    // 善后处理：如果策略是 "只回复一次(ONCE)" 或者整个过程完全没有下发过 replay 指令，
    // 这里必须保底发送至少一次 replay。因为有些错误可能没出现在缓冲区里（比如被硬件合并了），
    // 必须发送重试信号解锁 GPU 上被挂起的 Warp。
    if ((status == NV_OK && replayable_faults->replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_ONCE) ||
        num_replays == 0)
        status = push_replay_on_parent_gpu(parent_gpu, UVM_FAULT_REPLAY_TYPE_START, batch_context);

    // 清理追踪器
    uvm_tracker_deinit(&batch_context->tracker);

    // 如果最终状态是非正常退出，打印内核调试信息
    if (status != NV_OK)
        UVM_DBG_PRINT("Error servicing replayable faults on GPU: %s\n", uvm_parent_gpu_name(parent_gpu));
}

void uvm_parent_gpu_enable_prefetch_faults(uvm_parent_gpu_t *parent_gpu)
{
    UVM_ASSERT(parent_gpu->isr.replayable_faults.handling);
    UVM_ASSERT(parent_gpu->prefetch_fault_supported);

    if (!parent_gpu->fault_buffer.prefetch_faults_enabled) {
        parent_gpu->arch_hal->enable_prefetch_faults(parent_gpu);
        parent_gpu->fault_buffer.prefetch_faults_enabled = true;
    }
}

void uvm_parent_gpu_disable_prefetch_faults(uvm_parent_gpu_t *parent_gpu)
{
    UVM_ASSERT(parent_gpu->isr.replayable_faults.handling);
    UVM_ASSERT(parent_gpu->prefetch_fault_supported);

    if (parent_gpu->fault_buffer.prefetch_faults_enabled) {
        parent_gpu->arch_hal->disable_prefetch_faults(parent_gpu);
        parent_gpu->fault_buffer.prefetch_faults_enabled = false;
        parent_gpu->fault_buffer.disable_prefetch_faults_timestamp = NV_GETTIME();
    }
}

const char *uvm_perf_fault_replay_policy_string(uvm_perf_fault_replay_policy_t replay_policy)
{
    BUILD_BUG_ON(UVM_PERF_FAULT_REPLAY_POLICY_MAX != 4);

    switch (replay_policy) {
        UVM_ENUM_STRING_CASE(UVM_PERF_FAULT_REPLAY_POLICY_BLOCK);
        UVM_ENUM_STRING_CASE(UVM_PERF_FAULT_REPLAY_POLICY_BATCH);
        UVM_ENUM_STRING_CASE(UVM_PERF_FAULT_REPLAY_POLICY_BATCH_FLUSH);
        UVM_ENUM_STRING_CASE(UVM_PERF_FAULT_REPLAY_POLICY_ONCE);
        UVM_ENUM_STRING_DEFAULT();
    }
}

NV_STATUS uvm_test_get_prefetch_faults_reenable_lapse(UVM_TEST_GET_PREFETCH_FAULTS_REENABLE_LAPSE_PARAMS *params,
                                                      struct file *filp)
{
    params->reenable_lapse = uvm_perf_reenable_prefetch_faults_lapse_msec;

    return NV_OK;
}

NV_STATUS uvm_test_set_prefetch_faults_reenable_lapse(UVM_TEST_SET_PREFETCH_FAULTS_REENABLE_LAPSE_PARAMS *params,
                                                      struct file *filp)
{
    uvm_perf_reenable_prefetch_faults_lapse_msec = params->reenable_lapse;

    return NV_OK;
}

NV_STATUS uvm_test_drain_replayable_faults(UVM_TEST_DRAIN_REPLAYABLE_FAULTS_PARAMS *params, struct file *filp)
{
    uvm_gpu_t *gpu;
    NV_STATUS status = NV_OK;
    uvm_spin_loop_t spin;
    bool pending = true;
    uvm_va_space_t *va_space = uvm_va_space_get(filp);

    gpu = uvm_va_space_retain_gpu_by_uuid(va_space, &params->gpu_uuid);
    if (!gpu)
        return NV_ERR_INVALID_DEVICE;

    uvm_spin_loop_init(&spin);

    do {
        uvm_parent_gpu_replayable_faults_isr_lock(gpu->parent);
        pending = uvm_parent_gpu_replayable_faults_pending(gpu->parent);
        uvm_parent_gpu_replayable_faults_isr_unlock(gpu->parent);

        if (!pending)
            break;

        if (fatal_signal_pending(current)) {
            status = NV_ERR_SIGNAL_PENDING;
            break;
        }

        UVM_SPIN_LOOP(&spin);
    } while (uvm_spin_loop_elapsed(&spin) < params->timeout_ns);

    if (pending && status == NV_OK)
        status = NV_ERR_TIMEOUT;

    uvm_gpu_release(gpu);

    return status;
}

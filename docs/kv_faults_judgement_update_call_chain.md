# `batch_kv_faults` 等 KV 字段的“判断 + 更新”函数调用链

本文专门回答这句注释对应的数据来源：

```c
batch_kv_faults, // 本批次：KV类总缺页数
```

并给出 KV 相关字段在内核中的完整调用链、更新时机和计算公式。

---

## 1. 先看结论

`batch_kv_faults` 不是直接“计数器自增”得到的，它是：

1. 先在 fault 事件回调里累计全局 `total_kv_faults`（`num_kv_faults`）。
2. 再在 `uvm_parent_gpu_service_replayable_faults()` 每一轮处理结束时，用
   `本轮后 total_kv_faults - 本轮前 total_kv_faults`
   计算出本轮增量，赋值给 `batch_context->num_kv_faults`。
3. 最后作为 `log_replayable_fault_counters(...)` 的第 4 个参数打印成 `batch_kv_faults`。

---

## 2. KV 判断函数（“是不是 KV fault”）

## 2.1 判定函数

- `uvm_perf_fault_address_is_in_kv_cache(NvU64 fault_address)`  
  文件：`kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.c:147`

判定规则（闭区间）：

1. `uvm_perf_fault_kv_range_enable == 1`
2. `kv_start != 0 && kv_end != 0 && kv_end >= kv_start`
3. `kv_start <= fault_address <= kv_end`

相关参数定义：

- `uvm_perf_fault_kv_range_enable`：`:136`
- `uvm_perf_fault_kv_start`：`:141`
- `uvm_perf_fault_kv_end`：`:144`

---

## 3. 全局 KV 计数是如何更新的（total 级）

## 3.1 触发入口：fault 服务路径发 perf fault 事件

在 replayable fault 处理过程中，多个路径都会调用：

- `update_batch_and_notify_fault(...)`  
  定义：`uvm_gpu_replayable_faults.c:1517`

常见调用点：

1. block 路径：`uvm_gpu_replayable_faults.c:1831`
2. fatal notify 路径：`uvm_gpu_replayable_faults.c:2089`
3. ATS 路径：`uvm_gpu_replayable_faults.c:2224`

该函数内部调用：

- `uvm_perf_event_notify_gpu_fault(...)`  
  调用点：`uvm_gpu_replayable_faults.c:1548`  
  定义：`uvm_perf_events.h:357`

`uvm_perf_event_notify_gpu_fault()` 再转发：

- `uvm_perf_event_notify(..., UVM_PERF_EVENT_FAULT, ...)`  
  `uvm_perf_events.h:380`

## 3.2 回调注册（否则不会更新 total）

在 VA space 初始化时注册 fault 回调：

- `uvm_gpu_init_va_space(...)`  
  文件：`uvm_gpu.c:2025`
- 注册：`UVM_PERF_EVENT_FAULT -> update_stats_fault_cb`  
  `uvm_gpu.c:2030-2032`

注意：注册在 `if (uvm_procfs_is_debug_enabled())` 里（`uvm_gpu.c:2029`），如果该条件不成立，这条统计链不会跑。

## 3.3 真正的 total 更新函数

回调函数：

- `update_stats_fault_cb(...)`  
  文件：`uvm_gpu.c:1895`

它会调用：

- `update_stats_parent_gpu_fault_instance(...)`  
  文件：`uvm_gpu.c:1830`

在该函数中：

1. 调用 KV 判定：
   - `is_kv_fault = uvm_perf_fault_address_is_in_kv_cache(fault_entry->fault_address);`
   - 位置：`uvm_gpu.c:1883`
2. 如果是 KV fault：
   - `++parent_gpu->fault_buffer.replayable.stats.num_kv_faults;`
   - 位置：`uvm_gpu.c:1885`
3. 若同时是 duplicate（`is_duplicate || fault_entry->filtered`）：
   - `++num_kv_duplicate_faults`
   - 位置：`uvm_gpu.c:1890`

这一步更新的是 **total** 计数：

- `num_kv_faults`
- `num_kv_duplicate_faults`

---

## 4. `batch_kv_faults` 是如何从 total 推导出来的（batch 级）

函数：

- `uvm_parent_gpu_service_replayable_faults(...)`  
  文件：`uvm_gpu_replayable_faults.c:3352`

每轮 while 的关键步骤：

1. 记录“本轮开始前”的 total：
   - `total_kv_faults_before`
   - `total_kv_duplicates_before`
   - 位置：`uvm_gpu_replayable_faults.c:3371-3372`

2. 执行一轮 fault 服务：
   - `fetch_fault_buffer_entries` -> `preprocess_fault_batch` -> `service_fault_batch`
   - 位置：`3391`, `3402`, `3414`

3. 用差分得到本轮 batch 值：
   - `batch_context->num_kv_faults = total_after - total_before`
   - `batch_context->num_kv_duplicate_faults = total_after - total_before`
   - 位置：`3422-3428`

4. 打印时把这两个值传入：
   - `log_replayable_fault_counters(..., batch_context->num_kv_faults, batch_context->num_kv_duplicate_faults)`
   - 位置：`3431-3435`

因此：

- `batch_kv_faults == 本轮 total_kv_faults 的增量`
- `batch_kv_duplicates == 本轮 total_kv_duplicate_faults 的增量`

---

## 5. 打印字段如何对应到变量

打印函数：

- `log_replayable_fault_counters(...)`  
  文件：`uvm_gpu_replayable_faults.c:229`

参数对应：

1. `batch_faults`：`batch_context->num_cached_faults`
2. `batch_duplicates`：`batch_context->num_duplicate_faults`
3. `batch_kv_faults`：`batch_context->num_kv_faults`（差分结果）
4. `batch_kv_duplicates`：`batch_context->num_kv_duplicate_faults`（差分结果）

函数内衍生字段：

1. `batch_kv_after_dedup = batch_kv_faults - batch_kv_duplicates`（防下溢保护）
2. `batch_kv_ratio = batch_kv_faults / batch_faults`
3. `batch_kv_after_dedup_ratio = batch_kv_after_dedup / batch_after_dedup`

相关位置：

- `batch_kv_after_dedup`：`uvm_gpu_replayable_faults.c:241`
- 比例计算：`:249-250`
- total 读取：`:260-265`

---

## 6. 相关数据结构（batch 与 total 存放位置）

文件：`kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_gpu.h`

1. batch context：
   - `NvU32 num_kv_faults;`（`:323`）
   - `NvU32 num_kv_duplicate_faults;`（`:325`）

2. replayable stats（total）：
   - `NvU64 num_kv_faults;`（`:402`）
   - `NvU64 num_kv_duplicate_faults;`（`:404`）

---

## 7. 一张图看完调用链

```text
service_fault_batch_* paths
  -> update_batch_and_notify_fault()                      [uvm_gpu_replayable_faults.c]
     -> uvm_perf_event_notify_gpu_fault()                 [uvm_perf_events.h]
        -> uvm_perf_event_notify(... UVM_PERF_EVENT_FAULT)
           -> update_stats_fault_cb()                     [uvm_gpu.c]
              -> update_stats_parent_gpu_fault_instance() [uvm_gpu.c]
                 -> uvm_perf_fault_address_is_in_kv_cache()
                 -> total num_kv_faults / num_kv_duplicate_faults++

uvm_parent_gpu_service_replayable_faults()                [uvm_gpu_replayable_faults.c]
  before = total_kv_*
  ... run one batch ...
  batch_kv_* = total_kv_* - before
  -> log_replayable_fault_counters(... batch_kv_* ...)
```

---

## 8. 最容易误解的点

1. `batch_kv_faults` 不是“直接在 batch 中逐条自增”，而是 total 差分值。  
2. KV 判定只看 fault 地址是否落入 `[kv_start, kv_end]`。  
3. 若 fault 回调未注册（例如 debug/procfs 条件不满足），total 不更新，batch 差分自然也不会变。  


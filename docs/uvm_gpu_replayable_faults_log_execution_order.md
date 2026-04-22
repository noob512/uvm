# `uvm_gpu_replayable_faults.c` 日志打印执行顺序详解

本文只聚焦文件：

- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.c`

目标：说明“为了打印日志”，该文件内**哪些函数会执行、按什么顺序执行、每一步的触发条件是什么**。

---

## 1. 日志类型与入口概览

本文件里有两类日志：

1. `Replayable fault GPU ...`（每条 fault 地址日志）
2. `Replayable fault stats GPU ...`（每批次统计日志）

对应打印函数：

1. `log_replayable_fault_entry(...)`  
   定义：`uvm_gpu_replayable_faults.c:175`
2. `log_replayable_fault_counters(...)`  
   定义：`uvm_gpu_replayable_faults.c:220`

统一入口（ISR BH 主循环）：

- `uvm_parent_gpu_service_replayable_faults(...)`  
  定义：`uvm_gpu_replayable_faults.c:3240`

---

## 2. 先决条件（不满足就不会打印）

### 2.1 模块参数开关

1. `uvm_perf_fault_log_addresses`（每条 fault 地址日志开关）  
   定义：`uvm_gpu_replayable_faults.c:121`
2. `uvm_perf_fault_log_counters`（每批次统计日志开关）  
   定义：`uvm_gpu_replayable_faults.c:131`
3. `uvm_perf_fault_log_destination`（输出目的地）  
   定义：`uvm_gpu_replayable_faults.c:126`

目的地语义：

- `1` -> `trace_printk`（到 `trace_pipe`）
- `0` -> `UVM_INFO_PRINT`（到 `dmesg`）

### 2.2 是否有可处理 fault

`uvm_parent_gpu_service_replayable_faults()` 的 while 循环里，如果：

1. `fetch_fault_buffer_entries(...)` 返回错误，或
2. `batch_context->num_cached_faults == 0`

则当前轮不会产生日志。

---

## 3. 执行顺序 A：每条 fault 地址日志（`Replayable fault GPU ...`）

这条链路是“**在抓取硬件 fault buffer 条目时**”就打印。

## 3.1 严格顺序

1. `uvm_parent_gpu_service_replayable_faults(...)`  
   `uvm_gpu_replayable_faults.c:3240`
2. 调用 `fetch_fault_buffer_entries(parent_gpu, batch_context, FAULT_FETCH_MODE_BATCH_READY)`  
   调用点：`uvm_gpu_replayable_faults.c:3279`  
   函数定义：`uvm_gpu_replayable_faults.c:1033`
3. 在 `fetch_fault_buffer_entries(...)` 的条目解析循环中：
   1. `parse_replayable_entry(...)` 解析硬件条目
   2. 保存原始地址 `fault_address_raw`
   3. 对齐 `current_entry->fault_address`
4. 判断开关：`if (uvm_perf_fault_log_addresses)`  
   判断点：`uvm_gpu_replayable_faults.c:1126`
5. 调用 `log_replayable_fault_entry(parent_gpu, current_entry, fault_address_raw)`  
   调用点：`uvm_gpu_replayable_faults.c:1127`
6. `log_replayable_fault_entry(...)` 内部根据 `uvm_perf_fault_log_destination` 分支：
   1. `==1` -> `trace_printk(...)`
   2. `!=1` -> `UVM_INFO_PRINT(...)`  
   分支点：`uvm_gpu_replayable_faults.c:181`

## 3.2 关键特点

1. 这类日志发生在 `preprocess_fault_batch(...)` 和 `service_fault_batch(...)` 之前。
2. 它记录的是“抓到的硬件 fault 条目”，不是 batch 统计结果。

---

## 4. 执行顺序 B：每批次统计日志（`Replayable fault stats GPU ...`）

这条链路是“**batch 服务完成后**”打印。

## 4.1 严格顺序

1. 进入 `uvm_parent_gpu_service_replayable_faults(...)` while 一轮  
   `uvm_gpu_replayable_faults.c:3240`
2. 在轮开始记录基线：
   - `total_kv_faults_before`
   - `total_kv_duplicates_before`  
   位置：`uvm_gpu_replayable_faults.c:3259-3260`
3. 获取本轮条目：`fetch_fault_buffer_entries(...)`  
   调用点：`...:3279`
4. 预处理：`preprocess_fault_batch(...)`  
   调用点：`...:3290`
5. 服务 fault：`service_fault_batch(parent_gpu, FAULT_SERVICE_MODE_REGULAR, batch_context)`  
   调用点：`...:3302`  
   函数定义：`uvm_gpu_replayable_faults.c:2544`
6. 计算本轮 KV 增量（用 total 前后差）：
   - `batch_context->num_kv_faults`
   - `batch_context->num_kv_duplicate_faults`  
   位置：`...:3310-3315`
7. 判断开关：`if (uvm_perf_fault_log_counters)`  
   位置：`...:3318`
8. 调用 `log_replayable_fault_counters(...)`  
   调用点：`...:3319`
9. `log_replayable_fault_counters(...)` 内部按目的地分支：
   1. `==1` -> `trace_printk("Replayable fault stats GPU ...")`
   2. `!=1` -> `UVM_INFO_PRINT("Replayable fault stats GPU ...")`  
   分支点：`uvm_gpu_replayable_faults.c:240`

## 4.2 统计字段来源（本文件视角）

1. `batch_faults` <- `batch_context->num_cached_faults`
2. `batch_duplicates` <- `batch_context->num_duplicate_faults`
3. `batch_kv_faults` <- 本轮前后差分得到
4. `batch_kv_duplicates` <- 本轮前后差分得到
5. `total_*` <- `parent_gpu->stats` / `parent_gpu->fault_buffer.replayable.stats`

注意：`num_kv_faults`、`num_kv_duplicate_faults` 的累加在 `uvm_gpu.c` 的 fault 回调里完成，本文件只做“读取 total 并差分”。

---

## 5. `service_fault_batch(...)` 内与统计相关的实际执行函数

为了让 `batch_duplicates` 和外部 total 计数变化，本文件内关键执行链如下：

1. `service_fault_batch(...)`  
   `uvm_gpu_replayable_faults.c:2544`
2. `service_fault_batch_dispatch(...)`  
   `uvm_gpu_replayable_faults.c:2234`
3. 进入具体子路径（至少其一）：
   1. `service_fault_batch_block(...)` -> `service_fault_batch_block_locked(...)`  
      `...:1864`, `...:1622`
   2. `service_fault_batch_ats(...)` -> `service_fault_batch_ats_sub(...)`  
      `...:2177`, `...:2073`
   3. fatal 路径 `service_fault_batch_fatal_notify(...)`  
      `...:1963`
4. 在上述路径中调用 `update_batch_and_notify_fault(...)`：  
   定义：`...:1424`  
   典型调用点：
   - `...:1719`（block 路径）
   - `...:2112`（ATS 路径）
   - `...:1977`（fatal 通知路径）
5. `update_batch_and_notify_fault(...)` 做两件事：
   1. 更新 `batch_context->num_duplicate_faults`
   2. 调 `uvm_perf_event_notify_gpu_fault(...)` 触发 fault perf 事件

后续外部回调（不在本文件）会更新 `parent_gpu->...num_kv_faults` 等 total 计数，供本文件第 4 章差分使用。

---

## 6. 调用树（只保留日志关键节点）

```text
uvm_parent_gpu_service_replayable_faults (3240)
  -> fetch_fault_buffer_entries (1033)
       -> [if uvm_perf_fault_log_addresses] log_replayable_fault_entry (175)
            -> trace_printk / UVM_INFO_PRINT
  -> preprocess_fault_batch (1353)
  -> service_fault_batch (2544)
       -> service_fault_batch_dispatch (2234)
            -> ... -> update_batch_and_notify_fault (1424)
                 -> uvm_perf_event_notify_gpu_fault(...)
  -> compute batch_kv_* delta from total_kv_* before/after
  -> [if uvm_perf_fault_log_counters] log_replayable_fault_counters (220)
       -> trace_printk / UVM_INFO_PRINT
```

---

## 7. 实际排障时的顺序判断建议

如果你看不到 `Replayable fault stats GPU ...`，按这个顺序查：

1. `uvm_perf_fault_log_counters` 是否为 `1`
2. `uvm_parent_gpu_service_replayable_faults` 是否正在执行（是否有 replayable faults）
3. `batch_context->num_cached_faults` 是否一直为 `0`
4. 输出目的地是否匹配你的采集方式：
   - `destination=1` 看 `trace_pipe`
   - `destination=0` 看 `dmesg`

如果你看不到 `Replayable fault GPU ...`，按这个顺序查：

1. `uvm_perf_fault_log_addresses` 是否为 `1`
2. 是否真的进入了 `fetch_fault_buffer_entries` 的条目循环
3. 目的地配置是否正确


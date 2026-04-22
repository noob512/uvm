# `/tmp/uvm_kv_stats.log` 生成机制与全部相关改动函数说明

## 1. 结论先行

`/tmp/uvm_kv_stats.log` **不是内核直接写出的固定文件名**，而是：

1. 内核 UVM 驱动在 fault 服务路径里打印 `Replayable fault stats GPU ...` 统计行。
2. 用户态脚本把 `trace_pipe`（或 `dmesg -f`）中的这类行 `grep` 后重定向到目标文件（例如 `/tmp/uvm_kv_stats.log`）。

换句话说：
- **日志内容来源**：`nvidia-uvm` 内核模块统计打印。
- **日志文件落盘路径来源**：采集脚本参数（`--trace-log`）。

---

## 2. 端到端调用链（从 GPU fault 到 `/tmp/uvm_kv_stats.log`）

## 2.1 内核统计链

1. GPU 发生 replayable fault，进入 UVM fault 服务主循环：  
   `uvm_parent_gpu_service_replayable_faults()`  
   文件：`kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.c:3240`

2. fault 批次在服务过程中，fault 事件会通过：  
   `update_batch_and_notify_fault()` -> `uvm_perf_event_notify_gpu_fault(...)`  
   文件：`.../uvm_gpu_replayable_faults.c:1424`

3. VA space 上注册的 fault 回调 `update_stats_fault_cb()` 被触发，进一步调用：  
   `update_stats_parent_gpu_fault_instance()`  
   文件：`.../uvm_gpu.c:1895`, `.../uvm_gpu.c:1830`

4. 在 `update_stats_parent_gpu_fault_instance()` 中对每个 fault 做分类与累计：
   - 通过 `uvm_perf_fault_address_is_in_kv_cache(fault_address)` 判断是否 KV fault
   - 增加 `num_kv_faults` / `num_kv_duplicate_faults` / `num_replayable_faults` 等累计计数  
   文件：`.../uvm_gpu.c:1883` 起

5. 一个 batch 服务结束后，`uvm_parent_gpu_service_replayable_faults()` 用“本轮前后差分”得到：
   - `batch_kv_faults`
   - `batch_kv_duplicates`
   然后在 `uvm_perf_fault_log_counters=1` 时调用：  
   `log_replayable_fault_counters(...)`  
   文件：`.../uvm_gpu_replayable_faults.c:3310`, `...:3318`, `...:220`

6. `log_replayable_fault_counters(...)` 输出单行：
   `Replayable fault stats GPU ... batch_* ... total_* ...`
   - `uvm_perf_fault_log_destination=1` -> `trace_printk`（到 trace_pipe）
   - `uvm_perf_fault_log_destination=0` -> `UVM_INFO_PRINT`（到 dmesg）  
   文件：`.../uvm_gpu_replayable_faults.c:241`, `...:265`

## 2.2 文件采集链（生成 `/tmp/uvm_kv_stats.log`）

脚本 `workloads/vllm/run_kv_fault_ratio.sh` 中：

1. `--mode trace` 时：
   - `start_trace_capture_if_needed()` 启动：
     `cat trace_pipe | grep 'Replayable fault stats GPU' > "$TRACE_LOG"`  
   文件：`workloads/vllm/run_kv_fault_ratio.sh:231`

2. `--mode dmesg` 时：
   - `start_dmesg_capture_if_needed()` 启动：
     `dmesg --follow... | grep 'Replayable fault stats GPU' > "$TRACE_LOG"`  
   文件：`.../run_kv_fault_ratio.sh:248`

3. 因此当你运行：
   - `./workloads/vllm/run_kv_fault_ratio.sh --mode trace --trace-log /tmp/uvm_kv_stats.log`
   最终就得到 `/tmp/uvm_kv_stats.log`。

---

## 3. 为获得该日志而新增/修改的函数（完整清单）

下表只列“为了这条日志链路可用”而改动的关键函数与结构。

## 3.1 内核：`uvm_gpu_replayable_faults.c`

1. `uvm_perf_fault_address_is_in_kv_cache(NvU64 fault_address)`  
   位置：`.../uvm_gpu_replayable_faults.c:147`  
   作用：按模块参数 `kv_start/kv_end` 判断 fault 地址是否属于 KV 区间。

2. `percent_x100_u64(...)`  
   位置：`...:161`  
   作用：将比例转成保留两位小数的整数表示，供日志格式化。

3. `log_replayable_fault_counters(...)`  
   位置：`...:220`  
   作用：输出 `batch_faults/batch_kv_faults/total_kv_faults/...` 全字段统计行。

4. `uvm_parent_gpu_service_replayable_faults(...)`（改动点）  
   位置：`...:3240`  
   作用：
   - 记录 batch 前 `total_kv_faults_before/total_kv_duplicates_before`
   - batch 后做差得到 `batch_context->num_kv_faults/num_kv_duplicate_faults`
   - 在 `uvm_perf_fault_log_counters=1` 时打印统计日志

5. `log_replayable_fault_entry(...)`（地址日志相关，同一日志框架）  
   位置：`...:175`  
   作用：可选输出每条 fault 地址明细（不是 `uvm_kv_stats.log` 必需，但同一采集体系）。

### 同文件新增模块参数（决定日志能否出现）

- `uvm_perf_fault_log_destination`：`...:126`
- `uvm_perf_fault_log_counters`：`...:131`
- `uvm_perf_fault_kv_range_enable`：`...:136`
- `uvm_perf_fault_kv_start`：`...:141`
- `uvm_perf_fault_kv_end`：`...:144`

## 3.2 内核头文件：`uvm_gpu_replayable_faults.h`

1. `uvm_perf_fault_address_is_in_kv_cache(...)` 声明  
   位置：`kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.h:58`  
   作用：让其他编译单元（`uvm_gpu.c`）可调用 KV 判定函数。

## 3.3 内核：`uvm_gpu.c`

1. `update_stats_parent_gpu_fault_instance(...)`（核心计数逻辑）  
   位置：`.../uvm_gpu.c:1830`  
   改动作用：
   - 调用 `uvm_perf_fault_address_is_in_kv_cache(...)`
   - 增加 `num_kv_faults` 和 `num_kv_duplicate_faults`

2. `update_stats_fault_cb(...)`（fault 回调分发）  
   位置：`...:1895`  
   作用：将代表 fault 和 merged fault 实例都送入上面的计数函数。

3. `uvm_gpu_init_va_space(...)`（回调注册）  
   位置：`...:2025`  
   作用：注册 `UVM_PERF_EVENT_FAULT -> update_stats_fault_cb`，使统计路径生效。

4. `gpu_fault_stats_print_common(...)`（procfs 辅助观测）  
   位置：`...:692` 附近（引用处 `:699`, `:700`）  
   作用：输出 `kv_faults/kv_duplicates/kv_ratio`，便于和日志交叉校验。

## 3.4 内核结构体：`uvm_gpu.h`

1. `uvm_fault_service_batch_context_struct` 增加：
   - `num_kv_faults`
   - `num_kv_duplicate_faults`  
   位置：`.../uvm_gpu.h:323`, `:325`

2. replayable `stats` 增加：
   - `num_kv_faults`
   - `num_kv_duplicate_faults`  
   位置：`.../uvm_gpu.h:402`, `:404`

作用：分别承载 batch 级与 total 级 KV 计数。

## 3.5 vLLM：`gpu_model_runner.py`（给内核 KV 判定提供地址区间）

> 这部分不直接写 `/tmp/uvm_kv_stats.log`，但它提供 `kv_start/kv_end` 数据来源，是 KV 统计成立的前提。

1. `_is_uvm_address_logging_enabled()`  
   位置：`.../gpu_model_runner.py:3379`  
   作用：决定是否开启地址日志（默认跟随 `VLLM_USE_UVM`）。

2. `_uvm_address_log_file()`  
   位置：`...:3386`  
   作用：地址日志路径（默认 `vllm_uvm_address_regions.log`）。

3. `_append_uvm_address_log(...)`  
   位置：`...:3391`  
   作用：写出地址日志文件 section。

4. `_collect_weight_address_rows()`  
   位置：`...:3448`  
   作用：采集 weight/buffer 地址区间。

5. `_collect_kv_cache_address_rows(...)`  
   位置：`...:3510`  
   作用：采集 KV 地址，并汇总 `kv_cache:contiguous_range,all_layers,start,end,...`。

6. `_log_model_weight_addresses(...)`  
   位置：`...:3628`

7. `_log_kv_cache_addresses(...)`  
   位置：`...:3644`

8. `load_model(...)`（调用 `_log_model_weight_addresses("load_model")`）  
   位置：`...:3661`

9. `reload_weights(...)`（调用 `_log_model_weight_addresses("reload_weights")`）  
   位置：`...:3884`

10. `initialize_kv_cache_tensors(...)`（调用 `_log_kv_cache_addresses(..., "initialize_kv_cache")`）  
    位置：`...:5400`

## 3.6 采集脚本：`workloads/vllm/run_kv_fault_ratio.sh`

1. `extract_kv_range_from_log()`  
   位置：`.../run_kv_fault_ratio.sh:121`  
   作用：从 vLLM 地址日志抽取 `kv_cache:contiguous_range|span_range` 的 `start/end`。

2. `configure_uvm_params()`  
   位置：`...:355`  
   作用：写入：
   - `uvm_perf_fault_kv_start`
   - `uvm_perf_fault_kv_end`
   - `uvm_perf_fault_kv_range_enable=1`
   - `uvm_perf_fault_log_counters=1`
   - `uvm_perf_fault_log_destination`（trace/dmesg）

3. `start_trace_capture_if_needed()`  
   位置：`...:231`  
   作用：把 trace_pipe 中统计行保存到 `TRACE_LOG`（可设为 `/tmp/uvm_kv_stats.log`）。

4. `start_dmesg_capture_if_needed()`  
   位置：`...:248`  
   作用：dmesg 跟随采集并保存到 `TRACE_LOG`。

5. `reload_uvm_module_for_counter_reset()`  
   位置：`...:266`  
   作用：可选重载 `nvidia_uvm`，把 total 计数清零，便于单次实验对比。

6. `main()`  
   位置：`...:617`  
   作用：串联“启动 vLLM -> 等待 KV 区间 -> 配置参数 -> 采集日志 -> 跑 benchmark”全流程。

---

## 4. `/tmp/uvm_kv_stats.log` 的最小复现方式

```bash
cd /home/ubuntu/nvidia-uvm-gpu
./workloads/vllm/run_kv_fault_ratio.sh \
  --mode trace \
  --trace-log /tmp/uvm_kv_stats.log
```

若用 dmesg：

```bash
./workloads/vllm/run_kv_fault_ratio.sh \
  --mode dmesg \
  --trace-log /tmp/uvm_kv_stats.log
```

---

## 5. 常见误区

1. `batch_kv_faults` 不是累计值，而是“本 batch 增量”；`total_kv_faults` 才是累计值。  
2. `uvm_perf_fault_kv_range_enable=0` 时，KV 分类关闭，`batch_kv_faults` 会一直是 0。  
3. `kv_start/kv_end` 配错（或跨模型污染）会导致 KV 比例失真。  
4. `/tmp/uvm_kv_stats.log` 名字是采集脚本决定的，不是内核硬编码。


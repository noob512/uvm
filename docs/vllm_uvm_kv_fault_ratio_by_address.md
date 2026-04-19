# vLLM + UVM：基于缺页地址识别 KV Cache 缺失并统计占比（实现文档）

## 1. 目标

在 GPU 侧发生 replayable page fault 时：

1. 用 fault 地址判断是否命中 vLLM KV cache 地址区间。
2. 统计 KV fault 次数与占比。
3. 同时输出原始口径与去重后口径（dedup）统计，便于分析实际缺页压力。

---

## 2. 设计思路

前提：vLLM 已能输出 KV cache 地址日志，且本场景下 KV 地址空间是连续区间。

因此采用最直接判定：

- 设定 KV 区间 `[kv_start, kv_end]`（闭区间）。
- 对每个 replayable fault，检查 `fault_address` 是否落入该区间。
- 命中则记为 `kv_fault`，否则记为 `non_kv_fault`。

这样可以避免在 fault 热路径做复杂结构匹配，只做一次常量时间范围比较。

---

## 3. 修改文件

1. `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.h`
2. `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_gpu.h`
3. `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_gpu.c`
4. `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.c`
5. `workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py`

---

## 4. 详细实现

## 4.1 vLLM 侧：输出 KV 连续区间摘要

在 `GPUModelRunner._collect_kv_cache_address_rows()` 中新增了 KV 区间摘要行：

- `kv_cache:contiguous_range,all_layers,start,end,size...`：检测为连续区间。
- `kv_cache:span_range,all_layers,start,end,size...`：不连续时输出整体跨度。

这样可以直接从日志拿到用于内核参数配置的 `start/end`，不需要再手工汇总各层地址。

## 4.2 UVM 侧：新增 KV 区间参数与地址判定函数

在 `uvm_gpu_replayable_faults.c` 新增模块参数：

1. `uvm_perf_fault_kv_range_enable`：是否开启 KV 地址区间判定。
2. `uvm_perf_fault_kv_start`：KV 起始地址（含）。
3. `uvm_perf_fault_kv_end`：KV 结束地址（含）。

并新增导出函数：

- `bool uvm_perf_fault_address_is_in_kv_cache(NvU64 fault_address)`

供统计路径复用。

## 4.3 UVM 统计结构扩展

在 `uvm_gpu.h` 新增 replayable fault 统计字段：

1. `num_kv_faults`
2. `num_kv_duplicate_faults`

并在 batch context 中新增：

1. `num_kv_faults`
2. `num_kv_duplicate_faults`

用于批次日志输出。

## 4.4 按 fault 地址做 KV 分类计数

在 `uvm_gpu.c:update_stats_parent_gpu_fault_instance()` 中：

- 对 replayable fault 调用 `uvm_perf_fault_address_is_in_kv_cache(fault_entry->fault_address)`。
- 命中时累加 `num_kv_faults`。
- 若该 fault 同时被判定为 duplicate，则累加 `num_kv_duplicate_faults`。

这保证 KV 计数与现有 replayable/duplicate 统计口径一致。

## 4.5 批次日志增加 KV 统计与占比

在 `uvm_gpu_replayable_faults.c:log_replayable_fault_counters()` 中扩展输出：

1. `batch_kv_faults`
2. `batch_kv_duplicates`
3. `batch_kv_after_dedup`
4. `batch_kv_ratio`
5. `batch_kv_after_dedup_ratio`
6. `total_kv_faults`
7. `total_kv_duplicates`
8. `total_kv_after_dedup`
9. `total_kv_ratio`
10. `total_kv_after_dedup_ratio`

占比公式：

- `kv_ratio = kv_faults / replayable_faults`
- `kv_after_dedup_ratio = kv_after_dedup / replayable_after_dedup`

其中：

- `replayable_after_dedup = replayable_faults - duplicates`
- `kv_after_dedup = kv_faults - kv_duplicates`

## 4.6 procfs 统计增加 KV 维度

在 `uvm_gpu.c:gpu_fault_stats_print_common()` 中追加：

1. `kv_faults`
2. `kv_duplicates`
3. `kv_after_dedup`
4. `kv_ratio`
5. `kv_after_dedup_ratio`

可直接从 fault stats 文件查看累计占比。

---

## 5. 使用方式

## 5.1 从 vLLM 日志提取 KV 区间

确保启用地址日志：

```bash
export VLLM_USE_UVM=1
export VLLM_UVM_ADDRESS_LOG_ENABLE=1
export VLLM_UVM_ADDRESS_LOG_FILE=/tmp/vllm_uvm_address_regions.log
```

运行 vLLM 后，提取摘要行：

```bash
grep "kv_cache:contiguous_range,all_layers" /tmp/vllm_uvm_address_regions.log | tail -n 1
```

示例（伪）：

```text
kv_cache:contiguous_range,all_layers,0x700000000000,0x7007ffffffff,34359738368,32768.000
```

## 5.2 配置 UVM KV 区间判定

将上一步 `start/end` 写入模块参数：

```bash
# 开启 KV 区间判定
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_kv_range_enable

# 按实际日志值替换（支持 0x 前缀）
echo 0x700000000000 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_kv_start
echo 0x7007ffffffff | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_kv_end
```

## 5.3 开启 fault 统计日志

```bash
# 0=dmesg, 1=trace_pipe
echo 0 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_log_destination

# 每批次统计
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_log_counters
```

## 5.4 查看结果

```bash
dmesg -T | grep "Replayable fault stats GPU"
```

你会看到包含 KV 字段的新日志行。

---

## 6. 日志字段解释

单批次关键字段：

1. `batch_faults`：本批 replayable fault 总数。
2. `batch_duplicates`：本批 duplicate fault 数。
3. `batch_kv_faults`：本批命中 KV 地址区间的 fault 数。
4. `batch_kv_duplicates`：本批 KV fault 中的 duplicate 数。
5. `batch_kv_ratio`：`batch_kv_faults / batch_faults`。

累计关键字段：

1. `total_faults`：累计 replayable fault 总数。
2. `total_duplicates`：累计 duplicate fault 总数。
3. `total_kv_faults`：累计 KV fault 总数。
4. `total_kv_duplicates`：累计 KV duplicate fault 总数。
5. `total_kv_ratio`：`total_kv_faults / total_faults`。

---

## 7. 验证清单

1. vLLM 地址日志中存在 `kv_cache:contiguous_range` 行。
2. `uvm_perf_fault_kv_range_enable=1` 且 `start/end` 已设置。
3. `Replayable fault stats GPU ...` 日志中出现 `batch_kv_faults/total_kv_faults` 字段。
4. 在 decode 场景中，`total_kv_ratio` 应显著高于纯权重加载阶段。

---

## 8. 注意事项

1. `start/end` 为闭区间，建议直接使用日志中的十六进制值。
2. 若 `kv_end < kv_start` 或参数为 0，分类自动视为无效（不计 KV 命中）。
3. 多进程/多模型并发时，需确保区间对应当前 workload；否则比例会被污染。
4. 高 fault 压力下日志量较大，建议仅在诊断窗口开启 `uvm_perf_fault_log_counters`。

---

## 9. 回退

关闭 KV 判定与统计输出：

```bash
echo 0 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_kv_range_enable
echo 0 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_log_counters
```

该实现仅增加统计与日志，不改变原有 fault service 决策路径。

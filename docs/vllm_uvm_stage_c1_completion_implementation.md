# vLLM UVM Stage C1 Completion Implementation

本文档记录 Stage C1 补全实现：在 C1 attention-only `device_direct` 总预算已经通过多轮实验验证后，补齐 driver stats delta 解析和 budget sweep 自动化。

## 1. 背景

此前 C1 已经具备核心能力：

1. `gap_hot_runtime_scratch` strict gating。
2. `enabled:attention` phase 限定。
3. `cuda_malloc` device-direct backend。
4. `VLLM_UVM_DEVICE_DIRECT_MAX_TOTAL_BYTES` 总预算。
5. CAS reservation，保证并发下 live bytes 不越过预算。
6. `device_direct_budget_reject_records`、`device_direct_peak_live_bytes_observed` 等 metrics。

最近实验结论：

1. p10 / 1 MiB budget 下，C1 `cuda_malloc` 能稳定降低 gap2 faults。
2. 同配置 C1 sync vs C2 async 对照中，C1 `cuda_malloc` 当前更优。
3. C2 `cuda_malloc_async` 正确性已通过，但暂不应替代 C1 默认 backend。

因此当前最合适的工作不是扩大策略范围，而是补齐 C1 实验闭环。

## 2. 本次补全目标

本次补全包含两个部分：

1. 修复 `run_kv_fault_ratio.sh` 对中文 UVM stats line 的 delta parser。
2. 新增 C1 budget sweep 脚本和汇总器。

不包含：

1. 扩展到 `enabled:moe`。
2. 启用 `cuda_malloc_async` 为默认 backend。
3. 配置 CUDA memory pool release threshold。
4. 改变 C1 placement 策略。

## 3. Driver Stats Delta Parser 补全

### 3.1 问题

实验日志中经常出现：

```text
Failed to parse stats lines from /tmp/...
Hint: keep machine-readable key=value fields in driver stats logs
```

原因不是 stats 没采到，而是当前 driver stats 行是中文自然语言格式，例如：

```text
本批次总缺页实例数=256,去重后=28,KV类的总缺页数=0,去重后=0,...
|| 总缺页数=67123777,去重后=16158942,kv总错误数=198437,去重后=35240,...
```

旧 parser 的问题：

1. 已经能解析 `batch_faults`、`batch_after_dedup`、`total_faults`、`total_after_dedup`。
2. 但漏掉了 `batch_kv_after_dedup`。
3. 还把 `KV类的总缺页数=...,去重后=...` 误当成 `batch_kv_duplicates`。
4. localized line 实际暴露的是 after-dedup，不是 duplicates。

### 3.2 修改

修改文件：

```text
/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/run_kv_fault_ratio.sh
```

新增/修正逻辑：

1. `batch_kv_after_dedup` 从 `KV类的总缺页数=...,去重后=...` 解析。
2. `batch_kv_duplicates` 在中文日志路径中留空，由后续公式推导。
3. `total_kv_duplicates` 在中文日志路径中留空，由后续公式推导。
4. parser 先检查 fault / after-dedup 必需字段是否存在，再推导 duplicates。

推导公式：

```text
duplicates = faults - after_dedup
```

workload delta 公式保持不变：

```text
baseline = first_total - first_batch
delta = last_total - baseline
```

### 3.3 预期输出

修复后，中文 stats line 应能输出：

```text
===== Delta Replayable fault stats (this workload) =====
formula: baseline = first_total - first_batch; delta = last_total - baseline
delta_faults=...
delta_kv_faults=...
delta_kv_ratio=...
```

这不会替代 gap-watch JSON，但能补齐 driver-level replayable fault delta。

## 4. C1 Budget Sweep 自动化

### 4.1 新增脚本

新增：

```text
/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/run_stage_c_attention_c1_budget_sweep.sh
/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/compare_stage_c_budget_sweep.py
```

### 4.2 运行方式

默认运行：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm

./run_stage_c_attention_c1_budget_sweep.sh
```

默认参数：

```text
PROMPTS=10
REQUEST_RATE=5
OUTPUT_LEN=512
BUDGETS_CSV=524288,1048576,2097152,4194304
DEVICE_DIRECT_BACKEND=cuda_malloc
```

推荐显式运行：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm

PROMPTS=10 \
BUDGETS_CSV=524288,1048576,2097152,4194304 \
./run_stage_c_attention_c1_budget_sweep.sh
```

如果需要更快的 smoke test：

```bash
PROMPTS=5 \
BUDGETS_CSV=524288,1048576 \
./run_stage_c_attention_c1_budget_sweep.sh
```

### 4.3 输出结构

输出根目录：

```text
/tmp/vllm_stage_c1_budget_sweep_<timestamp>/
```

每个 budget 一个子目录：

```text
budget_524288/
budget_1048576/
budget_2097152/
budget_4194304/
```

每个子目录内部是完整 `run_stage_c_attention_p20_ab.sh` 输出，包括：

```text
vllm_stage_c_attention_p<PROMPTS>_ab_comparison.json
vllm_gap_watch_metrics_gap2_stage_c_attention_p<PROMPTS>.json
vllm_bench_gap2_stage_c_attention_p<PROMPTS>.log
vllm_uvm_allocator_trace_gap2_stage_c_attention_p<PROMPTS>.log
```

总汇总：

```text
vllm_stage_c1_budget_sweep_p<PROMPTS>.json
```

### 4.4 汇总指标

`compare_stage_c_budget_sweep.py` 会抽取：

1. `success_signal`
2. `effectiveness_signal`
3. `gap_fault_delta_pct_vs_trace`
4. `unknown_fault_delta_pct_vs_trace`
5. `output_throughput_delta_pct_vs_trace`
6. `mean_tpot_delta_pct_vs_trace`
7. `device_direct_actual_records`
8. `device_direct_budget_reject_records`
9. `device_direct_peak_live_bytes_observed`
10. `device_direct_min_budget_remaining_observed`
11. `device_direct_backend_counts`
12. `placement_backend_counts`

并给出一个 `best_budget_bytes`。

当前选择规则：

```text
Prefer success/effectiveness,
then higher output throughput,
then larger gap fault reduction,
then lower TPOT.
```

这个规则是实验辅助，不是最终论文结论。最终仍应人工检查 raw logs。

## 5. 当前推荐策略

基于已有实验：

```text
backend: cuda_malloc
phase: enabled:attention
target_class: gap_hot_runtime_scratch
min_bytes: 4096
max_bytes: 1048576
budget: 从 1 MiB 起做 sweep
```

当前不推荐：

```text
backend: cuda_malloc_async 作为默认
target_phases: enabled:moe
unlimited device_direct budget
```

## 6. 下一步

建议顺序：

1. 用新 parser 跑一次 p10 C1，确认 `Delta Replayable fault stats` 不再失败。
2. 跑 C1 budget sweep，找 512 KiB / 1 MiB / 2 MiB / 4 MiB 的拐点。
3. 如果 1 MiB 仍是最优或接近最优，固定为 C1 推荐 budget。
4. 如果更大 budget 带来明显 fault 下降且 TPOT 不恶化，再考虑 p20 验证。
5. 只有当 C1 budget 曲线稳定后，再重新评估 C2 memory pool threshold。

## 7. 验证命令

语法检查：

```bash
bash -n workloads/vllm/run_kv_fault_ratio.sh
bash -n workloads/vllm/run_stage_c_attention_c1_budget_sweep.sh
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile workloads/vllm/compare_stage_c_budget_sweep.py
```

离线汇总示例：

```bash
python3 workloads/vllm/compare_stage_c_budget_sweep.py \
  --run 1048576=/tmp/vllm_stage_c_attention_backend_ab_20260428_063138/cuda_malloc/vllm_stage_c_attention_p10_ab_comparison.json \
  --output-json /tmp/c1_budget_sweep_smoke.json
```

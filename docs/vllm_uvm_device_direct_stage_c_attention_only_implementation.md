# vLLM UVM Device-Direct Stage C Attention-Only Implementation

本文档记录本次 Stage C 小范围真实 `device_direct` 实现方案、代码修改、运行命令、输出字段和回退策略。

## 1. 背景

前置 Stage B 实验已经确认 gap2 具备进入小范围 Stage C 的条件：

1. gap2 在 probe 和 main 阶段稳定复现。
2. gap2 是 unknown fault 的主热点。
3. gap2 fault 以 write 为主，实验中为 100% write。
4. gap2 的 allocator 主导 phase 是 `enabled:attention`，其次是 `enabled:moe` 和 `enabled:model_forward`。
5. 严格 Stage B 下，`gap_hot_runtime_scratch` 候选对象数量充足。
6. 分配生命周期很短，main 阶段中位生命周期约 1ms 到 2ms，p95 约数毫秒。

因此 Stage C 不再只做 trace，而是在严格条件下允许 allocator 将部分 managed 分配替换为 GPU-only 分配。

## 2. Stage C 的安全边界

本次实现不是对整个 gap2 地址段一刀切，而是只允许满足所有条件的 allocation 进入真实 `device_direct`：

1. gap-watch policy action 必须是 `device_direct`。
2. `--uvm-device-direct-enable` 必须为 `1`。
3. target class 必须匹配 `gap_hot_runtime_scratch`。
4. allocation 必须命中当前 gap-watch 热点区间。
5. allocation phase 必须命中 allowlist。
6. 初始建议只使用 `enabled:attention`。
7. allocation size 必须在安全范围内，默认 `4096 <= size <= 1048576`。
8. 不满足条件时继续走 `cudaMallocManaged`。
9. `cudaMalloc` 失败时自动保留 managed 分配并记录 fallback reason。

默认 kill switch 仍然关闭：

```bash
--uvm-device-direct-enable 0
```

只有显式设置为 `1` 时，才可能触发真实 GPU-only allocation。

## 3. 代码修改

### 3.1 allocator 真实后端切换

文件：

`/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/uvm_allocator.cpp`

核心行为：

1. 仍先调用 `cudaMallocManaged` 获取 managed candidate。
2. 使用 candidate 地址计算是否与 gap-watch 区间重叠。
3. 执行 target class、phase、size、policy action、kill switch 判定。
4. 如果满足 Stage C 条件，则调用 `cudaMalloc` 申请 GPU-only device memory。
5. `cudaMalloc` 成功后释放 managed candidate，并将返回给 vLLM/PyTorch 的指针替换为 device pointer。
6. 如果 `cudaMalloc` 失败，或者释放 managed candidate 失败，则回退 managed candidate。

这样做的原因是：是否命中 gap2 依赖分配后的虚拟地址，不能在分配前可靠判断。因此 Stage C 使用“managed candidate -> 判定 -> 可选替换为 device-only”的保守方式。

### 3.2 新增统计指标

allocator session summary 现在会输出：

```text
Device-direct actual allocations
Device-direct actual bytes
Device-direct fallback allocations
Device-direct free success
```

含义：

1. `Device-direct actual allocations`：真实走 `cudaMalloc` 的次数。
2. `Device-direct actual bytes`：真实走 GPU-only backend 的累计字节数。
3. `Device-direct fallback allocations`：尝试 device backend 失败并回退 managed 的次数。
4. `Device-direct free success`：真实 device allocation 在 free 阶段成功释放的次数。

### 3.3 日志字段

`TRACE_POLICY`、`TRACE_GAP_WATCH_ALLOC`、`TRACE_GAP_WATCH_FREE` 继续输出：

```text
placement_backend=managed|device_direct
device_direct_eligible=0|1
device_direct_reason=<reason>
cpu_access_risk=<risk>
hot_gap_match=0|1
```

Stage C 成功时应看到：

```text
action=device_direct
placement_backend=device_direct
device_direct_eligible=1
device_direct_reason=device_direct_enabled
```

如果仍是 Stage B trace-only，应看到：

```text
action=device_direct_trace
placement_backend=managed
device_direct_eligible=1
device_direct_reason=trace_action_only
```

### 3.4 脚本参数

文件：

`/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/run_kv_fault_ratio.sh`

关键参数：

```bash
--uvm-device-direct-enable <0|1>
--uvm-device-direct-min-bytes <n>
--uvm-device-direct-max-bytes <n>
--uvm-device-direct-target-phases <csv>
--auto-gap-watch-policy-action-override device_direct
--auto-gap-watch-target-class-override gap_hot_runtime_scratch
```

默认 phase allowlist 是：

```text
enabled:attention,enabled:moe,enabled:model_forward
```

Stage C 初始实验建议显式收窄为：

```text
enabled:attention
```

### 3.5 分析脚本更新

文件：

`/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/summarize_gap_watch_metrics.py`

新增/修正：

1. 修复 `hot_gap_match_records` 对行尾字段解析不正确的问题。
2. 输出 `phase_record_ratios`。
3. 输出 `device_direct_actual_records`，来自 `placement_backend=device_direct` 的记录数。

文件：

`/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/discover_gap_watch.py`

修改：

1. 对 allocator correlation 优先使用 `TRACE_POLICY` 中的 `gap_overlap_bytes`。
2. 这样真实 `device_direct` 后，即使返回的 device pointer 不再落在原 managed gap2 地址，也仍能基于 allocator 的 gap-watch 记录做 post-main 分类。

## 4. 推荐实验流程

### 4.1 Stage C0-A：attention-only 小流量

先用少量 prompt 验证功能正确性：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm

./run_kv_fault_ratio.sh \
  --mode trace \
  --with-address-log \
  --trace-log /tmp/uvm_kv_fault_stats_gap2_stage_c_attention_p5.log \
  --address-trace-log /tmp/uvm_kv_fault_addrs_gap2_stage_c_attention_p5.log \
  --allocator-log /tmp/vllm_uvm_allocator_trace_gap2_stage_c_attention_p5.log \
  --uvm-trace-min-bytes 1048576 \
  --uvm-unknown-detail-enable 1 \
  --uvm-unknown-detail-min-bytes 4096 \
  --uvm-gap-watch-name same_run_gap2_stage_c_attention_p5 \
  --uvm-gap-watch-all-classes 1 \
  --uvm-gap-watch-min-bytes 4096 \
  --uvm-device-direct-enable 1 \
  --uvm-device-direct-min-bytes 4096 \
  --uvm-device-direct-max-bytes 1048576 \
  --uvm-device-direct-target-phases enabled:attention \
  --auto-gap-watch-enable 1 \
  --auto-gap-watch-probe-prompts 1 \
  --auto-gap-watch-target-gap 2 \
  --auto-gap-watch-policy-action-override device_direct \
  --auto-gap-watch-target-class-override gap_hot_runtime_scratch \
  --prompts 5 \
  --gap-watch-metrics-summary-json /tmp/vllm_gap_watch_metrics_gap2_stage_c_attention_p5.json \
  --auto-gap-watch-summary-json /tmp/vllm_auto_gap_watch_summary_gap2_stage_c_attention_p5.json \
  --auto-gap-watch-post-main-summary-json /tmp/vllm_auto_gap_watch_post_main_summary_gap2_stage_c_attention_p5.json
```

通过标准：

1. benchmark 完整结束。
2. `Failed requests = 0`。
3. allocator summary 中出现 `placement_backend=device_direct`。
4. `device_direct_actual_records > 0`。
5. `Device-direct fallback allocations = 0` 或很低。
6. 没有 CUDA illegal address、invalid device pointer、CPU 访问 device pointer 等错误。

### 4.2 Stage C0-B：attention-only 主实验

小流量稳定后，再跑 20 prompts：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm

./run_kv_fault_ratio.sh \
  --mode trace \
  --with-address-log \
  --trace-log /tmp/uvm_kv_fault_stats_gap2_stage_c_attention.log \
  --address-trace-log /tmp/uvm_kv_fault_addrs_gap2_stage_c_attention.log \
  --allocator-log /tmp/vllm_uvm_allocator_trace_gap2_stage_c_attention.log \
  --uvm-trace-min-bytes 1048576 \
  --uvm-unknown-detail-enable 1 \
  --uvm-unknown-detail-min-bytes 4096 \
  --uvm-gap-watch-name same_run_gap2_stage_c_attention \
  --uvm-gap-watch-all-classes 1 \
  --uvm-gap-watch-min-bytes 4096 \
  --uvm-device-direct-enable 1 \
  --uvm-device-direct-min-bytes 4096 \
  --uvm-device-direct-max-bytes 1048576 \
  --uvm-device-direct-target-phases enabled:attention \
  --auto-gap-watch-enable 1 \
  --auto-gap-watch-probe-prompts 1 \
  --auto-gap-watch-target-gap 2 \
  --auto-gap-watch-policy-action-override device_direct \
  --auto-gap-watch-target-class-override gap_hot_runtime_scratch \
  --prompts 20 \
  --gap-watch-metrics-summary-json /tmp/vllm_gap_watch_metrics_gap2_stage_c_attention.json \
  --auto-gap-watch-summary-json /tmp/vllm_auto_gap_watch_summary_gap2_stage_c_attention.json \
  --auto-gap-watch-post-main-summary-json /tmp/vllm_auto_gap_watch_post_main_summary_gap2_stage_c_attention.json
```

## 5. 重点观察字段

### 5.1 自动 gap 发现

文件：

`/tmp/vllm_auto_gap_watch_summary_gap2_stage_c_attention.json`

重点：

```text
fallback_used=false
selected_gap.gap_index=2
effective_target_class=gap_hot_runtime_scratch
effective_policy_action=device_direct
```

### 5.2 post-main gap 一致性

文件：

`/tmp/vllm_auto_gap_watch_post_main_summary_gap2_stage_c_attention.json`

重点：

```text
same_gap=true
same_gap_index=true
same_gap_start=true
same_gap_end=true
main_fallback_used=false
```

如果真实 `device_direct` 有效，gap2 fault 可能明显下降；如果下降到 target gap 不再是最热点，也需要结合 `selected_gap` 和 allocator metrics 一起判断。

### 5.3 gap-watch metrics

文件：

`/tmp/vllm_gap_watch_metrics_gap2_stage_c_attention.json`

重点：

```text
dominant_action=device_direct
dominant_target_class=gap_hot_runtime_scratch
placement_backend_counts
device_direct_actual_records
device_direct_eligible_records
device_direct_reason_counts
gap_policy_fail
median_lifetime_s
```

成功信号：

```text
placement_backend_counts={'managed': ..., 'device_direct': ...}
device_direct_actual_records > 0
device_direct_reason_counts 包含 device_direct_enabled
gap_policy_fail=0
```

### 5.4 fault 变化

重点比较三组：

1. `device_direct_trace` 严格 Stage B。
2. `device_direct` Stage C attention-only。
3. `prefetch`。

核心指标：

```text
gap2 faults
unknown faults
avg_faults_per_unique_page
TTFT
TPOT
output throughput
failed requests
```

Stage C 有效的最低标准：

1. `device_direct_actual_records > 0`。
2. `gap2 faults` 低于 Stage B trace-only。
3. `unknown faults` 低于 Stage B trace-only。
4. benchmark 无失败请求。
5. 性能没有显著恶化。

## 6. 回退与风险处理

如果出现 CUDA error、非法地址、进程崩溃或 failed requests：

1. 立即回退到：

```bash
--uvm-device-direct-enable 0
```

2. 保留：

```bash
--auto-gap-watch-policy-action-override device_direct_trace
```

3. 检查 allocator log 中的：

```text
placement_backend
device_direct_reason
device_direct_fallback_allocations
gap_policy_fail
```

4. 如果 attention-only 失败，不要扩展到 `enabled:moe`。

## 7. 后续扩展条件

只有当 attention-only 同时满足以下条件，才建议扩展到 MoE：

1. `Failed requests = 0`。
2. `placement_backend=device_direct` 出现且释放成功。
3. `device_direct_fallback_allocations` 很低。
4. gap2 fault 或 unknown fault 有下降。
5. TTFT/TPOT/throughput 没有明显恶化。

扩展命令只需把 phase allowlist 改成：

```bash
--uvm-device-direct-target-phases enabled:attention,enabled:moe
```

暂时不建议加入 `enabled:model_forward`，因为它是外层 phase，可能包含更多不同语义的临时分配。

## 8. 当前实现边界

1. Stage C 使用 `cudaMalloc`，不是 `cudaMallocAsync`。
2. 因为必须先知道 candidate managed 地址是否命中 gap2，所以当前实现采用 managed candidate 后再替换 backend 的方式。
3. 真实返回给 vLLM 的指针是 device pointer；日志中的 gap overlap 字段表示原 managed candidate 对 gap2 的命中证据。
4. 如果后续确认 `cudaMalloc/cudaFree` 同步开销明显，应评估 `cudaMallocAsync/cudaFreeAsync` 和 stream-ordered memory pool。
5. 当前不对 CPU access 做运行时硬检测，只依赖 phase、size、write-only fault 证据和 kill switch。

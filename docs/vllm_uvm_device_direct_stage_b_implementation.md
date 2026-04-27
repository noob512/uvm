# vLLM UVM Device-Direct Stage B Implementation

## 1. 阶段目标

阶段 B 的目标是新增 `device_direct` 动作表达能力，但默认保持 trace-only。

也就是说：

1. allocator 可以识别 `device_direct_trace` 和 `device_direct`。
2. control file 可以下发这两个动作。
3. 日志可以记录哪些 allocation 会成为 device-direct 候选。
4. 实际分配行为仍然保持 `cudaMallocManaged`。
5. 不引入 `cudaMalloc` / `cudaMallocAsync`。
6. 不改变 `uvm_free` 的释放路径。

这一步的核心价值是先回答：

1. 如果后续启用 GPU-only 分配，会选中多少对象？
2. 这些对象是否真的集中在 gap2？
3. 它们的 size、phase、class、lifetime 是否符合低风险条件？

---

## 2. 修改文件

本阶段修改了以下文件：

1. [uvm_allocator.cpp](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/uvm_allocator.cpp)
2. [run_kv_fault_ratio.sh](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/run_kv_fault_ratio.sh)
3. [discover_gap_watch.py](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/discover_gap_watch.py)
4. [summarize_gap_watch_metrics.py](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/summarize_gap_watch_metrics.py)

编译产物也已同步到 vLLM 实际加载路径：

1. [uvm_allocator.so](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/uvm_allocator.so)
2. [uvm_allocator.abi3.so](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/uvm_allocator.abi3.so)

---

## 3. `uvm_allocator.cpp` 修改

### 3.1 新增 `PolicyAction`

新增两个动作：

```cpp
DeviceDirectTrace
DeviceDirect
```

字符串映射：

```text
device_direct_trace -> DeviceDirectTrace
device_direct -> DeviceDirect
```

输出字符串：

```text
device_direct_trace
device_direct
```

### 3.2 行为边界

阶段 B 中，即使策略动作是：

```text
device_direct_trace
device_direct
```

实际分配仍然走：

```cpp
cudaMallocManaged(&ptr, size, cudaMemAttachGlobal)
```

因此阶段 B 不会改变：

1. 分配 API
2. pointer backend
3. free API
4. PyTorch allocator 契约

日志中会明确写：

```text
placement_backend=managed
```

这避免把 trace-only 阶段误读成已经使用 GPU-only allocation。

### 3.3 新增环境变量

新增：

```text
VLLM_UVM_DEVICE_DIRECT_ENABLE
VLLM_UVM_DEVICE_DIRECT_MIN_BYTES
VLLM_UVM_DEVICE_DIRECT_MAX_BYTES
```

默认值：

```text
VLLM_UVM_DEVICE_DIRECT_ENABLE=0
VLLM_UVM_DEVICE_DIRECT_MIN_BYTES=4096
VLLM_UVM_DEVICE_DIRECT_MAX_BYTES=1048576
```

当前阶段：

1. `VLLM_UVM_DEVICE_DIRECT_ENABLE=0` 表示明确 trace-only。
2. `VLLM_UVM_DEVICE_DIRECT_ENABLE=1` 也不会真正切换到 GPU-only。
3. 如果设置为 1，日志中的 reason 会显示 `eligible_but_stage_b_trace_only`。

这为阶段 C 留出开关，但阶段 B 不启用真实 backend 切换。

### 3.4 新增 `TRACE_POLICY` 字段

`TRACE_POLICY` 新增：

```text
placement_backend=<managed>
device_direct_eligible=<0|1>
device_direct_reason=<reason>
cpu_access_risk=<unknown>
hot_gap_match=<0|1>
```

字段含义：

1. `placement_backend`
   - 当前实际 placement backend。
   - 阶段 B 固定为 `managed`。
2. `device_direct_eligible`
   - 当前 allocation 是否满足 device-direct 候选条件。
3. `device_direct_reason`
   - 为什么 eligible 或为什么不 eligible。
4. `cpu_access_risk`
   - 当前阶段还没有 CPU 访问证明，固定为 `unknown`。
5. `hot_gap_match`
   - allocation 是否与当前 watched gap 有 overlap。

### 3.5 新增 `TRACE_GAP_WATCH_ALLOC/FREE` 字段

`TRACE_GAP_WATCH_ALLOC` 和 `TRACE_GAP_WATCH_FREE` 也会输出：

```text
placement_backend
device_direct_eligible
device_direct_reason
cpu_access_risk
hot_gap_match
```

这样可以直接从 gap-watch 视角统计：

1. 有多少 overlap allocation 会成为 device-direct 候选。
2. 候选是否来自目标 class。
3. 候选是否仍然短生命周期。

### 3.6 新增统计计数

新增 session summary 计数：

```text
Device-direct trace allocations
Device-direct eligible allocations
Device-direct requested bytes
```

含义：

1. `Device-direct trace allocations`
   - 请求 device-direct 类动作的 allocation 次数。
2. `Device-direct eligible allocations`
   - 满足当前 eligibility 条件的次数。
3. `Device-direct requested bytes`
   - 请求 device-direct 类动作的累计分配字节数。

---

## 4. eligibility 规则

阶段 B 的 device-direct eligibility 只用于日志，不用于改变行为。

当前规则：

1. policy action 是 `device_direct_trace` 或 `device_direct`
2. `device >= 0`
3. `hot_gap_match=1`
4. `gap_watch_class_match=1`
5. `size >= VLLM_UVM_DEVICE_DIRECT_MIN_BYTES`
6. `size <= VLLM_UVM_DEVICE_DIRECT_MAX_BYTES`

如果全部满足：

```text
device_direct_eligible=1
```

否则：

```text
device_direct_eligible=0
```

### 4.1 `device_direct_reason`

可能值：

```text
not_requested
invalid_device
no_hot_gap_match
target_class_mismatch
below_min_bytes
above_max_bytes
trace_only_not_enabled
eligible_but_stage_b_trace_only
```

其中：

1. `trace_only_not_enabled`
   - 满足候选条件，但 `VLLM_UVM_DEVICE_DIRECT_ENABLE=0`。
2. `eligible_but_stage_b_trace_only`
   - 即使打开 enable，阶段 B 也仍然不切换 backend。

---

## 5. `run_kv_fault_ratio.sh` 修改

### 5.1 新增参数

新增：

```text
--uvm-device-direct-enable <0|1>
--uvm-device-direct-min-bytes <n>
--uvm-device-direct-max-bytes <n>
```

新增 action 白名单：

```text
device_direct_trace
device_direct
```

因此现在可用：

```text
--uvm-gap-watch-policy-action device_direct_trace
--uvm-gap-watch-policy-action device_direct
--auto-gap-watch-policy-action-override device_direct_trace
--auto-gap-watch-policy-action-override device_direct
```

### 5.2 透传环境变量

server 启动时会透传：

```text
VLLM_UVM_DEVICE_DIRECT_ENABLE
VLLM_UVM_DEVICE_DIRECT_MIN_BYTES
VLLM_UVM_DEVICE_DIRECT_MAX_BYTES
```

---

## 6. `discover_gap_watch.py` 修改

`--policy-action-override` 现在支持：

```text
observe
prefetch
advise_prefetch
device_direct_trace
device_direct
```

因此 same-run 自动流程可以直接下发：

```text
policy_action=device_direct_trace
```

或：

```text
policy_action=device_direct
```

---

## 7. `summarize_gap_watch_metrics.py` 修改

新增解析和输出：

```text
placement_backend_counts
device_direct_reason_counts
device_direct_trace_records
device_direct_eligible_records
hot_gap_match_records
```

这些字段用于回答：

1. 当前是否仍然是 managed backend。
2. 有多少 allocation 请求了 device-direct 类动作。
3. 有多少 allocation 满足 device-direct 候选条件。
4. 这些 allocation 是否确实命中 hot gap。

---

## 8. 推荐验证命令

### 8.1 device-direct trace-only 实验

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm

./run_kv_fault_ratio.sh \
  --mode trace \
  --with-address-log \
  --trace-log /tmp/uvm_kv_fault_stats_gap2_device_trace.log \
  --address-trace-log /tmp/uvm_kv_fault_addrs_gap2_device_trace.log \
  --allocator-log /tmp/vllm_uvm_allocator_trace_gap2_device_trace.log \
  --uvm-trace-min-bytes 1048576 \
  --uvm-unknown-detail-enable 1 \
  --uvm-unknown-detail-min-bytes 4096 \
  --uvm-gap-watch-name same_run_gap2_device_trace \
  --uvm-gap-watch-all-classes 1 \
  --uvm-gap-watch-min-bytes 4096 \
  --uvm-device-direct-enable 0 \
  --uvm-device-direct-min-bytes 4096 \
  --uvm-device-direct-max-bytes 1048576 \
  --auto-gap-watch-enable 1 \
  --auto-gap-watch-probe-prompts 1 \
  --auto-gap-watch-target-gap 2 \
  --auto-gap-watch-policy-action-override device_direct_trace \
  --prompts 20 \
  --gap-watch-metrics-summary-json /tmp/vllm_gap_watch_metrics_gap2_device_trace.json \
  --auto-gap-watch-summary-json /tmp/vllm_auto_gap_watch_summary_gap2_device_trace.json \
  --auto-gap-watch-post-main-summary-json /tmp/vllm_auto_gap_watch_post_main_summary_gap2_device_trace.json
```

### 8.2 单独汇总

```bash
python3 /home/ubuntu/nvidia-uvm-gpu/workloads/vllm/summarize_gap_watch_metrics.py \
  --allocator-log /tmp/vllm_uvm_allocator_trace_gap2_device_trace.log \
  --summary-json /tmp/vllm_gap_watch_metrics_gap2_device_trace.json
```

---

## 9. 阶段 B 成功标准

运行完成后应满足：

1. vLLM 能完整跑完 probe + main。
2. allocator log 中出现：
   - `action=device_direct_trace`
   - `placement_backend=managed`
   - `device_direct_eligible=...`
   - `device_direct_reason=...`
3. `summarize_gap_watch_metrics.py` 输出：
   - `device_direct_trace_records > 0`
   - `placement_backend_counts` 中主要为 `managed`
   - `hot_gap_match_records > 0`
4. `gap2` fault 行为应接近 observe 组，而不是 prefetch 组。
5. 不应出现 CUDA illegal access、allocator crash 或 free mismatch。

---

## 10. 阶段 B 与阶段 C 的边界

阶段 B 不解决：

1. 真正 GPU-only 分配。
2. backend tracking。
3. `cudaMallocAsync` pool。
4. device pointer CPU access 风险。

阶段 C 才会引入：

1. `AllocationBackend`
2. backend-specific free
3. `cudaMallocAsync` / `cudaFreeAsync`
4. GPU-only memory pressure 监控

因此阶段 B 的判断标准不是 fault 必须下降，而是：

1. 候选对象是否足够集中。
2. 候选对象是否足够短命。
3. 候选对象 size 是否落在安全范围。
4. 日志是否能支撑后续真正切换 backend。

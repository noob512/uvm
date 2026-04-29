# vLLM UVM Stage D / D2 KV Budget Implementation

## 1. 背景

Stage C2 已经验证 `cuda_malloc_async` device-direct backend 可以在 attention runtime scratch 上工作：

1. C2 vs trace-only baseline 的 gap/unknown faults 下降。
2. `cuda_malloc_async` backend 真实命中。
3. CUDA mempool release threshold 配置成功。
4. device-direct live bytes 受 C1 总预算约束。
5. C2 相比 C1 sync backend 在最近 p5 检查中吞吐和 TPOT 更好，但 gap faults 略高，因此仍保持 opt-in。

Stage D 的目标不是继续扩大 runtime scratch device-direct，而是给 KV cache 建立独立预算视角，防止后续 scratch、weights、KV 三类对象在同一 UVM/VRAM 空间里互相抢占且无法解释。

## 2. 实现边界

第一版实现的是 Stage D0/D1 allocator-side KV telemetry + soft budget signal。

allocator 现在负责：

1. 识别 `initialize_kv_cache` 阶段产生的 KV cache allocation。
2. 统计 KV requested/live/peak bytes。
3. 记录 KV budget bytes、budget mode、remaining bytes。
4. 在超预算时输出明确 reason。
5. 在 `enforce` 模式下输出软拒绝计数。

allocator 仍然不做：

1. 不私自驱逐 KV block。
2. 不私自 swap KV block。
3. 不私自迁移正在使用的 KV。
4. 不修改 vLLM block table。
5. 不在 KV 超预算时返回 NULL 破坏 server 启动。

原因是 allocator 只知道地址和 size，不知道 request、block table、active sequence、可重算性、swap 安全点。真正的 KV eviction/swap/recompute 必须在 vLLM block manager 或 scheduler-aware KV manager 中完成。

本次继续实现 Stage D2 semantic KV budget enforcement。D2 的 enforcement 位置从 allocator 后移/上移到 vLLM KV config 生成路径，在真正分配 KV cache tensor 之前减少 KV block 数，确保实际 KV allocation 不超过预算。

## 3. 参数

### 3.1 环境变量

```text
VLLM_UVM_KV_BUDGET_BYTES=<bytes>
VLLM_UVM_KV_BUDGET_MODE=trace_only|enforce
```

语义：

1. `VLLM_UVM_KV_BUDGET_BYTES=0` 表示 KV budget 不限额，只记录 KV live/peak。
2. `trace_only` 表示只记录是否超预算，不产生 reject signal。
3. `enforce` 在 Stage D0/D1 中表示 allocator 产生 soft signal。
4. `enforce` 在 Stage D2 中还会触发 vLLM semantic KV budget enforcement，使 KV cache config 在分配前被预算 cap。

Stage D2 之后，`enforce` 不再只是 allocator soft signal。真正的硬约束发生在 vLLM KV cache config 层：如果 budget 足够容纳至少一个 KV block，则 vLLM 会减少 `num_blocks` 和 KV tensor size；如果 budget 连一个 KV block 都放不下，则 vLLM 会在 KV 初始化前失败，并输出明确错误。

### 3.2 runner 参数

`workloads/vllm/run_kv_fault_ratio.sh` 新增：

```bash
--uvm-kv-budget-bytes <n>
--uvm-kv-budget-mode trace_only|enforce
```

这些参数会被传入 vLLM server 进程：

```text
VLLM_UVM_KV_BUDGET_BYTES
VLLM_UVM_KV_BUDGET_MODE
```

## 4. D0/D1 allocator 实现

实现文件：

```text
workloads/vllm/vllm/uvm_test/uvm_allocator.cpp
```

### 4.1 KV allocation 识别

已有分类器会把如下 phase 识别为 KV：

```cpp
if (phase == "initialize_kv_cache") {
    return AllocationClass::KvPersistent;
}
```

vLLM 侧已有 phase wrapper：

```text
workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py
```

其中 `initialize_kv_cache()` 使用：

```python
with uvm_allocation_phase("initialize_kv_cache"):
    ...
```

因此 allocator 可以在 KV cache tensor 初始化阶段看到稳定的 `kv_persistent` 分类。

### 4.2 新增 telemetry counters

新增全局指标：

```text
kv_trace_allocs
kv_requested_bytes
kv_live_bytes
kv_peak_live_bytes
kv_budget_over_allocs
kv_budget_reject_allocs
kv_free_success_allocs
```

分配时：

1. `kv_trace_allocs += 1`
2. `kv_requested_bytes += size`
3. `kv_live_bytes += size`
4. `kv_peak_live_bytes = max(kv_peak_live_bytes, kv_live_bytes)`
5. 如果 `kv_budget_bytes > 0 && kv_live_bytes > kv_budget_bytes`，记录 over-budget。
6. 如果 budget mode 是 `enforce`，额外记录 soft reject。

释放时：

1. 如果该 allocation 被记录为 KV allocation，且 `cudaFree/cudaFreeAsync` 成功，则扣减 `kv_live_bytes`。
2. 增加 `kv_free_success_allocs`。

### 4.3 reason 设计

KV budget reason 包括：

```text
not_kv
kv_budget_unlimited
kv_budget_within_budget
kv_budget_exceeded_trace_only
kv_budget_exceeded_soft_enforce
```

含义：

1. `not_kv`：非 KV allocation。
2. `kv_budget_unlimited`：KV allocation，但 budget bytes 为 0。
3. `kv_budget_within_budget`：KV allocation 且未超预算。
4. `kv_budget_exceeded_trace_only`：KV allocation 超预算，但只观测。
5. `kv_budget_exceeded_soft_enforce`：KV allocation 超预算，产生 enforce soft signal，但不硬失败。

## 5. D2 vLLM 语义层实现

实现文件：

```text
workloads/vllm/vllm/vllm/v1/core/kv_cache_utils.py
workloads/vllm/vllm/vllm/v1/kv_cache_interface.py
```

### 5.1 enforcement 入口

vLLM 的 KV cache 初始化顺序是：

```text
engine/core.py::_initialize_kv_caches
  -> model_executor.determine_available_memory()
  -> get_kv_cache_configs(vllm_config, kv_cache_specs, available_memory)
  -> model_executor.initialize_from_config(kv_cache_configs)
  -> gpu_model_runner.initialize_kv_cache()
```

D2 在 `get_kv_cache_configs()` 内落地，因为这里还没有真正分配 KV tensor，但已经有完整的 KV spec、layer grouping 和 available memory。

### 5.2 available memory cap

新增逻辑：

```python
effective_available_memory = _apply_uvm_kv_budget_to_available_memory(
    available_memory
)
```

当：

```text
VLLM_UVM_KV_BUDGET_BYTES > 0
VLLM_UVM_KV_BUDGET_MODE == enforce
```

每个 worker 的 KV planning memory 会被 cap 到：

```text
min(profiled_available_memory, VLLM_UVM_KV_BUDGET_BYTES)
```

这会直接减少生成的 `num_blocks`，从源头减少 KV cache tensor 大小。

### 5.3 final guardrail

为了防止 `num_gpu_blocks_override` 或 hybrid KV alignment 绕过预算，D2 在每个 worker 的 `KVCacheConfig` 生成后再次检查：

```python
current_bytes = sum(tensor.size for tensor in kv_cache_config.kv_cache_tensors)
```

如果 `current_bytes > budget_bytes`，则按每 block 的总字节数缩小：

```text
new_num_blocks = floor(budget_bytes / bytes_per_block)
```

然后按比例缩小所有 `KVCacheTensor.size`。

如果 `budget_bytes < bytes_per_block`，说明连一个 KV block 都无法容纳，D2 会在初始化阶段抛出 `ValueError`。这是语义层的安全失败，比 allocator 分配后返回 NULL 更容易定位，也不会发生半初始化的 block table 状态。

### 5.4 KVCacheConfig metadata

`KVCacheConfig` 增加 D2 元数据字段：

```text
uvm_kv_budget_bytes
uvm_kv_budget_mode
uvm_kv_budget_enforced
uvm_kv_budget_original_num_blocks
uvm_kv_budget_original_bytes
uvm_kv_budget_final_bytes
```

这些字段用于日志、调试和后续 scheduler/block-manager 策略接入。

### 5.5 D2 与 allocator telemetry 的闭环

D2 的效果最终仍由 D0/D1 allocator telemetry 验证：

1. vLLM D2 先减少 `KVCacheConfig.num_blocks`。
2. `initialize_kv_cache` 分配更小的 KV tensors。
3. allocator 识别这些 tensors 为 `kv_persistent`。
4. `kv_peak_live_bytes_observed <= VLLM_UVM_KV_BUDGET_BYTES`。

因此 D2 成功标准不再是“出现 over-budget signal”，而是“enforce 模式下实际 KV peak live bytes 不超过预算”。

## 6. 日志与 JSON 指标

### 6.1 TRACE_POLICY 字段

`TRACE_POLICY` 增加：

```text
kv_budget_tracked=<0|1>
kv_budget_over_budget=<0|1>
kv_budget_reason=<reason>
kv_live_bytes=<n>
kv_budget_bytes=<n>
kv_budget_remaining=<n>
kv_budget_mode=<trace_only|enforce>
```

这些字段不依赖 gap watch，因此即使没有开启 Stage C gap watch，也可以在 KV 初始化阶段直接被解析。

### 6.2 TRACE_GAP_WATCH_ALLOC/FREE 字段

为了和 C/C2 gap watch 分析统一，gap watch alloc/free 日志也带上 KV budget 字段。正常 Stage D 验证主要看 `TRACE_POLICY`，但如果未来要观察 KV range 与某个 gap 的重叠，这些字段可以直接复用。

### 6.3 session summary 字段

allocator close 时输出：

```text
KV budget bytes
KV budget mode
KV trace allocations
KV requested bytes
KV live bytes
KV peak live bytes
KV budget remaining
KV budget over allocations
KV budget reject allocations
KV free success
```

Python 侧现在通过 `atexit` 调用 `uvm_close_log()`，让正常退出时 session summary 更稳定地落盘。不过 Stage D 的成功检查不依赖 summary，因为 server 运行中也会实时 flush `TRACE_POLICY`。

### 6.4 metrics JSON 字段

`workloads/vllm/summarize_gap_watch_metrics.py` 增加：

```json
{
  "kv_budget_bytes": 1048576,
  "kv_budget_mode": "trace_only",
  "kv_trace_allocations": 1,
  "kv_live_bytes": 123456789,
  "kv_peak_live_bytes_observed": 123456789,
  "kv_min_budget_remaining_observed": 0,
  "kv_budget_over_records": 1,
  "kv_budget_reject_records": 0,
  "kv_budget_reason_counts": {
    "kv_budget_exceeded_trace_only": 1
  }
}
```

## 7. 验证脚本

### 7.1 D0/D1 一键检查

新增：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./check_stage_d_success.py
```

默认行为：

1. 启动 vLLM server。
2. 等待 server ready 和 KV range 出现。
3. 不跑 benchmark，仅验证 KV 初始化阶段。
4. 设置 `--uvm-kv-budget-bytes 1048576`。
5. 设置 `--uvm-kv-budget-mode trace_only`。
6. 生成 allocator log 和 metrics JSON。
7. 检查 KV telemetry 是否存在。

默认使用 1 MiB budget 是为了让大 KV cache 初始化稳定触发 over-budget signal，证明 Stage D 的预算信号可见；由于是 `trace_only`，不会产生 soft reject，也不会影响服务正确性。

### 7.2 D0/D1 wrapper

新增：

```bash
./run_stage_d_kv_budget_check.sh
```

可配置环境变量：

```bash
KV_BUDGET_BYTES=1048576 \
KV_BUDGET_MODE=trace_only \
RUN_BENCH=0 \
./run_stage_d_kv_budget_check.sh
```

如果希望跑一个真实小 benchmark：

```bash
RUN_BENCH=1 PROMPTS=1 ./run_stage_d_kv_budget_check.sh
```

### 7.3 D2 semantic enforcement 检查

新增：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./check_stage_d2_success.py
```

默认行为：

1. 设置 `VLLM_UVM_KV_BUDGET_MODE=enforce`。
2. 设置 `VLLM_UVM_KV_BUDGET_BYTES=2147483648`，即 2 GiB。
3. 启动 vLLM server 到 KV cache 初始化完成。
4. 不跑 benchmark。
5. 用 allocator telemetry 验证实际 KV peak live bytes 不超过 budget。

wrapper：

```bash
KV_BUDGET_BYTES=2147483648 ./run_stage_d2_kv_budget_check.sh
```

如果希望跑一个真实小 benchmark：

```bash
RUN_BENCH=1 PROMPTS=1 ./run_stage_d2_kv_budget_check.sh
```

注意：如果 budget 小到连一个 KV block 都无法容纳，D2 会让 server 初始化失败，这是预期的硬约束行为。此时应增大 `KV_BUDGET_BYTES` 或降低模型上下文/并发配置。

### 7.4 离线验证

已有 allocator log：

```bash
./check_stage_d_success.py \
  --skip-run \
  --allocator-log /tmp/path/to/vllm_uvm_allocator_trace.log \
  --budget-bytes 1048576 \
  --budget-mode trace_only
```

已有 metrics JSON：

```bash
./check_stage_d_success.py \
  --skip-run \
  --metrics-json /tmp/path/to/vllm_stage_d_kv_budget_metrics.json \
  --budget-bytes 1048576 \
  --budget-mode trace_only
```

## 8. 成功标准

Stage D0/D1 当前成功标准：

1. server 能启动并完成 KV cache 初始化。
2. `kv_trace_allocations > 0`。
3. `kv_peak_live_bytes_observed > 0`。
4. `kv_budget_bytes` 与配置一致。
5. `kv_budget_mode` 与配置一致。
6. 如果 budget 非 0 且 peak live 超过 budget，则 `kv_budget_over_records > 0`。
7. `trace_only` 模式下 `kv_budget_reject_records == 0`。
8. `enforce` 模式下如果超预算，则 `kv_budget_reject_records > 0`，但这仍是软信号。

D2 成功标准：

1. `VLLM_UVM_KV_BUDGET_MODE=enforce`。
2. `kv_trace_allocations > 0`。
3. `kv_peak_live_bytes_observed > 0`。
4. `kv_peak_live_bytes_observed <= VLLM_UVM_KV_BUDGET_BYTES`。
5. `kv_budget_over_records == 0` 或不再增长。
6. server 初始化无半初始化错误。
7. 如果启用 benchmark，则 `Failed requests = 0`。

## 9. 与 Stage C2 的关系

Stage C2 的 device-direct budget 和 Stage D/D2 的 KV budget 是两套指标：

1. C1/C2 约束 runtime scratch device-direct 的真实 GPU-only live bytes。
2. Stage D0/D1 观测 KV cache managed allocation 的逻辑预算。
3. Stage D2 在 vLLM 层减少 KV blocks，使实际 KV allocation 不超过预算。
4. Stage D/D2 不会把 KV cache 改成 device-direct。
5. Stage D/D2 不会改变 C2 的 backend 选择。

这意味着可以同时运行：

```bash
DEVICE_DIRECT_BACKEND=cuda_malloc_async \
DEVICE_DIRECT_POOL_RELEASE_THRESHOLD=1048576 \
VLLM_UVM_KV_BUDGET_BYTES=1048576 \
VLLM_UVM_KV_BUDGET_MODE=trace_only \
./run_stage_c_attention_backend_ab.sh
```

生成的 C/C2 comparison JSON 会透传 `kv_*` metrics，方便确认 attention scratch 优化没有吞掉 KV 预算视角。

## 10. 后续 Stage D3

当前 D2 已经实现“初始化期语义预算”：通过减少 KV blocks 控制 KV cache 总大小。后续如果要实现运行时 KV pool 自适应，应继续进入 block manager：

1. 在 KV cache manager 或 block manager 中计算 block-level budget。
2. 在每次 request admission / block allocation 前检查剩余 budget。
3. 选择可 swap/evict/recompute 的 inactive block。
4. 更新 block table 和 prefix cache metadata。
5. 与 scheduler 协同，避免驱逐活跃 request。
6. 将 allocator 的 `kv_budget_exceeded_soft_enforce` 和 D2 metadata 作为上层策略输入，而不是让 allocator 直接做驱逐。

这一阶段才可以定义真正的：

```text
VLLM_UVM_KV_SWAP_ENABLE
VLLM_UVM_KV_EVICTION_POLICY
```

当前 D2 先把初始化期 KV pool 大小控制住，确保后续运行时 block-manager eviction/swap 有可靠的预算上限和 ground truth。

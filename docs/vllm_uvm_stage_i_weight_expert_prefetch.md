# vLLM UVM Stage I MoE Expert Weight Prefetch/Offload 执行实现说明

本文档记录 Stage I 的完整实现：在 Stage H 已经能生成 expert hot/cold trace-only plan 后，Stage I 在 MoE layer 的安全点上执行两类 opt-in 动作：

1. 对 hot active expert weight slice 发起 GPU prefetch。
2. 对 Stage H 计划中的 cold expert weight slice 发起可选 CPU preferred-location advise 或 CPU prefetch。

Stage I 的目标不是默认提升性能，而是证明“基于 MoE 语义的权重迁移动作可以安全接入运行时路径，并且有预算、trace 和一键验收闭环”。

## 1. 目标

Stage I 要证明：

1. MoE routing 后、expert kernel 前存在可用安全点。
2. 可以从 `topk_ids` 得到当前 layer 的 active experts。
3. 可以把 global expert id 映射到本 rank 的 local expert id。
4. 可以用 Stage H `prefetch_plan` 过滤 hot expert prefetch。
5. 可以用 Stage H `offload_plan` 选择 cold expert，并跳过当前 step active expert。
6. prefetch/offload 都受 per-layer-call 字节预算和 expert 数量预算限制。
7. benchmark failed requests 保持 0。

## 2. 修改文件

Stage I 主要修改：

```text
workloads/vllm/vllm/vllm/device_allocator/uvm.py
workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py
workloads/vllm/run_kv_fault_ratio.sh
workloads/vllm/check_stage_i_success.py
docs/vllm_uvm_stage_i_weight_expert_prefetch.md
```

## 3. 执行边界

Stage I 执行：

1. hot expert GPU prefetch。
2. cold expert CPU advise 或 CPU prefetch。
3. JSONL trace。
4. action budget。
5. Stage H plan-gated validation。

Stage I 不执行：

1. Tensor storage 替换。
2. 删除或卸载 PyTorch Parameter。
3. KV eviction/swap。
4. 全局 pressure coordinator。
5. 跨 token/跨 layer 预测式 prefetch。

## 4. 插入点

插入点在：

```text
vllm/model_executor/layers/fused_moe/layer.py
  FusedMoE.select_experts()
```

顺序：

```text
router_logits -> topk_ids
  -> Stage E MoE routing trace
  -> Stage I hot active expert GPU prefetch
  -> Stage I cold inactive expert CPU advise/offload
  -> quant_method.apply / fused expert kernel
```

这个位置的优势是：

1. 已经知道当前 token 会访问哪些 expert。
2. 还没有进入 expert GEMM kernel。
3. 不需要 allocator 猜测权重语义。
4. 不改变 tensor storage。

## 5. allocator Python wrapper

`vllm/device_allocator/uvm.py` 新增 raw range helper：

```python
prefetch_range_to_device(ptr, size, device)
prefetch_range_to_cpu(ptr, size)
advise_range_preferred_location(ptr, size, device)
```

其中：

1. `prefetch_range_to_device(..., device>=0)` 调用 allocator `.so` 的 `uvm_prefetch`，底层是 `cudaMemPrefetchAsync`。
2. `prefetch_range_to_cpu(..., device=-1)` 用 `cudaMemPrefetchAsync(..., cudaCpuDeviceId/-1)` 将 range 预取到 CPU。
3. `advise_range_preferred_location(..., device=-1)` 调用 `cudaMemAdviseSetPreferredLocation`，把 cold range 的 preferred location 设为 CPU。

这些 helper 返回“调用是否已发出”。底层 CUDA error 仍由 allocator shim 打印，Stage I 用 JSONL trace 和 failed requests 做端到端安全验证。

## 6. FusedMoE 实现

### 6.1 active expert 识别

Stage I 从 `topk_ids` 得到当前 step 的 active global expert id，再通过 `self.expert_map` 转成本 rank local expert id：

```text
topk_ids -> unique global expert ids -> expert_map -> local expert ids
```

如果没有 expert parallel map，则默认：

```text
local_expert_id = global_expert_id
```

### 6.2 weight slice 定位

当前支持两类 fused MoE expert weights：

```text
w13_weight -> moe_gate_up
w2_weight  -> moe_down
```

slice 计算方式：

```text
slice_bytes = tensor.numel() * tensor.element_size() / tensor.shape[0]
slice_ptr = tensor.data_ptr() + local_expert_id * slice_bytes
```

这与 Stage H 对 fused tensor 按第 0 维拆 logical expert slice 的方式一致。

### 6.3 Stage H plan-gated prefetch

可选配置：

```text
VLLM_UVM_WEIGHT_PREFETCH_PLAN_FILE=<stage_h_plan.json>
VLLM_UVM_WEIGHT_PREFETCH_REQUIRE_PLAN=0|1
```

如果 `PREFETCH_PLAN_FILE` 存在，Stage I 只对 `(layer_id, expert_id)` 出现在 `prefetch_plan` 中的 active expert 发起 prefetch。

如果 `REQUIRE_PLAN=1` 且没有可加载 plan，则不发 active-expert prefetch。这样验收脚本可以证明 Stage I 真正使用了 Stage H hot plan，而不是无差别预取所有 active expert。

### 6.4 cold expert offload/advise

可选配置：

```text
VLLM_UVM_WEIGHT_OFFLOAD_ENABLE=0|1
VLLM_UVM_WEIGHT_OFFLOAD_MODE=trace_only|advise_cpu|prefetch_cpu
VLLM_UVM_WEIGHT_OFFLOAD_PLAN_FILE=<stage_h_plan.json>
```

执行规则：

1. 只处理 Stage H `offload_plan` 中的 `(layer_id, expert_id)`。
2. 如果该 expert 是当前 step active expert，则跳过，避免把马上要用的权重推向 CPU。
3. 每个 `(layer_id, expert_id)` 在单个 layer 实例内只处理一次，避免 decode 阶段重复发 advise/offload。
4. `trace_only` 只写 `trace_offload_candidate`。
5. `advise_cpu` 发出 `cudaMemAdviseSetPreferredLocation(..., CPU)`。
6. `prefetch_cpu` 发出 `cudaMemPrefetchAsync(..., CPU)`。

默认验收使用 `advise_cpu`，因为它比真正 CPU prefetch 更温和，风险更低。

## 7. 配置项

runner 支持：

```text
--uvm-weight-prefetch-enable <0|1>
--uvm-weight-prefetch-mode <trace_only|prefetch>
--uvm-weight-prefetch-trace-file <path>
--uvm-weight-prefetch-max-bytes-per-step <n>
--uvm-weight-prefetch-max-experts-per-layer <n>
--uvm-weight-prefetch-target-roles <csv>
--uvm-weight-prefetch-device <n>
--uvm-weight-prefetch-plan-file <path>
--uvm-weight-prefetch-require-plan <0|1>
--uvm-weight-offload-enable <0|1>
--uvm-weight-offload-mode <trace_only|advise_cpu|prefetch_cpu>
--uvm-weight-offload-plan-file <path>
--uvm-weight-offload-max-bytes-per-step <n>
--uvm-weight-offload-max-experts-per-layer <n>
--uvm-weight-offload-target-roles <csv>
```

对应环境变量：

```text
VLLM_UVM_WEIGHT_PREFETCH_ENABLE
VLLM_UVM_WEIGHT_PREFETCH_MODE
VLLM_UVM_WEIGHT_PREFETCH_TRACE_FILE
VLLM_UVM_WEIGHT_PREFETCH_MAX_BYTES_PER_STEP
VLLM_UVM_WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER
VLLM_UVM_WEIGHT_PREFETCH_TARGET_ROLES
VLLM_UVM_WEIGHT_PREFETCH_DEVICE
VLLM_UVM_WEIGHT_PREFETCH_PLAN_FILE
VLLM_UVM_WEIGHT_PREFETCH_REQUIRE_PLAN
VLLM_UVM_WEIGHT_OFFLOAD_ENABLE
VLLM_UVM_WEIGHT_OFFLOAD_MODE
VLLM_UVM_WEIGHT_OFFLOAD_PLAN_FILE
VLLM_UVM_WEIGHT_OFFLOAD_MAX_BYTES_PER_STEP
VLLM_UVM_WEIGHT_OFFLOAD_MAX_EXPERTS_PER_LAYER
VLLM_UVM_WEIGHT_OFFLOAD_TARGET_ROLES
```

## 8. Trace 字段

Stage I trace 文件默认为：

```text
vllm_uvm_weight_prefetch_stage_i.jsonl
```

hot prefetch action：

```text
trace_prefetch_candidate
prefetch_issued
prefetch_skipped
budget_reject
step_summary
```

cold offload action：

```text
trace_offload_candidate
offload_advise_cpu_issued
offload_prefetch_cpu_issued
offload_skipped
offload_budget_reject
offload_step_summary
```

常见字段：

```text
layer_name
step
mode
action
role
global_expert_id
local_expert_id
ptr
bytes
device
success
issued_records
issued_bytes
attempted_bytes
budget_reject_records
max_bytes_per_step
```

## 9. 检查脚本

新增/升级：

```text
workloads/vllm/check_stage_i_success.py
```

默认命令：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./check_stage_i_success.py
```

默认流程：

1. 运行 planning probe。
2. 输出 weight map 和 MoE routing trace。
3. 调用 `plan_stage_h_weight_expert_actions.py` 生成 Stage H hot/cold plan。
4. 运行 execution probe。
5. execution probe 使用 `--uvm-weight-prefetch-plan-file` 和 `--uvm-weight-offload-plan-file`。
6. 开启 `--uvm-weight-prefetch-require-plan 1`，证明 prefetch 由 Stage H plan 驱动。
7. 默认开启 cold offload `advise_cpu`。
8. 解析 Stage I JSONL trace、metrics 和 benchmark log。

## 10. 成功标准

PASS 需要：

```text
prefetch_trace_present=True
prefetch_trace_records_present=True
prefetch_candidates_or_issued_present=True
prefetch_issued_present_in_prefetch_mode=True
prefetch_step_bytes_within_budget=True
plan_json_present=True
prefetch_plan_records_present=True
offload_plan_records_present=True
offload_action_present=True
offload_step_bytes_within_budget=True
moe_routing_trace_present=True
weight_map_present=True
benchmark_no_failed_requests=True
allocator_weight_metrics_present=True
pool_registry_enabled=True
runner_log_clean=True
```

如果使用：

```bash
./check_stage_i_success.py --disable-offload
```

则只验证 hot expert prefetch，不要求 cold offload action。

## 11. 复用已有日志

可以跳过新运行：

```bash
./check_stage_i_success.py \
  --skip-run \
  --prefetch-trace-jsonl /tmp/run/vllm_uvm_weight_prefetch_stage_i.jsonl \
  --plan-json /tmp/run/vllm_stage_i_weight_expert_plan.json \
  --moe-routing-jsonl /tmp/run/vllm_uvm_moe_routing_stage_i.jsonl \
  --weight-map-jsonl /tmp/run/vllm_uvm_weight_regions_stage_i.jsonl \
  --bench-log /tmp/run/vllm_bench_stage_i.log
```

## 12. 最终判断

Stage I 当前已经从“active expert prefetch 最小原型”升级为完整的 Stage I 执行闭环：

1. hot expert prefetch executor 已实现。
2. Stage H `prefetch_plan` 可作为 hot whitelist。
3. cold expert `offload_plan` 可执行 trace-only、CPU advise 或 CPU prefetch。
4. action budget 覆盖 prefetch 和 offload。
5. 当前 step active expert 不会被 cold offload。
6. 所有动作写入 JSONL trace。
7. `check_stage_i_success.py` 能自动生成 plan 并验证执行是否成功。

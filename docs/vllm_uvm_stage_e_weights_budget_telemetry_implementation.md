# vLLM UVM Stage E Weights Budget Telemetry Implementation

## 1. 背景

Stage D2 已经把 KV cache 的初始化大小纳入独立预算：`VLLM_UVM_KV_BUDGET_MODE=enforce` 时，vLLM 会在生成 KV cache config 前缩小 KV blocks，最终通过 allocator telemetry 验证 KV live/peak 不超过预算。

Stage E 的目标不是立刻实现完整 weights eviction/offload/prefetch，而是先把模型权重从同一个“managed memory 大池”里分出独立观测边界。这样后续做 per-pool eviction/prefetch 时，系统可以明确回答：

1. KV 超载时，只应影响 KV pool。
2. weights 超载时，只应影响 weights pool。
3. runtime scratch 的 device-direct 或 UVM 活动不应挤占 KV/weights 的解释口径。

## 2. 当前实现边界

当前 Stage E 实现的是 weights 初始化期预算遥测、权重语义地址 map，以及可选 MoE expert routing 热度 trace。

allocator 现在负责：

1. 识别 `load_model` 阶段产生的 `weight_persistent` allocation。
2. 统计 weights requested/live/peak bytes。
3. 记录 weights budget bytes、budget mode、remaining bytes。
4. 在超预算时输出明确 reason。
5. 在 `enforce` 模式下输出 soft reject 计数。

vLLM 侧现在负责：

1. 在 `load_model` 结束后输出 weight tensor -> address range 的 JSONL sidecar。
2. 为每条 weight 记录补充 `layer_id`、`expert_id`、`role`、`shard_id`、`dtype`、`shape` 等语义标签。
3. 在显式打开 MoE routing trace 时，按层/step 记录 expert token counts。

allocator 仍然不做：

1. 不私自驱逐 weights。
2. 不私自 offload weights。
3. 不私自 prefetch/offload MoE expert weights。
4. 不在 weights 超预算时返回 NULL 破坏模型加载。
5. 不改变 model loader 的参数加载顺序或 tensor placement。

原因是 allocator 只知道 allocation 的 phase、size 和地址，不知道权重名字、层号、expert id、下一步路由概率，也不知道某个权重 tensor 是否可安全迁移。真正的 weights-only eviction/offload/prefetch 需要 model loader、weight address map、MoE router trace 和 runtime scheduler 共同参与。

## 3. 参数

### 3.1 环境变量

```text
VLLM_UVM_WEIGHT_BUDGET_BYTES=<bytes>
VLLM_UVM_WEIGHT_BUDGET_MODE=trace_only|enforce
```

语义：

1. `VLLM_UVM_WEIGHT_BUDGET_BYTES=0` 表示 weights budget 不限额，只记录 live/peak。
2. `trace_only` 表示只记录是否超预算，不产生 reject signal。
3. `enforce` 在当前 Stage E 中仍是 allocator-side soft signal，不做硬失败、不做 eviction/offload。

### 3.2 runner 参数

`workloads/vllm/run_kv_fault_ratio.sh` 新增：

```bash
--uvm-weight-budget-bytes <n>
--uvm-weight-budget-mode trace_only|enforce
```

这些参数会被传入 vLLM server 进程：

```text
VLLM_UVM_WEIGHT_BUDGET_BYTES
VLLM_UVM_WEIGHT_BUDGET_MODE
```

KV 和 weights budget 是独立参数。Stage E 检查默认会把 `KV_BUDGET_BYTES` 设为 0，避免把 Stage D 的 KV enforcement 混进 weights 初始化验证。

## 4. allocator 实现

实现文件：

```text
workloads/vllm/vllm/uvm_test/uvm_allocator.cpp
```

### 4.1 weights allocation 识别

allocator 已有分类器会把如下 phase 识别为 weights：

```cpp
if (phase == "load_model") {
    return AllocationClass::WeightPersistent;
}
```

vLLM 侧 `gpu_model_runner.py::load_model()` 已使用：

```python
with uvm_allocation_phase("load_model"):
    self.model = model_loader.load_model(...)
```

因此模型加载期间的主要权重 allocation 会被标记为 `weight_persistent`。

### 4.2 新增 telemetry counters

新增全局指标：

```text
weight_trace_allocs
weight_requested_bytes
weight_live_bytes
weight_peak_live_bytes
weight_budget_over_allocs
weight_budget_reject_allocs
weight_free_success_allocs
```

分配时：

1. `weight_trace_allocs += 1`
2. `weight_requested_bytes += size`
3. `weight_live_bytes += size`
4. `weight_peak_live_bytes = max(weight_peak_live_bytes, weight_live_bytes)`
5. 如果 `weight_budget_bytes > 0 && weight_live_bytes > weight_budget_bytes`，记录 over-budget。
6. 如果 budget mode 是 `enforce`，额外记录 soft reject。

释放时：

1. 如果该 allocation 被记录为 weight allocation，且 `cudaFree/cudaFreeAsync` 成功，则扣减 `weight_live_bytes`。
2. 增加 `weight_free_success_allocs`。

### 4.3 reason 设计

weights budget reason 包括：

```text
not_weight
weight_budget_unlimited
weight_budget_within_budget
weight_budget_exceeded_trace_only
weight_budget_exceeded_soft_enforce
```

含义：

1. `not_weight`：非 weights allocation。
2. `weight_budget_unlimited`：weights allocation，但 budget bytes 为 0。
3. `weight_budget_within_budget`：weights allocation 且未超预算。
4. `weight_budget_exceeded_trace_only`：weights allocation 超预算，但只观测。
5. `weight_budget_exceeded_soft_enforce`：weights allocation 超预算，产生 enforce soft signal，但不硬失败。

### 4.4 trace 字段

`TRACE_POLICY` 和 `TRACE_GAP_WATCH_ALLOC/FREE` 新增：

```text
weight_budget_tracked
weight_budget_over_budget
weight_budget_reason
weight_live_bytes
weight_budget_bytes
weight_budget_remaining
weight_budget_mode
```

session summary 新增：

```text
Weight budget bytes
Weight budget mode
Weight trace allocations
Weight requested bytes
Weight live bytes
Weight peak live bytes
Weight budget remaining
Weight budget over allocations
Weight budget reject allocations
Weight free success
```

## 5. 汇总与检查脚本

### 5.1 summarizer

实现文件：

```text
workloads/vllm/summarize_gap_watch_metrics.py
```

新增 JSON 字段：

```text
weight_budget_bytes
weight_budget_mode
weight_trace_allocations
weight_requested_bytes
weight_live_bytes
weight_peak_live_bytes_observed
weight_min_budget_remaining_observed
weight_budget_over_records
weight_budget_reject_records
weight_budget_reason_counts
```

### 5.2 success checker

新增：

```text
workloads/vllm/check_stage_e_success.py
workloads/vllm/run_stage_e_weights_budget_check.sh
```

默认验证命令：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./check_stage_e_success.py
```

或：

```bash
WEIGHT_BUDGET_BYTES=1048576 WEIGHT_BUDGET_MODE=trace_only ./run_stage_e_weights_budget_check.sh
```

默认 `--no-bench`，即 server 启动并完成 model load / KV range 初始化后停止，不跑完整 benchmark。这样可以低成本验证 weights 初始化期遥测是否生效。

如需验证 MoE routing trace，需要跑 benchmark：

```bash
./check_stage_e_success.py --run-bench --enable-moe-routing-trace --require-moe-routing-trace
```

或：

```bash
RUN_BENCH=1 ENABLE_MOE_ROUTING_TRACE=1 REQUIRE_MOE_ROUTING_TRACE=1 ./run_stage_e_weights_budget_check.sh
```

### 5.3 weight semantic map

新增：

```text
workloads/vllm/summarize_stage_e_weight_map.py
```

默认 sidecar：

```text
vllm_uvm_weight_regions_stage_e.jsonl
```

每条 JSONL 记录包含：

```text
name
kind
start/end
size_bytes
dtype
shape
layer_id
expert_id
role
shard_id
is_moe_expert
```

这让后续离线分析可以把 fault address 归因到具体权重 tensor，并进一步区分 attention、MLP、MoE router、MoE expert w1/w2/w3/w13 等类别。

### 5.4 MoE routing trace

新增环境变量：

```text
VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0|1
VLLM_UVM_MOE_ROUTING_TRACE_FILE=<jsonl>
```

每条 JSONL 记录按层/step 聚合：

```text
layer_name
step
num_tokens
top_k
global_num_experts
expert_token_counts
active_experts
```

该 trace 默认关闭，避免每次推理都引入额外 CPU 同步。开启后用于判断 expert 热度是否稳定，以及是否具备预测下一 decode step expert 集合的基础。

## 6. 成功标准

Stage E 当前成功标准：

1. `weight_trace_allocations > 0`。
2. `weight_peak_live_bytes_observed > 0`。
3. `weight_budget_bytes` 与配置一致。
4. `weight_budget_mode` 与配置一致。
5. 非 0 budget 且 peak 超预算时，`weight_budget_over_records > 0`。
6. `trace_only` 模式下 `weight_budget_reject_records == 0`。
7. `enforce` 模式下如果超预算，`weight_budget_reject_records > 0`，但当前仍是 soft signal。
8. runner log 没有 parse failure 或 server early exit。
9. `weight_map_records > 0`。
10. weight map 至少能输出 `weight_map_moe_expert_records` 字段，用于判断当前模型是否包含可识别 MoE expert weights。
11. 如果要求 MoE routing trace，则 `moe_routing_records > 0`。

这证明 Stage E 已经建立 weights pool 的初始化期观测边界、权重语义地址 map，以及可选 MoE expert 热度观测。但它仍不证明已经实现 weights runtime eviction/offload/prefetch。

## 7. 与最终分池目标的关系

当前已经具备三类 pool 的第一层边界：

1. KV pool：Stage D/D2，已有初始化预算硬约束和 allocator telemetry。
2. Runtime scratch pool：Stage C/C2，已有 gap-watch + device-direct backend + 总预算。
3. Weights pool：Stage E，已有初始化期 allocator telemetry + 独立 budget soft signal + weight semantic map + MoE routing trace。

还没有完成的目标：

1. weights pool 的模型语义级硬预算。
2. 基于 fault address + weight map + routing trace 的自动 hot/cold 分类。
3. weights-only eviction/offload。
4. weights-only predictive prefetch。
5. 三个 pool 的统一 admission control 和互不驱逐保障。

## 8. 后续 Stage E2/E3 建议

下一步不建议直接 offload dense/shared weights。更稳的顺序是：

1. E4：把 fault address 与 weight semantic map、MoE routing trace 自动 join，生成 hot/cold expert 列表。
2. E5：只对低风险 expert weights 做 trace-only prefetch/offload 实验。
3. E6：引入 weights pool admission control，确保 weights 策略不会挤占 KV pool 和 runtime scratch pool。

这样做和当前项目目标一致：先把 KV、weights、runtime scratch 的初始化/运行边界拆清楚，再逐步实现各自独立的驱逐和预取策略。

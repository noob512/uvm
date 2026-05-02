# vLLM UVM Stage H Weights Hot/Cold Trace-only Plan 实现说明

本文档记录 Stage H 的实现：在 Stage E 的 weight semantic map、MoE routing trace，以及可选 UVM per-fault address log 基础上，生成 MoE expert weights 的 hot/cold 分类和 trace-only prefetch/offload 候选计划。

Stage H 不执行任何真实迁移，不调用 `cudaMemPrefetchAsync`，也不做 CPU offload。它的目标是把“哪些 expert 权重值得预取、哪些 expert 权重可能适合后续 offload”先稳定算出来，为 Stage I 的小范围执行做准备。

## 1. 目标

Stage H 要回答：

1. weight map 中是否能拿到 MoE expert 权重范围。
2. MoE routing trace 是否能和这些 expert 权重按 `(layer_id, expert_id)` 关联。
3. 哪些 expert 在本次 workload 中是 hot。
4. 哪些 expert 在本次 workload 中是 cold。
5. prefetch/offload 计划是否受字节预算约束。
6. trace-only 计划生成是否不影响 benchmark correctness。

## 2. 数据来源

Stage H 使用三类输入：

1. `vllm_uvm_weight_regions_stage_h.jsonl`
   - 来自 Stage E2 weight semantic map。
   - 包含 tensor name、地址范围、size、layer_id、expert_id、role、shape、dtype。

2. `vllm_uvm_moe_routing_stage_h.jsonl`
   - 来自 Stage E3 MoE routing trace。
   - 包含 layer_name、step、top_k、expert_token_counts、active_experts。

3. `uvm_kv_fault_addrs_stage_h.log`
   - 来自 `run_kv_fault_ratio.sh --with-address-log`。
   - 可选用于把 replayable fault address join 到 expert weight range。

Stage H 的主信号是 routing trace。fault join 是附加证据，因为真实 UVM fault 可能落在 weights、KV、runtime scratch 或其他 managed allocation 上，不保证每次小 benchmark 都能命中 expert weight range。

## 3. Fused Expert Tensor 处理

vLLM 常见 MoE 权重不是每个 expert 一个独立 tensor，而是 fused tensor，例如：

```text
model.layers.N.mlp.experts.w13_weight
model.layers.N.mlp.experts.w2_weight
```

这类 tensor 的第 0 维通常是 expert 维度。Stage E 原始 weight map 对它们只能记录一个完整 tensor，`expert_id` 可能为空。

Stage H 的 planner 会在后处理中把这类 fused tensor 按第 0 维拆成逻辑 expert slice：

```text
logical_fused_expert_slice:
  layer_id = N
  expert_id = slice index
  role = moe_gate_up / moe_down / ...
  start/end = tensor range 中按 expert 均分的逻辑地址范围
```

这一步不改变 tensor，也不改变真实分配，只是为了让 routing heat 可以落到 expert 粒度。

## 4. 新增脚本

### 4.1 `workloads/vllm/plan_stage_h_weight_expert_actions.py`

该脚本读取 Stage E 产物并输出 Stage H 计划。

核心功能：

1. 读取 weight map JSONL。
2. 过滤目标 role：
   - 默认 `moe_gate_up,moe_gate,moe_up,moe_down`
3. 展开 concrete expert tensor 或 fused expert logical slice。
4. 读取 MoE routing trace。
5. 按 `(layer_id, expert_id)` 聚合 routing tokens。
6. 可选读取 fault address log，并按地址范围 join 到 expert slice。
7. 输出 expert heat ranking。
8. 输出 trace-only `prefetch_plan` 和 `offload_plan`。

示例：

```bash
cd workloads/vllm
python3 plan_stage_h_weight_expert_actions.py \
  --weight-map /tmp/run/vllm_uvm_weight_regions_stage_h.jsonl \
  --moe-routing-trace /tmp/run/vllm_uvm_moe_routing_stage_h.jsonl \
  --fault-log /tmp/run/uvm_kv_fault_addrs_stage_h.log \
  --plan-json /tmp/run/vllm_stage_h_weight_expert_plan.json \
  --summary-json /tmp/run/vllm_stage_h_weight_expert_plan_summary.json
```

### 4.2 `workloads/vllm/check_stage_h_success.py`

该脚本是一键验收入口。

默认行为：

1. 启动单进程 vLLM server。
2. 开启 Stage E weight map。
3. 开启 Stage E MoE routing trace。
4. 开启 per-fault address log。
5. 跑一个小 benchmark。
6. 生成 allocator metrics summary。
7. 调用 Stage H planner 生成 trace-only plan。
8. 验证计划关键字段。

推荐命令：

```bash
cd workloads/vllm
./check_stage_h_success.py
```

也可以复用已有日志：

```bash
./check_stage_h_success.py \
  --skip-run \
  --weight-map-jsonl /tmp/run/vllm_uvm_weight_regions_stage_h.jsonl \
  --moe-routing-jsonl /tmp/run/vllm_uvm_moe_routing_stage_h.jsonl \
  --fault-log /tmp/run/uvm_kv_fault_addrs_stage_h.log
```

## 5. Plan JSON 字段

`vllm_stage_h_weight_expert_plan.json` 包含：

1. `mode`
   - 固定为 `trace_only`。

2. `expert_weight_range_records`
   - 可参与 Stage H 的 expert 权重 range 数量。

3. `logical_fused_expert_records`
   - 从 fused tensor 拆出来的逻辑 expert slice 数量。

4. `concrete_expert_weight_records`
   - weight map 中已经带 concrete `expert_id` 的记录数量。

5. `expert_heat_records`
   - 聚合后的 `(layer_id, expert_id)` 数量。

6. `routing_join_records`
   - routing trace 成功 join 到 expert heat record 的条目数。

7. `weight_fault_join_records`
   - fault address 成功 join 到 expert range 的条目数。

8. `prefetch_plan_records`
   - hot expert prefetch 候选数量。

9. `offload_plan_records`
   - cold expert offload 候选数量。

10. `top_hot_experts`
    - 按 heat score 排名前几的 expert。

11. `coldest_experts`
    - 没有 routing/fault 命中的冷 expert 排名。

每个 action record 都包含：

1. `action`
   - `prefetch_candidate` 或 `offload_candidate`。
2. `mode`
   - `trace_only`。
3. `layer_id`
4. `expert_id`
5. `score`
6. `bytes`
7. `routing_tokens`
8. `fault_count`
9. `unique_fault_pages`
10. `roles`
11. `ranges`

## 6. 成功标准

`check_stage_h_success.py` 的 PASS 标准包括：

1. `plan_json_present=True`
2. `trace_only_mode=True`
3. `expert_weight_ranges_present=True`
4. `logical_or_concrete_expert_signal_present=True`
5. `expert_heat_records_present=True`
6. `moe_routing_trace_present=True`
7. `routing_join_records_present=True`
8. `prefetch_plan_records_present=True`
9. `offload_plan_records_present=True`
10. `prefetch_plan_within_budget=True`
11. `offload_plan_within_budget=True`
12. `benchmark_no_failed_requests=True`

如果显式传入 `--require-fault-join`，还会要求：

```text
weight_fault_join_records > 0
```

默认不强制 fault join，因为短 benchmark 下 expert weights 可能已驻留或 faults 被其他 pool 主导。

## 7. 与 Stage I 的边界

Stage H 只生成计划，不执行计划。

当前没有实现：

1. `cudaMemPrefetchAsync` 到 GPU。
2. `cudaMemPrefetchAsync` 到 CPU。
3. `cudaMemAdviseSetPreferredLocation`。
4. tensor storage offload。
5. allocator 内部直接迁移 weights。

Stage I 才应该在 model runner / MoE layer 的安全点上小范围执行 hot expert prefetch，并且继续保持 opt-in 和 kill switch。

## 8. 风险与解释口径

1. Fused tensor 的 expert slice 是逻辑 range，基于第 0 维均分。它适合 Stage H 规划，但在 Stage I 执行前仍需确认实际内存 layout 与 kernel 访问方式。
2. Routing trace 会把 `topk_ids` 聚合到 CPU，默认只在验收或实验时开启，避免生产路径额外同步。
3. Cold expert 只表示本次 workload 未命中，不等于全局永远冷。
4. Offload plan 是候选列表，不代表现在可以安全 offload。
5. Dense/shared/router weights 默认不进入 offload/prefetch 候选，因为这些权重通常是公共路径或预测所需路径。

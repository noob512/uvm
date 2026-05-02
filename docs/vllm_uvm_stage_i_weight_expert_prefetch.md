# vLLM UVM Stage I Weights Expert Prefetch 实现说明

本文档记录 Stage I 的最小安全实现：在 Stage H 已经证明 expert hot/cold trace-only plan 可生成后，把 MoE routing 输出接入真实 `cudaMemPrefetchAsync`，对当前 layer 即将访问的 active expert weight slice 做小范围 GPU prefetch。

Stage I 当前只实现 prefetch，不实现 offload。

## 1. 目标

Stage I 要证明：

1. MoE routing 后、expert kernel 前存在可用安全点。
2. 可以从 `topk_ids` 得到当前 layer 的 active experts。
3. 可以把 active expert 映射到本 rank 的 local expert id。
4. 可以对 `w13_weight` / `w2_weight` 的 expert slice 发起 `cudaMemPrefetchAsync`。
5. prefetch 受 per-layer-call 字节预算和 expert 数量预算限制。
6. benchmark failed requests 保持 0。

## 2. 当前实现范围

已实现：

1. 在 `vllm/device_allocator/uvm.py` 新增 `prefetch_range_to_device(ptr, size, device)`。
2. 在 `FusedMoE.select_experts()` 路由完成后调用 `_maybe_prefetch_uvm_expert_weights(topk_ids)`。
3. 对 active expert 的 `w13_weight` 和 `w2_weight` slice 做 trace 或 prefetch。
4. 输出 Stage I JSONL trace。
5. 新增 `check_stage_i_success.py` 做一键验收。

未实现：

1. CPU offload。
2. `cudaMemAdviseSetPreferredLocation` 到 CPU。
3. 使用 Stage H 离线 plan 限定 hot expert 白名单。
4. 跨 layer / 跨 token 预测下一步 expert。
5. PCIe 带宽 coordinator。

## 3. 插入点

插入点在：

```text
vllm/model_executor/layers/fused_moe/layer.py
  FusedMoE.select_experts()
```

顺序为：

```text
router_logits -> topk_ids
  -> Stage E routing trace
  -> Stage I active expert weight prefetch
  -> quant_method.apply / fused expert kernel
```

这个位置的好处是：

1. 已经知道当前 token 会访问哪些 expert。
2. 还没有进入 expert weight GEMM kernel。
3. 不需要 allocator 猜测语义。
4. 不改变 tensor storage。

## 4. 配置项

通过 `run_kv_fault_ratio.sh` 注入：

```text
VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0|1
VLLM_UVM_WEIGHT_PREFETCH_MODE=trace_only|prefetch
VLLM_UVM_WEIGHT_PREFETCH_TRACE_FILE=<path>
VLLM_UVM_WEIGHT_PREFETCH_MAX_BYTES_PER_STEP=<n>
VLLM_UVM_WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER=<n>
VLLM_UVM_WEIGHT_PREFETCH_TARGET_ROLES=moe_gate_up,moe_down
VLLM_UVM_WEIGHT_PREFETCH_DEVICE=-1
```

含义：

1. `ENABLE=1` 才启用 Stage I 逻辑。
2. `MODE=trace_only` 只记录候选，不调用 prefetch。
3. `MODE=prefetch` 调用 allocator `.so` 里的 `uvm_prefetch`。
4. `MAX_BYTES_PER_STEP` 限制每次 MoE layer 调用最多 prefetch bytes。
5. `MAX_EXPERTS_PER_LAYER` 限制每次 MoE layer 调用最多处理几个 active experts。
6. `TARGET_ROLES` 当前支持 `moe_gate_up` 和 `moe_down`。
7. `DEVICE=-1` 表示使用当前 CUDA device。

## 5. Trace 字段

Stage I trace 文件默认为：

```text
vllm_uvm_weight_prefetch_stage_i.jsonl
```

每条 action 记录包含：

1. `layer_name`
2. `step`
3. `mode`
4. `action`
   - `trace_prefetch_candidate`
   - `prefetch_issued`
   - `prefetch_skipped`
   - `budget_reject`
   - `step_summary`
5. `role`
6. `local_expert_id`
7. `ptr`
8. `bytes`
9. `device`
10. `success`

`step_summary` 记录包含：

1. `active_local_experts`
2. `issued_records`
3. `issued_bytes`
4. `attempted_bytes`
5. `budget_reject_records`
6. `max_bytes_per_step`

## 6. 验收脚本

推荐命令：

```bash
cd workloads/vllm
./check_stage_i_success.py
```

默认会：

1. 启动单进程 vLLM server。
2. 开启 weight map。
3. 开启 MoE routing trace。
4. 开启 Stage I `mode=prefetch`。
5. 跑 1 条 benchmark 请求。
6. 检查 Stage I prefetch trace。

也可以只验证已有日志：

```bash
./check_stage_i_success.py \
  --skip-run \
  --prefetch-trace-jsonl /tmp/run/vllm_uvm_weight_prefetch_stage_i.jsonl \
  --bench-log /tmp/run/vllm_bench_stage_i.log
```

## 7. 成功标准

`check_stage_i_success.py` PASS 需要：

1. `prefetch_trace_present=True`
2. `prefetch_trace_records_present=True`
3. `prefetch_candidates_or_issued_present=True`
4. `prefetch_issued_present_in_prefetch_mode=True`
5. `prefetch_step_bytes_within_budget=True`
6. `moe_routing_trace_present=True`
7. `weight_map_present=True`
8. `benchmark_no_failed_requests=True`
9. allocator metrics 存在时：
   - `allocator_weight_metrics_present=True`
   - `pool_registry_enabled=True`

## 8. 注意事项

1. 当前 prefetch 是“当前 layer active expert”的即时预取，不是下一层/下一 token 的预测预取。
2. 由于 prefetch 插入点紧挨 expert kernel，收益可能有限，但这是验证安全执行链路的必要第一步。
3. 如果 trace 开销过大，可关闭 MoE routing trace，只保留 Stage I prefetch trace；验收脚本默认两者都开，方便证明链路。
4. `uvm_prefetch` 目前不返回 CUDA error，Stage I 用 trace 里的 `prefetch_issued` 表示调用已发出，并用 failed requests 做安全判断。
5. offload 仍然属于后续阶段，不能根据 Stage I 结果直接默认开启。

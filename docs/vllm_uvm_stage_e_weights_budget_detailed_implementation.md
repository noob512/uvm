# vLLM UVM Stage E Weights Budget / Weight Map / MoE Routing 详细实现说明

本文档说明当前项目中 Stage E 的实现方式、代码修改点、运行链路、日志字段和验收方法。它面向后续接手项目的人，目标是让读者能从文档顺着代码理解：模型权重如何从 UVM managed 大池中被单独识别、如何记录 weights 独立预算、如何生成 weight tensor 语义地址 map，以及如何通过 MoE routing trace 为后续专家级预取/驱逐做准备。

本文不展开原先的 eBPF 部分，只聚焦 vLLM UVM allocator、weights budget、weight semantic map、MoE routing trace、runner/check 脚本和相关文档。

## 1. Stage E 要解决什么问题

Stage D2 已经把 KV cache 的初始化大小纳入独立预算：当 `VLLM_UVM_KV_BUDGET_MODE=enforce` 时，vLLM 会在生成 KV cache config 前减少 KV blocks，最终 allocator telemetry 能验证 KV live/peak 不超过预算。

但 weights 仍然只是 UVM managed memory 大池里的一类对象。没有 Stage E 时，我们只能看到“有一堆 managed allocation”，却无法稳定回答：

1. 哪些 allocation 是模型权重？
2. 权重总 live/peak bytes 是多少？
3. 权重是否超过独立预算？
4. 某个地址范围对应哪个 tensor、哪一层、哪个 expert、哪个 shard？
5. MoE 推理时哪些 expert 热，哪些 expert 冷？

Stage E 的目标不是立刻实现完整 weights eviction/offload/prefetch，而是先建立 weights pool 的观测边界和语义索引。后续做 per-pool 驱逐/预取时，系统才有依据做到：

1. KV 超载时，只处理 KV pool。
2. weights 超载时，只处理 weights pool。
3. runtime scratch 的 device-direct/UVM 活动不混淆 KV/weights 的解释口径。
4. MoE expert 级别的策略可以根据 routing 热度和权重地址 map 做判断。

## 2. 阶段划分

当前项目里 Stage E 可以分成四层：

1. Stage E0：weights allocation 分类和独立 budget telemetry。
2. Stage E1：weights `trace_only/enforce` soft signal，和 Stage D 的 KV budget 参数互相独立。
3. Stage E2：weight tensor semantic address map，把权重地址和 tensor 名称、层号、expert id、role 等语义关联起来。
4. Stage E3：可选 MoE expert routing trace，记录每层每步 expert token counts。

这四层仍然是“观测与语义索引”阶段。当前 Stage E 没有实现：

1. weights 初始化硬预算缩容。
2. weights runtime eviction。
3. weights CPU offload。
4. expert weight predictive prefetch。
5. 基于 fault address 的自动 hot/cold expert 策略执行。

这一点和 Stage D2 不同。D2 已经在 vLLM KV config 层做了初始化预算硬约束；Stage E 目前的 `enforce` 仍是 allocator-side soft signal。

## 3. 为什么 Stage E 不在 allocator 层直接驱逐 weights

allocator 看到的是 pointer、size、device、stream、phase 和启发式 allocation class。它不知道：

1. pointer 对应哪个 PyTorch parameter 名称。
2. 这个 tensor 属于 attention、router、norm 还是 MoE expert。
3. MoE expert 下一步是否会被路由命中。
4. 当前 tensor 是否可安全 offload。
5. offload 后什么时候需要 prefetch 回 GPU。
6. dense/shared 权重是否会影响所有 token。
7. 正在运行的 CUDA kernel 是否还会读该权重。

如果 allocator 在超预算时直接 free/offload/prefetch 权重，很容易破坏模型执行，或者引入不可控的同步与 page fault。因此当前 Stage E 只在 allocator 层产生 telemetry 和 soft signal，真正的 weights 策略要交给 model loader、weight map、MoE router trace 和 runtime scheduler 后续共同实现。

## 4. 主要修改文件

### 4.1 `workloads/vllm/vllm/uvm_test/uvm_allocator.cpp`

这是 Stage E0/E1 的核心实现文件，负责识别 weights allocation、维护 weights live/peak 计数、输出 budget 字段，并在 free 时归还 weights live bytes。

Stage E 相关修改包括：

1. 新增配置项：
   - `weight_budget_bytes`
   - `weight_budget_mode`

2. 新增 weights telemetry counters：
   - `weight_trace_allocs`
   - `weight_requested_bytes`
   - `weight_live_bytes`
   - `weight_peak_live_bytes`
   - `weight_budget_over_allocs`
   - `weight_budget_reject_allocs`
   - `weight_free_success_allocs`

3. 新增/扩展 weights budget helper：
   - `normalize_weight_budget_mode()`
   - `is_weight_allocation()`
   - `update_weight_peak_live()`
   - `weight_budget_remaining_snapshot()`
   - `record_weight_allocation()`
   - `release_weight_budget()`

4. 扩展 `AllocationInfo`：
   - `weight_budget_tracked`
   - `weight_budget_over_budget`
   - `weight_budget_reason`

5. 扩展 trace log：
   - `TRACE_POLICY`
   - `TRACE_GAP_WATCH_ALLOC`
   - `TRACE_GAP_WATCH_FREE`
   - Session Summary

6. 扩展 reset/close summary：
   - `uvm_reset_all_stats()` 会清理 weights counters。
   - `uvm_close_log()` 会输出 weights budget summary。

### 4.2 `workloads/vllm/run_kv_fault_ratio.sh`

这是 Stage E 的底层 runner。虽然名字仍是 `run_kv_fault_ratio.sh`，但现在已经承载 Stage C/D/E 的统一实验入口。

Stage E 相关修改包括：

1. 新增 runner 变量：
   - `UVM_WEIGHT_BUDGET_BYTES`
   - `UVM_WEIGHT_BUDGET_MODE`
   - `UVM_WEIGHT_MAP_ENABLE`
   - `UVM_WEIGHT_MAP_FILE`
   - `UVM_MOE_ROUTING_TRACE_ENABLE`
   - `UVM_MOE_ROUTING_TRACE_FILE`

2. 新增命令行参数：
   - `--uvm-weight-budget-bytes <n>`
   - `--uvm-weight-budget-mode trace_only|enforce`
   - `--uvm-weight-map-enable <0|1>`
   - `--uvm-weight-map-file <path>`
   - `--uvm-moe-routing-trace-enable <0|1>`
   - `--uvm-moe-routing-trace-file <path>`

3. 参数校验：
   - weight budget bytes 必须是非负整数。
   - weight budget mode 必须是 `trace_only` 或 `enforce`。
   - weight map enable 必须是 0 或 1。
   - MoE routing trace enable 必须是 0 或 1。

4. server 环境变量注入：
   - `VLLM_UVM_WEIGHT_BUDGET_BYTES`
   - `VLLM_UVM_WEIGHT_BUDGET_MODE`
   - `VLLM_UVM_WEIGHT_MAP_ENABLE`
   - `VLLM_UVM_WEIGHT_MAP_FILE`
   - `VLLM_UVM_MOE_ROUTING_TRACE_ENABLE`
   - `VLLM_UVM_MOE_ROUTING_TRACE_FILE`

5. Stage E checker 默认把 `--uvm-kv-budget-bytes` 设为 0，避免 Stage D/D2 的 KV budget enforcement 影响 weights 检查。

### 4.3 `workloads/vllm/summarize_gap_watch_metrics.py`

该脚本从 allocator trace log 中聚合 Stage C/D/E metrics。Stage E 相关修改包括：

1. 新增 `record_weight_budget_fields()`。
2. 从 `TRACE_POLICY` 和 gap-watch alloc/free 记录中解析 weight budget fields。
3. 输出 weights summary 字段：
   - `weight_budget_bytes`
   - `weight_budget_mode`
   - `weight_trace_allocations`
   - `weight_requested_bytes`
   - `weight_live_bytes`
   - `weight_peak_live_bytes_observed`
   - `weight_min_budget_remaining_observed`
   - `weight_budget_over_records`
   - `weight_budget_reject_records`
   - `weight_budget_reason_counts`

### 4.4 `workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py`

这是 Stage E2 weight semantic map 的核心实现文件。

相关修改包括：

1. 在 `load_model()` 中用 `uvm_allocation_phase("load_model")` 包住模型加载，使 allocator 能把模型加载期 allocation 分类为 `weight_persistent`。

2. 新增 weight map 开关：
   - `_is_uvm_weight_map_enabled()`
   - `_uvm_weight_map_file()`
   - `_prepare_uvm_weight_map_log()`

3. 新增权重语义标签函数：
   - `_uvm_weight_semantic_tags(name)`

4. 新增 weight tensor 地址收集：
   - `_collect_weight_address_rows()`

5. 新增 JSONL sidecar 写入：
   - `_append_uvm_weight_map_log(phase, records)`

6. 在 `_log_model_weight_addresses("load_model")` 时同时写两类日志：
   - 旧的 CSV-style address log。
   - Stage E2 semantic JSONL sidecar。

### 4.5 `workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py`

这是 Stage E3 MoE routing trace 的核心实现文件。

相关修改包括：

1. 在 MoE layer 初始化时读取：
   - `VLLM_UVM_MOE_ROUTING_TRACE_ENABLE`
   - `VLLM_UVM_MOE_ROUTING_TRACE_FILE`

2. 新增 per-layer step counter：
   - `_uvm_moe_trace_step`

3. 在 top-k routing 结果生成后调用：
   - `_log_uvm_moe_routing_trace(...)`

4. `_log_uvm_moe_routing_trace()` 会聚合并写出：
   - `layer_name`
   - `step`
   - `num_tokens`
   - `top_k`
   - `global_num_experts`
   - `logical_num_experts`
   - `local_num_experts`
   - `topk_shape`
   - `router_logits_shape`
   - `expert_token_counts`
   - `active_experts`
   - `topk_weight_sum`
   - `enable_eplb`

这个 trace 默认关闭，因为它会把 tensor 数据 detach/cpu 聚合，带来额外同步和 CPU 开销。只有在 Stage E3 实验中显式打开。

### 4.6 `workloads/vllm/summarize_stage_e_weight_map.py`

这是 Stage E2/E3 的 summary 脚本。

它读取：

1. weight semantic map JSONL。
2. 可选 MoE routing trace JSONL。

输出字段包括：

1. `weight_map_records`
2. `weight_map_total_bytes`
3. `weight_map_moe_expert_records`
4. `weight_map_moe_expert_bytes`
5. `weight_map_layer_count`
6. `weight_map_expert_count`
7. `weight_map_kind_counts`
8. `weight_map_role_counts`
9. `weight_map_dtype_counts`
10. `weight_map_top_layers`
11. `weight_map_top_experts`
12. `moe_routing_records`
13. `moe_routing_total_tokens`
14. `moe_routing_layer_count`
15. `moe_routing_active_expert_count`
16. `moe_routing_top_layers`
17. `moe_routing_top_experts`

### 4.7 `workloads/vllm/check_stage_e_success.py`

这是 Stage E 的一键验收脚本。

默认行为：

1. 启动 vLLM server。
2. 等待 model load 和 KV range ready。
3. 默认不跑 benchmark。
4. 设置 weight budget 为 1 MiB、mode 为 `trace_only`。
5. 设置 KV budget 为 0，避免混入 Stage D。
6. 强制开启 weight map。
7. 默认关闭 MoE routing trace。
8. 汇总 allocator metrics 和 weight map summary。
9. 输出 `stage_e_success_check.json`。

支持参数：

1. `--budget-bytes`
2. `--budget-mode`
3. `--kv-budget-bytes`
4. `--kv-budget-mode`
5. `--run-bench`
6. `--enable-moe-routing-trace`
7. `--require-moe-routing-trace`
8. `--weight-map-jsonl`
9. `--moe-routing-jsonl`
10. `--weight-map-summary-json`

### 4.8 `workloads/vllm/run_stage_e_weights_budget_check.sh`

这是 Stage E 的 convenience wrapper。它把环境变量转成 `check_stage_e_success.py` 参数：

```bash
WEIGHT_BUDGET_BYTES="${WEIGHT_BUDGET_BYTES:-1048576}"
WEIGHT_BUDGET_MODE="${WEIGHT_BUDGET_MODE:-trace_only}"
KV_BUDGET_BYTES="${KV_BUDGET_BYTES:-0}"
KV_BUDGET_MODE="${KV_BUDGET_MODE:-trace_only}"
RUN_BENCH="${RUN_BENCH:-0}"
ENABLE_MOE_ROUTING_TRACE="${ENABLE_MOE_ROUTING_TRACE:-0}"
REQUIRE_MOE_ROUTING_TRACE="${REQUIRE_MOE_ROUTING_TRACE:-0}"
```

### 4.9 `docs/vllm_uvm_stage_e_weights_budget_telemetry_implementation.md`

这是已有 Stage E 摘要文档，记录了 Stage E 的目标、参数、allocator telemetry、weight map 和 MoE routing trace。本文档在它基础上进一步按读代码路径展开。

### 4.10 `docs/vllm_uvm_memory_pool_evolution_plan.md`

整体演进计划中记录了 Stage E0/E1/E2/E3 的当前落地状态、成功标准和后续方向。

## 5. E0/E1 allocator 运行流程

下面按一次 weights allocation 的执行顺序解释。

### 5.1 设置 load_model phase

在 `gpu_model_runner.py::load_model()` 中，模型加载被包在：

```python
with uvm_allocation_phase("load_model"):
    self.model = model_loader.load_model(...)
```

allocator 的分类逻辑会把 phase 为 `load_model` 的 allocation 识别为：

```cpp
AllocationClass::WeightPersistent
```

这就是 Stage E 能在 allocator 层区分 weights 和 KV/runtime scratch 的基础。

### 5.2 读取 weights budget 配置

allocator 初始化时读取：

```text
VLLM_UVM_WEIGHT_BUDGET_BYTES
VLLM_UVM_WEIGHT_BUDGET_MODE
```

`VLLM_UVM_WEIGHT_BUDGET_BYTES=0` 表示不限额。

`VLLM_UVM_WEIGHT_BUDGET_MODE` 通过 `normalize_weight_budget_mode()` 规范化，只有 `enforce` 会保留为 enforce，其余都回退到 `trace_only`。

### 5.3 识别 weight allocation

allocator 使用：

```cpp
static bool is_weight_allocation(AllocationClass alloc_class) {
    return alloc_class == AllocationClass::WeightPersistent;
}
```

在 `uvm_malloc()` 中：

```cpp
bool weight_budget_tracked = is_weight_allocation(alloc_class);
```

如果为 true，就进入 Stage E 的 weights budget 记账路径。

### 5.4 记录 allocation

`uvm_malloc()` 中调用：

```cpp
record_weight_allocation(size, &weight_budget_over_budget, &weight_budget_reason);
```

它做几件事：

1. `weight_trace_allocs += 1`
2. `weight_requested_bytes += size`
3. `weight_live_bytes += size`
4. 更新 `weight_peak_live_bytes`
5. 判断 `weight_budget_bytes > 0 && weight_live_bytes > weight_budget_bytes`
6. 如果超预算，`weight_budget_over_allocs += 1`
7. 如果超预算且 `weight_budget_mode == "enforce"`，`weight_budget_reject_allocs += 1`

注意：Stage E 的 reject 是 soft reject signal，不是 allocator 返回 NULL。

### 5.5 写入 reason

`record_weight_allocation()` 会给每次 weights allocation 写入 reason：

1. `weight_budget_unlimited`：budget bytes 为 0，不限额。
2. `weight_budget_within_budget`：budget 非 0，当前 weight live bytes 未超过预算。
3. `weight_budget_exceeded_trace_only`：超预算，但 mode 是 `trace_only`。
4. `weight_budget_exceeded_soft_enforce`：超预算，mode 是 `enforce`，allocator 产生软拒绝信号。
5. `not_weight`：不是 weights allocation。

### 5.6 保存 metadata

weights allocation 会被放入 `active_allocations`，因为 `store_active_info` 包含：

```cpp
weight_budget_tracked
```

对应 metadata 包括：

1. `weight_budget_tracked`
2. `weight_budget_over_budget`
3. `weight_budget_reason`

这样 free 时可以根据 metadata 判断这个 pointer 是否属于 weights pool，并归还 `weight_live_bytes`。

### 5.7 trace 输出

`TRACE_POLICY` 中会输出：

```text
weight_budget_tracked=<0|1>
weight_budget_over_budget=<0|1>
weight_budget_reason=<reason>
weight_live_bytes=<n>
weight_budget_bytes=<n>
weight_budget_remaining=<n>
weight_budget_mode=<trace_only|enforce>
```

`TRACE_GAP_WATCH_ALLOC` 和 `TRACE_GAP_WATCH_FREE` 也带有 weight budget 字段，便于未来把 fault/gap-watch 和 weight semantic map 统一分析。

### 5.8 free 时归还 weight live bytes

在 `uvm_free()` 中：

```cpp
if (has_info && info.weight_budget_tracked && err == cudaSuccess) {
    release_weight_budget(info.size);
    weight_free_success_allocs.fetch_add(1);
}
```

模型权重通常在 server 生命周期内长期存在，所以 no-bench 检查时 `weight_live_bytes` 常常接近 `weight_peak_live_bytes_observed`。

### 5.9 Session Summary

正常退出时 `uvm_close_log()` 会输出：

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

Stage E success check 主要依赖 `summarize_gap_watch_metrics.py` 聚合的 JSON；Session Summary 是辅助证据。

## 6. E2 weight semantic map 执行流程

allocator 只能知道 allocation 是 `weight_persistent`，不知道它对应哪个 tensor。因此 Stage E2 在 vLLM Python 层补充语义地图。

### 6.1 开关和输出文件

`gpu_model_runner.py` 读取：

```text
VLLM_UVM_WEIGHT_MAP_ENABLE
VLLM_UVM_WEIGHT_MAP_FILE
```

默认情况下，如果 UVM address logging 启用，weight map 也启用。Stage E checker 会显式传入：

```bash
--uvm-weight-map-enable 1
--uvm-weight-map-file <run_dir>/vllm_uvm_weight_regions_stage_e.jsonl
```

### 6.2 清理旧文件

`_prepare_uvm_weight_map_log()` 会在 global first rank 上清空旧 JSONL，然后在 distributed 初始化时做 barrier，避免多 rank 混用旧文件。

### 6.3 遍历模型参数和 buffer

`_collect_weight_address_rows()` 会遍历：

1. `model.named_parameters(recurse=True)`
2. `model.named_buffers(recurse=True)`

对每个 CUDA tensor 记录：

1. `data_ptr()`
2. `numel() * element_size()`
3. start/end 地址
4. dtype
5. shape
6. device

它会用 `_logged_weight_ptrs` 去重，避免 weight tying 或多个名字引用同一块内存时重复统计。

### 6.4 生成语义标签

`_uvm_weight_semantic_tags(name)` 根据 tensor name 推断：

1. `layer_id`
2. `expert_id`
3. `role`
4. `shard_id`
5. `is_moe_expert`

典型 role 包括：

1. `embedding`
2. `attention`
3. `moe_gate`
4. `moe_up`
5. `moe_gate_up`
6. `moe_down`
7. `moe_router`
8. `mlp`
9. `norm`
10. `other`

对于 Qwen3 MoE 这类模型，常见的 expert 权重会被标成 `moe_gate_up` 和 `moe_down`，router 权重会被标成 `moe_router`。

### 6.5 写 JSONL sidecar

`_append_uvm_weight_map_log()` 会为每条记录补充：

1. timestamp
2. phase
3. pid
4. model

然后写入 JSONL。每一行是一个权重 tensor 的语义地址记录。

典型字段：

```json
{
  "kind": "weight:param",
  "name": "...",
  "start": "0x...",
  "end": "0x...",
  "start_int": 123,
  "end_int": 456,
  "size_bytes": 789,
  "dtype": "torch.float8_e4m3fn",
  "shape": [...],
  "device": "cuda:0",
  "layer_id": 0,
  "expert_id": null,
  "role": "attention",
  "shard_id": null,
  "is_moe_expert": false,
  "phase": "load_model",
  "pid": 12345,
  "model": "Qwen/Qwen3-30B-A3B-FP8"
}
```

### 6.6 与旧 address log 的关系

`_log_model_weight_addresses("load_model")` 同时写：

1. 旧 CSV address log，便于沿用已有地址分析工具。
2. Stage E2 JSONL semantic map，便于后续按 layer/expert/role 分析。

JSONL 是 Stage E 之后更重要的语义 sidecar。

## 7. E3 MoE routing trace 执行流程

E2 解决“某个地址属于哪个权重 tensor”。E3 进一步回答“当前请求实际访问哪些 expert”。

### 7.1 开关和输出文件

`fused_moe/layer.py` 读取：

```text
VLLM_UVM_MOE_ROUTING_TRACE_ENABLE
VLLM_UVM_MOE_ROUTING_TRACE_FILE
```

默认关闭。Stage E checker 只有在传入：

```bash
--run-bench --enable-moe-routing-trace --require-moe-routing-trace
```

时才会要求 routing trace 必须存在且有记录。

### 7.2 trace 位置

MoE layer 在 top-k routing 得到 `topk_ids/topk_weights` 后调用：

```python
self._log_uvm_moe_routing_trace(...)
```

这意味着 trace 记录的是 router 选择出的专家分布，而不是后续 kernel 的底层访存。

### 7.3 trace 内容

`_log_uvm_moe_routing_trace()` 会：

1. detach `topk_ids`
2. 过滤 invalid expert id
3. 用 `torch.bincount()` 统计 expert token counts
4. 把结果搬到 CPU
5. 写 JSONL

每条记录包括：

1. `timestamp`
2. `pid`
3. `layer_name`
4. `step`
5. `num_tokens`
6. `top_k`
7. `global_num_experts`
8. `logical_num_experts`
9. `local_num_experts`
10. `topk_shape`
11. `router_logits_shape`
12. `expert_token_counts`
13. `active_experts`
14. `topk_weight_sum`
15. `enable_eplb`

### 7.4 为什么默认关闭

MoE routing trace 会引入额外 CPU 同步和 JSONL 写入。对于性能实验，它可能影响 TPOT/ITL。因此 Stage E 默认只检查 weight budget 和 weight map；只有专门验证 E3 时才开启 routing trace。

## 8. 汇总脚本如何工作

### 8.1 allocator metrics summary

`summarize_gap_watch_metrics.py` 聚合 allocator trace，输出：

1. `weight_budget_bytes`
2. `weight_budget_mode`
3. `weight_trace_allocations`
4. `weight_requested_bytes`
5. `weight_live_bytes`
6. `weight_peak_live_bytes_observed`
7. `weight_min_budget_remaining_observed`
8. `weight_budget_over_records`
9. `weight_budget_reject_records`
10. `weight_budget_reason_counts`

### 8.2 weight map summary

`summarize_stage_e_weight_map.py` 聚合 JSONL semantic map，输出：

1. record 数和总 bytes。
2. MoE expert record 数和 bytes。
3. layer/expert 数量。
4. kind/role/dtype 分布。
5. top layers/top experts。

### 8.3 MoE routing summary

如果提供 routing trace，`summarize_stage_e_weight_map.py` 同时输出：

1. `moe_routing_records`
2. `moe_routing_total_tokens`
3. `moe_routing_layer_count`
4. `moe_routing_active_expert_count`
5. `moe_routing_top_layers`
6. `moe_routing_top_experts`

这让一次 Stage E 实验同时具备“权重在哪里”和“哪些 expert 被访问”的两类证据。

## 9. 参数和环境变量

### 9.1 runner 参数

```bash
--uvm-weight-budget-bytes <n>
--uvm-weight-budget-mode trace_only|enforce
--uvm-weight-map-enable <0|1>
--uvm-weight-map-file <path>
--uvm-moe-routing-trace-enable <0|1>
--uvm-moe-routing-trace-file <path>
```

### 9.2 环境变量

```bash
VLLM_UVM_WEIGHT_BUDGET_BYTES=<bytes>
VLLM_UVM_WEIGHT_BUDGET_MODE=trace_only|enforce
VLLM_UVM_WEIGHT_MAP_ENABLE=0|1
VLLM_UVM_WEIGHT_MAP_FILE=<jsonl path>
VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0|1
VLLM_UVM_MOE_ROUTING_TRACE_FILE=<jsonl path>
```

### 9.3 推荐实验配置

Stage E no-bench telemetry：

```bash
--uvm-weight-budget-bytes 1048576
--uvm-weight-budget-mode trace_only
--uvm-kv-budget-bytes 0
--uvm-weight-map-enable 1
--no-bench
```

Stage E + MoE routing trace：

```bash
--uvm-weight-budget-bytes 1048576
--uvm-weight-budget-mode trace_only
--uvm-kv-budget-bytes 0
--uvm-weight-map-enable 1
--uvm-moe-routing-trace-enable 1
--run-bench
```

## 10. Metrics 字段如何理解

### 10.1 `weight_budget_bytes`

配置的 weights 独立预算。0 表示不限额。

### 10.2 `weight_budget_mode`

当前模式：

1. `trace_only`：只观测，不产生 soft reject。
2. `enforce`：当前 Stage E 中只产生 allocator-side soft reject signal，不做硬失败、不做 offload。

### 10.3 `weight_trace_allocations`

被识别为模型权重的 allocation 数。成功实现 Stage E 时应大于 0。

### 10.4 `weight_live_bytes`

当前仍存活的 weights bytes。模型权重通常长期存在，所以 no-bench 检查中该值通常接近 peak。

### 10.5 `weight_peak_live_bytes_observed`

本次运行中观察到的 weights live bytes 高水位。

### 10.6 `weight_budget_over_records`

weights allocation 后发现 live bytes 超过预算的记录数。

默认 1 MiB budget 对大模型权重来说极小，所以 no-bench trace-only 检查中该值通常大于 0。

### 10.7 `weight_budget_reject_records`

allocator-side soft reject 记录数。它统计 reason 为 `weight_budget_exceeded_soft_enforce` 的记录。

trace-only 模式下应为 0。

### 10.8 `weight_budget_reason_counts`

reason 分布，用来判断当前实验状态：

1. `weight_budget_unlimited`：budget 为 0。
2. `weight_budget_within_budget`：weights allocation 在预算内。
3. `weight_budget_exceeded_trace_only`：trace-only 下超预算。
4. `weight_budget_exceeded_soft_enforce`：enforce 下 allocator 观察到超预算。

### 10.9 `weight_map_records`

weight semantic map 中的记录数。应大于 0。它统计的是 tensor 语义记录，不一定等于 allocator allocation 数，因为一个 allocation 可能对应多个 tensor，或者多个 tensor 共享同一块 pointer 被去重。

### 10.10 `weight_map_moe_expert_records`

被语义标签识别为 MoE expert weight 的记录数。对于 MoE 模型应大于 0；对于 dense 模型可以为 0，但字段应存在。

### 10.11 `weight_map_role_counts`

权重 role 分布，例如：

```text
{'attention': 289, 'embedding': 1, 'moe_down': 96, 'moe_gate_up': 96, 'moe_router': 48, 'norm': 97, 'other': 1}
```

这是判断 E2 语义标签是否正常的关键字段。

### 10.12 `moe_routing_records`

MoE routing trace 的记录数。只有开启 routing trace 并运行 benchmark 时才应大于 0。

### 10.13 `moe_routing_top_experts`

按 token count 聚合后的热门 expert 列表。后续 expert prefetch/offload 可以用它作为热度输入之一。

## 11. 如何运行验收

### 11.1 Stage E 默认检查

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./check_stage_e_success.py
```

默认行为：

1. weight budget 为 1 MiB。
2. weight mode 为 `trace_only`。
3. KV budget 为 0。
4. 开启 weight map。
5. 不跑 benchmark。
6. 不要求 MoE routing trace。

预期结果：

1. `Stage E Success Check: PASS`
2. `weight_trace_allocations > 0`
3. `weight_peak_live_bytes_observed > 0`
4. `weight_budget_bytes == 1048576`
5. `weight_budget_mode == trace_only`
6. 因为 1 MiB 小于实际模型权重，`weight_budget_over_records > 0`
7. 因为是 trace-only，`weight_budget_reject_records == 0`
8. `weight_map_records > 0`

### 11.2 Stage E + MoE routing trace 检查

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./check_stage_e_success.py \
  --run-bench \
  --enable-moe-routing-trace \
  --require-moe-routing-trace
```

预期结果：

1. 默认 Stage E checks 全部通过。
2. benchmark successful requests > 0。
3. `moe_routing_records > 0`。
4. `moe_routing_active_expert_count > 0`。
5. `moe_routing_top_experts` 有内容。

### 11.3 wrapper 方式

默认 no-bench：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./run_stage_e_weights_budget_check.sh
```

带 benchmark 和 MoE routing trace：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
RUN_BENCH=1 \
ENABLE_MOE_ROUTING_TRACE=1 \
REQUIRE_MOE_ROUTING_TRACE=1 \
./run_stage_e_weights_budget_check.sh
```

## 12. 结合已有实验结果解读

用户之前给出的 Stage E no-bench 检查中：

```text
weight_budget_bytes=1048576
weight_budget_mode=trace_only
weight_trace_allocations=1314
weight_live_bytes=31185032525
weight_peak_live_bytes_observed=31185032901
weight_budget_over_records=1314
weight_budget_reject_records=0
weight_budget_reason_counts={'weight_budget_exceeded_trace_only': 1314}
Stage E Success Check: PASS
```

这说明：

1. allocator 成功识别到 1314 次模型权重 allocation。
2. 模型权重 live/peak 约 31.18 GB。
3. 配置预算只有 1 MiB，所以权重 allocation 全部触发 over-budget。
4. mode 是 `trace_only`，所以 soft reject 为 0。
5. 这证明 E0/E1 的 weights budget telemetry 和 reason 链路已经接通。

用户后续给出的 Stage E + benchmark + MoE routing trace 检查中：

```text
weight_map_records=628
weight_map_total_bytes=31185031168
weight_map_moe_expert_records=192
weight_map_moe_expert_bytes=28998107136
weight_map_layer_count=48
weight_map_role_counts={'attention': 289, 'embedding': 1, 'moe_down': 96, 'moe_gate_up': 96, 'moe_router': 48, 'norm': 97, 'other': 1}
moe_routing_records=49248
moe_routing_total_tokens=183168
moe_routing_layer_count=48
moe_routing_active_expert_count=128
moe_routing_top_experts=[('52', 23509), ('110', 22673), ...]
Stage E Success Check: PASS
```

这说明：

1. E2 weight semantic map 生成成功，记录了 628 个去重后的权重 tensor/buffer。
2. weight map 总 bytes 约 31.18 GB，和 allocator 观察到的 weight live/peak 基本一致。
3. 识别到 192 条 MoE expert weight 记录，expert 权重约 28.99 GB，说明该模型权重主体是 MoE experts。
4. layer count 为 48，与模型层数一致。
5. role counts 能区分 attention、moe_gate_up、moe_down、moe_router、norm 等权重角色。
6. E3 routing trace 生成成功，49248 条 routing record 覆盖 48 层。
7. active expert count 为 128，说明 benchmark 期间所有或大量 expert 都被路由触达。
8. top experts 列表给出了最热 expert，为后续 expert hot/cold 分类提供输入。

这组实验说明 Stage E 已经补全到“weights budget telemetry + weight semantic map + MoE routing heat trace”。但它仍不表示已经实现 weights eviction/offload/prefetch。

## 13. 成功标准

### 13.1 E0/E1 成功标准

1. metrics JSON 存在。
2. `weight_trace_allocations > 0`。
3. `weight_peak_live_bytes_observed > 0`。
4. `weight_live_bytes >= 0`。
5. `weight_budget_bytes` 与配置一致。
6. `weight_budget_mode` 与配置一致。
7. 如果预算非 0 且 peak 超预算，则 `weight_budget_over_records > 0`。
8. trace-only 模式下 `weight_budget_reject_records == 0`。
9. enforce 模式下如果超预算，则 `weight_budget_reject_records > 0`，但这仍是 soft signal。

### 13.2 E2 成功标准

1. weight map JSONL 存在。
2. `weight_map_records > 0`。
3. `weight_map_total_bytes > 0`。
4. role counts 字段存在。
5. 对 MoE 模型，`weight_map_moe_expert_records > 0`。
6. layer count 与模型结构大体匹配。

### 13.3 E3 成功标准

1. 必须开启 `--run-bench`。
2. 必须开启 `--enable-moe-routing-trace`。
3. 如果指定 `--require-moe-routing-trace`，则 `moe_routing_records > 0`。
4. `moe_routing_total_tokens > 0`。
5. `moe_routing_active_expert_count > 0`。
6. `moe_routing_top_experts` 有内容。

## 14. 常见失败模式和排查

### 14.1 `weight_trace_allocations` 为 0

可能原因：

1. UVM allocator 没启用。
2. allocator log 没正确传入。
3. vLLM 没跑到 model load。
4. `load_model` phase 没设置或丢失。
5. summary 解析了错误/旧的 allocator log。

排查：

1. 搜索 allocator log 中是否有 `phase=load_model`。
2. 搜索 `predicted_class=weight_persistent`。
3. 检查 server log 是否完成模型加载。
4. 检查 `VLLM_USE_UVM=1` 是否传入 server 进程。

### 14.2 trace-only 下没有 over-budget

默认 1 MiB budget 对大模型权重应稳定 over-budget。如果没有：

1. 检查 `weight_budget_bytes` 是否真是 1048576。
2. 检查 `weight_peak_live_bytes_observed` 是否大于 budget。
3. 检查 reason 是否是 `weight_budget_unlimited`。
4. 确认 metrics JSON 不是旧文件。

### 14.3 weight map 为空

可能原因：

1. `VLLM_UVM_WEIGHT_MAP_ENABLE=0`。
2. `VLLM_UVM_WEIGHT_MAP_FILE` 路径不可写。
3. `_log_model_weight_addresses("load_model")` 没执行。
4. 模型权重不在 CUDA 上。
5. 多进程/分布式同步导致只有非 first rank 写日志。

排查：

1. 检查 run dir 下 `vllm_uvm_weight_regions_stage_e.jsonl` 是否存在。
2. 检查 server log 是否有 weight map 写入 warning。
3. 检查 address log 是否有 weight rows。

### 14.4 MoE routing trace 为空

可能原因：

1. 没有 `--run-bench`，只启动 server 不会触发 routing。
2. 没有 `--enable-moe-routing-trace`。
3. 当前模型不是 MoE 模型，或执行路径没有进入 `FusedMoE`。
4. benchmark 请求失败或没有生成 token。
5. routing trace 文件路径不可写。

排查：

1. 检查 bench log 中 successful requests。
2. 检查 `VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=1` 是否传入。
3. 检查 `vllm_uvm_moe_routing_stage_e.jsonl` 是否存在。
4. 检查 server log 是否有 routing trace warning。

### 14.5 Stage E benchmark 变慢

开启 MoE routing trace 会把 routing tensor 统计搬到 CPU 并写 JSONL，可能明显拖慢 benchmark。用于性能对比时应关闭 E3 routing trace，只保留必要 telemetry。

## 15. Stage E 与 Stage C/D 的关系

Stage C、D、E 分别建立三个 pool 的边界：

1. Stage C/C2：runtime scratch pool，已有 opt-in `device_direct` backend 和总预算。
2. Stage D/D2：KV pool，已有初始化预算硬约束和 allocator telemetry。
3. Stage E/E3：weights pool，已有 allocator telemetry、weight semantic map 和 MoE routing trace。

这三套机制相互独立：

1. Stage E 不会改变 Stage C 的 device-direct backend。
2. Stage E checker 默认把 KV budget 设为 0，避免混入 Stage D。
3. Stage E 不会把 weights 改成 `device_direct`。
4. Stage E 不会执行 weights offload/eviction/prefetch。

后续真正要实现“kvcache、weights、临时缓冲区分区管理”，需要在这三个边界之上增加各自的策略执行器。

## 16. 后续建议

更稳妥的下一步不是直接 offload 所有 weights，而是：

1. 将 UVM fault address 与 weight semantic map 自动 join。
2. 将 MoE routing trace 与 expert weight map join。
3. 生成 hot/cold expert 列表。
4. 先对冷 expert 做 trace-only offload/prefetch 计划。
5. 再实现 opt-in expert-only prefetch/offload。
6. 最后引入 weights pool admission control，保证 weights 策略不挤占 KV pool 和 runtime scratch pool。

这样可以延续当前项目的演进原则：先观测、再语义化、再小范围执行，所有策略都带可回退路径。

## 17. 修改点速查表

| 文件 | Stage E 作用 |
| --- | --- |
| `workloads/vllm/vllm/uvm_test/uvm_allocator.cpp` | E0/E1 核心：weights allocation 识别、live/peak 计数、budget reason、soft enforce signal、free 回收 |
| `workloads/vllm/run_kv_fault_ratio.sh` | 注入 Stage E 参数，启动 vLLM，生成 allocator trace、weight map 和 MoE routing trace |
| `workloads/vllm/summarize_gap_watch_metrics.py` | 从 allocator trace 聚合 weights budget metrics |
| `workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py` | E2 核心：load_model phase、weight tensor 地址收集、semantic JSONL sidecar |
| `workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py` | E3 核心：MoE top-k routing trace |
| `workloads/vllm/summarize_stage_e_weight_map.py` | 聚合 weight map 和 MoE routing trace |
| `workloads/vllm/check_stage_e_success.py` | Stage E 一键验收 |
| `workloads/vllm/run_stage_e_weights_budget_check.sh` | Stage E convenience wrapper |
| `docs/vllm_uvm_stage_e_weights_budget_telemetry_implementation.md` | 既有 Stage E 摘要文档 |
| `docs/vllm_uvm_memory_pool_evolution_plan.md` | 记录 Stage E 在整体 memory pool 演进中的位置 |

## 18. 最小阅读路径

如果只想快速理解 Stage E，建议按下面顺序读：

1. 本文档第 1 到 10 节，先建立 E0/E1/E2/E3 的整体模型。
2. `workloads/vllm/check_stage_e_success.py`，理解验收条件。
3. `workloads/vllm/run_kv_fault_ratio.sh`，理解参数如何传给 vLLM server。
4. `workloads/vllm/vllm/uvm_test/uvm_allocator.cpp` 中 `weight_budget_*` 变量、`record_weight_allocation()`、`release_weight_budget()`、`uvm_malloc()`、`uvm_free()`。
5. `workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py` 中 `_collect_weight_address_rows()`、`_uvm_weight_semantic_tags()`、`_append_uvm_weight_map_log()`、`_log_model_weight_addresses()`。
6. `workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py` 中 `_log_uvm_moe_routing_trace()`。
7. `workloads/vllm/summarize_stage_e_weight_map.py`，理解 weight map 和 routing trace 如何被聚合。

读完这条路径，就能判断一次 Stage E 实验是“weights telemetry 未接上”“weight map 未生成”“MoE routing trace 未触发”还是“Stage E 已经完整通过”。

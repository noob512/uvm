# vLLM UVM 当前实现总览

本文档记录当前项目已经实现的 vLLM + UVM 内存管理能力、主要代码修改、运行方式、验收脚本和边界。它是一个“当前实现快照”，用于回答：

1. 项目现在到底实现了哪些 UVM 能力。
2. 每个阶段修改了哪些模块。
3. 哪些功能只是 telemetry / trace-only。
4. 哪些功能已经进入真实执行路径。
5. 后续继续 Stage J/K 或更高阶段时应该从哪里接。

当前实现的总体方向是：先把 vLLM 的 UVM allocation 按语义分成 `kv_cache`、`weights`、`runtime_scratch` 等独立 pool，再逐步在安全边界上接入 prefetch、device-direct、admission、prefix-cache eviction 和全局 coordinator。

## 1. 总体架构

当前项目由四层组成：

```text
vLLM Python runtime
  - MoE routing / expert weight action safe point
  - KV block manager / prefix-cache safe point
  - runner / success check scripts

Python UVM wrapper
  - enable custom allocator
  - set allocation phase
  - expose cudaMemPrefetchAsync / cudaMemAdvise helpers
  - Stage K coordinator

C++/CUDA UVM allocator shim
  - cudaMallocManaged default backend
  - optional device-direct backend
  - allocation classification
  - pool registry / per-pool counters
  - trace log and session summary

NVIDIA UVM kernel instrumentation and runner
  - reset UVM counters
  - configure KV address range
  - capture replayable fault stats
  - compute KV fault ratio
```

核心设计原则：

1. allocator 层只做它安全知道的事情：allocation size、phase、地址重叠、pool telemetry、device-direct 新分配准入。
2. KV cache 的运行期策略放在 vLLM block manager 边界，不在 allocator 层按裸指针驱逐。
3. weights 的运行期动作放在 MoE layer / model path 的语义安全点，不在 allocator 层盲目 offload。
4. scratch pool 优先做 admission control，不做 eviction。
5. Stage K 只协调已存在的高层安全动作，不直接迁移 raw pointer。

## 2. 关键文件总览

### 2.1 allocator 和 Python wrapper

```text
workloads/vllm/vllm/uvm_test/uvm_allocator.cpp
workloads/vllm/vllm/vllm/device_allocator/uvm.py
workloads/vllm/vllm/vllm/device_allocator/uvm_pool_coordinator.py
```

`uvm_allocator.cpp` 是 C++ allocator shim，负责：

1. 默认使用 `cudaMallocManaged` 分配。
2. 通过 `VLLM_UVM_LOG_FILE` 输出 allocation trace。
3. 读取 phase、size、gap-watch、device-direct、KV/weights budget、pool registry、scratch pool 等环境变量。
4. 给每次 allocation 标注 class、pool、backend、budget reason。
5. 输出 session summary，用于 `summarize_gap_watch_metrics.py` 聚合。

`uvm.py` 是 Python wrapper，负责：

1. 查找并加载 `uvm_allocator.abi3.so` / `uvm_allocator.so`。
2. 调用 `torch.cuda.memory.CUDAPluggableAllocator` 启用 custom allocator。
3. 提供 `uvm_allocation_phase()` / `uvm_enabled_allocation_phase()`。
4. 暴露 `prefetch_range_to_device()`、`prefetch_range_to_cpu()`、`advise_range_preferred_location()`。
5. 在进程退出时关闭 allocator log。

`uvm_pool_coordinator.py` 是 Stage K 新增控制面，负责：

1. 统一记录跨 pool action 请求。
2. 按 global / weights / KV / scratch budget 做 grant 或 deny。
3. 输出 `coordinator_config`、`coordinator_pressure`、`coordinator_request`、`coordinator_summary` JSONL。

### 2.2 vLLM 语义接入点

```text
workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py
workloads/vllm/vllm/vllm/v1/core/kv_cache_manager.py
workloads/vllm/vllm/vllm/v1/core/kv_cache_coordinator.py
workloads/vllm/vllm/vllm/v1/core/block_pool.py
workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py
```

`fused_moe/layer.py` 负责：

1. Stage E MoE routing trace。
2. Stage I hot expert weight GPU prefetch。
3. Stage I cold expert CPU advise / CPU prefetch。
4. Stage K weights action grant/deny 接入。

`kv_cache_manager.py` 负责：

1. 初始化 Stage J runtime KV policy。
2. 在 `allocate_slots()` 之前调用 Stage J policy 判断是否允许新 KV block admission。
3. 在无 free block 或 budget pressure 时记录 runtime trace。

`block_pool.py` 负责：

1. 提供 prefix-cache free block 安全 eviction executor。
2. 只清理 `ref_cnt == 0` 且处于 free queue 的 prefix-cache metadata。
3. 通过 Stage J policy 记录真实 prefix-cache eviction 事件。

`uvm_kv_runtime_policy.py` 负责：

1. Stage J runtime KV pressure 计算。
2. trace-only would-deny / would-evict candidate。
3. enforce 模式下 soft admission deny。
4. 调用 `BlockPool.evict_cached_free_blocks()`。
5. 向 Stage K 上报 KV pressure，并在 prefix eviction 前请求 coordinator grant。

### 2.3 runner 和验收脚本

```text
workloads/vllm/run_kv_fault_ratio.sh
workloads/vllm/summarize_gap_watch_metrics.py
workloads/vllm/plan_stage_h_weight_expert_actions.py
workloads/vllm/stage_i_ab_sweep.py
workloads/vllm/check_stage_c1_success.py
workloads/vllm/check_stage_c2_success.py
workloads/vllm/check_stage_d_success.py
workloads/vllm/check_stage_d2_success.py
workloads/vllm/check_stage_e_success.py
workloads/vllm/check_stage_f_success.py
workloads/vllm/check_stage_g_success.py
workloads/vllm/check_stage_h_success.py
workloads/vllm/check_stage_i_success.py
workloads/vllm/check_stage_j_success.py
workloads/vllm/check_stage_k_success.py
```

`run_kv_fault_ratio.sh` 是集成 runner，负责：

1. 清理 GPU 计算进程。
2. reset UVM kernel counters。
3. 启动单进程 vLLM server。
4. 等待 server ready 和 fresh KV range。
5. 配置 kernel UVM KV 地址范围。
6. 运行 `vllm bench serve`。
7. 收集 replayable fault stats。
8. 输出 delta fault stats。
9. 透传 Stage C-K 的环境变量。

`summarize_gap_watch_metrics.py` 负责从 allocator trace / session summary 中提取指标，例如：

1. gap-watch policy 命中情况。
2. device-direct 记录和预算拒绝。
3. KV / weights budget telemetry。
4. pool registry 三类对象指标。
5. scratch pool admission 指标。

`plan_stage_h_weight_expert_actions.py` 负责把 Stage E 的 weight map、MoE routing trace 和可选 fault address log 合成 Stage H hot/cold expert plan。

各 `check_stage_*_success.py` 是端到端或 self-test 验收入口。

## 3. Stage 0 / Stage 1：基础 UVM trace policy

### 3.1 已实现内容

早期阶段完成了 UVM allocator 的基础接入和 trace-only policy：

1. vLLM 可以通过 custom allocator 使用 UVM managed memory。
2. Python runtime 可以给 allocator 注入 allocation phase。
3. allocator 可以记录 allocation size、phase、地址、backend、classification。
4. 支持 warmup prefetch / preferred-location advise 的保守 policy 开关。
5. 支持 unknown allocation detail trace，为后续 gap 分析提供数据。

### 3.2 主要修改

```text
workloads/vllm/vllm/uvm_test/uvm_allocator.cpp
workloads/vllm/vllm/vllm/device_allocator/uvm.py
workloads/vllm/run_kv_fault_ratio.sh
```

典型环境变量：

```text
VLLM_USE_UVM=1
VLLM_UVM_LOG_FILE=/tmp/vllm_uvm_allocator_trace.log
VLLM_UVM_TRACE_MIN_BYTES=1048576
VLLM_UVM_POLICY_ENABLE=1
VLLM_UVM_POLICY_MODE=trace_only
VLLM_UVM_POLICY_WARMUP_PREFETCH_MIN_BYTES=1048576
VLLM_UVM_POLICY_WARMUP_ADVISE_GPU=0
VLLM_UVM_UNKNOWN_DETAIL_ENABLE=0|1
VLLM_UVM_UNKNOWN_DETAIL_MIN_BYTES=0
```

### 3.3 边界

这个阶段不做独立 pool，也不做运行期 eviction。它只建立“能看见 allocation，并且能解释部分 allocation phase”的基础。

## 4. Gap Watch / Unknown Gap：地址区间定位和目标策略

### 4.1 已实现内容

Gap Watch 用于解决 UVM fault / allocation 中大量 unknown gap 的归因问题：

1. 支持从地址日志里发现 unknown gap。
2. 支持指定 gap 起止地址进行 watch。
3. 支持统计 watch range 与 allocation 的 overlap。
4. 支持把 gap watch 与 policy action 绑定。
5. 支持 same-run auto gap watch：先 probe，再选择 gap，再 main benchmark。

### 4.2 主要修改

```text
workloads/vllm/vllm/uvm_test/uvm_allocator.cpp
workloads/vllm/run_kv_fault_ratio.sh
workloads/vllm/summarize_gap_watch_metrics.py
docs/vllm_uvm_unknown_gap_resolution.md
docs/vllm_uvm_same_run_auto_gap_watch.md
```

典型环境变量：

```text
VLLM_UVM_GAP_WATCH_ENABLE=0|1
VLLM_UVM_GAP_WATCH_NAME=gap_watch
VLLM_UVM_GAP_WATCH_START=0x...
VLLM_UVM_GAP_WATCH_END=0x...
VLLM_UVM_GAP_WATCH_ALL_CLASSES=1
VLLM_UVM_GAP_WATCH_MIN_BYTES=0
VLLM_UVM_GAP_WATCH_TARGET_CLASS=any
VLLM_UVM_GAP_WATCH_POLICY_ACTION=observe|prefetch|advise_prefetch|device_direct_trace|device_direct
VLLM_UVM_GAP_WATCH_CONTROL_FILE=/tmp/control.json
VLLM_UVM_GAP_WATCH_REFRESH_MS=250
```

### 4.3 边界

Gap Watch 是定位工具和保守策略入口，不等价于完整 pool 管理。它可以帮助发现热点 unknown allocation，但不能单独证明某个对象可以安全 eviction。

## 5. Stage B / C / C1 / C2：Device-direct 分配路径

### 5.1 已实现内容

Stage B-C 系列把部分短生命周期 runtime scratch 从 UVM managed allocation 中分流到 GPU-only backend：

1. Stage B 建立 device-direct eligibility：根据 phase、size、target class 判断是否适合走 GPU-only backend。
2. Stage C 增加真实 `cudaMalloc` backend。
3. Stage C1 增加 attention-only 或指定 phase 的保守启用方案。
4. Stage C2 增加 `cudaMallocAsync` backend 和 mempool release threshold 配置。
5. 支持全局 device-direct live bytes budget。
6. 预算不足、backend 失败或不满足条件时回退 managed UVM。

### 5.2 主要修改

```text
workloads/vllm/vllm/uvm_test/uvm_allocator.cpp
workloads/vllm/run_kv_fault_ratio.sh
workloads/vllm/check_stage_c1_success.py
workloads/vllm/check_stage_c2_success.py
docs/vllm_uvm_device_direct_stage_b_implementation.md
docs/vllm_uvm_device_direct_stage_c_attention_only_implementation.md
docs/vllm_uvm_device_direct_stage_c2_async_backend_implementation.md
```

典型环境变量：

```text
VLLM_UVM_DEVICE_DIRECT_ENABLE=0|1
VLLM_UVM_DEVICE_DIRECT_MIN_BYTES=4096
VLLM_UVM_DEVICE_DIRECT_MAX_BYTES=1048576
VLLM_UVM_DEVICE_DIRECT_MAX_TOTAL_BYTES=268435456
VLLM_UVM_DEVICE_DIRECT_BACKEND=cuda_malloc|cuda_malloc_async
VLLM_UVM_DEVICE_DIRECT_POOL_RELEASE_THRESHOLD=<empty|0|bytes>
VLLM_UVM_DEVICE_DIRECT_TARGET_PHASES=enabled:attention,enabled:moe,enabled:model_forward
```

### 5.3 Trace / metrics

allocator summary 中会出现：

```text
device_direct_trace_records
device_direct_eligible_records
device_direct_actual_records
device_direct_budget_reject_records
device_direct_backend_counts
device_direct_peak_live_bytes_observed
```

### 5.4 边界

Device-direct 只改变“新 allocation 的 backend”。它不迁移已有 UVM allocation，不做 eviction，也不处理 KV/weights 这类需要上层语义的对象。

## 6. Stage D / D2：KV Cache 初始化期预算 telemetry

### 6.1 已实现内容

Stage D 把 KV cache 从普通 UVM allocation 中识别出来，并建立独立预算 telemetry：

1. allocator 能识别 KV cache allocation。
2. 统计 KV requested/live/peak/free bytes。
3. 支持 `trace_only` 与 `enforce` 两种 budget mode。
4. `enforce` 在 allocator 层仍然是 soft signal，不直接让 vLLM 失败。
5. runner 可以结合 kernel UVM replayable fault counters 计算 KV fault ratio。
6. Stage D2 重点验证 semantic KV budget enforce 信号是否生效。

### 6.2 主要修改

```text
workloads/vllm/vllm/uvm_test/uvm_allocator.cpp
workloads/vllm/run_kv_fault_ratio.sh
workloads/vllm/check_stage_d_success.py
workloads/vllm/check_stage_d2_success.py
docs/vllm_uvm_stage_d_kv_budget_telemetry_implementation.md
docs/vllm_uvm_stage_d_kv_budget_detailed_implementation.md
```

典型环境变量：

```text
VLLM_UVM_KV_BUDGET_BYTES=0
VLLM_UVM_KV_BUDGET_MODE=trace_only|enforce
```

### 6.3 Trace / metrics

summary 中会出现：

```text
kv_budget_bytes
kv_budget_mode
kv_trace_allocations
kv_live_bytes
kv_peak_live_bytes_observed
kv_budget_over_records
kv_budget_reject_records
kv_budget_reason_counts
```

### 6.4 边界

Stage D 只覆盖 KV 初始化期 allocation budget telemetry。它不做运行期 block eviction，也不做 CPU KV swap。运行期 KV 控制在 Stage J。

## 7. Stage E：Weights budget telemetry、weight map、MoE routing trace

### 7.1 已实现内容

Stage E 把模型权重从 managed allocation 中识别出来，并补充权重语义：

1. allocator 识别 weights allocation，统计 requested/live/peak/free bytes。
2. 支持 weights 独立 budget telemetry。
3. 输出 weight tensor semantic address map JSONL。
4. 在 MoE layer 中记录 expert routing trace。
5. 后续 Stage H 可用 weight map + routing trace 生成 expert hot/cold plan。

### 7.2 主要修改

```text
workloads/vllm/vllm/uvm_test/uvm_allocator.cpp
workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py
workloads/vllm/run_kv_fault_ratio.sh
workloads/vllm/check_stage_e_success.py
docs/vllm_uvm_stage_e_weights_budget_telemetry_implementation.md
docs/vllm_uvm_stage_e_weights_budget_detailed_implementation.md
```

典型环境变量：

```text
VLLM_UVM_WEIGHT_BUDGET_BYTES=0
VLLM_UVM_WEIGHT_BUDGET_MODE=trace_only|enforce
VLLM_UVM_WEIGHT_MAP_ENABLE=0|1
VLLM_UVM_WEIGHT_MAP_FILE=/tmp/vllm_uvm_weight_regions.jsonl
VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0|1
VLLM_UVM_MOE_ROUTING_TRACE_FILE=/tmp/vllm_uvm_moe_routing_trace.jsonl
```

### 7.3 Trace / metrics

summary 中会出现：

```text
weight_budget_bytes
weight_budget_mode
weight_trace_allocations
weight_live_bytes
weight_peak_live_bytes_observed
weight_budget_over_records
weight_budget_reject_records
weight_budget_reason_counts
```

weight map JSONL 记录 tensor 名称、地址范围、大小、role、layer/expert 信息。MoE routing trace 记录 layer/expert 的 token 命中热度。

### 7.4 边界

Stage E 不执行 weights offload / eviction / prefetch。它只回答“weights 对象是谁、在哪里、热不热”。执行动作从 Stage I 开始。

## 8. Stage F：统一 Pool Registry

### 8.1 已实现内容

Stage F 在 allocator 层建立统一 pool registry，把对象归入：

```text
kv_cache
weights
runtime_scratch
other
```

已实现能力：

1. 每次 allocation 记录 pool kind。
2. 统计每类 pool 的 allocation/free/live/peak。
3. free 时按对象信息闭环更新 live bytes。
4. 输出 pool kind counts 和 pool alloc bytes by kind。

### 8.2 主要修改

```text
workloads/vllm/vllm/uvm_test/uvm_allocator.cpp
workloads/vllm/run_kv_fault_ratio.sh
workloads/vllm/check_stage_f_success.py
docs/vllm_uvm_stage_f_pool_registry_implementation.md
```

典型环境变量：

```text
VLLM_UVM_POOL_REGISTRY_ENABLE=0|1
```

### 8.3 Trace / metrics

summary 中会出现：

```text
pool_registry_enabled
pool_registry_alloc_records
pool_registry_free_records
pool_registry_live_objects
pool_registry_peak_live_objects
pool_kind_counts
pool_alloc_bytes_by_kind
pool_kv_live_bytes
pool_weight_live_bytes
pool_runtime_scratch_live_bytes
```

### 8.4 边界

Stage F 是 registry 和 telemetry，不改变 allocation policy，不做 eviction/prefetch/offload。

## 9. Stage G：Runtime Scratch Pool Admission Control

### 9.1 已实现内容

Stage G 是第一个 per-pool 执行型策略，但只作用于 `runtime_scratch` 新 allocation：

1. 判断 allocation 是否属于 runtime scratch pool。
2. 判断 phase 是否命中 target phases。
3. 判断是否满足 scratch pool device-direct admission。
4. 预算允许时走 Stage C/C2 device-direct backend。
5. 预算不足时：
   - `trace_only`：记录 over-budget，但仍允许 device-direct。
   - `enforce`：拒绝 device-direct，回退 managed UVM。
6. free 时释放 scratch pool device-direct live bytes。

### 9.2 主要修改

```text
workloads/vllm/vllm/uvm_test/uvm_allocator.cpp
workloads/vllm/run_kv_fault_ratio.sh
workloads/vllm/check_stage_g_success.py
docs/vllm_uvm_stage_g_runtime_scratch_pool_admission.md
```

典型环境变量：

```text
VLLM_UVM_SCRATCH_POOL_ENABLE=0|1
VLLM_UVM_SCRATCH_POOL_BUDGET_BYTES=1048576
VLLM_UVM_SCRATCH_POOL_MODE=trace_only|enforce
VLLM_UVM_SCRATCH_POOL_TARGET_PHASES=enabled:attention
```

### 9.3 Trace / metrics

summary 中会出现：

```text
scratch_pool_enabled
scratch_pool_budget_bytes
scratch_pool_mode
scratch_pool_trace_records
scratch_pool_eligible_records
scratch_pool_device_direct_records
scratch_pool_live_bytes
scratch_pool_peak_live_bytes
scratch_pool_budget_over_records
scratch_pool_budget_reject_records
scratch_pool_reason_counts
```

### 9.4 边界

Stage G 是 admission control，不是 eviction。它不会迁移或驱逐已经存在的 scratch allocation。

## 10. Stage H：Weights Hot/Cold Trace-only Plan

### 10.1 已实现内容

Stage H 基于 Stage E 产物生成 MoE expert weight hot/cold plan：

1. 读取 weight map JSONL。
2. 读取 MoE routing trace JSONL。
3. 可选读取 UVM fault address records。
4. 将 fused expert tensor 按 expert 维度拆成 logical expert slices。
5. 聚合 expert heat。
6. 生成 `prefetch_plan` 和 `offload_plan`。
7. 输出 plan summary 和 validation metrics。

### 10.2 主要修改

```text
workloads/vllm/plan_stage_h_weight_expert_actions.py
workloads/vllm/check_stage_h_success.py
docs/vllm_uvm_stage_h_weight_hot_cold_plan.md
```

Stage H plan JSON 典型字段：

```text
mode
weight_map_records
expert_weight_range_records
logical_fused_expert_records
expert_heat_records
moe_routing_records
routing_join_records
fault_address_records
weight_fault_join_records
prefetch_plan
offload_plan
```

### 10.3 边界

Stage H 不执行任何迁移动作，不调用 `cudaMemPrefetchAsync`，不做 CPU offload。它只生成 Stage I 可执行的候选计划。

## 11. Stage I：MoE Expert Weight Prefetch / Offload 执行

### 11.1 已实现内容

Stage I 在 MoE layer 的安全点执行 Stage H plan：

1. active hot expert GPU prefetch。
2. cold inactive expert preferred-location CPU advise。
3. cold inactive expert CPU prefetch。
4. 按 layer/expert/role 选择 expert weight slice。
5. 使用 Stage H `prefetch_plan` 作为 hot whitelist。
6. 使用 Stage H `offload_plan` 作为 cold candidate list。
7. 跳过当前 step active expert，避免把正在使用的 expert cold offload。
8. 支持 per-layer-call bytes budget。
9. 支持 per-layer experts budget。
10. 输出 Stage I JSONL trace。
11. 已接入 Stage K weights action coordinator。

### 11.2 主要修改

```text
workloads/vllm/vllm/vllm/device_allocator/uvm.py
workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py
workloads/vllm/run_kv_fault_ratio.sh
workloads/vllm/check_stage_i_success.py
workloads/vllm/stage_i_ab_sweep.py
docs/vllm_uvm_stage_i_weight_expert_prefetch.md
```

`uvm.py` 新增或暴露：

```text
prefetch_range_to_device(ptr, size, device)
prefetch_range_to_cpu(ptr, size)
advise_range_preferred_location(ptr, size, device)
```

`FusedMoE.select_experts()` 附近接入：

```text
MoE routing trace
-> _maybe_prefetch_uvm_expert_weights(topk_ids)
-> _maybe_offload_uvm_cold_expert_weights(topk_ids)
```

典型环境变量：

```text
VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0|1
VLLM_UVM_WEIGHT_PREFETCH_MODE=trace_only|prefetch
VLLM_UVM_WEIGHT_PREFETCH_TRACE_FILE=/tmp/vllm_uvm_weight_prefetch_stage_i.jsonl
VLLM_UVM_WEIGHT_PREFETCH_MAX_BYTES_PER_STEP=67108864
VLLM_UVM_WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER=2
VLLM_UVM_WEIGHT_PREFETCH_TARGET_ROLES=moe_gate_up,moe_down
VLLM_UVM_WEIGHT_PREFETCH_DEVICE=-1
VLLM_UVM_WEIGHT_PREFETCH_PLAN_FILE=/tmp/stage_h_plan.json
VLLM_UVM_WEIGHT_PREFETCH_REQUIRE_PLAN=0|1

VLLM_UVM_WEIGHT_OFFLOAD_ENABLE=0|1
VLLM_UVM_WEIGHT_OFFLOAD_MODE=trace_only|advise_cpu|prefetch_cpu
VLLM_UVM_WEIGHT_OFFLOAD_PLAN_FILE=/tmp/stage_h_plan.json
VLLM_UVM_WEIGHT_OFFLOAD_MAX_BYTES_PER_STEP=67108864
VLLM_UVM_WEIGHT_OFFLOAD_MAX_EXPERTS_PER_LAYER=1
VLLM_UVM_WEIGHT_OFFLOAD_TARGET_ROLES=moe_gate_up,moe_down
```

### 11.3 Trace action

Stage I trace 包含：

```text
trace_prefetch_candidate
prefetch_issued
prefetch_skipped
budget_reject
coordinator_reject
step_summary

trace_offload_candidate
offload_advise_cpu_issued
offload_prefetch_cpu_issued
offload_skipped
offload_budget_reject
offload_coordinator_reject
offload_step_summary
```

### 11.4 实验结果解释

Stage I A/B 已证明功能链路可用，但性能上可能退化。典型现象：

1. `no_stage_i` TPOT 最好。
2. prefetch trace / prefetch 执行会增加 Python hook、routing trace、JSONL、CUDA prefetch 调用开销。
3. cold offload advise / prefetch 可能进一步拉低 throughput。
4. 因此 Stage I 当前主要证明“语义安全点可执行”，不代表默认应开启提升性能。

### 11.5 边界

Stage I 只处理 MoE expert weight slice，不做 dense/shared weights 全量 offload，不做通用 weights eviction。

## 12. Stage J：KV Runtime Pressure / Prefix-cache Eviction

### 12.1 已实现内容

Stage J 把 KV 控制从初始化期 telemetry 推进到 vLLM block-manager runtime 边界：

1. 在 `KVCacheManager` 初始化 `UvmKvRuntimePolicy`。
2. 在 `allocate_slots()` 前计算 runtime KV block pressure。
3. trace-only 模式输出 `allocation_pressure`、candidate、`would_deny_allocation`。
4. enforce 模式可以通过返回 `None` 拒绝新的 KV block admission。
5. 支持 `lru_prefix_cache` / `scheduler_aware` policy 标签。
6. 支持 candidate snapshot。
7. 支持 prefix-cache free-block eviction executor。
8. executor 只清理 `ref_cnt == 0` 且处于 free queue 的 prefix-cache metadata。
9. 输出 `runtime_summary`。
10. 已接入 Stage K KV pressure 和 prefix eviction grant/deny。

### 12.2 主要修改

```text
workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py
workloads/vllm/vllm/vllm/v1/core/kv_cache_manager.py
workloads/vllm/vllm/vllm/v1/core/kv_cache_coordinator.py
workloads/vllm/vllm/vllm/v1/core/block_pool.py
workloads/vllm/run_kv_fault_ratio.sh
workloads/vllm/check_stage_j_success.py
docs/vllm_uvm_stage_j_kv_runtime_eviction.md
```

典型环境变量：

```text
VLLM_UVM_KV_RUNTIME_ENABLE=0|1
VLLM_UVM_KV_RUNTIME_MODE=trace_only|enforce
VLLM_UVM_KV_RUNTIME_BUDGET_BYTES=0
VLLM_UVM_KV_RUNTIME_BUDGET_BLOCKS=0
VLLM_UVM_KV_RUNTIME_TRACE_FILE=/tmp/vllm_uvm_kv_runtime_stage_j.jsonl
VLLM_UVM_KV_RUNTIME_EVICTION_POLICY=lru_prefix_cache|scheduler_aware
VLLM_UVM_KV_RUNTIME_CANDIDATE_LIMIT=16
VLLM_UVM_KV_RUNTIME_PREFIX_EVICT_ENABLE=0|1
VLLM_UVM_KV_RUNTIME_PREFIX_EVICT_MAX_BLOCKS=0
```

### 12.3 Trace action

Stage J trace 包含：

```text
runtime_config
allocation_pressure
would_reuse_free_block
would_evict_candidate
would_deny_allocation
deny_allocation
allocation_no_free_blocks
prefix_evict_attempt
prefix_evict_success
prefix_evict_noop
prefix_evict_failed
prefix_evict_coordinator_reject
evict_prefix_cache_block
runtime_summary
```

### 12.4 验收结果

`check_stage_j_success.py` 已支持：

```bash
./check_stage_j_success.py
./check_stage_j_success.py --self-test --require-prefix-eviction
```

已有实验中可看到：

```text
runtime enabled
allocation_pressure_records > 0
would_deny_records > 0
candidate_records > 0
prefix_evict_attempt_records > 0
prefix_evictions safe_ref_cnt_zero=True
failed_requests=0 in trace_only
```

### 12.5 边界

Stage J 当前不是完整 KV cache data eviction/offload/prefetch。它没有实现：

1. 活跃 KV block CPU swap。
2. offloaded KV block reload。
3. KV data prefetch-back。
4. 修改 active request block table。
5. allocator-side raw KV pointer migration。

这些应作为 Stage J2/J3 或 vLLM `kv_offload` / KV connector 接入继续做。

## 13. Stage K：Global Pool Coordinator

### 13.1 已实现内容

Stage K 是当前最新阶段，已经实现全局 high-level action coordinator：

1. 支持 `trace_only|enforce`。
2. 支持 global action budget。
3. 支持 weights / KV / scratch per-pool action budget。
4. 统一输出 coordinator JSONL。
5. Stage I expert weight prefetch/offload 前请求 weights grant。
6. Stage J runtime KV pressure 上报 coordinator pressure。
7. Stage J prefix-cache eviction 前请求 KV grant。
8. 默认关闭，不影响既有路径。
9. trace-only 下只记录 would-deny，不跳过动作。
10. enforce 下超预算 action 会被 skip / deny。

### 13.2 主要修改

```text
workloads/vllm/vllm/vllm/device_allocator/uvm_pool_coordinator.py
workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py
workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py
workloads/vllm/run_kv_fault_ratio.sh
workloads/vllm/check_stage_k_success.py
docs/vllm_uvm_stage_k_pool_coordinator.md
```

典型环境变量：

```text
VLLM_UVM_POOL_COORDINATOR_ENABLE=0|1
VLLM_UVM_POOL_COORDINATOR_MODE=trace_only|enforce
VLLM_UVM_POOL_COORDINATOR_TRACE_FILE=/tmp/vllm_uvm_pool_coordinator_stage_k.jsonl
VLLM_UVM_POOL_COORDINATOR_GLOBAL_BYTES_PER_STEP=0
VLLM_UVM_POOL_COORDINATOR_WEIGHT_BYTES_PER_STEP=0
VLLM_UVM_POOL_COORDINATOR_KV_BYTES_PER_STEP=0
VLLM_UVM_POOL_COORDINATOR_SCRATCH_BYTES_PER_STEP=0
VLLM_UVM_POOL_COORDINATOR_PRIORITY=kv,weights,scratch
```

### 13.3 Trace action

Stage K trace 包含：

```text
coordinator_config
coordinator_pressure
coordinator_request
coordinator_summary
coordinator_scope_reset
```

`coordinator_request` 关键字段：

```text
pool
requested_action
requested_bytes
allowed
would_deny
reason
scope_key
pool_budget_bytes
pool_used_bytes
global_budget_bytes
global_used_bytes
metadata
```

### 13.4 验收脚本

快速 self-test：

```bash
cd workloads/vllm
./check_stage_k_success.py
```

enforce self-test：

```bash
cd workloads/vllm
./check_stage_k_success.py --mode enforce
```

GPU 集成 probe：

```bash
cd workloads/vllm
./check_stage_k_success.py --gpu-run --mode trace_only
```

当前已验证：

```text
python3 -m py_compile ...
bash -n workloads/vllm/run_kv_fault_ratio.sh
./workloads/vllm/check_stage_k_success.py
./workloads/vllm/check_stage_k_success.py --mode enforce
git diff --check
```

### 13.5 边界

Stage K 第一版是全局协调器，不是完整跨 pool 数据迁移管理器。它当前不直接接 allocator C++ scratch pool budget，也不做 KV CPU swap。它协调的是已经存在的安全动作。

## 14. Kernel UVM fault stats 与 KV fault ratio

当前 runner 会配合 kernel-side UVM instrumentation 输出 replayable fault stats，例如：

```text
delta_faults
delta_duplicates
delta_after_dedup
delta_kv_faults
delta_kv_duplicates
delta_kv_after_dedup
delta_kv_ratio
delta_kv_after_dedup_ratio
```

运行流程：

1. 启动 server 后等待 fresh KV range。
2. 将 KV range 写入 nvidia_uvm 参数。
3. benchmark 与 server 使用同一个进程。
4. 收集 replayable fault stats。
5. 计算本 workload delta。

这套指标用于判断 UVM replayable faults 是否落在 KV cache range。当前多次实验中常见 `delta_kv_faults=0`，说明 fault 热点并不一定来自 KV cache，也解释了为什么不能只围绕 KV 做优化。

## 15. 一键实验入口

### 15.1 基础 KV fault ratio

```bash
cd workloads/vllm
./run_kv_fault_ratio.sh --mode trace --prompts 1 --output-len 128
```

### 15.2 Stage I A/B

```bash
cd workloads/vllm
./stage_i_ab_sweep.py
```

该脚本会：

1. 先跑 Stage H planning。
2. 生成 hot/cold expert plan。
3. 跑 no_stage_i。
4. 跑 prefetch trace-only。
5. 跑 prefetch execution。
6. 跑 prefetch + offload trace/advise。
7. 输出 CSV/JSON summary。

### 15.3 Stage J

```bash
cd workloads/vllm
./check_stage_j_success.py
./check_stage_j_success.py --self-test --require-prefix-eviction
```

### 15.4 Stage K

```bash
cd workloads/vllm
./check_stage_k_success.py
./check_stage_k_success.py --mode enforce
./check_stage_k_success.py --gpu-run --mode trace_only
```

## 16. 当前已经真实执行的功能

已经进入真实执行路径的内容：

1. UVM custom allocator 替换 PyTorch CUDA allocator。
2. allocator phase tracing。
3. gap-watch overlap tracing。
4. optional warmup prefetch / advise。
5. Stage C/C2 device-direct backend for eligible new allocations。
6. Stage G scratch pool device-direct admission / managed fallback。
7. Stage I MoE expert weight GPU prefetch。
8. Stage I cold expert CPU preferred-location advise。
9. Stage I cold expert CPU prefetch。
10. Stage J runtime KV admission soft deny in enforce mode。
11. Stage J prefix-cache free-block metadata eviction executor。
12. Stage K coordinator enforce deny for high-level actions。

## 17. 当前只做 telemetry / trace-only 的内容

这些内容已经可观测，但不是完整执行策略：

1. Stage D allocator-side KV budget enforce 是 soft signal。
2. Stage E allocator-side weights budget enforce 是 soft signal。
3. Stage H hot/cold expert plan 只生成计划。
4. Gap-watch 可以定位热点，但不证明对象可安全 eviction。
5. Stage K scratch pool 目前主要在 coordinator self-test / trace 层可见，C++ allocator scratch pool 尚未直接请求 Stage K grant。
6. Kernel KV fault ratio 只统计 fault 归因，不做策略。

## 18. 当前明确未实现的内容

尚未实现：

1. 完整 KV cache data CPU swap。
2. offloaded KV block reload / prefetch-back。
3. 活跃 KV block eviction。
4. KV connector / `vllm.v1.kv_offload` 完整集成。
5. dense/shared weights runtime offload。
6. 通用 weights eviction。
7. allocator 层按 raw pointer 驱逐 KV/weights。
8. Stage K 与 allocator C++ scratch pool 的直接 grant hook。
9. 时间窗口型 bandwidth coordinator。
10. 多请求/高并发下的 coordinator priority arbitration。

## 19. 当前推荐的后续路线

最稳妥的后续顺序：

1. 先跑 `check_stage_k_success.py --gpu-run --mode trace_only`，确认 Stage I/J/K 集成 trace 在真实 vLLM server 上闭环。
2. 完善 Stage K 对 allocator C++ scratch pool 的直接接入，让 runtime scratch 的真实 device-direct admission 也经过 coordinator。
3. 继续 Stage J2：接 vLLM KV offload/connector，而不是自己在 allocator 层迁移 raw KV pointer。
4. 做 Stage I performance cleanup：降低 Python hook、routing trace、JSONL 和重复 prefetch 的开销。
5. 最后再考虑更激进的 weights offload / KV swap / cross-pool arbitration。

## 20. 文档索引

当前相关文档：

```text
docs/vllm_uvm_memory_pool_evolution_plan.md
docs/vllm_uvm_independent_pool_eviction_prefetch_plan.md
docs/vllm_uvm_memory_pool_changes_walkthrough.md
docs/vllm_uvm_unknown_gap_resolution.md
docs/vllm_uvm_stage_d_kv_budget_detailed_implementation.md
docs/vllm_uvm_stage_e_weights_budget_detailed_implementation.md
docs/vllm_uvm_stage_f_pool_registry_implementation.md
docs/vllm_uvm_stage_g_runtime_scratch_pool_admission.md
docs/vllm_uvm_stage_h_weight_hot_cold_plan.md
docs/vllm_uvm_stage_i_weight_expert_prefetch.md
docs/vllm_uvm_stage_j_kv_runtime_eviction.md
docs/vllm_uvm_stage_k_pool_coordinator.md
```

如果只想快速理解当前状态，先读本文档；如果要继续修改某个阶段，再读对应 Stage 文档。

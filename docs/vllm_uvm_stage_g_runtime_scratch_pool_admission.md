# vLLM UVM Stage G Runtime Scratch Pool Admission Control 详细实现说明

本文档详细说明 Stage G 的实现方式、修改文件、运行链路、日志字段、metrics 汇总、验收脚本和实验结果解释。它面向后续继续实现 Stage H/I/J/K 的人，目标是让读者能从文档直接对应到代码，理解当前项目如何把 `runtime_scratch` 从“只被记录的 pool”推进到“可执行准入控制的独立 pool”。

Stage G 的关键词是：runtime scratch、独立预算、device-direct 准入、managed fallback、不驱逐。

## 1. 背景

Stage F 已经把 vLLM UVM allocation 统一登记到了 pool registry，并把核心对象分成三类：

```text
kv_cache
weights
runtime_scratch
```

这解决了“对象是谁、属于哪个 pool、free 时能否闭环”的问题，但 Stage F 本身不改变分配策略。后续如果要实现独立 pool 管理，不能一开始就对所有 pool 做真实驱逐，因为三类对象的语义完全不同：

1. `kv_cache` 的驱逐必须经过 vLLM block manager 或 scheduler，否则可能破坏 token block 映射。
2. `weights` 的 offload/prefetch 必须结合 model runner、MoE expert 路由和权重语义，否则可能迁移错误对象。
3. `runtime_scratch` 生命周期短、分配频繁，更适合 admission control，而不是 eviction。

因此 Stage G 选择风险最低的第一个执行型策略：只对 `runtime_scratch` 做独立准入控制。预算允许时走 device-direct fast path，预算不足时回退 managed UVM，不做驱逐，也不让请求失败。

## 2. Stage G 的目标

Stage G 的目标是把 Stage C/C2 的 device-direct 能力接到 Stage F 的 `runtime_scratch` pool 上：

```text
cudaMallocManaged candidate
  -> classify allocation
  -> PoolKind::RuntimeScratch
  -> scratch pool enabled?
  -> phase allowlist
  -> size gate
  -> scratch pool live-byte budget reservation
  -> budget ok: request Stage C device-direct path
  -> budget exceeded in enforce: keep managed candidate
  -> free: release global device-direct budget and scratch pool budget
```

成功标准不是性能一定提升，而是：

1. 能观察到 runtime scratch allocation。
2. 能从中筛出 eligible allocation。
3. 一部分 eligible allocation 能真实进入 device-direct。
4. enforce 模式下 scratch device-direct peak live bytes 不超过预算。
5. 超预算 allocation 能 fallback managed。
6. benchmark 没有 failed requests。

## 3. Stage G 明确不做什么

Stage G 不做：

1. scratch eviction。
2. scratch prefetch。
3. KV runtime eviction/swap。
4. weight offload/prefetch。
5. MoE expert weight prefetch/offload。
6. UVM driver page eviction policy 修改。
7. 全局 pressure coordinator。

这里最重要的边界是：Stage G 是 admission control，不是 eviction。它只决定“新的 runtime scratch allocation 是否允许进入 device-direct pool”，不迁移或驱逐已经存在的对象。

## 4. 修改文件总览

Stage G 主要修改这些文件：

```text
workloads/vllm/vllm/uvm_test/uvm_allocator.cpp
workloads/vllm/run_kv_fault_ratio.sh
workloads/vllm/summarize_gap_watch_metrics.py
workloads/vllm/check_stage_g_success.py
docs/vllm_uvm_independent_pool_eviction_prefetch_plan.md
docs/vllm_uvm_stage_g_runtime_scratch_pool_admission.md
```

各文件职责如下：

1. `uvm_allocator.cpp`
   - 读取 Stage G 环境变量。
   - 判断 allocation 是否属于 runtime scratch pool。
   - 执行 phase、size、budget gate。
   - 复用 Stage C/C2 的 device-direct 物理分配路径。
   - 在 free 路径释放 scratch pool budget。
   - 输出 Stage G trace 字段和 session summary。

2. `run_kv_fault_ratio.sh`
   - 增加 Stage G 命令行参数。
   - 校验参数合法性。
   - 把参数透传为 vLLM server 进程环境变量。
   - 与 Stage C device-direct 参数一起启动实验。

3. `summarize_gap_watch_metrics.py`
   - 解析 allocator trace 中的 `scratch_pool_*` 字段。
   - 汇总 scratch pool records、live bytes、peak live bytes、budget reject 和 reason counts。
   - 输出 JSON metrics，供 check 脚本和人工分析使用。

4. `check_stage_g_success.py`
   - 一键运行 Stage G admission probe。
   - 开启 pool registry、scratch pool 和 device-direct。
   - 运行 benchmark。
   - 生成 metrics JSON。
   - 校验 Stage G 是否满足成功条件。

5. `vllm_uvm_independent_pool_eviction_prefetch_plan.md`
   - 记录 Stage G 在总体路线图中的定位和验收标准。

## 5. 配置项

Stage G 在 allocator 侧新增四个核心配置。

### 5.1 `VLLM_UVM_SCRATCH_POOL_ENABLE`

是否启用 Stage G scratch pool admission control。

```bash
VLLM_UVM_SCRATCH_POOL_ENABLE=1
```

默认值为 `0`。关闭时 allocator 不会对 runtime scratch 做独立准入，只保留原有策略。

### 5.2 `VLLM_UVM_SCRATCH_POOL_BUDGET_BYTES`

scratch pool device-direct live bytes 预算。

```bash
VLLM_UVM_SCRATCH_POOL_BUDGET_BYTES=16777216
```

含义：

1. 只统计实际进入 scratch pool device-direct 的 live bytes。
2. 不统计 fallback managed 的 runtime scratch。
3. `0` 表示 unlimited，不限制 scratch pool device-direct live bytes。

注意这个预算是 Stage G 的独立预算，不等于 Stage C 的全局 device-direct 预算。真实进入 device-direct 时，两层预算都要通过。

### 5.3 `VLLM_UVM_SCRATCH_POOL_MODE`

支持两种模式：

```text
trace_only
enforce
```

`trace_only`：

1. 记录是否超过 scratch pool budget。
2. 即使超过预算，仍允许 reserve scratch pool budget。
3. 用于观察和调参，不用于强限制。

`enforce`：

1. 如果新 allocation 会让 scratch pool live bytes 超过预算，则拒绝 scratch pool device-direct。
2. 被拒绝的 allocation fallback managed。
3. benchmark 不应失败。

代码里也兼容 `soft_enforce` 输入，并归一化为 `enforce`。

### 5.4 `VLLM_UVM_SCRATCH_POOL_TARGET_PHASES`

scratch pool admission 的 phase allowlist。

```bash
VLLM_UVM_SCRATCH_POOL_TARGET_PHASES=enabled:attention,enabled:moe,enabled:model_forward
```

匹配方式是逗号分隔的 prefix match。例如 `enabled` 可以覆盖 `enabled:attention`、`enabled:moe`、`enabled:model_forward`、`enabled:compute_logits` 等运行时阶段。

默认值是：

```text
enabled:attention
```

这个默认比较保守，所以在早期实验中如果只开默认 phase，可能看不到 eligible allocation。

## 6. 与 Stage C/C2 的关系

Stage G 没有重新实现一套 device-direct allocator，而是复用 Stage C/C2 的能力。

Stage C/C2 已经提供：

1. `device_direct_enable`
2. `device_direct_min_bytes`
3. `device_direct_max_bytes`
4. `device_direct_max_total_bytes`
5. `device_direct_target_phases`
6. `cuda_malloc` / `cuda_malloc_async` backend
7. `cuda_malloc_async` mempool release threshold 配置
8. 全局 device-direct live bytes 预算
9. device-direct fallback managed 逻辑

Stage G 在这套机制前面加了一层“scratch pool admission”：

```text
Stage G scratch budget gate
  -> passed: force policy_action = DeviceDirect
  -> then Stage C physical device-direct gate and allocation
```

因此一次 scratch allocation 要真实进入 device-direct，必须同时满足：

1. 属于 runtime scratch allocation class。
2. `VLLM_UVM_SCRATCH_POOL_ENABLE=1`。
3. 当前 phase 命中 `VLLM_UVM_SCRATCH_POOL_TARGET_PHASES`。
4. size 在 `device_direct_min_bytes` 和 `device_direct_max_bytes` 之间。
5. `device_direct_enable=1`。
6. Stage G scratch pool budget reserve 成功。
7. Stage C 全局 device-direct budget reserve 成功。
8. `cudaMalloc` 或 `cudaMallocAsync` 成功。
9. 原 managed candidate 能成功 `cudaFree`。

任何一步失败，都应安全回退 managed。

## 7. 与 Stage F 的关系

Stage F 提供 pool registry，Stage G 基于这个语义继续做执行策略。两者关系可以理解为：

```text
Stage F:
  AllocationClass -> PoolKind -> TRACE_POOL_ALLOC/FREE -> pool metrics

Stage G:
  PoolKind::RuntimeScratch -> scratch admission -> device-direct or managed fallback
```

Stage G 仍然依赖 allocator 的 `AllocationClass` 判定来识别 runtime scratch。当前纳入 scratch pool 的类别包括：

```text
AllocationClass::RuntimeScratch
AllocationClass::RuntimeWorkspace
AllocationClass::WarmupWorkspace
```

这些类别在 Stage F 中会映射为：

```text
PoolKind::RuntimeScratch
pool_kind=runtime_scratch
```

Stage G 验收脚本会显式开启 Stage F：

```bash
--uvm-pool-registry-enable 1
```

这样可以同时验证 pool registry 记录和 scratch admission 记录是否一致。

## 8. `uvm_allocator.cpp` 实现详解

### 8.1 新增全局配置

allocator 顶部新增 Stage G 配置：

```cpp
static int scratch_pool_enable = 0;
static size_t scratch_pool_budget_bytes = 1 * 1024 * 1024;
static std::string scratch_pool_mode = "trace_only";
static std::string scratch_pool_target_phases = "enabled:attention";
```

默认关闭真实策略，默认预算 1 MiB，默认模式 `trace_only`，默认只允许 attention phase。

### 8.2 初始化读取环境变量

allocator 初始化阶段读取环境变量：

```text
VLLM_UVM_SCRATCH_POOL_ENABLE
VLLM_UVM_SCRATCH_POOL_BUDGET_BYTES
VLLM_UVM_SCRATCH_POOL_MODE
VLLM_UVM_SCRATCH_POOL_TARGET_PHASES
```

读取后会写入 session header，方便后处理脚本确认这次实验到底用的是什么配置。

### 8.3 新增 telemetry counters

Stage G 新增 counters：

```text
scratch_pool_trace_allocs
scratch_pool_eligible_allocs
scratch_pool_device_direct_allocs
scratch_pool_device_direct_bytes
scratch_pool_device_direct_live_bytes
scratch_pool_device_direct_peak_live_bytes
scratch_pool_budget_over_allocs
scratch_pool_budget_reject_allocs
scratch_pool_device_direct_free_success_allocs
```

这些 counters 分别回答：

1. 有多少 runtime scratch allocation 被 Stage G 观察到。
2. 有多少通过 phase、size、device-direct 开关检查。
3. 有多少真实进入 scratch pool device-direct。
4. scratch pool device-direct 累计字节数是多少。
5. 当前 scratch pool device-direct live bytes 是多少。
6. 峰值 live bytes 是多少。
7. 有多少 allocation 会超过 Stage G 预算。
8. enforce 模式下有多少因为预算被拒绝并 fallback managed。
9. free 路径是否释放了 Stage G 预算。

### 8.4 扩展 `AllocationInfo`

Stage G 在 allocation metadata 中新增字段：

```cpp
bool scratch_pool_tracked;
bool scratch_pool_eligible;
bool scratch_pool_device_direct;
bool scratch_pool_budget_over_budget;
std::string scratch_pool_reason;
```

含义：

1. `scratch_pool_tracked`
   - 该 allocation 是否参与 Stage G 观察。
   - 条件是 Stage G 开启、allocation 是 runtime scratch、device 合法。

2. `scratch_pool_eligible`
   - 是否通过 Stage G 的 phase、size、device-direct enable 检查。
   - eligible 不代表一定进入 device-direct，因为还可能被 budget 拒绝。

3. `scratch_pool_device_direct`
   - 是否真实进入 scratch pool device-direct。
   - free 路径只对这个字段为 true 的对象释放 scratch pool budget。

4. `scratch_pool_budget_over_budget`
   - 本次 allocation 是否会导致 scratch pool 预算超限。
   - trace-only 模式下可能为 true 但仍进入 device-direct。

5. `scratch_pool_reason`
   - 解释本次 allocation 的 Stage G 决策结果。

### 8.5 新增 helper

Stage G 新增或复用这些 helper：

```cpp
normalize_scratch_pool_mode()
is_scratch_pool_target_phase()
reserve_scratch_pool_device_direct_budget()
release_scratch_pool_device_direct_budget()
scratch_pool_budget_remaining_snapshot()
```

`normalize_scratch_pool_mode()` 把非法输入统一降级为 `trace_only`，把 `enforce` 和 `soft_enforce` 归一化为 `enforce`。

`is_scratch_pool_target_phase()` 使用逗号分隔 prefix match 判断当前 phase 是否允许进入 scratch pool fast path。

`reserve_scratch_pool_device_direct_budget()` 是 Stage G 的核心预算函数。它使用 CAS 更新 `scratch_pool_device_direct_live_bytes`：

```text
next_live = current_live + size
over_budget = budget > 0 && next_live > budget

if over_budget and mode == enforce:
  return false

otherwise:
  live_bytes = next_live
  update peak
  return true
```

`release_scratch_pool_device_direct_budget()` 在 free 或 device-direct swap 失败时归还 Stage G 预算。

`scratch_pool_budget_remaining_snapshot()` 用于日志输出，不改变状态。

## 9. malloc 路径逐步说明

Stage G 的 malloc 路径发生在：

1. allocation class 分类之后。
2. Stage D KV budget 和 Stage E weight budget 记录之后。
3. Stage F pool registry 记录之后。
4. Stage C device-direct 物理分配之前。

### 9.1 判断是否 tracked

代码先判断：

```cpp
bool scratch_pool_tracked =
    scratch_pool_enable &&
    is_runtime_scratch_pool_allocation(alloc_class) &&
    device >= 0;
```

如果不满足，`scratch_pool_reason=not_scratch_pool`。

### 9.2 phase gate

如果 tracked，但当前 phase 不在 allowlist 中：

```text
scratch_pool_reason=scratch_pool_phase_not_allowed
```

这类 allocation 会保持 managed，不进入 device-direct。

### 9.3 size gate

Stage G 复用 Stage C 的 device-direct size gate：

```text
size >= device_direct_min_bytes
size <= device_direct_max_bytes
```

过小：

```text
scratch_pool_reason=scratch_pool_below_min_bytes
```

过大：

```text
scratch_pool_reason=scratch_pool_above_max_bytes
```

这解释了早期实验里 `scratch_pool_above_max_bytes` 很多的原因：默认或传入的 `--max-bytes` 如果太小，大块 runtime scratch 会被排除。

### 9.4 device-direct enable gate

如果 Stage G 开启，但 Stage C device-direct 没开：

```text
scratch_pool_reason=scratch_pool_device_direct_disabled
```

Stage G 本身不单独申请 device-only memory，必须依赖 Stage C device-direct 开关。

### 9.5 scratch pool budget reservation

通过前面 gate 后：

```text
scratch_pool_eligible=true
scratch_pool_eligible_allocs += 1
reserve_scratch_pool_device_direct_budget(size, &over_budget)
```

结果分三类：

1. 预算内 reserve 成功：

```text
scratch_pool_reason=scratch_pool_device_direct_enabled
```

2. trace-only 模式下超预算但仍 reserve 成功：

```text
scratch_pool_reason=scratch_pool_trace_only_over_budget_device_direct
scratch_pool_over_budget=1
```

3. enforce 模式下超预算，reserve 失败：

```text
scratch_pool_reason=scratch_pool_budget_exceeded_fallback_managed
scratch_pool_over_budget=1
scratch_pool_budget_reject_allocs += 1
```

第三类不会进入 device-direct，会保留原 managed candidate。

### 9.6 接入 Stage C device-direct 请求

如果 Stage G eligible 且 scratch budget reserve 成功：

```cpp
policy_action = PolicyAction::DeviceDirect;
policy_source = "scratch_pool_policy";
device_direct_requested = true;
device_direct_reason = scratch_pool_reason;
```

这一步相当于 Stage G 对 Stage C 说：这个 runtime scratch allocation 已经通过 scratch pool 准入，请尝试用 device-direct 物理分配替换 managed candidate。

### 9.7 Stage C 全局 device-direct gate

Stage C 仍会检查全局 device-direct 预算：

```text
reserve_device_direct_budget(size)
```

如果全局预算不足：

```text
device_direct_reason=device_direct_budget_exceeded
scratch_pool_reason=scratch_pool_global_device_direct_budget_exceeded
```

并且 Stage G 会释放刚才 reserve 的 scratch pool budget，避免账本泄漏。

### 9.8 物理分配和指针替换

如果全局预算通过，Stage C 根据 backend 调用：

```text
cudaMalloc
cudaMallocAsync
```

成功后释放原始 managed candidate：

```text
cudaFree(managed_candidate)
```

然后把返回给 vLLM 的指针替换成 device-only pointer：

```text
placement_backend=device_direct
scratch_pool_device_direct=true
scratch_pool_device_direct_allocs += 1
scratch_pool_device_direct_bytes += size
```

如果 `cudaMalloc` / `cudaMallocAsync` 失败，或者释放 managed candidate 失败，则：

1. 释放 Stage C 全局 device-direct budget。
2. 释放 Stage G scratch pool budget。
3. 保留 managed candidate。
4. 记录 fallback reason。

常见 reason：

```text
scratch_pool_malloc_failed_fallback_managed
scratch_pool_malloc_async_failed_fallback_managed
scratch_pool_managed_candidate_free_failed
scratch_pool_device_direct_pool_config_failed
```

## 10. free 路径逐步说明

free 路径依赖 `AllocationInfo` 中保存的 Stage G 字段。

如果对象满足：

```text
placement_backend_name == "device_direct"
scratch_pool_device_direct == true
cudaFree/cudaFreeAsync success
```

则释放两层预算：

```cpp
release_device_direct_budget(info.size);
release_scratch_pool_device_direct_budget(info.size);
```

并增加：

```text
device_direct_free_success_allocs
scratch_pool_device_direct_free_success_allocs
```

这意味着 Stage G 维护的是独立 scratch pool live bytes，而不是直接复用 Stage C 全局 live bytes。Stage C 负责全局 device-direct 安全上限，Stage G 负责 runtime scratch 自己的 pool 上限。

## 11. 日志字段

Stage G 扩展了这些 trace：

```text
TRACE_POLICY
TRACE_GAP_WATCH_ALLOC
TRACE_GAP_WATCH_FREE
TRACE_POOL_ALLOC
TRACE_POOL_FREE
```

核心字段：

```text
scratch_pool_tracked
scratch_pool_eligible
scratch_pool_device_direct
scratch_pool_over_budget
scratch_pool_reason
scratch_pool_live_bytes
scratch_pool_budget_bytes
scratch_pool_budget_remaining
scratch_pool_mode
scratch_pool_enable
```

这些字段的读法：

1. `scratch_pool_tracked=1`
   - 说明该 allocation 属于 Stage G 观察范围。

2. `scratch_pool_eligible=1`
   - 说明 phase、size、device-direct enable 通过。

3. `scratch_pool_device_direct=1`
   - 说明真实进入 device-direct。

4. `scratch_pool_over_budget=1`
   - 说明如果纳入 device-direct 会超过 Stage G budget。

5. `scratch_pool_live_bytes`
   - 当前 scratch pool device-direct live bytes。

6. `scratch_pool_budget_remaining`
   - 当前预算剩余。预算为 0 时表示 unlimited，remaining 也记为 0。

常见 reason：

```text
not_scratch_pool
scratch_pool_not_evaluated
scratch_pool_phase_not_allowed
scratch_pool_below_min_bytes
scratch_pool_above_max_bytes
scratch_pool_device_direct_disabled
scratch_pool_budget_exceeded_fallback_managed
scratch_pool_trace_only_over_budget_device_direct
scratch_pool_device_direct_enabled
scratch_pool_global_device_direct_budget_exceeded
scratch_pool_malloc_failed_fallback_managed
scratch_pool_malloc_async_failed_fallback_managed
scratch_pool_managed_candidate_free_failed
scratch_pool_device_direct_pool_config_failed
```

## 12. Session summary

allocator session summary 会输出：

```text
Scratch pool enabled
Scratch pool budget bytes
Scratch pool mode
Scratch pool target phases
Scratch pool trace allocations
Scratch pool eligible allocations
Scratch pool device-direct allocations
Scratch pool device-direct bytes
Scratch pool device-direct live bytes
Scratch pool device-direct peak live bytes
Scratch pool budget remaining
Scratch pool budget over allocations
Scratch pool budget reject allocations
Scratch pool device-direct free success
```

这些字段可以在 trace 记录缺失或被过滤时作为 fallback 数据源。

## 13. runner 集成

`workloads/vllm/run_kv_fault_ratio.sh` 新增参数：

```bash
--uvm-scratch-pool-enable <0|1>
--uvm-scratch-pool-budget-bytes <n>
--uvm-scratch-pool-mode <trace_only|enforce>
--uvm-scratch-pool-target-phases <csv>
```

参数默认值：

```text
UVM_SCRATCH_POOL_ENABLE=0
UVM_SCRATCH_POOL_BUDGET_BYTES=1048576
UVM_SCRATCH_POOL_MODE=trace_only
UVM_SCRATCH_POOL_TARGET_PHASES=enabled:attention
```

runner 会进行基本校验：

```text
--uvm-scratch-pool-enable must be 0 or 1
--uvm-scratch-pool-budget-bytes must be a non-negative integer
--uvm-scratch-pool-mode must be trace_only or enforce
--uvm-scratch-pool-target-phases must not be empty
```

启动 vLLM server 时透传为环境变量：

```bash
VLLM_UVM_SCRATCH_POOL_ENABLE
VLLM_UVM_SCRATCH_POOL_BUDGET_BYTES
VLLM_UVM_SCRATCH_POOL_MODE
VLLM_UVM_SCRATCH_POOL_TARGET_PHASES
```

## 14. metrics 汇总

`workloads/vllm/summarize_gap_watch_metrics.py` 新增 `record_scratch_pool_fields()`，从 trace 中解析 `scratch_pool_*` 字段。

输出 JSON 包括：

```json
{
  "scratch_pool_enabled": true,
  "scratch_pool_budget_bytes": 16777216,
  "scratch_pool_mode": "enforce",
  "scratch_pool_trace_records": 3179,
  "scratch_pool_eligible_records": 483,
  "scratch_pool_device_direct_records": 146,
  "scratch_pool_device_direct_bytes": 75515904,
  "scratch_pool_live_bytes": 15853568,
  "scratch_pool_peak_live_bytes": 15853568,
  "scratch_pool_min_budget_remaining_observed": 923648,
  "scratch_pool_budget_over_records": 337,
  "scratch_pool_budget_reject_records": 337,
  "scratch_pool_reason_counts": {
    "scratch_pool_device_direct_enabled": 146,
    "scratch_pool_budget_exceeded_fallback_managed": 337,
    "scratch_pool_phase_not_allowed": 2648,
    "scratch_pool_above_max_bytes": 48
  }
}
```

字段解释：

1. `scratch_pool_trace_records`
   - Stage G 观察到的 runtime scratch records。

2. `scratch_pool_eligible_records`
   - 通过 phase、size、device-direct enable 的 records。

3. `scratch_pool_device_direct_records`
   - 实际进入 device-direct 的 records。

4. `scratch_pool_live_bytes`
   - 最后一条记录看到的 scratch pool device-direct live bytes。

5. `scratch_pool_peak_live_bytes`
   - 运行期间 scratch pool device-direct live bytes 峰值。

6. `scratch_pool_budget_over_records`
   - 会超过 scratch budget 的 records。

7. `scratch_pool_budget_reject_records`
   - enforce 模式下因为超过 scratch budget 被拒绝的 records。

8. `scratch_pool_reason_counts`
   - 解释为什么 allocation 进入、拒绝或未参与 Stage G。

## 15. success check 脚本

Stage G 新增：

```text
workloads/vllm/check_stage_g_success.py
```

默认运行：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./check_stage_g_success.py
```

默认参数：

```text
--budget-bytes 1048576
--mode enforce
--target-phases enabled:attention,enabled:moe,enabled:model_forward
--backend cuda_malloc_async
--min-bytes 4096
--max-bytes 16777216
--pool-release-threshold 1048576
--prompts 1
--request-rate 5
--output-len 512
```

脚本实际调用 `run_kv_fault_ratio.sh` 时会同时开启：

```bash
--uvm-pool-registry-enable 1
--uvm-scratch-pool-enable 1
--uvm-device-direct-enable 1
--uvm-device-direct-backend cuda_malloc_async
--uvm-device-direct-min-bytes 4096
--uvm-device-direct-max-bytes 16777216
--uvm-device-direct-max-total-bytes max(budget, 1048576)
--uvm-device-direct-target-phases <target-phases>
```

也就是说 Stage G check 不是只检查 trace 字段，而是要求 allocator 真的执行一部分 scratch device-direct。

### 15.1 成功检查项

`check_stage_g_success.py` 校验：

```text
metrics_json_present
scratch_pool_enabled
scratch_pool_budget_matches
scratch_pool_mode_matches
scratch_pool_trace_records_present
scratch_pool_eligible_records_present
scratch_pool_device_direct_records_present
scratch_pool_peak_within_budget_in_enforce
scratch_pool_fallback_signal_present_if_over_budget
benchmark_no_failed_requests
runner_log_clean
```

其中最关键的是：

```text
scratch_pool_eligible_records > 0
scratch_pool_device_direct_records > 0
scratch_pool_peak_live_bytes <= budget
failed_requests == 0
```

如果 enforce 模式下有 over-budget records，则必须看到 budget reject records，证明 over-budget 被 fallback managed，而不是默默越界。

### 15.2 复用已有日志验证

如果已经有 allocator log，可以跳过新 benchmark：

```bash
./check_stage_g_success.py \
  --skip-run \
  --allocator-log /tmp/vllm_stage_g_success_check_xxx/vllm_uvm_allocator_trace_stage_g.log \
  --bench-log /tmp/vllm_stage_g_success_check_xxx/vllm_bench_stage_g.log
```

如果传入 `--metrics-json`，脚本会直接读取 JSON。否则会调用 `summarize_gap_watch_metrics.py` 从 allocator log 生成 metrics JSON。

## 16. 推荐验收命令

根据实际实验，默认 1 MiB budget 对当前 Qwen/Qwen3-30B-A3B-FP8 workload 来说太小，可能无法产生 device-direct records。推荐的可通过命令是：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./check_stage_g_success.py \
  --target-phases enabled \
  --budget-bytes 16777216 \
  --max-bytes 16777216
```

这里：

1. `--target-phases enabled`
   - 覆盖所有 `enabled:*` 运行时 phase，避免只开 attention 时错过 runtime scratch。

2. `--budget-bytes 16777216`
   - 给 scratch pool 16 MiB device-direct live budget。

3. `--max-bytes 16777216`
   - 允许最大 16 MiB 的 runtime scratch allocation 进入候选集。

## 17. 实验结果解释

### 17.1 实验一：target phases 太窄导致没有 eligible

早期命令类似：

```bash
./check_stage_g_success.py \
  --target-phases enabled,enabled:attention,enabled:moe,enabled:model_forward,enabled:compute_logits,enabled:sampler \
  --budget-bytes 1048576
```

结果摘要：

```text
scratch_pool_trace_records=3179
scratch_pool_eligible_records=0
scratch_pool_device_direct_records=0
scratch_pool_reason_counts={
  "scratch_pool_phase_not_allowed": 2648,
  "scratch_pool_above_max_bytes": 531
}
```

解释：

1. Stage G 被开启了，也观察到了 3179 条 runtime scratch records。
2. 但没有任何 allocation 同时通过 phase 和 size gate。
3. 大部分被 `scratch_pool_phase_not_allowed` 拒绝。
4. 其余被 `scratch_pool_above_max_bytes` 拒绝。
5. 因为 eligible 为 0，所以不可能有 device-direct records，check 失败是正确的。

这个结果证明“trace 和分类已经工作”，但配置还没有让候选进入准入阶段。

### 17.2 实验二：有 eligible，但预算太小

命令类似：

```bash
./check_stage_g_success.py \
  --target-phases enabled \
  --budget-bytes 1048576 \
  --max-bytes 16777216
```

结果摘要：

```text
scratch_pool_trace_records=3179
scratch_pool_eligible_records=483
scratch_pool_device_direct_records=0
scratch_pool_budget_over_records=483
scratch_pool_budget_reject_records=483
scratch_pool_reason_counts={
  "scratch_pool_phase_not_allowed": 2648,
  "scratch_pool_budget_exceeded_fallback_managed": 483,
  "scratch_pool_above_max_bytes": 48
}
```

解释：

1. `--target-phases enabled` 成功扩大了 phase 覆盖面。
2. `--max-bytes 16777216` 也让更多 runtime scratch 通过 size gate。
3. 现在有 483 条 eligible records，说明 Stage G admission gate 已经能识别候选。
4. 但 1 MiB budget 太小，所有 483 条候选都会超过 budget。
5. enforce 模式下这些 allocation 全部 fallback managed。
6. 因此 device-direct records 仍然是 0，check 失败也是正确的。

这个结果证明“预算拒绝和 fallback managed 已经工作”，但预算不足以允许任何 scratch allocation 进入 device-direct。

### 17.3 实验三：16 MiB budget 成功通过

命令：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./check_stage_g_success.py \
  --target-phases enabled \
  --budget-bytes 16777216 \
  --max-bytes 16777216
```

结果摘要：

```text
Stage G Success Check: PASS
scratch_pool_enabled=True
scratch_pool_budget_bytes=16777216
scratch_pool_mode=enforce
scratch_pool_trace_records=3179
scratch_pool_eligible_records=483
scratch_pool_device_direct_records=146
scratch_pool_live_bytes=15853568
scratch_pool_peak_live_bytes=15853568
scratch_pool_budget_over_records=337
scratch_pool_budget_reject_records=337
scratch_pool_reason_counts={
  "scratch_pool_device_direct_enabled": 146,
  "scratch_pool_budget_exceeded_fallback_managed": 337,
  "scratch_pool_phase_not_allowed": 2648,
  "scratch_pool_above_max_bytes": 48
}
failed_requests=0
```

解释：

1. Stage G 完整开启，预算和 mode 与命令匹配。
2. allocator 观察到 3179 条 runtime scratch records。
3. 其中 483 条通过 phase 和 size gate，成为 Stage G eligible candidates。
4. 146 条成功进入 scratch pool device-direct。
5. 337 条由于 16 MiB budget 已满或会超限，被 enforce 策略拒绝并 fallback managed。
6. `scratch_pool_peak_live_bytes=15853568`，小于 `16777216`，证明预算没有越界。
7. `failed_requests=0`，说明 fallback managed 没有破坏 vLLM 请求。
8. check PASS，说明 Stage G 的准入、预算、device-direct、fallback 和 metrics 闭环都成立。

这个实验是 Stage G 最关键的成功证明。

## 18. 如何调参

如果 `scratch_pool_eligible_records=0`：

1. 扩大 `--target-phases`，例如使用 `enabled`。
2. 检查 `scratch_pool_reason_counts` 是否以 `scratch_pool_phase_not_allowed` 为主。
3. 检查 `--max-bytes` 是否太小。

如果 `scratch_pool_eligible_records>0` 但 `scratch_pool_device_direct_records=0`：

1. 检查是否以 `scratch_pool_budget_exceeded_fallback_managed` 为主。
2. 增大 `--budget-bytes`。
3. 确认 `--uvm-device-direct-enable 1`。
4. 确认 Stage C 全局 `--uvm-device-direct-max-total-bytes` 不小于 scratch budget。

如果 `scratch_pool_device_direct_records>0` 但 check 仍失败：

1. 检查 `scratch_pool_peak_live_bytes` 是否超过 enforce budget。
2. 检查 benchmark 是否有 failed requests。
3. 检查 runner log 是否出现 server early exit 或 parse failure。

如果 `cuda_malloc_async` 相关失败：

1. 尝试 `--backend cuda_malloc` 排除 CUDA mempool 配置问题。
2. 调大或关闭 `--pool-release-threshold`。
3. 查看 reason 是否为 `scratch_pool_device_direct_pool_config_failed` 或 `scratch_pool_malloc_async_failed_fallback_managed`。

## 19. 为什么 Stage G 不做 scratch eviction

runtime scratch 的生命周期通常非常短，经常是 attention、MoE、logits、sampler 等运行阶段的临时 workspace。对这类对象做 eviction 风险高、收益不稳定：

1. 对象可能马上被 kernel 使用，很难安全迁移。
2. 生命周期短，迁移成本可能高于收益。
3. allocator 不知道上层 kernel 是否仍依赖该 workspace。
4. 预算压力出现时，拒绝新对象进入 device-direct 比驱逐旧对象更安全。

因此 Stage G 采用：

```text
admit if safe
fallback if unsafe
free naturally
```

这也是后续 Stage J 处理 KV、Stage I 处理 weights 时需要保持的设计原则：不同 pool 的执行动作必须尊重对象语义，不能在 allocator 里盲目驱逐。

## 20. 后续阶段衔接

Stage G 完成后，系统已经具备：

1. Stage F：统一 pool registry。
2. Stage G：runtime scratch 独立准入控制。

后续建议：

1. Stage H：weights hot/cold trace-only planning。
2. Stage I：MoE expert weight prefetch/offload execution。
3. Stage J：KV runtime eviction/swap via vLLM block manager。
4. Stage K：global pressure coordinator。

Stage G 对后续阶段的价值是提供了一个可参考的执行策略模板：

```text
semantic classification
  -> pool-specific budget
  -> safe action if admitted
  -> managed/no-op fallback if rejected
  -> trace every branch
  -> check script proves behavior
```

后续 Stage I/J/K 应继续沿用这个原则，避免直接把 allocator 变成不理解语义的全局驱逐器。

## 21. 快速阅读路径

如果要从代码验证 Stage G，可按这个顺序阅读：

1. `docs/vllm_uvm_independent_pool_eviction_prefetch_plan.md`
   - 阅读 Stage G 在总体路线中的定位。

2. `workloads/vllm/vllm/uvm_test/uvm_allocator.cpp`
   - 搜索 `scratch_pool_enable` 看配置读取。
   - 搜索 `reserve_scratch_pool_device_direct_budget` 看预算逻辑。
   - 搜索 `scratch_pool_tracked` 看 malloc 路径。
   - 搜索 `scratch_pool_device_direct` 看 free 路径和日志字段。

3. `workloads/vllm/run_kv_fault_ratio.sh`
   - 搜索 `uvm-scratch-pool` 看 runner 参数、校验和环境变量透传。

4. `workloads/vllm/summarize_gap_watch_metrics.py`
   - 搜索 `record_scratch_pool_fields` 看 metrics 汇总逻辑。

5. `workloads/vllm/check_stage_g_success.py`
   - 看 Stage G check 如何配置实验，以及 PASS/FAIL 的判断标准。

## 22. 最终判断

当前 Stage G 已经完整实现了 runtime scratch pool admission control：

1. allocator 能识别 runtime scratch pool。
2. 支持独立 scratch pool budget。
3. 支持 `trace_only` 和 `enforce`。
4. 支持 phase allowlist。
5. 支持 min/max size gate。
6. 能把通过准入的 scratch allocation 接入 Stage C/C2 device-direct。
7. enforce 超预算时 fallback managed。
8. free 路径能释放 scratch pool budget。
9. trace 和 session summary 有完整字段。
10. metrics summarizer 能汇总 Stage G 结果。
11. success check 能一键证明 Stage G 是否生效。

从 16 MiB budget 的 PASS 实验看，Stage G 的行为符合设计：146 条 scratch allocation 进入 device-direct，337 条因预算超限 fallback managed，peak live bytes 保持在 budget 内，benchmark 没有失败请求。

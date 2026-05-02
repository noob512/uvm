# vLLM UVM Stage D / D2 KV Budget 详细实现说明

本文档说明当前项目中 Stage D 的实现方式、代码修改点、运行链路、日志字段和验收方法。它面向后续接手项目的人，目标是让读者能从文档顺着代码理解：KV cache 独立预算是如何被识别、记录、校验和在 D2 中真正限制初始化规模的。

本文不展开原先的 eBPF 部分，只聚焦 vLLM UVM allocator、KV cache 语义预算、runner/check 脚本和相关文档。

## 1. Stage D 要解决什么问题

Stage C 已经证明 allocator 可以对一小类 runtime scratch allocation 做真实 `device_direct`，并能用总预算约束这类 GPU-only allocation。但这仍然不是完整的“内存池分区管理”。

项目下一步目标是把显存/UVM 中的几类核心对象分区管理：

1. KV cache：服务请求过程中长期存在，和 block table、active sequence 绑定。
2. weights：模型权重，通常是持久对象。
3. runtime scratch / workspace：attention、MoE、logits、sampler 等阶段的短生命周期临时对象。

Stage D 聚焦 KV cache。它要回答两个问题：

1. 当前 vLLM 初始化时 KV cache 实际申请了多少字节？
2. 如果给 KV cache 一个独立预算，能不能先观测、再在初始化阶段限制 KV cache 大小？

因此 Stage D 不是继续扩展 Stage C 的 `device_direct`，也不是在 allocator 层做 KV eviction/swap/recompute。Stage D 的核心是“KV cache 独立预算视角”，D2 才进一步把预算接入 vLLM 的 KV cache config 生成路径，让初始化出来的 KV cache tensor bytes 不超过预算。

## 2. 阶段划分

当前项目里的 Stage D 分成两层：

1. Stage D0/D1：allocator-side KV budget telemetry + soft enforce signal。
2. Stage D2：vLLM semantic KV budget enforcement，在 KV cache tensor 真正分配前减少 KV blocks。

D0/D1 和 D2 的区别很关键：

1. D0/D1 只在 allocator 层记录 KV allocation 的 live/peak/budget/reason。即使 `mode=enforce`，allocator 也只输出 soft reject signal，不会返回 NULL，也不会驱逐 KV。
2. D2 在 vLLM 语义层生效。`VLLM_UVM_KV_BUDGET_MODE=enforce` 且 `VLLM_UVM_KV_BUDGET_BYTES>0` 时，vLLM 会在生成 `KVCacheConfig` 时减少可用 KV planning memory，并在最终 config 上做 guardrail，确保 KV tensor bytes 不超过预算。

换句话说，D0/D1 负责“看见并记录”，D2 负责“初始化时真的缩小”。

## 3. 为什么 allocator 不做 KV eviction

allocator 只能看到指针、大小、device、stream、phase 和一部分分类信息。它不知道：

1. 哪个 KV block 属于哪个 request。
2. 当前 block table 如何映射。
3. 哪些 sequence 正在 decode。
4. 哪些 KV block 可以安全 swap。
5. 哪些 token 可以重算。
6. scheduler 当前是否处在可迁移安全点。

如果 allocator 在超预算时直接 free、迁移或拒绝 KV allocation，很容易破坏 vLLM 的 block manager 和 request 状态。因此 D0/D1 的 `enforce` 被设计成 allocator-side soft signal，真正的 KV 缩容放到 D2 的 KV config 层完成。

这也解释了之前实验里 `trace_only` 和 `enforce` 的差异：

1. D0/D1 `trace_only`：超预算只记录 `kv_budget_exceeded_trace_only`。
2. D0/D1 `enforce`：超预算记录 `kv_budget_exceeded_soft_enforce` 和 reject counter，但不阻止 allocation。
3. D2 `enforce`：在 allocation 前缩小 KV blocks，使 allocator 观察到的 KV peak live 本身不超过预算。

## 4. 主要修改文件

### 4.1 `workloads/vllm/vllm/uvm_test/uvm_allocator.cpp`

这是 D0/D1 的核心实现文件，负责识别 KV allocation、维护 KV live/peak 计数、输出 budget 字段，并在 free 时归还 KV live bytes。

Stage D 相关修改包括：

1. 新增配置项：
   - `kv_budget_bytes`
   - `kv_budget_mode`

2. 新增 KV telemetry counters：
   - `kv_trace_allocs`
   - `kv_requested_bytes`
   - `kv_live_bytes`
   - `kv_peak_live_bytes`
   - `kv_budget_over_allocs`
   - `kv_budget_reject_allocs`
   - `kv_free_success_allocs`

3. 新增/扩展 KV budget helper：
   - `normalize_kv_budget_mode()`
   - `is_kv_allocation()`
   - `update_kv_peak_live()`
   - `kv_budget_remaining_snapshot()`
   - `record_kv_allocation()`
   - `release_kv_budget()`

4. 扩展 `AllocationInfo`：
   - `kv_budget_tracked`
   - `kv_budget_over_budget`
   - `kv_budget_reason`

5. 扩展 trace log：
   - `TRACE_POLICY`
   - `TRACE_GAP_WATCH_ALLOC`
   - `TRACE_GAP_WATCH_FREE`
   - Session Summary

6. 扩展 reset/close summary：
   - `uvm_reset_all_stats()` 会清理 KV counters。
   - `uvm_close_log()` 会输出 KV budget summary。

### 4.2 `workloads/vllm/run_kv_fault_ratio.sh`

这是 Stage D/D2 的底层 runner。它负责启动 vLLM server，把 KV budget 参数注入 server 进程，并在 `--no-bench` 模式下只跑到 KV cache 初始化完成。

Stage D 相关修改包括：

1. 新增 runner 变量：
   - `UVM_KV_BUDGET_BYTES`
   - `UVM_KV_BUDGET_MODE`

2. 新增命令行参数：
   - `--uvm-kv-budget-bytes <n>`
   - `--uvm-kv-budget-mode trace_only|enforce`

3. 参数校验：
   - budget bytes 必须是非负整数。
   - mode 必须是 `trace_only` 或 `enforce`。

4. server 环境变量注入：
   - `VLLM_UVM_KV_BUDGET_BYTES`
   - `VLLM_UVM_KV_BUDGET_MODE`

5. 支持 `--gap-watch-metrics-summary-json`：
   - Stage D 不一定需要 gap-watch，但复用 `summarize_gap_watch_metrics.py` 统一聚合 allocator log。

### 4.3 `workloads/vllm/summarize_gap_watch_metrics.py`

该脚本原本用于 gap-watch/Stage C metrics 汇总，现在也承担 Stage D 的 KV budget metrics 聚合。

Stage D 相关修改包括：

1. 新增 `record_kv_budget_fields()`。
2. 从 `TRACE_POLICY` 中解析 KV budget fields。
3. 在 gap-watch alloc/free 路径中也保留 KV 字段，方便未来分析 KV 与 watched gap 的重叠。
4. 输出 KV summary 字段：
   - `kv_budget_bytes`
   - `kv_budget_mode`
   - `kv_trace_allocations`
   - `kv_requested_bytes`
   - `kv_live_bytes`
   - `kv_peak_live_bytes_observed`
   - `kv_min_budget_remaining_observed`
   - `kv_budget_over_records`
   - `kv_budget_reject_records`
   - `kv_budget_reason_counts`

### 4.4 `workloads/vllm/check_stage_d_success.py`

这是 Stage D/D0/D1 的一键验收脚本，也被 D2 wrapper 复用。

默认行为：

1. 启动 vLLM server。
2. 等待 server ready 和 KV range 出现。
3. 配置 UVM params。
4. 默认 `--no-bench`，也就是只验证初始化阶段 KV cache allocation。
5. 汇总 allocator trace。
6. 检查 KV budget metrics。

默认参数：

1. `--budget-bytes 1048576`
2. `--budget-mode trace_only`
3. benchmark 默认关闭。

这个默认设置故意使用 1 MiB 的很小预算，使大模型 KV cache 初始化稳定触发 over-budget signal，从而证明 D0/D1 的 KV budget telemetry 是可见的。因为 mode 是 `trace_only`，所以不会软拒绝，也不会影响 server 正确启动。

### 4.5 `workloads/vllm/check_stage_d2_success.py`

这是 D2 的 wrapper。它调用 `check_stage_d_success.py`，但固定传入：

1. `--budget-mode enforce`
2. 默认 `--budget-bytes 2147483648`，即 2 GiB。

D2 的 PASS 依赖 `check_stage_d_success.py` 中的 enforce 检查：如果 `mode=enforce` 且 budget 非 0，则要求 `kv_peak_live_bytes_observed <= budget_bytes`。

### 4.6 `workloads/vllm/run_stage_d_kv_budget_check.sh`

这是 D0/D1 的 convenience wrapper。它把环境变量转成 `check_stage_d_success.py` 参数：

```bash
KV_BUDGET_BYTES="${KV_BUDGET_BYTES:-1048576}"
KV_BUDGET_MODE="${KV_BUDGET_MODE:-trace_only}"
RUN_BENCH="${RUN_BENCH:-0}"
```

### 4.7 `workloads/vllm/run_stage_d2_kv_budget_check.sh`

这是 D2 的 convenience wrapper。它把环境变量转成 `check_stage_d2_success.py` 参数：

```bash
KV_BUDGET_BYTES="${KV_BUDGET_BYTES:-2147483648}"
RUN_BENCH="${RUN_BENCH:-0}"
```

### 4.8 `workloads/vllm/vllm/vllm/v1/core/kv_cache_utils.py`

这是 D2 的核心实现文件。它不是 allocator，而是 vLLM KV cache config 生成路径。D2 在这里实现“初始化前硬约束”。

Stage D2 相关修改包括：

1. 新增环境变量读取：
   - `_read_uvm_kv_budget_bytes()`
   - `_read_uvm_kv_budget_mode()`
   - `_uvm_kv_budget_enforce_enabled()`

2. 新增 KV config byte 计算：
   - `_kv_cache_config_total_bytes()`

3. 在 block planning 前 cap available memory：
   - `_apply_uvm_kv_budget_to_available_memory()`
   - 对每个 worker 使用 `min(profiled_available_memory, VLLM_UVM_KV_BUDGET_BYTES)`。

4. 在最终 config 上做 guardrail：
   - `_enforce_uvm_kv_budget_on_config()`
   - 如果最终 KV tensor bytes 仍超过预算，则按比例缩小 `num_blocks` 和每个 `kv_cache_tensor.size`。
   - 如果 budget 连一个 KV block 都放不下，则在 KV 初始化前抛出明确 `ValueError`。

5. 在 `get_kv_cache_configs()` 中接入：
   - 先对 `available_memory` 应用 budget cap。
   - 生成每个 worker 的 `KVCacheConfig`。
   - 对每个 config 调用最终 guardrail。

6. 在日志中输出 D2 metadata：
   - mode
   - budget
   - enforced
   - final KV bytes
   - original KV bytes

### 4.9 `workloads/vllm/vllm/vllm/v1/kv_cache_interface.py`

这里扩展了 `KVCacheConfig` 数据结构，用于携带 D2 metadata：

1. `uvm_kv_budget_bytes`
2. `uvm_kv_budget_mode`
3. `uvm_kv_budget_enforced`
4. `uvm_kv_budget_original_num_blocks`
5. `uvm_kv_budget_original_bytes`
6. `uvm_kv_budget_final_bytes`

这些字段不是 allocator 运行时状态，而是 vLLM 语义层生成 KV config 时的证据。

### 4.10 `docs/vllm_uvm_memory_pool_evolution_plan.md`

演进计划文档中记录了 Stage D/D2 的当前状态：

1. D0/D1 已实现 allocator-side KV budget telemetry + soft enforce signal。
2. D2 已实现 vLLM semantic KV budget enforcement。
3. 当前阶段不在 allocator 层做 KV eviction/swap/recompute。

### 4.11 `docs/vllm_uvm_stage_d_kv_budget_telemetry_implementation.md`

这是已有 Stage D / D2 实现记录文档，偏实现摘要和实验说明。本文档在它基础上进一步按“读代码路径”展开。

## 5. D0/D1 allocator 运行流程

下面按一次 KV cache allocation 的执行顺序解释。

### 5.1 读取配置

allocator 初始化日志时读取：

```text
VLLM_UVM_KV_BUDGET_BYTES
VLLM_UVM_KV_BUDGET_MODE
```

`VLLM_UVM_KV_BUDGET_BYTES=0` 表示不限额。

`VLLM_UVM_KV_BUDGET_MODE` 通过 `normalize_kv_budget_mode()` 规范化，只有 `enforce` 会保留为 enforce，其余都回退到 `trace_only`。

### 5.2 识别 KV allocation

allocator 根据 allocation class 判断是否是 KV：

```cpp
static bool is_kv_allocation(AllocationClass alloc_class) {
    return alloc_class == AllocationClass::KvPersistent;
}
```

`KvPersistent` 的来源是 vLLM 初始化 KV cache 时设置的 phase。KV cache tensor 初始化期间，allocator 能看到 `initialize_kv_cache` 这类 phase，并把它归为 `kv_persistent`。

### 5.3 记录 allocation

在 `uvm_malloc()` 中，如果 `kv_budget_tracked=true`，就调用：

```cpp
record_kv_allocation(size, &kv_budget_over_budget, &kv_budget_reason);
```

它做几件事：

1. `kv_trace_allocs += 1`
2. `kv_requested_bytes += size`
3. `kv_live_bytes += size`
4. 更新 `kv_peak_live_bytes`
5. 判断 `kv_budget_bytes > 0 && kv_live_bytes > kv_budget_bytes`
6. 如果超预算，`kv_budget_over_allocs += 1`
7. 如果超预算且 `kv_budget_mode == "enforce"`，`kv_budget_reject_allocs += 1`

注意：D0/D1 的 reject 是 soft reject signal，不是 allocator 返回 NULL。

### 5.4 写入 reason

`record_kv_allocation()` 会给每次 KV allocation 写入 reason：

1. `kv_budget_unlimited`：budget bytes 为 0，不限额。
2. `kv_budget_within_budget`：budget 非 0，当前 KV live bytes 未超过预算。
3. `kv_budget_exceeded_trace_only`：超预算，但 mode 是 `trace_only`。
4. `kv_budget_exceeded_soft_enforce`：超预算，mode 是 `enforce`，allocator 产生软拒绝信号。
5. `not_kv`：不是 KV allocation。

### 5.5 保存 metadata

KV allocation 会被放入 `active_allocations`，因为 `store_active_info` 包含：

```cpp
kv_budget_tracked
```

对应 metadata 包括：

1. `kv_budget_tracked`
2. `kv_budget_over_budget`
3. `kv_budget_reason`

这一步很重要，因为 free 时需要知道这个 pointer 是否是 KV allocation，才能归还 `kv_live_bytes`。

### 5.6 trace 输出

`TRACE_POLICY` 中会输出：

```text
kv_budget_tracked=<0|1>
kv_budget_over_budget=<0|1>
kv_budget_reason=<reason>
kv_live_bytes=<n>
kv_budget_bytes=<n>
kv_budget_remaining=<n>
kv_budget_mode=<trace_only|enforce>
```

`TRACE_GAP_WATCH_ALLOC` 和 `TRACE_GAP_WATCH_FREE` 也带有 KV budget 字段，便于未来把 KV budget 与 gap-watch 统一分析。

### 5.7 free 时归还 KV live bytes

在 `uvm_free()` 中：

```cpp
if (has_info && info.kv_budget_tracked && err == cudaSuccess) {
    release_kv_budget(info.size);
    kv_free_success_allocs.fetch_add(1);
}
```

也就是说，只有 allocator metadata 表明这是 KV allocation，且 CUDA free 成功，才会扣减 `kv_live_bytes`。

### 5.8 Session Summary

正常退出时 `uvm_close_log()` 会输出：

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

不过 Stage D 的 success check 不只依赖 Session Summary。因为 server 运行中也会实时 flush `TRACE_POLICY`，`summarize_gap_watch_metrics.py` 可以从实时 trace 字段聚合出 metrics。

## 6. D2 vLLM 语义层执行流程

D2 发生在 vLLM 生成 KV cache config 的阶段，比 allocator 更早。

### 6.1 读取预算

`kv_cache_utils.py` 中：

```python
_read_uvm_kv_budget_bytes()
_read_uvm_kv_budget_mode()
_uvm_kv_budget_enforce_enabled()
```

只有当：

```text
VLLM_UVM_KV_BUDGET_BYTES > 0
VLLM_UVM_KV_BUDGET_MODE == enforce
```

D2 才启用。

### 6.2 block planning 前 cap available memory

`get_kv_cache_configs()` 开始时会调用：

```python
effective_available_memory = _apply_uvm_kv_budget_to_available_memory(
    available_memory
)
```

对于每个 worker：

```python
capped_memory = min(worker_memory, budget_bytes)
```

这会让后续 vLLM block planning 从一开始就以 KV budget 为上限，而不是先按 profiler 给出的全部可用显存规划，再让 allocator 事后发现超预算。

### 6.3 生成 KVCacheConfig

vLLM 按原有流程：

1. 检查每个 worker 是否有足够 KV memory。
2. merge 各 worker 的 KV cache specs。
3. 生成 KV cache groups。
4. 调用 `get_kv_cache_config_from_groups()` 计算 `num_blocks` 和 tensor sizes。

D2 没有重写 vLLM 的 KV block 规划算法，而是在输入 available memory 和输出 config 两端加约束。

### 6.4 最终 guardrail

生成每个 worker 的 `KVCacheConfig` 后，D2 调用：

```python
_enforce_uvm_kv_budget_on_config(...)
```

它会先计算：

```python
current_bytes = sum(tensor.size for tensor in kv_cache_config.kv_cache_tensors)
```

如果 `current_bytes <= budget_bytes`，直接返回。

如果仍然超过预算，说明可用内存 cap 后仍可能因为 override、alignment 或其它规则被放大。此时 D2 会：

1. 记录原始 `num_blocks` 和原始 bytes。
2. 计算 `bytes_per_block = current_bytes // old_num_blocks`。
3. 如果 budget 连一个 block 都放不下，抛出 `ValueError`，让 server 在 KV 初始化前失败。
4. 计算新的 `new_num_blocks`。
5. 按比例缩小每个 `kv_cache_tensor.size`。
6. 更新 `kv_cache_config.num_blocks`。
7. 写入 D2 metadata。

### 6.5 输出 metadata

D2 会在 `KVCacheConfig` 上写入：

1. `uvm_kv_budget_bytes`
2. `uvm_kv_budget_mode`
3. `uvm_kv_budget_enforced`
4. `uvm_kv_budget_original_num_blocks`
5. `uvm_kv_budget_original_bytes`
6. `uvm_kv_budget_final_bytes`

并在日志中输出类似：

```text
Stage D2 UVM KV budget: mode=enforce budget=... enforced=... final_kv_bytes=... original_kv_bytes=...
```

这些字段用于解释“vLLM 为什么少分配了 KV blocks”。

## 7. 参数和环境变量

### 7.1 runner 参数

```bash
--uvm-kv-budget-bytes <n>
--uvm-kv-budget-mode trace_only|enforce
```

### 7.2 环境变量

```bash
VLLM_UVM_KV_BUDGET_BYTES=<bytes>
VLLM_UVM_KV_BUDGET_MODE=trace_only|enforce
```

### 7.3 推荐实验配置

D0/D1 telemetry：

```bash
--uvm-kv-budget-bytes 1048576
--uvm-kv-budget-mode trace_only
--no-bench
```

D2 enforcement：

```bash
--uvm-kv-budget-bytes 2147483648
--uvm-kv-budget-mode enforce
--no-bench
```

## 8. Metrics 字段如何理解

### 8.1 `kv_budget_bytes`

配置的 KV 独立预算。0 表示不限额。

### 8.2 `kv_budget_mode`

当前模式：

1. `trace_only`：只观测，不产生 soft reject。
2. `enforce`：D0/D1 中产生 soft reject signal；D2 中还会触发 vLLM KV config 缩容。

### 8.3 `kv_trace_allocations`

被识别为 KV cache 的 allocation 数。成功实现 Stage D 时应大于 0。

### 8.4 `kv_live_bytes`

当前仍存活的 KV bytes。对于只启动 server、未退出的 `--no-bench` 检查，KV cache 通常仍然存活，所以该值会接近 peak。

### 8.5 `kv_peak_live_bytes_observed`

本次运行中观察到的 KV live bytes 高水位。D2 enforce 成功的关键证据是：

```text
kv_peak_live_bytes_observed <= kv_budget_bytes
```

### 8.6 `kv_min_budget_remaining_observed`

运行中观察到的最小剩余预算。D2 成功时通常非负；如果预算被刚好用满则可能为 0。

### 8.7 `kv_budget_over_records`

KV allocation 后发现 live bytes 超过预算的记录数。

D0/D1 trace-only 使用很小预算时，预期该值大于 0。

D2 enforce 成功时，预期该值为 0，或者至少不再因为初始化 KV cache 超预算而增长。

### 8.8 `kv_budget_reject_records`

allocator-side soft reject 记录数。它统计 reason 为 `kv_budget_exceeded_soft_enforce` 的记录。

注意：这个字段不是硬拒绝。D0/D1 enforce 里它表示“allocator 看到了超预算且 mode=enforce”，但 allocation 仍然成功。D2 正确缩容后，由于不再超预算，该值通常为 0。

### 8.9 `kv_budget_reason_counts`

reason 分布，用来判断当前实验到底处于哪种状态：

1. `kv_budget_unlimited`：budget 为 0。
2. `kv_budget_within_budget`：KV allocation 在预算内。
3. `kv_budget_exceeded_trace_only`：trace-only 下超预算。
4. `kv_budget_exceeded_soft_enforce`：enforce 下 allocator 观察到超预算。

## 9. 如何运行验收

### 9.1 Stage D0/D1 默认检查

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./check_stage_d_success.py
```

默认行为：

1. budget 为 1 MiB。
2. mode 为 `trace_only`。
3. 不跑 benchmark，只启动 server 到 KV 初始化完成。

预期结果：

1. `Stage D Success Check: PASS`
2. `kv_trace_allocations > 0`
3. `kv_peak_live_bytes_observed > 0`
4. `kv_budget_bytes == 1048576`
5. `kv_budget_mode == trace_only`
6. 因为 1 MiB 小于实际 KV cache，`kv_budget_over_records > 0`
7. 因为是 trace-only，`kv_budget_reject_records == 0`

### 9.2 Stage D2 默认检查

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./check_stage_d2_success.py
```

默认行为：

1. budget 为 2 GiB。
2. mode 为 `enforce`。
3. 不跑 benchmark，只启动 server 到 KV 初始化完成。

预期结果：

1. `Stage D Success Check: PASS`
2. `kv_trace_allocations > 0`
3. `kv_peak_live_bytes_observed > 0`
4. `kv_budget_mode == enforce`
5. `kv_peak_live_bytes_observed <= kv_budget_bytes`
6. `kv_budget_over_records == 0`
7. `kv_budget_reason_counts` 主要是 `kv_budget_within_budget`

### 9.3 wrapper 方式

D0/D1：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./run_stage_d_kv_budget_check.sh
```

D2：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./run_stage_d2_kv_budget_check.sh
```

### 9.4 跑 benchmark

默认检查只验证初始化。如果要带一个轻量 benchmark：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./check_stage_d2_success.py --run-bench --prompts 1
```

或：

```bash
RUN_BENCH=1 PROMPTS=1 ./run_stage_d2_kv_budget_check.sh
```

## 10. 结合已有实验结果解读

### 10.1 D0/D1 trace-only 实验

用户之前给出的 D0/D1 默认检查中：

```text
KV budget bytes: 1048576
KV budget mode: trace_only
kv_trace_allocations=48
kv_live_bytes=6442450944
kv_peak_live_bytes_observed=6442450944
kv_budget_over_records=48
kv_budget_reject_records=0
kv_budget_reason_counts={'kv_budget_exceeded_trace_only': 48}
Stage D Success Check: PASS
```

这说明：

1. allocator 成功识别到 48 次 KV cache allocation。
2. 实际 KV cache live/peak 约 6 GiB。
3. 配置预算只有 1 MiB，所以每个 KV allocation 都触发 over-budget。
4. mode 是 `trace_only`，所以没有 soft reject。
5. 这正是 D0/D1 想验证的内容：KV budget telemetry 能看见 KV cache，能计算 live/peak，能在超预算时输出清晰 reason。

这不表示 KV cache 被限制到 1 MiB。D0/D1 trace-only 本来就不限制初始化大小。

### 10.2 D2 enforce 实验

用户之前给出的 D2 默认检查中：

```text
KV budget bytes: 2147483648
KV budget mode: enforce
kv_trace_allocations=48
kv_live_bytes=2146959360
kv_peak_live_bytes_observed=2146959360
kv_min_budget_remaining_observed=524288
kv_budget_over_records=0
kv_budget_reject_records=0
kv_budget_reason_counts={'kv_budget_within_budget': 48}
Stage D Success Check: PASS
```

这说明：

1. allocator 仍然识别到 48 次 KV allocation，说明 KV telemetry 没丢。
2. KV peak live 约 2.14695936 GiB，低于 2 GiB budget 的字节值 `2147483648`。
3. 剩余预算约 524288 bytes，说明 D2 缩容后非常接近预算但没有超过。
4. `kv_budget_over_records=0`，说明初始化出的 KV cache 没有超过预算。
5. `kv_budget_reason_counts={'kv_budget_within_budget': 48}`，说明每个 KV allocation 都在预算内。

这说明 D2 已经实现了“KV pool 初始化大小受预算约束”。它不是运行时 eviction，也不是 recompute，而是在初始化规划阶段减少 KV blocks。

## 11. 成功标准

### 11.1 D0/D1 成功标准

1. metrics JSON 存在。
2. `kv_trace_allocations > 0`。
3. `kv_peak_live_bytes_observed > 0`。
4. `kv_live_bytes >= 0`。
5. `kv_budget_bytes` 与配置一致。
6. `kv_budget_mode` 与配置一致。
7. 如果预算非 0 且 peak 超预算，则 `kv_budget_over_records > 0`。
8. trace-only 模式下 `kv_budget_reject_records == 0`。
9. runner log 没有 parse failure 或 server exit。

### 11.2 D2 成功标准

1. `VLLM_UVM_KV_BUDGET_MODE=enforce`。
2. `kv_trace_allocations > 0`。
3. `kv_peak_live_bytes_observed > 0`。
4. `kv_peak_live_bytes_observed <= VLLM_UVM_KV_BUDGET_BYTES`。
5. `kv_budget_over_records == 0` 或至少不出现初始化阶段超预算。
6. `kv_budget_reason_counts` 主要为 `kv_budget_within_budget`。
7. 如果预算太小，server 应在 KV 初始化前明确失败，而不是 allocator 层静默破坏状态。

## 12. 常见失败模式和排查

### 12.1 `kv_trace_allocations` 为 0

可能原因：

1. UVM allocator 没启用。
2. allocator log 没正确传入。
3. vLLM 没跑到 KV cache 初始化阶段。
4. phase 标记缺失，KV allocation 没被分类为 `kv_persistent`。
5. summary 解析了错误或旧的 allocator log。

排查：

1. 看 server log 是否启动到 ready。
2. 看 address log 是否有 `kv_cache` range。
3. 看 allocator log 是否有 `TRACE_POLICY`。
4. 搜索 `phase=initialize_kv_cache` 或 `predicted_class=kv_persistent`。

### 12.2 trace-only 下没有 over-budget

如果使用默认 1 MiB budget，正常大模型应该 over-budget。如果没有，可能是：

1. 实际 budget 参数没传进去。
2. `kv_budget_bytes` 被解析成 0。
3. metrics JSON 来自旧运行。
4. 模型或配置产生的 KV cache 很小。

排查：

1. 检查 metrics 中 `kv_budget_bytes`。
2. 检查 `kv_peak_live_bytes_observed`。
3. 检查 `kv_budget_reason_counts` 是否为 `kv_budget_unlimited`。

### 12.3 D2 enforce 下 peak 超预算

这说明 D2 语义层约束没有生效或被后续规则放大。

排查：

1. server log 是否出现 `Stage D2 KV budget enforce`。
2. `VLLM_UVM_KV_BUDGET_MODE` 是否为 `enforce`。
3. `VLLM_UVM_KV_BUDGET_BYTES` 是否大于 0。
4. `KVCacheConfig` 是否有 D2 metadata。
5. 是否存在 `num_gpu_blocks_override` 或 alignment 规则导致最终 tensor size 被放大。

### 12.4 D2 budget 太小导致启动失败

这是预期保护。如果 budget 连一个 KV block 都放不下，D2 会抛出明确错误，避免创建无效 KV cache。

解决方式：

1. 增大 `KV_BUDGET_BYTES`。
2. 降低模型并发或上下文配置。
3. 调整 block size 或相关 vLLM KV 配置。

## 13. Stage D 与 Stage C/E 的关系

Stage C 的 device-direct budget 和 Stage D 的 KV budget 是两套独立机制：

1. Stage C 约束 runtime scratch 的真实 GPU-only backend live bytes。
2. Stage D 约束 KV cache 语义池。
3. Stage D/D2 不会把 KV cache 改成 `device_direct`。
4. Stage D/D2 不会改变 Stage C 的 backend 选择。

Stage E 在 D2 之后继续把 weights 纳入独立预算和语义地图。三者的关系可以理解为：

1. KV pool：Stage D/D2，已有初始化预算硬约束和 allocator telemetry。
2. Weight pool：Stage E，已有 weights budget telemetry 和权重语义地图。
3. Scratch pool：Stage C/C2，已有 attention runtime scratch 的 opt-in device-direct backend 和总预算。

后续真正的分区驱逐/预取策略应当基于这些语义池分别决策，而不是让一个全局 UVM 策略同时处理所有对象。

## 14. 修改点速查表

| 文件 | Stage D 作用 |
| --- | --- |
| `workloads/vllm/vllm/uvm_test/uvm_allocator.cpp` | D0/D1 核心：KV allocation 识别、live/peak 计数、budget reason、soft enforce signal、free 回收 |
| `workloads/vllm/run_kv_fault_ratio.sh` | 注入 KV budget 参数，启动 vLLM，生成 allocator trace 和 metrics summary |
| `workloads/vllm/summarize_gap_watch_metrics.py` | 从 allocator trace 聚合 KV budget metrics |
| `workloads/vllm/check_stage_d_success.py` | D0/D1 一键检查，也作为 D2 验收基础 |
| `workloads/vllm/check_stage_d2_success.py` | D2 enforce wrapper，固定使用 enforce 模式 |
| `workloads/vllm/run_stage_d_kv_budget_check.sh` | D0/D1 convenience wrapper |
| `workloads/vllm/run_stage_d2_kv_budget_check.sh` | D2 convenience wrapper |
| `workloads/vllm/vllm/vllm/v1/core/kv_cache_utils.py` | D2 核心：KV planning memory cap 和最终 KVCacheConfig guardrail |
| `workloads/vllm/vllm/vllm/v1/kv_cache_interface.py` | 扩展 KVCacheConfig，保存 D2 metadata |
| `docs/vllm_uvm_stage_d_kv_budget_telemetry_implementation.md` | 既有 Stage D/D2 实现摘要 |
| `docs/vllm_uvm_memory_pool_evolution_plan.md` | 记录 Stage D/D2 在整体 memory pool 演进中的位置 |

## 15. 最小阅读路径

如果只想快速理解 Stage D，建议按下面顺序读：

1. 本文档第 1 到 8 节，先建立 D0/D1 与 D2 的边界。
2. `workloads/vllm/check_stage_d_success.py`，理解验收条件。
3. `workloads/vllm/run_kv_fault_ratio.sh`，理解参数如何传给 vLLM server。
4. `workloads/vllm/vllm/uvm_test/uvm_allocator.cpp` 中 `kv_budget_*` 变量、`record_kv_allocation()`、`release_kv_budget()`、`uvm_malloc()`、`uvm_free()`。
5. `workloads/vllm/summarize_gap_watch_metrics.py` 中 `record_kv_budget_fields()` 和 KV summary 输出。
6. `workloads/vllm/vllm/vllm/v1/core/kv_cache_utils.py` 中 `_apply_uvm_kv_budget_to_available_memory()` 和 `_enforce_uvm_kv_budget_on_config()`。
7. `workloads/vllm/vllm/vllm/v1/kv_cache_interface.py` 中 D2 metadata 字段。

读完这条路径，就能判断一次 Stage D 实验是“telemetry 未接上”“trace-only 正常超预算”“D2 enforce 成功缩容”还是“D2 budget 太小导致合理失败”。

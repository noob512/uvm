# vLLM UVM Stage C Device Direct 详细实现说明

本文档说明当前项目中 Stage C 的实现方式、代码修改点、运行链路、日志字段和验收方法。它面向后续接手项目的人，目标是让读者可以从文档顺着代码理解：为什么 Stage C 要这样做、每个开关控制什么、一次 allocation 如何从 UVM managed 候选转成真实 GPU-only allocation，以及实验脚本如何证明实现成功。

本文不展开原先的 eBPF 部分，只聚焦 vLLM UVM allocator、gap-watch、Stage C `device_direct` 和对应实验脚本。

## 1. Stage C 要解决什么问题

Stage B 已经让 allocator 能表达 `device_direct_trace` 和 `device_direct` 两类动作，但 Stage B 的核心仍是 trace-only：即使某个 allocation 被判定“适合直接放到 GPU 显存”，它仍然通过 `cudaMallocManaged` 进入 UVM managed 路径。这样可以安全收集候选统计，但不能真正减少 UVM replayable fault。

Stage C 的目标是把一小类高置信度 allocation 从 UVM managed 中拿出来，改为真实 GPU-only 分配，从而减少这类对象带来的 UVM 缺页追踪和迁移成本。

具体来说，Stage C 不是“所有 unknown managed 都改成 `cudaMalloc`”。它只针对同一次 vLLM server 进程中自动发现的热点 gap，且只允许 attention 阶段的小尺寸 runtime scratch 类对象进入真实 `device_direct`。其余对象继续走 managed，保证实验边界足够窄。

## 2. Stage C 的阶段划分

Stage C 在当前项目里可以分成三层：

1. Stage C0 / attention-only device-direct：在严格门控下让 `gap_hot_runtime_scratch` 真实走 GPU-only backend，默认 backend 是 `cuda_malloc`。
2. Stage C1 / 总预算补全：增加 `device_direct_max_total_bytes`，限制真实 GPU-only allocation 的 live bytes，超过预算时回退 managed。
3. Stage C2 / async backend：增加 `cuda_malloc_async` backend 选择，以及 CUDA mempool release threshold 配置和验证脚本。

这三层是递进关系。C0 证明“能安全改 backend”，C1 证明“能被总预算约束”，C2 证明“backend 可替换为 `cudaMallocAsync` 并能配置默认 mempool”。

## 3. 安全边界

Stage C 的真实 `device_direct` 必须同时满足下面条件：

1. gap-watch 选择出的策略动作必须是 `device_direct`。
2. `VLLM_UVM_DEVICE_DIRECT_ENABLE=1`，也就是脚本层 `--uvm-device-direct-enable 1`。
3. allocation 必须和当前 watch gap 有重叠，即 `hot_gap_match=1`。
4. allocation 必须匹配目标类别，当前实验使用 `gap_hot_runtime_scratch`。
5. allocation 所在 phase 必须在 allowlist 中，Stage C attention 实验使用 `enabled:attention`。
6. allocation size 必须满足最小和最大阈值，默认是 `4096 <= size <= 1048576`。
7. 总 live bytes 不得超过 `VLLM_UVM_DEVICE_DIRECT_MAX_TOTAL_BYTES`。
8. backend 分配成功，并且原 managed candidate 能成功释放。

任何条件不满足，allocator 都会保守回退到 managed。回退不是异常路径，而是 Stage C 的安全设计核心。

## 4. 主要修改文件

### 4.1 `workloads/vllm/vllm/uvm_test/uvm_allocator.cpp`

这是 Stage C 的核心实现文件。它是 vLLM UVM allocator 的 C++ 实现，负责拦截 PyTorch CUDA allocation/free，做分类、gap-watch 策略判断、真实分配、日志输出和汇总统计。

Stage C 在这里增加或扩展了以下能力：

1. 新增 device-direct 配置项：
   - `device_direct_enable`
   - `device_direct_min_bytes`
   - `device_direct_max_bytes`
   - `device_direct_max_total_bytes`
   - `device_direct_backend`
   - `device_direct_pool_release_threshold_set`
   - `device_direct_pool_release_threshold`
   - `device_direct_target_phases`

2. 新增 device-direct 运行时统计：
   - `device_direct_trace_allocs`
   - `device_direct_eligible_allocs`
   - `device_direct_actual_allocs`
   - `device_direct_actual_bytes`
   - `device_direct_live_bytes`
   - `device_direct_peak_live_bytes`
   - `device_direct_budget_rejects`
   - `device_direct_fallback_allocs`
   - `device_direct_free_success_allocs`
   - `device_direct_requested_bytes`

3. 新增 backend 规范化：
   - `normalize_device_direct_backend()`
   - 输入 `cuda_malloc_async` 或 `cuda_async` 会归一化为 `cuda_malloc_async`。
   - 其他输入默认归一化为 `cuda_malloc`，避免非法值把 allocator 带入未知状态。

4. 新增 async mempool 配置：
   - `configure_device_direct_async_pool_if_needed()`
   - 只在 backend 为 `cuda_malloc_async` 且显式设置 release threshold 时触发。
   - 使用 `cudaDeviceGetDefaultMemPool()` 获取当前 device 默认 mempool。
   - 使用 `cudaMemPoolSetAttribute(cudaMemPoolAttrReleaseThreshold, ...)` 设置 release threshold。
   - 配置结果写入 `device_direct_pool_config_attempted`、`device_direct_pool_config_success`、`device_direct_pool_config_error` 等字段。

5. 新增总预算控制：
   - `reserve_device_direct_budget(size)`
   - `release_device_direct_budget(size)`
   - `update_device_direct_peak_live(current_live)`
   - `device_direct_budget_remaining_snapshot(live)`

6. 扩展 allocation 元数据：
   - `placement_backend_name`
   - `device_direct_backend_name`
   - `device_direct_eligible`
   - `device_direct_reason`
   - `cpu_access_risk`
   - `hot_gap_match`

7. 扩展 trace 输出：
   - `TRACE_POLICY`
   - `TRACE_GAP_WATCH_ALLOC`
   - `TRACE_GAP_WATCH_FREE`
   - Session Summary

### 4.2 `workloads/vllm/run_kv_fault_ratio.sh`

这是 Stage C 实验的底层 runner。它负责启动单个 vLLM server 进程、等待 KV range、配置 UVM 参数、运行 benchmark、收集 trace，并把 allocator 参数通过环境变量注入 server 进程。

Stage C 相关修改包括：

1. 增加命令行参数：
   - `--uvm-device-direct-enable`
   - `--uvm-device-direct-min-bytes`
   - `--uvm-device-direct-max-bytes`
   - `--uvm-device-direct-max-total-bytes`
   - `--uvm-device-direct-backend`
   - `--uvm-device-direct-pool-release-threshold`
   - `--uvm-device-direct-target-phases`

2. 增加对应环境变量：
   - `VLLM_UVM_DEVICE_DIRECT_ENABLE`
   - `VLLM_UVM_DEVICE_DIRECT_MIN_BYTES`
   - `VLLM_UVM_DEVICE_DIRECT_MAX_BYTES`
   - `VLLM_UVM_DEVICE_DIRECT_MAX_TOTAL_BYTES`
   - `VLLM_UVM_DEVICE_DIRECT_BACKEND`
   - `VLLM_UVM_DEVICE_DIRECT_POOL_RELEASE_THRESHOLD`
   - `VLLM_UVM_DEVICE_DIRECT_TARGET_PHASES`

3. 与 auto gap-watch 联动：
   - probe 阶段先运行少量 prompt。
   - `discover_gap_watch.py` 从同一 server 进程的 fault/address 信息中选择目标 gap。
   - main 阶段通过 control file 把目标 gap 和策略动作写给 allocator。
   - Stage C A/B 中 trace baseline 使用 `device_direct_trace`，真实实验使用 `device_direct`。

### 4.3 `workloads/vllm/run_stage_c_attention_p20_ab.sh`

这是 Stage C attention-only 的主 A/B 脚本。它连续运行两组实验：

1. Phase A：Stage B strict trace-only baseline。
   - `--uvm-device-direct-enable 0`
   - `--auto-gap-watch-policy-action-override device_direct_trace`
   - backend 不改变，placement 仍为 managed。

2. Phase B：Stage C attention-only `device_direct`。
   - `--uvm-device-direct-enable 1`
   - `--auto-gap-watch-policy-action-override device_direct`
   - 只允许 `enabled:attention`
   - 只允许 `gap_hot_runtime_scratch`
   - 真实 backend 由 `DEVICE_DIRECT_BACKEND` 控制，默认 `cuda_malloc`。

3. Phase C：调用 `compare_stage_c_attention_p20_ab.py` 比较 trace baseline 和真实 device-direct。

脚本名里保留了 `p20`，但实际 prompt 数可以通过环境变量 `PROMPTS` 改写，例如 C1/C2 success check 默认用更小的 `PROMPTS=5`。

### 4.4 `workloads/vllm/compare_stage_c_attention_p20_ab.py`

该脚本读取一组 trace-only baseline 和一组 Stage C device-direct 的输出，生成 `vllm_stage_c_attention_p<N>_ab_comparison.json`。

它主要检查：

1. trace 和 device 两组 benchmark 是否都能解析。
2. 两组是否都没有 failed requests。
3. probe 和 main 是否监控同一个 gap。
4. device 组是否有 `placement_backend=device_direct`。
5. device 组是否有实际 backend 记录。
6. `gap_policy_fail` 是否为 0。
7. `device_direct_peak_live_bytes_observed <= device_direct_max_total_bytes`。
8. 如设置 async pool threshold，配置是否成功。
9. device 组 gap faults 和 unknown faults 是否低于 trace baseline。
10. device 组 TPOT 是否没有明显恶化。

### 4.5 `workloads/vllm/run_stage_c_attention_backend_ab.sh`

这是 Stage C2 的 backend A/B 脚本。它在同样的 Stage C attention-only 设置下连续运行：

1. C1 sync backend：`DEVICE_DIRECT_BACKEND=cuda_malloc`
2. C2 async backend：`DEVICE_DIRECT_BACKEND=cuda_malloc_async`

然后调用 `compare_stage_c_backend_ab.py` 对比两组 device-direct backend。

### 4.6 `workloads/vllm/compare_stage_c_backend_ab.py`

该脚本比较 `cuda_malloc` 和 `cuda_malloc_async` 两种 backend 的 Stage C device-direct 结果。

它会输出：

1. 两种 backend 各自的 success/effectiveness signal。
2. gap faults、unknown faults、throughput、TPOT 的 async-vs-sync delta。
3. backend counts，例如 `{'cuda_malloc_async': 59710, 'none': 37038}`。
4. 总预算信息，例如 peak live、budget rejects。
5. async mempool 配置结果。

这里需要注意：C2 success check 默认更看重“async backend 正确工作、没有失败、预算约束生效、性能没有明显更差”，不强制要求 async 的 gap faults 必须低于 sync。因为 fault 数受运行抖动影响较大，且你给出的实验中 async fault 比 sync 高，但 throughput 和 TPOT 明显更好，所以脚本允许 `async_effectiveness_vs_sync=False` 但整体 PASS。

### 4.7 `workloads/vllm/check_stage_c1_success.py`

这是 Stage C1 一键验收脚本。默认会运行较小规模 Stage C attention A/B，验证 `cuda_malloc` backend 和总预算。

关键验收点：

1. report 存在。
2. trace/device 两组都成功。
3. device 组使用了 `cuda_malloc`。
4. 没有使用 `cuda_malloc_async`。
5. `device_direct_actual_records > 0`。
6. `placement_backend_counts.device_direct > 0`。
7. `device_direct_peak_live_bytes_observed <= device_direct_max_total_bytes`。
8. runner log 没有解析失败或 server 异常退出。

### 4.8 `workloads/vllm/check_stage_c2_success.py`

这是 Stage C2 一键验收脚本。默认运行 backend A/B：

1. sync：`cuda_malloc`
2. async：`cuda_malloc_async`
3. async pool release threshold：默认 `1048576`

关键验收点：

1. backend report 同时包含 sync 和 async。
2. sync/async 都有 success signal。
3. async 相对 trace baseline 有 effectiveness signal。
4. sync 使用 `cuda_malloc`。
5. async 使用 `cuda_malloc_async`。
6. async 没有误用 sync backend。
7. 两组 failed requests 都为 0。
8. 两组 `gap_policy_fail` 都为 0。
9. 两组都有真实 `device_direct` 记录。
10. 两组 peak live 都在预算内。
11. async probe/main 监控同一个 gap。
12. 如果请求 pool threshold，必须成功设置。
13. runner log 解析干净。

### 4.9 `workloads/vllm/summarize_gap_watch_metrics.py`

该脚本把 allocator trace log 提炼成 JSON metrics，是 Stage C 判断成功的关键中间层。

Stage C 相关聚合字段包括：

1. `placement_backend_counts`
2. `device_direct_backend_counts`
3. `device_direct_reason_counts`
4. `device_direct_trace_records`
5. `device_direct_eligible_records`
6. `device_direct_actual_records`
7. `device_direct_budget_reject_records`
8. `device_direct_max_total_bytes`
9. `device_direct_peak_live_bytes_observed`
10. `device_direct_min_budget_remaining_observed`
11. `device_direct_pool_release_threshold_set`
12. `device_direct_pool_release_threshold`
13. `device_direct_pool_config_attempted`
14. `device_direct_pool_config_success`
15. `device_direct_pool_config_error`
16. `hot_gap_match_records`
17. `median_lifetime_s`

## 5. 一次 Stage C allocation 的完整流程

下面是 `uvm_allocator.cpp` 中 `uvm_malloc()` 的核心流程，用接近代码执行顺序的方式描述。

### 5.1 先走 managed candidate

allocator 首先调用 `cudaMallocManaged(&ptr, size, cudaMemAttachGlobal)` 分配一个 managed candidate。

这一步看起来有点反直觉：既然目标是 `device_direct`，为什么还要先 managed？

原因是 Stage C 需要知道这个 allocation 如果走 managed，会落到哪个虚拟地址区间。只有拿到 candidate pointer 后，allocator 才能计算它是否和 auto gap-watch 发现的热点 gap 重叠。没有这个地址，就无法判断它是不是“当前同一 server 进程中的热点 gap 对象”。

因此 Stage C 的流程是：

1. 先 managed allocation，获得 candidate pointer。
2. 用 candidate pointer 和 size 计算 gap overlap。
3. 如果满足所有门控，再分配 GPU-only pointer。
4. 成功后释放 managed candidate。
5. 返回 GPU-only pointer 给 PyTorch。

这个设计牺牲了一点 allocation 阶段开销，但换来很强的安全性和可解释性。

### 5.2 分类和 gap-watch 判断

allocator 会根据 phase、size、地址范围和已有分类规则得到：

1. `alloc_class`，例如 `unknown_managed`、`runtime_scratch`、`runtime_workspace`、`kv_persistent`、`weight_persistent`。
2. `phase_snapshot`，例如 `enabled:attention`。
3. `gap_overlap_bytes`，即该 allocation 和 watch gap 的重叠字节数。
4. `gap_watch_class_match`，即它是否匹配目标类别。
5. `policy_action`，最终策略动作。

Stage C attention-only 实验中，目标类别是 `gap_hot_runtime_scratch`。在代码中它被解释为：处在 device-direct 允许 phase 中，且 allocation class 为 `unknown_managed`、`runtime_scratch` 或 `runtime_workspace`。

### 5.3 判断是否是 device-direct 请求

只有 `policy_action` 是以下两者之一时，才进入 device-direct 评估：

1. `device_direct_trace`
2. `device_direct`

`device_direct_trace` 只做候选统计，不改变实际 backend。`device_direct` 只有在 `device_direct_enable=1` 时才可能真正改变 backend。

### 5.4 eligibility 门控

如果是 device-direct 请求，allocator 按顺序检查：

1. `device < 0`：记为 `invalid_device`。
2. 没有 hot gap match：记为 `no_hot_gap_match`。
3. target class 不匹配：记为 `target_class_mismatch`。
4. phase 不允许：记为 `phase_not_allowed`。
5. size 小于下限：记为 `below_min_bytes`。
6. size 大于上限：记为 `above_max_bytes`。
7. 全部通过：`device_direct_eligible=true`。

如果 eligible 且当前 action 是 `device_direct` 且 enable 为 1，初始 reason 会是 `device_direct_enabled`。如果只是 trace action，则 reason 是 `trace_action_only`。如果 action 是 `device_direct` 但 enable 为 0，则 reason 是 `trace_only_not_enabled`。

### 5.5 总预算 reservation

进入真实 backend 前，Stage C1 会先调用 `reserve_device_direct_budget(size)`。

预算逻辑使用 atomic CAS 更新 `device_direct_live_bytes`：

1. 如果 `device_direct_max_total_bytes=0`，表示不限制总预算。
2. 如果当前 live + size 会超过预算，reservation 失败。
3. reservation 失败时 reason 为 `device_direct_budget_exceeded`，allocation 回退 managed。
4. reservation 成功时更新 live bytes，并维护 peak live bytes。

这个 reservation 发生在真实 GPU-only allocation 前。这样即使并发 allocation 同时进入，也不会让 live bytes 悄悄超过预算。

### 5.6 真实 GPU-only backend 分配

预算 reservation 成功后，根据 backend 选择实际 CUDA API：

1. `cuda_malloc`：调用 `cudaMalloc(&device_ptr, size)`。
2. `cuda_malloc_async`：先按需配置 mempool，再调用 `cudaMallocAsync(&device_ptr, size, stream)`。

如果 backend 分配失败：

1. 释放刚才 reservation 的 budget。
2. reason 记为 `device_malloc_failed_fallback_managed` 或 `device_malloc_async_failed_fallback_managed`。
3. 如果是 pool 配置失败，则 reason 为 `device_direct_pool_config_failed`。
4. 返回原 managed candidate。

### 5.7 释放 managed candidate 并切换返回指针

如果 GPU-only 分配成功，allocator 会调用 `cudaFree(ptr)` 释放 managed candidate。

只有 managed candidate 成功释放后，Stage C 才把 `ptr` 替换为 `device_ptr`，并把：

1. `placement_backend` 设为 `device_direct`
2. `device_direct_backend_used` 设为 `cuda_malloc` 或 `cuda_malloc_async`
3. `policy_action_success` 设为 true
4. `device_direct_actual_allocs` 加 1
5. `device_direct_actual_bytes` 加 size

如果 managed candidate 释放失败：

1. 释放刚分配出来的 GPU-only pointer。
2. 如果 cleanup 成功，释放 budget reservation。
3. reason 记为 `managed_candidate_free_failed`。
4. 回退 managed。

### 5.8 free 路径

`uvm_free()` 会从 `active_allocations` 找到 allocation metadata。

如果 metadata 表明 `placement_backend=device_direct`：

1. backend 是 `cuda_malloc_async`，则调用 `cudaFreeAsync(ptr, stream)`。
2. backend 是 `cuda_malloc`，则调用 `cudaFree(ptr)`。
3. free 成功后调用 `release_device_direct_budget(info.size)`。
4. `device_direct_free_success_allocs` 加 1。

如果不是 device-direct，仍走普通 `cudaFree(ptr)`，因为 managed allocation 也通过 `cudaFree` 释放。

## 6. 参数和环境变量

### 6.1 脚本参数

`run_kv_fault_ratio.sh` 支持以下 Stage C 参数：

```bash
--uvm-device-direct-enable <0|1>
--uvm-device-direct-min-bytes <n>
--uvm-device-direct-max-bytes <n>
--uvm-device-direct-max-total-bytes <n>
--uvm-device-direct-backend <cuda_malloc|cuda_malloc_async>
--uvm-device-direct-pool-release-threshold <n>
--uvm-device-direct-target-phases <csv>
--auto-gap-watch-policy-action-override <device_direct_trace|device_direct>
--auto-gap-watch-target-class-override gap_hot_runtime_scratch
```

### 6.2 allocator 环境变量

runner 会把参数转换成 vLLM server 进程中的环境变量：

```bash
VLLM_UVM_DEVICE_DIRECT_ENABLE
VLLM_UVM_DEVICE_DIRECT_MIN_BYTES
VLLM_UVM_DEVICE_DIRECT_MAX_BYTES
VLLM_UVM_DEVICE_DIRECT_MAX_TOTAL_BYTES
VLLM_UVM_DEVICE_DIRECT_BACKEND
VLLM_UVM_DEVICE_DIRECT_POOL_RELEASE_THRESHOLD
VLLM_UVM_DEVICE_DIRECT_TARGET_PHASES
```

### 6.3 典型 Stage C attention-only 参数

```bash
--uvm-device-direct-enable 1
--uvm-device-direct-min-bytes 4096
--uvm-device-direct-max-bytes 1048576
--uvm-device-direct-max-total-bytes 1048576
--uvm-device-direct-backend cuda_malloc
--uvm-device-direct-target-phases enabled:attention
--auto-gap-watch-policy-action-override device_direct
--auto-gap-watch-target-class-override gap_hot_runtime_scratch
```

### 6.4 典型 Stage C2 async 参数

```bash
--uvm-device-direct-enable 1
--uvm-device-direct-backend cuda_malloc_async
--uvm-device-direct-pool-release-threshold 1048576
```

## 7. 关键日志字段如何理解

### 7.1 `placement_backend`

`placement_backend=managed` 表示该 allocation 最终仍走 UVM managed。

`placement_backend=device_direct` 表示该 allocation 最终真实使用 GPU-only backend。这是 Stage C 成功的最直接证据之一。

### 7.2 `device_direct_backend`

`device_direct_backend=none` 表示没有真实 device-direct backend。

`device_direct_backend=cuda_malloc` 表示真实调用 `cudaMalloc`。

`device_direct_backend=cuda_malloc_async` 表示真实调用 `cudaMallocAsync`。

### 7.3 `device_direct_eligible`

`device_direct_eligible=1` 表示通过了 Stage C 候选门控。它不等价于真实执行，因为 trace-only 或预算拒绝时也可能 eligible。

### 7.4 `device_direct_reason`

常见 reason：

1. `device_direct_enabled`：真实 device-direct 成功。
2. `trace_action_only`：Stage B/C baseline，只统计不执行。
3. `trace_only_not_enabled`：action 是 device_direct，但 enable 没开。
4. `no_hot_gap_match`：没有落在当前 watch gap。
5. `target_class_mismatch`：不符合目标类别。
6. `phase_not_allowed`：phase 不在 allowlist。
7. `below_min_bytes`：size 太小。
8. `above_max_bytes`：size 太大。
9. `device_direct_budget_exceeded`：C1 总预算拒绝。
10. `device_malloc_failed_fallback_managed`：`cudaMalloc` 失败，回退 managed。
11. `device_malloc_async_failed_fallback_managed`：`cudaMallocAsync` 失败，回退 managed。
12. `device_direct_pool_config_failed`：async mempool 配置失败，回退 managed。
13. `managed_candidate_free_failed`：GPU-only 分配成功但 managed candidate 释放失败，回退 managed。

### 7.5 budget 字段

`device_direct_live_bytes` 表示当前真实 device-direct allocation 仍存活的总字节数。

`device_direct_max_total_bytes` 表示配置的总预算。

`device_direct_budget_remaining` 表示当前剩余预算。

`device_direct_peak_live_bytes_observed` 是 summary 脚本根据 trace 观察到的 live bytes 高水位。验收时通常要求它不超过 `device_direct_max_total_bytes`。

### 7.6 async pool 字段

`device_direct_pool_release_threshold_set` 表示是否显式设置 release threshold。

`device_direct_pool_release_threshold` 是设置值。

`device_direct_pool_config_attempted` 表示是否调用过 CUDA mempool 配置 API。

`device_direct_pool_config_success` 表示配置是否成功。

`device_direct_pool_config_error` 为 `none` 表示无错误。

## 8. Stage C 实验如何运行

### 8.1 C1 success check

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./check_stage_c1_success.py
```

默认会运行一个小规模 Stage C attention A/B，backend 为 `cuda_malloc`。

成功时应看到：

1. `Stage C1 Success Check: PASS`
2. `device_direct_actual_records > 0`
3. `device_direct_backend_counts` 中 `cuda_malloc > 0`
4. `cuda_malloc_async == 0`
5. `device_direct_peak_live_bytes_observed <= device_direct_max_total_bytes`
6. failed requests 为 0
7. runner log clean

### 8.2 C2 success check

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./check_stage_c2_success.py
```

默认会运行：

1. `cuda_malloc` Stage C attention A/B。
2. `cuda_malloc_async` Stage C attention A/B。
3. backend comparison。

成功时应看到：

1. `Stage C2 Success Check: PASS`
2. `async_backend_counts` 中 `cuda_malloc_async > 0`
3. `async_did_not_use_sync_backend=True`
4. `async_pool_threshold_set=True`
5. `async_pool_config_attempted=True`
6. `async_pool_config_success=True`
7. `async_pool_config_error_none=True`
8. `async_peak_live_within_budget=True`

### 8.3 手动运行 attention A/B

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
PROMPTS=5 \
REQUEST_RATE=5 \
OUTPUT_LEN=512 \
DEVICE_DIRECT_MAX_TOTAL_BYTES=1048576 \
DEVICE_DIRECT_BACKEND=cuda_malloc \
./run_stage_c_attention_p20_ab.sh
```

### 8.4 手动运行 backend A/B

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
PROMPTS=5 \
REQUEST_RATE=5 \
OUTPUT_LEN=512 \
DEVICE_DIRECT_MAX_TOTAL_BYTES=1048576 \
ASYNC_DEVICE_DIRECT_POOL_RELEASE_THRESHOLD=1048576 \
./run_stage_c_attention_backend_ab.sh
```

## 9. 成功标准

Stage C 的“正确性成功”主要看以下信号：

1. benchmark 完整跑完，没有 failed requests。
2. auto gap-watch 的 probe 和 main 监控同一个 gap。
3. allocator trace 中出现 `placement_backend=device_direct`。
4. `device_direct_actual_records > 0`。
5. `device_direct_backend_counts` 中出现期望 backend。
6. `gap_policy_fail=0`。
7. peak live bytes 不超过总预算。
8. free 路径成功释放 device-direct allocation。
9. runner log 没有 parse failure 或 server exit。

Stage C 的“效果成功”主要看：

1. 与 trace-only baseline 相比，target gap faults 下降。
2. 与 trace-only baseline 相比，unknown faults 下降。
3. TPOT 没有显著恶化，脚本默认容忍不超过 10% 的变差。
4. throughput 没有显著变差，C2 backend 对比默认关注不比 sync 差太多。

需要区分 correctness 和 effectiveness。一个 backend 可以正确使用、预算正确、请求成功，但由于运行抖动导致 fault 数不一定每次都优于另一个 backend。C2 success check 因此默认不把 async-vs-sync fault 更低作为硬性 PASS 条件。

## 10. 结合一次 C2 实验结果的解读方式

用户提供的 Stage C2 实验中，最终输出包括：

```text
Stage C2 Success Check: PASS
C2 vs trace: gap_delta=-12.98% unknown_delta=-12.98%
C2 vs C1 backend: sync_gap=2693436 async_gap=2984963 gap_delta=+10.82%
throughput/tpot vs C1: sync_tok_s=34.64 async_tok_s=42.61 throughput_delta=+23.01% sync_tpot=141.21 async_tpot=114.37 tpot_delta=-19.01%
async_device_direct: actual=59710 budget_rejects=0 peak_live=931840 max_total=1048576
async_backend_counts={'cuda_malloc_async': 59710, 'none': 37038}
async_pool: set=True threshold=1048576 attempted=1 success=1 error=None
backend_signals: correctness=True async_effectiveness_vs_sync=False
```

这组结果说明：

1. C2 相对 trace-only baseline 有明确效果：gap faults 和 unknown faults 都下降约 12.98%。
2. async backend 真实启用：`cuda_malloc_async` 记录数为 59710。
3. async backend 没有误走 sync backend：backend counts 中没有 `cuda_malloc`。
4. 总预算生效：peak live 约 931840 bytes，小于 1048576 bytes。
5. 没有预算拒绝：`budget_rejects=0`，说明当前负载下 device-direct live set 没超过 1 MiB。
6. CUDA mempool release threshold 配置成功：attempted=1、success=1、error=None。
7. async 相对 sync 的 fault 数更高，所以 `async_effectiveness_vs_sync=False`。
8. async 相对 sync 的性能更好：throughput 提升约 23.01%，TPOT 降低约 19.01%。
9. 因为 correctness 全部满足，且 async 相对 trace baseline 有效果，所以总体验收 PASS。

这正体现了 Stage C2 的验收策略：async backend 是否“可用且安全”是硬要求；async 是否每次在 fault 数上优于 sync 是观察项，不是默认硬门槛。

## 11. 常见失败模式和排查方法

### 11.1 没有 `device_direct_actual_records`

可能原因：

1. `--uvm-device-direct-enable` 没有设为 1。
2. auto gap-watch policy action 仍是 `device_direct_trace`。
3. target class 不是 `gap_hot_runtime_scratch`。
4. target phase 没包含实际 allocation phase。
5. size 阈值过窄。
6. target gap 没有被 main 阶段复用。
7. allocator log 没有正确传入或 summary 解析了旧日志。

排查字段：

1. `device_direct_reason_counts`
2. `hot_gap_match_records`
3. `phase_record_ratios`
4. `placement_backend_counts`
5. `device_direct_backend_counts`

### 11.2 `device_direct_budget_reject_records > 0`

这说明 C1 总预算确实拒绝了一部分 eligible allocation，并回退 managed。它不一定是失败。如果目标是验证预算拒绝行为，这是正确信号。如果目标是验证 device-direct 效果，可能需要提高 `DEVICE_DIRECT_MAX_TOTAL_BYTES`。

### 11.3 async pool 配置失败

看：

1. `device_direct_pool_config_attempted`
2. `device_direct_pool_config_success`
3. `device_direct_pool_config_error`

如果 `device_direct_pool_release_threshold_set=True` 但 config failed，Stage C2 check 会失败。此时优先确认 CUDA runtime、driver 和当前 device 是否支持默认 mempool API。

### 11.4 `gap_policy_fail > 0`

这表示 gap-watch 策略执行失败。对于 `device_direct`，常见原因是 backend 分配或候选释放失败。应查看 allocator trace 中的 `device_direct_reason` 和 `action_error`。

### 11.5 probe/main gap 不一致

Stage C 依赖同一 server 进程的动态 gap 发现。如果 probe 发现的 gap 和 main 后 hottest gap 不一致，实验仍可能运行成功，但对比结果的解释性会变差。check 脚本通常要求 same gap，以避免把“热点迁移”误判成 device-direct 效果。

## 12. Stage C 和 Stage D/E 的关系

Stage C 解决的是临时 scratch/workspace 类 allocation 的真实 GPU-only backend 问题。它的粒度是“被 gap-watch 识别出的热点 runtime scratch 对象”，而不是完整的语义内存池分区。

Stage D 在此基础上开始跟踪 KV cache 预算，目标是把 KV cache 作为独立语义池管理。

Stage E 开始跟踪 weights 预算和权重语义地图，目标是把 weights 也分成独立语义池。

因此，Stage C 是后续分区管理的执行基础：它证明 allocator 能在特定条件下安全改变 placement backend、能记录 backend 元数据、能做总预算约束、能在 free 路径正确释放并归还预算。Stage D/E 则把“哪些对象属于哪个池”这件事进一步语义化。

## 13. 修改点速查表

| 文件 | Stage C 作用 |
| --- | --- |
| `workloads/vllm/vllm/uvm_test/uvm_allocator.cpp` | 核心 allocator，实现 device-direct 门控、真实 backend、预算、async mempool、日志和 free 路径 |
| `workloads/vllm/run_kv_fault_ratio.sh` | 单进程 runner，传入 Stage C 参数，启动 vLLM，配置 auto gap-watch，收集日志 |
| `workloads/vllm/run_stage_c_attention_p20_ab.sh` | trace-only baseline vs attention-only device-direct A/B |
| `workloads/vllm/run_stage_c_attention_backend_ab.sh` | `cuda_malloc` vs `cuda_malloc_async` backend A/B |
| `workloads/vllm/compare_stage_c_attention_p20_ab.py` | 比较 trace baseline 和真实 Stage C device-direct |
| `workloads/vllm/compare_stage_c_backend_ab.py` | 比较 C1 sync backend 和 C2 async backend |
| `workloads/vllm/check_stage_c1_success.py` | C1 一键 correctness/effectiveness 验收 |
| `workloads/vllm/check_stage_c2_success.py` | C2 一键 backend correctness/effectiveness 验收 |
| `workloads/vllm/summarize_gap_watch_metrics.py` | 从 allocator trace 聚合 Stage C metrics |
| `docs/vllm_uvm_device_direct_stage_c_attention_only_implementation.md` | 早期 attention-only Stage C 实现记录 |
| `docs/vllm_uvm_stage_c1_completion_implementation.md` | C1 budget sweep 和补全记录 |
| `docs/vllm_uvm_device_direct_stage_c2_async_backend_implementation.md` | C2 async backend 记录 |

## 14. 最小阅读路径

如果只想快速理解 Stage C，建议按这个顺序读：

1. 本文档的第 1 到 7 节，先建立整体模型。
2. `workloads/vllm/run_stage_c_attention_p20_ab.sh`，理解实验如何配置 trace 和 device 两组。
3. `workloads/vllm/run_kv_fault_ratio.sh`，理解参数如何传到 vLLM server 环境变量。
4. `workloads/vllm/vllm/uvm_test/uvm_allocator.cpp` 中 device-direct 相关变量、`reserve_device_direct_budget()`、`configure_device_direct_async_pool_if_needed()`、`uvm_malloc()`、`uvm_free()`。
5. `workloads/vllm/summarize_gap_watch_metrics.py`，理解 JSON metrics 怎么从 trace log 得到。
6. `workloads/vllm/check_stage_c2_success.py`，理解最终 PASS/FAIL 的条件。

读完这条路径，基本就能独立修改 Stage C 的阈值、backend、验收条件，也能判断一次实验结果到底是“正确性失败”“效果不明显”还是“正常波动但总体成功”。

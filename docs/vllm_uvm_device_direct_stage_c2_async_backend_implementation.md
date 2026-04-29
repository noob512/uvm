# vLLM UVM Device-direct Stage C2 Async Backend Implementation

本文档记录 Stage C2 的实现：在 Stage C1 attention-only device-direct 总预算已经验证成功的基础上，新增可切换的 `cudaMallocAsync/cudaFreeAsync` 后端，并补齐可配置 CUDA memory pool release threshold 框架。

## 1. 背景

Stage C0/C1 已经证明：

1. `enabled:attention` 下的 `gap_hot_runtime_scratch` 可以安全走真实 `device_direct`。
2. C1 总预算可以限制 device-direct live bytes。
3. 小预算下超预算 fallback managed 正常。
4. gap2 faults 和 unknown faults 有明显下降。

但当前 C1 默认后端是：

```text
cudaMalloc
cudaFree
```

runtime scratch 的特点是：

1. 分配次数高。
2. 生命周期短。
3. 频繁出现在推理热路径。
4. 同时 live bytes 很小，但 allocation/free 调用非常密集。

因此 C2 的目标不是扩大策略范围，而是在相同 gating 下替换真实 device-direct 后端，并为后续 memory pool 调优提供稳定入口：

```text
cudaMalloc/cudaFree -> cudaMallocAsync/cudaFreeAsync
optional cudaMemPoolAttrReleaseThreshold
```

## 2. 实现范围

本阶段只实现 backend 切换，不改变 C1 策略边界。

保持不变：

```text
target_class = gap_hot_runtime_scratch
target_phase = enabled:attention
min_bytes = 4096
max_bytes = 1048576
max_total_bytes = C1 budget
```

新增能力：

```text
--uvm-device-direct-backend cuda_malloc
--uvm-device-direct-backend cuda_malloc_async
--uvm-device-direct-pool-release-threshold <bytes>
```

对应环境变量：

```text
VLLM_UVM_DEVICE_DIRECT_BACKEND=cuda_malloc|cuda_malloc_async
VLLM_UVM_DEVICE_DIRECT_POOL_RELEASE_THRESHOLD=<bytes>
```

默认值：

```text
cuda_malloc
```

这意味着现有 C1 实验默认行为不变。只有显式设置 `cuda_malloc_async` 时才进入 C2 async 路径。只有显式设置 pool release threshold 时才调用 CUDA mempool 配置 API；未设置时保持 CUDA runtime 默认行为。

## 3. 修改文件

核心实现：

```text
/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/uvm_allocator.cpp
```

实验 runner：

```text
/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/run_kv_fault_ratio.sh
/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/run_stage_c_attention_p20_ab.sh
/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/run_stage_c_attention_backend_ab.sh
```

结果解析：

```text
/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/summarize_gap_watch_metrics.py
/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/compare_stage_c_attention_p20_ab.py
/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/compare_stage_c_backend_ab.py
```

规划文档：

```text
/home/ubuntu/nvidia-uvm-gpu/docs/vllm_uvm_memory_pool_evolution_plan.md
```

## 4. Allocator 行为

### 4.1 Backend 配置

allocator 新增：

```cpp
static std::string device_direct_backend = "cuda_malloc";
static bool device_direct_pool_release_threshold_set = false;
static size_t device_direct_pool_release_threshold = 0;
```

初始化时读取：

```text
VLLM_UVM_DEVICE_DIRECT_BACKEND
VLLM_UVM_DEVICE_DIRECT_POOL_RELEASE_THRESHOLD
```

合法值：

```text
cuda_malloc
cuda_malloc_async
```

非法值会在 shell runner 侧被拒绝；allocator 侧也会 normalize 到默认 `cuda_malloc`，避免直接 env 注入导致未知行为。

pool release threshold 使用显式开关语义：

1. 环境变量不存在或为空：不配置 CUDA mempool，保持 CUDA 默认。
2. 环境变量存在且为 `0`：调用 `cudaMemPoolSetAttribute(..., 0)`，表示尽量激进释放。
3. 环境变量存在且为正整数：设置默认 CUDA mempool 的 release threshold。

### 4.2 Allocation 元数据

`AllocationInfo` 新增：

```cpp
std::string device_direct_backend_name;
```

原因是 free 阶段必须知道该 pointer 当初是用哪个 backend 分配的。不能只读当前全局配置，因为后续可能做动态配置或多实验复用进程。

### 4.3 分配路径

Stage C2 沿用 C1 的严格 gating：

```text
hot gap match
target class match
phase allowlist match
size within [min, max]
device_direct action requested
device_direct_enable = 1
budget reservation success
```

预算预留成功后：

```cpp
if backend == cuda_malloc_async:
    if pool threshold configured:
        cudaDeviceGetDefaultMemPool(...)
        cudaMemPoolSetAttribute(cudaMemPoolAttrReleaseThreshold, threshold)
    cudaMallocAsync(&device_ptr, size, stream)
else:
    cudaMalloc(&device_ptr, size)
```

pool threshold 配置是 lazy 的：只在 `cuda_malloc_async` backend 第一次真实 device-direct 分配前对当前 device 配置一次。这样 C1 `cuda_malloc` 路径完全不受影响。

如果 device allocation 成功，再释放 managed candidate：

```cpp
cudaFree(managed_candidate)
```

然后返回 device pointer，并记录：

```text
placement_backend=device_direct
device_direct_backend=cuda_malloc_async
device_direct_reason=device_direct_enabled
```

### 4.4 失败与 fallback

如果预算不足：

```text
device_direct_reason=device_direct_budget_exceeded
placement_backend=managed
```

如果 `cudaMallocAsync` 失败：

```text
device_direct_reason=device_malloc_async_failed_fallback_managed
placement_backend=managed
```

如果用户显式设置了 pool release threshold，但 CUDA mempool 配置失败：

```text
device_direct_reason=device_direct_pool_config_failed
placement_backend=managed
```

这种情况下不会继续伪装成成功的 async pool 实验，而是保守 fallback managed，并在 summary 中记录错误。

如果 `cudaMalloc` 失败：

```text
device_direct_reason=device_malloc_failed_fallback_managed
placement_backend=managed
```

失败时会释放已预留的 C1 budget。

### 4.5 Free 路径

C2 之前的 free 路径是先 `cudaFree(ptr)`，再查 metadata。C2 修改为：

1. 先查 `active_allocations`。
2. 判断是否是 `placement_backend=device_direct`。
3. 判断 `device_direct_backend_name`。
4. 再选择 free API：

```cpp
if backend == cuda_malloc_async:
    cudaFreeAsync(ptr, stream)
else:
    cudaFree(ptr)
```

free 成功后：

```text
device_direct_live_bytes -= size
device_direct_free_success_allocs += 1
```

## 5. 日志与观测

`TRACE_POLICY` 新增字段：

```text
device_direct_backend=<none|cuda_malloc|cuda_malloc_async>
device_direct_pool_release_threshold_set=<0|1>
device_direct_pool_release_threshold=<bytes>
```

`TRACE_GAP_WATCH_ALLOC` 新增字段：

```text
device_direct_backend=<none|cuda_malloc|cuda_malloc_async>
device_direct_pool_release_threshold_set=<0|1>
device_direct_pool_release_threshold=<bytes>
```

`TRACE_GAP_WATCH_FREE` 新增字段：

```text
device_direct_backend=<none|cuda_malloc|cuda_malloc_async>
```

session summary 新增：

```text
Device-direct backend: cuda_malloc_async
Device-direct pool release threshold set: 1
Device-direct pool release threshold: 1048576
Device-direct pool config attempted: 1
Device-direct pool config success: 1
Device-direct pool config error: none
```

`summarize_gap_watch_metrics.py` 新增：

```text
device_direct_backend_counts
device_direct_pool_release_threshold_set
device_direct_pool_release_threshold
device_direct_pool_config_attempted
device_direct_pool_config_success
device_direct_pool_config_error
```

预期 C2 成功时可以看到：

```text
placement_backend_counts={'device_direct': ..., 'managed': ...}
device_direct_backend_counts={'cuda_malloc_async': ..., 'none': ...}
device_direct_actual_records > 0
device_direct_peak_live_bytes_observed <= device_direct_max_total_bytes
device_direct_pool_config_success=1  # 当显式设置 threshold 时
```

## 6. 推荐验证命令

### 6.1 C1 baseline

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm

PROMPTS=10 \
DEVICE_DIRECT_MAX_TOTAL_BYTES=1048576 \
DEVICE_DIRECT_BACKEND=cuda_malloc \
./run_stage_c_attention_p20_ab.sh
```

### 6.2 C2 async backend

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm

PROMPTS=10 \
DEVICE_DIRECT_MAX_TOTAL_BYTES=1048576 \
DEVICE_DIRECT_BACKEND=cuda_malloc_async \
./run_stage_c_attention_p20_ab.sh
```

### 6.3 C2 async backend with pool release threshold

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm

PROMPTS=10 \
DEVICE_DIRECT_MAX_TOTAL_BYTES=1048576 \
DEVICE_DIRECT_BACKEND=cuda_malloc_async \
DEVICE_DIRECT_POOL_RELEASE_THRESHOLD=1048576 \
./run_stage_c_attention_p20_ab.sh
```

### 6.4 C1 vs C2 one-shot comparison

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm

PROMPTS=10 \
DEVICE_DIRECT_MAX_TOTAL_BYTES=1048576 \
ASYNC_DEVICE_DIRECT_POOL_RELEASE_THRESHOLD=1048576 \
./run_stage_c_attention_backend_ab.sh
```

### 6.5 C2 自动成功检查

新增脚本：

```text
/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/check_stage_c2_success.py
```

默认行为是运行一个较小的 C1 `cuda_malloc` vs C2 `cuda_malloc_async` backend A/B，并显式设置 async CUDA mempool release threshold：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm

./check_stage_c2_success.py
```

脚本会检查：

```text
correctness_signal=True
async_success_signal=True
async_effectiveness_signal=True  # C2 相对 trace-only baseline
sync backend 使用 cuda_malloc
async backend 使用 cuda_malloc_async
Failed requests = 0
gap_policy_fail = 0
device_direct_actual_records > 0
peak_live <= budget
pool threshold/config 字段匹配预期
```

离线复核已有 backend A/B 目录：

```bash
./check_stage_c2_success.py \
  --skip-run \
  --run-dir /tmp/vllm_stage_c_attention_backend_ab_<RUN_ID> \
  --prompts 10 \
  --pool-release-threshold 1048576
```

如果只想检查 C2 框架正确性，不把 fault reduction 作为硬门槛：

```bash
./check_stage_c2_success.py --no-require-effectiveness-vs-trace
```

如果要把 async 对 C1 sync 不退化也作为硬门槛，再额外加：

```bash
./check_stage_c2_success.py --require-not-worse-than-sync
```

### 6.6 p20 复核

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm

PROMPTS=20 \
DEVICE_DIRECT_MAX_TOTAL_BYTES=1048576 \
DEVICE_DIRECT_BACKEND=cuda_malloc_async \
./run_stage_c_attention_p20_ab.sh
```

## 7. 成功标准

C2 首轮成功必须满足：

```text
success_signal=True
effectiveness_signal=True
Failed requests = 0
gap_policy_fail = 0
device_peak_live_within_budget=True
device_direct_actual_records > 0
device_direct_backend_counts 中 cuda_malloc_async > 0
```

如果本轮显式设置了 pool release threshold，还必须满足：

```text
device_direct_pool_release_threshold_set=True
device_direct_pool_release_threshold=<expected bytes>
device_direct_pool_config_attempted=1
device_direct_pool_config_success=1
device_direct_pool_config_error=none
```

收益判断：

```text
gap_fault_delta_pct <= C1 baseline
unknown_fault_delta_pct <= C1 baseline
mean_tpot_delta_pct 不恶化
output_throughput_delta_pct 不恶化
```

如果 async backend 的 fault 收益类似 C1，但 TPOT/throughput 更好，说明分配释放开销确实下降。

## 8. 回退方式

如果出现 OOM、illegal address、invalid pointer、failed requests 或 async free 异常，直接回退：

```bash
DEVICE_DIRECT_BACKEND=cuda_malloc ./run_stage_c_attention_p20_ab.sh
```

或者完全关闭真实 device-direct：

```bash
./run_kv_fault_ratio.sh \
  --uvm-device-direct-enable 0 \
  --auto-gap-watch-policy-action-override device_direct_trace
```

## 9. 当前边界

C2 当前已经做到：

1. 自定义 CUDA memory pool release threshold。
2. `cuda_malloc_async` backend 可选。
3. C1 budget gating 复用。
4. pool 配置状态进入 trace / summary / comparison JSON。

C2 当前仍没有做：

1. 独立 per-phase pool。
2. 扩大到 `enabled:moe`。
3. 放宽 `--uvm-device-direct-max-bytes`。
4. 把 `cuda_malloc_async` 设为默认 backend。

这些都应在 C2 pool threshold 框架稳定后再做。

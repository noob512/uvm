# vLLM UVM Device-direct Stage C2 Async Backend Implementation

本文档记录 Stage C2 的实现：在 Stage C1 attention-only device-direct 总预算已经验证成功的基础上，新增可切换的 `cudaMallocAsync/cudaFreeAsync` 后端。

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

因此 C2 的目标不是扩大策略范围，而是在相同 gating 下替换真实 device-direct 后端：

```text
cudaMalloc/cudaFree -> cudaMallocAsync/cudaFreeAsync
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
```

对应环境变量：

```text
VLLM_UVM_DEVICE_DIRECT_BACKEND=cuda_malloc|cuda_malloc_async
```

默认值：

```text
cuda_malloc
```

这意味着现有 C1 实验默认行为不变。只有显式设置 `cuda_malloc_async` 时才进入 C2 路径。

## 3. 修改文件

核心实现：

```text
/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/uvm_allocator.cpp
```

实验 runner：

```text
/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/run_kv_fault_ratio.sh
/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/run_stage_c_attention_p20_ab.sh
```

结果解析：

```text
/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/summarize_gap_watch_metrics.py
/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/compare_stage_c_attention_p20_ab.py
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
```

初始化时读取：

```text
VLLM_UVM_DEVICE_DIRECT_BACKEND
```

合法值：

```text
cuda_malloc
cuda_malloc_async
```

非法值会在 shell runner 侧被拒绝；allocator 侧也会 normalize 到默认 `cuda_malloc`，避免直接 env 注入导致未知行为。

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
    cudaMallocAsync(&device_ptr, size, stream)
else:
    cudaMalloc(&device_ptr, size)
```

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
```

`TRACE_GAP_WATCH_ALLOC` 新增字段：

```text
device_direct_backend=<none|cuda_malloc|cuda_malloc_async>
```

`TRACE_GAP_WATCH_FREE` 新增字段：

```text
device_direct_backend=<none|cuda_malloc|cuda_malloc_async>
```

session summary 新增：

```text
Device-direct backend: cuda_malloc_async
```

`summarize_gap_watch_metrics.py` 新增：

```text
device_direct_backend_counts
```

预期 C2 成功时可以看到：

```text
placement_backend_counts={'device_direct': ..., 'managed': ...}
device_direct_backend_counts={'cuda_malloc_async': ..., 'none': ...}
device_direct_actual_records > 0
device_direct_peak_live_bytes_observed <= device_direct_max_total_bytes
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

### 6.3 p20 复核

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

C2 当前没有做：

1. 自定义 CUDA memory pool release threshold。
2. 独立 per-phase pool。
3. 扩大到 `enabled:moe`。
4. 放宽 `--uvm-device-direct-max-bytes`。

这些都应在 C2 async backend 稳定后再做。

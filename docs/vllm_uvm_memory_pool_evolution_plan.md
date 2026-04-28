# vLLM UVM Memory Pool Evolution Plan

本文档给出从当前 gap2 / `gap_hot_runtime_scratch` 实验继续演进到显存逻辑分区、device-direct 预算、异步内存池、KV 独立预算以及 MoE 权重冷热分层的详细实施方案。

核心原则：

1. 不一次性把所有 managed 内存切成 `cudaMalloc`。
2. 先解决最明确、最短命、CPU 访问风险最低的 runtime scratch。
3. 每一步都保留 kill switch。
4. 每一步都必须能通过 trace 证明生效范围、收益和风险。
5. KV eviction 和 weights offload 不在 allocator 层硬做，必须逐步接入 vLLM 语义。

## 1. 当前状态

当前已经具备以下基础能力：

1. gap2 auto discovery，可以在同一个 vLLM server 进程中先 probe，再发现 gap2，再跑 main。
2. allocator 可以记录 `TRACE_POLICY`、`TRACE_GAP_WATCH_ALLOC`、`TRACE_GAP_WATCH_FREE`。
3. allocator 已支持 `device_direct_trace` 和 `device_direct` action。
4. Stage B strict gating 已经支持 `gap_hot_runtime_scratch`。
5. Stage C attention-only 已经实现真实 `device_direct` backend，并由 `--uvm-device-direct-enable 0/1` 控制。
6. 细粒度 phase marker 已经能区分：

```text
enabled:attention
enabled:moe
enabled:model_forward
enabled:compute_logits
enabled:sampler
enabled:prefill_step
enabled:decode_step
```

最近实验显示 gap2 具备如下特征：

1. gap2 是 unknown fault 主热点。
2. gap2 fault 主要是 write。
3. main 阶段主导 phase 是 `enabled:attention`。
4. 大量候选分配生命周期在 1ms 到数 ms。
5. 严格 Stage B 下，小对象候选数充足。

因此演进路线应优先围绕 runtime scratch 做安全 device-direct，而不是立即进入 weights 全量 offload。

## 2. 总体架构目标

长期目标是建立逻辑显存分区，而不是依赖 CUDA/UVM driver 自己在所有对象之间抢占迁移。

建议逻辑 pool：

```text
GPU VRAM
├── Pool A: hot weights / resident weights
├── Pool B: KV cache
├── Pool C: runtime scratch / device_direct
├── Pool D: prefetch staging / cold weights window
└── Reserve: framework / CUDA graph / emergency
```

各 pool 的设计目标：

| Pool | 主要对象 | 初始 backend | 调度策略 |
| --- | --- | --- | --- |
| Pool A | shared weights、热 experts | managed + prefetch 或 device resident | 尽量常驻 GPU |
| Pool B | KV blocks | managed 或 vLLM KV allocator | 由 block manager 决定 swap/evict |
| Pool C | attention/MoE scratch | device_direct | 不驱逐，不足则 fallback |
| Pool D | cold expert weights / staging | CPU pinned / managed | 预测预取 |
| Reserve | CUDA runtime / graph / fragmentation | 不主动占用 | 防 OOM |

注意：这里的“分区”是逻辑预算，不是硬件物理分区。CUDA 不提供简单的 VRAM 硬分区接口，因此必须由 allocator、vLLM scheduler、KV block manager 共同执行预算。

## 3. 阶段 C0：Attention-only `device_direct`

### 3.1 目标

验证最小真实 device-direct 是否可行：

1. 只对 `enabled:attention`。
2. 只对 `gap_hot_runtime_scratch`。
3. 只对小对象，例如 `4 KiB <= size <= 1 MiB`。
4. 真实 backend 从 managed 替换成 `cudaMalloc`。
5. 不影响 KV cache 和 weights。
6. 出错可一键回退到 trace-only。

### 3.2 当前实现边界

当前 Stage C0 已实现：

1. allocator 先创建 managed candidate。
2. 用 managed candidate 地址判断是否命中 gap2。
3. 如果满足严格条件，调用 `cudaMalloc` 创建 device pointer。
4. 释放 managed candidate。
5. 返回 device pointer。
6. 日志记录 `placement_backend=device_direct`。

当前实现还不是最终形态，因为每次 `cudaMalloc/cudaFree` 可能带来同步开销和碎片风险。C0 的目标只是验证正确性和收益方向。

### 3.3 推荐命令

小流量：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm

./run_kv_fault_ratio.sh \
  --mode trace \
  --with-address-log \
  --trace-log /tmp/uvm_kv_fault_stats_gap2_stage_c_attention_p5.log \
  --address-trace-log /tmp/uvm_kv_fault_addrs_gap2_stage_c_attention_p5.log \
  --allocator-log /tmp/vllm_uvm_allocator_trace_gap2_stage_c_attention_p5.log \
  --uvm-trace-min-bytes 1048576 \
  --uvm-unknown-detail-enable 1 \
  --uvm-unknown-detail-min-bytes 4096 \
  --uvm-gap-watch-name same_run_gap2_stage_c_attention_p5 \
  --uvm-gap-watch-all-classes 1 \
  --uvm-gap-watch-min-bytes 4096 \
  --uvm-device-direct-enable 1 \
  --uvm-device-direct-min-bytes 4096 \
  --uvm-device-direct-max-bytes 1048576 \
  --uvm-device-direct-target-phases enabled:attention \
  --auto-gap-watch-enable 1 \
  --auto-gap-watch-probe-prompts 1 \
  --auto-gap-watch-target-gap 2 \
  --auto-gap-watch-policy-action-override device_direct \
  --auto-gap-watch-target-class-override gap_hot_runtime_scratch \
  --prompts 5 \
  --gap-watch-metrics-summary-json /tmp/vllm_gap_watch_metrics_gap2_stage_c_attention_p5.json \
  --auto-gap-watch-summary-json /tmp/vllm_auto_gap_watch_summary_gap2_stage_c_attention_p5.json \
  --auto-gap-watch-post-main-summary-json /tmp/vllm_auto_gap_watch_post_main_summary_gap2_stage_c_attention_p5.json
```

主实验：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm

./run_kv_fault_ratio.sh \
  --mode trace \
  --with-address-log \
  --trace-log /tmp/uvm_kv_fault_stats_gap2_stage_c_attention.log \
  --address-trace-log /tmp/uvm_kv_fault_addrs_gap2_stage_c_attention.log \
  --allocator-log /tmp/vllm_uvm_allocator_trace_gap2_stage_c_attention.log \
  --uvm-trace-min-bytes 1048576 \
  --uvm-unknown-detail-enable 1 \
  --uvm-unknown-detail-min-bytes 4096 \
  --uvm-gap-watch-name same_run_gap2_stage_c_attention \
  --uvm-gap-watch-all-classes 1 \
  --uvm-gap-watch-min-bytes 4096 \
  --uvm-device-direct-enable 1 \
  --uvm-device-direct-min-bytes 4096 \
  --uvm-device-direct-max-bytes 1048576 \
  --uvm-device-direct-target-phases enabled:attention \
  --auto-gap-watch-enable 1 \
  --auto-gap-watch-probe-prompts 1 \
  --auto-gap-watch-target-gap 2 \
  --auto-gap-watch-policy-action-override device_direct \
  --auto-gap-watch-target-class-override gap_hot_runtime_scratch \
  --prompts 20 \
  --gap-watch-metrics-summary-json /tmp/vllm_gap_watch_metrics_gap2_stage_c_attention.json \
  --auto-gap-watch-summary-json /tmp/vllm_auto_gap_watch_summary_gap2_stage_c_attention.json \
  --auto-gap-watch-post-main-summary-json /tmp/vllm_auto_gap_watch_post_main_summary_gap2_stage_c_attention.json
```

### 3.4 成功标准

必须同时满足：

1. benchmark 完整结束。
2. `Failed requests = 0`。
3. `placement_backend_counts` 中出现 `device_direct`。
4. `device_direct_actual_records > 0`。
5. `gap_policy_fail = 0`。
6. 没有 CUDA illegal address、invalid device pointer、OOM 等错误。

收益指标：

1. gap2 faults 低于 Stage B trace-only。
2. unknown faults 低于 Stage B trace-only。
3. TTFT / TPOT / throughput 不显著恶化。

### 3.5 回退条件

出现任一情况即回退：

1. vLLM server 启动失败。
2. CUDA OOM。
3. illegal address。
4. invalid device pointer。
5. failed requests > 0。
6. TTFT/TPOT 明显恶化且 fault 没下降。

回退方式：

```bash
--uvm-device-direct-enable 0
--auto-gap-watch-policy-action-override device_direct_trace
```

## 4. 阶段 C1：Device-direct 总预算

当前项目已实现 C1。实现位置包括：

1. `/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/uvm_allocator.cpp`
2. `/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/run_kv_fault_ratio.sh`
3. `/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/summarize_gap_watch_metrics.py`
4. `/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/run_stage_c_attention_p20_ab.sh`
5. `/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/compare_stage_c_attention_p20_ab.py`
6. `/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/run_stage_c_attention_c1_budget_sweep.sh`
7. `/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/compare_stage_c_budget_sweep.py`

C1 补全实现文档：

```text
/home/ubuntu/nvidia-uvm-gpu/docs/vllm_uvm_stage_c1_completion_implementation.md
```

### 4.1 为什么必须做预算

直接使用 `cudaMalloc` 会占用真实 GPU VRAM。即使每个对象小于 1 MiB，在高并发和重叠生命周期下也可能累计占用大量显存。

如果没有预算，风险包括：

1. scratch 抢占 KV cache 空间。
2. scratch 抢占 weights resident 空间。
3. CUDA runtime / graph reserve 被压缩。
4. 高并发下出现 OOM。
5. 频繁 fallback 造成性能波动。

C1 的目标是把 device-direct 从“按单个对象限制”升级为“按 pool 总量限制”。

### 4.2 新增参数

建议新增：

```bash
--uvm-device-direct-max-total-bytes <n>
```

对应环境变量：

```text
VLLM_UVM_DEVICE_DIRECT_MAX_TOTAL_BYTES
```

建议初始值：

```text
268435456    # 256 MiB
536870912    # 512 MiB
1073741824   # 1 GiB
```

默认值建议先保守：

```text
268435456
```

如果为 `0`，可表示不限制，但实验阶段不建议开放。

当前实现默认值为：

```text
268435456    # 256 MiB
```

也可以通过 A/B 脚本的环境变量覆盖：

```bash
DEVICE_DIRECT_MAX_TOTAL_BYTES=536870912 ./run_stage_c_attention_p20_ab.sh
```

### 4.3 allocator 设计

新增全局计数器：

```cpp
static size_t device_direct_max_total_bytes;
static std::atomic<size_t> device_direct_live_bytes;
static std::atomic<size_t> device_direct_peak_live_bytes;
static std::atomic<size_t> device_direct_budget_rejects;
```

分配前判定：

```text
if live_bytes + size > max_total_bytes:
    device_direct_reason=device_direct_budget_exceeded
    placement_backend=managed
```

真实 device allocation 成功后：

```text
device_direct_live_bytes += size
device_direct_peak_live_bytes = max(...)
```

free 成功后：

```text
device_direct_live_bytes -= size
```

注意：预算预留必须在 `cudaMalloc` 前完成，或者使用 CAS reservation，避免多线程并发同时越过预算。

推荐 CAS 流程：

```text
1. read live_bytes
2. if live + size > budget reject
3. compare_exchange live -> live + size
4. cudaMalloc
5. cudaMalloc 失败则 live_bytes -= size
```

当前实现采用上述 CAS reservation。关键行为：

1. 只有通过 C0 严格 gating 的候选才会进入预算预留。
2. `max_total_bytes > 0` 且 `live_bytes + size > max_total_bytes` 时拒绝本次 `device_direct`。
3. 拒绝时不会调用 `cudaMalloc`，会保留 managed candidate。
4. 日志中记录 `device_direct_reason=device_direct_budget_exceeded`。
5. `cudaMalloc` 失败或 managed candidate 释放失败时会释放已预留预算。
6. 真正 `device_direct` 对象在 `uvm_free` 成功后扣减 `device_direct_live_bytes`。

### 4.4 新增日志字段

`TRACE_POLICY` 和 `TRACE_GAP_WATCH_ALLOC` 建议增加：

```text
device_direct_live_bytes=<n>
device_direct_max_total_bytes=<n>
device_direct_budget_remaining=<n>
```

session summary 增加：

```text
Device-direct max total bytes
Device-direct live bytes
Device-direct peak live bytes
Device-direct budget rejects
```

当前实现已在 `TRACE_POLICY` 和 `TRACE_GAP_WATCH_ALLOC` 中输出这些字段，并在 allocator session summary 中输出上述汇总项。

`summarize_gap_watch_metrics.py` 会额外汇总：

```text
device_direct_budget_reject_records
device_direct_max_total_bytes
device_direct_peak_live_bytes_observed
device_direct_min_budget_remaining_observed
```

其中 `device_direct_peak_live_bytes_observed` 应该满足：

```text
device_direct_peak_live_bytes_observed <= device_direct_max_total_bytes
```

### 4.5 验证矩阵

建议按预算递增：

| 实验 | max total bytes | phase | prompts |
| --- | ---: | --- | ---: |
| C1-A | 256 MiB | enabled:attention | 5 |
| C1-B | 256 MiB | enabled:attention | 20 |
| C1-C | 512 MiB | enabled:attention | 20 |
| C1-D | 1 GiB | enabled:attention | 20 |

观察：

1. `device_direct_actual_records`
2. `device_direct_budget_rejects`
3. `device_direct_peak_live_bytes`
4. gap2 faults
5. OOM / failed requests
6. TTFT / TPOT

推荐 C1 p20 A/B 命令：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm

DEVICE_DIRECT_MAX_TOTAL_BYTES=268435456 ./run_stage_c_attention_p20_ab.sh
```

如果 256 MiB 下 `device_direct_budget_reject_records` 很高、收益明显变弱，但没有 OOM，可以继续：

```bash
DEVICE_DIRECT_MAX_TOTAL_BYTES=536870912 ./run_stage_c_attention_p20_ab.sh
```

如果 512 MiB 仍稳定，再测试：

```bash
DEVICE_DIRECT_MAX_TOTAL_BYTES=1073741824 ./run_stage_c_attention_p20_ab.sh
```

### 4.6 成功标准

1. 无 OOM。
2. `device_direct_peak_live_bytes <= max_total_bytes`。
3. gap2 faults 下降。
4. `device_direct_budget_rejects` 可以存在，但不应导致性能剧烈抖动。

额外建议同时看 A/B 报告中的：

1. `success_signal=True`
2. `effectiveness_signal=True`
3. `device_peak_live_within_budget=True`
4. `device_direct_actual_records > 0`
5. `gap_policy_fail=0`
6. `Failed requests = 0`

### 4.7 当前补全状态

当前 C1 已补齐两个实验闭环能力：

1. `run_kv_fault_ratio.sh` 的 delta stats parser 已支持当前中文 UVM stats 格式，可以从 `本批次总缺页实例数=...`、`KV类的总缺页数=...`、`总缺页数=...`、`kv总错误数=...` 中解析 batch、total、after-dedup，并推导 duplicate counters。
2. 新增 C1 budget sweep 一键脚本，可以固定 `cuda_malloc` backend，对多个 `DEVICE_DIRECT_MAX_TOTAL_BYTES` 连续运行 Stage C attention A/B，并生成预算横向汇总 JSON。

推荐 C1 budget sweep：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm

PROMPTS=10 \
BUDGETS_CSV=524288,1048576,2097152,4194304 \
./run_stage_c_attention_c1_budget_sweep.sh
```

输出目录：

```text
/tmp/vllm_stage_c1_budget_sweep_<timestamp>/
```

汇总文件：

```text
vllm_stage_c1_budget_sweep_p<PROMPTS>.json
```

当前 C1/C2 backend A/B 结论：

1. `cuda_malloc` 是当前推荐 C1 backend。
2. `cuda_malloc_async` 已通过正确性验证，但在 p10 / 1 MiB budget 对照中，gap faults 和 TPOT 均弱于 `cuda_malloc`，暂不设为默认。
3. 后续应先完成 C1 budget sweep 和 stats delta 稳定化，再考虑 C2 memory pool threshold 或 MoE 扩展。

## 5. 阶段 C2：`cudaMallocAsync` / CUDA Memory Pool

当前项目已实现 C2 的第一步：可切换 `cudaMallocAsync/cudaFreeAsync` backend。默认仍保持 C1 的 `cuda_malloc` 路径，只有显式设置 backend 时才启用 async 路径。

实现文档：

```text
/home/ubuntu/nvidia-uvm-gpu/docs/vllm_uvm_device_direct_stage_c2_async_backend_implementation.md
```

### 5.1 为什么需要 C2

C0/C1 使用 `cudaMalloc/cudaFree`，但 runtime scratch 的特点是：

1. 高频分配。
2. 高频释放。
3. 生命周期短。
4. 多发生在推理热路径。

`cudaMalloc/cudaFree` 可能引入同步、锁竞争和碎片。更合适的后端是：

```text
cudaMallocAsync
cudaFreeAsync
cudaMemPool
```

### 5.2 新增参数

```bash
--uvm-device-direct-backend cuda_malloc
--uvm-device-direct-backend cuda_malloc_async
```

对应环境变量：

```text
VLLM_UVM_DEVICE_DIRECT_BACKEND=cuda_malloc|cuda_malloc_async
```

默认：

```text
cuda_malloc
```

C2 实验时切换为：

```text
cuda_malloc_async
```

当前实现还支持 A/B 脚本环境变量：

```bash
DEVICE_DIRECT_BACKEND=cuda_malloc_async ./run_stage_c_attention_p20_ab.sh
```

### 5.3 allocator 设计

新增 enum：

```cpp
enum class DeviceDirectBackend {
    CudaMalloc,
    CudaMallocAsync,
};
```

分配：

```cpp
cudaMallocAsync(&ptr, size, stream);
```

释放：

```cpp
cudaFreeAsync(ptr, stream);
```

注意事项：

1. 必须记录 allocation 使用的 backend。
2. free 时根据 backend 选择 `cudaFree` 或 `cudaFreeAsync`。
3. 如果 stream 无效或为空，可 fallback 到 `cudaMalloc/cudaFree`。
4. 如果 vLLM/PyTorch allocator 调用 free 时传入的 stream 与 allocation stream 不一致，需要评估 stream-ordered 语义是否安全。

当前实现细节：

1. `VLLM_UVM_DEVICE_DIRECT_BACKEND=cuda_malloc` 时保持 C1 行为，使用 `cudaMalloc/cudaFree`。
2. `VLLM_UVM_DEVICE_DIRECT_BACKEND=cuda_malloc_async` 时，真实 device-direct 分配使用 `cudaMallocAsync(&ptr, size, stream)`。
3. 每个 allocation 会记录 `device_direct_backend_name`。
4. `uvm_free` 会先查询 allocation 元数据，再决定调用 `cudaFree` 还是 `cudaFreeAsync(ptr, stream)`。
5. C1 总预算逻辑完全保留：先 CAS reserve budget，再执行 CUDA allocation；失败或 cleanup 成功时释放预算；正常 free 成功后释放 live budget。
6. `TRACE_POLICY` 和 `TRACE_GAP_WATCH_ALLOC/FREE` 会输出 `device_direct_backend=<backend>`。
7. `summarize_gap_watch_metrics.py` 会输出 `device_direct_backend_counts`。

### 5.4 CUDA memory pool 调优

可选新增：

```text
VLLM_UVM_DEVICE_DIRECT_POOL_RELEASE_THRESHOLD
```

对应 CUDA API：

```cpp
cudaDeviceGetDefaultMemPool
cudaMemPoolSetAttribute
cudaMemPoolAttrReleaseThreshold
```

目标：

1. 减少频繁向 driver 申请/归还。
2. 降低分配延迟。
3. 控制 pool 常驻大小。

### 5.5 C2 成功标准

和 C1 相比：

1. gap2 fault 不回升。
2. TTFT/TPOT 改善或不恶化。
3. allocator latency 降低。
4. OOM 不增加。
5. free 成功率保持 100%。

推荐首轮 C2 验证命令：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm

PROMPTS=10 \
DEVICE_DIRECT_MAX_TOTAL_BYTES=1048576 \
DEVICE_DIRECT_BACKEND=cuda_malloc_async \
./run_stage_c_attention_p20_ab.sh
```

重点看：

```text
success_signal=True
effectiveness_signal=True
device_peak_live_within_budget=True
device_direct_backend_counts 中出现 cuda_malloc_async
device_direct_actual_records > 0
device_direct_budget_reject_records 可存在
gap_policy_fail=0
Failed requests=0
```

C2 对照实验建议固定相同预算和 prompts，分别跑：

```bash
PROMPTS=10 DEVICE_DIRECT_MAX_TOTAL_BYTES=1048576 DEVICE_DIRECT_BACKEND=cuda_malloc ./run_stage_c_attention_p20_ab.sh
PROMPTS=10 DEVICE_DIRECT_MAX_TOTAL_BYTES=1048576 DEVICE_DIRECT_BACKEND=cuda_malloc_async ./run_stage_c_attention_p20_ab.sh
```

如果 async backend 稳定，再扩展到 p20：

```bash
PROMPTS=20 DEVICE_DIRECT_MAX_TOTAL_BYTES=1048576 DEVICE_DIRECT_BACKEND=cuda_malloc_async ./run_stage_c_attention_p20_ab.sh
```

## 6. 阶段 D：KV 独立预算

### 6.1 为什么不能只在 allocator 层做 KV eviction

KV cache 有 vLLM 语义：

1. block 属于哪个 request。
2. block 是否还活跃。
3. block 是否可 swap。
4. block 是否可重算。
5. block table 如何更新。

allocator 只知道地址和 size，不知道请求语义。因此 allocator 不能单独决定驱逐哪个 KV block。

### 6.2 D 阶段目标

建立 KV 逻辑预算，防止 KV cache 与 runtime scratch、weights resident pool 互相抢占。

目标行为：

```text
KV cache 只能使用 KV budget
KV budget 不足时，只由 vLLM block manager 处理 KV swap/evict/recompute
runtime scratch 不得挤占 KV reserve
weights 不得挤占 KV reserve
```

### 6.3 需要接入的 vLLM 模块

需要重点调查和修改：

```text
vllm/v1/worker/gpu_worker.py
vllm/v1/core/kv_cache_manager.py
vllm/v1/worker/gpu_model_runner.py
vllm/v1/executor/*
vllm/config/cache.py
```

实际路径以当前 vLLM 版本为准，需要用 `rg "kv_cache"` 和 `rg "block"` 进一步定位。

### 6.4 参数设计

建议新增：

```bash
--uvm-kv-budget-bytes <n>
--uvm-kv-swap-enable <0|1>
--uvm-kv-eviction-policy lru|scheduler_aware
```

环境变量：

```text
VLLM_UVM_KV_BUDGET_BYTES
VLLM_UVM_KV_SWAP_ENABLE
VLLM_UVM_KV_EVICTION_POLICY
```

### 6.5 allocator 侧职责

allocator 只做：

1. 标记 KV allocation。
2. 记录 KV bytes。
3. 输出 KV budget telemetry。
4. 在超预算时返回明确 reason 或触发 vLLM 层回调。

allocator 不做：

1. 私自驱逐 KV block。
2. 私自迁移正在使用的 KV。
3. 私自修改 block table。

### 6.6 vLLM 层职责

vLLM block manager 做：

1. 计算 KV block budget。
2. 分配 block 时检查预算。
3. 选择可驱逐 block。
4. swap 到 CPU 或触发 recompute。
5. 更新 block table。
6. 与 scheduler 协同，避免驱逐活跃请求。

### 6.7 D 阶段验证

观察：

1. KV cache allocation bytes。
2. KV budget hit rate。
3. KV eviction count。
4. KV swap in/out bytes。
5. request latency。
6. failed requests。
7. gap2 runtime scratch 是否仍稳定。

成功标准：

1. scratch device-direct 不再导致 KV OOM。
2. KV eviction 只发生在 KV pool。
3. 没有请求级正确性问题。
4. 性能退化可解释、可控。

## 7. 阶段 E：Weights Hot/Cold 分类与 MoE Expert 预取

### 7.1 为什么不要先做完整 weights offload

完整 weights offload + predictive prefetch 是大工程，原因：

1. dense/shared weights 每 token 基本都会访问。
2. PCIe/NVLink 带宽可能成为瓶颈。
3. 预测错误会导致严重 stall。
4. 权重迁移和 runtime scratch / KV 会抢带宽。
5. 需要 scheduler、model runner、allocator、UVM fault trace 联动。

因此 E 阶段应先做统计，再只对 MoE expert weights 做小范围实验。

### 7.2 E0：权重访问热度统计

目标：

1. 识别哪些 weight pages 高频 fault。
2. 识别哪些 expert 权重低频。
3. 识别哪些层 / expert 与请求分布相关。

数据来源：

1. address log 中的 weight region。
2. fault address log。
3. allocator trace。
4. MoE router logits / selected experts。

需要新增分析脚本或扩展 deep dive：

```text
weight_faults_by_region
weight_faults_by_layer
weight_faults_by_expert
read/write ratio
faults_per_unique_page
reuse interval
```

输出示例：

```json
{
  "model.layers.47.mlp.experts.w2_weight": {
    "faults": 12345,
    "unique_pages": 678,
    "avg_faults_per_page": 18.2,
    "read_ratio": 1.0
  }
}
```

### 7.3 E1：MoE expert routing 观测

需要在 vLLM MoE 路径加 trace：

```text
layer_id
expert_id
token_count
batch_id
phase
timestamp
```

候选位置：

```text
vllm/model_executor/layers/fused_moe/layer.py
```

目标：

1. 统计每层 expert 命中频率。
2. 观察 expert 热度是否稳定。
3. 评估是否能预测下一 decode step 的 expert 集合。

### 7.4 E2：Expert 权重预取实验

先只做 trace / prefetch，不做强制 offload：

1. 识别下一步可能访问的 expert。
2. 对这些 expert weight managed range 执行 prefetch。
3. 比较 fault 和 latency。

策略：

```text
prefetch hot experts only
do not prefetch shared dense weights initially
budgeted prefetch staging area
avoid concurrent prefetch flood
```

新增参数：

```bash
--uvm-expert-prefetch-enable <0|1>
--uvm-expert-prefetch-max-bytes <n>
--uvm-expert-prefetch-lookahead-steps <n>
--uvm-expert-prefetch-min-confidence <float>
```

### 7.5 E3：Cold expert CPU residency

只有当 E2 证明 expert 预取有效后，才考虑让冷 expert 更偏向 CPU：

1. 使用 managed memory。
2. 对 cold expert 设置 CPU preferred location 或不主动 prefetch。
3. 对 predicted hot expert 提前 prefetch GPU。
4. 维持 hot expert resident cache。

不建议一开始使用普通 pageable CPU memory，因为恢复到 GPU 的路径和 PyTorch tensor 语义会复杂很多。

### 7.6 E 阶段成功标准

1. expert weight fault 下降。
2. MoE 层 latency 不恶化。
3. prefetch 带宽没有挤压 runtime scratch / KV。
4. 预测命中率可量化。
5. failed requests 为 0。

## 8. 实验矩阵

建议按如下顺序推进：

| 阶段 | 实验 | 关键开关 | 目标 |
| --- | --- | --- | --- |
| C0-A | attention-only p5 | `device_direct`, enable=1, phase=attention | 正确性 |
| C0-B | attention-only p20 | 同上 | fault/性能 |
| C1-A | budget 256MiB | max-total=256MiB | 防 OOM |
| C1-B | budget 512MiB | max-total=512MiB | 容量敏感性 |
| C2-A | cudaMallocAsync p5 | backend=async | 正确性 |
| C2-B | cudaMallocAsync p20 | backend=async | 分配开销 |
| D0 | KV telemetry | kv budget trace-only | 观测 |
| D1 | KV budget enforce | kv budget + vLLM block manager | 隔离 |
| E0 | weight hot/cold stats | analysis only | 识别目标 |
| E1 | expert routing trace | moe trace | 预测依据 |
| E2 | expert prefetch | prefetch only | 降低 weight fault |

## 9. 必须记录的指标

每次实验至少保留：

```text
allocator log
fault stats log
fault address log
auto gap-watch probe summary
auto gap-watch post-main summary
gap-watch metrics summary
benchmark log
deep-dive summary
```

核心指标：

```text
gap2 faults
unknown faults
weight faults
kv faults
avg_faults_per_unique_page
dominant_phase
phase_record_ratios
placement_backend_counts
device_direct_actual_records
device_direct_eligible_records
device_direct_reason_counts
device_direct_peak_live_bytes
device_direct_budget_rejects
TTFT
TPOT
ITL
throughput
failed requests
```

## 10. 风险清单

### 10.1 Device pointer 被 CPU 访问

风险：

```text
cudaMalloc 返回的 device pointer 不能被 CPU dereference
```

缓解：

1. 只允许 `enabled:attention`。
2. 只允许短生命周期 runtime scratch。
3. 出错立即关闭 `--uvm-device-direct-enable`。
4. 后续可加 CPU access sanitizer 或 debug trap。

### 10.2 OOM

风险：

```text
device_direct 占用真实 VRAM
```

缓解：

1. C1 加总预算。
2. 超预算 fallback managed。
3. 保留 reserve。
4. 先小 prompt，再主实验。

### 10.3 cudaMalloc/cudaFree 同步开销

风险：

```text
短命对象高频分配释放导致性能恶化
```

缓解：

1. C2 改 `cudaMallocAsync`。
2. 使用 CUDA memory pool。
3. 统计分配延迟和 p95/p99。

### 10.4 KV eviction 破坏请求正确性

风险：

```text
allocator 不知道 request/block 语义
```

缓解：

1. allocator 只做 telemetry。
2. eviction 由 vLLM block manager 实现。
3. 先 trace-only 再 enforce。

### 10.5 权重预测错误

风险：

```text
prefetch 错误 expert 浪费带宽
```

缓解：

1. 先做 expert routing trace。
2. 使用 confidence threshold。
3. 限制 prefetch bytes。
4. 不预取 shared dense weights。

## 11. 推荐近期任务拆分

### Task 1：完成 C0 attention-only 实验

目标：

1. 跑 p5。
2. 跑 p20。
3. 确认 `placement_backend=device_direct`。
4. 确认无 failed requests。

### Task 2：实现 C1 budget

修改：

1. allocator 新增 max total bytes。
2. run script 新增参数。
3. metrics summary 新增 budget 字段。
4. 文档补充实验命令。

### Task 3：实现 C2 async backend

修改：

1. allocator 新增 backend enum。
2. 支持 `cudaMallocAsync/cudaFreeAsync`。
3. 支持 memory pool release threshold。
4. 比较 cudaMalloc 和 cudaMallocAsync。

### Task 4：KV telemetry

修改：

1. 标记 KV allocation。
2. 输出 KV live bytes。
3. 输出 KV peak bytes。
4. 先不做 eviction。

### Task 5：weight / expert hotness analysis

修改：

1. 扩展 fault address analyzer。
2. 按 region/layer/expert 聚合 weight fault。
3. 在 MoE 层记录 expert routing。
4. 输出 expert hotness report。

## 12. 当前推荐结论

当前最稳妥路线是：

```text
C0 attention-only device_direct
-> C1 device_direct budget
-> C2 cudaMallocAsync/memory pool
-> D KV budget telemetry + block-manager eviction
-> E MoE expert hot/cold + predictive prefetch
```

不要跳过 C1。只要进入真实 `cudaMalloc`，预算就是防 OOM 的第一道安全阀。

不要直接做完整 weights offload。先通过 fault 和 expert routing 统计证明哪些权重值得预取，尤其优先关注 MoE experts，而不是 shared dense weights。

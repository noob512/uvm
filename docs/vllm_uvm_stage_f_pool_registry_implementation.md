# vLLM UVM Stage F Unified Pool Registry 详细实现说明

本文档详细说明 Stage F 的实现方式、修改文件、运行链路、日志字段、metrics 汇总和验收方法。它面向后续继续实现 Stage G/H/I/J/K 的人，目标是让读者能从文档直接对应到代码，理解当前项目如何把 KV cache、model weights、runtime scratch 三类 UVM allocation 纳入统一 pool registry。

Stage F 的关键词是：统一对象登记、per-pool telemetry、free 闭环。它不执行驱逐、swap、offload 或 prefetch。

## 1. 背景

Stage C/D/E 已经分别建立了三类对象的识别能力：

1. Stage C/C2：runtime scratch / workspace 可被识别为热点 gap 候选，并可在严格门控下走 device-direct。
2. Stage D/D2：KV cache allocation 可被识别，KV budget telemetry 已有，D2 还可以在初始化阶段限制 KV blocks。
3. Stage E：weights allocation 可被识别，weight budget telemetry、weight semantic map 和 MoE routing trace 已有。

但 C/D/E 的统计仍然分散。后续要做“哪个 pool 超载，只处理哪个 pool”的策略执行，需要一个统一入口回答：

1. 当前有哪些活跃 managed/device-direct 对象？
2. 每个对象属于 `kv_cache`、`weights` 还是 `runtime_scratch`？
3. 每个 pool 的 allocation/free 是否能闭环？
4. 每个 pool 当前 live bytes 和 peak live bytes 是多少？
5. 策略执行器后续应该从哪个对象集合中挑候选？

Stage F 正是为这些问题建立统一数据平面。

## 2. Stage F 的目标

Stage F 实现一个 allocator 侧的统一 Pool Registry：

```text
AllocationClass
  -> PoolKind
  -> AllocationInfo
  -> active_allocations object index
  -> TRACE_POOL_ALLOC / TRACE_POOL_FREE
  -> pool-level metrics JSON
```

Stage F 的成功不以性能提升为目标，而以“对象分类稳定、登记完整、free 可回收、metrics 可验证”为目标。

## 3. Stage F 明确不做什么

Stage F 不做：

1. KV runtime eviction。
2. KV swap / recompute。
3. weight offload。
4. weight prefetch。
5. MoE expert prefetch。
6. runtime scratch eviction。
7. runtime scratch admission fallback。
8. UVM driver 侧 page eviction 策略修改。

这些分别属于后续 Stage G/H/I/J/K。

## 4. Pool 抽象

Stage F 在 allocator 中新增 `PoolKind`：

```cpp
enum class PoolKind {
    KvCache,
    Weights,
    RuntimeScratch,
    OtherManaged,
};
```

当前映射关系：

```text
AllocationClass::KvPersistent
  -> PoolKind::KvCache
  -> "kv_cache"

AllocationClass::WeightPersistent
  -> PoolKind::Weights
  -> "weights"

AllocationClass::RuntimeScratch
AllocationClass::RuntimeWorkspace
AllocationClass::WarmupWorkspace
  -> PoolKind::RuntimeScratch
  -> "runtime_scratch"

其他类别
  -> PoolKind::OtherManaged
  -> "other_managed"
```

其中 `OtherManaged` 是兜底分类。当前 Stage F 默认不把它纳入成功标准，也不输出 `TRACE_POOL_ALLOC`，目的是避免 unknown allocation 噪音污染 KV/weights/scratch 三个核心 pool。

## 5. 为什么不是三个裸列表

最初设想可以是三个列表：

```text
kv_list
weights_list
scratch_list
```

但裸列表不够。后续策略至少需要：

1. 地址范围：`ptr/end/size`
2. 所属 pool：`kv_cache/weights/runtime_scratch`
3. 启发式类别：`predicted_class`
4. phase：比如 `load_model`、`enabled:attention`、`enabled:moe`
5. backend：`managed` 或 `device_direct`
6. lifetime：分配到释放之间的时间
7. budget 状态：KV/weight/scratch/device-direct 是否超预算
8. active/freed 状态
9. free 时的反向查找能力

所以 Stage F 没有单独维护裸 `std::list<void*>`，而是扩展已有 `AllocationInfo`，让 `active_allocations[ptr]` 作为统一对象索引。pool 只是 `AllocationInfo` 里的一个稳定维度。

## 6. 修改文件总览

Stage F 主要修改这些文件：

1. `workloads/vllm/vllm/uvm_test/uvm_allocator.cpp`
2. `workloads/vllm/run_kv_fault_ratio.sh`
3. `workloads/vllm/summarize_gap_watch_metrics.py`
4. `workloads/vllm/check_stage_f_success.py`
5. `docs/vllm_uvm_independent_pool_eviction_prefetch_plan.md`
6. `docs/vllm_uvm_stage_f_pool_registry_implementation.md`

其中 allocator 是核心，runner 负责开关注入，summarizer 负责日志转 JSON，check 脚本负责一键验收。

## 7. `uvm_allocator.cpp` 修改详解

### 7.1 新增配置开关

Stage F 新增环境变量：

```text
VLLM_UVM_POOL_REGISTRY_ENABLE=0|1
```

allocator 初始化时读取：

```cpp
pool_registry_enabled =
    read_bool_from_env("VLLM_UVM_POOL_REGISTRY_ENABLE", false) ? 1 : 0;
```

默认关闭，原因是：

1. 避免非 Stage F 实验额外输出大量 pool trace。
2. 避免无关运行保留更多 active allocation metadata。
3. 保持 Stage C/D/E 原有行为默认不变。

Stage F 验收脚本会显式传入：

```bash
--uvm-pool-registry-enable 1
```

### 7.2 session header 新增字段

allocator session header 会输出：

```text
pool_registry_enabled=<0|1>
```

这让后处理脚本即使没有看到 pool trace，也能知道这次实验是否真的开启了 Stage F。

### 7.3 新增 `PoolKind` 与映射函数

新增函数：

```cpp
static PoolKind pool_kind_for_allocation(AllocationClass alloc_class);
static const char* pool_kind_to_string(PoolKind pool_kind);
static PoolKind pool_kind_from_string(const std::string& value);
```

用途：

1. malloc 路径：`AllocationClass -> PoolKind -> string`
2. free 路径：从 `AllocationInfo.pool_kind_name` 还原为 `PoolKind`
3. trace 输出：统一输出小写字符串，如 `kv_cache`
4. metrics parser：按 `pool_kind` 字段聚合

### 7.4 扩展 `AllocationInfo`

Stage F 在 `AllocationInfo` 中新增：

```cpp
bool pool_registry_tracked;
std::string pool_kind_name;
std::string pool_object_state;
```

含义：

1. `pool_registry_tracked`
   - 该 allocation 是否被 Stage F registry 管理。
   - free 路径只对这个字段为 true 的对象回收 pool counters。

2. `pool_kind_name`
   - `kv_cache` / `weights` / `runtime_scratch`
   - 存成 string 是为了 free 路径不依赖原始 enum 生命周期，也方便日志输出。

3. `pool_object_state`
   - malloc 后为 `active`
   - free 成功后更新为 `freed`
   - 不纳入 registry 时为 `untracked`

### 7.5 新增全局 counters

Stage F 新增 registry 总计数：

```text
pool_registry_tracked_allocs
pool_registry_free_success_allocs
pool_registry_live_objects
pool_registry_peak_live_objects
```

新增 KV pool counters：

```text
pool_kv_allocs
pool_kv_requested_bytes
pool_kv_live_bytes
pool_kv_peak_live_bytes
pool_kv_free_success_allocs
```

新增 weight pool counters：

```text
pool_weight_allocs
pool_weight_requested_bytes
pool_weight_live_bytes
pool_weight_peak_live_bytes
pool_weight_free_success_allocs
```

新增 runtime scratch pool counters：

```text
pool_runtime_scratch_allocs
pool_runtime_scratch_requested_bytes
pool_runtime_scratch_live_bytes
pool_runtime_scratch_peak_live_bytes
pool_runtime_scratch_free_success_allocs
```

这些 counters 都是 atomic，适配 allocator 多线程调用场景。

### 7.6 分配登记函数

Stage F 新增：

```cpp
static void record_pool_registry_allocation(PoolKind pool_kind, size_t size);
```

执行逻辑：

1. `pool_registry_tracked_allocs += 1`
2. `pool_registry_live_objects += 1`
3. 更新 `pool_registry_peak_live_objects`
4. 按 pool 更新 allocation count、requested bytes、live bytes、peak live bytes

例如 weights：

```cpp
pool_weight_allocs.fetch_add(1);
pool_weight_requested_bytes.fetch_add(size);
size_t current_live = pool_weight_live_bytes.fetch_add(size) + size;
update_pool_peak_live(pool_weight_peak_live_bytes, current_live);
```

### 7.7 释放回收函数

Stage F 新增：

```cpp
static void release_pool_registry_allocation(PoolKind pool_kind, size_t size);
```

执行逻辑：

1. `pool_registry_live_objects -= 1`
2. `pool_registry_free_success_allocs += 1`
3. 按 pool 回收 live bytes
4. 按 pool 更新 free success

回收 live bytes 使用防下溢逻辑：

```cpp
release_pool_live_bytes(pool_weight_live_bytes, size);
```

这样即使出现异常重复释放或统计不一致，也不会把 live bytes 变成无符号整数下溢后的巨大值。

### 7.8 新增 `TRACE_POOL_ALLOC`

函数：

```cpp
trace_pool_alloc_event(...)
```

只有满足以下条件才输出：

```text
log_file 存在
pool_registry_enabled == 1
pool_kind != OtherManaged
```

输出字段包括：

```text
alloc_id
ptr
end
size_bytes
size_bucket
device
phase
predicted_class
pool_kind
pool_object_state
pool_registry_enabled
placement_backend
device_direct_backend
scratch_pool_tracked
scratch_pool_eligible
scratch_pool_device_direct
scratch_pool_over_budget
scratch_pool_reason
pool_registry_live_objects
pool_registry_peak_live_objects
pool_kv_live_bytes
pool_weight_live_bytes
pool_runtime_scratch_live_bytes
```

示例：

```text
TRACE_POOL_ALLOC alloc_id=123 ptr=0x... end=0x...
size_bytes=1048576 phase=load_model
predicted_class=weight_persistent pool_kind=weights
pool_object_state=active placement_backend=managed
pool_registry_live_objects=42 pool_weight_live_bytes=...
```

### 7.9 新增 `TRACE_POOL_FREE`

函数：

```cpp
trace_pool_free_event(...)
```

只有满足以下条件才输出：

```text
log_file 存在
pool_registry_enabled == 1
info != nullptr
info->pool_registry_tracked == true
```

输出字段包括：

```text
free_id
ptr
end
size_bytes
device
phase
alloc_id
alloc_phase
lifetime_s
predicted_class
pool_kind
pool_object_state=freed
placement_backend
device_direct_backend
pool_registry_live_objects
pool_registry_peak_live_objects
pool_kv_live_bytes
pool_weight_live_bytes
pool_runtime_scratch_live_bytes
scratch_pool_tracked
scratch_pool_eligible
scratch_pool_device_direct
scratch_pool_reason
```

`TRACE_POOL_FREE` 的核心价值是证明：

1. malloc 时登记过的对象，free 时能找到原始 metadata。
2. pool live bytes 能回收。
3. lifecycle / lifetime 可被后续策略使用。

### 7.10 malloc 路径时序

Stage F 在 `uvm_malloc` 中的时序如下：

```text
1. cudaMallocManaged 先分配 managed ptr
2. 根据 phase + size + device 分类 AllocationClass
3. Stage D/E 更新 KV / weight budget telemetry
4. AllocationClass -> PoolKind
5. 如果 pool registry 开启且 pool_kind 不是 OtherManaged：
   - record_pool_registry_allocation()
   - pool_object_state = active
6. Stage G scratch pool / Stage C device-direct 逻辑继续判断
7. 如果最终走 device-direct，placement_backend 会变成 device_direct
8. 将 AllocationInfo 写入 active_allocations
9. trace_pool_alloc_event()
10. 继续输出 TRACE_POLICY / TRACE_GAP_WATCH_ALLOC 等已有日志
```

关键点：

```cpp
bool store_active_info =
    size >= trace_min_bytes ||
    log_unknown_detail ||
    log_gap_watch ||
    placement_backend == "device_direct" ||
    kv_budget_tracked ||
    weight_budget_tracked ||
    pool_registry_tracked;
```

Stage F 把 `pool_registry_tracked` 加入 `store_active_info` 条件，确保即使对象不是大块 trace，也能在 free 时找到 metadata。

### 7.11 free 路径时序

Stage F 在 `uvm_free` 中的时序如下：

```text
1. 从 active_allocations 按 ptr 找 AllocationInfo
2. 执行真实 cudaFree/cudaFreeAsync
3. 如果 device-direct，回收 Stage C device-direct budget
4. 如果 KV，回收 Stage D KV budget
5. 如果 weights，回收 Stage E weight budget
6. 如果 pool_registry_tracked，回收 Stage F pool registry counters
7. info.pool_object_state = freed
8. 输出 TRACE_FREE / TRACE_POOL_FREE / TRACE_GAP_WATCH_FREE
```

Stage F 的回收逻辑：

```cpp
if (has_info && info.pool_registry_tracked && err == cudaSuccess) {
    release_pool_registry_allocation(
        pool_kind_from_string(info.pool_kind_name),
        info.size
    );
    info.pool_object_state = "freed";
}
```

注意：只有真实 CUDA free 成功后才回收 pool counters，避免 free 失败导致 registry 与实际对象状态不一致。

### 7.12 reset 和 close summary

`uvm_reset_all_stats()` 会清空 Stage F counters：

```text
pool_registry_tracked_allocs
pool_registry_free_success_allocs
pool_registry_live_objects
pool_registry_peak_live_objects
pool_kv_*
pool_weight_*
pool_runtime_scratch_*
```

`uvm_close_log()` 会输出 session summary：

```text
Pool registry enabled
Pool registry tracked allocations
Pool registry free success
Pool registry live objects
Pool registry peak live objects
Pool kv allocations
Pool kv requested bytes
Pool kv live bytes
Pool kv peak live bytes
Pool kv free success
Pool weight allocations
Pool weight requested bytes
Pool weight live bytes
Pool weight peak live bytes
Pool weight free success
Pool runtime scratch allocations
Pool runtime scratch requested bytes
Pool runtime scratch live bytes
Pool runtime scratch peak live bytes
Pool runtime scratch free success
```

summary 是 trace parser 的兜底来源。如果某些运行只拿到了 session summary，也能恢复部分 Stage F metrics。

## 8. `run_kv_fault_ratio.sh` 修改详解

Stage F 在 runner 中新增变量：

```bash
UVM_POOL_REGISTRY_ENABLE="${VLLM_UVM_POOL_REGISTRY_ENABLE:-0}"
```

新增 CLI 参数：

```bash
--uvm-pool-registry-enable <0|1>
```

新增校验：

```bash
[[ "$UVM_POOL_REGISTRY_ENABLE" =~ ^[01]$ ]] ||
  die "--uvm-pool-registry-enable must be 0 or 1"
```

启动 server 时注入：

```bash
VLLM_UVM_POOL_REGISTRY_ENABLE='$UVM_POOL_REGISTRY_ENABLE'
```

Stage F check 脚本会把它设为 1：

```bash
--uvm-pool-registry-enable 1
```

## 9. `summarize_gap_watch_metrics.py` 修改详解

虽然名字仍然叫 `summarize_gap_watch_metrics.py`，但它现在是 C/D/E/F/G 共用的 allocator trace summary 工具。

Stage F 增加两类解析：

```text
TRACE_POOL_ALLOC
TRACE_POOL_FREE
```

`TRACE_POOL_ALLOC` 解析逻辑：

1. `pool_alloc_records += 1`
2. 按 `pool_kind` 统计 `pool_kind_counts`
3. 按 `predicted_class` 统计 `pool_class_counter`
4. 按 `phase` 统计 `pool_phase_counter`
5. 按 `placement_backend` 统计 `pool_placement_counter`
6. 累加 `pool_alloc_bytes_by_kind`
7. 收集 live object / live bytes 快照

`TRACE_POOL_FREE` 解析逻辑：

1. `pool_free_records += 1`
2. 按 `pool_kind` 累加 `pool_free_bytes_by_kind`
3. 收集释放后的 live object / live bytes 快照

最终 JSON 中 Stage F 相关字段包括：

```json
{
  "pool_registry_enabled": true,
  "pool_registry_alloc_records": 4541,
  "pool_registry_free_records": 3184,
  "pool_registry_live_objects": 1356,
  "pool_registry_peak_live_objects": 1373,
  "pool_kind_counts": {
    "runtime_scratch": 3179,
    "weights": 1314,
    "kv_cache": 48
  },
  "pool_alloc_bytes_by_kind": {
    "runtime_scratch": 12914724594,
    "weights": 31237626924,
    "kv_cache": 6442450944
  },
  "pool_kv_live_bytes": 6442450944,
  "pool_weight_live_bytes": 31185032516,
  "pool_runtime_scratch_live_bytes": 16908288
}
```

这些字段是 `check_stage_f_success.py` 和后续 Stage G/H/I 的基础输入。

## 10. `check_stage_f_success.py` 修改详解

Stage F 新增一键验收脚本：

```text
workloads/vllm/check_stage_f_success.py
```

默认行为：

1. 创建 `/tmp/vllm_stage_f_success_check_<timestamp>`。
2. 启动 `run_kv_fault_ratio.sh`。
3. 开启 `--uvm-pool-registry-enable 1`。
4. 默认加 `--no-bench`，只等 server ready 和 KV range 出现。
5. 读取 allocator trace。
6. 调用 `summarize_gap_watch_metrics.py` 生成 metrics JSON。
7. 验证 KV 和 weights pool 记录。

默认命令：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./check_stage_f_success.py
```

带 benchmark 命令：

```bash
./check_stage_f_success.py --run-bench --require-runtime-scratch
```

复用已有日志：

```bash
./check_stage_f_success.py \
  --skip-run \
  --allocator-log /tmp/vllm_stage_f_success_check_xxx/vllm_uvm_allocator_trace_stage_f.log
```

或：

```bash
./check_stage_f_success.py \
  --skip-run \
  --metrics-json /tmp/vllm_stage_f_success_check_xxx/vllm_stage_f_pool_registry_metrics.json
```

## 11. Stage F 验收标准

默认验收：

```text
metrics_json_present=True
pool_registry_enabled=True
pool_registry_alloc_records_present=True
pool_registry_peak_live_objects_positive=True
kv_cache_pool_records_present=True
weights_pool_records_present=True
kv_and_weight_bytes_positive=True
runner_log_clean=True
```

如果传入 `--run-bench` 或 `--require-runtime-scratch`，还要求：

```text
runtime_scratch_pool_records_if_required=True
```

这样设计的原因：

1. 不跑 benchmark 时，server startup 一定会加载 weights 和初始化 KV，因此 weights/KV 应出现。
2. runtime scratch 多数发生在真实推理阶段，所以默认不强制要求。
3. 一旦跑 benchmark，runtime scratch 应该出现，否则说明 runtime phase 的对象没有被纳入 registry。

## 12. 典型实验结果解释

你之前运行过：

```bash
./check_stage_f_success.py --run-bench --require-runtime-scratch
```

实验结果核心字段：

```text
pool_registry_enabled=True
pool_registry_alloc_records=4541
pool_registry_peak_live_objects=1373
kv_cache_records=48
kv_cache_bytes=6442450944
weight_records=1314
weight_bytes=31237626924
runtime_scratch_records=3179
runtime_scratch_bytes=12914724594
pool_kv_live_bytes=6442450944
pool_weight_live_bytes=31185032516
pool_runtime_scratch_live_bytes=16908288
Stage F Success Check: PASS
```

解释：

1. `pool_registry_enabled=True`
   - Stage F 开关确实生效。

2. `pool_registry_alloc_records=4541`
   - 本次运行中共有 4541 次 allocation 被登记到 `kv_cache/weights/runtime_scratch` 三类 pool。

3. `pool_registry_peak_live_objects=1373`
   - 同一时刻最多有 1373 个 pool object 存活。

4. `kv_cache_records=48`
   - KV cache allocation 被稳定识别并登记。
   - `kv_cache_bytes=6442450944` 约为 6 GiB，和 Stage D/D2 的 KV cache 规模一致。

5. `weight_records=1314`
   - 模型权重 allocation 被稳定识别并登记。
   - `weight_bytes=31237626924` 约为 29.1 GiB。

6. `runtime_scratch_records=3179`
   - benchmark 运行阶段产生的大量临时 runtime allocation 被登记。
   - `runtime_scratch_bytes=12914724594` 是累计分配字节，不等于最终 live bytes。

7. `pool_runtime_scratch_live_bytes=16908288`
   - runtime scratch 生命周期短，大量对象已释放，所以最终 live bytes 只有约 16 MiB。

8. `runner_log_clean=True`
   - runner 没有 parse failure，也没有 server early exit。

结论：Stage F 的三类对象 registry 已经闭环，后续可以在这个对象索引之上继续做 Stage G scratch admission、Stage H weights hot/cold plan、Stage I expert prefetch。

## 13. 和 Stage C/D/E/G/H/I 的关系

Stage F 不替代 Stage C/D/E，而是把它们统一。

Stage C 提供：

1. gap-watch。
2. device-direct eligibility。
3. device-direct backend 和预算。

Stage D 提供：

1. KV allocation 识别。
2. KV budget telemetry。
3. KV 初始化预算约束。

Stage E 提供：

1. weight allocation 识别。
2. weight budget telemetry。
3. weight semantic map。
4. MoE routing trace。

Stage F 提供：

1. `kv_cache/weights/runtime_scratch` 统一登记。
2. `TRACE_POOL_ALLOC` / `TRACE_POOL_FREE`。
3. pool live objects / live bytes / peak live bytes。
4. 后续策略执行器的共同对象模型。

Stage G 使用 Stage F 的 runtime scratch pool 统计，进一步做 scratch admission/fallback。

Stage H 使用 Stage E semantic map 和 Stage F/allocator metrics，生成 weights hot/cold trace-only plan。

Stage I 在 Stage H 之后做 expert prefetch 执行验证，但仍依赖 Stage F/allocator metrics 判断 weights pool 是否正常。

## 14. 设计边界和风险控制

Stage F 的风险控制点：

1. 默认关闭，需要显式 `VLLM_UVM_POOL_REGISTRY_ENABLE=1`。
2. 只输出 telemetry，不改变 allocation backend。
3. 不对 `OtherManaged` 输出 pool trace，降低噪音。
4. free 只有在 CUDA free 成功后才回收 counters。
5. `pool_live_bytes` 回收使用防下溢逻辑。
6. 所有 counters 使用 atomic，适配多线程 allocator 调用。
7. 保留已有 Stage C/D/E 日志和 counters，不破坏前序阶段。

## 15. 常见问题

### 15.1 为什么默认 `check_stage_f_success.py` 不要求 runtime scratch？

因为默认不跑 benchmark，只启动 server 到 ready/KV range。这个阶段必然有 weights 和 KV，但不一定有真实推理路径中的 runtime scratch。加 `--run-bench` 后才应该要求 runtime scratch。

### 15.2 `pool_alloc_bytes_by_kind` 为什么大于 live bytes？

`pool_alloc_bytes_by_kind` 是累计分配字节。runtime scratch 会频繁分配和释放，因此累计 bytes 可以很大，但最终 live bytes 很小。

### 15.3 为什么 `pool_weight_live_bytes` 和 `pool_alloc_bytes_by_kind["weights"]` 不完全相等？

`pool_alloc_bytes_by_kind["weights"]` 来自 `TRACE_POOL_ALLOC` 累计，`pool_weight_live_bytes` 是当前 live 快照。权重通常长期存活，所以两者接近；差异可能来自少量临时/重复/释放对象或 trace 时刻不同。

### 15.4 Stage F 是否能直接做 eviction？

不能。Stage F 只知道对象的 allocator 级元数据，不知道 KV block 是否 active、weight tensor 是否可安全 offload、CUDA kernel 是否仍会访问该对象。真实执行必须在后续阶段通过 vLLM 上层安全点完成。

## 16. 最小阅读路径

建议按下面顺序读代码：

1. `workloads/vllm/vllm/uvm_test/uvm_allocator.cpp`
   - 搜索 `PoolKind`
   - 搜索 `record_pool_registry_allocation`
   - 搜索 `trace_pool_alloc_event`
   - 搜索 `release_pool_registry_allocation`
   - 搜索 `trace_pool_free_event`

2. `workloads/vllm/run_kv_fault_ratio.sh`
   - 搜索 `uvm-pool-registry-enable`
   - 搜索 `VLLM_UVM_POOL_REGISTRY_ENABLE`

3. `workloads/vllm/summarize_gap_watch_metrics.py`
   - 搜索 `TRACE_POOL_ALLOC`
   - 搜索 `TRACE_POOL_FREE`
   - 搜索 `pool_kind_counts`

4. `workloads/vllm/check_stage_f_success.py`
   - 看 `run_experiment()`
   - 看 `validate_metrics()`

5. `docs/vllm_uvm_independent_pool_eviction_prefetch_plan.md`
   - 看 Stage F 到 Stage K 的整体路线。

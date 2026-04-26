# vLLM UVM Gap2 Device-Direct Roadmap

## 1. 目标

本文档描述从当前 `gap2 + prefetch` 方案继续演进到 `device_direct` 的实施路线。

当前项目已经证明：

1. `gap2` 是稳定出现的主导 unknown 热点。
2. `gap2` 主要对应 `phase=enabled` 的短命 `unknown_managed` allocation。
3. same-run probe 能在同一 vLLM server 进程内定位本轮真实 `gap2` 地址。
4. 对 gap2 命中的目标 allocation 做 `managed_prefetch_gpu` 能显著降低缺页错误。

下一阶段目标不是立刻替换所有 UVM allocation，而是引入一条更保守、可回滚、可观测的 GPU-only 分配路径：

```text
observe -> managed_prefetch_gpu -> device_direct_trace -> device_direct
```

最终要回答的问题是：

1. 哪些 gap2 hotspot allocation 可以脱离 UVM？
2. 直接分配到 GPU 显存后，fault 是否继续下降？
3. 这样做是否会带来显存压力、CPU 访问错误或 allocator 开销？

---

## 2. 总体原则

`device_direct` 不能按固定虚拟地址硬编码。

原因是：

1. `gap2` 地址每次 vLLM 进程重启都会变化。
2. gap2 是“本轮地址空间中的热点地址池”，不是一个静态对象。
3. allocator 在分配前通常不知道这次分配最终会落到哪个 UVM gap。

因此正确策略是：

1. 用 same-run probe 发现本轮 gap2。
2. 用 allocator trace 确认命中 gap2 的对象类型。
3. 把这类对象抽象成新的 hot class。
4. 后续按 `phase + size + class + hot-gap feedback + lifetime` 决定 backend。

推荐引入的新语义类：

1. `gap_hot_runtime_scratch`
2. `gap_hot_runtime_workspace`

初始阶段可以只实现第一个。

---

## 3. 阶段 A：保留 managed + gap2 prefetch

### 3.1 当前状态

这一阶段已经基本完成。

已有能力：

1. `auto-gap-watch` same-run probe。
2. control file 热更新 watched gap。
3. `target_class + policy_action` 控制。
4. `managed_prefetch_gpu` 动作。
5. `summarize_gap_watch_metrics.py` 输出命中和成功指标。
6. observe/prefetch A/B 对照。

### 3.2 保留原因

这仍然是后续所有更激进策略的基线。

任何 `device_direct` 实验都必须至少和下面两组比较：

1. `observe`
2. `prefetch`

否则无法判断 device-direct 的收益是来自 GPU-only 分配，还是仅仅来自已有 prefetch。

### 3.3 成功标准

阶段 A 的成功标准是：

1. `observe` 组 `gap_policy_records=0`。
2. `prefetch` 组 `gap_policy_records>0`。
3. `prefetch` 组 `gap_policy_success == gap_policy_records`。
4. `prefetch` 组 `gap2 faults` 低于 `observe` 组。
5. `prefetch` 组 `total_unknown_faults` 低于 `observe` 组。

### 3.4 后续需要修正的小问题

当前 `post-main` summary 在 `--no-write-control` 模式下会重新计算推荐动作，
因此显示的 `effective_policy_action` 可能回到推荐值 `prefetch`。

这不影响策略是否执行的事实判断，因为真正的判断依据是：

1. `gap_policy_records`
2. `dominant_action`
3. allocator trace 中的 `policy_source`

但后续建议把 compare 脚本改为同时读取 probe summary 中的 effective action。

---

## 4. 阶段 B：新增 `device_direct` 动作，但默认 trace-only

### 4.1 目标

先让 allocator 能表达 `device_direct`，但不改变实际分配行为。

也就是说，阶段 B 只做：

1. 新增动作枚举。
2. 新增配置和日志字段。
3. 新增 dry-run 判定。
4. 输出“如果启用 device_direct，这次 allocation 会被选中”的证据。

不做：

1. 不调用 `cudaMalloc`。
2. 不改变 `cudaMallocManaged` 主路径。
3. 不改变 free 后端。

### 4.2 建议修改文件

主要修改：

1. [uvm_allocator.cpp](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/uvm_allocator.cpp)
2. [discover_gap_watch.py](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/discover_gap_watch.py)
3. [run_kv_fault_ratio.sh](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/run_kv_fault_ratio.sh)
4. [summarize_gap_watch_metrics.py](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/summarize_gap_watch_metrics.py)
5. [compare_gap_watch_ab.py](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/compare_gap_watch_ab.py)

### 4.3 `PolicyAction` 扩展

建议新增：

```cpp
enum class PolicyAction {
    ManagedDefault,
    ManagedPrefetchGpu,
    ManagedAdvisePrefetchGpu,
    DeviceDirectTrace,
    DeviceDirect,
};
```

动作语义：

1. `device_direct_trace`
   - 只标记 would-use-device-direct
   - 实际仍走 `cudaMallocManaged`
2. `device_direct`
   - 真正走 GPU-only 分配

### 4.4 control file 扩展

`policy_action` 支持新增值：

```text
policy_action=device_direct_trace
policy_action=device_direct
```

初期只允许脚本传入 `device_direct_trace`。

### 4.5 日志字段扩展

`TRACE_POLICY` 建议新增：

1. `placement_backend`
2. `device_direct_eligible`
3. `device_direct_reason`
4. `cpu_access_risk`
5. `hot_gap_match`

示例：

```text
TRACE_POLICY alloc_id=... predicted_class=gap_hot_runtime_scratch
action=device_direct_trace placement_backend=managed
device_direct_eligible=1 device_direct_reason=hot_gap_short_lived_small
cpu_access_risk=unknown hot_gap_match=1
```

### 4.6 成功标准

阶段 B 成功标准：

1. observe/prefetch 行为不变。
2. `device_direct_trace` 不改变 fault 和性能。
3. 日志能稳定统计 would-device-direct 的 allocation 数量。
4. would-device-direct 对象主要来自 `gap2`、`phase=enabled`、短生命周期对象。

---

## 5. 阶段 C：严格条件下启用 GPU-only allocation

### 5.1 目标

对满足严格条件的 gap2 hotspot allocation 使用 GPU-only 分配。

建议初始实现使用：

1. `cudaMallocAsync`
2. `cudaFreeAsync`

如果当前环境或 PyTorch pluggable allocator 流语义不稳定，再退回：

1. `cudaMalloc`
2. `cudaFree`

### 5.2 为什么优先 `cudaMallocAsync`

gap2 对象的中位生命周期大约是 1ms 级别。

如果用裸 `cudaMalloc/cudaFree`，高频小块分配可能引入很大同步和 allocator 开销。

`cudaMallocAsync` 更适合：

1. 高频短命 allocation。
2. 和 stream 绑定的生命周期。
3. 后续接入 CUDA memory pool。

### 5.3 必须新增 allocation backend 记录

当前 `uvm_free` 默认认为所有指针都来自 `cudaMallocManaged`。

启用 device direct 后必须记录每个指针的真实 backend：

```cpp
enum class AllocationBackend {
    Managed,
    DeviceDirect,
    DeviceDirectAsync,
};
```

`AllocationInfo` 需要新增：

1. `backend`
2. `backend_name`
3. `free_api`
4. `device_direct_eligible`
5. `device_direct_reason`

free 时必须按 backend 调用不同释放 API：

1. `Managed` -> `cudaFree`
2. `DeviceDirect` -> `cudaFree`
3. `DeviceDirectAsync` -> `cudaFreeAsync`

虽然 `Managed` 和 `DeviceDirect` 都可能用 `cudaFree`，但日志里必须分开。

### 5.4 初始启用条件

建议第一版只允许满足全部条件的 allocation 使用 device direct：

1. `phase == enabled`
2. `predicted_class == unknown_managed` 或 `gap_hot_runtime_scratch`
3. `gap_overlap_bytes > 0`
4. `gap_watch_target_class` 匹配
5. `size >= 4096`
6. `size <= 1 MiB`
7. `gap_watch_policy_action == device_direct`
8. `device_direct_enable == 1`
9. 没有 CPU access 证据

其中 `size <= 1 MiB` 是保守起点。

如果验证稳定，再扩展到：

1. `size <= 16 MiB`

### 5.5 CPU access 风险

GPU-only allocation 的最大风险是 CPU 访问。

`cudaMallocManaged` 允许 CPU/GPU 访问同一指针，但 `cudaMalloc` 不允许 CPU 直接解引用。

因此阶段 C 必须保守：

1. 只对 runtime 临时对象启用。
2. 不对 weight / kv / warmup persistent 启用。
3. 不对任何已知可能被 CPU 读取的对象启用。
4. 出现异常时可以一键回退到 managed。

### 5.6 建议新增运行时开关

环境变量：

```text
VLLM_UVM_DEVICE_DIRECT_ENABLE=0
VLLM_UVM_DEVICE_DIRECT_MAX_BYTES=1048576
VLLM_UVM_DEVICE_DIRECT_MIN_BYTES=4096
VLLM_UVM_DEVICE_DIRECT_REQUIRE_GAP_MATCH=1
VLLM_UVM_DEVICE_DIRECT_REQUIRE_TARGET_CLASS=1
VLLM_UVM_DEVICE_DIRECT_BACKEND=cudaMallocAsync
```

脚本参数：

```text
--uvm-device-direct-enable <0|1>
--uvm-device-direct-min-bytes <n>
--uvm-device-direct-max-bytes <n>
--uvm-device-direct-backend <cudaMalloc|cudaMallocAsync>
```

### 5.7 阶段 C 成功标准

功能成功：

1. `device_direct` 组能完整跑完 probe + main。
2. allocator log 中出现 `placement_backend=device_direct`。
3. `device_direct_allocs > 0`。
4. `device_direct_free_success == device_direct_allocs`。
5. 无 CUDA illegal access。
6. 无 PyTorch allocator crash。

效果成功：

1. `device_direct` 组 gap2 fault 低于 `prefetch` 组。
2. `device_direct` 组 total unknown fault 低于 `prefetch` 组。
3. TTFT / TPOT / ITL 不恶化。
4. GPU 显存峰值没有越过可接受边界。

---

## 6. 阶段 D：三组 A/B/C 实验

### 6.1 实验组

必须至少比较三组：

1. `observe`
2. `prefetch`
3. `device_direct`

可选加入：

1. `device_direct_trace`
2. `advise_prefetch`

### 6.2 每组固定条件

每组都保持：

1. 同模型
2. 同数据集
3. 同 `--prompts`
4. 同 `--request-rate`
5. 同 `--output-len`
6. 同 `--seed`
7. 同 `--auto-gap-watch-probe-prompts`
8. 同 `--auto-gap-watch-target-gap`
9. 同 `--uvm-gap-watch-target-class` 或 auto recommended class

唯一变量：

1. `policy_action`

### 6.3 主要指标

必须记录：

1. `gap2 faults`
2. `total_unknown_faults`
3. `avg_faults_per_unique_page`
4. `gap_policy_records`
5. `gap_policy_success`
6. `device_direct_allocs`
7. `device_direct_bytes`
8. `device_direct_free_success`
9. `Mean TTFT`
10. `Mean TPOT`
11. `Mean ITL`
12. `Benchmark duration`
13. `Output token throughput`
14. GPU memory peak

### 6.4 报告判断标准

判断 `prefetch` 有效：

1. `prefetch.gap2_faults < observe.gap2_faults`
2. `prefetch.total_unknown_faults < observe.total_unknown_faults`
3. `prefetch.gap_policy_success > 0`

判断 `device_direct` 有效：

1. `device_direct.gap2_faults < prefetch.gap2_faults`
2. `device_direct.total_unknown_faults < prefetch.total_unknown_faults`
3. `device_direct.device_direct_allocs > 0`
4. `device_direct` 性能不恶化
5. 无稳定性错误

---

## 7. 推荐实现顺序

### 7.1 第一步：修正 A/B 报告展示

当前 compare 脚本应读取 probe summary 的 effective action，而不是只看 post-main summary。

建议把 `compare_gap_watch_ab.py` 输入扩展为：

1. `--observe-probe`
2. `--prefetch-probe`

这样最终报告能准确显示：

1. observe 组 effective action 是 `observe`
2. prefetch 组 effective action 是 `prefetch`

### 7.2 第二步：实现 `device_direct_trace`

只增加日志，不改变行为。

这样可以回答：

1. 如果启用 device_direct，会选中多少 allocation？
2. 这些 allocation 的 size 分布是什么？
3. 是否确实集中在 gap2？
4. 是否都是短生命周期？

### 7.3 第三步：实现 backend tracking

在真正调用 `cudaMalloc` 前，必须先完成 backend tracking。

没有 backend tracking，`uvm_free` 无法可靠释放不同来源的指针。

### 7.4 第四步：小尺寸 device_direct

第一版只开：

1. `size <= 1 MiB`
2. `phase=enabled`
3. `target_class=unknown_managed`
4. `gap_overlap_bytes > 0`

### 7.5 第五步：扩展到更大 bucket

如果第一版稳定，再尝试：

1. `size <= 4 MiB`
2. `size <= 16 MiB`

每次只改一个阈值。

---

## 8. 风险和回滚

### 8.1 主要风险

1. CPU 访问 device pointer 导致崩溃。
2. GPU 显存压力增加。
3. 高频 `cudaMalloc/cudaFree` 造成额外开销。
4. stream 生命周期和 `cudaFreeAsync` 顺序不一致。
5. PyTorch 对 pluggable allocator 的隐式假设被破坏。

### 8.2 必须保留的回滚路径

任何时候都应该能通过环境变量回退：

```text
VLLM_UVM_DEVICE_DIRECT_ENABLE=0
VLLM_UVM_GAP_WATCH_POLICY_ACTION=prefetch
```

脚本层也要支持：

```text
--auto-gap-watch-policy-action-override observe
--auto-gap-watch-policy-action-override prefetch
```

### 8.3 初始默认值

建议默认：

```text
device_direct_enable=0
device_direct_max_bytes=1048576
device_direct_backend=cudaMallocAsync
device_direct_require_gap_match=1
device_direct_require_target_class=1
```

---

## 9. 预期最终命令形态

observe：

```bash
./run_kv_fault_ratio.sh \
  --auto-gap-watch-enable 1 \
  --auto-gap-watch-policy-action-override observe \
  --prompts 20
```

prefetch：

```bash
./run_kv_fault_ratio.sh \
  --auto-gap-watch-enable 1 \
  --auto-gap-watch-policy-action-override prefetch \
  --prompts 20
```

device-direct trace：

```bash
./run_kv_fault_ratio.sh \
  --auto-gap-watch-enable 1 \
  --auto-gap-watch-policy-action-override device_direct_trace \
  --uvm-device-direct-enable 0 \
  --prompts 20
```

device-direct：

```bash
./run_kv_fault_ratio.sh \
  --auto-gap-watch-enable 1 \
  --auto-gap-watch-policy-action-override device_direct \
  --uvm-device-direct-enable 1 \
  --uvm-device-direct-max-bytes 1048576 \
  --prompts 20
```

---

## 10. 当前建议结论

下一步不要直接全量切到 GPU-only。

推荐路径是：

1. 先把 `device_direct` 表达出来。
2. 再用 `device_direct_trace` 证明候选对象足够集中、短命、低风险。
3. 然后实现 backend tracking。
4. 最后只对小尺寸 gap2 hotspot 开启 GPU-only allocation。

这样可以最大化保留当前 prefetch 方案已经获得的收益，同时把更激进的优化风险控制在一个很窄的范围内。

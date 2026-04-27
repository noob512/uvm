# vLLM UVM Gap2 当前修改与脚本总览

## 1. 文档目的

本文档汇总当前项目中围绕 vLLM UVM replayable fault、gap2 热点发现、gap-watch targeted policy、prefetch A/B 实验以及 `device_direct` Stage B trace-only 的代码修改、脚本入口、实验产物和使用方式。

它适合作为后续继续修改项目时的入口文档。读完本文后，应该能回答：

1. 当前项目到底改了哪些核心文件。
2. 每个脚本负责什么，输入输出是什么。
3. 如何运行 single-run gap2 发现、observe/prefetch A/B、device_direct trace-only 实验。
4. 如何解释 `/tmp` 下生成的日志和 JSON。
5. 当前方案有哪些已知限制，下一阶段应该改哪里。

---

## 2. 当前主线

当前工作可以分成两条相关主线。

### 2.1 Gap2 targeted prefetch 主线

目标是先证明：

1. vLLM 运行时存在稳定的 unknown managed 热点 gap，当前重点是 `gap2`。
2. `gap2` 可以在同一个 vLLM server 进程内通过 probe 阶段自动发现。
3. allocator 可以通过 control file 热更新 watched gap。
4. 对命中 gap2 的目标 allocation 做 managed prefetch，能够减少后续 main 阶段的 replayable faults。

当前已有能力：

1. `auto-gap-watch` same-run probe。
2. probe 后自动写 gap-watch control file。
3. main 阶段复用同一个 vLLM server 进程。
4. `observe` 与 `prefetch` 策略 override。
5. gap-watch metrics summary。
6. observe/prefetch A/B 一键脚本。

### 2.2 Device-direct Stage B 主线

目标是为下一阶段 GPU-only allocation 做准备，但当前不真正改变分配后端。

Stage B 当前状态：

1. 已新增 `device_direct_trace` 和 `device_direct` 策略动作。
2. 已新增 device-direct eligibility 判定和日志字段。
3. `device_direct` 当前仍然是 trace-only，不会调用 `cudaMalloc` 或 `cudaMallocAsync`。
4. 实际分配仍然是 `cudaMallocManaged`。
5. 日志会记录哪些 allocation 如果进入 Stage C 会成为 device-direct 候选。

关键边界：

```text
Stage B = 可观测、可统计、可下发动作名，但不改变真实 placement backend。
```

---

## 3. 核心修改文件

### 3.1 `uvm_allocator.cpp`

路径：

```text
/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/uvm_allocator.cpp
```

作用：

1. vLLM/PyTorch pluggable allocator 的 C++ CUDA 实现。
2. 拦截 vLLM 的 GPU allocation，让其走 UVM managed allocation。
3. 记录 allocation/free/policy/gap-watch trace。
4. 执行 gap-watch 命中后的策略动作。

当前新增或强化的能力：

1. `PolicyAction` 支持 `observe`、`prefetch`、`advise_prefetch`、`device_direct_trace`、`device_direct`。
2. gap-watch control file 支持热更新 `start/end/target_class/policy_action`。
3. 命中 watched gap 时输出 `TRACE_GAP_WATCH_ALLOC` 和 `TRACE_GAP_WATCH_FREE`。
4. prefetch 策略会对命中目标 gap 的 managed allocation 执行 GPU prefetch。
5. Stage B 新增 device-direct trace 字段。

重要 trace 字段：

```text
TRACE_ALLOC
TRACE_FREE
TRACE_POLICY
TRACE_GAP_WATCH_ALLOC
TRACE_GAP_WATCH_FREE
TRACE_UNKNOWN_DETAIL
```

Stage B 新增字段：

```text
placement_backend
device_direct_eligible
device_direct_reason
cpu_access_risk
hot_gap_match
```

这些字段的含义：

1. `placement_backend` 表示真实分配后端，Stage B 固定为 `managed`。
2. `device_direct_eligible` 表示当前 allocation 是否满足 device-direct 候选条件。
3. `device_direct_reason` 表示 eligible 或 not eligible 的原因。
4. `cpu_access_risk` 当前还没有 CPU 访问证据，默认是 `unknown`。
5. `hot_gap_match` 表示 allocation 是否与当前 watched gap 有 overlap。

Stage B eligibility 当前条件：

1. policy action 是 `device_direct_trace` 或 `device_direct`。
2. `device >= 0`。
3. allocation 与 watched gap overlap。
4. allocation class 匹配 target class。
5. size 在 `VLLM_UVM_DEVICE_DIRECT_MIN_BYTES` 和 `VLLM_UVM_DEVICE_DIRECT_MAX_BYTES` 之间。

当前不会做的事情：

1. 不调用 `cudaMalloc`。
2. 不调用 `cudaMallocAsync`。
3. 不改变 `uvm_free` 的释放路径。
4. 不把任何对象真正改成 GPU-only allocation。

### 3.2 `run_kv_fault_ratio.sh`

路径：

```text
/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/run_kv_fault_ratio.sh
```

作用：

1. 启动单进程 vLLM server。
2. 等待 server ready 和 fresh KV range。
3. 配置 UVM driver trace 参数。
4. 启动 kernel trace 或 dmesg 采集。
5. 运行 benchmark。
6. 支持 same-run auto-gap-watch。
7. 结束后调用 summary 工具生成 JSON。

常用功能：

1. `--mode trace` 使用 tracefs 读取 UVM trace。
2. `--with-address-log` 记录 replayable fault 地址日志。
3. `--allocator-log` 记录 allocator 侧 trace。
4. `--auto-gap-watch-enable 1` 启用 probe -> discover -> control file -> main 流程。
5. `--auto-gap-watch-policy-action-override` 强制实验策略。
6. `--gap-watch-metrics-summary-json` 输出 gap-watch metrics summary。

新增 device-direct 相关参数：

```text
--uvm-device-direct-enable <0|1>
--uvm-device-direct-min-bytes <n>
--uvm-device-direct-max-bytes <n>
--auto-gap-watch-policy-action-override device_direct_trace
--auto-gap-watch-policy-action-override device_direct
```

当前解释：

1. `--uvm-device-direct-enable 0` 是推荐默认值，表示 trace-only。
2. `--uvm-device-direct-enable 1` 当前也不会真正启用 GPU-only allocation，因为 Stage C 尚未实现。
3. `device_direct_trace` 用来统计 would-device-direct 候选。
4. `device_direct` 目前也按 Stage B trace-only 处理。

### 3.3 `discover_gap_watch.py`

路径：

```text
/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/discover_gap_watch.py
```

作用：

1. 解析 replayable fault address log。
2. 找出目标 unknown gap，例如 `gap2`。
3. 关联 allocator trace，判断 gap 内主要 allocation class 和 phase。
4. 推荐 target class 和 policy action。
5. 写 gap-watch control file，让 allocator 在不重启 server 的情况下开始 watch 目标 gap。
6. 输出 discovery summary JSON。

当前支持的 policy override：

```text
observe
prefetch
advise_prefetch
device_direct_trace
device_direct
```

关键输出字段：

```text
selected_pid
target_gap
fallback_used
selected_gap
start
end
faults
unique_pages
fault_share_of_unknown
dominant_predicted_class
dominant_phase
recommended_target_class
recommended_policy_action
effective_target_class
effective_policy_action
control_file
control_written
```

如何理解：

1. `recommended_*` 是脚本根据当前日志推断的建议。
2. `effective_*` 是最终写入 control file 或本次实验实际使用的配置。
3. 如果使用 override，应优先看 `effective_policy_action`。

### 3.4 `summarize_gap_watch_metrics.py`

路径：

```text
/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/summarize_gap_watch_metrics.py
```

作用：

1. 解析 allocator trace log。
2. 统计 watched gap 的 allocation/free 命中。
3. 统计 policy 动作是否真正发生。
4. 输出 machine-readable JSON。

常见输出字段：

```text
gap_overlap_records
gap_policy_records
gap_policy_success
gap_policy_fail
gap_overlap_bytes
gap_policy_overlap_bytes
dominant_predicted_class
dominant_phase
dominant_action
dominant_target_class
median_lifetime_s
```

Stage B 新增输出字段：

```text
placement_backend_counts
device_direct_reason_counts
device_direct_trace_records
device_direct_eligible_records
hot_gap_match_records
```

如何判断 prefetch 是否生效：

1. `gap_policy_records > 0` 表示命中 gap 的对象触发了策略。
2. `gap_policy_success == gap_policy_records` 表示策略全部执行成功。
3. `dominant_action=managed_prefetch_gpu` 表示主要动作确实是 GPU prefetch。

如何判断 device-direct trace-only 是否正常：

1. `device_direct_trace_records > 0` 表示 device-direct 类动作被请求。
2. `device_direct_eligible_records > 0` 表示存在满足 Stage C 候选规则的对象。
3. `placement_backend_counts` 中应主要或全部是 `managed`。
4. 如果 Stage B 出现真实 device backend，说明行为边界被破坏，需要暂停。

### 3.5 `compare_gap_watch_ab.py`

路径：

```text
/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/compare_gap_watch_ab.py
```

作用：

1. 对比 observe 与 prefetch 两组实验。
2. 读取 post-main discovery summary 和 metrics summary。
3. 输出 gap faults、unknown faults、avg faults per unique page 的差值和比例。
4. 输出 `/tmp/vllm_gap_watch_ab_gap2.json`。

重要输出：

```text
gap_fault_delta
gap_fault_delta_pct
unknown_fault_delta
unknown_fault_delta_pct
avg_faults_per_unique_page_delta
avg_faults_per_unique_page_delta_pct
prefetch_policy_success_rate
```

当前已知显示问题：

1. `observe_action` 当前从 post-main summary 读取。
2. post-main summary 在 `--no-write-control` 模式下会重新计算推荐动作。
3. 因此 observe 组可能显示成 `prefetch`，即使 probe control file 实际写入的是 `observe`。
4. 判断策略是否执行时，应优先看 metrics JSON 中的 `gap_policy_records`、`dominant_action` 和 allocator trace。
5. 后续建议把 compare 脚本改为同时读取 probe summary，将 probe 阶段的 `effective_policy_action` 作为实验组标签。

### 3.6 `run_ab_test_gap2.sh`

路径：

```text
/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/run_ab_test_gap2.sh
```

作用：

1. 一键运行 gap2 observe/prefetch A/B。
2. Phase A 运行 `observe` baseline。
3. Phase B 运行 `prefetch` experimental。
4. Phase C 调用 `compare_gap_watch_ab.py` 生成对比 JSON。

默认产物：

```text
/tmp/uvm_kv_fault_stats_gap2_observe.log
/tmp/uvm_kv_fault_addrs_gap2_observe.log
/tmp/vllm_uvm_allocator_trace_gap2_observe.log
/tmp/vllm_gap_watch_metrics_gap2_observe.json
/tmp/vllm_auto_gap_watch_summary_gap2_observe.json
/tmp/vllm_auto_gap_watch_post_main_summary_gap2_observe.json
/tmp/uvm_kv_fault_stats_gap2_prefetch.log
/tmp/uvm_kv_fault_addrs_gap2_prefetch.log
/tmp/vllm_uvm_allocator_trace_gap2_prefetch.log
/tmp/vllm_gap_watch_metrics_gap2_prefetch.json
/tmp/vllm_auto_gap_watch_summary_gap2_prefetch.json
/tmp/vllm_auto_gap_watch_post_main_summary_gap2_prefetch.json
/tmp/vllm_gap_watch_ab_gap2.json
```

### 3.7 `analyze_uvm_fault_addresses.py`

路径：

```text
/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/analyze_uvm_fault_addresses.py
```

作用：

1. 解析 address trace log。
2. 把 fault address 与已知区域对齐。
3. 生成 fault address classification summary。
4. 用于判断 fault 是 KV、weight、unknown managed 还是其它区域。

适用场景：

1. 初步判断 replayable faults 分布。
2. 验证 KV range 是否真的贡献了主要 faults。
3. 发现 unknown gap。

### 3.8 `deep_dive_uvm_faults.py`

路径：

```text
/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/deep_dive_uvm_faults.py
```

作用：

1. 对 unknown gaps 做更深的地址和 allocator correlation。
2. 关联 `TRACE_ALLOC`、`TRACE_POLICY`、`TRACE_FREE`。
3. 分析 gap 内 allocation 的 phase、class、size、lifetime。

适用场景：

1. 解释为什么某个 unknown gap 很热。
2. 确认 gap2 是否由短命 runtime scratch/workspace allocation 主导。
3. 为 Stage C 的 `gap_hot_runtime_scratch` 分类提供证据。

### 3.9 `export_allocator_workspace_regions.py`

路径：

```text
/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/export_allocator_workspace_regions.py
```

作用：

1. 从 allocator trace 中导出 workspace/scratch allocation 地址区间。
2. 把 allocator 侧区间转换成 address-log-like regions。
3. 帮助 fault address analyzer 把 unknown 地址重新归类为 runtime workspace/scratch。

适用场景：

1. unknown gap 太大，需要进一步拆分。
2. 需要把原本 unknown 的 replayable faults 归因到 allocator runtime 对象。
3. 为新增 `gap_hot_runtime_scratch` 或 `gap_hot_runtime_workspace` 类做准备。

### 3.10 `Makefile`

路径：

```text
/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/Makefile
```

作用：

1. 编译 `uvm_allocator.so`。
2. 编译 LD_PRELOAD 版 `libcudamalloc_managed.so`。

常用命令：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test
make -B uvm
cp -f uvm_allocator.so ../vllm/uvm_allocator.abi3.so
```

注意：

1. `make uvm` 如果提示 `Nothing to be done for 'uvm'`，表示 make 认为目标文件比源文件新。
2. 强制重编译用 `make -B uvm`。
3. vLLM 实际加载的是 `../vllm/uvm_allocator.abi3.so`。
4. 只编译 `uvm_allocator.so` 不复制到 runtime path，vLLM 不一定会加载新版本。

---

## 4. 重要日志和 JSON

### 4.1 Kernel replayable fault stats log

示例：

```text
/tmp/uvm_kv_fault_stats_gap2_prefetch.log
```

来源：

1. UVM driver trace。
2. 记录 replayable fault batch stats。

当前注意事项：

1. 该日志目前包含中文文本格式。
2. `run_kv_fault_ratio.sh` 末尾的 delta parser 期望 machine-readable key=value。
3. 因此可能出现 `Failed to parse stats lines`。
4. 这不影响 address log、allocator log 和 gap-watch JSON 的分析。
5. 后续可以把 driver stats 也改成 key=value，以恢复自动 delta 解析。

### 4.2 Fault address log

示例：

```text
/tmp/uvm_kv_fault_addrs_gap2_prefetch.log
```

来源：

1. UVM driver address trace。
2. 记录 replayable fault 地址。

用途：

1. `discover_gap_watch.py` 用它发现 target gap。
2. `analyze_uvm_fault_addresses.py` 用它做地址分类。
3. `deep_dive_uvm_faults.py` 用它做 gap 深挖。

### 4.3 Allocator trace log

示例：

```text
/tmp/vllm_uvm_allocator_trace_gap2_prefetch.log
```

来源：

1. `uvm_allocator.cpp` 输出。
2. 由 `VLLM_UVM_LOG_FILE` 指定。

用途：

1. 证明策略是否真正执行。
2. 观察 allocation/free 生命周期。
3. 统计 gap overlap 和 policy records。
4. 统计 Stage B device-direct candidates。

### 4.4 Auto gap-watch probe summary

示例：

```text
/tmp/vllm_auto_gap_watch_summary_gap2_prefetch.json
```

含义：

1. probe 阶段发现 gap2。
2. 写入 control file。
3. 记录本次实验真正下发的 `effective_policy_action`。

### 4.5 Auto gap-watch post-main summary

示例：

```text
/tmp/vllm_auto_gap_watch_post_main_summary_gap2_prefetch.json
```

含义：

1. main 阶段结束后重新统计目标 gap。
2. 用于观察 main 阶段的 gap faults、unique pages、fault share。
3. 用于和 probe 阶段对比 same gap 是否稳定。

### 4.6 Gap-watch metrics summary

示例：

```text
/tmp/vllm_gap_watch_metrics_gap2_prefetch.json
```

含义：

1. allocator trace 的策略执行摘要。
2. 判断 observe/prefetch/device_direct_trace 的主要依据。

### 4.7 A/B comparison summary

示例：

```text
/tmp/vllm_gap_watch_ab_gap2.json
```

含义：

1. observe 与 prefetch 两组的差值。
2. 包括 gap faults、unknown faults、avg faults per unique page 的变化。

---

## 5. 标准实验流程

### 5.1 单次 auto-gap-watch prefetch 实验

用途：

1. 在同一个 server 进程内 probe gap2。
2. 对 gap2 命中对象启用 prefetch。
3. 观察 main 阶段 fault 和 policy metrics。

命令：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm

./run_kv_fault_ratio.sh \
  --mode trace \
  --with-address-log \
  --trace-log /tmp/uvm_kv_fault_stats_gap2_policy.log \
  --address-trace-log /tmp/uvm_kv_fault_addrs_gap2_policy.log \
  --allocator-log /tmp/vllm_uvm_allocator_trace_gap2_policy.log \
  --uvm-trace-min-bytes 1048576 \
  --uvm-unknown-detail-enable 1 \
  --uvm-unknown-detail-min-bytes 4096 \
  --uvm-gap-watch-name same_run_gap2 \
  --uvm-gap-watch-all-classes 1 \
  --uvm-gap-watch-min-bytes 4096 \
  --auto-gap-watch-enable 1 \
  --auto-gap-watch-probe-prompts 1 \
  --auto-gap-watch-target-gap 2 \
  --auto-gap-watch-policy-action-override prefetch \
  --prompts 20 \
  --gap-watch-metrics-summary-json /tmp/vllm_gap_watch_metrics_gap2_policy.json
```

成功判断：

1. `selected_gap=2`。
2. `same_gap=True`。
3. `gap_policy_records > 0`。
4. `gap_policy_success == gap_policy_records`。
5. `dominant_action=managed_prefetch_gpu`。

### 5.2 Observe/prefetch A/B 实验

用途：

1. 证明 prefetch 相比 observe 是否减少 gap2 faults。
2. 生成统一对比 JSON。

命令：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./run_ab_test_gap2.sh
```

最近一次已观察到的典型结果：

```text
observe_gap_faults=16869336
prefetch_gap_faults=4845380
gap_fault_delta=-12023956
gap_fault_delta_pct=-0.7127699632042424
observe_unknown_faults=17347156
prefetch_unknown_faults=5438654
unknown_fault_delta=-11908502
unknown_fault_delta_pct=-0.68648151893025
observe_policy_records=0
prefetch_policy_records=524703
prefetch_policy_success=524703
prefetch_policy_fail=0
```

解释：

1. prefetch 组 gap2 faults 下降约 71.3%。
2. prefetch 组 unknown faults 下降约 68.6%。
3. observe 组没有执行 policy。
4. prefetch 组执行了 524703 次 gap policy，且全部成功。
5. 这说明 gap2 targeted prefetch 对当前 workload 有明确正向作用。

### 5.3 Device-direct Stage B trace-only 实验

用途：

1. 不改变真实分配行为。
2. 统计如果进入 Stage C，哪些 gap2 allocation 会成为 device-direct 候选。
3. 验证 candidate 是否主要集中在 gap2、`phase=enabled`、`unknown_managed`、短生命周期对象。

命令：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm

./run_kv_fault_ratio.sh \
  --mode trace \
  --with-address-log \
  --trace-log /tmp/uvm_kv_fault_stats_gap2_device_trace.log \
  --address-trace-log /tmp/uvm_kv_fault_addrs_gap2_device_trace.log \
  --allocator-log /tmp/vllm_uvm_allocator_trace_gap2_device_trace.log \
  --uvm-trace-min-bytes 1048576 \
  --uvm-unknown-detail-enable 1 \
  --uvm-unknown-detail-min-bytes 4096 \
  --uvm-gap-watch-name same_run_gap2_device_trace \
  --uvm-gap-watch-all-classes 1 \
  --uvm-gap-watch-min-bytes 4096 \
  --uvm-device-direct-enable 0 \
  --uvm-device-direct-min-bytes 4096 \
  --uvm-device-direct-max-bytes 1048576 \
  --auto-gap-watch-enable 1 \
  --auto-gap-watch-probe-prompts 1 \
  --auto-gap-watch-target-gap 2 \
  --auto-gap-watch-policy-action-override device_direct_trace \
  --prompts 20 \
  --gap-watch-metrics-summary-json /tmp/vllm_gap_watch_metrics_gap2_device_trace.json \
  --auto-gap-watch-summary-json /tmp/vllm_auto_gap_watch_summary_gap2_device_trace.json \
  --auto-gap-watch-post-main-summary-json /tmp/vllm_auto_gap_watch_post_main_summary_gap2_device_trace.json
```

成功判断：

1. `effective_policy_action=device_direct_trace`。
2. `device_direct_trace_records > 0`。
3. `device_direct_eligible_records > 0`。
4. `placement_backend_counts` 中仍然是 `managed`。
5. `device_direct_reason_counts` 能解释哪些对象 eligible，哪些被过滤。

### 5.4 手动生成 metrics summary

命令：

```bash
python3 /home/ubuntu/nvidia-uvm-gpu/workloads/vllm/summarize_gap_watch_metrics.py \
  --allocator-log /tmp/vllm_uvm_allocator_trace_gap2_device_trace.log \
  --summary-json /tmp/vllm_gap_watch_metrics_gap2_device_trace.json
```

用途：

1. 不重跑 benchmark，只重新解析 allocator log。
2. 修改 summary 脚本后可以快速验证输出字段。

### 5.5 深挖 unknown gap

命令模板：

```bash
python3 /home/ubuntu/nvidia-uvm-gpu/workloads/vllm/deep_dive_uvm_faults.py \
  --address-log /tmp/uvm_kv_fault_addrs_gap2_prefetch.log \
  --allocator-log /tmp/vllm_uvm_allocator_trace_gap2_prefetch.log \
  --target-gap 2 \
  --summary-json /tmp/vllm_gap2_deep_dive.json
```

用途：

1. 分析 gap2 内对象的 class、phase、size、lifetime。
2. 为 Stage C 的严格筛选条件提供依据。

### 5.6 导出 allocator workspace regions

命令模板：

```bash
python3 /home/ubuntu/nvidia-uvm-gpu/workloads/vllm/export_allocator_workspace_regions.py \
  --allocator-log /tmp/vllm_uvm_allocator_trace_gap2_prefetch.log \
  --output /tmp/vllm_allocator_workspace_regions.log \
  --summary-json /tmp/vllm_allocator_workspace_regions_summary.json
```

用途：

1. 将 allocator trace 中的 workspace/scratch 地址段导出。
2. 后续可辅助 address analyzer 把 unknown fault 重新归类。

---

## 6. 如何解释关键指标

### 6.1 `faults`

`faults` 表示某个 gap 内观察到的 replayable fault 实例数。

注意：

1. 它不是唯一页数。
2. 同一页可能重复 fault 多次。
3. 因此需要同时看 `unique_pages` 和 `avg_faults_per_unique_page`。

### 6.2 `unique_pages`

`unique_pages` 表示 gap 内涉及多少不同 page。

解释：

1. 如果 `faults` 高但 `unique_pages` 不高，说明同一批页面反复迁移或反复 fault。
2. 这类情况通常更适合 prefetch 或 placement 优化。

### 6.3 `fault_share_of_unknown`

表示该 gap faults 占 unknown faults 的比例。

解释：

1. 接近 1 表示这个 gap 是 unknown faults 的主导来源。
2. gap2 实验中 probe 阶段经常接近 0.99，说明目标非常集中。

### 6.4 `gap_policy_records`

表示 watched gap 命中的 allocation 中，有多少条触发了策略动作。

解释：

1. observe 组应为 0。
2. prefetch 组应大于 0。
3. device_direct_trace 组应大于 0。

### 6.5 `gap_policy_success`

表示策略动作成功次数。

解释：

1. 对 prefetch 而言，成功意味着 CUDA prefetch API 返回成功。
2. 对 trace-only 动作而言，成功更多表示策略路径被记录并完成，不代表真实 backend 改变。

### 6.6 `dominant_action`

表示 allocator trace 中 gap-watch 命中的主导动作。

常见值：

```text
managed_default
managed_prefetch_gpu
device_direct_trace
device_direct
```

解释：

1. observe 组应主要是 `managed_default`。
2. prefetch 组应主要是 `managed_prefetch_gpu`。
3. Stage B trace 组应主要是 `device_direct_trace`，但 placement backend 仍是 `managed`。

### 6.7 `median_lifetime_s`

表示 gap-watch allocation 从 alloc 到 free 的中位生命周期。

解释：

1. gap2 当前常见中位生命周期约 1ms 级别。
2. 这支持“短命 runtime scratch/workspace”的判断。
3. 这也提示 Stage C 如果使用 `cudaMalloc/cudaFree` 可能有同步和 allocator 开销风险。
4. Stage C 更推荐先评估 `cudaMallocAsync/cudaFreeAsync`。

### 6.8 `device_direct_eligible_records`

表示 Stage B trace 中有多少 allocation 满足当前 device-direct 候选规则。

解释：

1. 这是 Stage C 的候选规模估计。
2. 数量过大可能意味着条件太宽。
3. 数量为 0 可能意味着 gap/class/size 条件太严，或 policy 没有下发成功。

### 6.9 `placement_backend_counts`

表示实际分配后端统计。

Stage B 正确结果：

```text
managed > 0
device/direct backend = 0
```

如果 Stage B 出现真实 device backend，应立即检查 `uvm_allocator.cpp`，因为这不符合 trace-only 目标。

---

## 7. 已知问题和注意事项

### 7.1 Driver stats delta parser 目前可能失败

现象：

```text
Failed to parse stats lines from /tmp/uvm_kv_fault_stats_*.log
Hint: keep machine-readable key=value fields in driver stats logs
```

原因：

1. UVM stats log 当前是中文自然语言文本。
2. shell 里的 delta parser 期望 `batch_faults=... total_faults=...` 这类 key=value。

影响：

1. 不影响 benchmark。
2. 不影响 address log。
3. 不影响 allocator trace。
4. 不影响 gap-watch summary JSON。

后续建议：

1. 修改 driver trace 输出，增加 machine-readable key=value 字段。
2. 或修改 shell parser 支持当前中文格式。

### 7.2 A/B compare 的 action 显示可能误导

现象：

```text
observe_action=prefetch
```

但 observe 组 metrics 显示：

```text
gap_policy_records=0
dominant_action=managed_default
```

原因：

1. compare 脚本读取 post-main summary 的 `effective_policy_action`。
2. post-main discovery 不写 control file，会重新给出推荐 action。
3. 推荐 action 可能是 `prefetch`，不等于 observe 组实际执行 action。

当前判断方式：

1. 以 probe summary 的 `effective_policy_action` 判断实验标签。
2. 以 metrics summary 的 `gap_policy_records` 和 `dominant_action` 判断实际执行行为。

后续建议：

1. `compare_gap_watch_ab.py` 增加 `--observe-probe-summary` 和 `--prefetch-probe-summary`。
2. 输出中区分 `configured_action` 和 `post_main_recommended_action`。

### 7.3 gap 地址不能跨进程复用

原因：

1. vLLM server 重启后虚拟地址会变化。
2. gap2 是本轮地址空间内的热点 gap，不是固定物理对象。

正确做法：

1. 每次实验都用 same-run probe。
2. probe 后在同一个 server 进程中写 control file。
3. main 阶段不要重启 server。

### 7.4 `device_direct` 当前不是 GPU-only allocation

当前阶段：

```text
device_direct = Stage B trace-only action
```

不能据此声称：

1. 已经使用 GPU 显存直接分配。
2. 已经绕过 UVM。
3. fault 下降来自 device allocation。

可以声称：

1. 已经具备 device_direct 动作表达能力。
2. 已经可以统计 would-device-direct 候选。
3. 已经可以为 Stage C 制定安全条件。

### 7.5 `.so` 编译后必须复制到 runtime path

编译路径：

```text
/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/uvm_allocator.so
```

vLLM 加载路径：

```text
/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/uvm_allocator.abi3.so
```

推荐命令：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test
make -B uvm
cp -f uvm_allocator.so ../vllm/uvm_allocator.abi3.so
```

---

## 8. 当前文档索引

### 8.1 Gap2 targeted policy

路径：

```text
/home/ubuntu/nvidia-uvm-gpu/docs/vllm_uvm_gap2_targeted_policy.md
```

内容：

1. gap2 targeted prefetch 的背景。
2. observe/prefetch A/B 命令。
3. 如何证明 prefetch 减少 faults。

### 8.2 Device-direct roadmap

路径：

```text
/home/ubuntu/nvidia-uvm-gpu/docs/vllm_uvm_gap2_device_direct_roadmap.md
```

内容：

1. 阶段 A/B/C/D 路线。
2. 为什么不能按固定 gap 地址做 GPU-only 分配。
3. Stage C 的安全条件。
4. A/B/C 三组比较设计。

### 8.3 Device-direct Stage B implementation

路径：

```text
/home/ubuntu/nvidia-uvm-gpu/docs/vllm_uvm_device_direct_stage_b_implementation.md
```

内容：

1. Stage B 已实现的代码修改。
2. 新增环境变量。
3. 新增 trace 字段。
4. device-direct eligibility 规则。
5. trace-only 实验命令。

### 8.4 Same-run auto gap watch

路径：

```text
/home/ubuntu/nvidia-uvm-gpu/docs/vllm_uvm_same_run_auto_gap_watch.md
```

内容：

1. same-run probe 的设计动机。
2. control file 热更新流程。
3. 为什么必须同一 server 进程内发现和执行。

### 8.5 Unknown gap resolution

路径：

```text
/home/ubuntu/nvidia-uvm-gpu/docs/vllm_uvm_unknown_gap_resolution.md
```

内容：

1. unknown gap 的分析方法。
2. allocator trace 与 fault address 的关联思路。
3. 后续分类细化方向。

### 8.6 Dynamic allocator status

路径：

```text
/home/ubuntu/nvidia-uvm-gpu/docs/vllm_uvm_dynamic_allocator_status.md
```

内容：

1. 当前动态 allocator 工作状态。
2. 已实现和未实现能力。
3. 运行时配置说明。

---

## 9. 下一阶段建议

### 9.1 先修正 compare 脚本标签问题

建议改动：

1. `compare_gap_watch_ab.py` 增加 probe summary 输入。
2. 输出 `configured_action`。
3. 输出 `post_main_recommended_action`。
4. 保留 metrics 中的 `dominant_action` 作为事实依据。

这样可以避免 observe 组显示成 prefetch 的误解。

### 9.2 增加 A/B/C 三组实验脚本

建议新增：

```text
run_abc_test_gap2.sh
compare_gap_watch_abc.py
```

三组：

```text
observe
prefetch
device_direct_trace
```

Stage C 真正启用后，第三组再切换为：

```text
device_direct
```

### 9.3 细化 gap_hot_runtime_scratch 分类

当前 gap2 主导 class 仍然是：

```text
unknown_managed
```

下一步建议：

1. 用 allocator workspace export 辅助分类。
2. 在 `uvm_allocator.cpp` 中增加更细的 predicted class。
3. 将满足条件的 unknown allocation 标成 `gap_hot_runtime_scratch`。

### 9.4 Stage C 前必须补 allocation backend registry

启用真实 device allocation 前必须做：

1. 给每个 pointer 记录 backend。
2. `uvm_free` 根据 backend 选择 `cudaFree`、`cudaFreeAsync` 或 managed free 路径。
3. 记录 allocation stream。
4. 记录 device id。
5. 增加 fallback 到 managed 的错误处理。

建议 backend 枚举：

```cpp
enum class AllocationBackend {
    Managed,
    DeviceDirect,
    DeviceDirectAsync,
};
```

### 9.5 Stage C 初始安全条件

建议只对满足以下条件的对象启用真实 device allocation：

```text
phase=enabled
predicted_class=gap_hot_runtime_scratch
hot_gap_match=1
size >= 4096
size <= 1048576 或 16777216
median_lifetime_s 极短
无 CPU access 证据
device memory pressure 可控
```

如果任何条件不满足，应 fallback 到 managed。

---

## 10. 快速排错表

### 10.1 server ready 超时

现象：

```text
ERROR: Timeout waiting for server readiness and KV range (>600s)
```

排查：

1. 查看 server log tail。
2. 确认 vLLM 是否加载了新的 allocator `.so`。
3. 确认 GPU 上是否有残留进程。
4. 确认模型加载是否卡住或 OOM。

### 10.2 `make uvm` 不重新编译

现象：

```text
make: Nothing to be done for 'uvm'.
```

处理：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test
make -B uvm
cp -f uvm_allocator.so ../vllm/uvm_allocator.abi3.so
```

### 10.3 prefetch 组没有 policy records

现象：

```text
gap_policy_records=0
```

排查：

1. `auto-gap-watch` 是否启用。
2. control file 是否写入。
3. `effective_policy_action` 是否是 `prefetch`。
4. target class 是否和 allocator trace 中 class 匹配。
5. watched gap start/end 是否与 main 阶段 gap 一致。

### 10.4 device_direct_trace 没有 eligible records

现象：

```text
device_direct_trace_records>0
device_direct_eligible_records=0
```

排查：

1. size 是否低于 `--uvm-device-direct-min-bytes`。
2. size 是否高于 `--uvm-device-direct-max-bytes`。
3. `hot_gap_match` 是否为 0。
4. target class 是否不匹配。
5. `device_direct_reason_counts` 中的主导 reason 是什么。

---

## 11. 当前结论

当前项目已经形成了一套相对完整的闭环：

1. 用 UVM address trace 找到 gap2。
2. 用 allocator trace 解释 gap2。
3. 用 same-run control file 把发现结果反馈给 allocator。
4. 用 observe/prefetch A/B 证明 prefetch 能减少 gap2 replayable faults。
5. 用 Stage B device-direct trace-only 为后续 GPU-only allocation 做候选筛选。

当前最可靠的事实依据是：

1. prefetch 组 `gap_policy_records` 大于 0。
2. prefetch 组 `gap_policy_success` 等于 `gap_policy_records`。
3. prefetch 组 gap2 faults 和 unknown faults 明显低于 observe 组。
4. Stage B 的 `placement_backend` 仍然是 `managed`，说明还没有越界启用 GPU-only allocation。

下一步最稳妥的推进顺序是：

1. 修正 A/B compare 标签显示。
2. 增加 A/B/C 三组脚本和 comparison。
3. 用 `device_direct_trace` 收集足够多轮候选统计。
4. 细化 `gap_hot_runtime_scratch` class。
5. 实现 Stage C 的 backend registry 和严格 fallback。
6. 小规模启用真实 `device_direct`，并与 observe/prefetch/device_direct_trace 对照。

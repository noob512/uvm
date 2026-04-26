# vLLM UVM Gap2 Targeted Policy

## 1. 文档目标

这次修改的目标不是只“盯住 gap2”，而是把整条链路补完整：

1. 自动检测当前运行中的 `gap2` 地址
2. 自动判断这块地址更像哪一类对象
3. 对命中这块地址、且类型匹配的 allocation 施加额外策略
4. 输出足够详细的指标，证明策略确实命中并执行了

本轮又继续补了两项能力：

1. `auto-gap-watch` 支持显式覆盖策略动作
2. 新增 observe/prefetch 的 A/B 对照汇总脚本

本次修改后，系统已经能做到：

1. `discover_gap_watch.py` 不只发现地址，还会给出：
   - `dominant_predicted_class`
   - `dominant_phase`
   - `recommended_target_class`
   - `recommended_policy_action`
2. `uvm_allocator.cpp` 不只记录 `TRACE_GAP_WATCH_ALLOC/FREE`
   - 还会对命中 gap 的 allocation 依据 `target_class + policy_action` 做额外动作
3. `run_kv_fault_ratio.sh` 会自动把“地址 + 类型 + 动作”写进 control file
4. 新增 `summarize_gap_watch_metrics.py`
   - 专门输出“是否真的命中、是否真的施策、是否成功”的证明摘要

---

## 2. 这次修改解决了什么问题

在此前实现中，项目已经具备：

1. 自动发现同一进程中的 `gap2`
2. allocator 侧 watch 某个地址段
3. 输出 `TRACE_GAP_WATCH_ALLOC/FREE`

但仍然缺两层关键能力：

### 2.1 缺“类型”

以前 auto gap-watch 只知道：

1. `gap2` 的 `start/end`

但不知道：

1. 这块 gap 更像 `runtime_scratch`
2. 还是更像 `runtime_workspace`
3. 还是更像 `warmup_workspace`

所以无法有针对性地下策略。

### 2.2 缺“动作证明”

以前即使打开了 gap-watch，也更像：

1. 只是记录谁命中了这个地址段

但无法直接回答：

1. allocator 有没有因为 gap2 而改变行为
2. 改变了多少次
3. 成功了多少次
4. 覆盖了多少 overlap bytes

这次修改把这两层补上了。

---

## 3. 修改文件

本次涉及四个核心文件和一个兼容性修复文件：

1. [uvm_allocator.cpp](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/uvm_allocator.cpp)
2. [discover_gap_watch.py](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/discover_gap_watch.py)
3. [run_kv_fault_ratio.sh](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/run_kv_fault_ratio.sh)
4. [summarize_gap_watch_metrics.py](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/summarize_gap_watch_metrics.py)
5. [deep_dive_uvm_faults.py](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/deep_dive_uvm_faults.py)

---

## 4. `discover_gap_watch.py` 的增强

### 4.1 新增 `--allocator-log`

现在 `discover_gap_watch.py` 会在发现 `gap2` 后，继续读取 allocator trace，
对这个 gap 做一次“轻量级类型判定”。

新增输入：

1. `--allocator-log /tmp/vllm_uvm_allocator_trace_xxx.log`

### 4.2 新增类型归因输出

脚本会分析 overlap 到 `gap2` 的 `TRACE_POLICY` / `TRACE_FREE`，得到：

1. `dominant_predicted_class`
2. `dominant_phase`
3. `phase_counts`
4. `predicted_class_counts`
5. `predicted_class_overlap_bytes`
6. `median_lifetime_s`

并生成：

1. `recommended_target_class`
2. `recommended_policy_action`

### 4.3 control file 不再只有地址

以前 control file 只有：

1. `enabled`
2. `name`
3. `start`
4. `end`
5. `all_classes`
6. `min_bytes`

现在会额外写入：

1. `target_class`
2. `policy_action`

### 4.4 新增 override 能力

为了做严格的 A/B 实验，`discover_gap_watch.py` 现在支持两类覆盖：

1. `--policy-action-override observe|prefetch|advise_prefetch`
2. `--target-class-override <name>`

这意味着：

1. probe 仍然负责发现“本次运行中的真实 gap2”
2. allocator 归因仍然负责给出推荐类别
3. 但最终写入 control file 的动作现在可以被实验者强制指定

这样就能保证：

1. `observe` 组和 `prefetch` 组使用同样的 auto-discovery 流程
2. 唯一改变的变量就是最终策略动作

这正是证明“prefetch 是否真的减少 fault”的必要条件。

---

## 5. `run_kv_fault_ratio.sh` 的增强

### 5.1 新增 auto-gap-watch override 参数

新增两个 CLI 参数：

1. `--auto-gap-watch-policy-action-override <mode>`
2. `--auto-gap-watch-target-class-override <name>`

前者用于控制 A/B 的核心变量：

1. `observe`
2. `prefetch`
3. `advise_prefetch`

后者用于在必要时固定目标类别，避免不同轮次因为推荐类别不同而导致对照失真。

### 5.2 作用方式

脚本在 probe 完成后，会调用 `discover_gap_watch.py` 写 control file。

现在这一步会把 override 一并透传过去：

1. 如果未设置 override，则继续使用自动推荐值
2. 如果设置了 override，则写入强制值

因此：

1. 默认工作流不受影响
2. 但实验模式下可以稳定形成 observe/prefetch 对照组

---

## 6. 新增 `compare_gap_watch_ab.py`

新增文件：

1. [compare_gap_watch_ab.py](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/compare_gap_watch_ab.py)

它的作用是把 observe 和 prefetch 两轮实验结果自动对齐比较。

### 6.1 输入

需要四个 JSON：

1. observe 组的 post-main summary
2. observe 组的 gap-watch metrics
3. prefetch 组的 post-main summary
4. prefetch 组的 gap-watch metrics

### 6.2 输出

它会直接给出：

1. `same_gap`
2. `gap_fault_delta`
3. `gap_fault_delta_pct`
4. `unknown_fault_delta`
5. `unknown_fault_delta_pct`
6. `avg_faults_per_unique_page_delta`
7. `avg_faults_per_unique_page_delta_pct`
8. `prefetch_policy_success`
9. `prefetch_policy_fail`

### 6.3 为什么它重要

这能把“策略真的执行了”和“fault 真的下降了”分开表达：

1. `gap_policy_records / gap_policy_success`
   - 证明策略确实执行
2. `gap_fault_delta / unknown_fault_delta`
   - 证明 fault 是否真的下降

---

## 7. 推荐的 A/B 实验命令

### 7.1 Observe 对照组

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm

./run_kv_fault_ratio.sh \
  --mode trace \
  --with-address-log \
  --trace-log /tmp/uvm_kv_fault_stats_gap2_observe.log \
  --address-trace-log /tmp/uvm_kv_fault_addrs_gap2_observe.log \
  --allocator-log /tmp/vllm_uvm_allocator_trace_gap2_observe.log \
  --uvm-trace-min-bytes 1048576 \
  --uvm-unknown-detail-enable 1 \
  --uvm-unknown-detail-min-bytes 4096 \
  --uvm-gap-watch-name same_run_gap2_observe \
  --uvm-gap-watch-all-classes 1 \
  --uvm-gap-watch-min-bytes 4096 \
  --auto-gap-watch-enable 1 \
  --auto-gap-watch-probe-prompts 1 \
  --auto-gap-watch-target-gap 2 \
  --auto-gap-watch-policy-action-override observe \
  --prompts 20 \
  --gap-watch-metrics-summary-json /tmp/vllm_gap_watch_metrics_gap2_observe.json \
  --auto-gap-watch-summary-json /tmp/vllm_auto_gap_watch_summary_gap2_observe.json \
  --auto-gap-watch-post-main-summary-json /tmp/vllm_auto_gap_watch_post_main_summary_gap2_observe.json
```

### 7.2 Prefetch 实验组

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm

./run_kv_fault_ratio.sh \
  --mode trace \
  --with-address-log \
  --trace-log /tmp/uvm_kv_fault_stats_gap2_prefetch.log \
  --address-trace-log /tmp/uvm_kv_fault_addrs_gap2_prefetch.log \
  --allocator-log /tmp/vllm_uvm_allocator_trace_gap2_prefetch.log \
  --uvm-trace-min-bytes 1048576 \
  --uvm-unknown-detail-enable 1 \
  --uvm-unknown-detail-min-bytes 4096 \
  --uvm-gap-watch-name same_run_gap2_prefetch \
  --uvm-gap-watch-all-classes 1 \
  --uvm-gap-watch-min-bytes 4096 \
  --auto-gap-watch-enable 1 \
  --auto-gap-watch-probe-prompts 1 \
  --auto-gap-watch-target-gap 2 \
  --auto-gap-watch-policy-action-override prefetch \
  --prompts 20 \
  --gap-watch-metrics-summary-json /tmp/vllm_gap_watch_metrics_gap2_prefetch.json \
  --auto-gap-watch-summary-json /tmp/vllm_auto_gap_watch_summary_gap2_prefetch.json \
  --auto-gap-watch-post-main-summary-json /tmp/vllm_auto_gap_watch_post_main_summary_gap2_prefetch.json
```

### 7.3 A/B 汇总

```bash
python3 /home/ubuntu/nvidia-uvm-gpu/workloads/vllm/compare_gap_watch_ab.py \
  --observe-post-main /tmp/vllm_auto_gap_watch_post_main_summary_gap2_observe.json \
  --observe-metrics /tmp/vllm_gap_watch_metrics_gap2_observe.json \
  --prefetch-post-main /tmp/vllm_auto_gap_watch_post_main_summary_gap2_prefetch.json \
  --prefetch-metrics /tmp/vllm_gap_watch_metrics_gap2_prefetch.json \
  --output-json /tmp/vllm_gap_watch_ab_gap2.json
```

---

## 8. 如何判定 prefetch 是否真的有效

如果满足下面几条，就可以高置信地说 prefetch 有效：

1. `same_gap=True`
2. `prefetch` 组 `gap_policy_records > 0`
3. `prefetch` 组 `gap_policy_success > 0`
4. `prefetch` 组的 `gap_faults` 低于 `observe` 组
5. `prefetch` 组的 `total_unknown_faults` 也同步下降
6. `avg_faults_per_unique_page` 下降

这样可以避免误判成：

1. gap 换了
2. 只是策略打日志但没真正执行
3. 只是 unique page 变多导致总 fault 看起来变少

示例：

```text
enabled=1
name=same_run_gap2
start=0x781181180000
end=0x781181ffffff
all_classes=1
min_bytes=4096
target_class=runtime_scratch
policy_action=advise_prefetch
```

这意味着 allocator 不再只是“看地址”，而是“看地址 + 看要命中的语义类别 + 看要执行的动作”。

---

## 5. `uvm_allocator.cpp` 的增强

这是这次修改的核心。

### 5.1 新增 gap-watch 专用配置

新增两个配置项：

1. `gap_watch_target_class`
2. `gap_watch_policy_action`

同时支持：

1. 环境变量读取
2. control file 热更新

对应环境变量：

1. `VLLM_UVM_GAP_WATCH_TARGET_CLASS`
2. `VLLM_UVM_GAP_WATCH_POLICY_ACTION`

### 5.2 新增动作类型

原来只有：

1. `managed_default`
2. `managed_prefetch_gpu`

现在新增：

1. `managed_advise_prefetch_gpu`

含义：

1. 先 `cudaMemAdviseSetPreferredLocation(device)`
2. 再 `cudaMemPrefetchAsync(..., device, stream)`

这比单纯 prefetch 更激进一点，但仍然没有改 allocator API 契约，仍然是 `cudaMallocManaged` 主路径。

### 5.3 新增 gap-watch 专用策略选择器

新增：

1. `choose_gap_watch_policy_action(...)`

它只在满足下列条件时生效：

1. allocation 与当前 watch range 有 overlap
2. 当前 allocation 的 `predicted_class` 命中 `target_class`
3. allocation size 大于等于 `gap_watch_min_bytes`
4. `policy_action` 不是 `observe`

如果条件满足，就让 gap-watch policy 覆盖普通 policy。

也就是说，当前策略优先级变成：

1. 先做普通 `phase + size` 分类
2. 再看是否命中 gap-watch 的“地址 + 类型”规则
3. 如果命中，则 gap-watch policy 覆盖 base policy

### 5.4 `TRACE_POLICY` 新增字段

为了证明策略来源，这次扩展了 `TRACE_POLICY`：

新增字段：

1. `policy_source`
2. `gap_watch_class_match`
3. `gap_overlap_bytes`

含义：

1. `policy_source=base_policy`
   - 表示这次动作来自原有普通策略
2. `policy_source=gap_watch_policy`
   - 表示这次动作是因为命中了 gap-watch 规则而触发
3. `gap_watch_class_match=1`
   - 表示 allocation 的 `predicted_class` 与 `target_class` 匹配
4. `gap_overlap_bytes`
   - 这次 allocation 在 watched gap 上实际覆盖了多少字节

### 5.5 `TRACE_GAP_WATCH_ALLOC/FREE` 新增字段

新增字段包括：

1. `policy_source`
2. `gap_watch_target_class`
3. `gap_watch_class_match`
4. `alloc_policy_source`
5. `alloc_policy_success`
6. `alloc_policy_error`
7. `gap_watch_policy_action`

这让 gap-watch 事件本身就能回答：

1. 命中的对象是什么类
2. 是否满足目标类型
3. 动作是来自普通策略还是 gap-watch 策略
4. 动作是否成功
5. 对应 lifetime 是多少

### 5.6 新增 allocator 内部统计计数器

新增原子计数器：

1. `gap_watch_overlap_allocs`
2. `gap_watch_overlap_bytes_total`
3. `gap_watch_target_class_match_allocs`
4. `gap_watch_policy_applied_allocs`
5. `gap_watch_policy_applied_overlap_bytes`
6. `gap_watch_policy_success_allocs`
7. `gap_watch_policy_failed_allocs`

这些计数器会在 session summary 里打印。

因此，就算不跑离线脚本，你也能在 allocator log 尾部直接看到：

1. 一共有多少 alloc overlap 到 watched gap
2. 其中多少 alloc 命中了 target class
3. 其中多少 alloc 真正触发了策略
4. 成功了多少，失败了多少

---

## 6. `run_kv_fault_ratio.sh` 的增强

### 6.1 新增参数

新增：

1. `--uvm-gap-watch-target-class <name>`
2. `--uvm-gap-watch-policy-action <mode>`
3. `--gap-watch-metrics-summary-json <path>`

其中：

1. `target-class` 决定只对哪一类 overlap allocation 施策
2. `policy-action` 决定施加什么动作
3. `gap-watch-metrics-summary-json` 决定是否在运行后自动汇总证明指标

### 6.2 auto gap-watch 会自动写入类型与动作

以前 auto 模式只做：

1. `probe -> discover address -> write start/end -> main`

现在变成：

1. `probe -> discover address`
2. `discover current dominant class / dominant phase`
3. `write start/end + target_class + policy_action`
4. allocator 热更新
5. `main`

也就是说，main 阶段拿到的已经不再只是一个“地址观察器”，而是一个“带动作的 targeted policy”。

### 6.3 运行后自动产出 gap-watch 指标摘要

如果传入：

```bash
--gap-watch-metrics-summary-json /tmp/vllm_gap_watch_metrics.json
```

脚本结束时会自动调用：

1. `summarize_gap_watch_metrics.py`

输出：

1. 终端摘要
2. JSON 摘要文件

---

## 7. 新增 `summarize_gap_watch_metrics.py`

这是这次新增的“证明脚本”。

### 7.1 输入

输入：

1. `--allocator-log /tmp/vllm_uvm_allocator_trace_xxx.log`

### 7.2 输出

输出内容包括：

1. `gap_overlap_records`
2. `gap_policy_records`
3. `gap_policy_success`
4. `gap_policy_fail`
5. `gap_overlap_bytes`
6. `gap_policy_overlap_bytes`
7. `dominant_predicted_class`
8. `dominant_phase`
9. `dominant_action`
10. `dominant_target_class`
11. `median_lifetime_s`
12. `session_summary`

### 7.3 它证明什么

这个脚本主要证明四件事：

1. watched gap 有没有被 runtime allocation 真正命中
2. 命中的对象主导类型是什么
3. gap-watch policy 有没有真正触发
4. 触发后的动作成功了多少次

所以它是“是否真正实现 gap2 targeted policy”的直接证据脚本。

---

## 8. `deep_dive_uvm_faults.py` 的兼容性修复

因为这次扩展了 `TRACE_POLICY` 的格式，原有正则会读不懂新字段。

所以额外做了一个兼容性修复：

1. `TRACE_POLICY_RE` 现在兼容新旧格式

这样：

1. 旧日志还能分析
2. 新日志也不会把 `deep_dive_uvm_faults.py` 弄坏

---

## 9. 推荐使用方式

### 9.1 同进程自动发现 + 自动施策 + 自动汇总

推荐命令：

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
  --prompts 20 \
  --gap-watch-metrics-summary-json /tmp/vllm_gap_watch_metrics_gap2_policy.json
```

这条命令会完成：

1. 启动 server
2. probe
3. 自动发现本轮 gap2
4. 自动判断 gap2 更像哪一类
5. 自动把 `target_class + policy_action` 写进 control file
6. allocator 在同一进程内热更新
7. main 阶段对命中 gap2 的对象施策
8. 输出 metrics summary

### 9.2 固定地址手工指定

如果你已经知道某轮 gap2 地址和类型，也可以手工指定：

```bash
./run_kv_fault_ratio.sh \
  --mode trace \
  --with-address-log \
  --trace-log /tmp/uvm_kv_fault_stats_manual_gap2.log \
  --address-trace-log /tmp/uvm_kv_fault_addrs_manual_gap2.log \
  --allocator-log /tmp/vllm_uvm_allocator_trace_manual_gap2.log \
  --uvm-gap-watch-enable 1 \
  --uvm-gap-watch-name manual_gap2 \
  --uvm-gap-watch-start 0x781181180000 \
  --uvm-gap-watch-end 0x781181ffffff \
  --uvm-gap-watch-all-classes 1 \
  --uvm-gap-watch-min-bytes 4096 \
  --uvm-gap-watch-target-class runtime_scratch \
  --uvm-gap-watch-policy-action advise_prefetch \
  --gap-watch-metrics-summary-json /tmp/vllm_gap_watch_metrics_manual_gap2.json
```

---

## 10. 如何判断是否“真的实现了”

本次实现是否成功，不看一句话结论，要看以下几组证据。

### 10.1 证据一：发现阶段输出

看 `discover_gap_watch.py` 的控制台输出或 JSON：

1. `selected_gap`
2. `start/end`
3. `dominant_predicted_class`
4. `recommended_target_class`
5. `recommended_policy_action`

这证明：

1. 当前运行中的 gap2 已经被找到
2. 并且已经拿到了类型判定与推荐动作

### 10.2 证据二：allocator session header

在 allocator log 开头看：

1. `gap_watch_target_class=...`
2. `gap_watch_policy_action=...`
3. `gap_watch_control_file=...`

这证明：

1. allocator 已经加载了 targeted gap policy

### 10.3 证据三：`TRACE_POLICY`

重点看：

1. `policy_source=gap_watch_policy`
2. `gap_watch_class_match=1`
3. `gap_overlap_bytes>0`
4. `action=managed_prefetch_gpu` 或 `managed_advise_prefetch_gpu`

这证明：

1. 不是只有普通 policy 在工作
2. 而是真的有 allocation 因为命中 gap2 而被施策

### 10.4 证据四：`TRACE_GAP_WATCH_ALLOC/FREE`

重点看：

1. `alloc_policy_source=gap_watch_policy`
2. `alloc_policy_success=1`
3. `gap_watch_target_class=...`
4. `gap_watch_policy_action=...`
5. `lifetime_s=...`

这证明：

1. 命中的对象来自目标类别
2. 动作执行成功
3. 对象生命周期和 gap2 的实际行为一致

### 10.5 证据五：session summary

看 allocator log 结尾：

1. `Gap-watch overlap allocations`
2. `Gap-watch target-class matches`
3. `Gap-watch policy applied`
4. `Gap-watch policy success`
5. `Gap-watch policy failed`

这证明：

1. 命中规模是多少
2. 动作执行规模是多少
3. 成功率如何

### 10.6 证据六：`summarize_gap_watch_metrics.py`

最后看：

1. `gap_policy_records`
2. `gap_policy_success`
3. `gap_policy_overlap_bytes`
4. `dominant_predicted_class`
5. `dominant_phase`

如果这些指标都非零，而且主导类型符合预期，就可以说：

1. 已经实现了“检测 gap2 地址和类型”
2. 也已经实现了“针对这块地址做额外策略”
3. 并且有可回放的日志和汇总指标证明这一点

---

## 11. 当前实现的边界

这次实现已经把 targeted gap policy 跑通，但仍然有边界。

### 11.1 仍然是 managed 主路径

当前动作仍然是：

1. `managed_default`
2. `managed_prefetch_gpu`
3. `managed_advise_prefetch_gpu`

也就是说：

1. 还没有真的把 gap2 命中对象切换到独立 GPU pool
2. 也没有直接改成 `cudaMalloc`

### 11.2 类型判断仍然是启发式

`discover_gap_watch.py` 的类型判定来自：

1. overlap 到 gap2 的 `TRACE_POLICY`
2. phase 分布
3. overlap bytes
4. lifetime 中位数

这是高实用性的启发式，但不是“调用点真名”。

### 11.3 更激进的下一步

如果后续要继续推进，最直接的方向是：

1. 新增真正的 `gpu_pool` 动作
2. 把 `gap_watch_policy_action` 从 `advise_prefetch` 扩展到 `gpu_pool`
3. 再比较：
   - fault 是否下降
   - overlap 对象是否还频繁触页

---

## 12. 最终结论

本次修改后，项目已经从：

1. 只能发现 gap2
2. 只能观察 gap2

推进到：

1. 自动发现 gap2 地址
2. 自动识别 gap2 主导类型
3. 自动把“地址 + 类型 + 动作”下发给 allocator
4. allocator 对命中 gap2 的目标对象执行额外策略
5. 运行后输出可证明的指标与摘要

也就是说，本次已经实现了一个真正可运行的：

1. `discover gap2`
2. `classify gap2`
3. `apply targeted policy`
4. `prove it happened`

的完整闭环。

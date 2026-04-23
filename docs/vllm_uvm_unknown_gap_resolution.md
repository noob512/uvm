# vLLM UVM Unknown Gap Resolution Workflow

## 1. 目标

这份文档描述一套把 `unknown` 区域继续“落实名字”的工作流，重点解决三个问题：

1. 把 allocator trace 中已经识别出的 `warmup_workspace` / `runtime_scratch` / `runtime_workspace` 地址段导出成地址区间清单。
2. 对某个热点 gap（例如 `gap#2`）抽出所有重叠的 `alloc_id`，按时间顺序观察其重复模式。
3. 解释为什么以前的 `/tmp/vllm_uvm_address_regions.log` 只能看到 weight / KV，而看不到这些 unknown gap 的真实来源。

本次实现新增了三个能力：

1. [export_allocator_workspace_regions.py](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/export_allocator_workspace_regions.py)
2. [deep_dive_uvm_faults.py](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/deep_dive_uvm_faults.py) 的 `focus-gap` 导出能力
3. [uvm_allocator.cpp](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/uvm_allocator.cpp) 的 allocator-side `unknown/gap-watch` 诊断事件

## 2. 现象回顾

在已有分析里，`unknown` 往往高度集中在少数 gap：

1. 热点 gap 不是“到处散落的漏记对象”，而是少数几个地址空档反复被 fault。
2. 这些 fault 往往是强写导向，并且单页重复 fault 很多次。
3. allocator trace 又能证明这些 gap 被大量短命 allocation 反复覆盖。

这说明：

1. `unknown` 并不等于“完全无来源”
2. 更像“没有被 tensor address-log 记名的 workspace / scratch / warmup buffer”

## 3. 为什么以前 address-log 看不到它们

根因不在 UVM 驱动本身，而在日志层次不同。

### 3.1 `/tmp/vllm_uvm_address_regions.log` 的来源

这个文件由 Python 侧的高层语义对象写出，核心逻辑在：

1. [gpu_model_runner.py](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py)
2. `_collect_weight_address_rows(...)`
3. `_log_kv_cache_addresses(...)`
4. `_append_uvm_address_log(...)`

它记录的是：

1. 权重张量
2. KV cache 张量

也就是说，它本质上是“有名字的 tensor 语义清单”。

workspace / scratch 的问题在于：

1. 它们往往不是长期存在的高层 tensor
2. 很多只是库内部或算子内部的短命 managed allocation
3. 即使有地址，也未必有稳定的 Python 张量名字可以写进 address-log

所以以前 address-log 看不到它们是正常的。

### 3.2 `uvm_allocator.cpp` 其实捕捉到了 phase，但不是 address-log

底层 phase 记录在：

1. [uvm_allocator.cpp](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/uvm_allocator.cpp)
2. `uvm_set_phase(...)`
3. `TRACE_ALLOC`
4. `TRACE_FREE`
5. `TRACE_POLICY`

也就是说：

1. `uvm_allocator.cpp` 记录了 allocation 的 `phase`
2. 也记录了 `predicted_class`
3. 但它输出的是 allocator trace，不是 tensor address-log

因此“没有捕捉到 unknown 区域的分配阶段”这句话更准确地说应该是：

1. allocator trace 已经捕捉到了很多 phase
2. 但这些 phase 没被导出成 address-log 风格的 region 清单
3. 同时对于大量运行时分配，phase 仍然只会落到粗粒度的 `enabled`

### 3.3 为什么很多 unknown 只显示成 `enabled`

这是第二层原因，也是最关键的原因。

当前 phase 切换主要发生在这些高层阶段：

1. `load_model`
2. `initialize_kv_cache`
3. `uvm_enable:cublas_preinit`
4. `kernel_warmup:*`
5. `profile_run`
6. `capture_model`
7. 默认回到 `enabled`

也就是说，如果某一批运行时 workspace/scratch 发生在这些显式 phase 之外，它们只能被记成：

1. `phase=enabled`

所以：

1. `enabled` 不是“没有 phase”
2. 而是“当前没有更细粒度的 phase 标签”

### 3.4 还有一类日志污染：`size_bytes=0`

这次实现还修复了一个很重要的分析偏差。

在 allocator trace 里，某些 `TRACE_POLICY` 记录会出现：

1. `ptr=0x0`
2. `size_bytes=0`

如果直接拿去做 gap overlap，`end = start + size - 1` 会退化成：

1. `0xffffffffffffffff`

这样会把整个 gap 错误地视为“被完全覆盖”。

现在 [deep_dive_uvm_faults.py](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/deep_dive_uvm_faults.py) 已经显式过滤：

1. `alloc.size_bytes <= 0`
2. `alloc.ptr == 0`

所以新的 overlap 分析不会再被这些零长度哨兵记录污染。

## 4. 本次新增的能力

### 4.1 导出 allocator workspace region 清单

新增脚本：

1. [export_allocator_workspace_regions.py](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/export_allocator_workspace_regions.py)

作用：

1. 读取 allocator trace
2. 挑出 `warmup_workspace` / `runtime_scratch` / `runtime_workspace`
3. 导出成 address-log 风格的区域文件

输出行示意：

1. `kind=allocator:warmup_workspace`
2. `name=alloc_id=...|class=...|phase=...|action=...`
3. `start/end/size_bytes`

注意：

1. 这是“地址区间清单”
2. 不是对 fault 的静态最终分类
3. 因为这些地址可能在时间上被反复复用

所以它更适合：

1. 看哪些 phase 真的分配过哪些 workspace 区间
2. 作为 gap 命名时的辅助证据

而不应该直接替代 weight/KV 的静态 address matching。

### 4.2 `deep_dive_uvm_faults.py` 增加 `focus-gap`

新增能力：

1. `--focus-gap`
2. `--gap-alloc-csv`
3. `--gap-reuse-csv`
4. `--gap-report-json`
5. `--gap-preview-limit`

它会为指定 gap 输出：

1. 按 `alloc_elapsed_s` 排序的重叠 allocation 列表
2. 每条 overlap 的 `alloc_id / phase / predicted_class / ptr / end / lifetime`
3. 同一地址段的重复复用模式 summary

这就能直接回答：

1. `gap#2` 是不是反复被同一类 runtime scratch 复用
2. 它是否先出现在 warmup/autotune，再被 enabled 阶段继续反复使用
3. 哪些 `alloc_id` 最值得单独跟踪

### 4.3 `uvm_allocator.cpp` 增加 allocator-side 细粒度追踪

这是本次新增的第三层能力，目的是把 `gap#2` 这种热点 gap 的特征直接在运行当下抓出来，而不是完全依赖离线重叠分析。

新增环境变量：

1. `VLLM_UVM_UNKNOWN_DETAIL_ENABLE`
2. `VLLM_UVM_UNKNOWN_DETAIL_MIN_BYTES`
3. `VLLM_UVM_GAP_WATCH_ENABLE`
4. `VLLM_UVM_GAP_WATCH_NAME`
5. `VLLM_UVM_GAP_WATCH_START`
6. `VLLM_UVM_GAP_WATCH_END`
7. `VLLM_UVM_GAP_WATCH_ALL_CLASSES`
8. `VLLM_UVM_GAP_WATCH_MIN_BYTES`

新增日志事件：

1. `TRACE_UNKNOWN_DETAIL`
2. `TRACE_UNKNOWN_DETAIL_FREE`
3. `TRACE_GAP_WATCH_ALLOC`
4. `TRACE_GAP_WATCH_FREE`

它们分别回答：

1. 哪些 `unknown_managed` allocation 正在大量出现
2. 这些 unknown allocation 的生命周期有多短
3. 哪些 allocation 实际命中了 `gap#2`
4. 命中 `gap#2` 的对象属于什么 `phase / predicted_class / action`
5. 某次命中对整个 gap 的覆盖比例是多少

## 5. 为什么 allocator-side 捕获对 `gap#2` 特别重要

纯离线 overlap 的核心局限是：

1. 只能从结果倒推来源
2. 会把“后来地址复用”的结果混在一起看

而 allocator-side watch 的优势是：

1. 它在 allocation 发生时立刻记录
2. 每条日志都带有当时的 `phase`
3. 可以同时记录 `predicted_class`
4. 可以直接知道是否命中 watch range
5. free 时还能拿到真实 `lifetime_s`

所以对于 `gap#2` 这种“14.5 MiB 很小，但 fault 极热”的地址段，最有效的办法就是：

1. 在运行时直接 watch 这个地址段
2. 把所有命中它的 allocation/free 都抓出来

## 6. 推荐使用方式

### 6.1 先做基础 deep dive

```bash
python3 /home/ubuntu/nvidia-uvm-gpu/workloads/vllm/deep_dive_uvm_faults.py \
  --address-log /tmp/vllm_uvm_address_regions.log \
  --fault-log /tmp/uvm_kv_fault_addrs_stage0.log \
  --allocator-log /tmp/vllm_uvm_allocator_trace_stage0.log \
  --top-gaps 20 \
  --gap-csv /tmp/vllm_unknown_gap_heat_stage0.csv \
  --summary-json /tmp/vllm_fault_deep_dive_summary_stage0.real.json
```

先从这里拿到：

1. hottest gaps
2. `likely_kind`
3. `dominant_phase`
4. `top_overlapping_allocations`

### 6.2 再导出 workspace region 清单

```bash
python3 /home/ubuntu/nvidia-uvm-gpu/workloads/vllm/export_allocator_workspace_regions.py \
  --allocator-log /tmp/vllm_uvm_allocator_trace_stage0.log \
  --output-log /tmp/vllm_uvm_workspace_regions_stage0.log \
  --summary-json /tmp/vllm_uvm_workspace_regions_stage0.json \
  --min-size-bytes 1048576
```

输出：

1. `/tmp/vllm_uvm_workspace_regions_stage0.log`
2. `/tmp/vllm_uvm_workspace_regions_stage0.json`

这个导出可以直接回答：

1. `enabled` 阶段有哪些 runtime scratch/workspace 地址
2. `kernel_warmup:flashinfer_autotune` 是否真的覆盖了某些 hot gap
3. `uvm_enable:cublas_preinit` 是否留下可见的 warmup region

### 6.3 对热点 gap 做精细时间序列

如果先用样本验证格式是否正常：

```bash
python3 /home/ubuntu/nvidia-uvm-gpu/workloads/vllm/deep_dive_uvm_faults.py \
  --address-log /tmp/vllm_uvm_address_regions.log \
  --fault-log /tmp/uvm_kv_fault_addrs_stage0.log \
  --allocator-log /tmp/vllm_uvm_allocator_trace_stage0.log \
  --top-gaps 3 \
  --focus-gap 2 \
  --gap-preview-limit 20 \
  --max-faults 50000
```

如果要导出完整 gap 明细：

```bash
python3 /home/ubuntu/nvidia-uvm-gpu/workloads/vllm/deep_dive_uvm_faults.py \
  --address-log /tmp/vllm_uvm_address_regions.log \
  --fault-log /tmp/uvm_kv_fault_addrs_stage0.log \
  --allocator-log /tmp/vllm_uvm_allocator_trace_stage0.log \
  --top-gaps 20 \
  --focus-gap 2 \
  --gap-alloc-csv /tmp/gap2_allocs_stage0.csv \
  --gap-reuse-csv /tmp/gap2_reuse_stage0.csv \
  --gap-report-json /tmp/gap2_report_stage0.json \
  --summary-json /tmp/vllm_fault_deep_dive_summary_stage0.real.json
```

说明：

1. `gap-alloc-csv` 可能非常大
2. 对于像 `gap#2` 这种高度复用的热点 gap，CSV 可以到百 MB 级
3. JSON 现在只保留 preview 和 summary，避免无意义爆炸

### 6.4 直接在 allocator 侧 watch `gap#2`

如果你已经知道 `gap#2` 的地址段，例如：

1. `start=0x7cd2f5180000`
2. `end=0x7cd2f5ffffff`

注意：

1. 这个地址段通常是“某一次具体运行”的结果
2. 换一个新进程重新跑，`gap#2` 的绝对地址可能变化
3. 因此更稳妥的方式通常是“两阶段”：
   第一阶段先做一次 trace-only 发现本次 hottest gap
   第二阶段再用该次分析出来的地址段开启 `gap-watch`

可以直接这样跑：

```bash
./workloads/vllm/run_kv_fault_ratio.sh \
  --mode trace \
  --with-address-log \
  --trace-log /tmp/uvm_kv_fault_stats_gap2_watch.log \
  --address-trace-log /tmp/uvm_kv_fault_addrs_gap2_watch.log \
  --allocator-log /tmp/vllm_uvm_allocator_trace_gap2_watch.log \
  --uvm-trace-min-bytes 1048576 \
  --uvm-unknown-detail-enable 1 \
  --uvm-unknown-detail-min-bytes 4096 \
  --uvm-gap-watch-enable 1 \
  --uvm-gap-watch-name gap2 \
  --uvm-gap-watch-start 0x7cd2f5180000 \
  --uvm-gap-watch-end 0x7cd2f5ffffff \
  --uvm-gap-watch-all-classes 1 \
  --uvm-gap-watch-min-bytes 4096
```

建议解释：

1. `--uvm-unknown-detail-enable 1`
   让 `unknown_managed` allocation 额外输出 `TRACE_UNKNOWN_DETAIL`
2. `--uvm-unknown-detail-min-bytes 4096`
   过滤掉大量极小噪声，但保留页面级和小 scratch 特征
3. `--uvm-gap-watch-enable 1`
   开启地址段观察模式
4. `--uvm-gap-watch-all-classes 1`
   不只看 `unknown_managed`，也看 warmup/runtime_scratch 是否命中 gap2

如果你想更保守，只抓 unknown：

```bash
--uvm-gap-watch-all-classes 0
```

## 7. 如何看新输出

### 7.1 allocator-side 新事件

#### `TRACE_UNKNOWN_DETAIL`

重点字段：

1. `alloc_id`
2. `phase`
3. `size_bytes`
4. `size_bucket`
5. `stream`
6. `overlaps_gap_watch`
7. `gap_overlap_bytes`

用途：

1. 看 unknown 是否集中在某类大小
2. 看 unknown 是否几乎都发生在 `enabled`
3. 看 unknown 是否直接命中 `gap#2`

#### `TRACE_UNKNOWN_DETAIL_FREE`

重点字段：

1. `alloc_id`
2. `alloc_phase`
3. `lifetime_s`
4. `gap_overlap_bytes`

用途：

1. 判断 unknown allocation 是短命 scratch 还是长期对象
2. 判断命中 `gap#2` 的 unknown 是否在毫秒级释放

#### `TRACE_GAP_WATCH_ALLOC`

重点字段：

1. `watch_name`
2. `phase`
3. `predicted_class`
4. `overlap_bytes`
5. `overlap_ratio_of_watch`

用途：

1. 直接看谁覆盖了 `gap#2`
2. 看覆盖来源是 `warmup_workspace` 还是 `runtime_scratch`
3. 看它对整个 gap 的覆盖比例到底有多大

#### `TRACE_GAP_WATCH_FREE`

重点字段：

1. `alloc_id`
2. `alloc_phase`
3. `alloc_predicted_class`
4. `overlap_bytes`
5. `lifetime_s`

用途：

1. 直接拿到命中 `gap#2` 的对象生命周期
2. 判断该 gap 更像 warmup 临时工作区还是 runtime 高频 scratch

### 7.2 workspace region log

重点看：

1. phase
2. kind
3. alloc_id
4. start/end

如果某一类 hot gap 附近长期出现：

1. `allocator:warmup_workspace`
2. `allocator:runtime_scratch`
3. `allocator:runtime_workspace`

说明这些地址段已经不再是“纯 unknown”。

### 7.3 focused gap 的 ordered overlap

重点看：

1. `alloc_elapsed_s`
2. `phase`
3. `predicted_class`
4. `overlap_ratio_of_gap`
5. `lifetime_s`

如果一个 gap：

1. 先在 `kernel_warmup:flashinfer_autotune` 里出现大覆盖率分配
2. 随后在 `enabled` 阶段被许多短命 allocation 高频重用

那么它通常意味着：

1. 这段地址先被 warmup/autotune 使用
2. 后面又被 runtime scratch/workspace 反复复用

### 7.4 focused gap 的 reuse pattern

重点看：

1. `ptr_hex..end_hex`
2. `reuse_count`
3. `top_phase`
4. `top_predicted_class`
5. `median_lifetime_s`

如果某个地址段：

1. `reuse_count` 非常高
2. `top_phase=enabled`
3. `median_lifetime_s` 很短

那么它更像：

1. runtime scratch
2. runtime workspace

而不是 weight/KV。

## 8. 对 `gap#2` 的实际解释框架

在实际样本里，`gap#2` 会呈现出这样一种结构：

1. 最早出现较大覆盖率的 `kernel_warmup:flashinfer_autotune`
2. 之后在 `enabled` 阶段被大量更小的区间反复复用
3. 中位寿命通常非常短

这意味着 `gap#2` 更像：

1. 一段被 warmup/autotune 首次使用过的地址
2. 之后在 runtime 中被 scratch/workspace 反复重用

换句话说，`gap#2` 并不是单一来源的“某个固定张量”，而更像：

1. 热点工作区地址池
2. 地址复用非常剧烈

## 9. 为什么还会残留 `unknown_managed`

即使现在有了新的导出和 focus-gap 工具，`unknown_managed` 也不会立刻完全消失，原因包括：

1. phase 粒度仍然不够细，很多运行时分配只能看到 `enabled`
2. 一些 allocation 虽然被 trace 到了，但没有更高层语义名
3. 某些很小的 managed 分配没有稳定 lifecycle 信息
4. 一部分 gap 可能来自 trace 没覆盖到的库内部行为

因此当前更合理的目标不是“把所有 unknown 归零”，而是：

1. 把 hottest unknown gap 落到 `warmup_workspace / runtime_scratch / runtime_workspace`
2. 明确哪些 residual unknown 只是粒度不够，而不是完全无来源

## 10. 当前结论

本次修改后，项目已经具备下面这套完整链路：

1. 用 tensor address-log 识别 `weight / kv`
2. 用 allocator trace 识别 `warmup_workspace / runtime_scratch / runtime_workspace`
3. 用 `focus-gap` 把热点 gap 对应的 `alloc_id` 时间序列拉平
4. 用 `reuse pattern` 判断它更像 warmup 还是 runtime scratch

也就是说，后续再看 `unknown` 时，不需要只停留在：

1. “它是 unknown”

而可以继续回答：

1. 它是不是 `flashinfer_autotune` 首次热身出来的 workspace
2. 它是不是 `enabled` 阶段反复复用的 runtime scratch
3. 它是否只是因为 address-log 只记录 tensor，而没有把 allocator 语义导出

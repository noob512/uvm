# vLLM UVM Fault Address Classification

## 1. 文档目标

本文档说明以下问题：

1. `/tmp/uvm_kv_fault_addrs.log` 和 `/tmp/vllm_uvm_address_regions.log` 是否可以直接比较。
2. 如何把 fault 地址分类为 `weight`、`kv_cache` 或 `unknown`。
3. 为什么会出现大量 `unknown`，以及如何进一步判断它们更像：
   - `warmup_workspace`
   - `runtime_scratch`
   - 其他未记录的临时缓冲
4. 如何使用脚本：
   - [analyze_uvm_fault_addresses.py](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/analyze_uvm_fault_addresses.py)
   - [deep_dive_uvm_faults.py](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/deep_dive_uvm_faults.py)
5. 如何解读本次真实运行的分析结果。

本文档以当前仓库中的 UVM allocator、phase tracing、fault 分类脚本和 allocator correlation 能力为准。

## 2. 三类日志分别记录什么

### 2.1 `/tmp/vllm_uvm_address_regions.log`

该日志由 vLLM 记录，用于描述已知张量有效载荷的 UVA 区间。

典型内容包括：
1. `weight:param`
2. `weight:buffer`
3. `kv_cache`
4. `kv_cache:contiguous_range`
5. `kv_cache:span_range`

其中：
1. `weight:*` 和具体的 `kv_cache` 行表示真实张量区间。
2. `kv_cache:contiguous_range` / `kv_cache:span_range` 是摘要行，适合做范围诊断，不适合当作具体张量标签。

### 2.2 `/tmp/uvm_kv_fault_addrs.log`

该日志由 NVIDIA UVM 驱动记录，表示 replayable fault 发生时的地址信息。

关键点：
1. 驱动会记录原始 fault 地址。
2. 随后会按页对齐，得到分类时使用的 page address。
3. 我们在分类时默认优先使用页对齐地址，而不是 raw address。

因此，真正的比较方式是：

`fault_page_address` vs `start <= addr <= end`

也就是：
1. 从 `/tmp/uvm_kv_fault_addrs.log` 取页对齐后的 fault 地址
2. 与 `/tmp/vllm_uvm_address_regions.log` 中的区间逐一匹配

### 2.3 `/tmp/vllm_uvm_allocator_trace.log`

该日志由当前改造后的 UVM allocator 记录，用于回答：

1. 某个地址区间是否真的来自 traced `cudaMallocManaged`
2. 它是在哪个 phase 分配的
3. 它后来是否被释放
4. 它的生命周期有多长

关键事件类型：
1. `TRACE_PHASE`
2. `TRACE_ALLOC`
3. `TRACE_FREE`

这份日志不是 fault 日志，而是 allocation lifecycle 日志。它的作用是把 `unknown` 从“只能猜测”提升为“可以通过 phase 和 lifetime 去归因”。

## 3. 为什么前两个日志可以直接比较

结论：

这两个日志来自不同层级，但它们都指向同一个 GPU UVA 空间，因此设计上就是可比较的。

原因如下：

1. `vllm_uvm_address_regions.log`
   - 记录的是张量 `data_ptr()` 对应的 UVA 区间
2. `uvm_kv_fault_addrs.log`
   - 记录的是 UVM fault 发生时的页地址
3. fault 的页地址与张量区间同处一个统一虚拟地址空间

所以：
1. 如果 fault 地址命中某个 `weight:*` 区间
   - 该 fault 可以归类为 `weight`
2. 如果 fault 地址命中某个 `kv_cache*` 区间
   - 该 fault 可以归类为 `kv_cache`
3. 如果没有命中任何已知区间
   - 只能归类为 `unknown`

这并不要求每个 fault 都必须命中某个已知张量。未命中的常见原因包括：

1. 分配器内部对齐或填充间隙
2. 未被 address log 记录的临时 workspace / scratch
3. 不同运行混在一起的旧日志
4. fault 发生时相关张量区间尚未被当前日志捕获

## 4. 这些地址是否经过 `cudaMallocManaged`

### 4.1 结论

在当前 `VLLM_USE_UVM=1` 模式下：

1. `vllm_uvm_address_regions.log` 中的权重和 KV cache 地址，极大概率都来自 `cudaMallocManaged`
2. `/tmp/uvm_kv_fault_addrs.log` 本身只是 fault 地址日志，不能单独证明某个地址一定由 `cudaMallocManaged` 分配
3. 但凡是命中 `weight` / `kv_cache` 已知区间的 fault，都可以追溯到当前 UVM allocator 管理的张量地址空间

### 4.2 分配链路

当前代码路径是：

`PyTorch CUDA allocation -> CUDAPluggableAllocator -> uvm_malloc -> cudaMallocManaged`

相关实现可参考：
1. [env_override.py](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/env_override.py)
2. [uvm.py](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/device_allocator/uvm.py)
3. [uvm_allocator.cpp](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/uvm_allocator.cpp)

### 4.3 权重和 KV cache 为什么共享同一条 allocator 路径

1. 权重加载阶段会创建 CUDA tensor storage
2. KV cache 初始化阶段也会创建 CUDA tensor storage
3. 当当前 CUDA allocator 已被替换为 UVM allocator 后，这两类分配最终都会进入 `uvm_malloc(...)`
4. `uvm_malloc(...)` 内部直接调用 `cudaMallocManaged(...)`

因此：
1. `weight` 和 `kv_cache` 在地址来源上是相通的
2. fault 如果命中这两类区间，说明它落在当前 UVM allocator 生成的已知张量地址空间上

## 5. address log 中有哪些对象

### 5.1 主要 phase

通常会看到两个主要阶段：

1. `phase=load_model`
   - 包含 `weight:param` 和 `weight:buffer`
2. `phase=initialize_kv_cache`
   - 包含 `kv_cache` 及其摘要行

### 5.2 分类时哪些行算“具体区间”

分类脚本只把这些行当作 concrete region：

1. `weight:param`
2. `weight:buffer`
3. `kv_cache`
4. `kv_cache:cross_layers`

而这些行只用于辅助：

1. `kv_cache:contiguous_range`
2. `kv_cache:span_range`

原因是摘要范围可能覆盖了实际张量之间的空洞，不能直接当作某个具体对象。

## 6. fault 分类规则

脚本采用以下规则：

1. 如果 fault 页地址命中 `weight:*`
   - 分类为 `weight`
2. 如果 fault 页地址命中 `kv_cache*`
   - 分类为 `kv_cache`
3. 如果未命中任何 concrete region
   - 分类为 `unknown`
4. 如果未命中具体 KV 行，但落在 KV 摘要范围内
   - 分类为 `unknown_in_kv_summary_span`

这套规则的核心思想是：
1. 对 `weight` 和 `kv_cache` 只做高置信度命中
2. 对无法证明的地址保留 `unknown`
3. 不做过度归类

## 7. `unknown` 为什么重要

如果没有 `unknown`，结论会被夸大。

一个 fault 未匹配的原因可能是：
1. 它落在张量区间之间的 gap
2. 它落在 address log 没有记录的 workspace / scratch
3. 它来自不同运行
4. 它来自 allocator 或 runtime 的临时对象

因此，`unknown` 的含义不是“无意义”，而是：

“当前无法证明它属于已记录的 weight/KV 张量有效载荷”

## 8. 两个分析脚本的职责

### 8.1 `analyze_uvm_fault_addresses.py`

负责基础分类：

1. 解析 `vllm_uvm_address_regions.log`
2. 默认选择最新 PID
3. 选择每个 phase 的最新 section
4. 将 region 归一化为可查询区间
5. 流式扫描 fault log
6. 输出：
   - 区域覆盖
   - fault 分类计数
   - 地址空间可比性评估
7. 可选输出：
   - regions CSV
   - faults CSV
   - summary JSON

### 8.2 `deep_dive_uvm_faults.py`

负责深挖 `unknown`：

1. 统计每类 fault 的：
   - faults
   - unique pages
   - avg faults per unique page
   - READ/WRITE 比例
   - fault type 分布
2. 构建 concrete region 之间的 gap
3. 把 `unknown` fault 归入 gap
4. 输出：
   - top unknown gaps
   - gap 热度
   - 左右邻接 region
   - warmup/workspace 启发式判断
5. 如果提供 `--allocator-log`
   - 将 hot gap 与 `TRACE_ALLOC/TRACE_FREE` 做重叠分析
   - 输出每个 gap 的：
     - `dominant_phase`
     - `likely_kind`
     - `unique_coverage_of_gap`
     - `cumulative_overlap_of_gap`
     - 生命周期统计
     - top overlapping allocations

## 9. allocator correlation 如何解读

### 9.1 `unique_coverage_of_gap`

表示：

allocator trace 中出现过的地址，对该 gap 的去重覆盖比例。

例如：
1. `94.45%`
   - 表示这个 gap 的绝大多数地址都至少被某个 traced allocation 覆盖过一次

### 9.2 `cumulative_overlap_of_gap`

表示：

把多次短命 allocation 对同一 gap 的重叠全部累计后的比例。

这个值可以大于 `100%`。

含义不是：
1. 这块 gap 静态上被覆盖了 200%

而是：
1. 这块 gap 被反复拿来当临时缓冲复用了很多次

### 9.3 `likely_kind`

这是脚本给出的 gap 类型推断，常见值包括：

1. `warmup_workspace`
2. `runtime_scratch`
3. `weight_or_load_workspace`
4. `kv_related_workspace`
5. `unresolved_unknown_gap`

### 9.4 `lifetime`

表示被释放的重叠 allocation 的生命周期统计：

1. `min`
2. `median`
3. `p95`
4. `max`

如果生命周期只有毫秒级，并且同一地址被反复申请/释放，这非常像 scratch/workspace，而不像长期常驻的 weight/KV。

### 9.5 `top_overlapping_allocations`

表示与当前 gap 重叠最明显的若干 traced allocations，字段包括：

1. `alloc_id`
2. `ptr_hex..end_hex`
3. `phase`
4. `freed`
5. `free_phase`
6. `lifetime_s`
7. `overlap_ratio_of_gap`

## 10. allocator trace 当前记录了哪些 phase

当前 phase tracing 已覆盖：

1. `uvm_enable:cublas_preinit`
2. `load_model`
3. `profile_run`
4. `initialize_kv_cache`
5. `compile_warmup:size=<N>`
6. `kernel_warmup:deep_gemm`
7. `kernel_warmup:flashinfer_autotune`
8. `kernel_warmup:flashinfer_attention`
9. `capture_model`
10. `enabled`

其中：
1. 显式 warmup / autotune phase 更容易判定为临时 workspace
2. `enabled` 表示当前没有更细粒度 phase 标签，但仍可通过生命周期和地址复用模式判断它更像 runtime scratch

## 11. 本次真实日志分析结果

本节基于以下真实输入：

1. `/tmp/uvm_kv_fault_addrs.log`
2. `/tmp/vllm_uvm_address_regions.log`
3. `/tmp/vllm_uvm_allocator_trace.log`

以及输出文件：

1. [/tmp/vllm_unknown_gap_heat.real.csv](/tmp/vllm_unknown_gap_heat.real.csv)
2. [/tmp/vllm_fault_deep_dive_summary.real.json](/tmp/vllm_fault_deep_dive_summary.real.json)

本次自动选中的 vLLM PID 是：

1. `4221`

### 11.1 分类总览

`weight`
1. faults: `410,601`
2. unique pages: `330,440`
3. avg faults per unique page: `1.242589`
4. access:
   - `READ`: `410,583`
   - `WRITE`: `18`
5. 特征：
   - 几乎完全是读 fault
   - 每页平均 fault 次数很低

`kv_cache`
1. faults: `2,623`
2. unique pages: `1,124`
3. avg faults per unique page: `2.333630`
4. access:
   - `READ`: `1,941`
   - `WRITE`: `682`
5. 特征：
   - 以读为主
   - 也存在一定比例写 fault

`unknown`
1. faults: `2,890,152`
2. unique pages: `9,489`
3. avg faults per unique page: `304.579197`
4. access:
   - `READ`: `1`
   - `WRITE`: `2,890,151`
5. 特征：
   - 几乎 `100% WRITE`
   - 热度远高于 `weight`

### 11.2 top unknown gaps

#### gap#2

1. 地址：`0x72161f180000 .. 0x72161fffffff`
2. 大小：`14.500 MiB`
3. faults：`2,848,990`
4. 占全部 unknown：`98.58%`
5. unique pages：`2,534`
6. avg faults per unique page：`1124.31`
7. 左邻接：`lm_head.weight`
8. 右邻接：`model.layers.47.mlp.experts.w2_weight`
9. 判定：`warmup_workspace`

这是本次分析的核心热点。

#### gap#3

1. 地址：`0x721645200000 .. 0x721645ffffff`
2. 大小：`14.000 MiB`
3. faults：`13,177`
4. 占全部 unknown：`0.46%`
5. 判定：`warmup_workspace`

#### gap#70

1. 地址：`0x72192fa00000 .. 0x72192fffffff`
2. 大小：`6.000 MiB`
3. faults：`9,442`
4. 判定：`runtime_scratch`

#### gap#8

1. 地址：`0x72168fc00000 .. 0x72168fffffff`
2. 大小：`4.000 MiB`
3. faults：`679`
4. 判定：`runtime_scratch`

#### gap#189

1. 地址：`0x721edc200000 .. 0x721edd3fffff`
2. 大小：`18.000 MiB`
3. faults：`26`
4. 主导 phase：`uvm_enable:cublas_preinit`
5. 判定：`warmup_workspace`

### 11.3 为什么 `gap#2` 几乎可以确定是 workspace

`gap#2` 的 allocator correlation 结果：

1. `dominant_phase=enabled`
2. `likely_kind=warmup_workspace`
3. `unique_coverage_of_gap=94.45%`
4. `cumulative_overlap_of_gap=202.05%`
5. `warmup_like_overlap_bytes=12,582,912`
6. `freed_overlap_bytes=3,072,103,168`
7. `live_overlap_bytes=0`
8. `freed_count=1343`
9. lifetime:
   - `min=0.000306s`
   - `median=0.001686s`
   - `p95=0.022487s`
   - `max=4.609722s`

它的 top overlapping allocations 中直接出现：

1. `phase=kernel_warmup:flashinfer_autotune`
2. 大量 `phase=enabled`
3. 且几乎全部 `freed=True`

这说明：

1. 该 gap 与 warmup/autotune 阶段存在直接重叠
2. 该 gap 在后续运行中还被当作短命 runtime buffer 反复复用
3. 它不像权重
4. 也不像长期常驻的 KV cache

### 11.4 为什么 `gap#3` 更像 warmup workspace

`gap#3` 的前几个重叠 allocation 几乎都来自：

1. `kernel_warmup:flashinfer_autotune`

并且：

1. 对 gap 的覆盖比例高达 `85.71%`
2. 生命周期中位数约 `0.007782s`
3. 全部都是 `freed=True`

因此它非常像 warmup/autotune 临时工作区。

### 11.5 为什么 `gap#70` 和 `gap#8` 更像 runtime scratch

这两个 gap 的共同特征是：

1. `dominant_phase=enabled`
2. 没有明显 warmup_like phase 重叠
3. 但存在稳定重复的短命 alloc/free
4. 生命周期中位数分别约：
   - `gap#70`: `0.005359s`
   - `gap#8`: `0.009256s`

因此脚本将它们归为：

1. `runtime_scratch`

这意味着它们更像正式推理阶段反复使用的小型工作区或中间缓冲。

### 11.6 `gap#189` 说明了什么

`gap#189` 的主导 phase 是：

1. `uvm_enable:cublas_preinit`

这说明部分 unknown 并不是推理阶段才出现，也可能来自启动早期的库预初始化工作区。

### 11.7 warmup/workspace 总体结论

脚本给出的总体 heuristic 结果是：

1. `conclusion=strongly_suggests_unlogged_warmup_or_workspace_buffers`
2. `confidence=high`
3. `score=14`

主要证据包括：

1. `unknown` 几乎完全是写 fault
2. 最热 gap 单独占了 `98.58%` 的 unknown
3. `unknown` 每页平均 fault 次数极高
4. allocator trace 与 `flashinfer_autotune` / `cublas_preinit` / `enabled` 临时分配存在明显重叠
5. 大量重叠 allocation 后续被释放

因此，本次真实日志下的 `unknown` 主体更像：

1. warmup/autotune workspace
2. runtime scratch buffer

而不像：

1. 漏记的权重 fault
2. 漏记的 KV cache fault
3. 随机噪声

## 12. 如何开启 allocator trace

建议设置：

```bash
export VLLM_USE_UVM=1
export VLLM_UVM_LOG_FILE=/tmp/vllm_uvm_allocator_trace.log
export VLLM_UVM_TRACE_MIN_BYTES=1048576
```

字段说明：

1. `VLLM_UVM_LOG_FILE`
   - allocator trace 输出位置
2. `VLLM_UVM_TRACE_MIN_BYTES`
   - 只记录大于等于该阈值的 allocation/free
   - `1048576 = 1 MiB`

如果想看更小的 scratch buffer，可把阈值降到：

```bash
export VLLM_UVM_TRACE_MIN_BYTES=4096
```

但日志体积会明显增大。

## 13. 使用方式

### 13.1 基础分类

```bash
python3 /home/ubuntu/nvidia-uvm-gpu/workloads/vllm/analyze_uvm_fault_addresses.py \
  --regions-csv /tmp/vllm_regions_normalized.csv \
  --summary-json /tmp/vllm_fault_classification_summary.json
```

### 13.2 深挖 unknown

```bash
python3 /home/ubuntu/nvidia-uvm-gpu/workloads/vllm/deep_dive_uvm_faults.py \
  --gap-csv /tmp/vllm_unknown_gap_heat.csv \
  --summary-json /tmp/vllm_fault_deep_dive_summary.json
```

### 13.3 结合 allocator trace 做 gap 归因

```bash
python3 /home/ubuntu/nvidia-uvm-gpu/workloads/vllm/deep_dive_uvm_faults.py \
  --allocator-log /tmp/vllm_uvm_allocator_trace.log \
  --gap-csv /tmp/vllm_unknown_gap_heat.csv \
  --summary-json /tmp/vllm_fault_deep_dive_summary.json
```

### 13.4 直接生成本次这种“真实 gap 归因报告”

```bash
python3 /home/ubuntu/nvidia-uvm-gpu/workloads/vllm/deep_dive_uvm_faults.py \
  --allocator-log /tmp/vllm_uvm_allocator_trace.log \
  --top-gaps 10 \
  --gap-csv /tmp/vllm_unknown_gap_heat.real.csv \
  --summary-json /tmp/vllm_fault_deep_dive_summary.real.json
```

### 13.5 使用运行脚本一键采集

推荐使用：

```bash
./workloads/vllm/run_kv_fault_ratio.sh \
  --mode trace \
  --with-address-log \
  --trace-log /tmp/uvm_kv_fault_stats.log \
  --address-trace-log /tmp/uvm_kv_fault_addrs.log \
  --allocator-log /tmp/vllm_uvm_allocator_trace.log \
  --uvm-trace-min-bytes 1048576
```

这会同时生成：

1. fault stats log
2. per-fault address log
3. vLLM address region log
4. allocator trace log

## 14. 输出文件建议

推荐保留这些文件：

1. `/tmp/vllm_regions_normalized.csv`
2. `/tmp/vllm_fault_classification_summary.json`
3. `/tmp/vllm_unknown_gap_heat.csv`
4. `/tmp/vllm_fault_deep_dive_summary.json`
5. `/tmp/vllm_uvm_allocator_trace.log`

对于本次真实分析，还额外保留：

1. [/tmp/vllm_unknown_gap_heat.real.csv](/tmp/vllm_unknown_gap_heat.real.csv)
2. [/tmp/vllm_fault_deep_dive_summary.real.json](/tmp/vllm_fault_deep_dive_summary.real.json)

## 15. 局限性

1. 脚本默认使用最新 PID。
   - 如果日志中混有多次运行，请使用 `--pid`
2. 分类只基于已记录的张量 payload 区间。
   - 不会自动重建 allocator 内部所有隐藏区间
3. `unknown` 不能自动等价成“错误”
   - 它经常正是 runtime workspace 的信号
4. allocator correlation 依赖新的 trace 格式。
   - 老的 `vllm_uvm_allocations.log` 只有大小摘要，不能做 gap 级别归因
5. `enabled` phase 不是强语义 phase。
   - 对 `enabled` 的判定仍需结合 lifetime、重复覆盖和 fault 模式一起看

## 16. 交付物

本次工作交付：

1. 脚本：
   - [analyze_uvm_fault_addresses.py](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/analyze_uvm_fault_addresses.py)
   - [deep_dive_uvm_faults.py](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/deep_dive_uvm_faults.py)
2. 文档：
   - [vllm_uvm_fault_address_classification.md](/home/ubuntu/nvidia-uvm-gpu/docs/vllm_uvm_fault_address_classification.md)
3. 本次真实分析结果：
   - [/tmp/vllm_unknown_gap_heat.real.csv](/tmp/vllm_unknown_gap_heat.real.csv)
   - [/tmp/vllm_fault_deep_dive_summary.real.json](/tmp/vllm_fault_deep_dive_summary.real.json)

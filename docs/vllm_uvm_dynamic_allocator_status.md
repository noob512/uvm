# vLLM UVM Dynamic Allocator Status

## 1. 文档目的

这份文档回答一个核心问题：

1. 我们当前是否已经实现了“根据分配大小、分配时机/阶段、分配作用，并最终结合生命周期来动态调整内存分配方式”的 allocator？
2. 如果还没有，当前已经走到哪一阶段？
3. 后续应该如何继续演进？

本文不是单点功能说明，而是对当前项目状态的阶段性总览。

---

## 2. 最终目标

项目的目标不是简单地“给某些 allocation 加一点 prefetch”，而是构建一个更完整的动态策略层：

```text
allocation event
  -> classify by phase + size + role + lifetime/reuse
  -> choose placement action
  -> emit trace for validation
  -> iteratively refine policy
```

换句话说，目标能力至少包含四层：

1. 能识别：
   - 这是 `weight`
   - 这是 `kv`
   - 这是 `warmup workspace`
   - 这是 `runtime scratch/workspace`
2. 能决策：
   - 继续 `cudaMallocManaged`
   - `cudaMallocManaged + prefetch`
   - `cudaMallocManaged + advise`
   - GPU 原生池化分配
   - 其他后续动作
3. 能验证：
   - 每次分类和动作都可追溯
   - 能和 fault 地址、gap、phase、lifetime 做闭环验证
4. 能迭代：
   - 根据热点、重用模式、生命周期统计继续更新策略

如果用一句话概括：

1. 我们要做的是“动态策略 allocator”
2. 而不是“在 managed allocator 上零散加几个特判”

---

## 3. 当前阶段结论

截至当前代码状态，项目可以明确划分为：

### 3.1 已完成：阶段 0

阶段 0 的目标是：

1. 不改变分配行为
2. 先把 allocation 按 `phase + size + role` 做语义分类
3. 把分类结果写入日志，建立后续策略闭环

这部分已经实现，并且已经被实验验证。

### 3.2 已实现但范围较窄：阶段 1

阶段 1 的目标是：

1. 先对最保守、最容易识别的一类对象动手
2. 即 `warmup_workspace`
3. 仍然保持 `cudaMallocManaged`
4. 但在分配后立刻执行 `cudaMemPrefetchAsync(..., device)`
5. 可选加 `cudaMemAdviseSetPreferredLocation(device)`

这部分代码已经落地，但属于“局部行为改变”，还不是完整动态 allocator。

### 3.3 尚未完成：阶段 2 及以后

真正意义上的“动态 allocator”还没有完全实现，尤其是下面这些关键能力仍未落地：

1. 运行时 `runtime_scratch` / `runtime_workspace` 的专门 placement 动作
2. 基于 `lifetime/reuse/fault heat` 的在线决策
3. GPU scratch pool / bucketed pool
4. 根据历史统计自动把 managed 对象升级为更激进的 GPU 放置方式
5. 基于 phase + size + lifetime 的闭环自适应策略引擎

因此，当前最准确的结论是：

1. 我们已经完成了动态 allocator 的“观测层、分类层、验证层”
2. 也实现了一个“warmup-only”的保守行为优化
3. 但还没有完成面向 runtime scratch/workspace 的完整动态分配器

---

## 4. 当前已经实现了什么

这一节按“能力层”来描述，而不是按文件罗列。

### 4.1 已实现：phase 感知

当前 allocator 已经能够感知高层运行阶段，包括但不限于：

1. `load_model`
2. `initialize_kv_cache`
3. `profile_run`
4. `kernel_warmup:*`
5. `uvm_enable:cublas_preinit`
6. `enabled`

这意味着 allocator 不再只看到“裸的 size”，而是已经具备“这块内存是在什么阶段被申请”的上下文。

这一步非常关键，因为：

1. 同样是 8 MiB
2. 在 `kernel_warmup:flashinfer_autotune` 和 `enabled`
3. 语义可能完全不同

### 4.2 已实现：AllocationClass 语义分类

当前代码中已经存在以下分类：

1. `weight_persistent`
2. `kv_persistent`
3. `warmup_workspace`
4. `runtime_scratch`
5. `runtime_workspace`
6. `unknown_managed`

这说明 allocator 已经不再把所有 managed allocation 一概而论，而是开始做“语义推断”。

当前分类规则的核心是：

1. `load_model` -> `weight_persistent`
2. `initialize_kv_cache` -> `kv_persistent`
3. `warmup/autotune/preinit/profile_run` -> `warmup_workspace`
4. `enabled` 且 `1 MiB .. 16 MiB` -> `runtime_scratch`
5. `enabled` 且 `16 MiB .. 128 MiB` -> `runtime_workspace`
6. 其他 -> `unknown_managed`

因此，今天的系统已经具备：

1. `phase + size` 驱动的第一版分类器

但还没有做到：

1. `phase + size + lifetime` 驱动的运行时决策器

### 4.3 已实现：PolicyAction 抽象

当前代码里已经引入了动作层抽象：

1. `managed_default`
2. `managed_prefetch_gpu`

这一步的意义是：

1. 分类结果和执行动作已经解耦
2. 后续继续增加：
   - `managed_advise_gpu_preferred`
   - `gpu_scratch_pool`
   - `gpu_direct`
3. 在架构上已经有落点

也就是说，当前代码结构已经为后续扩展动态策略留好了接口。

### 4.4 已实现：`TRACE_POLICY`

每次 allocation 后，allocator 都可以输出：

1. 当前 `phase`
2. 当前 `size`
3. `predicted_class`
4. `action`
5. 动作是否成功
6. 动作错误信息

这意味着：

1. 阶段 0 的核心目标已经完成
2. 我们已经可以对“策略命中情况”做离线验证
3. 这也是后续做 A/B 实验的基础

### 4.5 已实现：unknown deep dive 和 gap 归因

项目现在不仅能判断 `fault` 是不是命中 `weight` / `kv`，还已经具备进一步分析 `unknown` 的能力，包括：

1. `Top Unknown Gaps`
2. 每类 `unique pages`
3. `READ/WRITE` 比例
4. gap 热度
5. allocator trace overlap
6. phase 归因
7. lifetime 统计
8. workspace/scratch region 导出

这一步非常重要，因为它回答了两个关键问题：

1. `unknown` 不是随机噪声
2. `unknown` 的主体是可以被落实名字的

### 4.6 已实现：same-run auto gap-watch

之前的主要问题是：

1. `gap#2` 每次重启 vLLM 都可能换地址
2. 跨进程拿上一轮的 gap 地址去 watch 下一轮，容易完全打不中

现在项目已经支持：

1. 同一 vLLM 进程内先跑 `probe`
2. 自动发现本轮 hottest gap
3. 动态下发给 allocator
4. 再在同一进程里跑 `main`

这使得“gap 热点定位”从跨运行猜测，变成了同一地址空间里的直接观测。

---

## 5. 现阶段最重要的实验结论

当前阶段最关键的价值，不只是“代码里有分类器”，而是实验已经给出了一些明确结论。

### 5.1 结论一：weight / kv 不是 unknown 的主体

已有 fault 分类和 gap 归因结果已经说明：

1. 大量 `unknown fault` 并不落在 weight / kv 已知区间里
2. 它们主要集中在少数 gap
3. 这些 gap 不是“满天散落的碎片”

这意味着：

1. unknown 有结构
2. 可以继续被归因

### 5.2 结论二：gap#2 的主体不是权重，也不是 KV

基于 same-run 分析和 allocator trace overlap，当前已经有较强证据表明：

1. `gap#2` 不是 `weight_persistent`
2. `gap#2` 不是 `kv_persistent`
3. 它更像 runtime 中反复复用的 scratch/workspace 热点地址池

尤其是已经观察到：

1. overlap alloc 数量非常高
2. 主导 phase 几乎都是 `enabled`
3. lifetime 中位数很短
4. 地址区间被反复 alloc/free

这组证据更符合：

1. 运行时临时工作区
2. 高频写入
3. 地址池复用

而不是：

1. 长生命周期权重
2. 稳定常驻 KV 区

### 5.3 结论三：当前的“生命周期”主要是分析信号，不是在线决策信号

这是理解当前阶段最容易混淆的一点。

今天项目已经能从 trace 中分析出：

1. 某类 allocation 生命周期很短
2. 某些 gap 被反复复用
3. 某些 size bucket fault 很热

但是这些信息目前主要用于：

1. 离线分析
2. 文档归因
3. 决策下一步该改什么规则

而不是已经被直接用于：

1. 本次 allocation 时就立即根据 lifetime 预测选择不同 API

所以要非常明确：

1. `lifetime` 已经进入了分析面
2. 但还没有真正进入 allocator 的在线策略闭环

---

## 6. 当前还没有实现什么

这一节是阶段判断的关键。

### 6.1 尚未实现：runtime scratch/workspace 的差异化动作

当前真正会改变行为的只有：

1. `warmup_workspace`

而下面这些都还没有真正切换动作：

1. `runtime_scratch`
2. `runtime_workspace`
3. `unknown_managed`
4. `weight_persistent`
5. `kv_persistent`

也就是说，虽然分类器能把它们区分出来，但执行层还没有针对这些类别做进一步差异化。

### 6.2 尚未实现：GPU scratch pool

如果目标是降低 `gap#2` 这类 runtime scratch 热点带来的 UVM fault 干扰，真正有潜力的动作往往不是：

1. 继续放任其按 managed 缺页迁移

而是：

1. 建一个 GPU scratch pool
2. 把高频短命对象放进可复用的显存池

这一层目前还没有实现。

### 6.3 尚未实现：lifetime 驱动的在线动作选择

当前系统还不能做到下面这种在线规则：

1. 如果某个 size + phase 组合在最近 N 次分配里生命周期都小于 T
2. 且 fault 热度高
3. 且反复命中同一地址池
4. 那么把它从 `managed_default` 升级为 `gpu_pool`

这类“闭环在线策略”才是真正的动态 allocator 核心，但当前还没有落地。

### 6.4 尚未实现：role/use 的精细注入

从项目目标来说，“分配作用”最好不是完全靠 `phase + size` 猜，而是尽量拿到更明确的语义标签，例如：

1. 这是某个算子的 workspace
2. 这是 autotune 缓冲
3. 这是 decode runtime scratch
4. 这是长生命周期激活缓存

目前这部分还主要依赖启发式推断，而不是更细粒度的调用点语义注入。

所以今天的“作用识别”状态是：

1. 已经有第一版推断
2. 但还不够精细

---

## 7. 用阶段语言来判断：我们现在在哪

为了和之前的设计文档对齐，可以把当前状态归纳成下面这张表。

### 7.1 阶段 0：trace-only 建模

目标：

1. 只分类
2. 只写日志
3. 不改行为

状态：

1. 已完成

证据：

1. `AllocationClass` 已实现
2. `PolicyAction` 已实现
3. `TRACE_POLICY` 已实现
4. `run_kv_fault_ratio.sh` 已支持 `trace_only`

### 7.2 阶段 1：warmup 的保守优化

目标：

1. 不改 allocator API 契约
2. 仅对 `warmup_workspace` 做 `managed + prefetch`

状态：

1. 已实现
2. 但仍属于“局部策略”
3. 需要继续做系统性 A/B 验证

证据：

1. `choose_policy_action(...)` 已能对 `warmup_workspace` 返回 `managed_prefetch_gpu`
2. 已有 `cudaMemPrefetchAsync`
3. 可选 `cudaMemAdviseSetPreferredLocation(device)`
4. `run_kv_fault_ratio.sh` 已支持 `prefetch` / `warmup_prefetch`

### 7.3 阶段 2：runtime scratch/workspace 策略化

目标：

1. 把 `runtime_scratch` / `runtime_workspace` 真正区别对待
2. 让 hottest unknown gap 对应对象不再只是被分析，而是被优化

状态：

1. 尚未实现

### 7.4 阶段 3：phase + size + lifetime 闭环策略

目标：

1. 根据重用频率、生命周期、fault 热度动态调整 placement
2. 让策略逐步从启发式演进为数据驱动

状态：

1. 尚未实现

### 7.5 结论

如果必须用一句最准确的话描述当前项目阶段：

1. 当前项目已经完成“动态 allocator 的可观测与可分类底座”
2. 已经开始做“warmup-only 的保守策略干预”
3. 但还没有进入“runtime scratch/workspace 真正被动态分配策略接管”的阶段

---

## 8. 为什么说“还没完成真正的动态 allocator”

因为“动态 allocator”的关键词不只是“有分类器”，而是“分类会稳定地驱动后续内存放置动作”。

当前系统离这个目标还差三步：

### 8.1 从离线分析走向在线决策

今天我们已经知道：

1. 哪些 gap 热
2. 哪些 phase 热
3. 哪些 size bucket 热
4. 哪些 allocation 很短命

但这些更多是用于人来分析，而不是 allocator 自动调整。

### 8.2 从 warmup-only 走向 runtime

当前改行为的对象只覆盖：

1. `warmup_workspace`

而项目现在最痛的热点其实已经很清楚：

1. runtime `enabled` 阶段的 hotspot gap

如果不把 runtime scratch/workspace 纳入策略层，dynamic allocator 仍然不完整。

### 8.3 从单动作走向多动作

当前动作基本只有：

1. `managed_default`
2. `managed_prefetch_gpu`

而真正有用的策略空间应该包括：

1. managed 默认
2. managed + advise
3. managed + prefetch
4. GPU pool
5. GPU direct
6. 可能的 CPU preferred for cold objects

当前还没有展开到这个层级。

---

## 9. 当前最合理的下一阶段方向

结合已有实验和当前代码结构，后续最合理的路线不是继续泛泛地加规则，而是围绕热点问题推进。

### 9.1 第一优先级：把 runtime scratch/workspace 纳入真正的动作层

原因：

1. 当前 hottest unknown gap 的主体已经基本指向 runtime scratch/workspace
2. 如果继续只优化 warmup，主问题不会明显改善

建议动作：

1. 先做 `runtime_scratch -> candidate_gpu_pool`
2. 再做 `runtime_workspace -> candidate_prefetch_or_pool`

### 9.2 第二优先级：把 lifetime/reuse 从分析信号升级为策略输入

建议方式：

1. 先按 `phase + size bucket` 维护统计
2. 再看最近窗口内：
   - 分配次数
   - 生命周期中位数
   - overlap/fault 热度
3. 满足阈值才升级 placement 动作

这样会比“只看 size”更稳。

### 9.3 第三优先级：把 unknown_managed 继续压缩

目标不是强行让 unknown 变成 0，而是：

1. 把真正热的 unknown 命名出来
2. 让 unknown 逐步收敛到低频尾部

这一步对后续策略稳定性很重要。

---

## 10. 推荐的状态定义

为了后续写实验报告和提交实现说明，建议统一采用下面的状态表述。

### 10.1 可以说已经完成的

1. 已完成阶段 0：trace-only policy classification
2. 已完成分类、日志、gap deep dive、same-run auto gap discovery 基础设施
3. 已实现阶段 1 的 warmup prefetch 代码路径

### 10.2 更准确的表述

1. 当前系统已经具备 `phase + size` 驱动的分类能力
2. 当前系统已经具备基于 trace 的 `lifetime/reuse` 分析能力
3. 当前系统只对 `warmup_workspace` 实现了有限的行为改变
4. 当前系统尚未实现 `phase + size + lifetime` 驱动的完整 runtime dynamic allocator

### 10.3 不建议夸大的表述

下面这些说法当前还不准确：

1. “已经实现完整动态 allocator”
2. “已经按 phase + size + lifetime 自动分配不同内存”
3. “已经解决 runtime unknown hotspot”
4. “runtime scratch/workspace 已经被策略层接管”

---

## 11. 与已有文档的关系

如果需要查分项细节，可分别参考：

1. [vllm_uvm_phase_size_lifetime_strategy_plan.md](/home/ubuntu/nvidia-uvm-gpu/docs/vllm_uvm_phase_size_lifetime_strategy_plan.md)
   - 讲的是总体设计目标和后续路线
2. [vllm_uvm_stage0_trace_policy_implementation.md](/home/ubuntu/nvidia-uvm-gpu/docs/vllm_uvm_stage0_trace_policy_implementation.md)
   - 讲的是阶段 0 的分类与 trace-only 落地
3. [vllm_uvm_policy_stage0_stage1_implementation.md](/home/ubuntu/nvidia-uvm-gpu/docs/vllm_uvm_policy_stage0_stage1_implementation.md)
   - 讲的是阶段 0 / 1 的代码实现
4. [vllm_uvm_unknown_gap_resolution.md](/home/ubuntu/nvidia-uvm-gpu/docs/vllm_uvm_unknown_gap_resolution.md)
   - 讲的是 unknown gap 继续落实名字的方法
5. [vllm_uvm_same_run_auto_gap_watch.md](/home/ubuntu/nvidia-uvm-gpu/docs/vllm_uvm_same_run_auto_gap_watch.md)
   - 讲的是同一运行中自动发现并 watch hottest gap

本文档的定位是：

1. 把上述文档串起来
2. 给出“项目当前进度”的统一判断

---

## 12. 最终结论

截至当前版本，这个项目已经不再是“纯 managed allocator + 离线看 fault”的状态，而是已经拥有：

1. phase 感知
2. 语义分类
3. policy trace
4. unknown gap 深度归因
5. same-run gap hotspot 发现能力
6. warmup-only 的保守行为优化

但是，如果目标是：

1. 根据分配内存大小
2. 根据分配时机和阶段
3. 根据分配作用
4. 再结合生命周期和复用特征
5. 动态改变内存分配方式

那么当前最准确的状态判断是：

1. 我们已经完成了这个目标的“基础设施阶段”和“第一步可控优化阶段”
2. 也就是已经完成阶段 0，并已进入阶段 1
3. 但真正面向 runtime scratch/workspace 的动态 allocator 还没有完成
4. 后续工作的主战场将是：把 `phase + size + lifetime/reuse` 从分析框架升级为 runtime placement 策略


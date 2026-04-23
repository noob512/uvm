# vLLM UVM Phase + Size + Lifetime Strategy Plan

## 1. 目标

本文档给出一个完整的项目改造方案，用于把当前 vLLM 的 UVM 分配策略从：

1. 所有对象统一走 `cudaMallocManaged`

演进为：

1. 根据 `phase`
2. 根据 `allocation size`
3. 根据 `lifetime`

对不同类别的分配采取差异化策略，从而减少 replayable fault 对实验和性能分析的干扰。

本文档只给方案，不做代码修改。

## 2. 问题背景

当前项目已经具备以下能力：

1. 在 vLLM 中启用 UVM allocator
2. 记录权重和 KV cache 的地址区间
3. 记录 replayable fault 地址
4. 记录 allocator trace：
   - `TRACE_PHASE`
   - `TRACE_ALLOC`
   - `TRACE_FREE`
5. 对 `unknown` fault 做 gap 归因和 phase/lifetime 分析

根据现有实验结果，`unknown` 的主体并不是随机噪声，而是更像：

1. `warmup_workspace`
2. `runtime_scratch`

并且其关键特征是：

1. fault 主要是 `WRITE`
2. fault 热点集中在少数 gap
3. 与 `kernel_warmup:flashinfer_autotune`、`uvm_enable:cublas_preinit`、`enabled` 等 phase 重叠
4. 很多重叠 allocation 生命周期只有毫秒级甚至亚毫秒级
5. 同一地址区间被反复 alloc/free

这说明：

1. 不能再把所有 managed allocation 一视同仁
2. 也不能只靠“申请大小”一个维度来做判断
3. 更合理的做法是引入 `phase + size + lifetime` 的组合策略

## 3. 为什么不能只看 size

“按大小决定是否放 GPU”是一个直观思路，但不够稳。

原因有三类：

### 3.1 不同 phase 会复用相近大小的 allocation

例如：

1. warmup/autotune workspace 可能是 8 MiB、12 MiB、14 MiB
2. runtime scratch 也可能落在 1 MiB 到 16 MiB 之间
3. 某些长期对象的辅助缓冲也可能恰好落在这些大小区间

所以：

1. 相同 size 不代表相同语义

### 3.2 size 不包含“是否高频复用”的信息

一个 4 MiB allocation 可能是：

1. 一次性初始化缓冲
2. 每轮 decode 都反复申请/释放的 scratch
3. warmup/autotune 中的临时 workspace

这三种对象的优化策略并不相同。

### 3.3 size 不包含“是否值得常驻 GPU”的信息

一个 allocation 是否应该直接留在 GPU 上，还取决于：

1. 生命周期是否足够长
2. fault 是否高频
3. 是否会被重复访问
4. 是否会与更重要的 weight/KV 竞争显存

因此，size 只能作为信号之一，不能单独作为决策依据。

## 4. 设计目标

目标不是“一次性消灭所有 fault”，而是优先解决以下问题：

1. 减少 `unknown` 对实验分析的干扰
2. 将明显的 warmup/workspace/scratch fault 从权重/KV fault 中剥离
3. 对高频、短命、可识别的临时缓冲采取更合适的 placement 策略
4. 在不破坏当前 UVM 实验框架的前提下逐步演进

具体目标分为四档：

### 4.1 第一目标

让策略具备“识别能力”：

1. 能判断 allocation 更像 `weight`
2. 能判断 allocation 更像 `kv_cache`
3. 能判断 allocation 更像 `warmup_workspace`
4. 能判断 allocation 更像 `runtime_scratch`

### 4.2 第二目标

让策略具备“可配置能力”：

1. 每类对象可映射到不同动作
2. 不同 phase 可使用不同阈值
3. 策略可通过环境变量或配置文件调整

### 4.3 第三目标

让策略具备“可验证能力”：

1. 策略命中要能写日志
2. 每次 placement 决策都要可回放
3. 结果要能和 fault 日志做闭环对比

### 4.4 第四目标

在保证稳定性的前提下尝试减少 fault：

1. 对明显的 warmup workspace 做 GPU prefetch 或 GPU 原生分配
2. 对明显的 runtime scratch 建立 GPU resident pool
3. 对 weight/KV 保持当前主路径不被误伤

## 5. 总体思路

建议引入一个新的“策略层”：

`allocation event -> classify by phase + size + lifetime -> choose placement action`

这层逻辑不替代现有 UVM allocator，而是插在 allocator 之前或之内，对 allocation 做策略判断。

可以理解为：

1. 当前：
   - 所有对象都走 `cudaMallocManaged`
2. 改造后：
   - 一部分继续走 `cudaMallocManaged`
   - 一部分走 `cudaMallocManaged + advise/prefetch`
   - 一部分走 GPU scratch pool
   - 一部分在 warmup 阶段做特殊处理

## 6. 核心策略模型

### 6.1 输入信号

策略引擎的输入至少包括：

1. `phase`
2. `size_bytes`
3. `device`
4. `alloc_id`
5. `caller type` 或当前上下文标签
6. `history`
   - 同地址/同 size/同 phase 是否反复出现
7. `lifetime estimate`

其中：

1. `phase` 来自当前已有的 `uvm_allocation_phase(...)`
2. `size_bytes` 来自 allocator 本身
3. `history` 和 `lifetime estimate` 来自 allocator 维护的统计信息

### 6.2 输出类别

策略层先不直接决定 API，而是先产出一个语义类别。

建议的第一版类别：

1. `weight_persistent`
2. `kv_persistent`
3. `warmup_workspace`
4. `runtime_scratch`
5. `runtime_workspace`
6. `profile_workspace`
7. `unknown_managed`

### 6.3 输出动作

在类别基础上，再映射到动作。

建议的动作集合：

1. `managed_default`
   - 继续走 `cudaMallocManaged`
2. `managed_prefetch_gpu`
   - 走 `cudaMallocManaged`，分配后立刻 `cudaMemPrefetchAsync(..., device)`
3. `managed_advise_gpu_preferred`
   - 走 `cudaMallocManaged`，附加 `cudaMemAdviseSetPreferredLocation(device)`
4. `managed_advise_cpu_preferred`
   - 走 `cudaMallocManaged`，附加 CPU preferred location
5. `gpu_scratch_pool`
   - 不走 managed，改为 GPU 原生内存池
6. `gpu_direct`
   - 直接走 `cudaMalloc`
7. `trace_only`
   - 不改变行为，只记录命中情况

## 7. phase + size + lifetime 的分类规则

下面给出推荐的第一版启发式规则。

### 7.1 `weight_persistent`

判定信号：

1. `phase=load_model`
2. size 较大
3. 生命周期很长
4. 不会快速 free
5. 后续命中 fault 主要偏 `READ`

策略建议：

1. 保持 managed 主路径
2. 可选做 GPU prefetch
3. 可选做 preferred location = GPU

不建议：

1. 直接切到 scratch pool

### 7.2 `kv_persistent`

判定信号：

1. `phase=initialize_kv_cache`
2. 地址可与 `kv_cache` 区间对应
3. 生命周期长
4. 后续访问读写混合

策略建议：

1. 保持 managed 或受控预取
2. 可以考虑更强的 GPU preferred policy

### 7.3 `warmup_workspace`

判定信号：

1. `phase` 命中：
   - `uvm_enable:cublas_preinit`
   - `kernel_warmup:*`
   - `compile_warmup:*`
   - `profile_run`
2. size 处于中小范围或重复稳定范围
3. 生命周期短
4. 同 phase / 同 size / 同地址模式反复出现
5. 大量 `freed=True`

策略建议：

1. 第一阶段：
   - `managed_prefetch_gpu`
2. 第二阶段：
   - 对已确认热点改为 `gpu_scratch_pool`

### 7.4 `runtime_scratch`

判定信号：

1. `phase=enabled`
2. 生命周期极短
3. 地址或 size 呈现高频重复
4. 与 hottest unknown gaps 重叠
5. 多次 alloc/free 命中同一 gap

策略建议：

1. 优先改为 `gpu_scratch_pool`
2. 若先不换分配 API，则至少做：
   - `managed_prefetch_gpu`
   - `cudaMemAdviseSetPreferredLocation(device)`

### 7.5 `runtime_workspace`

判定信号：

1. `phase=enabled`
2. 生命周期比 `runtime_scratch` 稍长
3. 仍会被释放
4. 但未达到 persistent 对象级别

策略建议：

1. `managed_prefetch_gpu`
2. 必要时纳入 pool

## 8. lifetime 如何得到

当前 allocator 已经能在 `TRACE_FREE` 中记录：

1. `alloc_phase`
2. `free_phase`
3. `lifetime_s`

但如果策略要在“分配时”就生效，不能等对象 free 之后才知道 lifetime。

因此建议引入两级 lifetime：

### 8.1 离线 lifetime

来自现有 trace：

1. 用于分析
2. 用于构建规则
3. 用于生成策略白名单/黑名单

### 8.2 在线 lifetime 预测

在线时只能做估计，推荐方式：

1. 维护按 `(phase, rounded_size_bucket)` 聚合的历史统计
2. 记录：
   - 过去 N 次的 median lifetime
   - p95 lifetime
   - reuse count
3. 新 allocation 到来时，用历史模式预测它更像：
   - persistent
   - short-lived scratch
   - warmup workspace

也就是说：

1. 第一次看到某类 allocation 时，不做激进动作
2. 当某类 allocation 被重复观测到后，再升级策略

## 9. 需要改动的项目位置

以下是建议的改造位置。

### 9.1 `workloads/vllm/vllm/uvm_test/uvm_allocator.cpp`

这是策略的核心落点。

建议增加：

1. `AllocationPolicy`
   - 表示某次 allocation 的策略结果
2. `AllocationClass`
   - 表示语义分类
3. `classify_allocation(...)`
   - 输入 `phase + size + history`
   - 输出语义分类
4. `choose_allocation_action(...)`
   - 输入语义分类
   - 输出动作
5. `AllocationHistory`
   - 按 `(phase, size_bucket)` 聚合历史
6. `TRACE_POLICY`
   - 额外记录策略命中日志

### 9.2 `workloads/vllm/vllm/vllm/device_allocator/uvm.py`

建议增加：

1. Python 层策略开关
2. 环境变量解析
3. 允许从 Python 注入策略配置

例如：

1. `VLLM_UVM_POLICY_ENABLE=1`
2. `VLLM_UVM_POLICY_MODE=trace_only|prefetch|pool`
3. `VLLM_UVM_POLICY_CONFIG=/path/to/json`

### 9.3 phase 标注点

当前 phase 标注已经不错，但仍建议增强。

建议重点补充：

1. 更细的 prefill/decode phase
2. 运行时 operator 级 phase
3. 对大 scratch 来源再加一层语义标签

潜在位置包括：

1. [gpu_worker.py](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py)
2. [gpu_model_runner.py](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py)
3. [kernel_warmup.py](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/warmup/kernel_warmup.py)

## 10. 推荐的实施阶段

不建议直接一步切到“把一部分 allocation 改成 `cudaMalloc`”。

建议分四阶段实施。

### 阶段 0：继续做 trace-only 建模

目标：

1. 不改行为
2. 只增加策略分类日志

做法：

1. `classify_allocation(...)` 运行
2. 输出 `TRACE_POLICY`
3. 不改变分配 API

成功标准：

1. 能稳定把热点对象分成：
   - `warmup_workspace`
   - `runtime_scratch`
   - `weight_persistent`
   - `kv_persistent`

### 阶段 1：对 warmup workspace 做保守优化

目标：

1. 降低 warmup/workspace 的 UVM fault 干扰

做法：

1. 对命中 `warmup_workspace` 的 allocation
   - 仍走 `cudaMallocManaged`
   - 但立即做 `cudaMemPrefetchAsync(..., device)`
   - 可选再加 `cudaMemAdviseSetPreferredLocation(device)`

理由：

1. 侵入性低
2. 不改变 allocator API 契约
3. 先看 fault 能否显著下降

### 阶段 2：引入 GPU scratch pool

目标：

1. 把高频 runtime scratch 从 managed fault 域里挪出去

做法：

1. 为 `runtime_scratch` 建立固定 size bucket 的 GPU pool
2. 常见 bucket 示例：
   - 1 MiB
   - 2 MiB
   - 4 MiB
   - 8 MiB
   - 12 MiB
   - 16 MiB
3. 对命中 bucket 且高频复用的 allocation：
   - 从池中分配
   - 释放时回池，不回系统

理由：

1. 这类对象生命周期极短
2. 反复 alloc/free 成本高
3. 也是 unknown fault 的主要来源之一

### 阶段 3：对 weight / KV 做 selective advise

目标：

1. 保持长期对象稳定
2. 减少不必要 fault

做法：

1. `weight_persistent`
   - managed + GPU preferred
   - 或按层预取
2. `kv_persistent`
   - managed + selective prefetch
   - 避免与其他对象争抢 GPU 内存过猛

## 11. 推荐的数据结构

### 11.1 size bucket

建议把 size 做 bucket 化，而不是按原始字节值直接聚类。

例如：

1. `<64 KiB`
2. `64 KiB .. 256 KiB`
3. `256 KiB .. 1 MiB`
4. `1 MiB .. 2 MiB`
5. `2 MiB .. 4 MiB`
6. `4 MiB .. 8 MiB`
7. `8 MiB .. 16 MiB`
8. `>=16 MiB`

这样能提高规则稳定性。

### 11.2 history key

推荐用：

`(phase, size_bucket, device)`

作为历史聚合键。

可选再加：

1. caller tag
2. operator tag

### 11.3 runtime stats

每个 key 维护：

1. alloc count
2. free count
3. median lifetime
4. p95 lifetime
5. recent address reuse ratio
6. recent fault overlap score

## 12. 配置方案

建议通过环境变量和可选 JSON 配置文件控制。

### 12.1 环境变量

建议增加：

1. `VLLM_UVM_POLICY_ENABLE`
2. `VLLM_UVM_POLICY_MODE`
3. `VLLM_UVM_POLICY_TRACE_ONLY`
4. `VLLM_UVM_POLICY_WARMUP_PREFETCH_MIN_BYTES`
5. `VLLM_UVM_POLICY_SCRATCH_POOL_ENABLE`
6. `VLLM_UVM_POLICY_SCRATCH_POOL_MAX_BYTES`

### 12.2 配置文件

建议支持 JSON：

```json
{
  "default_action": "managed_default",
  "rules": [
    {
      "phase_prefix": "kernel_warmup:",
      "size_min_bytes": 1048576,
      "predicted_lifetime_max_s": 0.5,
      "class": "warmup_workspace",
      "action": "managed_prefetch_gpu"
    },
    {
      "phase": "enabled",
      "size_bucket": "1MiB-8MiB",
      "predicted_lifetime_max_s": 0.02,
      "reuse_min_count": 10,
      "class": "runtime_scratch",
      "action": "gpu_scratch_pool"
    }
  ]
}
```

## 13. 日志与可观测性方案

策略如果不可观测，就很难验证。

建议增加新的日志事件：

### 13.1 `TRACE_POLICY`

字段建议包括：

1. `alloc_id`
2. `ptr`
3. `end`
4. `size_bytes`
5. `phase`
6. `size_bucket`
7. `predicted_class`
8. `predicted_lifetime_s`
9. `action`
10. `history_alloc_count`
11. `history_median_lifetime_s`

### 13.2 `TRACE_POOL`

如果引入 scratch pool，建议再记录：

1. `pool_bucket`
2. `pool_hit`
3. `pool_miss`
4. `pool_bytes_in_use`

## 14. 如何验证该方案有效

建议从三个层面验证。

### 14.1 功能正确性

验证点：

1. 服务能正常启动
2. benchmark 能正常完成
3. 不引入 allocator 崩溃、双重释放、越界等问题

### 14.2 fault 统计改善

重点指标：

1. `unknown` fault 绝对数量
2. `unknown` 占总 fault 的比例
3. hottest unknown gaps 的 fault 热度
4. `enabled` phase 对 unknown gap 的重叠量

期望：

1. `warmup_workspace`
2. `runtime_scratch`

这两类的 fault 显著下降

### 14.3 对 weight/KV 信号的净化

重点看：

1. `weight` fault 占比是否更稳定
2. `kv_cache` fault 占比是否更可解释
3. `unknown` 是否不再淹没主信号

## 15. 风险与副作用

### 15.1 GPU 内存压力上升

如果把太多 workspace/scratch 直接放到 GPU：

1. 可能减少 page fault
2. 但也可能把 GPU 显存顶满

### 15.2 误分类风险

如果某类 allocation 被误判为 scratch 并从 pool 分配：

1. 可能改变原先的生命周期假设
2. 可能影响某些库的内部行为

### 15.3 与 CUDA graph / 外部库交互风险

某些 workspace 由外部库控制，行为未必完全稳定。

因此建议：

1. 先 trace-only
2. 再 prefetch
3. 最后才考虑改分配 API

## 16. 推荐的最小可行路线

如果目标是最快得到可验证收益，推荐顺序如下：

### 16.1 第一步

实现：

1. `TRACE_POLICY`
2. `classify_allocation(...)`

但不改分配动作。

### 16.2 第二步

只对这两类对象启用 `managed_prefetch_gpu`：

1. `kernel_warmup:*`
2. `uvm_enable:cublas_preinit`

### 16.3 第三步

对满足以下条件的 `enabled` phase allocation 启用 scratch pool：

1. size bucket 稳定
2. predicted lifetime 很短
3. 在 allocator trace 中高频重复出现
4. 与 hottest unknown gaps 强重叠

## 17. 预期结果

如果策略生效，预期会看到：

1. `unknown` 总 fault 数下降
2. hottest unknown gaps 热度下降
3. `warmup_workspace` 和 `runtime_scratch` 的 allocator trace 重叠区 fault 下降
4. `weight` / `kv_cache` 分类结果更稳定
5. prompt 数增长时，`unknown` 对总统计的污染减弱

## 18. 总结

这个方案的核心不是“猜哪些 allocation 小，就把它们放 GPU”，而是：

1. 先通过 `phase` 判断语义上下文
2. 再通过 `size` 判断候选类型
3. 再通过 `lifetime` 和复用历史判断是否值得改 placement

即：

`phase -> size -> lifetime/history -> class -> action`

这种方案比单纯按 size 切换 allocator 更稳，也更贴合当前项目已经具备的日志和分析能力。

它最大的优点是：

1. 可以逐步上线
2. 每一步都能用现有 fault/allocator trace 工具验证
3. 不会一开始就破坏当前的 UVM 实验框架

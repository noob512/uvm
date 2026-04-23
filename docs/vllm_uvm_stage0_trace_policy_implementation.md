# vLLM UVM Stage 0 Trace Policy Implementation

## 1. 文档目的

本文档专门解释阶段 0 是如何修改和实现的，对应的范围是：

1. 在 allocator 中引入语义分类
2. 在 allocator 中引入策略动作枚举
3. 在每次 allocation 后输出 `TRACE_POLICY`
4. 保证默认模式仍然是 `trace_only`
5. 通过运行脚本把策略环境变量传到 vLLM 进程

本阶段的核心原则是：

1. **不改变分配 API**
2. **不改变默认运行行为**
3. **只增加策略建模和日志能力**

## 2. 修改文件

阶段 0 涉及两个文件：

1. [uvm_allocator.cpp](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/uvm_allocator.cpp)
2. [run_kv_fault_ratio.sh](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/run_kv_fault_ratio.sh)

其中：

1. `uvm_allocator.cpp`
   - 负责真正实现 `AllocationClass`
   - 负责真正实现 `PolicyAction`
   - 负责真正实现 `classify_allocation(...)`
   - 负责真正输出 `TRACE_POLICY`
2. `run_kv_fault_ratio.sh`
   - 负责把策略开关和阈值通过环境变量传给 vLLM 进程
   - 负责给实验时切换 `trace_only` / `prefetch` 提供 CLI 参数

## 3. 为什么阶段 0 要改 allocator 而不是只改 Python

阶段 0 的目标是“每一次 allocation 都要被分类并写日志”。

如果只在 Python 层做：

1. 能看到高层 phase
2. 但拿不到最底层真实 allocation 的完整细节
3. 也无法确保所有 PyTorch CUDA 分配都被覆盖

而 `uvm_allocator.cpp` 是当前 pluggable allocator 的核心入口：

1. 所有经 UVM allocator 接管的 CUDA 分配最终都会进入 `uvm_malloc(...)`
2. 因此分类逻辑放在这里最完整、最稳定

## 4. 新增配置项

在 `uvm_allocator.cpp` 里新增了以下全局配置项：

可参考：
[uvm_allocator.cpp](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/uvm_allocator.cpp#L35)

新增项：

1. `policy_enabled`
2. `policy_warmup_prefetch_enabled`
3. `policy_warmup_advise_gpu`
4. `policy_warmup_prefetch_min_bytes`
5. `policy_mode`

它们的作用分别是：

1. `policy_enabled`
   - 是否开启策略层
   - 阶段 0 默认值为 `1`
2. `policy_mode`
   - 当前策略模式
   - 阶段 0 默认是 `trace_only`
3. `policy_warmup_prefetch_enabled`
   - 阶段 1 用
   - 阶段 0 默认不会触发真实 prefetch
4. `policy_warmup_prefetch_min_bytes`
   - 阶段 1 用
   - 但阶段 0 也会把它记录到日志 header
5. `policy_warmup_advise_gpu`
   - 阶段 1 用
   - 阶段 0 默认不使用

## 5. 新增语义分类 `AllocationClass`

可参考：
[uvm_allocator.cpp](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/uvm_allocator.cpp#L61)

新增的 `AllocationClass` 是：

1. `WeightPersistent`
2. `KvPersistent`
3. `WarmupWorkspace`
4. `RuntimeScratch`
5. `RuntimeWorkspace`
6. `UnknownManaged`

为什么要先引入这个枚举：

1. 因为策略层第一步不是直接做动作
2. 而是先把 allocation 归入一个“语义类别”
3. 这样后续阶段 1、阶段 2 才能在同样的分类结果上叠加不同动作

也就是说：

`AllocationClass` 解决的是“它像什么”

而不是：

“现在就怎么处理它”

## 6. 新增策略动作 `PolicyAction`

可参考：
[uvm_allocator.cpp](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/uvm_allocator.cpp#L70)

新增的 `PolicyAction` 是：

1. `ManagedDefault`
2. `ManagedPrefetchGpu`

阶段 0 中这两个动作的关系是：

1. `ManagedDefault`
   - 默认动作
   - 表示仍然保持 `cudaMallocManaged` 的原始行为
2. `ManagedPrefetchGpu`
   - 为阶段 1 预留
   - 阶段 0 默认不会触发

这样设计的好处是：

1. 阶段 0 和阶段 1 复用同一套动作框架
2. 阶段 0 只需要保证所有 allocation 都会得到一个动作结果
3. 但这个动作结果在默认模式下仍然是 `managed_default`

## 7. 新增辅助函数

### 7.1 环境变量读取

新增：

1. `read_bool_from_env(...)`
2. `read_size_from_env(...)`
3. `read_string_from_env(...)`

可参考：
[uvm_allocator.cpp](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/uvm_allocator.cpp#L111)

作用：

1. 从环境变量读取策略参数
2. 保证 allocator 初始化时就确定当前 policy mode
3. 让策略切换不需要改 Python 代码

### 7.2 分类/动作转字符串

新增：

1. `allocation_class_to_string(...)`
2. `policy_action_to_string(...)`

可参考：
[uvm_allocator.cpp](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/uvm_allocator.cpp#L147)

作用：

1. 把枚举值转换成日志中的稳定字符串
2. 让后续分析脚本可以直接按字符串解析

### 7.3 size bucket

新增：

1. `size_bucket_for(...)`

可参考：
[uvm_allocator.cpp](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/uvm_allocator.cpp#L175)

作用：

1. 把原始大小离散化为 bucket
2. 便于后续按 `(phase, size_bucket)` 观察热点模式

当前 bucket 划分为：

1. `<64KiB`
2. `64KiB-256KiB`
3. `256KiB-1MiB`
4. `1MiB-2MiB`
5. `2MiB-4MiB`
6. `4MiB-8MiB`
7. `8MiB-16MiB`
8. `>=16MiB`

## 8. `classify_allocation(...)` 是怎么实现的

可参考：
[uvm_allocator.cpp](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/uvm_allocator.cpp#L186)

该函数输入：

1. `phase`
2. `size`

输出：

1. `AllocationClass`

当前第一版规则是启发式规则：

### 8.1 `load_model -> weight_persistent`

规则：

1. 如果 `phase == "load_model"`
2. 返回 `WeightPersistent`

原因：

1. `load_model` 对应模型权重加载
2. 这类对象通常生命周期长
3. 语义上更接近持久权重

### 8.2 `initialize_kv_cache -> kv_persistent`

规则：

1. 如果 `phase == "initialize_kv_cache"`
2. 返回 `KvPersistent`

原因：

1. 对应 KV cache 初始化
2. 语义上明显是持久 KV 对象

### 8.3 warmup / autotune / preinit / profile_run -> warmup_workspace

规则：

1. phase 包含：
   - `warmup`
   - `autotune`
   - `preinit`
2. 或者 `phase == "profile_run"`
3. 返回 `WarmupWorkspace`

原因：

1. 现有实验里 hottest unknown gap 与这些 phase 高度重叠
2. 这些 phase 对应的 allocation 很多是短命临时工作区
3. 因此阶段 0 先统一归入 `warmup_workspace`

### 8.4 `enabled + 1MiB..16MiB -> runtime_scratch`

规则：

1. 如果 `phase == "enabled"`
2. 且大小在 `1 MiB .. 16 MiB`
3. 返回 `RuntimeScratch`

原因：

1. 现有 deep-dive 结果显示，很多运行期热点 unknown gap 都处于这个数量级
2. 且生命周期通常较短
3. 先把这部分视为运行期 scratch 候选

### 8.5 `enabled + 16MiB..128MiB -> runtime_workspace`

规则：

1. 如果 `phase == "enabled"`
2. 且大小在 `16 MiB .. 128 MiB`
3. 返回 `RuntimeWorkspace`

原因：

1. 这类对象可能仍然是临时区
2. 但比 `runtime_scratch` 更大、更重
3. 因此先单独区分

### 8.6 其他 -> unknown_managed

规则：

1. 所有不满足上述条件的对象
2. 返回 `UnknownManaged`

原因：

1. 阶段 0 的目标是先保守分类
2. 不强行把不确定对象归入更激进的类别

## 9. `choose_policy_action(...)` 是怎么实现的

可参考：
[uvm_allocator.cpp](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/uvm_allocator.cpp#L211)

这个函数输入：

1. `AllocationClass`
2. `size`
3. `device`

输出：

1. `PolicyAction`

阶段 0 的关键点在于：

1. 即使策略层开启
2. 默认 `policy_mode=trace_only`
3. 因此返回值仍然会是：
   - `ManagedDefault`

只有在后续阶段 1 显式开启时，`warmup_workspace` 才会返回：

1. `ManagedPrefetchGpu`

所以：

1. 阶段 0 中 `choose_policy_action(...)` 已经存在
2. 但默认只会产生“无行为变化”的动作

## 10. allocator 初始化时做了什么

在 `init_log_file()` 中新增了 policy 配置读取。

可参考：
[uvm_allocator.cpp](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/uvm_allocator.cpp#L228)

主要工作：

1. 读取 `VLLM_UVM_POLICY_ENABLE`
2. 读取 `VLLM_UVM_POLICY_MODE`
3. 读取 `VLLM_UVM_POLICY_WARMUP_PREFETCH_MIN_BYTES`
4. 读取 `VLLM_UVM_POLICY_WARMUP_ADVISE_GPU`

然后把这些值写到 allocator trace header 中。

这样做的意义是：

1. 每份 allocator trace 都自带当前策略模式
2. 后续分析时不用猜这次运行到底开没开 policy

## 11. `TRACE_POLICY` 是怎么接入分配路径的

### 11.1 新增 `trace_policy_event(...)`

可参考：
[uvm_allocator.cpp](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/uvm_allocator.cpp#L414)

这个函数负责把一次 allocation 的策略判断写成一条结构化日志。

日志字段包括：

1. `alloc_id`
2. `ptr`
3. `end`
4. `size_bytes`
5. `size_bucket`
6. `device`
7. `phase`
8. `predicted_class`
9. `action`
10. `action_success`
11. `action_error`

### 11.2 为什么 `TRACE_POLICY` 要在 `uvm_malloc(...)` 中输出

可参考：
[uvm_allocator.cpp](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/uvm_allocator.cpp#L547)

在 `uvm_malloc(...)` 中的顺序是：

1. 先完成 `cudaMallocManaged`
2. 更新 allocator 统计
3. 捕获当前 `phase_snapshot`
4. 写 `TRACE_ALLOC`
5. 调用 `classify_allocation(...)`
6. 调用 `choose_policy_action(...)`
7. 如有必要执行动作
8. 写 `TRACE_POLICY`

这样设计的原因是：

1. `TRACE_POLICY` 需要真实的 `ptr/end`
2. 也需要真实的 `alloc_id`
3. 所以必须在 allocation 成功之后记录

## 12. 阶段 0 为什么默认不改变行为

阶段 0 的默认策略是：

1. `policy_enabled = 1`
2. `policy_mode = trace_only`

这意味着：

1. 会做分类
2. 会写 `TRACE_POLICY`
3. 但不会主动 prefetch
4. 也不会改为 `cudaMalloc`
5. 也不会进 scratch pool

这样做的原因是：

1. 先验证分类是否靠谱
2. 避免一开始就污染已有实验 baseline
3. 让阶段 0 成为纯观察层

## 13. `run_kv_fault_ratio.sh` 为什么也要改

阶段 0 不只是 allocator 的问题，还需要把策略配置传到真正的 vLLM server 进程里。

所以脚本必须负责：

1. 接收 CLI 参数
2. 校验参数
3. 拼成环境变量
4. 注入到 `uv run vllm serve ...` 这个 server 进程

## 14. `run_kv_fault_ratio.sh` 具体改了什么

### 14.1 新增脚本级默认变量

可参考：
[run_kv_fault_ratio.sh](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/run_kv_fault_ratio.sh#L35)

新增：

1. `UVM_POLICY_ENABLE`
2. `UVM_POLICY_MODE`
3. `UVM_POLICY_WARMUP_PREFETCH_MIN_BYTES`
4. `UVM_POLICY_WARMUP_ADVISE_GPU`

### 14.2 新增帮助信息

可参考：
[run_kv_fault_ratio.sh](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/run_kv_fault_ratio.sh#L86)

新增参数说明：

1. `--uvm-policy-enable`
2. `--uvm-policy-mode`
3. `--uvm-policy-warmup-prefetch-min-bytes`
4. `--uvm-policy-warmup-advise-gpu`

### 14.3 启动 server 时注入环境变量

可参考：
[run_kv_fault_ratio.sh](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/run_kv_fault_ratio.sh#L172)

这里新增了：

1. `policy_env`

然后把：

1. `VLLM_UVM_POLICY_ENABLE`
2. `VLLM_UVM_POLICY_MODE`
3. `VLLM_UVM_POLICY_WARMUP_PREFETCH_MIN_BYTES`
4. `VLLM_UVM_POLICY_WARMUP_ADVISE_GPU`

一起注入到 server 启动命令。

### 14.4 新增参数解析

可参考：
[run_kv_fault_ratio.sh](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/run_kv_fault_ratio.sh#L523)

新增 `case` 分支用于解析新的 CLI 参数。

### 14.5 新增参数校验

可参考：
[run_kv_fault_ratio.sh](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/run_kv_fault_ratio.sh#L601)

新增校验：

1. `--uvm-policy-enable` 只能是 `0/1`
2. `--uvm-policy-mode` 只能是：
   - `trace_only`
   - `prefetch`
   - `warmup_prefetch`
3. warmup prefetch threshold 必须是整数
4. advise 开关必须是 `0/1`

## 15. 阶段 0 的运行方式

阶段 0 的典型运行命令是：

```bash
./run_kv_fault_ratio.sh \
  --mode trace \
  --with-address-log \
  --trace-log /tmp/uvm_kv_fault_stats_stage0.log \
  --address-trace-log /tmp/uvm_kv_fault_addrs_stage0.log \
  --allocator-log /tmp/vllm_uvm_allocator_trace_stage0.log \
  --uvm-trace-min-bytes 1048576 \
  --uvm-policy-enable 1 \
  --uvm-policy-mode trace_only
```

运行后验证：

```bash
rg 'TRACE_POLICY' /tmp/vllm_uvm_allocator_trace_stage0.log | head
```

预期现象：

1. allocator trace 中出现 `TRACE_POLICY`
2. `predicted_class` 会出现：
   - `warmup_workspace`
   - `runtime_scratch`
   - `runtime_workspace`
   - `weight_persistent`
   - `kv_persistent`
3. `action` 默认是：
   - `managed_default`

## 16. 阶段 0 的成功标准

阶段 0 成功，不是看 fault 有没有下降，而是看：

1. 是否每个大 allocation 都能输出 `TRACE_POLICY`
2. 热点对象是否能被稳定归类
3. 默认模式下是否保持原有行为不变
4. allocator trace 是否能支持后续离线分析

也就是说，阶段 0 更像：

1. 在 allocator 中引入“可观测策略层”
2. 而不是立即做性能优化

## 17. 实现总结

这段修改的核心链路可以概括为：

1. 在 `uvm_allocator.cpp` 中新增：
   - `AllocationClass`
   - `PolicyAction`
   - `classify_allocation(...)`
   - `choose_policy_action(...)`
   - `trace_policy_event(...)`
2. 在 `uvm_malloc(...)` 中把分类和日志挂到真实 allocation 路径上
3. 在 `run_kv_fault_ratio.sh` 中把 policy 参数传给 server 进程
4. 默认保持 `trace_only`

最终得到的结果是：

1. 阶段 0 已经具备完整实现
2. 它可以把热点对象分成：
   - `warmup_workspace`
   - `runtime_scratch`
   - `runtime_workspace`
   - `weight_persistent`
   - `kv_persistent`
3. 并通过 `TRACE_POLICY` 把这些判断完整写到 allocator trace 中

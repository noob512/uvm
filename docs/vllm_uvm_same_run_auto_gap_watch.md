# vLLM UVM Same-Run Auto Gap Watch

## 1. 背景

在之前的工作流里，热点 `unknown gap` 的定位通常分两步：

1. 第一次运行 vLLM，收集 `/tmp/uvm_kv_fault_addrs.log`、`/tmp/vllm_uvm_address_regions.log`、`/tmp/vllm_uvm_allocator_trace.log`
2. 离线分析出本轮最热的 `gap#2`
3. 第二次重新启动 vLLM，再把上一次得到的 `gap#2` 地址塞给 `--uvm-gap-watch-start/end`

这个流程有一个根本问题：

1. `gap#2` 是“本轮地址空间中的热点空档”
2. 它不是跨进程稳定的固定对象
3. 每次重新启动 vLLM，GPU UVA 地址布局都可能变化
4. 因此第二次运行里常常根本打不中上一次的 `gap#2`

这正是之前反复出现：

1. `Top Unknown Gaps` 里的 `gap#2` 地址每次都变
2. `TRACE_GAP_WATCH_ALLOC` / `TRACE_GAP_WATCH_FREE` 为 0

的根本原因。

## 2. 方案评估

围绕“如何避免跨运行地址漂移”，实际有两种候选方案。

### 2.1 方案 A：继续两次运行，但加大离线分析

做法：

1. 第一次运行收集更完整日志
2. 运行后做更深的 offline correlation
3. 不再强求第二次命中同一个绝对地址

优点：

1. 实现简单
2. 不需要改 allocator 行为

缺点：

1. 只能回答“这轮发生了什么”
2. 不能保证下一轮还能盯住同一片热点地址
3. 对于想抓“当前这个 server 进程里真实是谁在复用这段地址”的目标，不够直接

### 2.2 方案 B：同一 vLLM 进程内 probe -> discover -> main

做法：

1. 启动一次 vLLM server
2. 先跑一个很小的 probe workload
3. 立刻分析这次 probe 产生的 fault/address 日志
4. 在同一个 server 进程还活着的时候，动态把当前真实 `gap#2` 下发给 allocator
5. 再继续跑 main workload

优点：

1. `probe` 和 `main` 共享同一个 vLLM 进程
2. 共享同一个 GPU UVA 地址空间
3. 共享同一个 allocator 地址复用池
4. 不再依赖“下一次新进程正好复现同一地址”
5. 观测目标从“跨运行同地址”变成“同进程同地址”

缺点：

1. 需要让 allocator 支持运行时更新 watch 配置
2. 需要 runner 具备两阶段 benchmark 能力

### 2.3 最终结论

本项目现在选择 **方案 B**，因为它更贴合真实目标：

1. 我们要减少或消除 `gap#2` 这种热点 unknown 区间
2. 这要求我们尽量在“同一个地址空间、同一个 allocator 复用历史”里抓现行
3. 因此最优解不是“跨运行猜它是否复现”，而是“同一运行内先发现，再继续 watch”

## 3. 本次实现的修改

本次修改涉及三个文件：

1. [uvm_allocator.cpp](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/uvm_allocator.cpp)
2. [run_kv_fault_ratio.sh](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/run_kv_fault_ratio.sh)
3. [discover_gap_watch.py](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/discover_gap_watch.py)

### 3.1 `uvm_allocator.cpp`：新增运行时热更新的 gap-watch 控制

新增环境变量：

1. `VLLM_UVM_GAP_WATCH_CONTROL_FILE`
2. `VLLM_UVM_GAP_WATCH_REFRESH_MS`

新增能力：

1. allocator 在初始化时读取 control file 路径
2. 后续在分配路径上按固定轮询间隔检查 control file
3. 如果 control file 的 `mtime/size` 变化，就重新读取配置
4. 新的配置会立即覆盖当前 `gap_watch_start/end/name/all_classes/min_bytes`

也就是说，`gap-watch` 不再只能在进程启动前通过环境变量一次性注入，而是可以在进程活着的时候动态改。

### 3.2 `uvm_allocator.cpp`：新增 `TRACE_GAP_WATCH_CONFIG`

新增日志事件：

1. `TRACE_GAP_WATCH_CONFIG`

作用：

1. 记录 control file 的配置被何时应用
2. 记录新的 watch range 是多少
3. 记录当前是否启用
4. 记录配置变更原因，例如：
   - `control_file`
   - `disabled`
   - `invalid_range`

这样可以直接在 allocator trace 中确认：

1. auto discovery 是否真的把新的 `gap#2` 地址下发成功
2. allocator 是否真的在同一进程里切换到了新的 watch 配置

### 3.3 `run_kv_fault_ratio.sh`：新增 same-run auto workflow

新增参数：

1. `--uvm-gap-watch-control-file`
2. `--uvm-gap-watch-refresh-ms`
3. `--auto-gap-watch-enable`
4. `--auto-gap-watch-probe-prompts`
5. `--auto-gap-watch-target-gap`
6. `--auto-gap-watch-fallback-to-hottest`
7. `--auto-gap-watch-summary-json`

新增行为：

1. 当 `--auto-gap-watch-enable 1` 时，脚本不再只跑一次 benchmark
2. 而是对同一个 server 进程连续跑两个 benchmark phase：
   - `probe`
   - `main`

具体流程：

1. 启动同一个 vLLM server
2. 初始化一个 disabled 的 gap-watch control file
3. 开启 fault/address/allocator trace
4. 先跑 `probe`
5. 调用 `discover_gap_watch.py` 解析当前这次运行自己的日志
6. 把当前真实 `gap#2` 写入 control file
7. allocator 热更新配置
8. 继续跑 `main`

### 3.4 `discover_gap_watch.py`：新增同轮 hot gap 发现器

这个脚本专门用于“当前这一轮”。

输入：

1. 当前 run 的 `/tmp/vllm_uvm_address_regions.log`
2. 当前 run 的 per-fault address log

输出：

1. gap-watch control file
2. 可选 JSON summary

它会：

1. 基于当前 pid 的 weight/KV concrete region 构造 gap
2. 只统计当前 run 中落入 gap 的 unknown fault
3. 优先选择 `--target-gap` 指定的 gap index
4. 如果该 gap 没有 fault，且 `--fallback-to-hottest 1`，则回退到 hottest unknown gap
5. 把最终选择写成 allocator 可热更新的控制文件

## 4. 控制文件格式

allocator 读取的是一个简单的 `key=value` 文件。

示例：

```text
enabled=1
name=round_auto_gap2
start=0x725e71180000
end=0x725e71ffffff
all_classes=1
min_bytes=4096
```

字段含义：

1. `enabled`
   - `1` 表示启用 watch
   - `0` 表示关闭 watch
2. `name`
   - 这次 watch 的标签
3. `start/end`
   - 当前同进程里要盯住的真实 gap 区间
4. `all_classes`
   - `1` 表示任何命中 watch range 的 allocation 都记录
   - `0` 表示只看 `unknown_managed`
5. `min_bytes`
   - 命中 watch range 且尺寸达到阈值的 allocation 才记录

## 5. 为什么这个方案比“先跑一轮再重跑一轮”更好

关键差异在于地址空间是否连续。

### 5.1 旧方案的问题

旧方案中：

1. 第一次运行分析出 `gap#2 = 0xAAA..0xBBB`
2. 第二次运行时 vLLM 是一个新进程
3. allocator 的地址池、warmup 时序、workspace 重用路径都可能变
4. 于是第二次运行里的热点 gap 可能已经变成 `0xCCC..0xDDD`

结果就是：

1. watch 功能本身是开着的
2. 但你盯的地址已经不是这一轮真正热的地址
3. 所以 `TRACE_GAP_WATCH_ALLOC/FREE` 为 0

### 5.2 新方案的关键优势

新方案中：

1. `probe` 和 `main` 在同一个 server 进程里完成
2. `probe` 阶段发现的 `gap#2` 属于当前活着的这个地址空间
3. `main` 阶段继续使用同一个 allocator/同一个 managed 地址池
4. 所以命中概率远高于“跨运行复用绝对地址”

这并不保证 `main` 一定 100% 只用 `probe` 发现的那片地址，但它消除了最主要的失配来源：

1. 新进程带来的地址空间漂移

## 6. 新的推荐工作流

### 6.1 一次性完成 probe + main + dynamic watch

命令示例：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm

./run_kv_fault_ratio.sh \
  --mode trace \
  --with-address-log \
  --trace-log /tmp/uvm_kv_fault_stats_auto.log \
  --address-trace-log /tmp/uvm_kv_fault_addrs_auto.log \
  --allocator-log /tmp/vllm_uvm_allocator_trace_auto.log \
  --uvm-trace-min-bytes 1048576 \
  --uvm-unknown-detail-enable 1 \
  --uvm-unknown-detail-min-bytes 4096 \
  --uvm-gap-watch-name same_run_gap2 \
  --uvm-gap-watch-all-classes 1 \
  --uvm-gap-watch-min-bytes 4096 \
  --auto-gap-watch-enable 1 \
  --auto-gap-watch-probe-prompts 1 \
  --auto-gap-watch-target-gap 2 \
  --prompts 20
```

这个命令会自动完成：

1. 启动 server
2. probe 一次
3. 解析本轮真实 `gap#2`
4. 把结果写入 control file
5. 同进程继续 main benchmark

### 6.2 关键输出文件

1. `/tmp/uvm_kv_fault_stats_auto.log`
   - 整个 run 的 replayable fault 统计
2. `/tmp/uvm_kv_fault_addrs_auto.log`
   - 整个 run 的逐条 fault 地址
3. `/tmp/vllm_uvm_allocator_trace_auto.log`
   - allocator trace，包括：
     - `TRACE_POLICY`
     - `TRACE_UNKNOWN_DETAIL`
     - `TRACE_GAP_WATCH_CONFIG`
     - `TRACE_GAP_WATCH_ALLOC`
     - `TRACE_GAP_WATCH_FREE`
4. `/tmp/vllm_auto_gap_watch_summary_<pid>.json`
   - probe 阶段 discovery 输出
5. `/tmp/vllm_uvm_gap_watch_control_<pid>.conf`
   - allocator 动态读取的控制文件

## 7. 如何确认同进程动态切换真的生效

先看 allocator trace 里是否出现：

```bash
rg -n "TRACE_GAP_WATCH_CONFIG|TRACE_GAP_WATCH_ALLOC|TRACE_GAP_WATCH_FREE" \
  /tmp/vllm_uvm_allocator_trace_auto.log | head -n 100
```

重点看：

1. 是否出现 `TRACE_GAP_WATCH_CONFIG`
2. 其中的 `start/end` 是否等于 probe 发现的当前真实 `gap#2`
3. 之后是否出现 `TRACE_GAP_WATCH_ALLOC/FREE`

如果这三步都成立，就说明：

1. discovery 正常
2. control file 正常写入
3. allocator 正常热更新
4. main 阶段的 allocation 确实打中了同一进程内发现的热点 gap

## 8. 与旧文档的关系

旧文档仍然有效：

1. [vllm_uvm_fault_address_classification.md](/home/ubuntu/nvidia-uvm-gpu/docs/vllm_uvm_fault_address_classification.md)
   - 解释 weight/KV/unknown 的基础分类逻辑
2. [vllm_uvm_unknown_gap_resolution.md](/home/ubuntu/nvidia-uvm-gpu/docs/vllm_uvm_unknown_gap_resolution.md)
   - 解释 allocator correlation、workspace 导出、focus-gap 等离线分析能力

但新的 same-run auto workflow 解决的是更具体的问题：

1. 如何避免跨运行的 `gap#2` 地址漂移
2. 如何在同一 server 进程里先发现热点，再继续详细 watch

## 9. 现阶段的边界

这个方案已经显著优于“跨运行盯旧地址”，但仍有边界：

1. `probe` 发现的是当前进程早期的热点 gap
2. 后续 `main` 期间如果工作负载特征变化很大，热点仍可能进一步迁移
3. allocator 只对走到 `cudaMallocManaged` 这条路径的对象可见
4. 如果某些库内部行为没经过当前 allocator trace，仍可能只能在 fault 侧看到、在 allocator 侧看不到

因此当前更合理的目标是：

1. 先把同进程内最热 unknown gap 的来源显著缩小
2. 再基于 `phase + size + lifetime + reuse pattern` 去决定下一阶段的优化策略

## 10. 总结

本次改造后，项目已经从：

1. “先跑一轮，下一轮猜它还在不在同一个地址”

升级为：

1. “在同一个 vLLM 进程里先 probe，立即发现当前真实 `gap#2`，再继续 main workload 并详细 watch”

这使得 `gap#2` 分析从“跨运行推测”变成了“同进程定点观测”，更适合后续真正去减少或消除热点 unknown gap。

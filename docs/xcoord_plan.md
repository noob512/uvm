# gpu_ext 项目状态分析 + 新论文方向：跨 CPU-GPU eBPF 资源协调

## Context

gpu_ext 是一个通过 eBPF struct_ops 扩展 NVIDIA GPU 驱动的系统。当前论文已投稿待审。用户希望在 gpu_ext 基础上产出一篇**全新的**系统顶会论文，方向选择：**跨 CPU-GPU 子系统的统一 eBPF 资源协调**（sched_ext + gpu_ext）。

---

## 〇、xCoord Policy 总览 + 进度 + 下一步

### 0.1 已实现的 Policy

xCoord 分 GPU 侧（gpu_ext struct_ops）和 CPU 侧（sched_ext struct_ops）两部分。

**CPU 侧 (sched_ext)**:

| Policy | 文件 | 方法 | 问题 |
|--------|------|------|------|
| **baseline** | `sched_gpu_baseline.bpf.c` | GPU 进程 (TGID) + UVM worker (PID) → GPU_BOOST_DSQ (FIFO, 40ms)；非 GPU → SHARED_DSQ (全局 PRIQ) | 盲目 boost = `nice -20`；全局 SHARED_DSQ 破坏 CFS per-CPU 局部性；FAISS +67-759%；vLLM -62%（boost 错 PID + 全局 DSQ 开销） |
| **minimal** | `sched_gpu_minimal.bpf.c` | GPU 进程 → SCX_DSQ_LOCAL (boost timeslice)；非 GPU → SCX_DSQ_LOCAL (default) | 无全局 DSQ 开销，但 GPU 任务也走 local DSQ → 无法保证优先级 |
| **serving** | `sched_gpu_serving.bpf.c` | GPU 进程 → GPU_BOOST_DSQ；非 GPU → SCX_DSQ_LOCAL | baseline 和 minimal 的折中：GPU 走 boost DSQ，非 GPU 走 local。适合少线程 serving，不适合 batch |
| **xcoord** | `sched_gpu_xcoord.bpf.c` | 读 `gpu_state_map.is_thrashing` auto-detect GPU 进程 + `-p` PID always boost；非 GPU → SHARED_DSQ | 无需 `-p PID` 即可 boost UVM 进程（state_boost=15K 验证通过）。A/B vs baseline: 非 UVM 场景性能相同。**关键发现**: 非 GPU 任务不能用 SCX_DSQ_LOCAL（local DSQ 优先于 custom DSQ dispatch） |
| **coord v1** | `sched_gpu_coord.bpf.c` | 四个声称的创新: (1) Proportional timeslice; (2) Backpressure throttling; (3) Per-thread precision; (4) Multi-tenant anti-thrashing: LC注册时throttle BE GPU进程 | **⚠️ 诚实评估**: 核心机制是 **DSQ 队列分离**（LC→GPU_BOOST_DSQ, BE→SHARED_DSQ），本质仍是 priority scheduling。**Bug**: `global_thrashing` 单向锁存（set=1, never cleared）→ 不是 closed-loop。**数据问题**: coord_run1 thrashing=0（anti-thrashing 未激活）; xcoord_run2 state_boost≈0（auto-detect 未激活）→ 两者结果几乎相同。**需替换为 FPRS (v2)** |
| **coord v2 (FPRS)** | `sched_gpu_coord.bpf.c` ✅ | **Fault-Pressure Regulated Scheduling**: 用 LC 的 `fault_rate` 作为 QoS 反馈信号，积分控制器连续调节 BE 的 CPU 分配。`regulate()` 每 100ms 读 `lc_pid_array` 中 LC 的 fault_rate → error → integral → be_throttle_pct (0-1000)。8 stats。**E10 v2 结果**: 控制器正确运行但未激活 — LC (20B) fits in VRAM, fault_rate=1-8 << target=100。**需 LC 也 UVM thrashing 的场景** |
| **xcoord-noad** | `sched_gpu_xcoord_noad.bpf.c` ✅ | xcoord 去除 auto-detect: 只 boost -p PID + UVM workers，不 boost gpu_state_map 中 auto-detected 的进程。隔离 "不 boost aggressor" 效果。4 stats | |

**共同问题（所有 policy 含 coord v1）**:
1. ~~从未在完整链路下测过~~ ✅ E5+ 用 `prefetch_always_max_xcoord`（有 shared maps）
2. **本质都是 priority scheduling**: boost GPU / throttle non-GPU = `nice -20` 变体。缺乏真正的算法创新
3. **无反馈控制**: 当前所有 policy 都是 open-loop（读状态→做决策，不关心决策效果）。coord v1 的 `global_thrashing` 是 bug（单向锁存），不是 closed-loop
4. **缺失关键 baseline**: 未测 "xcoord-no-autodetect"。E10 中 xcoord 差是因为 auto-detect boost 了 aggressor，不 boost 就和 coord 差不多

**GPU 侧 (gpu_ext)**:

| Policy | 文件 | 与 xCoord 关系 |
|--------|------|---------------|
| `prefetch_always_max_cycle_moe` | `prefetch_always_max_cycle_moe.bpf.c` | **无 xCoord 功能** — 只有 prefetch + eviction hooks，无 `track_uvm_worker()`、无 `gpu_state_map` |
| `eviction_lfu_xcoord` | `eviction_lfu_xcoord.bpf.c` | xCoord GPU 侧（LFU eviction）— 有 `track_uvm_worker()`, `gpu_state_map`, `uvm_worker_pids`；但 LFU eviction 导致 120B serving OOM |
| **`prefetch_always_max_xcoord`** | `prefetch_always_max_xcoord.bpf.c` | **推荐 xCoord GPU 侧** — always_max prefetch + cycle_moe eviction + 完整 xCoord shared maps。pp=222, tg=85（与无 xCoord 版持平）|

**当前推荐**: `prefetch_always_max_xcoord` — GPU 侧最优策略 + xCoord 跨子系统反馈

### 0.2 关键事实

- UVM GPU1 BH 是 **SCHED_OTHER kthread**（ppid=2），sched_ext **可以** boost 它
- `uvm_worker_pids` 机制设计正确：gpu_ext hook 中 `bpf_get_current_pid_tgid()` 返回 UVM BH 的 PID，sched_ext 用 `p->tgid` 匹配
- **但从未在完整链路下测过**（`eviction_lfu_xcoord` + `sched_gpu_baseline` 同时运行）
- nsys E1 profiling 中 UVM BH 零 sched events 原因：nsys 只追踪目标进程子进程（UVM BH ppid=2 非子进程）+ 无 gpu_ext → 无 fault → BH 在睡眠

### 0.3 过去的重复/浪费

1. **4 轮 "跑 baseline → 结果差 → 分析原因" 循环**: E1→E2→E3→E4 每次都是发现 baseline 无效然后分析原因，但从未切换到 `eviction_lfu_xcoord` 验证完整链路
2. **nsys profiling 误判**: 把 nsys 追踪范围限制导致的零 sched events 误判为 "BH/interrupt context 不可调度"，在此错误结论上花了多轮修正 plan
3. **无 stress profiling → 否定 → 再验证**: #34 和 #35 花时间分析无 stress 场景机会，但结论受 nsys 误判影响。#34 (vLLM P99) 结论仍成立（主线程 ON-CPU 99.8%），#35 结论需修正（UVM BH 可调度）
4. **plan 文档膨胀**: 2100+ 行，大量历史实验记录和被推翻的结论混在一起

### 0.4 下一步（按优先级）

| # | 任务 | 为什么重要 | 前置依赖 |
|---|------|----------|---------|
| **1** | ~~完整链路测试~~ ✅ 已完成 (2026-02-28) | `gpu_state_map` 有数据 (fault_rate/eviction_count), `uvm_worker_pids` 追踪 UVM BH, `gpu_boosted=5515`. 120B batch OK; **120B serving OOM**（新旧策略均 OOM，llama.cpp build 问题）| — |
| **2** | ~~创建 `prefetch_always_max_xcoord`~~ ✅ 已完成 (2026-02-28) | always_max prefetch + cycle_moe eviction + xCoord shared maps。性能: pp=222, tg=85（与原策略持平）。完整链路验证通过。| 无 |
| **3** | ~~E5: 120B UVM batch + stress~~ ✅ (2026-02-28) | **负面结果**: stress-ng 20c 仅 -1.9% pp 降低，tg 在噪声内。sched_ext 零效果 (pp=217.15 vs 217.85)。**120B UVM batch 是 GPU/PCIe 瓶颈，CPU 调度延迟被 GPU pipeline 掩盖** | #2 |
| **4** | ~~实现 `sched_gpu_xcoord.bpf.c`~~ ✅ (2026-02-28) | Auto-detect thrashing GPU 进程 (state_boost=15K)。A/B 测试: 非 UVM 场景与 baseline 性能相同。关键发现: SCX_DSQ_LOCAL 不能用于非 GPU 任务 | #2 |
| **5** | ~~vLLM 30B serving + stress~~ ✅ (2026-02-28) | **正面结果**: c=4: TPOT -7.4% (29.5→27.3ms), throughput +8.5% (130→141 tok/s); c=8: TPOT -7.7%, throughput +8.9%. 恢复 baseline 25-28%。sched_ext 对 CPU-bound serving 有效 | #2 |
| **6** | ~~多 workload 共存 (LC + BE)~~ 初步测试 (2026-02-28) | 20B serving(LC) + 120B batch(BE) 共享 GPU。**120B 模型加载/pp 阶段**：TPOT 10.7→32.4ms(3x), TTFT 113→342ms(3x)。sched_ext: TTFT -35%(342→223ms)。**120B steady-state(tg)**：几乎无降级(TPOT≈10.7ms)。干扰是 phase-dependent 的。技术问题：两个 CUDA UVM 进程共享时 20B 会 CUDA OOM crash；需 20B 先启动占 VRAM，120B 后启动用 UVM | #4 |
| **7** | ~~FAISS SIFT100M + stress~~ ✅ (2026-03-01) | **中性结果**: sched_gpu_xcoord 零开销 (add: 53.4→52.3s/-2%, search nprobe1: 5.1→4.7s/-8%, nprobe4/16: 不变)。之前 sched_gpu_baseline +67-759% 开销是因为无 gpu_ext（stock nvidia module, add 331s）。有 gpu_ext 时 FAISS 是 GPU/PCIe-bound (add 53s)，sched_ext 开销被掩盖。Auto-detect: state_boost=73K | #4 |
| **8** | ~~E7: coord vs xcoord A/B on vLLM 30B~~ ✅ (2026-03-01) | **中性**: 30B 无 UVM thrashing → coord 的 proportional/backpressure 未激活。xcoord avg: TPOT=75.5ms, tput=155.7 tok/s; coord avg: TPOT=74.3ms, tput=155.5 tok/s（噪声内）。coord stats: `uvm_worker=24K, thrash=0, throttled=0`。**结论**: coord novelty 只在 UVM thrashing 场景才能与 xcoord 区分 | #4 |
| **9** | ~~FAISS + stress + coord~~ ✅ (2026-03-01) | 1ms throttle **负面**: nprobe1 +15% 回归（dispatch 开销）。**修正**: 改为 5ms throttle → 3×3 A/B 测试 xcoord vs coord: **中性** (Add -0.7%, nprobe1 +0.5%, nprobe4 -0.4%, nprobe16 -0.3% — 全在噪声内)。coord backpressure 激活 (throttled=452K, uvm_worker=44K, thrashing=1) 但 FAISS 是 GPU/PCIe-bound，CPU cache 不是瓶颈 | #8 |
| **10** | ~~E10: Multi-tenant coord v1 vs xcoord~~ ✅ (2026-03-01) | **⚠️ 初步正面但有严重方法学问题**。20B(LC)+120B(BE), 3×3 runs。**表面数据**: coord vs xcoord: TPOT -20.5%, TTFT -57.1%, P99 TTFT -86.7%。**诚实审查**: (1) coord_run1 thrashing=0 → anti-thrashing 未激活，相当于单租户（TPOT=11.45）; (2) xcoord_run2 state_boost≈0 → auto-detect 未激活（TPOT=12.28 ≈ coord_run2 的 12.09）; (3) 顺序偏差（no_sched→xcoord→coord, 不是随机）; (4) 只 3 runs, 方差极大; (5) 核心机制只是 DSQ 队列分离 = priority scheduling; (6) `global_thrashing` 单向锁存 bug。**需 FPRS v2 + 重做实验** | #9 |
| **11** | ~~实现 FPRS (coord v2)~~ ✅ (2026-03-01) | `sched_gpu_coord.bpf.c` 完全重写: 积分控制器 + `lc_pid_array` + `regulate()` 函数 + 8 stats。构建通过 + 加载运行验证通过 (throttle=0%, regulate calls 正常递增)。| #10 |
| **12** | ~~添加 xcoord-no-autodetect baseline~~ ✅ (2026-03-01) | `sched_gpu_xcoord_noad.bpf.c`: 移除 Priority 3 (auto-detect)，仅 boost -p PID + UVM workers。Makefile 更新，构建通过 | #10 |
| **13** | ~~E10 v2: 4 conditions × 5 runs (interleaved)~~ ✅ (2026-03-01) | **FPRS 未激活**: LC (20B) fault_rate=1-8/sec << target=100 (fits in VRAM)。所有条件呈双峰分布 (TPOT ~11ms vs ~17-28ms)。详见 §0.6 | #11, #12 |
| **14** | ~~E11: FPRS 激活场景~~ ✅ (2026-03-01) | **120B+120B OOM 崩溃** (120GB > 125GB RAM)。**改用 20B serving + FAISS SIFT100M**: FAISS 导致 20B fault_rate 从 52 → 3828 (VRAM 竞争确认)。连续流量测试: FPRS -4.8% mean / -4.9% P99 / +5.3% throughput。详见 §0.8 | #13 |
| **15** | FPRS 调优 + 更大规模实验 | 当前 FPRS 改善 ~5%。需要: (1) 控制器调优 (decay/demotion) (2) 多轮正式实验 (3) 分析 FPRS 的改善来自 UVM worker 优先还是 BE throttle | #14 |
| **16** | 实现 dual-actuator QoS eviction (`prefetch_always_max_qos`) | **核心算法改进**: 用 GPU eviction bias（直接 VRAM 控制）替代 CPU timeslice throttle（间接、弱效果）。LC fault_rate > target → 保护 LC pages 不被驱逐。GPU-bound BE 场景下效果远强于 CPU throttle | #14 |
| **17** | E12: 20B+FAISS dual-actuator 实验 | 测试 QoS eviction 是否能让 LC 完全避免 thrashing (fault_rate→0)，预期从 2x 延迟恢复到 ~1x | #16 |

**原则**: 不同算法放在不同文件里（`sched_gpu_baseline` / `sched_gpu_xcoord` / ...），不修改已有的 baseline 代码。每个 policy 都是独立可对比的。

### 0.5 算法创新：从 Priority Scheduling 到 Feedback Control

#### 诚实评估：为什么 coord v1 不算 novelty

| coord v1 声称的创新 | 实际情况 | 等效于 |
|---------------------|----------|--------|
| Proportional timeslice | 多租户下 LC=80ms, BE=5ms，proportional 逻辑对结果无影响 | 大小 timeslice 的 priority scheduling |
| Backpressure throttling | 非 GPU 已在 SHARED_DSQ（低优先级），throttle slice 是次要效果 | `nice` 值调整 |
| Per-thread precision | xcoord 也区分 level 1/2，coord 只是把 level 2 从 boost 改为 throttle | 改变了 auto-detect 策略的方向 |
| Multi-tenant anti-thrashing | **核心机制是 DSQ 队列分离**（LC→GPU_BOOST_DSQ, BE→SHARED_DSQ） | Static priority with GPU-triggered mode switch |
| Closed-loop feedback | **Bug**: `global_thrashing` 单向锁存，永不清零，不是 closed-loop | One-way latch (broken) |

**关键证据**: xcoord_run2（auto-detect 未激活, state_boost≈0）: TPOT=12.28ms ≈ coord_run2（anti-thrashing 激活）: TPOT=12.09ms。说明 coord 的改善主要来自"不 boost aggressor"而非其他 novelty。

#### 核心问题：CPU 调度对 GPU 内存的影响链

```
CPU 调度 BE 线程 ──→ BE 提交 GPU kernel ──→ GPU 执行触碰新页面
        │                                          │
        │                                    UVM page fault
        │                                          │
        │                                   驱逐 LC 的 GPU 页面
        │                                          │
        └──── 延迟 Δ (ms级) ────────────────→ LC 下次 kernel 卡住
```

**独特之处**: CPU 调度决策对 GPU 内存压力有**延迟的、间接的**因果关系。传统 CPU 调度效果是即时的（A 用 CPU 则 B 不能用），这里是跨子系统的延迟因果链。

**Priority scheduling 为什么不够**: 它是 open-loop——做决策后不关心效果。高优先级=永久高优先级，低优先级=永久低优先级。不会根据实际 GPU 内存压力的变化做调整。

#### FPRS: Fault-Pressure Regulated Scheduling（故障压力调节调度）

**核心思想**: 把 LC 的 `fault_rate` 当作 **QoS 指标**，用积分控制器（integral controller）**连续调节** BE 的 CPU 分配。不是二值的 boost/throttle，而是 feedback control。

##### 与 Priority Scheduling 的本质区别

| | Priority Scheduling (xcoord/coord v1) | FPRS |
|---|---|---|
| **决策依据** | "是不是 GPU 进程" (binary classification) | "LC 的 fault_rate 偏离 target 多少" (continuous error signal) |
| **控制方式** | 高/低优先级队列 (open-loop) | 积分控制器 + 衰减 (closed-loop) |
| **适应性** | 固定策略，thrashing→永久throttle(bug) | 自动适应: LC 压力↓ → BE 恢复, LC 压力↑ → BE 收紧 |
| **BE 的待遇** | 全 throttle 或全 boost (binary) | 0%→100% 连续调节 (continuous) |
| **目标函数** | 无 | 显式: `minimize(lc_fault_rate - target)` |
| **稳态行为** | coord v1: 永久 throttle (一旦触发) | 自动收敛到 LC QoS 和 BE throughput 的最优平衡 |
| **多 BE 公平性** | 统一 throttle | 可扩展为 per-BE fault attribution |

##### 算法设计

**BPF 全局状态**:
```c
// 积分控制器状态（BPF global variables）
volatile u64 pressure_integral = 0;    // 误差积分 (scaled ×1000)
volatile u32 be_throttle_pct = 0;      // 0-1000, 当前 throttle 比例
volatile u64 last_regulate_ns = 0;     // 上次调节时间戳
```

**Rodata 参数**:
```c
const volatile u64 target_lc_fault_rate = 100;     // LC 目标 fault rate (faults/sec)
const volatile u64 regulate_interval_ns = 100000000; // 100ms 调节周期
const volatile u64 ki_gain = 1;                     // 积分增益
const volatile u32 decay_shift = 2;                 // 恢复速度 (>>2 = ÷4 per interval)
const volatile u64 max_integral = 100000;           // anti-windup 上限
const volatile u64 min_be_slice_ns = 1000000;       // BE 最小 1ms slice
```

**调节步骤** (在 BE 任务 enqueue 时 lazy 触发):
```
REGULATE():
  now = bpf_ktime_get_ns()
  if (now - last_regulate_ns < regulate_interval_ns) return
  last_regulate_ns = now

  // 1. SENSOR: 读 LC 的 GPU 内存压力
  for each lc_pid in gpu_process_pids:
      lc_state = gpu_state_map.lookup(lc_pid)
      lc_fr = max(lc_fr, lc_state->fault_rate)

  // 2. ERROR SIGNAL: LC 实际 fault rate vs 目标
  if lc_fr > target_lc_fault_rate:
      error = lc_fr - target_lc_fault_rate
      pressure_integral += error * ki_gain      // 积累误差
      pressure_integral = min(pressure_integral, max_integral)  // anti-windup
  else:
      // LC 没问题 → DECAY: 让 BE 逐步恢复
      pressure_integral >>= decay_shift         // ÷4 per interval (~400ms 半衰期)

  // 3. ACTUATOR: 转换为 throttle 百分比
  be_throttle_pct = min(pressure_integral * 1000 / max_integral, 1000)
```

**Enqueue 决策**:
```
LC (level 3, -p PID):
    GPU_BOOST_DSQ, max_gpu_slice_ns              // 始终最高优先级

UVM BH worker (level 1):
    GPU_BOOST_DSQ, proportional_slice             // 解决已有 fault (帮助所有人)

BE GPU (level 2, auto-detected):
    be_slice = max_slice * (1000 - be_throttle_pct) / 1000
    be_slice = max(be_slice, min_be_slice_ns)
    if be_throttle_pct > 500:
        SHARED_DSQ, be_slice                      // 高压: 降优先级 + 缩 timeslice
    else:
        GPU_BOOST_DSQ, be_slice                   // 低压: 正常优先级, 受限 timeslice

Non-GPU:
    if be_throttle_pct > 800:
        SHARED_DSQ, throttle_slice_ns             // 重度 backpressure
    else:
        SHARED_DSQ, SCX_SLICE_DFL                 // 正常
```

##### 为什么这是真正的 Novelty

1. **Feedback Control in CPU Scheduling via GPU Signals**:
   - 从未有人用 GPU fault rate 作为 CPU 调度器的控制信号
   - 传统调度器只看 CPU 指标（runqueue length, load average, I/O wait）
   - 这是**跨子系统反馈控制**，不是跨子系统 priority hint

2. **Integral Controller with Delay Awareness**:
   - CPU→GPU 因果链有 ms 级延迟
   - Proportional controller 会振荡（调太多→LC 无 fault→放开→又 fault→...）
   - Integral controller 累积误差 + 衰减，天然适合有延迟的系统
   - `decay_shift` 控制恢复速度，避免 windup

3. **Explicit QoS Objective**:
   - Priority scheduling 没有目标函数。"高优先级"不回答"高到什么程度？"
   - FPRS 有明确优化目标: `minimize |lc_fault_rate - target|`
   - 可以配置 `target_lc_fault_rate` 来 tradeoff LC latency vs BE throughput

4. **Graceful Degradation / Pareto-Optimal**:
   - BE 不是被"杀死"——而是获得在维持 LC QoS 前提下的**最大可能 CPU 时间**
   - 当 LC 压力小时（fault_rate < target），integral 衰减 → BE 几乎不受影响
   - 自动找到 Pareto-optimal: LC QoS 满足 + BE throughput 最大化

5. **Control-Theoretic Foundation**:
   - 可以用控制理论分析稳定性（integral gain vs delay → stability margin）
   - 可以证明收敛性（bounded integral + decay → bounded steady-state error）
   - 这给论文提供了理论框架，不只是 heuristic

##### 可选扩展（进一步 novelty，按优先级）

| 扩展 | 描述 | 难度 | 价值 |
|------|------|------|------|
| Per-BE fault attribution | gpu_ext 追踪哪个 PID 导致 LC eviction → per-BE 独立调节 (polluter pays) | 高 | ★★★ |
| Phase-aware target | 检测 LLM prefill/decode 阶段 → 不同阶段不同 target_lc_fault_rate | 中 | ★★ |
| PI controller | 加入 proportional term (P+I) 加速响应 | 低 | ★ |
| Multi-LC weighted QoS | 多 LC 进程时 weighted max/sum 作 QoS 信号 | 低 | ★ |

##### 实验设计改进

| 改进 | 原因 |
|------|------|
| **交错/随机** condition ordering | 消除顺序偏差 |
| **5+ runs** per condition | 统计显著性（当前 3 runs 方差太大） |
| **等待 120B 稳态** 再开始 benchmark | 消除 loading phase 随机性（coord_run1 thrashing=0 问题） |
| **添加 xcoord-no-autodetect baseline** | 隔离 "不 boost aggressor" 的效果 vs FPRS 的效果 |
| **4 conditions**: no_sched / xcoord-no-autodetect / xcoord / FPRS | 完整对照 |
| **报告 scheduler stats** per run | 验证每个 run 的机制是否真正激活 |

### 0.6 E10 v2 结果：FPRS 未激活（2026-03-01）

**实验**: 4 conditions × 5 runs = 20 runs (interleaved)，20B(LC) + 120B(BE)

#### 原始数据

| Policy | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Median TPOT |
|--------|-------|-------|-------|-------|-------|-------------|
| no_sched | 18.34 | 11.56 | 11.14 | 16.82 | 17.32 | 16.82 |
| xcoord | 22.03 | 12.06 | 11.90 | 11.55 | 22.78 | 12.06 |
| xcoord_noad | 11.36 | 12.06 | 28.02 | 11.72 | 19.92 | 12.06 |
| fprs | 18.65 | 12.12 | 12.22 | 11.70 | 19.41 | 12.22 |

#### 关键发现

1. **FPRS 控制器正确运行但未激活**: throttle=0%, be_reg=0, integral=0 in ALL 5 runs
   - LC (20B) fault_rate = 1-8/sec，远低于 target=100
   - 原因: 20B (~10GB) fits entirely in 32GB VRAM，UVM 不会驱逐 LC 页面
   - FPRS 的逻辑是**正确的**: LC 无 GPU 压力 → 不 throttle BE → 行为 ≈ no_sched

2. **双峰分布**: 所有条件都呈 ~11ms (好) vs ~17-28ms (坏) 双峰分布
   - 这不是调度器效果，而是 120B thrashing 强度的随机变化
   - 坏 runs 中 120B 恰好在 serving benchmark 期间密集使用 GPU → GPU 竞争

3. **xcoord auto-detect 不一致**: state_boost 在 runs 间差异巨大 (0-13K)
   - run 2,4: state_boost=9641/13113 (120B 被 auto-detect 并 boost)
   - run 1,3,5: state_boost=0-38 (120B 未被 auto-detect)

4. **sched_ext 对此 workload mix 效果有限**: 中位数 TPOT 在所有条件间相近 (12.06-16.82ms)
   - 干扰主要是 GPU/PCIe 侧（120B 占用 GPU compute + PCIe bandwidth）
   - CPU 调度不是瓶颈: 20B serving 的 CPU 占用很低

#### 根因分析

```
20B (LC): VRAM-resident (10GB < 32GB)
    → fault_rate ≈ 0   ← FPRS 控制信号为零
    → UVM 不驱逐 LC 页面（cudaMalloc via UVM 但 fits in VRAM → 无 migration）

120B (BE): UVM thrashing (60GB >> 32GB)
    → 自己的页面在 VRAM 的剩余 22GB 中来回 swap
    → 干扰 LC 的方式: 占用 GPU compute 时间 + PCIe 带宽
    → 不是通过驱逐 LC 的页面来干扰 LC

结论: 干扰路径是 GPU compute contention, 不是 VRAM page contention
      → fault_rate 不是正确的 QoS 信号 for this workload mix
```

#### Scheduler 日志摘要

| Policy | lc (pid_boost) | be_reg | state_boost | throttle | lc_fr |
|--------|---------------|--------|-------------|----------|-------|
| xcoord | ~40K | n/a | 0-13K (不稳定) | n/a | n/a |
| xcoord_noad | ~40K | n/a | n/a (disabled) | n/a | n/a |
| fprs | ~37K (lc) | **0** (未激活) | n/a | **0%** | 1-8 |

### 0.7 E11 尝试与失败：120B+120B OOM 崩溃 (2026-03-01)

**方案 A1 (120B+120B)**: 两个 120B UVM 进程 = 120GB system RAM > 125GB total → **OOM 崩溃系统两次**。
- 两次死机需硬重启
- 结论: 不可行，必须换 workload

### 0.8 E11 实际结果：20B serving + FAISS SIFT100M (2026-03-01)

#### 场景设计

| 角色 | Workload | VRAM | System RAM | UVM |
|------|----------|------|-----------|-----|
| LC | 20B llama-server (UVM, c=65536) | ~13 GB | ~12 GB | ✅ |
| BE | FAISS SIFT100M IVF4096_Flat (-uvm) | ~50 GB index | ~50 GB | ✅ |
| **合计** | | **~63 GB >> 32 GB VRAM** | **~62 GB << 125 GB RAM** | ✅ |

**关键发现**: FAISS 的 index build + search 导致 20B 的 fault_rate 从 52 → **3828** (is_thrashing=1)。
FAISS 驱逐了 20B 的 VRAM 页面 → **FPRS 控制信号确认可用！**

#### 干扰量化 (单次请求测试)

| Phase | 20B 延迟 (64 tok) | 20B fault_rate | 说明 |
|-------|-------------------|----------------|------|
| Baseline (无 FAISS) | 0.23s | 52 | 正常 |
| FAISS add 阶段 (首次请求) | **2.5s** (10x) | → 3828 | VRAM 被 FAISS 占据，20B 大量 page fault |
| FAISS add 阶段 (后续请求) | 0.44s (2x) | ~1446 | 页面 warm up 后有所恢复 |
| FAISS 结束后 | 0.12s (恢复) | 衰减中 | 页面逐步回到 VRAM |

#### FPRS Bug 修复 (3 个)

| Bug | 症状 | 修复 |
|-----|------|------|
| **Non-GPU backpressure** | throttle=100% 时非 GPU 任务 (网络/系统服务) 也被 throttle → LC 首次请求 **45 秒** | 移除 non-GPU backpressure：只 throttle gpu_state_map 中的 BE 进程 |
| **gpu_state_map fault_rate 不衰减** | 进程空闲时 fault_rate 保持最后值（如 843），FPRS 误判为持续压力 → 永远 throttle=100% | 在 regulate() 中加 staleness check：`if (now - last_update_ns > 2s)` 则 treat fault_rate=0 |
| **max_integral 过大 + ki_gain 过低** | 100000 max + ki_gain=1 → 从 0% 到 50% 需 ~50s（FAISS 早已结束）| max_integral: 100K→10K, ki_gain: 1→10 → 响应时间 ~500ms |

#### 连续流量测试结果 (60s continuous traffic)

| 指标 | Baseline (无 FAISS) | FAISS (无 FPRS) | FAISS + FPRS | FPRS 改善 |
|------|------|------|------|------|
| **Mean 延迟** | 0.234s | 0.456s | **0.434s** | **-4.8%** |
| **P50 延迟** | 0.233s | 0.435s | **0.425s** | **-2.3%** |
| **P99 延迟** | 0.235s | 0.844s | **0.803s** | **-4.9%** |
| **Throughput** | 4.3 req/s | 2.2 req/s | **2.3 req/s** | **+5.3%** |
| **总请求数** | 129 | 132 | **139** | +5.3% |

#### FPRS 控制器行为分析

```
控制器统计:
  throttle 峰值: 100% (多次达到)
  lc_fr 峰值: 1713
  UVM worker boosts: 57,068 (大量 boost)
  BE regulated: 12,176 (FAISS 被调节)
  BE demoted: 0 (throttle 未持续 >50%，未降级到 SHARED_DSQ)
  backpressure: 0 (已移除 non-GPU backpressure)

问题: 控制器振荡
  请求期间: lc_fr > 200 → integral 累积 → throttle ↑
  请求间隙: lc_fr = 0 (staleness) → integral 衰减 → throttle ↓
  → throttle 在 0% 和 100% 之间快速振荡，无法维持稳态
```

#### 为什么只有 ~5% 改善

1. **FAISS 是 GPU-bound**: CPU throttle 对 GPU-bound workload 效果有限。FAISS 线程大部分时间在等 GPU，给它更少 CPU 不改变 GPU 使用率
2. **控制器振荡**: decay_shift=1 (÷2 per 100ms) 导致 throttle 在请求间迅速衰减，无法维持持续压制
3. **大部分改善来自 UVM worker 优先**: 57K UVM worker boosts 让 20B 的 page fault 更快解决。这是 "priority scheduling"，不是 FPRS 的 feedback control novelty
4. **be_demoted=0**: FAISS 从未被降级到 SHARED_DSQ，throttle 只影响了 timeslice 而非 DSQ 优先级

#### 待解决问题

1. **控制器调优**: decay_shift 需在 "快速恢复" 和 "持续压制" 之间找到平衡
2. **CPU throttle 对 GPU-bound BE 效果弱**: 可能需要更直接的 GPU 资源控制（如 GPU timeslice、compute partition）
3. **FPRS novelty vs UVM worker priority**: 需隔离两者效果——关闭 UVM worker boost 只看 BE throttle 的贡献
4. **更长时间的干扰**: FAISS 全流程只 ~50s，改用更持久的 BE workload (如 FAISS 循环运行) 让 FPRS 有更长的稳态窗口

### 0.9 算法改进：Dual-Actuator QoS Eviction (2026-03-01)

#### 核心问题：为什么 FPRS (CPU throttle) 只有 ~5%

FPRS 用 CPU timeslice 作为唯一执行器 (actuator)，因果链如下：

```
CPU throttle BE → BE 提交更少 GPU kernel → 更少 GPU page fault → LC 页面少被驱逐
       ^                                                               |
       |          延迟 ms 级，且 GPU-bound BE 上 CPU throttle ≈ 无效        |
       └──── SENSOR: LC fault_rate ← gpu_state_map ──────────────────────┘
```

**根本原因**: FAISS 是 GPU-bound workload。CPU throttle 不改变 FAISS 的 GPU kernel 提交速率（FAISS 线程大部分时间在等 GPU completion），因此对 GPU 内存压力的影响极弱。这不是调参问题，而是**执行器选错了**。

#### 解决方案：GPU eviction bias 作为直接执行器

```
SENSOR:     LC fault_rate (from gpu_state_map, updated in chunk_activate)
CONTROLLER: 积分控制器 (同 FPRS 逻辑，在 gpu_ext 中运行)
ACTUATOR:   eviction_bias → chunk_used 保护 LC pages (直接 VRAM 控制)
            ↓
            eviction_bias > 0 && chunk belongs to LC → unconditionally move_tail (protect)
            BE chunks → normal cycle_moe (可能被驱逐)
```

**关键区别**: 执行器直接控制 GPU VRAM 分配（谁的页面被驱逐），无需通过 CPU→GPU 的间接因果链。

#### 算法设计

**新文件**: `prefetch_always_max_qos.bpf.c`，基于 `prefetch_always_max_xcoord.bpf.c`，增加：

**新增 BPF 状态**:
```c
// LC PID 注册 (从 loader 通过 map 传入)
struct { ARRAY, max_entries=16 } lc_pid_array;   // LC 进程 PID 列表
const volatile u32 n_lc_pids = 0;                // LC PID 数量

// 积分控制器状态
volatile u64 eviction_integral = 0;              // 误差积分 (scaled)
volatile u32 eviction_bias = 0;                  // 0-1000, 当前保护强度
volatile u64 last_regulate_ns = 0;               // 上次调节时间
volatile u64 lc_fault_rate_observed = 0;         // 最近一次 LC fault_rate

// Rodata 参数
const volatile u64 target_fault_rate = 200;      // LC 目标 fault_rate
const volatile u64 ki_gain = 10;
const volatile u32 decay_shift = 2;              // ÷4 per interval
const volatile u64 max_integral = 10000;
const volatile u64 regulate_interval_ns = 100000000; // 100ms
```

**`is_lc_process(pid)`**: 线性扫描 `lc_pid_array`（最多 16 个，BPF verifier 可展开）。

**`regulate_eviction()`**: 在 `chunk_activate` (每次 page fault) 中 lazy 调用：
```
regulate_eviction():
  if (now - last_regulate_ns < regulate_interval_ns) return
  last_regulate_ns = now

  // 1. SENSOR: 读 LC 的最大 fault_rate
  max_lc_fr = 0
  for i in 0..n_lc_pids:
      state = gpu_state_map.lookup(lc_pid_array[i])
      if state && (now - state->last_update_ns < 2s):
          max_lc_fr = max(max_lc_fr, state->fault_rate)

  lc_fault_rate_observed = max_lc_fr

  // 2. CONTROLLER: 同 FPRS 积分逻辑
  if max_lc_fr > target_fault_rate:
      error = max_lc_fr - target_fault_rate
      eviction_integral += error * ki_gain / 1000
      eviction_integral = min(eviction_integral, max_integral)
  else:
      eviction_integral >>= decay_shift  // 衰减

  // 3. ACTUATOR: 转为 eviction bias
  eviction_bias = min(eviction_integral * 1000 / max_integral, 1000)
```

**核心修改: `gpu_block_access` (chunk_used)**:
```c
// QoS eviction protection: LC pages protected when under pressure
if (eviction_bias > 0) {
    u32 owner_pid = get_owner_pid_from_chunk(chunk);
    if (owner_pid && is_lc_process(owner_pid)) {
        bpf_gpu_block_move_tail(chunk, list);
        stat_inc(STAT_LC_PROTECTED);
        return 1; // BYPASS: LC protected
    }
}

// Normal cycle_moe T1 frequency protection for everything else
... (existing code)
```

#### 为什么这个算法有效

1. **20B+FAISS 场景**: LC (20B) WS ≈ 13GB, VRAM = 32GB, FAISS WS ≈ 50GB
   - 当 eviction_bias > 0: LC 的 13GB 全部 move_tail（LRU 尾部 = 最不容易被驱逐）
   - BE (FAISS) 的 chunks 走 normal cycle_moe → 更容易被驱逐
   - **预期效果**: LC fault_rate 从 3828 → ~0（LC 页面几乎不被驱逐）
   - **预期延迟**: 从 2x baseline → ~1x baseline（完全恢复）

2. **反馈控制保证公平**:
   - 当 LC fault_rate < target: eviction_bias 衰减 → BE 页面恢复正常保护
   - 不会永久 "惩罚" BE — 只在 LC 受压时激活

3. **直接 vs 间接**:
   - CPU throttle: 间接（CPU→GPU→VRAM），GPU-bound 时因果链断裂
   - Eviction bias: 直接控制 VRAM LRU 位置，因果链长度=0

#### E12 首次实验失败分析 (2026-03-01)

**实验日志关键数据**:
```
activate=1,825,329  used=0  t1=0  lc_prot=0  evict=374,341
bias=0%  integral=0  lc_fr=0  regulate=1,290
```

**3 个问题**:

| 问题 | 原因 | 修复 |
|------|------|------|
| `used=0` (gpu_block_access 从不触发) | `mark_root_chunk_used()` 只在 `block_set_resident_processor()` 中调用，条件：block_size == UVM_CHUNK_SIZE_MAX **且** resident bit 0→1（首次 resident）。已 resident 的页面不再触发。 | **保护逻辑移到 `chunk_activate`**（每次 page fault 必触发，activate=1.8M 验证） |
| `lc_fr=0` (LC 无 fault) | 20B server 在 FAISS 期间被 CUDA OOM 杀死（VRAM 耗尽）。20B 死后不再 fault | 需要更 robust 的启动顺序，或减小 FAISS dataset |
| 算法设计错误 | 把 LC 保护放在 gpu_block_access（不触发），不是 chunk_activate（触发） | 重写为 chunk_activate-based eviction bias |

**修正后的算法**: 在 `chunk_activate` (page fault) 中直接控制 LRU 位置：
- `eviction_bias > 0` + BE chunk → `bpf_gpu_block_move_head` (LRU 头 = 先驱逐)
- `eviction_bias > 0` + LC chunk → `bpf_gpu_block_move_tail` (LRU 尾 = 后驱逐)
- `eviction_bias == 0` → return 0 (kernel default: tail)

这个方案的正确性：
- BE chunk 一旦 fault in 就被放到 LRU 头部 → 下次驱逐首选 → BE 页面在 VRAM 中 cycle 极快
- LC chunk 被放到 LRU 尾部 → 最后才被驱逐 → LC 页面持久驻留
- 反馈控制保证 `eviction_bias` 在 LC 无压力时衰减 → BE 恢复正常

**避免重复错误的原则**: 写代码前先验证 hook 确实会被调用（检查 stat counter 非零）。

#### 与 FPRS 的关系

Dual-actuator = eviction bias (GPU 侧, 直接) + CPU priority (CPU 侧, 互补)
- GPU 侧: `prefetch_always_max_qos` — eviction bias 消除 LC thrashing
- CPU 侧: `sched_gpu_xcoord` — UVM worker priority 加速 fault 处理

两者可以独立运行，也可以组合。E12 先测 GPU 侧独立效果。

---

## 一、当前 gpu_ext 状态总结

### 可用资产

| 资产 | 状态 | 可直接复用 |
|------|------|-----------|
| NVIDIA UVM 内核模块 + 5 个 BPF hook | 可用 | 是 — 作为 GPU 侧基础设施 |
| 20 个 eBPF 策略（eviction/prefetch/scheduling） | 部分可用 | 是 — 作为策略库 |
| 4 个 Workload benchmark（llama.cpp/vLLM/PyTorch/FAISS） | 脚本就绪 | 是 — 作为评估基础 |
| GPU 调度 hooks（timeslice/preemption） | 原型 | 需扩展 |
| Device-side eBPF（SIMT verifier） | 主要是设计 | 不直接复用 |

### 关键限制

- ~~API 完整性 2/10~~ → **已解决**: BPF CO-RE 可读取所有 chunk 属性，实际完整性 7/10
- ~~唯一需要修复: depopulate hook 条件拦截 (5 LOC)~~ → **已解决**: 不存在此问题，元数据用 LRU_HASH 自动清理
- **内核修改: 0 LOC** ✅
- 实验 ~20% 完成 — 新论文需要全新的实验设计
- GPU 调度仅 timeslice 控制 — 需深化
- **新增**: CPU-GPU 耦合已有实验数据支撑（19-26% 性能下降，详见 Section 九）

---

## 二、新论文方向：跨 CPU-GPU eBPF 资源协调

### 2.1 核心洞察（Key Insight）

**GPU workload 性能同时取决于 CPU 和 GPU 两侧的资源管理决策，但现有系统将两者独立管理。**

具体体现：
1. **UVM page fault 处理依赖 CPU**: GPU 触发 page fault → CPU interrupt → CPU worker thread 处理 → PCIe 数据传输。CPU 调度直接影响 fault latency（host-side overhead 是传输时间的 7x）。
2. **GPU kernel launch 依赖 CPU thread**: CPU 线程被降调度 → GPU kernel launch 延迟 → tail latency 增加。
3. **Multi-tenant 竞争跨越 CPU-GPU**: LC inference tenant 的 GPU 内存被 BE training tenant 的 page fault 驱逐，同时两者的 CPU 线程也在争抢 CPU 时间。
4. **PCIe 带宽是共享资源**: CPU prefetch 和 GPU eviction 都用 PCIe，但两侧策略互不知情。

**sched_ext（Linux 6.12+）和 gpu_ext 分别用 BPF struct_ops 管理 CPU 和 GPU，但两者之间没有协调。** 这是一个明确的系统缺口。

### 2.2 论文定位

**Title**: `xCoord: Coordinated CPU-GPU Resource Management via Cross-Subsystem eBPF`

**一句话 claim** (待验证): GPU 内存子系统的运行时状态（page fault rate、eviction pressure）是 CPU 调度决策的关键缺失信号；通过 eBPF 跨子系统共享这些信号，xCoord 可以实现盲目优先级提升（blind boost）和静态 CPU 隔离（taskset）无法达到的性能保障。具体数值待 xCoord 调度器实现后验证。

**与 gpu_ext 论文的区别**:
- gpu_ext: "GPU driver as programmable OS subsystem"（单子系统可编程性）
- **xCoord**: "GPU memory awareness as first-class CPU scheduling signal"（跨子系统协调）
- gpu_ext 专注 GPU 内部策略；xCoord 专注 CPU↔GPU 策略联动
- gpu_ext 的评估是 "不同策略对比"；xCoord 的评估是 "协调 vs 独立 vs 朴素隔离"

### 2.3 系统架构

```
┌─────────────────────────────────────────────┐
│              User-space Control Plane         │
│   Policy Loader + Map Configuration + SLO    │
└────────────┬──────────────────┬──────────────┘
             │                  │
    ┌────────▼────────┐ ┌──────▼────────┐
    │   sched_ext BPF  │ │  gpu_ext BPF   │
    │                  │ │                │
    │ select_cpu()     │ │ chunk_activate │
    │ enqueue()        │ │ chunk_used     │
    │ dispatch()       │ │ evict_prepare  │
    │ running()        │ │ prefetch       │
    │ stopping()       │ │ sched_init     │
    └────────┬────────┘ └──────┬────────┘
             │                  │
      ┌──────▼──────────────────▼──────┐
      │        Shared BPF Maps          │
      │  (pinned in /sys/fs/bpf/)       │
      │                                 │
      │  gpu_state_map:                 │
      │    pid → {fault_rate, mem_usage,│
      │           eviction_count,       │
      │           prefetch_pending}     │
      │                                 │
      │  cpu_state_map:                 │
      │    pid → {cpu_priority,         │
      │           is_running,           │
      │           core_id,              │
      │           bandwidth_quota}      │
      │                                 │
      │  coordination_map:              │
      │    pid → {combined_priority,    │
      │           slo_target,           │
      │           tenant_class}         │
      └────────────────────────────────┘
```

### 2.4 协调策略设计

#### 策略 1: GPU-Aware CPU Scheduling

**场景**: Multi-tenant GPU，LC inference + BE training 共存

**机制**: gpu_ext 将 per-PID GPU 状态（fault rate、memory pressure、eviction count）写入 `gpu_state_map`。sched_ext 在 `select_cpu()` / `enqueue()` 中读取该 map，据此调整：
- **Fault handler 优先**: 当 GPU fault rate 高时，提升处理 UVM fault 的 kthread 的 CPU 优先级
- **LC tenant boost**: GPU 正在为 LC tenant 做 prefetch/compute 时，boost 该 tenant 的 CPU 线程（减少 kernel launch delay）
- **BE tenant throttle**: GPU 内存压力大时，降低 BE tenant 的 CPU 调度优先级（减少其 GPU 访问频率，缓解 thrashing）

#### 策略 2: CPU-Aware GPU Memory Management

**场景**: CPU-intensive preprocessing → GPU inference pipeline

**机制**: sched_ext 将 per-PID CPU 状态（is_running、core_id、vruntime）写入 `cpu_state_map`。gpu_ext 在 `evict_prepare()` 中读取：
- **Running process protection**: 如果 process 正在 CPU 上运行（即将 launch GPU kernel），保护其 GPU 内存不被 evict
- **Sleeping process demotion**: 如果 process 已被 CPU deschedule，其 GPU 内存优先 evict
- **NUMA-aware prefetch**: 根据 process 所在 CPU core 的 NUMA domain 选择 prefetch 策略

#### 策略 3: Coordinated Anti-Thrashing

**场景**: 多 tenant 争抢 GPU memory 导致 thrashing

**机制**: 双向协调
- gpu_ext 检测到 tenant A 的 fault rate 异常高 → 写入 `coordination_map`
- sched_ext 读取 → 临时降低 tenant A 的 CPU scheduling 频率
- 效果: 减少 tenant A 的 GPU 访问频率 → 降低 fault rate → 所有 tenant 收益

#### 策略 4: SLO-Driven Resource Partitioning

**场景**: LC tenant 有 latency SLO（如 P99 < 50ms）

**机制**:
- Control plane 设置 SLO target 在 `coordination_map`
- gpu_ext 持续监测 LC tenant 的 page fault latency
- 如果 latency 接近 SLO → 信号传递给 sched_ext → 抢占 BE tenant 的 CPU 时间 + 提升 LC 的 GPU 内存优先级
- 形成闭环: SLO → monitoring → cross-subsystem adjustment → SLO 满足

### 2.5 为什么这是新颖的

> **核心 novelty 不是 "shared BPF map"（那只是管道），而是 "GPU 内存状态作为 CPU 调度的第一类信号"。**
>
> 详细 novelty 分析见 Section 八。

| 对比 | 已有工作 | xCoord 差异 |
|------|----------|-------------|
| sched_ext (Meta) | 用 kprobes 识别 GPU 线程，但不知道 GPU 内存状态 | 将 fault_rate/eviction_pressure 作为调度信号 |
| gpu_ext | GPU 策略可编程，但不知道 CPU 调度状态 | 根据 is_running 决定 eviction 优先级 |
| **MSched** | **GPU 内部 proactive memory scheduling（预测+流水线迁移），不涉及 CPU 调度** | **跨子系统协调：MSched 的迁移线程仍需 CPU 调度支持，xCoord 填补此缺口** |
| GPREEMPT | 修改 GPU 驱动做 preemption，不协调 CPU | 跨子系统协调，不修改 GPU 驱动 |
| MIG/MPS | 静态 GPU 分区 | 动态、策略驱动的资源共享 |
| Pegasus | Xen hypervisor 级 CPU-GPU co-scheduling | OS 内核级 eBPF 协调（更轻量、更灵活） |
| CPU pinning (taskset) | 静态 CPU 隔离，对大模型无效（实验数据: 19.2% 降级） | 动态调度，根据 GPU 状态实时调整 |

**关键差异**:
1. Meta sched_ext 可以识别 GPU 线程但**看不到 GPU 内存状态**（fault rate、eviction pressure）
2. 静态 CPU 隔离（taskset）在已测试的大模型场景中**无效**（实验数据支撑，但未穷举所有配置）
3. MSched 优化 GPU 内部内存调度但**不涉及 CPU 调度**——其迁移工作由 CPU 线程执行，若这些线程被 Linux 调度器降优先级则效果被抵消
4. xCoord 的 GPU memory awareness 填补了上述所有缺口

---

## 三、技术可行性分析

### 3.1 sched_ext 侧（高可行性）

**API 完备**:
- `select_cpu()`, `enqueue()`, `dispatch()` — 控制 task 放置和优先级
- `running()`, `stopping()` — 感知 task 执行状态
- `BPF_MAP_TYPE_TASK_STORAGE` — per-task 自定义状态
- 支持 pinned BPF maps — 可与 gpu_ext 共享

**已有参考实现**:
- `scx_layered` — 按 task 分类应用不同策略
- `scx_rusty` — hybrid BPF/user-space 负载均衡
- `scx_lavd` — latency-criticality 感知调度

**工程量**: 实现一个 GPU-aware sched_ext scheduler ~500-800 LOC BPF + ~300 LOC userspace

### 3.2 gpu_ext 侧（高可行性，BPF CO-RE 已就绪）

**已有**:
- eviction/prefetch hooks — 可读取 `cpu_state_map` 做决策
- per-PID tracking（`eviction_freq_pid_decay.bpf.c`）— 可写入 `gpu_state_map`
- GPU scheduling hooks（timeslice 控制）— 可配合 sched_ext
- **BPF CO-RE 完整支持** — 可读取 chunk address, VA block, 链表遍历

**需补齐**:
- 写入 shared map 的逻辑 — 在每个 hook 中更新 `gpu_state_map`
- 元数据清理 — 使用 `BPF_MAP_TYPE_LRU_HASH` 或在 `eviction_prepare` 中手动清理

**工程量**:
- 内核修改: **0 LOC** ✅
- BPF 策略修改: ~250 LOC (添加 shared map 写入)

### 3.3 共享状态（高可行性）

**BPF maps pinning**:
```bash
# gpu_ext 策略创建并 pin map
bpftool map create /sys/fs/bpf/gpu_state_map type hash key 4 value 32 entries 1024
# sched_ext scheduler 打开同一 map
int fd = bpf_obj_get("/sys/fs/bpf/gpu_state_map");
```

**一致性**: BPF maps 提供原子读写（per-entry），足够 µs-ms 级策略决策。不需要强一致性。

### 3.4 风险点

| 风险 | 概率 | 缓解 |
|------|------|------|
| sched_ext 性能 overhead 过高 | 低 | sched_ext 已在 Meta 生产部署，hot-path overhead ~100ns |
| 共享 map 更新延迟影响决策 | 中 | GPU 策略以 ms 为单位，map 更新是 µs 级，延迟可接受 |
| GPU 调度 hooks 不够深 | 中 | 论文可聚焦 memory coordination（已有），scheduling 作为 bonus |
| 内核版本要求 (6.12+) | 低 | 当前内核 6.15.11 已支持 sched_ext |
| 工程复杂度超预期 | 中 | 分阶段实现，先做 strategy 1 验证概念 |

---

## 四、评估设计

### 4.1 Research Questions

- **RQ1 (Coordination Benefit)**: 跨子系统协调能否改善 multi-tenant GPU 性能？（vs. 独立策略）
- **RQ2 (Policy Space)**: 哪些协调策略在哪些场景下有效？
- **RQ3 (Overhead)**: 跨子系统通信的开销是多少？
- **RQ4 (Generality)**: 协调框架能否支持多种策略组合？

### 4.2 实验矩阵

**Scenario 1: Multi-Tenant LC+BE** (主要评估)
- Config: llama.cpp inference (LC) + PyTorch GNN training (BE)
- Baselines: (a) no policy, (b) gpu_ext only, (c) sched_ext only, (d) xCoord coordinated
- Metrics: LC P99 TPOT, BE epoch time, PCIe bandwidth utilization

**Scenario 2: UVM Fault Handler Prioritization**
- Config: Single-tenant GPU workload with heavy UVM fault
- Baselines: (a) default CFS, (b) sched_ext no GPU awareness, (c) xCoord with fault-aware CPU scheduling
- Metrics: page fault latency, total throughput

**Scenario 3: Pipeline Workload**
- Config: CPU preprocessing → GPU inference → CPU postprocessing
- Baselines: (a) default, (b) xCoord with pipeline-aware scheduling
- Metrics: end-to-end latency, GPU utilization

**Scenario 4: Anti-Thrashing**
- Config: 3 tenants 共享 GPU，memory heavily oversubscribed
- Baselines: (a) no policy, (b) gpu_ext eviction only, (c) xCoord + CPU throttling
- Metrics: total fault rate, per-tenant throughput fairness

**Scenario 5: SLO-Driven**
- Config: LC with P99 SLO target + BE background
- Baselines: (a) no SLO enforcement, (b) gpu_ext priority only, (c) xCoord SLO feedback loop
- Metrics: SLO violation rate, BE degradation

### 4.3 已有可复用的实验基础设施

- `workloads/llama.cpp/configs/bench.py`, `server_bench.py` — LC workload
- `workloads/pytorch/configs/gnn.py` — BE workload
- `scripts/run_trials.py`, `collect_results.py` — multi-trial aggregation
- `workloads/cleanup_gpu.py` — GPU 清理

---

## 五、执行计划（8-11 周）— 零内核修改！

> **状态**: 原始设计文档。Phase 1-2 已通过 POC-1 实现（详见 §10-§14），Phase 3-5 需基于 profiling 数据重新设计（详见 §15.4）。

> **重大简化 v2**:
> 1. BPF CO-RE 已支持所有必需功能（`BPF_CORE_READ(chunk, address)` 等），**无需新增 kfunc**。详见 `bpf_core_access_findings.md`。
> 2. Depopulate hook **根本不存在于代码中** (非文档错误)，元数据清理用 `BPF_MAP_TYPE_LRU_HASH` 或在 `eviction_prepare` 中手动清理。
> 3. **零内核修改** — 全部工作在 BPF 程序和用户空间完成！

~~### Phase 0: Minimal Kernel Fix（已删除）~~

**Phase 0 已证明不需要**:
- ❌ ~~修复 depopulate hook~~ → Hook 不存在，`BPF_MAP_TYPE_LRU_HASH` 自动清理元数据
- ❌ ~~新增 kfunc~~ → BPF CO-RE 已提供所有功能

### Phase 1: GPU-side Shared Map Integration（1-2 周）

**目标**: gpu_ext 策略写入 per-PID GPU state 到 shared map

**工作内容**:
1. 定义 `shared_maps.h` — GPU/CPU state 结构体 (~50 LOC)
2. 创建 `eviction_lfu_xcoord.bpf.c` — LFU + shared map 示例 (~250 LOC)
   - 用 `BPF_CORE_READ(chunk, address)` 作为 map key 追踪频率
   - 在 `chunk_used` 中更新 `gpu_state_map` (fault_rate, mem_usage, eviction_count)
   - Pin map 到 `/sys/fs/bpf/gpu_state_map`
3. 用户空间工具 `read_gpu_state` 验证 map 内容 (~100 LOC)

**Deliverable**: 运行 workload 时能从 `/sys/fs/bpf/gpu_state_map` 读到 per-PID GPU 状态

### Phase 2: sched_ext GPU-aware Scheduler 原型（2-3 周）

1. 基于 `scx_layered` 实现 GPU-aware scheduler (~600 LOC BPF)
2. 实现 `select_cpu()`: 读取 `gpu_state_map`，为 GPU fault handler 选择最优 CPU
3. 实现 `enqueue()`: 根据 GPU fault_rate 调整 task 优先级
   - High fault_rate → boost priority (减少 fault latency)
   - Low fault_rate → normal priority
4. 实现 `running()`/`stopping()`: 写入 `cpu_state_map` (is_running, core_id)
5. 单元测试: 同时运行 gpu_ext 和 sched_ext，验证 map 双向通信

**Deliverable**: 完整的 CPU-GPU 状态双向共享框架

### Phase 3: 协调策略实现（2-3 周）

1. **Strategy 1**: GPU fault-aware CPU scheduling
   - sched_ext 读取 `fault_rate` → boost UVM fault handler kthread 优先级
2. **Strategy 2**: CPU state-aware GPU eviction
   - gpu_ext 读取 `is_running` → 保护正在 CPU 上运行进程的 GPU 内存
3. **Strategy 3**: Coordinated anti-thrashing
   - gpu_ext 检测高 fault_rate → 写入 `coordination_map`
   - sched_ext 读取 → 临时降低该进程 CPU 调度频率
4. **Strategy 4**: SLO-driven resource partitioning
   - 用户空间设置 `slo_target` → 双侧策略联动保障 SLO

**Deliverable**: 4 种协调策略全部实现，可独立开关

### Phase 4: 全面评估（2-3 周）

1. **Scenario 1-5** × 10 trials × geomean
   - Multi-tenant LC+BE, UVM fault handler prioritization, Pipeline, Anti-thrashing, SLO-driven
2. **Ablation study**: no policy vs gpu_ext only vs sched_ext only vs xCoord coordinated
3. **Overhead measurement**:
   - BPF CO-RE read overhead (expected: <10ns per read)
   - Map access latency (expected: <100ns)
   - Scheduling decision overhead
4. **Sensitivity analysis**: 不同 SLO targets, 不同 memory oversubscription ratios

**Deliverable**: 完整实验数据 + 性能对比图表

### Phase 5: 论文撰写（2-3 周）

1. 新论文框架（引用 gpu_ext 但不复用文本）
2. 重点章节:
   - **Motivation**: CPU-GPU 资源管理的耦合问题
   - **Design**: Shared BPF maps 协调协议，BPF CO-RE 优势
   - **Implementation**: 4 种协调策略的具体实现
   - **Evaluation**: 5 scenarios 完整数据，overhead 分析
3. Related work: sched_ext, gpu_ext, Pegasus, MIG/MPS 对比

**Deliverable**: 可投稿论文 draft

### 关键 Milestone（零内核修改版）

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1-2 | GPU shared map | LFU + shared map 示例，pin 到 /sys/fs/bpf/ |
| 3-5 | sched_ext 原型 | GPU-aware scheduler 双向通信验证 |
| 6-8 | 协调策略完成 | 4 种策略全部实现，基础功能验证 |
| 9-10 | 评估完成 | 所有 scenario 有完整数据 + overhead 分析 |
| 11 | 论文 draft | 可投稿状态 |

**总工程量**:
- 内核修改: **0 LOC** ✅ (vs 原计划 55 LOC → 5 LOC → 0 LOC)
- BPF 程序: ~850 LOC (gpu_ext policies + sched_ext scheduler)
- 用户空间: ~100 LOC (测试工具)
- **总计**: ~950 LOC (vs 原计划 ~1255 LOC，减少 24%)

---

## 六、与 gpu_ext 论文的关系

| 维度 | gpu_ext（在审） | xCoord（新论文） |
|------|----------------|-----------------|
| 焦点 | GPU driver 可编程性 | CPU-GPU 跨子系统协调 |
| 技术栈 | gpu_ext BPF hooks + device-side eBPF | sched_ext + gpu_ext + shared maps |
| 评估重点 | 单 tenant 性能优化 | Multi-tenant 协调 |
| 创新点 | GPU driver extensibility | Cross-subsystem eBPF coordination |
| 依赖 | 独立 | 引用 gpu_ext 作为 GPU 侧基础设施 |

**关键**: xCoord 将 gpu_ext 作为组件（GPU 侧 hook），加上 sched_ext（CPU 侧 hook）和协调协议（shared BPF maps），构成完整的跨子系统协调系统。两篇论文互不冲突，xCoord 甚至可以 strengthen gpu_ext 的 story（"gpu_ext 不仅能单独使用，还能与 CPU scheduler 协调"）。

---

## 七、目标会议

| 会议 | Deadline 2026 | 匹配度 | 备注 |
|------|-------------|--------|------|
| **OSDI '27** | ~2026.11 | ★★★★★ | 系统顶会，cross-subsystem design 很 OSDI |
| **SOSP '27** | ~2026.04 (偶数年) | ★★★★ | 需看 SOSP 2027 是否在 cycle |
| **EuroSys '27** | ~2026.10 | ★★★★ | 偏欧洲，接受 system building |
| **ATC '27** | ~2027.01 | ★★★ | 偏实用，可以作为 backup |
| **ASPLOS '27** | ~2026.10 | ★★★★ | 偏 architecture，但接受 OS-level 工作 |

建议优先瞄准 **OSDI '27** 或 **EuroSys '27**。

---

## 八、Novelty 分析：xCoord 与 gpu_ext 的根本区别

### 8.1 当前 claim 的问题

当前描述 "通过 shared BPF maps 在 sched_ext 和 gpu_ext 之间共享状态" **缺乏 novelty**：
- Shared BPF map 只是管道（pipe），不是洞察（insight）
- 任何人读完 sched_ext 和 gpu_ext 的代码都能想到 "pin 一个 map 让两边读写"
- 真正的问题是：**什么问题只有跨子系统协调才能解决？**

### 8.2 关键洞察：CPU-GPU 耦合是不对称的

**核心发现（已有实验数据支撑）**:

1. **CPU 调度直接影响 GPU 性能，幅度远超预期**
   - 实验数据（`scripts/sched/`）：CPU stress 导致 GPU LLM 推理 **19-26% 吞吐量下降**
   - Qwen3-30B: 218.7 → 177.1 tok/s（CPU stress），218.7 → 160.8 tok/s（heavy load）
   - Context switches 增加 **21,990-36,414x**（0.1 → 2578-3179 per 1K CUDA launches）

2. **朴素的 CPU 隔离方案无效**
   - `taskset -c 0-3` + `nice -n -10` 无效：19.2% 降级（vs 无 pinning 的 19.0%），pinning 到争用核反而加剧竞争
   - 原因：pinning 到已争用的核反而加剧竞争
   - 需要内核级 `isolcpus` + IRQ affinity，但这是静态的、需要 root 权限、不可动态调整

3. **GPU 状态对 CPU 调度器不可见**
   - Meta sched_ext 部署（`scripts/sched/pdf_summary.md`）：用 kprobes 在 NVIDIA 驱动上识别 GPU-critical threads
   - 但 Meta **无法看到 GPU 内存状态**（fault rate、eviction pressure、memory utilization）
   - 这意味着 CPU 调度器在 "盲飞"：不知道 GPU 侧正在发生什么

4. **UVM page fault 的 CPU-GPU 耦合**
   - GPU 触发 page fault → CPU interrupt → CPU worker thread 处理 → PCIe 传输
   - Host-side overhead 是 PCIe 传输时间的 **7x**（来自 gpu_ext 论文数据）
   - **MSched 论文进一步量化**: 每次 fault 31.79μs，其中 **96% 是 CPU 控制面**（中断处理、TLB 管理），仅 4% 是数据传输
   - CPU 被抢占 = page fault 控制面延迟膨胀 = GPU 空闲等待 = 吞吐量下降
   - 这意味着 **CPU priority boost 对 fault 处理的影响比我们预期的更大**（控制面才是瓶颈）

5. **即使 GPU 内部调度最优，CPU 侧仍是盲区**（MSched 启发）
   - MSched 实现了 GPU 内部最优内存调度（Belady OPT + 流水线迁移，LLM 57.88x speedup）
   - 但 MSched 的所有迁移工作仍由 **CPU 线程执行**（D2H/H2D DMA setup、TLB 管理、中断处理）
   - 如果这些 CPU 线程被 noisy neighbor 抢占，MSched 的优化效果会被抵消
   - **没有任何现有系统解决 "GPU memory-aware CPU scheduling"**——这是 xCoord 的独特位置

### 8.3 xCoord 的真正 Novelty

**不是 "共享 map"，而是 "GPU memory awareness 作为 CPU 调度信号"**。

| 维度 | gpu_ext | xCoord |
|------|---------|--------|
| **问题** | GPU driver 不可编程 | CPU 和 GPU 资源管理互相 "盲飞" |
| **洞察** | eBPF struct_ops 可扩展 GPU driver | GPU 内存状态（fault rate、eviction pressure）是 CPU 调度的关键缺失信号 |
| **贡献** | GPU 侧策略可编程性 | 跨子系统闭环：GPU 状态 → CPU 调度 → GPU 性能改善 |
| **评估** | 单 tenant：不同 eviction/prefetch 策略效果 | Multi-tenant：协调 vs 独立 vs 朴素隔离 |
| **关键实验** | "用 LFU 替代 LRU" → hit rate 提升 | "CPU scheduler 看到 fault rate" → tail latency 降低 + CPU pinning 无法达到的效果 |

**一句话 novelty**: GPU 内存子系统的运行时状态（page fault rate、eviction pressure）是 CPU 调度决策的第一类（first-class）输入信号，通过 eBPF 跨子系统共享这些信号，可以实现朴素隔离方案无法达到的性能保障。

### 8.4 与 Meta sched_ext 的区别

Meta 的 `scx_layered` 在 AI 训练中部署（`scripts/sched/pdf_summary.md`）：
- ✅ 可以识别 GPU-critical threads（通过 kprobes 在 NVIDIA 驱动上）
- ✅ 恢复了被 noisy neighbors 抢走的 ~25% 性能
- ✅ Fleet-wide 节省 4% GPU capacity

**但 Meta 缺少的**:
- ❌ 不知道 GPU 内存状态（fault rate 高不高？eviction 在发生吗？）
- ❌ 不能根据 GPU memory pressure 动态调整 CPU 优先级
- ❌ 不能做反向协调（CPU 状态 → GPU eviction 决策）
- ❌ 对 UVM/managed memory workload 没有特殊处理

**xCoord 的补充**: 在 Meta 的 "识别 GPU 线程" 基础上，加入 "理解 GPU 内存状态"，实现更精细的协调。

### 8.5 与 MSched 的区别（GPU 内部 vs 跨子系统）

**MSched** (arxiv 2512.24637) 是 GPU 内存调度的 SOTA，通过预测 + 主动迁移实现 GPU 多任务下的最优页替换。

**详细分析**: [`docs/reference/msched_analysis.md`](../reference/msched_analysis.md)

| 维度 | MSched | xCoord |
|------|--------|--------|
| **优化层次** | GPU 子系统内部 | CPU↔GPU 跨子系统 |
| **核心机制** | 预测 kernel 访存 → Belady OPT → 流水线迁移 | GPU 内存状态 → CPU 调度信号 → priority boost |
| **CPU 调度** | ❌ 不涉及 | ✅ 核心贡献 |
| **GPU task 调度** | ✅ 扩展 XSched | ❌ 不控制 |
| **内存管理** | ✅ 核心（预测+迁移） | 间接（eviction 策略） |
| **可编程性** | ❌ 固定算法 | ✅ eBPF 运行时可换策略 |
| **驱动修改** | 需要（2 个新 ioctl） | 不需要（BPF struct_ops） |
| **Multi-tenant CPU 隔离** | ❌ 不处理 | ✅ per-PID GPU 状态驱动差异化调度 |

**互补关系**:

MSched 解决了 "GPU 内部该迁移哪些页"，但迁移工作由 CPU 线程执行。在多租户环境下，如果这些 CPU 线程被 noisy neighbor 抢占，MSched 的优化会被削弱。xCoord 恰好填补这个缺口——确保 GPU memory-critical 的 CPU 线程获得适当的调度优先级。

**论文叙事**:
> "Even with optimal GPU-side memory scheduling (MSched), CPU-GPU coordination gaps remain because the CPU scheduler is unaware of GPU memory pressure. xCoord fills this gap via cross-subsystem eBPF coordination."

---

## 九、已有实验证据（Motivation Data）

> 以下数据来自 `scripts/sched/` 目录下的实验，可直接作为论文 motivation section 的定量支撑。

### 9.1 CPU 调度对 GPU 推理性能的影响

**实验配置**: Qwen3-30B-A3B-FP8, ShareGPT 200 prompts, llama-server

| 场景 | tok/s | 降级 | Context Switch/1K | 增幅 |
|------|-------|------|-------------------|------|
| Baseline | 218.7 ± 0.1 | - | 0.1 | 1x |
| CPU Stress | 177.1 ± 3.6 | **19.0%** | 2578.4 | 21,990x |
| Network Stress | 218.2 ± 0.3 | 0.2% | 0.6 | 5x |
| Disk Stress | 218.8 ± 0.1 | 0.0% | 0.1 | 1x |
| **Heavy Load** | **160.8 ± 6.9** | **26.5%** | **2895.4** | **24,694x** |
| CPU Pinned | 176.8 ± 2.3 | **19.2%** | 3179.3 | 27,115x |

**关键发现**:
1. **CPU contention 是主导因素**：网络/磁盘几乎无影响，CPU 争用导致 19-26% 性能损失
2. **CPU pinning 无效**：不但没改善，反而略差（19.2% vs 19.0%），因为 pinning 到已争用的核加剧竞争
3. **Heavy load 非线性放大**：CPU+Net+Disk 组合（26.5%）远超各自之和

### 9.2 干净环境下的 baseline 分析

**实验配置**: Qwen3-0.6B, 无干扰环境

| 指标 | 值 | 说明 |
|------|-----|------|
| Context switches | 592 total (7.44 Hz) | 极低 |
| Kernel launches | 51,464 | |
| 受影响的 launch pairs | 62/51,463 (0.1%) | 极少 |
| 每次抢占的代价 | 15.3 ms (vs 正常 2 µs) | **7,650x 放大** |
| 总调度影响 | 1.2% | 干净环境下可忽略 |
| IRQ 总影响 | 0.0276% | 本地推理下可忽略 |

**关键发现**:
1. **干净环境 → 调度开销极小**（1.2%），但一旦有 noisy neighbors → 19-26%
2. **每次抢占代价极高**（7,650x），但发生概率低 → 尾延迟是关键战场
3. **IRQ 在本地推理中可忽略**，但分布式训练中 NET_RX 影响 5-20%（Meta 数据）

### 9.3 CPU Pinning 失败的根因分析

| 实验 | Model | Pinning 策略 | 效果 |
|------|-------|-------------|------|
| REPORT.md | Qwen3-0.6B | taskset -c 0-3 + nice -10 | **有效**: sched/1K 从 11,933 → 445 (96.3% 减少) |
| analysis_report | Qwen3-30B | taskset -c 0-3 + nice -10 | **无效**: sched/1K 从 2,578 → 3,179 (增加 23%) |

**分析**:
- 小模型（0.6B）：CPU 线程少，pinning 可以在 4 核上隔离
- 大模型（30B）：CPU 线程多（page fault handler、CUDA runtime、多个 worker），pinning 到 4 核反而加剧核内竞争
- **结论**: 静态 pinning 无法适应不同 workload 的动态需求 → 需要动态的 GPU-aware CPU 调度

### 9.4 Meta sched_ext 部署数据

来源: `scripts/sched/pdf_summary.md`（Meta AI Training 内部分析）

| 指标 | 值 |
|------|-----|
| 部署规模 | AI 训练集群 fleet-wide |
| 使用的 scheduler | scx_layered |
| Noisy neighbor 性能恢复 | ~25% |
| GPU capacity 节省 | 4% |
| GPU thread 识别方法 | kprobes on NVIDIA driver |

**Meta 的关键挑战**: 识别 GPU-critical threads。他们用 kprobes 在 NVIDIA 驱动上打点。
**Meta 的盲点**: 不知道 GPU 内存状态 → 不能根据 fault rate 调整 CPU 优先级。

---

## 十、详细 POC 方案

> 目标: 用**最小工程量**验证 xCoord 的核心 claim："GPU 内存状态作为 CPU 调度信号可以改善朴素方案无法达到的性能"。
>
> 原则: **先证明问题存在，再证明方案有效，最后证明方案优于替代方案。**

### 10.1 POC 总览

```
POC-0: 量化问题（已部分完成）         → 1 周
POC-1: 最小单向协调（GPU→CPU）        → 2 周
POC-2: 验证优于朴素方案               → 1 周
POC-3: 反向协调（CPU→GPU）           → 2 周（可选）
                                     ────────
                                     总计: 4-6 周
```

### 10.2 POC-0: 量化 CPU-GPU 耦合问题 ✅ 已完成

> **状态**: 2026-02-23 完成全部实验
> **结果目录**: `scripts/xcoord/results/poc0_20260223_152937/` (20B) 和 `scripts/xcoord/results/poc0_uvm_20260223_154122/` (120B UVM)

**目标**: 建立完整的 motivation data，证明 "CPU 调度影响 GPU 性能" 不是 corner case。

#### 10.2.1 实验 A: 20B 模型（模型完全装入 GPU，无 UVM paging）

**配置**: gpt-oss-20b-mxfp4 (~10GB，适配 32GB RTX 5090)，50 ShareGPT prompts

| 场景 | tok/s | 降级 | Requests OK | TPOT Mean | TPOT P99 |
|------|-------|------|-------------|-----------|----------|
| Baseline | **198.67** | - | 50/50 | 3.72ms | 6.29ms |
| CPU Stress | **175.40** | **-11.7%** | 40/50 | 4.54ms | 16.48ms |
| CPU Stress + Pinned | **179.71** | -9.5% | 40/50 | 4.62ms | 6.42ms |
| Heavy Load | **185.38** | -6.7% | 43/50 | 4.11ms | 13.68ms |

**GPU Hook 调用频率 (chunk_trace)**:

| 场景 | Activate/s | Used/s | Evict/s | Total Hooks |
|------|------------|--------|---------|-------------|
| Baseline | 67 | 453 | 0 | 49,363 |
| CPU Stress | 64 | 428 | 0 | 49,345 |
| CPU Pinned | 64 | 426 | 0 | 49,348 |
| Heavy Load | 59 | 396 | 0 | 49,365 |

**关键发现**:
1. **98.3% 的 ACTIVATE 事件发生在第 1 秒**（模型加载阶段），推理阶段几乎无 page fault
2. **零 eviction** — 模型完全装入 GPU，不存在 UVM thrashing
3. **CPU stress → 11.7% 吞吐量下降** — 纯粹的线程调度延迟（kernel launch delay）
4. **CPU pinning 仅恢复 2.5%** — 效果有限

#### 10.2.2 实验 B: 120B 模型（超出 GPU 内存，重度 UVM paging）⭐

**配置**: gpt-oss-120b-mxfp4 (~60GB，远超 32GB GPU)，20 ShareGPT prompts

| 场景 | tok/s | Requests OK | TPOT Mean | TPOT P99 | TTFT Mean | TTFT P99 |
|------|-------|-------------|-----------|----------|-----------|----------|
| UVM Baseline | 11.24 | 3/20 ⚠️ | 79.32ms | 86.05ms | 521.85ms | 1127.83ms |
| UVM + CPU Stress | 13.11 | 20/20 | 73.31ms | 132.38ms | **3160.10ms** | **7426.73ms** |
| UVM + CPU Stress + Pinned | 13.03 | 20/20 | 73.50ms | 132.31ms | **3169.66ms** | **7407.45ms** |

> ⚠️ Baseline 在 3 个请求后 segfault（llama-server UVM 代码不稳定）

**GPU Hook 调用频率**:

| 场景 | Duration | Activate/s | Used/s | Evict/s | Total Hooks |
|------|----------|------------|--------|---------|-------------|
| UVM Baseline | 214.8s | **2,225** | **5,279** | **2,150** | 2,073,192 |
| UVM + CPU Stress | 529.3s | **2,492** | **8,244** | **2,461** | 6,985,725 |
| UVM + CPU Stress + Pinned | 528.5s | **2,492** | **8,216** | **2,460** | 6,959,722 |

**与 20B 模型对比（关键数据）**:

| 指标 | 20B Model | 120B UVM Model | 比率 |
|------|-----------|----------------|------|
| Activate/s (baseline) | 67 | 2,225 | **33x** |
| Used/s (baseline) | 453 | 5,279 | **12x** |
| Evict/s (baseline) | 0 | 2,150 | **∞** |
| Total hooks (baseline) | 49,363 | 2,073,192 | **42x** |
| 推理阶段 page fault | ~0 | ~800,000+ | **∞** |

**⭐ 核心发现: TTFT Explosion + CPU Pinning 失效**:

1. **TTFT mean**: 521ms (baseline) → **3160ms** (CPU stress) = **6.1x 增加**
2. **TTFT P99**: 1128ms → **7427ms** = **6.6x 增加**
3. **CPU pinning 零效果**: 3160ms (unpinned) vs 3170ms (pinned) — 完全相同
4. **原因**: UVM page fault handler 运行在**内核 worker 线程**中，不是 llama-server 进程
   - `taskset` 只 pin 了用户空间线程
   - nvidia_uvm 内核模块的 fault handler 线程在任意 CPU 上运行
   - CPU stress 影响的是**这些内核线程**，不是被 pin 的用户空间线程
   - **因此 CPU pinning 对 UVM fault handler 无效** (taskset 只影响用户空间线程)

#### 10.2.3 两种 CPU-GPU 耦合机制

| 机制 | 证明实验 | 影响 | CPU Pinning 效果 |
|------|---------|------|-----------------|
| **线程调度延迟** | 20B 模型 | -11.7% throughput | 微弱 (+2.5%) |
| **UVM page fault 延迟** | 120B 模型 | TTFT 6.1x 增加 | 无效 (0%) — taskset 不影响内核线程 |

**xCoord 的核心价值**: 一个 GPU-aware CPU scheduler 需要知道**哪些线程是 UVM fault handler** 并提升其优先级，而不是仅仅 pin 用户空间 GPU 线程。

#### 10.2.4 POC-0 产出 ✅

1. ✅ "CPU 干扰 → GPU 性能" 因果表（两种机制量化）
2. ✅ UVM hook 频率对比（20B vs 120B，33-42x 差异）
3. ✅ "CPU pinning 为什么失败" 的定量解释（内核线程不受 taskset 影响）
4. ✅ 论文 Motivation section 的完整数据
5. ✅ 确认 120B UVM 是 POC-1 的理想测试场景

**POC-0 结论**: **问题真实存在且严重。继续 POC-1。**

**报告详情**: `scripts/xcoord/results/poc0_20260223_152937/REPORT.md` 和 `scripts/xcoord/results/poc0_uvm_20260223_154122/REPORT.md`

---

### 10.3 POC-1: 最小单向协调 GPU→CPU — ✅ 完成

> **状态**: 2026-02-27 完成。代码实现 + 3 轮调试 + 3 个 critical bug 修复 + 多 workload 验证。详细结果见 §14。
> **代码目录**: `extension/` (`eviction_lfu_xcoord*`, `sched_gpu_baseline*`, `shared_maps.h`)
> **实验结果目录**: `scripts/xcoord/results/poc1_xcoord_*/`

**目标**: 用最少代码验证 "gpu_ext 写入 GPU 状态 → sched_ext 读取并调整优先级 → 性能改善"。

#### 10.3.1 架构（实际实现版）

```
┌───────────────────────────┐     ┌───────────────────────────┐
│   gpu_ext BPF              │     │   sched_ext BPF            │
│   eviction_lfu_xcoord      │     │   sched_gpu_baseline          │
│                           │     │                           │
│  chunk_activate():         │     │  enqueue():                │
│    update fault_rate       │     │    1. check uvm_worker_pids│
│    track_uvm_worker()  ───┼──→  │       if active worker:    │
│  chunk_used():             │     │       → boost (ENQ_HEAD)  │
│    LFU frequency tracking  │     │    2. check gpu_state_map  │
│    track_uvm_worker()  ───┼──→  │       if thrashing:        │
│  eviction_prepare():       │     │       → throttle          │
│    LFU eviction            │     │    3. else: vtime-fair     │
│    track_uvm_worker()  ───┼──→  │                           │
│                           │     │  select_cpu():             │
│  Pinned maps:              │     │    → prev_cpu (cache warm) │
│    gpu_state_map           │     │  dispatch():               │
│    uvm_worker_pids         │     │    → dsq_move_to_local    │
└───────────────────────────┘     └───────────────────────────┘
          │                                │
          ├── /sys/fs/bpf/xcoord_gpu_state ┤
          └── /sys/fs/bpf/xcoord_uvm_workers ┘
```

#### 10.3.2 已完成的实现 ✅

**所有代码已编写并编译通过。** 文件清单:

| 文件 | LOC | 功能 | 状态 |
|------|-----|------|------|
| `shared_maps.h` | 48 | GPU/CPU 共享状态定义 + worker map 常量 | ✅ 完成 |
| `eviction_lfu_xcoord.bpf.c` | 313 | LFU 驱逐 + gpu_state_map 写入 + worker PID 追踪 | ✅ 完成 |
| `eviction_lfu_xcoord.c` | 252 | GPU 侧 loader，pin 两个共享 map | ✅ 完成 |
| `sched_gpu_baseline.bpf.c` | 180 | sched_ext BPF：enqueue 读取 worker map + gpu_state_map | ✅ 完成 |
| `sched_gpu_baseline.c` | 214 | sched_ext loader，bpf_obj_get + reuse_fd 连接两个 map | ✅ 完成 |
| `Makefile` (修改) | — | 添加 SCX_APPS 构建规则（双 vmlinux.h） | ✅ 完成 |
| `poc1_xcoord_scheduling.sh` | 274 | 3 场景自动化实验脚本 | ✅ 完成 |
| **总计** | **1281** | | |

**关键设计决策**:

1. **双 vmlinux.h 方案**: gpu_ext 用旧 `vmlinux/x86/vmlinux.h`（包含 NVIDIA UVM 类型），sched_ext 用新 `vmlinux/x86/scx/vmlinux.h`（包含 sched_ext 类型）
2. **直接 libbpf API**: sched_gpu_baseline.c 用直接 libbpf API（非 SCX_OPS_OPEN/LOAD/ATTACH），因为 UEI 宏要求 clang 18 不支持的 32-bit atomics
3. **SCX_ENUM_INIT 关键**: 必须在 open() 后调用 `SCX_ENUM_INIT(skel)` 和 `scx_hotplug_seq()`，否则所有 sched_ext 常量（SCX_DSQ_LOCAL, SCX_SLICE_DFL, SCX_ENQ_HEAD）均为 0，调度器完全不工作
4. **Skeleton 命名**: bpftool 从 `sched_gpu_baseline.bpf.o` 生成 `sched_gpu_baseline.skel.h`，skeleton 结构体名为 `sched_gpu_baseline_bpf`（带 `_bpf` 后缀）

#### 10.3.3 调试迭代记录（摘要）

> 详细经验教训见 §12.2。此处仅保留摘要。

3 轮调试迭代：
1. **编译问题** → skeleton 命名规则、`u64` 类型、libbpf API（详见 §12.2.1）
2. **SCX_ENUM_INIT 缺失** → 调度器 state=disabled，修复后正常
3. **gpu_boosted=0** → UVM worker PID 不匹配用户进程 PID，新增 `uvm_worker_pids` map 解决（详见 §12.2.2）

#### 10.3.4 中间版本实验结果（摘要）

> 已被 §14 最终结果取代。

- 120B UVM batch: CPU stress 影响仅 ~1.4%（GPU pipeline 隐藏 CPU 延迟）→ batch 不是 sched_ext 理想场景
- worker PID 追踪: gpu_boosted=1/39021（匹配率极低）→ 切换到 20B + `gpu_process_pids` 直接注册后解决

#### 10.3.5 阻塞问题（均已解决，摘要）

1. **Worker PID 追踪**: 机制功能正常但 120B batch 匹配率低 → 切换 20B 后正常（详见 §14.1）
2. **llama-server OOM**: 残留 GPU 进程占显存 → `cleanup_gpu.py` + 验证 GPU 0MiB（详见 §12.2.3）

**预防措施**: 实验脚本开头必须 `cleanup_gpu.py` + 验证 GPU 0MiB; 手动调试后必须清理; UVM 场景下任何残留进程都可能导致 OOM。

#### 10.3.6 下一步工作（120B 阶段，已全部完成）

> **状态**: 全部完成。items 4-5 通过切换到 20B + 新增 `gpu_process_pids` map 解决，详见 §14.1 Bug 修复。item 6 通过 stress-ng 全核 + concurrency=4 验证，详见 §14.4。

1. ~~**🔴 修复 llama-server 120B 崩溃问题**~~ → ✅ 已解决（残留 GPU 进程占用显存）
2. ~~**🔴 重写实验脚本使用标准 benchmark 工具**~~ → ✅ 已完成（`poc1_xcoord_bench.py`）
3. ~~**🟡 验证 worker PID 追踪效果**~~ → ✅ 已验证（匹配率极低 1/39021，120B UVM overhead 主导）
4. ~~**🔴 切换到 20B 模型重新验证**~~ → ✅ 已完成，详见 §14
5. ~~**🟡 分析 worker PID 匹配率低的原因**~~ → ✅ 根因：20B 无 UVM → worker map 空。改用 `gpu_process_pids` 直接注册 PID
6. ~~**🟢 更激进的 CPU 干扰方式**~~ → ✅ stress-ng 全 24 核 + concurrency=4 已验证

#### 10.3.7 POC-1 成功标准（已达成）

> **状态**: POC-1 于 2026-02-27 达成成功标准，详见 §14。

| 指标 | 120B 结果 | 20B 结果 (§14) | 判定 |
|------|-----------|---------------|------|
| gpu_boosted | 1/39021 | **12,322/468K (2.5%)** | ✅ 已解决 |
| CPU stress 影响 | ~0% | **+25% TPOT** | ✅ 显著 |
| xCoord vs stress | 无差异 | **TPOT 完全恢复 4.44→3.72ms** | ✅ 成功 |
| sched_ext state | enabled | enabled | ✅ 稳定 |

---

### 10.4 POC-2: 验证优于朴素方案（1 周）

**目标**: 用 ablation study 证明 "GPU awareness" 是关键，不是 sched_ext 本身。

#### 10.4.1 实验矩阵

```
Ablation 4 个变体:

(A) xCoord full:     sched_ext 读取 gpu_state → 动态调整优先级
(B) sched_ext blind:  sched_ext 不读取 gpu_state → 静态 boost 所有 GPU PID
(C) sched_ext oracle: sched_ext 读取 gpu_state → 但只用于 logging，不调整优先级
(D) taskset optimal:  用 cgroup cpuset 精心配置的 CPU 隔离

比较 (A) vs (B) → "GPU awareness" 的价值
比较 (A) vs (D) → 动态协调 vs 静态隔离
比较 (B) vs (C) → sched_ext 本身的价值（不含 GPU awareness）
```

#### 10.4.2 敏感性分析

```
变量:
(1) 干扰强度: stress-ng -c 4 / 8 / 16 / $(nproc)
(2) GPU 内存压力: 不同模型大小 (0.6B / 7B / 30B)
(3) Multi-tenant 数量: 2 / 3 / 4 个 GPU 进程
(4) fault_rate 阈值: 100 / 500 / 1000 / 5000 faults/sec

目标: 找到 xCoord 优势最大的 sweet spot
```

**POC-2 成功标准**:
- (A) xCoord full 比 (B) sched_ext blind 至少好 **3 个百分点**
- 证明 "GPU awareness" 是性能改善的关键因素

---

### 10.5 POC-3: 反向协调 CPU→GPU（2 周，可选）

**目标**: 验证双向协调比单向更好。

#### 10.5.1 CPU 状态写入

```c
// 在 sched_ext 的 running() hook 中:
void BPF_STRUCT_OPS(gpu_aware_running, struct task_struct *p) {
    u32 pid = p->tgid;
    struct cpu_pid_state *state = bpf_map_lookup_elem(&cpu_state_map, &pid);
    if (state) {
        state->is_running = 1;
        state->core_id = bpf_get_smp_processor_id();
        state->last_run_ns = bpf_ktime_get_ns();
    }
}

// 在 sched_ext 的 stopping() hook 中:
void BPF_STRUCT_OPS(gpu_aware_stopping, struct task_struct *p) {
    u32 pid = p->tgid;
    struct cpu_pid_state *state = bpf_map_lookup_elem(&cpu_state_map, &pid);
    if (state) {
        state->is_running = 0;
    }
}
```

#### 10.5.2 gpu_ext 读取 CPU 状态

```c
// 在 eviction_prepare 中:
SEC("struct_ops/gpu_evict_prepare")
int BPF_PROG(gpu_evict_prepare, ...) {
    // 遍历 eviction 候选列表
    struct list_head *pos = BPF_CORE_READ(list, next);
    #pragma unroll
    for (int i = 0; i < 128 && pos != list; i++) {
        uvm_gpu_chunk_t *chunk = container_of(pos, uvm_gpu_chunk_t, list);
        u32 pid = get_owner_pid_from_chunk(chunk);

        // 读取 CPU 状态
        struct cpu_pid_state *cpu = bpf_map_lookup_elem(&cpu_state_map, &pid);
        if (cpu && cpu->is_running) {
            // 进程正在 CPU 上运行 → 即将 launch GPU kernel → 保护其 GPU 内存
            // 跳过这个 chunk，不驱逐
            pos = BPF_CORE_READ(pos, next);
            continue;
        }

        // 进程不在 CPU 上运行 → 优先驱逐其 GPU 内存
        bpf_gpu_block_move_head(chunk, list);
        pos = BPF_CORE_READ(pos, next);
    }
    return 0;
}
```

**测试场景**: Multi-tenant (2 个 GPU 进程共享 GPU 内存)
- LC 进程正在 CPU 上运行 → gpu_ext 保护其 GPU 内存
- BE 进程被 CPU deschedule → gpu_ext 优先驱逐其 GPU 内存

**POC-3 成功标准**:
- 双向协调 (POC-3) 比单向 (POC-1) 在 multi-tenant 场景下额外改善 **>3%**
- 如果未达到: 说明单向已足够，论文聚焦 GPU→CPU 方向

---

### 10.6 POC 时间线 + 决策树

> 详细进度和下一步见 §0。此处仅保留里程碑摘要。

- **Week 1 (POC-0)**: ✅ 量化 CPU-GPU 耦合（20B: 11.7%, 120B: 6.1x TTFT, pinning 无效）
- **Week 2 (POC-1 实现)**: ✅ shared_maps + eviction_lfu_xcoord + sched_gpu_baseline（3 轮调试）
- **Week 3 (POC-1 验证)**: ✅ 20B TPOT 完全恢复; GNN +3%; FAISS +67-759%(不适合); → 适合 LLM serving
- **Week 4 (当前)**: ⏳ Profiling-first 方法 + 完整链路测试
- **POC-2/3**: ⏳ 待完整链路测试后决定

---

## 十一、工程量重新评估

### 11.1 POC 阶段（实际进展）

| 组件 | 预估 LOC | 实际 LOC | 预估天数 | 实际天数 | 状态 |
|------|---------|---------|---------|---------|------|
| `shared_maps.h` | ~50 | 48 | 0.5 | 0.2 | ✅ |
| `eviction_lfu_xcoord.bpf.c` | ~250 | 313 | 3 | 0.5 | ✅ |
| `eviction_lfu_xcoord.c` (loader) | — | 252 | — | 0.5 | ✅ |
| `sched_gpu_baseline.bpf.c` | ~400 | 180 | 5 | 0.5 | ✅ |
| `sched_gpu_baseline.c` (loader) | ~150 | 214 | 2 | 0.5 | ✅ |
| `poc1_xcoord_scheduling.sh` | ~200 | 274 | 2 | 0.3 | ✅ |
| 调试 + 3 轮迭代修复 | — | — | — | 1.0 | ✅ |
| **总计** | **~1050** | **1281** | **~12.5** | **~3.5** | **代码完成** |

**备注**: 实际代码量比预估多 22%（主要是 loader 比预估复杂：SCX_ENUM_INIT、双 map reuse、worker 追踪），但编码速度远快于预估（3.5 天 vs 12.5 天）。主要耗时在调试迭代（3 轮：编译修复 → SCX_ENUM_INIT → worker PID 追踪）。

### 11.2 完整论文阶段（原始预估，已过时）

> **注意**: 以下是 POC 阶段前的原始预估。实际进展表明 POC-1 用了 ~1 周（vs 预估 2-3 周），但发现了预估中未预见的问题（sched_ext overhead 对 batch workload 不可接受、blind boost 缺乏 novelty）。后续计划已调整为 profiling-first 方法，见 §15。

| Phase | 内容 | 原始预估 | 实际状态 |
|-------|------|---------|---------|
| Phase 1-2 (POC) | GPU shared map + sched_ext 原型 | 4-6 周 | ✅ ~1 周完成 |
| Phase 3 | 4 种协调策略完整实现 | 2-3 周 | ⏳ 需基于 profiling 重新设计 |
| Phase 4 | 5 scenarios 完整评估 | 2-3 周 | ⏳ |
| Phase 5 | 论文撰写 | 2-3 周 | ⏳ |

---

## 十二、风险与备选

### 12.1 POC 级风险（更新版，含实测验证）

| 风险 | 预估概率 | 实际结果 | 备注 |
|------|---------|---------|------|
| GPU fault latency 对 CPU 调度不敏感 | 低 | ✅ **已验证敏感** | POC-0: 6.1x TTFT 增加 |
| sched_ext 与 gpu_ext 同时加载冲突 | 中 | ✅ **不冲突** | 两组件同时运行稳定 |
| Shared map 延迟过高 | 低 | ✅ **可接受** | map 操作 µs 级，不影响调度 |
| sched_ext 导致系统不稳定 | 中 | ✅ **稳定** | state=enabled，安全 fallback 正常 |
| 改善幅度不够发论文 | 中 | ⚠️ **部分验证** | POC-1 blind boost 恢复 stress 退化（TPOT +25%→0%），但 blind boost 缺乏 novelty；主动优化（无 stress）对 GNN +3% overhead，对 FAISS 有害。需 profiling-first 设计针对性算法 |
| **新增**: UVM worker PID 不匹配 | 未预见 | ⚠️ **已修复代码** | 核心线程 tgid ≠ 用户进程 tgid |
| **新增**: 120B 模型 CUDA OOM | 未预见 | ✅ **已解决** | 根因: 残留 GPU 进程占显存 (2×552MiB)，清理后正常 |
| **新增**: SCX 编译工具链兼容性 | 未预见 | ✅ **已解决** | clang 18 不支持 32-bit atomics → 避开 UEI |

### 12.2 技术经验总结（Lessons Learned from POC-1）

#### 12.2.1 sched_ext 开发要点

1. **SCX_ENUM_INIT 是必须的**: 不调用则 SCX_DSQ_LOCAL=0, SCX_SLICE_DFL=0, SCX_ENQ_HEAD=0，调度器表面加载成功但实际不工作（state=disabled）
2. **Skeleton 命名规则**: `foo.bpf.o` → skeleton 结构体名 `foo_bpf`（不是 `foo`），头文件 `foo.skel.h`（不是 `foo.bpf.skel.h`）
3. **clang 18 + UEI 不兼容**: UEI 宏需要 32-bit atomics，clang 18 不支持 → 用直接 libbpf API 替代 SCX_OPS_OPEN/LOAD/ATTACH
4. **双 vmlinux.h 架构**: gpu_ext 需要 NVIDIA UVM 类型的 vmlinux.h，sched_ext 需要 sched_ext 类型的 vmlinux.h → Makefile 中 SCX_APPS 用不同 include path
5. **bpf_map__reuse_fd**: 共享 pinned map 必须在 `open()` 后 `load()` 前调用 `bpf_map__reuse_fd(skel->maps.XXX, fd)`

#### 12.2.2 UVM Worker 线程模型

**关键发现**: UVM page fault handler 运行在内核 kworker 线程中，而非触发 page fault 的用户进程。

- `gpu_state_map` 按内存所有者 PID（llama-server 的 tgid）索引
- 但 UVM fault handler 的 `bpf_get_current_pid_tgid() >> 32` 返回 kworker 的 tgid
- sched_ext 看到的是 kworker task，其 `p->tgid` 与 gpu_state_map 中的 key 不匹配
- **解决方案**: 新增 `uvm_worker_pids` map，在 gpu_ext hook 中记录当前执行线程的 tgid，sched_ext 先查此 map

**这一发现也解释了为什么 CPU pinning 对 120B UVM 无效**: `taskset` 只影响用户空间线程，不影响处理 UVM fault 的内核线程。

**⚠️ 后续 profiling 更新（#35 结论修正，2026-02-28）**: nsys SCHED_EVENTS 中 UVM GPU1 BH/KC 无调度记录，但**不是因为它们在中断上下文**。

实际情况（代码确认）：
- UVM GPU1 BH 是 **kthread**（`ppid=2`，`SCHED_OTHER`），完全可被 sched_ext 调度
- nsys 零 sched events 原因：nsys 默认只追踪目标进程的**子进程**，UVM BH ppid=2（kthreadd），非 llama-bench 子进程 → nsys 根本没捕获其调度事件
- E1 profiling 时无 gpu_ext → `uvm_worker_pids` map 为空 → UVM BH kthread 几乎无 fault 要处理 → 大部分时间在睡眠
- `uvm_worker_pids` 机制设计正确，有 gpu_ext 加载时 UVM BH 的 PID 会被记录，sched_ext 可以 boost 它

**结论修正**：sched_ext **能** boost UVM fault handler kthread，taskset 无效的原因是 taskset 只 pin 用户空间线程，但 xCoord 通过 `uvm_worker_pids` 可以做到 taskset 做不到的事。

#### 12.2.3 实验环境注意事项

- llama-server 120B 需要自定义 nvidia_uvm 内核模块（支持 BPF hook），加载前需停止 gdm3 + nvidia-persistenced + 卸载系统 nvidia 模块
- 实验脚本不能用 `sudo bash script.sh` 运行（会导致 root 用户找不到缓存的模型文件），应让脚本内部 `sudo` 单独的 BPF 工具
- **GPU 进程残留是 OOM 首因**：120B 模型 ~60GB via UVM，物理 GPU 32GB，显存已被 UVM 映射填满。任何残留 GPU 进程（哪怕只占 500MiB）都会导致 cuBLAS workspace 分配 OOM
  - 调试/测试后必须运行 `cleanup_gpu.py`
  - 实验脚本开头应验证 GPU 内存为 0MiB（`nvidia-smi --query-compute-apps=pid --format=csv,noheader` 应无输出）
  - 手动启动的 `nohup llama-server &` 进程容易遗忘，是最常见的残留来源
- 使用标准 benchmark 工具 (`server_bench.py` + `common.py`) 可避免服务器生命周期管理问题

### 12.3 Fallback 方案

**Fallback A**: 缩小范围到 "GPU-aware sched_ext scheduler"（单方向：CPU 感知 GPU）
- 工程量减半
- 仍然有 novelty（首个 GPU-aware sched_ext）
- 可投 EuroSys/ATC
- **与 POC-1 工程量重叠 80%**，不浪费前期工作

**Fallback B**: 转向 "LLM-focused gpu_ext"
- 复用已有 gpu_ext 基础设施
- 专注 MoE + KV-cache 场景
- 可投 MLSys/EuroSys

---

## 十三、MSched 启发的设计改进（2026-02-24 分析）

基于对 MSched 论文 (arxiv 2512.24637) 的深入分析，识别出 xCoord 当前设计的 5 个改进方向。

**详细分析文档**: [`docs/reference/msched_analysis.md`](../reference/msched_analysis.md)

### 13.1 改进 1：前瞻性信号替代滞后性 fault_rate

**问题**: 当前 `gpu_state_map` 中的 `fault_rate` 是**滞后指标**——fault rate 高时性能已经在下降。MSched 的核心洞察是等 fault 发生再反应是根本错误的，应该预测和主动准备。

**改进方案**: 在 `gpu_state_map` 中增加前瞻性信号：

```c
struct gpu_pid_state {
    /* 现有 */
    __u64 fault_count;
    __u64 fault_rate;        /* 滞后指标 */
    __u64 eviction_count;
    __u64 used_count;
    __u64 last_update_ns;
    __u32 is_thrashing;

    /* 新增前瞻性信号 */
    __u64 eviction_pressure;   /* eviction_count / used_count 比率 (×1000) */
    __u64 working_set_chunks;  /* 当前活跃 chunk 数量 */
    __u32 migration_pending;   /* 1 = 大规模迁移即将发生 */
    __u32 eviction_trend;      /* 0=stable, 1=rising, 2=spike */
};
```

**实现复杂度**: 低（在现有 chunk_activate/eviction_prepare hooks 中计算并写入）

**预期效果**: sched_ext 可以在 fault storm 到来之前提前 boost CPU priority，而不是等 fault rate 升高后才反应。

### 13.2 改进 2：区分不同类型的 CPU 线程

**问题**: 当前 xCoord 只区分 "UVM worker" vs "普通进程"。MSched 的分析显示 fault 的 96% 开销是 CPU 控制面，暗示不同线程类型对 CPU 的需求不同。

**改进方案**: 细粒度线程分类 + 差异化 boost：

| 线程类型 | 识别方法 | Boost 策略 |
|---------|---------|-----------|
| Fault handler (kworker) | `uvm_worker_pids` map | 最高优先级（控制面瓶颈，30μs→ms 膨胀） |
| Application thread | `gpu_state_map` PID 匹配 | 中等优先级（kernel launch、结果处理） |
| Migration/copy thread | 新增追踪 | 高优先级（DMA setup、TLB 管理） |
| 其他 | 默认 | 正常调度 |

**实现复杂度**: 中等（需要在 gpu_ext hooks 中区分 fault handler vs copy engine context）

### 13.3 改进 3：双向协调增加 CPU→GPU 方向

**问题**: 当前 xCoord 只有 GPU→CPU 方向（gpu_ext 写 fault rate → sched_ext 读取 boost）。MSched 的 Belady OPT 启发：如果 gpu_ext 知道哪个进程当前在 CPU 上运行，可以保护其 GPU 内存不被驱逐。

**改进方案**（对应 POC-3 设计）:

```
sched_ext 写入: cpu_running_pids map (PID → is_on_cpu)
gpu_ext 读取: eviction_prepare 时检查 chunk 所有者是否在 CPU 上运行
  → 在 CPU 上运行的进程: 保护其 GPU 内存（移到 LRU 尾部）
  → 不在 CPU 上运行: 优先驱逐
```

**与 MSched 的类比**: MSched 用 task timeline 决定页替换顺序，xCoord 用 CPU 调度状态决定 GPU 内存保护顺序——原理相同，但 xCoord 是跨子系统的。

**实现复杂度**: 低（POC-3 已有设计，额外 ~100 LOC）

### 13.4 改进 4：Pipeline-Aware CPU Boosting

**问题**: MSched 展示了流水线化迁移（D2H/H2D 并行）可以实现 347x 带宽提升。gpu_ext 的 prefetch 策略也会触发大规模数据迁移，这些迁移的 CPU 线程需要优先级保障。

**改进方案**: gpu_ext 在检测到大规模迁移即将发生时，在 `gpu_state_map` 中设置 `migration_pending=1`：

```
gpu_ext eviction_prepare hook:
  if eviction_count > threshold:
    gpu_state_map[pid].migration_pending = 1

sched_ext enqueue:
  if migration_pending:
    boost ALL threads of this PID (not just workers)
    → 确保 DMA setup、TLB flush、interrupt handling 都得到 CPU 时间
```

**实现复杂度**: 低（仅需在现有 hooks 中加 1 个条件判断 + 1 次 map 写入）

### 13.5 改进 5：BPF 在线访存模式学习

**问题**: MSched 使用离线 NVBit profiling 来预测 kernel 访存模式。这不适合 xCoord（我们的目标是运行时可编程、无需离线分析）。

**长期方向**: 利用 gpu_ext 的 chunk_activate/chunk_used hooks 在线学习访存模式：
- 在 BPF map 中记录每个 kernel 的 chunk activate 序列
- 检测 T1/T2/T3 模板（固定/线性/跨步）
- 将预测的 working set 大小写入 `gpu_state_map`
- sched_ext 根据预测的迁移量提前调整 CPU 优先级

**实现复杂度**: 高（需要 BPF 中的模式识别逻辑），属于论文 Phase 3 或 future work

### 13.6 优先级排序

| 改进 | 优先级 | 工程量 | 预期效果 | 阶段 |
|------|--------|--------|---------|------|
| 13.1 前瞻性信号 | 🔴 高 | ~50 LOC | 提前 boost → 减少 fault storm 延迟 | POC-1 改进 |
| 13.2 线程分类 | 🟡 中 | ~100 LOC | 更精准的 boost → 减少误 boost | POC-2 |
| 13.3 双向协调 | 🟡 中 | ~100 LOC | 保护运行中进程的 GPU 内存 | POC-3 |
| 13.4 Pipeline-aware | 🟢 低 | ~30 LOC | 大规模迁移时全面 boost | POC-2 |
| 13.5 在线学习 | ⚪ 远期 | ~300 LOC | 预测性内存管理 | Phase 3 / future |

### 13.7 对论文定位的影响

MSched 的发表加强了 xCoord 的论文定位：

1. **MSched 作为 related work**: GPU 内部内存调度的 SOTA，证明了 proactive scheduling 的巨大价值
2. **xCoord 的差异化**: "Even with optimal GPU-side scheduling, CPU-side coordination gaps remain"
3. **评估新维度**: 可以在实验中加入 "MSched-style proactive migration + xCoord CPU boost" 的组合场景
4. **新的 motivation 数据**: MSched 的 31.79μs/fault（96% CPU 控制面）直接支撑 xCoord 的核心论点——CPU 调度对 GPU fault 处理至关重要

### 13.8 下一步实验计划

> **已过时**: 原始计划被 §14 结果和 §15 profiling-first 方法论取代。当前下一步见 §0.4。

---

## 十四、POC-1 实验结果（2026-02-27）

### 14.1 Bug 修复

POC-1 之前代码有三个 critical bug，导致 gpu_boosted=1/39021：

1. **select_cpu bypass**: 当 CPU 空闲时，`scx_bpf_dsq_insert` 在 `select_cpu` 中直接分发任务到 SCX_DSQ_LOCAL，导致 `enqueue()` 不被调用，boost 逻辑完全跳过。修复：GPU 任务不在 `select_cpu` 中插入，让 `enqueue()` 处理。

2. **No direct PID matching**: 20B 模型 fit in VRAM，没有 UVM paging，`uvm_worker_pids` map 为空，无法匹配。修复：新增 `gpu_process_pids` map + `-p PID` CLI 参数直接注册 GPU 进程。

3. **DSQ FIFO/PRIQ conflict**: `scx_bpf_dsq_insert`（FIFO）和 `scx_bpf_dsq_insert_vtime`（PRIQ）不能混用同一 DSQ。修复：新增 `GPU_BOOST_DSQ=1`，GPU 任务用独立 FIFO DSQ，普通任务用 SHARED_DSQ PRIQ。`dispatch()` 先 drain GPU_BOOST_DSQ 再 SHARED_DSQ。

### 14.2 实验配置

- **模型**: 20B MoE (gpt-oss-20b-mxfp4, ~10GB, fits in VRAM)
- **GPU**: RTX 5090 32GB
- **CPU stress**: stress-ng on all 24 cores
- **xCoord**: sched_gpu_baseline only (skip eviction_lfu_xcoord, no UVM paging)
- **Benchmark**: vllm bench serve, ShareGPT dataset

### 14.3 Round 2: Low concurrency (concurrency=1, rate=0.2, 50 prompts)

| Metric | Baseline | CPU Stress | xCoord + Stress |
|--------|----------|------------|-----------------|
| **Mean TPOT (ms)** | **3.70** | **4.63 (+25%)** | **3.70 (0%)** |
| P99 TPOT (ms) | 3.91 | 6.75 (+73%) | 4.19 (+7%) |
| Mean TTFT (ms) | 58.6 | 71.1 | 76.2 |
| Throughput (tok/s) | 41.82 | 41.81 | 41.91 |

**结论**: 低并发下，xCoord **完全恢复** per-token latency 到 baseline 水平。TPOT 从 4.63ms（+25%）恢复到 3.70ms（+0%）。P99 TPOT 也从 6.75ms 恢复到 4.19ms（93% 恢复）。

### 14.4 Round 3: High concurrency (concurrency=4, rate=2.0, 50 prompts)

| Metric | Baseline | CPU Stress | xCoord + Stress |
|--------|----------|------------|-----------------|
| Throughput (tok/s) | 287.89 | 266.56 (-7.4%) | 271.90 (-5.6%) |
| Mean TPOT (ms) | 12.39 | 13.50 (+9.0%) | 13.06 (+5.4%) |
| Median TPOT (ms) | 12.24 | 13.23 (+8.1%) | 12.81 (+4.7%) |
| P99 TTFT (ms) | 242 | 348 (+44%) | 281 (+16%) |
| Mean TTFT (ms) | 129.6 | 156.6 (+20.8%) | 152.6 (+17.8%) |
| Requests OK | 50/50 | 48/50 | 50/50 |

**结论**: 高并发下，xCoord 恢复了：
- **Throughput**: 25% of stress-induced loss (266.56 → 271.90 tok/s)
- **TPOT**: 40% of degradation (13.50 → 13.06ms)
- **P99 TTFT**: **63%** of degradation (348 → 281ms)
- **Reliability**: 100% 成功 (50/50 vs 48/50 under stress)

### 14.5 关键发现

1. **xCoord CPU boosting 确实有效**: gpu_boosted 从 1/39021 增加到 12322/468K（2.5% 调度决策为 GPU boost），scheduler 正常工作。

2. **Per-token latency 是最清晰的改善信号**: 低并发 TPOT 完全恢复到 baseline（+0%），高并发恢复 40%。

3. **P99 TTFT 恢复 63%**: 尾部延迟改善显著——说明 xCoord 减少了调度毛刺。

4. **Reliability 改善**: xCoord 场景 50/50 请求成功 vs cpu_stress 场景 48/50。

5. **效果量在 2-7% 范围**: Throughput +2%，TPOT 改善 3-25%（取决于并发）。低并发下效果更显著。

### 14.6 POC-1 的根本问题

**POC-1 只是一个高级 baseline，不具备论文 novelty。**

核心问题：
1. **Blind boost = `nice -20`**: 只要是 GPU 进程就无条件 boost，等价于 `chrt -f 99` 或 `SCHED_FIFO`
2. **GPU 状态信号完全未使用**: `gpu_state_map` 里有 `fault_rate`、`is_thrashing`，但 boost 判断完全没读——不管 GPU 忙不忙都一样 boost
3. **CPU 干扰不真实**: stress-ng 纯 CPU 计算压力，和生产环境无关
4. **场景不对应论文 claim**: 20B fit in VRAM 无 UVM paging → gpu_ext 未参与 → 没有展示 "跨子系统协调"（但验证了 sched_ext 机制本身可工作）
5. **无自适应逻辑**: 不区分 prefill/decode、不区分 fault 强度、不区分 tenant 类型

**POC-1 的价值**: 验证了 sched_ext boost 机制本身 work（TPOT 恢复 25%），作为后续实验的 baseline。

### 14.7 POC-1+ 计划：多 workload 验证 + 算法探索

#### 14.7.1 实验策略

**原则**:
1. 先在单 tenant 场景逐个验证各 workload 的 CPU-GPU 耦合效应
2. 寻找 workload-specific 的最优调度算法
3. 每个实验控制在 ~10min 内完成
4. 最终汇总到 multi-tenant 场景

**单 tenant 实验矩阵**（每 workload 5 个 scenario）:

| # | Scenario | 目的 |
|---|----------|------|
| B0 | Baseline (no stress) | 性能上界 |
| B1 | + stress-ng | 量化 CPU 干扰影响 |
| B2 | + stress + taskset | 证明静态隔离无效 |
| B3 | + stress + blind boost (POC-1) | baseline 算法 |
| B4 | + stress + GPU-aware boost | **新算法** |

#### 14.7.2 候选算法

| 算法 | 描述 | Novelty | 适用场景 |
|------|------|---------|---------|
| **A0: Blind Boost** | GPU PID → 无条件 boost 40ms | 无 (= nice -20) | POC-1 baseline |
| **A1: Fault-Rate Adaptive** | fault_rate 越高 → slice 越长 | GPU 内存状态驱动 CPU 调度 | UVM workloads (GNN, FAISS, 120B) |
| **A2: UVM Worker Escalation** | UVM kworker 优先级 > GPU 用户线程 | 区分 critical path | UVM fault-heavy |
| **A3: Phase-Aware** | prefill 阶段 max boost, decode 阶段 normal | 推理阶段感知 | LLM inference |
| **A4: Anti-Thrashing Throttle** | fault_rate > thrashing → 降低该 tenant CPU time | 反直觉的跨子系统反馈 | Multi-tenant 过度竞争 |
| **A5: Weight-Proportional** | slice = f(fault_rate, gpu_utilization) | 连续自适应 | 通用 |

#### 14.7.3 单 Workload 实验计划

**W1: PyTorch GNN Training (UVM)**
- 配置: 5-8M nodes, UVM mode, 5-10 epochs (~10min)
- 特点: 有 UVM paging → gpu_ext 产生 fault_rate 信号 → 可测试 A1/A2
- Metric: epoch time (s)

**W2: FAISS Vector Search (UVM)**
- 配置: SIFT10M-20M, index build + search
- 特点: 批处理 + 搜索阶段 → 可测试 A3 (phase-aware)
- Metric: add time, search time (s)

**W3: llama.cpp 20B Inference (no UVM)**
- 配置: 20 prompts, concurrency=4 (~5min)
- 特点: 无 UVM，纯 CPU-GPU kernel launch latency
- Metric: TPOT, TTFT, throughput

**W4: Multi-Tenant (W1 + W3 同时)**
- LC: llama.cpp 20B serving
- BE: PyTorch GNN UVM training
- 特点: 自然 CPU 竞争（不需要 stress-ng）+ GPU 内存竞争
- Metric: LC TPOT/TTFT + BE epoch time
- 算法: A4 (tenant-differentiated) + A1 (fault-adaptive)

#### 14.7.4 新实验维度：无 stress 主动优化

> **详见 §15**。核心假设：即使无外部干扰，CPU-GPU 联合调度也可能改善性能。
> 新增 scenario B5: baseline + xCoord boost (无 stress)。B5 结果见 W1 Round 4。

#### 14.7.5 执行顺序

1. **立即**: W1 (GNN UVM) — 5 scenarios + B5 无 stress 优化
2. **然后**: W2 (FAISS UVM) — 同上
3. **然后**: W3 (llama 20B) — 同上
4. **最后**: W4 (Multi-Tenant) — 综合场景

#### 14.7.6 实验记录

**W1: GNN UVM (5M nodes, 5 epochs, warmup=1)**

Round 1 (2026-02-27): B0/B1/B2

| Scenario | Avg Epoch (s) | Median (s) | Duration (s) | vs Baseline |
|----------|--------------|------------|-------------|-------------|
| B0 baseline | 34.39 | 34.38 | 225.7 | - |
| B1 stress | 36.03 | 36.00 | 237.5 | +4.8% |
| B2 stress+taskset | 36.43 | 36.64 | 242.1 | +5.9% |

**发现**: stress 导致 4.8% 退化。taskset 比 plain stress 更差（+5.9%），因为限制了 UVM kworker 可用核心。

Round 2 (2026-02-28): B3/B4（旧版 scheduler，使用 SHARED_DSQ vtime 调度）

| Scenario | Avg Epoch (s) | Median (s) | Duration (s) | vs Baseline |
|----------|--------------|------------|-------------|-------------|
| B3 stress+blind_boost (旧) | **82.10** | 82.25 | 539.4 | **+138.7%** |
| B4 stress+gpuext+boost (旧) | **53.20** | 52.99 | 348.0 | **+54.7%** |

**重大发现: sched_ext overhead 导致灾难性退化！**
- B3 (sched_ext + stress) 比 B1 (CFS + stress) 慢 **2.3x** (82.10 vs 36.03)
- 原因: 所有 24 个 stress-ng 线程都经过 BPF enqueue/dispatch 路径，每次 2 个 hash map 查找 + vtime 计算。CFS 是 O(1)，sched_ext 对非 GPU 任务开销太大
- B4 比 B3 好 35%（53.20 vs 82.10），说明 gpu_ext eviction 策略本身在帮助 UVM 页管理

**修复**: 非 GPU 任务直接走 SCX_DSQ_LOCAL（跳过 vtime 调度），只对 GPU 任务使用 GPU_BOOST_DSQ。这样非 GPU 任务获得类 CFS 的低开销调度。

Round 3 (2026-02-28): 优化后 scheduler（非 GPU 任务 → SCX_DSQ_LOCAL）

| Scenario | Avg Epoch (s) | Median (s) | Duration (s) | vs Baseline |
|----------|--------------|------------|-------------|-------------|
| B0 baseline (re-run) | 34.40 | 34.40 | 225.8 | - |
| B3 stress+blind_boost (优化) | 40.05 | **35.76** | 791.7 | avg +16.4%, **median +4.0%** |

**sched_ext stats**: local=3,876,217 global=985,086 gpu_boosted=23,496 (0.5% of dispatches)

**重要发现**:
- **Median epoch = 35.76s** ≈ B1 stress (36.03s)。优化后 scheduler 在稳态下**不再增加额外开销**！
- **Avg 被第一个 epoch 拖高**: epoch_times = [**57.86**, 35.41, 35.76, 35.37, 35.83]
  - 第 1 epoch 57.86s（sched_ext 启动 + stress-ng 启动 + 初始 UVM 页分配）
  - 第 2-5 epoch 平均 35.59s ≈ CFS+stress 水平
- **Duration 异常**: 791.7s 远超 5×40s=200s。额外 ~590s 可能是 xCoord 启动/GPU PID 等待时间
- **SCX_DSQ_LOCAL 优化成功**: 从旧版 82.10s (2.3x baseline) 降到 median 35.76s (≈baseline)
- **但 avg 40.05s 仍高于 B1 36.03s**: 第一个 epoch 的 sched_ext 冷启动开销需要解决

Round 4 (2026-02-28): B5 无 stress + xCoord boost

| Scenario | Avg Epoch (s) | Median (s) | Duration (s) | vs Baseline |
|----------|--------------|------------|-------------|-------------|
| B5 no stress + boost | 35.44 | 35.40 | 232.3 | **+3.0%** |

**sched_ext stats**: local=1,334,535 global=29,073 gpu_boosted=4,043

**结论: xCoord 主动优化对 GNN UVM 无效，反而增加 3% 开销！**
- epoch_times = [35.40, 35.48, 35.39, 35.67, 35.26] — 无冷启动问题
- sched_ext 本身的 overhead (~3%) > CPU stress 恢复 (~2%)
- 无 stress 时 scheduling decisions 少 (local=1.3M vs stress=3.9M)，但 overhead 仍存在

**W1 GNN 汇总**:

| Scenario | Avg (s) | Median (s) | vs B0 | 说明 |
|----------|---------|-----------|-------|------|
| B0 baseline (CFS) | 34.39 | 34.38 | - | 性能上界 |
| B1 stress (CFS) | 36.03 | 36.00 | +4.8% | CPU 干扰 |
| B2 stress+taskset | 36.43 | 36.64 | +5.9% | 静态隔离更差 |
| B3 stress+boost (旧 vtime) | 82.10 | 82.25 | +138.7% | sched_ext vtime overhead 灾难 |
| B4 stress+gpuext+boost (旧) | 53.20 | 52.99 | +54.7% | gpu_ext 帮助但 vtime 仍在 |
| B3 stress+boost (优化 local) | 40.05 | **35.76** | median +4.0% | 稳态 ≈ CFS+stress |
| B5 no stress+boost | 35.44 | 35.40 | **+3.0%** | 主动优化无效 |

**W1 结论: GNN UVM 不是 xCoord 的理想 workload。**
- CPU stress 仅导致 4.8% 退化 → xCoord 最多恢复 4.8%
- sched_ext 本身开销 ~3% → net benefit ≈ 0
- 原因: GNN 是 compute-bound (22.5GB UVM on 32GB GPU)，UVM 页管理不是瓶颈
- **需要 CPU-GPU coupling 更强的 workload**

下一步: W2 FAISS 或 W3 llama.cpp 20B (POC-0 确认 11.7% coupling)

---

**W2: FAISS SIFT100M UVM (IVF4096,Flat, nprobe=1,4,16)**

Round 1 (2026-02-28): B0 warm cache / B1 stress / B3 多个 scheduler 变体

| Scenario | add (s) | np=1 (s) | np=4 (s) | np=16 (s) | Total (s) | add vs B0 |
|----------|---------|----------|----------|-----------|-----------|-----------|
| B0 baseline (CFS, warm) | 71.38 | 5.36 | 14.43 | 56.87 | 149.62 | - |
| B1 stress (CFS) | 85.42 | 5.38 | 14.07 | 55.18 | 162.33 | **+19.7%** |
| B5 sched_ext (no stress) | 72.60 | 5.27 | 14.33 | 56.39 | 150.18 | +1.7% |
| B3v1 stress+global_DSQ+40ms | 331.56 | 13.45 | 40.82 | 114.87 | 503.48 | **+364.5%** |
| B3v2 stress+local_DSQ+40ms | 190.42 | 6.16 | 14.08 | 66.79 | 279.62 | **+166.8%** |
| B3v3 stress+local_DSQ+default | 142.70 | 5.81 | 16.46 | 89.14 | 256.28 | **+100.0%** |

**关键发现**:
1. **CPU stress 对 FAISS add 阶段影响大**: +19.7%（比 GNN 的 4.8% 大 4 倍）→ FAISS 更适合 xCoord
2. **search 阶段不受 CPU stress 影响**: GPU-bound（nprobe=1/4 几乎无变化）
3. **sched_ext 无 stress 时开销 ≈ 0**: B5 vs B0 仅 +1.7%
4. **sched_ext + stress = 灾难**: 即使是最优的 B3v3（local DSQ + default slice）也比 B1 (CFS) 慢 67%

**sched_ext 变体性能分析**:

| 变体 | 问题 | add vs B1 |
|------|------|-----------|
| B3v1: global GPU_BOOST_DSQ | 24 个 OpenMP 线程全序列化到 1 个全局 DSQ | +288% |
| B3v2: local DSQ + 40ms | 40ms 时间片阻碍 OpenMP barrier 同步 | +123% |
| B3v3: local DSQ + default | is_gpu_task() hash 查找 + vtime_now 缓存行抖动 | +67% |

**根因分析**: sched_ext 在高 CPU contention 下性能远差于 CFS
- **无 stress**: sched_ext 开销仅 +1.7%（可忽略）
- **有 stress (24核)**: sched_ext 额外增加 67-288% 开销
- 原因 1: `running()`/`stopping()` 的全局 `vtime_now` 变量导致 24 核缓存行竞争
- 原因 2: 高 contention 时所有任务走 enqueue 路径（select_cpu 无空闲 CPU 快路径）
- 原因 3: 每次 dispatch 尝试 drain 两个空的全局 DSQ

**B3-minimal (sched_gpu_minimal: 移除 running/stopping/enable/dispatch)**:

| Scenario | add (s) | np=1 (s) | np=16 (s) | Total (s) | add vs B0 |
|----------|---------|----------|-----------|-----------|-----------|
| B3-minimal | **613.04** | 5.25 | 56.92 | 692.12 | **+758.9%** |

sched_ext stats: local=885,023 global=1,030,392 gpu_boosted=208

**发现**: 移除核心 hooks (running/stopping) 反而更差！说明 sched_ext 需要这些 hooks 来正确跟踪任务状态。没有 stopping 回调，任务可能不会正确被抢占，导致 OpenMP barrier 无限等待。

**W2 FAISS 结论**:

sched_ext 在高 CPU contention + 多线程 workload 下引入的 overhead 远超 GPU boost 的收益：
- **无 stress**: sched_ext 开销仅 +1.7%（可忽略）
- **有 stress**: 最好的变体仍比 CFS+stress 慢 67%
- **根本原因**: sched_ext 的 BPF 调度路径在高 contention 下效率远低于 CFS 原生路径
  - CFS 有高度优化的锁和 O(1) 调度（红黑树）
  - sched_ext 每次调度都要经过 BPF function call + map lookups
  - 对多线程 barrier-synchronized workload (OpenMP/FAISS) 影响特别大

**xCoord workload 适配矩阵**:

| Workload 类型 | CPU 线程数 | 同步模式 | sched_ext 影响 | xCoord 效果 |
|--------------|----------|---------|---------------|------------|
| LLM serving (llama.cpp) | 少 (~4-8) | 异步 | **低** | **✅ 有效** (TPOT 完全恢复: 4.44→3.72ms) |
| GNN training (PyTorch) | 中 (~8-12) | GPU-sync | **中** (+3%) | **≈ 0** (stress 仅 +4.8%) |
| Vector search (FAISS) | 多 (24+) | barrier | **高** (+67-759%) | **❌ 有害** |

**下一步**: 聚焦 llama.cpp serving（POC-1 已证明有效），用优化后的 scheduler (local DSQ, no global DSQ contention) 重新验证 + 尝试 GPU-aware 算法

---

**W3: llama.cpp 20B Serving — sched_gpu_serving vs sched_gpu_baseline**

测试 sched_gpu_serving (全局 GPU_BOOST_DSQ + 40ms timeslice) 是否比 sched_gpu_baseline (local DSQ + default timeslice) 更适合 LLM serving。

Round 1 (2026-02-28): sched_gpu_serving, c=1, 20 prompts

| Scenario | TTFT (ms) | TPOT (ms) | Throughput (tok/s) | 请求 |
|----------|-----------|-----------|-------------------|------|
| B0 baseline (CFS) | 53.11 | 3.68 | 49.33 | 20/20 |
| B1 stress (CFS) | 63.25 | 4.71 | 49.29 | 20/20 |
| B3 stress+serving | 67.97 | 4.95 | 49.19 | 20/20 |

vs POC-1 R2 sched_gpu_baseline (c=1, 50 prompts):

| Scenario | TTFT (ms) | TPOT (ms) | Throughput (tok/s) |
|----------|-----------|-----------|-------------------|
| B0 baseline | 58.6 | 3.70 | 41.8 |
| B1 stress | 71.1 | 4.63 | 41.8 |
| B3 stress+gpu_aware | 76.2 | **3.70** | 41.9 |

**sched_gpu_serving 发现**:
- serving TPOT: 4.95ms (比 stress 的 4.71ms 还差)
- serving 的非 GPU 任务走 SCX_DSQ_LOCAL → 缺少 SHARED_DSQ 的全局公平调度 → 无法实现优先级反转

Round 2 (2026-02-28): sched_gpu_baseline (修复版), c=1, 20 prompts

**BUG**: R4 发现 `SCX_ENQ_DSQ_PRIQ` flag 与 kernel `enq_flags` 冲突 → scheduler crash (runtime error: invalid enq_flags 0x200000000000009)。修复：移除 `SCX_ENQ_DSQ_PRIQ`，使用普通 FIFO。

R5 (修复后): sched_gpu_baseline = GPU→GPU_BOOST_DSQ(40ms) + 非GPU→SHARED_DSQ(FIFO)

| Scenario | TTFT (ms) | TPOT (ms) | Throughput (tok/s) | 请求 |
|----------|-----------|-----------|-------------------|------|
| B0 baseline (CFS) | 50.5 | 3.71 | 49.33 | 20/20 |
| B1 stress (CFS) | 61.3 | 4.44 | 46.80 | 19/20 |
| B3 stress+gpu_aware | 70.6 | **3.72** | 49.52 | 20/20 |

**TPOT 完全恢复！** 3.72ms ≈ baseline 3.71ms (stress 下 4.44ms → +20% 退化完全消除)

Scheduler 统计:
- local=8,643 (idle CPU fast path, 预热阶段)
- global=360,731 (非 GPU 任务 via SHARED_DSQ)
- gpu_boosted=12,028 (GPU 任务获得优先调度)

**核心架构洞察**:
1. **SHARED_DSQ (全局 FIFO) 是关键**: 非 GPU 任务（stress-ng）进入全局队列，GPU 任务在 GPU_BOOST_DSQ 获得优先 dispatch
2. **SCX_DSQ_LOCAL 不行**: sched_gpu_serving 把非 GPU 任务放 local DSQ → 每个 CPU 上 stress-ng 线程独占本地队列 → GPU 优先级反转不生效
3. **enq_flags 传递**: 必须保留 kernel 传入的 `enq_flags`，不能 OR 额外的 DSQ_PRIQ flag

**TTFT 退化**: xcoord TTFT (70.6ms) > stress (61.3ms)。原因: SHARED_DSQ 全局队列引入额外 dispatch 延迟。TPOT 恢复但 TTFT 代价略增（可接受，因为 TTFT 在绝对值上仍<100ms）。

**Bug fixes**:
1. **pkill self-kill**: `cleanup_xcoord_stale()` 中 `pkill -f sched_gpu_serving` 匹配了 Python 脚本自身命令行 → 改为 `pkill -x` (精确进程名匹配)
2. **SCX_ENQ_DSQ_PRIQ crash**: kernel enq_flags + PRIQ flag 冲突 → 移除 PRIQ flag
3. **CWD issue**: background Bash commands 继承上一次 cd 后的目录 → 使用绝对路径

---

**R5 c=4: sched_gpu_baseline (GPU_BOOST_DSQ + SHARED_DSQ), 50 prompts**

| Scenario | TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | Throughput (tok/s) | 请求 |
|----------|-----------|--------------|-----------|-------------------|------|
| B0 baseline (CFS) | 133.5 | 258.1 | 11.97 | 295.6 | 50/50 |
| B1 stress (CFS) | 167.4 | **420.8** | 12.92 | 270.9 | **47/50** |
| B3 stress+xcoord | 160.1 | **305.7** | 12.91 | 276.8 | **50/50** |

**c=4 改善**:
- TTFT: -4.4% (167.4→160.1ms)
- **P99 TTFT: -27.3%** (420.8→305.7ms) — 尾延迟大幅降低
- Throughput: +2.2% (270.9→276.8 tok/s)
- **Reliability: 47/50→50/50** — 6.4% more requests completed

**TPOT (c=4)**: 12.92→12.91ms (≈ 0%) — 高并发时 TPOT 不受影响（GPU compute 已饱和，CPU调度非瓶颈）

---

**综合 llama.cpp 20B 结果 (sched_gpu_baseline R5)**:

| Concurrency | Metric | Baseline | Stress | xCoord | 改善 |
|------------|--------|----------|--------|--------|------|
| c=1 | TPOT (ms) | 3.71 | 4.44 | **3.72** | **完全恢复** |
| c=1 | Throughput | 49.3 | 46.8 | **49.5** | **完全恢复** |
| c=1 | Reliability | 20/20 | 19/20 | **20/20** | 恢复 |
| c=4 | P99 TTFT (ms) | 258.1 | 420.8 | **305.7** | **-27.3%** |
| c=4 | Throughput | 295.6 | 270.9 | **276.8** | +2.2% |
| c=4 | Reliability | 50/50 | 47/50 | **50/50** | **恢复** |

**核心 takeaway**:
1. c=1: **TPOT 和 throughput 完全恢复到 baseline** → CPU调度是低并发时的关键瓶颈
2. c=4: **P99 尾延迟大幅降低 (-27%), reliability 完全恢复** → 高并发时主要帮助减少尾部 jitter
3. TTFT 略有退化 (SHARED_DSQ 全局队列开销) → 可接受的 trade-off

---

## 十五、现阶段问题总结 + 下一步方向

### 15.0 现阶段 5 大核心问题（2026-02-28 总结）

基于 POC-0/POC-1 全部实验数据，xCoord 当前面临以下高层次问题：

1. **算法无 novelty**: Blind boost = `nice -20` = `chrt -f 99`。`gpu_state_map` 里有 `fault_rate`、`is_thrashing` 信号，但调度决策完全没读——不管 GPU 忙不忙都一样 boost。论文 claim "GPU memory awareness as first-class CPU scheduling signal"，但实际实现是 PID-based priority boost，任何人用 `chrt` 一行命令就能做到。

2. **场景局限**: blind boost 只对 LLM serving（少线程、异步）有效。GNN (+3% overhead)、FAISS (+67~759% 退化) 在 blind boost 下表现差。但 blind boost 失败不等于 GPU-informed scheduling 失败——FAISS 的 barrier straggler 问题恰恰可能受益于更精细的调度（精确 boost straggler 线程而非所有线程）。

3. **无主动优化证据**: 目前证明的效果仅限于 "从 stress-ng 干扰中恢复"。无 stress 时 GNN B5 反而增加 3% overhead。还没有证据表明 xCoord 能**超越** CFS baseline。但注意：(a) stress-ng 模拟的 CPU 争用在多租户/co-located workload 场景中真实存在；(b) 无 stress 场景尚未被充分探索——当前 baseline 是 blind boost，更精细的 GPU-state-driven 调度可能在无 stress 时也有收益（如减少 tail latency、优化 CPU-GPU overlap）。

4. **缺乏 profiling 基础**: 所有算法都基于假设（"fault handler 需要优先级"）而非测量。不知道 CFS 在 GPU 关键路径上实际浪费了多少时间、浪费在哪里。没有数据支撑的优化 = 盲猜。

5. **sched_ext 固有开销天花板**: BPF 调度路径本身增加 1.7-3% baseline overhead（无 stress 时测量）。优化收益必须超过这个 overhead 才有 net positive。对高并发/barrier 同步 workload，overhead 远超收益。

**后续工作必须直面这 5 个问题**: Profiling-first（解决 #4）→ 针对性算法设计（解决 #1）→ 多场景验证（解决 #2）→ 无 stress 主动优化（解决 #3，详见 §15.5.2）→ 最小化 overhead（解决 #5）。

**特别注意**: 问题 #3 (无 stress 主动优化) 是决定论文 novelty 层级的关键分水岭。如果 xCoord 只能从 stress-ng 恢复，那它只是 "priority boost 工具"；如果能在无干扰时也改善 tail latency 或多 workload 协调，则是真正的 "cross-subsystem coordination"。

---

### 15.1 核心思路

POC-1 证明了 sched_ext 能在 stress-ng 干扰下恢复性能。但更有价值的问题是：
**即使没有外部干扰，CPU-GPU 联合调度本身能否提升 GPU workload 的性能？**

如果答案是 yes，这比 "防御干扰" 的 novelty 强得多：
- 不需要人为制造干扰场景来证明价值
- 任何 GPU workload 都能受益 → 更广的适用性
- 类比 DPDK：不只是防网络干扰，而是主动降低网络延迟

### 15.2 为什么默认 CFS 对 GPU workload 不是最优的？

> 基础 insight 见 §2.1 和 §8.2。以下是从 CFS 调度视角的具体分析。

Linux CFS 追求 CPU 公平性，但 GPU workload 有独特的 CPU-GPU 交互模式：

1. **UVM page fault 路径延迟**
   - GPU 触发 page fault → UVM GPU1 BH kthread 处理 → 迁移数据 → GPU 恢复
   - UVM GPU1 BH 是 **SCHED_OTHER kthread**（ppid=2），完全可被 CFS/sched_ext 调度
   - CFS 无法区分 GPU-critical UVM BH kthread 与普通系统 kthread → 这正是 xCoord 的切入点
   - `uvm_worker_pids` map：gpu_ext hook 中记录正在处理 fault 的 UVM BH PID，sched_ext 据此 boost
   - **E1 profiling 零 sched events 的真正原因**：nsys 只追踪目标进程子进程；且 E1 无 gpu_ext → 无 fault → UVM BH 大部分时间在睡眠

2. **GPU kernel launch 路径**
   - GPU 计算需要 CPU 提交 kernel → 如果 CPU 线程被 CFS 排到后面，GPU 空转
   - 对 serving 场景（低并发），CPU→GPU 提交延迟直接体现在 TPOT

3. **CPU-GPU pipeline bubble**
   - GPU 计算和 CPU 数据搬运可以 pipeline
   - 但 CFS 不知道 GPU 在等 CPU，无法主动调度相关 CPU 线程
   - 结果：GPU 完成计算后等 CPU 准备下一批数据，产生 bubble

4. **NUMA/PCIe 亲和性**
   - GPU 通过 PCIe 连接特定 NUMA node
   - CFS 可能把 GPU 相关线程调度到远端 NUMA → 跨 NUMA 内存访问 + PCIe 延迟增加
   - sched_ext 可以把 GPU 线程 pin 到 GPU 所在 NUMA node 的 CPU

### 15.3 具体优化策略

#### S1: UVM fault handler 优先调度（120B UVM 场景）⚠️ 机制可行，但效果待验证
- **方法**: gpu_ext 在 fault hook 中写入 `uvm_worker_pids`，sched_ext boost UVM GPU1 BH kthread
- **机制确认**: UVM GPU1 BH 是 SCHED_OTHER kthread（ppid=2），完全可被 sched_ext 调度
- 之前标记为"证伪"的原因**错误**：nsys 零 sched events 是 nsys 追踪范围限制 + E1 无 gpu_ext loaded
- **E1/E2 中为何无效（正确原因）**：PCIe 带宽（~20 GB/s）是瓶颈，不是 UVM BH 的调度延迟；fault handling 仅占 per-token 时间 6%（0.8ms），即使 boost 也无法改变 PCIe 限速
- **待验证**: 120B **serving** 场景（E5）— per-request latency 敏感，UVM BH 调度延迟可能更显著

#### S2: GPU kernel launch 亲和调度（20B 无 UVM 场景）
- **目标**: 减少 CPU→GPU kernel submission 延迟
- **方法**: boost llama.cpp server 进程 + NUMA pinning
- **测试**: 20B 模型（无 UVM），无 stress，对比 CFS baseline
- **预期**: TPOT 略有改善（可能 <5%，因为 20B 无 UVM 时 CPU 不是主要瓶颈）

#### S3: CPU-GPU pipeline 协调（120B UVM 场景）
- **目标**: 最大化 GPU compute 和 CPU data migration 的 overlap
- **方法**: 当 GPU 在计算时，主动 boost 下一批数据的 prefetch/migration CPU 线程
- **需要**: gpu_ext 的 chunk_activate/deactivate hook 提供 GPU 状态 → sched_ext 据此调度
- **这是 xCoord 最独特的 contribution**: 真正的跨子系统协调，而不只是 priority boost

#### S4: 系统线程降级
- **目标**: 减少系统后台线程对 GPU 路径的干扰
- **方法**: 非 GPU 相关系统线程（journald, snapd, crond 等）在 GPU 活跃时降低调度优先级
- **不需要 stress-ng**: 系统本身就有几十个后台线程在跑
- **测量**: 先 profiling 看 CFS baseline 下系统线程对 GPU 线程的 preemption 频率

### 15.4 方法论：Profiling-First（替代盲目 baseline 实验）

> **反思**: 之前 §15.4 的 E1-E4 实验计划是"先跑 baseline，希望有改善"。
> 正确方法是：**先用 profiling 工具量化 CPU 调度在 GPU 关键路径上的实际开销，
> 然后针对测量到的瓶颈设计算法**。

#### 15.4.1 Phase A: 性能分析（Profiling）

**目标**: 量化回答 "CFS 在 GPU 关键路径上浪费了多少时间？浪费在哪里？"

**工具矩阵**:

| 工具 | 测量目标 | 适用场景 |
|------|---------|---------|
| `gpu_sched_trace` + `analyze_preemption_impact.py` | GPU 线程被抢占的频率和代价 | 20B serving (kernel launch delay) |
| `chunk_trace` + `analyze_overhead.py` | UVM fault 的 CPU 控制面延迟 | 120B UVM (fault handling) |
| `perf sched latency` / `perf sched record` | CPU 调度延迟分布 | 所有场景 |
| `nsys profile` | CUDA kernel timeline + CPU 线程活动 | 精确的 GPU idle gap 分析 |
| `perf stat -e context-switches,cpu-migrations` | 上下文切换和 CPU 迁移频率 | 基础指标 |

**Profiling 实验**:

**P1: 120B UVM — 量化 fault handler 调度延迟**
1. 运行 `llama-bench -m 120B` + 同时 `sudo chunk_trace > /tmp/trace.csv`
2. 运行 `perf sched record -p $(pgrep -f llama-bench)` 捕获调度事件
3. 分析: `perf sched latency` 查看 UVM kworker 的 wait-time 和 max-delay
4. 关键问题: fault handler 被调度的 P50/P99 延迟是多少？占 fault 总延迟的比例？
5. 如果 >10% fault 延迟来自 CPU 调度 → S1 (fault handler 优先调度) 有价值

**P2: 20B Serving — 量化 kernel launch 调度延迟**
1. 运行 `llama-server 20B` + `sudo gpu_sched_trace > /tmp/sched.csv`
2. 运行 `analyze_preemption_impact.py /tmp/sched.csv` 计算抢占代价
3. 关键问题: GPU idle 中有多少是因为 CPU 线程未及时提交 kernel？
4. 如果 preemption penalty P99 > 1ms → S2 (kernel launch 亲和调度) 有价值

**P3: 120B UVM — GPU idle gap 分析**
1. 运行 `nsys profile llama-bench -m 120B` (如果 UVM 兼容)
2. 或用 `chunk_trace` 计算 fault-to-fault 间隔中 GPU 空闲时间
3. 关键问题: GPU 在等 CPU 还是 CPU 在等 GPU？pipeline bubble 有多大？

#### 15.4.2 Phase B: 算法设计（基于 Profiling 数据）

根据 Phase A 测量结果，选择最有潜力的优化策略：

| Profiling 发现 | 对应策略 | 实现方式 |
|---------------|---------|---------|
| fault handler wakeup 延迟 >10µs | S1: wakeup priority boost | enqueue() 中 UVM kworker → 最高优先级 |
| kernel launch 间 preemption penalty >1ms | S2: preemption guard | GPU 线程在 running() 时设 SCX_SLICE 更大 |
| GPU idle gap 中 CPU 在调度非 GPU 线程 | S3: pipeline-aware | gpu_ext 设 migration_pending → sched_ext boost |
| 系统线程 preemption 频率 >1% | S4: 系统线程降级 | 非 GPU 线程 → SHARED_DSQ 低优先级 |
| 所有项都 <阈值 | 停止 | CFS 对此 workload 已足够好，转向其他场景 |

**原则**: 没有 profiling 数据支撑的优化 = 盲猜。每个策略必须有测量到的瓶颈作为 motivation。

#### 15.4.3 Phase C: 实现 + 评估

1. 在 `sched_gpu_baseline.bpf.c` 中实现针对性算法
2. 用**同样的 profiling 工具**验证瓶颈是否被消除
3. 端到端 benchmark 测量实际性能改善
4. 比较: CFS baseline → blind boost (POC-1) → targeted algorithm → 量化 GPU awareness 的增量价值

#### 15.4.4 实验矩阵：每场景 Profile → Design → Evaluate

**总览**:

| # | Workload | 配置 | Profiling 目标 | Baseline 结果 | 状态 |
|---|----------|------|---------------|--------------|------|
| E1 | llama.cpp 120B (UVM) | batch, 无 stress | P1: fault handler 调度延迟 | CFS 已够好 (batch 无竞争) | ✅ profiled |
| E2 | llama.cpp 120B (UVM) | batch + stress-ng | P1+P3: fault + pipeline | batch 吞吐量仅 -1.4% (GPU pipeline 隐藏) | ✅ profiled |
| E3 | vLLM 30B (UVM) | serving + stress-ng | P1: serving tail latency | P99 +235%, baseline -62% (算法问题) | ✅ profiled |
| E4 | llama.cpp 20B (no UVM) | serving, 无 stress | P2: kernel launch 抢占 | CFS 已够好 (无竞争时) | ✅ profiled |
| E5 | llama.cpp 120B (UVM) | **serving** + stress | P1+P3: UVM serving latency | **未测试** — 最值得探索 | ⏳ |

**E1: 120B + sched_ext only** ✅ Profiling 完成

- Baseline: CFS, pp=137.59, tg=49.65 tok/s (更新值)
- **Profiling 工具**: `nsys profile` (perf 不可用, chunk_trace kprobe 符号不匹配)
- **Profiling 结果 (2026-02-28)**:

  **Timeline** (nsys 有 ~19% overhead: pp=111.16, tg=40.25 tok/s):
  - Model loading: 0-78.9s (~79s, 62.8GB H2D weight transfer)
  - pp512: 79.8-83.3s (3.56s, 1922 GPU kernels)
  - tg128: 83.7-88.1s (4.42s, 1971 GPU kernels)

  **pp512 phase**:
  - GPU: 95%+ utilization, UVM H2D=609K ops/32.2GB @20.1 GB/s, D2H=1.9K ops/3.9GB
  - CPU: **99.9% ON-CPU**, 仅 50 次调度切换, OFF-CPU P50=9.9us
  - CPU 调度 overhead: 5.1ms / 3550ms = **0.14%**

  **tg128 phase** (per-token: 15.4 kernels/token, 258 MB H2D + 258 MB D2H):
  - GPU: **99.8% utilization**, 仅 7ms idle gaps (DMA 与 compute 完全 overlap)
  - CPU: **99.7% ON-CPU**, 109 次调度切换, OFF-CPU P50=167us, P99=324us
  - CPU 调度 overhead: 14ms / 4210ms = **0.33%**

  **结论: 在单 workload 无干扰环境下，CPU scheduling overhead 极小（<0.4%）**
  - 主线程几乎不被抢占 (99.7-99.9% ON-CPU)
  - 调度延迟 P99 < 430us，已足够快
  - GPU 是瓶颈: PCIe 带宽 (~20 GB/s) 限制了 UVM 迁移速率
  - UVM 的 async migration 已将 DMA 与 GPU compute 有效 overlap
  - **注意**: 此结论仅适用于 batch (llama-bench) + 单 workload 场景。serving、多 workload 共存、或有其他系统负载时情况可能不同

- **设计决策**: E1 的 blind boost 在此场景无改善空间，但不排除更精细的调度策略（如 NUMA 感知、kworker 亲和性）的潜在收益
- **转向**: E2 (gpu_ext + sched_ext + stress-ng) 测试有干扰时的恢复能力
- llama-bench pp=512 tg=128, 10 runs geometric mean

**E2: 120B + gpu_ext + sched_ext (核心实验)** ✅ Profiling 完成

- Baseline: gpu_ext alone (always_max + cycle_moe), pp=138.08, tg=47.98 tok/s
- **Profiling 结果 (2026-02-28)**:

  **stress-ng 24-core (matrixprod) 影响测试**:
  | 配置 | pp | tg | 变化 |
  |------|-----|----|------|
  | gpu_ext, 无 stress | 138.08 | 47.98 | baseline |
  | gpu_ext + stress-ng 24c | 137.37 | 47.29 | **-0.5% / -1.4%** |

  **结论: stress-ng 对 120B UVM batch 影响可忽略** — "batch 联合优化"假设不成立
  - 120B batch 是 **GPU-bound**，不是 CPU-bound
  - GPU 95-99.8% 利用率，PCIe 带宽 (~20 GB/s) 是瓶颈
  - CPU 抢占被 GPU pipeline 隐藏: 即使 OFF-CPU 增加 9x (14ms→123ms), tg 仅降 1.4%
  - **限定**: 此结论仅适用于 **batch 模式** + **gpu_ext 已加载**；120B **serving** 模式（对 per-request latency 敏感）未测，不能外推

  **nsys 调度对比** (tg128 phase):
  |  | No stress | stress-ng |
  |--|-----------|-----------|
  | CPU ON-CPU | 99.7% | 97.2% |
  | OFF-CPU P99 | 324us | 3055us |
  | tg throughput | ~49.65 | ~47.29 |

  **"CPU-GPU 联合调度比单侧优化更好" — 在 batch 单 workload 无干扰下不成立**:
  - gpu_ext 单独已接近最优 (PCIe 瓶颈)
  - sched_ext blind boost 在此场景无额外价值（因为 CFS 已给足 CPU 时间）
  - 原因: 无干扰时 GPU pipeline 已有效隐藏 CPU scheduling 延迟
  - **但**: 多 workload 共存、serving 场景、或有系统级 CPU 负载时，结论可能不同

- **设计决策**: 放弃 E2 "batch + blind boost 联合优化超越单侧" 路线
- **转向**: (1) serving 场景（CPU 交互更频繁）; (2) 更精细的 GPU-state-driven 调度（非 blind boost）; (3) 多 workload 共存

**E3: vLLM 30B UVM**
- vLLM 从未测过 xCoord
- **先 Profile**: 同 P1 方法，量化 vLLM 的 UVM fault handler 调度延迟
- Qwen3-30B-A3B-FP8 (~29GB on 32GB) 有 UVM paging
- serve_bench.py --mode uvm 已就绪
- 验证 xCoord 不只对 llama.cpp 有效
- **在 E1/E2 最佳策略确定后做**

**E4: 20B proactive (补充)** ✅ Profiling 完成

- 20B fit in VRAM，无 UVM
- Baseline: pp=9827, tg=362 tok/s (with nsys: pp=9686, tg=357)
- **Profiling 结果 (2026-02-28)**:

  **Per-token breakdown** (tg128, cudaGraphLaunch span=358.6ms):
  - Per-token total: P50=**2.79ms**, P90=2.80ms, Max=4.06ms
  - GPU compute: ~0.40ms (14%)
  - CPU work (sampling/logits/prep): ~2.40ms (**86%**)
  - cudaGraphLaunch: ~22us (negligible)
  - GPU idle between tokens: 0us (kernels fully packed within each graph)

  **CPU scheduling during tg128**:
  - **零有效抢占**: 仅 3 次调度切换, 全部 <12.4us
  - ON-CPU: 100% during inference
  - CFS 已足够好（无竞争时）

  **观察: 无 CPU 竞争时, CFS 调度已充分 — baseline (blind boost) 无额外收益**
  - CPU work 主导 per-token time (86%), 但 CFS 在无竞争时不会抢占 GPU 线程
  - POC-1 数据: stress-ng 下 TPOT 3.71→4.44ms (+20%), baseline 恢复至 3.72ms
  - 注意: 这只说明"无竞争时 blind boost 无用"，不能推广为"sched_ext 对 20B 无价值"
    - 20B serving c=4+ 时请求自身产生 CPU 竞争 → baseline 仅恢复 63%
    - 更精准的 GPU-informed 调度可能改善 c=4+ 结果

### 15.5 vs POC-1 的区别 + Profiling 关键发现

| 维度 | POC-1 (stress recovery) | 主动优化 (Profiling-First) |
|------|------------------------|--------------------------|
| 干扰源 | 人为 stress-ng (24核) | 无 / 系统自身后台线程 |
| 目标 | 恢复到 baseline | **超越 baseline** |
| 方法 | 盲目 boost + 跑 benchmark | **先 profiling → 再设计算法** |
| Novelty | 防御性 → 增量贡献 | 主动协调 → **系统性贡献** |
| 适用性 | 需要干扰存在 | **任何 GPU workload** |
| 关键 workload | 20B (无 UVM) | **120B (重度 UVM)** |
| 关键引擎 | llama.cpp only | llama.cpp + **vLLM** |

**Profiling-First 数据汇总 (2026-02-28)**:

**Phase A 完成: E1/E2/E3/E4 全部 profiled。**

**Profiling 事实 (不含推断)**:

- E1 (120B UVM batch, 无 stress): CPU ON-CPU 99.7%, OFF-CPU P99=430us, GPU 99.8% utilized
- E2 (120B UVM batch + stress-ng 24c): 吞吐量变化 -0.5%/-1.4% (gpu_ext 已加载)
  - 注意: 仅在 batch 模式下测试, **未测 120B serving**; gpu_ext 的 prefetch 可能掩盖了部分影响
- E3 (vLLM 30B UVM serving + stress-ng 24c): **P99 TPOT +235%** (615→2062ms), median 不变
- E4 (20B no-UVM serving, 无 stress): CPU ON-CPU 100%, per-token 86% CPU work

**sched_gpu_baseline（盲目 boost）全场景表现**:

| 场景 | stress 影响 | baseline 效果 | 失败根因 |
|------|------------|--------------|---------|
| 20B serving c=1 + stress | TPOT +20% | ✅ 完全恢复 | N/A — 最简单 case，单线程 CPU-bound |
| 20B serving c=4 + stress | P99 +大 | ⚠️ 仅恢复 63% | 多线程竞争，盲目 boost 不精准 |
| vLLM 30B UVM + stress | P99 +235% | ❌ 反而 -62% | 多进程 boost 错误 + 全局 DSQ 开销 |
| FAISS batch + stress | 严重 | ❌ +67~759% 更差 | 全局 DSQ 开销 + 多线程 barrier |
| 120B UVM batch + stress | 仅 -1.4% | 未测 | (batch 模式下 GPU pipeline 隐藏 CPU 延迟) |
| GNN batch | 中等 | ⚠️ 中性 | 中等线程数 |

**baseline 的核心缺陷（不是方向问题，是实现问题）**:

1. **boost 对象错误**: 按 TGID 匹配整个进程所有线程，但真正需要 boost 的是特定 kworker/critical-path 线程
2. **全局 DSQ 开销**: 所有非 GPU 线程（含 stress-ng）都经过全局 SHARED_DSQ dispatch → 破坏 CFS per-CPU 局部性
3. **无 GPU 状态感知**: gpu_state_map 的 fault_rate/is_thrashing 字段在 enqueue 中未使用，实质上 = `nice -20`
4. **FIFO 不公平**: GPU_BOOST_DSQ 是 FIFO，idle 线程排在前面阻塞有实际工作的线程
5. **5 秒超时 ≈ 永远 boost**: UVM worker 超时太长，失去 reactive 特性

**⚠️ 关键区分: baseline 失败 ≠ 方向不可行**

以下场景不能因为 baseline 失败就判定"没有机会":
- **FAISS + stress**: baseline 的全局 DSQ 开销导致灾难性结果，但 FAISS barrier 同步的特性意味着一个 straggler 线程被延迟 → 所有线程等待 → 精准 boost straggler 是高价值场景。如果非 GPU 线程走 SCX_DSQ_LOCAL（零额外开销），sched_ext overhead 可以很低。
- **120B UVM batch + stress**: E2 仅在 batch 模式下测试且 gpu_ext 已加载。120B **serving** 场景（并发请求，per-request latency 敏感）未测试，不能排除 CPU 调度影响。
- **vLLM 30B + stress**: P99 +235% 证明 CPU 调度确实严重影响 serving tail latency。baseline 失败是因为 boost 了错误的线程并引入了全局 DSQ 开销，不是因为"boost 无用"。

### 15.5.1 超越 Baseline: xCoord 调度器设计方向

**核心问题: 什么 GPU 信号能让 CPU 调度器做出比 CFS 更好的决策？**

**Baseline 本质**: `nice -20` 的 BPF 版本 — 不利用任何 GPU 运行时状态。

**xCoord 需要解决的 4 个问题**:

| # | 问题 | Baseline 做法 | xCoord 目标 |
|---|------|-------------|------------|
| 1 | Boost 谁 | 整个进程 (TGID) | 仅 GPU fault handler / critical-path thread (TID) |
| 2 | 何时 Boost | 始终 (5s 超时 ≈ 永远) | 仅当 GPU 发出需求信号时 (fault_rate > threshold) |
| 3 | 非 GPU 线程 | 全局 SHARED_DSQ | SCX_DSQ_LOCAL (保留 CFS per-CPU 局部性, 零开销) |
| 4 | Boost 强度 | 固定 (FIFO + 40ms) | 按 GPU 需求自适应 (vtime fairness + 动态 timeslice) |

**候选方案**:

**方案 1: Per-Thread Targeted + Local DSQ（修复 baseline 核心缺陷）**
```
GPU侧 (gpu_ext): uvm_worker_pids 记录 UVM GPU1 BH kthread PID（fault 时自动记录）
                  gpu_process_pids 记录用户空间进程 PID（-p 参数注册）
CPU侧: boost 精确 TID/PID（UVM BH kthread + EngineCore 等关键线程）
        非 GPU 线程 → SCX_DSQ_LOCAL (零额外开销，保留 CFS 局部性)
```
- 直接修复 baseline 4 个核心缺陷
- UVM GPU1 BH IS 可被 sched_ext boost（kthread，SCHED_OTHER）
- `uvm_worker_pids` 机制设计正确，需 gpu_ext 同时加载才能工作
- 预期: vLLM/FAISS 场景不再恶化（SCX_DSQ_LOCAL 消除全局 DSQ 开销），UVM serving 场景有 recovery

**方案 2: Fault-Rate Reactive Scheduling（GPU 状态驱动）**
```
GPU侧: gpu_state_map[pid].fault_rate 实时更新, is_thrashing 标志
CPU侧: enqueue 检查 fault_rate:
        if is_thrashing → boost (GPU 正在 thrashing)
        else → 正常 CFS (不干预)
```
- 利用 gpu_ext 已有的 fault_rate 信号 (shared_maps.h 已定义)
- 空闲时零开销，thrashing 时精准干预
- **Novelty**: CFS/EEVDF 看不到 GPU fault rate — 这是 cross-subsystem 的独有信号

**方案 3: Pipeline-Aware Adaptive（最有 novelty）**
```
GPU侧: 检测 GPU kernel launch gap > threshold → "GPU 正在等 CPU"
        写入 gpu_stall_count
CPU侧: stall_count increasing → boost CUDA launch thread + 动态 timeslice
        stall_count stable → 正常调度
```
- GPU performance metric 直接驱动 CPU 调度决策
- 对 FAISS barrier 场景特别有价值: straggler thread 导致 GPU stall → 立即 boost

**建议实现路径**: 方案 1 → 方案 2 → 方案 3（渐进式，每步都可独立评估）

**最值得验证的场景（按 novelty 排序）**:
1. **vLLM 30B UVM + stress**: P99 +235% 证明问题真实, baseline 失败证明需要更好算法
2. **FAISS + stress**: barrier 同步 + straggler boost 是独特场景, 需要验证精准 boost 效果
3. **120B UVM serving + stress**: 未测试, 每 token 有 page fault → GPU-informed scheduling 持续有价值
4. **20B serving c=4+**: baseline 仅恢复 63%, 有提升空间

### 15.5.2 无 Stress 场景的机会分析

**核心问题**: xCoord 只能从人造干扰中恢复，还是在干净环境中也有价值？

**已有 profiling 数据中的线索**:

| 场景 | 无 stress profiling 发现 | 潜在机会 |
|------|--------------------------|---------|
| E1 (120B batch) | 主线程 ON-CPU 99.96%，Max OFF-CPU 324µs；nsys 中 UVM BH 零 sched events（nsys 追踪范围限制，非 BH context） | **机制可行但效果受限** — UVM BH 是 kthread 可被 boost，但 PCIe 是瓶颈（fault handling 仅占 6%），boost 无法改变 PCIe 限速 |
| E3 (vLLM 30B) | 无 stress P99 TPOT = 615.6ms vs median = 166.2ms; 稳态 EngineCore ON-CPU **99.8%**, OFF-CPU P99=254µs, Max=11ms | **CPU 调度不是 P99 来源** — 稳态无任何 >100ms OFF-CPU 事件，P99 来自 GPU/UVM 侧 |
| E4 (20B serving) | CPU ON-CPU 100% | 极小 — 模型 fit in VRAM，无 UVM |

**无 stress 但有机会的 4 个方向**:

**1. 自然 tail latency reduction（不需 stress-ng）** ❌ 已验证：vLLM 30B 无效
- ~~vLLM 30B 无 stress 时 P99 已是 median 的 3.7x → CFS 调度 jitter 本身导致尾延迟~~
- **实测否定**: 稳态 EngineCore ON-CPU=99.8%, OFF-CPU P99=254µs, Max=11ms, 零个>100ms事件
- TPOT P99=615ms 的来源是 GPU/UVM 侧（per-request UVM 冷启动、vLLM continuous batching）
- CPU 调度不是瓶颈，xCoord 无法通过调度降低此 P99
- **TTFT 机会保留**: TTFT P99=60s 有 UVM loading OFF-CPU，但这是服务启动一次性成本

**2. 多 workload 共存（最真实的生产场景）**
- 生产 GPU 服务器不会只跑一个任务
- 两个 GPU workload 共享机器时，CPU 线程自然竞争 — 不需 stress-ng
- 例: LC serving (vLLM) + BE training (GNN) 同时运行
- xCoord 可根据 GPU fault rate 区分 LC/BE，优先保障 LC 的 CPU 时间
- **这是最有论文价值的场景**: multi-tenant GPU resource coordination

**3. UVM BH kthread 优先级（UVM 场景特有）** — 机制可行，效果待 E5 验证
- UVM GPU1 BH **是 kthread（SCHED_OTHER）**，可被 CFS/sched_ext 调度 ✅
- 之前标注"已验证无效"是错误推断：nsys 零 sched events 是追踪范围限制 + E1 无 gpu_ext
- **E1/E2 batch 中无效的真正原因**：PCIe 带宽是瓶颈，fault handling 仅占 per-token 6%；且 always_max prefetch 减少了 fault 数量，BH kthread 大多在睡眠
- **E5 (120B serving)** 是关键验证：serving 场景每个新请求都有 UVM 冷启动，BH kthread 调度延迟直接影响 TTFT/TPOT；`uvm_worker_pids` + sched_ext boost 在此场景可能真正有效

**4. CPU-GPU phase-aware 调度**
- GPU compute 和 CPU DMA 可以 overlap
- CFS 不知道 GPU 当前处于 compute phase 还是 stall-on-fault phase
- gpu_ext 的 fault_rate 信号可以区分这两个 phase
- **⚠️ 注意**: "stall-on-fault" 阶段的 fault handler 在 BH context（不可调度），phase-aware 调度的价值在于调度**用户空间线程**（降低 GPU compute 时的 CPU 优先级，避免抢占 DMA 相关线程）
- 不依赖 stress-ng：多 workload 共存时，phase-aware 分配 CPU 时间更有意义

**优先级排序（按 novelty + 可行性，已更新）**:
1. **多 workload 共存** — novelty 最高，不依赖 stress-ng，CPU 竞争自然存在
2. **E5: 120B UVM serving + xCoord** — UVM BH kthread 可被 boost，serving 对 latency 敏感，`uvm_worker_pids` 机制已就绪；**这是 xCoord 最重要的未验证场景**
3. **vLLM 30B + stress (完整 xCoord)** — P99 +235% 已知，baseline 失败是实现问题
4. ~~**Tail latency reduction (no stress)**~~ — #34 已证伪：vLLM 稳态 ON-CPU 99.8%，P99 来自 GPU 侧

**nsys profiling 数据位置**:
- E1 (120B no-stress): `/tmp/xcoord_profiling/120b_nsys.sqlite`
- E2 (120B + stress): `/tmp/xcoord_profiling/e2_120b/120b_stress_nsys.sqlite`
- E4 (20B no-stress): `/tmp/xcoord_profiling/e4_20b/20b_nsys.sqlite`
- E3 (vLLM 30B UVM): `/tmp/xcoord_profiling/e3_vllm/vllm_nsys.sqlite`

### 15.6 E3: vLLM 30B UVM Profiling + Stress Impact (2026-02-28)

**✅ Profiling 完成 + Stress 测试完成**

**nsys Profiling 关键发现**:
- GPU utilization: **3.7%** (极低，30K kernels in 24s span, 仅893ms busy)
- 两阶段模式:
  - 模型加载 (t=8-14s): 31GB H2D UVM page migration, 每秒5-7GB
  - 推理 (t=14-25s): 几乎零 H2D → 模型已常驻VRAM
- EngineCore 主线程 (TID 1164649) 调度:
  - 模型加载阶段: ON-CPU 90.8%, OFF-CPU P50=121us, P99=4.2ms
  - Token 生成阶段: **ON-CPU 99.8-99.9%**, OFF-CPU P50=14us, P99=189us
- CUDA API: cudaLaunchKernel 27.5K次 (770ms), cudaMallocManaged 39K次, cudaFree 34.6K次
- **CUPTI buffer 限制**: 仅捕获前25s GPU活动 (全部232s sched events)

**Stress-ng 24核 Impact**:

| Metric | No Stress | Stress-24 | Delta |
|--------|-----------|-----------|-------|
| Output throughput (tok/s) | 49.59 | 45.35 | **-8.6%** |
| TPOT Mean (ms) | 224.9 | 336.3 | +49.5% |
| TPOT Median (ms) | 166.2 | 160.6 | -3.4% (不变) |
| **TPOT P99 (ms)** | **615.6** | **2062.2** | **+235%** |
| TTFT Mean (ms) | 33169 | 37616 | +13.4% |
| TTFT Median (ms) | 30510 | 30778 | +0.9% |

**关键发现: P99 tail latency 灾难性恶化 (+235%)，但 median 不变**
→ stress-ng 造成偶发性严重调度延迟，拖垮 P99

**sched_gpu_baseline 盲目 boost 测试 (仅 -p PID，无 gpu_ext)**:

| Metric | Stress-only | + sched_gpu_baseline |
|--------|------------|-------------------|
| Output throughput (tok/s) | 45.35 | **17.13 (-62%)** |
| TPOT Mean (ms) | 336.3 | **762.5 (+127%)** |
| TPOT P99 (ms) | 2062.2 | **5047 (+145%)** |
| TTFT Mean (ms) | 37616 | **45916 (+22%)** |

**sched_gpu_baseline 在 E3 的失败分析**:

1. **测试条件不完整**: 只跑了 `sched_gpu_baseline -p PID`，**未启动 eviction_lfu_xcoord (GPU侧BPF)** → `uvm_worker_pids` map 为空，cross-subsystem feedback 未激活
2. **吞吐量崩溃 (-62%)**: 全局 SHARED_DSQ 让 24 个 stress-ng 线程 + vLLM 内部线程全部经过 sched_ext dispatch，开销远超收益
3. **P99 恶化 (+720%)**: GPU_BOOST_DSQ(FIFO) 让 vLLM 的 idle 线程排在 stress-ng 前面，但这些 idle 线程没有实际工作，浪费 timeslice
4. **boost 对象错误**: `-p PID` 只匹配了 vLLM main process (PID 1175828)，EngineCore (另一个进程) 未被 boost；而真正的 UVM fault handler 是内核 kworker 线程

**E3 核心结论**:
- stress-ng 对 vLLM serving P99 影响显著 (+235%) → CPU 调度确实影响 GPU serving tail latency
- 但 sched_gpu_baseline 的盲目 boost 反而恶化 → 需要更精准的 GPU-informed 调度策略
- 详细设计方向见 §15.5.1

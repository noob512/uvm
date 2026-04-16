下面给你一套**非常具体、可操作**的实验设计清单：不引入任何新抽象（不谈 lease/principal/contract），只围绕你最初关心的现象——**CPU 调度/干扰如何影响 GPU 应用（GPU 吃不满、空转、尾延迟变差、分布式 straggler 放大）**——一步步做“可证伪”的验证。

我会先给出**通用实验框架**（保证可重复与可归因），然后列出一组从 microbench 到真实训练/推理的实验，每个实验都包含：**目标/假设、变量、步骤、采集指标、如何判定现象成立、现象不成立时该怎么改实验**。

---

## 0) 通用实验框架（所有实验都建议遵守）

### 0.1 固定环境（减少噪声）

* 固定 GPU：锁定 GPU 时钟（如支持 `nvidia-smi -lgc`）、关闭 auto-boost（可选）
* 固定 CPU：使用 `performance` governor，避免频率波动（或把它当实验变量，见后文）
* 关闭/固定后台服务：尤其是会产生周期性 CPU/IO/网络中断的服务
* 固定 NUMA：尽量先在单 socket 或明确绑核/绑内存（后面再做 NUMA 变量）

### 0.2 你必须采的两类“最小指标”

**A. CPU 侧调度指标（核心）**

* 目标线程（通常是 GPU “喂活”的线程、同步线程、通信线程）：

  * runnable→running 延迟分布（p50/p95/p99）
  * 被抢占次数、迁移次数
* 系统层面：

  * 每核 runqueue 长度、CPU 利用率
  * softirq/irq 时间（尤其同核干扰）

**B. GPU 侧利用与“空洞”指标（核心）**

* GPU utilization（粗指标）
* **GPU timeline 的 idle gaps**：空转片段的数量与总时长（最关键）

> 仅看 `nvidia-smi` 的 util 不够，必须要能看到时间线的“空洞”，否则你无法证明“CPU 延迟 → GPU 空转”的链条。

### 0.3 工具建议（不强制，但要能实现指标）

* GPU 时间线：Nsight Systems（强烈建议），或 CUPTI 采集活动；再不行用更粗的 sampling 也可，但证据弱
* CPU 调度延迟：

  * `perf sched record` + `perf sched latency`（快速上手）
  * ftrace/trace-cmd 的 `sched_switch`/`sched_wakeup`
  * eBPF/bpftrace 统计某 PID/TID 的 runnable→running 延迟（你做 eBPF 方向的话更自然）
* 软中断/中断：`/proc/interrupts`、`softirqs`、`perf top`、ftrace irq tracepoints

### 0.4 时间对齐（可选但非常加分）

如果你想做严谨因果图（OSDI 级别），要做 CPU 与 GPU 时间线对齐：

* Nsight Systems通常能同时采 CPU/GPU 事件并对齐，这会大幅减少工作量
* 如果自己采集，要做 offset/drift 校准（后面我也给了一个校准实验）

---

## 1) 实验 1：最基本的“CPU runnable 延迟会造成 GPU 空转吗？”

这是整个研究的第一块基石：如果这个都证明不了，后面都没意义。

### 假设

**H1**：当 GPU workload 是“host 提交受限/频繁提交”的时候，目标 CPU 线程的 runnable→running 延迟上升，会引入 GPU timeline 的 idle gaps（空洞增多/变长）。

### 工作负载（microbench）

选择一个**提交密集型** microbench（越简单越好）：

* 反复 launch 很短的 kernel（几十微秒级）
* 或反复做很小的 GEMM / elementwise，确保 kernel 很短、launch 频繁
* 关键点：GPU 本身不应该是瓶颈；瓶颈应该在 CPU 提交/调度

### 变量（treatment）

制造可控的 CPU 调度延迟（只动一个旋钮）：

* 在 feeder 线程同核上跑一个 CPU hog（busy loop）
* 通过 `nice`/`cgroup cpu.max` 调整 hog 的优先级/配额
* 或直接把 feeder 线程绑在一个“拥挤核”上（多线程 background load）

### 步骤（建议）

1. **控制组**：只跑 microbench，记录 GPU timeline 与 CPU runnable 延迟分布
2. **处理组**：加 CPU hog（从 0% → 20% → 50% → 80% duty cycle），每档跑 1–3 分钟
3. 所有档位重复 3 次（看方差）

### 采集指标

* feeder 线程 runnable→running 延迟 p50/p95/p99
* GPU idle gaps：总空转时长、空转次数、最大空转时长
* 端到端吞吐（iteration/s）

### 判定现象“成立”的标准

* runnable→running 的 p99 明显上升时，GPU idle gaps 的总时长/次数也显著上升
* 吞吐下降与 idle gaps 增加高度一致（可做相关系数或回归）

### 如果现象“不成立”，怎么调整实验？

通常是 microbench 不够“提交受限”，GPU 仍在长 kernel 里忙：

* 把 kernel 变短、launch 更频繁（更小 grid、更少工作）
* 确保 kernel 之间没有人为 sleep（否则你自己制造了 gap）
* 确保没有 GPU-side blocking（同步太频繁也会干扰判断）

---

## 2) 实验 2：`sched_yield`/频繁短睡眠会放大调度影响吗？

很多训练/推理 runtime 里关键线程会频繁 yield/sleep（比如等待队列、polling/backoff）。你要验证“频繁 yield 的线程在 CFS 下是否更容易被延后，进而让 GPU 断粮”。

### 假设

**H2**：对于频繁 `yield/sleep` 的 feeder/同步线程，CFS 更容易把它判定成“非急迫”，导致 runnable 延迟尾部更差，从而 GPU gap 增加。

### 工作负载

在实验 1 的 microbench 上增加一个“可调的 yield 行为”：

* 每提交 N 次 kernel 后 `sched_yield()` 或 `nanosleep(几十微秒)`
* N、sleep 时长作为可调参数

### 变量

* yield 频率：每 1 次 / 10 次 / 100 次
* sleep 时长：0 / 10µs / 50µs / 200µs
* 同时可叠加 CPU hog（模拟混部）

### 指标与判定

* 在相同 CPU hog 条件下，yield 越频繁，feeder runnable p99 越差，GPU gap 越明显
* 如果 yield 频率变化对 runnable p99 不敏感，说明你的系统干扰不够或 sleep 太小被合并了（需要调大 sleep/干扰）

---

## 3) 实验 3：CPU 频率策略对 GPU 吞吐/空转的影响（“低频核喂不饱 GPU”）

这不是调度本身，但在真实系统里经常和调度耦合：关键线程跑在低频核上会导致提交/同步变慢。

### 假设

**H3**：当 feeder/同步线程受 CPU 频率影响明显时，降低 CPU 频率会引入 GPU idle gaps 或显著降低吞吐，即使 CPU 仍有空闲。

### 工作负载

仍用实验 1 的提交密集 microbench（最敏感）。

### 变量

* governor：performance vs powersave
* 限频：把最大频率 clamp 到 60%/80%（或用 cpupower 限制）
* 把 feeder 线程绑到固定核，确保对比可重复

### 指标

* 每次提交耗时（可记录用户态时间戳差）
* runnable delay 是否变化（频率变化不一定改变 runnable delay，但会改变“running 时推进速度”）
* GPU idle gaps 与吞吐变化

### 判定

* 频率下降导致吞吐下降，并在 GPU 时间线上出现更密集/更长的 gap（尤其是短 kernel 场景）

---

## 4) 实验 4：IRQ/softirq 噪声是否会“抢走关键线程时间片”，造成 GPU gap？

这是很多生产系统的隐形杀手：网络/存储中断、softirq 处理会打断用户态关键线程。

### 假设

**H4**：把大量 IRQ/softirq 处理固定在 feeder 所在 CPU 核，会显著拉高 runnable 延迟/抢占时间，从而增加 GPU idle gaps；把 IRQ 迁走（irq affinity）会改善。

### 工作负载

实验 1 microbench 或一个小推理 workload（只要是提交/同步敏感）。

### 干扰注入

* 产生稳定可控的网络中断：例如固定速率的收发包（iperf/自发包工具）
* 或磁盘 IO 中断：固定速率读写

### 变量（关键）

* IRQ affinity：

  * **方案 A**：把相关 IRQ 固定到 feeder 所在核
  * **方案 B**：把 IRQ 固定到另一个核（或隔离核）
* 其他条件保持不变

### 指标

* feeder 核的 irq/softirq 时间占比
* feeder runnable 延迟与被抢占时间
* GPU idle gaps 与吞吐

### 判定

* A 明显更差、B 明显更好，并且 ledger（哪怕是手工对齐）能看到 gap 前发生 irq burst

---

## 5) 实验 5：GPU “完成检测/同步”路径是否受 CPU 调度影响（polling vs blocking）

这个实验更贴近真实框架：很多应用会频繁 `cudaEventQuery`（polling）或 `cudaEventSynchronize`（blocking）。CPU 调度会影响“完成被看见”的时间。

### 假设

**H5**：在 CPU 竞争/配额限制下，polling 更容易耗尽时间片并被 deschedule，导致完成检测延迟尾部更差；blocking 可能更稳定（或相反，取决于实现）。

### 工作负载

* GPU kernel 固定运行时间（例如 200µs/1ms/5ms 三档）
* CPU 线程循环：launch kernel → 等待完成（polling 或 blocking）→ 记录一次迭代延迟

### 变量

* 等待方式：polling / blocking / polling-then-blocking（混合）
* CPU 干扰：cgroup cpu.max 限额（例如 50% quota）或 CPU hog
* 时间片变化（可选）

### 指标

* 完成检测延迟（从 kernel 实际完成到 CPU 继续下一步的时间）
* 总迭代延迟的 p95/p99
* runnable delay 分布

### 判定

* 在 CPU 紧张时，某种等待方式显著更差，并且差异来自“CPU 没及时运行来观察完成”

---

## 6) 实验 6：多进程 GPU 竞争下，“CPU 调度 + GPU 队列”会不会互相放大？

这是验证“只优化 CPU 或只优化 GPU 不够”的前置证据，但你现在不谈抽象也能先验证现象：多进程下，一个 latency-sensitive job 会被另一个 BE job 放大尾部，CPU 调度只是其中一环。

### 假设

**H6**：当 GPU 上存在长 kernel 或大量 BE work，占用 GPU queue，会导致另一个 job 的短 kernel 延迟/吞吐明显变差；如果同时再给短 job 的 CPU feeder 加干扰，会出现更严重的放大（双重瓶颈）。

### 工作负载

* Job A（latency-sensitive）：短 kernel 高频提交（同 MB1）
* Job B（best-effort）：长 kernel 或大矩阵 GEMM

### 变量

* B 的占用强度：占用 20%/50%/80% GPU 时间
* A 的 CPU 干扰：无/轻/重
* 亲和性：A 的 feeder 是否绑定到拥挤核

### 指标

* A 的每次迭代延迟（p95/p99）
* A 的 GPU queue wait（用 GPU timeline 看）
* A 的 CPU runnable delay

### 判定

* 单独 GPU 竞争会使 A 变差；叠加 CPU 干扰会明显更差
* 这为后续“跨域机制必要性”提供事实基础（但此处不必提出机制）

---

## 7) 实验 7：Unified Memory / 内存迁移导致的 GPU stall，是否被 CPU 干扰放大？

这实验验证“memory 域的服务链也受 CPU 调度影响”，并且会表现成 GPU kernel 时间拉长或出现长 stall。

### 假设

**H7**：当 workload 触发大量 UM page fault / migration 时，CPU 干扰会拉长 fault 服务时间，导致 GPU stall 变多、尾部更长。

### 工作负载（MB3）

* 分配 managed memory（超过 GPU 显存或制造冷热反复）
* kernel 访问大量页面以触发 fault/migration
* 记录每次迭代 kernel 时间与同步时间

### 变量

* CPU 干扰：无/有（CPU hog 或 cpu.max quota）
* 预取策略：不预取 vs `cudaMemPrefetchAsync`（作为对照，确认确实是 fault/migration 导致）
* 访问模式：顺序 vs 随机（随机更容易产生差尾）

### 指标

* kernel runtime 分布（p99）
* fault/migration 相关 CPU 时间（能采到就采：相关内核线程/kworker/uvm 的运行时间；采不到就用间接指标如 stall 时间）
* GPU timeline 上的 stall/空转（Nsight Systems）

### 判定

* 在 CPU 干扰下，fault-heavy workload 的尾部显著更差；prefetch 能改善（说明根因确实在 migration）

---

## 8) 实验 8：NUMA 远端访存/跨 socket 迁移对 GPU 喂活的影响

你前面多次提到 NUMA，这里先不用抽象，只验证“跨 socket 的 CPU 控制线程与其数据/页不一致”会造成性能问题。

### 假设

**H8**：把 feeder/数据加载线程放在远端 NUMA，会显著降低提交/准备速度，导致 GPU idle gaps 或 step time 变差。

### 工作负载

* 小训练或推理 pipeline（包含明显 CPU 数据准备/拷贝/launch）
* 或 MB1 + 加入 CPU-side buffer fill

### 变量

* 绑定：

  * 情况 A：线程与内存都在 GPU 所在 socket（本地）
  * 情况 B：线程在远端 socket、内存在远端或随机
* 可用 `numactl --cpunodebind --membind` 或 cgroup cpuset

### 指标

* CPU side 数据准备时间
* GPU idle gaps
* LLC miss、remote NUMA 访存（如果能用 perf 采到）

### 判定

* 远端绑定显著变差，且变差主要表现为“GPU gap 增加/吞吐下降”

---

## 9) 实验 9：分布式训练里的 straggler 放大（一个 rank 的 CPU 干扰会拖全局）

这是把“单机 GPU gap”升级成“集群级别损失”的关键实验。

### 假设

**H9**：在同步型分布式训练（或频繁 allreduce/barrier）中，只对一个 rank 注入 CPU 调度干扰，就能显著放大全局 step time 尾部（straggler）。

### 工作负载

* 小规模 DDP（2–8 GPU）即可，关键是有同步点
* 模型可选轻量（ResNet/Transformer 小版），保证可重复

### 变量

* 只在 rank 0 注入干扰：CPU hog / quota / IRQ 噪声
* 干扰强度：轻/中/重
* 对照：在所有 rank 平均注入同等干扰（看是否更“均匀”）

### 指标

* 每个 rank 的 step time
* allreduce 时间
* 全局 step time 的 p95/p99
* 注入 rank 的 feeder runnable delay 与 GPU gap

### 判定

* 单点注入导致全局 tail 显著变差（远大于平均意义的影响），并且链条可解释（rank 0 的 CPU 延迟 → rank 0 GPU lag → allreduce 等待）

---

## 10) （可选但非常有力）时间对齐校准实验：把“因果链”从相关性做成证据

如果你想把论文做得特别硬（而不是“看起来像”），建议补一组对齐校准：

### 方法

* 在 CPU 侧记录“提交时间戳”与“完成回调时间戳”
* 在 GPU 侧记录 kernel start/end（Nsight/CUPTI）
* 做一组很短的 ping-pong（CPU→GPU→CPU），拟合 offset/drift
* 估计对齐误差分布

### 价值

你就可以在论文里画出这样的证据链图（非常有说服力）：

* **某次 GPU idle gap 发生前 X µs，feeder 线程处于 runnable 但未运行**
* **gap 长度与该 runnable 延迟高度一致**

---

# 怎么用这些实验去“验证设计正确/需要改进”？

你现在还不引入抽象，但你完全可以用这些实验做一个**迭代闭环**：

1. **先用实验 1–2 验证：问题确实存在，且可稳定复现**
2. **用实验 4–7 把根因类型分清楚：CPU 调度、IRQ、GPU 竞争、memory fault、NUMA**
3. **当你提出某个改动（比如只做 feeder 线程绑核/提高优先级/迁走 IRQ/改变等待方式）**

   * 回到对应实验验证：改善是否只在一种根因下有效？是否会把瓶颈迁到另一域？
4. **如果出现“改善 GPU gap 但 tail 更差”**

   * 多半是你触发了实验 7/8 那类 memory/NUMA 问题
5. **把每个失败模式沉淀成一个“可复现实验脚本 + 指标面板”**

   * 这会变成你后续写论文的 robustness/limitations 章节素材

---

# 我建议你从哪 3 个实验先做（最省时间、信息量最大）

如果你现在就要开始动手，我建议按这个顺序：

1. **实验 1（CPU runnable 延迟 → GPU idle gaps）**：先把最核心链条坐实
2. **实验 4（IRQ/softirq 抢占）**：生产系统里最常见、且最容易做 A/B 对照
3. **实验 7（UM fault/migration）**：一旦你将来做“联合调度”，这会是最容易翻车的域

---

如果你愿意，我可以把其中任意 2–3 个实验进一步写成“实验脚本级别”的操作步骤（包含：绑核/配额/注入方式、采集命令、数据如何处理、以及最终图应该怎么画），这样你可以直接开始跑并产出第一轮可发表的现象图。

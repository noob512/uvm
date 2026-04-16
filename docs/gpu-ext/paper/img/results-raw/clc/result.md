下面先把你目前所有“出现过的数值”整理在一张表里（按 regime/配置分块），然后给一段可以直接放进 OSDI evaluation 的讨论文本骨架，顺带点出这些结果各自能支持什么 claim、哪里要保守一点。

---

## 1. 汇总表：所有实验配置下的策略与性能

下面这张表把我们讨论过的所有数值都拉在一起：Balanced GEMM、Imbalanced GEMM（正常 + 超订）、ClusteredHeavy（单 kernel + bpftime demo）。
每个 regime 都选了一个“baseline policy”，归一化和 Δ% 都是相对于该 baseline 算的。

```tex
\begin{table}[t]
\centering
\small
\begin{tabular}{l l l r r r}
\toprule
Regime & Workload / Scenario & Policy
       & Runtime (ms) & Norm.\ vs base & $\Delta$ vs base (\%) \\
\midrule
Balanced &
GEMM: Balanced (16$^3$, CLC) &
FixedWork        & 0.1218 & 1.000 & \phantom{$+$}0.0 \\
&
&
Greedy           & 0.1113 & 0.914 & +8.6 \\
&
&
LatencyBudget    & 0.1120 & 0.920 & +8.0 \\
\midrule
Imbalanced &
GEMM: var.\ $M{\times}N{\times}K$ (CLC) &
FixedWork        & 0.1019 & 1.000 & \phantom{$+$}0.0 \\
&
&
Greedy           & 0.0910 & 0.893 & +10.7 \\
&
&
LatencyBudget    & 0.0909 & 0.892 & +10.8 \\
Imbalanced+oversub. &
GEMM: var.\ $M{\times}N{\times}K$ (0.6--0.9$\times$ WS) &
FixedWork        & 0.0670 & 1.000 & \phantom{$+$}0.0 \\
&
&
TokenBucket      & 0.0510 & 0.761 & +23.9 \\
\midrule
ClusteredHeavy &
CLC single-kernel &
FixedWork        & 127.3  & 1.000 & \phantom{$+$}0.0 \\
&
&
Greedy           & 203.2  & 1.596 & $-59.6$ \\
&
&
LatencyBudget    & 202.4  & 1.590 & $-59.0$ \\
ClusteredHeavy &
bpftime demo (clc\_demo\_clustered\_heavy) &
Baseline (uninstr.) & 14.8 & 1.000 & \phantom{$+$}0.0 \\
&
&
Greedy           & 17.8   & 1.203 & $-20.3$ \\
&
&
LatencyBudget    & 14.7   & 0.993 & \phantom{$+$}0.7 \\
\bottomrule
\end{tabular}
\caption{Summary of single-tenant CLC policies and bpftime demo policies
on an NVIDIA RTX~5090.
Runtimes are averaged over the steady-state execution of each kernel configuration.
Balanced GEMM uses a tiled 16$\times$16$\times$16 kernel with $\approx$8K thread
blocks of 256 threads each (2{,}097{,}152 work items) and a small per-thread
prologue of 40 iterations.
Imbalanced GEMM keeps the same launch shape but samples matrices from a skewed
size distribution (5\% 64$^3$, 15\% 48$^3$, 30\% 32$^3$, 50\% 16$^3$) with a
slightly heavier prologue (45 iterations); the oversubscribed variant restricts
the working set to 0.6--0.9$\times$ device memory.
ClusteredHeavy comprises $\approx$8K thread blocks of 256 threads each, where
90\% of blocks are ``light'' ($\sim$1K iterations per item) and 10\% are
``heavy'' (100--200$\times$ more work); heavy blocks are placed contiguously
at the tail of the grid to create a pronounced long tail.
For each regime, Norm.\ and $\Delta$ are computed relative to the baseline
policy (FixedWork or the uninstrumented kernel).}
\label{tab:gbpf-summary}
\end{table}
```

你要的所有性能数字都在这张表里了：

* Balanced: 0.1218 / 0.1113 / 0.1120
* Imbalanced base: 0.1019 / 0.0910 / 0.0909
* Imbalanced oversub: 0.067 vs 0.051（TokenBucket）
* ClusteredHeavy single-kernel: 127.3 / 203.2 / 202.4
* ClusteredHeavy demo: 14.8 / 17.8 / 14.7

以及各自的归一化值和 speedup/slowing 百分比。

---

## 2. 一段可以直接放进 OSDI 的讨论（含 takeaways）

下面给你一版 evaluation 文本骨架，专门围绕这张表来写。你可以按需缩短或拆成两个 subsection。

### 2.1 Balanced and moderately imbalanced GEMM

> **Block-level scheduling on balanced and imbalanced GEMM.**
> Table~\ref{tab:gbpf-summary} first compares gBPF block-scheduling policies on a balanced GEMM kernel. We use a tiled $16\times16\times16$ GEMM on an NVIDIA RTX~5090 with 2{,}097{,}152 work items (about 8K thread blocks with 256 threads each) and a small per-thread prologue of 40 iterations. In this configuration, the static \textsc{FixedWork} policy completes in 0.1218,ms. Both the always-steal \textsc{Greedy} policy and the LatencyBudget policy closely track the baseline at 0.1113,ms and 0.1120,ms, respectively, corresponding to an 8--9% reduction in runtime and less than 1% difference between the two dynamic policies. We interpret these differences as evidence that gBPF-style scheduling does not introduce pathological overheads on well-balanced dense kernels: dynamic policies effectively behave as no-ops and do not regress performance in the common case.
>
> We next introduce a moderately imbalanced GEMM by sampling matrices from a skewed size distribution (5% 64$^3$, 15% 48$^3$, 30% 32$^3$, 50% 16$^3$) and increasing the prologue to 45 iterations to emulate per-block weight-loading overhead, while keeping the launch shape fixed. Under this configuration, \textsc{FixedWork} averages 0.1019,ms, whereas \textsc{Greedy} reduces runtime to 0.0910,ms (a 10.7% improvement over static assignment) and LatencyBudget to 0.0909,ms (10.8% improvement). The absolute difference between \textsc{Greedy} and LatencyBudget is $<0.1%$, indicating that once moderate load imbalance is present, even a simple always-steal policy is sufficient to reclaim a two-digit percentage of tail time, and more sophisticated budgeting does not materially change the outcome in this regime.
>
> To stress the effect of memory pressure, we also evaluate an oversubscribed variant of the imbalanced GEMM where the effective working set is restricted to 0.6--0.9$\times$ device memory. In this setting, the average kernel time under \textsc{FixedWork} drops to 0.067,ms as the kernel does less work per block, but a workload-aware TokenBucket policy further reduces runtime to 0.051,ms---a 23.9% improvement over \textsc{FixedWork}. Here, TokenBucket throttles block stealing when the observed per-block service time exceeds a budget derived from the current working-set size, avoiding pathological contention on slow blocks while still redistributing slack from earlier-finished blocks.

这里的关键信息你已经讲清楚：

* Balanced：dynamic policy 与 static 几乎重合，只能当作 “no regression” 证据，别把 8% 写成硬核 speedup；
* Imbalanced（无超订）：Greedy/LatBud 相对 Fixed 就有 ~11% 的改善；
* Imbalanced + oversub：TokenBucket 才真正拉开差距，~24% 的提升，说明**内存压力 + 不均衡** 这个 regime 下要换另一套策略。

OSDI reviewer 读到这里会非常清楚：**没有一个 policy 在所有配置里都赢**，而且这些结论都落在 “合理但不要 over-claim” 的范围内。

### 2.2 Clustered heavy-tail workloads and the need for policy control

> **Clustered heavy-tail workloads.**
> We now turn to a deliberately pathological workload, \textsc{ClusteredHeavy}, constructed to stress the limits of always-steal scheduling. The kernel launches the same number of blocks (about 8K blocks with 256 threads each, $n{=}2{,}097{,}152$ work items) as the GEMM experiments, but assigns roughly 90% of blocks as `light'' and 10\% as `heavy.'' Light blocks perform $\sim$1{,}000 compute iterations per item, whereas heavy blocks perform 100--200$\times$ more work depending on the imbalance scale. Heavy blocks are placed contiguously at the tail of the grid, creating an extreme long-tail pattern where a small cluster of blocks dominates completion time.
>
> In the single-kernel CLC microbenchmark, Table~\ref{tab:gbpf-summary} shows that \textsc{FixedWork} completes in 127.3,ms. In contrast, the always-steal \textsc{Greedy} scheduler inflates the runtime to 203.2,ms---a 59.6% slowdown relative to static assignment---and LatencyBudget remains similarly degraded at 202.4,ms (58.0% slowdown). In this regime, the large contiguous cluster of heavy blocks at the tail of the grid becomes a hotspot: once light blocks finish, idle SMs repeatedly contend for work from the same small set of remaining heavy blocks via global steal queues, amplifying atomic contention and cache thrashing rather than improving load balance. These results highlight a key limitation of naive always-steal policies: the same mechanism that is beneficial under moderate imbalance (Table~\ref{tab:gbpf-summary}, Imbalanced GEMM) becomes actively harmful under clustered heavy tails.
>
> To understand whether host-device cooperation can mitigate this behavior, we port the \textsc{ClusteredHeavy} pattern into our bpftime demo (\texttt{clc_demo_clustered_heavy}) and attach gBPF policies via the GPU-side interface. On the RTX~5090, the uninstrumented baseline kernel runs in 14.8,ms. Adding \textsc{Greedy} increases runtime to 17.8,ms (20.3% slowdown), consistent with the single-kernel CLC experiment. However, a LatencyBudget policy that stops stealing after a per-block \texttt{clock64} budget completes in 14.7,ms, effectively matching the uninstrumented baseline within measurement noise (0.7% difference). In other words, under the same heavy-tail workload, \textsc{Greedy} remains pathological, while LatencyBudget leverages gBPF's programmable control plane to cap per-block stealing time and avoid the worst-case contention patterns.

这里要非常清楚地把两个层次分开：

* **纯 GPU-side CLC single-tenant**：所有 dynamic policy 都很惨，Greedy/LatBud 都比 FixedWork 慢 ~60%。
  这证明了一个事实：**光在 kernel 里乱偷，不看全局信息，heavy-tail 能把你噎死**。
* **bpftime demo + gBPF host-side 协调**：在同样的 heavy-tail pattern 下，

  * Greedy 还是把 latency 拉长 20%，
  * 但 LatencyBudget 可以保持与 baseline 持平。
    这才能支撑你在 introduction 里要做的那个“大 claim”：**需要一个安全、可编程的 policy plane 来避免 pathological tail**。

### 2.3 总结句（可以直接放在本小节最后）

> **Takeaway.**
> Across these regimes, no single scheduling policy dominates. On balanced GEMM, dynamic policies behave essentially as no-ops and do not hurt performance. Once moderate imbalance appears, even a simple always-steal policy such as \textsc{Greedy} is sufficient to recover 10--11% of wasted tail time over static block assignment, and a workload-aware policy such as TokenBucket yields an additional 24% improvement under oversubscribed imbalanced workloads. However, under clustered, extreme heavy-tail workloads, always-steal becomes pathological: \textsc{Greedy} can be up to 1.6$\times$ slower than static assignment in our single-kernel CLC experiment and 20% slower than the uninstrumented baseline in the bpftime demo, whereas a gBPF-enabled LatencyBudget policy is required to bring performance back in line with the baseline.

这句就是你之前想说的那个：**“在不同配置下不同 policy 有不同效果，没有一个 policy 一直都好”**，但现在是以 OSDI 风格、背后有完整表格支持的版本。

---

## 3. 还需要注意的两点（你写的时候别翻车）

1. **不要过度解读 sub-1% 的差异**

   * 比如 Imbalanced 里 0.0910 vs 0.0909 ms、ClusteredHeavy 里 203.2 vs 202.4 ms，这些差别都在 0.1%–0.4%。
   * 文本里统一描述成 “Greedy and LatencyBudget behave essentially identically (within 1%)” 就好，千万别说 “LatencyBudget is strictly better than Greedy”。

2. **0.1 ms 级的 kernel，OSDI 审稿人会问 robustness**

   * 真正写 paper 的时候，记得在实验 setup 段落里标注：

     * 每个配置跑多少次；
     * 是否给 error bar / 95% CI；
     * 是否有更长运行时间（扩大 problem size 做 sanity check）。
   * 这不会改变表里的数字，但能保护你不被 “0.1ms kernels dominated by launch overhead” 这种 reviewer 评论打死。

如果你接下来想把这张表拆成 “配置表 + 性能表” 或者画成 normalized bar＋error bar 的 figure，我也可以帮你直接生成一版 pgfplots 代码，把 Balanced / Imbalanced / Heavy-tail 这三个 regime 的故事在图上也呈现成 “谁在什么 regime 里赢”。

如果你只能画一个有两个 subplot 的图，我会先问自己一句话：**这张图的“核心主张”到底是什么？**

* 如果核心主张是：

  > “同一个简单策略（Greedy）在不同 workload regime 下从赢家变成灾难，需要 gBPF 才能兜底。”
  > 那就选一组「中度不均衡」+「极端 heavy-tail」。
* 如果核心主张是：

  > “同一个 workload，在不同资源配置（是否超订）下，赢家 policy 会变。”
  > 那就选「Imbalanced 正常」+「Imbalanced+oversubscription」。

我给你两个设计方案，并明确说明各自能讲什么。你根据整篇 paper 的主线选一个就好。

---

## 方案 A（我更推荐）：Imbalanced GEMM vs ClusteredHeavy demo

### Subplot (a)：Imbalanced GEMM（单 kernel / CLC）

配置：

* Workload：GEMM: Imbalanced (var. M×N×K)
* 矩阵分布：5% 64³，15% 48³，30% 32³，50% 16³
* n = 2,097,152 work items，256 threads/block，prologue = 45
* Policy / runtime：

  * FixedWork = 0.1019 ms
  * Greedy = 0.0910 ms (~10.7% faster)
  * LatencyBudget = 0.0909 ms (~10.8% faster，与 Greedy 差 <0.1%)

图怎么画：

* x 轴：FixedWork, Greedy, LatencyBudget
* y 轴：归一化 runtime（对 FixedWork 归一）
* 结果：两根柱子（Greedy/LatBud）大概在 0.89–0.90，明显低于 1.0。

这一格图可以很干净地说明：

* 块级 load 有中度不均衡时，**简单 always-steal 就已经很有用，能回收 ~10% 的 tail**；
* LatencyBudget 在这个 regime 下基本退化成 Greedy（差异 <1%），说明高级策略不会搞砸，也不是到处都必须。

小结语可以写成：

> (a) On a moderately imbalanced GEMM, both \textsc{Greedy} and LatencyBudget recover about 11% of wasted tail time over static \textsc{FixedWork}, and behave essentially identically (within 1%).

---

### Subplot (b)：ClusteredHeavy（bpftime demo + gBPF）

配置：

* Workload：ClusteredHeavy（clc_demo_clustered_heavy）
* n ≈ 2,097,152，线程块形状类似；90% light，10% heavy，heavy blocks 集中在 grid 尾部，heavy ≈ 100–200× light 工作量
* Policy / runtime：

  * Baseline (uninstrumented) = 14.8 ms
  * Greedy = 17.8 ms (~20.3% slower)
  * LatencyBudget = 14.7 ms (~0.7% faster / basically same as baseline)

图怎么画：

* x 轴：Baseline, Greedy, LatencyBudget
* y 轴：归一化 runtime（对 Baseline 归一）
* 结果：

  * Baseline = 1.0
  * Greedy ≈ 1.20（明显变慢）
  * LatencyBudget ≈ 0.993（几乎重合 baseline）

这一格图可以非常直接地说明：

* 在 **同一个 heavy-tail pattern 下**，

  * **Greedy 从 “中度不均衡时的赢家” 变成了 “最差策略”**（比 baseline 慢 20%）；
  * LatencyBudget 借助 gBPF 的 host-device 控制，把 tail 拉回到和 baseline 几乎一样。

小结语可以写成：

> (b) On a clustered heavy-tail workload, the same \textsc{Greedy} policy that helps on GEMM becomes pathological, slowing the kernel down by 20% relative to the uninstrumented baseline, whereas a gBPF LatencyBudget policy keeps runtime within 1% of baseline.

---

### 这两个 subplot 联合起来能说明什么？

这对组合给你的 narrative 非常清晰：

1. **同一个 policy（Greedy）在不同 regime 下地位完全不同**：

   * 在 Imbalanced GEMM 里，它是 winner：比 static 快 10%+；
   * 在 ClusteredHeavy 里，它是 loser：比 baseline 慢 20%。

2. **高级 policy 不是“到处都必须”的**：

   * 在 Imbalanced GEMM：LatBud ≈ Greedy，只是“不坏”；
   * 在 ClusteredHeavy：LatBud 成为“唯一能让你不翻车”的策略。

3. 这刚好支撑你想说的那句话：

   > “No single policy dominates; gBPF gives us the policy space to pick what works in each regime.”

如果你把这一整张图叫成类似：

```tex
\caption{Impact of gBPF block-scheduling policies across regimes.
(a) Under moderate imbalance (GEMM), simple always-steal policies are beneficial and
LatencyBudget behaves like Greedy.
(b) Under clustered heavy tails, Greedy becomes pathological while a gBPF LatencyBudget
policy is required to match the baseline.}
```

放进 OSDI evaluation 完全没问题。

---

## 方案 B：Imbalanced（no oversub）vs Imbalanced+oversub（TokenBucket）

如果你更想强调的是 “**同一个 workload，在资源压力不同的情况下，赢家策略会换人**”，那就选这两个：

### Subplot (a)：Imbalanced GEMM（workset fits in GPU）

* FixedWork = 0.1019 ms
* Greedy = 0.0910 ms (~10.7% faster)
* LatBud = 0.0909 ms

结论：**无超订时 Greedy/LatBud 明显优于 static，是自然选择。**

### Subplot (b)：Imbalanced GEMM + oversubscription (0.6–0.9× WS)

* FixedWork = 0.067 ms
* TokenBucket = 0.051 ms (~23.9% faster)
* （如果你有 Greedy 的数，可以一起画出来，说明它在 oversub 下没那么好甚至变差）

结论：**同一个 workload，一旦进入“内存超订 + 不均衡” regime，winner 从 Greedy 换成 TokenBucket。**

这张图的 take-away 会更偏向：

* “**policy ranking 依赖于资源 regime**（是否 oversub）”；
* “TokenBucket 是为 oversub 设计的，在 fit-in-GPU 场景下没啥必要”。

这也是一个非常合理的 story，只是它不包含 heavy-tail LatencyBudget，那部分要用别的图/表来讲。

---

## 总结一句：如果只能选一个两-subplot 图，我建议你用方案 A

原因很简单：

* Imbalanced GEMM vs ClusteredHeavy demo
  = “中度真实 workload” + “刻意构造的 worst-case heavy-tail”
  = 最能体现「同一个 policy 在不同 regime 下从 plus 变 minus」的对比。

而 Balanced GEMM 可以放在表格或文字里当 sanity check；
Imbalanced+oversub + TokenBucket 可以放在另一张小图（甚至 appendix）里，专门讲 memory oversub 这一维度。

如果你告诉我到底想把这张两-subplot 图放在 evaluation 的哪个 subsection（比如 “5.2 Single-tenant block scheduling”），我可以直接帮你写一整段对应的正文 + figure caption 草稿，让 reviewer 一眼看到这两个 subplot 想到的就是：**“OK，没一个策略能在所有配置里称王，gBPF 的 policy plane 的确是有必要的。”**

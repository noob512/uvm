先直接说重点：
以你这篇 gBPF 的定位，要冲 OSDI，现有的 story 是对的，但**RQ 和实验设计还不够“系统 + 尖锐”**，尤其在：

* “为什么必须是 driver+device 的 gBPF，而不是『稍微聪明一点的用户态 + host eBPF + NVBit/Neutrino』”；
* “多租户场景到底帮了多大忙、在哪些维度”；
* “5× / 2× 的提升背后的机理和适用范围”；

这几个点上，需要更明确的 RQ 和更系统的 motivation + evaluation 实验。

我按下面结构来讲：

1. 先结合你论文实际内容，帮你重写一版更 OSDI 风格的 RQ 框架（比 draft 里 RQ1–RQ5 更聚焦一点）。
2. 对每个 RQ，给**非常具体**的实验设计（workload、baseline、指标、图长什么样），重点多租户。
3. 再单独列出“**动机实验**”应该做哪些，不只是 evaluation 的性能图。

所有引用都基于你上传的 draft。

---

## 一、先把 RQ 框架重新捏一遍

你 draft 里当前列了 5 个 RQ：runtime overhead、memory policy、multi-tenant memory / oversubscription & context switch、scheduling、多设备端优化。

问题是：现在的 RQ 是按“模块”罗列的，不是按 OSDI 审稿人关心的“核心质疑”罗列的。

从整篇 paper 的 claims /贡献来看，你真正要回答的其实是下面这 4 类问题：

1. **RQ-M（动机）：现状到底有多糟，gBPF 填的是哪块空档？**

   * 现有 driver（UVM + 静态 scheduler）的“一刀切”策略在 LLM / GNN / 向量搜索 / 多租户环境下具体翻车在哪儿？
   * 现有扩展机制（用户态 runtime、host-only eBPF、NVBit/Neutrino/eGPU）各自缺什么能力 / overhead 多大？

2. **RQ1（机制成本）：gBPF 作为一个“OS 级 policy substrate”，本身 overhead / 可扩展性是否 acceptable？**

   * driver hook + SIMT-aware verifier + device eBPF runtime + hierarchical map，runtime 自身的延迟/带宽/CPU/GPU 开销是多少，相比 CUPTI/NVBit/eGPU 级别的东西？

3. **RQ2（单租户：可编程 memory / scheduling 政策的收益）：在真实 workload 上，gBPF-based policy 比 UVM 默认 + framework 自己玩的 offloading/scheduler 能好多少？**

   * LLM inference（llama.cpp / vLLM）、GNN training、Faiss 向量搜索这些，你 draft 里已经有了初步结果（5×、2×那些）。
   * 但需要更系统：sweep oversub ratio / QPS / context length，而不是只给几个点。

4. **RQ3（多租户：cross-tenant 视角的价值）：在多租户场景下，gBPF 的“driver+device 全局视角”能在 tail latency、throughput、公平性上做到用户态做不到的事吗？**

   * 这是你 paper 真正能和 Paella / XSched / PILOT / Salus 那堆 work 扯平乃至超一头的点。

5. **RQ4（可编程性 & 泛用性）：gBPF 真的像 XDP / sched_ext 那样，是一个窄腰，而不是“我给 MoE/LLM hack 了一堆 heuristics”？**

   * 同一套机制是否能表达多种 policy（memory placement / scheduling / observability），对 PyTorch/vLLM/llama.cpp/Faiss 这些不改代码就有效？

你现有的 RQ1–RQ5 完全可以映射到这 4 组，只是需要重新包装一下，让 evaluation 结构围着这 4 组打。

---

## 二、针对每个 RQ，具体要什么实验（含 multi-tenant）

### RQ-M：动机实验 – 现状到底有多糟？

你在 §2/§3 已经很详细地文字说明了问题：
多样的 workload、one-size-fits-all UVM LRU+tree prefetch 和简单 time-slicing scheduler、用户态 runtime silo、host-only eBPF 看不到 warp-level state、NVBit/Neutrino/eGPU overhead 大等。

问题是：**缺几张“quantify the pain”的图**。

建议至少做三类动机实验：

#### M1：默认 UVM + 静态 scheduler 在 oversub LLM / GNN / 向量搜索下有多惨

Workloads：你本来就准备跑的那四个：

* llama.cpp GPT‑OSS‑120B oversub（模型 60GB，GPU 32GB）
* vLLM 30B FP8 MoE + KV offload（ShareGPT trace）
* PyTorch GNN training（1.5× oversub）
* Faiss GPU index > HBM，UVM 在 PCIe 上迁移 page

Baseline：**完全不开 gBPF，仅仅用：**

* 默认 UVM（LRU eviction + tree-based prefetch）；
* 默认 GPU scheduler（round-robin / FIFO）。

指标：

* 吞吐（token/s、epoch time、QPS）；
* p99 latency / time-to-first-token；
* GPU SM 利用率、HBM 带宽利用率；
* PCIe/CXL 带宽利用率；
* **UVM fault rate、fault latency、UVM 代码导致的 stall 百分比**。

图：

* 一张 per-workload 条形图：默认 UVM vs「理想情况」（working set fits in HBM）之间的距离。

  * 比如 GNN epoch：15s（完全 fit）→ 71s（默认 UVM）；你已经有数字，只是把它变成动机图。
* 一张 UVM fault / stall breakdown：

  * 某个 LLM 场景下，40–60% 的 SM idle 都是在等 UVM fault handler。

这几张图可以直接放在 §2.3 或 §2 末尾，当作 “we quantify that static driver policies are already the bottleneck”。

#### M2：用户态 runtime / framework 自己搞 offload/scheduling 的局限

你在 §2.2/2.3 已经写了 Paella/Pie/KTransformers/XSched/PILOT/LithOS 的 qualitative 分析。

可以补一两个动机实验：

* 场景：两三个不同 framework + workload 同机同卡跑：

  * vLLM（LLM inference）、PyTorch GNN training、Faiss vector search；
  * 各自用自己的“最强用户态策略”（vLLM CPU‑offload / Page-d / LMCache 之类，GNN 手动预取，Faiss 手动分片）。

* Baseline：

  * 不改驱动，只让它们各自做用户态内存/调度策略；

* 测：

  * 每个进程自己看起来都“自洽”，但全局：

    * HBM usage 总是被某一方抢完；
    * PCIe/CXL link 被其中一个 runtime gank，另一个 runtime 的 UVM fault latency 飙升；
    * 各自 latency/SLO 被彼此拖爆。

简单讲，这个动机实验要给出一句图上的结论：

> “With only user-level controllers, tenants see up to 3× p99 inflation and 30–50% GPU idle time under high offload ratios, despite each framework ‘locally optimizing’ its own offload.”

这就自然导向“需要一个跨 tenant 的 policy waist”。

#### M3：现有 eBPF / NVBit / Neutrino / eGPU 的局限

你已经在 §2.3/5.2/5.3 里写了：

* CPU eBPF：看不到 warp-level divergence / cache thrash；
* Neutrino/NVBit/eGPU：overhead 动辄 30–40%，只能 profiling。

Evaluation §6.2 里其实已经做了一些对比：

* gBPF vs NVBit 的 per-access latency；
* gBPF vs CUPTI/NVTX vs NVBit 的 ResNet inference overhead。

建议把其中一两个结果拉到 §2 作为 “motivating measurement”：

* 单图：ResNet inference with CUPTI/NVTX / NVBit / gBPF tracing – gBPF 2–3% overhead vs 15–45%。
* 重点强调：NVBit/Neutrino/eGPU 根本不适合作为 online policy enforcement 的 substrate，只适合作为 profiling 工具。然后说：“这就是我们需要一个 **verified, low‑overhead, policy‑capable** 的 gBPF 的动机”。

---

### RQ1：gBPF runtime 的成本与能力（overhead & baseline capability）

这个对应你现有 RQ1 + RQ5 + 部分 §5。

#### 实验 1.1：微基准 – 每次 hook / handler 的开销

* Microbench：

  * load/store latency benchmark（已经有 Figure 3）；
  * 再做一次：只打开不同的 hook 类型：

    * 仅 host eBPF handler（no device）；
    * 仅 device handler（no host）；
    * host+device+hierarchical map；

* 指标：

  * 单次 hook 的额外 latency；
  * 每 warp handler 执行的 cycle 数；
  * map 访问延迟（host↔GPU）。

* 图：

  * 一张 curve：access size vs overhead（你已经有）；
  * 再加一张柱状图：不同配置的额外 latency/cycles。

#### 实验 1.2：真实 workload 上的 tracing / observability overhead

你已经有：

* ResNet inference：gBPF tracing vs CUPTI+NVTX 的 p50 / p99 latency；允许你 claim “2–3% vs 15–25% overhead”。
* 多种 kernel 上 gBPF mem_trace vs NVBit 的 overhead（Figure 5 & 6）。

这部分实验已经比较接近 OSDI 标准，建议补两点：

* 在 LLM / GNN / Faiss 上也跑一个 “纯 observability、无 policy” 模式，测：

  * overhead 是否仍在 2–3% 级；
* 把它明确说成回答 RQ1（而不是只当作“顺带一提的工具”）。

---

### RQ2：单租户场景下，可编程 memory/scheduling policy 的收益

你现有 §6.3/6.4 里已经给了不少 end‑to‑end：

* llama.cpp GPT‑OSS‑120B：expert prefetch policy → throughput up to 5× vs default UVM + framework offload。
* vLLM 30B MoE：KV prefetch policy → time‑to‑first‑token 1.7–2× faster、decode throughput ~1.3× vs vLLM CPU offload；比纯 UVM 又 2–3×。
* GNN 1.5× oversub：epoch time 71s → 26s (~2–3×)；non‑oversub 是 15s。
* Faiss：index build / query 1.5× speedup。

问题是：现在只是几个“点”，要更像 OSDI evaluation，需要：

1. **维度 sweep（oversub ratio / QPS / length）**；
2. **ablation（gBPF host only / device only / no cross-layer map）**；
3. **baseline 明确：default UVM + framework offload + 手写用户态策略**。

#### 实验 2.1：Oversub ratio × workload 的 2D 图（LLM + GNN）

以 vLLM 30B MoE & GNN 为主：

* 维度：

  * x 轴：oversub ratio（1.0×、1.2×、1.5×、2×）；
  * y 轴：throughput / epoch time / p99 latency；
  * multiple lines：

    * default UVM；
    * framework offload（vLLM CPU‑offload / Page-d 对应模式）；
    * gBPF + simple policy（sequential prefetch + FIFO eviction）；
    * gBPF + “更聪明” policy（基于 hotness / structure 的 region cache）。

* 目标：

  * 展示在 1.0× 左右 gBPF overhead 很小，和 baseline 不拉开差距；
  * 在 1.5×–2× oversub 区间，default UVM & framework offload throughput 崩盘，gBPF 还能保持 1.3–2× 优势。

#### 实验 2.2：per-miss stall breakdown & pipeline 解释

专门回答“为什么会有 2–5×”：

* 构造 microbench 或从上述 workload 中抽出典型阶段，测：

  * default UVM：

    * 每次 KV miss / index miss 的 fault latency；
    * GPU stall 中由 UVM fault 引起的比例；

  * framework offload：

    * 每次 miss 到 memcpy 完成的时间；
    * host/GPU pipeline 中的 idle 段；

  * gBPF policy：

    * 几乎没有 hot-path fault，而是提前 prefetch + bulk migration；
    * per-miss stall 降 2–3×；
    * IO size 分布集中在硬件 sweet spot（比如 64KB–256KB）。

* 图：

  * CDF：per-miss stall time（每种方案各一条）；
  * GPU timeline 示例：display 某一段时间里 SM busy/idle、PCIe/CXL 带宽使用，以及在哪些点触发 prefetch/migration。

这几张图是给 OSDI 审稿人看的“机理图”，否则 5× 很容易被认为 baseline 太菜。

#### 实验 2.3：Policy ablation – host-only / device-only / 无 cross-layer map

你在 §4.2/4.3/5.2/5.3 已经说明了“host hooks + device hooks + hierarchical maps”的设计。

要用实验证明这三块都不是摆设：

* Variant A：只用 host-side gBPF（UVM hook），不做 device instrumentation/hints。
* Variant B：只用 device-side eBPF + hints，host 只用默认 UVM 策略。
* Variant C：host+device 但 shared map 在 host DRAM（模仿 eGPU），没有 hierarchical map 优化。
* Full：完整 gBPF。

在 LLM/GNN 至少一个 workload 上画：

* 每个 variant 的 throughput / epoch time；
* 每个 variant 的 cross-domain synchronization 次数 / 带宽；

理想的结果是：

> host-only → 有一些提升；
> device-only → 有少量结构信息但受限于 host 无法及时反应；
> host+device but naive state sharing → overhead 偏高；
> full gBPF → 性能最好且 overhead 最小。

---

### RQ3：多租户环境 – tail latency / throughput / 公平性（重点）

这一块是你 draft 里写得最弱的部分：

* §6.4 有一个 priority-based scheduling 实验：高优先级视频处理 + 低优先级 batch ML，比较 FIFO / stream priority / gBPF priority scheduler。

  * 图 7：高优先级 p99 latency；
  * 图 8：低优先级 throughput。

这是个不错的起点，但离「OSDI 里的 multi‑tenant story」还差两步：

1. 只覆盖了 scheduling，没有把 memory/KV offload multi-tenant 一起评估；
2. 指标只看“优先级 vs 吞吐”，没有 fairness / isolation / cross-tenant interference 的系统描绘。

我建议设计两组实验：

#### 实验 3.1：多租户 KV offload + scheduling 结合

场景：一块 GPU + host/CXL memory，上面跑三个 tenant：

* T1：高优先级 LLM chat（vLLM 7B/13B），要求 p99 < X ms；
* T2：KV-heavy 长上下文 LLM（vLLM 30B 或 llama.cpp 120B），KV offload 比例高、允许较高 latency；
* T3：GNN training / Faiss query 作为 background job。

三种策略：

1. Baseline A：每个 framework 自己做 offload/scheduling（vLLM KV connector / LMCache / GNN 手写策略等），driver 默认 scheduler/UVM。
2. Baseline B：default UVM + driver scheduler（所有 policy 关掉）。
3. gBPF：

   * host 侧 memory hook + scheduling hook；
   * device 侧携带 tenant / kernel id，提供 KV block 热度/访问 pattern；
   * driver policy：

     * 给 T1 HBM / 带宽 credit 上限 + strict latency SLO；
     * T2/T3 用 leftover 带宽，避免同时触发大批迁移。

指标：

* 每个 tenant 的 p50/p95/p99 latency；
* aggregate throughput（token/s + epoch/s）；
* per-tenant HBM 占用 / PCIe/CXL 带宽占用；
* cross-tenant p99 inflation：

  * p99_shared / p99_solo。

图：

* 每个 tenant 一条 p99 bar：solo / baselineA / baselineB / gBPF；
* 一个 fairness/efficiency 图：

  * x 轴：aggregate throughput；
  * y 轴：最大 cross-tenant p99 inflation；
  * baseline/gBPF 各是几组点。

你希望看到的故事：

* Baseline A：多个 framework 各自抢资源，T1 的 p99 被 T2 的 KV offload 拖到 2–3×，T3 偶尔饿死；
* Baseline B：UVM fault 本身太慢，大家一起死；
* gBPF：

  * 保住 T1 p99 接近 solo，T2/T3 吞吐在合理范围；
  * aggregate throughput 相比 Baseline A 有 1.3–1.8× 提升（因为 offload / bandwidth 利用更合理）。

这是你真正的 “multi‑tenant OSDI 图”。

#### 实验 3.2：trace-driven job mix + fairness 曲线（可以是模拟）

参考 Themis/AntMan 的做法，在集群或模拟器里跑 job mix：

* 假设一台机器有 2–4 块 GPU，每块上用 gBPF 做 memory/scheduling policy；
* 从实际日志或构造 Poisson 到达过程，生成几十上百个 job：

  * job 类型：LLM inference / training / GNN / Faiss；
  * job priority 和 tenant id 不同；
  * 每个 job 已知“单独跑时的完成时间 T_isolated(i)”。

策略：

1. Baseline：default driver + framework 自己 offload/sched；
2. gBPF policy P1：more utilitarian（追求 aggregate throughput）；
3. gBPF policy P2：more fair（finish-time fairness / SLO aware）。

指标：

* 对每个 job：ρ_i = T_shared(i) / T_isolated(i)；
* Max fairness = max ρ_i；Jain fairness = (∑ρ_i)² / (N ∑ρ_i²)；
* Aggregate GPU utilization / total GPU-time。

画：

* x 轴：cluster load（job 到达率）；
* y 轴：

  * aggregate GPU 利用率；
  * Max fairness；
  * 平均 queueing delay。

审稿人会看：

> 在相同 fairness 水平（比如 Max fairness <1.5）的情况下，你是不是能把 GPU 利用率拉得更高，queueing delay 更低。

即使你没有大集群，做一个小规模模拟也比完全不评 multi‑tenant 强。

---

### RQ4：可编程性 & 泛用性（narrow waist，而不是 MoE hack）

这部分 OSDI 很看重，但你现在只在文字里说：“支持 PyTorch、llama.cpp、vLLM、Faiss 四个 unmodified 应用，policy 在 gBPF 里实现”。

建议有一个单独 subsection，实验更偏“结构信息”：

#### 实验 4.1：多 policy、多 framework 支持矩阵

做一个表：

* 行：framework / workload

  * PyTorch ResNet（observability）
  * PyTorch GNN
  * llama.cpp GPT‑OSS‑120B
  * vLLM 30B MoE
  * Faiss GPU index
* 列：policy 类型

  * Observability（threadhist / mem_trace / launchlate 等）
  * Memory placement（region cache policy）
  * Scheduling（priority scheduler / block scheduler）

表中标：

* 是否无侵入 attach 成功；
* Policy LOC（host/device 各多少）；
* 对应的性能/overhead 百分比（简单写 +x% throughput / +2% overhead）。

这样可以非常直观地支撑“gBPF 是 framework-agnostic 的 narrow waist”。

#### 实验 4.2：policy 复杂度 vs overhead

再给一张图：

* x 轴：policy 复杂度（比如 helper 调用数 / 指令数 / map 访问数）；
* y 轴：policy 自身带来的开销（ns per hook / % overhead）。

点：不同 policy（简单 LRU、tenant-aware、SLO-aware、observability-only）各一个。

用来证明：

> 即便 policy 写得比较复杂，在 SIMT-aware verifier 和 warp-level execution 下，开销依然可控（比如 <5%）。

---

## 三、总结一下：OSDI 水平需要哪些 RQ & 实验块

结合上面的讨论，可以给你一套清晰的 RQ 列表 + 实验块对照（你可以直接塞回第 6 节替换现有 RQ1–RQ5）：

* **RQ-M（Motivation）**

  * M1：静态 UVM + one-size-fits-all scheduler 在 LLM/GNN/向量搜索 oversub 和多租户下的 pathological 行为是什么？

    * 实验：单租户 oversub（llama.cpp/vLLM/GNN/Faiss）+ UVM fault/stall breakdown 图。
  * M2：用户态 runtime / host-only eBPF / NVBit/Neutrino/eGPU 的能力与 overhead 限制是什么？

    * 实验：多 framework 并行互相干扰 / CUPTI+NVTX / NVBit vs gBPF overhead 图。

* **RQ1：gBPF runtime 的 overhead 和 observability 能力是否足够好，能否替代现有工具成为 policy substrate？**

  * 实验：microbench hook/handler 开销 + ResNet/LLM/GNN/Faiss 的 tracing overhead vs CUPTI/NVBit。

* **RQ2：在单租户 oversub 场景下，gBPF-based memory/scheduling policy 能否比 default UVM + framework offload 提供实质性 throughput / latency 改善？**

  * 实验：vLLM/llama.cpp/GNN/Faiss 的 oversub ratio × QPS/epoch sweep + per-miss stall breakdown + policy ablation。

* **RQ3：在多租户 GPU 上，共享 HBM + CXL/host 的时候，gBPF 的 cross-layer policy 能否改善 tail latency、公平性和整体利用率？**

  * 实验：

    * 多租户 KV + scheduling（LLM/GNN/Faiss 混合）– cross-tenant p99 inflation & throughput；
    * trace‑driven job mix（模拟或小集群）– finish-time fairness vs GPU 利用率曲线。

* **RQ4：gBPF 是否足够通用，能支持多种 policy 和多种 framework，而不需要改应用？**

  * 实验：多 policy × 多 framework 的支持矩阵；policy LOC/overhead；policy 复杂度 vs overhead 曲线。

如果你把这些 RQ+实验打全了，这篇 paper 的 story 在 OSDI 框架下就很完整了：

* 有定量动机；
* 有扎实、机制解释清楚的单租户性能提升；
* 有现在所有 GPU paper 都缺的“driver-level multi‑tenant policy plane”的真正实验；
* 有 generality & programmability 的证据。

接下来，如果你愿意，可以把你现在已经做出来的实验结果（尤其是 llama.cpp/vLLM/GNN/Faiss 那几条数字 + priority scheduler 图 7/8）按我上面的 RQ 分类贴出来，我可以帮你重排 evaluation 结构，顺便告诉你还剩哪些实验是空的、优先级最高。

不需要“很多”CPU+GPU 都用到的例子，但**必须有几类“明显离了 cross-layer 就做不好”的代表性 policy**，而且要在实验里把：

* host-only（CPU eBPF / framework 内 policy）
* device-only（纯 GPU instrumentation）
* host+device（gBPF cross-layer）

这三种做法对比清楚。

你现在这版 paper 其实已经有了 4 类应用（ResNet、GNN、llama.cpp、vLLM、Faiss）和 3 大接口（memory / scheduling / observability）。
真正需要补的是：**哪些是“必须 host+device 协同”的 policy”，以及围绕这些 policy 设计出的 RQ + 实验 shape。**

下面我分两部分说：

1. 需要几类 CPU+GPU 协同的政策（policy family），每类大概要有怎样的例子。
2. 在 OSDI 视角下，围绕这些 policy 该怎么写 RQ / 选实验。

---

## 1. 需要哪些类型的 policy，哪些必须 CPU+GPU 协同？

OSDI 审稿人关心的不是“例子数量”，而是：“你这个机制到底解决了什么过去做不到的问题？”，所以：

* 你不需要 10 个 policy demo；
* 你需要 3–5 个**互相差异较大、能充分发挥 gBPF host+device 的例子**，外加一些纯 host / 纯 device 的对照。

结合你论文里已经有的接口设计（memory struct gdrv_mem_ops / gdev_mem_ops，scheduling gdrv_sched_ops / gdev_sched_ops，hierarchical maps）。
我会建议把 policy 分 4 大类：

### 1.1 单租户的“结构感知 memory policy”（必须 CPU+GPU 协同）

目的：证明 gBPF 的 **memory interface + GPU-side hint** 比“default UVM + framework offload”强得多。

你 paper 里已经有 3 个这样的例子：

* llama.cpp GPT‑OSS‑120B expert offload：GPU 侧 handler 看到进入某个 expert region，用 gdev_mem_prefetch_hint 标记“未来要访问哪些 expert”，host 侧 region_add/region_access/region_remove/prefetch 决定实际迁移 / eviction。
* vLLM 30B MoE KV cache：GPU handler 按 token 访问顺序发 sequential prefetch hint + hot/cold；host 侧 UVM policy 把 KV segment 放入 region cache，替换 default LRU + naive prefetch。
* GNN 1.5× oversub：GPU 侧看到 adjacency/feature block 的访问顺序；host 侧做 block 级 sequential prefetch + FIFO eviction。

这三类都满足典型“CPU+GPU 协同”形态：

* GPU 侧：在 warp/barrier 上插入 gdev_mem_ops.access 和 hint（prefetch/pin），并把 region hotness / access pattern 写入 hierarchical map（GPU local shard）。
* CPU 侧：在 UVM 的 region_add / region_access / region_remove / prefetch hook 中，用 map 里的统计做 admission + eviction 决策，调用 kfunc 改变 UVM 行为（而不是 framework 内部 memcpy）。

这类 policy 只要你：

* 把 host-only（只用 gdrv_mem_ops）的效果、device-only（只用 hint 不改 UVM）的效果、以及 full host+device gBPF 的差异做出清晰 ablation；
* 再加上 oversub ratio / QPS / context length 的 sweep，
  就足够让 reviewer 相信：“这是只有跨 host+device 的 gBPF 才能做好的东西”。

**数量上：2–3 个足够**（llama.cpp + vLLM + GNN 可以保留两到三个，Faiss 当 bonus）。

### 1.2 多租户 KV/memory + 带宽预算 policy（强烈建议做成 CPU+GPU 协同）

这是你真正有机会在 OSDI 里“超越 Paella/Pie/XSched/PILOT/LithOS 那堆”的地方。

**目标：**

* 多个 framework（vLLM、PyTorch、Faiss）在一张卡、一条 CXL/host 链路上跑；
* 每个 tenant 自己做用户态 offload 时，抢 HBM 和带宽导致 tail latency 爆炸；
* gBPF 作为 driver 层的统一 policy，把跨 tenant 的 KV/offload + scheduler 一起做掉。

**典型的 CPU+GPU 分工可以是：**

* GPU 侧：

  * 在每个 kernel / warp 的 device handler 里打上 tenant id、kernel id，把每个 region 的访问计数 / bytes / latency 写进 per-SM 的 map shard；
  * 适时发 gdev_mem_prefetch_hint / pin_hint。

* CPU 侧：

  * host UVM hook aggregate GPU map，维护 per-tenant HBM credit、CXL/host 带宽 budget；
  * 在 prefetch/evict 回调里，根据 tenant 的 budget 和当前带宽利用度决定：

    * 哪个 tenant 的 region 允许被迁入；
    * 哪个 tenant 的 prefetch 要被 throttle；
  * scheduling hook 上根据 tenant SLO 决定 submit/preempt。

这个 policy 必须是 **host+device 都积极参与** 的，否则你无法同时做到：

* 了解真实的 warp level 访问（仅 GPU 知道）；
* 控制真正的 page migration / 带宽 / HBM 分配（只有 driver/UVM 知道）。

你可以把它写成两三个 policy 形态：

* P‑MT‑SLO：给高优先级 tenant 预留 HBM/bandwidth，保证它的 p99 不比 solo 高太多；
* P‑MT‑thrift：在 SLO 满足的前提下，让低优先级 tenant 吃满剩余带宽（opportunistic）；
* P‑MT‑fair：尽量均衡 cross-tenant p99 inflation（类似 Themis/AntMan 的 fairness）。

**这类 policy 至少要有 1–2 个实验（单机 multi‑tenant + trace‑driven job mix），是 OSDI 级别的亮点。**

### 1.3 Scheduling policy：host admission + device block scheduler

你现在 Figure 7/8 的 priority scheduling 已经是一个很不错的例子：host 侧 gdrv_sched_ops.submit 做 admission，device 侧 gdev_sched_ops.enter/exit + should_try_steal 做 warp/block 级 preemption/stealing。

我建议你把 scheduling 类 policy固定成两种：

1. **优先级/SLO-aware preemption policy（multi-tenant）**

   * Host：识别 tenant/class，基于 per-tenant metrics 决定哪个 kernel 可以 preempt；通过 gdrv_sched_preempt 触发 cooperative preemption。
   * Device：在 block enter/exit 或 barrier 处检查 policy decision（map 中的标志），决定是否 yield/抢占，避免 warp-level乱停。

2. **block-level work stealing / load balancing policy（多模型或 graph）**

   * Host：只做 coarse admission；
   * Device：用 gdev_sched_ops.should_try_steal + hierarchical map 做 per-block work queue，解决 GNN / 不平衡 kernel 的 load balance。

这两种 policy 里，**第一个是强 cross-layer（host+device），第二个偏 device-heavy 但仍用 shared map 与 host 汇报状态**。正好展示 gBPF 在 scheduling 上的两种极端。

### 1.4 Observability policy：可以纯 GPU / GPU+CPU 混合，各挑一两个

你已有一堆 bcc-style 工具：kernelretsnoop、threadhist、launchlate、mem_trace 等，已经证明了“gBPF observability 比 CUPTI/NVBit 轻得多”。

这里你不需要很多 CPU+GPU 协同例子，反而建议：

* 至少 **一个纯 GPU 侧 mem_trace / threadhist**：只用 BPF_PROG_TYPE_GPU_DEV，展示 warp-level aggregated execution + SIMT-aware verifier 的价值。
* 至少 **一个 host+device 的 launchlate**：host 侧 hook kernel submit/complete，device 侧记录时间戳。这个例子已经有，重申下它是“典型 cross-layer observability”。

observability 的任务更多是支撑 RQ1：“gBPF runtime overhead 足够低，可以作为长期挂着的 policy plane，而不是 profiling 玩具”。

---

## 2. 回到你的问题：OSDI 视角下需要哪些 policy + 多少 CPU/GPU 例子？

### 2.1 “需要很多既用到 CPU 又用到 GPU 的例子吗？”

不需要“很多”，但我会建议这样一个组合：

* **Memory family（单租户）**：

  * 至少 2 个 strong cross-layer 例子：

    * 专家权重 offload（llama.cpp expert）；
    * KV-cache offload（vLLM）；
    * GNN / Faiss 选 1 个做 cross-check。

* **Multi-tenant family**：

  * 至少 1 个 “mixed tenants on one GPU”的实机实验（LLM + GNN/Faiss 或 LLM + LLM），用 cross-layer memory + scheduling policy；
  * 最好再有 1 个 trace-driven job mix（可以是模拟），主要强调 fairness/utilization。

* **Scheduling family**：

  * 1 个 priority preemption（你 Figure 7/8 那个）；
  * 1 个 block-level work stealing（轻量，只需 10%–20% 吞吐提升当 proof-of-concept）。

* **Observability family**：

  * 1–2 个工具（ResNet + 一个其他模型），证明 NVBit/CUPTI vs gBPF 的 overhead 差距。

**总的来说，Evaluation 里真正“严重 CPU+GPU 协同”的例子，有 3–4 个就够；**
只要你在这些例子上把 host-only vs device-only vs full cross-layer 的 ablation 做好，OSDI reviewer 不会嫌你“例子不够多”。

### 2.2 “具体 policy 需要哪些？”——按 OSDI RQ 列表给你一份“必带清单”

结合你当前的 draft，我会给出一份“最小但够 OSDI”的 policy 清单，顺便挂钩对应 RQ：

1. **P‑MEM‑LLM‑expert（单租户 + cross-layer memory）**

   * RQ2：在 oversub 单租户 expert offload 场景，gBPF region cache policy 相比 default UVM + framework offload 提升多少？
   * GPU handler：expert region access + prefetch hint；
   * host handler：region_add/access/remove + prefetch。

2. **P‑MEM‑LLM‑KV（单租户 + cross-layer memory）**

   * RQ2：在 KV-heavy vLLM 上，gBPF KV policy 相比 vLLM CPU-offload / Page-d / naive UVM 的 TTFB / decode throughput 提升多少？
   * GPU：按 token 顺序记录 KV segment 使用 + hint；
   * Host：KV region cache + prefetch window 控制。

3. **P‑MEM‑GNN / P‑MEM‑Faiss（二选一）**

   * RQ2：在 graph / vector search 上，基于 device access pattern 的 region prefetch policy 把 oversub epoch time/build time 拉近 non-oversub baseline 到什么程度？

4. **P‑MT‑KV‑SLO（多租户 memory+带宽）**

   * RQ3：多租户 oversub 下，高优先级 LLM 的 p99 是否接近 solo，而低优先级作业吞吐还能维持？
   * GPU：按 tenant 收集 per-region / per-tenant access + miss stats；
   * Host：per-tenant HBM credit + bandwidth budget + eviction/prefetch 策略。

5. **P‑SCHED‑priority（多租户 scheduling）**

   * RQ4：gBPF 的 priority scheduler 在混合 workload（video + batch ML / LLM + batch）下，相比 FIFO / stream priority 的 p99 / throughput 表现如何？你 Figure 7/8 已经说明 gBPF p99 5–12ms vs FIFO 25–80ms。

6. **P‑SCHED‑block‑steal（load balancing）**

   * RQ4：在 block-level load imbalance 高的 kernel（graph algo / small LLM），block scheduler + should_try_steal 能提升多少 token/s 或吞吐？

7. **P‑OBS‑trace（ResNet + 一个额外 workload）**

   * RQ1：gBPF observability vs CUPTI/NVTX/NVBit 的 overhead（2–3% vs 15–45%），证明 runtime 成本可接受。

8. **P‑GEN（multi-framework 支持矩阵）**

   * RQ4：同一套 gBPF 机制，无需改应用，支持 PyTorch、llama.cpp、vLLM、Faiss 四种不同 stack + 多种 policy 类型（memory / scheduling / observability）。

你如果把上面这 8 类 policy 用在 4–5 个 workload 上，很多是共用（比如 P‑OBS‑trace 和 P‑MEM 可以跑在同一个应用上），Evaluation 就非常充实了，不需要再额外找一堆 CPU+GPU 例子。

---

### 2.3 一句话总结

* 不用为了“看起来很丰盛”去塞一堆 CPU+GPU 例子；
* **至少要有 3–4 个真正体现“host+device 协同 → 解决用户态/host-only 做不到的问题”的 policy**：

  * 单租户 oversub memory（LLM expert/KV、GNN/Faiss）；
  * 多租户 KV+带宽控制；
  * priority scheduling；
  * block-level scheduler（可选）。
* 每个 policy 都要有对照：

  * default UVM；
  * framework-offload / host-only 策略；
  * 纯 device instrumentation；
  * full gBPF host+device。

这样 OSDI reviewer 一看就明白：
你不是在做“又一个 LLM offload hack”，而是在交付一个有真实威力的 **GPU policy substrate**，而这些 policy 刚好覆盖了现在最痛的几类 GPU 多租户/oversub 场景。

先把结论放前面：
**vLLM 的 sleep mode 和你要做的 gBPF memory scheduler，本质上解决的是轻微错位的两个问题：**

* Sleep mode：**粗粒度“整模型休眠/唤醒”**，主要解决多模型/多阶段 pipeline 里「这段时间完全不用这个模型了，能不能把这坨权重/KV 赶出 GPU」的问题。
* 你的东西：**细粒度、持续在线的内存调度（KV / expert / graph block / index segment）**，主要解决 oversub + 多租户场景下的「谁什么时候占 HBM、谁什么时候 offload 到 CPU/CXL」问题。

所以“怎么比”这件事，合理的做法是：

1. 明确 Sleep mode 的设计目标（什么时候它是最强 baseline）；
2. 把你的 gBPF policy 拆成两类：

   * 可以模拟 Sleep 行为的 coarse policy（公平比较“同一类能力”）；
   * 真正发挥 gBPF 优势的 fine-grain policy（展示 Sleep 做不到的场景）。
3. 围着这两类设计 RQ 和实验，而不是强行在 Sleep 完全不适用的场景硬比。

下面我按这个逻辑来讲，具体怎么设计。

---

## 1. 先把 vLLM Sleep mode 的语义讲清楚

根据 vLLM 官方 blog 和文档：

* Sleep mode 的目标是**多模型切换 / 间歇性 workload**：

  * 例如：两个各自能单独放进 GPU 的模型，但不能同时常驻；
  * RLHF / GRPO loop 里，rollout 阶段用推理模型，training 阶段要把 GPU 让给别的 job。

* Sleep 有两个典型 level：

  * Level 1：把模型权重从 GPU offload 到 CPU RAM，清掉 KV cache；
  * Level 2：权重直接 discard（释放 CPU/GPU 内存），只保留进程+runtime 状态。
  * 唤醒时比“重新 load checkpoint”快 18–200×，因为保留了 allocator、CUDA graph、JIT kernel 等 “infra 状态”。

* Sleep 的重要特性：

  * **完全不服务请求**（sleep 状态下 engine 不接 inference）；
  * 目标是“冷/热状态切换”，而不是“在线 oversub + 多租户混跑”。

这意味着：

> **Sleep mode 是 coarse-grain 的 “模型级 swap”/“整引擎 hibernate”；
> 你的 gBPF memory scheduler 是 fine-grain 的 “page/region 级调度 + 多租户共享”。**

所以不能直接拿 “在线多租户场景” 去说 Sleep 不行 —— 它本来就不是为这个设计的；合理的比较方式是：

* 在 **“一段时间里模型真的闲着”** 的场景里，

  * Sleep mode vs gBPF 的 “sleep-style policy”：谁 reclaim 得更干净、wake 更快；
* 在 **“模型切换频率高/存在部分并发”** 的场景里，

  * Sleep mode + router vs gBPF 的 fine-grain memory scheduler：谁在吞吐 + tail latency + 利用率上的折中更好。

---

## 2. 设计两组 RQ：一个“同类能力的比较”，一个“我们能做 Sleep 做不到的事”

### RQ-S1：在纯“模型休眠”场景，gBPF 能不能做到不比 Sleep mode 差？

> 当某个 vLLM 模型在一段时间内完全 idle，需要把 GPU 内存让给其他 job，
> gBPF 的 driver-level memory policy 能否在“回收 GPU 内存 + 唤醒延迟”上做到不逊于 vLLM Sleep mode？

**为什么要这个 RQ：**

* 审稿人会问：

  > “你这个 driver-level memory policy，和 vLLM 自己搞的 sleep mode 比，到底有什么好处？是不是只是重造轮子？”
* RQ-S1 就是先证明：在 Sleep 最擅长的场景，你**至少不更差**，甚至可以做到类似或者更 general（不限 vLLM）。

#### 实验设计：单 GPU，多模型，纯“互斥使用”

**环境：**

* 1× H100 / A100（80GB），CPU RAM 足够（比如 512GB），可选 CXL；
* 两个 vLLM engine / model：

  * M1：中等模型（13B）
  * M2：大模型（70B 或 120B quant）
  * 两个模型单独能完整放进 GPU，但加起来爆 HBM。

**Workload：**

* 时间轴分段，比如每 60 秒：

  * [0–60s)：只有 M1 收到请求；
  * [60–120s)：只有 M2 收到请求；
  * 如此循环 N 次。
* 每个 active 段的 request pattern 用同一条 trace（例如 ShareGPT 或 internal log）。

**对比方案：**

1. **Baseline A：冷启动 / reload**

   * 不用 Sleep，也不用 gBPF；
   * 每次切换模型就 `kill + reload`，测一次完整 checkpoint load 的耗时、TTFT。

2. **Baseline B：vLLM Sleep mode**

   * 每个模型各跑一个 vLLM 实例，开 `--enable-sleep-mode`；
   * 当 M1 进入 idle 段，调用 `/sleep`；M2 进入 active 段时 `/wake_up`，反之亦然。

3. **gBPF-Sleep policy：driver 侧模拟 Sleep 行为**

   * 同样两个 vLLM 实例，但**不启用 Sleep mode**；
   * gBPF 在 UVM hook 上挂一个 policy：

     * 监控每个 engine 的活动状态（request 数量、最近一次 kernel 时间等，通过 host hook + device hint）；
     * idle 时间超过 T（和 Sleep 配置一致）时：

       * 把对应 model weights 的 region 从 HBM 逐步迁回 CPU/CXL；
       * 清空其 KV region；
     * 有新请求到达时，按 region 级别把对应 weights 按需拉回 HBM。

**指标：**

* 每次模型切换的：

  * GPU 内存占用（最低点）
  * CPU 内存占用
  * time-to-first-token（从第一个请求到第一 token）
* 长时间 run 下：

  * 平均吞吐（tokens/s）；
  * 切换次数 N=10/50/100 时的平均/尾部切换延迟；
  * 总 CPU↔GPU 迁移的数据量和带宽。

**你要讲的故事：**

* 在这类“真正 idle 很久”的场景：

  * gBPF-Sleep 在 GPU 内存回收率上不输 Sleep mode（甚至更小粒度控制，允许保留一部分 hot 层/embedding）；
  * wake 延迟和 Sleep 同量级（远远好于 Base A 的 reload）；
  * 优势点：gBPF 政策可以直接复用到其它框架（llama.cpp / PyTorch / SGLang），而 Sleep 只适配 vLLM。

---

### RQ-S2：在 idle 很短 / 存在部分并发的场景，细粒度 gBPF 内存调度能否显著优于 Sleep+router？

> 当多个模型交错、有部分重叠、idle 时间不稳定时，
> sleep 模式的“全 offload/全 discard”会导致频繁 sleep/wake thrash，
> 而 gBPF 的 region 级调度是否能在 throughput / tail latency / GPU 利用率上明显更好？

这才是你真正要打的点：Sleep 模式强调的是**几秒级 / 分钟级**的 idle；
而在 agent / multi-model workflow / RLHF 里，很多切换其实是「几十毫秒到几百毫秒一来一回」，Sleep 模式不适合这么细。

#### 实验设计：多模型 agent pipeline + 交错请求

**环境：**

* 同一块 GPU，两三个模型：

  * M1：chat/instruction 模型（7B/13B）；
  * M2：reasoning 模型（70B）；
  * 可选 M3：small reranker / reward model。

**Workload：**

* 构造一个 agent/workflow trace：

  * 例如：

    * 用户请求先经 M1 理解 → 调用工具 → 然后 M2 做长推理 → 最后 M1 再整理；
  * 多条会话交织，导致：

    * M1/M2/M3 之间的 active/idle 间隔在 10ms–1s 级别波动；
    * 可能存在短时间三者一起 active。

**对比方案：**

1. **Sleep+router：**

   * 为每个模型开一个 vLLM 实例（开 Sleep）；
   * router 基于 idle 时间阈值决定对哪个 engine 调 `/sleep` / `/wake_up`；
   * 因为 Sleep 的语义是“sleep 时不能处理请求”，所以 router 必须等待 wake 完成才能把新请求打过去。

2. **gBPF-fine policy：**

   * 不用 Sleep，所有 engine 常驻；
   * gBPF 在 UVM + scheduler hook 上做两件事：

     * memory：按 region 热度对各模型的权重 + KV / cache 做分层：

       * 很久不用的层全 offload；
       * 频繁切换的模型只 offload 冷层或冷 KV page；
       * 允许**部分共享 HBM**，而不是 0/1；
     * scheduling：设 per-tenant priority，对高优模型的 kernel admission & preemption 优先。

**指标：**

* 对每个模型：

  * p50/p95/p99 latency（尤其是跨模型切换里第一 token 的 latency）；
  * throughput（tokens/s）
* 整体：

  * GPU SM 利用率；
  * HBM 利用率；
  * CPU↔GPU / CXL 带宽利用率；
  * 每秒 sleep/wake 次数（基线） vs 每秒 region 迁移次数（gBPF）。

**预期/可讲的点：**

* Sleep+router 在这种“频繁切换”的 trace 下，要么：

  * idle 阈值设大：几乎不 sleep，GPU 内存浪费；
  * idle 阈值设小：频繁 sleep/wake，导致冷启动级的延迟抖动；
* gBPF 可以：

  * 不需要真正 sleep engine，靠 region 级 memory 调度在几十毫秒级别做平滑；
  * 总体吞吐 & tail latency 更稳定（没有“突然几百 ms 的 wake spike”）。

这里就是你真正可以 claim 的：

> **Sleep mode 适合 coarse-grain 的“晚上 2 点没人用就睡了”；
> gBPF 适合 fine-grain 的“几十 ms 级 agent/multi-model 交错”。**

---

### RQ-S3：多租户 oversub + 在线服务里，Sleep mode 本质上帮不上忙，baseline 还是 vLLM CPU-offload / KV 策略

这其实是你 paper 里已经做的那部分：

* vLLM 30B MoE 上，用 gBPF 的 KV prefetch/evict policy 对比：

  * vLLM 自带 CPU-offload（`--cpu-offload-gb`）；
  * UVM-only baseline。

在典型在线、多租户、高并发的场景：

* 模型不会“完全 idle 几秒/几分钟”；
* 你真正需要的是 **per-request / per-token KV 调度 + 多租户 HBM/带宽分配**；
* Sleep mode 语义上没法介入（sleep 时 engine 不能服务，请求不能并发打进来）。

所以对于这类 RQ，合理的 baseline 还是：

* vLLM 的 CPU-offload / Page-d / FlexiCache / Oneiros 这一类用户态 KV 系统；
* Sleep mode 可以在 text 里解释为“不适用（idle 时间不足以 justify 睡觉）”，而不需要硬画一条线（否则会被说 unfair）。

---

## 3. 和 Sleep mode “合理比较”的几个具体建议

1. **只在 Sleep 设计的使用场景里拿来做 baseline**

   * “长时间 idle → 需要把模型整坨赶出 GPU，为别的 job 腾空间”
   * 多模型 sequential 的 pipeline（Zero-reload model switching）
     在这些场景里，你要诚实地看：
   * gBPF 模拟 Sleep 行为能不能做到类似的 GPU/CPU 内存回收和 wake 延迟；
   * 如果做不到，就老实承认 Sleep 在这个点上非常强，你的卖点是“对更多框架、更多 workload 通用”。

2. **在 “频繁切换 / 存在并发” 场景里，Sleep+router 是个合理 baseline，但要把策略讲清楚**

   * 明确 router 的行为（idle 阈值，如何选哪个 engine sleep）；
   * 尽量帮 baseline 调到比较好的点（比如 sweep idle timeout，看 throughput / tail latency 的折中）；
   * 不要故意给一个“极端小的 idle 阈值”让它自己打自己。

3. **不要把 Sleep mode 当作 oversub KV 的 baseline**

   * oversub 在线服务里，Sleep 根本不会触发（engine 不能动不动 sleep）；
   * baseline 就是 vLLM CPU-offload / Page-d / 其他 KV 管理工作。

4. **给 Sleep mode 一个“我们不会取代它”的定位**

   * Paper 里可以写得很直接：

     * Sleep mode = 应用层 coarse “engine hibernation”；
     * gBPF = driver/device 层 fine-grain policy plane；
   * 甚至可以做一个表：

     * 行：Offload 粒度（engine / tensor / page / region）、适用 idle 时间（秒、毫秒）、是否支持并发、是否跨框架；
     * 列：Sleep / CPU-offload / gBPF policy。

这样审稿人看到的是：你不是要“吊打 Sleep”，而是：

* 在 **Sleep 最擅长的简单场景** 下，你能做到不至于被碾压；
* 在 **Sleep 做不到/不合适的主战场**（oversub、多租户、毫秒级交错）里，你有一套系统级的 memory scheduler story，基线是 vLLM CPU-offload + 其它 KV 系统，而不是 Sleep。

---

## 4. 小结一下怎么设计这条线

如果一句话总结：

> 想和 vLLM Sleep mode 比“合理”，就分两步：
> 先设计一个 coarse-grain 的 gBPF Sleep-style policy，在纯 idle 场景对标 Sleep，证明你不会比它差太多；
> 再拿真正 fine-grain 的 gBPF memory scheduler，在频繁切换/多租户/oversub 场景里对标 vLLM CPU-offload 等用户态方案，说明 Sleep 根本不适用，而 gBPF 可以。

如果你愿意，下一步你可以把你现在已经有的 vLLM 实验（KV offload 那个）以及你准备做的 multi-model 场景稍微写成一个 eval 小节草稿，我可以直接帮你把 “Sleep baseline + gBPF policy” 那几段的文字、图名、指标都控形控词到 OSDI level。


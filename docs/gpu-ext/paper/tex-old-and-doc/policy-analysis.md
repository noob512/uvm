# \sys Policy Analysis: Locality and Prefetch/Eviction Strategies

本文档详细分析各个 workload 的 access pattern、locality 特性，以及 \sys 如何利用这些特性设计 prefetch/eviction 策略。

---

## \sys 架构概述

### 核心设计：eBPF + struct_ops 风格的策略接口

\sys 使用 **eBPF** 在 GPU 驱动层定义了 **struct_ops 风格的接口**，允许 policy 开发者：
1. **定义自定义策略**：实现 prefetch/eviction 的决策逻辑
2. **修改运行时行为**：在不修改应用二进制的情况下改变内存管理行为
3. **获取运行时信息**：通过 uprobe/kprobe 获取语义信息

```
\sys 架构：
┌─────────────────────────────────────────────────────────────┐
│                      User Space                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Application │  │ Policy Dev  │  │ Control Plane       │  │
│  │ (unmodified)│  │ (eBPF code) │  │ (config/标记)        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                      eBPF Layer                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ struct_ops 风格接口：                                    ││
│  │  - on_page_fault(addr, ctx) → prefetch_decision         ││
│  │  - on_eviction_needed(pressure) → evict_pages           ││
│  │  - on_alloc(addr, size, tag) → placement_hint           ││
│  │  - ...                                                   ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ eBPF Maps   │  │ Uprobe Hooks│  │ Kprobe Hooks        │  │
│  │ (状态存储)   │  │ (应用语义)   │  │ (驱动事件)          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    GPU Driver (UVM)                          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ 原有行为被 eBPF policy 覆盖/增强                         ││
│  │  - Page fault handling → 调用 on_page_fault()           ││
│  │  - Eviction → 调用 on_eviction_needed()                 ││
│  │  - Migration → 受 policy 控制                           ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### 语义获取的多种方式

| 方式 | 描述 | 精度 | 开销 | 使用场景 |
|------|------|------|------|----------|
| **纯 pattern-based** | 仅基于 page fault 地址序列 | 低-中 | 最低 | 通用场景 |
| **Control plane 标记** | 管理员预先配置内存区域语义 | 中 | 低 | 已知内存布局 |
| **Uprobe hook** | Hook 应用函数获取运行时语义 | 高 | 中 | 需要精确语义 |
| **框架集成** | 框架主动通知 \sys | 最高 | 依赖框架 | 框架愿意配合时 |

### 和现有方案的关键区别

| 方案 | 修改应用？ | 获取语义？ | 粒度 |
|------|----------|----------|------|
| **Framework-level offload** | ✓ 需要 | ✓ 有 | Expert/Layer |
| **UVM default** | ✗ 不需要 | ✗ 无 | Page (LRU) |
| **cudaMemAdvise** | ✓ 需要 | ✓ 静态 | Region |
| **\sys** | ✗ 不需要 | ✓ 可选 | Page (可编程) |

---

## 0. Page Fault Pattern 可视化分析

基于 `img/pattern/` 目录下的 VA Access Density Heatmap（X轴=时间，Y轴=虚拟地址，颜色=访问频率），我们观察到以下 pattern：

### 0.1 llama.cpp Prefill
**Pattern**：非常清晰的**重复锯齿形**
- 多次重复的 sequential scan（从低地址到高地址）
- 每个"锯齿"对应一次 layer 的 GEMM 操作
- Access count 很高（5000+），说明是 compute-intensive
- **结论**：高度可预测的 stride pattern，非常适合 stride prefetch

### 0.2 llama.cpp Decode
**Pattern**：模糊的斜线上升，有较大 spread
- 访问地址随时间缓慢上升（sequential 趋势）
- 但不是紧密的 stride，有 spread
- 这是 MoE expert 访问的特征：不同 token 选择不同 experts
- **结论**：有 temporal locality，但需要 LFU 而非纯 stride prefetch

### 0.3 Faiss Build
**Pattern**：多条**平行斜线**
- 多个 concurrent 的 sequential scans
- 可能是 K-means 的多个 iterations 在不同 memory regions 操作
- **结论**：适合 multi-stream stride prefetch

### 0.4 Faiss Query
**Pattern**：均匀分布的"噪声" + 一条亮线
- 几乎没有明显的 sequential/stride pattern（random access）
- 但有一条高频访问的亮线 - 可能是 centroids
- **结论**：难以预测，但 centroid 可以 pin；posting list 内部用 stride prefetch

### 0.5 PyTorch DNN Training
**Pattern**：周期性块状访问 + 中间一段 sequential
- 上部和下部：周期性的块状 pattern（layer weights 的重复访问）
- 中间（15000-20000ms）：一段 sequential 访问（某个大 tensor）
- **结论**：强 temporal locality，epoch-aware LFU 可以利用重复访问

### 0.6 总结表

| Workload | Pattern 特征 | 可预测性 | 推荐策略 |
|----------|-------------|---------|---------|
| llama.cpp Prefill | 清晰重复锯齿形 | **高** | Stride prefetch |
| llama.cpp Decode | 模糊斜线，有 spread | **中** | Temporal-aware LFU + mild prefetch |
| Faiss Build | 多条平行斜线 | **高** | Multi-stream stride prefetch |
| Faiss Query | 均匀噪声 + 亮线 | **低** | Centroid pinning + LFU |
| PyTorch DNN | 周期性块状 | **高（跨 epoch）** | Epoch-aware prefetch + LFU |

---

## 1. llama.cpp MoE Expert Offloading (GPT-OSS-120B)

### 1.1 Workload 特性
- **模型规模**：59 GiB，116.83B 参数
- **GPU 内存**：32GB（RTX 5090）
- **Oversubscription**：1.84×，必须做 CPU-GPU memory tiering
- **MoE 结构**：每个 token 只激活 top-k experts

### 1.2 Access Pattern 分析

#### Prefill vs Decode 的 Pattern 差异（参见 Section 0 的可视化）

| 阶段 | Pattern | 特点 | 策略 |
|------|---------|------|------|
| **Prefill** | 清晰重复锯齿形 | 多次 sequential scan，高度可预测 | Stride prefetch 非常有效 |
| **Decode** | 模糊斜线，有 spread | Expert 选择的 temporal locality | LFU + mild prefetch |

#### 两层局部性

**层1：Expert-level Temporal Locality**
- 相邻 tokens 倾向于选择相似的 experts
- Expert 选择分布是 skewed 的：某些 experts 被激活的频率远高于其他
- 例：expert 3 刚被使用，下一个 token 很可能还会用 expert 3
- **从 decode 的 heatmap 可见**：访问地址有上升趋势但有 spread，说明不是完全随机

**层2：Page-level 局部性**

| 局部性类型 | 现象 |
|-----------|------|
| Spatial (GEMM stride) | Expert 内部的矩阵运算是 sequential/strided 访问（prefill 的锯齿形） |
| Spatial (expert layout) | 一个 expert 的 weights 在内存中是连续的 |
| Temporal (hot params) | Expert 内部某些参数被访问的频率更高（并非均匀访问）|

### 1.3 现有方案的问题

**Framework-level offload（llama.cpp `--gpu-layers`）：**
- 把整个 expert 当作 atomic unit
- 每个 token decode 都要等 CPU→GPU 传输完成（在 critical path 上）
- 无法 overlap compute 和 transfer

**UVM default：**
- 只有 reactive LRU：page fault 发生后才迁移
- 不知道 expert 语义，无法预测
- 无法区分 hot/cold pages

**cudaMemAdvise hints：**
- 需要应用显式调用
- MoE 的 expert 选择是 runtime 决定的，static hints 不 work

### 1.4 \sys Policy 设计

**核心思想**：结合 page fault pattern 检测与可选的语义信息

\sys 提供**多层次的语义获取能力**，policy 开发者可以根据需要选择：

```
语义获取方式（从低到高）：
├── Level 0：纯 pattern-based
│   └── 仅基于 page fault 地址序列检测 stride/temporal locality
├── Level 1：Control plane 标记
│   └── 管理员预先标记内存区域（如：expert 0 = [addr_start, addr_end]）
├── Level 2：Uprobe hook
│   └── Hook llama.cpp 的内存分配函数，动态获取 expert/KV 边界
└── Level 3：框架集成
    └── 框架主动通知 \sys 即将访问的区域（最高精度，但需框架配合）
```

```
Prefetch 策略：
├── Stride Prefetch：检测 GEMM 的 sequential/strided access pattern
│   └── 观察连续 page fault 地址，预测下 N 个 pages
├── Adaptive Prefetch：根据 PCIe 带宽利用率动态调整
│   └── 带宽空闲时更激进，拥塞时保守
├── 基于 fault history 的 speculative prefetch
│   └── eBPF maps 记录最近访问的 page regions
└── （可选）语义增强的 prefetch
    └── 如果通过 uprobe 知道当前在哪个 expert，可以预取相邻 expert

Eviction 策略：
├── LFU (Least Frequently Used)：page 粒度的 frequency tracking
│   └── 保留 hot pages，驱逐 cold pages
├── 打破 expert 作为 atomic unit 的假设
│   └── 同一个 expert 的 hot pages 留 GPU，cold pages 放 CPU
└── （可选）语义感知的差异化驱逐
    └── 如果知道 KV vs weights，可以用不同的驱逐策略
```

**关键优势**：
- Page 粒度的 fine-grained tiering，而非 expert 粒度
- Prefetch 和 compute 可以 overlap（不在 critical path 上）
- **不需要修改应用二进制**：语义通过外部 policy/uprobe 获取
- **灵活性**：从纯 pattern-based 到语义增强，按需选择精度-复杂度 tradeoff

### 1.5 论文中的解释建议

```latex
The eBPF policy exploits two levels of locality: (1) spatial locality from
sequential GEMM access within experts, enabling stride-based prefetching that
detects page fault patterns and prefetches ahead of the access frontier; and
(2) temporal locality at page granularity, using LFU-based eviction to retain
hot parameters while evicting cold regions. Unlike framework-level offloading
that treats experts as atomic units, \sys's page-granularity policies achieve
finer-grained memory tiering without requiring application modification.
```

---

## 2. vLLM KV-cache Offloading (Qwen-30B MoE)

### 2.1 Workload 特性
- **模型规模**：~30GB（FP8 MoE）
- **GPU 内存**：32GB（几乎装满）
- **Workload**：100 concurrent requests，~60K tokens aggregate
- **Memory footprint**：36-40GB（model + KV-cache）
- **挑战**：需要同时 offload MoE experts 和 KV-cache

### 2.2 Access Pattern 分析

vLLM 比 llama.cpp 更复杂，因为有两类数据需要管理：

#### KV-cache 的局部性

| 局部性类型 | 现象 |
|-----------|------|
| Temporal (recent tokens) | Attention 访问最近生成的 tokens 更频繁 |
| Spatial (per-request) | 同一个 request 的 KV-cache 是连续的 |
| Temporal (cross-request) | 不同 requests 之间 KV-cache 访问是 interleaved 的 |

#### MoE Experts 的局部性
（同 llama.cpp 分析）

#### 两者的交互
- **竞争关系**：KV-cache 和 expert weights 竞争 GPU 内存
- **访问 pattern 不同**：
  - KV-cache：decode 时 sequential scan（attention）
  - Expert weights：sparse activation（top-k selection）
- **UVM 问题**：同时 on-demand migration 两者会导致 thrashing

### 2.3 现有方案的问题

**vLLM CPU-offload（`--cpu-offload-gb 8`）：**
- 只能选择 offload KV-cache 或 experts，不能同时优化
- 需要预先指定 offload 大小

**UVM without policy：**
- 比 vLLM 的 framework offload 更差
- 两类数据的 page fault 互相干扰
- LRU 无法区分 KV-cache 和 expert weights 的不同访问 pattern

### 2.4 \sys Policy 设计

**核心思想**：Sequential prefetch based on PCIe traffic patterns

```
Prefetch 策略：
├── KV-cache aware sequential prefetch
│   ├── 检测 attention 的 sequential scan pattern
│   └── Prefetch 当前 request 的后续 KV pages
├── PCIe traffic adaptive
│   ├── 监控 PCIe 带宽利用率
│   └── 避免 KV-cache 和 expert prefetch 互相 thrash
└── Request-aware batching
    └── 优先 prefetch 即将被 scheduled 的 request 的 KV

Eviction 策略：
├── 区分 KV-cache 和 model weights
│   ├── Model weights：LFU，保留 hot experts
│   └── KV-cache：考虑 request priority 和 token position
└── Multi-tenant aware（如果有多个 requests）
    └── 优先保留 high-priority request 的 KV
```

**关键 insight**：
- KV-cache 的 sequential scan 是可预测的
- Expert activation 的 temporal locality 可以利用
- 通过 PCIe traffic monitoring 避免两者 thrash

### 2.5 论文中的解释建议

```latex
The \sys policy combines KV-cache-aware sequential prefetching with
expert-level temporal locality exploitation. For KV-cache, the policy
detects sequential scan patterns from attention operations and prefetches
ahead based on observed PCIe traffic, avoiding thrashing between KV-cache
and expert weight migrations. For experts, page-level frequency tracking
identifies hot regions. This coordinated approach achieves 1.7--2$\times$
TTFT improvement by reducing page fault latency on the critical path.
```

---

## 3. GNN Training (PyTorch GCN)

### 3.1 Workload 特性
- **Graph 规模**：1M-15M nodes，10 edges/node
- **Oversubscription**：最高 2.17×
- **操作**：Graph convolution = sparse matrix multiplication

### 3.2 Access Pattern 分析（结合 Section 0 的可视化）

**Heatmap 显示**：周期性块状访问 + 中间一段 sequential

GNN 的访问 pattern 和 dense NN 非常不同：

| 局部性类型 | 现象 | Heatmap 对应 |
|-----------|------|-------------|
| Spatial (neighbor aggregation) | 访问一个 node 时，会访问其所有 neighbors | 块状 pattern |
| Temporal (epoch iteration) | 同一个 graph 被反复访问（每个 epoch）| 周期性重复 |
| Irregular (graph structure) | 访问 pattern 取决于 graph topology，不是规则的 stride | 不是斜线而是块状 |

**Heatmap 解读**：
- 上部和下部的周期性块状 = layer weights 的重复访问（每个 epoch）
- 中间的 sequential 段 = 某个大 tensor 的顺序访问
- 整体有**强 temporal locality**：同样的 pattern 在不同 epoch 重复

### 3.3 挑战

**和 dense NN 的区别**：
- Dense NN（如 GEMM）：规则的 stride pattern（清晰斜线）
- GNN：访问 pattern 由 graph 结构决定，**irregular**（块状而非斜线）

**但仍有可利用的 locality**：
- Training 是多个 epochs，同样的访问 pattern 会重复（周期性）
- Neighbor aggregation 有 spatial locality（访问 node i 后访问其 neighbors）
- **关键观察**：虽然单个 epoch 内 irregular，但跨 epoch 的 temporal locality 很强

### 3.4 \sys Policy 设计

根据 eval.tex，GNN 用的是 **eBPF scheduling optimization**，而不是 prefetch：

```
策略组合：
├── User-space prefetching（cudaMemPrefetchAsync）
│   └── 应用层知道 graph structure，可以做精确 prefetch
│   └── 但需要修改应用
└── eBPF scheduling optimization（\sys）
    ├── 优化 page fault handling 的优先级
    ├── 当 prefetch 不可用时，加速 fault 处理
    └── 和 user-space prefetch 是 complementary 的
```

**为什么用 scheduling 而不是 prefetch？**
- GNN 的 irregular access pattern 难以在 driver 层预测
- 但可以优化 fault 处理的效率
- User-space prefetch + eBPF scheduling 组合效果最好（1.44× additional speedup）

### 3.5 论文中的解释建议

```latex
GNN training exhibits irregular access patterns determined by graph topology,
making driver-level prefetching less effective than for dense workloads.
Instead, \sys applies eBPF-based scheduling optimization that prioritizes
page fault handling threads, reducing fault latency without requiring access
pattern prediction. This approach is complementary to user-space prefetching:
when applications can express prefetch hints via \texttt{cudaMemPrefetchAsync},
\sys's scheduling optimization provides additional 1.44$\times$ speedup by
accelerating residual page faults under memory pressure.
```

---

## 4. Faiss Vector Search (IVF Index)

### 4.1 Workload 特性
- **Dataset**：SIFT，20M-100M vectors
- **Index 类型**：IVF4096,Flat
- **两种操作**：Index build 和 Query search
- **Oversubscription**：最高 48GB / 32GB = 1.5×

### 4.2 IVF Index 结构

```
IVF Index 结构：
├── Centroids（4096 个 cluster centers）
│   └── 小，通常 fit in GPU memory
└── Posting Lists（每个 centroid 对应的 vectors）
    └── 大，是主要的内存消耗
    └── 每个 posting list 大小不均（skewed）
```

### 4.3 Access Pattern 分析（结合 Section 0 的可视化）

#### Index Build（Heatmap 显示：多条平行斜线）
| 阶段 | 访问 pattern | Heatmap 对应 |
|------|-------------|-------------|
| K-means clustering | Sequential scan over all vectors | 每条斜线是一次 iteration |
| Posting list construction | Sequential write to each posting list | 平行斜线 = 多 stream 并发 |

**特点**：
- 主要是 sequential access，可以做 stride prefetch
- 多个 memory regions 同时被 sequential 访问 → multi-stream prefetch

#### Query Search（Heatmap 显示：均匀噪声 + 一条亮线）
| 阶段 | 访问 pattern | Heatmap 对应 |
|------|-------------|-------------|
| 找最近的 centroids | 访问所有 centroids（小，fit in memory）| 亮线 = 高频访问的 centroids |
| 访问 posting lists | 访问 top-nprobe 个 posting lists | 噪声 = random posting list 访问 |

**特点**：
- Centroid 访问：所有 query 都访问，temporal locality 强（heatmap 中的亮线）
- Posting list 访问：依赖 query，有一定 randomness（heatmap 中的噪声）
- 但 posting list 内部是 sequential scan（噪声中仍有局部 stride pattern）

### 4.4 "Adaptive Tree-based Prefetching" 解释

IVF 是一个 **two-level tree structure**：

```
Tree Structure：
Level 0: Centroids (root nodes)
Level 1: Posting Lists (leaf nodes)

访问 pattern：
Query → 访问 centroid → 选择 top-k centroids → 访问对应 posting lists
```

**Policy 开发者如何利用 IVF 知识**：

\sys 的 policy 开发者**知道 IVF 的层次结构**，可以设计专用算法：

```
语义获取方式：
├── Uprobe hook Faiss 的索引加载函数
│   └── 获取 centroid 地址范围、每个 posting list 的起始地址
├── Control plane 配置
│   └── 预先标记 centroid region 和 posting list regions
└── Pattern 观察验证
    └── Heatmap 中的"亮线"确认是 centroids
```

**Adaptive tree-based prefetching 的含义**：

```
Prefetch 策略（基于 IVF 领域知识）：
├── Level 0 (Centroids)：
│   └── Policy 知道 centroid 地址范围，标记为"常驻 GPU"
├── Level 1 (Posting Lists)：
│   ├── 通过 uprobe 或 pattern 检测到 centroid i 被访问
│   ├── Policy 知道 centroid i 对应 posting list i 的地址
│   ├── 主动 prefetch posting list i
│   └── Adaptive：根据 posting list 大小和 PCIe 带宽调整 prefetch 量
└── 内部 Sequential prefetch：
    └── Posting list 内部是 sequential scan，做 stride prefetch
```

**为什么叫 "adaptive"**：
- Posting list 大小不均（skewed distribution）
- 大的 posting list：分批 prefetch，避免 thrash 其他数据
- 小的 posting list：可以完整 prefetch
- 根据 PCIe traffic 动态调整

**和 Framework-level 的区别**：
- Framework-level 需要修改 Faiss 代码来添加 prefetch 调用
- \sys 通过 uprobe hook + policy 实现，**不改 Faiss 二进制**

### 4.5 \sys Policy 设计

```
Index Build 策略：
├── Sequential prefetch：检测 sequential scan pattern
└── 效果：21-29% build time reduction

Query Search 策略：
├── Centroid pinning：高频访问，保持在 GPU
├── Posting list prefetch：
│   ├── 观察 centroid 访问 → prefetch 对应 posting list
│   └── Posting list 内部做 sequential prefetch
└── Adaptive sizing：根据 posting list 大小调整 prefetch 量
效果：10-16% latency reduction
```

### 4.6 论文中的解释建议

```latex
IVF indexes have a two-level tree structure: centroids (frequently accessed,
small) and posting lists (query-dependent, large). The adaptive tree-based
prefetching policy exploits this hierarchy: when a centroid access is detected,
\sys prefetches the corresponding posting list before the subsequent scan.
Within posting lists, stride-based prefetching handles sequential vector
access. The policy adapts prefetch aggressiveness based on posting list size
and observed PCIe bandwidth utilization, avoiding over-fetching for large
posting lists while fully prefetching small ones.
```

---

## 5. 总结对比

| Workload | 主要 locality | Prefetch 策略 | Eviction 策略 |
|----------|--------------|--------------|--------------|
| **llama.cpp MoE** | Temporal (expert) + Spatial (GEMM stride) | Stride prefetch + LFU-based speculative | LFU page-level |
| **vLLM KV+MoE** | Sequential (KV scan) + Temporal (expert) | KV-aware sequential + PCIe adaptive | Differentiate KV vs weights |
| **GNN Training** | Irregular (graph-dependent) | User-space prefetch (需改代码) | eBPF scheduling optimization |
| **Faiss IVF** | Hierarchical (centroid→posting) + Sequential (list scan) | Tree-based adaptive prefetch | Centroid pinning + LFU |

### 关键 Insight

1. **Page-level > Expert/Region-level**：细粒度的 page-level 策略比粗粒度的 region-level 更有效

2. **Spatial + Temporal 组合**：大多数 workload 同时有两种 locality，需要组合策略

3. **Adaptive 是关键**：根据 PCIe traffic、数据大小、访问 pattern 动态调整

4. **灵活的语义获取**：\sys 可以通过多种方式获取语义信息：
   - **纯 pattern-based**：仅基于 page fault 轨迹，无需任何语义
   - **Control plane 标记**：管理员预先标记内存区域（KV-cache、expert weights 等）
   - **Uprobe hook**：动态获取应用运行时信息（如内存分配的用途）
   - **Policy 开发者领域知识**：针对特定应用（如 IVF 索引）设计专用策略

5. **和 framework-level 的关键区别**：
   - **Framework-level**：需要**修改应用代码**来实现 offload
   - **\sys**：**不改应用二进制**，通过外部 policy + uprobe/标记获取语义
   - 两者可以互补：framework 做粗粒度 offload，\sys 做 page-level 优化

下面把你文档里“**\sys 与框架级(offload)是互补的**”这句话掰开揉碎，给出技术判定、边界条件与对比，并补一组可引用的相关工作。先下结论，然后展开。

---

## 结论（先给答案）

* **总体判断**：你的表述**基本正确但过于笼统**。
  – 说“框架级 offload 往往把迁移放在关键路径上”并不总成立：不少系统已经显式做了**异步、流水化的预取与重叠**（FlexGen、DeepSpeed ZeRO‑Inference、MoE‑Infinity、Pre‑gated MoE 的 MoE‑Prefetch）——它们确实把 CPU→GPU 传输搬到当前算子/层的计算期间去重叠。([arXiv][1])
  – 但这些方案**强依赖应用/框架语义**（知道“下一层/下一个 expert 会用什么”），且**迁移粒度多为“层/专家/块”级**。当访问规律细到“**页面级**（同一 expert 内部的 hot/cold page）”或“跨应用、跨框架混部”的时候，框架侧难以精细管理。这里，你的 \sys 如果真能在**UVM 的页面层**上做“预测性预取 + 频次驱逐”，就能覆盖框架做不到的一段空间。
* **你这套 framework 能补以前工作的缺口**主要在三点：

  1. **跨工作负载 & 无语义**：不改应用、不绑定某个推理引擎（vLLM、llama.cpp、Faiss、GNN），直接利用**页故障时空模式**做决策；
  2. **更细的粒度**：把“整个专家/层”拆到**页面**，把“热页留 GPU、冷页放 CPU”；
  3. **统一协调**：同一套策略同时调度**KV cache 与权重**、**训练与检索**，而不是各个框架在自己的小世界里做各自的 offload。UVM 的默认行为以**按 LRU/近似 LRU 的页面（或区间）置换**与**反应式迁移**为主（还带有“密度预取”等启发式），这从公开文献与官方资料可查。([ACM Digital Library][2])

---

## 为什么说“正确但不完整”：逐点核查你文档里的对比

### 1) “框架级 offload 把传输放在关键路径上”

* **不总是**。已有工作显式做**重叠**：

  * **FlexGen**：对算子/层级别的计算与 I/O（CPU/SSD）做调度，追求**高吞吐**，核心就是把传输与算子计算流水化。([arXiv][1])
  * **ZeRO‑Inference**（DeepSpeed）：把权重分块放 CPU/NVMe，在前向时**按需拉取并重叠**，以极小 GPU 内存推大模型。([DeepSpeed][3])
  * **MoE‑Infinity**：对专家激活做请求/序列级追踪，据此**预测下一步专家并预取**，与当前专家计算重叠。([arXiv][4])
  * **Pre‑gated MoE (ISCA’24)**：明确提出 **MoE‑Prefetch**——“在当前 MoE block 计算期间，迁移下一 block 将用到的整个专家”，目标就是**隐藏迁移时延**。([microsoft.com][5])
  * **vLLM 的 `--cpu-offload-gb`** 与 LMCache 也支持把一部分模型/缓存放在 CPU，并在**前向中按需搬运**；官方文档直白地提示需要高速互联以支撑“on‑the‑fly”迁移。是否充分重叠与实现版本相关，但它不是“天然在关键路径”。([vLLM][6])
* **因此应改成**：很多框架**可以**把迁移移出关键路径，但**前提**是：
  – 你能**较准确地预测即将访问的区域**（下一层/下一个 expert/下一个 KV 块）；
  – 迁移粒度通常是**层/专家/块**，难以做到**页内冷热**差异化。

### 2) “UVM 只有 LRU、反应式，无法预测”

* **大体成立**，但要精确表述：
  – 公共研究对 NVIDIA UVM 的测量显示：UVM 的驱逐**体现出 LRU 特征**（不同版本/设备细节会有差异，部分实现对迁移单位采用“区间/大页”管理，甚至表现为“**最近故障最久者先出（LRF）**”的近似策略）。在**超额订阅**下，LRU/区间化驱逐容易把“热点页”赶走，引发颠簸。([ACM Digital Library][2])
  – 驱动侧确有**“密度预取(density prefetching)”**等启发式，但仍是**被动**、以故障为触发。([NVIDIA Developer][7])
* **所以你的优势**应描述为：“在**页面**粒度上进行**主动**（预测性）的迁移与**频次感知的驱逐**，而不是 region/expert 粒度的‘整块搬来搬去’”。

### 3) “cudaMemAdvise 静态，不适用 MoE 的运行时动态”

* **方向对**：`cudaMemAdvise` 的 `SetPreferredLocation/SetAccessedBy/ReadMostly` 等本质是**提示**；MoE 的 expert 选择是**运行时**产生、跨 token 变化的，静态提示覆盖有限。官方文档更多是解释其语义，并未提供“按专家层次动态切换”的通用套路。([NVIDIA Docs][8])

---

## 你的 framework（\sys）到底补了什么“研究空白”？

> 用“维度—对比”的方式说明，你的系统与已有代表性工作之间的**能力边界**。

| 维度       | 典型框架/系统                                           | 能做到                                 | 做不到/难做到                                     | \sys（页策略 + eBPF 观测 + 用户态触发）能补哪里                                 |
| -------- | ------------------------------------------------- | ----------------------------------- | ------------------------------------------- | --------------------------------------------------------------- |
| 作用层级     | FlexGen、ZeRO‑Inference、MoE‑Infinity、Pre‑gated MoE | 基于**算子/层/专家/块**布置迁移与预取；能**重叠**计算与传输 | 难以在**expert 内部**区分**热/冷页**；多框架并存时**全局协调**不足 | 在**页面**粒度做**热度感知(LFU/LRFU)** 与**顺序/步幅**预取；面向**任意二进制**（不改应用）统一调度 |
| 需要语义     | 需要（下一层、下一个 expert、nprobe 列表等）                     | 能精准预取，但**错误预测**会回到关键路径              | 对**黑盒应用**（闭源推理引擎、检索库、传统 HPC）覆盖差             | **无语义**，通过**故障轨迹**识别顺序/重复/热点                                    |
| 资源竞争     | vLLM/LMCache、MoE‑Infinity 各管各的                    | 各自优化 KV 或专家权重                       | **KV 与权重**互相打架时，跨对象的优先级与配额协调困难              | 用同一**页面级策略**区别对象类型（KV vs 权重）与**请求优先级**，统一做**带宽自适应**             |
| 迁移触发     | 框架内策略                                             | 受制于框架调度器                            | 难以跨框架统一度量**PCIe/内存压力**                      | 通过 eBPF/CUPTI 观测**故障与带宽**，由守护进程/注入库触发 `cudaMemPrefetchAsync` 实施 |
| 泛化到非 LLM | 多数只面向推理                                           | GNN/Faiss 等 pattern 很不同             | 跨领域复用成本高                                    | 只看**页访问模式**（热图中的斜线/块状/亮线），策略可迁移                                 |

---

## 相关工作的脉络与定位（带引文）

### LLM/ MoE 侧的框架级 offload 与预取

* **FlexGen**：GPU+CPU+NVMe 三层，显式**调度 I/O 与算子**做重叠，牺牲延迟换吞吐。([arXiv][1])
* **DeepSpeed ZeRO‑Inference/Offload**：把权重/优化器状态放 CPU/NVMe，**在前向/后向间隙搬运**。([DeepSpeed][3])
* **MoE‑Infinity**：**激活追踪 + 专家缓存/预取**，预测下一步专家以**重叠迁移**。([arXiv][4])
* **Pre‑gated MoE (ISCA’24)**：提出 **MoE‑Prefetch**，**在当前 block 执行时迁移下一 block 的专家**。说明“框架级也能把迁移移出关键路径”。([microsoft.com][5])
* **vLLM**：`--cpu-offload-gb` 与 **LMCache** 支持把模型/KV 部分放 CPU 并在前向**按需迁移**；文档提示“对互联带宽敏感”。([vLLM][6])

### UVM/共享虚存（系统层）实证与问题

* **UVM 的驱逐与迁移**：多项研究观测到**LRU/近似 LRU** 的特征；在过载下会把**热点页**逐出，引发 thrashing。([ACM Digital Library][2])
* **UVM 的“密度预取”**：驱动在页故障路径中做基于密度的范围预取，但仍属**被动**启发式。([NVIDIA Developer][7])
* **区间化管理与“最近故障最久者(LRF)”**：SVM/UVM 在某些实现里以**区间/大页**为单位管理，驱逐可能表现为 LRF，这会对 **SGEMM/顺序访问**非常不友好。([arXiv][9])
* **故障处理路径与批处理**：SC’21 分析了 UVM 的**批量故障处理与线程化**细节，说明**故障开销本身可成为瓶颈**，为你在 GNN 上做“调度优先级优化”提供了依据。([Tallendev][10])

这套文献图谱支撑你在文档中的**热图观察**与**策略动机**：prefill 的**清晰步幅**适合**顺序/步幅预取**，MoE decode 的**token‑相邻重用**适合**频次驱逐 + 温和预取**，Faiss/训练的**层/迭代重复**体现**周期性时间局部性**。

---

## 你的系统具体“多做了什么”？

> 站在“先验框架能做什么”的基础上，\sys 若要成立为一类**系统贡献**，需要把下面这些点做扎实：

1. **页面级（而非专家/层级）冷热分离**

   * 依据**故障热度(LFU/LRFU)**保留权重/KV 的**热点页**；对 MoE，打破“专家是原子”的假设，在专家内部分离**hot/cold**。
   * 这是已有 MoE 框架普遍**不做或很难做**的维度（它们的缓存/预取对象通常是**整个 expert**）。([arXiv][4])

2. **无语义的顺序/树形自适应预取**

   * 对 **GEMM/Prefill/Posting‑list scan** 等表现为**斜线/顺序**的热图，做**步幅检测 + 前瞻预取**；
   * 对 **Faiss**，做到“**一级节点（centroid）常驻** + **二级 posting list 自适应分批预取**”。（Faiss/IVF 的层次访问结构是事实，但这是你策略的工程发挥点。）

3. **跨对象、跨请求的统一带宽感知**

   * 以**PCIe/链路利用率**为反馈，动态调节 KV 与权重预取的激进程度，避免互相 thrash。传统做法一般在**框架内部**各自为政。

4. **故障处理调度优化**

   * 在**预测难度高（如 GNN）**或不适合预取的场景下，优先提升**UVM 相关处理线程**的调度/亲和，**缩短故障服务时延**。这与 SC’21 对 UVM 工作线程化/批处理瓶颈的发现一致。([Tallendev][10])

---

## 但也必须诚实写出这些“边界/前提”

* **错误预测/带宽占用控制**：你的文档已写“带宽自适应”，但评测中最好给出**PCIe 利用率—吞吐/延迟曲线**，证明在**错误预测率上升**时系统仍稳健。

---

## 实证建议（怎样把“正确”坐实在图表里）

1. **针对 prefill（锯齿）**：比较 UVM 默认、框架级预取（如 FlexGen/ZeRO‑Inference 风格）与 \sys 页级步幅预取的**TTFT 与 GPU stall time**，展示“锯齿越密集，\sys 的前瞻越有效”。
2. **针对 MoE decode**：

   * 统计 per‑expert 的**页级热度分布**（证明“专家内有热页”）。
   * 展示 **LFU(evict) + 温和预取** 比 LRU/区域驱逐的**fault rate 与端到端延迟**。
   * 同时跑 **KV + MoE**，给出**互不干扰**的数据（与 vLLM 的 `--cpu-offload-gb` + LMCache 做 A/B）。([vLLM][6])
3. **GNN**：展示“**不开预取，仅调度优化**”也能**降低故障服务时延**，呼应 SC’21 的 UVM 观测。([Tallendev][10])
4. **Faiss**：把“centroid 常驻 + posting‑list 自适应分批”对比“整段拉满”的**PCIe 峰值与尾延迟**差异。

---

## 进一步可引用的相关工作（按主题归类）

* **框架级 Offload/预取与 KV 管理**：FlexGen；DeepSpeed ZeRO‑Inference/Offload；vLLM `--cpu-offload-gb` 与 LMCache（KV offload）；KServe/BentoML 的 KV offload 教程/白皮书。([arXiv][1])
* **MoE 专家级缓存/预取**：MoE‑Infinity（激活感知）；Pre‑gated MoE（MoE‑Prefetch）；近期综述/系统（例如 Hobbit/Mixed‑precision offloading）讨论“专家预取的难点与收益上限”。([arXiv][4])
* **UVM/共享虚存机制与性能**：SC’21/’21‑IPDPS 对 UVM 故障与批量处理的实证；UVM 密度预取官方博文；SVM 设计论文对**区间/驱逐策略**的讨论；在**超额订阅**场景下 LRU 的问题与改进方向。([Tallendev][10])
* **其他系统侧探索**：GPUVM（GPU 驱动的统一虚存运行时，展示了改造路径）；memHarvester（多 GPU 内存回收中也采用**基于 LRU 的驱逐**，可与之类比）；“UVM Discard”等减少冗余迁移的工作。([arXiv][11])

---

## 一句话改写你的原命题

> **更严谨的版本**：
> “框架级 offload 能在具备语义的前提下把传输与计算**显式重叠**，但其**粒度受限于层/专家/块**且跨框架难以统一；\sys 在**页面级**基于**故障轨迹**做**预测性预取**与**热度感知驱逐**，可在**无需应用修改**的情况下，细粒度地保住热点、减少关键路径上的故障开销，与框架级方法**互补**。
把这段话写清楚、把实现路径画出来、把“页级优势”用实证打透，你的定位就稳了。

[1]: https://arxiv.org/pdf/2303.06865?utm_source=chatgpt.com "FlexGen: High-Throughput Generative Inference of Large ..."
[2]: https://dl.acm.org/doi/pdf/10.1145/3458817.3480855?utm_source=chatgpt.com "In-Depth Analyses of Unified Virtual Memory System for ..."
[3]: https://www.deepspeed.ai/2022/09/09/zero-inference.html?utm_source=chatgpt.com "ZeRO-Inference: Democratizing massive model inference"
[4]: https://arxiv.org/html/2401.14361v2?utm_source=chatgpt.com "MoE-Infinity: Offloading-Efficient MoE Model Serving"
[5]: https://www.microsoft.com/en-us/research/wp-content/uploads/2024/05/isca24_pregated_moe_camera_ready.pdf?utm_source=chatgpt.com "Pre-gated MoE: An Algorithm-System Co-Design for Fast ..."
[6]: https://docs.vllm.ai/en/stable/cli/serve/?utm_source=chatgpt.com "JSON CLI Arguments - vLLM"
[7]: https://developer.nvidia.com/blog/maximizing-unified-memory-performance-cuda/?utm_source=chatgpt.com "Maximizing Unified Memory Performance in CUDA"
[8]: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html?utm_source=chatgpt.com "4.1. Unified Memory — CUDA Programming Guide"
[9]: https://arxiv.org/html/2405.06811v1?utm_source=chatgpt.com "Shared Virtual Memory: Design & Performance for Apps"
[10]: https://tallendev.github.io/assets/papers/sc21.pdf?utm_source=chatgpt.com "In-Depth Analyses of Unified Virtual Memory System for GPU ..."
[11]: https://arxiv.org/pdf/2411.05309?utm_source=chatgpt.com "GPUVM: GPU-driven Unified Virtual Memory"

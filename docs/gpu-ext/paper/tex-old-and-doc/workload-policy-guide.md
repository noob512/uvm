# \sys Workload Policy Guide

本文档按 workload 组织，详细分析每个应用的：
1. **Pattern 分析**：基于 heatmap 的访问模式观察
2. **Insight & Novelty**：\sys 的独特见解和创新点
3. **相关工作对比**：与现有方案的差异
4. **策略设计**：具体的 prefetch/eviction 策略
5. **实验结果解释**：为什么能获得性能提升

---

## \sys 架构回顾

\sys 使用 **eBPF + struct_ops 风格接口**，在 GPU 驱动层实现可编程的内存管理策略：

```
┌─────────────────────────────────────────────────────────────┐
│ Application (unmodified) │ Policy Dev │ Control Plane       │
├─────────────────────────────────────────────────────────────┤
│ eBPF Layer: struct_ops 接口 + Maps + Uprobe/Kprobe Hooks    │
├─────────────────────────────────────────────────────────────┤
│ GPU Driver (UVM): 行为被 eBPF policy 覆盖/增强              │
└─────────────────────────────────────────────────────────────┘
```

**核心区别**：不修改应用二进制，通过外部 policy + hook 获取语义

---

# Workload 1: llama.cpp MoE Expert Offloading

## 1.1 Workload 概述

| 项目 | 值 |
|------|-----|
| 模型 | GPT-OSS-120B MXFP4 MoE |
| 参数量 | 116.83B |
| 模型大小 | 59 GiB |
| GPU | RTX 5090 (32GB) |
| Oversubscription | 1.84× |

## 1.2 Pattern 分析

### Heatmap 观察

**Prefill 阶段**（见 `img/pattern/llama.cpp-prefill/page-fault-pattern.png`）：
```
Pattern: 清晰的重复锯齿形
        ↗ ↗ ↗ ↗ ↗  (多次从低地址到高地址的 sequential scan)
       ↗ ↗ ↗ ↗ ↗
      ↗ ↗ ↗ ↗ ↗
```
- 每个"锯齿"= 一次 layer 的 GEMM 操作
- Access count 高（5000+）= compute-intensive
- **高度可预测的 stride pattern**

**Decode 阶段**（见 `img/pattern/llama.cpp-decode/page-fault-pattern.png`）：
```
Pattern: 模糊斜线上升，有 spread
         .  . .  . .
        . .  .  . .
       .  . .  .  .    (不是紧密 stride，有分散)
```
- 访问地址随时间缓慢上升
- Spread = 不同 token 选择不同 experts
- **有 temporal locality，但非纯 sequential**

### 两层局部性分析

| 层级 | 局部性类型 | 现象 | Heatmap 证据 |
|------|-----------|------|-------------|
| Expert-level | Temporal | 相邻 token 选相似 expert | Decode 的斜线趋势 |
| Page-level | Spatial (stride) | GEMM 是 sequential 访问 | Prefill 的锯齿形 |
| Page-level | Temporal (hot/cold) | Expert 内部访问不均匀 | 某些区域颜色更亮 |

## 1.3 Insight & Novelty

### 核心 Insight

> **Expert 内部存在 hot/cold page 分布，打破"expert 是 atomic unit"的假设**

现有工作把整个 expert 当作迁移单位，但实际上：
- 一个 expert 的某些参数被频繁访问（hot）
- 另一些参数很少被访问（cold）
- **Page-level 的 LFU 可以只保留 hot pages，驱逐 cold pages**

### Novelty 对比

| 维度 | 现有方案 | \sys |
|------|---------|------|
| 迁移粒度 | Expert/Layer | **Page** |
| 语义需求 | 需要知道 expert 边界 | 可选（uprobe 获取或纯 pattern） |
| Hot/cold 区分 | 无（整体迁移） | **有（page-level LFU）** |

## 1.4 相关工作对比

### Framework-level Offload

| 系统 | 方法 | 局限 |
|------|------|------|
| **FlexGen** | GPU+CPU+NVMe 三层调度，overlap I/O 与计算 | 需要修改应用；粒度是 layer |
| **DeepSpeed ZeRO-Inference** | 权重分块放 CPU/NVMe，前向时拉取 | 粒度是 shard，非 page |
| **MoE-Infinity** | 激活追踪 + expert 预取 | 粒度是**整个 expert** |
| **Pre-gated MoE (ISCA'24)** | MoE-Prefetch：当前 block 执行时迁移下一个 expert | 粒度仍是 expert |
| **llama.cpp --gpu-layers** | 指定多少层放 GPU | 粗粒度，无动态调整 |

**\sys 的差异**：
- 不改应用，通过 uprobe hook 获取 expert 边界
- **Page 粒度**：同一 expert 内 hot pages 留 GPU，cold pages 放 CPU
- 结合 stride prefetch（prefill）和 LFU eviction（decode）

### UVM 默认行为

| 特性 | UVM 默认 | \sys |
|------|---------|------|
| 驱逐策略 | LRU/近似 LRU | **LFU（频次感知）** |
| 预取 | 被动（density prefetch 启发式） | **主动（stride 检测 + 语义增强）** |
| 语义 | 无 | 可选获取 |

## 1.5 策略设计

### 语义获取

```
方式 1：纯 pattern-based（无需任何配置）
├── 检测 stride pattern → prefill 的锯齿形
└── 检测 temporal locality → decode 的重复访问

方式 2：Uprobe hook（更精确）
├── Hook llama.cpp 的 expert 加载函数
├── 获取每个 expert 的地址范围
└── 在 eBPF maps 中记录 expert_id → [addr_start, addr_end]
```

### Prefetch 策略

```c
// struct_ops 接口：on_page_fault
int on_page_fault(u64 fault_addr, struct fault_ctx *ctx) {
    // 1. 检测 stride pattern
    u64 stride = fault_addr - last_fault_addr;
    if (is_consistent_stride(stride)) {
        // Prefill 阶段：清晰 stride，激进预取
        prefetch_ahead(fault_addr, stride, PREFETCH_DEPTH_HIGH);
    } else {
        // Decode 阶段：有 spread，温和预取
        prefetch_ahead(fault_addr, stride, PREFETCH_DEPTH_LOW);
    }

    // 2. (可选) 语义增强：如果知道 expert 边界
    if (expert_info_available) {
        int expert_id = lookup_expert(fault_addr);
        // 预取同一 expert 的相邻 pages
        prefetch_expert_region(expert_id, fault_addr);
    }

    // 3. 更新 page 访问频率（用于 LFU eviction）
    increment_access_count(fault_addr);

    return 0;
}
```

### Eviction 策略

```c
// struct_ops 接口：on_eviction_needed
int on_eviction_needed(u64 pressure_level) {
    // LFU：驱逐访问频率最低的 pages
    struct page_info *victim = find_lfu_page();

    // 关键：即使在同一个 expert 内，也只驱逐 cold pages
    // 打破 "expert 是 atomic unit" 的假设
    evict_page(victim);

    return 0;
}
```

## 1.6 实验结果解释

### 结果数据

| 配置 | Prefill (tok/s) | Decode (tok/s) | 相比 framework offload |
|------|----------------|----------------|----------------------|
| Framework offload (ncmoe=32) | 最高 | 基准 | 1.0× |
| UVM default | 低 | 很低 | - |
| UVM + hints | 更低 | 更低 | - |
| **\sys eBPF prefetch** | 略低 (-13%) | **4.8× 加速** | **整体最优** |

### 为什么 \sys 在 Decode 上获得 4.8× 加速？

1. **Framework offload 的问题**：
   - 每个 token 都要等 expert 从 CPU 搬到 GPU
   - 传输在 critical path 上，无法 overlap

2. **\sys 的优势**：
   - **Stride prefetch**：检测到 sequential pattern 后提前预取
   - **LFU eviction**：保留 hot pages，减少重复迁移
   - **Overlap**：prefetch 在后台进行，不阻塞计算

3. **为什么 Prefill 略低 (-13%)**：
   - Prefill 是 compute-bound，访问模式规律
   - Framework offload 可以精确控制数据位置
   - \sys 的 overhead 在 compute-bound 场景更明显
   - **但 decode 主导 end-to-end latency，4.8× 收益远超 13% 损失**

### 为什么 UVM hints 反而更差？

- `cudaMemAdvise` 是静态 hint
- MoE 的 expert 选择是动态的
- 静态 hint 导致错误的数据放置
- **\sys 的动态策略优于静态 hint**

---

# Workload 2: vLLM KV-cache Offloading

## 2.1 Workload 概述

| 项目 | 值 |
|------|-----|
| 模型 | Qwen-30B FP8 MoE |
| 模型大小 | ~30GB |
| GPU | RTX 5090 (32GB) |
| Workload | 100 concurrent requests, ~60K tokens |
| Memory footprint | 36-40GB (model + KV-cache) |

**挑战**：需要同时管理 MoE experts 和 KV-cache 的 offload

## 2.2 Pattern 分析

vLLM 比 llama.cpp 更复杂，有**两类数据**竞争 GPU 内存：

### KV-cache 访问模式

| 局部性 | 现象 |
|--------|------|
| Temporal (recent) | Attention 访问最近生成的 tokens 更频繁 |
| Spatial (per-request) | 同一 request 的 KV-cache 是连续的 |
| Interleaved | 不同 requests 的 KV 访问交错 |

### MoE Experts 访问模式

（同 llama.cpp 分析）

### 两者的交互问题

```
竞争场景：
KV-cache 增长 → 挤占 expert weights 空间 → expert page fault
Expert 迁入 → 挤占 KV-cache 空间 → KV page fault
→ 互相 thrashing
```

## 2.3 Insight & Novelty

### 核心 Insight

> **KV-cache 和 expert weights 有不同的访问模式，需要差异化策略；UVM 的统一 LRU 导致两者互相 thrash**

### Novelty

| 问题 | 现有方案 | \sys |
|------|---------|------|
| KV vs Weights 竞争 | 各管各的（vLLM 管 KV，MoE 框架管 experts） | **统一策略，区分对象类型** |
| 带宽分配 | 无协调 | **PCIe traffic adaptive** |

## 2.4 相关工作对比

| 系统 | 管理对象 | 局限 |
|------|---------|------|
| **vLLM --cpu-offload-gb** | KV-cache 或 model weights | 二选一，不能同时优化 |
| **LMCache** | KV-cache | 不管 expert weights |
| **MoE-Infinity** | Expert weights | 不管 KV-cache |

**\sys 的差异**：
- 统一管理 KV-cache 和 expert weights
- 通过 uprobe 或 address range 标记区分两类数据
- 不同类型使用不同策略

## 2.5 策略设计

### 语义获取

```
Uprobe hook vLLM 的内存分配：
├── KV-cache 分配 → 标记为 TYPE_KV
├── Model weights 加载 → 标记为 TYPE_WEIGHT
└── 在 eBPF maps 中记录 addr_range → type
```

### 差异化策略

```c
int on_page_fault(u64 fault_addr, struct fault_ctx *ctx) {
    int type = lookup_memory_type(fault_addr);

    if (type == TYPE_KV) {
        // KV-cache：sequential scan pattern
        // 检测 attention 的顺序访问，prefetch 后续 KV pages
        prefetch_sequential(fault_addr, KV_PREFETCH_DEPTH);
    } else if (type == TYPE_WEIGHT) {
        // Expert weights：temporal locality
        // 温和 prefetch + LFU 驱逐
        prefetch_mild(fault_addr);
    }

    return 0;
}

int on_eviction_needed(u64 pressure_level) {
    // 根据当前 phase 决定驱逐优先级
    if (is_decode_phase()) {
        // Decode 阶段：KV-cache 更重要，优先驱逐 cold weights
        evict_lfu_by_type(TYPE_WEIGHT);
    } else {
        // Prefill 阶段：weights 更重要
        evict_lru_by_type(TYPE_KV);
    }
    return 0;
}
```

### PCIe 带宽自适应

```c
// 监控 PCIe 利用率，避免 KV 和 weights prefetch 互相 thrash
int adjust_prefetch_aggressiveness() {
    float pcie_util = get_pcie_utilization();

    if (pcie_util > 0.8) {
        // 带宽紧张，减少 prefetch
        set_prefetch_depth(LOW);
    } else {
        // 带宽空闲，激进 prefetch
        set_prefetch_depth(HIGH);
    }
    return 0;
}
```

## 2.6 实验结果解释

### 结果数据

| 配置 | TTFT (mean) | TTFT (p99) | Decode throughput |
|------|-------------|------------|-------------------|
| vLLM CPU-offload | 基准 | 基准 | 基准 |
| UVM default | 更差 | 更差 | 更差 |
| **\sys** | **1.7-2× 改善** | **1.7-2× 改善** | **1.3× 改善** |
| LMCache | ~\sys | 比 \sys 差 | ~\sys |

### 为什么 \sys 比 vLLM offload 好 1.7-2×？

1. **vLLM offload 的问题**：
   - 静态指定 offload 多少 GB
   - 无法动态适应 KV-cache 增长
   - KV 和 weights 可能互相干扰

2. **\sys 的优势**：
   - **差异化策略**：KV 用 sequential prefetch，weights 用 LFU
   - **动态适应**：根据 PCIe 带宽调整
   - **统一协调**：避免两者互相 thrash

### 为什么 UVM default 比 vLLM offload 更差？

- UVM 的 LRU 无法区分 KV 和 weights
- 两者访问模式不同，统一 LRU 导致错误驱逐
- **\sys 的类型感知策略解决这个问题**

---

# Workload 3: GNN Training (PyTorch GCN)

## 3.1 Workload 概述

| 项目 | 值 |
|------|-----|
| 框架 | PyTorch |
| 模型 | GCN |
| Graph 规模 | 1M-15M nodes, 10 edges/node |
| Oversubscription | 最高 2.17× |

## 3.2 Pattern 分析

### Heatmap 观察（见 `img/pattern/pytorch-dnn/page-fault-pattern.png`）

```
Pattern: 周期性块状 + 中间一段 sequential

上部: ▓▓  ▓▓  ▓▓  ▓▓  (周期性块状 = layer weights)
中间: ╱╱╱╱╱╱╱╱╱╱╱╱╱╱  (sequential = 某个大 tensor)
下部: ▓▓  ▓▓  ▓▓  ▓▓  (周期性块状)
```

### 局部性分析

| 局部性 | 现象 | Heatmap 证据 |
|--------|------|-------------|
| **Temporal (跨 epoch)** | 每个 epoch 访问相同的 graph | 周期性重复的块状 |
| **Spatial (neighbor)** | 访问 node i 时访问其 neighbors | 块状 pattern |
| **Irregular** | 访问顺序由 graph 结构决定 | 不是斜线而是块状 |

### 关键观察

> **单个 epoch 内是 irregular，但跨 epoch 有强 temporal locality**

## 3.3 Insight & Novelty

### 核心 Insight

> **GNN 的 irregular pattern 使得 prefetch 难以预测，但可以优化 page fault handling 的效率**

### Novelty

| 问题 | 传统方法 | \sys |
|------|---------|------|
| Irregular access | 放弃 prefetch | **优化 fault handling 调度** |
| 跨 epoch locality | 无利用 | **LFU 保留热点** |

## 3.4 相关工作对比

| 系统 | 方法 | 局限 |
|------|------|------|
| **cudaMemPrefetchAsync** | 应用显式 prefetch | 需要修改应用 |
| **UVM default** | 被动 fault handling | 无优化 |

**\sys 的差异**：
- 不强求预测 irregular pattern
- 优化 fault handling 的调度优先级
- 与 user-space prefetch **互补**

## 3.5 策略设计

### 为什么不用 prefetch？

```
GNN 的访问模式：
Node 1 → Neighbors of 1 → Node 5 → Neighbors of 5 → ...

问题：下一个访问哪个 node 取决于 graph 结构
在 driver 层无法预测（没有 graph topology 信息）
```

### 策略：eBPF Scheduling Optimization

```c
// 不做 prefetch，而是优化 fault handling
int on_page_fault(u64 fault_addr, struct fault_ctx *ctx) {
    // 1. 记录访问频率（用于 LFU）
    increment_access_count(fault_addr);

    // 2. 提升 fault handling 线程的优先级
    boost_fault_handler_priority();

    return 0;
}
```

### 与 User-space Prefetch 互补

```
组合策略：
├── User-space (cudaMemPrefetchAsync)
│   └── 应用知道 graph 结构，可以精确 prefetch
│   └── 需要修改应用
└── \sys (eBPF scheduling)
    └── 加速 residual fault handling
    └── 不需要修改应用
    └── 两者组合：1.44× additional speedup
```

## 3.6 实验结果解释

### 结果数据

| 配置 | 相比 UVM default |
|------|-----------------|
| User prefetch only | 5.5× 加速 |
| \sys scheduling only | 2.65× 加速 |
| **User prefetch + \sys** | **额外 1.44× 加速** |

### 为什么 \sys scheduling 能获得 2.65× 加速（不用 prefetch）？

1. **UVM fault handling 的瓶颈**（参考 SC'21 论文）：
   - Fault handling 涉及多个内核线程
   - 线程调度影响 fault latency
   - 批处理 vs 单个处理的 tradeoff

2. **\sys 的优化**：
   - 提升 fault handler 线程优先级
   - 优化线程亲和性
   - 减少 fault service latency

### 为什么和 User prefetch 互补？

- User prefetch 减少 fault 数量
- 但在 memory pressure 下仍有 residual faults
- \sys 加速这些 residual faults 的处理
- **两者解决不同的问题**

---

# Workload 4: Faiss Vector Search (IVF Index)

## 4.1 Workload 概述

| 项目 | 值 |
|------|-----|
| Dataset | SIFT |
| Index 类型 | IVF4096,Flat |
| 规模 | 20M-100M vectors |
| Oversubscription | 最高 1.5× (48GB/32GB) |
| 操作 | Index build + Query search |

## 4.2 Pattern 分析

### IVF 索引结构

```
IVF Index (two-level tree):
├── Level 0: Centroids (4096 个 cluster centers, 小)
└── Level 1: Posting Lists (每个 centroid 对应的 vectors, 大)
```

### Heatmap 观察

**Index Build**（见 `img/pattern/faiss-build/page-fault-pattern.png`）：
```
Pattern: 多条平行斜线
    ╱    ╱    ╱
   ╱    ╱    ╱     (多个 concurrent sequential scans)
  ╱    ╱    ╱
```
- 每条斜线 = K-means 的一次 iteration
- 平行 = 多个 memory regions 同时被扫描
- **高度可预测，适合 multi-stream stride prefetch**

**Query Search**（见 `img/pattern/faiss-query/page-fault-pattern.png`）：
```
Pattern: 均匀噪声 + 一条亮线
  ░░░░░░░░░░░░░░░░░░
  ░░░░░░░░░░░░░░░░░░
  ████████████████████  ← 亮线 (high frequency)
  ░░░░░░░░░░░░░░░░░░
```
- 噪声 = posting list 的 random access
- 亮线 = centroids 的高频访问
- **Centroids 可 pin，posting list 难预测**

## 4.3 Insight & Novelty

### 核心 Insight

> **IVF 有 two-level hierarchy：centroids (热) 和 posting lists (冷/随机)。Policy 开发者可以利用这个领域知识设计专用策略。**

### Novelty

| 维度 | UVM default | \sys |
|------|------------|------|
| Centroid 处理 | 和其他 pages 一样 | **识别并 pin** |
| Posting list | 统一 LRU | **检测 sequential scan** |
| 层次感知 | 无 | **tree-based prefetch** |

## 4.4 相关工作对比

| 系统 | 方法 | 局限 |
|------|------|------|
| **Faiss GPU** | 假设数据 fit in memory | 无 offload |
| **UVM** | 被动 fault handling | 无层次感知 |

**\sys 的差异**：
- Policy 开发者知道 IVF 结构
- 通过 uprobe 获取 centroid/posting list 地址
- 针对性策略

## 4.5 策略设计

### 语义获取

```
Uprobe hook Faiss 的索引加载函数：
├── 获取 centroid 地址范围 → 标记为 REGION_CENTROID
├── 获取每个 posting list 的地址 → 标记为 REGION_POSTING
└── 在 eBPF maps 中记录 centroid_id → posting_list_addr
```

### Index Build 策略

```c
// Build 阶段：多条平行斜线 = multi-stream sequential
int on_page_fault_build(u64 fault_addr) {
    // 检测 stride pattern
    int stream_id = identify_stream(fault_addr);
    u64 stride = detect_stride(stream_id);

    // Multi-stream prefetch
    prefetch_ahead(fault_addr, stride, PREFETCH_DEPTH);

    return 0;
}
```

### Query Search 策略

```c
// Query 阶段：centroid pin + posting list prefetch
int on_page_fault_query(u64 fault_addr) {
    int region_type = lookup_region_type(fault_addr);

    if (region_type == REGION_CENTROID) {
        // Centroids：高频访问，标记为常驻
        mark_as_pinned(fault_addr);
    } else if (region_type == REGION_POSTING) {
        // Posting list：内部是 sequential scan
        prefetch_sequential_within_posting(fault_addr);
    }

    return 0;
}

// 基于层次结构的 prefetch
int on_centroid_access(int centroid_id) {
    // 当 centroid i 被访问时，预取对应的 posting list
    u64 posting_addr = lookup_posting_list(centroid_id);
    u64 posting_size = get_posting_size(centroid_id);

    // Adaptive：根据 posting list 大小调整
    if (posting_size < SMALL_THRESHOLD) {
        // 小 posting list：完整预取
        prefetch_full(posting_addr, posting_size);
    } else {
        // 大 posting list：分批预取，避免 thrash
        prefetch_partial(posting_addr, BATCH_SIZE);
    }

    return 0;
}
```

## 4.6 实验结果解释

### 结果数据

| 操作 | \sys vs UVM default |
|------|---------------------|
| Index Build | **21-29% 加速** |
| Query Search | **10-16% 加速** |

### 为什么 Build 获得 21-29% 加速？

1. **Heatmap 显示清晰的 multi-stream stride**
2. **\sys 检测并 prefetch 每个 stream**
3. **减少 fault stall time**

### 为什么 Query 只有 10-16% 加速（而非更高）？

1. **Posting list 访问有 randomness**（heatmap 的噪声）
2. **Centroid pinning 有效，但 posting list 难完全预测**
3. **但 posting list 内部的 sequential scan 仍可优化**

### 为什么收益随 memory pressure 增长（27% → 40%）？

- Memory pressure 低时，大部分数据 fit in GPU，fault 少
- Memory pressure 高时，fault 多，\sys 的 prefetch 收益更大
- **\sys 在 oversubscription 场景价值更高**

---

# 总结：跨 Workload 的 Pattern → 策略映射

| Workload | Pattern 特征 | 核心 Insight | 策略 | 结果 |
|----------|-------------|-------------|------|------|
| **llama.cpp MoE** | Prefill 锯齿 + Decode spread | Expert 内有 hot/cold pages | Stride + LFU | 4.8× decode |
| **vLLM KV+MoE** | KV sequential + Expert temporal | KV/Weights 需差异化 | 类型感知 + PCIe adaptive | 1.7-2× TTFT |
| **GNN Training** | 周期性块状 (irregular) | 难预测但可加速 fault handling | Scheduling optimization | 2.65× (alone), 1.44× (additional) |
| **Faiss IVF** | Build 平行斜线 + Query 噪声+亮线 | Two-level hierarchy | Centroid pin + tree prefetch | 21-29% build, 10-16% query |

## \sys 的统一价值

1. **Page-level 粒度**：比 expert/layer 更细
2. **不改应用**：通过 uprobe/标记获取语义
3. **可编程策略**：struct_ops 接口，按需定制
4. **跨 workload 复用**：同一套机制适用于 LLM/GNN/Vector Search

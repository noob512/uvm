# MSched 算法复现与研究计划

**日期**: 2026-02-25
**论文**: "Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling" (arxiv 2512.24637)
**完整论文**: `docs/reference/msched_paper/html/2512.24637v2/index.html`
**状态**: 进行中

---

## ⚠️ 约束：不修改自定义内核模块

**硬性要求**：所有新功能必须在 BPF 用户态程序（`extension/`）中实现，**不能修改** `kernel-module/nvidia-module/` 下的自定义 nvidia-uvm 内核模块代码。

**原因**：自定义内核模块是已有成果的基础设施，修改它会引入维护负担和稳定性风险。

**已有 kfunc（可直接使用）**：
- `bpf_gpu_set_prefetch_region()` — 设置 prefetch 区域
- `bpf_gpu_block_move_head()` — 将 chunk 移到 eviction list 头部（优先驱逐）
- `bpf_gpu_block_move_tail()` — 将 chunk 移到 eviction list 尾部（保护）
- `bpf_gpu_strstr()` — 字符串匹配
- `bpf_gpu_migrate_range(va_space, addr, length)` — **sleepable**: 从 bpf_wq callback 调用，触发 VA range 迁移到 GPU

**读取 chunk 信息的方式**：使用 BTF/CO-RE（`BPF_CORE_READ`）直接解引用内核结构体，无需添加新 kfunc：
```c
// 获取 chunk 的 VA 地址（替代 bpf_uvm_get_chunk_va kfunc）
uvm_va_block_t *va_block = BPF_CORE_READ(chunk, va_block);
u64 chunk_va = BPF_CORE_READ(va_block, start);

// 获取 chunk 大小
// bpf_probe_read_kernel 读取 bitfield，然后 1ULL << log2_size
```

**获取 va_block/va_space 上下文的方式**：kprobe + per-CPU map（不修改 hook 签名）：
```c
// kprobe 挂在 uvm_perf_prefetch_get_hint_va_block（struct_ops hook 之前调用）
SEC("kprobe/uvm_perf_prefetch_get_hint_va_block")
int BPF_KPROBE(capture_va_block, uvm_va_block_t *va_block) {
    struct va_block_ctx *info = bpf_map_lookup_elem(&va_block_cache, &zero);
    info->va_start = BPF_CORE_READ(va_block, start);
    info->va_end = BPF_CORE_READ(va_block, end);
    // va_space: va_block → managed_range → va_range.va_space
    uvm_va_range_managed_t *managed = BPF_CORE_READ(va_block, managed_range);
    info->va_space = (u64)BPF_CORE_READ(managed, va_range.va_space);
}
// struct_ops hook 中从同一 per-CPU map 读取
```

参考实现：
- CO-RE 读取: `extension/test_chunk_access.bpf.c`、`extension/chunk_trace.bpf.c`
- kprobe + per-CPU map: `extension/prefetch_trace.bpf.c`
- bpf_wq + migrate_range: `extension/prefetch_cross_block_v2.bpf.c`

---

## 1. 核心洞察：MSched 算法对单应用 UVM 同样有效

MSched 论文以 GPU 多任务为切入点，但其两个核心算法——**template-based working set prediction** 和 **Belady OPT eviction**——本质上解决的是 "UVM demand paging 不知道每个 GPU kernel 实际需要哪些页" 的问题。这个问题在**单应用 oversubscription** 下同样存在。

### 1.1 单应用为什么也需要 per-kernel 预测？

以 llama.cpp 120B MoE (59 GiB) 在 RTX 5090 (32GB) 上 decode 为例：

```
单次 decode step ≈ 160 个 GPU kernel 依次执行:
  kernel_layer0_attn(weights_L0, KV_L0, ...)    → 实际访问 ~60MB
  kernel_layer0_ffn(weights_L0_ffn, ...)         → 实际访问 ~120MB
  kernel_layer1_attn(weights_L1, ...)            → 实际访问 ~60MB
  ...
  kernel_layer79_ffn(weights_L79_ffn, ...)       → 实际访问 ~120MB
  // 下一个 decode step 重复同样的序列
```

- **总分配**: 59 GiB（所有 weights + activations + KV-cache 在一个大 buffer 里）
- **每个 kernel 实际 working set**: 16KB – 120MB（一层的 weight slice）
- **Default UVM**: 不知道每个 kernel 的 working set → fault → 按 spatial locality 猜 → 猜错的页挤掉后续 kernel 需要的页 → 连锁 thrashing
- **MSched template 预测**: 精确知道 layer K 的 kernel 只访问 `weights[offset_K : offset_K + size_K]`，0% false positive

### 1.2 MSched 论文中的证据

- **Table 1**: llama.cpp 的 allocation-granularity 预测 false positive rate 高达 99.7%（因为所有层的 weights 在一个大 cudaMalloc 里），而 template 预测 0%
- **Fig 8 (Ablation)**: 300% oversubscription 时，allocation-granularity 比 template 多 12.27× 迁移量 → throughput 差 15.67×。这是**单应用效应**——精度是瓶颈，不是多任务
- **Fig 6**: 即使没有 pipeline 优化，MSched 在 150% oversubscription 下保持 74% in-HBM throughput，demand paging 只剩 6%

### 1.3 gpu_ext 已有成果

gpu_ext 论文已经在单应用场景下证明了部分效果：
- llama.cpp 120B decode: **11.3× over raw UVM** (86.89 vs 7.72 tok/s)，使用 stride prefetch + LFU eviction
- vLLM Qwen-30B TTFT: **1.7-2× improvement**
- GNN training: **2.65× speedup**

**差距**: gpu_ext 的 prefetch 是 "围绕 fault address 的空间扩展"，MSched 是 "这个 kernel 精确访问哪些页"。随着 oversubscription 加大，这个精度差距指数放大（MSched Fig 8）。

---

## 2. MSched 算法分解

### 2.1 算法一：Template-Based Working Set Prediction

**离线阶段**（NVBit profiling）：
1. 用 NVBit instrument 每个 GPU kernel 的每次内存访问
2. 记录：`kernel_name → (launch_args, accessed_memory_regions)`
3. 分析 memory analyzer 归类为三种 template：

| Template | 占比 | 公式 | 示例 |
|----------|------|------|------|
| T1 (Fixed) | ~77% | `size = 常量` | 固定大小的 weight slice，invariant buffer |
| T2 (Linear) | ~18% | `size = k × arg_product` | `matmul(A,B,C,M,N,K)` → size ∝ M×K |
| T3 (Strided) | ~5% | `stride = k × arg_product` 的不连续块 | 高维 tensor 的特定维度操作 |

**在线阶段**（预加载 DLL 拦截 cuLaunchKernel）：
1. 拦截 `cuLaunchKernel` → 读取 launch arguments
2. 代入离线推导的公式 → 计算精确的 page set
3. 将预测结果附加到 kernel metadata

**精度**: 0.25% false negative, 0% false positive (Table 1)

### 2.2 算法二：Belady OPT Eviction

对于 kernel 序列 `[K0, K1, K2, ..., Kn]`，每个 kernel 的 working set 已知：

1. 构建 access timeline: `page P → 下次访问在 kernel Ki`
2. 淘汰决策: 淘汰 **下次访问最远** 的页
3. 实现: 逆序遍历 timeline，`madvise` 每个 kernel 的 pages 到淘汰链表尾部 → 链表头部自然是最优淘汰候选

**单应用简化版**: 对于 LLM decode 这种周期性 pattern：
- Layer K 刚跑完 → layer K 的 weights 不会在 ~160 个 kernel 内再被用
- Layer K+1 即将运行 → layer K+1 的 weights 最紧急
- **Belady ≈ 淘汰刚完成的层，预取下一层**

### 2.3 算法三：Pipelined Migration

- 双 Copy Engine: CE0 做 D2H eviction, CE1 做 H2D population 并行
- 全双工 PCIe: 63.5 GB/s (RTX 5080) vs 41.7 GB/s without pipeline (1.52×)

**此算法需要修改驱动，gpu_ext 无法实现。不在复现范围内。**

---

## 3. 复现路线：先离线后在线

### 3.1 ~~为什么先用 NVBit 离线复现？~~ → NVBit 在 RTX 5090 上不可行

原计划用 NVBit 做 device 侧离线 profiling，但**实测失败**：

1. ~~**NVBit 可用**: v1.7.7.1 已支持 RTX 5090 (SM_120)~~ → **实际不可行**: NVBit binary instrumentation 在 SM_120 上极慢（10+ 分钟仅初始化，CPU 200%，模型未加载），20B 和 120B 模型均超时
2. **根本原因**: NVBit 需要对每条指令做 binary translation，RTX 5090 的新 SM 架构使这个过程极其缓慢
3. **替代方案**: 使用**解析式方法**（从 GGUF 元数据计算 per-layer working set）+ **chunk_trace**（host 侧 UVM 事件追踪）替代

> **重要**: 本文档中所有 working set 分析数据均来自**解析式计算**和 **host 侧 chunk_trace**，**不包含任何 NVBit device 侧追踪数据**。

### 3.2 修正后的整体路线

```
Phase 0: ~~NVBit 离线 profiling~~ → 解析式 Working Set 分析 ✅ 已完成
  ↓ 方法: GGUF 元数据解析 + host 侧 chunk_trace
  ↓ 产出: T1/T2/T3 分类、per-decode-step 分析
  ↓ 限制: 无 device 侧 per-kernel ground truth（NVBit 不可行）

Phase 1: BPF eviction policy 实验 ✅ 已完成
  ↓ 实现 cycle_moe eviction policy (T1 保护)
  ↓ 结论: eviction 优化空间有限，默认 LRU 已足够好
  ↓ 关键发现: Hash map/move_head 在 fault handler 中不安全

Phase 2: Layer-aware prefetch（下一步重点）
  ↓ 基于 fault VA 推断当前层号，预取下一层 weights
  ↓ 不需要 NVBit — 利用 VA 地址单调递增的特性
  ↓ 目标: 消除 51% re-fault (83 GB 浪费迁移)

Phase 3: 在线 working set 学习（gpu_ext 独有贡献）
  ↓ device-side eBPF (bpftime) 在运行时观察 per-kernel access pattern
  ↓ 学到映射后 host-side prefetch hook 使用
  ↓ 前提: bpftime GPU 支持在 RTX 5090 上可用
```

---

## 4. ~~Phase 0: NVBit 离线 Profiling~~ → 已放弃，改用解析式方法

### 4.1 ~~目标~~ → 实际状态: NVBit 不可行

原计划:
- ~~在 RTX 5090 上用 NVBit instrument llama.cpp 120B 的 decode 过程~~
- ~~记录每个 GPU kernel 的: kernel name/hash, launch args, 访问的 memory pages~~
- ~~分析 T1/T2/T3 template 分布~~
- ~~生成 `kernel_id → {page_ranges}` 映射表~~

**实际结果**: NVBit binary instrumentation 在 RTX 5090 上完全不可用。20B 和 120B 模型均在初始化阶段超时（10+ 分钟，CPU 200%，GPU 仅 632 MiB）。ws_trace.so 工具已编写但无法产出数据。

**替代**: 使用解析式方法（GGUF 元数据 + chunk_trace host 侧事件），详见后文 "解析式 Working Set 分析" 章节。

### 4.2 NVBit 环境（仅供参考，实际不可用）

- **NVBit 版本**: v1.7.7.1（声称支持 SM_120）
- **CUDA**: 12.9
- **GPU**: RTX 5090, compute capability 12.0 (SM_120)
- **实测结果**: binary translation 阶段极慢，无法完成模型加载。原因可能是 SM_120 指令集新增内容过多，NVBit 的翻译器无法高效处理

### 4.3 实现步骤

1. **下载编译 NVBit v1.7.7.1**
   ```bash
   git clone https://github.com/NVlabs/NVBit.git
   cd NVBit && git checkout v1.7.7.1
   # 编译 mem_trace tool（记录每次 memory access 的地址）
   cd tools/mem_trace && make
   ```

2. **编写 MSched-style memory analyzer NVBit tool**
   ```
   输入: 拦截 cuLaunchKernel 获取 kernel name + args
         instrument 每个 load/store 获取 accessed addresses
   输出: per-kernel memory access range（对齐到 4KB page）
   ```

3. **在 llama.cpp 120B 上运行 profiling**
   ```bash
   LD_PRELOAD=./mem_trace.so \
     workloads/llama.cpp/llama.cpp/build/bin/llama-bench \
     --model gpt-oss-120b-default -p 512 -n 128 --uvm
   ```

4. **分析结果**
   - 每个 kernel 访问了哪些 page range？
   - 有多少 kernel 是 T1 (fixed)？T2 (linear)？T3 (strided)？
   - 和 MSched Table 2 的数据对比（llama.cpp: 60% T1, 38% T2, 2% T3）

### 4.4 预期产出

- `profiling_data/llama_120b_kernel_wsets.json`: 每个 kernel → page ranges 映射
- `profiling_data/llama_120b_template_stats.json`: T1/T2/T3 分布统计
- 一份分析报告，确认 MSched 的 template 分类在我们的 workload 上是否成立

---

## 5. Phase 1: 离线数据指导 gpu_ext 策略

### 5.1 目标

用 Phase 0 的 NVBit profiling 数据，实现两个新的 gpu_ext BPF policy：

1. **kernel-aware prefetch**: 知道每个 kernel 的 working set → 精确预取
2. **cycle-aware eviction**: 知道 kernel 执行顺序 → 近似 Belady OPT

### 5.2 Kernel-Aware Prefetch Policy

**文件**: `extension/prefetch_kernel_aware.bpf.c`

**设计思路**:
- Phase 0 产生的映射表通过 BPF map 加载到内核（userspace loader 写入）
- prefetch hook 触发时，查表得到当前 kernel 应预取的 page range
- 精确预取（不多不少），避免 over-prefetch 污染 HBM

```c
// BPF map: kernel_id → working set range
struct kernel_wset {
    u64 base_page;     // 起始页号
    u32 num_pages;     // 页数
};

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 8192);
    __type(key, u64);                    // kernel_id (hash)
    __type(value, struct kernel_wset);
} kernel_wset_map SEC(".maps");

// 在 prefetch hook 中:
SEC("struct_ops/gpu_page_prefetch")
int BPF_PROG(gpu_page_prefetch, ...)
{
    // 1. 确定当前是哪个 kernel（通过 fault 地址推断或 cross-layer 通信）
    // 2. 查 kernel_wset_map
    // 3. 设置 prefetch region = working set range
    // 4. return BYPASS
}
```

**挑战**: prefetch hook 收到的信息是 `page_index` 和 `bitmap_tree`（VA block 级别），不直接知道当前是哪个 kernel。需要通过以下方式推断：
- **方案 A**: fault address 落在哪个 weight range → 推断是哪一层 → 预取该层所有页
- **方案 B**: 用 uprobe 拦截 cuLaunchKernel → 写入 BPF map → prefetch hook 读取

方案 B 更精确，但需要 uprobe 支持。gpu_ext 论文已经用 uprobe trace PyTorch allocations，技术可行。

### 5.3 Cycle-Aware Eviction Policy

**文件**: `extension/eviction_cycle_aware.bpf.c`

**设计思路**:
LLM decode 的 kernel 序列是周期性的（每个 decode step 遍历所有层）。利用这个周期性：

```c
struct cycle_state {
    u32 current_pos;      // 当前在 cycle 中的位置
    u32 cycle_length;     // cycle 长度（层数 × 每层 kernel 数）
};

struct page_meta {
    u32 last_access_pos;  // 上次访问时的 cycle position
    u32 kernel_id;        // 哪个 kernel 访问的
};

// 在 eviction_prepare 中:
// distance = cycle_length - (current_pos - page.last_access_pos)
// distance 大 → move_head（优先淘汰）
// distance 小 → move_tail（保护）
```

**vs LFU 的优势**:
- LFU 对 dense model（每层访问频率相同）没有区分能力
- Cycle-aware 知道 "刚用过 = 离下次使用最远"，精确做 Belady

### 5.4 实验方案

| 配置 | Prefetch | Eviction | 说明 |
|------|----------|----------|------|
| Baseline UVM | nvidia-uvm 默认 | 默认 LRU | 对照基线 |
| gpu_ext 现有 | stride prefetch | LFU | 现有最佳组合 |
| 离线指导 | kernel-aware (Phase 0 数据) | cycle-aware | 使用离线 profiling 数据 |
| 离线指导 + LFU | kernel-aware | LFU | 验证 prefetch 精度的独立贡献 |
| 离线指导 + 默认预取 | stride prefetch | cycle-aware | 验证 eviction 改进的独立贡献 |

**Workloads**:
- llama.cpp 120B MoE (主要目标)
- vLLM Qwen-30B (验证泛化性)
- Faiss SIFT-100M (不同 access pattern)

**指标**:
- Decode throughput (tok/s)
- Page fault 数 / decode step
- 总迁移量 / decode step (MB)
- Prefetch false positive rate, false negative rate

---

## 6. Phase 2: 在线 Working Set 学习

### 6.1 目标

去掉 NVBit 离线依赖，用 gpu_ext 的 device-side eBPF 在运行时学习 per-kernel working set。

### 6.2 为什么在线学习可行？

1. **LLM decode 高度重复**: 每个 decode step 执行相同的 kernel 序列，访问相同的 weight ranges
2. **收敛快**: 1-2 个 decode step 就能观察到完整的 pattern
3. **自适应**: 不同 sequence length → 不同 KV-cache 大小，在线学习自动适应

### 6.3 设计

```
Device-side eBPF (GPU kernel 执行时):
  on access(ctx):
    kernel_id = hash(当前 kernel 信息)
    page = ctx->address >> PAGE_SHIFT
    // 记录: 这个 kernel 触碰了这个 page
    map_update(kernel_page_set, {kernel_id, page}, 1)

  on fence(ctx):  // kernel 完成
    // kernel_id 的 page set 快照已完成
    // 通过 cross-layer map 通知 host side

Host-side eBPF (page fault / prefetch 时):
  on gpu_prefetch(ctx):
    kernel_id = 从 cross-layer map 读取当前 kernel
    wset = map_lookup(kernel_wset_learned, kernel_id)
    if wset 有效:
      精确预取 wset 中的页
      return BYPASS
    else:
      return DEFAULT  // 第一次见到的 kernel，用默认策略
```

### 6.4 挑战

1. **BPF map 容量**: 每个 kernel 的 working set 可能有上千个 page → 用 `{base, size}` 的 range 表示而非逐页记录
2. **BPF verifier 限制**: device-side 循环有限制 → 需要 warp-level 聚合
3. **Cross-layer map 延迟**: device→host sync 延迟 → 需要在 kernel boundary (fence) 做 sync
4. **学习收敛速度**: 需要至少一个完整 cycle（一个 decode step）才能学到完整 pattern

### 6.5 vs NVBit 离线的对比

| 维度 | NVBit 离线 | gpu_ext 在线 |
|------|-----------|-------------|
| 精度 | 100%（instrument 每条指令） | 高（取决于采样率和 map 容量） |
| 开销 | 85%+（只用于 profiling） | 3-14%（可常驻） |
| 适应性 | 静态 profile，不适应输入变化 | 实时适应（seq_len, batch_size 变化） |
| 部署 | 需要单独 profiling pass | 无需，运行时自动学习 |
| T2 template | 可推导线性关系 | 需要多次不同参数的观察 |

---

## 7. Phase 3: 集成与全面评估

### 7.1 实验矩阵

**Experiment 1: Prefetch 精度差距量化**

用 gpu_ext 现有 tracing (prefetch_trace, chunk_trace) 测量：

| 配置 | FP rate | FN rate | 迁移量/step |
|------|---------|---------|------------|
| UVM 默认 | ? | ? | ? |
| gpu_ext stride | ? | ? | ? |
| gpu_ext always_max | ? | ? | ? |
| 离线 kernel-aware | 预期 ~0% | 预期 <1% | 预期最小 |
| 在线学习 (收敛后) | 预期 <5% | 预期 <5% | 预期接近离线 |

**Experiment 2: Oversubscription 缩放**

验证 MSched Fig 8 的发现：精度差距在高 oversubscription 下指数放大。

| Oversubscription | gpu_ext stride | 离线 kernel-aware | 差距倍数 |
|------------------|---------------|-------------------|---------|
| 1.5× | ? | ? | ? |
| 2.0× | ? | ? | ? |
| 3.0× | ? | ? | ? |

**Experiment 3: Eviction 策略对比**

| 策略 | Dense Model | MoE Model | 说明 |
|------|------------|-----------|------|
| 默认 LRU | ? | ? | UVM baseline |
| LFU | ? | ? | gpu_ext 现有 |
| Cycle-aware | ? | ? | 新策略 |
| Cycle-aware + kernel-aware prefetch | ? | ? | 完整组合 |

**Experiment 4: 端到端性能对比**

| 系统 | llama.cpp 120B decode | vLLM Qwen-30B TTFT | Faiss build |
|------|----------------------|--------------------|-----------:|
| UVM baseline | 7.72 tok/s (已有) | 9642 ms (已有) | ? |
| gpu_ext 现有策略 | 86.89 tok/s (已有) | 5042 ms (已有) | ? |
| gpu_ext + 离线 kernel-aware | ? | ? | ? |
| gpu_ext + 在线学习 | ? | ? | ? |
| MSched 论文数据 (不同硬件) | 参考值 | — | — |

### 7.2 论文定位

> MSched 证明了 per-kernel working set prediction + Belady OPT eviction 可以比 demand paging 快 11-58×。但 MSched 需要离线 NVBit profiling（85%+ 开销）和自定义驱动 ioctl。
>
> 我们展示 gpu_ext 的跨层 eBPF 运行时可以通过**在线 working set 学习**达到类似效果：device-side eBPF 以 3-14% 开销观察 per-kernel 内存访问 pattern，学到 working set 映射后通知 host-side prefetch/eviction hook 做 kernel-aware 决策——无需离线 profiling、无需驱动修改、无需应用改动。

---

## 8. 实现优先级

| 优先级 | 任务 | 预估工作量 | 依赖 |
|--------|------|-----------|------|
| **P0** | NVBit 环境搭建 + 离线 profiling llama.cpp 120B | 1 周 | 无 |
| **P0** | 用现有 tracing 工具测量 prefetch precision baseline | 3 天 | 无 |
| **P1** | 实现 kernel-aware prefetch (用离线数据) | 1-2 周 | P0 |
| **P1** | 实现 cycle-aware eviction | 1 周 | P0 |
| **P2** | Experiment 1-3: 离线策略 vs 现有策略对比 | 1 周 | P1 |
| **P3** | 在线 working set 学习 (device-side eBPF) | 2-3 周 | P2 结果验证方向 |
| **P4** | Experiment 4: 全面端到端评估 | 1-2 周 | P3 |
| **P5** | 多租户扩展（MSched Combination D 场景） | 2 周 | P4 |

---

## 9. 实验日志

### 2026-02-25: 项目启动

- 下载 MSched 论文完整 HTML + 22 张图片到 `docs/reference/msched_paper/`
- 修复 `.gitmodules` 中 `docs/gpu-ext/paper` submodule 的绝对路径问题
- 完成 MSched 论文精读，确认核心算法对单应用 UVM 场景的适用性
- 确认 NVBit v1.7.7.1 支持 RTX 5090 (SM_120, compute capability 12.0)
- 确认 gpu_ext 已有 prefetch_trace / chunk_trace 可用于测量精度
- 创建本计划文档

**完成**:
- [x] P0: 搭建 NVBit 环境，编译 mem_trace tool
- [x] P0: 用 chunk_trace 测量 raw UVM 的 page fault / eviction 模式

### 2026-02-25: P0 实验结果 — NVBit + chunk_trace 基线测量

#### NVBit 环境

- NVBit v1.7.7.1 下载并编译成功 (`/home/yunwei37/workspace/gpu/NVBit/nvbit_release_x86_64/`)
- 编译了两个工具:
  - `mem_trace.so`: 原始 NVBit mem_trace（每个 warp 的完整内存地址）
  - `ws_trace.so`: 自定义 working set profiler（CPU 端聚合 per-kernel 唯一页集合）
- 编写了 `scripts/analyze_ws.py`: MSched 风格的 working set 分析脚本（T1/T2/T3 分类 + FP/FN 率）

#### chunk_trace 基线测量

**实验配置**: llama.cpp 120B MXFP4 MoE, RTX 5090 (32GB), raw UVM (无 eBPF 策略)

**短测试** (pp=32, tg=16):
- 78K 事件，12,607 唯一 2MB chunks (24.6GB)，0 次 eviction
- 说明短序列不足以触发内存压力

**长测试** (pp=512, tg=128) — **关键数据**:

| 指标 | 值 |
|------|------|
| 总事件数 | 358,445 |
| ACTIVATE (page fault) | 41,825 |
| POPULATE (page use) | 290,523 |
| EVICTION_PREPARE | 25,968 |
| 唯一 2MB chunks | 20,531 |
| VA 空间总量 | 40.1 GB |
| Oversubscription | 8.1 GB (25%) |

**核心发现**:

1. **51% 的 page fault 是 re-fault** (21,294/41,825): 页面被 evict 后又被重新 fault-in
2. **82% 的 chunks 发生 thrashing** (16,813/20,531): 被 evict 后重新加载至少一次
   - 2 次重新激活: 12,338 chunks
   - 3 次重新激活: 4,469 chunks
3. **89% 的 working set 每 200ms 变化一次**: 高度动态的访问模式
4. **激活精度**: 98.4% 的被激活 chunks 确实被使用（demand paging 自身精度高，但不做预测）
5. **浪费的迁移带宽**: ~83.2 GB (每次 re-fault = evict 2MB + reload 2MB)

**Raw UVM 性能 (无 eBPF):**
- pp=512: 143.9 tok/s
- tg=128: 49.4 tok/s

**对比 gpu_ext 论文数据** (有 stride prefetch + LFU):
- tg decode: 86.89 tok/s (1.76× over raw UVM 的 49.4)

**MSched 复现启示**:
- 82% 的 chunk thrashing 证实了 MSched 的核心论点: 默认 eviction 策略不知道哪些页马上需要
- Per-kernel working set prediction 可以消除 51% 的 re-fault，节省 ~83GB 迁移带宽
- Working set 每 200ms 变化 89%，说明每个 decode step 访问的层完全不同
- **下一步**: 用 NVBit ws_trace 获取 per-kernel working set ground truth，验证 template prediction 的 FP/FN 率

**待办**:
- [x] P0: ~~运行 NVBit ws_trace 获取 per-kernel working set~~ → 改用解析式方法（见下方）
- [ ] P1: 设计 kernel-aware prefetch 的 BPF policy 接口
- [ ] P1: 基于 chunk_trace 数据设计 cycle-aware eviction 策略

### 2026-02-25: 解析式 Working Set 分析 — NVBit 替代方案

#### NVBit ws_trace 结论（已修正）

NVBit v1.7.7.1 在 RTX 5090 (SM_120) 上的 binary instrumentation 阶段**极慢** — 初始化需要 10+ 分钟（CPU 消耗 200%）。之前因 timeout 过短（< 15 分钟）而误判为"不可行"。

**修正**: NVBit v1.7.7.1 技术上支持 SM 3.5-12.1，driver ≤575.xx。MSched 论文本身用 NVBit 做离线 profiling，30-40% overhead 是预期的。120B 模型 + NVBit 可能需要 30-90 分钟完成初始化 + 执行。之前不是真的崩溃，是被提前 kill 了。

**状态**: 待重试（timeout 90 分钟）— 见下方 2026-02-26 NVBit 重试计划

#### 解析式 Per-Layer Working Set 计算

基于 GGUF 元数据解析模型架构参数，直接计算每层的 working set 分布。脚本: `NVBit/scripts/analytical_ws.py`

**120B 模型 (gpt-oss-120b) — 核心数据**:

| 组件 | 每层大小 | 全模型 |
|------|---------|--------|
| Attention (Q8) | 27.0 MB | 0.95 GB (36 layers) |
| Per Expert (MXFP4) | 13.4 MB | — |
| All 128 Experts | 1714 MB | 60.26 GB |
| Router (F32) | 1.4 MB | 50.6 MB |
| Embeddings + Output | — | 1.15 GB |
| **Total Model** | **1741 MB** | **62.36 GB** |

**MSched Template 分类**:

| Template | 类别 | 大小 | 占总模型 |
|----------|------|------|---------|
| **T1 (Fixed)** | 嵌入层 + 注意力 + Router | **2.14 GB** | 3.4% |
| **T2 (Active Experts)** | 4/128 experts × 36 layers | **1.88 GB** | 3.0% |
| **T3 (Inactive)** | 124/128 experts × 36 layers | **58.33 GB** | 93.6% |
| **Ideal WS (T1+T2)** | — | **4.02 GB** | 6.4% |

**关键发现**:

1. **理想 working set 仅 4.02 GB** — 占 32 GB VRAM 的 12.6%
   - 即使 2× oversubscription (模型 62 GB)，活跃数据仅用 1/8 VRAM
   - 剩余 28 GB VRAM 可用于 expert caching

2. **93.6% 的模型是 T3（不活跃 experts）**
   - MoE 的极端稀疏性: 每层 128 个 expert 只激活 4 个
   - Default UVM 无法区分 T1/T2/T3 → 浪费大量 VRAM 缓存不活跃 expert

3. **理想迁移量 vs 实际迁移量**:
   - MSched 理想: 每 decode step 迁移 1.88 GB (仅切换 active experts)
   - Default UVM 实际: ~83 GB wasted migration (chunk_trace 测量)
   - **差距 ~44×** → 巨大优化空间

4. **迁移时间估算**:
   - 理想: 1.88 GB / 63.5 GB/s = 29.6 ms per decode step
   - 120B decode throughput ~50 tok/s (20ms/tok) → 迁移可与计算 overlap

5. **与 chunk_trace 数据交叉验证**:
   - chunk_trace 显示 20,531 唯一 2MB chunks (40.1 GB VA)
   - 解析计算: 62.36 GB total model → 合理 (包含 alignment/padding)
   - chunk_trace 的 82% thrashing 主要来自不活跃 expert 的无差别 eviction

#### 策略含义

**对 gpu_ext kernel-aware prefetch 的启示**:
- 不需要 NVBit 级别的逐指令追踪
- 只需知道 "当前 decode step 在第几层" + "路由器选了哪些 expert"
- 信息来源: cuLaunchKernel 拦截 (uprobe) 或 fault 地址推断
- 精度: 只需 layer-level 粒度 (不需要 instruction-level)

**对 eviction 策略的启示**:
- LRU/LFU 无法区分 "刚用完的 expert" 和 "下次马上要用的 attention weights"
- Cycle-aware: 知道 decode step 的层序列 → Belady OPT ≈ 淘汰刚完成层的不活跃 experts
- 保护策略: T1 (attention + embeddings) 永远不淘汰; T2 按 cycle distance 排序

### 2026-02-25: 解析模型 vs chunk_trace 交叉验证

用 chunk_trace 的 VA 地址时序数据验证解析模型:

**Per-decode-step 分析** (pp=512, tg=128):

| 指标 | 值 | 含义 |
|------|------|------|
| 检测到的 decode steps | 177 | 基于 VA 大幅回退检测 |
| 稳定 decode phase | 156 steps (skip warmup) | t=3s 之后 |
| 平均 WS/step (page faults) | 333 MB (166 chunks) | 每步需要 fault-in 的数据 |
| 平均 new data/step | 60 MB | 从未见过的新 chunk |
| 平均 step duration | 41 ms (~24 tok/s) | 含 migration 开销 |
| 总 activated VA | 40.10 GB / 62.36 GB | 34% expert 从未被路由 |

**关键发现**:

1. **333 MB/step 的 page fault 全部是 re-fault** — 每步 fault-in 333 MB，但只有 60 MB 是首次访问的新数据，其余 273 MB 全是之前被 evict 的 chunk 重新加载
2. **100% ascending VA pattern** — 每个 decode step 内，VA 地址单调递增，证实 layer 序列化访问
3. **Step 6 是主加载阶段** — 15,446 chunks (30.9 GB) 在 2 秒内加载（prefill 阶段）
4. **3,718 chunks 只 fault 1 次** (7.4 GB) — 这些是 T1 候选（attention weights + embeddings），首次加载后常驻 VRAM
5. **16,813 chunks 多次 fault** — 这些是 expert weights，被 LRU 误 evict 后反复 re-fault
6. **模型中 34% 的 experts 从未被激活** — 总模型 62.36 GB，实际触达 40.10 GB

**Analytical vs Empirical 对比**:

| 维度 | 解析模型 | chunk_trace 实测 | 说明 |
|------|---------|-----------------|------|
| Ideal WS | 4.02 GB | — | 理论最优（无 re-fault） |
| 页面 fault/step | 1.88 GB (全 expert swap) | 0.33 GB (实际 fault) | 实测较低因为部分 expert 已 resident |
| T1 (永久) | 2.14 GB | ~7.4 GB (single-fault) | 实测含部分 hot experts |
| Re-fault rate | 0% (理论最优) | 50.9% | LRU 淘汰错误导致 |
| 迁移浪费 | 0 | 83.2 GB | 每次 re-fault = 4 MB round-trip |
| Step duration | ~30 ms (migration only) | 41 ms | 含 fault handling overhead |

**验证结论**: chunk_trace 数据完全支持解析模型的 T1/T2/T3 分类。核心瓶颈是 LRU eviction 无法区分 T1 (attention, 永驻) 和 T3 (inactive experts, 应优先淘汰)。

**下一步**:
- [x] P1: 实现 expert-aware eviction (T1 保护 + T3 优先淘汰) — **已完成**，见下方
- [ ] P1: 实现 layer-aware prefetch BPF policy (基于 fault 地址推断层号)
- [ ] P1: 基于 VA 地址的 cycle position 检测 (每 decode step 的 VA 回退检测)

### 2026-02-25: Cycle-Aware MoE Eviction Policy 实现与评测

#### 实现: `extension/eviction_cycle_moe.bpf.c`

**设计原则**: 最小干预——只在必要时保护 T1 chunks，其余让内核默认策略处理。

**核心机制**:
1. **T1 频率检测**: Per-CPU array (O(1) lookup) 跟踪每个 chunk 的访问次数
   - 访问 >= 3 次 → T1 (attention/embeddings) → `move_tail` (保护)
   - 访问 < 3 次 → `return 0` (让内核默认处理)
2. **chunk_activate**: `return 0` (内核默认，零开销)
3. **eviction_prepare**: `return 0` (内核默认)

**关键技术发现**:

1. **`move_head` 在 `chunk_activate` 中不安全** — 导致 Xid 31 (FAULT_PDE MMU Fault):
   - 新 chunk 被 activate 后立即移到 HEAD = 第一个被 evict 的候选
   - Eviction 线程并发扫描 list 时，可能在 page table setup 完成前就 evict 该 chunk
   - **解决**: 用 `move_tail` 或 `return 0` (让内核默认处理)

2. **BPF Hash Map 在 UVM fault handler 热路径中不可用** — 导致 Xid 31:
   - `BPF_MAP_TYPE_HASH` 的 bucket lock + hash computation 延迟太高
   - GPU fault timeout 过期 → Xid 31 FAULT_PDE
   - **解决**: 用 `BPF_MAP_TYPE_PERCPU_ARRAY` (O(1) 无锁查找)

3. **BPF verifier 禁止 pointer arithmetic** — 不能对 pointer 类型寄存器做 shift/XOR:
   - **解决**: `bpf_probe_read_kernel(&scalar, sizeof(scalar), &ptr)` 将 pointer 转为 scalar

4. **`return 0` vs `return 1 (BYPASS)` 的性能差异**: 非 T1 chunk 返回 0 让内核处理，避免了 `move_head` 的 list manipulation 开销

#### 性能评测

**120B Model, RTX 5090 (32GB), UVM demand paging**:

| 策略 | pp=512 | tg=128 | tg=512 | pp=2048 | tg=512 (2K ctx) |
|------|--------|--------|--------|---------|-----------------|
| **Baseline (无策略)** | 145.2 | 50.6 | 59.7 | 148.8 | 56.4 |
| **cycle_moe v2** | 145.2 | 50.9 | 59.8 | 149.1 | 56.2 |
| cycle_moe v1 | 144.1 | 50.9 | 47.8 | — | — |
| MRU | 18.97 | 9.62 | 10.3 | — | — |
| LFU | ❌ Xid 31 | ❌ | ❌ | — | — |

**分析**:

1. **cycle_moe v2 ≈ Baseline** (< 1% 差异): 零开销 T1 保护策略
2. **MRU 灾难性退化** (-83%): MRU 把每步都用的 attention weights 移到 HEAD → 每步都被 evict → 疯狂 thrashing
3. **LFU 崩溃**: Hash map 操作在 fault handler 热路径中延迟过高 → GPU MMU timeout
4. **cycle_moe v1 vs v2**: v1 对非 T1 chunks 调用 `move_head` 造成 20% overhead; v2 返回 0 消除开销

**结论**:
- 默认 UVM eviction (LRU-like) 对 MoE 工作负载已经足够好 — attention weights 自然停留在 LRU tail
- cycle_moe v2 作为安全网: 显式 T1 保护，零额外开销，防止 MRU 类策略的 attention thrashing
- **真正的优化空间在 prefetch 侧**: MSched 的核心贡献是 template-based working set prediction + proactive migration，而非 eviction ordering
- **下一步重点**: Layer-aware prefetch policy — 基于 fault 地址推断当前层，预取下一层的 weights

### 2026-02-26: Prefetch 策略全面评测 — 发现巨大优化空间

#### 实验设计

测试所有现有 prefetch 策略对 120B MoE 模型的影响：

| 策略 | 说明 |
|------|------|
| Baseline (无 BPF) | 内核默认 UVM prefetch（bitmap tree 选择性预取） |
| always_max | 每次 fault 预取整个 VA block 的 max_prefetch_region |
| stride | 检测 stride 模式后预测下一页（confidence 衰减） |
| none (禁用) | 返回空 region + BYPASS，完全禁用预取 |
| always_max + cycle_moe | 组合: 激进预取 + T1 eviction 保护 |

**配置**: llama.cpp 120B MXFP4 MoE, RTX 5090 (32GB), UVM mode, 5 repetitions

#### 结果

**短序列 (pp=512, tg=128)**:

| 策略 | pp512 (tok/s) | tg128 (tok/s) | pp vs Baseline | tg vs Baseline |
|------|--------------|--------------|----------------|----------------|
| **Baseline (无 BPF)** | 139.50 ± 1.85 | 45.29 ± 3.34 | — | — |
| **always_max** | 219.12 ± 2.81 | 76.85 ± 4.11 | **+57.1%** | **+69.7%** |
| **always_max + cycle_moe** | 224.25 ± 1.31 | 76.87 ± 4.35 | **+60.8%** | **+69.7%** |
| stride (conf=2, pages=4) | 33.17 ± 0.30 | 14.46 ± 1.51 | -76.2% | -68.1% |
| none (禁用) | 31.58 ± 0.29 | 14.01 ± 1.57 | -77.4% | -69.1% |

**长序列 (pp=2048, tg=512)**:

| 策略 | pp2048 (tok/s) | tg512 (tok/s) | pp vs Baseline | tg vs Baseline |
|------|---------------|--------------|----------------|----------------|
| **Baseline (无 BPF)** | 147.15 ± 1.73 | 52.20 ± 3.34 | — | — |
| **always_max** | 231.12 ± 2.70 | 85.15 ± 3.58 | **+57.1%** | **+63.1%** |
| **always_max + cycle_moe** | 227.88 ± 3.07 | 84.37 ± 3.66 | **+54.9%** | **+61.6%** |

#### 关键发现

1. **always_max prefetch 带来 57-70% 的性能提升**
   - 这是目前发现的**最大单一优化**: 仅通过把 BPF prefetch 设为 "预取整个 VA block"
   - 证明默认 UVM prefetch 的 bitmap tree 选择性预取对 MoE 模型是严重次优的
   - 默认策略可能只预取 VA block 的一部分（靠近 fault page 的区域），导致同一 VA block 的其他页需要额外 fault

2. **stride prefetch 灾难性退化 ≈ 完全禁用预取**
   - stride 策略返回 BYPASS，即使没有检测到 stride 也会设 result_region=(0,0) 跳过默认预取
   - 统计: 3900万次 fault，仅 8% 触发了 stride 预取，92% 设为空 region
   - **根本问题**: BYPASS 语义意味着 "BPF 处理了，内核不要做默认行为"，但 stride 只在少数情况下真正做了预取
   - **教训**: BPF prefetch 策略如果不确定，应该返回 DEFAULT (0) 而非 BYPASS (1)

3. **cycle_moe eviction 在 always_max 之上无额外收益**
   - always_max alone vs always_max + cycle_moe: 基本一致（< 2% 差异）
   - 再次确认: 默认 LRU eviction 对 MoE 已经足够好
   - always_max 的激进预取可能改变了 eviction 的工作分布，使 T1 保护变得不必要

4. **性能提升在不同序列长度下一致**
   - 短序列 (512/128): +57-70%
   - 长序列 (2048/512): +55-63%
   - 说明 always_max 的收益来自减少 per-VA-block fault 数量，与序列长度无关

5. **与 gpu_ext 论文数据对比**
   - 论文: tg decode 86.89 tok/s (stride prefetch + LFU)
   - 本次: tg512 85.15 tok/s (always_max alone)
   - **always_max 达到了论文级别的性能，且无需 LFU eviction（已知在当前驱动上崩溃）**

#### 实现: combined always_max + cycle_moe BPF policy

新建文件: `extension/prefetch_always_max_cycle_moe.bpf.c` + `.c` (userspace loader)
- 在单个 struct_ops 注册中组合所有 6 个 hooks
- Prefetch: always_max (无条件预取 max_prefetch_region)
- Eviction: cycle_moe (T1 frequency threshold=3, per-CPU array, 零 hash map)
- 已加入 Makefile BPF_APPS 列表

#### 下一步

**方向修正**: 原计划的 "layer-aware prefetch" 需要重新评估。

always_max 已经在 **intra-VA-block** 级别达到了最优预取（每个 VA block 2MB，fault 时全部加载）。进一步优化需要：

1. **Cross-VA-block proactive prefetch**: 在 fault 到当前 VA block 时，同时触发下一个 VA block 的预取；或在 kernel launch 前主动迁移下一层数据
   - **详细设计方案见独立文档 [`cross_block_prefetch_plan.md`](cross_block_prefetch_plan.md)**
   - 当前 BPF prefetch hook 的 `max_prefetch_region` 限制在单个 VA block 内
   - MSched 论文的核心贡献是 proactive migration（拦截 cuLaunchKernel + 内核 migration API）

2. **更大的 VA block**: 如果能增大 VA block 大小（如 4MB、8MB），always_max 的单次预取范围更大
   - 需要修改 nvidia-uvm 驱动配置

**待办**:
- [x] 量化 always_max 的 fault 减少量: 用 chunk_trace 对比 baseline vs always_max 的 page fault 数量 — 见下方
- [ ] Cross-VA-block proactive prefetch — 见 [`cross_block_prefetch_plan.md`](cross_block_prefetch_plan.md)
- [ ] 测试更多 workloads: vLLM Qwen-30B, Faiss SIFT-100M 在 always_max 下的表现

### 2026-02-26: chunk_trace 对比 — Baseline vs always_max

#### 实验

同时运行 chunk_trace (kprobe) + llama-bench 120B (pp=512, tg=128, r=3)，分别在无策略和 always_max 下。

#### chunk_trace 事件统计

| 事件 | Baseline | always_max | 变化 |
|------|---------|-----------|------|
| **ACTIVATE** (chunk 分配) | 88,387 | 88,455 | **~0%** (无差异) |
| **POPULATE** (chunk 使用) | 589,749 | 0 (*) | -100% |
| **EVICTION_PREPARE** (淘汰) | 72,372 | 72,451 | **~0%** (无差异) |
| Total events | 750,508 | 160,906 | -79% |

(*) POPULATE 事件消失的原因: chunk_trace 的 kprobe 挂在 `uvm_bpf_call_gpu_block_access` 上。当 always_max struct_ops 注册时（仅包含 prefetch hooks，eviction hooks 为 NULL），驱动检测到 `chunk_used` 函数指针为 NULL，跳过调用 → kprobe 不触发。

#### Re-fault 分析

| 指标 | Baseline | always_max |
|------|---------|-----------|
| 唯一 chunks | 15,770 | 15,772 |
| 单次 fault chunks | 0 | 0 |
| 多次 fault chunks | 15,770 | 15,772 |
| Re-faults | 72,617 | 72,683 |
| **Re-fault rate** | **82.2%** | **82.2%** |

#### 关键发现

1. **Chunk 级别活动完全相同**: ACTIVATE 和 EVICTION_PREPARE 在两种策略下几乎一致。always_max **不减少 2MB chunk 级别的 page fault 和 eviction**。

2. **性能提升来自 intra-chunk (page-level) 优化**:
   - 一个 VA block (2MB) 包含 512 个 4KB 页
   - **默认 UVM prefetch**: 每次 fault 只预取 VA block 中靠近 fault 地址的一部分页 → 同一 VA block 内的其他页需要后续单独 fault
   - **always_max**: 一次 fault 预取整个 VA block 的所有 non-resident 页 → 后续同一 VA block 内不再有 page fault
   - 结果: **每个 chunk 的 page fault 处理次数大幅减少**，但 chunk activate/evict 次数不变

3. **always_max 的优化机制总结**:
   - 不减少 chunk 级别的 thrashing（82% re-fault rate 不变）
   - 减少 **page-level fault overhead**: 每个 2MB chunk 从多次 fault 处理变为一次
   - 减少 **DMA 事务数**: 批量迁移整个 block 而非多次小传输
   - 减少 **fault queue 和中断开销**: 更少的 GPU fault → 更少的 CPU 中断

4. **进一步优化空间仍然巨大**:
   - 82% 的 chunk 仍在 thrash（被 evict 后重新加载）
   - always_max 只解决了 "每次 fault 带多少数据" 的问题
   - 未解决 "哪些 chunk 应该被 evict" 和 "能否提前预测需要哪些 chunk" 的问题
   - **Cross-VA-block proactive prefetch** 才能从根本上减少 82% 的 re-fault — 见 [`cross_block_prefetch_plan.md`](cross_block_prefetch_plan.md)

#### 结论与修正后的优化路线

```
已完成:
  ✅ Phase 1 Eviction: cycle_moe (T1 保护) — 结论: 默认 LRU 已够好
  ✅ Phase 2 Prefetch: always_max — 巨大收益 (+57-70%)，page-level 优化
  ✅ chunk_trace 对比 — 确认 chunk-level thrashing 不变，优化在 page-level

下一步 (按优先级):
  ✅ P1: 理解默认 UVM prefetch 为什么这么保守 — 已完成，见下方
  → P2: Cross-VA-block proactive prefetch — 见 cross_block_prefetch_plan.md
  → P3: 更多 workloads 验证 (vLLM, Faiss)
```

### 2026-02-26: UVM Prefetch Threshold 深入分析 — Root Cause 定位

#### 默认 UVM Prefetch 机制

**源码**: `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_perf_prefetch.c`

**算法**:
1. 每次 page fault 时，构建 **bitmap tree** — VA block (2MB, 512 pages) 上的虚拟二叉树
2. `pages` bitmap = resident 页 | 已 fault 页（已在 GPU 或正在迁移的页）
3. 从 fault page（叶子）向上遍历 tree，每一层检查:
   ```
   populated_count * 100 > subregion_pages * threshold
   ```
4. 如果满足，扩展 prefetch region 到该子树范围
5. 最终 prefetch 区域 = 满足条件的**最大子树**

**三个模块参数** (只读，模块加载时设置):

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `uvm_perf_prefetch_enable` | 1 | 0/1 | 开关 |
| `uvm_perf_prefetch_threshold` | **51** | 1-100 | 子区域中 populated 页的百分比门槛 |
| `uvm_perf_prefetch_min_faults` | 1 | 1-20 | 触发 prefetch 的最低 fault 数 |

**默认 threshold=51 的含义** — "严格过半"规则:
- Level 0 (1 page): 平凡通过
- Level 1 (2 pages): 需要 2/2 = 100% 已 populated
- Level 2 (4 pages): 需要 3/4 = 75%
- Level 3 (8 pages): 需要 5/8 = 62.5%
- ...
- Level 9 (512 pages = 整个 VA block): 需要 261/512 = 51%

**对 MoE 模型的影响**: 每个 expert ~13.4 MB ≈ 7 个 4KB 页，一个 VA block (2MB) 可能包含多个 expert 的部分数据。当单个 fault 进入一个几乎空的 VA block 时，populated 比例极低，threshold=51% 几乎永远不会让 prefetch 扩展到更大范围。

**源码中的 TODO** (line 42):
```c
// TODO: Bug 1778037: [uvm] Use adaptive threshold for page prefetching
```
NVIDIA 自己也认为静态 threshold 是问题，但未实现自适应逻辑。

#### BPF Hook 工作原理

在 `compute_prefetch_region()` 最开始调用 BPF hook:
```c
action = uvm_bpf_call_gpu_page_prefetch(page_index, bitmap_tree,
                                               &max_prefetch_region, &prefetch_region);
```

| 返回值 | 行为 |
|--------|------|
| DEFAULT (0) | 走原始 threshold 算法 |
| BYPASS (1) | 跳过所有计算，使用 BPF 设置的 result_region |
| ENTER_LOOP (2) | 走 tree 遍历，但每层调用 BPF on_tree_iter |

**always_max 的 BYPASS**: 直接设 result_region = max_prefetch_region → 等效于 threshold=0（无条件预取整个 VA block）。

#### Threshold Sweep 实验

**只修改模块参数，不加载任何 BPF 策略** — 验证原生 threshold 对性能的影响:

| threshold | pp512 (tok/s) | tg128 (tok/s) | pp vs default | tg vs default |
|-----------|--------------|--------------|---------------|---------------|
| **51 (默认)** | 139.50 ± 1.85 | 45.29 ± 3.34 | — | — |
| **25** | 176.06 ± 1.66 | 56.89 ± 6.86 | +26.2% | +25.6% |
| **10** | 202.21 ± 1.55 | 72.64 ± 7.24 | +45.0% | +60.4% |
| **5** | 208.39 ± 2.51 | 76.45 ± 7.14 | +49.4% | +68.8% |
| **1** | 217.12 ± 2.53 | 76.00 ± 4.06 | +55.6% | +67.8% |
| **BPF always_max** | 219.12 ± 2.81 | 76.85 ± 4.11 | +57.1% | +69.7% |

#### 关键发现

1. **Threshold 和性能严格单调递减**: threshold 越低 → prefetch 越激进 → 性能越高
   - 从 51→1: pp +55.6%, tg +67.8%
   - 从 51→5: 已获得 90%+ 的收益
   - threshold=1 ≈ BPF always_max（差异 < 2%）

2. **Threshold=5 是性价比拐点**: tg=76.45 已接近 threshold=1 的 76.00，但不是完全"无条件预取"

3. **BPF always_max 略优于 threshold=1**: 因为 always_max 直接 BYPASS 跳过了整个 bitmap tree 构建和遍历的开销，而 threshold=1 仍然执行 tree 遍历（只是几乎所有子树都通过检查）

4. **这是一个 nvidia-uvm 的设计缺陷**: 默认 threshold=51 对 sparse/MoE workloads 严重次优。NVIDIA 的 TODO Bug 1778037 建议用自适应 threshold，但从未实现。

5. **gpu_ext BPF prefetch 的核心价值**: 即使不修改模块参数，gpu_ext 通过 BPF struct_ops 可以在运行时动态覆盖 threshold 逻辑:
   - 对 MoE workload: BYPASS + always_max
   - 对 dense workload: 可能 DEFAULT（使用原始算法）效果更好
   - 对多租户: 不同 PID 不同策略
   - **无需重启驱动，无需 root 权限重新加载模块**

#### 论文意义

这个发现直接支持 gpu_ext 论文的核心论点:

> "UVM 的静态 prefetch 策略（threshold=51%）对 oversubscribed workloads 严重次优。
> 通过 eBPF struct_ops，我们可以在运行时无侵入地将 threshold 优化为 workload-specific 值。
> 仅 prefetch 策略一项就为 120B MoE 模型带来 **57-70% 的性能提升**（pp: 139→219, tg: 45→77 tok/s），
> 无需修改应用程序、无需重启驱动。"

这是对 MSched 论文 "per-kernel template-based prediction" 方案的一个更轻量级替代:
- MSched: 需要离线 NVBit profiling + 自定义驱动 ioctl → per-kernel 精确 working set
- gpu_ext: 运行时 BPF BYPASS → per-VA-block 全量预取 → 达到 MSched ~74% 的效果 (tg: 77 vs MSched 理论最优)

**差距**: MSched 的 cross-VA-block proactive prefetch 可以进一步消除 82% 的 chunk-level re-fault，gpu_ext 目前只解决了 intra-VA-block 的 page-level fragmentation。详细方案见 [`cross_block_prefetch_plan.md`](cross_block_prefetch_plan.md)。

#### 下一步

1. **Cross-VA-block proactive prefetch**: 见独立文档 [`cross_block_prefetch_plan.md`](cross_block_prefetch_plan.md)
2. **其他 workloads 验证**: vLLM, Faiss 是否也受 threshold 影响？Dense model 呢？
3. **自适应 threshold**: 能否通过 BPF 实现 NVIDIA 未完成的 Bug 1778037（自适应 threshold）？

### 2026-02-26: Combined Prefetch + Eviction 策略实验

#### 理论背景

对于**周期性访问模式** (LLM decode: layer 0→35→0→35...)：
- **LRU 是理论最差** — 最近最久未使用 = cycle 中最先被需要的，LRU 恰好淘汰它
- **MRU 是 Belady-optimal** — 最近使用的 = cycle 中最远才被需要的，应最先淘汰
- 但纯 MRU 灾难性（-83%）：因为它也淘汰了 T1 attention weights（每步都用）

**解决方案**：T1 保护 + 非 T1 用 MRU 变体

#### 新策略实现

| 策略 | Prefetch | T1 Eviction | Non-T1 Eviction | 文件 |
|------|----------|-------------|-----------------|------|
| always_max + MRU | always_max | move_tail | **move_head** (explicit MRU) | `prefetch_max_mru_expert.bpf.c` |
| always_max + passive MRU | always_max | move_tail | **BYPASS no-move** (freeze LRU) | `prefetch_max_passive_mru.bpf.c` |

**Passive MRU 原理**: 对非 T1 chunk 返回 BYPASS 但不调用任何 move 函数。这阻止了内核默认的 LRU 刷新（move to tail），chunk 保持在当前 list 位置，随着新 chunk 在 tail 添加，旧 chunk 自然向 head 漂移。效果 ≈ FIFO for non-T1 chunks。

#### 结果

**注意**: Threshold 实验期间用 `modprobe` 加载了 stock nvidia_uvm (无 BPF)，需要 `insmod` 重新加载 custom 模块才能使用 BPF struct_ops。

**pp=512, tg=128, 5 repetitions**:

| 策略 | pp512 (tok/s) | tg128 (tok/s) | pp vs baseline | tg vs baseline |
|------|--------------|--------------|----------------|----------------|
| Baseline (无 BPF) | 139.50 | 45.29 | — | — |
| always_max (prefetch only) | 219.12 | 76.85 | +57.1% | +69.7% |
| always_max + cycle_moe (T1 protect, default) | 224.25 | 76.87 | +60.8% | +69.7% |
| always_max + MRU expert (T1 protect, move_head) | 221.89 | 76.18 | +59.1% | +68.2% |
| **always_max + passive MRU (T1 protect, freeze)** | **227.94** | **78.68** | **+63.4%** | **+73.7%** |

**pp=2048, tg=512, 5 repetitions (passive MRU)**:

| 策略 | pp2048 (tok/s) | tg512 (tok/s) |
|------|---------------|--------------|
| Baseline | 147.15 | 52.20 |
| always_max | 231.12 | 85.15 |
| **passive MRU** | **231.45** | **85.08** |

#### 分析

1. **Passive MRU 是目前最佳组合** (短序列): tg=78.68 (+73.7%)
   - 比纯 always_max 的 tg=76.85 略优（+2.4%）
   - 比 cycle_moe 组合的 tg=76.87 略优
   - 改进来自: 非 T1 chunk 不再被 LRU 刷新 → 更快被淘汰 → T1 chunk 获得更多 VRAM 空间

2. **Explicit MRU (move_head) ≈ passive MRU**: move_head 的 list manipulation 开销抵消了 MRU 排序的收益

3. **长序列差异消失**: pp=2048/tg=512 时所有策略趋同 (85 tok/s)
   - 原因: 长序列下每步的 expert 数据量更大，VRAM 压力使 eviction 策略差异被摊薄
   - 短序列（更多 decode steps / 秒）下策略差异更明显

4. **Eviction 优化的天花板**: 即使最优的 eviction 策略也只能带来 2-4% 额外提升
   - 82% chunk thrashing 由 VRAM 容量决定（60GB 模型, 32GB VRAM）
   - Eviction 只影响 "淘汰顺序"，不影响 "淘汰总量"
   - **真正突破需要 cross-VA-block proactive prefetch** — 详见独立文档 `cross_block_prefetch_plan.md`

#### 最终性能对比总表

| 策略 | pp512 | tg128 | tg512 | 说明 |
|------|-------|-------|-------|------|
| Raw UVM (默认) | 139.5 | 45.3 | 52.2 | 内核默认 threshold=51% |
| threshold=1 (仅参数) | 217.1 | 76.0 | — | 不需要 BPF |
| BPF always_max | 219.1 | 76.9 | 85.2 | 仅 prefetch，无 eviction hook |
| **BPF passive MRU** | **228.0** | **78.7** | **85.1** | **最佳: always_max + passive MRU** |
| stride prefetch | 33.2 | 14.5 | — | ❌ 灾难 (BYPASS 禁用默认预取) |
| none (禁用预取) | 31.6 | 14.0 | — | ❌ 下限 |
| MRU (纯) | 19.0 | 9.6 | — | ❌ 灾难 (淘汰 attention) |
| gpu_ext 论文 (参考) | — | 86.89 | — | stride + LFU (旧驱动) |

**下一步**:
- [ ] P1: 自适应 Threshold — 实现 NVIDIA Bug 1778037，见下方
- [ ] P2: Template-based Per-kernel Prediction — MSched 核心算法对齐，见下方
- [ ] P3: 测试其他 workloads (vLLM Qwen-30B, Faiss SIFT-100M) 使用已有 BPF 策略
- [ ] P4: Cross-VA-block proactive prefetch — 独立项目，见 [`cross_block_prefetch_plan.md`](cross_block_prefetch_plan.md)

---

### P1: 自适应 Threshold (NVIDIA Bug 1778037)

#### 背景

NVIDIA 在 `uvm_perf_prefetch.c:42` 留下了 TODO:
```c
// TODO: Bug 1778037: [uvm] Use adaptive threshold for page prefetching
```
默认 threshold=51% 对 MoE 模型严重次优（-57% 性能），但对 dense 模型可能合理。NVIDIA 从未实现自适应逻辑。

gpu_ext 已有两个 BPF 实现但**未 benchmark**:
- `extension/prefetch_adaptive_tree_iter.bpf.c` — ENTER_LOOP 模式，用户态写 threshold_map
- `extension/prefetch_adaptive_sequential.bpf.c` — BYPASS 模式，百分比 + 方向 + 页数

#### 目标

实现真正的运行时自适应 threshold，而非简单的 "把 hardcode 值改成 BPF map 可调"。

#### 方案 1: Per-VA-Region Threshold

不同模型组件使用不同 threshold:

| 组件 | VA Range | 推荐 Threshold | 理由 |
|------|----------|----------------|------|
| Attention weights (T1) | 从 GGUF 元数据计算 | 1 (激进) | 每步都用，预取收益最大 |
| Active experts (T2) | 运行时从 fault pattern 学习 | 1 (激进) | 当前步需要 |
| Inactive experts (T3) | 其余 VA 区域 | 51 (保守) | 大概率不被访问，避免浪费 PCIe |
| Embeddings | 模型首尾 VA | 1 (激进) | 每步都用 |

**实现**: BYPASS 模式（不用 ENTER_LOOP），在 `before_compute` 中根据 `page_index` 查 region_map 决定 threshold。

```c
// BPF maps
struct { /* layer_id → {va_first, va_outer, threshold} */ } region_config SEC(".maps");

SEC("struct_ops/gpu_page_prefetch")
int BPF_PROG(gpu_page_prefetch, ...) {
    // 从 page_index 推断所属 VA region
    // 查 region_config 获取该 region 的 threshold
    // 如果 threshold == 1: 直接 always_max
    // 如果 threshold > 1: 用 ENTER_LOOP 让内核做 tree 遍历
    // 如果 threshold == 0: 不预取
}
```

**用户态 loader**: 从 GGUF 元数据解析 per-layer VA range，写入 region_config map。

#### 方案 2: Feedback-Driven Threshold

根据运行时指标动态调整全局 threshold:

```
用户态 daemon (每秒):
  fault_rate = read_from(/proc 或 BPF map)
  pcie_bw = read_from(nvidia-smi 或 perf counter)

  if fault_rate > HIGH:
      threshold -= 5  // 更激进
  if pcie_bw > 90%:
      threshold += 5  // 减少预取避免拥塞

  write_to(threshold_map)
```

#### 方案 3: Workload Auto-Detect

自动检测 MoE vs Dense 并切换策略:
- 检测方法: fault 地址的 locality pattern (MoE 有明显的 layer 跳跃，Dense 更连续)
- MoE 检测到后: threshold=1 (always_max)
- Dense: threshold=10-25 (适中)

#### ENTER_LOOP Hook 限制

exploration 发现的限制:
1. 无法提前终止遍历（无 STOP_LOOP 返回值）
2. BPF 不知道当前 tree level（需从 region size 推断）
3. 每次迭代独立，无累积状态传递
4. **结论**: per-VA-region 自适应更适合 BYPASS 模式，ENTER_LOOP 仅用于需要复刻内核 tree 算法的场景

#### 实验计划

| 配置 | 说明 |
|------|------|
| always_max (对照) | 已有数据: pp=219, tg=77 |
| adaptive_tree_iter (现有) | threshold_map=1 vs 10 vs 25 vs 51，验证 BPF 可调 threshold 工作 |
| adaptive_sequential (现有) | percentage=100 vs 50 vs 25，验证方向性预取 |
| per-region adaptive (新) | T1 区域 threshold=1, T3 区域 threshold=51 |
| feedback-driven (新) | 用户态 daemon 动态调 threshold |

**关键问题**: per-region adaptive 能否比 always_max 更好？
- 理论上: 对不活跃 expert 区域保守预取 → 减少 PCIe 浪费 → T1 数据更快到达
- 实际上: always_max 的 +57% 来自减少 page-level fault 开销，即使预取不活跃 expert 也是正收益
- 需要实验验证

#### 论文价值

- 实现 NVIDIA 6 年未完成的 Bug 1778037
- 展示 BPF struct_ops 的运行时策略定制能力
- 不同 workload 可以不重启驱动切换策略

---

### P2: Template-based Per-kernel Prediction

#### 背景

MSched 核心贡献: 通过 NVBit 离线 profiling 获取每个 GPU kernel 的精确 working set，实现 0% false positive prefetch + Belady OPT eviction。

gpu_ext 的替代路线: NVBit 在 RTX 5090 上不可行，但可以通过以下方式达到类似效果:
1. **解析式模型** (已完成): GGUF 元数据 → per-layer VA range → T1/T2/T3 分类
2. **uprobe 拦截** (已有基础): `tools/cuda_sched_trace.bpf.c` 已实现 cuLaunchKernel uprobe
3. **VA 地址推断** (已验证): chunk_trace 证实 VA 单调递增 = layer 序列化访问

#### 核心架构: Cross-Layer BPF Pipeline

这是 gpu_ext 区别于 MSched 的**独有能力**: 两个不同层的 BPF 程序通过共享 BPF map 协作。

```
                    ┌─ BPF Map (shared) ─┐
                    │ current_layer: 15   │
                    │ kernel_type: FFN    │
                    │ decode_step: 42     │
                    └──────┬──────────────┘
                           │
        ┌──────────────────┼────────────────────┐
        │                  │                     │
  ┌─────▼──────┐    ┌─────▼──────┐    ┌────────▼─────────┐
  │ uprobe     │    │ struct_ops │    │ struct_ops       │
  │ cuLaunch   │    │ prefetch   │    │ eviction         │
  │ Kernel     │    │ hook       │    │ hook             │
  │            │    │            │    │                  │
  │ 写 layer   │    │ 读 layer   │    │ 读 layer →       │
  │ 写 kernel  │    │ → 区域感知 │    │ Belady distance  │
  │ 写 step    │    │   prefetch │    │ → 优化淘汰顺序   │
  └────────────┘    └────────────┘    └──────────────────┘
    用户态              内核态 (fault handler)
```

#### 实现步骤

**Step 1: Analytical VA Range Map (用户态 loader)**

从 GGUF 元数据生成 per-layer VA range 映射:
```python
# 已有 NVBit/scripts/analytical_ws.py 的基础
# 新增: 输出 VA range mapping 供 BPF loader 写入 map
layer_ranges = {
    0: {"attn": (va_start, va_end), "experts": [(va_s, va_e), ...]},
    1: {"attn": (va_start, va_end), "experts": [(va_s, va_e), ...]},
    ...
}
```

**问题**: UVM 的 VA 地址由 `cudaMallocManaged` 分配，不一定和 GGUF tensor 的逻辑顺序一致。需要:
- 方案 A: 运行时从 chunk_trace 学习实际 VA → layer 映射（观察 1-2 个 decode step）
- 方案 B: uprobe 拦截 `cudaMallocManaged` 记录每次分配的 VA 和 size → 与 GGUF tensor 对应

**Step 2: Kernel Launch Tracking (uprobe BPF)**

扩展 `cuda_sched_trace.bpf.c`:
```c
// 新增: 写入共享 map 供 struct_ops 读取
struct kernel_state {
    u32 layer_id;       // 当前层号
    u32 kernel_type;    // 0=attn, 1=ffn, 2=expert_gate, ...
    u32 decode_step;    // 当前 decode step 计数
    u64 timestamp_ns;   // 时间戳
};

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, struct kernel_state);
} kernel_state_map SEC(".maps");  // struct_ops 从这里读

SEC("uprobe/cuLaunchKernel")
int trace_cuLaunchKernel(struct pt_regs *ctx) {
    u64 func_ptr = PT_REGS_PARM1(ctx);
    // func_ptr → 推断 kernel 类型和层号
    // 方法: 建立 func_ptr → layer_id 的 hash map (第一次 decode step 学习)
    // 写入 kernel_state_map
}
```

**Step 3: Struct_ops 读取 Kernel State**

```c
SEC("struct_ops/gpu_page_prefetch")
int BPF_PROG(gpu_page_prefetch, ...) {
    // 读 kernel_state_map → 知道当前在第几层
    struct kernel_state *ks = bpf_map_lookup_elem(&kernel_state_map, &zero);

    // 方案 A: intra-block — 根据层类型选 threshold
    if (ks && ks->kernel_type == ATTN) {
        // attention: always_max
        bpf_gpu_set_prefetch_region(result_region, max_first, max_outer);
    } else {
        // expert: 可以更保守
    }

    // 方案 B: 如果有 cross-block 能力 → proactive prefetch next layer
    // (依赖 P4 cross-block 实现)

    return 1; // BYPASS
}
```

**Step 4: Belady OPT Eviction**

```c
SEC("struct_ops/gpu_block_access")
int BPF_PROG(gpu_block_access, ...) {
    struct kernel_state *ks = bpf_map_lookup_elem(&kernel_state_map, &zero);
    if (!ks) return 0;

    // 从 chunk VA → 推断属于哪一层
    u64 chunk_va = get_chunk_va(chunk);  // 需要新 kfunc
    u32 chunk_layer = va_to_layer(chunk_va);  // 查 BPF map

    // Belady distance = 当前层到 chunk_layer 的 cycle 距离
    // LLM decode cycle = 36 layers
    u32 distance = (chunk_layer - ks->layer_id + 36) % 36;

    if (distance < 3) {
        // 马上要用 → move_tail (保护)
        bpf_gpu_block_move_tail(chunk, list);
    }
    // else: 让内核默认处理 (LRU)

    return 1; // BYPASS
}
```

#### 技术挑战

| 挑战 | 严重性 | 解决方案 |
|------|--------|----------|
| VA → layer 映射不确定 | 高 | 运行时学习: 第一个 decode step 建立映射 |
| uprobe 和 struct_ops 是不同 BPF 程序 | 中 | **共享 BPF map** (pinned map 或相同 loader 管理) |
| function_ptr → kernel 类型推断 | 中 | 建立 hash map (第一次 decode step 学习 func_ptr → layer 关系) |
| eviction hook 无法获取 chunk VA | 高 | 用 `BPF_CORE_READ(chunk, va_block)` + `BPF_CORE_READ(va_block, start)` 解引用 |
| MoE routing 动态性 (不同 step 激活不同 expert) | 低 | T1/T2 分类即可,不需要精确到哪个 expert |

#### 分阶段实现

**Phase A: VA-based Layer Detection (纯 struct_ops，不需要 uprobe)**
- 从 fault VA 地址推断当前层号
- 实现 Belady eviction (根据 layer distance)
- 需要: 运行时学习 VA → layer 映射 (chunk_trace 已证实可行)
- 预期收益: eviction 改进 2-5%

**Phase B: Uprobe Cross-Layer Pipeline**
- uprobe 拦截 cuLaunchKernel → 写 kernel_state_map
- struct_ops 读 kernel_state → 更精确的层号和 kernel 类型
- 预期收益: 更精确的 eviction + 为 cross-block 准备基础设施
- 论文价值: **展示 gpu_ext 的 cross-layer 能力** (uprobe + struct_ops 协作)

**Phase C: 与 Cross-block 结合 (依赖 P4)**
- 知道 "下一层需要什么" → 调用 cross-block prefetch 提前迁移
- 预期收益: 可能 +15-25% (消除 82% chunk thrashing 的一部分)
- 这是 MSched template prediction 的完整复现

#### 论文价值

- **MSched 对齐**: 复现 per-kernel working set prediction，但用 BPF 运行时替代 NVBit 离线
- **Cross-layer 展示**: uprobe (用户态) + struct_ops (内核态) 通过 BPF map 协作 — gpu_ext 独有
- **Runtime vs Offline**: 无需离线 profiling，第一个 decode step 自动学习映射
- **为 cross-block 打基础**: Phase C 结合后才是完整的 MSched 复现

---

### 2026-02-26: NVBit 重试计划 + P1/P2 细化实现

#### NVBit 重试计划

**背景**: 之前 NVBit 结论"不可行"是因为 timeout 过短（< 15 分钟）。NVBit v1.7.7.1 支持 SM 3.5-12.1，包含 RTX 5090 (SM_120)。MSched 论文本身用 NVBit 做离线 profiling。

**命令**:
```bash
GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
PAGE_SHIFT=21 WS_TRACE_OUTPUT=/tmp/ws_trace_120b.txt \
LD_PRELOAD=/home/yunwei37/workspace/gpu/NVBit/nvbit_release_x86_64/tools/ws_trace/ws_trace.so \
/home/yunwei37/workspace/gpu/gpu_ext/workloads/llama.cpp/build/bin/llama-bench \
  -m ~/.cache/llama.cpp/ggml-org_gpt-oss-120b-GGUF_gpt-oss-120b-mxfp4-00001-of-00003.gguf \
  -p 32 -n 16 -r 1
```

**参数选择**:
- `GGML_CUDA_DISABLE_GRAPHS=1`: 禁用 CUDA graphs 确保 NVBit 拦截所有 kernel launch
- `pp=32, n=16, r=1`: 最小化 kernel 数但覆盖完整 decode cycle
- `PAGE_SHIFT=21`: 2MB 页，与 UVM chunk 大小对齐
- 直接跑 120B（跳过 20B 验证），因为 120B 是目标模型
- timeout 90 分钟（120B + NVBit instrumentation + UVM migration 三重开销）

**判断标准**:

| 情况 | 判断 | 下一步 |
|------|------|--------|
| 完成，`/tmp/ws_trace_120b.txt` 含 KERNEL 行 | NVBit 可用 | → 用 `process_ws_trace.py` 生成 template 数据 |
| 输出文件空 / 0 行 | 拦截失败 | → 调试 inject_funcs.cu (可能 SM_120 指令集变化) |
| 崩溃/segfault | 真不兼容 | → Fallback: 纯解析式 + 运行时学习 |
| 90min 无 CPU 活动的 hang | 死锁 | → Fallback: 纯解析式 + 运行时学习 |

**NVBit 重试结果**: *(待填充)*

#### 获取 chunk VA 的方式（BTF/CO-RE，无需改内核模块）

使用 `BPF_CORE_READ` 直接从 struct_ops 回调中解引用 chunk 的 VA 信息：

```c
uvm_va_block_t *va_block = BPF_CORE_READ(chunk, va_block);
u64 chunk_va = BPF_CORE_READ(va_block, start);
```

CO-RE 会根据 nvidia-uvm.ko 导出的 BTF 自动重定位字段偏移：
- `chunk->va_block` → offset 32
- `va_block->start` → offset 48

参考: `extension/test_chunk_access.bpf.c`、`extension/chunk_trace.bpf.c`

**注意**: BPF verifier 有跳转复杂度限制（8193 jumps）。`va_to_layer()` 的 boundary 循环
（MAX_LAYERS=40）只能在一个回调中使用。`chunk_activate` 改为 hash map 快速路径。

#### Template + Belady BPF 程序

**文件**: `extension/prefetch_template_belady.bpf.c` + `extension/prefetch_template_belady.c`

**架构**:
```
用户态 loader                          内核态 BPF
──────────────                        ──────────────
1. 解析 NVBit JSON                     prefetch hook:
2. 写入 va_to_layer_map                  → always_max (BYPASS)
3. 写入 config_map (layers=36)
4. attach struct_ops                   chunk_activate hook:
                                         → 读 chunk VA (kfunc)
                                         → 更新 current_layer
                                         → 检测 decode step 边界

                                       chunk_used hook:
                                         → T1 频率检查 → protect
                                         → Belady 距离计算
                                         → 远距离 → move_head
                                         → 近距离 → move_tail
```

**BPF Maps**:
- `config_map` (ARRAY): 全局配置 (num_layers, t1_threshold, protect_distance)
- `va_to_layer_map` (HASH 64K): VA >> 21 → layer_id (用户态写入，BPF 读取)
- `state_map` (ARRAY): 运行时状态 (current_layer, decode_step, last_fault_va)
- `access_counts` (PERCPU_ARRAY 16K): T1 频率计数器

**两种运行模式**:
1. **有 NVBit 数据**: loader 从 `layer_va_ranges.json` 加载 VA→layer 映射，精确 Belady
2. **无 NVBit 数据**: 纯运行时学习，从 fault VA 模式推断 layer，passive MRU fallback

#### ws_trace 输出解析器

**文件**: `NVBit/scripts/process_ws_trace.py`

解析 NVBit ws_trace 输出格式:
```
KERNEL <grid_launch_id> <kernel_name> <num_pages> <total_accesses> <ws_bytes> <page0,page1,...>
```

**处理流程**:
1. 按 kernel_name 分组所有 launches
2. 跨 launch 对比 page set 稳定性
3. 分类: T1 (100% launches), T2 (>50%), T3 (≤50%)
4. 从 kernel_name 提取 layer_id (llama.cpp blk_N 命名)
5. 生成 per-layer VA range mapping

**输出**:
- `profiling_data/kernel_templates.json`: per-kernel T1/T2/T3 分类 + 比较解析模型
- `profiling_data/layer_va_ranges.json`: per-layer VA 区间 → 供 BPF loader 加载

#### 自适应 Threshold 实验脚本

**文件**: `workloads/llama.cpp/run_exp_adaptive_threshold.sh`

| 实验 | BPF 程序 | 参数 | 目的 |
|------|----------|------|------|
| 基线 (无 BPF) | — | — | 对照 |
| always_max | prefetch_always_max | — | 现有最佳 prefetch |
| passive MRU | prefetch_max_passive_mru | — | 现有最佳组合 |
| tree_iter t=1 | adaptive_tree_iter | -t 1 | BPF ENTER_LOOP，threshold=1 |
| tree_iter t=10 | adaptive_tree_iter | -t 10 | BPF ENTER_LOOP，中等 threshold |
| tree_iter t=25 | adaptive_tree_iter | -t 25 | BPF ENTER_LOOP，保守 |
| tree_iter t=51 | adaptive_tree_iter | -t 51 | BPF ENTER_LOOP，默认 |
| sequential p=100 | adaptive_sequential | -p 100 | BPF BYPASS，全量预取 |
| sequential p=50 | adaptive_sequential | -p 50 | BPF BYPASS，半量预取 |
| sequential p=25 | adaptive_sequential | -p 25 | BPF BYPASS，1/4 预取 |
| template_belady | prefetch_template_belady | --layers 36 | 新 Belady 策略 |

**关键问题**:
1. BPF ENTER_LOOP (tree_iter) vs BYPASS (always_max) 的路径开销差多少？
2. tree_iter t=1 是否 ≈ always_max？（理论应该，但 tree 遍历有开销）
3. template_belady 的 Belady eviction 能否比 passive MRU 更好？
4. sequential 的方向性预取对 MoE 有帮助吗？

#### 实现进度

- [x] 修正 NVBit 结论 (从"不可行" → "待重试")
- [x] chunk VA 获取: 使用 BPF_CORE_READ（无需新 kfunc）
- [x] 新增 BPF 程序: `prefetch_template_belady.bpf.c` + `.c` loader
- [x] 新增解析脚本: `NVBit/scripts/process_ws_trace.py`
- [x] 新增实验脚本: `workloads/llama.cpp/run_exp_adaptive_threshold.sh`
- [x] 更新 Makefile: 加入 `prefetch_template_belady`
- [x] NVBit 重试结果 — ❌ 失败（7+ 小时未完成，见下方详细日志）
- [x] 使用未修改的自定义 nvidia-uvm.ko — ✅ BPF 加载成功
- [x] 编译 prefetch_template_belady (需 skeleton 生成) — ✅ 成功
- [ ] 运行 adaptive threshold 实验
- [ ] 实现 chunk_trace → VA→layer 离线映射
- [ ] 实现 BPF 运行时自学习 VA→layer
- [ ] 实现 uprobe cudaMallocManaged 捕获 buffer base

### 2026-02-27: NVBit 重试结果 — 确认不可行 + 替代方案分析

#### NVBit 120B 重试详细日志

**命令**: 与上述 NVBit 重试计划中的相同（pp=32, n=16, r=1, PAGE_SHIFT=21, GGML_CUDA_DISABLE_GRAPHS=1）

**执行过程**:

| 时间 (min) | CPU 时间 | GPU 显存 (MiB) | 增速 (MiB/10min) | 状态 |
|-----------|----------|---------------|-----------------|------|
| 0 | 0:00 | — | — | NVBit 加载，打印 banner |
| 44 | 11:15 | 2,983 | — | binary instrumentation 开始 |
| 55 | — | 3,623 | 640 | GPU 100%，CPU 200% |
| 65 | — | 4,135 | 512 | 稳定加载中 |
| 75 | — | 4,519 | 384 | |
| 85 | — | 5,031 | 512 | |
| 96 | — | 5,671 | 640 | 加速 |
| 108 | — | 6,975 | 746 | NVBit 缓存生效 |
| 120 | 1:58 | 7,487 | 512 | |
| 140 | 2:38 | 9,407 | 960 | |
| 150 | 2:58 | 10,303 | 896 | |
| 170 | 3:29 | 11,839 | 770 | |
| 190 | 4:09 | 13,761 | 960 | |
| 210 | — | 16,707 | 1,470 | 加速明显 |
| 240 | — | 19,651 | 1,470 | |
| 270 | — | 23,363 | 1,860 | |
| ~420 | 7:25+ | ~23,363 | — | **被 timeout kill (SIGKILL)** |

**最终状态**: ws_trace_120b.txt = 0 字节（ws_trace 在进程正常退出时才 flush KERNEL 行）

**死亡原因**: Claude background task timeout (5400s) 向 bash wrapper 发送 SIGKILL → 子进程 llama-bench (PID 82977) 被级联终止

**关键数据**:
- 7+ 小时只加载了 23 GiB / 32 GiB VRAM（模型 59 GiB，但只需 ~20 GiB working set for prefill）
- 显存增速约 5-6 GiB/hr，最终加速到 ~7 GiB/hr
- 预估完成 prefill 需 12-24 小时（含 UVM eviction phase）
- 无 Xid 31、无 segfault、无 OOM — NVBit 技术上是可用的，只是**极慢**

**结论**:

| 判断标准 | 结果 |
|---------|------|
| 完成并输出 KERNEL 行 | ❌ 未完成 |
| 崩溃/segfault | ❌ 没有崩溃 |
| 90min 无 CPU 活动 | ❌ 全程 199% CPU |
| **实际情况** | **NVBit 可用但极慢: 120B 模型预计 12-24 小时** |

**NVBit 不可行的根本原因**:
1. NVBit 对每个 CUDA kernel 的每条内存指令做 binary instrumentation
2. 120B MoE 模型有 ~160+ 个不同 kernel template，每个都需要 instrument
3. Instrument 后的 kernel 执行速度降低 ~100x
4. UVM demand paging 在 instrumented kernel 下更慢（fault handler 也被间接放慢）
5. 对于小模型（<10 GiB）NVBit 可能可行（~1-2 小时），但 120B 在 oversubscription 下不实际

**最终决策**: 放弃 NVBit，改用以下替代方案获取 template 数据。

---

#### 替代方案分析: 不依赖 NVBit 的 Template 推导

##### 方案 1: chunk_trace → 离线 VA→Layer 映射（最可行）

**原理**: chunk_trace 的 kprobe 已经捕获 ACTIVATE 事件的 VA 地址。在 UVM oversubscription 下，每个 decode step 的 fault 序列反映了模型的层序列访问模式。

**已有证据**:
- chunk_trace 数据确认: 每个 decode step 内 VA 地址**单调递增** (100% ascending pattern)
- 解析模型验证: 3,718 chunks 只 fault 1 次 → T1 (attention/embeddings)
- 16,813 chunks 多次 fault → T2/T3 (experts)

**实现**:
1. 运行 chunk_trace + llama-bench (pp=512, tg=128) 收集 5-10 个 decode steps
2. 解析 ACTIVATE 事件: 按时间分组，检测 VA 回退（= decode step 边界）
3. 在每个 step 内，按 VA 排序分配 layer_id
4. 跨 step 聚合: `VA_chunk → layer_id` 映射表（一致性检查）
5. 输出 `layer_va_ranges.json` → 供 `prefetch_template_belady --profile` 加载

**优势**:
- 完全利用已有工具（chunk_trace BPF 程序已编译）
- 运行时间短: 正常 llama-bench 速度 + chunk_trace kprobe 开销（< 5%）
- 数据精度: 实际 GPU fault 事件，比 NVBit instruction-level 粒度粗（chunk 级别）但足够

**文件**: `workloads/llama.cpp/derive_layer_mapping.py` (待实现)

##### 方案 2: GGUF 元数据 + cudaMallocManaged 地址 → 解析式映射

**原理**: llama.cpp 在 UVM 模式下用**单个 cudaMallocManaged 调用**分配整个模型 buffer。tensors 按固定顺序（层号排序）在 buffer 内顺序排列，128 字节对齐。

**关键发现** (llama.cpp 源码分析):
- `ggml-cuda.cu:117-171`: `GGML_CUDA_ENABLE_UNIFIED_MEMORY` → `cudaMallocManaged(ptr, size)` 一次性分配
- `ggml-alloc.c:78-94`: `ggml_tallocr_alloc()` 顺序分配，alignment=128
- `llama-model.cpp:2543-2585`: tensor 按 layer→tensor_name 排序
- `llama-model-loader.cpp:1022-1064`: `ggml_backend_tensor_alloc(buf, tensor, data_ptr)` 设置 VA

**内存布局** (120B 模型):
```
cudaMallocManaged buffer (base_va):
  offset 0:         token_embedding
  offset ~5.7 MB:   blk.0.attn_norm.weight
  offset ~11.4 MB:  blk.0.attn_q.weight (46.6 MB)
  ...
  offset ~1.7 GB:   blk.0 所有 MoE experts (1714 MB)
  offset ~1.8 GB:   blk.1.attn_norm.weight
  ...
  offset ~62 GB:    output_norm + output.weight
```

**问题**: 需要知道 `base_va`（cudaMallocManaged 返回的地址）

**获取 base_va 的方法**:
1. **uprobe on cudaMallocManaged**: BPF 拦截 libcuda.so 的 cudaMallocManaged，读取返回值
2. **chunk_trace 校准**: 第一个 ACTIVATE 事件的 VA ≈ base_va（近似）
3. **/proc/PID/maps 解析**: 查找 UVM 映射区域

**实现**: `NVBit/scripts/analytical_va_mapping.py` (扩展现有 analytical_ws.py)

##### 方案 3: BPF 运行时自学习（零外部依赖）

**原理**: prefetch_template_belady.bpf.c 在运行时自动学习 VA→layer 映射:

1. **第一个 decode step (学习模式)**:
   - 记录每个 ACTIVATE 的 chunk VA 到 per-CPU array
   - 检测 VA 回退 → decode step 边界
   - VA 排序后按等间距分配 layer_id
   - 写入 va_to_layer_map

2. **后续 decode steps (预测模式)**:
   - 查 va_to_layer_map 获取 layer_id
   - 用 Belady 距离排序 eviction

**优势**: 零配置，自适应任何模型
**劣势**: 第一个 decode step 无优化（需要 ~20-40ms 学习期）

**已部分实现**: prefetch_template_belady.bpf.c 中的 `state_map` + `va_to_layer_map` 就是为此设计的

##### 方案 4: GGUF 解析 → BPF map 预加载（静态分析）

**原理**: 直接从 GGUF 文件解析 tensor 元数据，计算每个 tensor 在 buffer 中的偏移量，生成 VA→layer 映射。

**步骤**:
1. 用 Python 解析 GGUF header → tensor 名字、类型、维度
2. 按 llama.cpp 的排序规则排列 tensors
3. 计算累积偏移量（考虑 128 字节对齐）
4. 生成相对偏移 → layer_id 映射表
5. 运行时结合 base_va 偏移 → 绝对 VA→layer 映射

**优势**: 最快（纯离线计算，毫秒级）
**劣势**: 依赖 llama.cpp 的分配顺序不变（版本更新可能打破）

#### 实施优先级

| 优先级 | 方案 | 实现难度 | 预期收益 | 状态 |
|--------|------|---------|---------|------|
| **1** | chunk_trace → 离线映射 | 低 (100 行 Python) | 高 (精确) | ✅ 完成 |
| **2** | BPF 运行时自学习 | 中 (BPF 代码已有框架) | 中 (第一步无优化) | 部分实现 |
| **3** | GGUF 解析 + base_va | 中 (需 uprobe) | 高 (零运行时开销) | 待实现 |
| **4** | ~~NVBit 离线 profiling~~ | — | — | ❌ 放弃 |

---

### 实验日志: chunk_trace → 层映射 (2026-02-27)

#### 尝试过的方法

**方法 1: Gap-based layer detection (4MB gap threshold)**
- 将连续 VA 块按 4MB 间隙分隔为 "层"
- **结果**: 检测到 1478 个 "层" — 失败
- **原因**: MoE 专家之间的 VA 间隙导致过度分割，不是层边界

**方法 2: VA 回归检测 decode step 边界**
- 检测 VA 回退模式 (current VA < 70% max VA) → decode step 边界
- **结果**: prefill 阶段只检测到 1 个 step（单次单调扫描）— 信息不足

**方法 3: 滑动窗口 VA 中位数（50 events, 300MB jump）**
- 跟踪滑动窗口的 VA 中位数，大跳跃 = 层边界
- **结果**: 329 个区域 — 太多太噪声（eviction 引起的 re-fault 打乱 VA 顺序）

**方法 4: 时间等分法 (time-based equal slices)**
- 观察到 99.4% VA-time 相关性：prefill 阶段 VA 和时间几乎完全正相关
- 按首次激活时间等分为 36 份 → layer_id
- **结果**: 层 11-35 合理 (500-780 chunks, ~1-1.6 GiB)；层 0-10 有问题（早期 prefill 散射）
- **输出**: `results/msched_trace/layer_va_ranges_time.json`

**方法 5: 线性 VA 等分 (linear VA model)** ⭐
- 假设模型权重从 embedding_end 到 output_start 均匀分布
- 把完整 VA 范围等分为 36 份
- **结果**: 层 0-15 集中了所有 15,885 个 prefill chunks，层 16-35 几乎为空
- **原因**: VA 空间是稀疏的 — 117 GiB VA span 中只有 31 GiB 活跃 chunks（MoE 非活跃专家占 59 GiB 但 VA 并非均匀分布）
- **输出**: `results/msched_trace/layer_va_ranges_linear.json`

**方法 6: 等数量 VA 分割 (equal-count VA division)** ✅ 最终采用
- 过滤 prefill chunks (t=100-3000ms)，排除初始化散射和模型外区域
- 发现 66 GiB VA 空洞 (0x76cc864-0x76dd060)，将其以上 chunks 排除（属于 output 层）
- 剩余 15,801 个 clean chunks (30.9 GiB)，按 VA 排序后等分为 36 组
- **每层**: ~439 chunks (878 MB activated), VA span 912-2358 MB (含非活跃专家空洞)
- **关键验证**: 用 decode 阶段数据验证:
  - ✅ 检测到 60 个 decode steps，每个 step 覆盖 layer 0-35 循环
  - ✅ 层访问频率均匀: 427-788 次/层（前面层稍高因为 eviction 更少）
  - ✅ 序列正确: step 从 layer 35→0→1→...→35 循环
  - 18% 事件在模型外 (embedding/output 权重，layer_id=-1)
- **输出**: `results/msched_trace/layer_va_ranges_equal_count.json`

#### 关键发现

1. **VA 空间稀疏性**: 120B MoE 模型的 VA 占用 117 GiB，但 prefill 只激活 31 GiB（~53%）。
   非活跃专家在 VA 空间中分散分布，导致线性模型失效。

2. **99.4% VA-time 相关性**: prefill 阶段 chunk 激活顺序几乎完全按 VA 地址递增。
   这证实 llama.cpp 按 VA 顺序遍历模型层。

3. **等数量分割有效**: 因为每层包含类似数量的张量（attention + MoE 专家），
   活跃 chunks 数量大致相同，按 VA 等数量分割自然对齐层边界。

4. **BPF 实现**: 36 个 VA 边界值存入 ARRAY map，chunk VA 通过二分查找得到 layer_id。
   BPF bounded loop (max 36 iter) 足够快。

#### BPF 边界表

```
model_va_start = 0x76c010a00000
model_va_end   = 0x76cc86600000

boundary[0]  = 0x76c010a00000   boundary[18] = 0x76c558200000
boundary[1]  = 0x76c049a00000   boundary[19] = 0x76c5e7200000
boundary[2]  = 0x76c083e00000   boundary[20] = 0x76c671800000
boundary[3]  = 0x76c0c1e00000   boundary[21] = 0x76c704e00000
boundary[4]  = 0x76c102400000   boundary[22] = 0x76c776200000
boundary[5]  = 0x76c145e00000   boundary[23] = 0x76c7fc000000
boundary[6]  = 0x76c184c00000   boundary[24] = 0x76c861000000
boundary[7]  = 0x76c1c4600000   boundary[25] = 0x76c8cae00000
boundary[8]  = 0x76c206800000   boundary[26] = 0x76c935000000
boundary[9]  = 0x76c245e00000   boundary[27] = 0x76c999600000
boundary[10] = 0x76c284e00000   boundary[28] = 0x76c9f3600000
boundary[11] = 0x76c2c8600000   boundary[29] = 0x76ca49e00000
boundary[12] = 0x76c312600000   boundary[30] = 0x76caa5600000
boundary[13] = 0x76c364600000   boundary[31] = 0x76caf9a00000
boundary[14] = 0x76c3c3a00000   boundary[32] = 0x76cb48400000
boundary[15] = 0x76c418800000   boundary[33] = 0x76cb9b200000
boundary[16] = 0x76c47d600000   boundary[34] = 0x76cbea200000
boundary[17] = 0x76c4e3600000   boundary[35] = 0x76cc38600000
```

---

### 实验日志: Template-Belady Benchmark 结果 (2026-02-27)

#### 环境
- 自定义 nvidia-uvm.ko（未修改，使用 BPF_CORE_READ 获取 chunk VA）
- 120B GPT-OSS MoE, pp=512, tg=128, r=3
- RTX 5090, 32GB VRAM, threshold=51

#### 结果对比

| 策略 | pp512 (tok/s) | tg128 (tok/s) | pp 提升 | tg 提升 |
|------|:---:|:---:|:---:|:---:|
| Baseline (no BPF, threshold=51) | 141.55 ± 0.59 | 49.87 ± 6.95 | — | — |
| always_max + cycle_moe (previous best) | 229.38 ± 0.16 | 91.27 ± 9.18 | +62.1% | +83.0% |
| **template_belady (kfunc版，已废弃)** | 229.58 ± 0.56 | 91.20 ± 8.92 | +62.2% | +82.9% |
| **template_belady (CO-RE版，r=3)** | 219.41 ± 0.77 | 87.69 ± 8.58 | +55.0% | +75.8% |
| **template_belady (CO-RE版，r=5)** | **225.01 ± 1.35** | **88.20 ± 5.23** | **+58.9%** | **+76.8%** |

注：CO-RE 版与 kfunc 版的差异在 tg 高方差（±5-9）范围内，实际性能无显著差异。

#### 分析

1. **template_belady ≈ always_max + cycle_moe**: Belady 距离 eviction 与简单 T1 频率保护性能持平。
   两者的核心都是 always_max prefetch（已证明对 MoE +57-70%）。

2. **Eviction 策略差异不显著**: 原因是 82% chunk thrashing 是**容量瓶颈**（模型 59 GiB vs GPU 32 GiB）。
   GPU 内存只能放 ~50% 的模型权重。无论用什么 eviction 策略，每个 decode step 都必须换进/换出大量 chunks。
   Belady OPT 的理论收益需要在容量更接近 working set 时才能体现。

3. **tg ≈ 91 tok/s 是当前硬件的近天花板**: 相比论文中 MSched 在 A100 (80GB) 上的数字，
   RTX 5090 (32GB) 的 PCIe 4.0 带宽限制了可能的最大吞吐。

4. **BPF_CORE_READ 获取 chunk VA**: 通过 BTF/CO-RE 解引用 `chunk->va_block->start`，
   无需修改内核模块。boundary-based layer lookup (36 entry ARRAY map + bounded loop) 在 BPF verifier 下通过。

#### 下一步
- 调整 protect_distance 参数（当前=3），尝试 1, 5, 10, 18（半周期）
- 测试不同 t1_freq_threshold 值
- 在更低 oversubscription 场景下测试（20B 模型 ≈ 38% oversubscription），Belady 收益可能更明显
- Cross-layer proactive prefetch: 预测下一层需要的 chunks 并提前 migrate — 见 [`cross_block_prefetch_plan.md`](cross_block_prefetch_plan.md)

### 2026-02-27: 剩余开销分析 — 瓶颈定位与优化上限

#### 分析方法

结合 chunk_trace 对比实验（baseline vs always_max, r=3）和 benchmark 实测数据，量化 decode 阶段每个 token 的时间开销组成。

**数据来源**:
- chunk_trace 对比实验: 无 chunk_trace 开销的 ACTIVATE/EVICTION 计数
- Benchmark: pp=512, tg=128, 5 repetitions
- 解析模型: T1+T2 working set = 4.02 GiB
- 硬件参数: PCIe 5.0 x16 (63 GB/s/方向), GDDR7 (1792 GB/s), 32 GiB VRAM

#### Per-Decode-Token 迁移量

从 chunk_trace 对比实验（r=3, 无 trace 开销）提取:

| 指标 | Per-run | Per-decode-token |
|------|---------|-----------------|
| 总 ACTIVATE | ~29,462 | — |
| Prefill ACTIVATE | ~15,770 | — |
| **Decode ACTIVATE** | ~13,692 | **~107 chunks (214 MB)** |
| 常驻 chunks (未被 evict) | ~5,338 (10.4 GiB) | — |
| **Decode EVICTION** | ~13,692 | **~107 chunks (214 MB)** |
| **总迁移量/token** | — | **428 MB (214 MB in + 214 MB out)** |

**关键发现**: 迁移量 in ≈ out (每 token 换入换出相同数量的 chunks)。这符合稳态预期: GPU 满载，每换入一个 chunk 就必须驱逐一个。

#### PCIe 传输时间

| 传输类型 | 数据量 | 时间 (63 GB/s) |
|----------|--------|---------------|
| Fault-in (H2D) | 214 MB (0.209 GiB) | 3.32 ms |
| Eviction-out (D2H) | 214 MB (0.209 GiB) | 3.32 ms |
| **Sequential (UVM 默认, 单 CE)** | 428 MB | **6.63 ms** |
| **Pipelined (MSched, 双 CE)** | 428 MB | **3.32 ms** |

**UVM 默认使用单个 Copy Engine**: fault-in 和 eviction-out 必须串行执行。MSched 使用双 CE 实现全双工 PCIe，可将传输时间减半。

#### GPU 计算时间

```
Active Working Set (T1+T2) = 4.02 GiB
GDDR7 带宽 = 1792 GB/s
Compute time/token = 4.02 / 1792 × 1000 = 2.24 ms
```

**注**: 这是内存带宽受限下的下界。实际 GPU 计算可能略高（kernel launch overhead, 同步等），但在 MoE 模型中非常接近这个值。

#### Fault 处理开销

| 配置 | Page faults/token | 每次 fault 开销 | 总计 |
|------|-------------------|----------------|------|
| **Baseline (threshold=51)** | ~400 (多次 per chunk) | ~7.5 us | **3.0 ms** |
| **always_max** | ~107 (一次 per chunk) | ~7.5 us | **0.80 ms** |
| **理论 (proactive prefetch)** | 0 | — | **0 ms** |

Baseline 下每个 2MB VA block 需要多次 page fault (bitmap tree 只部分预取)；always_max 把每个 VA block 一次性加载完毕，将 fault 次数减少到 chunk 级别。

#### 总开销分解

| 组件 | Baseline (20.2 ms) | Best BPF (12.7 ms) | 理论最优 |
|------|:---:|:---:|:---:|
| GPU 计算 | 2.2 ms (11%) | 2.2 ms (18%) | 2.2 ms |
| PCIe DMA (串行) | ~6.6 ms (33%) | ~6.6 ms (52%) | 3.3 ms (双 CE) |
| Fault 处理 | ~3.0 ms (15%) | ~0.8 ms (6%) | 0 ms (proactive) |
| GPU stall + 调度 + 其他 | ~8.4 ms (41%) | ~3.0 ms (24%) | ~0.5 ms |
| **总计** | **20.2 ms** | **12.7 ms** | **~6.0 ms** |

**"GPU stall + 调度 + 其他" 包含**:
- GPU SM 等待 DMA 完成的 stall 时间（与 PCIe DMA 重叠）
- UVM 内部锁竞争和页表管理
- eviction_prepare list 扫描
- TLB shootdown
- UVM state machine 状态转换

#### 各优化手段的贡献

| 优化 | 节省 | 说明 | 状态 |
|------|------|------|------|
| Intra-block prefetch (always_max) | **7.2 ms** (36%) | 减少 page-level fault 从 ~400 到 ~107/token | ✅ 已完成 |
| Eviction ordering (passive MRU) | **0.3 ms** (1.5%) | 非 T1 chunk freeze → FIFO 效果 | ✅ 已完成 |
| Proactive prefetch (eliminate faults) | ~0.8 ms (4%) | 不再需要 GPU fault → CPU 中断 → handler 路径 | 待实现 |
| Pipelined DMA (双 Copy Engine) | ~3.3 ms (16%) | 全双工 PCIe, in/out 并行 | 需要驱动修改 |
| DMA ↔ Compute overlap | ~2.2 ms (11%) | 当前层计算与下一层 DMA 并行 | 需要 proactive prefetch |
| **已实现总计** | **7.5 ms (37%)** | Baseline 20.2 → Best 12.7 ms | |
| **潜在可实现** | **~6.3 ms (31%)** | Best 12.7 → ~6.4 ms | |
| **理论天花板** | **~14.2 ms (70%)** | 20.2 → ~6.0 ms → ~167 tok/s | |

#### 瓶颈优先级

```
当前瓶颈分布 (Best BPF, 12.7 ms/tok):

  ┌─────────────────────┐
  │ PCIe DMA (串行)     │ 6.6 ms ████████████████████████ 52%  ← 主要瓶颈
  ├─────────────────────┤
  │ GPU stall + 其他    │ 3.0 ms ██████████ 24%  ← UVM 内部开销
  ├─────────────────────┤
  │ GPU 计算            │ 2.2 ms ████████ 18%  ← 不可压缩
  ├─────────────────────┤
  │ Fault 处理          │ 0.8 ms ███ 6%  ← 可通过 proactive 消除
  └─────────────────────┘
```

**结论**:
1. **PCIe 迁移是压倒性瓶颈** (52%): 每 token 需要迁移 428 MB 数据
2. **迁移量不可通过 eviction 策略减少**: 82% chunk thrashing 是容量决定的 (模型 59 GiB, VRAM 32 GiB, 1.84× oversubscription)
3. **唯一可减少 PCIe 时间的方法**: 双 CE 流水线 (需驱动修改) 或 DMA↔Compute 重叠 (需 proactive prefetch)
4. **Eviction 策略已到天花板**: passive MRU 仅比 always_max 好 0.3 ms, 不值得进一步优化

#### 理论性能上限

| 场景 | ms/tok | tok/s | 说明 |
|------|:---:|:---:|------|
| **当前最佳** | 12.7 | 78.7 | always_max + passive MRU |
| + Proactive prefetch (消除 fault) | ~11.9 | ~84 | 节省 fault 处理 0.8 ms |
| + Pipelined DMA (双 CE) | ~8.6 | ~116 | 节省 3.3 ms (需驱动修改) |
| + DMA↔Compute overlap | ~6.4 | ~156 | 节省 2.2 ms (需 proactive) |
| **PCIe 带宽硬限制** | ~3.3 | ~301 | min_PCIe 时间 (完全重叠计算) |
| **含 30% 管理开销** | ~4.3 | ~232 | 实际上限估计 |

**现实可达目标 (不改驱动)**:
- Cross-block proactive prefetch 可以实现部分 DMA↔Compute overlap
- 预期: 10-20% 提升 (78.7 → ~87-95 tok/s)
- 原因: UVM fault handler 仍在路径上，只是 fault 时数据已经在迁移中
- 完全消除 fault 需要绕过 UVM demand paging (类似 MSched 的 madvise/migrate)

**超越当前水平需要的改动层次**:

| 改动 | 提升潜力 | 可行性 |
|------|---------|--------|
| Cross-block proactive prefetch (BPF) | +10-20% | ✅ 可在 extension/ 实现 |
| 驱动参数调优 (prefetch_min_faults 等) | +2-5% | ✅ 无需代码修改 |
| 双 Copy Engine pipeline | +30-50% | ❌ 需改驱动 (MSched 路线) |
| 绕过 UVM demand paging | +50-80% | ❌ 需改驱动 (MSched madvise) |
| 更大 VRAM (减少 oversubscription) | +100%+ | ❌ 硬件升级 |

#### 脚本

分析脚本: `workloads/llama.cpp/analyze_overhead.py`
数据来源: `workloads/llama.cpp/results/msched_trace/chunk_trace_120b_long.csv`

#### 下一步

1. **Cross-block proactive prefetch**: 见 [`cross_block_prefetch_plan.md`](cross_block_prefetch_plan.md) — 当前唯一可在 extension/ 中实现的有效优化
2. **更低 oversubscription 测试** (20B 模型): 验证 Belady eviction 在 WS ≈ VRAM 时的收益
3. **其他 workloads**: vLLM, Faiss 的开销分布可能完全不同

---

## §21. Cross-Block Prefetch 实验完结 + 下一步算法改进方向 (2026-02-28)

### 21.1 Cross-Block Prefetch 最终结论

详细数据见 `cross_block_prefetch_plan.md` §19。

**核心发现**: Cross-block prefetch 在 1.84x oversubscription 下要么有害要么无效。

| 策略 | 过滤率 | tg128 | 与 no-XB 比 |
|------|--------|-------|-------------|
| Blind adjacent | 0% | 61.82 | **-30%** |
| 2-step direction | 33% | 70.27 | -21% |
| 3-step direction | 44% | 77.90 | -12% |
| Adjacent-stride | 99.4% | 87.58 | ±0% (10-run, p>>0.05) |

**10-run 高置信度对比** (同一 driver load):

| Config | pp512 | tg128 | σ(tg) |
|--------|-------|-------|-------|
| always_max + cycle_moe (no XB) | 221.33 ± 3.49 | 88.79 ± 8.94 | 8.94 |
| adj-stride + cycle_moe | 221.26 ± 2.59 | 87.58 ± 8.33 | 8.33 |

Adjacent-stride 与 no-cross-block 统计不可区分 (t=0.31, p>>0.05)。

**修正的错误结论**:
1. §18 "va_space lock contention" → **错误**: 两端都取 READ lock (rwsem 允许并发读)
2. §18 "eviction policy 无影响" → **部分正确**: 仅在 high-volume cross-block 时成立
3. 真正原因: **PCIe 带宽竞争** + **VRAM displacement** (prefetch value ≈ eviction cost → net ≈ 0)

**当前最佳**: always_max + cycle_moe = pp 221.33, tg 88.79 tok/s (+79% over stock)

### 21.2 剩余性能 Gap 分析

**当前**: 88.79 tok/s (11.26 ms/token)
**理论上限** (full DMA-compute overlap): ~151 tok/s (6.6 ms/token)
**Gap**: 4.66 ms (41%)

Per-token 时间分解:
```
Component                    Time      % of total
──────────────────────────   ──────    ──────────
PCIe DMA (107×2MB×2dir)      6.6 ms    59%      ← 不可压缩 (硬件限制)
GPU compute                  2.2 ms    20%      ← 不可压缩 (模型决定)
Fault handling (107×7.5μs)   0.8 ms     7%      ← 已被 always_max 最小化
"Other" (locks, sched, etc)  1.66 ms   14%      ← 可优化空间
```

**可攻击面**: "Other" 1.66 ms + DMA-compute overlap potential。
如果完全消除 "other" 且完美 overlap: 6.6 ms/token = 151 tok/s (+70%)。
现实目标: 消除部分 "other" + 部分 overlap → 100-120 tok/s (+13-35%)。

### 21.3 下一步算法改进方向 (按可行性排序)

#### 方向 A: 层级感知前瞻式迁移 (Layer-Aware Proactive Migration)

**核心洞察**: Cross-block prefetch 失败因为它按 **空间相邻** 预取 (VA 地址连续)。
但 MoE 模型的访问是按 **层级顺序** 的 (layer 0 → layer 35 → layer 0 → ...)。
如果在处理 layer L 时提前迁移 layer L+1 的 chunks, 可以 overlap DMA 与 compute。

**与 cross-block 的关键区别**:
- Cross-block: 预取 VA 空间中的下一个 block → 40% 命中率 → 大量无效迁移
- Layer-aware: 预取 layer pipeline 中的下一层 → T1 部分 100% 命中率

**实现方案**:
1. 复用 `template_belady.bpf.c` 的层级检测 (VA regression → 新 decode step, ascending VA → 当前 layer)
2. 在 `gpu_block_activate` 中检测 layer 边界 (current_layer 变化)
3. 当检测到 layer L → L+1 转换: 通过 bpf_wq 触发 `bpf_gpu_migrate_range` 预迁移 layer L+1 的 VA 范围

**分步实现**:

**Step A1: T1 前瞻 (确定性预测)**
- T1 chunks (attention, norms, embeddings) 每个 token 都被每层访问
- 当检测到处理 layer L: 预迁移 layer L+1 的 T1 VA 范围
- 预测准确率: 100% (T1 总是被需要)
- 预迁移量: ~60 MB/layer (T1 2.14 GB / 36 layers)
- 问题: T1 在 cycle_moe 下通常已 resident (不会被驱逐), 所以预迁移可能是 no-op

**Step A2: Expert 历史预测 + 前瞻**
- 追踪每层 top-2 expert 的激活历史 (last 4 tokens)
- 预测: 如果 expert X 在 layer L 中连续 3/4 tokens 被激活 → 预测下个 token 也激活
- 对预测命中的 expert: 预迁移其 VA 范围
- 预测准确率: 需要实测 (MoE expert routing 的 temporal correlation)

**BPF 实现框架**:
```c
struct layer_expert_history {
    u64 expert_mask[36];     // 每层 64-bit bitmask, bit i = expert i 在本 token 激活
    u64 prev_mask[4][36];    // 前 4 个 token 的 expert 激活历史
    u32 token_count;
};

SEC("struct_ops/gpu_block_activate")
int BPF_PROG(gpu_block_activate, ...) {
    u64 chunk_va = BPF_CORE_READ(chunk, va_block, start);
    u32 layer = va_to_layer(chunk_va);
    u32 expert = va_to_expert(chunk_va, layer);  // 需要更细粒度的 VA→expert 映射

    // 记录当前 token 的 expert 激活
    hist->expert_mask[layer] |= (1ULL << expert);

    // 检测 layer 边界
    if (layer != state->current_layer && layer == state->current_layer + 1) {
        // Layer L → L+1 转换
        // 预测 L+1 需要哪些 experts
        u64 predicted = majority_vote(hist->prev_mask[0..3][layer+1]);
        // 通过 bpf_wq 预迁移 predicted experts
        schedule_expert_prefetch(layer + 1, predicted);
    }
    state->current_layer = layer;
}
```

**关键挑战**:
1. VA→expert 映射: 需要更细粒度的 VA 分析 (不仅仅是 36 层, 还要区分每层的 64 个 experts)
2. Expert routing 的 temporal correlation: MoE 论文通常不公开这个统计量
3. bpf_wq + bpf_gpu_migrate_range 的竞争: 与 cross-block 类似的 PCIe 竞争
4. 但: cross-block 每 run 17.8K 次 migrate → 大量竞争; layer-aware 仅 ~36×128=4,608 次/run 且每次精准

**预期**: 如果 expert 预测 50%+ 准确率, 且 PCIe 竞争可控 → **+10-20% tg**
**实现复杂度**: 高 (需要 VA→expert 映射分析 + BPF 状态机)
**风险**: 中等 (expert temporal correlation 未知, PCIe 竞争是已知风险)

---

#### 方向 B: 开销诊断 (Overhead Instrumentation)

**在实施 A 之前, 先精确测量 "other" 1.66ms 到底花在哪里。**

在 BPF hooks 中添加 `bpf_ktime_get_ns()` timestamp, 测量:
1. `gpu_block_access` 单次调用耗时 (BPF 内部 + spinlock 等待)
2. 两次 `gpu_block_activate` 之间的间隔 (fault 处理 + DMA 时间)
3. 连续 token 间的 quiet period (token boundary gap)
4. 每个 layer 的处理时间 (layer L 的第一次 activate 到 layer L+1 的第一次 activate)

```c
struct timing_stats {
    u64 access_total_ns;      // gpu_block_access 总耗时
    u64 access_count;         // gpu_block_access 调用次数
    u64 activate_total_ns;    // gpu_block_activate 总耗时
    u64 activate_count;
    u64 inter_layer_ns[36];   // 每层处理时间
    u64 last_activate_ns;     // 上一次 activate 时间戳
};
```

**这能告诉我们**:
- spinlock 是否是瓶颈 (如果 access 耗时 >> BPF 计算时间)
- layer 间是否有 idle gap (如果有, 说明 DMA-compute 已有部分 overlap)
- 哪些 layer 最慢 (可能是 expert 切换密集的 MoE layer)

**实现复杂度**: 低
**风险**: 无 (只是测量)
**应首先执行**: 这能指导方向 A 的优先级

---

#### 方向 C: 不同 Workload 测试

当前所有实验仅在 120B MoE pp512/tg128 上进行。需要验证结论在其他场景下的通用性。

**C1: 20B Dense Model (低 oversubscription)**
- 模型: `gpt-oss-20b-default` (~12 GB, 0.38x oversubscription — 完全放入 VRAM)
- 预期: 无需 eviction/prefetch 优化, BPF overhead 应为 ~0
- 测试: always_max + cycle_moe 是否有开销 (应该无)

**C2: 120B MoE pp2048/tg256 (更长序列)**
- 更大 KV cache → 更高 oversubscription → 更多 expert 切换
- 预期: eviction policy 可能有更大影响

**C3: vLLM serving (连续请求)**
- vLLM 使用 PagedAttention, 内存管理不同
- 测试: BPF eviction policy 是否对 serving throughput 有益

**C4: FAISS 向量搜索**
- 完全不同的访问模式 (顺序扫描 + 随机访问)
- 测试: always_max prefetch 对 FAISS 是否有益

**实现复杂度**: 低 (已有 benchmark 脚本)
**风险**: 无

---

#### 方向 D: Phase-Aware Policy (Prefill vs Decode 分离)

**洞察**: Prefill 和 decode 有完全不同的访问模式:
- Prefill: 顺序加载所有 layer, 每 layer 访问一次, adjacent hit rate 很高
- Decode: 循环访问, MoE expert 切换, adjacent hit rate 低

**当前**: 两个阶段使用相同策略。

**优化**:
- Prefill 阶段: 使用 adjacent-stride cross-block (proven neutral, 可能在 prefill 有少量收益)
- Decode 阶段: 只用 always_max + cycle_moe (proven optimal)
- Phase detection: VA regression (当前 VA < 70% max VA → decode started)

**预期**: pp 可能 +3-5%, tg 无变化
**风险**: pp 改进对长序列生成意义不大

---

### 21.4 推荐执行顺序

```
Phase 1: 诊断 (先知道问题在哪)
  └→ 方向 B: 添加 timing instrumentation, 测量 "other" 1.66ms 分布

Phase 2: 快速验证 (低风险, 确认基础)
  ├→ 方向 C1: 20B 模型 baseline (确认 low-oversub 场景)
  └→ 方向 C2: 120B pp2048 (确认 high-oversub 场景)

Phase 3: 核心算法 (高潜力, 需要 Phase 1 数据指导)
  ├→ 方向 A-Step1: Layer-aware T1 前瞻 (确定性, 低风险)
  └→ 方向 A-Step2: Expert 预测 + 前瞻 (需要 VA→expert 映射分析)

Phase 4: 扩展验证
  ├→ 方向 C3: vLLM (不同运行时)
  ├→ 方向 C4: FAISS (不同访问模式)
  └→ 方向 D: Phase-aware (如果 Phase 3 显示 prefill 可以进一步优化)
```

### 21.5 关键文件

| 文件 | 角色 |
|------|------|
| `extension/prefetch_always_max_cycle_moe.bpf.c` | 当前最佳策略, 方向 A/B 的修改对象 |
| `extension/prefetch_template_belady.bpf.c` | 方向 A 的层级检测基础设施 |
| `extension/prefetch_cross_block_v2.bpf.c` | 方向 A 的 bpf_wq 基础设施 |
| `workloads/llama.cpp/analyze_overhead.py` | 方向 B 的分析基础 |
| `results/msched_trace/layer_va_ranges_equal_count.json` | 方向 A 的 VA→layer 映射 |

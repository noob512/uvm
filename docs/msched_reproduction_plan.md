# MSched 算法复现与研究计划

**日期**: 2026-02-25
**论文**: "Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling" (arxiv 2512.24637)
**完整论文**: `docs/reference/msched_paper/html/2512.24637v2/index.html`
**状态**: 进行中

> **整理说明**: 本文档于 2026-02-28 从原始文档 (2122 行) 压缩整理为当前版本 (~420 行)，去除重复内容但保留所有实验数据和失败记录。
> 原始完整版存档于 [`docs/reference/msched_reproduction_plan_old.md`](reference/msched_reproduction_plan_old.md)，仅作为历史参考。
> **后续所有新实验和进展请直接 append 到本文档末尾。**

---

## ⚠️ 约束：不修改自定义内核模块

**硬性要求**：所有新功能必须在 BPF 用户态程序（`extension/`）中实现，**不能修改** `kernel-module/nvidia-module/` 下的自定义 nvidia-uvm 内核模块代码。

**已有 kfunc（可直接使用）**：
- `bpf_gpu_set_prefetch_region()` — 设置 prefetch 区域
- `bpf_gpu_block_move_head()` / `bpf_gpu_block_move_tail()` — eviction list 操作
- `bpf_gpu_strstr()` — 字符串匹配
- `bpf_gpu_migrate_range(va_space, addr, length)` — **sleepable**: 从 bpf_wq callback 调用

**读取 chunk 信息**：BTF/CO-RE（`BPF_CORE_READ`）直接解引用，无需新 kfunc：
```c
uvm_va_block_t *va_block = BPF_CORE_READ(chunk, va_block);
u64 chunk_va = BPF_CORE_READ(va_block, start);
// CO-RE 偏移: chunk->va_block=32, va_block->start=48
```

**获取 va_block/va_space 上下文**：kprobe + per-CPU map（不修改 hook 签名），参考 `extension/prefetch_trace.bpf.c`。

---

## 1. MSched 算法与单应用 UVM 的适用性

MSched 的两个核心算法——**template-based working set prediction** 和 **Belady OPT eviction**——本质上解决 "UVM demand paging 不知道每个 GPU kernel 实际需要哪些页" 的问题。在**单应用 oversubscription** 下同样存在。

以 llama.cpp 120B MoE (59 GiB) 在 RTX 5090 (32GB) 上 decode 为例：每个 decode step ≈ 160 个 GPU kernel，总分配 59 GiB 但每个 kernel 实际 WS 仅 16KB–120MB。Default UVM 按 spatial locality 猜 → 猜错连锁 thrashing。

**MSched 论文证据**: Table 1 — llama.cpp allocation-granularity 预测 FP rate 99.7%，template 预测 0%。Fig 8 — 300% oversub 时 allocation-granularity 比 template 多 12.27× 迁移量。

### MSched 三个算法

| 算法 | 方法 | gpu_ext 可行性 |
|------|------|---------------|
| Template WS Prediction | NVBit 离线 profiling → T1/T2/T3 分类 → cuLaunchKernel 拦截 | ✅ 用解析式+chunk_trace 替代 NVBit |
| Belady OPT Eviction | 已知 kernel 序列 → 淘汰下次访问最远的页 | ✅ BPF struct_ops 实现 |
| Pipelined Migration | 双 Copy Engine 全双工 PCIe | ❌ 需修改驱动 |

---

## 2. 实验时间线与结果

### 2.1 Phase 0: Working Set 分析 (2026-02-25)

#### NVBit 尝试与放弃

NVBit v1.7.7.1 编译成功（`/home/yunwei37/workspace/gpu/NVBit/nvbit_release_x86_64/`），编写了 `ws_trace.so` 和分析脚本 `scripts/analyze_ws.py`。

**首次尝试** (02-25): 超时 15 分钟，初始化阶段 CPU 200%，判断为不可行。

**重试** (02-27, timeout 7 小时):

| 时间 (min) | GPU 显存 (MiB) | 状态 |
|-----------|---------------|------|
| 44 | 2,983 | binary instrumentation 开始 |
| 120 | 7,487 | 稳定加载 |
| 270 | 23,363 | 加速到 ~1.8 GiB/10min |
| ~420 | ~23,363 | **被 timeout SIGKILL** |

**结果**: ws_trace_120b.txt = 0 字节（ws_trace 仅在正常退出时 flush）。NVBit 技术可用但极慢：120B 模型预计 12-24 小时，不实际。

**根本原因**: NVBit 对每条内存指令做 binary instrumentation，120B MoE ~160+ kernel template × instrument → 执行速度降 ~100x，叠加 UVM demand paging 开销。

**最终决策**: 放弃 NVBit，改用解析式方法 + chunk_trace。

#### 解析式 Per-Layer Working Set

脚本: `NVBit/scripts/analytical_ws.py`

**120B 模型 MSched Template 分类**:

| Template | 类别 | 大小 | 占总模型 |
|----------|------|------|---------|
| **T1 (Fixed)** | 嵌入层 + 注意力 + Router | **2.14 GB** | 3.4% |
| **T2 (Active Experts)** | 4/128 experts × 36 layers | **1.88 GB** | 3.0% |
| **T3 (Inactive)** | 124/128 experts × 36 layers | **58.33 GB** | 93.6% |
| **Ideal WS (T1+T2)** | — | **4.02 GB** | 6.4% |

理想 WS 仅 4.02 GB (VRAM 的 12.6%)，但 default UVM 实际浪费迁移 ~83 GB（差距 ~44×）。

#### chunk_trace 基线测量

**配置**: 120B MXFP4 MoE, RTX 5090 (32GB), raw UVM, pp=512, tg=128

| 指标 | 值 |
|------|------|
| 总事件数 | 358,445 |
| ACTIVATE (page fault) | 41,825 |
| EVICTION_PREPARE | 25,968 |
| 唯一 2MB chunks | 20,531 (40.1 GB VA) |
| **Re-fault rate** | **51%** (21,294/41,825) |
| **Chunk thrashing rate** | **82%** (16,813/20,531) |
| 浪费迁移带宽 | ~83.2 GB |

**Per-decode-step 分析** (177 steps detected):
- 平均 333 MB/step fault-in，其中 273 MB 是 re-fault
- 100% ascending VA pattern → 证实 layer 序列化访问
- 3,718 chunks 仅 fault 1 次 (T1 候选)；16,813 chunks 多次 fault
- 34% experts 从未被路由

**Raw UVM 性能**: pp=512: 143.9 tok/s, tg=128: 49.4 tok/s

---

### 2.2 Phase 1: Eviction Policy 实验 (2026-02-25)

#### cycle_moe eviction: `extension/eviction_cycle_moe.bpf.c`

**设计**: 最小干预 — T1 频率检测 (access≥3 → `move_tail` 保护)，其余 `return 0` 让内核默认。

**关键技术发现 (BPF 安全性 bugs)**:

1. **`move_head` 在 `chunk_activate` 中不安全** → Xid 31: 新 chunk 被 activate 后立即移到 HEAD，eviction 线程可能在 page table setup 前就 evict。**解决**: `move_tail` 或 `return 0`。

2. **BPF Hash Map 在 fault handler 热路径不可用** → Xid 31: bucket lock + hash 延迟导致 GPU fault timeout。**解决**: `BPF_MAP_TYPE_PERCPU_ARRAY`。

3. **BPF verifier 禁止 pointer arithmetic**: **解决**: `bpf_probe_read_kernel` 转为 scalar。

**性能评测** (120B, RTX 5090):

| 策略 | pp=512 | tg=128 |
|------|--------|--------|
| Baseline (无策略) | 145.2 | 50.6 |
| cycle_moe v2 | 145.2 | 50.9 |
| MRU | 18.97 | 9.62 |
| LFU | ❌ Xid 31 | ❌ |

**结论**: 默认 LRU 对 MoE 已足够好，cycle_moe 零开销安全网。MRU 灾难 (-83%)，LFU 崩溃。**优化空间在 prefetch 侧**。

---

### 2.3 Phase 2: Prefetch 策略全面评测 (2026-02-26)

**配置**: 120B MXFP4 MoE, RTX 5090 (32GB), UVM, 5 repetitions

**短序列 (pp=512, tg=128)**:

| 策略 | pp512 (tok/s) | tg128 (tok/s) | pp 提升 | tg 提升 |
|------|:---:|:---:|:---:|:---:|
| Baseline (无 BPF, threshold=51) | 139.50 ± 1.85 | 45.29 ± 3.34 | — | — |
| **always_max** | 219.12 ± 2.81 | 76.85 ± 4.11 | **+57.1%** | **+69.7%** |
| always_max + cycle_moe | 224.25 ± 1.31 | 76.87 ± 4.35 | +60.8% | +69.7% |
| stride (conf=2, pages=4) | 33.17 ± 0.30 | 14.46 ± 1.51 | -76.2% | -68.1% |
| none (禁用) | 31.58 ± 0.29 | 14.01 ± 1.57 | -77.4% | -69.1% |

**长序列 (pp=2048, tg=512)**: always_max: pp=231.12 (+57.1%), tg=85.15 (+63.1%)

**关键发现**:

1. **always_max = 最大单一优化 (+57-70%)**。默认 UVM bitmap tree threshold=51% 对 MoE 严重次优。
2. **stride 灾难 ≈ 禁用预取**: BYPASS 语义导致 92% fault 跳过默认预取。教训：不确定时返回 DEFAULT (0)。
3. **cycle_moe 在 always_max 之上无额外收益** (< 2%)。
4. **收益与序列长度无关**: 来自减少 per-VA-block page fault 数量。

#### chunk_trace 对比: Baseline vs always_max

| 指标 | Baseline | always_max |
|------|---------|-----------|
| ACTIVATE (chunk fault) | 88,387 | 88,455 (~0%) |
| EVICTION_PREPARE | 72,372 | 72,451 (~0%) |
| Re-fault rate | 82.2% | 82.2% |

**结论**: always_max **不减少** chunk 级别 thrashing (82% 不变)。性能提升来自 intra-chunk page-level 优化：每个 2MB VA block 从多次 fault 变为一次。

#### UVM Prefetch Threshold Root Cause

**源码**: `uvm_perf_prefetch.c` — bitmap tree 算法，`threshold=51` = "严格过半"规则。

NVIDIA 未实现的 TODO: `Bug 1778037: [uvm] Use adaptive threshold for page prefetching`

**Threshold Sweep** (仅模块参数，无 BPF):

| threshold | pp512 | tg128 | tg 提升 |
|-----------|:---:|:---:|:---:|
| 51 (默认) | 139.50 | 45.29 | — |
| 25 | 176.06 | 56.89 | +25.6% |
| 10 | 202.21 | 72.64 | +60.4% |
| 5 | 208.39 | 76.45 | +68.8% |
| 1 | 217.12 | 76.00 | +67.8% |
| BPF always_max | 219.12 | 76.85 | +69.7% |

严格单调：threshold 越低性能越高。always_max 略优于 threshold=1（跳过 tree 遍历开销）。

**BPF 的核心价值**: 运行时动态覆盖 threshold — MoE 用 BYPASS+always_max，dense 用 DEFAULT，多租户不同 PID 不同策略，无需重启驱动。

#### Combined Prefetch + Eviction 策略

| 策略 | pp512 | tg128 | tg 提升 |
|------|:---:|:---:|:---:|
| always_max only | 219.12 | 76.85 | +69.7% |
| + cycle_moe (T1 protect) | 224.25 | 76.87 | +69.7% |
| + MRU expert (T1 protect, move_head) | 221.89 | 76.18 | +68.2% |
| **+ passive MRU (T1 protect, freeze)** | **227.94** | **78.68** | **+73.7%** |

Passive MRU 原理: 非 T1 chunk 返回 BYPASS 但不 move → 阻止 LRU 刷新 → FIFO 效果。
长序列 (pp=2048/tg=512) 差异消失 (~85 tok/s)。

---

### 2.4 Layer Mapping + Template-Belady (2026-02-27)

#### chunk_trace → VA→Layer 映射

尝试了 6 种方法，前 5 种失败：

| 方法 | 结果 | 失败原因 |
|------|------|---------|
| Gap-based (4MB gap) | 1478 "层" | MoE 专家间 VA 间隙过度分割 |
| VA 回退检测 | prefill 仅 1 step | 信息不足 |
| 滑动窗口中位数 | 329 区域 | eviction re-fault 打乱顺序 |
| 时间等分 | 层 0-10 有问题 | 早期 prefill 散射 |
| 线性 VA 等分 | 层 0-15 集中 | VA 空间稀疏 (117 GiB span, 31 GiB 活跃) |
| **等数量 VA 分割** ✅ | 36 层 × ~439 chunks | 验证通过: 60 decode steps 循环 |

**等数量分割**: 过滤 prefill chunks (t=100-3000ms)，排除 66 GiB VA 空洞后剩余 15,801 chunks，按 VA 排序等分 36 组。Decode 验证: 层访问频率均匀 (427-788/层)，序列正确 (35→0→1→...→35)。

**输出**: `results/msched_trace/layer_va_ranges_equal_count.json`

#### Template-Belady Benchmark

**文件**: `extension/prefetch_template_belady.bpf.c` — always_max prefetch + VA-based layer detection + Belady distance eviction

| 策略 | pp512 | tg128 | tg 提升 |
|------|:---:|:---:|:---:|
| Baseline (threshold=51) | 141.55 ± 0.59 | 49.87 ± 6.95 | — |
| always_max + cycle_moe | 229.38 ± 0.16 | 91.27 ± 9.18 | +83.0% |
| template_belady (CO-RE, r=5) | 225.01 ± 1.35 | 88.20 ± 5.23 | +76.8% |

**结论**: template_belady ≈ always_max + cycle_moe。Belady eviction 与简单 T1 保护性能持平 — 82% chunk thrashing 是**容量瓶颈** (59 GiB vs 32 GiB)，eviction 策略无法减少迁移总量。

---

### 2.5 开销分析 (2026-02-27)

**Per-decode-token 迁移量** (from chunk_trace): ~107 chunks in + ~107 chunks out = **428 MB/token**

**Per-token 时间分解** (Best BPF, 10-run: 88.79 tok/s = 11.26 ms/tok):

```
Component                    Time      % of total
──────────────────────────   ──────    ──────────
PCIe DMA (107×2MB×2dir)      6.6 ms    59%      ← 不可压缩 (硬件限制)
GPU compute                  2.2 ms    20%      ← 不可压缩 (模型决定)
Fault handling (107×7.5μs)   0.8 ms     7%      ← 已被 always_max 最小化
"Other" (locks, sched, etc)  1.66 ms   14%      ← 可优化空间
```

**各优化手段贡献**:

| 优化 | 节省 | 状态 |
|------|------|------|
| Intra-block prefetch (always_max) | 7.2 ms (36%) | ✅ 已完成 |
| Eviction ordering (passive MRU) | 0.3 ms (1.5%) | ✅ 已完成 |
| Proactive prefetch (eliminate faults) | ~0.8 ms (4%) | 待实现 |
| Pipelined DMA (双 CE) | ~3.3 ms (16%) | ❌ 需改驱动 |
| DMA↔Compute overlap | ~2.2 ms (11%) | 需 proactive prefetch |

**理论性能上限**:

| 场景 | tok/s | 说明 |
|------|:---:|------|
| **当前最佳** | 88.79 | always_max + cycle_moe (10-run) |
| + proactive prefetch | ~84 | 节省 fault 0.8 ms |
| + pipelined DMA | ~116 | ❌ 需驱动修改 |
| + full overlap | ~156 | 理论天花板 |

分析脚本: `workloads/llama.cpp/analyze_overhead.py`

---

### 2.6 Cross-Block Prefetch 实验 (2026-02-27~28)

详细数据见 [`cross_block_prefetch_plan.md`](cross_block_prefetch_plan.md) §19。

**核心发现**: Cross-block prefetch 在 1.84x oversubscription 下**要么有害要么无效**。

| 策略 | 过滤率 | tg128 | 与 no-XB 比 |
|------|--------|:---:|-------------|
| Blind adjacent | 0% | 61.82 | **-30%** |
| 2-step direction | 33% | 70.27 | -21% |
| 3-step direction | 44% | 77.90 | -12% |
| Adjacent-stride | 99.4% | 87.58 | ±0% (10-run, p>>0.05) |

**10-run 高置信度** (同一 driver load):

| Config | pp512 | tg128 | σ(tg) |
|--------|:---:|:---:|:---:|
| always_max + cycle_moe (no XB) | 221.33 ± 3.49 | 88.79 ± 8.94 | 8.94 |
| adj-stride + cycle_moe | 221.26 ± 2.59 | 87.58 ± 8.33 | 8.33 |

Adjacent-stride 统计不可区分 (t=0.31, p>>0.05)。

**修正的错误结论**: §18 "lock contention" 是**错误**的 — 两端都取 READ lock (rwsem 允许并发读)。真正原因: **PCIe 带宽竞争** + **VRAM displacement** (prefetch value ≈ eviction cost → net ≈ 0)。

---

## 3. 最终性能总表

| 策略 | pp512 | tg128 | tg512 | 说明 |
|------|:---:|:---:|:---:|------|
| Raw UVM (threshold=51) | 139.5 | 45.3 | 52.2 | 内核默认 |
| threshold=1 (仅参数) | 217.1 | 76.0 | — | 不需要 BPF |
| BPF always_max | 219.1 | 76.9 | 85.2 | 仅 prefetch |
| BPF passive MRU | 228.0 | 78.7 | 85.1 | always_max + passive MRU |
| **BPF always_max + cycle_moe** | **221.3** | **88.8** | — | **最佳 (10-run)** |
| template_belady (CO-RE) | 225.0 | 88.2 | — | Belady ≈ cycle_moe |
| stride prefetch | 33.2 | 14.5 | — | ❌ 灾难 |
| MRU (纯) | 19.0 | 9.6 | — | ❌ 灾难 |
| gpu_ext 论文 (参考) | — | 86.89 | — | stride + LFU (旧驱动) |

---

## 4. 核心结论

1. **UVM prefetch threshold=51 是 MoE 模型的主要性能瓶颈**: BPF always_max 覆盖后 +57-70%，等效于 NVIDIA 6 年未实现的 Bug 1778037。

2. **Eviction 策略在高 oversubscription (1.84x) 下到天花板**: 82% chunk thrashing 是容量决定的，Belady OPT ≈ 简单 T1 保护 ≈ 默认 LRU，差异仅 0.3 ms/tok。


3. **Cross-block prefetch 在高 oversubscription (1.84x) + 循环访问下无效/有害**: prefetch value ≈ eviction cost (PCIe 零和)。中等 oversub + 线性访问模式下可能有效（见 cross_block_prefetch_plan §15.4）。

4. **PCIe DMA 是压倒性瓶颈** (59%): 428 MB/token 迁移量不可通过软件减少，需要双 CE pipeline (驱动修改) 或更大 VRAM。

5. **BPF 的核心价值**: 运行时策略定制 — 不同 workload 不同策略，无需重启驱动、无需改应用。

---

## 5. 未完成工作与下一步

### 5.1 未实现但已设计的方案

**自适应 Threshold (P1)**: 已有 `extension/prefetch_adaptive_tree_iter.bpf.c` 和 `extension/prefetch_adaptive_sequential.bpf.c`，**未 benchmark**。三种方案：Per-VA-Region threshold、Feedback-Driven、Workload Auto-Detect。ENTER_LOOP hook 限制: 无法提前终止遍历、无累积状态传递。

**Cross-Layer BPF Pipeline (P2)**: uprobe (cuLaunchKernel) + struct_ops 通过共享 BPF map 协作。已有 `cuda_sched_trace.bpf.c` 基础。VA→layer 映射已完成 (equal-count 方法)。待实现: uprobe 写 kernel_state_map → struct_ops 读取。

**VA→Layer 替代方案优先级**:

| 方案 | 状态 |
|------|------|
| chunk_trace → 等数量 VA 分割 | ✅ 完成 |
| BPF 运行时自学习 | 部分实现 (template_belady.bpf.c) |
| GGUF 解析 + uprobe base_va | 待实现 |
| ~~NVBit 离线 profiling~~ | ❌ 放弃 |

### 5.2 下一步算法改进方向 (按可行性排序)

#### 方向 A: 层级感知前瞻式迁移 (Layer-Aware Proactive Migration)

Cross-block 失败因为按**空间相邻**预取。层级感知按**层序列**预取 → T1 部分 100% 命中率。

**实现**: 复用 `template_belady.bpf.c` 层级检测，在 `gpu_block_activate` 检测 layer 边界 → bpf_wq 触发 `bpf_gpu_migrate_range` 预迁移 layer L+1。

- **Step A1 (T1 前瞻)**: 确定性预测，~60 MB/layer。但 T1 通常已 resident (may be no-op)。
- **Step A2 (Expert 历史预测)**: 追踪 top-2 expert 激活历史，majority vote 预测。需 VA→expert 映射。

**预期**: +10-20% tg (如果 expert 预测 50%+ 且 PCIe 竞争可控)
**风险**: PCIe 竞争 (cross-block 已证明), expert temporal correlation 未知

#### 方向 B: 开销诊断 (Overhead Instrumentation)

**先测量 "other" 1.66ms 的具体分布**, 再决定方向 A 优先级。

在 BPF hooks 加 `bpf_ktime_get_ns()` 测量: `gpu_block_access` 单次耗时、两次 activate 间隔、per-layer 处理时间、token boundary gap。

**实现复杂度**: 低。**风险**: 无。**应首先执行**。

#### 方向 C: 不同 Workload 测试

| 场景 | 目的 |
|------|------|
| 20B Dense (~12 GB, 完全放入 VRAM) | 验证低 oversub 下 BPF 零开销 |
| 120B MoE pp2048/tg256 | 验证高 oversub 下 eviction 影响 |
| vLLM serving | 不同内存管理 (PagedAttention) |
| FAISS SIFT-100M | 不同访问模式 (顺序+随机) |

#### 方向 D: Phase-Aware Policy (Prefill vs Decode 分离)

Prefill (顺序, high locality) vs Decode (循环, MoE 切换) 用不同策略。预期: pp +3-5%。

### 5.3 推荐执行顺序

```
Phase 1: 诊断
  └→ 方向 B: timing instrumentation, 测量 "other" 1.66ms 分布

Phase 2: 快速验证
  ├→ 方向 C1: 20B 模型 baseline
  └→ 方向 C2: 120B pp2048

Phase 3: 核心算法
  ├→ 方向 A-Step1: Layer-aware T1 前瞻
  └→ 方向 A-Step2: Expert 预测 + 前瞻

Phase 4: 扩展验证
  ├→ 方向 C3: vLLM
  ├→ 方向 C4: FAISS
  └→ 方向 D: Phase-aware
```

### 5.4 关键文件

| 文件 | 角色 |
|------|------|
| `extension/prefetch_always_max_cycle_moe.bpf.c` | 当前最佳策略 |
| `extension/prefetch_template_belady.bpf.c` | 层级检测基础设施 |
| `extension/prefetch_cross_block_v2.bpf.c` | bpf_wq 基础设施 |
| `workloads/llama.cpp/analyze_overhead.py` | 开销分析 |
| `results/msched_trace/layer_va_ranges_equal_count.json` | VA→layer 映射 |

---

## 6. 实验记录：Proactive Layer Migration (2026-02-27)

### 6.1 算法思路

**核心创新**: 在 GPU 处理 layer L 时，通过 bpf_wq 提前异步迁移 layer L+1 的数据到 GPU VRAM，使 DMA 传输与 GPU 计算重叠。

**与 cross-block prefetch 的区别**:
- Cross-block: 空间相邻 VA block（盲猜），已证明无效/有害
- Proactive layer: 语义级预取，利用 MoE 模型 layer 访问的确定性顺序 (0→1→...→N-1→0→...)

**实现组件**:
1. always_max prefetch (已验证 +57-70%)
2. cycle_moe T1 eviction (已验证安全)
3. O(1) layer transition detection (VA boundary check，无循环—BPF verifier 安全)
4. bpf_wq 异步预迁移 (kprobe 获取 va_space 上下文)

**BPF verifier 问题**: 最初用 `for (i = 0; i < MAX_LAYERS; i++)` 循环检测当前 layer，被 verifier 拒绝（"infinite loop detected"）。改为 O(1) 只检查 `layer_boundaries[current_layer + 1]`——因 layer 顺序访问，+1 检查即足。

### 6.2 文件

| 文件 | 说明 |
|------|------|
| `extension/prefetch_proactive_layer.bpf.c` | BPF 内核程序 (always_max + cycle_moe + proactive migration) |
| `extension/prefetch_proactive_layer.c` | 用户态 loader (CLI: `--profile`, `--layers`, `--prefetch-kb`, `--ahead`) |

### 6.3 Benchmark 结果 (llama.cpp 120B MoE, pp512/tg128, RTX 5090)

| 配置 | Runs | pp (tok/s) | tg (tok/s) | vs baseline |
|------|------|------------|------------|-------------|
| **Baseline** (always_max + cycle_moe) | 5 | 226.90 ± 2.36 | 89.45 ± 1.49 | — |
| **Proactive 4MB ahead=1** | 5 | 227.67 ± 0.67 | 89.96 ± 0.30 | +0.3% pp, +0.6% tg |
| **Proactive 8MB ahead=1** | 2 | 228.19 ± 0.31 | 90.90 ± 0.88 | +0.6% pp, +1.6% tg |
| Proactive 16MB ahead=2 | — | (未完成) | (未完成) | — |

### 6.4 分析

**结论: Proactive layer migration ≈ 基线性能，无统计显著差异。**

原因分析（与 cross-block prefetch 结论一致）:
1. **PCIe 带宽竞争**: 在 1.84x oversubscription 下，proactive migration 使用的 DMA 带宽与 demand paging 共享同一 PCIe 通道。预取的数据传输占用了本可用于当前 fault 的带宽。
2. **VRAM 位移**: 预取下一层数据必然挤出当前 VRAM 中的其他数据。在 WS > VRAM 场景下，这创建了"零和博弈"。
3. **UVM fault handling 已高度优化**: NVIDIA 的 demand paging 在检测到 fault 后立即批量迁移整个 VA block (2MB)，留给 proactive 的增量空间很小。

**关键洞察**: 在 PCIe 为瓶颈的场景下（占 per-token 时间的 59%），任何纯软件的 prefetch/eviction 策略都无法突破硬件带宽限制。真正有效的改进需要：
- 减少总迁移量（但 MoE 模型每 token 固有 ~428 MB 迁移需求）
- PCIe DMA 与 GPU 计算的流水线重叠（需驱动级修改，即 MSched 的 pipelined migration）
- 更高的 PCIe 带宽（硬件升级）

---

## 7. TODO: 后续实验方向

### 7.1 其他 workload 测试（优先级高）

不同 workload 有不同的内存访问模式，可能对 BPF 策略有不同响应:

| Workload | 预期访问模式 | 可能有效的策略 |
|----------|-------------|---------------|
| **FAISS SIFT-100M** | 大规模顺序扫描 + 随机 probe | 顺序 prefetch + LFU eviction |
| **GNN Training** (PyTorch Geometric) | 图遍历，随机性高 | 工作集估计 + 频率感知 eviction |
| **vLLM Serving** | PagedAttention，KV cache | phase-aware (prefill vs decode) |

### 7.2 新算法方向（纯 BPF，无需改 kernel 模块）

1. **Phase-Aware Policy**: 检测 prefill vs decode phase，分别使用不同 prefetch/eviction 策略
   - Prefill: 高 locality，激进 prefetch
   - Decode: MoE 循环，保守 prefetch + T1 保护

2. **Expert Prediction**: 利用 MoE 模型 top-k routing 的可预测性
   - 统计 expert 访问频率，预测下一步最可能的 expert
   - 需要先用 chunk_trace 分析 expert-level 访问模式

3. **Adaptive Threshold**: 已有 `prefetch_adaptive_tree_iter` 和 `prefetch_adaptive_sequential`，未 benchmark
   - 这两个策略使用 NVML 监控 GPU 利用率，动态调节 prefetch 阈值

4. **Working Set Compression**: 减少有效迁移量
   - 识别 "冷" chunk（长期不访问），优先 evict
   - 识别 "热" chunk（每 token 必访问），永不 evict

### 7.3 现有未测策略

| 策略 | 文件 | 状态 |
|------|------|------|
| `prefetch_adaptive_tree_iter` | `extension/prefetch_adaptive_tree_iter.bpf.c` | 已编译，未 benchmark |
| `prefetch_adaptive_sequential` | `extension/prefetch_adaptive_sequential.bpf.c` | 已编译，未 benchmark |
| `prefetch_stride` | `extension/prefetch_stride.bpf.c` | 已编译，未 benchmark |
| `eviction_lfu` | `extension/eviction_lfu.bpf.c` | 已编译，未 benchmark |
| `eviction_lfu_xcoord` | `extension/eviction_lfu_xcoord.bpf.c` | 已编译，未 benchmark |

### 7.4 llama.cpp 120B 完整性能对比表

| 策略 | pp (tok/s) | tg (tok/s) | 日期 | 备注 |
|------|-----------|-----------|------|------|
| Stock UVM (threshold=51) | 141.6 | 49.9 | 02-25 | 默认 NVIDIA 行为 |
| always_max (threshold=1) | 219.1 | 76.9 | 02-25 | BPF 或 module param |
| always_max + cycle_moe | 221.3 | 88.8 | 02-26 | 10-run mean |
| template_belady | 225.0 | 88.2 | 02-26 | always_max + Belady eviction (§2.4 5-run) |
| **Proactive layer 4MB** | 227.7 | 90.0 | 02-27 | ≈ baseline (PCIe bottleneck) |
| **Proactive layer 8MB** | 228.2 | 90.9 | 02-27 | ≈ baseline (2 runs only) |
| cross-block blind | — | -28.5% | 02-27 | 有害 (VRAM 位移) |
| cross-block adj-stride | — | ≈ baseline | 02-27 | 中性 (10-run 无显著差异) |

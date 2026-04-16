# MSched: Proactive Memory Scheduling for GPU Multitasking

**Paper**: "Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"
**Source**: https://arxiv.org/html/2512.24637
**Date analyzed**: 2026-02-24

---

## 1. Paper Summary

MSched is an OS-level GPU memory scheduler that replaces demand paging with **proactive working set preparation** during GPU context switches. It exploits the predictability of GPU memory access patterns to achieve Belady's optimal page replacement, delivering 11-58x speedup over CUDA Unified Memory under memory oversubscription.

### Core Claim

GPU memory management should transition from **reactive** (fault-driven) to **proactive** (prediction-driven). By co-designing the task scheduler and memory manager, MSched reconstructs a global memory access timeline and applies the optimal eviction policy (Belady's OPT).

---

## 2. Technical Design

### 2.1 Memory Access Prediction (Offline Profiling + Online Prediction)

Three templates covering ~99% of GPU kernels:

| Template | Frequency | Pattern | Example |
|----------|-----------|---------|---------|
| T1 (Fixed) | ~77% | Fixed regions per kernel | Invariant buffers |
| T2 (Linear) | ~18% | Size scales linearly with launch args | `vector_add(A, B, C, N)` → buffer size ∝ N |
| T3 (Strided) | ~5% | Discontiguous chunks, regular stride | Batched matrix ops |

**Offline**: NVBit instrumentation profiles kernel memory access → derives formulas mapping launch args to memory regions.
**Online**: Intercepts kernel launch API → evaluates formulas at μs cost.
**Accuracy**: 0.25% false negative, 0% false positive.

### 2.2 Belady OPT via Task Timeline

1. Scheduler exposes **task scheduling timeline** (ordered tasks + timeslice allocation)
2. During context switch, memory manager iterates timeline in **reverse order**
3. Each task's predicted working set pages are `madvise`'d to eviction list tail
4. Result: list head = optimal eviction candidates (pages used furthest in future)

### 2.3 Page Migration Pipeline

- **D2H eviction** (CE0) and **H2D population** (CE1) overlap using dual copy engines
- Pages ordered by predicted access order → **early execution**: compute starts as soon as immediate dependency pages arrive
- Full-duplex PCIe bandwidth utilization: 63.5 GB/s vs 0.12 GB/s for demand paging (347x)

### 2.4 GPU Driver Extensions (2 new ioctls)

| ioctl | Function |
|-------|----------|
| `madvise` | Move specific pages to eviction list tail (protect from eviction) |
| `migrate` | Batch evict head pages to host DRAM + populate specified pages to HBM |

### 2.5 Architecture (extends XSched)

```
┌─────────────────────────────────────────────────┐
│ MSched                                          │
│  ┌──────────┐  ┌──────────┐  ┌────────────────┐ │
│  │ Predictor│  │  Task    │  │ Memory Manager │ │
│  │(intercept│  │Scheduler │  │ (Belady OPT +  │ │
│  │ launch)  │  │(timeline)│  │  migration)    │ │
│  └──────────┘  └──────────┘  └────────────────┘ │
│                      │                │          │
│              task timeline      madvise/migrate  │
│                      │                │          │
│              ┌───────▼────────────────▼────────┐ │
│              │     GPU Driver (modified)       │ │
│              │  eviction list + copy engines   │ │
│              └─────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

---

## 3. Key Results

### 3.1 Performance vs Demand Paging

| Subscription | Scientific Speedup | LLM Speedup |
|-------------|-------------------|-------------|
| 150% | 11.05x | 57.88x |
| 200% | 9.35x | 44.79x |
| 300% | 7.52x | 33.60x |

### 3.2 Demand Paging Overhead Breakdown

- Per-fault overhead: **31.79 μs** total
  - Data transfer: 1.35 μs (4%)
  - **Control plane: 30.44 μs (96%)**
- Fault locks entire CU TLB → stalls thousands of threads
- LLM inference: 78x slowdown under concurrent demand paging

### 3.3 vs Other Systems

| Comparison | Result |
|-----------|--------|
| vs SUV (single-task UVM opt) | 7.18x better throughput (300% sub) |
| vs XSched (compute-only) | 4.06x P99 latency reduction, 2.43x throughput |

### 3.4 Platform

- RTX 5080 (16GB, PCIe 5.0) — primary
- RTX 3080 (10GB, PCIe 4.0) — secondary
- Workloads: Rodinia (scientific), PyTorch (DNN), llama.cpp (LLM)

---

## 4. Comparison with xCoord

### 4.1 Scope Comparison

| Dimension | MSched | xCoord |
|-----------|--------|--------|
| **Problem** | GPU memory oversubscription under multitasking | CPU-GPU scheduling coordination |
| **Mechanism** | Proactive page migration (predict + prefetch) | CPU scheduling aware of GPU state (fault rate → priority boost) |
| **GPU scheduling** | Extends XSched (GPU-side preemptive scheduler) | Does not control GPU scheduling |
| **CPU scheduling** | Not involved | Core contribution (sched_ext reads GPU state) |
| **Memory management** | Core contribution (Belady OPT) | Indirect (eviction policy via gpu_ext struct_ops) |
| **Prediction** | Offline profiling + online formula evaluation | No prediction (reactive: reads current fault rate) |
| **Driver modification** | 2 new ioctls (madvise, migrate) | BPF struct_ops hooks (no ioctl) |
| **Programmability** | Fixed algorithm | Programmable via eBPF |

### 4.2 Complementary, Not Competing

**MSched** optimizes the **GPU memory data plane** — predicting which pages to move and moving them proactively. It does NOT address CPU scheduling.

**xCoord** optimizes **cross-subsystem coordination** — making CPU scheduling aware of GPU memory pressure. It does NOT predict or control page migration.

These are orthogonal:
- MSched could benefit from xCoord: when GPU tasks need proactive migration, the CPU threads performing that migration should be prioritized → xCoord's sched_ext boosting
- xCoord could benefit from MSched's insight: instead of only boosting CPU priority for fault handlers, also proactively prepare working sets before context switches

### 4.3 Key Insights from MSched for xCoord

#### Insight 1: Control Plane Overhead is the Real Bottleneck (96% of fault latency)

MSched shows that per-fault overhead is 31.79 μs, of which **96% is control plane** (CPU interrupt handling, TLB management, scheduling). Only 4% is data transfer.

**Implication for xCoord**: Boosting CPU priority for UVM fault handlers is even MORE important than we thought — the bottleneck isn't PCIe bandwidth, it's CPU processing of faults. If fault handler threads get preempted by stress-ng, the 30 μs control plane latency could balloon to milliseconds.

#### Insight 2: Pipelined Migration >> Demand Paging (347x bandwidth)

MSched achieves 63.5 GB/s pipelined migration vs 0.12 GB/s demand paging.

**Implication for xCoord**: Our current approach (boost fault handler priority) only helps at the control plane level. For the data plane, we should consider a prefetch-aware CPU scheduling: if gpu_ext signals "about to prefetch large working set", sched_ext should boost the prefetch kernel threads too.

#### Insight 3: GPU Task Timeline Enables Belady OPT

MSched can achieve optimal page replacement because it knows the future task order.

**Implication for xCoord**: gpu_ext's eviction policies (LFU, freq_decay, etc.) are reactive. If we exposed the GPU task scheduler's timeline to gpu_ext, we could implement smarter eviction. This is a potential **new direction**: sched_ext → gpu_ext coordination (reverse direction of current xCoord).

#### Insight 4: Fault Rate is a Lagging Indicator

MSched's key insight is that waiting for faults to happen and then reacting is fundamentally wrong. Prediction enables proactive scheduling.

**Implication for xCoord**: Our current shared signal (`fault_rate` in `gpu_state_map`) is a **lagging indicator** — by the time fault rate is high, performance has already degraded. Better signals might be:
- **Predicted working set size** (from gpu_ext's chunk tracking)
- **Eviction pressure** (ratio of evictions to activations)
- **Working set transition** (context switch about to happen → burst of faults incoming)

#### Insight 5: LLM Workloads are Particularly Sensitive (78x slowdown)

MSched shows LLMs suffer 78x slowdown under demand paging (vs 7-11x for scientific computing). This is because LLMs have very large working sets (KV-cache + model weights) that thrash catastrophically.

**Implication for xCoord**: Our 120B UVM experiment confirms this — TTFT ~3500ms baseline is entirely UVM overhead. For xCoord to matter, we need scenarios where CPU scheduling is the marginal bottleneck, not UVM paging. The 20B model (fits mostly in GPU) is better because UVM paging is lighter, making CPU scheduling effects visible.

---

## 5. Gaps in MSched that xCoord Could Fill

### 5.1 No CPU Scheduling Awareness

MSched controls GPU task scheduling and memory migration, but the actual migration work is done by CPU threads. If those threads are preempted or deprioritized by the Linux scheduler, MSched's proactive migration would still be slow.

**xCoord fills this**: sched_ext can boost the migration threads' CPU priority based on gpu_ext's signal that a large migration is about to happen.

### 5.2 No Multi-Tenant Isolation on CPU Side

MSched handles GPU-side multitasking but doesn't prevent CPU interference between tenants. Two GPU tenants might have their fault handler threads competing for CPU time.

**xCoord fills this**: per-PID GPU state allows sched_ext to differentially prioritize tenants based on their GPU memory pressure.

### 5.3 No eBPF Programmability

MSched is a fixed algorithm — changing the eviction or migration policy requires recompiling the driver.

**gpu_ext fills this**: BPF struct_ops allows deploying new eviction/prefetch policies at runtime without driver changes.

### 5.4 Closed-Source GPU Driver Constraint

MSched modifies the NVIDIA driver (new ioctls), which requires access to proprietary source code and limits portability.

**gpu_ext approach**: Works via BPF hooks in the open-source nvidia-uvm module, requiring minimal kernel modification (0 LOC kernel change).

---

## 6. Potential Design Improvements for xCoord

### 6.1 Add Proactive Signals (vs Reactive Fault Rate)

**Current**: `gpu_state_map.fault_rate` — lagging indicator
**Improved**: Add forward-looking signals:
- `eviction_pressure`: ratio of `eviction_count / used_count` over window
- `working_set_size`: number of active chunks per PID
- `is_context_switching`: 1 when GPU context switch detected → fault storm incoming

### 6.2 Pipeline-Aware CPU Boosting

**Current**: Boost any task with high fault rate
**Improved**: Differentiate between:
- **Fault handler threads**: need CPU priority for control plane (30 μs overhead)
- **Migration threads**: need CPU priority for data plane (H2D/D2H transfers)
- **Application threads**: need CPU priority for kernel launch
→ Different boost levels for different thread types

### 6.3 Bidirectional Coordination with GPU Scheduler

**Current**: GPU → CPU only (gpu_ext writes, sched_ext reads)
**Improved**: Also CPU → GPU:
- sched_ext signals to gpu_ext which process is running on CPU
- gpu_ext uses this to protect running process's GPU memory from eviction
- This is exactly MSched's insight: align memory management with task scheduling

### 6.4 Working Set Prediction via BPF

MSched uses offline NVBit profiling. Could gpu_ext do online prediction?
- Track per-kernel memory access patterns in BPF maps
- Detect T1/T2/T3 templates at runtime
- Signal predicted working set size to sched_ext for proactive CPU priority boost

---

## 7. Implications for xCoord Paper Positioning

### 7.1 xCoord is NOT a Competitor to MSched

MSched operates within the GPU subsystem (GPU scheduler + GPU memory).
xCoord operates across subsystems (GPU memory ↔ CPU scheduler).

### 7.2 xCoord's Unique Contribution

No existing system (including MSched) addresses:
- CPU scheduling decisions informed by GPU memory state
- Cross-subsystem eBPF coordination (sched_ext + gpu_ext)
- Runtime-programmable GPU-aware CPU scheduling

### 7.3 MSched as Related Work

MSched should be cited as the state-of-the-art in GPU memory scheduling.
xCoord's positioning: "Even with optimal GPU-side memory scheduling (MSched), CPU-GPU coordination gaps remain because the CPU scheduler is unaware of GPU memory pressure. xCoord fills this gap."

### 7.4 Combined Vision

The ultimate system would combine:
1. **MSched**: Proactive GPU memory scheduling (prediction + migration)
2. **gpu_ext**: Programmable GPU memory policies (eBPF struct_ops)
3. **xCoord**: Cross-subsystem coordination (GPU state → CPU scheduler)
4. **sched_ext**: GPU-aware CPU scheduling (boost fault/migration threads)

---

## 8. References

- MSched paper: https://arxiv.org/abs/2512.24637
- XSched (GPU preemptive scheduler): referenced in MSched
- SUV (single-task UVM optimization): referenced in MSched
- NVIDIA Unified Memory: CUDA documentation
- Belady's algorithm: optimal page replacement (1966)

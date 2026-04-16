# 内核系统层可开展工作方案

> 基于 15+ 种已测试算法的实验数据，结合 gpu_ext BPF 框架能力和 vLLM/llama.cpp workload 特征，
> 本文档给出**经过实证检验的、可在内核系统层面落地的工作方向**。
>
> 关联文档：
> - [cross_block_prefetch_mechanism.md](cross_block_prefetch_mechanism.md) — 机制实现与实验数据
> - [cross_block_prefetch_plan.md](cross_block_prefetch_plan.md) — 算法设计与全量结果
> - [attention_aware_memory_subsystem_feasibility.md](attention_aware_memory_subsystem_feasibility.md) — 语义驱逐可行性

---

## 0. 核心前提：实验已证明的事实

在讨论任何新工作之前，必须先确认以下已被充分验证的结论，避免重复走弯路：

| 结论 | 证据 | 来源 |
|------|------|------|
| **驱逐排序不是瓶颈** | 6 种算法差异 < 1%（reuse_dist ≈ cycle_moe ≈ Belady ≈ LRU） | mechanism §8.10-8.11 |
| **always_max 是 fault-driven 天花板** | 15 种启发式均未超越 | plan §9.5 |
| **oversub ratio 决定 XB 有效性** | <1.5x 有效，>1.5x 有害 | mechanism §19.9 |
| **真正的瓶颈是 fault latency** | llama 理论上限 ~167 tok/s vs 实际 92 tok/s，差距来自串行 DMA | plan §10.1 |
| **proactive migration 可突破天花板** | uprobe microbench: always_max 2009ms → uprobe 788ms (**2.5x**) | plan §7.9 |

**因此，所有新工作应聚焦在"在 fault 发生之前就开始迁移数据"，而非"fault 发生后更聪明地处理"。**

---

## 1. 工作方向一：MoE Expert Proactive Prefetch（最高优先级）

### 1.1 动机与理论分析

MoE 模型（如 GPT-OSS-120B, Qwen3-30B-A3B）的 decode 阶段：每个 token 仅激活 top-k experts（通常 k=2）。Router/gate 在 GPU 上计算 softmax 后选出 expert ID，随后 GPU 执行该 expert 的 FFN 计算。

```
当前流程（串行）：
  Router(GPU) → 选出 expert_id → GPU 访问 expert weights → PAGE FAULT
  → CPU fault handler → DMA: CPU→GPU (expert weights) → GPU 恢复计算
  → GPU 执行 expert FFN → 完成 → 下一层...

  总延迟 = compute + fault_latency + DMA_latency（串行累加）
```

```
目标流程（pipeline overlap）：
  Router(GPU) → 选出 expert_id
  同时: CPU/BPF 得知 expert_id → bpf_wq → migrate(expert_N+1 weights)
  GPU 执行 expert_N FFN
  当 expert_N 完成时，expert_N+1 weights 已在 VRAM
  → 零 fault latency → 总延迟 ≈ max(compute, DMA)
```

**理论上限**：llama.cpp 120B 的 decode DMA 占比 59%（6.6ms/token 中的 3.9ms）。如果 DMA 与 compute 完全重叠，tg 可从 92 → ~150 tok/s（+63%）。

### 1.2 技术路径

有两条互补路径，一条基于 llama.cpp，一条基于 vLLM。

#### 路径 A：llama.cpp — fault bitmap replay 改进版

已有基础：`prefetch_moe_expert.bpf.c` 实现了 fault bitmap replay — 记录上一 token 的 fault 地址集合，下一 token 开始时 replay（提前迁移相同地址）。

**问题**：当前实现是"盲目 replay"（假设下一 token 访问同一组 expert），但 MoE routing 每 token 不同。

**改进方案**：

```
改进 1: Router-Informed Replay
  不是 replay 上一 token 的 fault bitmap，
  而是在 router 计算完成后，从 GPU 端拿到 expert_id，
  用 expert_id → VA range 映射表直接预取对应 expert。

改进 2: 双缓冲 + pipeline
  Token N: GPU 执行 expert_a → BPF 预取 expert_b (token N+1 已知)
  Token N+1: GPU 执行 expert_b → BPF 预取 expert_c (token N+2 已知)
```

**关键挑战**：

1. **如何从 GPU router 输出拿到 expert_id**？
   - llama.cpp 的 `topk-moe.cu` 在 GPU 上执行 top-k。结果在 GPU memory 中。
   - 方案 a：UVM 共享 buffer — router 结果写入 `cudaMallocManaged` 区域，CPU 可直接读
   - 方案 b：`cudaMemcpyAsync` D2H — router 结果异步拷回 CPU，uprobe 拦截完成事件
   - 方案 c：在 `ggml-cuda.cu` 的 dispatch 逻辑中，于 CPU 侧暴露 expert_id（需分析 llama.cpp 调度流程）

2. **Expert weights 的 VA 布局**：
   - 需要离线分析：哪个 expert 对应哪些 VA range
   - 已有工具：`chunk_trace` + `derive_layer_mapping.py` 可生成 VA→layer 映射
   - 需要扩展到 VA→expert 粒度

#### 路径 B：vLLM — FusedMoE layer 注入 prefetch hint

vLLM 的 MoE 实现在 `vllm/model_executor/layers/fused_moe/layer.py`。其 `forward_cuda()` 调用链：

```python
# layer.py 中 FusedMoEMethodBase.forward_cuda()
def forward_cuda(self, layer, x, ..., router_logits, ...):
    # 1. Router: topk_weights, topk_ids = FusedMoE.select_experts(...)
    #    → topk_ids 包含每个 token 选中的 expert ID
    # 2. Dispatch: fused_experts(hidden_states, w1, w2, topk_weights, topk_ids, ...)
    #    → GPU kernel 访问 w1[expert_id], w2[expert_id] 的权重
```

**注入点**：在 step 1 (router) 和 step 2 (dispatch) 之间，CPU 已经知道 `topk_ids`：

```python
# 修改 vLLM: 在 fused_moe forward 中注入 prefetch hint
def forward_cuda(self, layer, x, ..., router_logits, ...):
    topk_weights, topk_ids = self.select_experts(router_logits, ...)

    # ===== 新增: 通知 BPF 即将访问哪些 expert =====
    if uvm_enabled and hasattr(layer, 'expert_va_ranges'):
        for expert_id in topk_ids.unique().cpu().tolist():
            va_start, va_len = layer.expert_va_ranges[expert_id]
            uvm_lib.uvm_request_prefetch(va_start, va_len)
    # ================================================

    return fused_experts(x, layer.w13_weight, layer.w2_weight,
                        topk_weights, topk_ids, ...)
```

**BPF 侧**：复用已验证的 sleepable uprobe + `bpf_gpu_migrate_range()` 机制：

```c
// uprobe on uvm_request_prefetch(addr, len)
SEC("uprobe.s//path/to/uvm_allocator.so:uvm_request_prefetch")
int BPF_PROG(expert_prefetch, u64 addr, u64 len) {
    // 直接调用 sleepable kfunc 迁移 expert weights
    bpf_gpu_migrate_range(cached_va_space, addr, len);
    return 0;
}
```

### 1.3 实现计划

| 阶段 | 内容 | 修改范围 | 预计时间 |
|------|------|---------|---------|
| **A1** | Expert VA 布局分析 | `chunk_trace` + 新分析脚本 | 3 天 |
| **A2** | vLLM FusedMoE hint 注入 | `fused_moe/layer.py` + `uvm_allocator.cpp` | 3 天 |
| **A3** | BPF expert prefetch 策略 | 新建 `prefetch_expert_hint.bpf.c` | 3 天 |
| **A4** | vLLM 30B + llama.cpp 120B 实验 | 实验脚本 | 3 天 |
| **A5** | pipeline depth 调优 | rodata 参数化 | 2 天 |

### 1.4 内核侧改动

**无新增内核代码**。全部复用已有基础设施：
- `bpf_gpu_migrate_range()` kfunc（已存在）
- sleepable uprobe `SEC("uprobe.s/...")` 机制（已验证）
- kprobe 获取 `va_space`（已有模式）

**vLLM 侧改动**：
- `uvm_allocator.cpp`：新增 `uvm_request_prefetch(void* addr, size_t len)` — 空函数，仅作为 uprobe 挂载点
- `fused_moe/layer.py`：在 `forward_cuda()` 中注入 hint 调用
- `gpu_model_runner.py`：初始化时建立 expert → VA range 映射表

### 1.5 预期收益

| Workload | 当前最佳 | 预期提升 | 原理 |
|----------|---------|---------|------|
| llama.cpp 120B (1.84x) | tg 92 tok/s | **+30-60%** (120-150 tok/s) | DMA-compute overlap |
| vLLM 30B (1.175x) | TPOT 55ms | **+10-20%** (45-50ms) | 减少 expert fault stall |

---

## 2. 工作方向二：BPF 作为应用-内核语义通道

### 2.1 动机

当前 gpu_ext 的 BPF 策略完全是 fault-driven — BPF 只在 GPU page fault 时被触发，此时 GPU 已经 stall。uprobe 实验（§7.9 microbench 2.5x, §7.11 FAISS phase detection）证明：**应用层可以通过 uprobe 向 BPF 传递语义信息，BPF 据此做 proactive 决策**。

这不是一个单独的算法，而是一个**系统架构模式**：

```
┌──────────────────────────────────────────────┐
│  应用层 (vLLM / llama.cpp / FAISS / GNN)      │
│                                                │
│  在关键决策点调用 hint 函数:                     │
│    uvm_hint_prefetch(addr, len)               │
│    uvm_hint_phase(PREFILL / DECODE)           │
│    uvm_hint_expert_ids(ids[], count)          │
│    uvm_hint_batch_nodes(node_ids[], count)    │
│    uvm_hint_will_free(addr, len)              │
└────────────────────┬─────────────────────────┘
                     │ uprobe (零开销 when BPF not loaded)
┌────────────────────▼─────────────────────────┐
│  BPF 策略层 (可热加载/卸载)                     │
│                                                │
│  uprobe handler:                               │
│    解析 hint → 写入 BPF map / 触发 bpf_wq      │
│  struct_ops handler:                           │
│    读取 BPF map → 影响 prefetch/eviction 决策   │
│  bpf_wq callback:                              │
│    bpf_gpu_migrate_range() proactive 迁移      │
└────────────────────┬─────────────────────────┘
                     │ kfunc
┌────────────────────▼─────────────────────────┐
│  NVIDIA UVM 内核模块 (nvidia-uvm.ko)           │
│                                                │
│  uvm_migrate() → DMA H2D/D2H                  │
└──────────────────────────────────────────────┘
```

### 2.2 Hint API 设计

在 `uvm_allocator.cpp`（vLLM 的 UVM allocator 共享库）中新增以下 C 函数。这些函数本身是空实现 — 它们的唯一目的是作为 uprobe 挂载点：

```c
// uvm_allocator.cpp 新增

// 通知即将访问指定地址范围（proactive prefetch hint）
extern "C" void uvm_hint_prefetch(void* addr, size_t len) {
    // 空函数体。BPF uprobe 挂载时才有实际效果。
    // 编译器不会优化掉：extern "C" + 被 Python ctypes 调用。
}

// 通知即将释放指定地址范围（eviction 可安全跳过）
extern "C" void uvm_hint_will_free(void* addr, size_t len) {
    // 空函数体。
}

// 通知当前执行阶段（prefill=1, decode=2, ...）
extern "C" void uvm_hint_phase(int phase) {
    // 空函数体。（替代已有的 uvm_set_phase）
}

// 通知即将访问的 expert ID 列表
extern "C" void uvm_hint_expert_ids(const int* ids, int count,
                                     const void* weights_base) {
    // 空函数体。BPF 从参数读取 expert_id，
    // 结合 expert_va_map 计算 VA range，触发迁移。
}

// 通知即将访问的 KV block ID 列表（用于 KV cache 预取）
extern "C" void uvm_hint_kv_blocks(const int* block_ids, int count,
                                    const void* kv_base, size_t block_size) {
    // 空函数体。
}
```

### 2.3 应用侧集成点

#### vLLM 集成

```python
# vllm/device_allocator/uvm.py 新增 Python binding
def uvm_hint_prefetch(addr: int, length: int) -> None:
    if _uvm_lib is not None:
        _uvm_lib.uvm_hint_prefetch(ctypes.c_void_p(addr), ctypes.c_size_t(length))

def uvm_hint_expert_ids(ids: list[int], weights_base: int) -> None:
    if _uvm_lib is not None:
        arr = (ctypes.c_int * len(ids))(*ids)
        _uvm_lib.uvm_hint_expert_ids(arr, len(ids), ctypes.c_void_p(weights_base))
```

```python
# vllm/model_executor/layers/fused_moe/layer.py 注入点
class FusedMoE(torch.nn.Module):
    def forward_native(self, hidden_states, router_logits):
        # ... existing router logic ...
        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states, router_logits, ...)

        # ===== Hint: 告知 BPF 即将访问的 expert =====
        if is_uvm_enabled():
            unique_experts = topk_ids.unique()
            uvm_hint_expert_ids(
                unique_experts.cpu().tolist(),
                self.w13_weight.data_ptr()
            )
        # ================================================

        return fused_experts(hidden_states, ...)
```

#### llama.cpp 集成

llama.cpp 是 C++ 项目，可以直接在 GGML CUDA dispatch 中注入：

```c
// ggml/src/ggml-cuda/ggml-cuda.cu 中的 MoE dispatch 点
// 在 top-k 计算完成后、expert FFN 执行前，调用 hint 函数

// 方案：在 ggml_cuda_op_mul_mat_id() 中
// ids tensor 包含每个 token 选中的 expert index
// 读取 ids → 计算 expert weights 的 VA range → 调用 hint
```

### 2.4 BPF 策略模板

```c
// extension/prefetch_app_guided.bpf.c — 通用应用引导预取策略

#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "bpf_experimental.h"
#include "uvm_types.h"
#include "bpf_testmod.h"

char _license[] SEC("license") = "GPL";

/* 从 kprobe 缓存的 va_space handle */
struct { __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY); __uint(max_entries, 1);
         __type(key, u32); __type(value, u64); } va_space_cache SEC(".maps");

/* 应用 hint 产生的 pending prefetch 请求 */
struct prefetch_req { u64 addr; u64 len; };
struct { __uint(type, BPF_MAP_TYPE_QUEUE); __uint(max_entries, 64);
         __type(value, struct prefetch_req); } pending_queue SEC(".maps");

/* Expert VA range 映射表（用户态 loader 填充） */
struct expert_range { u64 offset; u64 len; };
struct { __uint(type, BPF_MAP_TYPE_ARRAY); __uint(max_entries, 256);
         __type(key, u32); __type(value, struct expert_range); } expert_map SEC(".maps");

/* ===== kprobe: 捕获 va_space ===== */
SEC("kprobe/uvm_perf_prefetch_get_hint_va_block")
int BPF_KPROBE(capture_va_space, void *va_block) {
    u64 va_space = /* BPF_CORE_READ chain to va_space */;
    u32 zero = 0;
    bpf_map_update_elem(&va_space_cache, &zero, &va_space, 0);
    return 0;
}

/* ===== sleepable uprobe: 响应应用 prefetch hint ===== */
SEC("uprobe.s//path/to/uvm_allocator.so:uvm_hint_prefetch")
int BPF_PROG(handle_hint_prefetch, u64 addr, u64 len) {
    u32 zero = 0;
    u64 *vs = bpf_map_lookup_elem(&va_space_cache, &zero);
    if (vs && *vs)
        bpf_gpu_migrate_range(*vs, addr, len);
    return 0;
}

/* ===== sleepable uprobe: 响应 expert ID hint ===== */
SEC("uprobe.s//path/to/uvm_allocator.so:uvm_hint_expert_ids")
int BPF_PROG(handle_hint_experts, const int *ids, int count, u64 weights_base) {
    u32 zero = 0;
    u64 *vs = bpf_map_lookup_elem(&va_space_cache, &zero);
    if (!vs || !*vs) return 0;

    /* 遍历 expert ids，查 expert_map，触发迁移 */
    for (int i = 0; i < count && i < 8; i++) {
        int eid = 0;
        bpf_probe_read_user(&eid, sizeof(eid), &ids[i]);
        u32 key = (u32)eid;
        struct expert_range *er = bpf_map_lookup_elem(&expert_map, &key);
        if (er && er->len > 0) {
            bpf_gpu_migrate_range(*vs, weights_base + er->offset, er->len);
        }
    }
    return 0;
}

/* ===== struct_ops: 照常 always_max + cycle_moe ===== */
SEC("struct_ops/gpu_page_prefetch")
int BPF_PROG(gpu_page_prefetch, ...) {
    /* always_max: 预取整个 VA block */
    bpf_gpu_set_prefetch_region(result_region, max_first, max_outer);
    return 1;
}

SEC("struct_ops/gpu_block_activate")
int BPF_PROG(gpu_block_activate, ...) {
    /* cycle_moe: T1 保护 */
    /* ... 标准 cycle_moe 逻辑 ... */
    return 1;
}
```

### 2.5 系统架构价值

这套 hint API 的学术价值在于：

1. **零侵入性**：hint 函数是空函数体，不加载 BPF 时零开销。应用不依赖 BPF — 是纯 opt-in。
2. **热加载**：BPF 策略可运行时加载/卸载，不重启应用或内核模块。
3. **可组合**：不同 BPF 策略可以选择性地 hook 不同 hint 函数。
4. **跨应用通用**：同一套 API 可用于 vLLM、llama.cpp、FAISS、PyTorch GNN — 每个应用只需在合适的点调 hint。

---

## 3. 工作方向三：BPF 接口扩展

### 3.1 动机

当前的 BPF 接口（6 个 struct_ops hook + 5 个 kfunc）在实验中暴露了多处限制。以下扩展每一个都有实验驱动的理由。

### 3.2 扩展清单

#### 3.2.1 新 kfunc：`bpf_gpu_get_pmm_pressure()`

```c
// 返回当前 GPU 内存压力指标
__bpf_kfunc u32 bpf_gpu_get_pmm_pressure(uvm_pmm_gpu_t *pmm) {
    // 返回 evictable chunk 占比 (0-100)
    // 或 free chunk 数量
}
```

**动机**：`prefetch_throttled_xb.bpf.c` 使用 fault rate 作为 PCIe 压力代理指标，但 fault rate 滞后且不精确（L5/L6 实验均有害）。直接查询 PMM 状态可以做出更准确的决策：

- 有空闲 chunk → 放心 proactive prefetch
- 接近满载 → 停止 proactive prefetch（避免零和博弈）
- 用于 §1 expert prefetch 的 rate limiting

**内核改动**：~10 行，读取 `pmm->root_chunks` 统计信息。

#### 3.2.2 新 kfunc：`bpf_gpu_get_chunk_va()`

```c
// 从 chunk 指针获取其对应的 VA start
__bpf_kfunc u64 bpf_gpu_get_chunk_va(uvm_gpu_chunk_t *chunk) {
    uvm_va_block_t *vb = chunk->va_block;
    return vb ? vb->start : 0;
}
```

**动机**：当前 BPF 中获取 chunk VA 需要两次 `BPF_CORE_READ`（`chunk→va_block→start`），每次需要处理 NULL 检查。这是所有 eviction 策略的 hot path 操作。一个专用 kfunc 更安全高效。

**内核改动**：5 行。

#### 3.2.3 新 hook：`gpu_migrate_complete` 回调

```c
// 在 uvm_migrate() 完成后调用
int (*gpu_migrate_complete)(u64 va_start, u64 va_end,
                            int direction,  // 0=H2D, 1=D2H
                            u64 bytes_migrated,
                            u64 elapsed_ns);
```

**动机**：当前 BPF 无法知道 proactive prefetch 是否完成。对于 §1 的 pipeline prefetch，需要知道"expert N+1 的数据是否已就位"才能安全地通知 GPU 继续。

更重要的是，这个 hook 可以用于：
- 统计实际 DMA 流量（当前只能估算上界）
- 测量 DMA 延迟（用于自适应 rate limiting）
- 构建精确的 PCIe 带宽利用率模型

**内核改动**：~20 行，在 `uvm_va_block_make_resident()` 完成后调用。

#### 3.2.4 扩展 `gpu_page_prefetch` hook 签名

```c
// 当前签名
int (*gpu_page_prefetch)(uvm_page_index_t page_index,
    uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
    uvm_va_block_region_t *max_prefetch_region,
    uvm_va_block_region_t *result_region);

// 扩展签名（可选，向后兼容）
int (*gpu_page_prefetch_v2)(uvm_page_index_t page_index,
    uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
    uvm_va_block_region_t *max_prefetch_region,
    uvm_va_block_region_t *result_region,
    u64 va_block_start,       // 当前 VA block 起始地址
    u64 va_space_handle,      // va_space 指针（for cross-block）
    u32 fault_count_in_block  // 该 block 的累计 fault 数
    );
```

**动机**：当前所有需要 VA 信息的 BPF 策略都必须使用 kprobe side channel（挂 `uvm_perf_prefetch_get_hint_va_block`，通过 per-CPU map 传递 va_block 指针给 struct_ops hook）。这在 mechanism §16 中被总结为"workaround"。直接在 hook 中传递可消除这个 hack。

**但**，根据 mechanism §16.3 确立的原则，优先使用 kprobe + CO-RE 模式。此扩展仅在性能分析表明 kprobe overhead 显著时才值得做。

### 3.3 接口扩展的论文叙事

每个新接口遵循以下结构：

```
1. Motivation: 展示当前 workaround 的代码复杂度
   → 对比: kprobe side channel hack (~30 行) vs 新 hook 参数 (0 行)
2. Interface Design: 新增签名 + 语义
3. Implementation: 内核改动行数
4. Evaluation: 同一算法在旧/新接口下的 A/B 对比
   → 代码行数减少 + 潜在的 kprobe overhead 消除
5. Case Study: 新接口 enable 了哪个之前做不到的策略
```

---

## 4. 工作方向四：Oversub-Aware 自动策略选择

### 4.1 动机

实验最明确的结论之一是：**最优策略取决于 oversubscription ratio**。

| Oversub | 最优 prefetch | 最优 eviction | XB |
|---------|-------------|---------------|-----|
| < 1.0x | 无需 BPF | 无需 BPF | 不触发 |
| 1.0-1.3x | always_max | cycle_moe | direction-aware 有效 |
| 1.3-1.5x | always_max | cycle_moe | 取决于访问模式 |
| > 1.5x | always_max | cycle_moe | 有害，禁用 |

当前用户必须手动选择策略并配置参数。一个自动化的策略选择器可以根据运行时 oversub ratio 动态切换。

### 4.2 实现方案

```c
// extension/prefetch_auto_adaptive.bpf.c

/* 运行时状态 */
struct adaptive_state {
    u64 total_faults;
    u64 total_evictions;
    u64 window_faults;         // 滑动窗口内的 fault 数
    u64 window_start_ns;
    u32 current_mode;          // 0=conservative, 1=aggressive, 2=xb_enabled
    u32 oversub_estimate;      // 估算的 oversub ratio × 100
};

SEC("struct_ops/gpu_block_activate")
int BPF_PROG(gpu_block_activate, ...) {
    struct adaptive_state *st = bpf_map_lookup_elem(&state_map, &zero);
    if (!st) return 0;

    st->total_faults++;
    st->window_faults++;

    u64 now = bpf_ktime_get_ns();
    u64 elapsed = now - st->window_start_ns;

    /* 每 1 秒重新评估策略 */
    if (elapsed > 1000000000ULL) {
        u64 fault_rate = st->window_faults * 1000000000ULL / elapsed;

        /* 根据 fault rate 估算 oversub:
         * fault_rate < 100/s → fits in VRAM, no action needed
         * fault_rate 100-1000/s → moderate oversub, XB may help
         * fault_rate > 1000/s → heavy oversub, disable XB
         */
        if (fault_rate < 100) {
            st->current_mode = 0;  // conservative: 不需要额外策略
        } else if (fault_rate < 1000) {
            st->current_mode = 2;  // aggressive + XB
        } else {
            st->current_mode = 1;  // aggressive, no XB
        }

        st->window_faults = 0;
        st->window_start_ns = now;
    }

    /* 根据 current_mode 执行策略 */
    /* ... cycle_moe + conditional XB ... */
    return 1;
}
```

**更好的方案**：如果 §3.2.1 的 `bpf_gpu_get_pmm_pressure()` kfunc 可用，可以直接查询 free chunk 比例，而非用 fault rate 间接估算。

### 4.3 与 vLLM 的集成

vLLM 可以在启动时通过环境变量控制策略：

```python
# vllm/device_allocator/uvm.py
def select_bpf_policy():
    """根据模型大小和 GPU 内存自动选择 BPF 策略"""
    model_size_gb = estimate_model_size()
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    oversub_ratio = model_size_gb / gpu_mem_gb

    if oversub_ratio < 1.0:
        return None  # 不需要 BPF
    elif oversub_ratio < 1.5:
        return "prefetch_always_max_cycle_moe"  # + XB direction
    else:
        return "prefetch_always_max_cycle_moe"  # 无 XB
```

---

## 5. 工作方向五：DMA-Compute Overlap 的内核级支持

### 5.1 动机

当前 UVM 的 page migration 是**同步阻塞**的：GPU fault → fault handler 暂停 GPU → DMA → GPU 恢复。这意味着 DMA 和 GPU compute 完全串行。

MSched 论文的核心贡献之一是利用**独立的 copy engine (CE)** 实现 DMA-compute overlap。gpu_ext 可以在 BPF 层实现类似效果。

### 5.2 当前机制分析

`bpf_gpu_migrate_range()` 通过 `bpf_wq` 在 process context 中执行，与 fault path 异步。这已经提供了一定程度的 overlap：

```
时间线 (当前):
  GPU: [compute]──[STALL: fault]──[compute]──[STALL: fault]──
  DMA:              [demand]                   [demand]
  WQ:                              [proactive]        [proactive]
                                    ↑ 可能与 compute overlap
```

**问题**：proactive DMA 和 demand DMA 共享同一 PCIe 链路和 UVM copy engine。在高 oversub 下，proactive DMA 与 demand DMA 互相竞争带宽。

### 5.3 可行改进

#### 5.3.1 优先级化 DMA 请求

在 `uvm_migrate_bpf()` wrapper 中标记 proactive vs demand：

```c
// uvm_migrate.c
NV_STATUS uvm_migrate_bpf(uvm_va_space_t *va_space, NvU64 base, NvU64 length)
{
    // 设置低优先级标志，让 demand migration 优先
    NvU32 flags = UVM_MIGRATE_FLAG_ASYNC | UVM_MIGRATE_FLAG_LOW_PRIORITY;
    return uvm_migrate(va_space, NULL, base, length, ..., flags, ...);
}
```

**注意**：NVIDIA UVM 的 `migrate_flags` 是否支持优先级需要调研。如果不支持，可以在 BPF 侧实现软优先级 — 当检测到 demand fault rate 升高时暂停 proactive prefetch。

#### 5.3.2 CE 通道分离

NVIDIA GPU 有多个 copy engine。理论上可以将 proactive prefetch 分配到与 demand fault 不同的 CE 上。但这需要修改 UVM 驱动的 CE 分配逻辑，改动较大。

**可行的软件层模拟**：使用 `cudaMemcpyAsync` 在用户态发起 proactive 迁移（绕过 UVM），让 CUDA runtime 分配独立 CE。这需要应用层参与（非纯 BPF）。

### 5.4 实验验证计划

| 实验 | 内容 | 指标 |
|------|------|------|
| E1 | 基准: always_max + cycle_moe (无 proactive) | tg tok/s |
| E2 | proactive expert prefetch (§1) via BPF | tg tok/s, DMA overlap % |
| E3 | proactive + demand 优先级分离 | tg tok/s, demand fault latency |
| E4 | 用户态 cudaMemPrefetchAsync + BPF eviction | tg tok/s |

---

## 6. 工作方向六：vLLM KV Cache 生命周期管理

### 6.1 动机

vLLM 在 UVM 模式下，KV cache blocks 与 model weights 共享 UVM 地址空间和 BPF 策略。但两者有完全不同的访问模式：

| 特征 | Model Weights | KV Cache |
|------|-------------|----------|
| 大小 | 固定（模型参数量） | 动态增长（随 request 数量） |
| 访问 | 循环（每层每 token 访问） | 追加 + 回读（写入新 token，attention 读旧 token） |
| 生命周期 | 永久（进程期间不变） | 短暂（request 完成后释放） |
| 重要性 | 均匀（每层权重同等重要） | 异质（recent token > old token） |

### 6.2 BPF 区分 Model Weights vs KV Cache

利用 uprobe 记录 KV cache 的 VA 范围：

```c
// 在 vLLM 的 CUDAPluggableAllocator 回调中
// KV cache 是通过 cudaMallocManaged 分配的，大小已知
// uprobe on uvm_malloc() → 记录 [addr, addr+size) 到 kv_range_map
```

```c
// BPF eviction 策略
SEC("struct_ops/gpu_block_activate")
int BPF_PROG(gpu_block_activate, ...) {
    u64 chunk_va = get_chunk_va(chunk);

    // 检查是否在 KV cache 范围内
    struct kv_range *kr = bpf_map_lookup_elem(&kv_range_map, &chunk_va);

    if (kr) {
        // KV cache chunk: 基于 token 位置排序
        // 越旧的 token → 越容易被驱逐
        if (kr->token_age > STALE_THRESHOLD) {
            bpf_gpu_block_move_head(chunk, list);
        } else {
            bpf_gpu_block_move_tail(chunk, list);
        }
    } else {
        // Model weight chunk: cycle_moe T1 保护
        // ... 标准 cycle_moe 逻辑 ...
    }
    return 1;
}
```

### 6.3 vLLM 侧修改

```python
# vllm/v1/worker/gpu_worker.py
# 在 KV cache 分配后，通知 BPF KV 的 VA 范围

def _init_kv_cache(self):
    # ... existing KV cache allocation ...
    if is_uvm_enabled():
        for layer_idx, (k_cache, v_cache) in enumerate(kv_caches):
            uvm_hint_kv_range(
                k_cache.data_ptr(), k_cache.numel() * k_cache.element_size(),
                layer_idx, 'key'
            )
            uvm_hint_kv_range(
                v_cache.data_ptr(), v_cache.numel() * v_cache.element_size(),
                layer_idx, 'value'
            )
```

### 6.4 预期效果

这个方向在**低 oversub (1.1-1.3x)** 下可能有价值：当 KV cache 增长导致 model weights 被驱逐时，BPF 可以优先驱逐旧的 KV tokens 而非 attention weights。

但根据实验数据，eviction 排序在高 oversub 下差异 < 1%，因此主要价值在低 oversub 的 serving 场景（vLLM 1.175x）。

---

## 7. 综合优先级与时间线

| 优先级 | 工作方向 | 预期改进 | 工程量 | 新增内核代码 | vLLM 修改 |
|--------|---------|---------|--------|------------|----------|
| **P0** | §1 MoE Expert Proactive Prefetch | tg **+30-60%** | 2 周 | 0 行 | ~50 行 Python + ~20 行 C |
| **P1** | §2 应用-内核语义通道 | 架构贡献 | 1 周 | 0 行 | ~100 行 C (hint API) |
| **P1** | §3.2.1 PMM pressure kfunc | 使 §4 和 §1 更精确 | 2 天 | ~10 行 | 0 |
| **P1** | §3.2.3 migrate_complete hook | 使 §5 pipeline 可控 | 3 天 | ~20 行 | 0 |
| **P2** | §4 自动策略选择 | 易用性 | 1 周 | 0 行 | ~30 行 Python |
| **P2** | §5 DMA-compute overlap | +10-20% | 2 周 | ~20 行 | 取决于方案 |
| **P3** | §6 KV cache 生命周期 | < 5% (低 oversub) | 1 周 | 0 行 | ~30 行 Python |
| **P3** | §3.2.2 chunk VA kfunc | 代码简化 | 1 天 | 5 行 | 0 |
| **低** | §3.2.4 hook 签名扩展 | 消除 kprobe hack | 3 天 | ~30 行 | 0 |

### 推荐执行顺序

```
Week 1-2: §1 MoE Expert Proactive Prefetch
  ├─ Day 1-3: Expert VA 布局分析 + vLLM hint 注入
  ├─ Day 4-6: BPF 策略实现 + 调试
  └─ Day 7-10: vLLM 30B + llama.cpp 120B 实验 + 调优

Week 3: §2 语义通道 + §3 接口扩展
  ├─ Day 1-2: Hint API 设计 + uvm_allocator 修改
  ├─ Day 3-4: PMM pressure kfunc + migrate_complete hook
  └─ Day 5: 通用 BPF 模板 + 文档

Week 4: §4 自动策略 + §5 DMA overlap 实验
  ├─ Day 1-3: 自适应策略实现 + 多 oversub 级别测试
  └─ Day 4-5: DMA overlap 实验 + 结果分析

Week 5 (可选): §6 KV cache + 论文写作
```

---

## 8. 与已有工作的关系

### 8.1 本方案 vs attention_aware_memory_subsystem_feasibility.md

| 原始提案 | 本方案对应 | 差异 |
|---------|-----------|------|
| GPU 端 Attention Score 采集 | §1 Router-informed expert prefetch | 不需要修改 attention kernel；用 router 输出替代 attention score |
| Score → BPF 通道 | §2 应用-内核语义通道 | 通用化：不限于 score，支持任意语义 hint |
| Level 1 语义驱逐 | 降低优先级（§6 KV 生命周期） | 实验证明驱逐排序差异 < 1% |
| Level 2 VRAM 内压缩 | 不在本方案范围 | 内核态无法 GPU 计算，留给应用层 |
| Level 3 热页保护 | 已实现（cycle_moe） | 无需额外工作 |

### 8.2 本方案 vs 已测试的 15+ 算法

| 已测试算法 | 结果 | 本方案如何超越 |
|-----------|------|--------------|
| 各种 eviction (reuse_dist, Belady, cooperative) | 差异 < 1% | 不再做 eviction 改进 |
| Cross-block blind/direction/stride | 高 oversub 有害 | §1 用语义精确预取替代盲猜 |
| Phase detection (uprobe/heuristic) | 机制正确但收益有限 | §2 扩展到更丰富的语义 |
| Proactive layer prefetch | VA boundary 粒度太粗 | §1 精确到 expert level |
| MoE bitmap replay | 假设下 token 访问同组 expert | §1 用 router 真实输出 |

### 8.3 核心范式转移

```
已完成的工作:
  Fault 发生 → BPF 决策 → 影响 DMA 方向/范围
  (reactive, 最多改善 fault 处理效率)

本方案的工作:
  应用语义 → BPF 预知 → DMA 在 fault 前启动
  (proactive, 消除 fault latency 本身)
```

这是从 **"更聪明地处理 fault"** 到 **"消除 fault"** 的范式转移。

---

## 9. 风险评估

| 风险 | 影响 | 缓解 |
|------|------|------|
| Expert hint 的 Python→C→uprobe 路径延迟过高 | 预取来不及完成 | 使用 `ctypes` 直接调用，避免 Python overhead；或 Cython |
| Router 输出 `topk_ids` 需要 `.cpu()` 同步 | 阻塞 GPU pipeline | 使用 `cudaMemcpyAsync` + 在 stream callback 中触发 hint |
| Expert VA 布局在不同运行之间不固定 | expert_map 无效 | 运行时通过 uprobe on `cudaMallocManaged` 动态构建 |
| `bpf_gpu_migrate_range()` 与 demand fault 竞争 PCIe | 负增益 | §3.2.1 PMM pressure kfunc 做 rate limiting |
| BPF HASH map 在 hot path 导致 Xid 31 | GPU crash | expert_map 用 ARRAY（expert 数量固定且小），不用 HASH |

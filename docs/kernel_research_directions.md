# 内核层研究方向：面向论文的创新点

> 前提约束：
> 1. 可以修改 nvidia-uvm.ko 内核模块代码，但限于**小幅修改**（单点 < 30 行），不重构整体架构
> 2. 核心目标是**发论文**，每个方向必须有清晰的 novelty claim
> 3. 所有方向均基于已有实验数据（15+ 种算法对比），有实证支撑
>
> 关联文档：
> - [cross_block_prefetch_plan.md](cross_block_prefetch_plan.md) — 15+ 种算法实验数据
> - [cross_block_prefetch_mechanism.md](cross_block_prefetch_mechanism.md) — 跨块预取机制
> - [future_work_kernel_system.md](future_work_kernel_system.md) — 工程落地工作方案

---

## 0. 为什么需要内核修改？当前接口的局限性

当前 gpu_ext 的 BPF 接口包含 6 个 struct_ops hook 和 5 个 kfunc：

| 接口 | 类型 | 能力 |
|------|------|------|
| `gpu_block_activate` | hook | chunk 变为可驱逐时调用，可通过 `move_head/tail` 调整 LRU 位置 |
| `gpu_block_access` | hook | ❌ 从未被调用（driver bug） |
| `gpu_evict_prepare` | hook | 驱逐前调用，可重排 used/unused 链表 |
| `gpu_page_prefetch` | hook | 控制 fault 后的预取范围 |
| `gpu_page_prefetch_iter` | hook | 预取迭代控制 |
| `bpf_gpu_migrate_range` | kfunc | 跨块主动迁移（sleepable） |
| `bpf_gpu_block_move_head/tail` | kfunc | 调整驱逐链表位置 |
| `bpf_gpu_set_prefetch_region` | kfunc | 设置预取区域 |

**已证明的局限性**（来自实验数据）：

1. **驱逐只能排序，不能拒绝**：BPF 只能把 chunk 移到链表头/尾，无法阻止特定 chunk 被驱逐。结果：proactive prefetch 的数据可被立即驱逐（cross-block -28%）
2. **无 chunk 级元数据**：BPF 必须用外部 hash map (keyed by VA) 关联语义信息，hot path 下开销大且脆弱
3. **无 fault 通知**：BPF 不知道哪个地址发生了 fault，只能在驱逐时被动响应
4. **无内存压力信息**：BPF 无法查询当前空闲 chunk 数量，只能用 fault rate 间接估算（所有基于 fault rate throttle 的策略均失败）
5. **无迁移完成通知**：BPF 无法知道 `bpf_gpu_migrate_range()` 是否完成

这些局限导致了一个根本性问题：**当前的 BPF 接口只能做 reactive policy（fault 后调整），无法做 well-informed proactive policy**。打破这些限制需要精准的内核修改。

---

## 1. 研究方向一：驱逐否决权 — BPF-Controlled Eviction Veto

### 1.1 Problem Statement

当前的 UVM 驱逐流程是一个刚性管道：

```
pick_root_chunk_to_evict():
  1. gpu_evict_prepare(used_list, unused_list)  ← BPF 重排链表
  2. chunk = list_first(unused_list)             ← 取链表头
  3. if (!chunk) chunk = list_first(used_list)   ← 取链表头
  4. chunk_start_eviction(chunk)                  ← 不可逆，开始驱逐
```

BPF 可以在 step 1 重排链表，但**无法阻止 step 2/3 选中的特定 chunk 被驱逐**。一旦 chunk 在链表头，就是它的死期。

**直接后果**（已有实验证据）：

| 场景 | 问题 | 证据 |
|------|------|------|
| Proactive prefetch 后立即被驱逐 | 预取数据 DMA 完成后被 demand fault 挤掉，白白浪费 PCIe 带宽 | cross-block 73.6% 命中率但净有害（+8.9GB DMA） |
| T1 保护不够强 | `move_tail` 只是放到队尾，极端压力下仍会被驱逐 | cycle_moe T1 保护在 1.84x oversub 下差异 < 1% |

### 1.2 Novel Mechanism: gpu_evict_decide Hook

在 `pick_root_chunk_to_evict()` 的 victim selection 循环中插入一个**逐 chunk 的决策 hook**：

```c
/* uvm_pmm_gpu.c: pick_root_chunk_to_evict() 修改 */

// 当前代码：直接取链表头
// chunk = list_first_chunk(&pmm->root_chunks.va_block_used);

// 修改后：遍历链表，对每个候选 chunk 询问 BPF
list_for_each_entry(chunk, &pmm->root_chunks.va_block_used, list) {
    int decision = uvm_bpf_call_gpu_evict_decide(pmm, chunk);
    if (decision != UVM_BPF_EVICT_REJECT) {
        // BPF 接受或无意见 → 选定此 chunk
        break;
    }
    // BPF 否决 → 跳过，检查下一个
    chunk = NULL;  // 标记未找到
}
```

新增 hook 定义：

```c
/* bpf_testmod.h / uvm_bpf_struct_ops.c */
struct gpu_mem_ops {
    // ... 已有 hooks ...
    int (*gpu_evict_decide)(uvm_pmm_gpu_t *pmm,
                            uvm_gpu_chunk_t *chunk);  // NEW
};

// 返回值语义：
// UVM_BPF_ACTION_DEFAULT (0) = 同意驱逐（默认行为）
// UVM_BPF_EVICT_REJECT   (2) = 否决此 chunk，选下一个
```

**内核改动量**：
- `uvm_pmm_gpu.c`：~15 行（循环 + hook 调用）
- `uvm_bpf_struct_ops.c`：~20 行（wrapper 函数 + CFI stub）
- `uvm_bpf_struct_ops.h`：~3 行（声明 + `UVM_BPF_EVICT_REJECT` 常量）
- **总计：~38 行**

### 1.3 Novelty Claim

**"首个支持 BPF 逐 chunk 否决 GPU 显存驱逐的系统"**

与已有工作的区分：

| 维度 | 已有方案 | 本方案 |
|------|---------|-------|
| CPU 内存 | `madvise(MADV_SEQUENTIAL)` — 静态提示 | BPF 运行时动态决策 |
| GPU 驱逐 | 链表排序（`move_head/tail`）— 全局重排 | 逐 chunk 否决 — 精细控制 |
| BPF 能力 | "建议"型（排序） | "决策"型（accept/reject） |
| 保护强度 | `move_tail` 只是推迟，仍可被驱逐 | `REJECT` 是绝对保护（同一轮驱逐内） |

**学术创新点**：

1. **接口范式转换**：从 "BPF 建议排序 → 内核按排序执行" 到 "内核提议候选 → BPF 批准/否决"。这是 GPU 内存管理中首次引入 "veto-based policy" 模型。
2. **解决 prefetch-eviction 冲突**：通过否决刚预取的 chunk，直接修复 cross-block 的核心失败模式。
3. **实现语义保护**：应用通过 uprobe 设置 chunk 的"重要性标记"（attention score、expert 活跃状态），BPF 在驱逐时查标记决定是否否决。

### 1.4 BPF 策略示例

```c
/* extension/eviction_veto_semantic.bpf.c */

struct chunk_meta {
    u64 last_prefetch_ns;  // 最近预取时间
    u32 semantic_tier;     // 0=可驱逐, 1=保护中, 2=绝对保护
    u32 access_count;      // 自预取后的访问计数
};

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 16384);
    __type(key, u32);               // chunk_va >> 21
    __type(value, struct chunk_meta);
} chunk_meta_map SEC(".maps");

SEC("struct_ops/gpu_evict_decide")
int BPF_PROG(gpu_evict_decide, uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk)
{
    uvm_va_block_t *vb = BPF_CORE_READ(chunk, va_block);
    if (!vb) return 0;  // 无 VA block → 同意驱逐

    u64 va = BPF_CORE_READ(vb, start);
    u32 key = (u32)(va >> 21);
    struct chunk_meta *meta = bpf_map_lookup_elem(&chunk_meta_map, &key);
    if (!meta) return 0;  // 无元数据 → 同意

    // 策略 1：预取保护窗口
    // 刚预取的数据在 50ms 内不允许被驱逐
    u64 now = bpf_ktime_get_ns();
    if (meta->last_prefetch_ns > 0 &&
        (now - meta->last_prefetch_ns) < 50000000ULL) {
        return 2;  // UVM_BPF_EVICT_REJECT
    }

    // 策略 2：语义保护
    // attention score 标记为"高重要性"的 chunk 不驱逐
    if (meta->semantic_tier >= 2) {
        return 2;  // REJECT
    }

    return 0;  // 同意驱逐
}
```

### 1.5 Paper Story

> **Problem**: GPU proactive prefetch has 73.6% accuracy but is net-harmful because the eviction subsystem immediately reclaims prefetched data. Existing BPF can only reorder eviction lists but cannot protect individual chunks.
>
> **Contribution**: We introduce `gpu_evict_decide`, a per-chunk BPF veto hook in the UVM eviction path. Combined with semantic metadata, BPF policies can protect prefetched/important chunks from premature eviction.
>
> **Evaluation**: Across 4 GPU workloads (LLM inference, GNN, vector search, MoE), eviction veto transforms cross-block prefetch from -28% to +X%, and enables attention-score-based protection that reduces total DMA traffic by Y%.

### 1.6 实验设计

| 实验 | 配置 | 对比基准 | 指标 |
|------|------|---------|------|
| E1 | Cross-block + 无 veto | always_max | DMA 总量, tg, 预取浪费率 |
| E2 | Cross-block + 预取保护窗口 (10/25/50/100ms) | E1 | 同上 |
| E3 | MoE expert prefetch + veto | expert prefetch 无 veto | tg, expert fault count |
| E4 | Attention score tier + veto | cycle_moe 无 veto | tg, eviction distribution |
| E5 | 多 oversub 级别 (1.0x–2.0x) | 各自 baseline | 跨级别鲁棒性 |

---

## 2. 研究方向二：Chunk 级 BPF 元数据标签

### 2.1 Problem Statement

当前 BPF 策略需要为每个 chunk 关联语义信息（access count、attention score、expert ID、phase marker），但 `uvm_gpu_chunk_t` 结构体中没有 BPF 可用字段。

**当前 workaround**：使用 BPF hash map keyed by `chunk_va >> 21`

```c
// 每个 BPF 策略都要重复这段 boilerplate
uvm_va_block_t *vb = BPF_CORE_READ(chunk, va_block);
if (!vb) return 0;
u64 va = BPF_CORE_READ(vb, start);
u32 key = (u32)(va >> 21);
struct my_metadata *meta = bpf_map_lookup_elem(&meta_map, &key);
```

**问题**：
1. **性能**：hash lookup 在 `gpu_block_activate`（spinlock 持有期间）的 hot path 上
2. **正确性**：chunk 被 split/merge 后 VA 可能改变，hash map 条目过期
3. **内存**：每种策略维护独立的 hash map，浪费 BPF map 内存
4. **代码冗余**：每个 BPF 程序前 5 行都是 VA 查找 boilerplate

### 2.2 Novel Mechanism: Per-Chunk BPF Tag

在 `uvm_gpu_chunk_struct` 中新增一个 `u64 bpf_tag` 字段：

```c
/* uvm_pmm_gpu.h */
struct uvm_gpu_chunk_struct
{
    NvU64 address;
    struct { ... };  // bitfields
    struct list_head list;
    uvm_va_block_t *va_block;
    uvm_gpu_chunk_t *parent;
    uvm_pmm_gpu_chunk_suballoc_t *suballoc;
    u64 bpf_tag;  // NEW: 8 bytes, BPF-accessible metadata
};
```

新增 kfunc：

```c
/* uvm_bpf_struct_ops.c */
__bpf_kfunc void bpf_gpu_chunk_set_tag(uvm_gpu_chunk_t *chunk, u64 tag) {
    if (chunk)
        chunk->bpf_tag = tag;
}

__bpf_kfunc u64 bpf_gpu_chunk_get_tag(uvm_gpu_chunk_t *chunk) {
    return chunk ? chunk->bpf_tag : 0;
}
```

**内核改动量**：
- `uvm_pmm_gpu.h`：+1 行（字段定义）
- `uvm_pmm_gpu.c`：+1 行（初始化 `bpf_tag = 0`）
- `uvm_bpf_struct_ops.c`：+15 行（两个 kfunc + BTF 注册）
- **总计：~17 行**

### 2.3 Novelty Claim

**"首个为 GPU 物理内存块提供 BPF 可编程内联元数据的系统"**

| 维度 | 类比 | 本方案 |
|------|------|-------|
| CPU | `struct page::private` — 文件系统使用 | `uvm_gpu_chunk_t::bpf_tag` — BPF 策略使用 |
| 传统 GPU | 无可编程元数据 | O(1) 内联访问 |
| 当前 gpu_ext | 外部 BPF hash map | 内联 u64 字段 |

**学术创新点**：

1. **内核-BPF 共享状态**：`bpf_tag` 是内核数据结构中的 BPF 可写字段，打破了传统"内核结构只读"的 BPF 约束。这是一个新的 BPF 编程模型。
2. **O(1) 语义查询**：驱逐决策的 hot path 从 hash lookup 降为直接字段读取。
3. **生命周期绑定**：tag 与 chunk 生命周期绑定（分配时初始化为 0，释放时自动清除），比外部 map 更安全。
4. **自由编码**：64 bit 可编码丰富语义 — low 16 bits 存 attention score，mid 16 bits 存 expert ID，high 32 bits 存 timestamp。

### 2.4 使用场景

```c
/* 场景 1：attention score 驱逐 */
SEC("struct_ops/gpu_block_activate")
int BPF_PROG(gpu_block_activate, uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk, struct list_head *list)
{
    u64 tag = bpf_gpu_chunk_get_tag(chunk);
    u16 score = tag & 0xFFFF;          // low 16 bits = score
    u16 tier  = (tag >> 16) & 0xFF;    // bits 16-23 = tier

    if (tier == 2)  // HOT
        bpf_gpu_block_move_tail(chunk, list);
    else if (score < TRASH_THRESHOLD)  // TRASH
        bpf_gpu_block_move_head(chunk, list);
    // COOL: 默认 LRU 位置
    return 1;
}

/* 场景 2：prefetch 时间戳（配合 evict_decide veto） */
// 在 proactive prefetch 完成后设置 tag
SEC("uprobe.s/...")
int BPF_PROG(on_migrate_done, u64 va_start, u64 va_end, ...) {
    u64 now_ms = bpf_ktime_get_ns() / 1000000;
    // tag = (timestamp_ms << 32) | PREFETCH_FLAG
    // 后续 gpu_evict_decide 中直接读 tag 判断保护窗口
}
```

### 2.5 与 Direction 1 的协同

`bpf_tag` + `gpu_evict_decide` 组合形成完整的语义驱逐框架：

```
应用层 → uprobe → BPF 设置 chunk tag (score/tier/timestamp)
                                          ↓
内核驱逐路径 → gpu_evict_decide → BPF 读取 tag → 决定 accept/reject
```

这比外部 hash map 方案更高效（无 hash 开销）、更安全（无过期条目）、更简洁（无 boilerplate）。

---

## 3. 研究方向三：Fault 通知 Hook — 从被动响应到主动感知

### 3.1 Problem Statement

当前 BPF 的信息获取方式：

| 信息 | 获取方法 | 问题 |
|------|---------|------|
| 哪个 chunk 被激活 | `gpu_block_activate` hook | 只知道 chunk，不知道为什么被激活 |
| 当前是哪层/哪个 expert | VA 地址 → 离线映射表 | 需预计算 VA→layer 映射，脆弱 |
| fault 发生位置 | ❌ 无直接接口（只能用 kprobe hack） | kprobe 不稳定，依赖内部函数名 |
| fault 频率 | BPF map 手动计数 | 滞后，不精确 |

**核心问题**：BPF 不知道 "GPU 刚在哪个地址 fault 了"。这使得所有 fault-pattern-based 策略都必须依赖 kprobe — 一种 fragile、implementation-specific 的 hack。

### 3.2 Novel Mechanism: gpu_fault_notify Hook

在 fault 服务路径中插入通知 hook：

```c
/* 新增 hook 定义 */
struct gpu_mem_ops {
    // ... 已有 hooks ...
    int (*gpu_evict_decide)(...);       // Direction 1
    int (*gpu_fault_notify)(u64 fault_va,    // fault 的起始虚拟地址
                            u64 block_size,   // VA block 大小 (通常 2MB)
                            u32 pages_faulted, // 本次 batch 中的 fault 页数
                            u32 is_write);     // 是否为写访问
};
```

插入点（`uvm_gpu_replayable_faults.c`）：

```c
/* service_fault_batch_block_locked() 中，在确定 fault 区域后 */

// 现有代码：
block_context->region = uvm_va_block_region(first_page_index, last_page_index + 1);
status = uvm_va_block_service_locked(gpu->id, va_block, va_block_retry, block_context);

// 新增（在 service_locked 调用前或后）：
uvm_bpf_call_gpu_fault_notify(
    va_block->start,                                     // fault 所在 VA block
    va_block->end - va_block->start + 1,                 // block size
    last_page_index - first_page_index + 1,              // faulted pages
    /* is_write derived from access_type */
);
```

**内核改动量**：
- `uvm_gpu_replayable_faults.c`：~5 行（hook 调用）
- `uvm_bpf_struct_ops.c`：~20 行（wrapper + CFI stub + 成员注册）
- **总计：~25 行**

### 3.3 Novelty Claim

**"GPU 内存管理中首个 BPF 原生 fault 通知接口"**

| 对比 | CPU 系统 | GPU 系统（本方案） |
|------|---------|-----------------|
| 机制 | `userfaultfd` — 用户态 fault 通知 | `gpu_fault_notify` — BPF fault 通知 |
| 粒度 | 页级 (4KB) | VA block 级 (2MB) |
| 延迟 | 内核→用户态上下文切换 | 内核内 BPF 调用（~100ns） |
| 可编程性 | 固定 API | BPF 可编程策略 |

**学术创新点**：

1. **Fault 作为语义信号**：GPU fault 不仅是"需要数据"的事件，更是"推断未来访问模式"的信号。将 fault 信息暴露给 BPF，使其成为一等公民。
2. **消除 kprobe 依赖**：所有当前的 fault-aware 策略（如 `prefetch_proactive_layer.bpf.c`）都使用 kprobe hack。`gpu_fault_notify` 提供稳定、语义明确的接口。
3. **Enable 新策略族**：
   - **Fault stride detection**：检测连续 fault 的 VA 间距，推断访问模式（sequential/stride/random）
   - **Layer transition detection**：MoE 模型中，连续 fault 跨越 layer boundary 意味着 "进入下一层" → 触发 next-layer proactive migration
   - **Hot region identification**：高频 fault 的 VA range 应被优先保护

### 3.4 BPF 策略示例：Fault Pattern Learning

```c
/* extension/fault_pattern_prefetch.bpf.c */

struct fault_history {
    u64 last_fault_va[4];  // 最近 4 次 fault 的 VA
    u32 stride;            // 检测到的步长
    u32 confidence;        // 步长置信度
};

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, struct fault_history);
} fault_hist SEC(".maps");

SEC("struct_ops/gpu_fault_notify")
int BPF_PROG(gpu_fault_notify, u64 fault_va, u64 block_size,
             u32 pages_faulted, u32 is_write)
{
    u32 zero = 0;
    struct fault_history *h = bpf_map_lookup_elem(&fault_hist, &zero);
    if (!h) return 0;

    // 计算与上一次 fault 的步长
    s64 delta = (s64)(fault_va - h->last_fault_va[0]);

    // 如果连续 3 次步长一致 → 高置信度 stride
    if (delta == h->stride && delta != 0) {
        h->confidence++;
        if (h->confidence >= 3) {
            // 预测下一个 fault VA，提前迁移
            u64 predict_va = fault_va + delta;
            u64 *vs = bpf_map_lookup_elem(&va_space_cache, &zero);
            if (vs && *vs)
                bpf_gpu_migrate_range(*vs, predict_va, block_size);
        }
    } else {
        h->stride = (u32)delta;
        h->confidence = 1;
    }

    // 更新 history 环形缓冲
    h->last_fault_va[3] = h->last_fault_va[2];
    h->last_fault_va[2] = h->last_fault_va[1];
    h->last_fault_va[1] = h->last_fault_va[0];
    h->last_fault_va[0] = fault_va;

    return 0;
}
```

### 3.5 关键优势：可与 Direction 1 & 2 组合

```
Fault Notify → 检测 stride/layer transition → 触发 proactive migrate
                                                        ↓
                                              设置 chunk bpf_tag (方向 2)
                                                        ↓
                              gpu_evict_decide 读取 tag → 保护刚预取的数据 (方向 1)
```

三个方向形成闭环：**感知 → 预取 → 保护**。

---

## 4. 研究方向四：内存压力导出 — BPF 的全局视野

### 4.1 Problem Statement

**关键实验发现**：cross-block prefetch 在 <1.5x oversub 有效，>1.5x 有害。原因是高 oversub 下 proactive prefetch 与 demand fault 争抢 PCIe 带宽，形成"零和博弈"。

理想的策略需要知道当前内存压力：
- 有空闲 chunk → 激进预取（预取不会挤掉任何人）
- 接近满载 → 保守预取（每次预取都触发驱逐）
- 完全满载 → 停止预取（只做 demand 迁移）

**当前 BPF 无法获知内存压力**。`prefetch_throttled_xb.bpf.c` 尝试用 fault rate 作为代理指标，但所有基于 throttle 的策略均失败（L5/L6 实验 tg 下降 5-9%），证明 fault rate 是一个**滞后且不准确的压力指标**。

### 4.2 Novel Mechanism: PMM Statistics Export

新增一个 kfunc，直接从 PMM 结构读取精确的内存状态：

```c
/* uvm_bpf_struct_ops.c */

struct bpf_gpu_pmm_stats {
    u32 total_root_chunks;    // 总 root chunk 数
    u32 free_chunks;          // 当前空闲（含 zero 和 non-zero）
    u32 used_chunks;          // va_block_used 链表长度
    u32 unused_chunks;        // va_block_unused 链表长度
    u32 pinned_chunks;        // 不可驱逐的 chunk 数
};

__bpf_kfunc int bpf_gpu_get_pmm_stats(uvm_pmm_gpu_t *pmm,
                                       struct bpf_gpu_pmm_stats *out)
{
    if (!pmm || !out) return -EINVAL;

    // 读取链表长度（近似值，非精确锁定）
    out->total_root_chunks = pmm->root_chunks.count;
    out->free_chunks = /* count from free_list */;
    out->used_chunks = /* count or approximate from va_block_used */;
    // ...
    return 0;
}
```

**设计考量**：由于 hook 在 spinlock 下调用，不能遍历整个链表计数。方案：
- **方案 A**：在 `uvm_pmm_gpu_t` 中新增 `u32 used_count` / `free_count` 原子计数器，在 chunk 入队/出队时维护（~10 行额外改动）
- **方案 B**：使用 `pmm->pma_stats`（NVIDIA PMA 已有统计），wrap 成 kfunc
- **方案 C**：只暴露 `pma_stats` 的 free pages 比例（最小改动，~5 行）

推荐方案 A（计数器方式），改动最少且最精确。

**内核改动量**：
- `uvm_pmm_gpu.h`：+3 行（计数器字段）
- `uvm_pmm_gpu.c`：+6 行（入队/出队时更新计数器）
- `uvm_bpf_struct_ops.c`：+15 行（kfunc + struct 定义 + BTF 注册）
- **总计：~24 行**

### 4.3 Novelty Claim

**"首个向 BPF 导出实时 GPU 内存压力的系统，实现压力感知的自适应策略"**

**学术创新点**：

1. **从 "one-size-fits-all" 到 "continuous adaptation"**：不同 oversub 级别需要不同策略（实验已充分证明）。PMM 统计使 BPF 能在运行时自动切换。
2. **解决 "prefetch 零和博弈"**：当 `free_chunks > threshold` 时预取不会触发驱逐，此时 cross-block 是纯增益。BPF 可以用精确阈值而非模糊 heuristic。
3. **GPU 特有的压力模型**：CPU 有 `/proc/meminfo`、`PSI`、`kswapd`。GPU 内存管理完全没有等价物。本方案是首个 GPU 内存压力可观测框架。

### 4.4 策略示例：Pressure-Gated Cross-Block Prefetch

```c
SEC("struct_ops/gpu_block_activate")
int BPF_PROG(gpu_block_activate, uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk, struct list_head *list)
{
    struct bpf_gpu_pmm_stats stats = {};
    bpf_gpu_get_pmm_stats(pmm, &stats);

    u32 pressure = 100 - (stats.free_chunks * 100 / stats.total_root_chunks);
    // pressure: 0 = 全空闲, 100 = 完全满载

    if (pressure < 70) {
        // 低压力：激进预取 + cross-block
        trigger_xb_prefetch(chunk);
    } else if (pressure < 90) {
        // 中等压力：只做 always_max，禁止 cross-block
        // (避免预取挤掉有用数据)
    } else {
        // 高压力：保守模式，减少 prefetch 范围
    }

    // 标准 cycle_moe 驱逐排序...
    return 1;
}
```

### 4.5 Paper Story

> **Problem**: Optimal GPU prefetch strategy varies dramatically with memory pressure — aggressive prefetch helps at low oversubscription but hurts at high oversubscription. No existing mechanism allows BPF policies to observe memory pressure.
>
> **Insight**: Fault rate is a poor proxy for pressure (all 3 throttled strategies degraded performance). Direct PMM statistics are accurate and available at near-zero cost.
>
> **Contribution**: We add `bpf_gpu_get_pmm_stats()`, a lightweight kfunc exporting real-time GPU memory pressure to BPF. This enables pressure-gated prefetching that automatically adapts across oversubscription levels.
>
> **Result**: A single BPF policy using pressure-gated cross-block achieves +X% at 1.0x oversub (where XB helps), 0% degradation at 1.5x (where XB was previously -28%), and +Y% overall via MoE expert prefetch rate limiting.

---

## 5. 研究方向五：迁移完成回调 — 闭环控制

### 5.1 Problem Statement

`bpf_gpu_migrate_range()` 是异步的（通过 `bpf_wq`）。BPF 发起迁移后无法知道：
- 迁移是否成功完成
- 实际迁移了多少字节（部分页面可能已在 GPU，无需迁移）
- 迁移花了多长时间（DMA 延迟）

这使得 BPF 无法实现：
- **Pipeline 控制**：expert N 的预取完成后才能开始 expert N+1
- **DMA 带宽估算**：需要 bytes/time 来做 rate limiting
- **预取效果评估**：需要知道预取是否被驱逐覆盖（预取完成 → 立即被驱逐 = 浪费）

### 5.2 Novel Mechanism: gpu_migrate_complete Hook

```c
/* 新增 hook */
struct gpu_mem_ops {
    // ... 已有 hooks ...
    int (*gpu_migrate_complete)(u64 va_start,       // 迁移的 VA 范围
                                u64 length,
                                u32 direction,       // 0=H2D (prefetch), 1=D2H (evict)
                                u64 bytes_migrated,  // 实际迁移字节数
                                u64 elapsed_ns);     // 迁移耗时 (纳秒)
};
```

插入点（`uvm_va_block.c` 或 `uvm_migrate.c`）：

```c
/* 在 uvm_migrate() 或 uvm_va_block_make_resident_read_duplicate() 完成后 */

// 在 uvm_migrate_bpf() 中添加：
NV_STATUS uvm_migrate_bpf(uvm_va_space_t *va_space, NvU64 base, NvU64 length)
{
    u64 start_ns = ktime_get_ns();
    NV_STATUS status = uvm_migrate(va_space, NULL, base, length, ...);
    u64 elapsed = ktime_get_ns() - start_ns;

    if (status == NV_OK) {
        uvm_bpf_call_gpu_migrate_complete(base, length,
                                           0,  // H2D
                                           length,  // 近似
                                           elapsed);
    }
    return status;
}
```

**内核改动量**：
- `uvm_migrate.c`：~8 行（计时 + hook 调用）
- `uvm_bpf_struct_ops.c`：~20 行（wrapper + CFI stub）
- **总计：~28 行**

### 5.3 Novelty Claim

**"首个为 BPF 提供 GPU DMA 迁移完成回调的系统，实现闭环 prefetch-compute pipeline"**

**学术创新点**：

1. **闭环控制**：当前所有 GPU prefetch 都是 open-loop（发出迁移请求后不知道结果）。`gpu_migrate_complete` 实现 closed-loop 控制。
2. **DMA 带宽建模**：BPF 可以实时计算 `bytes / elapsed_ns`，构建 PCIe 带宽利用率模型。当利用率接近饱和时自动降低预取激进度。
3. **Prefetch pipeline depth 控制**：类似 TCP 的拥塞窗口 — 维护一个 "in-flight prefetch" 计数器，完成回调减少计数，新预取增加计数，超过窗口大小时暂停。

### 5.4 策略示例：TCP-like Prefetch Window

```c
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, struct { u32 in_flight; u32 window_size; u64 total_bytes; });
} prefetch_ctrl SEC(".maps");

SEC("struct_ops/gpu_migrate_complete")
int BPF_PROG(migrate_complete, u64 va_start, u64 length,
             u32 direction, u64 bytes_migrated, u64 elapsed_ns)
{
    if (direction != 0) return 0;  // 只关心 H2D (prefetch)

    u32 zero = 0;
    struct { u32 in_flight; u32 window_size; u64 total_bytes; } *ctrl =
        bpf_map_lookup_elem(&prefetch_ctrl, &zero);
    if (!ctrl) return 0;

    // 迁移完成，减少 in-flight 计数
    if (ctrl->in_flight > 0)
        ctrl->in_flight--;
    ctrl->total_bytes += bytes_migrated;

    // 自适应窗口：类似 TCP AIMD
    // 如果 DMA 完成很快（带宽充裕）→ 增大窗口
    // 如果 DMA 完成很慢（带宽竞争）→ 缩小窗口
    u64 bandwidth_mbps = bytes_migrated * 1000 / elapsed_ns;  // MB/s
    if (bandwidth_mbps > 10000)  // > 10 GB/s
        ctrl->window_size = min(ctrl->window_size + 1, 8u);
    else if (bandwidth_mbps < 2000)  // < 2 GB/s
        ctrl->window_size = max(ctrl->window_size / 2, 1u);

    return 0;
}

// 在发起新 prefetch 前检查窗口
SEC("uprobe.s/...")
int BPF_PROG(maybe_prefetch, u64 addr, u64 len) {
    u32 zero = 0;
    auto *ctrl = bpf_map_lookup_elem(&prefetch_ctrl, &zero);
    if (!ctrl || ctrl->in_flight >= ctrl->window_size)
        return 0;  // 窗口满，暂停预取

    ctrl->in_flight++;
    bpf_gpu_migrate_range(cached_vs, addr, len);
    return 0;
}
```

---

## 6. 综合论文框架

### 6.1 独立发表 vs 组合发表

| 方案 | 包含方向 | 论文类型 | 适合会议 |
|------|---------|---------|---------|
| **Paper A** | 1 (veto) + 2 (tag) + 3 (fault notify) | 系统论文（新接口 + 新策略族） | OSDI/SOSP/ATC |
| **Paper B** | 1 (veto) + 4 (pressure) | 聚焦论文（prefetch-eviction coordination） | EuroSys/ASPLOS |
| **Paper C** | 3 (fault notify) + 5 (migrate complete) | 机制论文（闭环 GPU 内存管理） | NSDI/ATC |
| **Paper D** | 全部 5 个 | 完整系统论文 | OSDI/SOSP |

### 6.2 推荐组合：Paper A — "语义感知的 GPU 内存管理"

**Title**: _"BPF-Mediated Semantic GPU Memory Management: From Reactive Eviction to Proactive Protection"_

**核心叙事**：

> GPU 内存管理面临一个根本矛盾：**内核掌控物理内存但缺乏应用语义，应用拥有语义但无法影响内存决策**。
>
> 我们通过三个小幅内核修改（共 ~80 行 C 代码）打破这个矛盾：
>
> 1. **Eviction Veto**（~38 行）：BPF 可以否决特定 chunk 的驱逐，实现语义保护
> 2. **Chunk Tag**（~17 行）：BPF 可以在 chunk 上附加元数据，实现 O(1) 语义查询
> 3. **Fault Notification**（~25 行）：BPF 实时感知 fault 位置，实现模式学习
>
> 组合起来，这三个机制实现了一个新的 GPU 内存管理范式：
> **感知 → 标记 → 预取 → 保护**
>
> ```
> Fault Notify → BPF 学习访问模式 → 设置 chunk tag (score/tier)
>                                         ↓
>             Proactive Migrate → chunk 到达 GPU → tag 标记为 "刚预取"
>                                                        ↓
>             Evict Decide → 读取 tag → 否决刚预取的/高分的 chunk
> ```

**Paper 结构**：

```
§1 Introduction: GPU 内存过量使用越来越普遍 (LLM > VRAM)
§2 Background: UVM 驱逐模型 + BPF struct_ops
§3 Motivation:
   - 15 种算法实验证明 eviction ordering 差异 <1%
   - Cross-block prefetch 73.6% 准确率但净有害
   - 根因：缺乏语义感知 + prefetch-eviction 不协调
§4 Design:
   §4.1 Chunk Tag: 内联 BPF 元数据
   §4.2 Eviction Veto: 逐 chunk 决策 hook
   §4.3 Fault Notify: 实时 fault 通知
   §4.4 组合策略: 感知-预取-保护闭环
§5 Implementation: 内核修改详解 (~80 行 C)
§6 Evaluation:
   §6.1 Microbenchmark: hook overhead, tag vs hash map latency
   §6.2 LLM Inference (llama.cpp 120B MoE, vLLM 30B MoE)
   §6.3 GNN Training (PyTorch Geometric)
   §6.4 Vector Search (FAISS)
   §6.5 Cross-oversub robustness (1.0x-2.0x)
   §6.6 Ablation: 每个机制的独立贡献
§7 Related Work
§8 Conclusion
```

### 6.3 与已有 gpu_ext 工作的关系

| gpu_ext 已有贡献 | 本文新增 | 区别 |
|-----------------|---------|------|
| BPF struct_ops 框架 | 不变（基础设施） | 本文使用并扩展它 |
| 6 个 hook + 5 个 kfunc | +3 个 hook + 3 个 kfunc | 新增接口面 |
| `move_head/tail`（排序） | `evict_decide`（否决） | 范式升级 |
| 外部 BPF hash map | 内联 `bpf_tag` | 性能 + 安全性 |
| kprobe 获取 fault 信息 | `gpu_fault_notify` hook | 稳定性 + 语义 |
| always_max + cycle_moe | 语义驱逐 + 保护 + stride prefetch | 策略族升级 |

---

## 7. 内核修改总览

| 方向 | 修改文件 | 改动行数 | 修改性质 |
|------|---------|---------|---------|
| 1. Eviction Veto | `uvm_pmm_gpu.c`, `uvm_bpf_struct_ops.c/h` | ~38 行 | 新增 hook + 修改驱逐循环 |
| 2. Chunk Tag | `uvm_pmm_gpu.h/c`, `uvm_bpf_struct_ops.c` | ~17 行 | 新增字段 + 两个 kfunc |
| 3. Fault Notify | `uvm_gpu_replayable_faults.c`, `uvm_bpf_struct_ops.c` | ~25 行 | 新增 hook |
| 4. PMM Stats | `uvm_pmm_gpu.h/c`, `uvm_bpf_struct_ops.c` | ~24 行 | 新增计数器 + kfunc |
| 5. Migrate Complete | `uvm_migrate.c`, `uvm_bpf_struct_ops.c` | ~28 行 | 新增 hook |
| **合计** | | **~132 行** | 5 个独立 patch |

每个 patch 独立、可测试、向后兼容（不改变默认行为，BPF 未加载时所有 hook 返回 DEFAULT）。

---

## 8. 优先级与实施顺序

| 优先级 | 方向 | 论文价值 | 实验验证难度 | 推荐 |
|--------|------|---------|------------|------|
| **P0** | 1. Eviction Veto | ★★★★★ 最高（新范式） | 中（需实现 BPF 策略） | **必做** |
| **P0** | 2. Chunk Tag | ★★★★ 高（新抽象） | 低（纯接口扩展） | **必做** |
| **P1** | 3. Fault Notify | ★★★★ 高（新能力） | 中（需 pattern learning 策略） | **强烈推荐** |
| **P1** | 4. PMM Stats | ★★★ 中（实用性强） | 低（直接使用） | **推荐** |
| **P2** | 5. Migrate Complete | ★★★ 中（闭环控制） | 高（需 pipeline 策略调优） | **可选** |

### 实施计划

```
Week 1: 内核修改 (全部 5 个 patch)
  Day 1-2: Eviction Veto hook 实现 + 基础测试
  Day 3:   Chunk Tag 字段 + kfunc
  Day 4:   Fault Notify hook
  Day 5:   PMM Stats + Migrate Complete

Week 2: BPF 策略实现
  Day 1-2: Veto + Tag 联合策略（semantic eviction）
  Day 3:   Fault Pattern Learning 策略
  Day 4:   Pressure-Gated Prefetch 策略
  Day 5:   集成测试

Week 3-4: 实验评估
  4 个 workload × 5 个 oversub 级别 × 3 个策略配置
  Microbenchmark: hook overhead, tag vs hash 延迟
  Ablation study: 每个机制的独立贡献
```

---

## 9. 与已有学术工作的区分

| 已有工作 | 方法 | 本方案的区别 |
|---------|------|------------|
| FlexSM (EuroSys'23) | GPU 内存分层管理 | 无 BPF 可编程性，无应用语义 |
| G-Swap (ATC'20) | GPU 内存压缩+交换 | 无运行时策略切换，无 fault 感知 |
| Dragon (ASPLOS'22) | GPU-NVMe 分层 | 针对 SSD 层级，无 VRAM 内策略编程 |
| NVIDIA UVM | Demand paging + 固定 heuristic | 不可编程，无语义感知 |
| gpu_ext (base) | BPF struct_ops 排序 | **本文扩展**：veto + tag + fault notify |
| vAttention (SOSP'24) | CUDA VMM KV cache | 不处理 oversub，不涉及驱逐策略 |
| MSched | DMA-Compute overlap scheduler | 用户态调度，不涉及内核驱逐/预取策略 |

**本方案的独特定位**：

> 在 GPU 内存管理领域，首次实现了**应用语义 → 内核策略 → BPF 可编程**的完整闭环。
> 通过 132 行内核修改（5 个独立 patch），将 GPU UVM 从"固定策略的黑盒"
> 转变为"语义感知的可编程内存子系统"。

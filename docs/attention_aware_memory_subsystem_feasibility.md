# Attention-Aware GPU Memory Subsystem: 可行性评估

## 1. 执行摘要

**结论：架构整体方向可行，但需要对原始设计做重大调整。**

| 组件 | 可行性 | 说明 |
|------|--------|------|
| GPU 端 Attention Score 采集 | ✅ 可行 | 需修改 vLLM PagedAttention kernel |
| Score 从 GPU 传递到 BPF | ⚠️ 部分可行 | 需要用户态桥接层，无法直接 GPU→BPF |
| Level 3: 热页保护 / Swap to Host | ✅ 已实现 | gpu_ext 现有 T1 保护策略即此功能 |
| Level 1: 低分页丢弃并返回零值 | ❌ 原始方案不可行 | 需降级为"优先驱逐" + 应用层重算 |
| Level 2: VRAM 内压缩（逆投影） | ❌ 原始方案不可行 | 驱逐路径在内核态，无法发起 GPU 计算 |
| vAttention CUDA VMM 驱动层 | ⚠️ 需要新建 | 当前系统基于 UVM，非 VMM API |

---

## 2. 现有 gpu_ext 基础设施分析

### 2.1 已有能力

gpu_ext 是一套**打补丁的 NVIDIA 开源内核模块** + **eBPF struct_ops 策略框架**，提供以下可编程点：

**UVM 内存管理钩子 (`gpu_mem_ops`)**：
- `gpu_page_prefetch` / `gpu_page_prefetch_iter` — 缺页时决定预取范围（单 VA block 2MB 内）
- `gpu_block_activate` — chunk 进入 evictable 状态时触发（**主要策略入口**）
- `gpu_block_access` — chunk 被访问时触发（**已知 bug：从未被调用**）
- `gpu_evict_prepare` — 驱逐准备阶段

**可用 kfunc**：
- `bpf_gpu_block_move_head/tail` — 调整 chunk 在驱逐链表中的位置
- `bpf_gpu_set_prefetch_region` — 设置预取区域
- `bpf_gpu_migrate_range` — 异步跨 VA block 迁移（通过 `bpf_wq`）

**GPU 调度钩子 (`nv_gpu_sched_ops`)**：
- TSG 创建/绑定/销毁时的调度策略（timeslice、抢占等）

**vLLM UVM 集成**：
- 使用 `cudaMallocManaged` 实现显存超卖
- PyTorch `CUDAPluggableAllocator` 接口
- 环境变量 `VLLM_USE_UVM` 控制开关

### 2.2 已有的相关策略（与提案部分重叠）

| 已有策略 | 与提案的关系 |
|---------|-------------|
| `prefetch_always_max_cycle_moe.bpf.c` | T1 频率保护 ≈ Level 3 热页保护 |
| `prefetch_template_belady.bpf.c` | Belady 循环距离驱逐 ≈ 基于"未来使用距离"的驱逐 |
| `prefetch_max_mru_expert.bpf.c` | T1 保护 + MRU expert 驱逐 ≈ attention 层 vs expert 层分级 |
| `prefetch_moe_expert.bpf.c` | fault bitmap replay ≈ 基于历史模式的预取 |
| `prefetch_vllm_phase.bpf.c` | prefill/decode 阶段检测 ≈ 运行时 phase 感知 |

**关键发现**：提案中的 Level 3（热页保护/Swap to Host）已经在 gpu_ext 中以多种形式实现。真正的新贡献在 Level 1 和 Level 2，但这两个恰恰是最难实现的。

---

## 3. 逐步骤可行性分析

### 3.1 步骤 A：语义信息采集与传递

#### A1. GPU 端 Attention Score 计算 — ✅ 可行

**目标**：在 Attention Kernel 执行时，累计每个 KV Block 的 Attention Score。

**实现路径**：

```
vLLM PagedAttention Kernel (CUDA)
    ↓ 修改 kernel，每次 attention 后累加 score
    ↓
Score Buffer (cudaMallocManaged, UVM 共享地址)
    ↓ 用户态 daemon 定期读取
    ↓
BPF Map (HASH: page_id → score)
    ↓ BPF 策略在 eviction 时查询
    ↓
驱逐决策
```

**具体修改点**：

1. **修改 `paged_attention_v1.cu` / `paged_attention_v2.cu`**：
   - 在 softmax attention 计算后，对每个 KV block 的 attention weights 求和
   - 使用 `atomicAdd` 将 score 累加到 per-block score buffer
   - 类似 H2O (Heavy Hitter Oracle) 的方法

2. **Score Buffer 设计**：
   ```c
   // UVM 共享 buffer，GPU 写、CPU 读
   struct kv_block_score {
       float cumulative_score;    // 累计 attention score
       uint32_t access_count;     // 被 attend 的次数
       uint32_t last_step;        // 最后一次被 attend 的 decode step
   };
   // 按 KV block ID 索引，大小 = max_num_blocks
   ```

3. **开销评估**：
   - 额外 atomicAdd：每个 attention head 每次 decode step 对每个 KV block 一次
   - 对于 Llama-70B (80 heads, 4096 blocks)：~320K atomicAdd/step
   - 预估性能开销 < 2%（attention 本身的计算量远大于此）

**风险**：
- FlashAttention 的 online softmax 实现中，weight 不会被完整物化（只算 local max + sum），需要在 tiled 计算中额外跟踪
- 可能需要使用 FlashAttention 的 `logsumexp` 输出作为近似 score

#### A2. Score 从 GPU 传递到 BPF — ⚠️ 需要用户态桥接

**核心问题**：BPF 运行在 Linux 内核态（UVM 内核线程上下文），无法直接读取用户态进程映射的 CUDA UVM 地址空间。GPU 也无法直接写入 BPF map。

**不可行的方案**：
- ❌ GPU 直接写 BPF map（GPU 不能执行 `bpf_map_update_elem` 系统调用）
- ❌ BPF 直接读 UVM 地址（struct_ops 运行在 UVM 内核线程，无用户态 mm 上下文）
- ❌ 添加 kfunc 读取 UVM 页面（需要持有 `va_space` 锁，在 eviction 路径上有死锁风险）

**可行方案：用户态 Score Bridge Daemon**

```
┌─────────────────────────────────────────────────────────────┐
│                        GPU                                  │
│  PagedAttention Kernel → atomicAdd → Score Buffer (UVM)     │
└──────────────────────────┬──────────────────────────────────┘
                           │ cudaMemcpy / direct read (UVM coherent)
┌──────────────────────────▼──────────────────────────────────┐
│              Score Bridge Daemon (用户态)                     │
│  1. 每 N 个 decode step，读取 Score Buffer                    │
│  2. 归一化 + 分级 (Hot/Cool/Trash)                            │
│  3. 写入 BPF map: bpf_map_update_elem(score_map, &page, &s) │
│  周期：~10ms (每 decode step 结束时触发)                       │
└──────────────────────────┬──────────────────────────────────┘
                           │ BPF map (kernel-visible)
┌──────────────────────────▼──────────────────────────────────┐
│              BPF Eviction Policy (内核态)                     │
│  gpu_block_activate:                                         │
│    score = bpf_map_lookup_elem(&score_map, &chunk_va);       │
│    if (score < TRASH_THRESHOLD) move_head(chunk);            │
│    elif (score > HOT_THRESHOLD) move_tail(chunk);            │
└─────────────────────────────────────────────────────────────┘
```

**实现复杂度**：中等。核心是一个 Python/C 用户态 daemon + 一个修改过的 BPF 策略文件。

**延迟分析**：
- Score 传递延迟 = daemon 轮询周期（~10ms）
- 对于 LLM decode（每 step 30-100ms），一个 step 的延迟是可接受的
- Score 的"新鲜度"足够：KV block 的重要性变化缓慢（一个 token 不会剧烈改变累计 score）

#### A3. 替代方案：无需 GPU 端修改的近似方法

如果不想修改 PagedAttention kernel，可以用以下近似方法：

1. **基于访问频率的近似**（已有）：gpu_ext 的 `access_counts` per-CPU array 已经跟踪 chunk 激活频率。T1 chunks（attention 层权重）天然高频。

2. **基于 VA 地址的层级推断**（已有）：`prefetch_template_belady.bpf.c` 已实现 VA→layer 映射。Attention 层的 KV cache 在 VA 空间中有固定模式。

3. **基于 token 位置的重要性估计**：
   - StreamingLLM 发现：第一个 token（attention sink）+ 最近的 window 最重要
   - 不需要实际计算 attention score，只需要知道 KV block 对应的 token 位置
   - 位置信息可以从 vLLM 的 block table 中获取

**推荐**：先用方案 3（基于 token 位置的启发式）验证概念，再投入 GPU kernel 修改。

---

### 3.2 步骤 B：语义感知的缺页处理

#### Level 3 (Hot: 保留/Swap to Host) — ✅ 已实现

这就是 gpu_ext 现有的 T1 保护机制：

```c
// 来自 prefetch_always_max_cycle_moe.bpf.c
if (c + 1 >= T1_FREQ_THRESHOLD) {
    bpf_gpu_block_move_tail(chunk, list);  // 保护：移到链表尾部
    return 1;
}
```

- 高频 chunk 被保护在驱逐链表尾部
- 如果最终必须驱逐，UVM 自动将数据迁移到 Host RAM
- 重新访问时 UVM page fault 触发迁移回 GPU

**改进方向**：将 "频率 >= 3" 的硬编码阈值替换为基于 Attention Score 的动态阈值。



## 5. 可实现的方案设计

基于以上分析，以下是一个**实际可行**的实现方案，保留原始架构的核心思想，但适配 gpu_ext 的技术约束。

### 5.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│  vLLM + Modified PagedAttention (GPU)                           │
│  ┌─────────────────────┐  ┌─────────────────────────────────┐   │
│  │ PagedAttention CUDA  │  │ Score Buffer (UVM shared)       │   │
│  │ + score accumulation │→ │ [block_id] → cumulative_score   │   │
│  └─────────────────────┘  └────────────┬────────────────────┘   │
└────────────────────────────────────────┼────────────────────────┘
                                         │
┌────────────────────────────────────────▼────────────────────────┐
│  Score Bridge Daemon (用户态)                                    │
│  ┌──────────────┐   ┌───────────────┐   ┌────────────────────┐  │
│  │ Score Reader  │ → │ Score Ranker   │ → │ BPF Map Updater   │  │
│  │ (每 decode    │   │ Hot/Cool/Trash │   │ score_map,        │  │
│  │  step 同步)   │   │ 分级           │   │ config_map        │  │
│  └──────────────┘   └───────────────┘   └────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │ Pressure Monitor (读 BPF ringbuf / eviction 计数)        │    │
│  │ → 触发 Proactive Compressor (可选)                        │    │
│  └──────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────┬────────────────────────┘
                                         │ BPF maps
┌────────────────────────────────────────▼────────────────────────┐
│  BPF Policy: attention_aware_eviction.bpf.c (内核态)             │
│                                                                  │
│  gpu_block_activate:                                             │
│    1. 查询 score_map → 获取 chunk 的 attention score             │
│    2. score < TRASH_THRESHOLD  → move_head (优先驱逐)            │
│    3. score > HOT_THRESHOLD    → move_tail (保护)                │
│    4. 中间分数 → 不移动 (被动 LRU)                                │
│                                                                  │
│  gpu_page_prefetch:                                              │
│    always_max (整个 VA block 预取)                                │
│                                                                  │
│  (可选) gpu_evict_prepare:                                       │
│    通过 ringbuf 通知用户态 eviction 事件                          │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 实现分阶段计划

#### Phase 1: Score-Aware Eviction (2-3 周)

**目标**：用 Attention Score 指导驱逐顺序，无需修改 GPU kernel。

1. **近似 Score 来源**：利用 vLLM 已有的 block table 信息
   - 从 vLLM 的 `BlockSpaceManager` 获取每个 block 的 token 位置
   - 使用 StreamingLLM 启发式：`score(block) = f(recency, is_sink)`
   - 通过用户态 daemon 定期写入 BPF map

2. **新建 BPF 策略**：`extension/attention_aware_eviction.bpf.c`
   - score_map: `HASH<u32 page_id, struct block_score>`
   - 在 `gpu_block_activate` 中查询 score 做分级驱逐
   - 基于 `prefetch_always_max_cycle_moe.bpf.c` 模板

3. **用户态 Daemon**：`workloads/vllm/score_bridge.py`
   - 连接 vLLM 的 block manager
   - 定期更新 BPF map

**交付物**：
- `extension/attention_aware_eviction.bpf.c`
- `workloads/vllm/score_bridge.py`
- 实验脚本 + 基准对比

#### Phase 2: GPU-Side Score Collection (3-4 周)

**目标**：修改 PagedAttention kernel，采集真实 Attention Score。

1. **修改 `paged_attention_v1.cu`**：
   - 在 attention 计算后，累加每个 KV block 的 score
   - 写入 UVM 共享 score buffer

2. **升级 Score Bridge**：
   - 从 UVM score buffer 直接读取（替代 vLLM Python 层的近似值）
   - 更精确的 Hot/Cool/Trash 分级

3. **对比实验**：Phase 1 启发式 vs Phase 2 真实 score 的效果差异
-----------------------
#### Phase 3: Proactive Compression (可选, 4-6 周)

**目标**：在内存压力上升时，主动压缩低 score 的 KV blocks。

1. **KV Cache Quantization**（推荐先做）：
   - FP16 KV → INT4 KV，4x 压缩
   - 在用户态 CUDA context 中执行
   - 解压路径：dequantize on re-access

2. **逆投影压缩**（高难度，可选）：
   - KV → Hidden States (2x 压缩)
   - 需要缓存投影矩阵 W_K, W_V
   - 解压 = 重新投影

3. **压力触发器**：
   - BPF `gpu_evict_prepare` 通过 ringbuf 通知用户态
   - 或监控 eviction 频率（daemon 读取 BPF 计数器）

### 5.3 关键实现细节

#### BPF Score Map 设计

```c
// 在 BPF 策略中定义
struct block_score {
    u16 attention_score;  // 归一化到 [0, 65535]
    u8  tier;             // 0=TRASH, 1=COOL, 2=HOT
    u8  flags;            // bit0: is_sink, bit1: is_recent_window
};

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 65536);      // 支持 128GB VA / 2MB chunk
    __type(key, u32);                // chunk VA >> 21 (2MB aligned)
    __type(value, struct block_score);
} score_map SEC(".maps");
```

#### VA → Score 查询路径

```c
SEC("struct_ops/gpu_block_activate")
int BPF_PROG(gpu_block_activate, uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk, struct list_head *list)
{
    // 获取 chunk 的 VA
    uvm_va_block_t *va_block = BPF_CORE_READ(chunk, va_block);
    if (!va_block) return 0;
    u64 chunk_va = BPF_CORE_READ(va_block, start);
    
    // 查询 score
    u32 page_id = (u32)(chunk_va >> 21);
    struct block_score *bs = bpf_map_lookup_elem(&score_map, &page_id);
    
    if (!bs) {
        // 未知 chunk（非 KV cache，如模型权重）→ 用频率保护
        // ... 频率计数逻辑（同 cycle_moe）...
        return 0;
    }
    
    switch (bs->tier) {
    case 0: // TRASH
        bpf_gpu_block_move_head(chunk, list);  // 优先驱逐
        break;
    case 2: // HOT
        bpf_gpu_block_move_tail(chunk, list);  // 保护
        break;
    default: // COOL
        break;  // 被动 LRU
    }
    return 1;
}
```

#### Score Bridge Daemon 核心逻辑

```python
import ctypes
import struct
import time
from bcc import BPF  # 或直接操作 /sys/fs/bpf/

class ScoreBridge:
    def __init__(self, vllm_engine, bpf_map_fd):
        self.engine = vllm_engine
        self.map_fd = bpf_map_fd
        self.trash_threshold = 0.01   # bottom 1% score
        self.hot_threshold = 0.5      # top 50% score
    
    def update_scores(self):
        """每个 decode step 结束后调用"""
        block_scores = self.collect_scores()
        
        # 分位数分级
        scores_sorted = sorted(block_scores.values())
        if len(scores_sorted) > 0:
            trash_cutoff = scores_sorted[int(len(scores_sorted) * 0.1)]
            hot_cutoff = scores_sorted[int(len(scores_sorted) * 0.5)]
        
        for block_id, score in block_scores.items():
            chunk_va_page = block_id  # 需要 block_id → VA 的映射
            tier = 1  # COOL
            if score < trash_cutoff:
                tier = 0  # TRASH
            elif score > hot_cutoff:
                tier = 2  # HOT
            
            self.update_bpf_map(chunk_va_page, score, tier)
    
    def collect_scores(self):
        """从 vLLM block manager 或 GPU score buffer 收集"""
        # Phase 1: 基于 token 位置的启发式
        # Phase 2: 读取 GPU score buffer
        pass
```

---

## 6. 与原始提案的差异总结

| 原始提案 | 本方案 | 原因 |
|---------|--------|------|
| GPU 直接写 BPF map | GPU → UVM buffer → 用户态 → BPF map | GPU 无法执行 BPF 系统调用 |
| 缺页时 BPF 查询 Score | activate 时查询 Score | `gpu_block_access` bug 未被调用；activate 是唯一可靠入口 |
| 丢弃页面返回零值 | 优先驱逐 + 应用层 recompute | UVM 不支持零值返回；静默错误风险 |
| VRAM 内逆投影压缩 | 主动量化压缩（用户态 CUDA） | eviction 路径无法发起 GPU 计算 |
| vAttention CUDA VMM 驱动 | 继续使用 UVM | VMM 不走 UVM 路径，BPF 钩子不触发 |
| 缺页中断时实时处理 | activate 时预排序 + 主动压缩 | 内核态 eviction 必须快速完成 |

---

## 7. 风险与缓解

### 7.1 技术风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| Score map 查询延迟影响 eviction 性能 | HASH map 在高 eviction 压力下可能超时 (Xid 31) | 使用 PERCPU_ARRAY 替代 HASH；或限制 map 大小 |
| Score 信息滞后（daemon 未及时更新） | 错误的驱逐决策 | 设置保守阈值；未知 chunk 默认保护 |
| `move_head` 导致 Xid 31 FAULT_PDE | 新 chunk 在页表建立前被驱逐 | 已知问题：需要在 activate 中加入延迟计数（先不 move_head，等几次 activate 后再移动） |
| 修改 PagedAttention kernel 破坏 FlashAttention 优化 | 性能回退 | Phase 1 先用启发式验证概念 |

### 7.2 研究风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| Attention Score 不是 future importance 的好预测器 | 驱逐决策不比随机好 | 对比实验：Score-based vs 频率-based vs 随机 |
| "无损降级"不可能实现 | 任何驱逐都有延迟代价 | 目标调整为"最小化 tail latency 增加" |
| 压缩率不足以实质性缓解内存压力 | KV cache 只占总显存的一部分 | 量化 + 选择性驱逐组合使用 |

---

## 8. 结论

### 可以做的（推荐优先级排序）

1. **Score-guided eviction ordering**（Phase 1）：最容易实现，直接利用 gpu_ext 现有框架。预期 1-2 周可出原型。

2. **GPU-side score collection**（Phase 2）：技术上可行但工程量大，需修改 vLLM CUDA kernel。值得投入，因为提供了比频率计数更精确的语义信号。

3. **Proactive KV quantization**（Phase 3 简化版）：在内存压力上升时，主动将低 score KV blocks 从 FP16 量化到 INT4。比逆投影简单得多。

### 不能做的（需要放弃或根本性修改）

1. **内核层丢弃并返回零值**：UVM 架构不支持，且存在静默错误风险。

2. **驱逐路径中的 GPU 计算（压缩/逆投影）**：内核态无法发起 CUDA kernel。

3. **基于 CUDA VMM 的驱动层**：与 gpu_ext 的 UVM BPF 钩子不兼容。

### 最终建议

先实现 Phase 1（2-3 周），用启发式 score + 现有 BPF 框架验证"语义感知驱逐"的价值。如果效果显著，再投入 Phase 2 的 GPU kernel 修改。Phase 3 的压缩功能作为可选增强。

整个系统的核心价值不在于"无损"（这在物理意义上不可能），而在于**用语义信息最大化有限显存的效用**——确保最有价值的 KV blocks 最后被驱逐，从而将性能降级的"坡度"变得尽可能平缓。

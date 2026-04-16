# vLLM: 基于地址真值的 KV Cache / Weights 分流策略设计（Prefetch + Eviction）

状态: 设计文档（不包含实现）  
最后更新: 2026-04-14  
来源: `my_plan.md`（“用 vLLM 侧地址真值区分 KVCache 与权重，分而治之”）

---

## 0. 摘要

在 vLLM 的 UVM（Unified Virtual Memory）模式下，KV cache 与 model weights 共享 UVM 地址空间，UVM driver 的 prefetch/eviction 若采取“一刀切”策略，容易出现两类数据互相驱逐、tail latency 失控的问题。

本设计的核心是: **放弃通过 block 热度/访问模式“推断类型”**，改为 **由 vLLM 直接上报 KV cache 与 weights 的地址真值**。eBPF 侧只做 O(1) 地址分类，从而将策略拆成两条彼此独立的 fast-path:

- **KV cache**: 允许更激进的跨 VA-block 预取（cross-block prefetch），以降低 block 边界 stall。
- **Weights**: 以稳定为先，主要依赖 intra-block（2MB 内）预取，并在 eviction 中优先保护，避免被 KV 挤出显存导致反复 fault-back。

---

## 1. 目标与非目标

### 1.1 目标

- **G1. 类型识别稳定**: KV/Weights 分类不依赖热度阈值，直接基于 vLLM 上报的地址真值。
- **G2. 热路径开销低**: struct_ops 热路径仅做 `map lookup` + 少量分支，不做复杂推断。
- **G3. 最小可用闭环**: 先实现“KV vs Weights 分流”的最小策略闭环，再逐步细化（KV 的 age/recency、phase gating 等）。
- **G4. 与 gpu_ext 既有机制一致**:
  - 使用 `uvm_perf_prefetch_get_hint_va_block` kprobe 缓存 VA block 上下文（参考 [`../../../extension/prefetch_vllm_phase.bpf.c`](../../../extension/prefetch_vllm_phase.bpf.c)）。
  - 使用 `bpf_wq` + `bpf_gpu_migrate_range()` 做跨 VA block 预取（参考 [`../../cross_block_prefetch_plan.md`](../../cross_block_prefetch_plan.md)）。
  - eviction 逻辑主要写在 `gpu_block_activate`（`gpu_block_access` 在当前实现中不可用，见 [`../../../extension/README.md`](../../../extension/README.md)）。

### 1.2 非目标（第一版不做）

- 不实现“按 token 真实 attention score”级别的 KV 语义驱逐（这属于另一个方向，参考 [`../../../extension/attention_aware_eviction.bpf.c`](../../../extension/attention_aware_eviction.bpf.c)）。
- 不在 BPF 侧做大范围区间扫描或复杂数据结构（避免 verifier/性能风险）。
- 不在本设计里强绑定“UVM allocator 的预算/advise 策略”细节（分类只依赖地址真值，不依赖 `cudaMemAdvise` 推断）。

---

## 2. gpu_ext 侧约束（必须遵守）

这些约束直接决定了我们为什么选择“地址分流 + O(1) map lookup”的方案:

- **C1. 驱逐热路径放在 `gpu_block_activate`**: `gpu_block_access` 在当前框架中实际不会被调用，因此“基于访问更新”的策略不可依赖它（详见 [`../../../extension/README.md`](../../../extension/README.md)）。
- **C2. `move_head` 有风险**: 已知 `move_head` 可能导致 Xid 31 等问题，第一版 eviction 以 `move_tail`（保护）为主，避免激进“处决”。
- **C3. 指针算术受限**: BPF verifier 禁止对指针做算术；所有地址处理尽量在用户态对齐到 2MB 后再上报。
- **C4. struct_ops hot-path overhead 有上限**: 任何 per-fault 的复杂 loop/推断都容易超预算。

---

## 3. 总体架构

### 3.1 控制面与数据面分离

- **控制面（vLLM -> eBPF）**: vLLM 在已知语义点（weights load 完成、KV cache 初始化完成）上报地址集合。
- **数据面（UVM struct_ops）**: 对每次 fault/activate 事件，仅通过地址查询得到数据类型并执行分流策略。

### 3.2 语义通道（推荐: uprobe）

推荐将“hint ABI”放在 `uvm_allocator.abi3.so` 这类稳定可控的动态库里，通过 uprobe 捕获并写入 BPF map。理由:

- 已有先例: `uvm_set_phase(int)`（phase gating）在 [`../../../extension/prefetch_vllm_phase.bpf.c`](../../../extension/prefetch_vllm_phase.bpf.c) 中已采用 uprobe 方案。
- uprobe 运行于用户进程上下文，`bpf_get_current_pid_tgid()` 可直接获得真实 `tgid`，便于做 per-process key。

备选通道（不推荐但可用）:

- 用户态 daemon 直接更新 pinned BPF map（类似 `score_bridge.py` 写 `attention_score_map` 的模式），但需要额外进程与权限管理。

---

## 4. 地址分类的数据模型

### 4.1 Region Kind（类型）

最小集合建议三类:

- `REGION_UNKNOWN = 0`
- `REGION_WEIGHTS = 1`
- `REGION_KV = 2`

可扩展:

- `REGION_OTHER = 3`（workspace/临时 buffer）
- `REGION_INDEX = 4`（例如 Faiss index）

### 4.2 规范化到 2MB VA block（推荐在用户态完成）

UVM 的关键粒度是 **2MB VA block**:

- `VA_BLOCK_SIZE = 2MB = 1 << 21`
- `va_block_start = addr & ~(VA_BLOCK_SIZE - 1)`

用户态上报建议以 `va_block_start` 为单位（逐 block 上报），避免 BPF 侧做对齐与大范围循环。

---

## 5. BPF Map 设计（路径 A: 逐 2MB block 分类表）

### 5.1 `region_block_map`（核心）

用途: `tgid + 2MB va_block_start -> kind`，数据面 O(1) 查询。

Key:

```c
struct region_key {
    __u32 tgid;
    __u32 _pad;
    __u64 va_block_start;  // 2MB aligned UVA
};
```

Value:

```c
struct region_val {
    __u8  kind;     // KV/WEIGHTS/...
    __u8  flags;    // optional
    __u16 reserved;
    __u32 epoch;    // optional, 用于清理/防 PID 重用污染
};
```

Map 类型:

- `BPF_MAP_TYPE_HASH`
- `max_entries` 建议默认 `65536`（可覆盖 ~128GB/2MB 量级，同时留足碎片空间）

### 5.2 `region_epoch_map`（可选）

用途: 防止 PID 重用或模型 reload 导致旧条目残留。

- key: `tgid`
- value: `epoch`

写入策略:

- vLLM 每次初始化前先 `uvm_hint_begin(epoch++)`，BPF 记录该 epoch。
- `region_block_map` 写入时带 epoch；数据面查到 epoch 不匹配则视为 miss。

### 5.3 `region_stats`（建议）

最少统计:

- 命中: `hit_kv`, `hit_weights`, `miss`
- prefetch: `kv_xb_sched`, `weight_xb_skip`, `migrate_ok`, `migrate_fail`
- eviction: `protect_weight_tail`, `kv_default`, `unknown_default`

---

## 6. vLLM 侧: 地址真值如何获取与上报

> 这里只描述“在哪里拿地址、如何上报”的设计点，不要求当前代码已有实现。

### 6.1 KV cache 地址提取（确定性强）

KV cache 分配发生在 `initialize_kv_cache_tensors()`（见 vLLM UVM fork: [`../../../workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py`](../../../workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py)）。

做法:

- 遍历 `kv_caches.values()` 拿到 tensor（或 tensor list）。
- 对 `data_ptr()` 做去重（避免 shared KV cache 多次上报）。
- `addr = tensor.data_ptr()`
- `len = tensor.numel() * tensor.element_size()`
- 在用户态将 `[addr, addr+len)` 展开为 2MB blocks，逐 block 上报 `REGION_KV`。

### 6.2 Weights 地址提取（两种方式）

Weights 的“地址真值”也能由 vLLM 在用户态提取，但工程实现有两条路线:

- **W1（易实现）**: model load 完成后遍历 `named_parameters()` + 必要 buffers，按 storage 去重后上报为 `REGION_WEIGHTS`。
- **W2（更推荐，条目更可控）**: 在 allocator 层做“tagged allocation tracking”（weights 阶段 vs kv 阶段），只记录大块 allocation 并上报。该路线与“应用-内核语义通道”方向一致（参考 [`../../future_work_kernel_system.md`](../../future_work_kernel_system.md) §6）。

第一版建议先做 W1，验证闭环与收益后再收敛到 W2。

---

## 7. 语义通道 ABI（建议）

建议在 `uvm_allocator.abi3.so` 中导出最小 ABI（函数体可为空，仅供 uprobe 捕获参数）:

```c
// kind: 1=WEIGHTS, 2=KV
void uvm_hint_block(int kind, uint64_t va_block_start);

// 可选: 生命周期
void uvm_hint_begin(uint32_t epoch);
void uvm_hint_end(uint32_t epoch);
```

说明:

- `uvm_hint_block()` 强制调用方传入 2MB 对齐地址，避免 BPF 侧做对齐/展开循环。
- `epoch` 机制用于避免残留污染，不是必需项，但建议预留。

---

## 8. 数据面策略: Prefetch 分流（第一版）

核心原则:

- **所有类型**: intra-block 统一采用 `always_max`（2MB 内全量预取），作为稳定基线。
- **KV**: 允许 cross-block prefetch（例如 direction-aware / adjacent-stride），降低 block 边界 stall。
- **Weights**: 禁用 cross-block（避免污染与 displacement），只用 intra-block。

实现建议:

1. kprobe `uvm_perf_prefetch_get_hint_va_block` 缓存 `va_start/va_end/va_space/owner_tgid`（模式可参考 [`../../../extension/prefetch_vllm_phase.bpf.c`](../../../extension/prefetch_vllm_phase.bpf.c) 与 [`../../../extension/trace_helper.h`](../../../extension/trace_helper.h)）。
2. `gpu_page_prefetch()`:
   - `result_region = max_prefetch_region`（always_max）
   - 查 `region_block_map[(owner_tgid, va_start)]`
   - `kind==KV` 时触发 cross-block: `bpf_wq` + `bpf_gpu_migrate_range(va_space, next_block, len)`

跨 block prefetch 的具体模式可复用既有实现（direction history、dedup、rate-limit）:

- 参考: [`../../../extension/prefetch_vllm_phase.bpf.c`](../../../extension/prefetch_vllm_phase.bpf.c)
- 机制文档: [`../../cross_block_prefetch_mechanism.md`](../../cross_block_prefetch_mechanism.md)

---

## 9. 数据面策略: Eviction 分流（第一版）

核心目标: **Weights 优先保留，KV 不与之争抢**。

在 `gpu_block_activate()` 中:

- `REGION_WEIGHTS`: 强保护（`move_tail`），减少 weights 被挤出导致的反复 fault-back。
- `REGION_KV`: 弱保护或不保护（保持默认顺序），作为弹性工作集。
- `REGION_UNKNOWN`: fallback 到当前稳定组合（例如 cycle_moe 的 T1 保护），或直接 `return 0` 走默认。

注意:

- 第一版不引入 `move_head`，避免触发已知风险。
- 若未来要做“KV 最近性”驱逐，可结合上报的 KV block 元数据（token age / request id / sliding window），但不属于最小闭环。

---

## 10. 回退策略与失败模式

- **uprobe 未 attach / vLLM 未上报**: `region_miss` 全部走基线组合（例如 `prefetch_always_max_cycle_moe`）。
- **只上报 KV**: `KV` 命中走 KV 策略，其余视为 `UNKNOWN`（保守）。
- **map 容量不足**: 用户态统计溢出并报警，临时退回“区间上报 + 小数组匹配”（需要严格 bounded）。
- **PID 重用污染**: 使用 epoch 机制或 loader 侧清理旧条目。

---

## 11. 观测与验证建议

最小可观测闭环:

- region 命中率: `hit_kv/hit_weights/miss`
- prefetch 分流: `kv_xb_sched/weight_xb_skip/migrate_ok/migrate_fail`
- eviction 分流: `protect_weight_tail/kv_default/unknown_default`

性能指标建议沿用 vLLM benchmark:

- TTFT / TPOT / throughput（参见 [`../../../workloads/vllm/README.md`](../../../workloads/vllm/README.md)）

---

## 12. 实施里程碑（规划）

> 这里只给工程拆分顺序，避免一次性引入过多变量。本文不包含实现。

1. 打通语义通道（`uvm_hint_block` -> uprobe -> `region_block_map`）。
2. 只做 prefetch 分流（KV cross-block，weights 禁用 cross-block）。
3. 再做 eviction 分流（weights move_tail 保护，KV 默认）。
4. 引入 phase gating（可选，与 `uvm_set_phase` 机制兼容）。
5. 迭代 weights 上报方式（W1 -> W2），降低上报条目与启动成本。

---

## 13. 相关参考

- Cross-block 预取计划: [`../../cross_block_prefetch_plan.md`](../../cross_block_prefetch_plan.md)
- KV 生命周期区分设想: [`../../future_work_kernel_system.md`](../../future_work_kernel_system.md) §6
- 现有 vLLM phase uprobe 策略: [`../../../extension/prefetch_vllm_phase.bpf.c`](../../../extension/prefetch_vllm_phase.bpf.c)
- 基线策略与 hook 约束: [`../../../extension/README.md`](../../../extension/README.md)


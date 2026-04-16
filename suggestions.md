# gpu_ext Policy Suggestions（按 Workload / 目标快速选型）

状态: 指南（持续更新）  
最后更新: 2026-04-14

本文件用于把“该用哪个 eviction/prefetch/scheduling 策略”这件事，整理成一个可执行的选择入口。策略实现代码位于 `extension/`，此处只给建议与入口链接。

> 重要背景: 当前 gpu_ext 的 UVM hook 约束决定了很多策略的实现方式。尤其是 `gpu_block_access` 在现状中不可靠，eviction 策略应主要放在 `gpu_block_activate`（详见 [`../../../extension/README.md`](../../../extension/README.md)）。

---

## 1. 一句话默认推荐（不知道选什么就用它）

如果你没有明确的 workload 特征、或者只是想要一个稳定强基线:

- **Prefetch + Eviction 组合**: `prefetch_always_max_cycle_moe`
  - 代码: [`../../../extension/prefetch_always_max_cycle_moe.bpf.c`](../../../extension/prefetch_always_max_cycle_moe.bpf.c)
  - Loader: `extension/prefetch_always_max_cycle_moe`

原因: `always_max`（intra-block 2MB 全量预取）是已验证的最大单点收益；`cycle_moe` 的 T1 保护在多 workload 上稳定，且符合 `move_tail` 约束。

---

## 2. 按 Workload 选型

### 2.1 vLLM（UVM Serving / KV-heavy）

优先顺序建议:

1. 基线: `prefetch_always_max_cycle_moe`（先确保整体不退化）
2. Phase-aware（prefill/decode 分流）:
   - `prefetch_vllm_phase_transparent`（无需修改 vLLM 源码，靠 paged_attention uprobe）
     - BPF: [`../../../extension/prefetch_vllm_phase_transparent.bpf.c`](../../../extension/prefetch_vllm_phase_transparent.bpf.c)
     - Loader: [`../../../extension/prefetch_vllm_phase_transparent.c`](../../../extension/prefetch_vllm_phase_transparent.c)
   - 或 `prefetch_vllm_phase`（依赖 `uvm_set_phase()` 语义通道）
     - BPF: [`../../../extension/prefetch_vllm_phase.bpf.c`](../../../extension/prefetch_vllm_phase.bpf.c)
     - Loader: [`../../../extension/prefetch_vllm_phase.c`](../../../extension/prefetch_vllm_phase.c)
3. **KV / Weights 地址分流（推荐的下一步方向）**:
   - 设计文档: [`vllm_kv_weight_address_driven_policy.md`](vllm_kv_weight_address_driven_policy.md)
   - 目标: 用 vLLM 上报的地址真值区分 KV 与 weights，仅对 KV 启用 cross-block prefetch，并在 eviction 中优先保护 weights。

相关背景与计划:

- Cross-block vLLM 讨论: [`../../cross_block_prefetch_plan.md`](../../cross_block_prefetch_plan.md)
- KV 生命周期设想: [`../../future_work_kernel_system.md`](../../future_work_kernel_system.md) §6

### 2.2 llama.cpp（MoE expert weights offload）

建议:

- 基线: `prefetch_always_max_cycle_moe`
- 若需要更激进的主动预取/复用:
  - `prefetch_moe_expert`（bitmap replay 思路）
    - BPF: [`../../../extension/prefetch_moe_expert.bpf.c`](../../../extension/prefetch_moe_expert.bpf.c)
    - Loader: [`../../../extension/prefetch_moe_expert.c`](../../../extension/prefetch_moe_expert.c)
- 若需要跨 VA block 预取:
  - `prefetch_cross_block_v2` / `prefetch_stride_multiblock`（先在 microbench 验证再上真实 workload）

### 2.3 PyTorch GNN（epoch scan / 大 feature tensor）

建议:

- 基线: `prefetch_always_max_cycle_moe`
- 主动预取（分配后立即搬运）:
  - `prefetch_gnn_proactive`
    - BPF: [`../../../extension/prefetch_gnn_proactive.bpf.c`](../../../extension/prefetch_gnn_proactive.bpf.c)
    - Loader: [`../../../extension/prefetch_gnn_proactive.c`](../../../extension/prefetch_gnn_proactive.c)

### 2.4 FAISS（build vs search phase）

建议:

- 基线: `prefetch_always_max_cycle_moe`
- phase-aware:
  - `prefetch_faiss_phase`（heuristic: stride window）
  - `prefetch_faiss_uprobe`（更精确: uprobe 标记 search）

---

## 3. 按优化目标选型

- **吞吐优先（bulk transfer, sequential-ish）**:
  - 更偏 `always_max` / cross-block 预取（谨慎控制 decode/随机访问阶段）
- **Tail latency 优先（serving）**:
  - 先用 `prefetch_serving_adaptive` 做 fault-rate 门控，再叠加 phase/region 分流
- **多租户公平性（multi-tenant）**:
  - eviction: `eviction_pid_quota` / `eviction_freq_pid_decay`
  - prefetch: `prefetch_pid_tree`

---

## 4. 进一步阅读

- 策略全景与开发路径: [`../POLICY_OVERVIEW.md`](../POLICY_OVERVIEW.md)
- Hook/驱动约束（必读）: [`../../../extension/README.md`](../../../extension/README.md)
- vLLM benchmark setup: [`../../../workloads/vllm/README.md`](../../../workloads/vllm/README.md)


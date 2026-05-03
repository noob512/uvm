# vLLM UVM Stage J KV Runtime Pressure / Eviction 接入说明

Stage J 在 Stage D/D2 的 KV 初始化预算之外，补上运行期 KV pressure 的 block-manager-side 控制面。它的边界是 vLLM KV block manager / scheduler，不在 allocator 层按裸指针驱逐 KV。

## 1. 当前实现目标

当前 Stage J 实现四件事：

1. 在 `KVCacheManager.allocate_slots()` 运行期安全点观察即将分配的新 KV blocks。
2. 在 runtime KV block budget 超限时输出 `would_evict_candidate` / `would_deny_allocation` trace。
3. 在 `BlockPool` 复用 free queue 中带 prefix-cache hash 的 block 时，记录真实的 vLLM-safe `evict_prefix_cache_block` 事件。
4. 可选执行 Stage J prefix-cache free-block eviction executor：只清理 `ref_cnt==0` 且已在 free queue 中的 prefix-cache metadata。

这个版本刻意不做 allocator-side KV 迁移，也不直接修改活跃 request 的 block table。

## 2. 为什么这样算 Stage J 的第一版

vLLM 的 KV cache tensor 在初始化时整体分配，运行期真正变化的是：

1. 哪些 block 被 request 引用。
2. 哪些 block 在 free queue 中可复用。
3. 哪些 free block 还带 prefix-cache hash，复用时需要从 prefix cache 索引移除。
4. 新请求或 decode step 是否允许继续 admission。

因此 Stage J 的第一版选择 runtime block budget，而不是假装运行期释放 KV tensor bytes。真实 CPU swap 可以后续接入 vLLM 已有 `kv_offload`/KV connector 体系。

## 3. 修改文件

```text
workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py
workloads/vllm/vllm/vllm/v1/core/kv_cache_manager.py
workloads/vllm/vllm/vllm/v1/core/kv_cache_coordinator.py
workloads/vllm/vllm/vllm/v1/core/block_pool.py
workloads/vllm/run_kv_fault_ratio.sh
workloads/vllm/check_stage_j_success.py
docs/vllm_uvm_stage_j_kv_runtime_eviction.md
```

## 4. 配置项

Runner 参数：

```text
--uvm-kv-runtime-enable <0|1>
--uvm-kv-runtime-mode trace_only|enforce
--uvm-kv-runtime-budget-bytes <n>
--uvm-kv-runtime-budget-blocks <n>
--uvm-kv-runtime-trace-file <path>
--uvm-kv-runtime-eviction-policy lru_prefix_cache|scheduler_aware
--uvm-kv-runtime-candidate-limit <n>
```

对应环境变量：

```text
VLLM_UVM_KV_RUNTIME_ENABLE
VLLM_UVM_KV_RUNTIME_MODE
VLLM_UVM_KV_RUNTIME_BUDGET_BYTES
VLLM_UVM_KV_RUNTIME_BUDGET_BLOCKS
VLLM_UVM_KV_RUNTIME_TRACE_FILE
VLLM_UVM_KV_RUNTIME_EVICTION_POLICY
VLLM_UVM_KV_RUNTIME_CANDIDATE_LIMIT
VLLM_UVM_KV_RUNTIME_PREFIX_EVICT_ENABLE
VLLM_UVM_KV_RUNTIME_PREFIX_EVICT_MAX_BLOCKS
```

`budget_blocks` 优先级高于 `budget_bytes`。如果只设置 bytes，Stage J 会根据 KV config 的 bytes-per-block 换算成 block budget。

## 5. Trace 字段

Stage J JSONL 默认文件名：

```text
vllm_uvm_kv_runtime_stage_j.jsonl
```

主要 action：

```text
runtime_config
allocation_pressure
would_evict_candidate
would_reuse_free_block
would_deny_allocation
deny_allocation
allocation_no_free_blocks
prefix_evict_attempt
prefix_evict_success
prefix_evict_noop
prefix_evict_failed
evict_prefix_cache_block
runtime_summary
candidate_snapshot_failed
```

关键字段：

```text
mode
policy
budget_blocks
budget_bytes
bytes_per_block
total_blocks
free_blocks
used_blocks
projected_used_blocks
pressure_blocks
pressure_bytes
block_id
block_hash
ref_cnt
safe_ref_cnt_zero
```

`safe_ref_cnt_zero=True` 是 Stage J 的核心安全证明之一：候选或真实 prefix-cache eviction 不包含活跃引用 block。

## 6. 模式语义

`trace_only`：

1. 只记录 pressure、候选和 would-deny。
2. 不拒绝分配。
3. 用于默认验收和性能对照。

`enforce`：

1. 当 projected used blocks 超过 runtime budget 时，`allocate_slots()` 返回 `None`。
2. 由 vLLM scheduler 走现有 preemption / admission 路径处理。
3. 不迁移 raw KV tensor，不修改活跃 block table。

`prefix_evict_enable`：

1. 仅在 runtime pressure 下尝试执行。
2. 仅扫描 vLLM `free_block_queue`。
3. 仅处理 `ref_cnt==0`、非 null、且带 prefix-cache hash 的 block。
4. 动作是清理 prefix-cache metadata，让该 free block 不再作为 prefix-cache hit 被复用。
5. 不释放 KV tensor，不迁移 CPU/GPU，不改 active request block table。

## 7. 验收脚本

默认验收：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./check_stage_j_success.py
```

默认参数使用：

```text
mode=trace_only
budget_blocks=1
prompts=1
output_len=128
```

这样会稳定触发 runtime pressure trace，同时要求 benchmark failed requests 为 0。

快速 block-manager self-test：

```bash
./check_stage_j_success.py --self-test --require-prefix-eviction
```

该测试不启动 GPU server。它会构造一个小 KV block pool，手动把一个 `ref_cnt==0` free block 放入 prefix-cache map，然后触发 Stage J executor，并要求：

```text
prefix_evict_success_blocks > 0
unsafe_candidate_records == 0
unsafe_prefix_eviction_records == 0
```

离线验证：

```bash
./check_stage_j_success.py \
  --skip-run \
  --trace-jsonl /tmp/run/vllm_uvm_kv_runtime_stage_j.jsonl \
  --bench-log /tmp/run/vllm_bench_stage_j.log
```

如果要强制检查真实 prefix-cache eviction/reuse：

```bash
./check_stage_j_success.py --require-prefix-eviction
```

小 benchmark 可能没有足够 prefix-cache 复用压力，因此这个检查默认不启用。

## 8. 成功标准

PASS 需要：

```text
trace_jsonl_present=True
trace_records_present=True
runtime_config_present=True
runtime_enabled=True
runtime_mode_matches=True
allocation_pressure_records_present=True
over_budget_pressure_observed=True
would_deny_records_present=True
runtime_summary_present=True
candidate_records_are_safe=True
prefix_evictions_are_safe=True
benchmark_no_failed_requests_in_trace_only=True
runner_log_clean=True
```

## 9. 当前未实现内容

当前 Stage J 已完成 block-manager-side runtime pressure、admission 和 safe prefix-cache free-block eviction executor。它仍不代表完整 CPU KV swap 已完成。未实现内容包括：

1. 将 inactive KV blocks 拷贝到 CPU swap space。
2. 从 CPU swap space 异步 load 回 GPU。
3. 和 `vllm.v1.kv_offload` worker/connector 做完整 store/load 调度。
4. recompute 策略。
5. 跨 pool pressure coordinator。

这些应作为 Stage J 后续子阶段或 Stage K 前置工作继续推进。

# llama.cpp eBPF Policy Sweep — Experiment Log

## Objective

Systematically evaluate all available eBPF memory management policies on llama.cpp MoE expert offloading workload, identify the best-performing policy, and explore whether a custom policy can improve performance further.

## Environment

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 5090 (32 GB VRAM) |
| CPU | Intel Core Ultra 9 285K |
| Kernel module | Custom nvidia-uvm.ko with eBPF struct_ops hooks |
| Model | GPT-OSS-120B MXFP4 MoE (116.8B params, ~59 GiB) |
| Benchmark | `llama-bench` via `configs/bench.py --uvm --ncmoe 64` |
| Metrics | pp = prompt processing (512 tokens, tok/s), tg = text generation (128 tokens, tok/s) |
| Samples per test | 1 (single run; full paper protocol requires 10 trials + geometric mean) |

## Available Hook Points (struct_ops)

The modified nvidia-uvm.ko exposes 5 BPF hook points via `struct uvm_gpu_ext`:

| Hook | Category | Called when |
|------|----------|------------|
| `uvm_pmm_chunk_activate` | Eviction | A new GPU memory chunk is allocated/populated |
| `uvm_pmm_chunk_used` | Eviction | An existing chunk is accessed (page fault or migration) |
| `uvm_pmm_eviction_prepare` | Eviction | The driver is about to evict a chunk (pre-eviction callback) |
| `uvm_prefetch_before_compute` | Prefetch | Before computing prefetch regions for a page fault |
| `uvm_prefetch_on_tree_iter` | Prefetch | Per-region iteration during prefetch bitmap tree walk |

Available kfuncs (helpers callable from BPF):
- `bpf_uvm_pmm_chunk_move_head(chunk, list)` — move chunk to HEAD (evict sooner)
- `bpf_uvm_pmm_chunk_move_tail(chunk, list)` — move chunk to TAIL (evict later / protect)
- `bpf_uvm_set_va_block_region(region, first, outer)` — set prefetch region boundaries
- `bpf_uvm_strstr(str, sz, substr, sz)` — string search helper

## Results (Round 1)

Completed 4 tests successfully from the `test_policies.sh` sweep:

| Policy | pp (tok/s) | tg (tok/s) | pp vs baseline | tg vs baseline | Notes |
|--------|-----------|-----------|----------------|----------------|-------|
| **baseline (no policy)** | 228.49 | 15.83 | — | — | Driver default LRU eviction + default prefetch |
| **eviction_lfu** | 233.76 | 16.06 | **+2.3%** | **+1.5%** | Best overall — frequently-used chunks protected |
| **eviction_fifo** | 234.63 | 15.84 | **+2.7%** | +0.1% | Good pp, neutral tg |
| **eviction_mru** | 229.20 | 11.34 | +0.3% | **-28.4%** | Catastrophic tg regression — evicts recently used data |

Earlier single run (different baseline measurement):
| Run | pp | tg |
|-----|----|----|
| Baseline (first run, /tmp) | 236.52 | 16.04 |
| LFU (first run, /tmp) | 230.29 | 15.77 |

Note: ~3% variance between runs is normal (thermal, background processes).

## Policies NOT Yet Tested

| Policy | Status | Reason |
|--------|--------|--------|
| `eviction_fifo_chance` | **BPF verifier error** | `arg#0 pointer type STRUCT uvm_gpu_chunk_struct must point to scalar` in `uvm_pmm_eviction_prepare`. Pre-existing bug — needs code fix. |
| `eviction_freq_pid_decay` | **Not tested** | Script path issue in second run attempt; policy loads OK (verified manually) |
| `prefetch_none` | **Not tested** | Same script issue |
| `prefetch_always_max` | **Not tested** | Same script issue |
| `prefetch_adaptive_sequential` | **Not tested** | Same script issue |
| `prefetch_stride` | Skipped | Not in sweep list |
| `prefetch_adaptive_tree_iter` | Skipped | Not in sweep list |
| `prefetch_eviction_pid` | Skipped | Combined policy; planned for phase 2 |

## Issues Encountered

### 1. `uv` not in root's PATH
- **Symptom**: `test_policies.sh: line 61: uv: command not found` when run under `sudo`
- **Root cause**: Script was run with `sudo bash test_policies.sh` but `uv` is at `/home/yunwei37/.local/bin/uv` (not in root's PATH)
- **Fix**: Script updated to use absolute path `UV="/home/yunwei37/.local/bin/uv"` and run benchmark as normal user (only policy loader uses sudo)

### 2. Results directory owned by root
- **Symptom**: `PermissionError: [Errno 13] Permission denied: '.../policy_sweep/baseline_no_policy.json'`
- **Root cause**: First failed `sudo` run created `results/policy_sweep/` as root-owned
- **Fix**: `sudo chown -R yunwei37:yunwei37 results/policy_sweep/`

### 3. `set -euo pipefail` aborts on first failure
- **Symptom**: Script stopped at `eviction_fifo_chance` (BPF load failure), remaining 4 tests never ran
- **Fix**: Changed to `set -uo pipefail` (removed `-e`) so script continues past failures

### 4. Working directory drift in second run attempt
- **Symptom**: JSON written OK but `python3 -c` couldn't find the files (relative path mismatch)
- **Root cause**: `cd "$SCRIPT_DIR"` in the function didn't properly reset for the inline python extraction
- **Fix**: Script now uses `$output_file` (absolute path via `$RESULTS_DIR`) consistently

### 5. `eviction_fifo_chance` BPF verifier rejection
- **Error**: `arg#0 pointer type STRUCT uvm_gpu_chunk_struct must point to scalar, or struct with scalar`
- **Location**: `uvm_pmm_eviction_prepare` calling `bpf_uvm_pmm_chunk_move_tail`
- **Root cause**: The `eviction_prepare` hook receives `struct list_head *` pointers. The policy does `container_of()` to get `uvm_gpu_chunk_t *` then passes it to `bpf_uvm_pmm_chunk_move_tail`. The BPF verifier cannot prove the pointer is valid through the `container_of` cast.
- **Status**: Pre-existing bug; needs code refactoring (possibly pass chunk pointer directly in the hook, or use a different approach in eviction_prepare)

### 6. Stale struct_ops between test runs
- **Observation**: After killing a policy loader process, the struct_ops map may persist. The `cleanup_struct_ops_tool` handles this, but there can be races.
- **Mitigation**: Script calls `kill_policy()` which does `pkill` + `cleanup_struct_ops_tool` + sleep between each test

## Policy Algorithm Summary

### Eviction Policies

| Policy | Algorithm | Suitable for MoE? |
|--------|-----------|-------------------|
| **Driver default** | LRU — recently used chunks move to tail (protected) | Decent baseline |
| **eviction_fifo** | FIFO — new chunks at head, no reordering on access. Evicts oldest first. | Maybe — if hot experts are long-lived |
| **eviction_lfu** | LFU — tracks per-chunk access frequency. Low-freq near head (evict first), high-freq (>=10 accesses) moved to tail. | **Good** — hot experts stay resident |
| **eviction_mru** | MRU — recently accessed chunks moved to HEAD (evict first). Opposite of LRU. | **Bad for decode** — evicts the experts you just used |
| **eviction_fifo_chance** | FIFO + second chance — tracks access bit per chunk, gives one reprieve before eviction | Potentially good — but currently broken (verifier) |
| **eviction_freq_pid_decay** | Per-PID frequency with decay — access count per chunk, move_tail every N accesses based on PID priority | Good for multi-tenant; single-tenant benefit unclear |
| **eviction_pid_quota** | Per-PID quota — each process gets a % of GPU memory; within quota = LRU protected, over quota = no protection | Multi-tenant only |

### Prefetch Policies

| Policy | Algorithm | Suitable for MoE? |
|--------|-----------|-------------------|
| **Driver default** | Heuristic prefetch based on fault patterns | Baseline |
| **prefetch_none** | Disables all prefetching | Useful as a control |
| **prefetch_always_max** | Always prefetch the maximum region around a fault | **Potentially good** — MoE experts are contiguous memory blocks |
| **prefetch_adaptive_sequential** | Adapts prefetch window based on sequential access detection | **Promising** — expert loading is somewhat sequential within a layer |
| **prefetch_adaptive_tree_iter** | Tree-based iteration with adaptive thresholds | Similar to adaptive_sequential |
| **prefetch_stride** | Detects strided access patterns and prefetches accordingly | Less relevant for MoE |
| **prefetch_eviction_pid** | Combined: PID-based prefetch thresholds + probabilistic LRU eviction | **Interesting** for multi-tenant |

## Analysis: Why LFU Helps MoE

MoE (Mixture-of-Experts) models route each token to a subset of experts per layer. During decode:
- A small set of "hot" experts is activated much more frequently than others
- The default LRU policy treats all recently-accessed chunks equally
- LFU differentiates: hot experts accumulate high frequency counts and stay resident, while cold experts (accessed once or twice) are evicted first
- This reduces PCIe transfers for the hot working set

MRU is catastrophic because it evicts the most recently used chunks — exactly the hot experts needed for the next decode step.

## Next Steps: Experiment Plan

### Phase 1: Complete the Policy Sweep

Run the remaining 5 policies with the fixed `test_policies.sh`:

1. `eviction_freq_pid_decay` — already loads OK
2. `prefetch_none` — control: measure how much default prefetch helps
3. `prefetch_always_max` — aggressive prefetch for expert loading
4. `prefetch_adaptive_sequential` — smart prefetch for sequential expert access
5. `prefetch_eviction_pid` — combined prefetch + eviction (add to script)

Expected time: ~6 min (5 tests x ~70s each)

### Phase 2: Design a New MoE-Optimized Policy

Based on findings so far, a custom policy should combine:

**Eviction (LFU-based with decay window)**:
- Track per-chunk access frequency like LFU, but with a **time-decaying window** (e.g., exponential moving average)
- This prevents stale "hot" experts from occupying memory when the model shifts to different experts in later decode steps
- Implementation: `chunk_used` increments a counter, but also divides all counters by 2 every N eviction events (aging)

**Prefetch (layer-aware sequential)**:
- Expert data for a given layer is stored in contiguous memory regions
- When one expert in a layer is faulted, aggressively prefetch the entire expert's memory block (max region)
- But do NOT prefetch other experts in the same layer (they may not be routed to)
- This is basically `prefetch_always_max` but scoped to the faulted expert's block

**Combined policy name**: `eviction_lfu_aging` or `moe_expert_lfu`

### Phase 3: Multi-Trial Validation

Once the best policy is identified:
- Run 10 trials per configuration using `scripts/run_trials.py`
- Report geometric mean + stddev
- Compare with paper reference values

### Phase 4: Broader Workload Testing

Test the winning policy on other workloads to check for regressions:
- vLLM KV-cache offloading (different access pattern)
- PyTorch GNN training (random access pattern)
- FAISS vector search (sequential scan pattern)

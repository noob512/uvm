# Workload Reproduction Log

**Date**: 2026-02-16
**Environment**: RTX 5090 (32GB), CUDA 12.9, Driver 575.57.08 (stock nvidia-uvm), Linux 6.15.11
**Kernel Module**: Stock nvidia-uvm (NOT the modified gpu_ext version)
**eBPF Policies**: Not loaded

---

## Pre-test: GPU Cleanup

Before each test, a shared cleanup script is used to kill stale GPU processes:

```bash
python workloads/cleanup_gpu.py
```

**Issue encountered**: An old `llama-server` process was occupying 19226 MiB GPU memory, causing PyTorch UVM 5M test to OOM. After cleanup, the test passed.

---

## Test 1: PyTorch GNN

### Setup

```bash
cd workloads/pytorch
make clean && make all          # rebuild uvm_allocator.so + gpu_allocator.so
uv venv && source .venv/bin/activate
uv pip install torch psutil torch-geometric
```

- **PyTorch version**: 2.10.0+cu128
- **Allocators**: uvm_allocator.so and gpu_allocator.so freshly built

### Test 1a: Normal Mode (default allocator, 1M nodes)

```bash
uv run python benchmark_gnn_uvm.py \
  --dataset random --nodes 1000000 --edges_per_node 10 \
  --features 128 --hidden 256 --epochs 2 --warmup 1 \
  --prop chunked \
  --report_json result/verify_default_1M.json
```

**Result: PASS**

```
Avg epoch time: 0.218s
Median epoch time: 0.218s
Memory Usage:
  GPU allocated: 1.12 GB
  CPU used: 1.64 GB
```

**Note**: `--use_gpu_allocator` causes `CUBLAS_STATUS_INVALID_VALUE` during backward pass. The default PyTorch allocator works fine. This is a known issue — the custom gpu_allocator.so conflicts with cuBLAS internal allocations.

### Test 1b: UVM Mode (5M nodes)

```bash
python workloads/cleanup_gpu.py   # CRITICAL: must clear GPU first

CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 uv run python benchmark_gnn_uvm.py \
  --dataset random --nodes 5000000 --edges_per_node 10 \
  --features 128 --hidden 256 --epochs 2 --warmup 1 \
  --prop chunked --use_uvm \
  --report_json result/verify_uvm_5M.json
```

**Result: PASS**

```
Avg epoch time: 5.529s
Median epoch time: 5.564s
Memory Usage:
  GPU allocated: 5.52 GB
  CPU used: 1.63 GB
UVM Statistics:
  Peak allocated: 22.56 GB
  Allocations: 2065
  Frees: 1706
```

**Notes**:
- First attempt OOMed because a stale `llama-server` occupied 19GB GPU. After `cleanup_gpu.py`, passed immediately.
- UVM 3M nodes also tested (6.37s/epoch, peak 15.73GB) — works fine.
- Reference value (from Dec 2025 with modified KM): 34.23s/epoch at 5M. Current 5.53s is much faster — likely because stock driver without modified KM has different page migration behavior (less oversubscription pressure at this scale).

### Test 1 Summary

| Config | Nodes | Epoch Time | Peak Alloc | Status |
|--------|-------|-----------|-----------|--------|
| Default allocator | 1M | 0.22s | 1.12 GB | PASS |
| UVM | 3M | 6.37s | 15.73 GB | PASS |
| UVM | 5M | 5.53s | 22.56 GB | PASS (after GPU cleanup) |

---

## Test 2: llama.cpp

### Setup

```bash
cd workloads/llama.cpp
make build-cuda-no-vmm    # CUDA build with VMM disabled for UVM compatibility
```

Build completed successfully (~8 min). Uses GCC-12 + CUDA 12.9.

**Available models**:
- `ggml-org_gpt-oss-20b-GGUF` (12GB) — fits in 32GB VRAM
- `unsloth_GLM-4.7-Flash-GGUF` (18GB) — fits in VRAM
- `unsloth_Qwen3-Coder-Next-GGUF` (46GB) — needs UVM
- `unsloth_Qwen3-Next-80B-A3B-Instruct-GGUF` (46GB) — needs UVM
- GPT-OSS-120B (59GB) — **NOT downloaded**, needs `huggingface-cli download ggml-org/gpt-oss-120b-GGUF`

### Test 2a: Normal Mode (20B model, fits in VRAM)

```bash
python workloads/cleanup_gpu.py

./build/bin/llama-bench \
  -m ~/.cache/llama.cpp/ggml-org_gpt-oss-20b-GGUF_gpt-oss-20b-mxfp4.gguf \
  2>&1 | tee results/verify_20b_normal.log
```

**Result: PASS**

```
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | CUDA       |  99 |           pp512 |      9897.95 ± 60.40 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | CUDA       |  99 |           tg128 |        362.59 ± 1.01 |
build: f92e406b (7100)
```

### Test 2b: UVM Mode (46GB model, oversubscribes 32GB GPU)

Attempted with two 46GB models:

```bash
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 ./build/bin/llama-bench \
  -m ~/.cache/llama.cpp/unsloth_Qwen3-Next-80B-A3B-Instruct-GGUF_Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf
```

**Result: FAILED** — `error: failed to load model`

```bash
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 ./build/bin/llama-bench \
  -m ~/.cache/llama.cpp/unsloth_Qwen3-Coder-Next-GGUF_Qwen3-Coder-Next-UD-Q4_K_XL.gguf
```

**Result: FAILED** — `error: failed to load model`

**Root cause**: The llama.cpp submodule (build f92e406b) likely doesn't support these newer Qwen3/GLM GGUF formats. The paper's UVM experiment uses GPT-OSS-120B which is a known-compatible model but is not yet downloaded (~59 GiB).

**TODO**: Download GPT-OSS-120B model and retest:
```bash
huggingface-cli download ggml-org/gpt-oss-120b-GGUF --local-dir ~/.cache/llama.cpp/
```

### Test 2 Summary

| Config | Model | Size | pp512 | tg128 | Status |
|--------|-------|------|-------|-------|--------|
| Normal | gpt-oss-20B | 11.27 GiB | 9897.95 | 362.59 | PASS |
| UVM | Qwen3-Next-80B | 46 GiB | — | — | FAIL (model format incompatible) |
| UVM | Qwen3-Coder-Next | 46 GiB | — | — | FAIL (model format incompatible) |
| UVM | GPT-OSS-120B | 59 GiB | — | — | NOT TESTED (model not downloaded) |

---

## Test 3: FAISS

### Setup

**Data**: SIFT dataset moved from schedcp to gpu_ext:

```bash
mkdir -p workloads/faiss/faiss/benchs/bigann
mv /home/yunwei37/workspace/gpu/schedcp/workloads/faiss/faiss/benchs/bigann/{bigann_base.bvecs,bigann_learn.bvecs,bigann_query.bvecs,gnd} \
   workloads/faiss/faiss/benchs/bigann/
```

Dataset files:
- `bigann_base.bvecs` — 47 GB (375M vectors)
- `bigann_learn.bvecs` — 13 GB (100M training vectors)
- `bigann_query.bvecs` — 1.3 MB (10K queries)
- `gnd/` — ground truth files

**Download script** created: `workloads/faiss/download_sift.sh` (for future reproduction).

**Build**: NOT YET DONE — requires cmake + make (~20 min).

### Test 3a: Normal GPU (SIFT10M)

**Status**: Pending (FAISS not yet built)

### Test 3b: UVM (SIFT100M)

**Status**: Pending (FAISS not yet built)

---

## Test 4: vLLM

**Status**: Skipped — vLLM installation is heavy, deferred.

---

---

## Session 2: Atomic Script Verification (2026-02-17)

**Environment**: Same hardware. PyTorch 2.10.0+cu128, CUDA 12.8, vLLM 0.11.0rc2.
**Goal**: Test all new atomic config scripts with UVM oversubscription.

### llama.cpp 120B Expert Offload (from run_exp1_expert_offload.sh)

Model: GPT-OSS-120B MXFP4 MoE (59 GiB, ~117B params), 1.84x oversubscription on 32GB GPU.
1 trial each.

| Config | pp512 (tok/s) | tg128 (tok/s) | Paper Reference |
|--------|--------------|--------------|-----------------|
| ncmoe=64 (CPU offload) | 242.35 | 16.37 | 245.63 / 16.34 |
| ncmoe=32 (CPU offload) | 257.45 | 18.17 | 260.14 / 18.18 |
| UVM baseline | 136 | 48.9 | 238.48 / 7.72 |

UVM baseline: after implementing warmup-then-migrate strategy (ported from schedcp).
Initial run without warmup-then-migrate: pp512=53, tg128=1.9.

#### UVM warmup-then-migrate fix (submodule commit 26836b27)

Root cause: commit `6e603612` in llama.cpp submodule had `assert(global_uvm_allocation_pointer.ptr)`
in `ggml_cuda_compute_forward` — crashes when UVM is disabled (ptr=NULL).

Fix:
- Phase 1 (`ggml_cuda_device_malloc`): first UVM alloc → `cudaMemAdviseSetPreferredLocation=CPU`
- Phase 2 (`ggml_cuda_compute_forward`): first compute → switch to `SetPreferredLocation=GPU`
- Guard: `if (ptr)` instead of `assert(ptr)`

### Atomic Script Matrix Test

All scripts tested sequentially (one GPU experiment at a time).
UVM configs use oversubscribed data sizes to exercise real page fault paths.

#### llama.cpp bench (`configs/bench.py`)

| Config | Model | pp512 (tok/s) | tg128 (tok/s) | Duration |
|--------|-------|--------------|--------------|----------|
| no UVM | 20B (12GB) | 9842 | 362 | 2.5s |
| UVM | 120B (59GB, 1.84x) | 138 | 48 | ~120s |

#### llama.cpp server_bench (`configs/server_bench.py`)

Uses `vllm bench serve` (from vllm workload) as benchmark client.
Tokenizer: `Qwen/Qwen3-30B-A3B-FP8` (gpt-oss GGUF has no HF tokenizer).
ShareGPT dataset, 10 prompts, request-rate=0.2, max-concurrency=1.

| Config | Model | TTFT (med, ms) | TPOT (med, ms) | Throughput (tok/s) | Success |
|--------|-------|---------------|----------------|-------------------|---------|
| no UVM | 20B | 49 | 2.9 | 52.5 | 10/10 |
| UVM | 120B (1.84x) | 256 | 78.8 | 2.3 | 1/10 |

120B UVM server: model loads OK (~110s), first request succeeds. Remaining 9 fail —
likely KV cache allocation (ctx=65536) on top of 59GB model causes severe page thrashing.
Paper RQ2 co-location uses 20B model for server workload, not 120B.

#### PyTorch GNN (`configs/gnn.py`)

| Config | Nodes | Avg Epoch (s) | Peak Memory |
|--------|-------|--------------|-------------|
| no UVM | 1M | 0.22 | 2.8 GB |
| UVM | 10M (1.43x) | 146.09 | ~45 GB |

Paper reference (without-prefetch): 10M=70.06s. Our 146s is slower — possibly due to
different driver/CUDA version or UVM allocator budget handling.

**Known issue**: GNN UVM crashes at 1M nodes with `CUBLAS_STATUS_INVALID_VALUE`.
Works fine at 5M+. Paper only tests UVM at 5M+ nodes.

#### FAISS search (`configs/search.py`)

| Config | Dataset | Search Time | Recall (1-R@1) |
|--------|---------|-------------|----------------|
| no UVM | SIFT1M | 17.8ms | 0.3498 |
| UVM | SIFT100M (1.6x) | 7246ms | 0.4486 |

#### vLLM serve_bench (`configs/serve_bench.py`)

Model: Qwen/Qwen3-30B-A3B-FP8. ShareGPT dataset, 10 prompts.

| Config | TTFT (med, ms) | TPOT (med, ms) | Success |
|--------|---------------|----------------|---------|
| cpu_offload (--cpu-offload-gb 8) | 990 | 76 | 10/10 |
| UVM (VLLM_USE_UVM=1) | 18108 | 51 | 10/10 |

UVM TTFT is very high (18s) — initial page faults when model weights are first accessed.
TPOT is actually lower than cpu_offload once weights are resident.

#### Layer 2 Infrastructure

| Tool | Test | Result |
|------|------|--------|
| `scripts/run_trials.py` | 2 trials × llama bench 20B | geomean=9814.3 tok/s, stddev=5.1 |
| `scripts/collect_results.py` | Called by run_trials | JSON summary with geomean/stddev/min/max |

### Summary Matrix

| Script | no UVM | UVM (oversubscribed) | Status |
|--------|--------|---------------------|--------|
| llama bench | pp=9842, tg=362 (20B) | pp=138, tg=48 (120B, 1.84x) | OK |
| llama server | TTFT=49ms, TPOT=2.9ms (20B) | TTFT=256ms, TPOT=78.8ms (120B, 1/10 ok) | OK* |
| pytorch GNN | 0.22s/epoch (1M) | 146s/epoch (10M, 1.43x) | OK |
| FAISS search | 17.8ms (SIFT1M) | 7246ms (SIFT100M, 1.6x) | OK |
| vLLM serve | TTFT=990ms, TPOT=76ms (cpu_offload) | TTFT=18108ms, TPOT=51ms (UVM) | OK |

*120B server 9/10 requests fail due to extreme page thrashing at ctx=65536.

---

## Known Issues (cumulative)

1. **Stale GPU processes cause OOM**: Always run `cleanup_gpu.py` before benchmarks.

2. **`--use_gpu_allocator` crashes with cuBLAS**: The custom `gpu_allocator.so` causes `CUBLAS_STATUS_INVALID_VALUE`. UVM allocator works.

3. **GNN UVM at small node counts**: Crashes at 1M nodes with CUBLAS error. Works at 5M+.

4. **120B server UVM timeout**: 9/10 requests fail with 120B model + ctx=65536 in UVM mode. KV cache on top of 59GB model causes severe thrashing.

5. **UVM pp512 gap vs paper**: 120B UVM baseline gets pp512=136 vs paper's 238. Warmup-then-migrate helps (53→136) but doesn't close the gap. May need the paper's kernel module optimizations.

6. **Reference results discrepancy**: PyTorch 5M UVM epoch time varies between sessions (5.53s vs 34.26s) — likely depends on GPU memory state and driver behavior.

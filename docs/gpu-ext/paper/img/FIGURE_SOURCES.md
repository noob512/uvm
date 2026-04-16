# Figure Sources Documentation

This document tracks the source locations and generation scripts for all figures used in the paper.

## Background/Motivation Figures

| Figure | File | Source | Script |
|--------|------|--------|--------|
| Page fault patterns | `pattern/combined_patterns_1x5.pdf` | `img/pattern/` subdirs (faiss-build, faiss-query, llama.cpp-decode, llama.cpp-prefill, pytorch-dnn) | `img/pattern/combine_patterns.py` |
| Thread scheduling | `pattern/vector_add/thread_scheduling_motivation.pdf` | `img/pattern/vector_add/` | `plot_thread_scheduling.py` |
| Motivation silos | `motivation_silos.pdf` | `img/` | - |
| Architecture | `gpu-ebpf-arch.png` | `img/` | - |

**Thread scheduling figure data source**: `test_gpu_thread_exec.md` in the same directory. Raw data from `threadscheduling` eBPF tool output.

## RQ1: Single-Tenant Memory/Scheduling

### llama.cpp (GPT-OSS-120B Expert Offloading)

| Figure | File | Source | Script |
|--------|------|--------|--------|
| Expert offload throughput | `results-raw/llama.cpp/llama_uvm_combined_color.pdf` | `/home/yunwei37/workspace/gpu/schedcp/workloads/llama.cpp/uvm/` | `visbasic.py` |

**Data source**: `test-record-single.md` in the same directory

### vLLM (Qwen-30B KV-cache Offloading)

| Figure | File | Source | Script |
|--------|------|--------|--------|
| TTFT/TPOT comparison | `results-raw/vllm/ttft_tpot_combined.pdf` | `/home/yunwei37/workspace/gpu/schedcp/workloads/vllm/uvm/first-iter/` | `generate_figures.py` |

**Data source**: JSON files and `test_uvm_base.md` in the same directory

### PyTorch GNN Training

| Figure | File | Source | Script |
|--------|------|--------|--------|
| UVM benchmark comparison | `results-raw/pytorch/uvm_benchmark_comparison.pdf` | `/home/yunwei37/workspace/gpu/schedcp/workloads/pytorch/` | `visualize_all.py` |

**Data source**: `benchmark_gnn_uvm.py` output

### Faiss Index Building

| Figure | File | Source | Script |
|--------|------|--------|--------|
| Faiss benchmark results | `results-raw/faiss/faiss_benchmark_results.pdf` | `/home/yunwei37/workspace/gpu/schedcp/workloads/faiss/results/` | `plot_results.py` |

**Data source**: `SIFT*_IVF4096_Flat_*.json` files and `experiment_analysis.md`

## RQ2: Multi-Tenant Memory, Bandwidth, and Scheduling

### Multi-Tenant Scheduler (LC + BE)

| Figure | File | Source | Script |
|--------|------|--------|--------|
| Scheduler latency/throughput | `results-raw/multi-tenant/scheduler_latency_throughput.pdf` | `/home/yunwei37/workspace/gpu/co-processor-demo/gpu_ext_policy/docs/eval/multi-tenant-scheduler/` | `plot_figures.py` |

**Data source**: `simple_test_results/` directory and `README.md`

### Multi-Tenant Memory Priority

| Figure | File | Source | Script |
|--------|------|--------|--------|
| All kernels stacked | `results-raw/multi-tenant/all_kernels_stacked.pdf` | `/home/yunwei37/workspace/gpu/co-processor-demo/gpu_ext_policy/docs/eval/multi-tenant-memory/` | `plot_all_kernels_stacked.py` |

**Data source**: `results_hotspot/`, `results_gemm/`, `results_kmeans/` directories

### Two-Tenant Co-location (llama.cpp + GNN)

| Figure | File | Source | Script |
|--------|------|--------|--------|
| Co-located results | `results-raw/multi-tenant/fig_colocated_results.pdf` | `/home/yunwei37/workspace/gpu/schedcp/workloads/llama.cpp/uvm/` | `plot_colocated_results.py` |

**Data source**: `test-record-co-located.md`

## RQ3: Programmability and Mechanism Overhead

### Runtime Overhead

| Figure | File | Source | Script |
|--------|------|--------|--------|
| Microbench comparison | `results-raw/runtime/microbench_comparison.pdf` | `img/results-raw/runtime/` | `plot_microbench.py` |

**Data source**: `micro_vec_add_result.md`

## How to Regenerate Figures

1. **llama.cpp figures**:
   ```bash
   cd /home/yunwei37/workspace/gpu/schedcp/workloads/llama.cpp/uvm
   python visbasic.py  # or plot_colocated_results.py
   ```

2. **vLLM figures**:
   ```bash
   cd /home/yunwei37/workspace/gpu/schedcp/workloads/vllm/uvm/first-iter
   python generate_figures.py
   ```

3. **PyTorch figures**:
   ```bash
   cd /home/yunwei37/workspace/gpu/schedcp/workloads/pytorch
   python visualize_all.py
   ```

4. **Faiss figures**:
   ```bash
   cd /home/yunwei37/workspace/gpu/schedcp/workloads/faiss/results
   python plot_results.py
   ```

5. **Multi-tenant scheduler figures**:
   ```bash
   cd /home/yunwei37/workspace/gpu/co-processor-demo/gpu_ext_policy/docs/eval/multi-tenant-scheduler
   python plot_figures.py
   ```

6. **Multi-tenant memory figures**:
   ```bash
   cd /home/yunwei37/workspace/gpu/co-processor-demo/gpu_ext_policy/docs/eval/multi-tenant-memory
   python plot_all_kernels_stacked.py
   ```

7. **Pattern figures**:
   ```bash
   cd /home/yunwei37/workspace/gpu/co-processor-demo/gbpf-paper/img/pattern
   python combine_patterns.py
   ```

## Copy Commands

After regenerating, copy figures to paper directory:

```bash
# llama.cpp
cp /home/yunwei37/workspace/gpu/schedcp/workloads/llama.cpp/uvm/llama_uvm_combined_color.pdf img/results-raw/llama.cpp/
cp /home/yunwei37/workspace/gpu/schedcp/workloads/llama.cpp/uvm/fig_colocated_results.pdf img/results-raw/multi-tenant/

# vLLM
cp /home/yunwei37/workspace/gpu/schedcp/workloads/vllm/uvm/first-iter/ttft_tpot_combined.pdf img/results-raw/vllm/

# PyTorch
cp /home/yunwei37/workspace/gpu/schedcp/workloads/pytorch/uvm_benchmark_comparison.pdf img/results-raw/pytorch/

# Faiss
cp /home/yunwei37/workspace/gpu/schedcp/workloads/faiss/results/faiss_benchmark_results.pdf img/results-raw/faiss/

# Multi-tenant scheduler
cp /home/yunwei37/workspace/gpu/co-processor-demo/gpu_ext_policy/docs/eval/multi-tenant-scheduler/fig_main_result.pdf img/results-raw/multi-tenant/scheduler_latency_throughput.pdf

# Multi-tenant memory
cp /home/yunwei37/workspace/gpu/co-processor-demo/gpu_ext_policy/docs/eval/multi-tenant-memory/all_kernels_stacked.pdf img/results-raw/multi-tenant/
```

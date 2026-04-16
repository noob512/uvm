# Attention-Aware Memory Subsystem Experiments

This directory contains experiment scripts for evaluating the attention-aware memory subsystem against baseline UVM eviction policies.

## Overview

The attention-aware memory subsystem uses attention scores from the vLLM inference engine to guide GPU memory eviction decisions via eBPF policies. This should improve performance by:

- Keeping high-attention KV cache blocks in GPU memory
- Evicting low-attention blocks to CPU memory
- Reducing page faults during decode phase
- Improving tail latency (P99) for long-context workloads

## Files

### Main Experiment Script
- **`run_exp_attention_aware.sh`** - One-click experiment runner
  - Compares baseline UVM (LRU) vs attention-aware eviction
  - Runs vLLM serve + benchmark for each config
  - Generates comparison plots
  - Saves full logs with timestamps

### Standalone Scripts
- **`benchmark_attention_aware.py`** - Standalone benchmark script
  - Can run individual configs independently
  - Supports baseline, attention_aware, and cpu_offload modes
  - Outputs structured JSON results

- **`plot_attention_results.py`** - Plot generation script
  - Creates latency comparison plots (TTFT, TPOT)
  - Generates throughput comparison
  - Produces summary tables
  - Shows improvement percentages vs baseline

## Usage

### Quick Start (Recommended)

Run the full experiment suite:

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./run_exp_attention_aware.sh
```

Results will be saved to `results/exp_attention_aware/<timestamp>/`

### Custom Configuration

Set environment variables to customize:

```bash
# Use different model
export MODEL="meta-llama/Llama-2-70b-hf"

# Adjust workload parameters
export NUM_PROMPTS=100
export REQUEST_RATE=5.0
export MAX_CONCURRENCY=16

./run_exp_attention_aware.sh
```

### Standalone Benchmark

Run individual configs:

```bash
# Baseline UVM (LRU)
uv run python benchmark_attention_aware.py \
    --config baseline \
    --prompts 50 \
    --output results/baseline.json

# Attention-aware eviction
uv run python benchmark_attention_aware.py \
    --config attention_aware \
    --prompts 50 \
    --output results/attention.json

# CPU offload (reference)
uv run python benchmark_attention_aware.py \
    --config cpu_offload \
    --prompts 50 \
    --output results/cpu_offload.json
```

### Generate Plots

From a results directory:

```bash
uv run python plot_attention_results.py \
    --results-dir results/exp_attention_aware/20260409_123456
```

From individual JSON files:

```bash
uv run python plot_attention_results.py \
    --baseline results/baseline.json \
    --attention results/attention.json \
    --output results/comparison
```

## Prerequisites

1. **vLLM with UVM support** - Must be installed in the workload venv:
   ```bash
   cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
   uv sync
   uv pip install -e vllm/
   ```

2. **ShareGPT dataset** - Auto-downloaded on first run, or manually:
   ```bash
   python download_sharegpt.py --dataset vicuna
   ```

3. **eBPF program** (for attention-aware config):
   ```bash
   # Check if compiled
   ls -lh ../../extension/attention_aware_eviction
   
   # If not, compile it
   cd ../../extension
   make attention_aware_eviction
   ```

4. **Model access** - Default model is `Qwen/Qwen3-30B-A3B-FP8`
   - Auto-downloaded by vLLM on first run
   - Requires ~30GB disk space
   - Requires HuggingFace token for gated models

## Expected Results

With attention-aware eviction, you should see:

- **Lower TTFT P99**: Reduced cold-start latency (fewer initial page faults)
- **Lower TPOT P99**: More stable decode performance (fewer mid-decode faults)
- **Higher throughput**: Better memory utilization (keep hot pages resident)
- **Lower swap rate**: Fewer unnecessary evictions

Typical improvements (from paper):
- TTFT P99: 20-40% reduction
- TPOT P99: 15-30% reduction
- Throughput: 10-25% increase

## Output Structure

```
results/exp_attention_aware/<timestamp>/
├── full.log                          # Complete experiment log
├── baseline_uvm_lru.json             # Baseline results
├── baseline_uvm_lru.log              # Baseline benchmark log
├── attention_aware_ebpf.json         # Attention-aware results
├── attention_aware_ebpf.log          # Attention-aware benchmark log
├── bpf_attention_aware_eviction.log  # eBPF program log
├── vllm_server_baseline.log          # vLLM server logs
├── vllm_server_attention_aware.log
├── comparison_latency.png            # Latency comparison plot
├── comparison_throughput.png         # Throughput comparison plot
├── comparison_summary.png            # Summary table
└── comparison_improvement.png        # Improvement percentages
```

## Troubleshooting

### eBPF program not found
```
WARNING: attention_aware_eviction not found at ../../extension/attention_aware_eviction
```
**Solution**: Compile the eBPF program:
```bash
cd ../../extension
make attention_aware_eviction
```

### vLLM not installed
```
ERROR: vLLM not installed in workload venv
```
**Solution**: Install vLLM from local source:
```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
uv sync
uv pip install -e vllm/
```

### Dataset not found
```
ERROR: ShareGPT dataset not found
```
**Solution**: Download the dataset:
```bash
python download_sharegpt.py --dataset vicuna
```

### Server fails to start
Check the server log for details:
```bash
cat results/exp_attention_aware/<timestamp>/vllm_server_*.log
```

Common issues:
- Out of memory: Reduce `--max-num-seqs` or use smaller model
- Model not found: Check HuggingFace token and model name
- Port in use: Change `PORT` variable in script

## Integration with Score Bridge

The attention-aware eviction policy requires the score bridge to be running. The score bridge:

1. Collects attention scores from vLLM's PagedAttention kernels
2. Aggregates scores per KV cache block
3. Exposes scores to eBPF via BPF maps
4. Updates scores periodically during inference

See `README_SCORE_BRIDGE.md` for score bridge implementation details.

## Related Files

- `score_bridge_vllm.py` - Score bridge daemon
- `example_score_bridge_usage.py` - Score bridge usage examples
- `run_exp_attention_eviction.sh` - Original attention eviction experiment
- `../../docs/attention_aware_memory_subsystem_feasibility.md` - Design doc

## Notes

- Experiments run serially (one at a time) due to BPF struct_ops singleton constraint
- GPU cleanup runs before each config to ensure clean state
- Results are committed to git (small JSON files ~4KB each)
- Logs are gitignored (can be large)
- All scripts use `uv run` to ensure correct venv activation

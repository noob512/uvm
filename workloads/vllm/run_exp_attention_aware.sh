#!/usr/bin/env bash
# =============================================================================
# Experiment: Attention-Aware Memory Subsystem
#
# Compares attention-aware eviction policies against baseline LRU.
# Tests scenarios: long context (32K), multi-request concurrency, memory oversubscription
#
# Configs:
#   1. Baseline (UVM default LRU)
#   2. Attention-Aware (eBPF with attention score bridge)
#
# Reference: docs/attention_aware_memory_subsystem_feasibility.md
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKLOADS_DIR="$(dirname "$SCRIPT_DIR")"
EXT_DIR="$WORKLOADS_DIR/../extension"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$SCRIPT_DIR/results/exp_attention_aware/${TIMESTAMP}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
DATASET="$SCRIPT_DIR/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"

# Experiment parameters
NUM_PROMPTS="${NUM_PROMPTS:-50}"
REQUEST_RATE="${REQUEST_RATE:-2.0}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
PORT=8000

mkdir -p "$RESULTS_DIR"

# Log everything
exec > >(tee -a "$RESULTS_DIR/full.log") 2>&1

echo "============================================================"
echo "Experiment: Attention-Aware Memory Subsystem"
echo "============================================================"
echo "Timestamp: $TIMESTAMP"
echo "Model: $MODEL"
echo "Prompts: $NUM_PROMPTS @ ${REQUEST_RATE} RPS"
echo "Max concurrency: $MAX_CONCURRENCY"
echo "Results dir: $RESULTS_DIR"
echo ""

# --- Prerequisites ---
if [ ! -f "$DATASET" ]; then
    echo "ERROR: ShareGPT dataset not found at $DATASET"
    echo "Run: cd $SCRIPT_DIR && python download_sharegpt.py --dataset vicuna"
    exit 1
fi

# Check if vLLM is installed
if ! uv run --directory "$SCRIPT_DIR" python -c "import vllm" 2>/dev/null; then
    echo "ERROR: vLLM not installed in workload venv"
    echo "Run: cd $SCRIPT_DIR && uv sync && uv pip install -e vllm/"
    exit 1
fi

# ==========================================================
# Helper Functions
# ==========================================================

cleanup_gpu() {
    echo "Cleaning GPU processes..."
    python3 "$WORKLOADS_DIR/cleanup_gpu.py" 2>/dev/null || true
    sleep 2
}

start_bpf() {
    local binary="$1"
    shift
    local args=("$@")

    if [ ! -f "$EXT_DIR/$binary" ]; then
        echo "WARNING: $binary not found in $EXT_DIR, skipping..."
        return 1
    fi

    echo "  Loading BPF: $binary ${args[*]:-}"
    sudo "$EXT_DIR/$binary" "${args[@]}" > "$RESULTS_DIR/bpf_${binary}.log" 2>&1 &
    BPF_PID=$!
    sleep 3

    # Verify it loaded
    if ! kill -0 "$BPF_PID" 2>/dev/null; then
        echo "  ERROR: BPF program failed to load"
        cat "$RESULTS_DIR/bpf_${binary}.log"
        return 1
    fi
    echo "  BPF loaded (PID=$BPF_PID)"
    return 0
}

stop_bpf() {
    if [ -n "${BPF_PID:-}" ] && kill -0 "$BPF_PID" 2>/dev/null; then
        echo "  Stopping BPF program (PID $BPF_PID)..."
        sudo kill "$BPF_PID" 2>/dev/null || true
        wait "$BPF_PID" 2>/dev/null || true
    fi
    BPF_PID=""
    sleep 2
}

run_benchmark() {
    local config_name="$1"
    local output_json="$RESULTS_DIR/${config_name}.json"

    echo ""
    echo "=========================================="
    echo "Running: $config_name"
    echo "=========================================="

    cleanup_gpu

    # Run the benchmark using the atomic serve_bench.py script
    uv run --directory "$SCRIPT_DIR" python configs/serve_bench.py \
        --mode uvm \
        --prompts "$NUM_PROMPTS" \
        --port "$PORT" \
        --output "$output_json" \
        2>&1 | tee "$RESULTS_DIR/${config_name}.log"

    if [ -f "$output_json" ]; then
        echo "  ✓ Results saved to $output_json"
        # Extract key metrics
        python3 -c "
import json
with open('$output_json') as f:
    d = json.load(f)
    m = d.get('metrics', {})
    print(f\"  TTFT P50: {m.get('median_ttft_ms', 0):.2f}ms\")
    print(f\"  TTFT P99: {m.get('p99_ttft_ms', 0):.2f}ms\")
    print(f\"  TPOT P50: {m.get('median_tpot_ms', 0):.2f}ms\")
    print(f\"  TPOT P99: {m.get('p99_tpot_ms', 0):.2f}ms\")
    print(f\"  Throughput: {m.get('output_throughput_tok_s', 0):.2f} tok/s\")
" 2>/dev/null || echo "  (check JSON manually)"
    else
        echo "  ✗ Benchmark failed, no output JSON"
    fi
}

# ==========================================================
# Phase 1: Baseline (UVM default LRU)
# ==========================================================
echo ""
echo "============================================================"
echo "Phase 1: Baseline (UVM default LRU)"
echo "============================================================"

run_benchmark "baseline_uvm_lru"

# ==========================================================
# Phase 2: Attention-Aware (with eBPF)
# ==========================================================
echo ""
echo "============================================================"
echo "Phase 2: Attention-Aware (eBPF + Score Bridge)"
echo "============================================================"

# Check if attention-aware eviction binary exists
if [ -f "$EXT_DIR/attention_aware_eviction" ]; then
    if start_bpf "attention_aware_eviction"; then
        run_benchmark "attention_aware_ebpf"
        stop_bpf
    else
        echo "WARNING: Failed to load attention_aware_eviction, skipping..."
    fi
else
    echo "WARNING: attention_aware_eviction not found at $EXT_DIR/attention_aware_eviction"
    echo "This experiment requires the attention-aware eBPF program to be compiled."
    echo "Skipping attention-aware config..."
fi

# ==========================================================
# Phase 3: Generate Comparison Plots
# ==========================================================
echo ""
echo "============================================================"
echo "Phase 3: Generate Comparison Plots"
echo "============================================================"

if [ -f "$SCRIPT_DIR/plot_attention_results.py" ]; then
    echo "Generating plots..."
    uv run --directory "$SCRIPT_DIR" python plot_attention_results.py \
        --results-dir "$RESULTS_DIR" \
        --output "$RESULTS_DIR/comparison_plots.png" \
        2>&1 | tee "$RESULTS_DIR/plot.log" || {
        echo "WARNING: Plot generation failed"
    }
else
    echo "WARNING: plot_attention_results.py not found, skipping plots"
fi

# ==========================================================
# Summary
# ==========================================================
echo ""
echo "============================================================"
echo "RESULTS SUMMARY"
echo "============================================================"
echo ""

echo "Configuration Comparison:"
echo "-------------------------"

for config in baseline_uvm_lru attention_aware_ebpf; do
    json_file="$RESULTS_DIR/${config}.json"
    if [ -f "$json_file" ]; then
        echo ""
        echo "$config:"
        python3 -c "
import json
with open('$json_file') as f:
    d = json.load(f)
    m = d.get('metrics', {})
    print(f\"  TTFT (median): {m.get('median_ttft_ms', 0):.2f}ms\")
    print(f\"  TTFT (P99):    {m.get('p99_ttft_ms', 0):.2f}ms\")
    print(f\"  TPOT (median): {m.get('median_tpot_ms', 0):.2f}ms\")
    print(f\"  TPOT (P99):    {m.get('p99_tpot_ms', 0):.2f}ms\")
    print(f\"  Throughput:    {m.get('output_throughput_tok_s', 0):.2f} tok/s\")
    print(f\"  Duration:      {m.get('benchmark_duration_s', 0):.2f}s\")
" 2>/dev/null || echo "  (parse error)"
    fi
done

echo ""
echo "Expected improvements with attention-aware eviction:"
echo "  - Lower TTFT P99 (reduced cold-start latency)"
echo "  - Lower TPOT P99 (fewer page faults during decode)"
echo "  - Higher throughput (better memory utilization)"
echo ""
echo "Full logs and results saved to: $RESULTS_DIR/"
echo "Done."

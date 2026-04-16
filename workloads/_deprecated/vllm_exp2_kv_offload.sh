#!/usr/bin/env bash
# =============================================================================
# Experiment 2: vLLM KV-cache Offloading (Paper RQ1, Figure 7)
#
# Model: Qwen/Qwen3-30B-A3B-FP8 (~30GB)
# GPU: RTX 5090 (32GB), KV-cache at ~60K tokens → 36-40GB total
# Workload: 100 concurrent requests from ShareGPT (single-round, no prefix caching)
# Metrics: mean/p99 TTFT (ms), mean/p99 TPOT (ms), throughput (tok/s)
#
# Configs without modified KM: 1 (cpu_offload), 2 (uvm_baseline), 4 (lmcache)
# Configs needing modified KM: 3 (UVM + gpu_ext sequential prefetch)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKLOADS_DIR="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$SCRIPT_DIR/results/exp2_kv_offload/${TIMESTAMP}"
BASELINE_SCRIPT="$SCRIPT_DIR/uvm/test_uvm_baselines.py"
DATASET="$SCRIPT_DIR/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"

mkdir -p "$RESULTS_DIR"

# Log everything
exec > >(tee -a "$RESULTS_DIR/full.log") 2>&1

echo "============================================================"
echo "Experiment 2: vLLM KV-cache Offloading (Figure 7)"
echo "============================================================"
echo "Timestamp: $TIMESTAMP"
echo "Results dir: $RESULTS_DIR"
echo ""

# --- Prerequisites ---
uv run --directory "$SCRIPT_DIR" python -c "import vllm; print('vLLM version:', vllm.__version__)" 2>/dev/null || {
    echo "ERROR: vLLM not installed."
    echo "Run: cd $SCRIPT_DIR && uv venv && uv pip install vllm"
    exit 1
}

if [ ! -f "$DATASET" ]; then
    echo "Downloading ShareGPT dataset..."
    make -C "$SCRIPT_DIR" download-datasets || {
        echo "ERROR: Failed to download ShareGPT dataset"
        exit 1
    }
fi

# Paper benchmark args
BENCH_ARGS="--model Qwen/Qwen3-30B-A3B-FP8 --dataset-name sharegpt --num-prompts 100 --dataset-path $DATASET --sharegpt-output-len 512 --seed 42 --request-rate 5"

# --- GPU Cleanup ---
echo "Cleaning GPU processes..."
python3 "$WORKLOADS_DIR/cleanup_gpu.py" 2>/dev/null || true
sleep 2

# ==========================================================
# Config 1: CPU Offload (--cpu-offload-gb 8)
# ==========================================================
echo ""
echo "============================================================"
echo "Config 1/3: CPU Offload (8GB)"
echo "============================================================"
python3 "$WORKLOADS_DIR/cleanup_gpu.py" 2>/dev/null || true
sleep 2

uv run --directory "$SCRIPT_DIR" python "$BASELINE_SCRIPT" \
    --bench-args "$BENCH_ARGS" \
    --baselines cpu_offload \
    --output-dir "$RESULTS_DIR" \
    2>&1 | tee "$RESULTS_DIR/config1_cpu_offload.log" || {
    echo "Config 1 FAILED"
}

# ==========================================================
# Config 2: UVM Baseline
# ==========================================================
echo ""
echo "============================================================"
echo "Config 2/3: UVM Baseline"
echo "============================================================"
python3 "$WORKLOADS_DIR/cleanup_gpu.py" 2>/dev/null || true
sleep 5

uv run --directory "$SCRIPT_DIR" python "$BASELINE_SCRIPT" \
    --bench-args "$BENCH_ARGS" \
    --baselines uvm_baseline \
    --output-dir "$RESULTS_DIR" \
    2>&1 | tee "$RESULTS_DIR/config2_uvm_baseline.log" || {
    echo "Config 2 FAILED"
}

# ==========================================================
# Config 4: LMCache (if available)
# ==========================================================
echo ""
echo "============================================================"
echo "Config 3/3: LMCache (if available)"
echo "============================================================"
python3 "$WORKLOADS_DIR/cleanup_gpu.py" 2>/dev/null || true
sleep 5

LMCACHE_DIR="${LMCACHE_SERVER_DIR:-$HOME/workspace/gpu/LMCache}"
if [ -d "$LMCACHE_DIR" ]; then
    uv run --directory "$SCRIPT_DIR" python "$BASELINE_SCRIPT" \
        --bench-args "$BENCH_ARGS" \
        --baselines lmcache \
        --output-dir "$RESULTS_DIR" \
        2>&1 | tee "$RESULTS_DIR/config4_lmcache.log" || {
        echo "Config 4 (LMCache) FAILED"
    }
else
    echo "SKIPPED: LMCache not found at $LMCACHE_DIR"
    echo "Install LMCache to run this config."
fi

# --- Summary ---
echo ""
echo "============================================================"
echo "RESULTS SUMMARY (Experiment 2: KV-cache Offloading)"
echo "============================================================"
echo ""

# Parse JSON results
for f in "$RESULTS_DIR"/*.json; do
    [ -f "$f" ] || continue
    echo "$(basename "$f"):"
    python3 -c "
import json, sys
d = json.load(open('$f'))
print(f\"  Mean TTFT: {d.get('mean_ttft_ms', 'N/A'):.2f} ms\")
print(f\"  P99 TTFT:  {d.get('p99_ttft_ms', 'N/A'):.2f} ms\")
print(f\"  Mean TPOT: {d.get('mean_tpot_ms', 'N/A'):.2f} ms\")
print(f\"  Throughput: {d.get('output_token_throughput', 'N/A'):.2f} tok/s\")
" 2>/dev/null || echo "  (could not parse results)"
    echo ""
done

echo "Paper reference (100 concurrent requests):"
echo "  CPU offload: TTFT=8387.80ms  TPOT=324.13ms  throughput=391.14 tok/s"
echo "  UVM baseline: TTFT=9642.27ms  TPOT=374.23ms  throughput=307.26 tok/s"
echo "  LMCache:     TTFT=5401.71ms  TPOT=222.24ms  throughput=571.54 tok/s"
echo ""
echo "NOTE: Config 3 (UVM+gpu_ext) requires modified kernel module."
echo "Full logs saved to: $RESULTS_DIR/"

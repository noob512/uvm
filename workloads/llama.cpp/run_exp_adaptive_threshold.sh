#!/bin/bash
# Experiment: Adaptive Threshold BPF Policies
#
# Benchmarks existing adaptive BPF programs with various configurations.
# Compares with always_max baseline to quantify BPF path overhead and
# validate runtime-adjustable threshold functionality.
#
# Reference: NVIDIA Bug 1778037 (adaptive threshold, never implemented)
# Reference: MSched reproduction plan P1 (adaptive threshold)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXT_DIR="$SCRIPT_DIR/../../extension"
RESULTS_DIR="$SCRIPT_DIR/results/exp_adaptive_threshold/$(date +%Y%m%d_%H%M%S)"
MODEL="$HOME/.cache/llama.cpp/ggml-org_gpt-oss-120b-GGUF_gpt-oss-120b-mxfp4-00001-of-00003.gguf"
LLAMA_BENCH="$SCRIPT_DIR/build/bin/llama-bench"

# Benchmark parameters
PP=512
TG=128
REPS=5

mkdir -p "$RESULTS_DIR"

echo "============================================"
echo "Experiment: Adaptive Threshold BPF Policies"
echo "============================================"
echo "Model: gpt-oss-120b-mxfp4"
echo "Config: pp=$PP, tg=$TG, r=$REPS"
echo "Results: $RESULTS_DIR"
echo ""

# Check prerequisites
if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found at $MODEL"
    exit 1
fi
if [ ! -f "$LLAMA_BENCH" ]; then
    echo "ERROR: llama-bench not found at $LLAMA_BENCH"
    exit 1
fi

run_bench() {
    local name="$1"
    local logfile="$RESULTS_DIR/${name}.log"

    echo "--- Running: $name ---"
    python3 "$SCRIPT_DIR/../cleanup_gpu.py" 2>/dev/null || true
    sleep 2

    GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
    "$LLAMA_BENCH" \
        -m "$MODEL" \
        -p "$PP" -n "$TG" -r "$REPS" \
        -o json 2>&1 | tee "$logfile"

    echo ""
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
    sudo "$EXT_DIR/$binary" "${args[@]}" &
    BPF_PID=$!
    sleep 2

    # Verify it loaded
    if ! kill -0 "$BPF_PID" 2>/dev/null; then
        echo "  ERROR: BPF program failed to load"
        return 1
    fi
    return 0
}

stop_bpf() {
    if [ -n "${BPF_PID:-}" ] && kill -0 "$BPF_PID" 2>/dev/null; then
        echo "  Stopping BPF program (PID $BPF_PID)..."
        sudo kill "$BPF_PID" 2>/dev/null || true
        wait "$BPF_PID" 2>/dev/null || true
    fi
    BPF_PID=""
    sleep 1
}

# ===== Baseline: No BPF =====
echo "========== Baseline (no BPF) =========="
run_bench "baseline_no_bpf"

# ===== Baseline: always_max =====
echo "========== always_max (control) =========="
if start_bpf "prefetch_always_max"; then
    run_bench "always_max"
    stop_bpf
fi

# ===== Baseline: passive MRU (current best) =====
echo "========== passive MRU (current best) =========="
if start_bpf "prefetch_max_passive_mru"; then
    run_bench "passive_mru"
    stop_bpf
fi

# ===== adaptive_tree_iter with various thresholds =====
echo ""
echo "=========================================="
echo "  adaptive_tree_iter (ENTER_LOOP mode)"
echo "=========================================="

for THRESH in 1 10 25 51; do
    echo "========== tree_iter threshold=$THRESH =========="
    if start_bpf "prefetch_adaptive_tree_iter" -t "$THRESH"; then
        run_bench "tree_iter_t${THRESH}"
        stop_bpf
    fi
done

# ===== adaptive_sequential with various percentages =====
echo ""
echo "=========================================="
echo "  adaptive_sequential (BYPASS mode)"
echo "=========================================="

for PCT in 100 50 25; do
    echo "========== sequential pct=$PCT =========="
    if start_bpf "prefetch_adaptive_sequential" -p "$PCT"; then
        run_bench "sequential_p${PCT}"
        stop_bpf
    fi
done

# ===== template_belady (if available) =====
echo ""
echo "=========================================="
echo "  template_belady (Belady eviction)"
echo "=========================================="

echo "========== template_belady (no profile) =========="
if start_bpf "prefetch_template_belady" --layers 36 --protect 3; then
    run_bench "template_belady_default"
    stop_bpf
fi

# With profile if available
PROFILE_PATH="/home/yunwei37/workspace/gpu/NVBit/scripts/profiling_data/layer_va_ranges.json"
if [ -f "$PROFILE_PATH" ]; then
    echo "========== template_belady (with NVBit profile) =========="
    if start_bpf "prefetch_template_belady" --profile "$PROFILE_PATH" --layers 36 --protect 3; then
        run_bench "template_belady_profiled"
        stop_bpf
    fi
fi

# ===== Summary =====
echo ""
echo "============================================"
echo "                  SUMMARY"
echo "============================================"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Log files:"
for f in "$RESULTS_DIR"/*.log; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .log)
    # Extract pp and tg from JSON output
    pp_val=$(grep -o '"avg_ts":[0-9.]*' "$f" | head -1 | cut -d: -f2 || echo "N/A")
    tg_val=$(grep -o '"avg_ts":[0-9.]*' "$f" | tail -1 | cut -d: -f2 || echo "N/A")
    printf "  %-30s pp=%-10s tg=%-10s\n" "$name" "$pp_val" "$tg_val"
done

echo ""
echo "Reference values:"
echo "  Baseline (no BPF):    pp=139.5, tg=45.3"
echo "  always_max:           pp=219.1, tg=76.9"
echo "  passive MRU:          pp=228.0, tg=78.7"
echo "  threshold=1 (modprobe): pp=217.1, tg=76.0"
echo ""
echo "Done."

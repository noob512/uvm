#!/bin/bash
# =============================================================================
# Phase 1: Score-Aware Eviction Experiment
#
# Compares attention-aware eviction against baselines on vLLM serving.
#
# Configs:
#   1. cpu_offload       — vLLM --cpu-offload-gb 8, no UVM
#   2. uvm_baseline      — UVM, no BPF policy
#   3. uvm_cycle_moe     — UVM + cycle_moe BPF (existing best)
#   4. uvm_attention_no_score — UVM + attention_aware BPF (score_map empty)
#   5. uvm_attention_scored   — UVM + attention_aware BPF + score_bridge
#
# Usage:
#   cd workloads/vllm
#   sudo bash run_exp_attention_eviction.sh [--prompts N] [--trials N] [--skip-build]
#
# Prerequisites:
#   - RTX 5090 with patched nvidia-uvm.ko loaded
#   - extension/ built (or pass --skip-build to build here)
#   - vLLM installed: uv pip install -e vllm/
#   - ShareGPT dataset: make download-datasets
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKLOAD_DIR="$SCRIPT_DIR"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXTENSION_DIR="$ROOT_DIR/extension"

MODEL="Qwen/Qwen3-30B-A3B-FP8"
DATASET_PATH="$WORKLOAD_DIR/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"

NUM_PROMPTS=100
NUM_TRIALS=3
SKIP_BUILD=0
PORT=8000
REQUEST_RATE=5
OUTPUT_LEN=512
MAX_NUM_SEQS=16
SERVER_TIMEOUT=600

# Score bridge parameters (Phase 1: estimated from KV cache layout)
# These will be auto-detected at runtime when possible
SCORE_BRIDGE_BLOCK_SIZE_KB=256
SCORE_BRIDGE_TOKENS_PER_BLOCK=16
SCORE_BRIDGE_SINK_TOKENS=4
SCORE_BRIDGE_RECENT_WINDOW=256

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="$WORKLOAD_DIR/results/exp_attention_eviction/$TIMESTAMP"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompts)    NUM_PROMPTS="$2"; shift 2 ;;
        --trials)     NUM_TRIALS="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD=1; shift ;;
        --port)       PORT="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: sudo bash $0 [--prompts N] [--trials N] [--skip-build]"
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

cleanup_gpu() {
    log "Cleaning up GPU processes..."
    python3 "$ROOT_DIR/workloads/cleanup_gpu.py" --force 2>/dev/null || true
    sleep 2
}

cleanup_bpf() {
    log "Cleaning up BPF struct_ops..."
    if [[ -x "$EXTENSION_DIR/cleanup_struct_ops_tool" ]]; then
        "$EXTENSION_DIR/cleanup_struct_ops_tool" 2>/dev/null || true
    fi
    # Also kill any running loaders or score_bridge
    pkill -f "attention_aware_eviction" 2>/dev/null || true
    pkill -f "prefetch_always_max_cycle_moe" 2>/dev/null || true
    pkill -f "score_bridge.py" 2>/dev/null || true
    sleep 1
    # Unpin stale maps
    rm -f /sys/fs/bpf/attention_score_map 2>/dev/null || true
    rm -f /sys/fs/bpf/attention_stats_map 2>/dev/null || true
}

wait_for_server() {
    local timeout=$1
    local elapsed=0
    while [[ $elapsed -lt $timeout ]]; do
        if curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        if (( elapsed % 30 == 0 )); then
            log "  Waiting for server... (${elapsed}s / ${timeout}s)"
        fi
    done
    return 1
}

stop_server() {
    local pid=$1
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        log "Stopping server (PID $pid)..."
        kill -TERM -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        sleep 5
        kill -9 -"$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
    sleep 3
}

# Run a single benchmark trial
# Args: config_name trial_num [env_vars...] -- [server_extra_args...]
run_trial() {
    local config="$1"
    local trial="$2"
    shift 2

    local env_vars=()
    local server_extra_args=()
    local parsing_env=1

    for arg in "$@"; do
        if [[ "$arg" == "--" ]]; then
            parsing_env=0
            continue
        fi
        if [[ $parsing_env -eq 1 ]]; then
            env_vars+=("$arg")
        else
            server_extra_args+=("$arg")
        fi
    done

    local output_file="$RESULT_DIR/${config}_trial${trial}.json"
    local log_file="$RESULT_DIR/${config}_trial${trial}.log"

    log "--- $config trial $trial/$NUM_TRIALS ---"

    cleanup_gpu

    # Build server command
    local server_cmd="uv run vllm serve $MODEL --enforce-eager --port $PORT"
    for extra in "${server_extra_args[@]}"; do
        server_cmd="$server_cmd $extra"
    done

    # Build environment
    local env_str=""
    for ev in "${env_vars[@]}"; do
        env_str="$env_str $ev"
    done

    log "  Server: ${env_str} $server_cmd"

    # Start server
    (
        cd "$WORKLOAD_DIR"
        env ${env_vars[@]+"${env_vars[@]}"} $server_cmd
    ) > "$log_file" 2>&1 &
    local server_pid=$!

    # Wait for server
    if ! wait_for_server "$SERVER_TIMEOUT"; then
        log "  ERROR: Server failed to start"
        stop_server "$server_pid"
        echo '{"error": "server_start_failed"}' > "$output_file"
        return 1
    fi

    log "  Server ready. Running benchmark ($NUM_PROMPTS prompts)..."

    # Run benchmark
    local bench_output
    bench_output=$(cd "$WORKLOAD_DIR" && uv run vllm bench serve \
        --model "$MODEL" \
        --dataset-name sharegpt \
        --dataset-path "$DATASET_PATH" \
        --num-prompts "$NUM_PROMPTS" \
        --sharegpt-output-len "$OUTPUT_LEN" \
        --seed 42 \
        --request-rate "$REQUEST_RATE" \
        --port "$PORT" 2>&1) || true

    # Save raw output
    echo "$bench_output" >> "$log_file"

    # Parse metrics with Python
    python3 -c "
import re, json, sys
output = sys.stdin.read()
metrics = {}
patterns = {
    'successful_requests': r'Successful requests:\s+(\d+)',
    'benchmark_duration_s': r'Benchmark duration \(s\):\s+([\d.]+)',
    'total_input_tokens': r'Total input tokens:\s+(\d+)',
    'total_generated_tokens': r'Total generated tokens:\s+(\d+)',
    'request_throughput_rps': r'Request throughput \(req/s\):\s+([\d.]+)',
    'output_throughput_tok_s': r'Output token throughput \(tok/s\):\s+([\d.]+)',
    'mean_ttft_ms': r'Mean TTFT \(ms\):\s+([\d.]+)',
    'median_ttft_ms': r'Median TTFT \(ms\):\s+([\d.]+)',
    'p99_ttft_ms': r'P99 TTFT \(ms\):\s+([\d.]+)',
    'mean_tpot_ms': r'Mean TPOT \(ms\):\s+([\d.]+)',
    'median_tpot_ms': r'Median TPOT \(ms\):\s+([\d.]+)',
    'p99_tpot_ms': r'P99 TPOT \(ms\):\s+([\d.]+)',
}
for key, pattern in patterns.items():
    match = re.search(pattern, output)
    if match:
        val = match.group(1)
        metrics[key] = float(val) if '.' in val else int(val)
result = {
    'config': '$config', 'trial': $trial,
    'model': '$MODEL', 'num_prompts': $NUM_PROMPTS,
    'metrics': metrics,
}
json.dump(result, open('$output_file', 'w'), indent=2)
print(json.dumps(metrics, indent=2))
" <<< "$bench_output"

    log "  Result saved: $output_file"

    # Stop server
    stop_server "$server_pid"
    return 0
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

log "=============================================="
log " Phase 1: Score-Aware Eviction Experiment"
log "=============================================="
log "Model       : $MODEL"
log "Prompts     : $NUM_PROMPTS"
log "Trials      : $NUM_TRIALS"
log "Result dir  : $RESULT_DIR"
log ""

# Check root
if [[ $EUID -ne 0 ]]; then
    die "This script must be run as root (for BPF operations)"
fi

# Check nvidia-smi
if ! command -v nvidia-smi &>/dev/null; then
    die "nvidia-smi not found. Is NVIDIA driver installed?"
fi
log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# Check CUDA
if ! command -v nvcc &>/dev/null; then
    log "Warning: nvcc not found, CUDA toolkit may not be in PATH"
fi

# Check dataset
if [[ ! -f "$DATASET_PATH" ]]; then
    log "Downloading ShareGPT dataset..."
    (cd "$WORKLOAD_DIR" && make download-datasets)
fi
[[ -f "$DATASET_PATH" ]] || die "Dataset not found: $DATASET_PATH"

# Check vLLM
if ! (cd "$WORKLOAD_DIR" && uv run python -c "import vllm" 2>/dev/null); then
    die "vLLM not installed. Run: cd workloads/vllm && uv pip install -e vllm/"
fi

# ---------------------------------------------------------------------------
# Build extension (unless --skip-build)
# ---------------------------------------------------------------------------

if [[ $SKIP_BUILD -eq 0 ]]; then
    log "Building BPF extensions..."
    (cd "$EXTENSION_DIR" && make attention_aware_eviction prefetch_always_max_cycle_moe cleanup_struct_ops_tool -j"$(nproc)") \
        || die "BPF build failed"
    log "Build complete."
else
    log "Skipping build (--skip-build)"
fi

# Verify binaries
[[ -x "$EXTENSION_DIR/attention_aware_eviction" ]] \
    || die "Binary not found: $EXTENSION_DIR/attention_aware_eviction"
[[ -x "$EXTENSION_DIR/prefetch_always_max_cycle_moe" ]] \
    || die "Binary not found: $EXTENSION_DIR/prefetch_always_max_cycle_moe"

# Create result directory
mkdir -p "$RESULT_DIR"

# Save experiment metadata
cat > "$RESULT_DIR/experiment_info.json" <<EOFMETA
{
  "experiment": "phase1_score_aware_eviction",
  "timestamp": "$TIMESTAMP",
  "model": "$MODEL",
  "num_prompts": $NUM_PROMPTS,
  "num_trials": $NUM_TRIALS,
  "request_rate": $REQUEST_RATE,
  "output_len": $OUTPUT_LEN,
  "max_num_seqs": $MAX_NUM_SEQS,
  "gpu": "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)",
  "driver": "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)",
  "kernel": "$(uname -r)"
}
EOFMETA

log "Experiment metadata saved."
log ""

# ---------------------------------------------------------------------------
# Config 1: cpu_offload (baseline, no UVM)
# ---------------------------------------------------------------------------

log "===== Config 1/5: cpu_offload ====="
cleanup_bpf

for trial in $(seq 1 "$NUM_TRIALS"); do
    run_trial "cpu_offload" "$trial" \
        -- --cpu-offload-gb 8
done

# ---------------------------------------------------------------------------
# Config 2: uvm_baseline (UVM, no BPF policy)
# ---------------------------------------------------------------------------

log "===== Config 2/5: uvm_baseline ====="
cleanup_bpf

for trial in $(seq 1 "$NUM_TRIALS"); do
    run_trial "uvm_baseline" "$trial" \
        VLLM_USE_UVM=1 \
        -- --max-num-seqs "$MAX_NUM_SEQS"
done

# ---------------------------------------------------------------------------
# Config 3: uvm_cycle_moe (UVM + existing best BPF policy)
# ---------------------------------------------------------------------------

log "===== Config 3/5: uvm_cycle_moe ====="
cleanup_bpf

# Start cycle_moe BPF loader
log "Loading cycle_moe BPF policy..."
"$EXTENSION_DIR/prefetch_always_max_cycle_moe" > "$RESULT_DIR/cycle_moe_bpf.log" 2>&1 &
CYCLE_MOE_PID=$!
sleep 2

if ! kill -0 "$CYCLE_MOE_PID" 2>/dev/null; then
    log "WARNING: cycle_moe BPF loader failed, check $RESULT_DIR/cycle_moe_bpf.log"
    log "Skipping config 3."
else
    for trial in $(seq 1 "$NUM_TRIALS"); do
        run_trial "uvm_cycle_moe" "$trial" \
            VLLM_USE_UVM=1 \
            -- --max-num-seqs "$MAX_NUM_SEQS"
    done
fi

cleanup_bpf

# ---------------------------------------------------------------------------
# Config 4: uvm_attention_no_score (attention-aware BPF, empty score_map)
#   Tests: T1 fallback behavior when no scores are provided
# ---------------------------------------------------------------------------

log "===== Config 4/5: uvm_attention_no_score ====="
cleanup_bpf

log "Loading attention-aware eviction BPF (no scores)..."
"$EXTENSION_DIR/attention_aware_eviction" --stats-interval 0 \
    > "$RESULT_DIR/attention_no_score_bpf.log" 2>&1 &
ATTENTION_PID=$!
sleep 2

if ! kill -0 "$ATTENTION_PID" 2>/dev/null; then
    log "WARNING: attention_aware BPF loader failed"
    log "Skipping config 4."
else
    for trial in $(seq 1 "$NUM_TRIALS"); do
        run_trial "uvm_attention_no_score" "$trial" \
            VLLM_USE_UVM=1 \
            -- --max-num-seqs "$MAX_NUM_SEQS"
    done
fi

cleanup_bpf

# ---------------------------------------------------------------------------
# Config 5: uvm_attention_scored (attention-aware BPF + score_bridge)
#   Tests: Full score-aware eviction with StreamingLLM heuristic
# ---------------------------------------------------------------------------

log "===== Config 5/5: uvm_attention_scored ====="
cleanup_bpf

log "Loading attention-aware eviction BPF (with scores)..."
"$EXTENSION_DIR/attention_aware_eviction" --stats-interval 30 \
    > "$RESULT_DIR/attention_scored_bpf.log" 2>&1 &
ATTENTION_PID=$!
sleep 2

if ! kill -0 "$ATTENTION_PID" 2>/dev/null; then
    log "WARNING: attention_aware BPF loader failed"
    log "Skipping config 5."
else
    # Determine KV cache VA range.
    # For Phase 1 we use a conservative estimate:
    #   Qwen3-30B-A3B-FP8 with block_size=16, ~4096 blocks, 256KB/block
    #   KV cache is typically allocated in the upper half of UVM VA space.
    #   We use 0x7f0000000000 as a placeholder; in production, score_bridge
    #   should read the actual VA from vLLM's allocator log.
    KV_BASE_VA="0x7f0000000000"
    KV_NUM_BLOCKS=4096
    KV_NUM_TOKENS=2048

    for trial in $(seq 1 "$NUM_TRIALS"); do
        cleanup_gpu

        # Start score_bridge before server
        log "  Starting score_bridge daemon..."
        python3 "$WORKLOAD_DIR/score_bridge.py" standalone \
            --kv-base-va "$KV_BASE_VA" \
            --num-blocks "$KV_NUM_BLOCKS" \
            --block-size-kb "$SCORE_BRIDGE_BLOCK_SIZE_KB" \
            --num-tokens "$KV_NUM_TOKENS" \
            --tokens-per-block "$SCORE_BRIDGE_TOKENS_PER_BLOCK" \
            --sink-tokens "$SCORE_BRIDGE_SINK_TOKENS" \
            --recent-window "$SCORE_BRIDGE_RECENT_WINDOW" \
            --interval 2.0 \
            > "$RESULT_DIR/score_bridge_trial${trial}.log" 2>&1 &
        SCORE_PID=$!
        sleep 1

        run_trial "uvm_attention_scored" "$trial" \
            VLLM_USE_UVM=1 \
            -- --max-num-seqs "$MAX_NUM_SEQS"

        # Stop score_bridge
        kill "$SCORE_PID" 2>/dev/null || true
        wait "$SCORE_PID" 2>/dev/null || true
    done
fi

cleanup_bpf

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

log ""
log "=============================================="
log " Experiment Complete"
log "=============================================="
log ""
log "Results in: $RESULT_DIR"
log ""

# Generate summary table
python3 - "$RESULT_DIR" <<'EOFPY'
import json, sys, os
from pathlib import Path
from collections import defaultdict

result_dir = Path(sys.argv[1])
configs = defaultdict(list)

for f in sorted(result_dir.glob("*.json")):
    if f.name == "experiment_info.json" or f.name == "summary.json":
        continue
    try:
        data = json.loads(f.read_text())
        if "error" in data:
            continue
        config = data.get("config", "unknown")
        metrics = data.get("metrics", {})
        if metrics:
            configs[config].append(metrics)
    except Exception:
        pass

if not configs:
    print("No valid results found.")
    sys.exit(0)

# Compute geometric mean for each metric
import math

def geo_mean(values):
    if not values or any(v <= 0 for v in values):
        return 0
    return math.exp(sum(math.log(v) for v in values) / len(values))

summary = {}
print(f"{'Config':<30} {'Trials':>6} {'TTFT(ms)':>10} {'TPOT(ms)':>10} "
      f"{'Throughput':>12} {'Duration(s)':>12}")
print("-" * 82)

for config in ["cpu_offload", "uvm_baseline", "uvm_cycle_moe",
               "uvm_attention_no_score", "uvm_attention_scored"]:
    if config not in configs:
        continue
    trials = configs[config]
    n = len(trials)
    ttft = geo_mean([t.get("mean_ttft_ms", 0) for t in trials])
    tpot = geo_mean([t.get("mean_tpot_ms", 0) for t in trials])
    throughput = geo_mean([t.get("output_throughput_tok_s", 0) for t in trials])
    duration = geo_mean([t.get("benchmark_duration_s", 0) for t in trials])

    print(f"{config:<30} {n:>6} {ttft:>10.1f} {tpot:>10.1f} "
          f"{throughput:>12.1f} {duration:>12.1f}")

    summary[config] = {
        "trials": n,
        "mean_ttft_ms": round(ttft, 1),
        "mean_tpot_ms": round(tpot, 1),
        "output_throughput_tok_s": round(throughput, 1),
        "benchmark_duration_s": round(duration, 1),
    }

# Save summary
summary_file = result_dir / "summary.json"
json.dump(summary, open(summary_file, "w"), indent=2)
print(f"\nSummary saved: {summary_file}")
EOFPY

log ""
log "Done. Review logs in $RESULT_DIR/"

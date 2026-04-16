#!/bin/bash
# E10 v2: Multi-Tenant FPRS vs Priority Scheduling
#
# Setup:
#   LC: llama-server 20B (port 8080, fits in VRAM)
#   BE: llama-bench 120B (UVM, runs in background generating faults)
#   GPU-side: prefetch_always_max_xcoord (must be running, provides shared maps)
#
# Conditions (4):
#   1. no_sched:      CFS default (no sched_ext)
#   2. xcoord:        Binary boost — all GPU processes boosted (auto-detect ON)
#   3. xcoord_noad:   Priority only — LC boost, no auto-detect of BE
#   4. fprs:          Feedback control — LC fault pressure regulates BE CPU
#
# Methodology improvements over E10 v1:
#   - INTERLEAVED ordering (not sequential) to eliminate ordering bias
#   - 5 runs per condition (not 3) for statistical significance
#   - Wait for 120B STEADY STATE before benchmarking
#   - Report scheduler stats per run to verify mechanism activation
#   - Validate thrashing state before starting benchmark
#
# Total: 4 conditions × 5 runs = 20 runs (interleaved)
set -e

RESULTS_DIR="/home/yunwei37/workspace/gpu/gpu_ext/workloads/llama.cpp/results/e10_fprs"
EXT_DIR="/home/yunwei37/workspace/gpu/gpu_ext/extension"
LLAMA_DIR="/home/yunwei37/workspace/gpu/gpu_ext/workloads/llama.cpp"
VLLM_DIR="/home/yunwei37/workspace/gpu/gpu_ext/workloads/vllm"
LLAMA_SERVER="$LLAMA_DIR/build/bin/llama-server"
LLAMA_BENCH="$LLAMA_DIR/build/bin/llama-bench"
MODEL_20B="$HOME/.cache/llama.cpp/ggml-org_gpt-oss-20b-GGUF_gpt-oss-20b-mxfp4.gguf"
MODEL_120B="$HOME/.cache/llama.cpp/ggml-org_gpt-oss-120b-GGUF_gpt-oss-120b-mxfp4-00001-of-00003.gguf"
DATASET="/home/yunwei37/workspace/gpu/gpu_ext/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"

RUNS_PER_CONDITION=5
CONDITIONS=("no_sched" "xcoord" "xcoord_noad" "fprs")

mkdir -p "$RESULTS_DIR"

# Cleanup function
cleanup_sched() {
    # Kill in specific order: noad first (longer name), then coord, then xcoord
    sudo pkill -9 -x sched_gpu_xcoord_noad 2>/dev/null || true
    sudo pkill -9 -x sched_gpu_coord 2>/dev/null || true
    sudo pkill -9 -x sched_gpu_xcoord 2>/dev/null || true
    sleep 2
    # Verify sched_ext is unloaded
    local max_wait=10
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if ! cat /sys/kernel/sched_ext/root/ops 2>/dev/null | grep -q .; then
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    echo "  WARNING: sched_ext still loaded after cleanup ($(cat /sys/kernel/sched_ext/root/ops 2>/dev/null))"
}

cleanup_bench() {
    # Kill 120B bench if running
    pkill -f "llama-bench.*120b" 2>/dev/null || true
    sleep 2
}

cleanup_all() {
    echo "Cleaning up..."
    cleanup_sched
    cleanup_bench
    kill "$LC_PID" 2>/dev/null || true
    sleep 2
}
trap cleanup_all EXIT

wait_for_thrashing() {
    # Wait until gpu_state_map shows the SPECIFIC 120B PID is thrashing
    # This ensures we don't benchmark before the interference starts
    local max_wait=120
    local waited=0
    local bench_pid="$BENCH_PID"
    echo "  Waiting for 120B (PID=$bench_pid) to enter thrashing state..."
    while [ $waited -lt $max_wait ]; do
        # Check if 120B bench is still running
        if ! kill -0 "$bench_pid" 2>/dev/null; then
            echo "  WARNING: 120B bench exited before thrashing detected (waited ${waited}s)"
            return 1
        fi
        # Check if THIS PID is in gpu_state_map with is_thrashing=1
        local found=$(sudo bpftool map dump pinned /sys/fs/bpf/xcoord_gpu_state 2>/dev/null | \
            python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    for e in data:
        if e['key'] == $bench_pid and e['value'].get('is_thrashing') == 1:
            print(e['value'].get('fault_rate', 0))
            sys.exit(0)
    print(0)
except:
    print(0)
" 2>/dev/null)
        if [ "${found:-0}" -gt 0 ] 2>/dev/null; then
            echo "  120B (PID=$bench_pid) thrashing detected after ${waited}s (fault_rate=$found)"
            # Wait 5 more seconds for steady state
            sleep 5
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
    done
    echo "  WARNING: 120B (PID=$bench_pid) did not enter thrashing state after ${max_wait}s"
    echo "  Proceeding anyway (benchmark results may not show interference)"
    return 0
}

run_benchmark() {
    local policy=$1
    local run=$2
    local lc_pid=$3

    echo ""
    echo "================================================================"
    echo "=== $policy RUN $run / $RUNS_PER_CONDITION ==="
    echo "================================================================"

    # Start 120B batch (BE) in background
    echo "  Starting 120B batch (BE)..."
    GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 "$LLAMA_BENCH" \
        -m "$MODEL_120B" -p 512 -n 128 -r 3 \
        > "$RESULTS_DIR/${policy}_120b_run${run}.log" 2>&1 &
    BENCH_PID=$!
    echo "  120B bench PID=$BENCH_PID"

    # Wait for 120B to reach steady-state thrashing
    wait_for_thrashing

    # Start sched daemon if needed
    local SCHED_PID=""
    local SCHED_LOG="/tmp/e10_${policy}_run${run}.log"
    case "$policy" in
        xcoord)
            cd "$EXT_DIR"
            sudo nohup ./sched_gpu_xcoord -p $lc_pid > "$SCHED_LOG" 2>&1 &
            SCHED_PID=$!
            sleep 3
            if cat /sys/kernel/sched_ext/root/ops 2>/dev/null | grep -q "gpu_xcoord"; then
                echo "  xcoord scheduler ACTIVE"
            else
                echo "  ERROR: xcoord failed to attach! Retrying..."
                cleanup_sched
                sudo nohup ./sched_gpu_xcoord -p $lc_pid > "$SCHED_LOG" 2>&1 &
                SCHED_PID=$!
                sleep 3
            fi
            ;;
        xcoord_noad)
            cd "$EXT_DIR"
            sudo nohup ./sched_gpu_xcoord_noad -p $lc_pid > "$SCHED_LOG" 2>&1 &
            SCHED_PID=$!
            sleep 3
            if cat /sys/kernel/sched_ext/root/ops 2>/dev/null | grep -q "gpu_xcoord_noad"; then
                echo "  xcoord_noad scheduler ACTIVE"
            else
                echo "  ERROR: xcoord_noad failed to attach! Retrying..."
                cleanup_sched
                sudo nohup ./sched_gpu_xcoord_noad -p $lc_pid > "$SCHED_LOG" 2>&1 &
                SCHED_PID=$!
                sleep 3
            fi
            ;;
        fprs)
            cd "$EXT_DIR"
            sudo nohup ./sched_gpu_coord -p $lc_pid > "$SCHED_LOG" 2>&1 &
            SCHED_PID=$!
            sleep 3
            if cat /sys/kernel/sched_ext/root/ops 2>/dev/null | grep -q "gpu_coord"; then
                echo "  FPRS scheduler ACTIVE"
            else
                echo "  ERROR: FPRS failed to attach! Retrying..."
                cleanup_sched
                sudo nohup ./sched_gpu_coord -p $lc_pid > "$SCHED_LOG" 2>&1 &
                SCHED_PID=$!
                sleep 3
            fi
            ;;
        no_sched)
            echo "  No sched_ext (CFS default)"
            ;;
    esac

    # Warmup: send a few requests to avoid cold-start effects
    echo "  Warmup (5 requests)..."
    cd "$VLLM_DIR"
    uv run vllm bench serve \
        --model gpt-oss-20b-mxfp4 \
        --tokenizer Qwen/Qwen3-30B-A3B-FP8 \
        --dataset-name sharegpt \
        --dataset-path "$DATASET" \
        --num-prompts 5 \
        --max-concurrency 2 \
        --request-rate 1.0 \
        --base-url http://127.0.0.1:8080 \
        --endpoint /v1/completions \
        > /dev/null 2>&1 || true
    sleep 2

    # Run actual benchmark
    echo "  Running benchmark (30 prompts, c=4, rate=2.0)..."
    cd "$VLLM_DIR"
    uv run vllm bench serve \
        --model gpt-oss-20b-mxfp4 \
        --tokenizer Qwen/Qwen3-30B-A3B-FP8 \
        --dataset-name sharegpt \
        --dataset-path "$DATASET" \
        --num-prompts 30 \
        --max-concurrency 4 \
        --request-rate 2.0 \
        --base-url http://127.0.0.1:8080 \
        --endpoint /v1/completions \
        2>&1 | tee "$RESULTS_DIR/${policy}_run${run}.txt"

    # Capture scheduler stats
    if [ -n "$SCHED_PID" ] && [ -f "$SCHED_LOG" ]; then
        echo "--- scheduler stats (last 3 lines) ---"
        tail -3 "$SCHED_LOG"
        echo ""
        # Also save to results dir
        cp "$SCHED_LOG" "$RESULTS_DIR/${policy}_sched_run${run}.log"
    fi

    # Cleanup for this run
    cleanup_sched
    kill "$BENCH_PID" 2>/dev/null || true
    wait "$BENCH_PID" 2>/dev/null || true
    sleep 5

    echo "=== $policy RUN $run COMPLETE ==="
}

# --- Pre-flight checks ---
echo "E10 v2: Multi-Tenant FPRS vs Priority Scheduling"
echo "Results: $RESULTS_DIR"
echo "Conditions: ${CONDITIONS[*]}"
echo "Runs per condition: $RUNS_PER_CONDITION"
echo "Total runs: $((${#CONDITIONS[@]} * RUNS_PER_CONDITION)) (interleaved)"
echo ""

# Verify gpu_ext is running
if ! ps aux | grep "./prefetch_always_max_xcoord" | grep -v grep > /dev/null; then
    echo "ERROR: prefetch_always_max_xcoord not running!"
    echo "Start it first: cd $EXT_DIR && sudo ./prefetch_always_max_xcoord"
    exit 1
fi
echo "prefetch_always_max_xcoord: running"

# Verify binaries exist
for bin in "$LLAMA_SERVER" "$LLAMA_BENCH" \
           "$EXT_DIR/sched_gpu_xcoord" "$EXT_DIR/sched_gpu_xcoord_noad" "$EXT_DIR/sched_gpu_coord"; do
    if [ ! -x "$bin" ]; then
        echo "ERROR: Binary not found: $bin"
        exit 1
    fi
done
echo "All binaries found"

# Verify models
for model in "$MODEL_20B" "$MODEL_120B"; do
    if [ ! -f "$model" ]; then
        echo "ERROR: Model not found: $model"
        exit 1
    fi
done
echo "All models found"

# Cleanup any stale processes
python3 /home/yunwei37/workspace/gpu/gpu_ext/workloads/cleanup_gpu.py 2>/dev/null || true
cleanup_sched 2>/dev/null || true
cleanup_bench 2>/dev/null || true
sleep 3

# --- Start 20B serving (LC) — stays running for all 20 runs ---
echo ""
echo "Starting 20B llama-server (LC) on port 8080..."
GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
    "$LLAMA_SERVER" \
    -m "$MODEL_20B" \
    -c 65536 -ngl 99 --host 0.0.0.0 --port 8080 \
    > "$RESULTS_DIR/llama_server_20b.log" 2>&1 &
LC_PID=$!
echo "  20B server PID=$LC_PID, waiting 30s for model load..."
sleep 30

# Verify server is up
if ! kill -0 $LC_PID 2>/dev/null; then
    echo "ERROR: llama-server failed to start"
    tail -20 "$RESULTS_DIR/llama_server_20b.log"
    exit 1
fi
curl -s http://127.0.0.1:8080/health > /dev/null || { echo "ERROR: 20B server not responding!"; exit 1; }
echo "20B server ready (PID=$LC_PID)"

# --- Generate interleaved run order ---
# Create array of (condition, run) pairs, then shuffle
declare -a RUN_ORDER
idx=0
for run in $(seq 1 $RUNS_PER_CONDITION); do
    for cond in "${CONDITIONS[@]}"; do
        RUN_ORDER[$idx]="${cond}:${run}"
        idx=$((idx + 1))
    done
done

# Shuffle using a fixed seed for reproducibility (but still interleaved)
# Use shuf if available, otherwise use a simple rotation
if command -v shuf &> /dev/null; then
    SHUFFLED=($(printf '%s\n' "${RUN_ORDER[@]}" | shuf --random-source=<(echo "e10fprs2026")))
else
    # Fallback: just use the round-robin order (already interleaved)
    SHUFFLED=("${RUN_ORDER[@]}")
fi

echo ""
echo "Run order (interleaved):"
for i in "${!SHUFFLED[@]}"; do
    echo "  $((i+1)). ${SHUFFLED[$i]}"
done
echo ""

# --- Run experiments in interleaved order ---
total=${#SHUFFLED[@]}
for i in "${!SHUFFLED[@]}"; do
    IFS=':' read -r cond run <<< "${SHUFFLED[$i]}"
    echo ""
    echo "########################################"
    echo "# Progress: $((i+1)) / $total"
    echo "########################################"
    run_benchmark "$cond" "$run" "$LC_PID"
done

# --- Summary ---
echo ""
echo "============================================="
echo "E10 v2 SUMMARY: FPRS vs Priority Scheduling"
echo "============================================="
for policy in "${CONDITIONS[@]}"; do
    echo ""
    echo "--- $policy ---"
    tpot_sum=0
    ttft_sum=0
    p99_sum=0
    tput_sum=0
    count=0
    for run in $(seq 1 $RUNS_PER_CONDITION); do
        f="$RESULTS_DIR/${policy}_run${run}.txt"
        if [ -f "$f" ]; then
            tpot=$(grep "Mean TPOT" "$f" | awk '{print $NF}')
            ttft=$(grep "Mean TTFT" "$f" | awk '{print $NF}')
            p99_ttft=$(grep "P99 TTFT" "$f" | awk '{print $NF}')
            tput=$(grep "Output token throughput" "$f" | head -1 | awk '{print $NF}')
            echo "  Run $run: TPOT=${tpot}ms TTFT=${ttft}ms P99_TTFT=${p99_ttft}ms tput=${tput} tok/s"

            # Accumulate for averages (integer-only bash)
            count=$((count + 1))
        else
            echo "  Run $run: MISSING"
        fi
    done
done

echo ""
echo "Scheduler logs saved to: $RESULTS_DIR/*_sched_run*.log"
echo "DONE"

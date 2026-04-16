#!/bin/bash
# POC-1: xCoord GPU→CPU Coordination via Shared BPF Maps
#
# Tests whether GPU-aware CPU scheduling (sched_ext reading GPU fault rate)
# improves UVM workload performance under CPU contention.
#
# Scenarios:
#   1. uvm_baseline        - 120B UVM, no CPU stress (reference)
#   2. uvm_cpu_stress      - 120B UVM + CPU stress, no xCoord (same as POC-0)
#   3. uvm_xcoord          - 120B UVM + CPU stress + xCoord (both gpu_ext + sched_ext)
#
# Success criteria (from POC-0 baseline):
#   - TTFT: from ~3160ms (cpu_stress) to <2000ms (xCoord)
#   - Throughput: from ~175.40 tok/s (cpu_stress) to >190 tok/s (xCoord)

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHUNK_TRACE="$REPO_ROOT/extension/chunk_trace"
EVICTION_XCOORD="$REPO_ROOT/extension/eviction_lfu_xcoord"
SCHED_GPU_AWARE="$REPO_ROOT/extension/sched_gpu_baseline"
LLAMA_SERVER="$REPO_ROOT/workloads/llama.cpp/build/bin/llama-server"
DATASET="$REPO_ROOT/workloads/llama.cpp/datasets/sharegpt_vicuna.json"
VLLM_WORKDIR="$REPO_ROOT/workloads/vllm"
TOKENIZER="Qwen/Qwen3-30B-A3B-FP8"

SERVER_PORT=8013
NUM_PROMPTS=20          # Match POC-0
MAX_CONCURRENCY=1
REQUEST_RATE=0.2
SERVER_WAIT=60

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$SCRIPT_DIR/results/poc1_xcoord_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

LLAMA_PID=""
TRACE_PID=""
STRESS_PID=""
XCOORD_GPU_PID=""
XCOORD_CPU_PID=""

echo "============================================================"
echo "POC-1: xCoord GPU→CPU Coordination"
echo "============================================================"
echo "Results: $RESULTS_DIR"
echo "Prompts: $NUM_PROMPTS"
echo ""

# Check prerequisites
for bin in "$LLAMA_SERVER" "$EVICTION_XCOORD" "$SCHED_GPU_AWARE" "$CHUNK_TRACE"; do
    if [ ! -x "$bin" ]; then
        echo "ERROR: Missing binary: $bin"
        echo "  Build with: cd extension && make"
        exit 1
    fi
done

if ! command -v stress-ng &>/dev/null; then
    echo "ERROR: stress-ng not installed"
    exit 1
fi

cleanup_all() {
    echo "  Cleaning up..."
    [ -n "$LLAMA_PID" ] && kill $LLAMA_PID 2>/dev/null || true
    [ -n "$TRACE_PID" ] && sudo kill $TRACE_PID 2>/dev/null || true
    [ -n "$STRESS_PID" ] && kill $STRESS_PID 2>/dev/null || true
    [ -n "$XCOORD_CPU_PID" ] && sudo kill $XCOORD_CPU_PID 2>/dev/null || true
    [ -n "$XCOORD_GPU_PID" ] && sudo kill $XCOORD_GPU_PID 2>/dev/null || true
    killall stress-ng 2>/dev/null || true
    [ -n "$LLAMA_PID" ] && wait $LLAMA_PID 2>/dev/null || true
    [ -n "$TRACE_PID" ] && wait $TRACE_PID 2>/dev/null || true
    [ -n "$STRESS_PID" ] && wait $STRESS_PID 2>/dev/null || true
    [ -n "$XCOORD_CPU_PID" ] && wait $XCOORD_CPU_PID 2>/dev/null || true
    [ -n "$XCOORD_GPU_PID" ] && wait $XCOORD_GPU_PID 2>/dev/null || true
    LLAMA_PID="" ; TRACE_PID="" ; STRESS_PID=""
    XCOORD_GPU_PID="" ; XCOORD_CPU_PID=""
    # Clean up pinned maps
    sudo rm -f /sys/fs/bpf/xcoord_gpu_state 2>/dev/null || true
    sudo rm -f /sys/fs/bpf/xcoord_uvm_workers 2>/dev/null || true
}

trap cleanup_all EXIT

start_xcoord() {
    local llama_pid=$1
    local scenario=$2

    echo "  Starting xCoord gpu_ext side (eviction_lfu_xcoord)..."
    sudo "$EVICTION_XCOORD" -p "$llama_pid" -P 1 -d 5 \
        > "$RESULTS_DIR/xcoord_gpu_${scenario}.log" 2>&1 &
    XCOORD_GPU_PID=$!
    sleep 2

    # Verify pinned map exists
    if ! sudo ls /sys/fs/bpf/xcoord_gpu_state &>/dev/null; then
        echo "  ERROR: gpu_state_map not pinned!"
        return 1
    fi
    echo "  gpu_ext loaded (PID: $XCOORD_GPU_PID)"

    echo "  Starting xCoord sched_ext side (sched_gpu_baseline)..."
    sudo "$SCHED_GPU_AWARE" -t 1000 \
        > "$RESULTS_DIR/xcoord_cpu_${scenario}.log" 2>&1 &
    XCOORD_CPU_PID=$!
    sleep 2

    # Verify sched_ext is active
    local sched_state=$(cat /sys/kernel/sched_ext/state 2>/dev/null || echo "unknown")
    echo "  sched_ext state: $sched_state (PID: $XCOORD_CPU_PID)"
    echo "  xCoord fully active."
}

stop_xcoord() {
    if [ -n "$XCOORD_CPU_PID" ]; then
        echo "  Stopping sched_gpu_baseline..."
        sudo kill $XCOORD_CPU_PID 2>/dev/null || true
        wait $XCOORD_CPU_PID 2>/dev/null || true
        XCOORD_CPU_PID=""
    fi
    if [ -n "$XCOORD_GPU_PID" ]; then
        echo "  Stopping eviction_lfu_xcoord..."
        sudo kill $XCOORD_GPU_PID 2>/dev/null || true
        wait $XCOORD_GPU_PID 2>/dev/null || true
        XCOORD_GPU_PID=""
    fi
    sudo rm -f /sys/fs/bpf/xcoord_gpu_state 2>/dev/null || true
    sudo rm -f /sys/fs/bpf/xcoord_uvm_workers 2>/dev/null || true
}

run_scenario() {
    local scenario=$1
    local with_stress=$2   # "yes" or "no"
    local with_xcoord=$3   # "yes" or "no"

    echo ""
    echo "============================================================"
    echo "Scenario: $scenario"
    echo "  CPU stress: $with_stress"
    echo "  xCoord: $with_xcoord"
    echo "============================================================"

    python3 "$REPO_ROOT/workloads/cleanup_gpu.py" 2>/dev/null || true
    sleep 2

    # Start chunk_trace
    echo "  Starting chunk_trace..."
    sudo "$CHUNK_TRACE" > "$RESULTS_DIR/trace_${scenario}.csv" 2>/dev/null &
    TRACE_PID=$!
    sleep 1

    # Start llama-server with UVM enabled
    echo "  Starting llama-server (120B, UVM)..."
    GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 GGML_CUDA_DISABLE_GRAPHS=1 \
        $LLAMA_SERVER --gpt-oss-120b-default -c 4096 -ngl 99 --host 0.0.0.0 --port $SERVER_PORT \
        > "$RESULTS_DIR/server_${scenario}.log" 2>&1 &
    LLAMA_PID=$!

    echo "  Waiting ${SERVER_WAIT}s for 120B model to load..."
    sleep "$SERVER_WAIT"

    # Check server health
    local retries=0
    while [ $retries -lt 20 ]; do
        if curl -s http://127.0.0.1:$SERVER_PORT/health > /dev/null 2>&1; then
            echo "  llama-server ready (PID: $LLAMA_PID)"
            break
        fi
        retries=$((retries + 1))
        if [ $retries -ge 20 ]; then
            echo "  ERROR: llama-server failed to become ready"
            cleanup_all
            return 1
        fi
        sleep 5
    done

    # Warmup
    echo "  Warmup (2 prompts)..."
    uv run --directory "$VLLM_WORKDIR" vllm bench serve \
        --model "gpt-oss-120b-mxfp4" \
        --tokenizer "$TOKENIZER" \
        --dataset-name sharegpt \
        --dataset-path "$DATASET" \
        --base-url "http://127.0.0.1:$SERVER_PORT" \
        --num-prompts 2 \
        --max-concurrency 1 \
        --request-rate 0.1 \
        > /dev/null 2>&1 || true
    sleep 3

    # Start xCoord if enabled (after warmup, before stress)
    if [ "$with_xcoord" = "yes" ]; then
        start_xcoord "$LLAMA_PID" "$scenario"
    fi

    # Start interference if needed
    if [ "$with_stress" = "yes" ]; then
        echo "  Starting CPU stress (all $(nproc) cores)..."
        stress-ng -c $(nproc) --timeout 600 > /dev/null 2>&1 &
        STRESS_PID=$!
        sleep 2
    fi

    # Run benchmark
    echo "  Running benchmark ($NUM_PROMPTS prompts)..."
    uv run --directory "$VLLM_WORKDIR" vllm bench serve \
        --model "gpt-oss-120b-mxfp4" \
        --tokenizer "$TOKENIZER" \
        --dataset-name sharegpt \
        --dataset-path "$DATASET" \
        --base-url "http://127.0.0.1:$SERVER_PORT" \
        --num-prompts $NUM_PROMPTS \
        --max-concurrency $MAX_CONCURRENCY \
        --request-rate $REQUEST_RATE \
        > "$RESULTS_DIR/bench_${scenario}.txt" 2>&1 || true

    echo "  Benchmark saved."

    # Stop stress
    if [ -n "$STRESS_PID" ]; then
        kill $STRESS_PID 2>/dev/null || true
        killall stress-ng 2>/dev/null || true
        STRESS_PID=""
    fi

    # Stop xCoord
    if [ "$with_xcoord" = "yes" ]; then
        stop_xcoord
    fi

    # Stop server
    kill $LLAMA_PID 2>/dev/null || true
    wait $LLAMA_PID 2>/dev/null || true
    LLAMA_PID=""

    # Stop trace
    sudo kill -INT $TRACE_PID 2>/dev/null || true
    sleep 2
    sudo kill $TRACE_PID 2>/dev/null || true
    wait $TRACE_PID 2>/dev/null || true
    TRACE_PID=""

    echo "  Scenario $scenario complete."
    sleep 5
}

# ============================================================
# Run scenarios
# ============================================================

# 1. Baseline: UVM only, no stress
run_scenario "uvm_baseline" "no" "no"

# 2. CPU stress without xCoord (reproduce POC-0 result)
run_scenario "uvm_cpu_stress" "yes" "no"

# 3. CPU stress WITH xCoord
run_scenario "uvm_xcoord" "yes" "yes"

echo ""
echo "============================================================"
echo "POC-1 All scenarios complete!"
echo "============================================================"
echo "Results: $RESULTS_DIR"
ls -la "$RESULTS_DIR/"
echo ""
echo "Quick comparison:"
echo "  grep -E 'TTFT|throughput' $RESULTS_DIR/bench_*.txt"
echo ""
echo "Run analysis:"
echo "  python3 $SCRIPT_DIR/analyze_poc0.py $RESULTS_DIR"

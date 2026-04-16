#!/bin/bash
# POC-0 Part 2: UVM-Heavy Experiment with 120B model
# The 120B model (~60GB) exceeds 32GB GPU memory → triggers UVM page faults during inference.
# This directly demonstrates CPU-GPU coupling via page fault handling.
#
# Uses --gpt-oss-120b-default flag (auto-selects model from cache)

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHUNK_TRACE="$REPO_ROOT/extension/chunk_trace"
LLAMA_SERVER="$REPO_ROOT/workloads/llama.cpp/build/bin/llama-server"
DATASET="$REPO_ROOT/workloads/llama.cpp/datasets/sharegpt_vicuna.json"
VLLM_WORKDIR="$REPO_ROOT/workloads/vllm"
TOKENIZER="Qwen/Qwen3-30B-A3B-FP8"

SERVER_PORT=8013
NUM_PROMPTS=20          # Fewer prompts - 120B is slow with UVM
MAX_CONCURRENCY=1
REQUEST_RATE=0.2        # Slower rate for large model
SERVER_WAIT=60          # Longer wait for 120B model load

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$SCRIPT_DIR/results/poc0_uvm_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

LLAMA_PID=""
TRACE_PID=""
STRESS_PID=""

echo "============================================================"
echo "POC-0 UVM-Heavy: 120B Model (exceeds GPU memory)"
echo "============================================================"
echo "Results: $RESULTS_DIR"
echo "Prompts: $NUM_PROMPTS"
echo ""

cleanup_all() {
    echo "  Cleaning up..."
    [ -n "$LLAMA_PID" ] && kill $LLAMA_PID 2>/dev/null || true
    [ -n "$TRACE_PID" ] && sudo kill $TRACE_PID 2>/dev/null || true
    [ -n "$STRESS_PID" ] && kill $STRESS_PID 2>/dev/null || true
    killall stress-ng 2>/dev/null || true
    [ -n "$LLAMA_PID" ] && wait $LLAMA_PID 2>/dev/null || true
    [ -n "$TRACE_PID" ] && wait $TRACE_PID 2>/dev/null || true
    [ -n "$STRESS_PID" ] && wait $STRESS_PID 2>/dev/null || true
    LLAMA_PID="" ; TRACE_PID="" ; STRESS_PID=""
}

trap cleanup_all EXIT

run_scenario() {
    local scenario=$1
    local with_stress=$2   # "yes" or "no"
    local pinned=$3        # "yes" or "no"

    echo ""
    echo "============================================================"
    echo "Scenario: $scenario"
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
    local server_cmd="$LLAMA_SERVER --gpt-oss-120b-default -c 4096 -ngl 99 --host 0.0.0.0 --port $SERVER_PORT"

    if [ "$pinned" = "yes" ]; then
        GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 GGML_CUDA_DISABLE_GRAPHS=1 \
            taskset -c 0-5 nice -n -10 $server_cmd \
            > "$RESULTS_DIR/server_${scenario}.log" 2>&1 &
    else
        GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 GGML_CUDA_DISABLE_GRAPHS=1 \
            $server_cmd \
            > "$RESULTS_DIR/server_${scenario}.log" 2>&1 &
    fi
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

    # Warmup (2 prompts)
    echo "  Warmup..."
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

    # Start interference if needed
    if [ "$with_stress" = "yes" ]; then
        echo "  Starting CPU stress..."
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

    # Stop everything
    [ -n "$STRESS_PID" ] && kill $STRESS_PID 2>/dev/null || true
    killall stress-ng 2>/dev/null || true
    STRESS_PID=""

    kill $LLAMA_PID 2>/dev/null || true
    wait $LLAMA_PID 2>/dev/null || true
    LLAMA_PID=""

    sudo kill -INT $TRACE_PID 2>/dev/null || true
    sleep 2
    sudo kill $TRACE_PID 2>/dev/null || true
    wait $TRACE_PID 2>/dev/null || true
    TRACE_PID=""

    echo "  Scenario $scenario complete."
    sleep 5
}

# Run scenarios
run_scenario "uvm_baseline" "no" "no"
run_scenario "uvm_cpu_stress" "yes" "no"
run_scenario "uvm_cpu_stress_pinned" "yes" "yes"

echo ""
echo "============================================================"
echo "All UVM-heavy scenarios complete!"
echo "============================================================"
echo "Results: $RESULTS_DIR"
ls -la "$RESULTS_DIR/"
echo ""
echo "Run analysis:"
echo "  python3 $SCRIPT_DIR/analyze_poc0.py $RESULTS_DIR"

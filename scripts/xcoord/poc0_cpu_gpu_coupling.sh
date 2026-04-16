#!/bin/bash
# POC-0: Quantify CPU-GPU Coupling
# Measures GPU hook call rates (activate/used/eviction_prepare) under different
# CPU interference scenarios to demonstrate CPU-GPU performance coupling.
#
# Prerequisites:
# - Custom nvidia_uvm module with BPF hooks loaded
# - chunk_trace built: cd extension && make chunk_trace
# - llama-server built: workloads/llama.cpp/build/bin/llama-server
# - ShareGPT dataset: workloads/llama.cpp/datasets/sharegpt_vicuna.json
# - vllm bench in workloads/vllm/.venv

set -eo pipefail

# === Configuration ===
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHUNK_TRACE="$REPO_ROOT/extension/chunk_trace"
LLAMA_SERVER="$REPO_ROOT/workloads/llama.cpp/build/bin/llama-server"
MODEL="$HOME/.cache/llama.cpp/ggml-org_gpt-oss-20b-GGUF_gpt-oss-20b-mxfp4.gguf"
DATASET="$REPO_ROOT/workloads/llama.cpp/datasets/sharegpt_vicuna.json"
VLLM_WORKDIR="$REPO_ROOT/workloads/vllm"
LLAMA_WORKDIR="$REPO_ROOT/workloads/llama.cpp"
TOKENIZER="Qwen/Qwen3-30B-A3B-FP8"

SERVER_PORT=8013
NUM_PROMPTS=50        # Fewer prompts for POC (faster iteration)
MAX_CONCURRENCY=1
REQUEST_RATE=1
SERVER_WAIT=30        # seconds to wait for model load

# Results directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$SCRIPT_DIR/results/poc0_$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "POC-0: CPU-GPU Coupling Experiment"
echo "============================================================"
echo "Results: $RESULTS_DIR"
echo "Model: $(basename $MODEL)"
echo "Prompts: $NUM_PROMPTS"
echo "Timestamp: $TIMESTAMP"
echo ""

# === Prerequisite checks ===
check_prereqs() {
    local ok=1
    [ -f "$CHUNK_TRACE" ] || { echo "ERROR: chunk_trace not found at $CHUNK_TRACE"; ok=0; }
    [ -f "$LLAMA_SERVER" ] || { echo "ERROR: llama-server not found at $LLAMA_SERVER"; ok=0; }
    [ -f "$MODEL" ] || { echo "ERROR: Model not found at $MODEL"; ok=0; }
    [ -f "$DATASET" ] || { echo "ERROR: Dataset not found at $DATASET"; ok=0; }
    sudo grep -q uvm_bpf_call /proc/kallsyms || { echo "ERROR: Custom nvidia_uvm module not loaded (no BPF hooks)"; ok=0; }
    which stress-ng > /dev/null 2>&1 || { echo "ERROR: stress-ng not installed"; ok=0; }
    [ $ok -eq 1 ] || { echo "Fix prerequisites and re-run."; exit 1; }
    echo "All prerequisites OK."
}

# === State variables ===
LLAMA_PID=""
TRACE_PID=""
STRESS_PID=""

# === Helper functions ===

cleanup_all() {
    echo "  Cleaning up..."
    [ -n "$LLAMA_PID" ] && kill $LLAMA_PID 2>/dev/null || true
    [ -n "$TRACE_PID" ] && sudo kill $TRACE_PID 2>/dev/null || true
    [ -n "$STRESS_PID" ] && kill $STRESS_PID 2>/dev/null || true
    killall stress-ng 2>/dev/null || true
    killall iperf3 2>/dev/null || true
    [ -n "$LLAMA_PID" ] && wait $LLAMA_PID 2>/dev/null || true
    [ -n "$TRACE_PID" ] && wait $TRACE_PID 2>/dev/null || true
    [ -n "$STRESS_PID" ] && wait $STRESS_PID 2>/dev/null || true
    LLAMA_PID="" ; TRACE_PID="" ; STRESS_PID=""
}

start_chunk_trace() {
    local output_file=$1
    echo "  Starting chunk_trace → $output_file"
    sudo "$CHUNK_TRACE" > "$output_file" 2>/dev/null &
    TRACE_PID=$!
    sleep 1
    if ! sudo kill -0 $TRACE_PID 2>/dev/null; then
        echo "  ERROR: chunk_trace failed to start"
        return 1
    fi
    echo "  chunk_trace running (PID: $TRACE_PID)"
}

stop_chunk_trace() {
    if [ -n "${TRACE_PID:-}" ]; then
        echo "  Stopping chunk_trace..."
        sudo kill -INT $TRACE_PID 2>/dev/null || true
        sleep 2
        sudo kill $TRACE_PID 2>/dev/null || true
        wait $TRACE_PID 2>/dev/null || true
        TRACE_PID=""
    fi
}

start_llama_server() {
    local mode=$1  # "normal" or "pinned"
    echo "  Starting llama-server (mode=$mode)..."

    local env_vars="GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 GGML_CUDA_DISABLE_GRAPHS=1"

    if [ "$mode" = "pinned" ]; then
        env $env_vars taskset -c 0-5 nice -n -10 "$LLAMA_SERVER" \
            -m "$MODEL" -c 65536 -ngl 99 --host 0.0.0.0 --port $SERVER_PORT \
            > "$RESULTS_DIR/server_${mode}.log" 2>&1 &
    else
        env $env_vars "$LLAMA_SERVER" \
            -m "$MODEL" -c 65536 -ngl 99 --host 0.0.0.0 --port $SERVER_PORT \
            > "$RESULTS_DIR/server_${mode}.log" 2>&1 &
    fi
    LLAMA_PID=$!

    echo "  Waiting ${SERVER_WAIT}s for model to load..."
    sleep "$SERVER_WAIT"

    # Check server health
    local retries=0
    while [ $retries -lt 10 ]; do
        if curl -s http://127.0.0.1:$SERVER_PORT/health > /dev/null 2>&1; then
            echo "  llama-server ready (PID: $LLAMA_PID)"
            return 0
        fi
        retries=$((retries + 1))
        sleep 5
    done
    echo "  ERROR: llama-server failed to become ready"
    return 1
}

stop_llama_server() {
    if [ -n "${LLAMA_PID:-}" ]; then
        echo "  Stopping llama-server..."
        kill $LLAMA_PID 2>/dev/null || true
        wait $LLAMA_PID 2>/dev/null || true
        LLAMA_PID=""
        sleep 3
    fi
}

run_benchmark() {
    local scenario=$1
    local bench_output="$RESULTS_DIR/bench_${scenario}.txt"

    echo "  Running vllm bench serve ($NUM_PROMPTS prompts)..."
    local model_name=$(basename "$MODEL" .gguf)

    uv run --directory "$VLLM_WORKDIR" vllm bench serve \
        --model "$model_name" \
        --tokenizer "$TOKENIZER" \
        --dataset-name sharegpt \
        --dataset-path "$DATASET" \
        --base-url "http://127.0.0.1:$SERVER_PORT" \
        --num-prompts $NUM_PROMPTS \
        --max-concurrency $MAX_CONCURRENCY \
        --request-rate $REQUEST_RATE \
        > "$bench_output" 2>&1 || true

    echo "  Benchmark output saved to $bench_output"
}

start_interference() {
    local scenario=$1
    case "$scenario" in
        baseline)
            echo "  No interference (baseline)"
            ;;
        cpu_stress)
            echo "  Starting CPU stress (all $(nproc) cores)..."
            stress-ng -c $(nproc) --timeout 600 > /dev/null 2>&1 &
            STRESS_PID=$!
            sleep 2
            ;;
        heavy_load)
            echo "  Starting heavy load (CPU + Network + Disk)..."
            stress-ng -c $(nproc) --timeout 600 > /dev/null 2>&1 &
            STRESS_PID=$!
            # Add network interference if iperf3 server available
            iperf3 -c 127.0.0.1 -t 600 -P 4 > /dev/null 2>&1 &
            sleep 2
            ;;
        cpu_stress_pinned)
            echo "  Starting CPU stress (pinned llama-server mode)..."
            stress-ng -c $(nproc) --timeout 600 > /dev/null 2>&1 &
            STRESS_PID=$!
            sleep 2
            ;;
    esac
}

stop_interference() {
    echo "  Stopping interference..."
    kill $STRESS_PID 2>/dev/null || true
    killall stress-ng 2>/dev/null || true
    killall iperf3 2>/dev/null || true
    STRESS_PID=""
    sleep 2
}

# === Run one scenario ===
run_scenario() {
    local scenario=$1
    local server_mode=$2  # "normal" or "pinned"

    echo ""
    echo "============================================================"
    echo "Scenario: $scenario (server_mode=$server_mode)"
    echo "============================================================"

    # Clean GPU
    python3 "$REPO_ROOT/workloads/cleanup_gpu.py" 2>/dev/null || true
    sleep 2

    # Start chunk_trace
    start_chunk_trace "$RESULTS_DIR/trace_${scenario}.csv"

    # Start llama-server
    if ! start_llama_server "$server_mode"; then
        stop_chunk_trace
        return 1
    fi

    # Warmup (5 prompts)
    echo "  Warmup (5 prompts)..."
    local model_name=$(basename "$MODEL" .gguf)
    uv run --directory "$VLLM_WORKDIR" vllm bench serve \
        --model "$model_name" \
        --tokenizer "$TOKENIZER" \
        --dataset-name sharegpt \
        --dataset-path "$DATASET" \
        --base-url "http://127.0.0.1:$SERVER_PORT" \
        --num-prompts 5 \
        --max-concurrency 1 \
        --request-rate 1 \
        > /dev/null 2>&1 || true
    sleep 2

    # Start interference
    start_interference "$scenario"

    # Run benchmark
    run_benchmark "$scenario"

    # Stop everything
    stop_interference
    stop_llama_server
    stop_chunk_trace

    echo "  Scenario $scenario complete."
}

# === Main ===
trap cleanup_all EXIT

check_prereqs

# Scenarios to run
SCENARIOS=(
    "baseline:normal"
    "cpu_stress:normal"
    "cpu_stress_pinned:pinned"
    "heavy_load:normal"
)

for entry in "${SCENARIOS[@]}"; do
    scenario="${entry%%:*}"
    server_mode="${entry##*:}"
    run_scenario "$scenario" "$server_mode"
done

echo ""
echo "============================================================"
echo "All scenarios complete!"
echo "============================================================"
echo ""
echo "Results directory: $RESULTS_DIR"
echo ""
echo "Files:"
ls -la "$RESULTS_DIR/"
echo ""
echo "Run analysis:"
echo "  python3 $SCRIPT_DIR/analyze_poc0.py $RESULTS_DIR"

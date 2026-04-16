#!/bin/bash
# E12: QoS-Driven Eviction vs Baseline
#
# Setup:
#   LC: llama-server 20B (UVM, port 8080, already running)
#   BE: FAISS SIFT100M IVF4096_Flat (-uvm)
#   GPU-side: prefetch_always_max_qos (QoS eviction protection)
#
# Conditions (3):
#   1. baseline:   prefetch_always_max_xcoord (no QoS eviction protection)
#   2. qos:        prefetch_always_max_qos -l <LC_PID> (QoS eviction protection)
#   3. no_faiss:   No FAISS (LC-only baseline for reference)
#
# Methodology:
#   - FAISS runs in background, generating GPU memory pressure
#   - LC receives continuous traffic (30 prompts, c=4)
#   - QoS policy protects LC pages from eviction via feedback controller
#   - Compare LC latency with/without QoS eviction protection
#
# Expected: QoS should dramatically reduce LC latency under FAISS pressure
#   (from ~2x degradation to ~1x, vs FPRS's ~5%)
set -e

RESULTS_DIR="/home/yunwei37/workspace/gpu/gpu_ext/workloads/llama.cpp/results/e12_qos_eviction"
EXT_DIR="/home/yunwei37/workspace/gpu/gpu_ext/extension"
LLAMA_DIR="/home/yunwei37/workspace/gpu/gpu_ext/workloads/llama.cpp"
FAISS_DIR="/home/yunwei37/workspace/gpu/gpu_ext/workloads/faiss"
VLLM_DIR="/home/yunwei37/workspace/gpu/gpu_ext/workloads/vllm"
LLAMA_SERVER="$LLAMA_DIR/build/bin/llama-server"
MODEL_20B="$HOME/.cache/llama.cpp/ggml-org_gpt-oss-20b-GGUF_gpt-oss-20b-mxfp4.gguf"
DATASET="/home/yunwei37/workspace/gpu/gpu_ext/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"

RUNS_PER_CONDITION=3
CONDITIONS=("no_faiss" "baseline" "qos")

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${RESULTS_DIR}/${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "E12: QoS-Driven Eviction vs Baseline"
echo "Results: $RESULTS_DIR"
echo ""

# --- Pre-flight checks ---
# Verify 20B server is running
LC_PID=$(pgrep -f "llama-server.*20b" | head -1)
if [ -z "$LC_PID" ]; then
    echo "ERROR: llama-server 20B not running!"
    echo "Start it first with GGML_CUDA_ENABLE_UNIFIED_MEMORY=1"
    exit 1
fi
curl -s http://127.0.0.1:8080/health > /dev/null || { echo "ERROR: 20B server not responding!"; exit 1; }
echo "20B server: PID=$LC_PID, healthy"

# Verify binaries
for bin in "$EXT_DIR/prefetch_always_max_qos" "$EXT_DIR/prefetch_always_max_xcoord"; do
    if [ ! -x "$bin" ]; then
        echo "ERROR: Binary not found: $bin"
        exit 1
    fi
done
echo "All binaries found"

# Cleanup functions
FAISS_PID=""
GPU_POLICY_PID=""

cleanup_gpu_policy() {
    if [ -n "$GPU_POLICY_PID" ]; then
        sudo kill "$GPU_POLICY_PID" 2>/dev/null || true
        sleep 3
    fi
    # Also try by name
    sudo pkill -x prefetch_always_max_qos 2>/dev/null || true
    sudo pkill -x prefetch_always_max_xcoord 2>/dev/null || true
    sleep 2
    GPU_POLICY_PID=""
}

cleanup_faiss() {
    if [ -n "$FAISS_PID" ]; then
        kill "$FAISS_PID" 2>/dev/null || true
        wait "$FAISS_PID" 2>/dev/null || true
    fi
    # Kill any remaining FAISS processes
    pkill -f "bench_gpu_1bn.*uvm" 2>/dev/null || true
    sleep 2
    FAISS_PID=""
}

cleanup_all() {
    echo "Cleaning up..."
    cleanup_faiss
    # Don't kill the xcoord policy — leave it for user to manage
}
trap cleanup_all EXIT

start_faiss() {
    echo "  Starting FAISS SIFT100M (BE, UVM mode)..."
    cd "$FAISS_DIR"
    uv run python bench_gpu_1bn.py \
        SIFT100M IVF4096,Flat \
        -uvm \
        -nprobe 1,4,16 \
        > "$RESULTS_DIR/faiss_${1}_run${2}.log" 2>&1 &
    FAISS_PID=$!
    echo "  FAISS PID=$FAISS_PID"
}

wait_for_faiss_interference() {
    # Wait until gpu_state_map shows high fault rate for FAISS
    local max_wait=180
    local waited=0
    echo "  Waiting for FAISS to create GPU memory pressure..."
    while [ $waited -lt $max_wait ]; do
        if ! kill -0 "$FAISS_PID" 2>/dev/null; then
            echo "  WARNING: FAISS exited before interference (waited ${waited}s)"
            return 1
        fi
        # Check LC fault_rate via gpu_state_map
        local lc_fr=$(sudo bpftool map dump pinned /sys/fs/bpf/xcoord_gpu_state 2>/dev/null | \
            python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    for e in data:
        k = e.get('formatted',{}).get('key', e.get('key'))
        v = e.get('formatted',{}).get('value', e.get('value',{}))
        if str(k) == '$LC_PID':
            print(v.get('fault_rate', 0))
            sys.exit(0)
    print(0)
except:
    print(0)
" 2>/dev/null)
        if [ "${lc_fr:-0}" -gt 100 ] 2>/dev/null; then
            echo "  LC (PID=$LC_PID) under pressure: fault_rate=$lc_fr (waited ${waited}s)"
            sleep 5
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
    done
    echo "  WARNING: LC did not show pressure after ${max_wait}s, proceeding anyway"
    return 0
}

run_lc_benchmark() {
    local tag=$1
    local run=$2

    echo "  Warmup (3 requests)..."
    cd "$VLLM_DIR"
    uv run vllm bench serve \
        --model gpt-oss-20b-mxfp4 \
        --tokenizer Qwen/Qwen3-30B-A3B-FP8 \
        --dataset-name sharegpt \
        --dataset-path "$DATASET" \
        --num-prompts 3 \
        --max-concurrency 2 \
        --request-rate 1.0 \
        --base-url http://127.0.0.1:8080 \
        --endpoint /v1/completions \
        > /dev/null 2>&1 || true
    sleep 2

    echo "  Running LC benchmark (30 prompts, c=4, rate=2.0)..."
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
        2>&1 | tee "$RESULTS_DIR/${tag}_run${run}.txt"
}

# ================================================================
# Run experiments
# ================================================================

for run in $(seq 1 $RUNS_PER_CONDITION); do
    for cond in "${CONDITIONS[@]}"; do
        echo ""
        echo "================================================================"
        echo "=== $cond RUN $run / $RUNS_PER_CONDITION ==="
        echo "================================================================"

        case "$cond" in
            no_faiss)
                # LC-only baseline — just make sure existing xcoord policy is loaded
                # (no need to swap policy; we just don't start FAISS)
                echo "  Condition: no_faiss (LC-only reference)"

                # Make sure existing policy is running
                if ! ps aux | grep "./prefetch_always_max_xcoord" | grep -v grep > /dev/null 2>&1; then
                    echo "  Starting prefetch_always_max_xcoord..."
                    cd "$EXT_DIR"
                    sudo nohup ./prefetch_always_max_xcoord > "$RESULTS_DIR/xcoord_policy.log" 2>&1 &
                    GPU_POLICY_PID=$!
                    sleep 5
                fi

                run_lc_benchmark "no_faiss" "$run"
                ;;

            baseline)
                # FAISS + xcoord (no QoS protection)
                echo "  Condition: baseline (FAISS + xcoord, no QoS eviction protection)"

                # Make sure we're running xcoord (not qos)
                cleanup_gpu_policy
                echo "  Starting prefetch_always_max_xcoord..."
                cd "$EXT_DIR"
                sudo nohup ./prefetch_always_max_xcoord > "$RESULTS_DIR/xcoord_baseline_run${run}.log" 2>&1 &
                GPU_POLICY_PID=$!
                sleep 5

                start_faiss "baseline" "$run"
                wait_for_faiss_interference
                run_lc_benchmark "baseline" "$run"

                # Save gpu_state_map snapshot
                sudo bpftool map dump pinned /sys/fs/bpf/xcoord_gpu_state 2>/dev/null \
                    > "$RESULTS_DIR/baseline_gpustate_run${run}.json" || true

                cleanup_faiss
                sleep 5
                ;;

            qos)
                # FAISS + QoS eviction protection
                echo "  Condition: qos (FAISS + QoS eviction protection for LC PID=$LC_PID)"

                # Swap to QoS policy
                cleanup_gpu_policy
                echo "  Starting prefetch_always_max_qos -l $LC_PID..."
                cd "$EXT_DIR"
                sudo nohup ./prefetch_always_max_qos -l "$LC_PID" -v \
                    > "$RESULTS_DIR/qos_policy_run${run}.log" 2>&1 &
                GPU_POLICY_PID=$!
                sleep 5

                start_faiss "qos" "$run"
                wait_for_faiss_interference
                run_lc_benchmark "qos" "$run"

                # Save gpu_state_map snapshot + controller state
                sudo bpftool map dump pinned /sys/fs/bpf/xcoord_gpu_state 2>/dev/null \
                    > "$RESULTS_DIR/qos_gpustate_run${run}.json" || true
                echo "--- QoS controller state ---" >> "$RESULTS_DIR/qos_policy_run${run}.log"
                tail -20 "$RESULTS_DIR/qos_policy_run${run}.log"

                cleanup_faiss
                sleep 5
                ;;
        esac

        echo "=== $cond RUN $run COMPLETE ==="
        sleep 3
    done
done

# Restore original xcoord policy
cleanup_gpu_policy
echo ""
echo "Restoring prefetch_always_max_xcoord..."
cd "$EXT_DIR"
sudo nohup ./prefetch_always_max_xcoord > /dev/null 2>&1 &
sleep 3

# --- Summary ---
echo ""
echo "============================================="
echo "E12 SUMMARY: QoS-Driven Eviction"
echo "============================================="
for cond in "${CONDITIONS[@]}"; do
    echo ""
    echo "--- $cond ---"
    for run in $(seq 1 $RUNS_PER_CONDITION); do
        f="$RESULTS_DIR/${cond}_run${run}.txt"
        if [ -f "$f" ]; then
            tpot=$(grep "Mean TPOT" "$f" 2>/dev/null | awk '{print $NF}' || echo "N/A")
            ttft=$(grep "Mean TTFT" "$f" 2>/dev/null | awk '{print $NF}' || echo "N/A")
            p99_ttft=$(grep "P99 TTFT" "$f" 2>/dev/null | awk '{print $NF}' || echo "N/A")
            tput=$(grep "Output token throughput" "$f" 2>/dev/null | head -1 | awk '{print $NF}' || echo "N/A")
            echo "  Run $run: TPOT=${tpot}ms TTFT=${ttft}ms P99_TTFT=${p99_ttft}ms tput=${tput} tok/s"
        else
            echo "  Run $run: MISSING"
        fi
    done
done

echo ""
echo "QoS controller logs: $RESULTS_DIR/qos_policy_run*.log"
echo "FAISS logs: $RESULTS_DIR/faiss_*.log"
echo "GPU state snapshots: $RESULTS_DIR/*_gpustate_run*.json"
echo "DONE"

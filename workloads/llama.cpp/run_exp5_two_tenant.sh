#!/usr/bin/env bash
# =============================================================================
# Experiment 5: Two-Tenant Co-location (Paper RQ2, Figure 12)
#
# T1 (LC): llama.cpp inference server (gpt-oss-20b, UVM, ctx=65536)
#          Serving 100 ShareGPT prompts at 0.2 RPS
# T2 (BE): PyTorch GNN training (8M nodes, UVM, 25 epochs, 36GB peak)
#
# Configs without modified KM: 1 (default UVM)
# Configs needing modified KM: 2 (gpu_ext per-tenant policies)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKLOADS_DIR="$(dirname "$SCRIPT_DIR")"
PYTORCH_DIR="$WORKLOADS_DIR/pytorch"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$SCRIPT_DIR/results/exp5_two_tenant/${TIMESTAMP}"
LLAMA_SERVER="$SCRIPT_DIR/build/bin/llama-server"
MODEL_20B="${MODEL_20B_CACHE:-$HOME/.cache/llama.cpp/ggml-org_gpt-oss-20b-GGUF_gpt-oss-20b-mxfp4.gguf}"
DATASET="$SCRIPT_DIR/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
GNN_BENCHMARK="$PYTORCH_DIR/benchmark_gnn_uvm.py"
GNN_EPOCHS="${GNN_EPOCHS:-25}"
GNN_NODES="${GNN_NODES:-8000000}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
REQUEST_RATE="${REQUEST_RATE:-0.2}"

mkdir -p "$RESULTS_DIR"

# Log everything
exec > >(tee -a "$RESULTS_DIR/full.log") 2>&1

echo "============================================================"
echo "Experiment 5: Two-Tenant Co-location (Figure 12)"
echo "============================================================"
echo "Timestamp: $TIMESTAMP"
echo "T1 (LC): llama.cpp 20B, UVM, ctx=65536, $NUM_PROMPTS prompts @ ${REQUEST_RATE} RPS"
echo "T2 (BE): GNN ${GNN_NODES} nodes, UVM, $GNN_EPOCHS epochs"
echo "Results dir: $RESULTS_DIR"
echo ""

# --- Prerequisites ---
if [ ! -f "$LLAMA_SERVER" ]; then
    echo "ERROR: llama-server not found at $LLAMA_SERVER"
    echo "Run: cd $SCRIPT_DIR && make build-cuda-no-vmm"
    exit 1
fi

if [ ! -f "$MODEL_20B" ]; then
    echo "ERROR: GPT-OSS-20B model not found at $MODEL_20B"
    exit 1
fi

if [ ! -f "$GNN_BENCHMARK" ]; then
    echo "ERROR: benchmark_gnn_uvm.py not found at $GNN_BENCHMARK"
    exit 1
fi

# Download ShareGPT if needed
if [ ! -f "$DATASET" ]; then
    echo "Downloading ShareGPT dataset..."
    python3 "$SCRIPT_DIR/download_sharegpt.py" --dataset vicuna
fi

# --- GPU Cleanup ---
echo "Cleaning GPU processes..."
python3 "$WORKLOADS_DIR/cleanup_gpu.py" 2>/dev/null || true
sleep 2

# ==========================================================
# Phase 1: Single-tenant baselines
# ==========================================================

# --- Single llama.cpp baseline ---
echo ""
echo "============================================================"
echo "Phase 1a: Single-tenant llama.cpp baseline"
echo "============================================================"

GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
    "$LLAMA_SERVER" \
    -m "$MODEL_20B" \
    -c 65536 -ngl 99 --host 0.0.0.0 --port 8013 \
    > "$RESULTS_DIR/llama_server_single.log" 2>&1 &
LLAMA_PID=$!
echo "Started llama-server (PID=$LLAMA_PID), waiting for model load..."
sleep 30

# Check if server is up
if ! kill -0 $LLAMA_PID 2>/dev/null; then
    echo "ERROR: llama-server failed to start"
    cat "$RESULTS_DIR/llama_server_single.log"
    exit 1
fi

# Run benchmark client
uv run --directory "$SCRIPT_DIR" vllm bench serve \
    --model gpt-oss-20b \
    --dataset-name sharegpt --num-prompts "$NUM_PROMPTS" \
    --dataset-path "$DATASET" \
    --base-url http://127.0.0.1:8013 \
    --max-concurrency=1 --request-rate "$REQUEST_RATE" \
    2>&1 | tee "$RESULTS_DIR/single_llama_bench.log" || true

kill $LLAMA_PID 2>/dev/null || true
wait $LLAMA_PID 2>/dev/null || true
sleep 5

# --- Single GNN baseline ---
echo ""
echo "============================================================"
echo "Phase 1b: Single-tenant GNN baseline"
echo "============================================================"
python3 "$WORKLOADS_DIR/cleanup_gpu.py" 2>/dev/null || true
sleep 2

CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 uv run --directory "$PYTORCH_DIR" python "$GNN_BENCHMARK" \
    --dataset random --nodes "$GNN_NODES" \
    --edges_per_node 10 --features 128 --hidden 256 \
    --epochs "$GNN_EPOCHS" --warmup 1 --prop chunked --use_uvm \
    --report_json "$RESULTS_DIR/single_gnn.json" \
    2>&1 | tee "$RESULTS_DIR/single_gnn.log" || {
    echo "Single GNN baseline FAILED"
}

# ==========================================================
# Phase 2: Co-located (default UVM, no policy)
# ==========================================================
echo ""
echo "============================================================"
echo "Phase 2: Co-located (default UVM)"
echo "============================================================"
python3 "$WORKLOADS_DIR/cleanup_gpu.py" 2>/dev/null || true
sleep 2

# Start llama-server
GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
    "$LLAMA_SERVER" \
    -m "$MODEL_20B" \
    -c 65536 -ngl 99 --host 0.0.0.0 --port 8013 \
    > "$RESULTS_DIR/llama_server_colocated.log" 2>&1 &
LLAMA_PID=$!
echo "Started llama-server (PID=$LLAMA_PID), waiting for model load..."
sleep 30

if ! kill -0 $LLAMA_PID 2>/dev/null; then
    echo "ERROR: llama-server failed to start"
    cat "$RESULTS_DIR/llama_server_colocated.log"
    exit 1
fi

# Start GNN training in background
echo "Starting GNN training (background)..."
CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 uv run --directory "$PYTORCH_DIR" python "$GNN_BENCHMARK" \
    --dataset random --nodes "$GNN_NODES" \
    --edges_per_node 10 --features 128 --hidden 256 \
    --epochs "$GNN_EPOCHS" --warmup 1 --prop chunked --use_uvm \
    --report_json "$RESULTS_DIR/colocated_gnn_baseline.json" \
    > "$RESULTS_DIR/colocated_gnn_baseline.log" 2>&1 &
GNN_PID=$!
echo "Started GNN training (PID=$GNN_PID)"
sleep 10

# Run llama benchmark client
echo "Starting llama.cpp benchmark client..."
uv run --directory "$SCRIPT_DIR" vllm bench serve \
    --model gpt-oss-20b \
    --dataset-name sharegpt --num-prompts "$NUM_PROMPTS" \
    --dataset-path "$DATASET" \
    --base-url http://127.0.0.1:8013 \
    --max-concurrency=1 --request-rate "$REQUEST_RATE" \
    2>&1 | tee "$RESULTS_DIR/colocated_llama_bench.log" || true

# Wait for GNN to finish
echo "Waiting for GNN training to finish..."
wait $GNN_PID 2>/dev/null || true

# Cleanup
kill $LLAMA_PID 2>/dev/null || true
wait $LLAMA_PID 2>/dev/null || true

# --- Summary ---
echo ""
echo "============================================================"
echo "RESULTS SUMMARY (Experiment 5: Two-Tenant Co-location)"
echo "============================================================"
echo ""

echo "Single-tenant llama.cpp:"
grep -E "TPOT|TTFT|throughput" "$RESULTS_DIR/single_llama_bench.log" 2>/dev/null || echo "  (check log manually)"
echo ""

echo "Single-tenant GNN:"
if [ -f "$RESULTS_DIR/single_gnn.json" ]; then
    python3 -c "import json; d=json.load(open('$RESULTS_DIR/single_gnn.json')); print(f\"  Avg epoch: {d['avg_epoch_time_s']:.3f}s\")" 2>/dev/null || echo "  (check log manually)"
fi
echo ""

echo "Co-located (default UVM) llama.cpp:"
grep -E "TPOT|TTFT|throughput" "$RESULTS_DIR/colocated_llama_bench.log" 2>/dev/null || echo "  (check log manually)"
echo ""

echo "Co-located (default UVM) GNN:"
if [ -f "$RESULTS_DIR/colocated_gnn_baseline.json" ]; then
    python3 -c "import json; d=json.load(open('$RESULTS_DIR/colocated_gnn_baseline.json')); print(f\"  Avg epoch: {d['avg_epoch_time_s']:.3f}s\")" 2>/dev/null || echo "  (check log manually)"
fi
echo ""

echo "Paper reference:"
echo "  Single llama.cpp TPOT: 3.67ms"
echo "  Co-located (UVM) TPOT: 19.73ms"
echo "  Co-located (UVM) GNN epoch: 23.23s"
echo ""
echo "NOTE: Config 2 (gpu_ext per-tenant) requires modified kernel module."
echo "Full logs saved to: $RESULTS_DIR/"

#!/usr/bin/env bash
# =============================================================================
# Experiment 3: PyTorch GNN Training (Paper RQ1, Figure 8)
#
# Model: 2-layer GCN, random graphs, 10 edges/node, 128 features, hidden 256
# GPU: RTX 5090 (32GB)
# Metrics: epoch time (s), memory usage
# Trials: 10 epochs measured (2 warmup + 10 measured)
#
# Configs without modified KM:
#   1. Native GPU (default allocator, 1M-7M nodes)
#   2. UVM baseline (5M-15M nodes)
# Configs needing modified KM:
#   3. UVM + gpu_ext eBPF (5M-15M nodes)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKLOADS_DIR="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$SCRIPT_DIR/results/exp3_gnn/${TIMESTAMP}"
BENCHMARK="$SCRIPT_DIR/benchmark_gnn_uvm.py"
EPOCHS="${EPOCHS:-10}"
WARMUP="${WARMUP:-2}"

mkdir -p "$RESULTS_DIR"

# Log everything
exec > >(tee -a "$RESULTS_DIR/full.log") 2>&1

echo "============================================================"
echo "Experiment 3: PyTorch GNN Training (Figure 8)"
echo "============================================================"
echo "Timestamp: $TIMESTAMP"
echo "Epochs: $EPOCHS (warmup: $WARMUP)"
echo "Results dir: $RESULTS_DIR"
echo ""

# --- Prerequisites ---
if [ ! -f "$BENCHMARK" ]; then
    echo "ERROR: benchmark_gnn_uvm.py not found at $BENCHMARK"
    exit 1
fi

# Check for allocator .so files
if [ ! -f "$SCRIPT_DIR/uvm_allocator.so" ]; then
    echo "Building allocators..."
    make -C "$SCRIPT_DIR" all
fi

# --- GPU Cleanup ---
echo "Cleaning GPU processes..."
python3 "$WORKLOADS_DIR/cleanup_gpu.py" 2>/dev/null || true
sleep 2

# Common args
COMMON_ARGS="--dataset random --edges_per_node 10 --features 128 --hidden 256 --epochs $EPOCHS --warmup $WARMUP --prop chunked"

# ==========================================================
# Config 1: Native GPU (default allocator, 1M-7M)
# ==========================================================
echo ""
echo "============================================================"
echo "Config 1: Native GPU (default allocator)"
echo "============================================================"

NATIVE_NODES="1000000 3000000 5000000 7000000"
mkdir -p "$RESULTS_DIR/native_gpu"

for NODES in $NATIVE_NODES; do
    NODES_M=$((NODES / 1000000))
    echo ""
    echo "--- Native GPU: ${NODES_M}M nodes ---"
    python3 "$WORKLOADS_DIR/cleanup_gpu.py" 2>/dev/null || true
    sleep 2

    uv run --directory "$SCRIPT_DIR" python "$BENCHMARK" \
        $COMMON_ARGS \
        --nodes "$NODES" \
        --report_json "$RESULTS_DIR/native_gpu/${NODES_M}M.json" \
        2>&1 | tee "$RESULTS_DIR/native_gpu/${NODES_M}M.log" || {
        echo "  FAILED (likely OOM) at ${NODES_M}M nodes"
    }
done

# ==========================================================
# Config 2: UVM Baseline (5M-15M)
# ==========================================================
echo ""
echo "============================================================"
echo "Config 2: UVM Baseline (default driver, no policy)"
echo "============================================================"

UVM_NODES="5000000 7000000 8000000 10000000 12000000 15000000"
mkdir -p "$RESULTS_DIR/uvm_baseline"

for NODES in $UVM_NODES; do
    NODES_M=$((NODES / 1000000))
    echo ""
    echo "--- UVM Baseline: ${NODES_M}M nodes ---"
    python3 "$WORKLOADS_DIR/cleanup_gpu.py" 2>/dev/null || true
    sleep 2

    CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 uv run --directory "$SCRIPT_DIR" python "$BENCHMARK" \
        $COMMON_ARGS \
        --nodes "$NODES" \
        --use_uvm \
        --report_json "$RESULTS_DIR/uvm_baseline/${NODES_M}M.json" \
        2>&1 | tee "$RESULTS_DIR/uvm_baseline/${NODES_M}M.log" || {
        echo "  FAILED at ${NODES_M}M nodes"
    }
done

# --- Summary ---
echo ""
echo "============================================================"
echo "RESULTS SUMMARY (Experiment 3: GNN Training)"
echo "============================================================"
echo ""
echo "Config 1: Native GPU (default allocator)"
echo "| Nodes | Avg Epoch Time | GPU Alloc |"
echo "|-------|---------------|-----------|"
for NODES in $NATIVE_NODES; do
    NODES_M=$((NODES / 1000000))
    JSON="$RESULTS_DIR/native_gpu/${NODES_M}M.json"
    if [ -f "$JSON" ]; then
        AVG=$(python3 -c "import json; d=json.load(open('$JSON')); print(f\"{d['avg_epoch_time_s']:.3f}s\")" 2>/dev/null || echo "N/A")
        MEM=$(python3 -c "import json; d=json.load(open('$JSON')); print(f\"{d['memory']['gpu_allocated_GB']:.2f} GB\")" 2>/dev/null || echo "N/A")
        echo "| ${NODES_M}M | $AVG | $MEM |"
    else
        echo "| ${NODES_M}M | FAILED | — |"
    fi
done

echo ""
echo "Config 2: UVM Baseline"
echo "| Nodes | Avg Epoch Time | Peak UVM Alloc |"
echo "|-------|---------------|----------------|"
for NODES in $UVM_NODES; do
    NODES_M=$((NODES / 1000000))
    JSON="$RESULTS_DIR/uvm_baseline/${NODES_M}M.json"
    if [ -f "$JSON" ]; then
        AVG=$(python3 -c "import json; d=json.load(open('$JSON')); print(f\"{d['avg_epoch_time_s']:.3f}s\")" 2>/dev/null || echo "N/A")
        PEAK=$(python3 -c "import json; d=json.load(open('$JSON')); print(f\"{d.get('uvm_stats',{}).get('peak_allocated_GB','N/A'):.2f} GB\")" 2>/dev/null || echo "N/A")
        echo "| ${NODES_M}M | $AVG | $PEAK |"
    else
        echo "| ${NODES_M}M | FAILED | — |"
    fi
done

echo ""
echo "Paper reference (without user prefetch, epoch time):"
echo "  Native GPU:    5M=1.14s  7M=1.79s  8M=OOM"
echo "  UVM baseline:  5M=34.23s 7M=48.28s 8M=55.36s 10M=70.06s 12M=93.71s 15M=292.77s"
echo ""
echo "NOTE: Configs 4 (UVM+gpu_ext) and 5 (UVM+prefetch+gpu_ext) require modified kernel module."
echo "Full logs saved to: $RESULTS_DIR/"

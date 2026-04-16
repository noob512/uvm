#!/usr/bin/env bash
# =============================================================================
# Experiment 4: FAISS Vector Search (Paper RQ1, Figure 9)
#
# Dataset: SIFT1B (bigann), IVF4096,Flat index
# GPU: RTX 5090 (32GB)
# Sizes: SIFT20M (9.5GB), SIFT50M (24GB), SIFT100M (48GB, oversubscribed)
# Metrics: index build time (s), search time per nprobe (s), recall
# Trials: each search nprobe tested once (build is single run)
#
# Configs without modified KM: 1 (UVM baseline)
# Configs needing modified KM: 2 (UVM + gpu_ext eBPF adaptive prefetch)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKLOADS_DIR="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$SCRIPT_DIR/results/exp4_vector_search/${TIMESTAMP}"
BENCH_SCRIPT="$SCRIPT_DIR/bench_gpu_1bn.py"
DATA_DIR="$SCRIPT_DIR/faiss/benchs/bigann"

mkdir -p "$RESULTS_DIR"

# Log everything
exec > >(tee -a "$RESULTS_DIR/full.log") 2>&1

echo "============================================================"
echo "Experiment 4: FAISS Vector Search (Figure 9)"
echo "============================================================"
echo "Timestamp: $TIMESTAMP"
echo "Results dir: $RESULTS_DIR"
echo ""

# --- Prerequisites ---
if [ ! -f "$BENCH_SCRIPT" ]; then
    echo "ERROR: bench_gpu_1bn.py not found at $BENCH_SCRIPT"
    exit 1
fi

python3 -c "import faiss" 2>/dev/null || {
    echo "ERROR: faiss Python module not installed."
    echo "Build FAISS first: cd faiss && cmake -B build -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=ON ..."
    exit 1
}

if [ ! -f "$DATA_DIR/bigann_base.bvecs" ]; then
    echo "ERROR: SIFT dataset not found at $DATA_DIR/"
    echo "Run: bash $SCRIPT_DIR/download_sift.sh"
    exit 1
fi

# --- GPU Cleanup ---
echo "Cleaning GPU processes..."
python3 "$WORKLOADS_DIR/cleanup_gpu.py" 2>/dev/null || true
sleep 2

NPROBES="1,4,16"

# ==========================================================
# SIFT20M — fits in GPU memory (9.5GB), baseline reference
# ==========================================================
echo ""
echo "============================================================"
echo "SIFT20M: UVM Baseline (9.5GB, fits in 32GB GPU)"
echo "============================================================"
python3 "$WORKLOADS_DIR/cleanup_gpu.py" 2>/dev/null || true
sleep 2

cd "$SCRIPT_DIR"
uv run python "$BENCH_SCRIPT" SIFT20M IVF4096,Flat -nprobe $NPROBES -uvm \
    2>&1 | tee "$RESULTS_DIR/SIFT20M_uvm_baseline.log" || {
    echo "SIFT20M FAILED"
}

# ==========================================================
# SIFT50M — near memory boundary (24GB)
# ==========================================================
echo ""
echo "============================================================"
echo "SIFT50M: UVM Baseline (24GB, near boundary)"
echo "============================================================"
python3 "$WORKLOADS_DIR/cleanup_gpu.py" 2>/dev/null || true
sleep 2

uv run python "$BENCH_SCRIPT" SIFT50M IVF4096,Flat -nprobe $NPROBES -uvm \
    2>&1 | tee "$RESULTS_DIR/SIFT50M_uvm_baseline.log" || {
    echo "SIFT50M FAILED"
}

# ==========================================================
# SIFT100M — oversubscribed (48GB on 32GB GPU)
# ==========================================================
echo ""
echo "============================================================"
echo "SIFT100M: UVM Baseline (48GB, oversubscribed)"
echo "============================================================"
python3 "$WORKLOADS_DIR/cleanup_gpu.py" 2>/dev/null || true
sleep 2

uv run python "$BENCH_SCRIPT" SIFT100M IVF4096,Flat -nprobe $NPROBES -uvm \
    2>&1 | tee "$RESULTS_DIR/SIFT100M_uvm_baseline.log" || {
    echo "SIFT100M FAILED"
}

# --- Summary ---
echo ""
echo "============================================================"
echo "RESULTS SUMMARY (Experiment 4: Vector Search)"
echo "============================================================"
echo ""
for SIZE in SIFT20M SIFT50M SIFT100M; do
    LOG="$RESULTS_DIR/${SIZE}_uvm_baseline.log"
    echo "$SIZE:"
    if [ -f "$LOG" ]; then
        grep -E "add time|probe=" "$LOG" 2>/dev/null || echo "  (no results found in log)"
    else
        echo "  NOT RUN"
    fi
    echo ""
done

echo "Paper reference (SIFT100M, UVM baseline):"
echo "  Add time:       68.41s"
echo "  Search nprobe=1:  5.14s"
echo "  Search nprobe=4:  14.39s"
echo "  Search nprobe=16: 56.51s"
echo ""
echo "NOTE: Config 2 (UVM+gpu_ext) requires modified kernel module."
echo "Full logs saved to: $RESULTS_DIR/"

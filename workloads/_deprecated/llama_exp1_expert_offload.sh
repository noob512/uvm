#!/usr/bin/env bash
# =============================================================================
# Experiment 1: llama.cpp Expert Offloading (Paper RQ1, Figure 6)
#
# Model: GPT-OSS-120B MXFP4 MoE (59 GiB, 116.83B params)
# GPU: RTX 5090 (32GB) → 1.84x oversubscription
# Metrics: pp512 (prefill tok/s), tg128 (decode tok/s)
# Trials: 10 runs per config (paper methodology)
#
# Configs that run without modified KM: 1 (ncmoe=64), 2 (ncmoe=32), 3 (UVM baseline)
# Configs that need modified KM: 4 (UVM+hints), 5 (UVM+gpu_ext eBPF)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKLOADS_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$SCRIPT_DIR/results/exp1_expert_offload"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${RESULTS_DIR}/${TIMESTAMP}"
LLAMA_BENCH="$SCRIPT_DIR/build/bin/llama-bench"
MODEL="${MODEL_120B_CACHE:-$HOME/.cache/llama.cpp/ggml-org_gpt-oss-120b-GGUF_gpt-oss-120b-mxfp4-00001-of-00003.gguf}"
NRUNS="${NRUNS:-10}"

mkdir -p "$RESULTS_DIR"

# Log everything
exec > >(tee -a "$RESULTS_DIR/full.log") 2>&1

echo "============================================================"
echo "Experiment 1: llama.cpp Expert Offloading (Figure 6)"
echo "============================================================"
echo "Timestamp: $TIMESTAMP"
echo "Model: $MODEL"
echo "Runs per config: $NRUNS"
echo "Results dir: $RESULTS_DIR"
echo ""

# --- Prerequisites ---
if [ ! -f "$LLAMA_BENCH" ]; then
    echo "ERROR: llama-bench not found at $LLAMA_BENCH"
    echo "Run: cd $SCRIPT_DIR && make build-cuda-no-vmm"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "ERROR: GPT-OSS-120B model not found at $MODEL"
    echo "Run: huggingface-cli download ggml-org/gpt-oss-120b-GGUF --local-dir ~/.cache/llama.cpp/ggml-org_gpt-oss-120b-GGUF"
    exit 1
fi

# --- GPU Cleanup ---
echo "Cleaning GPU processes..."
python3 "$WORKLOADS_DIR/cleanup_gpu.py" 2>/dev/null || true
sleep 2

# --- Config 1: Framework CPU Offload (ncmoe=64) ---
echo ""
echo "============================================================"
echo "Config 1/3: Framework CPU Offload (ncmoe=64)"
echo "============================================================"
$LLAMA_BENCH \
    -m "$MODEL" \
    -r "$NRUNS" \
    -ncmoe 64 \
    2>&1 | tee "$RESULTS_DIR/config1_ncmoe64.log"

python3 "$WORKLOADS_DIR/cleanup_gpu.py" 2>/dev/null || true
sleep 5

# --- Config 2: Framework CPU Offload (ncmoe=32) ---
echo ""
echo "============================================================"
echo "Config 2/3: Framework CPU Offload (ncmoe=32)"
echo "============================================================"
$LLAMA_BENCH \
    -m "$MODEL" \
    -r "$NRUNS" \
    -ncmoe 32 \
    2>&1 | tee "$RESULTS_DIR/config2_ncmoe32.log"

python3 "$WORKLOADS_DIR/cleanup_gpu.py" 2>/dev/null || true
sleep 5

# --- Config 3: UVM Baseline ---
echo ""
echo "============================================================"
echo "Config 3/3: UVM Baseline (no policy)"
echo "============================================================"
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 $LLAMA_BENCH \
    -m "$MODEL" \
    -r "$NRUNS" \
    2>&1 | tee "$RESULTS_DIR/config3_uvm_baseline.log"

# --- Summary ---
echo ""
echo "============================================================"
echo "RESULTS SUMMARY (Experiment 1: Expert Offloading)"
echo "============================================================"
echo ""
echo "Config 1 (ncmoe=64):"
grep -E "pp|tg" "$RESULTS_DIR/config1_ncmoe64.log" | tail -2 || echo "  (no results)"
echo ""
echo "Config 2 (ncmoe=32):"
grep -E "pp|tg" "$RESULTS_DIR/config2_ncmoe32.log" | tail -2 || echo "  (no results)"
echo ""
echo "Config 3 (UVM baseline):"
grep -E "pp|tg" "$RESULTS_DIR/config3_uvm_baseline.log" | tail -2 || echo "  (no results)"
echo ""
echo "Paper reference values (tok/s):"
echo "  ncmoe=64:     pp512=245.63  tg128=16.34"
echo "  ncmoe=32:     pp512=260.14  tg128=18.18"
echo "  UVM baseline: pp512=238.48  tg128=7.72"
echo ""
echo "NOTE: Configs 4 (UVM+hints) and 5 (UVM+gpu_ext) require modified kernel module."
echo "Full logs saved to: $RESULTS_DIR/"

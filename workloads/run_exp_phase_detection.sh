#!/bin/bash
# Experiment: Phase-Adaptive Prefetch Detection (uprobe-based)
#
# Tests phase-detection BPF policies across all workloads:
# 1. llama.cpp: decode_prefetch_mode 0-3, with/without XB
# 2. FAISS: uprobe path fix + comparison with heuristic
# 3. vLLM: decode modes + xb_decode_enable
#
# All experiments are SERIAL (struct_ops singleton, shared GPU).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
EXT_DIR="$ROOT_DIR/extension"
RESULTS_BASE="$ROOT_DIR/workloads/results_phase_detection/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_BASE"

# ===== Common helpers =====

BPF_PID=""

cleanup_gpu() {
    python3 "$ROOT_DIR/workloads/cleanup_gpu.py" 2>/dev/null || true
    sleep 2
}

stop_bpf() {
    if [ -n "${BPF_PID:-}" ] && kill -0 "$BPF_PID" 2>/dev/null; then
        echo "  Stopping BPF (PID $BPF_PID)..."
        sudo kill "$BPF_PID" 2>/dev/null || true
        wait "$BPF_PID" 2>/dev/null || true
    fi
    BPF_PID=""
    sleep 1
}

start_bpf() {
    local binary="$1"
    shift
    echo "  Loading BPF: $binary $*"
    sudo "$EXT_DIR/$binary" "$@" &
    BPF_PID=$!
    sleep 3
    if ! kill -0 "$BPF_PID" 2>/dev/null; then
        echo "  ERROR: BPF program failed to load"
        return 1
    fi
    return 0
}

# ===== PART 1: llama.cpp Phase Detection =====

echo "============================================"
echo "  PART 1: llama.cpp Phase Detection"
echo "============================================"

MODEL="$HOME/.cache/llama.cpp/ggml-org_gpt-oss-120b-GGUF_gpt-oss-120b-mxfp4-00001-of-00003.gguf"
LLAMA_BENCH="$ROOT_DIR/workloads/llama.cpp/build/bin/llama-bench"
LIBLLAMA="$ROOT_DIR/workloads/llama.cpp/build/bin/libllama.so"
LLAMA_DIR="$RESULTS_BASE/llama"
mkdir -p "$LLAMA_DIR"

PP=512
TG=128
REPS=3

run_llama_bench() {
    local name="$1"
    local logfile="$LLAMA_DIR/${name}.log"
    echo "--- Running llama.cpp: $name ---"
    cleanup_gpu
    GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
    "$LLAMA_BENCH" \
        -m "$MODEL" \
        -p "$PP" -n "$TG" -r "$REPS" \
        -o json 2>&1 | tee "$logfile"
    echo ""
}

# 1a. Baseline: always_max_cycle_moe (no phase detection)
echo "========== llama baseline: always_max_cycle_moe =========="
if start_bpf "prefetch_always_max_cycle_moe"; then
    run_llama_bench "baseline_always_max_cycle_moe"
    stop_bpf
fi

# 1b. Phase mode 0, xb=0 (always_max both phases, no XB — should match baseline)
echo "========== llama phase: mode0 xb=0 =========="
if start_bpf "prefetch_llama_phase" "$LIBLLAMA" 0 32 0; then
    run_llama_bench "phase_mode0_noxb"
    stop_bpf
fi

# 1c. Phase mode 1 (narrow decode, radius=32), xb=0
echo "========== llama phase: mode1 r=32 xb=0 =========="
if start_bpf "prefetch_llama_phase" "$LIBLLAMA" 1 32 0; then
    run_llama_bench "phase_mode1_r32_noxb"
    stop_bpf
fi

# 1d. Phase mode 1 (narrow decode, radius=8), xb=0
echo "========== llama phase: mode1 r=8 xb=0 =========="
if start_bpf "prefetch_llama_phase" "$LIBLLAMA" 1 8 0; then
    run_llama_bench "phase_mode1_r8_noxb"
    stop_bpf
fi

# 1e. Phase mode 2 (default kernel during decode), xb=0
echo "========== llama phase: mode2 xb=0 =========="
if start_bpf "prefetch_llama_phase" "$LIBLLAMA" 2 32 0; then
    run_llama_bench "phase_mode2_noxb"
    stop_bpf
fi

# 1f. Phase mode 3 (forward-only decode), xb=0
echo "========== llama phase: mode3 xb=0 =========="
if start_bpf "prefetch_llama_phase" "$LIBLLAMA" 3 32 0; then
    run_llama_bench "phase_mode3_noxb"
    stop_bpf
fi

# 1g. Phase mode 0 + XB during prefill (reproduce previous result)
echo "========== llama phase: mode0 xb=1 =========="
if start_bpf "prefetch_llama_phase" "$LIBLLAMA" 0 32 1; then
    run_llama_bench "phase_mode0_xb"
    stop_bpf
fi

# ===== llama.cpp Summary =====
echo ""
echo "=== llama.cpp Phase Detection Summary ==="
for f in "$LLAMA_DIR"/*.log; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .log)
    pp_val=$(grep -o '"avg_ts":[0-9.]*' "$f" | head -1 | cut -d: -f2 || echo "N/A")
    tg_val=$(grep -o '"avg_ts":[0-9.]*' "$f" | tail -1 | cut -d: -f2 || echo "N/A")
    printf "  %-40s pp=%-10s tg=%-10s\n" "$name" "$pp_val" "$tg_val"
done

# ===== PART 2: FAISS Uprobe Phase Detection =====

echo ""
echo "============================================"
echo "  PART 2: FAISS Uprobe Phase Detection"
echo "============================================"

FAISS_DIR="$RESULTS_BASE/faiss"
mkdir -p "$FAISS_DIR"

# Find correct FAISS .so path
FAISS_SO="$ROOT_DIR/workloads/faiss/faiss/build/faiss/python/faiss/_swigfaiss.so"
if [ ! -f "$FAISS_SO" ]; then
    echo "WARNING: FAISS .so not found at $FAISS_SO, skipping FAISS tests"
else
    # Find symbols
    ADD_SYM=$(nm -D "$FAISS_SO" 2>/dev/null | grep "T.*GpuIndex12add_with_ids" | awk '{print $3}' | head -1)
    SEARCH_SYM=$(nm -D "$FAISS_SO" 2>/dev/null | grep "T.*GpuIndex6search" | awk '{print $3}' | head -1)

    if [ -z "$ADD_SYM" ] || [ -z "$SEARCH_SYM" ]; then
        echo "WARNING: Could not find FAISS symbols, skipping FAISS tests"
        echo "  ADD_SYM=$ADD_SYM"
        echo "  SEARCH_SYM=$SEARCH_SYM"
    else
        echo "  FAISS .so: $FAISS_SO"
        echo "  add symbol: $ADD_SYM"
        echo "  search symbol: $SEARCH_SYM"

        run_faiss_bench() {
            local name="$1"
            local logfile="$FAISS_DIR/${name}.log"
            echo "--- Running FAISS: $name ---"
            cleanup_gpu
            cd "$ROOT_DIR/workloads/faiss"
            uv run python bench_gpu_1bn.py SIFT100M IVF4096,Flat -nnn 10 -nprobe 1,4,16 -uvm 2>&1 | tee "$logfile"
            cd "$ROOT_DIR/extension"
            echo ""
        }

        # 2a. FAISS baseline: no BPF
        echo "========== FAISS baseline: no BPF =========="
        run_faiss_bench "baseline_no_bpf"

        # 2b. FAISS: always_max_cycle_moe
        echo "========== FAISS: always_max_cycle_moe =========="
        if start_bpf "prefetch_always_max_cycle_moe"; then
            run_faiss_bench "always_max_cycle_moe"
            stop_bpf
        fi

        # 2c. FAISS: uprobe phase detection (correct path)
        echo "========== FAISS: uprobe phase =========="
        if start_bpf "prefetch_faiss_uprobe" "$FAISS_SO" "$ADD_SYM" "$SEARCH_SYM"; then
            run_faiss_bench "faiss_uprobe"
            stop_bpf
        fi

        # 2d. FAISS: heuristic phase (existing)
        echo "========== FAISS: heuristic phase =========="
        if start_bpf "prefetch_faiss_phase"; then
            run_faiss_bench "faiss_heuristic"
            stop_bpf
        fi

        # Summary
        echo ""
        echo "=== FAISS Phase Detection Summary ==="
        for f in "$FAISS_DIR"/*.log; do
            [ -f "$f" ] || continue
            name=$(basename "$f" .log)
            # Extract search times for nprobe=1,4,16
            echo "  $name:"
            grep -o '"search_time": [0-9.]*' "$f" | head -3 || echo "    (parse error)"
        done
    fi
fi

# ===== PART 3: vLLM Phase Detection =====

echo ""
echo "============================================"
echo "  PART 3: vLLM Phase Detection"
echo "============================================"

VLLM_DIR="$RESULTS_BASE/vllm"
mkdir -p "$VLLM_DIR"

ALLOCATOR_SO="$ROOT_DIR/workloads/vllm/vllm/vllm/uvm_allocator.abi3.so"

run_vllm_bench() {
    local name="$1"
    local output="$VLLM_DIR/${name}.json"
    echo "--- Running vLLM: $name ---"
    cleanup_gpu
    cd "$ROOT_DIR/workloads/vllm"
    uv run python configs/serve_bench.py --mode uvm --prompts 100 --no-cleanup -o "$output" 2>&1
    cd "$ROOT_DIR/extension"
    echo ""
}

# 3a. vLLM baseline: no BPF
echo "========== vLLM baseline: no BPF =========="
run_vllm_bench "baseline_no_bpf"

# 3b. vLLM: always_max_cycle_moe
echo "========== vLLM: always_max_cycle_moe =========="
if start_bpf "prefetch_always_max_cycle_moe"; then
    run_vllm_bench "always_max_cycle_moe"
    stop_bpf
fi

# 3c. vLLM: phase mode 0, xb_decode=0 (same as always_max + XB prefill only)
echo "========== vLLM phase: mode0 xb_decode=0 =========="
if start_bpf "prefetch_vllm_phase" "$ALLOCATOR_SO" 0 32 0; then
    run_vllm_bench "phase_mode0_noxb"
    stop_bpf
fi

# 3d. vLLM: phase mode 0, xb_decode=1 (XB both phases)
echo "========== vLLM phase: mode0 xb_decode=1 =========="
if start_bpf "prefetch_vllm_phase" "$ALLOCATOR_SO" 0 32 1; then
    run_vllm_bench "phase_mode0_xb_both"
    stop_bpf
fi

# 3e. vLLM: phase mode 1 (narrow decode), xb_decode=0
echo "========== vLLM phase: mode1 r=32 xb_decode=0 =========="
if start_bpf "prefetch_vllm_phase" "$ALLOCATOR_SO" 1 32 0; then
    run_vllm_bench "phase_mode1_r32"
    stop_bpf
fi

# 3f. vLLM: phase mode 2 (default kernel decode), xb_decode=0
echo "========== vLLM phase: mode2 xb_decode=0 =========="
if start_bpf "prefetch_vllm_phase" "$ALLOCATOR_SO" 2 32 0; then
    run_vllm_bench "phase_mode2"
    stop_bpf
fi

# Summary
echo ""
echo "=== vLLM Phase Detection Summary ==="
for f in "$VLLM_DIR"/*.json; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .json)
    tpot=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('metrics',{}).get('mean_tpot_ms','N/A'))" 2>/dev/null || echo "N/A")
    tput=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('metrics',{}).get('output_throughput','N/A'))" 2>/dev/null || echo "N/A")
    printf "  %-35s TPOT=%-10s tput=%-10s\n" "$name" "$tpot" "$tput"
done

echo ""
echo "============================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "============================================"
echo "Results directory: $RESULTS_BASE"

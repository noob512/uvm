#!/bin/bash
# Uprobe Prefetch Benchmark Experiment
#
# Compares 4 configurations:
#   A: No BPF (baseline UVM on-demand faults)
#   B: BPF always_max only (struct_ops prefetch, no uprobe)
#   C: BPF uprobe + always_max (application-guided + struct_ops)
#   D: BPF uprobe loaded but no hints (control group)
#
# Each config runs the benchmark with oversubscribed sequential chunked access.

set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXT_DIR="$REPO_ROOT/extension"
BENCH_DIR="$REPO_ROOT/microbench/memory"
RESULT_DIR="$REPO_ROOT/microbench/memory/results/uprobe_bench"

TOTAL_MB=${1:-40000}    # 40GB default (1.25x oversubscription on 32GB GPU)
CHUNK_MB=${2:-4}        # 4MB chunks
ITERATIONS=${3:-5}

mkdir -p "$RESULT_DIR"

# Build benchmark
echo "=== Building benchmark ==="
cd "$BENCH_DIR"
nvcc -O2 -o uprobe_bench uprobe_bench.cu -lcudart -Wno-deprecated-gpu-targets
BENCH="$BENCH_DIR/uprobe_bench"
echo "Built: $BENCH"

# Build BPF tools if needed
cd "$EXT_DIR"
if [ ! -f prefetch_always_max ]; then
    echo "Building prefetch_always_max..."
    make prefetch_always_max
fi
if [ ! -f test_uprobe_prefetch ]; then
    echo "Building test_uprobe_prefetch..."
    make test_uprobe_prefetch
fi

# Helper: clean up stale struct_ops
cleanup_bpf() {
    sudo "$EXT_DIR/cleanup_struct_ops_tool" 2>/dev/null || true
    sudo bpftool map list -j 2>/dev/null | python3 -c "
import json,sys,os
try:
    for m in json.load(sys.stdin):
        if m.get('type')=='struct_ops':
            os.system(f'sudo bpftool map delete id {m[\"id\"]} key 0 0 0 0 2>/dev/null')
except: pass
" 2>/dev/null || true
    sleep 1
}

# Helper: kill any running BPF loaders
kill_bpf() {
    sudo pkill -x prefetch_always 2>/dev/null || true
    sudo pkill -x test_uprobe_pr 2>/dev/null || true
    sleep 2
    cleanup_bpf
}

echo ""
echo "=========================================="
echo " Config A: No BPF (baseline)"
echo "=========================================="
kill_bpf
echo "Running benchmark (no BPF, hints are no-ops)..."
"$BENCH" $TOTAL_MB $CHUNK_MB $ITERATIONS 2>&1 | tee "$RESULT_DIR/config_a_no_bpf.log"

echo ""
echo "=========================================="
echo " Config B: always_max only (no uprobe)"
echo "=========================================="
kill_bpf
echo "Starting prefetch_always_max..."
sudo "$EXT_DIR/prefetch_always_max" > /tmp/always_max.log 2>&1 &
BPF_PID=$!
sleep 3

if kill -0 $BPF_PID 2>/dev/null; then
    echo "BPF always_max attached (PID=$BPF_PID)"
    "$BENCH" $TOTAL_MB $CHUNK_MB $ITERATIONS --no-hint 2>&1 | tee "$RESULT_DIR/config_b_always_max.log"
    sudo kill $BPF_PID 2>/dev/null; wait $BPF_PID 2>/dev/null || true
else
    echo "ERROR: prefetch_always_max failed to start"
    cat /tmp/always_max.log
fi

echo ""
echo "=========================================="
echo " Config C: uprobe + always_max (direct)"
echo "=========================================="
kill_bpf
echo "Starting test_uprobe_prefetch --direct..."
sudo "$EXT_DIR/test_uprobe_prefetch" "$BENCH" --direct > /tmp/uprobe_bpf.log 2>&1 &
BPF_PID=$!
sleep 3

if kill -0 $BPF_PID 2>/dev/null; then
    echo "BPF uprobe+always_max attached (PID=$BPF_PID)"
    "$BENCH" $TOTAL_MB $CHUNK_MB $ITERATIONS 2>&1 | tee "$RESULT_DIR/config_c_uprobe.log"
    sudo kill $BPF_PID 2>/dev/null; wait $BPF_PID 2>/dev/null || true
else
    echo "ERROR: test_uprobe_prefetch failed to start"
    grep -v "^libbpf:" /tmp/uprobe_bpf.log
fi

echo ""
echo "=========================================="
echo " Config D: uprobe loaded, no hints (control)"
echo "=========================================="
kill_bpf
echo "Starting test_uprobe_prefetch --direct (control: hints disabled)..."
sudo "$EXT_DIR/test_uprobe_prefetch" "$BENCH" --direct > /tmp/uprobe_bpf2.log 2>&1 &
BPF_PID=$!
sleep 3

if kill -0 $BPF_PID 2>/dev/null; then
    echo "BPF uprobe+always_max attached but hints disabled (PID=$BPF_PID)"
    "$BENCH" $TOTAL_MB $CHUNK_MB $ITERATIONS --no-hint 2>&1 | tee "$RESULT_DIR/config_d_uprobe_nohint.log"
    sudo kill $BPF_PID 2>/dev/null; wait $BPF_PID 2>/dev/null || true
else
    echo "ERROR: test_uprobe_prefetch failed to start"
    grep -v "^libbpf:" /tmp/uprobe_bpf2.log
fi

# Final cleanup
kill_bpf

echo ""
echo "=========================================="
echo " SUMMARY"
echo "=========================================="
echo ""
for f in "$RESULT_DIR"/config_*.log; do
    config=$(basename "$f" .log)
    echo "--- $config ---"
    grep "iter " "$f" | tail -$ITERATIONS
    echo ""
done

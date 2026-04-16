#!/bin/bash
# GNN 10M Novel Algorithms Benchmark
# All output redirected to log files to avoid pipe issues

WORKDIR="/home/yunwei37/workspace/gpu/gpu_ext/workloads/pytorch"
EXTDIR="/home/yunwei37/workspace/gpu/gpu_ext/extension"
CLEANUP="/home/yunwei37/workspace/gpu/gpu_ext/workloads/cleanup_gpu.py"
RESULTDIR="/home/yunwei37/workspace/gpu/gpu_ext/workloads/pytorch/result"
LOGDIR="/tmp/gnn_novel_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$LOGDIR"
cd "$WORKDIR"

SUMMARY="$LOGDIR/summary_${TIMESTAMP}.txt"
echo "=== GNN 10M Novel Algorithms Benchmark ===" > "$SUMMARY"
echo "Started: $(date)" >> "$SUMMARY"
echo "" >> "$SUMMARY"

run_benchmark() {
    local config_name="$1"
    local json_out="$2"
    shift 2

    echo "$(date): Starting $config_name" >> "$SUMMARY"

    # Kill any existing BPF
    sudo pkill -9 -f "prefetch_" 2>/dev/null
    # Also kill any stale llama-bench
    sudo pkill -9 -f "llama-bench" 2>/dev/null
    sleep 2

    # Clean GPU
    sudo python3 "$CLEANUP" > "$LOGDIR/${config_name}_cleanup.log" 2>&1

    # Clean stale struct_ops by reloading if needed
    # (handled by the BPF loader itself)

    BPF_PID=""

    # Start BPF if command provided
    if [ $# -gt 0 ]; then
        echo "  BPF: $@" >> "$SUMMARY"
        sudo "$@" > "$LOGDIR/${config_name}_bpf.log" 2>&1 &
        BPF_PID=$!
        sleep 3

        # Check if BPF loaded successfully
        if ! sudo kill -0 "$BPF_PID" 2>/dev/null; then
            echo "  ERROR: BPF failed to load" >> "$SUMMARY"
            echo "  BPF log tail:" >> "$SUMMARY"
            tail -5 "$LOGDIR/${config_name}_bpf.log" >> "$SUMMARY" 2>/dev/null
            echo "" >> "$SUMMARY"
            return 1
        fi
        echo "  BPF loaded, PID=$BPF_PID" >> "$SUMMARY"
    else
        echo "  No BPF (baseline)" >> "$SUMMARY"
    fi

    # Run benchmark with ALL output to log file
    CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 uv run python benchmark_gnn_uvm.py \
        --dataset random --nodes 10000000 --prop chunked \
        --epochs 3 --use_uvm \
        --report_json "$RESULTDIR/$json_out" \
        > "$LOGDIR/${config_name}_bench.log" 2>&1
    BENCH_EXIT=$?

    if [ $BENCH_EXIT -ne 0 ]; then
        echo "  WARNING: benchmark exited with code $BENCH_EXIT" >> "$SUMMARY"
    fi

    # Extract results from JSON
    if [ -f "$RESULTDIR/$json_out" ]; then
        AVG=$(python3 -c "import json; d=json.load(open('$RESULTDIR/$json_out')); print(f'{d[\"avg_epoch_time_s\"]:.2f}')")
        EPOCHS=$(python3 -c "import json; d=json.load(open('$RESULTDIR/$json_out')); print([round(t,2) for t in d['epoch_times_s']])")
        echo "  RESULT: avg=${AVG}s | epochs=$EPOCHS" >> "$SUMMARY"
    else
        echo "  RESULT: FAILED (no output JSON)" >> "$SUMMARY"
    fi

    # Stop BPF
    if [ -n "$BPF_PID" ]; then
        sudo kill "$BPF_PID" 2>/dev/null
        wait "$BPF_PID" 2>/dev/null
        sleep 2
    fi

    echo "" >> "$SUMMARY"
}

# ============================================================
# Config A1: Baseline (no BPF)
# ============================================================
run_benchmark "A1_baseline" "novel_a1_baseline_${TIMESTAMP}.json"

# ============================================================
# Config A2: always_max+cycle_moe (reference)
# ============================================================
run_benchmark "A2_always_max_cycle_moe" "novel_a2_always_max_${TIMESTAMP}.json" \
    "$EXTDIR/prefetch_always_max_cycle_moe"

# ============================================================
# Config A3: XB direction-aware (previous best, 3.29x)
# ============================================================
run_benchmark "A3_xb_direction" "novel_a3_xb_direction_${TIMESTAMP}.json" \
    "$EXTDIR/prefetch_cross_block_v2" 1 2048 0

# ============================================================
# Config N1-K2: stride multiblock K=2
# ============================================================
run_benchmark "N1_K2_stride_multiblock" "novel_n1k2_stride_${TIMESTAMP}.json" \
    "$EXTDIR/prefetch_stride_multiblock" 2

# ============================================================
# Config N1-K3: stride multiblock K=3
# ============================================================
run_benchmark "N1_K3_stride_multiblock" "novel_n1k3_stride_${TIMESTAMP}.json" \
    "$EXTDIR/prefetch_stride_multiblock" 3

# ============================================================
# Config N3: cooperative prefetch-eviction (protect_radius=2)
# ============================================================
run_benchmark "N3_cooperative_r2" "novel_n3_cooperative_r2_${TIMESTAMP}.json" \
    "$EXTDIR/prefetch_cooperative" 2

# ============================================================
# Config N3b: cooperative prefetch-eviction (protect_radius=4)
# ============================================================
run_benchmark "N3b_cooperative_r4" "novel_n3b_cooperative_r4_${TIMESTAMP}.json" \
    "$EXTDIR/prefetch_cooperative" 4

# ============================================================
# Config N2: reuse distance eviction (50ms threshold, with XB)
# ============================================================
run_benchmark "N2_reuse_dist_50ms" "novel_n2_reuse_dist_50_${TIMESTAMP}.json" \
    "$EXTDIR/prefetch_reuse_dist" 50 1

# ============================================================
# Config N2b: reuse distance eviction (20ms threshold, with XB)
# ============================================================
run_benchmark "N2b_reuse_dist_20ms" "novel_n2b_reuse_dist_20_${TIMESTAMP}.json" \
    "$EXTDIR/prefetch_reuse_dist" 20 1

# Final cleanup
sudo pkill -9 -f "prefetch_" 2>/dev/null
sudo pkill -9 -f "llama-bench" 2>/dev/null

echo "========================================" >> "$SUMMARY"
echo "=== FINAL SUMMARY TABLE ===" >> "$SUMMARY"
echo "Completed: $(date)" >> "$SUMMARY"
echo "" >> "$SUMMARY"

# Print summary table
python3 << 'PYEOF' >> "$SUMMARY" 2>&1
import json, glob, os, sys

timestamp = os.environ.get('TIMESTAMP', '')
if not timestamp:
    # Find from files
    import re
    files = sorted(glob.glob('/home/yunwei37/workspace/gpu/gpu_ext/workloads/pytorch/result/novel_a1_baseline_*.json'))
    if files:
        m = re.search(r'_(\d{8}_\d{6})\.json$', files[-1])
        if m:
            timestamp = m.group(1)

result_dir = '/home/yunwei37/workspace/gpu/gpu_ext/workloads/pytorch/result'
files = sorted(glob.glob(f'{result_dir}/novel_*_{timestamp}.json'))

if not files:
    print('No result files found!')
    sys.exit(0)

baseline_avg = None
results = []
for f in files:
    try:
        d = json.load(open(f))
        config = os.path.basename(f).replace('novel_', '').replace(f'_{timestamp}.json', '')
        avg = d['avg_epoch_time_s']
        epochs = d['epoch_times_s']
        if 'a1_baseline' in os.path.basename(f):
            baseline_avg = avg
        results.append((config, avg, epochs))
    except Exception as e:
        config = os.path.basename(f)
        results.append((config, -1, []))

print(f"{'Config':<35} | {'Avg (s)':>8} | {'Speedup':>8} | Epoch Times")
print('-' * 100)
for config, avg, epochs in results:
    epoch_str = ', '.join([f'{t:.2f}' for t in epochs])
    if avg > 0 and baseline_avg and baseline_avg > 0:
        speedup = baseline_avg / avg
        print(f"{config:<35} | {avg:>8.2f} | {speedup:>7.2f}x | [{epoch_str}]")
    elif avg > 0:
        print(f"{config:<35} | {avg:>8.2f} | {'N/A':>8} | [{epoch_str}]")
    else:
        print(f"{config:<35} | {'FAILED':>8} | {'N/A':>8} | []")
PYEOF

echo "" >> "$SUMMARY"
echo "Timestamp: $TIMESTAMP" >> "$SUMMARY"
echo "Summary: $SUMMARY" >> "$SUMMARY"
echo "Logs: $LOGDIR/" >> "$SUMMARY"

# Print the summary to stdout as final output
cat "$SUMMARY"
echo "BENCHMARK_COMPLETE"

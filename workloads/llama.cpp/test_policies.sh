#!/bin/bash
# Serial policy benchmark for llama.cpp MoE expert offloading
# Tests each eBPF policy one-by-one, measures pp and tg throughput
#
# NOTE: Run as normal user (NOT sudo). Script uses sudo only for policy loading.
set -uo pipefail
# NOTE: not using -e so we continue past policy load failures (e.g. BPF verifier errors)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXT_DIR="$(cd "$SCRIPT_DIR/../../extension" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results/policy_sweep"
UV="/home/yunwei37/.local/bin/uv"
mkdir -p "$RESULTS_DIR"

BENCH_CMD="$UV run python configs/bench.py --uvm --ncmoe 64"

# Helper: kill any running policy loader and clean up
kill_policy() {
    sudo pkill -f 'eviction_|prefetch_|gpu_preempt|gpu_sched_set' 2>/dev/null || true
    sleep 2
    # Use cleanup tool if needed
    if sudo bpftool prog list 2>/dev/null | grep -q struct_ops; then
        sudo "$EXT_DIR/cleanup_struct_ops_tool" 2>/dev/null || true
        sleep 1
    fi
}

# Helper: run one policy test
run_test() {
    local policy_name="$1"
    local policy_binary="$2"  # empty string means no policy (baseline)
    local output_file="$RESULTS_DIR/${policy_name}.json"

    echo "============================================"
    echo "Testing: $policy_name"
    echo "============================================"

    # Cleanup any previous policy
    kill_policy

    # Start policy if specified
    if [ -n "$policy_binary" ]; then
        echo "  Loading policy: $policy_binary"
        sudo "$policy_binary" > /dev/null 2>&1 &
        sleep 3  # Wait for BPF program to attach

        # Verify it's attached
        if ! sudo bpftool prog list 2>/dev/null | grep -q struct_ops; then
            echo "  ERROR: Policy failed to attach!"
            kill_policy
            return 1
        fi
        echo "  Policy attached successfully"
    else
        echo "  No policy (baseline)"
    fi

    # Run benchmark (as normal user, NOT sudo)
    echo "  Running benchmark..."
    cd "$SCRIPT_DIR"
    if $BENCH_CMD -o "$output_file" 2>&1; then
        echo "  Benchmark completed: $output_file"
        # Extract key metrics
        pp=$(python3 -c "import json; d=json.load(open('$output_file')); print(f\"{d['metrics']['pp_tok_s']:.2f}\")")
        tg=$(python3 -c "import json; d=json.load(open('$output_file')); print(f\"{d['metrics']['tg_tok_s']:.2f}\")")
        echo "  >> pp=$pp tok/s, tg=$tg tok/s"
    else
        echo "  ERROR: Benchmark failed!"
    fi

    # Cleanup policy
    kill_policy
    echo ""
}

# ============================================
# Run tests serially
# ============================================

echo "Starting serial policy benchmark sweep"
echo "Results directory: $RESULTS_DIR"
echo ""

# 1. Baseline (no policy)
run_test "baseline_no_policy" ""

# 2. Eviction policies (only eviction, driver default prefetch)
run_test "eviction_lfu" "$EXT_DIR/eviction_lfu"
run_test "eviction_fifo" "$EXT_DIR/eviction_fifo"
run_test "eviction_mru" "$EXT_DIR/eviction_mru"
run_test "eviction_fifo_chance" "$EXT_DIR/eviction_fifo_chance"
run_test "eviction_freq_pid_decay" "$EXT_DIR/eviction_freq_pid_decay"

# 3. Prefetch policies (only prefetch, driver default eviction)
run_test "prefetch_none" "$EXT_DIR/prefetch_none"
run_test "prefetch_always_max" "$EXT_DIR/prefetch_always_max"
run_test "prefetch_adaptive_sequential" "$EXT_DIR/prefetch_adaptive_sequential"

echo "============================================"
echo "All tests completed!"
echo "============================================"

# Print summary table
echo ""
echo "Summary:"
echo "-------------------------------------------"
printf "%-30s %12s %12s\n" "Policy" "pp (tok/s)" "tg (tok/s)"
echo "-------------------------------------------"
for f in "$RESULTS_DIR"/*.json; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .json)
    pp=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d['metrics']['pp_tok_s']:.2f}\")" 2>/dev/null || echo "N/A")
    tg=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d['metrics']['tg_tok_s']:.2f}\")" 2>/dev/null || echo "N/A")
    printf "%-30s %12s %12s\n" "$name" "$pp" "$tg"
done
echo "-------------------------------------------"

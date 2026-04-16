#!/bin/bash
# GPU Priority Scheduling Demo: BPF kfunc preempt
#
# Tests THREE scenarios to show when BPF preempt helps:
#
#   Scenario 1: BE partially occupies GPU (<<<512,256>>>)
#     → LC can run alongside BE on free SMs → preempt NOT needed
#
#   Scenario 2: BE fully saturates GPU (persistent kernel on ALL SMs)
#     → LC MUST wait for hardware timeslice → preempt HELPS
#
#   Scenario 3: Same as 2, but with BPF preempt
#     → BE preempted on each LC launch → LC gets GPU faster
#
# The key insight: BPF preempt helps only when GPU is fully saturated
# AND hardware timeslice is the bottleneck.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(dirname "$0")/../../../extension"

# Build CUDA test binaries if needed
if [ ! -x "$SCRIPT_DIR/test_priority_demo" ]; then
    echo "Building test_priority_demo..."
    nvcc -o "$SCRIPT_DIR/test_priority_demo" "$SCRIPT_DIR/test_priority_demo.cu" -lcuda
fi
PRIORITY_DEMO="$SCRIPT_DIR/test_priority_demo"

N_TRIALS=${1:-20}
echo "============================================================"
echo "  GPU Priority Scheduling Demo (uprobe kfunc preempt)"
echo "============================================================"
echo "  N_TRIALS=$N_TRIALS per scenario"
echo ""

cleanup() {
    # Kill all background processes
    kill $BE_PID 2>/dev/null || true
    sudo pkill -f test_uprobe_preempt 2>/dev/null || true
    sudo pkill -f test_preempt_kfunc 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

capture_be_handles() {
    # Start kfunc tool to capture handles via kprobes
    { sleep 120; } | sudo ./test_preempt_kfunc > /dev/null 2>&1 &
    KF_PID=$!
    sleep 3

    # Start BE (it will be captured by kprobes)
    "$@" > /tmp/be_output.txt 2>&1 &
    BE_PID=$!
    sleep 5

    if ! kill -0 $BE_PID 2>/dev/null; then
        echo "ERROR: BE failed to start"
        cat /tmp/be_output.txt
        exit 1
    fi
    echo "  BE PID=$BE_PID"

    # Find tsg_map and extract handles
    TSG_MAP=$(sudo bpftool map list -j 2>/dev/null | python3 -c "
import sys, json
for m in json.loads(sys.stdin.read()):
    if m.get('name') == 'tsg_map': print(m['id']); break
" 2>/dev/null)

    HANDLES=$(sudo bpftool map dump id $TSG_MAP -j 2>/dev/null | python3 -c "
import sys, json
data = json.loads(sys.stdin.read())
for entry in data:
    v = entry.get('value', [])
    vals = [int(x, 16) if isinstance(x, str) else x for x in v]
    if len(vals) < 16: continue
    hC = vals[0] | (vals[1]<<8) | (vals[2]<<16) | (vals[3]<<24)
    hT = vals[4] | (vals[5]<<8) | (vals[6]<<16) | (vals[7]<<24)
    eng = vals[8] | (vals[9]<<8) | (vals[10]<<16) | (vals[11]<<24)
    pid = vals[12] | (vals[13]<<8) | (vals[14]<<16) | (vals[15]<<24)
    if eng == 1 and pid == $BE_PID:
        print(f'0x{hC:x} 0x{hT:x}'); break
" 2>/dev/null)

    HC=$(echo $HANDLES | awk '{print $1}')
    HT=$(echo $HANDLES | awk '{print $2}')
    echo "  BE TSG: hClient=$HC hTsg=$HT"

    # Kill kfunc tool (kprobes no longer needed)
    sudo kill $KF_PID 2>/dev/null || true
    wait $KF_PID 2>/dev/null || true
}

# ==============================================================
# Scenario 1: BE partial saturation (loop mode)
# ==============================================================
echo "============================================================"
echo "  Scenario 1: BE partial GPU (<<<512,256>>> loop)"
echo "  Expected: LC runs alongside BE → preempt NOT needed"
echo "============================================================"
echo ""

echo "[1a] Starting BE (loop mode)..."
capture_be_handles $PRIORITY_DEMO be_loop

echo "[1b] LC baseline (no preempt)..."
$PRIORITY_DEMO lc $N_TRIALS 2>&1 | tee /tmp/s1_baseline.txt

# Kill BE
kill $BE_PID 2>/dev/null || true
wait $BE_PID 2>/dev/null || true
sleep 2

echo ""

# ==============================================================
# Scenario 2: BE full saturation, NO preempt
# ==============================================================
echo "============================================================"
echo "  Scenario 2: BE FULL GPU (persistent kernel, ALL SMs)"
echo "  Expected: LC waits for hardware timeslice → HIGH latency"
echo "============================================================"
echo ""

echo "[2a] Starting BE (persistent mode — fills ALL SMs)..."
capture_be_handles $PRIORITY_DEMO be

echo "[2b] LC baseline (no preempt, BE on all SMs)..."
$PRIORITY_DEMO lc $N_TRIALS 2>&1 | tee /tmp/s2_baseline.txt

echo ""

# ==============================================================
# Scenario 3: BE full saturation, WITH preempt
# ==============================================================
echo "============================================================"
echo "  Scenario 3: Same as 2, WITH BPF preempt"
echo "  Expected: LC gets GPU faster (preempt cost < timeslice wait)"
echo "============================================================"
echo ""

echo "[3a] Attaching uprobe to ALL cuLaunchKernel → preempt BE..."
sudo sh -c 'echo > /sys/kernel/tracing/trace'
sudo ./test_uprobe_preempt $HC $HT > /tmp/uprobe_output.txt 2>&1 &
UPROBE_PID=$!
sleep 2

LINK=$(sudo bpftool link list 2>/dev/null | grep -c "uprobe" || true)
if [ "$LINK" -eq 0 ]; then
    echo "ERROR: uprobe not attached"
    cat /tmp/uprobe_output.txt
    exit 1
fi
echo "  uprobe active: cuLaunchKernel → preempt $HC:$HT"
echo ""

echo "[3b] LC with preempt (BE on all SMs)..."
$PRIORITY_DEMO lc $N_TRIALS 2>&1 | tee /tmp/s3_preempt.txt

# Stop uprobe
sudo kill $UPROBE_PID 2>/dev/null || true
wait $UPROBE_PID 2>/dev/null || true

# Kill BE
kill $BE_PID 2>/dev/null || true
wait $BE_PID 2>/dev/null || true

# ==============================================================
# Summary
# ==============================================================
echo ""
echo "============================================================"
echo "  SUMMARY"
echo "============================================================"
echo ""

S1=$(grep "^  avg=" /tmp/s1_baseline.txt || true)
S2=$(grep "^  avg=" /tmp/s2_baseline.txt || true)
S3=$(grep "^  avg=" /tmp/s3_preempt.txt || true)

echo "  S1 (BE partial, no preempt): $S1"
echo "  S2 (BE full,    no preempt): $S2"
echo "  S3 (BE full,  WITH preempt): $S3"

# Parse values
S1_AVG=$(echo "$S1" | grep -o 'avg=[0-9]*' | grep -o '[0-9]*')
S2_AVG=$(echo "$S2" | grep -o 'avg=[0-9]*' | grep -o '[0-9]*')
S3_AVG=$(echo "$S3" | grep -o 'avg=[0-9]*' | grep -o '[0-9]*')
S1_MED=$(echo "$S1" | grep -o 'median=[0-9]*' | grep -o '[0-9]*')
S2_MED=$(echo "$S2" | grep -o 'median=[0-9]*' | grep -o '[0-9]*')
S3_MED=$(echo "$S3" | grep -o 'median=[0-9]*' | grep -o '[0-9]*')

echo ""
echo "  | Scenario              | Avg (us)   | Median (us) |"
echo "  |-----------------------|------------|-------------|"
echo "  | S1: BE partial, no P  | $S1_AVG     | $S1_MED      |"
echo "  | S2: BE full, no P     | $S2_AVG     | $S2_MED      |"
echo "  | S3: BE full, WITH P   | $S3_AVG     | $S3_MED      |"

if [ -n "$S2_AVG" ] && [ -n "$S3_AVG" ] && [ "$S3_AVG" -gt 0 ]; then
    echo ""
    echo "  S2→S3 improvement: $(python3 -c "
s2, s3 = $S2_AVG, $S3_AVG
if s3 < s2:
    print(f'{s2/s3:.1f}x speedup ({s2} → {s3} us, -{(s2-s3)/s2*100:.0f}%)')
else:
    print(f'No improvement ({s2} → {s3} us, +{(s3-s2)/s2*100:.0f}%)')
")"
fi

echo ""
echo "  Uprobe preempt stats:"
grep -E "hits=|preempt_|avg_" /tmp/uprobe_output.txt 2>/dev/null | tail -5

echo ""
echo "  Analysis:"
echo "  - S1 fast: GPU has spare SMs, LC runs alongside BE"
echo "  - S2 slow: ALL SMs occupied, LC waits for HW timeslice"
echo "  - S3: BPF preempt should reduce S2's timeslice wait"
echo "    (improvement visible when timeslice_wait > preempt_cost ~300us)"
echo ""
echo "============================================================"

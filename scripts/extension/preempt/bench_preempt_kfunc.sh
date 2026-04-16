#!/bin/bash
# Benchmark: kfunc preempt latency
#
# Strategy:
# 1. Start CUDA-A (target to preempt)
# 2. Start BPF tool with trigger armed + repeat=N
# 3. Create/destroy CUDA contexts in a loop → fires struct_ops → triggers kfunc
# 4. Read latency_samples map for precise per-call kfunc timing
set -o pipefail
cd "$(dirname "$0")/../../../extension"

N_PREEMPTS=${1:-10}
echo "=== kfunc Preempt Latency Benchmark (N=$N_PREEMPTS) ==="

cleanup() {
    sudo pkill -f test_preempt_kfunc 2>/dev/null || true
    kill $CUDA_A 2>/dev/null || true
    kill $TRIGGER_PID 2>/dev/null || true
}
trap cleanup EXIT

# Step 1: Start target CUDA workload
echo "[1] Starting target CUDA workload..."
python3 -c "
import ctypes, os, time
libcuda = ctypes.CDLL('libcuda.so')
libcuda.cuInit(0)
dev = ctypes.c_int()
libcuda.cuDeviceGet(ctypes.byref(dev), 0)
ctx = ctypes.c_void_p()
libcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
print(f'  Target PID={os.getpid()}', flush=True)
time.sleep(120)
" &
CUDA_A=$!
sleep 3

# Step 2: Start BPF tool — capture A's handles, arm trigger with repeat
echo "[2] Loading BPF, arming trigger (repeat=$N_PREEMPTS)..."
{
    sleep 4  # wait for handle capture
    echo "l"
    sleep 1
    echo "p 0"  # arm trigger for idx=0
    sleep $((N_PREEMPTS * 2 + 20))  # wait for all preempts
    echo "s"
    sleep 1
    echo "q"
} | sudo timeout $((N_PREEMPTS * 2 + 40)) ./test_preempt_kfunc 2>&1 > /tmp/kfunc_bench.txt &
TOOL_PID=$!
sleep 6

# Set repeat count via bpftool
echo "[3] Setting repeat count=$N_PREEMPTS via bpftool..."
# Find the trigger map
TRIGGER_MAP=$(sudo bpftool map list 2>/dev/null | grep -B1 '"trigger"' | head -1 | awk '{print $1}' | tr -d ':')
if [ -z "$TRIGGER_MAP" ]; then
    echo "ERROR: Cannot find trigger map"
    cat /tmp/kfunc_bench.txt
    exit 1
fi
echo "  trigger map id=$TRIGGER_MAP"

# Write repeat count (key=1, value=N_PREEMPTS)
sudo bpftool map update id $TRIGGER_MAP key 01 00 00 00 value $(printf '%02x %02x %02x %02x' $((N_PREEMPTS & 0xFF)) $(((N_PREEMPTS >> 8) & 0xFF)) $(((N_PREEMPTS >> 16) & 0xFF)) $(((N_PREEMPTS >> 24) & 0xFF)))
echo "  repeat=$N_PREEMPTS set"

# Step 3: Fire struct_ops hooks by creating CUDA contexts
echo "[4] Creating $N_PREEMPTS CUDA contexts (triggers struct_ops → kfunc)..."
python3 -c "
import ctypes, sys, time
N = int(sys.argv[1])
libcuda = ctypes.CDLL('libcuda.so')
libcuda.cuInit(0)
dev = ctypes.c_int()
libcuda.cuDeviceGet(ctypes.byref(dev), 0)
for i in range(N):
    ctx = ctypes.c_void_p()
    libcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
    time.sleep(0.5)  # let bpf_wq complete before next
    libcuda.cuCtxDestroy_v2(ctx)
    time.sleep(0.5)
    if (i+1) % 10 == 0:
        print(f'  {i+1}/{N} contexts created', flush=True)
print(f'  All {N} contexts done', flush=True)
" $N_PREEMPTS &
TRIGGER_PID=$!

# Wait for completion
wait $TRIGGER_PID 2>/dev/null
sleep 2

# Step 4: Read results
echo ""
echo "[5] Reading latency samples..."

# Read latency_samples map
LATENCY_MAP=$(sudo bpftool map list 2>/dev/null | grep -B1 '"latency_samples"' | head -1 | awk '{print $1}' | tr -d ':')
if [ -z "$LATENCY_MAP" ]; then
    echo "WARN: Cannot find latency_samples map"
else
    echo "  latency_samples map id=$LATENCY_MAP"
    echo ""
    echo "  kfunc latency samples (ns):"
    for i in $(seq 0 $((N_PREEMPTS - 1))); do
        KEY=$(printf '%02x %02x %02x %02x' $((i & 0xFF)) $(((i >> 8) & 0xFF)) $(((i >> 16) & 0xFF)) $(((i >> 24) & 0xFF)))
        VAL=$(sudo bpftool map lookup id $LATENCY_MAP key $KEY 2>/dev/null | grep -o '"value":.*' | grep -o '[0-9a-f][0-9a-f]' | head -8 | tac | tr -d '\n' | sed 's/^/0x/')
        if [ -n "$VAL" ] && [ "$VAL" != "0x" ]; then
            NS=$((VAL))
            if [ $NS -gt 0 ]; then
                US=$((NS / 1000))
                echo "    [$i] ${NS} ns = ${US} us"
            fi
        fi
    done
fi

echo ""
echo "=== Tool output ==="
wait $TOOL_PID 2>/dev/null || true
cat /tmp/kfunc_bench.txt

echo ""
echo "=== Trace (last entries) ==="
sudo timeout 2 cat /sys/kernel/tracing/trace_pipe 2>/dev/null || true
echo ""
echo "=== Benchmark Complete ==="

#!/bin/bash
# Non-interactive end-to-end test for bpf_nv_gpu_preempt_tsg kfunc
#
# Uses the tool with sleep-delimited commands piped via subshell.
# CUDA workloads are started at the right timing.
set -o pipefail
cd "$(dirname "$0")/../../../extension"

cleanup() {
    sudo pkill -f test_preempt_kfunc 2>/dev/null || true
    kill $CUDA_A $CUDA_B 2>/dev/null || true
}
trap cleanup EXIT

echo "=== GPU Preempt kfunc E2E Test ==="

# Step 1: Start tool in background, feed it commands
echo "[1] Loading BPF..."
{
    # Wait for tool to initialize and capture CUDA-A's TSG
    sleep 6
    echo "l"
    sleep 1
    echo "s"
    sleep 1
    # Arm preempt trigger
    echo "p 0"
    # Wait for CUDA-B to trigger struct_ops → bpf_wq → kfunc
    sleep 20
    echo "s"
    sleep 1
    echo "q"
} | sudo timeout 50 ./test_preempt_kfunc 2>&1 > /tmp/preempt_result.txt &
TOOL_PID=$!
sleep 2

# Verify tool is running
if ! kill -0 $TOOL_PID 2>/dev/null; then
    echo "ERROR: Tool failed to start"
    cat /tmp/preempt_result.txt
    exit 1
fi
echo "  Tool running (PID $TOOL_PID)"

# Step 2: Start CUDA-A (will be captured by kprobes)
echo "[2] Starting CUDA workload A..."
python3 -c "
import ctypes, os, time
libcuda = ctypes.CDLL('libcuda.so')
libcuda.cuInit(0)
dev = ctypes.c_int()
libcuda.cuDeviceGet(ctypes.byref(dev), 0)
ctx = ctypes.c_void_p()
libcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
print(f'  CUDA-A PID={os.getpid()}', flush=True)
time.sleep(60)
" &
CUDA_A=$!
echo "  CUDA-A started (PID $CUDA_A)"

# Wait for TSG capture + trigger arming (tool's sleep 6 + l + s + p)
sleep 12

# Step 3: Start CUDA-B (triggers struct_ops → bpf_wq → kfunc)
echo "[3] Starting CUDA workload B (triggers preempt)..."
python3 -c "
import ctypes, os, time
libcuda = ctypes.CDLL('libcuda.so')
libcuda.cuInit(0)
dev = ctypes.c_int()
libcuda.cuDeviceGet(ctypes.byref(dev), 0)
ctx = ctypes.c_void_p()
libcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
print(f'  CUDA-B PID={os.getpid()}', flush=True)
time.sleep(10)
" &
CUDA_B=$!
echo "  CUDA-B started (PID $CUDA_B)"

# Wait for tool to complete
echo "[4] Waiting for tool to finish..."
wait $TOOL_PID 2>/dev/null || true

echo ""
echo "=== Results ==="
cat /tmp/preempt_result.txt

echo ""
echo "=== Trace Pipe (recent entries) ==="
sudo timeout 2 cat /sys/kernel/tracing/trace_pipe 2>/dev/null || true

echo ""
echo "=== Test Complete ==="

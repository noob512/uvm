#!/bin/bash
# End-to-end test for bpf_nv_gpu_preempt_tsg kfunc
#
# Flow:
# 1. Load BPF (kprobes + struct_ops)
# 2. Start CUDA workload A → captures hClient/hTsg
# 3. Arm trigger via bpftool map update
# 4. Start CUDA workload B → struct_ops fires → bpf_wq → kfunc → preempt A
# 5. Check stats

set -e
cd "$(dirname "$0")/../../../extension"

echo "=== GPU Preempt kfunc E2E Test ==="

# Clean up
echo "[1] Cleanup..."
sudo pkill -f test_preempt_kfunc 2>/dev/null || true
pkill -f "cuCtxCreate.*sleep" 2>/dev/null || true
sleep 1

# Start the tool with commands piped via FIFO
echo "[2] Loading BPF program..."
rm -f /tmp/preempt_fifo
mkfifo /tmp/preempt_fifo

# Open write end to keep FIFO alive, then start tool
(
    exec 7>/tmp/preempt_fifo
    sudo ./test_preempt_kfunc < /tmp/preempt_fifo > /tmp/preempt_out.txt 2>&1 &
    TOOL_PID=$!
    echo "Tool PID: $TOOL_PID"

    sleep 2

    echo "[3] Starting CUDA workload A..."
    python3 -c "
import ctypes, os, time
libcuda = ctypes.CDLL('libcuda.so')
libcuda.cuInit(0)
dev = ctypes.c_int()
libcuda.cuDeviceGet(ctypes.byref(dev), 0)
ctx = ctypes.c_void_p()
libcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
print(f'  CUDA-A created, PID={os.getpid()}', flush=True)
time.sleep(120)
" &
    CUDA_A=$!
    sleep 3

    # List captured TSGs
    echo "l" >&7
    sleep 1
    echo "s" >&7
    sleep 1

    echo "[4] Tool output so far:"
    cat /tmp/preempt_out.txt
    echo "---"

    # Arm trigger for TSG index 0
    echo "[5] Arming preempt trigger..."
    echo "p 0" >&7
    sleep 1

    echo "[6] Starting CUDA workload B (trigger fires on TSG init)..."
    python3 -c "
import ctypes, os, time
libcuda = ctypes.CDLL('libcuda.so')
libcuda.cuInit(0)
dev = ctypes.c_int()
libcuda.cuDeviceGet(ctypes.byref(dev), 0)
ctx = ctypes.c_void_p()
libcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
print(f'  CUDA-B created, PID={os.getpid()}', flush=True)
time.sleep(10)
" &
    CUDA_B=$!

    # Wait for preempt to complete
    sleep 5

    echo "s" >&7
    sleep 1

    echo "[7] Final output:"
    cat /tmp/preempt_out.txt
    echo "---"

    echo "[8] Trace pipe (last entries):"
    sudo timeout 2 cat /sys/kernel/tracing/trace_pipe 2>/dev/null || true
    echo ""

    # Cleanup
    echo "q" >&7
    sleep 1
    kill $CUDA_A $CUDA_B 2>/dev/null || true
    wait $TOOL_PID 2>/dev/null || true
    exec 7>&-
)

rm -f /tmp/preempt_fifo /tmp/preempt_out.txt
echo "=== Test Complete ==="

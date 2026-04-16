#!/bin/bash
# FAISS B3: stress + sched_gpu_baseline boost
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RESULT_DIR="$REPO_ROOT/workloads/faiss/xcoord_results"
mkdir -p "$RESULT_DIR"

echo "=== FAISS B3: Stress + xCoord Boost ===" >&2

# Start stress-ng
NPROC=$(nproc)
stress-ng -c "$NPROC" --timeout 600 &
STRESS_PID=$!
echo "stress-ng PID: $STRESS_PID ($NPROC cores)" >&2
sleep 2

# Start FAISS in background
cd "$REPO_ROOT"
CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 uv run --directory workloads/faiss python configs/search.py \
  --dataset SIFT100M --uvm --nprobe "1,4,16" --no-cleanup \
  -o "$RESULT_DIR/B3_stress_boost.json" > "$RESULT_DIR/B3_faiss.log" 2>&1 &
FAISS_PID=$!
echo "FAISS background PID: $FAISS_PID" >&2

# Wait for GPU process to appear
GPU_PID=""
for i in $(seq 1 30); do
  GPU_PID=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  if [ -n "$GPU_PID" ]; then
    echo "GPU PID found: $GPU_PID" >&2
    break
  fi
  echo "Waiting for GPU ($i)..." >&2
  sleep 2
done

if [ -z "$GPU_PID" ]; then
  echo "ERROR: No GPU PID found, using FAISS_PID" >&2
  GPU_PID=$FAISS_PID
fi

# Start sched_gpu_baseline
sudo "$REPO_ROOT/extension/sched_gpu_baseline" -p "$GPU_PID" > "$RESULT_DIR/B3_sched.log" 2>&1 &
SCHED_PID=$!
sleep 2
echo "sched_ext state: $(cat /sys/kernel/sched_ext/state 2>/dev/null)" >&2
echo "sched_gpu_baseline PID: $SCHED_PID" >&2

# Wait for FAISS
wait $FAISS_PID
FAISS_RC=$?
echo "FAISS finished (rc=$FAISS_RC)" >&2

# Cleanup
sudo kill "$SCHED_PID" 2>/dev/null || true
kill "$STRESS_PID" 2>/dev/null || true
killall stress-ng 2>/dev/null || true
sleep 1

# Show sched stats
echo "=== sched stats ===" >&2
tail -3 "$RESULT_DIR/B3_sched.log" >&2

echo "=== Done ===" >&2

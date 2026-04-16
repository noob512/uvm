#!/bin/bash
# Run FAISS with a specified sched_ext scheduler under stress
# Usage: ./run_faiss_with_sched.sh <scheduler_binary> <result_name>
set -euo pipefail

SCHED_BIN="${1:?Usage: $0 <scheduler_binary> <result_name>}"
RESULT_NAME="${2:?Usage: $0 <scheduler_binary> <result_name>}"
WITH_STRESS="${3:-yes}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RESULT_DIR="$REPO_ROOT/workloads/faiss/xcoord_results"
mkdir -p "$RESULT_DIR"

cd "$REPO_ROOT"
python3 workloads/cleanup_gpu.py

cleanup() {
    sudo pkill -f "$(basename "$SCHED_BIN")" 2>/dev/null || true
    killall stress-ng 2>/dev/null || true
}
trap cleanup EXIT

if [ "$WITH_STRESS" = "yes" ]; then
    NPROC=$(nproc)
    stress-ng -c "$NPROC" --timeout 600 &
    echo "stress-ng started ($NPROC cores)" >&2
    sleep 2
fi

# Start FAISS in background
uv run --directory workloads/faiss python configs/search.py \
  --dataset SIFT100M --uvm --nprobe "1,4,16" --no-cleanup \
  -o "$RESULT_DIR/${RESULT_NAME}.json" > "$RESULT_DIR/${RESULT_NAME}_faiss.log" 2>&1 &
FAISS_PID=$!
echo "FAISS PID: $FAISS_PID" >&2

# Wait for GPU process
GPU_PID=""
for i in $(seq 1 30); do
    GPU_PID=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    [ -n "$GPU_PID" ] && break
    sleep 2
done
echo "GPU PID: ${GPU_PID:-NOT_FOUND}" >&2

# Start scheduler
sudo "$SCHED_BIN" -p "${GPU_PID:-$FAISS_PID}" > "$RESULT_DIR/${RESULT_NAME}_sched.log" 2>&1 &
SCHED_PID=$!
sleep 2
echo "sched_ext: $(cat /sys/kernel/sched_ext/state 2>/dev/null)" >&2

# Wait for FAISS to finish
wait $FAISS_PID
RC=$?
echo "FAISS done (rc=$RC)" >&2

# Show stats
echo "=== sched stats ===" >&2
tail -5 "$RESULT_DIR/${RESULT_NAME}_sched.log" >&2

echo "=== Result: $RESULT_DIR/${RESULT_NAME}.json ===" >&2

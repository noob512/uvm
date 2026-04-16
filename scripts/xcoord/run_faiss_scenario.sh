#!/bin/bash
# Run a single FAISS scenario for xCoord benchmarking
# Usage: ./run_faiss_scenario.sh <scenario> [stress_cores]
# Scenarios: B0 (baseline), B1 (stress), B3 (stress+boost), B5 (boost only)

set -euo pipefail

SCENARIO="${1:-B0}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RESULT_DIR="$REPO_ROOT/workloads/faiss/xcoord_results"
SCHED="$REPO_ROOT/extension/sched_gpu_baseline"
mkdir -p "$RESULT_DIR"

WITH_STRESS=false
WITH_BOOST=false

case "$SCENARIO" in
  B0) ;;
  B1) WITH_STRESS=true ;;
  B3) WITH_STRESS=true; WITH_BOOST=true ;;
  B5) WITH_BOOST=true ;;
  *) echo "Unknown scenario: $SCENARIO"; exit 1 ;;
esac

echo "=== FAISS $SCENARIO ===" >&2
cd "$REPO_ROOT"
python3 workloads/cleanup_gpu.py 2>&1

PIDS_TO_KILL=()
trap 'for p in "${PIDS_TO_KILL[@]}"; do sudo kill "$p" 2>/dev/null || kill "$p" 2>/dev/null || true; done; killall stress-ng 2>/dev/null || true' EXIT

if $WITH_STRESS; then
  NPROC=$(nproc)
  stress-ng -c "$NPROC" --timeout 600 &
  PIDS_TO_KILL+=($!)
  echo "stress-ng started ($NPROC cores)" >&2
  sleep 2
fi

if $WITH_BOOST; then
  # Start FAISS in background, get GPU PID, attach scheduler
  uv run --directory workloads/faiss python configs/search.py \
    --dataset SIFT100M --uvm --nprobe "1,4,16" --no-cleanup \
    -o "$RESULT_DIR/${SCENARIO}.json" > "$RESULT_DIR/${SCENARIO}_faiss.log" 2>&1 &
  FAISS_PID=$!
  echo "FAISS PID: $FAISS_PID" >&2

  GPU_PID=""
  for i in $(seq 1 30); do
    GPU_PID=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    [ -n "$GPU_PID" ] && break
    sleep 2
  done
  echo "GPU PID: $GPU_PID" >&2

  sudo "$SCHED" -p "${GPU_PID:-$FAISS_PID}" > "$RESULT_DIR/${SCENARIO}_sched.log" 2>&1 &
  PIDS_TO_KILL+=($!)
  sleep 2
  echo "sched_ext: $(cat /sys/kernel/sched_ext/state 2>/dev/null)" >&2

  wait $FAISS_PID
  echo "FAISS done" >&2
  echo "=== sched stats ===" >&2
  tail -3 "$RESULT_DIR/${SCENARIO}_sched.log" >&2
else
  # Direct run (no scheduler)
  uv run --directory workloads/faiss python configs/search.py \
    --dataset SIFT100M --uvm --nprobe "1,4,16" --no-cleanup \
    -o "$RESULT_DIR/${SCENARIO}.json" 2>&1
fi

echo "=== $SCENARIO done ===" >&2

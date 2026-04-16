#!/bin/bash
# Run llama.cpp 20B benchmark with sched_gpu_serving policy
# Each scenario runs separately to avoid timeout issues
set -e

cd /home/yunwei37/workspace/gpu/gpu_ext

SCHEDULER="extension/sched_gpu_serving"
OUTPUT_DIR="scripts/xcoord/results/poc1_serving_r1"
COMMON_ARGS="--model-20b --prompts 20 --max-concurrency 1 --request-rate 0.2 --skip-gpu-ext --scheduler ${SCHEDULER}"

echo "=== Running POC-1 with sched_gpu_serving ==="
echo "=== Scenario: ALL (baseline, stress, xcoord) ==="
date

uv run --directory workloads/llama.cpp python ../../scripts/xcoord/poc1_xcoord_bench.py \
    ${COMMON_ARGS} \
    --output-dir "${OUTPUT_DIR}" 2>&1

echo "=== Done ==="
date

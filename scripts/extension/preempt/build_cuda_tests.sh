#!/bin/bash
# Build CUDA test programs for preempt demo
set -euo pipefail
cd "$(dirname "$0")"

echo "Building CUDA test programs..."
nvcc -o test_priority_demo test_priority_demo.cu -lcuda
nvcc -o test_cuda_launch test_cuda_launch.cu -lcuda
echo "Done: test_priority_demo, test_cuda_launch"

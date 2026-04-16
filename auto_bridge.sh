#!/bin/bash

set -euo pipefail

BRIDGE_SCRIPT="workloads/vllm/score_bridge_vllm.py"

# 使用外部环境变量覆盖默认值
KV_PTR="${KV_PTR:-}"
NUM_BLOCKS="${NUM_BLOCKS:-1024}"
BLOCK_SIZE_KB="${BLOCK_SIZE_KB:-256}"
INTERVAL="${INTERVAL:-1.0}"
KV_SCORE="${KV_SCORE:-20000}"
KV_TIER="${KV_TIER:-1}"
WEIGHT_SCORE="${WEIGHT_SCORE:-65535}"
WEIGHT_TIER="${WEIGHT_TIER:-2}"

if [ -z "$KV_PTR" ]; then
    echo "❌ 缺少 KV_PTR。"
    echo "请先导出 KV 基址，例如："
    echo "  export KV_PTR=0x7f8a12345000"
    exit 1
fi

echo "✅ 使用 KV 基址: $KV_PTR"
echo "🚀 启动位置/类别固定赋分模式的 score bridge..."

sudo workloads/vllm/.venv/bin/python "$BRIDGE_SCRIPT" daemon \
    --kv-cache-ptr "$KV_PTR" \
    --num-blocks "$NUM_BLOCKS" \
    --block-size-kb "$BLOCK_SIZE_KB" \
    --kv-score "$KV_SCORE" \
    --kv-tier "$KV_TIER" \
    --weight-score "$WEIGHT_SCORE" \
    --weight-tier "$WEIGHT_TIER" \
    --interval "$INTERVAL" \
    --stats

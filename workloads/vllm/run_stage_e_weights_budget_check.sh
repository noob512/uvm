#!/usr/bin/env bash
# Convenience wrapper for Stage E model-weights budget telemetry validation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROMPTS="${PROMPTS:-1}"
REQUEST_RATE="${REQUEST_RATE:-5}"
OUTPUT_LEN="${OUTPUT_LEN:-512}"
WEIGHT_BUDGET_BYTES="${WEIGHT_BUDGET_BYTES:-1048576}"
WEIGHT_BUDGET_MODE="${WEIGHT_BUDGET_MODE:-trace_only}"
KV_BUDGET_BYTES="${KV_BUDGET_BYTES:-0}"
KV_BUDGET_MODE="${KV_BUDGET_MODE:-trace_only}"
RUN_BENCH="${RUN_BENCH:-0}"
ENABLE_MOE_ROUTING_TRACE="${ENABLE_MOE_ROUTING_TRACE:-0}"
REQUIRE_MOE_ROUTING_TRACE="${REQUIRE_MOE_ROUTING_TRACE:-0}"
OUT_DIR="${OUT_DIR:-/tmp/vllm_stage_e_weights_budget_check_$(date +%Y%m%d_%H%M%S)}"

cmd=(
  python3 "$SCRIPT_DIR/check_stage_e_success.py"
  --run-dir "$OUT_DIR"
  --budget-bytes "$WEIGHT_BUDGET_BYTES"
  --budget-mode "$WEIGHT_BUDGET_MODE"
  --kv-budget-bytes "$KV_BUDGET_BYTES"
  --kv-budget-mode "$KV_BUDGET_MODE"
  --prompts "$PROMPTS"
  --request-rate "$REQUEST_RATE"
  --output-len "$OUTPUT_LEN"
)

if [ "$RUN_BENCH" = "1" ]; then
  cmd+=(--run-bench)
fi

if [ "$ENABLE_MOE_ROUTING_TRACE" = "1" ]; then
  cmd+=(--enable-moe-routing-trace)
fi

if [ "$REQUIRE_MOE_ROUTING_TRACE" = "1" ]; then
  cmd+=(--require-moe-routing-trace)
fi

"${cmd[@]}"

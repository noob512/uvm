#!/usr/bin/env bash
# Convenience wrapper for Stage D KV independent budget telemetry validation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROMPTS="${PROMPTS:-1}"
REQUEST_RATE="${REQUEST_RATE:-5}"
OUTPUT_LEN="${OUTPUT_LEN:-512}"
KV_BUDGET_BYTES="${KV_BUDGET_BYTES:-1048576}"
KV_BUDGET_MODE="${KV_BUDGET_MODE:-trace_only}"
RUN_BENCH="${RUN_BENCH:-0}"
OUT_DIR="${OUT_DIR:-/tmp/vllm_stage_d_kv_budget_check_$(date +%Y%m%d_%H%M%S)}"

cmd=(
  python3 "$SCRIPT_DIR/check_stage_d_success.py"
  --run-dir "$OUT_DIR"
  --budget-bytes "$KV_BUDGET_BYTES"
  --budget-mode "$KV_BUDGET_MODE"
  --prompts "$PROMPTS"
  --request-rate "$REQUEST_RATE"
  --output-len "$OUTPUT_LEN"
)

if [ "$RUN_BENCH" = "1" ]; then
  cmd+=(--run-bench)
fi

"${cmd[@]}"

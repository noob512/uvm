#!/usr/bin/env bash
# Run Stage C1 attention-only cuda_malloc A/B over a list of live-byte budgets.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PROMPTS="${PROMPTS:-10}"
REQUEST_RATE="${REQUEST_RATE:-5}"
OUTPUT_LEN="${OUTPUT_LEN:-512}"
BUDGETS_CSV="${BUDGETS_CSV:-524288,1048576,2097152,4194304}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
ROOT_OUT_DIR="${ROOT_OUT_DIR:-/tmp/vllm_stage_c1_budget_sweep_${RUN_ID}}"
SUMMARY_JSON="$ROOT_OUT_DIR/vllm_stage_c1_budget_sweep_p${PROMPTS}.json"

mkdir -p "$ROOT_OUT_DIR"

IFS=',' read -r -a BUDGETS <<< "$BUDGETS_CSV"
RUN_ARGS=()

echo "==========================================================="
echo " Stage C1 Attention Budget Sweep"
echo " Root output dir: $ROOT_OUT_DIR"
echo " Prompts: $PROMPTS"
echo " Request rate: $REQUEST_RATE"
echo " Output len: $OUTPUT_LEN"
echo " Budgets: $BUDGETS_CSV"
echo " Backend: cuda_malloc"
echo "==========================================================="

for budget in "${BUDGETS[@]}"; do
  if ! [[ "$budget" =~ ^[0-9]+$ ]]; then
    echo "Invalid budget in BUDGETS_CSV: $budget" >&2
    exit 1
  fi

  out_dir="$ROOT_OUT_DIR/budget_${budget}"
  report_json="$out_dir/vllm_stage_c_attention_p${PROMPTS}_ab_comparison.json"

  echo
  echo "==========================================================="
  echo " Stage C1 budget: $budget bytes"
  echo "==========================================================="
  PROMPTS="$PROMPTS" \
  REQUEST_RATE="$REQUEST_RATE" \
  OUTPUT_LEN="$OUTPUT_LEN" \
  DEVICE_DIRECT_MAX_TOTAL_BYTES="$budget" \
  DEVICE_DIRECT_BACKEND="cuda_malloc" \
  OUT_DIR="$out_dir" \
  RUN_ID="${RUN_ID}_budget_${budget}" \
  ./run_stage_c_attention_p20_ab.sh

  RUN_ARGS+=(--run "$budget=$report_json")
done

echo
echo "==========================================================="
echo " Summarize C1 budget sweep"
echo "==========================================================="
python3 "$SCRIPT_DIR/compare_stage_c_budget_sweep.py" \
  "${RUN_ARGS[@]}" \
  --output-json "$SUMMARY_JSON"

echo
echo "==========================================================="
echo " Done"
echo " Budget sweep summary: $SUMMARY_JSON"
echo " Raw logs: $ROOT_OUT_DIR"
echo "==========================================================="

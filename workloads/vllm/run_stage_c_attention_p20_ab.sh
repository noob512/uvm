#!/usr/bin/env bash
# Run two p20 experiments back-to-back and compare:
# 1) Stage B strict trace-only baseline
# 2) Stage C attention-only device_direct

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PROMPTS="${PROMPTS:-20}"
REQUEST_RATE="${REQUEST_RATE:-5}"
OUTPUT_LEN="${OUTPUT_LEN:-512}"
DEVICE_DIRECT_MAX_TOTAL_BYTES="${DEVICE_DIRECT_MAX_TOTAL_BYTES:-268435456}"
DEVICE_DIRECT_BACKEND="${DEVICE_DIRECT_BACKEND:-cuda_malloc}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-/tmp/vllm_stage_c_attention_p20_ab_${RUN_ID}}"

mkdir -p "$OUT_DIR"

echo "==========================================================="
echo " Stage C Attention A/B p${PROMPTS}"
echo " Output dir: $OUT_DIR"
echo " Device-direct max total bytes: $DEVICE_DIRECT_MAX_TOTAL_BYTES"
echo " Device-direct backend: $DEVICE_DIRECT_BACKEND"
echo "==========================================================="

echo
echo "==========================================================="
echo " Phase A: Stage B strict trace-only baseline"
echo "==========================================================="
./run_kv_fault_ratio.sh \
  --mode trace \
  --with-address-log \
  --trace-log "$OUT_DIR/uvm_kv_fault_stats_gap2_strict_attention_trace_p${PROMPTS}.log" \
  --address-trace-log "$OUT_DIR/uvm_kv_fault_addrs_gap2_strict_attention_trace_p${PROMPTS}.log" \
  --allocator-log "$OUT_DIR/vllm_uvm_allocator_trace_gap2_strict_attention_trace_p${PROMPTS}.log" \
  --bench-log "$OUT_DIR/vllm_bench_gap2_strict_attention_trace_p${PROMPTS}.log" \
  --uvm-trace-min-bytes 1048576 \
  --uvm-unknown-detail-enable 1 \
  --uvm-unknown-detail-min-bytes 4096 \
  --uvm-gap-watch-name "same_run_gap2_strict_attention_trace_p${PROMPTS}" \
  --uvm-gap-watch-all-classes 1 \
  --uvm-gap-watch-min-bytes 4096 \
  --uvm-device-direct-enable 0 \
  --uvm-device-direct-min-bytes 4096 \
  --uvm-device-direct-max-bytes 1048576 \
  --uvm-device-direct-max-total-bytes "$DEVICE_DIRECT_MAX_TOTAL_BYTES" \
  --uvm-device-direct-backend "$DEVICE_DIRECT_BACKEND" \
  --uvm-device-direct-target-phases enabled:attention \
  --auto-gap-watch-enable 1 \
  --auto-gap-watch-probe-prompts 1 \
  --auto-gap-watch-target-gap 2 \
  --auto-gap-watch-policy-action-override device_direct_trace \
  --auto-gap-watch-target-class-override gap_hot_runtime_scratch \
  --prompts "$PROMPTS" \
  --request-rate "$REQUEST_RATE" \
  --output-len "$OUTPUT_LEN" \
  --gap-watch-metrics-summary-json "$OUT_DIR/vllm_gap_watch_metrics_gap2_strict_attention_trace_p${PROMPTS}.json" \
  --auto-gap-watch-summary-json "$OUT_DIR/vllm_auto_gap_watch_summary_gap2_strict_attention_trace_p${PROMPTS}.json" \
  --auto-gap-watch-post-main-summary-json "$OUT_DIR/vllm_auto_gap_watch_post_main_summary_gap2_strict_attention_trace_p${PROMPTS}.json"

echo
echo "==========================================================="
echo " Phase B: Stage C attention-only device_direct"
echo "==========================================================="
./run_kv_fault_ratio.sh \
  --mode trace \
  --with-address-log \
  --trace-log "$OUT_DIR/uvm_kv_fault_stats_gap2_stage_c_attention_p${PROMPTS}.log" \
  --address-trace-log "$OUT_DIR/uvm_kv_fault_addrs_gap2_stage_c_attention_p${PROMPTS}.log" \
  --allocator-log "$OUT_DIR/vllm_uvm_allocator_trace_gap2_stage_c_attention_p${PROMPTS}.log" \
  --bench-log "$OUT_DIR/vllm_bench_gap2_stage_c_attention_p${PROMPTS}.log" \
  --uvm-trace-min-bytes 1048576 \
  --uvm-unknown-detail-enable 1 \
  --uvm-unknown-detail-min-bytes 4096 \
  --uvm-gap-watch-name "same_run_gap2_stage_c_attention_p${PROMPTS}" \
  --uvm-gap-watch-all-classes 1 \
  --uvm-gap-watch-min-bytes 4096 \
  --uvm-device-direct-enable 1 \
  --uvm-device-direct-min-bytes 4096 \
  --uvm-device-direct-max-bytes 1048576 \
  --uvm-device-direct-max-total-bytes "$DEVICE_DIRECT_MAX_TOTAL_BYTES" \
  --uvm-device-direct-backend "$DEVICE_DIRECT_BACKEND" \
  --uvm-device-direct-target-phases enabled:attention \
  --auto-gap-watch-enable 1 \
  --auto-gap-watch-probe-prompts 1 \
  --auto-gap-watch-target-gap 2 \
  --auto-gap-watch-policy-action-override device_direct \
  --auto-gap-watch-target-class-override gap_hot_runtime_scratch \
  --prompts "$PROMPTS" \
  --request-rate "$REQUEST_RATE" \
  --output-len "$OUTPUT_LEN" \
  --gap-watch-metrics-summary-json "$OUT_DIR/vllm_gap_watch_metrics_gap2_stage_c_attention_p${PROMPTS}.json" \
  --auto-gap-watch-summary-json "$OUT_DIR/vllm_auto_gap_watch_summary_gap2_stage_c_attention_p${PROMPTS}.json" \
  --auto-gap-watch-post-main-summary-json "$OUT_DIR/vllm_auto_gap_watch_post_main_summary_gap2_stage_c_attention_p${PROMPTS}.json"

echo
echo "==========================================================="
echo " Phase C: Compare p${PROMPTS} results"
echo "==========================================================="
python3 "$SCRIPT_DIR/compare_stage_c_attention_p20_ab.py" \
  --trace-probe "$OUT_DIR/vllm_auto_gap_watch_summary_gap2_strict_attention_trace_p${PROMPTS}.json" \
  --trace-post-main "$OUT_DIR/vllm_auto_gap_watch_post_main_summary_gap2_strict_attention_trace_p${PROMPTS}.json" \
  --trace-metrics "$OUT_DIR/vllm_gap_watch_metrics_gap2_strict_attention_trace_p${PROMPTS}.json" \
  --trace-bench-log "$OUT_DIR/vllm_bench_gap2_strict_attention_trace_p${PROMPTS}.log" \
  --device-probe "$OUT_DIR/vllm_auto_gap_watch_summary_gap2_stage_c_attention_p${PROMPTS}.json" \
  --device-post-main "$OUT_DIR/vllm_auto_gap_watch_post_main_summary_gap2_stage_c_attention_p${PROMPTS}.json" \
  --device-metrics "$OUT_DIR/vllm_gap_watch_metrics_gap2_stage_c_attention_p${PROMPTS}.json" \
  --device-bench-log "$OUT_DIR/vllm_bench_gap2_stage_c_attention_p${PROMPTS}.log" \
  --output-json "$OUT_DIR/vllm_stage_c_attention_p${PROMPTS}_ab_comparison.json"

echo
echo "==========================================================="
echo " Done"
echo " Report: $OUT_DIR/vllm_stage_c_attention_p${PROMPTS}_ab_comparison.json"
echo " Raw logs: $OUT_DIR"
echo "==========================================================="

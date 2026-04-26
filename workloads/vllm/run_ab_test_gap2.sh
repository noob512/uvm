#!/usr/bin/env bash
# 此脚本用于一键运行 UVM Gap 2 的 "Observe" 与 "Prefetch" A/B 对比测试

# 开启严格错误捕获：任何命令失败都会立刻终止脚本
set -euo pipefail

WORK_DIR="/home/ubuntu/nvidia-uvm-gpu/workloads/vllm"
echo "[Info] 切换到工作目录: $WORK_DIR"
cd "$WORK_DIR" || { echo "无法切换到工作目录 $WORK_DIR"; exit 1; }

echo "==========================================================="
echo " Phase A: 运行基准组 (Baseline) - 策略: observe"
echo "==========================================================="
./run_kv_fault_ratio.sh \
  --mode trace \
  --with-address-log \
  --trace-log /tmp/uvm_kv_fault_stats_gap2_observe.log \
  --address-trace-log /tmp/uvm_kv_fault_addrs_gap2_observe.log \
  --allocator-log /tmp/vllm_uvm_allocator_trace_gap2_observe.log \
  --uvm-trace-min-bytes 1048576 \
  --uvm-unknown-detail-enable 1 \
  --uvm-unknown-detail-min-bytes 4096 \
  --uvm-gap-watch-name same_run_gap2_observe \
  --uvm-gap-watch-all-classes 1 \
  --uvm-gap-watch-min-bytes 4096 \
  --auto-gap-watch-enable 1 \
  --auto-gap-watch-probe-prompts 1 \
  --auto-gap-watch-target-gap 2 \
  --auto-gap-watch-policy-action-override observe \
  --prompts 20 \
  --gap-watch-metrics-summary-json /tmp/vllm_gap_watch_metrics_gap2_observe.json \
  --auto-gap-watch-summary-json /tmp/vllm_auto_gap_watch_summary_gap2_observe.json \
  --auto-gap-watch-post-main-summary-json /tmp/vllm_auto_gap_watch_post_main_summary_gap2_observe.json


echo "==========================================================="
echo " Phase B: 运行实验组 (Experimental) - 策略: prefetch"
echo "==========================================================="
./run_kv_fault_ratio.sh \
  --mode trace \
  --with-address-log \
  --trace-log /tmp/uvm_kv_fault_stats_gap2_prefetch.log \
  --address-trace-log /tmp/uvm_kv_fault_addrs_gap2_prefetch.log \
  --allocator-log /tmp/vllm_uvm_allocator_trace_gap2_prefetch.log \
  --uvm-trace-min-bytes 1048576 \
  --uvm-unknown-detail-enable 1 \
  --uvm-unknown-detail-min-bytes 4096 \
  --uvm-gap-watch-name same_run_gap2_prefetch \
  --uvm-gap-watch-all-classes 1 \
  --uvm-gap-watch-min-bytes 4096 \
  --auto-gap-watch-enable 1 \
  --auto-gap-watch-probe-prompts 1 \
  --auto-gap-watch-target-gap 2 \
  --auto-gap-watch-policy-action-override prefetch \
  --prompts 20 \
  --gap-watch-metrics-summary-json /tmp/vllm_gap_watch_metrics_gap2_prefetch.json \
  --auto-gap-watch-summary-json /tmp/vllm_auto_gap_watch_summary_gap2_prefetch.json \
  --auto-gap-watch-post-main-summary-json /tmp/vllm_auto_gap_watch_post_main_summary_gap2_prefetch.json


echo "==========================================================="
echo " Phase C: 生成 A/B 测试对比报告"
echo "==========================================================="
python3 compare_gap_watch_ab.py \
  --observe-post-main /tmp/vllm_auto_gap_watch_post_main_summary_gap2_observe.json \
  --observe-metrics /tmp/vllm_gap_watch_metrics_gap2_observe.json \
  --prefetch-post-main /tmp/vllm_auto_gap_watch_post_main_summary_gap2_prefetch.json \
  --prefetch-metrics /tmp/vllm_gap_watch_metrics_gap2_prefetch.json \
  --output-json /tmp/vllm_gap_watch_ab_gap2.json

echo "==========================================================="
echo "[Success] 所有任务执行完毕！"
echo "请查看最终对比结果: /tmp/vllm_gap_watch_ab_gap2.json"
echo "==========================================================="

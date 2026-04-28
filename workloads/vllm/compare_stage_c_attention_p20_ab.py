#!/usr/bin/env python3
"""Compare Stage B trace-only vs Stage C attention-only device_direct p20 runs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


BENCH_PATTERNS: dict[str, tuple[str, type]] = {
    "successful_requests": (r"Successful requests:\s+(\d+)", int),
    "failed_requests": (r"Failed requests:\s+(\d+)", int),
    "benchmark_duration_s": (r"Benchmark duration \(s\):\s+([\d.]+)", float),
    "request_throughput_req_s": (r"Request throughput \(req/s\):\s+([\d.]+)", float),
    "output_throughput_tok_s": (r"Output token throughput \(tok/s\):\s+([\d.]+)", float),
    "peak_output_throughput_tok_s": (
        r"Peak output token throughput \(tok/s\):\s+([\d.]+)",
        float,
    ),
    "peak_concurrent_requests": (r"Peak concurrent requests:\s+([\d.]+)", float),
    "total_token_throughput_tok_s": (r"Total Token throughput \(tok/s\):\s+([\d.]+)", float),
    "mean_ttft_ms": (r"Mean TTFT \(ms\):\s+([\d.]+)", float),
    "median_ttft_ms": (r"Median TTFT \(ms\):\s+([\d.]+)", float),
    "p99_ttft_ms": (r"P99 TTFT \(ms\):\s+([\d.]+)", float),
    "mean_tpot_ms": (r"Mean TPOT \(ms\):\s+([\d.]+)", float),
    "median_tpot_ms": (r"Median TPOT \(ms\):\s+([\d.]+)", float),
    "p99_tpot_ms": (r"P99 TPOT \(ms\):\s+([\d.]+)", float),
    "mean_itl_ms": (r"Mean ITL \(ms\):\s+([\d.]+)", float),
    "median_itl_ms": (r"Median ITL \(ms\):\s+([\d.]+)", float),
    "p99_itl_ms": (r"P99 ITL \(ms\):\s+([\d.]+)", float),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a p20 Stage B trace-only run with a p20 Stage C "
            "attention-only device_direct run."
        )
    )
    parser.add_argument("--trace-probe", required=True)
    parser.add_argument("--trace-post-main", required=True)
    parser.add_argument("--trace-metrics", required=True)
    parser.add_argument("--trace-bench-log", required=True)
    parser.add_argument("--device-probe", required=True)
    parser.add_argument("--device-post-main", required=True)
    parser.add_argument("--device-metrics", required=True)
    parser.add_argument("--device-bench-log", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def load_json(path: str) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        raise SystemExit(f"file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def selected_gap(summary: dict[str, Any]) -> dict[str, Any]:
    gap = summary.get("selected_gap")
    return gap if isinstance(gap, dict) else {}


def gap_identity(summary: dict[str, Any]) -> tuple[Any, Any, Any]:
    gap = selected_gap(summary)
    return gap.get("gap_index"), gap.get("start_hex"), gap.get("end_hex")


def pct_delta(new_value: Any, old_value: Any) -> float | None:
    if new_value is None or old_value in (None, 0):
        return None
    return (float(new_value) - float(old_value)) / float(old_value)


def numeric_delta(new_value: Any, old_value: Any) -> float | int | None:
    if new_value is None or old_value is None:
        return None
    return new_value - old_value


def parse_bench_main(path: str) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    sections = re.split(r"===== Benchmark phase:\s+", text)
    main_sections = [section for section in sections if section.startswith("main =====")]
    section = main_sections[-1] if main_sections else text

    result: dict[str, Any] = {"bench_log": path, "phase_found": bool(main_sections)}
    for key, (pattern, caster) in BENCH_PATTERNS.items():
        matches = re.findall(pattern, section)
        if matches:
            result[key] = caster(matches[-1])
        else:
            result[key] = None
    return result


def run_summary(
    *,
    name: str,
    probe: dict[str, Any],
    post: dict[str, Any],
    metrics: dict[str, Any],
    bench: dict[str, Any],
) -> dict[str, Any]:
    gap = selected_gap(post)
    probe_gap = selected_gap(probe)
    classification = post.get("selected_gap_allocator_classification", {})
    if not isinstance(classification, dict):
        classification = {}

    placement = metrics.get("placement_backend_counts", {})
    if not isinstance(placement, dict):
        placement = {}

    write_ratio = None
    access_ratios = gap.get("access_ratios", {})
    if isinstance(access_ratios, dict):
        write_ratio = access_ratios.get("UVM_FAULT_ACCESS_TYPE_WRITE")

    return {
        "name": name,
        "effective_policy_action": post.get("effective_policy_action"),
        "effective_target_class": post.get("effective_target_class"),
        "probe_main_same_gap": gap_identity(probe) == gap_identity(post),
        "probe_gap": {
            "gap_index": probe_gap.get("gap_index"),
            "start_hex": probe_gap.get("start_hex"),
            "end_hex": probe_gap.get("end_hex"),
            "faults": probe_gap.get("faults"),
            "fault_share_of_unknown": probe_gap.get("fault_share_of_unknown"),
        },
        "main_gap": {
            "gap_index": gap.get("gap_index"),
            "start_hex": gap.get("start_hex"),
            "end_hex": gap.get("end_hex"),
            "size_mib": gap.get("size_mib"),
            "faults": gap.get("faults"),
            "unique_pages": gap.get("unique_pages"),
            "avg_faults_per_unique_page": gap.get("avg_faults_per_unique_page"),
            "fault_share_of_unknown": gap.get("fault_share_of_unknown"),
            "write_ratio": write_ratio,
            "left_region": gap.get("left_region"),
            "right_region": gap.get("right_region"),
        },
        "total_unknown_faults": post.get("total_unknown_faults"),
        "allocator_classification": {
            "dominant_phase": classification.get("dominant_phase"),
            "dominant_predicted_class": classification.get("dominant_predicted_class"),
            "median_lifetime_s": classification.get("median_lifetime_s"),
            "p95_lifetime_s": classification.get("p95_lifetime_s"),
            "phase_counts": classification.get("phase_counts"),
        },
        "metrics": {
            "dominant_action": metrics.get("dominant_action"),
            "dominant_phase": metrics.get("dominant_phase"),
            "dominant_target_class": metrics.get("dominant_target_class"),
            "gap_overlap_records": metrics.get("gap_overlap_records"),
            "gap_policy_records": metrics.get("gap_policy_records"),
            "gap_policy_success": metrics.get("gap_policy_success"),
            "gap_policy_fail": metrics.get("gap_policy_fail"),
            "device_direct_eligible_records": metrics.get(
                "device_direct_eligible_records"
            ),
            "device_direct_actual_records": metrics.get("device_direct_actual_records"),
            "device_direct_backend_counts": metrics.get(
                "device_direct_backend_counts"
            ),
            "device_direct_budget_reject_records": metrics.get(
                "device_direct_budget_reject_records"
            ),
            "device_direct_max_total_bytes": metrics.get(
                "device_direct_max_total_bytes"
            ),
            "device_direct_peak_live_bytes_observed": metrics.get(
                "device_direct_peak_live_bytes_observed"
            ),
            "device_direct_min_budget_remaining_observed": metrics.get(
                "device_direct_min_budget_remaining_observed"
            ),
            "device_direct_reason_counts": metrics.get("device_direct_reason_counts"),
            "hot_gap_match_records": metrics.get("hot_gap_match_records"),
            "median_lifetime_s": metrics.get("median_lifetime_s"),
            "phase_record_ratios": metrics.get("phase_record_ratios"),
            "placement_backend_counts": placement,
        },
        "bench_main": bench,
    }


def verdict(trace_run: dict[str, Any], device_run: dict[str, Any]) -> dict[str, Any]:
    trace_bench = trace_run["bench_main"]
    device_bench = device_run["bench_main"]
    trace_gap = trace_run["main_gap"]
    device_gap = device_run["main_gap"]
    device_metrics = device_run["metrics"]
    placement = device_metrics.get("placement_backend_counts") or {}

    gap_fault_delta = numeric_delta(device_gap.get("faults"), trace_gap.get("faults"))
    gap_fault_delta_pct = pct_delta(device_gap.get("faults"), trace_gap.get("faults"))
    unknown_delta = numeric_delta(
        device_run.get("total_unknown_faults"), trace_run.get("total_unknown_faults")
    )
    unknown_delta_pct = pct_delta(
        device_run.get("total_unknown_faults"), trace_run.get("total_unknown_faults")
    )
    tpot_delta_pct = pct_delta(
        device_bench.get("mean_tpot_ms"), trace_bench.get("mean_tpot_ms")
    )
    throughput_delta_pct = pct_delta(
        device_bench.get("output_throughput_tok_s"),
        trace_bench.get("output_throughput_tok_s"),
    )

    checks = {
        "trace_main_bench_parsed": bool(trace_bench.get("phase_found")),
        "device_main_bench_parsed": bool(device_bench.get("phase_found")),
        "trace_no_failed_requests": trace_bench.get("failed_requests") == 0,
        "device_no_failed_requests": device_bench.get("failed_requests") == 0,
        "trace_probe_main_same_gap": bool(trace_run.get("probe_main_same_gap")),
        "device_probe_main_same_gap": bool(device_run.get("probe_main_same_gap")),
        "device_direct_records_present": (
            (device_metrics.get("device_direct_actual_records") or 0) > 0
        ),
        "device_direct_backend_present": (placement.get("device_direct") or 0) > 0,
        "device_policy_fail_zero": device_metrics.get("gap_policy_fail") == 0,
        "device_peak_live_within_budget": (
            device_metrics.get("device_direct_max_total_bytes") in (None, 0)
            or (
                device_metrics.get("device_direct_peak_live_bytes_observed") is not None
                and device_metrics.get("device_direct_peak_live_bytes_observed")
                <= device_metrics.get("device_direct_max_total_bytes")
            )
        ),
        "device_gap_faults_lower_than_trace": (
            gap_fault_delta is not None and gap_fault_delta < 0
        ),
        "device_unknown_faults_lower_than_trace": (
            unknown_delta is not None and unknown_delta < 0
        ),
        "device_tpot_not_more_than_10pct_worse": (
            tpot_delta_pct is not None and tpot_delta_pct <= 0.10
        ),
    }

    return {
        "gap_fault_delta": gap_fault_delta,
        "gap_fault_delta_pct": gap_fault_delta_pct,
        "unknown_fault_delta": unknown_delta,
        "unknown_fault_delta_pct": unknown_delta_pct,
        "avg_faults_per_unique_page_delta": numeric_delta(
            device_gap.get("avg_faults_per_unique_page"),
            trace_gap.get("avg_faults_per_unique_page"),
        ),
        "avg_faults_per_unique_page_delta_pct": pct_delta(
            device_gap.get("avg_faults_per_unique_page"),
            trace_gap.get("avg_faults_per_unique_page"),
        ),
        "output_throughput_delta_pct": throughput_delta_pct,
        "mean_tpot_delta_pct": tpot_delta_pct,
        "mean_ttft_delta_pct": pct_delta(
            device_bench.get("mean_ttft_ms"), trace_bench.get("mean_ttft_ms")
        ),
        "checks": checks,
        "success_signal": (
            checks["device_no_failed_requests"]
            and checks["device_direct_records_present"]
            and checks["device_direct_backend_present"]
            and checks["device_policy_fail_zero"]
            and checks["device_peak_live_within_budget"]
        ),
        "effectiveness_signal": (
            checks["device_gap_faults_lower_than_trace"]
            and checks["device_unknown_faults_lower_than_trace"]
        ),
    }


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:+.2f}%"


def main() -> int:
    args = parse_args()

    trace_probe = load_json(args.trace_probe)
    trace_post = load_json(args.trace_post_main)
    trace_metrics = load_json(args.trace_metrics)
    device_probe = load_json(args.device_probe)
    device_post = load_json(args.device_post_main)
    device_metrics = load_json(args.device_metrics)

    trace_run = run_summary(
        name="trace_only",
        probe=trace_probe,
        post=trace_post,
        metrics=trace_metrics,
        bench=parse_bench_main(args.trace_bench_log),
    )
    device_run = run_summary(
        name="stage_c_attention_device_direct",
        probe=device_probe,
        post=device_post,
        metrics=device_metrics,
        bench=parse_bench_main(args.device_bench_log),
    )
    comparison = {
        "trace_only": trace_run,
        "stage_c_attention_device_direct": device_run,
        "comparison": verdict(trace_run, device_run),
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2, sort_keys=True)

    comp = comparison["comparison"]
    trace_gap = trace_run["main_gap"]
    device_gap = device_run["main_gap"]
    trace_bench = trace_run["bench_main"]
    device_bench = device_run["bench_main"]

    print("Stage C Attention p20 A/B Comparison")
    print(f"- output_json={output_path}")
    print(
        f"- trace_gap_faults={trace_gap.get('faults')} "
        f"device_gap_faults={device_gap.get('faults')} "
        f"delta={comp['gap_fault_delta']} "
        f"delta_pct={fmt_pct(comp['gap_fault_delta_pct'])}"
    )
    print(
        f"- trace_unknown_faults={trace_run.get('total_unknown_faults')} "
        f"device_unknown_faults={device_run.get('total_unknown_faults')} "
        f"delta={comp['unknown_fault_delta']} "
        f"delta_pct={fmt_pct(comp['unknown_fault_delta_pct'])}"
    )
    print(
        f"- trace_output_tok_s={trace_bench.get('output_throughput_tok_s')} "
        f"device_output_tok_s={device_bench.get('output_throughput_tok_s')} "
        f"delta_pct={fmt_pct(comp['output_throughput_delta_pct'])}"
    )
    print(
        f"- trace_mean_tpot_ms={trace_bench.get('mean_tpot_ms')} "
        f"device_mean_tpot_ms={device_bench.get('mean_tpot_ms')} "
        f"delta_pct={fmt_pct(comp['mean_tpot_delta_pct'])}"
    )
    print(
        "- device_direct_actual_records="
        f"{device_run['metrics'].get('device_direct_actual_records')} "
        "placement_backend_counts="
        f"{device_run['metrics'].get('placement_backend_counts')}"
    )
    print(
        "- device_direct_backend_counts="
        f"{device_run['metrics'].get('device_direct_backend_counts')}"
    )
    print(
        "- device_direct_budget="
        f"{device_run['metrics'].get('device_direct_max_total_bytes')} "
        "peak_live_observed="
        f"{device_run['metrics'].get('device_direct_peak_live_bytes_observed')} "
        "budget_reject_records="
        f"{device_run['metrics'].get('device_direct_budget_reject_records')}"
    )
    print(
        f"- success_signal={comp['success_signal']} "
        f"effectiveness_signal={comp['effectiveness_signal']}"
    )
    print("- checks:")
    for key, value in comp["checks"].items():
        print(f"  {key}={value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

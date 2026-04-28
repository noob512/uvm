#!/usr/bin/env python3
"""Compare Stage C attention device-direct backends from two A/B run reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two run_stage_c_attention_p20_ab.py JSON reports, normally "
            "cuda_malloc vs cuda_malloc_async with identical prompts and budget."
        )
    )
    parser.add_argument("--sync-json", required=True, help="cuda_malloc A/B report")
    parser.add_argument("--async-json", required=True, help="cuda_malloc_async A/B report")
    parser.add_argument("--output-json", required=True, help="comparison output path")
    return parser.parse_args()


def load_json(path: str) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.is_file():
        raise SystemExit(f"file not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pct_delta(new_value: Any, old_value: Any) -> float | None:
    if new_value is None or old_value in (None, 0):
        return None
    return (float(new_value) - float(old_value)) / float(old_value)


def numeric_delta(new_value: Any, old_value: Any) -> float | int | None:
    if new_value is None or old_value is None:
        return None
    return new_value - old_value


def get_path(data: dict[str, Any], *path: str) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def device_run(report: dict[str, Any]) -> dict[str, Any]:
    run = report.get("stage_c_attention_device_direct")
    if not isinstance(run, dict):
        raise SystemExit("report missing stage_c_attention_device_direct")
    return run


def compact_run(report: dict[str, Any]) -> dict[str, Any]:
    run = device_run(report)
    metrics = run.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    bench = run.get("bench_main", {})
    if not isinstance(bench, dict):
        bench = {}
    gap = run.get("main_gap", {})
    if not isinstance(gap, dict):
        gap = {}
    comparison = report.get("comparison", {})
    if not isinstance(comparison, dict):
        comparison = {}

    return {
        "success_signal": comparison.get("success_signal"),
        "effectiveness_signal": comparison.get("effectiveness_signal"),
        "gap_fault_delta_pct_vs_trace": comparison.get("gap_fault_delta_pct"),
        "unknown_fault_delta_pct_vs_trace": comparison.get("unknown_fault_delta_pct"),
        "output_throughput_delta_pct_vs_trace": comparison.get(
            "output_throughput_delta_pct"
        ),
        "mean_tpot_delta_pct_vs_trace": comparison.get("mean_tpot_delta_pct"),
        "checks": comparison.get("checks"),
        "gap_faults": gap.get("faults"),
        "unknown_faults": run.get("total_unknown_faults"),
        "unique_pages": gap.get("unique_pages"),
        "avg_faults_per_unique_page": gap.get("avg_faults_per_unique_page"),
        "successful_requests": bench.get("successful_requests"),
        "failed_requests": bench.get("failed_requests"),
        "output_throughput_tok_s": bench.get("output_throughput_tok_s"),
        "total_token_throughput_tok_s": bench.get("total_token_throughput_tok_s"),
        "mean_tpot_ms": bench.get("mean_tpot_ms"),
        "median_tpot_ms": bench.get("median_tpot_ms"),
        "p99_tpot_ms": bench.get("p99_tpot_ms"),
        "mean_ttft_ms": bench.get("mean_ttft_ms"),
        "p99_ttft_ms": bench.get("p99_ttft_ms"),
        "device_direct_actual_records": metrics.get("device_direct_actual_records"),
        "device_direct_eligible_records": metrics.get("device_direct_eligible_records"),
        "device_direct_budget_reject_records": metrics.get(
            "device_direct_budget_reject_records"
        ),
        "device_direct_max_total_bytes": metrics.get("device_direct_max_total_bytes"),
        "device_direct_peak_live_bytes_observed": metrics.get(
            "device_direct_peak_live_bytes_observed"
        ),
        "device_direct_min_budget_remaining_observed": metrics.get(
            "device_direct_min_budget_remaining_observed"
        ),
        "device_direct_backend_counts": metrics.get("device_direct_backend_counts"),
        "device_direct_reason_counts": metrics.get("device_direct_reason_counts"),
        "placement_backend_counts": metrics.get("placement_backend_counts"),
        "gap_policy_fail": metrics.get("gap_policy_fail"),
        "dominant_phase": metrics.get("dominant_phase"),
        "dominant_target_class": metrics.get("dominant_target_class"),
        "phase_record_ratios": metrics.get("phase_record_ratios"),
        "median_lifetime_s": metrics.get("median_lifetime_s"),
    }


def backend_count(run: dict[str, Any], backend: str) -> int:
    counts = run.get("device_direct_backend_counts")
    if not isinstance(counts, dict):
        return 0
    return int(counts.get(backend) or 0)


def build_comparison(sync_report: dict[str, Any], async_report: dict[str, Any]) -> dict[str, Any]:
    sync_run = compact_run(sync_report)
    async_run = compact_run(async_report)

    sync_gap_faults = sync_run.get("gap_faults")
    async_gap_faults = async_run.get("gap_faults")
    sync_unknown_faults = sync_run.get("unknown_faults")
    async_unknown_faults = async_run.get("unknown_faults")
    sync_tpot = sync_run.get("mean_tpot_ms")
    async_tpot = async_run.get("mean_tpot_ms")
    sync_throughput = sync_run.get("output_throughput_tok_s")
    async_throughput = async_run.get("output_throughput_tok_s")

    checks = {
        "sync_success_signal": sync_run.get("success_signal") is True,
        "async_success_signal": async_run.get("success_signal") is True,
        "sync_effectiveness_signal": sync_run.get("effectiveness_signal") is True,
        "async_effectiveness_signal": async_run.get("effectiveness_signal") is True,
        "sync_backend_used": backend_count(sync_run, "cuda_malloc") > 0,
        "async_backend_used": backend_count(async_run, "cuda_malloc_async") > 0,
        "sync_no_failed_requests": sync_run.get("failed_requests") == 0,
        "async_no_failed_requests": async_run.get("failed_requests") == 0,
        "sync_gap_policy_fail_zero": sync_run.get("gap_policy_fail") == 0,
        "async_gap_policy_fail_zero": async_run.get("gap_policy_fail") == 0,
        "async_gap_faults_not_higher_than_sync": (
            async_gap_faults is not None
            and sync_gap_faults is not None
            and async_gap_faults <= sync_gap_faults
        ),
        "async_tpot_not_more_than_10pct_worse_than_sync": (
            pct_delta(async_tpot, sync_tpot) is not None
            and pct_delta(async_tpot, sync_tpot) <= 0.10
        ),
        "async_throughput_not_more_than_10pct_worse_than_sync": (
            pct_delta(async_throughput, sync_throughput) is not None
            and pct_delta(async_throughput, sync_throughput) >= -0.10
        ),
    }

    return {
        "cuda_malloc": sync_run,
        "cuda_malloc_async": async_run,
        "backend_comparison": {
            "gap_fault_delta_async_minus_sync": numeric_delta(
                async_gap_faults, sync_gap_faults
            ),
            "gap_fault_delta_pct_async_vs_sync": pct_delta(
                async_gap_faults, sync_gap_faults
            ),
            "unknown_fault_delta_async_minus_sync": numeric_delta(
                async_unknown_faults, sync_unknown_faults
            ),
            "unknown_fault_delta_pct_async_vs_sync": pct_delta(
                async_unknown_faults, sync_unknown_faults
            ),
            "output_throughput_delta_pct_async_vs_sync": pct_delta(
                async_throughput, sync_throughput
            ),
            "mean_tpot_delta_pct_async_vs_sync": pct_delta(async_tpot, sync_tpot),
            "mean_ttft_delta_pct_async_vs_sync": pct_delta(
                async_run.get("mean_ttft_ms"), sync_run.get("mean_ttft_ms")
            ),
            "device_direct_actual_delta_async_minus_sync": numeric_delta(
                async_run.get("device_direct_actual_records"),
                sync_run.get("device_direct_actual_records"),
            ),
            "budget_reject_delta_async_minus_sync": numeric_delta(
                async_run.get("device_direct_budget_reject_records"),
                sync_run.get("device_direct_budget_reject_records"),
            ),
            "checks": checks,
            "correctness_signal": all(
                checks[key]
                for key in (
                    "sync_success_signal",
                    "async_success_signal",
                    "sync_backend_used",
                    "async_backend_used",
                    "sync_no_failed_requests",
                    "async_no_failed_requests",
                    "sync_gap_policy_fail_zero",
                    "async_gap_policy_fail_zero",
                )
            ),
            "async_effectiveness_signal": (
                checks["async_effectiveness_signal"]
                and checks["async_gap_faults_not_higher_than_sync"]
                and checks["async_tpot_not_more_than_10pct_worse_than_sync"]
                and checks["async_throughput_not_more_than_10pct_worse_than_sync"]
            ),
        },
    }


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:+.2f}%"


def main() -> None:
    args = parse_args()
    sync_report = load_json(args.sync_json)
    async_report = load_json(args.async_json)
    result = build_comparison(sync_report, async_report)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")

    cmp = result["backend_comparison"]
    sync_run = result["cuda_malloc"]
    async_run = result["cuda_malloc_async"]

    print("Stage C Backend Comparison")
    print(f"- output_json={output_path}")
    print(
        "- sync_backend_counts="
        f"{sync_run.get('device_direct_backend_counts')} "
        f"async_backend_counts={async_run.get('device_direct_backend_counts')}"
    )
    print(
        "- sync_gap_faults="
        f"{sync_run.get('gap_faults')} async_gap_faults={async_run.get('gap_faults')} "
        f"delta={cmp.get('gap_fault_delta_async_minus_sync')} "
        f"delta_pct={fmt_pct(cmp.get('gap_fault_delta_pct_async_vs_sync'))}"
    )
    print(
        "- sync_unknown_faults="
        f"{sync_run.get('unknown_faults')} "
        f"async_unknown_faults={async_run.get('unknown_faults')} "
        f"delta={cmp.get('unknown_fault_delta_async_minus_sync')} "
        f"delta_pct={fmt_pct(cmp.get('unknown_fault_delta_pct_async_vs_sync'))}"
    )
    print(
        "- sync_output_tok_s="
        f"{sync_run.get('output_throughput_tok_s')} "
        f"async_output_tok_s={async_run.get('output_throughput_tok_s')} "
        f"delta_pct={fmt_pct(cmp.get('output_throughput_delta_pct_async_vs_sync'))}"
    )
    print(
        "- sync_mean_tpot_ms="
        f"{sync_run.get('mean_tpot_ms')} "
        f"async_mean_tpot_ms={async_run.get('mean_tpot_ms')} "
        f"delta_pct={fmt_pct(cmp.get('mean_tpot_delta_pct_async_vs_sync'))}"
    )
    print(
        "- sync_actual_records="
        f"{sync_run.get('device_direct_actual_records')} "
        f"async_actual_records={async_run.get('device_direct_actual_records')} "
        f"delta={cmp.get('device_direct_actual_delta_async_minus_sync')}"
    )
    print(
        "- sync_budget_rejects="
        f"{sync_run.get('device_direct_budget_reject_records')} "
        f"async_budget_rejects={async_run.get('device_direct_budget_reject_records')} "
        f"delta={cmp.get('budget_reject_delta_async_minus_sync')}"
    )
    print(
        "- correctness_signal="
        f"{cmp.get('correctness_signal')} "
        f"async_effectiveness_signal={cmp.get('async_effectiveness_signal')}"
    )
    print("- checks:")
    checks = cmp.get("checks", {})
    if isinstance(checks, dict):
        for key, value in checks.items():
            print(f"  {key}={value}")


if __name__ == "__main__":
    main()

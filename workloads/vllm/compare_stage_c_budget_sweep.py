#!/usr/bin/env python3
"""Summarize C1 cuda_malloc budget sweep reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize multiple Stage C attention A/B reports by C1 budget."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="BUDGET_BYTES=REPORT_JSON",
        help="Budget and corresponding run_stage_c_attention_p20_ab report JSON.",
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def pct(value: Any) -> float | None:
    if value is None:
        return None
    return float(value) * 100.0


def fmt_pct(value: Any) -> str:
    value = pct(value)
    if value is None:
        return "n/a"
    return f"{value:+.2f}%"


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def compact_report(budget: int, path: Path, report: dict[str, Any]) -> dict[str, Any]:
    comparison = get_dict(report.get("comparison"))
    device = get_dict(report.get("stage_c_attention_device_direct"))
    trace = get_dict(report.get("trace_only"))
    metrics = get_dict(device.get("metrics"))
    bench = get_dict(device.get("bench_main"))
    gap = get_dict(device.get("main_gap"))
    trace_gap = get_dict(trace.get("main_gap"))
    trace_bench = get_dict(trace.get("bench_main"))

    return {
        "budget_bytes": budget,
        "report_json": str(path),
        "success_signal": comparison.get("success_signal"),
        "effectiveness_signal": comparison.get("effectiveness_signal"),
        "gap_fault_delta_pct_vs_trace": comparison.get("gap_fault_delta_pct"),
        "unknown_fault_delta_pct_vs_trace": comparison.get("unknown_fault_delta_pct"),
        "output_throughput_delta_pct_vs_trace": comparison.get(
            "output_throughput_delta_pct"
        ),
        "mean_tpot_delta_pct_vs_trace": comparison.get("mean_tpot_delta_pct"),
        "trace_gap_faults": trace_gap.get("faults"),
        "device_gap_faults": gap.get("faults"),
        "trace_output_tok_s": trace_bench.get("output_throughput_tok_s"),
        "device_output_tok_s": bench.get("output_throughput_tok_s"),
        "trace_mean_tpot_ms": trace_bench.get("mean_tpot_ms"),
        "device_mean_tpot_ms": bench.get("mean_tpot_ms"),
        "failed_requests": bench.get("failed_requests"),
        "device_direct_actual_records": metrics.get("device_direct_actual_records"),
        "device_direct_eligible_records": metrics.get("device_direct_eligible_records"),
        "device_direct_budget_reject_records": metrics.get(
            "device_direct_budget_reject_records"
        ),
        "device_direct_peak_live_bytes_observed": metrics.get(
            "device_direct_peak_live_bytes_observed"
        ),
        "device_direct_min_budget_remaining_observed": metrics.get(
            "device_direct_min_budget_remaining_observed"
        ),
        "device_direct_backend_counts": metrics.get("device_direct_backend_counts"),
        "placement_backend_counts": metrics.get("placement_backend_counts"),
        "gap_policy_fail": metrics.get("gap_policy_fail"),
        "median_lifetime_s": metrics.get("median_lifetime_s"),
        "phase_record_ratios": metrics.get("phase_record_ratios"),
        "checks": comparison.get("checks"),
    }


def score_run(run: dict[str, Any]) -> tuple[int, float, float, float]:
    success = 1 if run.get("success_signal") and run.get("effectiveness_signal") else 0
    throughput = float(run.get("device_output_tok_s") or 0.0)
    tpot = float(run.get("device_mean_tpot_ms") or 1.0e18)
    fault_reduction = -float(run.get("gap_fault_delta_pct_vs_trace") or 0.0)
    return success, throughput, fault_reduction, -tpot


def main() -> None:
    args = parse_args()
    runs: list[dict[str, Any]] = []

    for item in args.run:
        if "=" not in item:
            raise SystemExit(f"--run must be BUDGET_BYTES=REPORT_JSON: {item}")
        budget_text, path_text = item.split("=", 1)
        try:
            budget = int(budget_text)
        except ValueError as exc:
            raise SystemExit(f"invalid budget: {budget_text}") from exc
        path = Path(path_text)
        runs.append(compact_report(budget, path, load_json(path)))

    runs.sort(key=lambda run: run["budget_bytes"])
    best_run = max(runs, key=score_run) if runs else None
    result = {
        "runs": runs,
        "best_budget_bytes": best_run.get("budget_bytes") if best_run else None,
        "best_report_json": best_run.get("report_json") if best_run else None,
        "selection_rule": (
            "Prefer success/effectiveness, then higher output throughput, then "
            "larger gap fault reduction, then lower TPOT."
        ),
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("Stage C1 Budget Sweep Summary")
    print(f"- output_json={output_path}")
    print(f"- best_budget_bytes={result['best_budget_bytes']}")
    print("- runs:")
    for run in runs:
        print(
            "  "
            f"budget={run['budget_bytes']} "
            f"success={run.get('success_signal')} "
            f"effectiveness={run.get('effectiveness_signal')} "
            f"gap_delta={fmt_pct(run.get('gap_fault_delta_pct_vs_trace'))} "
            f"unknown_delta={fmt_pct(run.get('unknown_fault_delta_pct_vs_trace'))} "
            f"output_tok_s={run.get('device_output_tok_s')} "
            f"tpot_ms={run.get('device_mean_tpot_ms')} "
            f"actual={run.get('device_direct_actual_records')} "
            f"rejects={run.get('device_direct_budget_reject_records')} "
            f"peak_live={run.get('device_direct_peak_live_bytes_observed')}"
        )


if __name__ == "__main__":
    main()

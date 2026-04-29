#!/usr/bin/env python3
"""Run or validate a Stage E weights budget telemetry success check."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether Stage E model-weights independent budget telemetry "
            "is wired. By default this starts vLLM until model load/KV init has "
            "completed, then validates allocator-side weight budget metrics."
        )
    )
    parser.add_argument("--metrics-json", help="Existing summarize_gap_watch_metrics JSON.")
    parser.add_argument("--allocator-log", help="Existing or new allocator trace log.")
    parser.add_argument("--weight-map-jsonl", help="Existing or new Stage E weight map JSONL.")
    parser.add_argument("--moe-routing-jsonl", help="Existing or new Stage E MoE routing JSONL.")
    parser.add_argument("--weight-map-summary-json", help="Existing or new weight map summary JSON.")
    parser.add_argument("--run-dir", help="Existing or new Stage E run directory.")
    parser.add_argument("--output-json", help="Where to write the check summary JSON.")
    parser.add_argument("--budget-bytes", type=int, default=1048576)
    parser.add_argument(
        "--budget-mode",
        choices=("trace_only", "enforce"),
        default="trace_only",
    )
    parser.add_argument(
        "--kv-budget-bytes",
        type=int,
        default=0,
        help="Keep Stage D KV budget independent from this Stage E check.",
    )
    parser.add_argument(
        "--kv-budget-mode",
        choices=("trace_only", "enforce"),
        default="trace_only",
    )
    parser.add_argument(
        "--run-bench",
        action="store_true",
        help="Run the benchmark after server startup. Default stops after ready.",
    )
    parser.add_argument(
        "--enable-moe-routing-trace",
        action="store_true",
        help="Enable Stage E MoE routing trace while running the benchmark.",
    )
    parser.add_argument(
        "--require-moe-routing-trace",
        action="store_true",
        help="Fail if the MoE routing trace has no records.",
    )
    parser.add_argument("--prompts", type=int, default=1)
    parser.add_argument("--request-rate", default="5")
    parser.add_argument("--output-len", default="512")
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Do not start a new vLLM run; requires --metrics-json or --allocator-log.",
    )
    return parser.parse_args()


def default_run_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return Path(f"/tmp/vllm_stage_e_success_check_{stamp}")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def summarize_allocator_log(allocator_log: Path, metrics_json: Path) -> None:
    subprocess.run(
        [
            "python3",
            str(SCRIPT_DIR / "summarize_gap_watch_metrics.py"),
            "--allocator-log",
            str(allocator_log),
            "--summary-json",
            str(metrics_json),
        ],
        cwd=SCRIPT_DIR,
        check=True,
    )


def summarize_weight_map(
    weight_map_jsonl: Path,
    moe_routing_jsonl: Path | None,
    summary_json: Path,
) -> None:
    cmd = [
        "python3",
        str(SCRIPT_DIR / "summarize_stage_e_weight_map.py"),
        "--weight-map",
        str(weight_map_jsonl),
        "--summary-json",
        str(summary_json),
    ]
    if moe_routing_jsonl is not None:
        cmd.extend(["--moe-routing-trace", str(moe_routing_jsonl)])
    subprocess.run(cmd, cwd=SCRIPT_DIR, check=True)


def resolve_existing_inputs(
    args: argparse.Namespace,
) -> tuple[Path | None, Path | None, Path | None, Path | None, Path | None]:
    metrics_json = Path(args.metrics_json) if args.metrics_json else None
    allocator_log = Path(args.allocator_log) if args.allocator_log else None
    weight_map_jsonl = Path(args.weight_map_jsonl) if args.weight_map_jsonl else None
    moe_routing_jsonl = Path(args.moe_routing_jsonl) if args.moe_routing_jsonl else None
    weight_map_summary_json = (
        Path(args.weight_map_summary_json)
        if args.weight_map_summary_json
        else None
    )
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if metrics_json is None:
            candidate = run_dir / "vllm_stage_e_weight_budget_metrics.json"
            if candidate.is_file():
                metrics_json = candidate
        if allocator_log is None:
            candidate = run_dir / "vllm_uvm_allocator_trace_stage_e.log"
            if candidate.is_file():
                allocator_log = candidate
        if weight_map_jsonl is None:
            candidate = run_dir / "vllm_uvm_weight_regions_stage_e.jsonl"
            if candidate.is_file():
                weight_map_jsonl = candidate
        if moe_routing_jsonl is None:
            candidate = run_dir / "vllm_uvm_moe_routing_stage_e.jsonl"
            if candidate.is_file():
                moe_routing_jsonl = candidate
        if weight_map_summary_json is None:
            candidate = run_dir / "vllm_stage_e_weight_map_summary.json"
            if candidate.is_file():
                weight_map_summary_json = candidate
    return (
        metrics_json,
        allocator_log,
        weight_map_jsonl,
        moe_routing_jsonl,
        weight_map_summary_json,
    )


def run_experiment(
    args: argparse.Namespace, run_dir: Path
) -> tuple[Path, Path, Path, Path, Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    allocator_log = Path(args.allocator_log) if args.allocator_log else (
        run_dir / "vllm_uvm_allocator_trace_stage_e.log"
    )
    metrics_json = Path(args.metrics_json) if args.metrics_json else (
        run_dir / "vllm_stage_e_weight_budget_metrics.json"
    )
    weight_map_jsonl = Path(args.weight_map_jsonl) if args.weight_map_jsonl else (
        run_dir / "vllm_uvm_weight_regions_stage_e.jsonl"
    )
    moe_routing_jsonl = Path(args.moe_routing_jsonl) if args.moe_routing_jsonl else (
        run_dir / "vllm_uvm_moe_routing_stage_e.jsonl"
    )
    weight_map_summary_json = (
        Path(args.weight_map_summary_json)
        if args.weight_map_summary_json
        else run_dir / "vllm_stage_e_weight_map_summary.json"
    )
    runner_log = run_dir / "stage_e_success_check_runner.log"

    cmd = [
        "./run_kv_fault_ratio.sh",
        "--mode",
        "trace",
        "--allocator-log",
        str(allocator_log),
        "--trace-log",
        str(run_dir / "uvm_kv_fault_stats_stage_e.log"),
        "--address-log",
        str(run_dir / "vllm_uvm_address_regions_stage_e.log"),
        "--server-log",
        str(run_dir / "vllm_server_stage_e.log"),
        "--bench-log",
        str(run_dir / "vllm_bench_stage_e.log"),
        "--uvm-kv-budget-bytes",
        str(args.kv_budget_bytes),
        "--uvm-kv-budget-mode",
        args.kv_budget_mode,
        "--uvm-weight-budget-bytes",
        str(args.budget_bytes),
        "--uvm-weight-budget-mode",
        args.budget_mode,
        "--uvm-weight-map-enable",
        "1",
        "--uvm-weight-map-file",
        str(weight_map_jsonl),
        "--uvm-moe-routing-trace-enable",
        "1" if args.enable_moe_routing_trace else "0",
        "--uvm-moe-routing-trace-file",
        str(moe_routing_jsonl),
        "--gap-watch-metrics-summary-json",
        str(metrics_json),
        "--prompts",
        str(args.prompts),
        "--request-rate",
        str(args.request_rate),
        "--output-len",
        str(args.output_len),
    ]
    if not args.run_bench:
        cmd.append("--no-bench")

    print("===========================================================")
    print(" Stage E Success Check: running weight budget telemetry probe")
    print(f" Output dir: {run_dir}")
    print(f" Weight budget bytes: {args.budget_bytes}")
    print(f" Weight budget mode: {args.budget_mode}")
    print(f" KV budget bytes: {args.kv_budget_bytes}")
    print(f" Benchmark enabled: {args.run_bench}")
    print(f" Weight map: {weight_map_jsonl}")
    print(f" MoE routing trace enabled: {args.enable_moe_routing_trace}")
    print("===========================================================")

    with runner_log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            cwd=SCRIPT_DIR,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        rc = process.wait()

    if rc != 0:
        raise SystemExit(f"Stage E probe failed with exit code {rc}; log={runner_log}")
    if not metrics_json.is_file():
        if allocator_log.is_file():
            summarize_allocator_log(allocator_log, metrics_json)
        else:
            raise SystemExit(f"Stage E metrics were not produced: {metrics_json}")
    if weight_map_jsonl.is_file():
        summarize_weight_map(
            weight_map_jsonl,
            moe_routing_jsonl if moe_routing_jsonl.is_file() else None,
            weight_map_summary_json,
        )
    return (
        metrics_json,
        allocator_log,
        weight_map_jsonl,
        moe_routing_jsonl,
        weight_map_summary_json,
        runner_log,
    )


def validate_metrics(
    *,
    metrics_json: Path,
    allocator_log: Path | None,
    weight_map_summary_json: Path | None,
    runner_log: Path | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    metrics = load_json(metrics_json)
    weight_trace_allocs = as_int(metrics.get("weight_trace_allocations"))
    weight_budget_bytes = as_int(metrics.get("weight_budget_bytes"))
    weight_peak_live = as_int(metrics.get("weight_peak_live_bytes_observed"))
    weight_live = as_int(metrics.get("weight_live_bytes"))
    weight_over = as_int(metrics.get("weight_budget_over_records"))
    weight_rejects = as_int(metrics.get("weight_budget_reject_records"))
    weight_mode = metrics.get("weight_budget_mode")
    weight_map_summary = (
        load_json(weight_map_summary_json)
        if weight_map_summary_json is not None and weight_map_summary_json.is_file()
        else {}
    )
    weight_map_records = as_int(weight_map_summary.get("weight_map_records"))
    weight_map_moe_expert_records = as_int(
        weight_map_summary.get("weight_map_moe_expert_records")
    )
    moe_routing_records = as_int(weight_map_summary.get("moe_routing_records"))

    expected_over_budget = (
        args.budget_bytes > 0
        and weight_peak_live is not None
        and weight_peak_live > args.budget_bytes
    )

    checks = [
        check(
            "metrics_json_present",
            metrics_json.is_file(),
            f"metrics_json={metrics_json}",
        ),
        check(
            "weight_trace_allocations_present",
            weight_trace_allocs is not None and weight_trace_allocs > 0,
            f"weight_trace_allocations={weight_trace_allocs}",
        ),
        check(
            "weight_peak_live_positive",
            weight_peak_live is not None and weight_peak_live > 0,
            f"weight_peak_live_bytes_observed={weight_peak_live}",
        ),
        check(
            "weight_live_non_negative",
            weight_live is None or weight_live >= 0,
            f"weight_live_bytes={weight_live}",
        ),
        check(
            "weight_budget_bytes_matches",
            weight_budget_bytes == args.budget_bytes,
            f"weight_budget_bytes={weight_budget_bytes} expected={args.budget_bytes}",
        ),
        check(
            "weight_budget_mode_matches",
            weight_mode == args.budget_mode,
            f"weight_budget_mode={weight_mode} expected={args.budget_mode}",
        ),
        check(
            "weight_over_budget_signal_if_expected",
            not expected_over_budget or (weight_over is not None and weight_over > 0),
            (
                f"expected_over_budget={expected_over_budget} "
                f"weight_budget_over_records={weight_over}"
            ),
        ),
        check(
            "trace_only_has_no_soft_rejects",
            args.budget_mode != "trace_only" or weight_rejects == 0,
            f"weight_budget_reject_records={weight_rejects}",
        ),
        check(
            "enforce_emits_soft_rejects_if_over_budget",
            (
                args.budget_mode != "enforce"
                or not expected_over_budget
                or (weight_rejects is not None and weight_rejects > 0)
            ),
            (
                f"expected_over_budget={expected_over_budget} "
                f"weight_budget_reject_records={weight_rejects}"
            ),
        ),
        check(
            "weight_map_records_present",
            weight_map_records is not None and weight_map_records > 0,
            f"weight_map_records={weight_map_records}",
        ),
        check(
            "weight_map_has_moe_expert_signal",
            weight_map_moe_expert_records is not None,
            f"weight_map_moe_expert_records={weight_map_moe_expert_records}",
        ),
        check(
            "moe_routing_trace_present_if_required",
            (
                not args.require_moe_routing_trace
                or (moe_routing_records is not None and moe_routing_records > 0)
            ),
            f"moe_routing_records={moe_routing_records}",
        ),
    ]

    runner_log_flags = {
        "contains_parse_failure": False,
        "contains_server_exit": False,
    }
    if runner_log and runner_log.is_file():
        text = runner_log.read_text(encoding="utf-8", errors="replace")
        runner_log_flags["contains_parse_failure"] = "could not parse" in text.lower()
        runner_log_flags["contains_server_exit"] = "Server exited early" in text
        checks.append(
            check(
                "runner_log_clean",
                not runner_log_flags["contains_parse_failure"]
                and not runner_log_flags["contains_server_exit"],
                str(runner_log_flags),
            )
        )

    passed = all(item["passed"] for item in checks)
    return {
        "passed": passed,
        "metrics_json": str(metrics_json),
        "allocator_log": str(allocator_log) if allocator_log else None,
        "weight_map_summary_json": (
            str(weight_map_summary_json) if weight_map_summary_json else None
        ),
        "weight_budget_bytes": weight_budget_bytes,
        "weight_budget_mode": weight_mode,
        "weight_trace_allocations": weight_trace_allocs,
        "weight_live_bytes": weight_live,
        "weight_peak_live_bytes_observed": weight_peak_live,
        "weight_budget_over_records": weight_over,
        "weight_budget_reject_records": weight_rejects,
        "weight_budget_reason_counts": metrics.get("weight_budget_reason_counts"),
        "weight_map_records": weight_map_records,
        "weight_map_total_bytes": weight_map_summary.get("weight_map_total_bytes"),
        "weight_map_moe_expert_records": weight_map_moe_expert_records,
        "weight_map_moe_expert_bytes": weight_map_summary.get(
            "weight_map_moe_expert_bytes"
        ),
        "weight_map_role_counts": weight_map_summary.get("weight_map_role_counts"),
        "moe_routing_records": moe_routing_records,
        "moe_routing_total_tokens": weight_map_summary.get(
            "moe_routing_total_tokens"
        ),
        "moe_routing_top_experts": weight_map_summary.get(
            "moe_routing_top_experts"
        ),
        "expected_over_budget": expected_over_budget,
        "checks": checks,
    }


def print_summary(summary: dict[str, Any], output_json: Path) -> None:
    status = "PASS" if summary["passed"] else "FAIL"
    print("===========================================================")
    print(f" Stage E Success Check: {status}")
    print(f" Metrics: {summary['metrics_json']}")
    print(f" Allocator log: {summary['allocator_log']}")
    print(f" Weight map summary: {summary['weight_map_summary_json']}")
    print("===========================================================")
    print(
        "- weight_budget: "
        f"bytes={summary.get('weight_budget_bytes')} "
        f"mode={summary.get('weight_budget_mode')}"
    )
    print(
        "- weight_live: "
        f"trace_allocs={summary.get('weight_trace_allocations')} "
        f"live={summary.get('weight_live_bytes')} "
        f"peak={summary.get('weight_peak_live_bytes_observed')}"
    )
    print(
        "- weight_budget_signals: "
        f"over={summary.get('weight_budget_over_records')} "
        f"rejects={summary.get('weight_budget_reject_records')} "
        f"expected_over_budget={summary.get('expected_over_budget')}"
    )
    print(f"- weight_budget_reason_counts={summary.get('weight_budget_reason_counts')}")
    print(
        "- weight_map: "
        f"records={summary.get('weight_map_records')} "
        f"bytes={summary.get('weight_map_total_bytes')} "
        f"moe_expert_records={summary.get('weight_map_moe_expert_records')}"
    )
    print(f"- weight_map_role_counts={summary.get('weight_map_role_counts')}")
    print(
        "- moe_routing: "
        f"records={summary.get('moe_routing_records')} "
        f"tokens={summary.get('moe_routing_total_tokens')} "
        f"top_experts={summary.get('moe_routing_top_experts')}"
    )
    print("- checks:")
    for item in summary["checks"]:
        print(f"  {item['name']}={item['passed']} ({item['detail']})")
    print(f"- check_json={output_json}")


def main() -> int:
    args = parse_args()
    if args.budget_bytes < 0:
        raise SystemExit("--budget-bytes must be non-negative")
    if args.kv_budget_bytes < 0:
        raise SystemExit("--kv-budget-bytes must be non-negative")

    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir()
    (
        metrics_json,
        allocator_log,
        weight_map_jsonl,
        moe_routing_jsonl,
        weight_map_summary_json,
    ) = resolve_existing_inputs(args)
    runner_log: Path | None = None

    if not args.skip_run:
        (
            metrics_json,
            allocator_log,
            weight_map_jsonl,
            moe_routing_jsonl,
            weight_map_summary_json,
            runner_log,
        ) = run_experiment(args, run_dir)
    elif metrics_json is None:
        if allocator_log is None:
            raise SystemExit("--skip-run requires --metrics-json or --allocator-log")
        metrics_json = run_dir / "vllm_stage_e_weight_budget_metrics.json"
        run_dir.mkdir(parents=True, exist_ok=True)
        summarize_allocator_log(allocator_log, metrics_json)
    if weight_map_summary_json is None and weight_map_jsonl is not None:
        weight_map_summary_json = run_dir / "vllm_stage_e_weight_map_summary.json"
        run_dir.mkdir(parents=True, exist_ok=True)
        summarize_weight_map(
            weight_map_jsonl,
            moe_routing_jsonl if moe_routing_jsonl and moe_routing_jsonl.is_file() else None,
            weight_map_summary_json,
        )

    assert metrics_json is not None
    summary = validate_metrics(
        metrics_json=metrics_json,
        allocator_log=allocator_log,
        weight_map_summary_json=weight_map_summary_json,
        runner_log=runner_log,
        args=args,
    )
    output_json = Path(args.output_json) if args.output_json else (
        run_dir / "stage_e_success_check.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print_summary(summary, output_json)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

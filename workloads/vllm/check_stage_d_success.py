#!/usr/bin/env python3
"""Run or validate a Stage D KV budget telemetry success check."""

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
            "Check whether Stage D KV independent budget telemetry is wired. "
            "By default this starts vLLM only until KV cache initialization, "
            "then validates allocator-side KV budget metrics. Use --metrics-json "
            "or --allocator-log with --skip-run for offline validation."
        )
    )
    parser.add_argument("--metrics-json", help="Existing summarize_gap_watch_metrics JSON.")
    parser.add_argument("--allocator-log", help="Existing or new allocator trace log.")
    parser.add_argument("--run-dir", help="Existing or new Stage D run directory.")
    parser.add_argument("--output-json", help="Where to write the check summary JSON.")
    parser.add_argument("--budget-bytes", type=int, default=1048576)
    parser.add_argument(
        "--budget-mode",
        choices=("trace_only", "enforce"),
        default="trace_only",
    )
    parser.add_argument(
        "--run-bench",
        action="store_true",
        help="Run the benchmark after KV initialization. Default stops after server ready.",
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
    return Path(f"/tmp/vllm_stage_d_success_check_{stamp}")


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
    cmd = [
        "python3",
        str(SCRIPT_DIR / "summarize_gap_watch_metrics.py"),
        "--allocator-log",
        str(allocator_log),
        "--summary-json",
        str(metrics_json),
    ]
    subprocess.run(cmd, cwd=SCRIPT_DIR, check=True)


def resolve_existing_inputs(args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    metrics_json = Path(args.metrics_json) if args.metrics_json else None
    allocator_log = Path(args.allocator_log) if args.allocator_log else None
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if metrics_json is None:
            candidate = run_dir / "vllm_stage_d_kv_budget_metrics.json"
            if candidate.is_file():
                metrics_json = candidate
        if allocator_log is None:
            candidate = run_dir / "vllm_uvm_allocator_trace_stage_d.log"
            if candidate.is_file():
                allocator_log = candidate
    return metrics_json, allocator_log


def run_experiment(args: argparse.Namespace, run_dir: Path) -> tuple[Path, Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    allocator_log = Path(args.allocator_log) if args.allocator_log else (
        run_dir / "vllm_uvm_allocator_trace_stage_d.log"
    )
    metrics_json = Path(args.metrics_json) if args.metrics_json else (
        run_dir / "vllm_stage_d_kv_budget_metrics.json"
    )
    runner_log = run_dir / "stage_d_success_check_runner.log"

    cmd = [
        "./run_kv_fault_ratio.sh",
        "--mode",
        "trace",
        "--allocator-log",
        str(allocator_log),
        "--trace-log",
        str(run_dir / "uvm_kv_fault_stats_stage_d.log"),
        "--address-log",
        str(run_dir / "vllm_uvm_address_regions_stage_d.log"),
        "--server-log",
        str(run_dir / "vllm_server_stage_d.log"),
        "--bench-log",
        str(run_dir / "vllm_bench_stage_d.log"),
        "--uvm-kv-budget-bytes",
        str(args.budget_bytes),
        "--uvm-kv-budget-mode",
        args.budget_mode,
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
    print(" Stage D Success Check: running KV budget telemetry probe")
    print(f" Output dir: {run_dir}")
    print(f" KV budget bytes: {args.budget_bytes}")
    print(f" KV budget mode: {args.budget_mode}")
    print(f" Benchmark enabled: {args.run_bench}")
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
        raise SystemExit(f"Stage D probe failed with exit code {rc}; log={runner_log}")
    if not metrics_json.is_file():
        if allocator_log.is_file():
            summarize_allocator_log(allocator_log, metrics_json)
        else:
            raise SystemExit(f"Stage D metrics were not produced: {metrics_json}")
    return metrics_json, allocator_log, runner_log


def validate_metrics(
    *,
    metrics_json: Path,
    allocator_log: Path | None,
    runner_log: Path | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    metrics = load_json(metrics_json)
    kv_trace_allocs = as_int(metrics.get("kv_trace_allocations"))
    kv_budget_bytes = as_int(metrics.get("kv_budget_bytes"))
    kv_peak_live = as_int(metrics.get("kv_peak_live_bytes_observed"))
    kv_live = as_int(metrics.get("kv_live_bytes"))
    kv_over = as_int(metrics.get("kv_budget_over_records"))
    kv_rejects = as_int(metrics.get("kv_budget_reject_records"))
    kv_mode = metrics.get("kv_budget_mode")

    expected_over_budget = (
        args.budget_bytes > 0
        and kv_peak_live is not None
        and kv_peak_live > args.budget_bytes
    )

    checks = [
        check(
            "metrics_json_present",
            metrics_json.is_file(),
            f"metrics_json={metrics_json}",
        ),
        check(
            "kv_trace_allocations_present",
            kv_trace_allocs is not None and kv_trace_allocs > 0,
            f"kv_trace_allocations={kv_trace_allocs}",
        ),
        check(
            "kv_peak_live_positive",
            kv_peak_live is not None and kv_peak_live > 0,
            f"kv_peak_live_bytes_observed={kv_peak_live}",
        ),
        check(
            "kv_live_non_negative",
            kv_live is None or kv_live >= 0,
            f"kv_live_bytes={kv_live}",
        ),
        check(
            "kv_budget_bytes_matches",
            kv_budget_bytes == args.budget_bytes,
            f"kv_budget_bytes={kv_budget_bytes} expected={args.budget_bytes}",
        ),
        check(
            "kv_budget_mode_matches",
            kv_mode == args.budget_mode,
            f"kv_budget_mode={kv_mode} expected={args.budget_mode}",
        ),
        check(
            "kv_over_budget_signal_if_expected",
            not expected_over_budget or (kv_over is not None and kv_over > 0),
            f"expected_over_budget={expected_over_budget} kv_budget_over_records={kv_over}",
        ),
        check(
            "enforce_peak_live_within_budget",
            args.budget_mode != "enforce"
            or args.budget_bytes == 0
            or (kv_peak_live is not None and kv_peak_live <= args.budget_bytes),
            (
                "kv_peak_live_bytes_observed="
                f"{kv_peak_live} budget_bytes={args.budget_bytes}"
            ),
        ),
        check(
            "trace_only_has_no_soft_rejects",
            args.budget_mode != "trace_only" or kv_rejects in (0, None),
            f"kv_budget_reject_records={kv_rejects}",
        ),
        check(
            "enforce_emits_soft_rejects_if_over_budget",
            args.budget_mode != "enforce"
            or not expected_over_budget
            or (kv_rejects is not None and kv_rejects > 0),
            f"expected_over_budget={expected_over_budget} kv_budget_reject_records={kv_rejects}",
        ),
    ]

    runner_log_check = None
    if runner_log and runner_log.is_file():
        text = runner_log.read_text(encoding="utf-8", errors="replace")
        runner_log_check = {
            "contains_parse_failure": "Failed to parse stats lines" in text,
            "contains_server_exit": "Server exited before ready" in text,
        }
        checks.append(
            check(
                "runner_log_clean",
                not runner_log_check["contains_parse_failure"]
                and not runner_log_check["contains_server_exit"],
                str(runner_log_check),
            )
        )

    passed = all(item["passed"] for item in checks)
    return {
        "passed": passed,
        "metrics_json": str(metrics_json),
        "allocator_log": str(allocator_log) if allocator_log else None,
        "runner_log": str(runner_log) if runner_log else None,
        "summary": {
            "kv_budget_bytes": kv_budget_bytes,
            "kv_budget_mode": kv_mode,
            "kv_trace_allocations": kv_trace_allocs,
            "kv_live_bytes": kv_live,
            "kv_peak_live_bytes_observed": kv_peak_live,
            "kv_budget_over_records": kv_over,
            "kv_budget_reject_records": kv_rejects,
            "kv_budget_reason_counts": metrics.get("kv_budget_reason_counts"),
            "expected_over_budget": expected_over_budget,
        },
        "checks": checks,
        "runner_log_check": runner_log_check,
    }


def print_result(result: dict[str, Any]) -> None:
    status = "PASS" if result["passed"] else "FAIL"
    summary = result["summary"]
    print("===========================================================")
    print(f" Stage D Success Check: {status}")
    print(f" Metrics: {result['metrics_json']}")
    if result.get("allocator_log"):
        print(f" Allocator log: {result['allocator_log']}")
    print("===========================================================")
    print(
        "- kv_budget: "
        f"bytes={summary.get('kv_budget_bytes')} "
        f"mode={summary.get('kv_budget_mode')}"
    )
    print(
        "- kv_live: "
        f"trace_allocs={summary.get('kv_trace_allocations')} "
        f"live={summary.get('kv_live_bytes')} "
        f"peak={summary.get('kv_peak_live_bytes_observed')}"
    )
    print(
        "- kv_budget_signals: "
        f"over={summary.get('kv_budget_over_records')} "
        f"rejects={summary.get('kv_budget_reject_records')} "
        f"expected_over_budget={summary.get('expected_over_budget')}"
    )
    print(f"- kv_budget_reason_counts={summary.get('kv_budget_reason_counts')}")
    print("- checks:")
    for item in result["checks"]:
        print(f"  {item['name']}={item['passed']} ({item['detail']})")


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir()
    metrics_json, allocator_log = resolve_existing_inputs(args)
    runner_log: Path | None = None

    if args.skip_run:
        if metrics_json is None and allocator_log is None:
            raise SystemExit("--skip-run requires --metrics-json or --allocator-log")
        if metrics_json is None:
            assert allocator_log is not None
            metrics_json = run_dir / "vllm_stage_d_kv_budget_metrics.json"
            summarize_allocator_log(allocator_log, metrics_json)
    else:
        metrics_json, allocator_log, runner_log = run_experiment(args, run_dir)

    assert metrics_json is not None
    result = validate_metrics(
        metrics_json=metrics_json,
        allocator_log=allocator_log,
        runner_log=runner_log,
        args=args,
    )

    output_json = Path(args.output_json) if args.output_json else (
        run_dir / "stage_d_success_check.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    result["output_json"] = str(output_json)

    print_result(result)
    print(f"- check_json={output_json}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

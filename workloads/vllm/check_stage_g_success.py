#!/usr/bin/env python3
"""Run or validate a Stage G runtime scratch pool admission check."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether Stage G runtime scratch pool admission control is "
            "wired. The check enables scratch-pool device-direct fast path with "
            "managed fallback and validates allocator telemetry."
        )
    )
    parser.add_argument("--metrics-json", help="Existing summarize_gap_watch_metrics JSON.")
    parser.add_argument("--allocator-log", help="Existing or new allocator trace log.")
    parser.add_argument("--bench-log", help="Existing or new benchmark log.")
    parser.add_argument("--run-dir", help="Existing or new Stage G run directory.")
    parser.add_argument("--output-json", help="Where to write the check summary JSON.")
    parser.add_argument("--budget-bytes", type=int, default=1048576)
    parser.add_argument(
        "--mode",
        choices=("trace_only", "enforce"),
        default="enforce",
        help="Scratch pool mode. enforce should show fallback if pressure exceeds budget.",
    )
    parser.add_argument(
        "--target-phases",
        default="enabled:attention,enabled:moe,enabled:model_forward",
        help="Comma-separated phase prefixes for scratch pool admission.",
    )
    parser.add_argument(
        "--backend",
        choices=("cuda_malloc", "cuda_malloc_async"),
        default="cuda_malloc_async",
    )
    parser.add_argument(
        "--min-bytes",
        type=int,
        default=4096,
        help="Minimum allocation size allowed for Stage G device-direct admission.",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=16777216,
        help=(
            "Maximum allocation size allowed for Stage G device-direct admission. "
            "Default matches the RuntimeScratch classifier upper bound."
        ),
    )
    parser.add_argument("--pool-release-threshold", type=int, default=1048576)
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
    return Path(f"/tmp/vllm_stage_g_success_check_{stamp}")


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


def parse_failed_requests(bench_log: Path | None) -> int | None:
    if bench_log is None or not bench_log.is_file():
        return None
    text = bench_log.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Failed requests:\s+(\d+)", text)
    if not matches:
        return None
    return int(matches[-1])


def resolve_existing_inputs(
    args: argparse.Namespace,
) -> tuple[Path | None, Path | None, Path | None]:
    metrics_json = Path(args.metrics_json) if args.metrics_json else None
    allocator_log = Path(args.allocator_log) if args.allocator_log else None
    bench_log = Path(args.bench_log) if args.bench_log else None
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if metrics_json is None:
            candidate = run_dir / "vllm_stage_g_scratch_pool_metrics.json"
            if candidate.is_file():
                metrics_json = candidate
        if allocator_log is None:
            candidate = run_dir / "vllm_uvm_allocator_trace_stage_g.log"
            if candidate.is_file():
                allocator_log = candidate
        if bench_log is None:
            candidate = run_dir / "vllm_bench_stage_g.log"
            if candidate.is_file():
                bench_log = candidate
    return metrics_json, allocator_log, bench_log


def run_experiment(
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    allocator_log = Path(args.allocator_log) if args.allocator_log else (
        run_dir / "vllm_uvm_allocator_trace_stage_g.log"
    )
    metrics_json = Path(args.metrics_json) if args.metrics_json else (
        run_dir / "vllm_stage_g_scratch_pool_metrics.json"
    )
    bench_log = Path(args.bench_log) if args.bench_log else (
        run_dir / "vllm_bench_stage_g.log"
    )
    runner_log = run_dir / "stage_g_success_check_runner.log"

    cmd = [
        "./run_kv_fault_ratio.sh",
        "--mode",
        "trace",
        "--allocator-log",
        str(allocator_log),
        "--trace-log",
        str(run_dir / "uvm_kv_fault_stats_stage_g.log"),
        "--address-log",
        str(run_dir / "vllm_uvm_address_regions_stage_g.log"),
        "--server-log",
        str(run_dir / "vllm_server_stage_g.log"),
        "--bench-log",
        str(bench_log),
        "--uvm-kv-budget-bytes",
        "0",
        "--uvm-weight-budget-bytes",
        "0",
        "--uvm-pool-registry-enable",
        "1",
        "--uvm-scratch-pool-enable",
        "1",
        "--uvm-scratch-pool-budget-bytes",
        str(args.budget_bytes),
        "--uvm-scratch-pool-mode",
        args.mode,
        "--uvm-scratch-pool-target-phases",
        args.target_phases,
        "--uvm-device-direct-enable",
        "1",
        "--uvm-device-direct-backend",
        args.backend,
        "--uvm-device-direct-min-bytes",
        str(args.min_bytes),
        "--uvm-device-direct-max-bytes",
        str(args.max_bytes),
        "--uvm-device-direct-max-total-bytes",
        str(max(args.budget_bytes, 1048576) if args.budget_bytes > 0 else 0),
        "--uvm-device-direct-target-phases",
        args.target_phases,
        "--uvm-device-direct-pool-release-threshold",
        str(args.pool_release_threshold),
        "--gap-watch-metrics-summary-json",
        str(metrics_json),
        "--prompts",
        str(args.prompts),
        "--request-rate",
        str(args.request_rate),
        "--output-len",
        str(args.output_len),
    ]

    print("===========================================================")
    print(" Stage G Success Check: running scratch pool admission probe")
    print(f" Output dir: {run_dir}")
    print(f" Scratch budget bytes: {args.budget_bytes}")
    print(f" Scratch mode: {args.mode}")
    print(f" Scratch target phases: {args.target_phases}")
    print(f" Device-direct backend: {args.backend}")
    print(f" Device-direct min bytes: {args.min_bytes}")
    print(f" Device-direct max bytes: {args.max_bytes}")
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
        raise SystemExit(f"Stage G probe failed with exit code {rc}; log={runner_log}")
    if not metrics_json.is_file():
        if allocator_log.is_file():
            summarize_allocator_log(allocator_log, metrics_json)
        else:
            raise SystemExit(f"Stage G metrics were not produced: {metrics_json}")
    return metrics_json, allocator_log, bench_log, runner_log


def validate_metrics(
    *,
    metrics_json: Path,
    allocator_log: Path | None,
    bench_log: Path | None,
    runner_log: Path | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    metrics = load_json(metrics_json)
    scratch_enabled = metrics.get("scratch_pool_enabled")
    scratch_budget = as_int(metrics.get("scratch_pool_budget_bytes"))
    scratch_mode = metrics.get("scratch_pool_mode")
    scratch_trace = as_int(metrics.get("scratch_pool_trace_records"))
    scratch_eligible = as_int(metrics.get("scratch_pool_eligible_records"))
    scratch_direct = as_int(metrics.get("scratch_pool_device_direct_records"))
    scratch_peak = as_int(metrics.get("scratch_pool_peak_live_bytes"))
    scratch_live = as_int(metrics.get("scratch_pool_live_bytes"))
    scratch_over = as_int(metrics.get("scratch_pool_budget_over_records"))
    scratch_rejects = as_int(metrics.get("scratch_pool_budget_reject_records"))
    failed_requests = parse_failed_requests(bench_log)
    reason_counts = metrics.get("scratch_pool_reason_counts") or {}

    checks = [
        check("metrics_json_present", metrics_json.is_file(), f"metrics_json={metrics_json}"),
        check(
            "scratch_pool_enabled",
            scratch_enabled is True,
            f"scratch_pool_enabled={scratch_enabled}",
        ),
        check(
            "scratch_pool_budget_matches",
            scratch_budget == args.budget_bytes,
            f"scratch_pool_budget_bytes={scratch_budget} expected={args.budget_bytes}",
        ),
        check(
            "scratch_pool_mode_matches",
            scratch_mode == args.mode,
            f"scratch_pool_mode={scratch_mode} expected={args.mode}",
        ),
        check(
            "scratch_pool_trace_records_present",
            scratch_trace is not None and scratch_trace > 0,
            f"scratch_pool_trace_records={scratch_trace}",
        ),
        check(
            "scratch_pool_eligible_records_present",
            scratch_eligible is not None and scratch_eligible > 0,
            f"scratch_pool_eligible_records={scratch_eligible}",
        ),
        check(
            "scratch_pool_device_direct_records_present",
            scratch_direct is not None and scratch_direct > 0,
            f"scratch_pool_device_direct_records={scratch_direct}",
        ),
        check(
            "scratch_pool_peak_within_budget_in_enforce",
            (
                args.mode != "enforce"
                or args.budget_bytes == 0
                or (scratch_peak is not None and scratch_peak <= args.budget_bytes)
            ),
            f"scratch_pool_peak_live_bytes={scratch_peak} budget={args.budget_bytes}",
        ),
        check(
            "scratch_pool_fallback_signal_present_if_over_budget",
            (
                args.mode != "enforce"
                or scratch_over is None
                or scratch_over == 0
                or (scratch_rejects is not None and scratch_rejects > 0)
            ),
            (
                f"scratch_pool_budget_over_records={scratch_over} "
                f"scratch_pool_budget_reject_records={scratch_rejects}"
            ),
        ),
        check(
            "benchmark_no_failed_requests",
            failed_requests in (None, 0),
            f"failed_requests={failed_requests}",
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
        "bench_log": str(bench_log) if bench_log else None,
        "scratch_pool_enabled": scratch_enabled,
        "scratch_pool_budget_bytes": scratch_budget,
        "scratch_pool_mode": scratch_mode,
        "scratch_pool_trace_records": scratch_trace,
        "scratch_pool_eligible_records": scratch_eligible,
        "scratch_pool_device_direct_records": scratch_direct,
        "scratch_pool_live_bytes": scratch_live,
        "scratch_pool_peak_live_bytes": scratch_peak,
        "scratch_pool_budget_over_records": scratch_over,
        "scratch_pool_budget_reject_records": scratch_rejects,
        "scratch_pool_reason_counts": reason_counts,
        "failed_requests": failed_requests,
        "checks": checks,
    }


def print_summary(summary: dict[str, Any], output_json: Path) -> None:
    status = "PASS" if summary["passed"] else "FAIL"
    print("===========================================================")
    print(f" Stage G Success Check: {status}")
    print(f" Metrics: {summary['metrics_json']}")
    print(f" Allocator log: {summary['allocator_log']}")
    print(f" Bench log: {summary['bench_log']}")
    print("===========================================================")
    print(
        "- scratch_pool: "
        f"enabled={summary.get('scratch_pool_enabled')} "
        f"budget={summary.get('scratch_pool_budget_bytes')} "
        f"mode={summary.get('scratch_pool_mode')}"
    )
    print(
        "- scratch_pool_records: "
        f"trace={summary.get('scratch_pool_trace_records')} "
        f"eligible={summary.get('scratch_pool_eligible_records')} "
        f"device_direct={summary.get('scratch_pool_device_direct_records')}"
    )
    print(
        "- scratch_pool_live: "
        f"live={summary.get('scratch_pool_live_bytes')} "
        f"peak={summary.get('scratch_pool_peak_live_bytes')}"
    )
    print(
        "- scratch_pool_budget_signals: "
        f"over={summary.get('scratch_pool_budget_over_records')} "
        f"rejects={summary.get('scratch_pool_budget_reject_records')}"
    )
    print(f"- scratch_pool_reason_counts={summary.get('scratch_pool_reason_counts')}")
    print(f"- failed_requests={summary.get('failed_requests')}")
    print("- checks:")
    for item in summary["checks"]:
        print(f"  {item['name']}={item['passed']} ({item['detail']})")
    print(f"- check_json={output_json}")


def main() -> int:
    args = parse_args()
    if args.budget_bytes < 0:
        raise SystemExit("--budget-bytes must be non-negative")
    if args.pool_release_threshold < 0:
        raise SystemExit("--pool-release-threshold must be non-negative")
    if args.min_bytes < 0:
        raise SystemExit("--min-bytes must be non-negative")
    if args.max_bytes < 0:
        raise SystemExit("--max-bytes must be non-negative")
    if args.max_bytes and args.min_bytes > args.max_bytes:
        raise SystemExit("--min-bytes must be <= --max-bytes")

    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir()
    metrics_json, allocator_log, bench_log = resolve_existing_inputs(args)
    runner_log: Path | None = None

    if not args.skip_run:
        metrics_json, allocator_log, bench_log, runner_log = run_experiment(
            args,
            run_dir,
        )
    elif metrics_json is None:
        if allocator_log is None:
            raise SystemExit("--skip-run requires --metrics-json or --allocator-log")
        metrics_json = run_dir / "vllm_stage_g_scratch_pool_metrics.json"
        run_dir.mkdir(parents=True, exist_ok=True)
        summarize_allocator_log(allocator_log, metrics_json)

    assert metrics_json is not None
    summary = validate_metrics(
        metrics_json=metrics_json,
        allocator_log=allocator_log,
        bench_log=bench_log,
        runner_log=runner_log,
        args=args,
    )
    output_json = Path(args.output_json) if args.output_json else (
        run_dir / "stage_g_success_check.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print_summary(summary, output_json)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

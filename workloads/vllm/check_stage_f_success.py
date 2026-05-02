#!/usr/bin/env python3
"""Run or validate a Stage F unified pool registry telemetry check."""

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
            "Check whether Stage F unified pool registry telemetry is wired. "
            "By default this starts vLLM until model load/KV init completes, "
            "then validates allocator-side kv_cache and weights pool objects. "
            "Use --run-bench to also require runtime_scratch pool activity."
        )
    )
    parser.add_argument("--metrics-json", help="Existing summarize_gap_watch_metrics JSON.")
    parser.add_argument("--allocator-log", help="Existing or new allocator trace log.")
    parser.add_argument("--run-dir", help="Existing or new Stage F run directory.")
    parser.add_argument("--output-json", help="Where to write the check summary JSON.")
    parser.add_argument("--run-bench", action="store_true")
    parser.add_argument(
        "--require-runtime-scratch",
        action="store_true",
        help="Fail unless runtime_scratch pool records are present.",
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
    return Path(f"/tmp/vllm_stage_f_success_check_{stamp}")


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


def resolve_existing_inputs(args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    metrics_json = Path(args.metrics_json) if args.metrics_json else None
    allocator_log = Path(args.allocator_log) if args.allocator_log else None
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if metrics_json is None:
            candidate = run_dir / "vllm_stage_f_pool_registry_metrics.json"
            if candidate.is_file():
                metrics_json = candidate
        if allocator_log is None:
            candidate = run_dir / "vllm_uvm_allocator_trace_stage_f.log"
            if candidate.is_file():
                allocator_log = candidate
    return metrics_json, allocator_log


def run_experiment(args: argparse.Namespace, run_dir: Path) -> tuple[Path, Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    allocator_log = Path(args.allocator_log) if args.allocator_log else (
        run_dir / "vllm_uvm_allocator_trace_stage_f.log"
    )
    metrics_json = Path(args.metrics_json) if args.metrics_json else (
        run_dir / "vllm_stage_f_pool_registry_metrics.json"
    )
    runner_log = run_dir / "stage_f_success_check_runner.log"

    cmd = [
        "./run_kv_fault_ratio.sh",
        "--mode",
        "trace",
        "--allocator-log",
        str(allocator_log),
        "--trace-log",
        str(run_dir / "uvm_kv_fault_stats_stage_f.log"),
        "--address-log",
        str(run_dir / "vllm_uvm_address_regions_stage_f.log"),
        "--server-log",
        str(run_dir / "vllm_server_stage_f.log"),
        "--bench-log",
        str(run_dir / "vllm_bench_stage_f.log"),
        "--uvm-kv-budget-bytes",
        "0",
        "--uvm-weight-budget-bytes",
        "0",
        "--uvm-pool-registry-enable",
        "1",
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
    print(" Stage F Success Check: running unified pool registry probe")
    print(f" Output dir: {run_dir}")
    print(f" Benchmark enabled: {args.run_bench}")
    print(f" Require runtime scratch: {args.require_runtime_scratch}")
    print("===========================================================")

    env = os.environ.copy()
    with runner_log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            cwd=SCRIPT_DIR,
            env=env,
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
        raise SystemExit(f"Stage F probe failed with exit code {rc}; log={runner_log}")
    if not metrics_json.is_file():
        if allocator_log.is_file():
            summarize_allocator_log(allocator_log, metrics_json)
        else:
            raise SystemExit(f"Stage F metrics were not produced: {metrics_json}")
    return metrics_json, allocator_log, runner_log


def validate_metrics(
    *,
    metrics_json: Path,
    allocator_log: Path | None,
    runner_log: Path | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    metrics = load_json(metrics_json)
    pool_kind_counts = metrics.get("pool_kind_counts") or {}
    pool_alloc_bytes = metrics.get("pool_alloc_bytes_by_kind") or {}
    pool_registry_alloc_records = as_int(metrics.get("pool_registry_alloc_records"))
    pool_registry_peak_live_objects = as_int(
        metrics.get("pool_registry_peak_live_objects")
    )
    pool_registry_enabled = metrics.get("pool_registry_enabled")
    kv_records = as_int(pool_kind_counts.get("kv_cache"))
    weight_records = as_int(pool_kind_counts.get("weights"))
    runtime_records = as_int(pool_kind_counts.get("runtime_scratch"))
    kv_bytes = as_int(pool_alloc_bytes.get("kv_cache"))
    weight_bytes = as_int(pool_alloc_bytes.get("weights"))
    runtime_bytes = as_int(pool_alloc_bytes.get("runtime_scratch"))
    require_runtime = args.require_runtime_scratch or args.run_bench

    checks = [
        check(
            "metrics_json_present",
            metrics_json.is_file(),
            f"metrics_json={metrics_json}",
        ),
        check(
            "pool_registry_enabled",
            pool_registry_enabled is True,
            f"pool_registry_enabled={pool_registry_enabled}",
        ),
        check(
            "pool_registry_alloc_records_present",
            pool_registry_alloc_records is not None and pool_registry_alloc_records > 0,
            f"pool_registry_alloc_records={pool_registry_alloc_records}",
        ),
        check(
            "pool_registry_peak_live_objects_positive",
            pool_registry_peak_live_objects is not None
            and pool_registry_peak_live_objects > 0,
            f"pool_registry_peak_live_objects={pool_registry_peak_live_objects}",
        ),
        check(
            "kv_cache_pool_records_present",
            kv_records is not None and kv_records > 0,
            f"kv_cache_records={kv_records} kv_cache_bytes={kv_bytes}",
        ),
        check(
            "weights_pool_records_present",
            weight_records is not None and weight_records > 0,
            f"weight_records={weight_records} weight_bytes={weight_bytes}",
        ),
        check(
            "runtime_scratch_pool_records_if_required",
            not require_runtime or (runtime_records is not None and runtime_records > 0),
            (
                f"require_runtime={require_runtime} "
                f"runtime_scratch_records={runtime_records} "
                f"runtime_scratch_bytes={runtime_bytes}"
            ),
        ),
        check(
            "kv_and_weight_bytes_positive",
            (kv_bytes is not None and kv_bytes > 0)
            and (weight_bytes is not None and weight_bytes > 0),
            f"kv_bytes={kv_bytes} weight_bytes={weight_bytes}",
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
        "pool_registry_enabled": pool_registry_enabled,
        "pool_registry_alloc_records": pool_registry_alloc_records,
        "pool_registry_free_records": metrics.get("pool_registry_free_records"),
        "pool_registry_live_objects": metrics.get("pool_registry_live_objects"),
        "pool_registry_peak_live_objects": pool_registry_peak_live_objects,
        "pool_kind_counts": pool_kind_counts,
        "pool_alloc_bytes_by_kind": pool_alloc_bytes,
        "pool_placement_backend_counts": metrics.get(
            "pool_placement_backend_counts"
        ),
        "pool_kv_live_bytes": metrics.get("pool_kv_live_bytes"),
        "pool_weight_live_bytes": metrics.get("pool_weight_live_bytes"),
        "pool_runtime_scratch_live_bytes": metrics.get(
            "pool_runtime_scratch_live_bytes"
        ),
        "require_runtime_scratch": require_runtime,
        "checks": checks,
    }


def print_summary(summary: dict[str, Any], output_json: Path) -> None:
    status = "PASS" if summary["passed"] else "FAIL"
    print("===========================================================")
    print(f" Stage F Success Check: {status}")
    print(f" Metrics: {summary['metrics_json']}")
    print(f" Allocator log: {summary['allocator_log']}")
    print("===========================================================")
    print(
        "- pool_registry: "
        f"enabled={summary.get('pool_registry_enabled')} "
        f"alloc_records={summary.get('pool_registry_alloc_records')} "
        f"free_records={summary.get('pool_registry_free_records')} "
        f"live_objects={summary.get('pool_registry_live_objects')} "
        f"peak_live_objects={summary.get('pool_registry_peak_live_objects')}"
    )
    print(f"- pool_kind_counts={summary.get('pool_kind_counts')}")
    print(f"- pool_alloc_bytes_by_kind={summary.get('pool_alloc_bytes_by_kind')}")
    print(f"- pool_placement_backend_counts={summary.get('pool_placement_backend_counts')}")
    print(
        "- pool_live_bytes: "
        f"kv={summary.get('pool_kv_live_bytes')} "
        f"weights={summary.get('pool_weight_live_bytes')} "
        f"runtime_scratch={summary.get('pool_runtime_scratch_live_bytes')}"
    )
    print(f"- require_runtime_scratch={summary.get('require_runtime_scratch')}")
    print("- checks:")
    for item in summary["checks"]:
        print(f"  {item['name']}={item['passed']} ({item['detail']})")
    print(f"- check_json={output_json}")


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir()
    metrics_json, allocator_log = resolve_existing_inputs(args)
    runner_log: Path | None = None

    if not args.skip_run:
        metrics_json, allocator_log, runner_log = run_experiment(args, run_dir)
    elif metrics_json is None:
        if allocator_log is None:
            raise SystemExit("--skip-run requires --metrics-json or --allocator-log")
        metrics_json = run_dir / "vllm_stage_f_pool_registry_metrics.json"
        run_dir.mkdir(parents=True, exist_ok=True)
        summarize_allocator_log(allocator_log, metrics_json)

    assert metrics_json is not None
    summary = validate_metrics(
        metrics_json=metrics_json,
        allocator_log=allocator_log,
        runner_log=runner_log,
        args=args,
    )
    output_json = Path(args.output_json) if args.output_json else (
        run_dir / "stage_f_success_check.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print_summary(summary, output_json)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

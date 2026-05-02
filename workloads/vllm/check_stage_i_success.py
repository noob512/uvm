#!/usr/bin/env python3
"""Run or validate a Stage I expert weight prefetch success check."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether Stage I MoE expert weight prefetch is wired. The "
            "default run enables routing-informed expert weight slice prefetch "
            "and validates the Stage I JSONL trace plus benchmark health."
        )
    )
    parser.add_argument("--metrics-json", help="Existing summarize_gap_watch_metrics JSON.")
    parser.add_argument("--allocator-log", help="Existing or new allocator trace log.")
    parser.add_argument("--prefetch-trace-jsonl", help="Existing or new Stage I prefetch JSONL.")
    parser.add_argument("--moe-routing-jsonl", help="Existing or new MoE routing JSONL.")
    parser.add_argument("--weight-map-jsonl", help="Existing or new weight map JSONL.")
    parser.add_argument("--bench-log", help="Existing or new benchmark log.")
    parser.add_argument("--run-dir", help="Existing or new Stage I run directory.")
    parser.add_argument("--output-json", help="Where to write the check summary JSON.")
    parser.add_argument(
        "--mode",
        choices=("trace_only", "prefetch"),
        default="prefetch",
        help="Stage I mode. prefetch issues cudaMemPrefetchAsync calls.",
    )
    parser.add_argument("--max-bytes-per-step", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-experts-per-layer", type=int, default=2)
    parser.add_argument("--target-roles", default="moe_gate_up,moe_down")
    parser.add_argument("--prompts", type=int, default=1)
    parser.add_argument("--request-rate", default="5")
    parser.add_argument("--output-len", default="512")
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Do not start a new vLLM run; requires --prefetch-trace-jsonl.",
    )
    return parser.parse_args()


def default_run_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return Path(f"/tmp/vllm_stage_i_success_check_{stamp}")


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


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError:
                continue
    return records


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
) -> tuple[Path | None, Path | None, Path | None, Path | None, Path | None, Path | None]:
    metrics_json = Path(args.metrics_json) if args.metrics_json else None
    allocator_log = Path(args.allocator_log) if args.allocator_log else None
    prefetch_trace = (
        Path(args.prefetch_trace_jsonl) if args.prefetch_trace_jsonl else None
    )
    moe_routing_jsonl = Path(args.moe_routing_jsonl) if args.moe_routing_jsonl else None
    weight_map_jsonl = Path(args.weight_map_jsonl) if args.weight_map_jsonl else None
    bench_log = Path(args.bench_log) if args.bench_log else None
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if metrics_json is None:
            candidate = run_dir / "vllm_stage_i_allocator_metrics.json"
            if candidate.is_file():
                metrics_json = candidate
        if allocator_log is None:
            candidate = run_dir / "vllm_uvm_allocator_trace_stage_i.log"
            if candidate.is_file():
                allocator_log = candidate
        if prefetch_trace is None:
            candidate = run_dir / "vllm_uvm_weight_prefetch_stage_i.jsonl"
            if candidate.is_file():
                prefetch_trace = candidate
        if moe_routing_jsonl is None:
            candidate = run_dir / "vllm_uvm_moe_routing_stage_i.jsonl"
            if candidate.is_file():
                moe_routing_jsonl = candidate
        if weight_map_jsonl is None:
            candidate = run_dir / "vllm_uvm_weight_regions_stage_i.jsonl"
            if candidate.is_file():
                weight_map_jsonl = candidate
        if bench_log is None:
            candidate = run_dir / "vllm_bench_stage_i.log"
            if candidate.is_file():
                bench_log = candidate
    return (
        metrics_json,
        allocator_log,
        prefetch_trace,
        moe_routing_jsonl,
        weight_map_jsonl,
        bench_log,
    )


def run_experiment(
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    allocator_log = Path(args.allocator_log) if args.allocator_log else (
        run_dir / "vllm_uvm_allocator_trace_stage_i.log"
    )
    metrics_json = Path(args.metrics_json) if args.metrics_json else (
        run_dir / "vllm_stage_i_allocator_metrics.json"
    )
    prefetch_trace = (
        Path(args.prefetch_trace_jsonl)
        if args.prefetch_trace_jsonl
        else run_dir / "vllm_uvm_weight_prefetch_stage_i.jsonl"
    )
    moe_routing_jsonl = Path(args.moe_routing_jsonl) if args.moe_routing_jsonl else (
        run_dir / "vllm_uvm_moe_routing_stage_i.jsonl"
    )
    weight_map_jsonl = Path(args.weight_map_jsonl) if args.weight_map_jsonl else (
        run_dir / "vllm_uvm_weight_regions_stage_i.jsonl"
    )
    bench_log = Path(args.bench_log) if args.bench_log else (
        run_dir / "vllm_bench_stage_i.log"
    )
    runner_log = run_dir / "stage_i_success_check_runner.log"

    cmd = [
        "./run_kv_fault_ratio.sh",
        "--mode",
        "trace",
        "--allocator-log",
        str(allocator_log),
        "--trace-log",
        str(run_dir / "uvm_kv_fault_stats_stage_i.log"),
        "--address-log",
        str(run_dir / "vllm_uvm_address_regions_stage_i.log"),
        "--server-log",
        str(run_dir / "vllm_server_stage_i.log"),
        "--bench-log",
        str(bench_log),
        "--uvm-kv-budget-bytes",
        "0",
        "--uvm-weight-budget-bytes",
        "0",
        "--uvm-weight-map-enable",
        "1",
        "--uvm-weight-map-file",
        str(weight_map_jsonl),
        "--uvm-moe-routing-trace-enable",
        "1",
        "--uvm-moe-routing-trace-file",
        str(moe_routing_jsonl),
        "--uvm-weight-prefetch-enable",
        "1",
        "--uvm-weight-prefetch-mode",
        args.mode,
        "--uvm-weight-prefetch-trace-file",
        str(prefetch_trace),
        "--uvm-weight-prefetch-max-bytes-per-step",
        str(args.max_bytes_per_step),
        "--uvm-weight-prefetch-max-experts-per-layer",
        str(args.max_experts_per_layer),
        "--uvm-weight-prefetch-target-roles",
        args.target_roles,
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

    print("===========================================================")
    print(" Stage I Success Check: running expert weight prefetch probe")
    print(f" Output dir: {run_dir}")
    print(f" Prefetch mode: {args.mode}")
    print(f" Max bytes per step: {args.max_bytes_per_step}")
    print(f" Max experts per layer: {args.max_experts_per_layer}")
    print(f" Target roles: {args.target_roles}")
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
        raise SystemExit(f"Stage I probe failed with exit code {rc}; log={runner_log}")
    if not metrics_json.is_file():
        if allocator_log.is_file():
            summarize_allocator_log(allocator_log, metrics_json)
        else:
            raise SystemExit(f"Stage I metrics were not produced: {metrics_json}")
    return (
        metrics_json,
        allocator_log,
        prefetch_trace,
        moe_routing_jsonl,
        weight_map_jsonl,
        bench_log,
        runner_log,
    )


def summarize_prefetch_trace(records: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    issued_bytes = 0
    issued_records = 0
    prefetch_issued = 0
    trace_candidates = 0
    budget_rejects = 0
    max_step_issued_bytes = 0
    max_summary_attempted_bytes = 0
    layers: set[str] = set()
    experts: set[tuple[str, int]] = set()

    for record in records:
        action = str(record.get("action", "unknown"))
        action_counts[action] += 1
        layer_name = str(record.get("layer_name", "unknown"))
        layers.add(layer_name)
        local_expert = as_int(record.get("local_expert_id"))
        if local_expert is not None:
            experts.add((layer_name, local_expert))
        role = record.get("role")
        if role is not None:
            role_counts[str(role)] += 1
        if action in {"prefetch_issued", "trace_prefetch_candidate", "prefetch_skipped"}:
            issued_records += 1
            issued_bytes += int(record.get("bytes") or 0)
        if action == "prefetch_issued":
            prefetch_issued += 1
        if action == "trace_prefetch_candidate":
            trace_candidates += 1
        if action == "budget_reject":
            budget_rejects += 1
        if action == "step_summary":
            max_step_issued_bytes = max(
                max_step_issued_bytes,
                int(record.get("issued_bytes") or 0),
            )
            max_summary_attempted_bytes = max(
                max_summary_attempted_bytes,
                int(record.get("attempted_bytes") or 0),
            )

    return {
        "prefetch_trace_records": len(records),
        "prefetch_action_counts": dict(action_counts),
        "prefetch_role_counts": dict(role_counts),
        "prefetch_issued_records": prefetch_issued,
        "trace_candidate_records": trace_candidates,
        "prefetch_candidate_or_issued_records": issued_records,
        "prefetch_candidate_or_issued_bytes": issued_bytes,
        "budget_reject_records": budget_rejects,
        "prefetch_layer_count": len(layers),
        "prefetch_expert_count": len(experts),
        "max_step_issued_bytes": max_step_issued_bytes,
        "max_summary_attempted_bytes": max_summary_attempted_bytes,
    }


def validate(
    *,
    metrics_json: Path | None,
    allocator_log: Path | None,
    prefetch_trace: Path,
    moe_routing_jsonl: Path | None,
    weight_map_jsonl: Path | None,
    bench_log: Path | None,
    runner_log: Path | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    records = read_jsonl(prefetch_trace)
    trace_summary = summarize_prefetch_trace(records)
    metrics = load_json(metrics_json) if metrics_json and metrics_json.is_file() else {}
    failed_requests = parse_failed_requests(bench_log)
    weight_trace_allocs = as_int(metrics.get("weight_trace_allocations"))
    pool_registry_enabled = metrics.get("pool_registry_enabled")
    max_step_issued_bytes = as_int(trace_summary.get("max_step_issued_bytes"))
    prefetch_issued_records = as_int(trace_summary.get("prefetch_issued_records"))
    trace_candidate_records = as_int(trace_summary.get("trace_candidate_records"))
    candidate_or_issued = as_int(
        trace_summary.get("prefetch_candidate_or_issued_records")
    )

    checks = [
        check(
            "prefetch_trace_present",
            prefetch_trace.is_file(),
            f"prefetch_trace={prefetch_trace}",
        ),
        check(
            "prefetch_trace_records_present",
            trace_summary["prefetch_trace_records"] > 0,
            f"prefetch_trace_records={trace_summary['prefetch_trace_records']}",
        ),
        check(
            "prefetch_candidates_or_issued_present",
            candidate_or_issued is not None and candidate_or_issued > 0,
            f"prefetch_candidate_or_issued_records={candidate_or_issued}",
        ),
        check(
            "prefetch_issued_present_in_prefetch_mode",
            args.mode != "prefetch"
            or (prefetch_issued_records is not None and prefetch_issued_records > 0),
            f"prefetch_issued_records={prefetch_issued_records}",
        ),
        check(
            "trace_candidates_present_in_trace_mode",
            args.mode != "trace_only"
            or (trace_candidate_records is not None and trace_candidate_records > 0),
            f"trace_candidate_records={trace_candidate_records}",
        ),
        check(
            "prefetch_step_bytes_within_budget",
            max_step_issued_bytes is not None
            and max_step_issued_bytes <= args.max_bytes_per_step,
            (
                f"max_step_issued_bytes={max_step_issued_bytes} "
                f"budget={args.max_bytes_per_step}"
            ),
        ),
        check(
            "moe_routing_trace_present",
            moe_routing_jsonl is not None and moe_routing_jsonl.is_file(),
            f"moe_routing_jsonl={moe_routing_jsonl}",
        ),
        check(
            "weight_map_present",
            weight_map_jsonl is not None and weight_map_jsonl.is_file(),
            f"weight_map_jsonl={weight_map_jsonl}",
        ),
        check(
            "benchmark_no_failed_requests",
            failed_requests in (None, 0),
            f"failed_requests={failed_requests}",
        ),
    ]

    if metrics:
        checks.extend(
            [
                check(
                    "allocator_weight_metrics_present",
                    weight_trace_allocs is not None and weight_trace_allocs > 0,
                    f"weight_trace_allocations={weight_trace_allocs}",
                ),
                check(
                    "pool_registry_enabled",
                    pool_registry_enabled is True,
                    f"pool_registry_enabled={pool_registry_enabled}",
                ),
            ]
        )

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
        "metrics_json": str(metrics_json) if metrics_json else None,
        "allocator_log": str(allocator_log) if allocator_log else None,
        "prefetch_trace_jsonl": str(prefetch_trace),
        "moe_routing_jsonl": str(moe_routing_jsonl) if moe_routing_jsonl else None,
        "weight_map_jsonl": str(weight_map_jsonl) if weight_map_jsonl else None,
        "bench_log": str(bench_log) if bench_log else None,
        "mode": args.mode,
        "max_bytes_per_step": args.max_bytes_per_step,
        "max_experts_per_layer": args.max_experts_per_layer,
        "target_roles": args.target_roles,
        "failed_requests": failed_requests,
        "weight_trace_allocations": weight_trace_allocs,
        "pool_registry_enabled": pool_registry_enabled,
        **trace_summary,
        "checks": checks,
    }


def print_summary(summary: dict[str, Any], output_json: Path) -> None:
    status = "PASS" if summary["passed"] else "FAIL"
    print("===========================================================")
    print(f" Stage I Success Check: {status}")
    print(f" Prefetch trace: {summary['prefetch_trace_jsonl']}")
    print(f" Metrics: {summary['metrics_json']}")
    print(f" Bench log: {summary['bench_log']}")
    print("===========================================================")
    print(
        "- prefetch: "
        f"mode={summary.get('mode')} "
        f"records={summary.get('prefetch_trace_records')} "
        f"issued={summary.get('prefetch_issued_records')} "
        f"candidates={summary.get('trace_candidate_records')} "
        f"bytes={summary.get('prefetch_candidate_or_issued_bytes')}"
    )
    print(
        "- prefetch_scope: "
        f"layers={summary.get('prefetch_layer_count')} "
        f"experts={summary.get('prefetch_expert_count')} "
        f"roles={summary.get('prefetch_role_counts')}"
    )
    print(
        "- budget: "
        f"max_step_issued_bytes={summary.get('max_step_issued_bytes')} "
        f"max_bytes_per_step={summary.get('max_bytes_per_step')} "
        f"rejects={summary.get('budget_reject_records')}"
    )
    print(f"- action_counts={summary.get('prefetch_action_counts')}")
    print(f"- failed_requests={summary.get('failed_requests')}")
    print("- checks:")
    for item in summary["checks"]:
        print(f"  {item['name']}={item['passed']} ({item['detail']})")
    print(f"- check_json={output_json}")


def main() -> int:
    args = parse_args()
    if args.max_bytes_per_step < 0:
        raise SystemExit("--max-bytes-per-step must be non-negative")
    if args.max_experts_per_layer < 0:
        raise SystemExit("--max-experts-per-layer must be non-negative")
    if args.prompts < 0:
        raise SystemExit("--prompts must be non-negative")

    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir()
    (
        metrics_json,
        allocator_log,
        prefetch_trace,
        moe_routing_jsonl,
        weight_map_jsonl,
        bench_log,
    ) = resolve_existing_inputs(args)
    runner_log: Path | None = None

    if not args.skip_run:
        (
            metrics_json,
            allocator_log,
            prefetch_trace,
            moe_routing_jsonl,
            weight_map_jsonl,
            bench_log,
            runner_log,
        ) = run_experiment(args, run_dir)
    else:
        if prefetch_trace is None:
            raise SystemExit("--skip-run requires --prefetch-trace-jsonl or --run-dir")
        if metrics_json is None and allocator_log is not None:
            metrics_json = run_dir / "vllm_stage_i_allocator_metrics.json"
            run_dir.mkdir(parents=True, exist_ok=True)
            summarize_allocator_log(allocator_log, metrics_json)

    assert prefetch_trace is not None
    summary = validate(
        metrics_json=metrics_json,
        allocator_log=allocator_log,
        prefetch_trace=prefetch_trace,
        moe_routing_jsonl=moe_routing_jsonl,
        weight_map_jsonl=weight_map_jsonl,
        bench_log=bench_log,
        runner_log=runner_log,
        args=args,
    )
    output_json = Path(args.output_json) if args.output_json else (
        run_dir / "stage_i_success_check.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print_summary(summary, output_json)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

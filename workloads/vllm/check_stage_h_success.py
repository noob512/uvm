#!/usr/bin/env python3
"""Run or validate a Stage H weights hot/cold trace-only planning check."""

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
            "Check whether Stage H weight expert hot/cold trace-only planning "
            "is wired. By default this runs a tiny benchmark with Stage E "
            "weight map and MoE routing trace enabled, then builds a Stage H "
            "prefetch/offload candidate plan without executing migration."
        )
    )
    parser.add_argument("--metrics-json", help="Existing summarize_gap_watch_metrics JSON.")
    parser.add_argument("--allocator-log", help="Existing or new allocator trace log.")
    parser.add_argument("--weight-map-jsonl", help="Existing or new Stage E weight map JSONL.")
    parser.add_argument("--moe-routing-jsonl", help="Existing or new Stage E MoE routing JSONL.")
    parser.add_argument("--fault-log", help="Existing or new per-fault address log.")
    parser.add_argument("--bench-log", help="Existing or new benchmark log.")
    parser.add_argument("--plan-json", help="Existing or new Stage H plan JSON.")
    parser.add_argument("--plan-summary-json", help="Existing or new Stage H plan summary JSON.")
    parser.add_argument("--run-dir", help="Existing or new Stage H run directory.")
    parser.add_argument("--output-json", help="Where to write the check summary JSON.")
    parser.add_argument("--prompts", type=int, default=1)
    parser.add_argument("--request-rate", default="5")
    parser.add_argument("--output-len", default="512")
    parser.add_argument(
        "--target-roles",
        default="moe_gate_up,moe_gate,moe_up,moe_down",
        help="Comma-separated expert roles eligible for Stage H planning.",
    )
    parser.add_argument("--hot-top-k", type=int, default=16)
    parser.add_argument("--cold-bottom-k", type=int, default=16)
    parser.add_argument("--hot-min-routing-tokens", type=int, default=1)
    parser.add_argument("--prefetch-plan-max-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--offload-plan-max-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument(
        "--require-fault-join",
        action="store_true",
        help="Also require replayable fault addresses to join with expert weights.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Do not start a new vLLM run; requires existing weight map inputs.",
    )
    return parser.parse_args()


def default_run_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return Path(f"/tmp/vllm_stage_h_success_check_{stamp}")


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
) -> tuple[Path | None, Path | None, Path | None, Path | None, Path | None, Path | None, Path | None]:
    metrics_json = Path(args.metrics_json) if args.metrics_json else None
    allocator_log = Path(args.allocator_log) if args.allocator_log else None
    weight_map_jsonl = Path(args.weight_map_jsonl) if args.weight_map_jsonl else None
    moe_routing_jsonl = Path(args.moe_routing_jsonl) if args.moe_routing_jsonl else None
    fault_log = Path(args.fault_log) if args.fault_log else None
    bench_log = Path(args.bench_log) if args.bench_log else None
    plan_json = Path(args.plan_json) if args.plan_json else None
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if metrics_json is None:
            candidate = run_dir / "vllm_stage_h_allocator_metrics.json"
            if candidate.is_file():
                metrics_json = candidate
        if allocator_log is None:
            candidate = run_dir / "vllm_uvm_allocator_trace_stage_h.log"
            if candidate.is_file():
                allocator_log = candidate
        if weight_map_jsonl is None:
            candidate = run_dir / "vllm_uvm_weight_regions_stage_h.jsonl"
            if candidate.is_file():
                weight_map_jsonl = candidate
        if moe_routing_jsonl is None:
            candidate = run_dir / "vllm_uvm_moe_routing_stage_h.jsonl"
            if candidate.is_file():
                moe_routing_jsonl = candidate
        if fault_log is None:
            candidate = run_dir / "uvm_kv_fault_addrs_stage_h.log"
            if candidate.is_file():
                fault_log = candidate
        if bench_log is None:
            candidate = run_dir / "vllm_bench_stage_h.log"
            if candidate.is_file():
                bench_log = candidate
        if plan_json is None:
            candidate = run_dir / "vllm_stage_h_weight_expert_plan.json"
            if candidate.is_file():
                plan_json = candidate
    return (
        metrics_json,
        allocator_log,
        weight_map_jsonl,
        moe_routing_jsonl,
        fault_log,
        bench_log,
        plan_json,
    )


def run_experiment(
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    allocator_log = Path(args.allocator_log) if args.allocator_log else (
        run_dir / "vllm_uvm_allocator_trace_stage_h.log"
    )
    metrics_json = Path(args.metrics_json) if args.metrics_json else (
        run_dir / "vllm_stage_h_allocator_metrics.json"
    )
    weight_map_jsonl = Path(args.weight_map_jsonl) if args.weight_map_jsonl else (
        run_dir / "vllm_uvm_weight_regions_stage_h.jsonl"
    )
    moe_routing_jsonl = Path(args.moe_routing_jsonl) if args.moe_routing_jsonl else (
        run_dir / "vllm_uvm_moe_routing_stage_h.jsonl"
    )
    fault_log = Path(args.fault_log) if args.fault_log else (
        run_dir / "uvm_kv_fault_addrs_stage_h.log"
    )
    bench_log = Path(args.bench_log) if args.bench_log else (
        run_dir / "vllm_bench_stage_h.log"
    )
    plan_json = Path(args.plan_json) if args.plan_json else (
        run_dir / "vllm_stage_h_weight_expert_plan.json"
    )
    plan_summary_json = Path(args.plan_summary_json) if args.plan_summary_json else (
        run_dir / "vllm_stage_h_weight_expert_plan_summary.json"
    )
    runner_log = run_dir / "stage_h_success_check_runner.log"

    cmd = [
        "./run_kv_fault_ratio.sh",
        "--mode",
        "trace",
        "--with-address-log",
        "--allocator-log",
        str(allocator_log),
        "--trace-log",
        str(run_dir / "uvm_kv_fault_stats_stage_h.log"),
        "--address-trace-log",
        str(fault_log),
        "--address-log",
        str(run_dir / "vllm_uvm_address_regions_stage_h.log"),
        "--server-log",
        str(run_dir / "vllm_server_stage_h.log"),
        "--bench-log",
        str(bench_log),
        "--uvm-kv-budget-bytes",
        "0",
        "--uvm-weight-budget-bytes",
        "0",
        "--uvm-weight-budget-mode",
        "trace_only",
        "--uvm-weight-map-enable",
        "1",
        "--uvm-weight-map-file",
        str(weight_map_jsonl),
        "--uvm-moe-routing-trace-enable",
        "1",
        "--uvm-moe-routing-trace-file",
        str(moe_routing_jsonl),
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
    print(" Stage H Success Check: running weight hot/cold planning probe")
    print(f" Output dir: {run_dir}")
    print(f" Target roles: {args.target_roles}")
    print(f" Prefetch plan max bytes: {args.prefetch_plan_max_bytes}")
    print(f" Offload plan max bytes: {args.offload_plan_max_bytes}")
    print(" Trace-only: no weight prefetch/offload execution")
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
        raise SystemExit(f"Stage H probe failed with exit code {rc}; log={runner_log}")
    if not metrics_json.is_file():
        if allocator_log.is_file():
            summarize_allocator_log(allocator_log, metrics_json)
        else:
            raise SystemExit(f"Stage H metrics were not produced: {metrics_json}")
    build_plan(
        args=args,
        weight_map_jsonl=weight_map_jsonl,
        moe_routing_jsonl=moe_routing_jsonl,
        fault_log=fault_log,
        plan_json=plan_json,
        plan_summary_json=plan_summary_json,
    )
    return (
        metrics_json,
        allocator_log,
        weight_map_jsonl,
        moe_routing_jsonl,
        fault_log,
        bench_log,
        plan_json,
        runner_log,
    )


def build_plan(
    *,
    args: argparse.Namespace,
    weight_map_jsonl: Path,
    moe_routing_jsonl: Path | None,
    fault_log: Path | None,
    plan_json: Path,
    plan_summary_json: Path | None,
) -> None:
    cmd = [
        "python3",
        str(SCRIPT_DIR / "plan_stage_h_weight_expert_actions.py"),
        "--weight-map",
        str(weight_map_jsonl),
        "--plan-json",
        str(plan_json),
        "--target-roles",
        args.target_roles,
        "--hot-top-k",
        str(args.hot_top_k),
        "--cold-bottom-k",
        str(args.cold_bottom_k),
        "--hot-min-routing-tokens",
        str(args.hot_min_routing_tokens),
        "--prefetch-plan-max-bytes",
        str(args.prefetch_plan_max_bytes),
        "--offload-plan-max-bytes",
        str(args.offload_plan_max_bytes),
    ]
    if moe_routing_jsonl is not None and moe_routing_jsonl.is_file():
        cmd.extend(["--moe-routing-trace", str(moe_routing_jsonl), "--require-routing"])
    if fault_log is not None and fault_log.is_file():
        cmd.extend(["--fault-log", str(fault_log)])
    if plan_summary_json is not None:
        cmd.extend(["--summary-json", str(plan_summary_json)])
    subprocess.run(cmd, cwd=SCRIPT_DIR, check=True)


def validate_plan(
    *,
    metrics_json: Path | None,
    allocator_log: Path | None,
    weight_map_jsonl: Path | None,
    moe_routing_jsonl: Path | None,
    fault_log: Path | None,
    bench_log: Path | None,
    plan_json: Path,
    runner_log: Path | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    plan = load_json(plan_json)
    metrics = load_json(metrics_json) if metrics_json and metrics_json.is_file() else {}
    failed_requests = parse_failed_requests(bench_log)

    expert_ranges = as_int(plan.get("expert_weight_range_records"))
    logical_slices = as_int(plan.get("logical_fused_expert_records"))
    heat_records = as_int(plan.get("expert_heat_records"))
    routing_records = as_int(plan.get("moe_routing_records"))
    routing_join = as_int(plan.get("routing_join_records"))
    fault_join = as_int(plan.get("weight_fault_join_records"))
    prefetch_records = as_int(plan.get("prefetch_plan_records"))
    offload_records = as_int(plan.get("offload_plan_records"))
    prefetch_bytes = as_int(plan.get("prefetch_plan_bytes"))
    offload_bytes = as_int(plan.get("offload_plan_bytes"))
    weight_trace_allocs = as_int(metrics.get("weight_trace_allocations"))
    pool_registry_enabled = metrics.get("pool_registry_enabled")

    checks = [
        check("plan_json_present", plan_json.is_file(), f"plan_json={plan_json}"),
        check(
            "trace_only_mode",
            plan.get("mode") == "trace_only",
            f"mode={plan.get('mode')}",
        ),
        check(
            "weight_map_jsonl_present",
            weight_map_jsonl is not None and weight_map_jsonl.is_file(),
            f"weight_map_jsonl={weight_map_jsonl}",
        ),
        check(
            "expert_weight_ranges_present",
            expert_ranges is not None and expert_ranges > 0,
            f"expert_weight_range_records={expert_ranges}",
        ),
        check(
            "logical_or_concrete_expert_signal_present",
            (logical_slices is not None and logical_slices > 0)
            or (as_int(plan.get("concrete_expert_weight_records")) or 0) > 0,
            (
                f"logical_fused_expert_records={logical_slices} "
                f"concrete_expert_weight_records={plan.get('concrete_expert_weight_records')}"
            ),
        ),
        check(
            "expert_heat_records_present",
            heat_records is not None and heat_records > 0,
            f"expert_heat_records={heat_records}",
        ),
        check(
            "moe_routing_trace_present",
            routing_records is not None and routing_records > 0,
            f"moe_routing_records={routing_records}",
        ),
        check(
            "routing_join_records_present",
            routing_join is not None and routing_join > 0,
            f"routing_join_records={routing_join}",
        ),
        check(
            "prefetch_plan_records_present",
            prefetch_records is not None and prefetch_records > 0,
            f"prefetch_plan_records={prefetch_records}",
        ),
        check(
            "offload_plan_records_present",
            offload_records is not None and offload_records > 0,
            f"offload_plan_records={offload_records}",
        ),
        check(
            "prefetch_plan_within_budget",
            prefetch_bytes is not None and prefetch_bytes <= args.prefetch_plan_max_bytes,
            (
                f"prefetch_plan_bytes={prefetch_bytes} "
                f"max={args.prefetch_plan_max_bytes}"
            ),
        ),
        check(
            "offload_plan_within_budget",
            offload_bytes is not None and offload_bytes <= args.offload_plan_max_bytes,
            f"offload_plan_bytes={offload_bytes} max={args.offload_plan_max_bytes}",
        ),
        check(
            "fault_join_present_if_required",
            not args.require_fault_join or (fault_join is not None and fault_join > 0),
            f"weight_fault_join_records={fault_join}",
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
        "weight_map_jsonl": str(weight_map_jsonl) if weight_map_jsonl else None,
        "moe_routing_jsonl": str(moe_routing_jsonl) if moe_routing_jsonl else None,
        "fault_log": str(fault_log) if fault_log else None,
        "bench_log": str(bench_log) if bench_log else None,
        "plan_json": str(plan_json),
        "mode": plan.get("mode"),
        "expert_weight_range_records": expert_ranges,
        "logical_fused_expert_records": logical_slices,
        "concrete_expert_weight_records": plan.get("concrete_expert_weight_records"),
        "expert_heat_records": heat_records,
        "expert_weight_bytes": plan.get("expert_weight_bytes"),
        "moe_routing_records": routing_records,
        "routing_join_records": routing_join,
        "weight_fault_join_records": fault_join,
        "prefetch_plan_records": prefetch_records,
        "prefetch_plan_bytes": prefetch_bytes,
        "offload_plan_records": offload_records,
        "offload_plan_bytes": offload_bytes,
        "top_hot_experts": plan.get("top_hot_experts", [])[:5],
        "coldest_experts": plan.get("coldest_experts", [])[:5],
        "weight_trace_allocations": weight_trace_allocs,
        "pool_registry_enabled": pool_registry_enabled,
        "failed_requests": failed_requests,
        "checks": checks,
    }


def print_summary(summary: dict[str, Any], output_json: Path) -> None:
    status = "PASS" if summary["passed"] else "FAIL"
    print("===========================================================")
    print(f" Stage H Success Check: {status}")
    print(f" Plan: {summary['plan_json']}")
    print(f" Metrics: {summary['metrics_json']}")
    print(f" Weight map: {summary['weight_map_jsonl']}")
    print(f" MoE routing: {summary['moe_routing_jsonl']}")
    print("===========================================================")
    print(
        "- expert_weights: "
        f"ranges={summary.get('expert_weight_range_records')} "
        f"logical_fused={summary.get('logical_fused_expert_records')} "
        f"concrete={summary.get('concrete_expert_weight_records')} "
        f"bytes={summary.get('expert_weight_bytes')}"
    )
    print(
        "- heat_sources: "
        f"routing_records={summary.get('moe_routing_records')} "
        f"routing_join={summary.get('routing_join_records')} "
        f"fault_join={summary.get('weight_fault_join_records')}"
    )
    print(
        "- trace_only_plan: "
        f"prefetch_records={summary.get('prefetch_plan_records')} "
        f"prefetch_bytes={summary.get('prefetch_plan_bytes')} "
        f"offload_records={summary.get('offload_plan_records')} "
        f"offload_bytes={summary.get('offload_plan_bytes')}"
    )
    print(f"- failed_requests={summary.get('failed_requests')}")
    print("- checks:")
    for item in summary["checks"]:
        print(f"  {item['name']}={item['passed']} ({item['detail']})")
    print(f"- check_json={output_json}")


def main() -> int:
    args = parse_args()
    if args.prompts < 0:
        raise SystemExit("--prompts must be non-negative")
    if args.hot_top_k < 0 or args.cold_bottom_k < 0:
        raise SystemExit("--hot-top-k and --cold-bottom-k must be non-negative")
    if args.prefetch_plan_max_bytes < 0 or args.offload_plan_max_bytes < 0:
        raise SystemExit("plan byte budgets must be non-negative")

    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir()
    (
        metrics_json,
        allocator_log,
        weight_map_jsonl,
        moe_routing_jsonl,
        fault_log,
        bench_log,
        plan_json,
    ) = resolve_existing_inputs(args)
    runner_log: Path | None = None

    if not args.skip_run:
        (
            metrics_json,
            allocator_log,
            weight_map_jsonl,
            moe_routing_jsonl,
            fault_log,
            bench_log,
            plan_json,
            runner_log,
        ) = run_experiment(args, run_dir)
    else:
        if weight_map_jsonl is None:
            raise SystemExit("--skip-run requires --weight-map-jsonl or --run-dir")
        if plan_json is None:
            plan_json = run_dir / "vllm_stage_h_weight_expert_plan.json"
            run_dir.mkdir(parents=True, exist_ok=True)
            build_plan(
                args=args,
                weight_map_jsonl=weight_map_jsonl,
                moe_routing_jsonl=moe_routing_jsonl,
                fault_log=fault_log,
                plan_json=plan_json,
                plan_summary_json=(
                    Path(args.plan_summary_json) if args.plan_summary_json else None
                ),
            )
        if metrics_json is None and allocator_log is not None:
            metrics_json = run_dir / "vllm_stage_h_allocator_metrics.json"
            run_dir.mkdir(parents=True, exist_ok=True)
            summarize_allocator_log(allocator_log, metrics_json)

    assert plan_json is not None
    summary = validate_plan(
        metrics_json=metrics_json,
        allocator_log=allocator_log,
        weight_map_jsonl=weight_map_jsonl,
        moe_routing_jsonl=moe_routing_jsonl,
        fault_log=fault_log,
        bench_log=bench_log,
        plan_json=plan_json,
        runner_log=runner_log,
        args=args,
    )
    output_json = Path(args.output_json) if args.output_json else (
        run_dir / "stage_h_success_check.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print_summary(summary, output_json)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

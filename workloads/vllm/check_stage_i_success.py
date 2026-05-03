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
            "default run first generates a Stage H hot/cold expert plan, then "
            "executes Stage I plan-gated hot prefetch and optional cold offload."
        )
    )
    parser.add_argument("--metrics-json", help="Existing summarize_gap_watch_metrics JSON.")
    parser.add_argument("--allocator-log", help="Existing or new allocator trace log.")
    parser.add_argument("--prefetch-trace-jsonl", help="Existing or new Stage I prefetch JSONL.")
    parser.add_argument("--moe-routing-jsonl", help="Existing or new MoE routing JSONL.")
    parser.add_argument("--weight-map-jsonl", help="Existing or new weight map JSONL.")
    parser.add_argument("--plan-json", help="Existing or new Stage H plan JSON used by Stage I.")
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
    parser.add_argument(
        "--offload-mode",
        choices=("trace_only", "advise_cpu", "prefetch_cpu"),
        default="advise_cpu",
        help="Cold expert action mode driven by Stage H offload_plan.",
    )
    parser.add_argument("--offload-max-bytes-per-step", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--offload-max-experts-per-layer", type=int, default=1)
    parser.add_argument("--offload-target-roles", default="moe_gate_up,moe_down")
    parser.add_argument(
        "--disable-offload",
        action="store_true",
        help="Only validate Stage I hot expert prefetch, not cold offload/advise.",
    )
    parser.add_argument("--prompts", type=int, default=1)
    parser.add_argument(
        "--plan-prompts",
        type=int,
        default=1,
        help="Benchmark prompts for the Stage H planning probe.",
    )
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
) -> tuple[
    Path | None,
    Path | None,
    Path | None,
    Path | None,
    Path | None,
    Path | None,
    Path | None,
]:
    metrics_json = Path(args.metrics_json) if args.metrics_json else None
    allocator_log = Path(args.allocator_log) if args.allocator_log else None
    prefetch_trace = (
        Path(args.prefetch_trace_jsonl) if args.prefetch_trace_jsonl else None
    )
    moe_routing_jsonl = Path(args.moe_routing_jsonl) if args.moe_routing_jsonl else None
    weight_map_jsonl = Path(args.weight_map_jsonl) if args.weight_map_jsonl else None
    plan_json = Path(args.plan_json) if args.plan_json else None
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
        if plan_json is None:
            candidate = run_dir / "vllm_stage_i_weight_expert_plan.json"
            if candidate.is_file():
                plan_json = candidate
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
        plan_json,
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
    plan_json = Path(args.plan_json) if args.plan_json else (
        run_dir / "vllm_stage_i_weight_expert_plan.json"
    )
    plan_summary_json = run_dir / "vllm_stage_i_weight_expert_plan_summary.json"
    bench_log = Path(args.bench_log) if args.bench_log else (
        run_dir / "vllm_bench_stage_i.log"
    )
    runner_log = run_dir / "stage_i_success_check_runner.log"
    planning_allocator_log = run_dir / "vllm_uvm_allocator_trace_stage_i_planning.log"
    planning_metrics_json = run_dir / "vllm_stage_i_allocator_metrics_planning.json"
    planning_moe_routing_jsonl = run_dir / "vllm_uvm_moe_routing_stage_i_planning.jsonl"
    planning_weight_map_jsonl = run_dir / "vllm_uvm_weight_regions_stage_i_planning.jsonl"
    planning_bench_log = run_dir / "vllm_bench_stage_i_planning.log"
    planning_runner_log = run_dir / "stage_i_planning_runner.log"

    planning_cmd = [
        "./run_kv_fault_ratio.sh",
        "--mode",
        "trace",
        "--allocator-log",
        str(planning_allocator_log),
        "--trace-log",
        str(run_dir / "uvm_kv_fault_stats_stage_i_planning.log"),
        "--address-log",
        str(run_dir / "vllm_uvm_address_regions_stage_i_planning.log"),
        "--server-log",
        str(run_dir / "vllm_server_stage_i_planning.log"),
        "--bench-log",
        str(planning_bench_log),
        "--uvm-kv-budget-bytes",
        "0",
        "--uvm-weight-budget-bytes",
        "0",
        "--uvm-weight-map-enable",
        "1",
        "--uvm-weight-map-file",
        str(planning_weight_map_jsonl),
        "--uvm-moe-routing-trace-enable",
        "1",
        "--uvm-moe-routing-trace-file",
        str(planning_moe_routing_jsonl),
        "--uvm-weight-prefetch-enable",
        "0",
        "--uvm-weight-offload-enable",
        "0",
        "--uvm-pool-registry-enable",
        "1",
        "--gap-watch-metrics-summary-json",
        str(planning_metrics_json),
        "--prompts",
        str(args.plan_prompts),
        "--request-rate",
        str(args.request_rate),
        "--output-len",
        str(args.output_len),
    ]

    print("===========================================================")
    print(" Stage I Success Check: planning hot/cold expert actions")
    print(f" Output dir: {run_dir}")
    print(f" Plan JSON: {plan_json}")
    print("===========================================================")
    with planning_runner_log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            planning_cmd,
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
        raise SystemExit(
            f"Stage I planning probe failed with exit code {rc}; "
            f"log={planning_runner_log}"
        )

    subprocess.run(
        [
            "python3",
            str(SCRIPT_DIR / "plan_stage_h_weight_expert_actions.py"),
            "--weight-map",
            str(planning_weight_map_jsonl),
            "--moe-routing-trace",
            str(planning_moe_routing_jsonl),
            "--plan-json",
            str(plan_json),
            "--summary-json",
            str(plan_summary_json),
            "--target-roles",
            args.target_roles,
            "--hot-top-k",
            "1024",
            "--cold-bottom-k",
            "1024",
            "--require-routing",
        ],
        cwd=SCRIPT_DIR,
        check=True,
    )

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
        "--uvm-weight-prefetch-plan-file",
        str(plan_json),
        "--uvm-weight-prefetch-require-plan",
        "1",
        "--uvm-weight-offload-enable",
        "0" if args.disable_offload else "1",
        "--uvm-weight-offload-mode",
        args.offload_mode,
        "--uvm-weight-offload-plan-file",
        str(plan_json),
        "--uvm-weight-offload-max-bytes-per-step",
        str(args.offload_max_bytes_per_step),
        "--uvm-weight-offload-max-experts-per-layer",
        str(args.offload_max_experts_per_layer),
        "--uvm-weight-offload-target-roles",
        args.offload_target_roles,
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
    print(f" Plan JSON: {plan_json}")
    print(f" Prefetch mode: {args.mode}")
    print(f" Max bytes per step: {args.max_bytes_per_step}")
    print(f" Max experts per layer: {args.max_experts_per_layer}")
    print(f" Target roles: {args.target_roles}")
    print(f" Offload enabled: {not args.disable_offload}")
    print(f" Offload mode: {args.offload_mode}")
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
        plan_json,
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
    offload_issued = 0
    trace_offload_candidates = 0
    budget_rejects = 0
    offload_budget_rejects = 0
    max_step_issued_bytes = 0
    max_offload_step_issued_bytes = 0
    max_summary_attempted_bytes = 0
    max_offload_summary_attempted_bytes = 0
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
        if action in {"offload_advise_cpu_issued", "offload_prefetch_cpu_issued"}:
            offload_issued += 1
        if action == "trace_offload_candidate":
            trace_offload_candidates += 1
        if action == "budget_reject":
            budget_rejects += 1
        if action == "offload_budget_reject":
            offload_budget_rejects += 1
        if action == "step_summary":
            max_step_issued_bytes = max(
                max_step_issued_bytes,
                int(record.get("issued_bytes") or 0),
            )
            max_summary_attempted_bytes = max(
                max_summary_attempted_bytes,
                int(record.get("attempted_bytes") or 0),
            )
        if action == "offload_step_summary":
            max_offload_step_issued_bytes = max(
                max_offload_step_issued_bytes,
                int(record.get("issued_bytes") or 0),
            )
            max_offload_summary_attempted_bytes = max(
                max_offload_summary_attempted_bytes,
                int(record.get("attempted_bytes") or 0),
            )

    return {
        "prefetch_trace_records": len(records),
        "prefetch_action_counts": dict(action_counts),
        "prefetch_role_counts": dict(role_counts),
        "prefetch_issued_records": prefetch_issued,
        "trace_candidate_records": trace_candidates,
        "offload_issued_records": offload_issued,
        "trace_offload_candidate_records": trace_offload_candidates,
        "prefetch_candidate_or_issued_records": issued_records,
        "prefetch_candidate_or_issued_bytes": issued_bytes,
        "budget_reject_records": budget_rejects,
        "offload_budget_reject_records": offload_budget_rejects,
        "prefetch_layer_count": len(layers),
        "prefetch_expert_count": len(experts),
        "max_step_issued_bytes": max_step_issued_bytes,
        "max_offload_step_issued_bytes": max_offload_step_issued_bytes,
        "max_summary_attempted_bytes": max_summary_attempted_bytes,
        "max_offload_summary_attempted_bytes": max_offload_summary_attempted_bytes,
    }


def validate(
    *,
    metrics_json: Path | None,
    allocator_log: Path | None,
    prefetch_trace: Path,
    moe_routing_jsonl: Path | None,
    weight_map_jsonl: Path | None,
    plan_json: Path | None,
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
    max_offload_step_issued_bytes = as_int(
        trace_summary.get("max_offload_step_issued_bytes")
    )
    prefetch_issued_records = as_int(trace_summary.get("prefetch_issued_records"))
    trace_candidate_records = as_int(trace_summary.get("trace_candidate_records"))
    offload_issued_records = as_int(trace_summary.get("offload_issued_records"))
    trace_offload_candidate_records = as_int(
        trace_summary.get("trace_offload_candidate_records")
    )
    candidate_or_issued = as_int(
        trace_summary.get("prefetch_candidate_or_issued_records")
    )
    plan = load_json(plan_json) if plan_json and plan_json.is_file() else {}
    prefetch_plan_records = as_int(plan.get("prefetch_plan_records"))
    offload_plan_records = as_int(plan.get("offload_plan_records"))

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
            "plan_json_present",
            plan_json is not None and plan_json.is_file(),
            f"plan_json={plan_json}",
        ),
        check(
            "prefetch_plan_records_present",
            prefetch_plan_records is not None and prefetch_plan_records > 0,
            f"prefetch_plan_records={prefetch_plan_records}",
        ),
        check(
            "offload_plan_records_present",
            args.disable_offload
            or (offload_plan_records is not None and offload_plan_records > 0),
            f"offload_plan_records={offload_plan_records}",
        ),
        check(
            "offload_action_present",
            args.disable_offload
            or (
                args.offload_mode == "trace_only"
                and trace_offload_candidate_records is not None
                and trace_offload_candidate_records > 0
            )
            or (
                args.offload_mode != "trace_only"
                and offload_issued_records is not None
                and offload_issued_records > 0
            ),
            (
                f"offload_mode={args.offload_mode} "
                f"offload_issued_records={offload_issued_records} "
                f"trace_offload_candidate_records={trace_offload_candidate_records}"
            ),
        ),
        check(
            "offload_step_bytes_within_budget",
            args.disable_offload
            or (
                max_offload_step_issued_bytes is not None
                and max_offload_step_issued_bytes <= args.offload_max_bytes_per_step
            ),
            (
                f"max_offload_step_issued_bytes={max_offload_step_issued_bytes} "
                f"budget={args.offload_max_bytes_per_step}"
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
        "plan_json": str(plan_json) if plan_json else None,
        "bench_log": str(bench_log) if bench_log else None,
        "mode": args.mode,
        "offload_mode": args.offload_mode,
        "offload_enabled": not args.disable_offload,
        "max_bytes_per_step": args.max_bytes_per_step,
        "offload_max_bytes_per_step": args.offload_max_bytes_per_step,
        "max_experts_per_layer": args.max_experts_per_layer,
        "target_roles": args.target_roles,
        "prefetch_plan_records": prefetch_plan_records,
        "offload_plan_records": offload_plan_records,
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
    print(f" Plan: {summary['plan_json']}")
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
        "- offload: "
        f"enabled={summary.get('offload_enabled')} "
        f"mode={summary.get('offload_mode')} "
        f"issued={summary.get('offload_issued_records')} "
        f"trace_candidates={summary.get('trace_offload_candidate_records')}"
    )
    print(
        "- plan: "
        f"prefetch_records={summary.get('prefetch_plan_records')} "
        f"offload_records={summary.get('offload_plan_records')}"
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
        f"rejects={summary.get('budget_reject_records')} "
        f"max_offload_step_issued_bytes={summary.get('max_offload_step_issued_bytes')} "
        f"offload_max_bytes_per_step={summary.get('offload_max_bytes_per_step')} "
        f"offload_rejects={summary.get('offload_budget_reject_records')}"
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
    if args.offload_max_bytes_per_step < 0:
        raise SystemExit("--offload-max-bytes-per-step must be non-negative")
    if args.offload_max_experts_per_layer < 0:
        raise SystemExit("--offload-max-experts-per-layer must be non-negative")
    if args.prompts < 0:
        raise SystemExit("--prompts must be non-negative")
    if args.plan_prompts < 0:
        raise SystemExit("--plan-prompts must be non-negative")

    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir()
    (
        metrics_json,
        allocator_log,
        prefetch_trace,
        moe_routing_jsonl,
        weight_map_jsonl,
        plan_json,
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
            plan_json,
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
        plan_json=plan_json,
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

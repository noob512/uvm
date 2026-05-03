#!/usr/bin/env python3
"""Run or validate a Stage K global UVM pool coordinator success check."""

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
            "Check whether Stage K coordinates high-level UVM pool actions. "
            "The default self-test is fast and validates global/per-pool "
            "grant and would-deny behavior. Use --gpu-run for an integrated "
            "Stage I + Stage J probe."
        )
    )
    parser.add_argument("--run-dir", help="Existing or new Stage K run directory.")
    parser.add_argument("--output-json", help="Where to write the check summary JSON.")
    parser.add_argument("--trace-jsonl", help="Existing or new Stage K JSONL trace.")
    parser.add_argument("--bench-log", help="Existing or new benchmark log.")
    parser.add_argument("--server-log", help="Existing or new server log.")
    parser.add_argument(
        "--mode",
        choices=("trace_only", "enforce"),
        default="trace_only",
        help="Stage K mode. trace_only records would-deny without skipping actions.",
    )
    parser.add_argument("--global-bytes-per-step", type=int, default=1024)
    parser.add_argument("--weight-bytes-per-step", type=int, default=1024)
    parser.add_argument("--kv-bytes-per-step", type=int, default=256)
    parser.add_argument("--scratch-bytes-per-step", type=int, default=128)
    parser.add_argument("--priority", default="kv,weights,scratch")
    parser.add_argument(
        "--gpu-run",
        action="store_true",
        help="Run a full vLLM probe that wires Stage I and Stage J to Stage K.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Validate an existing trace; use --trace-jsonl or --run-dir.",
    )
    parser.add_argument("--plan-json", help="Existing or new Stage H plan JSON.")
    parser.add_argument("--prompts", type=int, default=1)
    parser.add_argument("--plan-prompts", type=int, default=1)
    parser.add_argument("--request-rate", default="5")
    parser.add_argument("--output-len", default="128")
    return parser.parse_args()


def default_run_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return Path(f"/tmp/vllm_stage_k_success_check_{stamp}")


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


def parse_failed_requests(bench_log: Path | None) -> int | None:
    if bench_log is None or not bench_log.is_file():
        return None
    text = bench_log.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Failed requests:\s+(\d+)", text)
    if not matches:
        return None
    return int(matches[-1])


def resolve_existing_inputs(args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    trace_jsonl = Path(args.trace_jsonl) if args.trace_jsonl else None
    bench_log = Path(args.bench_log) if args.bench_log else None
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if trace_jsonl is None:
            candidate = run_dir / "vllm_uvm_pool_coordinator_stage_k.jsonl"
            if candidate.is_file():
                trace_jsonl = candidate
        if bench_log is None:
            candidate = run_dir / "vllm_bench_stage_k.log"
            if candidate.is_file():
                bench_log = candidate
    return trace_jsonl, bench_log


def coordinator_env(args: argparse.Namespace, trace_jsonl: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    env.update(
        {
            "VLLM_UVM_POOL_COORDINATOR_ENABLE": "1",
            "VLLM_UVM_POOL_COORDINATOR_MODE": args.mode,
            "VLLM_UVM_POOL_COORDINATOR_TRACE_FILE": str(trace_jsonl),
            "VLLM_UVM_POOL_COORDINATOR_GLOBAL_BYTES_PER_STEP": str(
                args.global_bytes_per_step
            ),
            "VLLM_UVM_POOL_COORDINATOR_WEIGHT_BYTES_PER_STEP": str(
                args.weight_bytes_per_step
            ),
            "VLLM_UVM_POOL_COORDINATOR_KV_BYTES_PER_STEP": str(
                args.kv_bytes_per_step
            ),
            "VLLM_UVM_POOL_COORDINATOR_SCRATCH_BYTES_PER_STEP": str(
                args.scratch_bytes_per_step
            ),
            "VLLM_UVM_POOL_COORDINATOR_PRIORITY": args.priority,
        }
    )
    return env


def run_self_test(args: argparse.Namespace, run_dir: Path) -> tuple[Path, Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_jsonl = Path(args.trace_jsonl) if args.trace_jsonl else (
        run_dir / "vllm_uvm_pool_coordinator_stage_k.jsonl"
    )
    bench_log = Path(args.bench_log) if args.bench_log else (
        run_dir / "vllm_bench_stage_k.log"
    )
    runner_log = run_dir / "stage_k_self_test_runner.log"
    bench_log.write_text("Failed requests:                         0\n", encoding="utf-8")

    code = r'''
import os

from vllm.device_allocator.uvm_pool_coordinator import (
    record_uvm_pool_pressure,
    request_uvm_pool_action,
)

trace_file = os.environ["VLLM_UVM_POOL_COORDINATOR_TRACE_FILE"]
try:
    os.unlink(trace_file)
except FileNotFoundError:
    pass

mode = os.environ["VLLM_UVM_POOL_COORDINATOR_MODE"]
record_uvm_pool_pressure(
    pool="scratch",
    pressure_bytes=256,
    pressure_ratio=2.0,
    action_queue_depth=1,
    metadata={"stage": "stage_k_self_test"},
)
d1 = request_uvm_pool_action(
    pool="weights",
    action="expert_prefetch_gpu",
    requested_bytes=768,
    scope_key="decode_step:0",
    metadata={"stage": "stage_k_self_test"},
)
d2 = request_uvm_pool_action(
    pool="kv",
    action="prefix_cache_evict",
    requested_bytes=512,
    scope_key="decode_step:0",
    metadata={"stage": "stage_k_self_test"},
)
d3 = request_uvm_pool_action(
    pool="scratch",
    action="device_direct_admission",
    requested_bytes=256,
    scope_key="decode_step:1",
    metadata={"stage": "stage_k_self_test"},
)
assert d1.allowed and not d1.would_deny
assert d2.would_deny
assert d3.would_deny
if mode == "enforce":
    assert not d2.allowed and not d3.allowed
else:
    assert d2.allowed and d3.allowed
'''

    cmd = ["uv", "run", "--directory", str(SCRIPT_DIR), "python", "-c", code]
    print("===========================================================")
    print(" Stage K Success Check: running coordinator self-test")
    print(f" Output dir: {run_dir}")
    print(f" Trace JSONL: {trace_jsonl}")
    print(f" Mode: {args.mode}")
    print("===========================================================")
    with runner_log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            cwd=SCRIPT_DIR,
            env=coordinator_env(args, trace_jsonl),
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
            f"Stage K self-test failed with exit code {rc}; log={runner_log}"
        )
    return trace_jsonl, bench_log, runner_log


def run_gpu_probe(args: argparse.Namespace, run_dir: Path) -> tuple[Path, Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_jsonl = Path(args.trace_jsonl) if args.trace_jsonl else (
        run_dir / "vllm_uvm_pool_coordinator_stage_k.jsonl"
    )
    bench_log = Path(args.bench_log) if args.bench_log else (
        run_dir / "vllm_bench_stage_k.log"
    )
    runner_log = run_dir / "stage_k_success_check_runner.log"
    plan_json = Path(args.plan_json) if args.plan_json else (
        run_dir / "vllm_stage_k_weight_expert_plan.json"
    )
    planning_weight_map = run_dir / "vllm_uvm_weight_regions_stage_k_planning.jsonl"
    planning_moe = run_dir / "vllm_uvm_moe_routing_stage_k_planning.jsonl"
    planning_metrics = run_dir / "vllm_stage_k_allocator_metrics_planning.json"
    planning_allocator = run_dir / "vllm_uvm_allocator_trace_stage_k_planning.log"
    plan_summary = run_dir / "vllm_stage_k_weight_expert_plan_summary.json"

    planning_cmd = [
        "./run_kv_fault_ratio.sh",
        "--mode",
        "trace",
        "--allocator-log",
        str(planning_allocator),
        "--trace-log",
        str(run_dir / "uvm_kv_fault_stats_stage_k_planning.log"),
        "--address-log",
        str(run_dir / "vllm_uvm_address_regions_stage_k_planning.log"),
        "--server-log",
        str(run_dir / "vllm_server_stage_k_planning.log"),
        "--bench-log",
        str(run_dir / "vllm_bench_stage_k_planning.log"),
        "--uvm-weight-map-enable",
        "1",
        "--uvm-weight-map-file",
        str(planning_weight_map),
        "--uvm-moe-routing-trace-enable",
        "1",
        "--uvm-moe-routing-trace-file",
        str(planning_moe),
        "--uvm-pool-registry-enable",
        "1",
        "--gap-watch-metrics-summary-json",
        str(planning_metrics),
        "--prompts",
        str(args.plan_prompts),
        "--request-rate",
        str(args.request_rate),
        "--output-len",
        str(args.output_len),
    ]

    print("===========================================================")
    print(" Stage K Success Check: planning Stage I weight actions")
    print(f" Output dir: {run_dir}")
    print(f" Coordinator trace: {trace_jsonl}")
    print("===========================================================")
    with runner_log.open("w", encoding="utf-8") as handle:
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
        raise SystemExit(f"Stage K planning probe failed with exit code {rc}")

    subprocess.run(
        [
            "python3",
            str(SCRIPT_DIR / "plan_stage_h_weight_expert_actions.py"),
            "--weight-map",
            str(planning_weight_map),
            "--moe-routing-trace",
            str(planning_moe),
            "--plan-json",
            str(plan_json),
            "--summary-json",
            str(plan_summary),
            "--target-roles",
            "moe_gate_up,moe_down",
            "--hot-top-k",
            "1024",
            "--cold-bottom-k",
            "1024",
            "--require-routing",
        ],
        cwd=SCRIPT_DIR,
        check=True,
    )

    main_cmd = [
        "./run_kv_fault_ratio.sh",
        "--mode",
        "trace",
        "--allocator-log",
        str(run_dir / "vllm_uvm_allocator_trace_stage_k.log"),
        "--trace-log",
        str(run_dir / "uvm_kv_fault_stats_stage_k.log"),
        "--address-log",
        str(run_dir / "vllm_uvm_address_regions_stage_k.log"),
        "--server-log",
        str(run_dir / "vllm_server_stage_k.log"),
        "--bench-log",
        str(bench_log),
        "--uvm-kv-runtime-enable",
        "1",
        "--uvm-kv-runtime-mode",
        "trace_only",
        "--uvm-kv-runtime-budget-blocks",
        "1",
        "--uvm-kv-runtime-trace-file",
        str(run_dir / "vllm_uvm_kv_runtime_stage_k.jsonl"),
        "--uvm-kv-runtime-prefix-evict-enable",
        "1",
        "--uvm-weight-prefetch-enable",
        "1",
        "--uvm-weight-prefetch-mode",
        "trace_only",
        "--uvm-weight-prefetch-trace-file",
        str(run_dir / "vllm_uvm_weight_prefetch_stage_k.jsonl"),
        "--uvm-weight-prefetch-plan-file",
        str(plan_json),
        "--uvm-weight-prefetch-require-plan",
        "1",
        "--uvm-weight-offload-enable",
        "1",
        "--uvm-weight-offload-mode",
        "trace_only",
        "--uvm-weight-offload-plan-file",
        str(plan_json),
        "--uvm-pool-coordinator-enable",
        "1",
        "--uvm-pool-coordinator-mode",
        args.mode,
        "--uvm-pool-coordinator-trace-file",
        str(trace_jsonl),
        "--uvm-pool-coordinator-global-bytes-per-step",
        str(args.global_bytes_per_step),
        "--uvm-pool-coordinator-weight-bytes-per-step",
        str(args.weight_bytes_per_step),
        "--uvm-pool-coordinator-kv-bytes-per-step",
        str(args.kv_bytes_per_step),
        "--uvm-pool-coordinator-scratch-bytes-per-step",
        str(args.scratch_bytes_per_step),
        "--uvm-pool-coordinator-priority",
        args.priority,
        "--uvm-pool-registry-enable",
        "1",
        "--prompts",
        str(args.prompts),
        "--request-rate",
        str(args.request_rate),
        "--output-len",
        str(args.output_len),
    ]

    with runner_log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            main_cmd,
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
        raise SystemExit(f"Stage K integrated probe failed with exit code {rc}")
    return trace_jsonl, bench_log, runner_log


def summarize_trace(records: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    pools: Counter[str] = Counter()
    would_deny_records = 0
    denied_records = 0
    request_records = 0
    pressure_records = 0
    config: dict[str, Any] = {}
    summary: dict[str, Any] = {}

    for record in records:
        action = str(record.get("action", "unknown"))
        action_counts[action] += 1
        if action == "coordinator_config":
            config = record
        if action == "coordinator_summary":
            summary = record
        if action == "coordinator_pressure":
            pressure_records += 1
            pool = str(record.get("pool", "unknown"))
            pools[pool] += 1
        if action == "coordinator_request":
            request_records += 1
            pool = str(record.get("pool", "unknown"))
            pools[pool] += 1
            outcome_counts[str(record.get("outcome", "unknown"))] += 1
            if record.get("would_deny") is True:
                would_deny_records += 1
            if record.get("allowed") is False:
                denied_records += 1

    return {
        "trace_records": len(records),
        "action_counts": dict(action_counts),
        "outcome_counts": dict(outcome_counts),
        "pools": dict(pools),
        "request_records": request_records,
        "pressure_records": pressure_records,
        "would_deny_records": would_deny_records,
        "denied_records": denied_records,
        "config": config,
        "summary": summary,
    }


def validate(
    *,
    trace_jsonl: Path,
    bench_log: Path | None,
    runner_log: Path | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    records = read_jsonl(trace_jsonl)
    trace_summary = summarize_trace(records)
    config = trace_summary["config"]
    failed_requests = parse_failed_requests(bench_log)
    pools = set(trace_summary["pools"])
    expected_pools = {"weights", "kv"} if args.gpu_run else {"weights", "kv", "scratch"}

    checks = [
        check("trace_jsonl_present", trace_jsonl.is_file(), f"trace_jsonl={trace_jsonl}"),
        check(
            "trace_records_present",
            trace_summary["trace_records"] > 0,
            f"trace_records={trace_summary['trace_records']}",
        ),
        check("coordinator_config_present", bool(config), f"config={config}"),
        check("coordinator_enabled", config.get("enabled") is True, f"config={config}"),
        check(
            "coordinator_mode_matches",
            config.get("mode") == args.mode,
            f"mode={config.get('mode')} expected={args.mode}",
        ),
        check(
            "request_records_present",
            trace_summary["request_records"] > 0,
            f"request_records={trace_summary['request_records']}",
        ),
        check(
            "expected_pools_visible",
            expected_pools.issubset(pools),
            f"pools={sorted(pools)} expected_subset={sorted(expected_pools)}",
        ),
        check(
            "would_deny_records_present",
            trace_summary["would_deny_records"] > 0,
            f"would_deny_records={trace_summary['would_deny_records']}",
        ),
        check(
            "deny_records_present_in_enforce_mode",
            args.mode != "enforce" or trace_summary["denied_records"] > 0,
            f"denied_records={trace_summary['denied_records']}",
        ),
        check(
            "benchmark_no_failed_requests_in_trace_only",
            args.mode != "trace_only" or failed_requests in (None, 0),
            f"failed_requests={failed_requests}",
        ),
    ]

    if runner_log and runner_log.is_file():
        text = runner_log.read_text(encoding="utf-8", errors="replace").lower()
        checks.append(
            check(
                "runner_log_clean",
                "server exited early" not in text and "traceback" not in text,
                f"runner_log={runner_log}",
            )
        )

    passed = all(item["passed"] for item in checks)
    output_json = Path(args.output_json) if args.output_json else (
        trace_jsonl.parent / "stage_k_success_check.json"
    )
    result = {
        "passed": passed,
        "trace_jsonl": str(trace_jsonl),
        "bench_log": str(bench_log) if bench_log else None,
        "failed_requests": failed_requests,
        "trace_summary": trace_summary,
        "checks": checks,
        "output_json": str(output_json),
    }
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def print_summary(result: dict[str, Any]) -> None:
    trace = result["trace_summary"]
    status = "PASS" if result["passed"] else "FAIL"
    print("===========================================================")
    print(f" Stage K Success Check: {status}")
    print(f" Trace JSONL: {result['trace_jsonl']}")
    print("===========================================================")
    print(f"- pools={trace['pools']}")
    print(f"- action_counts={trace['action_counts']}")
    print(f"- outcome_counts={trace['outcome_counts']}")
    print(
        "- requests="
        f"{trace['request_records']} would_deny={trace['would_deny_records']} "
        f"denied={trace['denied_records']} pressure={trace['pressure_records']}"
    )
    print(f"- failed_requests={result['failed_requests']}")
    print("- checks:")
    for item in result["checks"]:
        print(f"  {item['name']}={item['passed']} ({item['detail']})")
    print(f"- check_json={result['output_json']}")


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir()
    runner_log: Path | None = None
    if args.skip_run:
        trace_jsonl, bench_log = resolve_existing_inputs(args)
        if trace_jsonl is None:
            raise SystemExit("--skip-run requires --trace-jsonl or --run-dir")
    elif args.gpu_run:
        trace_jsonl, bench_log, runner_log = run_gpu_probe(args, run_dir)
    else:
        trace_jsonl, bench_log, runner_log = run_self_test(args, run_dir)

    result = validate(
        trace_jsonl=trace_jsonl,
        bench_log=bench_log,
        runner_log=runner_log,
        args=args,
    )
    print_summary(result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

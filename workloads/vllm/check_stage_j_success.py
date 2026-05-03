#!/usr/bin/env python3
"""Run or validate a Stage J runtime KV pressure policy success check."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether Stage J KV runtime pressure policy is wired at the "
            "vLLM block-manager boundary. Default mode is trace_only with a tiny "
            "runtime block budget, so it emits would-evict/would-deny records "
            "while preserving benchmark correctness."
        )
    )
    parser.add_argument("--run-dir", help="Existing or new Stage J run directory.")
    parser.add_argument("--output-json", help="Where to write the check summary JSON.")
    parser.add_argument("--trace-jsonl", help="Existing or new Stage J runtime JSONL.")
    parser.add_argument("--bench-log", help="Existing or new benchmark log.")
    parser.add_argument("--server-log", help="Existing or new server log.")
    parser.add_argument(
        "--mode",
        choices=("trace_only", "enforce"),
        default="trace_only",
        help="Stage J mode. Default trace_only is non-disruptive.",
    )
    parser.add_argument(
        "--budget-blocks",
        type=int,
        default=1,
        help="Runtime KV block budget. Default 1 intentionally creates pressure.",
    )
    parser.add_argument(
        "--budget-bytes",
        type=int,
        default=0,
        help="Runtime KV byte budget. Ignored when --budget-blocks is > 0.",
    )
    parser.add_argument("--candidate-limit", type=int, default=16)
    parser.add_argument("--prefix-evict-enable", action="store_true")
    parser.add_argument("--prefix-evict-max-blocks", type=int, default=0)
    parser.add_argument(
        "--eviction-policy",
        choices=("lru_prefix_cache", "scheduler_aware"),
        default="lru_prefix_cache",
    )
    parser.add_argument("--prompts", type=int, default=1)
    parser.add_argument("--request-rate", default="5")
    parser.add_argument("--output-len", default="128")
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Do not start a new vLLM run; requires --trace-jsonl or --run-dir.",
    )
    parser.add_argument(
        "--require-prefix-eviction",
        action="store_true",
        help=(
            "Require a real prefix-cache eviction/reuse event. This is optional "
            "because small one-shot probes often have enough free blocks."
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "Run a fast in-process block-manager test that seeds prefix-cache "
            "free blocks and verifies the Stage J executor evicts only safe blocks."
        ),
    )
    return parser.parse_args()


def default_run_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return Path(f"/tmp/vllm_stage_j_success_check_{stamp}")


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


def resolve_existing_inputs(
    args: argparse.Namespace,
) -> tuple[Path | None, Path | None, Path | None]:
    trace_jsonl = Path(args.trace_jsonl) if args.trace_jsonl else None
    bench_log = Path(args.bench_log) if args.bench_log else None
    server_log = Path(args.server_log) if args.server_log else None
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if trace_jsonl is None:
            candidate = run_dir / "vllm_uvm_kv_runtime_stage_j.jsonl"
            if candidate.is_file():
                trace_jsonl = candidate
        if bench_log is None:
            candidate = run_dir / "vllm_bench_stage_j.log"
            if candidate.is_file():
                bench_log = candidate
        if server_log is None:
            candidate = run_dir / "vllm_server_stage_j.log"
            if candidate.is_file():
                server_log = candidate
    return trace_jsonl, bench_log, server_log


def run_experiment(
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_jsonl = Path(args.trace_jsonl) if args.trace_jsonl else (
        run_dir / "vllm_uvm_kv_runtime_stage_j.jsonl"
    )
    bench_log = Path(args.bench_log) if args.bench_log else (
        run_dir / "vllm_bench_stage_j.log"
    )
    server_log = Path(args.server_log) if args.server_log else (
        run_dir / "vllm_server_stage_j.log"
    )
    runner_log = run_dir / "stage_j_success_check_runner.log"

    cmd = [
        "./run_kv_fault_ratio.sh",
        "--mode",
        "trace",
        "--allocator-log",
        str(run_dir / "vllm_uvm_allocator_trace_stage_j.log"),
        "--trace-log",
        str(run_dir / "uvm_kv_fault_stats_stage_j.log"),
        "--address-log",
        str(run_dir / "vllm_uvm_address_regions_stage_j.log"),
        "--server-log",
        str(server_log),
        "--bench-log",
        str(bench_log),
        "--uvm-kv-budget-bytes",
        "0",
        "--uvm-kv-runtime-enable",
        "1",
        "--uvm-kv-runtime-mode",
        args.mode,
        "--uvm-kv-runtime-budget-bytes",
        str(args.budget_bytes),
        "--uvm-kv-runtime-budget-blocks",
        str(args.budget_blocks),
        "--uvm-kv-runtime-trace-file",
        str(trace_jsonl),
        "--uvm-kv-runtime-eviction-policy",
        args.eviction_policy,
        "--uvm-kv-runtime-candidate-limit",
        str(args.candidate_limit),
        "--uvm-kv-runtime-prefix-evict-enable",
        "1" if args.prefix_evict_enable else "0",
        "--uvm-kv-runtime-prefix-evict-max-blocks",
        str(args.prefix_evict_max_blocks),
        "--prompts",
        str(args.prompts),
        "--request-rate",
        str(args.request_rate),
        "--output-len",
        str(args.output_len),
    ]

    print("===========================================================")
    print(" Stage J Success Check: running runtime KV pressure probe")
    print(f" Output dir: {run_dir}")
    print(f" Trace JSONL: {trace_jsonl}")
    print(f" Mode: {args.mode}")
    print(f" Runtime budget blocks: {args.budget_blocks}")
    print(f" Runtime budget bytes: {args.budget_bytes}")
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
        raise SystemExit(f"Stage J probe failed with exit code {rc}; log={runner_log}")
    return trace_jsonl, bench_log, server_log, runner_log


def run_self_test(
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_jsonl = Path(args.trace_jsonl) if args.trace_jsonl else (
        run_dir / "vllm_uvm_kv_runtime_stage_j.jsonl"
    )
    bench_log = Path(args.bench_log) if args.bench_log else (
        run_dir / "vllm_bench_stage_j.log"
    )
    server_log = Path(args.server_log) if args.server_log else (
        run_dir / "vllm_server_stage_j.log"
    )
    runner_log = run_dir / "stage_j_self_test_runner.log"
    bench_log.write_text("Failed requests:                         0\n", encoding="utf-8")
    server_log.write_text("Stage J self-test does not start a server.\n", encoding="utf-8")

    code = r'''
import os
import torch

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)

trace_file = os.environ["VLLM_UVM_KV_RUNTIME_TRACE_FILE"]
try:
    os.unlink(trace_file)
except FileNotFoundError:
    pass

config = KVCacheConfig(
    num_blocks=8,
    kv_cache_tensors=[KVCacheTensor(size=8 * 1024, shared_by=["layer.0"])],
    kv_cache_groups=[
        KVCacheGroupSpec(
            layer_names=["layer.0"],
            kv_cache_spec=FullAttentionSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float16,
            ),
        )
    ],
)
manager = KVCacheManager(
    config,
    max_model_len=128,
    hash_block_size=16,
    enable_caching=True,
)
block_pool = manager.block_pool
free_block = block_pool.free_block_queue.get_all_free_blocks()[0]
assert free_block.ref_cnt == 0 and not free_block.is_null
block_hash = make_block_hash_with_group_id(BlockHash(b"stage-j-self-test"), 0)
free_block.block_hash = block_hash
block_pool.cached_block_hash_to_block.insert(block_hash, free_block)
allowed = manager.uvm_kv_runtime_policy.should_allow_allocation(
    request_id="stage-j-self-test",
    num_blocks_to_allocate=2,
    block_pool=block_pool,
)
if os.environ["VLLM_UVM_KV_RUNTIME_MODE"] == "enforce":
    assert not allowed
else:
    assert allowed
assert free_block.block_hash is None
'''

    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    env.update(
        {
            "VLLM_UVM_KV_RUNTIME_ENABLE": "1",
            "VLLM_UVM_KV_RUNTIME_MODE": args.mode,
            "VLLM_UVM_KV_RUNTIME_BUDGET_BYTES": str(args.budget_bytes),
            "VLLM_UVM_KV_RUNTIME_BUDGET_BLOCKS": str(args.budget_blocks),
            "VLLM_UVM_KV_RUNTIME_TRACE_FILE": str(trace_jsonl),
            "VLLM_UVM_KV_RUNTIME_EVICTION_POLICY": args.eviction_policy,
            "VLLM_UVM_KV_RUNTIME_CANDIDATE_LIMIT": str(args.candidate_limit),
            "VLLM_UVM_KV_RUNTIME_PREFIX_EVICT_ENABLE": "1",
            "VLLM_UVM_KV_RUNTIME_PREFIX_EVICT_MAX_BLOCKS": str(
                args.prefix_evict_max_blocks or 1
            ),
        }
    )
    args.prefix_evict_enable = True
    args.prefix_evict_max_blocks = args.prefix_evict_max_blocks or 1

    cmd = [
        "uv",
        "run",
        "--directory",
        str(SCRIPT_DIR),
        "python",
        "-c",
        code,
    ]
    print("===========================================================")
    print(" Stage J Success Check: running block-manager self-test")
    print(f" Output dir: {run_dir}")
    print(f" Trace JSONL: {trace_jsonl}")
    print(f" Mode: {args.mode}")
    print("===========================================================")

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
        raise SystemExit(
            f"Stage J self-test failed with exit code {rc}; log={runner_log}"
        )
    return trace_jsonl, bench_log, server_log, runner_log


def summarize_trace(records: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts: Counter[str] = Counter()
    over_budget_records = 0
    candidate_records = 0
    unsafe_candidate_records = 0
    prefix_eviction_records = 0
    unsafe_prefix_eviction_records = 0
    prefix_evict_attempt_records = 0
    prefix_evict_success_records = 0
    prefix_evict_success_blocks = 0
    prefix_evict_noop_records = 0
    prefix_evict_failed_records = 0
    deny_allocation_records = 0
    deny_records = 0
    allocation_pressure_records = 0
    max_pressure_blocks = 0
    max_pressure_bytes = 0
    runtime_config: dict[str, Any] = {}
    runtime_summary: dict[str, Any] = {}

    for record in records:
        action = str(record.get("action", "unknown"))
        action_counts[action] += 1
        if action == "runtime_config":
            runtime_config = record
        if action == "runtime_summary":
            runtime_summary = record
        if action == "allocation_pressure":
            allocation_pressure_records += 1
        if action == "would_deny_allocation":
            deny_records += 1
        if action == "deny_allocation":
            deny_allocation_records += 1
        if action == "prefix_evict_attempt":
            prefix_evict_attempt_records += 1
        if action == "prefix_evict_success":
            prefix_evict_success_records += 1
            prefix_evict_success_blocks += as_int(record.get("evicted_blocks")) or 0
        if action == "prefix_evict_noop":
            prefix_evict_noop_records += 1
        if action == "prefix_evict_failed":
            prefix_evict_failed_records += 1
        if bool(record.get("over_budget")):
            over_budget_records += 1
        pressure_blocks = as_int(record.get("pressure_blocks")) or 0
        pressure_bytes = as_int(record.get("pressure_bytes")) or 0
        max_pressure_blocks = max(max_pressure_blocks, pressure_blocks)
        max_pressure_bytes = max(max_pressure_bytes, pressure_bytes)
        if action in {"would_evict_candidate", "would_reuse_free_block"}:
            candidate_records += 1
            if record.get("safe_ref_cnt_zero") is not True:
                unsafe_candidate_records += 1
        if action == "evict_prefix_cache_block":
            prefix_eviction_records += 1
            if record.get("safe_ref_cnt_zero") is not True:
                unsafe_prefix_eviction_records += 1

    return {
        "trace_records": len(records),
        "action_counts": dict(action_counts),
        "runtime_config": runtime_config,
        "runtime_summary": runtime_summary,
        "allocation_pressure_records": allocation_pressure_records,
        "over_budget_records": over_budget_records,
        "would_deny_records": deny_records,
        "deny_allocation_records": deny_allocation_records,
        "candidate_records": candidate_records,
        "unsafe_candidate_records": unsafe_candidate_records,
        "prefix_evict_attempt_records": prefix_evict_attempt_records,
        "prefix_evict_success_records": prefix_evict_success_records,
        "prefix_evict_success_blocks": prefix_evict_success_blocks,
        "prefix_evict_noop_records": prefix_evict_noop_records,
        "prefix_evict_failed_records": prefix_evict_failed_records,
        "prefix_eviction_records": prefix_eviction_records,
        "unsafe_prefix_eviction_records": unsafe_prefix_eviction_records,
        "max_pressure_blocks": max_pressure_blocks,
        "max_pressure_bytes": max_pressure_bytes,
    }


def validate(
    *,
    trace_jsonl: Path,
    bench_log: Path | None,
    server_log: Path | None,
    runner_log: Path | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    records = read_jsonl(trace_jsonl)
    trace_summary = summarize_trace(records)
    config = trace_summary["runtime_config"]
    runtime_summary = trace_summary["runtime_summary"]
    failed_requests = parse_failed_requests(bench_log)
    budget_blocks = as_int(config.get("budget_blocks"))
    mode = config.get("mode")
    enabled = config.get("enabled")
    bytes_per_block = as_int(config.get("bytes_per_block"))
    prefix_evict_enabled = config.get("prefix_evict_enable")

    checks = [
        check(
            "trace_jsonl_present",
            trace_jsonl.is_file(),
            f"trace_jsonl={trace_jsonl}",
        ),
        check(
            "trace_records_present",
            trace_summary["trace_records"] > 0,
            f"trace_records={trace_summary['trace_records']}",
        ),
        check(
            "runtime_config_present",
            bool(config),
            f"runtime_config={config}",
        ),
        check(
            "runtime_enabled",
            enabled is True,
            f"enabled={enabled}",
        ),
        check(
            "runtime_mode_matches",
            mode == args.mode,
            f"mode={mode} expected={args.mode}",
        ),
        check(
            "runtime_budget_blocks_matches",
            budget_blocks == args.budget_blocks or args.budget_blocks == 0,
            f"budget_blocks={budget_blocks} expected={args.budget_blocks}",
        ),
        check(
            "bytes_per_block_positive",
            bytes_per_block is not None and bytes_per_block > 0,
            f"bytes_per_block={bytes_per_block}",
        ),
        check(
            "allocation_pressure_records_present",
            trace_summary["allocation_pressure_records"] > 0,
            (
                "allocation_pressure_records="
                f"{trace_summary['allocation_pressure_records']}"
            ),
        ),
        check(
            "over_budget_pressure_observed",
            trace_summary["over_budget_records"] > 0,
            f"over_budget_records={trace_summary['over_budget_records']}",
        ),
        check(
            "would_deny_records_present",
            trace_summary["would_deny_records"] > 0,
            f"would_deny_records={trace_summary['would_deny_records']}",
        ),
        check(
            "deny_records_present_in_enforce_mode",
            args.mode != "enforce" or trace_summary["deny_allocation_records"] > 0,
            f"deny_allocation_records={trace_summary['deny_allocation_records']}",
        ),
        check(
            "candidate_records_present",
            args.candidate_limit == 0 or trace_summary["candidate_records"] > 0,
            f"candidate_records={trace_summary['candidate_records']}",
        ),
        check(
            "runtime_summary_present",
            bool(runtime_summary),
            f"runtime_summary={runtime_summary}",
        ),
        check(
            "prefix_evict_config_matches",
            prefix_evict_enabled is bool(args.prefix_evict_enable),
            (
                f"prefix_evict_enable={prefix_evict_enabled} "
                f"expected={args.prefix_evict_enable}"
            ),
        ),
        check(
            "prefix_evict_attempted_if_enabled",
            not args.prefix_evict_enable
            or trace_summary["prefix_evict_attempt_records"] > 0,
            (
                "prefix_evict_attempt_records="
                f"{trace_summary['prefix_evict_attempt_records']}"
            ),
        ),
        check(
            "prefix_evict_no_failures",
            trace_summary["prefix_evict_failed_records"] == 0,
            (
                "prefix_evict_failed_records="
                f"{trace_summary['prefix_evict_failed_records']}"
            ),
        ),
        check(
            "candidate_records_are_safe",
            trace_summary["unsafe_candidate_records"] == 0,
            (
                "unsafe_candidate_records="
                f"{trace_summary['unsafe_candidate_records']}"
            ),
        ),
        check(
            "prefix_evictions_are_safe",
            trace_summary["unsafe_prefix_eviction_records"] == 0,
            (
                "unsafe_prefix_eviction_records="
                f"{trace_summary['unsafe_prefix_eviction_records']}"
            ),
        ),
        check(
            "prefix_eviction_present_if_required",
            not args.require_prefix_eviction
            or trace_summary["prefix_eviction_records"] > 0
            or trace_summary["prefix_evict_success_blocks"] > 0,
            (
                f"prefix_eviction_records={trace_summary['prefix_eviction_records']} "
                "prefix_evict_success_blocks="
                f"{trace_summary['prefix_evict_success_blocks']}"
            ),
        ),
        check(
            "benchmark_no_failed_requests_in_trace_only",
            args.mode != "trace_only" or failed_requests in (None, 0),
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
        "trace_jsonl": str(trace_jsonl),
        "bench_log": str(bench_log) if bench_log else None,
        "server_log": str(server_log) if server_log else None,
        "mode": args.mode,
        "budget_blocks": args.budget_blocks,
        "budget_bytes": args.budget_bytes,
        "candidate_limit": args.candidate_limit,
        "prefix_evict_enable": args.prefix_evict_enable,
        "prefix_evict_max_blocks": args.prefix_evict_max_blocks,
        "eviction_policy": args.eviction_policy,
        "failed_requests": failed_requests,
        **trace_summary,
        "checks": checks,
    }


def print_summary(summary: dict[str, Any], output_json: Path) -> None:
    status = "PASS" if summary["passed"] else "FAIL"
    print("===========================================================")
    print(f" Stage J Success Check: {status}")
    print(f" Trace JSONL: {summary['trace_jsonl']}")
    print(f" Bench log: {summary['bench_log']}")
    print("===========================================================")
    print(
        "- runtime: "
        f"mode={summary.get('mode')} "
        f"budget_blocks={summary.get('budget_blocks')} "
        f"budget_bytes={summary.get('budget_bytes')} "
        f"policy={summary.get('eviction_policy')}"
    )
    print(
        "- pressure: "
        f"allocation_records={summary.get('allocation_pressure_records')} "
        f"over_budget_records={summary.get('over_budget_records')} "
        f"would_deny={summary.get('would_deny_records')} "
        f"max_pressure_blocks={summary.get('max_pressure_blocks')} "
        f"max_pressure_bytes={summary.get('max_pressure_bytes')}"
    )
    print(
        "- candidates: "
        f"records={summary.get('candidate_records')} "
        f"unsafe={summary.get('unsafe_candidate_records')} "
        f"prefix_evictions={summary.get('prefix_eviction_records')} "
        f"unsafe_prefix_evictions={summary.get('unsafe_prefix_eviction_records')}"
    )
    print(
        "- prefix_executor: "
        f"enabled={summary.get('prefix_evict_enable')} "
        f"max_blocks={summary.get('prefix_evict_max_blocks')} "
        f"attempts={summary.get('prefix_evict_attempt_records')} "
        f"success_records={summary.get('prefix_evict_success_records')} "
        f"success_blocks={summary.get('prefix_evict_success_blocks')} "
        f"noop={summary.get('prefix_evict_noop_records')} "
        f"failed={summary.get('prefix_evict_failed_records')}"
    )
    print(f"- action_counts={summary.get('action_counts')}")
    print(f"- failed_requests={summary.get('failed_requests')}")
    print("- checks:")
    for item in summary["checks"]:
        print(f"  {item['name']}={item['passed']} ({item['detail']})")
    print(f"- check_json={output_json}")


def main() -> int:
    args = parse_args()
    if args.budget_blocks < 0:
        raise SystemExit("--budget-blocks must be non-negative")
    if args.budget_bytes < 0:
        raise SystemExit("--budget-bytes must be non-negative")
    if args.candidate_limit < 0:
        raise SystemExit("--candidate-limit must be non-negative")
    if args.prefix_evict_max_blocks < 0:
        raise SystemExit("--prefix-evict-max-blocks must be non-negative")
    if args.prompts < 0:
        raise SystemExit("--prompts must be non-negative")

    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir()
    if args.self_test:
        trace_jsonl, bench_log, server_log, runner_log = run_self_test(args, run_dir)
        summary = validate(
            trace_jsonl=trace_jsonl,
            bench_log=bench_log,
            server_log=server_log,
            runner_log=runner_log,
            args=args,
        )
        output_json = Path(args.output_json) if args.output_json else (
            run_dir / "stage_j_success_check.json"
        )
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print_summary(summary, output_json)
        return 0 if summary["passed"] else 1

    trace_jsonl, bench_log, server_log = resolve_existing_inputs(args)
    runner_log: Path | None = None

    if not args.skip_run:
        trace_jsonl, bench_log, server_log, runner_log = run_experiment(args, run_dir)
    elif trace_jsonl is None:
        raise SystemExit("--skip-run requires --trace-jsonl or --run-dir")

    assert trace_jsonl is not None
    summary = validate(
        trace_jsonl=trace_jsonl,
        bench_log=bench_log,
        server_log=server_log,
        runner_log=runner_log,
        args=args,
    )
    output_json = Path(args.output_json) if args.output_json else (
        run_dir / "stage_j_success_check.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print_summary(summary, output_json)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

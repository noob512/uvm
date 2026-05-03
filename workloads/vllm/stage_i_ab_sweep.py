#!/usr/bin/env python3
"""Run a one-shot Stage I A/B sweep.

The script generates one Stage H hot/cold expert plan, then reuses that plan
across several Stage I execution variants so the slowdown source can be
separated into prefetch, offload/advise, and trace/planning overhead.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Variant:
    name: str
    prefetch_enable: bool
    prefetch_mode: str
    max_experts_per_layer: int
    offload_enable: bool
    offload_mode: str
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage I A/B variants from one shared Stage H plan and summarize "
            "latency, throughput, prefetch, and offload/advise signals."
        )
    )
    parser.add_argument("--run-dir", help="Output directory. Default uses /tmp.")
    parser.add_argument("--plan-prompts", type=int, default=1)
    parser.add_argument("--prompts", type=int, default=1)
    parser.add_argument("--request-rate", default="5")
    parser.add_argument("--output-len", default="512")
    parser.add_argument("--target-roles", default="moe_gate_up,moe_down")
    parser.add_argument("--max-bytes-per-step", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--offload-max-bytes-per-step", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--offload-max-experts-per-layer", type=int, default=1)
    parser.add_argument(
        "--expert-sweep",
        default="1,2,4",
        help="Comma-separated max-experts-per-layer values for prefetch-only sweep.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip no_stage_i runtime baseline.",
    )
    parser.add_argument(
        "--skip-offload",
        action="store_true",
        help="Skip offload trace/advise variants.",
    )
    return parser.parse_args()


def default_run_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return Path(f"/tmp/vllm_stage_i_ab_sweep_{stamp}")


def parse_int_csv(value: str) -> list[int]:
    out: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed = int(item)
        if parsed < 0:
            raise SystemExit("--expert-sweep values must be non-negative")
        out.append(parsed)
    if not out:
        raise SystemExit("--expert-sweep must contain at least one integer")
    return out


def build_variants(args: argparse.Namespace) -> list[Variant]:
    variants: list[Variant] = []
    if not args.skip_baseline:
        variants.append(
            Variant(
                name="no_stage_i",
                prefetch_enable=False,
                prefetch_mode="trace_only",
                max_experts_per_layer=0,
                offload_enable=False,
                offload_mode="trace_only",
                description="Routing/weight-map runtime without Stage I actions.",
            )
        )
    variants.append(
        Variant(
            name="prefetch_trace_only_no_offload",
            prefetch_enable=True,
            prefetch_mode="trace_only",
            max_experts_per_layer=2,
            offload_enable=False,
            offload_mode="trace_only",
            description="Stage I trace overhead without CUDA prefetch/offload actions.",
        )
    )
    for max_experts in parse_int_csv(args.expert_sweep):
        variants.append(
            Variant(
                name=f"prefetch_no_offload_e{max_experts}",
                prefetch_enable=True,
                prefetch_mode="prefetch",
                max_experts_per_layer=max_experts,
                offload_enable=False,
                offload_mode="trace_only",
                description=(
                    "Hot expert GPU prefetch only; no cold offload/advise."
                ),
            )
        )
    if not args.skip_offload:
        variants.extend(
            [
                Variant(
                    name="prefetch_offload_trace_e2",
                    prefetch_enable=True,
                    prefetch_mode="prefetch",
                    max_experts_per_layer=2,
                    offload_enable=True,
                    offload_mode="trace_only",
                    description="Hot prefetch plus cold offload trace only.",
                ),
                Variant(
                    name="prefetch_offload_advise_e2",
                    prefetch_enable=True,
                    prefetch_mode="prefetch",
                    max_experts_per_layer=2,
                    offload_enable=True,
                    offload_mode="advise_cpu",
                    description="Hot prefetch plus cold expert CPU preferred-location advise.",
                ),
            ]
        )
    return variants


def run_command(cmd: list[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
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
        raise SystemExit(f"command failed with exit code {rc}; log={log_path}")


def run_stage_h_planning(args: argparse.Namespace, run_dir: Path) -> dict[str, Path]:
    planning_dir = run_dir / "00_planning"
    paths = {
        "allocator_log": planning_dir / "vllm_uvm_allocator_trace_stage_i_planning.log",
        "metrics_json": planning_dir / "vllm_stage_i_allocator_metrics_planning.json",
        "moe_routing_jsonl": planning_dir / "vllm_uvm_moe_routing_stage_i_planning.jsonl",
        "weight_map_jsonl": planning_dir / "vllm_uvm_weight_regions_stage_i_planning.jsonl",
        "bench_log": planning_dir / "vllm_bench_stage_i_planning.log",
        "runner_log": planning_dir / "stage_i_planning_runner.log",
        "plan_json": run_dir / "vllm_stage_i_weight_expert_plan.json",
        "plan_summary_json": run_dir / "vllm_stage_i_weight_expert_plan_summary.json",
    }
    cmd = [
        "./run_kv_fault_ratio.sh",
        "--mode",
        "trace",
        "--allocator-log",
        str(paths["allocator_log"]),
        "--trace-log",
        str(planning_dir / "uvm_kv_fault_stats_stage_i_planning.log"),
        "--address-log",
        str(planning_dir / "vllm_uvm_address_regions_stage_i_planning.log"),
        "--server-log",
        str(planning_dir / "vllm_server_stage_i_planning.log"),
        "--bench-log",
        str(paths["bench_log"]),
        "--uvm-kv-budget-bytes",
        "0",
        "--uvm-weight-budget-bytes",
        "0",
        "--uvm-weight-map-enable",
        "1",
        "--uvm-weight-map-file",
        str(paths["weight_map_jsonl"]),
        "--uvm-moe-routing-trace-enable",
        "1",
        "--uvm-moe-routing-trace-file",
        str(paths["moe_routing_jsonl"]),
        "--uvm-weight-prefetch-enable",
        "0",
        "--uvm-weight-offload-enable",
        "0",
        "--uvm-pool-registry-enable",
        "1",
        "--gap-watch-metrics-summary-json",
        str(paths["metrics_json"]),
        "--prompts",
        str(args.plan_prompts),
        "--request-rate",
        str(args.request_rate),
        "--output-len",
        str(args.output_len),
    ]

    print("===========================================================")
    print(" Stage I A/B Sweep: planning shared Stage H hot/cold plan")
    print(f" Output dir: {run_dir}")
    print(f" Plan JSON: {paths['plan_json']}")
    print("===========================================================")
    run_command(cmd, cwd=SCRIPT_DIR, log_path=paths["runner_log"])

    plan_cmd = [
        "python3",
        str(SCRIPT_DIR / "plan_stage_h_weight_expert_actions.py"),
        "--weight-map",
        str(paths["weight_map_jsonl"]),
        "--moe-routing-trace",
        str(paths["moe_routing_jsonl"]),
        "--plan-json",
        str(paths["plan_json"]),
        "--summary-json",
        str(paths["plan_summary_json"]),
        "--target-roles",
        args.target_roles,
        "--hot-top-k",
        "1024",
        "--cold-bottom-k",
        "1024",
        "--require-routing",
    ]
    run_command(plan_cmd, cwd=SCRIPT_DIR, log_path=planning_dir / "stage_h_plan.log")
    return paths


def run_variant(
    args: argparse.Namespace,
    run_dir: Path,
    plan_json: Path,
    variant: Variant,
) -> dict[str, Path]:
    variant_dir = run_dir / variant.name
    paths = {
        "allocator_log": variant_dir / "vllm_uvm_allocator_trace_stage_i.log",
        "metrics_json": variant_dir / "vllm_stage_i_allocator_metrics.json",
        "prefetch_trace": variant_dir / "vllm_uvm_weight_prefetch_stage_i.jsonl",
        "moe_routing_jsonl": variant_dir / "vllm_uvm_moe_routing_stage_i.jsonl",
        "weight_map_jsonl": variant_dir / "vllm_uvm_weight_regions_stage_i.jsonl",
        "bench_log": variant_dir / "vllm_bench_stage_i.log",
        "runner_log": variant_dir / "stage_i_runner.log",
    }

    cmd = [
        "./run_kv_fault_ratio.sh",
        "--mode",
        "trace",
        "--allocator-log",
        str(paths["allocator_log"]),
        "--trace-log",
        str(variant_dir / "uvm_kv_fault_stats_stage_i.log"),
        "--address-log",
        str(variant_dir / "vllm_uvm_address_regions_stage_i.log"),
        "--server-log",
        str(variant_dir / "vllm_server_stage_i.log"),
        "--bench-log",
        str(paths["bench_log"]),
        "--uvm-kv-budget-bytes",
        "0",
        "--uvm-weight-budget-bytes",
        "0",
        "--uvm-weight-map-enable",
        "1",
        "--uvm-weight-map-file",
        str(paths["weight_map_jsonl"]),
        "--uvm-moe-routing-trace-enable",
        "1",
        "--uvm-moe-routing-trace-file",
        str(paths["moe_routing_jsonl"]),
        "--uvm-weight-prefetch-enable",
        "1" if variant.prefetch_enable else "0",
        "--uvm-weight-prefetch-mode",
        variant.prefetch_mode,
        "--uvm-weight-prefetch-trace-file",
        str(paths["prefetch_trace"]),
        "--uvm-weight-prefetch-max-bytes-per-step",
        str(args.max_bytes_per_step),
        "--uvm-weight-prefetch-max-experts-per-layer",
        str(variant.max_experts_per_layer),
        "--uvm-weight-prefetch-target-roles",
        args.target_roles,
        "--uvm-weight-prefetch-plan-file",
        str(plan_json),
        "--uvm-weight-prefetch-require-plan",
        "1" if variant.prefetch_enable else "0",
        "--uvm-weight-offload-enable",
        "1" if variant.offload_enable else "0",
        "--uvm-weight-offload-mode",
        variant.offload_mode,
        "--uvm-weight-offload-plan-file",
        str(plan_json),
        "--uvm-weight-offload-max-bytes-per-step",
        str(args.offload_max_bytes_per_step),
        "--uvm-weight-offload-max-experts-per-layer",
        str(args.offload_max_experts_per_layer),
        "--uvm-weight-offload-target-roles",
        args.target_roles,
        "--uvm-pool-registry-enable",
        "1",
        "--gap-watch-metrics-summary-json",
        str(paths["metrics_json"]),
        "--prompts",
        str(args.prompts),
        "--request-rate",
        str(args.request_rate),
        "--output-len",
        str(args.output_len),
    ]

    print("===========================================================")
    print(f" Stage I A/B Sweep: running variant {variant.name}")
    print(f" Description: {variant.description}")
    print("===========================================================")
    run_command(cmd, cwd=SCRIPT_DIR, log_path=paths["runner_log"])
    return paths


def parse_bench_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    fields = {
        "successful_requests": r"Successful requests:\s+([0-9.]+)",
        "failed_requests": r"Failed requests:\s+([0-9.]+)",
        "benchmark_duration_s": r"Benchmark duration \(s\):\s+([0-9.]+)",
        "output_tok_per_s": r"Output token throughput \(tok/s\):\s+([0-9.]+)",
        "total_tok_per_s": r"Total Token throughput \(tok/s\):\s+([0-9.]+)",
        "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([0-9.]+)",
        "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([0-9.]+)",
        "mean_itl_ms": r"Mean ITL \(ms\):\s+([0-9.]+)",
    }
    parsed: dict[str, Any] = {}
    for key, pattern in fields.items():
        matches = re.findall(pattern, text)
        parsed[key] = float(matches[-1]) if matches else None
    return parsed


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
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


def summarize_stage_i_trace(path: Path) -> dict[str, Any]:
    action_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    prefetch_bytes = 0
    offload_bytes = 0
    max_prefetch_step_bytes = 0
    max_offload_step_bytes = 0
    for record in read_jsonl(path):
        action = str(record.get("action", "unknown"))
        action_counts[action] += 1
        role = record.get("role")
        if role:
            role_counts[str(role)] += 1
        size = int(record.get("bytes") or 0)
        if action == "prefetch_issued":
            prefetch_bytes += size
        if action in {"offload_advise_cpu_issued", "offload_prefetch_cpu_issued"}:
            offload_bytes += size
        if action == "step_summary":
            max_prefetch_step_bytes = max(
                max_prefetch_step_bytes,
                int(record.get("issued_bytes") or 0),
            )
        if action == "offload_step_summary":
            max_offload_step_bytes = max(
                max_offload_step_bytes,
                int(record.get("issued_bytes") or 0),
            )
    return {
        "trace_records": sum(action_counts.values()),
        "action_counts": dict(action_counts),
        "role_counts": dict(role_counts),
        "prefetch_issued": action_counts.get("prefetch_issued", 0),
        "prefetch_bytes": prefetch_bytes,
        "trace_prefetch_candidates": action_counts.get("trace_prefetch_candidate", 0),
        "offload_advise_cpu_issued": action_counts.get("offload_advise_cpu_issued", 0),
        "offload_prefetch_cpu_issued": action_counts.get("offload_prefetch_cpu_issued", 0),
        "trace_offload_candidates": action_counts.get("trace_offload_candidate", 0),
        "offload_bytes": offload_bytes,
        "max_prefetch_step_bytes": max_prefetch_step_bytes,
        "max_offload_step_bytes": max_offload_step_bytes,
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_variant(
    variant: Variant,
    paths: dict[str, Path],
    plan_summary: dict[str, Any],
) -> dict[str, Any]:
    bench = parse_bench_log(paths["bench_log"])
    trace = summarize_stage_i_trace(paths["prefetch_trace"])
    metrics = load_json(paths["metrics_json"])
    return {
        "variant": variant.name,
        "description": variant.description,
        "prefetch_enable": variant.prefetch_enable,
        "prefetch_mode": variant.prefetch_mode,
        "max_experts_per_layer": variant.max_experts_per_layer,
        "offload_enable": variant.offload_enable,
        "offload_mode": variant.offload_mode,
        "prefetch_plan_records": plan_summary.get("prefetch_plan_records"),
        "offload_plan_records": plan_summary.get("offload_plan_records"),
        "weight_trace_allocations": metrics.get("weight_trace_allocations"),
        "pool_registry_enabled": metrics.get("pool_registry_enabled"),
        **bench,
        **trace,
        "bench_log": str(paths["bench_log"]),
        "prefetch_trace": str(paths["prefetch_trace"]),
        "metrics_json": str(paths["metrics_json"]),
    }


def write_outputs(run_dir: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    summary_json = run_dir / "stage_i_ab_sweep_summary.json"
    summary_csv = run_dir / "stage_i_ab_sweep_summary.csv"
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump({"variants": rows}, handle, indent=2, sort_keys=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary_json, summary_csv


def print_summary(rows: list[dict[str, Any]], summary_json: Path, summary_csv: Path) -> None:
    print("===========================================================")
    print(" Stage I A/B Sweep Summary")
    print(f" Summary JSON: {summary_json}")
    print(f" Summary CSV:  {summary_csv}")
    print("===========================================================")
    header = (
        f"{'variant':34} {'TPOT':>8} {'TTFT':>8} {'out tok/s':>10} "
        f"{'prefetch':>9} {'off_adv':>8} {'off_trace':>9} {'failed':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['variant'][:34]:34} "
            f"{str(row.get('mean_tpot_ms')):>8} "
            f"{str(row.get('mean_ttft_ms')):>8} "
            f"{str(row.get('output_tok_per_s')):>10} "
            f"{row.get('prefetch_issued', 0):>9} "
            f"{row.get('offload_advise_cpu_issued', 0):>8} "
            f"{row.get('trace_offload_candidates', 0):>9} "
            f"{str(row.get('failed_requests')):>7}"
        )


def main() -> int:
    args = parse_args()
    if args.plan_prompts < 0 or args.prompts < 0:
        raise SystemExit("--plan-prompts and --prompts must be non-negative")
    if args.max_bytes_per_step < 0 or args.offload_max_bytes_per_step < 0:
        raise SystemExit("byte budgets must be non-negative")
    if args.offload_max_experts_per_layer < 0:
        raise SystemExit("--offload-max-experts-per-layer must be non-negative")

    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    planning_paths = run_stage_h_planning(args, run_dir)
    plan_summary = load_json(planning_paths["plan_summary_json"])

    rows: list[dict[str, Any]] = []
    for variant in build_variants(args):
        variant_paths = run_variant(args, run_dir, planning_paths["plan_json"], variant)
        rows.append(summarize_variant(variant, variant_paths, plan_summary))

    summary_json, summary_csv = write_outputs(run_dir, rows)
    print_summary(rows, summary_json, summary_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

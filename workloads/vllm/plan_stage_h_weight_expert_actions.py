#!/usr/bin/env python3
"""Build Stage H trace-only hot/cold expert weight action plans.

Stage H intentionally does not migrate memory. It joins the Stage E weight
semantic map with MoE routing heat and optional replayable fault addresses,
then emits trace-only prefetch/offload candidates for later execution stages.
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


RAW_ADDR_RE = re.compile(r"(?:\braw|精确原始地址)=(0x[0-9a-fA-F]+)")
PAGE_ADDR_RE = re.compile(r"(?:\bpage|对齐到页地址)=(0x[0-9a-fA-F]+)")
LAYER_RE = re.compile(r"(?:^|\.)(?:layers|layer)\.(\d+)(?:\.|$)")


@dataclass
class ExpertRange:
    layer_id: int
    expert_id: int
    role: str
    name: str
    start: int
    end: int
    size_bytes: int
    source: str


@dataclass
class ExpertHeat:
    layer_id: int
    expert_id: int
    roles: Counter[str] = field(default_factory=Counter)
    weight_names: set[str] = field(default_factory=set)
    ranges: list[ExpertRange] = field(default_factory=list)
    bytes: int = 0
    routing_tokens: int = 0
    routing_records: int = 0
    first_step: int | None = None
    last_step: int | None = None
    fault_count: int = 0
    unique_fault_pages: set[int] = field(default_factory=set)

    @property
    def unique_fault_count(self) -> int:
        return len(self.unique_fault_pages)


class RangeLookup:
    def __init__(self, ranges: Iterable[ExpertRange]) -> None:
        self.ranges = sorted(ranges, key=lambda item: (item.start, item.end))
        self.starts = [item.start for item in self.ranges]

    def find(self, address: int) -> ExpertRange | None:
        idx = bisect.bisect_right(self.starts, address) - 1
        if idx < 0:
            return None
        item = self.ranges[idx]
        if address <= item.end:
            return item
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Stage H trace-only MoE expert hot/cold plans."
    )
    parser.add_argument("--weight-map", required=True, help="Stage E weight map JSONL.")
    parser.add_argument("--moe-routing-trace", help="Stage E MoE routing JSONL.")
    parser.add_argument("--fault-log", help="Optional per-fault address trace log.")
    parser.add_argument("--plan-json", required=True, help="Output Stage H plan JSON.")
    parser.add_argument("--summary-json", help="Optional summary JSON path.")
    parser.add_argument(
        "--target-roles",
        default="moe_gate_up,moe_gate,moe_up,moe_down",
        help="Comma-separated MoE expert weight roles eligible for planning.",
    )
    parser.add_argument("--hot-top-k", type=int, default=16)
    parser.add_argument("--cold-bottom-k", type=int, default=16)
    parser.add_argument("--hot-min-routing-tokens", type=int, default=1)
    parser.add_argument("--fault-score-weight", type=float, default=16.0)
    parser.add_argument("--prefetch-plan-max-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--offload-plan-max-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument(
        "--require-routing",
        action="store_true",
        help="Fail if no MoE routing trace records can be joined.",
    )
    return parser.parse_args()


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


def parse_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value, 16) if value.startswith("0x") else int(value)
        except ValueError:
            return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def layer_id_from_name(name: str) -> int | None:
    match = LAYER_RE.search(name)
    if match:
        return int(match.group(1))
    return None


def normalize_role(role: Any) -> str:
    return str(role or "unknown")


def expand_weight_record(
    record: dict[str, Any],
    target_roles: set[str],
) -> list[ExpertRange]:
    role = normalize_role(record.get("role"))
    if role not in target_roles:
        return []
    if not record.get("is_moe_expert"):
        return []

    name = str(record.get("name") or "unknown")
    layer_id = parse_int(record.get("layer_id"))
    if layer_id is None:
        layer_id = layer_id_from_name(name)
    if layer_id is None:
        return []

    start = parse_int(record.get("start_int")) or parse_int(record.get("start"))
    end = parse_int(record.get("end_int")) or parse_int(record.get("end"))
    size_bytes = parse_int(record.get("size_bytes"))
    if start is None or end is None or size_bytes is None or size_bytes <= 0:
        return []
    if end < start:
        return []

    expert_id = parse_int(record.get("expert_id"))
    if expert_id is not None:
        return [
            ExpertRange(
                layer_id=layer_id,
                expert_id=expert_id,
                role=role,
                name=name,
                start=start,
                end=end,
                size_bytes=size_bytes,
                source="concrete_expert_tensor",
            )
        ]

    shape = record.get("shape")
    if not isinstance(shape, list) or not shape:
        return []
    num_experts = parse_int(shape[0])
    if num_experts is None or num_experts <= 1:
        return []

    slice_bytes = size_bytes // num_experts
    if slice_bytes <= 0:
        return []

    ranges: list[ExpertRange] = []
    for local_expert_id in range(num_experts):
        slice_start = start + local_expert_id * slice_bytes
        slice_end = (
            start + (local_expert_id + 1) * slice_bytes - 1
            if local_expert_id + 1 < num_experts
            else end
        )
        if slice_end < slice_start:
            continue
        ranges.append(
            ExpertRange(
                layer_id=layer_id,
                expert_id=local_expert_id,
                role=role,
                name=name,
                start=slice_start,
                end=slice_end,
                size_bytes=slice_end - slice_start + 1,
                source="logical_fused_expert_slice",
            )
        )
    return ranges


def parse_fault_addresses(path: Path | None) -> list[int]:
    if path is None or not path.is_file():
        return []
    addresses: list[int] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            page_match = PAGE_ADDR_RE.search(line)
            raw_match = RAW_ADDR_RE.search(line)
            match = page_match or raw_match
            if match:
                addresses.append(int(match.group(1), 16))
    return addresses


def routing_layer_id(record: dict[str, Any]) -> int | None:
    layer_name = str(record.get("layer_name") or "")
    return layer_id_from_name(layer_name)


def add_step_bounds(heat: ExpertHeat, step: int | None) -> None:
    if step is None:
        return
    if heat.first_step is None or step < heat.first_step:
        heat.first_step = step
    if heat.last_step is None or step > heat.last_step:
        heat.last_step = step


def heat_score(heat: ExpertHeat, fault_score_weight: float) -> float:
    return (
        float(heat.routing_tokens)
        + fault_score_weight * float(heat.unique_fault_count)
        + 0.01 * float(heat.fault_count)
    )


def action_record(
    heat: ExpertHeat,
    *,
    action: str,
    reason: str,
    score: float,
    max_ranges: int = 8,
) -> dict[str, Any]:
    return {
        "action": action,
        "reason": reason,
        "mode": "trace_only",
        "layer_id": heat.layer_id,
        "expert_id": heat.expert_id,
        "score": round(score, 6),
        "bytes": heat.bytes,
        "routing_tokens": heat.routing_tokens,
        "routing_records": heat.routing_records,
        "fault_count": heat.fault_count,
        "unique_fault_pages": heat.unique_fault_count,
        "roles": dict(heat.roles),
        "first_step": heat.first_step,
        "last_step": heat.last_step,
        "ranges": [
            {
                "name": item.name,
                "role": item.role,
                "start": f"0x{item.start:x}",
                "end": f"0x{item.end:x}",
                "bytes": item.size_bytes,
                "source": item.source,
            }
            for item in heat.ranges[:max_ranges]
        ],
        "range_count": len(heat.ranges),
    }


def select_with_byte_limit(
    candidates: list[tuple[float, ExpertHeat]],
    *,
    max_items: int,
    max_bytes: int,
    reverse: bool,
) -> list[tuple[float, ExpertHeat]]:
    selected: list[tuple[float, ExpertHeat]] = []
    total_bytes = 0
    for score, heat in sorted(candidates, key=lambda item: item[0], reverse=reverse):
        if len(selected) >= max_items:
            break
        if max_bytes > 0 and total_bytes + heat.bytes > max_bytes and selected:
            continue
        selected.append((score, heat))
        total_bytes += heat.bytes
    return selected


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    weight_map_path = Path(args.weight_map)
    routing_path = Path(args.moe_routing_trace) if args.moe_routing_trace else None
    fault_path = Path(args.fault_log) if args.fault_log else None
    target_roles = {
        item.strip() for item in args.target_roles.split(",") if item.strip()
    }

    weight_records = read_jsonl(weight_map_path)
    routing_records = read_jsonl(routing_path)
    expert_ranges: list[ExpertRange] = []
    for record in weight_records:
        expert_ranges.extend(expand_weight_record(record, target_roles))

    heat_by_key: dict[tuple[int, int], ExpertHeat] = {}
    for item in expert_ranges:
        key = (item.layer_id, item.expert_id)
        heat = heat_by_key.setdefault(
            key, ExpertHeat(layer_id=item.layer_id, expert_id=item.expert_id)
        )
        heat.roles[item.role] += 1
        heat.weight_names.add(item.name)
        heat.ranges.append(item)
        heat.bytes += item.size_bytes

    routing_join_records = 0
    routing_unmatched_records = 0
    for record in routing_records:
        layer_id = routing_layer_id(record)
        counts = record.get("expert_token_counts")
        if layer_id is None or not isinstance(counts, dict):
            routing_unmatched_records += 1
            continue
        step = parse_int(record.get("step"))
        for expert_id_text, count_value in counts.items():
            expert_id = parse_int(expert_id_text)
            count = parse_int(count_value)
            if expert_id is None or count is None or count <= 0:
                continue
            heat = heat_by_key.get((layer_id, expert_id))
            if heat is None:
                routing_unmatched_records += 1
                continue
            heat.routing_tokens += count
            heat.routing_records += 1
            add_step_bounds(heat, step)
            routing_join_records += 1

    fault_addresses = parse_fault_addresses(fault_path)
    lookup = RangeLookup(expert_ranges)
    fault_join_records = 0
    fault_unmatched_records = 0
    for address in fault_addresses:
        item = lookup.find(address)
        if item is None:
            fault_unmatched_records += 1
            continue
        heat = heat_by_key.get((item.layer_id, item.expert_id))
        if heat is None:
            fault_unmatched_records += 1
            continue
        heat.fault_count += 1
        heat.unique_fault_pages.add(address)
        fault_join_records += 1

    all_heat = list(heat_by_key.values())
    scored = [(heat_score(item, args.fault_score_weight), item) for item in all_heat]
    hot_candidates = [
        (score, item)
        for score, item in scored
        if item.routing_tokens >= args.hot_min_routing_tokens
        or item.unique_fault_count > 0
    ]
    cold_candidates = [
        (score, item)
        for score, item in scored
        if item.routing_tokens == 0 and item.unique_fault_count == 0
    ]

    prefetch_selected = select_with_byte_limit(
        hot_candidates,
        max_items=args.hot_top_k,
        max_bytes=args.prefetch_plan_max_bytes,
        reverse=True,
    )
    offload_selected = select_with_byte_limit(
        cold_candidates,
        max_items=args.cold_bottom_k,
        max_bytes=args.offload_plan_max_bytes,
        reverse=False,
    )

    prefetch_plan = [
        action_record(
            heat,
            action="prefetch_candidate",
            reason="hot_by_routing_or_fault",
            score=score,
        )
        for score, heat in prefetch_selected
    ]
    offload_plan = [
        action_record(
            heat,
            action="offload_candidate",
            reason="cold_no_routing_no_fault",
            score=score,
        )
        for score, heat in offload_selected
    ]

    role_counts = Counter()
    source_counts = Counter()
    for item in expert_ranges:
        role_counts[item.role] += 1
        source_counts[item.source] += 1

    summary = {
        "stage": "H",
        "mode": "trace_only",
        "weight_map": str(weight_map_path),
        "moe_routing_trace": str(routing_path) if routing_path else None,
        "fault_log": str(fault_path) if fault_path else None,
        "target_roles": sorted(target_roles),
        "weight_map_records": len(weight_records),
        "expert_weight_range_records": len(expert_ranges),
        "concrete_expert_weight_records": source_counts.get("concrete_expert_tensor", 0),
        "logical_fused_expert_records": source_counts.get(
            "logical_fused_expert_slice", 0
        ),
        "expert_heat_records": len(all_heat),
        "expert_weight_bytes": sum(item.bytes for item in all_heat),
        "expert_role_counts": dict(role_counts),
        "expert_range_source_counts": dict(source_counts),
        "moe_routing_records": len(routing_records),
        "routing_join_records": routing_join_records,
        "routing_unmatched_records": routing_unmatched_records,
        "fault_address_records": len(fault_addresses),
        "weight_fault_join_records": fault_join_records,
        "fault_unmatched_records": fault_unmatched_records,
        "prefetch_plan_records": len(prefetch_plan),
        "prefetch_plan_bytes": sum(item["bytes"] for item in prefetch_plan),
        "prefetch_plan_max_bytes": args.prefetch_plan_max_bytes,
        "offload_plan_records": len(offload_plan),
        "offload_plan_bytes": sum(item["bytes"] for item in offload_plan),
        "offload_plan_max_bytes": args.offload_plan_max_bytes,
        "top_hot_experts": [
            action_record(heat, action="rank_only", reason="top_heat", score=score)
            for score, heat in sorted(scored, key=lambda item: item[0], reverse=True)[:10]
        ],
        "coldest_experts": [
            action_record(heat, action="rank_only", reason="lowest_heat", score=score)
            for score, heat in sorted(scored, key=lambda item: item[0])[:10]
        ],
    }

    return {
        **summary,
        "prefetch_plan": prefetch_plan,
        "offload_plan": offload_plan,
    }


def print_summary(plan: dict[str, Any]) -> None:
    print("Stage H Weight Expert Plan Summary")
    print(f"- mode={plan['mode']}")
    print(f"- weight_map_records={plan['weight_map_records']}")
    print(f"- expert_weight_range_records={plan['expert_weight_range_records']}")
    print(f"- logical_fused_expert_records={plan['logical_fused_expert_records']}")
    print(f"- expert_heat_records={plan['expert_heat_records']}")
    print(f"- moe_routing_records={plan['moe_routing_records']}")
    print(f"- routing_join_records={plan['routing_join_records']}")
    print(f"- fault_address_records={plan['fault_address_records']}")
    print(f"- weight_fault_join_records={plan['weight_fault_join_records']}")
    print(
        "- prefetch_plan="
        f"records={plan['prefetch_plan_records']} bytes={plan['prefetch_plan_bytes']}"
    )
    print(
        "- offload_plan="
        f"records={plan['offload_plan_records']} bytes={plan['offload_plan_bytes']}"
    )


def main() -> int:
    args = parse_args()
    if args.hot_top_k < 0:
        raise SystemExit("--hot-top-k must be non-negative")
    if args.cold_bottom_k < 0:
        raise SystemExit("--cold-bottom-k must be non-negative")
    if args.prefetch_plan_max_bytes < 0:
        raise SystemExit("--prefetch-plan-max-bytes must be non-negative")
    if args.offload_plan_max_bytes < 0:
        raise SystemExit("--offload-plan-max-bytes must be non-negative")

    plan = build_plan(args)
    if args.require_routing and plan["routing_join_records"] <= 0:
        raise SystemExit("no MoE routing records joined with expert weights")

    plan_json = Path(args.plan_json)
    plan_json.parent.mkdir(parents=True, exist_ok=True)
    with plan_json.open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, sort_keys=True)

    if args.summary_json:
        summary = dict(plan)
        summary.pop("prefetch_plan", None)
        summary.pop("offload_plan", None)
        summary_json = Path(args.summary_json)
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        with summary_json.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)

    print_summary(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

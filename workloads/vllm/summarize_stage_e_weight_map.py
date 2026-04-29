#!/usr/bin/env python3
"""Summarize Stage E weight semantic map and optional MoE routing trace."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Stage E weight map and MoE routing JSONL files."
    )
    parser.add_argument("--weight-map", required=True, help="Weight map JSONL path.")
    parser.add_argument("--moe-routing-trace", help="Optional MoE routing JSONL path.")
    parser.add_argument("--summary-json", help="Optional JSON output path.")
    return parser.parse_args()


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


def summarize_weight_map(records: list[dict[str, Any]]) -> dict[str, Any]:
    kind_counter: Counter[str] = Counter()
    role_counter: Counter[str] = Counter()
    layer_counter: Counter[str] = Counter()
    expert_counter: Counter[str] = Counter()
    dtype_counter: Counter[str] = Counter()
    total_bytes = 0
    moe_expert_bytes = 0

    for record in records:
        size = int(record.get("size_bytes") or 0)
        total_bytes += size
        kind_counter[str(record.get("kind", "unknown"))] += 1
        role_counter[str(record.get("role", "unknown"))] += 1
        dtype_counter[str(record.get("dtype", "unknown"))] += 1
        layer_id = record.get("layer_id")
        if layer_id is not None:
            layer_counter[str(layer_id)] += 1
        expert_id = record.get("expert_id")
        if expert_id is not None:
            expert_counter[str(expert_id)] += 1
        if record.get("is_moe_expert"):
            moe_expert_bytes += size

    return {
        "weight_map_records": len(records),
        "weight_map_total_bytes": total_bytes,
        "weight_map_moe_expert_records": sum(
            1 for record in records if record.get("is_moe_expert")
        ),
        "weight_map_moe_expert_bytes": moe_expert_bytes,
        "weight_map_layer_count": len(layer_counter),
        "weight_map_expert_count": len(expert_counter),
        "weight_map_kind_counts": dict(kind_counter),
        "weight_map_role_counts": dict(role_counter),
        "weight_map_dtype_counts": dict(dtype_counter),
        "weight_map_top_layers": layer_counter.most_common(10),
        "weight_map_top_experts": expert_counter.most_common(10),
    }


def summarize_moe_routing(records: list[dict[str, Any]]) -> dict[str, Any]:
    layer_counter: Counter[str] = Counter()
    expert_counter: Counter[str] = Counter()
    total_tokens = 0

    for record in records:
        layer_counter[str(record.get("layer_name", "unknown"))] += 1
        total_tokens += int(record.get("num_tokens") or 0)
        counts = record.get("expert_token_counts") or {}
        if isinstance(counts, dict):
            for expert_id, count in counts.items():
                expert_counter[str(expert_id)] += int(count)

    return {
        "moe_routing_records": len(records),
        "moe_routing_total_tokens": total_tokens,
        "moe_routing_layer_count": len(layer_counter),
        "moe_routing_active_expert_count": len(expert_counter),
        "moe_routing_top_layers": layer_counter.most_common(10),
        "moe_routing_top_experts": expert_counter.most_common(10),
    }


def main() -> int:
    args = parse_args()
    weight_map = Path(args.weight_map)
    moe_trace = Path(args.moe_routing_trace) if args.moe_routing_trace else None
    weight_records = read_jsonl(weight_map)
    moe_records = read_jsonl(moe_trace) if moe_trace else []

    summary = {
        "weight_map": str(weight_map),
        "moe_routing_trace": str(moe_trace) if moe_trace else None,
        **summarize_weight_map(weight_records),
        **summarize_moe_routing(moe_records),
    }

    if args.summary_json:
        with Path(args.summary_json).open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)

    print("Stage E Weight Map Summary")
    print(f"- weight_map_records={summary['weight_map_records']}")
    print(f"- weight_map_total_bytes={summary['weight_map_total_bytes']}")
    print(f"- weight_map_moe_expert_records={summary['weight_map_moe_expert_records']}")
    print(f"- weight_map_moe_expert_bytes={summary['weight_map_moe_expert_bytes']}")
    print(f"- weight_map_layer_count={summary['weight_map_layer_count']}")
    print(f"- weight_map_expert_count={summary['weight_map_expert_count']}")
    print(f"- weight_map_role_counts={summary['weight_map_role_counts']}")
    print(f"- moe_routing_records={summary['moe_routing_records']}")
    print(f"- moe_routing_total_tokens={summary['moe_routing_total_tokens']}")
    print(f"- moe_routing_layer_count={summary['moe_routing_layer_count']}")
    print(f"- moe_routing_active_expert_count={summary['moe_routing_active_expert_count']}")
    print(f"- moe_routing_top_experts={summary['moe_routing_top_experts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export allocator-traced workspace/scratch allocations into an address-log-like file.

This tool is intentionally address-oriented, not fault-oriented. It converts
`TRACE_POLICY` / `TRACE_ALLOC` information from the allocator trace into a
region inventory grouped by phase, so the investigator can see which address
ranges were used by warmup/runtime workspace-like allocations.

Important: because allocator address ranges can be reused many times across the
run, the exported file should be treated as a named region catalog for manual
inspection, not as a time-agnostic substitute for weight/KV static matching.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from deep_dive_uvm_faults import parse_allocator_log


DEFAULT_CLASSES = (
    "warmup_workspace",
    "runtime_scratch",
    "runtime_workspace",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export workspace-like allocator-traced allocations into an "
            "address-log-like region file."
        )
    )
    parser.add_argument(
        "--allocator-log",
        default="/tmp/vllm_uvm_allocator_trace.log",
        help="Path to the allocator trace log.",
    )
    parser.add_argument(
        "--output-log",
        default="/tmp/vllm_uvm_workspace_regions.log",
        help="Output path for the synthesized address-log-like region file.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional JSON summary for exported regions.",
    )
    parser.add_argument(
        "--min-size-bytes",
        type=int,
        default=1 * 1024 * 1024,
        help="Only export allocations with size >= this threshold.",
    )
    parser.add_argument(
        "--classes",
        default=",".join(DEFAULT_CLASSES),
        help=(
            "Comma-separated predicted classes to export. "
            f"Default: {','.join(DEFAULT_CLASSES)}"
        ),
    )
    parser.add_argument(
        "--phases",
        default=None,
        help="Optional comma-separated phase allowlist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    allocator_log_path = Path(args.allocator_log)
    if not allocator_log_path.is_file():
        raise SystemExit(f"allocator log not found: {allocator_log_path}")

    selected_classes = {
        item.strip() for item in args.classes.split(",") if item.strip()
    }
    selected_phases = (
        {item.strip() for item in args.phases.split(",") if item.strip()}
        if args.phases
        else None
    )

    allocator_report = parse_allocator_log(allocator_log_path)
    allocations = allocator_report["allocations"]

    grouped_rows: dict[str, list[tuple[str, str, int, int, int, str | None]]] = defaultdict(list)
    phase_counts = Counter()
    class_counts = Counter()
    exported = 0

    for alloc in allocations:
        if alloc.predicted_class not in selected_classes:
            continue
        if alloc.size_bytes < args.min_size_bytes:
            continue
        if selected_phases is not None and alloc.phase not in selected_phases:
            continue

        kind = f"allocator:{alloc.predicted_class}"
        name = (
            f"alloc_id={alloc.alloc_id}"
            f"|class={alloc.predicted_class}"
            f"|phase={alloc.phase}"
            f"|action={alloc.action or 'unknown'}"
        )
        grouped_rows[alloc.phase].append(
            (kind, name, alloc.ptr, alloc.end, alloc.size_bytes, alloc.alloc_wall_time)
        )
        phase_counts[alloc.phase] += 1
        class_counts[str(alloc.predicted_class)] += 1
        exported += 1

    output_path = Path(args.output_log)
    with output_path.open("w", encoding="utf-8") as handle:
        for phase in sorted(grouped_rows):
            rows = grouped_rows[phase]
            timestamp = next(
                (wall_ts for *_, wall_ts in rows if wall_ts is not None),
                "allocator_trace",
            )
            handle.write(
                f"[{timestamp}] phase={phase} pid=0 model=allocator_trace\n"
            )
            handle.write("kind,name,start,end,size_bytes,size_mb\n")
            for kind, name, start, end, size_bytes, _wall_ts in sorted(
                rows, key=lambda item: (item[2], item[3], item[1])
            ):
                handle.write(
                    f"{kind},{name},0x{start:x},0x{end:x},{size_bytes},"
                    f"{size_bytes / (1024 * 1024):.3f}\n"
                )
            handle.write("\n")

    summary = {
        "allocator_log": str(allocator_log_path),
        "output_log": str(output_path),
        "min_size_bytes": args.min_size_bytes,
        "selected_classes": sorted(selected_classes),
        "selected_phases": sorted(selected_phases) if selected_phases is not None else None,
        "exported_region_rows": exported,
        "phase_counts": dict(phase_counts),
        "class_counts": dict(class_counts),
        "allocator_trace_allocation_count": allocator_report["allocation_count"],
        "allocator_trace_matched_lines": allocator_report["matched_trace_lines"],
        "warnings": allocator_report["warnings"],
    }

    if args.summary_json:
        with Path(args.summary_json).open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)

    print("Allocator Workspace Region Export")
    print(f"- allocator_log={allocator_log_path}")
    print(f"- output_log={output_path}")
    print(f"- exported_region_rows={exported}")
    for phase, count in phase_counts.most_common():
        print(f"- phase={phase} rows={count}")
    for predicted_class, count in class_counts.most_common():
        print(f"- class={predicted_class} rows={count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

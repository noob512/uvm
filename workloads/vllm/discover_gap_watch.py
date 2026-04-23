#!/usr/bin/env python3
"""Discover the hottest unknown gap from the current run and emit a watch file.

This script is designed for same-process workflows:
1. Run a small probe workload against a live vLLM server.
2. Parse the current fault/address logs from that *same* run.
3. Write a gap-watch control file that the allocator can hot-reload.
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
from collections import Counter
from pathlib import Path

from analyze_uvm_fault_addresses import (
    Region,
    parse_address_log,
    parse_fault_record,
    select_sections,
)
from deep_dive_uvm_faults import build_concrete_regions, build_gaps


ACCESS_TYPE_RE = re.compile(r"(?:access_type|访问类型)=(?P<value>[^,\s]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover the current run's unknown hot gap and emit a watch control file."
    )
    parser.add_argument(
        "--address-log",
        default="/tmp/vllm_uvm_address_regions.log",
        help="Path to the vLLM address region log.",
    )
    parser.add_argument(
        "--fault-log",
        default="/tmp/uvm_kv_fault_addrs.log",
        help="Path to the per-fault address log.",
    )
    parser.add_argument(
        "--control-file",
        default=None,
        help="Output control file consumed by the allocator.",
    )
    parser.add_argument(
        "--no-write-control",
        action="store_true",
        help="Only print/write summary JSON; do not update the allocator control file.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional JSON summary file.",
    )
    parser.add_argument(
        "--target-gap",
        type=int,
        default=2,
        help="Preferred gap index to watch. Default: 2.",
    )
    parser.add_argument(
        "--fallback-to-hottest",
        type=int,
        choices=(0, 1),
        default=1,
        help="If target gap has no faults, fall back to the hottest unknown gap.",
    )
    parser.add_argument(
        "--pid",
        type=int,
        default=None,
        help="Optional pid override for address-log section selection.",
    )
    parser.add_argument(
        "--use-raw-address",
        action="store_true",
        help="Classify using raw addresses instead of page-aligned addresses.",
    )
    parser.add_argument(
        "--watch-name",
        default="auto_gap_watch",
        help="Name to store in the control file.",
    )
    parser.add_argument(
        "--all-classes",
        type=int,
        choices=(0, 1),
        default=1,
        help="Whether the dynamic watch should capture all classes.",
    )
    parser.add_argument(
        "--min-bytes",
        type=int,
        default=4096,
        help="Minimum allocation size to trace for watched overlaps.",
    )
    parser.add_argument(
        "--start-line",
        type=int,
        default=1,
        help="Only analyze fault-log lines starting from this 1-based line number.",
    )
    return parser.parse_args()


def build_gap_lookup(gaps: list) -> tuple[list[int], list]:
    ordered = sorted(gaps, key=lambda gap: gap.start)
    return [gap.start for gap in ordered], ordered


def find_gap(starts: list[int], gaps: list, address: int):
    idx = bisect.bisect_right(starts, address) - 1
    if idx < 0:
        return None
    candidate = gaps[idx]
    if address <= candidate.end:
        return candidate
    return None


def access_label(line: str) -> str:
    match = ACCESS_TYPE_RE.search(line)
    if match is None:
        return "unknown"
    return match.group("value").upper()


def summarize_unknown_gaps(
    fault_log: Path,
    gaps: list,
    use_raw_address: bool,
    start_line: int,
) -> tuple[dict[int, dict[str, object]], int]:
    gap_starts, ordered_gaps = build_gap_lookup(gaps)
    per_gap: dict[int, dict[str, object]] = {}
    total_unknown_faults = 0

    with fault_log.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line_no < start_line:
                continue
            record = parse_fault_record(line, line_no)
            if record is None:
                continue
            address = record.selected_address(use_raw_address)
            if address is None:
                continue

            gap = find_gap(gap_starts, ordered_gaps, address)
            if gap is None:
                continue

            total_unknown_faults += 1
            entry = per_gap.setdefault(
                gap.index,
                {
                    "gap_index": gap.index,
                    "start_hex": gap.start_hex,
                    "end_hex": gap.end_hex,
                    "size_bytes": gap.size_bytes,
                    "size_mib": gap.size_mib,
                    "faults": 0,
                    "unique_pages": set(),
                    "access_counts": Counter(),
                    "left_region": gap.left_region.name if gap.left_region else None,
                    "right_region": gap.right_region.name if gap.right_region else None,
                },
            )
            entry["faults"] += 1
            entry["unique_pages"].add(address)
            entry["access_counts"][access_label(line)] += 1

    for entry in per_gap.values():
        entry["unique_pages"] = len(entry["unique_pages"])
        access_counts: Counter = entry["access_counts"]
        total = sum(access_counts.values())
        entry["access_counts"] = dict(access_counts)
        entry["access_ratios"] = {
            key: (value / total if total > 0 else 0.0)
            for key, value in access_counts.items()
        }
        entry["avg_faults_per_unique_page"] = (
            entry["faults"] / entry["unique_pages"]
            if entry["unique_pages"] > 0
            else 0.0
        )
    return per_gap, total_unknown_faults


def write_control_file(
    path: Path,
    enabled: bool,
    name: str,
    start_hex: str,
    end_hex: str,
    all_classes: int,
    min_bytes: int,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"enabled={1 if enabled else 0}\n")
        handle.write(f"name={name}\n")
        handle.write(f"start={start_hex}\n")
        handle.write(f"end={end_hex}\n")
        handle.write(f"all_classes={all_classes}\n")
        handle.write(f"min_bytes={min_bytes}\n")


def main() -> int:
    args = parse_args()
    address_log = Path(args.address_log)
    fault_log = Path(args.fault_log)
    control_file = Path(args.control_file) if args.control_file else None

    if not address_log.is_file():
        raise SystemExit(f"address log not found: {address_log}")
    if not fault_log.is_file():
        raise SystemExit(f"fault log not found: {fault_log}")

    sections, warnings = parse_address_log(address_log)
    selected_pid, selected_sections = select_sections(sections, args.pid)
    concrete_regions: list[Region] = build_concrete_regions(selected_sections)
    gaps = build_gaps(concrete_regions)
    if not gaps:
        raise SystemExit("no concrete gaps found in the current address log")

    gap_summaries, total_unknown_faults = summarize_unknown_gaps(
        fault_log,
        gaps,
        args.use_raw_address,
        max(args.start_line, 1),
    )

    requested_gap = gap_summaries.get(args.target_gap)
    selected_gap = requested_gap
    fallback_used = False
    if selected_gap is None or selected_gap["faults"] <= 0:
        if args.fallback_to_hottest:
            hottest = sorted(
                gap_summaries.values(),
                key=lambda item: (item["faults"], item["unique_pages"]),
                reverse=True,
            )
            if not hottest:
                raise SystemExit("no unknown faults were found in any gap")
            selected_gap = hottest[0]
            fallback_used = True
        else:
            raise SystemExit(
                f"target gap #{args.target_gap} has no unknown faults and fallback is disabled"
            )

    assert selected_gap is not None
    if not args.no_write_control:
        if control_file is None:
            raise SystemExit("--control-file is required unless --no-write-control is set")
        write_control_file(
            control_file,
            True,
            args.watch_name,
            selected_gap["start_hex"],
            selected_gap["end_hex"],
            args.all_classes,
            args.min_bytes,
        )

    selected_gap = dict(selected_gap)
    selected_gap["fault_share_of_unknown"] = (
        selected_gap["faults"] / total_unknown_faults if total_unknown_faults > 0 else 0.0
    )

    summary = {
        "address_log": str(address_log),
        "fault_log": str(fault_log),
        "control_file": str(control_file) if control_file is not None else None,
        "control_written": not args.no_write_control,
        "selected_pid": selected_pid,
        "start_line": max(args.start_line, 1),
        "target_gap": args.target_gap,
        "fallback_used": fallback_used,
        "total_unknown_faults": total_unknown_faults,
        "selected_gap": selected_gap,
        "top_gaps_by_faults": sorted(
            (
                {
                    "gap_index": entry["gap_index"],
                    "start_hex": entry["start_hex"],
                    "end_hex": entry["end_hex"],
                    "size_mib": entry["size_mib"],
                    "faults": entry["faults"],
                    "unique_pages": entry["unique_pages"],
                }
                for entry in gap_summaries.values()
            ),
            key=lambda item: (item["faults"], item["unique_pages"]),
            reverse=True,
        )[:10],
        "warnings": warnings,
    }

    if args.summary_json:
        with Path(args.summary_json).open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)

    print("Auto Gap Watch Discovery")
    print(f"- selected_pid={selected_pid}")
    print(f"- target_gap={args.target_gap}")
    print(f"- fallback_used={fallback_used}")
    print(f"- selected_gap={selected_gap['gap_index']}")
    print(f"- start={selected_gap['start_hex']}")
    print(f"- end={selected_gap['end_hex']}")
    print(f"- faults={selected_gap['faults']}")
    print(f"- unique_pages={selected_gap['unique_pages']}")
    print(f"- fault_share_of_unknown={selected_gap['fault_share_of_unknown']:.4f}")
    print(f"- control_file={control_file if control_file is not None else 'none'}")
    print(f"- control_written={0 if args.no_write_control else 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

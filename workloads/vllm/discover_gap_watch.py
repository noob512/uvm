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
TRACE_POLICY_RE = re.compile(
    r"TRACE_POLICY alloc_id=(?P<alloc_id>\d+) ptr=(?P<ptr>0x[0-9a-fA-F]+) "
    r"end=(?P<end>0x[0-9a-fA-F]+) size_bytes=(?P<size_bytes>\d+) "
    r"size_bucket=(?P<size_bucket>[^ ]+) device=(?P<device>-?\d+) "
    r"phase=(?P<phase>\S+) predicted_class=(?P<predicted_class>[^ ]+) "
    r"action=(?P<action>[^ ]+) policy_source=(?P<policy_source>[^ ]+) "
    r"gap_watch_class_match=(?P<gap_watch_class_match>\d+) "
    r"gap_overlap_bytes=(?P<gap_overlap_bytes>\d+) "
    r"action_success=(?P<action_success>\d+) action_error=(?P<action_error>[^ ]+)"
)
TRACE_FREE_RE = re.compile(
    r"TRACE_FREE free_id=(?P<free_id>\d+) ptr=(?P<ptr>0x[0-9a-fA-F]+) "
    r"end=(?P<end>0x[0-9a-fA-F]+) size_bytes=(?P<size_bytes>\d+) "
    r"size_mb=(?P<size_mb>[0-9.]+) device=(?P<device>-?\d+) "
    r"phase=(?P<phase>\S+) alloc_id=(?P<alloc_id>\d+) "
    r"alloc_phase=(?P<alloc_phase>\S+) lifetime_s=(?P<lifetime_s>-?[0-9.]+)"
)


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
        "--allocator-log",
        default=None,
        help="Optional allocator trace log used to classify the selected gap.",
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
    parser.add_argument(
        "--policy-action-override",
        default=None,
        choices=(
            "observe",
            "prefetch",
            "advise_prefetch",
            "device_direct_trace",
            "device_direct",
        ),
        help="Override the recommended policy action written to the control file.",
    )
    parser.add_argument(
        "--target-class-override",
        default=None,
        help="Override the recommended target class written to the control file.",
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


def classify_gap_kind_from_allocator(
    allocator_log: Path,
    gap_start: int,
    gap_end: int,
) -> dict[str, object]:
    if not allocator_log.is_file():
        return {
            "allocator_log_found": False,
            "dominant_phase": None,
            "dominant_predicted_class": None,
            "recommended_target_class": "any",
            "recommended_policy_action": "prefetch",
            "overlap_allocations": 0,
        }

    overlap_allocations = 0
    phase_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    overlap_bytes_by_class: Counter[str] = Counter()
    lifetime_by_alloc_id: dict[int, float] = {}
    overlap_alloc_ids: set[int] = set()

    with allocator_log.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            policy_match = TRACE_POLICY_RE.search(line)
            if policy_match is not None:
                logged_overlap_bytes = policy_match.group("gap_overlap_bytes")
                if logged_overlap_bytes is not None:
                    overlap_bytes = int(logged_overlap_bytes)
                    if overlap_bytes <= 0:
                        continue
                else:
                    start = int(policy_match.group("ptr"), 16)
                    end = int(policy_match.group("end"), 16)
                    overlap_start = max(gap_start, start)
                    overlap_end = min(gap_end, end)
                    if overlap_start > overlap_end:
                        continue
                    overlap_bytes = overlap_end - overlap_start + 1
                alloc_id = int(policy_match.group("alloc_id"))
                predicted_class = policy_match.group("predicted_class")
                phase = policy_match.group("phase")
                overlap_allocations += 1
                overlap_alloc_ids.add(alloc_id)
                phase_counts[phase] += 1
                class_counts[predicted_class] += 1
                overlap_bytes_by_class[predicted_class] += overlap_bytes
                continue

            free_match = TRACE_FREE_RE.search(line)
            if free_match is not None:
                alloc_id = int(free_match.group("alloc_id"))
                if alloc_id in overlap_alloc_ids:
                    lifetime_s = float(free_match.group("lifetime_s"))
                    if lifetime_s >= 0:
                        lifetime_by_alloc_id[alloc_id] = lifetime_s

    dominant_phase = phase_counts.most_common(1)[0][0] if phase_counts else None
    dominant_predicted_class = (
        overlap_bytes_by_class.most_common(1)[0][0]
        if overlap_bytes_by_class
        else (class_counts.most_common(1)[0][0] if class_counts else None)
    )
    recommended_target_class = dominant_predicted_class or "any"
    recommended_policy_action = "prefetch"

    median_lifetime_s = None
    p95_lifetime_s = None
    if lifetime_by_alloc_id:
        ordered = sorted(lifetime_by_alloc_id.values())
        median_lifetime_s = ordered[len(ordered) // 2]
        p95_lifetime_s = ordered[max(len(ordered) - 1, 0) * 95 // 100]

    dominant_is_runtime_phase = (
        dominant_phase == "enabled"
        or (isinstance(dominant_phase, str) and dominant_phase.startswith("enabled:"))
    )
    if (
        dominant_is_runtime_phase
        and dominant_predicted_class in {
            "unknown_managed",
            "runtime_scratch",
            "runtime_workspace",
        }
        and median_lifetime_s is not None
        and median_lifetime_s < 0.01
    ):
        recommended_target_class = "gap_hot_runtime_scratch"

    if dominant_predicted_class in {"runtime_scratch", "runtime_workspace"}:
        recommended_policy_action = (
            "advise_prefetch"
            if median_lifetime_s is not None and median_lifetime_s < 0.01
            else "prefetch"
        )
    elif dominant_predicted_class == "warmup_workspace":
        recommended_policy_action = "advise_prefetch"

    return {
        "allocator_log_found": True,
        "dominant_phase": dominant_phase,
        "dominant_predicted_class": dominant_predicted_class,
        "recommended_target_class": recommended_target_class,
        "recommended_policy_action": recommended_policy_action,
        "overlap_allocations": overlap_allocations,
        "phase_counts": dict(phase_counts),
        "predicted_class_counts": dict(class_counts),
        "predicted_class_overlap_bytes": dict(overlap_bytes_by_class),
        "median_lifetime_s": median_lifetime_s,
        "p95_lifetime_s": p95_lifetime_s,
    }


def write_control_file(
    path: Path,
    enabled: bool,
    name: str,
    start_hex: str,
    end_hex: str,
    all_classes: int,
    min_bytes: int,
    target_class: str,
    policy_action: str,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"enabled={1 if enabled else 0}\n")
        handle.write(f"name={name}\n")
        handle.write(f"start={start_hex}\n")
        handle.write(f"end={end_hex}\n")
        handle.write(f"all_classes={all_classes}\n")
        handle.write(f"min_bytes={min_bytes}\n")
        handle.write(f"target_class={target_class}\n")
        handle.write(f"policy_action={policy_action}\n")


def main() -> int:
    args = parse_args()
    address_log = Path(args.address_log)
    fault_log = Path(args.fault_log)
    control_file = Path(args.control_file) if args.control_file else None
    allocator_log = Path(args.allocator_log) if args.allocator_log else None

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

    selected_gap = dict(selected_gap)
    selected_gap["fault_share_of_unknown"] = (
        selected_gap["faults"] / total_unknown_faults if total_unknown_faults > 0 else 0.0
    )
    allocator_gap_kind = (
        classify_gap_kind_from_allocator(
            allocator_log,
            int(selected_gap["start_hex"], 16),
            int(selected_gap["end_hex"], 16),
        )
        if allocator_log is not None
        else {
            "allocator_log_found": False,
            "dominant_phase": None,
            "dominant_predicted_class": None,
            "recommended_target_class": "any",
            "recommended_policy_action": "prefetch",
            "overlap_allocations": 0,
        }
    )

    recommended_target_class = str(
        allocator_gap_kind.get("recommended_target_class", "any")
    )
    recommended_policy_action = str(
        allocator_gap_kind.get("recommended_policy_action", "prefetch")
    )
    effective_target_class = (
        args.target_class_override
        if args.target_class_override is not None
        else recommended_target_class
    )
    effective_policy_action = (
        args.policy_action_override
        if args.policy_action_override is not None
        else recommended_policy_action
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
            effective_target_class,
            effective_policy_action,
        )

    summary = {
        "address_log": str(address_log),
        "fault_log": str(fault_log),
        "allocator_log": str(allocator_log) if allocator_log is not None else None,
        "control_file": str(control_file) if control_file is not None else None,
        "control_written": not args.no_write_control,
        "selected_pid": selected_pid,
        "start_line": max(args.start_line, 1),
        "target_gap": args.target_gap,
        "fallback_used": fallback_used,
        "total_unknown_faults": total_unknown_faults,
        "selected_gap": selected_gap,
        "selected_gap_allocator_classification": allocator_gap_kind,
        "effective_target_class": effective_target_class,
        "effective_policy_action": effective_policy_action,
        "target_class_overridden": args.target_class_override is not None,
        "policy_action_overridden": args.policy_action_override is not None,
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
    print(f"- dominant_predicted_class={allocator_gap_kind.get('dominant_predicted_class')}")
    print(f"- dominant_phase={allocator_gap_kind.get('dominant_phase')}")
    print(f"- recommended_target_class={recommended_target_class}")
    print(f"- recommended_policy_action={recommended_policy_action}")
    print(f"- effective_target_class={effective_target_class}")
    print(f"- effective_policy_action={effective_policy_action}")
    print(f"- control_file={control_file if control_file is not None else 'none'}")
    print(f"- control_written={0 if args.no_write_control else 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

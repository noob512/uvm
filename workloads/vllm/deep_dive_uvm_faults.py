#!/usr/bin/env python3
"""Deep-dive UVM fault analysis for unknown gaps and access patterns."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from analyze_uvm_fault_addresses import (
    Region,
    RegionLookup,
    merge_intervals,
    parse_address_log,
    parse_fault_record,
    select_sections,
)


FAULT_TYPE_RE = re.compile(r"(?:fault_type|中断类型)=(?P<value>[^,\s]+)")
ACCESS_TYPE_RE = re.compile(r"(?:access_type|访问类型)=(?P<value>[^,\s]+)")
TRACE_ALLOC_RE = re.compile(
    r"TRACE_ALLOC alloc_id=(?P<alloc_id>\d+) ptr=(?P<ptr>0x[0-9a-fA-F]+) "
    r"end=(?P<end>0x[0-9a-fA-F]+) size_bytes=(?P<size_bytes>\d+) "
    r"size_mb=(?P<size_mb>[0-9.]+) device=(?P<device>-?\d+) "
    r"phase=(?P<phase>\S+) total_bytes=(?P<total_bytes>\d+) "
    r"peak_bytes=(?P<peak_bytes>\d+)"
)
TRACE_FREE_RE = re.compile(
    r"TRACE_FREE free_id=(?P<free_id>\d+) ptr=(?P<ptr>0x[0-9a-fA-F]+) "
    r"end=(?P<end>0x[0-9a-fA-F]+) size_bytes=(?P<size_bytes>\d+) "
    r"size_mb=(?P<size_mb>[0-9.]+) device=(?P<device>-?\d+) "
    r"phase=(?P<phase>\S+) alloc_id=(?P<alloc_id>\d+) "
    r"alloc_phase=(?P<alloc_phase>\S+) lifetime_s=(?P<lifetime_s>-?[0-9.]+) "
    r"total_bytes=(?P<total_bytes>\d+)"
)
TRACE_POLICY_RE = re.compile(
    r"TRACE_POLICY alloc_id=(?P<alloc_id>\d+) ptr=(?P<ptr>0x[0-9a-fA-F]+) "
    r"end=(?P<end>0x[0-9a-fA-F]+) size_bytes=(?P<size_bytes>\d+) "
    r"size_bucket=(?P<size_bucket>[^ ]+) device=(?P<device>-?\d+) "
    r"phase=(?P<phase>\S+) predicted_class=(?P<predicted_class>[^ ]+) "
    r"action=(?P<action>[^ ]+) (?:policy_source=(?P<policy_source>[^ ]+) "
    r"gap_watch_class_match=(?P<gap_watch_class_match>\d+) "
    r"gap_overlap_bytes=(?P<gap_overlap_bytes>\d+) )?"
    r"action_success=(?P<action_success>\d+) "
    r"action_error=(?P<action_error>[^ ]+)"
)
TRACE_PREFIX_RE = re.compile(
    r"^\[(?P<wall_ts>[^\]]+)\]\s+\[\+(?P<elapsed_s>[0-9.]+)s\]"
)


@dataclass
class AllocationTrace:
    alloc_id: int
    ptr: int
    end: int
    size_bytes: int
    device: int
    phase: str
    predicted_class: str | None = None
    action: str | None = None
    action_success: bool | None = None
    action_error: str | None = None
    size_bucket: str | None = None
    alloc_wall_time: str | None = None
    alloc_elapsed_s: float | None = None
    freed: bool = False
    free_phase: str | None = None
    lifetime_s: float | None = None
    free_wall_time: str | None = None
    free_elapsed_s: float | None = None

    @property
    def ptr_hex(self) -> str:
        return f"0x{self.ptr:x}"

    @property
    def end_hex(self) -> str:
        return f"0x{self.end:x}"


@dataclass(frozen=True)
class Gap:
    index: int
    start: int
    end: int
    left_region: Region | None
    right_region: Region | None

    @property
    def size_bytes(self) -> int:
        return self.end - self.start + 1

    @property
    def size_mib(self) -> float:
        return self.size_bytes / (1024 * 1024)

    @property
    def start_hex(self) -> str:
        return f"0x{self.start:x}"

    @property
    def end_hex(self) -> str:
        return f"0x{self.end:x}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deep-dive UVM faults, unknown gaps, and warmup/workspace heuristics."
    )
    parser.add_argument(
        "--address-log",
        default="/tmp/vllm_uvm_address_regions.log",
        help="Path to vLLM address region log.",
    )
    parser.add_argument(
        "--fault-log",
        default="/tmp/uvm_kv_fault_addrs.log",
        help="Path to UVM replayable fault address log.",
    )
    parser.add_argument(
        "--allocator-log",
        default=None,
        help=(
            "Optional allocator trace log emitted by the UVM allocator. "
            "When provided, the script correlates unknown gaps with traced allocations."
        ),
    )
    parser.add_argument(
        "--pid",
        type=int,
        default=None,
        help="Restrict address-log sections to a specific pid. Default: latest pid in the log.",
    )
    parser.add_argument(
        "--use-raw-address",
        action="store_true",
        help="Match faults using raw addresses instead of page-aligned addresses.",
    )
    parser.add_argument(
        "--top-gaps",
        type=int,
        default=10,
        help="How many unknown gaps to show in the report.",
    )
    parser.add_argument(
        "--gap-csv",
        default=None,
        help="Optional output CSV for per-gap heat statistics.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional output JSON summary file.",
    )
    parser.add_argument(
        "--max-faults",
        type=int,
        default=None,
        help="Stop after analyzing N parsed fault records.",
    )
    parser.add_argument(
        "--focus-gap",
        type=int,
        default=None,
        help="Optional gap index to export a detailed allocator overlap timeline for.",
    )
    parser.add_argument(
        "--gap-alloc-csv",
        default=None,
        help="Optional CSV output for per-allocation overlap details of --focus-gap.",
    )
    parser.add_argument(
        "--gap-reuse-csv",
        default=None,
        help="Optional CSV output for repeated-address reuse patterns of --focus-gap.",
    )
    parser.add_argument(
        "--gap-report-json",
        default=None,
        help=(
            "Optional JSON output for the summarized --focus-gap report. "
            "Use --gap-alloc-csv for the full ordered overlap list."
        ),
    )
    parser.add_argument(
        "--gap-preview-limit",
        type=int,
        default=200,
        help="How many ordered overlaps / reuse patterns to keep in JSON summaries.",
    )
    return parser.parse_args()


def build_concrete_regions(selected_sections: list) -> list[Region]:
    return [
        region
        for section in selected_sections
        for region in section.rows
        if region.major_type in {"weight", "kv_cache"} and not region.is_summary
    ]


def find_region_before(sorted_regions: list[Region], address: int) -> Region | None:
    idx = bisect.bisect_left([region.start for region in sorted_regions], address) - 1
    while idx >= 0:
        region = sorted_regions[idx]
        if region.end < address:
            return region
        idx -= 1
    return None


def find_region_after(sorted_regions: list[Region], address: int) -> Region | None:
    idx = bisect.bisect_right([region.start for region in sorted_regions], address)
    while idx < len(sorted_regions):
        region = sorted_regions[idx]
        if region.start > address:
            return region
        idx += 1
    return None


def build_gaps(concrete_regions: list[Region]) -> list[Gap]:
    if not concrete_regions:
        return []

    sorted_regions = sorted(concrete_regions, key=lambda region: (region.start, region.end))
    merged = merge_intervals(sorted_regions)
    gaps: list[Gap] = []
    for index, ((left_start, left_end), (right_start, right_end)) in enumerate(
        zip(merged, merged[1:]),
        start=1,
    ):
        gap_start = left_end + 1
        gap_end = right_start - 1
        if gap_start > gap_end:
            continue
        gaps.append(
            Gap(
                index=index,
                start=gap_start,
                end=gap_end,
                left_region=find_region_before(sorted_regions, gap_start),
                right_region=find_region_after(sorted_regions, gap_end),
            )
        )
    return gaps


def merge_plain_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []

    sorted_intervals = sorted(intervals)
    merged: list[tuple[int, int]] = [sorted_intervals[0]]
    for start, end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def parse_allocator_log(path: Path) -> dict[str, object]:
    allocations_by_id: dict[int, AllocationTrace] = {}
    warnings: list[str] = []
    total_lines = 0
    matched_lines = 0

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for total_lines, line in enumerate(handle, start=1):
            prefix_match = TRACE_PREFIX_RE.search(line)
            wall_ts = prefix_match.group("wall_ts") if prefix_match else None
            elapsed_s = (
                float(prefix_match.group("elapsed_s"))
                if prefix_match is not None
                else None
            )

            alloc_match = TRACE_ALLOC_RE.search(line)
            if alloc_match is not None:
                matched_lines += 1
                alloc_id = int(alloc_match.group("alloc_id"))
                allocations_by_id[alloc_id] = AllocationTrace(
                    alloc_id=alloc_id,
                    ptr=int(alloc_match.group("ptr"), 16),
                    end=int(alloc_match.group("end"), 16),
                    size_bytes=int(alloc_match.group("size_bytes")),
                    device=int(alloc_match.group("device")),
                    phase=alloc_match.group("phase"),
                    alloc_wall_time=wall_ts,
                    alloc_elapsed_s=elapsed_s,
                )
                continue

            policy_match = TRACE_POLICY_RE.search(line)
            if policy_match is not None:
                matched_lines += 1
                alloc_id = int(policy_match.group("alloc_id"))
                allocation = allocations_by_id.get(alloc_id)
                if allocation is None:
                    allocation = AllocationTrace(
                        alloc_id=alloc_id,
                        ptr=int(policy_match.group("ptr"), 16),
                        end=int(policy_match.group("end"), 16),
                        size_bytes=int(policy_match.group("size_bytes")),
                        device=int(policy_match.group("device")),
                        phase=policy_match.group("phase"),
                        alloc_wall_time=wall_ts,
                        alloc_elapsed_s=elapsed_s,
                    )
                    allocations_by_id[alloc_id] = allocation
                allocation.predicted_class = policy_match.group("predicted_class")
                allocation.action = policy_match.group("action")
                allocation.action_success = policy_match.group("action_success") == "1"
                allocation.action_error = policy_match.group("action_error")
                allocation.size_bucket = policy_match.group("size_bucket")
                if allocation.alloc_wall_time is None:
                    allocation.alloc_wall_time = wall_ts
                if allocation.alloc_elapsed_s is None:
                    allocation.alloc_elapsed_s = elapsed_s
                continue

            free_match = TRACE_FREE_RE.search(line)
            if free_match is None:
                continue

            matched_lines += 1
            alloc_id = int(free_match.group("alloc_id"))
            allocation = allocations_by_id.get(alloc_id)
            if allocation is None:
                warnings.append(
                    f"line {total_lines}: free references unknown alloc_id={alloc_id}"
                )
                continue
            allocation.freed = True
            allocation.free_phase = free_match.group("phase")
            allocation.lifetime_s = float(free_match.group("lifetime_s"))
            allocation.free_wall_time = wall_ts
            allocation.free_elapsed_s = elapsed_s

    return {
        "path": str(path),
        "total_lines": total_lines,
        "matched_trace_lines": matched_lines,
        "allocation_count": len(allocations_by_id),
        "allocations": list(allocations_by_id.values()),
        "warnings": warnings,
    }


def phase_looks_like_warmup_or_workspace(phase: str) -> bool:
    lowered = phase.lower()
    keywords = (
        "warmup",
        "autotune",
        "profile",
        "preinit",
        "capture_model",
        "compile_warmup",
    )
    return any(keyword in lowered for keyword in keywords)


def correlate_allocator_allocations(
    gap_summaries: list[dict[str, object]],
    allocator_report: dict[str, object],
    top_gaps: int,
) -> dict[str, object]:
    allocations: list[AllocationTrace] = allocator_report["allocations"]  # type: ignore[assignment]
    per_gap: list[dict[str, object]] = []

    for gap in gap_summaries[:top_gaps]:
        gap_start = int(gap["start_hex"], 16)
        gap_end = int(gap["end_hex"], 16)
        gap_size = int(gap["size_bytes"])
        overlapping: list[dict[str, object]] = []
        phase_overlap_bytes: Counter[str] = Counter()
        class_overlap_bytes: Counter[str] = Counter()
        freed_overlap_bytes = 0
        live_overlap_bytes = 0
        exact_match_count = 0
        overlap_intervals: list[tuple[int, int]] = []
        freed_lifetimes_s: list[float] = []

        for alloc in allocations:
            if alloc.size_bytes <= 0 or alloc.ptr == 0:
                continue
            overlap_start = max(gap_start, alloc.ptr)
            overlap_end = min(gap_end, alloc.end)
            if overlap_start > overlap_end:
                continue

            overlap_bytes = overlap_end - overlap_start + 1
            phase_overlap_bytes[alloc.phase] += overlap_bytes
            if alloc.predicted_class is not None:
                class_overlap_bytes[alloc.predicted_class] += overlap_bytes
            overlap_intervals.append((overlap_start, overlap_end))
            if alloc.freed:
                freed_overlap_bytes += overlap_bytes
                if alloc.lifetime_s is not None and alloc.lifetime_s >= 0:
                    freed_lifetimes_s.append(alloc.lifetime_s)
            else:
                live_overlap_bytes += overlap_bytes
            if alloc.ptr == gap_start and alloc.end == gap_end:
                exact_match_count += 1

            overlapping.append(
                {
                    "alloc_id": alloc.alloc_id,
                    "ptr_hex": alloc.ptr_hex,
                    "end_hex": alloc.end_hex,
                    "size_bytes": alloc.size_bytes,
                    "phase": alloc.phase,
                    "predicted_class": alloc.predicted_class,
                    "action": alloc.action,
                    "action_success": alloc.action_success,
                    "action_error": alloc.action_error,
                    "size_bucket": alloc.size_bucket,
                    "alloc_wall_time": alloc.alloc_wall_time,
                    "alloc_elapsed_s": alloc.alloc_elapsed_s,
                    "freed": alloc.freed,
                    "free_phase": alloc.free_phase,
                    "lifetime_s": alloc.lifetime_s,
                    "free_wall_time": alloc.free_wall_time,
                    "free_elapsed_s": alloc.free_elapsed_s,
                    "overlap_bytes": overlap_bytes,
                    "overlap_ratio_of_gap": overlap_bytes / gap_size if gap_size else 0.0,
                    "exact_match": alloc.ptr == gap_start and alloc.end == gap_end,
                }
            )

        overlapping.sort(key=lambda item: (-item["overlap_bytes"], item["alloc_id"]))
        dominant_phase = phase_overlap_bytes.most_common(1)[0][0] if phase_overlap_bytes else None
        dominant_phase_overlap = (
            phase_overlap_bytes.most_common(1)[0][1] if phase_overlap_bytes else 0
        )
        dominant_predicted_class = (
            class_overlap_bytes.most_common(1)[0][0] if class_overlap_bytes else None
        )
        warmup_overlap_bytes = sum(
            overlap_bytes
            for phase, overlap_bytes in phase_overlap_bytes.items()
            if phase_looks_like_warmup_or_workspace(phase)
        )
        cumulative_overlap_bytes = sum(item["overlap_bytes"] for item in overlapping)
        merged_overlap_intervals = merge_plain_intervals(overlap_intervals)
        unique_coverage_bytes = sum(
            interval_end - interval_start + 1
            for interval_start, interval_end in merged_overlap_intervals
        )
        lifetime_stats = {
            "freed_count": len(freed_lifetimes_s),
            "min_s": min(freed_lifetimes_s) if freed_lifetimes_s else None,
            "median_s": statistics.median(freed_lifetimes_s) if freed_lifetimes_s else None,
            "p95_s": (
                sorted(freed_lifetimes_s)[max(len(freed_lifetimes_s) - 1, 0) * 95 // 100]
                if freed_lifetimes_s
                else None
            ),
            "max_s": max(freed_lifetimes_s) if freed_lifetimes_s else None,
        }

        if warmup_overlap_bytes > 0 and freed_overlap_bytes >= max(warmup_overlap_bytes // 2, 1):
            likely_kind = "warmup_workspace"
        elif dominant_predicted_class in {
            "runtime_scratch",
            "runtime_workspace",
            "warmup_workspace",
            "kv_persistent",
            "weight_persistent",
        }:
            likely_kind = dominant_predicted_class
        elif dominant_phase == "enabled" and freed_lifetimes_s:
            median_lifetime = lifetime_stats["median_s"] or 0.0
            if median_lifetime < 0.01:
                likely_kind = "runtime_scratch"
            else:
                likely_kind = "runtime_workspace"
        elif dominant_phase == "initialize_kv_cache":
            likely_kind = "kv_related_workspace"
        elif dominant_phase == "load_model":
            likely_kind = "weight_or_load_workspace"
        elif dominant_phase == "uvm_enable:cublas_preinit":
            likely_kind = "warmup_workspace"
        elif dominant_phase is not None:
            likely_kind = f"phase_scoped:{dominant_phase}"
        else:
            likely_kind = "unresolved_unknown_gap"

        if exact_match_count > 0 and warmup_overlap_bytes > 0 and freed_overlap_bytes >= warmup_overlap_bytes:
            assessment = "strong_evidence_of_temporary_workspace_or_scratch"
        elif warmup_overlap_bytes > 0 and freed_overlap_bytes >= max(warmup_overlap_bytes // 2, 1):
            assessment = "likely_temporary_workspace_or_scratch"
        elif dominant_phase is not None:
            assessment = f"overlaps_traced_allocations_from_{dominant_phase}"
        else:
            assessment = "no_traced_allocator_overlap"

        per_gap.append(
            {
                "gap_index": gap["gap_index"],
                "start_hex": gap["start_hex"],
                "end_hex": gap["end_hex"],
                "dominant_phase": dominant_phase,
                "dominant_predicted_class": dominant_predicted_class,
                "dominant_phase_overlap_bytes": dominant_phase_overlap,
                "warmup_like_overlap_bytes": warmup_overlap_bytes,
                "freed_overlap_bytes": freed_overlap_bytes,
                "live_overlap_bytes": live_overlap_bytes,
                "cumulative_overlap_bytes": cumulative_overlap_bytes,
                "cumulative_overlap_ratio_of_gap": (
                    cumulative_overlap_bytes / gap_size if gap_size else 0.0
                ),
                "unique_coverage_bytes": unique_coverage_bytes,
                "unique_coverage_ratio_of_gap": (
                    unique_coverage_bytes / gap_size if gap_size else 0.0
                ),
                "exact_match_count": exact_match_count,
                "phase_overlap_bytes": dict(phase_overlap_bytes),
                "predicted_class_overlap_bytes": dict(class_overlap_bytes),
                "lifetime_stats": lifetime_stats,
                "likely_kind": likely_kind,
                "assessment": assessment,
                "top_overlapping_allocations": overlapping[:10],
            }
        )

    return {
        "allocator_log": allocator_report["path"],
        "allocation_count": allocator_report["allocation_count"],
        "matched_trace_lines": allocator_report["matched_trace_lines"],
        "warnings": allocator_report["warnings"],
        "per_gap": per_gap,
    }


def find_gap_summary(gap_summaries: list[dict[str, object]], gap_index: int) -> dict[str, object] | None:
    for gap_summary in gap_summaries:
        if int(gap_summary["gap_index"]) == gap_index:
            return gap_summary
    return None


def overlap_record_for_gap(
    gap_start: int,
    gap_end: int,
    gap_size: int,
    alloc: AllocationTrace,
) -> dict[str, object] | None:
    if alloc.size_bytes <= 0 or alloc.ptr == 0:
        return None
    overlap_start = max(gap_start, alloc.ptr)
    overlap_end = min(gap_end, alloc.end)
    if overlap_start > overlap_end:
        return None

    overlap_bytes = overlap_end - overlap_start + 1
    return {
        "alloc_id": alloc.alloc_id,
        "phase": alloc.phase,
        "predicted_class": alloc.predicted_class,
        "action": alloc.action,
        "action_success": alloc.action_success,
        "action_error": alloc.action_error,
        "size_bucket": alloc.size_bucket,
        "ptr_hex": alloc.ptr_hex,
        "end_hex": alloc.end_hex,
        "size_bytes": alloc.size_bytes,
        "overlap_start_hex": f"0x{overlap_start:x}",
        "overlap_end_hex": f"0x{overlap_end:x}",
        "overlap_bytes": overlap_bytes,
        "overlap_ratio_of_gap": overlap_bytes / gap_size if gap_size else 0.0,
        "alloc_wall_time": alloc.alloc_wall_time,
        "alloc_elapsed_s": alloc.alloc_elapsed_s,
        "freed": alloc.freed,
        "free_phase": alloc.free_phase,
        "free_wall_time": alloc.free_wall_time,
        "free_elapsed_s": alloc.free_elapsed_s,
        "lifetime_s": alloc.lifetime_s,
        "exact_match": alloc.ptr == gap_start and alloc.end == gap_end,
    }


def summarize_reuse_patterns(overlaps: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    for overlap in overlaps:
        key = (str(overlap["ptr_hex"]), str(overlap["end_hex"]))
        entry = grouped.setdefault(
            key,
            {
                "ptr_hex": overlap["ptr_hex"],
                "end_hex": overlap["end_hex"],
                "count": 0,
                "phases": Counter(),
                "predicted_classes": Counter(),
                "alloc_ids": [],
                "overlap_bytes_max": 0,
                "exact_match_count": 0,
                "alloc_elapsed_s": [],
                "lifetime_s": [],
            },
        )
        entry["count"] = int(entry["count"]) + 1
        entry["phases"][str(overlap["phase"])] += 1
        if overlap["predicted_class"] is not None:
            entry["predicted_classes"][str(overlap["predicted_class"])] += 1
        entry["alloc_ids"].append(int(overlap["alloc_id"]))
        entry["overlap_bytes_max"] = max(
            int(entry["overlap_bytes_max"]), int(overlap["overlap_bytes"])
        )
        if overlap["exact_match"]:
            entry["exact_match_count"] = int(entry["exact_match_count"]) + 1
        if overlap["alloc_elapsed_s"] is not None:
            entry["alloc_elapsed_s"].append(float(overlap["alloc_elapsed_s"]))
        if overlap["lifetime_s"] is not None and float(overlap["lifetime_s"]) >= 0:
            entry["lifetime_s"].append(float(overlap["lifetime_s"]))

    summaries: list[dict[str, object]] = []
    for entry in grouped.values():
        alloc_elapsed_values = sorted(entry["alloc_elapsed_s"])
        lifetime_values = sorted(entry["lifetime_s"])
        summaries.append(
            {
                "ptr_hex": entry["ptr_hex"],
                "end_hex": entry["end_hex"],
                "reuse_count": entry["count"],
                "top_phase": entry["phases"].most_common(1)[0][0]
                if entry["phases"]
                else None,
                "top_predicted_class": entry["predicted_classes"].most_common(1)[0][0]
                if entry["predicted_classes"]
                else None,
                "phases": dict(entry["phases"]),
                "predicted_classes": dict(entry["predicted_classes"]),
                "alloc_id_sample": entry["alloc_ids"][:10],
                "overlap_bytes_max": entry["overlap_bytes_max"],
                "exact_match_count": entry["exact_match_count"],
                "first_alloc_elapsed_s": alloc_elapsed_values[0]
                if alloc_elapsed_values
                else None,
                "last_alloc_elapsed_s": alloc_elapsed_values[-1]
                if alloc_elapsed_values
                else None,
                "median_lifetime_s": statistics.median(lifetime_values)
                if lifetime_values
                else None,
            }
        )

    summaries.sort(
        key=lambda item: (
            -int(item["reuse_count"]),
            -(item["median_lifetime_s"] or 0.0),
            str(item["ptr_hex"]),
        )
    )
    return summaries


def build_focused_gap_report(
    gap_index: int,
    gap_summaries: list[dict[str, object]],
    allocator_report: dict[str, object],
) -> dict[str, object]:
    gap_summary = find_gap_summary(gap_summaries, gap_index)
    if gap_summary is None:
        raise ValueError(f"gap#{gap_index} was not found in unknown gap summaries")

    gap_start = int(gap_summary["start_hex"], 16)
    gap_end = int(gap_summary["end_hex"], 16)
    gap_size = int(gap_summary["size_bytes"])
    allocations: list[AllocationTrace] = allocator_report["allocations"]  # type: ignore[assignment]

    overlaps = [
        overlap
        for alloc in allocations
        if (overlap := overlap_record_for_gap(gap_start, gap_end, gap_size, alloc)) is not None
    ]
    overlaps.sort(
        key=lambda item: (
            item["alloc_elapsed_s"] if item["alloc_elapsed_s"] is not None else float("inf"),
            int(item["alloc_id"]),
        )
    )
    phase_counter = Counter(str(item["phase"]) for item in overlaps)
    class_counter = Counter(
        str(item["predicted_class"])
        for item in overlaps
        if item["predicted_class"] is not None
    )
    class_overlap_bytes = Counter()
    for item in overlaps:
        if item["predicted_class"] is not None:
            class_overlap_bytes[str(item["predicted_class"])] += int(item["overlap_bytes"])

    return {
        "gap_summary": gap_summary,
        "overlap_allocation_count": len(overlaps),
        "dominant_phase": phase_counter.most_common(1)[0][0] if phase_counter else None,
        "dominant_predicted_class": (
            class_counter.most_common(1)[0][0] if class_counter else None
        ),
        "dominant_predicted_class_by_bytes": (
            class_overlap_bytes.most_common(1)[0][0] if class_overlap_bytes else None
        ),
        "phase_counts": dict(phase_counter),
        "predicted_class_counts": dict(class_counter),
        "predicted_class_overlap_bytes": dict(class_overlap_bytes),
        "ordered_overlaps": overlaps,
        "range_reuse_patterns": summarize_reuse_patterns(overlaps),
    }


def compact_focused_gap_report(
    focused_gap: dict[str, object], preview_limit: int
) -> dict[str, object]:
    compact = dict(focused_gap)
    compact["ordered_overlap_preview"] = focused_gap["ordered_overlaps"][:preview_limit]
    compact["range_reuse_preview"] = focused_gap["range_reuse_patterns"][:preview_limit]
    compact.pop("ordered_overlaps", None)
    compact.pop("range_reuse_patterns", None)
    return compact


def parse_fault_type(line: str) -> str:
    match = FAULT_TYPE_RE.search(line)
    return match.group("value") if match else "UNKNOWN"


def parse_access_type(line: str) -> str:
    match = ACCESS_TYPE_RE.search(line)
    return match.group("value") if match else "UNKNOWN"


def empty_class_stats() -> dict[str, object]:
    return {
        "faults": 0,
        "unique_pages": set(),
        "access_counts": Counter(),
        "fault_type_counts": Counter(),
    }


def gap_stats_template(gap: Gap) -> dict[str, object]:
    return {
        "gap": gap,
        "faults": 0,
        "unique_pages": set(),
        "access_counts": Counter(),
        "fault_type_counts": Counter(),
        "top_pages": Counter(),
    }


def classification_summary(stats: dict[str, object]) -> dict[str, object]:
    faults = int(stats["faults"])
    unique_pages = len(stats["unique_pages"])
    access_counts = dict(stats["access_counts"])
    fault_type_counts = dict(stats["fault_type_counts"])
    return {
        "faults": faults,
        "unique_pages": unique_pages,
        "avg_faults_per_unique_page": (faults / unique_pages) if unique_pages else 0.0,
        "access_counts": access_counts,
        "access_ratios": {
            access: (count / faults) if faults else 0.0
            for access, count in access_counts.items()
        },
        "fault_type_counts": fault_type_counts,
    }


def summarize_gap(gap_stat: dict[str, object], total_unknown_faults: int) -> dict[str, object]:
    gap: Gap = gap_stat["gap"]  # type: ignore[assignment]
    faults = int(gap_stat["faults"])
    unique_pages = len(gap_stat["unique_pages"])
    access_counts = dict(gap_stat["access_counts"])
    return {
        "gap_index": gap.index,
        "start_hex": gap.start_hex,
        "end_hex": gap.end_hex,
        "size_bytes": gap.size_bytes,
        "size_mib": gap.size_mib,
        "faults": faults,
        "fault_share_of_unknown": (faults / total_unknown_faults) if total_unknown_faults else 0.0,
        "unique_pages": unique_pages,
        "avg_faults_per_unique_page": (faults / unique_pages) if unique_pages else 0.0,
        "access_counts": access_counts,
        "access_ratios": {
            access: (count / faults) if faults else 0.0
            for access, count in access_counts.items()
        },
        "fault_type_counts": dict(gap_stat["fault_type_counts"]),
        "top_pages": [
            {"address_hex": f"0x{address:x}", "faults": count}
            for address, count in gap_stat["top_pages"].most_common(10)
        ],
        "left_region": (
            {
                "kind": gap.left_region.kind,
                "name": gap.left_region.name,
                "start_hex": gap.left_region.start_hex,
                "end_hex": gap.left_region.end_hex,
            }
            if gap.left_region is not None
            else None
        ),
        "right_region": (
            {
                "kind": gap.right_region.kind,
                "name": gap.right_region.name,
                "start_hex": gap.right_region.start_hex,
                "end_hex": gap.right_region.end_hex,
            }
            if gap.right_region is not None
            else None
        ),
    }


def warmup_workspace_assessment(
    class_summaries: dict[str, dict[str, object]],
    top_gaps: list[dict[str, object]],
    first_occurrence_line: dict[str, int | None],
    allocator_correlation: dict[str, object] | None = None,
) -> dict[str, object]:
    unknown_summary = class_summaries.get("unknown", {})
    weight_summary = class_summaries.get("weight", {})
    unknown_faults = int(unknown_summary.get("faults", 0))
    unknown_write_ratio = float(
        unknown_summary.get("access_ratios", {}).get("UVM_FAULT_ACCESS_TYPE_WRITE", 0.0)
    )
    unknown_avg_faults = float(unknown_summary.get("avg_faults_per_unique_page", 0.0))
    weight_avg_faults = float(weight_summary.get("avg_faults_per_unique_page", 0.0))
    top_gap_share = float(top_gaps[0]["fault_share_of_unknown"]) if top_gaps else 0.0

    evidence: list[str] = []
    score = 0

    if unknown_faults > 0 and unknown_write_ratio >= 0.95:
        score += 3
        evidence.append(
            f"unknown faults are write-dominated ({unknown_write_ratio:.2%}), unlike weight faults which are mostly read faults"
        )
    if top_gap_share >= 0.80:
        score += 3
        evidence.append(
            f"the hottest unknown gap alone accounts for {top_gap_share:.2%} of unknown faults"
        )
    if unknown_avg_faults >= 50:
        score += 2
        evidence.append(
            f"unknown pages are re-faulted heavily ({unknown_avg_faults:.2f} faults per unique page)"
        )
    if weight_avg_faults and unknown_avg_faults >= weight_avg_faults * 20:
        score += 1
        evidence.append(
            f"unknown page hotness is much higher than weight page hotness ({unknown_avg_faults:.2f} vs {weight_avg_faults:.2f})"
        )
    if (
        first_occurrence_line.get("unknown") is not None
        and first_occurrence_line.get("weight") is not None
        and first_occurrence_line["unknown"] < first_occurrence_line["weight"]
    ):
        score += 1
        evidence.append("unknown faults appear before the first recorded weight hit")
    if (
        first_occurrence_line.get("kv_cache") is not None
        and first_occurrence_line.get("unknown") is not None
        and first_occurrence_line["unknown"] < first_occurrence_line["kv_cache"]
    ):
        score += 1
        evidence.append("unknown faults appear well before the first recorded KV-cache hit")

    if allocator_correlation is not None:
        per_gap = allocator_correlation.get("per_gap", [])
        if per_gap:
            any_warmup_overlap = any(
                int(gap.get("warmup_like_overlap_bytes", 0)) > 0 for gap in per_gap
            )
            any_exact_match = any(
                int(gap.get("exact_match_count", 0)) > 0 for gap in per_gap
            )
            any_freed_overlap = any(
                int(gap.get("freed_overlap_bytes", 0)) > 0
                and int(gap.get("warmup_like_overlap_bytes", 0)) > 0
                for gap in per_gap
            )
            if any_warmup_overlap:
                score += 2
                evidence.append(
                    "allocator trace shows overlap between unknown gaps and warmup/autotune/cublas-preinit phases"
                )
            if any_exact_match:
                score += 1
                evidence.append(
                    "allocator trace contains an exact allocation-range match for at least one hot unknown gap"
                )
            if any_freed_overlap:
                score += 1
                evidence.append(
                    "overlapping traced allocations were later freed, fitting temporary workspace behavior"
                )

    if score >= 8:
        conclusion = "strongly_suggests_unlogged_warmup_or_workspace_buffers"
        confidence = "high"
    elif score >= 5:
        conclusion = "likely_unlogged_warmup_or_workspace_buffers"
        confidence = "medium"
    else:
        conclusion = "not_enough_signal_for_warmup_workspace_claim"
        confidence = "low"

    explanation = (
        "The unknown faults are concentrated in a few small gaps, are overwhelmingly write faults, "
        "and are repeatedly faulted on the same pages. That pattern fits temporary workspaces, "
        "scratch buffers, or warmup/autotune allocations better than model weights or KV cache."
    )

    return {
        "conclusion": conclusion,
        "confidence": confidence,
        "score": score,
        "explanation": explanation,
        "evidence": evidence,
    }


def analyze_faults(
    fault_log_path: Path,
    concrete_lookup: RegionLookup,
    gaps: list[Gap],
    use_raw_address: bool,
    max_faults: int | None,
) -> dict[str, object]:
    class_stats = {
        "weight": empty_class_stats(),
        "kv_cache": empty_class_stats(),
        "unknown": empty_class_stats(),
    }
    first_occurrence_line: dict[str, int | None] = {
        "weight": None,
        "kv_cache": None,
        "unknown": None,
    }
    gap_starts = [gap.start for gap in gaps]
    gap_stats = {gap.index: gap_stats_template(gap) for gap in gaps}
    unknown_outside = Counter()

    parsed_faults = 0
    total_lines = 0
    ignored_lines = 0

    overall_min = gaps[0].start if gaps else None
    overall_max = gaps[-1].end if gaps else None

    with fault_log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for total_lines, line in enumerate(handle, start=1):
            record = parse_fault_record(line, total_lines)
            if record is None:
                ignored_lines += 1
                continue

            parsed_faults += 1
            selected_address = record.selected_address(use_raw_address)
            if selected_address is None:
                continue

            access_type = parse_access_type(line)
            fault_type = parse_fault_type(line)

            matched_region = concrete_lookup.find(selected_address)
            classification = matched_region.major_type if matched_region is not None else "unknown"

            stats = class_stats[classification]
            stats["faults"] = int(stats["faults"]) + 1
            stats["unique_pages"].add(selected_address)
            stats["access_counts"][access_type] += 1
            stats["fault_type_counts"][fault_type] += 1

            if first_occurrence_line[classification] is None:
                first_occurrence_line[classification] = total_lines

            if classification == "unknown":
                gap_index = bisect.bisect_right(gap_starts, selected_address) - 1
                if gap_index >= 0:
                    gap = gaps[gap_index]
                    if gap.start <= selected_address <= gap.end:
                        current_gap_stats = gap_stats[gap.index]
                        current_gap_stats["faults"] = int(current_gap_stats["faults"]) + 1
                        current_gap_stats["unique_pages"].add(selected_address)
                        current_gap_stats["access_counts"][access_type] += 1
                        current_gap_stats["fault_type_counts"][fault_type] += 1
                        current_gap_stats["top_pages"][selected_address] += 1
                    elif overall_min is not None and selected_address < overall_min:
                        unknown_outside["below_first_gap"] += 1
                    elif overall_max is not None and selected_address > overall_max:
                        unknown_outside["above_last_gap"] += 1
                    else:
                        unknown_outside["inside_region_span_but_not_in_gap"] += 1
                elif overall_min is not None and selected_address < overall_min:
                    unknown_outside["below_first_gap"] += 1
                elif overall_max is not None and selected_address > overall_max:
                    unknown_outside["above_last_gap"] += 1
                else:
                    unknown_outside["inside_region_span_but_not_in_gap"] += 1

            if max_faults is not None and parsed_faults >= max_faults:
                break

    class_summaries = {
        name: classification_summary(stats) for name, stats in class_stats.items()
    }
    unknown_faults = int(class_summaries["unknown"]["faults"])
    gap_summaries = [
        summarize_gap(gap_stat, unknown_faults)
        for gap_stat in gap_stats.values()
        if int(gap_stat["faults"]) > 0
    ]
    gap_summaries.sort(key=lambda item: (-item["faults"], item["start_hex"]))

    return {
        "parsed_fault_records": parsed_faults,
        "ignored_lines": ignored_lines,
        "total_lines_seen": total_lines,
        "class_summaries": class_summaries,
        "first_occurrence_line": first_occurrence_line,
        "unknown_gap_summaries": gap_summaries,
        "unknown_outside_counts": dict(unknown_outside),
    }


def write_gap_csv(path: Path, gap_summaries: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "gap_index",
                "start_hex",
                "end_hex",
                "size_bytes",
                "size_mib",
                "faults",
                "fault_share_of_unknown",
                "unique_pages",
                "avg_faults_per_unique_page",
                "write_ratio",
                "read_ratio",
                "left_region_kind",
                "left_region_name",
                "right_region_kind",
                "right_region_name",
            ]
        )
        for gap in gap_summaries:
            access_ratios = gap["access_ratios"]
            left_region = gap["left_region"] or {}
            right_region = gap["right_region"] or {}
            writer.writerow(
                [
                    gap["gap_index"],
                    gap["start_hex"],
                    gap["end_hex"],
                    gap["size_bytes"],
                    f"{gap['size_mib']:.3f}",
                    gap["faults"],
                    f"{gap['fault_share_of_unknown']:.6f}",
                    gap["unique_pages"],
                    f"{gap['avg_faults_per_unique_page']:.6f}",
                    f"{access_ratios.get('UVM_FAULT_ACCESS_TYPE_WRITE', 0.0):.6f}",
                    f"{access_ratios.get('UVM_FAULT_ACCESS_TYPE_READ', 0.0):.6f}",
                    left_region.get("kind", ""),
                    left_region.get("name", ""),
                    right_region.get("kind", ""),
                    right_region.get("name", ""),
                ]
            )


def print_report(report: dict[str, object], top_gaps: int) -> None:
    print("Deep Dive Summary")
    print(
        f"- parsed_fault_records={report['fault_analysis']['parsed_fault_records']} "
        f"ignored_lines={report['fault_analysis']['ignored_lines']} "
        f"total_lines_seen={report['fault_analysis']['total_lines_seen']}"
    )

    print("\nUnique Pages By Classification")
    for name, summary in report["fault_analysis"]["class_summaries"].items():
        print(
            f"- {name}: faults={summary['faults']} unique_pages={summary['unique_pages']} "
            f"avg_faults_per_unique_page={summary['avg_faults_per_unique_page']:.6f}"
        )

    print("\nREAD/WRITE Ratios")
    for name, summary in report["fault_analysis"]["class_summaries"].items():
        ratios = summary["access_ratios"]
        print(
            f"- {name}: "
            f"read={ratios.get('UVM_FAULT_ACCESS_TYPE_READ', 0.0):.2%} "
            f"write={ratios.get('UVM_FAULT_ACCESS_TYPE_WRITE', 0.0):.2%}"
        )

    print("\nTop Unknown Gaps")
    gap_summaries = report["fault_analysis"]["unknown_gap_summaries"]
    for gap in gap_summaries[:top_gaps]:
        print(
            f"- gap#{gap['gap_index']} {gap['start_hex']}..{gap['end_hex']} "
            f"size={gap['size_mib']:.3f} MiB faults={gap['faults']} "
            f"share_of_unknown={gap['fault_share_of_unknown']:.2%} "
            f"unique_pages={gap['unique_pages']} "
            f"avg_faults_per_unique_page={gap['avg_faults_per_unique_page']:.2f} "
            f"read={gap['access_ratios'].get('UVM_FAULT_ACCESS_TYPE_READ', 0.0):.2%} "
            f"write={gap['access_ratios'].get('UVM_FAULT_ACCESS_TYPE_WRITE', 0.0):.2%}"
        )
        left_region = gap["left_region"]
        right_region = gap["right_region"]
        if left_region or right_region:
            print(
                f"  neighbors: left={left_region['name'] if left_region else 'None'} "
                f"right={right_region['name'] if right_region else 'None'}"
            )

    outside = report["fault_analysis"]["unknown_outside_counts"]
    if outside:
        print("\nUnknown Outside Gap Accounting")
        for name, count in outside.items():
            print(f"- {name}: {count}")

    allocator_correlation = report.get("allocator_correlation")
    if allocator_correlation is not None:
        print("\nAllocator Correlation")
        print(
            f"- allocator_log={allocator_correlation['allocator_log']} "
            f"allocations={allocator_correlation['allocation_count']} "
            f"matched_trace_lines={allocator_correlation['matched_trace_lines']}"
        )
        for gap in allocator_correlation["per_gap"]:
            print(
                f"- gap#{gap['gap_index']} dominant_phase={gap['dominant_phase']} "
                f"dominant_predicted_class={gap['dominant_predicted_class']} "
                f"likely_kind={gap['likely_kind']} "
                f"unique_coverage_of_gap={gap['unique_coverage_ratio_of_gap']:.2%} "
                f"cumulative_overlap_of_gap={gap['cumulative_overlap_ratio_of_gap']:.2%} "
                f"warmup_like_overlap_bytes={gap['warmup_like_overlap_bytes']} "
                f"freed_overlap_bytes={gap['freed_overlap_bytes']} "
                f"exact_match_count={gap['exact_match_count']} "
                f"assessment={gap['assessment']}"
            )
            lifetime_stats = gap["lifetime_stats"]
            if lifetime_stats["freed_count"] > 0:
                print(
                    "  lifetime: "
                    f"freed_count={lifetime_stats['freed_count']} "
                    f"min={lifetime_stats['min_s']:.6f}s "
                    f"median={lifetime_stats['median_s']:.6f}s "
                    f"p95={lifetime_stats['p95_s']:.6f}s "
                    f"max={lifetime_stats['max_s']:.6f}s"
                )
            for alloc in gap["top_overlapping_allocations"][:5]:
                print(
                    "  overlap: "
                    f"alloc_id={alloc['alloc_id']} "
                    f"{alloc['ptr_hex']}..{alloc['end_hex']} "
                    f"phase={alloc['phase']} "
                    f"predicted_class={alloc['predicted_class']} "
                    f"freed={alloc['freed']} "
                    f"lifetime_s={alloc['lifetime_s']} "
                    f"overlap_ratio_of_gap={alloc['overlap_ratio_of_gap']:.2%}"
                )

    focused_gap = report.get("focused_gap")
    if focused_gap is not None:
        gap_summary = focused_gap["gap_summary"]
        print("\nFocused Gap Timeline")
        print(
            f"- gap#{gap_summary['gap_index']} {gap_summary['start_hex']}..{gap_summary['end_hex']} "
            f"faults={gap_summary['faults']} unique_pages={gap_summary['unique_pages']} "
            f"dominant_phase={focused_gap['dominant_phase']} "
            f"dominant_predicted_class={focused_gap['dominant_predicted_class']} "
            f"dominant_predicted_class_by_bytes={focused_gap['dominant_predicted_class_by_bytes']}"
        )
        for overlap in focused_gap["ordered_overlap_preview"][:10]:
            print(
                "  ordered_overlap: "
                f"alloc_id={overlap['alloc_id']} "
                f"elapsed={overlap['alloc_elapsed_s']}s "
                f"phase={overlap['phase']} "
                f"predicted_class={overlap['predicted_class']} "
                f"{overlap['ptr_hex']}..{overlap['end_hex']} "
                f"overlap={overlap['overlap_ratio_of_gap']:.2%} "
                f"lifetime_s={overlap['lifetime_s']}"
            )
        for reuse in focused_gap["range_reuse_preview"][:10]:
            print(
                "  reuse_pattern: "
                f"{reuse['ptr_hex']}..{reuse['end_hex']} "
                f"reuse_count={reuse['reuse_count']} "
                f"top_phase={reuse['top_phase']} "
                f"top_predicted_class={reuse['top_predicted_class']} "
                f"exact_match_count={reuse['exact_match_count']}"
            )

    print("\nWarmup/Workspace Heuristic")
    assessment = report["warmup_workspace_assessment"]
    print(f"- conclusion: {assessment['conclusion']}")
    print(f"- confidence: {assessment['confidence']}")
    print(f"- score: {assessment['score']}")
    for item in assessment["evidence"]:
        print(f"- evidence: {item}")


def main() -> int:
    args = parse_args()
    address_log_path = Path(args.address_log)
    fault_log_path = Path(args.fault_log)

    if not address_log_path.is_file():
        print(f"address log not found: {address_log_path}", file=sys.stderr)
        return 2
    if not fault_log_path.is_file():
        print(f"fault log not found: {fault_log_path}", file=sys.stderr)
        return 2

    sections, warnings = parse_address_log(address_log_path)
    selected_pid, selected_sections = select_sections(sections, args.pid)
    concrete_regions = build_concrete_regions(selected_sections)
    concrete_lookup = RegionLookup(concrete_regions)
    gaps = build_gaps(concrete_regions)

    fault_analysis = analyze_faults(
        fault_log_path=fault_log_path,
        concrete_lookup=concrete_lookup,
        gaps=gaps,
        use_raw_address=args.use_raw_address,
        max_faults=args.max_faults,
    )
    top_gap_summaries = fault_analysis["unknown_gap_summaries"][: args.top_gaps]
    allocator_correlation = None
    allocator_report = None
    if args.allocator_log:
        allocator_log_path = Path(args.allocator_log)
        if not allocator_log_path.is_file():
            print(f"allocator log not found: {allocator_log_path}", file=sys.stderr)
            return 2
        allocator_report = parse_allocator_log(allocator_log_path)
        allocator_correlation = correlate_allocator_allocations(
            gap_summaries=fault_analysis["unknown_gap_summaries"],
            allocator_report=allocator_report,
            top_gaps=args.top_gaps,
        )
    heuristic = warmup_workspace_assessment(
        class_summaries=fault_analysis["class_summaries"],
        top_gaps=top_gap_summaries,
        first_occurrence_line=fault_analysis["first_occurrence_line"],
        allocator_correlation=allocator_correlation,
    )
    focused_gap = None
    focused_gap_full = None
    if args.focus_gap is not None:
        if allocator_report is None:
            print("--focus-gap requires --allocator-log", file=sys.stderr)
            return 2
        try:
            focused_gap_full = build_focused_gap_report(
                gap_index=args.focus_gap,
                gap_summaries=fault_analysis["unknown_gap_summaries"],
                allocator_report=allocator_report,
            )
            focused_gap = compact_focused_gap_report(
                focused_gap_full,
                preview_limit=args.gap_preview_limit,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    report = {
        "selected_pid": selected_pid,
        "address_log": str(address_log_path),
        "fault_log": str(fault_log_path),
        "used_raw_addresses": bool(args.use_raw_address),
        "warnings": warnings,
        "gap_count": len(gaps),
        "fault_analysis": fault_analysis,
        "allocator_correlation": allocator_correlation,
        "focused_gap": focused_gap,
        "warmup_workspace_assessment": heuristic,
    }

    if args.gap_csv:
        write_gap_csv(Path(args.gap_csv), fault_analysis["unknown_gap_summaries"])
    if args.gap_alloc_csv and focused_gap_full is not None:
        with Path(args.gap_alloc_csv).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "alloc_id",
                    "phase",
                    "predicted_class",
                    "action",
                    "action_success",
                    "size_bucket",
                    "ptr_hex",
                    "end_hex",
                    "size_bytes",
                    "overlap_start_hex",
                    "overlap_end_hex",
                    "overlap_bytes",
                    "overlap_ratio_of_gap",
                    "alloc_wall_time",
                    "alloc_elapsed_s",
                    "freed",
                    "free_phase",
                    "free_wall_time",
                    "free_elapsed_s",
                    "lifetime_s",
                    "exact_match",
                ]
            )
            for overlap in focused_gap_full["ordered_overlaps"]:
                writer.writerow(
                    [
                        overlap["alloc_id"],
                        overlap["phase"],
                        overlap["predicted_class"],
                        overlap["action"],
                        overlap["action_success"],
                        overlap["size_bucket"],
                        overlap["ptr_hex"],
                        overlap["end_hex"],
                        overlap["size_bytes"],
                        overlap["overlap_start_hex"],
                        overlap["overlap_end_hex"],
                        overlap["overlap_bytes"],
                        f"{overlap['overlap_ratio_of_gap']:.6f}",
                        overlap["alloc_wall_time"],
                        overlap["alloc_elapsed_s"],
                        overlap["freed"],
                        overlap["free_phase"],
                        overlap["free_wall_time"],
                        overlap["free_elapsed_s"],
                        overlap["lifetime_s"],
                        overlap["exact_match"],
                    ]
                )
    if args.gap_reuse_csv and focused_gap_full is not None:
        with Path(args.gap_reuse_csv).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "ptr_hex",
                    "end_hex",
                    "reuse_count",
                    "top_phase",
                    "top_predicted_class",
                    "overlap_bytes_max",
                    "exact_match_count",
                    "first_alloc_elapsed_s",
                    "last_alloc_elapsed_s",
                    "median_lifetime_s",
                    "alloc_id_sample",
                ]
            )
            for reuse in focused_gap_full["range_reuse_patterns"]:
                writer.writerow(
                    [
                        reuse["ptr_hex"],
                        reuse["end_hex"],
                        reuse["reuse_count"],
                        reuse["top_phase"],
                        reuse["top_predicted_class"],
                        reuse["overlap_bytes_max"],
                        reuse["exact_match_count"],
                        reuse["first_alloc_elapsed_s"],
                        reuse["last_alloc_elapsed_s"],
                        reuse["median_lifetime_s"],
                        "|".join(str(value) for value in reuse["alloc_id_sample"]),
                    ]
                )
    if args.gap_report_json and focused_gap is not None:
        with Path(args.gap_report_json).open("w", encoding="utf-8") as handle:
            json.dump(focused_gap, handle, indent=2, sort_keys=True)
    if args.summary_json:
        with Path(args.summary_json).open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)

    print_report(report, args.top_gaps)
    if warnings:
        print("\nWarnings")
        for warning in warnings[:20]:
            print(f"- {warning}")
        if len(warnings) > 20:
            print(f"- ... {len(warnings) - 20} more warnings omitted")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

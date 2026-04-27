#!/usr/bin/env python3
"""Summarize gap-watch policy effectiveness from allocator trace logs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


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
TRACE_GAP_ALLOC_RE = re.compile(
    r"TRACE_GAP_WATCH_ALLOC alloc_id=(?P<alloc_id>\d+) watch_name=(?P<watch_name>[^ ]+) "
    r"ptr=(?P<ptr>0x[0-9a-fA-F]+) end=(?P<end>0x[0-9a-fA-F]+) size_bytes=(?P<size_bytes>\d+) "
    r"size_bucket=(?P<size_bucket>[^ ]+) device=(?P<device>-?\d+) phase=(?P<phase>\S+) "
    r"predicted_class=(?P<predicted_class>[^ ]+) action=(?P<action>[^ ]+) "
    r"policy_source=(?P<policy_source>[^ ]+) gap_watch_target_class=(?P<target_class>[^ ]+) "
    r"gap_watch_class_match=(?P<class_match>\d+) .* overlap_bytes=(?P<overlap_bytes>\d+)"
)
TRACE_GAP_FREE_RE = re.compile(
    r"TRACE_GAP_WATCH_FREE free_id=(?P<free_id>\d+) watch_name=(?P<watch_name>[^ ]+) "
    r"ptr=(?P<ptr>0x[0-9a-fA-F]+) end=(?P<end>0x[0-9a-fA-F]+) size_bytes=(?P<size_bytes>\d+) "
    r"device=(?P<device>-?\d+) phase=(?P<phase>\S+) alloc_id=(?P<alloc_id>\d+) "
    r"alloc_phase=(?P<alloc_phase>\S+) alloc_predicted_class=(?P<predicted_class>[^ ]+) "
    r"alloc_action=(?P<action>[^ ]+) alloc_policy_source=(?P<policy_source>[^ ]+) "
    r"alloc_policy_success=(?P<policy_success>\d+) alloc_policy_error=(?P<policy_error>[^ ]+) "
    r"gap_watch_target_class=(?P<target_class>[^ ]+) gap_watch_policy_action=(?P<policy_action>[^ ]+) "
    r".* overlap_bytes=(?P<overlap_bytes>\d+) .* lifetime_s=(?P<lifetime_s>-?[0-9.]+)"
)
SUMMARY_KEY_RE = re.compile(r"^\s{2}(?P<key>[^:]+): (?P<value>.+)$")
KV_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>[^ ]+)")


def parse_kv_fields(line: str) -> dict[str, str]:
    return {
        match.group("key"): match.group("value").strip()
        for match in KV_RE.finditer(line)
    }


def counter_ratios(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in counter.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize whether gap-watch policy hit and succeeded."
    )
    parser.add_argument(
        "--allocator-log",
        required=True,
        help="Path to the allocator trace log.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    allocator_log = Path(args.allocator_log)
    if not allocator_log.is_file():
        raise SystemExit(f"allocator log not found: {allocator_log}")

    policy_records = 0
    gap_overlap_records = 0
    gap_policy_records = 0
    gap_policy_success = 0
    gap_policy_fail = 0
    gap_overlap_bytes = 0
    gap_policy_overlap_bytes = 0
    class_match_records = 0
    action_counter: Counter[str] = Counter()
    class_counter: Counter[str] = Counter()
    phase_counter: Counter[str] = Counter()
    target_class_counter: Counter[str] = Counter()
    policy_source_counter: Counter[str] = Counter()
    placement_backend_counter: Counter[str] = Counter()
    device_direct_reason_counter: Counter[str] = Counter()
    lifetime_values: list[float] = []
    session_summary: dict[str, str] = {}
    in_summary = False
    device_direct_trace_records = 0
    device_direct_eligible_records = 0
    hot_gap_match_records = 0

    with allocator_log.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "Session Summary" in line:
                in_summary = True
                continue
            if in_summary:
                if line.startswith("========================================"):
                    in_summary = False
                    continue
                summary_match = SUMMARY_KEY_RE.search(line)
                if summary_match is not None:
                    session_summary[summary_match.group("key")] = summary_match.group("value")

            policy_match = TRACE_POLICY_RE.search(line)
            if policy_match is not None:
                policy_records += 1
                continue

            gap_alloc_match = TRACE_GAP_ALLOC_RE.search(line)
            if gap_alloc_match is not None:
                kv_fields = parse_kv_fields(line)
                gap_overlap_records += 1
                overlap_bytes = int(gap_alloc_match.group("overlap_bytes"))
                gap_overlap_bytes += overlap_bytes
                predicted_class = gap_alloc_match.group("predicted_class")
                phase = gap_alloc_match.group("phase")
                action = gap_alloc_match.group("action")
                policy_source = gap_alloc_match.group("policy_source")
                target_class = gap_alloc_match.group("target_class")
                class_match = gap_alloc_match.group("class_match") == "1"
                class_counter[predicted_class] += 1
                phase_counter[phase] += 1
                action_counter[action] += 1
                policy_source_counter[policy_source] += 1
                target_class_counter[target_class] += 1
                placement_backend_counter[kv_fields.get("placement_backend", "unknown")] += 1
                device_direct_reason_counter[
                    kv_fields.get("device_direct_reason", "not_recorded")
                ] += 1
                if action in {"device_direct_trace", "device_direct"}:
                    device_direct_trace_records += 1
                if kv_fields.get("device_direct_eligible") == "1":
                    device_direct_eligible_records += 1
                if kv_fields.get("hot_gap_match") == "1":
                    hot_gap_match_records += 1
                if class_match:
                    class_match_records += 1
                if policy_source == "gap_watch_policy" and action != "managed_default":
                    gap_policy_records += 1
                    gap_policy_overlap_bytes += overlap_bytes
                continue

            gap_free_match = TRACE_GAP_FREE_RE.search(line)
            if gap_free_match is not None:
                if gap_free_match.group("policy_source") == "gap_watch_policy" and gap_free_match.group("action") != "managed_default":
                    if gap_free_match.group("policy_success") == "1":
                        gap_policy_success += 1
                    else:
                        gap_policy_fail += 1
                lifetime_s = float(gap_free_match.group("lifetime_s"))
                if lifetime_s >= 0:
                    lifetime_values.append(lifetime_s)

    summary = {
        "allocator_log": str(allocator_log),
        "policy_records": policy_records,
        "gap_overlap_records": gap_overlap_records,
        "gap_overlap_bytes": gap_overlap_bytes,
        "gap_policy_records": gap_policy_records,
        "gap_policy_overlap_bytes": gap_policy_overlap_bytes,
        "gap_policy_success": gap_policy_success,
        "gap_policy_fail": gap_policy_fail,
        "gap_watch_class_match_records": class_match_records,
        "dominant_predicted_class": class_counter.most_common(1)[0][0] if class_counter else None,
        "dominant_phase": phase_counter.most_common(1)[0][0] if phase_counter else None,
        "dominant_action": action_counter.most_common(1)[0][0] if action_counter else None,
        "dominant_target_class": target_class_counter.most_common(1)[0][0] if target_class_counter else None,
        "policy_source_counts": dict(policy_source_counter),
        "predicted_class_counts": dict(class_counter),
        "phase_counts": dict(phase_counter),
        "phase_record_ratios": counter_ratios(phase_counter),
        "action_counts": dict(action_counter),
        "target_class_counts": dict(target_class_counter),
        "placement_backend_counts": dict(placement_backend_counter),
        "device_direct_reason_counts": dict(device_direct_reason_counter),
        "device_direct_trace_records": device_direct_trace_records,
        "device_direct_eligible_records": device_direct_eligible_records,
        "device_direct_actual_records": placement_backend_counter.get(
            "device_direct", 0
        ),
        "hot_gap_match_records": hot_gap_match_records,
        "median_lifetime_s": (
            sorted(lifetime_values)[len(lifetime_values) // 2]
            if lifetime_values
            else None
        ),
        "session_summary": session_summary,
    }

    if args.summary_json:
        with Path(args.summary_json).open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)

    print("Gap Watch Metrics Summary")
    print(f"- gap_overlap_records={gap_overlap_records}")
    print(f"- gap_policy_records={gap_policy_records}")
    print(f"- gap_policy_success={gap_policy_success}")
    print(f"- gap_policy_fail={gap_policy_fail}")
    print(f"- gap_overlap_bytes={gap_overlap_bytes}")
    print(f"- gap_policy_overlap_bytes={gap_policy_overlap_bytes}")
    print(f"- dominant_predicted_class={summary['dominant_predicted_class']}")
    print(f"- dominant_phase={summary['dominant_phase']}")
    print(f"- dominant_action={summary['dominant_action']}")
    print(f"- dominant_target_class={summary['dominant_target_class']}")
    print(f"- phase_record_ratios={summary['phase_record_ratios']}")
    print(f"- placement_backend_counts={summary['placement_backend_counts']}")
    print(f"- device_direct_trace_records={device_direct_trace_records}")
    print(f"- device_direct_eligible_records={device_direct_eligible_records}")
    print(f"- device_direct_actual_records={summary['device_direct_actual_records']}")
    print(f"- hot_gap_match_records={hot_gap_match_records}")
    print(f"- median_lifetime_s={summary['median_lifetime_s']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

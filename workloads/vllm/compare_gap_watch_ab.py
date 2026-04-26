#!/usr/bin/env python3
"""Compare observe vs prefetch gap-watch experiment outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two same-run gap-watch experiments, typically observe vs prefetch."
        )
    )
    parser.add_argument(
        "--observe-post-main",
        required=True,
        help="Post-main summary JSON from the observe run.",
    )
    parser.add_argument(
        "--observe-metrics",
        required=True,
        help="Gap-watch metrics JSON from the observe run.",
    )
    parser.add_argument(
        "--prefetch-post-main",
        required=True,
        help="Post-main summary JSON from the prefetch run.",
    )
    parser.add_argument(
        "--prefetch-metrics",
        required=True,
        help="Gap-watch metrics JSON from the prefetch run.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output JSON path.",
    )
    return parser.parse_args()


def load_json(path: str) -> dict:
    file_path = Path(path)
    if not file_path.is_file():
        raise SystemExit(f"file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pct_delta(new_value: float | int | None, old_value: float | int | None):
    if new_value is None or old_value is None:
        return None
    if old_value == 0:
        return None
    return (float(new_value) - float(old_value)) / float(old_value)


def gap_identity(summary: dict) -> tuple[int | None, str | None, str | None]:
    selected = summary.get("selected_gap", {})
    return (
        selected.get("gap_index"),
        selected.get("start_hex"),
        selected.get("end_hex"),
    )


def main() -> int:
    args = parse_args()

    observe_post = load_json(args.observe_post_main)
    observe_metrics = load_json(args.observe_metrics)
    prefetch_post = load_json(args.prefetch_post_main)
    prefetch_metrics = load_json(args.prefetch_metrics)

    observe_gap = observe_post.get("selected_gap", {})
    prefetch_gap = prefetch_post.get("selected_gap", {})

    same_gap = gap_identity(observe_post) == gap_identity(prefetch_post)

    comparison = {
        "observe": {
            "effective_policy_action": observe_post.get("effective_policy_action"),
            "effective_target_class": observe_post.get("effective_target_class"),
            "gap_index": observe_gap.get("gap_index"),
            "gap_start": observe_gap.get("start_hex"),
            "gap_end": observe_gap.get("end_hex"),
            "gap_faults": observe_gap.get("faults"),
            "total_unknown_faults": observe_post.get("total_unknown_faults"),
            "avg_faults_per_unique_page": observe_gap.get("avg_faults_per_unique_page"),
            "fault_share_of_unknown": observe_gap.get("fault_share_of_unknown"),
            "gap_policy_records": observe_metrics.get("gap_policy_records"),
            "gap_policy_success": observe_metrics.get("gap_policy_success"),
            "gap_policy_fail": observe_metrics.get("gap_policy_fail"),
            "dominant_action": observe_metrics.get("dominant_action"),
        },
        "prefetch": {
            "effective_policy_action": prefetch_post.get("effective_policy_action"),
            "effective_target_class": prefetch_post.get("effective_target_class"),
            "gap_index": prefetch_gap.get("gap_index"),
            "gap_start": prefetch_gap.get("start_hex"),
            "gap_end": prefetch_gap.get("end_hex"),
            "gap_faults": prefetch_gap.get("faults"),
            "total_unknown_faults": prefetch_post.get("total_unknown_faults"),
            "avg_faults_per_unique_page": prefetch_gap.get("avg_faults_per_unique_page"),
            "fault_share_of_unknown": prefetch_gap.get("fault_share_of_unknown"),
            "gap_policy_records": prefetch_metrics.get("gap_policy_records"),
            "gap_policy_success": prefetch_metrics.get("gap_policy_success"),
            "gap_policy_fail": prefetch_metrics.get("gap_policy_fail"),
            "dominant_action": prefetch_metrics.get("dominant_action"),
        },
        "comparison": {
            "same_gap": same_gap,
            "gap_fault_delta": (
                prefetch_gap.get("faults") - observe_gap.get("faults")
                if observe_gap.get("faults") is not None and prefetch_gap.get("faults") is not None
                else None
            ),
            "gap_fault_delta_pct": pct_delta(
                prefetch_gap.get("faults"), observe_gap.get("faults")
            ),
            "unknown_fault_delta": (
                prefetch_post.get("total_unknown_faults") - observe_post.get("total_unknown_faults")
                if observe_post.get("total_unknown_faults") is not None and prefetch_post.get("total_unknown_faults") is not None
                else None
            ),
            "unknown_fault_delta_pct": pct_delta(
                prefetch_post.get("total_unknown_faults"),
                observe_post.get("total_unknown_faults"),
            ),
            "avg_faults_per_unique_page_delta": (
                prefetch_gap.get("avg_faults_per_unique_page") -
                observe_gap.get("avg_faults_per_unique_page")
                if observe_gap.get("avg_faults_per_unique_page") is not None and prefetch_gap.get("avg_faults_per_unique_page") is not None
                else None
            ),
            "avg_faults_per_unique_page_delta_pct": pct_delta(
                prefetch_gap.get("avg_faults_per_unique_page"),
                observe_gap.get("avg_faults_per_unique_page"),
            ),
            "prefetch_policy_success_rate": (
                float(prefetch_metrics.get("gap_policy_success", 0)) /
                float(prefetch_metrics.get("gap_policy_records", 1))
                if prefetch_metrics.get("gap_policy_records", 0) > 0
                else None
            ),
        },
    }

    if args.output_json:
        with Path(args.output_json).open("w", encoding="utf-8") as handle:
            json.dump(comparison, handle, indent=2, sort_keys=True)

    print("Gap Watch A/B Comparison")
    print(f"- same_gap={comparison['comparison']['same_gap']}")
    print(
        f"- observe_action={comparison['observe']['effective_policy_action']} "
        f"prefetch_action={comparison['prefetch']['effective_policy_action']}"
    )
    print(
        f"- observe_gap_faults={comparison['observe']['gap_faults']} "
        f"prefetch_gap_faults={comparison['prefetch']['gap_faults']}"
    )
    print(
        f"- gap_fault_delta={comparison['comparison']['gap_fault_delta']} "
        f"gap_fault_delta_pct={comparison['comparison']['gap_fault_delta_pct']}"
    )
    print(
        f"- observe_unknown_faults={comparison['observe']['total_unknown_faults']} "
        f"prefetch_unknown_faults={comparison['prefetch']['total_unknown_faults']}"
    )
    print(
        f"- unknown_fault_delta={comparison['comparison']['unknown_fault_delta']} "
        f"unknown_fault_delta_pct={comparison['comparison']['unknown_fault_delta_pct']}"
    )
    print(
        f"- observe_avg_faults_per_page={comparison['observe']['avg_faults_per_unique_page']} "
        f"prefetch_avg_faults_per_page={comparison['prefetch']['avg_faults_per_unique_page']}"
    )
    print(
        f"- avg_faults_per_page_delta={comparison['comparison']['avg_faults_per_unique_page_delta']} "
        f"avg_faults_per_page_delta_pct={comparison['comparison']['avg_faults_per_unique_page_delta_pct']}"
    )
    print(
        f"- observe_policy_records={comparison['observe']['gap_policy_records']} "
        f"prefetch_policy_records={comparison['prefetch']['gap_policy_records']}"
    )
    print(
        f"- prefetch_policy_success={comparison['prefetch']['gap_policy_success']} "
        f"prefetch_policy_fail={comparison['prefetch']['gap_policy_fail']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

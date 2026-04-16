#!/usr/bin/env python3
"""
Plot Attention-Aware Memory Subsystem Results

Generates comparison plots for attention-aware vs baseline eviction policies.

Usage:
    uv run python plot_attention_results.py --results-dir results/exp_attention_aware/20260409_123456
    uv run python plot_attention_results.py --baseline baseline.json --attention attention.json
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent


def load_result(json_path: Path) -> Dict:
    """Load a benchmark result JSON file."""
    with open(json_path) as f:
        return json.load(f)


def find_results_in_dir(results_dir: Path) -> Dict[str, Path]:
    """Find all result JSON files in a directory."""
    results = {}
    for json_file in results_dir.glob("*.json"):
        # Skip if it's a nested directory result
        if json_file.stem in ["baseline_uvm_lru", "attention_aware_ebpf", "cpu_offload"]:
            results[json_file.stem] = json_file
    return results


def plot_latency_comparison(results: Dict[str, Dict], output_path: Path):
    """Generate latency comparison plots (TTFT and TPOT)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    configs = list(results.keys())
    colors = plt.cm.Set2(range(len(configs)))

    # TTFT comparison
    ax = axes[0]
    x_pos = np.arange(len(configs))
    width = 0.35

    ttft_p50 = [results[c]["metrics"].get("median_ttft_ms", 0) / 1000 for c in configs]
    ttft_p99 = [results[c]["metrics"].get("p99_ttft_ms", 0) / 1000 for c in configs]

    ax.bar(x_pos - width / 2, ttft_p50, width, label="P50", color=colors, alpha=0.8)
    ax.bar(x_pos + width / 2, ttft_p99, width, label="P99", color=colors, alpha=0.5)

    ax.set_xlabel("Configuration", fontsize=12)
    ax.set_ylabel("Time to First Token (s)", fontsize=12)
    ax.set_title("TTFT Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([c.replace("_", "\n") for c in configs], fontsize=10)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # TPOT comparison
    ax = axes[1]
    tpot_p50 = [results[c]["metrics"].get("median_tpot_ms", 0) for c in configs]
    tpot_p99 = [results[c]["metrics"].get("p99_tpot_ms", 0) for c in configs]

    ax.bar(x_pos - width / 2, tpot_p50, width, label="P50", color=colors, alpha=0.8)
    ax.bar(x_pos + width / 2, tpot_p99, width, label="P99", color=colors, alpha=0.5)

    ax.set_xlabel("Configuration", fontsize=12)
    ax.set_ylabel("Time per Output Token (ms)", fontsize=12)
    ax.set_title("TPOT Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([c.replace("_", "\n") for c in configs], fontsize=10)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved latency comparison to {output_path}")


def plot_throughput_comparison(results: Dict[str, Dict], output_path: Path):
    """Generate throughput comparison plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    configs = list(results.keys())
    colors = plt.cm.Set2(range(len(configs)))

    throughput = [results[c]["metrics"].get("output_throughput_tok_s", 0) for c in configs]
    peak_throughput = [results[c]["metrics"].get("peak_throughput_tok_s", 0) for c in configs]

    x_pos = np.arange(len(configs))
    width = 0.35

    ax.bar(x_pos - width / 2, throughput, width, label="Average", color=colors, alpha=0.8)
    ax.bar(x_pos + width / 2, peak_throughput, width, label="Peak", color=colors, alpha=0.5)

    ax.set_xlabel("Configuration", fontsize=12)
    ax.set_ylabel("Throughput (tokens/s)", fontsize=12)
    ax.set_title("Throughput Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([c.replace("_", "\n") for c in configs], fontsize=10)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved throughput comparison to {output_path}")


def plot_summary_table(results: Dict[str, Dict], output_path: Path):
    """Generate a summary table figure."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("tight")
    ax.axis("off")

    # Prepare table data
    headers = ["Config", "TTFT P50 (s)", "TTFT P99 (s)", "TPOT P50 (ms)", "TPOT P99 (ms)", "Throughput (tok/s)"]
    rows = []

    for config_name, result in results.items():
        m = result["metrics"]
        rows.append([
            config_name.replace("_", " ").title(),
            f"{m.get('median_ttft_ms', 0) / 1000:.2f}",
            f"{m.get('p99_ttft_ms', 0) / 1000:.2f}",
            f"{m.get('median_tpot_ms', 0):.2f}",
            f"{m.get('p99_tpot_ms', 0):.2f}",
            f"{m.get('output_throughput_tok_s', 0):.2f}",
        ])

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f0f0f0")

    plt.title("Attention-Aware Memory Subsystem: Results Summary", fontsize=14, fontweight="bold", pad=20)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved summary table to {output_path}")


def plot_improvement_bars(results: Dict[str, Dict], baseline_key: str, output_path: Path):
    """Generate improvement percentage bars relative to baseline."""
    if baseline_key not in results:
        print(f"WARNING: Baseline '{baseline_key}' not found, skipping improvement plot")
        return

    baseline = results[baseline_key]["metrics"]
    other_configs = {k: v for k, v in results.items() if k != baseline_key}

    if not other_configs:
        print("WARNING: No other configs to compare, skipping improvement plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    metrics_to_compare = [
        ("median_ttft_ms", "TTFT P50", True),  # True = lower is better
        ("p99_ttft_ms", "TTFT P99", True),
        ("median_tpot_ms", "TPOT P50", True),
        ("p99_tpot_ms", "TPOT P99", True),
        ("output_throughput_tok_s", "Throughput", False),  # False = higher is better
    ]

    configs = list(other_configs.keys())
    x_pos = np.arange(len(metrics_to_compare))
    width = 0.8 / len(configs)
    colors = plt.cm.Set2(range(len(configs)))

    for i, config_name in enumerate(configs):
        improvements = []
        for metric_key, _, lower_is_better in metrics_to_compare:
            baseline_val = baseline.get(metric_key, 0)
            config_val = other_configs[config_name]["metrics"].get(metric_key, 0)

            if baseline_val == 0:
                improvement = 0
            else:
                if lower_is_better:
                    improvement = ((baseline_val - config_val) / baseline_val) * 100
                else:
                    improvement = ((config_val - baseline_val) / baseline_val) * 100

            improvements.append(improvement)

        ax.bar(
            x_pos + i * width,
            improvements,
            width,
            label=config_name.replace("_", " ").title(),
            color=colors[i],
            alpha=0.8,
        )

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Improvement over Baseline (%)", fontsize=12)
    ax.set_title(f"Performance Improvement vs {baseline_key.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos + width * (len(configs) - 1) / 2)
    ax.set_xticklabels([label for _, label, _ in metrics_to_compare], fontsize=10)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved improvement comparison to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot attention-aware memory subsystem results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Directory containing result JSON files",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Baseline result JSON file",
    )
    parser.add_argument(
        "--attention",
        type=Path,
        help="Attention-aware result JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory or file prefix (default: results_dir/plots)",
    )
    parser.add_argument(
        "--baseline-key",
        default="baseline_uvm_lru",
        help="Key name for baseline config (default: baseline_uvm_lru)",
    )

    args = parser.parse_args()

    # Load results
    results = {}

    if args.results_dir:
        if not args.results_dir.exists():
            print(f"ERROR: Results directory not found: {args.results_dir}", file=sys.stderr)
            sys.exit(1)

        found_results = find_results_in_dir(args.results_dir)
        if not found_results:
            print(f"ERROR: No result JSON files found in {args.results_dir}", file=sys.stderr)
            sys.exit(1)

        for name, path in found_results.items():
            results[name] = load_result(path)

        output_dir = args.output or args.results_dir
    elif args.baseline and args.attention:
        if not args.baseline.exists():
            print(f"ERROR: Baseline file not found: {args.baseline}", file=sys.stderr)
            sys.exit(1)
        if not args.attention.exists():
            print(f"ERROR: Attention file not found: {args.attention}", file=sys.stderr)
            sys.exit(1)

        results["baseline"] = load_result(args.baseline)
        results["attention_aware"] = load_result(args.attention)

        output_dir = args.output or SCRIPT_DIR / "results"
    else:
        print("ERROR: Must provide either --results-dir or both --baseline and --attention", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    if output_dir.suffix:
        # It's a file path, use its parent
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        prefix = output_dir.stem
        output_dir = output_dir.parent
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = "comparison"

    print(f"Loaded {len(results)} result files")
    print(f"Generating plots in {output_dir}")

    # Generate plots
    plot_latency_comparison(results, output_dir / f"{prefix}_latency.png")
    plot_throughput_comparison(results, output_dir / f"{prefix}_throughput.png")
    plot_summary_table(results, output_dir / f"{prefix}_summary.png")
    plot_improvement_bars(results, args.baseline_key, output_dir / f"{prefix}_improvement.png")

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()

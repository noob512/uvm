#!/usr/bin/env python3
"""Collect trial JSON results and compute summary statistics.

Usage:
    python scripts/collect_results.py \
        --results-dir results/exp1/ncmoe64/ \
        --output results/exp1/ncmoe64/summary.json

    python scripts/collect_results.py \
        --results-dir results/exp1/ \
        --output results/exp1/summary.json \
        --format table

Reads all trial_*.json files, computes geometric mean and stddev for numeric metrics.
"""
import argparse
import json
import math
import sys
from pathlib import Path


def geometric_mean(values: list[float]) -> float:
    """Compute geometric mean of positive values."""
    if not values or any(v <= 0 for v in values):
        # Fall back to arithmetic mean for non-positive values
        return sum(values) / len(values) if values else 0.0
    log_sum = sum(math.log(v) for v in values)
    return math.exp(log_sum / len(values))


def arithmetic_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = arithmetic_mean(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def collect_trials(results_dir: Path) -> list[dict]:
    """Load all trial_*.json files from a directory."""
    trial_files = sorted(results_dir.glob("trial_*.json"))
    trials = []
    for f in trial_files:
        try:
            trials.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: skipping {f}: {e}", file=sys.stderr)
    return trials


def collect_subdirs(results_dir: Path) -> dict[str, list[dict]]:
    """Collect trials from subdirectories (for multi-config experiments)."""
    configs = {}
    for subdir in sorted(results_dir.iterdir()):
        if subdir.is_dir():
            trials = collect_trials(subdir)
            if trials:
                configs[subdir.name] = trials
    return configs


def summarize_trials(trials: list[dict]) -> dict:
    """Compute summary statistics from a list of trial results."""
    if not trials:
        return {}

    # Use first trial as template
    template = trials[0]

    # Collect all numeric metrics across trials
    all_metrics = {}
    for trial in trials:
        for key, value in trial.get("metrics", {}).items():
            if isinstance(value, (int, float)) and value is not None:
                all_metrics.setdefault(key, []).append(float(value))

    # Compute stats
    stats = {}
    for key, values in all_metrics.items():
        stats[key] = {
            "geomean": round(geometric_mean(values), 4),
            "mean": round(arithmetic_mean(values), 4),
            "stddev": round(stddev(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "n": len(values),
        }

    # Duration stats
    durations = [t.get("duration_s", 0) for t in trials if t.get("duration_s")]
    duration_stats = {}
    if durations:
        duration_stats = {
            "mean": round(arithmetic_mean(durations), 2),
            "stddev": round(stddev(durations), 2),
        }

    return {
        "workload": template.get("workload"),
        "config": template.get("config"),
        "params": template.get("params"),
        "n_trials": len(trials),
        "metrics": stats,
        "duration": duration_stats,
    }


def format_table(summary: dict) -> str:
    """Format summary as a readable text table."""
    lines = []
    header = f"  {summary.get('workload', '?')} / {summary.get('config', '?')} ({summary.get('n_trials', 0)} trials)"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    metrics = summary.get("metrics", {})
    if metrics:
        # Find max key length for alignment
        max_key = max(len(k) for k in metrics)
        for key, stats in metrics.items():
            geomean = stats.get("geomean", 0)
            sd = stats.get("stddev", 0)
            lines.append(f"  {key:<{max_key}}  geomean={geomean:>10.4f}  stddev={sd:>8.4f}  n={stats.get('n', 0)}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Collect and summarize trial results")
    parser.add_argument("--results-dir", required=True, help="Directory containing trial_*.json or config subdirs")
    parser.add_argument("--output", "-o", help="Output summary JSON path")
    parser.add_argument("--format", choices=["json", "table", "csv"], default="json",
                        help="Output format for stdout (default: json)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: {results_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # Check if this directory has trial files directly or config subdirs
    direct_trials = collect_trials(results_dir)
    if direct_trials:
        # Single config: trial files directly in results_dir
        summary = summarize_trials(direct_trials)
        output = summary
    else:
        # Multi-config: each subdir has trial files
        configs = collect_subdirs(results_dir)
        if not configs:
            print(f"ERROR: No trial_*.json found in {results_dir} or its subdirectories", file=sys.stderr)
            sys.exit(1)
        summaries = {}
        for config_name, trials in configs.items():
            summaries[config_name] = summarize_trials(trials)
        output = {"configs": summaries}

    # Write JSON output
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(output, indent=2))
        print(f"Summary written to {args.output}", file=sys.stderr)

    # Print to stdout in requested format
    if args.format == "json":
        print(json.dumps(output, indent=2))
    elif args.format == "table":
        if "configs" in output:
            for config_name, summary in output["configs"].items():
                print(format_table(summary))
                print()
        else:
            print(format_table(output))
    elif args.format == "csv":
        # Flat CSV: config, metric, geomean, stddev, n
        if "configs" in output:
            print("config,metric,geomean,stddev,n")
            for config_name, summary in output["configs"].items():
                for metric, stats in summary.get("metrics", {}).items():
                    print(f"{config_name},{metric},{stats['geomean']},{stats['stddev']},{stats['n']}")
        else:
            print("metric,geomean,stddev,n")
            for metric, stats in output.get("metrics", {}).items():
                print(f"{metric},{stats['geomean']},{stats['stddev']},{stats['n']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generic N-trial runner: runs an atomic config script N times, saves per-trial JSON.

Usage:
    python scripts/run_trials.py \
        --trials 10 \
        --command "uv run python llama.cpp/configs/bench.py --ncmoe 64" \
        --results-dir results/exp1/ncmoe64/

    python scripts/run_trials.py \
        --trials 3 \
        --command "uv run python pytorch/configs/gnn.py --nodes 5000000" \
        --results-dir results/exp3/gnn_5M/

Each trial result is saved as trial_01.json, trial_02.json, etc.
After all trials, runs collect_results.py to generate summary.json.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WORKLOADS_DIR = SCRIPT_DIR.parent
from common import cleanup_gpu


def run_trial(command: str, trial_num: int, results_dir: Path) -> dict:
    """Run one trial of the atomic script."""
    output_path = results_dir / f"trial_{trial_num:02d}.json"

    # Append --output and --no-cleanup (cleanup is done by us)
    full_cmd = f"{command} --output {output_path} --no-cleanup"

    print(f"  Trial {trial_num}: {full_cmd}", file=sys.stderr)
    start = time.time()
    result = subprocess.run(
        full_cmd, shell=True, cwd=str(WORKLOADS_DIR),
        capture_output=True, text=True,
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  Trial {trial_num} FAILED (exit {result.returncode}):", file=sys.stderr)
        if result.stderr:
            print(result.stderr[-1000:], file=sys.stderr)
        return None

    if output_path.exists():
        data = json.loads(output_path.read_text())
        print(f"  Trial {trial_num} done ({elapsed:.1f}s)", file=sys.stderr)
        return data
    else:
        print(f"  Trial {trial_num}: no output file generated", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description="Run N trials of an atomic benchmark script")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials")
    parser.add_argument("--command", required=True, help="Atomic script command (without --output)")
    parser.add_argument("--results-dir", required=True, help="Directory to store trial results")
    parser.add_argument("--no-cleanup", action="store_true", help="Skip GPU cleanup between trials")
    parser.add_argument("--no-summary", action="store_true", help="Skip auto-generating summary")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {args.trials} trials → {results_dir}/", file=sys.stderr)
    print(f"Command: {args.command}", file=sys.stderr)

    successes = 0
    failures = 0

    for i in range(1, args.trials + 1):
        # Cleanup GPU before each trial
        if not args.no_cleanup:
            cleanup_gpu()

        result = run_trial(args.command, i, results_dir)
        if result is not None:
            successes += 1
        else:
            failures += 1

    print(f"\nCompleted: {successes}/{args.trials} succeeded, {failures} failed", file=sys.stderr)

    # Auto-run collect_results.py if available
    if not args.no_summary and successes > 0:
        collect_script = SCRIPT_DIR / "collect_results.py"
        if collect_script.exists():
            summary_path = results_dir / "summary.json"
            print(f"Generating summary → {summary_path}", file=sys.stderr)
            subprocess.run(
                [sys.executable, str(collect_script),
                 "--results-dir", str(results_dir),
                 "--output", str(summary_path)],
                cwd=str(WORKLOADS_DIR),
            )

    if failures > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

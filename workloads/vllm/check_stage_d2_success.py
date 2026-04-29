#!/usr/bin/env python3
"""Stage D2 semantic KV budget enforcement success check."""

from __future__ import annotations

import argparse
import subprocess
from datetime import UTC, datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage D2 semantic KV budget enforcement validation. This wraps "
            "check_stage_d_success.py with VLLM_UVM_KV_BUDGET_MODE=enforce and "
            "requires observed KV peak live bytes to stay within the budget."
        )
    )
    parser.add_argument("--budget-bytes", type=int, default=2 * 1024 * 1024 * 1024)
    parser.add_argument("--run-dir", help="Output directory for the D2 check.")
    parser.add_argument("--output-json", help="Where to write the check summary JSON.")
    parser.add_argument("--run-bench", action="store_true")
    parser.add_argument("--prompts", type=int, default=1)
    parser.add_argument("--request-rate", default="5")
    parser.add_argument("--output-len", default="512")
    return parser.parse_args()


def default_run_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return Path(f"/tmp/vllm_stage_d2_success_check_{stamp}")


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir()
    output_json = Path(args.output_json) if args.output_json else (
        run_dir / "stage_d2_success_check.json"
    )

    cmd = [
        str(SCRIPT_DIR / "check_stage_d_success.py"),
        "--run-dir",
        str(run_dir),
        "--output-json",
        str(output_json),
        "--budget-bytes",
        str(args.budget_bytes),
        "--budget-mode",
        "enforce",
        "--prompts",
        str(args.prompts),
        "--request-rate",
        str(args.request_rate),
        "--output-len",
        str(args.output_len),
    ]
    if args.run_bench:
        cmd.append("--run-bench")

    return subprocess.call(cmd, cwd=SCRIPT_DIR)


if __name__ == "__main__":
    raise SystemExit(main())

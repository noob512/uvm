#!/usr/bin/env python3
"""
Attention-Aware Memory Subsystem Benchmark

Standalone benchmark script for testing attention-aware eviction policies.
Can be used independently or called from run_exp_attention_aware.sh.

Usage:
    uv run python benchmark_attention_aware.py --config baseline --output results/baseline.json
    uv run python benchmark_attention_aware.py --config attention_aware --output results/attention.json
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
WORKLOAD_DIR = SCRIPT_DIR
WORKLOADS_DIR = SCRIPT_DIR.parent
EXT_DIR = WORKLOADS_DIR.parent / "extension"

sys.path.insert(0, str(WORKLOADS_DIR / "scripts"))
from common import cleanup_gpu, wait_for_server, stop_server, parse_vllm_bench_output

# Configuration
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-30B-A3B-FP8")
DATASET_PATH = os.environ.get(
    "DATASET_PATH",
    str(WORKLOAD_DIR / "datasets" / "ShareGPT_V3_unfiltered_cleaned_split.json"),
)
VLLM_SERVER_DIR = str(WORKLOAD_DIR / "vllm")

# Benchmark configurations
CONFIGS = {
    "baseline": {
        "name": "Baseline UVM (LRU)",
        "description": "Default UVM with LRU eviction",
        "server_cmd": f"uv run vllm serve {MODEL} --enforce-eager --max-num-seqs 16",
        "env": {"VLLM_USE_UVM": "1"},
        "bpf_program": None,
    },
    "attention_aware": {
        "name": "Attention-Aware Eviction",
        "description": "eBPF-based attention-aware eviction policy",
        "server_cmd": f"uv run vllm serve {MODEL} --enforce-eager --max-num-seqs 16",
        "env": {"VLLM_USE_UVM": "1"},
        "bpf_program": "attention_aware_eviction",
    },
    "cpu_offload": {
        "name": "CPU Offload (Reference)",
        "description": "vLLM native CPU offload for comparison",
        "server_cmd": f"uv run vllm serve {MODEL} --enforce-eager --cpu-offload-gb 8",
        "env": {},
        "bpf_program": None,
    },
}

SERVER_STARTUP_TIMEOUT = 600
SERVER_CHECK_INTERVAL = 5


def start_bpf_program(program_name, log_file):
    """Start a BPF program in the background."""
    bpf_path = EXT_DIR / program_name
    if not bpf_path.exists():
        print(f"WARNING: BPF program not found: {bpf_path}", file=sys.stderr)
        return None

    print(f"Starting BPF program: {program_name}", file=sys.stderr)
    try:
        proc = subprocess.Popen(
            ["sudo", str(bpf_path)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        time.sleep(3)  # Give BPF time to attach

        if proc.poll() is not None:
            print(f"ERROR: BPF program exited immediately", file=sys.stderr)
            return None

        print(f"BPF program started (PID={proc.pid})", file=sys.stderr)
        return proc
    except Exception as e:
        print(f"ERROR: Failed to start BPF program: {e}", file=sys.stderr)
        return None


def stop_bpf_program(proc):
    """Stop a BPF program."""
    if proc and proc.poll() is None:
        print(f"Stopping BPF program (PID={proc.pid})", file=sys.stderr)
        try:
            os.killpg(os.getpgid(proc.pid), 15)  # SIGTERM
            proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), 9)  # SIGKILL
                proc.wait(timeout=5)
            except Exception:
                pass
        time.sleep(2)


def run_benchmark(
    config_name: str,
    num_prompts: int = 50,
    request_rate: float = 2.0,
    port: int = 8000,
    max_concurrency: int = None,
) -> dict:
    """Run a single benchmark configuration."""
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config: {config_name}")

    config = CONFIGS[config_name]
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Running: {config['name']}", file=sys.stderr)
    print(f"Description: {config['description']}", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    # Prepare environment
    env = os.environ.copy()
    env.update(config["env"])

    # Start BPF program if needed
    bpf_proc = None
    bpf_log_file = None
    if config["bpf_program"]:
        bpf_log_path = WORKLOAD_DIR / f"bpf_{config['bpf_program']}.log"
        bpf_log_file = open(bpf_log_path, "w")
        bpf_proc = start_bpf_program(config["bpf_program"], bpf_log_file)
        if not bpf_proc:
            print("WARNING: Continuing without BPF program", file=sys.stderr)

    # Start vLLM server
    server_cmd = config["server_cmd"] + f" --port {port}"
    print(f"Starting vLLM server (port={port})...", file=sys.stderr)

    server_log_file = open(WORKLOAD_DIR / f"vllm_server_{config_name}.log", "w")
    server_proc = subprocess.Popen(
        server_cmd,
        shell=True,
        cwd=str(WORKLOAD_DIR),
        env=env,
        stdout=server_log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

    try:
        # Wait for server to be ready
        if not wait_for_server(
            port=port,
            timeout=SERVER_STARTUP_TIMEOUT,
            check_interval=SERVER_CHECK_INTERVAL,
            process=server_proc,
        ):
            print("ERROR: Server failed to start", file=sys.stderr)
            stop_server(server_proc)
            if bpf_proc:
                stop_bpf_program(bpf_proc)
            sys.exit(1)

        print(f"Server ready. Running benchmark ({num_prompts} prompts)...", file=sys.stderr)

        # Build benchmark command
        bench_cmd = (
            f"uv run vllm bench serve "
            f"--model {MODEL} "
            f"--dataset-name sharegpt "
            f"--dataset-path {DATASET_PATH} "
            f"--num-prompts {num_prompts} "
            f"--sharegpt-output-len 512 --seed 42 "
            f"--request-rate {request_rate} "
            f"--port {port}"
        )

        if max_concurrency:
            bench_cmd += f" --max-concurrency {max_concurrency}"

        start_time = time.time()
        bench_result = subprocess.run(
            bench_cmd,
            shell=True,
            cwd=str(WORKLOAD_DIR),
            capture_output=True,
            text=True,
        )
        duration = time.time() - start_time

        # Parse results
        metrics = parse_vllm_bench_output(bench_result.stdout)

        result = {
            "workload": "vllm",
            "config": config_name,
            "config_description": config["description"],
            "params": {
                "model": MODEL,
                "num_prompts": num_prompts,
                "request_rate": request_rate,
                "max_concurrency": max_concurrency,
                "bpf_program": config["bpf_program"],
            },
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "duration_s": duration,
            "raw": {
                "bench_stdout": bench_result.stdout,
                "bench_stderr": bench_result.stderr,
            },
        }

        return result

    finally:
        # Cleanup
        print("Stopping server...", file=sys.stderr)
        stop_server(server_proc)
        server_log_file.close()

        if bpf_proc:
            stop_bpf_program(bpf_proc)
            if bpf_log_file:
                bpf_log_file.close()

        time.sleep(2)


def main():
    parser = argparse.ArgumentParser(
        description="Attention-Aware Memory Subsystem Benchmark"
    )
    parser.add_argument(
        "--config",
        required=True,
        choices=list(CONFIGS.keys()),
        help="Benchmark configuration",
    )
    parser.add_argument(
        "--prompts", type=int, default=50, help="Number of prompts (default: 50)"
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=2.0,
        help="Request rate in RPS (default: 2.0)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum concurrent requests (default: unlimited)",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Server port (default: 8000)"
    )
    parser.add_argument(
        "--output", "-o", help="Output JSON path (default: stdout)"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Skip GPU cleanup before benchmark",
    )

    args = parser.parse_args()

    # Validate prerequisites
    if not Path(VLLM_SERVER_DIR).exists():
        print(f"ERROR: vLLM not found at {VLLM_SERVER_DIR}", file=sys.stderr)
        sys.exit(1)

    if not Path(DATASET_PATH).exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}", file=sys.stderr)
        print("Run: cd workloads/vllm && python download_sharegpt.py --dataset vicuna", file=sys.stderr)
        sys.exit(1)

    # Cleanup GPU
    if not args.no_cleanup:
        cleanup_gpu()

    # Run benchmark
    result = run_benchmark(
        args.config,
        num_prompts=args.prompts,
        request_rate=args.request_rate,
        port=args.port,
        max_concurrency=args.max_concurrency,
    )

    # Output results
    output_json = json.dumps(result, indent=2)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output_json)
        print(f"Results written to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()

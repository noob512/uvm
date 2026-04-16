#!/usr/bin/env python3
"""POC-1: xCoord GPU->CPU Coordination benchmark.

Uses the same benchmark methodology as server_bench.py (common.py utilities),
adding xCoord lifecycle management (eviction_lfu_xcoord + sched_gpu_baseline)
and CPU stress interference.

Scenarios:
  1. uvm_baseline    - 120B UVM, no CPU stress (reference)
  2. uvm_cpu_stress  - 120B UVM + CPU stress, no xCoord
  3. uvm_xcoord      - 120B UVM + CPU stress + xCoord

Usage:
    uv run --directory workloads/llama.cpp python ../../scripts/xcoord/poc1_xcoord_bench.py \
        --output-dir scripts/xcoord/results/poc1_test

    # Single scenario:
    uv run --directory workloads/llama.cpp python ../../scripts/xcoord/poc1_xcoord_bench.py \
        --output-dir scripts/xcoord/results/poc1_test --scenarios uvm_xcoord

Note: xCoord components (eviction_lfu_xcoord, sched_gpu_baseline) require sudo.
      The script will call sudo internally for BPF tools only.
      Do NOT run the entire script with sudo (model cache is under $HOME).
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
WORKLOADS_DIR = REPO_ROOT / "workloads"
LLAMA_WORKLOAD_DIR = WORKLOADS_DIR / "llama.cpp"

# Import common utilities (same as server_bench.py)
sys.path.insert(0, str(WORKLOADS_DIR / "scripts"))
from common import cleanup_gpu, wait_for_server, stop_server, parse_vllm_bench_output

# Paths (matching server_bench.py layout)
VLLM_WORKLOAD_DIR = WORKLOADS_DIR / "vllm"
LLAMA_SERVER = LLAMA_WORKLOAD_DIR / "build" / "bin" / "llama-server"
DATASET_DIR = LLAMA_WORKLOAD_DIR / "datasets"
DATASET_PATH = DATASET_DIR / "sharegpt_vicuna.json"
DEFAULT_TOKENIZER = "Qwen/Qwen3-30B-A3B-FP8"

# Model paths
DEFAULT_MODEL_120B = (
    Path.home()
    / ".cache/llama.cpp"
    / "ggml-org_gpt-oss-120b-GGUF_gpt-oss-120b-mxfp4-00001-of-00003.gguf"
)
DEFAULT_MODEL_20B = (
    Path.home()
    / ".cache/llama.cpp"
    / "ggml-org_gpt-oss-20b-GGUF_gpt-oss-20b-mxfp4.gguf"
)

# xCoord binaries (default; can be overridden via --scheduler)
EVICTION_XCOORD = REPO_ROOT / "extension" / "eviction_lfu_xcoord"
SCHED_GPU_AWARE = REPO_ROOT / "extension" / "sched_gpu_baseline"
SCHED_GPU_SERVING = REPO_ROOT / "extension" / "sched_gpu_serving"

# xCoord pinned map paths
XCOORD_GPU_STATE_PIN = Path("/sys/fs/bpf/xcoord_gpu_state")
XCOORD_UVM_WORKERS_PIN = Path("/sys/fs/bpf/xcoord_uvm_workers")

SERVER_STARTUP_TIMEOUT = 300
SERVER_CHECK_INTERVAL = 5

# Scenario definitions: (with_stress, with_xcoord)
SCENARIOS = {
    "uvm_baseline": (False, False),
    "uvm_cpu_stress": (True, False),
    "uvm_xcoord": (True, True),
}


def cleanup_xcoord_stale():
    """Kill any stale xCoord BPF processes and clean pinned maps."""
    # Use pkill -x (exact process name match) to avoid killing ourselves.
    # pkill -f matches the full command line and would kill this python script
    # if the scheduler name appears as an argument.
    for name in ["eviction_lfu_xcoord", "sched_gpu_baseline", "sched_gpu_serving",
                 "sched_gpu_minimal"]:
        subprocess.run(["sudo", "pkill", "-x", name], capture_output=True)
    time.sleep(1)
    # Remove stale pinned maps
    for pin in [XCOORD_GPU_STATE_PIN, XCOORD_UVM_WORKERS_PIN]:
        subprocess.run(["sudo", "rm", "-f", str(pin)], capture_output=True)


def verify_gpu_clean():
    """Verify no GPU compute processes are running."""
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    pids = result.stdout.strip()
    if pids:
        print(f"  WARNING: GPU still has processes: {pids}", file=sys.stderr)
        return False
    return True


def start_xcoord(server_pid, output_dir, skip_gpu_ext=False, sched_binary=None):
    """Start xCoord GPU and CPU components. Returns dict of processes or None on failure.

    Args:
        server_pid: PID of the GPU server process to boost.
        output_dir: Directory for log files.
        skip_gpu_ext: If True, skip eviction_lfu_xcoord (for workloads that fit in VRAM).
    """
    procs = {}

    if not skip_gpu_ext:
        # GPU side: eviction_lfu_xcoord
        gpu_log = open(output_dir / "xcoord_gpu.log", "w")
        gpu_proc = subprocess.Popen(
            ["sudo", str(EVICTION_XCOORD), "-p", str(server_pid), "-P", "1", "-d", "5"],
            stdout=gpu_log, stderr=subprocess.STDOUT,
        )
        procs["gpu"] = (gpu_proc, gpu_log)
        time.sleep(2)

        # Verify pinned map exists (needs sudo to stat /sys/fs/bpf/)
        check = subprocess.run(["sudo", "test", "-e", str(XCOORD_GPU_STATE_PIN)])
        if check.returncode != 0:
            print("  ERROR: gpu_state_map not pinned!", file=sys.stderr)
            stop_xcoord(procs)
            return None
        print(f"  eviction_lfu_xcoord loaded (PID: {gpu_proc.pid})", file=sys.stderr)
    else:
        print("  Skipping eviction_lfu_xcoord (model fits in VRAM)", file=sys.stderr)

    # CPU side: sched_ext scheduler (always needed)
    # Pass server PID via -p for direct process boosting
    sched_bin = sched_binary or SCHED_GPU_AWARE
    sched_cmd = ["sudo", str(sched_bin), "-p", str(server_pid)]
    if str(sched_bin).endswith("sched_gpu_baseline"):
        sched_cmd += ["-t", "1000"]
    cpu_log = open(output_dir / "xcoord_cpu.log", "w")
    cpu_proc = subprocess.Popen(
        sched_cmd,
        stdout=cpu_log, stderr=subprocess.STDOUT,
    )
    procs["cpu"] = (cpu_proc, cpu_log)
    time.sleep(2)

    # Check sched_ext state
    try:
        state = Path("/sys/kernel/sched_ext/state").read_text().strip()
        print(f"  sched_ext state: {state}", file=sys.stderr)
    except Exception:
        print("  sched_ext state: unknown", file=sys.stderr)

    print(f"  sched_gpu_baseline loaded (PID: {cpu_proc.pid}, boosting server PID: {server_pid})",
          file=sys.stderr)
    return procs


def stop_xcoord(procs):
    """Stop xCoord components and clean up pinned maps."""
    if not procs:
        return
    # Stop in reverse order: CPU scheduler first, then GPU
    for name in ["cpu", "gpu"]:
        if name not in procs:
            continue
        proc, log = procs[name]
        if proc.poll() is None:
            subprocess.run(["sudo", "kill", str(proc.pid)], capture_output=True)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                subprocess.run(["sudo", "kill", "-9", str(proc.pid)], capture_output=True)
                proc.wait(timeout=5)
        log.close()
    # Clean up pinned maps
    for pin in [XCOORD_GPU_STATE_PIN, XCOORD_UVM_WORKERS_PIN]:
        subprocess.run(["sudo", "rm", "-f", str(pin)], capture_output=True)


def start_stress():
    """Start CPU stress on all cores. Returns process."""
    nproc = os.cpu_count() or 1
    proc = subprocess.Popen(
        ["stress-ng", "-c", str(nproc), "--timeout", "600"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print(f"  stress-ng started (PID: {proc.pid}, {nproc} cores)", file=sys.stderr)
    return proc


def stop_stress(proc):
    """Stop stress-ng."""
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    subprocess.run(["killall", "stress-ng"], capture_output=True)


def run_scenario(scenario, model, ctx, port, prompts, max_concurrency,
                 request_rate, output_dir, skip_gpu_ext=False, sched_binary=None):
    """Run a single benchmark scenario. Returns result dict or None."""
    with_stress, with_xcoord = SCENARIOS[scenario]

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Scenario: {scenario}", file=sys.stderr)
    print(f"  CPU stress: {with_stress}, xCoord: {with_xcoord}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    # Clean stale xCoord processes first
    cleanup_xcoord_stale()

    # GPU cleanup (same as server_bench.py)
    cleanup_gpu()
    if not verify_gpu_clean():
        print("  ERROR: GPU not clean, aborting scenario", file=sys.stderr)
        return None

    # Build server command (same as server_bench.py)
    cmd = [
        str(LLAMA_SERVER),
        "-m", model,
        "-c", str(ctx),
        "-ngl", "99",
        "--host", "0.0.0.0",
        "--port", str(port),
    ]
    env = os.environ.copy()
    env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"
    env["GGML_CUDA_DISABLE_GRAPHS"] = "1"

    # Start server (same pattern as server_bench.py, but save logs)
    server_log = open(output_dir / f"server_{scenario}.log", "w")
    print(f"  Starting llama-server (uvm=True, ctx={ctx}, port={port})...", file=sys.stderr)
    server_proc = subprocess.Popen(
        cmd, env=env,
        stdout=server_log, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

    xcoord_procs = None
    stress_proc = None
    metrics = {}
    bench_output = ""
    elapsed = 0

    try:
        # Wait for server (same as server_bench.py)
        if not wait_for_server("127.0.0.1", port, SERVER_STARTUP_TIMEOUT,
                               check_interval=SERVER_CHECK_INTERVAL,
                               process=server_proc):
            print("  ERROR: Server failed to start", file=sys.stderr)
            return None

        print(f"  Server ready (PID: {server_proc.pid}).", file=sys.stderr)

        # Download dataset if needed (same as server_bench.py)
        if not DATASET_PATH.exists():
            print("  Downloading ShareGPT dataset...", file=sys.stderr)
            subprocess.run(
                [sys.executable,
                 str(LLAMA_WORKLOAD_DIR / "download_sharegpt.py"),
                 "--dataset", "vicuna"],
                cwd=str(LLAMA_WORKLOAD_DIR),
            )

        # Warmup (2 prompts, not counted)
        model_name = Path(model).stem
        print("  Warmup (2 prompts)...", file=sys.stderr)
        warmup_cmd = (
            f"uv run --directory {VLLM_WORKLOAD_DIR} vllm bench serve "
            f"--model {model_name} "
            f"--tokenizer {DEFAULT_TOKENIZER} "
            f"--dataset-name sharegpt "
            f"--dataset-path {DATASET_PATH} "
            f"--base-url http://127.0.0.1:{port} "
            f"--num-prompts 2 "
            f"--max-concurrency 1 "
            f"--request-rate 0.1"
        )
        subprocess.run(warmup_cmd, shell=True, capture_output=True)
        time.sleep(3)

        # Start xCoord (after warmup, before stress)
        if with_xcoord:
            print("  Starting xCoord...", file=sys.stderr)
            xcoord_procs = start_xcoord(server_proc.pid, output_dir,
                                        skip_gpu_ext=skip_gpu_ext,
                                        sched_binary=sched_binary)
            if xcoord_procs is None:
                print("  ERROR: Failed to start xCoord", file=sys.stderr)
                return None

        # Start CPU stress
        if with_stress:
            print("  Starting CPU stress...", file=sys.stderr)
            stress_proc = start_stress()
            time.sleep(2)

        # Post-stress warmup: let scheduler stabilize under load
        if with_xcoord and with_stress:
            print("  Post-stress warmup (3 prompts, not counted)...", file=sys.stderr)
            post_warmup_cmd = (
                f"uv run --directory {VLLM_WORKLOAD_DIR} vllm bench serve "
                f"--model {model_name} "
                f"--tokenizer {DEFAULT_TOKENIZER} "
                f"--dataset-name sharegpt "
                f"--dataset-path {DATASET_PATH} "
                f"--base-url http://127.0.0.1:{port} "
                f"--num-prompts 3 "
                f"--max-concurrency 1 "
                f"--request-rate 0.5"
            )
            subprocess.run(post_warmup_cmd, shell=True, capture_output=True)
            time.sleep(2)

        # Run benchmark (same as server_bench.py)
        print(f"  Running benchmark ({prompts} prompts)...", file=sys.stderr)
        bench_cmd = (
            f"uv run --directory {VLLM_WORKLOAD_DIR} vllm bench serve "
            f"--model {model_name} "
            f"--tokenizer {DEFAULT_TOKENIZER} "
            f"--dataset-name sharegpt "
            f"--dataset-path {DATASET_PATH} "
            f"--base-url http://127.0.0.1:{port} "
            f"--num-prompts {prompts} "
            f"--max-concurrency {max_concurrency} "
            f"--request-rate {request_rate}"
        )

        start = time.time()
        bench_result = subprocess.run(
            bench_cmd, shell=True,
            capture_output=True, text=True,
        )
        elapsed = time.time() - start

        bench_output = bench_result.stdout + bench_result.stderr

        if bench_result.returncode != 0:
            print(f"  Benchmark failed (exit {bench_result.returncode}):", file=sys.stderr)
            print(bench_output[-2000:], file=sys.stderr)

        # Save raw bench output
        (output_dir / f"bench_{scenario}.txt").write_text(bench_output)

        # Parse metrics (same as server_bench.py)
        metrics = parse_vllm_bench_output(bench_output)

    finally:
        # Cleanup in reverse order
        if stress_proc:
            print("  Stopping stress-ng...", file=sys.stderr)
            stop_stress(stress_proc)
        if xcoord_procs:
            print("  Stopping xCoord...", file=sys.stderr)
            stop_xcoord(xcoord_procs)
        print("  Stopping server...", file=sys.stderr)
        stop_server(server_proc)
        server_log.close()
        time.sleep(5)

    print(f"  Scenario {scenario} complete.", file=sys.stderr)

    return {
        "workload": "llama.cpp",
        "config": f"poc1_{scenario}",
        "params": {
            "model": Path(model).name,
            "uvm": True,
            "ctx": ctx,
            "prompts": prompts,
            "max_concurrency": max_concurrency,
            "request_rate": request_rate,
            "cpu_stress": with_stress,
            "xcoord": with_xcoord,
        },
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
        "duration_s": round(elapsed, 2),
        "raw": {"bench_output": bench_output},
    }


def print_summary(results):
    """Print comparison table across scenarios."""
    print(f"\n{'='*60}", file=sys.stderr)
    print("POC-1 Results Summary", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    header = f"{'Scenario':<20} {'TTFT(ms)':>10} {'Median':>10} {'Throughput':>12} {'Requests':>10}"
    print(header, file=sys.stderr)
    print("-" * len(header), file=sys.stderr)

    for name, result in results.items():
        m = result.get("metrics", {})
        ttft = m.get("mean_ttft_ms", "N/A")
        median = m.get("median_ttft_ms", "N/A")
        tput = m.get("output_throughput_tok_s", "N/A")
        reqs = m.get("successful_requests", "N/A")

        ttft_s = f"{ttft:.1f}" if isinstance(ttft, (int, float)) else ttft
        median_s = f"{median:.1f}" if isinstance(median, (int, float)) else median
        tput_s = f"{tput:.2f}" if isinstance(tput, (int, float)) else tput

        print(f"{name:<20} {ttft_s:>10} {median_s:>10} {tput_s:>12} {reqs:>10}",
              file=sys.stderr)

    print("", file=sys.stderr)

    # Show improvement if all three scenarios present
    if "uvm_cpu_stress" in results and "uvm_xcoord" in results:
        stress_m = results["uvm_cpu_stress"].get("metrics", {})
        xcoord_m = results["uvm_xcoord"].get("metrics", {})
        stress_ttft = stress_m.get("mean_ttft_ms")
        xcoord_ttft = xcoord_m.get("mean_ttft_ms")
        if stress_ttft and xcoord_ttft:
            improvement = (stress_ttft - xcoord_ttft) / stress_ttft * 100
            print(f"xCoord TTFT improvement vs cpu_stress: {improvement:+.1f}%", file=sys.stderr)
        stress_tput = stress_m.get("output_throughput_tok_s")
        xcoord_tput = xcoord_m.get("output_throughput_tok_s")
        if stress_tput and xcoord_tput:
            improvement = (xcoord_tput - stress_tput) / stress_tput * 100
            print(f"xCoord throughput improvement vs cpu_stress: {improvement:+.1f}%",
                  file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="POC-1: xCoord GPU->CPU Coordination benchmark")
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", default=None,
                             help="Model path (explicit)")
    model_group.add_argument("--model-20b", action="store_true",
                             help="Use 20B model (~10GB, fits in VRAM)")
    parser.add_argument("--ctx", type=int, default=None,
                        help="Context size (default: 4096 for 120B, 65536 for 20B)")
    parser.add_argument("--port", type=int, default=8013, help="Server port")
    parser.add_argument("--prompts", type=int, default=20,
                        help="Number of ShareGPT prompts per scenario")
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--request-rate", type=float, default=0.2)
    parser.add_argument("--output-dir", "-o", required=True,
                        help="Output directory for results")
    parser.add_argument("--scenarios", nargs="+",
                        default=list(SCENARIOS.keys()),
                        choices=list(SCENARIOS.keys()),
                        help="Scenarios to run (default: all)")
    parser.add_argument("--skip-gpu-ext", action="store_true",
                        help="Skip eviction_lfu_xcoord (for models that fit in VRAM)")
    parser.add_argument("--scheduler", type=Path,
                        help="Path to sched_ext scheduler binary (default: sched_gpu_baseline)")
    args = parser.parse_args()

    # Model selection
    if args.model_20b:
        args.model = str(DEFAULT_MODEL_20B)
        if args.ctx is None:
            args.ctx = 65536
        # 20B fits in VRAM — no UVM paging, skip gpu_ext by default
        if not args.skip_gpu_ext:
            args.skip_gpu_ext = True
            print("  Note: auto-enabling --skip-gpu-ext for 20B model", file=sys.stderr)
    else:
        if args.model is None:
            args.model = str(DEFAULT_MODEL_120B)
        if args.ctx is None:
            args.ctx = 4096

    # Prerequisites
    if not LLAMA_SERVER.exists():
        print(f"ERROR: llama-server not found at {LLAMA_SERVER}", file=sys.stderr)
        print(f"Run: cd {LLAMA_WORKLOAD_DIR} && make build-cuda-no-vmm", file=sys.stderr)
        sys.exit(1)
    if not Path(args.model).exists():
        print(f"ERROR: Model not found at {args.model}", file=sys.stderr)
        sys.exit(1)
    if not VLLM_WORKLOAD_DIR.exists():
        print(f"ERROR: vllm workload not found at {VLLM_WORKLOAD_DIR}", file=sys.stderr)
        sys.exit(1)
    for xcoord_scenario in args.scenarios:
        if SCENARIOS[xcoord_scenario][1]:  # needs xCoord
            required = [SCHED_GPU_AWARE]
            if not args.skip_gpu_ext:
                required.append(EVICTION_XCOORD)
            for binary in required:
                if not binary.exists():
                    print(f"ERROR: {binary.name} not found at {binary}", file=sys.stderr)
                    print(f"Run: cd {REPO_ROOT / 'extension'} && make {binary.name}",
                          file=sys.stderr)
                    sys.exit(1)
            break

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"poc1_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}", file=sys.stderr)
    print("POC-1: xCoord GPU->CPU Coordination", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  Model: {Path(args.model).name}", file=sys.stderr)
    print(f"  Context: {args.ctx}", file=sys.stderr)
    print(f"  Prompts: {args.prompts}", file=sys.stderr)
    print(f"  Scenarios: {args.scenarios}", file=sys.stderr)
    print(f"  Output: {output_dir}", file=sys.stderr)

    results = {}
    for scenario in args.scenarios:
        result = run_scenario(
            scenario, args.model, args.ctx, args.port,
            args.prompts, args.max_concurrency, args.request_rate,
            output_dir, skip_gpu_ext=args.skip_gpu_ext,
            sched_binary=args.scheduler,
        )
        if result:
            results[scenario] = result
            # Save individual result JSON
            result_path = output_dir / f"result_{scenario}.json"
            result_path.write_text(json.dumps(result, indent=2))
            print(f"  Result: {result_path}", file=sys.stderr)

    # Save combined results
    combined_path = output_dir / "poc1_results.json"
    combined_path.write_text(json.dumps(results, indent=2))

    # Print summary
    print_summary(results)

    print(f"\nAll results saved to: {output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""xCoord single-workload benchmark: test CPU-GPU coupling across workloads.

Runs a workload under multiple scenarios to measure CPU stress impact
and xCoord recovery. Each scenario ~2-5min, total ~10-20min.

Scenarios:
  B0: baseline (no stress)
  B1: + stress-ng (24 cores)
  B2: + stress + taskset (pin workload to cores 0-3)
  B3: + stress + blind boost (sched_gpu_baseline, current POC-1)
  B4: + stress + gpu_ext + blind boost (with eviction_lfu_xcoord for UVM workloads)

Usage:
    # GNN UVM workload:
    uv run --directory workloads/pytorch python ../../scripts/xcoord/xcoord_workload_bench.py \
        --workload gnn --gnn-nodes 5000000 --gnn-epochs 5 -o results/xcoord_gnn

    # llama.cpp 20B:
    uv run --directory workloads/llama.cpp python ../../scripts/xcoord/xcoord_workload_bench.py \
        --workload llama20b --llama-prompts 20 -o results/xcoord_llama20b
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
EXTENSION_DIR = REPO_ROOT / "extension"

# xCoord binaries
SCHED_GPU_AWARE = EXTENSION_DIR / "sched_gpu_baseline"
EVICTION_XCOORD = EXTENSION_DIR / "eviction_lfu_xcoord"

# Pin paths
XCOORD_GPU_STATE_PIN = Path("/sys/fs/bpf/xcoord_gpu_state")
XCOORD_UVM_WORKERS_PIN = Path("/sys/fs/bpf/xcoord_uvm_workers")

SCENARIOS = ["B0_baseline", "B1_stress", "B2_stress_taskset", "B3_stress_blind_boost",
             "B4_stress_gpuext_boost"]


def cleanup_xcoord():
    """Kill stale xCoord processes and clean pinned maps."""
    subprocess.run(["sudo", "pkill", "-f", "eviction_lfu_xcoord"], capture_output=True)
    subprocess.run(["sudo", "pkill", "-f", "sched_gpu_baseline"], capture_output=True)
    time.sleep(1)
    for pin in [XCOORD_GPU_STATE_PIN, XCOORD_UVM_WORKERS_PIN]:
        subprocess.run(["sudo", "rm", "-f", str(pin)], capture_output=True)


def cleanup_gpu():
    """Run GPU cleanup."""
    subprocess.run([sys.executable, str(REPO_ROOT / "workloads" / "cleanup_gpu.py")],
                   capture_output=True)


def start_stress():
    """Start stress-ng on all cores."""
    nproc = os.cpu_count() or 1
    proc = subprocess.Popen(
        ["stress-ng", "-c", str(nproc), "--timeout", "600"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print(f"    stress-ng started (PID: {proc.pid}, {nproc} cores)", file=sys.stderr)
    return proc


def stop_stress(proc):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    subprocess.run(["killall", "stress-ng"], capture_output=True)


def start_sched_gpu_baseline(workload_pid, log_dir):
    """Start sched_gpu_baseline with -p PID."""
    log = open(log_dir / "sched_gpu_baseline.log", "w")
    proc = subprocess.Popen(
        ["sudo", str(SCHED_GPU_AWARE), "-p", str(workload_pid)],
        stdout=log, stderr=subprocess.STDOUT,
    )
    time.sleep(2)
    # Verify sched_ext loaded
    try:
        state = Path("/sys/kernel/sched_ext/state").read_text().strip()
        print(f"    sched_ext state: {state}", file=sys.stderr)
    except Exception:
        pass
    print(f"    sched_gpu_baseline loaded (PID: {proc.pid}, boosting: {workload_pid})", file=sys.stderr)
    return proc, log


def start_eviction_xcoord(workload_pid, log_dir):
    """Start eviction_lfu_xcoord for UVM workloads."""
    log = open(log_dir / "eviction_xcoord.log", "w")
    proc = subprocess.Popen(
        ["sudo", str(EVICTION_XCOORD), "-p", str(workload_pid), "-P", "1", "-d", "5"],
        stdout=log, stderr=subprocess.STDOUT,
    )
    time.sleep(2)
    # Check pinned map
    check = subprocess.run(["sudo", "test", "-e", str(XCOORD_GPU_STATE_PIN)])
    if check.returncode == 0:
        print(f"    eviction_lfu_xcoord loaded (PID: {proc.pid})", file=sys.stderr)
    else:
        print(f"    WARNING: gpu_state_map not pinned", file=sys.stderr)
    return proc, log


def stop_xcoord_procs(procs):
    """Stop xCoord processes."""
    for name, (proc, log) in procs.items():
        if proc.poll() is None:
            subprocess.run(["sudo", "kill", str(proc.pid)], capture_output=True)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                subprocess.run(["sudo", "kill", "-9", str(proc.pid)], capture_output=True)
        log.close()
    for pin in [XCOORD_GPU_STATE_PIN, XCOORD_UVM_WORKERS_PIN]:
        subprocess.run(["sudo", "rm", "-f", str(pin)], capture_output=True)


# ===== Workload runners =====

def run_gnn(nodes, uvm, epochs, warmup, output_path, taskset_cores=None, env_extra=None):
    """Run GNN training benchmark. Returns result dict."""
    workload_dir = REPO_ROOT / "workloads" / "pytorch"
    cmd_prefix = []
    if taskset_cores:
        cmd_prefix = ["taskset", "-c", taskset_cores]

    cmd = cmd_prefix + [
        "uv", "run", "--directory", str(workload_dir),
        "python", "configs/gnn.py",
        "--nodes", str(nodes),
        "--epochs", str(epochs),
        "--warmup", str(warmup),
        "--no-cleanup",
        "-o", str(output_path),
    ]
    if uvm:
        cmd.append("--uvm")

    env = os.environ.copy()
    if uvm:
        env["CUDA_MANAGED_FORCE_DEVICE_ALLOC"] = "1"
    if env_extra:
        env.update(env_extra)

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    GNN failed: {result.stderr[-500:]}", file=sys.stderr)
        return None

    return json.loads(Path(output_path).read_text())


def get_gnn_pid():
    """Find the running GNN python process PID."""
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    pids = result.stdout.strip().split("\n")
    for pid in pids:
        pid = pid.strip()
        if pid:
            return int(pid)
    return None


def run_llama_bench(model, ctx, port, prompts, max_concurrency, request_rate,
                    output_dir, scenario_name, taskset_cores=None):
    """Run llama.cpp server benchmark. Returns result dict."""
    llama_dir = REPO_ROOT / "workloads" / "llama.cpp"
    vllm_dir = REPO_ROOT / "workloads" / "vllm"
    llama_server = llama_dir / "build" / "bin" / "llama-server"
    dataset_path = llama_dir / "datasets" / "sharegpt_vicuna.json"

    cmd = [str(llama_server), "-m", model, "-c", str(ctx), "-ngl", "99",
           "--host", "0.0.0.0", "--port", str(port)]
    if taskset_cores:
        cmd = ["taskset", "-c", taskset_cores] + cmd

    env = os.environ.copy()
    env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"
    env["GGML_CUDA_DISABLE_GRAPHS"] = "1"

    server_log = open(output_dir / f"server_{scenario_name}.log", "w")
    server_proc = subprocess.Popen(cmd, env=env, stdout=server_log, stderr=subprocess.STDOUT,
                                   preexec_fn=os.setsid)

    # Import common utilities
    sys.path.insert(0, str(REPO_ROOT / "workloads" / "scripts"))
    from common import wait_for_server, stop_server

    if not wait_for_server("127.0.0.1", port, 300, check_interval=5, process=server_proc):
        print("    Server failed to start", file=sys.stderr)
        server_log.close()
        return None, None

    return server_proc, server_log


# ===== Main scenario runner =====

def run_scenario_gnn(scenario, args, output_dir):
    """Run a single GNN scenario."""
    print(f"\n  === {scenario} ===", file=sys.stderr)

    cleanup_xcoord()
    cleanup_gpu()
    time.sleep(2)

    stress_proc = None
    xcoord_procs = {}
    taskset_cores = None

    # Determine scenario flags
    with_stress = scenario != "B0_baseline"
    with_taskset = scenario == "B2_stress_taskset"
    with_boost = scenario in ("B3_stress_blind_boost", "B4_stress_gpuext_boost")
    with_gpuext = scenario == "B4_stress_gpuext_boost"

    if with_taskset:
        taskset_cores = "0-3"
        print(f"    taskset cores: {taskset_cores}", file=sys.stderr)

    result_path = output_dir / f"result_{scenario}.json"

    try:
        # For xCoord scenarios, we need to start the workload first to get PID,
        # then attach xCoord. But GNN is a batch job that runs and exits.
        # Solution: start stress first, then run GNN with xCoord wrapping.

        if with_stress:
            stress_proc = start_stress()
            time.sleep(2)

        # For boost scenarios, we need to start sched_gpu_baseline before the workload.
        # We'll use PID=0 trick — actually we need to know the PID.
        # For GNN, the process starts and runs, so we can't get PID before it starts.
        # Alternative: start GNN in background, get PID, start xCoord, wait for GNN.

        if with_boost or with_gpuext:
            # Run GNN in background
            workload_dir = REPO_ROOT / "workloads" / "pytorch"
            gnn_cmd = [
                "uv", "run", "--directory", str(workload_dir),
                "python", "configs/gnn.py",
                "--nodes", str(args.gnn_nodes),
                "--epochs", str(args.gnn_epochs),
                "--warmup", str(args.gnn_warmup),
                "--no-cleanup",
                "-o", str(result_path),
            ]
            if args.gnn_uvm:
                gnn_cmd.append("--uvm")

            gnn_env = os.environ.copy()
            if args.gnn_uvm:
                gnn_env["CUDA_MANAGED_FORCE_DEVICE_ALLOC"] = "1"

            gnn_log = open(output_dir / f"gnn_{scenario}.log", "w")
            gnn_proc = subprocess.Popen(gnn_cmd, env=gnn_env,
                                        stdout=gnn_log, stderr=subprocess.STDOUT)

            # Wait for GPU process to appear
            print("    Waiting for GPU process...", file=sys.stderr)
            gpu_pid = None
            for _ in range(60):
                time.sleep(2)
                gpu_pid = get_gnn_pid()
                if gpu_pid:
                    break
            if not gpu_pid:
                print("    WARNING: Could not find GPU PID, using GNN proc PID", file=sys.stderr)
                gpu_pid = gnn_proc.pid

            print(f"    GPU PID: {gpu_pid}", file=sys.stderr)

            if with_gpuext and args.gnn_uvm:
                evict_proc, evict_log = start_eviction_xcoord(gpu_pid, output_dir)
                xcoord_procs["evict"] = (evict_proc, evict_log)

            sched_proc, sched_log = start_sched_gpu_baseline(gpu_pid, output_dir)
            xcoord_procs["sched"] = (sched_proc, sched_log)

            # Wait for GNN to finish
            gnn_proc.wait()
            gnn_log.close()

        else:
            # Simple run (no xCoord)
            result = run_gnn(args.gnn_nodes, args.gnn_uvm, args.gnn_epochs,
                             args.gnn_warmup, result_path,
                             taskset_cores=taskset_cores)

    finally:
        if stress_proc:
            stop_stress(stress_proc)
        if xcoord_procs:
            stop_xcoord_procs(xcoord_procs)

    # Read result
    if result_path.exists():
        result = json.loads(result_path.read_text())
        result["scenario"] = scenario
        result_path.write_text(json.dumps(result, indent=2))
        m = result.get("metrics", {})
        print(f"    avg_epoch: {m.get('avg_epoch_time_s', 'N/A'):.2f}s", file=sys.stderr)
        return result
    return None


def print_summary(results):
    """Print comparison table."""
    print(f"\n{'='*70}", file=sys.stderr)
    print("xCoord Workload Benchmark Results", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)

    header = f"{'Scenario':<30} {'Avg Epoch(s)':>12} {'Median(s)':>10} {'Duration(s)':>12}"
    print(header, file=sys.stderr)
    print("-" * len(header), file=sys.stderr)

    baseline_val = None
    for name, r in results.items():
        m = r.get("metrics", {})
        avg = m.get("avg_epoch_time_s", 0)
        med = m.get("median_epoch_time_s", 0)
        dur = r.get("duration_s", 0)

        if "B0" in name:
            baseline_val = avg

        delta = ""
        if baseline_val and avg and "B0" not in name:
            pct = (avg - baseline_val) / baseline_val * 100
            delta = f" ({pct:+.1f}%)"

        print(f"{name:<30} {avg:>12.2f} {med:>10.2f} {dur:>12.1f}{delta}", file=sys.stderr)

    print("", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="xCoord single-workload benchmark")
    parser.add_argument("--workload", required=True, choices=["gnn", "llama20b", "faiss"])
    parser.add_argument("--output-dir", "-o", required=True)
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS, choices=SCENARIOS)

    # GNN args
    parser.add_argument("--gnn-nodes", type=int, default=5_000_000)
    parser.add_argument("--gnn-epochs", type=int, default=5)
    parser.add_argument("--gnn-warmup", type=int, default=1)
    parser.add_argument("--gnn-uvm", action="store_true", default=True)
    parser.add_argument("--gnn-no-uvm", action="store_true")

    args = parser.parse_args()
    if args.gnn_no_uvm:
        args.gnn_uvm = False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"xcoord_{args.workload}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}", file=sys.stderr)
    print(f"xCoord Workload Benchmark: {args.workload}", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    print(f"  Scenarios: {args.scenarios}", file=sys.stderr)
    print(f"  Output: {output_dir}", file=sys.stderr)

    results = {}
    for scenario in args.scenarios:
        if args.workload == "gnn":
            result = run_scenario_gnn(scenario, args, output_dir)
        else:
            print(f"  Workload {args.workload} not yet implemented", file=sys.stderr)
            continue

        if result:
            results[scenario] = result

    # Save combined
    combined_path = output_dir / "combined_results.json"
    combined_path.write_text(json.dumps(results, indent=2))

    print_summary(results)
    print(f"\nResults saved to: {output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()

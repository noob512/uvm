#!/usr/bin/env python3
"""Shared utilities for workload atomic scripts.

Import from any atomic script via:
    sys.path.insert(0, str(WORKLOADS_DIR / "scripts"))
    from common import cleanup_gpu, wait_for_server, stop_server, parse_vllm_bench_output
"""
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

WORKLOADS_DIR = Path(__file__).resolve().parent.parent


def cleanup_gpu():
    """Kill stale GPU processes via cleanup_gpu.py."""
    cleanup_script = WORKLOADS_DIR / "cleanup_gpu.py"
    if cleanup_script.exists():
        subprocess.run(
            [sys.executable, str(cleanup_script)],
            capture_output=True,
        )
        time.sleep(2)


def wait_for_server(host="127.0.0.1", port=8000, timeout=300,
                    check_interval=5, process=None):
    """Wait for an HTTP server to become ready (socket connect + /health 200).

    Args:
        host: Server hostname.
        port: Server port.
        timeout: Maximum seconds to wait.
        check_interval: Seconds between checks.
        process: Optional subprocess.Popen; returns False if it exits early.
    """
    start = time.time()
    while time.time() - start < timeout:
        if process and process.poll() is not None:
            return False
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            if sock.connect_ex((host, port)) == 0:
                sock.close()
                try:
                    req = urllib.request.Request(f"http://{host}:{port}/health")
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        if resp.status == 200:
                            return True
                except Exception:
                    pass
            else:
                sock.close()
        except Exception:
            pass
        elapsed = int(time.time() - start)
        print(f"  Waiting for server... ({elapsed}s / {timeout}s)", file=sys.stderr)
        time.sleep(check_interval)
    return False


def stop_server(process):
    """Stop a server subprocess (SIGTERM to process group, then SIGKILL)."""
    if process and process.poll() is None:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=30)
        except Exception:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait(timeout=10)
            except Exception:
                pass


def parse_vllm_bench_output(output):
    """Parse vllm bench serve output into a metrics dict."""
    metrics = {}
    patterns = {
        "successful_requests": r"Successful requests:\s+(\d+)",
        "benchmark_duration_s": r"Benchmark duration \(s\):\s+([\d.]+)",
        "total_input_tokens": r"Total input tokens:\s+(\d+)",
        "total_generated_tokens": r"Total generated tokens:\s+(\d+)",
        "request_throughput_rps": r"Request throughput \(req/s\):\s+([\d.]+)",
        "output_throughput_tok_s": r"Output token throughput \(tok/s\):\s+([\d.]+)",
        "peak_throughput_tok_s": r"Peak output token throughput \(tok/s\):\s+([\d.]+)",
        "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([\d.]+)",
        "median_ttft_ms": r"Median TTFT \(ms\):\s+([\d.]+)",
        "p99_ttft_ms": r"P99 TTFT \(ms\):\s+([\d.]+)",
        "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([\d.]+)",
        "median_tpot_ms": r"Median TPOT \(ms\):\s+([\d.]+)",
        "p99_tpot_ms": r"P99 TPOT \(ms\):\s+([\d.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            val = match.group(1)
            metrics[key] = float(val) if "." in val else int(val)
    return metrics

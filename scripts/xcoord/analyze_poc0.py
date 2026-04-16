#!/usr/bin/env python3
"""Analyze POC-0 results: chunk_trace CSV + vllm bench output.

Usage:
    python3 analyze_poc0.py results/poc0_YYYYMMDD_HHMMSS/
"""
import csv
import re
import sys
from pathlib import Path
from collections import defaultdict


def parse_trace_csv(path: Path) -> dict:
    """Parse chunk_trace CSV and compute hook rates."""
    stats = {
        "ACTIVATE": 0,
        "POPULATE": 0,
        "EVICTION_PREPARE": 0,
    }
    first_ts = None
    last_ts = None
    pid_activations = defaultdict(int)  # per-PID fault (activate) count
    inter_activate_gaps = []  # time between consecutive activates (ns)
    last_activate_ts = None

    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                hook = row.get("hook_type", "")
                if hook in stats:
                    stats[hook] += 1

                ts_str = row.get("time_ms", "0")
                try:
                    ts_ms = int(ts_str)
                except (ValueError, TypeError):
                    continue

                if first_ts is None:
                    first_ts = ts_ms
                last_ts = ts_ms

                if hook == "ACTIVATE":
                    pid = row.get("owner_pid", "")
                    if pid:
                        pid_activations[pid] += 1
                    if last_activate_ts is not None:
                        gap = ts_ms - last_activate_ts
                        if gap > 0:
                            inter_activate_gaps.append(gap)
                    last_activate_ts = ts_ms

    except Exception as e:
        print(f"  Warning: Failed to parse {path}: {e}")
        return {}

    duration_s = (last_ts - first_ts) / 1000.0 if first_ts is not None and last_ts is not None and last_ts > first_ts else 0

    result = {
        "duration_s": round(duration_s, 2),
        "total_hooks": sum(stats.values()),
    }

    for hook, count in stats.items():
        result[f"{hook}_count"] = count
        result[f"{hook}_rate"] = round(count / duration_s, 1) if duration_s > 0 else 0

    # Inter-activate gap statistics
    if inter_activate_gaps:
        sorted_gaps = sorted(inter_activate_gaps)
        n = len(sorted_gaps)
        result["activate_gap_p50_ms"] = sorted_gaps[n // 2]
        result["activate_gap_p95_ms"] = sorted_gaps[int(n * 0.95)]
        result["activate_gap_p99_ms"] = sorted_gaps[int(n * 0.99)]
        result["activate_gap_mean_ms"] = round(sum(sorted_gaps) / n, 2)

    # Per-PID info
    result["num_pids"] = len(pid_activations)
    if pid_activations:
        top_pid = max(pid_activations, key=pid_activations.get)
        result["top_pid"] = top_pid
        result["top_pid_activations"] = pid_activations[top_pid]

    return result


def parse_bench_output(path: Path) -> dict:
    """Parse vllm bench serve output for key metrics."""
    result = {}
    try:
        text = path.read_text()
    except Exception:
        return result

    # Extract key metrics from vllm bench output
    # Format: "Output token throughput (tok/s):         198.67    "
    patterns = {
        "request_throughput": r"Request throughput \(req/s\):\s*([\d.]+)",
        "output_throughput": r"Output token throughput \(tok/s\):\s*([\d.]+)",
        "total_token_throughput": r"Total Token throughput \(tok/s\):\s*([\d.]+)",
        "successful_requests": r"Successful requests:\s*(\d+)",
        "benchmark_duration": r"Benchmark duration \(s\):\s*([\d.]+)",
        "tpot_mean": r"Mean TPOT \(ms\):\s*([\d.]+)",
        "tpot_median": r"Median TPOT \(ms\):\s*([\d.]+)",
        "tpot_p99": r"P99 TPOT \(ms\):\s*([\d.]+)",
        "ttft_mean": r"Mean TTFT \(ms\):\s*([\d.]+)",
        "ttft_median": r"Median TTFT \(ms\):\s*([\d.]+)",
        "ttft_p99": r"P99 TTFT \(ms\):\s*([\d.]+)",
        "itl_mean": r"Mean ITL \(ms\):\s*([\d.]+)",
        "itl_p99": r"P99 ITL \(ms\):\s*([\d.]+)",
    }

    for key, pattern in patterns.items():
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            result[key] = float(m.group(1))

    return result


def print_report(results_dir: Path):
    """Generate analysis report."""
    # Auto-detect scenarios from trace files
    trace_files = sorted(results_dir.glob("trace_*.csv"))
    if not trace_files:
        print("ERROR: No trace files found")
        return

    scenarios = [f.stem.replace("trace_", "") for f in trace_files]
    data = {}

    for scenario in scenarios:
        trace_file = results_dir / f"trace_{scenario}.csv"
        bench_file = results_dir / f"bench_{scenario}.txt"

        entry = {"scenario": scenario}
        if trace_file.exists():
            print(f"  Parsing trace_{scenario}.csv ({trace_file.stat().st_size // 1024 // 1024}MB)...",
                  file=sys.stderr)
            entry["trace"] = parse_trace_csv(trace_file)
        if bench_file.exists():
            entry["bench"] = parse_bench_output(bench_file)
        data[scenario] = entry

    # Print report
    print("=" * 80)
    print("POC-0: CPU-GPU Coupling Analysis Report")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print()

    # === Table 1: Throughput comparison ===
    print("## 1. Throughput Comparison")
    print()
    print(f"{'Scenario':<25} {'tok/s':>10} {'Slowdown':>10} {'TPOT Mean':>12} {'TPOT P99':>12} {'TTFT Mean':>12}")
    print("-" * 80)

    baseline_toks = None
    for scenario in scenarios:
        bench = data.get(scenario, {}).get("bench", {})
        toks = bench.get("output_throughput", 0) or bench.get("total_token_throughput", 0)
        tpot_mean = bench.get("tpot_mean", 0)
        tpot_p99 = bench.get("tpot_p99", 0)
        ttft_mean = bench.get("ttft_mean", 0)

        if scenario == "baseline":
            baseline_toks = toks
            slowdown = "-"
        elif baseline_toks and baseline_toks > 0:
            pct = (1 - toks / baseline_toks) * 100
            slowdown = f"{pct:+.1f}%"
        else:
            slowdown = "N/A"

        toks_str = f"{toks:.1f}" if toks else "N/A"
        tpot_mean_str = f"{tpot_mean:.2f} ms" if tpot_mean else "N/A"
        tpot_p99_str = f"{tpot_p99:.2f} ms" if tpot_p99 else "N/A"
        ttft_mean_str = f"{ttft_mean:.1f} ms" if ttft_mean else "N/A"

        print(f"{scenario:<25} {toks_str:>10} {slowdown:>10} {tpot_mean_str:>12} {tpot_p99_str:>12} {ttft_mean_str:>12}")

    print()

    # === Table 2: GPU Hook Rates ===
    print("## 2. GPU Hook Call Rates (from chunk_trace)")
    print()
    print(f"{'Scenario':<25} {'Duration':>10} {'Activate/s':>12} {'Used/s':>12} {'Evict/s':>12} {'Total Hooks':>12}")
    print("-" * 80)

    for scenario in scenarios:
        trace = data.get(scenario, {}).get("trace", {})
        duration = trace.get("duration_s", 0)
        activate_rate = trace.get("ACTIVATE_rate", 0)
        used_rate = trace.get("POPULATE_rate", 0)
        evict_rate = trace.get("EVICTION_PREPARE_rate", 0)
        total = trace.get("total_hooks", 0)

        print(f"{scenario:<25} {duration:>8.1f}s {activate_rate:>12.1f} {used_rate:>12.1f} {evict_rate:>12.1f} {total:>12}")

    print()

    # === Table 3: Activate (Fault) Inter-Event Gaps ===
    print("## 3. Chunk Activate Inter-Event Gap (proxy for fault latency)")
    print()
    print(f"{'Scenario':<25} {'Count':>10} {'Mean':>10} {'P50':>10} {'P95':>10} {'P99':>10}")
    print("-" * 80)

    for scenario in scenarios:
        trace = data.get(scenario, {}).get("trace", {})
        count = trace.get("ACTIVATE_count", 0)
        mean_ms = trace.get("activate_gap_mean_ms", 0)
        p50 = trace.get("activate_gap_p50_ms", 0)
        p95 = trace.get("activate_gap_p95_ms", 0)
        p99 = trace.get("activate_gap_p99_ms", 0)

        print(f"{scenario:<25} {count:>10} {mean_ms:>8.1f}ms {p50:>8}ms {p95:>8}ms {p99:>8}ms")

    print()

    # === Table 4: Per-PID Info ===
    print("## 4. Per-PID Activation Info")
    print()
    for scenario in scenarios:
        trace = data.get(scenario, {}).get("trace", {})
        num_pids = trace.get("num_pids", 0)
        top_pid = trace.get("top_pid", "N/A")
        top_count = trace.get("top_pid_activations", 0)
        print(f"  {scenario:<25} {num_pids} PIDs, top PID={top_pid} ({top_count} activations)")

    print()

    # === Summary ===
    print("=" * 80)
    print("## Summary")
    print("=" * 80)
    print()

    # Pairwise comparison: find baseline, stress, and pinned scenarios
    bl_name = next((s for s in scenarios if "baseline" in s), None)
    st_name = next((s for s in scenarios if "stress" in s and "pinned" not in s), None)
    pn_name = next((s for s in scenarios if "pinned" in s), None)
    hv_name = next((s for s in scenarios if "heavy" in s), None)

    def get_toks(name):
        if not name:
            return 0
        bench = data.get(name, {}).get("bench", {})
        return bench.get("output_throughput", 0) or bench.get("total_token_throughput", 0)

    def get_ttft(name):
        if not name:
            return 0
        return data.get(name, {}).get("bench", {}).get("ttft_mean", 0)

    bl_toks = get_toks(bl_name)
    st_toks = get_toks(st_name)
    pn_toks = get_toks(pn_name)
    hv_toks = get_toks(hv_name)

    if bl_toks and st_toks:
        stress_drop = (1 - st_toks / bl_toks) * 100
        print(f"1. CPU stress causes {stress_drop:+.1f}% throughput change ({bl_toks:.1f} → {st_toks:.1f} tok/s)")

    if st_toks and pn_toks:
        pin_diff = (pn_toks - st_toks) / st_toks * 100
        direction = "improvement" if pin_diff > 0 else "degradation"
        print(f"2. CPU pinning: {abs(pin_diff):.1f}% {direction} vs unpinned ({pn_toks:.1f} vs {st_toks:.1f} tok/s)")

    if bl_toks and hv_toks:
        heavy_drop = (1 - hv_toks / bl_toks) * 100
        print(f"3. Heavy load causes {heavy_drop:+.1f}% throughput change ({bl_toks:.1f} → {hv_toks:.1f} tok/s)")

    # TTFT comparison (especially important for UVM workloads)
    bl_ttft = get_ttft(bl_name)
    st_ttft = get_ttft(st_name)
    pn_ttft = get_ttft(pn_name)
    if bl_ttft and st_ttft:
        ttft_increase = (st_ttft - bl_ttft) / bl_ttft * 100
        print(f"4. TTFT under CPU stress: {bl_ttft:.1f}ms → {st_ttft:.1f}ms ({ttft_increase:+.1f}%)")
    if st_ttft and pn_ttft:
        print(f"5. TTFT pinned vs unpinned: {pn_ttft:.1f}ms vs {st_ttft:.1f}ms")

    # Trace comparison
    bl_trace = data.get(bl_name, {}).get("trace", {}) if bl_name else {}
    st_trace = data.get(st_name, {}).get("trace", {}) if st_name else {}

    bl_activate = bl_trace.get("ACTIVATE_rate", 0)
    st_activate = st_trace.get("ACTIVATE_rate", 0)
    if bl_activate and st_activate:
        ratio = st_activate / bl_activate
        print(f"6. Activate rate: {bl_activate:.1f}/s → {st_activate:.1f}/s (CPU stress), {ratio:.1f}x")

    bl_evict = bl_trace.get("EVICTION_PREPARE_rate", 0)
    st_evict = st_trace.get("EVICTION_PREPARE_rate", 0)
    if bl_evict and st_evict:
        ratio = st_evict / bl_evict
        print(f"7. Eviction rate: {bl_evict:.1f}/s → {st_evict:.1f}/s (CPU stress), {ratio:.1f}x")

    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.exists():
        print(f"ERROR: {results_dir} does not exist")
        sys.exit(1)

    print_report(results_dir)

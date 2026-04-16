#!/usr/bin/env python3
"""
Analyze remaining overhead in UVM demand paging for 120B MoE model.

Quantifies per-decode-step:
- Migration volume (fault-in + eviction-out)
- PCIe bandwidth utilization
- Fault handling overhead
- Compute vs migration time split
- Theoretical minimum latency

Usage:
  uv run python3 analyze_overhead.py results/msched_trace/chunk_trace_120b_long.csv
"""

import argparse
import csv
import json
import sys
from collections import defaultdict

# Hardware constants
PCIE5_BW_GBS = 63.0  # PCIe 5.0 x16 per direction (GB/s)
CHUNK_SIZE_MB = 2     # UVM chunk size
CHUNK_SIZE_BYTES = CHUNK_SIZE_MB * 1024 * 1024
GPU_VRAM_GB = 32      # RTX 5090
MODEL_SIZE_GB = 59    # 120B MoE model

# Model config
NUM_LAYERS = 36
ANALYTICAL_T1_GB = 2.14   # attention + embeddings
ANALYTICAL_T2_GB = 1.88   # active experts per step (4/128)
ANALYTICAL_T3_GB = 58.33  # inactive experts
ANALYTICAL_IDEAL_WS_GB = ANALYTICAL_T1_GB + ANALYTICAL_T2_GB  # 4.02 GB

# Performance data (from experiments)
PERF_DATA = {
    "baseline_no_bpf": {"pp512": 139.5, "tg128": 45.3, "tg512": 52.2},
    "threshold_1": {"pp512": 217.1, "tg128": 76.0},
    "always_max": {"pp512": 219.1, "tg128": 76.9, "tg512": 85.2},
    "passive_mru": {"pp512": 228.0, "tg128": 78.7, "tg512": 85.1},
    "template_belady_core": {"pp512": 225.0, "tg128": 88.2},
    "always_max_cycle_moe": {"pp512": 229.4, "tg128": 91.3},
}


def parse_events(filepath):
    """Parse chunk_trace CSV into typed events."""
    events = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            va_start_str = row.get('va_start', '').strip()
            try:
                va_start = int(va_start_str, 16) if va_start_str.startswith('0x') else int(va_start_str)
            except (ValueError, AttributeError):
                va_start = 0
            events.append({
                'time_ms': float(row['time_ms']),
                'hook_type': row['hook_type'],
                'va_start': va_start,
                'chunk_addr': row.get('chunk_addr', ''),
            })
    return events


def detect_decode_steps(activate_events, min_step_events=20):
    """Detect decode step boundaries from VA regression in ACTIVATE events."""
    steps = []
    current_step = []
    max_va = 0

    for ev in activate_events:
        va = ev['va_start']
        if va == 0:
            continue
        # Detect VA regression: current VA drops significantly below max
        if current_step and va < max_va * 0.7 and len(current_step) > min_step_events:
            steps.append(current_step)
            current_step = []
            max_va = 0
        current_step.append(ev)
        max_va = max(max_va, va)

    if current_step and len(current_step) > min_step_events:
        steps.append(current_step)

    return steps


def analyze_per_step(events_by_type, steps):
    """Analyze per-decode-step overhead metrics."""
    results = []

    # Skip first step (prefill) and analyze decode steps
    decode_steps = steps[1:] if len(steps) > 1 else steps

    for i, step in enumerate(decode_steps):
        if not step:
            continue
        t_start = step[0]['time_ms']
        t_end = step[-1]['time_ms']
        duration_ms = t_end - t_start if t_end > t_start else 1.0

        # Count unique chunks activated in this step
        chunk_addrs = set(ev['chunk_addr'] for ev in step)
        num_chunks = len(chunk_addrs)
        migration_in_mb = num_chunks * CHUNK_SIZE_MB

        results.append({
            'step': i,
            'duration_ms': duration_ms,
            'num_activations': len(step),
            'unique_chunks': num_chunks,
            'migration_in_mb': migration_in_mb,
        })

    return results


def compute_overhead_breakdown(step_stats):
    """Compute overhead breakdown across all decode steps."""
    if not step_stats:
        return {}

    durations = [s['duration_ms'] for s in step_stats]
    migrations = [s['migration_in_mb'] for s in step_stats]
    chunks = [s['unique_chunks'] for s in step_stats]
    activations = [s['num_activations'] for s in step_stats]

    avg_duration = sum(durations) / len(durations)
    avg_migration_mb = sum(migrations) / len(migrations)
    avg_chunks = sum(chunks) / len(chunks)
    avg_activations = sum(activations) / len(activations)

    # PCIe transfer time (DMA only)
    # Fault-in: migration_in MB at PCIe bandwidth
    # Eviction-out: approximately same volume
    pcie_fault_in_ms = (avg_migration_mb / 1024) / PCIE5_BW_GBS * 1000
    pcie_eviction_out_ms = pcie_fault_in_ms  # approximate: evict similar volume
    pcie_total_ms = pcie_fault_in_ms + pcie_eviction_out_ms

    # UVM uses single Copy Engine (no pipeline), so DMA is sequential
    # With MSched pipelined migration: max(fault_in, eviction_out)
    pcie_pipelined_ms = max(pcie_fault_in_ms, pcie_eviction_out_ms)

    # Fault handling overhead estimate
    # Each GPU page fault: ~5-10 us (interrupt + handler + page table update)
    # With always_max: 1 fault per chunk → num_chunks faults per step
    # Without always_max: many faults per chunk (each 4KB page faults separately)
    fault_overhead_per_fault_us = 7.5  # estimate
    fault_overhead_always_max_ms = avg_chunks * fault_overhead_per_fault_us / 1000
    # Baseline: ~5x more faults (512 pages per 2MB chunk, but some locality)
    fault_overhead_baseline_ms = avg_activations * fault_overhead_per_fault_us / 1000

    return {
        'num_steps': len(step_stats),
        'avg_duration_ms': avg_duration,
        'avg_migration_in_mb': avg_migration_mb,
        'avg_migration_total_mb': avg_migration_mb * 2,  # in + out
        'avg_chunks_per_step': avg_chunks,
        'avg_activations_per_step': avg_activations,
        # PCIe
        'pcie_fault_in_ms': pcie_fault_in_ms,
        'pcie_eviction_out_ms': pcie_eviction_out_ms,
        'pcie_total_sequential_ms': pcie_total_ms,
        'pcie_total_pipelined_ms': pcie_pipelined_ms,
        # Fault overhead
        'fault_overhead_always_max_ms': fault_overhead_always_max_ms,
        'fault_overhead_baseline_ms': fault_overhead_baseline_ms,
    }


def compute_theoretical_limits():
    """Compute theoretical performance limits."""
    results = {}

    # 1. Migration volume per decode step (from analytical model)
    # Each decode step: switch 4 active experts per layer = 4*36 * 13.4 MB = 1.93 GB
    # But only if ALL experts change. In practice, some remain from prev step.
    # Conservative estimate: half change = ~1 GB
    expert_switch_gb = ANALYTICAL_T2_GB  # 1.88 GB worst case

    # 2. PCIe transfer time for migration
    pcie_time_sequential_ms = (expert_switch_gb / PCIE5_BW_GBS) * 1000 * 2  # in+out
    pcie_time_pipelined_ms = (expert_switch_gb / PCIE5_BW_GBS) * 1000  # overlap

    # 3. Ideal decode latency (no oversubscription)
    # 120B MoE on RTX 5090: compute-bound estimate
    # Active params per token: ~4.02 GB → memory bandwidth bound
    # RTX 5090 HBM bandwidth: ~1.79 TB/s (GDDR7 1792 GB/s)
    hbm_bw_gbs = 1792.0  # GDDR7 bandwidth
    active_params_gb = ANALYTICAL_IDEAL_WS_GB
    compute_ms = (active_params_gb / hbm_bw_gbs) * 1000

    # 4. Total theoretical minimum per-token time
    # With perfect prefetch: compute + overlap migration with compute
    # Best case: migration fully hidden behind compute
    # Worst case: compute + migration (sequential)
    theoretical_best_ms = max(compute_ms, pcie_time_pipelined_ms)
    theoretical_worst_ms = compute_ms + pcie_time_sequential_ms

    results['expert_switch_gb'] = expert_switch_gb
    results['pcie_sequential_ms'] = pcie_time_sequential_ms
    results['pcie_pipelined_ms'] = pcie_time_pipelined_ms
    results['compute_ms'] = compute_ms
    results['hbm_bw_gbs'] = hbm_bw_gbs
    results['theoretical_best_tok_per_s'] = 1000 / theoretical_best_ms
    results['theoretical_worst_tok_per_s'] = 1000 / theoretical_worst_ms
    results['theoretical_best_ms_per_tok'] = theoretical_best_ms
    results['theoretical_worst_ms_per_tok'] = theoretical_worst_ms

    return results


def main():
    parser = argparse.ArgumentParser(description='Analyze UVM overhead for 120B MoE')
    parser.add_argument('trace_file', help='chunk_trace CSV file')
    parser.add_argument('--output', '-o', default=None, help='Output JSON file')
    args = parser.parse_args()

    print(f"=== UVM Overhead Analysis for 120B MoE on RTX 5090 ===\n")

    # Parse events
    print(f"Parsing {args.trace_file}...")
    events = parse_events(args.trace_file)
    print(f"  Total events: {len(events):,}")

    # Split by type
    by_type = defaultdict(list)
    for ev in events:
        by_type[ev['hook_type']].append(ev)

    for hook, evts in sorted(by_type.items()):
        print(f"  {hook}: {len(evts):,}")

    # Basic stats
    activate = by_type.get('ACTIVATE', [])
    populate = by_type.get('POPULATE', [])
    eviction = by_type.get('EVICTION_PREPARE', [])

    unique_chunks = set(ev['chunk_addr'] for ev in activate)
    total_time_ms = events[-1]['time_ms'] - events[0]['time_ms'] if events else 0

    print(f"\n  Unique chunks: {len(unique_chunks):,} ({len(unique_chunks) * CHUNK_SIZE_MB / 1024:.1f} GiB)")
    print(f"  Total trace time: {total_time_ms/1000:.1f} s")

    # Re-fault analysis
    chunk_activate_count = defaultdict(int)
    for ev in activate:
        chunk_activate_count[ev['chunk_addr']] += 1

    single_fault = sum(1 for c in chunk_activate_count.values() if c == 1)
    multi_fault = sum(1 for c in chunk_activate_count.values() if c > 1)
    total_refaults = sum(c - 1 for c in chunk_activate_count.values() if c > 1)
    refault_rate = total_refaults / len(activate) * 100 if activate else 0

    print(f"\n--- Re-fault Analysis ---")
    print(f"  Single-fault chunks: {single_fault:,} ({single_fault * CHUNK_SIZE_MB / 1024:.1f} GiB) — T1 candidates")
    print(f"  Multi-fault chunks:  {multi_fault:,} ({multi_fault * CHUNK_SIZE_MB / 1024:.1f} GiB) — thrashing")
    print(f"  Total re-faults:     {total_refaults:,}")
    print(f"  Re-fault rate:       {refault_rate:.1f}%")
    print(f"  Wasted migration:    {total_refaults * CHUNK_SIZE_MB * 2 / 1024:.1f} GiB (each re-fault = evict 2MB + reload 2MB)")

    # Detect decode steps
    print(f"\n--- Decode Step Detection ---")
    steps = detect_decode_steps(activate)
    print(f"  Detected steps: {len(steps)}")
    if steps:
        print(f"  Step 0 (prefill): {len(steps[0]):,} activations, "
              f"{steps[0][-1]['time_ms'] - steps[0][0]['time_ms']:.0f} ms")
        if len(steps) > 1:
            decode_steps = steps[1:]
            print(f"  Decode steps: {len(decode_steps)}")

    # Per-step analysis
    step_stats = analyze_per_step(by_type, steps)

    if step_stats:
        print(f"\n--- Per-Decode-Step Overhead (avg of {len(step_stats)} steps) ---")
        breakdown = compute_overhead_breakdown(step_stats)

        print(f"  Duration:           {breakdown['avg_duration_ms']:.1f} ms/step → {1000/breakdown['avg_duration_ms']:.1f} tok/s (baseline)")
        print(f"  Activations:        {breakdown['avg_activations_per_step']:.0f}/step")
        print(f"  Unique chunks:      {breakdown['avg_chunks_per_step']:.0f}/step")
        print(f"  Migration in:       {breakdown['avg_migration_in_mb']:.0f} MB/step")
        print(f"  Migration total:    {breakdown['avg_migration_total_mb']:.0f} MB/step (in + eviction out)")

        print(f"\n--- PCIe Transfer Time ---")
        print(f"  Fault-in DMA:       {breakdown['pcie_fault_in_ms']:.2f} ms")
        print(f"  Eviction-out DMA:   {breakdown['pcie_eviction_out_ms']:.2f} ms")
        print(f"  Total (sequential): {breakdown['pcie_total_sequential_ms']:.2f} ms (UVM default: single CE)")
        print(f"  Total (pipelined):  {breakdown['pcie_total_pipelined_ms']:.2f} ms (MSched: dual CE)")
        print(f"  PCIe utilization:   {breakdown['avg_migration_total_mb'] / 1024 / (breakdown['avg_duration_ms']/1000) / PCIE5_BW_GBS * 100:.1f}% of {PCIE5_BW_GBS} GB/s")

        print(f"\n--- Fault Handling Overhead (estimated) ---")
        print(f"  Per-fault: ~7.5 us (interrupt + handler + page table)")
        print(f"  With always_max ({breakdown['avg_chunks_per_step']:.0f} faults/step): {breakdown['fault_overhead_always_max_ms']:.2f} ms")
        print(f"  Baseline ({breakdown['avg_activations_per_step']:.0f} activations/step): {breakdown['fault_overhead_baseline_ms']:.2f} ms")
    else:
        breakdown = {}

    # Theoretical limits
    print(f"\n--- Theoretical Performance Limits ---")
    theory = compute_theoretical_limits()
    print(f"  GPU HBM bandwidth:     {theory['hbm_bw_gbs']:.0f} GB/s (GDDR7)")
    print(f"  Active WS per token:   {ANALYTICAL_IDEAL_WS_GB:.2f} GiB (T1+T2)")
    print(f"  Compute time/token:    {theory['compute_ms']:.3f} ms (memory-bandwidth bound)")
    print(f"  Expert switch volume:  {theory['expert_switch_gb']:.2f} GiB/step (worst case)")
    print(f"  PCIe transfer/step:    {theory['pcie_sequential_ms']:.2f} ms (sequential) / {theory['pcie_pipelined_ms']:.2f} ms (pipelined)")
    print(f"  Theoretical best:      {theory['theoretical_best_ms_per_tok']:.2f} ms/tok → {theory['theoretical_best_tok_per_s']:.0f} tok/s")
    print(f"  Theoretical worst:     {theory['theoretical_worst_ms_per_tok']:.2f} ms/tok → {theory['theoretical_worst_tok_per_s']:.0f} tok/s")

    # Gap analysis
    print(f"\n{'='*70}")
    print(f"=== OVERHEAD BREAKDOWN SUMMARY ===")
    print(f"{'='*70}")

    best_tg = 91.3  # always_max + cycle_moe
    best_ms = 1000 / best_tg
    baseline_tg = 45.3
    baseline_ms = 1000 / baseline_tg

    print(f"\n  Current best:        {best_tg:.1f} tok/s ({best_ms:.1f} ms/tok)")
    print(f"  Baseline (no BPF):   {baseline_tg:.1f} tok/s ({baseline_ms:.1f} ms/tok)")
    print(f"  Theoretical ceiling: {theory['theoretical_best_tok_per_s']:.0f} tok/s ({theory['theoretical_best_ms_per_tok']:.2f} ms/tok)")

    if breakdown:
        actual_step_ms = breakdown['avg_duration_ms']
        pcie_ms = breakdown['pcie_total_sequential_ms']
        fault_ms = breakdown['fault_overhead_always_max_ms']
        compute_ms = theory['compute_ms']

        # always_max step time estimate
        best_step_ms = best_ms  # 1000/91.3 = 10.95 ms

        print(f"\n  --- Per-token time decomposition (estimated) ---")
        print(f"  Component                     | Baseline  | Best BPF  | Theoretical")
        print(f"  ------------------------------|-----------|-----------|------------")
        print(f"  GPU compute                   | {compute_ms:6.2f} ms | {compute_ms:6.2f} ms | {compute_ms:6.2f} ms")
        print(f"  PCIe DMA (fault-in+evict-out) | {pcie_ms:6.2f} ms | {pcie_ms:6.2f} ms | {theory['pcie_pipelined_ms']:6.2f} ms")
        print(f"  Fault handling overhead        | {breakdown['fault_overhead_baseline_ms']:6.2f} ms | {fault_ms:6.2f} ms | 0.00 ms")
        print(f"  Scheduling + misc             | {baseline_ms - compute_ms - pcie_ms - breakdown['fault_overhead_baseline_ms']:6.2f} ms | {best_step_ms - compute_ms - pcie_ms - fault_ms:6.2f} ms | 0.00 ms")
        print(f"  ------------------------------|-----------|-----------|------------")
        print(f"  TOTAL                         | {baseline_ms:6.2f} ms | {best_step_ms:6.2f} ms | {theory['theoretical_best_ms_per_tok']:6.2f} ms")

        # Improvement opportunities
        print(f"\n  --- Improvement Opportunities ---")

        saved_by_prefetch = baseline_ms - best_step_ms
        remaining_overhead = best_step_ms - theory['theoretical_best_ms_per_tok']

        print(f"  1. Intra-block prefetch (always_max): saved {saved_by_prefetch:.1f} ms/tok ({saved_by_prefetch/baseline_ms*100:.0f}%) ✅ Done")
        print(f"  2. Remaining overhead:                {remaining_overhead:.1f} ms/tok")

        # Break down remaining overhead
        overhead_pcie = pcie_ms - theory['pcie_pipelined_ms']
        overhead_fault = fault_ms
        overhead_other = remaining_overhead - overhead_pcie - overhead_fault
        if overhead_other < 0:
            overhead_other = 0

        print(f"     a. PCIe sequential vs pipelined:   {overhead_pcie:.2f} ms (needs dual CE → driver change)")
        print(f"     b. Fault handling (interrupt+TLB):  {fault_ms:.2f} ms (needs proactive prefetch → no faults)")
        print(f"     c. Re-fault overhead (82% thrash):  NOT reducible by eviction (capacity-bound)")
        print(f"     d. Misc (scheduling, queuing):      {overhead_other:.2f} ms")
        print(f"\n  Cross-block proactive prefetch potential:")
        print(f"     - Eliminate fault handling overhead: -{fault_ms:.2f} ms")
        print(f"     - Enable pipelined DMA:             -{overhead_pcie:.2f} ms")
        print(f"     - Potential new perf: {1000/(best_step_ms - fault_ms - overhead_pcie):.0f} tok/s")
        print(f"     - But: cannot reduce migration volume (82% thrash is capacity-bound)")

    # Oversubscription analysis
    print(f"\n  --- Oversubscription Impact ---")
    oversub = MODEL_SIZE_GB / GPU_VRAM_GB
    resident_frac = GPU_VRAM_GB / MODEL_SIZE_GB
    print(f"  Model size:          {MODEL_SIZE_GB} GiB")
    print(f"  GPU VRAM:            {GPU_VRAM_GB} GiB")
    print(f"  Oversubscription:    {oversub:.2f}x")
    print(f"  Resident fraction:   {resident_frac*100:.0f}%")
    print(f"  Ideal WS (T1+T2):   {ANALYTICAL_IDEAL_WS_GB:.2f} GiB ({ANALYTICAL_IDEAL_WS_GB/GPU_VRAM_GB*100:.0f}% of VRAM)")
    print(f"  Headroom for cache:  {GPU_VRAM_GB - ANALYTICAL_IDEAL_WS_GB:.1f} GiB for expert caching")
    print(f"  Max cacheable experts: {(GPU_VRAM_GB - ANALYTICAL_IDEAL_WS_GB) / 0.0134:.0f} / {128*36} total")

    # Save results
    if args.output:
        output = {
            'trace_file': args.trace_file,
            'total_events': len(events),
            'unique_chunks': len(unique_chunks),
            'refault_rate': refault_rate,
            'breakdown': breakdown,
            'theoretical': theory,
            'perf_data': PERF_DATA,
        }
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()

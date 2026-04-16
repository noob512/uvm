#!/usr/bin/env python3
"""
VRAM Simulator: Replay chunk_trace to evaluate prefetch × eviction strategy combinations.

Replays ACTIVATE events from chunk_trace CSV as a cache simulation.
Configurable VRAM capacity, eviction policy, and prefetch strategy.

Key questions answered:
1. How does cross-block prefetch interact with different eviction policies?
2. What's the prefetch hit rate and displacement cost?
3. Is cross-block harmful because of eviction policy or access pattern?

Usage:
  python3 simulate_vram.py results/msched_trace/chunk_trace_120b_long.csv
  python3 simulate_vram.py results/msched_trace/chunk_trace_120b_long.csv --capacity 14000
"""
import argparse
import csv
import json
import sys
from collections import OrderedDict, defaultdict

VA_BLOCK_SIZE = 2 * 1024 * 1024  # 2MB
CHUNK_SIZE_MB = 2


class VRAMCache:
    """Simulates GPU VRAM as a chunk-level cache."""

    def __init__(self, capacity, eviction_policy='lru', t1_threshold=3,
                 prefetch_placement='mru'):
        """
        prefetch_placement: where to insert prefetched data in the eviction list
          'mru' = tail (most protected, like real UVM kernel behavior)
          'lru' = head (least protected, proposed optimization)
        """
        self.capacity = capacity
        self.eviction_policy = eviction_policy
        self.t1_threshold = t1_threshold
        self.prefetch_placement = prefetch_placement

        # VRAM: OrderedDict maintains insertion/access order for LRU
        # Key: va_start, Value: access_count
        self.vram = OrderedDict()
        self.access_counts = defaultdict(int)

        # Prefetch tracking
        self.prefetched = set()  # va_starts loaded by prefetch, not yet demand-accessed

        # Global stats
        self.stats = {
            'demand_faults': 0, 'demand_hits': 0,
            'evictions': 0,
            'prefetch_issued': 0, 'prefetch_hits': 0, 'prefetch_wasted': 0,
            'prefetch_already_present': 0,
        }

    def _find_eviction_victim(self):
        """Find chunk to evict according to policy."""
        if self.eviction_policy == 'lru':
            return next(iter(self.vram))

        elif self.eviction_policy == 'fifo':
            return next(iter(self.vram))

        elif self.eviction_policy == 't1_protect':
            # Scan from LRU end, evict first non-T1 chunk
            for va in self.vram:
                if self.access_counts[va] < self.t1_threshold:
                    return va
            # All T1 — fallback to LRU
            return next(iter(self.vram))

        elif self.eviction_policy == 'mru':
            # Evict most recently used (last item)
            return next(reversed(self.vram))

        else:
            raise ValueError(f"Unknown policy: {self.eviction_policy}")

    def _evict_one(self):
        victim = self._find_eviction_victim()
        del self.vram[victim]
        self.stats['evictions'] += 1
        if victim in self.prefetched:
            self.stats['prefetch_wasted'] += 1
            self.prefetched.discard(victim)
        return victim

    def demand_access(self, va_start):
        """Process a demand fault. Returns True if miss (new load)."""
        if va_start in self.vram:
            # Hit
            self.stats['demand_hits'] += 1
            if va_start in self.prefetched:
                self.stats['prefetch_hits'] += 1
                self.prefetched.discard(va_start)
            # Update LRU position (not for FIFO)
            if self.eviction_policy != 'fifo':
                self.vram.move_to_end(va_start)
            self.access_counts[va_start] += 1
            return False

        # Miss
        self.stats['demand_faults'] += 1
        while len(self.vram) >= self.capacity:
            self._evict_one()
        self.vram[va_start] = True
        self.vram.move_to_end(va_start)
        self.access_counts[va_start] += 1
        return True

    def prefetch(self, va_start):
        """Proactively load a chunk. Returns True if actually loaded (miss)."""
        if va_start in self.vram:
            self.stats['prefetch_already_present'] += 1
            return False

        self.stats['prefetch_issued'] += 1
        while len(self.vram) >= self.capacity:
            self._evict_one()
        self.vram[va_start] = True
        if self.prefetch_placement == 'lru':
            self.vram.move_to_end(va_start, last=False)  # LRU end (vulnerable)
        else:
            self.vram.move_to_end(va_start, last=True)   # MRU end (protected, real UVM behavior)
        self.prefetched.add(va_start)
        self.access_counts[va_start] += 1
        return True


def parse_activate_sequence(filepath):
    """Parse ACTIVATE events into (time_ms, va_start) sequence."""
    events = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['hook_type'] != 'ACTIVATE':
                continue
            va_str = row.get('va_start', '').strip()
            if not va_str:
                continue
            try:
                va = int(va_str, 16) if va_str.startswith('0x') else int(va_str)
            except ValueError:
                continue
            if va == 0:
                continue
            events.append((float(row['time_ms']), va))
    return events


def detect_steps(events, min_gap_ms=2.0):
    """Detect decode step boundaries from time gaps."""
    steps = []
    current_step = []
    last_time = 0

    for time_ms, va in events:
        if current_step and time_ms - last_time > min_gap_ms:
            steps.append(current_step)
            current_step = []
        current_step.append((time_ms, va))
        last_time = time_ms

    if current_step:
        steps.append(current_step)
    return steps


def detect_steps_by_va_regression(events, min_step_events=20):
    """Detect steps by VA address regression (same as analyze_overhead.py)."""
    steps = []
    current_step = []
    max_va = 0

    for time_ms, va in events:
        if current_step and va < max_va * 0.7 and len(current_step) > min_step_events:
            steps.append(current_step)
            current_step = []
            max_va = 0
        current_step.append((time_ms, va))
        max_va = max(max_va, va)

    if current_step and len(current_step) > min_step_events:
        steps.append(current_step)
    return steps


def run_simulation(events, capacity, eviction_policy, prefetch_strategy,
                   t1_threshold=3, prefetch_placement='mru'):
    """Run VRAM simulation with given parameters.

    prefetch_strategy:
      'none': no cross-block prefetch
      'adjacent_1': prefetch 1 adjacent block on every fault
      'adjacent_1_dedup': prefetch 1 adjacent, but only on new block entry
      'selective': prefetch adjacent only after 2+ consecutive adjacent faults

    prefetch_placement:
      'mru': insert prefetched data at MRU end (real UVM behavior, protected)
      'lru': insert at LRU end (proposed optimization, vulnerable)
    """
    cache = VRAMCache(capacity, eviction_policy, t1_threshold, prefetch_placement)

    last_block = None
    last_was_adjacent = False
    consecutive_adjacent = 0

    per_step_faults = []  # demand faults per step
    step_events = detect_steps_by_va_regression(events)

    for step_idx, step in enumerate(step_events):
        step_demand_faults = 0
        step_hits = 0

        for _, va in step:
            # Process demand access
            was_miss = cache.demand_access(va)
            if was_miss:
                step_demand_faults += 1
            else:
                step_hits += 1

            # Cross-block prefetch
            if prefetch_strategy == 'none':
                pass

            elif prefetch_strategy == 'adjacent_1':
                # Always prefetch next adjacent block
                cache.prefetch(va + VA_BLOCK_SIZE)

            elif prefetch_strategy == 'adjacent_1_dedup':
                # Only prefetch when entering a new block
                if va != last_block:
                    cache.prefetch(va + VA_BLOCK_SIZE)

            elif prefetch_strategy == 'selective':
                # Prefetch only after 2+ consecutive adjacent accesses
                if last_block is not None:
                    diff = va - last_block
                    if diff == VA_BLOCK_SIZE:
                        consecutive_adjacent += 1
                    else:
                        consecutive_adjacent = 0

                    if consecutive_adjacent >= 2:
                        cache.prefetch(va + VA_BLOCK_SIZE)

            last_block = va

        per_step_faults.append({
            'step': step_idx,
            'demand_faults': step_demand_faults,
            'hits': step_hits,
            'total_events': len(step),
        })

    return cache.stats, per_step_faults


def load_layer_boundaries(json_path):
    """Load layer VA boundaries from JSON."""
    with open(json_path) as f:
        data = json.load(f)
    return [int(va, 16) for va in data['boundary_vas']]


def main():
    parser = argparse.ArgumentParser(description='VRAM Simulator for cross-block prefetch analysis')
    parser.add_argument('trace_file', help='chunk_trace CSV file')
    parser.add_argument('--capacity', type=int, default=15000,
                        help='VRAM capacity in chunks (default: 15000 = ~30GB)')
    parser.add_argument('--layer-mapping',
                        default='results/msched_trace/layer_va_ranges_equal_count.json',
                        help='Layer VA mapping JSON')
    args = parser.parse_args()

    print(f"=== VRAM Simulator ===")
    print(f"Trace: {args.trace_file}")
    print(f"Capacity: {args.capacity} chunks ({args.capacity * CHUNK_SIZE_MB / 1024:.1f} GB)")

    # Parse events
    print(f"\nParsing trace...")
    events = parse_activate_sequence(args.trace_file)
    print(f"  ACTIVATE events: {len(events):,}")

    unique_blocks = len(set(va for _, va in events))
    print(f"  Unique VA blocks: {unique_blocks:,} ({unique_blocks * CHUNK_SIZE_MB / 1024:.1f} GB)")

    # Detect steps
    steps = detect_steps_by_va_regression(events)
    print(f"  Detected steps: {len(steps)}")
    if len(steps) > 1:
        print(f"  Step 0 (prefill): {len(steps[0])} events")
        decode_sizes = [len(s) for s in steps[1:]]
        print(f"  Decode steps: {len(decode_sizes)}, avg {sum(decode_sizes)/len(decode_sizes):.0f} events/step")

    # Run simulation matrix
    eviction_policies = ['lru', 't1_protect', 'fifo']
    prefetch_strategies = ['none', 'adjacent_1_dedup', 'selective']
    placements = ['mru', 'lru']  # MRU = real UVM behavior, LRU = proposed optimization

    print(f"\n{'='*110}")
    print(f"{'Policy':>14} × {'Prefetch':<20} {'Place':>5} | {'Faults':>8} {'Hits':>8} {'Evict':>8} | {'PF-Issued':>10} {'PF-Hit':>7} {'PF-Waste':>9} | {'Net DMA Δ':>10}")
    print(f"{'='*110}")

    baseline_faults = {}  # eviction_policy -> baseline faults

    for eviction in eviction_policies:
        for prefetch in prefetch_strategies:
            for placement in placements:
                if prefetch == 'none' and placement == 'lru':
                    continue  # no prefetch = placement irrelevant

                stats, per_step = run_simulation(
                    events, args.capacity, eviction, prefetch,
                    prefetch_placement=placement
                )

                if prefetch == 'none':
                    baseline_faults[eviction] = stats['demand_faults']

                # Net DMA change = (demand_faults + prefetch_issued) - baseline_demand_faults
                base = baseline_faults.get(eviction, stats['demand_faults'])
                total_dma = stats['demand_faults'] + stats['prefetch_issued']
                net_change = total_dma - base
                net_mb = net_change * CHUNK_SIZE_MB

                place_str = placement if prefetch != 'none' else '-'
                print(f"{eviction:>14} × {prefetch:<20} {place_str:>5} | "
                      f"{stats['demand_faults']:>8,} {stats['demand_hits']:>8,} {stats['evictions']:>8,} | "
                      f"{stats['prefetch_issued']:>10,} {stats['prefetch_hits']:>7,} {stats['prefetch_wasted']:>9,} | "
                      f"{'+' if net_mb >= 0 else ''}{net_mb:>9,} MB")

        print(f"{'-'*110}")

    # Detailed per-step analysis for key configurations
    print(f"\n{'='*70}")
    print(f"=== Per-Step Analysis (decode only, skip step 0) ===")
    print(f"{'='*70}")

    key_configs = [
        ('lru', 'none', 'mru', 'Baseline (LRU, no prefetch)'),
        ('t1_protect', 'none', 'mru', 'T1-protect, no prefetch'),
        ('t1_protect', 'adjacent_1_dedup', 'mru', 'T1-protect + cross-block (MRU insert, real UVM)'),
        ('t1_protect', 'adjacent_1_dedup', 'lru', 'T1-protect + cross-block (LRU insert, optimized)'),
        ('fifo', 'adjacent_1_dedup', 'mru', 'FIFO + cross-block (MRU insert)'),
        ('fifo', 'adjacent_1_dedup', 'lru', 'FIFO + cross-block (LRU insert)'),
    ]

    for eviction, prefetch, placement, label in key_configs:
        stats, per_step = run_simulation(events, args.capacity, eviction, prefetch,
                                         prefetch_placement=placement)

        # Skip step 0 (prefill)
        decode_steps = per_step[1:] if len(per_step) > 1 else per_step
        if not decode_steps:
            continue

        avg_faults = sum(s['demand_faults'] for s in decode_steps) / len(decode_steps)
        avg_hits = sum(s['hits'] for s in decode_steps) / len(decode_steps)
        total_dma = stats['demand_faults'] + stats['prefetch_issued']

        print(f"\n  {label}:")
        print(f"    Avg demand faults/step: {avg_faults:.1f}")
        print(f"    Avg hits/step:          {avg_hits:.1f}")
        print(f"    Hit rate:               {stats['demand_hits']/(stats['demand_hits']+stats['demand_faults'])*100:.1f}%")
        print(f"    Prefetch hits:          {stats['prefetch_hits']:,} / {stats['prefetch_issued']:,} "
              f"({stats['prefetch_hits']/max(1,stats['prefetch_issued'])*100:.1f}%)")
        print(f"    Prefetch wasted:        {stats['prefetch_wasted']:,} "
              f"({stats['prefetch_wasted']/max(1,stats['prefetch_issued'])*100:.1f}%)")
        print(f"    Total DMA chunks:       {total_dma:,} (demand + prefetch)")
        print(f"    Total DMA volume:       {total_dma * CHUNK_SIZE_MB / 1024:.1f} GB")

    # Summary table
    print(f"\n{'='*70}")
    print(f"=== Summary: Cross-Block Prefetch Impact by Eviction Policy ===")
    print(f"{'='*70}")
    print(f"\nQuestion: Does eviction policy change whether cross-block helps or hurts?")
    print(f"Metric: demand_faults (lower = better, means prefetch eliminated faults)")
    print(f"         But also check total_dma = demand_faults + prefetch_issued")
    print()

    for eviction in eviction_policies:
        stats_none, _ = run_simulation(events, args.capacity, eviction, 'none')
        faults_none = stats_none['demand_faults']

        for placement in ['mru', 'lru']:
            stats_xb, _ = run_simulation(events, args.capacity, eviction, 'adjacent_1_dedup',
                                          prefetch_placement=placement)

            faults_xb = stats_xb['demand_faults']
            faults_saved = faults_none - faults_xb
            pf_issued = stats_xb['prefetch_issued']
            pf_hit = stats_xb['prefetch_hits']
            pf_waste = stats_xb['prefetch_wasted']

            net_dma_change = pf_issued - faults_saved

            print(f"  {eviction:>14} + cross-block ({placement} insert):")
            print(f"    Demand faults: {faults_none:,} → {faults_xb:,} (saved {faults_saved:,})")
            print(f"    Prefetch: {pf_issued:,} issued, {pf_hit:,} hit ({pf_hit/max(1,pf_issued)*100:.1f}%), "
                  f"{pf_waste:,} wasted ({pf_waste/max(1,pf_issued)*100:.1f}%)")
            print(f"    Net DMA change: {'+' if net_dma_change > 0 else ''}{net_dma_change:,} chunks "
                  f"({'+' if net_dma_change > 0 else ''}{net_dma_change * CHUNK_SIZE_MB:,} MB)")
            print(f"    → {'HARMFUL' if net_dma_change > 0 else 'BENEFICIAL'}: "
                  f"{'more' if net_dma_change > 0 else 'less'} total DMA")
            print()


if __name__ == '__main__':
    main()

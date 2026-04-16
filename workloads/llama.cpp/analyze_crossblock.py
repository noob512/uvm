#!/usr/bin/env python3
"""Analyze cross-block prefetch hit rate from chunk trace data.

For each POPULATE (fault) event, check if the next POPULATE is in an adjacent VA block.
This simulates what the cross-block prefetch would have prefetched and whether it would
have been useful.
"""
import csv
import sys
from collections import defaultdict

VA_BLOCK_SIZE = 2 * 1024 * 1024  # 2MB

def analyze(trace_file):
    # Read all POPULATE events (actual faults)
    populates = []
    with open(trace_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['hook_type'] == 'POPULATE':
                populates.append({
                    'time_ms': float(row['time_ms']),
                    'va_start': int(row['va_start'], 16),
                    'va_end': int(row['va_end'], 16),
                })

    print(f"Total POPULATE events: {len(populates)}")

    # Analyze: for each fault, would prefetching the next adjacent block help?
    adjacent_hit = 0
    adjacent_miss = 0
    same_block = 0
    total = len(populates) - 1

    # Track unique blocks accessed
    block_sequence = []
    seen_blocks = set()

    for i in range(len(populates)):
        block_start = populates[i]['va_start']
        if block_start not in seen_blocks or (block_sequence and block_sequence[-1] != block_start):
            block_sequence.append(block_start)
        seen_blocks.add(block_start)

    print(f"Unique VA blocks accessed: {len(seen_blocks)}")
    print(f"Block access sequence length: {len(block_sequence)}")

    # For each block transition, check if next block is adjacent
    transitions = 0
    adjacent_transitions = 0
    backward_transitions = 0
    forward_transitions = 0
    far_transitions = 0

    for i in range(len(block_sequence) - 1):
        cur = block_sequence[i]
        nxt = block_sequence[i + 1]
        if cur == nxt:
            continue
        transitions += 1
        diff = nxt - cur
        if diff == VA_BLOCK_SIZE:
            adjacent_transitions += 1
            forward_transitions += 1
        elif diff == -VA_BLOCK_SIZE:
            adjacent_transitions += 1
            backward_transitions += 1
        elif abs(diff) <= 4 * VA_BLOCK_SIZE:
            far_transitions += 1  # within 4 blocks
        # else: far jump

    print(f"\n=== Block Transition Analysis ===")
    print(f"Total transitions: {transitions}")
    print(f"Adjacent (+1 block): {forward_transitions} ({100*forward_transitions/transitions:.1f}%)")
    print(f"Adjacent (-1 block): {backward_transitions} ({100*backward_transitions/transitions:.1f}%)")
    print(f"Near (±2-4 blocks): {far_transitions} ({100*far_transitions/transitions:.1f}%)")
    far_jumps = transitions - adjacent_transitions - far_transitions
    print(f"Far jumps: {far_jumps} ({100*far_jumps/transitions:.1f}%)")

    # Analyze per-phase: prefill vs decode
    # Prefill is the initial burst, decode is steady-state
    # Find the boundary: large time gap or change in access pattern
    print(f"\n=== Phase Analysis ===")

    # Simple heuristic: first N events where time < some threshold
    # Or: find the point where block reuse starts
    first_reuse_idx = None
    block_first_seen = {}
    for i, block in enumerate(block_sequence):
        if block in block_first_seen and i - block_first_seen[block] > 10:
            first_reuse_idx = i
            break
        if block not in block_first_seen:
            block_first_seen[block] = i

    if first_reuse_idx:
        print(f"First block reuse at sequence index {first_reuse_idx} (≈ end of prefill)")

        # Prefill transitions
        prefill_adj = 0
        prefill_total = 0
        for i in range(min(first_reuse_idx, len(block_sequence) - 1)):
            cur = block_sequence[i]
            nxt = block_sequence[i + 1]
            if cur != nxt:
                prefill_total += 1
                if abs(nxt - cur) == VA_BLOCK_SIZE:
                    prefill_adj += 1

        print(f"Prefill: {prefill_adj}/{prefill_total} adjacent ({100*prefill_adj/max(1,prefill_total):.1f}%)")

        # Decode transitions
        decode_adj = 0
        decode_total = 0
        for i in range(first_reuse_idx, len(block_sequence) - 1):
            cur = block_sequence[i]
            nxt = block_sequence[i + 1]
            if cur != nxt:
                decode_total += 1
                if abs(nxt - cur) == VA_BLOCK_SIZE:
                    decode_adj += 1

        print(f"Decode: {decode_adj}/{decode_total} adjacent ({100*decode_adj/max(1,decode_total):.1f}%)")

    # Distribution of jump distances
    print(f"\n=== Jump Distance Distribution (top 10) ===")
    jump_dist = defaultdict(int)
    for i in range(len(block_sequence) - 1):
        cur = block_sequence[i]
        nxt = block_sequence[i + 1]
        if cur != nxt:
            blocks_jumped = (nxt - cur) // VA_BLOCK_SIZE
            jump_dist[blocks_jumped] += 1

    for dist, count in sorted(jump_dist.items(), key=lambda x: -x[1])[:10]:
        pct = 100 * count / transitions
        print(f"  {dist:+6d} blocks: {count:5d} ({pct:5.1f}%)")

if __name__ == '__main__':
    trace_file = sys.argv[1] if len(sys.argv) > 1 else \
        '/home/yunwei37/workspace/gpu/gpu_ext/workloads/llama.cpp/results/msched_trace/chunk_trace_120b_raw.csv'
    analyze(trace_file)

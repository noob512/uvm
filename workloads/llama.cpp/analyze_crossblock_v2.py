#!/usr/bin/env python3
"""Deeper analysis of MoE block access patterns for smarter prefetch strategies."""
import csv
import sys
from collections import defaultdict

VA_BLOCK_SIZE = 2 * 1024 * 1024  # 2MB

def analyze(trace_file):
    populates = []
    with open(trace_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['hook_type'] == 'POPULATE':
                populates.append({
                    'time_ms': float(row['time_ms']),
                    'va_start': int(row['va_start'], 16),
                })

    # Build block sequence (deduplicated consecutive)
    block_seq = []
    for p in populates:
        b = p['va_start']
        if not block_seq or block_seq[-1] != b:
            block_seq.append(b)

    print(f"Block sequence length: {len(block_seq)}")

    # Skip prefill (first ~32 transitions), focus on decode
    DECODE_START = 100  # safe margin
    decode_seq = block_seq[DECODE_START:]
    print(f"Decode sequence length: {len(decode_seq)}")

    # 1. Adjacent hit rate at different lookahead distances
    print(f"\n=== Prefetch Hit Rate by Lookahead ===")
    print(f"(If we prefetch N adjacent blocks ahead, what % of next accesses are covered?)")
    for ahead in [1, 2, 3, 5, 10]:
        hits = 0
        total = 0
        for i in range(len(decode_seq) - 1):
            cur = decode_seq[i]
            nxt = decode_seq[i + 1]
            if cur == nxt:
                continue
            total += 1
            # Check if nxt is within 'ahead' adjacent blocks
            for a in range(1, ahead + 1):
                if nxt == cur + a * VA_BLOCK_SIZE:
                    hits += 1
                    break
        print(f"  Ahead {ahead:2d} blocks: {hits}/{total} = {100*hits/total:.1f}%")

    # 2. Find repeating patterns in block transitions (cycle detection)
    print(f"\n=== Cycle Detection ===")
    # Look for repeating sequences of block jumps
    jump_seq = []
    for i in range(len(decode_seq) - 1):
        if decode_seq[i] != decode_seq[i+1]:
            jump_seq.append(decode_seq[i+1] - decode_seq[i])

    # Check: is there a cycle of length N?
    for cycle_len in [36, 72, 100, 200, 500]:
        if len(jump_seq) < 2 * cycle_len:
            continue
        # Compare first cycle with second cycle
        matches = 0
        for i in range(cycle_len):
            if jump_seq[i] == jump_seq[i + cycle_len]:
                matches += 1
        print(f"  Cycle len {cycle_len}: {matches}/{cycle_len} match ({100*matches/cycle_len:.1f}%)")

    # 3. History-based prediction: can we predict next block from last N transitions?
    print(f"\n=== History-Based Prediction ===")
    for history_len in [1, 2, 3]:
        # Build transition table: (last N jumps) -> next jump
        table = defaultdict(lambda: defaultdict(int))
        for i in range(history_len, len(jump_seq)):
            key = tuple(jump_seq[i-history_len:i])
            table[key][jump_seq[i]] += 1

        # Evaluate prediction accuracy
        correct = 0
        total = 0
        for i in range(history_len, len(jump_seq) - 1):
            key = tuple(jump_seq[i-history_len:i])
            if key in table:
                predicted = max(table[key], key=table[key].get)
                actual = jump_seq[i]
                if predicted == actual:
                    correct += 1
            total += 1

        print(f"  History {history_len}: {correct}/{total} = {100*correct/total:.1f}% accuracy")
        print(f"    Unique patterns: {len(table)}")

    # 4. Consecutive adjacent runs
    print(f"\n=== Consecutive Adjacent Block Runs ===")
    run_lengths = []
    current_run = 0
    for i in range(len(decode_seq) - 1):
        diff = decode_seq[i+1] - decode_seq[i]
        if diff == VA_BLOCK_SIZE:
            current_run += 1
        else:
            if current_run > 0:
                run_lengths.append(current_run)
            current_run = 0
    if current_run > 0:
        run_lengths.append(current_run)

    if run_lengths:
        print(f"  Total runs: {len(run_lengths)}")
        print(f"  Mean run length: {sum(run_lengths)/len(run_lengths):.1f}")
        print(f"  Max run length: {max(run_lengths)}")
        dist = defaultdict(int)
        for r in run_lengths:
            dist[min(r, 10)] += 1
        for l in sorted(dist):
            print(f"    Length {l if l < 10 else '10+'}: {dist[l]} runs")

    # 5. VA region clustering — group blocks into contiguous ranges
    print(f"\n=== VA Region Analysis ===")
    unique_blocks = sorted(set(decode_seq))
    print(f"  Unique decode blocks: {len(unique_blocks)}")

    # Find contiguous regions
    regions = []
    if unique_blocks:
        region_start = unique_blocks[0]
        region_end = unique_blocks[0]
        for b in unique_blocks[1:]:
            if b == region_end + VA_BLOCK_SIZE:
                region_end = b
            else:
                regions.append((region_start, region_end))
                region_start = b
                region_end = b
        regions.append((region_start, region_end))

    print(f"  Contiguous VA regions: {len(regions)}")
    for i, (s, e) in enumerate(regions[:10]):
        nblocks = (e - s) // VA_BLOCK_SIZE + 1
        print(f"    Region {i}: 0x{s:x} - 0x{e+VA_BLOCK_SIZE-1:x} ({nblocks} blocks, {nblocks*2} MB)")

    # 6. 关键分析：cross-block prefetch 的 VRAM 净影响
    print(f"\n=== Cross-Block Prefetch VRAM Impact Simulation ===")
    # Simulate: for each new block entry, we prefetch next block
    # Track: how many prefetched blocks are actually used before evicted
    prefetched = set()
    used_before_evict = 0
    wasted = 0
    total_prefetches = 0
    seen = set()

    for i in range(len(decode_seq) - 1):
        cur = decode_seq[i]
        nxt = decode_seq[i+1]
        seen.add(cur)

        if cur not in seen or i == 0:  # new block
            # Prefetch next adjacent
            prefetch_target = cur + VA_BLOCK_SIZE
            total_prefetches += 1
            prefetched.add(prefetch_target)

        if nxt in prefetched:
            used_before_evict += 1
            prefetched.discard(nxt)

    print(f"  Total prefetches issued: {total_prefetches}")
    print(f"  Prefetches used: {used_before_evict} ({100*used_before_evict/max(1,total_prefetches):.1f}%)")
    wasted = total_prefetches - used_before_evict
    print(f"  Prefetches wasted: {wasted} ({100*wasted/max(1,total_prefetches):.1f}%)")

if __name__ == '__main__':
    trace_file = sys.argv[1] if len(sys.argv) > 1 else \
        '/home/yunwei37/workspace/gpu/gpu_ext/workloads/llama.cpp/results/msched_trace/chunk_trace_120b_raw.csv'
    analyze(trace_file)

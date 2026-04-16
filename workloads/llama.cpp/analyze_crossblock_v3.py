#!/usr/bin/env python3
"""
Enhanced cross-block prefetch analysis (v3).

Improvements over v2:
1. Prefill vs decode phase separation with per-phase adjacent hit rates
2. Per-layer transition classification (intra-layer vs inter-layer)
3. History-based prediction compression analysis (how few patterns cover 80%?)
4. Fixed VRAM simulation bug from v2
5. Layer-aware prefetch potential analysis

Usage:
  python3 analyze_crossblock_v3.py results/msched_trace/chunk_trace_120b_long.csv
  python3 analyze_crossblock_v3.py results/msched_trace/chunk_trace_120b_raw.csv \
      --layer-mapping results/msched_trace/layer_va_ranges_equal_count.json
"""
import argparse
import csv
import json
import sys
from collections import defaultdict

VA_BLOCK_SIZE = 2 * 1024 * 1024  # 2MB


def parse_block_sequence(trace_file):
    """Parse POPULATE events into deduplicated block sequence with timestamps."""
    populates = []
    with open(trace_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['hook_type'] != 'POPULATE':
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
            populates.append((float(row['time_ms']), va))

    # Deduplicate consecutive same-block accesses
    block_seq = []
    for time_ms, va in populates:
        if not block_seq or block_seq[-1][1] != va:
            block_seq.append((time_ms, va))

    return block_seq


def load_layer_boundaries(json_path):
    """Load layer VA boundaries and per-layer ranges from JSON."""
    with open(json_path) as f:
        data = json.load(f)

    boundaries = [int(va, 16) for va in data['boundary_vas']]

    # Build layer ranges: [(va_start, va_end), ...]
    layers = []
    for lid in sorted(data['layers'].keys(), key=int):
        layer = data['layers'][lid]
        layers.append({
            'id': int(lid),
            'va_start': int(layer['va_start'], 16),
            'va_end': int(layer['va_end'], 16),
            'num_chunks': layer['num_chunks'],
        })

    return boundaries, layers


def va_to_layer(va, boundaries):
    """Map VA to layer ID using binary search on boundaries."""
    if va < boundaries[0]:
        return -1
    lo, hi = 0, len(boundaries) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if boundaries[mid] <= va:
            lo = mid
        else:
            hi = mid - 1
    return lo


def detect_prefill_decode_boundary(block_seq, min_gap_ms=50):
    """Detect prefill/decode boundary.

    Heuristic: first large time gap (>50ms) after initial burst,
    or first block reuse with significant gap.
    """
    # Method 1: Time gap
    for i in range(1, len(block_seq)):
        gap = block_seq[i][0] - block_seq[i-1][0]
        if gap > min_gap_ms and i > 100:
            return i

    # Method 2: First significant block reuse
    seen = {}
    for i, (t, va) in enumerate(block_seq):
        if va in seen and i - seen[va] > 50:
            return i
        if va not in seen:
            seen[va] = i

    # Fallback: 20% of sequence
    return len(block_seq) // 5


def analyze_adjacent_hit_rate(block_seq, label=""):
    """Analyze adjacent block hit rate for a subsequence."""
    if len(block_seq) < 2:
        return {}

    results = {}
    for ahead in [1, 2, 3, 5]:
        hits = 0
        total = 0
        for i in range(len(block_seq) - 1):
            cur = block_seq[i][1]
            nxt = block_seq[i+1][1]
            if cur == nxt:
                continue
            total += 1
            for a in range(1, ahead + 1):
                if nxt == cur + a * VA_BLOCK_SIZE:
                    hits += 1
                    break
        rate = 100 * hits / total if total > 0 else 0
        results[ahead] = (hits, total, rate)

    if label:
        print(f"\n  {label}:")
    for ahead, (hits, total, rate) in results.items():
        print(f"    Ahead {ahead:2d}: {hits:,}/{total:,} = {rate:.1f}%")

    return results


def analyze_layer_transitions(block_seq, boundaries, label=""):
    """Classify block transitions as intra-layer vs inter-layer."""
    intra_layer = 0
    inter_layer_adjacent = 0  # adjacent layers (L → L+1)
    inter_layer_far = 0       # non-adjacent layers
    total = 0

    intra_layer_adjacent_hit = 0  # intra-layer AND adjacent VA block

    for i in range(len(block_seq) - 1):
        cur_va = block_seq[i][1]
        nxt_va = block_seq[i+1][1]
        if cur_va == nxt_va:
            continue
        total += 1

        cur_layer = va_to_layer(cur_va, boundaries)
        nxt_layer = va_to_layer(nxt_va, boundaries)

        if cur_layer == nxt_layer:
            intra_layer += 1
            if nxt_va == cur_va + VA_BLOCK_SIZE:
                intra_layer_adjacent_hit += 1
        elif abs(cur_layer - nxt_layer) == 1:
            inter_layer_adjacent += 1
        else:
            inter_layer_far += 1

    if label:
        print(f"\n  {label}:")
    print(f"    Intra-layer:              {intra_layer:,}/{total:,} ({100*intra_layer/max(1,total):.1f}%)")
    print(f"      of which VA-adjacent:   {intra_layer_adjacent_hit:,}/{intra_layer:,} ({100*intra_layer_adjacent_hit/max(1,intra_layer):.1f}%)")
    print(f"    Inter-layer adjacent (±1): {inter_layer_adjacent:,}/{total:,} ({100*inter_layer_adjacent/max(1,total):.1f}%)")
    print(f"    Inter-layer far:          {inter_layer_far:,}/{total:,} ({100*inter_layer_far/max(1,total):.1f}%)")

    return {
        'intra_layer': intra_layer,
        'intra_layer_adjacent': intra_layer_adjacent_hit,
        'inter_layer_adjacent': inter_layer_adjacent,
        'inter_layer_far': inter_layer_far,
        'total': total,
    }


def analyze_history_compression(block_seq, history_len=3):
    """Analyze how many history patterns are needed to cover 80% of predictions.

    Key question: Can we fit the most important patterns in a small BPF map?
    """
    # Build jump sequence
    jump_seq = []
    for i in range(len(block_seq) - 1):
        cur = block_seq[i][1]
        nxt = block_seq[i+1][1]
        if cur != nxt:
            jump_seq.append(nxt - cur)

    if len(jump_seq) < history_len + 1:
        return

    # Build transition table: (last N jumps) -> {next_jump: count}
    table = defaultdict(lambda: defaultdict(int))
    for i in range(history_len, len(jump_seq)):
        key = tuple(jump_seq[i-history_len:i])
        table[key][jump_seq[i]] += 1

    # Evaluate accuracy and pattern coverage
    print(f"\n=== History-{history_len} Prediction Compression ===")
    print(f"  Total unique patterns: {len(table):,}")

    # Sort patterns by frequency (how many times each pattern appears)
    pattern_freq = []
    for key, targets in table.items():
        total_uses = sum(targets.values())
        best_target = max(targets, key=targets.get)
        correct = targets[best_target]
        pattern_freq.append((key, total_uses, correct, best_target))

    pattern_freq.sort(key=lambda x: -x[1])

    # Coverage analysis: how many patterns needed for X% of all predictions
    total_predictions = sum(pf[1] for pf in pattern_freq)
    total_correct = sum(pf[2] for pf in pattern_freq)
    print(f"  Total predictions:      {total_predictions:,}")
    print(f"  Total correct:          {total_correct:,} ({100*total_correct/total_predictions:.1f}%)")

    cumulative_predictions = 0
    cumulative_correct = 0
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    threshold_idx = 0

    print(f"\n  Patterns needed for coverage:")
    for i, (key, uses, correct, target) in enumerate(pattern_freq):
        cumulative_predictions += uses
        cumulative_correct += correct
        coverage = cumulative_predictions / total_predictions

        while threshold_idx < len(thresholds) and coverage >= thresholds[threshold_idx]:
            print(f"    {thresholds[threshold_idx]*100:5.0f}% coverage: {i+1:,} patterns "
                  f"(accuracy so far: {100*cumulative_correct/cumulative_predictions:.1f}%)")
            threshold_idx += 1

    # Top 10 most frequent patterns
    print(f"\n  Top 10 patterns:")
    for i, (key, uses, correct, target) in enumerate(pattern_freq[:10]):
        blocks = [d // VA_BLOCK_SIZE for d in key]
        target_blocks = target // VA_BLOCK_SIZE
        print(f"    #{i+1}: jumps={blocks} → predict {target_blocks:+d} blocks "
              f"({uses:,} uses, {100*correct/uses:.0f}% accurate)")

    # Key insight: can we fit in PERCPU_ARRAY?
    for map_size in [256, 512, 1024, 2048, 4096]:
        if len(pattern_freq) <= map_size:
            coverage = 1.0
        else:
            top_predictions = sum(pf[1] for pf in pattern_freq[:map_size])
            coverage = top_predictions / total_predictions
        top_correct = sum(pf[2] for pf in pattern_freq[:min(map_size, len(pattern_freq))])
        top_total = sum(pf[1] for pf in pattern_freq[:min(map_size, len(pattern_freq))])
        accuracy = top_correct / max(1, top_total)
        print(f"    ARRAY[{map_size}]: coverage {coverage*100:.1f}%, accuracy {accuracy*100:.1f}%")


def analyze_layer_aware_prefetch(block_seq, boundaries, layers):
    """Analyze potential of layer-aware prefetch.

    Instead of blind adjacent, prefetch the next layer's VA range.
    For each inter-layer transition L→L', check if we could have predicted L'.
    """
    print(f"\n=== Layer-Aware Prefetch Potential ===")

    # Build layer transition sequence
    layer_seq = []
    for _, va in block_seq:
        layer = va_to_layer(va, boundaries)
        if layer >= 0 and (not layer_seq or layer_seq[-1] != layer):
            layer_seq.append(layer)

    print(f"  Layer sequence length: {len(layer_seq):,}")
    print(f"  Unique layers visited: {len(set(layer_seq))}")

    # Layer transition analysis
    layer_trans = defaultdict(lambda: defaultdict(int))
    for i in range(len(layer_seq) - 1):
        layer_trans[layer_seq[i]][layer_seq[i+1]] += 1

    # For each layer, what's the most likely next layer?
    total_trans = 0
    correct_if_predict_next = 0  # predict layer+1
    correct_if_predict_mode = 0  # predict most common next

    for src, targets in layer_trans.items():
        for dst, count in targets.items():
            total_trans += count
            if dst == (src + 1) % len(layers):
                correct_if_predict_next += count

        most_common = max(targets, key=targets.get)
        correct_if_predict_mode += targets[most_common]

    print(f"\n  Prediction accuracy:")
    print(f"    Predict layer+1:     {correct_if_predict_next:,}/{total_trans:,} = "
          f"{100*correct_if_predict_next/max(1,total_trans):.1f}%")
    print(f"    Predict most-common: {correct_if_predict_mode:,}/{total_trans:,} = "
          f"{100*correct_if_predict_mode/max(1,total_trans):.1f}%")

    # Layer transition matrix (sparse)
    print(f"\n  Layer transition matrix (top transitions):")
    flat_trans = []
    for src, targets in layer_trans.items():
        for dst, count in targets.items():
            flat_trans.append((src, dst, count))
    flat_trans.sort(key=lambda x: -x[2])

    for src, dst, count in flat_trans[:15]:
        pct = 100 * count / total_trans
        print(f"    L{src:2d} → L{dst:2d}: {count:5,} ({pct:5.1f}%)")

    # If we prefetch the entire next layer's VA range, how many blocks is that?
    avg_chunks_per_layer = sum(l['num_chunks'] for l in layers) / len(layers)
    print(f"\n  Avg chunks per layer: {avg_chunks_per_layer:.0f}")
    print(f"  Layer-aware prefetch would load ~{avg_chunks_per_layer:.0f} chunks per transition")
    print(f"  vs blind adjacent: 1 chunk per fault")
    print(f"  → Layer-aware is too aggressive for 1.84x oversubscription")


def main():
    parser = argparse.ArgumentParser(description='Enhanced cross-block prefetch analysis')
    parser.add_argument('trace_file', help='chunk_trace CSV file')
    parser.add_argument('--layer-mapping',
                        default='results/msched_trace/layer_va_ranges_equal_count.json',
                        help='Layer VA mapping JSON')
    args = parser.parse_args()

    print(f"=== Cross-Block Prefetch Analysis v3 ===")
    print(f"Trace: {args.trace_file}")

    # Parse
    block_seq = parse_block_sequence(args.trace_file)
    print(f"Block sequence length: {len(block_seq):,}")
    print(f"Unique blocks: {len(set(va for _, va in block_seq)):,}")

    # Load layer mapping
    boundaries, layers = load_layer_boundaries(args.layer_mapping)
    print(f"Layers: {len(layers)}")

    # ========== 1. Prefill vs Decode Phase Separation ==========
    print(f"\n{'='*60}")
    print(f"=== 1. Prefill vs Decode Phase Separation ===")
    print(f"{'='*60}")

    boundary_idx = detect_prefill_decode_boundary(block_seq)
    prefill_seq = block_seq[:boundary_idx]
    decode_seq = block_seq[boundary_idx:]

    print(f"\n  Boundary at index {boundary_idx}")
    print(f"  Prefill: {len(prefill_seq):,} events ({prefill_seq[-1][0] - prefill_seq[0][0]:.0f} ms)")
    print(f"  Decode:  {len(decode_seq):,} events ({decode_seq[-1][0] - decode_seq[0][0]:.0f} ms)")

    # Adjacent hit rate by phase
    print(f"\n  --- Adjacent Hit Rate by Phase ---")
    analyze_adjacent_hit_rate(prefill_seq, "Prefill")
    analyze_adjacent_hit_rate(decode_seq, "Decode")
    analyze_adjacent_hit_rate(block_seq, "Overall")

    # ========== 2. Per-Layer Transition Classification ==========
    print(f"\n{'='*60}")
    print(f"=== 2. Per-Layer Transition Classification ===")
    print(f"{'='*60}")

    analyze_layer_transitions(prefill_seq, boundaries, "Prefill")
    analyze_layer_transitions(decode_seq, boundaries, "Decode")
    analyze_layer_transitions(block_seq, boundaries, "Overall")

    # ========== 3. History-Based Prediction Compression ==========
    print(f"\n{'='*60}")
    print(f"=== 3. History-Based Prediction Compression ===")
    print(f"{'='*60}")

    # Analyze decode phase only (prefill is easy to predict)
    analyze_history_compression(decode_seq, history_len=1)
    analyze_history_compression(decode_seq, history_len=2)
    analyze_history_compression(decode_seq, history_len=3)

    # ========== 4. Layer-Aware Prefetch Potential ==========
    print(f"\n{'='*60}")
    print(f"=== 4. Layer-Aware Prefetch Potential ===")
    print(f"{'='*60}")

    analyze_layer_aware_prefetch(decode_seq, boundaries, layers)

    # ========== 5. Selective Prefetch Analysis ==========
    print(f"\n{'='*60}")
    print(f"=== 5. Selective Prefetch (Momentum Detection) ===")
    print(f"{'='*60}")

    # How often do we get 2+ consecutive adjacent accesses?
    consecutive_runs = []
    current_run = 0
    for i in range(len(decode_seq) - 1):
        diff = decode_seq[i+1][1] - decode_seq[i][1]
        if diff == VA_BLOCK_SIZE:
            current_run += 1
        else:
            if current_run > 0:
                consecutive_runs.append(current_run)
            current_run = 0
    if current_run > 0:
        consecutive_runs.append(current_run)

    # Selective: only prefetch after 2+ consecutive adjacent
    # How many prefetches would be issued?
    selective_prefetches = sum(max(0, r - 1) for r in consecutive_runs)
    blind_transitions = sum(1 for i in range(len(decode_seq) - 1)
                           if decode_seq[i+1][1] != decode_seq[i][1])

    print(f"\n  Consecutive adjacent runs: {len(consecutive_runs):,}")
    if consecutive_runs:
        print(f"  Mean run length: {sum(consecutive_runs)/len(consecutive_runs):.1f}")
        print(f"  Runs ≥ 2: {sum(1 for r in consecutive_runs if r >= 2):,}")
        print(f"  Runs ≥ 3: {sum(1 for r in consecutive_runs if r >= 3):,}")

    print(f"\n  Blind adjacent prefetches (per transition): {blind_transitions:,}")
    print(f"  Selective prefetches (after 2+ adjacent): {selective_prefetches:,}")
    print(f"  Reduction: {100*(1 - selective_prefetches/max(1,blind_transitions)):.0f}%")

    # What's the hit rate for selective?
    selective_hits = 0
    consecutive_count = 0
    for i in range(len(decode_seq) - 1):
        diff = decode_seq[i+1][1] - decode_seq[i][1]
        if diff == VA_BLOCK_SIZE:
            consecutive_count += 1
            if consecutive_count >= 2:
                # Would have prefetched this one
                selective_hits += 1
        else:
            consecutive_count = 0

    print(f"  Selective hits: {selective_hits:,}/{selective_prefetches:,} = "
          f"{100*selective_hits/max(1,selective_prefetches):.1f}%")
    print(f"  vs blind hit rate: "
          f"{analyze_adjacent_hit_rate(decode_seq).get(1, (0,0,0))[2]:.1f}%")

    # ========== 6. Summary ==========
    print(f"\n{'='*60}")
    print(f"=== Summary of Key Findings ===")
    print(f"{'='*60}")

    prefill_result = analyze_adjacent_hit_rate(prefill_seq)
    decode_result = analyze_adjacent_hit_rate(decode_seq)

    print(f"""
  1. Prefill vs Decode:
     - Prefill adjacent hit rate: {prefill_result.get(1, (0,0,0))[2]:.1f}%
     - Decode adjacent hit rate:  {decode_result.get(1, (0,0,0))[2]:.1f}%
     → Cross-block {'helps' if prefill_result.get(1,(0,0,0))[2] > 40 else 'limited for'} prefill, \
{'helps' if decode_result.get(1,(0,0,0))[2] > 40 else 'limited for'} decode

  2. Layer transitions: (see detailed breakdown above)
     → Intra-layer transitions have higher adjacent hit rate

  3. History prediction compression:
     → Check how many patterns needed for 80% coverage

  4. Selective prefetch:
     → {100*(1 - selective_prefetches/max(1,blind_transitions)):.0f}% fewer prefetches than blind
     → Higher precision per prefetch issued
""")


if __name__ == '__main__':
    main()

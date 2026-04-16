#!/usr/bin/env python3
"""
Derive VA→Layer mapping from chunk_trace ACTIVATE events.

Analyzes chunk_trace CSV output to detect:
1. Decode step boundaries (VA address regression)
2. Per-step layer sequence (monotonically ascending VA → layer_id)
3. Consistent VA→layer mapping across multiple decode steps
4. T1/T2/T3 classification from access frequency

Output: layer_va_ranges.json compatible with prefetch_template_belady loader.

Usage:
  python3 derive_layer_mapping.py chunk_trace_120b_long.csv [--output layer_va_ranges.json]
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict


def parse_chunk_trace(filepath):
    """Parse chunk_trace CSV, return list of ACTIVATE events sorted by time."""
    events = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['hook_type'] != 'ACTIVATE':
                continue
            va_start_str = row.get('va_start', '').strip()
            if not va_start_str or va_start_str == '0x0':
                continue
            try:
                va_start = int(va_start_str, 16) if va_start_str.startswith('0x') else int(va_start_str)
            except ValueError:
                continue
            if va_start == 0:
                continue
            events.append({
                'time_ms': float(row['time_ms']),
                'va_start': va_start,
                'va_end': int(row['va_end'], 16),
                'va_page_index': int(row['va_page_index']),
                'chunk_addr': row['chunk_addr'],
            })
    events.sort(key=lambda e: e['time_ms'])
    return events


def detect_decode_steps(events, min_faults_per_step=10):
    """Detect decode step boundaries from VA regression patterns.

    A new decode step starts when:
    - VA drops significantly (current VA < 90% of recent max VA)
    - AND at least min_faults_per_step faults since last boundary
    """
    steps = []
    current_step = []
    max_va_seen = 0

    for evt in events:
        va = evt['va_start']

        # Detect regression: VA drops below 90% of max seen
        if (max_va_seen > 0 and
            va < max_va_seen * 0.7 and
            len(current_step) >= min_faults_per_step):
            steps.append(current_step)
            current_step = []
            max_va_seen = 0

        current_step.append(evt)
        max_va_seen = max(max_va_seen, va)

    if len(current_step) >= min_faults_per_step:
        steps.append(current_step)

    return steps


def extract_layer_boundaries(step_events):
    """From a single decode step's ACTIVATE events, extract layer boundaries.

    Within a decode step, VA addresses are monotonically ascending.
    Group contiguous VA chunks into layers based on VA gaps.
    """
    if not step_events:
        return []

    # Sort by VA
    sorted_evts = sorted(step_events, key=lambda e: e['va_start'])

    # Group into contiguous VA ranges (gap > 4MB = new layer)
    CHUNK_SIZE = 2 * 1024 * 1024  # 2MB
    GAP_THRESHOLD = 4 * 1024 * 1024  # 4MB gap = new layer

    layers = []
    current_layer_start = sorted_evts[0]['va_start']
    current_layer_end = sorted_evts[0]['va_end']
    current_chunks = [sorted_evts[0]]

    for evt in sorted_evts[1:]:
        gap = evt['va_start'] - current_layer_end
        if gap > GAP_THRESHOLD:
            # New layer
            layers.append({
                'va_start': current_layer_start,
                'va_end': current_layer_end,
                'num_chunks': len(current_chunks),
                'size_mb': (current_layer_end - current_layer_start + 1) / (1024 * 1024),
            })
            current_layer_start = evt['va_start']
            current_layer_end = evt['va_end']
            current_chunks = [evt]
        else:
            current_layer_end = max(current_layer_end, evt['va_end'])
            current_chunks.append(evt)

    # Last layer
    layers.append({
        'va_start': current_layer_start,
        'va_end': current_layer_end,
        'num_chunks': len(current_chunks),
        'size_mb': (current_layer_end - current_layer_start + 1) / (1024 * 1024),
    })

    return layers


def build_consistent_mapping(steps, min_consistency=0.5):
    """Build VA→layer mapping that's consistent across decode steps.

    For each step, derive layer boundaries. Then merge across steps
    to find consistent VA→layer assignments.
    """
    # Extract layers from each step
    all_step_layers = []
    for i, step in enumerate(steps):
        layers = extract_layer_boundaries(step)
        all_step_layers.append(layers)

    if not all_step_layers:
        return {}, []

    # Use the step with most layers as reference
    ref_idx = max(range(len(all_step_layers)), key=lambda i: len(all_step_layers[i]))
    ref_layers = all_step_layers[ref_idx]

    # Build VA→layer map from reference
    va_to_layer = {}
    CHUNK_SIZE = 2 * 1024 * 1024

    for layer_id, layer in enumerate(ref_layers):
        va = layer['va_start']
        while va <= layer['va_end']:
            va_key = va >> 21  # 2MB page number
            va_to_layer[va_key] = layer_id
            va += CHUNK_SIZE

    # Count chunk access frequency across all steps for T1/T2/T3
    chunk_step_count = defaultdict(int)  # va_key → how many steps it appears in
    for step in steps:
        seen_this_step = set()
        for evt in step:
            va_key = evt['va_start'] >> 21
            if va_key not in seen_this_step:
                seen_this_step.add(va_key)
                chunk_step_count[va_key] += 1

    return va_to_layer, ref_layers, chunk_step_count


def classify_chunks(chunk_step_count, num_steps):
    """Classify chunks into T1/T2/T3 based on access frequency across decode steps."""
    t1 = set()  # Accessed in 100% of steps
    t2 = set()  # Accessed in >50% of steps
    t3 = set()  # Accessed in ≤50% of steps

    for va_key, count in chunk_step_count.items():
        frac = count / num_steps
        if frac >= 0.95:
            t1.add(va_key)
        elif frac > 0.5:
            t2.add(va_key)
        else:
            t3.add(va_key)

    return t1, t2, t3


def main():
    parser = argparse.ArgumentParser(description='Derive VA→Layer mapping from chunk_trace')
    parser.add_argument('trace_file', help='Path to chunk_trace CSV file')
    parser.add_argument('--output', default='layer_va_ranges.json',
                        help='Output JSON file (default: layer_va_ranges.json)')
    parser.add_argument('--skip-warmup', type=int, default=5,
                        help='Skip first N decode steps as warmup (default: 5)')
    parser.add_argument('--gap-mb', type=float, default=4.0,
                        help='VA gap threshold for layer boundary (MB, default: 4.0)')
    args = parser.parse_args()

    if not os.path.exists(args.trace_file):
        print(f"Error: {args.trace_file} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing {args.trace_file}...")
    events = parse_chunk_trace(args.trace_file)
    print(f"  {len(events)} ACTIVATE events")

    if not events:
        print("Error: No ACTIVATE events found", file=sys.stderr)
        sys.exit(1)

    va_min = min(e['va_start'] for e in events)
    va_max = max(e['va_start'] for e in events)
    print(f"  VA range: {va_min:#x} - {va_max:#x} ({(va_max - va_min) / (1024**3):.2f} GiB)")

    print(f"\nDetecting decode steps...")
    steps = detect_decode_steps(events)
    print(f"  {len(steps)} decode steps detected")

    for i, step in enumerate(steps[:10]):
        va_s = min(e['va_start'] for e in step)
        va_e = max(e['va_start'] for e in step)
        t_s = step[0]['time_ms']
        t_e = step[-1]['time_ms']
        print(f"    Step {i}: {len(step)} faults, VA {va_s:#x}-{va_e:#x}, "
              f"t={t_s:.0f}-{t_e:.0f}ms ({t_e - t_s:.0f}ms)")

    if len(steps) > 10:
        print(f"    ... ({len(steps) - 10} more steps)")

    # Skip warmup steps
    analysis_steps = steps[args.skip_warmup:]
    if len(analysis_steps) < 3:
        print(f"Warning: Only {len(analysis_steps)} steps after skipping warmup. Using all steps.")
        analysis_steps = steps

    print(f"\nBuilding VA→Layer mapping from {len(analysis_steps)} steps...")
    va_to_layer, ref_layers, chunk_step_count = build_consistent_mapping(analysis_steps)

    print(f"  {len(ref_layers)} layers detected")
    print(f"  {len(va_to_layer)} VA entries in mapping")

    # Print layer info
    print(f"\nLayer boundaries:")
    for i, layer in enumerate(ref_layers):
        print(f"  Layer {i:3d}: {layer['va_start']:#14x} - {layer['va_end']:#14x} "
              f"({layer['size_mb']:8.1f} MB, {layer['num_chunks']:4d} chunks)")

    # T1/T2/T3 classification
    print(f"\nChunk classification (across {len(analysis_steps)} decode steps):")
    t1, t2, t3 = classify_chunks(chunk_step_count, len(analysis_steps))
    CHUNK_MB = 2
    print(f"  T1 (every step):  {len(t1):5d} chunks ({len(t1) * CHUNK_MB:8.1f} MB)")
    print(f"  T2 (>50% steps):  {len(t2):5d} chunks ({len(t2) * CHUNK_MB:8.1f} MB)")
    print(f"  T3 (≤50% steps):  {len(t3):5d} chunks ({len(t3) * CHUNK_MB:8.1f} MB)")
    print(f"  Total:            {len(chunk_step_count):5d} chunks "
          f"({len(chunk_step_count) * CHUNK_MB:8.1f} MB)")

    # Compare with analytical model
    print(f"\n  Analytical model comparison (120B GPT-OSS):")
    print(f"    Analytical T1: 2.14 GB, Measured T1: {len(t1) * CHUNK_MB / 1024:.2f} GB")
    print(f"    Analytical T2: 1.88 GB, Measured T2: {len(t2) * CHUNK_MB / 1024:.2f} GB")
    print(f"    Analytical T3: 58.33 GB, Measured T3: {len(t3) * CHUNK_MB / 1024:.2f} GB")

    # Output layer_va_ranges.json
    output_layers = {}
    for i, layer in enumerate(ref_layers):
        output_layers[str(i)] = {
            'layer_id': i,
            'va_start': hex(layer['va_start']),
            'va_end': hex(layer['va_end']),
            'size_mb': layer['size_mb'],
            'num_chunks': layer['num_chunks'],
            'kernels': [],
        }

    output = {
        'source': 'chunk_trace',
        'trace_file': os.path.basename(args.trace_file),
        'num_decode_steps': len(analysis_steps),
        'page_shift': 21,
        'num_layers': len(ref_layers),
        'classification': {
            't1_chunks': len(t1),
            't1_mb': len(t1) * CHUNK_MB,
            't2_chunks': len(t2),
            't2_mb': len(t2) * CHUNK_MB,
            't3_chunks': len(t3),
            't3_mb': len(t3) * CHUNK_MB,
        },
        'layers': output_layers,
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {args.output}")

    # Also output a simple va→layer lookup table for BPF loader
    lookup_file = args.output.replace('.json', '_lookup.json')
    lookup = {}
    for va_key, layer_id in va_to_layer.items():
        lookup[hex(va_key << 21)] = layer_id

    with open(lookup_file, 'w') as f:
        json.dump({
            'page_shift': 21,
            'num_entries': len(lookup),
            'va_to_layer': lookup,
        }, f, indent=2)
    print(f"Saved: {lookup_file}")

    print("\nDone.")


if __name__ == '__main__':
    main()

# Thread Scheduling Example

This example demonstrates how to use bpftime's GPU eBPF tracing to map CUDA threads to their hardware execution units: Streaming Multiprocessors (SMs), warps, and lanes.

## Overview

The example consists of two main components:

1. **Vector Addition CUDA Application** (`vec_add`): A simple CUDA application that repeatedly performs vector addition on the GPU.

2. **eBPF CUDA Probe** (`threadscheduling`): An eBPF program that attaches to CUDA kernel functions, monitoring their execution and thread scheduling.

When CUDA kernels execute, the GPU scheduler assigns thread blocks to Streaming Multiprocessors (SMs). Within each SM, threads are grouped into warps of 32 threads that execute in lockstep.

## GPU Hardware Concepts

| Concept | Description | PTX Register |
|---------|-------------|--------------|
| **SM (Streaming Multiprocessor)** | Physical processing unit on the GPU. Multiple SMs run in parallel. | `%smid` |
| **Warp** | A group of 32 threads that execute in lockstep on an SM. | `%warpid` |
| **Lane** | A thread's position (0-31) within its warp. | `%laneid` |

## eBPF Helpers

This example uses three GPU eBPF helpers:

| Helper ID | Function | Description |
|-----------|----------|-------------|
| 509 | `bpf_get_sm_id()` | Returns the SM ID executing this thread |
| 510 | `bpf_get_warp_id()` | Returns the warp ID within the SM |
| 511 | `bpf_get_lane_id()` | Returns the lane ID (0-31) within the warp |

## Building the Example

```bash
# From the bpftime root directory, build with CUDA support
cmake -Bbuild -DBPFTIME_ENABLE_CUDA_ATTACH=1 -DBPFTIME_CUDA_ROOT=/usr/local/cuda .
make -C build -j$(nproc)

# Build this example
make -C example/gpu/threadscheduling
```

## Running the Example

You need two terminals:

### Terminal 1: Launch the eBPF Program (Server)

```bash
BPFTIME_LOG_OUTPUT=console LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so \
  example/gpu/threadscheduling/threadscheduling
```

### Terminal 2: Run the CUDA Application (Client)

```bash
BPFTIME_LOG_OUTPUT=console LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
  example/gpu/threadscheduling/vec_add [num_blocks] [threads_per_block]
```

Optional arguments:
- `num_blocks`: Number of thread blocks (default: 4)
- `threads_per_block`: Threads per block (default: 64)

Example with custom configuration:
```bash
# Run with 8 blocks and 128 threads per block
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
  example/gpu/threadscheduling/vec_add 8 128
```

## Trace example

BPFTIME_LOG_OUTPUT=console LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
  example/gpu/threadscheduling/vec_add 16 64

```

╔════════════════════════════════════════════════════════════════════╗
║              SM / Warp / Lane Mapping Report                       ║
╚════════════════════════════════════════════════════════════════════╝

  Timestamp: 2025-12-08 19:46:47

┌─ SM Utilization Histogram ─────────────────────────────────────────┐
│                                                                    │
│  SM  0: █████████████████████████                      239 threads │
│  SM  1: █                                                8 threads │
│  SM  2: ██████████████                                 138 threads │
│  SM  3: ████                                            45 threads │
│  SM  4: ███                                             36 threads │
│  SM  5: █                                               14 threads │
│  SM  6: █                                                3 threads │
│  SM  7: █                                                6 threads │
│  SM  8: █                                                6 threads │
│  SM  9: █████████████                                  126 threads │
│  SM 10: ██                                              24 threads │
│  SM 11: █████████                                       94 threads │
│  SM 12: █████                                           51 threads │
│  SM 13: █                                                5 threads │
│  SM 14: █                                                7 threads │
│  SM 15: ████████████████████████████████████████       382 threads │
│                                                                    │
│  Total threads: 1184      Active SMs: 16                           │
└────────────────────────────────────────────────────────────────────┘

  Load Balance Score: 48.6% (100% = perfect distribution)

┌─ Warp Distribution per SM ─────────────────────────────────────────┐
│  SM   │ Warp ID │ Thread Count                                     │
├───────┼─────────┼──────────────────────────────────────────────────┤
│    0  │     0   │       47                                         │
│    7  │     2   │       54                                         │
│    4  │     1   │       23                                         │
│   11  │     3   │      114                                         │
│    1  │     0   │       62                                         │
│    8  │     2   │       10                                         │
│    5  │     1   │       17                                         │
│   12  │     3   │      108                                         │
│    2  │     0   │       39                                         │
│    9  │     2   │       57                                         │
│    6  │     1   │       30                                         │
│   13  │     3   │       72                                         │
│    3  │     0   │       22                                         │
│   10  │     2   │        3                                         │
│    7  │     1   │       33                                         │
│   14  │     3   │       21                                         │
│    4  │     0   │        5                                         │
│   11  │     2   │       61                                         │
│    8  │     1   │       81                                         │
│   15  │     3   │       32                                         │
│    5  │     0   │      110                                         │
│   12  │     2   │       78                                         │
│    0  │     8   │       22                                         │
│    9  │     1   │       22                                         │
│    6  │     0   │       19                                         │
│   13  │     2   │       63                                         │
│   10  │     1   │      161                                         │
│    7  │     0   │       12                                         │
│   14  │     2   │       21                                         │
│   11  │     1   │       21                                         │
│    8  │     0   │      120                                         │
│   15  │     2   │       41                                         │
│   12  │     1   │       31                                         │
│    0  │     7   │      110                                         │
│    9  │     0   │       12                                         │
│   13  │     1   │       10                                         │
│   10  │     0   │      103                                         │
│   14  │     1   │       42                                         │
│   11  │     0   │        1                                         │
│   15  │     1   │       12                                         │
│   12  │     0   │       23                                         │
│    0  │     6   │       30                                         │
│   13  │     0   │       19                                         │
│   14  │     0   │       44                                         │
│    0  │    15   │        4                                         │
│   15  │     0   │       31                                         │
│    0  │     5   │      152                                         │
│    0  │    14   │      134                                         │
│    0  │     4   │       19                                         │
│    0  │     3   │       63                                         │
│    1  │     3   │       37                                         │
│    0  │    12   │      191                                         │
│    2  │     3   │       24                                         │
│    3  │     3   │        6                                         │
│    0  │     2   │       18                                         │
│    4  │     3   │       16                                         │
│    1  │     2   │       53                                         │
│    5  │     3   │       29                                         │
│    0  │    11   │       50                                         │
│    2  │     2   │        5                                         │
│    6  │     3   │      116                                         │
│    3  │     2   │       10                                         │
│    0  │     1   │       29                                         │
│    7  │     3   │       27                                         │
│    4  │     2   │       67                                         │
│    1  │     1   │       26                                         │
│    8  │     3   │       17                                         │
│    0  │    10   │      312                                         │
│    2  │     1   │       16                                         │
│    9  │     3   │       63                                         │
│    6  │     2   │       60                                         │
│    3  │     1   │       41                                         │
└───────┴─────────┴──────────────────────────────────────────────────┘

┌─ Thread-to-Hardware Mapping Samples ───────────────────────────────┐
│  Block(x,y,z)  │ Thread(x,y,z) │  SM  │ Warp │ Lane │              │
├────────────────┼───────────────┼──────┼──────┼──────┼──────────────┤
│  ( 14, 0, 0)   │  ( 32, 0, 0)  │   14 │    1 │    0 │              │
│  ( 14, 0, 0)   │  ( 32, 0, 0)  │    1 │    1 │    0 │              │
│  ( 14, 0, 0)   │  ( 32, 0, 0)  │    1 │    1 │    0 │              │
│  ( 14, 0, 0)   │  ( 32, 0, 0)  │   14 │    1 │    0 │              │
│  ( 14, 0, 0)   │  ( 32, 0, 0)  │    1 │    1 │    0 │              │
│  ( 14, 0, 0)   │  ( 32, 0, 0)  │    1 │    1 │    0 │              │
│  ( 14, 0, 0)   │  ( 32, 0, 0)  │    1 │    1 │    0 │              │
│  ( 14, 0, 0)   │  ( 32, 0, 0)  │    1 │    1 │    0 │              │
│  ( 14, 0, 0)   │  ( 32, 0, 0)  │    1 │    1 │    0 │              │
│  ( 14, 0, 0)   │  ( 32, 0, 0)  │    1 │    1 │    0 │              │
│  ...           │ ...           │ ...  │ ...  │ ...  │ (10 samples) │
└────────────────┴───────────────┴──────┴──────┴──────┴──────────────┘

  Press Ctrl+C to exit.
```

## Understanding the Output

The probe displays:

### SM Utilization Histogram
Shows how threads are distributed across SMs:
```
┌─ SM Utilization Histogram ─────────────────────────────────────────┐
│  SM  0: ████████████████████████████████████████     64 threads    │
│  SM  1: ████████████████████████████████████████     64 threads    │
│  SM  2: ████████████████████████████████████████     64 threads    │
│  SM  3: ████████████████████████████████████████     64 threads    │
│                                                                    │
│  Total threads: 256       Active SMs: 4                            │
└────────────────────────────────────────────────────────────────────┘
```

### Warp Distribution per SM
Shows which warps are active on each SM:
```
┌─ Warp Distribution per SM ─────────────────────────────────────────┐
│  SM   │ Warp ID │ Thread Count                                     │
├───────┼─────────┼──────────────────────────────────────────────────┤
│    0  │     0   │       32                                         │
│    0  │     1   │       32                                         │
│    1  │     0   │       32                                         │
│    1  │     1   │       32                                         │
└───────┴─────────┴──────────────────────────────────────────────────┘
```

### Thread-to-Hardware Mapping Samples
Shows individual thread assignments:
```
┌─ Thread-to-Hardware Mapping Samples ───────────────────────────────┐
│  Block(x,y,z)  │ Thread(x,y,z) │  SM  │ Warp │ Lane │              │
├────────────────┼───────────────┼──────┼──────┼──────┼──────────────┤
│  (  0, 0, 0)   │  (  0, 0, 0)  │    2 │    0 │    0 │              │
│  (  0, 0, 0)   │  ( 31, 0, 0)  │    2 │    0 │   31 │              │
│  (  1, 0, 0)   │  (  0, 0, 0)  │    5 │    0 │    0 │              │
└────────────────┴───────────────┴──────┴──────┴──────┴──────────────┘
```

### Load Balance Score
A percentage indicating how evenly threads are distributed across SMs:
- 100% = perfect distribution (all SMs have equal load)
- Lower values indicate imbalanced distribution

## Use Cases

### 1. Verify Block-to-SM Distribution
Run with different block counts to see how the GPU scheduler distributes work:
```bash
# Few blocks - may not use all SMs
./vec_add 2 64

# Many blocks - should distribute across all SMs
./vec_add 16 64
```

### 2. Debug Persistent Kernels
For persistent kernel designs (one block per SM), verify each block maps to a unique SM by running one block per SM:

Number of SMs varies by GPU, so you will adjust your `block` size accordingly:
```bash
# Example for RTX 3090 (82 SMs)
./vec_add 82 64

# Example for RTX 4090 (128 SMs)
./vec_add 128 64
```

### 3. Analyze Warp Occupancy
Check how warps are distributed within blocks:
```bash
# 128 threads = 4 warps per block
./vec_add 4 128

# 256 threads = 8 warps per block
./vec_add 4 256
```

## Key Observations from the Figure

From the thread scheduling visualization (Figure `thread_scheduling_motivation.pdf`), we can draw several important conclusions:

### (a) SM Load Distribution

1. **Severe Load Imbalance**: The default GPU scheduler produces highly uneven thread distribution across SMs. SM 15 handles 382 threads while SM 6 only has 3 threads - a 127x difference.

2. **Low Load Balance Score (48.6%)**: Less than half of the ideal balanced distribution, indicating significant scheduling inefficiency.

3. **Underutilized SMs**: Many SMs (1, 5, 6, 7, 8, 13, 14) have fewer than 15 threads, far below the ideal of 74 threads per SM.

### (b) Warp Activity per SM

1. **Concentrated Warp Activity**: SM 0 has unusually high activity in high-numbered warps (warp 10-15), with warp 10 alone having 312 thread executions.

2. **Sparse Warp Utilization**: Most SMs only use warps 0-3, while higher warp IDs (4-15) are mostly idle except for SM 0.

3. **Non-uniform Scheduling**: The same logical warp ID shows vastly different activity levels across SMs, suggesting unpredictable scheduling behavior.

### Motivation for Programmable Observability

These observations motivate the need for:

1. **Fine-grained GPU observability**: Default profiling tools (CUPTI, NSight) don't expose real-time SM/warp-level scheduling information.

2. **Programmable tracing**: eBPF-based tools can capture per-thread hardware mapping without kernel modification.

3. **Custom scheduling policies**: Understanding the default scheduler's behavior is the first step toward implementing better load-balancing strategies.

## Implementation Details

The eBPF probe (`threadscheduling.bpf.c`) attaches to the `vectorAdd` CUDA kernel and:

1. Reads hardware scheduling registers via helpers 509-511
2. Records thread-to-hardware mapping in a BPF hash map
3. Maintains SM and warp histograms for analysis
4. Outputs debug information for the first thread of each warp

The userspace loader (`threadscheduling.c`) periodically:

1. Reads the BPF maps
2. Computes statistics and histograms
3. Displays the mapping visualization


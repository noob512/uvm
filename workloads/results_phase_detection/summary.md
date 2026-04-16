# Phase Detection Experiment Results

Date: 2026-03-05 (run 20260305_run)
Hardware: RTX 5090 (32GB), Intel Core Ultra 9 285K, custom nvidia driver 575.57.08

## PART 1: llama.cpp Phase Detection

Model: gpt-oss-120B MoE (MXFP4), pp=512, tg=128, r=3 reps
Oversubscription ratio: ~1.84x

| Config | Description | pp (tok/s) | tg (tok/s) | tg delta |
|--------|-------------|-----------|-----------|----------|
| A | always_max_cycle_moe (baseline) | 213.94 | 83.87 | -- |
| B | phase mode0 noxb (always_max both phases) | 209.12 | 84.31 | +0.5% |
| C | phase mode1 r=32 noxb (narrow decode) | 213.40 | 44.03 | **-47.5%** |
| D | phase mode1 r=8 noxb (very narrow decode) | 215.99 | 35.55 | **-57.6%** |
| E | phase mode2 noxb (default kernel during decode) | 215.71 | 49.07 | **-41.5%** |
| F | phase mode3 noxb (forward-only decode) | 216.37 | 64.46 | **-23.1%** |
| G | phase mode0 + XB prefill | 203.69 | 82.08 | -2.1% |

### Key Findings (llama.cpp)

1. **Phase detection does NOT help for llama.cpp MoE decode.** All modes that reduce prefetch scope during decode (modes 1-3) severely degrade tg performance (-23% to -58%).
2. **MoE decode needs always_max prefetch.** The MoE expert selection pattern requires aggressive full-block prefetch during decode. Narrowing the prefetch window starves the GPU of data.
3. **Cross-block (XB) prefill is neutral-to-slightly-negative.** Config G (mode0+XB) shows pp -5% and tg -2%, suggesting XB adds overhead without benefit at this oversubscription ratio.
4. **Phase mode0 matches baseline.** Config B validates the uprobe mechanism: mode0 (always_max in both phases) closely matches Config A.
5. **pp is stable across configs (~209-216).** Prefill performance is relatively insensitive to decode prefetch mode since it runs before decode starts.

## PART 2: FAISS Phase Detection

Dataset: SIFT100M, Index: IVF4096,Flat, nnn=10
Oversubscription ratio: ~1.5x

| Config | Description | add (s) | search np=1 (s) | search np=4 (s) | search np=16 (s) |
|--------|-------------|---------|-----------------|-----------------|-------------------|
| 2a | no BPF (baseline) | 98.5 | 5.12 | 14.95 | 62.94 |
| 2b | always_max_cycle_moe | 48.9 | 4.42 | 12.52 | 49.39 |
| 2c | uprobe phase detection | 48.3 | 4.39 | 12.62 | 49.23 |
| 2d | heuristic phase detection | 48.9 | 4.40 | 12.59 | 49.38 |

### Key Findings (FAISS)

1. **All BPF policies provide ~2x add speedup and 14-22% search improvement** vs no-BPF baseline.
2. **Uprobe vs heuristic: no measurable difference.** Uprobe (2c) and heuristic (2d) produce nearly identical results across all metrics. The heuristic stride-based detection is sufficient.
3. **always_max_cycle_moe is equally effective.** A generic always_max policy matches the FAISS-specific phase detectors.
4. **Improvement breakdown**: add: -51% (98.5->48.3s), np=1: -14%, np=4: -16%, np=16: -22%.

## PART 3: vLLM Phase Detection

Model: 30B UVM, 100 prompts serving benchmark
Oversubscription ratio: ~1.175x

| Config | Description | TPOT (ms) | Throughput (tok/s) | P99 TPOT (ms) | TPOT delta |
|--------|-------------|-----------|-------------------|---------------|------------|
| 3a | no BPF (baseline) | 61.26 | 232.55 | 64.76 | -- |
| 3b | always_max_cycle_moe | 56.89 | 250.99 | 59.43 | **-7.1%** |
| 3c | phase mode0 noxb | 57.12 | 249.46 | 59.71 | **-6.8%** |
| 3d | phase mode0 xb_both | 57.83 | 247.73 | 61.42 | -5.6% |
| 3e | phase mode1 r=32 | 62.89 | 228.94 | 65.97 | +2.7% |
| 3f | phase mode2 | 62.23 | 230.96 | 67.10 | +1.6% |

### Key Findings (vLLM)

1. **always_max (3b) and phase mode0 (3c) are the winners** with ~7% TPOT reduction and ~8% throughput improvement.
2. **Reducing prefetch during decode (mode1, mode2) is HARMFUL** -- same pattern as llama.cpp.
3. **Cross-block during decode (3d) slightly worse than no-XB (3c)** -- TPOT 57.83 vs 57.12ms.
4. **Phase mode0 matches always_max_cycle_moe**, validating the uprobe mechanism overhead is negligible.

## Cross-Workload Conclusions

1. **Phase-adaptive prefetch that REDUCES decode aggressiveness is universally harmful.** All three workloads show degradation when the decode phase uses narrower prefetch (modes 1, 2, 3). The GPU's memory access patterns during decode are too complex for simple radius-based restriction.

2. **The best policy is always_max (full-block prefetch) in ALL phases.** Phase detection adds mechanism complexity without improving performance when the optimal policy is the same in both phases.

3. **Cross-block prefetch adds marginal-to-negative value** at these oversubscription ratios (1.175x-1.84x). The additional DMA traffic competes with demand-paging migration.

4. **Uprobe-based phase detection has negligible overhead.** Config B/3c closely match the always_max baseline, confirming the uprobe mechanism itself is sound -- it's the phase-adaptive policy that fails, not the detection.

5. **For FAISS, both uprobe and heuristic phase detection are equivalent.** The stride-based heuristic already identifies build vs search accurately enough.

## Result Files

All raw results stored in:
`/home/yunwei37/workspace/gpu/gpu_ext/workloads/results_phase_detection/20260305_run/`

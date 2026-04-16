
# CUDA Benchmark Results

**Device:** NVIDIA GeForce RTX 5090  
**Timestamp:** 2025-12-03T22:53:54.945217  

## Workload Configuration

| Workload | Binary | Elements | Iterations | Threads | Blocks |
|----------|--------|----------|------------|---------|--------|
| large | benchmark/gpu/workload/vec_add | 100000 | 1000 | 512 | 196 |
| medium | benchmark/gpu/workload/vec_add | 10000 | 10000 | 256 | 40 |
| minimal | benchmark/gpu/workload/vec_add | 32 | 3 | 32 | 1 |
| small | benchmark/gpu/workload/vec_add | 1000 | 10000 | 256 | 4 |
| small_64x16 | benchmark/gpu/workload/vec_add | 1000 | 10000 | 64 | 16 |
| tiny | benchmark/gpu/workload/vec_add | 32 | 10000 | 32 | 1 |
| xlarge | benchmark/gpu/workload/vec_add | 1000000 | 1000 | 512 | 1954 |

## Benchmark Results

| Test Name | Workload | Avg Time (μs) | vs Baseline | Overhead |
|-----------|----------|---------------|-------------|----------|
| Baseline (tiny) | tiny | 5.15 | - | - |
| Baseline (small) | small | 5.23 | - | - |
| Baseline (medium) | medium | 5.27 | - | - |
| Baseline (large) | large | 5.57 | - | - |
| Baseline (xlarge) | xlarge | 7.03 | - | - |
| Empty probe (tiny) | tiny | 5.53 | 5.15 | 1.07x (+7.4%) |
| Empty probe (small) | small | 5.58 | 5.23 | 1.07x (+6.7%) |
| Empty probe (medium) | medium | 5.61 | 5.27 | 1.06x (+6.5%) |
| Empty probe (large) | large | 5.94 | 5.57 | 1.07x (+6.6%) |
| Empty probe (xlarge) | xlarge | 7.53 | 7.03 | 1.07x (+7.1%) |
| Entry probe (tiny) | tiny | 5.49 | 5.15 | 1.07x (+6.6%) |
| Entry probe (small) | small | 5.55 | 5.23 | 1.06x (+6.1%) |
| Entry probe (medium) | medium | 6.22 | 5.27 | 1.18x (+18.0%) |
| Entry probe (xlarge) | xlarge | 7.50 | 7.03 | 1.07x (+6.7%) |
| Exit probe (tiny) | tiny | 5.35 | 5.15 | 1.04x (+3.9%) |
| Exit probe (small) | small | 5.38 | 5.23 | 1.03x (+2.9%) |
| Exit probe (large) | large | 5.80 | 5.57 | 1.04x (+4.1%) |
| Entry+Exit (tiny) | tiny | 6.06 | 5.15 | 1.18x (+17.7%) |
| Entry+Exit (small_64x16) | small_64x16 | 6.24 | 5.23 | 1.19x (+19.3%) |
| Entry+Exit (small) | small | 6.16 | 5.23 | 1.18x (+17.8%) |
| Entry+Exit (medium) | medium | 5.65 | 5.27 | 1.07x (+7.2%) |
| Entry+Exit (large) | large | 5.91 | 5.57 | 1.06x (+6.1%) |
| Entry+Exit (xlarge) | xlarge | 7.55 | 7.03 | 1.07x (+7.4%) |
| GPU Ringbuf (tiny) | tiny | 6.86 | 5.15 | 1.33x (+33.2%) |
| GPU Ringbuf (small_64x16) | small_64x16 | 8.88 | 5.23 | 1.70x (+69.8%) |
| GPU Ringbuf (small) | small | 8.92 | 5.23 | 1.71x (+70.6%) |
| GPU Ringbuf (medium) | medium | 33.03 | 5.27 | 6.27x (+526.8%) |
| GPU Ringbuf (large) | large | 1.96 | 5.57 | 0.35x (-64.8%) |
| GPU Ringbuf (xlarge) | xlarge | 1.86 | 7.03 | 0.26x (-73.5%) |
| Global timer (tiny) | tiny | 5.53 | 5.15 | 1.07x (+7.4%) |
| Global timer (small) | small | 5.71 | 5.23 | 1.09x (+9.2%) |
| Global timer (medium) | medium | 6.42 | 5.27 | 1.22x (+21.8%) |
| Global timer (xlarge) | xlarge | 25.97 | 7.03 | 3.69x (+269.4%) |
| Per-GPU-thread array (tiny) | tiny | 6.11 | 5.15 | 1.19x (+18.6%) |
| Per-GPU-thread array (small_64x16) | small_64x16 | 6.23 | 5.23 | 1.19x (+19.1%) |
| Per-GPU-thread array (small) | small | 6.56 | 5.23 | 1.25x (+25.4%) |
| Per-GPU-thread array (medium) | medium | FAILED | - | - |
| Per-GPU-thread array (large) | large | FAILED | - | - |
| Per-GPU-thread array (xlarge) | xlarge | FAILED | - | - |
| Memtrace (tiny) | tiny | 6.11 | 5.15 | 1.19x (+18.6%) |
| Memtrace (small) | small | 8.77 | 5.23 | 1.68x (+67.7%) |
| Memtrace (medium) | medium | 8.81 | 5.27 | 1.67x (+67.2%) |
| Memtrace (large) | large | 29.62 | 5.57 | 5.32x (+431.8%) |
| GPU Array map update (tiny) | tiny | 6.95 | 5.15 | 1.35x (+35.0%) |
| GPU Array map update (small_64x16) | small_64x16 | 6.61 | 5.23 | 1.26x (+26.4%) |
| GPU Array map update (small) | small | 7.55 | 5.23 | 1.44x (+44.4%) |
| GPU Array map update (medium) | medium | 7.63 | 5.27 | 1.45x (+44.8%) |
| GPU Array map update (large) | large | 15.38 | 5.57 | 2.76x (+176.1%) |
| GPU Array map update (xlarge) | xlarge | 231.19 | 7.03 | 32.89x (+3188.6%) |
| GPU Array map lookup (tiny) | tiny | 5.75 | 5.15 | 1.12x (+11.7%) |
| GPU Array map lookup (small_64x16) | small_64x16 | 5.88 | 5.23 | 1.12x (+12.4%) |
| GPU Array map lookup (small) | small | 5.97 | 5.23 | 1.14x (+14.1%) |
| GPU Array map lookup (medium) | medium | 6.01 | 5.27 | 1.14x (+14.0%) |
| GPU Array map lookup (large) | large | 9.17 | 5.57 | 1.65x (+64.6%) |
| GPU Array map lookup (xlarge) | xlarge | 33.39 | 7.03 | 4.75x (+375.0%) |
| CPU Array map update (minimal) | minimal | 33886.86 | 5.15 | 6579.97x (+657897.3%) |
| CPU Array map lookup (minimal) | minimal | 33738.82 | 5.15 | 6551.23x (+655022.7%) |
| CPU Hash map update (minimal) | minimal | 34903.79 | 5.15 | 6777.43x (+677643.5%) |
| CPU Hash map lookup (minimal) | minimal | 33724.10 | 5.15 | 6548.37x (+654736.9%) |
| CPU Hash map delete (minimal) | minimal | 33763.20 | 5.15 | 6555.96x (+655496.1%) |


# vLLM UVM Baseline Tests

## Quick Start

```bash
# Run all three baselines (cpu_offload, uvm_baseline, lmcache)
python test_uvm_baselines.py

# Run specific baselines
python test_uvm_baselines.py --baselines cpu_offload uvm_baseline

# Custom benchmark args
python test_uvm_baselines.py \
  --bench-args "--model Qwen/Qwen3-30B-A3B-FP8 \
    --dataset-name sharegpt --num-prompts 100 \
    --dataset-path ../datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
    --sharegpt-output-len 512 --seed 42 --request-rate 5"
```

## Generate Figures

```bash
cd first-iter && python generate_figures.py
```

See [../README.md](../README.md) for full instructions and reference results.

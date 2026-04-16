# Deprecated Scripts

These scripts were superseded by the atomic script architecture (Layer 1 + Layer 2).

## Replacements

| Old Script | Replaced By |
|---|---|
| llama_exp1_expert_offload.sh | llama.cpp/configs/bench.py + scripts/run_trials.py |
| pytorch_exp3_gnn.sh | pytorch/configs/gnn.py + scripts/run_trials.py |
| faiss_exp4_vector_search.sh | faiss/configs/search.py + scripts/run_trials.py |
| vllm_exp2_kv_offload.sh | vllm/configs/serve_bench.py + scripts/run_trials.py |
| llama_uvm_test_baselines.py | llama.cpp/configs/server_bench.py |
| vllm_uvm_test_baselines.py | vllm/configs/serve_bench.py |

## Notable References

- `vllm_uvm_test_baselines.py` contains LMCache server config (kv-transfer-config JSON, LMCACHE_* env vars)
- All scripts contain paper reference values in their summary sections

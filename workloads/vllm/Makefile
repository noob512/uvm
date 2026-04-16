# Simplified Makefile for vLLM benchmarks

VENV ?= .venv/bin/activate
DATASET_PATH ?= datasets/ShareGPT_V3_unfiltered_cleaned_split.json

.PHONY: bench download-datasets

# Run vLLM benchmark using CLI
bench:
	@. $(VENV) && vllm bench serve \
		--model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
		--dataset-name sharegpt \
		--num-prompts 1000 \
		--dataset-path $(DATASET_PATH)

# Download benchmark datasets
download-datasets:
	@mkdir -p datasets
	@if [ ! -f "datasets/ShareGPT_V3_unfiltered_cleaned_split.json" ]; then \
		echo "Downloading ShareGPT dataset..."; \
		wget -P datasets https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json; \
	fi
	@echo "Dataset ready in ./datasets/"

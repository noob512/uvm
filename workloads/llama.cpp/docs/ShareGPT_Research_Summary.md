# ShareGPT Dataset Research Summary

## Overview

ShareGPT is a dataset consisting of real-world conversations between users and AI assistants (primarily ChatGPT), collected through the ShareGPT browser extension before the service was discontinued. The dataset has become a standard benchmark in the ML systems community for evaluating LLM serving performance and model quality.

## Dataset Characteristics

### Core Properties
- **Type**: Real-world user-AI conversations
- **Collection Method**: Via ShareGPT Chrome Extension (users could share conversations with one click)
- **Content**: Diverse conversations covering various topics and use cases
- **Versions**: 
  - ShareGPT52K (collected before April 2023)
  - ShareGPT92K (collected from launch to June 2023)
  - ShareGPT_V3_unfiltered_cleaned_split.json (commonly used cleaned version)

### Statistical Properties (from vLLM benchmarks)
- **Average Input Tokens**: ~202
- **Average Output Tokens**: ~179
- **Usage**: Typically 500 prompts randomly sampled with fixed seed for reproducible benchmarks

## Usage in ML Systems Research

### 1. LLM Serving Performance Benchmarking

ShareGPT has become the de facto standard for benchmarking LLM serving systems, used by:
- **vLLM**: Achieves 1.8-2.7x throughput improvements, reaching state-of-the-art performance
- **TensorRT-LLM**: Shows highest throughput in many configurations
- **SGLang**: Comparable performance to vLLM on TTFT and TPOT metrics
- **Other frameworks**: TGI, LMDeploy, DeepSpeed-MII, ScaleLLM

### 2. Key Metrics Evaluated

When using ShareGPT for benchmarking, systems typically measure:
- **Throughput Metrics**:
  - Request throughput (req/s)
  - Output token throughput (tok/s)
  - Total token throughput (tok/s)

- **Latency Metrics**:
  - TTFT (Time To First Token): mean, median, p99
  - ITL (Inter-Token Latency): time between consecutive tokens
  - TPOT (Time Per Output Token): (Total_latency - TTFT) / Total_output_tokens

### 3. Model Training and Fine-tuning

- **Vicuna**: Trained by fine-tuning LLaMA on 70K ShareGPT conversations
- **ShareGPT4V**: Extended to multi-modal with 1.2M highly descriptive captions
- **Various fine-tuned models**: Used as training data for instruction-following capabilities

## Research Papers Using ShareGPT

### 1. ShareGPT4V: Improving Large Multi-Modal Models (arXiv:2311.12793)
- Creates 1.2M caption dataset from ShareGPT conversations
- Demonstrates improvements on MME and MMBench benchmarks
- Gains of 222.8/22.0/22.3 and 2.7/1.3/1.5 respectively

### 2. Early ChatGPT User Portrait through the Lens of Data (arXiv:2312.10078)
- Analyzes user behavior patterns from ShareGPT data
- Studies ShareGPT52K and ShareGPT92K datasets
- Provides insights into real-world AI usage patterns

### 3. Benchmarking Large Language Models on Controllable Generation (arXiv:2401.00690)
- Uses ShareGPT to measure instruction diversity
- Compares synthetic instructions against real-world ShareGPT data
- Validates that ShareGPT represents realistic usage scenarios

## Benchmark Results Examples

### vLLM v0.6.0 on H100 (Llama 3.1 70B, TP=8)
- Request throughput: 16.05 req/s
- Output token throughput: 2993.85 tok/s
- Mean TTFT: 564.63 ms
- P99 TTFT: 2301.28 ms

### Comparative Performance
- vLLM achieves highest throughput on ShareGPT dataset for Llama-3 models on H100
- TensorRT-LLM shows highest throughput overall, with vLLM second
- SGLang and vLLM have comparable TTFT and TPOT performance

## Why ShareGPT is Valuable for ML Systems

1. **Real-world Distribution**: Captures actual user interaction patterns
2. **Diverse Content**: Covers wide range of topics and conversation lengths
3. **Reproducibility**: Fixed sampling with seeds ensures consistent benchmarks
4. **Industry Standard**: Widely adopted across major LLM serving frameworks
5. **Balanced Workload**: Mix of prompt processing and generation tasks

## Common Usage Patterns

### For Benchmarking
```bash
python benchmark_serving.py \
  --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 500
```

### Dataset Variants
- **Cleaned versions**: Remove problematic conversations
- **Filtered versions**: Focus on specific languages or topics
- **Synthetic extensions**: GPT-4 enhanced versions for specific tasks

## Implications for Scheduler Evaluation

ShareGPT is particularly useful for testing Linux schedulers with LLM workloads because:
1. **Variable Request Patterns**: Tests scheduler's ability to handle diverse workloads
2. **Mixed Compute Requirements**: Prompt processing (parallel) vs generation (sequential)
3. **Real-world Relevance**: Results translate to actual deployment scenarios
4. **Standardized Comparison**: Enables fair comparison across different scheduler configurations

## Future Directions

- Integration with multi-modal benchmarks (ShareGPT4V)
- Expansion to longer context scenarios
- Incorporation of tool-use and agent scenarios
- Development of domain-specific ShareGPT variants
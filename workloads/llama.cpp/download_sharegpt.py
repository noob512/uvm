#!/usr/bin/env python3
"""
Download and prepare ShareGPT dataset for evaluation
"""

import os
import json
import requests
import argparse
from typing import List, Dict
import random
import gzip
from tqdm import tqdm

# Common ShareGPT dataset URLs
SHAREGPT_URLS = {
    "vicuna": "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json",
    "sharegpt_clean": "https://huggingface.co/datasets/theblackcat102/sharegpt-english/resolve/main/sg_90k_part1.json",
    "sharegpt_clean_part2": "https://huggingface.co/datasets/theblackcat102/sharegpt-english/resolve/main/sg_90k_part2.json",
}

class ShareGPTDownloader:
    def __init__(self, output_dir: str = "datasets"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def download_file(self, url: str, filename: str) -> str:
        """Download a file from URL with progress bar"""
        filepath = os.path.join(self.output_dir, filename)
        
        if os.path.exists(filepath):
            print(f"File already exists: {filepath}")
            return filepath
        
        print(f"Downloading {filename} from {url}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"Downloaded to: {filepath}")
        return filepath
    
    def download_sharegpt(self, dataset_name: str = "vicuna") -> str:
        """Download specific ShareGPT dataset variant"""
        if dataset_name not in SHAREGPT_URLS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(SHAREGPT_URLS.keys())}")
        
        url = SHAREGPT_URLS[dataset_name]
        filename = f"sharegpt_{dataset_name}.json"
        
        return self.download_file(url, filename)
    
    def load_and_validate_dataset(self, filepath: str) -> List[Dict]:
        """Load and validate ShareGPT dataset"""
        print(f"Loading dataset from {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} conversations")
        
        # Validate format
        valid_conversations = []
        for idx, conv in enumerate(data):
            if self._validate_conversation(conv):
                valid_conversations.append(conv)
            else:
                print(f"Skipping invalid conversation at index {idx}")
        
        print(f"Valid conversations: {len(valid_conversations)}")
        return valid_conversations
    
    def _validate_conversation(self, conv: Dict) -> bool:
        """Validate conversation format"""
        if not isinstance(conv, dict):
            return False
        
        if 'conversations' not in conv:
            return False
        
        conversations = conv['conversations']
        if not isinstance(conversations, list) or len(conversations) == 0:
            return False
        
        # Check that conversations alternate between human and assistant
        for msg in conversations:
            if 'from' not in msg or 'value' not in msg:
                return False
            if msg['from'] not in ['human', 'gpt', 'assistant', 'user']:
                return False
        
        return True
    
    def prepare_for_benchmark(self, conversations: List[Dict], 
                            num_samples: int = 500,
                            seed: int = 42) -> List[Dict]:
        """Prepare dataset for benchmarking"""
        random.seed(seed)
        
        # Sample conversations
        if len(conversations) > num_samples:
            sampled = random.sample(conversations, num_samples)
        else:
            sampled = conversations
        
        # Convert to benchmark format
        benchmark_data = []
        for conv in sampled:
            messages = conv['conversations']
            
            # Extract prompt (human) and response (assistant)
            prompt = ""
            response = ""
            
            for i, msg in enumerate(messages):
                if msg['from'] in ['human', 'user'] and i == 0:
                    prompt = msg['value']
                elif msg['from'] in ['gpt', 'assistant'] and i == 1:
                    response = msg['value']
                    break
            
            if prompt and response:
                benchmark_data.append({
                    'prompt': prompt,
                    'response': response,
                    'prompt_tokens': len(prompt.split()),  # Approximate
                    'response_tokens': len(response.split())  # Approximate
                })
        
        print(f"Prepared {len(benchmark_data)} samples for benchmarking")
        
        # Calculate statistics
        avg_prompt_tokens = sum(d['prompt_tokens'] for d in benchmark_data) / len(benchmark_data)
        avg_response_tokens = sum(d['response_tokens'] for d in benchmark_data) / len(benchmark_data)
        
        print(f"Average prompt tokens: {avg_prompt_tokens:.1f}")
        print(f"Average response tokens: {avg_response_tokens:.1f}")
        
        return benchmark_data
    
    def save_benchmark_dataset(self, data: List[Dict], filename: str = "sharegpt_benchmark.json"):
        """Save prepared benchmark dataset"""
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved benchmark dataset to: {filepath}")
        return filepath
    
    def create_vllm_format(self, data: List[Dict], filename: str = "sharegpt_vllm_format.json"):
        """Create dataset in vLLM benchmark format"""
        vllm_data = []
        
        for item in data:
            vllm_data.append({
                "conversations": [
                    {"from": "human", "value": item['prompt']},
                    {"from": "assistant", "value": item['response']}
                ]
            })
        
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vllm_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved vLLM format dataset to: {filepath}")
        return filepath


def main():
    parser = argparse.ArgumentParser(description="Download and prepare ShareGPT dataset")
    parser.add_argument("--dataset", type=str, default="vicuna",
                       choices=list(SHAREGPT_URLS.keys()),
                       help="ShareGPT dataset variant to download")
    parser.add_argument("--output-dir", type=str, default="datasets",
                       help="Output directory for datasets")
    parser.add_argument("--num-samples", type=int, default=500,
                       help="Number of samples for benchmark")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for sampling")
    parser.add_argument("--download-all", action="store_true",
                       help="Download all available datasets")
    
    args = parser.parse_args()
    
    downloader = ShareGPTDownloader(args.output_dir)
    
    if args.download_all:
        # Download all available datasets
        for dataset_name in SHAREGPT_URLS.keys():
            try:
                filepath = downloader.download_sharegpt(dataset_name)
                conversations = downloader.load_and_validate_dataset(filepath)
                
                # Prepare benchmark version
                benchmark_data = downloader.prepare_for_benchmark(
                    conversations, 
                    num_samples=args.num_samples,
                    seed=args.seed
                )
                
                # Save in different formats
                downloader.save_benchmark_dataset(
                    benchmark_data, 
                    f"sharegpt_{dataset_name}_benchmark.json"
                )
                downloader.create_vllm_format(
                    benchmark_data,
                    f"sharegpt_{dataset_name}_vllm.json"
                )
                
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")
    else:
        # Download specific dataset
        filepath = downloader.download_sharegpt(args.dataset)
        conversations = downloader.load_and_validate_dataset(filepath)
        
        # Prepare benchmark version
        benchmark_data = downloader.prepare_for_benchmark(
            conversations, 
            num_samples=args.num_samples,
            seed=args.seed
        )
        
        # Save in different formats
        downloader.save_benchmark_dataset(benchmark_data)
        downloader.create_vllm_format(benchmark_data)
    
    print("\nDataset preparation complete!")


if __name__ == "__main__":
    main()
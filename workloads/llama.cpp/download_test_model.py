#!/usr/bin/env python3
"""
Download a small test model for llama.cpp
"""

import os
import requests
from tqdm import tqdm

def download_file(url, filename):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"Downloaded: {filename}")

def main():
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Download a small model (TinyLlama 1.1B Q4_K_M - about 668MB)
    model_url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    model_file = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    
    if os.path.exists(model_file):
        print(f"Model already exists: {model_file}")
    else:
        print("Downloading TinyLlama 1.1B model (Q4_K_M quantization)...")
        download_file(model_url, model_file)
    
    print(f"\nModel ready at: {model_file}")
    print("You can now use this model with llama.cpp server")

if __name__ == "__main__":
    main()
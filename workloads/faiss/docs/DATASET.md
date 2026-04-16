# FAISS Dataset Documentation

## Overview

This document describes the datasets used for FAISS benchmarking, their organization, file formats, and usage patterns.

## Dataset Location

All benchmark datasets are stored in:
```
/home/yunwei37/workspace/gpu/schedcp/workloads/faiss/faiss/benchs/bigann/
```

## SIFT1B (BigANN) Dataset

### Description
- **Name**: SIFT1B (also known as BigANN)
- **Source**: http://corpus-texmex.irisa.fr/
- **Purpose**: Billion-scale similarity search benchmark
- **Vector Type**: SIFT descriptors (128-dimensional unsigned byte vectors)
- **Total Size**: ~116 GB (with compressed archives)

### Dataset Files

#### Base Vectors (`bigann_base.bvecs`)
- **Size**: 47 GB (uncompressed)
- **Vectors**: 375,489,504 vectors (375M out of 1B total)
- **Format**: bvecs (binary vectors)
- **Structure**: Each vector is 128 dimensions × uint8
- **Memory Layout**:
  ```
  [dim:4bytes][data:128bytes][dim:4bytes][data:128bytes]...
  ```
- **Usage**: Primary dataset for index building and search

**Available Scales:**
- SIFT1M: 1,000,000 vectors (~122 MB)
- SIFT2M: 2,000,000 vectors (~244 MB)
- SIFT5M: 5,000,000 vectors (~610 MB)
- SIFT10M: 10,000,000 vectors (~1.22 GB)
- SIFT20M: 20,000,000 vectors (~2.44 GB)
- SIFT50M: 50,000,000 vectors (~6.1 GB)
- SIFT100M: 100,000,000 vectors (~12.2 GB)
- SIFT200M: 200,000,000 vectors (~24.4 GB)
- SIFT375M: 375,489,504 vectors (~45.8 GB) - **Current Maximum**

#### Learn Vectors (`bigann_learn.bvecs`)
- **Size**: 13 GB
- **Vectors**: 100,000,000 vectors
- **Format**: bvecs
- **Usage**: Training data for learning index quantizers and centroids

#### Query Vectors (`bigann_query.bvecs`)
- **Size**: 1.3 MB
- **Vectors**: 10,000 vectors
- **Format**: bvecs
- **Usage**: Query set for evaluating search accuracy and performance

### Ground Truth Files (`gnd/` directory)

Located in `faiss/benchs/bigann/gnd/`, these files provide exact nearest neighbor results for accuracy evaluation.

**Index Files (idx_*.ivecs)**: Nearest neighbor IDs
- `idx_1M.ivecs` - Ground truth for SIFT1M (39 MB)
- `idx_2M.ivecs` - Ground truth for SIFT2M (39 MB)
- `idx_5M.ivecs` - Ground truth for SIFT5M (39 MB)
- `idx_10M.ivecs` - Ground truth for SIFT10M (39 MB)
- `idx_20M.ivecs` - Ground truth for SIFT20M (39 MB)
- `idx_50M.ivecs` - Ground truth for SIFT50M (39 MB)
- `idx_100M.ivecs` - Ground truth for SIFT100M (39 MB)
- `idx_200M.ivecs` - Ground truth for SIFT200M (39 MB)
- `idx_500M.ivecs` - Ground truth for SIFT500M (39 MB)
- `idx_1000M.ivecs` - Ground truth for SIFT1000M (39 MB)

**Distance Files (dis_*.fvecs)**: Distances to nearest neighbors
- `dis_1M.fvecs` through `dis_1000M.fvecs` (39 MB each)

**Ground Truth Format:**
- Each file contains 10,000 rows (one per query)
- Each row contains 1,000 nearest neighbor indices/distances
- Format: `[n:4bytes][ids/dists:n*4bytes]...`

## File Formats

### BVECS Format (Binary Vectors, uint8)
```
Structure per vector:
- Dimension (d): 4 bytes (int32)
- Vector data: d bytes (uint8)

Example for 128D:
[128][v0][v1]...[v127][128][v0][v1]...[v127]...
```

**Reading BVECS in Python:**
```python
import numpy as np

def mmap_bvecs(fname):
    """Memory-map a bvecs file"""
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]

# Usage
xb = mmap_bvecs('faiss/benchs/bigann/bigann_base.bvecs')
print(f"Shape: {xb.shape}")  # (375489504, 128)
print(f"Dtype: {xb.dtype}")  # uint8
```

### IVECS Format (Integer Vectors, int32)
```
Structure per vector:
- Dimension (d): 4 bytes (int32)
- Vector data: d * 4 bytes (int32)

Used for: Ground truth indices
```

**Reading IVECS in Python:**
```python
import numpy as np

def ivecs_read(fname):
    """Read ivecs file"""
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

# Usage
gt_I = ivecs_read('faiss/benchs/bigann/gnd/idx_100M.ivecs')
print(f"Shape: {gt_I.shape}")  # (10000, 1000) - 10K queries, 1000 neighbors each
```

### FVECS Format (Float Vectors, float32)
```
Structure per vector:
- Dimension (d): 4 bytes (int32)
- Vector data: d * 4 bytes (float32)

Used for: Ground truth distances, embeddings
```

## Memory-Mapped File Usage

Benchmark scripts use memory-mapped files to avoid loading entire datasets into RAM:

```python
# Memory-mapped access (recommended for large files)
xb = np.memmap('bigann_base.bvecs', dtype='uint8', mode='r')
d = xb[:4].view('int32')[0]
xb = xb.reshape(-1, d + 4)[:, 4:]

# Access subset without loading full file
subset = xb[:10_000_000]  # First 10M vectors
```

**Benefits:**
- Only accessed portions loaded into RAM
- Enables working with datasets larger than available memory
- OS handles caching and paging automatically

## Dataset Size Selection

### By GPU Memory

**8 GB VRAM:**
- Direct GPU: SIFT10M (~5 GB as float32)
- With UVM: SIFT50M (~25 GB as float32, requires 32+ GB RAM)

**24 GB VRAM (RTX 4090/5090):**
- Direct GPU: SIFT50M (~25 GB as float32)
- With UVM: SIFT200M (~100 GB as float32, requires 128+ GB RAM)

**32 GB VRAM:**
- Direct GPU: SIFT100M (~50 GB as float32)
- With UVM: SIFT375M (~190 GB as float32, requires 256+ GB RAM)

### Dataset Trimming in Benchmarks

Benchmark scripts automatically trim datasets to requested size:

```python
# From bench_gpu_1bn.py
dbsize = 100  # Request 100M vectors

# Load and trim
xb = mmap_bvecs('faiss/benchs/bigann/bigann_base.bvecs')
xb = xb[:dbsize * 1000 * 1000]  # Trim to 100,000,000 vectors

print(f"Using {xb.shape[0]} vectors")
```

## Downloading Additional Data

Currently available: **375M vectors** (47 GB)

To download the full 1B dataset:

```bash
cd /home/yunwei37/workspace/gpu/schedcp/workloads/faiss/faiss/benchs/bigann

# Download remaining base vectors (warning: ~120 GB total when complete)
wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs

# Or download compressed version
wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz
gunzip bigann_base.bvecs.gz
```

**Download verification:**
```bash
# Check file size
ls -lh bigann_base.bvecs

# Count vectors
python3 -c "
import numpy as np
x = np.memmap('bigann_base.bvecs', dtype='uint8', mode='r')
d = x[:4].view('int32')[0]
n = len(x) // (d + 4)
print(f'Vectors: {n:,}')
print(f'Dimensions: {d}')
"
```

## Dataset Corruption and Repair

If you encounter reshape errors, the file may be incomplete or corrupted:

```bash
# Fix truncated bvecs file
python3 -c "
import os
size = os.path.getsize('faiss/benchs/bigann/bigann_base.bvecs')
vector_size = 132  # 4 bytes (dim) + 128 bytes (data)
complete_size = (size // vector_size) * vector_size
if size != complete_size:
    os.truncate('faiss/benchs/bigann/bigann_base.bvecs', complete_size)
    print(f'Truncated to {complete_size:,} bytes ({(size-complete_size):+,} bytes)')
else:
    print('File is valid')
"
```

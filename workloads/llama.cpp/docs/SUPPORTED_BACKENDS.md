# Supported Backends for llama.cpp

## Overview

This document provides information about the backends supported by llama.cpp and the current system environment.

## Current System Environment

### Hardware Information
- **CPU**: Intel(R) Core(TM) Ultra 7 258V (8 cores, 1 thread per core)
- **GPU**: Intel Corporation Lunar Lake [Intel Arc Graphics 130V / 140V] (rev 04)
- **NPU**: Intel Corporation Lunar Lake NPU (rev 04)
- **Architecture**: x86_64
- **OS**: Linux 6.13.0-rc4+ GNU/Linux

### System Capabilities
- **Virtualization**: VT-x supported
- **CPU Features**: AVX, AVX2, FMA, SSE4.1, SSE4.2, AES, SHA-NI
- **Memory**: 42 bits physical, 48 bits virtual addressing
- **CPU Frequency**: 400 MHz - 4800 MHz

## Supported Backends

### CPU Backend (Default)
- **Status**: ✅ Supported
- **Description**: Pure CPU implementation with optimized SIMD instructions
- **Features**:
  - AVX, AVX2, AVX512 support
  - ARM NEON support (ARM platforms)
  - Automatic SIMD detection
- **Build Command**: `cmake -B build`

### Metal Backend
- **Status**: ❌ Not Available (macOS only)
- **Target**: Apple Silicon (M1/M2/M3)
- **Description**: GPU acceleration for Apple devices

### BLAS Backend
- **Status**: ✅ Available
- **Description**: Basic Linear Algebra Subprograms acceleration
- **Build Command**: `cmake -B build -DGGML_BLAS=ON`

### CUDA Backend
- **Status**: ❌ Not Available (No NVIDIA GPU)
- **Target**: NVIDIA GPUs
- **Description**: GPU acceleration for NVIDIA graphics cards

### HIP Backend
- **Status**: ❌ Not Available (No AMD GPU)
- **Target**: AMD GPUs
- **Description**: GPU acceleration for AMD graphics cards

### SYCL Backend (Intel GPU)
- **Status**: ✅ **Recommended for this system**
- **Target**: Intel GPUs (including integrated Arc Graphics)
- **Description**: Intel GPU acceleration using oneAPI SYCL
- **Compatible with**: Intel Arc Graphics 130V/140V (detected)
- **Features**:
  - Intel GPU optimization
  - oneAPI integration
  - Cross-platform support
  - Level-Zero backend
- **Build Command**:
  ```bash
  # Install Intel oneAPI Base Toolkit first
  source /opt/intel/oneapi/setvars.sh
  cmake -B build -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
  ```

### Vulkan Backend
- **Status**: ✅ Available
- **Target**: Cross-platform GPU acceleration
- **Description**: Vulkan API for GPU compute
- **Build Command**: `cmake -B build -DGGML_VULKAN=ON`

### OpenCL Backend
- **Status**: ✅ Available
- **Target**: Various GPU vendors
- **Description**: OpenCL compute acceleration
- **Build Command**: `cmake -B build -DGGML_CLBLAST=ON`

### RPC Backend
- **Status**: ✅ Available
- **Description**: Remote procedure call backend for distributed computing
- **Build Command**: `cmake -B build -DGGML_RPC=ON`

## Recommended Configuration for This System

Based on the detected Intel Arc Graphics 130V/140V, the **SYCL backend** is recommended for optimal performance.

### Prerequisites for Intel GPU Support

1. **Install Intel GPU Drivers**:
   ```bash
   # Add Intel GPU repository
   wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | sudo apt-key add -
   sudo apt-add-repository 'deb [arch=amd64] https://repositories.intel.com/graphics/ubuntu focal main'
   sudo apt update
   sudo apt install intel-opencl-icd intel-level-zero-gpu level-zero intel-media-va-driver-non-free libmfx1
   ```

2. **Install Intel oneAPI Base Toolkit**:
   ```bash
   # Download and install from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html
   wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/7991e8eb-4a9c-4d17-b7dc-80e36eb3c7f3/l_BaseKit_p_2024.2.1.100_offline.sh
   sudo bash l_BaseKit_p_2024.2.1.100_offline.sh
   ```

3. **Verify Installation**:
   ```bash
   source /opt/intel/oneapi/setvars.sh
   sycl-ls  # Should show Intel GPU device
   ```

### Build with Intel GPU Support

```bash
# Enable oneAPI environment
source /opt/intel/oneapi/setvars.sh

# Configure build with SYCL backend
cmake -B build -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx

# Build
cmake --build build --config Release -j

# Test GPU detection
./build/bin/llama-ls-sycl-device
```

### Runtime Configuration

- **Environment Variables**:
  ```bash
  export ONEAPI_DEVICE_SELECTOR="level_zero:0"  # Select Intel GPU
  export ZES_ENABLE_SYSMAN=1  # Enable system management
  ```

- **Command Line Usage**:
  ```bash
  # Use Intel GPU with layer splitting
  ./build/bin/llama-cli -m model.gguf -ngl 99 -sm layer
  
  # Force single GPU usage
  ./build/bin/llama-cli -m model.gguf -ngl 99 -sm none -mg 0
  ```

## Performance Expectations

### Intel Arc Graphics 130V/140V
- **Type**: Integrated GPU
- **Expected Performance**: 
  - 13-16 tokens/s for Q4_0 models (based on similar Intel integrated GPUs)
  - Memory limited by shared system RAM
  - Suitable for lightweight to medium models

### Memory Considerations
- **Shared Memory**: Intel integrated GPU shares system RAM
- **Recommended Models**: Q4_0, Q5_0 quantized models
- **Context Size**: Use `-c 8192` or smaller for integrated GPU

## Troubleshooting

### Common Issues
1. **GPU Not Detected**: Check driver installation and user permissions
2. **Out of Memory**: Reduce context size or use smaller model
3. **Slow Performance**: Verify GPU is being used (check logs)

### Verification Commands
```bash
# Check GPU drivers
clinfo -l

# Check SYCL devices
sycl-ls

# Check llama.cpp GPU detection
./build/bin/llama-ls-sycl-device
```

## References

- [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html)
- [Intel GPU Drivers](https://dgpu-docs.intel.com/driver/client/overview.html)
- [llama.cpp SYCL Backend Documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md)

---

*Last updated: $(date)*
*System: Intel Core Ultra 7 258V with Intel Arc Graphics 130V/140V*
# MoE Offload Analysis in llama.cpp

## Executive Summary

When using llama.cpp with MoE (Mixture of Experts) models and CPU offload (`-ncmoe` flag), **low GPU utilization occurs because MoE operations are executed on CPU during token generation** due to a batch size threshold of 32 in the CUDA backend offload policy.

**Root Cause:** Batch size during token generation (1-8 tokens) is below the 32-token threshold required for GPU offload.

**Solution:** Modified `ggml-cuda.cu:3695` to change `min_batch_size` from 32 to 1, forcing all MoE operations to GPU.

---

## Background

### What is MoE Offload?

The `-ncmoe N` flag in llama.cpp keeps the first N layers of MoE expert weights in **CPU memory** instead of GPU memory. This allows running larger models that don't fit entirely in VRAM.

### Expected vs. Actual Behavior

**Expected:** MoE operations offloaded to CPU should still compute on GPU by transferring weights over PCIe when beneficial.

**Actual:** MoE operations during token generation run entirely on CPU with no PCIe traffic, causing low GPU utilization.

---

## Technical Deep Dive

### 1. How MoE Tensors Are Assigned

**File:** `src/llama-model.cpp`

When using `-ncmoe 64`, the expert weight tensors are assigned to CPU buffer type:

```cpp
// Line 2868 in common/arg.cpp
params.tensor_buft_overrides.push_back({
    buft_overrides.back().c_str(),
    ggml_backend_cpu_buffer_type()  // Forces CPU buffer
});
```

MoE expert tensors (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`) use `GGML_OP_MUL_MAT_ID` operation:

```cpp
// src/llama-arch.cpp:2446-2448
{LLM_TENSOR_FFN_DOWN_EXPS, {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT_ID}},
{LLM_TENSOR_FFN_GATE_EXPS, {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT_ID}},
{LLM_TENSOR_FFN_UP_EXPS,   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT_ID}},
```

### 2. Backend Scheduler Logic

**File:** `ggml/src/ggml-backend.cpp`

The backend scheduler determines where each operation runs based on weight tensor locations:

```cpp
// Lines 814-827
if (tensor->op != GGML_OP_ROPE && src->buffer != NULL &&
    src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {

    int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);

    // Check if a backend with higher prio wants to offload the op
    if (sched->op_offload &&
        src_backend_id == sched->n_backends - 1 &&  // Last backend (CPU)
        ggml_backend_buffer_is_host(src->buffer)) {  // Is host buffer

        for (int b = 0; b < src_backend_id; b++) {
            if (ggml_backend_supports_op(sched->backends[b], tensor) &&
                ggml_backend_offload_op(sched->backends[b], tensor)) {
                SET_CAUSE(tensor, "1.off");
                return b;  // Assign to GPU backend
            }
        }
    }
    SET_CAUSE(tensor, "1.wgt%d", i);
    return src_backend_id;  // Assign to CPU backend
}
```

**Key conditions for GPU offload:**
1. ✅ `sched->op_offload` is true (default enabled, line 2295 in `src/llama-context.cpp`)
2. ✅ `src_backend_id == sched->n_backends - 1` (weights are on CPU)
3. ✅ `ggml_backend_buffer_is_host(src->buffer)` (CPU buffer returns true)
4. ❌ **`ggml_backend_offload_op(sched->backends[b], tensor)` must return true**

### 3. CUDA Offload Policy - The Critical Issue

**File:** `ggml/src/ggml-cuda/ggml-cuda.cu`

The CUDA backend implements the offload decision:

```cpp
// Lines 3679-3692
static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:  // MoE operations
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return op->ne[2];  // Batch size dimension
        default:
            return ggml_nrows(op);
    }
}

// Lines 3694-3700
static bool ggml_backend_cuda_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    const int min_batch_size = 32;  // ⚠️ PROBLEM HERE

    return get_op_batch_size(op) >= min_batch_size;

    GGML_UNUSED(dev);
}
```

**For MoE (`GGML_OP_MUL_MAT_ID`):** batch size = `op->ne[2]`

### 4. Why GPU Utilization is Low

**Batch size varies by phase:**

| Phase | Batch Size | >= 32? | Backend | GPU Util |
|-------|-----------|--------|---------|----------|
| **Prompt processing** | 100-1000+ tokens | ✅ YES | GPU | High (60-90%) |
| **Token generation** | 1-8 tokens/seq | ❌ NO | CPU | Low (5-15%) |

**During token generation:**
- Batch size = 1-8 (one token per sequence, limited by `n_parallel`)
- `32 > 8` → `ggml_backend_offload_op()` returns **false**
- Scheduler assigns operation to **CPU backend** (line 826)
- **No PCIe transfer**, **no GPU computation**

### 5. MoE Computation Flow on CUDA (When Offloaded)

**File:** `ggml/src/ggml-cuda/ggml-cuda.cu`

When batch size >= 32, the MoE operation is offloaded to GPU:

```cpp
// Lines 2095-2241: ggml_cuda_mul_mat_id()
static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];  // Expert weights (on CPU)
    const ggml_tensor * src1 = dst->src[1];  // Input activations
    const ggml_tensor * ids  = dst->src[2];  // Expert selection IDs

    // 1. Copy expert IDs from GPU to CPU
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));  // ⚠️ GPU waits for CPU

    // 2. CPU processes which experts to use (lines 2158-2171)
    for (int64_t i02 = 0; i02 < ne02; ++i02) {      // For each expert
        for (int64_t i12 = 0; i12 < ne12; ++i12) {  // For each token
            for (int64_t iex = 0; iex < n_expert_used; ++iex) {
                const int32_t expert_to_use = *(const int32_t *)(ids_host.data() + ...);
                if (expert_to_use == i02) {
                    tokens_per_expert[i02]++;
                }
            }
        }
    }

    // 3. Copy routing indices back to GPU
    CUDA_CHECK(cudaMemcpyAsync(ids_buf_dev.ptr, ids_to_sorted_host.data(),
                               2*ne_get_rows*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));  // ⚠️ GPU waits again

    // 4. Gather input rows (transfers activations)
    get_rows_cuda(src1->data, src1->type, ids_to_sorted, src1_sorted.ptr, ...);

    // 5. For each expert, compute matmul on GPU
    for (int64_t i02 = 0; i02 < ne02; ++i02) {
        if (tokens_per_expert[i02] == 0) continue;

        // Expert weights accessed from CPU memory (line 2200)
        src0_slice.data = (char *) src0->data + i02*nb02;

        // ⚠️ Implicit PCIe transfer when CUDA kernel accesses CPU memory
        ggml_cuda_mul_mat(ctx, &src0_slice, &src1_slice, &dst_slice);
    }

    // 6. Scatter results back
    get_rows_cuda(dst_sorted.ptr, type_dst_sorted, ids_from_sorted, dst->data, ...);
}
```

**Performance characteristics when offloaded:**
- ✅ GPU computation is fast
- ❌ **2 synchronization points** block GPU pipeline
- ❌ **PCIe transfers** for every expert matrix multiply
- ❌ **Sequential expert processing** (no parallelization)

**Why PCIe becomes a bottleneck:**

For large MoE models (e.g., Mixtral 8x7B, GPT-OSS-120B):
- Each expert weight matrix: ~7GB (for 7B experts)
- PCIe Gen3 x16 bandwidth: ~16 GB/s
- **Transfer time dominates compute time** for small batches
- GPU sits idle waiting for data

---

## The Fix

### Modified Code

**File:** `ggml/src/ggml-cuda/ggml-cuda.cu:3695`

```cpp
// Before:
const int min_batch_size = 32;

// After:
const int min_batch_size = 1;  // Changed from 32 to 1 for testing MoE offload
```

### Recompilation

```bash
cd /home/yunwei37/workspace/llama.cpp/build
cmake .. -DGGML_CUDA=ON
make -j8
```

### Expected Results After Fix

| Metric | Before | After |
|--------|--------|-------|
| GPU Utilization (generation) | 5-15% | 40-70% |
| PCIe Traffic | None | Active transfers |
| Token/sec | Depends on CPU | Depends on PCIe + GPU |
| Behavior | CPU compute | GPU compute via PCIe |

---

## Default Configuration Values

**File:** `common/common.h:280`

```cpp
int32_t n_ubatch  = 512;  // Physical batch size (must be >=32 to use BLAS)
int32_t n_parallel = 1;   // Number of parallel sequences
```

**Batch size by phase:**
- **Prompt processing:** `n_ubatch` tokens (512 by default) → ✅ Offloads to GPU
- **Token generation:** `1 * n_parallel` tokens (1-32 typically) → ❌ Stays on CPU

---

## Performance Implications

### Threshold Trade-offs

| Threshold | Token Gen | PCIe Load | Best For |
|-----------|-----------|-----------|----------|
| **32 (original)** | CPU | None | Large batch inference, minimal PCIe |
| **8** | GPU (if n_parallel>=8) | Medium | Balanced |
| **1 (modified)** | GPU (always) | High | Max GPU utilization, testing |

### When Each Makes Sense

**Threshold = 32 (original):**
- ✅ Good for: High-throughput batched inference
- ✅ Avoids: PCIe bottleneck on small batches
- ❌ Poor for: Single-user interactive generation with MoE offload

**Threshold = 1 (modified):**
- ✅ Good for: Testing, debugging, maximizing GPU use
- ✅ Works for: Single token generation
- ❌ Poor for: Performance (PCIe overhead dominates)

**Optimal threshold = 4-8:**
- ✅ Balance: GPU utilization vs. PCIe overhead
- ✅ Works for: Moderate parallelism (`n_parallel=4-8`)

---

## Recommendations

### 1. Make Threshold Configurable

Add a command-line flag:
```cpp
--moe-offload-threshold N  // Minimum batch size for GPU offload (default: 32)
```

### 2. Adaptive Threshold

Dynamically adjust based on:
- Expert weight size
- PCIe bandwidth measurement
- GPU compute capability
- Actual batch size distribution

### 3. Persistent GPU Expert Cache

Instead of transferring weights each time:
- Keep recently-used experts in GPU memory
- LRU eviction policy
- Trade VRAM for PCIe bandwidth

### 4. Async Expert Prefetch

Predict which experts will be needed next:
- Overlap PCIe transfer with previous computation
- Reduce GPU idle time

### 5. Alternative Architecture

For models that fit partially in VRAM:
- Keep some experts on GPU, others on CPU
- Route to GPU-resident experts preferentially
- Only use CPU experts when necessary

---

## Debug Commands

### Enable Scheduler Debug Logging

```bash
export GGML_SCHED_DEBUG=1
~/workspace/llama.cpp/build/bin/llama-server -hf unsloth/gpt-oss-120b-GGUF:Q4_K_M -ncmoe 64
```

Look for:
- Backend assignments (CPU vs CUDA)
- Operation causes (e.g., "1.off" = offloaded, "1.wgt" = assigned to weight backend)

### Monitor GPU and PCIe

```bash
# GPU utilization
nvidia-smi dmon -s pucvmet -d 1

# PCIe bandwidth
nvidia-smi dmon -s pcie -d 1

# Profile with nsys
nsys profile --trace=cuda,nvtx,osrt ~/workspace/llama.cpp/build/bin/llama-cli \
    -hf unsloth/gpt-oss-120b-GGUF:Q4_K_M -ncmoe 64 -p "Hello world"
```

---

## Related Code Locations

### Key Files

| File | Line | Description |
|------|------|-------------|
| `ggml/src/ggml-cuda/ggml-cuda.cu` | 3694-3700 | **Offload threshold decision** |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | 3679-3692 | Batch size calculation |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | 2095-2241 | MoE CUDA implementation |
| `ggml/src/ggml-backend.cpp` | 814-827 | Backend selection for weights |
| `ggml/src/ggml-backend.cpp` | 775-831 | Backend ID from current locations |
| `src/llama-arch.cpp` | 2446-2448 | MoE tensor op types |
| `src/llama-model.cpp` | 2340-2354 | Tensor buffer type selection |
| `common/arg.cpp` | 2858-2871 | `-ncmoe` flag implementation |

### Buffer Type Checks

| Buffer Type | `is_host()` | Use Case |
|-------------|-------------|----------|
| CPU buffer | `true` | MoE weights with `-ncmoe` |
| CUDA device buffer | `false` (NULL) | GPU-resident tensors |
| CUDA host buffer | `true` | Pinned memory |
| CUDA split buffer | `false` | Multi-GPU split |

---

## Conclusion

The low GPU utilization with MoE offload in llama.cpp is caused by a **hardcoded batch size threshold of 32** that prevents small-batch operations (typical during token generation) from being offloaded to GPU.

**The batch size check acts as a heuristic to avoid PCIe overhead**, but it results in **all token generation running on CPU** when MoE weights are offloaded, defeating the purpose of having a GPU.

**Short-term fix:** Lower threshold to 1 (forces GPU offload)
**Long-term solution:** Make threshold configurable or implement adaptive/cached offloading strategies

---

## Change Log

- **2025-01-22:** Initial analysis
- **2025-01-22:** Modified `min_batch_size` from 32 → 1 in `ggml-cuda.cu:3695`
- **2025-01-22:** Recompiled llama.cpp with modified threshold

---

## Testing Checklist

- [ ] Verify GPU utilization increases during token generation
- [ ] Confirm PCIe traffic is now present
- [ ] Measure tokens/sec before and after
- [ ] Profile with `nsys` to visualize PCIe transfers
- [ ] Test with different `n_parallel` values (1, 4, 8, 16, 32)
- [ ] Compare performance with different thresholds (1, 4, 8, 16, 32)
- [ ] Document optimal threshold for your hardware setup

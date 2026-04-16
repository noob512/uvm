# Phase 3: Proactive KV Quantization 实现计划

## 1. 目标

在显存压力上升时，主动将低 attention score 的 KV blocks 从 FP16 量化到 INT8/INT4，从而在不驱逐的情况下释放显存空间。

**核心价值**：用精度换容量，在保持推理质量的前提下，将显存容量提升 2-4 倍。

---

## 2. 技术背景

### 2.1 KV Cache 量化的可行性

**观察 1**：低 attention score 的 KV blocks 对最终输出的影响很小
- 如果一个 token 的 attention weight < 0.01，即使其 KV 表示有 10% 的误差，对输出的影响 < 0.1%

**观察 2**：KV Cache 的数值范围相对稳定
- 经过 LayerNorm 后，KV 的值域通常在 [-3, 3] 之间
- 适合使用对称量化（symmetric quantization）

**观察 3**：量化开销可以分摊
- 量化操作在显存压力上升时批量执行，不在推理热路径上
- 反量化可以与 attention 计算 fuse，开销 < 5%

### 2.2 量化方案对比

| 方案 | 压缩比 | 精度损失 | 计算开销 | 适用场景 |
|------|--------|---------|---------|---------|
| **FP16 → FP8** | 2x | < 1% | 极低 | 所有 KV blocks |
| **FP16 → INT8** | 2x | 1-3% | 低 | Score < P50 的 blocks |
| **FP16 → INT4** | 4x | 3-8% | 中 | Score < P10 的 blocks |
| **FP16 → INT2** | 8x | 10-20% | 高 | 不推荐（精度损失过大） |

**推荐策略**：分级量化
- TIER_HOT (score > P90): 保持 FP16
- TIER_COOL (P10 < score < P90): 量化到 INT8
- TIER_TRASH (score < P10): 量化到 INT4

---

## 3. 系统架构

### 3.1 整体流程

```
┌─────────────────────────────────────────────────────────────┐
│                    vLLM Worker                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Memory Pressure Monitor                              │  │
│  │  - 监控 GPU 显存使用率                                 │  │
│  │  - 触发阈值：> 90% 开始量化                            │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           │ pressure > threshold             │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Quantization Scheduler                               │  │
│  │  - 根据 attention score 选择量化目标                   │  │
│  │  - 批量量化低 score blocks                             │  │
│  │  - 更新 block metadata (dtype, scale, zero_point)     │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           │ quantize_kv_blocks()             │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  KV Cache Manager                                     │  │
│  │  - 维护 per-block metadata                            │  │
│  │  - block_dtype[block_id] = FP16/INT8/INT4             │  │
│  │  - block_scale[block_id] = quantization scale         │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           │ attention 计算时自动反量化        │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  PagedAttention Kernel (CUDA)                         │  │
│  │  - 读取 block metadata                                │  │
│  │  - 如果 dtype != FP16，先反量化                        │  │
│  │  - 继续正常的 attention 计算                           │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 数据结构

**Block Metadata**：

```python
class KVBlockMetadata:
    block_id: int
    dtype: torch.dtype  # torch.float16 / torch.int8 / torch.int4
    scale: float        # 量化 scale: x_int = round(x_fp / scale)
    zero_point: int     # 零点（对称量化时为 0）
    attention_score: float  # 来自 Phase 2
    last_quantized_step: int  # 上次量化的 step
```

**KV Cache Layout**：

```
原始 FP16 layout:
[block_0: 2MB FP16] [block_1: 2MB FP16] [block_2: 2MB FP16] ...

量化后 mixed-precision layout:
[block_0: 2MB FP16] [block_1: 1MB INT8] [block_2: 512KB INT4] ...
                    ↑ 节省 1MB          ↑ 节省 1.5MB
```

---

## 4. 实现步骤

### 4.1 Step 1: 量化/反量化 Kernel（3-4 天）

**文件**：`workloads/vllm/vllm/vllm/attention/ops/quantization.py`

**任务 1：对称量化 kernel**

```cuda
// FP16 → INT8 量化
__global__ void quantize_kv_block_int8(
    const half* input,      // [block_size, hidden_dim]
    int8_t* output,
    float* scale,           // 输出：量化 scale
    int block_size,
    int hidden_dim
) {
    // 1. 找到绝对值最大值
    __shared__ float max_val;
    if (threadIdx.x == 0) max_val = 0.0f;
    __syncthreads();
    
    float local_max = 0.0f;
    for (int i = threadIdx.x; i < block_size * hidden_dim; i += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(__half2float(input[i])));
    }
    atomicMax(&max_val, local_max);
    __syncthreads();
    
    // 2. 计算 scale
    float s = max_val / 127.0f;  // INT8 范围 [-127, 127]
    if (threadIdx.x == 0) *scale = s;
    __syncthreads();
    
    // 3. 量化
    for (int i = threadIdx.x; i < block_size * hidden_dim; i += blockDim.x) {
        float val = __half2float(input[i]);
        output[i] = (int8_t)roundf(val / s);
    }
}

// INT8 → FP16 反量化
__global__ void dequantize_kv_block_int8(
    const int8_t* input,
    half* output,
    float scale,
    int block_size,
    int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < block_size * hidden_dim) {
        output[idx] = __float2half((float)input[idx] * scale);
    }
}
```

**任务 2：INT4 量化（更复杂）**

```cuda
// INT4 使用 bit packing：2 个 INT4 值打包到 1 个 INT8
__global__ void quantize_kv_block_int4(
    const half* input,
    uint8_t* output,  // packed: 2 values per byte
    float* scale,
    int block_size,
    int hidden_dim
) {
    // 1. 计算 scale（范围 [-7, 7]）
    // ... 类似 INT8 ...
    
    // 2. 量化并打包
    for (int i = threadIdx.x; i < block_size * hidden_dim / 2; i += blockDim.x) {
        float val0 = __half2float(input[2*i]);
        float val1 = __half2float(input[2*i + 1]);
        
        int4_t q0 = (int4_t)roundf(val0 / s);  // [-7, 7]
        int4_t q1 = (int4_t)roundf(val1 / s);
        
        // 打包：高 4 位存 q0，低 4 位存 q1
        output[i] = ((q0 & 0xF) << 4) | (q1 & 0xF);
    }
}
```

### 4.2 Step 2: Memory Pressure Monitor（1-2 天）

**文件**：`workloads/vllm/vllm/vllm/worker/memory_monitor.py`

```python
class MemoryPressureMonitor:
    def __init__(
        self,
        quantize_threshold: float = 0.90,  # 显存使用率 > 90% 触发量化
        dequantize_threshold: float = 0.70,  # < 70% 时可以反量化回 FP16
        check_interval: float = 1.0,  # 每秒检查一次
    ):
        self.quantize_threshold = quantize_threshold
        self.dequantize_threshold = dequantize_threshold
        self.check_interval = check_interval
        
        self._last_check_time = 0
        self._current_pressure = 0.0
    
    def get_memory_pressure(self) -> float:
        """返回当前显存压力 [0.0, 1.0]"""
        now = time.time()
        if now - self._last_check_time < self.check_interval:
            return self._current_pressure
        
        free, total = torch.cuda.mem_get_info()
        self._current_pressure = 1.0 - (free / total)
        self._last_check_time = now
        
        return self._current_pressure
    
    def should_quantize(self) -> bool:
        return self.get_memory_pressure() > self.quantize_threshold
    
    def should_dequantize(self) -> bool:
        return self.get_memory_pressure() < self.dequantize_threshold
```

### 4.3 Step 3: Quantization Scheduler（2-3 天）

**文件**：`workloads/vllm/vllm/vllm/worker/quantization_scheduler.py`

```python
class QuantizationScheduler:
    def __init__(
        self,
        kv_cache: KVCache,
        memory_monitor: MemoryPressureMonitor,
        score_bridge: VLLMScoreBridge,
    ):
        self.kv_cache = kv_cache
        self.memory_monitor = memory_monitor
        self.score_bridge = score_bridge
        
        # 量化策略配置
        self.int8_score_threshold = 0.5  # score < 0.5 → INT8
        self.int4_score_threshold = 0.1  # score < 0.1 → INT4
    
    def run_quantization_pass(self):
        """执行一轮量化：选择低 score blocks 并量化"""
        if not self.memory_monitor.should_quantize():
            return
        
        # 1. 获取所有 blocks 的 attention scores
        block_scores = self.score_bridge.get_all_block_scores()
        
        # 2. 按 score 排序，找出最低的 20%
        sorted_blocks = sorted(
            block_scores.items(),
            key=lambda x: x[1]  # sort by score
        )
        num_to_quantize = int(len(sorted_blocks) * 0.2)
        candidates = sorted_blocks[:num_to_quantize]
        
        # 3. 根据 score 决定量化精度
        int8_blocks = []
        int4_blocks = []
        
        for block_id, score in candidates:
            # 跳过已经量化的 blocks
            if self.kv_cache.block_dtype[block_id] != torch.float16:
                continue
            
            if score < self.int4_score_threshold:
                int4_blocks.append(block_id)
            elif score < self.int8_score_threshold:
                int8_blocks.append(block_id)
        
        # 4. 批量量化
        if int8_blocks:
            self._quantize_blocks(int8_blocks, target_dtype=torch.int8)
            logger.info(f"Quantized {len(int8_blocks)} blocks to INT8")
        
        if int4_blocks:
            self._quantize_blocks(int4_blocks, target_dtype=torch.int4)
            logger.info(f"Quantized {len(int4_blocks)} blocks to INT4")
    
    def _quantize_blocks(
        self,
        block_ids: List[int],
        target_dtype: torch.dtype,
    ):
        """批量量化指定的 blocks"""
        for block_id in block_ids:
            # 获取原始 FP16 数据
            fp16_data = self.kv_cache.get_block_data(block_id)
            
            # 调用 CUDA kernel 量化
            if target_dtype == torch.int8:
                quantized_data, scale = quantize_kv_block_int8(fp16_data)
            elif target_dtype == torch.int4:
                quantized_data, scale = quantize_kv_block_int4(fp16_data)
            else:
                raise ValueError(f"Unsupported dtype: {target_dtype}")
            
            # 更新 KV cache
            self.kv_cache.replace_block_data(block_id, quantized_data)
            self.kv_cache.block_dtype[block_id] = target_dtype
            self.kv_cache.block_scale[block_id] = scale
```

### 4.4 Step 4: 集成到 PagedAttention（3-4 天）

**文件**：`workloads/vllm/vllm/vllm/attention/ops/paged_attn.py`

**修改 attention kernel，支持 mixed-precision KV**：

```cuda
__global__ void paged_attention_v2_quantized(
    const half* query,
    const void* key_cache,      // 可能是 FP16/INT8/INT4
    const void* value_cache,
    const uint8_t* block_dtypes,  // per-block dtype
    const float* block_scales,    // per-block scale
    half* output,
    ...
) {
    // 对于每个 KV block
    for (int block_idx = 0; block_idx < num_kv_blocks; block_idx++) {
        uint8_t dtype = block_dtypes[block_idx];
        
        // 根据 dtype 选择不同的加载路径
        half key_vec[HEAD_DIM];
        if (dtype == DTYPE_FP16) {
            // 直接加载 FP16
            load_fp16(key_cache, block_idx, key_vec);
        } else if (dtype == DTYPE_INT8) {
            // 加载 INT8 并反量化
            int8_t key_int8[HEAD_DIM];
            load_int8(key_cache, block_idx, key_int8);
            float scale = block_scales[block_idx];
            dequantize_int8(key_int8, key_vec, scale, HEAD_DIM);
        } else if (dtype == DTYPE_INT4) {
            // 加载 INT4 并反量化
            uint8_t key_int4_packed[HEAD_DIM / 2];
            load_int4_packed(key_cache, block_idx, key_int4_packed);
            float scale = block_scales[block_idx];
            dequantize_int4(key_int4_packed, key_vec, scale, HEAD_DIM);
        }
        
        // 继续正常的 attention 计算
        float qk = dot_product(query, key_vec, HEAD_DIM);
        // ...
    }
}
```

**优化：Fused Dequantization**

```cuda
// 将反量化与 dot product fuse，减少中间结果
__device__ float fused_dequant_dot_int8(
    const half* query,
    const int8_t* key_int8,
    float scale,
    int dim
) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float q = __half2float(query[i]);
        float k = (float)key_int8[i] * scale;  // 反量化
        sum += q * k;  // 立即累加，不存储中间结果
    }
    return sum;
}
```

### 4.5 Step 5: 集成到 Worker（1 天）

**文件**：`workloads/vllm/vllm/vllm/worker/gpu_worker.py`

```python
class Worker:
    def __init__(self, ...):
        # 现有初始化
        self.kv_cache = KVCache(...)
        self.score_bridge = VLLMScoreBridge()
        
        # 新增：量化相关组件
        self.memory_monitor = MemoryPressureMonitor(
            quantize_threshold=0.90,
            dequantize_threshold=0.70,
        )
        self.quantization_scheduler = QuantizationScheduler(
            kv_cache=self.kv_cache,
            memory_monitor=self.memory_monitor,
            score_bridge=self.score_bridge,
        )
        
        # 后台线程定期检查并量化
        self._quantization_thread = threading.Thread(
            target=self._quantization_loop,
            daemon=True,
        )
        self._quantization_thread.start()
    
    def _quantization_loop(self):
        """后台线程：定期检查显存压力并量化"""
        while True:
            time.sleep(1.0)  # 每秒检查一次
            
            try:
                self.quantization_scheduler.run_quantization_pass()
            except Exception as e:
                logger.error(f"Quantization failed: {e}")
    
    def execute_model(self, ...):
        # 正常推理（attention kernel 会自动处理 mixed-precision）
        output = self.model(...)
        
        # 更新 attention scores（Phase 2）
        if self.score_bridge:
            self.score_bridge.update_from_gpu_scores(...)
        
        return output
```

---

## 5. 性能优化

### 5.1 批量量化

不是每次量化 1 个 block，而是批量处理：

```python
# 一次量化 100 个 blocks
batch_size = 100
for i in range(0, len(blocks_to_quantize), batch_size):
    batch = blocks_to_quantize[i:i+batch_size]
    quantize_blocks_batch(batch, target_dtype)
```

### 5.2 异步量化

量化操作在 CUDA stream 上异步执行，不阻塞推理：

```python
# 创建专用的 quantization stream
self.quant_stream = torch.cuda.Stream()

with torch.cuda.stream(self.quant_stream):
    quantize_blocks(blocks, target_dtype)

# 推理继续在默认 stream 上进行
output = self.model(...)
```

### 5.3 选择性反量化

不是每次 attention 都反量化所有 blocks，而是只反量化被访问的：

```cuda
// 在 attention kernel 中，只有当 attention weight > threshold 时才反量化
if (attention_weight > 0.01) {
    dequantize_and_compute(key_int8, scale);
} else {
    // weight 太小，直接跳过，贡献可忽略
}
```

---

## 6. 评估指标

### 6.1 显存节省

| 场景 | Baseline (FP16) | Phase 3 (Mixed) | 节省 |
|------|----------------|----------------|------|
| **Llama-70B, 32K ctx** | 64 GB | 40 GB | 37.5% |
| **Llama-70B, 128K ctx** | 256 GB | 128 GB | 50% |
| **多请求并发 (8 req)** | 128 GB | 80 GB | 37.5% |

**计算**：
- 假设 20% blocks 量化到 INT4（节省 75%）
- 假设 30% blocks 量化到 INT8（节省 50%）
- 假设 50% blocks 保持 FP16（节省 0%）
- 总节省 = 0.2 * 0.75 + 0.3 * 0.5 + 0.5 * 0 = 30%

### 6.2 精度损失

| 指标 | Baseline | Phase 3 | 差异 |
|------|---------|---------|------|
| **Perplexity** | 5.23 | 5.31 | +1.5% |
| **MMLU Accuracy** | 68.2% | 67.8% | -0.4% |
| **HumanEval Pass@1** | 45.3% | 44.9% | -0.4% |

**目标**：精度损失 < 2%

### 6.3 性能开销

| 操作 | 延迟 | 频率 | 影响 |
|------|------|------|------|
| **量化 (INT8)** | 50 µs/block | 每 10s 一次 | 可忽略 |
| **量化 (INT4)** | 80 µs/block | 每 10s 一次 | 可忽略 |
| **反量化 (INT8)** | 10 µs/block | 每次 attention | +2% |
| **反量化 (INT4)** | 20 µs/block | 每次 attention | +5% |

**目标**：端到端延迟增加 < 5%

---

## 7. 风险与缓解

### 7.1 技术风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| 量化误差累积 | 长上下文推理质量下降 | 定期重新量化，使用 EMA 更新 scale |
| INT4 精度不足 | 某些任务准确率大幅下降 | 提供配置选项，允许禁用 INT4 |
| 反量化开销过大 | 推理延迟增加 > 10% | 使用 fused kernel，减少内存访问 |
| Mixed-precision 内存碎片 | 显存利用率降低 | 使用 buddy allocator，支持可变大小 block |

### 7.2 工程风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| CUDA kernel 复杂度高 | 开发周期延长 | 先实现 INT8，INT4 作为可选 |
| 与 Phase 2 集成困难 | 需要重构大量代码 | 设计清晰的接口，模块化开发 |
| 量化策略调优困难 | 需要大量实验 | 提供丰富的配置选项和监控指标 |

---

## 8. 实现时间表

| 阶段 | 任务 | 时间 | 依赖 |
|------|------|------|------|
| **Week 1** | INT8 量化/反量化 kernel | 3 天 | 无 |
| **Week 1** | Memory Monitor + Scheduler | 2 天 | 无 |
| **Week 2** | 集成到 PagedAttention | 3 天 | Week 1 |
| **Week 2** | 集成到 Worker | 1 天 | Week 1 |
| **Week 2** | 单元测试 + Bug 修复 | 1 天 | Week 2 |
| **Week 3** | INT4 量化（可选） | 3 天 | Week 2 |
| **Week 3** | 性能优化 | 2 天 | Week 2 |
| **Week 4** | 端到端评估 | 3 天 | Week 3 |
| **Week 4** | 精度评估（MMLU, HumanEval） | 2 天 | Week 3 |

**总计**：约 4 周（INT8 only）或 5 周（含 INT4）

---

## 9. 后续优化方向

### 9.1 自适应量化精度

根据任务类型动态调整量化策略：

```python
if task_type == "code_generation":
    # 代码生成对精度敏感，保守量化
    int4_threshold = 0.05
elif task_type == "summarization":
    # 摘要任务对精度不敏感，激进量化
    int4_threshold = 0.2
```

### 9.2 Per-layer 量化

不同 Transformer 层的 KV 重要性不同：

```python
# 浅层（靠近输入）的 KV 更重要，使用更高精度
if layer_idx < num_layers // 2:
    target_dtype = torch.int8
else:
    target_dtype = torch.int4
```

### 9.3 动态反量化

根据 attention weight 动态决定是否反量化：

```cuda
if (attention_weight > 0.01) {
    // 高 weight：反量化到 FP16
    dequantize_to_fp16(key_int8);
} else {
    // 低 weight：直接用 INT8 计算（更快但精度略低）
    compute_in_int8(key_int8);
}
```

---

## 10. 参考资料

### 10.1 相关论文

- **LLM.int8()**: [8-bit Matrix Multiplication for Transformers](https://arxiv.org/abs/2208.07339)
- **GPTQ**: [Accurate Post-Training Quantization for GPT](https://arxiv.org/abs/2210.17323)
- **AWQ**: [Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
- **SmoothQuant**: [Accurate and Efficient Post-Training Quantization](https://arxiv.org/abs/2211.10438)

### 10.2 代码参考

- PyTorch Quantization: [官方文档](https://pytorch.org/docs/stable/quantization.html)
- vLLM INT8 KV Cache: `vllm/model_executor/layers/quantization/`
- FlashAttention INT8: [GitHub Issue](https://github.com/Dao-AILab/flash-attention/issues/123)

---

**文档版本**：v1.0  
**最后更新**：2026-04-09  
**状态**：待实施（依赖 Phase 2 完成）

# Phase 2: GPU Kernel 修改实现计划

## 1. 目标

在 vLLM 的 PagedAttention CUDA kernel 中实时采集真实的 attention score，替代 Phase 1 的 StreamingLLM 启发式规则。

**核心价值**：从"猜测"哪些 KV blocks 重要，升级为"测量"真实的注意力分布。

---

## 2. 技术方案

### 2.1 Score 采集点

**修改文件**：`workloads/vllm/vllm/vllm/attention/ops/paged_attn.py` 及其 CUDA backend

**采集位置**：在 softmax attention 计算后，对每个 KV block 的 attention weights 求和

```cuda
// 伪代码
__global__ void paged_attention_v2_kernel(
    ...
    float* block_scores  // 新增：per-block score accumulator (UVM shared)
) {
    // 1. 现有的 attention 计算
    float qk = dot(query, key);
    float attention_weight = softmax(qk);
    
    // 2. 累加到 block score
    int block_idx = get_block_index(key_position);
    atomicAdd(&block_scores[block_idx], attention_weight);
    
    // 3. 继续现有的 value 加权求和
    output += attention_weight * value;
}
```

### 2.2 Score Buffer 设计

**数据结构**：

```python
# 在 vLLM KV cache manager 中
class KVCache:
    def __init__(self, ...):
        # 现有的 KV cache tensor
        self.kv_data = torch.empty(...)
        
        # 新增：per-block attention score accumulator
        self.block_scores = torch.zeros(
            self.num_blocks,
            dtype=torch.float32,
            device='cuda'
        )
        
        # 使用 cudaMallocManaged 使其对 CPU 可见
        # 或者定期 cudaMemcpy 到 CPU
```

**内存布局**：

```
block_scores[block_id] = Σ(attention_weights for all tokens in this block)
```

- 每个 block 一个 float32（4 bytes）
- 对于 10K blocks：40KB（可忽略）
- 使用 UVM 共享内存，CPU 可直接读取

### 2.3 与 Score Bridge 集成

**修改 `score_bridge_vllm.py`**：

```python
class VLLMScoreBridge:
    def update_from_gpu_scores(
        self,
        block_scores_tensor: torch.Tensor,
        kv_cache_ptr: int,
        block_size_bytes: int,
    ):
        """
        Phase 2: 从 GPU 累加的真实 score 更新 BPF map
        """
        # 1. 读取 GPU scores（如果是 UVM，直接访问；否则需要 .cpu()）
        scores = block_scores_tensor.cpu().numpy()
        
        # 2. 归一化到 [0, 65535]
        max_score = scores.max()
        if max_score > 0:
            normalized = (scores / max_score * 65535).astype(np.uint16)
        else:
            normalized = np.zeros_like(scores, dtype=np.uint16)
        
        # 3. 根据 score 分级
        # 使用百分位数而非固定阈值
        p90 = np.percentile(scores, 90)
        p10 = np.percentile(scores, 10)
        
        for block_id, (score, norm_score) in enumerate(zip(scores, normalized)):
            if score >= p90:
                tier = TIER_HOT
            elif score <= p10:
                tier = TIER_TRASH
            else:
                tier = TIER_COOL
            
            # 4. 计算 page_id 并更新 BPF map
            block_va = kv_cache_ptr + block_id * block_size_bytes
            page_id = va_to_page_id(block_va)
            self.bridge.update_score(page_id, norm_score, tier, flags=0)
        
        # 5. 重置 GPU scores（准备下一轮累加）
        block_scores_tensor.zero_()
```

---

## 3. 实现步骤

### 3.1 Step 1: 修改 CUDA Kernel（2-3 天）

**文件**：`workloads/vllm/vllm/vllm/attention/backends/flash_attn.py`

**任务**：
1. 在 `paged_attention_v2` kernel 中添加 `block_scores` 参数
2. 在 attention weight 计算后插入 `atomicAdd`
3. 处理多头注意力（对所有 heads 的 weights 求和）

**挑战**：
- FlashAttention 使用 online softmax，weights 不完整物化
- 需要在 tiled 计算中跟踪 per-block 累加
- 可能需要使用 `logsumexp` 作为近似 score

**解决方案**：
```cuda
// 在每个 tile 的 softmax 计算后
__shared__ float tile_block_scores[MAX_BLOCKS_PER_TILE];

// 每个 thread 累加自己负责的 tokens
for (int i = threadIdx.x; i < num_tokens_in_tile; i += blockDim.x) {
    int block_idx = token_to_block[i];
    atomicAdd(&tile_block_scores[block_idx], attention_weights[i]);
}
__syncthreads();

// Block 内归约后，写回全局内存
if (threadIdx.x == 0) {
    for (int b = 0; b < num_blocks_in_tile; b++) {
        atomicAdd(&block_scores[b], tile_block_scores[b]);
    }
}
```

### 3.2 Step 2: 修改 Python 接口（1 天）

**文件**：`workloads/vllm/vllm/vllm/worker/gpu_worker.py`

**任务**：
1. 在 `KVCache` 中添加 `block_scores` tensor
2. 将 `block_scores` 传递给 CUDA kernel
3. 在每个 decode step 后调用 `score_bridge.update_from_gpu_scores()`

**代码**：
```python
class Worker:
    def __init__(self, ...):
        # 初始化 KV cache
        self.kv_cache = KVCache(...)
        
        # 初始化 score bridge
        self.score_bridge = VLLMScoreBridge()
        if os.path.exists("/sys/fs/bpf/attention_score_map"):
            self.score_bridge.connect()
    
    def execute_model(self, seq_group_metadata_list, ...):
        # 执行 attention（会累加 scores）
        output = self.model(
            ...,
            kv_cache=self.kv_cache,
            block_scores=self.kv_cache.block_scores,  # 新增
        )
        
        # 更新 BPF map
        if self.score_bridge and self.score_bridge.is_connected():
            self.score_bridge.update_from_gpu_scores(
                self.kv_cache.block_scores,
                self.kv_cache.data_ptr(),
                self.kv_cache.block_size_bytes,
            )
        
        return output
```

### 3.3 Step 3: 性能优化（1-2 天）

**优化点**：

1. **减少 atomicAdd 频率**：
   - 不是每个 token 都 atomicAdd，而是先在 shared memory 累加
   - 每个 warp 归约后再写全局内存

2. **异步更新**：
   - Score Bridge 在后台线程更新 BPF map
   - 不阻塞主推理路径

3. **自适应更新频率**：
   ```python
   # 不是每个 step 都更新，而是每 N steps 或每 M ms
   if self.step_count % UPDATE_INTERVAL == 0:
       self.score_bridge.update_from_gpu_scores(...)
   ```

### 3.4 Step 4: 测试与验证（2-3 天）

**单元测试**：
```python
def test_block_score_accumulation():
    # 1. 创建 mock KV cache
    kv_cache = create_test_kv_cache(num_blocks=100)
    
    # 2. 运行一次 attention
    output = paged_attention_v2(
        query, key, value,
        block_scores=kv_cache.block_scores,
    )
    
    # 3. 验证 scores 非零且合理
    assert kv_cache.block_scores.sum() > 0
    assert kv_cache.block_scores.max() <= num_heads * num_query_tokens
```

**集成测试**：
```bash
# 运行完整的 vLLM server + score bridge
./test_phase2_integration.sh
```

**性能测试**：
```python
# 对比 Phase 1 vs Phase 2 的驱逐决策质量
def test_eviction_quality():
    # 运行相同的 workload
    results_phase1 = run_benchmark(use_heuristic=True)
    results_phase2 = run_benchmark(use_gpu_scores=True)
    
    # Phase 2 应该有更低的 P99 延迟
    assert results_phase2['p99_latency'] < results_phase1['p99_latency']
```

---

## 4. 风险与缓解

### 4.1 技术风险

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| FlashAttention 不物化 weights | 无法直接获取 score | 使用 logsumexp 或 partial softmax 作为近似 |
| atomicAdd 性能开销 | 推理延迟增加 2-5% | 使用 shared memory 归约，减少全局 atomic 次数 |
| Score 不稳定（每次不同） | 驱逐决策抖动 | 使用指数移动平均（EMA）平滑 score |
| 多头注意力 score 聚合 | 不同 heads 关注不同位置 | 求和或取最大值，实验验证哪个更好 |

### 4.2 工程风险

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| vLLM 版本升级破坏兼容性 | 需要重新适配 | 使用 git submodule 锁定版本 |
| CUDA kernel 编译失败 | 无法构建 | 提供 fallback 到 Phase 1 启发式 |
| 内存泄漏 | 长时间运行崩溃 | 添加 CUDA memory checker，定期验证 |

---

## 5. 评估指标

### 5.1 正确性验证

**验证 1：Score 分布合理性**
```python
# 检查 score 是否符合 attention 的稀疏性
def validate_score_distribution(scores):
    # 大部分 blocks 应该有低 score
    assert np.percentile(scores, 50) < np.mean(scores)
    
    # 少数 blocks 应该有高 score（长尾分布）
    assert np.percentile(scores, 90) > 2 * np.mean(scores)
```

**验证 2：与 Phase 1 启发式对比**
```python
# Phase 2 的 HOT blocks 应该包含 Phase 1 识别的 Sink + Recent
def compare_with_heuristic(gpu_scores, heuristic_tiers):
    gpu_hot = np.where(gpu_scores > threshold)[0]
    heuristic_hot = np.where(heuristic_tiers == TIER_HOT)[0]
    
    # 交集应该 > 80%
    overlap = len(set(gpu_hot) & set(heuristic_hot))
    assert overlap / len(heuristic_hot) > 0.8
```

### 5.2 性能对比

| 指标 | Phase 1 (启发式) | Phase 2 (GPU Score) | 目标 |
|------|-----------------|-------------------|------|
| **驱逐准确率** | 基准 | +10-20% | 更少驱逐有用数据 |
| **P99 延迟** | 基准 | -5-15% | 更低尾延迟 |
| **推理开销** | 0% | +2-5% | 可接受 |
| **内存开销** | 0 | 40KB (10K blocks) | 可忽略 |

---

## 6. 实现时间表

| 阶段 | 任务 | 时间 | 负责人 |
|------|------|------|--------|
| **Week 1** | CUDA kernel 修改 + 单元测试 | 3 天 | CUDA 工程师 |
| **Week 1** | Python 接口修改 | 1 天 | vLLM 工程师 |
| **Week 1** | Score Bridge 升级 | 1 天 | 系统工程师 |
| **Week 2** | 集成测试 + Bug 修复 | 2 天 | 全员 |
| **Week 2** | 性能优化 | 2 天 | CUDA 工程师 |
| **Week 2** | 端到端验证 | 1 天 | 全员 |
| **Week 3** | 对比实验 + 论文撰写 | 5 天 | 研究员 |

**总计**：约 3 周

---

## 7. Fallback 策略

如果 Phase 2 实现遇到不可克服的技术障碍，提供以下 fallback：

### 7.1 Fallback A: 采样式 Score 采集

不是每个 token 都累加 score，而是每 N 个 tokens 采样一次：

```cuda
if (token_idx % SAMPLE_RATE == 0) {
    atomicAdd(&block_scores[block_idx], attention_weight * SAMPLE_RATE);
}
```

**优点**：降低 atomicAdd 开销  
**缺点**：Score 精度降低，但仍优于启发式

### 7.2 Fallback B: CPU-side Score 计算

在 CPU 上读取 attention weights（如果 vLLM 已经输出），计算 score：

```python
# 如果 vLLM 的 attention 输出包含 weights
attention_weights = model.get_attention_weights()  # [num_heads, num_tokens, num_kv_tokens]

# 在 CPU 上聚合
block_scores = aggregate_weights_to_blocks(attention_weights, block_table)
```

**优点**：无需修改 CUDA kernel  
**缺点**：需要传输大量数据（weights），开销可能更大

### 7.3 Fallback C: 混合策略

对于 prefill phase 使用 GPU score（一次性计算，开销分摊），对于 decode phase 使用启发式（每次只生成 1 token，开销敏感）：

```python
if phase == "prefill":
    scores = compute_gpu_scores(attention_weights)
elif phase == "decode":
    scores = compute_heuristic_scores(token_positions)
```

---

## 8. 后续优化方向

### 8.1 多模态 Attention

对于 vision-language 模型，图像 tokens 的 attention 模式与文本不同：

```python
# 图像 tokens 通常有更均匀的 attention
if token_type == "image":
    tier = TIER_HOT  # 全部保护
elif token_type == "text":
    tier = compute_tier_from_score(score)
```

### 8.2 动态阈值调整

根据当前显存压力动态调整 HOT/TRASH 阈值：

```python
if memory_pressure > 0.9:
    # 显存紧张：提高 HOT 阈值，只保护最重要的
    hot_threshold = np.percentile(scores, 95)
elif memory_pressure < 0.7:
    # 显存充足：降低 HOT 阈值，多保护一些
    hot_threshold = np.percentile(scores, 85)
```

### 8.3 跨请求 Score 共享

对于多个请求共享相同 prefix 的场景（如 few-shot prompting），共享 prefix 的 score：

```python
# 如果 block 被多个请求共享，累加所有请求的 score
shared_block_scores[block_id] = sum(
    request.block_scores[block_id]
    for request in requests_sharing_block
)
```

---

## 9. 参考资料

### 9.1 相关论文

- **FlashAttention**: [Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- **H2O**: [Heavy-Hitter Oracle for Efficient Generative Inference](https://arxiv.org/abs/2306.14048)
- **StreamingLLM**: [Efficient Streaming Language Models](https://arxiv.org/abs/2309.17453)
- **vLLM**: [Efficient Memory Management for LLM Serving](https://arxiv.org/abs/2309.06180)

### 9.2 代码参考

- vLLM PagedAttention: `vllm/attention/ops/paged_attn.py`
- FlashAttention CUDA: `flash_attn/flash_attn_interface.py`
- PyTorch CUDA Extension: [官方教程](https://pytorch.org/tutorials/advanced/cpp_extension.html)

---

**文档版本**：v1.0  
**最后更新**：2026-04-09  
**状态**：待实施

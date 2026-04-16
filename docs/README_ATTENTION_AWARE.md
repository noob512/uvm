# 注意力感知GPU内存子系统 - 项目总览

## 快速导航

本项目实现了一套基于注意力机制的智能GPU显存管理系统，专为大语言模型（LLM）推理优化。

### 📚 核心文档

| 文档 | 描述 | 适用人群 |
|------|------|---------|
| [可行性分析](attention_aware_memory_subsystem_feasibility.md) | 技术可行性评估，架构设计决策 | 架构师、研究员 |
| [完整实现文档](attention_aware_implementation.md) | Phase 1 实现细节、使用指南、监控调试 | 开发者、运维 |
| [Phase 2 实现计划](phase2_gpu_kernel_modification_plan.md) | GPU Kernel 修改路线图 | CUDA 工程师 |
| [Phase 3 实现计划](phase3_kv_quantization_plan.md) | KV 量化方案设计 | 系统工程师 |

### 🔧 实现代码

| 组件 | 位置 | 说明 |
|------|------|------|
| **BPF 驱逐策略** | [extension/attention_aware_eviction.bpf.c](../extension/attention_aware_eviction.bpf.c) | 内核态驱逐决策逻辑 |
| **用户态加载器** | [extension/attention_aware_eviction.c](../extension/attention_aware_eviction.c) | BPF 程序加载和管理 |
| **Score Bridge** | [workloads/vllm/score_bridge_vllm.py](../workloads/vllm/score_bridge_vllm.py) | 用户态分数桥接守护进程 |
| **集成指南** | [workloads/vllm/SCORE_BRIDGE_INTEGRATION.md](../workloads/vllm/SCORE_BRIDGE_INTEGRATION.md) | vLLM 集成步骤 |
| **实验脚本** | [workloads/vllm/run_exp_attention_aware.sh](../workloads/vllm/run_exp_attention_aware.sh) | 对比实验自动化脚本 |

---

## 项目概述

### 核心问题

传统的 GPU 显存管理使用 LRU（Least Recently Used）策略，对所有数据一视同仁。但在 LLM 推理中，不同的 KV Cache blocks 对最终输出的贡献差异巨大：

- **Attention Sink**（首 4 tokens）：几乎所有 token 都会关注它们
- **Recent Window**（末 128 tokens）：最近的上下文，高度相关
- **Middle Tokens**（中间部分）：远离 Sink 和 Recent，贡献很小

传统 LRU 可能会驱逐 Attention Sink，导致推理质量严重下降。

### 解决方案

**注意力感知驱逐**：用语义信息（attention score）指导显存驱逐决策。

```
高 Attention Score → 保护（move_tail，最后驱逐）
低 Attention Score → 优先驱逐（move_head，最先驱逐）
```

### 三阶段实现

| Phase | 功能 | 状态 | 预期收益 |
|-------|------|------|---------|
| **Phase 1** | Score-guided eviction + StreamingLLM 启发式 | ✅ 已实现 | P99 延迟 -20~40% |
| **Phase 2** | GPU-side real attention score collection | ⏳ 计划中 | 驱逐准确率 +10~20% |
| **Phase 3** | Proactive KV quantization | ⏳ 计划中 | 显存容量 +2~4x |

---

## 快速开始

### 前置条件

- NVIDIA GPU with UVM support
- 已编译的 gpu_ext 内核模块（参见主 README）
- vLLM 环境（使用 `uv` 管理）

### 5 分钟体验

**步骤 1：编译并启动 BPF 策略**

```bash
cd /home/ubuntu/nvidia-uvm-gpu/extension
make attention_aware_eviction
sudo ./attention_aware_eviction --stats-interval 10
```

**步骤 2：运行 vLLM 推理**

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm

# 启动 vLLM server（会自动使用 attention-aware 策略）
uv run python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --gpu-memory-utilization 0.95
```

**步骤 3：观察统计信息**

在步骤 1 的终端中，每 10 秒会打印统计信息：

```
--- Stats (t=10s) ---
  activate_total         1523
  score_hit              1204  (79% 命中率)
  move_head_trash         456  (30% 被优先驱逐)
  move_tail_hot           623  (41% 被保护)
  tier_cool               125  (8% 走默认 LRU)
  t1_protect              319  (21% 模型权重保护)
```

### 运行对比实验

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
./run_exp_attention_aware.sh
```

结果保存在 `results/exp_attention_aware/<timestamp>/`，包含：
- 延迟对比图（P50/P99）
- 吞吐对比图
- 详细的 JSON 结果

---

## 系统架构

### 数据流

```
┌─────────────────────────────────────────────────────────────┐
│                      vLLM Worker                            │
│                                                             │
│  [Phase 1] StreamingLLM Heuristic                          │
│  [Phase 2] GPU Attention Kernel → Real Scores              │
│                          ↓                                  │
│              Score Bridge Daemon                            │
│              (score_bridge_vllm.py)                         │
│                          ↓                                  │
└──────────────────────────┼──────────────────────────────────┘
                           │ bpf_map_update_elem()
                           ↓
              ┌────────────────────────────┐
              │  BPF Map (Kernel Space)    │
              │  /sys/fs/bpf/              │
              │  attention_score_map       │
              │                            │
              │  Key: page_id (VA >> 21)   │
              │  Value: {score, tier}      │
              └────────────────────────────┘
                           │
                           │ 驱逐时查询
                           ↓
              ┌────────────────────────────┐
              │  eBPF Program              │
              │  (attention_aware_eviction)│
              │                            │
              │  TIER_HOT   → move_tail    │
              │  TIER_TRASH → move_head    │
              │  TIER_COOL  → default LRU  │
              └────────────────────────────┘
                           │
                           ↓
              ┌────────────────────────────┐
              │  NVIDIA UVM Driver         │
              │  (实际页面迁移)             │
              └────────────────────────────┘
```

### 三层分级

| Tier | 语义 | 驱逐优先级 | 典型场景 |
|------|------|-----------|---------|
| **TIER_HOT** | 核心数据 | 最低 | Attention Sink, Recent Window |
| **TIER_COOL** | 普通数据 | 中等 | 中间位置的 KV blocks |
| **TIER_TRASH** | 垃圾数据 | 最高 | 远离 Sink 和 Recent 的 tokens |

---

## 性能预期

### Phase 1（已实现）

**场景 1：长上下文推理（32K tokens）**
- P99 延迟：-20~40%
- 原因：保护 Attention Sink，避免重新计算

**场景 2：多请求并发（显存超卖 1.5x）**
- 吞吐：+15~30%
- 原因：优先驱逐低价值 KV，高价值请求不受影响

### Phase 2（计划中）

- 驱逐准确率：+10~20%
- 原因：真实 attention score 比启发式更精确

### Phase 3（计划中）

- 显存容量：+2~4x
- 精度损失：< 2%
- 原因：低 score blocks 量化到 INT8/INT4

---

## 监控与调试

### 实时统计

```bash
# BPF 程序统计
sudo ./extension/attention_aware_eviction --stats-interval 10

# Score Bridge 统计
uv run --directory workloads/vllm python score_bridge_vllm.py daemon --stats
```

### 关键指标

| 指标 | 说明 | 健康值 |
|------|------|--------|
| `score_hit / activate_total` | KV Cache 识别率 | > 70% |
| `move_head_trash` | 成功优先驱逐的垃圾数据 | > 20% |
| `move_tail_hot` | 成功保护的核心数据 | > 30% |

### 调试工具

```bash
# 查看 BPF map 内容
sudo bpftool map dump pinned /sys/fs/bpf/attention_score_map

# 追踪 UVM 事件
sudo bpftrace -e 'kprobe:uvm_gpu_chunk_evict { @[comm] = count(); }'
```

---

## 常见问题

### Q1: BPF 程序加载失败

```
Failed to attach struct_ops: Device or resource busy
```

**解决**：清理旧的 struct_ops 实例
```bash
sudo ./extension/cleanup_struct_ops_tool
```

### Q2: 统计信息全是 0

**原因**：Score Bridge 未运行或未正确更新 map

**解决**：
```bash
# 检查 score_bridge 进程
ps aux | grep score_bridge

# 查看 map 内容
sudo bpftool map dump pinned /sys/fs/bpf/attention_score_map
```

### Q3: 性能没有提升

**可能原因**：
1. 显存压力不足（< 80%），驱逐很少发生
2. 工作负载不适合（如短上下文推理）
3. Score Bridge 更新频率过低

**解决**：
- 增加 `--gpu-memory-utilization` 到 0.95
- 使用长上下文 benchmark（> 16K tokens）
- 降低 Score Bridge 的 `--interval` 到 0.5s

---

## 贡献指南

### 代码结构

```
nvidia-uvm-gpu/
├── extension/
│   ├── attention_aware_eviction.bpf.c    # BPF 驱逐策略
│   ├── attention_aware_eviction.c        # 用户态加载器
│   └── uvm_types.h                       # UVM 数据结构定义
├── workloads/vllm/
│   ├── score_bridge_vllm.py              # Score Bridge 主程序
│   ├── SCORE_BRIDGE_INTEGRATION.md       # 集成指南
│   ├── run_exp_attention_aware.sh        # 实验脚本
│   └── benchmark_attention_aware.py      # Benchmark 工具
└── docs/
    ├── attention_aware_implementation.md # 完整实现文档
    ├── phase2_gpu_kernel_modification_plan.md
    └── phase3_kv_quantization_plan.md
```

### 开发流程

1. **修改 BPF 程序**：编辑 `.bpf.c` 文件，运行 `make` 重新编译
2. **修改 Score Bridge**：编辑 `.py` 文件，无需重新编译
3. **测试**：运行 `./run_exp_attention_aware.sh` 验证
4. **提交**：遵循项目的 commit 规范

### 添加新功能

**示例：添加新的分级策略**

1. 修改 `attention_aware_eviction.bpf.c`：
   ```c
   // 添加新的 tier 定义
   #define TIER_CRITICAL 3
   
   // 在 gpu_block_activate 中添加逻辑
   if (tier == TIER_CRITICAL) {
       // 绝对不驱逐
       bpf_gpu_block_move_tail(chunk, list);
       return 1;
   }
   ```

2. 修改 `score_bridge_vllm.py`：
   ```python
   TIER_CRITICAL = 3
   
   def compute_tier(self, score):
       if score > 0.95:
           return TIER_CRITICAL
       # ... 现有逻辑 ...
   ```

3. 测试并提交

---

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@inproceedings{gpu_ext_attention_aware,
  title={Attention-Aware GPU Memory Management for LLM Inference},
  author={gpu_ext Project Team},
  booktitle={To appear},
  year={2026}
}
```

---

## 相关资源

### 论文

- [StreamingLLM: Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)
- [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048)
- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

### 代码

- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [StreamingLLM GitHub](https://github.com/mit-han-lab/streaming-llm)
- [eBPF Documentation](https://ebpf.io/)

---

## 许可证

本项目遵循 GPL-2.0 许可证（与 Linux 内核一致）。

---

## 联系方式

- **Issues**: [GitHub Issues](https://github.com/your-org/nvidia-uvm-gpu/issues)
- **讨论**: [GitHub Discussions](https://github.com/your-org/nvidia-uvm-gpu/discussions)

---

**最后更新**：2026-04-09  
**项目状态**：Phase 1 已完成，Phase 2/3 计划中

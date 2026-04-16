# gpu_ext Policy Overview

## 1. 策略分类

gpu_ext 提供两类可编程策略：
- **Eviction Policies**：GPU 内存驱逐策略（10 个）
- **Prefetch Policies**：GPU 内存预取策略（9 个）

---

## 2. Eviction 策略表

| 策略 | 文件 | 描述 | 适用场景 | 复杂度 |
|------|------|------|---------|--------|
| **FIFO** | `eviction_fifo.bpf.c` | 先进先出（基准） | 基准测试 | O(1) |
| **LRU** | (内核默认) | 最近最少使用 | 通用场景 | O(1) |
| **MRU** | `eviction_mru.bpf.c` | 最近最多使用（反向扫描） | 全扫描负载 | O(1) |
| **LFU** | `eviction_lfu.bpf.c` | 最少使用频率 | 热点数据保护 | O(1) 驱逐 + O(1) 更新 |
| **PID Quota** | `eviction_pid_quota.bpf.c` | 进程配额隔离 | Multi-tenant | O(1) |
| **Freq Decay** | `eviction_freq_pid_decay.bpf.c` | 频率指数衰减 | 时间局部性强 | O(1) |
| **FIFO Chance** | `eviction_fifo_chance.bpf.c` | FIFO + 二次机会 | 混合负载 | O(1) |

**详细分析**：[`reference/EVICTION_POLICIES.md`](reference/EVICTION_POLICIES.md)

**代码位置**：`../../extension/eviction_*.bpf.c`

---

## 3. Prefetch 策略表

| 策略 | 文件 | 描述 | 适用场景 |
|------|------|------|---------|
| **None** | `prefetch_none.bpf.c` | 禁用预取 | 随机访问 |
| **Always Max** | `prefetch_always_max.bpf.c` | 总是最大预取 | 顺序访问 |
| **Stride** | `prefetch_stride.bpf.c` | 步长预测 | 固定步长访问 |
| **Adaptive Sequential** | `prefetch_adaptive_sequential.bpf.c` | 自适应顺序 | 动态顺序访问 |
| **Adaptive Tree** | `prefetch_adaptive_tree_iter.bpf.c` | 树感知 | 树结构遍历 |
| **PID Tree** | `prefetch_pid_tree.bpf.c` | PID 感知树 | Multi-tenant 树访问 |
| **Trace-based** | `prefetch_trace.bpf.c` | 基于访问痕迹 | 重复访问模式 |

**详细推荐**：[`policy/suggestions.md`](policy/suggestions.md)

**代码位置**：`../../extension/prefetch_*.bpf.c`

---

## 4. 快速选择指南

### 4.1 按工作负载选择

| 工作负载 | 推荐 Eviction | 推荐 Prefetch | 原因 |
|---------|--------------|--------------|------|
| **LLM Inference**<br/>(llama.cpp) | LFU | Adaptive Sequential | 热点 KV-cache + 顺序 token 生成 |
| **GNN Training**<br/>(PyTorch) | Freq Decay | PID Tree | 时间局部性 + 图结构遍历 |
| **Vector Search**<br/>(FAISS) | MRU | Stride | 全扫描模式 + 固定步长 |
| **vLLM Multi-tenant** | PID Quota | Adaptive Sequential | 租户隔离 + 顺序推理 |

详细工作负载分析：[`profiling/WORKLOAD_ANALYSIS.md`](profiling/WORKLOAD_ANALYSIS.md)

### 4.2 按内存压力选择

```
低压力 (<50% GPU memory used)  → LRU (内核默认)
                                  适用于大部分场景，无需自定义

中压力 (50-80%)                → LFU / Freq Decay
                                  保护热点数据，减少 thrashing

高压力 (>80%)                  → PID Quota / FIFO Chance
                                  多租户隔离，公平性保障
```

### 4.3 按优化目标选择

| 优化目标 | 推荐策略 | 说明 |
|---------|---------|------|
| **吞吐量** | LFU, Freq Decay | 最大化 hit rate |
| **延迟** | FIFO, LRU | 简单快速，开销低 |
| **公平性** | PID Quota | 保障多租户资源隔离 |
| **能耗** | MRU | 减少不必要的驱逐 |

---

## 5. 策略开发路径

### 5.1 从简单到复杂

**Level 1：基础策略（新手入门）**
- `eviction_fifo.bpf.c` — 最简单，只用 `move_tail`，~80 LOC
- `prefetch_none.bpf.c` — 空实现，理解 hook 结构

**Level 2：中级策略（使用 BPF map）**
- `eviction_lfu.bpf.c` — 使用 BPF map 追踪频率，~200 LOC
- `prefetch_stride.bpf.c` — 步长检测，~150 LOC

**Level 3：高级策略（多 PID + 衰减）**
- `eviction_freq_pid_decay.bpf.c` — 复杂状态管理，~350 LOC
- `prefetch_adaptive_tree_iter.bpf.c` — 树感知预取，~280 LOC

### 5.2 开发指南文档

1. **架构理解**：[`driver_docs/lru/UVM_LRU_POLICY.md`](driver_docs/lru/UVM_LRU_POLICY.md)
   - LRU 框架完整分析（105KB）
   - Chunk 生命周期理解
   - PMM 数据结构详解

2. **Hook 详解**：[`driver_docs/lru/HOOK_CALL_PATTERN_ANALYSIS.md`](driver_docs/lru/HOOK_CALL_PATTERN_ANALYSIS.md)
   - 5 个 hook 的调用时机（51KB）
   - Hook 参数含义
   - 调用频率分析

3. **开发实践**：[`driver_docs/lru/UVM_LRU_USAGE_GUIDE.md`](driver_docs/lru/UVM_LRU_USAGE_GUIDE.md)
   - 策略开发步骤（27KB）
   - 常见错误和调试
   - 性能优化技巧

---

## 6. 论文参考

- **gpu_ext 论文**（在审）：GPU driver programmability via eBPF struct_ops
- **策略详解**：[`reference/EVICTION_POLICIES.md`](reference/EVICTION_POLICIES.md)
- **Related work**：[`reference/related.md`](reference/related.md)

---

## 7. 实验和评估

### 7.1 Workload Profiling

- **Workload 分析**：[`profiling/WORKLOAD_ANALYSIS.md`](profiling/WORKLOAD_ANALYSIS.md)
  - llama.cpp：KV-cache 热点分析
  - PyTorch GNN：图遍历模式
  - FAISS：向量搜索访问模式

### 7.2 Multi-tenant 评估

- **Memory 评估**：[`eval/multi-tenant-memory/EVALUATION_REPORT.md`](eval/multi-tenant-memory/EVALUATION_REPORT.md)
- **Scheduler 评估**：[`eval/multi-tenant-scheduler/README.md`](eval/multi-tenant-scheduler/README.md)

### 7.3 策略推荐

- **按 workload 选择**：[`policy/suggestions.md`](policy/suggestions.md)
- **策略对比**：[`reference/EVICTION_POLICIES.md`](reference/EVICTION_POLICIES.md)

---

## 8. 快速开始

### 8.1 编译和加载策略

```bash
# 1. 编译策略（示例：LFU）
cd extension
make eviction_lfu

# 2. 加载策略
sudo ./eviction_lfu

# 3. 验证加载
sudo bpftool prog list | grep uvm
```

### 8.2 运行 Benchmark

```bash
# 1. llama.cpp inference
cd workloads/llama.cpp
uv run python bench.py

# 2. 查看结果
cat results/*.json

# 3. 清理 GPU（下次运行前）
cd ..
uv run python cleanup_gpu.py
```

### 8.3 开发新策略

```bash
# 1. 复制参考实现
cd extension
cp eviction_fifo.bpf.c eviction_my_policy.bpf.c

# 2. 修改策略逻辑
vim eviction_my_policy.bpf.c

# 3. 更新 Makefile
vim Makefile
# 添加：eviction_my_policy

# 4. 编译测试
make eviction_my_policy
sudo ./eviction_my_policy
```

---

## 9. BPF Struct_Ops 架构

当前使用的 `struct gpu_mem_ops` 定义了 6 个 hook：

```c
struct gpu_mem_ops {
    // 测试 hook
    int (*gpu_test_trigger)(...);

    // Prefetch hooks
    int (*gpu_page_prefetch)(...);
    int (*gpu_page_prefetch_iter)(...);

    // Eviction hooks
    int (*gpu_block_activate)(...);     // 低频 (~6.7k/s)
    int (*gpu_block_access)(...);         // 高频 (~70k/s)
    int (*gpu_evict_prepare)(...);   // 低频 (~6.7k/s)
};
```

**注意**：
- `chunk_used` 是最高频 hook（84% 调用），性能关键
- `activate` 和 `eviction_prepare` 频率 1:1（驱逐后立即重分配）
- 所有策略必须实现为 BPF struct_ops，不是回调函数框架

详见：[`driver_docs/lru/HOOK_CALL_PATTERN_ANALYSIS.md`](driver_docs/lru/HOOK_CALL_PATTERN_ANALYSIS.md)

---

## 10. FAQ

### Q1: 如何选择 Eviction 策略？

**A**: 先确定优化目标（吞吐/延迟/公平性），再看内存压力：
- 低压力 → 用默认 LRU
- 中压力 + 有热点 → LFU
- 高压力 + 多租户 → PID Quota

### Q2: Prefetch 策略可以单独用吗？

**A**: 可以，Eviction 和 Prefetch 是独立的。默认组合：
- Eviction: LRU（内核默认）
- Prefetch: None（无预取）

### Q3: 如何调试 BPF 策略？

**A**: 使用 `bpf_printk` + `bpftrace`：
```bash
# 在 BPF 代码中
bpf_printk("chunk addr: %llx, freq: %u", addr, freq);

# 在终端查看
sudo bpftrace -e 'tracepoint:bpf:bpf_trace_printk { printf("%s\n", args->msg); }'
```

### Q4: 策略加载失败怎么办？

**A**: 常见原因：
1. BPF verifier 拒绝 → 检查循环和指针访问
2. Hook 签名不匹配 → 参考 `eviction_common.h`
3. 内核模块未加载 → `lsmod | grep nvidia_uvm`

---

## 11. 相关资源

### 代码仓库

- **Extension 代码**：`../../extension/`
- **Workload 脚本**：`../../workloads/`
- **内核模块**：`../../kernel-module/nvidia-module/`

### 文档导航

- **顶层 README**：[`../README.md`](../README.md)
- **驱动架构**：[`driver_docs/UVM_MODULE_ARCHITECTURE_CN.md`](driver_docs/UVM_MODULE_ARCHITECTURE_CN.md)
- **Prefetch 分析**：[`driver_docs/prefetch/UVM_PREFETCH_POLICY_ANALYSIS.md`](driver_docs/prefetch/UVM_PREFETCH_POLICY_ANALYSIS.md)
- **GPU Scheduling**：[`driver_docs/sched/GPU_SCHEDULING_CONTROL_ANALYSIS.md`](driver_docs/sched/GPU_SCHEDULING_CONTROL_ANALYSIS.md)

### 外部资源

- BPF CO-RE：https://nakryiko.com/posts/bpf-core-reference-guide/
- sched_ext：https://github.com/sched-ext/scx
- NVIDIA UVM：https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#unified-memory-programming

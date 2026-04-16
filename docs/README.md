# gpu_ext Documentation

## 快速导航

### 新用户入口

**我想开发 GPU 策略**：
1. [`gpu-ext/POLICY_OVERVIEW.md`](gpu-ext/POLICY_OVERVIEW.md) — 从这里开始
2. 选择参考策略（FIFO → LFU → Freq Decay）
3. 阅读开发指南：[`gpu-ext/driver_docs/lru/UVM_LRU_USAGE_GUIDE.md`](gpu-ext/driver_docs/lru/UVM_LRU_USAGE_GUIDE.md)
4. 查看代码：`extension/eviction_*.bpf.c`

**我想理解驱动架构**：
1. [`gpu-ext/driver_docs/UVM_MODULE_ARCHITECTURE_CN.md`](gpu-ext/driver_docs/UVM_MODULE_ARCHITECTURE_CN.md) — 模块总览
2. [`gpu-ext/driver_docs/lru/UVM_LRU_POLICY.md`](gpu-ext/driver_docs/lru/UVM_LRU_POLICY.md) — LRU 框架
3. [`gpu-ext/driver_docs/lru/HOOK_CALL_PATTERN_ANALYSIS.md`](gpu-ext/driver_docs/lru/HOOK_CALL_PATTERN_ANALYSIS.md) — Hook 调用

**我想跑实验**：
1. [`../workloads/README.md`](../workloads/README.md) — Workload 说明
2. [`gpu-ext/eval/multi-tenant-memory/README.md`](gpu-ext/eval/multi-tenant-memory/README.md) — 实验脚本
3. [`gpu-ext/policy/suggestions.md`](gpu-ext/policy/suggestions.md) — 策略选择

---

### 按任务导航

**开发 Eviction 策略**:
```
gpu-ext/POLICY_OVERVIEW.md (策略全景)
  └─ gpu-ext/driver_docs/lru/UVM_LRU_USAGE_GUIDE.md (开发指南)
      └─ gpu-ext/reference/EVICTION_POLICIES.md (策略详解)
          └─ extension/eviction_fifo.bpf.c (参考实现)
```

**开发 Prefetch 策略**:
```
gpu-ext/POLICY_OVERVIEW.md (策略全景)
  └─ gpu-ext/policy/suggestions.md (策略推荐)
      └─ extension/prefetch_adaptive_sequential.bpf.c (参考实现)
```

**理解 BPF Struct_Ops 架构**:
```
gpu-ext/driver_docs/lru/UVM_LRU_POLICY.md (LRU 框架)
  └─ gpu-ext/driver_docs/lru/HOOK_CALL_PATTERN_ANALYSIS.md (Hook 分析)
      └─ gpu-ext/driver_docs/lru/PMM_CHUNK_LIFECYCLE_ANALYSIS.md (生命周期)
```

---

## Directory Structure

### [`gpu-ext/`](gpu-ext/) — gpu_ext 论文（在审）

GPU driver programmability via eBPF struct_ops. 所有与 gpu_ext 论文相关的文档。

| Subdirectory | Description |
|---|---|
| [`paper/`](gpu-ext/paper/) | LaTeX 论文源文件、图表数据、构建脚本 |
| [`driver_docs/`](gpu-ext/driver_docs/) | NVIDIA UVM 驱动内部分析（架构、LRU、prefetch、scheduling、call graph） |
| [`eval/`](gpu-ext/eval/) | 评估报告（multi-tenant memory/scheduler 实验结果） |
| [`policy/`](gpu-ext/policy/) | 策略推荐矩阵（per-workload 最优 eviction/prefetch 策略） |
| [`profiling/`](gpu-ext/profiling/) | Workload 内存访问模式分析（llama.cpp, faiss, pytorch） |
| [`test-verify/`](gpu-ext/test-verify/) | GPU eBPF 验证分析与 benchmark |
| [`reference/`](gpu-ext/reference/) | 参考资料（eviction 策略总览、chunk trace 格式、related work） |
| [`experiment/`](gpu-ext/experiment/) | 实验记录与脚本架构设计 |
| [`archive/`](gpu-ext/archive/) | 旧版/草稿文档归档 |

### 项目计划文档

| File | Description |
|---|---|
| [`msched_reproduction_plan.md`](msched_reproduction_plan.md) | MSched 算法复现与研究计划 |
| [`cross_block_prefetch_plan.md`](cross_block_prefetch_plan.md) | Cross-block proactive prefetch 计划 |
| [`xcoord_plan.md`](xcoord_plan.md) | xCoord CPU-GPU 协调计划（零内核修改） |

### [`xcoord/`](xcoord/) — xCoord 参考资料

| File | Description |
|---|---|
| [`bpf_core_access_findings.md`](xcoord/bpf_core_access_findings.md) | ✅ BPF CO-RE 发现（无需 kfunc） |

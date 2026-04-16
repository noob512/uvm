# vLLM KV Cache / Weights 地址驱动分流策略 - 详细实施方案

**版本**: v1.0  
**日期**: 2026-04-16  
**状态**: 实施计划（待执行）  
**基于文档**: 
- [vllm_kv_weight_address_driven_policy.md](vllm_kv_weight_address_driven_policy.md)
- [gpt_plan.md](gpt_plan.md)
- [suggestions.md](suggestions.md)

---

## 执行摘要

本文档详细规划了如何在 `gpu_ext` 项目中实现基于地址真值的 KV Cache / Weights 分流策略。核心思想是：**放弃通过热度推断数据类型，改为由 vLLM 直接上报 KV cache 与 model weights 的地址真值**，eBPF 侧只做 O(1) 地址分类，从而实现精确的分流策略。

### 核心目标

1. **可靠分类**: 通过 vLLM 上报的地址真值区分 KV Cache 和 Weights
2. **分流策略**: KV 允许激进的跨 block 预取，Weights 优先保护避免驱逐
3. **低开销**: struct_ops 热路径仅做 map lookup，不做复杂推断
4. **兼容现有框架**: 复用 `prefetch_vllm_phase.bpf.c` 的 uprobe 模式和 cross-block 机制

---

## 第一部分：架构设计

### 1.1 总体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    vLLM Application                          │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │  Model Weights   │         │   KV Cache       │         │
│  │  Load Complete   │         │   Allocation     │         │
│  └────────┬─────────┘         └────────┬─────────┘         │
│           │                             │                    │
│           └─────────────┬───────────────┘                    │
│                         ▼                                    │
│              ┌─────────────────────┐                        │
│              │ uvm_hint_block()    │ ◄─ C ABI in            │
│              │ (in uvm_allocator   │    uvm_allocator.so    │
│              │  .abi3.so)          │                        │
│              └──────────┬──────────┘                        │
└─────────────────────────┼───────────────────────────────────┘
                          │
                          │ uprobe attach
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    eBPF Programs                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  UPROBE: uvm_hint_block(kind, va_block_start)       │  │
│  │  → Write to region_block_map[(tgid, va_block)] = kind│  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  KPROBE: uvm_perf_prefetch_get_hint_va_block        │  │
│  │  → Cache va_block context (va_start, va_end, ...)   │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  STRUCT_OPS: gpu_page_prefetch                       │  │
│  │  1. Lookup region_block_map[(tgid, va_start)]       │  │
│  │  2. If KV → always_max + schedule cross-block       │  │
│  │  3. If WEIGHTS → always_max only (no cross-block)   │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  STRUCT_OPS: gpu_block_activate (eviction)          │  │
│  │  1. Lookup region_block_map[(tgid, chunk_va)]       │  │
│  │  2. If WEIGHTS → move_tail (protect)                │  │
│  │  3. If KV → default (allow eviction)                │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 数据流

**控制面（初始化阶段）**:
1. vLLM 加载模型权重 → 计算覆盖的 2MB VA blocks → 调用 `uvm_hint_block(REGION_WEIGHTS, va_block_start)`
2. vLLM 初始化 KV cache → 计算覆盖的 2MB VA blocks → 调用 `uvm_hint_block(REGION_KV, va_block_start)`
3. eBPF uprobe 捕获调用 → 写入 `region_block_map`

**数据面（运行时）**:
1. Page fault 触发 → kprobe 缓存 va_block 上下文
2. `gpu_page_prefetch` → 查询 `region_block_map` → 根据类型执行分流策略
3. `gpu_block_activate` → 查询 `region_block_map` → 根据类型调整驱逐优先级

---

## 第二部分：文件结构与修改清单

### 2.1 新增文件

```
extension/
├── prefetch_vllm_address_driven.bpf.c    # 核心 eBPF 程序
├── prefetch_vllm_address_driven.c        # Userspace loader
└── uvm_hint_abi.h                        # C ABI 头文件（vLLM 和 eBPF 共享）

workloads/vllm/
├── vllm_address_hint.py                  # Python 包装器（调用 C ABI）
└── patch_vllm_worker.py                  # vLLM worker 集成脚本

docs/
└── vllm_address_driven_implementation.md # 本文档的技术细节版本
```

### 2.2 修改文件

```
workloads/vllm/
├── vllm/vllm/v1/worker/gpu_worker.py     # 集成地址上报逻辑
└── vllm/vllm/uvm_allocator.abi3.so       # 添加 uvm_hint_block() 函数

extension/Makefile                         # 添加新策略的编译目标
```

---

## 第三部分：详细实施步骤

### Phase 1: 基础设施搭建（语义通道）

#### Step 1.1: 定义 C ABI 接口

**文件**: `extension/uvm_hint_abi.h`

```c
/* SPDX-License-Identifier: GPL-2.0 */
#ifndef UVM_HINT_ABI_H
#define UVM_HINT_ABI_H

#include <stdint.h>

/* Region kinds (must match eBPF side) */
#define REGION_UNKNOWN  0
#define REGION_WEIGHTS  1
#define REGION_KV       2
#define REGION_OTHER    3

/* 
 * uvm_hint_block - Report memory region type to eBPF
 * @kind: REGION_WEIGHTS, REGION_KV, etc.
 * @va_block_start: 2MB-aligned virtual address
 * 
 * This function is called from vLLM to inform the eBPF policy
 * about the semantic type of memory regions.
 * 
 * Implementation: Empty stub in uvm_allocator.so, actual logic
 * is in eBPF uprobe attached to this function.
 */
void uvm_hint_block(uint32_t kind, uint64_t va_block_start);

/* Helper: Convert tensor address to 2MB-aligned VA block start */
static inline uint64_t va_to_block_start(uint64_t addr) {
    return addr & ~((1ULL << 21) - 1);  /* 2MB = 1 << 21 */
}

#endif /* UVM_HINT_ABI_H */
```

#### Step 1.2: 在 uvm_allocator.so 中添加 stub 函数

**文件**: `workloads/vllm/vllm/device_allocator/uvm_allocator.c` (需要找到源文件)

```c
#include "uvm_hint_abi.h"

/* Empty stub - actual logic is in eBPF uprobe */
void uvm_hint_block(uint32_t kind, uint64_t va_block_start) {
    /* This function is intentionally empty.
     * The eBPF uprobe attached to this function will intercept
     * the call and extract arguments from registers.
     */
}
```

**编译**: 重新编译 `uvm_allocator.abi3.so`

```bash
cd workloads/vllm/vllm/device_allocator
# 根据实际构建系统调整
make uvm_allocator.abi3.so
```

#### Step 1.3: Python 包装器

**文件**: `workloads/vllm/vllm_address_hint.py`

```python
#!/usr/bin/env python3
"""
vLLM Address Hint Bridge - Python wrapper for uvm_hint_block() C ABI
"""

import ctypes
import os
from typing import Optional

# Region kinds (must match C header and eBPF)
REGION_UNKNOWN = 0
REGION_WEIGHTS = 1
REGION_KV = 2
REGION_OTHER = 3

VA_BLOCK_SIZE = 2 * 1024 * 1024  # 2MB
VA_BLOCK_MASK = ~(VA_BLOCK_SIZE - 1)


class UVMHintBridge:
    """Bridge to report memory region types to eBPF via uvm_hint_block()"""
    
    def __init__(self, lib_path: Optional[str] = None):
        """
        Args:
            lib_path: Path to uvm_allocator.abi3.so. If None, auto-detect.
        """
        if lib_path is None:
            # Auto-detect: look for uvm_allocator.abi3.so in vllm package
            import vllm
            vllm_dir = os.path.dirname(vllm.__file__)
            lib_path = os.path.join(vllm_dir, "uvm_allocator.abi3.so")
        
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"uvm_allocator.so not found: {lib_path}")
        
        self.lib = ctypes.CDLL(lib_path)
        
        # void uvm_hint_block(uint32_t kind, uint64_t va_block_start)
        self.lib.uvm_hint_block.argtypes = [ctypes.c_uint32, ctypes.c_uint64]
        self.lib.uvm_hint_block.restype = None
        
        self.enabled = True
    
    def hint_block(self, kind: int, va_block_start: int):
        """Report a single 2MB VA block to eBPF"""
        if not self.enabled:
            return
        self.lib.uvm_hint_block(kind, va_block_start)
    
    def hint_tensor(self, kind: int, addr: int, size_bytes: int):
        """
        Report all 2MB blocks covered by a tensor.
        
        Args:
            kind: REGION_WEIGHTS, REGION_KV, etc.
            addr: Tensor data pointer (from tensor.data_ptr())
            size_bytes: Tensor size in bytes
        """
        if not self.enabled or size_bytes == 0:
            return
        
        # Calculate covered 2MB blocks
        start_block = addr & VA_BLOCK_MASK
        end_addr = addr + size_bytes
        end_block = (end_addr + VA_BLOCK_SIZE - 1) & VA_BLOCK_MASK
        
        # Report each block
        current = start_block
        while current < end_block:
            self.hint_block(kind, current)
            current += VA_BLOCK_SIZE
    
    def hint_tensor_list(self, kind: int, tensors):
        """
        Report multiple tensors (e.g., all weight tensors in a model).
        
        Args:
            kind: REGION_WEIGHTS, REGION_KV, etc.
            tensors: List of torch.Tensor objects
        """
        import torch
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                addr = tensor.data_ptr()
                size = tensor.numel() * tensor.element_size()
                self.hint_tensor(kind, addr, size)


# Global singleton
_bridge: Optional[UVMHintBridge] = None

def get_bridge() -> UVMHintBridge:
    """Get or create global UVMHintBridge instance"""
    global _bridge
    if _bridge is None:
        try:
            _bridge = UVMHintBridge()
        except Exception as e:
            print(f"[UVMHint] Failed to initialize bridge: {e}")
            # Create a dummy bridge that does nothing
            _bridge = UVMHintBridge.__new__(UVMHintBridge)
            _bridge.enabled = False
    return _bridge
```

---

### Phase 2: eBPF 程序实现

#### Step 2.1: 核心 eBPF 程序

**文件**: `extension/prefetch_vllm_address_driven.bpf.c`

由于内容较长，我将分块展示关键部分：


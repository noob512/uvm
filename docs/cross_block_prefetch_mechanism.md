# Cross-VA-Block Prefetch 设计方案

**日期**: 2026-02-26
**状态**: 设计中（待实现）
**前提**: MSched 复现中的 intra-block prefetch (always_max) + passive MRU eviction 已验证有效
**关联**: 从 `msched_reproduction_plan.md` 中独立出来的子项目

---

## 1. 背景

### 为什么需要 Cross-block Prefetch

在 MSched 复现实验中已确认：
- BPF always_max prefetch + passive MRU eviction 达到 pp=228, tg=78.7 tok/s（短序列 tg128 +74% over Phase 2 baseline；长序列 tg512 +63%）
- 但 chunk_trace 显示 **82% chunk thrashing 未改变** — 因为 intra-block prefetch 只扩展当前 2MB VA block 内的预取范围
- Eviction 优化天花板约 2-4% 额外提升（只改淘汰顺序，不改淘汰总量）
- **真正突破需要 cross-VA-block proactive prefetch** — 提前迁移相邻 VA block 的数据

### 驱动调研结论

已完成对 nvidia-uvm 驱动 migration API 的深入调研。实现 cross-block prefetch 的三种方案:

**方案 A: 新增 `bpf_uvm_prefetch_va_range()` kfunc**
- 内部调用 `uvm_migrate()` (在 `uvm_migrate.c:635`)
- 需要 `va_space` 指针（目前未暴露给 BPF）
- 锁序问题: fault path 已持有当前 block lock，adjacent block 加锁可能死锁
- 需要使用 `UVM_VA_BLOCK_LOCK_RETRY` pattern

**方案 B: Fault-batch 级别的新 struct_ops hook**
- 在 `service_fault_batch_dispatch()` 中添加 hook
- BPF 可以返回 "额外需要预取的 VA 地址列表"
- 驱动负责在当前 block 处理完后，依次迁移 adjacent blocks
- 避免锁序问题，但需要修改 fault 批处理逻辑

**方案 C: Deferred prefetch work queue**
- kfunc 将预取请求放入内核 work queue
- Worker thread 在 fault path 之外调用 `uvm_migrate()`
- 完全避免锁序问题
- 预取与当前 fault 处理并行
- **最安全、最容易实现、且自然提供 compute-migration overlap**

**推荐**: 方案 C (deferred work queue)。

---

## 2. 第一版实现（已废弃）

第一版使用自定义 kernel work queue，已实现并编译/加载成功，但 code review 发现严重问题：

| 问题 | 严重性 | 说明 |
|------|--------|------|
| **va_space 生命周期** | 高 | worker 持有 va_space 裸指针，进程退出后 use-after-free |
| **per-CPU 抢占安全** | 中 | set_context → BPF hook 之间无抢占保护，可能读错 CPU 的 context |
| **每次 fault 做 kmalloc** | 中 | 热路径 kmalloc(GFP_KERNEL) 有性能和 sleep 风险 |
| **违反 gpu_ext 设计原则** | 高 | 新增 ~100 行内核 C 代码（work queue、手动锁管理），BPF 仅做触发器。gpu_ext 的核心论点是 BPF 提供安全扩展，如果扩展需要大量内核修改就失去了意义 |

**结论**：旧方案本质上是 "用 BPF 触发一段不安全的内核代码"，需要重新设计。

---

## 3. 新方案：BPF Workqueue + `uvm_migrate()` kfunc

### 核心思路

用 BPF 自带的 workqueue 机制（`bpf_wq`，Linux 6.10+，我们是 6.15）替代自写的内核 work queue。BPF wq callback 跑在 **process context**（可 sleep），可以调用 sleepable kfunc。kfunc 内部包装 `uvm_migrate()` — UVM_MIGRATE ioctl 使用的同一函数。

```
旧方案（自写内核代码）：
  BPF hook → per-CPU buffer → 自写 kernel workqueue → kmalloc →
  手动 uvm_va_space_down_read → 手动 uvm_va_block_find →
  手动 UVM_VA_BLOCK_LOCK_RETRY → uvm_va_block_migrate_locked
  （~100 行新内核代码，va_space 生命周期问题，per-CPU 抢占问题）

新方案（BPF workqueue）：
  BPF hook → BPF map → bpf_wq_start() → BPF wq callback →
  bpf_gpu_migrate_range() kfunc → uvm_migrate_bpf() → uvm_migrate()
  （内核侧新增：uvm_migrate.c ~20 行 wrapper + uvm_bpf_struct_ops.c ~10 行 kfunc）
```

### `uvm_migrate()` API 分析（关键发现）

经过代码审查，`uvm_migrate()` 的实际情况：

```c
// uvm_migrate.c:635 — 10 个参数，static 函数
static NV_STATUS uvm_migrate(uvm_va_space_t *va_space,
                             struct mm_struct *mm,
                             NvU64 base, NvU64 length,
                             uvm_processor_id_t dest_id,
                             int dest_nid,
                             NvU32 migrate_flags,
                             uvm_va_range_managed_t *first_managed_range,
                             uvm_tracker_t *out_tracker,
                             uvm_processor_mask_t *gpus_to_check_for_nvlink_errors);
```

**关键点**：
1. **`static` 函数** — 不可直接从其他 .c 文件调用，需要在 `uvm_migrate.c` 中添加 wrapper
2. **不自己加锁** — 调用前 caller 必须已持有 `va_space read lock`；如果 `mm != NULL` 还需要 `mmap_lock`
3. **`mm=NULL` 可行** — 只要提供 `first_managed_range`（通过 `uvm_va_space_iter_managed_first()` 获取）。UVM managed allocation 不需要 mm（不走 HMM 路径）
4. **`uvm_api_migrate()` 的调用模式**（line 922-1039）：
   ```c
   mm = uvm_va_space_mm_or_current_retain_lock(va_space);  // 获取 mm + mmap_lock
   uvm_va_space_down_read(va_space);                        // 加 va_space read lock
   dest_id = dest_gpu ? dest_gpu->id : UVM_ID_CPU;         // 构造 processor_id
   status = uvm_migrate(va_space, mm, base, length, dest_id, ...);
   uvm_va_space_up_read(va_space);                          // 释放
   uvm_va_space_mm_or_current_release_unlock(va_space, mm); // 释放
   ```
5. **processor_id 构造**：`uvm_gpu_id_from_index(0)` 产生 GPU 0 的 id（值=1，因为 `UVM_ID_GPU0_VALUE=1`）

### 架构

```
┌─────────────────────────────────────────────────┐
│  BPF struct_ops hook (fault path, 非 sleepable)  │
│                                                   │
│  1. always_max intra-block prefetch               │
│  2. bpf_gpu_get_va_space() → 拿 va_space handle  │
│  3. bpf_gpu_get_block_end_va() → 当前 block 边界 │
│  4. 写 (va_space, next_addr, length) 到 BPF map   │
│  5. bpf_wq_start() → 调度异步 work               │
└────────────────────┬────────────────────────────┘
                     │ BPF map (预分配，无 kmalloc)
                     ▼
┌─────────────────────────────────────────────────┐
│  BPF wq callback (process context, 可 sleep)     │
│                                                   │
│  1. 从 BPF map 读 prefetch 请求                  │
│  2. bpf_gpu_migrate_range(va_space, addr, len)   │
│     └→ kfunc 调 uvm_migrate_bpf()               │
│        └→ mm = uvm_va_space_mm_or_current_...()  │
│        └→ uvm_va_space_down_read(va_space)       │
│        └→ uvm_migrate(...) — 处理 block find,    │
│           block lock + retry, migration           │
│        └→ uvm_va_space_up_read + mm release      │
└─────────────────────────────────────────────────┘
```

---

## 4. 内核侧改动

### 文件 1: `uvm_migrate.c` — 新增 wrapper 函数（~20 行）

因为 `uvm_migrate()` 是 static，需要在同文件中添加一个公开的 wrapper。这个 wrapper 复制 `uvm_api_migrate()` 的锁管理模式：

```c
// 新增在 uvm_migrate.c 中，uvm_api_migrate() 之后
NV_STATUS uvm_migrate_bpf(uvm_va_space_t *va_space, NvU64 base, NvU64 length,
                          uvm_processor_id_t dest_id)
{
    struct mm_struct *mm;
    NV_STATUS status;

    // 复制 uvm_api_migrate() 的锁模式
    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_read(va_space);

    status = uvm_migrate(va_space,
                         mm,
                         base,
                         length,
                         dest_id,
                         NUMA_NO_NODE,                              // dest_nid
                         0,                                         // migrate_flags
                         uvm_va_space_iter_managed_first(va_space, base, base),
                         NULL,                                      // out_tracker (同步)
                         NULL);                                     // nvlink errors

    uvm_va_space_up_read(va_space);
    if (mm)
        uvm_va_space_mm_or_current_release_unlock(va_space, mm);

    return status;
}
```

声明添加到 `uvm_migrate.h`：
```c
NV_STATUS uvm_migrate_bpf(uvm_va_space_t *va_space, NvU64 base, NvU64 length,
                          uvm_processor_id_t dest_id);
```

### 文件 2: `uvm_bpf_struct_ops.c` — 新增 2 个 kfunc

新增 kfunc：
```c
#include "uvm_migrate.h"

// kfunc 1: 获取当前 fault 的 va_space (从 per-CPU context)
__bpf_kfunc u64 bpf_gpu_get_va_space(void)
{
    struct uvm_bpf_prefetch_ctx *ctx = this_cpu_ptr(&bpf_prefetch_ctx);
    if (!ctx->va_block)
        return 0;
    return (u64)uvm_va_block_get_va_space(ctx->va_block);
}

// kfunc 2: sleepable — 从 bpf_wq callback 调用
__bpf_kfunc int bpf_gpu_migrate_range(u64 va_space_handle, u64 addr, u64 length)
{
    uvm_va_space_t *va_space = (uvm_va_space_t *)va_space_handle;
    if (!va_space || !length)
        return -EINVAL;
    return (int)uvm_migrate_bpf(va_space, addr, length,
                                uvm_gpu_id_from_index(0));
}
```

BTF 注册：
```c
BTF_ID_FLAGS(func, bpf_gpu_get_va_space)
BTF_ID_FLAGS(func, bpf_gpu_migrate_range, KF_SLEEPABLE)
```

### 文件 3: `uvm_perf_prefetch.c` — per-CPU context 设置

需要在 prefetch 计算前后设置 per-CPU context（保留现有 `set/clear_context` 机制）：
```c
struct uvm_bpf_prefetch_ctx {
    uvm_va_block_t *va_block;  // 只保留 block 指针
};
```

---

## 5. BPF 侧

新建 `extension/prefetch_cross_block_v2.bpf.c`，需要 `bpf_experimental.h`（从 `~/workspace/bpf-developer-tutorial/src/features/bpf_wq/` 复制）。

```c
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "bpf_experimental.h"
#include "uvm_types.h"
#include "bpf_testmod.h"

#define VA_BLOCK_SIZE (2ULL * 1024 * 1024)
#define PREFETCH_AHEAD_BLOCKS 2

// kfunc 声明
extern void bpf_gpu_set_prefetch_region(...) __ksym;
extern u64  bpf_gpu_get_va_space(void) __ksym __weak;
extern u64  bpf_gpu_get_block_end_va(void) __ksym __weak;
extern int  bpf_gpu_migrate_range(u64 va_space, u64 addr, u64 length) __ksym __weak;
extern void bpf_gpu_block_move_tail(...) __ksym;

// BPF maps
struct wq_elem { struct bpf_wq work; };
struct prefetch_req { u64 va_space; u64 addr; u64 length; };

struct { __uint(type, BPF_MAP_TYPE_ARRAY); __uint(max_entries, 4);
         __type(key, int); __type(value, struct wq_elem); } wq_map SEC(".maps");
struct { __uint(type, BPF_MAP_TYPE_ARRAY); __uint(max_entries, 4);
         __type(key, int); __type(value, struct prefetch_req); } req_map SEC(".maps");

// WQ callback — process context, 可 sleep, 调 sleepable kfunc
static int do_prefetch(void *map, int *key, void *value) {
    struct prefetch_req *req = bpf_map_lookup_elem(&req_map, key);
    if (req && req->va_space)
        bpf_gpu_migrate_range(req->va_space, req->addr, req->length);
    return 0;
}

SEC("struct_ops/gpu_page_prefetch")
int BPF_PROG(gpu_page_prefetch, ...) {
    // 1) intra-block: always_max
    uvm_page_index_t max_first = BPF_CORE_READ(max_prefetch_region, first);
    uvm_page_index_t max_outer = BPF_CORE_READ(max_prefetch_region, outer);
    bpf_gpu_set_prefetch_region(result_region, max_first, max_outer);

    // 2) cross-block: BPF workqueue 调度异步 prefetch
    u64 va_space = bpf_gpu_get_va_space();
    u64 block_end = bpf_gpu_get_block_end_va();
    if (va_space && block_end) {
        for (int i = 0; i < PREFETCH_AHEAD_BLOCKS && i < 4; i++) {
            struct prefetch_req req = {
                .va_space = va_space,
                .addr = block_end + 1 + (u64)i * VA_BLOCK_SIZE,
                .length = VA_BLOCK_SIZE,
            };
            bpf_map_update_elem(&req_map, &i, &req, 0);

            struct wq_elem *elem = bpf_map_lookup_elem(&wq_map, &i);
            if (elem) {
                bpf_wq_init(&elem->work, &wq_map, 0);
                bpf_wq_set_callback(&elem->work, do_prefetch, 0);
                bpf_wq_start(&elem->work, 0);
            }
        }
    }
    return 1; // BYPASS
}

// + passive MRU eviction hooks (同 prefetch_always_max_passive_mru.bpf.c)
```

---

## 6. 安全考虑

### va_space 生命周期
- 仍存在风险：BPF wq callback 执行时 va_space 可能已被释放
- 缓解措施（按优先级）：
  1. **实际场景安全**：benchmark 期间 llama-bench 不会退出，va_space 始终有效
  2. **kfunc 内验证**（v2）：`bpf_gpu_migrate_range()` 检查 va_space 是否在全局 active 集合中
  3. **最严格方案**（v3）：在 `uvm_va_space_destroy()` 中等待 pending BPF wq callback 完成
- **v1 先用方案 1**（benchmark 安全），后续按需加验证

### bpf_wq 在 struct_ops 中的可用性
- `bpf_wq` 是 Linux 6.10+ 的 kfunc，注册在 `bpf_common_kfunc_set` 中
- 需要验证 struct_ops 程序能否调用这些 kfunc（kernel 6.15 应该支持）
- 如果不支持：使用 fallback 方案

### sleepable kfunc
- `bpf_gpu_migrate_range` 用 `KF_SLEEPABLE` flag 注册
- 只有 sleepable context (bpf_wq callback) 能调用
- struct_ops hook 本身不能调用（非 sleepable）— 这正好保证安全

---

## 7. Fallback 方案

### Fallback A: 内核侧 kthread + BPF map
- 仍只新增 `bpf_gpu_migrate_range()` kfunc（不变）
- 在 struct_ops `reg()` 时创建 kthread，`unreg()` 时停止
- BPF hook 写请求到 BPF map + 设置一个标志位
- kthread 轮询标志位，有请求时调 `bpf_gpu_migrate_range()` 对应的内核函数
- 比旧方案安全：kthread 生命周期绑定 struct_ops，用高层 `uvm_migrate_bpf()` API

### Fallback B: 纯用户态（LD_PRELOAD + ringbuf）
- BPF hook 写 fault 地址到 ring buffer
- LD_PRELOAD 注入后台线程到 llama-bench 进程
- 后台线程读 ringbuf，调 `cuMemPrefetchAsync()`
- 完全不改内核，但需要 LD_PRELOAD

---

## 8. 模块安全机制

**原则**：自定义 nvidia-uvm.ko 仅通过 `insmod` 显式加载，**重启后自动恢复 stock 模块**。

**当前机制**（已满足安全要求）：
```
Stock 模块: /lib/modules/6.15.11-061511-generic/updates/dkms/nvidia-uvm.ko.zst
  srcversion: D6CE1151F2DCBD6B83A6E0A
Custom 模块: ~/workspace/gpu/gpu_ext/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko

modprobe nvidia_uvm → 永远加载 stock（DKMS 安装路径）
insmod /path/to/custom/nvidia-uvm.ko → 加载 custom
重启 → systemd/udev 用 modprobe → stock ✓
```

**操作规程**：
```bash
# 加载 custom 模块（测试前）
sudo rmmod nvidia_uvm nvidia_modeset nvidia
sudo insmod ~/workspace/gpu/gpu_ext/kernel-module/nvidia-module/kernel-open/nvidia.ko
sudo insmod ~/workspace/gpu/gpu_ext/kernel-module/nvidia-module/kernel-open/nvidia-modeset.ko
sudo insmod ~/workspace/gpu/gpu_ext/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko

# 恢复 stock 模块（测试后或出问题时）
sudo rmmod nvidia_uvm nvidia_modeset nvidia
sudo modprobe nvidia   # 加载 stock

# 验证当前加载的模块
cat /sys/module/nvidia_uvm/srcversion
# stock: D6CE1151F2DCBD6B83A6E0A
# custom: 不同值

# 紧急恢复（如果 GPU hang）
# 方法 1: 重启（自动加载 stock）
sudo reboot
# 方法 2: 强制卸载
sudo rmmod -f nvidia_uvm && sudo modprobe nvidia
```

**额外安全措施**：
- 永远不要把 custom .ko 安装到 `/lib/modules/` 目录
- 永远不要运行 `make install` 或 `dkms install` 用 custom 模块
- 编译 custom 模块时只在源码目录 `make modules`，不做 install

---

## 9. 实现步骤

1. **在 `uvm_migrate.c` 中添加 `uvm_migrate_bpf()` wrapper**
   - 复制 `uvm_api_migrate()` 的锁管理模式
   - mm 通过 `uvm_va_space_mm_or_current_retain_lock()` 获取
   - va_space read lock 通过 `uvm_va_space_down_read()` 获取
   - 调用 static `uvm_migrate()` 然后释放锁
   - 在 `uvm_migrate.h` 中添加声明

2. **新增 2 个 kfunc**（`uvm_bpf_struct_ops.c`）
   - `bpf_gpu_get_va_space()` — 返回当前 fault 的 va_space handle (u64)
   - `bpf_gpu_migrate_range()` — sleepable kfunc, 调用 `uvm_migrate_bpf()`
   - 注册到 BTF kfunc set，`bpf_gpu_migrate_range` 加 `KF_SLEEPABLE` flag

3. **验证 bpf_wq 兼容性**
   - 写一个最小测试：struct_ops hook 中调 `bpf_wq_start()`
   - 确认 kernel 6.15 允许 struct_ops 程序使用 `bpf_wq` kfuncs
   - 如果不允许，实施 Fallback A（kthread）

4. **编写新 BPF 程序** `extension/prefetch_cross_block_v2.bpf.c`
   - 复制 `bpf_experimental.h` 到 `extension/`
   - Prefetch: always_max + bpf_wq cross-block
   - Eviction: passive MRU（从 `prefetch_always_max_passive_mru.bpf.c` 复制）

5. **编译测试**
   - 先恢复 stock 模块确认 GPU 正常
   - 编译 custom nvidia-uvm.ko（改了 `uvm_migrate.c` 和 `uvm_bpf_struct_ops.c`）
   - `insmod` 加载 custom 模块
   - 加载 BPF 程序，验证所有 kfunc 解析（包括 `bpf_gpu_migrate_range`）
   - 运行 llama-bench 120B 快速测试（2 reps）

6. **Benchmark**
   - Baseline: 无 BPF (`GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`, pp=512, tg=128, r=5)
   - passive MRU: always_max + passive MRU eviction（已有数据: pp=228, tg=78.7）
   - cross-block v2: always_max + bpf_wq cross-block + passive MRU
   - 对比分析

---

## 10. 预期对比

| 方案 | 内核新增代码 | 安全性 | 论文契合度 |
|------|------------|--------|-----------|
| 旧方案 (kernel wq) | ~100 行 | 差 (va_space, kmalloc) | 弱 |
| **新方案 (bpf_wq)** | **~30 行** (20 行 wrapper + 10 行 kfunc) | **好** (uvm_migrate 处理迁移，wrapper 处理锁) | **强** (BPF 全栈) |

---

## 11. 待确认风险

- [ ] `bpf_wq` 是否可用于 struct_ops 程序 → **未实际尝试**，v1 跳过直接用了内核 workqueue。bpf_wq 在 kernel 6.15 可用（已确认 API 存在），需要写 v2 验证
- [ ] bpf_wq callback 能否调用模块注册的 KF_SLEEPABLE kfunc → 待验证
- [x] `uvm_va_space_mm_or_current_retain_lock()` → 绕过，`uvm_migrate_bpf()` 使用 `mm=NULL` + `first_managed_range`
- [x] va_space 裸指针 → v1 benchmark 期间安全（llama-bench 不退出）

---

## 12. 所有测试过的算法详解

本节对项目中测试过的所有 prefetch 和 eviction 算法做简要说明，方便理解后续实验结果表中各配置的含义。

### 12.1 背景：UVM Demand Paging 工作流程

GPU 通过 UVM（Unified Virtual Memory）访问超过显存容量的数据。当 GPU 访问不在显存中的数据时：

```
GPU 执行 kernel → 访问不在 VRAM 的页 → GPU MMU 产生 page fault
→ 中断通知 CPU → UVM fault handler:
    1. 选择要预取的范围 (prefetch policy)
    2. 如果 VRAM 已满，选择要驱逐的 chunk (eviction policy)
    3. DMA 传输：evict D2H + fault-in H2D
→ GPU 恢复执行
```

UVM 将虚拟地址空间划分为 **2MB VA block**，每个 VA block 包含 512 个 4KB 页。Prefetch policy 决定"一次 fault 带多少数据进 VRAM"，eviction policy 决定"VRAM 满时先淘汰谁"。

BPF struct_ops hook 在两个决策点介入：
- `gpu_page_prefetch`: 在 bitmap tree 遍历前调用，可 BYPASS（跳过内核算法，使用 BPF 设定的区域）、DEFAULT（走内核默认）或 ENTER_LOOP（让内核遍历但每层调 BPF）
- `gpu_block_access`: 在 chunk 被使用时调用，可控制 chunk 在 eviction list 中的位置（move_head = 优先淘汰，move_tail = 保护，BYPASS 不移动 = 冻结当前位置）

---

### 12.2 Prefetch 算法

#### Baseline（无 BPF，threshold=51）

**UVM 内核默认算法**。每次 page fault 时：
1. 在当前 2MB VA block 上构建 bitmap tree（虚拟二叉树，512 个叶子）
2. 从 fault page 所在叶子向上遍历，每层检查：`populated_count * 100 > subregion_pages * 51`
3. 只要子树中已有页占比超过 51%，就扩展 prefetch 范围到该子树
4. 最终 prefetch 区域 = 满足条件的最大连续子树

**对 MoE 的问题**：一个几乎空的 VA block 第一次 fault 时，populated 占比极低（1/512 = 0.2%），threshold=51% 几乎不会扩展，导致每个 VA block 需要大量 fault 才能把数据全部搬入。

```c
// 内核默认逻辑 (uvm_perf_prefetch.c)
for (level = 0; level < tree_height; level++) {
    if (populated * 100 > subregion_pages * threshold)
        expand region to this subtree;
}
```

#### always_max（BPF BYPASS，预取整个 VA block）

**最简单也是收益最大的策略**。每次 fault 时，直接把 prefetch 区域设为整个 VA block 的 `max_prefetch_region`，跳过 bitmap tree 遍历。

```c
// prefetch_always_max.bpf.c
SEC("struct_ops/gpu_page_prefetch")
int BPF_PROG(gpu_page_prefetch, ...) {
    // 取 max_prefetch_region 的 first 和 outer（整个 VA block 范围）
    uvm_page_index_t max_first = BPF_CORE_READ(max_prefetch_region, first);
    uvm_page_index_t max_outer = BPF_CORE_READ(max_prefetch_region, outer);
    // 设 result = max → 预取整个 VA block 所有 non-resident 页
    bpf_gpu_set_prefetch_region(result_region, max_first, max_outer);
    return 1; // BYPASS — 跳过内核 bitmap tree 算法
}
```

**效果**：等效于 threshold=0。一次 fault 把整个 2MB VA block 的所有页搬入 VRAM，消除同一 block 内后续的 page fault。不减少 chunk 级别的 thrashing（82% re-fault 不变），但大幅减少 page-level fault 数量（从 ~400/token 降到 ~107/token）。

#### stride（步长模式预测）

检测连续 fault 地址的步长模式，预测下一次 fault 位置并预取。使用置信度衰减机制：连续命中增加置信度，miss 降低置信度。

```c
// 简化逻辑
stride = current_fault_addr - last_fault_addr;
if (stride == predicted_stride)
    confidence++;  // 命中
else
    confidence--;  // miss, 衰减
if (confidence >= threshold)
    prefetch_region = [current + stride, current + stride * pages];
    return 1; // BYPASS
```

**MoE 的灾难**：MoE 模型的内存访问在层间跳跃（非线性），stride 检测极少成功（仅 8% fault 触发预取）。关键问题是 stride 策略返回 BYPASS 即使未做预取 — 这阻止了内核默认预取，等效于禁用预取。

#### none（完全禁用预取）

返回空 region + BYPASS，强制每个页单独 fault。用作性能下限参考。

#### cross-block aggressive（跨 VA block 激进预取，每次 fault 预取 2 个相邻 block）

在 always_max（intra-block 全量预取）基础上，额外通过 kernel workqueue 异步迁移当前 VA block 之后的 2 个相邻 2MB block。每次 fault 都触发。

```c
// prefetch_cross_block.bpf.c (aggressive 版本)
SEC("struct_ops/gpu_page_prefetch")
int BPF_PROG(gpu_page_prefetch, ...) {
    // 1) Intra-block: always_max
    bpf_gpu_set_prefetch_region(result_region, max_first, max_outer);

    // 2) Cross-block: 预取后面 2 个 VA block
    u64 block_end = bpf_gpu_get_block_end_va();
    if (block_end > 0) {
        // 请求迁移 block_end+1 开始的 4MB (2 blocks × 2MB)
        bpf_gpu_request_prefetch_range(block_end + 1, 2 * VA_BLOCK_SIZE);
    }
    return 1; // BYPASS
}
```

**内核侧实现**：`bpf_gpu_request_prefetch_range()` 将请求放入 64-slot ring buffer，workqueue worker 线程在 fault handler 外调用 `uvm_migrate_bpf()` 执行实际迁移。

#### cross-block rate-limited（跨 VA block 限速预取，每个新 block 仅 1 次）

相比 aggressive 版本，增加去重：用 ARRAY map 记录上次预取的 block VA，只在进入**新** VA block 时才预取 1 个相邻 block。同一 block 内的后续 fault 不触发额外预取。

```c
// prefetch_cross_block.bpf.c (rate-limited 版本)
u64 block_end = bpf_gpu_get_block_end_va();
if (block_end > 0) {
    u32 zero = 0;
    u64 *last = bpf_map_lookup_elem(&last_prefetch_block, &zero);
    if (last && *last != block_end) {
        *last = block_end;  // 记录已预取此 block
        bpf_gpu_request_prefetch_range(block_end + 1, VA_BLOCK_SIZE); // 仅 1 block
    }
}
```

---

### 12.3 Eviction 算法

#### LRU（UVM 默认）

内核默认：chunk 使用时移到 eviction list 尾部（最后淘汰），头部优先淘汰。经典 LRU。

#### MRU（纯 Most Recently Used）

最近使用的 chunk 移到 eviction list 头部（最先淘汰）。理论上对周期性访问模式（LLM decode 循环遍历所有层）是 Belady-optimal，因为"刚用过 = 离下次使用最远"。

**灾难性后果**：MRU 无差别地把 attention weights（每步都用的 T1 数据）也移到头部淘汰 → 每步都要重新搬入 attention → -83% 性能。

#### cycle_moe（T1 频率保护 + 默认处理其余）

**核心思想**：只保护高频 chunk（attention + embeddings），其余让内核默认 LRU 处理。

```c
// eviction_cycle_moe.bpf.c
SEC("struct_ops/gpu_block_access")
int BPF_PROG(gpu_block_access, ...) {
    u32 idx = chunk_hash(chunk);
    u8 *count = bpf_map_lookup_elem(&access_counts, &idx);
    if (!count) return 0;
    if (++(*count) >= T1_FREQ_THRESHOLD) {  // T1_FREQ_THRESHOLD = 3
        bpf_gpu_block_move_tail(chunk, list);  // 保护（移到尾部）
        return 1; // BYPASS
    }
    return 0;  // DEFAULT — 让内核 LRU 处理
}
```

**使用 PERCPU_ARRAY 而非 HASH map**：HASH map 在 fault handler 热路径中延迟过高，导致 GPU MMU timeout → Xid 31 crash。

#### MRU expert（T1 保护 + 非 T1 用 move_head）

T1 chunk（freq ≥ 3）用 `move_tail` 保护；非 T1 chunk 用 `move_head` 显式移到头部优先淘汰。

```c
if (freq >= T1_THRESHOLD) {
    bpf_gpu_block_move_tail(chunk, list);  // T1: 保护
} else {
    bpf_gpu_block_move_head(chunk, list);  // 非 T1: 优先淘汰
}
return 1; // BYPASS
```

**问题**：`move_head` 的 list manipulation 开销（spinlock + pointer update）抵消了 MRU 排序的理论收益。

#### passive MRU（T1 保护 + 非 T1 冻结 LRU 位置）

T1 chunk 用 `move_tail` 保护；非 T1 chunk 返回 BYPASS **但不调用任何 move 函数** — 这阻止了内核默认的 LRU 刷新（move_tail），chunk 保持在当前 list 位置。随着新 chunk 在 tail 端添加，旧 chunk 自然向 head 漂移，效果 ≈ FIFO。

```c
if (freq >= T1_THRESHOLD) {
    bpf_gpu_block_move_tail(chunk, list);  // T1: 保护
    return 1; // BYPASS
}
// 非 T1: BYPASS 但不 move — 冻结当前 LRU 位置
return 1; // BYPASS, 无 move = passive MRU
```

**关键优势**：零 list manipulation 开销（不调用任何 move 函数），仅靠 "不做默认动作" 就实现了近似 MRU。

#### template_belady（Belady 距离 eviction）

**核心思想**：MSched 的 Belady OPT eviction 的 BPF 实现。从 chunk 的 VA 地址推断其所属层号（通过离线 chunk_trace 建立的 VA→layer 映射表），然后计算到当前层的**周期距离**（cycle distance），距离远的优先淘汰。

```c
SEC("struct_ops/gpu_block_access")
int BPF_PROG(gpu_block_access, ...) {
    // 1) T1 频率保护（同 passive MRU）
    if (freq >= T1_THRESHOLD) {
        bpf_gpu_block_move_tail(chunk, list);
        return 1;
    }
    // 2) 获取 chunk VA → 查 boundary table → layer_id
    uvm_va_block_t *va_block = BPF_CORE_READ(chunk, va_block);
    u64 chunk_va = BPF_CORE_READ(va_block, start);
    u32 chunk_layer = va_to_layer(chunk_va);  // linear scan 36 boundaries

    // 3) Belady 距离 = 到下次使用的 cycle 距离
    u32 distance = (chunk_layer - current_layer + NUM_LAYERS) % NUM_LAYERS;

    if (distance <= protect_distance) {
        bpf_gpu_block_move_tail(chunk, list);  // 即将使用 → 保护
    } else {
        bpf_gpu_block_move_head(chunk, list);  // 远距离 → 优先淘汰
    }
    return 1;
}
```

**VA→layer 映射表**：由离线 `derive_layer_mapping.py` 从 chunk_trace 数据生成。将 prefill 阶段激活的 15,801 个 chunk 按 VA 排序后等分为 36 组（对应 36 层），生成 36 个 VA 边界值存入 BPF ARRAY map。

---

### 12.4 组合策略总表

| 配置名称 | Prefetch 算法 | Eviction 算法 | 文件 |
|---------|--------------|--------------|------|
| Baseline (no BPF) | 内核默认 threshold=51 | 内核默认 LRU | — |
| threshold=N | 内核 threshold=N | 内核默认 LRU | 模块参数 |
| always_max | always_max (BYPASS) | 内核默认 LRU | `prefetch_always_max.bpf.c` |
| stride | stride 模式 (BYPASS) | 内核默认 LRU | `prefetch_stride.bpf.c` |
| none | 空 region (BYPASS) | 内核默认 LRU | — |
| always_max + cycle_moe | always_max | T1 protect + DEFAULT | `prefetch_always_max_cycle_moe.bpf.c` |
| always_max + MRU expert | always_max | T1 protect + move_head | `prefetch_max_mru_expert.bpf.c` |
| always_max + passive MRU | always_max | T1 protect + freeze | `prefetch_max_passive_mru.bpf.c` |
| template_belady | always_max | T1 protect + Belady distance | `prefetch_template_belady.bpf.c` |
| cross-block aggressive | always_max + 2 adjacent blocks | T1 protect + freeze | `prefetch_cross_block.bpf.c` |
| cross-block rate-limited | always_max + 1 adjacent block (dedup) | T1 protect + freeze | `prefetch_cross_block.bpf.c` |
| MRU (纯) | 内核默认 | move_head (全部) | — |

---

## 13. 实验结果（2026-02-27）

### 实现方案

最终采用**内核 workqueue** 方案（非 bpf_wq）：
- 3 个新 kfunc: `bpf_gpu_get_block_start_va()`, `bpf_gpu_get_block_end_va()`, `bpf_gpu_request_prefetch_range()`
- Per-CPU prefetch context: 在 `rcu_read_lock()` 下设置，保证抢占安全
- Ring buffer (64 slots) + workqueue worker: BPF kfunc 入队，worker 调用 `uvm_migrate_bpf()`
- `uvm_migrate_bpf()`: 包装 static `uvm_migrate()`，自动管理 va_space read lock
- 内核侧新增 ~80 行，BPF 侧复用已有 `prefetch_cross_block.bpf.c`
- 编译通过，加载成功，无 Xid 错误

### Benchmark 结果

| 配置 | pp (tok/s) | tg (tok/s) | vs always_max |
|------|-----------|-----------|---------------|
| Baseline (no BPF, threshold=51) | 142.93 ± 0.47 | 47.24 ± 6.05 | — |
| always_max + passive MRU | 223.86 ± 0.51 | 78.39 ± 6.88 | — (baseline) |
| **cross-block 2 blocks (aggressive)** | 191.85 ± 4.30 | 62.61 ± 7.62 | **-14% pp, -20% tg** |
| **cross-block 1 block (rate-limited)** | 206.67 ± 3.43 | 60.78 ± 7.04 | **-8% pp, -22% tg** |

### 分析：为什么 cross-block prefetch 有害

**根本原因: 1.84x oversubscription 下，proactive prefetch 是零和博弈。**

每个 proactively prefetched 的 2MB block 必然 evict 一个 useful block：
- 当前 VRAM 容量 = ~30 GB（32 GB 减去系统开销）
- 模型大小 = 59 GB
- Per token 需要 107 chunks (214 MB) 迁入 + 107 chunks (214 MB) 逐出 = 428 MB 总 DMA
- Cross-block aggressive: ~100 extra blocks/token ≈ +200 MB DMA (+47% 额外流量)
- Cross-block rate-limited: ~100 extra blocks/token（每个新 block 仅一次，但仍然太多）

被 evict 的 block 之后还会被 demand-page 回来，形成恶性循环：
```
prefetch block N+1 → evict block M → later fault on M → migrate M back → evict block K → ...
```

### 什么条件下 cross-block prefetch 有效

| 条件 | 当前情况 | 需要的情况 |
|------|---------|-----------|
| Oversubscription | 1.84x | < 1.0x（全部放得下） |
| VRAM headroom | 0（满） | 有空闲槽位放 prefetch |
| Prefetch 精度 | 盲目 adjacent | 精确 next-layer-only |
| DMA 通道 | 共享（单 CE） | 独立（双 CE，MSched 方式） |

### 初步结论（仅 llama.cpp 120B，1.84x oversubscription）

1. **技术实现成功**: kfunc + 内核 workqueue 方案工作正常，无 crash/Xid，zero overhead when BPF not active
2. **性能负增益**: 在 1.84x oversubscription 下，盲目 adjacent prefetch 有害
3. **Cross-block prefetch 的价值场景**: oversubscription < 1.2x 时可能有正增益（VRAM 有余量放 prefetch）
4. **MSched 方式的优势**: 它用独立 CE + 精确 template 做 proactive migration，绕过 UVM demand paging，而非在 UVM 内部做额外 migration

**待验证**（§14）：
- v1 只测了 llama.cpp 120B 一个 workload，结论可能不具普适性
- 应在不同 oversubscription 级别和访问模式下测试（使用 `microbench/memory` 工具）
- 应尝试 bpf_wq 方案替代内核 workqueue，减少内核代码量

---

## 14. 下一步：bpf_wq 方案 + 多 workload 验证

### 14.1 为什么需要 bpf_wq 版本

v1 使用内核 workqueue 存在以下问题：

| 问题 | 内核 workqueue (v1) | bpf_wq (v2) |
|------|-------------------|-------------|
| 内核新增代码 | ~80 行（ring buffer + spinlock + worker + wq lifecycle） | ~15 行（仅 2 个 kfunc） |
| 安全性 | ring buffer 满时静默丢请求 | BPF verifier 保证安全 |
| 符合 gpu_ext 哲学 | 弱（大量内核基础设施代码） | 强（逻辑在 BPF 中） |
| 可扩展性 | 修改逻辑需重编内核模块 | 修改 BPF 程序即可 |

### 14.2 bpf_wq 方案设计

**内核侧改动**（替换 v1 的 ring buffer/worker，净减 ~45 行）：

```c
// 删除: ring buffer, spinlock, kernel workqueue, bpf_prefetch_worker()
// 删除: bpf_gpu_request_prefetch_range() kfunc

// 新增 kfunc 1: 获取当前 fault 的 va_space handle
__bpf_kfunc u64 bpf_gpu_get_va_space(void)
{
    struct uvm_bpf_prefetch_ctx *ctx = this_cpu_ptr(&bpf_prefetch_ctx);
    return ctx->va_space ? (u64)ctx->va_space : 0;
}

// 新增 kfunc 2: sleepable — 从 bpf_wq callback 调用
__bpf_kfunc int bpf_gpu_migrate_range(u64 va_space_handle, u64 addr, u64 length)
{
    uvm_va_space_t *va_space = (uvm_va_space_t *)va_space_handle;
    if (!va_space || !length)
        return -EINVAL;
    return (int)uvm_migrate_bpf(va_space, addr, length);
}

// BTF 注册:
BTF_ID_FLAGS(func, bpf_gpu_get_va_space)
BTF_ID_FLAGS(func, bpf_gpu_migrate_range, KF_SLEEPABLE)
```

保留：per-CPU context、`bpf_gpu_get_block_start_va()`、`bpf_gpu_get_block_end_va()`、`uvm_migrate_bpf()` wrapper。

**BPF 侧**（`extension/prefetch_cross_block_v2.bpf.c`）：

```c
#include "bpf_experimental.h"  // bpf_wq API

struct prefetch_data {
    u64 va_space;
    u64 addr;
    u64 length;
    struct bpf_wq work;
};

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, int);
    __type(value, struct prefetch_data);
} wq_map SEC(".maps");

// bpf_wq callback — 进程上下文，可 sleep，调用 sleepable kfunc
static int do_prefetch(void *map, int *key, void *value)
{
    struct prefetch_data *data = value;
    if (data && data->va_space)
        bpf_gpu_migrate_range(data->va_space, data->addr, data->length);
    return 0;
}

SEC("struct_ops/gpu_page_prefetch")
int BPF_PROG(gpu_page_prefetch, ...) {
    // 1) intra-block: always_max
    bpf_gpu_set_prefetch_region(result_region, max_first, max_outer);

    // 2) cross-block: bpf_wq 调度异步 prefetch
    u64 va_space = bpf_gpu_get_va_space();
    u64 block_end = bpf_gpu_get_block_end_va();
    if (va_space && block_end > 0) {
        u32 zero = 0;
        u64 *last = bpf_map_lookup_elem(&last_prefetch_block, &zero);
        if (last && *last != block_end) {
            *last = block_end;
            int key = 0;
            struct prefetch_data *data = bpf_map_lookup_elem(&wq_map, &key);
            if (data) {
                data->va_space = va_space;
                data->addr = block_end + 1;
                data->length = VA_BLOCK_SIZE;
                bpf_wq_init(&data->work, &wq_map, 0);
                bpf_wq_set_callback(&data->work, do_prefetch, 0);
                bpf_wq_start(&data->work, 0);
            }
        }
    }
    return 1; // BYPASS
}
```

**待验证的关键问题**：
1. struct_ops BPF 程序能否调用 `bpf_wq_init/set_callback/start`（属于 `bpf_common_kfunc_set`）
2. bpf_wq callback 能否调用 nvidia-uvm.ko 注册的 `KF_SLEEPABLE` kfunc
3. 如果不行，fallback: 保留当前 v1 内核 workqueue 方案

### 14.3 多 workload 验证计划

v1 结论仅基于 llama.cpp 120B（1.84x oversubscription），需要用 `microbench/memory` 在不同条件下验证。

**实验矩阵**：

| Kernel | 访问模式 | size_factor | 预期 oversubscription |
|--------|---------|-------------|----------------------|
| `seq_stream` | 顺序流式 | 0.5, 1.0, 1.5, 2.0 | 0x → 2x |
| `hotspot` | 5-point stencil | 0.5, 1.0, 1.5 | 0x → 1.5x |
| `gemm` | LLM 式多层权重复用 | 0.5, 1.0, 1.5 | 0x → 1.5x |
| `rand_stream` | 随机访问 | 0.5, 1.0, 1.5 | 0x → 1.5x |

**对比配置**：
1. Baseline (no BPF) — UVM 默认 threshold=51
2. always_max + passive MRU — 当前最佳
3. cross-block v1 (内核 wq) 或 v2 (bpf_wq) — 待测

**关键假设验证**：
- size_factor=0.5（全部放得下 VRAM）: cross-block 应无影响（不触发 eviction）
- size_factor=1.0（刚好满）: cross-block 可能有正收益（少量 headroom）
- size_factor=1.5-2.0（严重 oversubscription）: cross-block 预期有害（与 120B 结论一致）

### 14.4 实施步骤

1. **bpf_wq 可行性验证**: 修改内核模块 → 编译 → 写 v2 BPF 程序 → 编译加载测试
2. **microbench baseline**: stock module，no BPF，各 kernel × size_factor
3. **microbench + BPF**: custom module，always_max vs cross-block，各 kernel × size_factor
4. **结果分析**: 找到 cross-block 的收益/损害临界点
5. **更新结论**: 将 §13 的 "初步结论" 升级为完整结论

---

## 15. bpf_wq v2 实现与验证结果（2026-02-27）

### 15.1 bpf_wq 可行性：已验证通过

**关键发现**：struct_ops BPF 程序可以使用 `bpf_wq` kfunc，且 bpf_wq callback 可以调用模块注册的 `KF_SLEEPABLE` kfunc。

kfunc 解析日志：
```
bpf_gpu_get_va_space     → nvidia_uvm [150212]
bpf_gpu_migrate_range    → nvidia_uvm [150215]  (KF_SLEEPABLE)
bpf_wq_init              → vmlinux [73557]
bpf_wq_set_callback_impl → vmlinux [73559]
bpf_wq_start             → vmlinux [73561]
```

**实现细节**：
- 内核模块改动：删除 ring buffer/spinlock/kernel workqueue (~55 行)，新增 `bpf_gpu_get_va_space()` + `bpf_gpu_migrate_range(KF_SLEEPABLE)` (~20 行)，净减 ~35 行
- BPF 程序：`extension/prefetch_cross_block_v2.bpf.c`，使用 `bpf_experimental.h` 中的 bpf_wq API
- 需要从 running kernel BTF 重新生成 `vmlinux.h`（旧版本缺少 `struct bpf_wq` 定义）
- 编译/加载/运行均无错误，dmesg 无 Xid，无 GPU fault

### 15.2 Microbench 对比结果

**测试环境**：custom nvidia-uvm.ko (v2 kfuncs)，RTX 5090 32GB，kernel 6.15.11

#### Baseline (stock module, no BPF, threshold=51)

| Kernel | size_factor | median_ms | bandwidth GB/s |
|--------|------------|-----------|---------------|
| seq_stream | 0.5 | 0.570 | 29534 |
| seq_stream | 1.0 | 120.4 | 280 |
| seq_stream | 1.5 | 1858.6 | 27 |
| seq_stream | 2.0 | TIMEOUT | — |
| hotspot | 0.5 | 343.0 | 490 |
| hotspot | 1.0 | 2169 | 155 |
| gemm | 0.5 | 1228 | 137 |

#### always_max + passive MRU vs cross-block v2

| Workload | size_factor | always_max (ms) | cross-block v2 (ms) | 提升 |
|----------|-----------|-----------------|---------------------|------|
| seq_stream | 0.5 | 0.570 | 0.572 | **0%** (fits in VRAM) |
| **seq_stream** | **1.0** | **53.1** | **32.6** | **+63%** |
| seq_stream | 1.5 | 2520 | 2340 | **+8%** |
| hotspot | 0.5 | 343.0 | 342.9 | **0%** (fits in VRAM) |
| **hotspot** | **1.0** | **1796** | **1562** | **+15%** |

### 15.3 分析

**cross-block prefetch 在低/中 oversubscription 有显著正收益**：

1. **sf=0.5 (fits in VRAM)**: 零开销 — 数据全部在 VRAM，不触发 fault，cross-block 不触发。验证了 BPF hook 的 zero-overhead 特性。

2. **sf=1.0 (刚好满)**: **显著正收益** — seq_stream +63%, hotspot +15%。
   - 此时 VRAM 接近但不严重超额，proactive prefetch 能预先搬入即将使用的数据
   - 对 seq_stream（线性访问）效果最好：下一个 VA block 几乎必定命中
   - 对 hotspot（stencil 访问）也有收益：相邻 VA block 与空间局部性吻合

3. **sf=1.5 (1.5x oversubscription)**: **仍有 +8% 正收益**（seq_stream）。
   - 这与 llama.cpp 120B (1.84x) 的 -20% 负增益形成对比
   - 原因：seq_stream 是单次遍历（不循环），prefetched block 不会被后续 fault 驱逐
   - llama.cpp 120B 的 MoE decode 是循环遍历所有层，prefetched block 在一轮内必被驱逐

4. **关键区别 — 访问模式决定 cross-block 的价值**：
   - **单次/线性遍历** (seq_stream, hotspot stencil): cross-block 有效 — prefetch 的 block 会在驱逐前被使用
   - **循环遍历** (MoE decode): cross-block 有害 — VRAM 满时每次 prefetch 都驱逐有用数据，且 prefetched block 在下一轮才用到

### 15.4 结论更新

**修正 §13 的初步结论**：cross-block prefetch 并非 "在 oversubscription 下总是有害"，而是 **取决于访问模式**：

| 条件 | cross-block 效果 | 原因 |
|------|-----------------|------|
| fits in VRAM (sf≤0.8) | 零开销 | 不触发 fault/eviction |
| 刚满 (sf≈1.0) | **+15~63%** | 少量 headroom + prefetch 命中率高 |
| 中等 oversub + 线性访问 (sf=1.5) | **+8%** | prefetch block 在驱逐前被使用 |
| 高 oversub + 循环访问 (llama.cpp 1.84x) | **-20%** | prefetch 驱逐有用 block，形成 thrashing |

**bpf_wq vs 内核 workqueue**：
- bpf_wq 完全可行，且代码量大幅减少（内核侧净减 35 行）
- 性能表现等效（同一方案，仅执行路径不同）
- 更符合 gpu_ext 的 "BPF 全栈" 设计哲学
- **推荐 v2 (bpf_wq) 替代 v1 (内核 workqueue) 作为默认实现**

### 15.5 llama.cpp 120B 验证

cross-block v2 在 120B MoE 模型上确认负增益，与 v1 结论一致：

| 配置 | pp (tok/s) | tg (tok/s) | vs always_max |
|------|-----------|-----------|---------------|
| always_max + passive MRU | 221.4 ± 0.4 | 87.3 ± 7.1 | — |
| **cross-block v2 (bpf_wq)** | **214.0 ± 0.6** | **65.6 ± 7.1** | **-3% pp, -25% tg** |

**结论**：bpf_wq vs 内核 workqueue 对性能无影响（v1 -22% tg vs v2 -25% tg，在 noise range 内），问题出在 "盲目 adjacent block" 策略本身。

### 15.6 120B MoE Block 访问模式深度分析

基于 chunk_trace_120b_raw.csv（65,719 POPULATE 事件，12,131 unique blocks）的分析（`analyze_crossblock_v2.py`）：

#### Adjacent Block 命中率

| Prefetch Lookahead | 命中率 |
|-------------------|--------|
| +1 block (当前策略) | **26.4%** |
| +2 blocks | 31.3% |
| +3 blocks | 34.6% |
| +5 blocks | 39.5% |
| +10 blocks | 50.3% |

**结论**：盲目 adjacent 只能覆盖 ~26% 的 block transitions。即使 prefetch 10 个 adjacent blocks 也只覆盖 50%。MoE 模型的 block 访问高度非线性。

#### Jump Distance 分布

```
+1 blocks: 17131 (26.5%)   ← adjacent 命中
+2 blocks:  3167 ( 4.9%)
+3 blocks:  2117 ( 3.3%)
+5~+11:     各 1100-1700 (2-3%)
Far jumps:  40621 (62.8%)   ← 跳到完全不同的 VA 区域
```

**62.8% 的 block transitions 是 far jump**（跨越多个 VA region），这些对应 MoE expert 之间的切换。

#### History-Based Prediction

| History Length | Prediction Accuracy | Unique Patterns |
|---------------|-------------------|----------------|
| 1 (last jump) | 27.9% | 932 |
| 2 (last 2 jumps) | 39.5% | 12,330 |
| **3 (last 3 jumps)** | **67.7%** | **36,254** |

**关键发现**：最近 3 跳历史可达到 **67.7% 预测准确率**（vs adjacent 的 26.4%），但需要 36K patterns — 这可以用 BPF HASH map 存储，但在 fault handler 热路径查找可能太慢（之前已验证 HASH map 在热路径会导致 Xid 31）。

#### Consecutive Adjacent Run 长度

```
Mean: 1.3 blocks
78% of runs: length 1 (single adjacent step, then far jump)
Max: 22 blocks
```

Adjacent 步序非常短 — 大部分只走 1 步就跳到远处。这解释了为什么 cross-block prefetch 的 26% 命中率不能补偿 74% 的浪费。

#### VA Region 碎片化

- 12,113 unique blocks 分布在 **2,917 个 contiguous VA regions**
- 平均 region 大小: ~4 blocks (8 MB)
- 这意味着模型权重在 VA 空间中高度碎片化，不适合 adjacent prefetch

### 15.7 §15.6-§15.7 结论修正 (2026-02-28 深度分析)

**§15.6 的分析使用了短 trace (chunk_trace_120b_raw.csv, 65K events)，长 trace 分析 (chunk_trace_120b_long.csv, 358K events) 得出显著不同的结论。**

---

## 17. 深度分析结果 (2026-02-28)

### 17.1 方法论

两个新分析工具：

1. **`simulate_vram.py`** — VRAM 缓存模拟器
   - 将 ACTIVATE 事件作为 cache miss 重放
   - 可配置: VRAM 容量、eviction policy (LRU/FIFO/T1-protect)、prefetch strategy (none/adjacent/selective)
   - 新增 **prefetch_placement** 参数: MRU insert (真实 UVM 行为) vs LRU insert (优化方案)
   - 跟踪: demand faults、prefetch hits/wastes、net DMA change

2. **`analyze_crossblock_v3.py`** — 增强版访问模式分析
   - Prefill vs Decode 阶段分离
   - Per-layer transition 分类 (intra-layer vs inter-layer)
   - History-based prediction 压缩分析 (短 trace vs 长 trace 对比)

### 17.2 短 trace vs 长 trace 结论差异

| 指标 | 短 trace (65K events) | 长 trace (358K events) | 差异 |
|------|---------------------|----------------------|------|
| Adjacent hit rate (overall) | 26.4% | 16.5% | **-37% 下降** |
| History-1 accuracy | 27.9% | 17.4% | -38% |
| History-2 accuracy | 39.5% | 23.9% | -39% |
| History-3 accuracy | **67.7%** | **48.5%** | **-28% 下降** |
| History-3 unique patterns | 36K | **108K** | 3x 膨胀 |

**原因**: 短 trace 中 decode 阶段数据不足，prefill 占比过高导致 adjacent hit rate 虚高。长 trace 有更完整的 decode steady-state。

**结论**: §15.6 的 "67.7% 历史预测准确率" 被过度报告。实际 decode 阶段只有 48.5%，且 108K patterns 不可能装入 BPF map。

### 17.3 Prefill vs Decode 阶段分离

| 阶段 | Adjacent +1 hit rate | Adjacent +5 hit rate |
|------|---------------------|---------------------|
| Prefill (110 events, 634ms) | 35.8% | 59.6% |
| **Decode (287K events, 9045ms)** | **16.5%** | **39.2%** |
| Overall | 16.5% | 39.3% |

**关键发现**:
- Prefill 阶段的 adjacent hit rate 是 decode 的 **2x**
- 但 prefill 只占总事件的 0.04%，对性能影响微乎其微
- **Cross-block 对 decode（性能瓶颈阶段）帮助有限**: 83.5% 的 decode 转移不是 adjacent

### 17.4 Per-Layer Transition 分析

| 转移类型 | Decode 占比 | Adjacent hit rate |
|---------|------------|-------------------|
| Intra-layer (同层内) | 95.3% | 17.3% VA-adjacent |
| Inter-layer adjacent (L→L±1) | 4.4% | — |
| Inter-layer far | 0.3% | — |

**关键发现**:
- 95.3% 的 block transitions 发生在**同一层内** — 即 expert 之间切换
- 层内的 VA-adjacent rate 仅 17.3% — MoE expert 在 VA 空间中不连续分布
- Cross-block prefetch 本质上是在预取"同层内的下一个 VA block"，但同层 expert 之间不是连续排列的

### 17.5 VRAM 模拟器核心发现

#### Prefetch 插入位置的影响

在真实 UVM 中，`gpu_block_activate` 将新 chunk 放在 eviction list **尾部 (MRU端，受保护)**。模拟器测试了两种策略：

| 策略 | Eviction | Prefetch Hit Rate | Waste Rate | Net DMA Change |
|------|---------|-------------------|------------|----------------|
| **MRU insert (真实 UVM)** | T1-protect | **73.6%** | 20.8% | **+8,916 MB** (有害) |
| LRU insert (优化) | T1-protect | 61.7% | 35.3% | +15,092 MB (更有害) |
| **MRU insert** | LRU | **74.5%** | 17.1% | **+14,442 MB** (有害) |
| LRU insert | FIFO | 47.4% | 52.6% | **-8,686 MB** (有益!) |

**惊人发现**: MRU insert（真实行为）给出 73.6% prefetch hit rate — 远高于静态分析的 16.5% adjacent hit rate！这是因为 MRU insert 保护 prefetched data 不被立即淘汰，给它足够时间等到被 demand access。

**但即使 73.6% hit rate，cross-block 在所有配置下仍然 net harmful**（除了 FIFO+LRU insert 这个特殊组合）。原因：26,054 prefetches 只消除了 21,596 demand faults，多出的 4,458 prefetches 是纯浪费 DMA。

#### Eviction Policy × Prefetch 交互矩阵

```
                         Net DMA change (MB)
                    No prefetch | MRU insert | LRU insert
    LRU:                    0   |   +14,442  |   +33,562
    T1-protect:             0   |    +8,916  |   +15,092
    FIFO:                   0   |   +14,450  |    -8,686  ← 唯一有益!
```

**解读**:
1. **T1-protect + MRU insert 是最不坏的有害配置** (+8.9 GB)：T1 保护减少了对关键数据的驱逐
2. **FIFO + LRU insert 是唯一有益配置** (-8.7 GB)：FIFO 不在 access 时提升 chunk，LRU insert 让无用 prefetch 快速被淘汰
3. **LRU + LRU insert 是最差配置** (+33.6 GB)：demand access 会把 prefetched data 从 LRU 提升到 MRU，displacing useful data

### 17.6 对 §13-§15 结论的修正

| 原结论 | 修正 | 原因 |
|--------|------|------|
| "盲目 adjacent 命中率 26.4%" | **Decode 阶段仅 16.5%**（短 trace 过度乐观） | 长 trace 更多 decode 数据 |
| "History-3 可达 67.7%" | **实际 48.5%**（108K patterns 不可行） | 短 trace pattern 重复率高 |
| "每个 prefetch 驱逐一个 useful block" | **取决于 placement**: MRU insert 73.6% hit rate | 静态分析忽略了 prefetched data 在 VRAM 中的存活时间 |
| "Cross-block 仅在 oversub < 1.2x 有效" | 也取决于 **(eviction policy, insertion position)** | FIFO+LRU insert 在 1.84x 仍有益 |
| "HASH map 在热路径太慢" | **prefetch hook 不是热路径**，但 108K patterns 太大 | Xid 31 发生在 gpu_block_access，不在 gpu_page_prefetch |

### 17.7 改进方向（更新版）

基于深度分析，之前的三个改进方向需要重新评估：

#### ❌ History-3 prediction — 不可行
- 108K unique patterns，即使 4096-slot ARRAY 也只覆盖 46.7%，accuracy 23.0%
- 长 trace 显示 pattern 不稳定，不适合固定大小 hash table

#### ⚠️ Layer-aware prefetch — 理论可行但粒度太粗
- Layer+1 预测 47.1% accurate，most-common 预测 57.6%
- 但每层 ~439 chunks（878 MB），prefetch 一整层太激进
- 需要 **per-expert VA mapping**（更细粒度），当前 infrastructure 不支持

#### ⚠️ Selective (momentum) — 理论清晰但影响太小
- 97% 减少无效 prefetch，67.3% hit rate
- 但绝对量仅 2,154 prefetches（vs 41,814 baseline faults），影响 < 1%

#### ✅ 新方向: Prefetch insertion policy 优化
- 在 `gpu_block_activate` 中检测 chunk 是否来自 proactive prefetch
- 如果是：**放在 LRU 端**（让无用 prefetch 快速淘汰）
- 如果不是：正常放在 MRU 端
- 实现简单：BPF map 记录 pending prefetch 地址，activate 时查询
- 模拟结果: T1-protect + MRU-demand/LRU-prefetch 可能接近 FIFO+LRU 的效果

#### ✅ 新方向: FIFO eviction + cross-block
- 模拟显示 FIFO+LRU insert 是唯一 net beneficial 配置 (-8.7 GB)
- FIFO 的核心优势：不在 access 时提升 chunk，避免 prefetched data 被 promoted
- 可以实现为: `gpu_block_access` return BYPASS (不执行默认 move_tail)
- 但 FIFO 对非 cross-block 场景可能不如 LRU — 需要实际 benchmark 验证

### 17.8 分析工具

- `workloads/llama.cpp/simulate_vram.py` — VRAM 缓存模拟器
- `workloads/llama.cpp/analyze_crossblock_v3.py` — 增强版访问模式分析

---

## 18. 真实 Benchmark 结果 (2026-02-28)

### 18.1 关键 Bug 发现：kprobe 未被 attach

Cross-block v2 的 loader (`prefetch_cross_block_v2.c`) 只调用了 `bpf_map__attach_struct_ops()`
来 attach struct_ops map，但**从未 attach kprobe program** (`capture_va_block`)。
这意味着 `va_block_cache` 永远为空，cross-block prefetch **从未实际执行过**。

Debug counters 确认:
```
kprobe fires: 0          ← kprobe 没有 attach!
prefetch_hook fires: 67818
wq scheduled: 0
wq callback ran: 0
```

**修复**: 在 loader 中添加 `bpf_program__attach(skel->progs.capture_va_block)` 手动 attach kprobe。

### 18.2 修复后 Benchmark 结果

测试配置: 120B MoE, pp512/tg128, r=5, RTX 5090 32GB, 1.84x oversubscription

| 配置 | pp512 (t/s) | tg128 (t/s) | pp delta | tg delta |
|------|------------|------------|----------|----------|
| **Baseline** (always_max + cycle_moe, no XB) | **214.97±3.11** | **83.95±4.65** | - | - |
| XB + passive_mru (mode 0) | 204.84±2.09 | 59.99±2.75 | -4.7% | **-28.5%** |
| XB + cycle_moe (mode 1) | 205.54±2.69 | 60.49±2.37 | -4.4% | **-28.0%** |
| XB + default_lru (mode 2) | 203.83±2.77 | 60.53±2.57 | -5.2% | **-27.9%** |
| XB + FIFO (mode 3) | 204.69±2.12 | 60.23±2.53 | -4.8% | **-28.3%** |

### 18.3 Debug Counter 数据 (5 runs)

| 模式 | kprobe | prefetch_hook | wq scheduled | wq callback | callback rate |
|------|--------|---------------|-------------|-------------|---------------|
| passive_mru | 228,590 | 198,664 | 104,549 | 88,908 | 85.0% |
| cycle_moe | 228,641 | 198,715 | 104,581 | 89,125 | 85.2% |
| default_lru | 228,539 | 198,613 | 104,510 | 89,080 | 85.2% |
| FIFO | 228,331 | 198,405 | 104,335 | 89,117 | 85.4% |

**观察**:
- 所有模式的 stats 几乎完全相同（说明 eviction policy 不影响 fault 数量）
- 每次 run 约 ~17.8K 次 cross-block migrate，每次 2MB = **~35.6 GB 额外 PCIe 流量/run**
- 85% wq callback 执行率 — 15% 被丢弃（可能因为新的 wq_init 覆盖了未完成的旧请求）

### 18.4 核心结论

1. **Eviction policy 完全不影响 cross-block 性能** — 四种 eviction 模式给出相同的 ~60 tg。
   这证实了 §17.5 VRAM 模拟的预测：**在 2x oversubscription 下，eviction policy 影响 < 1%**。

2. **Cross-block 的 -28% 回归原因**:
   - ~35.6 GB 额外 PCIe 流量 / run → 每 token 额外 ~56 MB migration
   - 原始 per-token migration ~428 MB，增加 ~13%
   - 但 tg 下降 28%，比纯流量增加更严重
   - **额外损害来自 PCIe 带宽竞争**: async prefetch DMA 和 demand fault DMA 竞争同一 PCIe 链路
   - 以及 **lock contention**: `bpf_gpu_migrate_range()` 需要 `va_space` 锁，与 fault handler 竞争

3. **VRAM 模拟的预测 vs 真实结果**:
   - 模拟预测：FIFO+LRU insert 是唯一有益组合 (-8.7 GB)
   - 真实结果：FIFO 同样有害 (-28%)
   - **差异原因**: 模拟没有考虑 PCIe 竞争和锁争用，只考虑了 VRAM 内容变化
   - 模拟对于评估 "额外流量" 有用，但无法预测并发竞争的影响

4. **Blind adjacent cross-block prefetch 在 120B MoE 1.84x oversubscription 下确定无益**:
   - 不是 eviction policy 的问题
   - 不是 insertion position 的问题
   - 是 **PCIe 竞争 + 锁争用** 的根本问题
   - 即使 prefetch 命中率达到 73%+，额外的竞争开销超过了减少 fault 的收益

### 18.5 §18 结论修正

**§18.4.2 "lock contention" 结论完全错误**：
- `uvm_migrate_bpf()` 取 `uvm_va_space_down_read()` — READ lock
- Fault handler 也取 `uvm_va_space_down_read()` — READ lock
- Linux rwsem 允许多个 reader 同时持锁 — 读-读不冲突
- Cross-block 目标是 block N+1，fault 在 block N — per-block mutex 也不冲突
- **真正的竞争只有 PCIe 带宽共享和 GPU 内存分配（eviction）**

**§18.3 "~35.6 GB 额外 PCIe traffic" 是上界估算**：
- 假设所有 17.8K callback 都迁移完整 2MB
- 实际 `uvm_va_block_make_resident()` 跳过已 resident 的 page
- 需要内核级 trace 才能知道实际 DMA 量

---

## 19. Cross-Block 迭代优化实验 (2026-02-28)

### 19.1 实验配置

所有实验: 120B MoE, pp512/tg128, r=5, RTX 5090 32GB, ~1.84x oversubscription
Eviction: cycle_moe (除非另注)
BPF 改进: kprobe 正确 attach, migrate 返回值追踪, per-CPU wq_map(max_entries=64)

### 19.2 Granularity 实验

| 粒度 | pp512 | tg128 | tg delta | wq sched | migrate ok | rate skip |
|------|-------|-------|----------|----------|------------|-----------|
| Baseline (no XB) | 214.97 | 83.95 | — | — | — | — |
| **2MB** | 203.69 | 60.33 | **-28.2%** | 104,637 | 88,817 | 94,035 |
| 512KB | 207.80 | 56.40 | -32.8% | 145,675 | 121,532 | 77,053 |
| 256KB | 207.76 | 51.97 | -38.1% | 150,488 | 125,818 | 77,948 |
| 64KB | 205.76 | 47.44 | -43.5% | 154,595 | 129,878 | 78,373 |

**结论**: 更小粒度 = 更差性能。原因：
1. 小粒度 cross-block 只覆盖下一 block 的部分 → 剩余部分仍需 demand fault
2. 更多 demand fault → 更多 prefetch hook 触发 → 更多 cross-block prefetch 请求
3. Per-CPU rate-limiting 导致同一 block 从多个 CPU 重复 prefetch
4. 级联效应: 小粒度 paradoxically 增加总 DMA 量

### 19.3 Direction-Aware Prefetch

**设计**: 追踪 per-CPU 最近 2 个 block VA（prev, cur），只在连续 2+ 次同方向转换时 prefetch。
同时使用 global dedup (ARRAY map) 防止跨 CPU 重复 prefetch。

| Policy | pp512 | tg128 | tg delta | wq sched | migrate ok | dir skip | dedup |
|--------|-------|-------|----------|----------|------------|----------|-------|
| Baseline (no XB) | 214.97 | 83.95 | — | — | — | — | — |
| Blind adjacent (global dedup) | 213.52 | 61.82 | -26.4% | 104,603 | 88,917 | 0 | 25 |
| **Direction-aware** | **212.74** | **70.27** | **-16.3%** | **82,273** | **69,133** | **27,118** | 6 |

**Direction-aware 改进效果**:
- tg: 61.82 → 70.27 (**+13.7%** 相对改进)
- 过滤掉 27,118 次无效 prefetch (26% 的 hook 调用)
- 总 migration 减少 22%: 88,917 → 69,133
- 仍然 -16.3% vs baseline，说明即使方向正确的 prefetch 也有 displacement 代价

### 19.4 Direction-Aware × Eviction Policy

| Policy | pp512 | tg128 | tg delta | wq sched | migrate ok |
|--------|-------|-------|----------|----------|------------|
| Dir-aware + cycle_moe | 212.74 | 70.27 | -16.3% | 82,273 | 69,133 |
| Dir-aware + FIFO | 213.37 | 70.02 | -16.6% | 82,559 | 68,946 |

**结论**: Eviction policy 对 direction-aware 同样无影响（70.27 vs 70.02 在误差范围内）。
再次验证：**在 2x oversubscription 下，eviction policy 不是瓶颈**。

### 19.5 当前进展总结与分析

**已验证的优化方向**:
1. ✅ **Direction-aware prefetch**: 从 -28% 回归改善到 -16%，有效减少 ~22% 无效 prefetch
2. ❌ **Smaller granularity**: 反而更差（cascading effect — 更多 demand fault）
3. ❌ **Eviction policy 变换**: 四种 eviction 模式均无显著差异

**Key instrumentation 数据** (所有实验 migrate 成功率 >99.9%, 失败仅 ~50 次):
- migrate success/fail 分布说明 cross-block target 地址有效，DMA 确实在发生
- 99.9% 成功率 ≠ 99.9% 有用：成功只表示 `uvm_migrate()` 没报错，不代表目标 page 不在 VRAM
- 需要内核级 trace 才能区分 "实际 DMA" vs "已 resident 的 no-op migrate"

**§18 结论修正**:
- ❌ "lock contention" → 错误：fault 和 prefetch 都取 READ lock，不冲突
- ⚠️ "35.6 GB 额外流量" → 上界估算，实际可能更小
- ⚠️ "eviction policy 无影响" → 正确，但原因不是 lock contention，而是 displacement 代价在 VRAM 满载时是常数
- ⚠️ "cross-block 确定无益" → 过于武断：direction-aware 已将回归从 -28% 缩小到 -16%

### 19.6 Momentum Strictness 实验

| Filter | pp512 | tg128 | tg delta | wq sched | migrate ok | dir skip |
|--------|-------|-------|----------|----------|------------|----------|
| Baseline (no XB) | 214.97 | 83.95 | — | — | — | — |
| Blind adjacent | 213.52 | 61.82 | -26.4% | 104,603 | 88,917 | 0 |
| 2-step direction | 212.74 | 70.27 | -16.3% | 82,273 | 69,133 | 27,118 |
| 3-step direction | 216.17 | 77.90 | -7.2% | 71,765 | 59,769 | 38,039 |
| Dir-aware + 512KB | 213.25 | 64.73 | -22.9% | 108,473 | 88,451 | 33,868 |
| Dir-aware + FIFO | 213.37 | 70.02 | -16.6% | 82,559 | 68,946 | 27,251 |
| **Adjacent-stride** | **228.86** | **88.31** | **+5.2%** | **1,523** | **1,415** | **133,480** |

### 19.7 Adjacent-Stride: 突破性发现

**Adjacent-stride filter**: 要求连续 3 次 block 转换都是恰好 ±1 VA block（2MB 步长）。

**结果**: pp=228.86 (+6.5%), tg=88.31 (**+5.2% 超过 baseline!**)

**为什么有效**:
1. 极度 selective — 5 runs 仅 1,523 次 prefetch（vs blind 的 104K）
2. 几乎只在 **prefill 阶段**触发（顺序加载模型权重时产生连续 adjacent 访问）
3. Decode 阶段 MoE expert 切换是非 adjacent 的 → adjacent-stride 自动静默
4. 283 次/run × 2MB = ~566 MB 额外 prefetch — 忽略不计
5. 这些 prefetch 命中率极高（真正的顺序访问模式），有效减少 prefill fault 数量
6. 更少 demand fault → 更少 prefetch hook 调用 → 减少 BPF overhead
7. pp 从 215 → 229 (+6.5%) 说明 prefill 确实受益于 cross-block

**Adjacent-stride 本质上是一个自动 prefill-only cross-block**:
不需要显式检测 prefill/decode 阶段，adjacent-stride 的严格条件自然过滤掉 decode 阶段的所有 prefetch。

**与 always_max + cycle_moe 的组合效果**:
- always_max: 每次 fault 时 prefetch 当前 block 内所有 page (intra-block)
- adjacent-stride: 检测到连续顺序访问时 prefetch 下一个 block (cross-block, prefill only)
- cycle_moe: T1 数据保护 + non-T1 使用 kernel LRU

### 19.8 Adjacent-Stride 确认实验 (2026-02-28 续)

同一次 driver 加载下的完整对比（排除 driver reload 变量干扰）：

| Configuration | pp512 | tg128 | Notes |
|---------------|-------|-------|-------|
| No BPF (threshold=51) | 144.21 ± 0.91 | 49.50 ± 3.85 | 纯 kernel 默认 |
| always_max + cycle_moe (no XB) | 220.86 ± 1.70 | 87.32 ± 5.80 | intra-block only |
| adj-stride + cycle_moe | 227.77 ± 1.89 | 88.21 ± 5.68 | **最佳组合** |
| adj-stride + default_lru | 221.88 ± 1.45 | 85.74 ± 5.05 | kernel LRU |
| adj-stride + FIFO | 222.07 ± 1.93 | 85.86 ± 5.45 | 冻结位置 |

**Adjacent-stride + cycle_moe stats (5 runs)**:
```
kprobe fires:        284,263
prefetch_hook fires: 254,337
wq scheduled:          1,520  (0.6% of hook fires)
wq callback ran:       1,416
migrate success:       1,414  (99.9% success rate)
migrate failed:            2
rate-limit skip:     119,281
direction skip:      133,536  (99.4% filtered)
dedup skip:                0
```

**关键发现**:

1. **Adjacent-stride 确认可复现**: 两次独立测试 tg=88.31 和 88.21，高度一致
2. **pp 提升有意义**: cycle_moe + adj-stride pp=227.77 vs no-XB pp=220.86 = **+3.1%**
3. **tg 无害**: 88.21 vs 87.32 = +1.0%（在 σ=5.7 的噪声范围内）
4. **cycle_moe 仍是最佳 eviction**: adj-stride + cycle_moe (88.21) > default_lru (85.74) ≈ FIFO (85.86)
5. **Cross-block 数量极低时 eviction policy 仍然有差异**: cycle_moe 的 T1 保护在 decode 阶段提供额外收益，与 cross-block 无关

**Adjacent-stride 的工程价值**:
- 完全消除了 blind cross-block prefetch 的 -28% 回归
- 在 prefill 阶段提供了额外的 +3% pp 收益
- ~280 次/run 的 prefetch 活动对系统资源几乎无影响
- 不需要手动区分 prefill/decode — 利用访问模式自动识别

### 19.9 §18-§19 最终结论

1. **§18 "lock contention" 结论 → 完全否定**
   - 代码分析证明: fault handler 和 `uvm_migrate_bpf()` 都取 `va_space_down_read()` (READ lock)
   - Linux rwsem 允许多个 reader 同时持锁，不存在读-读竞争
   - 真正原因: **PCIe 带宽竞争** + **VRAM displacement** (每次 prefetch 触发 eviction)

2. **§18 "eviction policy 无影响" 结论 → 修正为 "仅在 high-volume cross-block 时无影响"**
   - 当 cross-block prefetch 数量很高 (~17K/run) 时，eviction policy 确实被 PCIe 竞争掩盖
   - 但当 cross-block 数量极低 (~280/run, adjacent-stride) 时，cycle_moe 仍比 default_lru/FIFO 好 2-3%

3. **Cross-block prefetch 关键成功因素: 选择性 > 覆盖率**

   | 策略 | 过滤率 | tg128 | 与 baseline 比 |
   |------|--------|-------|---------------|
   | Blind adjacent | 0% | 61.82 | -29% |
   | 2-step direction | 33% | 70.27 | -19% |
   | 3-step direction | 44% | 77.90 | -11% |
   | **Adjacent-stride** | **99.4%** | **88.21** | **+1.0%** |

   越严格的过滤 → 越少的无效 prefetch → 越少的 VRAM displacement → 越好的性能

4. **更小粒度 (512KB/256KB/64KB) 不是解决方案**
   - 部分 coverage 导致更多 demand fault → 级联增加 DMA 总量
   - 2MB (完整 VA block) 仍是最佳粒度

5. **更大粒度 (4MB/8MB multi-block) 也不是解决方案**
   - 4MB adjacent-stride: pp=222.53, tg=86.58 (vs 2MB: pp=227.77, tg=88.21)
   - 相同 prefetch 次数 (~1,480) 但每次 4MB → 2x DMA 总量 → 更多 displacement
   - migrate failed 从 2 增加到 11（某些地址超出 managed range）
   - **2MB (单个 VA block) 是最优粒度**：恰好覆盖下一个 block，不多不少

### 19.10 最佳配置总结

**生产推荐**: `./prefetch_cross_block_v2 1 2048 3`
- 1 = cycle_moe eviction (T1 保护 + non-T1 kernel LRU)
- 2048 = 2MB prefetch (单个 VA block)
- 3 = adjacent-stride filter (99.4% 过滤率)

**完整性能对比** (120B MoE, pp512/tg128, 1.84x oversubscription):
```
Configuration                      pp512     tg128     vs stock   runs
─────────────────────────────────  ───────   ───────   ────────   ────
Stock kernel (threshold=51)        144.21     49.50    baseline    5
always_max + cycle_moe             220.86     87.32    +76% tg     5
adj-stride + cycle_moe (2MB)       227.77     88.21    +78% tg     5
adj-stride + cycle_moe (4MB)       222.53     86.58    +75% tg     5
adj-stride + default_lru (2MB)     221.88     85.74    +73% tg     5
adj-stride + FIFO (2MB)            222.07     85.86    +73% tg     5
```

### 19.11 高置信度实验 (10 runs)

| Config | pp512 | tg128 | σ(tg) |
|--------|-------|-------|-------|
| always_max + cycle_moe (no XB) | 221.33 ± 3.49 | 88.79 ± 8.94 | 8.94 |
| adj-stride + cycle_moe (2MB) | 221.26 ± 2.59 | 87.58 ± 8.33 | 8.33 |

**Adjacent-stride stats (10 runs)**:
```
wq scheduled:   2,879  (~288/run, consistent with 5-run data)
migrate success: 2,667  (99.9% success)
migrate failed:      3
```

**统计分析**:
- tg 差异: 88.79 - 87.58 = 1.21 tok/s
- pooled σ ≈ 8.6, t-statistic = 1.21 / (8.6 × √(2/10)) ≈ 0.31
- **p >> 0.05, 差异不显著**

**修正后的结论**: Adjacent-stride 在 1.84x oversubscription 下**统计上等效于无 cross-block prefetch**。
- 不造成任何可测量的伤害（vs blind 的 -28% 回归）
- 也不提供任何可测量的收益
- ~288 次/run 的 prefetch 太少，不足以影响 prefill 速度
- 之前 5-run 数据显示的 "+3.1% pp" 在 σ≈3.5 的背景下不显著

**核心发现: 在 1.84x oversubscription 下，cross-block prefetch 要么有害（blind）要么无效（filtered）**。
性能的主要驱动力是 **intra-block always_max + cycle_moe eviction**，cross-block 无法在此基础上进一步提升。

### 19.12 为什么 cross-block 无法帮助

1. **VRAM 严重过压** (1.84x): 每个 prefetch 的 2MB 必须驱逐 2MB 有用数据
2. **Prefetch 的 net value = 命中收益 - 驱逐损失**: 在高过压下，驱逐损失 ≈ 命中收益 → net ≈ 0
3. **Intra-block already_max 已足够**: 当前 block 内的 always_max 已覆盖所有 page-level fault
4. **Adjacent block 的价值取决于后续访问**: MoE 模型的 decode 阶段是 expert-switching，下一个 block 几乎不是 adjacent
5. **Prefill 阶段的 sequential 访问确实触发 adjacent-stride，但**: prefill 只占总运行时间的小部分，且 prefetch 带宽与 demand fault 竞争

**Cross-block prefetch 可能有价值的场景**:
- **低过压** (< 1.3x): VRAM 有余裕吸收 prefetch 数据不必立即驱逐
- **强序列性工作负载**: 非 MoE 的 dense model，整个推理过程是 sequential weight loading
- **PCIe 带宽有余** (Gen5 x16): 当前 Gen5 x16 已经足够，但如果 compute 更快则 PCIe 再次成为瓶颈

---

## 16. 代码清理 (2026-02-27)

### 16.1 原则：不修改内核模块获取上下文信息

**核心发现**: va_block、va_space 等上下文信息不需要通过添加 kfunc 或修改 hook 签名获取。
BPF 程序可以通过 **kprobe + per-CPU map** 模式，在调用链上游函数中捕获这些信息。

**已有示例**: `extension/prefetch_trace.bpf.c`
- kprobe 挂在 `uvm_perf_prefetch_get_hint_va_block`（在 struct_ops hook 之前调用）
- 从函数参数拿到 `va_block`，用 CO-RE 读取 `start`/`end`/`va_space`
- 写入 `BPF_MAP_TYPE_PERCPU_ARRAY`（同 CPU 上 struct_ops hook 会读到）

```
调用链: uvm_perf_prefetch_get_hint_va_block(va_block, ...)
          ↓
        compute_prefetch_mask(va_block, ...)
          ↓
        compute_prefetch_region(va_block, ...)
          ↓
        uvm_bpf_call_gpu_page_prefetch(...)
          ↓
        ops->gpu_page_prefetch(...)  ← BPF struct_ops hook
```

**kprobe 在上游捕获 va_block → per-CPU map → struct_ops hook 读取**

CO-RE 读取链:
- `BPF_CORE_READ(va_block, start)` → block start VA
- `BPF_CORE_READ(va_block, end)` → block end VA
- `BPF_CORE_READ(va_block, managed_range, va_range.va_space)` → va_space 指针

### 16.2 清理结果

**移除的内核代码** (从 `uvm_bpf_struct_ops.c`):
- ~~`struct uvm_bpf_prefetch_ctx` + `DEFINE_PER_CPU`~~ (per-CPU context)
- ~~`bpf_gpu_get_block_start_va()`~~ → 用 kprobe + CO-RE 替代
- ~~`bpf_gpu_get_block_end_va()`~~ → 同上
- ~~`bpf_gpu_get_va_space()`~~ → 同上
- ~~`bpf_gpu_request_prefetch_range()`~~ (v1 kernel workqueue) → 用 bpf_wq 替代
- ~~kernel workqueue / ring buffer / spinlock 基础设施~~ (v1)
- ~~hook 签名修改 (va_block 参数)~~ → 不需要，kprobe 获取

**保留的内核代码** (最小化内核修改):
- `bpf_gpu_migrate_range()` kfunc — action kfunc，调用内部 `uvm_migrate()`，无法用 CO-RE 替代
- `uvm_migrate_bpf()` (uvm_migrate.c) — migrate_range 的后端

**删除的 BPF 程序**:
- `extension/prefetch_cross_block.bpf.c` (v1) — 依赖已删除的 kernel workqueue kfunc

**更新的 BPF 程序**:
- `extension/prefetch_cross_block_v2.bpf.c` — 改用 kprobe + per-CPU map 获取 va_block 信息

**删除的一次性脚本**:
- `microbench/memory/run_baseline.sh`, `run_compare.sh`, `run_compare_oversub.sh`, `run_llama_compare.sh`
- `microbench/memory/results/`

### 16.3 规则：未来获取上下文信息的方式

**不应该做**:
- 添加 getter kfunc 从 per-CPU context 返回内核指针
- 修改 struct_ops hook 签名传递额外参数
- 在内核模块中添加 per-CPU context set/clear 代码

**应该做**:
- kprobe 挂在调用链上游，从函数参数用 CO-RE 读取
- 写入 `BPF_MAP_TYPE_PERCPU_ARRAY`，struct_ops hook 中读取
- 参考 `extension/prefetch_trace.bpf.c` 的实现模式

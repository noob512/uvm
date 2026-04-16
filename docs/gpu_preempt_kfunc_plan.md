# GPU Preempt kfunc 设计文档：从内核 BPF 上下文直接抢占 GPU TSG

## 1. 背景与动机

### 1.1 当前 GPU 抢占架构

目前系统中有三种 GPU TSG 抢占路径：

**路径 A：Userspace ioctl 路径（`gpu_preempt_ctrl` 工具）**

```
BPF tracepoint (内核态)
    → 捕获 hClient/hTsg → 存入 BPF ring buffer
    → 用户态 poll ring buffer
    → 用户态发起 ioctl(NV_ESC_RM_CONTROL)
    → escape.c 安全检查绕过 (privLevel=KERNEL)
    → Nv04ControlWithSecInfo()
    → RM dispatch → kchangrpapiCtrlCmdPreempt_IMPL()
    → NV_RM_RPC_CONTROL → GSP firmware
    → 等待 RPC response
    → 返回用户态
```

实测延迟：**~356us**（含 userspace context switch + ioctl 往返 + GSP RPC）

**路径 B：bpf_wq 路径（用于 cross-block prefetch 等场景）**

```
struct_ops hook (non-sleepable, 持有 spinlock)
    → bpf_wq_init() + bpf_wq_set_callback() + bpf_wq_start()
    → kworker 线程被调度执行
    → callback 中调用 sleepable kfunc（如 bpf_gpu_migrate_range）
```

bpf_wq 当前仅用于 UVM 内存迁移，未用于 GPU 抢占。

**路径 C：struct_ops + kfunc 路径（on_task_init timeslice 设置）**

```
nv-kernel.o 内部 TSG 创建
    → 调用 nv_gpu_sched_task_init() [EXPORT_SYMBOL]
    → RCU dispatch → BPF struct_ops on_task_init()
    → BPF 调用 bpf_nv_gpu_set_timeslice(ctx, us) [kfunc]
    → 修改 ctx->timeslice
    → nv-kernel.o 读取修改后的 ctx，应用 timeslice
```

此路径延迟极低（纯内核态、无 RPC），但只能在 TSG **创建时** 设置参数，不能运行时调整。

### 1.2 问题分析

| 路径 | 优点 | 缺点 |
|------|------|------|
| A: ioctl | 功能完整（test_preempt_ioctl 已验证） | 延迟高(356us+)、需要 userspace 参与、无法自动化。注：`gpu_preempt_ctrl` 因依赖不存在的 tracepoints 从未加载过（见 §8.1） |
| B: bpf_wq | 纯内核态 | 需要 kworker 调度延迟、当前未支持 preempt |
| C: kfunc | 延迟最低 | 只能在 init 时设置、无运行时 preempt 能力 |

**核心问题**：BPF 程序无法从内核上下文直接触发 GPU TSG 抢占。所有抢占操作都必须经过 userspace ioctl 路径。

### 1.3 目标

添加 sleepable kfunc `bpf_nv_gpu_preempt_tsg(hClient, hTsg)`，使 BPF 程序可以从内核上下文（sleepable BPF context，如 bpf_wq callback）直接触发 GPU TSG 抢占，实现**跨进程 preempt**。

**为什么需要 kfunc？** 同进程内已可通过 fd 复用 + ioctl 实现 preempt（见 §8 验证），但 BPF 调度策略需要从内核态 preempt **任意进程**的 TSG，无法复用目标进程的 fd → 必须通过 kfunc 绕过 ioctl 安全检查。

**Handle 捕获**：已验证 kprobe 3-probe 策略可在 stock nvidia module 上捕获任意进程的 hClient/hTsg（见 §8.3），**无需修改内核**。kfunc 只需实现执行侧。

**新架构（目标）**：

```
BPF kprobe 捕获目标进程 hClient/hTsg → 存入 BPF map
    ↓
事件触发（page fault / timer / struct_ops hook）
    → BPF 程序检测到需要抢占
    → bpf_wq callback（如在 non-sleepable 上下文）
        或直接调用（如在 sleepable struct_ops 上下文）
    → bpf_nv_gpu_preempt_tsg(hClient, hTsg) [sleepable kfunc]
    → nv_gpu_sched_do_preempt() [nvidia.ko 内部]
    → RM dispatch → GSP RPC → 抢占完成
```

预期延迟：消除 userspace context switch（~50-100us），总延迟从 ~356us 降至 ~200-250us。**实测修正**（见 §8.10）：kfunc 低延迟带 177us（2x 提升），但总平均 540us 因 GSP 双峰分布反而高于 ioctl。

## 2. 现有代码分析

### 2.1 GPU 调度 struct_ops 框架

**文件**: `kernel-module/nvidia-module/kernel-open/nvidia/nv-gpu-sched-hooks.c`

当前已实现的 struct_ops 和 kfuncs：

```c
// struct_ops 定义
struct nv_gpu_sched_ops {
    int (*on_task_init)(struct nv_gpu_task_init_ctx *ctx);     // TSG 创建
    int (*on_bind)(struct nv_gpu_bind_ctx *ctx);               // TSG 绑定到 runlist
    int (*on_task_destroy)(struct nv_gpu_task_destroy_ctx *ctx); // TSG 销毁
};

// 已注册的 kfuncs
bpf_nv_gpu_set_timeslice(ctx, timeslice_us)   // KF_TRUSTED_ARGS, 仅 init 时
bpf_nv_gpu_set_interleave(ctx, level)          // KF_TRUSTED_ARGS, 仅 init 时
bpf_nv_gpu_reject_bind(ctx)                    // KF_TRUSTED_ARGS, 仅 bind 时
```

这些 kfuncs 都是 **non-sleepable** 的（只修改 context struct 字段），且只在特定 hook 时机可用。

### 2.2 UVM BPF struct_ops 框架（参考）

**文件**: `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_bpf_struct_ops.c`

UVM 侧已有 sleepable kfunc 的先例：

```c
// 已注册的 sleepable kfunc
bpf_gpu_migrate_range(va_space, addr, length)  // KF_SLEEPABLE
```

此 kfunc 通过 `bpf_wq` 从 non-sleepable 的 eviction/prefetch hooks 间接调用，是本文档实现前唯一的 sleepable GPU kfunc（本文档新增 `bpf_nv_gpu_preempt_tsg` 为第二个）。

### 2.3 RM 抢占实现路径

**文件**: `kernel-module/nvidia-module/src/nvidia/src/kernel/gpu/fifo/kernel_channel_group_api.c`

```c
// 抢占 API（通过 ioctl dispatch 调用）
NV_STATUS kchangrpapiCtrlCmdPreempt_IMPL(
    KernelChannelGroupApi *pKernelChannelGroupApi,
    NVA06C_CTRL_PREEMPT_PARAMS *pPreemptParams
);
```

此函数在 RM 对象系统内部运行，需要有效的 `KernelChannelGroupApi` 对象引用。在 GSP 系统（RTX 5090）上，通过 `NV_RM_RPC_CONTROL` 发送 RPC 到 GSP firmware，**会 sleep**。

### 2.4 ioctl 到 RM 的调用链

```
用户态 ioctl(fd, NV_ESC_RM_CONTROL, &params)
  → nvidia_ioctl() [kernel-open/nvidia/nv.c:2432]
  → rm_ioctl(sp, nv, nvfp, cmd, arg, size) [nv-kernel.o]
  → RM resource dispatch
  → kchangrpapiCtrlCmdPreempt_IMPL()
  → IS_GSP_CLIENT(pGpu) → NV_RM_RPC_CONTROL()
  → GSP firmware 执行抢占
  → 返回 NV_STATUS
```

关键：`rm_ioctl()` 是 `kernel-open` 调用 `nv-kernel.o` 的入口，声明在 `nv.h:959`：
```c
NV_STATUS NV_API_CALL rm_ioctl(nvidia_stack_t *, nv_state_t *, nv_file_private_t *, NvU32, void *, NvU32);
```

### 2.5 现有 GPU 抢占 BPF 工具

**文件**: `extension/gpu_preempt_ctrl.bpf.c`

使用 tracepoints 监控 TSG 生命周期：
- `tracepoint/nvidia/nvidia_gpu_tsg_create` → 捕获 hClient, hTsg
- `tracepoint/nvidia/nvidia_gpu_tsg_schedule` → 监控调度事件
- `tracepoint/nvidia/nvidia_gpu_tsg_destroy` → 清理跟踪

BPF maps 存储 TSG handle 映射（`tsg_map` by hTsg, `pid_tsg_map` by PID），供用户态读取后发送 ioctl preempt。

### 2.6 GPreempt 论文参考

**文件**: `docs/gpu-ext/driver_docs/sched/gpreempt-analysis/`

GPreempt (ATC'25) 的关键技术：
- TSG timeslice 设为 1us → 硬件快速 yield → <40us 抢占延迟
- 通过注释 `rpcRecvPoll()` 实现异步 RPC（fire-and-forget）
- 使用 GDRCopy (~1us) 实现 CPU→GPU hint-based pre-preemption

我们的方案与 GPreempt 的区别：
- GPreempt 修改驱动使 RPC 异步化（高风险，不等待 response）
- 我们保持 RPC 同步，仅消除 userspace 往返（低风险）
- 后续可选探索异步 RPC 作为 Phase 2

## 3. 技术约束

### 3.1 Sleepable vs Non-sleepable

**为什么 preempt kfunc 必须是 sleepable？**

RTX 5090 是 GSP 系统。GPU preempt 调用链：
```
bpf_nv_gpu_preempt_tsg()
  → nv_gpu_sched_do_preempt()
  → RM dispatch (可能获取 mutex/semaphore)
  → NV_RM_RPC_CONTROL() (等待 GSP response, completion_wait)
```

GSP RPC 使用 completion 机制等待 firmware 响应，**必须在可 sleep 的上下文中调用**。

因此：
- kfunc 标记为 `KF_SLEEPABLE`
- 从 non-sleepable hooks（eviction/prefetch，持有 spinlock）调用时，**必须** 通过 `bpf_wq`
- 从 sleepable hooks（如新增的 `on_sched_tick`）或 `bpf_wq` callback 中可直接调用

### 3.2 kernel-open 与 nv-kernel.o 的边界

```
kernel-open/nvidia/              ← 开源，可自由修改
    nv-gpu-sched-hooks.c         ← kfunc 定义在这里
    nv.c                         ← nvidia_ioctl 在这里

src/nvidia/                      ← RM 源码（开源但编译为 nv-kernel.o）
    kernel_channel_group_api.c   ← preempt 实现在这里

链接关系：
    nvidia.ko = kernel-open/*.o + nv-kernel.o
```

`kernel-open` 可以调用 `nv-kernel.o` 导出的函数（如 `rm_ioctl`）。反之，`nv-kernel.o` 调用 `kernel-open` 导出的函数（如 `nv_gpu_sched_task_init`）。

**新增 kfunc 的调用方向**：`kernel-open` (kfunc) → `nv-kernel.o` (RM preempt 实现)

有两种方式穿越这个边界：

1. **通过 rm_ioctl**：构造完整的 ioctl 参数，调用 `rm_ioctl()`。需要 `nvidia_stack_t`、`nv_state_t`、`nv_file_private_t`，获取这些上下文比较复杂。但 §8.1 的实践表明可以通过 `osapi.c` 中 `NV_ENTER_RM_RUNTIME` + `Nv04ControlWithSecInfo` 路径实现。
2. **新增 RM 导出函数**：在 RM 源码中添加 `nv_gpu_sched_do_preempt()` 并 `EXPORT_SYMBOL`。更干净但需要深入 RM 内部 API。

### 3.3 RM 上下文获取

kfunc 只需要 `hClient` 和 `hTsg` 两个参数（均为 NvU32），由 BPF kprobe 捕获（见 §8.3 的 3-probe 策略，零内核修改）。RM 内部通过 `(hClient, hTsg)` 查找 `KernelChannelGroupApi` 对象并执行 preempt。

## 4. 实现方案

### Phase 1: 添加 sleepable kfunc `bpf_nv_gpu_preempt_tsg`

**最小内核修改集**：仅 3 个文件。Handle 捕获由 BPF kprobe 完成（零内核修改，见 §8.3）。

#### 方案选择：osapi.c 新函数 vs escape.c bypass

两种方案本质相同，最终都调用 `Nv04ControlWithSecInfo` + `RS_PRIV_LEVEL_KERNEL` 实现跨进程 preempt。区别在于权限提升的位置和作用域：

| | osapi.c 方案（选定） | escape.c 方案（弃用） |
|---|---|---|
| 核心操作 | `Nv04ControlWithSecInfo(RS_PRIV_LEVEL_KERNEL)` | 相同 |
| 权限提升位置 | 新增专用内核函数内部 | 修改现有 ioctl 路径 |
| 谁能触发跨进程 preempt | **仅 kfunc**（BPF 程序） | kfunc + **任意 root 进程通过 ioctl** |
| 文件改动 | 3 处（osapi.c + exports + hooks.c） | 2 处（escape.c + hooks.c） |
| 安全性 | 权限提升不暴露给 userspace | ioctl 表面扩大，任何 root 进程可跨进程操控 GPU TSG |

**选择 osapi.c 方案**：虽然多 1 处文件改动，但权限提升被封装在内核函数内部，不对 userspace ioctl 路径引入安全漏洞。escape.c 方案会使任何 root 进程都能通过 ioctl 跨进程 preempt 其他进程的 GPU TSG，属于不必要的攻击面扩大。

#### 4.1 Step 1: RM 层新增内部 preempt 函数

**文件**: `kernel-module/nvidia-module/src/nvidia/arch/nvalloc/unix/src/osapi.c`

以内核特权级绕过 fd/client 安全检查，调用 `Nv04ControlWithSecInfo`：

```c
NvU32 NV_API_CALL nv_gpu_sched_do_preempt(
    nvidia_stack_t *sp, NvU32 hClient, NvU32 hTsg)
{
    void *fp;
    NVOS54_PARAMETERS params = { 0 };
    NVA06C_CTRL_PREEMPT_PARAMS preemptParams = { 0 };
    API_SECURITY_INFO secInfo = { 0 };

    NV_ENTER_RM_RUNTIME(sp, fp);

    preemptParams.bWait = NV_TRUE;
    preemptParams.bManualTimeout = NV_TRUE;
    preemptParams.timeoutUs = 100000;

    params.hClient = hClient;
    params.hObject = hTsg;
    params.cmd = 0xa06c0105; /* NVA06C_CTRL_CMD_PREEMPT */
    params.params = NvP64_VALUE(&preemptParams);
    params.paramsSize = sizeof(preemptParams);

    secInfo.privLevel = RS_PRIV_LEVEL_KERNEL;
    secInfo.paramLocation = PARAM_LOCATION_KERNEL;

    Nv04ControlWithSecInfo(&params, secInfo);

    NV_EXIT_RM_RUNTIME(sp, fp);
    return params.status;
}
```

**关键点**：
- `RS_PRIV_LEVEL_KERNEL` 绕过 fd/client ownership 检查 → 跨进程 preempt
- `PARAM_LOCATION_KERNEL` 告诉 RM 参数在内核空间（非 user pointer）
- sp (nvidia_stack_t) 由 kfunc 分配，函数只做 RM runtime 包装
- 不需要 threadState（与 escape.c 同级调用路径）
- 在 `exports_link_command.txt` 中添加导出（不需要修改 nv.h，用 extern 前向声明）

#### 4.2 Step 2: kfunc 注册

**文件**: `kernel-module/nvidia-module/kernel-open/nvidia/nv-gpu-sched-hooks.c`

在现有 kfunc 定义区域添加：

```c
/* Forward declaration - implemented in nv-kernel.o (osapi.c) */
extern NvU32 nv_gpu_sched_do_preempt(nvidia_stack_t *sp, NvU32 hClient, NvU32 hTsg);

__bpf_kfunc int bpf_nv_gpu_preempt_tsg(u32 hClient, u32 hTsg)
{
    nvidia_stack_t *sp = NULL;
    NvU32 status;

    if (!hClient || !hTsg)
        return -EINVAL;
    if (nv_kmem_cache_alloc_stack(&sp) != 0)
        return -ENOMEM;

    status = nv_gpu_sched_do_preempt(sp, hClient, hTsg);
    nv_kmem_cache_free_stack(sp);

    return (status == 0) ? 0 : -EIO;
}
```

在 BTF kfunc ID set 中注册（**添加到现有 set**）：

```c
BTF_KFUNCS_START(nv_gpu_sched_kfunc_ids_set)
BTF_ID_FLAGS(func, bpf_nv_gpu_set_timeslice, KF_TRUSTED_ARGS)
BTF_ID_FLAGS(func, bpf_nv_gpu_set_interleave, KF_TRUSTED_ARGS)
BTF_ID_FLAGS(func, bpf_nv_gpu_reject_bind, KF_TRUSTED_ARGS)
/* 新增: sleepable kfunc，可从 bpf_wq callback 调用 */
BTF_ID_FLAGS(func, bpf_nv_gpu_preempt_tsg, KF_SLEEPABLE)
BTF_KFUNCS_END(nv_gpu_sched_kfunc_ids_set)
```

#### 4.3 Step 3: BPF 测试程序

Handle 捕获使用已验证的 kprobe 3-probe 策略（见 §8.3），代码复用 `test_preempt_demo.bpf.c`。

kfunc 调用通过 `bpf_wq` 实现（kprobe 是 non-sleepable 上下文）：

```c
/* kfunc 声明 */
extern int bpf_nv_gpu_preempt_tsg(u32 hClient, u32 hTsg) __ksym;

/* bpf_wq callback: sleepable 上下文中调用 kfunc */
static int do_preempt_wq(void *map, int *key, void *val)
{
    struct preempt_req *req = val;
    return bpf_nv_gpu_preempt_tsg(req->hClient, req->hTsg);
}

/* 在 kprobe/kretprobe 中捕获到 TSG 后，触发 bpf_wq */
// bpf_wq_init(&req->work, &wq_map, 0);
// bpf_wq_set_callback(&req->work, do_preempt_wq, 0);
// bpf_wq_start(&req->work, 0);
```

### Phase 2（可选）: 添加 `on_sched_tick` sleepable hook

在 `nv_gpu_sched_ops` 中增加周期性 sleepable callback，由内核 timer 触发。BPF 程序在此 hook 中可直接调用 `bpf_nv_gpu_preempt_tsg()` 而不需要 bpf_wq。

**优势**：真正实现"无 workqueue"的 preempt 路径
**挑战**：需要在 nvidia.ko 内部设置 hrtimer/timer_list 周期性触发 hook

### Phase 3（探索性）: Fire-and-Forget Async RPC

参考 GPreempt 的做法，修改 `src/nvidia/src/kernel/vgpu/rpc.c` 使 preempt RPC 不等待 GSP response，使 kfunc 可标记为 non-sleepable。

**风险**：不知道 preempt 是否成功、可能引发 RM 状态不一致。

## 5. 关键文件清单

**内核修改（仅 3 处）**：

| 文件 | 操作 | 内容 |
|------|------|------|
| `src/nvidia/arch/nvalloc/unix/src/osapi.c` | 修改 | 添加 `nv_gpu_sched_do_preempt()` + EXPORT_SYMBOL |
| `kernel-open/nvidia/nv-gpu-sched-hooks.c` | 修改 | 添加 `bpf_nv_gpu_preempt_tsg` kfunc 定义和注册 |
| `src/nvidia/exports_link_command.txt` | 修改 | 导出 `nv_gpu_sched_do_preempt` 符号（无需修改 `nv.h`，用 extern 前向声明） |

Handle 捕获由 BPF kprobe 完成（零内核修改，见 §8.3）。`nv-gpu-sched-hooks.h`、`kernel_channel_group*.c`、`escape.c` 均不需要修改。

## 6. 风险与注意事项

### 6.1 RM API 调用安全性

`nv_gpu_sched_do_preempt()` 使用 `RS_PRIV_LEVEL_KERNEL` 绕过 client ownership 检查，实现跨进程 preempt。这与 ioctl 路径中 escape.c 的安全检查绕过逻辑相同，但在内核态完成，不暴露给用户空间。

### 6.2 GPU Lock 竞争

RM control 调用可能需要获取 GPU lock（`rmGpuLockAcquire`）。如果 preempt 在高频调用场景下使用，可能与正常的 RM 操作产生 lock 竞争。需要测试不同调用频率下的影响。

### 6.3 hClient/hTsg 有效性

BPF tracepoint 捕获的 handles 可能在调用 preempt 时已失效（TSG 已销毁）。`pRmApi->Control` 应该会返回 `NV_ERR_INVALID_OBJECT`，kfunc 需要正确传播此错误。

### 6.4 BPF verifier 兼容性

`KF_SLEEPABLE` kfuncs 在 `BPF_PROG_TYPE_STRUCT_OPS` 中使用时，需要：
- struct_ops 的 SEC 标注为 `struct_ops.s`（sleepable）
- 或从 `bpf_wq` callback 中调用（天然 sleepable）

需要验证 BPF verifier 是否允许在 `nv_gpu_sched_ops` 的 non-sleepable hooks 中通过 bpf_wq 间接调用 sleepable kfunc。

## 7. 验证方案

### 7.1 编译验证
```bash
cd kernel-module/nvidia-module
make -j$(nproc) modules
# 确认无编译错误
```

### 7.2 加载验证
```bash
sudo rmmod nvidia_uvm nvidia_modeset nvidia
sudo insmod kernel-open/nvidia.ko
sudo insmod kernel-open/nvidia-modeset.ko
sudo insmod kernel-open/nvidia-uvm.ko
dmesg | grep "GPU sched struct_ops"
# 应看到 "nvidia: GPU sched struct_ops initialized"
```

### 7.3 kfunc 可用性验证
```bash
cd extension
make gpu_sched_preempt
sudo ./gpu_sched_preempt
# BPF 程序应能通过 verifier 加载
```

### 7.4 功能测试
```bash
# 终端 1: 启动 preempt 工具
sudo ./gpu_sched_preempt -v

# 终端 2: 启动 long-running CUDA 程序
./test_cuda_kernel  # 运行一个持续几秒的 CUDA kernel

# 终端 1: 触发 preempt (通过命令行或自动)
# 观察 preempt 结果和延迟

# 对比：使用原有 gpu_preempt_ctrl 工具的 ioctl 路径延迟
```

### 7.5 延迟对比测试
```bash
# 测量 kfunc 路径延迟（bpf_wq callback → kfunc → RM → 返回）
# 测量 ioctl 路径延迟（userspace → ioctl → RM → 返回 userspace）
# 预期 kfunc 路径快 50-100us（消除 userspace context switch）
```

## 8. 实施进展记录

### 8.1 已完成工作 (2026-03-03)

#### 发现1: `gpu_preempt_ctrl` 无法工作
- `gpu_preempt_ctrl.bpf.c` 依赖的 tracepoints（`nvidia_gpu_tsg_create/schedule/destroy`）**不存在于驱动源码**
- `nv-tracepoint.h` 中仅定义了 `nvidia_dev_xid` 一个 tracepoint
- 因此 `gpu_preempt_ctrl` **从未能正常加载过**

#### 发现2: `NV_ESC_RM_CONTROL` 值错误
- `gpu_preempt_ctrl_event.h` 中定义 `NV_ESC_RM_CONTROL = 0x2A`（正确）
- 但如果其他代码中误用 `NV_ESC_RM_CONTROL = 2`（之前的 test_preempt_ioctl 就犯了这个错误），ioctl 会返回 EINVAL
- 正确值来自 `src/nvidia/arch/nvalloc/unix/include/nv_escape.h:30`

#### 发现3: ioctl preempt 直接测试成功

##### 测试步骤

**Step 1: 使用 bpftrace 从 kprobe 捕获 RM handles**

bpftrace 无法解析 nvidia 模块自定义 struct（BTF 类型带后缀如 `u32___2`），因此使用 raw memory offset 读取：

```bash
sudo bpftrace -e 'kprobe:nv_gpu_sched_task_init {
    printf("hClient=0x%x hTsg=0x%x tsg_id=%llu engine=%u\n",
        *(uint32*)(arg0+32), *(uint32*)(arg0+36),
        *(uint64*)arg0, *(uint32*)(arg0+8));
}'
```

`nv_gpu_task_init_ctx` struct 内存布局（加入 hClient/hTsg 后）：
```
offset 0:  u64 tsg_id
offset 8:  u32 engine_type
offset 16: u64 default_timeslice
offset 24: u32 default_interleave
offset 28: u32 runlist_id
offset 32: u32 hClient        ← 新增
offset 36: u32 hTsg            ← 新增
offset 40: u64 timeslice
offset 48: u32 interleave_level
```

**Step 2: 启动 CUDA 工作负载，触发 TSG 创建**

```bash
python3 -c "
import torch, time, os
print(f'PID: {os.getpid()}')
x = torch.randn(4096, 4096, device='cuda')
print('CUDA initialized, running matmuls...')
while True:
    y = torch.mm(x, x)
    torch.cuda.synchronize()
    time.sleep(0.1)
" &
```

bpftrace 输出（每个 CUDA 进程创建多个 TSG）：
```
hClient=0xc1e00050 hTsg=0xd          engine=13    # COPY engine
hClient=0xc1e00051 hTsg=0xf          engine=13    # COPY engine
hClient=0xc1e00052 hTsg=0xcaf00002   engine=1     # GRAPHICS/COMPUTE
hClient=0xc1d00053 hTsg=0xcaf00001   engine=1     # GRAPHICS/COMPUTE
hClient=0xc1d00002 hTsg=0xcaf0017b   engine=13    # COPY engine (UVM)
hClient=0xc1d00002 hTsg=0xcaf00193   engine=14    # COPY engine (UVM)
hClient=0xc1d0004b hTsg=0x5c000013   engine=1     # GRAPHICS/COMPUTE (应用)
hClient=0xc1d0004b hTsg=0x5c000038   engine=13    # COPY engine (应用)
hClient=0xc1d0004b hTsg=0x5c000046   engine=14    # COPY engine (应用)
```

**Engine types**: 1=GR (Graphics/Compute), 13/14=CE (Copy Engine)。注：§8.9 kfunc 测试中观察到 engine 13=NVJPEG, 14=OFA — engine type 到物理 engine 的映射取决于 GPU 型号和 driver 配置

**Step 3: 运行 `test_preempt_ioctl` 测试各种控制命令**

```bash
# 编译测试工具
cd extension && make test_preempt_ioctl

# PREEMPT: 强制抢占 GPU TSG
sudo ./test_preempt_ioctl preempt 0xc1d0004b 0x5c000013

# SET_TIMESLICE: 动态修改 TSG 时间片
sudo ./test_preempt_ioctl timeslice 0xc1d0004b 0x5c000013 1000  # 1000us
sudo ./test_preempt_ioctl timeslice 0xc1d0004b 0x5c000013 1     # 1us (GPreempt风格)

# PREEMPT on COPY engine TSG
sudo ./test_preempt_ioctl preempt 0xc1d0004b 0x5c000038

# SET_INTERLEAVE
sudo ./test_preempt_ioctl interleave 0xc1d0004b 0x5c000013 1
```

##### 实测结果 (2026-03-03)

```
=== Test 1: PREEMPT on engine=1 TSG ===
PREEMPT hClient=0xc1d0004b hTsg=0x5c000013 status=0 duration=669 us

=== Test 2: PREEMPT again (same TSG) ===
PREEMPT hClient=0xc1d0004b hTsg=0x5c000013 status=0 duration=316 us

=== Test 3: SET_TIMESLICE to 1000us ===
SET_TIMESLICE hClient=0xc1d0004b hTsg=0x5c000013 timeslice=1000 us status=0 duration=313 us

=== Test 4: SET_TIMESLICE to 1us (GPreempt-style) ===
SET_TIMESLICE hClient=0xc1d0004b hTsg=0x5c000013 timeslice=1 us status=0 duration=267 us

=== Test 5: PREEMPT on engine=13 (COPY engine) TSG ===
PREEMPT hClient=0xc1d0004b hTsg=0x5c000038 status=0 duration=316 us

=== Test 6: SET_INTERLEAVE ===
SET_INTERLEAVE hClient=0xc1d0004b hTsg=0x5c000013 level=1 status=31 duration=5 us
```

CUDA 进程在所有 preempt 操作后继续正常运行（`nvidia-smi` 确认进程仍存在）。

##### 结果汇总

| 操作 | 结果 | 延迟 | 说明 |
|------|------|------|------|
| PREEMPT (首次) | **成功** status=0 | 669us | 首次调用包含冷路径开销 |
| PREEMPT (重复) | **成功** status=0 | 316us | 热路径稳定 ~300us |
| SET_TIMESLICE 1000us | **成功** status=0 | 313us | 可设为任意值 |
| SET_TIMESLICE 1us | **成功** status=0 | 267us | GPreempt 风格最小时间片 |
| PREEMPT (COPY engine) | **成功** status=0 | 316us | CE TSG 也可以 preempt |
| SET_INTERLEAVE | **失败** status=31 | 5us | NV_ERR_INVALID_ARGUMENT |

**关键发现**：
- Preempt 延迟稳定在 ~300us（不含首次冷路径 ~670us），与之前测试一致
- 延迟主要来自 GSP RPC 往返（RTX 5090 是 GSP 架构）
- SET_INTERLEAVE 失败是因为 interleave level 值不在有效范围内（不影响核心功能）
- CUDA 进程 preempt 后自动恢复执行，无 crash 或异常

##### ioctl 调用路径分析

```
Userspace test_preempt_ioctl
  │
  ├── open("/dev/nvidiactl")
  │
  ├── ioctl(fd, _IOWR('F', 211, nv_ioctl_xfer_t))
  │     cmd = NV_ESC_IOCTL_XFER_CMD (NV_IOCTL_BASE+11 = 211)
  │     payload = { cmd=0x2A (NV_ESC_RM_CONTROL), size, ptr→NVOS54_PARAMETERS }
  │
  │   NVOS54_PARAMETERS:
  │     hClient = 0xc1d0004b    ← 从 bpftrace 捕获
  │     hObject = 0x5c000013    ← TSG handle
  │     cmd     = 0xa06c0105    ← NVA06C_CTRL_CMD_PREEMPT
  │     params  → { bWait=1, bManualTimeout=1, timeoutUs=100000 }
  │
  └── 内核路径:
        nvidia_ioctl()                          [kernel-open/nvidia/nv.c]
          → rm_ioctl(sp, nv, nvfp, cmd, arg)    [nv-kernel.o / osapi.c]
            → case NV_ESC_RM_CONTROL:           [escape.c]
              → 安全绕过: (cmd & 0xffff0000) == 0xa06c0000
                → privLevel = RS_PRIV_LEVEL_KERNEL
                → clientOSInfo = NULL
              → Nv04ControlWithSecInfo(pApi, secInfo)
                → RM resource server dispatch
                  → rpcCtrlPreempt_HAL()        [rpc.c:4472]
                    → GSP RPC (等待 firmware response, ~300us)
                      → GPU 硬件执行 TSG preempt
```

**安全绕过机制** (`escape.c:762`):
正常 RM control 调用会检查调用者是否拥有目标 RM 对象（通过 fd 匹配 clientOSInfo）。
自定义修改将 `0xa06c****` 范围命令（TSG 调度控制）提升为内核特权级别，允许跨进程 preempt。

**GSP RPC 路径** (`rpc.c:4472`):
RTX 5090 的 RM 不直接操作 GPU 硬件，而是通过 RPC 发送到 GSP firmware。
`rpcCtrlPreempt_HAL()` 构造 RPC 消息 → 等待 completion → GSP firmware 执行实际抢占。
这是 ~300us 延迟的主要来源，kfunc 路径也无法消除此 RPC 延迟。

#### 已实现的代码修改（kernel module 侧，初版 — 含多余修改）

**注意**: 初版修改包含了不必要的变更。经过 §8.3-§8.4 的验证，最终只需要以下 **3 处修改**：

**需要保留的修改**（kfunc 核心）：

1. **`src/.../osapi.c`**
   - 新增 `nv_gpu_sched_do_preempt()`
   - 走 `NV_ENTER_RM_RUNTIME` + `Nv04ControlWithSecInfo` + `RS_PRIV_LEVEL_KERNEL` 路径（不使用 threadState，见 §8.7 最终实现）

2. **`kernel-open/nvidia/nv-gpu-sched-hooks.c`**
   - 新增 `bpf_nv_gpu_preempt_tsg()` (KF_SLEEPABLE kfunc)

3. **`src/nvidia/exports_link_command.txt`**
   - 导出 `nv_gpu_sched_do_preempt` 符号（无需修改 `nv.h`，用 extern 前向声明）

**应撤回的多余修改**（handle 捕获由 kprobe 完成，不需要内核改动）：

- ~~`nv-gpu-sched-hooks.h`: `nv_gpu_task_init_ctx` 新增 hClient/hTsg 字段~~ → kprobe 3-probe 策略已解决
- ~~`kernel_channel_group.c`: 移除 hook 调用~~ → 不需要
- ~~`kernel_channel_group_api.c`: 添加 hook 调用~~ → 不需要
- ~~`escape.c`: 安全检查绕过~~ → kfunc 用 `RS_PRIV_LEVEL_KERNEL`，不走 ioctl
- ~~`osapi.c`: `nv_gpu_sched_do_set_timeslice()`~~ → 不需要 set_timeslice kfunc
- ~~`nv-gpu-sched-hooks.c`: `bpf_nv_gpu_set_timeslice_runtime()`~~ → 不需要

#### 验证状态（初版，含多余修改）
- [x] 模块编译成功
- [x] 模块加载成功，dmesg 确认 struct_ops + kfunc 初始化
- [x] kallsyms 确认 `bpf_nv_gpu_preempt_tsg` 已注册
- [x] 现有 struct_ops 程序 (`gpu_sched_set_timeslices`) 在新模块上正常工作
- [ ] **撤回多余修改后重新编译验证**
- [ ] **kfunc 端到端测试**（BPF 程序 → bpf_wq → kfunc → RM → GSP → preempt）
- [ ] **回归测试**：现有 UVM BPF 程序（prefetch/eviction）在新模块上正常工作

### 8.2 `test_preempt_demo` — 自包含 BPF+CUDA+ioctl 测试工具 (2026-03-03)

**设计**：单进程单二进制，集成 BPF kprobe（捕获 TSG handles）+ CUDA driver API（运行 GPU kernel）+ ioctl preempt（抢占测试），一条命令展示完整 preempt 效果。

**文件**：
- `extension/test_preempt_demo.bpf.c` — BPF 3-probe 策略（kprobe + kretprobe on ioctl 捕获 hClient/hTsg，见 §8.3）
- `extension/test_preempt_demo.c` — BPF loader + CUDA driver API + ioctl + 三项测试
- `extension/Makefile` — 新增 `CUDA_APPS` 链接规则（`-lcuda -lpthread`）

**实测结果** (2026-03-03)：

| 测试 | 结果 |
|------|------|
| **A: Preempt 延迟** | avg=337us, min=319, max=400 (10/10 成功) |
| **B: Timeslice=1us** | 单 TSG 无影响 (-0.0%)，需要多 TSG 竞争 |
| **C: Burst 100x** | 5749 preempts/sec, 100/100 成功, GPU alive |
| **D: 持续 preempt 影响** | **kernel 时间 +184%** (260ms → 739ms) |

**Test D 关键数据** — 证明 preempt 真正中断 GPU 执行：
```
Phase 1 (无 preempt):  10 kernels, avg=260151 min=259008 max=270364 us
Phase 2 (持续 preempt): 10 kernels, avg=739293 min=711454 max=745119 us
Preempts issued: 43211 in 7.4s (5839 preempts/sec)

*** kernel time +184.2% (260151 → 739293 us) ***
→ PREEMPT IS INTERRUPTING GPU EXECUTION
```

**Test D 分析**：
- 每次 preempt 触发 GPU 硬件保存 TSG 上下文 → 重新调度 → 恢复执行
- 单 TSG 场景下，preempt 后 GPU scheduler 立即恢复同一 TSG，但 save/restore 有固定开销
- 43211 次 preempt 使 10 个 kernel 额外花费 4.79s → 每次 preempt 约 111us GPU 侧开销
- 这 111us 是 GPU 硬件 context save/restore 的真实代价（host 侧看到的 ~350us 还包含 GSP RPC 往返）

**技术细节**：
- **系统 TSG vs 应用 TSG**：CUDA init 创建 ~6 个系统 TSG（`0xcaf*` handles），应用 TSG（`0x5c*` handles）最后创建。必须选最后一个 GR TSG。
- **BPF kprobe**：用 raw memory offset 读取（非 struct_ops），避免 vmlinux.h BTF 问题
- **PTX anti-DCE**：busy_loop 必须写 global memory，否则 GPU JIT 做 dead-code elimination
- **依赖**：~~需要自定义 nvidia 模块（hClient/hTsg 字段 + escape.c 安全绕过）~~ 见 §8.3 无需内核修改的方案

### 8.3 无需内核修改的 Handle 捕获方案 (2026-03-03)

**问题**：原始方案在 `nv_gpu_task_init_ctx` 中添加 hClient/hTsg 字段，需要修改内核模块。

**发现**：`nv-kernel.o` 中所有函数（`rm_ioctl`、`Nv04AllocWithSecInfo`、`kchangrpapiConstruct_IMPL` 等）都标记为 "notrace"，**无法 kprobe**。只有 `kernel-open/` 目录中编译的函数（`nv_gpu_sched_task_init`、`nvidia_unlocked_ioctl` 等）可以 kprobe。

**解决方案**：3-probe 策略，从 ioctl 路径拦截 TSG 分配：

```
Probe 1: kprobe/nvidia_unlocked_ioctl (入口)
  ├── 读取 user-space nv_ioctl_xfer_t → 过滤 inner cmd == 0x2B (NV_ESC_RM_ALLOC)
  ├── 读取 user-space NVOS21_PARAMETERS → 过滤 hClass == 0xa06c (TSG class)
  └── 保存 hRoot (=hClient) + NVOS21 user pointer 到 per-TID pending map

Probe 2: kprobe/nv_gpu_sched_task_init (中间)
  ├── 在 rm_ioctl 执行期间被调用（kchangrpInit_IMPL → nv_gpu_sched_task_init）
  └── 补充 engine_type + tsg_id 到 pending map entry

Probe 3: kretprobe/nvidia_unlocked_ioctl (出口)
  ├── nvidia_ioctl 已执行 copy_to_user 将结果写回 user space
  ├── 从 user-space NVOS21 offset 8 读取 hObjectNew (= hTsg)
  └── 输出完整 {hClient, hTsg, engine_type, tsg_id, pid} entry
```

**NVOS21_PARAMETERS 内存布局**（同 NVOS64 前 4 字段）：
```
offset 0:  u32 hRoot         (= hClient, 输入)
offset 4:  u32 hObjectParent
offset 8:  u32 hObjectNew    (= hTsg, 输出，由 RM alloc 填写)
offset 12: u32 hClass        (= 0xa06c for TSG)
```

**nv_ioctl_xfer_t 内存布局**（user space）：
```
offset 0:  u32 cmd   (inner command, e.g., 0x2B)
offset 4:  u32 size
offset 8:  u64 ptr   (user-space pointer to NVOS21)
```

**实测结果** (2026-03-03，stock nvidia module)：

```
Captured 3 TSG(s):
  [0] hClient=0xc1d0033e hTsg=0x5c000013 engine=GR(1) tsg_id=1
  [1] hClient=0xc1d0033e hTsg=0x5c000038 engine=CE(13) tsg_id=3
  [2] hClient=0xc1d0033e hTsg=0x5c000046 engine=CE2(14) tsg_id=1

Test A: avg=358us, 10/10 成功
Test C: 5612 preempts/sec, 100/100 成功
Test D: kernel time +180.2% (260120 → 728914 us)
  → PREEMPT IS INTERRUPTING GPU EXECUTION
```

**与旧方案对比**：

| 维度 | 旧方案 (custom module) | 新方案 (stock module) |
|------|----------------------|---------------------|
| Handle 捕获 | kprobe on `nv_gpu_sched_task_init` + custom fields (offset 32/36) | kprobe/kretprobe on `nvidia_unlocked_ioctl` + user-space read |
| 内核修改 | 需要在 `nv_gpu_task_init_ctx` 添加 hClient/hTsg | **不需要** |
| 捕获时机 | TSG 初始化时（struct_ops hook 调用点） | ioctl 返回时（NVOS21 copy_to_user 后） |
| Engine type | 直接从 custom ctx 字段读取 | 从 `nv_gpu_sched_task_init` kprobe 间接获取 |
| 捕获 TSG 数量 | 9 个（含系统 TSG） | 3 个（仅应用 TSG，系统 TSG 不走 class=0xa06c） |
| escape.c bypass | 需要（cross-fd preempt ioctl） | ~~仍需要~~ **不需要**（见 §8.4） |
| 性能 | 相同 | 相同（preempt 延迟相同） |

### 8.4 无需 escape.c bypass 的 Preempt 执行 (2026-03-03)

**问题**：§8.3 解决了 handle 捕获的内核修改依赖，但 preempt ioctl 仍然因为 cross-fd 访问被 RM 安全检查拦截（status=35）。

**根本原因**：`test_preempt_demo` 自己 `open("/dev/nvidiactl")` 得到新 fd，其 `nvfp`（file private）与 CUDA 创建 hClient 时的 fd 不匹配。RM `Nv04ControlWithSecInfo` 校验 `client->clientOSInfo == secInfo.clientOSInfo` 失败。

**解决方案**：既然 CUDA 和 test_preempt_demo 在**同一进程**，CUDA 打开的 nvidia fd 就在进程的 fd 表里。扫描 `/proc/self/fd`，逐个尝试 preempt，成功的就是 CUDA 的控制 fd（nvfp 匹配 hClient）。

```c
// 扫描 /proc/self/fd → 找到 /dev/nvidia* fds → 尝试 preempt → 成功即匹配
for (int fd = 3; fd < 1024; fd++) {
    readlink("/proc/self/fd/%d", link);
    if (strncmp(link, "/dev/nvidia", 11) != 0) continue;
    status = rm_control(fd, hClient, hTsg, NVA06C_CTRL_CMD_PREEMPT, ...);
    if (status == 0) return fd;  // Found CUDA's fd!
}
```

**实测结果** (2026-03-03，stock nvidia module，零内核修改)：

```
Scanning /proc/self/fd for CUDA's nvidia fd...
  fd=22 (/dev/nvidiactl): preempt status=0
Found CUDA's fd=22 — nvfp matches hClient

Test A: avg=358us, 10/10 成功
Test C: 5779 preempts/sec, 100/100 成功
Test D: kernel time +164.6% (260091 → 688310 us)
  → PREEMPT IS INTERRUPTING GPU EXECUTION
```

**最终架构**（零内核修改）：

```
test_preempt_demo (单进程, 单二进制, sudo)
  │
  ├── BPF Probe 1: kprobe/nvidia_unlocked_ioctl
  │     读取 user-space nv_ioctl_xfer_t + NVOS21
  │     过滤 cmd=0x2B, hClass=0xa06c → 保存 hClient
  │
  ├── BPF Probe 2: kprobe/nv_gpu_sched_task_init
  │     补充 engine_type, tsg_id
  │
  ├── BPF Probe 3: kretprobe/nvidia_unlocked_ioctl
  │     读取 hObjectNew (= hTsg) → 输出完整 entry
  │
  ├── CUDA Driver API: cuInit + cuCtxCreate + cuLaunchKernel
  │     → CUDA 内部 open("/dev/nvidiactl") → fd=22
  │
  └── Preempt ioctl: 复用 CUDA 的 fd=22
        → nvfp 匹配 hClient → RM 安全检查通过 → preempt 成功
```

**关键依赖总结**：

| 依赖 | 是否需要内核修改 |
|------|----------------|
| Handle 捕获 (hClient/hTsg) | 否（nvidia_unlocked_ioctl kprobe） |
| Engine type 捕获 | 否（nv_gpu_sched_task_init kprobe） |
| Preempt ioctl 执行 | 否（复用 CUDA 的 fd） |
| Timeslice 设置 | 否（同上） |
| **总计** | **零内核修改** |

**限制**：此方案要求 preempt 调用者和 CUDA 在**同一进程**内（共享 fd 表）。**跨进程 preempt 是 kfunc 的核心价值** — BPF 调度策略需要 preempt 任意进程的 TSG，必须通过 kfunc 路径（见 §4）。

### 8.5 代码重构：`gpu_preempt.h` 公共头文件 (2026-03-03)

将 preempt 机制代码提取为可复用头文件 `extension/gpu_preempt.h`，原始 `test_preempt_demo` 保留 Tests A-D，新增 `test_preempt_multi` 包含 Tests E-F。

**`gpu_preempt.h` 公共接口**（`gp_` 前缀）：

| 函数/宏 | 说明 |
|---------|------|
| `gp_rm_control(fd, hClient, hObject, cmd, params, size)` | RM control ioctl 封装 |
| `gp_preempt(fd, hClient, hTsg)` | TSG preempt (bWait=1, timeout=100ms) |
| `gp_set_timeslice(fd, hClient, hTsg, us)` | 动态设置 TSG timeslice |
| `gp_find_cuda_fd(hClient, hTsg, verbose)` | 扫描 /proc/self/fd 找 CUDA 的 nvidia fd |
| `gp_get_time_us()` | CLOCK_MONOTONIC 微秒时间戳 |
| `gp_engine_str(engine_type)` | Engine type → 字符串 (GR/CE/CE2) |
| `GP_CHECK_CUDA(call)` | CUDA 错误检查宏 |
| `gp_ptx_source` | busy_loop PTX kernel 源码 |
| `struct gp_worker` + `gp_worker_init/start/join` | GPU worker 线程基础设施（含 sample 采集） |
| `gp_cuda_warmup(ctx)` | CUDA context warmup（触发 TSG 创建） |

**文件清单**：
- `extension/gpu_preempt.h` — 公共头文件（所有 preempt 机制代码）
- `extension/test_preempt_demo.bpf.c` — BPF 3-probe 策略（不变）
- `extension/test_preempt_demo.c` — 重构使用 gpu_preempt.h，保留 Tests A-D
- `extension/test_preempt_multi.bpf.c` — 同 test_preempt_demo.bpf.c（独立 skeleton）
- `extension/test_preempt_multi.c` — 多 context 测试 Tests E-F

### 8.6 多 Context 竞争测试 `test_preempt_multi` (2026-03-03)

**目的**：证明 TSG preempt 在多 context 竞争场景下能有效提升优先任务的 GPU 时间分配。

**架构**：两个 CUDA context（A 和 B）在同一 GPU 上竞争，BPF 捕获双方 TSG handles，preempt B 的 TSG 观察 A 的性能变化。

**实测结果** (RTX 5090, stock nvidia module, 零内核修改)：

```
=== GPU Preempt Multi-Context Test ===

[Phase 2] Creating CUDA context A...
  Context A GR TSG: hClient=0xc1d0002b hTsg=0x5c000013

[Phase 3] Creating CUDA context B...
  Context B GR TSG: hClient=0xc1d0002b hTsg=0x5c000088
```

#### Test E: 两个等量 context 竞争 — preempt B → A 获得更多 GPU 时间

两个 context 运行相同工作负载（100M iterations, ~300ms/kernel）：

```
Phase 1 (无 preempt):
  A: avg=539455 min=539411 max=539501 stddev=33 us
  B: avg=539383 us
  → 两者均分 GPU 时间（~50%），kernel 因等待调度从 260ms 升至 539ms

Phase 2 (持续 preempt B):
  A: avg=278392 min=276919 max=280077 stddev=1013 us
  Preempts issued: 23411

  *** RESULT E: A kernel time -48.4% (539455 → 278392 us) ***
  → PREEMPTING B GIVES A MORE GPU TIME
```

**分析**：
- 无 preempt 时，GPU round-robin 两个 TSG，每个 context 获得 ~50% GPU 时间
  - kernel 本身 ~260ms，但需等待另一个 TSG 的 timeslice（~260ms），总计 ~540ms
- 持续 preempt B 后，B 的 kernel 被频繁中断，A 获得接近 100% GPU 时间
  - A 的 kernel 时间从 539ms 降回 278ms（接近无竞争的 260ms baseline + preempt 扰动开销）
- **48.4% 延迟降低 = 从 50% GPU 份额提升到接近 100%**

#### Test F: 短 kernel A + 长 kernel B — preempt 降低 A 的调度等待

Context A 运行短 kernel（1M iterations, ~3ms），Context B 运行长 kernel（100M iterations, ~300ms）：

```
Phase 1 (无 preempt):
  A: avg=7091 min=7089 max=7093 stddev=1 us
  → A 的 kernel 仅 ~3ms，但 observed time 是 7ms（含 ~4ms 调度等待）

Phase 2 (持续 preempt B):
  A: avg=3005 min=2811 max=4992 stddev=456 us
  Preempts issued: 469

  *** RESULT F: A kernel time -57.6% (7091 → 3005 us) ***
  → PREEMPT REDUCES A's SCHEDULING WAIT
```

**分析**：
- 无 preempt 时，A 的短 kernel 需等待 B 的 timeslice 到期才能被调度
  - observed time (7ms) = kernel time (~3ms) + scheduling wait (~4ms)
  - scheduling wait < 16ms timeslice 是因为 A 可能在 B timeslice 中间到达
- 持续 preempt B 后，B 被立即中断，A 几乎不用等待调度
  - observed time 降至 ~3ms ≈ 纯 kernel 执行时间
  - **调度等待从 ~4ms 降至接近 0**
- Preempt 仅发出 469 次（vs Test E 的 23411 次），因为 A 的 kernel 很短，每次 preempt B 后 A 很快完成

#### 结果汇总

| 测试 | 场景 | 无 preempt | 有 preempt | 改善 |
|------|------|-----------|-----------|------|
| **E** | 两等量 context | A=539ms | A=278ms | **-48.4%** |
| **F** | 短 A + 长 B | A=7.1ms | A=3.0ms | **-57.6%** |
| D (参考) | 单 context 连续 preempt | 260ms | 739ms | +184.2% (preempt 开销) |

**结论**：
1. **TSG preempt = 有效的 kernel 级 preempt**：preempt TSG 在 kernel 运行时调用 = 打断该 kernel 执行
2. **多 context 场景下 preempt 有明确收益**：可实现优先级调度（优先任务获得更多 GPU 时间）和延迟保护（latency-sensitive 任务不被 throughput 任务阻塞）
3. **零内核修改**：所有功能在 stock nvidia module 上实现

### 8.7 kfunc 实现（最小修改版）(2026-03-04)

从干净的 `test-sched` 分支重新实现，仅 3 处修改（初版的多余修改已全部不需要）：

**修改 1: `src/nvidia/arch/nvalloc/unix/src/osapi.c`**
- 新增 `nv_gpu_sched_do_preempt(sp, hClient, hTsg)`
- 走 `NV_ENTER_RM_RUNTIME` → `Nv04ControlWithSecInfo(RS_PRIV_LEVEL_KERNEL)` 路径
- 不使用 threadState（跟 escape.c 同级别调用，不需要）
- `PARAM_LOCATION_KERNEL` 表明参数在内核空间
- 新增 `#include <ctrl/ctrla06c.h>` 获取 preempt 参数类型

**修改 2: `kernel-open/nvidia/nv-gpu-sched-hooks.c`**
- 新增 `bpf_nv_gpu_preempt_tsg(hClient, hTsg)` kfunc
- 在 `__bpf_kfunc_start_defs()` / `__bpf_kfunc_end_defs()` 之间
- BTF 注册：`BTF_ID_FLAGS(func, bpf_nv_gpu_preempt_tsg, KF_SLEEPABLE)`
- kfunc 负责 `nv_kmem_cache_alloc_stack` / `free_stack`，osapi 函数接收 sp 参数
- `extern NvU32 nv_gpu_sched_do_preempt(nvidia_stack_t *sp, NvU32, NvU32)` 前向声明

**修改 3: `src/nvidia/exports_link_command.txt`**
- 新增 `--undefined=nv_gpu_sched_do_preempt`
- 无需修改 nv.h（使用 extern 前向声明即可）

**验证结果**：
- [x] `make -j$(nproc) modules` 编译成功（仅重编译 osapi.c + nv-gpu-sched-hooks.o）
- [x] 三个模块加载成功（nvidia + nvidia-modeset + nvidia-uvm）
- [x] `dmesg`: "nvidia: GPU sched struct_ops initialized"
- [x] `kallsyms`: `bpf_nv_gpu_preempt_tsg` 和 `nv_gpu_sched_do_preempt` 均可见
- [x] **kfunc 端到端测试** ✅ 详见 §8.9

### 8.8 TODO

1. ~~多 TSG 竞争测试~~ ✅ Test E + Test F 完成（同进程 ioctl 验证）
2. ~~撤回多余内核修改~~ ✅ 从干净分支重新实现，仅 3 处修改
3. ~~kfunc 编译 + 加载~~ ✅ 模块编译成功，kfunc 注册到 kallsyms
4. ~~kfunc 端到端测试~~ ✅ BPF → bpf_wq → kfunc → RM → GSP → preempt 全链路验证
5. ~~跨进程 preempt 验证~~ ✅ CUDA-A 的 TSG 被 CUDA-B 触发的 struct_ops → bpf_wq → kfunc 成功 preempt
6. **集成到 BPF 调度策略** — 在 UVM eviction/prefetch hooks 中通过 bpf_wq 调用 kfunc preempt
7. **动态 preempt 策略** — 根据 fault rate / QoS 信号自动决定 preempt 目标和频率

### 8.9 kfunc 端到端测试验证 (2026-03-04)

**测试程序**: `extension/test_preempt_kfunc.bpf.c` + `test_preempt_kfunc.c`

**架构**（解决 kprobe 不能使用 bpf_wq 的限制）：
- **kprobe 层**（non-sleepable）：3-probe 策略捕获 hClient/hTsg handles
  - kprobe/nvidia_unlocked_ioctl → 拦截 TSG alloc (class 0xa06c)
  - kprobe/nv_gpu_sched_task_init → 捕获 tsg_id + engine_type
  - kretprobe/nvidia_unlocked_ioctl → 从 user-space NVOS21 读取 hTsg
- **struct_ops 层**（支持 bpf_wq）：on_task_init/on_bind 检查 trigger map → bpf_wq → kfunc
- **共享 maps**：tsg_map（handles）、trigger（触发信号）、stats（统计）

**关键发现**：
- `bpf_wq_init()` 必须在 `bpf_wq_set_callback_impl()` 之前调用，否则返回 -EINVAL
- kprobe 程序不能使用 bpf_wq（"tracing progs cannot use bpf_wq yet"）
- struct_ops 程序可以使用 bpf_wq，且与 kprobe 在同一 BPF 对象中共存
- `BPF_PROG` 宏内部使用 `ctx` 作为第一个参数名，struct_ops 参数不能命名为 `ctx`

**测试流程**：
1. 加载 BPF（kprobes + struct_ops 同时注册）
2. 启动 CUDA-A → kprobes 捕获 3 个 TSG（engine 1=GRAPHICS, 13=NVJPEG, 14=OFA）
3. 用户 arm trigger（trigger[0] = tsg_idx + 1）
4. 启动 CUDA-B → struct_ops on_task_init 触发 → check trigger → bpf_wq → kfunc
5. kworker 执行 `bpf_nv_gpu_preempt_tsg(hClient, hTsg)` → RM dispatch → GSP → 成功

**结果**：
```
tsg_captured: 6   (3 from CUDA-A, 3 from CUDA-B)
preempt_ok:   1   ← kfunc 成功 preempt CUDA-A 的 GRAPHICS TSG
preempt_err:  0
wq_fired:     1
struct_ops:   14
```

**trace_pipe 关键日志**：
```
python3-131060 [010] check_trigger: trig=1 tsg_idx=0
python3-131060 [010] trigger: wq_set_callback ret=0
python3-131060 [010] trigger: wq_start ret=0 hClient=0xc1d0003b hTsg=0x5c000013
kworker/10:0   [010] kfunc preempt OK: hClient=0xc1d0003b hTsg=0x5c000013
```

**跨进程 preempt 确认**：
- CUDA-A PID=130830 创建的 TSG（hClient=0xc1d0003b, hTsg=0x5c000013）
- 被 CUDA-B PID=131060 触发的 struct_ops → bpf_wq → kfunc 成功 preempt
- 使用 RS_PRIV_LEVEL_KERNEL 绕过 RM ownership 检查，实现真正的跨进程 preempt

### 8.10 kfunc vs ioctl 延迟对比基准测试 (2026-03-04)

**测试方法**：
- 创建目标 CUDA 上下文（target TSG），然后反复 preempt 同一 TSG
- kfunc 路径：bpf_wq callback 中 `bpf_ktime_get_ns()` 精确计时 kfunc 调用
- ioctl 路径：`test_preempt_demo` 使用 `clock_gettime(CLOCK_MONOTONIC)` 计时
- 每次 preempt 之间有 0.3-0.5s 间隔，避免 bpf_wq 竞态

**kfunc 路径延迟 (31 samples, bpf_wq → kfunc → RM → GSP)**：
```
排除冷启动 (首次 50ms):
  N=30, avg=540 us, median=206 us, min=147 us, max=1380 us

双峰分布:
  低延迟带 (<400us): N=17, avg=177 us    ← GSP 已空闲时
  高延迟带 (≥400us): N=13, avg=1014 us   ← GSP 忙碌/上下文切换
```

**ioctl 路径延迟 (test_preempt_demo)**：
```
标准模式 (10 rounds): avg=354 us, min=313 us, max=398 us
突发模式 (100x rapid): avg=181 us/preempt, throughput=5519 preempts/sec
```

**对比总结**：

| 路径 | Avg | Min | Max | 说明 |
|------|-----|-----|-----|------|
| ioctl (标准) | 354 us | 313 us | 398 us | userspace→ioctl→RM→GSP |
| ioctl (突发) | 181 us | — | — | 100x 连续 rapid preempt |
| kfunc (全部,warm) | 540 us | 147 us | 1380 us | bpf_wq→kfunc→RM→GSP |
| kfunc (低延迟带) | 177 us | 147 us | 207 us | GSP 空闲时，纯 RM+GSP |
| kfunc (高延迟带) | 1014 us | 772 us | 1380 us | GSP 忙碌/上下文切换 |

**关键发现**：

1. **kfunc 低延迟带 (177us) 比 ioctl 标准 (354us) 快 2x** — 消除了 userspace→kernel 往返
2. **kfunc 有明显双峰分布** — GSP 状态决定了延迟是 ~177us 还是 ~1014us
3. **ioctl 突发模式 (181us) 接近 kfunc 低延迟带** — 连续 ioctl 时 GSP 保持热状态
4. **bpf_wq 调度开销极小** — wq_ns ≈ kfunc_ns（差距 <20ns），几乎全是 RM+GSP RPC 时间
5. **首次调用有冷启动惩罚** — 第一次 kfunc 调用 50ms，之后稳定在 147-1380us

**为什么 kfunc avg (540us) > ioctl avg (354us)?**
- ioctl 测试有 **持续运行的 GPU kernel**（baseline iteration ~260ms），GPU 一直在工作
- kfunc 测试的 target 是**空闲 CUDA context**（无 running kernel）
- 当 target TSG 无 running kernel 时，preempt 仍然走完 RM→GSP 路径但有更多变数
- 双峰分布说明 GSP firmware 内部状态（是否有活跃调度）影响 preempt 延迟
- **公平对比应看 kfunc 低延迟带 (177us) vs ioctl 标准 (354us) = 2x 提升**

**kfunc 的真正价值**：
- 不是绝对延迟更低（RM→GSP RPC 是固定开销）
- 而是**消除 userspace 往返 + ring buffer poll 延迟**
- 在 BPF 策略中，检测到需要 preempt → bpf_wq → kfunc，全程内核态
- 对比 ioctl 路径：检测 → ring buffer → userspace poll → ioctl → kernel → RM → GSP
- 关键差异在**检测到执行的延迟**，不是 RM+GSP 本身

**测试工具**：
- `scripts/extension/preempt/bench_preempt_kfunc.py` — Python 自动化 kfunc 基准测试
- `scripts/extension/preempt/bench_preempt_kfunc.sh` — Shell 版本（备用）
- `test_preempt_demo` — ioctl 路径基准测试（含 burst 模式）

### 8.11 Sleepable uprobe 直接调用 kfunc 验证 (2026-03-04)

**发现**：sleepable uprobe (`SEC("uprobe.s/...")`) 可以**直接调用** sleepable kfunc，无需 bpf_wq。

**原理**：
- uprobe 运行在进程上下文（process context），标记 `BPF_F_SLEEPABLE` 后可 sleep
- kfunc 注册了 `BPF_PROG_TYPE_UNSPEC`（覆盖所有 prog type），uprobe 可以调用
- 无需 struct_ops + bpf_wq 中间层

**测试**：`extension/test_uprobe_preempt.bpf.c` + `test_uprobe_preempt.c`
- uprobe hook `cuLaunchKernel` → 直接 `bpf_nv_gpu_preempt_tsg()` → 成功
- 每次 `cuLaunchKernel` 调用精确触发 preempt

**结果 (5 samples)**：
```
uprobe preempt OK: 276735 ns (277 us)
uprobe preempt OK: 318187 ns (318 us)
uprobe preempt OK: 320077 ns (320 us)
uprobe preempt OK: 316094 ns (316 us)
uprobe preempt OK: 328407 ns (328 us)
→ avg=312 us, min=277 us, max=328 us
```

**三路径对比**：

| 路径 | Avg | 触发时机 | bpf_wq | 架构复杂度 |
|------|-----|---------|--------|-----------|
| ioctl (userspace) | 354 us | 手动/polling | 不需要 | 高（ring buffer + userspace） |
| struct_ops + bpf_wq + kfunc | 540 us | context 创建 | **需要** | 中（struct_ops + wq） |
| **uprobe.s + kfunc** | **312 us** | **kernel launch** | **不需要** | **低（直接调用）** |

**uprobe 路径优势**：
1. **延迟最低** (312 us) — 无 bpf_wq kworker 调度开销
2. **触发时机最精准** — hook `cuLaunchKernel`，在 GPU kernel 提交时 preempt
3. **架构最简** — 不需要 struct_ops、bpf_wq、trigger map、重复计数
4. **可 hook 任意 CUDA API** — `cuLaunchKernel`、`cuMemcpyAsync`、`cudaMalloc` 等

**实际调度场景**：
- uprobe 在 BE 进程的 `cuLaunchKernel` 上 → 检测 LC fault rate → 决定是否 preempt BE
- 或 uprobe 在 LC 进程的 `cuLaunchKernel` 上 → 在 LC 提交 kernel 前先 preempt BE
- 这实现了**精确到 kernel launch 粒度的 GPU preemption**

### 8.12 优先级调度实验 (2026-03-04)

#### 8.12.1 早期实验：per-launch preempt 反而变慢

**实验 A: uprobe on LC's cuLaunchKernel → preempt BE (BE <<<512,256>>> loop)**

| 配置 | LC avg | LC median | LC P99 |
|------|--------|-----------|--------|
| 无 preempt (baseline) | 323 us | 137 us | 2235 us |
| **有 preempt** | **2427 us** | **2569 us** | **2502 us** |

结果：LC **变慢 7.5x**。原因：
- preempt 是**同步**的（280us kfunc call 阻塞 cuLaunchKernel 返回）
- baseline 时 RTX 5090 硬件 compute preemption 已经很好地处理 SM 时间片
- BE 用 512 blocks 只占 GPU 38% 容量 → LC 可以并行跑

**实验 B: uprobe on BE's cuLaunchKernel → preempt BE (BE 占满 GPU)**

| 配置 | LC avg | LC median | LC P99 |
|------|--------|-----------|--------|
| 无 preempt | 3069 us | 2340 us | 6680 us |
| **有 preempt** | **4108 us** | **4290 us** | **8770 us** |

结果：LC 仍然**变慢**。原因：
- 每次 BE launch 都 preempt → BE 变慢 → GPU 占用时间更长 → LC 等更久
- TSG preemption 是重量级操作（暂停/恢复整个 TSG context）

**验证 preempt 确实生效**（test_preempt_demo 数据）：
```
无 preempt: BE kernel = 261678 us
持续 preempt: BE kernel = 716415 us (+173.8%)
preempt throughput: 5504 preempts/sec
```
→ TSG preempt 确实中断了 GPU 执行，但在上述场景中是一种惩罚而非优化。

#### 8.12.2 修正实验：persistent kernel 占满 GPU

**实验设计**：3 场景对比，分析 BPF preempt 在什么条件下有效。

**为什么之前 preempt 反而变慢？**

之前的实验中 BE 用 `<<<512, 256>>>` 只占 GPU 38% 容量（512 blocks / 1360 max），LC 可以在空闲 SM 上**并行**运行。RTX 5090 的硬件 compute preemption 在 SM 级别做 warp 交错（微秒级），远快于 BPF TSG preempt 的 300us 同步开销。结果：preempt 的 300us 开销 > 并行运行的 0us → LC 变慢。

**正确的测试**：BE 必须**完全占满所有 SM**（persistent kernel 填满每个 SM slot），LC 才被迫等待硬件 timeslice。

**三场景对比**：

| 场景 | BE 模式 | LC Avg | LC Median | LC Max | LC P95 |
|------|---------|--------|-----------|--------|--------|
| S1: BE 部分占 GPU | `<<<512,256>>>` loop | 5184 us | 4960 us | 9542 us | 9542 us |
| S2: BE **占满 GPU** | persistent, ALL SMs | 6141 us | 5802 us | **8302 us** | **8302 us** |
| S3: S2 + **BPF preempt** | persistent + uprobe | **4389 us** | **4859 us** | **4899 us** | **4899 us** |

**S2→S3 改善**：
- **avg: -29%** (6141 → 4389 us)
- **max: -41%** (8302 → 4899 us，消除尾延迟)
- **P95: -41%** (8302 → 4899 us)
- **方差: -66%** (spread 5240 → 1769 us，延迟更可预测)

**关键分析**：

1. **S1 的 LC 延迟高于预期** — `<<<512,256>>>` loop 模式下，BE 在 kernel 间有 gap（sync + relaunch），但 GPU 争抢仍然导致 5ms avg 延迟。这说明即使 GPU 有空闲 SM，TSG 级别的调度器也会造成延迟

2. **S2 的 max=8302us** — 这接近硬件 timeslice 周期。当 BE persistent kernel 占满所有 SM，LC 必须等到 GPU 硬件 round-robin timeslice 轮转才能拿到 SM 时间。S1 的更高 max=9542us 来自 kernel relaunch gap 叠加 timeslice 等待

3. **S3 消除了尾延迟** — BPF preempt 强制中断 BE 的 TSG，使 LC 不再受硬件 timeslice 随机性影响：
   - max 从 8302us 降至 4899us（-41%）
   - 方差大幅缩小（延迟可预测）
   - 代价：preempt 同步开销 ~300us + TSG context switch ~4.5ms

4. **S3 的双峰分布** — ~4860us（preempt + GPU context switch + kernel） vs ~3300us（BE 恰好在 yield 窗口）。4860us = 300us preempt + ~4500us GPU context switch 开销

**BPF preempt 什么时候有效**：

| 条件 | BPF preempt 效果 | 原因 |
|------|-----------------|------|
| BE 部分占 GPU | **无效/有害** | LC 可并行跑，preempt 300us 是纯开销 |
| BE 占满 GPU + 短 timeslice | **中等** | 减少尾延迟，但 avg 改善有限 |
| BE 占满 GPU + 长 timeslice | **显著** | LC 等待从 ms 级降至 preempt 开销 |
| **UVM eviction** | **必要** | 唯一能释放 BE VRAM 页的方法 |
| **QoS enforcement** | **有效** | 一次性惩罚超时 BE |

**什么样的 kernel 场景最能体现 preempt 价值**：
- **Persistent kernel**（占满所有 SM、长时间运行，如 CUDA Graph 循环）
- **超长 kernel**（>10ms，如大矩阵乘法、推理 forward pass）
- **Cooperative kernel**（grid sync 要求所有 SM 同时运行，不可被部分 preempt）
- **UVM 场景**（BE 持有 VRAM 页，LC 需要 page migration）

**测试程序**：`scripts/extension/preempt/test_priority_demo.cu`
- `be` mode：persistent kernel 填满所有 SM（`cudaOccupancyMaxActiveBlocksPerMultiprocessor` 计算 max blocks）
- `be_loop` mode：loop 模式（kernel 间有 gap）
- `lc` mode：短 kernel 延迟测量

**脚本**：`scripts/extension/preempt/run_priority_demo.sh` — 自动化 3 场景对比

## 9. 参考文档

- [gpu_preempt_ctrl_design.md](./gpu-ext/driver_docs/sched/gpu_preempt_ctrl_design.md) — 现有 userspace preempt 工具设计
- [ebpf_preempt_design.md](./gpu-ext/driver_docs/sched/ebpf_preempt_design.md) — eBPF preempt 可行性分析
- [GPreempt_Implementation_Analysis.md](./gpu-ext/driver_docs/sched/gpreempt-analysis/GPreempt_Implementation_Analysis.md) — GPreempt 论文实现分析
- [hook_enhancement_analysis.md](./gpu-ext/driver_docs/sched/hook_enhancement_analysis.md) — GPU 调度 hook 增强分析
- `kernel-open/nvidia/nv-gpu-sched-hooks.c` — 当前 GPU sched struct_ops 实现
- `kernel-open/nvidia-uvm/uvm_bpf_struct_ops.c` — UVM BPF struct_ops（sleepable kfunc 参考）
- `extension/gpu_preempt.h` — preempt 机制公共头文件（handle 捕获 + ioctl 封装）
- `extension/test_preempt_demo.bpf.c` — BPF 3-probe handle 捕获策略
- `extension/test_preempt_multi.c` — 多 context 竞争测试（Test E/F）
- `extension/test_preempt_kfunc.bpf.c` — kfunc 端到端测试 BPF（kprobe + struct_ops + bpf_wq）
- `extension/test_preempt_kfunc.c` — kfunc 测试 loader（交互式 + 脚本化）
- `extension/test_uprobe_preempt.bpf.c` — sleepable uprobe 直接调用 kfunc 验证
- `extension/test_uprobe_preempt.c` — uprobe preempt loader（支持 `-p PID` 过滤）
- `scripts/extension/preempt/test_priority_demo.cu` — LC/BE 优先级调度 demo（CUDA 程序）
- `scripts/extension/preempt/run_priority_demo.sh` — 优先级调度自动化测试脚本
- `scripts/extension/preempt/run_preempt_kfunc_test.sh` — 非交互式端到端测试脚本
- `scripts/extension/preempt/bench_preempt_kfunc.py` — kfunc 延迟基准测试（自动化）
- `scripts/extension/preempt/bench_preempt_kfunc.sh` — kfunc 延迟基准测试（shell 版）
- `extension/prefetch_cross_block_v2.bpf.c` — bpf_wq 使用模式参考

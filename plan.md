# Plan: 自包含 BPF+CUDA+ioctl GPU Preempt Demo 工具

## 目标

写一个**单进程单二进制**的 `test_preempt_demo` 工具，同时包含：
1. **BPF kprobe** — 自动捕获 CUDA 上下文创建时的 RM handles (hClient/hTsg)
2. **CUDA driver API** — 启动长时间运行的 GPU kernel
3. **ioctl preempt** — 用捕获到的 handles 发起 preempt，测量延迟和效果

运行后不依赖任何外部工具（bpftrace、python 等），一条命令即可看到 preempt 的完整效果。

## 架构

```
test_preempt_demo (单进程)
│
├── Phase 1: 加载 BPF kprobe
│   └── attach kprobe:nv_gpu_sched_task_init
│       → 捕获 hClient/hTsg/engine_type 存入 BPF hash map
│
├── Phase 2: 初始化 CUDA
│   └── cuInit() → cuCtxCreate()
│       → 驱动创建 TSGs → 触发 kprobe → BPF 自动记录 handles
│
├── Phase 3: 读取 BPF map，找到 engine=1 (COMPUTE) 的 TSG handle
│
├── Phase 4: 启动 GPU kernel (worker thread)
│   └── 循环执行 busy_loop PTX kernel，每次记录完成时间
│
├── Phase 5: Preempt 测试
│   ├── Test A: Preempt 延迟（10 次 preempt，统计 min/avg/max）
│   ├── Test B: Timeslice 修改（设为 1us，测量 kernel 完成时间变化）
│   └── Test C: 多次快速 preempt（100 次连续，测量吞吐）
│
└── Phase 6: 输出结果汇总表
```

## 文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `extension/test_preempt_demo.bpf.c` | 新建 | BPF kprobe 程序，捕获 TSG handles |
| `extension/test_preempt_demo.c` | 新建 | 主程序：BPF loader + CUDA + ioctl + 测试 |
| `extension/Makefile` | 修改 | 添加 CUDA_APPS 链接规则 |

## 详细设计

### 1. BPF 程序 (`test_preempt_demo.bpf.c`)

使用 **kprobe**（不是 struct_ops），避免 vmlinux.h 不含 nvidia module BTF 的问题。
用 `bpf_probe_read_kernel` + raw memory offset 读取 `nv_gpu_task_init_ctx` 字段。

```c
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

struct tsg_entry {
    u32 hClient;
    u32 hTsg;
    u32 engine_type;
    u32 pid;
    u64 tsg_id;
};

// hash map: 顺序 index → tsg_entry
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 64);
    __type(key, u32);           // sequential index
    __type(value, struct tsg_entry);
} tsg_map SEC(".maps");

// 计数器
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, u32);
} tsg_count SEC(".maps");

// 目标 PID（从 userspace 设置，只捕获自己进程的 TSG）
const volatile u32 target_pid = 0;

SEC("kprobe/nv_gpu_sched_task_init")
int capture_tsg(struct pt_regs *ctx)
{
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (target_pid && pid != target_pid)
        return 0;

    void *init_ctx = (void *)PT_REGS_PARM1(ctx);
    struct tsg_entry entry = {};
    u32 zero = 0, *cnt;

    // struct nv_gpu_task_init_ctx 内存布局:
    //   offset 0:  u64 tsg_id
    //   offset 8:  u32 engine_type
    //   offset 32: u32 hClient
    //   offset 36: u32 hTsg
    bpf_probe_read_kernel(&entry.tsg_id, sizeof(u64), init_ctx);
    bpf_probe_read_kernel(&entry.engine_type, sizeof(u32), init_ctx + 8);
    bpf_probe_read_kernel(&entry.hClient, sizeof(u32), init_ctx + 32);
    bpf_probe_read_kernel(&entry.hTsg, sizeof(u32), init_ctx + 36);
    entry.pid = pid;

    cnt = bpf_map_lookup_elem(&tsg_count, &zero);
    if (!cnt) return 0;

    u32 idx = __sync_fetch_and_add(cnt, 1);
    bpf_map_update_elem(&tsg_map, &idx, &entry, BPF_ANY);
    return 0;
}

char LICENSE[] SEC("license") = "GPL";
```

**为什么用 kprobe 不用 struct_ops**：
- struct_ops 需要 nvidia module BTF 类型（`nv_gpu_task_init_ctx` 等），当前 vmlinux.h 中不存在
- kprobe + raw memory offset 已在 bpftrace 测试中验证可行
- kprobe 更轻量，不需要注册/注销 struct_ops，不影响现有已加载的 BPF 策略

### 2. 主程序 (`test_preempt_demo.c`)

**链接依赖**：libbpf（BPF skeleton）+ libcuda（CUDA driver API）+ pthread

#### 2.1 BPF 加载

```c
// 设置 target_pid 为自身 PID
skel->rodata->target_pid = getpid();
// open → load → attach kprobe
skel = test_preempt_demo_bpf__open();
skel->rodata->target_pid = getpid();
test_preempt_demo_bpf__load(skel);
test_preempt_demo_bpf__attach(skel);
```

#### 2.2 CUDA 初始化

使用 CUDA **driver API**（`<cuda.h>`, `libcuda.so`），不用 runtime API：
- 不需要 nvcc 编译，普通 gcc 即可
- `libcuda.so` 随 nvidia 驱动安装，总是可用
- 嵌入 PTX 字符串，CUDA driver 会 JIT 编译为目标 GPU 的 native code

```c
cuInit(0);
cuDeviceGet(&device, 0);
cuCtxCreate(&ctx, 0, device);
// → 此时 kprobe 已触发，handles 已在 BPF map 中
```

#### 2.3 读取 BPF map

遍历 `tsg_map`，找出 `engine_type == 1` (GR/COMPUTE) 且 `pid == getpid()` 的 entry。

#### 2.4 GPU Worker Thread

在独立线程中循环运行 busy_loop PTX kernel，每次记录完成耗时。

PTX kernel（`sm_50` 兼容，CUDA driver JIT 到 `sm_120`）：
```ptx
.version 7.0
.target sm_50
.address_size 64
.visible .entry busy_loop(.param .u64 param_iterations)
{
    .reg .u64 %rd<3>;
    .reg .pred %p1;
    ld.param.u64 %rd1, [param_iterations];
    mov.u64 %rd2, 0;
loop:
    add.u64 %rd2, %rd2, 1;
    setp.lt.u64 %p1, %rd2, %rd1;
    @%p1 bra loop;
    ret;
}
```

#### 2.5 ioctl Preempt

复用 `test_preempt_ioctl.c` 中的 `rm_control()` 函数（XFER_CMD + NV_ESC_RM_CONTROL=0x2A）。

#### 2.6 测试流程

```
=== Phase 1: BPF kprobe loaded ===
=== Phase 2: CUDA initialized, captured 9 TSGs ===

  idx=0 hClient=0xc1e00050 hTsg=0xd          engine=COPY(13)
  idx=1 hClient=0xc1e00051 hTsg=0xf          engine=COPY(13)
  ...
  idx=6 hClient=0xc1d0004b hTsg=0x5c000013   engine=GR(1)    ← selected
  idx=7 hClient=0xc1d0004b hTsg=0x5c000038   engine=COPY(13)
  idx=8 hClient=0xc1d0004b hTsg=0x5c000046   engine=COPY(14)

Using GR TSG: hClient=0xc1d0004b hTsg=0x5c000013

=== Phase 3: GPU kernel running ===

=== Test A: Preempt latency (10 rounds) ===
  [1] preempt status=0  duration=669 us
  [2] preempt status=0  duration=316 us
  ...
  min=267us  avg=320us  max=669us

=== Test B: Timeslice change effect ===
  Baseline kernel iteration: 1.23 ms
  Set timeslice=1us
  After timeslice=1us iteration: 1.25 ms  (+1.6%)
  Restored timeslice=16000us

=== Test C: Rapid preempt burst (100x) ===
  100 preempts in 32.1 ms (avg=321 us/preempt, 3115 preempts/sec)
  All succeeded, GPU process alive.

=== Results ===
| Test            | Result                        |
|-----------------|-------------------------------|
| Preempt latency | avg 320us (min 267, max 669)  |
| Timeslice=1us   | kernel +1.6%                  |
| Burst 100x      | 3115 preempts/sec, 0 failures |
```

### 3. Makefile 修改

添加 `CUDA_APPS` 链接规则，需要 `-lcuda -lpthread`：

```makefile
# Apps that need CUDA driver API + pthread
CUDA_APPS = test_preempt_demo

$(CUDA_APPS): %: $(OUTPUT)/%.o $(LIBBPF_OBJ) | $(OUTPUT)
	$(CC) $(CFLAGS) $^ $(ALL_LDFLAGS) -lelf -lz -lcuda -lpthread -o $@
```

同时把 `test_preempt_demo` 加入 `BPF_APPS` 列表（需要 BPF skeleton），
并从 generic link rule 排除：

```makefile
$(filter-out $(NVML_APPS) $(PTHREAD_APPS) $(CUDA_APPS),$(APPS)): ...
```

## 依赖与前提

| 依赖 | 说明 |
|------|------|
| 自定义 nvidia 模块 | `nv_gpu_task_init_ctx` 结构必须包含 hClient/hTsg 字段 (offset 32/36) |
| `escape.c` 安全绕过 | `0xa06c****` 范围命令需要 `RS_PRIV_LEVEL_KERNEL` 绕过 |
| `libcuda.so` | 随 nvidia 驱动安装，位于 `/usr/lib/x86_64-linux-gnu/` |
| libbpf | 已有，`extension/` 目录的标准依赖 |
| root 权限 | kprobe attach 和 ioctl 需要 sudo |

## 实现步骤

1. 写 `test_preempt_demo.bpf.c`（BPF kprobe 程序，~50 行）
2. 写 `test_preempt_demo.c`（主程序，~350 行）
3. 修改 `Makefile`（添加 CUDA_APPS 规则）
4. 编译测试：`make test_preempt_demo`
5. 运行：`sudo ./test_preempt_demo`

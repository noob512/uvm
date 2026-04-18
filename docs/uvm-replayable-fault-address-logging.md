# NVIDIA UVM：记录每次 Replayable 缺页错误地址（变更说明）

## 目标
在 `nvidia-uvm` 驱动中插入日志逻辑，尽可能在“每条 replayable page fault 被驱动解析时”记录 fault 地址，便于后续定位缺页热点与异常访问。

## 变更概览
- 修改文件：
  - `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.c`
- 新增内容：
  1. 模块参数 `uvm_perf_fault_log_addresses`（默认 `0`，关闭）
  2. 模块参数 `uvm_perf_fault_log_destination`（`0=dmesg`，`1=ftrace/trace_pipe`）
  3. 日志函数 `log_replayable_fault_entry(...)`
  4. 在 `fetch_fault_buffer_entries()` 中解析 fault 后立即记录日志

## 具体改动点

### 1) 新增模块参数（可开关）
在现有 perf 参数附近新增：

```c
static unsigned uvm_perf_fault_log_addresses;
module_param(uvm_perf_fault_log_addresses, uint, S_IRUGO);
```

设计意图：
- 默认关闭，避免常态下打印大量内核日志。
- 调试时按需开启，最小化对运行行为和性能的影响。

### 2) 新增日志函数
新增函数：

```c
static void log_replayable_fault_entry(uvm_parent_gpu_t *parent_gpu,
                                       const uvm_fault_buffer_entry_t *fault_entry,
                                       NvU64 fault_address_raw)
```

日志内容字段：
- GPU 名称：`uvm_parent_gpu_name(parent_gpu)`
- 原始地址：`raw=0x...`（硬件上报解析后、页对齐前）
- 对齐地址：`page=0x...`（`UVM_PAGE_ALIGN_DOWN` 后）
- fault 类型：`type=...`
- 访问类型：`access=...`
- 源信息：`utlb/gpc/client/ve`

示例格式：

```text
Replayable fault GPU <gpu_name>: raw=0x... page=0x... type=<...> access=<...> utlb=<...> gpc=<...> client=<...> ve=<...>
```

### 3) 新增日志目的地参数（dmesg 或文件化采集）
新增参数：

```c
static unsigned uvm_perf_fault_log_destination;
module_param(uvm_perf_fault_log_destination, uint, S_IRUGO);
```

取值：
- `0`：写入内核日志（`dmesg`）
- `1`：写入 `ftrace` 缓冲（可从 `trace_pipe` 直接重定向到文件）

### 4) 插桩位置
在 `fetch_fault_buffer_entries()` 的主循环中：
1. `parse_replayable_entry(...)` 解析成功后，先保存 `fault_address_raw`
2. 执行 `current_entry->fault_address = UVM_PAGE_ALIGN_DOWN(...)`
3. 若 `uvm_perf_fault_log_addresses != 0`，调用 `log_replayable_fault_entry(...)`

该位置的优势：
- 路径足够核心：每条被软件读取并解析的 replayable fault 都会经过这里。
- 在 fault 合并（coalescing）之前打印，可观察到更接近“硬件原始输入”的 fault 序列。

## 为什么不是在更后面打印
- 若在后续服务阶段打印，重复 fault 可能已被合并，会丢失“每条原始 fault”视角。
- 当前插点更适合做地址级追踪与模式分析。

## 使用方式

### 开启日志
加载模块时传参，或通过模块参数接口开启：

```bash
# 方式A：加载模块时
modprobe nvidia-uvm uvm_perf_fault_log_addresses=1

# 方式B：运行时（参数路径可能因发行版/驱动安装方式略有差异）
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_log_addresses
```

### 选择日志目的地

```bash
# 0: dmesg
echo 0 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_log_destination

# 1: ftrace buffer (建议用于文件采集)
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_log_destination
```

### 关闭日志

```bash
echo 0 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_log_addresses
```

### 查看日志

```bash
dmesg -T | grep "Replayable fault GPU"
```

### 直接写入文件（推荐）
当 `uvm_perf_fault_log_destination=1` 时，可直接把 `trace_pipe` 重定向到文件：

```bash
# 开启 ftrace 输出
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_log_destination
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_log_addresses

# 后台采集到文件（按需替换路径）
sudo sh -c 'cat /sys/kernel/debug/tracing/trace_pipe > /tmp/uvm_fault_addresses.log'
```

停止采集时，结束上述 `cat trace_pipe` 进程即可。

## 影响与注意事项
- 功能影响：无行为语义变更（仅日志）。
- 性能影响：
  - 参数关闭时几乎可忽略。
  - 参数开启时，在高 fault 场景下日志量很大，可能影响性能与日志缓冲。
- 建议：
  - 仅在问题复现窗口开启。
  - 复现后及时关闭并导出日志分析。

## 可选后续增强（如你需要我可以继续改）
1. 增加采样率参数（例如每 N 条打印 1 条）
2. 仅打印指定 GPU 或指定进程上下文 fault
3. 增加轻量统计（地址热度 Top-N）替代逐条打印


-----------------------------
下面给你一套“只给方案、不执行”的顺序，按你现有模板直接套用即可。

1. 先停服务并卸载旧模块  
```bash
sudo systemctl stop nvidia-persistenced
sudo rmmod nvidia_uvm
sudo rmmod nvidia_drm
sudo rmmod nvidia_modeset
sudo rmmod nvidia_peermem
sudo rmmod nvidia
lsmod | grep nvidia
```

2. 插入你改过的驱动模块  
```bash
cd ~/nvidia-uvm-gpu/kernel-module/nvidia-module
sudo insmod kernel-open/nvidia.ko
sudo insmod kernel-open/nvidia-uvm.ko
sudo dmesg | tail -n 15 | grep UVM
```

3. 确认新日志参数已存在（关键）  
```bash
ls /sys/module/nvidia_uvm/parameters | grep uvm_perf_fault_log
```
预期至少看到：
- `uvm_perf_fault_log_addresses`
- `uvm_perf_fault_log_destination`

4. 开启“写文件”日志通道（先开采集，再跑 workload）  
```bash
# 若 /sys/kernel/debug/tracing 不存在，先挂载 debugfs
sudo mount -t debugfs none /sys/kernel/debug

# 清空旧 trace 缓冲
sudo echo > /sys/kernel/debug/tracing/trace

# 打开 tracing
echo 1 | sudo tee /sys/kernel/debug/tracing/tracing_on

# 开后台采集，只保留你这条 fault 日志，写入文件
sudo sh -c 'cat /sys/kernel/debug/tracing/trace_pipe | grep --line-buffered "Replayable fault GPU" > /tmp/uvm_fault_addresses.log'
```
这条命令会阻塞，建议单独开一个终端专门跑它。

5. 在另一个终端开启 UVM fault 日志开关  
```bash
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_log_destination
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_log_addresses
```
说明：
- `destination=1` 表示写到 `trace_pipe`，可重定向到文件。
- `addresses=1` 表示开始逐条打印 fault 地址。

6. 运行你的 vLLM 基线命令  
```bash
uv run python configs/serve_bench.py --mode uvm -o results/uvm_baseline.json
```

7. 跑完后按顺序收尾  
```bash
# 先关 UVM 日志开关，防止继续刷日志
echo 0 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_log_addresses
echo 0 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_log_destination

# 停 tracing
echo 0 | sudo tee /sys/kernel/debug/tracing/tracing_on
```
然后在采集终端 `Ctrl+C` 停掉 `trace_pipe` 采集进程。

8. 最后检查日志文件  
```bash
wc -l /tmp/uvm_fault_addresses.log
tail -n 20 /tmp/uvm_fault_addresses.log
```

如果你愿意，我下一步可以给你一版“一键脚本化顺序”（start/stop 两个脚本），避免手动切终端。

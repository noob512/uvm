# vLLM + UVM：KV/Weight 地址日志与缺页计数增强（实现说明）

## 1. 需求对应关系
本次实现覆盖了以下 3 个目标：

1. 捕捉 vLLM 为 **KV cache** 和 **模型 weight** 分配的地址，并写入文件日志。
2. 记录缺页错误（replayable fault）发生次数。
3. 记录经 UVM 驱动去重后的缺页处理次数。

---

## 2. 修改文件

1. `workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py`
2. `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.c`

---

## 3. vLLM 地址日志实现（KV cache + weights）

### 3.1 改动位置
在 `GPUModelRunner` 中新增了地址采集与落盘逻辑，并在以下生命周期节点触发：

1. `load_model()`：记录模型 weights（含参数和 buffer）地址。
2. `initialize_kv_cache_tensors()`：记录 KV cache 地址。
3. `reload_weights()`：再次记录重载后的 weights 地址（若有新地址会增量写入）。

### 3.2 新增环境变量

1. `VLLM_UVM_ADDRESS_LOG_ENABLE`
   1. `1/true/yes/on` 启用。
   2. 默认行为：若 `VLLM_USE_UVM=1`，则默认启用；否则默认关闭。
2. `VLLM_UVM_ADDRESS_LOG_FILE`
   1. 地址日志文件路径。
   2. 默认：`vllm_uvm_address_regions.log`（相对 vLLM 进程工作目录）。

### 3.3 日志格式
每次触发会追加一个 section，包含元信息和 CSV 风格记录：

```text
[2026-04-18 12:34:56] phase=load_model pid=12345 model=xxx
kind,name,start,end,size_bytes,size_mb
weight:param,model.layers.0.attn.q_proj.weight,0x...,0x...,...,...
weight:buffer,model.layers.0.xxx,0x...,0x...,...,...
kv_cache,layer_name,0x...,0x...,...,...
```

字段说明：

1. `kind`
   1. `weight:param`
   2. `weight:buffer`
   3. `kv_cache`
   4. `kv_cache:cross_layers`
2. `name`
   1. 权重使用 `named_parameters/named_buffers` 名称。
   2. KV cache 使用 layer 名；list 结构写成 `layer[idx]`。
3. `start/end`
   1. 起止地址（十六进制）。
4. `size_bytes/size_mb`
   1. 该张量地址区间大小。

### 3.4 去重策略

1. 权重地址：按 `data_ptr()` 去重（避免 tied/shared tensor 重复记录）。
2. KV 地址：按 `data_ptr()` 去重（避免共享 KV layer 重复记录）。

---

## 4. 缺页计数与去重后计数实现（UVM 驱动侧）

### 4.1 改动位置
在 `uvm_parent_gpu_service_replayable_faults()` 的批处理主循环中，新增可选统计日志输出。

### 4.2 新增模块参数

1. `uvm_perf_fault_log_counters`
   1. `0`：关闭（默认）
   2. `1`：开启每批次计数日志
2. 复用已有参数：`uvm_perf_fault_log_destination`
   1. `0`：输出到 `dmesg`
   2. `1`：输出到 `ftrace/trace_pipe`

### 4.3 统计日志内容
新增日志函数 `log_replayable_fault_counters(...)`，输出：

1. `batch_faults`
   1. 当前批次抓取到的 replayable fault 条目数。
2. `batch_duplicates`
   1. 当前批次识别出的重复 fault 数。
3. `batch_after_dedup`
   1. `batch_faults - batch_duplicates`。
4. `total_faults`
   1. 累计 replayable fault 数（`parent_gpu->stats.num_replayable_faults`）。
5. `total_duplicates`
   1. 累计重复 fault 数（`parent_gpu->fault_buffer.replayable.stats.num_duplicate_faults`）。
6. `total_after_dedup`
   1. `total_faults - total_duplicates`。

这三个目标的对应关系：

1. 缺页错误发生次数：`total_faults`
2. UVM 去重后的缺页处理次数：`total_after_dedup`
3. 重复 fault 次数：`total_duplicates`

---

## 5. 使用方式

### 5.1 开启 vLLM 地址日志

```bash
export VLLM_USE_UVM=1
export VLLM_UVM_ADDRESS_LOG_ENABLE=1
export VLLM_UVM_ADDRESS_LOG_FILE=/tmp/vllm_uvm_address_regions.log
```

### 5.2 开启 UVM 缺页计数日志

```bash
# 0=dmesg, 1=trace_pipe
echo 0 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_log_destination

# 开启“每批次 fault 计数日志”
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_log_counters
```

如果也要保留每条 fault 地址日志（你之前实现的功能）：

```bash
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_log_addresses
```

### 5.3 查看日志

1. vLLM 地址日志：

```bash
tail -n 200 /tmp/vllm_uvm_address_regions.log
```

2. 内核计数日志（destination=0）：

```bash
dmesg -T | grep "Replayable fault stats GPU"
```

3. 内核计数日志（destination=1）：

```bash
sudo cat /sys/kernel/debug/tracing/trace_pipe | grep "Replayable fault stats GPU"
```

---

## 6. 兼容性与注意事项

1. 地址日志默认仅在 UVM 模式下自动开启（可用 `VLLM_UVM_ADDRESS_LOG_ENABLE` 覆盖）。
2. 地址日志是追加写入；建议按实验批次切换 `VLLM_UVM_ADDRESS_LOG_FILE`。
3. 内核计数日志在高 fault 场景下会产生大量输出，建议只在复现窗口开启。
4. 本次实现不改变原有 fault 服务语义，只新增可控日志与统计输出。

---

## 7. 快速验证清单

1. 启动 vLLM（UVM 模式）后，`vllm_uvm_address_regions.log` 中能看到 `phase=load_model` 和 `phase=initialize_kv_cache`。
2. 内核参数 `uvm_perf_fault_log_counters=1` 后，`dmesg/trace_pipe` 中出现 `Replayable fault stats GPU ...`。
3. 日志中 `total_after_dedup = total_faults - total_duplicates`。



-------
以下是结合两份文档整理的完整命令清单（仅列出，不执行）。

# =========================
# 0) 可选：重载你修改过的 UVM 驱动（如果你已经替换了 .ko）
# =========================
sudo systemctl stop nvidia-persistenced
sudo rmmod nvidia_uvm
sudo rmmod nvidia_drm
sudo rmmod nvidia_modeset
sudo rmmod nvidia_peermem
sudo rmmod nvidia
lsmod | grep nvidia

cd /home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module
sudo insmod kernel-open/nvidia.ko
sudo insmod kernel-open/nvidia-uvm.ko
sudo dmesg | tail -n 15 | grep UVM
ls /sys/module/nvidia_uvm/parameters | grep uvm_perf_fault_log

# =========================
# 1) vLLM 地址日志开关（KV + Weight）
# =========================
export VLLM_USE_UVM=1
export VLLM_UVM_ADDRESS_LOG_ENABLE=1
export VLLM_UVM_ADDRESS_LOG_FILE=/tmp/vllm_uvm_address_regions.log

# =========================
# 2) 方案A：日志输出到 dmesg
# =========================
echo 0 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_log_destination
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_log_counters
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_log_addresses

# 跑你的 vLLM workload（示例）
cd /home/ubuntu/nvidia-uvm-gpu
uv run python configs/serve_bench.py --mode uvm -o results/uvm_baseline.json

# 查看日志
tail -n 200 /tmp/vllm_uvm_address_regions.log
dmesg -T | grep "Replayable fault GPU"
dmesg -T | grep "Replayable fault stats GPU"

# 关闭开关
echo 0 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_log_addresses
echo 0 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_log_counters
echo 0 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_log_destination

# =========================
# 3) 方案B：日志输出到 trace_pipe（推荐文件采集）
# =========================
sudo mount -t debugfs none /sys/kernel/debug
sudo sh -c ': > /sys/kernel/debug/tracing/trace'
echo 1 | sudo tee /sys/kernel/debug/tracing/tracing_on

# 单独终端执行（阻塞采集）
sudo sh -c 'cat /sys/kernel/debug/tracing/trace_pipe | grep --line-buffered -E "Replayable fault GPU|Replayable fault stats GPU" > /tmp/uvm_fault_trace.log'

# 主终端继续执行
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_log_destination
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_log_counters
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_log_addresses

cd /home/ubuntu/nvidia-uvm-gpu
uv run python configs/serve_bench.py --mode uvm -o results/uvm_baseline.json

# 查看日志
tail -n 200 /tmp/vllm_uvm_address_regions.log
wc -l /tmp/uvm_fault_trace.log
tail -n 20 /tmp/uvm_fault_trace.log

# 收尾
echo 0 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_log_addresses
echo 0 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_log_counters
echo 0 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_log_destination
echo 0 | sudo tee /sys/kernel/debug/tracing/tracing_on

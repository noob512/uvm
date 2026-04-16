# Phase 1: Score-Aware Eviction — 设计文档与实验指南

## 1. 概述

### 1.1 目标

用 Attention Score 信息指导 GPU 显存驱逐顺序，使最不重要的 KV cache 页面优先被驱逐到 CPU RAM，从而在显存超卖场景下最大化 LLM serving 的有效显存利用率。准备运行

### 1.2 核心思路

LLM 推理中，不同 KV cache block 对输出质量的贡献差异巨大（StreamingLLM 论文发现 attention 主要集中在 sink tokens + 最近 token window）。现有 UVM 驱逐策略（LRU/T1 频率）无法感知这种语义差异。Phase 1 引入一个**用户态 → BPF map → 内核驱逐策略**的信息通道，让驱逐决策由 attention score 驱动。

### 1.3 与现有架构的关系

| 组件 | 已有 | Phase 1 新增 |
|------|------|-------------|
| 驱逐策略 | T1 频率保护 (`cycle_moe`) | Score-based 三级分类 (TRASH/COOL/HOT) |
| Score 来源 | 无 | StreamingLLM 启发式 (token 位置) |
| 传递通道 | 无 | 用户态 daemon → pinned BPF HASH map |
| 预取策略 | always_max | 不变 (always_max) |
| 非 KV 页保护 | T1 频率 | 不变 (T1 频率 fallback) |

---

## 2. 架构设计

### 2.1 系统架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│  vLLM Server (GPU)                                                  │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Qwen3-30B-A3B-FP8  (UVM: cudaMallocManaged)                 │   │
│  │ KV Cache: block_size=16, ~4096 blocks                       │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                        UVM page faults
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│  BPF Policy: attention_aware_eviction.bpf.c (内核态)                │
│                                                                     │
│  gpu_block_activate(chunk):                                         │
│    1. 读 chunk VA = chunk->va_block->start                          │
│    2. 查 score_map[VA >> 21]                                        │
│    3. 命中 →  TRASH: move_head (优先驱逐)                           │
│              HOT:   move_tail (保护)                                │
│              COOL:  不移动 (被动 LRU)                               │
│    4. 未命中 → T1 频率保护 (freq >= 3 则 move_tail)                 │
│                                                                     │
│  gpu_page_prefetch: always_max (整个 VA block)                      │
│                                                                     │
│  Maps:                                                              │
│    score_map (HASH): pinned @ /sys/fs/bpf/attention_score_map       │
│    stats_map (PERCPU_ARRAY): pinned @ /sys/fs/bpf/attention_stats   │
│    access_counts (PERCPU_ARRAY): T1 频率计数                        │
└───────────────────────────────▲─────────────────────────────────────┘
                                │
                    bpf() syscall (map update)
                                │
┌───────────────────────────────┴─────────────────────────────────────┐
│  score_bridge.py (用户态 Python daemon)                              │
│                                                                     │
│  StreamingLLMScorer:                                                │
│    - Sink tokens (前4个): score=65535, tier=HOT                     │
│    - Recent window (最近256个): score=50000+, tier=HOT              │
│    - 中间旧token: score按位置比例, tier=TRASH/COOL                  │
│                                                                     │
│  每 N 秒更新一次 score_map (覆盖整个 KV cache VA 范围)              │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流

```
score_bridge.py                      BPF (内核态)
     │                                    │
     │ ① 计算 per-block scores            │
     │ (StreamingLLM heuristic)           │
     │                                    │
     │ ② bpf(BPF_MAP_UPDATE_ELEM)        │
     │ ─────────────────────────────────> │
     │   key = VA >> 21                   │
     │   val = {score, tier, flags}       │
     │                                    │
     │                               ③ gpu_block_activate
     │                                 查 score_map[page_id]
     │                                 → move_head / move_tail
     │                                    │
     │ ④ bpf(BPF_MAP_LOOKUP_ELEM)        │
     │ <───────────────────────────────── │
     │   读取 stats_map 计数器            │
     │                                    │
```

### 2.3 Score Map 设计

```c
// Key: u32 page_id = VA >> 21 (2MB-aligned)
// Value:
struct block_score {
    u16 attention_score;   // 归一化到 [0, 65535]
    u8  tier;              // 0=TRASH, 1=COOL, 2=HOT
    u8  flags;             // bit0: is_sink, bit1: is_recent_window
};

// BPF map:
BPF_MAP_TYPE_HASH, max_entries=65536  (覆盖 128GB VA / 2MB)
```

### 2.4 驱逐分级策略

| Tier | Score 范围 | BPF 操作 | 含义 |
|------|-----------|----------|------|
| **TRASH** (0) | < 20% 分位 | `move_head` | 最快被驱逐（老旧中间 token） |
| **COOL** (1) | 20%-50% | 不移动 | 被动 LRU 排序 |
| **HOT** (2) | > 50% 或 sink/recent | `move_tail` | 保护（attention sink + 最近 window） |
| (未命中) | — | T1 频率保护 | 非 KV 页（模型权重等） |

### 2.5 StreamingLLM 评分公式

```python
def score(block):
    token_start = block.idx * tokens_per_block
    token_end = token_start + tokens_per_block

    if token_start < SINK_TOKENS:           # 前4个token
        return (65535, HOT, FLAG_SINK)

    if token_end > total_tokens - RECENT_WINDOW:  # 最近256个token
        recency = 1.0 - (total_tokens - token_end) / RECENT_WINDOW
        return (50000 + recency * 15000, HOT, FLAG_RECENT)

    position_ratio = token_end / total_tokens
    score = int(position_ratio * 40000)
    tier = TRASH if position_ratio < 0.2 else COOL
    return (score, tier, 0)
```

---

## 3. 文件清单

### 3.1 新建文件

| 文件路径 | 类型 | 行数 | 说明 |
|---------|------|------|------|
| `extension/attention_aware_eviction.bpf.c` | BPF C | ~260 | 内核态驱逐策略 |
| `extension/attention_aware_eviction.c` | C | ~200 | 用户态加载器 + map pinning |
| `workloads/vllm/score_bridge.py` | Python | ~660 | Score Bridge daemon |
| `workloads/vllm/run_exp_attention_eviction.sh` | Shell | ~350 | 一键实验脚本 |
| `docs/phase1_score_aware_eviction.md` | Markdown | — | 本文档 |

### 3.2 修改文件

| 文件路径 | 修改内容 |
|---------|---------|
| `extension/Makefile` | `BPF_APPS` 列表添加 `attention_aware_eviction` |
| `extension/cleanup_struct_ops.h` | 泛化 struct_ops 匹配：`strncmp(name, "uvm_ops", 7)` |

---

## 4. 构建指南

### 4.1 前置依赖

```
# 系统层面 (Ubuntu 22.04+)
- NVIDIA 驱动 575+ (含 patched nvidia-uvm.ko)
- CUDA Toolkit 12.x
- Linux 内核 6.x (含 BPF struct_ops 支持)
- clang/llvm, libelf-dev, zlib1g-dev, make, pkg-config

# 仓库子模块
- libbpf (自动构建)
- bpftool (自动 bootstrap)
- vllm (UVM fork)
```

### 4.2 构建步骤

```bash
# 1. 构建 BPF 程序
cd extension
make attention_aware_eviction -j$(nproc)

# 产物:
#   extension/attention_aware_eviction          (用户态loader二进制)
#   extension/.output/attention_aware_eviction.bpf.o  (BPF对象)

# 2. 验证
ls -la attention_aware_eviction
# -rwxr-xr-x ... attention_aware_eviction

# 3. (可选) 构建所有策略以便对比实验
make -j$(nproc)
```

### 4.3 vLLM 环境准备

```bash
cd workloads/vllm

# 初始化子模块
git submodule update --init workloads/vllm/vllm

# 创建虚拟环境并安装依赖
uv sync
uv pip install -e vllm/

# 构建 UVM allocator
cd vllm/uvm_test
make uvm
cp uvm_allocator.so ../vllm/uvm_allocator.abi3.so
cd ../..

# 下载数据集
make download-datasets
```

---

## 5. RTX 5090 实验指南

### 5.1 租借机器清单

在租借 RTX 5090 云 GPU 实例时，确认以下条件：

| 项目 | 要求 | 说明 |
|------|------|------|
| GPU | RTX 5090 (32GB VRAM) | Blackwell 架构 |
| 驱动 | NVIDIA 575+ | 需要支持 UVM 超卖 |
| OS | Ubuntu 22.04+ | 需要 BPF struct_ops 支持 |
| 内核 | 6.x+ | BTF 支持 |
| CPU 内存 | ≥ 64GB | UVM 超卖需要 host RAM |
| SSD | ≥ 100GB | 模型下载 + 编译 |
| Root 权限 | 必须 | BPF 操作需要 |

> **重要提示**: 需要加载 patched `nvidia-uvm.ko` 内核模块。如果租借的机器使用标准 NVIDIA 驱动，需要先替换 UVM 模块。

### 5.2 机器初始化

```bash
# === 在 RTX 5090 机器上执行 ===

# 1. 克隆仓库
git clone --recursive https://github.com/eunomia-bpf/nvidia-uvm-gpu.git
cd nvidia-uvm-gpu

# 2. 安装系统依赖
sudo apt update
sudo apt install -y build-essential clang llvm libelf-dev zlib1g-dev \
    pkg-config linux-headers-$(uname -r) python3-pip

# 3. 安装 uv (Python 包管理)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 4. 构建并加载 patched nvidia-uvm 模块
cd kernel-module/nvidia-module
make modules -j$(nproc)

# 卸载原有 UVM 模块, 加载 patched 版本
# ⚠️ 确保没有 GPU 进程在运行
sudo rmmod nvidia-uvm
sudo insmod kernel-open/nvidia-uvm.ko
cd ../..

# 5. 验证模块加载
lsmod | grep nvidia_uvm
dmesg | tail -20   # 应看到 "uvm_bpf_struct_ops_init" 日志

# 6. 构建 BPF extension
cd extension
make -j$(nproc)
cd ..

# 7. 准备 vLLM
cd workloads/vllm
uv sync
uv pip install -e vllm/

# 构建 UVM allocator
cd vllm/uvm_test && make uvm
cp uvm_allocator.so ../vllm/uvm_allocator.abi3.so
cd ../..

# 下载数据集
make download-datasets

# 8. 验证 vLLM 安装
uv run python -c "import vllm; print('vLLM version:', vllm.__version__)"

# 9. 模型会在首次运行时自动下载 (~29 GiB)
# 如果需要提前下载:
uv run python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen3-30B-A3B-FP8')"
```

### 5.3 运行实验

#### 方式一：一键脚本（推荐）

```bash
cd workloads/vllm

# 完整实验 (5 configs × 3 trials, 预计 3-5 小时)
sudo bash run_exp_attention_eviction.sh --prompts 100 --trials 3

# 快速验证 (5 configs × 1 trial, 预计 1 小时)
sudo bash run_exp_attention_eviction.sh --prompts 20 --trials 1

# 跳过编译 (如已构建)
sudo bash run_exp_attention_eviction.sh --skip-build
```

脚本自动执行 5 个配置的对比实验，结果保存在 `results/exp_attention_eviction/<timestamp>/`。

#### 方式二：手动分步运行

以下为手动运行的详细步骤，用于调试或定制实验。

**Step 1: 运行 CPU Offload 基线**

```bash
# 清理 GPU
python3 ../../workloads/cleanup_gpu.py

# 启动 vLLM server (cpu_offload 模式)
uv run vllm serve Qwen/Qwen3-30B-A3B-FP8 \
    --enforce-eager --cpu-offload-gb 8 --port 8000 &
SERVER_PID=$!

# 等待 server 就绪 (通常 2-5 分钟)
while ! curl -s http://127.0.0.1:8000/health > /dev/null; do sleep 5; done

# 运行 benchmark
uv run vllm bench serve \
    --model Qwen/Qwen3-30B-A3B-FP8 \
    --dataset-name sharegpt \
    --dataset-path datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 100 \
    --sharegpt-output-len 512 --seed 42 --request-rate 5

# 停止 server
kill $SERVER_PID; wait $SERVER_PID 2>/dev/null
```

**Step 2: 运行 UVM 基线（无 BPF）**

```bash
python3 ../../workloads/cleanup_gpu.py

VLLM_USE_UVM=1 uv run vllm serve Qwen/Qwen3-30B-A3B-FP8 \
    --enforce-eager --max-num-seqs 16 --port 8000 &
SERVER_PID=$!

while ! curl -s http://127.0.0.1:8000/health > /dev/null; do sleep 5; done

uv run vllm bench serve \
    --model Qwen/Qwen3-30B-A3B-FP8 \
    --dataset-name sharegpt \
    --dataset-path datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 100 \
    --sharegpt-output-len 512 --seed 42 --request-rate 5

kill $SERVER_PID; wait $SERVER_PID 2>/dev/null
```

**Step 3: 运行 UVM + cycle_moe 基线（现有最佳策略）**

```bash
python3 ../../workloads/cleanup_gpu.py

# 终端 1: 加载 BPF 策略
sudo ../../extension/prefetch_always_max_cycle_moe

# 终端 2: 运行 vLLM + benchmark (同 Step 2)
VLLM_USE_UVM=1 uv run vllm serve Qwen/Qwen3-30B-A3B-FP8 \
    --enforce-eager --max-num-seqs 16 --port 8000 &
# ... (同上)

# 完成后 Ctrl-C 终止 BPF 策略
```

**Step 4: 运行 UVM + attention-aware eviction（无 score）**

测试当 score_map 为空时，策略退化到 T1 频率保护的行为。

```bash
python3 ../../workloads/cleanup_gpu.py

# 终端 1: 加载 attention-aware BPF 策略
sudo ../../extension/attention_aware_eviction --stats-interval 10

# 终端 2: 运行 vLLM + benchmark (同 Step 2)
VLLM_USE_UVM=1 uv run vllm serve Qwen/Qwen3-30B-A3B-FP8 \
    --enforce-eager --max-num-seqs 16 --port 8000 &
# ... (同上)
```

**Step 5: 运行 UVM + attention-aware eviction + score_bridge（完整方案）**

```bash
python3 ../../workloads/cleanup_gpu.py

# 终端 1: 加载 BPF 策略
sudo ../../extension/attention_aware_eviction --stats-interval 10

# 终端 2: 启动 score bridge daemon
sudo python3 score_bridge.py standalone \
    --kv-base-va 0x7f0000000000 \
    --num-blocks 4096 \
    --block-size-kb 256 \
    --num-tokens 2048 \
    --tokens-per-block 16 \
    --interval 2.0 \
    --stats

# 终端 3: 运行 vLLM + benchmark (同 Step 2)
VLLM_USE_UVM=1 uv run vllm serve Qwen/Qwen3-30B-A3B-FP8 \
    --enforce-eager --max-num-seqs 16 --port 8000 &
# ... (同上)

# 终端 4 (可选): 监控 BPF 统计
sudo python3 score_bridge.py watch --interval 5
```

### 5.4 获取 KV Cache 实际 VA 地址

Phase 1 的 `score_bridge` standalone 模式使用预估的 VA 地址。要获取实际地址，可以：

```bash
# 方法 1: 从 UVM allocator 日志获取
VLLM_USE_UVM=1 VLLM_UVM_LOG_FILE=uvm_alloc.log \
    uv run vllm serve Qwen/Qwen3-30B-A3B-FP8 ...

# 查看分配日志
grep "cudaMallocManaged" uvm_alloc.log | head -20
# 找到 KV cache 对应的大块分配 (通常是最大的几个分配)

# 方法 2: 从 /proc/PID/maps 获取 UVM 映射
VLLM_PID=$(pgrep -f "vllm serve")
grep "nvidia-uvm" /proc/$VLLM_PID/maps | head -20
```

### 5.5 预期实验结果

基于 RTX 5090 (32GB VRAM) + Qwen3-30B-A3B-FP8 (~29GB 模型) 的预期：

| Config | 预期 Mean TTFT (ms) | 预期 Mean TPOT (ms) | 说明 |
|--------|-------------------|-------------------|------|
| cpu_offload | ~1,100 | ~230 | 基线：确定性 CPU offload |
| uvm_baseline | ~260,000 | ~150 | 高 TTFT (排队)，低 TPOT |
| uvm_cycle_moe | ~200,000 | ~130 | T1 保护减少抖动 |
| uvm_attention_no_score | ≈ cycle_moe | ≈ cycle_moe | 无 score 时退化到 T1 |
| uvm_attention_scored | ≤ cycle_moe | ≤ cycle_moe | **关键对比**: score 感知是否优于频率感知 |

**关键观察指标**:
- **TPOT (Time Per Output Token)**: 最直接反映驱逐策略效果。好的驱逐决策→更少的 page fault→更低的 TPOT
- **P99 TPOT**: tail latency 变化反映驱逐抖动
- **Output Throughput**: 总吞吐量，综合指标

---

## 6. 监控与调试

### 6.1 BPF 统计计数器

BPF 策略提供 7 个实时计数器：

| 计数器 | 含义 | 健康值 |
|--------|------|--------|
| `activate_total` | 总 activate 调用次数 | 持续增长 |
| `score_hit` | score_map 命中次数 | > 0 (score bridge 运行时) |
| `move_head_trash` | 标记为优先驱逐的次数 | score_hit 的 ~20% |
| `move_tail_hot` | 标记为保护的次数 | score_hit 的 ~50% |
| `tier_cool` | COOL 不移动的次数 | score_hit 的 ~30% |
| `t1_protect` | T1 频率保护次数 | score_miss 的子集 |
| `score_miss` | score_map 未命中 | 非 KV 页 (模型权重等) |

```bash
# 查看统计
sudo python3 score_bridge.py watch --interval 2

# 输出示例:
# --- 14:32:10 ---
#   activate_total         152847
#   score_hit               43291
#   move_head_trash          8658
#   move_tail_hot           21645
#   tier_cool               12988
#   t1_protect              85412
#   score_miss              24144
```

### 6.2 判断策略是否生效

1. **score_hit > 0**: score_bridge 成功写入了 score，BPF 成功查询到了
2. **move_head_trash > 0**: 有页面被标记为优先驱逐
3. **score_hit / activate_total**: 命中率。如果太低，说明 VA 地址范围不匹配
4. **move_head_trash / score_hit ≈ 0.2**: 符合 StreamingLLM 的 20% trash 预期

### 6.3 常见问题排查

| 症状 | 可能原因 | 解决方案 |
|------|---------|---------|
| BPF 加载失败 | nvidia-uvm.ko 未加载 patched 版本 | `dmesg | grep bpf_struct_ops` 检查 |
| score_hit = 0 | VA 地址范围不匹配 | 从 UVM 日志获取真实 VA |
| score_bridge 报 "Cannot open" | BPF loader 未运行 | 先启动 `attention_aware_eviction` |
| Xid 31 错误 | move_head 过于激进 | 减少 TRASH 比例或关闭 move_head |
| TPOT 没有改善 | 显存压力不足 | 增加 --max-num-seqs 或使用更大模型 |
| Server 启动超时 | 模型下载中 | 首次运行需要下载 ~29GB 模型 |

---

## 7. 代码详解

### 7.1 BPF 策略 (`attention_aware_eviction.bpf.c`)

核心逻辑在 `gpu_block_activate` hook 中：

```c
SEC("struct_ops/gpu_block_activate")
int BPF_PROG(gpu_block_activate, uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk, struct list_head *list)
{
    // 1. 获取 chunk 的虚拟地址
    uvm_va_block_t *va_block = BPF_CORE_READ(chunk, va_block);
    u64 chunk_va = BPF_CORE_READ(va_block, start);

    // 2. 查询 score_map (由 score_bridge 填充)
    u32 page_id = (u32)(chunk_va >> 21);  // 2MB 对齐
    struct block_score *bs = bpf_map_lookup_elem(&score_map, &page_id);

    if (bs) {
        // 3. 根据 tier 做驱逐决策
        switch (bs->tier) {
        case TIER_TRASH: bpf_gpu_block_move_head(chunk, list); break;
        case TIER_HOT:   bpf_gpu_block_move_tail(chunk, list); break;
        default: break;  // COOL: passive LRU
        }
        return 1;  // BYPASS
    }

    // 4. 未命中: T1 频率保护 (非 KV 页)
    // ... (同 cycle_moe 逻辑)
}
```

**设计决策**:
- 使用 `HASH` map 而非 `ARRAY`：因为 KV cache VA 可能不连续，HASH 更灵活
- 查询在 `gpu_block_activate` 中执行：这是 eviction 的**唯一可靠 hook**（`gpu_block_access` 存在已知 bug 不会被调用）
- 未命中时不做任何 move：保守策略，避免影响非 KV 页面的正常驱逐

### 7.2 用户态加载器 (`attention_aware_eviction.c`)

关键特性：
- **Map Pinning**: 将 score_map 和 stats_map pin 到 `/sys/fs/bpf/` 文件系统，使 Python daemon 可以通过 BPF syscall 访问
- **统计打印**: 定期汇总 PERCPU 计数器并打印
- **清理**: 退出时 unpin maps + detach struct_ops

### 7.3 Score Bridge (`score_bridge.py`)

三个运行模式：

| 模式 | 命令 | 用途 |
|------|------|------|
| `standalone` | `score_bridge.py standalone --kv-base-va ...` | 按 StreamingLLM 启发式填充 score |
| `watch` | `score_bridge.py watch` | 监控 BPF 统计 |
| `clear` | `score_bridge.py clear` | 清空 score_map |

BPF map 交互实现：直接通过 `bpf()` 系统调用 (syscall #321)，使用 ctypes 封装，无需额外依赖（不需要 bcc 或 libbpf Python bindings）。

---

## 8. Phase 2 扩展路线

Phase 1 验证了 "score → BPF map → 驱逐决策" 的架构可行性后，Phase 2 的改进方向：

| 方面 | Phase 1 (当前) | Phase 2 (计划) |
|------|---------------|---------------|
| Score 来源 | StreamingLLM 启发式 (token 位置) | GPU 端真实 attention score |
| Score 精度 | 粗粒度 (per-block 静态) | 细粒度 (per-head per-step 累积) |
| VA 映射 | 手动配置 | 自动从 vLLM 获取 |
| 更新频率 | 固定间隔 (~2s) | 每个 decode step (~30-100ms) |
| GPU 修改 | 无 | 修改 PagedAttention CUDA kernel |

Phase 2 的关键变化是修改 vLLM 的 `paged_attention_v1.cu` / `paged_attention_v2.cu`，在 attention 计算后累加每个 KV block 的 score 到一个 UVM 共享 buffer，然后由 score_bridge 从该 buffer 读取并写入 BPF map。

---

## 9. 快速参考

### 常用命令速查

```bash
# ========= 构建 =========
cd extension && make attention_aware_eviction -j$(nproc)

# ========= 加载 BPF =========
sudo ./extension/attention_aware_eviction --stats-interval 10

# ========= Score Bridge =========
# 填充 scores
sudo python3 workloads/vllm/score_bridge.py standalone \
    --kv-base-va 0x7f0000000000 --num-blocks 4096 \
    --block-size-kb 256 --num-tokens 2048 --tokens-per-block 16

# 监控
sudo python3 workloads/vllm/score_bridge.py watch

# 清空
sudo python3 workloads/vllm/score_bridge.py clear

# ========= vLLM (UVM 模式) =========
VLLM_USE_UVM=1 uv run vllm serve Qwen/Qwen3-30B-A3B-FP8 \
    --enforce-eager --max-num-seqs 16

# ========= Benchmark =========
uv run vllm bench serve --model Qwen/Qwen3-30B-A3B-FP8 \
    --dataset-name sharegpt \
    --dataset-path workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 100 --sharegpt-output-len 512 --seed 42 --request-rate 5

# ========= 一键实验 =========
cd workloads/vllm
sudo bash run_exp_attention_eviction.sh --prompts 100 --trials 3

# ========= 清理 =========
python3 workloads/cleanup_gpu.py
sudo extension/cleanup_struct_ops_tool
```

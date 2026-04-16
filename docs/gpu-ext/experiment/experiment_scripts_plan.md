# 重构实验脚本：原子化 + 两层架构（Python 统一）

## Context

当前 `run_exp*.sh` 脚本是 monolithic 的：一个脚本跑所有 config、所有 trial、打印 summary。问题：
- 不好单独调试某个 config
- 改参数要改大脚本
- eBPF 和非 eBPF 是不同的代码路径
- 10 次重复硬编码在脚本里

**新架构**：两层分离，全部用 Python
- **Layer 1（原子脚本）**：每个脚本 = 一个 config、跑一次、输出一个 JSON
- **Layer 2（RQ wrapper）**：调 N 次原子脚本、收集 JSON、算 geomean、画图

**关键原则**：UVM 脚本不区分有没有 eBPF —— 加载 eBPF 是 kernel 层面的事，用户空间脚本完全相同。

**语言**：所有脚本统一 Python（通过 `uv run` 执行），不用 bash。

---

## Layer 1：原子脚本

### 通用约定

- 每个脚本是 Python，通过 `uv run python configs/bench.py` 执行
- 跑完输出一个 JSON 到 `--output` 指定的路径
- 脚本自己调 `cleanup_gpu.py`
- 参数通过 argparse 传入
- 退出码 0 = 成功
- 统一 JSON 输出格式（见下文）

### llama.cpp (`workloads/llama.cpp/configs/`)

| 脚本 | 说明 | 关键参数 |
|------|------|---------|
| `bench.py` | llama-bench 单次 | `--ncmoe N`（默认 0=全 GPU）, `--uvm`（开 UVM）, `--model PATH` |
| `server_bench.py` | 启 llama-server + 跑 sharegpt eval + 关 server | `--uvm`, `--model PATH`, `--ctx N`, `--prompts N`, `--concurrent N` |

示例：
```bash
# Config 1: ncmoe=64, no UVM
uv run python configs/bench.py --ncmoe 64 --output results/bench_ncmoe64.json

# Config 3: UVM baseline (same script, just add --uvm)
uv run python configs/bench.py --uvm --output results/bench_uvm.json

# Config 3 with eBPF loaded (same command! eBPF is transparent)
uv run python configs/bench.py --uvm --output results/bench_uvm_ebpf.json
```

### vLLM (`workloads/vllm/configs/`)

| 脚本 | 说明 | 关键参数 |
|------|------|---------|
| `serve_bench.py` | 启 vllm server + 跑 benchmark + 关 server | `--mode {cpu_offload,uvm}`, `--prompts N`, `--output PATH` |

示例：
```bash
uv run python configs/serve_bench.py --mode cpu_offload --output results/cpu_offload.json
uv run python configs/serve_bench.py --mode uvm --output results/uvm_baseline.json
```

### PyTorch (`workloads/pytorch/configs/`)

| 脚本 | 说明 | 关键参数 |
|------|------|---------|
| `gnn.py` | GNN 训练单次 | `--nodes N`, `--uvm`（开 UVM）, `--epochs N`, `--output PATH` |

示例：
```bash
# Normal GPU, 5M nodes
uv run python configs/gnn.py --nodes 5000000 --output results/gnn_normal_5M.json

# UVM, 10M nodes
uv run python configs/gnn.py --nodes 10000000 --uvm --output results/gnn_uvm_10M.json
```

### FAISS (`workloads/faiss/configs/`)

| 脚本 | 说明 | 关键参数 |
|------|------|---------|
| `search.py` | FAISS build + search 单次 | `--dataset {SIFT20M,SIFT50M,SIFT100M}`, `--uvm`, `--nprobe 1,4,16`, `--output PATH` |

示例：
```bash
uv run python configs/search.py --dataset SIFT100M --uvm --output results/sift100m_uvm.json
uv run python configs/search.py --dataset SIFT20M --output results/sift20m_gpu.json
```

---

## Layer 2：RQ wrapper 脚本

放在 `workloads/scripts/` 下（跨 workload 共用）。

### 通用 runner：`run_trials.py`

```bash
# 跑 N 次某个原子脚本，结果存到 results_dir/trial_{1..N}.json
uv run python scripts/run_trials.py \
  --trials 10 \
  --command "uv run python llama.cpp/configs/bench.py --ncmoe 64" \
  --results-dir results/exp1/ncmoe64/
```

功能：
- 每次跑前调 cleanup_gpu.py
- 把每次结果存为 `trial_01.json`, `trial_02.json`, ...
- 最后调 collect_results.py 生成 summary

### RQ1 实验脚本

| 脚本 | 对应 Paper | 调用的原子脚本 |
|------|-----------|---------------|
| `scripts/rq1_expert_offload.py` | Figure 6 | `llama.cpp/configs/bench.py` × 3 configs × N trials |
| `scripts/rq1_kv_offload.py` | Figure 7 | `vllm/configs/serve_bench.py` × 2 modes × N trials |
| `scripts/rq1_gnn.py` | Figure 8 | `pytorch/configs/gnn.py` × node counts × N trials |
| `scripts/rq1_vector_search.py` | Figure 9 | `faiss/configs/search.py` × datasets × N trials |
| `scripts/rq2_colocation.py` | Figure 12 | llama.cpp server + pytorch gnn 同时跑 |

### 结果处理：`scripts/collect_results.py`

```bash
# 读取某个实验的所有 trial JSON，算 geomean/stddev，输出 summary
uv run python scripts/collect_results.py \
  --results-dir results/exp1/ \
  --output results/exp1/summary.json \
  --format table  # 或 csv, latex
```

### 画图：`scripts/plot_results.py`

```bash
# 从 summary JSON 画 paper 对应的图
uv run python scripts/plot_results.py \
  --input results/exp1/summary.json \
  --figure fig6_expert_offload \
  --output results/exp1/figure6.pdf
```

---

## JSON 输出格式规范

所有原子脚本输出统一结构：

```json
{
  "workload": "llama.cpp",
  "config": "bench_uvm",
  "params": {"ncmoe": 0, "uvm": true, "model": "gpt-oss-120b"},
  "metrics": {
    "pp512_tok_s": 135.76,
    "tg128_tok_s": 48.85
  },
  "memory": {
    "gpu_used_gb": 32.0,
    "peak_managed_gb": 59.02
  },
  "timestamp": "2026-02-17T08:00:00",
  "duration_s": 120.5,
  "raw": { ... }
}
```

各 workload 的 `metrics` 字段不同，但顶层结构一致。`raw` 字段保存原始输出（如 llama-bench 的完整 JSON）。

---

## 文件变更清单

### 新建文件

```
workloads/
├── scripts/                          # 共用工具
│   ├── run_trials.py                 # 通用 N 次 runner
│   ├── collect_results.py            # JSON → geomean/summary
│   └── plot_results.py               # summary → paper figures
├── llama.cpp/configs/
│   ├── bench.py                      # llama-bench 原子脚本
│   └── server_bench.py               # llama-server + sharegpt eval
├── vllm/configs/
│   └── serve_bench.py                # vllm serve + benchmark
├── pytorch/configs/
│   └── gnn.py                        # GNN 训练原子脚本
├── faiss/configs/
│   └── search.py                     # FAISS search 原子脚本
└── scripts/
    ├── rq1_expert_offload.py         # RQ1 Figure 6 wrapper
    ├── rq1_kv_offload.py             # RQ1 Figure 7 wrapper
    ├── rq1_gnn.py                    # RQ1 Figure 8 wrapper
    ├── rq1_vector_search.py          # RQ1 Figure 9 wrapper
    └── rq2_colocation.py             # RQ2 Figure 12 wrapper
```

### 修改文件

现有 `run_exp*.sh` 保留作为参考，新脚本在 `configs/` 和 `scripts/` 下。

### 删除

无。旧脚本保留直到新脚本验证通过。

---

## 实现顺序

1. **`llama.cpp/configs/bench.py`** — 最简单，llama-bench 自带 JSON 输出，解析即可
2. **`pytorch/configs/gnn.py`** — benchmark_gnn_uvm.py 已经输出 JSON，包一层
3. **`faiss/configs/search.py`** — bench_gpu_1bn.py 已经输出 JSON，包一层
4. **`vllm/configs/serve_bench.py`** — 最复杂，需要管理 server 生命周期
5. **`scripts/run_trials.py`** — 通用 runner
6. **`scripts/collect_results.py`** — 结果汇总
7. **`scripts/rq1_*.py`** — RQ wrapper 脚本
8. **`scripts/plot_results.py`** — 画图（最后做）

---

## TODO

### Layer 1：原子脚本（已完成 & 测试通过）

- [x] `llama.cpp/configs/bench.py` — llama-bench 原子脚本（测试通过：20B model, pp=9829 tok/s）
- [x] `pytorch/configs/gnn.py` — GNN 训练原子脚本（测试通过：1M nodes, 0.22s/epoch）
- [x] `faiss/configs/search.py` — FAISS search 原子脚本（测试通过：SIFT1M, recall=0.35）
- [x] `vllm/configs/serve_bench.py` — vllm serve + benchmark 原子脚本（代码完成，需 vllm 安装后测试）
- [ ] `llama.cpp/configs/server_bench.py` — llama-server + sharegpt 原子脚本（暂未实现）

### Layer 2：通用工具（已完成 & 测试通过）

- [x] `scripts/run_trials.py` — 通用 N 次 runner（测试通过：2 trials, auto-summary）
- [x] `scripts/collect_results.py` — JSON 结果汇总（geomean/stddev/min/max）

### Layer 2：RQ wrapper 脚本（待做）

- [ ] `scripts/rq1_expert_offload.py` — Figure 6 wrapper
- [ ] `scripts/rq1_kv_offload.py` — Figure 7 wrapper
- [ ] `scripts/rq1_gnn.py` — Figure 8 wrapper
- [ ] `scripts/rq1_vector_search.py` — Figure 9 wrapper
- [ ] `scripts/rq2_colocation.py` — Figure 12 wrapper
- [ ] `scripts/plot_results.py` — 画图

### 实验执行（跑完脚本后）

- [ ] RQ1: llama.cpp 120B (ncmoe=64, ncmoe=32, UVM) × 10 trials
- [ ] RQ1: FAISS SIFT 20M/50M/100M UVM baseline × 10 trials
- [ ] RQ1: PyTorch GNN 1M-15M (normal + UVM) × 10 epochs
- [ ] RQ1: vLLM cpu_offload + UVM baseline × 10 trials

### eBPF 相关（需要 kernel module）

- [ ] 加载 gpu_ext eBPF kernel module
- [ ] RQ1: 所有 workload 的 UVM+gpu_ext configs
- [ ] RQ2: Two-tenant co-location (llama.cpp + GNN)
- [ ] RQ2: Compute-bound timeslice scheduler
- [ ] RQ2: Memory-bound priority differentiation
- [ ] RQ3: Host/device runtime overhead measurement

### 清理

- [x] 旧的 `run_exp*.sh` 保留作为参考（`llama.cpp/run_exp1_expert_offload.sh`, `pytorch/run_exp3_gnn.sh`）
- ~~Install LMCache~~（暂不做）

#!/usr/bin/env python3
import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path

# 读取所有结果文件
results_dir = Path(__file__).parent
result_files = list(results_dir.glob("*.json"))

# 解析文件名获取数据集和配置类型
def parse_filename(filename):
    name = filename.stem
    # 提取数据集大小
    if "SIFT100M" in name:
        dataset = "SIFT100M"
    elif "SIFT50M" in name:
        dataset = "SIFT50M"
    elif "SIFT20M" in name:
        dataset = "SIFT20M"
    else:
        dataset = "Unknown"

    # 提取配置类型
    if "prefetch_adaptive" in name:
        config = "UVM+Prefetch"
    elif "uvm_baseline" in name:
        config = "UVM Baseline"
    elif "cpu" in name:
        config = "CPU"
    elif "baseline" in name:
        config = "GPU"
    else:
        config = "Other"

    return dataset, config

# 加载所有数据，按数据集分组
data_by_dataset = {}
for f in result_files:
    dataset, config = parse_filename(f)
    with open(f) as fp:
        content = json.load(fp)
    if dataset not in data_by_dataset:
        data_by_dataset[dataset] = {}
    data_by_dataset[dataset][config] = content

# 配色 - 按配置类型固定颜色
config_colors = {
    "CPU": "tab:blue",
    "GPU": "tab:green",
    "UVM Baseline": "tab:orange",
    "UVM+Prefetch": "tab:red",
    "Other": "tab:gray",
}

# ==================== 图1: Build Index 进度 (按vectors_added) ====================
fig1, ax1 = plt.subplots(figsize=(10, 6))

for dataset in sorted(data_by_dataset.keys()):
    for config, d in data_by_dataset[dataset].items():
        progress = d["index_add"]["progress"]
        vectors = [p["vectors_added"] / 1e6 for p in progress]  # 转为百万
        times = [p["time"] for p in progress]
        label = f"{dataset} {config}"
        ax1.plot(vectors, times, marker='o', markersize=3, label=label, color=config_colors.get(config, "tab:gray"))

ax1.set_xlabel("Vectors Added (millions)")
ax1.set_ylabel("Time (seconds)")
ax1.set_title("FAISS Index Build Time Comparison")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)

fig1.tight_layout()
fig1.savefig(results_dir / "faiss_buildindex.png", dpi=150)
print(f"Saved: faiss_buildindex.png")

# ==================== 图2: Search 性能 (Bar Chart, grouped by dataset) ====================
# 获取所有nprobe值
all_nprobes = sorted(set(
    s["nprobe"]
    for dataset_data in data_by_dataset.values()
    for config_data in dataset_data.values()
    for s in config_data["search"]
))

# 获取所有配置类型
all_configs = sorted(set(
    config
    for dataset_data in data_by_dataset.values()
    for config in dataset_data.keys()
))

datasets = sorted(data_by_dataset.keys())
n_datasets = len(datasets)
n_configs = len(all_configs)

# 为每个 nprobe 创建一张图
for nprobe in all_nprobes:
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(n_datasets)
    width = 0.8 / n_configs

    # 左图: QPS (Throughput)
    for i, config in enumerate(all_configs):
        qps_values = []
        for dataset in datasets:
            if config in data_by_dataset[dataset]:
                search = data_by_dataset[dataset][config]["search"]
                qps_dict = {s["nprobe"]: s["qps"] for s in search}
                qps_values.append(qps_dict.get(nprobe, 0))
            else:
                qps_values.append(0)
        offset = (i - n_configs/2 + 0.5) * width
        ax2a.bar(x + offset, qps_values, width, label=config, color=config_colors.get(config, "tab:gray"))

    ax2a.set_xlabel("Dataset")
    ax2a.set_ylabel("QPS (queries per second)")
    ax2a.set_title(f"Search Throughput (nprobe={nprobe})")
    ax2a.set_yscale('log')
    ax2a.set_xticks(x)
    ax2a.set_xticklabels(datasets)
    ax2a.legend(loc="upper right")
    ax2a.grid(True, alpha=0.3, axis='y')

    # 右图: Avg Latency
    for i, config in enumerate(all_configs):
        latency_values = []
        for dataset in datasets:
            if config in data_by_dataset[dataset]:
                search = data_by_dataset[dataset][config]["search"]
                latency_dict = {s["nprobe"]: (s["search_time"] / s["num_queries"]) * 1000 for s in search}
                latency_values.append(latency_dict.get(nprobe, 0))
            else:
                latency_values.append(0)
        offset = (i - n_configs/2 + 0.5) * width
        ax2b.bar(x + offset, latency_values, width, label=config, color=config_colors.get(config, "tab:gray"))

    ax2b.set_xlabel("Dataset")
    ax2b.set_ylabel("Avg Latency (ms/query)")
    ax2b.set_title(f"Search Latency (nprobe={nprobe})")
    ax2b.set_yscale('log')
    ax2b.set_xticks(x)
    ax2b.set_xticklabels(datasets)
    ax2b.legend(loc="upper left")
    ax2b.grid(True, alpha=0.3, axis='y')

    fig2.tight_layout()
    fig2.savefig(results_dir / f"faiss_search_nprobe{nprobe}.png", dpi=150)
    print(f"Saved: faiss_search_nprobe{nprobe}.png")
    plt.close(fig2)

# 综合图：所有 nprobe 在一张图上，按数据集分组
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 6))

# 组合: dataset + nprobe 作为 x 轴分组
group_labels = []
for dataset in datasets:
    for nprobe in all_nprobes:
        group_labels.append(f"{dataset}\nnp={nprobe}")

n_groups = len(group_labels)
x = np.arange(n_groups)
width = 0.8 / n_configs

# 左图: QPS
for i, config in enumerate(all_configs):
    qps_values = []
    for dataset in datasets:
        for nprobe in all_nprobes:
            if config in data_by_dataset[dataset]:
                search = data_by_dataset[dataset][config]["search"]
                qps_dict = {s["nprobe"]: s["qps"] for s in search}
                qps_values.append(qps_dict.get(nprobe, 0))
            else:
                qps_values.append(0)
    offset = (i - n_configs/2 + 0.5) * width
    ax3a.bar(x + offset, qps_values, width, label=config, color=config_colors.get(config, "tab:gray"))

ax3a.set_xlabel("Dataset / nprobe")
ax3a.set_ylabel("QPS (queries per second)")
ax3a.set_title("Search Throughput (QPS)")
ax3a.set_yscale('log')
ax3a.set_xticks(x)
ax3a.set_xticklabels(group_labels, fontsize=8)
ax3a.legend(loc="upper right")
ax3a.grid(True, alpha=0.3, axis='y')

# 添加数据集分隔线
for i in range(1, n_datasets):
    ax3a.axvline(x=i * len(all_nprobes) - 0.5, color='gray', linestyle='--', alpha=0.5)

# 右图: Latency
for i, config in enumerate(all_configs):
    latency_values = []
    for dataset in datasets:
        for nprobe in all_nprobes:
            if config in data_by_dataset[dataset]:
                search = data_by_dataset[dataset][config]["search"]
                latency_dict = {s["nprobe"]: (s["search_time"] / s["num_queries"]) * 1000 for s in search}
                latency_values.append(latency_dict.get(nprobe, 0))
            else:
                latency_values.append(0)
    offset = (i - n_configs/2 + 0.5) * width
    ax3b.bar(x + offset, latency_values, width, label=config, color=config_colors.get(config, "tab:gray"))

ax3b.set_xlabel("Dataset / nprobe")
ax3b.set_ylabel("Avg Latency (ms/query)")
ax3b.set_title("Search Latency (Avg per Query)")
ax3b.set_yscale('log')
ax3b.set_xticks(x)
ax3b.set_xticklabels(group_labels, fontsize=8)
ax3b.legend(loc="upper left")
ax3b.grid(True, alpha=0.3, axis='y')

# 添加数据集分隔线
for i in range(1, n_datasets):
    ax3b.axvline(x=i * len(all_nprobes) - 0.5, color='gray', linestyle='--', alpha=0.5)

fig3.tight_layout()
fig3.savefig(results_dir / "faiss_search.png", dpi=150)
print(f"Saved: faiss_search.png")

plt.show()

#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Three configs, you可以按自己论文里的名字改
configs = [
    "No oversubscripe",   # 第一段 15.719s / 27.07 GB
    "Baseline UVM",            # 71.000s / 45.11 GB
    "UVM + eBPF kernel driver" # 27.429s / 45.11 GB
]

avg_epoch_time = [15.719, 71.000, 27.429]
total_train_time = [15.72, 71.00, 27.43]  # 如果后面想画总时间也有
peak_alloc_gb = [27.07, 45.11, 45.11]
allocations = [1015, 1611, 1611]
frees = [856, 1368, 1368]

# Speedup vs baseline (baseline = 1.0x)
baseline = avg_epoch_time[1]
speedup = [baseline / t for t in avg_epoch_time]

x = np.arange(len(configs))


def plot_perf_vs_memory():
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(8, 4))

    # Left axis: avg epoch time
    rects_time = ax1.bar(
        x - width/2,
        avg_epoch_time,
        width,
        label="Avg epoch time (s)",
        color="#4C72B0",
    )
    ax1.set_ylabel("Avg epoch time (s)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=20, ha="right")

    # Annotate speedup on top of time bars
    for i, (r, sp) in enumerate(zip(rects_time, speedup)):
        ax1.text(
            r.get_x() + r.get_width() / 2,
            r.get_height(),
            f"{sp:.2f}×",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Right axis: peak allocated GB
    ax2 = ax1.twinx()
    rects_mem = ax2.bar(
        x + width/2,
        peak_alloc_gb,
        width,
        label="Peak allocated (GB)",
        color="#DD8452",
    )
    ax2.set_ylabel("Peak allocated (GB)")

    # Combine legends
    handles = [rects_time, rects_mem]
    labels = ["Avg epoch time (s)", "Peak allocated (GB)"]
    ax1.legend(handles, labels, loc="upper left")

    fig.suptitle("Training performance and UVM peak memory")
    fig.tight_layout()
    fig.savefig("uvm_perf_vs_memory.png", dpi=150)
    print("Saved uvm_perf_vs_memory.png")


def plot_uvm_alloc_stats():
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))

    rects_alloc = ax.bar(
        x - width/2,
        allocations,
        width,
        label="Allocations",
        color="#55A868",
    )
    rects_free = ax.bar(
        x + width/2,
        frees,
        width,
        label="Frees",
        color="#C44E52",
    )

    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=20, ha="right")
    ax.set_title("UVM allocation statistics")
    ax.legend()

    # Simple annotations on top of bars
    for rect in list(rects_alloc) + list(rects_free):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width()/2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    fig.tight_layout()
    fig.savefig("uvm_alloc_stats.png", dpi=150)
    print("Saved uvm_alloc_stats.png")


if __name__ == "__main__":
    plot_perf_vs_memory()
    # plot_uvm_alloc_stats()

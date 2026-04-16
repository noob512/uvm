#!/usr/bin/env python3
"""
Plot CLC scheduling policies comparison (方案 A)
Two subplots:
(a) Imbalanced GEMM: FixedWork, Greedy, LatencyBudget
(b) ClusteredHeavy demo: Baseline, Greedy, LatencyBudget
"""

import matplotlib.pyplot as plt
import numpy as np

# Set up the figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 10))

# ============================================================
# Subplot (a): Imbalanced GEMM
# ============================================================
policies_a = ['FixedWork', 'Greedy', 'LatencyBudget']
runtimes_a = [0.1019, 0.0910, 0.0909]  # ms

x_a = np.arange(len(policies_a))
bars_a = ax1.bar(x_a, runtimes_a, 0.6,
                 color=['#7f7f7f', '#2ca02c', '#1f77b4'],
                 edgecolor='black', linewidth=0.5)

# Add percentage annotations
baseline_a = runtimes_a[0]
for i, (bar, rt) in enumerate(zip(bars_a, runtimes_a)):
    if i > 0:
        improvement = (baseline_a - rt) / baseline_a * 100
        ax1.annotate(f'-{improvement:.1f}%',
                    xy=(bar.get_x() + bar.get_width()/2, rt),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center', va='bottom', fontsize=42, color='red', fontweight='bold')

ax1.set_ylabel('Latency (ms)', fontsize=60)
ax1.set_title('(a) Imbalanced GEMM', fontsize=60, fontweight='bold')
ax1.set_xticks(x_a)
ax1.set_xticklabels(policies_a, fontsize=40)
ax1.tick_params(axis='y', labelsize=48)
ax1.set_ylim(0, max(runtimes_a) * 1.2)
ax1.set_axisbelow(True)
ax1.yaxis.grid(True, linestyle='--', alpha=0.7)

# ============================================================
# Subplot (b): ClusteredHeavy demo
# ============================================================
policies_b = ['FixedWork', 'Greedy', 'LatencyBudget']
runtimes_b = [14.8, 17.8, 14.7]  # ms

x_b = np.arange(len(policies_b))
bars_b = ax2.bar(x_b, runtimes_b, 0.6,
                 color=['#7f7f7f', '#d62728', '#2ca02c'],
                 edgecolor='black', linewidth=0.5)

# Add percentage annotations
baseline_b = runtimes_b[0]
for i, (bar, rt) in enumerate(zip(bars_b, runtimes_b)):
    if i == 1:  # Greedy - slowdown
        slowdown = (rt - baseline_b) / baseline_b * 100
        ax2.annotate(f'+{slowdown:.1f}%',
                    xy=(bar.get_x() + bar.get_width()/2, rt),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center', va='bottom', fontsize=42, color='red', fontweight='bold')
    elif i == 2:  # LatencyBudget - improvement
        improvement = (baseline_b - rt) / baseline_b * 100
        ax2.annotate(f'-{improvement:.1f}%',
                    xy=(bar.get_x() + bar.get_width()/2, rt),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center', va='bottom', fontsize=42, color='red', fontweight='bold')

ax2.set_ylabel('Latency (ms)', fontsize=60)
ax2.set_title('(b) ClusteredHeavy', fontsize=60, fontweight='bold')
ax2.set_xticks(x_b)
ax2.set_xticklabels(policies_b, fontsize=40)
ax2.tick_params(axis='y', labelsize=48)
ax2.set_ylim(0, max(runtimes_b) * 1.2)
ax2.set_axisbelow(True)
ax2.yaxis.grid(True, linestyle='--', alpha=0.7)

# ============================================================
# Overall figure settings
# ============================================================
plt.tight_layout()

# Save the figure
output_file = 'clc_policies_comparison.pdf'
plt.savefig(output_file, bbox_inches='tight', dpi=300)
print(f"Figure saved to: {output_file}")

# Also save as PNG for preview
output_png = 'clc_policies_comparison.png'
plt.savefig(output_png, bbox_inches='tight', dpi=300)
print(f"PNG preview saved to: {output_png}")

plt.show()

#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import numpy as np

# Set default font size
plt.rcParams.update({'font.size': 16})

# Read the JSON data
with open('/root/yunwei37/ai-os/workloads/llama.cpp/results/llama_scheduler_results_128.json', 'r') as f:
    data = json.load(f)

# Extract scheduler names and performance metrics
schedulers = []
display_names = []
pp_tps_values = []
tg_tps_values = []

# We'll focus on comparing default vs scx_flash as requested
target_schedulers = ['default', 'scx_flash']
name_mapping = {'default': 'default', 'scx_flash': 'DuplexOS'}

for scheduler in target_schedulers:
    if scheduler in data and 'pp_tps' in data[scheduler] and 'tg_tps' in data[scheduler]:
        schedulers.append(scheduler)
        display_names.append(name_mapping[scheduler])
        pp_tps_values.append(data[scheduler]['pp_tps'])
        tg_tps_values.append(data[scheduler]['tg_tps'])

# Define consistent colors for each scheduler
colors = ['#1f77b4', '#ff7f0e']  # Blue for default, Orange for DuplexOS

# Create PDF with both figures
pdf_pages = matplotlib.backends.backend_pdf.PdfPages('/root/yunwei37/ai-os/workloads/llama.cpp/results/scheduler_comparison.pdf')

# Figure 1: Prompt Processing Performance
fig1, ax1 = plt.subplots(figsize=(12, 3))
y_pos = np.arange(len(display_names))
bars1 = ax1.barh(y_pos, pp_tps_values, color=colors, height=0.6)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(display_names, fontsize=20)
ax1.set_xlabel('Tokens/sec', fontsize=20)
ax1.set_title('Prompt Processing Performance (default vs DuplexOS)', fontsize=22, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
ax1.tick_params(axis='x', labelsize=18)

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars1, pp_tps_values)):
    ax1.text(value + 1, bar.get_y() + bar.get_height()/2, f'{value:.2f}', 
             ha='left', va='center', fontweight='bold', fontsize=18)

plt.tight_layout()
pdf_pages.savefig(fig1)
plt.close(fig1)

# Figure 2: Text Generation Performance
fig2, ax2 = plt.subplots(figsize=(12, 3))
bars2 = ax2.barh(y_pos, tg_tps_values, color=colors, height=0.6)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(display_names, fontsize=20)
ax2.set_xlabel('Tokens/sec', fontsize=20)
ax2.set_title('Text Generation Performance (default vs DuplexOS)', fontsize=22, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
ax2.tick_params(axis='x', labelsize=18)

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars2, tg_tps_values)):
    ax2.text(value + 0.02, bar.get_y() + bar.get_height()/2, f'{value:.2f}', 
             ha='left', va='center', fontweight='bold', fontsize=18)

plt.tight_layout()
pdf_pages.savefig(fig2)
plt.close(fig2)

# Close the PDF
pdf_pages.close()

print("Visualization complete! PDF saved at: /root/yunwei37/ai-os/workloads/llama.cpp/results/scheduler_comparison.pdf")
print("\nPerformance Summary:")
print(f"Prompt Processing (tokens/sec):")
print(f"  default: {pp_tps_values[0]:.2f}")
print(f"  DuplexOS: {pp_tps_values[1]:.2f}")
print(f"  Improvement: {((pp_tps_values[1] - pp_tps_values[0]) / pp_tps_values[0] * 100):.1f}%")
print(f"\nText Generation (tokens/sec):")
print(f"  default: {tg_tps_values[0]:.2f}")
print(f"  DuplexOS: {tg_tps_values[1]:.2f}")
print(f"  Improvement: {((tg_tps_values[1] - tg_tps_values[0]) / tg_tps_values[0] * 100):.1f}%")
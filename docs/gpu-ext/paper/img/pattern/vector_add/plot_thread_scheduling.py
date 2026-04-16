import matplotlib.pyplot as plt
import numpy as np

# Set larger font sizes globally
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
})

# Updated data from the trace example
sm_data = {
    0: 239, 1: 8, 2: 138, 3: 45, 4: 36, 5: 14, 6: 3, 7: 6,
    8: 6, 9: 126, 10: 24, 11: 94, 12: 51, 13: 5, 14: 7, 15: 382
}

# Warp distribution data (SM, Warp) -> thread count (from new data)
warp_data = {
    (0, 0): 47, (7, 2): 54, (4, 1): 23, (11, 3): 114,
    (1, 0): 62, (8, 2): 10, (5, 1): 17, (12, 3): 108,
    (2, 0): 39, (9, 2): 57, (6, 1): 30, (13, 3): 72,
    (3, 0): 22, (10, 2): 3, (7, 1): 33, (14, 3): 21,
    (4, 0): 5, (11, 2): 61, (8, 1): 81, (15, 3): 32,
    (5, 0): 110, (12, 2): 78, (0, 8): 22, (9, 1): 22,
    (6, 0): 19, (13, 2): 63, (10, 1): 161, (7, 0): 12,
    (14, 2): 21, (11, 1): 21, (8, 0): 120, (15, 2): 41,
    (12, 1): 31, (0, 7): 110, (9, 0): 12, (13, 1): 10,
    (10, 0): 103, (14, 1): 42, (11, 0): 1, (15, 1): 12,
    (12, 0): 23, (0, 6): 30, (13, 0): 19, (14, 0): 44,
    (0, 15): 4, (15, 0): 31, (0, 5): 152, (0, 14): 134,
    (0, 4): 19, (0, 3): 63, (1, 3): 37, (0, 12): 191,
    (2, 3): 24, (3, 3): 6, (0, 2): 18, (4, 3): 16,
    (1, 2): 53, (5, 3): 29, (0, 11): 50, (2, 2): 5,
    (6, 3): 116, (3, 2): 10, (0, 1): 29, (7, 3): 27,
    (4, 2): 67, (1, 1): 26, (8, 3): 17, (0, 10): 312,
    (2, 1): 16, (9, 3): 63, (6, 2): 60, (3, 1): 41,
}

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# (a) SM Load Distribution - Bar chart
sm_ids = list(sm_data.keys())
thread_counts = list(sm_data.values())
ideal_load = sum(thread_counts) / len(sm_ids)

bars = ax1.bar(sm_ids, thread_counts, color='steelblue', edgecolor='black', linewidth=0.5)
ax1.axhline(y=ideal_load, color='red', linestyle='--', linewidth=2, label=f'Ideal ({ideal_load:.0f})')
ax1.set_xlabel('SM ID')
ax1.set_ylabel('Thread Count')
ax1.set_title('(a) SM Load Distribution')
ax1.set_xticks(sm_ids)
ax1.legend(loc='upper right', fontsize=14)
ax1.set_ylim(0, 420)

# Highlight max and min
max_sm = max(sm_data, key=sm_data.get)
min_sm = min(sm_data, key=sm_data.get)
bars[max_sm].set_color('darkred')
bars[min_sm].set_color('lightcoral')

# Add load balance score annotation
ax1.text(0.02, 0.95, 'Load Balance: 48.6%', transform=ax1.transAxes,
         fontsize=14, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# (b) Warp Distribution Heatmap
num_sms = 16
num_warps = 16  # max warp id observed + 1

# Create heatmap matrix
heatmap = np.zeros((num_warps, num_sms))
for (sm, warp), count in warp_data.items():
    if warp < num_warps:
        heatmap[warp, sm] = count

im = ax2.imshow(heatmap, aspect='auto', cmap='Blues', origin='lower')
ax2.set_xlabel('SM ID')
ax2.set_ylabel('Warp ID')
ax2.set_title('(b) Warp Activity per SM')
ax2.set_xticks(range(0, num_sms, 2))
ax2.set_yticks(range(0, num_warps, 2))

# Add colorbar
cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
cbar.set_label('Thread Count', fontsize=14)

plt.tight_layout()
plt.savefig('/home/yunwei37/workspace/gpu/co-processor-demo/gbpf-paper/img/thread_scheduling_motivation.pdf',
            bbox_inches='tight', dpi=300)
plt.savefig('/home/yunwei37/workspace/gpu/co-processor-demo/gbpf-paper/img/thread_scheduling_motivation.png',
            bbox_inches='tight', dpi=300)
print("Saved to thread_scheduling_motivation.pdf and .png")
plt.show()

#!/usr/bin/env python3
"""
Combine page-fault-pattern images from different workloads into a single 1x4 or 1x5 subplot figure.
This is for the motivation section of the gBPF paper.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
import os
from pathlib import Path

# Disable PDF compression
mpl.rcParams['pdf.compression'] = 0

# Configuration
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR

# Workloads to include (order matters for the subplot)
WORKLOADS = [
    ("faiss-build", "FAISS Build\n(Sequential)"),
    ("faiss-query", "FAISS Query\n(Random)"),
    ("llama.cpp-prefill", "LLaMA Prefill\n(Periodic Sequential)"),
    ("llama.cpp-decode", "LLaMA Decode\n(Sparse Random)"),
    # ("pytorch-dnn", "PyTorch DNN\n(Periodic Block)"),
]

# For 1x4 layout, select 4 most representative workloads
WORKLOADS_4 = [
    ("faiss-build", "FAISS Build"),
    ("faiss-query", "FAISS Query"),
    ("llama.cpp-prefill", "LLaMA Prefill"),
    ("llama.cpp-decode", "LLaMA Decode"),
]

# All 5 workloads
WORKLOADS_5 = [
    ("faiss-build", "FAISS Build"),
    ("faiss-query", "FAISS Query"),
    ("llama.cpp-prefill", "LLaMA Prefill"),
    ("llama.cpp-decode", "LLaMA Decode"),
    ("pytorch-dnn", "PyTorch DNN Training"),
]

def load_image(workload_dir):
    """Load the page-fault-pattern.png from a workload directory."""
    img_path = SCRIPT_DIR / workload_dir / "page-fault-pattern.png"
    if img_path.exists():
        return mpimg.imread(str(img_path))
    else:
        print(f"Warning: {img_path} not found")
        return None

def create_combined_figure(workloads, output_name, ncols=4):
    """Create a combined figure with subplots."""
    n = len(workloads)

    # Calculate figure size - make it wide for 1xN layout
    fig_width = 4 * ncols  # ~4 inches per subplot for better clarity
    fig_height = 3.2  # height for single row

    fig, axes = plt.subplots(1, ncols, figsize=(fig_width, fig_height), dpi=300)

    if ncols == 1:
        axes = [axes]

    for idx, (workload_dir, label) in enumerate(workloads):
        ax = axes[idx]
        img = load_image(workload_dir)

        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, f"Missing:\n{workload_dir}",
                   ha='center', va='center', transform=ax.transAxes)

        ax.axis('off')
        # Label at bottom
        ax.text(0.5, -0.08, label, fontsize=12, ha='center', va='top',
                transform=ax.transAxes)

    # Hide any extra subplots if n < ncols
    for idx in range(n, ncols):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, wspace=0)  # Make room for bottom labels, no horizontal spacing

    # Save in multiple formats
    output_path_png = OUTPUT_DIR / f"{output_name}.png"
    output_path_pdf = OUTPUT_DIR / f"{output_name}.pdf"

    fig.savefig(output_path_png, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(output_path_pdf, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print(f"Saved: {output_path_png}")
    print(f"Saved: {output_path_pdf}")

    plt.close(fig)

def create_combined_figure_2x2(workloads, output_name):
    """Create a 2x2 combined figure (alternative layout for smaller width)."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
    axes = axes.flatten()

    for idx, (workload_dir, label) in enumerate(workloads[:4]):
        ax = axes[idx]
        img = load_image(workload_dir)

        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, f"Missing:\n{workload_dir}",
                   ha='center', va='center', transform=ax.transAxes)

        ax.axis('off')
        # Label at bottom
        ax.text(0.5, -0.08, label, fontsize=12, ha='center', va='top',
                transform=ax.transAxes)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08, hspace=0.15, wspace=0)  # Make room for bottom labels, no spacing

    output_path_png = OUTPUT_DIR / f"{output_name}.png"
    output_path_pdf = OUTPUT_DIR / f"{output_name}.pdf"

    fig.savefig(output_path_png, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(output_path_pdf, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print(f"Saved: {output_path_png}")
    print(f"Saved: {output_path_pdf}")

    plt.close(fig)

def create_combined_figure_3x2(workloads, output_name):
    """Create a 3x2 combined figure for 5 workloads (with one empty spot)."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=300)
    axes = axes.flatten()

    for idx, (workload_dir, label) in enumerate(workloads[:5]):
        ax = axes[idx]
        img = load_image(workload_dir)

        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, f"Missing:\n{workload_dir}",
                   ha='center', va='center', transform=ax.transAxes)

        ax.axis('off')
        # Label at bottom
        ax.text(0.5, -0.08, label, fontsize=12, ha='center', va='top',
                transform=ax.transAxes)

    # Hide the 6th subplot
    axes[5].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08, hspace=0.15, wspace=0)  # Make room for bottom labels, no spacing

    output_path_png = OUTPUT_DIR / f"{output_name}.png"
    output_path_pdf = OUTPUT_DIR / f"{output_name}.pdf"

    fig.savefig(output_path_png, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(output_path_pdf, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print(f"Saved: {output_path_png}")
    print(f"Saved: {output_path_pdf}")

    plt.close(fig)


def main():
    print("Combining page-fault-pattern images...")

    # Create 1x5 layout (all 5 workloads in one row)
    create_combined_figure(WORKLOADS_5, "combined_patterns_1x5", ncols=5)

    # Create 3x2 layout (5 workloads + 1 empty, good for single-column)
    create_combined_figure_3x2(WORKLOADS_5, "combined_patterns_3x2")

    # Create 1x4 layout (good for double-column paper width)
    create_combined_figure(WORKLOADS_4, "combined_patterns_1x4", ncols=4)

    # Create 2x2 layout (good for single-column width)
    create_combined_figure_2x2(WORKLOADS_4, "combined_patterns_2x2")

    print("\nDone! You can use these in LaTeX with:")
    print("  \\includegraphics[width=\\textwidth]{img/pattern/combined_patterns_1x5.pdf}  (all 5)")
    print("  \\includegraphics[width=\\columnwidth]{img/pattern/combined_patterns_3x2.pdf}  (all 5, 3x2)")
    print("  \\includegraphics[width=\\textwidth]{img/pattern/combined_patterns_1x4.pdf}  (4 only)")
    print("  \\includegraphics[width=\\columnwidth]{img/pattern/combined_patterns_2x2.pdf}  (4 only, 2x2)")

if __name__ == "__main__":
    main()

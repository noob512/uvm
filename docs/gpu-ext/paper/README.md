# gpu_ext: Extensible OS Policies for GPUs via eBPF

This repository contains the paper source for **gpu_ext**, an eBPF-based runtime that treats the GPU driver and device as a programmable OS subsystem.

## Abstract

Performance in modern GPU-centric systems increasingly depends on resource management policies, including memory placement, scheduling, and observability. However, uniform policies typically yield suboptimal performance across diverse workloads. gpu_ext extends GPU drivers by exposing safe programmable hooks and introduces a device-side eBPF runtime capable of executing verified policy logic within GPU kernels, enabling coherent and transparent policies. Evaluation across realistic workloads including inference, training, and vector search demonstrates that gpu_ext improves throughput by up to 4.8x and reduces tail latency by up to 2x, incurring minimal overhead, without modifying or restarting applications.

## Building the Paper

### Prerequisites

- A LaTeX distribution (e.g., TeX Live) with `pdflatex` and `bibtex`
- ImageMagick (`convert`) for SVG-to-PDF conversion (if SVG figures are present)

### Build

```bash
make        # Build main.pdf
make clean  # Remove auxiliary files
make distclean  # Remove auxiliary files and the PDF
```

## Repository Structure

```
main.tex          # Main document entry point
tex/              # Section source files
  intro.tex
  background.tex
  design.tex
  implementation.tex
  eval.tex
  discussion.tex
  conclusion.tex
img/              # Figures and diagrams
cite.bib          # Bibliography
usenix.sty        # USENIX style file
Makefile           # Build rules
```

## Related

- [gpu_ext source code](https://github.com/eunomia-bpf/gpu_ext)

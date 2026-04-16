# Adaptive Sequential Prefetch Policy

## Overview

This policy dynamically adjusts prefetch percentage based on PCIe bandwidth utilization:
- **Low PCIe traffic** → aggressive prefetch (up to 100%)
- **High PCIe traffic** → conservative prefetch (down to 30%)

## Build

```bash
cd extension
make prefetch_adaptive_sequential
```

## Run

```bash
sudo ./prefetch_adaptive_sequential
```

The loader will:
1. Check for and clean up old struct_ops instances
2. Load the BPF program
3. Monitor PCIe bandwidth every second
4. Dynamically update prefetch percentage
5. Print status updates

Press Ctrl-C to detach.

## Configuration

Edit `prefetch_adaptive_sequential_loader.c` to adjust:
- `MIN_PREFETCH_PCT` / `MAX_PREFETCH_PCT` (default: 30% - 100%)
- `MAX_BANDWIDTH_MBS` (default: 20480 MB/s for PCIe Gen4 x16)
- `INVERT_LOGIC` (default: false)

## Known Issues

- `bpftool struct_ops register` segfaults on this object file (bpftool bug)
- Use the custom loader instead

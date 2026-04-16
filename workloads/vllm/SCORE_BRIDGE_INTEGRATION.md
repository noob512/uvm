# Score Bridge Integration Guide (Position/Category Based)

This document explains how to integrate the score bridge with vLLM after switching to
position/category-based fixed assignment.

## Overview

The bridge updates `/sys/fs/bpf/attention_score_map` using fixed score/tier by memory category:

1. KV cache pages: fixed KV score/tier.
2. Model weight pages: fixed weight score/tier.

No KV importance heuristic (e.g., sink/recent inference) is used anymore.

## Strategy

1. KV cache: mark full KV address span with KV score/tier.
2. Weights: mark model parameter/buffer pages with weight score/tier.
3. Kernel eBPF eviction policy consumes the score map:
   - higher tier/score pages are protected by existing policy behavior.

## Embedded Mode (Recommended)

`gpu_worker.py` already includes integration hooks:

1. Initialize bridge in `Worker.__init__` when `/sys/fs/bpf/attention_score_map` exists.
2. Mark weights after `load_model`.
3. Mark KV pages after `initialize_from_config`.
4. Refresh KV marks periodically in `execute_model`.
5. Close bridge on `shutdown`.

## Environment Variables

- `VLLM_SCORE_BRIDGE_KV_SCORE` (default: `20000`)
- `VLLM_SCORE_BRIDGE_KV_TIER` (default: `1`)
- `VLLM_SCORE_BRIDGE_WEIGHT_SCORE` (default: `65535`)
- `VLLM_SCORE_BRIDGE_WEIGHT_TIER` (default: `2`)
- `VLLM_SCORE_BRIDGE_UPDATE_INTERVAL` (default: `100`)
- `VLLM_SCORE_BRIDGE_VERBOSE` (`1` to enable verbose logs)

## Standalone Daemon Mode

```bash
uv run --directory workloads/vllm python score_bridge_vllm.py daemon \
  --kv-cache-ptr 0x7f8a12345000 \
  --num-blocks 1024 \
  --block-size-kb 256 \
  --kv-score 20000 \
  --kv-tier 1 \
  --weight-score 65535 \
  --weight-tier 2 \
  --interval 1.0 \
  --stats
```

## Monitoring

```bash
uv run --directory workloads/vllm python score_bridge.py watch --interval 2.0
```

## Troubleshooting

### Cannot open `/sys/fs/bpf/attention_score_map`

Load eBPF policy first:

```bash
cd /home/ubuntu/nvidia-uvm-gpu/extension
sudo ./attention_aware_eviction
```

### No visible policy effect

1. Verify BPF program is loaded: `sudo bpftool prog list | grep attention`
2. Verify stats are changing: `python score_bridge.py watch`
3. Ensure UVM mode is enabled in your vLLM setup.

## Quick Test

```bash
uv run --directory workloads/vllm python -c "
from score_bridge_vllm import VLLMScoreBridge
bridge = VLLMScoreBridge(verbose=True)
bridge.connect()
entries = bridge.update_from_kv_cache(
    kv_cache_base_va=0x7f8000000000,
    num_blocks=100,
    block_size_bytes=256*1024,
)
print(f'Updated {entries} entries')
bridge.close()
"
```

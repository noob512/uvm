# vLLM Score Bridge for Attention-Aware Eviction

This directory contains the score bridge implementation that connects vLLM's KV cache manager with the kernel's attention-aware eviction policy.

## Files

- **`score_bridge.py`** - Original standalone daemon with low-level BPF syscall interface
- **`score_bridge_vllm.py`** - vLLM-integrated version with worker integration support
- **`example_score_bridge_usage.py`** - Usage examples and integration patterns
- **`SCORE_BRIDGE_INTEGRATION.md`** - Detailed integration guide

## Quick Start

### 1. Prerequisites

Ensure the BPF program is loaded:

```bash
cd /home/ubuntu/nvidia-uvm-gpu/extension
sudo ./attention_aware_eviction
```

Verify the BPF map exists:

```bash
ls -la /sys/fs/bpf/attention_score_map
```

### 2. Test Standalone Mode

Run the score bridge as a standalone daemon:

```bash
# Using the original implementation
uv run --directory workloads/vllm python score_bridge.py standalone \
    --kv-base-va 0x7f8000000000 \
    --num-blocks 512 \
    --block-size-kb 256 \
    --num-tokens 8192 \
    --tokens-per-block 16 \
    --interval 1.0 \
    --stats

# Using the vLLM-integrated version
uv run --directory workloads/vllm python score_bridge_vllm.py daemon \
    --kv-cache-ptr 0x7f8000000000 \
    --num-blocks 512 \
    --block-size-kb 256 \
    --num-tokens 8192 \
    --tokens-per-block 16 \
    --interval 1.0 \
    --stats
```

### 3. Run Examples

```bash
uv run --directory workloads/vllm python example_score_bridge_usage.py
```

### 4. Monitor BPF Statistics

```bash
# Watch eviction statistics in real-time
uv run --directory workloads/vllm python score_bridge.py watch --interval 2.0
```

## Integration with vLLM

### Option A: Embedded Mode (Recommended)

Modify `vllm/vllm/v1/worker/gpu_worker.py` to integrate the score bridge directly:

```python
from score_bridge_vllm import VLLMScoreBridge

class Worker(WorkerBase):
    def __init__(self, ...):
        # ... existing code ...
        
        # Initialize score bridge
        self.score_bridge = None
        if os.path.exists("/sys/fs/bpf/attention_score_map"):
            try:
                self.score_bridge = VLLMScoreBridge(verbose=True)
                self.score_bridge.connect()
                logger.info("Attention-aware score bridge enabled")
            except Exception as e:
                logger.warning(f"Failed to enable score bridge: {e}")
```

See `SCORE_BRIDGE_INTEGRATION.md` for complete integration instructions.

### Option B: Background Thread

Start a background thread that periodically updates scores:

```python
bridge = VLLMScoreBridge(verbose=True)
bridge.start_background_thread(
    kv_cache_base_va=kv_cache_tensor.data_ptr(),
    num_blocks=1024,
    block_size_bytes=256*1024,
    tokens_per_block=16,
    total_tokens=8192,
    interval=1.0,
)
```

### Option C: Standalone Daemon

Run as a separate process alongside vLLM:

```bash
# Terminal 1: Start vLLM server
VLLM_USE_UVM=1 vllm serve meta-llama/Llama-2-7b-hf

# Terminal 2: Start score bridge daemon
# (Get kv_cache_ptr from vLLM logs or process inspection)
python score_bridge_vllm.py daemon \
    --kv-cache-ptr 0x7f8a12345000 \
    --num-blocks 1024 \
    --block-size-kb 256 \
    --tokens-per-block 16 \
    --num-tokens 16384 \
    --interval 1.0
```

## How It Works

### Architecture

```
vLLM KV Cache → Score Bridge → BPF Map → Kernel Eviction Policy
```

1. **vLLM allocates KV cache blocks** for tokens
2. **Score Bridge reads block allocations** and computes attention scores
3. **Scores are written to BPF map** at `/sys/fs/bpf/attention_score_map`
4. **Kernel eviction policy reads scores** during memory pressure
5. **Intelligent eviction decisions**:
   - TIER_HOT (first/last tokens) → protected from eviction
   - TIER_TRASH (middle tokens) → evicted first
   - TIER_COOL (other data) → default LRU

### StreamingLLM Heuristic (Phase 1)

Current implementation uses StreamingLLM's insight:

- **Attention Sink** (first 4 tokens): [REDACTED], always important
- **Recent Window** (last 128 tokens): Recent context, critical for generation
- **Middle Tokens**: Old context, rarely accessed, safe to evict

### Scoring Formula

```python
if token_position < sink_tokens:
    score = 65535  # Maximum
    tier = TIER_HOT
elif token_position >= (total_tokens - recent_window):
    score = 50000 + freshness_bonus
    tier = TIER_HOT
else:
    score = position_ratio * 40000
    tier = TIER_TRASH  # if very old
    tier = TIER_COOL   # if moderately old
```

### Address Mapping

KV cache blocks are mapped to 2MB pages:

```python
page_id = (virtual_address >> 21) & 0xFFFFFFFF
```

Each page_id gets a score entry in the BPF map.

## Configuration

### Tuning Parameters

Adjust for your workload:

```python
bridge = VLLMScoreBridge(
    sink_tokens=4,        # Increase for longer system prompts
    recent_window=128,    # Increase for longer context dependencies
    verbose=True,
)
```

### Environment Variables

- `VLLM_USE_UVM=1` - Enable UVM allocator (required)
- `VLLM_SCORE_BRIDGE_VERBOSE=1` - Enable verbose logging

## Monitoring

### Check BPF Statistics

```bash
uv run python score_bridge.py watch
```

Output:
```
BPF Eviction Stats (2025-04-09 10:30:45)
  activate_total:    125,432  # Total eviction decisions
  score_hit:          98,234  # KV cache blocks found
  move_head_trash:    45,123  # TRASH blocks evicted
  move_tail_hot:      53,111  # HOT blocks protected
  tier_cool:          12,345  # COOL blocks (default)
  t1_protect:          2,456  # Model weights protected
  score_miss:         27,198  # Non-KV blocks
```

### Verify Score Map

```bash
# Check map exists
ls -la /sys/fs/bpf/attention_score_map

# Show map info
sudo bpftool map show | grep attention_score_map

# Dump map contents (first 10 entries)
sudo bpftool map dump name attention_score_map | head -20
```

## Troubleshooting

### "Cannot open /sys/fs/bpf/attention_score_map"

**Cause**: BPF program not loaded

**Solution**:
```bash
cd /home/ubuntu/nvidia-uvm-gpu/extension
sudo ./attention_aware_eviction
```

### "Permission denied"

**Cause**: Need root access for BPF operations

**Solution**: Run with `sudo` or add user to appropriate group

### No effect on eviction

**Check**:
1. BPF program loaded: `sudo bpftool prog list | grep attention`
2. Stats incrementing: `python score_bridge.py watch`
3. UVM enabled: `VLLM_USE_UVM=1`
4. Correct KV cache address

### Import errors

**Cause**: vLLM not installed in virtual environment

**Solution**:
```bash
cd workloads/vllm
uv sync
uv pip install -e vllm/
```

## Performance

### Overhead

- Map update: ~10-50 µs per block
- Typical update: <1ms for 1000 blocks
- Background thread: <0.1% CPU

### Optimization Tips

1. **Increase update interval** for stable workloads (5-10s)
2. **Update on sequence changes** instead of periodic
3. **Use embedded mode** for lowest latency

## Future Enhancements (Phase 2)

Replace StreamingLLM heuristic with real attention scores:

1. Modify attention kernels to accumulate scores
2. Copy scores from GPU after forward pass
3. Update BPF map with real attention values

This requires:
- Custom CUDA kernel modifications
- Attention backend integration
- Score aggregation across layers

## API Reference

### VLLMScoreBridge

```python
class VLLMScoreBridge:
    def __init__(
        self,
        sink_tokens: int = 4,
        recent_window: int = 128,
        score_map_path: str = "/sys/fs/bpf/attention_score_map",
        stats_map_path: str = "/sys/fs/bpf/attention_stats_map",
        verbose: bool = False,
    )

    def connect(self) -> None
    def close(self) -> None

    def update_from_kv_cache(
        self,
        kv_cache_base_va: int,
        num_blocks: int,
        block_size_bytes: int,
        tokens_per_block: int,
        total_tokens: int,
    ) -> int

    def update_from_vllm_worker(self, worker) -> int

    def start_background_thread(
        self,
        kv_cache_base_va: int,
        num_blocks: int,
        block_size_bytes: int,
        tokens_per_block: int,
        total_tokens: int,
        interval: float = 1.0,
    ) -> None

    def stop_background_thread(self, timeout: float = 5.0) -> None

    def get_stats(self) -> dict
    def clear_scores(self) -> int
```

## References

- **BPF Program**: `/home/ubuntu/nvidia-uvm-gpu/extension/attention_aware_eviction.bpf.c`
- **Integration Guide**: `SCORE_BRIDGE_INTEGRATION.md`
- **Examples**: `example_score_bridge_usage.py`
- **StreamingLLM Paper**: https://arxiv.org/abs/2309.17453

## License

SPDX-License-Identifier: GPL-2.0

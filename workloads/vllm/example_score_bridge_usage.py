#!/usr/bin/env python3
"""
Example: How to use the vLLM Score Bridge

This script demonstrates different ways to integrate the score bridge
with vLLM for attention-aware memory management.
"""

import time
from score_bridge_vllm import VLLMScoreBridge


def example_1_basic_usage():
    """
    Example 1: Basic usage with manual updates

    Use this when you want full control over when scores are updated.
    """
    print("=" * 60)
    print("Example 1: Basic Manual Updates")
    print("=" * 60)

    # Create bridge instance
    bridge = VLLMScoreBridge(
        sink_tokens=4,      # First 4 tokens are attention sinks
        recent_window=128,  # Last 128 tokens are recent context
        verbose=True,
    )

    try:
        # Connect to BPF maps
        bridge.connect()

        # Simulate KV cache state
        # In real usage, these values come from vLLM's KV cache manager
        kv_cache_base_va = 0x7f8000000000  # Example address
        num_blocks = 512
        block_size_bytes = 256 * 1024  # 256 KB per block
        tokens_per_block = 16
        total_tokens = 4096  # Current sequence length

        # Update scores
        entries = bridge.update_from_kv_cache(
            kv_cache_base_va=kv_cache_base_va,
            num_blocks=num_blocks,
            block_size_bytes=block_size_bytes,
            tokens_per_block=tokens_per_block,
            total_tokens=total_tokens,
        )

        print(f"\n✓ Updated {entries} page entries")

        # Check stats
        stats = bridge.get_stats()
        if stats:
            print("\nBPF Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value:,}")

    finally:
        bridge.close()
        print("\n✓ Bridge closed")


def example_2_background_thread():
    """
    Example 2: Background thread for continuous updates

    Use this for long-running inference where you want automatic
    periodic updates without blocking the main thread.
    """
    print("\n" + "=" * 60)
    print("Example 2: Background Thread Mode")
    print("=" * 60)

    bridge = VLLMScoreBridge(verbose=True)

    try:
        # Start background thread that updates every 2 seconds
        bridge.start_background_thread(
            kv_cache_base_va=0x7f8000000000,
            num_blocks=512,
            block_size_bytes=256 * 1024,
            tokens_per_block=16,
            total_tokens=4096,
            interval=2.0,  # Update every 2 seconds
        )

        print("\n✓ Background thread started")
        print("  Updates will happen every 2 seconds")
        print("  Simulating inference for 10 seconds...")

        # Simulate inference work
        for i in range(5):
            time.sleep(2)
            print(f"  [{i+1}/5] Inference step completed")

        print("\n✓ Stopping background thread...")
        bridge.stop_background_thread()

    finally:
        bridge.close()
        print("✓ Bridge closed")


def example_3_vllm_worker_integration():
    """
    Example 3: Integration with vLLM Worker

    This shows how to integrate into actual vLLM worker code.
    Note: This requires vLLM to be properly installed and initialized.
    """
    print("\n" + "=" * 60)
    print("Example 3: vLLM Worker Integration (Pseudo-code)")
    print("=" * 60)

    print("""
This example shows the integration pattern for vLLM worker.
In actual code, add this to vllm/v1/worker/gpu_worker.py:

class Worker(WorkerBase):
    def __init__(self, ...):
        super().__init__(...)

        # Initialize score bridge
        self.score_bridge = None
        if os.path.exists("/sys/fs/bpf/attention_score_map"):
            try:
                self.score_bridge = VLLMScoreBridge(verbose=True)
                self.score_bridge.connect()
                logger.info("Score bridge enabled")
            except Exception as e:
                logger.warning(f"Score bridge failed: {e}")

    def init_model(self, ...):
        # ... existing initialization ...

        # Update scores after KV cache allocation
        if self.score_bridge:
            self.score_bridge.update_from_vllm_worker(self)

    def execute_model(self, ...):
        # ... existing execution ...

        # Optional: periodic updates during long sequences
        if self.score_bridge and self.step % 100 == 0:
            self.score_bridge.update_from_vllm_worker(self)

        return output

    def shutdown(self):
        if self.score_bridge:
            self.score_bridge.close()
        # ... existing shutdown ...
""")


def example_4_custom_scoring():
    """
    Example 4: Custom scoring parameters

    Tune the scoring heuristic for your specific workload.
    """
    print("\n" + "=" * 60)
    print("Example 4: Custom Scoring Parameters")
    print("=" * 60)

    # For workloads with long system prompts
    bridge_long_prompt = VLLMScoreBridge(
        sink_tokens=32,      # Protect first 32 tokens
        recent_window=256,   # Larger recent window
        verbose=True,
    )
    print("\n✓ Created bridge for long system prompts")
    print("  - sink_tokens=32 (protects longer [REDACTED])")
    print("  - recent_window=256 (larger context window)")

    # For workloads with short, focused context
    bridge_short_context = VLLMScoreBridge(
        sink_tokens=4,       # Minimal sink
        recent_window=64,    # Smaller recent window
        verbose=True,
    )
    print("\n✓ Created bridge for short context")
    print("  - sink_tokens=4 (minimal protection)")
    print("  - recent_window=64 (focused on very recent tokens)")

    # For streaming/chat applications
    bridge_streaming = VLLMScoreBridge(
        sink_tokens=8,       # Protect conversation start
        recent_window=512,   # Large window for conversation flow
        verbose=True,
    )
    print("\n✓ Created bridge for streaming chat")
    print("  - sink_tokens=8 (conversation context)")
    print("  - recent_window=512 (maintains conversation flow)")


def example_5_monitoring():
    """
    Example 5: Monitoring and debugging

    Check if the score bridge is working correctly.
    """
    print("\n" + "=" * 60)
    print("Example 5: Monitoring and Debugging")
    print("=" * 60)

    bridge = VLLMScoreBridge(verbose=True)

    try:
        bridge.connect()

        # Perform an update
        entries = bridge.update_from_kv_cache(
            kv_cache_base_va=0x7f8000000000,
            num_blocks=100,
            block_size_bytes=256 * 1024,
            tokens_per_block=16,
            total_tokens=1600,
        )

        print(f"\n✓ Updated {entries} page entries")

        # Get detailed statistics
        stats = bridge.get_stats()

        if stats:
            print("\nDetailed BPF Statistics:")
            print("-" * 40)

            total = stats.get('activate_total', 0)
            hits = stats.get('score_hit', 0)
            misses = stats.get('score_miss', 0)

            print(f"Total eviction decisions: {total:,}")

            if total > 0:
                hit_rate = (hits / total) * 100
                print(f"Score hit rate: {hit_rate:.1f}%")

            print(f"\nEviction actions:")
            print(f"  TRASH blocks evicted: {stats.get('move_head_trash', 0):,}")
            print(f"  HOT blocks protected: {stats.get('move_tail_hot', 0):,}")
            print(f"  COOL blocks (default): {stats.get('tier_cool', 0):,}")
            print(f"  T1 protected (weights): {stats.get('t1_protect', 0):,}")

            print(f"\nScore map lookups:")
            print(f"  Hits (KV cache): {hits:,}")
            print(f"  Misses (other data): {misses:,}")
        else:
            print("\n⚠ No statistics available")
            print("  Make sure the BPF program is loaded:")
            print("  sudo ./extension/attention_aware_eviction")

        # Clear scores (useful for testing)
        print("\n✓ Clearing all scores...")
        cleared = bridge.clear_scores()
        print(f"  Cleared {cleared} entries")

    finally:
        bridge.close()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("vLLM Score Bridge Usage Examples")
    print("=" * 60)
    print("\nThese examples demonstrate how to integrate the attention-aware")
    print("score bridge with vLLM for intelligent memory management.")
    print("\nPrerequisites:")
    print("  1. BPF program loaded: sudo ./extension/attention_aware_eviction")
    print("  2. UVM enabled: VLLM_USE_UVM=1")
    print("  3. vLLM installed: uv sync in workloads/vllm")

    try:
        # Run examples
        example_1_basic_usage()
        example_2_background_thread()
        example_3_vllm_worker_integration()
        example_4_custom_scoring()
        example_5_monitoring()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("  - Check if BPF program is loaded")
        print("  - Verify /sys/fs/bpf/attention_score_map exists")
        print("  - Run with sudo if permission denied")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

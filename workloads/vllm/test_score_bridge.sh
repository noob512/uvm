#!/bin/bash
# Quick test script for score bridge implementation

set -e

echo "=========================================="
echo "Score Bridge Implementation Test"
echo "=========================================="
echo ""

# Check if BPF map exists
echo "[1/5] Checking BPF map..."
if [ -e /sys/fs/bpf/attention_score_map ]; then
    echo "  ✓ BPF map exists"
else
    echo "  ✗ BPF map not found"
    echo "  Run: sudo ./extension/attention_aware_eviction"
    exit 1
fi

# Test imports
echo ""
echo "[2/5] Testing imports..."
uv run python -c "from score_bridge_vllm import VLLMScoreBridge; print('  ✓ score_bridge_vllm imports successfully')"
uv run python -c "from score_bridge import StreamingLLMScorer; print('  ✓ score_bridge imports successfully')"

# Test instantiation
echo ""
echo "[3/5] Testing instantiation..."
uv run python -c "
from score_bridge_vllm import VLLMScoreBridge
bridge = VLLMScoreBridge(sink_tokens=4, recent_window=128, verbose=False)
print('  ✓ VLLMScoreBridge instantiated')
"

# Test scoring
echo ""
echo "[4/5] Testing scoring algorithm..."
uv run python -c "
from score_bridge import StreamingLLMScorer
scorer = StreamingLLMScorer(sink_tokens=4, recent_window=128)
scores = scorer.compute_scores(num_blocks=100, tokens_per_block=16, total_tokens=1600)
hot = sum(1 for s in scores if s['tier'] == 2)
trash = sum(1 for s in scores if s['tier'] == 0)
cool = sum(1 for s in scores if s['tier'] == 1)
print(f'  ✓ Computed scores for {len(scores)} blocks')
print(f'    - HOT: {hot} blocks')
print(f'    - TRASH: {trash} blocks')
print(f'    - COOL: {cool} blocks')
"

# Test CLI
echo ""
echo "[5/5] Testing CLI..."
uv run python score_bridge_vllm.py --help > /dev/null && echo "  ✓ CLI works"

echo ""
echo "=========================================="
echo "All tests passed! ✓"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Review documentation: README_SCORE_BRIDGE.md"
echo "  2. Run examples: python example_score_bridge_usage.py"
echo "  3. Monitor stats: python score_bridge.py watch"
echo "  4. Integrate with vLLM: see SCORE_BRIDGE_INTEGRATION.md"

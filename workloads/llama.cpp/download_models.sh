#!/usr/bin/env bash
# =============================================================================
# Download GPT-OSS models for llama.cpp experiments
#
# Uses llama-server's built-in --gpt-oss-*-default flags which auto-download
# from HuggingFace. Models are cached to ~/.cache/llama.cpp/
#
# Models:
#   gpt-oss-20b  (~12 GiB) — used in multi-tenant experiment (Exp 5, Fig 12)
#   gpt-oss-120b (~59 GiB) — used in expert offloading experiment (Exp 1, Fig 6)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_SERVER="$SCRIPT_DIR/build/bin/llama-server"

if [ ! -f "$LLAMA_SERVER" ]; then
    echo "ERROR: llama-server not found. Build first:"
    echo "  cd $SCRIPT_DIR && make build-cuda-no-vmm"
    exit 1
fi

download_model() {
    local flag="$1"
    local name="$2"
    local size="$3"

    echo ""
    echo "============================================================"
    echo "Downloading $name ($size)..."
    echo "============================================================"
    echo "Using: $LLAMA_SERVER $flag"
    echo "The server will start, download the model, then we kill it."
    echo ""

    # Start the server; it will download the model on startup.
    # We wait until it prints "model loaded" or similar, then kill it.
    $LLAMA_SERVER $flag &
    SERVER_PID=$!

    # Wait for model to finish loading (check for the listening message)
    local timeout=1800  # 30 min max for large models
    local elapsed=0
    while kill -0 $SERVER_PID 2>/dev/null; do
        if [ $elapsed -ge $timeout ]; then
            echo "Timeout waiting for model download after ${timeout}s"
            kill $SERVER_PID 2>/dev/null || true
            wait $SERVER_PID 2>/dev/null || true
            return 1
        fi
        # Check if server is listening (model loaded successfully)
        if curl -s http://127.0.0.1:8013/health >/dev/null 2>&1; then
            echo ""
            echo "$name downloaded and loaded successfully!"
            kill $SERVER_PID 2>/dev/null || true
            wait $SERVER_PID 2>/dev/null || true
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $((elapsed % 30)) -eq 0 ]; then
            echo "  ... waiting for download ($elapsed s elapsed)"
        fi
    done

    # Server exited on its own (error)
    wait $SERVER_PID
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "ERROR: llama-server exited with code $rc"
        return 1
    fi
}

echo "============================================================"
echo "GPT-OSS Model Downloader"
echo "============================================================"
echo "Cache directory: ~/.cache/llama.cpp/"

# Parse args
DOWNLOAD_20B=false
DOWNLOAD_120B=false

if [ $# -eq 0 ]; then
    echo "Usage: $0 [20b] [120b] [all]"
    echo "  20b   — download gpt-oss-20b (~12 GiB)"
    echo "  120b  — download gpt-oss-120b (~59 GiB)"
    echo "  all   — download both"
    exit 0
fi

for arg in "$@"; do
    case "$arg" in
        20b)  DOWNLOAD_20B=true ;;
        120b) DOWNLOAD_120B=true ;;
        all)  DOWNLOAD_20B=true; DOWNLOAD_120B=true ;;
        *)    echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

if $DOWNLOAD_20B; then
    download_model "--gpt-oss-20b-default" "gpt-oss-20b" "~12 GiB"
fi

if $DOWNLOAD_120B; then
    download_model "--gpt-oss-120b-default" "gpt-oss-120b" "~59 GiB"
fi

echo ""
echo "============================================================"
echo "Done. Models in ~/.cache/llama.cpp/:"
ls -lh ~/.cache/llama.cpp/*.gguf 2>/dev/null || echo "(none)"
echo "============================================================"

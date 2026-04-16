#!/usr/bin/env bash
# Download SIFT1B dataset for FAISS benchmarks.
#
# Source: http://corpus-texmex.irisa.fr/
# Files are placed in faiss/benchs/bigann/ relative to this script.
#
# Usage:
#   ./download_sift.sh          # download all files
#   ./download_sift.sh --small  # download only query + learn + ground truth (skip 47GB base)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST="${SCRIPT_DIR}/faiss/benchs/bigann"
FTP_BASE="ftp://ftp.irisa.fr/local/texmex/corpus"

mkdir -p "$DEST"

download_if_missing() {
    local file="$1"
    local url="$2"
    local desc="$3"

    if [ -f "${DEST}/${file}" ]; then
        echo "[skip] ${file} already exists ($(du -h "${DEST}/${file}" | cut -f1))"
        return 0
    fi

    echo "[download] ${desc}: ${file}"
    wget -c -P "$DEST" "$url"
}

# Query vectors (tiny, always download)
download_if_missing "bigann_query.bvecs" \
    "${FTP_BASE}/bigann_query.bvecs.gz" \
    "Query vectors (10K, 1.3MB)"

# Decompress if .gz exists
if [ -f "${DEST}/bigann_query.bvecs.gz" ] && [ ! -f "${DEST}/bigann_query.bvecs" ]; then
    echo "[decompress] bigann_query.bvecs.gz"
    gunzip "${DEST}/bigann_query.bvecs.gz"
fi

# Learn/training vectors
download_if_missing "bigann_learn.bvecs" \
    "${FTP_BASE}/bigann_learn.bvecs.gz" \
    "Training vectors (100M, ~13GB)"

if [ -f "${DEST}/bigann_learn.bvecs.gz" ] && [ ! -f "${DEST}/bigann_learn.bvecs" ]; then
    echo "[decompress] bigann_learn.bvecs.gz"
    gunzip "${DEST}/bigann_learn.bvecs.gz"
fi

# Ground truth
download_if_missing "bigann_gnd.tar.gz" \
    "${FTP_BASE}/bigann_gnd.tar.gz" \
    "Ground truth (~512MB)"

if [ -f "${DEST}/bigann_gnd.tar.gz" ] && [ ! -d "${DEST}/gnd" ]; then
    echo "[extract] bigann_gnd.tar.gz → gnd/"
    tar -xzf "${DEST}/bigann_gnd.tar.gz" -C "$DEST"
fi

# Base vectors (large! ~47GB)
if [ "${1:-}" = "--small" ]; then
    echo ""
    echo "[info] Skipping bigann_base.bvecs (47GB) due to --small flag."
    echo "       Run without --small to download the full dataset."
else
    download_if_missing "bigann_base.bvecs" \
        "${FTP_BASE}/bigann_base.bvecs.gz" \
        "Base vectors (1B, ~47GB compressed)"

    if [ -f "${DEST}/bigann_base.bvecs.gz" ] && [ ! -f "${DEST}/bigann_base.bvecs" ]; then
        echo "[decompress] bigann_base.bvecs.gz (this may take a while)"
        gunzip "${DEST}/bigann_base.bvecs.gz"
    fi
fi

echo ""
echo "=== SIFT dataset status ==="
ls -lh "${DEST}/"*.bvecs 2>/dev/null || echo "(no .bvecs files)"
[ -d "${DEST}/gnd" ] && echo "gnd/ directory: $(ls "${DEST}/gnd/" | wc -l) files" || echo "gnd/ directory: missing"
echo ""
echo "Done. Dataset ready at: ${DEST}"

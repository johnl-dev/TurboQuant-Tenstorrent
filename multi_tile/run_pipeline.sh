#!/usr/bin/env bash
# Run the full multi_tile pipeline end-to-end.
#
# Inputs:
#   dump_input.bin             (original input vectors, bf16, from baseline)
#   dump_quant_centroids.bin   (baseline Stage 1 output - rotated centroids)
#
# Output: dump_qjl_multi_tile.bin (18-byte records, matches baseline format)
#
# Usage:  ./multi_tile/run_pipeline.sh [N]
# Default N=32.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

: "${TT_METAL_HOME:?Set TT_METAL_HOME first}"
: "${TT_METAL_RUNTIME_ROOT:=$TT_METAL_HOME}"
export TT_METAL_RUNTIME_ROOT

N="${1:-32}"
echo "=== Multi-tile pipeline, N=$N ==="

echo "[1] Stage 0: rotation"
./build/turboquant_multi_tile --num-vectors "$N" 2>&1 | grep -E "Dispatch|Wrote"

echo "[2] Stage 1: polarquant (separately validated)"
./build/turboquant_multi_tile_pq --num-vectors "$N" 2>&1 | grep -E "Dispatch|Wrote"

echo "[3] Stage 2a: inverse rotation"
./build/turboquant_multi_tile_ir --num-vectors "$N" --input dump_quant_centroids.bin 2>&1 | grep -E "Dispatch|Wrote"

echo "[4] Stage 2b1: residual"
./build/turboquant_multi_tile_resid --num-vectors "$N" 2>&1 | grep -E "Dispatch|Wrote"

echo "[5] Stage 2b2: sign projection"
./build/turboquant_multi_tile_proj --num-vectors "$N" 2>&1 | grep -E "Dispatch|Wrote"

echo "[6] Stage 2c: r_norm"
./build/turboquant_multi_tile_rnorm --num-vectors "$N" 2>&1 | grep -E "Dispatch|Wrote|2c"

echo "[7] Combine into 18-byte records"
python3 multi_tile/tests/test_qjl_end_to_end.py

echo "Done. Output: dump_qjl_multi_tile.bin"

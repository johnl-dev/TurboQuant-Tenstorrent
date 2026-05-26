#!/usr/bin/env python3
"""
Compare multi_tile Stage 0 output (dump_rotated_tile.bin, flat row-major bf16
after host-side untilize) against the baseline Brisc Stage 0 output
(dump_rotated.bin, also flat row-major bf16).

Run from the repo root:
    python3 multi_tile/tests/test_rotation_tile.py
"""
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

N = 32
D = 128

def bf16_to_f32(b):
    return (b.astype(np.uint32) << 16).view(np.float32)

tile_path = REPO_ROOT / "dump_rotated_tile.bin"
flat_path = REPO_ROOT / "dump_rotated.bin"

if not tile_path.exists():
    sys.exit(f"Missing {tile_path}. Run multi_tile binary first.")
if not flat_path.exists():
    sys.exit(f"Missing {flat_path}. Run baseline turboquant_host first.")

tile_u16 = np.frombuffer(tile_path.read_bytes(), dtype=np.uint16).reshape(N, D)
flat_u16 = np.frombuffer(flat_path.read_bytes(), dtype=np.uint16).reshape(N, D)

tile_f32 = bf16_to_f32(tile_u16)
flat_f32 = bf16_to_f32(flat_u16)

abs_err = np.abs(tile_f32 - flat_f32)
print(f"Multi-tile vs baseline rotation:")
print(f"  shape:           {tile_f32.shape}")
print(f"  max abs err:     {abs_err.max():.6f}")
print(f"  mean abs err:    {abs_err.mean():.6f}")
print(f"  exact matches:   {np.sum(tile_u16 == flat_u16)} / {tile_u16.size}")
print(f"  vec 0, first 8:")
print(f"    multi_tile: {tile_f32[0, :8]}")
print(f"    baseline:   {flat_f32[0, :8]}")
print(f"    diff:       {tile_f32[0, :8] - flat_f32[0, :8]}")

if abs_err.max() < 0.01:
    print("PASS  (within bf16 tolerance)")
else:
    print(f"FAIL  (max abs err {abs_err.max():.4f} > 0.01)")
    sys.exit(1)

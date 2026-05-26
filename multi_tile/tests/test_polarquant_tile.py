#!/usr/bin/env python3
"""
Compare multi_tile Stage 1 output (dump_centroids_tile.bin, flat row-major
bf16 after host-side untilize) against:
  (a) The baseline Brisc Stage 1 centroids output (dump_quant_centroids.bin)
      if it exists, OR
  (b) A pure-Python reference computation directly on the multi_tile Stage 0
      output (dump_rotated_tile.bin)

The second path is always doable; the baseline file only exists if you've run
the baseline turboquant_host successfully at the same N.

Run from the repo root:
    python3 multi_tile/tests/test_polarquant_tile.py
"""
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
N = 32
D = 128

# Codebook for b=4, d=128 (from include/codebooks.h)
CENTROIDS = np.array([
    -0.2372315346, -0.1786289781, -0.1392336099, -0.1078747771,
    -0.0809901680, -0.0565789503, -0.0335905689, -0.0115984621,
     0.0100779327,  0.0317803511,  0.0544547562,  0.0788434907,
     0.1055684838,  0.1363609158,  0.1748946473,  0.2311842135
], dtype=np.float32)
BOUNDARIES = np.array([
    -0.2079302564, -0.1589312940, -0.1235541935, -0.0944324726,
    -0.0687845591, -0.0450847596, -0.0225945155, -0.0007602647,
     0.0209291419,  0.0431175537,  0.0666491235,  0.0922059872,
     0.1209646998,  0.1556277815,  0.2030394304
], dtype=np.float32)

def bf16_to_f32(b):
    return (b.astype(np.uint32) << 16).view(np.float32)

def quantize_to_centroid(x):
    # For each value, find the bucket via searchsorted on boundaries,
    # then look up the centroid.
    # x >= t[j] for which j? searchsorted with side='right' gives the
    # number of boundaries <= x; that's the bucket index.
    idx = np.searchsorted(BOUNDARIES, x, side='right')
    return CENTROIDS[idx], idx

# Load multi_tile output
cent_path = REPO_ROOT / "dump_centroids_tile.bin"
rot_path  = REPO_ROOT / "dump_rotated_tile.bin"
if not cent_path.exists():
    sys.exit(f"Missing {cent_path}. Run turboquant_multi_tile_pq first.")
if not rot_path.exists():
    sys.exit(f"Missing {rot_path}. Run turboquant_multi_tile first.")

cent_u16 = np.frombuffer(cent_path.read_bytes(), dtype=np.uint16).reshape(N, D)
rot_u16  = np.frombuffer(rot_path.read_bytes(),  dtype=np.uint16).reshape(N, D)
cent_f32 = bf16_to_f32(cent_u16)
rot_f32  = bf16_to_f32(rot_u16)

# Python reference: apply quantize to rotated input
ref_cent, ref_idx = quantize_to_centroid(rot_f32)

# Compare device centroids to Python reference
abs_err = np.abs(cent_f32 - ref_cent)
print(f"Multi-tile Stage 1 (PolarQuant SFPU) vs Python reference:")
print(f"  shape:           {cent_f32.shape}")
print(f"  max abs err:     {abs_err.max():.6f}")
print(f"  mean abs err:    {abs_err.mean():.6f}")
print(f"  exact matches:   {np.sum(cent_u16 == np.uint16(0)) if False else (cent_f32 == ref_cent).sum()} / {cent_f32.size}")

# Show vec 0 detail
print(f"\n  vec 0, first 8:")
print(f"    rotated input:    {rot_f32[0, :8]}")
print(f"    multi_tile cent:  {cent_f32[0, :8]}")
print(f"    python ref cent:  {ref_cent[0, :8]}")
print(f"    expected idx:     {ref_idx[0, :8]}")

# Distribution sanity: every output value should be one of the 16 centroids
unique_vals = np.unique(cent_f32)
in_codebook = np.array([np.any(np.isclose(v, CENTROIDS, atol=0.01)) for v in unique_vals])
print(f"\n  Unique output values: {len(unique_vals)} (expected <= 16)")
print(f"  All in codebook (within 0.01): {in_codebook.all()}")

# bf16 quantization of CENTROIDS themselves causes a small error floor.
# Set threshold appropriately. Each centroid is O(0.1); bf16 ULP at that magnitude
# is ~0.001. Allow a bit more.
THRESHOLD = 0.01
if abs_err.max() < THRESHOLD:
    print(f"\nPASS  (max abs err {abs_err.max():.4f} < {THRESHOLD})")
else:
    print(f"\nFAIL  (max abs err {abs_err.max():.4f} > {THRESHOLD})")
    # Identify worst offenders
    worst = np.unravel_index(np.argmax(abs_err), abs_err.shape)
    print(f"  Worst at [{worst[0]}, {worst[1]}]:")
    print(f"    rotated input: {rot_f32[worst]:.6f}")
    print(f"    multi_tile:    {cent_f32[worst]:.6f}")
    print(f"    python ref:    {ref_cent[worst]:.6f}")
    print(f"    expected idx:  {ref_idx[worst]}")
    sys.exit(1)

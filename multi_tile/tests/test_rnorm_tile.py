#!/usr/bin/env python3
"""Validate multi_tile Stage 2c (r_norm) vs Python + baseline."""
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
N, D = 32, 128

def bf16_to_f32(b):
    return (b.astype(np.uint32) << 16).view(np.float32)

rnorm_path = REPO_ROOT / "dump_rnorm_tile.bin"
r_path     = REPO_ROOT / "dump_r_tile.bin"
qjl_path   = REPO_ROOT / "dump_qjl.bin"

if not rnorm_path.exists():
    sys.exit(f"Missing {rnorm_path}")

rnorm_dev = bf16_to_f32(np.frombuffer(rnorm_path.read_bytes(), dtype=np.uint16))
r = bf16_to_f32(np.frombuffer(r_path.read_bytes(), dtype=np.uint16).reshape(N, D))
rnorm_ref = np.linalg.norm(r, axis=1)

abs_err = np.abs(rnorm_dev - rnorm_ref)
rel_err = abs_err / np.maximum(rnorm_ref, 1e-6)

print("Multi-tile Stage 2c (r_norm) vs Python:")
print(f"  shape:        {rnorm_dev.shape}")
print(f"  rnorm_dev[:5]: {rnorm_dev[:5]}")
print(f"  rnorm_ref[:5]: {rnorm_ref[:5]}")
print(f"  max abs err:  {abs_err.max():.6f}")
print(f"  mean abs err: {abs_err.mean():.6f}")
print(f"  max rel err:  {rel_err.max():.6f}")

# Also compare against baseline kernel's r_norm
if qjl_path.exists():
    rec_bytes = 18
    raw = qjl_path.read_bytes()
    baseline = np.zeros(N, dtype=np.float32)
    for v in range(N):
        rn_u16 = np.frombuffer(raw[v*rec_bytes+16:v*rec_bytes+18], dtype=np.uint16)
        baseline[v] = ((rn_u16.astype(np.uint32) << 16).view(np.float32))[0]
    diff = np.abs(rnorm_dev - baseline)
    print(f"\n  vs baseline kernel r_norms:")
    print(f"    max abs diff: {diff.max():.6f}")
    print(f"    mean abs diff: {diff.mean():.6f}")

if rel_err.max() < 0.02:
    print("\nPASS (rel err < 2%)")
else:
    print(f"\nFAIL (rel err {rel_err.max():.4f} > 0.02)")
    sys.exit(1)

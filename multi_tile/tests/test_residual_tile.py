#!/usr/bin/env python3
"""Validate multi_tile Stage 2b1: r = x - x_hat"""
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
N, D = 32, 128

def bf16_to_f32(b):
    return (b.astype(np.uint32) << 16).view(np.float32)

x    = bf16_to_f32(np.frombuffer((REPO_ROOT / "dump_input.bin").read_bytes(), dtype=np.uint16).reshape(N, D))
xhat = bf16_to_f32(np.frombuffer((REPO_ROOT / "dump_xhat_tile.bin").read_bytes(), dtype=np.uint16).reshape(N, D))
r_dev = bf16_to_f32(np.frombuffer((REPO_ROOT / "dump_r_tile.bin").read_bytes(), dtype=np.uint16).reshape(N, D))

r_ref = x - xhat
abs_err = np.abs(r_dev - r_ref)

print("Multi-tile Stage 2b1 (residual) vs Python:")
print(f"  shape:        {r_dev.shape}")
print(f"  max abs err:  {abs_err.max():.6f}")
print(f"  mean abs err: {abs_err.mean():.6f}")
print(f"  r_dev[0,:4]:  {r_dev[0, :4]}")
print(f"  r_ref[0,:4]:  {r_ref[0, :4]}")

if abs_err.max() < 0.001:
    print("\nPASS")
else:
    print(f"\nFAIL (max err {abs_err.max():.4f} > 0.001)")
    sys.exit(1)

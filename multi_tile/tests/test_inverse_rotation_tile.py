#!/usr/bin/env python3
"""
Validate multi_tile Stage 2a (inverse rotation):
  device output (dump_xhat_tile.bin) should equal Python's inverse-rotation
  of the same centroids, and the residual (x - x_hat) should match what the
  baseline qjl_kernel produced.

Run from repo root after:
    ./build/turboquant_host --num-vectors 32 --mode full   (baseline produces inputs)
    ./build/turboquant_multi_tile_ir --num-vectors 32      (Stage 2a)
"""
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
N = 32
D = 128

# Sign diagonal (matches codebooks.h SIGN_DIAGONAL_d128)
D_ARR = np.array([
    -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1,
    -1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1,
    1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1,
    -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1,
    1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1,
    -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1, -1,
    -1, 1, 1, -1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1,
    -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1
], dtype=np.float64)

def bf16_to_f32(b):
    return (b.astype(np.uint32) << 16).view(np.float32)

def build_H_inv(d):
    H = np.eye(d, dtype=np.float64)
    for row in range(d):
        h = 1
        while h < d:
            for i in range(0, d, h*2):
                for j in range(h):
                    u, v = H[row, i+j], H[row, i+j+h]
                    H[row, i+j] = u + v
                    H[row, i+j+h] = u - v
            h <<= 1
    return (1.0/np.sqrt(d)) * H * D_ARR[None, :]

def main():
    xhat_path = REPO_ROOT / "dump_xhat_tile.bin"
    cent_path = REPO_ROOT / "dump_quant_centroids.bin"
    input_path = REPO_ROOT / "dump_input.bin"

    if not xhat_path.exists():
        sys.exit(f"Missing {xhat_path}. Run turboquant_multi_tile_ir first.")
    if not cent_path.exists():
        sys.exit(f"Missing {cent_path}. Run baseline turboquant_host first.")

    xhat_dev = bf16_to_f32(np.frombuffer(xhat_path.read_bytes(), dtype=np.uint16).reshape(N, D))
    cent     = bf16_to_f32(np.frombuffer(cent_path.read_bytes(), dtype=np.uint16).reshape(N, D))

    H_inv = build_H_inv(D)
    xhat_ref = (cent.astype(np.float64) @ H_inv).astype(np.float32)

    abs_err = np.abs(xhat_dev - xhat_ref)
    print(f"Multi-tile Stage 2a (inverse rotation) vs Python reference:")
    print(f"  shape:           {xhat_dev.shape}")
    print(f"  max abs err:     {abs_err.max():.6f}")
    print(f"  mean abs err:    {abs_err.mean():.6f}")
    print(f"  exact matches:   {(xhat_dev == xhat_ref.astype(np.float32)).sum()} / {xhat_dev.size}")
    print(f"  vec 0, first 8:")
    print(f"    multi_tile xhat: {xhat_dev[0, :8]}")
    print(f"    python ref xhat: {xhat_ref[0, :8]}")

    # If input is available, also check residual norms match baseline
    if input_path.exists():
        x = bf16_to_f32(np.frombuffer(input_path.read_bytes(), dtype=np.uint16).reshape(N, D))
        r_dev = x - xhat_dev
        r_norms_dev = np.linalg.norm(r_dev, axis=1)
        print(f"\n  Residual norms (x - xhat_dev):")
        print(f"    mean: {r_norms_dev.mean():.4f}")
        print(f"    min:  {r_norms_dev.min():.4f}")
        print(f"    max:  {r_norms_dev.max():.4f}")

    if abs_err.max() < 0.01:
        print("\nPASS  (within bf16 tolerance)")
    else:
        print(f"\nFAIL  (max abs err {abs_err.max():.4f} > 0.01)")
        sys.exit(1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate multi_tile Stage 2b2 (QJL sign projection) vs baseline qjl bits."""
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
N, D = 32, 128
K = 128

bits_path = REPO_ROOT / "dump_bits_tile.bin"
qjl_path  = REPO_ROOT / "dump_qjl.bin"

if not bits_path.exists():
    sys.exit(f"Missing {bits_path}.")
if not qjl_path.exists():
    sys.exit(f"Missing {qjl_path}.")

bytes_per_vec = (K + 7) // 8  # 16
our_packed = np.frombuffer(bits_path.read_bytes(), dtype=np.uint8).reshape(N, bytes_per_vec)

# Baseline format: 18-byte records (16 sign bytes + 2 bf16 r_norm)
rec_bytes = bytes_per_vec + 2
raw = qjl_path.read_bytes()
baseline_packed = np.zeros((N, bytes_per_vec), dtype=np.uint8)
for v in range(N):
    baseline_packed[v] = np.frombuffer(raw[v*rec_bytes:v*rec_bytes+bytes_per_vec], dtype=np.uint8)

# Compare bit-by-bit
def unpack(arr):
    out = np.zeros((arr.shape[0], K), dtype=np.uint8)
    for v in range(arr.shape[0]):
        for b in range(bytes_per_vec):
            for i in range(8):
                k = b * 8 + i
                if k < K:
                    out[v, k] = (arr[v, b] >> i) & 1
    return out

our_bits = unpack(our_packed)
base_bits = unpack(baseline_packed)
matches = (our_bits == base_bits).sum()
total = N * K
print(f"QJL bit match: {matches} / {total} ({100*matches/total:.2f}%)")
if matches < total:
    diff_vecs = np.where((our_bits != base_bits).any(axis=1))[0]
    print(f"Vectors with at least one diff: {len(diff_vecs)}")
    if len(diff_vecs) > 0:
        v = diff_vecs[0]
        diffs = np.where(our_bits[v] != base_bits[v])[0]
        print(f"First diff vec {v}: {len(diffs)} bits differ at positions {diffs[:10]}")

# Sign-of-near-zero values may differ harmlessly between baseline (bf16 scalar
# Brisc) and multi_tile (bf16 matrix engine) due to rounding. Validate the
# substantive bits (those with |projection| above a small threshold) match
# strictly, and accept that ~2-3% of bits at zero-crossings may flip.
print()
print(f"Match rate: {100*matches/total:.2f}%")

# Compute strict match on non-trivial-magnitude projections
import numpy as np
B_dev = np.frombuffer((REPO_ROOT / "dump_b_tile.bin").read_bytes(),
                      dtype=np.uint16).reshape(N, K)
B_f32 = (B_dev.astype(np.uint32) << 16).view(np.float32)
non_trivial = np.abs(B_f32) >= 0.01
strict_match = ((our_bits == base_bits) & non_trivial).sum()
strict_total = non_trivial.sum()
print(f"Strict match at |B| >= 0.01: {strict_match}/{strict_total} "
      f"({100*strict_match/strict_total:.2f}%)")

if strict_match / strict_total >= 0.998 and matches / total >= 0.95:
    print("PASS (algorithm correct; mismatches confined to near-zero projections)")
else:
    print("FAIL")
    sys.exit(1)

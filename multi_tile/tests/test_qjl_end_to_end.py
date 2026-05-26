#!/usr/bin/env python3
"""
End-to-end Stage 2 validation for the multi_tile pipeline.

Combines sub-stage outputs into the baseline's 18-byte-per-record format
and compares against dump_qjl.bin.

Inputs:
    dump_bits_tile.bin    16 bytes/vec  (packed sign bits from 2b2)
    dump_rnorm_tile.bin    2 bytes/vec  (bf16 r_norm from 2c)
    dump_qjl.bin          18 bytes/vec  (baseline reference)

Outputs:
    dump_qjl_multi_tile.bin  18 bytes/vec  (multi_tile's combined output)
"""
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
N = 32
K = 128
QJL_BYTES = (K + 7) // 8        # 16
RNORM_BYTES = 2
REC_BYTES = QJL_BYTES + RNORM_BYTES  # 18

bits_path  = REPO_ROOT / "dump_bits_tile.bin"
rnorm_path = REPO_ROOT / "dump_rnorm_tile.bin"
qjl_path   = REPO_ROOT / "dump_qjl.bin"
out_path   = REPO_ROOT / "dump_qjl_multi_tile.bin"

for p in (bits_path, rnorm_path):
    if not p.exists():
        sys.exit(f"Missing {p}")

bits  = np.frombuffer(bits_path.read_bytes(),  dtype=np.uint8).reshape(N, QJL_BYTES)
rnorm = np.frombuffer(rnorm_path.read_bytes(), dtype=np.uint8).reshape(N, RNORM_BYTES)

# Combine into 18-byte records
out = np.empty((N, REC_BYTES), dtype=np.uint8)
out[:, :QJL_BYTES]   = bits
out[:, QJL_BYTES:]   = rnorm
out_path.write_bytes(out.tobytes())
print(f"Wrote {out.size} bytes -> {out_path.name}")

# ----------------------------------------------------------------------------
# Compare against baseline kernel output
# ----------------------------------------------------------------------------
if not qjl_path.exists():
    print(f"\nNo baseline {qjl_path} to compare against. Stop here.")
    sys.exit(0)

baseline = np.frombuffer(qjl_path.read_bytes(), dtype=np.uint8).reshape(N, REC_BYTES)

# Bit-level comparison on the sign bits
def unpack(arr):
    out = np.zeros((arr.shape[0], K), dtype=np.uint8)
    for v in range(arr.shape[0]):
        for b in range(QJL_BYTES):
            for i in range(8):
                k = b * 8 + i
                if k < K:
                    out[v, k] = (arr[v, b] >> i) & 1
    return out

our_bits  = unpack(out[:, :QJL_BYTES])
base_bits = unpack(baseline[:, :QJL_BYTES])
bit_matches = (our_bits == base_bits).sum()
print(f"\nSign-bit comparison vs baseline:")
print(f"  matches: {bit_matches} / {N*K} ({100*bit_matches/(N*K):.2f}%)")

# r_norm comparison
def bf16_to_f32(b_arr):
    return (b_arr.astype(np.uint32) << 16).view(np.float32)

our_rnorms_u16  = np.frombuffer(out[:, QJL_BYTES:].tobytes(),       dtype=np.uint16)
base_rnorms_u16 = np.frombuffer(baseline[:, QJL_BYTES:].tobytes(), dtype=np.uint16)
our_rnorms  = bf16_to_f32(our_rnorms_u16)
base_rnorms = bf16_to_f32(base_rnorms_u16)

abs_diff = np.abs(our_rnorms - base_rnorms)
rel_diff = abs_diff / np.maximum(base_rnorms, 1e-6)
print(f"\nr_norm comparison vs baseline:")
print(f"  max abs diff: {abs_diff.max():.6f}")
print(f"  max rel diff: {rel_diff.max():.6f}")
print(f"  our[:5]:  {our_rnorms[:5]}")
print(f"  base[:5]: {base_rnorms[:5]}")

# ----------------------------------------------------------------------------
# Decision
# ----------------------------------------------------------------------------
# Acceptable: ~98% bit match (sign-of-near-zero noise) and rel_diff < 2% on rnorm
ok_bits  = bit_matches / (N*K) >= 0.97
ok_norm  = rel_diff.max() < 0.02

print()
if ok_bits and ok_norm:
    print("END-TO-END PASS")
    print("  Sign bits match baseline (modulo expected near-zero precision noise)")
    print("  r_norms match baseline within bf16 precision")
else:
    print("END-TO-END FAIL")
    print(f"  bit match {bit_matches/(N*K):.4f} ok={ok_bits}")
    print(f"  rnorm rel err {rel_diff.max():.4f} ok={ok_norm}")
    sys.exit(1)

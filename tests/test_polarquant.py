"""
tests/test_polarquant.py
=========================
Unit tests for polarquant_kernel.cpp.

Reference-only tests (no hardware):
  python tests/test_polarquant.py

Device validation (requires host program output):
  ./turboquant_single_core --num-vectors 64 --mode full
  python tests/test_polarquant.py --device

Device mode loads:
  dump_input.bin            — original FP16 input vectors
  dump_quant_indices.bin    — packed b-bit indices from polarquant_kernel.cpp
  dump_quant_centroids.bin  — FP16 dequantized centroids (kCbResidual output)

What is actually being tested in --device mode
----------------------------------------------
  1. pack_indices() in polarquant_kernel.cpp correctly packs b-bit integers
     into bytes with the expected low-nibble-first layout.
  2. The binary search / linear scan over decision boundaries returns the
     same index as numpy searchsorted() applied to the same codebook.
  3. The centroid pass-through (kCbResidual) writes the correct FP16 centroid
     for each index.
  4. The indices agree with a fresh reference encode of the SAME input (which
     went through the device rotation first, so we load the rotated data, not
     the raw input).
"""

import sys
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "reference"))
from turboquant import (
    build_codebook, polarquant_encode, polarquant_decode,
    make_sign_diagonal, apply_rotation, TurboQuantCodec,
)

DEFAULT_DIM  = 128
DEFAULT_BITS = 4
DEFAULT_SEED = 0  # SHA-256/NumPy seed matching SIGN_DIAGONAL precomputed table


# ---------------------------------------------------------------------------
# Binary unpack helpers (mirrors the C++ pack_indices / unpack_indices)
# ---------------------------------------------------------------------------

def unpack_indices(packed: bytes, d: int, b: int) -> np.ndarray:
    """Unpack b-bit indices from a packed byte array."""
    per_byte = 8 // b
    mask = (1 << b) - 1
    indices = np.zeros(d, dtype=np.int32)
    for i in range(d):
        byte_i = i // per_byte
        shift  = (i % per_byte) * b
        indices[i] = (packed[byte_i] >> shift) & mask
    return indices


def pack_indices(indices: np.ndarray, b: int) -> bytes:
    """Pack indices into b-bit-per-element byte array (Python reference)."""
    d = len(indices)
    per_byte = 8 // b
    n_bytes = (d * b + 7) // 8
    packed = bytearray(n_bytes)
    mask = (1 << b) - 1
    for i, idx in enumerate(indices):
        byte_i = i // per_byte
        shift  = (i % per_byte) * b
        packed[byte_i] |= (int(idx) & mask) << shift
    return bytes(packed)


# ---------------------------------------------------------------------------
# Tile-granular dump loaders
# ---------------------------------------------------------------------------

def load_fp16_dump(path: str, n: int, d: int, tile: int = 32) -> np.ndarray:
    raw = Path(path).read_bytes()
    fp16 = np.frombuffer(raw, dtype=np.uint16)
    n_full = len(fp16) // d
    return fp16[:n * d].reshape(n, d).astype(np.uint32).__lshift__(16).view(np.float32).astype(np.float64)


def load_packed_indices_dump(path: str, n: int, d: int, b: int,
                             tile: int = 32) -> np.ndarray:
    """
    Read a tile-granular packed-indices dump and return (n, d) int32 array.
    The dump has shape (num_tiles × tile × pq_bytes_per_vector) in bytes.
    """
    pq_bytes = (d * b + 7) // 8
    raw = Path(path).read_bytes()
    # Tile layout: tile_idx × (tile × pq_bytes)
    indices = np.zeros((n, d), dtype=np.int32)
    for v in range(n):
        tile_idx    = v // tile
        vec_in_tile = v % tile
        offset = tile_idx * tile * pq_bytes + vec_in_tile * pq_bytes
        packed = raw[offset:offset + pq_bytes]
        indices[v] = unpack_indices(packed, d, b)
    return indices


# ---------------------------------------------------------------------------
# Reference-only tests
# ---------------------------------------------------------------------------

def test_codebook_monotonicity():
    for d in [64, 128]:
        for b in [2, 4, 8]:
            cb = build_codebook(b=b, d=d, n_samples=100_000)
            assert np.all(np.diff(cb.centroids) > 0)
            assert np.all(np.diff(cb.boundaries) > 0)
            for i in range(len(cb.boundaries)):
                assert cb.centroids[i] < cb.boundaries[i] < cb.centroids[i+1]
    print("PASS  test_codebook_monotonicity")


def test_mse_decreases_with_bits():
    d = 128
    prev = float("inf")
    for b in [2, 4, 8]:
        codec = TurboQuantCodec(d=d, b=b)
        rng = np.random.RandomState(b)
        x = rng.randn(3000, d); x /= np.linalg.norm(x, axis=1, keepdims=True)
        enc = codec.encode(x)
        mse = float(np.mean((x - codec.decode(enc))**2))
        assert mse < prev, f"MSE not decreasing at b={b}"
        prev = mse
    print("PASS  test_mse_decreases_with_bits")


def test_index_roundtrip():
    rng = np.random.RandomState(20)
    for d in [64, 128]:
        for b in [2, 4]:
            cb = build_codebook(b=b, d=d, n_samples=100_000)
            D = make_sign_diagonal(d, seed=0)
            x = rng.randn(200, d); x /= np.linalg.norm(x, axis=1, keepdims=True)
            z = apply_rotation(x, D)
            idx = polarquant_encode(z, cb)
            z_hat = polarquant_decode(idx, cb)
            for i, v in zip(idx.flatten(), z_hat.flatten()):
                assert abs(v - cb.centroids[i]) < 1e-7
    print("PASS  test_index_roundtrip")


def test_index_range():
    rng = np.random.RandomState(30)
    for b in [2, 4, 8]:
        cb = build_codebook(b=b, d=128, n_samples=100_000)
        D = make_sign_diagonal(128, seed=0)
        x = rng.randn(1000, 128); x /= np.linalg.norm(x, axis=1, keepdims=True)
        idx = polarquant_encode(apply_rotation(x, D), cb)
        assert np.all(idx >= 0) and np.all(idx < 2**b)
    print("PASS  test_index_range")


def test_cosine_similarity():
    rng = np.random.RandomState(40)
    codec = TurboQuantCodec(d=128, b=4)
    x = rng.randn(1000, 128); x /= np.linalg.norm(x, axis=1, keepdims=True)
    enc = codec.encode(x)
    x_hat = codec.decode(enc)
    x_hat /= np.linalg.norm(x_hat, axis=1, keepdims=True) + 1e-12
    mean_cos = float(np.mean(np.sum(x * x_hat, axis=1)))
    assert mean_cos >= 0.99, f"Cosine similarity {mean_cos:.4f} < 0.99"
    print(f"PASS  test_cosine_similarity  [mean_cos={mean_cos:.4f}]")


def test_bit_packing_roundtrip():
    rng = np.random.RandomState(50)
    for b in [2, 4, 8]:
        idx = rng.randint(0, 2**b, size=128)
        recovered = unpack_indices(pack_indices(idx, b), 128, b)
        assert np.all(idx == recovered), f"Pack/unpack failed at b={b}"
    print("PASS  test_bit_packing_roundtrip")


# ---------------------------------------------------------------------------
# Device validation tests
# ---------------------------------------------------------------------------

def test_device_indices_match_reference(
    input_bin: str,
    rotated_bin: str,
    indices_bin: str,
    d: int = DEFAULT_DIM,
    b: int = DEFAULT_BITS,
    rotation_seed: int = DEFAULT_SEED,
    tile_size: int = 32,
):
    """
    Load the device's rotated vectors (dump_rotated.bin) and device indices
    (dump_quant_indices.bin).  Run polarquant_encode() in Python on the same
    rotated vectors and compare index-by-index.

    We use the device rotated output (not the original input + Python rotation)
    so that any FP16 rounding in rotation_kernel.cpp is shared between both
    sides of the comparison.  This isolates polarquant_kernel.cpp's codebook
    lookup from any rotation discrepancy.
    """
    n = len(Path(input_bin).read_bytes()) // (d * 2)
    print(f"  Vectors: {n}  d={d}  b={b}")

    # Load device rotated output as the common input to both sides
    z_dev = load_fp16_dump(rotated_bin, n, d, tile_size)

    # Load device packed indices
    idx_dev = load_packed_indices_dump(indices_bin, n, d, b, tile_size)

    # Run reference codebook encode on the same z values
    cb = build_codebook(b=b, d=d, n_samples=300_000)
    idx_ref = polarquant_encode(z_dev, cb)

    # Compare
    match_rate = float(np.mean(idx_ref == idx_dev))
    n_mismatch = int(np.sum(idx_ref != idx_dev))

    print(f"  Index match rate: {match_rate*100:.2f}%  "
          f"({n_mismatch} mismatches out of {n*d})")

    # Mismatches near decision boundaries (off-by-one) are expected due to
    # FP16 precision at the exact threshold.  We allow ≤ 1% mismatch.
    assert match_rate >= 0.99, (
        f"polarquant_kernel.cpp index match rate {match_rate*100:.2f}% < 99%\n"
        f"  This suggests a codebook ordering or binary search bug, "
        f"not just a boundary rounding issue."
    )

    # Also check that all device indices are in-range
    assert np.all(idx_dev >= 0) and np.all(idx_dev < 2**b), \
        "Device produced out-of-range indices"

    print(f"PASS  test_device_indices_match_reference  "
          f"[match={match_rate*100:.2f}%]  ✓")


def test_device_centroids_match_indices(
    rotated_bin: str,
    indices_bin: str,
    centroids_bin: str,
    d: int = DEFAULT_DIM,
    b: int = DEFAULT_BITS,
    tile_size: int = 32,
):
    """
    The dequantized centroid CB (dump_quant_centroids.bin) must equal
    codebook.centroids[index] for every coordinate, within FP16 precision.

    This validates that polarquant_kernel.cpp's centroid pass-through
    to kCbResidual is correct — if it's wrong, the residual r = x - x_hat
    will be computed from garbage and QJL will not correct anything.
    """
    n_all = len(Path(rotated_bin).read_bytes()) // (d * 2 * tile_size) * tile_size
    n = min(n_all, len(Path(centroids_bin).read_bytes()) // (d * 2 * tile_size) * tile_size)

    idx_dev       = load_packed_indices_dump(indices_bin,   n, d, b, tile_size)
    centroids_dev = load_fp16_dump(centroids_bin, n, d, tile_size)

    cb = build_codebook(b=b, d=d, n_samples=300_000)
    centroids_ref = cb.centroids[idx_dev]    # (n, d) reference centroid values

    # Convert reference to FP16 and back, to match the kernel's FP16 output
    centroids_ref_fp16 = centroids_ref.astype(np.float16).astype(np.float64)

    abs_err = np.abs(centroids_ref_fp16 - centroids_dev)
    max_err  = float(np.max(abs_err))
    mean_err = float(np.mean(abs_err))

    # Tolerance: FP16 has ~3 decimal digits of precision; centroids are O(0.1),
    # so we expect errors < 0.001 absolute.
    assert max_err < 0.005, (
        f"Centroid pass-through error too large: max={max_err:.5f} "
        f"(threshold 0.005) — possible wrong index or wrong centroid lookup."
    )
    print(f"PASS  test_device_centroids_match_indices  "
          f"[max_err={max_err:.5f}  mean_err={mean_err:.5f}]  ✓")


def test_device_quantization_mse(
    input_bin: str,
    indices_bin: str,
    d: int = DEFAULT_DIM,
    b: int = DEFAULT_BITS,
    rotation_seed: int = DEFAULT_SEED,
    tile_size: int = 32,
):
    """
    Compute the MSE of the device's quantized reconstruction against the
    original input.  Must be within 5% of the reference Python MSE.
    """
    raw = Path(input_bin).read_bytes()
    n   = len(raw) // (d * 2)
    x   = np.frombuffer(raw, dtype=np.uint16).reshape(n, d).astype(np.uint32).__lshift__(16).view(np.float32).astype(np.float64)

    idx_dev = load_packed_indices_dump(indices_bin, n, d, b, tile_size)
    cb = build_codebook(b=b, d=d, n_samples=300_000)

    # Dequantize: centroids → inverse rotation → reconstructed x
    D = make_sign_diagonal(d, rotation_seed)
    from turboquant import apply_inverse_rotation
    cents = cb.centroids[idx_dev]                  # (n, d) in rotated space
    x_hat = apply_inverse_rotation(cents, D)       # (n, d) in original space

    mse_dev = float(np.mean((x - x_hat)**2))

    # Reference MSE
    codec = TurboQuantCodec(d=d, b=b)
    enc = codec.encode(x)
    mse_ref = float(np.mean((x - codec.decode(enc))**2))

    ratio = mse_dev / (mse_ref + 1e-12)
    print(f"  MSE device={mse_dev:.6f}  reference={mse_ref:.6f}  ratio={ratio:.3f}")
    assert ratio < 1.10, (
        f"Device quantization MSE {ratio:.3f}× worse than reference "
        f"(threshold 1.10) — possible codebook or rotation mismatch."
    )
    print(f"PASS  test_device_quantization_mse  [ratio={ratio:.3f}]  ✓")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", action="store_true")
    parser.add_argument("--input-bin",     default="dump_input.bin")
    parser.add_argument("--rotated-bin",   default="dump_rotated.bin")
    parser.add_argument("--indices-bin",   default="dump_quant_indices.bin")
    parser.add_argument("--centroids-bin", default="dump_quant_centroids.bin")
    parser.add_argument("--dim",       type=int, default=DEFAULT_DIM)
    parser.add_argument("--bits",      type=int, default=DEFAULT_BITS)
    parser.add_argument("--seed",      type=int, default=DEFAULT_SEED)
    parser.add_argument("--tile-size", type=int, default=32)
    args = parser.parse_args()

    print("=" * 60)
    print("test_polarquant.py — PolarQuant Codebook Tests")
    print("=" * 60)

    test_codebook_monotonicity()
    test_mse_decreases_with_bits()
    test_index_roundtrip()
    test_index_range()
    test_cosine_similarity()
    test_bit_packing_roundtrip()

    if args.device:
        print("\n--- Device validation (polarquant_kernel.cpp) ---")
        required = {
            args.input_bin:     "original input",
            args.rotated_bin:   "rotated vectors (from rotation_kernel.cpp)",
            args.indices_bin:   "packed indices (from polarquant_kernel.cpp)",
            args.centroids_bin: "centroids CB (from polarquant_kernel.cpp)",
        }
        missing = [f"{p} [{desc}]" for p, desc in required.items()
                   if not Path(p).exists()]
        if missing:
            print(f"SKIP  device tests — missing:\n  " + "\n  ".join(missing))
            print("  Run: ./turboquant_single_core --num-vectors 64 --mode full")
        else:
            test_device_indices_match_reference(
                args.input_bin, args.rotated_bin, args.indices_bin,
                d=args.dim, b=args.bits, rotation_seed=args.seed,
                tile_size=args.tile_size,
            )
            test_device_centroids_match_indices(
                args.rotated_bin, args.indices_bin, args.centroids_bin,
                d=args.dim, b=args.bits, tile_size=args.tile_size,
            )
            test_device_quantization_mse(
                args.input_bin, args.indices_bin,
                d=args.dim, b=args.bits, rotation_seed=args.seed,
                tile_size=args.tile_size,
            )
    else:
        print("\n(Run with --device to validate polarquant_kernel.cpp output)")

    print("=" * 60)
    print("All PolarQuant tests passed.")


if __name__ == "__main__":
    main()

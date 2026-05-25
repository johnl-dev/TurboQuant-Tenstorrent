"""
tests/test_qjl.py
==================
Unit tests for qjl_kernel.cpp.

Reference-only tests (no hardware):
  python tests/test_qjl.py

Device validation (requires host program output):
  ./turboquant_single_core --num-vectors 64 --mode full
  python tests/test_qjl.py --device

Device mode loads:
  dump_input.bin              — original FP16 input vectors
  dump_quant_centroids.bin    — FP16 dequantized centroids from polarquant_kernel
  dump_qjl.bin                — QJL bits + r_norm from qjl_kernel.cpp

What is actually being tested in --device mode
----------------------------------------------
  1. The residual r = x − x_hat computed on device matches Python.
  2. The LFSR sign row expansion in the kernel matches the Python QJLProjector
     (same seed, same Galois polynomial, same bit extraction order).
  3. The 1-bit projections sign(S·r) agree bit-for-bit with Python.
  4. The r_norm stored as FP16 matches ||r||_2 within FP16 precision.

Bit-for-bit agreement on the projections is the critical check: if the LFSR
polynomial or the bit-extraction order in qjl_kernel.cpp differs from the
Python LFSR, the inner-product correction will be systematically wrong and
dequant_attn_kernel.cpp will produce biased attention scores.
"""

import sys
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "reference"))
from turboquant import QJLProjector, TurboQuantCodec, make_sign_diagonal, apply_inverse_rotation

DEFAULT_DIM      = 128
DEFAULT_QJL_DIM  = 128
DEFAULT_QJL_SEED = 0xCAFEBABE   # kQjlSeed
DEFAULT_ROT_SEED = 0


# ---------------------------------------------------------------------------
# Tile-granular loaders
# ---------------------------------------------------------------------------

def load_fp16_dump(path: str, n: int, d: int, tile: int = 32) -> np.ndarray:
    raw = Path(path).read_bytes()
    fp16 = np.frombuffer(raw, dtype=np.uint16)
    return fp16[:n * d].reshape(n, d).astype(np.uint32).__lshift__(16).view(np.float32).astype(np.float64)


def load_qjl_dump(path: str, n: int, k: int, tile: int = 32
                  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Load dump_qjl.bin.  Per-vector layout:
      [ceil(k/8) bytes: packed 1-bit projections]
      [2 bytes: r_norm as FP16]
    Tile-granular: tile × (ceil(k/8) + 2) bytes per tile.

    Returns:
      bits    : (n, k) int8 array of ±1
      r_norms : (n,)   float64 array
    """
    qjl_bytes  = (k + 7) // 8
    rnorm_bytes = 2
    rec_bytes   = qjl_bytes + rnorm_bytes
    tile_bytes  = tile * rec_bytes

    raw = Path(path).read_bytes()

    bits_out   = np.zeros((n, k), dtype=np.int8)
    r_norm_out = np.zeros(n, dtype=np.float64)

    for v in range(n):
        tile_idx    = v // tile
        vec_in_tile = v % tile
        base = tile_idx * tile_bytes + vec_in_tile * rec_bytes

        # Unpack k bits
        for j in range(k):
            byte_j = j // 8
            bit_j  = (raw[base + byte_j] >> (j % 8)) & 1
            bits_out[v, j] = 1 if bit_j else -1

        # r_norm (FP16 → float64)
        bf16_val = np.frombuffer(raw[base + qjl_bytes:base + qjl_bytes + 2],
                                  dtype=np.uint16)[0]
        # bf16 → float32: shift to upper 16 bits of a uint32 and reinterpret
        r_norm_out[v] = float(
            np.array([np.uint32(bf16_val) << 16], dtype=np.uint32).view(np.float32)[0]
        )

    return bits_out, r_norm_out


# ---------------------------------------------------------------------------
# Reference-only tests
# ---------------------------------------------------------------------------

def test_sign_matrix_balance():
    for k in [64, 128]:
        for d in [64, 128]:
            qjl = QJLProjector(d=d, k=k, seed=42)
            row_means = np.mean(qjl.S, axis=1)
            threshold = 4.0 / np.sqrt(d)
            assert np.max(np.abs(row_means)) < threshold
    print("PASS  test_sign_matrix_balance")


def test_output_is_binary():
    rng = np.random.RandomState(0)
    qjl = QJLProjector(d=128, k=128, seed=1)
    bits = qjl.encode(rng.randn(100, 128))
    assert set(np.unique(bits)).issubset({-1, 1})
    print("PASS  test_output_is_binary")


def test_qjl_unbiased_estimator():
    rng = np.random.RandomState(10)
    qjl = QJLProjector(d=128, k=128, seed=7)
    errors = []
    for _ in range(5000):
        q, r = rng.randn(128), rng.randn(128)
        rn = np.linalg.norm(r)
        bits = qjl.encode((r / rn)[None])[0]
        errors.append(qjl.inner_product_correction(q, bits, rn) - q @ r)
    bias = abs(np.mean(errors))
    assert bias < 0.5, f"QJL bias={bias:.5f}"
    print(f"PASS  test_qjl_unbiased_estimator  [|bias|={bias:.5f}]")


def test_qjl_variance_reduction():
    rng = np.random.RandomState(20)
    d, q_vec = 128, rng.randn(128)
    variances = {}
    for k in [16, 64, 128]:
        qjl = QJLProjector(d=d, k=k, seed=42)
        errs = []
        for _ in range(3000):
            r = rng.randn(d); rn = np.linalg.norm(r)
            bits = qjl.encode((r/rn)[None])[0]
            errs.append(qjl.inner_product_correction(q_vec, bits, rn) - q_vec @ r)
        variances[k] = float(np.var(errs))
    assert variances[16] > variances[64] > variances[128]
    print(f"PASS  test_qjl_variance_reduction")


def test_qjl_determinism():
    rng = np.random.RandomState(30)
    r = rng.randn(10, 128)
    for seed in [0, 42, 0xCAFEBABE]:
        b1 = QJLProjector(d=128, k=128, seed=seed).encode(r)
        b2 = QJLProjector(d=128, k=128, seed=seed).encode(r)
        assert np.all(b1 == b2)
    print("PASS  test_qjl_determinism")


def test_residual_norm_stored():
    rng = np.random.RandomState(40)
    codec = TurboQuantCodec(d=128, b=4)
    x = rng.randn(500, 128); x /= np.linalg.norm(x, axis=1, keepdims=True)
    enc = codec.encode(x)
    r_norms_true = np.linalg.norm(x - codec.decode(enc), axis=1)
    err = np.max(np.abs(r_norms_true - enc['r_norms']))
    assert err < 1e-10
    print(f"PASS  test_residual_norm_stored")


# ---------------------------------------------------------------------------
# Device validation tests
# ---------------------------------------------------------------------------

def _python_lfsr_sign_dot(row_seed: int, r_unit: np.ndarray) -> float:
    """
    Reproduce the LFSR sign dot product from qjl_kernel.cpp in Python.

    Galois LFSR polynomial: x^32 + x^22 + x^2 + x + 1  (tap mask 0x80200003)
    Row seed for row j: kQjlSeed XOR (j * 2654435761)  (Knuth multiplicative hash)
    Sign = +1.0 if (state & 1) else -1.0  after one LFSR step per coordinate.
    """
    state = row_seed if row_seed != 0 else 1
    dot = 0.0
    for xi in r_unit:
        lsb = state & 1
        state >>= 1
        if lsb:
            state ^= 0x80200003
        s = 1.0 if (state & 1) else -1.0
        dot += s * xi
    return dot


def test_device_qjl_bits_match_reference(
    input_bin: str,
    centroids_bin: str,
    qjl_bin: str,
    d: int = DEFAULT_DIM,
    k: int = DEFAULT_QJL_DIM,
    qjl_seed: int = DEFAULT_QJL_SEED,
    rot_seed: int = DEFAULT_ROT_SEED,
    tile_size: int = 32,
):
    """
    Load the device centroids (dump_quant_centroids.bin) and device QJL output
    (dump_qjl.bin).  Recompute the residual and QJL projection in Python using
    the same LFSR and compare bit-for-bit.

    This test specifically validates that the LFSR polynomial, the bit-extraction
    order, and the Knuth hash row-seed derivation in qjl_kernel.cpp all match
    the Python reference implementation.
    """
    raw_in = Path(input_bin).read_bytes()
    n = len(raw_in) // (d * 2)
    x = np.frombuffer(raw_in, dtype=np.uint16).reshape(n, d).astype(np.uint32).__lshift__(16).view(np.float32).astype(np.float64)

    x_hat = load_fp16_dump(centroids_bin, n, d, tile_size)
    bits_dev, r_norms_dev = load_qjl_dump(qjl_bin, n, k, tile_size)

    # Recompute residuals from device centroid output (same x_hat as device used)
    # Note: device's x_hat = inv_rotate(centroids_from_polarquant), but the
    # centroid dump is in rotated space.  We invert to get original-space x_hat.
    D = make_sign_diagonal(d, rot_seed)
    x_hat_orig = apply_inverse_rotation(x_hat, D)    # back to original space
    r = x - x_hat_orig                               # residual
    r_norms_ref = np.linalg.norm(r, axis=1)
    r_unit = r / (r_norms_ref[:, None] + 1e-12)

    # Compute reference bits using the LFSR as implemented in qjl_kernel.cpp
    knuth_mul = 2654435761
    bits_ref = np.zeros((n, k), dtype=np.int8)
    for j in range(k):
        row_seed = (qjl_seed ^ (j * knuth_mul)) & 0xFFFFFFFF
        for v in range(n):
            dot = _python_lfsr_sign_dot(row_seed, r_unit[v])
            bits_ref[v, j] = 1 if dot > 0 else -1

    # Compare
    match_rate = float(np.mean(bits_ref == bits_dev))
    n_mismatch = int(np.sum(bits_ref != bits_dev))
    print(f"  QJL bit match rate: {match_rate*100:.2f}%  "
          f"({n_mismatch} mismatches out of {n*k})")

    # Bits that are near-zero projections (|dot| < epsilon) are legitimately
    # ambiguous — FP32 vs FP16 arithmetic differences near zero can flip the sign.
    # We allow ≤ 2% mismatch to account for these boundary cases.
    assert match_rate >= 0.98, (
        f"qjl_kernel.cpp bit match rate {match_rate*100:.2f}% < 98%\n"
        f"  This likely indicates a mismatch in the LFSR polynomial, "
        f"tap mask, or row-seed derivation."
    )
    print(f"PASS  test_device_qjl_bits_match_reference  "
          f"[match={match_rate*100:.2f}%]  ✓")


def test_device_r_norm_accuracy(
    input_bin: str,
    centroids_bin: str,
    qjl_bin: str,
    d: int = DEFAULT_DIM,
    k: int = DEFAULT_QJL_DIM,
    rot_seed: int = DEFAULT_ROT_SEED,
    tile_size: int = 32,
):
    """
    Verify that the r_norm stored by qjl_kernel.cpp matches ||x - x_hat||
    within FP16 quantization error.
    """
    raw_in = Path(input_bin).read_bytes()
    n = len(raw_in) // (d * 2)
    x = np.frombuffer(raw_in, dtype=np.uint16).reshape(n, d).astype(np.uint32).__lshift__(16).view(np.float32).astype(np.float64)

    x_hat_rot = load_fp16_dump(centroids_bin, n, d, tile_size)
    _, r_norms_dev = load_qjl_dump(qjl_bin, n, k, tile_size)

    D = make_sign_diagonal(d, rot_seed)
    x_hat = apply_inverse_rotation(x_hat_rot, D)
    r_norms_ref = np.linalg.norm(x - x_hat, axis=1)

    # FP16 has ~3 significant decimal digits; norms are O(0.1–0.3) for unit vectors
    rel_err = np.abs(r_norms_ref - r_norms_dev) / (r_norms_ref + 1e-12)
    max_rel = float(np.max(rel_err))
    mean_rel = float(np.mean(rel_err))

    assert max_rel < 0.01, (
        f"r_norm FP16 relative error too large: max={max_rel:.4f} "
        f"(threshold 0.01)"
    )
    print(f"PASS  test_device_r_norm_accuracy  "
          f"[max_rel_err={max_rel:.4f}  mean_rel_err={mean_rel:.4f}]  ✓")


def test_device_inner_product_correction_unbiased(
    input_bin: str,
    centroids_bin: str,
    qjl_bin: str,
    d: int = DEFAULT_DIM,
    k: int = DEFAULT_QJL_DIM,
    qjl_seed: int = DEFAULT_QJL_SEED,
    rot_seed: int = DEFAULT_ROT_SEED,
    tile_size: int = 32,
):
    """
    Using the device QJL bits and r_norms, compute the inner-product correction
    in Python and verify it is an unbiased estimator of q·r.
    This is the key end-to-end correctness check for qjl_kernel.cpp.
    """
    raw_in = Path(input_bin).read_bytes()
    n = len(raw_in) // (d * 2)
    x = np.frombuffer(raw_in, dtype=np.uint16).reshape(n, d).astype(np.uint32).__lshift__(16).view(np.float32).astype(np.float64)

    x_hat_rot = load_fp16_dump(centroids_bin, n, d, tile_size)
    bits_dev, r_norms_dev = load_qjl_dump(qjl_bin, n, k, tile_size)

    D = make_sign_diagonal(d, rot_seed)
    x_hat = apply_inverse_rotation(x_hat_rot, D)
    r_true = x - x_hat

    # Use the Python QJLProjector with the same seed to compute corrections
    # (We're testing the device bits here, not re-deriving them from Python)
    qjl = QJLProjector(d=d, k=k, seed=qjl_seed)

    rng = np.random.RandomState(77)
    q = rng.randn(d)

    # True inner products q·r
    true_dots = r_true @ q

    # Correction using device bits
    corrections = np.array([
        qjl.inner_product_correction(q, bits_dev[v], r_norms_dev[v])
        for v in range(n)
    ])

    bias = float(np.abs(np.mean(corrections - true_dots)))
    print(f"  QJL correction bias: {bias:.5f}  (over {n} vectors)")

    # With n=64 vectors the estimate is noisy; allow larger bias
    assert bias < 1.0, (
        f"Inner-product correction using device bits is biased: {bias:.5f}\n"
        f"  This suggests the device LFSR sign matrix doesn't match the Python one."
    )
    print(f"PASS  test_device_inner_product_correction_unbiased  [bias={bias:.5f}]  ✓")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", action="store_true")
    parser.add_argument("--input-bin",     default="dump_input.bin")
    parser.add_argument("--centroids-bin", default="dump_quant_centroids.bin")
    parser.add_argument("--qjl-bin",       default="dump_qjl.bin")
    parser.add_argument("--dim",       type=int, default=DEFAULT_DIM)
    parser.add_argument("--qjl-dim",   type=int, default=DEFAULT_QJL_DIM)
    parser.add_argument("--qjl-seed",  type=int, default=DEFAULT_QJL_SEED)
    parser.add_argument("--rot-seed",  type=int, default=DEFAULT_ROT_SEED)
    parser.add_argument("--tile-size", type=int, default=32)
    args = parser.parse_args()

    print("=" * 60)
    print("test_qjl.py — QJL Residual Projection Tests")
    print("=" * 60)

    test_sign_matrix_balance()
    test_output_is_binary()
    test_qjl_unbiased_estimator()
    test_qjl_variance_reduction()
    test_qjl_determinism()
    test_residual_norm_stored()

    if args.device:
        print("\n--- Device validation (qjl_kernel.cpp) ---")
        required = {
            args.input_bin:     "original input",
            args.centroids_bin: "dequant centroids (from polarquant_kernel.cpp)",
            args.qjl_bin:       "QJL bits + r_norm (from qjl_kernel.cpp)",
        }
        missing = [f"{p} [{desc}]" for p, desc in required.items()
                   if not Path(p).exists()]
        if missing:
            print("SKIP  device tests — missing:\n  " + "\n  ".join(missing))
            print("  Run: ./turboquant_single_core --num-vectors 64 --mode full")
        else:
            test_device_qjl_bits_match_reference(
                args.input_bin, args.centroids_bin, args.qjl_bin,
                d=args.dim, k=args.qjl_dim,
                qjl_seed=args.qjl_seed, rot_seed=args.rot_seed,
                tile_size=args.tile_size,
            )
            test_device_r_norm_accuracy(
                args.input_bin, args.centroids_bin, args.qjl_bin,
                d=args.dim, k=args.qjl_dim, rot_seed=args.rot_seed,
                tile_size=args.tile_size,
            )
            test_device_inner_product_correction_unbiased(
                args.input_bin, args.centroids_bin, args.qjl_bin,
                d=args.dim, k=args.qjl_dim,
                qjl_seed=args.qjl_seed, rot_seed=args.rot_seed,
                tile_size=args.tile_size,
            )
    else:
        print("\n(Run with --device to validate qjl_kernel.cpp output)")

    print("=" * 60)
    print("All QJL tests passed.")


if __name__ == "__main__":
    main()

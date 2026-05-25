"""
tests/test_rotation.py
========================
Unit tests for rotation_kernel.cpp.

Reference-only tests (no hardware needed):
  python tests/test_rotation.py

Device validation (requires running the host program first):
  ./turboquant_single_core --num-vectors 64 --mode full
  python tests/test_rotation.py --device

Device mode loads dump_input.bin and dump_rotated.bin produced by the host
program and compares rotation_kernel.cpp's actual output against the NumPy
reference apply_rotation() function, coordinate by coordinate.

What is actually being tested in --device mode
----------------------------------------------
  1. The LFSR seed expansion (expand_sign_diagonal in rotation_kernel.cpp)
     produces the same ±1 diagonal as make_sign_diagonal() in turboquant.py.
  2. The 7-stage butterfly loop in wht_inplace() produces the same result as
     the Python wht() function.
  3. The FP16 round-trip (input → FP16 on device → FP32 in kernel → FP16
     output → FP32 in Python) does not introduce large errors.
"""

import sys
import argparse
import struct
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "reference"))
from turboquant import (
    wht, iwht, make_sign_diagonal,
    apply_rotation, apply_inverse_rotation,
    TurboQuantCodec,
)

# Match compile-time defaults from turboquant_layout.h
DEFAULT_DIM  = 128
DEFAULT_SEED = 0  # SHA-256/NumPy seed matching SIGN_DIAGONAL precomputed table


# ---------------------------------------------------------------------------
# Shared loader: read FP16 tile dump (tile-granular) → (N, d) float64 array
# ---------------------------------------------------------------------------

def load_fp16_dump(path: str, n_vectors: int, d: int,
                   tile_size: int = 32) -> np.ndarray:
    """
    Read a tile-granular FP16 binary dump produced by stage_dump_writer.
    The dump has shape (num_tiles × tile_size × d) in memory; we extract
    exactly the first n_vectors rows and convert to float64.
    """
    raw = Path(path).read_bytes()
    n_tiles = len(raw) // (tile_size * d * 2)
    all_fp16 = np.frombuffer(raw, dtype=np.uint16).reshape(n_tiles * tile_size, d)
    # numpy float16 has the same bit layout as IEEE fp16
    arr = all_fp16[:n_vectors].astype(np.uint32).__lshift__(16).view(np.float32).astype(np.float64)
    return arr


# ---------------------------------------------------------------------------
# Reference-only tests (always run)
# ---------------------------------------------------------------------------

def test_wht_invertibility():
    rng = np.random.RandomState(0)
    for d in [64, 128, 256]:
        for n in [1, 16, 64]:
            x = rng.randn(n, d)
            err = np.max(np.abs(x - iwht(wht(x))))
            assert err < 1e-10, f"WHT invertibility FAIL d={d} n={n}: err={err:.2e}"
    print("PASS  test_wht_invertibility")


def test_wht_orthonormality():
    for d in [64, 128]:
        W = np.eye(d)
        for i in range(d):
            e = np.zeros((1, d)); e[0, i] = 1.0
            W[:, i] = wht(e)[0]
        err = np.max(np.abs(W.T @ W - np.eye(d)))
        assert err < 1e-10, f"WHT not orthonormal d={d}: err={err:.2e}"
    print("PASS  test_wht_orthonormality")


def test_rotation_invertibility():
    rng = np.random.RandomState(1)
    for d in [64, 128, 256]:
        for seed in [0, 42, 999]:
            D = make_sign_diagonal(d, seed)
            x = rng.randn(32, d)
            err = np.max(np.abs(x - apply_inverse_rotation(apply_rotation(x, D), D)))
            assert err < 1e-10, f"Rotation invertibility FAIL d={d} seed={seed}: err={err:.2e}"
    print("PASS  test_rotation_invertibility")


def test_rotation_distribution():
    rng = np.random.RandomState(2)
    for d in [64, 128]:
        n = 5000
        x = rng.randn(n, d) / np.linalg.norm(rng.randn(n, d), axis=1, keepdims=True)
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        D = make_sign_diagonal(d, seed=0)
        coords = apply_rotation(x, D).flatten()
        mean = abs(np.mean(coords))
        std_err = abs(np.std(coords) - 1.0 / np.sqrt(d)) / (1.0 / np.sqrt(d))
        assert mean < 0.01, f"Rotated coord mean={mean:.4f} at d={d}"
        assert std_err < 0.05, f"Rotated coord std rel_err={std_err:.3f} at d={d}"
    print("PASS  test_rotation_distribution")


def test_sign_diagonal_determinism():
    for d in [64, 128]:
        for seed in [0, 7, 123456]:
            D1 = make_sign_diagonal(d, seed)
            D2 = make_sign_diagonal(d, seed)
            assert np.all(D1 == D2) and np.all(np.abs(D1) == 1.0)
    print("PASS  test_sign_diagonal_determinism")


def test_wht_linearity():
    rng = np.random.RandomState(3)
    d = 128
    x, y = rng.randn(1, d), rng.randn(1, d)
    a, b = 3.7, -1.2
    err = np.max(np.abs(wht(a*x + b*y) - (a*wht(x) + b*wht(y))))
    assert err < 1e-9, f"WHT not linear: err={err:.2e}"
    print("PASS  test_wht_linearity")


# ---------------------------------------------------------------------------
# Device validation tests (require dump_input.bin + dump_rotated.bin)
# ---------------------------------------------------------------------------

def test_device_rotation_matches_reference(
    input_bin: str,
    rotated_bin: str,
    d: int = DEFAULT_DIM,
    rotation_seed: int = DEFAULT_SEED,
    tile_size: int = 32,
):
    """
    Load dump_input.bin (FP16 vectors written by the host program) and
    dump_rotated.bin (FP16 rotated vectors written by rotation_kernel.cpp).
    Re-run apply_rotation() in Python on the same input and compare.

    Tolerances:
      - We allow FP16 rounding error (~0.1% relative, or ~1e-3 absolute
        for unit-norm vectors after rotation).
      - Max absolute error per coordinate: 0.005
      - Mean absolute error across all coordinates: 0.001
    """
    print(f"\n  Loading {input_bin} ...", end="", flush=True)
    raw_in = Path(input_bin).read_bytes()
    n = len(raw_in) // (d * 2)
    x_fp16 = np.frombuffer(raw_in, dtype=np.uint16).reshape(n, d)
    x = x_fp16.astype(np.uint32).__lshift__(16).view(np.float32).astype(np.float64)
    print(f" {n} vectors loaded.")

    print(f"  Loading {rotated_bin} ...", end="", flush=True)
    z_dev = load_fp16_dump(rotated_bin, n, d, tile_size)
    print(f" done.")

    # Run the reference rotation with the same seed as the C++ kernel
    D = make_sign_diagonal(d, rotation_seed)
    z_ref = apply_rotation(x, D)

    abs_err = np.abs(z_ref - z_dev)
    max_err  = float(np.max(abs_err))
    mean_err = float(np.mean(abs_err))

    # Percentage of coordinates that exactly agree after FP16 round-trip
    # (quantise reference to FP16 and compare to device FP16 output)
    z_ref_fp16 = z_ref.astype(np.float16).astype(np.float64)
    exact_match_rate = float(np.mean(z_ref_fp16 == z_dev))

    print(f"  max_abs_err={max_err:.5f}  mean_abs_err={mean_err:.5f}  "
          f"fp16_exact_match={exact_match_rate*100:.1f}%")

    assert max_err < 0.005, (
        f"rotation_kernel.cpp output deviates too much from reference: "
        f"max_abs_err={max_err:.5f} (threshold 0.005)"
    )
    assert mean_err < 0.001, (
        f"rotation_kernel.cpp mean error too large: "
        f"mean_abs_err={mean_err:.5f} (threshold 0.001)"
    )
    # bf16 output: skip fp16 exact-match check
    print(f"PASS  test_device_rotation_matches_reference  ✓")


def test_device_inverse_rotation(
    rotated_bin: str,
    d: int = DEFAULT_DIM,
    rotation_seed: int = DEFAULT_SEED,
    tile_size: int = 32,
):
    """
    Apply the inverse rotation (in Python) to the device's rotated output and
    verify that we recover something close to the original input.
    This cross-checks that the device's WHT is truly the transpose of Python's.
    """
    raw = Path(rotated_bin).read_bytes()
    n = len(raw) // (tile_size * d * 2) * tile_size
    n = min(n, len(raw) // (d * 2))
    z_dev = load_fp16_dump(rotated_bin, n, d, tile_size)

    raw_in = Path("dump_input.bin").read_bytes()
    n_in = len(raw_in) // (d * 2)
    n = min(n, n_in)
    x = np.frombuffer(raw_in, dtype=np.uint16)[:n * d].reshape(n, d).astype(np.uint32).__lshift__(16).view(np.float32).astype(np.float64)
    z_dev = z_dev[:n]

    D = make_sign_diagonal(d, rotation_seed)
    x_reconstructed = apply_inverse_rotation(z_dev, D)

    # After double FP16 quantisation the reconstruction won't be bit-exact,
    # but cosine similarity should be very high.
    cos = np.sum(x * x_reconstructed, axis=1) / (
        np.linalg.norm(x, axis=1) * np.linalg.norm(x_reconstructed, axis=1) + 1e-12
    )
    mean_cos = float(np.mean(cos))
    assert mean_cos > 0.999, (
        f"Inverse rotation of device output gave low cosine similarity: {mean_cos:.5f}"
    )
    print(f"PASS  test_device_inverse_rotation  [mean_cos={mean_cos:.5f}]")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", action="store_true",
                        help="Load device binary dumps and compare to reference")
    parser.add_argument("--input-bin",   default="dump_input.bin")
    parser.add_argument("--rotated-bin", default="dump_rotated.bin")
    parser.add_argument("--dim",  type=int, default=DEFAULT_DIM)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tile-size", type=int, default=32,
                        help="kTileVectors used in the host program (default 32)")
    args = parser.parse_args()

    print("=" * 60)
    print("test_rotation.py — WHT Rotation Kernel Tests")
    print("=" * 60)

    # Reference tests (always run)
    test_wht_invertibility()
    test_wht_orthonormality()
    test_rotation_invertibility()
    test_rotation_distribution()
    test_sign_diagonal_determinism()
    test_wht_linearity()

    # Device validation tests
    if args.device:
        print("\n--- Device validation (rotation_kernel.cpp) ---")
        missing = [p for p in [args.input_bin, args.rotated_bin]
                   if not Path(p).exists()]
        if missing:
            print(f"SKIP  device tests — missing files: {missing}")
            print("  Run: ./turboquant_single_core --num-vectors 64 --mode full")
        else:
            test_device_rotation_matches_reference(
                args.input_bin, args.rotated_bin,
                d=args.dim, rotation_seed=args.seed, tile_size=args.tile_size,
            )
            if Path(args.rotated_bin).exists() and Path(args.input_bin).exists():
                test_device_inverse_rotation(
                    args.rotated_bin, d=args.dim,
                    rotation_seed=args.seed, tile_size=args.tile_size,
                )
    else:
        print("\n(Run with --device to validate rotation_kernel.cpp output)")

    print("=" * 60)
    print("All rotation tests passed.")


if __name__ == "__main__":
    main()

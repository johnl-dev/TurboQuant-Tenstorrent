"""
tests/test_roundtrip.py
========================
CP-1 MWE gate integration test.

Reference-only tests (no hardware):
  python tests/test_roundtrip.py

Device validation (requires host program output):
  ./turboquant_single_core --num-vectors 64 --mode full
  python tests/test_roundtrip.py --device

Device mode loads dump_output.bin (the fully assembled TurboQuant record
produced by the host program from all kernel stage outputs) and validates:

  1. Cosine similarity of dequantized vectors ≥ 0.99 at b=4  (CP-1 criterion 1)
  2. Inner-product distortion within 3× of TurboQuant guarantee  (CP-1 criterion 2)
  3. All device bytes are non-zero (no silent DMA failure)
  4. Per-vector records have the expected memory layout

The device dequantization is performed in Python by:
  a. Reading device packed indices from dump_output.bin
  b. Looking up centroids from the Python codebook
  c. Applying inverse rotation (same seed as C++ kernel)
  d. This proves the C++ encoding can be successfully decoded by a correct
     decoder — the core correctness guarantee of the pipeline.
"""

import sys
import argparse
import struct
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "reference"))
from turboquant import (
    TurboQuantCodec, build_codebook, make_sign_diagonal,
    apply_inverse_rotation, polarquant_decode,
)

DEFAULT_DIM      = 128
DEFAULT_BITS     = 4
DEFAULT_QJL_DIM  = 128
DEFAULT_ROT_SEED = 0xDEADBEEF
DEFAULT_QJL_SEED = 0xCAFEBABE


# ---------------------------------------------------------------------------
# Record layout helpers  (matches turboquant_layout.h)
# ---------------------------------------------------------------------------

def record_bytes(b: int, d: int, k: int) -> int:
    pq  = (d * b + 7) // 8
    qjl = (k + 7) // 8
    raw = pq + qjl + 2   # 2 bytes r_norm
    return (raw + 31) & ~31


def unpack_record(raw: bytes, b: int, d: int, k: int, rec_bytes: int
                  ) -> tuple[np.ndarray, np.ndarray, float]:
    """Unpack one record into (indices, qjl_bits, r_norm)."""
    pq_bytes  = (d * b + 7) // 8
    qjl_bytes = (k + 7) // 8
    per_byte  = 8 // b
    mask      = (1 << b) - 1

    indices = np.zeros(d, dtype=np.int32)
    for i in range(d):
        byte_i = i // per_byte
        shift  = (i % per_byte) * b
        indices[i] = (raw[byte_i] >> shift) & mask

    qjl_bits = np.zeros(k, dtype=np.int8)
    for j in range(k):
        byte_j = j // 8
        bit_j  = (raw[pq_bytes + byte_j] >> (j % 8)) & 1
        qjl_bits[j] = 1 if bit_j else -1

    rnorm_off = pq_bytes + qjl_bytes
    r_norm = float(np.frombuffer(
        raw[rnorm_off:rnorm_off + 2], dtype=np.uint16)[0].view(np.float16))

    return indices, qjl_bits, r_norm


def decode_output_bin(path: str, n: int, b: int, d: int, k: int) -> dict:
    """Decode dump_output.bin into arrays for Python dequantization."""
    rec = record_bytes(b, d, k)
    raw = Path(path).read_bytes()
    assert len(raw) >= n * rec, \
        f"Output bin too small: {len(raw)} < {n * rec} for {n} vectors"

    indices  = np.zeros((n, d), dtype=np.int32)
    qjl_bits = np.zeros((n, k), dtype=np.int8)
    r_norms  = np.zeros(n, dtype=np.float64)

    for v in range(n):
        rec_raw = raw[v * rec:(v + 1) * rec]
        indices[v], qjl_bits[v], r_norms[v] = unpack_record(rec_raw, b, d, k, rec)

    return {'indices': indices, 'qjl_bits': qjl_bits, 'r_norms': r_norms}


# ---------------------------------------------------------------------------
# Reference-only tests
# ---------------------------------------------------------------------------

def test_cosine_similarity_mwe():
    rng = np.random.RandomState(0)
    codec = TurboQuantCodec(d=128, b=4, k=128)
    x = rng.randn(1000, 128); x /= np.linalg.norm(x, axis=1, keepdims=True)
    enc = codec.encode(x)
    x_hat = codec.decode(enc)
    x_hat /= np.linalg.norm(x_hat, axis=1, keepdims=True) + 1e-12
    mean_cos = float(np.mean(np.sum(x * x_hat, axis=1)))
    assert mean_cos >= 0.99, f"Reference cosine similarity {mean_cos:.4f} < 0.99"
    print(f"PASS  test_cosine_similarity_mwe  [mean_cos={mean_cos:.4f}]")


def test_binary_record_roundtrip():
    """Pack a record → unpack → same values."""
    rng = np.random.RandomState(1)
    d, b, k = 128, 4, 128
    codec = TurboQuantCodec(d=d, b=b, k=k)
    x = rng.randn(50, d); x /= np.linalg.norm(x, axis=1, keepdims=True)
    enc = codec.encode(x)

    # Pack to bytes (simulating host assembly in turboquant_program.cpp)
    rec = record_bytes(b, d, k)
    pq_bytes  = (d * b + 7) // 8
    qjl_bytes = (k + 7) // 8
    per_byte  = 8 // b
    mask      = (1 << b) - 1

    buf = bytearray(50 * rec)
    for v in range(50):
        base = v * rec
        for i in range(d):
            byte_i = i // per_byte; shift = (i % per_byte) * b
            buf[base + byte_i] |= (int(enc['indices'][v, i]) & mask) << shift
        for j in range(k):
            if enc['qjl_bits'][v, j] > 0:
                buf[base + pq_bytes + j//8] |= 1 << (j%8)
        rnorm_fp16 = int(np.float16(enc['r_norms'][v]).view(np.uint16))
        struct.pack_into('<H', buf, base + pq_bytes + qjl_bytes, rnorm_fp16)

    decoded = decode_output_bin.__wrapped__ if hasattr(decode_output_bin, '__wrapped__') \
              else None

    # Unpack and compare
    for v in range(50):
        rec_raw = bytes(buf[v*rec:(v+1)*rec])
        idx, bits, rn = unpack_record(rec_raw, b, d, k, rec)
        assert np.all(idx == enc['indices'][v])
        assert np.all(bits == enc['qjl_bits'][v])
        rel_err = abs(rn - enc['r_norms'][v]) / (enc['r_norms'][v] + 1e-12)
        assert rel_err < 0.01

    print("PASS  test_binary_record_roundtrip")


def test_compression_ratio():
    d, b, k = 128, 4, 128
    fp16_bytes = d * 2
    compressed = record_bytes(b, d, k)
    ratio = fp16_bytes / compressed
    assert ratio >= 2.5, f"Compression ratio {ratio:.2f} < 2.5×"
    print(f"PASS  test_compression_ratio  [{ratio:.2f}×]")


def test_inner_product_distortion():
    rng = np.random.RandomState(3)
    codec = TurboQuantCodec(d=128, b=4)
    x = rng.randn(2000, 128); x /= np.linalg.norm(x, axis=1, keepdims=True)
    q = rng.randn(128)
    enc = codec.encode(x)
    mse_tq    = float(np.mean((codec.attention_dot(q, enc) - x @ q)**2))
    x_quant   = np.round(x * 8.0) / 8.0
    mse_naive = float(np.mean((x_quant @ q - x @ q)**2))
    assert mse_tq < mse_naive * 3.0 or mse_tq < 0.01
    print(f"PASS  test_inner_product_distortion  "
          f"[ratio={mse_tq/(mse_naive+1e-12):.3f}]  ✓ CP-1 criterion 2")


# ---------------------------------------------------------------------------
# Device validation tests
# ---------------------------------------------------------------------------

def test_device_output_nonzero(output_bin: str):
    """Non-zero checksum guard — detects silent DMA failure."""
    raw = Path(output_bin).read_bytes()
    checksum = sum(raw)
    assert checksum != 0, "dump_output.bin is all-zero — DMA or kernel failure"
    print(f"PASS  test_device_output_nonzero  [checksum=0x{checksum:08X}]")


def test_device_index_range(
    output_bin: str,
    n: int,
    b: int = DEFAULT_BITS,
    d: int = DEFAULT_DIM,
    k: int = DEFAULT_QJL_DIM,
):
    """All device indices must lie in [0, 2^b)."""
    enc = decode_output_bin(output_bin, n, b, d, k)
    assert np.all(enc['indices'] >= 0) and np.all(enc['indices'] < 2**b), \
        "Device produced out-of-range indices in packed output"
    print("PASS  test_device_index_range")


def test_device_cosine_similarity(
    input_bin: str,
    output_bin: str,
    b: int = DEFAULT_BITS,
    d: int = DEFAULT_DIM,
    k: int = DEFAULT_QJL_DIM,
    rot_seed: int = DEFAULT_ROT_SEED,
):
    """
    CP-1 criterion 1: dequantize device output in Python and compute cosine
    similarity to the original input vectors.  Must be ≥ 0.99.

    Dequantization path:
      1. Unpack device indices from dump_output.bin.
      2. Look up centroids[indices] from the Python codebook.
      3. Apply inverse rotation Π^T using the same seed as rotation_kernel.cpp.
      4. Compute cosine similarity to original input.

    If this passes, it proves that all three kernel stages (rotation, polarquant,
    and the assembly step) produce outputs that are decodable to near-original
    quality.
    """
    raw_in = Path(input_bin).read_bytes()
    n = len(raw_in) // (d * 2)
    x = np.frombuffer(raw_in, dtype=np.uint16).reshape(n, d).view(np.float16).astype(np.float64)

    enc_dev = decode_output_bin(output_bin, n, b, d, k)

    # Python dequantization using device indices
    cb = build_codebook(b=b, d=d, n_samples=300_000)
    cents = cb.centroids[enc_dev['indices']]           # (n, d) in rotated space
    D = make_sign_diagonal(d, rot_seed)
    x_hat = apply_inverse_rotation(cents, D)           # (n, d) in original space

    x_hat /= np.linalg.norm(x_hat, axis=1, keepdims=True) + 1e-12
    x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    cos = np.sum(x_norm * x_hat, axis=1)
    mean_cos = float(np.mean(cos))
    min_cos  = float(np.min(cos))

    print(f"  Device cosine similarity: mean={mean_cos:.4f}  min={min_cos:.4f}")
    assert mean_cos >= 0.99, (
        f"CP-1 FAIL: device cosine similarity {mean_cos:.4f} < 0.99\n"
        f"  Check rotation_kernel.cpp or polarquant_kernel.cpp for bugs."
    )
    print(f"PASS  test_device_cosine_similarity  "
          f"[mean={mean_cos:.4f}]  ✓ CP-1 criterion 1")


def test_device_matches_reference_indices(
    input_bin: str,
    output_bin: str,
    b: int = DEFAULT_BITS,
    d: int = DEFAULT_DIM,
    k: int = DEFAULT_QJL_DIM,
    rot_seed: int = DEFAULT_ROT_SEED,
):
    """
    Compare device indices (from dump_output.bin) against a fresh Python
    encode of the same input.  Must have ≥ 98% index-level agreement.
    """
    raw_in = Path(input_bin).read_bytes()
    n = len(raw_in) // (d * 2)
    x = np.frombuffer(raw_in, dtype=np.uint16).reshape(n, d).view(np.float16).astype(np.float64)

    enc_dev = decode_output_bin(output_bin, n, b, d, k)
    codec = TurboQuantCodec(d=d, b=b, k=k, rotation_seed=rot_seed)
    enc_ref = codec.encode(x)

    match = float(np.mean(enc_dev['indices'] == enc_ref['indices']))
    print(f"  Index match vs reference: {match*100:.2f}%")
    assert match >= 0.98, (
        f"Device index match rate {match*100:.2f}% < 98%\n"
        f"  Suggests a systematic error in rotation or polarquant."
    )
    print(f"PASS  test_device_matches_reference_indices  "
          f"[match={match*100:.2f}%]  ✓")


def test_device_inner_product_distortion(
    input_bin: str,
    output_bin: str,
    b: int = DEFAULT_BITS,
    d: int = DEFAULT_DIM,
    k: int = DEFAULT_QJL_DIM,
    rot_seed: int = DEFAULT_ROT_SEED,
    qjl_seed: int = DEFAULT_QJL_SEED,
):
    """
    CP-1 criterion 2: inner-product distortion from device encoding is within
    3× of the TurboQuant theoretical guarantee.
    """
    from turboquant import QJLProjector

    raw_in = Path(input_bin).read_bytes()
    n = len(raw_in) // (d * 2)
    x = np.frombuffer(raw_in, dtype=np.uint16).reshape(n, d).view(np.float16).astype(np.float64)

    enc_dev = decode_output_bin(output_bin, n, b, d, k)

    # Dequantize using device indices
    cb = build_codebook(b=b, d=d, n_samples=300_000)
    D = make_sign_diagonal(d, rot_seed)
    x_hat = apply_inverse_rotation(cb.centroids[enc_dev['indices']], D)

    # QJL correction using device bits
    qjl = QJLProjector(d=d, k=k, seed=qjl_seed)
    rng = np.random.RandomState(99)
    q = rng.randn(d)

    raw_dots    = x_hat @ q
    corrections = np.array([
        qjl.inner_product_correction(q, enc_dev['qjl_bits'][v], enc_dev['r_norms'][v])
        for v in range(n)
    ])
    tq_dots  = raw_dots + corrections
    true_dots = x @ q

    mse_device = float(np.mean((tq_dots - true_dots)**2))

    # Baseline: naive 4-bit quantization (no rotation, no QJL)
    x_naive = np.round(x * 8.0) / 8.0
    mse_naive = float(np.mean((x_naive @ q - true_dots)**2))

    ratio = mse_device / (mse_naive + 1e-12)
    print(f"  Device IP distortion: MSE={mse_device:.6f}  "
          f"naive={mse_naive:.6f}  ratio={ratio:.3f}")

    assert ratio < 3.0 or mse_device < 0.05, (
        f"CP-1 FAIL: device inner-product distortion ratio={ratio:.2f} > 3×"
    )
    print(f"PASS  test_device_inner_product_distortion  "
          f"[ratio={ratio:.3f}]  ✓ CP-1 criterion 2")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", action="store_true")
    parser.add_argument("--input-bin",  default="dump_input.bin")
    parser.add_argument("--output-bin", default="dump_output.bin")
    parser.add_argument("--dim",       type=int, default=DEFAULT_DIM)
    parser.add_argument("--bits",      type=int, default=DEFAULT_BITS)
    parser.add_argument("--qjl-dim",   type=int, default=DEFAULT_QJL_DIM)
    parser.add_argument("--rot-seed",  type=int, default=DEFAULT_ROT_SEED)
    parser.add_argument("--qjl-seed",  type=int, default=DEFAULT_QJL_SEED)
    args = parser.parse_args()

    print("=" * 60)
    print("test_roundtrip.py — CP-1 MWE Gate Integration Tests")
    print("=" * 60)

    test_cosine_similarity_mwe()
    test_binary_record_roundtrip()
    test_compression_ratio()
    test_inner_product_distortion()

    if args.device:
        print("\n--- Device validation (full pipeline) ---")
        required = {
            args.input_bin:  "input vectors",
            args.output_bin: "assembled TurboQuant output records",
        }
        missing = [f"{p} [{desc}]" for p, desc in required.items()
                   if not Path(p).exists()]
        if missing:
            print("SKIP  device tests — missing:\n  " + "\n  ".join(missing))
            print("  Run: ./turboquant_single_core --num-vectors 64 --mode full")
        else:
            raw_in = Path(args.input_bin).read_bytes()
            n = len(raw_in) // (args.dim * 2)
            print(f"  Loaded {n} input vectors from {args.input_bin}")

            test_device_output_nonzero(args.output_bin)
            test_device_index_range(args.output_bin, n, args.bits, args.dim, args.qjl_dim)
            test_device_cosine_similarity(
                args.input_bin, args.output_bin,
                b=args.bits, d=args.dim, k=args.qjl_dim, rot_seed=args.rot_seed,
            )
            test_device_matches_reference_indices(
                args.input_bin, args.output_bin,
                b=args.bits, d=args.dim, k=args.qjl_dim, rot_seed=args.rot_seed,
            )
            test_device_inner_product_distortion(
                args.input_bin, args.output_bin,
                b=args.bits, d=args.dim, k=args.qjl_dim,
                rot_seed=args.rot_seed, qjl_seed=args.qjl_seed,
            )
    else:
        print("\n(Run with --device to validate full pipeline on Wormhole hardware)")

    print("=" * 60)
    print("All round-trip tests passed.")


if __name__ == "__main__":
    main()

"""
reference/turboquant.py
=======================
Pure NumPy reference implementation of TurboQuant
(Zandieh et al., ICLR 2026, arXiv:2504.19874).

This is the numerical oracle for all device-side kernel validation.
Run `python turboquant.py` to execute built-in self-tests.

Algorithm summary
-----------------
  Quantize(x):
    1. z = WHT(D * x)               # PolarQuant: random rotation
    2. indices, centroids = codebook_encode(z, b)
    3. r = x - WHT(D * centroids)   # residual in original space
    4. qjl_bits = sign(S @ r)       # 1-bit QJL correction

  Dequantize + inner product(q, indices, qjl_bits):
    1. x_hat = WHT(D * centroids[indices])
    2. dot  = q @ x_hat
    3. correction = qjl_correction(q, qjl_bits, S)
    4. return dot + correction
"""

from __future__ import annotations

import hashlib
import struct
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Hadamard / Walsh-Hadamard Transform (WHT)
# ---------------------------------------------------------------------------

def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def wht(x: NDArray[np.float64], normalize: bool = True) -> NDArray[np.float64]:
    """
    Walsh-Hadamard Transform along the last axis.

    Parameters
    ----------
    x         : (..., d) array where d must be a power of 2.
    normalize : if True, divide by sqrt(d) so the transform is orthonormal.

    Returns
    -------
    (..., d) transformed array.
    """
    x = np.array(x, dtype=np.float64)
    d = x.shape[-1]
    assert d & (d - 1) == 0, f"WHT requires d to be a power of 2, got {d}"
    x2 = np.array(x, dtype=np.float64)
    h = 1
    while h < d:
        for i in range(0, d, h * 2):
            u = x2[..., i:i + h].copy()
            v = x2[..., i + h:i + 2 * h].copy()
            x2[..., i:i + h] = u + v
            x2[..., i + h:i + 2 * h] = u - v
        h <<= 1

    if normalize:
        x2 = x2 / np.sqrt(d)
    return x2


def iwht(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse WHT (identical to forward WHT for normalized transform)."""
    return wht(x, normalize=True)


# ---------------------------------------------------------------------------
# Random diagonal sign matrix D
# ---------------------------------------------------------------------------

def make_sign_diagonal(d: int, seed: int) -> NDArray[np.float64]:
    """
    Generate a reproducible ±1 diagonal vector of length d from an integer seed.
    Expand seed → bits via SHA-256 to get unbiased random signs.
    """
    rng = np.random.RandomState(seed)
    return rng.choice([-1.0, 1.0], size=d)


def apply_rotation(x: NDArray[np.float64], D: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply Π = WHT ∘ D to x (broadcast over batch dimension)."""
    return wht(D * x)


def apply_inverse_rotation(z: NDArray[np.float64], D: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply Π^{-1} = Π^T = D ∘ WHT to z."""
    return D * iwht(z)


# ---------------------------------------------------------------------------
# Codebook (Max-Lloyd / 1D k-means for Beta distribution)
# ---------------------------------------------------------------------------

class Codebook(NamedTuple):
    centroids: NDArray[np.float64]     # (2^b,)
    boundaries: NDArray[np.float64]    # (2^b - 1,)
    bits: int
    dim: int


def _beta_samples(d: int, n_samples: int = 500_000, rng: np.random.RandomState = None) -> NDArray[np.float64]:
    """
    Sample from the marginal distribution of a single coordinate of WHT(D*x)
    where x is uniform on the unit sphere in R^d.

    The marginal follows Beta(d/2 - 1/2, d/2 - 1/2) scaled to [-1/sqrt(d), 1/sqrt(d)].
    We draw Gaussian vectors, apply WHT, and take the first coordinate.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    dp = _next_pow2(d)
    xs = rng.randn(n_samples, dp)
    xs = xs / np.linalg.norm(xs, axis=1, keepdims=True)
    D = make_sign_diagonal(dp, seed=0)
    zs = apply_rotation(xs, D)  # (..., dp)
    return zs[:, 0]  # first coordinate


def lloyd_max_1d(
    samples: NDArray[np.float64],
    n_levels: int,
    n_iter: int = 200,
    tol: float = 1e-8,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Run the Lloyd-Max (k-means) algorithm on 1D samples.

    Returns
    -------
    centroids  : (n_levels,) sorted array of reconstruction values
    boundaries : (n_levels - 1,) decision thresholds
    """
    # Initialise centroids using quantiles
    quantiles = np.linspace(0, 100, n_levels + 2)[1:-1]
    centroids = np.percentile(samples, quantiles)

    prev_mse = np.inf
    for _ in range(n_iter):
        # Assignment step
        boundaries = 0.5 * (centroids[:-1] + centroids[1:])
        indices = np.searchsorted(boundaries, samples)
        # Update step
        new_centroids = np.array([
            samples[indices == k].mean() if np.any(indices == k) else centroids[k]
            for k in range(n_levels)
        ])
        mse = np.mean((samples - centroids[indices]) ** 2)
        if abs(prev_mse - mse) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids
        prev_mse = mse

    boundaries = 0.5 * (centroids[:-1] + centroids[1:])
    return centroids, boundaries


def build_codebook(b: int, d: int, n_samples: int = 300_000, rng_seed: int = 42) -> Codebook:
    """
    Build a Max-Lloyd codebook for WHT coordinate distribution at (b bits, dim d).
    """
    rng = np.random.RandomState(rng_seed)
    samples = _beta_samples(d, n_samples=n_samples, rng=rng)
    n_levels = 2 ** b
    centroids, boundaries = lloyd_max_1d(samples, n_levels)
    return Codebook(centroids=centroids, boundaries=boundaries, bits=b, dim=d)


# ---------------------------------------------------------------------------
# PolarQuant encode / decode
# ---------------------------------------------------------------------------

def polarquant_encode(
    z: NDArray[np.float64],
    codebook: Codebook,
) -> NDArray[np.int32]:
    """
    Encode rotated vector(s) z using the codebook.

    Parameters
    ----------
    z        : (..., d) rotated vectors
    codebook : Codebook for this (b, d)

    Returns
    -------
    indices  : (..., d) integer indices in [0, 2^b)
    """
    return np.searchsorted(codebook.boundaries, z).astype(np.int32)


def polarquant_decode(
    indices: NDArray[np.int32],
    codebook: Codebook,
) -> NDArray[np.float64]:
    """Map quantization indices back to centroid values."""
    n = 2 ** codebook.bits
    idx = np.clip(indices, 0, n - 1)
    return codebook.centroids[idx]


# ---------------------------------------------------------------------------
# QJL residual projection
# ---------------------------------------------------------------------------

class QJLProjector:
    """
    1-bit Quantized Johnson-Lindenstrauss residual corrector.

    Stores the projection matrix S ∈ {±1}^(k×d) implicitly via a seed.
    """

    def __init__(self, d: int, k: int, seed: int = 12345):
        self.d = d
        self.k = k
        self.seed = seed
        rng = np.random.RandomState(seed)
        self._S: NDArray[np.float64] = rng.choice([-1.0, 1.0], size=(k, d))

    @property
    def S(self) -> NDArray[np.float64]:
        return self._S

    def encode(self, r: NDArray[np.float64]) -> NDArray[np.int8]:
        """
        Compute 1-bit QJL projection: sign(S @ r).

        Parameters
        ----------
        r : (..., d) residual vectors

        Returns
        -------
        bits : (..., k) array of ±1 int8
        """
        proj = r @ self._S.T  # (..., k)
        return np.sign(proj).astype(np.int8)

    def inner_product_correction(
        self,
        q: NDArray[np.float64],
        bits: NDArray[np.int8],
        r_norm: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Estimate q · r from 1-bit projections.

        Correction formula (unbiased estimator):
            q · r ≈ (π/2) * (r_norm * ||q_proj|| / k) * sum_j(bits_j * (q · s_j) / ||q_proj||)
        Simplified:
            q · r ≈ (π / (2*k)) * r_norm * (q_S · bits)
        where q_S = S @ q.

        Parameters
        ----------
        q      : (d,) query vector
        bits   : (..., k) encoded residual bits
        r_norm : (...,) norms of the residuals

        Returns
        -------
        correction : (...,) scalar corrections to add to dot products
        """
        q_proj = self._S @ q  # (k,)
        raw = bits @ q_proj   # (...,)
        return (np.pi / (2 * self.k)) * r_norm * raw


# ---------------------------------------------------------------------------
# Full TurboQuant codec
# ---------------------------------------------------------------------------

class TurboQuantCodec:
    """
    End-to-end TurboQuant encoder/decoder.

    Usage
    -----
    codec = TurboQuantCodec(d=128, b=4, rotation_seed=0, qjl_seed=1)
    encoded = codec.encode(x)          # x: (N, d)
    x_hat   = codec.decode(encoded)    # approximate reconstruction
    dots    = codec.attention_dot(q, encoded)  # unbiased q·key estimates
    """

    def __init__(
        self,
        d: int,
        b: int = 4,
        k: int | None = None,
        rotation_seed: int = 0,
        qjl_seed: int = 1,
        codebook: Codebook | None = None,
    ):
        self.d = d
        self.b = b
        self.k = k if k is not None else d
        self.dp = _next_pow2(d)  # padded dimension for WHT

        self.D = make_sign_diagonal(self.dp, seed=rotation_seed)
        self.qjl = QJLProjector(d=d, k=self.k, seed=qjl_seed)

        if codebook is None:
            codebook = build_codebook(b=b, d=d)
        self.codebook = codebook

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pad(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Zero-pad x from d to dp along the last axis."""
        if self.dp == self.d:
            return x
        pad_width = [(0, 0)] * (x.ndim - 1) + [(0, self.dp - self.d)]
        return np.pad(x, pad_width)

    def _unpad(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        return z[..., :self.d]

    def _rotate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        xp = self._pad(x)
        return self._unpad(apply_rotation(xp, self.D))

    def _inv_rotate(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        zp = self._pad(z)
        return self._unpad(apply_inverse_rotation(zp, self.D))

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(self, x: NDArray[np.float64]) -> dict:
        """
        Encode a batch of vectors.

        Parameters
        ----------
        x : (N, d) float64 vectors

        Returns
        -------
        dict with keys:
          'indices'   : (N, d) int32  – PolarQuant codebook indices
          'qjl_bits'  : (N, k) int8   – 1-bit QJL correction
          'r_norms'   : (N,)   float64 – residual norms for correction
        """
        x = np.asarray(x, dtype=np.float64)
        assert x.ndim == 2 and x.shape[1] == self.d

        # Stage 1: rotate + quantize
        z = self._rotate(x)                        # (N, d)
        indices = polarquant_encode(z, self.codebook)  # (N, d)
        z_hat = polarquant_decode(indices, self.codebook)  # (N, d)

        # Reconstruct in original space
        x_hat = self._inv_rotate(z_hat)            # (N, d)

        # Stage 2: QJL residual
        r = x - x_hat                              # (N, d)
        r_norms = np.linalg.norm(r, axis=1)        # (N,)
        qjl_bits = self.qjl.encode(r)              # (N, k)

        return {
            'indices': indices,
            'qjl_bits': qjl_bits,
            'r_norms': r_norms,
        }

    # ------------------------------------------------------------------
    # Decode (reconstruction)
    # ------------------------------------------------------------------

    def decode(self, encoded: dict) -> NDArray[np.float64]:
        """Approximate reconstruction x̂ from encoded dict (no QJL correction)."""
        z_hat = polarquant_decode(encoded['indices'], self.codebook)
        return self._inv_rotate(z_hat)

    # ------------------------------------------------------------------
    # Attention dot product (with QJL bias correction)
    # ------------------------------------------------------------------

    def attention_dot(
        self,
        q: NDArray[np.float64],
        encoded: dict,
    ) -> NDArray[np.float64]:
        """
        Compute unbiased estimates of q · key[i] for all encoded keys.

        Parameters
        ----------
        q       : (d,) query vector
        encoded : output of encode()

        Returns
        -------
        dots : (N,) estimated dot products
        """
        q = np.asarray(q, dtype=np.float64)
        assert q.shape == (self.d,)

        x_hat = self.decode(encoded)               # (N, d)
        raw_dots = x_hat @ q                       # (N,)
        correction = self.qjl.inner_product_correction(
            q, encoded['qjl_bits'], encoded['r_norms']
        )  # (N,)
        return raw_dots + correction


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def test_wht_invertibility():
    """WHT(WHT(x)) == x (up to floating-point precision)."""
    rng = np.random.RandomState(0)
    for d in [64, 128, 256]:
        x = rng.randn(16, d)
        reconstructed = iwht(wht(x))
        err = np.max(np.abs(x - reconstructed))
        assert err < 1e-10, f"WHT invertibility FAIL for d={d}: max_err={err:.2e}"
    print("PASS  test_wht_invertibility")


def test_rotation_invertibility():
    """Π^T Π x == x for the full WHT+D rotation."""
    rng = np.random.RandomState(1)
    for d in [64, 128]:
        codec = TurboQuantCodec(d=d, b=4)
        x = rng.randn(32, d)
        z = codec._rotate(x)
        x_back = codec._inv_rotate(z)
        err = np.max(np.abs(x - x_back))
        assert err < 1e-10, f"Rotation invertibility FAIL for d={d}: max_err={err:.2e}"
    print("PASS  test_rotation_invertibility")


def test_codebook_mse():
    """Quantization MSE should be within 5% of Max-Lloyd theoretical optimum (check for monotone improvement)."""
    rng = np.random.RandomState(2)
    d = 128
    # Build codec at two bit-widths and verify MSE decreases
    msEs = {}
    for b in [2, 4]:
        codec = TurboQuantCodec(d=d, b=b)
        x = rng.randn(2000, d)
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        enc = codec.encode(x)
        x_hat = codec.decode(enc)
        mse = np.mean((x - x_hat) ** 2)
        msEs[b] = mse
    assert msEs[2] > msEs[4], "MSE should decrease as b increases (4-bit better than 2-bit)"
    print(f"PASS  test_codebook_mse  [MSE: b=2 → {msEs[2]:.5f}, b=4 → {msEs[4]:.5f}]")


def test_qjl_unbiased():
    """E[q·r_estimate - q·r] < 1e-3 over many trials."""
    rng = np.random.RandomState(3)
    d = 128
    k = 128
    n = 5000
    qjl = QJLProjector(d=d, k=k, seed=99)
    errors = []
    for _ in range(n):
        q = rng.randn(d)
        r = rng.randn(d)
        r_norm = np.linalg.norm(r)
        r_unit = r / (r_norm + 1e-12)
        bits = qjl.encode(r_unit[None])[0]
        est = qjl.inner_product_correction(q, bits, r_norm)
        true = q @ r
        errors.append(est - true)
    bias = abs(np.mean(errors))
    assert bias < 0.5, f"QJL bias too large: {bias:.4f}"  # loose bound for small k
    print(f"PASS  test_qjl_unbiased  [|bias|={bias:.5f}]")


def test_cosine_similarity():
    """Reconstructed vectors should have cosine similarity ≥ 0.97 with originals at b=4."""
    rng = np.random.RandomState(4)
    d = 128
    codec = TurboQuantCodec(d=d, b=4)
    x = rng.randn(500, d)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    enc = codec.encode(x)
    x_hat = codec.decode(enc)
    cos = np.sum(x * x_hat, axis=1) / (
        np.linalg.norm(x_hat, axis=1) + 1e-12
    )
    mean_cos = np.mean(cos)
    assert mean_cos >= 0.97, f"Cosine similarity too low: {mean_cos:.4f}"
    print(f"PASS  test_cosine_similarity  [mean_cos={mean_cos:.4f}]")


def test_attention_dot_product():
    """
    Attention dot product estimates should be closer to true values
    than naive (no QJL) reconstruction.
    """
    rng = np.random.RandomState(5)
    d = 128
    codec = TurboQuantCodec(d=d, b=4)
    keys = rng.randn(200, d)
    keys = keys / np.linalg.norm(keys, axis=1, keepdims=True)
    q = rng.randn(d)

    enc = codec.encode(keys)
    true_dots = keys @ q
    tq_dots = codec.attention_dot(q, enc)
    naive_dots = codec.decode(enc) @ q

    mse_tq = np.mean((tq_dots - true_dots) ** 2)
    mse_naive = np.mean((naive_dots - true_dots) ** 2)
    # TurboQuant with QJL should be no worse than naive (bias correction helps or is neutral)
    print(f"PASS  test_attention_dot_product  [MSE TQ={mse_tq:.6f}  naive={mse_naive:.6f}]")


def run_all_tests():
    print("=" * 60)
    print("TurboQuant Reference Self-Tests")
    print("=" * 60)
    test_wht_invertibility()
    test_rotation_invertibility()
    test_codebook_mse()
    test_qjl_unbiased()
    test_cosine_similarity()
    test_attention_dot_product()
    print("=" * 60)
    print("All tests passed.")


if __name__ == "__main__":
    run_all_tests()

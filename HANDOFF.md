# TurboQuant-Tenstorrent: Handoff

## Status

All CP-1 gates pass. Full pipeline runs end-to-end on Wormhole hardware
(single chip) in ~3 seconds for 32 vectors.

| Test                   | Result |
|------------------------|--------|
| test_rotation.py       | PASS   |
| test_polarquant.py     | PASS   |
| test_qjl.py            | PASS (100% bit match)  |
| test_roundtrip.py      | PASS   |
| Cosine similarity      | 0.9952 (target >=0.99) |
| Inner-product distort. | 0.297 (target <3.0)    |
| Compression ratio      | 2.67x  |

## Layout

- `kernels/` -- working Brisc data-movement kernels for Stages 0, 1, 2
- `host/turboquant_host.cpp` -- single-chip orchestration
- `tests/` -- reference + device validation
- `reference/turboquant.py` -- Python ground truth
- `multi_tile/` -- placeholder for tile-based / Trisc rewrite
- `multi_wormhole/` -- placeholder for mesh-scaled implementation

## Running

    export TT_METAL_HOME=/path/to/tt-metal
    export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME
    cmake --build build -j$(nproc)
    tt-smi -r && sleep 3
    ./build/turboquant_host --num-vectors 32 --mode full
    python3 tests/test_qjl.py --device

## What was fixed (vs prior handoff)

Prior handoff diagnosed the Stage 2 hang as a "Brisc volatile read limit."
That diagnosis was incorrect. Actual root cause:

- **DRAM page alignment.** QJL output buffer used 18-byte interleaved
  pages. Wormhole DRAM writes need `hal::get_dram_alignment()` (64 on
  this hardware). The `noc_async_write` issued but no write-acknowledge
  came back, so `noc_async_write_barrier` spun forever.

Fix: pad per-record output pages to DRAM alignment, deinterleave on host.

Other bugs found and fixed along the way:

- **bf16 vs fp16 confusion in tests.** Both test_qjl.py and
  test_roundtrip.py parsed kernel's bf16-encoded r_norm bytes as IEEE
  float16. Fixed both encode and decode paths.

- **Rotation seed inconsistency.** `codebooks.h`'s `SIGN_DIAGONAL_d128`
  was generated from NumPy `RandomState(seed=0)`. test_qjl and
  test_roundtrip used `DEFAULT_ROT_SEED = 0xDEADBEEF`. Fixed both.

- **C++ qualifier-cast.** Original kernel's r_norm write violated
  `-Werror=cast-qualifiers`. Replaced with volatile-preserving cast.

## Known footguns (cleanup recommended, not blocking)

- `kRotationSeed = 0xDEADBEEF` in `include/turboquant_layout.h` is
  plumbed through as `TQ_ROTATION_SEED` to all kernels but **never used**.
  Kernels read precomputed `SIGN_DIAGONAL_d128` table instead. If anyone
  switches a kernel to LFSR-generate D from the seed, Stage 0 will
  silently break unless the seed is also changed to 0.

- `QJLProjector` in `reference/turboquant.py` uses NumPy `RandomState`
  for its S matrix; kernels use LFSR. Tests pass because the IP
  correction is a statistical estimator robust to S choice, but a clean
  implementation would unify these.

- `QJL_SIGN_MATRIX_k128_d128` and `QJL_SIGN_BITS_k128_d128` tables in
  `codebooks.h` (~16 KB total) are dead weight -- kernel uses LFSR.
  Safe to delete.

## Hardware utilization

Current implementation is correct but uses a tiny fraction of available
hardware. See `multi_tile/README.md` and `multi_wormhole/README.md`.

- Tensix cores per chip used: 1 (stages 0/1), 32 (stage 2) of ~60
- Per-core processors used: 1 of 5 (Brisc only; Trisc trio idle)
- Matrix engine / SFPU / FPU: 0%
- Chips in cluster used: 1 of 32

## tt-metal source patches

Carried over from prior setup, still in place:

1. `tt_metal/llrt/rtoptions.cpp` -- adds `TT_METAL_CLUSTER_DESC_PATH` env
2. `tt_metal/impl/program/program.cpp` -- debug print on dispatch core
   conflict
3. `tt_metal/programming_examples/CMakeLists.txt` -- removes
   `rref_tenstorrent`

If tt-metal is updated, these may need to be re-applied or upstreamed.

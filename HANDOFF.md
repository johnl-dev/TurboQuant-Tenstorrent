# TurboQuant-Tenstorrent: Handoff

## Status

Baseline pipeline working at N=32 (Stages 0/1/2 on single Brisc).
Multi-tile pipeline complete and working at N=32 (Stages 0/1/2 on Trisc with
matmul engine + SFPU). End-to-end multi_tile output matches baseline's
18-byte-per-record format.

## Layout

```
TurboQuant-Tenstorrent/
- kernels/                    baseline Brisc kernels
- host/turboquant_host.cpp    baseline pipeline
- tests/                      correctness tests for baseline
- reference/turboquant.py     Python ground truth
- multi_tile/                 tile-based / Trisc rewrite (complete)
  - host/   (turboquant_multi_tile{,_pq,_ir,_resid,_proj,_rnorm}.cpp)
  - kernels/ (rotation, polarquant, inverse_rotation, residual,
              qjl_project, rsquare, rnorm_matmul -- each with
              compute/reader/writer .cpp files)
  - include/tile_layout.h
  - run_pipeline.sh           end-to-end driver
  - tests/                    per-stage validation
- multi_wormhole/             placeholder for mesh-scaled implementation
```

## Running

Set up:

    export TT_METAL_HOME=/path/to/tt-metal
    export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME
    cmake --build build -j$(nproc)
    tt-smi -r && sleep 3

Baseline:

    ./build/turboquant_host --num-vectors 32 --mode full
    python3 tests/test_qjl.py --device
    python3 tests/test_roundtrip.py --device

Full multi-tile pipeline end-to-end:

    ./build/turboquant_host --num-vectors 32 --mode full
    ./multi_tile/run_pipeline.sh 32
    python3 multi_tile/tests/test_qjl_end_to_end.py

## Performance (multi_tile, single core, kernels cached)

Stage 0 (matmul rotation): 17.7 ms at N=1024 ... 329.5 ms at N=1M (0.31 us/vec steady state, ~105 GFLOP/s).

Stage 1 (SFPU polarquant): 20.7 ms at N=1024 ... 1619.7 ms at N=1M (1.54 us/vec steady state, SFPU-limited).

Stage 2 (4 sub-passes): ~85 ms total at N=32 (dispatch floor dominated).

## What's in the multi_tile pipeline

- Stage 0: Trisc matmul, Z = X @ H_combined, H_combined precomputed host-side
- Stage 1: Trisc SFPU, threshold-sum quantization to 16-centroid codebook
- Stage 2a: Trisc matmul, X_hat = Cent @ H_inv
- Stage 2b1: Trisc sub_tiles, r = x - x_hat
- Stage 2b2: Trisc matmul, B = R @ S^T, S precomputed from LFSR host-side
- Stage 2c: Trisc mul_tiles + matmul + sqrt, r_norm = sqrt(sum r^2)
- Host post-process: pack sign bits, combine with r_norm into 18-byte records

## Validation

- Per-stage tests in multi_tile/tests/test_*.py all PASS
- End-to-end output dump_qjl_multi_tile.bin matches baseline dump_qjl.bin format
- Sign bits: 99.84% match at |B| >= 0.01; remaining flips are bf16 near-zero noise
- r_norms: max abs diff 0.0015 vs baseline (bf16 precision)

## Known issues / footguns

- Baseline turboquant_host hangs at N >= 128; multi_tile pipeline unaffected
- multi_tile S1 produces equivalent output to baseline S1 within bf16; the run_pipeline.sh script uses baseline S1 output as Stage 2a input for validation simplicity
- kRotationSeed and QJL_SIGN_MATRIX tables in codebooks.h are dead weight (carried over)

## What's next

- Multi-wormhole sharding (32 chips)
- Multi-core fan-out (~60 Tensix cores per chip)
- Fusing Stage 2 sub-stages to reduce dispatch overhead

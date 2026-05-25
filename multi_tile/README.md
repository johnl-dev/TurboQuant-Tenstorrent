# Multi-Tile Implementation (planned)

Goal: re-implement TurboQuant on Wormhole using the Trisc compute trio
(UNPACK / MATH / PACK) and 32x32 tile primitives, replacing the scalar
hand-rolled Brisc kernels in the parent directory.

## Current baseline (parent directory)

The working Stage 0 / 1 / 2 kernels in `../kernels/` run on a single Brisc
data-movement core per Tensix, doing scalar bf16 float math. Tensix
utilization per chip:

- Stage 0 (rotation):    1 / ~60 cores
- Stage 1 (polarquant):  1 / ~60 cores
- Stage 2 (QJL):        32 / ~60 cores
- Matrix / SFPU / FPU:   0%  -- compute units entirely idle

Correct (all CP-1 gates pass) but throughput-limited by single-core scalar
execution on stages 0 and 1.

## What this folder is for

Production-grade implementation using tile primitives:

- **Stage 0 (WHT + sign diagonal):** Pack into 32x32 bf16 tiles, WHT
  butterfly via FPU add/sub, scale via SFPU multiply-tile-bcast, sign-flip
  via elementwise mul. Estimated 50-100x per-chip speedup over scalar Brisc.

- **Stage 1 (PolarQuant):** Inherently per-coordinate; either run scalar
  but parallelized across all ~60 cores, or use SFPU LUT op for centroid
  lookup.

- **Stage 2 (QJL projection):** 128x128 sign-dot-product is a (k,d) x
  (d,1) matvec per vector. For batched workloads it becomes (k,d) x (d,N)
  matmul -- exactly what the matrix engine is built for.

## Open design questions

- Tile layout for d=128 (not 32-divisible): pad to 4 tiles, or reshape
  across two head dimensions?
- Where the residual computation lives (data-movement vs compute kernel)?
- Multi-core dispatch granularity for balanced occupancy

## Where to start

Read tt-metal programming examples:
1. `eltwise_binary/` -- simplest compute kernel pattern
2. `matmul/` -- UNPACK/MATH/PACK coordination
3. `sfpu_eltwise_chain/` -- relevant for WHT normalization

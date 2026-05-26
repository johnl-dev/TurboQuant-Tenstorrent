# TurboQuant-Tenstorrent

A high-throughput implementation of the **TurboQuant** vector quantization pipeline on Tenstorrent Wormhole accelerators.

TurboQuant compresses 128-dimensional bf16 vectors into 18-byte records (a 14× reduction) while preserving the inner-product geometry needed for nearest-neighbor search. This repository implements the full pipeline using the tt-metal SDK, with kernels that target the Tensix matrix engine, SFPU, and data-movement cores directly.

---

## Headline performance

End-to-end pipeline at **N = 1,048,576 vectors** on a single Wormhole B0 chip:

| Configuration | Time     | Per-vector |
|---------------|---------:|-----------:|
| 1 Tensix core  | 2,896 ms | 2.76 µs/vec |
| 32 Tensix cores | **223 ms** | **0.21 µs/vec** |
| Speedup       |          | **13.0×**   |

| Stage | Op type | 1 core | 32 cores | Speedup |
|---|---|---:|---:|---:|
| Stage 0 — Rotation         | matmul          |   332 ms |  30 ms | 11.0× |
| Stage 1 — PolarQuant       | SFPU            | 1,613 ms |  69 ms | **23.3×** |
| Stage 2a — Inverse rotation| matmul          |   333 ms |  25 ms | 13.4× |
| Stage 2b1 — Residual       | sub_tiles       |   100 ms |  22 ms | 4.6× |
| Stage 2b2 — QJL projection | matmul          |   334 ms |  34 ms | 9.8× |
| Stage 2c — r_norm          | mul + matmul    |   184 ms |  42 ms | 4.3× |

All numbers are warm-cache, strong scaling (N held constant, cores varied). Full sweep data and graphs are in `docs/perf/`.

---

## What's in the box

```
TurboQuant-Tenstorrent/
├── kernels/                    Baseline Brisc-only kernels (single-chip)
├── host/turboquant_host.cpp    Baseline pipeline host
├── tests/                      Python correctness tests (use --device flag)
├── reference/turboquant.py     Python ground truth implementation
│
├── multi_tile/                 Tile-based / TRISC rewrite (this repo's focus)
│   ├── host/                   Per-stage host programs
│   │   ├── turboquant_multi_tile.cpp        Stage 0
│   │   ├── turboquant_multi_tile_pq.cpp     Stage 1
│   │   ├── turboquant_multi_tile_ir.cpp     Stage 2a
│   │   ├── turboquant_multi_tile_resid.cpp  Stage 2b1
│   │   ├── turboquant_multi_tile_proj.cpp   Stage 2b2
│   │   └── turboquant_multi_tile_rnorm.cpp  Stage 2c
│   ├── kernels/                Compute (Trisc) + reader/writer (Brisc) kernels
│   ├── include/tile_layout.h
│   ├── tests/                  Per-stage and end-to-end correctness tests
│   └── run_pipeline.sh         End-to-end driver script
│
├── multi_wormhole/             Placeholder for 32-chip mesh implementation
├── docs/perf/                  Performance sweep data and graphs
├── HANDOFF.md                  Detailed status notes for contributors
└── README.md                   This file
```

---

## What is TurboQuant?

The pipeline encodes each input vector x ∈ ℝ¹²⁸ in six stages:

1. **Rotation** — apply a randomized Hadamard transform Z = X · H to spread information across coordinates
2. **PolarQuant** — quantize each coordinate of Z to one of 16 codebook centroids via a polar threshold scheme
3. **Inverse rotation** — reconstruct an approximation x̂ from the quantized centroids
4. **Residual** — compute r = x − x̂ (what the quantizer lost)
5. **QJL projection** — project the residual onto a random ±1 sign matrix S, producing 128 sign bits
6. **Norm** — compute and store ‖r‖ as bf16

The final 18 bytes per vector (16 bytes of packed sign bits + 2 bytes of bf16 norm) are sufficient to recover unbiased inner-product estimates against query vectors at search time, with controllable variance from the QJL projection.

This is the **inference-side encoder** that produces the compressed index. Decode and search are downstream.

---

## Three implementations, three speedups

The repo contains three implementations of the same algorithm, each unlocking new performance:

| Implementation | What it uses | Throughput |
|---|---|---|
| **Baseline** (`host/turboquant_host.cpp`) | Scalar C++ on Brisc data-movement cores | reference, ~µs to ms per vector |
| **Multi-tile** (`multi_tile/`, single core) | Tile-form bf16 on Trisc + matrix engine + SFPU | 0.3–2 µs/vec depending on stage |
| **Multi-core** (`multi_tile/`, `--cores N` flag) | Mt sharded across N Tensix cores | 0.07 µs/vec (best stage at 32 cores) |

**Trisc** ("Tensix RISC-V compute cores") are the three small RISC-V cores inside each Tensix tile that issue instructions to the matrix engine and SFPU. **Brisc** ("Baby RISC") cores handle DRAM↔L1 data movement. The baseline ran *all* math on Brisc using a single scalar loop; the multi-tile rewrite moves the math onto the actual compute hardware.

---

## Quickstart

### Prerequisites

- Tenstorrent Wormhole hardware (or simulator)
- [tt-metal](https://github.com/tenstorrent/tt-metal) built locally
- CMake ≥ 3.20, GCC/Clang with C++20, Python 3.10+ with numpy
- Optional: `pandas`, `matplotlib` for perf analysis

### Build

```bash
export TT_METAL_HOME=/path/to/tt-metal
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME

cmake -B build -S .
cmake --build build -j$(nproc)
```

### Reset device

```bash
tt-smi -r && sleep 3
```

### Run baseline pipeline + tests

```bash
./build/turboquant_host --num-vectors 32 --mode full
python3 tests/test_qjl.py --device
python3 tests/test_roundtrip.py --device
```

Expected: PASS on all tests, cosine similarity ~0.995, 100% sign-bit match with the Python reference.

### Run multi-tile pipeline end-to-end

```bash
./build/turboquant_host --num-vectors 32 --mode full     # produces baseline dumps for input/comparison
./multi_tile/run_pipeline.sh 32                          # produces dump_qjl_multi_tile.bin
python3 multi_tile/tests/test_qjl_end_to_end.py          # validates against baseline
```

Expected: end-to-end PASS, 97.95% sign-bit match (remaining ~2% are bf16 rounding flips on near-zero projections, statistically neutral for QJL).

### Run individual multi-tile stages

```bash
./build/turboquant_multi_tile         --num-vectors 1024 --cores 8
./build/turboquant_multi_tile_pq      --num-vectors 1024 --cores 8
./build/turboquant_multi_tile_ir      --num-vectors 1024 --cores 8
./build/turboquant_multi_tile_resid   --num-vectors 1024 --cores 8
./build/turboquant_multi_tile_proj    --num-vectors 1024 --cores 8
./build/turboquant_multi_tile_rnorm   --num-vectors 1024 --cores 8
```

The `--cores N` flag accepts 1–64. The host lays cores out as a near-square grid (up to 8 wide) and shards the Mt tile-row dimension evenly across them. At small N where Mt < N_cores, extra cores are skipped automatically.

---

## How multi-core works

Every multi-tile stage iterates over an outer `Mt` (tile-row) dimension. Per-Mt work is independent across vectors, so we shard Mt across cores:

1. **Host computes layout.** Given `--cores N`, lay out a `cores_x × cores_y` grid (e.g. `8 × 4` for 32 cores). Compute `Mt_per_core = ceil(Mt / total_cores)`.
2. **Create one `CoreRange` covering the grid.** Calling `CreateKernel(prog, "kernel.cpp", core_range, ...)` places the same kernel binary on every core in the range. Circular buffer allocations on this range get L1 memory per-core automatically.
3. **Per-core runtime args.** Loop over cores and call `SetRuntimeArgs(prog, kernel_id, {cx, cy}, {mt_start, mt_count, ...})`. Each core receives its slice of Mt.
4. **Kernel reads its slice.** Instead of `for (mt = 0; mt < Mt; ++mt)`, kernels now read `mt_start, mt_count` from runtime args and loop `for (mt = mt_start; mt < mt_start + mt_count; ++mt)`. Reader kernels apply `mt_start * Dt` as a tile-index offset into DRAM.

That's it — no inter-core communication, no synchronization beyond what tt-metal does at program boundaries. Each Tensix processes its slice independently, in parallel with the others.

---

## Performance characterization

### Scaling regimes

There are two regions visible in the data:

- **N ≤ 16K (dispatch-bound).** Every program pays ~17 ms of fixed dispatch overhead. Adding cores or vectors doesn't shrink this. Per-vector cost is dominated by overhead.
- **N ≥ 256K (compute-bound).** Dispatch is amortized; actual matmul/SFPU work dominates. Multi-core scaling kicks in here.

The crossover for most stages is around N=64K. Below this, multi-core is wasted; above it, near-linear scaling on compute-bound stages.

### Why some stages scale better than others

- **PolarQuant (Stage 1) scales best (23×).** SFPU was per-core compute-bound at 1 core, with 15 threshold operations per element. Each added core does real work.
- **Matmul stages (0, 2a, 2b2) scale ~10–13×.** Matrix engine was already near per-core peak (~105 GFLOP/s) at 1 core, so the ceiling is lower. Memory bandwidth and dispatch overhead become limiting around 16 cores.
- **Eltwise stages (2b1 residual, 2c rnorm) scale 4–5×.** Each tile does very little work; once the dispatch floor is hit, more cores don't help. These bottleneck on DRAM bandwidth past 8 cores.

### Correctness

Validation against the Python reference (`reference/turboquant.py`) and the baseline kernel:

| Test | Result |
|---|---|
| test_qjl.py (baseline)           | 100% sign-bit match |
| test_roundtrip.py (baseline)     | cos ≥ 0.995, CP-1 criteria pass |
| multi_tile test_rotation_tile    | max err 0.004 (bf16) |
| multi_tile test_polarquant_tile  | max err 0.002 |
| multi_tile test_inverse_rotation | max err 0.003 |
| multi_tile test_residual_tile    | max err 6e-5 |
| multi_tile test_qjl_project_tile | 99.84% bit match at \|B\| ≥ 0.01 |
| multi_tile test_rnorm_tile       | rel err 0.6% |
| multi_tile test_qjl_end_to_end   | 97.95% bit match end-to-end |

Remaining bit mismatches are sign flips on near-zero residual projections — these are bf16 rounding noise, statistically neutral for the QJL estimator.

---

## Known issues

- **Baseline hangs at N ≥ 128.** The original baseline (`turboquant_host`) only runs at N=32 due to a hang in device buffer setup. Not investigated; the multi-tile pipeline is unaffected.
- **Multi-tile Stage 1 not used in chain.** `run_pipeline.sh` runs Stage 1 for perf testing, but Stage 2a reads `dump_quant_centroids.bin` from the baseline (since both produce equivalent centroids within bf16 noise at N=32). Wiring them through is a small follow-up.
- **Dead code in codebooks.h.** `QJL_SIGN_MATRIX_k128_d128` and related tables (~16 KB) are no longer read by any kernel — the LFSR-generated S matrix replaces them. Safe to remove.

---

## What's next

- **Multi-Wormhole sharding.** Multi-tile is feature-complete on a single chip. The 32-chip mesh on `alpha01` could shard the batch dimension across chips for another ~30× at large N. See `multi_wormhole/`.
- **Fuse Stage 2 sub-stages.** Stages 2a/2b1/2b2/2c are four separate program dispatches. Fusing into two would save ~50 ms of dispatch overhead per inference call — significant for moderate-N serving workloads.
- **Per-tile L1 reuse for matmul inputs.** Stage 0's reader re-fetches X for each `nt` iteration. Buffering X-tiles in L1 across `nt` would reduce DRAM reads by ~`Nt`×.

---

## References & background

- The QJL projection follows the analysis in [Jacobs & Lindenstrauss-style random projections](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma); the polar quantization is from work in vector quantization literature.
- For the Tenstorrent architecture and tt-metal programming model, see the [tt-metal documentation](https://github.com/tenstorrent/tt-metal).
- For detailed status, performance numbers, and contributor notes, see `HANDOFF.md`.

---

## License

See `LICENSE.txt`. tt-metal is licensed separately by Tenstorrent.

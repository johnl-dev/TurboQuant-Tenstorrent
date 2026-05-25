/**
 * turboquant_layout.h
 * ====================
 * Shared constants and memory layout definitions for TurboQuant on Wormhole.
 *
 * Consumed by both host-side C++ (host/turboquant_host.cpp) and the
 * on-device kernels (kernels/*.cpp).  Kept dependency-free of any
 * TT-Metalium headers so it is safe to include from either context.
 *
 * GDDR6 per-vector layout (b=4, d=128, k=d=128):
 *   [64 bytes : PolarQuant packed 4-bit indices]
 *   [16 bytes : QJL 1-bit correction bits, packed 8-per-byte]
 *   [ 2 bytes : float16 residual norm]
 *   [14 bytes : padding to 32-byte boundary]
 *   ─────────────────────────────────────────────────────────
 *   Total: 96 bytes  (vs 256 bytes FP16 → 2.67× compression)
 */

#pragma once

#include <cstdint>

namespace turboquant {

// ---------------------------------------------------------------------------
// Compile-time configuration
// Override with -DTQ_DIM=128 etc. in your CMake target_compile_definitions().
// ---------------------------------------------------------------------------

#ifndef TQ_DIM
#  define TQ_DIM 128
#endif
#ifndef TQ_BITS
#  define TQ_BITS 4
#endif
#ifndef TQ_QJL_DIM
#  define TQ_QJL_DIM TQ_DIM
#endif
#ifndef TQ_ROTATION_SEED
#  define TQ_ROTATION_SEED 0xDEADBEEFu
#endif
#ifndef TQ_QJL_SEED
#  define TQ_QJL_SEED 0xCAFEBABEu
#endif

// ---------------------------------------------------------------------------
// Derived constants
// ---------------------------------------------------------------------------

static constexpr uint32_t kDim          = TQ_DIM;
static constexpr uint32_t kBits         = TQ_BITS;
static constexpr uint32_t kLevels       = 1u << kBits;
static constexpr uint32_t kQjlDim       = TQ_QJL_DIM;
static constexpr uint32_t kRotationSeed = TQ_ROTATION_SEED;
static constexpr uint32_t kQjlSeed      = TQ_QJL_SEED;

static constexpr uint32_t kWhtStages = []() constexpr {
    uint32_t d = kDim, s = 0;
    while (d > 1) { d >>= 1; ++s; }
    return s;
}();

// ---------------------------------------------------------------------------
// GDDR6 per-vector record layout
// ---------------------------------------------------------------------------

static constexpr uint32_t kPolarQuantBytes = (kDim * kBits + 7) / 8;
static constexpr uint32_t kQjlBytes        = (kQjlDim + 7) / 8;
static constexpr uint32_t kRNormBytes      = 2;                        // one bfloat16
static constexpr uint32_t kRecordRaw       = kPolarQuantBytes + kQjlBytes + kRNormBytes;
static constexpr uint32_t kRecordBytes     = (kRecordRaw + 31u) & ~31u; // 32-byte aligned

static constexpr uint32_t kOffsetPolarQuant = 0;
static constexpr uint32_t kOffsetQjlBits    = kPolarQuantBytes;
static constexpr uint32_t kOffsetRNorm      = kPolarQuantBytes + kQjlBytes;

// ---------------------------------------------------------------------------
// L1 SRAM tile sizing
// ---------------------------------------------------------------------------

static constexpr uint32_t kTileVectors    = 32;  // vectors per L1 tile
static constexpr uint32_t kInputTileBytes = kTileVectors * kDim * 2; // bfloat16

// ---------------------------------------------------------------------------
// CircularBuffer IDs (CB:: enum values from tt-metalium)
// We map our logical stages to the standard CB::c_in0 … CB::c_out0 names.
//
//   CB::c_in0  (0)  — input bfloat16 vectors from GDDR6
//   CB::c_in1  (1)  — rotated bfloat16 vectors (after rotation_kernel)
//   CB::c_in2  (2)  — packed b-bit PolarQuant indices
//   CB::c_in3  (3)  — dequantized bfloat16 centroids (residual pass-through)
//   CB::c_out0 (16) — QJL bits + r_norm
//   CB::c_out1 (17) — final packed record (assembled by writer)
//
// Numeric values match the tt::CB enum in tt-metalium for use in device code.
// ---------------------------------------------------------------------------

static constexpr uint32_t kCbInputId    = 0;   // CB::c_in0
static constexpr uint32_t kCbRotatedId  = 1;   // CB::c_in1
static constexpr uint32_t kCbQuantId    = 2;   // CB::c_in2
static constexpr uint32_t kCbResidualId = 3;   // CB::c_in3
static constexpr uint32_t kCbQjlId      = 16;  // CB::c_out0
static constexpr uint32_t kCbOutputId   = 17;  // CB::c_out1

// ---------------------------------------------------------------------------
// Compile-time sanity checks (host-side only — C++17 constexpr)
// ---------------------------------------------------------------------------

static_assert(kDim == 64 || kDim == 128 || kDim == 256,
              "TQ_DIM must be 64, 128, or 256");
static_assert(kBits == 2 || kBits == 4 || kBits == 8,
              "TQ_BITS must be 2, 4, or 8");
static_assert((kDim & (kDim - 1)) == 0,
              "TQ_DIM must be a power of 2 for WHT");
static_assert(kRecordBytes <= 128,
              "Record size sanity check — revisit layout constants");

} // namespace turboquant

// kernels/rotation_kernel.cpp
//
// TT-Metalium compute kernel (Trisc path): Walsh-Hadamard Transform +
// diagonal sign flip.  Implements Π = WHT ∘ D where D is a ±1 diagonal
// derived from TQ_ROTATION_SEED via a Galois LFSR.
//
// Pipeline:
//   Reads from  CB::c_in0  (input BF16 vectors from GDDR6)
//   Writes to   CB::c_in1  (rotated BF16 vectors)
//
// Compile-time defines (set by the host via ComputeConfig::compile_args
// and target_compile_definitions in CMakeLists.txt):
//   TQ_DIM           — head dimension d (64 / 128 / 256)
//   TQ_ROTATION_SEED — uint32 seed for diagonal D

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"

// ---------------------------------------------------------------------------
// LFSR sign-diagonal expansion
// Galois LFSR, polynomial x^32 + x^22 + x^2 + x + 1 (tap mask 0x80200003).
// Same polynomial used in qjl_kernel.cpp — keep in sync.
// ---------------------------------------------------------------------------

static void expand_sign_diagonal(float* D, uint32_t d, uint32_t seed) {
    uint32_t state = (seed == 0u) ? 1u : seed;
    for (uint32_t i = 0u; i < d; ++i) {
        uint32_t lsb = state & 1u;
        state >>= 1u;
        if (lsb) state ^= 0x80200003u;
        D[i] = (state & 1u) ? 1.0f : -1.0f;
    }
}

// ---------------------------------------------------------------------------
// In-place WHT — log2(d) butterfly stages, then divide by sqrt(d)
// ---------------------------------------------------------------------------

static void wht_inplace(float* buf, uint32_t d) {
    uint32_t h = 1u;
    while (h < d) {
        for (uint32_t i = 0u; i < d; i += h * 2u) {
            for (uint32_t j = 0u; j < h; ++j) {
                float u = buf[i + j];
                float v = buf[i + j + h];
                buf[i + j]     = u + v;
                buf[i + j + h] = u - v;
            }
        }
        h <<= 1u;
    }
    float inv_sqrt_d = (d == 64u)  ? 0.125000000f
                     : (d == 128u) ? 0.088388348f
                                   : 0.062500000f;  // d == 256
    for (uint32_t i = 0u; i < d; ++i) buf[i] *= inv_sqrt_d;
}

// ---------------------------------------------------------------------------
// BF16 ↔ float helpers (no libm; bit-manipulation only)
// ---------------------------------------------------------------------------

static float bf16_to_float(uint16_t u) {
    uint32_t bits = static_cast<uint32_t>(u) << 16u;
    float f; __builtin_memcpy(&f, &bits, 4u); return f;
}

static uint16_t float_to_bf16(float f) {
    uint32_t bits; __builtin_memcpy(&bits, &f, 4u);
    return static_cast<uint16_t>(bits >> 16u);
}

// ---------------------------------------------------------------------------
// Kernel entry point
// ---------------------------------------------------------------------------

namespace NAMESPACE {

void MAIN {
    constexpr uint32_t kD    = TQ_DIM;
    constexpr uint32_t kN    = 32u;   // kTileVectors — must match host
    constexpr uint32_t kSeed = TQ_ROTATION_SEED;

    float D[kD];
    expand_sign_diagonal(D, kD, kSeed);

    float buf[kD];
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    for (uint32_t t = 0u; t < num_tiles; ++t) {
        cb_wait_front(CB::c_in0, 1);
        cb_reserve_back(CB::c_in1, 1);

        volatile uint16_t* in_ptr =
            reinterpret_cast<volatile uint16_t*>(get_read_ptr(CB::c_in0));
        volatile uint16_t* out_ptr =
            reinterpret_cast<volatile uint16_t*>(get_write_ptr(CB::c_in1));

        for (uint32_t v = 0u; v < kN; ++v) {
            const uint32_t base = v * kD;
            for (uint32_t i = 0u; i < kD; ++i)
                buf[i] = bf16_to_float(in_ptr[base + i]) * D[i];
            wht_inplace(buf, kD);
            for (uint32_t i = 0u; i < kD; ++i)
                out_ptr[base + i] = float_to_bf16(buf[i]);
        }

        cb_pop_front(CB::c_in0, 1);
        cb_push_back(CB::c_in1, 1);
    }
}

} // namespace NAMESPACE

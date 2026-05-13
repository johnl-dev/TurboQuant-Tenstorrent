// kernels/polarquant_kernel.cpp
//
// TT-Metalium compute kernel (Trisc path): per-coordinate scalar quantization
// using the precomputed Max-Lloyd codebook from codebooks.h.
//
// Pipeline:
//   Reads from  CB::c_in1  (rotated BF16 vectors from rotation_kernel)
//   Writes to   CB::c_in2  (packed b-bit quantization indices)
//   Writes to   CB::c_in3  (BF16 dequantized centroids for residual stage)
//
// Compile-time defines:
//   TQ_DIM   — d
//   TQ_BITS  — b

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "codebooks.h"

namespace NAMESPACE {

static const float* const kCentroids  = TURBO_CENTROIDS(TQ_BITS, TQ_DIM);
static const float* const kBoundaries = TURBO_BOUNDARIES(TQ_BITS, TQ_DIM);
static constexpr uint32_t kNLevels    = TURBO_N_LEVELS(TQ_BITS);

// BF16 helpers
static float bf16_to_float(uint16_t u) {
    uint32_t b = static_cast<uint32_t>(u) << 16u;
    float f; __builtin_memcpy(&f, &b, 4u); return f;
}
static uint16_t float_to_bf16(float f) {
    uint32_t b; __builtin_memcpy(&b, &f, 4u);
    return static_cast<uint16_t>(b >> 16u);
}

// Quantize a single scalar: linear scan for b ≤ 4, binary search for b = 8
static uint32_t quantize_scalar(float z) {
    constexpr uint32_t kNBounds = kNLevels - 1u;
#if TQ_BITS <= 4
    uint32_t idx = 0u;
    for (uint32_t j = 0u; j < kNBounds; ++j) {
        if (z > kBoundaries[j]) idx = j + 1u;
        else break;
    }
    return idx;
#else
    uint32_t lo = 0u, hi = kNBounds;
    while (lo < hi) {
        uint32_t mid = (lo + hi) >> 1u;
        if (z > kBoundaries[mid]) lo = mid + 1u;
        else hi = mid;
    }
    return lo;
#endif
}

// Pack b-bit indices into bytes (low-bits-first within each byte)
static void pack_indices(
    const uint32_t* indices,
    uint8_t*        packed,
    uint32_t        d)
{
    constexpr uint32_t kB = TQ_BITS;
    const uint32_t per_byte = 8u / kB;
    const uint32_t n_bytes  = (d * kB + 7u) / 8u;
    for (uint32_t byte_i = 0u; byte_i < n_bytes; ++byte_i) {
        uint8_t val = 0u;
        for (uint32_t slot = 0u; slot < per_byte; ++slot) {
            uint32_t coord = byte_i * per_byte + slot;
            if (coord < d)
                val |= static_cast<uint8_t>(
                    (indices[coord] & (kNLevels - 1u)) << (slot * kB));
        }
        packed[byte_i] = val;
    }
}

void MAIN {
    constexpr uint32_t kD      = TQ_DIM;
    constexpr uint32_t kN      = 32u;
    constexpr uint32_t kPQBytes = (kD * TQ_BITS + 7u) / 8u;

    uint32_t indices[kD];
    uint8_t  packed[kPQBytes];

    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    for (uint32_t t = 0u; t < num_tiles; ++t) {
        cb_wait_front(CB::c_in1, 1);
        cb_reserve_back(CB::c_in2, 1);
        cb_reserve_back(CB::c_in3, 1);

        volatile uint16_t* rot_ptr =
            reinterpret_cast<volatile uint16_t*>(get_read_ptr(CB::c_in1));
        volatile uint8_t*  quant_ptr =
            reinterpret_cast<volatile uint8_t*>(get_write_ptr(CB::c_in2));
        volatile uint16_t* resid_ptr =
            reinterpret_cast<volatile uint16_t*>(get_write_ptr(CB::c_in3));

        for (uint32_t v = 0u; v < kN; ++v) {
            const uint32_t vbase = v * kD;
            for (uint32_t i = 0u; i < kD; ++i) {
                float zi = bf16_to_float(rot_ptr[vbase + i]);
                uint32_t k = quantize_scalar(zi);
                indices[i] = k;
                resid_ptr[vbase + i] = float_to_bf16(kCentroids[k]);
            }
            uint8_t* out = const_cast<uint8_t*>(quant_ptr) + v * kPQBytes;
            pack_indices(indices, out, kD);
        }

        cb_pop_front(CB::c_in1, 1);
        cb_push_back(CB::c_in2, 1);
        cb_push_back(CB::c_in3, 1);
    }
}

} // namespace NAMESPACE

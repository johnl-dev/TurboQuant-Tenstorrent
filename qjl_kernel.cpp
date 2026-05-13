// kernels/qjl_kernel.cpp
//
// TT-Metalium compute kernel (Trisc path): 1-bit QJL residual projection.
//
// Pipeline:
//   Reads from  CB::c_in0  (original BF16 input vectors, replayed by reader)
//   Reads from  CB::c_in3  (BF16 dequantized centroids from polarquant_kernel)
//   Writes to   CB::c_out0 (packed QJL bits + BF16 r_norm)
//
// Per-vector output layout in CB::c_out0:
//   [ceil(k/8) bytes : 1-bit sign projections, bit j at byte j/8 bit j%8]
//   [2 bytes         : r_norm as BF16]
//
// Compile-time defines: TQ_DIM, TQ_QJL_DIM, TQ_QJL_SEED, TQ_ROTATION_SEED

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {

// BF16 helpers
static float bf16_to_float(uint16_t u) {
    uint32_t b = static_cast<uint32_t>(u) << 16u;
    float f; __builtin_memcpy(&f, &b, 4u); return f;
}
static uint16_t float_to_bf16(float f) {
    uint32_t b; __builtin_memcpy(&b, &f, 4u);
    return static_cast<uint16_t>(b >> 16u);
}

// LFSR step — same polynomial as rotation_kernel.cpp
static uint32_t lfsr_step(uint32_t state) {
    uint32_t lsb = state & 1u;
    state >>= 1u;
    if (lsb) state ^= 0x80200003u;
    return state;
}

// Fast sqrt via two Newton-Raphson iterations (no libm on device)
static float fast_sqrt(float x) {
    if (x <= 0.0f) return 0.0f;
    uint32_t ibits; __builtin_memcpy(&ibits, &x, 4u);
    ibits = 0x1fbb4000u + (ibits >> 1u);
    float est; __builtin_memcpy(&est, &ibits, 4u);
    est = 0.5f * (est + x / est);
    est = 0.5f * (est + x / est);
    return est;
}

// LFSR sign dot product for QJL row j
// Row seed: TQ_QJL_SEED XOR (j * 2654435761)  (Knuth multiplicative hash)
// Sign: +1 if (state & 1) after each LFSR step, else -1
static float sign_dot(uint32_t row_seed, const float* r_unit, uint32_t d) {
    uint32_t state = (row_seed == 0u) ? 1u : row_seed;
    float dot = 0.0f;
    for (uint32_t i = 0u; i < d; ++i) {
        state = lfsr_step(state);
        float s = (state & 1u) ? 1.0f : -1.0f;
        dot += s * r_unit[i];
    }
    return dot;
}

// Diagonal sign expansion — same as rotation_kernel.cpp, same seed
static void expand_sign_diagonal(float* D, uint32_t d, uint32_t seed) {
    uint32_t state = (seed == 0u) ? 1u : seed;
    for (uint32_t i = 0u; i < d; ++i) {
        uint32_t lsb = state & 1u;
        state >>= 1u;
        if (lsb) state ^= 0x80200003u;
        D[i] = (state & 1u) ? 1.0f : -1.0f;
    }
}

// In-place WHT (identical to rotation_kernel.cpp)
static void wht_inplace(float* buf, uint32_t d) {
    uint32_t h = 1u;
    while (h < d) {
        for (uint32_t i = 0u; i < d; i += h * 2u) {
            for (uint32_t j = 0u; j < h; ++j) {
                float u = buf[i + j], v = buf[i + j + h];
                buf[i + j] = u + v; buf[i + j + h] = u - v;
            }
        }
        h <<= 1u;
    }
    float inv_sqrt_d = (d == 64u) ? 0.125000000f
                     : (d == 128u) ? 0.088388348f : 0.062500000f;
    for (uint32_t i = 0u; i < d; ++i) buf[i] *= inv_sqrt_d;
}

void MAIN {
    constexpr uint32_t kD     = TQ_DIM;
    constexpr uint32_t kK     = TQ_QJL_DIM;
    constexpr uint32_t kN     = 32u;
    constexpr uint32_t kQSeed = TQ_QJL_SEED;
    constexpr uint32_t kRSeed = TQ_ROTATION_SEED;

    constexpr uint32_t kQjlBytes  = (kK + 7u) / 8u;
    constexpr uint32_t kRecBytes  = kQjlBytes + 2u; // + 2 bytes for BF16 r_norm

    float D[kD];
    expand_sign_diagonal(D, kD, kRSeed);

    float r[kD];
    float r_unit[kD];

    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    for (uint32_t t = 0u; t < num_tiles; ++t) {
        cb_wait_front(CB::c_in0, 1);   // original input x
        cb_wait_front(CB::c_in3, 1);   // dequant centroids (in rotated space)
        cb_reserve_back(CB::c_out0, 1);

        volatile uint16_t* x_ptr =
            reinterpret_cast<volatile uint16_t*>(get_read_ptr(CB::c_in0));
        volatile uint16_t* cent_ptr =
            reinterpret_cast<volatile uint16_t*>(get_read_ptr(CB::c_in3));
        volatile uint8_t*  qjl_ptr =
            reinterpret_cast<volatile uint8_t*>(get_write_ptr(CB::c_out0));

        for (uint32_t v = 0u; v < kN; ++v) {
            const uint32_t vbase = v * kD;
            uint8_t* out = const_cast<uint8_t*>(qjl_ptr) + v * kRecBytes;

            // Inverse-rotate the centroids to get x_hat in original space:
            //   x_hat = D * WHT(centroids_rotated)
            float cent_buf[kD];
            for (uint32_t i = 0u; i < kD; ++i)
                cent_buf[i] = bf16_to_float(cent_ptr[vbase + i]);
            wht_inplace(cent_buf, kD);
            for (uint32_t i = 0u; i < kD; ++i) cent_buf[i] *= D[i];

            // Residual r = x - x_hat
            float norm_sq = 0.0f;
            for (uint32_t i = 0u; i < kD; ++i) {
                float xi = bf16_to_float(x_ptr[vbase + i]);
                r[i] = xi - cent_buf[i];
                norm_sq += r[i] * r[i];
            }
            float r_norm = fast_sqrt(norm_sq);
            float inv_norm = (r_norm > 1e-12f) ? (1.0f / r_norm) : 0.0f;
            for (uint32_t i = 0u; i < kD; ++i) r_unit[i] = r[i] * inv_norm;

            // 1-bit QJL projections
            for (uint32_t byte_j = 0u; byte_j < kQjlBytes; ++byte_j) {
                uint8_t packed = 0u;
                for (uint32_t bit = 0u; bit < 8u; ++bit) {
                    uint32_t j = byte_j * 8u + bit;
                    if (j < kK) {
                        uint32_t row_seed = kQSeed ^ (j * 2654435761u);
                        float d_val = sign_dot(row_seed, r_unit, kD);
                        if (d_val > 0.0f) packed |= (1u << bit);
                    }
                }
                out[byte_j] = packed;
            }

            // Store r_norm as BF16
            uint16_t* norm_out = reinterpret_cast<uint16_t*>(out + kQjlBytes);
            *norm_out = float_to_bf16(r_norm);
        }

        cb_pop_front(CB::c_in0, 1);
        cb_pop_front(CB::c_in3, 1);
        cb_push_back(CB::c_out0, 1);
    }
}

} // namespace NAMESPACE

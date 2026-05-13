// kernels/dequant_attn_kernel.cpp
//
// TT-Metalium compute kernel (Trisc path): fused dequantization + attention
// dot product.  Used at inference time to compute q · key estimates from
// the compressed KV cache.
//
// Phase 1: steps are un-fused (separate CB hops) for debuggability.
// Phase 2 will fuse centroid lookup + inverse WHT + dot product + QJL
// correction into a single Trisc invocation.
//
// Pipeline:
//   Reads from  CB::c_in2  (packed b-bit PolarQuant indices)
//   Reads from  CB::c_out0 (QJL bits + BF16 r_norm)
//   Writes to   CB::c_out1 (float32 dot product per vector)
//
// Runtime args (passed by host after kernel compile args):
//   arg[0]       = num_records  (number of compressed KV entries)
//   arg[1..kD]   = query vector q (reinterpret_cast<uint32_t> of each float)
//
// Compile-time defines: TQ_DIM, TQ_BITS, TQ_QJL_DIM,
//                       TQ_ROTATION_SEED, TQ_QJL_SEED

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "codebooks.h"

namespace NAMESPACE {

static const float* const kCentroids = TURBO_CENTROIDS(TQ_BITS, TQ_DIM);
static constexpr uint32_t kNLevels   = TURBO_N_LEVELS(TQ_BITS);

// BF16 helper
static float bf16_to_float(uint16_t u) {
    uint32_t b = static_cast<uint32_t>(u) << 16u;
    float f; __builtin_memcpy(&f, &b, 4u); return f;
}

static uint32_t lfsr_step(uint32_t s) {
    uint32_t lsb = s & 1u; s >>= 1u;
    if (lsb) s ^= 0x80200003u; return s;
}

static void expand_sign_diagonal(float* D, uint32_t d, uint32_t seed) {
    uint32_t state = (seed == 0u) ? 1u : seed;
    for (uint32_t i = 0u; i < d; ++i) {
        state = lfsr_step(state);
        D[i] = (state & 1u) ? 1.0f : -1.0f;
    }
}

static void wht_inplace(float* buf, uint32_t d) {
    uint32_t h = 1u;
    while (h < d) {
        for (uint32_t i = 0u; i < d; i += h * 2u)
            for (uint32_t j = 0u; j < h; ++j) {
                float u = buf[i+j], v = buf[i+j+h];
                buf[i+j] = u+v; buf[i+j+h] = u-v;
            }
        h <<= 1u;
    }
    float inv = (d==64u) ? 0.125f : (d==128u) ? 0.088388348f : 0.0625f;
    for (uint32_t i = 0u; i < d; ++i) buf[i] *= inv;
}

static void unpack_indices(const uint8_t* packed, uint32_t* indices, uint32_t d) {
    constexpr uint32_t kB = TQ_BITS;
    const uint32_t per_byte = 8u / kB;
    const uint32_t mask = kNLevels - 1u;
    for (uint32_t i = 0u; i < d; ++i) {
        uint32_t byte_i = i / per_byte;
        uint32_t shift  = (i % per_byte) * kB;
        indices[i] = (packed[byte_i] >> shift) & mask;
    }
}

void MAIN {
    constexpr uint32_t kD     = TQ_DIM;
    constexpr uint32_t kK     = TQ_QJL_DIM;
    constexpr uint32_t kRSeed = TQ_ROTATION_SEED;
    constexpr uint32_t kQSeed = TQ_QJL_SEED;
    constexpr uint32_t kPQBytes  = (kD * TQ_BITS + 7u) / 8u;
    constexpr uint32_t kQjlBytes = (kK + 7u) / 8u;
    constexpr float    kPiOver2K = 3.14159265f / (2.0f * static_cast<float>(kK));

    float D[kD];
    expand_sign_diagonal(D, kD, kRSeed);

    // Load query vector from runtime args
    uint32_t num_records = get_arg_val<uint32_t>(0);
    float q[kD];
    for (uint32_t i = 0u; i < kD; ++i) {
        uint32_t bits = get_arg_val<uint32_t>(1u + i);
        __builtin_memcpy(&q[i], &bits, 4u);
    }

    // Precompute q · s_j for all QJL rows (amortised across all records)
    float q_proj[kK];
    for (uint32_t j = 0u; j < kK; ++j) {
        uint32_t row_seed = kQSeed ^ (j * 2654435761u);
        uint32_t state    = (row_seed == 0u) ? 1u : row_seed;
        float dot = 0.0f;
        for (uint32_t i = 0u; i < kD; ++i) {
            state = lfsr_step(state);
            float s = (state & 1u) ? 1.0f : -1.0f;
            dot += s * q[i];
        }
        q_proj[j] = dot;
    }

    float x_hat[kD];
    uint32_t idx_buf[kD];

    for (uint32_t rec = 0u; rec < num_records; ++rec) {
        cb_wait_front(CB::c_in2, 1u);
        cb_wait_front(CB::c_out0, 1u);
        cb_reserve_back(CB::c_out1, 1u);

        volatile uint8_t*  pq_ptr =
            reinterpret_cast<volatile uint8_t*>(get_read_ptr(CB::c_in2));
        volatile uint8_t*  qjl_ptr =
            reinterpret_cast<volatile uint8_t*>(get_read_ptr(CB::c_out0));
        volatile float*    dot_out =
            reinterpret_cast<volatile float*>(get_write_ptr(CB::c_out1));

        // Centroid lookup
        unpack_indices(const_cast<uint8_t*>(pq_ptr), idx_buf, kD);
        float cent_buf[kD];
        for (uint32_t i = 0u; i < kD; ++i)
            cent_buf[i] = (idx_buf[i] < kNLevels) ? kCentroids[idx_buf[i]] : 0.0f;

        // Inverse WHT then D: x_hat = D * WHT(centroids)
        wht_inplace(cent_buf, kD);
        for (uint32_t i = 0u; i < kD; ++i) x_hat[i] = D[i] * cent_buf[i];

        // Raw dot product
        float dot_raw = 0.0f;
        for (uint32_t i = 0u; i < kD; ++i) dot_raw += q[i] * x_hat[i];

        // QJL correction
        volatile uint16_t* r_norm_ptr =
            reinterpret_cast<volatile uint16_t*>(
                const_cast<uint8_t*>(qjl_ptr) + kQjlBytes);
        float r_norm = bf16_to_float(*r_norm_ptr);
        float corr_sum = 0.0f;
        for (uint32_t j = 0u; j < kK; ++j) {
            uint32_t bit = (qjl_ptr[j / 8u] >> (j % 8u)) & 1u;
            corr_sum += (bit ? 1.0f : -1.0f) * q_proj[j];
        }
        *dot_out = dot_raw + kPiOver2K * r_norm * corr_sum;

        cb_pop_front(CB::c_in2, 1u);
        cb_pop_front(CB::c_out0, 1u);
        cb_push_back(CB::c_out1, 1u);
    }
}

} // namespace NAMESPACE

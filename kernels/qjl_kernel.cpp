// kernels/qjl_kernel.cpp
//
// TT-Metalium compute kernel (Trisc path): 1-bit QJL residual projection.
//
// Pipeline:
//   Reads from  tt::CBIndex::c_0  (original BF16 input vectors, replayed by reader)
//   Reads from  tt::CBIndex::c_3  (BF16 dequantized centroids from polarquant_kernel)
//   Writes to   tt::CBIndex::c_16 (packed QJL bits + BF16 r_norm)
//
// Per-vector output layout in tt::CBIndex::c_16:
//   [ceil(k/8) bytes : 1-bit sign projections, bit j at byte j/8 bit j%8]
//   [2 bytes         : r_norm as BF16]
//
// Compile-time defines: TQ_DIM, TQ_QJL_DIM, TQ_QJL_SEED, TQ_ROTATION_SEED

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "codebooks.h"



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

void kernel_main() {
    // QJL real math
    constexpr uint32_t kD       = TQ_DIM;
    constexpr uint32_t kK       = TQ_QJL_DIM;
    constexpr uint32_t kQjlBytes = (kK + 7u) / 8u;
    constexpr uint32_t kRecBytes3 = kQjlBytes + 2u;
    constexpr float    kInvSqrtD = 0.088388348f;  // 1/sqrt(128)
    constexpr uint32_t kKnuth   = 2654435761u;
    constexpr uint32_t kQjlSeed = TQ_QJL_SEED;

    uint32_t src_addr  = get_arg_val<uint32_t>(0);
    uint32_t cent_addr = get_arg_val<uint32_t>(4);
    uint32_t vec_idx   = get_arg_val<uint32_t>(5);
    uint32_t out_addr  = get_arg_val<uint32_t>(6);
    uint32_t out_stride = get_arg_val<uint32_t>(7);

    const uint32_t bpv = kD * sizeof(uint16_t);

    const InterleavedAddrGen<true> src_gen  = {
        .bank_base_address = src_addr,  .page_size = bpv };
    const InterleavedAddrGen<true> cent_gen = {
        .bank_base_address = cent_addr, .page_size = bpv };
    const InterleavedAddrGen<true> out_gen  = {
        .bank_base_address = out_addr,  .page_size = out_stride };

    // Diagonal sign vector (matches rotation_kernel: static table from codebooks.h)
    static const int8_t* D_table = TURBO_SIGN_DIAGONAL(TQ_DIM);

    // L1 scratch buffers
    static volatile float buf[kD];      // working buf for WHT (volatile per rotation_kernel pattern)
    static float r_unit[kD];            // unit residual

    // ----- Read input x into c_0 -----
    cb_reserve_back(tt::CBIndex::c_0, 1);
    noc_async_read(get_noc_addr(vec_idx, src_gen),
                   get_write_ptr(tt::CBIndex::c_0), bpv);
    noc_async_read_barrier();
    cb_push_back(tt::CBIndex::c_0, 1);
    cb_wait_front(tt::CBIndex::c_0, 1);
    volatile uint16_t* xp =
        reinterpret_cast<volatile uint16_t*>(get_read_ptr(tt::CBIndex::c_0));

    // ----- Read centroid (rotated-space) into buf via c_in2 -----
    // We don't actually need a CB for this; read directly to buf via L1.
    // But noc_async_read needs an L1 destination address, not a stack array address.
    // Use c_in2 as a scratch CB to land the centroid bytes.
    cb_reserve_back(tt::CBIndex::c_2, 1);
    noc_async_read(get_noc_addr(vec_idx, cent_gen),
                   get_write_ptr(tt::CBIndex::c_2), bpv);
    noc_async_read_barrier();
    cb_push_back(tt::CBIndex::c_2, 1);
    cb_wait_front(tt::CBIndex::c_2, 1);
    volatile uint16_t* cp =
        reinterpret_cast<volatile uint16_t*>(get_read_ptr(tt::CBIndex::c_2));

    // Convert centroid to float in buf
    for (uint32_t i = 0u; i < kD; ++i) {
        buf[i] = bf16_to_float(cp[i]);
    }

    // Inverse rotation: x_hat = D * WHT(c) / sqrt(d)
    // wht_inplace already divides by sqrt(d) at the end.
    wht_inplace(const_cast<float*>(buf), kD);

    // Compute residual r = x - x_hat (where x_hat[i] = D[i] * buf[i]), and ||r||^2
    float nsq = 0.0f;
    for (uint32_t i = 0u; i < kD; ++i) {
        float xi    = bf16_to_float(xp[i]);
        float xhati = (float)D_table[i] * buf[i];
        float ri    = xi - xhati;
        r_unit[i]   = ri;     // stash residual; normalize in-place below
        nsq        += ri * ri;
    }
    float r_norm = fast_sqrt(nsq);
    float inv_rn = (r_norm > 0.0f) ? (1.0f / r_norm) : 0.0f;
    for (uint32_t i = 0u; i < kD; ++i) r_unit[i] *= inv_rn;

    cb_pop_front(tt::CBIndex::c_0, 1);
    cb_pop_front(tt::CBIndex::c_2, 1);

    // ----- 128 QJL sign projections -----
    cb_reserve_back(tt::CBIndex::c_16, 1);
    volatile uint8_t* op =
        reinterpret_cast<volatile uint8_t*>(get_write_ptr(tt::CBIndex::c_16));

    // Zero the output region first (sign bytes + norm bytes)
    for (uint32_t i = 0u; i < kRecBytes3; ++i) op[i] = 0u;

    for (uint32_t j = 0u; j < kK; ++j) {
        uint32_t row_seed = kQjlSeed ^ (j * kKnuth);
        float dot = sign_dot(row_seed, r_unit, kD);
        if (dot > 0.0f) {
            // Set bit j in op[j/8]
            op[j >> 3u] |= (uint8_t)(1u << (j & 7u));
        }
    }

    // Write r_norm as bf16 right after the 16 sign bytes
    volatile uint16_t* np = reinterpret_cast<volatile uint16_t*>(op + kQjlBytes);
    *np = float_to_bf16(r_norm);

    // ----- NOC write to DRAM, padded stride -----
    noc_async_write(get_write_ptr(tt::CBIndex::c_16),
                    get_noc_addr(vec_idx, out_gen),
                    kRecBytes3);
    noc_async_write_barrier();
    cb_push_back(tt::CBIndex::c_16, 1);
}

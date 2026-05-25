// kernels/rotation_kernel.cpp
//
// TT-Metalium compute kernel (Trisc path): Walsh-Hadamard Transform +
// diagonal sign flip.  Implements Π = WHT ∘ D where D is a ±1 diagonal
// derived from TQ_ROTATION_SEED via a Galois LFSR.
//
// Pipeline:
//   Reads from  tt::CBIndex::c_0  (input BF16 vectors from GDDR6)
//   Writes to   tt::CBIndex::c_1  (rotated BF16 vectors)
//
// Compile-time defines (set by the host via ComputeConfig::compile_args
// and target_compile_definitions in CMakeLists.txt):
//   TQ_DIM           — head dimension d (64 / 128 / 256)
//   TQ_ROTATION_SEED — uint32 seed for diagonal D

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "codebooks.h"

// ---------------------------------------------------------------------------
// LFSR sign-diagonal expansion
// Galois LFSR, polynomial x^32 + x^22 + x^2 + x + 1 (tap mask 0x80200003).
// Same polynomial used in qjl_kernel.cpp — keep in sync.
// ---------------------------------------------------------------------------

// Sign diagonal loaded from precomputed table in codebooks.h

// ---------------------------------------------------------------------------
// In-place WHT — log2(d) butterfly stages, then divide by sqrt(d)
// ---------------------------------------------------------------------------

static void wht_inplace(float* buf, uint32_t d) {
    uint32_t h = 1u;
    while (h < d) {
        uint32_t h2 = h * 2u;
        for (uint32_t i = 0u; i < d; i += h2) {
            for (uint32_t j = 0u; j < h; ++j) {
                float u = buf[i + j];
                float v = buf[i + j + h];
                buf[i + j]     = u + v;
                buf[i + j + h] = u - v;
            }
        }
        h = h2;
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



void kernel_main() {
    constexpr uint32_t kD    = TQ_DIM;
    constexpr uint32_t kN    = 32u;   // kTileVectors — must match host
    constexpr uint32_t kSeed = TQ_ROTATION_SEED;

    // Load precomputed sign diagonal
    static const int8_t* D_table = TURBO_SIGN_DIAGONAL(TQ_DIM);
    static float D[kD];
    static volatile float buf[kD];
    for (uint32_t i = 0; i < kD; ++i) D[i] = (float)D_table[i];


    // Runtime args: src_addr, num_vectors, dim, vectors_per_tile
    uint32_t src_addr    = get_arg_val<uint32_t>(0);
    uint32_t num_vectors = get_arg_val<uint32_t>(1);
    uint32_t num_tiles   = (num_vectors + kN - 1) / kN;

    const uint32_t bytes_per_vec  = kD * sizeof(uint16_t);
    const uint32_t bytes_per_tile = kN * bytes_per_vec;

    const InterleavedAddrGen<true> src_gen = {
        .bank_base_address = src_addr,
        .page_size         = bytes_per_vec,
    };

    uint32_t vec_idx = 0;
    for (uint32_t t = 0u; t < num_tiles; ++t) {
        uint32_t vecs_this_tile = ((vec_idx + kN) <= num_vectors)
                                  ? kN : (num_vectors - vec_idx);

        // Read tile from DRAM into c_0
        cb_reserve_back(tt::CBIndex::c_0, 1);
        uint32_t l1_in = get_write_ptr(tt::CBIndex::c_0);
        for (uint32_t v = 0; v < vecs_this_tile; ++v) {
            uint64_t noc_src = get_noc_addr(vec_idx + v, src_gen);
            noc_async_read(noc_src, l1_in + v * bytes_per_vec, bytes_per_vec);
        }
        noc_async_read_barrier();
        cb_push_back(tt::CBIndex::c_0, 1);

        // Now process
        cb_wait_front(tt::CBIndex::c_0, 1);
        cb_reserve_back(tt::CBIndex::c_1, 1);

        volatile uint16_t* in_ptr =
            reinterpret_cast<volatile uint16_t*>(get_read_ptr(tt::CBIndex::c_0));
        volatile uint16_t* out_ptr =
            reinterpret_cast<volatile uint16_t*>(get_write_ptr(tt::CBIndex::c_1));

        for (uint32_t v = 0u; v < vecs_this_tile; ++v) {
            const uint32_t base = v * kD;
            // Use out_ptr L1 region as float scratch BEFORE writing bf16 results
            // out_ptr is kN*kD*2 bytes = 8192 bytes = room for 2048 floats
            // Apply D, WHT, normalize using static buf
            for (uint32_t i = 0u; i < kD; ++i)
                buf[i] = bf16_to_float(in_ptr[base + i]) * (float)D[i];
            for (uint32_t h = 1u; h < kD; h <<= 1u) {
                uint32_t h2 = h * 2u;
                for (uint32_t i = 0u; i < kD; i += h2) {
                    for (uint32_t j = 0u; j < h; ++j) {
                        float a = buf[i + j];
                        float b = buf[i + j + h];
                        buf[i + j]     = a + b;
                        buf[i + j + h] = a - b;
                    }
                }
            }
            constexpr float inv_sqrt_d = 0.088388348f;
            for (uint32_t i = 0u; i < kD; ++i)
                out_ptr[base + i] = float_to_bf16(buf[i] * inv_sqrt_d);
        }

        cb_pop_front(tt::CBIndex::c_0, 1);
        cb_push_back(tt::CBIndex::c_1, 1);
        vec_idx += vecs_this_tile;
    }
}


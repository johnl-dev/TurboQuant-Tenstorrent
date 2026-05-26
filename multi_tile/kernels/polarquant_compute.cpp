// polarquant_compute.cpp
//
// Trisc SFPU compute kernel for Stage 1 (PolarQuant) in tile form.
//
// Math (per element x):
//   cent_out(x) = c[0] + sum_{j=0..14} (c[j+1] - c[j]) * (x >= t[j])
//
// Equivalent to: find bucket j such that t[j-1] <= x < t[j], output c[j].
// Expressed as a sum of weighted threshold indicators so we can implement it
// with SFPU comparison + multiply + accumulate, no gather needed.
//
// The 16 centroids and 15 boundaries for b=4, d=128 are baked in as
// compile-time float constants (taken from include/codebooks.h).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"

// ---------------------------------------------------------------------------
// Codebook constants (b=4, d=128) -- copy of values from include/codebooks.h
// ---------------------------------------------------------------------------
constexpr float kC[16] = {
    -0.2372315346f, -0.1786289781f, -0.1392336099f, -0.1078747771f,
    -0.0809901680f, -0.0565789503f, -0.0335905689f, -0.0115984621f,
     0.0100779327f,  0.0317803511f,  0.0544547562f,  0.0788434907f,
     0.1055684838f,  0.1363609158f,  0.1748946473f,  0.2311842135f
};

constexpr float kT[15] = {
    -0.2079302564f, -0.1589312940f, -0.1235541935f, -0.0944324726f,
    -0.0687845591f, -0.0450847596f, -0.0225945155f, -0.0007602647f,
     0.0209291419f,  0.0431175537f,  0.0666491235f,  0.0922059872f,
     0.1209646998f,  0.1556277815f,  0.2030394304f
};

// Precomputed deltas: kCDelta[j] = kC[j+1] - kC[j]
constexpr float kCDelta[15] = {
    kC[1]  - kC[0],  kC[2]  - kC[1],  kC[3]  - kC[2],  kC[4]  - kC[3],
    kC[5]  - kC[4],  kC[6]  - kC[5],  kC[7]  - kC[6],  kC[8]  - kC[7],
    kC[9]  - kC[8],  kC[10] - kC[9],  kC[11] - kC[10], kC[12] - kC[11],
    kC[13] - kC[12], kC[14] - kC[13], kC[15] - kC[14]
};

// Reinterpret float bit pattern as uint32 (for SFPU scalar params)
static constexpr uint32_t f2u(float f) {
    // constexpr __builtin_bit_cast — supported on this toolchain
    return __builtin_bit_cast(uint32_t, f);
}

void kernel_main() {
    constexpr uint32_t Mt = get_compile_time_arg_val(0);  // tile rows
    constexpr uint32_t Dt = get_compile_time_arg_val(1);  // d / 32 = 4

    constexpr auto cb_x    = tt::CBIndex::c_0;   // input rotated tiles
    constexpr auto cb_cent = tt::CBIndex::c_16;  // output centroid tiles

    // DST register slots:
    //   slot 0: accumulator
    //   slot 1: working comparison/multiply tile
    constexpr uint32_t DST_ACC = 0;
    constexpr uint32_t DST_WORK = 1;

    init_sfpu(cb_x, cb_cent);
    unary_ge_tile_init();
    binop_with_scalar_tile_init();
    add_binary_tile_init();

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t dt = 0; dt < Dt; ++dt) {
            cb_wait_front(cb_x, 1);
            cb_reserve_back(cb_cent, 1);

            tile_regs_acquire();

            // Initialize accumulator = c[0] everywhere:
            //   copy x into DST_ACC, zero it, then add c[0].
            copy_tile(cb_x, 0, DST_ACC);
            mul_unary_tile(DST_ACC, f2u(0.0f));
            add_unary_tile(DST_ACC, f2u(kC[0]));

            // For each threshold j: acc += c_delta[j] * (x >= t[j])
            for (uint32_t j = 0; j < 15; ++j) {
                copy_tile(cb_x, 0, DST_WORK);
                unary_ge_tile(DST_WORK, f2u(kT[j]));
                mul_unary_tile(DST_WORK, f2u(kCDelta[j]));
                add_binary_tile(DST_ACC, DST_WORK, DST_ACC);  // acc += work
            }

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(DST_ACC, cb_cent);
            cb_push_back(cb_cent, 1);
            tile_regs_release();

            cb_pop_front(cb_x, 1);
        }
    }
}

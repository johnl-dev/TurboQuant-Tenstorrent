// rotation_compute.cpp
//
// Trisc compute kernel for Stage 0 (rotation) in tile form.
//
// Math:  Z = X @ H_combined
//   where H_combined = (1/sqrt(d)) * D_diag @ H_d   (precomputed by host)
//
// X shape: (Mt*32, Kt*32) = (N, 128)        -> Mt * Kt tiles
// H shape: (Kt*32, Nt*32) = (128, 128)      -> Kt * Nt tiles
// Z shape: (Mt*32, Nt*32) = (N, 128)        -> Mt * Nt tiles
//
// For N=32 (the initial test case): Mt=1, Kt=4, Nt=4.
//
// Outer-product matmul: for each output tile (mt, nt), accumulate over kt
// the product X[mt,kt] @ H[kt,nt]. Reader feeds (X, H) tile-pairs in the
// inner loop order; this kernel just does the matmul over Kt iterations.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"

void kernel_main() {
    constexpr uint32_t Mt = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Nt = get_compile_time_arg_val(2);

    constexpr auto cb_x = tt::CBIndex::c_0;   // input X tiles (from reader)
    constexpr auto cb_h = tt::CBIndex::c_1;   // input H tiles (from reader)
    constexpr auto cb_z = tt::CBIndex::c_16;  // output Z tiles (to writer)

    mm_init(cb_x, cb_h, cb_z);

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            tile_regs_acquire();

            for (uint32_t kt = 0; kt < Kt; ++kt) {
                cb_wait_front(cb_x, 1);
                cb_wait_front(cb_h, 1);

                matmul_tiles(cb_x, cb_h, 0, 0, 0);

                cb_pop_front(cb_x, 1);
                cb_pop_front(cb_h, 1);
            }

            tile_regs_commit();
            tile_regs_wait();

            cb_reserve_back(cb_z, 1);
            pack_tile(0, cb_z);
            cb_push_back(cb_z, 1);

            tile_regs_release();
        }
    }
}


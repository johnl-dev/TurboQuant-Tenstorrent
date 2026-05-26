// inverse_rotation_compute.cpp
//
// Trisc compute kernel for Stage 2a (inverse rotation) in tile form.
//
// Math:  X_hat = Cent @ H_inv
//   where H_inv = (1/sqrt(d)) * H_d * diag(D)   (precomputed by host)
//
// Same matmul-form as Stage 0. Only the precomputed matrix differs.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"

void kernel_main() {
    constexpr uint32_t Mt = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Nt = get_compile_time_arg_val(2);

    constexpr auto cb_cent = tt::CBIndex::c_0;
    constexpr auto cb_h    = tt::CBIndex::c_1;
    constexpr auto cb_xhat = tt::CBIndex::c_16;

    mm_init(cb_cent, cb_h, cb_xhat);

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            tile_regs_acquire();

            for (uint32_t kt = 0; kt < Kt; ++kt) {
                cb_wait_front(cb_cent, 1);
                cb_wait_front(cb_h, 1);

                matmul_tiles(cb_cent, cb_h, 0, 0, 0);

                cb_pop_front(cb_cent, 1);
                cb_pop_front(cb_h, 1);
            }

            tile_regs_commit();
            tile_regs_wait();

            cb_reserve_back(cb_xhat, 1);
            pack_tile(0, cb_xhat);
            cb_push_back(cb_xhat, 1);

            tile_regs_release();
        }
    }
}

// residual_compute.cpp
//
// Trisc compute kernel for Stage 2b1 (residual).
// Math: r = x - x_hat   (elementwise per tile)

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"

void kernel_main() {
    constexpr uint32_t Mt = get_compile_time_arg_val(0);
    constexpr uint32_t Dt = get_compile_time_arg_val(1);

    constexpr auto cb_x    = tt::CBIndex::c_0;
    constexpr auto cb_xhat = tt::CBIndex::c_1;
    constexpr auto cb_r    = tt::CBIndex::c_16;

    binary_op_init_common(cb_x, cb_xhat, cb_r);
    sub_tiles_init(cb_x, cb_xhat);

    uint32_t total = Mt * Dt;
    for (uint32_t i = 0; i < total; ++i) {
        cb_wait_front(cb_x,    1);
        cb_wait_front(cb_xhat, 1);

        tile_regs_acquire();
        sub_tiles(cb_x, cb_xhat, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_r, 1);
        pack_tile(0, cb_r);
        cb_push_back(cb_r, 1);
        tile_regs_release();

        cb_pop_front(cb_x,    1);
        cb_pop_front(cb_xhat, 1);
    }
}

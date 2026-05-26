// rsquare_compute.cpp
// Trisc: r_sq = r * r (elementwise)

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"

void kernel_main() {
    constexpr uint32_t Mt = get_compile_time_arg_val(0);
    constexpr uint32_t Dt = get_compile_time_arg_val(1);

    constexpr auto cb_r    = tt::CBIndex::c_0;
    constexpr auto cb_rsq  = tt::CBIndex::c_16;

    // mul_tiles uses two CB inputs; we pass cb_r twice. binary_op_init_common
    // takes (input0_cb, input1_cb, output_cb); with the same CB twice it just
    // reads the front tile of cb_r as both operands of the multiply.
    binary_op_init_common(cb_r, cb_r, cb_rsq);
    mul_tiles_init(cb_r, cb_r);

    uint32_t total = Mt * Dt;
    for (uint32_t i = 0; i < total; ++i) {
        cb_wait_front(cb_r, 1);

        tile_regs_acquire();
        mul_tiles(cb_r, cb_r, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_rsq, 1);
        pack_tile(0, cb_rsq);
        cb_push_back(cb_rsq, 1);
        tile_regs_release();

        cb_pop_front(cb_r, 1);
    }
}

// rnorm_matmul_compute.cpp
// Trisc: r_norm_sq = r_sq @ ones_d_col   ;   r_norm = sqrt(r_norm_sq)
//
// Output is (Mt, 1) tiles: per Mt, one output tile whose column 0 holds
// the r_norm values for the 32 vectors in that tile.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_unary/sqrt.h"

void kernel_main() {
    constexpr uint32_t Mt = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);  // = Dt of input r_sq

    constexpr auto cb_rsq  = tt::CBIndex::c_0;
    constexpr auto cb_ones = tt::CBIndex::c_1;
    constexpr auto cb_norm = tt::CBIndex::c_16;

    mm_init(cb_rsq, cb_ones, cb_norm);
    sqrt_tile_init();

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        tile_regs_acquire();

        // Accumulate K (=Dt) matmul updates into DST[0]
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            cb_wait_front(cb_rsq,  1);
            cb_wait_front(cb_ones, 1);
            matmul_tiles(cb_rsq, cb_ones, 0, 0, 0);
            cb_pop_front(cb_rsq,  1);
            cb_pop_front(cb_ones, 1);
        }

        // sqrt on DST[0]
        sqrt_tile(0);

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_norm, 1);
        pack_tile(0, cb_norm);
        cb_push_back(cb_norm, 1);
        tile_regs_release();
    }
}

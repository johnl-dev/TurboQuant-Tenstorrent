// qjl_project_compute.cpp
//
// Trisc compute kernel for Stage 2b2 (QJL sign projection).
// Math: B = R @ S_T   where S_T is (d, k) bf16, prebuilt from LFSR on host.
//
// Same structure as rotation_compute.cpp / inverse_rotation_compute.cpp.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"

void kernel_main() {
    constexpr uint32_t Mt = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);  // = Dt of input r
    constexpr uint32_t Nt = get_compile_time_arg_val(2);  // = Kt of QJL output

    constexpr auto cb_r = tt::CBIndex::c_0;
    constexpr auto cb_s = tt::CBIndex::c_1;
    constexpr auto cb_b = tt::CBIndex::c_16;

    mm_init(cb_r, cb_s, cb_b);

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                cb_wait_front(cb_r, 1);
                cb_wait_front(cb_s, 1);
                matmul_tiles(cb_r, cb_s, 0, 0, 0);
                cb_pop_front(cb_r, 1);
                cb_pop_front(cb_s, 1);
            }
            tile_regs_commit();
            tile_regs_wait();

            cb_reserve_back(cb_b, 1);
            pack_tile(0, cb_b);
            cb_push_back(cb_b, 1);
            tile_regs_release();
        }
    }
}

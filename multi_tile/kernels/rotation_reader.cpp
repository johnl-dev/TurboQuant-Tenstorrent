// rotation_reader.cpp
//
// Brisc data-movement reader for Stage 0.
//
// Feeds the compute kernel one (X_tile, H_tile) pair per Kt step, in the
// order required by the compute kernel's outer-product matmul loop:
//   for mt in 0..Mt:
//     for nt in 0..Nt:
//       for kt in 0..Kt:
//         push X[mt, kt]    (re-read for each nt, since DRAM is the source of truth)
//         push H[kt, nt]
//
// This is the canonical "reuse-X-across-nt" pattern but with X re-read from
// DRAM each time. A more sophisticated reader would buffer X in L1 once per
// mt; we'll add that optimization later.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t x_addr = get_arg_val<uint32_t>(0);
    uint32_t h_addr = get_arg_val<uint32_t>(1);
    uint32_t Mt     = get_arg_val<uint32_t>(2);
    uint32_t Kt     = get_arg_val<uint32_t>(3);
    uint32_t Nt     = get_arg_val<uint32_t>(4);

    constexpr auto cb_x = tt::CBIndex::c_0;
    constexpr auto cb_h = tt::CBIndex::c_1;

    constexpr uint32_t tile_bytes = 2048;  // 32 * 32 * sizeof(bf16)

    const InterleavedAddrGenFast<true> x_gen = {
        .bank_base_address = x_addr,
        .page_size         = tile_bytes,
        .data_format       = DataFormat::Float16_b,
    };
    const InterleavedAddrGenFast<true> h_gen = {
        .bank_base_address = h_addr,
        .page_size         = tile_bytes,
        .data_format       = DataFormat::Float16_b,
    };

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                // X[mt, kt] — page index = mt*Kt + kt
                cb_reserve_back(cb_x, 1);
                uint32_t x_l1 = get_write_ptr(cb_x);
                noc_async_read_tile(mt * Kt + kt, x_gen, x_l1);

                // H[kt, nt] — page index = kt*Nt + nt
                cb_reserve_back(cb_h, 1);
                uint32_t h_l1 = get_write_ptr(cb_h);
                noc_async_read_tile(kt * Nt + nt, h_gen, h_l1);

                noc_async_read_barrier();

                cb_push_back(cb_x, 1);
                cb_push_back(cb_h, 1);
            }
        }
    }
}

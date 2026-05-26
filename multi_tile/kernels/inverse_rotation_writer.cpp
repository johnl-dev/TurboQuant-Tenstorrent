// inverse_rotation_writer.cpp
//
// NCRISC writer for Stage 2a. Drains output tiles to DRAM.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t xhat_addr = get_arg_val<uint32_t>(0);
    uint32_t Mt        = get_arg_val<uint32_t>(1);
    uint32_t Nt        = get_arg_val<uint32_t>(2);

    constexpr auto cb_xhat = tt::CBIndex::c_16;
    constexpr uint32_t tile_bytes = 2048;

    const InterleavedAddrGenFast<true> xhat_gen = {
        .bank_base_address = xhat_addr,
        .page_size         = tile_bytes,
        .data_format       = DataFormat::Float16_b,
    };

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            cb_wait_front(cb_xhat, 1);
            noc_async_write_tile(mt * Nt + nt, xhat_gen, get_read_ptr(cb_xhat));
            noc_async_write_barrier();
            cb_pop_front(cb_xhat, 1);
        }
    }
}

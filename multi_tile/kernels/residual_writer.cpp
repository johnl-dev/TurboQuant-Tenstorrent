// residual_writer.cpp
//
// NCRISC writer: drains cb_r tiles to DRAM.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t r_addr = get_arg_val<uint32_t>(0);
    uint32_t Mt     = get_arg_val<uint32_t>(1);
    uint32_t Dt     = get_arg_val<uint32_t>(2);

    constexpr auto cb_r = tt::CBIndex::c_16;
    constexpr uint32_t tile_bytes = 2048;

    const InterleavedAddrGenFast<true> r_gen = {
        .bank_base_address = r_addr,
        .page_size         = tile_bytes,
        .data_format       = DataFormat::Float16_b,
    };

    uint32_t total = Mt * Dt;
    for (uint32_t i = 0; i < total; ++i) {
        cb_wait_front(cb_r, 1);
        noc_async_write_tile(i, r_gen, get_read_ptr(cb_r));
        noc_async_write_barrier();
        cb_pop_front(cb_r, 1);
    }
}

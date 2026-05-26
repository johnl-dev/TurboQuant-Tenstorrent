// polarquant_reader.cpp
//
// Brisc reader for Stage 1. Streams (Mt * Dt) tiles of rotated input X from
// DRAM into cb_x in tile-row-major order (mt outer, dt inner).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t x_addr = get_arg_val<uint32_t>(0);
    uint32_t Mt     = get_arg_val<uint32_t>(1);
    uint32_t Dt     = get_arg_val<uint32_t>(2);

    constexpr auto cb_x = tt::CBIndex::c_0;
    constexpr uint32_t tile_bytes = 2048;

    const InterleavedAddrGenFast<true> x_gen = {
        .bank_base_address = x_addr,
        .page_size         = tile_bytes,
        .data_format       = DataFormat::Float16_b,
    };

    uint32_t total = Mt * Dt;
    for (uint32_t i = 0; i < total; ++i) {
        cb_reserve_back(cb_x, 1);
        uint32_t l1 = get_write_ptr(cb_x);
        noc_async_read_tile(i, x_gen, l1);
        noc_async_read_barrier();
        cb_push_back(cb_x, 1);
    }
}

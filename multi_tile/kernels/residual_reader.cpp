// residual_reader.cpp
//
// Brisc reader: streams x and x_hat tiles into cb_x and cb_xhat.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t x_addr    = get_arg_val<uint32_t>(0);
    uint32_t xhat_addr = get_arg_val<uint32_t>(1);
    uint32_t Mt        = get_arg_val<uint32_t>(2);
    uint32_t Dt        = get_arg_val<uint32_t>(3);

    constexpr auto cb_x    = tt::CBIndex::c_0;
    constexpr auto cb_xhat = tt::CBIndex::c_1;
    constexpr uint32_t tile_bytes = 2048;

    const InterleavedAddrGenFast<true> x_gen = {
        .bank_base_address = x_addr,
        .page_size         = tile_bytes,
        .data_format       = DataFormat::Float16_b,
    };
    const InterleavedAddrGenFast<true> xhat_gen = {
        .bank_base_address = xhat_addr,
        .page_size         = tile_bytes,
        .data_format       = DataFormat::Float16_b,
    };

    uint32_t total = Mt * Dt;
    for (uint32_t i = 0; i < total; ++i) {
        cb_reserve_back(cb_x,    1);
        cb_reserve_back(cb_xhat, 1);
        noc_async_read_tile(i, x_gen,    get_write_ptr(cb_x));
        noc_async_read_tile(i, xhat_gen, get_write_ptr(cb_xhat));
        noc_async_read_barrier();
        cb_push_back(cb_x,    1);
        cb_push_back(cb_xhat, 1);
    }
}

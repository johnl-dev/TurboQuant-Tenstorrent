#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t rsq_addr  = get_arg_val<uint32_t>(0);
    uint32_t ones_addr = get_arg_val<uint32_t>(1);
    uint32_t Mt        = get_arg_val<uint32_t>(2);
    uint32_t Kt        = get_arg_val<uint32_t>(3);

    constexpr auto cb_rsq  = tt::CBIndex::c_0;
    constexpr auto cb_ones = tt::CBIndex::c_1;
    constexpr uint32_t tile_bytes = 2048;

    const InterleavedAddrGenFast<true> rsq_gen = {
        .bank_base_address = rsq_addr,
        .page_size         = tile_bytes,
        .data_format       = DataFormat::Float16_b,
    };
    const InterleavedAddrGenFast<true> ones_gen = {
        .bank_base_address = ones_addr,
        .page_size         = tile_bytes,
        .data_format       = DataFormat::Float16_b,
    };

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            cb_reserve_back(cb_rsq, 1);
            noc_async_read_tile(mt * Kt + kt, rsq_gen, get_write_ptr(cb_rsq));
            cb_reserve_back(cb_ones, 1);
            // ones has only Kt tiles total (one per K dimension);
            // each is the same all-ones (d=32, 1 col valid) but tile is 32x32.
            noc_async_read_tile(kt, ones_gen, get_write_ptr(cb_ones));
            noc_async_read_barrier();
            cb_push_back(cb_rsq, 1);
            cb_push_back(cb_ones, 1);
        }
    }
}

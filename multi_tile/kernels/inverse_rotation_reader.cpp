// inverse_rotation_reader.cpp
//
// Brisc reader for Stage 2a. Streams (Cent, H_inv) tile pairs in the order
// required by outer-product matmul.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t cent_addr = get_arg_val<uint32_t>(0);
    uint32_t h_addr    = get_arg_val<uint32_t>(1);
    uint32_t Mt        = get_arg_val<uint32_t>(2);
    uint32_t Kt        = get_arg_val<uint32_t>(3);
    uint32_t Nt        = get_arg_val<uint32_t>(4);

    constexpr auto cb_cent = tt::CBIndex::c_0;
    constexpr auto cb_h    = tt::CBIndex::c_1;
    constexpr uint32_t tile_bytes = 2048;

    const InterleavedAddrGenFast<true> cent_gen = {
        .bank_base_address = cent_addr,
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
                cb_reserve_back(cb_cent, 1);
                noc_async_read_tile(mt * Kt + kt, cent_gen, get_write_ptr(cb_cent));
                cb_reserve_back(cb_h, 1);
                noc_async_read_tile(kt * Nt + nt, h_gen, get_write_ptr(cb_h));
                noc_async_read_barrier();
                cb_push_back(cb_cent, 1);
                cb_push_back(cb_h, 1);
            }
        }
    }
}

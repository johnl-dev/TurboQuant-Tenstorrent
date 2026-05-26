// qjl_project_reader.cpp

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t r_addr = get_arg_val<uint32_t>(0);
    uint32_t s_addr = get_arg_val<uint32_t>(1);
    uint32_t Mt     = get_arg_val<uint32_t>(2);
    uint32_t Kt     = get_arg_val<uint32_t>(3);
    uint32_t Nt     = get_arg_val<uint32_t>(4);

    constexpr auto cb_r = tt::CBIndex::c_0;
    constexpr auto cb_s = tt::CBIndex::c_1;
    constexpr uint32_t tile_bytes = 2048;

    const InterleavedAddrGenFast<true> r_gen = {
        .bank_base_address = r_addr,
        .page_size         = tile_bytes,
        .data_format       = DataFormat::Float16_b,
    };
    const InterleavedAddrGenFast<true> s_gen = {
        .bank_base_address = s_addr,
        .page_size         = tile_bytes,
        .data_format       = DataFormat::Float16_b,
    };

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                cb_reserve_back(cb_r, 1);
                noc_async_read_tile(mt * Kt + kt, r_gen, get_write_ptr(cb_r));
                cb_reserve_back(cb_s, 1);
                noc_async_read_tile(kt * Nt + nt, s_gen, get_write_ptr(cb_s));
                noc_async_read_barrier();
                cb_push_back(cb_r, 1);
                cb_push_back(cb_s, 1);
            }
        }
    }
}

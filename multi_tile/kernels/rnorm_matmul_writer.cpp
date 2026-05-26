#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t norm_addr = get_arg_val<uint32_t>(0);
    uint32_t Mt        = get_arg_val<uint32_t>(1);

    constexpr auto cb_norm = tt::CBIndex::c_16;
    constexpr uint32_t tile_bytes = 2048;

    const InterleavedAddrGenFast<true> norm_gen = {
        .bank_base_address = norm_addr,
        .page_size         = tile_bytes,
        .data_format       = DataFormat::Float16_b,
    };

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        cb_wait_front(cb_norm, 1);
        noc_async_write_tile(mt, norm_gen, get_read_ptr(cb_norm));
        noc_async_write_barrier();
        cb_pop_front(cb_norm, 1);
    }
}

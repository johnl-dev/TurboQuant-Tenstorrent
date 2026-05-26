// rotation_writer.cpp
//
// NCRISC data-movement writer for Stage 0.
//
// Pulls one output tile at a time from c_16 and writes to DRAM in
// (tile_row, tile_col) row-major order.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t z_addr = get_arg_val<uint32_t>(0);
    uint32_t Mt     = get_arg_val<uint32_t>(1);
    uint32_t Nt     = get_arg_val<uint32_t>(2);

    constexpr auto cb_z = tt::CBIndex::c_16;

    constexpr uint32_t tile_bytes = 2048;

    const InterleavedAddrGenFast<true> z_gen = {
        .bank_base_address = z_addr,
        .page_size         = tile_bytes,
        .data_format       = DataFormat::Float16_b,
    };

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            cb_wait_front(cb_z, 1);
            uint32_t z_l1 = get_read_ptr(cb_z);
            noc_async_write_tile(mt * Nt + nt, z_gen, z_l1);
            noc_async_write_barrier();
            cb_pop_front(cb_z, 1);
        }
    }
}

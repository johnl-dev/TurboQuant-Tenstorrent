// kernels/dataflow/writer_kernel.cpp
//
// Ncrisc (RISCV_1) data movement kernel.
// Streams packed TurboQuant records from L1 CB::c_out1 → GDDR6.
// Used only in the full-pipeline production path (not in stage-dump mode).
//
// Runtime args:
//   arg[0] = dst_dram_addr     base address of output buffer in GDDR6
//   arg[1] = num_vectors
//   arg[2] = record_bytes      bytes per packed record (kRecordBytes)
//   arg[3] = vectors_per_tile

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr         = get_arg_val<uint32_t>(0);
    uint32_t num_vectors      = get_arg_val<uint32_t>(1);
    uint32_t record_bytes     = get_arg_val<uint32_t>(2);
    uint32_t vectors_per_tile = get_arg_val<uint32_t>(3);

    const InterleavedAddrGen<true> dst_gen = {
        .bank_base_address = dst_addr,
        .page_size         = record_bytes,
    };

    uint32_t vec_idx = 0;
    while (vec_idx < num_vectors) {
        uint32_t vecs_this_tile = ((vec_idx + vectors_per_tile) <= num_vectors)
                                      ? vectors_per_tile
                                      : (num_vectors - vec_idx);

        cb_wait_front(CB::c_out1, 1);
        uint32_t l1_addr = get_read_ptr(CB::c_out1);
        for (uint32_t v = 0; v < vecs_this_tile; ++v) {
            uint64_t noc_dst = get_noc_addr(vec_idx + v, dst_gen);
            noc_async_write(l1_addr + v * record_bytes, noc_dst, record_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(CB::c_out1, 1);

        vec_idx += vecs_this_tile;
    }
}

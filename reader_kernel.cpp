// kernels/dataflow/reader_kernel.cpp
//
// Brisc (RISCV_0) data movement kernel.
// Streams BF16 input vectors from GDDR6 → L1 CB::c_in0.
// Also replays the same input tile into CB::c_in0 again before the QJL
// stage, so qjl_kernel can read x alongside the centroids from CB::c_in3.
//
// Runtime args:
//   arg[0] = src_dram_addr      base address of input buffer in GDDR6
//   arg[1] = num_vectors        total number of BF16 vectors
//   arg[2] = dim                d, elements per vector
//   arg[3] = vectors_per_tile   kTileVectors

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr         = get_arg_val<uint32_t>(0);
    uint32_t num_vectors      = get_arg_val<uint32_t>(1);
    uint32_t dim              = get_arg_val<uint32_t>(2);
    uint32_t vectors_per_tile = get_arg_val<uint32_t>(3);

    const uint32_t bytes_per_vector = dim * sizeof(uint16_t);
    const uint32_t bytes_per_tile   = vectors_per_tile * bytes_per_vector;

    const InterleavedAddrGen<true> src_gen = {
        .bank_base_address = src_addr,
        .page_size         = bytes_per_vector,
    };

    uint32_t vec_idx = 0;
    while (vec_idx < num_vectors) {
        uint32_t vecs_this_tile = ((vec_idx + vectors_per_tile) <= num_vectors)
                                      ? vectors_per_tile
                                      : (num_vectors - vec_idx);

        // Feed CB::c_in0 for rotation_kernel
        cb_reserve_back(CB::c_in0, 1);
        uint32_t l1_addr = get_write_ptr(CB::c_in0);
        for (uint32_t v = 0; v < vecs_this_tile; ++v) {
            uint64_t noc_src = get_noc_addr(vec_idx + v, src_gen);
            noc_async_read(noc_src,
                           l1_addr + v * bytes_per_vector,
                           bytes_per_vector);
        }
        noc_async_read_barrier();
        cb_push_back(CB::c_in0, 1);

        vec_idx += vecs_this_tile;
    }
}

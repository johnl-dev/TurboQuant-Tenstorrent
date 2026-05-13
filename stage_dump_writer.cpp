// kernels/dataflow/stage_dump_writer.cpp
//
// Ncrisc (RISCV_1) data movement kernel for per-stage validation dumps.
//
// Reads raw bytes from a configurable CB and writes them to a GDDR6 debug
// buffer so the host can read them back and compare against the Python
// reference implementation.
//
// Compile-time defines (set per-stage via DataMovementConfig::defines):
//   TQ_DUMP_CB_ID      CB enum integer value to capture
//   TQ_DUMP_PAGE_BYTES byte size of one page in that CB
//
// Runtime args:
//   arg[0] = dst_dram_addr
//   arg[1] = num_pages      (= num_tiles)
//   arg[2] = page_bytes     (must equal TQ_DUMP_PAGE_BYTES)

#include <stdint.h>
#include "dataflow_api.h"

#ifndef TQ_DUMP_CB_ID
#  define TQ_DUMP_CB_ID 1          // default: CB::c_in1 (rotated)
#endif
#ifndef TQ_DUMP_PAGE_BYTES
#  define TQ_DUMP_PAGE_BYTES 8192  // 32 vectors × 128 × 2 bytes
#endif

void kernel_main() {
    uint32_t dst_addr   = get_arg_val<uint32_t>(0);
    uint32_t num_pages  = get_arg_val<uint32_t>(1);
    uint32_t page_bytes = get_arg_val<uint32_t>(2);

    const InterleavedAddrGen<true> dst_gen = {
        .bank_base_address = dst_addr,
        .page_size         = page_bytes,
    };

    for (uint32_t pg = 0; pg < num_pages; ++pg) {
        cb_wait_front(TQ_DUMP_CB_ID, 1);
        uint32_t l1_addr = get_read_ptr(TQ_DUMP_CB_ID);

        uint64_t noc_dst = get_noc_addr(pg, dst_gen);
        noc_async_write(l1_addr, noc_dst, page_bytes);
        noc_async_write_barrier();

        cb_pop_front(TQ_DUMP_CB_ID, 1);
    }
}

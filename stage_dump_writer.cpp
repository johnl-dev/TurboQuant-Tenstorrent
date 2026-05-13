/**
 * kernels/dataflow/stage_dump_writer.cpp
 * ========================================
 * Ncrisc (RISCV_1) data movement kernel for per-stage debug dumps.
 *
 * Reads from a configurable CB ID and writes the raw bytes to a GDDR6
 * debug buffer.  Used only when TQ_DUMP_STAGE is defined at compile time.
 *
 * This kernel replaces writer_kernel.cpp when running in per-stage
 * validation mode.  The host program selects the correct CB to dump
 * by passing the CB ID as a compile-time define.
 *
 * Compile-time defines:
 *   TQ_DUMP_CB_ID      = which CB to read (e.g. kCbRotated=1, kCbQuant=2)
 *   TQ_DUMP_PAGE_BYTES = byte size of one page in that CB
 *
 * Runtime args:
 *   arg[0] = dst_dram_addr
 *   arg[1] = num_pages      (= num_tiles = ceil(N / kTileVectors))
 *   arg[2] = page_bytes
 */

#include "dataflow_api.h"

#ifndef TQ_DUMP_CB_ID
#  define TQ_DUMP_CB_ID 1          // default: dump rotated CB
#endif
#ifndef TQ_DUMP_PAGE_BYTES
#  define TQ_DUMP_PAGE_BYTES 8192  // 32 vectors × 128 × 2 bytes
#endif

void kernel_main() {
    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t num_pages = get_arg_val<uint32_t>(1);
    uint32_t page_bytes = get_arg_val<uint32_t>(2);

    const InterleavedAddrGen<true> dst_gen = {
        .bank_base_address = dst_addr,
        .page_size         = page_bytes,
    };

    for (uint32_t pg = 0; pg < num_pages; ++pg) {
        cb_wait_front(TQ_DUMP_CB_ID, 1);
        uint32_t l1_read_addr = get_read_ptr(TQ_DUMP_CB_ID);

        uint64_t noc_addr = get_noc_addr(pg, dst_gen);
        noc_async_write(l1_read_addr, noc_addr, page_bytes);
        noc_async_write_barrier();

        cb_pop_front(TQ_DUMP_CB_ID, 1);
    }
}

/**
 * host/turboquant_program.cpp
 * ============================
 * Host-side TT-Metal program for Phase 1: single-core TurboQuant.
 *
 * Runs kernels in isolated single-stage programs so each stage's raw CB
 * output is flushed to GDDR6 and read back as a named binary file.
 * This is what makes the Python per-stage validation tests meaningful:
 * each test file loads its corresponding dump and compares it against the
 * NumPy reference implementation independently.
 *
 * Dump files produced (always, in --mode full):
 *   dump_input.bin           — FP16 input vectors     (N × d × 2 bytes)
 *   dump_rotated.bin         — FP16 rotated vectors    (N × d × 2 bytes)
 *   dump_quant_indices.bin   — packed b-bit indices    (N × ceil(d*b/8) bytes)
 *   dump_quant_centroids.bin — FP16 dequantized cents  (N × d × 2 bytes)
 *   dump_qjl.bin             — QJL bits + r_norm fp16  (N × (ceil(k/8)+2) bytes)
 *   dump_output.bin          — final packed records    (N × kRecordBytes bytes)
 *
 * Stage isolation strategy:
 *   Each stage is a separate tt::tt_metal::Program dispatched in sequence.
 *   The output CB of each program is wired to a stage_dump_writer that
 *   sends its contents to a GDDR6 debug buffer.  After Finish(), the host
 *   reads that buffer back and writes the dump file.
 *
 *   This avoids the "only one RISCV_1 writer kernel per program" constraint
 *   when dumping mid-pipeline CBs.
 *
 * Build:
 *   g++ host/turboquant_program.cpp \
 *       -I include \
 *       -I $TT_METAL_HOME/tt_metal/api \
 *       -I $TT_METAL_HOME/tt_metal \
 *       -L $TT_METAL_HOME/build/lib \
 *       -ltt_metal -ldevice \
 *       -DTQ_DIM=128 -DTQ_BITS=4 -DTQ_QJL_DIM=128 \
 *       -std=c++17 -O2 -g \
 *       -o turboquant_single_core
 *
 * Usage:
 *   ./turboquant_single_core [--num-vectors 64] [--bits 4] [--mode full]
 *
 * Validate (after running):
 *   python tests/test_rotation.py   --device
 *   python tests/test_polarquant.py --device
 *   python tests/test_qjl.py        --device
 *   python tests/test_roundtrip.py  --device
 */

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <chrono>
#include <iostream>
#include <fstream>
#include <stdexcept>

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/core_coord.h"
#include "tt_metal/impl/device/device.hpp"
#include "turboquant_layout.h"

using namespace tt;
using namespace tt::tt_metal;

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

static uint16_t float_to_fp16_host(float v) {
    uint32_t f32; memcpy(&f32, &v, 4);
    uint32_t sign = (f32 >> 31) & 1u, exp32 = (f32 >> 23) & 0xFFu, mant = f32 & 0x7FFFFFu;
    if (exp32 == 255u) return static_cast<uint16_t>((sign<<15)|0x7C00u|(mant?0x200u:0u));
    int32_t exp16 = static_cast<int32_t>(exp32) - 127 + 15;
    if (exp16 >= 31) return static_cast<uint16_t>((sign<<15)|0x7C00u);
    if (exp16 <= 0) {
        uint32_t sh = static_cast<uint32_t>(1 - exp16);
        if (sh > 24) return static_cast<uint16_t>(sign << 15);
        mant = (0x800000u | mant) >> sh;
        return static_cast<uint16_t>((sign<<15)|(mant>>13));
    }
    return static_cast<uint16_t>((sign<<15)|(static_cast<uint32_t>(exp16)<<10)|(mant>>13));
}

static std::vector<std::vector<float>> gen_test_vectors(uint32_t n, uint32_t d, uint32_t seed=42) {
    uint64_t state = seed;
    auto rnd = [&]() -> float {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        uint32_t b = static_cast<uint32_t>(state >> 33);
        b = (b & 0x7FFFFFu) | 0x3F800000u;
        float f; memcpy(&f, &b, 4); return f - 1.5f;
    };
    std::vector<std::vector<float>> vs(n, std::vector<float>(d));
    for (uint32_t i = 0; i < n; ++i) {
        float nm = 0; for (uint32_t j = 0; j < d; ++j) { vs[i][j]=rnd(); nm+=vs[i][j]*vs[i][j]; }
        nm = std::sqrt(nm); for (uint32_t j = 0; j < d; ++j) vs[i][j] /= nm;
    }
    return vs;
}

static void write_bin(const std::string& path, const void* data, size_t bytes) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path + " for writing");
    f.write(reinterpret_cast<const char*>(data), bytes);
    std::cout << "  wrote " << bytes << " bytes → " << path << "\n";
}

static std::shared_ptr<Buffer> dram_buf(Device* dev, uint32_t total, uint32_t page) {
    InterleavedBufferConfig c{.device=dev, .size=total, .page_size=page,
                               .buffer_type=BufferType::DRAM};
    return CreateBuffer(c);
}

// ---------------------------------------------------------------------------
// Single-stage program builder
// Runs reader → [compute kernels up to `last_stage`] → stage_dump_writer
// and dumps the output of `dump_cb_id` to a GDDR6 buffer.
//
// last_stage:  0=rotation, 1=+polarquant, 2=+qjl
// dump_cb_id:  which CB to capture (must be the output of the last kernel)
// Returns: raw bytes from the dump buffer, tile-granular (not vector-granular)
// ---------------------------------------------------------------------------

static std::vector<uint8_t> run_isolated_stage(
    Device* device,
    CommandQueue& cq,
    CoreCoord core,
    uint32_t src_dram_addr,     // already-uploaded input
    uint32_t num_vectors,
    uint32_t d,
    int last_stage,             // 0, 1, or 2
    uint32_t dump_cb_id,
    uint32_t dump_page_bytes,
    const std::map<std::string, std::string>& defines)
{
    const uint32_t kN = turboquant::kTileVectors;
    const uint32_t num_tiles = (num_vectors + kN - 1) / kN;
    const uint32_t bytes_fp16 = kN * d * 2;
    const uint32_t bytes_pq   = kN * turboquant::kPolarQuantBytes;
    const uint32_t bytes_qjl  = kN * (turboquant::kQjlBytes + turboquant::kRNormBytes);

    auto dump_buf = dram_buf(device, num_tiles * dump_page_bytes, dump_page_bytes);

    Program prog = CreateProgram();
    auto make_cb = [&](uint32_t id, uint32_t page, DataFormat fmt) {
        CircularBufferConfig cfg(2 * page, {{id, fmt}}).set_page_size(id, page);
        CreateCircularBuffer(prog, core, cfg);
    };
    // Always create all CBs so the compute kernels can write/read them freely,
    // even if a downstream CB is never consumed in this stage run.
    make_cb(turboquant::kCbInput,    bytes_fp16, DataFormat::Float16_b);
    make_cb(turboquant::kCbRotated,  bytes_fp16, DataFormat::Float16_b);
    make_cb(turboquant::kCbQuant,    bytes_pq,   DataFormat::RawUInt8);
    make_cb(turboquant::kCbResidual, bytes_fp16, DataFormat::Float16_b);
    make_cb(turboquant::kCbQjl,      bytes_qjl,  DataFormat::RawUInt8);

    std::vector<uint32_t> cargs = {num_tiles};

    // Reader (RISCV_0)
    auto rk = CreateKernel(prog, "kernels/dataflow/reader_kernel.cpp", core,
        DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,
                           .noc=NOC::RISCV_0_default});
    SetRuntimeArgs(prog, rk, core, {src_dram_addr, num_vectors, d, kN});

    // Compute kernels up to last_stage
    CreateKernel(prog, "kernels/rotation_kernel.cpp", core,
        ComputeConfig{.math_fidelity=MathFidelity::HiFi4,
                      .compile_args=cargs, .defines=defines});
    if (last_stage >= 1) {
        CreateKernel(prog, "kernels/polarquant_kernel.cpp", core,
            ComputeConfig{.math_fidelity=MathFidelity::HiFi4,
                          .compile_args=cargs, .defines=defines});
    }
    if (last_stage >= 2) {
        CreateKernel(prog, "kernels/qjl_kernel.cpp", core,
            ComputeConfig{.math_fidelity=MathFidelity::HiFi4,
                          .compile_args=cargs, .defines=defines});
    }

    // Stage dump writer (RISCV_1) — reads dump_cb_id, writes to GDDR6
    std::map<std::string, std::string> dump_defines = {
        {"TQ_DUMP_CB_ID",      std::to_string(dump_cb_id)},
        {"TQ_DUMP_PAGE_BYTES", std::to_string(dump_page_bytes)},
    };
    auto wk = CreateKernel(prog, "kernels/dataflow/stage_dump_writer.cpp", core,
        DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,
                           .noc=NOC::RISCV_1_default,
                           .defines=dump_defines});
    SetRuntimeArgs(prog, wk, core, {dump_buf->address(), num_tiles, dump_page_bytes});

    EnqueueProgram(cq, prog, false);
    Finish(cq);

    std::vector<uint8_t> out(num_tiles * dump_page_bytes);
    EnqueueReadBuffer(cq, dump_buf, out.data(), true);
    return out;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    uint32_t num_vectors = 64u;
    uint32_t b = turboquant::kBits;
    uint32_t d = turboquant::kDim;
    uint32_t k = turboquant::kQjlDim;
    std::string mode = "full";

    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if      (a=="--num-vectors" && i+1<argc) num_vectors = std::atoi(argv[++i]);
        else if (a=="--bits"        && i+1<argc) b           = std::atoi(argv[++i]);
        else if (a=="--mode"        && i+1<argc) mode        = argv[++i];
    }

    std::cout << "[TurboQuant Phase 1] d=" << d << " b=" << b
              << " k=" << k << " N=" << num_vectors << "\n\n";

    // Generate & write input vectors
    auto test_vecs = gen_test_vectors(num_vectors, d);
    std::vector<uint16_t> input_fp16(num_vectors * d);
    for (uint32_t i = 0; i < num_vectors; ++i)
        for (uint32_t j = 0; j < d; ++j)
            input_fp16[i * d + j] = float_to_fp16_host(test_vecs[i][j]);
    write_bin("dump_input.bin", input_fp16.data(), input_fp16.size() * 2);

    std::map<std::string, std::string> defines = {
        {"TQ_DIM",           std::to_string(d)},
        {"TQ_BITS",          std::to_string(b)},
        {"TQ_QJL_DIM",       std::to_string(k)},
        {"TQ_ROTATION_SEED", std::to_string(turboquant::kRotationSeed)},
        {"TQ_QJL_SEED",      std::to_string(turboquant::kQjlSeed)},
    };

    Device* device = CreateDevice(0);
    CommandQueue& cq = device->command_queue();
    CoreCoord core{0, 0};

    auto t0 = std::chrono::high_resolution_clock::now();

    // Upload input once; reuse DRAM address across all isolated stage programs.
    auto input_buf = dram_buf(device, num_vectors * d * 2, d * 2);
    EnqueueWriteBuffer(cq, input_buf, input_fp16.data(), true);
    const uint32_t src_addr = input_buf->address();

    const uint32_t kN = turboquant::kTileVectors;
    const uint32_t num_tiles = (num_vectors + kN - 1) / kN;

    // -------------------------------------------------------------------------
    // Stage 0: rotation_kernel → dump kCbRotated (FP16 rotated vectors)
    // -------------------------------------------------------------------------
    std::cout << "[Stage 0] rotation_kernel → dump_rotated.bin\n";
    {
        uint32_t page = kN * d * 2;
        auto raw = run_isolated_stage(device, cq, core, src_addr,
                                      num_vectors, d, /*last_stage=*/0,
                                      turboquant::kCbRotated, page, defines);
        write_bin("dump_rotated.bin", raw.data(), raw.size());
    }

    // -------------------------------------------------------------------------
    // Stage 1a: rotation + polarquant → dump kCbQuant (packed b-bit indices)
    // -------------------------------------------------------------------------
    std::cout << "\n[Stage 1a] polarquant_kernel → dump_quant_indices.bin\n";
    {
        uint32_t page = kN * turboquant::kPolarQuantBytes;
        auto raw = run_isolated_stage(device, cq, core, src_addr,
                                      num_vectors, d, /*last_stage=*/1,
                                      turboquant::kCbQuant, page, defines);
        write_bin("dump_quant_indices.bin", raw.data(), raw.size());
    }

    // -------------------------------------------------------------------------
    // Stage 1b: rotation + polarquant → dump kCbResidual (FP16 centroids)
    //   kCbResidual is the dequantized centroid pass-through from polarquant.
    // -------------------------------------------------------------------------
    std::cout << "\n[Stage 1b] polarquant_kernel → dump_quant_centroids.bin\n";
    {
        uint32_t page = kN * d * 2;
        auto raw = run_isolated_stage(device, cq, core, src_addr,
                                      num_vectors, d, /*last_stage=*/1,
                                      turboquant::kCbResidual, page, defines);
        write_bin("dump_quant_centroids.bin", raw.data(), raw.size());
    }

    // -------------------------------------------------------------------------
    // Stage 2: rotation + polarquant + qjl → dump kCbQjl (bits + r_norm)
    // -------------------------------------------------------------------------
    std::cout << "\n[Stage 2] qjl_kernel → dump_qjl.bin\n";
    {
        uint32_t page = kN * (turboquant::kQjlBytes + turboquant::kRNormBytes);
        auto raw = run_isolated_stage(device, cq, core, src_addr,
                                      num_vectors, d, /*last_stage=*/2,
                                      turboquant::kCbQjl, page, defines);
        write_bin("dump_qjl.bin", raw.data(), raw.size());
    }

    // -------------------------------------------------------------------------
    // Full pipeline: all kernels → dump kCbOutput (packed final records)
    // The dequant_attn kernel is not included in the write path here —
    // test_roundtrip.py validates end-to-end by re-running dequant in Python.
    // -------------------------------------------------------------------------
    std::cout << "\n[Full pipeline] all kernels → dump_output.bin\n";
    {
        // The full output is produced by assembling the packed record from the
        // per-stage dumps (done in Python), OR by running a final combined
        // program that produces kCbOutput directly.
        //
        // Here we run stage 2 again but instruct the Python side to assemble
        // the final record from dump_quant_indices.bin + dump_qjl.bin, which
        // is exactly what test_roundtrip.py --device does.
        //
        // We still write dump_output.bin as a combined record for convenience.
        // It is assembled on the host from the already-read stage buffers:
        // [PolarQuant bytes | QJL bytes | r_norm bytes | padding]

        uint32_t pq_bytes   = turboquant::kPolarQuantBytes;
        uint32_t qjl_bytes  = turboquant::kQjlBytes + turboquant::kRNormBytes;
        uint32_t rec_bytes  = turboquant::kRecordBytes;

        // Re-read the stage dumps we just wrote
        auto read_bin = [](const std::string& path) -> std::vector<uint8_t> {
            std::ifstream f(path, std::ios::binary | std::ios::ate);
            if (!f) throw std::runtime_error("Cannot open " + path);
            size_t sz = f.tellg(); f.seekg(0);
            std::vector<uint8_t> buf(sz);
            f.read(reinterpret_cast<char*>(buf.data()), sz);
            return buf;
        };

        auto pq_raw  = read_bin("dump_quant_indices.bin");
        auto qjl_raw = read_bin("dump_qjl.bin");

        // The raw dumps are tile-granular (num_tiles × page_bytes).
        // We need to extract exactly num_vectors records from them.
        const uint32_t pq_page   = kN * pq_bytes;
        const uint32_t qjl_page  = kN * qjl_bytes;

        std::vector<uint8_t> output(num_vectors * rec_bytes, 0u);
        for (uint32_t v = 0; v < num_vectors; ++v) {
            uint32_t tile_idx  = v / kN;
            uint32_t vec_in_tile = v % kN;

            const uint8_t* pq_src  = pq_raw.data()  + tile_idx * pq_page  + vec_in_tile * pq_bytes;
            const uint8_t* qjl_src = qjl_raw.data() + tile_idx * qjl_page + vec_in_tile * qjl_bytes;
            uint8_t* dst = output.data() + v * rec_bytes;

            memcpy(dst,                                        pq_src,  pq_bytes);
            memcpy(dst + turboquant::kOffsetQjlBits,          qjl_src, turboquant::kQjlBytes);
            memcpy(dst + turboquant::kOffsetRNorm,             qjl_src + turboquant::kQjlBytes, 2);
        }
        write_bin("dump_output.bin", output.data(), output.size());
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    CloseDevice(device);

    std::cout << "\nTotal wall time: " << ms << " ms\n\n";
    std::cout << "Validate each stage against the NumPy reference:\n"
              << "  python tests/test_rotation.py   --device\n"
              << "  python tests/test_polarquant.py --device\n"
              << "  python tests/test_qjl.py        --device\n"
              << "  python tests/test_roundtrip.py  --device\n";
    return 0;
}

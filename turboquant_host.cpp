// host/turboquant_host.cpp
//
// Host-side TurboQuant driver for Tenstorrent Wormhole.
// Uses the TT-Metalium Mesh API (MeshDevice / MeshBuffer / MeshWorkload),
// matching the style of rref_tenstorrent/rref_host.cpp.
//
// Runs all four compute kernels in isolated single-stage programs, one per
// pipeline stage, and dumps each stage's CB output to GDDR6 for Python-side
// validation.  Dump files produced (in --mode full):
//
//   dump_input.bin            FP16 input vectors         (N × d × 2 bytes)
//   dump_rotated.bin          BF16 rotated vectors       (tile-granular)
//   dump_quant_indices.bin    packed b-bit indices       (tile-granular)
//   dump_quant_centroids.bin  BF16 dequantized centroids (tile-granular)
//   dump_qjl.bin              QJL bits + r_norm          (tile-granular)
//   dump_output.bin           final assembled records    (N × kRecordBytes)
//
// Build via CMake (see CMakeLists.txt):
//   cmake -B build -DCMAKE_BUILD_TYPE=Release
//   cmake --build build
//
// Run:
//   ./build/turboquant_host [--num-vectors 64] [--bits 4] [--mode full]
//
// Validate each stage:
//   python tests/test_rotation.py   --device
//   python tests/test_polarquant.py --device
//   python tests/test_qjl.py        --device
//   python tests/test_roundtrip.py  --device

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdexcept>
#include <algorithm>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>

#include "turboquant_layout.h"

using namespace tt;
using namespace tt::tt_metal;

// ---------------------------------------------------------------------------
// Constants — must match the compute kernels
// ---------------------------------------------------------------------------

static constexpr uint32_t kD  = turboquant::kDim;
static constexpr uint32_t kB  = turboquant::kBits;
static constexpr uint32_t kK  = turboquant::kQjlDim;
static constexpr uint32_t kN  = turboquant::kTileVectors;  // vectors per tile

// ---------------------------------------------------------------------------
// BF16 conversion (host-side, mirrors bfloat16.hpp usage in rref_host.cpp)
// ---------------------------------------------------------------------------

static uint16_t float_to_bf16(float v) {
    return tt::bfloat16(v).to_uint16();
}

static float bf16_to_float(uint16_t u) {
    return tt::bfloat16(u).to_float();
}

// ---------------------------------------------------------------------------
// Test vector generation — unit-norm Gaussian, identical to Python reference
// ---------------------------------------------------------------------------

static std::vector<std::vector<float>> gen_test_vectors(
    uint32_t n, uint32_t d, uint32_t seed = 42)
{
    uint64_t state = seed;
    auto rnd = [&]() -> float {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        uint32_t bits = static_cast<uint32_t>(state >> 33);
        bits = (bits & 0x7FFFFFu) | 0x3F800000u;
        float f; memcpy(&f, &bits, 4); return f - 1.5f;
    };
    std::vector<std::vector<float>> vs(n, std::vector<float>(d));
    for (uint32_t i = 0; i < n; ++i) {
        float nm = 0.0f;
        for (uint32_t j = 0; j < d; ++j) { vs[i][j] = rnd(); nm += vs[i][j] * vs[i][j]; }
        nm = std::sqrt(nm);
        for (uint32_t j = 0; j < d; ++j) vs[i][j] /= nm;
    }
    return vs;
}

// ---------------------------------------------------------------------------
// Binary file I/O
// ---------------------------------------------------------------------------

static void write_bin(const std::string& path, const void* data, size_t bytes) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open for writing: " + path);
    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    std::cout << "  wrote " << bytes << " bytes → " << path << "\n";
}

static std::vector<uint8_t> read_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    size_t sz = static_cast<size_t>(f.tellg()); f.seekg(0);
    std::vector<uint8_t> buf(sz);
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(sz));
    return buf;
}

// ---------------------------------------------------------------------------
// MeshBuffer helpers — mirrors rref_host.cpp's buffer allocation pattern
// ---------------------------------------------------------------------------

static std::shared_ptr<distributed::MeshBuffer> make_mesh_buf(
    distributed::MeshDevice* dev,
    uint32_t total_bytes,
    uint32_t page_bytes)
{
    distributed::ReplicatedBufferConfig rep_cfg{ .size = total_bytes };
    distributed::DeviceLocalBufferConfig local_cfg{
        .page_size   = page_bytes,
        .buffer_type = BufferType::DRAM,
    };
    return distributed::MeshBuffer::create(rep_cfg, local_cfg, dev);
}

// ---------------------------------------------------------------------------
// Isolated stage runner
//
// Runs reader + [compute kernels up to last_stage] + stage_dump_writer on
// core {0,0} and returns the raw bytes dumped from dump_cb.
//
// last_stage:  0 = rotation only
//              1 = rotation + polarquant
//              2 = rotation + polarquant + qjl
// dump_cb:     CB enum value to capture (CB::c_in1, CB::c_in2, etc.)
// dump_page:   page size of that CB in bytes
// ---------------------------------------------------------------------------

static std::vector<uint8_t> run_isolated_stage(
    distributed::MeshDevice*   mesh_dev,
    distributed::MeshCommandQueue& cq,
    const distributed::MeshCoordinateRange& dev_range,
    uint32_t               src_addr,
    uint32_t               num_vectors,
    int                    last_stage,
    CB                     dump_cb,
    uint32_t               dump_page_bytes)
{
    const uint32_t num_tiles = (num_vectors + kN - 1) / kN;
    const uint32_t bytes_fp16 = kN * kD * 2;
    const uint32_t bytes_pq   = kN * turboquant::kPolarQuantBytes;
    const uint32_t bytes_qjl  = kN * (turboquant::kQjlBytes + turboquant::kRNormBytes);

    auto dump_buf = make_mesh_buf(mesh_dev,
                                  num_tiles * dump_page_bytes, dump_page_bytes);

    Program prog = CreateProgram();
    CoreCoord core{0, 0};
    CoreRange core_range(core, core);

    // CircularBuffers — create all so compute kernels can freely push/pop
    auto make_cb = [&](CB id, uint32_t page, tt::DataFormat fmt) {
        CircularBufferConfig cfg(2 * page, {{id, fmt}}).set_page_size(id, page);
        CreateCircularBuffer(prog, core_range, cfg);
    };
    make_cb(CB::c_in0,  bytes_fp16, tt::DataFormat::Float16_b); // input
    make_cb(CB::c_in1,  bytes_fp16, tt::DataFormat::Float16_b); // rotated
    make_cb(CB::c_in2,  bytes_pq,   tt::DataFormat::RawUInt8);  // PQ indices
    make_cb(CB::c_in3,  bytes_fp16, tt::DataFormat::Float16_b); // centroids
    make_cb(CB::c_out0, bytes_qjl,  tt::DataFormat::RawUInt8);  // QJL output

    std::vector<uint32_t> cargs = {num_tiles};

    // Data movement — reader (RISCV_0)
    auto rk = CreateKernel(prog,
        "kernels/dataflow/reader_kernel.cpp", core_range,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc       = NOC::RISCV_0_default,
        });
    SetRuntimeArgs(prog, rk, core, {src_addr, num_vectors, kD, kN});

    // Compute kernels up to last_stage
    CreateKernel(prog, "kernels/rotation_kernel.cpp", core_range,
        ComputeConfig{
            .math_fidelity    = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args     = cargs,
        });

    if (last_stage >= 1) {
        CreateKernel(prog, "kernels/polarquant_kernel.cpp", core_range,
            ComputeConfig{
                .math_fidelity    = MathFidelity::HiFi4,
                .fp32_dest_acc_en = false,
                .math_approx_mode = false,
                .compile_args     = cargs,
            });
    }
    if (last_stage >= 2) {
        CreateKernel(prog, "kernels/qjl_kernel.cpp", core_range,
            ComputeConfig{
                .math_fidelity    = MathFidelity::HiFi4,
                .fp32_dest_acc_en = false,
                .math_approx_mode = false,
                .compile_args     = cargs,
            });
    }

    // Stage dump writer (RISCV_1) — captures dump_cb to GDDR6
    std::map<std::string, std::string> dump_defines = {
        {"TQ_DUMP_CB_ID",      std::to_string(static_cast<uint32_t>(dump_cb))},
        {"TQ_DUMP_PAGE_BYTES", std::to_string(dump_page_bytes)},
    };
    auto wk = CreateKernel(prog,
        "kernels/dataflow/stage_dump_writer.cpp", core_range,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc       = NOC::RISCV_1_default,
            .defines   = dump_defines,
        });
    SetRuntimeArgs(prog, wk, core,
                   {dump_buf->address(), num_tiles, dump_page_bytes});

    // Wrap in MeshWorkload and enqueue (non-blocking for throughput)
    distributed::MeshWorkload workload;
    workload.add_program(dev_range, std::move(prog));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);

    // Blocking read to flush queue and retrieve dump
    std::vector<uint8_t> out;
    distributed::EnqueueReadMeshBuffer(cq, out, dump_buf, /*blocking=*/true);
    return out;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    uint32_t num_vectors = 64u;
    std::string mode = "full";

    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if      (a == "--num-vectors" && i + 1 < argc) num_vectors = std::atoi(argv[++i]);
        else if (a == "--mode"        && i + 1 < argc) mode        = argv[++i];
    }

    std::cout << "[TurboQuant] d=" << kD << " b=" << kB
              << " k=" << kK << " N=" << num_vectors << "\n\n";

    // Generate input and write dump_input.bin immediately
    auto test_vecs = gen_test_vectors(num_vectors, kD);
    std::vector<uint16_t> input_bf16(num_vectors * kD);
    for (uint32_t i = 0; i < num_vectors; ++i)
        for (uint32_t j = 0; j < kD; ++j)
            input_bf16[i * kD + j] = float_to_bf16(test_vecs[i][j]);
    write_bin("dump_input.bin", input_bf16.data(), input_bf16.size() * 2);

    // Open device — unit mesh wrapping device 0, matching rref_host.cpp style
    auto mesh_dev = distributed::MeshDevice::create_unit_mesh(/*device_id=*/0);
    distributed::MeshCommandQueue& cq = mesh_dev->mesh_command_queue();
    distributed::MeshCoordinateRange dev_range(mesh_dev->shape());

    auto t0 = std::chrono::high_resolution_clock::now();

    // Upload input once; reuse address across all stage programs
    auto input_buf = make_mesh_buf(mesh_dev.get(), num_vectors * kD * 2, kD * 2);
    distributed::EnqueueWriteMeshBuffer(cq, input_buf, input_bf16, /*blocking=*/true);
    const uint32_t src_addr = input_buf->address();

    const uint32_t num_tiles   = (num_vectors + kN - 1) / kN;
    const uint32_t bytes_fp16  = kN * kD * 2;
    const uint32_t bytes_pq    = kN * turboquant::kPolarQuantBytes;
    const uint32_t bytes_qjl   = kN * (turboquant::kQjlBytes + turboquant::kRNormBytes);

    // -------------------------------------------------------------------------
    // Stage 0: rotation_kernel → dump CB::c_in1 (rotated BF16 vectors)
    // -------------------------------------------------------------------------
    std::cout << "[Stage 0] rotation_kernel → dump_rotated.bin\n";
    {
        auto raw = run_isolated_stage(mesh_dev.get(), cq, dev_range,
                                      src_addr, num_vectors, /*last_stage=*/0,
                                      CB::c_in1, bytes_fp16);
        write_bin("dump_rotated.bin", raw.data(), raw.size());
    }

    // -------------------------------------------------------------------------
    // Stage 1a: + polarquant → dump CB::c_in2 (packed b-bit indices)
    // -------------------------------------------------------------------------
    std::cout << "\n[Stage 1a] polarquant_kernel → dump_quant_indices.bin\n";
    {
        auto raw = run_isolated_stage(mesh_dev.get(), cq, dev_range,
                                      src_addr, num_vectors, /*last_stage=*/1,
                                      CB::c_in2, bytes_pq);
        write_bin("dump_quant_indices.bin", raw.data(), raw.size());
    }

    // -------------------------------------------------------------------------
    // Stage 1b: + polarquant → dump CB::c_in3 (BF16 dequantized centroids)
    // -------------------------------------------------------------------------
    std::cout << "\n[Stage 1b] polarquant_kernel → dump_quant_centroids.bin\n";
    {
        auto raw = run_isolated_stage(mesh_dev.get(), cq, dev_range,
                                      src_addr, num_vectors, /*last_stage=*/1,
                                      CB::c_in3, bytes_fp16);
        write_bin("dump_quant_centroids.bin", raw.data(), raw.size());
    }

    // -------------------------------------------------------------------------
    // Stage 2: + qjl → dump CB::c_out0 (QJL bits + BF16 r_norm)
    // -------------------------------------------------------------------------
    std::cout << "\n[Stage 2] qjl_kernel → dump_qjl.bin\n";
    {
        auto raw = run_isolated_stage(mesh_dev.get(), cq, dev_range,
                                      src_addr, num_vectors, /*last_stage=*/2,
                                      CB::c_out0, bytes_qjl);
        write_bin("dump_qjl.bin", raw.data(), raw.size());
    }

    // -------------------------------------------------------------------------
    // Assemble dump_output.bin on the host from the per-stage dumps.
    // Layout: [PQ bytes | QJL bytes | r_norm bytes | padding] × N vectors.
    // -------------------------------------------------------------------------
    std::cout << "\n[Assemble] building dump_output.bin from stage dumps\n";
    {
        auto pq_raw  = read_bin("dump_quant_indices.bin");
        auto qjl_raw = read_bin("dump_qjl.bin");

        const uint32_t pq_page_bytes  = bytes_pq;
        const uint32_t qjl_page_bytes = bytes_qjl;
        const uint32_t pq_per_vec     = turboquant::kPolarQuantBytes;
        const uint32_t qjl_per_vec    = turboquant::kQjlBytes + turboquant::kRNormBytes;
        const uint32_t rec_bytes      = turboquant::kRecordBytes;

        std::vector<uint8_t> output(num_vectors * rec_bytes, 0u);
        for (uint32_t v = 0; v < num_vectors; ++v) {
            uint32_t tile_idx    = v / kN;
            uint32_t vec_in_tile = v % kN;

            const uint8_t* pq_src  = pq_raw.data()
                                   + tile_idx * pq_page_bytes
                                   + vec_in_tile * pq_per_vec;
            const uint8_t* qjl_src = qjl_raw.data()
                                   + tile_idx * qjl_page_bytes
                                   + vec_in_tile * qjl_per_vec;
            uint8_t* dst = output.data() + v * rec_bytes;

            memcpy(dst + turboquant::kOffsetPolarQuant, pq_src,  pq_per_vec);
            memcpy(dst + turboquant::kOffsetQjlBits,   qjl_src, turboquant::kQjlBytes);
            memcpy(dst + turboquant::kOffsetRNorm,
                   qjl_src + turboquant::kQjlBytes, turboquant::kRNormBytes);
        }
        write_bin("dump_output.bin", output.data(), output.size());
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    mesh_dev->close();

    std::cout << "\nTotal time: " << std::fixed << std::setprecision(1)
              << ms << " ms\n\n"
              << "Validate with:\n"
              << "  python tests/test_rotation.py   --device\n"
              << "  python tests/test_polarquant.py --device\n"
              << "  python tests/test_qjl.py        --device\n"
              << "  python tests/test_roundtrip.py  --device\n";
    return 0;
}

// host/turboquant_host.cpp
//
// Host-side TurboQuant driver for Tenstorrent Wormhole.
// Uses the TT-Metalium Mesh API (MeshDevice / MeshBuffer / MeshWorkload).

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

#include "turboquant_layout.h"

#include <tt-metalium/hal.hpp>
using namespace tt::tt_metal;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr uint32_t kD = turboquant::kDim;
static constexpr uint32_t kB = turboquant::kBits;
static constexpr uint32_t kK = turboquant::kQjlDim;
static constexpr uint32_t kN = turboquant::kTileVectors;

// ---------------------------------------------------------------------------
// BF16 conversion
// bfloat16 is in the global namespace (not tt:: or tt::tt_metal::)

// ---------------------------------------------------------------------------

static uint16_t float_to_bf16(float v) {
    // BF16 = upper 16 bits of float32
    uint32_t bits; memcpy(&bits, &v, 4); return static_cast<uint16_t>(bits >> 16);
}

static float bf16_to_float(uint16_t u) {
    // Expand BF16 to float32 by zero-extending lower 16 bits
    uint32_t bits = static_cast<uint32_t>(u) << 16; float f; memcpy(&f, &bits, 4); return f;
}

// ---------------------------------------------------------------------------
// Test vector generation
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
    std::cout << "  wrote " << bytes << " bytes -> " << path << "\n";
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
// MeshBuffer helper
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
// ---------------------------------------------------------------------------

static std::vector<uint8_t> run_isolated_stage(
    distributed::MeshDevice*                 mesh_dev,
    distributed::MeshCommandQueue&           cq,
    const distributed::MeshCoordinateRange&  dev_range,
    uint64_t                                 src_addr,   // DeviceAddr is uint64
    uint32_t                                 num_vectors,
    int                                      last_stage,
    tt::CB                                   dump_cb,
    uint32_t                                 dump_page_bytes,
    uint64_t                                 cent_addr = 0)
{
    const uint32_t num_tiles  = (num_vectors + kN - 1) / kN;
    const uint32_t bytes_fp16 = kN * kD * 2;
    const uint32_t bytes_pq   = kN * turboquant::kPolarQuantBytes;
    const uint32_t bytes_qjl  = kN * (turboquant::kQjlBytes + turboquant::kRNormBytes);

    auto dump_buf = make_mesh_buf(mesh_dev,
                                  num_tiles * dump_page_bytes, dump_page_bytes);

    Program prog = CreateProgram();
    tt::tt_metal::CoreCoord core{0, 0};
    tt::tt_metal::CoreRange core_range(core, core);

    // CircularBuffers — build config without method chaining (older API style)
    auto make_cb = [&](tt::CB id, uint32_t page, tt::DataFormat fmt) {
        std::map<uint8_t, tt::DataFormat> data_format_spec = {{static_cast<uint8_t>(id), fmt}};
        CircularBufferConfig cfg(2 * page, data_format_spec);
        cfg.set_page_size(static_cast<uint8_t>(id), page);
        CreateCircularBuffer(prog, core_range, cfg);
    };
    make_cb(tt::CB::c_in0,  bytes_fp16, tt::DataFormat::Float16_b);
    make_cb(tt::CB::c_in1,  bytes_fp16, tt::DataFormat::Float16_b);
    make_cb(tt::CB::c_in2,  bytes_pq,   tt::DataFormat::RawUInt8);
    make_cb(tt::CB::c_in3,  bytes_fp16, tt::DataFormat::Float16_b);
    make_cb(tt::CB::c_out0, bytes_qjl,  tt::DataFormat::RawUInt8);

    std::vector<uint32_t> cargs = {num_tiles};

    // Common defines for all compute kernels
    std::map<std::string, std::string> kdefines = {
        {"TQ_DIM",           std::to_string(kD)},
        {"TQ_ROTATION_SEED", std::to_string(turboquant::kRotationSeed)},
        {"TQ_BITS",          std::to_string(kB)},
        {"TQ_K_VECS",        std::to_string(kN)},
        {"TQ_QJL_DIM",       std::to_string(kK)},
        {"TQ_QJL_SEED",      std::to_string(turboquant::kQjlSeed)},
    };

    // Each stage runs exactly one kernel on RISCV_0
    if (last_stage == 0) {
        // Stage 0: rotation — reads DRAM directly, writes c_in1
        auto rk = CreateKernel(prog,
            "kernels/rotation_kernel.cpp", core_range,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc       = NOC::RISCV_0_default,
                .defines   = kdefines,
            });
        SetRuntimeArgs(prog, rk, core,
                       {static_cast<uint32_t>(src_addr), num_vectors, kD, kN});
    } else if (last_stage == 1) {
        // Stage 1: polarquant — reads c_in1 (written by stage 0 via dump reload)
        auto rk = CreateKernel(prog,
            "kernels/polarquant_kernel.cpp", core_range,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc       = NOC::RISCV_0_default,
                .defines   = kdefines,
            });
        SetRuntimeArgs(prog, rk, core,
                       {static_cast<uint32_t>(src_addr), num_vectors, kD, kN});
    } else if (last_stage == 2) {
        // Stage 2: qjl - one core per vector for parallelism
        const uint32_t cores_x = 8u, cores_y = 4u;
        tt::tt_metal::CoreRange multi_core_range(
            tt::tt_metal::CoreCoord{0, 0},
            tt::tt_metal::CoreCoord{cores_x - 1, cores_y - 1});
        auto rk = CreateKernel(prog,
            "kernels/qjl_kernel.cpp", multi_core_range,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc       = NOC::RISCV_0_default,
                .defines   = kdefines,
            });
        // Each core processes exactly 1 vector
        for (uint32_t cy = 0u; cy < cores_y; ++cy) {
            for (uint32_t cx = 0u; cx < cores_x; ++cx) {
                uint32_t vec_idx = cy * cores_x + cx;
                tt::tt_metal::CoreCoord c{cx, cy};
                SetRuntimeArgs(prog, rk, c,
                    {static_cast<uint32_t>(src_addr), 1u, kD, 1u,
                     static_cast<uint32_t>(cent_addr), vec_idx});
            }
        }
    }

    // Stage dump writer (RISCV_1)
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
                   {static_cast<uint32_t>(dump_buf->address()),
                    num_tiles, dump_page_bytes});

    distributed::MeshWorkload workload;
    workload.add_program(dev_range, std::move(prog));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);

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

    auto test_vecs = gen_test_vectors(num_vectors, kD);
    std::vector<uint16_t> input_bf16(num_vectors * kD);
    for (uint32_t i = 0; i < num_vectors; ++i)
        for (uint32_t j = 0; j < kD; ++j)
            input_bf16[i * kD + j] = float_to_bf16(test_vecs[i][j]);
    write_bin("dump_input.bin", input_bf16.data(), input_bf16.size() * 2);

    auto mesh_dev = distributed::MeshDevice::create_unit_mesh(6);
    distributed::MeshCommandQueue& cq = mesh_dev->mesh_command_queue();
    distributed::MeshCoordinateRange dev_range(mesh_dev->shape());

    auto t0 = std::chrono::high_resolution_clock::now();

    auto input_buf = make_mesh_buf(mesh_dev.get(), num_vectors * kD * 2, kD * 2);
    distributed::EnqueueWriteMeshBuffer(cq, input_buf, input_bf16, /*blocking=*/true);
    const uint64_t src_addr = input_buf->address();

    const uint32_t bytes_fp16 = kN * kD * 2;
    const uint32_t bytes_pq   = kN * turboquant::kPolarQuantBytes;
    const uint32_t bytes_qjl  = kN * (turboquant::kQjlBytes + turboquant::kRNormBytes);

    std::cout << "[Stage 0] rotation_kernel -> dump_rotated.bin\n";
    {
        auto raw = run_isolated_stage(mesh_dev.get(), cq, dev_range,
                                      src_addr, num_vectors, 0,
                                      tt::CB::c_in1, bytes_fp16);
        write_bin("dump_rotated.bin", raw.data(), raw.size());
    }

    // Upload rotated vectors back to DRAM for polarquant to read
    auto rot_raw = read_bin("dump_rotated.bin");
    auto rot_buf = make_mesh_buf(mesh_dev.get(), rot_raw.size(), kD * 2);
    distributed::EnqueueWriteMeshBuffer(cq, rot_buf,
        std::vector<uint16_t>(reinterpret_cast<uint16_t*>(rot_raw.data()),
                              reinterpret_cast<uint16_t*>(rot_raw.data() + rot_raw.size())),
        /*blocking=*/true);
    const uint64_t rot_addr = rot_buf->address();
    std::cout << "\n[Stage 1a] polarquant_kernel -> dump_quant_indices.bin\n";
    {
        auto raw = run_isolated_stage(mesh_dev.get(), cq, dev_range,
                                      rot_addr, num_vectors, 1,
                                      tt::CB::c_in2, bytes_pq);
        write_bin("dump_quant_indices.bin", raw.data(), raw.size());
    }

    std::cout << "\n[Stage 1b] polarquant_kernel -> dump_quant_centroids.bin\n";
    {
        auto raw = run_isolated_stage(mesh_dev.get(), cq, dev_range,
                                      rot_addr, num_vectors, 1,
                                      tt::CB::c_in3, bytes_fp16);
        write_bin("dump_quant_centroids.bin", raw.data(), raw.size());
    }

    std::cout << "\n[Stage 2] qjl_kernel -> dump_qjl.bin\n";
    {
        // Upload centroids back to DRAM for QJL to read
        auto cent_raw = read_bin("dump_quant_centroids.bin");
        auto cent_buf = make_mesh_buf(mesh_dev.get(), cent_raw.size(), kD * 2);
        distributed::EnqueueWriteMeshBuffer(cq, cent_buf,
            std::vector<uint16_t>(reinterpret_cast<uint16_t*>(cent_raw.data()),
                                  reinterpret_cast<uint16_t*>(cent_raw.data() + cent_raw.size())),
            /*blocking=*/true);
        const uint64_t cent_addr = cent_buf->address();

        // Allocate QJL output buffer: num_vectors * kRecBytes
        const uint32_t kRecBytes       = (kK + 7) / 8 + 2;

        const uint32_t kDramAlign     = tt::tt_metal::hal::get_dram_alignment();

        const uint32_t kRecBytesPadded = (kRecBytes + kDramAlign - 1) & ~(kDramAlign - 1);

        auto qjl_out_buf = make_mesh_buf(mesh_dev.get(),

                                         num_vectors * kRecBytesPadded,

                                         kRecBytesPadded);

        const uint64_t qjl_out_addr = qjl_out_buf->address();
        // Multi-core QJL: 32 cores, one per vector
        const uint32_t cores_x = 8u, cores_y = 4u;
        Program prog2 = CreateProgram();
        // Use device compute grid, skip row 0 col 0 (dispatch)
        auto grid = mesh_dev->compute_with_storage_grid_size();
        // Use rows 0..cores_y-1, cols 1..cores_x (skip col 0 dispatch)
        tt::tt_metal::CoreRange qjl_range(
            tt::tt_metal::CoreCoord{0, 2},
            tt::tt_metal::CoreCoord{7, 5});

        std::map<std::string, std::string> qjl_defines = {
            {"TQ_DIM",           std::to_string(kD)},
            {"TQ_ROTATION_SEED", std::to_string(turboquant::kRotationSeed)},
            {"TQ_BITS",          std::to_string(kB)},
            {"TQ_K_VECS",        std::to_string(kN)},
            {"TQ_QJL_DIM",       std::to_string(kK)},
            {"TQ_QJL_SEED",      std::to_string(turboquant::kQjlSeed)},
        };

        // CB for each core: c_0=input, c_3=centroid, c_16=output scratch
        auto make_cb2 = [&](tt::CB id, uint32_t page, tt::DataFormat fmt) {
            std::map<uint8_t, tt::DataFormat> spec = {{static_cast<uint8_t>(id), fmt}};
            CircularBufferConfig cfg(2 * page, spec);
            cfg.set_page_size(static_cast<uint8_t>(id), page);
            CreateCircularBuffer(prog2, qjl_range, cfg);
        };
        make_cb2(tt::CB::c_in0,  kD * 2,    tt::DataFormat::Float16_b);
        make_cb2(tt::CB::c_in2,  kD * 2,    tt::DataFormat::Float16_b);
        make_cb2(tt::CB::c_in3,  kD * 2,    tt::DataFormat::Float16_b);
        make_cb2(tt::CB::c_out0, kRecBytes,  tt::DataFormat::RawUInt8);

        auto qjl_k = CreateKernel(prog2, "kernels/qjl_kernel.cpp", qjl_range,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc       = NOC::RISCV_0_default,
                .defines   = qjl_defines,
            });

        for (uint32_t cy = 0u; cy < cores_y; ++cy) {
            for (uint32_t cx = 0u; cx < cores_x; ++cx) {
                uint32_t vi = cy * cores_x + cx;
                tt::tt_metal::CoreCoord c{cx, cy + 2};
                SetRuntimeArgs(prog2, qjl_k, c,
                    {static_cast<uint32_t>(src_addr), 1u, kD, 1u,
                     static_cast<uint32_t>(cent_addr), vi,
                     static_cast<uint32_t>(qjl_out_addr),
                     kRecBytesPadded});
            }
        }

        distributed::MeshWorkload wl2;
        // Only run on local chip (coord {0,0}), not remote chip 16
        distributed::MeshCoordinateRange local_range(
            distributed::MeshCoordinate{0, 0},
            distributed::MeshCoordinate{0, 0});
        wl2.add_program(local_range, std::move(prog2));
        distributed::EnqueueMeshWorkload(cq, wl2, /*blocking=*/true);

        // Read QJL output (deinterleave padding)


        std::vector<uint8_t> raw_padded;


        distributed::EnqueueReadMeshBuffer(cq, raw_padded, qjl_out_buf, /*blocking=*/true);


        std::vector<uint8_t> raw(num_vectors * kRecBytes);


        for (uint32_t i = 0; i < num_vectors; ++i) {


            std::memcpy(raw.data()      + i * kRecBytes,


                        raw_padded.data() + i * kRecBytesPadded,


                        kRecBytes);


        }


        write_bin("dump_qjl.bin", raw.data(), raw.size());
    }
    {
        auto pq_raw  = read_bin("dump_quant_indices.bin");
        auto qjl_raw = read_bin("dump_qjl.bin");

        const uint32_t pq_page  = bytes_pq;
        const uint32_t qjl_page = bytes_qjl;
        const uint32_t pq_vec   = turboquant::kPolarQuantBytes;
        const uint32_t qjl_vec  = turboquant::kQjlBytes + turboquant::kRNormBytes;
        const uint32_t rec      = turboquant::kRecordBytes;

        std::vector<uint8_t> output(num_vectors * rec, 0u);
        for (uint32_t v = 0; v < num_vectors; ++v) {
            uint32_t ti = v / kN, vi = v % kN;
            const uint8_t* pq_src  = pq_raw.data()  + ti * pq_page  + vi * pq_vec;
            const uint8_t* qjl_src = qjl_raw.data() + ti * qjl_page + vi * qjl_vec;
            uint8_t* dst = output.data() + v * rec;
            memcpy(dst + turboquant::kOffsetPolarQuant, pq_src,  pq_vec);
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

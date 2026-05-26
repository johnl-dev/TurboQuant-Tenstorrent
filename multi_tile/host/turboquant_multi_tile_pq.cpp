// turboquant_multi_tile_pq.cpp
//
// Standalone Stage 1 (PolarQuant centroids) tile-form host.
//
// Reads dump_rotated_tile.bin (flat row-major bf16, output of
// turboquant_multi_tile Stage 0), tilizes it, runs the SFPU polarquant
// compute kernel, untilizes the output, writes dump_centroids_tile.bin.
//
// Compare against the baseline's dump_quant_centroids.bin (when available)
// via multi_tile/tests/test_polarquant_tile.py.

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>

#include "tile_layout.h"

namespace tq = turboquant::multi_tile;

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

// -----------------------------------------------------------------------------
// Load flat row-major bf16 file -> tilized bytes
// -----------------------------------------------------------------------------
static std::vector<uint16_t> load_flat_as_tiles(
    const std::string& path, uint32_t n_rows, uint32_t n_cols)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::abort(); }
    std::vector<uint16_t> raw(n_rows * n_cols);
    f.read(reinterpret_cast<char*>(raw.data()), n_rows * n_cols * 2);

    std::vector<bfloat16> as_bf(raw.size());
    static_assert(sizeof(bfloat16) == sizeof(uint16_t), "bf16 must be u16-sized");
    std::memcpy(as_bf.data(), raw.data(), raw.size() * sizeof(uint16_t));

    std::vector<bfloat16> tiled = tilize_nfaces(as_bf, n_rows, n_cols);
    std::vector<uint16_t> out(tiled.size());
    std::memcpy(out.data(), tiled.data(), tiled.size() * sizeof(uint16_t));
    return out;
}

// -----------------------------------------------------------------------------
// MeshBuffer helper
// -----------------------------------------------------------------------------
static std::shared_ptr<MeshBuffer> make_mesh_buf(
    MeshDevice* dev, uint32_t total_bytes, uint32_t page_bytes)
{
    ReplicatedBufferConfig rep_cfg{ .size = total_bytes };
    DeviceLocalBufferConfig local_cfg{
        .page_size   = page_bytes,
        .buffer_type = BufferType::DRAM,
    };
    return MeshBuffer::create(rep_cfg, local_cfg, dev);
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    uint32_t num_vectors = 32;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--num-vectors" && i + 1 < argc) {
            num_vectors = std::atoi(argv[++i]);
        }
    }
    constexpr uint32_t d = tq::D;
    if (num_vectors % tq::TILE_HEIGHT != 0) {
        std::fprintf(stderr, "num_vectors (%u) must be a multiple of 32\n", num_vectors);
        return 1;
    }
    const uint32_t Mt = num_vectors / tq::TILE_HEIGHT;
    const uint32_t Dt = d / tq::TILE_WIDTH;  // 4

    std::cout << "[multi_tile Stage 1] N=" << num_vectors
              << " d=" << d << " Mt=" << Mt << " Dt=" << Dt << "\n";

    // Load Stage 0 output as Stage 1 input
    auto x_tiles = load_flat_as_tiles("dump_rotated_tile.bin", num_vectors, d);
    std::cout << "  Loaded " << x_tiles.size() << " bf16 values from dump_rotated_tile.bin\n";

    auto mesh_dev = MeshDevice::create_unit_mesh(6);
    auto& cq = mesh_dev->mesh_command_queue();
    MeshCoordinateRange dev_range(MeshCoordinate{0, 0}, MeshCoordinate{0, 0});

    const uint32_t tile_bytes = tq::TILE_BYTES_BF16;
    auto x_buf    = make_mesh_buf(mesh_dev.get(), Mt * Dt * tile_bytes, tile_bytes);
    auto cent_buf = make_mesh_buf(mesh_dev.get(), Mt * Dt * tile_bytes, tile_bytes);

    EnqueueWriteMeshBuffer(cq, x_buf, x_tiles, /*blocking=*/true);

    Program prog = CreateProgram();
    tt::tt_metal::CoreCoord core{0, 0};
    tt::tt_metal::CoreRange core_range(core, core);

    auto make_cb = [&](tt::CB id, uint32_t n_pages) {
        std::map<uint8_t, tt::DataFormat> spec = {
            {static_cast<uint8_t>(id), tt::DataFormat::Float16_b}
        };
        CircularBufferConfig cfg(n_pages * tile_bytes, spec);
        cfg.set_page_size(static_cast<uint8_t>(id), tile_bytes);
        CreateCircularBuffer(prog, core_range, cfg);
    };
    make_cb(tt::CB::c_in0,  2);  // input
    make_cb(tt::CB::c_out0, 2);  // centroid output

    auto reader_id = CreateKernel(prog,
        "multi_tile/kernels/polarquant_reader.cpp", core_range,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc       = NOC::RISCV_0_default,
        });
    auto writer_id = CreateKernel(prog,
        "multi_tile/kernels/polarquant_writer.cpp", core_range,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc       = NOC::RISCV_1_default,
        });
    std::vector<uint32_t> compute_args = {Mt, Dt};
    CreateKernel(prog,
        "multi_tile/kernels/polarquant_compute.cpp", core_range,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .compile_args  = compute_args,
        });

    SetRuntimeArgs(prog, reader_id, core,
        {static_cast<uint32_t>(x_buf->address()), Mt, Dt});
    SetRuntimeArgs(prog, writer_id, core,
        {static_cast<uint32_t>(cent_buf->address()), Mt, Dt});

    MeshWorkload wl;
    wl.add_program(dev_range, std::move(prog));

    std::cout << "  Dispatching workload...\n";
    auto t0 = std::chrono::steady_clock::now();
    EnqueueMeshWorkload(cq, wl, /*blocking=*/true);
    auto t1 = std::chrono::steady_clock::now();
    double dispatch_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "  Dispatch+execute time: " << dispatch_ms << " ms\n";

    // Read output back, untilize, write flat bf16
    std::vector<uint16_t> cent_tiles(Mt * Dt * tq::TILE_ELEMENTS);
    EnqueueReadMeshBuffer(cq, cent_tiles, cent_buf, /*blocking=*/true);

    std::vector<bfloat16> cent_as_bf(cent_tiles.size());
    std::memcpy(cent_as_bf.data(), cent_tiles.data(),
                cent_tiles.size() * sizeof(uint16_t));
    std::vector<bfloat16> cent_flat = untilize_nfaces(cent_as_bf, num_vectors, d);
    std::vector<uint16_t> cent_out(cent_flat.size());
    std::memcpy(cent_out.data(), cent_flat.data(),
                cent_flat.size() * sizeof(uint16_t));

    std::ofstream out("dump_centroids_tile.bin", std::ios::binary);
    out.write(reinterpret_cast<const char*>(cent_out.data()), cent_out.size() * 2);
    std::cout << "  Wrote " << (cent_out.size() * 2)
              << " bytes -> dump_centroids_tile.bin (flat row-major)\n";

    std::cout << "[multi_tile Stage 1] done.\n";
    return 0;
}

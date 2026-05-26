// turboquant_multi_tile_rnorm.cpp
//
// Stage 2c (r_norm) host. Two passes:
//   2c1: r_sq = r * r  (eltwise mul)  -- writes to internal DRAM buf
//   2c2: r_norm = sqrt(r_sq @ ones)   -- writes dump_rnorm_tile.bin

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>

#include <cstdint>
#include <cstring>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>

#include "tile_layout.h"

namespace tq = turboquant::multi_tile;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

static std::vector<uint16_t> load_flat_as_tiles(
    const std::string& path, uint32_t n_rows, uint32_t n_cols)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::abort(); }
    std::vector<uint16_t> raw(n_rows * n_cols);
    f.read(reinterpret_cast<char*>(raw.data()), n_rows * n_cols * 2);
    std::vector<bfloat16> as_bf(raw.size());
    static_assert(sizeof(bfloat16) == sizeof(uint16_t));
    std::memcpy(as_bf.data(), raw.data(), raw.size() * 2);
    std::vector<bfloat16> tiled = tilize_nfaces(as_bf, n_rows, n_cols);
    std::vector<uint16_t> out(tiled.size());
    std::memcpy(out.data(), tiled.data(), tiled.size() * 2);
    return out;
}

static std::vector<uint16_t> tilize_bf16(
    const std::vector<float>& flat, uint32_t n_rows, uint32_t n_cols)
{
    std::vector<bfloat16> as_bf(flat.size());
    for (size_t i = 0; i < flat.size(); ++i) as_bf[i] = bfloat16(flat[i]);
    std::vector<bfloat16> tiled = tilize_nfaces(as_bf, n_rows, n_cols);
    std::vector<uint16_t> out(tiled.size());
    std::memcpy(out.data(), tiled.data(), tiled.size() * 2);
    return out;
}

static std::shared_ptr<MeshBuffer> make_mesh_buf(
    MeshDevice* dev, uint32_t total_bytes, uint32_t page_bytes)
{
    ReplicatedBufferConfig rep{ .size = total_bytes };
    DeviceLocalBufferConfig local{ .page_size = page_bytes, .buffer_type = BufferType::DRAM };
    return MeshBuffer::create(rep, local, dev);
}

int main(int argc, char** argv) {
    uint32_t num_vectors = 32;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--num-vectors" && i + 1 < argc)
            num_vectors = std::atoi(argv[++i]);
    }
    constexpr uint32_t d = tq::D;
    if (num_vectors % tq::TILE_HEIGHT != 0) {
        std::fprintf(stderr, "num_vectors must be multiple of 32\n"); return 1;
    }
    const uint32_t Mt = num_vectors / tq::TILE_HEIGHT;
    const uint32_t Dt = d / tq::TILE_WIDTH;

    std::cout << "[multi_tile Stage 2c] N=" << num_vectors
              << " Mt=" << Mt << " Dt=" << Dt << "\n";

    // Load r tiles from 2b1's output
    auto r_tiles = load_flat_as_tiles("dump_r_tile.bin", num_vectors, d);

    // Build ones column: Dt tiles, each is 32x32 with column 0 = 1.0 and rest = 0
    // Actually the matmul `r_sq @ ones` works fine if "ones" tile is all 1.0,
    // because matmul_tiles computes the full 32x32 outer product accumulation.
    // We only care about column 0 of the output, so it's fine if the matmul also
    // writes garbage into other columns. Use full all-1.0 tiles.
    std::vector<float> ones_flat(d * d, 1.0f);
    // Wait, the ones operand is shape (d, 1). For the matmul we tile it as a
    // (d, 32) layout — Kt tiles of size 32x32, where each tile has all entries = 1.0.
    // The output of (32, 32) @ (32, 32) where the second operand is all-1 is
    // each output element = sum of input row. Same value across the row, so
    // column 0 of output is what we want.
    auto ones_tiles = tilize_bf16(ones_flat, d, d);

    auto mesh_dev = MeshDevice::create_unit_mesh(6);
    auto& cq = mesh_dev->mesh_command_queue();
    MeshCoordinateRange dev_range(MeshCoordinate{0, 0}, MeshCoordinate{0, 0});

    const uint32_t tile_bytes = tq::TILE_BYTES_BF16;
    auto r_buf    = make_mesh_buf(mesh_dev.get(), Mt * Dt * tile_bytes, tile_bytes);
    auto rsq_buf  = make_mesh_buf(mesh_dev.get(), Mt * Dt * tile_bytes, tile_bytes);
    auto ones_buf = make_mesh_buf(mesh_dev.get(), Dt * Dt * tile_bytes, tile_bytes);
    auto norm_buf = make_mesh_buf(mesh_dev.get(), Mt      * tile_bytes, tile_bytes);

    EnqueueWriteMeshBuffer(cq, r_buf,    r_tiles,    true);
    EnqueueWriteMeshBuffer(cq, ones_buf, ones_tiles, true);

    // ------------ Pass 2c1: square ------------
    {
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
        make_cb(tt::CB::c_in0,  2);
        make_cb(tt::CB::c_out0, 2);

        auto reader_id = CreateKernel(prog,
            "multi_tile/kernels/rsquare_reader.cpp", core_range,
            DataMovementConfig{ .processor = DataMovementProcessor::RISCV_0,
                                .noc = NOC::RISCV_0_default });
        auto writer_id = CreateKernel(prog,
            "multi_tile/kernels/rsquare_writer.cpp", core_range,
            DataMovementConfig{ .processor = DataMovementProcessor::RISCV_1,
                                .noc = NOC::RISCV_1_default });
        std::vector<uint32_t> cargs = {Mt, Dt};
        CreateKernel(prog,
            "multi_tile/kernels/rsquare_compute.cpp", core_range,
            ComputeConfig{ .math_fidelity = MathFidelity::HiFi4, .compile_args = cargs });

        SetRuntimeArgs(prog, reader_id, core,
            {static_cast<uint32_t>(r_buf->address()), Mt, Dt});
        SetRuntimeArgs(prog, writer_id, core,
            {static_cast<uint32_t>(rsq_buf->address()), Mt, Dt});

        MeshWorkload wl;
        wl.add_program(dev_range, std::move(prog));
        std::cout << "  Dispatching 2c1 (square)...\n";
        auto t0 = std::chrono::steady_clock::now();
        EnqueueMeshWorkload(cq, wl, true);
        auto t1 = std::chrono::steady_clock::now();
        std::cout << "  2c1 time: "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count()
                  << " ms\n";
    }

    // ------------ Pass 2c2: matmul + sqrt ------------
    {
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
        make_cb(tt::CB::c_in0,  2);
        make_cb(tt::CB::c_in1,  2);
        make_cb(tt::CB::c_out0, 2);

        auto reader_id = CreateKernel(prog,
            "multi_tile/kernels/rnorm_matmul_reader.cpp", core_range,
            DataMovementConfig{ .processor = DataMovementProcessor::RISCV_0,
                                .noc = NOC::RISCV_0_default });
        auto writer_id = CreateKernel(prog,
            "multi_tile/kernels/rnorm_matmul_writer.cpp", core_range,
            DataMovementConfig{ .processor = DataMovementProcessor::RISCV_1,
                                .noc = NOC::RISCV_1_default });
        std::vector<uint32_t> cargs = {Mt, Dt};   // Kt == Dt
        CreateKernel(prog,
            "multi_tile/kernels/rnorm_matmul_compute.cpp", core_range,
            ComputeConfig{ .math_fidelity = MathFidelity::HiFi4, .compile_args = cargs });

        SetRuntimeArgs(prog, reader_id, core,
            {static_cast<uint32_t>(rsq_buf->address()),
             static_cast<uint32_t>(ones_buf->address()), Mt, Dt});
        SetRuntimeArgs(prog, writer_id, core,
            {static_cast<uint32_t>(norm_buf->address()), Mt});

        MeshWorkload wl;
        wl.add_program(dev_range, std::move(prog));
        std::cout << "  Dispatching 2c2 (matmul+sqrt)...\n";
        auto t0 = std::chrono::steady_clock::now();
        EnqueueMeshWorkload(cq, wl, true);
        auto t1 = std::chrono::steady_clock::now();
        std::cout << "  2c2 time: "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count()
                  << " ms\n";
    }

    // Read back the norm tiles and extract column 0 of each
    std::vector<uint16_t> norm_tiles(Mt * tq::TILE_ELEMENTS);
    EnqueueReadMeshBuffer(cq, norm_tiles, norm_buf, true);

    std::vector<bfloat16> as_bf(norm_tiles.size());
    std::memcpy(as_bf.data(), norm_tiles.data(), norm_tiles.size() * 2);
    // Output buffer has Mt tiles each of shape (32, 32). Untilize as (N, 32).
    // Column 0 of the result holds r_norm; other columns are garbage.
    constexpr uint32_t TILE_W = 32;
    std::vector<bfloat16> flat = untilize_nfaces(as_bf, num_vectors, TILE_W);
    std::vector<uint16_t> full(flat.size());
    std::memcpy(full.data(), flat.data(), flat.size() * 2);
    std::vector<uint16_t> rnorms(num_vectors);
    for (uint32_t v = 0; v < num_vectors; ++v) rnorms[v] = full[v * TILE_W];

    std::ofstream of("dump_rnorm_tile.bin", std::ios::binary);
    of.write(reinterpret_cast<const char*>(rnorms.data()), rnorms.size() * 2);
    std::cout << "  Wrote " << (rnorms.size() * 2)
              << " bytes -> dump_rnorm_tile.bin (one bf16 r_norm per vector)\n";
    return 0;
}

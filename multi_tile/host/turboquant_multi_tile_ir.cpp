// turboquant_multi_tile_ir.cpp
//
// Standalone Stage 2a (inverse rotation) host for tile-form pipeline.
//
// Reads dump_quant_centroids.bin (baseline Stage 1 output - rotated-space
// centroids, flat row-major bf16), inverse-rotates via matmul against
// precomputed H_inv = (1/sqrt(d)) * H_d * diag(D), writes dump_xhat_tile.bin.

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>

#include "tile_layout.h"

namespace tq = turboquant::multi_tile;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

// Same sign diagonal as include/codebooks.h
static const int8_t SIGN_DIAGONAL_d128[128] = {
    -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1,
    -1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1,
    1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1,
    -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1,
    1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1,
    -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1, -1,
    -1, 1, 1, -1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1,
    -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1
};

// Build H_inv: (1/sqrt(d)) * H_d * diag(D) -- D scales columns of H
static std::vector<float> build_hadamard_inverse(uint32_t d) {
    std::vector<float> H(d * d, 0.0f);
    for (uint32_t i = 0; i < d; ++i) H[i * d + i] = 1.0f;

    // WHT each row to get H_d (Hadamard matrix)
    for (uint32_t row = 0; row < d; ++row) {
        float* r = &H[row * d];
        uint32_t h = 1;
        while (h < d) {
            for (uint32_t i = 0; i < d; i += h * 2) {
                for (uint32_t j = 0; j < h; ++j) {
                    float u = r[i + j];
                    float v = r[i + j + h];
                    r[i + j]     = u + v;
                    r[i + j + h] = u - v;
                }
            }
            h <<= 1;
        }
    }

    // Scale COLUMNS of H by D (different from forward, which scaled rows)
    for (uint32_t i = 0; i < d; ++i) {
        for (uint32_t j = 0; j < d; ++j) {
            float dj = static_cast<float>(SIGN_DIAGONAL_d128[j]);
            H[i * d + j] *= dj;
        }
    }

    // Final scaling by 1/sqrt(d)
    float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
    for (uint32_t k = 0; k < d * d; ++k) H[k] *= inv_sqrt_d;

    return H;
}

static std::vector<uint16_t> tilize_bf16(
    const std::vector<float>& flat, uint32_t n_rows, uint32_t n_cols)
{
    std::vector<bfloat16> as_bf(flat.size());
    for (size_t i = 0; i < flat.size(); ++i) as_bf[i] = bfloat16(flat[i]);
    std::vector<bfloat16> tiled = tilize_nfaces(as_bf, n_rows, n_cols);
    std::vector<uint16_t> out(tiled.size());
    std::memcpy(out.data(), tiled.data(), tiled.size() * sizeof(uint16_t));
    return out;
}

static std::vector<uint16_t> load_flat_as_tiles(
    const std::string& path, uint32_t n_rows, uint32_t n_cols)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::abort(); }
    std::vector<uint16_t> raw(n_rows * n_cols);
    f.read(reinterpret_cast<char*>(raw.data()), n_rows * n_cols * 2);

    std::vector<bfloat16> as_bf(raw.size());
    static_assert(sizeof(bfloat16) == sizeof(uint16_t));
    std::memcpy(as_bf.data(), raw.data(), raw.size() * sizeof(uint16_t));
    std::vector<bfloat16> tiled = tilize_nfaces(as_bf, n_rows, n_cols);
    std::vector<uint16_t> out(tiled.size());
    std::memcpy(out.data(), tiled.data(), tiled.size() * sizeof(uint16_t));
    return out;
}

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

int main(int argc, char** argv) {
    uint32_t num_vectors = 32;
    std::string input_path = "dump_quant_centroids.bin";
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--num-vectors" && i + 1 < argc) num_vectors = std::atoi(argv[++i]);
        else if (a == "--input" && i + 1 < argc) input_path = argv[++i];
    }
    constexpr uint32_t d = tq::D;
    if (num_vectors % tq::TILE_HEIGHT != 0) {
        std::fprintf(stderr, "num_vectors must be multiple of 32\n"); return 1;
    }
    const uint32_t Mt = num_vectors / tq::TILE_HEIGHT;
    const uint32_t Kt = d / tq::TILE_WIDTH;
    const uint32_t Nt = d / tq::TILE_WIDTH;

    std::cout << "[multi_tile Stage 2a] N=" << num_vectors
              << " Mt=" << Mt << " Kt=" << Kt << " Nt=" << Nt
              << " input=" << input_path << "\n";

    auto cent_tiles = load_flat_as_tiles(input_path, num_vectors, d);
    auto h_flat     = build_hadamard_inverse(d);
    auto h_tiles    = tilize_bf16(h_flat, d, d);
    std::cout << "  Loaded centroids, built H_inv, tilized.\n";

    auto mesh_dev = MeshDevice::create_unit_mesh(6);
    auto& cq = mesh_dev->mesh_command_queue();
    MeshCoordinateRange dev_range(MeshCoordinate{0, 0}, MeshCoordinate{0, 0});

    const uint32_t tile_bytes = tq::TILE_BYTES_BF16;
    auto cent_buf = make_mesh_buf(mesh_dev.get(), Mt * Kt * tile_bytes, tile_bytes);
    auto h_buf    = make_mesh_buf(mesh_dev.get(), Kt * Nt * tile_bytes, tile_bytes);
    auto xhat_buf = make_mesh_buf(mesh_dev.get(), Mt * Nt * tile_bytes, tile_bytes);

    EnqueueWriteMeshBuffer(cq, cent_buf, cent_tiles, true);
    EnqueueWriteMeshBuffer(cq, h_buf,    h_tiles,    true);

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
        "multi_tile/kernels/inverse_rotation_reader.cpp", core_range,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc       = NOC::RISCV_0_default,
        });
    auto writer_id = CreateKernel(prog,
        "multi_tile/kernels/inverse_rotation_writer.cpp", core_range,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc       = NOC::RISCV_1_default,
        });
    std::vector<uint32_t> compute_args = {Mt, Kt, Nt};
    CreateKernel(prog,
        "multi_tile/kernels/inverse_rotation_compute.cpp", core_range,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .compile_args  = compute_args,
        });

    SetRuntimeArgs(prog, reader_id, core,
        {static_cast<uint32_t>(cent_buf->address()),
         static_cast<uint32_t>(h_buf->address()),
         Mt, Kt, Nt});
    SetRuntimeArgs(prog, writer_id, core,
        {static_cast<uint32_t>(xhat_buf->address()), Mt, Nt});

    MeshWorkload wl;
    wl.add_program(dev_range, std::move(prog));

    std::cout << "  Dispatching workload...\n";
    auto t0 = std::chrono::steady_clock::now();
    EnqueueMeshWorkload(cq, wl, true);
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "  Dispatch+execute time: "
              << std::chrono::duration<double, std::milli>(t1 - t0).count()
              << " ms\n";

    std::vector<uint16_t> xhat_tiles(Mt * Nt * tq::TILE_ELEMENTS);
    EnqueueReadMeshBuffer(cq, xhat_tiles, xhat_buf, true);

    std::vector<bfloat16> as_bf(xhat_tiles.size());
    std::memcpy(as_bf.data(), xhat_tiles.data(), xhat_tiles.size() * 2);
    std::vector<bfloat16> flat = untilize_nfaces(as_bf, num_vectors, d);
    std::vector<uint16_t> out(flat.size());
    std::memcpy(out.data(), flat.data(), flat.size() * 2);

    std::ofstream of("dump_xhat_tile.bin", std::ios::binary);
    of.write(reinterpret_cast<const char*>(out.data()), out.size() * 2);
    std::cout << "  Wrote " << (out.size() * 2)
              << " bytes -> dump_xhat_tile.bin (flat row-major)\n";
    std::cout << "[multi_tile Stage 2a] done.\n";
    return 0;
}

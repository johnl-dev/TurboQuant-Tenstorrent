// turboquant_multi_tile_proj.cpp
//
// Standalone Stage 2b2 (QJL sign projection) host:
// reads dump_r_tile.bin (residuals from 2b1), builds S_T = transpose of LFSR
// sign matrix (k=128, d=128), runs matmul B = R @ S_T, writes
// dump_b_tile.bin (the raw bf16 projections) and dump_bits_tile.bin (packed
// sign bits as 16 bytes per vector).

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

#ifndef TQ_QJL_SEED
#  define TQ_QJL_SEED 0xCAFEBABEu
#endif

static uint32_t lfsr_step(uint32_t state) {
    uint32_t lsb = state & 1u;
    state >>= 1u;
    if (lsb) state ^= 0x80200003u;
    return state;
}

// Build S^T of shape (d, k): column j is the LFSR sequence with the per-row seed.
// S[j, i] = i-th LFSR output for row j; S_T[i, j] = S[j, i].
static std::vector<float> build_qjl_S_T(uint32_t d, uint32_t k, uint32_t qjl_seed) {
    constexpr uint32_t kKnuth = 2654435761u;
    std::vector<float> S_T(d * k, 0.0f);
    for (uint32_t j = 0; j < k; ++j) {
        uint32_t row_seed = qjl_seed ^ (j * kKnuth);
        uint32_t state = (row_seed == 0u) ? 1u : row_seed;
        for (uint32_t i = 0; i < d; ++i) {
            state = lfsr_step(state);
            float s = (state & 1u) ? 1.0f : -1.0f;
            S_T[i * k + j] = s;
        }
    }
    return S_T;
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

static std::shared_ptr<MeshBuffer> make_mesh_buf(
    MeshDevice* dev, uint32_t total_bytes, uint32_t page_bytes)
{
    ReplicatedBufferConfig rep{ .size = total_bytes };
    DeviceLocalBufferConfig local{ .page_size = page_bytes, .buffer_type = BufferType::DRAM };
    return MeshBuffer::create(rep, local, dev);
}

// bf16 -> sign bit (high bit of the bf16 representation)
static uint8_t bf16_sign_positive(uint16_t b) {
    // Positive (including +0) -> 1, negative -> 0. Matches baseline: bit set if proj > 0.
    // bf16 sign bit is bit 15; 0 means non-negative, 1 means negative.
    return ((b >> 15) & 1u) ? 0u : 1u;
}

int main(int argc, char** argv) {
    uint32_t num_vectors = 32;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--num-vectors" && i + 1 < argc) {
            num_vectors = std::atoi(argv[++i]);
        }
    }
    constexpr uint32_t d = tq::D;
    constexpr uint32_t k = tq::D;  // QJL_DIM == d for our config
    if (num_vectors % tq::TILE_HEIGHT != 0) {
        std::fprintf(stderr, "num_vectors must be multiple of 32\n"); return 1;
    }
    const uint32_t Mt = num_vectors / tq::TILE_HEIGHT;
    const uint32_t Kt = d / tq::TILE_WIDTH;
    const uint32_t Nt = k / tq::TILE_WIDTH;

    std::cout << "[multi_tile Stage 2b2] N=" << num_vectors
              << " Mt=" << Mt << " Kt=" << Kt << " Nt=" << Nt
              << " qjl_seed=0x" << std::hex << uint32_t(TQ_QJL_SEED) << std::dec << "\n";

    auto r_tiles = load_flat_as_tiles("dump_r_tile.bin", num_vectors, d);
    auto S_T_flat = build_qjl_S_T(d, k, TQ_QJL_SEED);
    auto s_tiles  = tilize_bf16(S_T_flat, d, k);
    std::cout << "  Loaded r tiles, built S^T from LFSR, tilized.\n";

    auto mesh_dev = MeshDevice::create_unit_mesh(6);
    auto& cq = mesh_dev->mesh_command_queue();
    MeshCoordinateRange dev_range(MeshCoordinate{0, 0}, MeshCoordinate{0, 0});

    const uint32_t tile_bytes = tq::TILE_BYTES_BF16;
    auto r_buf = make_mesh_buf(mesh_dev.get(), Mt * Kt * tile_bytes, tile_bytes);
    auto s_buf = make_mesh_buf(mesh_dev.get(), Kt * Nt * tile_bytes, tile_bytes);
    auto b_buf = make_mesh_buf(mesh_dev.get(), Mt * Nt * tile_bytes, tile_bytes);

    EnqueueWriteMeshBuffer(cq, r_buf, r_tiles, true);
    EnqueueWriteMeshBuffer(cq, s_buf, s_tiles, true);

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
        "multi_tile/kernels/qjl_project_reader.cpp", core_range,
        DataMovementConfig{ .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default });
    auto writer_id = CreateKernel(prog,
        "multi_tile/kernels/qjl_project_writer.cpp", core_range,
        DataMovementConfig{ .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default });
    std::vector<uint32_t> compute_args = {Mt, Kt, Nt};
    CreateKernel(prog,
        "multi_tile/kernels/qjl_project_compute.cpp", core_range,
        ComputeConfig{ .math_fidelity = MathFidelity::HiFi4, .compile_args = compute_args });

    SetRuntimeArgs(prog, reader_id, core,
        {static_cast<uint32_t>(r_buf->address()),
         static_cast<uint32_t>(s_buf->address()), Mt, Kt, Nt});
    SetRuntimeArgs(prog, writer_id, core,
        {static_cast<uint32_t>(b_buf->address()), Mt, Nt});

    MeshWorkload wl;
    wl.add_program(dev_range, std::move(prog));

    std::cout << "  Dispatching workload...\n";
    auto t0 = std::chrono::steady_clock::now();
    EnqueueMeshWorkload(cq, wl, true);
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "  Dispatch+execute time: "
              << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

    std::vector<uint16_t> b_tiles(Mt * Nt * tq::TILE_ELEMENTS);
    EnqueueReadMeshBuffer(cq, b_tiles, b_buf, true);

    // Untilize to flat row-major (N, k) bf16
    std::vector<bfloat16> as_bf(b_tiles.size());
    std::memcpy(as_bf.data(), b_tiles.data(), b_tiles.size() * 2);
    std::vector<bfloat16> flat = untilize_nfaces(as_bf, num_vectors, k);
    std::vector<uint16_t> b_flat(flat.size());
    std::memcpy(b_flat.data(), flat.data(), flat.size() * 2);

    // Write raw bf16 projections
    std::ofstream of_b("dump_b_tile.bin", std::ios::binary);
    of_b.write(reinterpret_cast<const char*>(b_flat.data()), b_flat.size() * 2);
    std::cout << "  Wrote " << (b_flat.size() * 2) << " bytes -> dump_b_tile.bin\n";

    // Pack to sign bits: 16 bytes per vector (k=128 bits)
    const uint32_t bytes_per_vec = (k + 7) / 8;
    std::vector<uint8_t> packed(num_vectors * bytes_per_vec, 0);
    for (uint32_t v = 0; v < num_vectors; ++v) {
        for (uint32_t j = 0; j < k; ++j) {
            uint8_t bit = bf16_sign_positive(b_flat[v * k + j]);
            if (bit) packed[v * bytes_per_vec + (j >> 3)] |= (1u << (j & 7));
        }
    }
    std::ofstream of_bits("dump_bits_tile.bin", std::ios::binary);
    of_bits.write(reinterpret_cast<const char*>(packed.data()), packed.size());
    std::cout << "  Wrote " << packed.size() << " bytes -> dump_bits_tile.bin (packed signs)\n";

    return 0;
}

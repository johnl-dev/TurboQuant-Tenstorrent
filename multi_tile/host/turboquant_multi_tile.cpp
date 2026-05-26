// turboquant_multi_tile.cpp
//
// Stage 0 (rotation) standalone host binary using tile-form matmul on Trisc.
//
// Computes Z = X @ H_combined where:
//   - X has shape (N, 128) in tile layout
//   - H_combined = (1/sqrt(d)) * (D_diag) @ H_d   built on host, uploaded to DRAM
//
// Inputs:  reads dump_input.bin from the parent dir (same as baseline)
// Outputs: writes dump_rotated_tile.bin (tile-layout bf16, NOT flat)
//
// Compare against parent dir's dump_rotated.bin via tests/diff_rotation.py
// which untilizes the tile-layout output before diffing.

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>

#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <vector>
#include <fstream>
#include <chrono>
#include <iostream>

#include "tile_layout.h"

namespace tq = turboquant::multi_tile;

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

// -----------------------------------------------------------------------------
// bf16 conversion helpers
// -----------------------------------------------------------------------------
static uint16_t float_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    return static_cast<uint16_t>(bits >> 16);
}

static float bf16_to_float(uint16_t u) {
    uint32_t bits = static_cast<uint32_t>(u) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// -----------------------------------------------------------------------------
// SIGN_DIAGONAL_d128 -- copy of the table from parent codebooks.h
// (Could include codebooks.h directly, but that pulls in a lot. Inlined.)
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Build the Hadamard matrix H_128 via WHT-on-identity, then compose with D
// and 1/sqrt(d). Result: row-major (128, 128) bf16 array, ready to be tilized.
// -----------------------------------------------------------------------------
static std::vector<float> build_hadamard_combined(uint32_t d) {
    // Start with H = I_d
    std::vector<float> H(d * d, 0.0f);
    for (uint32_t i = 0; i < d; ++i) H[i * d + i] = 1.0f;

    // Apply WHT along the rows (gives H_d, since WHT of I is H_d)
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

    // Now multiply each ROW of H by D[row] (this gives D @ H).
    // Wait — we want (X ⊙ D) @ H_normalized, which means scaling each ROW of
    // X by D before the matmul. Equivalently, each row r of X gets D[i]*x[i].
    // In the matmul X @ H_combined, the i-th column of H_combined is used as
    // a linear combination of x[0..d-1]. So we want H_combined[i,j] = D[i] * H[i,j]
    // (scale row i of H by D[i]). Yes — scale ROWS of H by D.
    for (uint32_t i = 0; i < d; ++i) {
        float di = static_cast<float>(SIGN_DIAGONAL_d128[i]);
        for (uint32_t j = 0; j < d; ++j) {
            H[i * d + j] *= di;
        }
    }

    // Final scaling by 1/sqrt(d)
    float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
    for (uint32_t k = 0; k < d * d; ++k) H[k] *= inv_sqrt_d;

    return H;
}

// -----------------------------------------------------------------------------
// Tilize a flat row-major (n_rows, n_cols) bf16 array into tile-layout bytes.
// Result is a flat byte buffer: tile_row-major over (tile_row, tile_col),
// each tile being 32*32*2 bytes in row-major order within the tile.
// (Note: this is the LOGICAL tile layout; the device's on-chip face layout
// is further interleaved, but tt-metal's tile-aware NOC reads handle the
// conversion when reading from interleaved DRAM with tile-aligned page sizes.)
// -----------------------------------------------------------------------------
static std::vector<uint16_t> tilize_bf16(
    const std::vector<float>& flat,
    uint32_t n_rows,
    uint32_t n_cols)
{
    // Use tt-metal's canonical tilize_nfaces to get the correct on-device
    // face layout (4 quadrants of 16x16 bf16 values per 32x32 tile).
    std::vector<bfloat16> as_bf(flat.size());
    for (size_t i = 0; i < flat.size(); ++i) as_bf[i] = bfloat16(flat[i]);

    std::vector<bfloat16> tiled = tilize_nfaces(as_bf, n_rows, n_cols);

    // Reinterpret bfloat16 vector as uint16_t for the rest of the pipeline
    // (memcpy because there's no .to_packed() on this tt-metal branch)
    std::vector<uint16_t> out(tiled.size());
    std::memcpy(out.data(), tiled.data(), tiled.size() * sizeof(uint16_t));
    return out;
}

// -----------------------------------------------------------------------------
// Load dump_input.bin (flat bf16, (N, d)) and convert to tile layout.
// -----------------------------------------------------------------------------
static std::vector<uint16_t> load_input_as_tiles(
    const std::string& path,
    uint32_t n,
    uint32_t d)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::abort(); }
    std::vector<uint16_t> raw(n * d);
    f.read(reinterpret_cast<char*>(raw.data()), n * d * 2);

    std::vector<float> as_float(n * d);
    for (uint32_t i = 0; i < n * d; ++i) as_float[i] = bf16_to_float(raw[i]);

    return tilize_bf16(as_float, n, d);
}

// -----------------------------------------------------------------------------
// MeshBuffer helper (DRAM-resident, replicated across mesh = single chip for now)
// -----------------------------------------------------------------------------
static std::shared_ptr<MeshBuffer> make_mesh_buf(
    MeshDevice* dev,
    uint32_t total_bytes,
    uint32_t page_bytes)
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
    const uint32_t Kt = d / tq::TILE_WIDTH;       // 4
    const uint32_t Nt = d / tq::TILE_WIDTH;       // 4

    std::cout << "[multi_tile Stage 0] N=" << num_vectors
              << " d=" << d << " Mt=" << Mt << " Kt=" << Kt << " Nt=" << Nt << "\n";

    // -------------------------------------------------------------------------
    // Prepare data on host
    // -------------------------------------------------------------------------
    auto x_tiles = load_input_as_tiles("dump_input.bin", num_vectors, d);
    std::cout << "  Loaded " << x_tiles.size() << " bf16 values from dump_input.bin\n";

    auto h_flat  = build_hadamard_combined(d);
    auto h_tiles = tilize_bf16(h_flat, d, d);
    std::cout << "  Built H_combined and tilized to " << h_tiles.size() << " bf16 values\n";

    // -------------------------------------------------------------------------
    // Open device
    // -------------------------------------------------------------------------
    auto mesh_dev = MeshDevice::create_unit_mesh(6);  // chip 6 (PCIe-accessible L-die)
    auto& cq = mesh_dev->mesh_command_queue();
    MeshCoordinateRange dev_range(MeshCoordinate{0, 0}, MeshCoordinate{0, 0});

    // -------------------------------------------------------------------------
    // Allocate DRAM buffers
    // -------------------------------------------------------------------------
    const uint32_t tile_bytes = tq::TILE_BYTES_BF16;
    auto x_buf = make_mesh_buf(mesh_dev.get(), Mt * Kt * tile_bytes, tile_bytes);
    auto h_buf = make_mesh_buf(mesh_dev.get(), Kt * Nt * tile_bytes, tile_bytes);
    auto z_buf = make_mesh_buf(mesh_dev.get(), Mt * Nt * tile_bytes, tile_bytes);

    EnqueueWriteMeshBuffer(cq, x_buf, x_tiles, /*blocking=*/true);
    EnqueueWriteMeshBuffer(cq, h_buf, h_tiles, /*blocking=*/true);

    // -------------------------------------------------------------------------
    // Build program: single core, three kernels
    // -------------------------------------------------------------------------
    Program prog = CreateProgram();
    CoreCoord core{0, 0};
    CoreRange core_range(core, core);

    // CBs: input X (cb_in0 == c_0), input H (cb_in1 == c_1), output Z (cb_out0 == c_16)
    auto make_cb = [&](tt::CB id, uint32_t n_pages) {
        std::map<uint8_t, tt::DataFormat> spec = {
            {static_cast<uint8_t>(id), tt::DataFormat::Float16_b}
        };
        CircularBufferConfig cfg(n_pages * tile_bytes, spec);
        cfg.set_page_size(static_cast<uint8_t>(id), tile_bytes);
        CreateCircularBuffer(prog, core_range, cfg);
    };
    make_cb(tt::CB::c_in0,  2);   // double-buffered X
    make_cb(tt::CB::c_in1,  2);   // double-buffered H
    make_cb(tt::CB::c_out0, 2);   // double-buffered Z

    // Reader on Brisc
    auto reader_id = CreateKernel(prog,
        "multi_tile/kernels/rotation_reader.cpp", core_range,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc       = NOC::RISCV_0_default,
        });

    // Writer on NCRISC
    auto writer_id = CreateKernel(prog,
        "multi_tile/kernels/rotation_writer.cpp", core_range,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc       = NOC::RISCV_1_default,
        });

    // Compute on Trisc
    std::vector<uint32_t> compute_args = {Mt, Kt, Nt};
    CreateKernel(prog,
        "multi_tile/kernels/rotation_compute.cpp", core_range,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .compile_args  = compute_args,
        });

    // Runtime args
    SetRuntimeArgs(prog, reader_id, core,
        {static_cast<uint32_t>(x_buf->address()),
         static_cast<uint32_t>(h_buf->address()),
         Mt, Kt, Nt});
    SetRuntimeArgs(prog, writer_id, core,
        {static_cast<uint32_t>(z_buf->address()), Mt, Nt});

    // -------------------------------------------------------------------------
    // Run
    // -------------------------------------------------------------------------
    MeshWorkload wl;
    wl.add_program(dev_range, std::move(prog));
    std::cout << "  Dispatching workload...\n";
    auto t0 = std::chrono::steady_clock::now();
    EnqueueMeshWorkload(cq, wl, /*blocking=*/true);
    auto t1 = std::chrono::steady_clock::now();
    double dispatch_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "  Dispatch+execute time: " << dispatch_ms << " ms\n";

    // -------------------------------------------------------------------------
    // Read output back
    // -------------------------------------------------------------------------
    std::vector<uint16_t> z_tiles(Mt * Nt * tq::TILE_ELEMENTS);
    EnqueueReadMeshBuffer(cq, z_tiles, z_buf, /*blocking=*/true);

    // Untilize on host: tile-layout -> flat row-major (matches dump_rotated.bin)
    // bfloat16 is layout-compatible with uint16_t on this branch (single
    // uint16_t member, trivially copyable), so memcpy round-trips raw bits
    // without going through the lossy float->bf16 rounding path.
    static_assert(sizeof(bfloat16) == sizeof(uint16_t),
        "bfloat16 must be uint16_t-sized for raw memcpy");
    std::vector<bfloat16> z_as_bf(z_tiles.size());
    std::memcpy(z_as_bf.data(), z_tiles.data(), z_tiles.size() * sizeof(uint16_t));
    const uint32_t z_rows = num_vectors;
    const uint32_t z_cols = d;
    std::vector<bfloat16> z_flat = untilize_nfaces(z_as_bf, z_rows, z_cols);

    std::vector<uint16_t> z_out(z_flat.size());
    std::memcpy(z_out.data(), z_flat.data(), z_flat.size() * sizeof(uint16_t));

    std::ofstream out("dump_rotated_tile.bin", std::ios::binary);
    out.write(reinterpret_cast<const char*>(z_out.data()), z_out.size() * 2);
    std::cout << "  Wrote " << (z_out.size() * 2)
              << " bytes -> dump_rotated_tile.bin (flat row-major)\n";

    std::cout << "[multi_tile Stage 0] done.\n";
    return 0;
}

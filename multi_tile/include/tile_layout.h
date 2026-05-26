// tile_layout.h
//
// Helpers for the tt-metalium 32x32 tile layout used by all multi_tile stages.
//
// A "tile" is a 32x32 array of bf16 values, stored in row-major order within
// the tile (the on-device face layout is more complex — interleaved 16x16
// sub-faces — but tt-metal's tilize/untilize handles the conversion when
// reading/writing from DRAM through InterleavedAddrGen with tile-sized pages).
//
// For our matmul-form Stage 0:
//   - Input  X has shape (N, d) where N = batch_size, d = 128
//   - Output Z has shape (N, d)
//   - In tiles: (N/32) rows, (d/32) = 4 cols, total Mt * Nt tiles
//
// Tile ordering on disk: row-major over (tile_row, tile_col).

#pragma once

#include <cstdint>

namespace turboquant::multi_tile {

constexpr uint32_t TILE_HEIGHT  = 32;
constexpr uint32_t TILE_WIDTH   = 32;
constexpr uint32_t TILE_ELEMENTS = TILE_HEIGHT * TILE_WIDTH;  // 1024
constexpr uint32_t TILE_BYTES_BF16 = TILE_ELEMENTS * 2;       // 2048

constexpr uint32_t D = 128;
constexpr uint32_t D_TILES = D / TILE_WIDTH;  // 4

inline uint32_t tile_index(uint32_t tile_row, uint32_t tile_col, uint32_t n_tile_cols) {
    return tile_row * n_tile_cols + tile_col;
}

}  // namespace turboquant::multi_tile

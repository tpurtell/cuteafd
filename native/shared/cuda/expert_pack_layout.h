#pragma once
#include <stdint.h>
#ifdef __CUDACC__
#define CUTEAFD_PACK_HD __host__ __device__
#else
#define CUTEAFD_PACK_HD
#endif
/* Shared inverse lane mapping, also exercised by the CPU native fixture. */
static CUTEAFD_PACK_HD inline void cuteafd_expert_pack_coordinates(
    uint64_t index, uint32_t k, int scales, uint32_t* row, uint32_t* col) {
  const uint32_t tile_elements = scales ? 1024 : 4096;
  const uint64_t tile = index / tile_elements;
  const uint32_t local = index % tile_elements;
  const uint32_t k_tiles = (k + 127) / 128;
  const uint32_t nt = tile / k_tiles, kt = tile % k_tiles;
  if (scales) {
    *row = nt * 256 + local / 4;
    *col = kt * 4 + local % 4;
  } else {
    const uint32_t n8i = local & 3;
    const uint32_t combined = (local >> 2) & 31;
    const uint32_t n8c = (local >> 7) & 7;
    const uint32_t k32 = local >> 10;
    *row = nt * 256 + n8c * 32 + n8i * 8 + (combined >> 2);
    *col = kt * 16 + k32 * 4 + (combined & 3);
  }
}
/* Compact N128 tiles, with whole-K32 row/column tails and independent FC1
 * halves. Aligned widths retain the historical N256 bytes above. */
static CUTEAFD_PACK_HD inline void cuteafd_expert_pack_layout_coordinates(
    uint64_t index, uint32_t n, uint32_t k, uint32_t half, int scales,
    int compact, uint32_t* row, uint32_t* col) {
  if (!compact) {
    cuteafd_expert_pack_coordinates(index, k, scales, row, col);
    return;
  }
  const uint32_t divisor = scales ? 32 : 8;
  const uint32_t columns = k / divisor;
  uint32_t first = 0;
  if (half) {
    const uint64_t elements = (uint64_t)half * columns;
    first = (uint32_t)(index / elements) * half;
    index %= elements;
    n = half;
  }
  const uint32_t nt = (uint32_t)(index / ((uint64_t)128 * columns));
  index %= (uint64_t)128 * columns;
  const uint32_t rows = n - nt * 128 < 128 ? n - nt * 128 : 128;
  const uint32_t block_columns = scales ? 4 : 16;
  const uint32_t kt = (uint32_t)(index / (rows * block_columns));
  const uint32_t local = (uint32_t)(index % (rows * block_columns));
  const uint32_t tail_columns = columns - kt * block_columns;
  const uint32_t cols = tail_columns < block_columns ? tail_columns : block_columns;
  if (scales) {
    *row = first + nt * 128 + local / cols;
    *col = kt * block_columns + local % cols;
  } else {
    const uint32_t n32 = rows / 32;
    *row = first + nt * 128 + ((local / 128) % n32) * 32
        + (local % 4) * 8 + (local / 16) % 8;
    *col = kt * block_columns + local / (128 * n32) * 4 + (local / 4) % 4;
  }
}
static CUTEAFD_PACK_HD inline uint64_t cuteafd_nvfp4_scale_offset(
    uint64_t row, uint64_t col, uint64_t padded_cols) {
  return (row / 128) * (padded_cols * 128) + (col / 4) * 512
      + (row % 32) * 16 + ((row / 32) % 4) * 4 + col % 4;
}
#undef CUTEAFD_PACK_HD

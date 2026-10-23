// CPU stand-ins for load-only native kernels, linked into the fake native DSO:
// the same shared rule the CUDA kernels run, executed sequentially.
#include <stdint.h>

#include "../../../../../native/shared/cuda/fp8_load_quant.h"

extern "C" int32_t cuteafd_fp8_load_quant(const void* jobs, int32_t count, void* stream) {
  (void)stream;
  if (count < 0) return 1;
  const auto* list = static_cast<const cuteafd_load_quant::Job*>(jobs);
  for (int32_t i = 0; i < count; ++i) cuteafd_load_quant::run_job(list[i]);
  return 0;
}

extern "C" int32_t cuteafd_fp8_load_quant_abi(uint64_t* job_bytes, uint64_t* segment_bytes, uint64_t* max_cols) {
  *job_bytes = sizeof(cuteafd_load_quant::Job);
  *segment_bytes = sizeof(cuteafd_load_quant::Segment);
  *max_cols = 512 * 128;
  return 0;
}

extern "C" int32_t cuteafd_load_widen_bf16_f32(const void* src, void* dst, int64_t n, void* stream) {
  (void)stream;
  for (int64_t i = 0; i < n; ++i)
    static_cast<uint32_t*>(dst)[i] = cuteafd_load_quant::widen_bf16(static_cast<const uint16_t*>(src)[i]);
  return 0;
}

extern "C" int32_t cuteafd_load_narrow_f32_bf16(const void* src, void* dst, int64_t n, void* stream) {
  (void)stream;
  for (int64_t i = 0; i < n; ++i)
    static_cast<uint16_t*>(dst)[i] = cuteafd_load_quant::narrow_f32(static_cast<const uint32_t*>(src)[i]);
  return 0;
}

extern "C" int32_t cuteafd_load_kvb_absorb(const void* kv_b, void* uk, void* uv, int64_t h0, int64_t heads,
                                           int64_t nope, int64_t v, int64_t lat, void* stream) {
  (void)stream;
  const auto* source = static_cast<const uint16_t*>(kv_b);
  for (int64_t i = 0; i < heads * lat * nope; ++i)
    static_cast<uint16_t*>(uk)[i] = cuteafd_load_quant::absorb_uk(source, h0, nope, v, lat, i);
  for (int64_t i = 0; i < heads * v * lat; ++i)
    static_cast<uint16_t*>(uv)[i] = cuteafd_load_quant::absorb_uv(source, h0, nope, v, lat, i);
  return 0;
}

extern "C" int32_t cuteafd_load_kvb_split_fp8(const void* kv_b, const void* grid, void* uk, void* uk_scale, void* uv,
                                              void* uv_scale, int64_t row0, int64_t heads, int64_t nope, int64_t v,
                                              int64_t lat, void* stream) {
  (void)stream;
  namespace lq = cuteafd_load_quant;
  const auto* q = static_cast<const uint8_t*>(kv_b);
  const auto* g = static_cast<const float*>(grid);
  for (int64_t i = 0; i < heads * lat * nope; ++i) static_cast<uint8_t*>(uk)[i] = lq::absorb_uk_fp8(q, 0, nope, v, lat, i);
  for (int64_t i = 0; i < heads * v * lat; ++i) static_cast<uint8_t*>(uv)[i] = lq::absorb_uv_fp8(q, 0, nope, v, lat, i);
  for (int64_t i = 0; i < heads * lat * (nope / 64); ++i)
    static_cast<float*>(uk_scale)[i] = lq::absorb_uk_scale(g, row0, nope, v, lat, i);
  for (int64_t i = 0; i < heads * v * (lat / 64); ++i)
    static_cast<float*>(uv_scale)[i] = lq::absorb_uv_scale(g, row0, nope, v, lat, i);
  return 0;
}

extern "C" int32_t cuteafd_load_bf16_product_f32(const void* a, const void* b, void* dst, int64_t n, void* stream) {
  (void)stream;
  for (int64_t i = 0; i < n; ++i)
    static_cast<uint32_t*>(dst)[i] = cuteafd_load_quant::product_bf16_f32(static_cast<const uint16_t*>(a)[i],
                                                                          static_cast<const uint16_t*>(b)[i]);
  return 0;
}

extern "C" int32_t cuteafd_load_trellis_tiles(const void* src, void* dst, int64_t rows, int64_t k0, int64_t tiles,
                                              int64_t tile_bytes, int32_t side, int64_t k16, void* stream) {
  (void)stream;
  if (rows < 1 || k0 < 0 || tiles < 1 || tile_bytes < 16 || tile_bytes % 16 != 0 || (side != 0 && side != 1)
      || k0 + rows > k16 || tiles % 8 != 0) return 1;
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t c = 0; c < tiles; ++c)
      memcpy(static_cast<uint8_t*>(dst) + cuteafd_load_quant::trellis_tile(side, k0 + r, c, tiles, k16) * tile_bytes,
             static_cast<const uint8_t*>(src) + (r * tiles + c) * tile_bytes, size_t(tile_bytes));
  return 0;
}

#include "../../../../../native/shared/cuda/load_scale_gather.h"

extern "C" int32_t cuteafd_load_scale_gather(const void* grid, const void* index, void* row_major, void* k_major,
                                             int64_t rows, int64_t kb, int64_t grid_cols, int64_t col0, void* stream) {
  (void)stream;
  if (!grid || !index || rows < 1 || kb < 1 || col0 < 0 || col0 + kb > grid_cols || (!row_major && !k_major)) return 1;
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t b = 0; b < kb; ++b) {
      const float v = cuteafd_scale_gather::gathered(static_cast<const float*>(grid),
          static_cast<const uint32_t*>(index), grid_cols, col0, r, b);
      if (row_major) static_cast<float*>(row_major)[r * kb + b] = v;
      if (k_major) static_cast<float*>(k_major)[b * rows + r] = v;
    }
  return 0;
}

// CPU stand-ins for load-only native kernels, linked into the fake native DSO:
// the same shared rule the CUDA kernels run, executed sequentially.
#include <stdint.h>

#include "../../../../../native/shared/cuda/fp8_load_quant.h"

#include <cmath>

// Preserve the warp's accumulation/reduction order for deferred-load tests.
static int32_t markov_norms(const void* source, void* output, int32_t rows, int32_t cols, bool slack) {
  if (!source || !output || rows < 1 || cols != (slack ? 512 : 256)) return 1;
  const auto* values = static_cast<const uint16_t*>(source);
  for (int32_t r = 0; r < rows; ++r) {
    float lanes[32] = {};
    for (int32_t lane = 0; lane < 32; ++lane)
      for (int32_t c = lane; c < cols; c += 32) {
        const uint32_t bits = uint32_t(values[int64_t(r) * cols + c]) << 16;
        float value;
        memcpy(&value, &bits, sizeof(value));
        lanes[lane] = std::fma(value, value, lanes[lane]);
      }
    for (int32_t offset = 16; offset; offset >>= 1) {
      float next[32];
      for (int32_t lane = 0; lane < 32; ++lane) next[lane] = lanes[lane] + lanes[lane ^ offset];
      memcpy(lanes, next, sizeof(lanes));
    }
    const float norm = std::sqrt(lanes[0]);
    static_cast<float*>(output)[r] = slack ? std::fma(norm, 1.0001f, 1e-6f) : norm;
  }
  return 0;
}

extern "C" int32_t cuteafd_glmf_dspark_markov_norms(const void* source, void* output, int32_t rows,
                                                  int32_t cols, void* stream) {
  (void)stream;
  return markov_norms(source, output, rows, cols, false);
}

extern "C" int32_t cuteafd_ml4_dspark_markov_norms(const void* source, void* output, int32_t rows,
                                                 int32_t cols, void* stream) {
  (void)stream;
  return markov_norms(source, output, rows, cols, true);
}

// A synthetic device count keeps owner construction CPU-only.
extern "C" int32_t cuteafd_sm_count() { return 1; }

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

extern "C" int32_t cuteafd_load_narrow_f32_bf16(const void* src, void* dst, int64_t n, int32_t sms, void* stream) {
  (void)stream;
  if (n < 1 || sms < 1) return 1;
  for (int64_t i = 0; i < n; ++i)
    static_cast<uint16_t*>(dst)[i] = cuteafd_load_quant::narrow_f32(static_cast<const uint32_t*>(src)[i]);
  return 0;
}

extern "C" int32_t cuteafd_load_narrow_i64_i32(const void* src, void* dst, int64_t n,
                                               uint32_t* invalid, void* stream) {
  (void)stream;
  if (!src || !dst || !invalid || n < 1) return 1;
  for (int64_t i = 0; i < n; ++i)
    if (!cuteafd_load_quant::narrow_i64(static_cast<const int64_t*>(src)[i], static_cast<int32_t*>(dst) + i))
      *invalid = 1;
  return 0;
}

extern "C" int32_t cuteafd_bf16_f32_add_async(const uint16_t* rank0, const uint16_t* rank1,
                                              float* dst, size_t n, int32_t sms, void* stream) {
  (void)stream;
  if (n < 1 || sms < 1) return 1;
  for (size_t i = 0; i < n; ++i)
    dst[i] = cuteafd_load_quant::partial_add_bf16_f32(rank0[i], rank1[i]);
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

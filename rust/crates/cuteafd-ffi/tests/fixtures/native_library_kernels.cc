// CPU stand-ins for load-only native kernels, linked into the fake native DSO:
// the same shared rule the CUDA kernels run, executed sequentially.
#include <stdint.h>
#include <algorithm>

#include "../../../../../native/shared/cuda/fp8_load_quant.h"

// A synthetic device count keeps owner construction CPU-only.
extern "C" int32_t cuteafd_sm_count() { return 1; }

// Metadata-only drafter queries and its one load-time derived array. These
// stand-ins exercise owner construction, not attention or draft numerics.
extern "C" uint64_t cuteafd_block_dspark_attention_workspace(int32_t sequences, int32_t heads,
    int32_t kv_heads, int32_t block, int32_t keys, int32_t dim) {
  if (sequences < 1 || heads < 1 || kv_heads < 1 || heads % kv_heads || block < 1 || keys < block ||
      (dim != 64 && dim != 128)) return 0;
  const int64_t slices = (int64_t(heads / kv_heads) * block + 31) / 32;
  const int64_t base = kv_heads * slices;
  const int64_t partitions = std::min<int64_t>((keys + 31) / 32, (2 + base - 1) / base);
  const int64_t scalar = (std::min(keys, 1040) + 127) / 128 * (dim == 64 ? 32 : 16);
  return uint64_t(sequences) * kv_heads * std::max(slices * partitions * 32, scalar) * (dim + 2) * 4;
}

extern "C" uint64_t cuteafd_glmf_dspark_markov_workspace(int32_t sequences, int32_t block) {
  return uint64_t(sequences) * 296 * 8 + uint64_t(sequences) * block * 4;
}

// Mirrors native/families/glm5/cuda/glm_dflash.cu:429-430 and :176-177;
// the CUDA-linked library is not loaded by this CPU-only ownership fixture.
extern "C" uint64_t cuteafd_glm_dflash_topk_workspace(int32_t rows) {
  return uint64_t(rows) * 64 * 16 * 8;
}

// The native d128/d64 entry points each reserve two query slices.
extern "C" uint64_t cuteafd_glm_dflash_attention_workspace(int32_t sequences, int32_t kv_heads,
    int32_t max_keys) {
  return cuteafd_block_dspark_attention_workspace(sequences, 2 * kv_heads, kv_heads, 32, max_keys, 128);
}
extern "C" uint64_t cuteafd_glm_dflash_attention_workspace_d64(int32_t sequences, int32_t kv_heads,
    int32_t max_keys) {
  return cuteafd_block_dspark_attention_workspace(sequences, 2 * kv_heads, kv_heads, 32, max_keys, 64);
}

extern "C" int32_t cuteafd_glmf_dspark_markov_norms(const void* w2, void* norms, int32_t vocab,
    int32_t rank, void* stream) {
  (void)stream;
  if (!w2 || !norms || vocab < 1 || rank != 256) return 1;
  for (int row = 0; row < vocab; ++row) {
    float sum = 0;
    for (int col = 0; col < rank; ++col) {
      const uint32_t bits = uint32_t(static_cast<const uint16_t*>(w2)[row * rank + col]) << 16;
      float value;
      memcpy(&value, &bits, sizeof(value));
      sum += value * value;
    }
    static_cast<float*>(norms)[row] = sum;
  }
  return 0;
}

extern "C" size_t cuteafd_fp8_linear_workspace(int32_t rows, int32_t k, int32_t n, int32_t mode) {
  if (rows < 1 || k < 128 || n < 16 || mode < 0 || mode > 2) return 0;
  const auto align = [](size_t bytes) { return (bytes + 255) / 256 * 256; };
  const int blocks = k / 128;
  const int wanted = std::min(blocks, std::max(1, (16 + n / 16 - 1) / (n / 16)));
  const int span = (blocks + wanted - 1) / wanted;
  const int splits = (blocks + span - 1) / span;
  const int cap = mode == 0 ? 64 : 128;
  return align(2 * cap * 4) + align((k + 255) / 256 * cap * 4) + align(size_t(k) * cap * 2) +
      (mode == 2 ? align(size_t(k / 128) * cap * 4) : 0) +
      (splits > 1 ? align(size_t(splits) * std::min(rows, cap) * n * 4) : 0);
}

extern "C" size_t cuteafd_fp8_w8a16_workspace(int32_t rows, int32_t k, int32_t n) {
  return cuteafd_fp8_linear_workspace(rows, k, n, 0);
}

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

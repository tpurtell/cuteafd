// Load-time BF16 -> FP8 quantization straight from host-mapped push ring
// slots into the final device operand, bit-for-bit the coordinator loaders'
// host rule (fp8_load_quant.h). One CTA per scale unit (a 128-row Block band,
// or one row for Row128/Channel). Order-independent steps (amax, per-element
// quantization, each row's float64 error) run in parallel; the order-dependent
// fold of row errors runs on one thread in the host's order.
#include <cuda_runtime.h>
#include <stdint.h>

#include <algorithm>

#include "fp8_load_quant.h"

namespace lq = cuteafd_load_quant;

namespace {
using Job = lq::Job;

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 512;  // cols <= 65536

__global__ void load_quant_kernel(const Job* jobs) {
  const Job job = jobs[blockIdx.x];
  const lq::Strided<lq::Segmented> src{lq::Segmented{job.segments, job.count}, job.cols, job.col0, job.pitch};
  const int64_t rows = job.rows, cols = job.cols, kb = cols / 128;
  __shared__ float scales[kMaxBlocks];
  __shared__ double ra[128], rp[128];
  // Per-block amax: warp w reduces blocks w, w + warps, ...
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  for (int64_t b = warp; b < kb; b += kThreads / 32) {
    float m = 0.0f;
    for (int64_t i = lane; i < rows * 128; i += 32) m = lq::fmax32(m, lq::fabs32(src.at((i / 128) * cols + b * 128 + i % 128)));
    for (int offset = 16; offset; offset >>= 1) m = lq::fmax32(m, __shfl_xor_sync(0xffffffffu, m, offset));
    if (lane == 0) scales[b] = m;
  }
  __syncthreads();
  if (job.rule != lq::kBest) {
    if (threadIdx.x == 0) lq::select_scales(src, scales, rows, cols, job.layout, job.rule, ra, rp);
  } else {
    // Best: Channel collapses to one candidate pair on block 0's columns.
    if (threadIdx.x == 0 && job.layout == lq::kChannel) {
      float row_amax = 0.0f;
      for (int64_t b = 0; b < kb; ++b) row_amax = lq::fmax32(row_amax, scales[b]);
      for (int64_t b = 0; b < kb; ++b) scales[b] = row_amax;
    }
    __syncthreads();
    const int64_t blocks = job.layout == lq::kChannel ? 1 : kb;
    for (int64_t b = 0; b < blocks; ++b) {
      const float amax = scales[b];
      const float sa = lq::amax_scale(amax), sp = lq::pow2_scale(amax);
      if (sp != sa) {
        for (int64_t r = threadIdx.x; r < rows; r += kThreads) {
          ra[r] = lq::row_error(src, cols, r, b, sa);
          rp[r] = lq::row_error(src, cols, r, b, sp);
        }
        __syncthreads();
        if (threadIdx.x == 0) scales[b] = lq::best_scale(sa, sp, ra, rp, rows);
        __syncthreads();
      } else if (threadIdx.x == 0) {
        scales[b] = sa;
      }
      __syncthreads();
    }
    if (threadIdx.x == 0 && job.layout == lq::kChannel)
      for (int64_t b = 1; b < kb; ++b) scales[b] = scales[0];
  }
  __syncthreads();
  for (int64_t b = threadIdx.x; b < kb; b += kThreads) job.scales[b * job.scale_stride] = scales[b];
  for (int64_t i = threadIdx.x; i < rows * cols; i += kThreads)
    job.q[(i / cols) * job.q_stride + i % cols] = lq::e4m3(lq::fdiv(src.at(i), scales[(i % cols) / 128]));
}
}  // namespace

// `jobs`: `count` units in device-visible (pinned host or device) memory, each
// reading host-mapped fragments and writing its final E4M3 rows and scales.
extern "C" int32_t cuteafd_fp8_load_quant(const void* jobs, int32_t count, void* stream) {
  if (count < 0) return cudaErrorInvalidValue;
  if (count == 0) return cudaSuccess;
  load_quant_kernel<<<unsigned(count), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const Job*>(jobs));
  return cudaGetLastError();
}

namespace {
__global__ void widen_kernel(const uint16_t* src, uint32_t* dst, int64_t n) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
    dst[i] = lq::widen_bf16(src[i]);
}
__global__ void narrow_kernel(const uint32_t* src, uint16_t* dst, int64_t n) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
    dst[i] = lq::narrow_f32(src[i]);
}
__global__ void absorb_kernel(const uint16_t* kv_b, uint16_t* uk, uint16_t* uv, int64_t h0, int64_t heads,
                              int64_t nope, int64_t v, int64_t lat) {
  const int64_t nk = heads * lat * nope, nv = heads * v * lat;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < nk + nv; i += int64_t(gridDim.x) * blockDim.x) {
    if (i < nk) uk[i] = lq::absorb_uk(kv_b, h0, nope, v, lat, i);
    else uv[i - nk] = lq::absorb_uv(kv_b, h0, nope, v, lat, i - nk);
  }
}
__global__ void absorb_fp8_kernel(const uint8_t* kv_b, const float* grid, uint8_t* uk, float* uk_s, uint8_t* uv,
                                  float* uv_s, int64_t row0, int64_t heads, int64_t nope, int64_t v, int64_t lat) {
  const int64_t nk = heads * lat * nope, nv = heads * v * lat;
  const int64_t sk = heads * lat * (nope / 64), sv = heads * v * (lat / 64);
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < nk + nv + sk + sv;
       i += int64_t(gridDim.x) * blockDim.x) {
    if (i < nk) uk[i] = lq::absorb_uk_fp8(kv_b, 0, nope, v, lat, i);
    else if (i < nk + nv) uv[i - nk] = lq::absorb_uv_fp8(kv_b, 0, nope, v, lat, i - nk);
    else if (i < nk + nv + sk) uk_s[i - nk - nv] = lq::absorb_uk_scale(grid, row0, nope, v, lat, i - nk - nv);
    else uv_s[i - nk - nv - sk] = lq::absorb_uv_scale(grid, row0, nope, v, lat, i - nk - nv - sk);
  }
}
__global__ void product_kernel(const uint16_t* a, const uint16_t* b, uint32_t* dst, int64_t n) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
    dst[i] = lq::product_bf16_f32(a[i], b[i]);
}
__global__ void trellis_tiles_kernel(const uint4* src, uint4* dst, int64_t rows, int64_t k0, int64_t tiles,
                                     int64_t words, int32_t side, int64_t k16) {
  const int64_t n = rows * tiles * words;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
    const int64_t tile = i / words, r = tile / tiles, c = tile % tiles;
    dst[lq::trellis_tile(side, k0 + r, c, tiles, k16) * words + i % words] = src[i];
  }
}
unsigned grid_for(int64_t n) {
  int device = 0, sms = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  return unsigned(std::min<int64_t>((n + 255) / 256, int64_t(sms > 0 ? sms : 1) * 8));
}
}  // namespace

// BF16 `src` (device-visible, e.g. pinned host) widened to F32 `dst`, `n` elements.
extern "C" int32_t cuteafd_load_widen_bf16_f32(const void* src, void* dst, int64_t n, void* stream) {
  if (n < 1) return cudaErrorInvalidValue;
  widen_kernel<<<grid_for(n), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint16_t*>(src), static_cast<uint32_t*>(dst), n);
  return cudaGetLastError();
}

// F32 -> BF16 (round to nearest even) of n values (see fp8_load_quant.h).
extern "C" int32_t cuteafd_load_narrow_f32_bf16(const void* src, void* dst, int64_t n, void* stream) {
  if (n < 1) return cudaErrorInvalidValue;
  narrow_kernel<<<grid_for(n), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint32_t*>(src), static_cast<uint16_t*>(dst), n);
  return cudaGetLastError();
}

// MLA kv_b absorbed per head into uk/uv for heads [h0, h0 + heads) (see fp8_load_quant.h).
extern "C" int32_t cuteafd_load_kvb_absorb(const void* kv_b, void* uk, void* uv, int64_t h0, int64_t heads,
                                           int64_t nope, int64_t v, int64_t lat, void* stream) {
  if (heads < 1 || nope < 1 || v < 1 || lat < 1 || h0 < 0) return cudaErrorInvalidValue;
  absorb_kernel<<<grid_for(heads * (nope + v) * lat), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint16_t*>(kv_b), static_cast<uint16_t*>(uk), static_cast<uint16_t*>(uv), h0, heads, nope,
      v, lat);
  return cudaGetLastError();
}

// FP8 kv_b rows of `heads` heads (the first at kv_b row `row0`) and the whole
// 128x128 grid split into the GLM FP8 uk/uv operands (see fp8_load_quant.h);
// nope a whole number of 64-wide tiles, lat of 128-wide grid blocks.
extern "C" int32_t cuteafd_load_kvb_split_fp8(const void* kv_b, const void* grid, void* uk, void* uk_scale, void* uv,
                                              void* uv_scale, int64_t row0, int64_t heads, int64_t nope, int64_t v,
                                              int64_t lat, void* stream) {
  if (heads < 1 || nope < 64 || v < 1 || lat < 128 || row0 < 0 || nope % 64 != 0 || lat % 128 != 0)
    return cudaErrorInvalidValue;
  const int64_t n = heads * ((nope + v) * lat + lat * (nope / 64) + v * (lat / 64));
  absorb_fp8_kernel<<<grid_for(n), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(kv_b), static_cast<const float*>(grid), static_cast<uint8_t*>(uk),
      static_cast<float*>(uk_scale), static_cast<uint8_t*>(uv), static_cast<float*>(uv_scale), row0, heads, nope, v, lat);
  return cudaGetLastError();
}

// BF16 `a` x `b` promoted to F32 products in `dst`, n values (see fp8_load_quant.h).
extern "C" int32_t cuteafd_load_bf16_product_f32(const void* a, const void* b, void* dst, int64_t n, void* stream) {
  if (n < 1) return cudaErrorInvalidValue;
  product_kernel<<<grid_for(n), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint16_t*>(a), static_cast<const uint16_t*>(b), static_cast<uint32_t*>(dst), n);
  return cudaGetLastError();
}

// Staged EXL3 trellis rows reordered into a native shard (see fp8_load_quant.h).
extern "C" int32_t cuteafd_load_trellis_tiles(const void* src, void* dst, int64_t rows, int64_t k0, int64_t tiles,
                                              int64_t tile_bytes, int32_t side, int64_t k16, void* stream) {
  if (rows < 1 || k0 < 0 || tiles < 1 || tile_bytes < 16 || tile_bytes % 16 != 0 || (side != 0 && side != 1)
      || k0 + rows > k16 || (side == 1 && tiles % 8 != 0) || (side == 0 && tiles % 8 != 0)) return cudaErrorInvalidValue;
  const int64_t words = tile_bytes / 16;
  trellis_tiles_kernel<<<grid_for(rows * tiles * words), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint4*>(src), static_cast<uint4*>(dst), rows, k0, tiles, words, side, k16);
  return cudaGetLastError();
}

// The Job ABI, for the Rust side to check before launching.
extern "C" int32_t cuteafd_fp8_load_quant_abi(uint64_t* job_bytes, uint64_t* segment_bytes, uint64_t* max_cols) {
  *job_bytes = sizeof(Job);
  *segment_bytes = sizeof(lq::Segment);
  *max_cols = uint64_t(kMaxBlocks) * 128;
  return 0;
}

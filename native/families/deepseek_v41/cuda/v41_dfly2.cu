#include "cuteafd_v41_dfly2.h"
#include "v41_capacity.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <mma.h>
#include <stdint.h>

namespace {
using namespace nvcuda;
using bf16 = __nv_bfloat16;

static_assert(sizeof(cuteafd_v41_dfly2_request_t) == 80);

constexpr int kVocabSize = 129280;
constexpr int kHiddenSize = 5120;
constexpr int kHeadDim = 512;
constexpr int kHeads = 64;
constexpr int kKvStride = 520;
constexpr int kOutputStride = 516;
constexpr int kProbabilityStride = 80;
constexpr int kKvBytes = 64 * kKvStride * 2;
constexpr int kOutputBytes = 16 * kOutputStride * 4;
constexpr int kSharedBytes = kKvBytes + kOutputBytes + 192 + 512;
constexpr float kScale = 0.04419417382415922f;  // 1 / sqrt(512)
constexpr int kVocabTiles = 64;

bool span(const void* ptr, uint64_t n, uint32_t a) {
  const auto p = reinterpret_cast<uintptr_t>(ptr);
  return p && p % a == 0 && p <= UINTPTR_MAX - n;
}
bool disjoint(const void* a, uint64_t n, const void* b, uint64_t m) {
  const auto x = reinterpret_cast<uintptr_t>(a);
  const auto y = reinterpret_cast<uintptr_t>(b);
  return x + n <= y || y + m <= x;
}

#if defined(__CUDA_ARCH_SPECIFIC__) && __CUDA_ARCH_SPECIFIC__ == 1200
__device__ __forceinline__ uint32_t packed_fp8_pair(uint16_t input, uint32_t factors) {
  uint32_t output;
  asm("{ .reg .b32 values; cvt.rn.bf16x2.e4m3x2 values, %1; mul.bf16x2 %0, values, %2; }"
      : "=r"(output)
      : "h"(input), "r"(factors));
  return output;
}
#endif

__device__ __forceinline__ uint16_t fp4_bf16_bits(unsigned value) {
  const unsigned magnitude = value & 7;
  return uint16_t(((value & 8) << 12) | (magnitude < 2 ? (magnitude ? 0x3f00 : 0) : 0x3f00 + magnitude * 64));
}

__device__ __forceinline__ uint64_t packed_fp4_quad(uint16_t input, uint8_t scale) {
#if defined(__CUDA_ARCH_SPECIFIC__) && __CUDA_ARCH_SPECIFIC__ == 1200
  uint32_t low, high;
  const uint16_t packed_low = input & 255, packed_high = input >> 8;
  asm("{ .reg .b8 x; cvt.u8.u16 x, %1; cvt.rn.bf16x2.e2m1x2 %0, x; }" : "=r"(low) : "h"(packed_low));
  asm("{ .reg .b8 x; cvt.u8.u16 x, %1; cvt.rn.bf16x2.e2m1x2 %0, x; }" : "=r"(high) : "h"(packed_high));
  uint32_t factors, a, b;
  const uint16_t scales = uint16_t(scale) | (uint16_t(scale) << 8);
  asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(factors) : "h"(scales));
  asm("mul.bf16x2 %0, %1, %2;" : "=r"(a) : "r"(low), "r"(factors));
  asm("mul.bf16x2 %0, %1, %2;" : "=r"(b) : "r"(high), "r"(factors));
  return uint64_t(a) | (uint64_t(b) << 32);
#else
  __nv_fp8_e4m3 f;
  f.__x = scale;
  uint64_t result = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float value = __bfloat162float(__ushort_as_bfloat16(fp4_bf16_bits((input >> (j * 4)) & 15)));
    result |= uint64_t(__bfloat16_as_ushort(__float2bfloat16_rn(__fmul_rn(value, float(f))))) << (j * 16);
  }
  return result;
#endif
}

__device__ __forceinline__ float warp_max(float x) {
  for (int n = 16; n; n >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, n));
  return x;
}
__device__ __forceinline__ float warp_sum(float x) {
  for (int n = 16; n; n >>= 1) x = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, n));
  return x;
}

// Top two bits of ref: 0 = window ring (FP8), 1 = source pool (FP8 or FP4),
// 2 = bidirectional BF16 draft block.
__device__ __forceinline__ uint64_t locate_dfly2(
    const cuteafd_v41_dfly2_request_t& d, int key, int win_count,
    uint64_t win_first, bool has_draft, int block_size) {
  if (key < win_count) {
    const uint64_t pos = win_first + uint64_t(key);
    return pos % 128;
  }
  const int sel_idx = key - win_count;
  if (sel_idx < 512) {
    const int32_t id = d.selected[sel_idx];
    if (id < 0 || uint64_t(id) >= d.committed_end) return UINT64_MAX;
    const uint64_t pos = uint64_t(id);
    if (pos >= uint64_t(d.page_stride) * 256) return UINT64_MAX;
    const uint64_t physical = uint64_t(d.source_pages[pos / 256]) * 256 + (pos % 256);
    return physical < d.source_capacity ? ((1ull << 62) | physical) : UINT64_MAX;
  }
  if (has_draft) {
    const int draft_idx = sel_idx - 512;
    if (draft_idx < block_size) return (2ull << 62) | uint64_t(draft_idx);
  }
  return UINT64_MAX;
}

template <bool Split, bool SourceFP4>
__global__ __launch_bounds__(128, 1) void dfly2_attend(
    const bf16* query, const bf16* draft_kv, const float* sink,
    const cuteafd_v41_dfly2_request_t* requests_dev, bf16* output,
    uint8_t* alive_out, float* partial, int block_size) {
  const int row = blockIdx.x, group = blockIdx.y;
  const int req = row / block_size;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const uint64_t base = (uint64_t(row) * kHeads + group * 16) * kHeadDim;
  const uint64_t partial_base = ((uint64_t(row) * gridDim.z + blockIdx.z) * kHeads + group * 16) * 514;
  const auto d = requests_dev[req];

  const uint64_t ring_start = d.committed_end > 128 ? d.committed_end - 128 : 0;
  const uint64_t win_first = d.window_begin > ring_start ? d.window_begin : ring_start;
  const int win_count = d.committed_end > win_first ? int(d.committed_end - win_first) : 0;
  const bool has_draft = draft_kv != nullptr;
  const int count = win_count + 512 + (has_draft ? block_size : 0);

  extern __shared__ __align__(32) unsigned char memory[];
  auto* kv = reinterpret_cast<bf16*>(memory);
  auto* scratch = reinterpret_cast<float*>(memory + kKvBytes);
  auto* probability = reinterpret_cast<bf16*>(scratch);
  auto* maximum = reinterpret_cast<float*>(memory + kKvBytes + kOutputBytes);
  float* sum = maximum + 16;
  float* rescale = sum + 16;
  auto* refs = reinterpret_cast<uint64_t*>(memory + kKvBytes + kOutputBytes + 192);

  if (tid < 16) {
    maximum[tid] = -1e30f;
    sum[tid] = 0.0f;
  }
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
#pragma unroll
  for (int t = 0; t < 8; ++t) wmma::fill_fragment(acc[t], 0.0f);

  bool empty = true;
  const int tiles = (count + 63) / 64;
  const int per_part = Split ? (tiles + int(gridDim.z) - 1) / int(gridDim.z) : tiles;
  const int first = Split ? int(blockIdx.z) * per_part * 64 : 0;
  const int last = Split ? min(count, int(blockIdx.z + 1) * per_part * 64) : count;

  for (int start = first; start < last; start += 64) {
    if (tid < 64) {
      refs[tid] = start + tid < count
                      ? locate_dfly2(d, start + tid, win_count, win_first, has_draft, block_size)
                      : UINT64_MAX;
    }
    if (!__syncthreads_or(tid < 64 && refs[tid] != UINT64_MAX)) continue;

    for (int key = warp; key < 64; key += 4) {
      const uint64_t ref = refs[key];
      for (int block = 0; block < 4; ++block) {
        const int col = block * 128 + lane * 4;
        uint64_t packed = 0;
        if (ref != UINT64_MAX) {
          const int tag = int(ref >> 62);
          const uint64_t physical = ref & ((1ull << 62) - 1);
          if (tag == 2) {
            const bf16* src = draft_kv + (uint64_t(req) * block_size + physical) * kHeadDim + col;
            packed = *reinterpret_cast<const uint64_t*>(src);
          } else if (SourceFP4 && tag == 1) {
            const uint8_t* src = d.source_values + physical * 256 + col / 2;
            const uint16_t bytes = (reinterpret_cast<uintptr_t>(src) & 1)
                                       ? (uint16_t(src[0]) | (uint16_t(src[1]) << 8))
                                       : *reinterpret_cast<const uint16_t*>(src);
            packed = packed_fp4_quad(bytes, d.source_scales[physical * 32 + col / 16]);
          } else {
            const uint8_t* values = tag == 0 ? d.window_values : d.source_values;
            const uint8_t* scales = tag == 0 ? d.window_scales : d.source_scales;
            const uint8_t* src = values + physical * 512 + col;
            uint32_t bytes;
            if ((reinterpret_cast<uintptr_t>(src) & 3) == 0)
              bytes = *reinterpret_cast<const uint32_t*>(src);
            else
              bytes = uint32_t(src[0]) | (uint32_t(src[1]) << 8) | (uint32_t(src[2]) << 16) |
                      (uint32_t(src[3]) << 24);
            const uint8_t exponent = scales[physical * 16 + col / 32];
#if defined(__CUDA_ARCH_SPECIFIC__) && __CUDA_ARCH_SPECIFIC__ == 1200
            const uint32_t factor = exponent ? (uint32_t(exponent) << 7) : 0x40;
            const uint32_t factors = factor | (factor << 16);
            packed = uint64_t(packed_fp8_pair(uint16_t(bytes), factors)) |
                     (uint64_t(packed_fp8_pair(uint16_t(bytes >> 16), factors)) << 32);
#else
            const float scale = exponent == 0 ? 0x1p-127f : __uint_as_float(uint32_t(exponent) << 23);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              __nv_fp8_e4m3 f;
              f.__x = uint8_t(bytes >> (j * 8));
              const auto value = __float2bfloat16_rn(__fmul_rn(float(f), scale));
              packed |= uint64_t(__bfloat16_as_ushort(value)) << (j * 16);
            }
#endif
          }
        }
        *reinterpret_cast<uint64_t*>(kv + key * kKvStride + col) = packed;
      }
    }
    __syncthreads();

    {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> scores;
      wmma::fill_fragment(scores, 0.0f);
      for (int k = 0; k < kHeadDim; k += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> b;
        wmma::load_matrix_sync(a, query + base + k, kHeadDim);
        wmma::load_matrix_sync(b, kv + warp * 16 * kKvStride + k, kKvStride);
        wmma::mma_sync(scores, a, b, scores);
      }
      wmma::store_matrix_sync(scratch + warp * 16, scores, 64, wmma::mem_row_major);
    }
    __syncthreads();

    bf16 p0a{}, p0b{}, p1a{}, p1b{}, p2a{}, p2b{}, p3a{}, p3b{};
    for (int h = warp; h < 16; h += 4) {
      const float a = refs[lane] != UINT64_MAX ? scratch[h * 64 + lane] * kScale : -CUDART_INF_F;
      const float b = refs[lane + 32] != UINT64_MAX ? scratch[h * 64 + lane + 32] * kScale : -CUDART_INF_F;
      const float prev = maximum[h], m = fmaxf(prev, warp_max(fmaxf(a, b)));
      const float scale = expf(prev - m), pa = expf(a - m), pb = expf(b - m);
      const float total = warp_sum(pa + pb);
      const auto a16 = __float2bfloat16_rn(pa), b16 = __float2bfloat16_rn(pb);
      if (h < 4) {
        p0a = a16;
        p0b = b16;
      } else if (h < 8) {
        p1a = a16;
        p1b = b16;
      } else if (h < 12) {
        p2a = a16;
        p2b = b16;
      } else {
        p3a = a16;
        p3b = b16;
      }
      if (lane == 0) {
        maximum[h] = m;
        rescale[h] = scale;
        sum[h] = fmaf(sum[h], scale, total);
      }
    }
    __syncthreads();

    if (!empty) {
      for (int i = tid; i < 16 * 16; i += 128) scratch[i] = rescale[i / 16];
      __syncthreads();
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> factors;
      wmma::load_matrix_sync(factors, scratch, 16, wmma::mem_row_major);
#pragma unroll
      for (int t = 0; t < 8; ++t) {
#pragma unroll
        for (int i = 0; i < acc[t].num_elements; ++i)
          acc[t].x[i] = __fmul_rn(acc[t].x[i], factors.x[i]);
      }
      __syncthreads();
    }
    empty = false;

    probability[warp * kProbabilityStride + lane] = p0a;
    probability[warp * kProbabilityStride + lane + 32] = p0b;
    probability[(warp + 4) * kProbabilityStride + lane] = p1a;
    probability[(warp + 4) * kProbabilityStride + lane + 32] = p1b;
    probability[(warp + 8) * kProbabilityStride + lane] = p2a;
    probability[(warp + 8) * kProbabilityStride + lane + 32] = p2b;
    probability[(warp + 12) * kProbabilityStride + lane] = p3a;
    probability[(warp + 12) * kProbabilityStride + lane + 32] = p3b;
    __syncthreads();

    for (int k = 0; k < 64; k += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> p;
      wmma::load_matrix_sync(p, probability + k, kProbabilityStride);
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> v;
        wmma::load_matrix_sync(v, kv + k * kKvStride + warp * 128 + t * 16, kKvStride);
        wmma::mma_sync(acc[t], p, v, acc[t]);
      }
    }
    __syncthreads();
  }

  if constexpr (Split) {
#pragma unroll
    for (int t = 0; t < 8; ++t)
      wmma::store_matrix_sync(scratch + warp * 128 + t * 16, acc[t], kOutputStride, wmma::mem_row_major);
    if (tid < 16) {
      partial[partial_base + tid * 514 + 512] = empty ? -1e30f : maximum[tid];
      partial[partial_base + tid * 514 + 513] = empty ? 0.0f : sum[tid];
    }
    __syncthreads();
    for (int i = tid; i < 16 * kHeadDim; i += 128)
      partial[partial_base + (i / kHeadDim) * 514 + (i % kHeadDim)] =
          empty ? 0.0f : scratch[(i / kHeadDim) * kOutputStride + (i % kHeadDim)];
  } else {
    if (alive_out && group == 0 && tid == 0) alive_out[row] = empty ? 0u : 1u;
    if (empty) {
      for (int i = tid; i < 16 * kHeadDim; i += 128) output[base + i] = __float2bfloat16(0.0f);
      return;
    }
#pragma unroll
    for (int t = 0; t < 8; ++t)
      wmma::store_matrix_sync(scratch + warp * 128 + t * 16, acc[t], kOutputStride, wmma::mem_row_major);
    if (tid < 16) sum[tid] += expf(sink[group * 16 + tid] - maximum[tid]);
    __syncthreads();
    for (int i = tid; i < 16 * kHeadDim; i += 128)
      output[base + i] = __float2bfloat16_rn(scratch[(i / kHeadDim) * kOutputStride + (i % kHeadDim)] / sum[i / kHeadDim]);
  }
}

__global__ void dfly2_merge(const float* partial, const float* sink, bf16* output,
                            uint8_t* alive_out, int parts) {
  const int row = blockIdx.x, head = blockIdx.y, col = threadIdx.x;
  const uint64_t b = (uint64_t(row) * parts * kHeads + head) * 514;
  const uint64_t dest = (uint64_t(row) * kHeads + head) * kHeadDim;
  float maximum = -1e30f;
  bool any = false;
  for (int p = 0; p < parts; ++p) {
    const uint64_t i = b + uint64_t(p) * kHeads * 514;
    if (partial[i + 513] > 0.0f) {
      any = true;
      maximum = fmaxf(maximum, partial[i + 512]);
    }
  }
  if (alive_out && head == 0 && col == 0) alive_out[row] = any ? 1u : 0u;
  if (!any) {
    output[dest + col] = __float2bfloat16(0.0f);
    output[dest + col + 256] = __float2bfloat16(0.0f);
    return;
  }
  float sum = 0.0f, a = 0.0f, c = 0.0f;
  for (int p = 0; p < parts; ++p) {
    const uint64_t i = b + uint64_t(p) * kHeads * 514;
    const float ps = partial[i + 513];
    if (ps <= 0.0f) continue;
    const float factor = expf(partial[i + 512] - maximum);
    sum = fmaf(ps, factor, sum);
    a = fmaf(partial[i + col], factor, a);
    c = fmaf(partial[i + col + 256], factor, c);
  }
  sum += expf(sink[head] - maximum);
  output[dest + col] = __float2bfloat16_rn(a / sum);
  output[dest + col + 256] = __float2bfloat16_rn(c / sum);
}

__global__ void embed_block_kernel(const bf16* table, const uint32_t* anchor_ids,
                                   bf16* block_out, bf16* anchor_out,
                                   int block_size, uint32_t mask_token_id) {
  const int row = blockIdx.x;
  const int req = row / block_size;
  const int slot = row % block_size;
  const uint32_t token = slot == 0 ? anchor_ids[req] : mask_token_id;
  const bool valid = token < uint32_t(kVocabSize);
  const bf16* src = valid ? table + uint64_t(token) * kHiddenSize : nullptr;
  bf16* dst = block_out + uint64_t(row) * kHiddenSize;
  bf16* adst = (slot == 0 && anchor_out) ? (anchor_out + uint64_t(req) * kHiddenSize) : nullptr;
  for (int i = threadIdx.x * 8; i < kHiddenSize; i += blockDim.x * 8) {
    uint4 v = valid ? *reinterpret_cast<const uint4*>(src + i) : make_uint4(0, 0, 0, 0);
    *reinterpret_cast<uint4*>(dst + i) = v;
    if (adst) *reinterpret_cast<uint4*>(adst + i) = v;
  }
}

__global__ void embed_tokens_kernel(const bf16* table, const uint32_t* token_ids, bf16* out) {
  const int req = blockIdx.x;
  const uint32_t token = token_ids[req];
  const bool valid = token < uint32_t(kVocabSize);
  const bf16* src = valid ? table + uint64_t(token) * kHiddenSize : nullptr;
  bf16* dst = out + uint64_t(req) * kHiddenSize;
  for (int i = threadIdx.x * 8; i < kHiddenSize; i += blockDim.x * 8) {
    *reinterpret_cast<uint4*>(dst + i) =
        valid ? *reinterpret_cast<const uint4*>(src + i) : make_uint4(0, 0, 0, 0);
  }
}

__global__ void concat_bf16_kernel(const bf16* a, const bf16* b, bf16* out,
                                   int width_a, int width_b) {
  const uint64_t row = blockIdx.x;
  const int total = width_a + width_b;
  bf16* dst = out + row * total;
  const bf16* sa = a + row * width_a;
  const bf16* sb = b + row * width_b;
  for (int i = threadIdx.x * 8; i < width_a; i += blockDim.x * 8)
    *reinterpret_cast<uint4*>(dst + i) = *reinterpret_cast<const uint4*>(sa + i);
  for (int i = threadIdx.x * 8; i < width_b; i += blockDim.x * 8)
    *reinterpret_cast<uint4*>(dst + width_a + i) = *reinterpret_cast<const uint4*>(sb + i);
}

__global__ void mtp_fuse_and_seed_hc_kernel(
    const bf16* block_x, const bf16* fused, const bf16* gate,
    const uint64_t* positions, bf16* residual, float* pre_mix, int block_size) {
  const int row = blockIdx.x;
  const int req = row / block_size;
  const int slot = row % block_size;
  const bool do_fuse = (slot == 0) && (positions[req] > 0);
  const bf16* src = block_x + uint64_t(row) * kHiddenSize;
  const bf16* fsrc = fused + uint64_t(req) * kHiddenSize;
  bf16* rdst = residual + uint64_t(row) * 4 * kHiddenSize;
  for (int col = threadIdx.x; col < kHiddenSize; col += blockDim.x) {
    bf16 v = src[col];
    if (do_fuse) {
      const bf16 delta = __float2bfloat16_rn(__bfloat162float(gate[col]) * __bfloat162float(fsrc[col]));
      v = __float2bfloat16_rn(__bfloat162float(v) + __bfloat162float(delta));
    }
    rdst[col] = v;
    rdst[kHiddenSize + col] = v;
    rdst[2 * kHiddenSize + col] = v;
    rdst[3 * kHiddenSize + col] = v;
  }
  if (threadIdx.x < 4) pre_mix[uint64_t(row) * 4 + threadIdx.x] = 0.25f;
}

__global__ void swiglu_kernel(const bf16* gate, const bf16* up, bf16* out,
                              uint64_t total, float limit) {
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += uint64_t(gridDim.x) * blockDim.x) {
    float g = __bfloat162float(gate[i]);
    float u = __bfloat162float(up[i]);
    if (limit > 0.0f) {
      g = fminf(g, limit);
      u = fmaxf(-limit, fminf(u, limit));
    }
    const float sg = __bfloat162float(__float2bfloat16_rn(g / (1.0f + expf(-g))));
    out[i] = __float2bfloat16_rn(sg * u);
  }
}

__global__ void add2_bf16_kernel(const bf16* a, const bf16* b, bf16* out, uint64_t count) {
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += uint64_t(gridDim.x) * blockDim.x) {
    out[i] = __float2bfloat16_rn(__bfloat162float(a[i]) + __bfloat162float(b[i]));
  }
}

__global__ void add3_bf16_kernel(const bf16* a, const bf16* b, const bf16* c, bf16* out, uint64_t count) {
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += uint64_t(gridDim.x) * blockDim.x) {
    const float bc = __bfloat162float(__float2bfloat16_rn(__bfloat162float(b[i]) + __bfloat162float(c[i])));
    out[i] = __float2bfloat16_rn(__bfloat162float(a[i]) + bc);
  }
}

__global__ void slice_slot_kernel(const bf16* block_hidden, bf16* out,
                                  int block_size, int slot, int hidden_size) {
  const int req = blockIdx.x;
  const bf16* src = block_hidden + uint64_t(req * block_size + slot) * hidden_size;
  bf16* dst = out + uint64_t(req) * hidden_size;
  for (int i = threadIdx.x * 8; i < hidden_size; i += blockDim.x * 8)
    *reinterpret_cast<uint4*>(dst + i) = *reinterpret_cast<const uint4*>(src + i);
}

__global__ void scatter_kv_slot_kernel(const bf16* kv_row, bf16* block_keys,
                                       int max_slots, int slot) {
  const int req = blockIdx.x;
  const bf16* src = kv_row + uint64_t(req) * kHeadDim;
  bf16* dst = block_keys + uint64_t(req * max_slots + slot) * kHeadDim;
  const int i = threadIdx.x * 8;
  if (i < kHeadDim)
    *reinterpret_cast<uint4*>(dst + i) = *reinterpret_cast<const uint4*>(src + i);
}

// One block of 256 threads per (req, head); each thread owns dims tid and tid+256.
__global__ void block_gated_add_kernel(
    const bf16* query, const bf16* block_keys, const bf16* ctx_read,
    const uint8_t* ctx_alive, const bf16* gate_logits, const float* rel_bias,
    const float* block_sink, const float* gate_bias, bf16* out,
    int max_slots, int slot) {
  const int req = blockIdx.x, head = blockIdx.y, tid = threadIdx.x;
  const int lane = tid % 32, warp = tid / 32;
  const uint64_t q_base = (uint64_t(req) * kHeads + head) * kHeadDim;
  const float q0 = __bfloat162float(query[q_base + tid]);
  const float q1 = __bfloat162float(query[q_base + tid + 256]);

  __shared__ float warp_partials[8];
  __shared__ float logits[16];

  for (int k = 0; k <= slot; ++k) {
    const uint64_t k_base = uint64_t(req * max_slots + k) * kHeadDim;
    const float k0 = __bfloat162float(block_keys[k_base + tid]);
    const float k1 = __bfloat162float(block_keys[k_base + tid + 256]);
    float dot = fmaf(q0, k0, q1 * k1);
    dot = warp_sum(dot);
    if (lane == 0) warp_partials[warp] = dot;
    __syncthreads();
    if (tid == 0) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < 8; ++w) sum += warp_partials[w];
      logits[k] = sum * kScale + rel_bias[head * 16 + (slot - k)];
    }
    __syncthreads();
  }

  __shared__ float weights[16];
  __shared__ float gate_val;
  if (tid == 0) {
    const float sink = block_sink[head];
    float m = sink;
    for (int k = 0; k <= slot; ++k) m = fmaxf(m, logits[k]);
    float denom = expf(sink - m);
    for (int k = 0; k <= slot; ++k) {
      const float e = expf(logits[k] - m);
      weights[k] = e;
      denom += e;
    }
    const float inv_denom = 1.0f / denom;
    for (int k = 0; k <= slot; ++k) weights[k] *= inv_denom;
    gate_val = tanhf(__bfloat162float(gate_logits[req * kHeads + head]) + gate_bias[head * 16 + slot]);
  }
  __syncthreads();

  float acc0 = 0.0f, acc1 = 0.0f;
  for (int k = 0; k <= slot; ++k) {
    const float w = weights[k];
    const uint64_t k_base = uint64_t(req * max_slots + k) * kHeadDim;
    acc0 = fmaf(w, __bfloat162float(block_keys[k_base + tid]), acc0);
    acc1 = fmaf(w, __bfloat162float(block_keys[k_base + tid + 256]), acc1);
  }
  const bool alive = ctx_alive[req] != 0;
  const float base0 = alive ? __bfloat162float(ctx_read[q_base + tid]) : 0.0f;
  const float base1 = alive ? __bfloat162float(ctx_read[q_base + tid + 256]) : 0.0f;
  out[q_base + tid] = __float2bfloat16_rn(fmaf(acc0, gate_val, base0));
  out[q_base + tid + 256] = __float2bfloat16_rn(fmaf(acc1, gate_val, base1));
}

struct VocabTileStat {
  uint32_t id;
  float max_val;
  float sum_exp;
  uint32_t pad;
};
static_assert(sizeof(VocabTileStat) == 16);

__device__ __forceinline__ void combine_stat(
    uint32_t& id, float& m, float& s, uint32_t oid, float om, float os) {
  if (os == 0.0f && oid == UINT32_MAX) return;
  if (s == 0.0f && id == UINT32_MAX) {
    id = oid;
    m = om;
    s = os;
    return;
  }
  if (om > m || (om == m && oid < id)) {
    s = fmaf(s, expf(m - om), os);
    m = om;
    id = oid;
  } else {
    s = fmaf(os, expf(om - m), s);
  }
}

__global__ void shard_stats_tiles_kernel(const float* logits, VocabTileStat* tiles, int vocab_cols) {
  const int tile = blockIdx.x, row = blockIdx.y, tid = threadIdx.x;
  const int lane = tid % 32, warp = tid / 32;
  const int per = (vocab_cols + kVocabTiles - 1) / kVocabTiles;
  const int first = tile * per;
  const int last = min(vocab_cols, first + per);
  const float* in = logits + uint64_t(row) * vocab_cols;

  uint32_t best_id = UINT32_MAX;
  float m = -CUDART_INF_F;
  float s = 0.0f;
  for (int col = first + tid; col < last; col += 256) {
    const float v = in[col];
    if (s == 0.0f && best_id == UINT32_MAX) {
      best_id = uint32_t(col);
      m = v;
      s = 1.0f;
    } else if (v > m || (v == m && uint32_t(col) < best_id)) {
      s = fmaf(s, expf(m - v), 1.0f);
      m = v;
      best_id = uint32_t(col);
    } else {
      s += expf(v - m);
    }
  }
  for (int o = 16; o; o >>= 1) {
    const uint32_t oid = __shfl_down_sync(0xffffffffu, best_id, o);
    const float om = __shfl_down_sync(0xffffffffu, m, o);
    const float os = __shfl_down_sync(0xffffffffu, s, o);
    combine_stat(best_id, m, s, oid, om, os);
  }
  __shared__ uint32_t s_id[8];
  __shared__ float s_m[8];
  __shared__ float s_s[8];
  if (lane == 0) {
    s_id[warp] = best_id;
    s_m[warp] = m;
    s_s[warp] = s;
  }
  __syncthreads();
  if (warp == 0) {
    best_id = lane < 8 ? s_id[lane] : UINT32_MAX;
    m = lane < 8 ? s_m[lane] : -CUDART_INF_F;
    s = lane < 8 ? s_s[lane] : 0.0f;
    for (int o = 4; o; o >>= 1) {
      const uint32_t oid = __shfl_down_sync(0xffffffffu, best_id, o);
      const float om = __shfl_down_sync(0xffffffffu, m, o);
      const float os = __shfl_down_sync(0xffffffffu, s, o);
      combine_stat(best_id, m, s, oid, om, os);
    }
    if (lane == 0) {
      tiles[uint64_t(row) * kVocabTiles + tile] = VocabTileStat{best_id, m, s, 0};
    }
  }
}

__global__ void shard_stats_reduce_kernel(const VocabTileStat* tiles, uint32_t* out_ids,
                                          float* out_max, float* out_sum_exp) {
  const int row = blockIdx.x, tid = threadIdx.x;
  const int lane = tid % 32, warp = tid / 32;
  const VocabTileStat st = tiles[uint64_t(row) * kVocabTiles + tid];
  uint32_t best_id = st.id;
  float m = st.max_val;
  float s = st.sum_exp;
  for (int o = 16; o; o >>= 1) {
    const uint32_t oid = __shfl_down_sync(0xffffffffu, best_id, o);
    const float om = __shfl_down_sync(0xffffffffu, m, o);
    const float os = __shfl_down_sync(0xffffffffu, s, o);
    combine_stat(best_id, m, s, oid, om, os);
  }
  __shared__ uint32_t s_id[2];
  __shared__ float s_m[2];
  __shared__ float s_s[2];
  if (lane == 0) {
    s_id[warp] = best_id;
    s_m[warp] = m;
    s_s[warp] = s;
  }
  __syncthreads();
  if (tid == 0) {
    best_id = s_id[0];
    m = s_m[0];
    s = s_s[0];
    combine_stat(best_id, m, s, s_id[1], s_m[1], s_s[1]);
    out_ids[row] = best_id;
    out_max[row] = m;
    out_sum_exp[row] = s;
  }
}

__global__ void finalize_slot_kernel(const uint32_t* ids, const float* sum_exp,
                                     uint32_t* curr_tokens, uint32_t* all_tokens,
                                     float* all_probs, int rows, int drafts, int slot_idx) {
  const int r = threadIdx.x;
  if (r >= rows) return;
  const uint32_t tok = ids[r];
  const float s = sum_exp[r];
  const float prob = (isfinite(s) && s > 0.0f) ? (1.0f / s) : 0.0f;
  curr_tokens[r] = tok;
  all_tokens[r * drafts + slot_idx] = tok;
  all_probs[r * drafts + slot_idx] = prob;
}

__global__ void merge_stats_kernel(
    const uint32_t* ids0, const float* max0, const float* sum0,
    const uint32_t* ids1, const float* max1, const float* sum1,
    uint32_t* curr_tokens, uint32_t* all_tokens, float* all_probs,
    int rows, uint32_t split, int drafts, int slot_idx) {
  const int r = threadIdx.x;
  if (r >= rows) return;
  const uint32_t a = ids0[r], b = ids1[r];
  const float m0 = max0[r], s0 = sum0[r];
  const float m1 = max1[r], s1 = sum1[r];
  if (a >= split || b >= uint32_t(kVocabSize) - split ||
      !isfinite(m0) || !isfinite(m1) || !isfinite(s0) || !isfinite(s1)) {
    curr_tokens[r] = UINT32_MAX;
    all_tokens[r * drafts + slot_idx] = UINT32_MAX;
    all_probs[r * drafts + slot_idx] = 0.0f;
    return;
  }
  const bool first = m0 >= m1;
  const float m = first ? m0 : m1;
  const float total = fmaf(s0, expf(m0 - m), s1 * expf(m1 - m));
  const uint32_t tok = first ? a : (split + b);
  const float prob = (isfinite(total) && total > 0.0f) ? (1.0f / total) : 0.0f;
  curr_tokens[r] = tok;
  all_tokens[r * drafts + slot_idx] = tok;
  all_probs[r * drafts + slot_idx] = prob;
}

inline int grid_for(uint64_t total, int threads) {
  const uint64_t blocks = (total + threads - 1) / threads;
  return int(blocks < 8192 ? (blocks ? blocks : 1) : 8192);
}

}  // namespace

extern "C" int32_t cuteafd_v41_dfly2_attention_initialize(void) {
  auto s = cudaFuncSetAttribute(dfly2_attend<false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes);
  if (s != cudaSuccess) return s;
  s = cudaFuncSetAttribute(dfly2_attend<false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes);
  if (s != cudaSuccess) return s;
  s = cudaFuncSetAttribute(dfly2_attend<true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes);
  if (s != cudaSuccess) return s;
  return cudaFuncSetAttribute(dfly2_attend<true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes);
}

extern "C" int32_t cuteafd_v41_dfly2_attention(
    const uint16_t* query, const uint16_t* draft_kv, const float* sink,
    const cuteafd_v41_dfly2_request_t* requests_dev, uint16_t* output,
    uint8_t* alive_out, float* partial, uint64_t scratch_bytes,
    int32_t requests, int32_t block_size, int32_t parts, int32_t source_fp4,
    void* stream) {
  if (requests < 1 || requests > CUTEAFD_V41_MAX_DRAFT_MEMBERS ||
      block_size < 1 || block_size > 16 || parts < 1 || parts > 11 ||
      (source_fp4 != 0 && source_fp4 != 1))
    return cudaErrorInvalidValue;
  const int32_t rows = requests * block_size;
  const uint64_t q_bytes = uint64_t(rows) * kHeads * kHeadDim * 2;
  if (!span(query, q_bytes, 32) || !span(output, q_bytes, 32) ||
      !span(sink, kHeads * 4, 4) ||
      !span(requests_dev, uint64_t(requests) * sizeof(*requests_dev), 8) ||
      !disjoint(query, q_bytes, output, q_bytes) ||
      !disjoint(sink, kHeads * 4, output, q_bytes))
    return cudaErrorInvalidValue;
  if (draft_kv) {
    const uint64_t d_bytes = uint64_t(rows) * kHeadDim * 2;
    if (!span(draft_kv, d_bytes, 8) || !disjoint(draft_kv, d_bytes, output, q_bytes))
      return cudaErrorInvalidValue;
  }
  if (alive_out && (!span(alive_out, uint64_t(rows), 1) || !disjoint(alive_out, uint64_t(rows), output, q_bytes)))
    return cudaErrorInvalidValue;
  auto s = reinterpret_cast<cudaStream_t>(stream);
  if (partial) {
    const uint64_t req_scratch = uint64_t(rows) * parts * kHeads * 514 * sizeof(float);
    if (scratch_bytes < req_scratch || !span(partial, req_scratch, 4) ||
        !disjoint(partial, req_scratch, output, q_bytes))
      return cudaErrorInvalidValue;
    if (source_fp4) {
      dfly2_attend<true, true><<<dim3(rows, 4, parts), 128, kSharedBytes, s>>>(
          reinterpret_cast<const bf16*>(query), reinterpret_cast<const bf16*>(draft_kv),
          sink, requests_dev, reinterpret_cast<bf16*>(output), alive_out, partial, block_size);
    } else {
      dfly2_attend<true, false><<<dim3(rows, 4, parts), 128, kSharedBytes, s>>>(
          reinterpret_cast<const bf16*>(query), reinterpret_cast<const bf16*>(draft_kv),
          sink, requests_dev, reinterpret_cast<bf16*>(output), alive_out, partial, block_size);
    }
    auto err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    dfly2_merge<<<dim3(rows, kHeads), 256, 0, s>>>(
        partial, sink, reinterpret_cast<bf16*>(output), alive_out, parts);
  } else {
    if (source_fp4) {
      dfly2_attend<false, true><<<dim3(rows, 4), 128, kSharedBytes, s>>>(
          reinterpret_cast<const bf16*>(query), reinterpret_cast<const bf16*>(draft_kv),
          sink, requests_dev, reinterpret_cast<bf16*>(output), alive_out, nullptr, block_size);
    } else {
      dfly2_attend<false, false><<<dim3(rows, 4), 128, kSharedBytes, s>>>(
          reinterpret_cast<const bf16*>(query), reinterpret_cast<const bf16*>(draft_kv),
          sink, requests_dev, reinterpret_cast<bf16*>(output), alive_out, nullptr, block_size);
    }
  }
  return cudaGetLastError();
}

extern "C" int32_t cuteafd_v41_dfly2_embed_block(
    const uint16_t* embed_table, const uint32_t* anchor_ids,
    uint16_t* block_out, uint16_t* anchor_out,
    int32_t requests, int32_t block_size, uint32_t mask_token_id, void* stream) {
  if (requests < 1 || requests > CUTEAFD_V41_MAX_DRAFT_MEMBERS ||
      block_size < 1 || block_size > 16 || mask_token_id >= uint32_t(kVocabSize))
    return cudaErrorInvalidValue;
  const uint64_t t_bytes = uint64_t(kVocabSize) * kHiddenSize * 2;
  const uint64_t b_bytes = uint64_t(requests) * block_size * kHiddenSize * 2;
  const uint64_t a_bytes = uint64_t(requests) * kHiddenSize * 2;
  if (!span(embed_table, t_bytes, 16) || !span(anchor_ids, uint64_t(requests) * 4, 4) ||
      !span(block_out, b_bytes, 16) || (anchor_out && !span(anchor_out, a_bytes, 16)))
    return cudaErrorInvalidValue;
  embed_block_kernel<<<requests * block_size, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const bf16*>(embed_table), anchor_ids,
      reinterpret_cast<bf16*>(block_out), reinterpret_cast<bf16*>(anchor_out),
      block_size, mask_token_id);
  return cudaGetLastError();
}

extern "C" int32_t cuteafd_v41_dfly2_embed_tokens(
    const uint16_t* embed_table, const uint32_t* token_ids, uint16_t* out,
    int32_t requests, void* stream) {
  if (requests < 1 || requests > CUTEAFD_V41_MAX_DRAFT_MEMBERS) return cudaErrorInvalidValue;
  if (!span(embed_table, uint64_t(kVocabSize) * kHiddenSize * 2, 16) ||
      !span(token_ids, uint64_t(requests) * 4, 4) ||
      !span(out, uint64_t(requests) * kHiddenSize * 2, 16))
    return cudaErrorInvalidValue;
  embed_tokens_kernel<<<requests, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const bf16*>(embed_table), token_ids, reinterpret_cast<bf16*>(out));
  return cudaGetLastError();
}

extern "C" int32_t cuteafd_v41_dfly2_concat_bf16(
    const uint16_t* a, const uint16_t* b, uint16_t* out,
    int32_t rows, int32_t width_a, int32_t width_b, void* stream) {
  if (rows < 1 || rows > 4096 || width_a < 8 || width_b < 8 ||
      (width_a % 8) != 0 || (width_b % 8) != 0)
    return cudaErrorInvalidValue;
  const uint64_t ba = uint64_t(rows) * width_a * 2;
  const uint64_t bb = uint64_t(rows) * width_b * 2;
  const uint64_t bo = ba + bb;
  if (!span(a, ba, 16) || !span(b, bb, 16) || !span(out, bo, 16) ||
      !disjoint(a, ba, out, bo) || !disjoint(b, bb, out, bo))
    return cudaErrorInvalidValue;
  concat_bf16_kernel<<<rows, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const bf16*>(a), reinterpret_cast<const bf16*>(b),
      reinterpret_cast<bf16*>(out), width_a, width_b);
  return cudaGetLastError();
}

extern "C" int32_t cuteafd_v41_dfly2_mtp_fuse_and_seed_hc(
    const uint16_t* block_x, const uint16_t* fused, const uint16_t* gate,
    const uint64_t* positions, uint16_t* residual, float* pre_mix,
    int32_t requests, int32_t block_size, void* stream) {
  if (requests < 1 || requests > CUTEAFD_V41_MAX_DRAFT_MEMBERS ||
      block_size < 1 || block_size > 16)
    return cudaErrorInvalidValue;
  const int32_t rows = requests * block_size;
  const uint64_t bx = uint64_t(rows) * kHiddenSize * 2;
  const uint64_t f_bytes = uint64_t(requests) * kHiddenSize * 2;
  const uint64_t g_bytes = uint64_t(kHiddenSize) * 2;
  const uint64_t r_bytes = uint64_t(rows) * 4 * kHiddenSize * 2;
  const uint64_t p_bytes = uint64_t(rows) * 4 * 4;
  if (!span(block_x, bx, 2) || !span(fused, f_bytes, 2) || !span(gate, g_bytes, 2) ||
      !span(positions, uint64_t(requests) * 8, 8) || !span(residual, r_bytes, 2) ||
      !span(pre_mix, p_bytes, 4))
    return cudaErrorInvalidValue;
  mtp_fuse_and_seed_hc_kernel<<<rows, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const bf16*>(block_x), reinterpret_cast<const bf16*>(fused),
      reinterpret_cast<const bf16*>(gate), positions,
      reinterpret_cast<bf16*>(residual), pre_mix, block_size);
  return cudaGetLastError();
}

extern "C" int32_t cuteafd_v41_dfly2_swiglu(
    const uint16_t* gate, const uint16_t* up, uint16_t* out,
    int32_t rows, int32_t inter, float limit, void* stream) {
  if (rows < 1 || rows > 4096 || inter < 1) return cudaErrorInvalidValue;
  const uint64_t total = uint64_t(rows) * inter;
  const uint64_t bytes = total * 2;
  if (!span(gate, bytes, 2) || !span(up, bytes, 2) || !span(out, bytes, 2))
    return cudaErrorInvalidValue;
  swiglu_kernel<<<grid_for(total, 256), 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const bf16*>(gate), reinterpret_cast<const bf16*>(up),
      reinterpret_cast<bf16*>(out), total, limit);
  return cudaGetLastError();
}

extern "C" int32_t cuteafd_v41_dfly2_add2_bf16(
    const uint16_t* a, const uint16_t* b, uint16_t* out,
    uint64_t count, void* stream) {
  if (count == 0) return cudaErrorInvalidValue;
  const uint64_t bytes = count * 2;
  if (!span(a, bytes, 2) || !span(b, bytes, 2) || !span(out, bytes, 2))
    return cudaErrorInvalidValue;
  add2_bf16_kernel<<<grid_for(count, 256), 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const bf16*>(a), reinterpret_cast<const bf16*>(b),
      reinterpret_cast<bf16*>(out), count);
  return cudaGetLastError();
}

extern "C" int32_t cuteafd_v41_dfly2_add3_bf16(
    const uint16_t* a, const uint16_t* b, const uint16_t* c, uint16_t* out,
    uint64_t count, void* stream) {
  if (count == 0) return cudaErrorInvalidValue;
  const uint64_t bytes = count * 2;
  if (!span(a, bytes, 2) || !span(b, bytes, 2) || !span(c, bytes, 2) || !span(out, bytes, 2))
    return cudaErrorInvalidValue;
  add3_bf16_kernel<<<grid_for(count, 256), 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const bf16*>(a), reinterpret_cast<const bf16*>(b),
      reinterpret_cast<const bf16*>(c), reinterpret_cast<bf16*>(out), count);
  return cudaGetLastError();
}

extern "C" int32_t cuteafd_v41_dfly2_slice_slot(
    const uint16_t* block_hidden, uint16_t* out,
    int32_t requests, int32_t block_size, int32_t slot, int32_t hidden_size,
    void* stream) {
  if (requests < 1 || requests > CUTEAFD_V41_MAX_DRAFT_MEMBERS ||
      block_size < 1 || block_size > 16 || slot < 0 || slot >= block_size ||
      hidden_size < 8 || (hidden_size % 8) != 0)
    return cudaErrorInvalidValue;
  const uint64_t ib = uint64_t(requests) * block_size * hidden_size * 2;
  const uint64_t ob = uint64_t(requests) * hidden_size * 2;
  if (!span(block_hidden, ib, 16) || !span(out, ob, 16) || !disjoint(block_hidden, ib, out, ob))
    return cudaErrorInvalidValue;
  slice_slot_kernel<<<requests, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const bf16*>(block_hidden), reinterpret_cast<bf16*>(out),
      block_size, slot, hidden_size);
  return cudaGetLastError();
}

extern "C" int32_t cuteafd_v41_dfly2_scatter_kv_slot(
    const uint16_t* kv_row, uint16_t* block_keys,
    int32_t requests, int32_t max_slots, int32_t slot, void* stream) {
  if (requests < 1 || requests > CUTEAFD_V41_MAX_DRAFT_MEMBERS ||
      max_slots < 1 || max_slots > 16 || slot < 0 || slot >= max_slots)
    return cudaErrorInvalidValue;
  const uint64_t ib = uint64_t(requests) * kHeadDim * 2;
  const uint64_t ob = uint64_t(requests) * max_slots * kHeadDim * 2;
  if (!span(kv_row, ib, 16) || !span(block_keys, ob, 16) || !disjoint(kv_row, ib, block_keys, ob))
    return cudaErrorInvalidValue;
  scatter_kv_slot_kernel<<<requests, 64, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const bf16*>(kv_row), reinterpret_cast<bf16*>(block_keys),
      max_slots, slot);
  return cudaGetLastError();
}

extern "C" int32_t cuteafd_v41_dfly2_block_gated_add(
    const uint16_t* query, const uint16_t* block_keys,
    const uint16_t* ctx_read, const uint8_t* ctx_alive,
    const uint16_t* gate_logits, const float* rel_bias,
    const float* block_sink, const float* gate_bias,
    uint16_t* out, int32_t requests, int32_t max_slots, int32_t slot,
    void* stream) {
  if (requests < 1 || requests > CUTEAFD_V41_MAX_DRAFT_MEMBERS ||
      max_slots < 2 || max_slots > 16 || slot < 1 || slot >= max_slots)
    return cudaErrorInvalidValue;
  const uint64_t q_bytes = uint64_t(requests) * kHeads * kHeadDim * 2;
  const uint64_t k_bytes = uint64_t(requests) * max_slots * kHeadDim * 2;
  const uint64_t gl_bytes = uint64_t(requests) * kHeads * 2;
  const uint64_t rb_bytes = uint64_t(kHeads) * 16 * 4;
  if (!span(query, q_bytes, 2) || !span(block_keys, k_bytes, 2) ||
      !span(ctx_read, q_bytes, 2) || !span(ctx_alive, uint64_t(requests), 1) ||
      !span(gate_logits, gl_bytes, 2) || !span(rel_bias, rb_bytes, 4) ||
      !span(block_sink, kHeads * 4, 4) || !span(gate_bias, rb_bytes, 4) ||
      !span(out, q_bytes, 2))
    return cudaErrorInvalidValue;
  block_gated_add_kernel<<<dim3(requests, kHeads), 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const bf16*>(query), reinterpret_cast<const bf16*>(block_keys),
      reinterpret_cast<const bf16*>(ctx_read), ctx_alive,
      reinterpret_cast<const bf16*>(gate_logits), rel_bias, block_sink, gate_bias,
      reinterpret_cast<bf16*>(out), max_slots, slot);
  return cudaGetLastError();
}

extern "C" int32_t cuteafd_v41_dfly2_shard_stats_f32(
    const float* logits, void* scratch, uint32_t* out_ids,
    float* out_max, float* out_sum_exp,
    int32_t rows, int32_t vocab_cols, void* stream) {
  if (rows < 1 || rows > CUTEAFD_V41_MAX_DRAFT_MEMBERS ||
      vocab_cols < 1 || vocab_cols > kVocabSize)
    return cudaErrorInvalidValue;
  const uint64_t l_bytes = uint64_t(rows) * vocab_cols * 4;
  const uint64_t s_bytes = uint64_t(rows) * kVocabTiles * sizeof(VocabTileStat);
  const uint64_t r_bytes = uint64_t(rows) * 4;
  if (!span(logits, l_bytes, 4) || !span(scratch, s_bytes, 16) ||
      !span(out_ids, r_bytes, 4) || !span(out_max, r_bytes, 4) ||
      !span(out_sum_exp, r_bytes, 4))
    return cudaErrorInvalidValue;
  auto* tiles = reinterpret_cast<VocabTileStat*>(scratch);
  auto s = reinterpret_cast<cudaStream_t>(stream);
  shard_stats_tiles_kernel<<<dim3(kVocabTiles, rows), 256, 0, s>>>(logits, tiles, vocab_cols);
  auto err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  shard_stats_reduce_kernel<<<rows, kVocabTiles, 0, s>>>(tiles, out_ids, out_max, out_sum_exp);
  return cudaGetLastError();
}

extern "C" int32_t cuteafd_v41_dfly2_finalize_slot_f32(
    const uint32_t* ids, const float* sum_exp,
    uint32_t* curr_tokens, uint32_t* all_tokens, float* all_probs,
    int32_t rows, int32_t drafts, int32_t slot_idx, void* stream) {
  if (rows < 1 || rows > CUTEAFD_V41_MAX_DRAFT_MEMBERS ||
      drafts < 1 || drafts > 15 || slot_idx < 0 || slot_idx >= drafts)
    return cudaErrorInvalidValue;
  const uint64_t r_bytes = uint64_t(rows) * 4;
  const uint64_t a_bytes = uint64_t(rows) * drafts * 4;
  if (!span(ids, r_bytes, 4) || !span(sum_exp, r_bytes, 4) ||
      !span(curr_tokens, r_bytes, 4) || !span(all_tokens, a_bytes, 4) ||
      !span(all_probs, a_bytes, 4))
    return cudaErrorInvalidValue;
  finalize_slot_kernel<<<1, CUTEAFD_V41_MAX_DRAFT_MEMBERS, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      ids, sum_exp, curr_tokens, all_tokens, all_probs, rows, drafts, slot_idx);
  return cudaGetLastError();
}

extern "C" int32_t cuteafd_v41_dfly2_merge_stats_f32(
    const uint32_t* ids0, const float* max0, const float* sum0,
    const uint32_t* ids1, const float* max1, const float* sum1,
    uint32_t* curr_tokens, uint32_t* all_tokens, float* all_probs,
    int32_t rows, int32_t split, int32_t drafts, int32_t slot_idx, void* stream) {
  if (rows < 1 || rows > CUTEAFD_V41_MAX_DRAFT_MEMBERS ||
      split < 1 || split >= kVocabSize ||
      drafts < 1 || drafts > 15 || slot_idx < 0 || slot_idx >= drafts)
    return cudaErrorInvalidValue;
  const uint64_t r_bytes = uint64_t(rows) * 4;
  const uint64_t a_bytes = uint64_t(rows) * drafts * 4;
  if (!span(ids0, r_bytes, 4) || !span(max0, r_bytes, 4) || !span(sum0, r_bytes, 4) ||
      !span(ids1, r_bytes, 4) || !span(max1, r_bytes, 4) || !span(sum1, r_bytes, 4) ||
      !span(curr_tokens, r_bytes, 4) || !span(all_tokens, a_bytes, 4) ||
      !span(all_probs, a_bytes, 4))
    return cudaErrorInvalidValue;
  merge_stats_kernel<<<1, CUTEAFD_V41_MAX_DRAFT_MEMBERS, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      ids0, max0, sum0, ids1, max1, sum1, curr_tokens, all_tokens, all_probs,
      rows, uint32_t(split), drafts, slot_idx);
  return cudaGetLastError();
}

#include "blas_owner.h"
#include <cublas_v2.h>
#include <new>

namespace {
constexpr uint64_t kDfly2BlasWorkspace = 32ull * 1024ull * 1024ull;

struct Dfly2BlasHandle {
  int device = -1;
  std::shared_ptr<cuteafd::BlasOwner> owner;
  void* workspace = nullptr;
  uint64_t bytes = 0;
};

int32_t dfly2_blas_status(cublasStatus_t status) {
  return status == CUBLAS_STATUS_SUCCESS ? 0 : 10000 + static_cast<int32_t>(status);
}
}  // namespace

extern "C" int32_t cuteafd_v41_dfly2_blas_create(void* workspace, uint64_t bytes, void** output) {
  if (!output) return cudaErrorInvalidValue;
  *output = nullptr;
  if (bytes < kDfly2BlasWorkspace || !span(workspace, kDfly2BlasWorkspace, 256))
    return cudaErrorInvalidValue;
  auto* handle = new (std::nothrow) Dfly2BlasHandle{};
  if (!handle) return cudaErrorMemoryAllocation;
  auto status = cudaGetDevice(&handle->device);
  if (status != cudaSuccess) { delete handle; return status; }
  const auto created = cuteafd::borrow_blas(false, handle->owner, "v41/blas/dfly2");
  if (created != CUBLAS_STATUS_SUCCESS) { delete handle; return dfly2_blas_status(created); }
  handle->workspace = workspace;
  handle->bytes = bytes;
  *output = handle;
  return 0;
}

extern "C" int32_t cuteafd_v41_dfly2_blas_destroy(void* opaque) {
  if (!opaque) return cudaErrorInvalidValue;
  delete static_cast<Dfly2BlasHandle*>(opaque);
  return 0;
}

extern "C" int32_t cuteafd_v41_dfly2_gemm_bf16(
    void* opaque, const uint16_t* x, const uint16_t* w, uint16_t* out,
    int32_t rows, int32_t k, int32_t n, void* stream) {
  if (!opaque || rows < 1 || k < 1 || n < 1) return cudaErrorInvalidValue;
  auto* handle = static_cast<Dfly2BlasHandle*>(opaque);
  int device = -1;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  if (device != handle->device) return cudaErrorInvalidDevice;
  const uint64_t bx = uint64_t(rows) * k * 2;
  const uint64_t bw = uint64_t(n) * k * 2;
  const uint64_t bo = uint64_t(rows) * n * 2;
  if (!span(x, bx, 2) || !span(w, bw, 2) || !span(out, bo, 2) ||
      !disjoint(out, bo, x, bx) || !disjoint(out, bo, w, bw))
    return cudaErrorInvalidValue;
  const auto bound = cuteafd::bind_blas(
      handle->owner, reinterpret_cast<cudaStream_t>(stream), handle->workspace, handle->bytes);
  if (bound != CUBLAS_STATUS_SUCCESS) return dfly2_blas_status(bound);
  const float alpha = 1.0f, beta = 0.0f;
  return dfly2_blas_status(cublasGemmEx(
      handle->owner->gemm, CUBLAS_OP_T, CUBLAS_OP_N,
      n, rows, k, &alpha,
      w, CUDA_R_16BF, k,
      x, CUDA_R_16BF, k,
      &beta,
      out, CUDA_R_16BF, n,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

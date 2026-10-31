#include "../../../shared/cuda/draft_attention.cuh"
#include "../../../shared/cuda/draft_kv.cuh"
// DFlash2 drafters (incoai/GLM-5.3-DFlash2, incoai/GLM-5.3-Flash-DFlash2): the
// non-GEMM pieces of the context update, the block body and the candidate
// selector.
// GEMMs run through cuBLAS (cuteafd_cuda_linear_bf16_cublas_async); the
// vocabulary head is the engine's.
//
// Every elementwise statement keeps upstream's BF16 boundaries (Qwen3RMSNorm
// rounds the unit-RMS value before the weight; the dynamic convolution rounds
// each of its four accumulations; RoPE rounds each product and the sum; the
// selector's edge and score are BF16). glmrt measured that reassociating these
// in FP32 compounds through the six layers and destroys greedy acceptance.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

namespace {

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float f(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float r(float v) { return __bfloat162float(__float2bfloat16(v)); }

template <int THREADS>
__device__ float block_sum(float v, float* shared) {
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  if (threadIdx.x % 32 == 0) shared[threadIdx.x / 32] = v;
  __syncthreads();
  float total = 0;
  for (int i = 0; i < THREADS / 32; ++i) total += shared[i];
  __syncthreads();
  return total;
}

constexpr int kNormThreads = 512;
constexpr int kMaxHidden = 8192;

// out = bf16(bf16(x * rsqrt(mean(x^2) + eps)) * w), one block per row.
__global__ void rmsnorm_kernel(const bf16* x, const bf16* w, bf16* out, int width, float eps) {
  __shared__ float partial[kNormThreads / 32];
  const uint64_t row = blockIdx.x;
  float sum = 0;
  for (int i = threadIdx.x; i < width; i += kNormThreads) {
    const float v = f(x[row * width + i]);
    sum += v * v;
  }
  const float inv = rsqrtf(block_sum<kNormThreads>(sum, partial) / width + eps);
  for (int i = threadIdx.x; i < width; i += kNormThreads)
    out[row * width + i] = __float2bfloat16(r(f(x[row * width + i]) * inv) * f(w[i]));
}

// One side of the two-tap grouped dynamic convolution of block rows.
__device__ __forceinline__ float conv_value(const bf16* source, const bf16* dynamic, const bf16* base, uint64_t row,
                                            int column, int block_rows, int hidden, int group, int side) {
  const int groups = hidden / group, g = column / group;
  const bf16* dyn = dynamic + row * 4 * groups + side * 2 * groups + g;
  const float current = f(source[row * hidden + column]);
  float v = r(current * f(base[(2 * side) * hidden + column]));
  v = r(v + current * f(dyn[0]));
  if (row % block_rows > 0) {
    const float previous = f(source[(row - 1) * hidden + column]);
    v = r(v + r(previous * f(base[(2 * side + 1) * hidden + column])));
    v = r(v + previous * f(dyn[groups]));
  }
  return v;
}

__global__ void conv_kernel(const bf16* source, const bf16* dynamic, const bf16* base, bf16* out, uint64_t total,
                            int block_rows, int hidden, int group) {
  for (uint64_t i = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x; i < total; i += uint64_t(gridDim.x) * blockDim.x)
    out[i] = __float2bfloat16(conv_value(source, dynamic, base, i / hidden, int(i % hidden), block_rows, hidden,
                                         group, 0));
}

// residual_out = bf16(residual + conv1(source)); normalized = RMSNorm(residual_out) * w.
__global__ void conv_residual_norm_kernel(const bf16* source, const bf16* dynamic, const bf16* base,
                                          const bf16* residual, const bf16* w, bf16* residual_out, bf16* normalized,
                                          int block_rows, int hidden, int group, float eps) {
  __shared__ float partial[kNormThreads / 32];
  __shared__ float values[kMaxHidden];
  const uint64_t row = blockIdx.x;
  float sum = 0;
  for (int i = threadIdx.x; i < hidden; i += kNormThreads) {
    const float conv = conv_value(source, dynamic, base, row, i, block_rows, hidden, group, 1);
    const float v = r(f(residual[row * hidden + i]) + conv);
    values[i] = v;
    sum += v * v;
  }
  const float inv = rsqrtf(block_sum<kNormThreads>(sum, partial) / hidden + eps);
  for (int i = threadIdx.x; i < hidden; i += kNormThreads) {
    residual_out[row * hidden + i] = __float2bfloat16(values[i]);
    normalized[row * hidden + i] = __float2bfloat16(r(values[i] * inv) * f(w[i]));
  }
}

// One warp per (row, head) of [q heads | k heads]: per-head RMSNorm, then
// rotate_half RoPE at the row's position; k and v rows go to `slots[row]`
// (skipped when negative) of the [slot, kv_heads, head_dim] caches.
constexpr int kHeadDim = 128;

template <int dim, bool fp8>
__device__ __forceinline__ void qk_rope_body(const bf16* qkv, const bf16* q_norm, const bf16* k_norm, const int64_t* positions,
                               const int32_t* slots, bf16* q_out, bf16* k_out, bf16* v_out, int rows, int heads,
                               int kv_heads, float theta, float eps) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  const int per_row = heads + kv_heads;
  if (warp >= rows * per_row) return;
  const int row = warp / per_row, head = warp % per_row;
  const bool is_q = head < heads;
  const int slot = slots != nullptr ? slots[row] : row;
  if (!is_q && slot < 0) return;
  const uint64_t width = uint64_t(heads + 2 * kv_heads) * dim;
  const bf16* in = qkv + row * width + uint64_t(head) * dim;
  const bf16* norm = is_q ? q_norm : k_norm;
  // Each lane holds dim / 32 elements; RoPE pairs the two head halves.
  float x[dim / 32], sum = 0;
#pragma unroll
  for (int j = 0; j < dim / 32; ++j) {
    x[j] = f(in[lane + 32 * j]);
    sum += x[j] * x[j];
  }
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  const float inv = rsqrtf(sum / dim + eps);
#pragma unroll
  for (int j = 0; j < dim / 32; ++j) x[j] = r(r(x[j] * inv) * f(norm[lane + 32 * j]));
  const float position = float(positions[row]);
  float y[dim / 32];
#pragma unroll
  for (int j = 0; j < dim / 32; ++j) {
    const int e = lane + 32 * j, i = e % (dim / 2);
    const float inv_freq = 1.0f / powf(theta, float(2 * i) / dim);
    const float angle = position * inv_freq;
    const float c = r(cosf(angle)), s = r(sinf(angle));
    const float pair = x[(j + dim / 64) % (dim / 32)];
    y[j] = j < dim / 64 ? r(r(x[j] * c) - r(pair * s)) : r(r(x[j] * c) + r(pair * s));
  }
  if (is_q) {
    bf16* out = q_out + (uint64_t(row) * heads + head) * dim;
#pragma unroll
    for (int j = 0; j < dim / 32; ++j) out[lane + 32 * j] = __float2bfloat16(y[j]);
    return;
  }
  const int kv = head - heads;
  bf16* k = k_out + (uint64_t(slot) * kv_heads + kv) * dim;
  bf16* v = v_out + (uint64_t(slot) * kv_heads + kv) * dim;
  const bf16* v_in = qkv + row * width + uint64_t(heads + kv_heads + kv) * dim;
  if constexpr (fp8) {
    float values[dim / 32];
#pragma unroll
    for (int j = 0; j < dim / 32; ++j) values[j] = f(v_in[lane + 32 * j]);
    cuteafd::draft_kv::store<dim>(k_out, slot, kv, kv_heads, y);
    cuteafd::draft_kv::store<dim>(v_out, slot, kv, kv_heads, values);
  } else {
#pragma unroll
  for (int j = 0; j < dim / 32; ++j) {
    k[lane + 32 * j] = __float2bfloat16(y[j]);
    v[lane + 32 * j] = v_in[lane + 32 * j];
  }
  }
}

template <bool fp8>
__global__ void qk_rope_kernel(const bf16* qkv, const bf16* q_norm, const bf16* k_norm, const int64_t* positions,
                               const int32_t* slots, bf16* q_out, bf16* k_out, bf16* v_out, int rows, int heads,
                               int kv_heads, float theta, float eps) {
  qk_rope_body<128, fp8>(qkv, q_norm, k_norm, positions, slots, q_out, k_out, v_out,
                           rows, heads, kv_heads, theta, eps);
}

template <bool fp8>
__global__ void qk_rope_d64_kernel(const bf16* qkv, const bf16* q_norm, const bf16* k_norm, const int64_t* positions,
                               const int32_t* slots, bf16* q_out, bf16* k_out, bf16* v_out, int rows, int heads,
                               int kv_heads, float theta, float eps) {
  qk_rope_body<64, fp8>(qkv, q_norm, k_norm, positions, slots, q_out, k_out, v_out,
                           rows, heads, kv_heads, theta, eps);
}

__global__ void silu_mul_kernel(const bf16* gate_up, bf16* out, uint64_t rows, int inter) {
  const uint64_t total = rows * inter;
  for (uint64_t i = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x; i < total; i += uint64_t(gridDim.x) * blockDim.x) {
    const uint64_t row = i / inter;
    const int c = int(i % inter);
    const float g = f(gate_up[row * 2 * inter + c]), u = f(gate_up[row * 2 * inter + inter + c]);
    out[i] = __float2bfloat16(r(g / (1.0f + expf(-g))) * u);
  }
}

// Top-16 of each drafted row's BF16-rounded logits (largest first, lower
// index on ties): 64 chunk-local top-16s, then a merge.
constexpr int kTopK = 16;
constexpr int kTopChunks = 64;
constexpr int kTopThreads = 256;

__device__ __forceinline__ bool better(float v, int i, float bv, int bi) { return v > bv || (v == bv && i < bi); }

__device__ void block_argmax(float& v, int& i) {
  for (int o = 16; o; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, v, o);
    const int oi = __shfl_xor_sync(0xffffffffu, i, o);
    if (better(ov, oi, v, i)) {
      v = ov;
      i = oi;
    }
  }
  __shared__ float wv[kTopThreads / 32];
  __shared__ int wi[kTopThreads / 32];
  if (threadIdx.x % 32 == 0) {
    wv[threadIdx.x / 32] = v;
    wi[threadIdx.x / 32] = i;
  }
  __syncthreads();
  v = wv[0];
  i = wi[0];
  for (int w = 1; w < kTopThreads / 32; ++w)
    if (better(wv[w], wi[w], v, i)) {
      v = wv[w];
      i = wi[w];
    }
  __syncthreads();
}

// Drafted row n is block row 1 + n % drafts of sequence n / drafts.
__global__ void topk_partial_kernel(const float* logits, float* values, int* indices, int vocab, int block_rows,
                                    int drafts) {
  const int chunk = blockIdx.x, n = blockIdx.y;
  const uint64_t row = uint64_t(n / drafts) * block_rows + 1 + n % drafts;
  const int per = (vocab + kTopChunks - 1) / kTopChunks, first = chunk * per, last = min(vocab, first + per);
  constexpr int kLocal = 12;
  float local[kLocal];
  const float* in = logits + row * vocab;
  for (int j = 0; j < kLocal; ++j) {
    const int idx = first + threadIdx.x + j * kTopThreads;
    local[j] = idx < last ? r(in[idx]) : -CUDART_INF_F;
  }
  for (int k = 0; k < kTopK; ++k) {
    float bv = -CUDART_INF_F;
    int bi = INT32_MAX;
    for (int j = 0; j < kLocal; ++j) {
      const int idx = first + threadIdx.x + j * kTopThreads;
      if (idx < last && better(local[j], idx, bv, bi)) {
        bv = local[j];
        bi = idx;
      }
    }
    block_argmax(bv, bi);
    if (threadIdx.x == 0) {
      values[(uint64_t(n) * kTopChunks + chunk) * kTopK + k] = bv;
      indices[(uint64_t(n) * kTopChunks + chunk) * kTopK + k] = bi;
    }
    const int owner = bi - first;
    if (owner >= 0 && owner % kTopThreads == int(threadIdx.x)) local[owner / kTopThreads] = -CUDART_INF_F;
  }
}

__global__ void topk_merge_kernel(const float* values, const int* indices, float* unary, int* candidates) {
  const int n = blockIdx.x;
  constexpr int kLocal = kTopChunks * kTopK / kTopThreads;
  float v[kLocal];
  int ix[kLocal];
  for (int j = 0; j < kLocal; ++j) {
    const uint64_t at = uint64_t(n) * kTopChunks * kTopK + threadIdx.x + j * kTopThreads;
    v[j] = values[at];
    ix[j] = indices[at];
  }
  for (int k = 0; k < kTopK; ++k) {
    float bv = -CUDART_INF_F;
    int bi = INT32_MAX;
    for (int j = 0; j < kLocal; ++j)
      if (ix[j] != INT32_MAX && better(v[j], ix[j], bv, bi)) {
        bv = v[j];
        bi = ix[j];
      }
    block_argmax(bv, bi);
    if (threadIdx.x == 0) {
      unary[n * kTopK + k] = bv;
      candidates[n * kTopK + k] = bi;
    }
    for (int j = 0; j < kLocal; ++j)
      if (ix[j] == bi) ix[j] = INT32_MAX;
  }
}

// One block per sequence walks its drafted rows: score_k = bf16(unary_k +
// bf16(sum_r bf16(pred[prev, r] * proj[r]) * succ[cand_k, r])); the best
// (lowest rank on ties) becomes the next predecessor. Also writes the margin,
// best probability, entropy and rank over the 16 scores per row.
constexpr int kRank = 256;

__global__ void select_kernel(const bf16* pred_cb, const bf16* succ_cb, const bf16* projected, const int* candidates,
                              const float* unary, const uint32_t* anchors, uint32_t* tokens, float* features,
                              int block_rows, int drafts) {
  const int s = blockIdx.x, t = threadIdx.x, lane = t % 32, warp = t / 32;
  __shared__ float conditioned[kRank];
  __shared__ float edges[kTopK];
  __shared__ uint32_t previous;
  if (t == 0) previous = anchors[s];
  __syncthreads();
  for (int i = 0; i < drafts; ++i) {
    const int n = s * drafts + i;
    const bf16* proj = projected + (uint64_t(s) * block_rows + 1 + i) * kRank;
    conditioned[t] = r(f(pred_cb[uint64_t(previous) * kRank + t]) * f(proj[t]));
    __syncthreads();
    // Warp w scores candidates w and w + 8.
    for (int k = warp; k < kTopK; k += kRank / 32) {
      const bf16* succ = succ_cb + uint64_t(candidates[n * kTopK + k]) * kRank;
      float dot = 0;
      for (int j = lane; j < kRank; j += 32) dot = fmaf(conditioned[j], f(succ[j]), dot);
      for (int o = 16; o; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
      if (lane == 0) edges[k] = r(unary[n * kTopK + k] + r(dot));
    }
    __syncthreads();
    if (t == 0) {
      int best = 0;
      for (int k = 1; k < kTopK; ++k)
        if (edges[k] > edges[best]) best = k;
      float runner = -CUDART_INF_F, total = 0, weighted = 0;
      for (int k = 0; k < kTopK; ++k) {
        if (k != best) runner = fmaxf(runner, edges[k]);
        const float shifted = edges[k] - edges[best], mass = expf(shifted);
        total += mass;
        weighted += mass * shifted;
      }
      float* feature = features + uint64_t(n) * 4;
      feature[0] = edges[best] - runner;
      feature[1] = 1.0f / total;
      feature[2] = logf(total) - weighted / total;
      feature[3] = float(best);
      previous = uint32_t(candidates[n * kTopK + best]);
      tokens[n] = previous;
    }
    __syncthreads();
  }
}

// dst[row, offset : offset + width] = src[row, :] (BF16).
__global__ void tap_kernel(const bf16* src, bf16* dst, int width, int stride, int offset) {
  const uint64_t row = blockIdx.y;
  for (int i = (blockIdx.x * blockDim.x + threadIdx.x) * 8; i < width; i += gridDim.x * blockDim.x * 8)
    *reinterpret_cast<uint4*>(dst + row * stride + offset + i) = *reinterpret_cast<const uint4*>(src + row * width + i);
}

// dst[row, offset : offset + width] = bf16(mean over `hc` of src[row, i, :]) (the
// mHC stream contraction; FP32 sum in stream order, then the division).
__global__ void tap_mean_kernel(const bf16* src, bf16* dst, int width, int hc, int stride, int offset) {
  const uint64_t row = blockIdx.y;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < width; i += gridDim.x * blockDim.x) {
    float sum = 0;
    for (int k = 0; k < hc; ++k) sum += f(src[(row * hc + k) * width + i]);
    dst[row * stride + offset + i] = __float2bfloat16(sum / float(hc));
  }
}

inline int grid_for(uint64_t total, int threads) {
  const uint64_t blocks = (total + threads - 1) / threads;
  return int(blocks < 8192 ? (blocks ? blocks : 1) : 8192);
}

}  // namespace

extern "C" int32_t cuteafd_glm_dflash_rmsnorm(const void* x, const void* w, void* out, int32_t rows, int32_t width,
                                              float eps, void* stream) {
  if (rows < 1 || width < 1) return cudaErrorInvalidValue;
  rmsnorm_kernel<<<rows, kNormThreads, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const bf16*>(x), static_cast<const bf16*>(w), static_cast<bf16*>(out), width, eps);
  return cudaGetLastError();
}

// out = conv side 0 of `source` [rows, hidden] with `dynamic` [rows, 4 * hidden / group]
// and `base` [2, 2, hidden]; rows form blocks of `block_rows`.
extern "C" int32_t cuteafd_glm_dflash_conv(const void* source, const void* dynamic, const void* base, void* out,
                                           int32_t rows, int32_t block_rows, int32_t hidden, int32_t group,
                                           void* stream) {
  if (rows < 1 || block_rows < 1 || hidden % group) return cudaErrorInvalidValue;
  const uint64_t total = uint64_t(rows) * hidden;
  conv_kernel<<<grid_for(total, 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const bf16*>(source), static_cast<const bf16*>(dynamic), static_cast<const bf16*>(base),
      static_cast<bf16*>(out), total, block_rows, hidden, group);
  return cudaGetLastError();
}

// residual_out = residual + conv side 1 of `source`; normalized = RMSNorm(residual_out) * w.
// `residual_out` may alias `residual`.
extern "C" int32_t cuteafd_glm_dflash_conv_residual_norm(const void* source, const void* dynamic, const void* base,
                                                         const void* residual, const void* w, void* residual_out,
                                                         void* normalized, int32_t rows, int32_t block_rows,
                                                         int32_t hidden, int32_t group, float eps, void* stream) {
  if (rows < 1 || block_rows < 1 || hidden % group || hidden > kMaxHidden) return cudaErrorInvalidValue;
  conv_residual_norm_kernel<<<rows, kNormThreads, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const bf16*>(source), static_cast<const bf16*>(dynamic), static_cast<const bf16*>(base),
      static_cast<const bf16*>(residual), static_cast<const bf16*>(w), static_cast<bf16*>(residual_out),
      static_cast<bf16*>(normalized), block_rows, hidden, group, eps);
  return cudaGetLastError();
}

// qkv [rows, (heads + 2 kv_heads) * 128]: q heads -> q_out [rows, heads, 128]
// (normed, roped); k (normed, roped) and v -> row slots[row] (or row when
// slots is null; negative slots are skipped) of k_out/v_out [*, kv_heads, 128].
extern "C" int32_t cuteafd_glm_dflash_qk_rope(const void* qkv, const void* q_norm, const void* k_norm,
                                              const void* positions, const void* slots, void* q_out, void* k_out,
                                              void* v_out, int32_t rows, int32_t heads, int32_t kv_heads,
                                              float theta, float eps, void* stream) {
  if (rows < 1 || kv_heads < 1 || heads < 0) return cudaErrorInvalidValue;
  const int warps = rows * (heads + kv_heads);
  qk_rope_kernel<false><<<(warps * 32 + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const bf16*>(qkv), static_cast<const bf16*>(q_norm), static_cast<const bf16*>(k_norm),
      static_cast<const int64_t*>(positions), static_cast<const int32_t*>(slots), static_cast<bf16*>(q_out),
      static_cast<bf16*>(k_out), static_cast<bf16*>(v_out), rows, heads, kv_heads, theta, eps);
  return cudaGetLastError();
}

// Workspace bytes cuteafd_glm_dflash_attention needs.
extern "C" uint64_t cuteafd_glm_dflash_attention_workspace(int32_t sequences, int32_t kv_heads, int32_t max_keys) {
  // This ABI admits up to 64 queries per KV head, including smaller launches.
  return cuteafd::draft_attention::workspace(sequences, kv_heads, 2, max_keys, kHeadDim);
}

// out [sequences * block_rows, heads * 128] = softmax(q k^T * scale) v over each
// sequence's context entries (ring slots seq_slots[s] * ring + position % ring
// for positions ctx_ends[s] - ctx_lengths[s] .. ctx_ends[s] - 1; with window > 0
// only those within `window` of the block row's position ctx_ends[s] + row) and
// its block rows (k_block/v_block [rows, kv_heads, 128]). heads / kv_heads *
// block_rows must be at most 64; max_keys bounds ctx_lengths + block_rows.
extern "C" int32_t cuteafd_glm_dflash_attention(const void* q, const void* k_block, const void* v_block,
                                                const void* k_ring, const void* v_ring, const void* seq_slots,
                                                const void* ctx_lengths, const void* ctx_ends, void* out,
                                                void* workspace, int32_t sequences, int32_t block_rows,
                                                int32_t heads, int32_t kv_heads, int32_t ring, int32_t max_keys,
                                                int32_t window, float scale, void* stream) {
  return cuteafd::draft_attention::launch(q, k_block, v_block, k_ring, v_ring, seq_slots,
      ctx_lengths, ctx_ends, nullptr, out, workspace, sequences, block_rows, heads, kv_heads,
      ring, max_keys, window, false, kHeadDim, 2, false, scale, static_cast<cudaStream_t>(stream));
}

extern "C" int32_t cuteafd_glm_dflash_silu_mul(const void* gate_up, void* out, int32_t rows, int32_t inter,
                                               void* stream) {
  if (rows < 1 || inter < 1) return cudaErrorInvalidValue;
  silu_mul_kernel<<<grid_for(uint64_t(rows) * inter, 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const bf16*>(gate_up), static_cast<bf16*>(out), uint64_t(rows), inter);
  return cudaGetLastError();
}

// Workspace bytes cuteafd_glm_dflash_topk needs for `rows` drafted rows.
extern "C" uint64_t cuteafd_glm_dflash_topk_workspace(int32_t rows) {
  return uint64_t(rows) * kTopChunks * kTopK * 8;
}

// unary/candidates [sequences * drafts, 16] from logits [sequences * block_rows, vocab]
// FP32 (block rows 1..drafts of each sequence).
extern "C" int32_t cuteafd_glm_dflash_topk(const void* logits, void* unary, void* candidates, void* workspace,
                                           int32_t sequences, int32_t block_rows, int32_t drafts, int32_t vocab,
                                           void* stream) {
  if (sequences < 1 || drafts < 1 || drafts >= block_rows || (vocab + kTopChunks - 1) / kTopChunks > 12 * kTopThreads)
    return cudaErrorInvalidValue;
  const int rows = sequences * drafts;
  auto* values = static_cast<float*>(workspace);
  auto* indices = reinterpret_cast<int*>(values + uint64_t(rows) * kTopChunks * kTopK);
  auto s = static_cast<cudaStream_t>(stream);
  topk_partial_kernel<<<dim3(kTopChunks, rows), kTopThreads, 0, s>>>(static_cast<const float*>(logits), values,
                                                                     indices, vocab, block_rows, drafts);
  topk_merge_kernel<<<rows, kTopThreads, 0, s>>>(values, indices, static_cast<float*>(unary),
                                                 static_cast<int*>(candidates));
  return cudaGetLastError();
}

// tokens [sequences, drafts] U32 and features [sequences, drafts, 4] F32 from the
// selector codebooks [vocab, 256], projected [sequences * block_rows, 256] and the
// top-16 of each drafted row.
extern "C" int32_t cuteafd_glm_dflash_select(const void* pred_cb, const void* succ_cb, const void* projected,
                                             const void* candidates, const void* unary, const void* anchors,
                                             void* tokens, void* features, int32_t sequences, int32_t block_rows,
                                             int32_t drafts, int32_t rank, void* stream) {
  if (sequences < 1 || rank != kRank || drafts < 1 || drafts >= block_rows) return cudaErrorInvalidValue;
  select_kernel<<<sequences, kRank, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const bf16*>(pred_cb), static_cast<const bf16*>(succ_cb), static_cast<const bf16*>(projected),
      static_cast<const int*>(candidates), static_cast<const float*>(unary), static_cast<const uint32_t*>(anchors),
      static_cast<uint32_t*>(tokens), static_cast<float*>(features), block_rows, drafts);
  return cudaGetLastError();
}

// dst[row, offset : offset + width] = src[row, :] for `rows` BF16 rows (width,
// stride and offset multiples of 8).
extern "C" int32_t cuteafd_glm_dflash_tap(const void* src, void* dst, int32_t rows, int32_t width, int32_t stride,
                                          int32_t offset, void* stream) {
  if (rows < 1 || width % 8 || stride % 8 || offset % 8 || offset + width > stride) return cudaErrorInvalidValue;
  tap_kernel<<<dim3((width / 8 + 255) / 256, rows), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const bf16*>(src), static_cast<bf16*>(dst), width, stride, offset);
  return cudaGetLastError();
}

// dst[row, offset : offset + width] = bf16(mean_i src[row, i]) for `rows` rows of
// `hc` BF16 streams of `width` (the mHC contraction DFlash2 taps).
extern "C" int32_t cuteafd_glm_dflash_tap_mean(const void* src, void* dst, int32_t rows, int32_t width, int32_t hc,
                                               int32_t stride, int32_t offset, void* stream) {
  if (rows < 1 || hc < 1 || width < 1 || offset + width > stride) return cudaErrorInvalidValue;
  tap_mean_kernel<<<dim3((width + 255) / 256, rows), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const bf16*>(src), static_cast<bf16*>(dst), width, hc, stride, offset);
  return cudaGetLastError();
}


extern "C" int32_t cuteafd_glm_dflash_qk_rope_fp8(const void* qkv, const void* q_norm, const void* k_norm,
                                              const void* positions, const void* slots, void* q_out, void* k_out,
                                              void* v_out, int32_t rows, int32_t heads, int32_t kv_heads,
                                              float theta, float eps, void* stream) {
  if (rows < 1 || kv_heads < 1 || heads < 0) return cudaErrorInvalidValue;
  const int warps = rows * (heads + kv_heads);
  qk_rope_kernel<true><<<(warps * 32 + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const bf16*>(qkv), static_cast<const bf16*>(q_norm), static_cast<const bf16*>(k_norm),
      static_cast<const int64_t*>(positions), static_cast<const int32_t*>(slots), static_cast<bf16*>(q_out),
      static_cast<bf16*>(k_out), static_cast<bf16*>(v_out), rows, heads, kv_heads, theta, eps);
  return cudaGetLastError();
}

extern "C" int32_t cuteafd_glm_dflash_attention_fp8(const void* q, const void* k_block, const void* v_block,
                                                const void* k_ring, const void* v_ring, const void* seq_slots,
                                                const void* ctx_lengths, const void* ctx_ends, void* out,
                                                void* workspace, int32_t sequences, int32_t block_rows,
                                                int32_t heads, int32_t kv_heads, int32_t ring, int32_t max_keys,
                                                int32_t window, float scale, void* stream) {
  return cuteafd::draft_attention::launch(q, k_block, v_block, k_ring, v_ring, seq_slots,
      ctx_lengths, ctx_ends, nullptr, out, workspace, sequences, block_rows, heads, kv_heads,
      ring, max_keys, window, false, kHeadDim, 2, true, scale, static_cast<cudaStream_t>(stream));
}

// ABI-isomorphic d64 variants: norm weights have 64 elements, RoPE pairs the
// two 32-element halves, and the attention arena admits 64 queries per KV head.
extern "C" int32_t cuteafd_glm_dflash_qk_rope_d64(const void* qkv, const void* q_norm, const void* k_norm,
                                              const void* positions, const void* slots, void* q_out, void* k_out,
                                              void* v_out, int32_t rows, int32_t heads, int32_t kv_heads,
                                              float theta, float eps, void* stream) {
  if (rows < 1 || kv_heads < 1 || heads < 0) return cudaErrorInvalidValue;
  const int warps = rows * (heads + kv_heads);
  qk_rope_d64_kernel<false><<<(warps * 32 + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const bf16*>(qkv), static_cast<const bf16*>(q_norm), static_cast<const bf16*>(k_norm),
      static_cast<const int64_t*>(positions), static_cast<const int32_t*>(slots), static_cast<bf16*>(q_out),
      static_cast<bf16*>(k_out), static_cast<bf16*>(v_out), rows, heads, kv_heads, theta, eps);
  return cudaGetLastError();
}

extern "C" int32_t cuteafd_glm_dflash_qk_rope_d64_fp8(const void* qkv, const void* q_norm, const void* k_norm,
                                              const void* positions, const void* slots, void* q_out, void* k_out,
                                              void* v_out, int32_t rows, int32_t heads, int32_t kv_heads,
                                              float theta, float eps, void* stream) {
  if (rows < 1 || kv_heads < 1 || heads < 0) return cudaErrorInvalidValue;
  const int warps = rows * (heads + kv_heads);
  qk_rope_d64_kernel<true><<<(warps * 32 + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const bf16*>(qkv), static_cast<const bf16*>(q_norm), static_cast<const bf16*>(k_norm),
      static_cast<const int64_t*>(positions), static_cast<const int32_t*>(slots), static_cast<bf16*>(q_out),
      static_cast<bf16*>(k_out), static_cast<bf16*>(v_out), rows, heads, kv_heads, theta, eps);
  return cudaGetLastError();
}

extern "C" int32_t cuteafd_glm_dflash_attention_d64(const void* q, const void* k_block, const void* v_block,
                                                const void* k_ring, const void* v_ring, const void* seq_slots,
                                                const void* ctx_lengths, const void* ctx_ends, void* out,
                                                void* workspace, int32_t sequences, int32_t block_rows,
                                                int32_t heads, int32_t kv_heads, int32_t ring, int32_t max_keys,
                                                int32_t window, float scale, void* stream) {
  return cuteafd::draft_attention::launch(q, k_block, v_block, k_ring, v_ring, seq_slots,
      ctx_lengths, ctx_ends, nullptr, out, workspace, sequences, block_rows, heads, kv_heads,
      ring, max_keys, window, false, 64, 2, false, scale, static_cast<cudaStream_t>(stream));
}

extern "C" int32_t cuteafd_glm_dflash_attention_d64_fp8(const void* q, const void* k_block, const void* v_block,
                                                const void* k_ring, const void* v_ring, const void* seq_slots,
                                                const void* ctx_lengths, const void* ctx_ends, void* out,
                                                void* workspace, int32_t sequences, int32_t block_rows,
                                                int32_t heads, int32_t kv_heads, int32_t ring, int32_t max_keys,
                                                int32_t window, float scale, void* stream) {
  return cuteafd::draft_attention::launch(q, k_block, v_block, k_ring, v_ring, seq_slots,
      ctx_lengths, ctx_ends, nullptr, out, workspace, sequences, block_rows, heads, kv_heads,
      ring, max_keys, window, false, 64, 2, true, scale, static_cast<cudaStream_t>(stream));
}

extern "C" uint64_t cuteafd_glm_dflash_attention_workspace_d64(int32_t sequences, int32_t kv_heads, int32_t max_keys) {
  // This ABI admits up to 64 queries per KV head, including smaller launches.
  return cuteafd::draft_attention::workspace(sequences, kv_heads, 2, max_keys, 64);
}

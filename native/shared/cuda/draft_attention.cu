#include "draft_attention.cuh"
#include "draft_kv.cuh"
#include <cuda_bf16.h>
#include <math_constants.h>
#include <mma.h>
#include <algorithm>
#include <limits.h>

namespace cuteafd::draft_attention {
namespace {
using bf16 = __nv_bfloat16;
namespace wmma = nvcuda::wmma;
constexpr int tile_keys = 32;
constexpr int threads = 128;
constexpr int scalar_keys = 128;
constexpr int scalar_queries = 16;
constexpr int scalar_context = 1024;

int sm_count() {
  // Cache per calling thread and device, not one card's SM count globally.
  static thread_local int cached_device = -1, cached_count = 0;
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return 0;
  if (device != cached_device) {
    int count = 0;
    if (cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess)
      return 0;
    cached_device = device;
    cached_count = count;
  }
  return cached_count;
}

int partition_capacity(int kv_heads, int slices, int max_keys, int sms) {
  const int tiles = (max_keys + tile_keys - 1) / tile_keys;
  return std::min(tiles, (2 * sms + kv_heads * slices - 1) / (kv_heads * slices));
}

// Four warps cover a 32-query x 32-key score tile. PV reuses those warps
// for two 16-query tiles and two disjoint halves of the value dimension.
template <int dim, bool fp8, bool paged>
__global__ void partial_kernel(const bf16* q, const bf16* k_block, const bf16* v_block,
    const bf16* k_ring, const bf16* v_ring, const int32_t* slots, const int32_t* lengths,
    const int32_t* ends, const int32_t* pages, int table_width, float* partial_o, float* partial_ml, int heads, int kv_heads,
    int block_rows, int ring, int slices, int partitions, int max_partitions,
    int max_keys, int window, bool causal, float scale) {
  constexpr int stride = dim + 8;
  constexpr int score_stride = 40;
  const int partition = blockIdx.x, kvh = blockIdx.y / slices, slice = blockIdx.y % slices;
  const int sequence = blockIdx.z, t = threadIdx.x, warp = t / 32;
  const int local = t / 4, sub = t % 4, query = slice * queries_per_slice + local;
  const int group = heads / kv_heads, queries = group * block_rows;
  const int ctx = lengths[sequence], end = ends[sequence], slot = slots[sequence];
  // Partition each sequence from its own key length and the device capacity,
  // independent of batch size and other sequences' lengths, so every request's
  // WMMA/softmax reduction order is bitwise invariant across batch compositions.
  const int total = min(ctx + block_rows, max_keys);
  const int seq_tiles = (total + tile_keys - 1) / tile_keys;
  const int seq_wanted = min(seq_tiles, max_partitions);
  const int seq_tpp = seq_wanted > 0 ? (seq_tiles + seq_wanted - 1) / seq_wanted : 1;
  const int first = partition * seq_tpp * tile_keys;
  const int last = min(total, first + seq_tpp * tile_keys);
  const uint64_t base = (((uint64_t(sequence) * kv_heads + kvh) * slices + slice)
      * partitions + partition) * queries_per_slice;
  if (first >= last) {
    if (sub == 0) {
      partial_ml[(base + local) * 2] = -CUDART_INF_F;
      partial_ml[(base + local) * 2 + 1] = 0;
    }
    for (int i = t; i < queries_per_slice * dim; i += threads) partial_o[base * dim + i] = 0;
    return;
  }
  __shared__ __align__(32) bf16 qs[queries_per_slice][stride];
  __shared__ __align__(32) bf16 ks[tile_keys][stride];
  __shared__ __align__(32) bf16 vs[tile_keys][stride];
  __shared__ __align__(32) bf16 probability[queries_per_slice][score_stride];
  // Scores and logical-row accumulator rescaling occupy this arena in turn.
  __shared__ __align__(32) float output[queries_per_slice][dim];
  __shared__ float factors[queries_per_slice];
  for (int i = t; i < queries_per_slice * dim; i += threads) {
    const int qi = slice * queries_per_slice + i / dim, d = i % dim;
    const int row = qi / group, head = kvh * group + qi % group;
    qs[i / dim][d] = qi < queries
        ? q[((uint64_t(sequence) * block_rows + row) * heads + head) * dim + d]
        : __float2bfloat16(0);
  }
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accum[dim / 32];
#pragma unroll
  for (int d = 0; d < dim / 32; ++d) wmma::fill_fragment(accum[d], 0.0f);
  float m = -CUDART_INF_F, l = 0;
  const int lowest = window > 0 ? end + min(query / group, block_rows - 1) - window + 1 : INT32_MIN;
  for (int tile = first; tile < last; tile += tile_keys) {
    __syncthreads();
    for (int i = t; i < tile_keys * (dim / 8); i += threads) {
      const int key = i / (dim / 8), d = i % (dim / 8) * 8, j = tile + key;
      uint4 k4 = make_uint4(0, 0, 0, 0), v4 = make_uint4(0, 0, 0, 0);
      if (j < last) {
        const bf16 *kp, *vp;
        if (j < ctx) {
          const int position = end - ctx + j;
          const uint64_t row = paged
              ? uint64_t(pages[uint64_t(sequence) * table_width + position / 64]) * 64 + position % 64
              : uint64_t(slot) * ring + position % ring;
          if constexpr (fp8) {
            k4 = draft_kv::load8<dim>(k_ring, row, kvh, kv_heads, d);
            v4 = draft_kv::load8<dim>(v_ring, row, kvh, kv_heads, d);
          } else {
            const uint64_t at = (row * kv_heads + kvh) * dim + d;
            kp = k_ring + at; vp = v_ring + at;
          }
        } else {
          const uint64_t at = ((uint64_t(sequence) * block_rows + j - ctx) * kv_heads + kvh) * dim + d;
          kp = k_block + at; vp = v_block + at;
        }
        if (!fp8 || j >= ctx) {
          k4 = *reinterpret_cast<const uint4*>(kp);
          v4 = *reinterpret_cast<const uint4*>(vp);
        }
      }
      *reinterpret_cast<uint4*>(&ks[key][d]) = k4;
      *reinterpret_cast<uint4*>(&vs[key][d]) = v4;
    }
    __syncthreads();
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> score;
    wmma::fill_fragment(score, 0.0f);
#pragma unroll
    for (int k = 0; k < dim; k += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> b;
      wmma::load_matrix_sync(a, &qs[(warp / 2) * 16][k], stride);
      wmma::load_matrix_sync(b, &ks[(warp % 2) * 16][k], stride);
      wmma::mma_sync(score, a, b, score);
    }
    wmma::store_matrix_sync(&output[(warp / 2) * 16][(warp % 2) * 16], score,
        dim, wmma::mem_row_major);
    __syncthreads();
    float logits[8], tile_max = -CUDART_INF_F;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int key = sub * 8 + j, position = tile + key;
      const bool visible = query < queries && position < last &&
          (position < ctx ? end - ctx + position >= lowest : !causal || position - ctx <= query / group);
      logits[j] = visible ? output[local][key] * scale : -CUDART_INF_F;
      tile_max = fmaxf(tile_max, logits[j]);
    }
    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, 1));
    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, 2));
    const float next_m = fmaxf(m, tile_max);
    const float factor = next_m == -CUDART_INF_F ? 1.0f : expf(m - next_m);
    float tile_sum = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float p = next_m == -CUDART_INF_F ? 0.0f : expf(logits[j] - next_m);
      probability[local][sub * 8 + j] = __float2bfloat16(p);
      tile_sum += p;
    }
    tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, 1);
    tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, 2);
    l = l * factor + tile_sum;
    m = next_m;
    if (sub == 0) factors[local] = factor;
    __syncthreads();
    // Fragment lane/row mapping is opaque. Rescale by logical row in shared
    // storage, then reload, as in the existing V4.1 WMMA attention path.
#pragma unroll
    for (int d = 0; d < dim / 32; ++d)
      wmma::store_matrix_sync(&output[(warp / 2) * 16][(warp % 2) * (dim / 2) + d * 16],
          accum[d], dim, wmma::mem_row_major);
    __syncthreads();
    for (int i = t; i < queries_per_slice * dim; i += threads) output[i / dim][i % dim] *= factors[i / dim];
    __syncthreads();
#pragma unroll
    for (int d = 0; d < dim / 32; ++d)
      wmma::load_matrix_sync(accum[d], &output[(warp / 2) * 16][(warp % 2) * (dim / 2) + d * 16],
          dim, wmma::mem_row_major);
#pragma unroll
    for (int k = 0; k < tile_keys; k += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> p;
      wmma::load_matrix_sync(p, &probability[(warp / 2) * 16][k], score_stride);
#pragma unroll
      for (int d = 0; d < dim / 32; ++d) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> v;
        wmma::load_matrix_sync(v, &vs[k][(warp % 2) * (dim / 2) + d * 16], stride);
        wmma::mma_sync(accum[d], p, v, accum[d]);
      }
    }
  }
#pragma unroll
  for (int d = 0; d < dim / 32; ++d)
    wmma::store_matrix_sync(&output[(warp / 2) * 16][(warp % 2) * (dim / 2) + d * 16],
        accum[d], dim, wmma::mem_row_major);
  __syncthreads();
  for (int i = t; i < queries_per_slice * dim; i += threads) partial_o[base * dim + i] = output[i / dim][i % dim];
  if (sub == 0) {
    partial_ml[(base + local) * 2] = m;
    partial_ml[(base + local) * 2 + 1] = l;
  }
}

// The original warp-per-query scalar arithmetic wins on skinny, short-context
// shapes. Keep its 128-key chunks, FP32 probabilities and reduction order.
template <int dim, bool windowed, bool has_sink, bool fp8, bool paged>
__global__ void scalar_partial_kernel(const bf16* q, const bf16* k_block, const bf16* v_block,
    const bf16* k_ring, const bf16* v_ring, const int32_t* slots, const int32_t* lengths,
    const int32_t* ends, const int32_t* pages, int table_width, float* partial_o, float* partial_ml, int heads, int kv_heads,
    int block_rows, int ring, int chunks, int max_keys, int window, bool causal, float scale) {
  const int chunk = blockIdx.x, kvh = blockIdx.y, sequence = blockIdx.z;
  const int group = heads / kv_heads, query = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int row = query / group, head = kvh * group + query % group;
  const int ctx = lengths[sequence], end = ends[sequence], slot = slots[sequence];
  constexpr bool original_layout = dim == 64 && !windowed && !has_sink;
  constexpr int partial_rows = original_layout ? queries_per_slice : scalar_queries;
  const int total = original_layout ? ctx + block_rows : min(ctx + block_rows, max_keys);
  const int first = chunk * scalar_keys, last = min(total, first + scalar_keys);
  const uint64_t base = ((uint64_t(sequence) * kv_heads + kvh) * chunks + chunk) * partial_rows + query;
  __shared__ float qs[partial_rows][dim];
  __shared__ __nv_bfloat162 ks[tile_keys][dim / 2 + 1];
  __shared__ __nv_bfloat162 vs[tile_keys][dim / 2];
  const bf16* q_row = q + ((uint64_t(sequence) * block_rows + row) * heads + head) * dim;
#pragma unroll
  for (int d = lane; d < dim; d += 32) qs[query][d] = __bfloat162float(q_row[d]);
  float m = -CUDART_INF_F, l = 0, acc[dim / 64] = {};
  float acc_pair[dim / 64] = {};
  const int lowest = windowed ? end + row - window + 1 : INT32_MIN;
  for (int tile = first; tile < last; tile += tile_keys) {
    const int n = min(tile_keys, last - tile);
    __syncthreads();
    for (int i = threadIdx.x; i < tile_keys * (dim / 8); i += blockDim.x) {
      const int key = i / (dim / 8), d = i % (dim / 8) * 8;
      uint4 k4 = make_uint4(0, 0, 0, 0), v4 = make_uint4(0, 0, 0, 0);
      if (key < n) {
        const int j = tile + key;
        const bf16 *kp, *vp;
        if (j < ctx) {
          const int position = end - ctx + j;
          const uint64_t row = paged
              ? uint64_t(pages[uint64_t(sequence) * table_width + position / 64]) * 64 + position % 64
              : uint64_t(slot) * ring + position % ring;
          if constexpr (fp8) {
            k4 = draft_kv::load8<dim>(k_ring, row, kvh, kv_heads, d);
            v4 = draft_kv::load8<dim>(v_ring, row, kvh, kv_heads, d);
          } else {
            const uint64_t at = (row * kv_heads + kvh) * dim + d;
            kp = k_ring + at; vp = v_ring + at;
          }
        } else {
          const uint64_t at = ((uint64_t(sequence) * block_rows + j - ctx) * kv_heads + kvh) * dim + d;
          kp = k_block + at; vp = v_block + at;
        }
        if (!fp8 || j >= ctx) {
          k4 = *reinterpret_cast<const uint4*>(kp);
          v4 = *reinterpret_cast<const uint4*>(vp);
        }
      }
      const auto* k2 = reinterpret_cast<const __nv_bfloat162*>(&k4);
#pragma unroll
      for (int e = 0; e < 4; ++e) ks[key][d / 2 + e] = k2[e];
      *reinterpret_cast<uint4*>(&vs[key][d / 2]) = v4;
    }
    __syncthreads();
    float dot = 0;
#pragma unroll 8
    for (int d = 0; d < dim / 2; ++d) {
      const float2 b = __bfloat1622float2(ks[lane][d]);
      dot = fmaf(qs[query][2 * d], b.x, dot);
      dot = fmaf(qs[query][2 * d + 1], b.y, dot);
    }
    const int j = tile + lane;
    const bool visible = lane < n && (j < ctx || !causal || j - ctx <= row)
        && (!windowed || j >= ctx || end - ctx + j >= lowest);
    const float score = visible ? dot * scale : -CUDART_INF_F;
    float tile_max = score;
    for (int o = 16; o; o >>= 1) tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, o));
    const float next_m = fmaxf(m, tile_max);
    const float factor = next_m == -CUDART_INF_F ? 1.0f : expf(m - next_m);
    const float p = next_m == -CUDART_INF_F ? 0.0f : expf(score - next_m);
    float tile_sum = p;
    for (int o = 16; o; o >>= 1) tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, o);
    l = l * factor + tile_sum;
    m = next_m;
#pragma unroll
    for (int d = 0; d < dim / 64; ++d) { acc[d] *= factor; acc_pair[d] *= factor; }
    for (int key = 0; key < n; ++key) {
      const float pk = __shfl_sync(0xffffffffu, p, key);
#pragma unroll
      for (int d = 0; d < dim / 64; ++d) {
        const float2 v = __bfloat1622float2(vs[key][lane + 32 * d]);
        acc[d] = fmaf(pk, v.x, acc[d]);
        acc_pair[d] = fmaf(pk, v.y, acc_pair[d]);
      }
    }
  }
  if (lane == 0) { partial_ml[base * 2] = m; partial_ml[base * 2 + 1] = l; }
#pragma unroll
  for (int d = 0; d < dim / 64; ++d) {
    partial_o[base * dim + 2 * lane + 64 * d] = acc[d];
    partial_o[base * dim + 2 * lane + 64 * d + 1] = acc_pair[d];
  }
}

template <int dim, int query_rows = queries_per_slice, bool has_sink = true, bool sliced = true>
__global__ void merge_kernel(const float* partial_o, const float* partial_ml, const bf16* sinks,
    bf16* out, int heads, int kv_heads, int block_rows, int slices, int partitions) {
  const int query = blockIdx.x, kvh = blockIdx.y, sequence = blockIdx.z, d = threadIdx.x;
  const int group = heads / kv_heads, row = query / group, head = kvh * group + query % group;
  const int slice = sliced ? query / query_rows : 0, local = sliced ? query % query_rows : query;
  const uint64_t base = sliced
      ? ((uint64_t(sequence) * kv_heads + kvh) * slices + slice) * partitions
      : (uint64_t(sequence) * kv_heads + kvh) * partitions;
  const float sink = has_sink && sinks ? __bfloat162float(sinks[head]) : -CUDART_INF_F;
  float m = sink;
  for (int p = 0; p < partitions; ++p)
    m = fmaxf(m, partial_ml[((base + p) * query_rows + local) * 2]);
  float l = has_sink && sinks ? expf(sink - m) : 0, value = 0;
  for (int p = 0; p < partitions; ++p) {
    const uint64_t at = (base + p) * query_rows + local;
    const float pm = partial_ml[at * 2];
    if (pm == -CUDART_INF_F) continue;
    const float factor = expf(pm - m);
    l += partial_ml[at * 2 + 1] * factor;
    value += partial_o[at * dim + d] * factor;
  }
  out[((uint64_t(sequence) * block_rows + row) * heads + head) * dim + d] = __float2bfloat16(value / l);
}

template <int dim, bool fp8, bool paged>
int dispatch_wmma(const void* q, const void* k_block, const void* v_block, const void* k_ring,
    const void* v_ring, const void* slots, const void* lengths, const void* ends, const void* sinks,
    void* out, void* scratch, int sequences, int block_rows, int heads, int kv_heads, int ring,
    int max_keys, int window, bool causal, int slices, int sms, float scale, cudaStream_t stream, const int32_t* pages, int table_width) {
  const int max_partitions = (2 * sms + kv_heads * slices - 1) / (kv_heads * slices);
  int partitions = partition_capacity(kv_heads, slices, max_keys, sms);
  if (sequences == 1) {
    const int tiles = (max_keys + tile_keys - 1) / tile_keys;
    const int wanted = std::min(tiles, max_partitions);
    const int tpp = wanted > 0 ? (tiles + wanted - 1) / wanted : 1;
    partitions = (tiles + tpp - 1) / tpp;
  }
  auto* partial_o = static_cast<float*>(scratch);
  auto* partial_ml = partial_o + uint64_t(sequences) * kv_heads * slices * partitions * queries_per_slice * dim;
  partial_kernel<dim, fp8, paged><<<dim3(partitions, kv_heads * slices, sequences), threads, 0, stream>>>(
      static_cast<const bf16*>(q), static_cast<const bf16*>(k_block), static_cast<const bf16*>(v_block),
      static_cast<const bf16*>(k_ring), static_cast<const bf16*>(v_ring), static_cast<const int32_t*>(slots),
      static_cast<const int32_t*>(lengths), static_cast<const int32_t*>(ends), pages, table_width, partial_o, partial_ml, heads,
      kv_heads, block_rows, ring, slices, partitions, max_partitions, max_keys, window, causal, scale);
  auto status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  const int queries = heads / kv_heads * block_rows;
  merge_kernel<dim><<<dim3(queries, kv_heads, sequences), dim, 0, stream>>>(partial_o, partial_ml,
      static_cast<const bf16*>(sinks), static_cast<bf16*>(out), heads, kv_heads, block_rows, slices, partitions);
  return cudaGetLastError();
}
template <int dim, bool windowed, bool has_sink, bool fp8, bool paged>
int dispatch_scalar(const void* q, const void* k_block, const void* v_block, const void* k_ring,
    const void* v_ring, const void* slots, const void* lengths, const void* ends, const void* sinks,
    void* out, void* scratch, int sequences, int block_rows, int heads, int kv_heads, int ring,
    int max_keys, int window, bool causal, float scale, cudaStream_t stream, const int32_t* pages, int table_width) {
  constexpr int partial_rows = dim == 64 && !windowed && !has_sink ? queries_per_slice : scalar_queries;
  const int queries = heads / kv_heads * block_rows;
  const int chunks = (max_keys + scalar_keys - 1) / scalar_keys;
  auto* partial_o = static_cast<float*>(scratch);
  auto* partial_ml = partial_o + uint64_t(sequences) * kv_heads * chunks * partial_rows * dim;
  scalar_partial_kernel<dim, windowed, has_sink, fp8, paged><<<dim3(chunks, kv_heads, sequences), queries * 32, 0, stream>>>(
      static_cast<const bf16*>(q), static_cast<const bf16*>(k_block), static_cast<const bf16*>(v_block),
      static_cast<const bf16*>(k_ring), static_cast<const bf16*>(v_ring), static_cast<const int32_t*>(slots),
      static_cast<const int32_t*>(lengths), static_cast<const int32_t*>(ends), pages, table_width, partial_o, partial_ml,
      heads, kv_heads, block_rows, ring, chunks, max_keys, window, causal, scale);
  auto status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  merge_kernel<dim, partial_rows, has_sink, false><<<dim3(queries, kv_heads, sequences), dim, 0, stream>>>(
      partial_o, partial_ml, static_cast<const bf16*>(sinks), static_cast<bf16*>(out),
      heads, kv_heads, block_rows, 1, chunks);
  return cudaGetLastError();
}

template <int dim, bool fp8, bool paged>
int route(const void* q, const void* k_block, const void* v_block, const void* k_ring,
    const void* v_ring, const void* slots, const void* lengths, const void* ends, const void* sinks,
    void* out, void* scratch, int sequences, int block_rows, int heads, int kv_heads, int ring,
    int max_keys, int window, bool causal, int slices, int sms, float scale, cudaStream_t stream, const int32_t* pages, int table_width) {
  const int queries = heads / kv_heads * block_rows;
  // max_keys includes the new block; the boundary is resident context only.
  if (queries <= scalar_queries && max_keys - block_rows <= scalar_context) {
#define SCALAR(W, S) dispatch_scalar<dim, W, S, fp8, paged>(q, k_block, v_block, k_ring, v_ring, slots, lengths, ends, \
    sinks, out, scratch, sequences, block_rows, heads, kv_heads, ring, max_keys, window, causal, scale, stream, pages, table_width)
    if (window > 0) return sinks ? SCALAR(true, true) : SCALAR(true, false);
    return sinks ? SCALAR(false, true) : SCALAR(false, false);
#undef SCALAR
  }
  return dispatch_wmma<dim, fp8, paged>(q, k_block, v_block, k_ring, v_ring, slots, lengths, ends, sinks,
      out, scratch, sequences, block_rows, heads, kv_heads, ring, max_keys, window, causal, slices, sms, scale, stream, pages, table_width);
}
}  // namespace

uint64_t workspace(int sequences, int kv_heads, int slices, int max_keys, int head_dim) {
  if (sequences < 1 || kv_heads < 1 || slices < 1 || max_keys < 1 ||
      (head_dim != 64 && head_dim != 128)) return 0;
  const int sms = sm_count();
  if (sms < 1) return 0;
  const int capacity = partition_capacity(kv_heads, slices, max_keys, sms);
  // Scalar queries bound block_rows too, so its largest total key count is 1040.
  const int scalar_chunks = (std::min(max_keys, scalar_context + scalar_queries) + scalar_keys - 1) / scalar_keys;
  const int scalar_rows = head_dim == 64 ? queries_per_slice : scalar_queries;
  const int rows = std::max(slices * capacity * queries_per_slice, scalar_chunks * scalar_rows);
  return uint64_t(sequences) * kv_heads * rows * (head_dim + 2) * sizeof(float);
}

int launch(const void* q, const void* k_block, const void* v_block, const void* k_ring,
    const void* v_ring, const void* slots, const void* lengths, const void* ends, const void* sinks,
    void* out, void* scratch, int sequences, int block_rows, int heads, int kv_heads, int ring,
    int max_keys, int window, bool causal, int head_dim, int slices, bool fp8, float scale, cudaStream_t stream, const int32_t* pages, int table_width) {
  if (sequences < 1 || block_rows < 1 || heads < 1 || kv_heads < 1 || heads % kv_heads ||
      ring < 1 || max_keys < block_rows || window < 0 || slices < 1 ||
      (heads / kv_heads * block_rows + queries_per_slice - 1) / queries_per_slice > slices ||
      !q || !k_block || !v_block || !k_ring || !v_ring || !slots || !lengths || !ends || !out || !scratch)
    return cudaErrorInvalidValue;
  const int sms = sm_count();
  if (sms < 1) return cudaErrorInvalidDevice;
#define ROUTE(D, F, P) route<D, F, P>(q, k_block, v_block, k_ring, v_ring, slots, lengths, ends, \
    sinks, out, scratch, sequences, block_rows, heads, kv_heads, ring, max_keys, window, causal, \
    slices, sms, scale, stream, pages, table_width)
  if (pages) {
    if (table_width < (max_keys - block_rows + 63) / 64 || window != 0) return cudaErrorInvalidValue;
    if (fp8 && head_dim == 128) return ROUTE(128, true, true);
    if (fp8 && head_dim == 64) return ROUTE(64, true, true);
    if (head_dim == 128) return ROUTE(128, false, true);
    if (head_dim == 64) return ROUTE(64, false, true);
  } else {
    if (fp8 && head_dim == 128) return ROUTE(128, true, false);
    if (fp8 && head_dim == 64) return ROUTE(64, true, false);
    if (head_dim == 128) return ROUTE(128, false, false);
    if (head_dim == 64) return ROUTE(64, false, false);
  }
#undef ROUTE
  return cudaErrorInvalidValue;
}
}  // namespace cuteafd::draft_attention

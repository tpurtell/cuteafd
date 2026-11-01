// Token input and output on the device for every family's decode loop:
// embedding rows gathered from a resident BF16 table by device token ids, and
// greedy token selection (with an optional log-probability) from FP32 logits.
//
// The gather is a byte copy of whole table rows, so its output is bit-identical
// to reading the same rows from the checkpoint. The greedy selection orders by
// logit, then by ascending token id, exactly like the host argmax it replaces
// (`TargetSamplingParams::select_token` for greedy rows and
// `logits_argmax_f32_wide_kernel`): NaN is never chosen and ties go to the
// lower id.
#include "common.h"

namespace {

constexpr int kGatherThreads = 256;
constexpr int kSelectThreads = 1024;

// One block per input row: `copies` contiguous copies of table[ids[row]] (with
// `index`, table[ids[index[row]]]), or of `fallback` for an id outside the
// table (zeros without one).
__global__ void __launch_bounds__(kGatherThreads) embed_gather_bf16_kernel(
    const uint16_t* __restrict__ table, size_t vocab, size_t hidden, const uint32_t* __restrict__ ids,
    const uint32_t* __restrict__ index, size_t copies, const uint16_t* __restrict__ fallback,
    uint16_t* __restrict__ out, int vectorized) {
  const size_t row = blockIdx.x;
  const uint32_t token = ids[index != nullptr ? index[row] : row];
  const uint16_t* source = token < vocab ? table + static_cast<size_t>(token) * hidden : fallback;
  uint16_t* destination = out + row * copies * hidden;
  if (vectorized) {
    const size_t vectors = hidden / 8;
    for (size_t i = threadIdx.x; i < vectors; i += blockDim.x) {
      const uint4 value = source != nullptr ? __ldg(reinterpret_cast<const uint4*>(source) + i)
                                            : make_uint4(0, 0, 0, 0);
      for (size_t c = 0; c < copies; ++c) {
        reinterpret_cast<uint4*>(destination + c * hidden)[i] = value;
      }
    }
    return;
  }
  for (size_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    const uint16_t value = source != nullptr ? source[i] : 0;
    for (size_t c = 0; c < copies; ++c) {
      destination[c * hidden + i] = value;
    }
  }
}

__device__ __forceinline__ uint32_t ordered_bits(float value) {
  value = value == 0.0f ? 0.0f : value;
  const uint32_t bits = __float_as_uint(value);
  const uint32_t mask = static_cast<uint32_t>(-static_cast<int32_t>(bits >> 31)) | 0x80000000U;
  return bits ^ mask;
}

__device__ __forceinline__ float from_ordered_bits(uint32_t ordered) {
  const uint32_t mask = (ordered >> 31) == 0 ? 0xffffffffU : 0x80000000U;
  return __uint_as_float(ordered ^ mask);
}

// Descending integer order: larger logits first, then lower token ids.
__device__ __forceinline__ uint64_t select_key(float logit, uint32_t token) {
  return (static_cast<uint64_t>(ordered_bits(logit)) << 32) | static_cast<uint64_t>(~token);
}

// Running log-sum-exp over finite logits: (max, sum of exp(x - max)).
struct Lse {
  float max;
  float sum;
};

__device__ __forceinline__ Lse lse_merge(Lse a, Lse b) {
  if (a.max == -CUDART_INF_F) return b;
  if (b.max == -CUDART_INF_F) return a;
  const float m = fmaxf(a.max, b.max);
  return Lse{m, a.sum * expf(a.max - m) + b.sum * expf(b.max - m)};
}

// One block per row. `out_status[row]` is 1 when the row holds a non-finite
// logit (the host argmax refuses such a row), else 0. `out_logprob` (nullable)
// receives log_softmax(row)[id].
template <bool LogProb>
__global__ void __launch_bounds__(kSelectThreads) logits_greedy_f32_kernel(
    const float* __restrict__ logits, size_t stride, size_t vocab, uint32_t* __restrict__ out_ids,
    float* __restrict__ out_logprob, uint32_t* __restrict__ out_status, int vectorized) {
  __shared__ uint64_t warp_best[kSelectThreads / 32];
  __shared__ int warp_invalid[kSelectThreads / 32];
  __shared__ Lse warp_lse[kSelectThreads / 32];
  const float* row = logits + blockIdx.x * stride;
  uint64_t best = select_key(-CUDART_INF_F, 0);
  int invalid = 0;
  Lse lse{-CUDART_INF_F, 0.0f};
  auto consider = [&](float score, uint32_t token) {
    invalid |= !isfinite(score);
    if (!isnan(score)) {
      const uint64_t key = select_key(score, token);
      best = key > best ? key : best;
    }
    if constexpr (LogProb) {
      if (isfinite(score)) {
        if (score > lse.max) {
          lse.sum = lse.sum * expf(lse.max - score) + 1.0f;
          lse.max = score;
        } else {
          lse.sum += expf(score - lse.max);
        }
      }
    }
  };
  if (vectorized) {
    const float4* row4 = reinterpret_cast<const float4*>(row);
    const uint32_t vectors = static_cast<uint32_t>(vocab / 4);
#pragma unroll 4
    for (uint32_t i = threadIdx.x; i < vectors; i += kSelectThreads) {
      const float4 v = __ldg(row4 + i);
      const uint32_t token = i * 4;
      consider(v.x, token);
      consider(v.y, token + 1);
      consider(v.z, token + 2);
      consider(v.w, token + 3);
    }
  } else {
    for (size_t i = threadIdx.x; i < vocab; i += kSelectThreads) {
      consider(row[i], static_cast<uint32_t>(i));
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    const uint64_t other = __shfl_down_sync(0xffffffffU, best, offset);
    best = other > best ? other : best;
    invalid |= __shfl_down_sync(0xffffffffU, invalid, offset);
    if constexpr (LogProb) {
      const Lse other_lse{__shfl_down_sync(0xffffffffU, lse.max, offset),
                          __shfl_down_sync(0xffffffffU, lse.sum, offset)};
      lse = lse_merge(lse, other_lse);
    }
  }
  const int warp = threadIdx.x / 32;
  if ((threadIdx.x & 31) == 0) {
    warp_best[warp] = best;
    warp_invalid[warp] = invalid;
    if constexpr (LogProb) warp_lse[warp] = lse;
  }
  __syncthreads();
  if (warp != 0) {
    return;
  }
  best = warp_best[threadIdx.x];
  invalid = warp_invalid[threadIdx.x];
  if constexpr (LogProb) lse = warp_lse[threadIdx.x];
  for (int offset = 16; offset > 0; offset >>= 1) {
    const uint64_t other = __shfl_down_sync(0xffffffffU, best, offset);
    best = other > best ? other : best;
    invalid |= __shfl_down_sync(0xffffffffU, invalid, offset);
    if constexpr (LogProb) {
      const Lse other_lse{__shfl_down_sync(0xffffffffU, lse.max, offset),
                          __shfl_down_sync(0xffffffffU, lse.sum, offset)};
      lse = lse_merge(lse, other_lse);
    }
  }
  if (threadIdx.x == 0) {
    out_ids[blockIdx.x] = ~static_cast<uint32_t>(best);
    out_status[blockIdx.x] = invalid ? 1u : 0u;
    if constexpr (LogProb) {
      const float score = from_ordered_bits(static_cast<uint32_t>(best >> 32));
      out_logprob[blockIdx.x] = score - (lse.max + logf(lse.sum));
    }
  }
}

}  // namespace

extern "C" cuteafd_status_t cuteafd_cuda_embed_gather_bf16_async(
    const uint16_t* table, size_t vocab, size_t hidden, const uint32_t* token_ids,
    const uint32_t* index, size_t rows, size_t copies, const uint16_t* fallback, uint16_t* out,
    void* cuda_stream) {
  if (table == nullptr || token_ids == nullptr || out == nullptr || rows == 0 || vocab == 0 ||
      hidden == 0 || copies == 0 || rows > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return CUTEAFD_STATUS_INVALID_ARGUMENT;
  }
  size_t ignored = 0;
  if (!checked_mul(vocab, hidden, &ignored) || !checked_mul(rows, copies, &ignored) ||
      !checked_mul(ignored, hidden, &ignored)) {
    return CUTEAFD_STATUS_INVALID_ARGUMENT;
  }
  const auto aligned = [](const void* p) { return reinterpret_cast<uintptr_t>(p) % 16 == 0; };
  const int vectorized = hidden % 8 == 0 && aligned(table) && aligned(out) &&
                         (fallback == nullptr || aligned(fallback));
  embed_gather_bf16_kernel<<<static_cast<int>(rows), kGatherThreads, 0,
                             reinterpret_cast<cudaStream_t>(cuda_stream)>>>(
      table, vocab, hidden, token_ids, index, copies, fallback, out, vectorized);
  return status_from_cuda(cudaGetLastError());
}

// Final-logit soft cap in place: x = tanh(x / cap) * cap in FP32 (Gemma 2,
// Mistral Large 4: `logits / c; tanh; * c`, each an FP32 op, as the reference).
__global__ void logits_softcap_f32_kernel(float* __restrict__ logits, size_t rows, size_t vocab,
                                          size_t stride, float cap) {
  const size_t row = blockIdx.y;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < vocab; i += size_t(gridDim.x) * blockDim.x) {
    float* x = logits + row * stride + i;
    *x = tanhf(__fdiv_rn(*x, cap)) * cap;
  }
}

extern "C" cuteafd_status_t cuteafd_cuda_logits_softcap_f32_async(
    float* logits, size_t rows, size_t vocab, size_t stride, float cap, void* cuda_stream) {
  if (logits == nullptr || rows == 0 || vocab == 0 || stride < vocab || !(cap > 0.0f) ||
      rows > 65535) {
    return CUTEAFD_STATUS_INVALID_ARGUMENT;
  }
  const dim3 grid(static_cast<unsigned>((vocab + 255) / 256 < 1024 ? (vocab + 255) / 256 : 1024),
                  static_cast<unsigned>(rows));
  logits_softcap_f32_kernel<<<grid, 256, 0, reinterpret_cast<cudaStream_t>(cuda_stream)>>>(
      logits, rows, vocab, stride, cap);
  return status_from_cuda(cudaGetLastError());
}

extern "C" cuteafd_status_t cuteafd_cuda_logits_greedy_f32_async(
    const float* logits, size_t rows, size_t vocab, size_t stride, uint32_t* out_ids,
    float* out_logprob, uint32_t* out_status, void* cuda_stream) {
  if (logits == nullptr || out_ids == nullptr || out_status == nullptr || rows == 0 || vocab == 0 ||
      stride < vocab || vocab > std::numeric_limits<uint32_t>::max() ||
      rows > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return CUTEAFD_STATUS_INVALID_ARGUMENT;
  }
  const int vectorized =
      vocab % 4 == 0 && stride % 4 == 0 && reinterpret_cast<uintptr_t>(logits) % 16 == 0;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  if (out_logprob != nullptr) {
    logits_greedy_f32_kernel<true><<<static_cast<int>(rows), kSelectThreads, 0, stream>>>(
        logits, stride, vocab, out_ids, out_logprob, out_status, vectorized);
  } else {
    logits_greedy_f32_kernel<false><<<static_cast<int>(rows), kSelectThreads, 0, stream>>>(
        logits, stride, vocab, out_ids, out_logprob, out_status, vectorized);
  }
  return status_from_cuda(cudaGetLastError());
}

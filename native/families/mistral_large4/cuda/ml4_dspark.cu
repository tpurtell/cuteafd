// Mistral Large 4 mtp_dspark: the recurrent (RNN) Markov head, greedy.
//
// For each draft step k (sequential within a proposal, parallel over
// sequences), with prev the anchor at k = 0, else draft k - 1:
//   joint = bf16(joint_w . [state | embedding[prev] | h_k] + joint_b)  (1536)
//   gate = sigmoid(joint[0:512]), cand = tanh(joint[512:1024]), out = tanh(joint[1024:])
//   state = gate * state + (1 - gate) * cand        (each op rounded to BF16)
//   e = out;  draft_k = argmax_v base[s, k, v] + bf16(output[v] . e)
// python/reference/families/mistral_large4/dspark.py (vLLM bd4fa263's RNNHead)
// is the oracle. BF16 boundaries follow the reference's BF16 tensor ops; the
// joint projection accumulates in FP32. The vocabulary argmax prunes rows
// whose base logit plus |output[v]| |e| (Cauchy-Schwarz, with slack for the
// BF16 rounding of the bias) cannot reach the value of the row's base argmax;
// surviving rows are scored exactly. Ties go to the lowest index.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

namespace {

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float f(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float r(float v) { return __bfloat162float(__float2bfloat16(v)); }

constexpr int kRank = 512;
constexpr int kMaxSequences = 32;
constexpr int kBlocks = 296;
constexpr int kThreads = 256;

// One block per sequence: joint = W . [state | w1[prev] | h] + b, then the gates.
// state/e are BF16 [sequences, 512]; h is BF16 [sequences * block, hidden].
__global__ void joint_kernel(const bf16* w1, const bf16* joint_w, const bf16* joint_b, const bf16* hidden,
                             const uint32_t* anchors, const uint32_t* drafts, bf16* state, bf16* embed,
                             int hidden_width, int block, int step) {
  extern __shared__ float x[];  // [2 * kRank + hidden_width]
  __shared__ float joint[3 * kRank];
  const int s = blockIdx.x;
  const uint32_t prev = step == 0 ? anchors[s] : drafts[s * block + step - 1];
  const int width = 2 * kRank + hidden_width;
  const bf16* h = hidden + (uint64_t(s) * block + step) * hidden_width;
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    float v;
    if (i < kRank) v = step == 0 ? 0.0f : f(state[s * kRank + i]);
    else if (i < 2 * kRank) v = f(w1[uint64_t(prev) * kRank + (i - kRank)]);
    else v = f(h[i - 2 * kRank]);
    x[i] = v;
  }
  __syncthreads();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, warps = blockDim.x / 32;
  for (int o = warp; o < 3 * kRank; o += warps) {
    const bf16* row = joint_w + uint64_t(o) * width;
    float sum = 0;
    for (int i = lane; i < width; i += 32) sum = fmaf(f(row[i]), x[i], sum);
    for (int k = 16; k; k >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, k);
    if (lane == 0) joint[o] = r(sum + f(joint_b[o]));
  }
  __syncthreads();
  for (int i = threadIdx.x; i < kRank; i += blockDim.x) {
    const float g = r(1.0f / (1.0f + __expf(-joint[i])));
    const float c = r(tanhf(joint[kRank + i]));
    const float old = x[i];
    const float next = r(r(g * old) + r(r(1.0f - g) * c));
    state[s * kRank + i] = __float2bfloat16(next);
    embed[s * kRank + i] = __float2bfloat16(r(tanhf(joint[2 * kRank + i])));
  }
}

// Lane's share of output[v] . e (16 elements per lane), warp-summed.
__device__ __forceinline__ float dot512(const bf16* w2, uint32_t v, const bf16* e, int lane) {
  float dot = 0;
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int base = 256 * half + 8 * lane;
    const uint4 raw = *reinterpret_cast<const uint4*>(w2 + uint64_t(v) * kRank + base);
    const __nv_bfloat162* pair = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float2 w = __bfloat1622float2(pair[k]);
      dot = fmaf(w.x, f(e[base + 2 * k]), dot);
      dot = fmaf(w.y, f(e[base + 2 * k + 1]), dot);
    }
  }
  for (int o = 16; o; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
  return r(dot);
}

__global__ void vocab_partial_kernel(const float* logits, const bf16* w2, const float* norms, const uint32_t* top,
                                     const bf16* embed, float* partial_value, uint32_t* partial_index,
                                     int sequences, int vocab, int block, int step) {
  // BF16 embeddings (already BF16-rounded): 32 KiB of static shared memory.
  __shared__ bf16 e[kMaxSequences][kRank];
  __shared__ float floor_value[kMaxSequences], reach[kMaxSequences];
  for (int i = threadIdx.x; i < sequences * kRank; i += blockDim.x) e[i / kRank][i % kRank] = embed[i];
  __syncthreads();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, warps = blockDim.x / 32;
  for (int s = warp; s < sequences; s += warps) {
    float sum = 0;
    for (int c = lane; c < kRank; c += 32) sum += f(e[s][c]) * f(e[s][c]);
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    const uint64_t row = uint64_t(s) * block + step;
    const uint32_t t = top[row];
    const float value = logits[row * vocab + t] + dot512(w2, t, e[s], lane);
    if (lane == 0) {
      floor_value[s] = value;
      // BF16 rounding of the bias adds at most 2^-8 relative.
      reach[s] = sqrtf(sum) * 1.01f + 1e-5f;
    }
  }
  __syncthreads();
  float best[kMaxSequences];
  uint32_t best_index[kMaxSequences];
#pragma unroll
  for (int s = 0; s < kMaxSequences; ++s) {
    best[s] = -CUDART_INF_F;
    best_index[s] = UINT32_MAX;
  }
  for (int v = blockIdx.x * warps + warp; v < vocab; v += gridDim.x * warps) {
    const float n = norms[v];
    bool any = false;
#pragma unroll
    for (int s = 0; s < kMaxSequences; ++s) {
      if (s >= sequences) break;
      any |= logits[(uint64_t(s) * block + step) * vocab + v] + n * reach[s] >= floor_value[s];
    }
    if (!any) continue;
#pragma unroll
    for (int s = 0; s < kMaxSequences; ++s) {
      if (s >= sequences) break;
      const float value = logits[(uint64_t(s) * block + step) * vocab + v] + dot512(w2, v, e[s], lane);
      if (value > best[s] || (value == best[s] && uint32_t(v) < best_index[s])) {
        best[s] = value;
        best_index[s] = v;
      }
    }
  }
  __shared__ float values[kThreads / 32][kMaxSequences];
  __shared__ uint32_t indices[kThreads / 32][kMaxSequences];
  if (lane == 0) {
#pragma unroll
    for (int s = 0; s < kMaxSequences; ++s) {
      values[warp][s] = best[s];
      indices[warp][s] = best_index[s];
    }
  }
  __syncthreads();
  if (threadIdx.x < sequences) {
    const int s = threadIdx.x;
    float b = values[0][s];
    uint32_t bi = indices[0][s];
    for (int i = 1; i < warps; ++i)
      if (values[i][s] > b || (values[i][s] == b && indices[i][s] < bi)) {
        b = values[i][s];
        bi = indices[i][s];
      }
    partial_value[s * gridDim.x + blockIdx.x] = b;
    partial_index[s * gridDim.x + blockIdx.x] = bi;
  }
}

__global__ void row_argmax_kernel(const float* logits, uint32_t* top, int vocab) {
  const uint64_t row = blockIdx.x;
  float b = -CUDART_INF_F;
  uint32_t bi = UINT32_MAX;
  for (int v = threadIdx.x; v < vocab; v += blockDim.x) {
    const float x = logits[row * vocab + v];
    if (x > b) {
      b = x;
      bi = v;
    }
  }
  for (int o = 16; o; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, b, o);
    const uint32_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
    if (ov > b || (ov == b && oi < bi)) {
      b = ov;
      bi = oi;
    }
  }
  __shared__ float wv[32];
  __shared__ uint32_t wi[32];
  if (threadIdx.x % 32 == 0) {
    wv[threadIdx.x / 32] = b;
    wi[threadIdx.x / 32] = bi;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 1; i < int(blockDim.x / 32); ++i)
      if (wv[i] > b || (wv[i] == b && wi[i] < bi)) {
        b = wv[i];
        bi = wi[i];
      }
    top[row] = bi;
  }
}

__global__ void row_norms_kernel(const bf16* w2, float* norms, int vocab) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  if (warp >= vocab) return;
  float sum = 0;
  for (int c = lane; c < kRank; c += 32) {
    const float x = f(w2[uint64_t(warp) * kRank + c]);
    sum += x * x;
  }
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  if (lane == 0) norms[warp] = sqrtf(sum) * 1.0001f + 1e-6f;
}

__global__ void final_kernel(const float* partial_value, const uint32_t* partial_index, uint32_t* drafts,
                             int partials, int block, int step) {
  const int s = blockIdx.x, lane = threadIdx.x;
  float b = -CUDART_INF_F;
  uint32_t bi = UINT32_MAX;
  for (int i = lane; i < partials; i += 32) {
    const float v = partial_value[s * partials + i];
    const uint32_t index = partial_index[s * partials + i];
    if (v > b || (v == b && index < bi)) {
      b = v;
      bi = index;
    }
  }
  for (int o = 16; o; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, b, o);
    const uint32_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
    if (ov > b || (ov == b && oi < bi)) {
      b = ov;
      bi = oi;
    }
  }
  if (lane == 0) drafts[s * block + step] = bi;
}

}  // namespace

// Workspace bytes cuteafd_ml4_dspark_markov needs for `sequences` x `block` rows.
extern "C" uint64_t cuteafd_ml4_dspark_markov_workspace(int32_t sequences, int32_t block) {
  // partials (value + index), top per row, state and embedding (BF16).
  return uint64_t(sequences) * kBlocks * 8 + uint64_t(sequences) * block * 4 + uint64_t(sequences) * kRank * 4;
}

// norms [vocab] F32 = slackened row norms of the output projection (BF16 [vocab, 512]), once at load.
extern "C" int32_t cuteafd_ml4_dspark_markov_norms(const void* output, void* norms, int32_t vocab, int32_t rank,
                                                   void* stream) {
  if (vocab < 1 || rank != kRank) return cudaErrorInvalidValue;
  row_norms_kernel<<<(vocab * 32 + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const bf16*>(output), static_cast<float*>(norms), vocab);
  return cudaGetLastError();
}

// drafts [sequences, block] U32, greedy, from the base logits (FP32 [sequences * block,
// vocab], the soft-capped target head) and the final-norm rows `hidden` (BF16
// [sequences * block, hidden_width]); embedding/output BF16 [vocab, 512], joint_w BF16
// [1536, 1024 + hidden_width], joint_b BF16 [1536]; `norms` from
// cuteafd_ml4_dspark_markov_norms.
extern "C" int32_t cuteafd_ml4_dspark_markov(const void* logits, const void* hidden, const void* embedding,
                                             const void* joint_w, const void* joint_b, const void* output,
                                             const void* norms, const void* anchors, void* drafts, void* workspace,
                                             int32_t sequences, int32_t block, int32_t vocab, int32_t rank,
                                             int32_t hidden_width, void* stream) {
  if (sequences < 1 || sequences > kMaxSequences || block < 1 || rank != kRank || hidden_width < 1 ||
      hidden_width % 8)
    return cudaErrorInvalidValue;
  auto s = static_cast<cudaStream_t>(stream);
  auto* values = static_cast<float*>(workspace);
  auto* indices = reinterpret_cast<uint32_t*>(values + uint64_t(sequences) * kBlocks);
  auto* top = indices + uint64_t(sequences) * kBlocks;
  auto* state = reinterpret_cast<bf16*>(top + uint64_t(sequences) * block);
  auto* embed = state + uint64_t(sequences) * kRank;
  const size_t shared = size_t(2 * kRank + hidden_width) * sizeof(float);
  if (shared > 48 * 1024) {
    const cudaError_t error = cudaFuncSetAttribute(joint_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                   int(shared));
    if (error != cudaSuccess) return error;
  }
  row_argmax_kernel<<<sequences * block, 1024, 0, s>>>(static_cast<const float*>(logits), top, vocab);
  for (int step = 0; step < block; ++step) {
    joint_kernel<<<sequences, 512, shared, s>>>(
        static_cast<const bf16*>(embedding), static_cast<const bf16*>(joint_w), static_cast<const bf16*>(joint_b),
        static_cast<const bf16*>(hidden), static_cast<const uint32_t*>(anchors),
        static_cast<const uint32_t*>(drafts), state, embed, hidden_width, block, step);
    vocab_partial_kernel<<<kBlocks, kThreads, 0, s>>>(
        static_cast<const float*>(logits), static_cast<const bf16*>(output), static_cast<const float*>(norms), top,
        embed, values, indices, sequences, vocab, block, step);
    final_kernel<<<sequences, 32, 0, s>>>(values, indices, static_cast<uint32_t*>(drafts), kBlocks, block, step);
  }
  return cudaGetLastError();
}

// Load-time BF16 -> FP8 quantization straight from host-mapped push ring
// slots into the final device operand, bit-for-bit the coordinator loaders'
// host rule (fp8_load_quant.h). One CTA per scale unit (a 128-row Block band,
// or one row for Row128/Channel). Order-independent steps (amax, per-element
// quantization, each row's float64 error) run in parallel; the order-dependent
// fold of row errors runs on one thread in the host's order.
#include <cuda_runtime.h>
#include <stdint.h>

#include "fp8_load_quant.h"

namespace lq = cuteafd_load_quant;

namespace {
struct Job {
  const lq::Segment* segments;  // device-visible (pinned host) fragment list
  int32_t count;
  int32_t rows;  // 128 (Block) or 1 (Row128/Channel)
  int32_t cols;
  int32_t layout;
  int32_t rule;
  uint8_t* q;          // rows x cols E4M3, row stride q_stride
  int64_t q_stride;
  float* scales;       // kb scales (one unit)
};

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 512;  // cols <= 65536

__global__ void load_quant_kernel(const Job* jobs) {
  const Job job = jobs[blockIdx.x];
  const lq::Segmented src{job.segments, job.count};
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
  for (int64_t b = threadIdx.x; b < kb; b += kThreads) job.scales[b] = scales[b];
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

// The Job ABI, for the Rust side to check before launching.
extern "C" int32_t cuteafd_fp8_load_quant_abi(uint64_t* job_bytes, uint64_t* segment_bytes, uint64_t* max_cols) {
  *job_bytes = sizeof(Job);
  *segment_bytes = sizeof(lq::Segment);
  *max_cols = uint64_t(kMaxBlocks) * 128;
  return 0;
}

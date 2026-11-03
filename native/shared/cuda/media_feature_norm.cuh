#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {
// Official public media features use weightless FP32 RMSNorm, then one BF16
// store. Keep this after raw-tower observers, with no extra device allocation.
__global__ void media_feature_norm_kernel(uint16_t* data, int width, float epsilon) {
  auto* row = reinterpret_cast<__nv_bfloat16*>(data) + size_t(blockIdx.x) * width;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < width; col += blockDim.x) {
    float value = __bfloat162float(row[col]);
    sum = __fadd_rn(sum, __fmul_rn(value, value));
  }
  for (int delta = 16; delta; delta /= 2)
    sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, delta));
  __shared__ float warps[8];
  if ((threadIdx.x & 31) == 0) warps[threadIdx.x / 32] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    sum = threadIdx.x < 8 ? warps[threadIdx.x] : 0.0f;
    for (int delta = 16; delta; delta /= 2)
      sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, delta));
    if (threadIdx.x == 0)
      warps[0] = powf(__fadd_rn(sum / float(width), epsilon), -0.5f);
  }
  __syncthreads();
  float inverse = warps[0];
  for (int col = threadIdx.x; col < width; col += blockDim.x)
    row[col] = __float2bfloat16_rn(__fmul_rn(__bfloat162float(row[col]), inverse));
}
int media_feature_norm(uint16_t* data, int rows, int width, float epsilon, cudaStream_t stream) {
  media_feature_norm_kernel<<<rows, 256, 0, stream>>>(data, width, epsilon);
  return cudaGetLastError();
}
}

#include "cuteafd_experts.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>
#include "expert_pack_layout.h"
#include "cuteafd_mxfp4_layouts.h"

namespace {

// Aligned widths retain N256/K128; exact tails use compact N128/K32 tiles.
template<bool Gated, bool Scales>
__global__ void pack(const uint8_t* first, const uint8_t* second, uint8_t* output,
    uint32_t n, uint32_t k, uint32_t n_pad, uint64_t count, bool compact) {
  for (uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += uint64_t(gridDim.x) * blockDim.x) {
    uint32_t row, col;
    cuteafd_expert_pack_layout_coordinates(index, Gated ? 2 * n_pad : n, k,
        n_pad, Scales, compact, &row, &col);
    const uint8_t* source = first;
    if constexpr (Gated) {
      if (row >= n_pad) { row -= n_pad; source = second; }
    }
    constexpr uint32_t divisor = Scales ? 32 : 8;
    const bool valid = row < n && col < k / divisor;
    if constexpr (Scales) {
      output[index] = valid ? source[uint64_t(row) * (k / divisor) + col] : 0;
    } else {
      reinterpret_cast<uint32_t*>(output)[index] = valid
        ? reinterpret_cast<const uint32_t*>(source)[uint64_t(row) * (k / divisor) + col] : 0;
    }
  }
}

__global__ void pack_scale_codes(const cuteafd_scale_code_job_t* jobs,
    uint64_t* error) {
  const auto job = jobs[blockIdx.y];
  const unsigned lane = threadIdx.x % 32;
  const unsigned warp = threadIdx.x / 32;
  const uint64_t tiles = (uint64_t(job.columns) + 3) / 4;
  const uint64_t code_bytes = uint64_t(job.stored_rows) * tiles * 2;
  for (uint64_t row = uint64_t(blockIdx.x) * 8 + warp;
       row < job.rows; row += uint64_t(gridDim.x) * 8) {
    const auto source = job.source + row * job.pitch;
    unsigned low = 255, high = 0;
    for (unsigned col = lane; col < job.columns; col += 32) {
      low = min(low, unsigned(source[col]));
      high = max(high, unsigned(source[col]));
    }
    for (unsigned delta = 16; delta; delta /= 2) {
      low = min(low, __shfl_xor_sync(0xffffffff, low, delta));
      high = max(high, __shfl_xor_sync(0xffffffff, high, delta));
    }
    if (high - low > 15) {
      if (!lane) atomicMin(reinterpret_cast<unsigned long long*>(error),
          (uint64_t(blockIdx.y) << 32) | row);
      continue;
    }
    const uint64_t resident = uint64_t(job.start_row) + row;
    if (!lane) job.destination[code_bytes + resident] = uint8_t(low);
    for (uint64_t tile = lane; tile < tiles; tile += 32) {
      uint16_t code = 0;
      for (unsigned byte = 0; byte < 4; ++byte) {
        const uint64_t col = tile * 4 + byte;
        if (col < job.columns) code |= uint16_t(source[col] - low) << (byte * 4);
      }
      const uint64_t tile_start = resident / 256 * 256;
      const uint64_t tile_rows = min(uint64_t(256), uint64_t(job.stored_rows) - tile_start);
      const uint64_t index = tile_start * tiles + tile * tile_rows + resident % 256;
      reinterpret_cast<uint16_t*>(job.destination)[index] = code;
    }
  }
}

bool overlaps(const void* a, uint64_t an, const void* b, uint64_t bn) {
  const auto av = reinterpret_cast<uintptr_t>(a), bv = reinterpret_cast<uintptr_t>(b);
  return av <= bv ? bv - av < an : av - bv < bn;
}
}

extern "C" int32_t cuteafd_v41_pack_scale_codes_async(
    const cuteafd_scale_code_job_t* jobs, uint32_t count, uint64_t* error,
    void* stream) {
  if (!jobs || !count || count > 65535 || !error ||
      reinterpret_cast<uintptr_t>(error) % 8) return cudaErrorInvalidValue;
  uint32_t max_rows = 0;
  for (uint32_t i = 0; i < count; ++i) {
    const auto& job = jobs[i];
    const uint64_t tiles = (uint64_t(job.columns) + 3) / 4;
    const uint64_t bytes = uint64_t(job.stored_rows) * (tiles * 2 + 1);
    if (!job.source || !job.destination || !job.rows || !job.columns ||
        job.stored_rows % 32 || job.start_row > job.stored_rows ||
        job.rows > job.stored_rows - job.start_row || job.pitch < job.columns ||
        job.columns > job.source_bytes ||
        uint64_t(job.rows - 1) > (job.source_bytes - job.columns) / job.pitch ||
        bytes > job.destination_bytes || reinterpret_cast<uintptr_t>(job.destination) % 2 ||
        overlaps(job.source, job.source_bytes, job.destination, bytes))
      return cudaErrorInvalidValue;
    max_rows = std::max(max_rows, job.rows);
  }
  int device, sms;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  status = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  if (status != cudaSuccess) return status;
  if (sms <= 0) return cudaErrorInvalidDevice;
  const unsigned blocks = std::min((max_rows + 7) / 8, unsigned(sms) * 4);
  pack_scale_codes<<<dim3(blocks, count), 256, 0, static_cast<cudaStream_t>(stream)>>>(jobs, error);
  return cudaGetLastError();
}

extern "C" int32_t cuteafd_expert_packed_sizes(uint32_t intermediate,
    uint64_t bytes[4]) {
  // Storage and exported kernels use the same measured geometry contract.
  const uint64_t hidden = cuteafd_expert_hidden();
  if (!bytes || !intermediate || intermediate % 32 != 0 || intermediate > 8192)
    return cudaErrorInvalidValue;
  const uint64_t padded = cuteafd_mxfp4_stored_width(hidden, intermediate);
  if (!padded) return cudaErrorInvalidValue;
  bytes[0] = padded * hidden;
  bytes[1] = padded * hidden / 16;
  bytes[2] = hidden * padded / 2;
  bytes[3] = hidden * padded / 32;
  return cudaSuccess;
}

// Coalesced omitted row tails; never touch the delivered prefix of a row.
extern "C" int32_t cuteafd_zero_pitched_async(void* destination,
    uint64_t destination_bytes, uint64_t pitch, uint64_t width,
    uint64_t rows, void* stream) {
  if (!destination || !width || !rows || pitch < width ||
      width > destination_bytes ||
      rows - 1 > (destination_bytes - width) / pitch)
    return cudaErrorInvalidValue;
  if (rows == 1)
    return cudaMemsetAsync(destination, 0, width, static_cast<cudaStream_t>(stream));
  return cudaMemset2DAsync(destination, pitch, 0, width, rows,
      static_cast<cudaStream_t>(stream));
}

// One plane of the already-fused source image. Reuse the exact mount pack
// instantiations; the caller owns one out-of-place scratch across all planes.
extern "C" int32_t cuteafd_pack_expert_plane_async(
    const uint8_t* source, uint64_t source_bytes, uint8_t* destination,
    uint64_t destination_bytes, uint32_t intermediate, uint32_t plane, void* stream) {
  uint64_t sizes[4];
  if (plane >= 4 || !source || !destination ||
      cuteafd_expert_packed_sizes(intermediate, sizes)) return cudaErrorInvalidValue;
  const auto hidden = cuteafd_expert_hidden();
  const auto padded = cuteafd_mxfp4_stored_width(hidden, intermediate);
  const bool compact = padded == intermediate && intermediate % 128;
  const bool gated = plane < 2, scales = plane % 2;
  const uint64_t required = gated ? sizes[plane]
      : uint64_t(hidden) * intermediate / (scales ? 32 : 2);
  if (source_bytes < required || destination_bytes < sizes[plane] ||
      reinterpret_cast<uintptr_t>(source) % 4 ||
      reinterpret_cast<uintptr_t>(destination) % 16 ||
      overlaps(source, required, destination, sizes[plane])) return cudaErrorInvalidValue;
  int device, sms;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  status = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  if (status != cudaSuccess) return status;
  if (sms <= 0) return cudaErrorInvalidDevice;
  const uint64_t count = sizes[plane] / (scales ? 1 : 4);
  const unsigned blocks = static_cast<unsigned>(
      std::min<uint64_t>((count + 255) / 256, uint64_t(sms) * 4));
  auto s = static_cast<cudaStream_t>(stream);
  if (gated && !scales)
    pack<false, false><<<blocks, 256, 0, s>>>(source, nullptr, destination,
        2 * padded, hidden, compact ? padded : 0, sizes[plane] / 4, compact);
  else if (gated)
    pack<false, true><<<blocks, 256, 0, s>>>(source, nullptr, destination,
        2 * padded, hidden, compact ? padded : 0, sizes[plane], compact);
  else if (!scales)
    pack<false, false><<<blocks, 256, 0, s>>>(source, nullptr, destination,
        hidden, intermediate, 0, sizes[plane] / 4, compact);
  else
    pack<false, true><<<blocks, 256, 0, s>>>(source, nullptr, destination,
        hidden, intermediate, 0, sizes[plane], compact);
  return cudaGetLastError();
}

extern "C" int32_t cuteafd_pack_expert_weights_async(
    const uint8_t* const sources[3], uint8_t* const destinations[2],
    uint32_t intermediate, void* stream) {
  uint64_t sizes[4];
  if (!sources || !destinations || cuteafd_expert_packed_sizes(intermediate, sizes))
    return cudaErrorInvalidValue;
  const auto hidden = cuteafd_expert_hidden();
  const uint64_t source_bytes = uint64_t(intermediate) * hidden / 2;
  for (int i = 0; i < 3; ++i)
    if (!sources[i] || reinterpret_cast<uintptr_t>(sources[i]) % 4)
      return cudaErrorInvalidValue;
  for (int i = 0; i < 2; ++i) {
    if (!destinations[i] || reinterpret_cast<uintptr_t>(destinations[i]) % 16)
      return cudaErrorInvalidValue;
    for (int j = 0; j < 3; ++j)
      if (overlaps(destinations[i], sizes[i * 2], sources[j], source_bytes))
        return cudaErrorInvalidValue;
  }
  if (overlaps(destinations[0], sizes[0], destinations[1], sizes[2]))
    return cudaErrorInvalidValue;
  int device, sms;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  status = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  if (status != cudaSuccess) return status;
  if (sms <= 0) return cudaErrorInvalidDevice;
  const auto blocks = unsigned(std::min<uint64_t>((sizes[0] / 4 + 255) / 256,
      uint64_t(sms) * 4));
  const auto padded = cuteafd_mxfp4_stored_width(hidden, intermediate);
  const bool compact = padded == intermediate && intermediate % 128;
  auto s = static_cast<cudaStream_t>(stream);
  pack<true, false><<<blocks, 256, 0, s>>>(sources[1], sources[0], destinations[0],
      intermediate, hidden, padded, sizes[0] / 4, compact);
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  pack<false, false><<<blocks, 256, 0, s>>>(sources[2], nullptr, destinations[1],
      hidden, intermediate, 0, sizes[2] / 4, compact);
  return cudaGetLastError();
}

extern "C" int32_t cuteafd_pack_expert_async(const uint8_t* const sources[6],
    uint8_t* const destinations[4], uint32_t intermediate, void* stream) {
  uint64_t sizes[4];
  if (!sources || !destinations || cuteafd_expert_packed_sizes(intermediate, sizes))
    return cudaErrorInvalidValue;
  const uint32_t hidden = cuteafd_expert_hidden();
  const uint64_t weight_bytes = uint64_t(intermediate) * hidden / 2;
  const uint64_t source_sizes[] = {weight_bytes, weight_bytes, weight_bytes,
    weight_bytes / 16, weight_bytes / 16, weight_bytes / 16};
  for (int i = 0; i < 6; ++i)
    if (!sources[i] || (i < 3 && reinterpret_cast<uintptr_t>(sources[i]) % 4))
      return cudaErrorInvalidValue;
  for (int i = 0; i < 4; ++i) {
    if (!destinations[i] || reinterpret_cast<uintptr_t>(destinations[i]) % 16)
      return cudaErrorInvalidValue;
    for (int j = 0; j < 6; ++j)
      if (overlaps(destinations[i], sizes[i], sources[j], source_sizes[j])) return cudaErrorInvalidValue;
    for (int j = 0; j < i; ++j)
      if (overlaps(destinations[i], sizes[i], destinations[j], sizes[j])) return cudaErrorInvalidValue;
  }
  const uint32_t padded = cuteafd_mxfp4_stored_width(hidden, intermediate);
  const bool compact = padded == intermediate && intermediate % 128;
  auto cuda_stream = static_cast<cudaStream_t>(stream);
  // First half is up (W3), second half gate (W1), stored independently.
  pack<true, false><<<256, 256, 0, cuda_stream>>>(sources[1], sources[0], destinations[0],
      intermediate, hidden, padded, sizes[0] / 4, compact);
  auto status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  pack<true, true><<<256, 256, 0, cuda_stream>>>(sources[4], sources[3], destinations[1],
      intermediate, hidden, padded, sizes[1], compact);
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  pack<false, false><<<256, 256, 0, cuda_stream>>>(sources[2], nullptr, destinations[2],
      hidden, intermediate, 0, sizes[2] / 4, compact);
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  pack<false, true><<<256, 256, 0, cuda_stream>>>(sources[5], nullptr, destinations[3],
      hidden, intermediate, 0, sizes[3], compact);
  return cudaGetLastError();
}

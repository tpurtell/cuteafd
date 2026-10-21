// GPU replay of the host-rule fixture through cuteafd_fp8_load_quant: every
// unit is read from fragmented, registered host memory (as from push ring
// slots, including fragments that split a BF16 element) and must match the
// host rule byte for byte.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

#include "../shared/cuda/fp8_load_quant.h"

extern "C" int32_t cuteafd_fp8_load_quant(const void* jobs, int32_t count, void* stream);
extern "C" int32_t cuteafd_fp8_load_quant_abi(uint64_t* job_bytes, uint64_t* segment_bytes, uint64_t* max_cols);

namespace lq = cuteafd_load_quant;

namespace {
struct Job {
  const lq::Segment* segments;
  int32_t count, rows, cols, layout, rule;
  uint8_t* q;
  int64_t q_stride;
  float* scales;
};

template <typename T>
T take(const std::vector<uint8_t>& bytes, size_t& at) {
  T value;
  std::memcpy(&value, bytes.data() + at, sizeof(T));
  at += sizeof(T);
  return value;
}

#define CHECK(call)                                                                       \
  do {                                                                                    \
    cudaError_t e = (call);                                                               \
    if (e != cudaSuccess) {                                                               \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e)); \
      return 1;                                                                           \
    }                                                                                     \
  } while (0)
}  // namespace

int main(int argc, char** argv) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
  uint64_t job_bytes = 0, segment_bytes = 0, max_cols = 0;
  cuteafd_fp8_load_quant_abi(&job_bytes, &segment_bytes, &max_cols);
  if (job_bytes != sizeof(Job) || segment_bytes != sizeof(lq::Segment)) {
    std::fprintf(stderr, "load-quant ABI differs\n");
    return 1;
  }
  const char* path = argc > 1 ? argv[1] : CUTEAFD_LOAD_QUANT_FIXTURE;
  std::ifstream file(path, std::ios::binary);
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  if (bytes.size() < 8 || std::memcmp(bytes.data(), "LQF1", 4) != 0) return 2;
  size_t at = 4;
  const uint32_t cases = take<uint32_t>(bytes, at);
  for (uint32_t c = 0; c < cases; ++c) {
    const int32_t layout = take<int32_t>(bytes, at), rule = take<int32_t>(bytes, at);
    const uint32_t rows = take<uint32_t>(bytes, at), cols = take<uint32_t>(bytes, at);
    const size_t n = size_t(rows) * cols, kb = cols / 128;
    const int32_t unit = layout == lq::kBlock ? 128 : 1;
    const size_t units = rows / unit;
    // Registered host source, as a ring mapping.
    uint8_t* host = nullptr;
    CHECK(cudaHostAlloc(&host, n * 2, cudaHostAllocPortable | cudaHostAllocMapped));
    std::memcpy(host, bytes.data() + at, n * 2);
    at += n * 2;
    const uint8_t* expected_q = bytes.data() + at;
    at += n;
    std::vector<float> expected_s(units * kb);
    std::memcpy(expected_s.data(), bytes.data() + at, expected_s.size() * 4);
    at += expected_s.size() * 4;
    uint64_t state = 0x9E3779B97F4A7C15ULL ^ c;
    auto next = [&]() { state ^= state << 13; state ^= state >> 7; state ^= state << 17; return state; };
    std::vector<std::vector<lq::Segment>> lists(units);
    for (size_t u = 0; u < units; ++u)
      for (uint64_t off = 0, total = uint64_t(unit) * cols * 2; off < total;) {
        const uint64_t len = std::min<uint64_t>(total - off, 1 + next() % 8191);
        lists[u].push_back({off, host + u * unit * cols * 2 + off, len});
        off += len;
      }
    size_t segments_total = 0;
    for (auto& l : lists) segments_total += l.size();
    lq::Segment* segments = nullptr;
    Job* jobs = nullptr;
    uint8_t* q = nullptr;
    float* s = nullptr;
    CHECK(cudaHostAlloc(&segments, segments_total * sizeof(lq::Segment), cudaHostAllocPortable | cudaHostAllocMapped));
    CHECK(cudaHostAlloc(&jobs, units * sizeof(Job), cudaHostAllocPortable | cudaHostAllocMapped));
    CHECK(cudaMalloc(&q, n));
    CHECK(cudaMalloc(&s, units * kb * 4));
    size_t cursor = 0;
    for (size_t u = 0; u < units; ++u) {
      std::copy(lists[u].begin(), lists[u].end(), segments + cursor);
      jobs[u] = Job{segments + cursor, int32_t(lists[u].size()), unit, int32_t(cols), layout, rule,
                    q + u * unit * cols, int64_t(cols), s + u * kb};
      cursor += lists[u].size();
    }
    CHECK(cudaError_t(cuteafd_fp8_load_quant(jobs, int32_t(units), nullptr)));
    CHECK(cudaDeviceSynchronize());
    std::vector<uint8_t> actual_q(n);
    std::vector<float> actual_s(units * kb);
    CHECK(cudaMemcpy(actual_q.data(), q, n, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(actual_s.data(), s, actual_s.size() * 4, cudaMemcpyDeviceToHost));
    if (std::memcmp(actual_q.data(), expected_q, n) != 0 ||
        std::memcmp(actual_s.data(), expected_s.data(), actual_s.size() * 4) != 0) {
      std::fprintf(stderr, "case %u layout %d rule %d: GPU differs from the host rule\n", c, layout, rule);
      return 1;
    }
    cudaFree(q);
    cudaFree(s);
    cudaFreeHost(jobs);
    cudaFreeHost(segments);
    cudaFreeHost(host);
  }
  std::printf("GPU fp8 load quant: %u cases byte-exact from fragmented host memory\n", cases);
  return 0;
}

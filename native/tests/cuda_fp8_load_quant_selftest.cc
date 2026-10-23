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
extern "C" int32_t cuteafd_load_widen_bf16_f32(const void* src, void* dst, int64_t n, void* stream);
extern "C" int32_t cuteafd_load_kvb_absorb(const void* kv_b, void* uk, void* uv, int64_t h0, int64_t heads,
                                           int64_t nope, int64_t v, int64_t lat, void* stream);
extern "C" int32_t cuteafd_load_kvb_split_fp8(const void* kv_b, const void* grid, void* uk, void* uk_scale, void* uv,
                                              void* uv_scale, int64_t row0, int64_t heads, int64_t nope, int64_t v,
                                              int64_t lat, void* stream);

namespace lq = cuteafd_load_quant;

namespace {
using Job = lq::Job;

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

int split_fp8();

// BF16 widening and kv_b absorption from mapped host memory against the
// header's host definitions (every BF16 bit pattern; a two-rank head split).
int transforms() {
  const int64_t n = 1 << 16;
  uint16_t* src = nullptr;
  CHECK(cudaHostAlloc(&src, n * 2, cudaHostAllocPortable | cudaHostAllocMapped));
  for (int64_t i = 0; i < n; ++i) src[i] = uint16_t(i);
  uint32_t* wide = nullptr;
  CHECK(cudaMalloc(&wide, n * 4));
  CHECK(cudaError_t(cuteafd_load_widen_bf16_f32(src, wide, n, nullptr)));
  std::vector<uint32_t> got(n);
  CHECK(cudaMemcpy(got.data(), wide, n * 4, cudaMemcpyDeviceToHost));
  for (int64_t i = 0; i < n; ++i)
    if (got[i] != lq::widen_bf16(src[i])) { std::fprintf(stderr, "widen differs at %lld\n", (long long)i); return 1; }
  const int64_t heads = 8, nope = 192, v = 128, lat = 512;
  uint16_t* kv_b = nullptr;
  CHECK(cudaHostAlloc(&kv_b, heads * (nope + v) * lat * 2, cudaHostAllocPortable | cudaHostAllocMapped));
  for (int64_t i = 0; i < heads * (nope + v) * lat; ++i) kv_b[i] = uint16_t(i * 40503u);
  const int64_t share = heads / 2;
  uint16_t *uk = nullptr, *uv = nullptr;
  CHECK(cudaMalloc(&uk, share * lat * nope * 2));
  CHECK(cudaMalloc(&uv, share * v * lat * 2));
  for (int64_t rank = 0; rank < 2; ++rank) {
    CHECK(cudaError_t(cuteafd_load_kvb_absorb(kv_b, uk, uv, rank * share, share, nope, v, lat, nullptr)));
    std::vector<uint16_t> k(share * lat * nope), w(share * v * lat);
    CHECK(cudaMemcpy(k.data(), uk, k.size() * 2, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(w.data(), uv, w.size() * 2, cudaMemcpyDeviceToHost));
    for (int64_t i = 0; i < int64_t(k.size()); ++i)
      if (k[i] != lq::absorb_uk(kv_b, rank * share, nope, v, lat, i)) { std::fprintf(stderr, "uk differs\n"); return 1; }
    for (int64_t i = 0; i < int64_t(w.size()); ++i)
      if (w[i] != lq::absorb_uv(kv_b, rank * share, nope, v, lat, i)) { std::fprintf(stderr, "uv differs\n"); return 1; }
  }
  // Independent spot check of the index math: head 5, latent 7, key dim 3 / value row 2.
  std::vector<uint16_t> k(share * lat * nope), w(share * v * lat);
  CHECK(cudaMemcpy(k.data(), uk, k.size() * 2, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(w.data(), uv, w.size() * 2, cudaMemcpyDeviceToHost));
  if (k[(1 * lat + 7) * nope + 3] != kv_b[(5 * (nope + v) + 3) * lat + 7] ||
      w[(1 * v + 2) * lat + 7] != kv_b[(5 * (nope + v) + nope + 2) * lat + 7]) {
    std::fprintf(stderr, "absorb layout differs from kv_b\n");
    return 1;
  }
  cudaFree(uk); cudaFree(uv); cudaFree(wide); cudaFreeHost(kv_b); cudaFreeHost(src);
  std::printf("GPU load widen + kv_b absorb exact\n");
  return split_fp8();
}

// FP8 kv_b split at the GLM 5.3 geometry (448 rows per head: heads span 3.5
// grid blocks) over a two-rank head split, against the header rule and an
// independent restatement of the mount loader's host transpose.
int split_fp8() {
  const int64_t heads = 8, nope = 192, v = 256, lat = 512, rows = heads * (nope + v);
  const int64_t kb = lat / 128, grid_rows = (rows + 127) / 128;
  uint8_t* kv_b = nullptr;
  float* grid = nullptr;
  CHECK(cudaHostAlloc(&kv_b, rows * lat, cudaHostAllocPortable | cudaHostAllocMapped));
  CHECK(cudaHostAlloc(&grid, grid_rows * kb * 4, cudaHostAllocPortable | cudaHostAllocMapped));
  for (int64_t i = 0; i < rows * lat; ++i) kv_b[i] = uint8_t(i * 2654435761u >> 13);
  for (int64_t i = 0; i < grid_rows * kb; ++i) grid[i] = 1.0f + float(i) * 0.25f;
  const int64_t share = heads / 2;
  uint8_t *uk = nullptr, *uv = nullptr;
  float *uk_s = nullptr, *uv_s = nullptr;
  CHECK(cudaMalloc(&uk, share * lat * nope));
  CHECK(cudaMalloc(&uv, share * v * lat));
  CHECK(cudaMalloc(&uk_s, share * lat * (nope / 64) * 4));
  CHECK(cudaMalloc(&uv_s, share * v * (lat / 64) * 4));
  for (int64_t rank = 0; rank < 2; ++rank) {
    const int64_t h0 = rank * share;
    // The staged rows begin at this rank's first head; scales still index the whole grid.
    CHECK(cudaError_t(cuteafd_load_kvb_split_fp8(kv_b + h0 * (nope + v) * lat, grid, uk, uk_s, uv, uv_s,
                                                 h0 * (nope + v), share, nope, v, lat, nullptr)));
    CHECK(cudaDeviceSynchronize());
    std::vector<uint8_t> k(share * lat * nope), w(share * v * lat);
    std::vector<float> ks(share * lat * (nope / 64)), ws(share * v * (lat / 64));
    CHECK(cudaMemcpy(k.data(), uk, k.size(), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(w.data(), uv, w.size(), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(ks.data(), uk_s, ks.size() * 4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(ws.data(), uv_s, ws.size() * 4, cudaMemcpyDeviceToHost));
    auto scale = [&](int64_t row, int64_t col) { return grid[(row / 128) * kb + col / 128]; };
    for (int64_t h = 0; h < share; ++h) {
      const int64_t base = (h0 + h) * (nope + v);
      for (int64_t c = 0; c < lat; ++c) {
        for (int64_t d = 0; d < nope; ++d)
          if (k[(h * lat + c) * nope + d] != kv_b[(base + d) * lat + c]) { std::fprintf(stderr, "fp8 uk differs\n"); return 1; }
        for (int64_t t = 0; t < nope / 64; ++t)
          if (ks[(h * lat + c) * (nope / 64) + t] != scale(base + t * 64, c)) { std::fprintf(stderr, "uk scale differs\n"); return 1; }
      }
      for (int64_t r = 0; r < v; ++r) {
        for (int64_t c = 0; c < lat; ++c)
          if (w[(h * v + r) * lat + c] != kv_b[(base + nope + r) * lat + c]) { std::fprintf(stderr, "fp8 uv differs\n"); return 1; }
        for (int64_t t = 0; t < lat / 64; ++t)
          if (ws[(h * v + r) * (lat / 64) + t] != scale(base + nope + r, t * 64)) { std::fprintf(stderr, "uv scale differs\n"); return 1; }
      }
    }
  }
  cudaFree(uk); cudaFree(uv); cudaFree(uk_s); cudaFree(uv_s); cudaFreeHost(kv_b); cudaFreeHost(grid);
  std::printf("GPU FP8 kv_b split exact\n");
  return 0;
}
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
      jobs[u] = Job{segments + cursor, int32_t(lists[u].size()), unit, int32_t(cols), layout, rule, 0, 0,
                    int64_t(cols), q + u * unit * cols, int64_t(cols), s + u * kb, 1};
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
  return transforms();
}

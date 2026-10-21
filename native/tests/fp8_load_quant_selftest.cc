// CPU self-test of the load-time FP8 quantization shared with the CUDA load
// kernels: every case of the host-rule fixture (written by the Rust test
// native_load_quant_fixture_matches_host_rule) must reproduce byte for byte.
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

#include "../shared/cuda/fp8_load_quant.h"

namespace lq = cuteafd_load_quant;

template <typename T>
static T take(const std::vector<uint8_t>& bytes, size_t& at) {
  T value;
  std::memcpy(&value, bytes.data() + at, sizeof(T));
  at += sizeof(T);
  return value;
}

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : CUTEAFD_LOAD_QUANT_FIXTURE;
  std::ifstream file(path, std::ios::binary);
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  if (bytes.size() < 8 || std::memcmp(bytes.data(), "LQF1", 4) != 0) {
    std::fprintf(stderr, "missing or invalid fixture %s\n", path);
    return 2;
  }
  size_t at = 4;
  const uint32_t cases = take<uint32_t>(bytes, at);
  size_t pow2_picked = 0, amax_kept = 0;
  for (uint32_t c = 0; c < cases; ++c) {
    const int32_t layout = take<int32_t>(bytes, at), rule = take<int32_t>(bytes, at);
    const uint32_t rows = take<uint32_t>(bytes, at), cols = take<uint32_t>(bytes, at);
    const size_t n = size_t(rows) * cols, kb = cols / 128;
    const int64_t unit = layout == lq::kBlock ? 128 : 1;
    const size_t scale_rows = rows / unit;
    std::vector<uint16_t> w(n);
    std::memcpy(w.data(), bytes.data() + at, n * 2);
    at += n * 2;
    const uint8_t* expected_q = bytes.data() + at;
    at += n;
    std::vector<float> expected_s(scale_rows * kb);
    std::memcpy(expected_s.data(), bytes.data() + at, expected_s.size() * 4);
    at += expected_s.size() * 4;
    std::vector<uint8_t> q(n);
    std::vector<float> s(scale_rows * kb);
    for (size_t u = 0; u < scale_rows; ++u)
      lq::quantize_unit(w.data() + u * unit * cols, q.data() + u * unit * cols, s.data() + u * kb, unit, cols,
                        layout, rule);
    if (std::memcmp(q.data(), expected_q, n) != 0 || std::memcmp(s.data(), expected_s.data(), s.size() * 4) != 0) {
      size_t i = 0;
      while (i < n && q[i] == expected_q[i]) ++i;
      std::fprintf(stderr, "case %u layout %d rule %d: first value mismatch at %zu\n", c, layout, rule, i);
      return 1;
    }
    // The same units read through ring-slot fragments: arbitrary lengths,
    // including odd ones that split a BF16 element across two fragments.
    uint64_t state = 0x2545F4914F6CDD1DULL ^ c;
    auto next = [&]() { state ^= state << 13; state ^= state >> 7; state ^= state << 17; return state; };
    std::vector<uint8_t> qs(n);
    std::vector<float> ss(scale_rows * kb);
    double ra[128], rp[128];
    for (size_t u = 0; u < scale_rows; ++u) {
      const uint8_t* unit_bytes = reinterpret_cast<const uint8_t*>(w.data() + u * unit * cols);
      const uint64_t total = uint64_t(unit) * cols * 2;
      std::vector<lq::Segment> segments;
      for (uint64_t at = 0; at < total;) {
        const uint64_t len = std::min<uint64_t>(total - at, 1 + next() % 4099);
        segments.push_back({at, unit_bytes + at, len});
        at += len;
      }
      lq::quantize_unit_from(lq::Segmented{segments.data(), int32_t(segments.size())}, qs.data() + u * unit * cols,
                             ss.data() + u * kb, unit, cols, layout, rule, ra, rp);
    }
    if (std::memcmp(qs.data(), expected_q, n) != 0 || std::memcmp(ss.data(), expected_s.data(), ss.size() * 4) != 0) {
      std::fprintf(stderr, "case %u layout %d rule %d: segmented source differs\n", c, layout, rule);
      return 1;
    }
    if (rule == lq::kBest)
      for (size_t i = 0; i < s.size(); ++i) {
        const uint32_t b = lq::f32_bits(s[i]);
        ((b & 0x7FFFFFu) == 0 && s[i] != 1.0f ? pow2_picked : amax_kept) += 1;
      }
  }
  if (at != bytes.size()) {
    std::fprintf(stderr, "fixture has %zu trailing bytes\n", bytes.size() - at);
    return 1;
  }
  // Both Best outcomes must be exercised for the tie-break order to be tested.
  if (pow2_picked == 0 || amax_kept == 0) {
    std::fprintf(stderr, "Best cases picked pow2 %zu, amax %zu: both branches required\n", pow2_picked, amax_kept);
    return 1;
  }
  std::printf("fp8 load quant: %u cases byte-exact (contiguous and fragmented); Best picked pow2 %zu, amax/448 %zu\n", cases, pow2_picked,
              amax_kept);
  return 0;
}

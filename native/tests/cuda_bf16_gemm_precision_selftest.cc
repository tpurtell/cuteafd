// The shared cuBLAS BF16 linear keeps FP32 reductions at decode and verify widths: every output
// is within half a BF16 ulp of its FP64 dot product plus FP32 accumulation slack. Without
// CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION on the shared handle, cuBLAS may reduce a
// split-K GEMM's partials in BF16, which at 17..128 rows over K = 4096 left outputs up to about
// 75x past this bound (an RTX 5090 measurement).
#include "cuteafd_native.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
static void check(bool value, const char* what) {
  if (!value) { std::cerr << "FAIL: " << what << "\n"; std::abort(); }
}
static void ok(cuteafd_status_t value) { check(value == CUTEAFD_STATUS_OK, "native call"); }
static float bf16_value(uint16_t bits) {
  const uint32_t word = uint32_t(bits) << 16;
  float value;
  std::memcpy(&value, &word, 4);
  return value;
}
int main() {
  constexpr size_t n = 4096, k = 4096, max_rows = 128;
  uint64_t state = 0x9e3779b97f4a7c15ull;
  auto next = [&] { state = state * 6364136223846793005ull + 1442695040888963407ull; return uint32_t(state >> 33); };
  // Weights near 2^-6 and activations near 1, both signs: sums with real cancellation.
  std::vector<uint16_t> w(n * k), x(max_rows * k);
  for (auto& v : w) { const uint32_t b = next(); v = uint16_t(((b & 1) << 15) | ((118 + (b >> 1) % 6) << 7) | ((b >> 5) & 127)); }
  for (auto& v : x) { const uint32_t b = next(); v = uint16_t(((b & 1) << 15) | ((124 + (b >> 1) % 4) << 7) | ((b >> 5) & 127)); }
  cuteafd_device_buffer_t dw{}, dx{}, dy{};
  ok(cuteafd_alloc_device_buffer(w.size() * 2, &dw, CUTEAFD_MEMORY_KIND_WORKSPACE, "selftest"));
  ok(cuteafd_alloc_device_buffer(x.size() * 2, &dx, CUTEAFD_MEMORY_KIND_WORKSPACE, "selftest"));
  ok(cuteafd_alloc_device_buffer(max_rows * n * 2, &dy, CUTEAFD_MEMORY_KIND_WORKSPACE, "selftest"));
  ok(cuteafd_copy_h2d(dw, w.data(), w.size() * 2));
  ok(cuteafd_copy_h2d(dx, x.data(), x.size() * 2));
  void* stream = nullptr;
  ok(cuteafd_cuda_stream_create(&stream));
  std::vector<uint16_t> y(max_rows * n);
  for (size_t rows : {1, 8, 17, 33, 64, 65, 128}) {
    ok(cuteafd_cuda_linear_bf16_cublas_async(static_cast<const uint16_t*>(dx.ptr), static_cast<const uint16_t*>(dw.ptr),
                                             nullptr, static_cast<uint16_t*>(dy.ptr), rows, k, n, stream));
    ok(cuteafd_cuda_stream_synchronize(stream));
    ok(cuteafd_copy_d2h(y.data(), dy, rows * n * 2, stream));
    double worst = 0.0;
    for (size_t r = 0; r < rows; ++r) {
      for (size_t c = 0; c < n; c += 61) {
        double exact = 0.0, scale = 0.0;
        for (size_t i = 0; i < k; ++i) {
          const double t = double(bf16_value(x[r * k + i])) * double(bf16_value(w[c * k + i]));
          exact += t;
          scale += std::fabs(t);
        }
        const double err = std::fabs(double(bf16_value(y[r * n + c])) - exact);
        worst = std::max(worst, err / (std::ldexp(std::fabs(exact), -8) + 1e-6 * scale + 1e-30));
      }
    }
    std::cout << "cuBLAS BF16 linear rows " << rows << ": worst error / bound " << worst << "\n";
    check(worst <= 1.0, "a BF16 linear output is past its FP64 bound (reduced-precision reductions?)");
  }
  ok(cuteafd_cuda_stream_destroy(stream));
  ok(cuteafd_free_device_buffer(&dy));
  ok(cuteafd_free_device_buffer(&dx));
  ok(cuteafd_free_device_buffer(&dw));
  std::cout << "the shared cuBLAS BF16 linear keeps FP32 reductions at 1..128 rows\n";
}

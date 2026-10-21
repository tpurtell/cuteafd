// Load-time BF16 -> FP8 E4M3 quantization, bit-for-bit the coordinator
// loaders' host rule (rust/.../glm5_flash/fp8.rs `quantize`): the same scale
// rules (amax / 448, power of two, or the better of the two by a float64
// squared error summed in the host's order) and the same layouts (128x128
// blocks, one scale per row and 128-wide K block, or one per row). Shared by
// the CUDA load kernels and a CPU self-test, so every float operation is
// spelled out without contraction: nvcc would otherwise fuse a*b - c into an
// FMA and change the error sums that pick a scale.
#pragma once
#include <stdint.h>
#include <string.h>

#if defined(__CUDACC__)
#define CUTEAFD_LQ_FN __host__ __device__ __forceinline__
#else
#define CUTEAFD_LQ_FN inline
#endif

namespace cuteafd_load_quant {

enum Layout : int32_t { kBlock = 0, kRow128 = 1, kChannel = 2 };
enum Rule : int32_t { kAmax = 0, kPow2 = 1, kBest = 2 };

CUTEAFD_LQ_FN uint32_t f32_bits(float x) { uint32_t b; memcpy(&b, &x, 4); return b; }
CUTEAFD_LQ_FN float bits_f32(uint32_t b) { float x; memcpy(&x, &b, 4); return x; }

CUTEAFD_LQ_FN float fmul(float a, float b) {
#if defined(__CUDA_ARCH__)
  return __fmul_rn(a, b);
#else
  volatile float r = a * b; return r;
#endif
}
CUTEAFD_LQ_FN float fsub(float a, float b) {
#if defined(__CUDA_ARCH__)
  return __fsub_rn(a, b);
#else
  volatile float r = a - b; return r;
#endif
}
CUTEAFD_LQ_FN float fdiv(float a, float b) {
#if defined(__CUDA_ARCH__)
  return __fdiv_rn(a, b);
#else
  volatile float r = a / b; return r;
#endif
}
CUTEAFD_LQ_FN float fadd(float a, float b) {
#if defined(__CUDA_ARCH__)
  return __fadd_rn(a, b);
#else
  volatile float r = a + b; return r;
#endif
}
CUTEAFD_LQ_FN double dmul(double a, double b) {
#if defined(__CUDA_ARCH__)
  return __dmul_rn(a, b);
#else
  volatile double r = a * b; return r;
#endif
}
CUTEAFD_LQ_FN double dadd(double a, double b) {
#if defined(__CUDA_ARCH__)
  return __dadd_rn(a, b);
#else
  volatile double r = a + b; return r;
#endif
}

CUTEAFD_LQ_FN float bf16(uint16_t v) { return bits_f32(uint32_t(v) << 16); }
CUTEAFD_LQ_FN float fabs32(float x) { return bits_f32(f32_bits(x) & 0x7FFFFFFFu); }
// Rust f32::max: the non-NaN operand when one is NaN.
CUTEAFD_LQ_FN float fmax32(float a, float b) {
  if (a != a) return b;
  if (b != b) return a;
  return a > b ? a : b;
}

// The smallest power of two >= amax / 448 (1 for zero or NaN).
CUTEAFD_LQ_FN float pow2_scale(float amax) {
  if (!(amax > 0.0f)) return 1.0f;
  const uint32_t b = f32_bits(amax);
  int x = int(b >> 23) - 135 + int((b & 0x7FFFFFu) > 0x600000u);
  x = x < -126 ? -126 : (x > 127 ? 127 : x);
  return bits_f32(uint32_t(x + 127) << 23);
}

// E4M3 (fn), round to nearest even, saturating at 448; NaN -> 0x7F.
CUTEAFD_LQ_FN uint8_t e4m3(float x) {
  const uint32_t xb = f32_bits(x);
  const uint8_t sign = (xb >> 31) ? 0x80 : 0;
  const float a = fabs32(x);
  if (a != a) return 0x7F;
  if (a >= 464.0f) return sign | 0x7E;
  if (a < 0.015625f) {
    // Multiples of 2^-9, round half to even (exact: a * 512 is exact in f32).
    const float scaled = fmul(a, 512.0f);
    uint32_t n = uint32_t(scaled);
    const float frac = fsub(scaled, float(n));
    if (frac > 0.5f || (frac == 0.5f && (n & 1u))) n += 1;
    return sign | uint8_t(n);
  }
  const uint32_t bits = f32_bits(a);
  const uint32_t rounded = bits + 0x7FFFFu + ((bits >> 20) & 1u);
  const int exponent = int((rounded >> 23) & 0xFFu) - 127;
  const uint8_t mantissa = uint8_t((rounded >> 20) & 7u);
  return sign | uint8_t(uint8_t(exponent + 7) << 3) | mantissa;
}

CUTEAFD_LQ_FN float e4m3_value(uint8_t q) {
  const float sign = (q & 0x80) ? -1.0f : 1.0f;
  const int e = (q >> 3) & 0xF;
  const float m = float(q & 7);
  // 2^k as an exact float.
  const float p = bits_f32(uint32_t((e == 0 ? -6 : e - 7) + 127) << 23);
  const float frac = fdiv(m, 8.0f);
  return fmul(sign, e == 0 ? fmul(frac, p) : fmul(fadd(1.0f, frac), p));
}

// One float64 squared error term exactly as the host computes it:
// f64::from(e4m3_value(e4m3(x / s)) * s - x).powi(2).
CUTEAFD_LQ_FN double sq_error(float x, float s) {
  const float d = fsub(fmul(e4m3_value(e4m3(fdiv(x, s))), s), x);
  const double dd = double(d);
  return dmul(dd, dd);
}

CUTEAFD_LQ_FN float amax_scale(float amax) { return amax > 0.0f ? fdiv(amax, 448.0f) : 1.0f; }

// Rows [row0, row0 + rows) of `w` (row stride `cols`) quantized as one scale
// unit: `rows` is 128 for Block, 1 for Row128/Channel. Writes `q` for those
// rows (same stride) and the unit's `kb` scales (Channel repeats one value).
// The float64 sums for Best run over the unit's rows (outer) then 128 columns
// (inner), sequentially, as the host's iterator sums do.
CUTEAFD_LQ_FN void quantize_unit(const uint16_t* w, uint8_t* q, float* scales, int64_t rows, int64_t cols,
                                 int32_t layout, int32_t rule) {
  const int64_t kb = cols / 128;
  // Per-block amax over the unit's rows.
  float row_amax = 0.0f;
  for (int64_t b = 0; b < kb; ++b) {
    float m = 0.0f;
    for (int64_t r = 0; r < rows; ++r)
      for (int64_t c = b * 128; c < (b + 1) * 128; ++c) m = fmax32(m, fabs32(bf16(w[r * cols + c])));
    scales[b] = m;
    row_amax = fmax32(row_amax, m);
  }
  if (layout == kChannel)
    for (int64_t b = 0; b < kb; ++b) scales[b] = row_amax;
  // amax per block retained for the power-of-two candidate.
  for (int64_t b = 0; b < kb; ++b) {
    const float amax = scales[b];
    float s = rule == kPow2 ? pow2_scale(amax) : amax_scale(amax);
    if (rule == kBest && (layout != kChannel || b == 0)) {
      const float sa = s, sp = pow2_scale(amax);
      if (sp != sa) {
        double ea = 0.0, ep = 0.0;
        for (int64_t r = 0; r < rows; ++r) {
          double ra = 0.0, rp = 0.0;
          for (int64_t c = b * 128; c < (b + 1) * 128; ++c) {
            const float x = bf16(w[r * cols + c]);
            ra = dadd(ra, sq_error(x, sa));
            rp = dadd(rp, sq_error(x, sp));
          }
          ea = dadd(ea, ra);
          ep = dadd(ep, rp);
        }
        if (ep < ea) s = sp;
      }
    }
    scales[b] = s;
  }
  if (layout == kChannel && rule == kBest)
    for (int64_t b = 1; b < kb; ++b) scales[b] = scales[0];
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t c = 0; c < cols; ++c) q[r * cols + c] = e4m3(fdiv(bf16(w[r * cols + c]), scales[c / 128]));
}

}  // namespace cuteafd_load_quant

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

// A unit's BF16 source: contiguous, or the ordered fragments of host ring
// slots that hold it (an element may straddle two fragments). `at(i)` is the
// unit's i-th element, row-major over the unit's rows and `cols`.
struct Contiguous {
  const uint16_t* w;
  CUTEAFD_LQ_FN float at(int64_t i) const { return bf16(w[i]); }
};
struct Segment {
  uint64_t offset;  // byte offset of this fragment within the unit's source
  const uint8_t* ptr;
  uint64_t bytes;
};
struct Segmented {
  const Segment* segments;
  int32_t count;
  CUTEAFD_LQ_FN uint8_t byte(uint64_t at) const {
    int32_t lo = 0, hi = count - 1;
    while (lo < hi) {  // last segment with offset <= at
      const int32_t mid = (lo + hi + 1) / 2;
      if (segments[mid].offset <= at) lo = mid; else hi = mid - 1;
    }
    return segments[lo].ptr[at - segments[lo].offset];
  }
  CUTEAFD_LQ_FN float at(int64_t i) const {
    const uint64_t b = uint64_t(i) * 2;
    return bf16(uint16_t(byte(b)) | uint16_t(uint16_t(byte(b + 1)) << 8));
  }
};

// Scale-block amax over the unit's rows (order-independent).
template <typename Src>
CUTEAFD_LQ_FN float block_amax(const Src& src, int64_t rows, int64_t cols, int64_t b) {
  float m = 0.0f;
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t c = b * 128; c < (b + 1) * 128; ++c) m = fmax32(m, fabs32(src.at(r * cols + c)));
  return m;
}

// One row's float64 error over block b's 128 columns, summed in column order.
template <typename Src>
CUTEAFD_LQ_FN double row_error(const Src& src, int64_t cols, int64_t r, int64_t b, float s) {
  double e = 0.0;
  for (int64_t c = b * 128; c < (b + 1) * 128; ++c) e = dadd(e, sq_error(src.at(r * cols + c), s));
  return e;
}

// Whether Best keeps amax / 448 or takes the power of two for one block,
// given each row's errors summed in row order (`ra`, `rp`: rows entries).
CUTEAFD_LQ_FN float best_scale(float sa, float sp, const double* ra, const double* rp, int64_t rows) {
  double ea = 0.0, ep = 0.0;
  for (int64_t r = 0; r < rows; ++r) { ea = dadd(ea, ra[r]); ep = dadd(ep, rp[r]); }
  return ep < ea ? sp : sa;
}

// Final scales from per-block amax `scales[kb]` (overwritten), sequentially.
// `row_errors(b, sa, sp, ra, rp)` fills each row's two error sums for block b.
template <typename Src>
CUTEAFD_LQ_FN void select_scales(const Src& src, float* scales, int64_t rows, int64_t cols, int32_t layout,
                                 int32_t rule, double* ra, double* rp) {
  const int64_t kb = cols / 128;
  if (layout == kChannel) {
    float row_amax = 0.0f;
    for (int64_t b = 0; b < kb; ++b) row_amax = fmax32(row_amax, scales[b]);
    for (int64_t b = 0; b < kb; ++b) scales[b] = row_amax;
  }
  for (int64_t b = 0; b < kb; ++b) {
    const float amax = scales[b];
    float s = rule == kPow2 ? pow2_scale(amax) : amax_scale(amax);
    if (rule == kBest && (layout != kChannel || b == 0)) {
      const float sp = pow2_scale(amax);
      if (sp != s) {
        for (int64_t r = 0; r < rows; ++r) {
          ra[r] = row_error(src, cols, r, b, s);
          rp[r] = row_error(src, cols, r, b, sp);
        }
        s = best_scale(s, sp, ra, rp, rows);
      }
    }
    scales[b] = s;
  }
  if (layout == kChannel && rule == kBest)
    for (int64_t b = 1; b < kb; ++b) scales[b] = scales[0];
}

// The whole unit sequentially (CPU reference; the CUDA kernel runs the same
// steps with the order-independent ones in parallel).
template <typename Src>
CUTEAFD_LQ_FN void quantize_unit_from(const Src& src, uint8_t* q, float* scales, int64_t rows, int64_t cols,
                                      int32_t layout, int32_t rule, double* ra, double* rp) {
  const int64_t kb = cols / 128;
  for (int64_t b = 0; b < kb; ++b) scales[b] = block_amax(src, rows, cols, b);
  select_scales(src, scales, rows, cols, layout, rule, ra, rp);
  for (int64_t i = 0; i < rows * cols; ++i) q[i] = e4m3(fdiv(src.at(i), scales[(i % cols) / 128]));
}

#if !defined(__CUDA_ARCH__)
// Rows [0, rows) of contiguous `w` as one scale unit: `rows` is 128 for Block,
// 1 for Row128/Channel. Host convenience over quantize_unit_from.
inline void quantize_unit(const uint16_t* w, uint8_t* q, float* scales, int64_t rows, int64_t cols,
                          int32_t layout, int32_t rule) {
  double ra[128], rp[128];
  quantize_unit_from(Contiguous{w}, q, scales, rows, cols, layout, rule, ra, rp);
}
#endif

}  // namespace cuteafd_load_quant

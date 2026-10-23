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

// A unit that is a column window of wider source rows: unit element (r, c)
// is source element r * pitch + col0 + c (all in elements).
template <typename Src>
struct Strided {
  Src src;
  int64_t cols, col0, pitch;
  CUTEAFD_LQ_FN float at(int64_t i) const { return src.at((i / cols) * pitch + col0 + i % cols); }
};

// One unit of a load: `rows` (128 Block, 1 Row128/Channel) by `cols` source
// elements at column `col0` of rows `pitch` elements wide, read from host
// fragments; E4M3 rows at `q` (row stride `q_stride`) and its `cols / 128`
// scales at `scales` with stride `scale_stride` (1 row-major; the row count for
// K-block-major). Shared by the CUDA kernel, the CPU fake DSO and the self-tests.
struct Job {
  const Segment* segments;
  int32_t count;
  int32_t rows;
  int32_t cols;
  int32_t layout;
  int32_t rule;
  int32_t pad;
  int64_t col0;
  int64_t pitch;
  uint8_t* q;
  int64_t q_stride;
  float* scales;
  int64_t scale_stride;
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

// BF16 -> F32 widening (exact).
CUTEAFD_LQ_FN uint32_t widen_bf16(uint16_t v) { return uint32_t(v) << 16; }

// F32 -> BF16, round to nearest even; NaN keeps its sign and top payload bits
// and becomes quiet (cuteafd_core::OutputPrecision::Bf16::round).
CUTEAFD_LQ_FN uint16_t narrow_f32(uint32_t bits) {
  if ((bits & 0x7fffffffu) > 0x7f800000u) return uint16_t(((bits & 0xffff0000u) | 0x00400000u) >> 16);
  return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

// Two BF16 values promoted to F32 and multiplied once, round to nearest
// (the Kimi residual score `norm * proj`, never a BF16 product).
CUTEAFD_LQ_FN uint32_t product_bf16_f32(uint16_t a, uint16_t b) {
  return f32_bits(fmul(bits_f32(widen_bf16(a)), bits_f32(widen_bf16(b))));
}

// EXL3 trellis tiles (16x16, `tile_bytes` each) of a staged window: `rows`
// K16 rows from local row `k0`, `tiles` N16 tiles each, in checkpoint order
// [K16][N16]. Staged tile (r, c) lands at this tile index of the native
// shard: side 0 (output) [N128][K16][8], side 1 (input) [K128][N128][8][8].
CUTEAFD_LQ_FN int64_t trellis_tile(int32_t side, int64_t kr, int64_t c, int64_t tiles, int64_t k16) {
  if (side == 0) return ((c / 8) * k16 + kr) * 8 + c % 8;
  return ((kr / 8) * (tiles / 8) + c / 8) * 64 + (kr % 8) * 8 + c % 8;
}

// MLA kv_b [heads * (nope + v), lat] BF16 split per head into
// uk [heads][lat][nope] (key rows transposed) and uv [heads][v][lat] (value
// rows copied), for heads [h0, h0 + count): element `i` of that output range.
CUTEAFD_LQ_FN uint16_t absorb_uk(const uint16_t* kv_b, int64_t h0, int64_t nope, int64_t v, int64_t lat, int64_t i) {
  const int64_t h = h0 + i / (lat * nope), c = (i / nope) % lat, d = i % nope;
  return kv_b[(h * (nope + v) + d) * lat + c];
}
CUTEAFD_LQ_FN uint16_t absorb_uv(const uint16_t* kv_b, int64_t h0, int64_t nope, int64_t v, int64_t lat, int64_t i) {
  const int64_t h = h0 + i / (v * lat), r = (i / lat) % v, c = i % lat;
  return kv_b[(h * (nope + v) + nope + r) * lat + c];
}

// FP8 kv_b rows of `heads` heads (E4M3 bytes, the first at kv_b row `row0`)
// with the whole tensor's FP32 128x128 grid [ceil(rows / 128), lat / 128] split per head into the GLM FP8 programs'
// operands (the mount loader's host rule): uk [heads][lat][nope] bytes with one
// scale per (lat row, 64-wide nope tile), uv [heads][v][lat] bytes with one scale
// per (v row, 64-wide lat tile). Values move unchanged; scales are grid entries.
CUTEAFD_LQ_FN uint8_t absorb_uk_fp8(const uint8_t* kv_b, int64_t h0, int64_t nope, int64_t v, int64_t lat, int64_t i) {
  const int64_t h = h0 + i / (lat * nope), c = (i / nope) % lat, d = i % nope;
  return kv_b[(h * (nope + v) + d) * lat + c];
}
CUTEAFD_LQ_FN uint8_t absorb_uv_fp8(const uint8_t* kv_b, int64_t h0, int64_t nope, int64_t v, int64_t lat, int64_t i) {
  const int64_t h = h0 + i / (v * lat), r = (i / lat) % v, c = i % lat;
  return kv_b[(h * (nope + v) + nope + r) * lat + c];
}
// Scales index the grid by absolute kv_b row: `row0` is the first staged row.
CUTEAFD_LQ_FN float absorb_uk_scale(const float* grid, int64_t row0, int64_t nope, int64_t v, int64_t lat, int64_t i) {
  const int64_t tiles = nope / 64, h = i / (lat * tiles), c = (i / tiles) % lat, t = i % tiles;
  return grid[((row0 + h * (nope + v) + t * 64) / 128) * (lat / 128) + c / 128];
}
CUTEAFD_LQ_FN float absorb_uv_scale(const float* grid, int64_t row0, int64_t nope, int64_t v, int64_t lat, int64_t i) {
  const int64_t tiles = lat / 64, h = i / (v * tiles), r = (i / tiles) % v, t = i % tiles;
  return grid[((row0 + h * (nope + v) + nope + r) / 128) * (lat / 128) + (t * 64) / 128];
}

#if !defined(__CUDA_ARCH__)
// One job sequentially (the CPU fake DSO's load kernel and the self-test oracle).
inline void run_job(const Job& job) {
  const Strided<Segmented> src{Segmented{job.segments, job.count}, job.cols, job.col0, job.pitch};
  const int64_t rows = job.rows, cols = job.cols, kb = cols / 128;
  float scales[512];
  double ra[128], rp[128];
  for (int64_t b = 0; b < kb; ++b) scales[b] = block_amax(src, rows, cols, b);
  select_scales(src, scales, rows, cols, job.layout, job.rule, ra, rp);
  for (int64_t b = 0; b < kb; ++b) job.scales[b * job.scale_stride] = scales[b];
  for (int64_t i = 0; i < rows * cols; ++i)
    job.q[(i / cols) * job.q_stride + i % cols] = e4m3(fdiv(src.at(i), scales[(i % cols) / 128]));
}

// Rows [0, rows) of contiguous `w` as one scale unit: `rows` is 128 for Block,
// 1 for Row128/Channel. Host convenience over quantize_unit_from.
inline void quantize_unit(const uint16_t* w, uint8_t* q, float* scales, int64_t rows, int64_t cols,
                          int32_t layout, int32_t rule) {
  double ra[128], rp[128];
  quantize_unit_from(Contiguous{w}, q, scales, rows, cols, layout, rule, ra, rp);
}
#endif

}  // namespace cuteafd_load_quant

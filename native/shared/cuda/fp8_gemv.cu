// W8A16 linear for skinny row counts (drafters, FP8 LM heads): E4M3 weights
// with one FP32 scale per output row and 128-wide K block ([n, k / 128], the
// fp8_row_quant layout), BF16 activations, FP32 accumulation on tensor cores.
//
// Weights are packed at load into mma.m16n8k16 A-fragment order: tiles of
// 16 rows x 32 k, each 32 lanes x 16 bytes, so a warp streams its tile row
// with one coalesced 16-byte load per lane and converts E4M3 pairs straight
// to the f16x2 registers the MMA takes (exact: every E4M3 value is an f16).
// Activations take the MMA's B side (swap AB: out^T = W x^T): each row is
// scaled by a power of two into f16 range (exact within it) and packed in
// B-fragment order once per call. Rows beyond 64 run in chunks of 64.
// Split-K (a deterministic second pass) keeps narrow outputs busy.
//
// cuteafd_fp8_linear (end of file) runs the same packed weights in two more
// modes, with kernels and scratch of their own; the W8A16 entry points keep
// theirs. `wide` takes passes of 128 rows with the same bits as passes of 64
// (every output element's MMAs and sums are its own): one pass over the
// weights and one launch chain where passes of 64 take two. `w8a8` also takes
// E4M3 activations past 8 rows (per row and 128-wide K block, amax / 448, the
// dynamic scheme of FP8 checkpoints) on mma.m16n8k32: half the MMAs of the
// f16 path at twice the rate. Its B fragments follow the packed weights' K
// order, so the weights are read as packed: a lane's 16 bytes of a 16 x 32
// tile hold k c, c + 1, c + 8, c + 9 (+16) of rows g and g + 8, which byte
// permutes turn into the k32 A fragment of the same K order. Past 64 rows that
// W8A8 path is one 2D CTA grid over the M and N tiles with a single K split,
// writing BF16 (or FP32) directly; 64 rows and below keep the split-K passes.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Host-side launch counter for the wide single-pass path (the harness reads it).
static int g_fp8_linear_launches = 0;

namespace {

constexpr int kWarps = 4;
constexpr int kMaxRows = 64;

// Smallest power of two >= x (x > 0). Keep libdevice's edge behavior for
// subnormals and the top exponent, without its calls on ordinary norm rows.
__device__ __forceinline__ float pow2_ceil(float x) {
  const uint32_t bits = __float_as_uint(x), exponent = bits & 0x7f800000u;
  if (exponent != 0 && exponent < 0x7f000000u)
    return __uint_as_float(exponent + ((bits & 0x007fffffu) != 0 ? 0x00800000u : 0));
  int e;
  const float m = frexpf(x, &e);  // x = m * 2^e, m in [0.5, 1)
  return ldexpf(1.0f, m == 0.5f ? e - 1 : e);
}

// The smallest power of two >= amax / 448 (Hugh Madden's glm53f-afd
// pow2_scale, bit for bit): BF16 weights with at most E4M3's 3 mantissa bits
// then quantize exactly.
__device__ __forceinline__ float pow2_scale(float amax) {
  const uint32_t b = __float_as_uint(amax);
  int x = int(b >> 23) - 135 + int((b & 0x7FFFFFu) > 0x600000u);
  x = x < -126 ? -126 : (x > 127 ? 127 : x);
  return __uint_as_float(uint32_t(x + 127) << 23);
}

__device__ __forceinline__ uint8_t e4m3(float x) {
  return __nv_fp8_e4m3(x).__x;
}

__device__ __forceinline__ float quant_error(float x, float s) {
  const float d = float(__nv_fp8_e4m3(x / s)) * s - x;
  return d * d;
}

// One warp per (16-row tile, 128-wide K block): per-row scales, then the
// block's four 16 x 32 tiles in A-fragment order.
// `rule`: 0 amax / 448, 1 pow2_scale, 2 whichever leaves the smaller squared
// error over the row's 128 values (1 for an all-zero block).
__global__ void pack_kernel(const __nv_bfloat16* __restrict__ w, uint8_t* __restrict__ packed,
                            float* __restrict__ scale, int n, int k, int rule) {
  const int kb = blockIdx.x, rt = blockIdx.y, lane = threadIdx.x;
  const int kbs = k / 128;
  __shared__ float s_scale[16];
  for (int r = 0; r < 16; ++r) {
    const __nv_bfloat16* row = w + size_t(rt * 16 + r) * k + kb * 128 + lane * 4;
    float x[4], amax = 0.0f;
    for (int j = 0; j < 4; ++j) {
      x[j] = __bfloat162float(row[j]);
      amax = fmaxf(amax, fabsf(x[j]));
    }
    for (int offset = 16; offset; offset >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
    const float sa = amax > 0.0f ? amax / 448.0f : 1.0f, sp = amax > 0.0f ? pow2_scale(amax) : 1.0f;
    float s = rule == 1 ? sp : sa;
    if (rule == 2) {
      float ea = 0.0f, ep = 0.0f;
      for (int j = 0; j < 4; ++j) {
        ea += quant_error(x[j], sa);
        ep += quant_error(x[j], sp);
      }
      for (int offset = 16; offset; offset >>= 1) {
        ea += __shfl_xor_sync(0xffffffffu, ea, offset);
        ep += __shfl_xor_sync(0xffffffffu, ep, offset);
      }
      s = ep < ea ? sp : sa;
    }
    if (lane == 0) {
      s_scale[r] = s;
      scale[size_t(rt * 16 + r) * kbs + kb] = s;
    }
  }
  __syncwarp();
  const int g = lane / 4, c = (lane % 4) * 2;
  const float s_lo = s_scale[g], s_hi = s_scale[g + 8];
  const __nv_bfloat16* lo = w + size_t(rt * 16 + g) * k;
  const __nv_bfloat16* hi = w + size_t(rt * 16 + g + 8) * k;
  auto q = [&](const __nv_bfloat16* row, float s, int col) { return e4m3(__bfloat162float(row[col]) / s); };
  for (int kt = 0; kt < 4; ++kt) {
    uint8_t out[16];
    for (int step = 0; step < 2; ++step) {
      const int k0 = kb * 128 + kt * 32 + step * 16 + c;
      uint8_t* o = out + step * 8;
      o[0] = q(lo, s_lo, k0);
      o[1] = q(lo, s_lo, k0 + 1);
      o[2] = q(hi, s_hi, k0);
      o[3] = q(hi, s_hi, k0 + 1);
      o[4] = q(lo, s_lo, k0 + 8);
      o[5] = q(lo, s_lo, k0 + 9);
      o[6] = q(hi, s_hi, k0 + 8);
      o[7] = q(hi, s_hi, k0 + 9);
    }
    uint4 v;
    memcpy(&v, out, 16);
    reinterpret_cast<uint4*>(packed)[(size_t(rt) * (k / 32) + kb * 4 + kt) * 32 + lane] = v;
  }
}

// Split each row's maximum across K; the packing launch completes the max.
// A row-wide CTA serializes long K loads and leaves most SMs idle at C1.
constexpr int kScaleCols = 256;
__global__ void row_max_kernel(const __nv_bfloat16* __restrict__ x, float* __restrict__ maxima,
                               int rows, int k, int splits) {
  const int m = blockIdx.y, split = blockIdx.x;
  const int col = split * kScaleCols + threadIdx.x * 2;
  float amax = 0.0f;
  if (m < rows && col < k) {
    const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(x + size_t(m) * k + col);
    amax = fmaxf(fabsf(__low2float(v)), fabsf(__high2float(v)));
  }
  for (int offset = 16; offset; offset >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
  __shared__ float warp_max[kScaleCols / 64];
  if ((threadIdx.x & 31) == 0) warp_max[threadIdx.x >> 5] = amax;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 0; i < kScaleCols / 64; ++i) amax = fmaxf(amax, warp_max[i]);
    maxima[size_t(m) * splits + split] = amax;
  }
}

int row_max_splits(int k) { return (k + kScaleCols - 1) / kScaleCols; }
size_t row_max_bytes(int k, int stride) { return size_t(row_max_splits(k)) * stride * sizeof(float); }

// Activations [rows, k] -> f16 B fragments: per 32-wide k tile and 8-row
// group, 32 lanes x 16 bytes (two k16 steps of b0b1, b2b3).
__global__ void pack_x_kernel(const __nv_bfloat16* __restrict__ x, const float* __restrict__ maxima,
                              float* __restrict__ sx, uint4* __restrict__ xp, int rows, int k, int groups,
                              int stride, int splits) {
  const int kt = blockIdx.x, group = blockIdx.y, lane = threadIdx.x;
  const int m = group * 8 + lane / 4, c = (lane % 4) * 2;
  float amax = 0.0f;
  for (int i = lane % 4; i < splits; i += 4) amax = fmaxf(amax, maxima[size_t(m) * splits + i]);
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
  // Same power-of-two scale as the row-wide reduction; max reordering is exact.
  const float s = amax > 0.0f ? 16384.0f / pow2_ceil(amax) : 1.0f;
  if (kt == 0 && lane % 4 == 0) {
    sx[m] = s;
    sx[stride + m] = 1.0f / s;
  }
  auto h = [&](int col) -> __half {
    return m < rows ? __float2half_rn(__bfloat162float(x[size_t(m) * k + col]) * s) : __float2half_rn(0.0f);
  };
  __half v[8];
  for (int step = 0; step < 2; ++step) {
    const int k0 = kt * 32 + step * 16 + c;
    v[step * 4 + 0] = h(k0);
    v[step * 4 + 1] = h(k0 + 1);
    v[step * 4 + 2] = h(k0 + 8);
    v[step * 4 + 3] = h(k0 + 9);
  }
  uint4 out;
  memcpy(&out, v, 16);
  xp[(size_t(kt) * groups + group) * 32 + lane] = out;
}

__device__ __forceinline__ uint32_t fp8x2_to_f16x2(uint32_t pair) {
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(pair), __NV_E4M3);
  return uint32_t(h.x) | (uint32_t(h.y) << 16);
}

__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 load_stream(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

// One warp per (16-row tile, K split). GROUPS 8-row activation groups.
template <int GROUPS>
__global__ void __launch_bounds__(kWarps * 32) gemv_kernel(
    const uint4* __restrict__ wp, const float* __restrict__ scale, const uint4* __restrict__ xp,
    const float* __restrict__ sx, void* __restrict__ out, float* __restrict__ partial, int out_f32, int rows, int n,
    int k, int kb_per_split, int splits) {
  const int lane = threadIdx.x & 31;
  const int warp = blockIdx.x * kWarps + threadIdx.x / 32;
  const int tiles = n / 16;
  if (warp >= tiles * splits) return;
  const int rt = warp % tiles, split = warp / tiles;
  const int kbs = k / 128;
  const int kb0 = split * kb_per_split, kb1 = min(kbs, kb0 + kb_per_split);
  const int g = lane / 4, c = (lane % 4) * 2;
  float acc[GROUPS][4];
#pragma unroll
  for (int i = 0; i < GROUPS; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;
  const uint4* wrow = wp + size_t(rt) * (k / 32) * 32 + lane;
  const float* s_lo = scale + size_t(rt * 16 + g) * kbs;
  const float* s_hi = scale + size_t(rt * 16 + g + 8) * kbs;
  uint4 a[4];
  if (kb0 < kb1) {
#pragma unroll
    for (int kt = 0; kt < 4; ++kt) a[kt] = load_stream(wrow + size_t(kb0 * 4 + kt) * 32);
  }
  for (int kb = kb0; kb < kb1; ++kb) {
    uint4 next[4];
    if (kb + 1 < kb1) {
#pragma unroll
      for (int kt = 0; kt < 4; ++kt) next[kt] = load_stream(wrow + size_t((kb + 1) * 4 + kt) * 32);
    }
    float blk[GROUPS][4];
#pragma unroll
    for (int i = 0; i < GROUPS; ++i) blk[i][0] = blk[i][1] = blk[i][2] = blk[i][3] = 0.0f;
#pragma unroll
    for (int kt = 0; kt < 4; ++kt) {
      const uint32_t a0[4] = {fp8x2_to_f16x2(a[kt].x & 0xffff), fp8x2_to_f16x2(a[kt].x >> 16),
                              fp8x2_to_f16x2(a[kt].y & 0xffff), fp8x2_to_f16x2(a[kt].y >> 16)};
      const uint32_t a1[4] = {fp8x2_to_f16x2(a[kt].z & 0xffff), fp8x2_to_f16x2(a[kt].z >> 16),
                              fp8x2_to_f16x2(a[kt].w & 0xffff), fp8x2_to_f16x2(a[kt].w >> 16)};
      const uint4* b = xp + (size_t(kb * 4 + kt) * GROUPS) * 32 + lane;
#pragma unroll
      for (int i = 0; i < GROUPS; ++i) {
        const uint4 bv = __ldg(b + i * 32);
        mma(blk[i], a0, bv.x, bv.y);
        mma(blk[i], a1, bv.z, bv.w);
      }
    }
    const float lo = __ldg(s_lo + kb), hi = __ldg(s_hi + kb);
#pragma unroll
    for (int i = 0; i < GROUPS; ++i) {
      acc[i][0] += blk[i][0] * lo;
      acc[i][1] += blk[i][1] * lo;
      acc[i][2] += blk[i][2] * hi;
      acc[i][3] += blk[i][3] * hi;
    }
    if (kb + 1 < kb1) {
#pragma unroll
      for (int kt = 0; kt < 4; ++kt) a[kt] = next[kt];
    }
  }
  // C fragment: c0, c1 weight row g, activation rows c, c + 1; c2, c3 weight row g + 8.
  const int row_lo = rt * 16 + g, row_hi = row_lo + 8;
#pragma unroll
  for (int i = 0; i < GROUPS; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int m = i * 8 + c + (j & 1);
      const int col = j < 2 ? row_lo : row_hi;
      if (m >= rows) continue;
      if (splits > 1) {
        partial[(size_t(split) * rows + m) * n + col] = acc[i][j];
      } else {
        const float v = acc[i][j] * sx[kMaxRows + m];
        if (out_f32) {
          static_cast<float*>(out)[size_t(m) * n + col] = v;
        } else {
          static_cast<__nv_bfloat16*>(out)[size_t(m) * n + col] = __float2bfloat16(v);
        }
      }
    }
  }
}

__global__ void reduce_kernel(const float* __restrict__ partial, const float* __restrict__ sx, void* __restrict__ out,
                              int out_f32, int rows, int n, int splits) {
  const size_t total = size_t(rows) * n;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += size_t(gridDim.x) * blockDim.x) {
    float v = 0.0f;
    for (int s = 0; s < splits; ++s) v += partial[size_t(s) * total + i];
    v *= sx[kMaxRows + int(i / n)];
    if (out_f32) {
      static_cast<float*>(out)[i] = v;
    } else {
      static_cast<__nv_bfloat16*>(out)[i] = __float2bfloat16(v);
    }
  }
}

int sm_count() {
  static int count = 0;
  if (count == 0) {
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device);
    if (count <= 0) count = 128;
  }
  return count;
}

void split_plan(int n, int k, int* kb_per_split, int* splits) {
  const int tiles = n / 16, kbs = k / 128;
  const int wanted = (sm_count() * 16 + tiles - 1) / tiles;
  int s = wanted < 1 ? 1 : (wanted > kbs ? kbs : wanted);
  *kb_per_split = (kbs + s - 1) / s;
  *splits = (kbs + *kb_per_split - 1) / *kb_per_split;
}

size_t align256(size_t bytes) { return (bytes + 255) / 256 * 256; }

// --- cuteafd_fp8_linear's wide and W8A8 passes ------------------------------

// Rows per pass of the wide and W8A8 modes.
constexpr int kWideRows = 128;
// Most rows the W8A8 mode keeps on the f16 path (one draft block).
constexpr int kW8a16Rows = 8;

// Four floats to E4M3 (round to nearest, saturating), first in the low byte.
__device__ __forceinline__ uint32_t e4m3x4(float a, float b, float c, float d) {
  const uint32_t lo = __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
  const uint32_t hi = __nv_cvt_float2_to_fp8x2(make_float2(c, d), __NV_SATFINITE, __NV_E4M3);
  return lo | (hi << 16);
}

// W8A8 activations: per row and 128-wide K block, E4M3 at amax / 448 (1 for a
// zero block) in mma.m16n8k32 B-fragment order under the packed weights' K
// order: per 32-wide k tile and 8-row group, 32 lanes x 8 bytes, lane l (row
// group * 8 + l / 4, c = 2 (l % 4)) holding k c, c + 1, c + 8, c + 9, then the
// same + 16. Scales go to sxa [k / 128][stride] (padding rows: 1). One warp per
// (128-wide K block, 8-row group); the four lanes of a row hold its 128 values.
__global__ void pack_x_e4m3_kernel(const __nv_bfloat16* __restrict__ x, uint2* __restrict__ xq,
                                   float* __restrict__ sxa, int rows, int k, int groups, int stride) {
  const int kb = blockIdx.x, group = blockIdx.y, lane = threadIdx.x;
  const int m = group * 8 + lane / 4, c = (lane % 4) * 2;
  float v[4][8];
  float amax = 0.0f;
#pragma unroll
  for (int kt = 0; kt < 4; ++kt) {
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      float lo = 0.0f, hi = 0.0f;
      if (m < rows) {
        const __nv_bfloat162 pair =
            *reinterpret_cast<const __nv_bfloat162*>(x + size_t(m) * k + kb * 128 + kt * 32 + c + 8 * p);
        lo = __low2float(pair);
        hi = __high2float(pair);
      }
      v[kt][2 * p] = lo;
      v[kt][2 * p + 1] = hi;
      amax = fmaxf(amax, fmaxf(fabsf(lo), fabsf(hi)));
    }
  }
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
  const float s = amax > 0.0f ? amax / 448.0f : 1.0f;
  if (lane % 4 == 0) sxa[size_t(kb) * stride + m] = s;
#pragma unroll
  for (int kt = 0; kt < 4; ++kt) {
    uint2 q;
    q.x = e4m3x4(v[kt][0] / s, v[kt][1] / s, v[kt][2] / s, v[kt][3] / s);
    q.y = e4m3x4(v[kt][4] / s, v[kt][5] / s, v[kt][6] / s, v[kt][7] / s);
    xq[(size_t(kb * 4 + kt) * groups + group) * 32 + lane] = q;
  }
}

__device__ __forceinline__ void mma_e4m3(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// gemv_kernel with E4M3 activations: one m16n8k32 per k tile and group, the
// A fragment (row g: k c, c + 1, c + 8, c + 9 | +16; row g + 8 the same)
// permuted from the packed bytes. Each 128-wide K block's sums take the weight
// row's and the activation row's scales before they join the FP32 total, so
// the outputs and split partials need no row inverse.
template <int GROUPS>
__global__ void __launch_bounds__(kWarps * 32) gemv_w8a8_kernel(
    const uint4* __restrict__ wp, const float* __restrict__ scale, const uint2* __restrict__ xq,
    const float* __restrict__ sxa, void* __restrict__ out, float* __restrict__ partial, int out_f32, int rows, int n,
    int k, int kb_per_split, int splits, int stride) {
  const int lane = threadIdx.x & 31;
  const int warp = blockIdx.x * kWarps + threadIdx.x / 32;
  const int tiles = n / 16;
  if (warp >= tiles * splits) return;
  const int rt = warp % tiles, split = warp / tiles;
  const int kbs = k / 128;
  const int kb0 = split * kb_per_split, kb1 = min(kbs, kb0 + kb_per_split);
  const int g = lane / 4, c = (lane % 4) * 2;
  float acc[GROUPS][4];
#pragma unroll
  for (int i = 0; i < GROUPS; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;
  const uint4* wrow = wp + size_t(rt) * (k / 32) * 32 + lane;
  const float* s_lo = scale + size_t(rt * 16 + g) * kbs;
  const float* s_hi = scale + size_t(rt * 16 + g + 8) * kbs;
  uint4 a[4];
  if (kb0 < kb1) {
#pragma unroll
    for (int kt = 0; kt < 4; ++kt) a[kt] = load_stream(wrow + size_t(kb0 * 4 + kt) * 32);
  }
  for (int kb = kb0; kb < kb1; ++kb) {
    uint4 next[4];
    if (kb + 1 < kb1) {
#pragma unroll
      for (int kt = 0; kt < 4; ++kt) next[kt] = load_stream(wrow + size_t((kb + 1) * 4 + kt) * 32);
    }
    float blk[GROUPS][4];
#pragma unroll
    for (int i = 0; i < GROUPS; ++i) blk[i][0] = blk[i][1] = blk[i][2] = blk[i][3] = 0.0f;
#pragma unroll
    for (int kt = 0; kt < 4; ++kt) {
      const uint32_t af[4] = {__byte_perm(a[kt].x, a[kt].y, 0x5410), __byte_perm(a[kt].x, a[kt].y, 0x7632),
                              __byte_perm(a[kt].z, a[kt].w, 0x5410), __byte_perm(a[kt].z, a[kt].w, 0x7632)};
      const uint2* b = xq + (size_t(kb * 4 + kt) * GROUPS) * 32 + lane;
#pragma unroll
      for (int i = 0; i < GROUPS; ++i) {
        const uint2 bv = __ldg(b + i * 32);
        mma_e4m3(blk[i], af, bv.x, bv.y);
      }
    }
    const float lo = __ldg(s_lo + kb), hi = __ldg(s_hi + kb);
    const float* sk = sxa + size_t(kb) * stride + c;
#pragma unroll
    for (int i = 0; i < GROUPS; ++i) {
      const float2 sa = __ldg(reinterpret_cast<const float2*>(sk + i * 8));
      acc[i][0] += blk[i][0] * (lo * sa.x);
      acc[i][1] += blk[i][1] * (lo * sa.y);
      acc[i][2] += blk[i][2] * (hi * sa.x);
      acc[i][3] += blk[i][3] * (hi * sa.y);
    }
    if (kb + 1 < kb1) {
#pragma unroll
      for (int kt = 0; kt < 4; ++kt) a[kt] = next[kt];
    }
  }
  const int row_lo = rt * 16 + g, row_hi = row_lo + 8;
#pragma unroll
  for (int i = 0; i < GROUPS; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int m = i * 8 + c + (j & 1);
      const int col = j < 2 ? row_lo : row_hi;
      if (m >= rows) continue;
      if (splits > 1) {
        partial[(size_t(split) * rows + m) * n + col] = acc[i][j];
      } else if (out_f32) {
        static_cast<float*>(out)[size_t(m) * n + col] = acc[i][j];
      } else {
        static_cast<__nv_bfloat16*>(out)[size_t(m) * n + col] = __float2bfloat16(acc[i][j]);
      }
    }
  }
}

// gemv_w8a8_kernel<16> over every M tile at once. One CTA owns kWarps N-tiles
// of 16 rows and one 128-row M tile (GROUPS groups of 8). Grid (m_tiles,
// n_ctas) so blockIdx.x walks M first: the CTAs that share an N tile, and
// therefore the same streamed weights, are neighbors. K is not split; the
// FP32 tile is the output.
template <int GROUPS>
__global__ void __launch_bounds__(kWarps * 32) gemv_w8a8_wide_kernel(
    const uint4* __restrict__ wp, const float* __restrict__ scale, const uint2* __restrict__ xq,
    const float* __restrict__ sxa, void* __restrict__ out, int out_f32, int rows, int n, int k, int stride,
    int groups) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int rt = int(blockIdx.y) * kWarps + warp;
  const int tiles = n / 16;
  if (rt >= tiles) return;
  const int m_tile = int(blockIdx.x);
  const int row_base = m_tile * (GROUPS * 8);
  const int group_base = m_tile * GROUPS;
  const int kbs = k / 128;
  const int g = lane / 4, c = (lane % 4) * 2;
  float acc[GROUPS][4];
#pragma unroll
  for (int i = 0; i < GROUPS; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;
  const uint4* wrow = wp + size_t(rt) * (k / 32) * 32 + lane;
  const float* s_lo = scale + size_t(rt * 16 + g) * kbs;
  const float* s_hi = scale + size_t(rt * 16 + g + 8) * kbs;
  uint4 a[4];
  if (kbs > 0) {
#pragma unroll
    for (int kt = 0; kt < 4; ++kt) a[kt] = load_stream(wrow + size_t(kt) * 32);
  }
  for (int kb = 0; kb < kbs; ++kb) {
    uint4 next[4];
    if (kb + 1 < kbs) {
#pragma unroll
      for (int kt = 0; kt < 4; ++kt) next[kt] = load_stream(wrow + size_t((kb + 1) * 4 + kt) * 32);
    }
    float blk[GROUPS][4];
#pragma unroll
    for (int i = 0; i < GROUPS; ++i) blk[i][0] = blk[i][1] = blk[i][2] = blk[i][3] = 0.0f;
#pragma unroll
    for (int kt = 0; kt < 4; ++kt) {
      const uint32_t af[4] = {__byte_perm(a[kt].x, a[kt].y, 0x5410), __byte_perm(a[kt].x, a[kt].y, 0x7632),
                              __byte_perm(a[kt].z, a[kt].w, 0x5410), __byte_perm(a[kt].z, a[kt].w, 0x7632)};
      const uint2* b = xq + (size_t(kb * 4 + kt) * groups + group_base) * 32 + lane;
#pragma unroll
      for (int i = 0; i < GROUPS; ++i) {
        const uint2 bv = __ldg(b + i * 32);
        mma_e4m3(blk[i], af, bv.x, bv.y);
      }
    }
    const float lo = __ldg(s_lo + kb), hi = __ldg(s_hi + kb);
    const float* sk = sxa + size_t(kb) * stride + row_base + c;
#pragma unroll
    for (int i = 0; i < GROUPS; ++i) {
      const float2 sa = __ldg(reinterpret_cast<const float2*>(sk + i * 8));
      acc[i][0] += blk[i][0] * (lo * sa.x);
      acc[i][1] += blk[i][1] * (lo * sa.y);
      acc[i][2] += blk[i][2] * (hi * sa.x);
      acc[i][3] += blk[i][3] * (hi * sa.y);
    }
    if (kb + 1 < kbs) {
#pragma unroll
      for (int kt = 0; kt < 4; ++kt) a[kt] = next[kt];
    }
  }
  const int row_lo = rt * 16 + g, row_hi = row_lo + 8;
#pragma unroll
  for (int i = 0; i < GROUPS; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int m = row_base + i * 8 + c + (j & 1);
      const int col = j < 2 ? row_lo : row_hi;
      if (m >= rows) continue;
      if (out_f32) {
        static_cast<float*>(out)[size_t(m) * n + col] = acc[i][j];
      } else {
        static_cast<__nv_bfloat16*>(out)[size_t(m) * n + col] = __float2bfloat16(acc[i][j]);
      }
    }
  }
}

// reduce_kernel for W8A8 partials, which carry their scales: the K splits'
// sum in split order.
__global__ void reduce_w8a8_kernel(const float* __restrict__ partial, void* __restrict__ out, int out_f32, int rows,
                                   int n, int splits) {
  const size_t total = size_t(rows) * n;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += size_t(gridDim.x) * blockDim.x) {
    float v = 0.0f;
    for (int s = 0; s < splits; ++s) v += partial[size_t(s) * total + i];
    if (out_f32) {
      static_cast<float*>(out)[i] = v;
    } else {
      static_cast<__nv_bfloat16*>(out)[i] = __float2bfloat16(v);
    }
  }
}

// Scratch of a wide or W8A8 call: the row scales and their inverses
// (2 x kWideRows FP32), the B fragments (k x kWideRows x 2 bytes; the E4M3
// ones take its first half), with W8A8 the activation scales
// ([k / 128][kWideRows] FP32), and the split partials.
size_t wide_bytes(int rows, int k, int n, bool w8a8) {
  const int m = rows < kWideRows ? rows : kWideRows;
  int kb_per_split, splits;
  split_plan(n, k, &kb_per_split, &splits);
  return align256(2 * kWideRows * sizeof(float)) + align256(row_max_bytes(k, kWideRows)) +
         align256(size_t(k) * kWideRows * 2) +
         (w8a8 ? align256(size_t(k / 128) * kWideRows * sizeof(float)) : 0) +
         (splits > 1 ? align256(size_t(splits) * m * n * sizeof(float)) : 0);
}

// Rows of the single-pass W8A8 activation arena: one byte of E4M3 per element,
// padded to a whole 128-row tile so the last tile's GROUPS=16 reads stay in range.
int w8a8_stride(int rows) { return ((rows + kWideRows - 1) / kWideRows) * kWideRows; }

size_t w8a8_single_bytes(int rows, int k) {
  const size_t stride = size_t(w8a8_stride(rows));
  return align256(size_t(k) * stride) + align256(size_t(k / 128) * stride * sizeof(float));
}

// Rows <= 64 keep the split-K arena. Wider W8A8 takes the single-pass arena,
// or the split-K one when that is larger, so a buffer admitted for W8A8 still
// covers the wide and W8A16 modes on the same shape.
size_t w8a8_workspace(int rows, int k, int n) {
  const size_t legacy = wide_bytes(rows, k, n, true);
  if (rows <= kMaxRows) return legacy;
  const size_t single = w8a8_single_bytes(rows, k);
  return single > legacy ? single : legacy;
}

// cuteafd_fp8_linear's modes.
constexpr int kModeW8a16 = 0, kModeWide = 1, kModeW8a8 = 2;

}  // namespace

// Packs a BF16 [n, k] weight (n % 16 == 0, k % 128 == 0) into `packed`
// (n * k bytes, A-fragment order) and `scale` ([n, k / 128] FP32) under
// scale `rule` (0 amax / 448, 1 power of two, 2 the better per block).
extern "C" int32_t cuteafd_fp8_w8a16_pack(const void* w, void* packed, void* scale, int32_t n, int32_t k,
                                          int32_t rule, void* stream) {
  if (n < 16 || n % 16 || k < 128 || k % 128 || rule < 0 || rule > 2) return cudaErrorInvalidValue;
  pack_kernel<<<dim3(k / 128, n / 16), 32, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(w), static_cast<uint8_t*>(packed), static_cast<float*>(scale), n, k, rule);
  return cudaGetLastError();
}

// Scratch bytes cuteafd_fp8_w8a16_linear needs for these shapes.
extern "C" size_t cuteafd_fp8_w8a16_workspace(int32_t rows, int32_t k, int32_t n) {
  if (n < 16 || k < 128) return 0;
  const int chunk = rows < kMaxRows ? rows : kMaxRows;
  int kb_per_split, splits;
  split_plan(n, k, &kb_per_split, &splits);
  return align256(2 * kMaxRows * sizeof(float)) + align256(row_max_bytes(k, kMaxRows)) +
         align256(size_t(k) * kMaxRows * 2) +
         (splits > 1 ? align256(size_t(splits) * chunk * n * sizeof(float)) : 0);
}

// out [rows, n] (BF16, or FP32 with out_f32) = x [rows, k] BF16 @ W^T for a
// packed weight; `workspace` holds cuteafd_fp8_w8a16_workspace bytes.
extern "C" int32_t cuteafd_fp8_w8a16_linear(const void* x, const void* packed, const void* scale, void* out,
                                            int32_t out_f32, int32_t rows, int32_t k, int32_t n, void* workspace,
                                            size_t workspace_bytes, void* stream) {
  if (rows < 1 || n < 16 || n % 16 || k < 128 || k % 128) return cudaErrorInvalidValue;
  if (workspace_bytes < cuteafd_fp8_w8a16_workspace(rows, k, n)) return cudaErrorInvalidValue;
  const cudaStream_t s = static_cast<cudaStream_t>(stream);
  uint8_t* ws = static_cast<uint8_t*>(workspace);
  float* sx = reinterpret_cast<float*>(ws);
  float* maxima = reinterpret_cast<float*>(ws + align256(2 * kMaxRows * sizeof(float)));
  uint4* xp = reinterpret_cast<uint4*>(ws + align256(2 * kMaxRows * sizeof(float)) +
                                     align256(row_max_bytes(k, kMaxRows)));
  float* partial = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(xp) +
                                            align256(size_t(k) * kMaxRows * 2));
  const int max_splits = row_max_splits(k);
  int kb_per_split, splits;
  split_plan(n, k, &kb_per_split, &splits);
  const size_t out_row = size_t(n) * (out_f32 ? 4 : 2);
  for (int first = 0; first < rows; first += kMaxRows) {
    const int m = rows - first < kMaxRows ? rows - first : kMaxRows;
    const int groups = m <= 8 ? 1 : m <= 16 ? 2 : m <= 32 ? 4 : 8;
    const __nv_bfloat16* xm = static_cast<const __nv_bfloat16*>(x) + size_t(first) * k;
    void* om = static_cast<uint8_t*>(out) + size_t(first) * out_row;
    row_max_kernel<<<dim3(max_splits, groups * 8), kScaleCols / 2, 0, s>>>(xm, maxima, m, k, max_splits);
    pack_x_kernel<<<dim3(k / 32, groups), 32, 0, s>>>(xm, maxima, sx, xp, m, k, groups, kMaxRows, max_splits);
    const int warps = (n / 16) * splits;
    const dim3 grid((warps + kWarps - 1) / kWarps), block(kWarps * 32);
    const uint4* wp = static_cast<const uint4*>(packed);
    const float* sc = static_cast<const float*>(scale);
    switch (groups) {
      case 1: gemv_kernel<1><<<grid, block, 0, s>>>(wp, sc, xp, sx, om, partial, out_f32, m, n, k, kb_per_split, splits); break;
      case 2: gemv_kernel<2><<<grid, block, 0, s>>>(wp, sc, xp, sx, om, partial, out_f32, m, n, k, kb_per_split, splits); break;
      case 4: gemv_kernel<4><<<grid, block, 0, s>>>(wp, sc, xp, sx, om, partial, out_f32, m, n, k, kb_per_split, splits); break;
      default: gemv_kernel<8><<<grid, block, 0, s>>>(wp, sc, xp, sx, om, partial, out_f32, m, n, k, kb_per_split, splits); break;
    }
    if (splits > 1) {
      const size_t total = size_t(m) * n;
      const int blocks = int((total + 255) / 256 < 4096 ? (total + 255) / 256 : 4096);
      reduce_kernel<<<blocks, 256, 0, s>>>(partial, sx, om, out_f32, m, n, splits);
    }
  }
  return cudaGetLastError();
}

namespace {

// W8A8 past 64 rows: one pack of every row (padded to 128) and one GEMM.
// The split-K scratch is not touched. Rows <= 64 do not call this.
int32_t run_w8a8_single(const void* x, const void* packed, const void* scale, void* out, int32_t out_f32,
                        int32_t rows, int32_t k, int32_t n, void* workspace, size_t workspace_bytes, void* stream) {
  if (rows <= kMaxRows || n < 16 || n % 16 || k < 128 || k % 128) return cudaErrorInvalidValue;
  if (workspace_bytes < w8a8_workspace(rows, k, n)) return cudaErrorInvalidValue;
  const int stride = w8a8_stride(rows);
  const int groups = stride / 8;
  const int m_tiles = stride / kWideRows;
  const int n_ctas = (n / 16 + kWarps - 1) / kWarps;
  if (m_tiles > 65535 || groups > 65535 || n_ctas > 65535) return cudaErrorInvalidValue;
  const cudaStream_t s = static_cast<cudaStream_t>(stream);
  uint8_t* ws = static_cast<uint8_t*>(workspace);
  uint2* xq = reinterpret_cast<uint2*>(ws);
  float* sxa = reinterpret_cast<float*>(ws + align256(size_t(k) * size_t(stride)));
  ++g_fp8_linear_launches;
  pack_x_e4m3_kernel<<<dim3(k / 128, groups), 32, 0, s>>>(
      static_cast<const __nv_bfloat16*>(x), xq, sxa, rows, k, groups, stride);
  ++g_fp8_linear_launches;
  gemv_w8a8_wide_kernel<16><<<dim3(m_tiles, n_ctas), kWarps * 32, 0, s>>>(
      static_cast<const uint4*>(packed), static_cast<const float*>(scale), xq, sxa, out, out_f32, rows, n, k, stride,
      groups);
  return cudaGetLastError();
}

// out = x @ W^T in passes of kWideRows rows; with `w8a8`, a pass of more than
// kW8a16Rows rows takes E4M3 activations. A pass of up to kMaxRows rows on the
// f16 path is one chunk of cuteafd_fp8_w8a16_linear and runs there (the same
// kernels and bits); the wide passes launch the kernels above at 16 groups.
// W8A8 with more than kMaxRows rows is one split-free 2D grid instead.
int32_t run_wide(const void* x, const void* packed, const void* scale, void* out, int32_t out_f32, int32_t rows,
                 int32_t k, int32_t n, bool w8a8, void* workspace, size_t workspace_bytes, void* stream) {
  if (w8a8 && rows > kMaxRows) {
    return run_w8a8_single(x, packed, scale, out, out_f32, rows, k, n, workspace, workspace_bytes, stream);
  }
  if (rows < 1 || n < 16 || n % 16 || k < 128 || k % 128) return cudaErrorInvalidValue;
  if (workspace_bytes < wide_bytes(rows, k, n, w8a8)) return cudaErrorInvalidValue;
  const cudaStream_t s = static_cast<cudaStream_t>(stream);
  uint8_t* ws = static_cast<uint8_t*>(workspace);
  float* sx = reinterpret_cast<float*>(ws);
  // gemv_kernel and reduce_kernel read a row's inverse at [kMaxRows + m].
  const float* inverse = sx + (kWideRows - kMaxRows);
  float* maxima = reinterpret_cast<float*>(ws + align256(2 * kWideRows * sizeof(float)));
  uint8_t* fragments = ws + align256(2 * kWideRows * sizeof(float)) + align256(row_max_bytes(k, kWideRows));
  const int max_splits = row_max_splits(k);
  uint4* xp = reinterpret_cast<uint4*>(fragments);
  uint2* xq = reinterpret_cast<uint2*>(fragments);
  float* sxa = reinterpret_cast<float*>(fragments + align256(size_t(k) * kWideRows * 2));
  float* partial = reinterpret_cast<float*>(fragments + align256(size_t(k) * kWideRows * 2) +
                                            (w8a8 ? align256(size_t(k / 128) * kWideRows * sizeof(float)) : 0));
  int kb_per_split, splits;
  split_plan(n, k, &kb_per_split, &splits);
  const size_t out_row = size_t(n) * (out_f32 ? 4 : 2);
  const int warps = (n / 16) * splits;
  const dim3 grid((warps + kWarps - 1) / kWarps), block(kWarps * 32);
  const uint4* wp = static_cast<const uint4*>(packed);
  const float* sc = static_cast<const float*>(scale);
  for (int first = 0; first < rows; first += kWideRows) {
    const int m = rows - first < kWideRows ? rows - first : kWideRows;
    const __nv_bfloat16* xm = static_cast<const __nv_bfloat16*>(x) + size_t(first) * k;
    void* om = static_cast<uint8_t*>(out) + size_t(first) * out_row;
    const bool e4m3 = w8a8 && m > kW8a16Rows;
    if (!e4m3 && m <= kMaxRows) {
      // The wide layout holds the W8A16 one (its scratch is no smaller).
      const int32_t status =
          cuteafd_fp8_w8a16_linear(xm, packed, scale, om, out_f32, m, k, n, workspace, workspace_bytes, stream);
      if (status != 0) return status;
      continue;
    }
    if (e4m3) {
      const int groups = m <= 16 ? 2 : m <= 32 ? 4 : m <= 64 ? 8 : 16;
      pack_x_e4m3_kernel<<<dim3(k / 128, groups), 32, 0, s>>>(xm, xq, sxa, m, k, groups, kWideRows);
      switch (groups) {
        case 2: gemv_w8a8_kernel<2><<<grid, block, 0, s>>>(wp, sc, xq, sxa, om, partial, out_f32, m, n, k, kb_per_split, splits, kWideRows); break;
        case 4: gemv_w8a8_kernel<4><<<grid, block, 0, s>>>(wp, sc, xq, sxa, om, partial, out_f32, m, n, k, kb_per_split, splits, kWideRows); break;
        case 8: gemv_w8a8_kernel<8><<<grid, block, 0, s>>>(wp, sc, xq, sxa, om, partial, out_f32, m, n, k, kb_per_split, splits, kWideRows); break;
        default: gemv_w8a8_kernel<16><<<grid, block, 0, s>>>(wp, sc, xq, sxa, om, partial, out_f32, m, n, k, kb_per_split, splits, kWideRows); break;
      }
    } else {
      // kMaxRows < m <= kWideRows: 16 groups of 8 rows.
      row_max_kernel<<<dim3(max_splits, kWideRows), kScaleCols / 2, 0, s>>>(xm, maxima, m, k, max_splits);
      pack_x_kernel<<<dim3(k / 32, 16), 32, 0, s>>>(xm, maxima, sx, xp, m, k, 16, kWideRows, max_splits);
      gemv_kernel<16><<<grid, block, 0, s>>>(wp, sc, xp, inverse, om, partial, out_f32, m, n, k, kb_per_split, splits);
    }
    if (splits > 1) {
      const size_t total = size_t(m) * n;
      const int blocks = int((total + 255) / 256 < 4096 ? (total + 255) / 256 : 4096);
      if (e4m3) {
        reduce_w8a8_kernel<<<blocks, 256, 0, s>>>(partial, om, out_f32, m, n, splits);
      } else {
        reduce_kernel<<<blocks, 256, 0, s>>>(partial, inverse, om, out_f32, m, n, splits);
      }
    }
  }
  return cudaGetLastError();
}

}  // namespace

// Scratch bytes cuteafd_fp8_linear needs in `mode`: 0 the W8A16 passes of 64
// rows above (cuteafd_fp8_w8a16_workspace), 1 wide: passes of 128 rows with
// the same bits, 2 W8A8: the f16 path up to 8 rows (the same bits), E4M3
// activations past them. Past 64 rows mode 2 is one split-free pass and its
// arena is the larger of that pass and the split-K pass. 0 for an unknown
// mode or shape.
extern "C" size_t cuteafd_fp8_linear_workspace(int32_t rows, int32_t k, int32_t n, int32_t mode) {
  if (n < 16 || k < 128) return 0;
  switch (mode) {
    case kModeW8a16: return cuteafd_fp8_w8a16_workspace(rows, k, n);
    case kModeWide: return wide_bytes(rows, k, n, false);
    case kModeW8a8: return w8a8_workspace(rows, k, n);
    default: return 0;
  }
}

extern "C" void cuteafd_fp8_linear_reset_launches(void) { g_fp8_linear_launches = 0; }

extern "C" int32_t cuteafd_fp8_linear_launch_count(void) { return g_fp8_linear_launches; }

// cuteafd_fp8_w8a16_linear in `mode` (see cuteafd_fp8_linear_workspace); mode
// 0 is cuteafd_fp8_w8a16_linear.
extern "C" int32_t cuteafd_fp8_linear(const void* x, const void* packed, const void* scale, void* out, int32_t out_f32,
                                      int32_t rows, int32_t k, int32_t n, int32_t mode, void* workspace,
                                      size_t workspace_bytes, void* stream) {
  switch (mode) {
    case kModeW8a16:
      return cuteafd_fp8_w8a16_linear(x, packed, scale, out, out_f32, rows, k, n, workspace, workspace_bytes, stream);
    case kModeWide:
      return run_wide(x, packed, scale, out, out_f32, rows, k, n, false, workspace, workspace_bytes, stream);
    case kModeW8a8:
      return run_wide(x, packed, scale, out, out_f32, rows, k, n, true, workspace, workspace_bytes, stream);
    default: return cudaErrorInvalidValue;
  }
}

// Standalone W8A8 wide-GEMM harness for kernel duel #100.
// Not a CMake target: compile with nvcc and pass the candidate and baseline
// libfp8_gemv.so paths. The baseline is the unmodified fp8_gemv.cu.
//
// Correctness: single-pass output against an FP64 sum of the same dequantized
// E4M3 products (device quant, checked against the bytes pack_kernel and
// pack_x_e4m3 actually stored). Narrow rows must be bit-identical to baseline.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <dlfcn.h>

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#define CHECK(expr)                                                                 \
  do {                                                                              \
    cudaError_t _st = (expr);                                                       \
    if (_st != cudaSuccess) {                                                       \
      std::fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_st)); \
      std::exit(1);                                                                 \
    }                                                                               \
  } while (0)

using PackFn = int32_t (*)(const void*, void*, void*, int32_t, int32_t, int32_t, void*);
using WorkspaceFn = size_t (*)(int32_t, int32_t, int32_t, int32_t);
using LinearFn = int32_t (*)(const void*, const void*, const void*, void*, int32_t, int32_t, int32_t, int32_t,
                             int32_t, void*, size_t, void*);
using ResetFn = void (*)();
using CountFn = int32_t (*)();

struct Lib {
  void* handle = nullptr;
  PackFn pack = nullptr;
  WorkspaceFn workspace = nullptr;
  LinearFn linear = nullptr;
  ResetFn reset = nullptr;
  CountFn count = nullptr;

  void open(const char* path, bool launches) {
    handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
      std::fprintf(stderr, "dlopen %s: %s\n", path, dlerror());
      std::exit(1);
    }
    auto sym = [&](const char* name) {
      void* p = dlsym(handle, name);
      if (!p) {
        std::fprintf(stderr, "dlsym %s from %s: %s\n", name, path, dlerror());
        std::exit(1);
      }
      return p;
    };
    pack = reinterpret_cast<PackFn>(sym("cuteafd_fp8_w8a16_pack"));
    workspace = reinterpret_cast<WorkspaceFn>(sym("cuteafd_fp8_linear_workspace"));
    linear = reinterpret_cast<LinearFn>(sym("cuteafd_fp8_linear"));
    if (launches) {
      reset = reinterpret_cast<ResetFn>(sym("cuteafd_fp8_linear_reset_launches"));
      count = reinterpret_cast<CountFn>(sym("cuteafd_fp8_linear_launch_count"));
    }
  }
};

static cudaStream_t g_stream;

__device__ __forceinline__ uint32_t e4m3x4(float a, float b, float c, float d) {
  const uint32_t lo = __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
  const uint32_t hi = __nv_cvt_float2_to_fp8x2(make_float2(c, d), __NV_SATFINITE, __NV_E4M3);
  return lo | (hi << 16);
}

__device__ __forceinline__ float e4m3_f(uint8_t byte) {
  __nv_fp8_e4m3 v;
  v.__x = byte;
  return float(v);
}

// Rule 0 (amax / 448) dense dequant of a BF16 [n, k] weight. Same reduction and
// constructor rounding as pack_kernel.
__global__ void quant_w_dense(const __nv_bfloat16* __restrict__ w, float* __restrict__ dense, int k) {
  const int kb = blockIdx.x, rt = blockIdx.y, lane = threadIdx.x;
  for (int r = 0; r < 16; ++r) {
    const __nv_bfloat16* row = w + (size_t(rt) * 16 + r) * k + kb * 128 + lane * 4;
    float x[4], amax = 0.0f;
    for (int j = 0; j < 4; ++j) {
      x[j] = __bfloat162float(row[j]);
      amax = fmaxf(amax, fabsf(x[j]));
    }
    for (int offset = 16; offset; offset >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
    const float s = amax > 0.0f ? amax / 448.0f : 1.0f;
    for (int j = 0; j < 4; ++j) {
      dense[(size_t(rt) * 16 + r) * k + kb * 128 + lane * 4 + j] = float(__nv_fp8_e4m3(x[j] / s)) * s;
    }
  }
}

// Inverse of pack_kernel's A-fragment store, scaled by the packed scales.
__global__ void unpack_w(const uint8_t* __restrict__ packed, const float* __restrict__ scale, float* __restrict__ dense,
                         int k) {
  const int kb = blockIdx.x, rt = blockIdx.y, lane = threadIdx.x;
  const int kbs = k / 128;
  const int g = lane / 4, c = (lane % 4) * 2;
  const float s_lo = scale[(size_t(rt) * 16 + g) * kbs + kb];
  const float s_hi = scale[(size_t(rt) * 16 + g + 8) * kbs + kb];
  for (int kt = 0; kt < 4; ++kt) {
    const uint4 v = reinterpret_cast<const uint4*>(packed)[(size_t(rt) * (k / 32) + kb * 4 + kt) * 32 + lane];
    uint8_t o[16];
    memcpy(o, &v, 16);
    const int row_lo = rt * 16 + g, row_hi = row_lo + 8;
    for (int step = 0; step < 2; ++step) {
      const int k0 = kb * 128 + kt * 32 + step * 16 + c;
      const uint8_t* b = o + step * 8;
      const int cols[8] = {k0, k0 + 1, k0, k0 + 1, k0 + 8, k0 + 9, k0 + 8, k0 + 9};
      const int rows[8] = {row_lo, row_lo, row_hi, row_hi, row_lo, row_lo, row_hi, row_hi};
      const float ss[8] = {s_lo, s_lo, s_hi, s_hi, s_lo, s_lo, s_hi, s_hi};
      for (int i = 0; i < 8; ++i) dense[size_t(rows[i]) * k + cols[i]] = e4m3_f(b[i]) * ss[i];
    }
  }
}

// Dense dequant of pack_x_e4m3_kernel (cvt rounding, per row x 128-K scale).
__global__ void quant_x_dense(const __nv_bfloat16* __restrict__ x, float* __restrict__ dense, int rows, int k) {
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
  if (m >= rows) return;
#pragma unroll
  for (int kt = 0; kt < 4; ++kt) {
    const uint32_t qx = e4m3x4(v[kt][0] / s, v[kt][1] / s, v[kt][2] / s, v[kt][3] / s);
    const uint32_t qy = e4m3x4(v[kt][4] / s, v[kt][5] / s, v[kt][6] / s, v[kt][7] / s);
    const int base = kb * 128 + kt * 32;
    const int cols[8] = {c, c + 1, c + 8, c + 9, c + 16, c + 17, c + 24, c + 25};
    const uint32_t words[2] = {qx, qy};
    for (int w = 0; w < 2; ++w) {
      for (int b = 0; b < 4; ++b) {
        const int i = w * 4 + b;
        dense[size_t(m) * k + base + cols[i]] = e4m3_f(uint8_t(words[w] >> (8 * b))) * s;
      }
    }
  }
}

// Inverse of the single-pass pack_x layout (xq at workspace + 0, scales after it).
__global__ void unpack_x(const uint2* __restrict__ xq, const float* __restrict__ sxa, float* __restrict__ dense, int rows,
                         int k, int stride) {
  const int kb = blockIdx.x, group = blockIdx.y, lane = threadIdx.x;
  const int groups = stride / 8;
  const int m = group * 8 + lane / 4, c = (lane % 4) * 2;
  if (m >= rows) return;
  const float s = sxa[size_t(kb) * stride + m];
  for (int kt = 0; kt < 4; ++kt) {
    const uint2 q = xq[(size_t(kb * 4 + kt) * groups + group) * 32 + lane];
    uint8_t b[8];
    memcpy(b, &q, 8);
    const int cols[8] = {c, c + 1, c + 8, c + 9, c + 16, c + 17, c + 24, c + 25};
    for (int i = 0; i < 8; ++i) dense[size_t(m) * k + kb * 128 + kt * 32 + cols[i]] = e4m3_f(b[i]) * s;
  }
}

__global__ void e4m3_table(float* out) {
  const int i = threadIdx.x;
  __nv_fp8_e4m3 v;
  v.__x = static_cast<uint8_t>(i);
  out[i] = float(v);
}

__global__ void ctor_vs_cvt(const float* in, int* ctor_bytes, int* cvt_bytes, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  ctor_bytes[i] = int(__nv_fp8_e4m3(in[i]).__x);
  const uint32_t pair = __nv_cvt_float2_to_fp8x2(make_float2(in[i], 0.0f), __NV_SATFINITE, __NV_E4M3);
  cvt_bytes[i] = int(pair & 0xff);
}

__global__ void f32_to_bf16_kernel(const float* in, __nv_bfloat16* out, size_t n) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2bfloat16(in[i]);
}

__global__ void first_diff(const float* a, const float* b, size_t n, unsigned long long* count, unsigned long long* index) {
  const size_t stride = size_t(gridDim.x) * blockDim.x;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
    if (__float_as_uint(a[i]) != __float_as_uint(b[i])) {
      atomicAdd(reinterpret_cast<unsigned long long*>(count), 1ull);
      atomicCAS(index, ~0ull, i);
    }
  }
}

static int sm_count() {
  int device = 0, count = 0;
  CHECK(cudaGetDevice(&device));
  CHECK(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device));
  return count;
}

static int splits_for(int n, int k, int sms) {
  const int tiles = n / 16, kbs = k / 128;
  const int wanted = (sms * 16 + tiles - 1) / tiles;
  const int s = wanted < 1 ? 1 : (wanted > kbs ? kbs : wanted);
  const int kb = (kbs + s - 1) / s;
  return (kbs + kb - 1) / kb;
}

// One 128-row W8A8 pass is pack + gemm + reduce when K is split.
static int old_kernels(int rows, int n, int k, int sms) {
  const int passes = (rows + 127) / 128;
  const int splits = splits_for(n, k, sms);
  return passes * (2 + (splits > 1 ? 1 : 0));
}

static uint16_t f32_to_bf16(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  const uint32_t lsb = (u >> 16) & 1u;
  u += 0x7fffu + lsb;
  return static_cast<uint16_t>(u >> 16);
}

static float host_e4m3(uint8_t byte) {
  const int sign = byte >> 7;
  const int exp = (byte >> 3) & 0xf;
  const int man = byte & 7;
  float v;
  if (exp == 0) {
    v = std::ldexp(float(man), -9);
  } else if (exp == 15 && man == 7) {
    v = std::numeric_limits<float>::quiet_NaN();
  } else {
    v = std::ldexp(1.0f + float(man) / 8.0f, exp - 7);
  }
  return sign ? -v : v;
}

static void check_e4m3_table() {
  float* d = nullptr;
  CHECK(cudaMalloc(&d, 256 * sizeof(float)));
  e4m3_table<<<1, 256>>>(d);
  CHECK(cudaGetLastError());
  float h[256];
  CHECK(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int i = 0; i < 256; ++i) {
    const float ref = host_e4m3(static_cast<uint8_t>(i));
    const bool nan_ok = std::isnan(ref) && std::isnan(h[i]);
    uint32_t a, b;
    std::memcpy(&a, &ref, 4);
    std::memcpy(&b, h + i, 4);
    if (!nan_ok && a != b) {
      if (bad < 4) std::printf("e4m3 code %d host %g device %g\n", i, ref, h[i]);
      ++bad;
    }
  }
  std::printf("e4m3 table mismatches %d\n", bad);
  if (bad) std::exit(1);

  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
  const int n = 4096;
  std::vector<float> in(n);
  for (float& v : in) v = dist(rng);
  float* din = nullptr;
  int *dctor = nullptr, *dcvt = nullptr;
  CHECK(cudaMalloc(&din, n * sizeof(float)));
  CHECK(cudaMalloc(&dctor, n * sizeof(int)));
  CHECK(cudaMalloc(&dcvt, n * sizeof(int)));
  CHECK(cudaMemcpy(din, in.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  ctor_vs_cvt<<<(n + 255) / 256, 256>>>(din, dctor, dcvt, n);
  CHECK(cudaGetLastError());
  std::vector<int> ctor(n), cvt(n);
  CHECK(cudaMemcpy(ctor.data(), dctor, n * sizeof(int), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(cvt.data(), dcvt, n * sizeof(int), cudaMemcpyDeviceToHost));
  int differ = 0;
  for (int i = 0; i < n; ++i)
    if (ctor[i] != cvt[i]) ++differ;
  std::printf("ctor vs cvt mismatches on %d random floats: %d (weight uses ctor, activation uses cvt)\n", n, differ);
  CHECK(cudaFree(d));
  CHECK(cudaFree(din));
  CHECK(cudaFree(dctor));
  CHECK(cudaFree(dcvt));
}

static unsigned long long diff_count(const float* a, const float* b, size_t n, unsigned long long* first) {
  unsigned long long *dcount = nullptr, *dindex = nullptr;
  CHECK(cudaMalloc(&dcount, sizeof(unsigned long long)));
  CHECK(cudaMalloc(&dindex, sizeof(unsigned long long)));
  const unsigned long long zero = 0, none = ~0ull;
  CHECK(cudaMemcpy(dcount, &zero, sizeof(zero), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dindex, &none, sizeof(none), cudaMemcpyHostToDevice));
  const int threads = 256;
  const int blocks = std::min(2048, int((n + threads - 1) / threads));
  first_diff<<<blocks, threads>>>(a, b, n, dcount, dindex);
  CHECK(cudaGetLastError());
  unsigned long long count = 0, index = none;
  CHECK(cudaMemcpy(&count, dcount, sizeof(count), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(&index, dindex, sizeof(index), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(dcount));
  CHECK(cudaFree(dindex));
  if (first) *first = index;
  return count;
}

static void must_match(const char* what, const float* a, const float* b, size_t n) {
  unsigned long long first = 0;
  const unsigned long long count = diff_count(a, b, n, &first);
  std::printf("%s mismatches %llu", what, count);
  if (count) {
    float hv[2];
    CHECK(cudaMemcpy(hv, a + first, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hv + 1, b + first, sizeof(float), cudaMemcpyDeviceToHost));
    std::printf(" first %llu a %g b %g", first, hv[0], hv[1]);
  }
  std::printf("\n");
  if (count) std::exit(1);
}

// C[m, n] = sum_k A[m, k] * B[n, k], accumulated in FP64. A and B are the
// dequantized E4M3 values (code converted to FP32, times the block scale).
static void gemm_fp64(const float* A, const float* B, std::vector<double>& C, int M, int N, int K) {
  constexpr int BM = 8, BN = 16, BK = 64;
  C.assign(size_t(M) * N, 0.0);
#pragma omp parallel for schedule(static)
  for (int m0 = 0; m0 < M; m0 += BM) {
    const int mi = std::min(BM, M - m0);
    for (int n0 = 0; n0 < N; n0 += BN) {
      double acc[BM][BN];
      for (int i = 0; i < BM; ++i)
        for (int j = 0; j < BN; ++j) acc[i][j] = 0.0;
      for (int k0 = 0; k0 < K; k0 += BK) {
        double panel[BK][BN];
        for (int j = 0; j < BN; ++j) {
          const float* brow = B + (size_t(n0) + j) * K + k0;
          for (int kk = 0; kk < BK; ++kk) panel[kk][j] = double(brow[kk]);
        }
        for (int i = 0; i < mi; ++i) {
          const float* a = A + (size_t(m0) + i) * K + k0;
          for (int kk = 0; kk < BK; ++kk) {
            const double av = double(a[kk]);
            for (int j = 0; j < BN; ++j) acc[i][j] += av * panel[kk][j];
          }
        }
      }
      for (int i = 0; i < mi; ++i)
        for (int j = 0; j < BN; ++j) C[(size_t(m0) + i) * N + n0 + j] = acc[i][j];
    }
  }
}

struct Stats {
  double max_abs = 0, max_rel = 0, max_rel_big = 0, cosine = 0;
  bool finite = true;
};

static Stats compare(const std::vector<float>& got, const std::vector<double>& ref) {
  Stats s;
  long double dot = 0, ng = 0, nr = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    if (!std::isfinite(got[i]) || !std::isfinite(ref[i])) {
      s.finite = false;
      continue;
    }
    const double d = std::fabs(double(got[i]) - ref[i]);
    s.max_abs = std::max(s.max_abs, d);
    const double denom = std::max(std::fabs(ref[i]), 1e-6);
    s.max_rel = std::max(s.max_rel, d / denom);
    if (std::fabs(ref[i]) > 1e-3) s.max_rel_big = std::max(s.max_rel_big, d / std::fabs(ref[i]));
    const long double gv = static_cast<long double>(got[i]);
    dot += gv * static_cast<long double>(ref[i]);
    ng += gv * gv;
    nr += ref[i] * ref[i];
  }
  s.cosine = (ng > 0 && nr > 0) ? double(dot / std::sqrt(ng * nr)) : 0;
  return s;
}

static Stats compare_float(const std::vector<float>& got, const std::vector<float>& other) {
  std::vector<double> ref(other.size());
  for (size_t i = 0; i < other.size(); ++i) ref[i] = double(other[i]);
  return compare(got, ref);
}

static void fill_bf16(std::mt19937& rng, std::vector<uint16_t>& bits, float scale) {
  std::normal_distribution<float> normal(0.0f, 1.0f);
  for (size_t i = 0; i < bits.size(); ++i) {
    float v = normal(rng) * scale;
    if ((i % 997) == 0) v = 0.0f;
    bits[i] = f32_to_bf16(v);
  }
}

static void fill_ternary_w(std::mt19937& rng, std::vector<uint16_t>& bits, int n, int k) {
  std::uniform_int_distribution<int> digit(-1, 1);
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < k; ++col) {
      const int mag = (col % 128 == 0) ? 1 : digit(rng);
      bits[size_t(row) * k + col] = f32_to_bf16(float(mag) * std::exp2(-6.0f));
    }
  }
}

static void fill_ternary_x(std::mt19937& rng, std::vector<uint16_t>& bits, int rows, int k) {
  std::uniform_int_distribution<int> digit(-1, 1);
  std::uniform_int_distribution<int> exp_j(-12, -3);
  for (int row = 0; row < rows; ++row) {
    for (int kb = 0; kb < k / 128; ++kb) {
      const float scale = 448.0f * std::exp2(float(exp_j(rng)));
      for (int c = 0; c < 128; ++c) {
        const int mag = c == 0 ? 1 : digit(rng);
        bits[size_t(row) * k + kb * 128 + c] = f32_to_bf16(float(mag) * scale);
      }
    }
  }
}

static std::string uuid_string() {
  cudaDeviceProp prop{};
  CHECK(cudaGetDeviceProperties(&prop, 0));
  const unsigned char* b = reinterpret_cast<const unsigned char*>(prop.uuid.bytes);
  char buf[64];
  std::snprintf(buf, sizeof(buf), "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", b[0], b[1],
                b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
  return buf;
}

static void require_ok(const char* what, int32_t status) {
  if (status != 0) {
    std::fprintf(stderr, "%s status %d\n", what, status);
    std::exit(1);
  }
}

static int call_linear(Lib& lib, const void* x, const void* packed, const void* scale, void* out, int out_f32, int rows,
                       int k, int n, int mode, void* ws, size_t bytes) {
  return lib.linear(x, packed, scale, out, out_f32, rows, k, n, mode, ws, bytes, g_stream);
}

int main(int argc, char** argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: %s candidate.so baseline.so expected-uuid results.txt\n", argv[0]);
    return 2;
  }
  std::ofstream out(argv[4]);
  auto both = [&](const std::string& line) {
    std::cout << line << std::endl;
    out << line << std::endl;
  };

  CHECK(cudaSetDevice(0));
  CHECK(cudaStreamCreate(&g_stream));
  cudaDeviceProp prop{};
  CHECK(cudaGetDeviceProperties(&prop, 0));
  const int sms = sm_count();
  const std::string uuid = uuid_string();
  both("device " + std::string(prop.name) + " sms " + std::to_string(sms) + " uuid " + uuid);
  if (uuid != argv[3]) {
    both("FAIL uuid expected " + std::string(argv[3]));
    return 1;
  }
  check_e4m3_table();

  Lib candidate, baseline;
  candidate.open(argv[1], true);
  baseline.open(argv[2], false);

  const int shapes[][2] = {{20480, 4096}, {4096, 2048}};
  const char* names[] = {"fc", "kv"};
  const int narrow[] = {1, 8, 16, 64};
  const int wide_rows[] = {65, 128, 129, 1024, 2048};
  const int max_m = 2048;
  bool correct = true;

  auto workspace_of = [&](Lib& lib, int rows, int k, int n, int mode) {
    return lib.workspace(rows, k, n, mode);
  };

  for (int s = 0; s < 2; ++s) {
    const int k = shapes[s][0], n = shapes[s][1];
    for (int rows : {1, 8, 16, 64, 65, 128, 129, 1024, 2048}) {
      for (int mode = 0; mode < 3; ++mode) {
        const size_t a = workspace_of(candidate, rows, k, n, mode);
        const size_t b = workspace_of(baseline, rows, k, n, mode);
        if (mode != 2 || rows <= 64) {
          if (a != b) {
            both("FAIL workspace " + std::string(names[s]) + " rows " + std::to_string(rows) + " mode " +
                 std::to_string(mode) + " candidate " + std::to_string(a) + " baseline " + std::to_string(b));
            correct = false;
          }
        } else if (a < b) {
          both("FAIL wide workspace shrank " + std::string(names[s]) + " rows " + std::to_string(rows));
          correct = false;
        }
        const size_t cap = workspace_of(candidate, max_m, k, n, mode);
        if (a > cap) {
          both("FAIL workspace not monotone " + std::string(names[s]) + " rows " + std::to_string(rows) + " mode " +
               std::to_string(mode));
          correct = false;
        }
      }
    }
    both("workspace " + std::string(names[s]) + " mode2 rows64 " +
         std::to_string(workspace_of(candidate, 64, k, n, 2)) + " rows2048 " +
         std::to_string(workspace_of(candidate, 2048, k, n, 2)) + " baseline2048 " +
         std::to_string(workspace_of(baseline, 2048, k, n, 2)));
  }
  if (!correct) return 1;

  const size_t max_k = 20480, max_n = 4096;
  uint16_t *d_w = nullptr, *d_x = nullptr;
  uint8_t *d_packed = nullptr, *d_ws_c = nullptr, *d_ws_b = nullptr;
  float *d_scale = nullptr, *d_wq = nullptr, *d_unpack = nullptr, *d_xq = nullptr, *d_out = nullptr, *d_out_b = nullptr;
  __nv_bfloat16 *d_bf = nullptr, *d_bf_b = nullptr, *d_bf_round = nullptr;
  CHECK(cudaMalloc(&d_w, max_n * max_k * sizeof(uint16_t)));
  CHECK(cudaMalloc(&d_x, size_t(max_m) * max_k * sizeof(uint16_t)));
  CHECK(cudaMalloc(&d_packed, max_n * max_k));
  CHECK(cudaMalloc(&d_scale, max_n * (max_k / 128) * sizeof(float)));
  CHECK(cudaMalloc(&d_wq, max_n * max_k * sizeof(float)));
  CHECK(cudaMalloc(&d_unpack, std::max(max_n * max_k, size_t(max_m) * max_k) * sizeof(float)));
  CHECK(cudaMalloc(&d_xq, size_t(max_m) * max_k * sizeof(float)));
  CHECK(cudaMalloc(&d_out, size_t(max_m) * max_n * sizeof(float)));
  CHECK(cudaMalloc(&d_out_b, size_t(max_m) * max_n * sizeof(float)));
  CHECK(cudaMalloc(&d_bf, size_t(max_m) * max_n * sizeof(__nv_bfloat16)));
  CHECK(cudaMalloc(&d_bf_b, size_t(max_m) * max_n * sizeof(__nv_bfloat16)));
  CHECK(cudaMalloc(&d_bf_round, size_t(max_m) * max_n * sizeof(__nv_bfloat16)));

  size_t ws_cap_c = 0, ws_cap_b = 0;
  for (int s = 0; s < 2; ++s) {
    for (int mode = 0; mode < 3; ++mode) {
      ws_cap_c = std::max(ws_cap_c, candidate.workspace(max_m, shapes[s][0], shapes[s][1], mode));
      ws_cap_b = std::max(ws_cap_b, baseline.workspace(max_m, shapes[s][0], shapes[s][1], mode));
    }
  }
  CHECK(cudaMalloc(&d_ws_c, ws_cap_c));
  CHECK(cudaMalloc(&d_ws_b, ws_cap_b));
  both("workspace cap candidate " + std::to_string(ws_cap_c) + " baseline " + std::to_string(ws_cap_b));

  std::vector<uint16_t> h_w(max_n * max_k), h_x(size_t(max_m) * max_k);
  std::vector<float> h_out, h_base, h_wq, h_xq;

  for (int s = 0; s < 2; ++s) {
    const int k = shapes[s][0], n = shapes[s][1];
    std::mt19937 rng(100 + s);
    h_w.resize(size_t(n) * k);
    fill_bf16(rng, h_w, 0.02f);
    CHECK(cudaMemcpy(d_w, h_w.data(), h_w.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
    require_ok("pack", candidate.pack(d_w, d_packed, d_scale, n, k, 0, g_stream));
    CHECK(cudaStreamSynchronize(g_stream));
    quant_w_dense<<<dim3(k / 128, n / 16), 32>>>(reinterpret_cast<const __nv_bfloat16*>(d_w), d_wq, k);
    unpack_w<<<dim3(k / 128, n / 16), 32>>>(d_packed, d_scale, d_unpack, k);
    CHECK(cudaGetLastError());
    must_match((std::string(names[s]) + " weight quant vs packed bytes").c_str(), d_wq, d_unpack, size_t(n) * k);
    h_wq.resize(size_t(n) * k);
    CHECK(cudaMemcpy(h_wq.data(), d_wq, h_wq.size() * sizeof(float), cudaMemcpyDeviceToHost));

    auto run_pair = [&](int rows, int mode, int out_f32, void* out_c, void* out_b) {
      const size_t bc = candidate.workspace(rows, k, n, mode);
      const size_t bb = baseline.workspace(rows, k, n, mode);
      require_ok("candidate", call_linear(candidate, d_x, d_packed, d_scale, out_c, out_f32, rows, k, n, mode, d_ws_c, bc));
      require_ok("baseline", call_linear(baseline, d_x, d_packed, d_scale, out_b, out_f32, rows, k, n, mode, d_ws_b, bb));
      CHECK(cudaStreamSynchronize(g_stream));
    };

    for (int rows : narrow) {
      std::mt19937 xrng(1000 + rows + s * 17);
      h_x.resize(size_t(rows) * k);
      fill_bf16(xrng, h_x, 0.5f);
      CHECK(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
      for (int mode = 0; mode < 3; ++mode) {
        run_pair(rows, mode, 1, d_out, d_out_b);
        std::vector<uint8_t> a(size_t(rows) * n * 4), b(a.size());
        CHECK(cudaMemcpy(a.data(), d_out, a.size(), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(b.data(), d_out_b, b.size(), cudaMemcpyDeviceToHost));
        const bool same = a == b;
        both("bits " + std::string(names[s]) + " M " + std::to_string(rows) + " mode " + std::to_string(mode) +
             " f32 " + (same ? "identical" : "DIFFER"));
        if (!same) correct = false;
        run_pair(rows, mode, 0, d_bf, d_bf_b);
        a.resize(size_t(rows) * n * 2);
        b.resize(a.size());
        CHECK(cudaMemcpy(a.data(), d_bf, a.size(), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(b.data(), d_bf_b, b.size(), cudaMemcpyDeviceToHost));
        const bool same16 = a == b;
        both("bits " + std::string(names[s]) + " M " + std::to_string(rows) + " mode " + std::to_string(mode) +
             " bf16 " + (same16 ? "identical" : "DIFFER"));
        if (!same16) correct = false;
      }
      // Untouched W8A16 / wide routes at the wide row count, plus mode 2 at 64 already covered.
    }
    for (int mode : {0, 1}) {
      const int rows = 2048;
      std::mt19937 xrng(2000 + mode + s * 19);
      h_x.resize(size_t(rows) * k);
      fill_bf16(xrng, h_x, 0.5f);
      CHECK(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
      run_pair(rows, mode, 1, d_out, d_out_b);
      std::vector<uint8_t> a(size_t(rows) * n * 4), b(a.size());
      CHECK(cudaMemcpy(a.data(), d_out, a.size(), cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(b.data(), d_out_b, b.size(), cudaMemcpyDeviceToHost));
      const bool same = a == b;
      both("bits " + std::string(names[s]) + " M 2048 mode " + std::to_string(mode) + " f32 " +
           (same ? "identical" : "DIFFER"));
      if (!same) correct = false;
    }

    // Ternary operands: narrow mode 2 stays bit-identical to mode 0. Wide is reported.
    {
      std::mt19937 wrng(3000 + s), xrng(4000 + s);
      h_w.resize(size_t(n) * k);
      fill_ternary_w(wrng, h_w, n, k);
      CHECK(cudaMemcpy(d_w, h_w.data(), h_w.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
      require_ok("pack ternary", candidate.pack(d_w, d_packed, d_scale, n, k, 1, g_stream));
      CHECK(cudaStreamSynchronize(g_stream));
      for (int rows : {64, 128, 2048}) {
        h_x.resize(size_t(rows) * k);
        fill_ternary_x(xrng, h_x, rows, k);
        CHECK(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
        const size_t bc0 = candidate.workspace(rows, k, n, 0);
        const size_t bc2 = candidate.workspace(rows, k, n, 2);
        require_ok("tern0", call_linear(candidate, d_x, d_packed, d_scale, d_out, 1, rows, k, n, 0, d_ws_c, bc0));
        require_ok("tern2", call_linear(candidate, d_x, d_packed, d_scale, d_out_b, 1, rows, k, n, 2, d_ws_c, bc2));
        CHECK(cudaStreamSynchronize(g_stream));
        h_out.resize(size_t(rows) * n);
        h_base.resize(h_out.size());
        CHECK(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(h_base.data(), d_out_b, h_base.size() * sizeof(float), cudaMemcpyDeviceToHost));
        int differ = 0;
        double max_abs = 0;
        for (size_t i = 0; i < h_out.size(); ++i) {
          uint32_t a, b;
          std::memcpy(&a, &h_out[i], 4);
          std::memcpy(&b, &h_base[i], 4);
          if (a != b) ++differ;
          max_abs = std::max(max_abs, std::fabs(double(h_out[i]) - double(h_base[i])));
        }
        both("ternary " + std::string(names[s]) + " M " + std::to_string(rows) + " mode2_vs_mode0 differ " +
             std::to_string(differ) + " max_abs " + std::to_string(max_abs));
        if (rows <= 64 && differ) correct = false;
      }
      // Restore rule-0 weights for the numeric and perf sections.
      rng.seed(100 + s);
      h_w.resize(size_t(n) * k);
      fill_bf16(rng, h_w, 0.02f);
      CHECK(cudaMemcpy(d_w, h_w.data(), h_w.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
      require_ok("repack", candidate.pack(d_w, d_packed, d_scale, n, k, 0, g_stream));
      CHECK(cudaStreamSynchronize(g_stream));
      quant_w_dense<<<dim3(k / 128, n / 16), 32>>>(reinterpret_cast<const __nv_bfloat16*>(d_w), d_wq, k);
      CHECK(cudaGetLastError());
      CHECK(cudaMemcpy(h_wq.data(), d_wq, h_wq.size() * sizeof(float), cudaMemcpyDeviceToHost));
    }

    for (int rows : wide_rows) {
      std::mt19937 xrng(5000 + rows + s * 23);
      h_x.resize(size_t(rows) * k);
      fill_bf16(xrng, h_x, 0.5f);
      CHECK(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
      const int stride = ((rows + 127) / 128) * 128;
      quant_x_dense<<<dim3(k / 128, stride / 8), 32>>>(reinterpret_cast<const __nv_bfloat16*>(d_x), d_xq, rows, k);
      CHECK(cudaGetLastError());
      const size_t bc = candidate.workspace(rows, k, n, 2);
      const size_t bb = baseline.workspace(rows, k, n, 2);
      require_ok("wide f32", call_linear(candidate, d_x, d_packed, d_scale, d_out, 1, rows, k, n, 2, d_ws_c, bc));
      CHECK(cudaStreamSynchronize(g_stream));
      const size_t xq_bytes = size_t(k) * stride;
      const size_t sxa_off = (xq_bytes + 255) & ~size_t(255);
      unpack_x<<<dim3(k / 128, stride / 8), 32>>>(reinterpret_cast<const uint2*>(d_ws_c),
                                                 reinterpret_cast<const float*>(d_ws_c + sxa_off), d_unpack, rows, k,
                                                 stride);
      CHECK(cudaGetLastError());
      must_match((std::string(names[s]) + " M " + std::to_string(rows) + " act quant vs workspace").c_str(), d_xq,
                 d_unpack, size_t(rows) * k);

      h_xq.resize(size_t(rows) * k);
      h_out.resize(size_t(rows) * n);
      CHECK(cudaMemcpy(h_xq.data(), d_xq, h_xq.size() * sizeof(float), cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost));
      const double t0 = omp_get_wtime();
      std::vector<double> ref;
      gemm_fp64(h_xq.data(), h_wq.data(), ref, rows, n, k);
      both("ref " + std::string(names[s]) + " M " + std::to_string(rows) + " seconds " +
           std::to_string(omp_get_wtime() - t0));
      Stats fp32 = compare(h_out, ref);
      require_ok("wide baseline", call_linear(baseline, d_x, d_packed, d_scale, d_out_b, 1, rows, k, n, 2, d_ws_b, bb));
      CHECK(cudaStreamSynchronize(g_stream));
      h_base.resize(h_out.size());
      CHECK(cudaMemcpy(h_base.data(), d_out_b, h_base.size() * sizeof(float), cudaMemcpyDeviceToHost));
      Stats vs_old = compare_float(h_out, h_base);

      require_ok("wide bf16", call_linear(candidate, d_x, d_packed, d_scale, d_bf, 0, rows, k, n, 2, d_ws_c, bc));
      f32_to_bf16_kernel<<<int((h_out.size() + 255) / 256), 256>>>(d_out, d_bf_round, h_out.size());
      CHECK(cudaGetLastError());
      CHECK(cudaStreamSynchronize(g_stream));
      std::vector<uint16_t> got16(h_out.size()), round16(h_out.size());
      CHECK(cudaMemcpy(got16.data(), d_bf, got16.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(round16.data(), d_bf_round, round16.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
      const bool bf16_is_rounded = got16 == round16;
      std::vector<float> h16(h_out.size());
      for (size_t i = 0; i < h16.size(); ++i) {
        uint32_t u = uint32_t(got16[i]) << 16;
        std::memcpy(&h16[i], &u, 4);
      }
      Stats bf = compare(h16, ref);
      char line[640];
      std::snprintf(line, sizeof(line),
                    "correct %s M %d f32 abs %.3e rel %.3e rel_gt1e-3 %.3e cos %.8f finite %d | bf16 abs %.3e rel %.3e "
                    "cos %.8f rounded %d | vs_old abs %.3e rel %.3e cos %.8f",
                    names[s], rows, fp32.max_abs, fp32.max_rel, fp32.max_rel_big, fp32.cosine, int(fp32.finite),
                    bf.max_abs, bf.max_rel, bf.cosine, int(bf16_is_rounded), vs_old.max_abs, vs_old.max_rel,
                    vs_old.cosine);
      both(line);
      // max_rel uses a 1e-6 floor, so near-zero outputs dominate it. The gate is the
      // relative error where |ref| > 1e-3, plus cosine and a BF16 absolute bound
      // (one ulp at magnitude 2 is ~1.6e-2; a bad fragment is O(1)).
      if (!fp32.finite || !bf.finite || !bf16_is_rounded || fp32.max_rel_big > 2e-2 || bf.max_rel_big > 5e-2 ||
          fp32.max_abs > 1e-2 || bf.max_abs > 5e-2 || fp32.cosine < 0.999 || bf.cosine < 0.999) {
        correct = false;
      }
    }
  }

  // Launch counter stays off the unchanged routes.
  candidate.reset();
  {
    const int k = 4096, n = 2048, rows = 64;
    require_ok("count narrow", call_linear(candidate, d_x, d_packed, d_scale, d_out, 1, rows, k, n, 2, d_ws_c,
                                           candidate.workspace(rows, k, n, 2)));
    require_ok("count mode0", call_linear(candidate, d_x, d_packed, d_scale, d_out, 1, 2048, k, n, 0, d_ws_c,
                                          candidate.workspace(2048, k, n, 0)));
    CHECK(cudaStreamSynchronize(g_stream));
    const int leaks = candidate.count();
    both("launch leaks on unchanged routes " + std::to_string(leaks));
    if (leaks) correct = false;
  }

  auto median = [](std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };
  cudaEvent_t ev0, ev1;
  CHECK(cudaEventCreate(&ev0));
  CHECK(cudaEventCreate(&ev1));
  constexpr int kIters = 51;
  auto time_ms = [&](auto&& fn) {
    fn();
    CHECK(cudaStreamSynchronize(g_stream));
    std::vector<float> samples;
    samples.reserve(kIters);
    for (int i = 0; i < kIters; ++i) {
      CHECK(cudaEventRecord(ev0, g_stream));
      fn();
      CHECK(cudaEventRecord(ev1, g_stream));
      CHECK(cudaEventSynchronize(ev1));
      float ms = 0;
      CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
      samples.push_back(ms);
    }
    return median(samples);
  };

  // Repack both shapes' weights; perf uses the fc weight left... reload per shape inside.
  // Time 1 fc, 1 kv, and the 1 fc + 5 kv wave. Weights differ, so the wave rebinds pointers.
  uint8_t *packed_fc = nullptr, *packed_kv = nullptr;
  float *scale_fc = nullptr, *scale_kv = nullptr;
  uint16_t *w_fc = nullptr, *w_kv = nullptr;
  CHECK(cudaMalloc(&packed_fc, size_t(4096) * 20480));
  CHECK(cudaMalloc(&scale_fc, size_t(4096) * (20480 / 128) * sizeof(float)));
  CHECK(cudaMalloc(&w_fc, size_t(4096) * 20480 * sizeof(uint16_t)));
  CHECK(cudaMalloc(&packed_kv, size_t(2048) * 4096));
  CHECK(cudaMalloc(&scale_kv, size_t(2048) * (4096 / 128) * sizeof(float)));
  CHECK(cudaMalloc(&w_kv, size_t(2048) * 4096 * sizeof(uint16_t)));
  auto pack_shape = [&](uint16_t* w, uint8_t* packed, float* scale, int k, int n, int seed) {
    std::mt19937 rng(seed);
    std::vector<uint16_t> bits(size_t(n) * k);
    fill_bf16(rng, bits, 0.02f);
    CHECK(cudaMemcpy(w, bits.data(), bits.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
    require_ok("perf pack", candidate.pack(w, packed, scale, n, k, 0, g_stream));
  };
  pack_shape(w_fc, packed_fc, scale_fc, 20480, 4096, 11);
  pack_shape(w_kv, packed_kv, scale_kv, 4096, 2048, 12);
  {
    std::mt19937 rng(13);
    h_x.resize(size_t(2048) * 20480);
    fill_bf16(rng, h_x, 0.5f);
    CHECK(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
  }
  CHECK(cudaStreamSynchronize(g_stream));

  const size_t fc_c = candidate.workspace(2048, 20480, 4096, 2);
  const size_t fc_b = baseline.workspace(2048, 20480, 4096, 2);
  const size_t kv_c = candidate.workspace(2048, 4096, 2048, 2);
  const size_t kv_b = baseline.workspace(2048, 4096, 2048, 2);
  auto one = [&](Lib& lib, void* ws, size_t bytes, uint8_t* packed, float* scale, int k, int n) {
    require_ok("perf", lib.linear(d_x, packed, scale, d_bf, 0, 2048, k, n, 2, ws, bytes, g_stream));
  };
  const float fc_new = time_ms([&] { one(candidate, d_ws_c, fc_c, packed_fc, scale_fc, 20480, 4096); });
  const float fc_old = time_ms([&] { one(baseline, d_ws_b, fc_b, packed_fc, scale_fc, 20480, 4096); });
  const float kv_new = time_ms([&] { one(candidate, d_ws_c, kv_c, packed_kv, scale_kv, 4096, 2048); });
  const float kv_old = time_ms([&] { one(baseline, d_ws_b, kv_b, packed_kv, scale_kv, 4096, 2048); });
  auto wave = [&](Lib& lib, void* ws, size_t bfc, size_t bkv) {
    require_ok("wave fc", lib.linear(d_x, packed_fc, scale_fc, d_bf, 0, 2048, 20480, 4096, 2, ws, bfc, g_stream));
    for (int i = 0; i < 5; ++i)
      require_ok("wave kv", lib.linear(d_x, packed_kv, scale_kv, d_bf, 0, 2048, 4096, 2048, 2, ws, bkv, g_stream));
  };
  const float wave_new = time_ms([&] { wave(candidate, d_ws_c, fc_c, kv_c); });
  const float wave_old = time_ms([&] { wave(baseline, d_ws_b, fc_b, kv_b); });

  candidate.reset();
  one(candidate, d_ws_c, fc_c, packed_fc, scale_fc, 20480, 4096);
  CHECK(cudaStreamSynchronize(g_stream));
  const int fc_launches = candidate.count();
  candidate.reset();
  one(candidate, d_ws_c, kv_c, packed_kv, scale_kv, 4096, 2048);
  CHECK(cudaStreamSynchronize(g_stream));
  const int kv_launches = candidate.count();
  candidate.reset();
  wave(candidate, d_ws_c, fc_c, kv_c);
  CHECK(cudaStreamSynchronize(g_stream));
  const int wave_launches = candidate.count();

  const int fc_old_k = old_kernels(2048, 4096, 20480, sms);
  const int kv_old_k = old_kernels(2048, 2048, 4096, sms);
  const int wave_old_k = fc_old_k + 5 * kv_old_k;
  char line[256];
  std::snprintf(line, sizeof(line), "perf fc old_ms %.4f new_ms %.4f speedup %.3f old_kernels %d new_kernels %d", fc_old,
                fc_new, fc_old / fc_new, fc_old_k, fc_launches);
  both(line);
  std::snprintf(line, sizeof(line), "perf kv old_ms %.4f new_ms %.4f speedup %.3f old_kernels %d new_kernels %d", kv_old,
                kv_new, kv_old / kv_new, kv_old_k, kv_launches);
  both(line);
  std::snprintf(line, sizeof(line),
                "perf wave old_ms %.4f new_ms %.4f speedup %.3f old_kernels %d new_kernels %d", wave_old, wave_new,
                wave_old / wave_new, wave_old_k, wave_launches);
  both(line);
  const bool fast = wave_old / wave_new >= 2.0f;
  both(std::string("stop_bar ") + (fast ? "PASS" : "FAIL") + " speedup " + std::to_string(wave_old / wave_new));
  both(std::string("correctness ") + (correct ? "PASS" : "FAIL"));
  if (fc_launches != 2 || kv_launches != 2 || wave_launches != 12) {
    both("FAIL launch count");
    correct = false;
  }
  return correct ? 0 : 1;
}

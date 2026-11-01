// p10 WP1 selftest: in-place graph argument update through the driver API.
//
// The graph mixes a runtime kernel (<<<>>> host stub) with a kernel launched
// from a driver-loaded library (cuLibraryLoadData + cuLaunchKernel), which is
// the shape of the CuTe-DSL AOT score overlay in V4.1 index selection: the
// runtime cudaGraphKernelNodeGetParams cannot read that node (C1). Checks:
//   1. pointer update of both nodes; replay == eager (host-replicated, exact)
//   2. 4-byte scalar update on the driver node
//   3. mislabelled kinds refused before any node is touched
//   4. forced failure at each driver stage clears the CUDA last-error
//   5. a value absent from the graph is reported not-found (found[i] == 0)
// Exit 77 (skip) without a CUDA device.
#include "cuteafd_native.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
  do {                                                                           \
    if (!(cond)) {                                                               \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
      ++g_failures;                                                              \
    }                                                                            \
  } while (0)

#define REQUIRE_CUDA(expr)                                                       \
  do {                                                                           \
    const cudaError_t e_ = (expr);                                               \
    if (e_ != cudaSuccess) {                                                     \
      std::fprintf(stderr, "FATAL %s:%d: %s -> %s\n", __FILE__, __LINE__, #expr, \
                   cudaGetErrorString(e_));                                      \
      return 1;                                                                  \
    }                                                                            \
  } while (0)

#define REQUIRE_CU(expr)                                                         \
  do {                                                                           \
    const CUresult r_ = (expr);                                                  \
    if (r_ != CUDA_SUCCESS) {                                                    \
      std::fprintf(stderr, "FATAL %s:%d: %s -> %d\n", __FILE__, __LINE__, #expr, \
                   static_cast<int>(r_));                                        \
      return 1;                                                                  \
    }                                                                            \
  } while (0)

// out[i] = in[i] * k for i < n. Hand-written PTX, JIT-compiled by the driver
// (validated offline with ptxas). Parameters: u64, u64, f32, u32.
const char kDriverKernelPtx[] = R"PTX(
.version 8.0
.target sm_80
.address_size 64

.visible .entry wp1_scale(
  .param .u64 wp1_in,
  .param .u64 wp1_out,
  .param .f32 wp1_k,
  .param .u32 wp1_n)
{
  .reg .pred %p<2>;
  .reg .b32 %r<6>;
  .reg .f32 %f<4>;
  .reg .b64 %rd<8>;

  ld.param.u64 %rd1, [wp1_in];
  ld.param.u64 %rd2, [wp1_out];
  ld.param.f32 %f1, [wp1_k];
  ld.param.u32 %r1, [wp1_n];
  mov.u32 %r2, %ctaid.x;
  mov.u32 %r3, %ntid.x;
  mov.u32 %r4, %tid.x;
  mad.lo.s32 %r5, %r2, %r3, %r4;
  setp.ge.u32 %p1, %r5, %r1;
  @%p1 bra WP1_DONE;
  cvta.to.global.u64 %rd3, %rd1;
  cvta.to.global.u64 %rd4, %rd2;
  mul.wide.u32 %rd5, %r5, 4;
  add.s64 %rd6, %rd3, %rd5;
  ld.global.f32 %f2, [%rd6];
  mul.rn.f32 %f3, %f2, %f1;
  add.s64 %rd7, %rd4, %rd5;
  st.global.f32 [%rd7], %f3;
WP1_DONE:
  ret;
}
)PTX";

__global__ void runtime_scale(const float* in, float* out, float k, unsigned n) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { out[i] = __fmul_rn(in[i], k); }
}

constexpr unsigned kN = 1000;
constexpr unsigned kThreads = 128;
constexpr unsigned kBlocks = (kN + kThreads - 1) / kThreads;

std::string last_error() {
  char buffer[1024] = {};
  cuteafd_last_error(buffer, sizeof(buffer));
  return buffer;
}

uint64_t bits(float value) {
  uint32_t out = 0;
  std::memcpy(&out, &value, sizeof(out));
  return out;
}

// Seeds a (non-sticky) runtime last-error so a test can prove a failure path
// clears it: without the clear, cudaGetLastError would still return it.
bool seed_runtime_error() {
  (void)cudaSetDevice(-1);
  return cudaPeekAtLastError() != cudaSuccess;
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    (void)cudaGetLastError();
    std::printf("SKIP: no CUDA device\n");
    return 77;
  }
  REQUIRE_CUDA(cudaSetDevice(0));
  REQUIRE_CUDA(cudaFree(nullptr));

  CUlibrary library = nullptr;
  REQUIRE_CU(cuLibraryLoadData(&library, kDriverKernelPtx, nullptr, nullptr, 0,
                               nullptr, nullptr, 0));
  CUkernel kernel = nullptr;
  REQUIRE_CU(cuLibraryGetKernel(&kernel, library, "wp1_scale"));

  std::vector<float> host_a(kN), host_b(kN);
  for (unsigned i = 0; i < kN; ++i) {
    host_a[i] = 0.25f * static_cast<float>(i) + 1.0f;
    host_b[i] = -0.5f * static_cast<float>(i) + 3.0f;
  }
  const size_t bytes = kN * sizeof(float);
  float *in_a, *mid_a, *out_a, *in_b, *mid_b, *out_b;
  REQUIRE_CUDA(cudaMalloc(&in_a, bytes));
  REQUIRE_CUDA(cudaMalloc(&mid_a, bytes));
  REQUIRE_CUDA(cudaMalloc(&out_a, bytes));
  REQUIRE_CUDA(cudaMalloc(&in_b, bytes));
  REQUIRE_CUDA(cudaMalloc(&mid_b, bytes));
  REQUIRE_CUDA(cudaMalloc(&out_b, bytes));
  REQUIRE_CUDA(cudaMemcpy(in_a, host_a.data(), bytes, cudaMemcpyHostToDevice));
  REQUIRE_CUDA(cudaMemcpy(in_b, host_b.data(), bytes, cudaMemcpyHostToDevice));
  REQUIRE_CUDA(cudaMemset(out_a, 0, bytes));
  REQUIRE_CUDA(cudaMemset(out_b, 0, bytes));
  cudaStream_t stream = nullptr;
  REQUIRE_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  const float k_runtime = 2.0f;
  float k_driver = 3.0f;
  unsigned n = kN;
  CHECK(cuteafd_cuda_graph_begin_capture(stream) == CUTEAFD_STATUS_OK);
  runtime_scale<<<kBlocks, kThreads, 0, stream>>>(in_a, mid_a, k_runtime, n);
  {
    void* args[] = {&mid_a, &out_a, &k_driver, &n};
    REQUIRE_CU(cuLaunchKernel(reinterpret_cast<CUfunction>(kernel), kBlocks, 1, 1, kThreads, 1, 1,
                              0, reinterpret_cast<CUstream>(stream), args, nullptr));
  }
  cuteafd_cuda_graph_capture_info_t info = {};
  CHECK(cuteafd_cuda_graph_end_capture_retained(stream, &info) == CUTEAFD_STATUS_OK);
  CHECK(info.kernel_node_count == 2);
  if (info.graph == nullptr || info.graph_exec == nullptr) {
    std::fprintf(stderr, "FATAL: capture failed: %s\n", last_error().c_str());
    return 1;
  }

  // C1 evidence (informational): how many kernel nodes the RUNTIME view fails on.
  {
    size_t count = 0;
    cudaGraphGetNodes(static_cast<cudaGraph_t>(info.graph), nullptr, &count);
    std::vector<cudaGraphNode_t> nodes(count);
    cudaGraphGetNodes(static_cast<cudaGraph_t>(info.graph), nodes.data(), &count);
    int runtime_failures = 0;
    for (cudaGraphNode_t node : nodes) {
      cudaKernelNodeParams params = {};
      if (cudaGraphKernelNodeGetParams(node, &params) != cudaSuccess) { ++runtime_failures; }
    }
    (void)cudaGetLastError();
    std::printf("INFO runtime cudaGraphKernelNodeGetParams failures=%d of %zu nodes\n",
                runtime_failures, count);
  }

  const auto replay_and_read = [&](float* out, std::vector<float>* host) {
    if (cuteafd_cuda_graph_launch(info.graph_exec, stream) != CUTEAFD_STATUS_OK) { return false; }
    if (cudaStreamSynchronize(stream) != cudaSuccess) { return false; }
    host->resize(kN);
    return cudaMemcpy(host->data(), out, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
  };
  std::vector<float> got;

  // 1. Pointer update across both nodes; replay == eager (host-replicated).
  {
    const uint64_t old_values[] = {reinterpret_cast<uint64_t>(in_a), reinterpret_cast<uint64_t>(mid_a),
                                   reinterpret_cast<uint64_t>(out_a)};
    const uint64_t new_values[] = {reinterpret_cast<uint64_t>(in_b), reinterpret_cast<uint64_t>(mid_b),
                                   reinterpret_cast<uint64_t>(out_b)};
    const uint8_t widths[] = {8, 8, 8};
    const uint8_t kinds[] = {1, 1, 1};
    uint8_t found[] = {0, 0, 0};
    const cuteafd_status_t status = cuteafd_cuda_graph_exec_update_pointers(
        info.graph_exec, info.graph, old_values, new_values, widths, kinds, found, 3);
    CHECK(status == CUTEAFD_STATUS_OK);
    if (status != CUTEAFD_STATUS_OK) { std::fprintf(stderr, "  %s\n", last_error().c_str()); }
    CHECK(found[0] == 1 && found[1] == 1 && found[2] == 1);
    CHECK(replay_and_read(out_b, &got));
    bool equal = true;
    for (unsigned i = 0; i < kN; ++i) {
      const float expected = (host_b[i] * k_runtime) * k_driver;
      equal = equal && got[i] == expected;
    }
    CHECK(equal);
    std::vector<float> untouched(kN);
    CHECK(cudaMemcpy(untouched.data(), out_a, bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
    bool zero = true;
    for (float v : untouched) { zero = zero && v == 0.0f; }
    CHECK(zero);
  }

  // 2. 4-byte scalar on the driver-loaded node: k 3.0 -> 5.0.
  {
    const uint64_t old_values[] = {bits(3.0f)};
    const uint64_t new_values[] = {bits(5.0f)};
    const uint8_t widths[] = {4};
    const uint8_t kinds[] = {0};
    uint8_t found[] = {0};
    CHECK(cuteafd_cuda_graph_exec_update_pointers(info.graph_exec, info.graph, old_values,
                                                     new_values, widths, kinds, found, 1)
          == CUTEAFD_STATUS_OK);
    CHECK(found[0] == 1);
    CHECK(replay_and_read(out_b, &got));
    bool equal = true;
    for (unsigned i = 0; i < kN; ++i) { equal = equal && got[i] == (host_b[i] * k_runtime) * 5.0f; }
    CHECK(equal);
  }

  // 3. Mislabelled kinds are refused before any node is touched.
  {
    uint8_t found[] = {0};
    const uint8_t width8[] = {8};
    // scalar kind holding a changing device pointer
    const uint64_t dev_old[] = {reinterpret_cast<uint64_t>(in_b)};
    const uint64_t dev_new[] = {reinterpret_cast<uint64_t>(in_a)};
    const uint8_t scalar[] = {0};
    CHECK(seed_runtime_error());
    CHECK(cuteafd_cuda_graph_exec_update_pointers(info.graph_exec, info.graph, dev_old, dev_new,
                                                     width8, scalar, found, 1)
          == CUTEAFD_STATUS_INVALID_ARGUMENT);
    CHECK(last_error().find("aliases a CUDA pointer") != std::string::npos);
    CHECK(cudaGetLastError() == cudaSuccess);
    // pointer kind holding a host (non-CUDA) value
    uint64_t host_word = 0;
    const uint64_t host_old[] = {reinterpret_cast<uint64_t>(&host_word)};
    const uint64_t host_new[] = {reinterpret_cast<uint64_t>(&host_word) + 8};
    const uint8_t pointer[] = {1};
    CHECK(cuteafd_cuda_graph_exec_update_pointers(info.graph_exec, info.graph, host_old, host_new,
                                                     width8, pointer, found, 1)
          == CUTEAFD_STATUS_INVALID_ARGUMENT);
    CHECK(last_error().find("not a CUDA pointer") != std::string::npos);
    // nothing was rewritten: replay still reads in_b
    CHECK(replay_and_read(out_b, &got));
    bool equal = true;
    for (unsigned i = 0; i < kN; ++i) { equal = equal && got[i] == (host_b[i] * k_runtime) * 5.0f; }
    CHECK(equal);
  }

  // 4. Forced failure at each driver stage: status INTERNAL_ERROR, message
  //    names the driver call, and the CUDA last-error is clear afterwards; the
  //    next runtime launch's status check sees success (the p8 crash shape).
  const char* stage_names[] = {"", "cuGraphKernelNodeGetParams", "cuGraphKernelNodeSetParams",
                               "cuGraphExecKernelNodeSetParams"};
  for (int stage = 1; stage <= 3; ++stage) {
    const uint64_t old_values[] = {reinterpret_cast<uint64_t>(in_b)};
    const uint64_t new_values[] = {reinterpret_cast<uint64_t>(in_a)};
    const uint8_t widths[] = {8};
    const uint8_t kinds[] = {1};
    uint8_t found[] = {0};
    CHECK(seed_runtime_error());
    cuteafd_cuda_graph_update_pointers_inject_fault(stage);
    CHECK(cuteafd_cuda_graph_exec_update_pointers(info.graph_exec, info.graph, old_values,
                                                     new_values, widths, kinds, found, 1)
          == CUTEAFD_STATUS_INTERNAL_ERROR);
    CHECK(last_error().find(stage_names[stage]) != std::string::npos);
    CHECK(cudaGetLastError() == cudaSuccess);
    runtime_scale<<<kBlocks, kThreads, 0, stream>>>(in_a, mid_a, k_runtime, n);
    CHECK(cudaGetLastError() == cudaSuccess);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
  }
  cuteafd_cuda_graph_update_pointers_inject_fault(0);

  // 5. A value that is not in the graph is reported not-found, not an error.
  {
    const uint64_t old_values[] = {reinterpret_cast<uint64_t>(out_a) + 64};
    const uint64_t new_values[] = {reinterpret_cast<uint64_t>(out_b) + 64};
    const uint8_t widths[] = {8};
    const uint8_t kinds[] = {1};
    uint8_t found[] = {1};
    CHECK(cuteafd_cuda_graph_exec_update_pointers(info.graph_exec, info.graph, old_values,
                                                     new_values, widths, kinds, found, 1)
          == CUTEAFD_STATUS_OK);
    CHECK(found[0] == 0);
  }

  CHECK(cuteafd_cuda_graph_exec_destroy(info.graph_exec) == CUTEAFD_STATUS_OK);
  CHECK(cuteafd_cuda_graph_destroy(info.graph) == CUTEAFD_STATUS_OK);
  cudaStreamDestroy(stream);
  cudaFree(in_a);
  cudaFree(mid_a);
  cudaFree(out_a);
  cudaFree(in_b);
  cudaFree(mid_b);
  cudaFree(out_b);
  cuLibraryUnload(library);
  if (g_failures != 0) {
    std::fprintf(stderr, "cuda_graph_update_selftest: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("cuda_graph_update_selftest: OK\n");
  return 0;
}

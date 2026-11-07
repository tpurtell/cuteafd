// Named CUDA owners. Run in the matching toolchain image on idle, locked GPUs.
// nvcc -O2 -o probe cuda-context-probe.cu -lcublas -lcublasLt -arch=sm_120
// probe handles|lt|lt-with-blas [device=0] [count=4]
// probe mappings [device0=0] [device1=1]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <sys/mman.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

static void cuda_check(cudaError_t status, const char* call) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", call, cudaGetErrorString(status));
    std::exit(1);
  }
}
static void blas_check(cublasStatus_t status, const char* call) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "%s: cuBLAS status %d\n", call, int(status));
    std::exit(1);
  }
}
#define CUDA(call) cuda_check((call), #call)
#define BLAS(call) blas_check((call), #call)

static long long mem_available_kb() {
  FILE* f = std::fopen("/proc/meminfo", "r");
  char line[256];
  long long value = -1;
  while (f && std::fgets(line, sizeof(line), f)) {
    if (std::sscanf(line, "MemAvailable: %lld kB", &value) == 1) break;
  }
  if (f) std::fclose(f);
  return value;
}
static size_t used(int device) {
  CUDA(cudaSetDevice(device));
  CUDA(cudaDeviceSynchronize());
  size_t free = 0, total = 0;
  CUDA(cudaMemGetInfo(&free, &total));
  return total - free;
}
static void identity(int device) {
  CUDA(cudaSetDevice(device));
  cudaDeviceProp prop{};
  CUDA(cudaGetDeviceProperties(&prop, device));
  std::printf("{\"mode\":\"identity\",\"device\":%d,\"name\":\"%s\",\"uuid\":\"GPU-", device, prop.name);
  for (int i = 0; i < 16; ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10) std::putchar('-');
    std::printf("%02x", (unsigned char)prop.uuid.bytes[i]);
  }
  std::printf("\"}\n");
}
static void handles(int device, int count) {
  CUDA(cudaSetDevice(device));
  CUDA(cudaFree(nullptr));
  cudaDeviceProp prop{};
  CUDA(cudaGetDeviceProperties(&prop, device));
  int driver = 0;
  CUDA(cudaDriverGetVersion(&driver));
  // Allocate disjoint A/B/C before sampling: buffers are not handle residency.
  float* p = nullptr;
  CUDA(cudaMalloc(&p, 3 * 64 * 64 * sizeof(float)));
  CUDA(cudaMemset(p, 0, 3 * 64 * 64 * sizeof(float)));
  const auto baseline = used(device);
  const auto host_baseline = mem_available_kb();
  std::vector<cublasHandle_t> owners(count);
  const float alpha = 1, beta = 0;
  for (int i = 0; i < count; ++i) {
    const auto before = used(device);
    BLAS(cublasCreate(&owners[i]));
    const auto created = used(device);
    BLAS(cublasSgemm(owners[i], CUBLAS_OP_N, CUBLAS_OP_N, 64, 64, 64,
        &alpha, p, 64, p + 4096, 64, &beta, p + 8192, 64));
    const auto after = used(device);
    std::printf("{\"mode\":\"handle\",\"device\":%d,\"sm\":\"sm_%d%d\",\"driver\":%d,"
        "\"count\":%d,\"create_bytes\":%lld,\"gemm_bytes\":%lld,\"delta_bytes\":%lld,"
        "\"total_bytes\":%lld,\"host_total_bytes\":%lld}\n", device, prop.major, prop.minor,
        driver, i + 1, (long long)created - before, (long long)after - created,
        (long long)after - before, (long long)after - baseline,
        1024 * (host_baseline - mem_available_kb()));
    std::fflush(stdout);
  }
  for (auto owner : owners) BLAS(cublasDestroy(owner));
  CUDA(cudaFree(p));
}
static void lt_handles(int device, int count, bool with_blas) {
  CUDA(cudaSetDevice(device));
  CUDA(cudaFree(nullptr));
  cudaDeviceProp prop{};
  CUDA(cudaGetDeviceProperties(&prop, device));
  int driver = 0;
  CUDA(cudaDriverGetVersion(&driver));
  constexpr int n = 4096, k = 4096, rows = 16;
  constexpr size_t weight_bytes = size_t(n) * k * 2;
  constexpr size_t input_bytes = size_t(k) * rows * 2;
  constexpr size_t output_bytes = size_t(n) * rows * 2;
  constexpr size_t workspace_bytes = 4 << 20;
  void *weight = nullptr, *input = nullptr, *output = nullptr;
  CUDA(cudaMalloc(&weight, weight_bytes));
  CUDA(cudaMalloc(&input, input_bytes));
  CUDA(cudaMalloc(&output, output_bytes));
  CUDA(cudaMemset(weight, 0, weight_bytes));
  CUDA(cudaMemset(input, 0, input_bytes));
  const auto before_blas = used(device);
  cublasHandle_t blas = nullptr;
  if (with_blas) BLAS(cublasCreate(&blas));
  const auto baseline = used(device);
  std::vector<cublasLtHandle_t> owners(count);
  std::vector<void*> workspaces(count);
  std::vector<cublasLtMatmulDesc_t> operations(count);
  std::vector<cublasLtMatrixLayout_t> weights(count), inputs(count), outputs(count);
  const float alpha = 1, beta = 0;
  for (int i = 0; i < count; ++i) {
    const auto before = used(device);
    BLAS(cublasLtCreate(&owners[i]));
    const auto created = used(device);
    CUDA(cudaMalloc(&workspaces[i], workspace_bytes));
    const auto workspace = used(device);
    BLAS(cublasLtMatmulDescCreate(&operations[i], CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t transpose = CUBLAS_OP_T;
    BLAS(cublasLtMatmulDescSetAttribute(operations[i], CUBLASLT_MATMUL_DESC_TRANSA,
        &transpose, sizeof(transpose)));
    BLAS(cublasLtMatrixLayoutCreate(&weights[i], CUDA_R_16BF, k, n, k));
    BLAS(cublasLtMatrixLayoutCreate(&inputs[i], CUDA_R_16BF, k, rows, k));
    BLAS(cublasLtMatrixLayoutCreate(&outputs[i], CUDA_R_16BF, n, rows, n));
    cublasLtMatmulPreference_t preference = nullptr;
    BLAS(cublasLtMatmulPreferenceCreate(&preference));
    BLAS(cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)));
    cublasLtMatmulHeuristicResult_t candidate{};
    int found = 0;
    BLAS(cublasLtMatmulAlgoGetHeuristic(owners[i], operations[i], weights[i], inputs[i],
        outputs[i], outputs[i], preference, 1, &candidate, &found));
    if (found != 1 || candidate.state != CUBLAS_STATUS_SUCCESS) {
      std::fprintf(stderr, "no BF16 M16/K4096/N4096 Lt algorithm\n");
      std::exit(1);
    }
    BLAS(cublasLtMatmulPreferenceDestroy(preference));
    const auto planned = used(device);
    BLAS(cublasLtMatmul(owners[i], operations[i], &alpha, weight, weights[i], input,
        inputs[i], &beta, output, outputs[i], output, outputs[i], &candidate.algo,
        workspaces[i], workspace_bytes, nullptr));
    const auto after = used(device);
    std::printf("{\"mode\":\"lt\",\"with_blas\":%s,\"device\":%d,\"sm\":\"sm_%d%d\","
        "\"driver\":%d,\"count\":%d,\"existing_blas_bytes\":%lld,\"create_bytes\":%lld,"
        "\"workspace_bytes\":%lld,\"plan_bytes\":%lld,\"matmul_bytes\":%lld,"
        "\"total_bytes\":%lld,\"algorithm_workspace_bytes\":%zu}\n",
        with_blas ? "true" : "false", device, prop.major, prop.minor, driver, i + 1,
        (long long)baseline - before_blas, (long long)created - before,
        (long long)workspace - created, (long long)planned - workspace,
        (long long)after - planned, (long long)after - baseline, candidate.workspaceSize);
    std::fflush(stdout);
  }
  for (int i = 0; i < count; ++i) {
    BLAS(cublasLtMatmulDescDestroy(operations[i]));
    BLAS(cublasLtMatrixLayoutDestroy(weights[i]));
    BLAS(cublasLtMatrixLayoutDestroy(inputs[i]));
    BLAS(cublasLtMatrixLayoutDestroy(outputs[i]));
    BLAS(cublasLtDestroy(owners[i]));
    CUDA(cudaFree(workspaces[i]));
  }
  if (blas) BLAS(cublasDestroy(blas));
  CUDA(cudaFree(weight));
  CUDA(cudaFree(input));
  CUDA(cudaFree(output));
}
static void mappings(int device0, int device1) {
  if (device0 == device1) {
    std::fprintf(stderr, "mappings requires two distinct GPUs\n");
    std::exit(2);
  }
  for (int device : {device0, device1}) {
    CUDA(cudaSetDevice(device));
    CUDA(cudaFree(nullptr));
  }
  const size_t gib = size_t(1) << 30;
  for (const char* mode : {"hostalloc", "hostregister", "malloc"}) {
    for (size_t size : {gib, 4 * gib, 16 * gib, 32 * gib}) {
      const auto before0 = used(device0), before1 = used(device1);
      const auto host_before = mem_available_kb();
      if (std::strcmp(mode, "malloc") != 0 && host_before * 1024 < (long long)(size + 8 * gib)) {
        std::fprintf(stderr, "insufficient host MemAvailable for %s %zu\n", mode, size);
        std::exit(1);
      }
      CUDA(cudaSetDevice(device0));
      void* p = nullptr;
      if (std::strcmp(mode, "hostalloc") == 0) {
        CUDA(cudaHostAlloc(&p, size, cudaHostAllocPortable | cudaHostAllocMapped));
      } else if (std::strcmp(mode, "hostregister") == 0) {
        p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) { std::perror("mmap"); std::exit(1); }
        CUDA(cudaHostRegister(p, size, cudaHostRegisterPortable));
      } else {
        CUDA(cudaMalloc(&p, size));
      }
      // Resolve the portable mapping on both contexts, as prefix/transport do.
      if (std::strcmp(mode, "malloc") != 0) {
        for (int device : {device0, device1}) {
          CUDA(cudaSetDevice(device));
          void* alias = nullptr;
          CUDA(cudaHostGetDevicePointer(&alias, p, 0));
        }
      }
      const auto after0 = used(device0), after1 = used(device1);
      std::printf("{\"mode\":\"%s\",\"bytes\":%zu,\"devices\":[%d,%d],"
          "\"delta_bytes\":[%lld,%lld],\"driver_bytes\":[%lld,%lld],\"host_bytes\":%lld}\n",
          mode, size, device0, device1, (long long)after0 - before0, (long long)after1 - before1,
          (long long)after0 - before0 - (std::strcmp(mode, "malloc") == 0 ? size : 0),
          (long long)after1 - before1, 1024 * (host_before - mem_available_kb()));
      std::fflush(stdout);
      CUDA(cudaSetDevice(device0));
      if (std::strcmp(mode, "hostalloc") == 0) CUDA(cudaFreeHost(p));
      else if (std::strcmp(mode, "hostregister") == 0) {
        CUDA(cudaHostUnregister(p));
        if (munmap(p, size)) { std::perror("munmap"); std::exit(1); }
      } else CUDA(cudaFree(p));
    }
  }
}
int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "handles";
  const int device = argc > 2 ? std::atoi(argv[2]) : 0;
  identity(device);
  if (std::strcmp(mode, "mappings") == 0) identity(argc > 3 ? std::atoi(argv[3]) : 1);
  const int last = argc > 3 ? std::atoi(argv[3]) : (std::strcmp(mode, "mappings") == 0 ? 1 : 4);
  if (std::strcmp(mode, "handles") == 0 && last > 0 && last <= 64) handles(device, last);
  else if ((std::strcmp(mode, "lt") == 0 || std::strcmp(mode, "lt-with-blas") == 0)
      && last > 0 && last <= 64) lt_handles(device, last, std::strcmp(mode, "lt-with-blas") == 0);
  else if (std::strcmp(mode, "mappings") == 0) mappings(device, last);
  else { std::fprintf(stderr, "usage: probe handles|lt|lt-with-blas [device] [count] | mappings [device0] [device1]\n"); return 2; }
}

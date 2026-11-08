// Named CUDA owners. Run in the matching toolchain image on idle, locked GPUs.
// nvcc -O2 -o probe cuda-context-probe.cu -lcublas -lcublasLt -lcuda -arch=sm_120
// probe handles|lt|lt-with-blas [device=0] [count=4] [prewarm]
// probe mappings [device0=0] [device1=1]
// probe allocations [device=0] [retained_count=32]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <sys/mman.h>
#include <cuda.h>
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
  const char* loading = std::getenv("CUDA_MODULE_LOADING");
  std::printf("\",\"module_loading\":\"%s\"}\n", loading ? loading : "default");
}
static void prewarm_sgemm(int device) {
  const auto before = used(device);
  const auto host_before = mem_available_kb();
  float* p = nullptr;
  CUDA(cudaMalloc(&p, 3 * 64 * 64 * sizeof(float)));
  CUDA(cudaMemset(p, 0, 3 * 64 * 64 * sizeof(float)));
  const auto buffered = used(device);
  cublasHandle_t handle = nullptr;
  BLAS(cublasCreate(&handle));
  const auto created = used(device);
  const float alpha = 1, beta = 0;
  BLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 64, 64, 64,
      &alpha, p, 64, p + 4096, 64, &beta, p + 8192, 64));
  const auto multiplied = used(device);
  BLAS(cublasDestroy(handle));
  CUDA(cudaFree(p));
  const auto after = used(device);
  std::printf("{\"mode\":\"prewarm\",\"device\":%d,\"operation\":\"sgemm_64\","
      "\"create_bytes\":%lld,\"gemm_bytes\":%lld,\"retained_bytes\":%lld,"
      "\"host_retained_bytes\":%lld}\n", device, (long long)created - buffered,
      (long long)multiplied - created, (long long)after - before,
      1024 * (host_before - mem_available_kb()));
  std::fflush(stdout);
}
static void handles(int device, int count, bool prewarm) {
  CUDA(cudaSetDevice(device));
  CUDA(cudaFree(nullptr));
  cudaDeviceProp prop{};
  CUDA(cudaGetDeviceProperties(&prop, device));
  int driver = 0;
  CUDA(cudaDriverGetVersion(&driver));
  if (prewarm) prewarm_sgemm(device);
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
static void lt_handles(int device, int count, bool with_blas, bool prewarm) {
  CUDA(cudaSetDevice(device));
  CUDA(cudaFree(nullptr));
  cudaDeviceProp prop{};
  CUDA(cudaGetDeviceProperties(&prop, device));
  int driver = 0;
  CUDA(cudaDriverGetVersion(&driver));
  if (prewarm) prewarm_sgemm(device);
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
  const auto host_before_blas = mem_available_kb();
  cublasHandle_t blas = nullptr;
  if (with_blas) BLAS(cublasCreate(&blas));
  const auto baseline = used(device);
  const auto host_baseline = mem_available_kb();
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
        "\"total_bytes\":%lld,\"algorithm_workspace_bytes\":%zu,"
        "\"host_existing_blas_bytes\":%lld,\"host_total_bytes\":%lld}\n",
        with_blas ? "true" : "false", device, prop.major, prop.minor, driver, i + 1,
        (long long)baseline - before_blas, (long long)created - before,
        (long long)workspace - created, (long long)planned - workspace,
        (long long)after - planned, (long long)after - baseline, candidate.workspaceSize,
        1024 * (host_before_blas - host_baseline), 1024 * (host_baseline - mem_available_kb()));
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
// One page write per lane: exercise the mapping without transfer-sized GPU work.
__global__ static void touch_mapping(unsigned char* p, size_t bytes) {
  for (size_t i = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 4096;
       i < bytes; i += size_t(gridDim.x) * blockDim.x * 4096) p[i] = 1;
}
static void chunks(int device, int peer) {
  const size_t mib = size_t(1) << 20, gib = size_t(1) << 30;
  std::vector<int> devices{device};
  if (peer >= 0 && peer != device) devices.push_back(peer);
  for (int d : devices) {
    CUDA(cudaSetDevice(d)); CUDA(cudaFree(nullptr));
    touch_mapping<<<1, 1>>>(nullptr, 0); CUDA(cudaDeviceSynchronize());
  }
  struct Shape { size_t count, bytes; };
  for (const auto shape : {Shape{1, 32*gib}, Shape{89, 368*mib},
                          Shape{512, 64*mib}, Shape{128, 256*mib}}) {
    if (mem_available_kb() * 1024 < (long long)(shape.count * shape.bytes + 8*gib)) {
      std::fprintf(stderr, "insufficient host memory for chunks\n"); std::exit(1);
    }
    std::vector<size_t> baseline;
    for (int d : devices) baseline.push_back(used(d));
    auto sample = [&](const char* stage, size_t count, size_t touched) {
      for (size_t j = 0; j < devices.size(); ++j) {
        size_t free = 0, total = 0;
        CUDA(cudaSetDevice(devices[j])); CUDA(cudaDeviceSynchronize());
        CUDA(cudaMemGetInfo(&free, &total));
        std::printf("{\"mode\":\"chunks\",\"allocator\":%d,\"device\":%d,"
            "\"chunk_bytes\":%zu,\"target_count\":%zu,\"count\":%zu,\"stage\":\"%s\","
            "\"touched_bytes\":%zu,\"free\":%zu,\"total\":%zu,\"delta_bytes\":%lld}\n",
            device, devices[j], shape.bytes, shape.count, count, stage, touched,
            free, total, (long long)(total-free) - baseline[j]);
      }
      std::fflush(stdout);
    };
    sample("before", 0, 0);
    std::vector<void*> pointers;
    for (size_t i = 0; i < shape.count; ++i) {
      CUDA(cudaSetDevice(device)); void* p = nullptr;
      CUDA(cudaHostAlloc(&p, shape.bytes, cudaHostAllocPortable | cudaHostAllocMapped));
      pointers.push_back(p);
      sample("allocated", i+1, 0);
    }
    for (int d : devices) {
      CUDA(cudaSetDevice(d));
      for (void* p : pointers) { void* alias = nullptr; CUDA(cudaHostGetDevicePointer(&alias, p, 0)); }
    }
    sample("aliases", shape.count, 0);
    for (size_t fraction : {size_t(4), size_t(1)}) {
      for (void* p : pointers) std::memset(p, 1, shape.bytes/fraction);
      sample("cpu_touch", shape.count, shape.count*shape.bytes/fraction);
    }
    for (int d : devices) {
      CUDA(cudaSetDevice(d));
      for (void* p : pointers) touch_mapping<<<256, 256>>>((unsigned char*)p, shape.bytes);
      CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
    }
    sample("gpu_touch", shape.count, shape.count*shape.bytes);
    CUDA(cudaSetDevice(device));
    for (void* p : pointers) CUDA(cudaFreeHost(p));
    sample("freed", 0, 0);
  }
}
static void driver_check(CUresult status, const char* call) {
  if (status != CUDA_SUCCESS) {
    const char* message = nullptr;
    cuGetErrorString(status, &message);
    std::fprintf(stderr, "%s: %s\n", call, message ? message : "driver error");
    std::exit(1);
  }
}
#define DRIVER(call) driver_check((call), #call)

static void allocation_pool(const char* stage, int device) {
  cudaMemPool_t pool = nullptr;
  CUDA(cudaDeviceGetMemPool(&pool, device));
  uint64_t reserved = 0, allocated = 0;
  CUDA(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved));
  CUDA(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &allocated));
  std::printf("{\"mode\":\"allocation_pool\",\"stage\":\"%s\",\"reserved_bytes\":%llu,"
      "\"used_bytes\":%llu}\n", stage, (unsigned long long)reserved, (unsigned long long)allocated);
}

static void allocation_batch(int device, const char* name, const std::vector<size_t>& sizes, int count) {
  constexpr size_t limit = size_t(8) << 30;
  size_t per = 0;
  for (size_t size : sizes) {
    if (size == 0 || size > limit - per) { std::fprintf(stderr, "invalid allocation batch\n"); std::exit(2); }
    per += size;
  }
  if (size_t(count) > limit / per) { std::fprintf(stderr, "allocation batch exceeds 8 GiB\n"); std::exit(2); }
  const auto baseline = used(device);
  std::vector<void*> owners;
  owners.reserve(sizes.size() * count);
  size_t requested = 0;
  auto previous = baseline;
  for (int i = 0; i < count; ++i) {
    for (size_t size : sizes) {
      void* p = nullptr;
      CUDA(cudaMalloc(&p, size));
      owners.push_back(p);
      requested += size;
    }
    if (i < 3 || i + 1 == count || (i + 1) % (count > 64 ? 128 : 1) == 0) {
      const auto after = used(device);
      CUdeviceptr base = 0;
      size_t range = 0;
      DRIVER(cuMemGetAddressRange(&base, &range, (CUdeviceptr)owners.back()));
      std::printf("{\"mode\":\"allocation_batch\",\"case\":\"%s\",\"groups\":%d,"
          "\"allocations\":%zu,\"requested_bytes\":%zu,\"raw_delta_bytes\":%lld,"
          "\"raw_since_sample_bytes\":%lld,\"unclassified_bytes\":%lld,"
          "\"last_address\":%llu,\"last_address_range_bytes\":%zu}\n",
          name, i + 1, owners.size(), requested, (long long)after - baseline,
          (long long)after - previous, (long long)after - baseline - (long long)requested,
          (unsigned long long)base, range);
      previous = after;
      std::fflush(stdout);
    }
  }
  for (auto owner : owners) CUDA(cudaFree(owner));
  const auto after = used(device);
  std::printf("{\"mode\":\"allocation_released\",\"case\":\"%s\",\"retained_bytes\":%lld}\n",
      name, (long long)after - baseline);
}

static void allocations(int device, int count) {
  CUDA(cudaSetDevice(device));
  CUDA(cudaFree(nullptr));
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  size_t minimum = 0, recommended = 0;
  DRIVER(cuMemGetAllocationGranularity(&minimum, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  DRIVER(cuMemGetAllocationGranularity(&recommended, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  int driver = 0, runtime = 0;
  CUDA(cudaDriverGetVersion(&driver));
  CUDA(cudaRuntimeGetVersion(&runtime));
  std::printf("{\"mode\":\"allocation_granularity\",\"device\":%d,\"driver\":%d,\"runtime\":%d,"
      "\"vmm_minimum_bytes\":%zu,\"vmm_recommended_bytes\":%zu,\"batch_limit_bytes\":%zu}\n",
      device, driver, runtime, minimum, recommended, size_t(8) << 30);
  allocation_pool("before", device);
  constexpr size_t kib = 1024, mib = 1024 * kib;
  for (size_t size : {size_t(384), 14 * kib, 24 * kib, 64 * kib - 1, 64 * kib,
      64 * kib + 1, 576 * kib, mib - 1, mib, mib + 1, 2 * mib - 1, 2 * mib,
      2 * mib + 1, 3 * mib, 3136 * kib, 63 * mib, 64 * mib}) {
    char name[64];
    std::snprintf(name, sizeof(name), "size_%zu", size);
    allocation_batch(device, name, {size}, size < mib ? count * 32 : count);
  }
  allocation_batch(device, "k6_payload_suh_svh", {63 * mib, 14 * kib, 24 * kib}, count);
  allocation_batch(device, "small_large_interleaved", {14 * kib, 63 * mib, 24 * kib, 3 * mib, 384}, count);
  allocation_pool("after", device);
}

int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "handles";
  const int device = argc > 2 ? std::atoi(argv[2]) : 0;
  identity(device);
  if (std::strcmp(mode, "mappings") == 0) identity(argc > 3 ? std::atoi(argv[3]) : 1);
  const int last = argc > 3 ? std::atoi(argv[3]) :
      (std::strcmp(mode, "mappings") == 0 ? 1 : std::strcmp(mode, "allocations") == 0 ? 32 : 4);
  const bool prewarm = argc == 5 && std::strcmp(argv[4], "prewarm") == 0;
  if (argc > 5 || (argc == 5 && (!prewarm || std::strcmp(mode, "mappings") == 0))) {
    std::fprintf(stderr, "invalid prewarm argument\n");
    return 2;
  }
  if (std::strcmp(mode, "handles") == 0 && last > 0 && last <= 64) handles(device, last, prewarm);
  else if ((std::strcmp(mode, "lt") == 0 || std::strcmp(mode, "lt-with-blas") == 0)
      && last > 0 && last <= 64) lt_handles(device, last, std::strcmp(mode, "lt-with-blas") == 0, prewarm);
  else if (std::strcmp(mode, "mappings") == 0) mappings(device, last);
  else if (std::strcmp(mode, "allocations") == 0 && !prewarm && last > 0 && last <= 64) allocations(device, last);
  else if (std::strcmp(mode, "chunks") == 0 && !prewarm) chunks(device, argc > 3 ? last : -1);
  else { std::fprintf(stderr, "usage: probe handles|lt|lt-with-blas [device] [count] [prewarm] | mappings [device0] [device1] | allocations [device] [retained_count] | chunks [device] [peer=-1]\n"); return 2; }
}

// Per-arch CUDA context + cuBLAS cost for placement::inventory::ARCH_CONTEXTS. Prints one JSON line.
// Build: nvcc -O2 -o probe cuda-context-probe.cu -lcublas -arch=sm_120 (sm_121 on GB10); run on an idle device.
// GB10 has unified memory: read host_context_kb/host_cublas_kb (MemAvailable deltas), not cudaMemGetInfo.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h>
static long long mem_available_kb() {
  FILE* f = fopen("/proc/meminfo", "r"); char key[64]; long long v; char unit[8];
  while (f && fscanf(f, "%63s %lld %7s", key, &v, unit) == 3) if (!strcmp(key, "MemAvailable:")) { fclose(f); return v; }
  if (f) fclose(f); return -1;
}
int main(int argc, char** argv) {
  int dev = argc > 1 ? atoi(argv[1]) : 0;
  long long host0 = mem_available_kb();
  cudaSetDevice(dev);
  size_t f0 = 0, t0 = 0; cudaFree(0); cudaMemGetInfo(&f0, &t0);
  long long host1 = mem_available_kb();
  cublasHandle_t h; cublasCreate(&h);
  void* p; cudaMalloc(&p, 1 << 20); float a = 1, b = 0;
  cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, 64, 64, 64, &a, (float*)p, 64, (float*)p, 64, &b, (float*)p, 64);
  cudaDeviceSynchronize();
  size_t f1 = 0, t1 = 0; cudaMemGetInfo(&f1, &t1);
  long long host2 = mem_available_kb();
  cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
  int driver = 0; cudaDriverGetVersion(&driver);
  printf("{\"name\":\"%s\",\"sm\":\"sm_%d%d\",\"sms\":%d,\"total\":%zu,\"used_after_context\":%zu,"
         "\"used_after_cublas\":%zu,\"cublas_delta\":%zu,\"host_context_kb\":%lld,\"host_cublas_kb\":%lld,\"driver\":%d}\n",
         prop.name, prop.major, prop.minor, prop.multiProcessorCount, t0, t0 - f0, t1 - f1, (t1 - f1) - (t0 - f0) - (1 << 20),
         host0 - host1, host1 - host2, driver);
  return 0;
}

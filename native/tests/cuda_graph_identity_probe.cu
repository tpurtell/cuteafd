// Graph executable size/identity and concurrent-slot experiment; JSONL output.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

#define CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e)); \
  std::exit(1); } } while (0)
constexpr size_t MiB = 1024 * 1024;

template <int Bytes> struct Args {
  uint32_t* output;
  uint32_t words[(Bytes - sizeof(uint32_t*)) / sizeof(uint32_t)];
};
template <int Bytes> __global__ void kernel_one(Args<Bytes> args) {
  uint32_t sum = 0;
  for (uint32_t value : args.words) sum += value;
  *args.output = sum;
}
template <int Bytes> __global__ void kernel_two(Args<Bytes> args) {
  uint32_t sum = 0;
  for (int i = sizeof(args.words) / sizeof(uint32_t) - 1; i >= 0; --i) {
    uint32_t value = args.words[i];
    asm volatile("add.u32 %0, %0, 0;" : "+r"(value));
    sum += value;
  }
  *args.output = sum;
}
struct Shape { int nodes; int bytes; int kernel; bool fan; };
static size_t free_bytes(cudaStream_t stream) {
  CUDA(cudaStreamSynchronize(stream));
  size_t free, total;
  CUDA(cudaMemGetInfo(&free, &total));
  return free;
}
static void initialize(cudaStream_t* stream, uint32_t** output, int nodes) {
  CUDA(cudaSetDevice(0));
  CUDA(cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking));
  CUDA(cudaMalloc(output, nodes * sizeof(uint32_t)));
  Args<32> small{};
  small.output = *output;
  Args<512> large{};
  large.output = *output;
  kernel_one<32><<<1, 1, 0, *stream>>>(small);
  kernel_two<32><<<1, 1, 0, *stream>>>(small);
  kernel_one<512><<<1, 1, 0, *stream>>>(large);
  kernel_two<512><<<1, 1, 0, *stream>>>(large);
  CUDA(cudaStreamSynchronize(*stream));
}
template <int Bytes> static cudaGraphExec_t create_typed(
    const Shape& shape, uint32_t* output, uint32_t seed, cudaStream_t stream) {
  cudaGraph_t graph;
  CUDA(cudaGraphCreate(&graph, 0));
  cudaGraphNode_t root;
  CUDA(cudaGraphAddEmptyNode(&root, graph, nullptr, 0));
  std::vector<cudaGraphNode_t> nodes(shape.nodes);
  cudaGraphNode_t previous = root;
  for (int i = 0; i < shape.nodes; ++i) {
    Args<Bytes> args{};
    args.output = output + i;
    for (size_t j = 0; j < sizeof(args.words) / sizeof(uint32_t); ++j)
      args.words[j] = seed + j + 1;
    void* params[] = {&args};
    cudaKernelNodeParams kernel{};
    kernel.func = shape.kernel == 1 ? reinterpret_cast<void*>(kernel_one<Bytes>)
                                   : reinterpret_cast<void*>(kernel_two<Bytes>);
    kernel.gridDim = dim3(1);
    kernel.blockDim = dim3(1);
    kernel.kernelParams = params;
    cudaGraphNode_t dependency = shape.fan ? root : previous;
    CUDA(cudaGraphAddKernelNode(&nodes[i], graph, &dependency, 1, &kernel));
    previous = nodes[i];
  }
  cudaGraphNode_t tail;
  if (shape.fan) CUDA(cudaGraphAddEmptyNode(&tail, graph, nodes.data(), nodes.size()));
  else CUDA(cudaGraphAddEmptyNode(&tail, graph, &previous, 1));
  cudaGraphExec_t exec;
  CUDA(cudaGraphInstantiate(&exec, graph, 0));
  CUDA(cudaGraphDestroy(graph));
  CUDA(cudaGraphUpload(exec, stream));
  CUDA(cudaGraphLaunch(exec, stream));
  CUDA(cudaStreamSynchronize(stream));
  std::vector<uint32_t> actual(shape.nodes);
  CUDA(cudaMemcpy(actual.data(), output, actual.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  uint32_t n = (Bytes - sizeof(uint32_t*)) / sizeof(uint32_t);
  uint32_t expected = n * seed + n * (n + 1) / 2;
  for (uint32_t value : actual) if (value != expected) {
    std::fprintf(stderr, "bad graph replay %u != %u\n", value, expected);
    std::exit(1);
  }
  return exec;
}
static cudaGraphExec_t create(const Shape& shape, uint32_t* output,
                              uint32_t seed, cudaStream_t stream) {
  return shape.bytes == 32 ? create_typed<32>(shape, output, seed, stream)
                           : create_typed<512>(shape, output, seed, stream);
}
static void destroy(std::vector<cudaGraphExec_t>& graphs, cudaStream_t stream) {
  CUDA(cudaStreamSynchronize(stream));
  for (auto graph : graphs) CUDA(cudaGraphExecDestroy(graph));
  graphs.clear();
}
static void child_ok(pid_t pid) {
  int status;
  if (waitpid(pid, &status, 0) != pid || !WIFEXITED(status) || WEXITSTATUS(status)) std::exit(1);
}
static void slots(const Shape& shape) {
  pid_t pid = fork();
  if (pid < 0) std::exit(1);
  if (pid == 0) {
    cudaStream_t stream;
    uint32_t* output;
    initialize(&stream, &output, shape.nodes);
    size_t baseline = free_bytes(stream), previous = baseline;
    std::vector<cudaGraphExec_t> live;
    int steps = 0;
    for (int n = 1; n <= 256; ++n) {
      live.push_back(create(shape, output, n, stream));
      size_t free = free_bytes(stream);
      if (free < previous) ++steps;
      std::printf("{\"kind\":\"slot\",\"nodes\":%d,\"args\":%d,\"kernel\":%d,\"fan\":%d,"
                  "\"n\":%d,\"free_baseline\":%zu,\"free_before\":%zu,\"free_after\":%zu}\n",
                  shape.nodes, shape.bytes, shape.kernel, int(shape.fan), n, baseline, previous, free);
      previous = free;
      if (steps >= 4 && n >= 16) break;
    }
    size_t n = live.size(), k = n / 2;
    for (size_t i = 0; i < k; ++i) CUDA(cudaGraphExecDestroy(live[i]));
    live.erase(live.begin(), live.begin() + k);
    size_t after_destroy = free_bytes(stream);
    for (size_t i = 0; i < k; ++i) live.push_back(create(shape, output, 1000 + i, stream));
    size_t after_replace = free_bytes(stream);
    CUDA(cudaGraphLaunch(live.front(), stream));
    CUDA(cudaStreamSynchronize(stream));
    live.push_back(create(shape, output, 2000, stream));
    size_t after_extra = free_bytes(stream);
    std::printf("{\"kind\":\"replace\",\"nodes\":%d,\"args\":%d,\"kernel\":%d,\"fan\":%d,"
                "\"n\":%zu,\"k\":%zu,\"free_before\":%zu,\"free_destroyed\":%zu,"
                "\"free_replaced\":%zu,\"free_nplus1\":%zu}\n",
                shape.nodes, shape.bytes, shape.kernel, int(shape.fan), n, k, previous,
                after_destroy, after_replace, after_extra);
    destroy(live, stream);
    CUDA(cudaDeviceReset());
    std::fflush(stdout);
    _exit(0);
  }
  child_ok(pid);
}
static void identity(int nodes, int bytes, bool reverse) {
  pid_t pid = fork();
  if (pid < 0) std::exit(1);
  if (pid == 0) {
    cudaStream_t stream;
    uint32_t* output;
    initialize(&stream, &output, nodes);
    size_t baseline = free_bytes(stream);
    Shape shapes[] = {{nodes, bytes, 1, false}, {nodes, bytes, 2, false},
                      {nodes, bytes, 1, true}, {nodes, bytes, 2, true}};
    for (int phase = 0; phase < 8; ++phase) {
      int index = phase % 4;
      if (reverse) index = 3 - index;
      const Shape& shape = shapes[index];
      size_t before = free_bytes(stream);
      std::vector<cudaGraphExec_t> graphs;
      for (int i = 0; i < 32; ++i) graphs.push_back(create(shape, output, phase * 100 + i, stream));
      size_t after = free_bytes(stream);
      destroy(graphs, stream);
      std::printf("{\"kind\":\"identity\",\"nodes\":%d,\"args\":%d,\"reverse\":%d,\"phase\":%d,"
                  "\"kernel\":%d,\"fan\":%d,\"free_baseline\":%zu,\"free_before\":%zu,"
                  "\"free_after\":%zu,\"free_destroyed\":%zu}\n",
                  nodes, bytes, int(reverse), phase, shape.kernel, int(shape.fan),
                  baseline, before, after, free_bytes(stream));
    }
    CUDA(cudaDeviceReset());
    std::fflush(stdout);
    _exit(0);
  }
  child_ok(pid);
}
int main() {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  // No parent CUDA calls: every child has an independent empty context.
  for (int nodes : {64, 512, 2048}) for (int bytes : {32, 512}) {
    for (int kernel : {1, 2}) for (bool fan : {false, true}) slots({nodes, bytes, kernel, fan});
    identity(nodes, bytes, false);
    identity(nodes, bytes, true);
  }
  std::puts("{\"kind\":\"identity_done\",\"ok\":true}");
  return 0;
}

// Standalone graph-state lifetime experiment; emits JSONL to stdout.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <random>
#include <vector>

#define CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e)); \
  std::exit(1); } } while (0)

using Clock = std::chrono::steady_clock;
constexpr size_t MiB = 1024 * 1024;

template <int Bytes> struct Args {
  uint32_t* output;
  uint32_t words[(Bytes - sizeof(uint32_t*)) / sizeof(uint32_t)];
};
static_assert(sizeof(Args<32>) == 32 && sizeof(Args<512>) == 512);

template <int Bytes> __global__ void probe_kernel(Args<Bytes> args) {
  uint32_t sum = 0;
  for (uint32_t word : args.words) sum += word;
  *args.output = sum;
}

struct Shape { int nodes; int args; size_t charge = 0; };
struct Graph {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  size_t charge = 0;
};

static size_t free_bytes(cudaStream_t stream) {
  CUDA(cudaStreamSynchronize(stream));
  size_t free, total;
  CUDA(cudaMemGetInfo(&free, &total));
  return free;
}
static double elapsed_us(Clock::time_point start) {
  return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}
static void destroy(Graph& graph, cudaStream_t stream) {
  CUDA(cudaStreamSynchronize(stream));
  CUDA(cudaGraphExecDestroy(graph.exec));
  CUDA(cudaGraphDestroy(graph.graph));
  graph = {};
}

template <int Bytes> static void capture(cudaGraph_t* graph, int nodes,
                                        uint32_t* output, cudaStream_t stream) {
  Args<Bytes> args{};
  args.output = output;
  for (size_t i = 0; i < sizeof(args.words) / sizeof(uint32_t); ++i)
    args.words[i] = static_cast<uint32_t>(i + 1);
  CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  CUDA(cudaMemsetAsync(output, 0, 4096, stream));
  CUDA(cudaMemcpyAsync(output + 1024, output, 4096, cudaMemcpyDeviceToDevice, stream));
  for (int i = 0; i < nodes; ++i) probe_kernel<Bytes><<<1, 1, 0, stream>>>(args);
  CUDA(cudaMemsetAsync(output + 1024, 7, 4096, stream));
  CUDA(cudaStreamEndCapture(stream, graph));
}
static Graph create(const Shape& shape, uint32_t* output, cudaStream_t stream,
                    bool phases = false) {
  Graph graph;
  graph.charge = shape.charge;
  const size_t before = free_bytes(stream);
  auto start = Clock::now();
  if (shape.args == 32) capture<32>(&graph.graph, shape.nodes, output, stream);
  else capture<512>(&graph.graph, shape.nodes, output, stream);
  double capture_us = elapsed_us(start);
  const size_t captured = free_bytes(stream);
  start = Clock::now();
  CUDA(cudaGraphInstantiate(&graph.exec, graph.graph, 0));
  double instantiate_us = elapsed_us(start);
  const size_t instantiated = free_bytes(stream);
  start = Clock::now();
  CUDA(cudaGraphUpload(graph.exec, stream));
  CUDA(cudaStreamSynchronize(stream));
  double upload_us = elapsed_us(start);
  const size_t uploaded = free_bytes(stream);
  start = Clock::now();
  CUDA(cudaGraphLaunch(graph.exec, stream));
  CUDA(cudaStreamSynchronize(stream));
  double replay_us = elapsed_us(start);
  const size_t replayed = free_bytes(stream);
  if (phases) {
    size_t count;
    CUDA(cudaGraphGetNodes(graph.graph, nullptr, &count));
    std::printf("{\"kind\":\"shape\",\"nodes\":%d,\"total_nodes\":%zu,\"args\":%d,"
                "\"free_before\":%zu,\"free_capture\":%zu,\"free_instantiate\":%zu,"
                "\"free_upload\":%zu,\"free_replay\":%zu,\"capture_us\":%.3f,"
                "\"instantiate_us\":%.3f,\"upload_us\":%.3f,\"replay_us\":%.3f,\"miss_us\":%.3f}\n",
                shape.nodes, count, shape.args, before, captured, instantiated, uploaded,
                replayed, capture_us, instantiate_us, upload_us, replay_us,
                capture_us + instantiate_us + upload_us);
  }
  return graph;
}
static void verify(const Shape& shape, uint32_t* output) {
  uint32_t actual;
  CUDA(cudaMemcpy(&actual, output, sizeof(actual), cudaMemcpyDeviceToHost));
  uint32_t n = (shape.args - sizeof(uint32_t*)) / sizeof(uint32_t);
  uint32_t expected = n * (n + 1) / 2;
  if (actual != expected) {
    std::fprintf(stderr, "bad replay: %u != %u\n", actual, expected);
    std::exit(1);
  }
}
static void trim(cudaStream_t stream, const char* stage) {
  size_t before = free_bytes(stream);
  cudaError_t status = cudaDeviceGraphMemTrim(0);
  size_t after = free_bytes(stream);
  cuuint64_t used = 0, reserved = 0;
  CUDA(cudaDeviceGetGraphMemAttribute(0, cudaGraphMemAttrUsedMemCurrent, &used));
  CUDA(cudaDeviceGetGraphMemAttribute(0, cudaGraphMemAttrReservedMemCurrent, &reserved));
  std::printf("{\"kind\":\"trim\",\"stage\":\"%s\",\"status\":%d,"
              "\"free_before\":%zu,\"free_after\":%zu,\"graph_alloc_used\":%llu,"
              "\"graph_alloc_reserved\":%llu}\n", stage, int(status), before, after,
              static_cast<unsigned long long>(used), static_cast<unsigned long long>(reserved));
  if (status != cudaSuccess) cudaGetLastError();
}

int main(int argc, char** argv) {
  int cycles = argc > 1 ? std::atoi(argv[1]) : 1000;
  size_t cap = (argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 64) * MiB;
  if (cycles < 1000 || cap < 16 * MiB) return 2;
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  CUDA(cudaSetDevice(0));
  cudaDeviceProp prop;
  CUDA(cudaGetDeviceProperties(&prop, 0));
  int runtime, driver;
  CUDA(cudaRuntimeGetVersion(&runtime));
  CUDA(cudaDriverGetVersion(&driver));
  cudaStream_t stream;
  CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  uint32_t* output;
  CUDA(cudaMalloc(&output, 8192));
  Shape warm{1, 512};
  Graph warm_graph = create(warm, output, stream);
  verify(warm, output);
  destroy(warm_graph, stream);
  size_t baseline = free_bytes(stream);
  std::printf("{\"kind\":\"start\",\"device\":\"%s\",\"sm\":%d,\"runtime\":%d,"
              "\"driver\":%d,\"cycles\":%d,\"cap\":%zu,\"free_baseline\":%zu}\n",
              prop.name, prop.major * 10 + prop.minor, runtime, driver, cycles, cap, baseline);
  std::vector<Shape> shapes;
  for (int bytes : {32, 512}) for (int nodes : {1, 10, 100, 1000, 5000}) {
    shapes.push_back({nodes, bytes});
    Graph graph = create(shapes.back(), output, stream, true);
    verify(shapes.back(), output);
    size_t before = free_bytes(stream);
    destroy(graph, stream);
    size_t after = free_bytes(stream);
    std::printf("{\"kind\":\"shape_destroy\",\"nodes\":%d,\"args\":%d,"
                "\"free_before\":%zu,\"free_after\":%zu}\n", nodes, bytes, before, after);
  }
  // Amortize chunk quantization over 16 live graphs. The floor intentionally
  // overcharges tiny shapes instead of pretending their zero delta means free.
  for (auto& shape : shapes) {
    size_t before = free_bytes(stream);
    std::vector<Graph> samples;
    for (int i = 0; i < 16; ++i) samples.push_back(create(shape, output, stream));
    size_t after = free_bytes(stream);
    size_t growth = before > after ? before - after : 0;
    shape.charge = std::max<size_t>(2 * MiB / 16, (growth + 15) / 16);
    for (auto& graph : samples) destroy(graph, stream);
    size_t released = free_bytes(stream);
    std::printf("{\"kind\":\"calibrate\",\"nodes\":%d,\"args\":%d,\"charge\":%zu,"
                "\"free_before\":%zu,\"free_live16\":%zu,\"free_destroyed\":%zu}\n",
                shape.nodes, shape.args, shape.charge, before, after, released);
  }
  trim(stream, "after_calibration");
  size_t loop_baseline = free_bytes(stream);
  std::printf("{\"kind\":\"loop_start\",\"free\":%zu}\n", loop_baseline);
  std::mt19937 random(1605);
  std::deque<Graph> live;
  size_t live_bytes = 0;
  for (int cycle = 0; cycle < cycles; ++cycle) {
    const Shape& shape = shapes[random() % shapes.size()];
    size_t before = free_bytes(stream);
    int evictions = 0;
    while (!live.empty() && live_bytes + shape.charge > cap) {
      size_t pre_destroy = free_bytes(stream);
      live_bytes -= live.front().charge;
      destroy(live.front(), stream);
      live.pop_front();
      size_t post_destroy = free_bytes(stream);
      std::printf("{\"kind\":\"evict\",\"cycle\":%d,\"live_charge\":%zu,"
                  "\"free_before\":%zu,\"free_after\":%zu}\n",
                  cycle, live_bytes, pre_destroy, post_destroy);
      ++evictions;
    }
    size_t after_evictions = free_bytes(stream);
    live.push_back(create(shape, output, stream));
    live_bytes += shape.charge;
    size_t after_capture = free_bytes(stream);
    verify(shape, output);
    std::printf("{\"kind\":\"cycle\",\"cycle\":%d,\"nodes\":%d,\"args\":%d,"
                "\"live_count\":%zu,\"live_charge\":%zu,\"evictions\":%d,\"free_before\":%zu,"
                "\"free_after_evictions\":%zu,\"free_after_capture\":%zu}\n",
                cycle, shape.nodes, shape.args, live.size(), live_bytes, evictions,
                before, after_evictions, after_capture);
    if (cycle == cycles / 2) {
      trim(stream, "graphs_live");
      CUDA(cudaGraphLaunch(live.back().exec, stream));
      CUDA(cudaStreamSynchronize(stream));
      verify(shape, output);
    }
  }
  size_t before_destroy_all = free_bytes(stream);
  for (auto& graph : live) destroy(graph, stream);
  live.clear();
  size_t after_destroy_all = free_bytes(stream);
  trim(stream, "all_destroyed");
  CUDA(cudaFree(output));
  CUDA(cudaStreamDestroy(stream));
  CUDA(cudaDeviceSynchronize());
  size_t final_free, total;
  CUDA(cudaMemGetInfo(&final_free, &total));
  std::printf("{\"kind\":\"done\",\"free_before_destroy_all\":%zu,"
              "\"free_after_destroy_all\":%zu,\"free_after_buffers\":%zu,\"ok\":true}\n",
              before_destroy_all, after_destroy_all, final_free);
  return 0;
}

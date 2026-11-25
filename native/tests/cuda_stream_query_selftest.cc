#include "cuteafd_native.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

static void check(bool condition) {
  if (!condition) std::abort();
}
struct Gate { std::atomic<bool> release{false}; };
static void CUDART_CB hold(void* state) {
  auto& gate = *static_cast<Gate*>(state);
  while (!gate.release.load(std::memory_order_acquire)) std::this_thread::yield();
}
int main() {
  const char* tmp = std::getenv("TMPDIR");
  const auto profile = std::filesystem::absolute(std::filesystem::path(tmp ? tmp : ".") /
      ("graph-census-selftest-" + std::to_string(getpid())));
  std::filesystem::create_directories(profile);
  check(setenv("CUTEAFD_GRAPH_CENSUS", "1", 1) == 0);
  check(setenv("CUTEAFD_PROFILE_DIR", profile.c_str(), 1) == 0);
  check(setenv("CUTEAFD_PROFILE_HOST", "selftest", 1) == 0);
  void* pending = nullptr;
  void* peer = nullptr;
  check(cuteafd_cuda_stream_create(&pending) == CUTEAFD_STATUS_OK);
  check(cuteafd_cuda_stream_create(&peer) == CUTEAFD_STATUS_OK);
  check(cuteafd_cuda_stream_query(pending, nullptr) == CUTEAFD_STATUS_INVALID_ARGUMENT);
  int32_t ready = -1;
  check(cuteafd_cuda_stream_query(pending, &ready) == CUTEAFD_STATUS_OK && ready == 1);
  void* completion[2] = {};
  for (auto& event : completion) check(cuteafd_cuda_event_create_ordering(&event) == CUTEAFD_STATUS_OK);
  Gate gate;
  check(cudaLaunchHostFunc(static_cast<cudaStream_t>(pending), hold, &gate) == cudaSuccess);
  // The gate cannot complete until after these queries. A blocking implementation
  // hits the CTest deadline instead of silently passing as a completion query.
  check(cuteafd_cuda_stream_query(pending, &ready) == CUTEAFD_STATUS_OK && ready == 0);
  check(cuteafd_cuda_stream_query(peer, &ready) == CUTEAFD_STATUS_OK && ready == 1);
  check(cuteafd_cuda_event_record(completion[0], pending) == CUTEAFD_STATUS_OK);
  check(cuteafd_cuda_event_record(completion[1], peer) == CUTEAFD_STATUS_OK);
  check(cuteafd_cuda_event_query(completion[0], &ready) == CUTEAFD_STATUS_OK && ready == 0);
  check(cuteafd_cuda_stream_synchronize(peer) == CUTEAFD_STATUS_OK);
  check(cuteafd_cuda_event_query(completion[1], &ready) == CUTEAFD_STATUS_OK && ready == 1);
  gate.release.store(true, std::memory_order_release);
  check(cuteafd_cuda_stream_synchronize(pending) == CUTEAFD_STATUS_OK);
  check(cuteafd_cuda_stream_query(pending, &ready) == CUTEAFD_STATUS_OK && ready == 1);
  check(cuteafd_cuda_event_query(completion[0], &ready) == CUTEAFD_STATUS_OK && ready == 1);
  for (auto event : completion) {
    check(cuteafd_cuda_event_record(event, pending) == CUTEAFD_STATUS_OK);
  }
  check(cuteafd_cuda_stream_synchronize(pending) == CUTEAFD_STATUS_OK);
  for (auto event : completion) {
    check(cuteafd_cuda_event_query(event, &ready) == CUTEAFD_STATUS_OK && ready == 1);
    check(cuteafd_cuda_event_destroy(event) == CUTEAFD_STATUS_OK);
  }
  void* start = nullptr;
  void* end = nullptr;
  void* graph = nullptr;
  cuteafd_device_buffer_t buffer{};
  check(cuteafd_alloc_device_buffer(1 << 20, &buffer, CUTEAFD_MEMORY_KIND_WORKSPACE, "selftest") == CUTEAFD_STATUS_OK);
  check(cuteafd_cuda_event_create(&start) == CUTEAFD_STATUS_OK);
  check(cuteafd_cuda_event_create(&end) == CUTEAFD_STATUS_OK);
  check(cuteafd_cuda_event_record_external(nullptr, pending) == CUTEAFD_STATUS_INVALID_ARGUMENT);
  // Real drafters record these markers eagerly before their first capture.
  check(cuteafd_cuda_event_record_external(start, pending) == CUTEAFD_STATUS_OK);
  check(cudaMemsetAsync(buffer.ptr, 0x2a, buffer.bytes, static_cast<cudaStream_t>(pending)) == cudaSuccess);
  check(cuteafd_cuda_event_record_external(end, pending) == CUTEAFD_STATUS_OK);
  check(cuteafd_cuda_stream_synchronize(pending) == CUTEAFD_STATUS_OK);
  float eager_elapsed = -1.0f;
  check(cuteafd_cuda_event_elapsed_ms(start, end, &eager_elapsed) == CUTEAFD_STATUS_OK && eager_elapsed >= 0.0f);
  check(cuteafd_cuda_graph_begin_capture(pending) == CUTEAFD_STATUS_OK);
  check(cuteafd_cuda_event_record_external(start, pending) == CUTEAFD_STATUS_OK);
  check(cudaMemsetAsync(buffer.ptr, 0x5a, buffer.bytes, static_cast<cudaStream_t>(pending)) == cudaSuccess);
  check(cuteafd_cuda_event_record_external(end, pending) == CUTEAFD_STATUS_OK);
  check(cuteafd_cuda_graph_end_capture(pending, &graph) == CUTEAFD_STATUS_OK);
  for (int replay = 0; replay < 3; ++replay) {
    check(cuteafd_cuda_graph_launch(graph, pending) == CUTEAFD_STATUS_OK);
    check(cuteafd_cuda_stream_synchronize(pending) == CUTEAFD_STATUS_OK);
    float elapsed = -1.0f;
    check(cuteafd_cuda_event_elapsed_ms(start, end, &elapsed) == CUTEAFD_STATUS_OK && elapsed >= 0.0f);
    unsigned char first = 0;
    check(cudaMemcpy(&first, buffer.ptr, 1, cudaMemcpyDeviceToHost) == cudaSuccess && first == 0x5a);
    // An uncaptured shape can run between replays while the graph still owns
    // references to the same retained events.
    check(cuteafd_cuda_event_record_external(start, pending) == CUTEAFD_STATUS_OK);
    check(cudaMemsetAsync(buffer.ptr, 0x2a, buffer.bytes, static_cast<cudaStream_t>(pending)) == cudaSuccess);
    check(cuteafd_cuda_event_record_external(end, pending) == CUTEAFD_STATUS_OK);
    check(cuteafd_cuda_stream_synchronize(pending) == CUTEAFD_STATUS_OK);
    check(cuteafd_cuda_event_elapsed_ms(start, end, &eager_elapsed) == CUTEAFD_STATUS_OK && eager_elapsed >= 0.0f);
    check(cudaMemcpy(&first, buffer.ptr, 1, cudaMemcpyDeviceToHost) == cudaSuccess && first == 0x2a);
  }
  check(cuteafd_cuda_graph_exec_destroy(graph) == CUTEAFD_STATUS_OK);
  // A driver-loaded kernel has a known ABI and no runtime host-function entry.
  const char* ptx = R"(.version 7.0
.target sm_80
.address_size 64
.visible .entry census_store(.param .u32 value, .param .u64 dst) {
  .reg .u64 %rd;
  .reg .u32 %r;
  ld.param.u64 %rd, [dst];
  ld.param.u32 %r, [value];
  st.global.u32 [%rd], %r;
  ret;
})";
  CUmodule module = nullptr;
  CUfunction function = nullptr;
  check(cuModuleLoadData(&module, ptx) == CUDA_SUCCESS);
  check(cuModuleGetFunction(&function, module, "census_store") == CUDA_SUCCESS);
  CUdeviceptr dst = reinterpret_cast<CUdeviceptr>(buffer.ptr);
  unsigned value = 42;
  void* args[] = {&value, &dst};
  check(cuteafd_cuda_graph_begin_capture(pending) == CUTEAFD_STATUS_OK);
  check(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0,
      reinterpret_cast<CUstream>(pending), args, nullptr) == CUDA_SUCCESS);
  check(cuteafd_cuda_graph_end_capture(pending, &graph) == CUTEAFD_STATUS_OK);
  check(cuteafd_cuda_graph_launch(graph, pending) == CUTEAFD_STATUS_OK);
  check(cuteafd_cuda_stream_synchronize(pending) == CUTEAFD_STATUS_OK);
  unsigned actual = 0;
  check(cudaMemcpy(&actual, buffer.ptr, sizeof(actual), cudaMemcpyDeviceToHost) == cudaSuccess && actual == value);
  std::ifstream captures(profile / "graph-census-captures-selftest.jsonl");
  std::ostringstream records;
  records << captures.rdbuf();
  const auto text = records.str();
  check(text.find("\"cuda_driver_version\":") != std::string::npos);
  check(text.find("\"kernel\":\"census_store\"") != std::string::npos);
  check(text.find("\"param_count\":2,\"param_bytes\":16,\"param_payload_bytes\":12,\"param_status\":0") != std::string::npos);
  check(text.find("\"constant_status\":0") != std::string::npos);
  check(text.find("\"local_status\":0") != std::string::npos);
  check(cudaGetLastError() == cudaSuccess);
  check(cuteafd_cuda_graph_exec_destroy(graph) == CUTEAFD_STATUS_OK);
  check(cuModuleUnload(module) == CUDA_SUCCESS);
  std::filesystem::remove_all(profile);
  check(cuteafd_cuda_event_destroy(end) == CUTEAFD_STATUS_OK);
  check(cuteafd_cuda_event_destroy(start) == CUTEAFD_STATUS_OK);
  check(cuteafd_free_device_buffer(&buffer) == CUTEAFD_STATUS_OK);
  check(cuteafd_cuda_stream_destroy(peer) == CUTEAFD_STATUS_OK);
  check(cuteafd_cuda_stream_destroy(pending) == CUTEAFD_STATUS_OK);
  std::cout << "stream query distinguishes pending and complete without joining the peer\n";
}

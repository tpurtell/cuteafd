#include "cuteafd_native.h"
#include <cuda_runtime_api.h>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>

#define check(condition) do { if (!(condition)) { \
  char error[1024]{}; cuteafd_last_error(error, sizeof(error)); \
  std::cerr << "check failed at " << __LINE__ << ": " << #condition << '\n'; \
  std::cerr << error << '\n'; \
  std::abort(); \
} } while (false)
struct Gate { std::atomic<bool> release{false}; };
static void CUDART_CB hold(void* state) {
  auto& gate = *static_cast<Gate*>(state);
  while (!gate.release.load(std::memory_order_acquire)) std::this_thread::yield();
}

int main() {
  int count = 0;
  check(cudaGetDeviceCount(&count) == cudaSuccess);
  if (count < 2) { std::cout << "requires two CUDA devices\n"; return 77; }
  check(cuteafd_cuda_get_device(nullptr) == CUTEAFD_STATUS_INVALID_ARGUMENT);
  check(cuteafd_cuda_set_device(-1) == CUTEAFD_STATUS_INVALID_ARGUMENT);
  constexpr size_t bytes = 1024 * 1024;
  for (int source = 0; source < 2; ++source) {
    const int destination = 1 - source;
    cuteafd_device_buffer_t src{}, dst{};
    void *producer = nullptr, *consumer = nullptr, *pending = nullptr, *event = nullptr;
    check(cuteafd_cuda_set_device(source) == CUTEAFD_STATUS_OK);
    check(cuteafd_cuda_enable_peer(destination) == CUTEAFD_STATUS_OK);
    check(cuteafd_cuda_enable_peer(destination) == CUTEAFD_STATUS_OK);
    check(cuteafd_alloc_device_buffer(bytes, &src) == CUTEAFD_STATUS_OK);
    check(src.device_id == source);
    check(cuteafd_cuda_stream_create(&producer) == CUTEAFD_STATUS_OK);
    check(cuteafd_cuda_event_create(&event) == CUTEAFD_STATUS_OK);
    check(cuteafd_cuda_set_device(destination) == CUTEAFD_STATUS_OK);
    check(cuteafd_cuda_enable_peer(source) == CUTEAFD_STATUS_OK);
    check(cuteafd_alloc_device_buffer(bytes, &dst) == CUTEAFD_STATUS_OK);
    check(dst.device_id == destination);
    check(cuteafd_cuda_stream_create(&consumer) == CUTEAFD_STATUS_OK);
    check(cuteafd_cuda_stream_create(&pending) == CUTEAFD_STATUS_OK);
    int owner = -1, current = -1;
    check(cuteafd_cuda_stream_device(producer, nullptr) == CUTEAFD_STATUS_INVALID_ARGUMENT);
    check(cuteafd_cuda_stream_device(producer, &owner) == CUTEAFD_STATUS_OK && owner == source);
    check(cuteafd_cuda_stream_device(consumer, &owner) == CUTEAFD_STATUS_OK && owner == destination);
    check(cuteafd_cuda_get_device(&current) == CUTEAFD_STATUS_OK && current == destination);
    check(cuteafd_copy_peer_async(dst, src, bytes + 1, consumer) == CUTEAFD_STATUS_INVALID_ARGUMENT);
    check(cuteafd_copy_peer_async(dst, src, bytes, nullptr) == CUTEAFD_STATUS_INVALID_ARGUMENT);
    std::vector<unsigned char> host(bytes);
    for (int pattern : {37, 191, 0}) {
      check(cuteafd_cuda_set_device(source) == CUTEAFD_STATUS_OK);
      check(cuteafd_copy_peer_async(dst, src, bytes, consumer) == CUTEAFD_STATUS_INVALID_ARGUMENT);
      check(cudaMemsetAsync(src.ptr, pattern, bytes, static_cast<cudaStream_t>(producer)) == cudaSuccess);
      check(cuteafd_cuda_event_record(event, producer) == CUTEAFD_STATUS_OK);
      check(cuteafd_cuda_set_device(destination) == CUTEAFD_STATUS_OK);
      Gate gate;
      check(cudaLaunchHostFunc(static_cast<cudaStream_t>(pending), hold, &gate) == cudaSuccess);
      check(cuteafd_cuda_stream_wait_event(consumer, event) == CUTEAFD_STATUS_OK);
      check(cuteafd_copy_peer_async(dst, src, bytes, consumer) == CUTEAFD_STATUS_OK);
      int32_t ready = 0;
      while (!ready) {
        check(cuteafd_cuda_stream_query(consumer, &ready) == CUTEAFD_STATUS_OK);
        std::this_thread::yield();
      }
      check(cuteafd_cuda_stream_query(pending, &ready) == CUTEAFD_STATUS_OK && ready == 0);
      gate.release.store(true, std::memory_order_release);
      check(cuteafd_cuda_stream_synchronize(pending) == CUTEAFD_STATUS_OK);
      check(cuteafd_copy_d2h(host.data(), dst, bytes) == CUTEAFD_STATUS_OK);
      for (auto value : host) check(value == pattern);
    }
    cuteafd_device_buffer_t local{};
    check(cuteafd_alloc_device_buffer(bytes, &local) == CUTEAFD_STATUS_OK);
    std::vector<unsigned char> values(bytes);
    for (size_t i = 0; i < bytes; ++i) values[i] = (i * 17 + 11) % 251;
    check(cuteafd_copy_h2d(local, values.data(), bytes) == CUTEAFD_STATUS_OK);
    check(cuteafd_cuda_set_device(source) == CUTEAFD_STATUS_OK);
    check(cuteafd_copy_h2d(src, values.data(), bytes) == CUTEAFD_STATUS_OK);
    check(cuteafd_copy_device_rows_async(dst, src, 17, 19, 43, 31, consumer) == CUTEAFD_STATUS_INVALID_ARGUMENT);
    check(cuteafd_cuda_set_device(destination) == CUTEAFD_STATUS_OK);
    check(cuteafd_copy_device_rows_async(dst, src, 17, 19, 16, 31, consumer) == CUTEAFD_STATUS_INVALID_ARGUMENT);
    check(cuteafd_copy_device_rows_async(dst, src, 17, 19, 43, 16, consumer) == CUTEAFD_STATUS_INVALID_ARGUMENT);
    check(cuteafd_copy_device_rows_async(dst, src, 17, 0, 43, 31, consumer) == CUTEAFD_STATUS_INVALID_ARGUMENT);
    check(cuteafd_copy_device_rows_async(dst, src, 0, 19, 43, 31, consumer) == CUTEAFD_STATUS_INVALID_ARGUMENT);
    check(cuteafd_copy_device_rows_async(dst, src, 17, std::numeric_limits<size_t>::max(), 43, 31, consumer) == CUTEAFD_STATUS_INVALID_ARGUMENT);
    check(cuteafd_copy_device_rows_async(dst, dst, 17, 19, 43, 31, consumer) == CUTEAFD_STATUS_INVALID_ARGUMENT);
    auto short_src = src;
    short_src.bytes = 18 * 31 + 16;
    check(cuteafd_copy_device_rows_async(dst, short_src, 17, 19, 43, 31, consumer) == CUTEAFD_STATUS_INVALID_ARGUMENT);
    for (auto input : {src, local}) {
      Gate gate;
      check(cudaLaunchHostFunc(static_cast<cudaStream_t>(pending), hold, &gate) == cudaSuccess);
      check(cudaMemsetAsync(dst.ptr, 255, bytes, static_cast<cudaStream_t>(consumer)) == cudaSuccess);
      check(cuteafd_copy_device_rows_async(dst, input, 17, 19, 43, 31, consumer) == CUTEAFD_STATUS_OK);
      check(cuteafd_cuda_stream_synchronize(consumer) == CUTEAFD_STATUS_OK);
      int32_t ready = 0;
      check(cuteafd_cuda_stream_query(pending, &ready) == CUTEAFD_STATUS_OK && ready == 0);
      gate.release.store(true, std::memory_order_release);
      check(cuteafd_cuda_stream_synchronize(pending) == CUTEAFD_STATUS_OK);
      check(cuteafd_copy_d2h(host.data(), dst, bytes) == CUTEAFD_STATUS_OK);
      for (size_t i = 0; i < bytes; ++i) {
        const size_t row = i / 43, column = i % 43;
        check(host[i] == (row < 19 && column < 17 ? values[row * 31 + column] : 255));
      }
    }
    check(cuteafd_free_device_buffer(&local) == CUTEAFD_STATUS_OK);
    check(cuteafd_cuda_stream_destroy(pending) == CUTEAFD_STATUS_OK);
    check(cuteafd_cuda_stream_destroy(consumer) == CUTEAFD_STATUS_OK);
    check(cuteafd_free_device_buffer(&dst) == CUTEAFD_STATUS_OK);
    check(cuteafd_cuda_set_device(source) == CUTEAFD_STATUS_OK);
    check(cuteafd_cuda_event_destroy(event) == CUTEAFD_STATUS_OK);
    check(cuteafd_cuda_stream_destroy(producer) == CUTEAFD_STATUS_OK);
    check(cuteafd_free_device_buffer(&src) == CUTEAFD_STATUS_OK);
  }
  std::cout << "bidirectional peer and local pitched copies preserve data/padding, reject invalid extents and do not join a pending lane\n";
}

// Verifies that cuteafd_copy_d2h orders against a non-blocking producing stream
// without synchronizing unrelated streams, rejects a null stream, and that the
// old legacy default-stream cudaMemcpy fails the same test by reading stale
// data ahead of an in-flight kernel on a non-blocking stream.
#include "cuteafd_native.h"
#include "cuteafd_peer_exchange.h"
#include <cuda_runtime_api.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>

__global__ void write_u32_kernel(uint32_t* dst, uint32_t value) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *dst = value;
  }
}

static void check(bool condition, const char* what) {
  if (!condition) {
    std::cerr << "FAILED: " << what << "\n";
    std::exit(1);
  }
}

static std::atomic<uint32_t>& word_at(uint32_t* word) {
  return *reinterpret_cast<std::atomic<uint32_t>*>(word);
}

int main(int argc, char** argv) {
  bool legacy_mode = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--legacy") == 0) {
      legacy_mode = true;
    }
  }

  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::cout << "no CUDA device\n";
    return 77;
  }

  cuteafd_host_buffer_t words{};
  check(cuteafd_alloc_host_buffer(4096, &words, CUTEAFD_MEMORY_KIND_WORKSPACE,
                                  "d2h-order-selftest") == CUTEAFD_STATUS_OK,
        "alloc mapped pinned host words");
  std::memset(words.ptr, 0, words.bytes);
  auto* flag = static_cast<uint32_t*>(words.ptr);
  auto* other_flag = flag + 16;

  void* stream = nullptr;
  void* other = nullptr;
  check(cuteafd_cuda_stream_create(&stream) == CUTEAFD_STATUS_OK, "create producing stream");
  check(cuteafd_cuda_stream_create(&other) == CUTEAFD_STATUS_OK, "create unrelated stream");

  cuteafd_device_buffer_t device{};
  check(cuteafd_alloc_device_buffer(4096, &device, CUTEAFD_MEMORY_KIND_WORKSPACE,
                                    "d2h-order-selftest") == CUTEAFD_STATUS_OK,
        "alloc device buffer");

  // Null stream is rejected by cuteafd_copy_d2h.
  uint32_t probe = 0;
  check(cuteafd_copy_d2h(&probe, device, sizeof(probe), nullptr) == CUTEAFD_STATUS_INVALID_ARGUMENT,
        "null stream rejected");

  constexpr uint32_t kInitial = 0x11111111u;
  constexpr uint32_t kWritten = 0x22222222u;

  // Initialize device word to kInitial on `stream` and drain.
  write_u32_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<uint32_t*>(device.ptr), kInitial);
  check(cudaGetLastError() == cudaSuccess, "launch init kernel");
  check(cuteafd_cuda_stream_synchronize(stream) == CUTEAFD_STATUS_OK, "sync init kernel");

  // Gate `stream` behind `flag == 1`, then launch kernel writing kWritten on `stream`.
  check(cuteafd_cuda_stream_wait_value32(stream, flag, 1) == CUTEAFD_STATUS_OK,
        "gate producing stream");
  write_u32_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<uint32_t*>(device.ptr), kWritten);
  check(cudaGetLastError() == cudaSuccess, "launch gated write kernel");

  if (legacy_mode) {
    // Demonstrate that the old legacy-stream cudaMemcpy races ahead of the
    // non-blocking stream and observes stale kInitial instead of kWritten.
    std::thread releaser([&]() {
      std::this_thread::sleep_for(std::chrono::milliseconds(40));
      word_at(flag).store(1, std::memory_order_release);
    });
    uint32_t legacy_read = 0;
    check(cudaMemcpy(&legacy_read, device.ptr, sizeof(legacy_read), cudaMemcpyDeviceToHost) ==
              cudaSuccess,
          "legacy cudaMemcpy");
    releaser.join();
    cuteafd_cuda_stream_synchronize(stream);
    if (legacy_read != kWritten) {
      std::cerr << "FAILED: legacy cudaMemcpy observed stale value 0x" << std::hex << legacy_read
                << " instead of 0x" << kWritten << std::dec << "\n";
      cuteafd_free_device_buffer(&device);
      cuteafd_cuda_stream_destroy(other);
      cuteafd_cuda_stream_destroy(stream);
      cuteafd_free_host_buffer(&words);
      return 1;
    }
    return 0;
  }

  // Negative control: while `flag == 0`, legacy default-stream cudaMemcpy does
  // not order against non-blocking `stream` and reads stale kInitial.
  uint32_t legacy_read = 0;
  check(cudaMemcpy(&legacy_read, device.ptr, sizeof(legacy_read), cudaMemcpyDeviceToHost) ==
            cudaSuccess,
        "legacy cudaMemcpy negative control");
  check(legacy_read == kInitial,
        "legacy default-stream cudaMemcpy raced non-blocking stream and read stale value");

  // Gate an unrelated non-blocking stream `other` so a device-wide synchronize
  // would deadlock or stall; cuteafd_copy_d2h must sync only `stream`.
  check(cuteafd_cuda_stream_wait_value32(other, other_flag, 1) == CUTEAFD_STATUS_OK,
        "gate unrelated stream");

  std::atomic<bool> copy_finished{false};
  std::atomic<bool> stayed_blocked{false};
  std::thread releaser([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    stayed_blocked.store(!copy_finished.load(std::memory_order_acquire), std::memory_order_release);
    word_at(flag).store(1, std::memory_order_release);
  });

  uint32_t ordered_read = 0;
  check(cuteafd_copy_d2h(&ordered_read, device, sizeof(ordered_read), stream) == CUTEAFD_STATUS_OK,
        "stream-ordered cuteafd_copy_d2h");
  copy_finished.store(true, std::memory_order_release);
  releaser.join();

  check(stayed_blocked.load(std::memory_order_acquire),
        "cuteafd_copy_d2h waited for producing stream before returning");
  check(ordered_read == kWritten,
        "cuteafd_copy_d2h observed kernel write on non-blocking stream");

  int32_t other_ready = -1;
  check(cuteafd_cuda_stream_query(other, &other_ready) == CUTEAFD_STATUS_OK && other_ready == 0,
        "cuteafd_copy_d2h did not synchronize unrelated stream");

  // Release `other`, drain, and confirm quiescent read sees the drained value.
  word_at(other_flag).store(1, std::memory_order_release);
  check(cuteafd_cuda_stream_synchronize(other) == CUTEAFD_STATUS_OK, "sync unrelated stream");
  uint32_t quiescent_read = 0;
  check(cuteafd_copy_d2h_quiescent(&quiescent_read, device, sizeof(quiescent_read)) ==
            CUTEAFD_STATUS_OK,
        "cuteafd_copy_d2h_quiescent after drain");
  check(quiescent_read == kWritten, "quiescent copy observed drained value");

  check(cuteafd_free_device_buffer(&device) == CUTEAFD_STATUS_OK, "free device buffer");
  check(cuteafd_cuda_stream_destroy(other) == CUTEAFD_STATUS_OK, "destroy other stream");
  check(cuteafd_cuda_stream_destroy(stream) == CUTEAFD_STATUS_OK, "destroy producing stream");
  check(cuteafd_free_host_buffer(&words) == CUTEAFD_STATUS_OK, "free host words");

  std::cout << "cuteafd_cuda_d2h_stream_order_selftest passed\n";
  std::cout << "test result: ok. 1 passed; 0 failed\n";
  return 0;
}

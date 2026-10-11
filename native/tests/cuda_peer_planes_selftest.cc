// cuteafd_peer_push_planes against cuteafd_peer_push_signal: the same link
// (flag, send state) alternates between the two kernels and every wait sees
// the bytes it was signalled for; strided local and peer planes land exactly
// and leave row padding alone; invalid planes are refused.
#include "cuteafd_peer_exchange.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#define check(condition) do { if (!(condition)) { \
  std::cerr << "check failed at " << __LINE__ << ": " << #condition << '\n'; \
  std::abort(); \
} } while (false)

namespace {
struct Side {
  int device;
  cudaStream_t stream = nullptr;
  uint8_t* buffer = nullptr;   // sources
  uint8_t* recv = nullptr;     // pushes from the peer land here
  uint8_t* local = nullptr;    // local plane destination
  uint32_t* control = nullptr; // [flag, send seq, arrivals, recv seq]
};

constexpr size_t kBytes = 1 << 20;

std::vector<uint8_t> read(int device, const uint8_t* p, size_t bytes) {
  std::vector<uint8_t> out(bytes);
  check(cudaSetDevice(device) == cudaSuccess);
  check(cudaMemcpy(out.data(), p, bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
  return out;
}
}  // namespace

int main() {
  int count = 0;
  check(cudaGetDeviceCount(&count) == cudaSuccess);
  if (count < 2) { std::cout << "requires two CUDA devices\n"; return 77; }
  Side side[2] = {{0}, {1}};
  for (auto& s : side) {
    check(cudaSetDevice(s.device) == cudaSuccess);
    int can = 0;
    check(cudaDeviceCanAccessPeer(&can, s.device, 1 - s.device) == cudaSuccess);
    if (!can) { std::cout << "no peer access\n"; return 77; }
    const cudaError_t enabled = cudaDeviceEnablePeerAccess(1 - s.device, 0);
    check(enabled == cudaSuccess || enabled == cudaErrorPeerAccessAlreadyEnabled);
    cudaGetLastError();
    check(cuteafd_peer_exchange_initialize() == cudaSuccess);
    check(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking) == cudaSuccess);
    for (uint8_t** p : {&s.buffer, &s.recv, &s.local}) check(cudaMalloc(p, kBytes) == cudaSuccess);
    check(cudaMalloc(&s.control, 64) == cudaSuccess);
    check(cudaMemset(s.control, 0, 64) == cudaSuccess);
    check(cudaMemset(s.recv, 0xEE, kBytes) == cudaSuccess);
    check(cudaMemset(s.local, 0xEE, kBytes) == cudaSuccess);
    std::vector<uint8_t> pattern(kBytes);
    for (size_t i = 0; i < kBytes; ++i) pattern[i] = uint8_t((i * 131 + 7 * s.device + 3) % 251);
    check(cudaMemcpy(s.buffer, pattern.data(), kBytes, cudaMemcpyHostToDevice) == cudaSuccess);
    check(cudaDeviceSynchronize() == cudaSuccess);
  }
  const auto source = [&](int device) { return read(device, side[device].buffer, kBytes); };
  const std::vector<uint8_t> src[2] = {source(0), source(1)};

  // 1. One link (A's send state, B's flag) used by push_signal, then push_planes with one
  //    contiguous plane, then push_signal again: each wait sees its own bytes in order.
  for (int a = 0; a < 2; ++a) {
    Side& A = side[a];
    Side& B = side[1 - a];
    const uint64_t sizes[3] = {4096, 36864, 16 * 1024};
    for (int round = 0; round < 3; ++round) {
      check(cudaSetDevice(A.device) == cudaSuccess);
      const uint64_t bytes = sizes[round], offset = round * 65536;
      if (round == 1) {
        cuteafd_peer_plane_t plane{B.recv + offset, A.buffer + offset, 1, bytes, bytes, bytes};
        check(cuteafd_peer_push_planes(&plane, 1, B.control, A.control + 1, 0, A.stream) == cudaSuccess);
      } else {
        check(cuteafd_peer_push_signal(B.recv + offset, A.buffer + offset, bytes, B.control, A.control + 1, 0,
            A.stream) == cudaSuccess);
      }
      check(cudaSetDevice(B.device) == cudaSuccess);
      check(cuteafd_peer_wait(B.control, B.control + 3, B.stream) == cudaSuccess);
      check(cudaStreamSynchronize(B.stream) == cudaSuccess);
      const auto got = read(B.device, B.recv + offset, bytes);
      for (uint64_t i = 0; i < bytes; ++i) check(got[i] == src[a][offset + i]);
    }
    check(cudaSetDevice(A.device) == cudaSuccess);
    check(cudaStreamSynchronize(A.stream) == cudaSuccess);
    uint32_t words[4];
    check(cudaMemcpy(words, A.control, 16, cudaMemcpyDeviceToHost) == cudaSuccess);
    check(words[1] == 3 && words[2] == 0);  // three sequences sent, arrival counter reset
  }

  // 2. Strided planes in one launch: a local plane, a peer plane, an empty plane and a second
  //    peer plane; padding between rows keeps its 0xEE fill.
  {
    Side& A = side[0];
    Side& B = side[1];
    check(cudaSetDevice(A.device) == cudaSuccess);
    check(cudaMemset(A.local, 0xEE, kBytes) == cudaSuccess);
    check(cudaSetDevice(B.device) == cudaSuccess);
    check(cudaMemset(B.recv, 0xEE, kBytes) == cudaSuccess);
    check(cudaDeviceSynchronize() == cudaSuccess);
    check(cudaSetDevice(A.device) == cudaSuccess);
    const cuteafd_peer_plane_t planes[4] = {
        {A.local, A.buffer, 7, 1024, 2048, 1152},                      // local: q's own half
        {B.recv, A.buffer + 16384, 7, 1024, 2048, 1152},               // peer: q's half at the peer
        {B.recv + 100000, A.buffer, 0, 0, 0, 0},                       // empty
        {B.recv + 1024 + 32768, A.buffer + 65536, 5, 512, 2048, 512},  // peer, offset column
    };
    check(cuteafd_peer_push_planes(planes, 4, B.control, A.control + 1, 3, A.stream) == cudaSuccess);
    check(cudaSetDevice(B.device) == cudaSuccess);
    check(cuteafd_peer_wait(B.control, B.control + 3, B.stream) == cudaSuccess);
    check(cudaStreamSynchronize(B.stream) == cudaSuccess);
    const auto local = read(A.device, A.local, kBytes);
    const auto recv = read(B.device, B.recv, kBytes);
    for (size_t i = 0; i < 7 * 2048; ++i) {
      const size_t row = i / 2048, col = i % 2048;
      check(local[i] == (col < 1024 ? src[0][row * 1152 + col] : 0xEE));
      check(recv[i] == (col < 1024 ? src[0][16384 + row * 1152 + col] : 0xEE));
    }
    for (size_t i = 0; i < 5 * 2048; ++i) {
      const size_t row = i / 2048, col = i % 2048, at = 1024 + 32768 + i;
      check(recv[at] == (col < 512 ? src[0][65536 + row * 512 + col] : 0xEE));
    }
    check(recv[100000] == 0xEE);
    // Copy only (no flag): nothing published, bytes land.
    check(cudaSetDevice(A.device) == cudaSuccess);
    const cuteafd_peer_plane_t only{A.local + 524288, A.buffer, 2, 64, 64, 64};
    check(cuteafd_peer_push_planes(&only, 1, nullptr, nullptr, 0, A.stream) == cudaSuccess);
    check(cudaStreamSynchronize(A.stream) == cudaSuccess);
    const auto copied = read(A.device, A.local + 524288, 128);
    for (size_t i = 0; i < 128; ++i) check(copied[i] == src[0][i]);
    uint32_t words[4];
    check(cudaMemcpy(words, A.control, 16, cudaMemcpyDeviceToHost) == cudaSuccess);
    check(words[1] == 4);
  }

  // 3. Invalid planes are refused before launch.
  {
    Side& A = side[0];
    check(cudaSetDevice(A.device) == cudaSuccess);
    cuteafd_peer_plane_t bad{A.local, A.buffer, 1, 24, 32, 32};
    check(cuteafd_peer_push_planes(&bad, 1, nullptr, nullptr, 0, A.stream) == cudaErrorInvalidValue);
    bad = {A.local, A.buffer, 2, 64, 32, 64};
    check(cuteafd_peer_push_planes(&bad, 1, nullptr, nullptr, 0, A.stream) == cudaErrorInvalidValue);
    bad = {A.local + 8, A.buffer, 1, 16, 16, 16};
    check(cuteafd_peer_push_planes(&bad, 1, nullptr, nullptr, 0, A.stream) == cudaErrorInvalidValue);
    bad = {A.local, A.buffer, 1, 16, 16, 16};
    check(cuteafd_peer_push_planes(&bad, 5, nullptr, nullptr, 0, A.stream) == cudaErrorInvalidValue);
    check(cuteafd_peer_push_planes(&bad, 0, nullptr, nullptr, 0, A.stream) == cudaErrorInvalidValue);
    check(cuteafd_peer_push_planes(&bad, 1, A.control, nullptr, 0, A.stream) == cudaErrorInvalidValue);
  }
  for (auto& s : side) {
    check(cudaSetDevice(s.device) == cudaSuccess);
    check(cudaStreamSynchronize(s.stream) == cudaSuccess);
    check(cudaStreamDestroy(s.stream) == cudaSuccess);
    for (void* p : {static_cast<void*>(s.buffer), static_cast<void*>(s.recv), static_cast<void*>(s.local),
         static_cast<void*>(s.control)}) check(cudaFree(p) == cudaSuccess);
  }
  std::cout << "multi-plane push shares a link with push_signal, lands strided local and peer planes exactly, "
               "keeps padding and refuses invalid planes\n";
  return 0;
}

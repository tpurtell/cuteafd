// Two-GPU exchange over peer memory (see cuteafd_peer_exchange.h) and the
// `cuteafd fabric --p2p` probe that measures it against the copy engine and a
// pinned-host bounce.
#include "cuteafd_peer_exchange.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>
#include <algorithm>
#include <atomic>
#include <thread>
#include <chrono>
#include <vector>

namespace {

__device__ __forceinline__ void store_release_sys(uint32_t* address, uint32_t value) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(address), "r"(value) : "memory");
}

__device__ __forceinline__ uint32_t load_acquire_sys(const uint32_t* address) {
  uint32_t value;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  return value;
}

__global__ void push_signal(uint4* destination, const uint4* source, uint64_t units, uint32_t* flag,
    uint32_t* state) {
  const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < units; i += stride)
    destination[i] = source[i];
  // Every block's peer stores are ordered before its arrival; the last block
  // to arrive publishes the sequence (fence cumulativity carries the others').
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    if (atomicAdd(&state[1], 1u) == gridDim.x - 1) {
      __threadfence_system();
      state[1] = 0;
      const uint32_t sequence = state[0] + 1;
      state[0] = sequence;
      store_release_sys(flag, sequence);
    }
  }
}

// Up to four strided byte planes (each local or on the peer), then one
// signal exactly as `push_signal` publishes it (the same `send_state`
// sequence and arrival counter, so the two kernels may share a flag).
struct Planes {
  cuteafd_peer_plane_t plane[CUTEAFD_PEER_MAX_PLANES];
  uint64_t first_unit[CUTEAFD_PEER_MAX_PLANES + 1];
  uint32_t count;
};

__global__ void push_planes(Planes planes, uint32_t* flag, uint32_t* state) {
  const uint64_t units = planes.first_unit[planes.count];
  const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < units; i += stride) {
    uint32_t p = 0;
    while (i >= planes.first_unit[p + 1]) ++p;
    const cuteafd_peer_plane_t& plane = planes.plane[p];
    const uint64_t per_row = plane.row_bytes / 16, unit = i - planes.first_unit[p];
    const uint64_t row = unit / per_row, column = unit % per_row;
    reinterpret_cast<uint4*>(static_cast<uint8_t*>(plane.destination) + row * plane.destination_pitch)[column] =
        reinterpret_cast<const uint4*>(static_cast<const uint8_t*>(plane.source) + row * plane.source_pitch)[column];
  }
  if (!flag) return;
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    if (atomicAdd(&state[1], 1u) == gridDim.x - 1) {
      __threadfence_system();
      state[1] = 0;
      const uint32_t sequence = state[0] + 1;
      state[0] = sequence;
      store_release_sys(flag, sequence);
    }
  }
}

__device__ __forceinline__ uint64_t global_ns() {
  uint64_t ns;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns));
  return ns;
}

// A wait gives up after this long (the peer's stream failed or was never fed):
// the trap faults this stream, so the host sees an error instead of a hang.
constexpr uint64_t kWaitTimeoutNs = 60ull * 1000 * 1000 * 1000;

__global__ void wait_flag(const uint32_t* flag, uint32_t* state) {
  if (threadIdx.x == 0) {
    const uint32_t expected = state[0] + 1;
    const uint64_t start = global_ns();
    uint32_t polls = 0;
    while (int32_t(load_acquire_sys(flag) - expected) < 0) {
      if ((++polls & 1023) == 0 && global_ns() - start > kWaitTimeoutNs) {
        printf("peer_wait: flag %p still %u, waiting for %u after 60 s\n", flag, load_acquire_sys(flag), expected);
        __trap();
      }
    }
    state[0] = expected;
  }
  __syncthreads();
}

// Publishes a host mailbox: `words` descriptor words, then the next sequence
// (release, system scope) for the host proxy spinning on `flag`. Work earlier
// on the stream (the D2H copies of the payload) has completed when it runs.
__global__ void host_signal(uint32_t* flag, uint32_t* send_state, uint32_t* descriptor, uint4 words) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    volatile uint32_t* out = descriptor;
    out[0] = words.x;
    out[1] = words.y;
    out[2] = words.z;
    out[3] = words.w;
    __threadfence_system();
    const uint32_t sequence = send_state[0] + 1;
    send_state[0] = sequence;
    store_release_sys(flag, sequence);
  }
}

__device__ __forceinline__ uint64_t load_acquire_sys_u64(const uint64_t* address) {
  uint64_t value;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(address) : "memory");
  return value;
}

// Waits for every rank's NIC-written completion flag (low 32 bits = the next
// sequence of `state`), records ranks that flagged an error, then advances
// the sequence. One warp; lane r watches rank r.
__global__ void wait_written(const uint64_t* flags, uint32_t ranks, uint32_t stride_words, uint32_t* state,
    uint32_t* error) {
  const uint32_t expected = state[0] + 1;
  const uint32_t rank = threadIdx.x;
  if (rank < ranks) {
    const uint64_t* flag = flags + uint64_t(rank) * stride_words;
    const uint64_t start = global_ns();
    uint32_t polls = 0;
    uint64_t value = load_acquire_sys_u64(flag);
    while (uint32_t(value) != expected) {
      if ((++polls & 1023) == 0 && global_ns() - start > kWaitTimeoutNs) {
        printf("spark wait: rank %u flag %llx, waiting for %u after 60 s\n", rank, (unsigned long long)value, expected);
        __trap();
      }
      value = load_acquire_sys_u64(flag);
    }
    if (value >> 63) atomicOr(error, 1u << rank);
  }
  __syncwarp();
  if (threadIdx.x == 0) {
    __threadfence();
    state[0] = expected;
  }
}

__global__ void wait_flag_abortable(const uint32_t* flag, uint32_t* state, const uint32_t* aborted) {
  if (threadIdx.x == 0) {
    const uint32_t expected = state[0] + 1;
    const uint64_t start = global_ns();
    uint32_t polls = 0;
    for (;;) {
      // The cancellation word is independent of push/wait sequence state.
      // Every later queued/captured wait also exits once it is published.
      if (load_acquire_sys(aborted)) break;
      if (int32_t(load_acquire_sys(flag) - expected) >= 0) {
        state[0] = expected;
        break;
      }
      if ((++polls & 1023) == 0 && global_ns() - start > kWaitTimeoutNs) {
        printf("peer_wait_abortable: missing push or terminal abort after 60 s\n");
        __trap();
      }
    }
  }
  __syncthreads();
}

__global__ void publish_abort(uint32_t* aborted) {
  store_release_sys(aborted, 1);
}

__global__ void add_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, uint64_t count) {
  const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += stride)
    out[i] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(a[i]), __bfloat162float(b[i])));
}

__global__ void sm_copy(uint4* destination, const uint4* source, uint64_t units) {
  const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < units; i += stride)
    destination[i] = source[i];
}

unsigned default_blocks(uint64_t units, uint32_t blocks) {
  if (blocks) return blocks;
  const uint64_t wanted = (units + 1023) / 1024;
  return unsigned(std::max<uint64_t>(1, std::min<uint64_t>(wanted, 32)));
}

#define CHECK(call) do { const cudaError_t status_ = (call); if (status_ != cudaSuccess) return status_; } while (0)

// Device-side resources of one probe call; freed on every return path.
struct Side {
  int device = 0;
  cudaStream_t stream = nullptr, ingress = nullptr;
  cudaEvent_t start = nullptr, stop = nullptr, mark = nullptr;
  uint8_t *buffer = nullptr, *recv = nullptr, *ingress_device = nullptr;
  uint32_t* control = nullptr;  // [flag, send seq, arrivals, recv seq]
  cudaGraphExec_t graph = nullptr;
  ~Side() {
    cudaSetDevice(device);
    if (graph) cudaGraphExecDestroy(graph);
    if (stream) cudaStreamSynchronize(stream);
    if (ingress) cudaStreamSynchronize(ingress);
    for (auto* event : {start, stop, mark}) if (event) cudaEventDestroy(event);
    for (auto* s : {stream, ingress}) if (s) cudaStreamDestroy(s);
    for (void* p : {static_cast<void*>(buffer), static_cast<void*>(recv), static_cast<void*>(ingress_device),
         static_cast<void*>(control)}) if (p) cudaFree(p);
  }
  uint32_t* flag() const { return control; }
  uint32_t* send() const { return control + 1; }
  uint32_t* waited() const { return control + 3; }
};

struct Pinned {
  void* p = nullptr;
  ~Pinned() { if (p) cudaFreeHost(p); }
};

cudaError_t enable_peer(int device, int peer) {
  CHECK(cudaSetDevice(device));
  int capable = 0;
  CHECK(cudaDeviceCanAccessPeer(&capable, device, peer));
  if (!capable) return cudaErrorPeerAccessUnsupported;
  const cudaError_t status = cudaDeviceEnablePeerAccess(peer, 0);
  if (status == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); return cudaSuccess; }
  return status;
}

cudaError_t setup(Side& side, int device, uint64_t bytes) {
  side.device = device;
  CHECK(cudaSetDevice(device));
  CHECK(cudaStreamCreateWithFlags(&side.stream, cudaStreamNonBlocking));
  CHECK(cudaEventCreate(&side.start));
  CHECK(cudaEventCreate(&side.stop));
  CHECK(cudaEventCreateWithFlags(&side.mark, cudaEventDisableTiming));
  CHECK(cudaMalloc(&side.buffer, bytes));
  CHECK(cudaMalloc(&side.recv, bytes));
  CHECK(cudaMemset(side.buffer, 1 + device, bytes));
  CHECK(cudaMemset(side.recv, 0, bytes));
  CHECK(cudaMalloc(&side.control, 64));
  CHECK(cudaMemset(side.control, 0, 64));
  return cudaDeviceSynchronize();
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

}  // namespace

extern "C" int32_t cuteafd_peer_exchange_initialize() {
  // Loads the exchange kernels on the current device now: a lazily loaded kernel's
  // first launch may wait for the device to idle, which a spinning wait never does.
  cudaFuncAttributes attributes{};
  for (const void* kernel : {reinterpret_cast<const void*>(push_signal), reinterpret_cast<const void*>(push_planes),
       reinterpret_cast<const void*>(wait_flag),
       reinterpret_cast<const void*>(add_bf16), reinterpret_cast<const void*>(host_signal),
       reinterpret_cast<const void*>(wait_written)}) {
    const cudaError_t status = cudaFuncGetAttributes(&attributes, kernel);
    if (status != cudaSuccess) return status;
  }
  return cudaSuccess;
}

extern "C" int32_t cuteafd_peer_add_bf16_async(const void* a, const void* b, void* out, uint64_t count, void* stream) {
  const auto in_range = [count](const void* p, const void* q) {
    const auto x = reinterpret_cast<uintptr_t>(p), y = reinterpret_cast<uintptr_t>(q);
    return x + 2 * count <= y || y + 2 * count <= x;
  };
  if (!a || !b || !out || !stream || !count || count > (1ull << 36) ||
      ((reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b) | reinterpret_cast<uintptr_t>(out)) & 1) ||
      !in_range(a, out) || !in_range(b, out))
    return cudaErrorInvalidValue;
  const uint64_t blocks = std::min<uint64_t>((count + 255) / 256, 4096);
  add_bf16<<<unsigned(blocks), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(a), static_cast<const __nv_bfloat16*>(b), static_cast<__nv_bfloat16*>(out), count);
  return cudaGetLastError();
}

extern "C" int32_t cuteafd_peer_push_signal(void* destination, const void* source, uint64_t bytes,
    uint32_t* flag, uint32_t* send_state, uint32_t blocks, void* stream) {
  if (!destination || !source || !flag || !send_state || !stream || (bytes & 15) ||
      ((reinterpret_cast<uintptr_t>(destination) | reinterpret_cast<uintptr_t>(source)) & 15) || bytes > (1ull << 40))
    return cudaErrorInvalidValue;
  const uint64_t units = bytes / 16;
  push_signal<<<default_blocks(units, blocks), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<uint4*>(destination), static_cast<const uint4*>(source), units, flag, send_state);
  return cudaGetLastError();
}

extern "C" int32_t cuteafd_peer_push_planes(const cuteafd_peer_plane_t* planes, uint32_t count, uint32_t* flag,
    uint32_t* send_state, uint32_t blocks, void* stream) {
  if (!planes || !count || count > CUTEAFD_PEER_MAX_PLANES || !stream || (flag && !send_state) ||
      (!flag && send_state))
    return cudaErrorInvalidValue;
  Planes packed{};
  packed.count = count;
  for (uint32_t p = 0; p < count; ++p) {
    const cuteafd_peer_plane_t& plane = planes[p];
    const auto misaligned = (reinterpret_cast<uintptr_t>(plane.destination) | reinterpret_cast<uintptr_t>(plane.source) |
        plane.row_bytes | plane.destination_pitch | plane.source_pitch) & 15;
    if (misaligned || (plane.rows && (!plane.destination || !plane.source || !plane.row_bytes ||
        plane.row_bytes > plane.destination_pitch || plane.row_bytes > plane.source_pitch)) ||
        plane.rows > (1ull << 32) || plane.row_bytes > (1ull << 32))
      return cudaErrorInvalidValue;
    packed.plane[p] = plane;
    packed.first_unit[p + 1] = packed.first_unit[p] + (plane.rows ? plane.rows * (plane.row_bytes / 16) : 0);
  }
  const uint64_t units = packed.first_unit[count];
  for (uint32_t p = count; p < CUTEAFD_PEER_MAX_PLANES; ++p) packed.first_unit[p + 1] = units;
  push_planes<<<default_blocks(units, blocks), 256, 0, static_cast<cudaStream_t>(stream)>>>(packed, flag, send_state);
  return cudaGetLastError();
}

extern "C" int32_t cuteafd_peer_wait(const uint32_t* flag, uint32_t* recv_state, void* stream) {
  if (!flag || !recv_state || !stream) return cudaErrorInvalidValue;
  wait_flag<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(flag, recv_state);
  return cudaGetLastError();
}

extern "C" int32_t cuteafd_host_signal(uint32_t* flag, uint32_t* send_state, uint32_t* descriptor,
    const uint32_t* words, void* stream) {
  if (!flag || !send_state || !descriptor || !words || !stream) return cudaErrorInvalidValue;
  host_signal<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(flag, send_state, descriptor,
      make_uint4(words[0], words[1], words[2], words[3]));
  return cudaGetLastError();
}

extern "C" int32_t cuteafd_spark_wait_written(const uint64_t* flags, uint32_t ranks, uint32_t stride_words,
    uint32_t* state, uint32_t* error, void* stream) {
  if (!flags || !state || !error || !stream || !ranks || ranks > 32 || !stride_words) return cudaErrorInvalidValue;
  wait_written<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(flags, ranks, stride_words, state, error);
  return cudaGetLastError();
}

extern "C" int32_t cuteafd_peer_abort_initialize() {
  cudaFuncAttributes attributes{};
  for (const void* kernel : {reinterpret_cast<const void*>(wait_flag_abortable),
       reinterpret_cast<const void*>(publish_abort)}) {
    const cudaError_t status = cudaFuncGetAttributes(&attributes, kernel);
    if (status != cudaSuccess) return status;
  }
  return cudaSuccess;
}

extern "C" int32_t cuteafd_peer_wait_abortable(const uint32_t* flag, uint32_t* recv_state,
    const uint32_t* aborted, void* stream) {
  if (!flag || !recv_state || !aborted || !stream) return cudaErrorInvalidValue;
  wait_flag_abortable<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(flag, recv_state, aborted);
  return cudaGetLastError();
}

extern "C" int32_t cuteafd_peer_abort_publish(uint32_t* aborted, void* independent_stream) {
  if (!aborted || !independent_stream) return cudaErrorInvalidValue;
  publish_abort<<<1, 1, 0, static_cast<cudaStream_t>(independent_stream)>>>(aborted);
  return cudaGetLastError();
}

extern "C" int32_t cuteafd_p2p_probe(int32_t a, int32_t b, uint64_t bytes, int32_t test, uint32_t ingress,
    uint32_t iterations, uint32_t blocks, double* microseconds) {
  if (!microseconds || a == b || a < 0 || b < 0 || !bytes || (bytes & 15) || !iterations || test < 0 || test > 10)
    return cudaErrorInvalidValue;
  int previous = 0;
  CHECK(cudaGetDevice(&previous));
  CHECK(enable_peer(a, b));
  CHECK(enable_peer(b, a));
  Side sa, sb;
  CHECK(setup(sa, a, bytes));
  CHECK(setup(sb, b, bytes));
  Pinned bounce;
  if (test == 3) CHECK(cudaHostAlloc(&bounce.p, bytes, cudaHostAllocPortable));
  const uint64_t units = bytes / 16;
  const unsigned grid = default_blocks(units, blocks);
  auto u4 = [](uint8_t* p) { return reinterpret_cast<uint4*>(p); };

  // One operation (or one round) enqueued on the streams.
  auto enqueue = [&]() -> cudaError_t {
    switch (test) {
      case 0:
        CHECK(cudaSetDevice(b));
        return cudaMemcpyAsync(sb.recv, sa.buffer, bytes, cudaMemcpyDefault, sb.stream);
      case 1:
        CHECK(cudaSetDevice(b));
        sm_copy<<<grid, 256, 0, sb.stream>>>(u4(sb.recv), u4(sa.buffer), units);
        return cudaGetLastError();
      case 2:
        CHECK(cudaSetDevice(a));
        sm_copy<<<grid, 256, 0, sa.stream>>>(u4(sb.recv), u4(sa.buffer), units);
        return cudaGetLastError();
      case 4: case 5: case 10:
        CHECK(cudaSetDevice(a));
        if (test == 5) sm_copy<<<grid, 256, 0, sa.stream>>>(u4(sb.recv), u4(sa.buffer), units);
        else CHECK(cudaMemcpyAsync(sb.recv, sa.buffer, bytes, cudaMemcpyDefault, sa.stream));
        CHECK(cudaEventRecord(sa.mark, sa.stream));
        CHECK(cudaSetDevice(b));
        CHECK(cudaStreamWaitEvent(sb.stream, sa.mark, 0));
        if (test == 5) sm_copy<<<grid, 256, 0, sb.stream>>>(u4(sa.recv), u4(sb.buffer), units);
        else CHECK(cudaMemcpyAsync(sa.recv, sb.buffer, bytes, cudaMemcpyDefault, sb.stream));
        CHECK(cudaEventRecord(sb.mark, sb.stream));
        CHECK(cudaSetDevice(a));
        return cudaStreamWaitEvent(sa.stream, sb.mark, 0);
      case 6: case 7:
        CHECK(cudaSetDevice(a));
        push_signal<<<grid, 256, 0, sa.stream>>>(u4(sb.recv), u4(sa.buffer), units, sb.flag(), sa.send());
        wait_flag<<<1, 32, 0, sa.stream>>>(sa.flag(), sa.waited());
        CHECK(cudaGetLastError());
        CHECK(cudaSetDevice(b));
        wait_flag<<<1, 32, 0, sb.stream>>>(sb.flag(), sb.waited());
        push_signal<<<grid, 256, 0, sb.stream>>>(u4(sa.recv), u4(sb.buffer), units, sa.flag(), sb.send());
        return cudaGetLastError();
      case 8: case 9:
        CHECK(cudaSetDevice(a));
        push_signal<<<grid, 256, 0, sa.stream>>>(u4(sb.recv), u4(sa.buffer), units, sb.flag(), sa.send());
        wait_flag<<<1, 32, 0, sa.stream>>>(sa.flag(), sa.waited());
        CHECK(cudaGetLastError());
        CHECK(cudaSetDevice(b));
        push_signal<<<grid, 256, 0, sb.stream>>>(u4(sa.recv), u4(sb.buffer), units, sa.flag(), sb.send());
        wait_flag<<<1, 32, 0, sb.stream>>>(sb.flag(), sb.waited());
        return cudaGetLastError();
    }
    return cudaErrorInvalidValue;
  };
  const bool ping_pong = test >= 4 && test <= 7 || test == 10;
  const bool graphs = test == 7 || test == 9 || test == 10;
  const uint32_t batch = std::min<uint32_t>(iterations, graphs ? 256 : iterations);

  if (graphs) {
    // Capture `batch` operations: per device (flags) or across both (events).
    if (test == 10) {
      CHECK(cudaSetDevice(a));
      CHECK(cudaStreamBeginCapture(sa.stream, cudaStreamCaptureModeRelaxed));
      for (uint32_t i = 0; i < batch; ++i) CHECK(enqueue());
      cudaGraph_t graph = nullptr;
      CHECK(cudaSetDevice(a));
      CHECK(cudaStreamEndCapture(sa.stream, &graph));
      const cudaError_t status = cudaGraphInstantiate(&sa.graph, graph, 0);
      cudaGraphDestroy(graph);
      CHECK(status);
    } else {
      for (int pass = 0; pass < 2; ++pass) {
        Side& side = pass ? sb : sa;
        CHECK(cudaSetDevice(side.device));
        CHECK(cudaStreamBeginCapture(side.stream, cudaStreamCaptureModeRelaxed));
        for (uint32_t i = 0; i < batch; ++i) {
          Side& peer = pass ? sa : sb;
          if (test == 7 && pass == 1) wait_flag<<<1, 32, 0, side.stream>>>(side.flag(), side.waited());
          push_signal<<<grid, 256, 0, side.stream>>>(u4(peer.recv), u4(side.buffer), units, peer.flag(), side.send());
          if (!(test == 7 && pass == 1)) wait_flag<<<1, 32, 0, side.stream>>>(side.flag(), side.waited());
        }
        cudaGraph_t graph = nullptr;
        CHECK(cudaStreamEndCapture(side.stream, &graph));
        const cudaError_t status = cudaGraphInstantiate(&side.graph, graph, 0);
        cudaGraphDestroy(graph);
        CHECK(status);
      }
    }
  }

  auto launch_batch = [&]() -> cudaError_t {
    if (!graphs) {
      for (uint32_t i = 0; i < batch; ++i) CHECK(enqueue());
      return cudaSuccess;
    }
    if (test == 10) { CHECK(cudaSetDevice(a)); return cudaGraphLaunch(sa.graph, sa.stream); }
    CHECK(cudaSetDevice(b));
    CHECK(cudaGraphLaunch(sb.graph, sb.stream));
    CHECK(cudaSetDevice(a));
    return cudaGraphLaunch(sa.graph, sa.stream);
  };

  // Host->device ingress streams into the selected devices.
  Pinned ingress_host;
  const uint64_t ingress_bytes = 256ull << 20;
  if (ingress) {
    CHECK(cudaHostAlloc(&ingress_host.p, ingress_bytes, cudaHostAllocPortable));
    for (Side* side : {&sa, &sb}) {
      if (!(ingress & (side == &sa ? 1u : 2u))) continue;
      CHECK(cudaSetDevice(side->device));
      CHECK(cudaStreamCreateWithFlags(&side->ingress, cudaStreamNonBlocking));
      CHECK(cudaMalloc(&side->ingress_device, ingress_bytes));
    }
  }
  // A host thread per loaded device keeps two copies queued until stopped.
  std::atomic<bool> stop_ingress{false};
  std::vector<std::thread> feeders;
  cudaError_t feeder_status = cudaSuccess;
  auto start_ingress = [&]() {
    stop_ingress = false;
    for (Side* side : {&sa, &sb}) {
      if (!side->ingress) continue;
      feeders.emplace_back([&, side]() {
        cudaSetDevice(side->device);
        cudaEvent_t done[2];
        for (auto& event : done) cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
        for (uint64_t i = 0; !stop_ingress; ++i) {
          if (i >= 2) cudaEventSynchronize(done[i & 1]);
          const cudaError_t status = cudaMemcpyAsync(side->ingress_device, ingress_host.p, ingress_bytes,
              cudaMemcpyHostToDevice, side->ingress);
          if (status != cudaSuccess) { feeder_status = status; break; }
          cudaEventRecord(done[i & 1], side->ingress);
        }
        cudaStreamSynchronize(side->ingress);
        for (auto& event : done) cudaEventDestroy(event);
      });
    }
    // Let the copies reach full rate before timing.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  };
  auto stop_feeders = [&]() {
    stop_ingress = true;
    for (auto& feeder : feeders) feeder.join();
    feeders.clear();
  };

  auto run = [&](double& elapsed_us) -> cudaError_t {
    if (test == 3) {
      // Host-timed: D2H on A, H2D on B, synchronized per operation.
      std::vector<double> times;
      for (uint32_t i = 0; i < iterations; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        CHECK(cudaSetDevice(a));
        CHECK(cudaMemcpyAsync(bounce.p, sa.buffer, bytes, cudaMemcpyDeviceToHost, sa.stream));
        CHECK(cudaEventRecord(sa.mark, sa.stream));
        CHECK(cudaSetDevice(b));
        CHECK(cudaStreamWaitEvent(sb.stream, sa.mark, 0));
        CHECK(cudaMemcpyAsync(sb.recv, bounce.p, bytes, cudaMemcpyHostToDevice, sb.stream));
        CHECK(cudaStreamSynchronize(sb.stream));
        times.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
      }
      elapsed_us = median(times) * iterations;
      return cudaSuccess;
    }
    // Timed on the device that issues the first operation of each round.
    Side& timer = (test == 0 || test == 1) ? sb : sa;
    CHECK(cudaSetDevice(timer.device));
    CHECK(cudaEventRecord(timer.start, timer.stream));
    for (uint32_t done = 0; done < iterations; done += batch) CHECK(launch_batch());
    CHECK(cudaSetDevice(timer.device));
    CHECK(cudaEventRecord(timer.stop, timer.stream));
    CHECK(cudaSetDevice(b));
    CHECK(cudaStreamSynchronize(sb.stream));
    CHECK(cudaSetDevice(a));
    CHECK(cudaStreamSynchronize(sa.stream));
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, timer.start, timer.stop));
    elapsed_us = double(ms) * 1e3;
    return cudaSuccess;
  };

  // Round the iterations to whole graph batches, warm up, then 5 timed repeats.
  iterations = (iterations + batch - 1) / batch * batch;
  double warm = 0;
  CHECK(run(warm));
  std::vector<double> samples;
  for (int repeat = 0; repeat < 5; ++repeat) {
    if (ingress) start_ingress();
    double elapsed = 0;
    const cudaError_t status = run(elapsed);
    if (ingress) stop_feeders();
    CHECK(status);
    CHECK(feeder_status);
    samples.push_back(elapsed / iterations / (ping_pong ? 2.0 : 1.0));
  }
  // The pushed bytes landed (A's pattern in B's buffer, B's in A's for two-way tests).
  uint8_t probe_a = 0, probe_b = 0;
  CHECK(cudaSetDevice(b));
  CHECK(cudaMemcpy(&probe_b, sb.recv + bytes - 1, 1, cudaMemcpyDeviceToHost));
  CHECK(cudaSetDevice(a));
  CHECK(cudaMemcpy(&probe_a, sa.recv + bytes - 1, 1, cudaMemcpyDeviceToHost));
  if (probe_b != uint8_t(1 + a) || (test >= 4 && test != 3 && probe_a != uint8_t(1 + b)))
    return cudaErrorUnknown;
  *microseconds = median(samples);
  CHECK(cudaSetDevice(previous));
  return cudaSuccess;
}

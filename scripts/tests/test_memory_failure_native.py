"""CPU proof of native allocation-boundary capture before constructor cleanup."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_native_failure_wrappers_capture_live_state_and_preserve_raw_codes(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    source = tmp_path / "memory_failure.cc"
    source.write_text(r'''
#include "memory_failure.h"
#include <cassert>
#include <cstring>
static int live = 0, captured_live = 0, calls = 0, code = 0;
static bool graph_alive = true;
static const char* api = nullptr;
void event(int32_t, uintptr_t, size_t, int32_t released, const char* file, int32_t line,
    uint32_t kind, const char* owner) {
  // An allocation names its tag; a release names none.
  if (!released) assert(file && line > 0 && kind == CUTEAFD_MEMORY_KIND_KV && owner && std::strcmp(owner, "test/kv") == 0);
  else assert(kind == 0 && owner == nullptr);
  live += released ? -1 : 1;
}
void failure(const cuteafd_memory_failure_t* trigger) {
  captured_live = live; code = trigger->code; api = trigger->api;
  if (std::strcmp(api, "cudaGraphInstantiate") == 0) {
    assert(graph_alive && trigger->handle == 0x1234 && !trigger->bytes_known);
  }
}
int allocate(void** output) {
  ++calls;
  if (calls == 2) return 2;
  *output = reinterpret_cast<void*>(0x1000); return 0;
}
int graph_failure() {
  struct Graph { ~Graph() { graph_alive = false; } } graph;
  return CUTEAFD_CHECK_MEMORY("cudaGraphInstantiate", "cuda", 2, 0, 0, reinterpret_cast<void*>(0x1234));
}
int main() {
  cuteafd_set_memory_event_callback_v3(event);
  cuteafd_set_memory_failure_callback(failure);
  void* first = nullptr;
  void* second = nullptr;
  assert(CUTEAFD_TRACK_ALLOC("cudaMalloc", &first, 4096, 0, 0, "test/kv", allocate(&first)) == 1);
  assert(CUTEAFD_TRACK_ALLOC("cudaMalloc", &first, 4096, 0, CUTEAFD_MEMORY_KIND_KV, "", allocate(&first)) == 1);
  assert(calls == 0 && first == nullptr && live == 0);
  assert(CUTEAFD_TRACK_MALLOC(64, 0, "test/kv") == nullptr);
  assert(CUTEAFD_TRACK_ALLOC("cudaMalloc", &first, 4096, 0, CUTEAFD_MEMORY_KIND_KV, "test/kv", allocate(&first)) == 0);
  assert(CUTEAFD_TRACK_ALLOC("cudaMalloc", &second, 8192, 0, CUTEAFD_MEMORY_KIND_KV, "test/kv", allocate(&second)) == 2);
  uint32_t kind = 0; char owner[32] = {}; size_t required = 0;
  assert(cuteafd_last_allocation_tag(&kind, owner, sizeof(owner), &required) == 0);
  assert(kind == CUTEAFD_MEMORY_KIND_KV && std::strcmp(owner, "test/kv") == 0);
  assert(calls == 2 && captured_live == 1 && code == 2);
  assert(CUTEAFD_TRACK_FREE(first, 0) == 0 && live == 0);
  assert(graph_failure() == 2);
  assert(!graph_alive);
  assert(CUTEAFD_CHECK_MEMORY("cublasCreate", "cublas", 3, 0, 0, nullptr) == 3);
  assert(code == 3 && std::strcmp(api, "cublasCreate") == 0);
  assert(!graph_alive);
  assert(CUTEAFD_CHECK_MEMORY("cudaProbe", "cuda", 801, 0, 0, nullptr) == 801);
  assert(code == 3);
  size_t bytes = 1;
  void* host = CUTEAFD_MALLOC(bytes++);
  assert(bytes == 2);
  free(host);
}
''')
    executable = tmp_path / "memory_failure"
    built = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                            "-I", str(ROOT / "native/shared/include"), str(source), str(ROOT / "native/shared/src/memory_failure.cc"),
                            "-o", str(executable)], capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(executable)], capture_output=True, text=True)
    assert ran.returncode == 0, ran.stderr


def test_allocation_failure_format_preserves_original_error(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    source = tmp_path / "allocation_failure.cc"
    source.write_text(r'''
#include "memory_failure.h"
#include <cassert>
#include <limits>
int main() {
  cuteafd_allocation_diagnostics diagnostics;
  diagnostics.device = 1;
  diagnostics.free_bytes = 4096;
  diagnostics.total_bytes = 98304;
  const std::string original = "cudaErrorMemoryAllocation (out of memory)";
  const auto format = [&](const char* owner) {
    return cuteafd_format_allocation_failure("cudaMalloc failed", 8192, owner, original, diagnostics);
  };
  assert(format("kv/pages") == "cudaMalloc failed: cudaErrorMemoryAllocation (out of memory); "
      "requested_bytes=8192; owner=kv/pages; device=1; free_bytes=4096; total_bytes=98304");
  diagnostics.memory_error = "cudaErrorUnknown (unknown error)";
  assert(format("weights/core") == "cudaMalloc failed: cudaErrorMemoryAllocation (out of memory); "
      "requested_bytes=8192; owner=weights/core; device=1; free/total unavailable: cudaErrorUnknown (unknown error)");
  diagnostics.device_error = "cudaErrorNoDevice (no CUDA-capable device is detected)";
  const auto unavailable = format(nullptr);
  assert(unavailable == "cudaMalloc failed: cudaErrorMemoryAllocation (out of memory); "
      "requested_bytes=8192; owner=unavailable; device unavailable: cudaErrorNoDevice (no CUDA-capable device is detected); "
      "free/total unavailable: cudaErrorUnknown (unknown error)");
  assert(format("") == unavailable);
  diagnostics.memory_error.clear();
  assert(format("host/staging").find("free_bytes=4096; total_bytes=98304") != std::string::npos);
  const size_t large = std::numeric_limits<size_t>::max();
  const auto wide = cuteafd_format_allocation_failure("cudaHostAlloc failed", large, "transport/rings", original, diagnostics);
  assert(wide.find("requested_bytes=" + std::to_string(large)) != std::string::npos);
  assert(wide.find("cudaHostAlloc failed: " + original) == 0);
}
''')
    executable = tmp_path / "allocation_failure"
    built = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                            "-I", str(ROOT / "native/shared/include"), str(source),
                            str(ROOT / "native/shared/src/memory_failure.cc"), "-o", str(executable)],
                           capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(executable)], capture_output=True, text=True)
    assert ran.returncode == 0, ran.stderr


def test_graph_capture_failure_clears_last_error_before_peer_eager_retry(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    native = (ROOT / "native/shared/src/cuteafd_native.cc").read_text()
    # Compile the production entry points against a CPU runtime which retains
    # recoverable errors exactly until cudaGetLastError consumes them.
    entry_points = native.split('extern "C" cuteafd_status_t cuteafd_cuda_graph_begin_capture(', 1)[1]
    entry_points = 'extern "C" cuteafd_status_t cuteafd_cuda_graph_begin_capture(' + entry_points.split(
        'extern "C" cuteafd_status_t cuteafd_cuda_graph_launch(', 1)[0]
    # These legacy-wrapper failure oracles stub only their original CUDA APIs;
    # source-only capture/update has separate ownership tests.
    begin = entry_points.index("#if CUTEAFD_NATIVE_ENABLE_CUDA\nnamespace {\nstruct GraphHandle")
    end = entry_points.index('extern "C" cuteafd_status_t cuteafd_cuda_graph_end_capture_retained(', begin)
    entry_points = entry_points[:begin] + entry_points[end:]
    diagnostics = (ROOT / "native/shared/include/cuteafd_graph_diagnostics.h").read_text()
    clear_error = diagnostics.split("inline cudaError_t diagnostic_cuda", 1)[1].split("using GraphEdgesQuery", 1)[0]
    source = tmp_path / "graph_capture_failure.cc"
    source.write_text(r'''
#include "cuteafd_native.h"
#include <cassert>
#include <string>
#define CUTEAFD_NATIVE_ENABLE_CUDA 1
#define CUTEAFD_CHECK_MEMORY(api, domain, call, bytes, known, handle) (call)
using cudaError_t = int;
using cudaStream_t = void*;
using cudaGraph_t = void*;
using cudaGraphExec_t = void*;
constexpr int cudaSuccess = 0, cudaErrorMemoryAllocation = 2, cudaErrorStreamCaptureInvalidated = 901;
constexpr int cudaErrorIllegalAddress = 700, cudaStreamCaptureModeThreadLocal = 1;
static int last_error = 0, sticky_error = 0, injected = 0, point = 0, live_graphs = 0, live_execs = 0;
static size_t graph_nodes = 3;
static int instantiate_calls = 0;
static bool active = false, partial_exec = false, census = false;
static std::string message;
int cudaGetLastError() { int status = sticky_error ? sticky_error : last_error; last_error = 0; return status; }
int outcome(int at) {
  const int status = sticky_error ? sticky_error : (point == at ? injected : 0);
  if (status) last_error = status; // Successful CUDA calls do not consume a previous error.
  return status;
}
int cudaStreamBeginCapture(void*, int) { int status = outcome(1); if (!status) active = true; return status; }
int cudaStreamEndCapture(void*, void** graph) {
  assert(active); active = false;
  const int status = outcome(2);
  if (!status) { *graph = reinterpret_cast<void*>(0x1000); ++live_graphs; }
  return status;
}
int cudaGraphInstantiate(void** exec, void*, int) {
  ++instantiate_calls;
  const int status = outcome(3);
  if (!status || partial_exec) { *exec = reinterpret_cast<void*>(0x2000); ++live_execs; }
  return status;
}
int cudaGraphUpload(void*, void*) { return outcome(6); }
int cudaGraphDestroy(void*) { assert(live_graphs > 0); --live_graphs; return outcome(4); }
int cudaGraphExecDestroy(void*) { assert(live_execs > 0); --live_execs; return outcome(5); }
int cudaGraphGetNodes(void*, void**, size_t* nodes) { *nodes = graph_nodes; return 0; }
static cuteafd_status_t end_capture_retained(void*, cuteafd_cuda_graph_capture_info_t*, bool);
int cudaMemGetInfo(size_t* free, size_t* total) { *free = 1 << 20; *total = 2 << 20; return 0; }
bool graph_census_enabled() { return census; }
void record_graph_census(void*, void*, int, size_t, int, size_t) {}
cuteafd_status_t ok() { message.clear(); return CUTEAFD_STATUS_OK; }
cuteafd_status_t fail(cuteafd_status_t status, const char* action) { message = action; return status; }
cuteafd_status_t fail_cuda(cuteafd_status_t status, const char* action, int error) {
  message = std::string(action) + ": " + std::to_string(error); return status;
}
cuteafd_status_t fill_cuda_graph_capture_info(void* graph, void* exec, cuteafd_cuda_graph_capture_info_t* out) {
  out->graph = graph; out->graph_exec = exec; out->node_count = 3; return ok();
}
namespace cuteafd {
''' + "inline cudaError_t diagnostic_cuda" + clear_error + r'''
int inspect_graph_topology(void*, size_t* count, size_t* width, int* spans) {
  *count = 3; *width = 1; *spans = 1; return 0;
}
}
''' + entry_points + r'''
int peer_wait_abortable() { return cudaGetLastError(); }
int main() {
  void* stream = reinterpret_cast<void*>(0x3000);
  // A rejected recording must never reach instantiate, even under the census.
  for (bool with_census : {false, true}) for (size_t count : {0u, 1u, 2u}) {
    census = with_census;
    graph_nodes = count;
    for (int wrapper : {0, 1, 2}) {
      assert(cuteafd_cuda_graph_begin_capture(stream) == CUTEAFD_STATUS_OK);
      void* exec = nullptr; size_t nodes = 99, width = 99; int spans = 99;
      cuteafd_cuda_graph_capture_info_t info{};
      const int before = instantiate_calls;
      const auto status = wrapper == 0 ? cuteafd_cuda_graph_end_capture(stream, &exec)
        : wrapper == 1 ? cuteafd_cuda_graph_end_capture_topology(stream, &exec, &nodes, &width, &spans)
        : cuteafd_cuda_graph_end_capture_retained(stream, &info);
      assert(status == CUTEAFD_STATUS_GRAPH_TOO_SMALL && instantiate_calls == before);
      assert(!active && live_graphs == 0 && live_execs == 0 && exec == nullptr);
      if (wrapper == 1) assert(nodes == count && width == 0 && spans == 0);
      if (wrapper == 2) assert(info.node_count == count && !info.graph && !info.graph_exec);
      assert(peer_wait_abortable() == 0);
    }
    assert(cuteafd_cuda_graph_begin_capture(stream) == CUTEAFD_STATUS_OK);
    void* exec = nullptr;
    const int before = instantiate_calls;
    assert(cuteafd_cuda_graph_end_capture_for_test(stream, &exec) == CUTEAFD_STATUS_OK);
    assert(instantiate_calls == before + 1 && live_graphs == 0 && live_execs == 1);
    assert(cudaGraphExecDestroy(exec) == 0 && peer_wait_abortable() == 0);
  }
  graph_nodes = 3;
  for (bool with_census : {false, true}) {
    census = with_census;
    // Both ranks share this host thread's last-error slot. A valid eager wait
    // after either rank's failed instantiate must not poison the TP2 submission.
    for (int failing_rank : {0, 1}) {
      int captures[2] = {}, eager[2] = {}, cancelled[2] = {};
      for (int rank : {0, 1}) {
        point = rank == failing_rank ? 3 : 0; injected = cudaErrorMemoryAllocation;
        assert(cuteafd_cuda_graph_begin_capture(stream) == CUTEAFD_STATUS_OK);
        void* exec = nullptr; size_t nodes = 0, width = 0; int spans = 0;
        const auto status = cuteafd_cuda_graph_end_capture_topology(stream, &exec, &nodes, &width, &spans);
        assert(!active && live_graphs == 0);
        if (status != CUTEAFD_STATUS_OK) {
          assert(message == "cudaGraphInstantiate failed; lower GPU_MEMORY_CAP or increase GPU_RESERVE to leave graph headroom: 2" && exec == nullptr);
          ++cancelled[rank]; ++eager[rank];
        } else {
          assert(nodes == 3 && width == 1 && spans == 1);
          ++captures[rank]; assert(cudaGraphExecDestroy(exec) == 0);
        }
        assert(peer_wait_abortable() == 0 && live_execs == 0);
      }
      assert(cancelled[failing_rank] == 1 && eager[failing_rank] == 1);
      assert(captures[1 - failing_rank] == 1); // Never rerun the peer's queued work.
      assert(peer_wait_abortable() == 0); // A subsequent submission remains usable.
    }
  }
  census = false;
  // All three end-capture APIs share recovery, including partially returned execs.
  for (int wrapper : {0, 1, 2}) for (int failure : {2, 3, 4, 6}) {
    point = failure; injected = failure == 2 ? cudaErrorStreamCaptureInvalidated : cudaErrorMemoryAllocation;
    partial_exec = true;
    assert(cuteafd_cuda_graph_begin_capture(stream) == CUTEAFD_STATUS_OK);
    void* exec = nullptr; size_t nodes = 0, width = 0; int spans = 0;
    cuteafd_cuda_graph_capture_info_t info{};
    const auto status = wrapper == 0 ? cuteafd_cuda_graph_end_capture(stream, &exec)
      : wrapper == 1 ? cuteafd_cuda_graph_end_capture_topology(stream, &exec, &nodes, &width, &spans)
      : cuteafd_cuda_graph_end_capture_retained(stream, &info);
    // The retained API transfers its source graph, so it has no destroy boundary.
    if (wrapper == 2 && failure == 4) {
      assert(status == CUTEAFD_STATUS_OK);
      cudaGraphDestroy(info.graph); cudaGraphExecDestroy(info.graph_exec); cudaGetLastError();
    } else {
      const auto expected = (failure == 3 || failure == 6) ? CUTEAFD_STATUS_ALLOCATION_FAILED : CUTEAFD_STATUS_INTERNAL_ERROR;
      assert(status == expected && exec == nullptr && info.graph == nullptr);
    }
    assert(!active && live_graphs == 0 && live_execs == 0 && peer_wait_abortable() == 0);
  }
  point = 1; injected = cudaErrorMemoryAllocation;
  assert(cuteafd_cuda_graph_begin_capture(stream) == CUTEAFD_STATUS_INTERNAL_ERROR);
  assert(!active && message == "cudaStreamBeginCapture failed: 2" && peer_wait_abortable() == 0);
  // Clearing a recoverable error slot cannot erase a genuine sticky GPU fault.
  point = 0;
  assert(cuteafd_cuda_graph_begin_capture(stream) == CUTEAFD_STATUS_OK);
  sticky_error = cudaErrorIllegalAddress;
  cuteafd_cuda_graph_capture_info_t info{};
  assert(cuteafd_cuda_graph_end_capture_retained(stream, &info) == CUTEAFD_STATUS_INTERNAL_ERROR);
  assert(!active && peer_wait_abortable() == cudaErrorIllegalAddress);
}
''')
    executable = tmp_path / "graph_capture_failure"
    built = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                            "-I", str(ROOT / "native/shared/include"), str(source), "-o", str(executable)],
                           capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(executable)], capture_output=True, text=True)
    assert ran.returncode == 0, ran.stderr


def test_transferable_signature_handles_d2d_memcpy_nodes(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    native = (ROOT / "native/shared/src/cuteafd_native.cc").read_text()
    signature_fn = native.split("std::vector<uint64_t> transferable_signature(", 1)[1]
    signature_fn = "std::vector<uint64_t> transferable_signature(" + signature_fn.split("\n}\n}\n#endif", 1)[0] + "\n}\n"
    diagnostics = (ROOT / "native/shared/include/cuteafd_graph_diagnostics.h").read_text()
    clear_error = diagnostics.split("inline cudaError_t diagnostic_cuda", 1)[1].split("using GraphEdgesQuery", 1)[0]
    source = tmp_path / "transferable_signature_d2d.cc"
    source.write_text(r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <vector>
using cudaError_t = int;
using CUresult = int;
using cudaGraph_t = struct FakeGraph*;
using cudaGraphNode_t = struct FakeNode*;
using CUgraphNode = cudaGraphNode_t;
constexpr int cudaSuccess = 0, CUDA_SUCCESS = 0, cudaErrorInvalidValue = 1;
enum cudaGraphNodeType { cudaGraphNodeTypeKernel = 0, cudaGraphNodeTypeMemcpy = 1, cudaGraphNodeTypeMemset = 2, cudaGraphNodeTypeHost = 3 };
enum cudaMemcpyKind { cudaMemcpyHostToHost = 0, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3, cudaMemcpyDefault = 4 };
enum cudaMemoryType { cudaMemoryTypeUnregistered = 0, cudaMemoryTypeHost = 1, cudaMemoryTypeDevice = 2 };
struct cudaPos { size_t x = 0, y = 0, z = 0; };
struct cudaPitchedPtr { void* ptr = nullptr; size_t pitch = 0, xsize = 0, ysize = 0; };
struct cudaExtent { size_t width = 0, height = 0, depth = 0; };
struct cudaMemcpy3DParms {
  void* srcArray = nullptr;
  cudaPos srcPos{};
  cudaPitchedPtr srcPtr{};
  void* dstArray = nullptr;
  cudaPos dstPos{};
  cudaPitchedPtr dstPtr{};
  cudaExtent extent{};
  cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;
};
struct cudaMemsetParams {
  void* dst = nullptr;
  size_t pitch = 0;
  unsigned int value = 0;
  unsigned int elementSize = 0;
  size_t width = 0;
  size_t height = 0;
};
struct CUDA_KERNEL_NODE_PARAMS { void* func = nullptr; };
struct cudaPointerAttributes { cudaMemoryType type = cudaMemoryTypeUnregistered; int device = -1; };
struct cudaGraphEdgeData { uint8_t from_port = 0, to_port = 0, type = 0; };
struct FakeNode {
  cudaGraphNodeType kind = cudaGraphNodeTypeKernel;
  void* func = nullptr;
  cudaMemcpy3DParms memcpy{};
  cudaMemsetParams memset{};
};
struct FakeGraph {
  std::vector<FakeNode*> nodes;
  std::vector<std::pair<FakeNode*, FakeNode*>> edges;
};
static int current_device = 0;
static std::unordered_map<const void*, cudaPointerAttributes> ptr_attrs;
int cudaGetLastError() { return cudaSuccess; }
int cudaGetDevice(int* dev) { *dev = current_device; return cudaSuccess; }
int cudaGraphGetNodes(cudaGraph_t g, cudaGraphNode_t* out, size_t* count) {
  if (!out) { *count = g->nodes.size(); return cudaSuccess; }
  const size_t n = std::min(*count, g->nodes.size());
  for (size_t i = 0; i < n; ++i) out[i] = g->nodes[i];
  *count = n;
  return cudaSuccess;
}
int cudaGraphNodeGetType(cudaGraphNode_t n, cudaGraphNodeType* kind) { *kind = n->kind; return cudaSuccess; }
CUresult cuGraphKernelNodeGetParams(CUgraphNode n, CUDA_KERNEL_NODE_PARAMS* p) { p->func = n->func; return CUDA_SUCCESS; }
int cudaGraphMemcpyNodeGetParams(cudaGraphNode_t n, cudaMemcpy3DParms* p) { *p = n->memcpy; return cudaSuccess; }
int cudaGraphMemsetNodeGetParams(cudaGraphNode_t n, cudaMemsetParams* p) { *p = n->memset; return cudaSuccess; }
int cudaPointerGetAttributes(cudaPointerAttributes* attr, const void* ptr) {
  auto it = ptr_attrs.find(ptr);
  if (it == ptr_attrs.end()) return cudaErrorInvalidValue;
  *attr = it->second;
  return cudaSuccess;
}
int cudaGraphGetEdges(cudaGraph_t g, cudaGraphNode_t* from, cudaGraphNode_t* to, cudaGraphEdgeData* data, size_t* count) {
  const size_t n = std::min(*count, g->edges.size());
  for (size_t i = 0; i < n; ++i) { from[i] = g->edges[i].first; to[i] = g->edges[i].second; data[i] = {}; }
  *count = g->edges.size();
  return cudaSuccess;
}
namespace cuteafd {
''' + "inline cudaError_t diagnostic_cuda" + clear_error + r'''
}
''' + signature_fn + r'''
int main() {
  std::vector<uint8_t> old_src(4096, 0x11), old_dst(4096, 0), new_src(4096, 0x5a), new_dst(4096, 0);
  std::vector<uint8_t> host_buf(4096, 0x22), peer_buf(4096, 0x33);
  ptr_attrs[old_src.data()] = {cudaMemoryTypeDevice, 0};
  ptr_attrs[old_dst.data()] = {cudaMemoryTypeDevice, 0};
  ptr_attrs[new_src.data()] = {cudaMemoryTypeDevice, 0};
  ptr_attrs[new_dst.data()] = {cudaMemoryTypeDevice, 0};
  ptr_attrs[host_buf.data()] = {cudaMemoryTypeHost, -1};
  ptr_attrs[peer_buf.data()] = {cudaMemoryTypeDevice, 1};
  auto make_d2d = [&](void* src, void* dst, size_t bytes, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice) {
    FakeNode k1{cudaGraphNodeTypeKernel, reinterpret_cast<void*>(0x101), {}};
    FakeNode m{cudaGraphNodeTypeMemcpy, nullptr, {}};
    m.memcpy.srcPtr.ptr = src; m.memcpy.dstPtr.ptr = dst;
    m.memcpy.extent = {bytes, 1, 1}; m.memcpy.kind = kind;
    FakeNode k2{cudaGraphNodeTypeKernel, reinterpret_cast<void*>(0x202), {}};
    return std::array<FakeNode, 3>{k1, m, k2};
  };
  auto run_sig = [&](std::array<FakeNode, 3> nodes) {
    FakeGraph g{{&nodes[0], &nodes[1], &nodes[2]}, {{&nodes[0], &nodes[1]}, {&nodes[1], &nodes[2]}}};
    return transferable_signature(&g, 3);
  };
  auto donor_nodes = make_d2d(old_src.data(), old_dst.data(), 4096);
  auto target_nodes = make_d2d(new_src.data(), new_dst.data(), 4096);
  auto default_nodes = make_d2d(new_src.data(), new_dst.data(), 4096, cudaMemcpyDefault);
  const auto sig_donor = run_sig(donor_nodes);
  const auto sig_target = run_sig(target_nodes);
  const auto sig_default = run_sig(default_nodes);
  assert(!sig_donor.empty());
  assert(sig_donor == sig_target);
  assert(sig_donor == sig_default);
  // Signature includes node type, direction (D2D) and device, omitting 1D byte length so
  // different batch row counts share donors across cudaGraphExecUpdate.
  assert(sig_donor[3] == cudaGraphNodeTypeMemcpy && sig_donor[4] == cudaMemcpyDeviceToDevice &&
         sig_donor[5] == 0);
  // Different 1D copy size matches signature; canonical 1D pitched layout
  // (pitch == xsize == extent.width, ysize == 1, height == depth == 1) also matches.
  auto smaller_nodes = make_d2d(new_src.data(), new_dst.data(), 2048);
  const auto sig_smaller = run_sig(smaller_nodes);
  assert(sig_smaller == sig_donor);
  auto pitched_1d = make_d2d(new_src.data(), new_dst.data(), 4096);
  pitched_1d[1].memcpy.srcPtr.pitch = 4096; pitched_1d[1].memcpy.srcPtr.xsize = 4096; pitched_1d[1].memcpy.srcPtr.ysize = 1;
  pitched_1d[1].memcpy.dstPtr.pitch = 4096; pitched_1d[1].memcpy.dstPtr.xsize = 4096; pitched_1d[1].memcpy.dstPtr.ysize = 1;
  assert(run_sig(pitched_1d) == sig_donor);
  // CUDA 13.2 cudaMemcpy3DAsync omits xsize in CUDA_MEMCPY3D (returns xsize == 0 with pitch == width, ysize == 1).
  auto pitched_1d_zero_xsize = pitched_1d;
  pitched_1d_zero_xsize[1].memcpy.srcPtr.xsize = 0; pitched_1d_zero_xsize[1].memcpy.dstPtr.xsize = 0;
  assert(run_sig(pitched_1d_zero_xsize) == sig_donor);
  auto pitched_1d_smaller = make_d2d(new_src.data(), new_dst.data(), 2048);
  pitched_1d_smaller[1].memcpy.srcPtr.pitch = 2048; pitched_1d_smaller[1].memcpy.srcPtr.xsize = 2048; pitched_1d_smaller[1].memcpy.srcPtr.ysize = 1;
  pitched_1d_smaller[1].memcpy.dstPtr.pitch = 2048; pitched_1d_smaller[1].memcpy.dstPtr.xsize = 2048; pitched_1d_smaller[1].memcpy.dstPtr.ysize = 1;
  assert(run_sig(pitched_1d_smaller) == sig_donor);
  current_device = 1;
  ptr_attrs[new_src.data()] = {cudaMemoryTypeDevice, 1};
  ptr_attrs[new_dst.data()] = {cudaMemoryTypeDevice, 1};
  const auto sig_dev1 = run_sig(target_nodes);
  assert(!sig_dev1.empty() && sig_dev1 != sig_donor && sig_dev1[5] == 1);
  current_device = 0;
  ptr_attrs[new_src.data()] = {cudaMemoryTypeDevice, 0};
  ptr_attrs[new_dst.data()] = {cudaMemoryTypeDevice, 0};
  // Strided/pitched, mismatched xsize/ysize/height/depth, non-D2D, cross-device peer, 2D/3D, non-zero offset, zero-width, array and memset nodes stay non-transferable.
  auto pitched = make_d2d(new_src.data(), new_dst.data(), 4096);
  pitched[1].memcpy.srcPtr.pitch = 8192; pitched[1].memcpy.srcPtr.xsize = 4096; pitched[1].memcpy.srcPtr.ysize = 1;
  assert(run_sig(pitched).empty());
  auto bad_xsize = pitched_1d;
  bad_xsize[1].memcpy.srcPtr.xsize = 2048;
  assert(run_sig(bad_xsize).empty());
  auto bad_ysize = pitched_1d;
  bad_ysize[1].memcpy.dstPtr.ysize = 2;
  assert(run_sig(bad_ysize).empty());
  auto zero_ysize = pitched_1d;
  zero_ysize[1].memcpy.srcPtr.ysize = 0;
  assert(run_sig(zero_ysize).empty());
  auto zero_height = pitched_1d;
  zero_height[1].memcpy.extent.height = 0;
  assert(run_sig(zero_height).empty());
  auto zero_depth = pitched_1d;
  zero_depth[1].memcpy.extent.depth = 0;
  assert(run_sig(zero_depth).empty());
  auto h2d = make_d2d(host_buf.data(), new_dst.data(), 4096, cudaMemcpyHostToDevice);
  assert(run_sig(h2d).empty());
  auto d2h = make_d2d(new_src.data(), host_buf.data(), 4096, cudaMemcpyDeviceToHost);
  assert(run_sig(d2h).empty());
  auto default_host = make_d2d(host_buf.data(), new_dst.data(), 4096, cudaMemcpyDefault);
  assert(run_sig(default_host).empty());
  auto peer = make_d2d(peer_buf.data(), new_dst.data(), 4096, cudaMemcpyDeviceToDevice);
  assert(run_sig(peer).empty());
  auto copy2d = make_d2d(new_src.data(), new_dst.data(), 4096);
  copy2d[1].memcpy.extent.height = 2;
  assert(run_sig(copy2d).empty());
  auto offset = make_d2d(new_src.data(), new_dst.data(), 4096);
  offset[1].memcpy.srcPos.x = 16;
  assert(run_sig(offset).empty());
  auto zero = make_d2d(new_src.data(), new_dst.data(), 0);
  assert(run_sig(zero).empty());
  auto arr = make_d2d(new_src.data(), new_dst.data(), 4096);
  arr[1].memcpy.srcArray = reinterpret_cast<void*>(0x999);
  assert(run_sig(arr).empty());
  auto host_nodes = make_d2d(new_src.data(), new_dst.data(), 4096);
  host_nodes[1].kind = cudaGraphNodeTypeHost;
  assert(run_sig(host_nodes).empty());

  // 1D same-device memset nodes transfer across dst pointer, width and fill value while
  // keying elementSize and device; 2D, strided pitch, zero/invalid elementSize, host and peer stay rejected.
  auto make_memset = [&](void* dst, size_t width, unsigned int elem_size, unsigned int value = 0,
                         size_t height = 1, size_t pitch = 0) {
    FakeNode k1{cudaGraphNodeTypeKernel, reinterpret_cast<void*>(0x101), {}, {}};
    FakeNode m{cudaGraphNodeTypeMemset, nullptr, {}, {dst, pitch, value, elem_size, width, height}};
    FakeNode k2{cudaGraphNodeTypeKernel, reinterpret_cast<void*>(0x202), {}, {}};
    return std::array<FakeNode, 3>{k1, m, k2};
  };
  auto ms_donor = make_memset(old_dst.data(), 1024, 4, 0);
  auto ms_target = make_memset(new_dst.data(), 512, 4, 0x5a5a5a5aU);
  auto ms_pitched = make_memset(new_dst.data(), 512, 4, 0, 1, 512 * 4);
  const auto sig_ms = run_sig(ms_donor);
  assert(!sig_ms.empty());
  assert(sig_ms[3] == cudaGraphNodeTypeMemset && sig_ms[4] == 4 && sig_ms[5] == 0);
  assert(run_sig(ms_target) == sig_ms);
  assert(run_sig(ms_pitched) == sig_ms);
  auto ms_elem1 = make_memset(new_dst.data(), 4096, 1, 0);
  const auto sig_ms1 = run_sig(ms_elem1);
  assert(!sig_ms1.empty() && sig_ms1 != sig_ms && sig_ms1[4] == 1);
  auto ms_elem2 = make_memset(new_dst.data(), 2048, 2, 0);
  const auto sig_ms2 = run_sig(ms_elem2);
  assert(!sig_ms2.empty() && sig_ms2 != sig_ms && sig_ms2[4] == 2);
  current_device = 1;
  ptr_attrs[new_dst.data()] = {cudaMemoryTypeDevice, 1};
  const auto sig_ms_dev1 = run_sig(ms_target);
  assert(!sig_ms_dev1.empty() && sig_ms_dev1 != sig_ms && sig_ms_dev1[5] == 1);
  current_device = 0;
  ptr_attrs[new_dst.data()] = {cudaMemoryTypeDevice, 0};
  assert(run_sig(make_memset(nullptr, 1024, 4)).empty());
  assert(run_sig(make_memset(new_dst.data(), 0, 4)).empty());
  assert(run_sig(make_memset(new_dst.data(), 1024, 4, 0, 0)).empty());
  assert(run_sig(make_memset(new_dst.data(), 1024, 4, 0, 2)).empty());
  assert(run_sig(make_memset(new_dst.data(), 1024, 4, 0, 1, 8192)).empty());
  assert(run_sig(make_memset(new_dst.data(), 1024, 0)).empty());
  assert(run_sig(make_memset(new_dst.data(), 1024, 8)).empty());
  assert(run_sig(make_memset(host_buf.data(), 1024, 4)).empty());
  assert(run_sig(make_memset(peer_buf.data(), 1024, 4)).empty());
}
''')
    executable = tmp_path / "transferable_signature_d2d"
    built = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                            str(source), "-o", str(executable)],
                           capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(executable)], capture_output=True, text=True)
    assert ran.returncode == 0, ran.stderr


def test_cmake_rejects_wrapped_source_without_real_callbacks(tmp_path):
    cmake = shutil.which("cmake")
    if not cmake:
        pytest.skip("CMake unavailable")
    audit = (ROOT / "native/CMakeLists.txt").read_text().split(
        "# Follow local implementation includes too (the vision test includes a .cu).", 1)[1]
    (tmp_path / "wrapped.cc").write_text('#include "memory_failure.h"\nint main() {}\n')
    (tmp_path / "indirect.cc").write_text('#include "wrapped.cc"\n')
    project = "cmake_minimum_required(VERSION 3.24)\nproject(fixture CXX)\n"
    project += "add_executable(missing indirect.cc)\n" + audit
    (tmp_path / "CMakeLists.txt").write_text(project)
    result = subprocess.run([cmake, "-S", str(tmp_path), "-B", str(tmp_path / "build")],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "missing uses wrapped allocators but does not link" in result.stderr
    project = project.replace("add_executable(missing indirect.cc)",
        'add_library(cuteafd_memory_callbacks OBJECT "' + str(ROOT / "native/shared/src/memory_failure.cc") + '")\n'
        'target_include_directories(cuteafd_memory_callbacks PRIVATE "' + str(ROOT / "native/shared/include") + '")\n'
        'add_executable(missing indirect.cc)\n'
        'target_link_libraries(missing PRIVATE cuteafd_memory_callbacks)')
    (tmp_path / "CMakeLists.txt").write_text(project)
    result = subprocess.run([cmake, "-S", str(tmp_path), "-B", str(tmp_path / "linked")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

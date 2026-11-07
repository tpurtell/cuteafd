"""CPU checks for nvcc registration forwarding and workspace-free TP2 preload."""
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "native/shared/src/tp2_kernel_inventory.cc"
ABI = """#pragma once
#include <cstdint>
#define CUDARTAPI
struct uint3 { unsigned x, y, z; };
struct dim3 { unsigned x, y, z; };
struct cudaFuncAttributes {};
enum cudaError_t { cudaSuccess = 0, cudaErrorInvalidDeviceFunction = 98 };
extern "C" void __cudaRegisterFunction(void**, const char*, char*, const char*,
    int, uint3*, uint3*, dim3*, dim3*, int*);
extern "C" cudaError_t cudaFuncGetAttributes(cudaFuncAttributes*, const void*);
"""
CALLER = r'''
#include <cuteafd_cuda_registration_abi.h>
extern "C" void register_kernel(const char* host, const char* name) {
  __cudaRegisterFunction(nullptr, host, nullptr, name, -1, nullptr, nullptr,
      nullptr, nullptr, nullptr);
}
'''
MOCK = r'''
#include <cuteafd_cuda_registration_abi.h>
#include <cassert>
#include <vector>
extern "C" void register_kernel(const char*, const char*);
extern "C" int32_t cuteafd_tp2_native_initialize();
static int device = 0, calls = 0, fail_at = -1;
static std::vector<const void*> registered;
static std::vector<const void*> visited[2];
extern "C" void __cudaRegisterFunction(void** fat, const char* host, char* fun,
    const char* name, int limit, uint3* tid, uint3* bid, dim3* block,
    dim3* grid, int* warp) {
  assert(!fat && host && !fun && name && limit == -1);
  assert(!tid && !bid && !block && !grid && !warp);
  registered.push_back(host);
}
extern "C" cudaError_t cudaFuncGetAttributes(cudaFuncAttributes*, const void* host) {
  if (calls++ == fail_at) return static_cast<cudaError_t>(719);
  visited[device].push_back(host);
  return cudaSuccess;
}
int main(int argc, char**) {
  assert(cuteafd_tp2_native_initialize() == 98); // wrap absent/empty fails closed
  const char* names[] = {"select_fast_kernel", "softmax_select_kernel",
      "rtx_tp2_sum", "wait_flag_abortable", "brand_new_helper"};
  static char identities[5];
  for (int i = 0; i < 5; ++i) {
    register_kernel(&identities[i], names[i]);
    if (i < 3) assert(cuteafd_tp2_native_initialize() == 98);
  }
  if (argc > 1) {
    assert(cuteafd_tp2_native_initialize() == 98); // no --wrap is never accepted
    return 0;
  }
  register_kernel(&identities[0], names[0]);
  for (device = 0; device < 2; ++device) {
    visited[device].clear();
    assert(cuteafd_tp2_native_initialize() == 0);
    assert(visited[device].size() == 5); // duplicate forwarding, unique inventory
    for (int i = 0; i < 5; ++i) assert(visited[device][i] == &identities[i]);
  }
  device = 0; fail_at = calls + 2;
  assert(cuteafd_tp2_native_initialize() == 719);
  assert(registered.size() == 6);
}
'''


def test_compiled_registration_inventory_forwards_and_preloads_every_new_kernel(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    (tmp_path / "cuteafd_cuda_registration_abi.h").write_text(ABI)
    (tmp_path / "caller.cc").write_text(CALLER)
    (tmp_path / "mock.cc").write_text(MOCK)
    command = [compiler, "-std=c++17", "-pthread", "-I", str(tmp_path), str(SOURCE),
               str(tmp_path / "caller.cc"), str(tmp_path / "mock.cc")]
    for wrapped in [True, False]:
        executable = tmp_path / ("wrapped" if wrapped else "unwrapped")
        subprocess.run(command + (["-Wl,--wrap=__cudaRegisterFunction"] if wrapped else
                       ["-Wl,--defsym=__real___cudaRegisterFunction=__cudaRegisterFunction"])
                       + ["-o", str(executable)], check=True, capture_output=True)
        subprocess.run([str(executable)] + ([] if wrapped else ["no-wrap"]), check=True,
                       capture_output=True)
    (tmp_path / "cuteafd_cuda_registration_abi.h").write_text(
        ABI.replace("int, uint3*", "long, uint3*"))
    result = subprocess.run([compiler, "-std=c++17", "-I", str(tmp_path),
                             "-fsyntax-only", str(SOURCE)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "CUDA registration ABI changed" in result.stderr


def test_shared_bf16_add_native_wrapper_accepts_buffer_sized_widths(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    native = (ROOT / "native/shared/cuda/route_reduce.cu").read_text()
    overlap = native.split("bool overlaps(", 1)[1].split("// Deterministic", 1)[0]
    wrapper = native.split('extern "C" int32_t cuteafd_add_tp2_shared_async', 1)[1].split(
        'extern "C" int32_t cuteafd_reduce_tp2_experts_async', 1)[0]
    # Compile the real host validation and grid sizing, replacing only CUDA launch.
    wrapper, replacements = re.subn(
        r"add_tp2_shared<<<blocks, 256, 0, static_cast<cudaStream_t>\(stream\)>>>\(.*?count\);",
        "launch_add(blocks, count);", wrapper, flags=re.S)
    assert replacements == 1
    source = tmp_path / "shared-add.cc"
    source.write_text(r'''
#include <cassert>
#include <cstddef>
#include <cstdint>
static constexpr int cudaErrorInvalidValue = 1;
static unsigned launches = 0, launched_blocks = 0;
static size_t launched_count = 0;
static void launch_add(unsigned blocks, size_t count) {
  ++launches; launched_blocks = blocks; launched_count = count;
}
static int cudaGetLastError() { return 0; }
''' + "bool overlaps(" + overlap + '\nextern "C" int32_t cuteafd_add_tp2_shared_async' + wrapper + r'''
int main() {
  for (size_t hidden : {size_t(5120), size_t(7168)}) {
    for (size_t rows : {size_t(1), size_t(16), size_t(4096)}) {
      const size_t count = rows * hidden, bytes = count * 2;
      const auto a = reinterpret_cast<const uint16_t*>(uintptr_t(0x1000));
      const auto b = reinterpret_cast<const uint16_t*>(uintptr_t(0x1000 + bytes));
      auto output = reinterpret_cast<uint16_t*>(uintptr_t(0x1000 + 2 * bytes));
      assert(cuteafd_add_tp2_shared_async(a, b, output, count, nullptr) == 0);
      assert(launched_count == count);
      assert(launched_blocks == ((count + 255) / 256 < 4096 ? (count + 255) / 256 : 4096));
      const auto before = launches;
      assert(cuteafd_add_tp2_shared_async(a, b, output, 0, nullptr) == 1);
      assert(cuteafd_add_tp2_shared_async(a, b, output, SIZE_MAX / 2 + 1, nullptr) == 1);
      assert(cuteafd_add_tp2_shared_async(a, b, output, SIZE_MAX, nullptr) == 1);
      assert(cuteafd_add_tp2_shared_async(a, b, const_cast<uint16_t*>(a), count, nullptr) == 1);
      const auto wrapped = reinterpret_cast<const uint16_t*>(UINTPTR_MAX - 1);
      assert(cuteafd_add_tp2_shared_async(wrapped, b, output, count, nullptr) == 1);
      assert(launches == before);
    }
  }
}
''')
    # Initializer-list support is needed only by the standalone test harness.
    result = subprocess.run([compiler, "-std=c++17", "-include", "initializer_list", str(source),
                             "-o", str(tmp_path / "shared-add")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    subprocess.run([str(tmp_path / "shared-add")], check=True, capture_output=True)


def test_cuda_build_attests_real_header_and_always_wraps_registration():
    cmake = (ROOT / "native/CMakeLists.txt").read_text()
    assert 'crt/host_runtime.h" CUTEAFD_CUDA_HOST_RUNTIME' in cmake
    assert 'message(FATAL_ERROR "Cannot attest the CUDA registration ABI' in cmake
    assert 'list(APPEND CUTEAFD_NATIVE_SOURCES shared/src/tp2_kernel_inventory.cc)' in cmake
    assert 'target_link_options(cuteafd_native PRIVATE "-Wl,--wrap=__cudaRegisterFunction")' in cmake
    constructor = (ROOT / "rust/crates/cuteafd-daemon/src/shared/peer_split.rs").read_text()
    constructor = constructor.split("pub fn new(", 1)[1].split("pub fn new_abortable", 1)[0]
    assert constructor.index("library.tp2_native_initialize()?") < constructor.index("let zeroed")
    assert 'elapsed_ms = started.elapsed().as_secs_f64() * 1000.0' in constructor


def test_peer_watchdog_reports_counters_and_opt_in_exchange_identity():
    native = (ROOT / "native/shared/cuda/peer_exchange.cu").read_text()
    wait = native.split("__global__ void wait_flag_abortable", 1)[1].split(
        "__global__ void publish_abort", 1)[0]
    timeout = wait.split("global_ns() - start > kWaitTimeoutNs", 1)[1]
    for field in ["flag=%p", "state=%p", "aborted=%p", "expected=%u", "observed=%u",
                  "receive=%u", "abort=%u"]:
        assert field in timeout
    assert timeout.index("printf(") < timeout.index("__trap()")
    shared = (ROOT / "rust/crates/cuteafd-daemon/src/shared/peer_split.rs").read_text()
    identity = shared.split("pub fn trace_identity", 1)[1].split(
        "pub fn terminal_state", 1)[0]
    assert 'if self.ends[0].progress.is_none()' in identity
    assert 'var("CUTEAFD_PEER_WAIT_DIAGNOSTICS")' in shared
    for field in ["device =", "stream =", "flag_index =", "receive_state =", "abort ="]:
        assert field in identity
    assert "synchronize(" not in identity
    assert "copy_d2h" not in identity
    family = ROOT / "rust/crates/cuteafd-daemon/src/families/glm5_flash"
    engine = (family / "engine.rs").read_text()
    assert 'routes.trace_identity("glmf-tp2-routes")' in engine
    assert 'exchange.trace_identity("glmf-target-main")' in engine
    assert '"glmf-remote-dflash:taps,input,hidden,logits"' in (family / "mod.rs").read_text()
    assert "exchange.trace_identity(identity)" in (
        family.parent.parent / "shared/draft/remote.rs").read_text()


def test_peer_diagnostics_keep_graph_identity_and_dump_both_rank_frontiers(tmp_path):
    native = (ROOT / "native/shared/cuda/peer_exchange.cu").read_text()
    shared = (ROOT / "rust/crates/cuteafd-daemon/src/shared/peer_split.rs").read_text()
    ffi = (ROOT / "rust/crates/cuteafd-ffi/src/shared/peer_exchange.rs").read_text()
    assert "template<bool Diagnostic = false>" in native
    for kernel in ["push_signal", "push_planes", "wait_flag_abortable"]:
        assert f"reinterpret_cast<const void*>({kernel}<true>)" in native
        assert f"{kernel}<false><<<" in native
    dump = native.split("__device__ void diagnostic_dump", 1)[1].split(
        "__device__ void diagnostic_push_enter", 1)[0]
    for field in ["d.peer_control", "d.local_control", "d.peer", "d.local",
                  "push_issue=%llu", "wait_issue=%llu", "published=%u", "observed=%u"]:
        assert field in dump
    issue = shared.split("fn diagnostic(&self", 1)[1].split("pub fn terminal_state", 1)[0]
    assert "PEER_ISSUE.fetch_add" in issue
    assert 'seq = "device-runtime"' in issue
    assert '"peer host issue begin"' in issue
    assert '"peer host issue returned"' in issue
    assert "copy_d2h" not in issue and "synchronize(" not in issue
    for symbol in ["peer_push_signal_diagnostic", "peer_push_planes_diagnostic", "peer_wait_diagnostic"]:
        assert symbol in shared and symbol in ffi
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    source = tmp_path / "peer-abi.cc"
    source.write_text('''#include "cuteafd_peer_exchange.h"
#include <cstddef>
static_assert(sizeof(cuteafd_peer_progress_t) == 48);
static_assert(offsetof(cuteafd_peer_progress_t, published) == 24);
static_assert(offsetof(cuteafd_peer_diagnostic_t, issue) == 32);
static_assert(offsetof(cuteafd_peer_diagnostic_t, rank) == 48);
static_assert(sizeof(cuteafd_peer_diagnostic_t) == 64);
''')
    result = subprocess.run([compiler, "-std=c++17", "-I", str(ROOT / "native/shared/include"),
                             "-fsyntax-only", str(source)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_remote_context_wait_is_ordered_after_cooperative_target_producer():
    family = ROOT / "rust/crates/cuteafd-daemon/src/families/glm5_flash"
    remote = (family.parent.parent / "shared/draft/remote_dflash.rs").read_text()
    update = remote.split("pub fn update(&self", 1)[1].split("pub fn put_taps", 1)[0]
    assert "self.synchronize()?" not in update
    assert "cuda_stream_synchronize" not in update
    assert update.index("self.bridge.publish_context(&[plane])") < update.index("self.bridge.acknowledge_context()")
    # The shared bridge owns the ordering edge for every family binding.
    bridge = (ROOT / "rust/crates/cuteafd-daemon/src/shared/draft/remote.rs").read_text()
    publish = bridge.split("pub fn publish_context(&self", 1)[1].split("pub fn acknowledge_context", 1)[0]
    assert "cuda_stream_synchronize" not in publish
    event = publish.index("cuda_event_record(self.target_ready.raw, self.target_peer.stream)")
    wait = publish.index("cuda_stream_wait_event(self.stream.raw, self.target_ready.raw)")
    assert publish.index("self.exchange.wait(0, CONTEXT)?") < publish.index("self.exchange.push_planes(0, CONTEXT") < event < wait
    assert wait < publish.index("self.exchange.wait(1, CONTEXT)")
    assert "chunk-N+1 waits" in publish
    ack = bridge.split("pub fn acknowledge_context(&self", 1)[1].split("\n    }", 1)[0]
    assert "self.exchange.signal(1, CONTEXT)?" in ack
    assert "cuda_event_create_ordering" in bridge
    owner = bridge.split("pub(crate) struct RemoteBridge", 1)[1]
    assert owner.index("target_ready: Event<'a>") < owner.index("pub stream: Stream<'a>")
    startup = (family / "mod.rs").read_text()
    assert startup.index("engine.drafter = self.load_drafter") < startup.index("draft.validate_target_peer(")
    engine = (family / "engine.rs").read_text()
    assert "self.exchange()?.require_cooperative_producer_stream(rank, producer)?" in engine
    assert "routes.require_cooperative_producer_stream(rank, producer)?" in engine
    # Preserve intermediate context waves, and apply the same edge to decode.
    draft_dispatch = (family / "dspark.rs").read_text()
    assert "Self::RemoteDflash(d) => d.update(rows)" in draft_dispatch
    bench = (family / "speculate.rs").read_text().split(
        "draft_prefill_bench(args, &golden", 1)[1].split("fn draft_prefill_bench", 1)[0]
    assert bench.index("engine.prefill_device(") < bench.index("drafter.update(")
    assert "for chunk in prompt.chunks" in bench


def test_owner_pair_diagnostic_timing_waits_for_end_before_elapsed_query():
    source = (ROOT / "rust/crates/cuteafd-daemon/src/shared/experts/rtx/pair.rs").read_text()
    owner = source.split("impl<'a> OwnerPair<'a>", 1)[1]
    constructor = owner.split("pub(crate) fn new(", 1)[1].split(
        "pub(crate) fn uses_bf16_input", 1)[0]
    assert "timing: bool" in constructor
    assert re.search(r"timing:\s*timing\s*\.then\(\|\|", constructor)
    assert "CUTEAFD_TP2_TIMING" not in source
    capacity_check = constructor.index("weight.max_rows() == max_rows as usize")
    assert capacity_check < constructor.index("weights[rank].execution(modules[rank].take())?")
    launcher = (ROOT / "scripts/launch/run-family.sh").read_text()
    assert "CUTEAFD_TP2_TOKEN_SUMS" not in launcher
    execute = owner.split("pub(crate) async unsafe fn execute(", 1)[1].split(
        "#[cfg(test)]", 1)[0]
    diagnostics = execute.split("for rank in drain.ranks.iter() {", 1)[1].split(
        "drain.complete = true", 1)[0]
    assert "if let Some([start, end]) = &rank.timing {" in diagnostics
    scoped = diagnostics.split("rank.stream.device.run(|| unsafe {", 1)[1].split(
        "})?;", 1)[0]
    assert scoped.index("library.cuda_event_synchronize(end.raw)?;") < scoped.index(
        "library.cuda_event_elapsed_ms(start.raw, end.raw)")
    # Ordinary deferred submission must not gain a host timing wait.
    assert execute.count("cuda_event_synchronize(") == 1
    assert execute.count("cuda_event_elapsed_ms(") == 1
    assert "start.record(&rank.stream)?;" in execute
    assert "end.record(&rank.stream)?;" in execute


def test_compiled_bf16_blas_handles_follow_device_and_retry_failed_creation(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    source = (ROOT / "native/shared/cuda/linear.cu").read_text()
    helper = source.split("cuteafd_status_t cublas_handle(", 1)[1].split(
        "cuteafd_status_t validate_linear_args", 1)[0]
    mock = r'''
#include <cassert>
#include <unordered_map>
#include <string>
using cudaStream_t = void*;
using cudaStreamCaptureStatus = int;
constexpr int cudaStreamCaptureStatusNone = 0;
constexpr int CUTEAFD_STATUS_CAPTURE_REQUIRES_PREPARE = 9;
int cudaStreamIsCapturing(void*, int* out) { *out = 0; return 0; }
void cuteafd_set_last_error_message(const char*) {}
using cuteafd_status_t = int;
using cublasHandle_t = int*;
using cublasStatus_t = int;
using cudaError_t = int;
constexpr int CUTEAFD_STATUS_OK = 0, CUTEAFD_STATUS_INVALID_ARGUMENT = 1;
constexpr int cudaSuccess = 0, CUBLAS_STATUS_SUCCESS = 0;
static int device = 0, device_error = 0, create_error = 0, creates = 0, logs = 0;
static int owners[8];
int cudaGetDevice(int* out) { *out = device; return device_error; }
int cublasCreate(cublasHandle_t* out) {
  ++creates;
  if (create_error) return create_error;
  owners[creates] = device;
  *out = &owners[creates];
  return 0;
}
int status_from_cuda(int status) { return status; }
int status_from_cublas(int status) { return status; }
void log_blas_handle_created(const char*) { ++logs; }
'''
    main = r'''
int main() {
  cublasHandle_t first = nullptr, second = nullptr, again = nullptr;
  assert(cublas_handle(nullptr) == CUTEAFD_STATUS_INVALID_ARGUMENT);
  assert(creates == 0);
  assert(cublas_handle(&first) == 0 && *first == 0);
  device = 1; create_error = 7;
  assert(cublas_handle(&second) == 7 && second == nullptr);
  create_error = 0;
  assert(cublas_handle(&second) == 0 && *second == 1 && first != second);
  device = 0;
  assert(cublas_handle(&again) == 0 && again == first);
  device = 1;
  assert(cublas_handle(&again) == 0 && again == second);
  device_error = 19;
  assert(cublas_handle(&again) == 19 && again == second);
  assert(creates == 3 && logs == 2);
}
'''
    fixture = tmp_path / "blas_handles.cc"
    fixture.write_text(mock + "cuteafd_status_t cublas_handle(" + helper + main)
    executable = tmp_path / "blas_handles"
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    str(fixture), "-o", str(executable)], check=True, capture_output=True)
    subprocess.run([str(executable)], check=True, capture_output=True)


def test_compiled_graph_capture_requires_explicit_blas_prepare_without_changing_stream(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    linear = (ROOT / "native/shared/cuda/linear.cu").read_text()
    native = (ROOT / "native/shared/src/cuteafd_native.cc").read_text()
    helper = "cuteafd_status_t cublas_handle(" + linear.split(
        "cuteafd_status_t cublas_handle(", 1)[1].split(
        "cuteafd_status_t validate_linear_args", 1)[0]
    prepare = 'extern "C" cuteafd_status_t cuteafd_cuda_prepare_cublas_handle()' + linear.split(
        'extern "C" cuteafd_status_t cuteafd_cuda_prepare_cublas_handle()', 1)[1].split(
        'extern "C" cuteafd_status_t cuteafd_cuda_graph_update_linear_bf16_node', 1)[0]
    capture = 'extern "C" cuteafd_status_t cuteafd_cuda_graph_begin_capture(' + native.split(
        'extern "C" cuteafd_status_t cuteafd_cuda_graph_begin_capture(', 1)[1].split(
        'extern "C" cuteafd_status_t cuteafd_cuda_graph_end_capture(', 1)[0]
    # Exercise the real GEMM handle/stream binding, excluding only its CUDA bias kernel.
    launch = "cuteafd_status_t launch_linear_bf16_cublas(" + linear.split(
        "cuteafd_status_t launch_linear_bf16_cublas(", 1)[1].split(
        "  if (bias != nullptr) {", 1)[0] + "  (void)bias; return 0;\n}\n"
    mock = r'''
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <thread>
#include <unordered_map>
#include <string>
#define CUTEAFD_NATIVE_ENABLE_CUDA 1
using cudaStreamCaptureStatus = int;
constexpr int cudaStreamCaptureStatusNone = 0;
constexpr int CUTEAFD_STATUS_CAPTURE_REQUIRES_PREPARE = 9;
static std::string last_error;
void cuteafd_set_last_error_message(const char* message) { last_error = message; }
using cuteafd_status_t = int;
using cudaError_t = int;
using cublasStatus_t = int;
using cudaStream_t = void*;
struct Handle { int device; std::thread::id thread; void* stream; };
using cublasHandle_t = Handle*;
constexpr int CUTEAFD_STATUS_OK = 0, CUTEAFD_STATUS_INVALID_ARGUMENT = 1;
constexpr int CUTEAFD_STATUS_INTERNAL_ERROR = 2, cudaSuccess = 0;
constexpr int CUBLAS_STATUS_SUCCESS = 0, cudaStreamCaptureModeThreadLocal = 1;
constexpr int CUBLAS_OP_T = 0, CUBLAS_OP_N = 1, CUDA_R_16BF = 0;
constexpr int CUBLAS_COMPUTE_32F = 0, CUBLAS_GEMM_DEFAULT_TENSOR_OP = 0;
static thread_local int device = 0, device_error = 0;
static thread_local bool capturing = false;
static int creates = 0, begins = 0, bindings = 0, gemms = 0;
static int create_error = 0, capture_error = 0;
static Handle handles[16];
static cublasHandle_t last_gemm = nullptr;
int cudaGetDevice(int* out) { *out = device; return device_error; }
int cublasCreate(cublasHandle_t* out) {
  assert(!capturing);
  ++creates;
  if (create_error) return create_error;
  handles[creates] = {device, std::this_thread::get_id(), nullptr};
  *out = &handles[creates];
  return 0;
}
int cublasSetStream(cublasHandle_t handle, void* stream) {
  assert(handle->device == device && handle->thread == std::this_thread::get_id());
  ++bindings; handle->stream = stream; return 0;
}
template<class... Args> int cublasGemmEx(cublasHandle_t handle, Args...) {
  ++gemms; last_gemm = handle; return 0;
}
int cudaStreamBeginCapture(void* stream, int mode) {
  assert(stream && mode == cudaStreamCaptureModeThreadLocal && !capturing);
  ++begins;
  capturing = capture_error == 0;
  return capture_error;
}
int cudaStreamIsCapturing(void*, int* out) { *out = capturing ? 1 : 0; return 0; }
int status_from_cuda(int status) { return status; }
int status_from_cublas(int status) { return status; }
void log_blas_handle_created(const char*) {}
int fail(int status, const char*) { return status; }
int fail_cuda(int status, const char*, int) { return status; }
int ok() { return 0; }
int validate_linear_bf16_args(const uint16_t*, const uint16_t*, uint16_t*,
                              size_t, size_t, size_t) { return 0; }
'''
    main = r'''
int main() {
  void* capture_stream = reinterpret_cast<void*>(uintptr_t(0x10));
  void* eager_stream = reinterpret_cast<void*>(uintptr_t(0x20));
  uint16_t buffer = 0;
  auto gemm = [&](void* stream) {
    assert(launch_linear_bf16_cublas(&buffer, &buffer, nullptr, &buffer,
                                   1, 1, 1, stream) == 0);
    assert(last_gemm->stream == stream);
  };
  assert(cuteafd_cuda_graph_begin_capture(nullptr) == 1 && creates == 0);
  assert(cuteafd_cuda_graph_begin_capture(capture_stream) == 0 && creates == 0);
  assert(launch_linear_bf16_cublas(&buffer, &buffer, nullptr, &buffer,
                                1, 1, 1, capture_stream) == 9);
  assert(capturing && creates == 0 && bindings == 0 && gemms == 0);
  assert(last_error.find("cuteafd_cuda_linear_bf16_cublas_async") != std::string::npos);
  assert(last_error.find("prepare_cublas_handle before capture") != std::string::npos);
  capturing = false;
  device_error = 19;
  assert(cuteafd_cuda_prepare_cublas_handle() == 19 && creates == 0);
  device_error = 0; create_error = 7;
  assert(cuteafd_cuda_prepare_cublas_handle() == 7 && creates == 1);
  create_error = 0;
  assert(cuteafd_cuda_prepare_cublas_handle() == 0 && creates == 2);
  assert(cuteafd_cuda_graph_begin_capture(capture_stream) == 0);
  assert(creates == 2 && bindings == 0);
  gemm(capture_stream);
  const auto first = last_gemm;
  capturing = false;
  gemm(eager_stream);
  const int bound = bindings;
  assert(cuteafd_cuda_graph_begin_capture(capture_stream) == 0);
  assert(first->stream == eager_stream && bindings == bound && creates == 2);
  gemm(capture_stream);
  capturing = false;
  gemm(eager_stream);
  device = 1;
  assert(cuteafd_cuda_prepare_cublas_handle() == 0 && creates == 3);
  assert(cuteafd_cuda_graph_begin_capture(capture_stream) == 0 && creates == 3);
  gemm(capture_stream);
  assert(last_gemm != first && last_gemm->device == 1);
  capturing = false;
  device = 0;
  assert(cuteafd_cuda_graph_begin_capture(capture_stream) == 0 && creates == 3);
  capturing = false;
  std::thread worker([&] {
    assert(cuteafd_cuda_prepare_cublas_handle() == 0 && creates == 4);
    assert(cuteafd_cuda_graph_begin_capture(capture_stream) == 0 && creates == 4);
    gemm(capture_stream);
    assert(last_gemm != first && last_gemm->device == 0);
    capturing = false;
  });
  worker.join();
  capture_error = 71;
  assert(cuteafd_cuda_graph_begin_capture(capture_stream) == CUTEAFD_STATUS_INTERNAL_ERROR);
  assert(!capturing && creates == 4 && begins == 7 && gemms == 6);
  assert(first->stream == eager_stream);
}
'''
    fixture = tmp_path / "capture_blas.cc"
    fixture.write_text(mock + helper + prepare + capture + launch + main)
    executable = tmp_path / "capture_blas"
    built = subprocess.run([compiler, "-std=c++17", "-pthread", "-Wall", "-Wextra", "-Werror",
                            str(fixture), "-o", str(executable)], capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(executable)], capture_output=True, text=True)
    assert ran.returncode == 0, ran.stderr


def test_inventory_signature_matches_installed_toolkit_without_cuda_initialization(tmp_path):
    header = Path("/usr/local/cuda/include/crt/host_runtime.h")
    compiler = shutil.which("g++")
    if not header.exists() or not compiler:
        pytest.skip("CUDA headers or compiler unavailable")
    declaration = re.search(r"extern void CUDARTAPI __cudaRegisterFunction\([^;]*;", header.read_text())
    assert declaration, "toolkit registration ABI missing"
    (tmp_path / "cuteafd_cuda_registration_abi.h").write_text(
        '#include <cuda_runtime.h>\nextern "C" {\n' + declaration[0] + '\n}\n')
    subprocess.run([compiler, "-std=c++17", "-I", str(tmp_path),
                    "-I", str(header.parents[1]), "-fsyntax-only", str(SOURCE)],
                   check=True, capture_output=True)

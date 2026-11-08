"""CPU checks for nvcc registration forwarding and workspace-free TP2 preload."""
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
FAILURE_HEADER = '#include "' + str(ROOT / "native/shared/include/memory_failure.h") + '"\n'
CALLBACK_SOURCE = ROOT / "native/shared/src/memory_failure.cc"
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


BLAS_MOCK = r"""
#pragma once
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <thread>
using cudaStream_t = void*;
using cudaStreamCaptureStatus = int;
using cudaError_t = int;
using cuteafd_status_t = int;
using cublasStatus_t = int;
using cublasMath_t = int;
using cublasAtomicsMode_t = int;
struct MockBlas { int device; std::thread::id thread; void* stream; void* workspace;
                  int math = 0, atomics = 0; };
using cublasHandle_t = MockBlas*;
using cublasLtHandle_t = MockBlas*;
constexpr int cudaSuccess = 0, cudaStreamCaptureStatusNone = 0;
constexpr int CUBLAS_STATUS_SUCCESS = 0, CUBLAS_STATUS_NOT_INITIALIZED = 1;
constexpr int CUBLAS_STATUS_ALLOC_FAILED = 3, CUBLAS_STATUS_INVALID_VALUE = 7;
constexpr int CUBLAS_DEFAULT_MATH = 0, CUBLAS_ATOMICS_ALLOWED = 0;
constexpr int CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = 16;
constexpr int CUTEAFD_STATUS_OK = 0, CUTEAFD_STATUS_INVALID_ARGUMENT = 1;
constexpr int CUTEAFD_STATUS_INTERNAL_ERROR = 2, CUTEAFD_STATUS_CAPTURE_REQUIRES_PREPARE = 9;
constexpr int CUBLAS_OP_T = 0, CUBLAS_OP_N = 1, CUDA_R_16BF = 0;
constexpr int CUBLAS_COMPUTE_32F = 0, CUBLAS_GEMM_DEFAULT_TENSOR_OP = 0;
constexpr int cudaStreamCaptureModeThreadLocal = 1;
inline thread_local int device = 0, device_error = 0;
inline thread_local bool capturing = false;
inline int creates = 0, lt_creates = 0, destroys = 0, lt_destroys = 0;
inline int allocations = 0, frees = 0, bindings = 0, gemms = 0, begins = 0, drains = 0;
inline int create_error = 0, alloc_error = 0, capture_error = 0;
inline std::string last_error;
inline cublasHandle_t last_gemm = nullptr;
inline int cudaGetDevice(int* out) { *out = device; return device_error; }
inline int cudaSetDevice(int id) { device = id; return 0; }
inline int cudaDeviceSynchronize() { ++drains; return 0; }
inline int cudaStreamIsCapturing(void*, int* out) { *out = capturing ? 1 : 0; return 0; }
inline int cudaMalloc(void** out, size_t bytes) {
  assert(!capturing && bytes == (4 << 20));
  if (alloc_error) return alloc_error;
  ++allocations; *out = malloc(1); return *out ? 0 : 2;
}
inline int cudaFree(void* ptr) { ++frees; free(ptr); return 0; }
inline int cublasCreate(cublasHandle_t* out) {
  assert(!capturing); ++creates;
  if (create_error) return create_error;
  *out = new MockBlas{device, std::this_thread::get_id(), nullptr, nullptr}; return 0;
}
inline int cublasLtCreate(cublasLtHandle_t* out) {
  assert(!capturing); ++lt_creates;
  *out = new MockBlas{device, std::this_thread::get_id(), nullptr, nullptr}; return 0;
}
inline int cublasDestroy(cublasHandle_t h) { assert(device == h->device); ++destroys; delete h; return 0; }
inline int cublasLtDestroy(cublasLtHandle_t h) { assert(device == h->device); ++lt_destroys; delete h; return 0; }
inline int cublasSetStream(cublasHandle_t h, void* s) {
  assert(h->device == device && h->thread == std::this_thread::get_id());
  ++bindings; h->stream = s; h->workspace = nullptr; return 0;
}
inline int cublasSetWorkspace(cublasHandle_t h, void* w, size_t bytes) {
  assert(w && bytes == (4 << 20)); h->workspace = w; return 0;
}
inline int cublasSetMathMode(cublasHandle_t h, int mode) { h->math = mode; return 0; }
inline int cublasSetAtomicsMode(cublasHandle_t h, int mode) { h->atomics = mode; return 0; }
template<class... Args> int cublasGemmEx(cublasHandle_t h, Args...) {
  assert(h->workspace && h->math == CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION && h->atomics == 0);
  ++gemms; last_gemm = h; return 0;
}
inline int cudaStreamBeginCapture(void* s, int mode) {
  assert(s && mode == cudaStreamCaptureModeThreadLocal && !capturing);
  ++begins; capturing = capture_error == 0; return capture_error;
}
inline void cuteafd_set_last_error_message(const char* s) { last_error = s; }
inline int status_from_cuda(int s) { return s; }
inline int status_from_cublas(int s) { return s; }
inline int fail(int status, const char*) { return status; }
inline int fail_cuda(int status, const char*, int) { return status; }
inline int ok() { return 0; }
inline int validate_linear_bf16_args(const uint16_t*, const uint16_t*, uint16_t*, size_t, size_t, size_t) { return 0; }
"""


def _compile_blas_fixture(tmp_path, main, extra=""):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    (tmp_path / "mock_blas.h").write_text(BLAS_MOCK)
    for name in ("cuda_runtime.h", "cublas_v2.h", "cublasLt.h"):
        (tmp_path / name).write_text('#include "mock_blas.h"\n')
    linear = (ROOT / "native/shared/cuda/linear.cu").read_text()
    helper = "constexpr size_t kLinearBlasWorkspace" + linear.split(
        "constexpr size_t kLinearBlasWorkspace", 1)[1].split(
        "cuteafd_status_t validate_linear_args", 1)[0]
    prepare = 'extern "C" cuteafd_status_t cuteafd_cuda_prepare_cublas_stream(' + linear.split(
        'extern "C" cuteafd_status_t cuteafd_cuda_prepare_cublas_stream(', 1)[1].split(
        'extern "C" cuteafd_status_t cuteafd_cuda_graph_update_linear_bf16_node', 1)[0]
    native = (ROOT / "native/shared/src/cuteafd_native.cc").read_text()
    capture = 'extern "C" cuteafd_status_t cuteafd_cuda_graph_begin_capture(' + native.split(
        'extern "C" cuteafd_status_t cuteafd_cuda_graph_begin_capture(', 1)[1].split(
        'extern "C" cuteafd_status_t cuteafd_cuda_graph_end_capture(', 1)[0]
    launch = "cuteafd_status_t launch_linear_bf16_cublas(" + linear.split(
        "cuteafd_status_t launch_linear_bf16_cublas(", 1)[1].split(
        "  if (bias != nullptr) {", 1)[0] + "  (void)bias; return 0;\n}\n"
    fixture = tmp_path / "blas.cc"
    fixture.write_text('#define CUTEAFD_NATIVE_ENABLE_CUDA 1\n#include "blas_owner.h"\n' +
                       helper + prepare + capture + launch + main)
    sources = [str(fixture), str(CALLBACK_SOURCE)]
    if extra:
        other = tmp_path / "borrower.cc"
        other.write_text('#include "blas_owner.h"\n' + extra)
        sources.append(str(other))
    executable = tmp_path / "blas"
    built = subprocess.run([compiler, "-std=c++17", "-pthread", "-Wall", "-Wextra", "-Werror",
                            *sources, "-I", str(tmp_path), "-I", str(ROOT / "native/shared/include"),
                            "-o", str(executable)], capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(executable)], capture_output=True, text=True)
    assert ran.returncode == 0, ran.stderr
    return ran.stderr


def test_compiled_bf16_blas_handles_follow_device_and_retry_failed_creation(tmp_path):
    logs = _compile_blas_fixture(tmp_path, r"""
std::shared_ptr<cuteafd::BlasOwner> other_borrower(bool lt);
int main() {
  cublasHandle_t first = nullptr, second = nullptr, again = nullptr;
  std::shared_ptr<cuteafd::BlasOwner> invalid;
  assert(cuteafd::borrow_blas(false, invalid, "") == 7 && creates == 0);
  assert(cublas_handle(nullptr) == 1 && creates == 0);
  assert(cublas_handle(&first) == 0 && first->device == 0);
  uint32_t kind = 0; char name[128] = {}; size_t required = 0;
  assert(cuteafd_last_allocation_tag(&kind, name, sizeof(name), &required) == 0);
  assert(kind == CUTEAFD_MEMORY_KIND_WORKSPACE && std::string(name) == "shared/blas/linear");
  auto shared = other_borrower(false);
  assert(shared->gemm_owner == "shared/blas/linear");
  assert(shared->gemm == first && creates == 1);
  assert(other_borrower(true) == shared && lt_creates == 1);
  void* workspace = first->workspace;
  assert(cuteafd::bind_blas(shared, nullptr, workspace, 4 << 20, 5, 1) == 0);
  assert(first->math == 5 && first->atomics == 1);
  assert(cublas_handle(&again) == 0 && again == first);
  assert(first->math == CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION
      && first->atomics == 0 && first->workspace == workspace);
  device = 1; create_error = 7;
  assert(cublas_handle(&second) == 7 && second == nullptr);
  create_error = 0; alloc_error = 2;
  assert(cublas_handle(&second) == 2 && second == nullptr);
  alloc_error = 0;
  assert(cublas_handle(&second) == 0 && second->device == 1 && first != second);
  assert(creates == 3 && allocations == 2);
  device = 0;
  assert(cublas_handle(&again) == 0 && again == first);
  device_error = 19;
  assert(cublas_handle(&again) == 19 && again == first);
  device_error = 0;
  std::thread worker([&] {
    assert(cuteafd::bind_blas(shared, nullptr, workspace, 4 << 20) == 7);
    auto local = other_borrower(false);
    assert(local != shared && local->gemm != first);
  });
  worker.join();
  assert(creates == 4 && destroys == 1);
}
""", r"""
std::shared_ptr<cuteafd::BlasOwner> other_borrower(bool lt) {
  std::shared_ptr<cuteafd::BlasOwner> owner;
  assert(cuteafd::borrow_blas(lt, owner, "test/blas/serving") == 0);
  return owner;
}
""")
    assert logs.count("event=create backend=gemm") == 3
    assert logs.count("event=create backend=lt") == 1
    assert logs.count("event=destroy backend=gemm") == 3
    assert logs.count("event=destroy backend=lt") == 1
    assert logs.count("blas_workspace ") == 2
    assert "thread_id=" in logs and "thread_name=" in logs


def test_compiled_graph_capture_requires_explicit_blas_prepare_without_changing_stream(tmp_path):
    _compile_blas_fixture(tmp_path, r"""
int main() {
  void* capture_stream = reinterpret_cast<void*>(uintptr_t(0x10));
  void* eager_stream = reinterpret_cast<void*>(uintptr_t(0x20));
  uint16_t buffer = 0;
  auto gemm = [&](void* stream) {
    assert(launch_linear_bf16_cublas(&buffer, &buffer, nullptr, &buffer, 1, 1, 1, stream) == 0);
    assert(last_gemm->stream == stream);
  };
  assert(cuteafd_cuda_graph_begin_capture(nullptr) == 1 && creates == 0);
  assert(cuteafd_cuda_graph_begin_capture(capture_stream) == 0 && creates == 0);
  assert(launch_linear_bf16_cublas(&buffer, &buffer, nullptr, &buffer, 1, 1, 1, capture_stream) == 9);
  assert(capturing && creates == 0 && allocations == 0 && bindings == 0 && gemms == 0);
  assert(last_error.find("cuteafd_cuda_linear_bf16_cublas_async") != std::string::npos);
  assert(last_error.find("before capture") != std::string::npos);
  capturing = false;
  assert(cuteafd_cuda_prepare_cublas_stream(capture_stream) == 0);
  cublasHandle_t first = nullptr;
  gemm(capture_stream); first = last_gemm;
  auto captured_workspace = first->workspace;
  assert(cuteafd_cuda_prepare_cublas_stream(eager_stream) == 0);
  auto eager_workspace = first->workspace;
  assert(eager_workspace != captured_workspace && allocations == 2 && creates == 1);
  const auto bound = bindings;
  assert(cuteafd_cuda_graph_begin_capture(capture_stream) == 0);
  assert(bindings == bound && first->stream == eager_stream);
  gemm(capture_stream);
  assert(first->workspace == captured_workspace && allocations == 2 && creates == 1);
  capturing = false;
  gemm(eager_stream);
  assert(first->workspace == eager_workspace);
  device = 1;
  assert(cuteafd_cuda_prepare_cublas_stream(capture_stream) == 0 && creates == 2);
  assert(cuteafd_cuda_graph_begin_capture(capture_stream) == 0);
  gemm(capture_stream);
  assert(last_gemm != first && last_gemm->device == 1);
  capturing = false; device = 0;
  std::thread worker([&] {
    assert(cuteafd_cuda_prepare_cublas_stream(capture_stream) == 0 && creates == 3);
    assert(cuteafd_cuda_graph_begin_capture(capture_stream) == 0);
    gemm(capture_stream);
    assert(last_gemm != first && last_gemm->device == 0);
    capturing = false;
  });
  worker.join();
  assert(frees == 1 && destroys == 1 && drains == 1);
  capture_error = 71;
  assert(cuteafd_cuda_graph_begin_capture(capture_stream) == 2);
  assert(!capturing && creates == 3 && begins == 5 && gemms == 5);
  assert(first->stream == eager_stream);
}
""")


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


def test_every_family_measures_shared_native_preload_before_its_baseline():
    daemon = ROOT / "rust/crates/cuteafd-daemon/src"
    inventory = (daemon / "shared/inventory.rs").read_text()
    window = inventory.split("let (before, _) = library.cuda_memory_info()?;", 1)[1].split("let tracked", 1)[0]
    assert window.index("if phase == InventoryPhase::PreWeight { preload_native_modules(library)?; }") < window.index("|| library.cuda_memory_info()")
    assert "library.tp2_native_initialize()" in inventory.split("pub(crate) fn preload_native_modules", 1)[1]
    for family, paths in {
        "deepseek_v4": ["mod.rs"],
        "deepseek_v41": ["v41_native_serve.rs", "v41_native_serve/distributed.rs"],
        "glm5": ["mod.rs"], "glm5_flash": ["mod.rs"],
        "kimi_k3": ["startup.rs"], "muse_spark": ["serve.rs"],
        "qwen4": ["mod.rs", "dual.rs"], "step5": ["serve.rs"],
    }.items():
        for path in paths:
            source = (daemon / "families" / family / path).read_text()
            assert "RuntimeInventory::measure(" in source, (family, path)
            assert "InventoryPhase::PreWeight" in source, (family, path)
            assert "library.tp2_native_initialize()?" not in source, (family, path)
    for family, path, sample in [
        ("mimo_v2", "mod.rs", "let after = self.library.cuda_memory_info()"),
        ("qwen3_5", "engine.rs", "let (free, total) = library.cuda_memory_info()"),
    ]:
        source = (daemon / "families" / family / path).read_text()
        source = source.split("// Load only selected modules, then measure the admission baseline before tensors.", 1)[-1]
        assert source.index("crate::shared::inventory::preload_native_modules(") < source.index(sample)
    spark = (daemon / "shared/spark_waves.rs").read_text()
    guard = spark.split("fn native_initialize_before_waits", 1)[1].split("impl<'a> Waves", 1)[0]
    assert guard.index("ensure!(library.native_modules_preloaded()?") < guard.index("library.tp2_native_initialize()")
    assert "native_initialize_before_waits(library)?;" in spark.split("pub(crate) fn start", 1)[1]


def test_spark_wait_guard_callers_have_an_earlier_preweight_window():
    daemon = ROOT / "rust/crates/cuteafd-daemon/src"
    # Same-function ordering, with split-rank loops retained around inventory.
    for path, end in [
        ("families/deepseek_v4/mod.rs", "transports.push(SparkLink::"),
        ("families/deepseek_v41/v41_native_serve.rs", "transport.bootstrap("),
        ("families/deepseek_v41/v41_native_serve/distributed.rs", "transport.get_mut().bootstrap("),
        ("families/glm5/mod.rs", "SparkLink::prepared("),
        ("families/glm5_flash/mod.rs", "crate::shared::spark_intake::SparkLink::new_waves("),
        ("families/qwen4/mod.rs", "crate::shared::spark_intake::SparkLink::new_waves("),
        ("families/step5/serve.rs", "SparkLink::new_waves("),
        ("families/muse_spark/serve.rs", "super::engine::MuseEngine::connect("),
        ("families/kimi_k3/startup.rs", "Engine::load_admitted("),
    ]:
        source = (daemon / path).read_text()
        assert source.index("InventoryPhase::PreWeight") < source.index(end), path
    mimo = (daemon / "families/mimo_v2/mod.rs").read_text().split(
        "// Load only selected modules, then measure the admission baseline before tensors.", 1)[1]
    assert mimo.index("preload_native_modules(") < mimo.index("crate::shared::spark_intake::SparkLink::new(")
    # EXLR8 and ML4 execute the same GLM engine body; no alternate Spark startup.
    glm = (daemon / "families/glm5/mod.rs").read_text()
    assert "fn with_resolved_engine_media" in glm
    assert "CheckpointFormat::Exlr8" in glm and "ml4_draft" in glm
    serve = (daemon / "families/glm5/serve.rs").read_text()
    assert "opened.with_resolved_engine_media(" in serve
    # The ordinary Spark workers (all native/FP8/EXL3/EXLR8 roles) are servers,
    # not coordinator Wave consumers; they must never acquire this wait guard.
    worker = daemon / "shared/experts/service"
    for path in [daemon / "shared/experts/service.rs", *worker.rglob("*.rs")]:
        source = path.read_text()
        assert "SparkLink::" not in source and "Waves::start(" not in source, path
    # Qualification-only coordinators explicitly enter the shared preload too.
    for path in ["families/step5/golden.rs", "families/step5/chain.rs",
                 "shared/experts/tp_gate.rs", "shared/experts/exlr8_decode_gate.rs"]:
        assert "crate::shared::inventory::preload_native_modules(" in (daemon / path).read_text(), path


def test_step5_qualification_preloads_on_the_selected_device():
    source = (ROOT / "rust/crates/cuteafd-daemon/src/families/step5/chain.rs").read_text()
    # cache_test_device may select GPU1 while leaving GPU0 current.
    windows = source.split("let device = crate::shared::memory::device::cache_test_device(&library)?;")[1:]
    assert len(windows) == 2
    for window in windows:
        preload = window.split("let programs", 1)[0]
        assert "device.run(|| crate::shared::inventory::preload_native_modules(&library))?;" in preload

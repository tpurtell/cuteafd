"""CPU checks for the shared GPU-landing diagnostic and ordering boundary."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_landing_sentinel_checks_actual_loaded_bits_only_when_enabled(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    native = (ROOT / "native/shared/cuda/route_reduce.cu").read_text()
    helper = native.split("template<bool Diagnostic>\n", 1)[1].split(
        "template<int Ranks, bool Diagnostic = false>", 1)[0]
    source = tmp_path / "landing-value.cc"
    source.write_text(r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#define __device__
#define __forceinline__ inline
struct __nv_bfloat16 { uint16_t bits; };
static uint16_t __bfloat16_as_ushort(__nv_bfloat16 value) { return value.bits; }
static float __bfloat162float(__nv_bfloat16 value) { return float(value.bits); }
''' + "template<bool Diagnostic>\n" + helper + r'''
int main(int argc, char**) {
  for (uint16_t generation = 0; generation < 64; ++generation) {
    const uint16_t bits = 0x7f81 | generation;
    const __nv_bfloat16 plane[] = {{0x3f80}, {bits}, {0x4000}};
    assert(landing_value<false>(plane, 1, 3) == float(bits));
    if (argc > 1) assert(landing_value<true>(plane, 1, 3) == float(bits));
    assert(landing_value<true>(plane, 0, 3) == float(0x3f80));
    assert(landing_value<true>(plane, 2, 3) == float(0x4000));
  }
}
''')
    executable = tmp_path / "landing-value"
    result = subprocess.run([compiler, "-std=c++17", str(source), "-o", str(executable)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    quiet = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    assert quiet.stdout == ""
    detected = subprocess.run([str(executable), "diagnostic"], check=True,
                              capture_output=True, text=True)
    lines = detected.stdout.splitlines()
    assert len(lines) == 64
    assert all("landing sentinel: rank=3 plane=" in line and "offset=1 bits=" in line
               for line in lines)


def test_landing_poison_follows_prior_wave_drains_and_precedes_dispatch():
    shared = (ROOT / "rust/crates/cuteafd-daemon/src/shared/spark_intake.rs").read_text()
    before = shared.split("pub(crate) fn before_dispatch(&self)", 1)[1].split(
        "/// Whether a wave", 1)[0]
    poison = before.index("self.library.copy_h2d(destination")
    assert before.index("cuda_event_synchronize(self.consumed)") < poison
    assert before.index("cuda_event_synchronize(self.copied)") < poison
    assert before.index("self.mode == IntakeMode::Gpu && self.diagnostic_sentinel") < poison
    assert 'var("CUTEAFD_DIAG_LANDING_SENTINEL")' in shared
    native = (ROOT / "native/shared/cuda/route_reduce.cu").read_text()
    reducer = native.split("__global__ void reduce_compact_eight", 1)[1].split(
        '\nextern "C"', 1)[0]
    for rank in range(8):
        assert f"landing_value<Diagnostic>(p{rank}, offset, {rank})" in reducer
    assert "reduce_compact_eight<RANKS, true><<<" in native
    assert "reduce_compact_eight<RANKS, false><<<" in native


def test_no_relaxed_ordering_trial_only_changes_device_receive_registration(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    native = (ROOT / "native/shared/src/cuteafd_native.cc").read_text()
    helper = native.split("int device_receive_access() {", 1)[1].split("\n}", 1)[0]
    source = tmp_path / "device-receive-access.cc"
    source.write_text(r'''
#include <cassert>
#include <cstdlib>
#include <cstring>
constexpr int IBV_ACCESS_LOCAL_WRITE = 1;
constexpr int IBV_ACCESS_RELAXED_ORDERING = 1 << 20;
''' + "int device_receive_access() {" + helper + r'''
}
int main() {
  const char* key = "CUTEAFD_DIAG_LANDING_NO_RELAXED_ORDERING";
  assert(unsetenv(key) == 0);
  assert(device_receive_access() == (IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_RELAXED_ORDERING));
  for (const char* value : {"", "0", "true", "01"}) {
    assert(setenv(key, value, 1) == 0);
    assert(device_receive_access() == (IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_RELAXED_ORDERING));
  }
  assert(setenv(key, "1", 1) == 0);
  assert(device_receive_access() == IBV_ACCESS_LOCAL_WRITE);
}
''')
    executable = tmp_path / "device-receive-access"
    subprocess.run([compiler, "-std=c++17", "-include", "initializer_list", str(source),
                    "-o", str(executable)], check=True, capture_output=True, text=True)
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    attach = native.split('extern "C" cuteafd_status_t cuteafd_rdma_rc_endpoint_set_recv_landing', 1)[1].split(
        'extern "C"', 1)[0]
    assert "register_device_dmabuf(endpoint->pd, device_ptr, bytes, &mr,\n" in attach
    assert "device_receive_access());" in attach
    probe = native.split('extern "C" cuteafd_status_t cuteafd_rdma_gpu_landing_probe', 1)[1]
    assert "register_device_dmabuf(pd, gpu, bytes, &gpu_mr,\n" in probe
    assert "device_receive_access());" in probe
    assert native.count("device_receive_access()") == 3
    flag_write = native.split('extern "C" cuteafd_status_t cuteafd_rdma_rc_endpoint_expose_device', 1)[1].split(
        'extern "C"', 1)[0]
    assert "IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);" in flag_write
    assert "RELAXED_ORDERING" not in flag_write

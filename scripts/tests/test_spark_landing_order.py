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

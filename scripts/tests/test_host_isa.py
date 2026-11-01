"""Host CPU targets: scripts/build/host-isa.sh and its use by the build scripts.

The daemon-side selection (fake /proc/cpuinfo with and without AVX-512, the
override, the exec plan) is tested in rust/crates/cuteafd-ffi/src/host_isa.rs.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / 'scripts/build/host-isa.sh'


def run(body: str, arch: str = 'x86_64', cpuinfo: str | None = None, tmp_path: Path | None = None) -> subprocess.CompletedProcess:
    """Source the helper with uname/grep's view of the machine faked."""
    prelude = f'uname() {{ printf "%s\\n" {arch}; }}\n'
    if cpuinfo is not None:
        assert tmp_path is not None
        info = tmp_path / 'cpuinfo'
        info.write_text(cpuinfo)
        prelude += f'grep() {{ if [[ "${{@: -1}}" == /proc/cpuinfo ]]; then command grep "${{@:1:$#-1}}" {info}; else command grep "$@"; fi; }}\n'
    script = f'set -euo pipefail\n{prelude}source {HELPER}\n{body}'
    return subprocess.run(['bash', '-c', script], capture_output=True, text=True)


AVX512 = 'flags\t\t: fpu sse2 avx2 fma bmi2 avx512f avx512dq avx512cd avx512bw avx512vl\n'
AVX2 = 'flags\t\t: fpu sse2 avx2 fma bmi2\n'


def test_release_coordinator_builds_portable_base_then_avx512_variant():
    result = run('cuteafd_host_isa_list release ""')
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ['x86-64-v2', 'x86-64-v4']


@pytest.mark.parametrize('mode', ['release', 'wip'])
def test_spark_builds_one_gb10_target(mode):
    result = run(f'cuteafd_host_isa_list {mode} ""', arch='aarch64')
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ['grace']


@pytest.mark.parametrize(('cpuinfo', 'expected'), [(AVX512, 'x86-64-v4'), (AVX2, 'x86-64-v2')])
def test_wip_coordinator_builds_only_this_hosts_isa(tmp_path, cpuinfo, expected):
    result = run('cuteafd_host_isa_list wip ""', cpuinfo=cpuinfo, tmp_path=tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == [expected]


def test_override_lists_are_validated_against_the_build_architecture():
    both = run('cuteafd_host_isa_list wip "x86-64-v2,x86-64-v4"')
    assert both.stdout.split() == ['x86-64-v2', 'x86-64-v4']
    wrong = run('cuteafd_host_isa_list wip grace')
    assert wrong.returncode != 0 and 'cannot be built on x86_64' in wrong.stderr
    unknown = run('cuteafd_host_isa_list release znver5')
    assert unknown.returncode != 0


def test_rust_flags_and_triples():
    result = run('for isa in x86-64-v2 x86-64-v4 grace; do printf "%s|%s\\n" "$(cuteafd_host_isa_rustflags $isa)" "$(cuteafd_host_isa_triple $isa)"; done')
    assert result.returncode == 0, result.stderr
    v3, v4, grace = result.stdout.splitlines()
    assert v3 == '-C target-cpu=x86-64-v2|x86_64-unknown-linux-gnu'
    assert v4 == '-C target-cpu=x86-64-v4|x86_64-unknown-linux-gnu'
    # GB10: X925 tuning, the crypto it has, and no MTE (which it lacks).
    assert grace == '-C target-cpu=cortex-x925 -C target-feature=+aes,+sha2,+sha3,-mte|aarch64-unknown-linux-gnu'


def test_cargo_build_is_locked_per_isa_target_dir_with_explicit_target(tmp_path):
    shims = tmp_path / 'shims'
    shims.mkdir()
    log = tmp_path / 'cargo.log'
    (shims / 'cargo').write_text(f'#!/bin/sh\nprintf "%s|%s|%s\\n" "$CARGO_TARGET_DIR" "$RUSTFLAGS" "$*" >>{log}\n')
    (shims / 'cargo').chmod(0o755)
    result = run(f'PATH={shims}:$PATH; RUSTFLAGS=-Awarnings cuteafd_host_isa_cargo_build /t/cargo-target x86-64-v4 /src/Cargo.toml --quiet; '
                 'cuteafd_host_isa_daemon /t/cargo-target x86-64-v4')
    assert result.returncode == 0, result.stderr
    target, flags, args = log.read_text().strip().split('|')
    assert target == '/t/cargo-target-x86-64-v4'
    assert flags == '-Awarnings -C target-cpu=x86-64-v4'
    assert args.startswith('build --locked --manifest-path /src/Cargo.toml -p cuteafd-daemon --release --target x86_64-unknown-linux-gnu')
    assert args.endswith('--quiet')
    assert result.stdout == '/t/cargo-target-x86-64-v4/x86_64-unknown-linux-gnu/release/cuteafd'


def test_install_places_base_and_variants_and_always_creates_variant_dirs(tmp_path):
    target, native, output = tmp_path / 'ct', tmp_path / 'native', tmp_path / 'out'
    for isa in ('x86-64-v2', 'x86-64-v4'):
        daemon = tmp_path / f'ct-{isa}/x86_64-unknown-linux-gnu/release/cuteafd'
        daemon.parent.mkdir(parents=True)
        daemon.write_text(isa)
    (native / 'x86-64-v4').mkdir(parents=True)
    (native / 'libcuteafd_native.so').write_text('lib v3')
    (native / 'x86-64-v4/libcuteafd_native.so').write_text('lib v4')
    result = run(f'cuteafd_host_isa_install {output} {native} {target} x86-64-v2 x86-64-v4')
    assert result.returncode == 0, result.stderr
    assert (output / 'cuteafd').read_text() == 'x86-64-v2'
    assert (output / 'libcuteafd_native.so').read_text() == 'lib v3'
    assert (output / 'bin-variants/x86-64-v4/cuteafd').read_text() == 'x86-64-v4'
    assert (output / 'lib-variants/x86-64-v4/libcuteafd_native.so').read_text() == 'lib v4'
    single = tmp_path / 'single'
    result = run(f'cuteafd_host_isa_install {single} {native} {target} x86-64-v2')
    assert result.returncode == 0, result.stderr
    assert (single / 'bin-variants').is_dir() and not any((single / 'bin-variants').iterdir())
    assert (single / 'lib-variants').is_dir() and not any((single / 'lib-variants').iterdir())


def test_build_scripts_pass_the_isa_list_to_cargo_and_cmake():
    for name in ('build-release-artifacts.sh', 'build-wip-artifacts.sh'):
        text = (ROOT / 'scripts/build' / name).read_text()
        assert 'source "$(dirname "${BASH_SOURCE[0]}")/host-isa.sh"' in text, name
        assert '-DCUTEAFD_HOST_ISA="${host_isa_cmake:-}"' in text, name
        assert 'cuteafd_host_isa_cargo_build "$' in text, name
    release = (ROOT / 'scripts/build/build-release-artifacts.sh').read_text()
    assert 'cuteafd_host_isa_list release "${CUTEAFD_RELEASE_HOST_ISA:-}"' in release
    wip = (ROOT / 'scripts/build/build-wip-artifacts.sh').read_text()
    assert 'cuteafd_host_isa_list wip "${CUTEAFD_WIP_HOST_ISA:-}"' in wip
    # The resolved list reaches the native-store key (every CUTEAFD_WIP_* variable is hashed).
    assert 'export CUTEAFD_WIP_HOST_ISA_RESOLVED=' in wip
    assert 'lib-variants' in wip.split('native_items=(')[1].split(')')[0]


def test_wip_build_key_and_staging_carry_the_variants():
    wip = (ROOT / 'wip.sh').read_text()
    key = wip[wip.index('wip_build_key="$('):wip.index('} | sha256sum', wip.index('wip_build_key="$('))]
    assert 'host-isa coordinator' in key
    for path in ('scripts/lib/release-common.sh', 'scripts/launch/run-family.sh'):
        text = (ROOT / path).read_text()
        assert '/bin-variants/." ' in text and '/lib-variants/." ' in text, path
    finalize = (ROOT / 'scripts/build/finalize-wip-slot.sh').read_text()
    assert 'for variants in bin-variants lib-variants' in finalize
    dockerfile = (ROOT / 'docker/Dockerfile.release').read_text()
    assert 'COPY .cuteafd-release-image/bin-variants/ /opt/cuteafd/bin/' in dockerfile
    assert 'COPY .cuteafd-release-image/lib-variants/ /opt/cuteafd/lib/' in dockerfile


def test_cmake_knows_each_isa_and_keeps_fp_contraction_off():
    text = (ROOT / 'native/CMakeLists.txt').read_text()
    assert 'set(CUTEAFD_HOST_ISA_FLAGS_x86-64-v2 -march=x86-64-v2' in text
    assert 'set(CUTEAFD_HOST_ISA_FLAGS_x86-64-v4 -march=x86-64-v4' in text
    assert 'set(CUTEAFD_HOST_ISA_FLAGS_grace -mcpu=grace)' in text
    assert '-ffp-contract=off' in text
    assert 'cmake_language(DEFER CALL cuteafd_host_isa_variants)' in text


def test_wip_cargo_features_are_allowlisted_keyed_and_reach_cargo():
    wip = (ROOT / 'wip.sh').read_text()
    key = wip[wip.index('wip_build_key="$('):wip.index('} | sha256sum', wip.index('wip_build_key="$('))]
    assert 'CARGO_FEATURES; do' in key
    assert 'expected jemalloc or dev-* features' in wip
    assert '-e "CUTEAFD_WIP_CARGO_FEATURES=${wip_cargo_features:-}"' in wip
    build = (ROOT / 'scripts/build/build-wip-artifacts.sh').read_text()
    assert '--quiet "${cargo_feature_args[@]}"' in build
    # The Rust phase marker changes with the feature set; the native key ignores it.
    assert 'features=%s' in build.split('wip_rust_fingerprint=')[1].split('\n')[0]
    assert "'^CUTEAFD_WIP_(CARGO_FEATURES|" in build
    start = wip.index('wip_cargo_features="')
    check = 'release_die() { exit 3; }\n' + wip[start:wip.index('\nwip_build_key=', start)]
    for features, expected in [('dev-test,dev-x', 0), ('jemalloc', 0),
                               ('jemalloc,dev-test', 0), ('prod', 3),
                               ('dev-test default', 3), ('dev_x', 3)]:
        result = subprocess.run(['bash', '-c', check],
                                env={**os.environ, 'CUTEAFD_WIP_CARGO_FEATURES': features})
        assert result.returncode == expected, features


def test_performance_cores_detects_runtime_big_core_frequency(tmp_path):
    for cpu in range(20):
        path = tmp_path / f'cpu{cpu}/cpufreq'
        path.mkdir(parents=True)
        (path / 'cpuinfo_max_freq').write_text('3900000' if cpu % 2 else '2808000')
    result = run(f'cuteafd_performance_cores {tmp_path}', arch='aarch64')
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '10'


def test_build_parallelism_only_changes_gb10():
    for arch, expected in [('aarch64', '10 10'), ('x86_64', '16 16')]:
        result = run('CMAKE_BUILD_PARALLEL_LEVEL=16; CARGO_BUILD_JOBS=16; '
                     'cuteafd_performance_cores() { echo 10; }; '
                     'cuteafd_build_parallelism; '
                     'echo "$CMAKE_BUILD_PARALLEL_LEVEL $CARGO_BUILD_JOBS"', arch=arch)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == expected

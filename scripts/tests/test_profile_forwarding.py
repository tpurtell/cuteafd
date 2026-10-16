"""PROFILE=... forwards one set of profiling switches to the coordinator and every
Spark worker in every family (run-family.sh) and on the V4.1 path (run.sh)."""
import shutil
import socket
import subprocess
from pathlib import Path

import pytest

from test_run_family_dispatch import GLMF_TWO_LAYER, _family_launch_result

ROOT = Path(__file__).resolve().parents[2]
HOST = socket.gethostname().split('.')[0]


def _helper(*args):
    script = 'source scripts/lib/release-common.sh; release_profile_docker_args "$@"; printf "%s\\n" ${RELEASE_PROFILE_ARGS[@]+"${RELEASE_PROFILE_ARGS[@]}"}'
    return subprocess.run(['bash', '-c', script, 'test', *args], cwd=ROOT, capture_output=True, text=True)


def _wrapper(*args):
    # NUL-separated: the container script is one argv element and contains newlines.
    script = ('source scripts/lib/release-common.sh; release_profile_wrapper "$@"; '
              'printf "W:%s\\0" ${RELEASE_PROFILE_WRAPPER[@]+"${RELEASE_PROFILE_WRAPPER[@]}"}; '
              'printf "D:%s\\0" ${RELEASE_PROFILE_WRAPPER_DOCKER_ARGS[@]+"${RELEASE_PROFILE_WRAPPER_DOCKER_ARGS[@]}"}')
    out = subprocess.run(['bash', '-c', script, 'test', *args], cwd=ROOT, capture_output=True, text=True)

    def take(prefix):
        return [item[len(prefix):] for item in out.stdout.split('\0')
                if item.startswith(prefix) and len(item) > len(prefix)]

    return out, take('W:'), take('D:')


def _install(root: Path, relative: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/bin/sh\nprintf "%s\\n" "$0" "$@"\n')
    path.chmod(0o755)
    return path


def _resolve(tool, root: Path):
    script = 'source scripts/lib/release-common.sh; release_profile_resolve_tool "$1" "$2"'
    return subprocess.run(['bash', '-c', script, 'test', tool, str(root)],
                          cwd=ROOT, capture_output=True, text=True)


def test_helper_maps_switches_host_and_directory():
    out = _helper('census,transport-timing,expert-timing,draft-timing', '/data/prof', 'kiwi')
    assert out.returncode == 0, out.stderr
    args = out.stdout.split('\n')
    for flag in ('CUTEAFD_GRAPH_CENSUS=1', 'CUTEAFD_PROTOCOL_V2_TCP_TIMING=1', 'CUTEAFD_EXPERT_TIMING=1',
                 'CUTEAFD_DRAFT_TIMING=1', 'CUTEAFD_PROFILE_HOST=kiwi', 'CUTEAFD_PROFILE_DIR=/data/prof', '/data/prof:/data/prof'):
        assert flag in args
    assert _helper('', '', 'kiwi').stdout == '\n'


def test_existing_switches_are_unchanged_and_add_no_wrapper():
    out = _helper('census,transport-timing,expert-timing', '/data/prof', 'kiwi')
    assert out.stdout.split('\n')[:-1] == [
        '-e', 'CUTEAFD_GRAPH_CENSUS=1', '-e', 'RUST_LOG=info,cuteafd::graph_capture=debug',
        '-e', 'CUTEAFD_STARTUP_MEMORY_PROBE=1', '-e', 'CUTEAFD_PROTOCOL_V2_TCP_TIMING=1', '-e', 'CUTEAFD_EXPERT_TIMING=1',
        '-e', 'CUTEAFD_PROFILE_HOST=kiwi', '-v', '/data/prof:/data/prof', '-e', 'CUTEAFD_PROFILE_DIR=/data/prof']
    for profile in ('', 'census,transport-timing,expert-timing,draft-timing'):
        out, wrapper, docker = _wrapper(profile, '/data/prof', 'raptor', 'coordinator')
        assert out.returncode == 0, out.stderr
        assert wrapper == [] and docker == []
        out, wrapper, docker = _wrapper(profile, '/data/prof', 'kiwi', 'spark')
        assert out.returncode == 0 and wrapper == [] and docker == []


def test_census_carries_the_startup_memory_probe_and_nothing_else_does():
    out = _helper('census', '/data/prof', 'kiwi')
    assert out.returncode == 0, out.stderr
    assert out.stdout.split('\n')[:-1] == [
        '-e', 'CUTEAFD_GRAPH_CENSUS=1', '-e', 'RUST_LOG=info,cuteafd::graph_capture=debug',
        '-e', 'CUTEAFD_STARTUP_MEMORY_PROBE=1',
        '-e', 'CUTEAFD_PROFILE_HOST=kiwi', '-v', '/data/prof:/data/prof', '-e', 'CUTEAFD_PROFILE_DIR=/data/prof']
    for profile in ('transport-timing,expert-timing,draft-timing', 'nsys', 'ncu'):
        out = _helper(profile, '/data/prof', 'kiwi')
        assert out.returncode == 0, out.stderr
        assert 'CUTEAFD_STARTUP_MEMORY_PROBE' not in out.stdout
    assert _helper('', '', 'kiwi').stdout == '\n'


def test_family_launch_census_probe_reaches_coordinator_and_workers_only_when_asked(tmp_path):
    prof = tmp_path / 'prof'
    config = 'GLM5_FLASH_FP8_MODEL_ID=off\n'
    (tmp_path / 'plain').mkdir()
    (tmp_path / 'census').mkdir()
    plain = _family_launch_result(tmp_path / 'plain', GLMF_TWO_LAYER, 'test/glmf', config)
    assert plain.returncode == 0, plain.stderr
    assert 'CUTEAFD_STARTUP_MEMORY_PROBE' not in plain.stderr
    assert 'CUTEAFD_PROFILE' not in plain.stderr
    census = _family_launch_result(tmp_path / 'census', GLMF_TWO_LAYER, 'test/glmf',
                                   config + f'PROFILE=census\nPROFILE_DIR={prof}\n')
    assert census.returncode == 0, census.stderr
    lines = census.stderr.splitlines()
    coordinator = next(line for line in lines if 'cuteafd serve-glmf' in line)
    workers = [line for line in lines if 'cuteafd expertd-native' in line]
    assert workers
    for line in [coordinator, *workers]:
        for flag in ('CUTEAFD_GRAPH_CENSUS=1', 'CUTEAFD_STARTUP_MEMORY_PROBE=1', f'CUTEAFD_PROFILE_DIR={prof}'):
            assert flag in line, (flag, line)


CAPTURE_PATH_CASES = r'''
#include "graph_census_path.h"
#include <cstdio>
int main() {
  const char* cases[][3] = {
    {nullptr, "raptor", "/tmp/cuteafd-graph-census-captures.jsonl"},
    {"", "raptor", "/tmp/cuteafd-graph-census-captures.jsonl"},
    {"relative/dir", "raptor", "/tmp/cuteafd-graph-census-captures.jsonl"},
    {"/data/prof", "raptor", "/data/prof/graph-census-captures-raptor.jsonl"},
    {"/data/prof/", "kiwi", "/data/prof/graph-census-captures-kiwi.jsonl"},
    {"/data/prof", "a.b/../c", "/data/prof/graph-census-captures-a_b____c.jsonl"},
    {"/data/prof", nullptr, "/data/prof/graph-census-captures-unknown.jsonl"},
    {"/data/prof", "", "/data/prof/graph-census-captures-unknown.jsonl"},
  };
  int failed = 0;
  for (auto& c : cases) {
    const std::string got = cuteafd_graph_census_captures_path(c[0], c[1]);
    if (got != c[2]) { std::printf("%s -> %s\n", c[2], got.c_str()); failed = 1; }
  }
  return failed;
}
'''


def test_graph_census_captures_follow_the_profile_dir(tmp_path):
    compiler = shutil.which('g++')
    if not compiler:
        pytest.skip('C++ compiler unavailable')
    (tmp_path / 'cases.cc').write_text(CAPTURE_PATH_CASES)
    executable = tmp_path / 'cases'
    subprocess.run([compiler, '-std=c++17', '-I', str(ROOT / 'native/shared/src'), str(tmp_path / 'cases.cc'),
                    '-o', str(executable)], check=True, capture_output=True)
    result = subprocess.run([str(executable)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout
    native = (ROOT / 'native/shared/src/cuteafd_native.cc').read_text()
    assert 'cuteafd_graph_census_captures_path(std::getenv("CUTEAFD_PROFILE_DIR"), host)' in native
    assert 'std::fopen("/tmp/cuteafd-graph-census-captures.jsonl"' not in native


def test_nsys_wraps_the_coordinator_paused_into_profile_dir():
    out = _helper('nsys,draft-timing', '/data/prof', 'raptor')
    assert out.returncode == 0, out.stderr
    assert 'CUTEAFD_DRAFT_TIMING=1' in out.stdout and 'nsys' not in out.stdout
    out, wrapper, docker = _wrapper('nsys,draft-timing', '/data/prof', 'raptor', 'coordinator')
    assert out.returncode == 0, out.stderr
    assert wrapper[:2] == ['/bin/bash', '-c']
    assert wrapper[3:6] == ['release-profile-resolve', 'nsys', 'profile']
    assert 'release_profile_resolve_tool "$tool"' in wrapper[2]
    for flag in ('--trace=cuda,nvtx,osrt', '--cuda-graph-trace=node', '--sample=none', '--cpuctxsw=none',
                 '--start-later=true', '--session-new=cuteafd-profile', '--output=/data/prof/nsys-raptor'):
        assert flag in wrapper
    assert docker == ['--stop-signal', 'SIGINT', '--stop-timeout', '300', '--label', 'cuteafd.graceful-stop=1']


def test_nsys_gpu_metrics_samples_the_named_devices():
    out, wrapper, _ = _wrapper('nsys', '/data/prof', 'raptor', 'coordinator', '', '', '', '0,1')
    assert out.returncode == 0, out.stderr
    assert '--gpu-metrics-devices=0,1' in wrapper
    out, wrapper, _ = _wrapper('nsys', '/data/prof', 'raptor', 'coordinator', '', '', '', 'all')
    assert '--gpu-metrics-devices=all' in wrapper
    out, wrapper, _ = _wrapper('nsys', '/data/prof', 'raptor', 'coordinator')
    assert not [flag for flag in wrapper if flag.startswith('--gpu-metrics')]


@pytest.mark.parametrize('args,message', [
    (('nsys', '/data/prof', 'raptor', 'coordinator', '', '', '', 'all;rm'), 'GPU index list'),
    (('ncu', '/data/prof', 'raptor', 'coordinator', 'gemm', '', '', '0'), 'needs PROFILE=nsys'),
    (('', '', 'raptor', 'coordinator', '', '', '', '0'), 'needs PROFILE=nsys'),
])
def test_nsys_gpu_metrics_fails_closed(args, message):
    out, _, _ = _wrapper(*args)
    assert out.returncode != 0 and message in out.stderr


def test_ncu_wraps_with_filter_count_and_set():
    out, wrapper, _ = _wrapper('ncu', '/data/prof', 'raptor', 'coordinator', 'gemm|moe', '', '')
    assert out.returncode == 0, out.stderr
    assert wrapper[:2] == ['/bin/bash', '-c']
    assert wrapper[3:5] == ['release-profile-resolve', 'ncu']
    assert wrapper[0] != '/usr/local/bin/ncu'
    joined = ' '.join(wrapper[4:])
    for part in ('--kernel-name regex:gemm|moe', '--launch-count 20', '--set full', '--export /data/prof/ncu-raptor'):
        assert part in joined
    out, wrapper, _ = _wrapper('ncu', '/data/prof', 'raptor', 'coordinator', 'gemm', '5', 'basic')
    assert '--launch-count 5 --set basic' in ' '.join(wrapper[4:])


@pytest.mark.parametrize('args,message', [
    (('nsys', '/data/prof', 'kiwi', 'spark'), 'wraps the coordinator only'),
    (('ncu', '/data/prof', 'kiwi', 'spark', 'gemm'), 'wraps the coordinator only'),
    (('ncu', '/data/prof', 'raptor', 'coordinator'), 'PROFILE=ncu needs PROFILE_NCU_KERNEL'),
    (('ncu', '/data/prof', 'raptor', 'coordinator', 'a b'), 'must not contain whitespace'),
    (('ncu', '/data/prof', 'raptor', 'coordinator', 'gemm', '0'), 'PROFILE_NCU_COUNT must be'),
    (('ncu', '/data/prof', 'raptor', 'coordinator', 'gemm', '5', 'Full;x'), 'PROFILE_NCU_SET must be'),
    (('nsys', '/data/prof', 'raptor', 'coordinator', 'gemm'), 'need PROFILE=ncu'),
    (('census', '/data/prof', 'raptor', 'coordinator', 'gemm'), 'need PROFILE=ncu'),
])
def test_wrapper_refuses_bad_values(args, message):
    out, wrapper, _ = _wrapper(*args)
    assert out.returncode != 0 and message in out.stderr, out.stderr


def test_resolve_tool_uses_the_stated_layouts(tmp_path):
    out, ncu_wrapper, _ = _wrapper('ncu', '/data/prof', 'raptor', 'coordinator', 'gemm')
    assert out.returncode == 0, out.stderr
    out, nsys_wrapper, _ = _wrapper('nsys', '/data/prof', 'raptor', 'coordinator')
    assert out.returncode == 0, out.stderr
    for wrapper in (ncu_wrapper, nsys_wrapper):
        assert 'release_profile_resolve_tool "$tool"' in wrapper[2]
        assert wrapper[2].rstrip().endswith('release_profile_exec "$@"')

    stable = tmp_path / 'stable'
    _install(stable, 'usr/local/bin/ncu')
    _install(stable, 'opt/nvidia/nsight-compute/2026.9.9/ncu')
    found = _resolve('ncu', stable)
    assert found.returncode == 0, found.stderr
    assert found.stdout.strip() == str(stable / 'usr/local/bin/ncu')

    versioned = tmp_path / 'versioned'
    _install(versioned, 'opt/nvidia/nsight-compute/2026.1.9/ncu')
    _install(versioned, 'opt/nvidia/nsight-compute/2026.1.10/ncu')
    _install(versioned, 'opt/nvidia/nsight-compute/2025.4/ncu')
    found = _resolve('ncu', versioned)
    assert found.returncode == 0, found.stderr
    assert found.stdout.strip() == str(versioned / 'opt/nvidia/nsight-compute/2026.1.10/ncu')

    missing = _resolve('ncu', tmp_path / 'empty')
    assert missing.returncode == 127
    assert missing.stdout == ''
    assert 'ncu not found: looked for /usr/local/bin/ncu and /opt/nvidia/nsight-compute/*/ncu' in missing.stderr

    nsys_bin = tmp_path / 'nsys-bin'
    _install(nsys_bin, 'usr/local/bin/nsys')
    _install(nsys_bin, 'usr/local/cuda/bin/nsys')
    _install(nsys_bin, 'opt/nvidia/nsight-systems/2026.9/nsys')
    found = _resolve('nsys', nsys_bin)
    assert found.returncode == 0, found.stderr
    assert found.stdout.strip() == str(nsys_bin / 'usr/local/bin/nsys')

    nsys_cuda = tmp_path / 'nsys-cuda'
    _install(nsys_cuda, 'usr/local/cuda/bin/nsys')
    _install(nsys_cuda, 'opt/nvidia/nsight-systems/2026.9/nsys')
    found = _resolve('nsys', nsys_cuda)
    assert found.returncode == 0, found.stderr
    assert found.stdout.strip() == str(nsys_cuda / 'usr/local/cuda/bin/nsys')

    nsys_ver = tmp_path / 'nsys-ver'
    _install(nsys_ver, 'opt/nvidia/nsight-systems/2026.1.9/nsys')
    _install(nsys_ver, 'opt/nvidia/nsight-systems/2026.1.10/nsys')
    found = _resolve('nsys', nsys_ver)
    assert found.returncode == 0, found.stderr
    assert found.stdout.strip() == str(nsys_ver / 'opt/nvidia/nsight-systems/2026.1.10/nsys')

    missing = _resolve('nsys', tmp_path / 'no-nsys')
    assert missing.returncode == 127
    assert missing.stdout == ''
    assert 'nsys not found: looked for /usr/local/bin/nsys, /usr/local/cuda/bin/nsys, and /opt/nvidia/nsight-systems/*/nsys' in missing.stderr


def test_profile_exec_replaces_the_shell_with_the_resolved_binary(tmp_path):
    binary = _install(tmp_path, 'fake-ncu')
    script = '''
source scripts/lib/release-common.sh
fake=$1
shift
release_profile_resolve_tool() { printf "%s\\n" "$fake"; }
release_profile_exec "$@"
'''
    result = subprocess.run(['bash', '-c', script, 'test', str(binary), 'ncu', '--set', 'full', 'cuteafd', 'serve'],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [str(binary), '--set', 'full', 'cuteafd', 'serve']
    missing = '''
source scripts/lib/release-common.sh
release_profile_resolve_tool() { printf "ncu not found: looked for /usr/local/bin/ncu\\n" >&2; return 127; }
release_profile_exec "$@"
'''
    result = subprocess.run(['bash', '-c', missing, 'test', 'ncu', 'cuteafd'],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 127
    assert result.stdout == ''
    assert 'ncu not found: looked for /usr/local/bin/ncu' in result.stderr


@pytest.mark.parametrize('profile,directory,message', [
    ('census,flamegraph', '', 'unknown PROFILE switch flamegraph'),
    ('census;rm', '', 'PROFILE must be a comma list'),
    ('', '/data/prof', 'PROFILE_DIR needs PROFILE'),
    ('census', 'relative', 'PROFILE_DIR must be an absolute directory'),
    ('nsys', '', 'PROFILE=nsys needs PROFILE_DIR'),
    ('ncu', '', 'PROFILE=ncu needs PROFILE_DIR'),
    ('nsys,ncu', '/data/prof', 'nsys or ncu, not both'),
])
def test_helper_refuses_bad_values(profile, directory, message):
    out = _helper(profile, directory, 'kiwi')
    assert out.returncode != 0 and message in out.stderr


def test_family_launch_forwards_profile_to_coordinator_and_each_worker(tmp_path):
    prof = tmp_path / 'prof'
    result = _family_launch_result(tmp_path, GLMF_TWO_LAYER, 'test/glmf',
                                   f'GLM5_FLASH_FP8_MODEL_ID=off\nPROFILE=transport-timing,expert-timing,draft-timing\nPROFILE_DIR={prof}\n')
    assert result.returncode == 0, result.stderr
    lines = result.stderr.splitlines()
    coordinator = next(line for line in lines if 'cuteafd serve-glmf' in line)
    workers = [line for line in lines if 'cuteafd expertd-native' in line]
    assert workers
    for line, host in [(coordinator, HOST), *[(w, 'h0') for w in workers]]:
        for flag in ('CUTEAFD_PROTOCOL_V2_TCP_TIMING=1', 'CUTEAFD_EXPERT_TIMING=1', 'CUTEAFD_DRAFT_TIMING=1', f'CUTEAFD_PROFILE_HOST={host}',
                     f'CUTEAFD_PROFILE_DIR={prof}', f'{prof}:{prof}'):
            assert flag in line, (flag, line)
    assert all(f'mkdir -p {prof} && release_require_writable_state {prof} && docker run' in w for w in workers)
    assert prof.is_dir()


def test_family_launch_wraps_only_the_coordinator_under_nsys(tmp_path):
    prof = tmp_path / 'prof'
    result = _family_launch_result(tmp_path, GLMF_TWO_LAYER, 'test/glmf',
                                   f'GLM5_FLASH_FP8_MODEL_ID=off\nPROFILE=nsys,expert-timing\nPROFILE_DIR={prof}\n')
    assert result.returncode == 0, result.stderr
    lines = result.stderr.splitlines()
    workers = [line for line in lines if 'cuteafd expertd-native' in line]
    assert workers
    text = result.stderr
    marker = 'cuteafd serve-glmf'
    assert text.count(marker) == 1
    start = text.rfind('docker ', 0, text.index(marker))
    coordinator_cmd = text[start:text.index('\n', text.index(marker))]
    assert '--stop-signal SIGINT' in coordinator_cmd and 'cuteafd.graceful-stop=1' in coordinator_cmd
    assert 'release_profile_exec' in coordinator_cmd and ' nsys profile ' in coordinator_cmd
    assert f'--output={prof}/nsys-{HOST}' in coordinator_cmd
    assert coordinator_cmd.index('--force-overwrite=true') < coordinator_cmd.index(marker)
    for worker in workers:
        assert 'release_profile' not in worker and 'bin/nsys' not in worker
        assert 'graceful-stop' not in worker and 'CUTEAFD_EXPERT_TIMING=1' in worker


def test_family_launch_ncu_without_filter_fails_closed(tmp_path):
    result = _family_launch_result(tmp_path, GLMF_TWO_LAYER, 'test/glmf',
                                   f'GLM5_FLASH_FP8_MODEL_ID=off\nPROFILE=ncu\nPROFILE_DIR={tmp_path / "prof"}\n')
    assert result.returncode != 0 and 'PROFILE=ncu needs PROFILE_NCU_KERNEL' in result.stderr
    assert 'cuteafd serve-glmf' not in result.stderr


def test_family_launch_without_profile_adds_nothing(tmp_path):
    result = _family_launch_result(tmp_path, GLMF_TWO_LAYER, 'test/glmf', 'GLM5_FLASH_FP8_MODEL_ID=off\n')
    assert result.returncode == 0, result.stderr
    assert 'CUTEAFD_PROFILE' not in result.stderr and 'CUTEAFD_EXPERT_TIMING' not in result.stderr


def test_v41_worker_takes_profile_args_after_every_positional():
    source = (ROOT / 'scripts/launch/run-family.sh').read_text()
    invocation, remote = source.split('echo "== starting native Spark experts =="', 1)[1].split("<<'REMOTE' &", 1)
    remote = remote.split('\nREMOTE', 1)[0]
    assert '--profile-args' in invocation
    assert '"${rdma_args[@]}" "${wip_args[@]}" ${profile_args[@]+"${profile_args[@]}"}' in remote
    positional = ['image', 'name', '1', '4096', '107374182400', '19441', 'snapshot', 'fingerprint', '3', '4',
                  '0', '', '', '', '', '', 'info', '__none__']
    for tail, expected in [([], []), (['--profile-args', '-e', 'CUTEAFD_EXPERT_TIMING=1', '-e', 'CUTEAFD_PROFILE_HOST=emu'],
                                      ['CUTEAFD_EXPERT_TIMING=1', 'CUTEAFD_PROFILE_HOST=emu'])]:
        result = subprocess.run(['bash', '-euc', 'source scripts/lib/release-common.sh; source scripts/lib/cdi-gpus.sh; docker() { printf "%s\\n" "$@"; }\n' + remote.replace(' >/dev/null', ''),
                                 'test', *positional, *tail], cwd=ROOT, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        arguments = result.stdout.splitlines()
        assert arguments[arguments.index('--world') + 1] == '4'
        for flag in expected:
            assert flag in arguments
        assert ('CUTEAFD_PROFILE_HOST=emu' in arguments) == bool(expected)
    coordinator = source.split('start_coordinator() {', 1)[1].split('\n}\n', 1)[0] if 'start_coordinator() {' in source else source
    assert '${profile_args[@]+"${profile_args[@]}"}' in coordinator
    assert '"$COORDINATOR_DOCKER_INFERENCE" ${profile_wrapper[@]+"${profile_wrapper[@]}"} cuteafd' in coordinator
    assert '${profile_wrapper_args[@]+"${profile_wrapper_args[@]}"}' in coordinator
    # V4.1 takes the coordinator wrapper from the shared helper every family uses.
    assert source.count('release_profile_wrapper "$(get PROFILE)" "$(get PROFILE_DIR)" "$(hostname -s)" coordinator') == 1
    v41 = source[source.index('launch_deepseek_v41() {'):]
    assert 'shared_profile_args' in v41[:v41.index('start_coordinator() {')]

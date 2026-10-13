"""wip.sh's coordinator export: off-queue modes first (remote export host,
capped beside serving), the fair GPU turn last, falling through on exit 75."""
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WIP = (ROOT / 'wip.sh').read_text()


def _function(name):
    start = WIP.index(name + '() {')
    return WIP[start:WIP.index('\n}\n', start) + 3]


def _dispatch():
    start = WIP.index('    local mode exported=0')
    return WIP[start:WIP.index('    ((exported)) || release_die', start)]


@pytest.mark.parametrize('setting,host,expected', [
    ('auto', '', 'capped turn'),
    ('auto', 'oliver', 'capped remote turn'),
    ('remote', '', 'remote turn'),
    ('capped', 'oliver', 'capped turn'),
    ('turn', 'oliver', 'turn'),
])
def test_mode_order(setting, host, expected):
    script = 'release_die() { echo "DIE $*"; exit 9; }\n' + _function('wip_coordinator_export_modes') + \
        f'CUTEAFD_WIP_COORDINATOR_EXPORT={setting} CUTEAFD_EXPORT_HOST={host} wip_coordinator_export_modes\n'
    result = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.split() == expected.split()


def test_unknown_mode_is_refused():
    script = 'release_die() { echo "DIE $*"; exit 9; }\n' + _function('wip_coordinator_export_modes') + \
        'CUTEAFD_WIP_COORDINATOR_EXPORT=bogus wip_coordinator_export_modes\n'
    result = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 9 and 'must be auto, remote, capped or turn' in result.stdout


@pytest.mark.parametrize('capped,remote,expected_done,expected_calls', [
    (0, 0, 'capped', ['capped']),
    (75, 0, 'remote', ['capped', 'remote']),
    (75, 75, 'turn', ['capped', 'remote', 'turn']),
])
def test_dispatch_falls_through_to_the_fair_turn(tmp_path, capped, remote, expected_done, expected_calls):
    trace = tmp_path / 'trace'
    script = f'''
release_die() {{ echo "DIE $*"; exit 9; }}
wip_coordinator_export_modes() {{ printf 'capped remote turn\\n'; }}
wip_remote_export() {{ echo remote >> {trace}; return {remote}; }}
wip_capped_export() {{ echo capped >> {trace}; return {capped}; }}
wip_hwlock() {{ echo turn >> {trace}; }}
wip_export_gpu_lock=gpu1 coordinator_build=(docker exec c)
run() {{
{_dispatch()}
}}
run
'''
    result = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f'done by {expected_done}' in result.stdout
    assert trace.read_text().split() == expected_calls


def test_capped_export_runs_only_on_a_gpu_whose_lock_it_holds_and_is_watched():
    capped = _function('wip_capped_export')
    # Never queues (no fair-turn wait) and never picks a GPU across locks.
    assert 'wip_hwlock' not in capped
    assert 'export-placement.py" hold --gpus "${CUTEAFD_WIP_EXPORT_GPUS:-1,0}"' in capped
    on = _function('wip_capped_export_on')
    assert 'CUTEAFD_EXPORT_GPU' in on and 'memory.free' not in on and 'sort -t,' not in on
    assert '"$placement" admit' in on and 'watch --gpu "$gpu" --cap-mib "$cap" --cgroup "$id"' in on
    remote = _function('wip_remote_export')
    assert 'wip_hwlock' not in remote and 'CUTEAFD_WIP_NATIVE_REQUIRE_HIT=1' in remote


def test_remote_export_admits_from_the_hosts_wsl_smi_never_the_container():
    remote = (ROOT / 'scripts/build/export-remote.sh').read_text()
    assert 'smi=/usr/lib/wsl/lib/nvidia-smi' in remote
    admit = remote.index('"$placement" admit --smi "$smi"')
    container = remote.index('docker run --rm --gpus all --user')
    assert admit < container  # admission precedes the export container
    assert 'mem_get_info' not in remote
    # Bulk data only through sparknest, never one ssh pipe.
    assert 'nest replicate --hosts "$host"' in remote and 'wip-native/x86_64/coordinator-120' in remote
    # The interim SM-count guard went with #130: no export carries the card's SM count.
    assert 'sm-check' not in remote and 'sm-check' not in (ROOT / 'scripts/build/export-placement.py').read_text()


def test_capped_export_runs_the_native_phase_in_a_one_off_container_on_the_held_gpu(tmp_path):
    trace = tmp_path / 'trace'
    locks = tmp_path / 'locks'
    locks.mkdir()
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    (bindir / 'nvidia-smi').write_text('''#!/usr/bin/env bash
case "$1" in
  --query-gpu=index,uuid) printf '0, GPU-zero\\n1, GPU-one\\n' ;;
  --query-gpu=*) printf '0, GPU-zero, 90000, 97887, 7887\\n1, GPU-one, 500, 97887, 97387\\n' ;;
  --query-compute-apps=*) ;;
esac
''')
    (bindir / 'docker').write_text(f'#!/usr/bin/env bash\necho "docker $*" >> {trace}\n[[ "$1" == run ]] && echo cid123\nexit 0\n')
    for tool in ('nvidia-smi', 'docker'):
        (bindir / tool).chmod(0o755)
    common = (ROOT / 'scripts/lib/release-common.sh').read_text()
    helper = common[common.index('release_gpu_device_args() {'):common.index('\n}\n', common.index('release_gpu_device_args() {')) + 3]
    script = helper + _function('wip_capped_export') + _function('wip_capped_export_on') + f'''
repo_root={str(ROOT)!r} coordinator_container=wipc COORDINATOR_DOCKER_DEV=img
wip_capped_export docker exec -e CUTEAFD_WIP_DSV4_AOT=ON -e 'CUTEAFD_WIP_EXPERT_FAMILIES=a;b' wipc /x coordinator
'''
    result = subprocess.run(['bash', '-c', 'set -euo pipefail\n' + script], capture_output=True, text=True, timeout=60,
                            env={**os.environ, 'PATH': f'{bindir}:{os.environ["PATH"]}', 'CUTEAFD_LOCK_DIR': str(locks),
                                 'CUTEAFD_HW_LOCKS_HELD': ''})
    assert result.returncode == 0, result.stdout + result.stderr
    calls = trace.read_text()
    run = next(line for line in calls.splitlines() if line.startswith('docker run'))
    assert '--gpus device=GPU-one' in run and '--volumes-from wipc' in run
    assert '-e CUTEAFD_WIP_DSV4_AOT=ON' in run and '-e CUTEAFD_WIP_EXPERT_FAMILIES=a;b' in run
    assert 'docker exec cid123 /wip/source/scripts/build/build-wip-artifacts.sh' in calls and calls.rstrip().endswith('rm -f cid123')
    assert 'capped on gpu1' in result.stdout


def test_the_export_containers_gpu_binding_has_one_call_site():
    # The CDI switch (--device nvidia.com/gpu=...) edits release_gpu_device_args only.
    on = _function('wip_capped_export_on')
    assert 'release_gpu_device_args "$gpu"' in on and '--gpus' not in on


def test_remote_export_matches_the_dev_image_by_layer_content_not_local_id():
    # docker save | load gave oliver the same layers under another image ID; a
    # local-ID check refused a usable host. Content identity is the layer list,
    # while the native key keeps raptor's ID (CUTEAFD_WIP_DEV_IMAGE_ID via ENV_FILE).
    remote = (ROOT / 'scripts/build/export-remote.sh').read_text()
    assert "{{json .RootFS.Layers}}" in remote and "{{.Id}}" not in remote
    assert '{{json .RootFS.Layers}}' in _function('wip_remote_export')


def test_remote_staging_seals_with_one_path_per_nest_call():
    # nest seal takes a single <PATH> (-r for a tree); two paths exited 2.
    import re
    remote = (ROOT / 'scripts/build/export-remote.sh').read_text()
    for line in remote.splitlines():
        if re.search(r'\bnest seal\b', line):
            args = [a for a in line.split('nest seal', 1)[1].split('>')[0].split() if not a.startswith('-')]
            assert len(args) == 1, line.strip()

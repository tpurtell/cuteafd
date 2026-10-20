"""Coordinator export placement: admission from the host's nvidia-smi, the
memory cap, the SM-count check and the watchdog (scripts/build/export-placement.py)."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/build/export-placement.py'
spec = importlib.util.spec_from_file_location('export_placement', SCRIPT)
placement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(placement)


def fake_smi(tmp_path, gpus, apps='', name='nvidia-smi'):
    """An nvidia-smi stub: GPU rows (index, uuid, used, total, free) and compute apps
    (uuid, pid, used). A `state` file, when present, replaces the GPU rows, so a
    test can change what the stub reports while a watch runs."""
    state = tmp_path / (name + '.state')
    state.write_text(gpus)
    app_rows = tmp_path / (name + '.apps')
    app_rows.write_text(apps)
    stub = tmp_path / name
    stub.write_text(f'''#!/usr/bin/env bash
case "$1" in
  --query-gpu=*) cat {state} ;;
  --query-compute-apps=*) cat {app_rows} ;;
esac
''')
    stub.chmod(0o755)
    return stub, state


def run(*args, timeout=60):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=timeout)


@pytest.mark.parametrize('free,expected', [
    (97000, (placement.MAX_CAP_MIB, (placement.MAX_CAP_MIB - placement.SIZING_MIB) // placement.CONTEXT_MIB)),
    (32000, (placement.MAX_CAP_MIB, (placement.MAX_CAP_MIB - placement.SIZING_MIB) // placement.CONTEXT_MIB)),
    (10000, (10000 - placement.HEADROOM_MIB, (10000 - placement.HEADROOM_MIB - placement.SIZING_MIB) // placement.CONTEXT_MIB)),
    (placement.MIN_CAP_MIB + placement.HEADROOM_MIB - 1, None),
    (2000, None),
])
def test_cap_and_jobs_follow_the_free_memory_at_that_moment(free, expected):
    assert placement.admit(free) == expected


def test_admit_reads_the_host_smi_and_refuses_what_does_not_fit(tmp_path):
    smi, _ = fake_smi(tmp_path, '0, GPU-a, 65485, 97887, 31966\n1, GPU-b, 94127, 97887, 3760\n')
    ok = run('admit', '--smi', str(smi), '--gpu', '0')
    assert ok.returncode == 0, ok.stderr
    cap, jobs, used = map(int, ok.stdout.split())
    assert cap == placement.MAX_CAP_MIB and used == 65485 and jobs >= 1
    busy = run('admit', '--smi', str(smi), '--gpu', 'GPU-b')
    assert busy.returncode == placement.NO_TURN and 'MiB free' in busy.stderr


def test_wsl_admission_never_trusts_the_containers_view(tmp_path):
    # On oliver a container's cudaMemGetInfo reported 93.9 of 95.6 GiB free while
    # the host's WSL nvidia-smi showed TJ's job holding 65.5 GB. Admission takes
    # its reading from the HOST smi binary it is given; export-remote.sh passes
    # oliver's /usr/lib/wsl/lib/nvidia-smi and admits before any container starts.
    host, _ = fake_smi(tmp_path, '0, GPU-o, 65485, 97887, 4000\n', name='host-smi')
    container_view, _ = fake_smi(tmp_path, '0, GPU-o, 1700, 97887, 96187\n', name='container-smi')
    assert run('admit', '--smi', str(container_view), '--gpu', '0').returncode == 0
    assert run('admit', '--smi', str(host), '--gpu', '0').returncode == placement.NO_TURN
    remote = (ROOT / 'scripts/build/export-remote.sh').read_text()
    assert 'smi=/usr/lib/wsl/lib/nvidia-smi' in remote
    assert 'mem_get_info' not in SCRIPT.read_text().split('"""', 2)[2]


def test_watch_passes_through_an_export_that_stays_under_its_cap(tmp_path):
    smi, _ = fake_smi(tmp_path, '0, GPU-o, 65600, 97887, 31000\n')
    done = run('watch', '--smi', str(smi), '--gpu', '0', '--cap-mib', '4000', '--baseline-mib', '65485',
               '--interval', '0.1', '--', 'bash', '-c', 'sleep 0.5; exit 3')
    assert done.returncode == 3 and 'export peak 115 MiB' in done.stderr


def test_watch_kills_an_export_over_its_cap_and_reports_no_turn(tmp_path):
    smi, state = fake_smi(tmp_path, '0, GPU-o, 65485, 97887, 31966\n')
    marker = tmp_path / 'survived'
    # The "export" grows past the cap after a moment; the watchdog must stop it.
    command = f'printf "0, GPU-o, 75000, 97887, 22000\\n" > {state}; sleep 30; touch {marker}'
    stopped = run('watch', '--smi', str(smi), '--gpu', '0', '--cap-mib', '4000', '--baseline-mib', '65485',
                  '--interval', '0.1', '--', 'bash', '-c', command, timeout=45)
    assert stopped.returncode == placement.NO_TURN and 'over its 4000 MiB cap' in stopped.stderr
    assert not marker.exists()


def test_watch_counts_only_the_export_containers_processes(tmp_path, monkeypatch):
    # Serving processes on the same GPU are not the export's: only pids in the
    # export container's cgroup count against its cap.
    smi, _ = fake_smi(tmp_path, '1, GPU-r, 88649, 97887, 9238\n',
                      apps='GPU-r, 100, 80000\nGPU-r, 200, 600\nGPU-r, 201, 1900\n')
    monkeypatch.setattr(placement, 'cgroup_pids', lambda cid: {200, 201})
    assert placement.export_mib(str(smi), '1', 'abc', None) == 2500


def test_watch_stops_on_a_failed_reading(tmp_path, monkeypatch):
    calls = iter([subprocess.SubprocessError('smi hiccup'), 10, 10])
    def reading(*args):
        value = next(calls, 10)
        if isinstance(value, Exception):
            raise value
        return value
    monkeypatch.setattr(placement, 'export_mib', reading)
    args = placement.argparse.Namespace(smi='x', gpu='0', cgroup=None, baseline_mib=0, cap_mib=100,
                                        interval=0.05, command=['bash', '-c', 'sleep 0.3'])
    assert placement.watch(args) == placement.NO_TURN


def _hold(tmp_path, held=(), env_held='', gpus='1,0'):
    """Run `hold` with `held` GPU locks taken by another process; report the GPU used."""
    locks = tmp_path / 'locks'
    locks.mkdir(exist_ok=True)
    holders = []
    for gpu in held:
        holders.append(subprocess.Popen(['flock', str(locks / f'gpu{gpu}.lock'), 'sleep', '30']))
    try:
        import time
        time.sleep(0.3)
        probe = ('import fcntl,os,sys; g=os.environ["CUTEAFD_EXPORT_GPU"]; '
                 'print(g, os.environ.get("CUTEAFD_HW_LOCKS_HELD",""))')
        env = {**os.environ, 'CUTEAFD_LOCK_DIR': str(locks), 'CUTEAFD_HW_LOCKS_HELD': env_held}
        return subprocess.run([sys.executable, str(SCRIPT), 'hold', '--gpus', gpus, '--wait', '0', '--', sys.executable, '-c', probe],
                              capture_output=True, text=True, timeout=30, env=env)
    finally:
        for holder in holders:
            holder.kill()


def test_hold_uses_a_gpu_the_caller_already_holds_without_probing(tmp_path):
    # The caller holds gpu0 for its own run (an export beside it); even with
    # gpu0.lock held (by the caller) that GPU is the one used.
    result = _hold(tmp_path, held=('0',), env_held='gpu0')
    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[0] == '0' and 'already holds gpu0' in result.stderr


def test_hold_queues_for_first_gpu_and_honors_timeout(tmp_path):
    first = _hold(tmp_path)
    assert first.returncode == 0 and first.stdout.split() == ['1', 'gpu1']
    second = _hold(tmp_path, held=('1',))
    assert second.returncode == placement.NO_TURN and not second.stdout


def test_hold_never_uses_a_gpu_someone_else_holds(tmp_path):
    # Both locks held (another job, or TJ's interactive gpu0): exit 75 so wip.sh
    # moves on to the export host or the fair queue.
    result = _hold(tmp_path, held=('1', '0'))
    assert result.returncode == placement.NO_TURN and 'no turn for gpu1' in result.stderr


def test_reservations_share_baseline_and_cannot_double_admit(tmp_path):
    path = tmp_path / 'ledger.json'
    row = ['0', 'GPU-a', '65000', '98000', '33000']
    first = placement.reserve(path, 'a', row, 64000, 1, max_cap=12288, max_jobs=8)
    second = placement.reserve(path, 'b', row, 64000, 2, max_cap=12288, max_jobs=8)
    assert first and second
    baseline, budget = placement.reservation_budget(path, 'a')
    assert baseline == 65000 and budget == first[0] + second[0]
    state = __import__('json').loads(path.read_text())
    assert sum(v['ram'] for v in state['leases'].values()) <= 64000 - placement.RAM_HEADROOM_MIB
    assert first[0] == first[1] * (placement.CONTEXT_MIB + placement.SIZING_MIB)
    # Baseline is shared, even when the second entrant sees allocated growth.
    assert placement.reserve(path, 'c', ['0', 'GPU-a', '76000', '98000', '22000'], 32000, 3) is None
    placement.release_reservation(path, 'a')
    assert placement.reservation_budget(path, 'b')[1] == second[0]


def test_ram_limits_concurrent_sizing_workers(tmp_path):
    row = ['0', 'GPU-a', '1000', '98000', '97000']
    assert placement.reserve(tmp_path / 'ledger.json', 'a', row, 16000, 1) is None


@pytest.mark.parametrize("signum", [15, 1, 3])
def test_hold_signal_retains_mutex_until_child_cleanup(tmp_path, signum):
    import fcntl
    import signal
    import time
    lock = tmp_path / 'gpu0.lock'
    ready, stopping, done = [tmp_path / name for name in ('ready', 'stopping', 'done')]
    command = f"trap 'touch {stopping}; sleep 0.5; touch {done}; exit 0' TERM HUP QUIT; touch {ready}; while true; do sleep 0.1; done"
    process = subprocess.Popen([sys.executable, str(SCRIPT), 'hold', '--gpus', '0', '--', 'bash', '-c', command],
                               env={**os.environ, 'CUTEAFD_LOCK_DIR': str(tmp_path)}, start_new_session=True)

    def wait_for(path):
        deadline = time.monotonic() + 30
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        return path.exists()

    try:
        assert wait_for(ready)
        process.send_signal(signum)
        assert wait_for(stopping)
        with lock.open('a') as handle:
            # The mutex may only be free once the child's cleanup has finished.
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                assert done.exists()
                fcntl.flock(handle, fcntl.LOCK_UN)
            assert process.wait(timeout=30) == 0
            assert done.exists()
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def test_capped_and_remote_cleanup_require_successful_absence_query():
    remote = (ROOT / 'scripts/build/export-remote.sh').read_text()
    cleanup = remote.split('cleanup_export() {', 1)[1].split('trap cleanup_export', 1)[0]
    assert 'docker ps -a' in cleanup and 'sleep 60' in cleanup
    assert 'docker container inspect' not in cleanup
    wip = (ROOT / 'wip.sh').read_text().split('wip_capped_export_on() (', 1)[1].split('build_coordinator()', 1)[0]
    assert 'trap cleanup_capped EXIT' in wip
    assert 'timeout --kill-after=30 7200' in wip
    assert '-e CUTEAFD_EXPORT_HOST_SLOT_DIR=/export-slots' in wip
    assert 'CUTEAFD_EXPORT_GPU_BUDGET_MIB=$cap' in wip
    assert 'docker rm -f "$id" >/dev/null 2>&1 || true' not in wip


def test_capped_cleanup_ignores_second_signal_and_retains_gpu_lock(tmp_path):
    import fcntl
    import signal
    import time
    wip = (ROOT / 'wip.sh').read_text()
    cleanup = wip[wip.index('  cleanup_capped() {'):wip.index('  trap cleanup_capped EXIT')]
    tools = tmp_path / 'bin'
    tools.mkdir()
    entered, allow, ready = [tmp_path / name for name in ('entered', 'allow', 'ready')]
    docker = tools / 'docker'
    docker.write_text(f'#!/bin/sh\ntouch {entered}\ntest -f {allow} && exit 0\nexit 1\n')
    docker.chmod(0o755)
    pause = tools / 'sleep'
    pause.write_text('#!/bin/sh\nexec /bin/sleep 0.05\n')
    pause.chmod(0o755)
    command = cleanup + f"\nname=owned; trap cleanup_capped EXIT; trap 'exit 143' TERM INT HUP QUIT; touch {ready}; while true; do sleep 60; done"
    process = subprocess.Popen([sys.executable, str(SCRIPT), 'hold', '--gpus', '0', '--', 'bash', '-c', command],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               env={**os.environ, 'PATH': f'{tools}:{os.environ["PATH"]}',
                                    'CUTEAFD_LOCK_DIR': str(tmp_path)})
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        process.send_signal(signal.SIGTERM)
        while not entered.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert entered.exists()
        process.send_signal(signal.SIGTERM)
        time.sleep(0.15)
        assert process.poll() is None
        with (tmp_path / 'gpu0.lock').open('a') as handle:
            with pytest.raises(BlockingIOError):
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            allow.touch()
            assert process.wait(timeout=5) == 143
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        allow.touch()
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=5)


def test_run_held_forwards_a_signal_that_arrives_before_the_child_exists(tmp_path, monkeypatch):
    import importlib.util
    import signal
    import time
    spec = importlib.util.spec_from_file_location('export_placement', SCRIPT)
    placement = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(placement)
    stopping = tmp_path / 'stopping'
    command = ['bash', '-c', f"trap 'touch {stopping}; exit 0' HUP; while true; do sleep 0.1; done"]
    real_popen = placement.subprocess.Popen

    def popen_after_signal(*args, **kwargs):
        # The signal lands in the window between installing the forwarders and the spawn.
        os.kill(os.getpid(), signal.SIGHUP)
        process = real_popen(*args, **kwargs)
        time.sleep(0.3)  # let bash install its trap before the deferred forward
        return process

    monkeypatch.setattr(placement.subprocess, 'Popen', popen_after_signal)
    assert placement.run_held(command, dict(os.environ)) == 0
    assert stopping.exists()


def test_queued_export_releases_seed_and_build_locks_for_serving_ticket(tmp_path):
    import time
    locks = tmp_path / 'locks'
    locks.mkdir()
    ready, request_seed, served, exported = [tmp_path / name for name in ('ready', 'request-seed', 'served', 'exported')]
    env = {**os.environ, 'CUTEAFD_LOCK_DIR': str(locks), 'CUTEAFD_HW_LOCKS_HELD': ''}
    serving_command = f"touch '{ready}'; until test -f '{request_seed}'; do sleep 0.01; done; flock -w 10 '{locks}/kakapo.lock' touch '{served}'"
    serving = subprocess.Popen([sys.executable, str(ROOT / 'scripts/hw/hwlock.py'), '--locks', 'gpu0,gpu1',
                                '--wait', '15', '--', 'bash', '-c', serving_command], env=env)
    text = (ROOT / 'wip.sh').read_text()
    flow = text.split('wip_build_selected_roles() {', 1)[1].split('\n}\n', 1)[0]
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        script = f"""set -euo pipefail
role=both; wip_reused_coordinator=0; wip_reused_expert=0
wip_retain_counterpart() {{ :; }}
wip_build_resource_acquire() {{ exec {{seed_fd}}>'{locks}/kakapo.lock'; flock "$seed_fd"; }}
build_expert() {{ touch '{request_seed}'; }}
wip_build_resource_release() {{ exec {{seed_fd}}>&-; }}
wip_distribute_sealed_expert() {{ :; }}
build_coordinator() {{
  if [[ $1 == rust ]]; then
    flock -w 5 '{locks}/build.lock' true
  else
    flock -n '{locks}/build.lock' true
    python3 '{SCRIPT}' hold --gpus 1,0 --wait 15 -- touch '{exported}'
  fi
}}
{flow}
"""
        result = subprocess.run(['bash', '-c', script], capture_output=True, text=True, env=env, timeout=25)
        assert result.returncode == 0, result.stdout + result.stderr
        assert serving.wait(timeout=10) == 0
        assert served.exists() and exported.exists()
        assert not list((locks / 'hwq').glob('*.ticket'))
    finally:
        if serving.poll() is None:
            serving.kill()
            serving.wait()


def test_queue_waits_when_both_gpus_are_busy_then_exports(tmp_path):
    import time
    ready = tmp_path / 'ready'
    env = {**os.environ, 'CUTEAFD_LOCK_DIR': str(tmp_path), 'CUTEAFD_HW_LOCKS_HELD': ''}
    holder = subprocess.Popen([sys.executable, str(ROOT / 'scripts/hw/hwlock.py'), '--locks', 'gpu0,gpu1',
                               '--wait', '5', '--', 'bash', '-c', f'touch "{ready}"; sleep 1'], env=env)
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        result = subprocess.run([sys.executable, str(SCRIPT), 'hold', '--gpus', '1,0', '--wait', '10',
                                 '--', 'bash', '-c', 'echo "$CUTEAFD_EXPORT_GPU $CUTEAFD_HW_LOCKS_HELD"'],
                                capture_output=True, text=True, env=env, timeout=15)
        assert result.returncode == 0, result.stderr
        assert 'queued for gpu1' in result.stderr and result.stdout.strip() == '1 gpu1'
        assert holder.wait(timeout=5) == 0
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait()

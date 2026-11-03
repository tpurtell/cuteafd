"""Build-only resources count leases; serving hosts remain exclusive."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python/tools/lib'))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


resource = load('build_resource', ROOT / 'scripts/build/build-resource.py')
export = load('export_jobs', ROOT / 'python/tools/lib/_export_jobs.py')


@pytest.fixture(autouse=True)
def quiet_host(tmp_path_factory, monkeypatch):
    # In-process admissions never read the gate host's live PSI or nvidia-smi.
    monkeypatch.setattr(export, 'PRESSURE_PATH', tmp_path_factory.mktemp('psi') / 'absent')
    monkeypatch.setattr(export, 'live_cuda_processes', lambda: 'stubbed')
    yield


LIVE_CUDA_PROCESSES = export.live_cuda_processes


def test_seed_capacity_uses_cores_and_memory():
    assert resource.capacity(20, 124608) == 3
    assert resource.capacity(64, 65536) == 2
    assert resource.capacity(2, 4096) == 1


def test_counting_slots_preserve_legacy_exclusion_and_release(tmp_path):
    leases = [resource.acquire(tmp_path, 'kakapo', 3) for _ in range(3)]
    try:
        assert all(leases)
        assert resource.acquire(tmp_path, 'kakapo', 3) is None
        assert resource.try_lock(tmp_path / 'kakapo.lock') is None
        for handle in leases.pop():
            handle.close()
        extra = resource.acquire(tmp_path, 'kakapo', 3)
        assert extra
        leases.append(extra)
    finally:
        for lease in leases:
            for handle in lease:
                handle.close()
    exclusive = resource.try_lock(tmp_path / 'kakapo.lock')
    assert exclusive
    try:
        assert resource.acquire(tmp_path, 'kakapo', 3) is None
    finally:
        exclusive.close()


def test_busy_kakapo_falls_back_only_under_serving_host_lock(tmp_path):
    held = resource.acquire(tmp_path, 'kakapo', 1)
    rhea = resource.try_lock(tmp_path / 'rhea.lock')
    lease = tmp_path / 'lease'
    lease.mkdir()
    os.mkfifo(lease / 'release')
    process = subprocess.Popen(['python3', str(ROOT / 'scripts/build/build-resource.py'),
                               '--slots', '1', '--fallback', 'rhea,moa', '--wait', '0',
                               '--lease-dir', str(lease)],
                              env={**os.environ, 'CUTEAFD_LOCK_DIR': str(tmp_path)})
    try:
        deadline = time.monotonic() + 5
        while not (lease / 'granted').exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert json.loads((lease / 'granted').read_text())['host'] == 'moa'
        assert resource.try_lock(tmp_path / 'moa.lock') is None
        with (lease / 'release').open('w') as handle:
            handle.write('x')
        assert process.wait(timeout=5) == 0
        released = resource.try_lock(tmp_path / 'moa.lock')
        assert released
        released.close()
    finally:
        process.terminate()
        process.wait(timeout=5)
        rhea.close()
        for handle in held:
            handle.close()


@pytest.mark.parametrize('kind,cores,expected', [('raptor', 64, 16), ('remote', 20, 8), ('remote', 4, 4)])
def test_export_budget(kind, cores, expected, monkeypatch):
    monkeypatch.setattr(os, 'cpu_count', lambda: cores)
    monkeypatch.delenv('CUTEAFD_EXPORT_JOBS', raising=False)
    monkeypatch.delenv('CARGO_BUILD_JOBS', raising=False)
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_CLASS', kind)
    assert export.export_jobs() == expected
    monkeypatch.setenv('CARGO_BUILD_JOBS', '7')
    assert export.export_jobs() == expected
    monkeypatch.setenv('CUTEAFD_EXPORT_JOBS', '3')
    assert export.export_jobs() == 3
    monkeypatch.setenv('CUTEAFD_EXPORT_JOBS', '0')
    with pytest.raises(ValueError):
        export.export_jobs()


def test_export_slot_released_after_failure(tmp_path, monkeypatch):
    monkeypatch.setenv('CUTEAFD_EXPORT_SLOT_DIR', str(tmp_path))
    monkeypatch.setenv('CUTEAFD_EXPORT_JOBS', '1')
    with pytest.raises(RuntimeError):
        with export.export_slot():
            assert resource.try_lock(tmp_path / '0.lock') is None
            raise RuntimeError('compiler failure')
    with export.export_slot():
        assert resource.try_lock(tmp_path / '0.lock') is None


def test_build_slot_default_admits_five_and_keeps_nice(tmp_path):
    held = []
    try:
        for name in ['build.lock', 'build.lock.1', 'build.lock.2', 'build.lock.3']:
            held.append(resource.try_lock(tmp_path / name))
        result = subprocess.run(['bash', str(ROOT / 'scripts/build/build-slot.sh'), 'bash', '-c',
                                 'ps -o ni= -p $$'], capture_output=True, text=True, timeout=5,
                                env={**os.environ, 'CUTEAFD_LOCK_DIR': str(tmp_path), 'CUTEAFD_BUILD_SLOT_WAIT': '0'})
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == '19'
        assert 'build slot 4:' in result.stderr
    finally:
        for handle in held:
            handle.close()


def test_failed_probe_is_unavailable(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(255, 'ssh')
    monkeypatch.setattr(resource.subprocess, 'run', fail)
    assert resource.probe('kakapo') is None


def test_owner_death_before_grant_does_not_acquire(tmp_path):
    held = resource.acquire(tmp_path, 'kakapo', 1)
    lease = tmp_path / 'lease'
    lease.mkdir()
    os.mkfifo(lease / 'release')
    owner = subprocess.Popen(['sleep', '30'])
    process = subprocess.Popen(['python3', str(ROOT / 'scripts/build/build-resource.py'), '--slots', '1',
                                '--owner', str(owner.pid), '--wait', '30', '--lease-dir', str(lease)],
                               env={**os.environ, 'CUTEAFD_LOCK_DIR': str(tmp_path)})
    try:
        time.sleep(0.1)
        owner.terminate()
        owner.wait(timeout=5)
        assert process.wait(timeout=5) == 0
        assert not (lease / 'granted').exists()
    finally:
        process.terminate()
        process.wait(timeout=5)
        for handle in held:
            handle.close()


def test_lease_has_no_elapsed_unlock_deadline_and_stops_before_unlock():
    text = (ROOT / 'scripts/build/build-resource.py').read_text()
    granted = text[text.index("(args.lease_dir / 'granted')"):]
    assert 'deadline' not in granted and '7260' not in granted
    assert granted.index('docker stop -t 10') < granted.index('handle.close()')


def test_host_worker_admission_is_shared_across_builds(tmp_path, monkeypatch):
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_SLOT_DIR', str(tmp_path / 'host'))
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_WORKERS', '1')
    monkeypatch.setenv('CUTEAFD_EXPORT_JOBS', '8')
    monkeypatch.setenv('CUTEAFD_EXPORT_SLOT_DIR', str(tmp_path / 'build-a'))
    with export.export_slot():
        assert resource.try_lock(tmp_path / 'host/0.lock') is None
        assert resource.try_lock(tmp_path / 'build-a/0.lock') is None
    with export.export_slot():
        assert resource.try_lock(tmp_path / 'host/0.lock') is None


def test_low_memfree_high_cache_admits_all_gb10_workers_with_spacing(tmp_path, monkeypatch):
    monkeypatch.setattr(export, '_context_lease', None)
    monkeypatch.setattr(export, '_gb10_host', lambda: True)
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_SLOT_DIR', str(tmp_path))
    monkeypatch.setenv('CUTEAFD_EXPORT_WORKER_RAM_MIB', '1536')
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_WORKERS', '20')
    monkeypatch.delenv('CUTEAFD_EXPORT_HOST_CONTEXTS', raising=False)
    monkeypatch.delenv('CUTEAFD_EXPORT_CONTEXT_SPACING_S', raising=False)
    original = Path.read_text
    def text(path, *args, **kwargs):
        if str(path) == '/proc/self/cgroup':
            return ''
        if str(path) == '/proc/meminfo':
            return 'MemFree: 512000 kB\nMemAvailable: 125829120 kB\nCached: 94371840 kB\n'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', text)
    clock = [0.0]
    monkeypatch.setattr(export.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(export.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    held, stamps = [], []
    try:
        for _ in range(export.GB10_EXPORT_CONTEXTS):
            export.context_admission()
            held.append(export._context_lease)
            stamps.append(export._context_admitted_at)
            export._context_lease = None
        assert len(held) == 8
        assert all(b - a >= export.CONTEXT_SPACING_S for a, b in zip(stamps, stamps[1:]))
        assert json.loads((tmp_path / 'context-baseline.json').read_text())['available'] == 122880
        details = {}
        assert export._admit_locked(tmp_path, 1536, 8, 1.5, details) is None
        assert 'leases=8/8' in details['status']
    finally:
        for handle in held:
            handle.close()
        export._context_lease = None


@pytest.mark.parametrize('reason', ['memory', 'pressure', 'leases', 'lock'])
def test_context_wait_warns_every_30s_and_fails_within_10_minutes(tmp_path, monkeypatch, capsys, reason):
    monkeypatch.setattr(export, '_context_lease', None)
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_SLOT_DIR', str(tmp_path))
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_CONTEXTS', '1')
    monkeypatch.setenv('CUTEAFD_EXPORT_WORKER_RAM_MIB', '1536')
    monkeypatch.setattr(export, 'host_allocatable_memory', lambda: (1024 if reason == 'memory' else 120000) * 1024**2)
    monkeypatch.setattr(export, 'memory_pressure', lambda: 25 if reason == 'pressure' else 0)
    clock = [0.0]
    monkeypatch.setattr(export.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(export.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    (tmp_path / 'context-baseline.json').write_text(json.dumps({'available': 120000}))
    held = None
    if reason in ('leases', 'lock'):
        held = (tmp_path / ('context.0.lock' if reason == 'leases' else 'context-admission.lock')).open('w+')
        held.write('1536')
        held.flush()
        fcntl.flock(held, fcntl.LOCK_EX)
    try:
        with pytest.raises(TimeoutError, match='after 600s:.*available=.*needed=.*leases='):
            export.context_admission()
        warnings = capsys.readouterr().out.splitlines()
        assert len(warnings) == 21
        for stamp, line in zip(range(0, 601, 30), warnings):
            assert f'WARN AOT context admission waiting {stamp}s:' in line
            assert 'available=' in line and 'needed=2560 MiB' in line and 'leases=' in line
        assert clock[0] == 600 and export._context_lease is None
    finally:
        if held:
            held.close()


@pytest.mark.parametrize('available,reservation,expected', [(120000, 1536, True), (7000, 4096, False)])
def test_context_admission_accounts_live_reservations_across_worker_limits(tmp_path, monkeypatch, available, reservation, expected):
    monkeypatch.setattr(export, '_context_lease', None)
    monkeypatch.setattr(export, 'host_allocatable_memory', lambda: available * 1024**2)
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_SLOT_DIR', str(tmp_path))
    monkeypatch.setenv('CUTEAFD_EXPORT_WORKER_RAM_MIB', str(reservation))
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_WORKERS', '2')
    (tmp_path / 'context-baseline.json').write_text(json.dumps({'available': available}))
    held = (tmp_path / 'context.9.lock').open('w+')
    held.write('4096')
    held.flush()
    fcntl.flock(held, fcntl.LOCK_EX)
    original = Path.read_text
    def text(path, *args, **kwargs):
        if str(path) == '/proc/self/cgroup':
            return ''
        if str(path) == '/proc/meminfo':
            return f'MemFree: 512000 kB\nMemAvailable: {available * 1024} kB\n'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', text)
    def stop_wait(seconds):
        assert seconds == 1
        raise TimeoutError('test stops the bounded admission wait')
    monkeypatch.setattr(export.time, 'sleep', stop_wait)
    try:
        if expected:
            export.context_admission()
            assert export._context_lease
            assert (tmp_path / 'context.0.lock').read_text() == str(reservation)
        else:
            with pytest.raises(TimeoutError):
                export.context_admission()
            assert export._context_lease is None
    finally:
        held.close()
        if export._context_lease:
            export._context_lease.close()
        export._context_lease = None


_ADMIT_CHILD = """
import json, os, sys, time
sys.path.insert(0, sys.argv[1])
import _export_jobs
# The guard's own spaced stamp; a post-return clock read adds scheduler jitter.
admitted = _export_jobs.context_admission()
time.sleep(float(sys.argv[2]))
print(json.dumps([admitted, time.monotonic()]), flush=True)
"""


def _admit_processes(tmp_path, count, hold, extra_env):
    env = {**os.environ, 'CUTEAFD_EXPORT_HOST_SLOT_DIR': str(tmp_path),
           'CUTEAFD_EXPORT_WORKER_RAM_MIB': '1', **extra_env}
    processes = [subprocess.Popen([sys.executable, '-c', _ADMIT_CHILD, str(ROOT / 'python/tools/lib'), str(hold)],
                                  env=env, stdout=subprocess.PIPE, text=True) for _ in range(count)]
    spans = []
    for process in processes:
        out, _ = process.communicate(timeout=60)
        assert process.returncode == 0
        spans.append(json.loads(out.strip().splitlines()[-1]))
    return sorted(spans)


def test_context_cap_and_spacing_hold_across_processes(tmp_path):
    spans = _admit_processes(tmp_path, 5, 1.6, {'CUTEAFD_EXPORT_HOST_CONTEXTS': '2',
                                                 'CUTEAFD_EXPORT_CONTEXT_SPACING_S': '0.25'})
    starts = [start for start, _ in spans]
    assert all(b - a >= 0.24 for a, b in zip(starts, starts[1:]))
    for start, _ in spans:
        live = sum(1 for other_start, other_end in spans if other_start <= start < other_end)
        assert live <= 2
    # Every lease was released on exit: all are lockable again.
    leases = sorted(tmp_path.glob('context.*.lock'))
    assert [path.name for path in leases] == ['context.0.lock', 'context.1.lock']
    for path in leases:
        assert resource.try_lock(path) is not None


@pytest.mark.parametrize('cls,machine,workers,expected', [
    ('remote', 'x86_64', '20', 8), ('raptor', 'aarch64', '20', 8),
    ('raptor', 'x86_64', '64', 64), ('remote', 'x86_64', '4', 4)])
def test_host_context_cap_defaults(monkeypatch, cls, machine, workers, expected):
    import platform
    monkeypatch.delenv('CUTEAFD_EXPORT_HOST_CONTEXTS', raising=False)
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_CLASS', cls)
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_WORKERS', workers)
    monkeypatch.setattr(platform, 'machine', lambda: machine)
    assert export.host_context_cap() == expected
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_CONTEXTS', '3')
    assert export.host_context_cap() == 3


def test_parked_planner_frees_its_lease_but_keeps_its_reservation(tmp_path, monkeypatch):
    monkeypatch.setattr(export, '_context_lease', None)
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_SLOT_DIR', str(tmp_path))
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_CONTEXTS', '1')
    monkeypatch.setenv('CUTEAFD_EXPORT_CONTEXT_SPACING_S', '0')
    monkeypatch.setenv('CUTEAFD_EXPORT_WORKER_RAM_MIB', '1024')
    export.context_admission()
    try:
        export.park_context_lease()
        assert resource.try_lock(tmp_path / 'context.0.lock') is not None
        parked = list(tmp_path.glob('context.parked-*.lock'))
        assert len(parked) == 1 and parked[0].read_text() == '1024'
        # A worker still gets the single lease while the planner is parked.
        _admit_processes(tmp_path, 1, 0, {'CUTEAFD_EXPORT_HOST_CONTEXTS': '1',
                                          'CUTEAFD_EXPORT_CONTEXT_SPACING_S': '0'})
    finally:
        export._context_lease.close()
        export._context_lease = None
    # The next admission reaps the exited planner's parked file.
    _admit_processes(tmp_path, 1, 0, {'CUTEAFD_EXPORT_HOST_CONTEXTS': '1',
                                      'CUTEAFD_EXPORT_CONTEXT_SPACING_S': '0'})
    assert not list(tmp_path.glob('context.parked-*.lock'))


@pytest.mark.parametrize('busy_lock', [False, True])
def test_gpu_admission_wait_is_visible_and_bounded(tmp_path, monkeypatch, capsys, busy_lock):
    monkeypatch.setattr(export, '_gpu_lease', None)
    monkeypatch.setenv('CUTEAFD_EXPORT_GPU_SLOT_DIR', str(tmp_path))
    monkeypatch.setenv('CUTEAFD_EXPORT_GPU_BUDGET_MIB', '2048')
    monkeypatch.setenv('CUTEAFD_EXPORT_WORKER_GPU_MIB', '1536')
    held = (tmp_path / ('admission.lock' if busy_lock else 'context.live.lock')).open('w+')
    held.write('1536')
    held.flush()
    fcntl.flock(held, fcntl.LOCK_EX)
    clock = [0.0]
    monkeypatch.setattr(export.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(export.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    try:
        with pytest.raises(TimeoutError, match='GPU admission timed out after 600s:.*free=.*needed=.*leases='):
            export.gpu_admission()
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == 21 and clock[0] == 600
        assert all('WARN AOT GPU admission waiting' in line for line in lines)
    finally:
        held.close()


def test_weighted_gpu_budget_shared_across_package_workers(tmp_path, monkeypatch):
    monkeypatch.setattr(export, '_gpu_lease', None)
    monkeypatch.setenv('CUTEAFD_EXPORT_GPU_SLOT_DIR', str(tmp_path))
    monkeypatch.setenv('CUTEAFD_EXPORT_GPU_BUDGET_MIB', '12288')
    monkeypatch.setenv('CUTEAFD_EXPORT_WORKER_GPU_MIB', '2688')
    held = (tmp_path / 'context.999.lock').open('w+')
    held.write('10752')
    held.flush()
    fcntl.flock(held, fcntl.LOCK_EX)
    def stop_wait(seconds):
        raise TimeoutError('budget full')
    monkeypatch.setattr(export.time, 'sleep', stop_wait)
    try:
        with pytest.raises(TimeoutError):
            export.gpu_admission()
        assert export._gpu_lease is None
    finally:
        held.close()
    export.gpu_admission()
    assert export._gpu_lease
    assert (tmp_path / 'context.999.lock').read_text() == '2688'
    export._gpu_lease.close()
    export._gpu_lease = None


@pytest.mark.parametrize('signum', [1, 3])
def test_seed_hup_quit_stops_container_before_unlock(tmp_path, signum):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    marker = tmp_path / 'stopped'
    ssh = bin_dir / 'ssh'
    ssh.write_text(f'#!/bin/sh\nsleep 0.3\ntouch {marker}\n')
    ssh.chmod(0o755)
    lease = tmp_path / 'lease'
    lease.mkdir()
    os.mkfifo(lease / 'release')
    (lease / 'active').touch()
    process = subprocess.Popen(['python3', str(ROOT / 'scripts/build/build-resource.py'), '--host', 'rhea',
                                '--slots', '1', '--container', 'owned', '--lease-dir', str(lease)],
                               env={**os.environ, 'PATH': f'{bin_dir}:{os.environ["PATH"]}',
                                    'CUTEAFD_LOCK_DIR': str(tmp_path)})
    try:
        deadline = time.monotonic() + 5
        while not (lease / 'granted').exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (lease / 'granted').exists()
        process.send_signal(signum)
        assert resource.try_lock(tmp_path / 'rhea.lock') is None
        assert process.wait(timeout=5) == 0
        assert marker.exists()
        handle = resource.try_lock(tmp_path / 'rhea.lock')
        assert handle
        handle.close()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def test_gpu_lease_pid_collision_never_truncates_live_reservation(tmp_path, monkeypatch):
    monkeypatch.setattr(export, '_gpu_lease', None)
    monkeypatch.setenv('CUTEAFD_EXPORT_GPU_SLOT_DIR', str(tmp_path))
    monkeypatch.setenv('CUTEAFD_EXPORT_GPU_BUDGET_MIB', '12288')
    monkeypatch.setenv('CUTEAFD_EXPORT_WORKER_GPU_MIB', '640')
    live = tmp_path / f'context.{os.getpid()}.lock'
    held = live.open('w+')
    held.write('2688')
    held.flush()
    fcntl.flock(held, fcntl.LOCK_EX)
    try:
        export.gpu_admission()
        assert live.read_text() == '2688'
        assert Path(export._gpu_lease.name) != live
        assert export._gpu_lease.readable()
    finally:
        held.close()
        if export._gpu_lease:
            export._gpu_lease.close()
        export._gpu_lease = None


def _pressure(tmp_path, monkeypatch, some):
    path = tmp_path / 'pressure'
    path.write_text(f'some avg10={some} avg60=0.00 avg300=0.00 total=1\n'
                    'full avg10=0.00 avg60=0.00 avg300=0.00 total=1\n')
    monkeypatch.setattr(export, 'PRESSURE_PATH', path)
    return path


@pytest.mark.parametrize('text,expected', [
    ('some avg10=12.50 avg60=3.00 avg300=1.00 total=9\nfull avg10=4.00 avg60=0 avg300=0 total=1\n', 12.5),
    ('full avg10=40.00 avg60=0 avg300=0 total=1\n', 0.0), ('garbage\n', 0.0), (None, 0.0)])
def test_memory_pressure_reads_some_avg10(tmp_path, monkeypatch, text, expected):
    path = tmp_path / 'pressure'
    if text is not None:
        path.write_text(text)
    monkeypatch.setattr(export, 'PRESSURE_PATH', path)  # absent file: a kernel without PSI
    assert export.memory_pressure() == expected


def test_context_admission_waits_out_memory_pressure_then_logs_census(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(export, '_context_lease', None)
    slots = tmp_path / 'slots'
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_SLOT_DIR', str(slots))
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_CONTEXTS', '4')
    monkeypatch.setenv('CUTEAFD_EXPORT_CONTEXT_SPACING_S', '0')
    monkeypatch.setenv('CUTEAFD_EXPORT_WORKER_RAM_MIB', '1024')
    monkeypatch.setattr(export, 'live_cuda_processes', lambda: '2 compute processes (11:900 12:700)')
    pressure = _pressure(tmp_path, monkeypatch, '25.00')
    slots.mkdir()
    (slots / 'context-baseline.json').write_text(json.dumps({'available': 1 << 30}))
    # One leased worker and one parked planner already hold contexts.
    held = []
    for name in ('context.0.lock', 'context.parked-abc.lock'):
        handle = (slots / name).open('w+')
        handle.write('1024')
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_EX)
        held.append(handle)
    waits = []
    def sleep(seconds):
        waits.append(seconds)
        if len(waits) == 3:  # pressure subsides after three one-second checks
            pressure.write_text('some avg10=4.00 avg60=0 avg300=0 total=1\n')
    monkeypatch.setattr(export.time, 'sleep', sleep)
    try:
        export.context_admission()
        assert waits == [1, 1, 1]
        assert (slots / 'context.1.lock').read_text() == '1024'
        out = capsys.readouterr().out
        assert out.count('WARN AOT context admission waiting') == 1
        assert 'memory pressure some avg10=25.00 (limit 10)' in out
        assert 'live contexts 1 leased + 1 parked before this one' in out
        assert 'host compute: 2 compute processes (11:900 12:700)' in out
    finally:
        for handle in held:
            handle.close()
        if export._context_lease:
            export._context_lease.close()
        export._context_lease = None


def test_memory_pressure_at_threshold_still_admits(tmp_path, monkeypatch):
    monkeypatch.setattr(export, '_context_lease', None)
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_SLOT_DIR', str(tmp_path / 'slots'))
    monkeypatch.setenv('CUTEAFD_EXPORT_CONTEXT_SPACING_S', '0')
    monkeypatch.setattr(export, 'live_cuda_processes', lambda: 'none')
    _pressure(tmp_path, monkeypatch, '10.00')
    def no_wait(seconds):
        raise AssertionError('admission at the threshold must not wait')
    monkeypatch.setattr(export.time, 'sleep', no_wait)
    export.context_admission()
    try:
        assert export._context_lease
    finally:
        export._context_lease.close()
        export._context_lease = None


def test_live_cuda_processes_formats_nvidia_smi(monkeypatch):
    monkeypatch.setattr(export.shutil, 'which', lambda name: '/usr/bin/nvidia-smi')
    class Done:
        stdout = '4242, 3072\n4243, 812\n'
    monkeypatch.setattr(export.subprocess, 'run', lambda *a, **k: Done())
    assert LIVE_CUDA_PROCESSES() == '2 compute processes (4242:3072 4243:812)'
    monkeypatch.setattr(export.shutil, 'which', lambda name: None)
    assert LIVE_CUDA_PROCESSES() == 'nvidia-smi unavailable'


@pytest.mark.parametrize('cls,machine,jobs,expected', [
    ('remote', 'aarch64', 8, 4), ('remote', 'aarch64', 2, 2), ('raptor', 'aarch64', 8, 4),
    ('raptor', 'x86_64', 16, 16), ('remote', 'x86_64', 8, 4)])
def test_gpu_workers_cap_gb10_packages(monkeypatch, cls, machine, jobs, expected):
    import platform
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_CLASS', cls)
    monkeypatch.setattr(platform, 'machine', lambda: machine)
    assert export.gpu_workers(jobs) == expected


_POOL_CHILD = """
import os, sys
sys.path.insert(0, sys.argv[1])
import package_exl3_aot as package
pids = package.compile_variants([None] * 9, 2)
# Each worker set up its session once and served several variants.
assert len(set(pids)) <= 2 and len(pids) == 9, pids
print(len(set(pids)))
"""


def test_exl3_variant_workers_persist_across_variants(tmp_path):
    # A variant stand-in that reports its worker pid: the real pool, spawn
    # context and initializer, without CUDA.
    shim = tmp_path / 'shim.py'
    shim.write_text('import os, time\n'
                    'def variant(payload):\n'
                    '    assert os.environ["CUTEAFD_EXPORT_WORKER_RAM_MIB"] == "4096"\n'
                    '    assert os.getsid(0) == os.getpid()\n'
                    '    time.sleep(0.05)\n'
                    '    return os.getpid()\n')
    child = _POOL_CHILD.replace('import package_exl3_aot as package\n',
                                'import package_exl3_aot as package\nimport shim\n'
                                'package.compile_variant = shim.variant\n')
    env = {**os.environ, 'PYTHONPATH': str(tmp_path)}
    result = subprocess.run([sys.executable, '-c', child, str(ROOT / 'python/tools/aot')], env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert int(result.stdout.strip()) in (1, 2)
    text = (ROOT / 'python/tools/aot/package_exl3_aot.py').read_text()
    assert 'max_tasks_per_child' not in text
    assert 'initializer=init_variant_worker' in text


def test_spark_native_builds_brake_ninja_on_load():
    host_isa = ROOT / 'scripts/build/host-isa.sh'
    for machine, expected in (('aarch64', ['-l', '10']), ('x86_64', [])):
        out = subprocess.run(['bash', '-c', f'uname() {{ echo {machine}; }}; source "$1"; cuteafd_performance_cores() {{ echo 10; }}; cuteafd_native_build_args',
                              'host-isa', str(host_isa)], capture_output=True, text=True, check=True).stdout
        assert out.split() == expected
    for script, root in (('build-wip-artifacts.sh', 'build_dir'), ('build-release-artifacts.sh', 'build_root')):
        text = (ROOT / 'scripts/build' / script).read_text()
        assert f'cmake --build "${root}/native" -- "${{native_build_args[@]}}"' in text
        assert text.index('cuteafd_native_build_args') < text.index('cmake --build')


# LAN-only seeds: counting CPU leases, one GPU export at a time per seed.

def _lease(tmp_path, name, *args, ssh_ok=True):
    """A real build-resource.py lease with stubbed ssh (probe: 20 cores, 121 GiB)."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    ssh = bin_dir / 'ssh'
    if not ssh.exists():
        ssh.write_text('#!/bin/sh\n[ -e "$SSH_DOWN_MARK" ] && case "$*" in *cassowary*) exit 255;; esac\n'
                       'case "$*" in *nproc*) printf "20\\n123904\\n";; esac\nexit 0\n')
        ssh.chmod(0o755)
    lease = tmp_path / name
    lease.mkdir()
    os.mkfifo(lease / 'release')
    owner = subprocess.Popen(['sleep', '60'])
    env = {**os.environ, 'PATH': f'{bin_dir}:{os.environ["PATH"]}', 'CUTEAFD_LOCK_DIR': str(tmp_path / 'locks'),
           'SSH_DOWN_MARK': str(tmp_path / ('ok' if ssh_ok else 'ssh-down'))}
    if not ssh_ok:
        (tmp_path / 'ssh-down').touch()
    process = subprocess.Popen(['python3', str(ROOT / 'scripts/build/build-resource.py'), '--owner', str(owner.pid),
                                '--wait', '20', '--lease-dir', str(lease), *args], env=env)
    return lease, process, owner


def _wait_file(path, timeout=10):
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert time.monotonic() < deadline, f'{path.name} never appeared'
        time.sleep(0.02)


def _end(lease, process, owner):
    try:
        with (lease / 'release').open('w') as handle:
            handle.write('x')
        assert process.wait(timeout=10) == 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        owner.terminate()
        owner.wait(timeout=5)


@pytest.fixture
def locks(tmp_path):
    (tmp_path / 'locks').mkdir()
    return tmp_path / 'locks'


def test_seed_export_progresses_with_an_older_queued_serving_ticket(tmp_path, locks):
    lease = _lease(tmp_path, 'build', '--host', 'kakapo')
    serving = None
    try:
        _wait_file(lease[0] / 'granted')
        # The older serving ticket queues while the build holds the kakapo.lock
        # umbrella. On export-request the build drops the umbrella, gets its
        # disjoint kakapo-export turn without waiting behind the ticket, and
        # keeps holding its counting slot while the umbrella is gone.
        marker = tmp_path / 'served'
        serving = subprocess.Popen([sys.executable, str(ROOT / 'scripts/hw/hwlock.py'),
                                    '--locks', 'sparks,kakapo', '--wait', '20', '--',
                                    'touch', str(marker)],
                                   env={**os.environ, 'CUTEAFD_LOCK_DIR': str(locks)})
        deadline = time.monotonic() + 5
        while not list((locks / 'hwq').glob('*.ticket')):
            assert serving.poll() is None and time.monotonic() < deadline
            time.sleep(0.02)
        (lease[0] / 'export-request').touch()
        _wait_file(lease[0] / 'export-granted', timeout=5)
        assert serving.wait(timeout=5) == 0 and marker.exists()
        probe = resource.try_lock(locks / 'kakapo.lock')
        assert probe is not None
        probe.close()
        assert resource.try_lock(locks / 'kakapo.build.0.lock') is None
        (lease[0] / 'export-done').touch()
        _end(*lease)
        lease = None
    finally:
        if lease:
            _end(*lease)
        if serving and serving.poll() is None:
            serving.terminate()
            serving.wait(timeout=5)


def test_seed_export_releases_host_umbrella_before_queueing_and_avoids_deadlock(tmp_path, locks, monkeypatch):
    # Direct sequence check: the host umbrella is released before hwlock.hold is
    # entered for SEED-export, while the counting slot stays held.
    import contextlib
    import types
    handles = resource.acquire(locks, 'cassowary', 3)
    assert handles is not None and len(handles) == 2
    assert resource.try_lock(locks / 'cassowary.lock') is None
    assert resource.try_lock(locks / 'cassowary.build.0.lock') is None
    real_hwlock = resource._hwlock()
    events = []

    @contextlib.contextmanager
    def checked_hold(names, wait, label='', **kwargs):
        probe = resource.try_lock(locks / 'cassowary.lock')
        assert probe is not None, 'cassowary.lock still held when queueing for cassowary-export'
        probe.close()
        assert resource.try_lock(locks / 'cassowary.build.0.lock') is None
        events.append(list(names))
        real_hwlock.LOCK_DIR = locks
        with real_hwlock.hold(names, wait, label, **kwargs) as held:
            yield held

    stub = types.SimpleNamespace(LOCK_DIR=locks, Cancelled=real_hwlock.Cancelled, hold=checked_hold)
    monkeypatch.setattr(resource, '_hwlock', lambda: stub)
    lease_dir = tmp_path / 'unit-lease'
    lease_dir.mkdir()
    stack = contextlib.ExitStack()
    try:
        state = resource.take_export_turn(stack, locks, 'cassowary',
                                          types.SimpleNamespace(wait=5, lease_dir=lease_dir),
                                          lambda: False, handles)
        assert state == 'held' and events == [['cassowary-export']]
        assert len(handles) == 1
        assert resource.try_lock(locks / 'cassowary-export.lock') is None
    finally:
        stack.close()
        for handle in handles:
            handle.close()

    # End-to-end #1038 reproduction: a fair-queued job wanting both cassowary and
    # cassowary-export queues while a build holds cassowary's CPU lease; when the
    # build then requests cassowary-export, releasing cassowary.lock before queueing
    # lets both complete without deadlocking.
    lease = _lease(tmp_path, 'build-both', '--host', 'cassowary')
    queued = None
    try:
        _wait_file(lease[0] / 'granted')
        marker = tmp_path / 'both-done'
        queued = subprocess.Popen([sys.executable, str(ROOT / 'scripts/hw/hwlock.py'),
                                   '--locks', 'cassowary,cassowary-export', '--wait', '20', '--',
                                   'touch', str(marker)],
                                  env={**os.environ, 'CUTEAFD_LOCK_DIR': str(locks)})
        deadline = time.monotonic() + 5
        while not list((locks / 'hwq').glob('*.ticket')):
            assert queued.poll() is None and time.monotonic() < deadline
            time.sleep(0.02)
        (lease[0] / 'export-request').touch()
        assert queued.wait(timeout=10) == 0 and marker.exists()
        _wait_file(lease[0] / 'export-granted', timeout=10)
        (lease[0] / 'export-done').touch()
    finally:
        _end(*lease)
        if queued and queued.poll() is None:
            queued.terminate()
            queued.wait(timeout=5)


def test_two_builds_on_one_seed_share_cpu_slots_but_serialize_the_export(tmp_path, locks):
    first = _lease(tmp_path, 'a', '--host', 'kakapo')
    second = _lease(tmp_path, 'b', '--host', 'kakapo')
    try:
        for lease, _, _ in (first, second):
            _wait_file(lease / 'granted')
            assert json.loads((lease / 'granted').read_text()) == {'host': 'kakapo', 'slots': 3}
        # Both hold a counting slot at once: their Cargo phases overlap.
        assert resource.held_slots(locks, 'kakapo', 3) == 2
        (first[0] / 'export-request').touch()
        _wait_file(first[0] / 'export-granted')
        assert resource.try_lock(locks / 'kakapo-export.lock') is None
        (second[0] / 'export-request').touch()
        time.sleep(1.0)
        assert not (second[0] / 'export-granted').exists(), 'two exports on one GB10 at once'
        (first[0] / 'export-done').touch()
        _wait_file(second[0] / 'export-granted')
        # The first build's CPU slot outlives its export turn.
        assert resource.held_slots(locks, 'kakapo', 3) == 2
        (second[0] / 'export-done').touch()
    finally:
        _end(*first)
        _end(*second)
    handle = resource.try_lock(locks / 'kakapo-export.lock')
    assert handle
    handle.close()


def test_export_turns_on_different_seeds_do_not_wait_for_each_other(tmp_path, locks):
    first = _lease(tmp_path, 'a', '--host', 'kakapo')
    second = _lease(tmp_path, 'b', '--host', 'cassowary')
    try:
        for lease, _, _ in (first, second):
            _wait_file(lease / 'granted')
            (lease / 'export-request').touch()
        for lease, _, _ in (first, second):
            _wait_file(lease / 'export-granted')
    finally:
        _end(*first)
        _end(*second)


def test_lease_teardown_stops_the_container_before_releasing_the_export_turn(tmp_path, locks):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    marker = tmp_path / 'stopped'
    (bin_dir / 'ssh').write_text(f'#!/bin/sh\ncase "$*" in *docker*) sleep 0.5; touch {marker};; '
                                 '*nproc*) printf "20\\n123904\\n";; esac\nexit 0\n')
    (bin_dir / 'ssh').chmod(0o755)
    lease, process, owner = _lease(tmp_path, 'a', '--host', 'kakapo', '--container', 'owned')
    try:
        _wait_file(lease / 'granted')
        (lease / 'active').touch()
        (lease / 'export-request').touch()
        _wait_file(lease / 'export-granted')
        owner.terminate()  # wip.sh died mid-export
        owner.wait(timeout=5)
        deadline = time.monotonic() + 5
        while not marker.exists():
            assert resource.try_lock(locks / 'kakapo-export.lock') is None, 'export turn freed before the stop'
            assert time.monotonic() < deadline
            time.sleep(0.02)
        assert process.wait(timeout=10) == 0
        handle = resource.try_lock(locks / 'kakapo-export.lock')
        assert handle
        handle.close()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def test_exclusive_seed_lease_grants_the_export_at_once(tmp_path, locks):
    # rhea and moa leases are exclusive already: no second export can be there.
    lease, process, owner = _lease(tmp_path, 'a', '--host', 'rhea', '--slots', '1')
    try:
        _wait_file(lease / 'granted')
        (lease / 'export-request').touch()
        _wait_file(lease / 'export-granted')
        assert not (locks / 'rhea-export.lock').exists()
    finally:
        _end(lease, process, owner)


def test_balance_picks_the_less_loaded_lan_seed(tmp_path, locks):
    held = resource.acquire(locks, 'kakapo', 3)
    try:
        lease, process, owner = _lease(tmp_path, 'a', '--host', 'kakapo', '--balance')
        try:
            _wait_file(lease / 'granted')
            assert json.loads((lease / 'granted').read_text())['host'] == 'cassowary'
        finally:
            _end(lease, process, owner)
    finally:
        for handle in held:
            handle.close()
    # A tie keeps the named seed.
    lease, process, owner = _lease(tmp_path, 'b', '--host', 'kakapo', '--balance')
    try:
        _wait_file(lease / 'granted')
        assert json.loads((lease / 'granted').read_text())['host'] == 'kakapo'
    finally:
        _end(lease, process, owner)


def test_two_balanced_builds_arriving_together_land_on_both_seeds(tmp_path, locks):
    leases = [_lease(tmp_path, name, '--host', 'kakapo', '--balance') for name in ('a', 'b')]
    try:
        for lease, _, _ in leases:
            _wait_file(lease / 'granted')
        hosts = sorted(json.loads((lease / 'granted').read_text())['host'] for lease, _, _ in leases)
        assert hosts == ['cassowary', 'kakapo']
    finally:
        for lease in leases:
            _end(*lease)


def test_balance_skips_an_unreachable_seed_and_falls_back_when_all_are_full(tmp_path, locks):
    lease, process, owner = _lease(tmp_path, 'a', '--host', 'cassowary', '--balance', ssh_ok=False)
    try:
        _wait_file(lease / 'granted')
        assert json.loads((lease / 'granted').read_text())['host'] == 'kakapo'
    finally:
        _end(lease, process, owner)
    (tmp_path / 'ssh-down').unlink()
    held = [resource.acquire(locks, 'kakapo', 1), resource.acquire(locks, 'cassowary', 1)]
    rhea = resource.try_lock(locks / 'rhea.lock')
    try:
        lease, process, owner = _lease(tmp_path, 'b', '--host', 'kakapo', '--balance', '--slots', '1',
                                       '--fallback', 'rhea,moa')
        try:
            _wait_file(lease / 'granted')
            assert json.loads((lease / 'granted').read_text())['host'] == 'moa'
        finally:
            _end(lease, process, owner)
    finally:
        rhea.close()
        for lease_handles in held:
            for handle in lease_handles:
                handle.close()


def test_pick_names_the_balanced_seed_without_holding_anything(tmp_path, locks, monkeypatch):
    held = resource.acquire(locks, 'kakapo', 3)
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    (bin_dir / 'ssh').write_text('#!/bin/sh\ncase "$*" in *nproc*) printf "20\\n123904\\n";; esac\n')
    (bin_dir / 'ssh').chmod(0o755)
    env = {**os.environ, 'PATH': f'{bin_dir}:{os.environ["PATH"]}', 'CUTEAFD_LOCK_DIR': str(locks)}
    try:
        result = subprocess.run(['python3', str(ROOT / 'scripts/build/build-resource.py'), '--pick', '--balance',
                                 '--host', 'kakapo', '--fallback', 'rhea,moa', '--lease-dir', '/nonexistent'],
                                env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0 and result.stdout.strip() == 'cassowary'
        assert resource.held_slots(locks, 'cassowary', 3) == 0
    finally:
        for handle in held:
            handle.close()
    # Every LAN seed unreachable: the first reachable fallback.
    (bin_dir / 'ssh').write_text('#!/bin/sh\ncase "$*" in *kakapo*|*cassowary*) exit 255;; *nproc*) printf "20\\n123904\\n";; esac\n')
    result = subprocess.run(['python3', str(ROOT / 'scripts/build/build-resource.py'), '--pick', '--balance',
                             '--host', 'kakapo', '--fallback', 'rhea,moa', '--lease-dir', '/nonexistent'],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and result.stdout.strip() == 'rhea'


def test_gb10_charge_uses_measured_host_footprint(tmp_path, monkeypatch):
    monkeypatch.setattr(export, '_context_lease', None)
    monkeypatch.setattr(export, '_gb10_host', lambda: True)
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_SLOT_DIR', str(tmp_path))
    monkeypatch.setenv('CUTEAFD_EXPORT_WORKER_RAM_MIB', '')
    monkeypatch.setenv('CUTEAFD_EXPORT_CONTEXT_SPACING_S', '0')
    monkeypatch.setattr(export, 'host_allocatable_memory', lambda: 120000 * 1024**2)
    (tmp_path / 'context-footprint-mib').write_text('7168')
    export.context_admission()
    try:
        assert (tmp_path / 'context.0.lock').read_text() == '7168'
    finally:
        export._context_lease.close()
        export._context_lease = None


def test_context_hook_initializes_stream_under_host_guard(tmp_path, monkeypatch):
    import types
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_SLOT_DIR', str(tmp_path))
    monkeypatch.setattr(export, '_gb10_host', lambda: True)
    events = []
    cuda = types.SimpleNamespace(_lazy_init=lambda: events.append('torch'))
    def stream():
        cuda._lazy_init()  # torch's recursive lazy-init path must not deadlock
        assert resource.try_lock(tmp_path / 'context-init.lock') is None
        events.append('stream')
        return types.SimpleNamespace(synchronize=lambda: events.append('sync'))
    cuda.current_stream = stream
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(export, 'context_admission', lambda: events.append('admit'))
    monkeypatch.setattr(export, 'gpu_admission', lambda: events.append('gpu'))
    monkeypatch.setattr(export, '_record_footprint', lambda *args: 'measured')
    callbacks = []
    monkeypatch.setattr(export.atexit, 'register', lambda callback: callbacks.append(callback))
    export.install_context_admission()
    try:
        cuda._lazy_init()
        assert events == ['admit', 'gpu', 'torch', 'torch', 'stream', 'sync']
        assert resource.try_lock(tmp_path / 'context-init.lock') is not None
        cuda._lazy_init()
        assert events.count('stream') == 1
    finally:
        for callback in callbacks:
            callback()


def test_measured_footprint_updates_shared_charge_and_live_lease(tmp_path, monkeypatch):
    import types
    monkeypatch.setattr(export.resource, 'getrusage', lambda kind: types.SimpleNamespace(ru_maxrss=3072 * 1024))
    monkeypatch.setattr(export, 'host_allocatable_memory', lambda: 110000 * 1024**2)
    cuda = types.SimpleNamespace(max_memory_allocated=lambda: 768 * 1024**2,
                                 max_memory_reserved=lambda: 1024 * 1024**2)
    lease = (tmp_path / 'context.0.lock').open('w+')
    lease.write('1536')
    lease.flush()
    monkeypatch.setattr(export, '_context_lease', lease)
    try:
        text = export._record_footprint(types.SimpleNamespace(cuda=cuda), tmp_path, 120000)
        assert 'rss_peak=3072 MiB' in text
        assert 'torch_allocated_peak=768 MiB' in text
        assert 'torch_reserved_peak=1024 MiB' in text
        assert (tmp_path / 'context-footprint-mib').read_text() == '4096'
        assert (tmp_path / 'context.0.lock').read_text() == '4096'
        assert 'init_dip=10000 MiB' in text
        (tmp_path / 'context-footprint-mib').write_text('8192')
        export._record_footprint(types.SimpleNamespace(cuda=cuda), tmp_path, 120000)
        assert (tmp_path / 'context-footprint-mib').read_text() == '8192'
    finally:
        lease.close()


def test_discrete_context_hook_preserves_existing_lazy_initialization(tmp_path, monkeypatch):
    import types
    events = []
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_SLOT_DIR', str(tmp_path))
    monkeypatch.setattr(export, '_gb10_host', lambda: False)
    cuda = types.SimpleNamespace(_lazy_init=lambda: events.append('torch'))
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(export, 'context_admission', lambda: events.append('admit'))
    monkeypatch.setattr(export, 'gpu_admission', lambda: events.append('gpu'))
    export.install_context_admission()
    cuda._lazy_init()
    assert events == ['admit', 'gpu', 'torch']
    assert not (tmp_path / 'context-init.lock').exists()


def test_standalone_gb10_installs_host_default_pool(tmp_path, monkeypatch):
    import types
    # Record the key so monkeypatch restores it: the code under test assigns it.
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_SLOT_DIR', '')
    monkeypatch.delenv('CUTEAFD_EXPORT_HOST_SLOT_DIR')
    monkeypatch.setattr(export, '_gb10_host', lambda: True)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    cuda = types.SimpleNamespace(_lazy_init=lambda: None)
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace(cuda=cuda))
    export.install_context_admission()
    assert os.environ['CUTEAFD_EXPORT_HOST_SLOT_DIR'] == str(tmp_path / '.cache/cuteafd/export-workers')
    assert getattr(cuda._lazy_init, '_cuteafd_context_admission')

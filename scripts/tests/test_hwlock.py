import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'hw' / 'hwlock.py'
spec = importlib.util.spec_from_file_location('hwlock', SCRIPT)
hwlock = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hwlock)


@pytest.fixture
def lockdir(tmp_path, monkeypatch):
    monkeypatch.setattr(hwlock, 'LOCK_DIR', tmp_path)
    monkeypatch.setattr(hwlock, 'POLL_S', 0.05)
    return tmp_path


def spawn(lockdir, locks, seconds, marker):
    env = {**os.environ, 'CUTEAFD_LOCK_DIR': str(lockdir)}
    cmd = [sys.executable, str(SCRIPT), '--locks', locks, '--wait', '30', '--',
           sys.executable, '-c', f'import time,pathlib; pathlib.Path({str(marker)!r}).write_text(str(time.time())); time.sleep({seconds})']
    return subprocess.Popen(cmd, env=env, stderr=subprocess.DEVNULL)


def wait_for(path, timeout=10):
    end = time.time() + timeout
    while not path.exists():
        assert time.time() < end, f'{path} never appeared'
        time.sleep(0.02)
    return float(path.read_text())


@pytest.mark.parametrize('gate', [False, True])
def test_preflight_fails_before_ticket(lockdir, gate):
    env = {**os.environ, 'CUTEAFD_LOCK_DIR': str(lockdir)}
    result = subprocess.run([sys.executable, str(SCRIPT), '--locks', 'gpu1',
                             '--preflight', 'exit 9', *(['--gate'] if gate else []),
                             '--', 'true'], env=env, capture_output=True, text=True)
    assert result.returncode == 78
    assert 'hwlock: preflight failed (exit 9)' in result.stderr
    assert not (lockdir / 'hwq').exists()


@pytest.mark.parametrize('gate', [False, True])
def test_preflight_at_grant_skips_the_enqueue_check(lockdir, gate):
    count = lockdir / 'checks'
    marker = lockdir / 'command-ran'
    code = (f'import pathlib; p=pathlib.Path({str(count)!r}); '
            'p.write_text(str((int(p.read_text()) if p.exists() else 0)+1))')
    import shlex
    env = {**os.environ, 'CUTEAFD_LOCK_DIR': str(lockdir)}
    result = subprocess.run([sys.executable, str(SCRIPT), '--locks', 'gpu1', '--preflight-at-grant',
                             '--preflight', shlex.join([sys.executable, '-c', code]),
                             *(['--gate'] if gate else []), '--', 'touch', str(marker)],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert count.read_text() == '1' and marker.exists()


@pytest.mark.parametrize('gate', [False, True])
def test_preflight_fails_at_grant_releasing_ticket_and_locks(lockdir, gate):
    count = lockdir / 'checks'
    marker = lockdir / 'command-ran'
    code = (f'import pathlib,sys; p=pathlib.Path({str(count)!r}); '
            'n=int(p.read_text()) if p.exists() else 0; p.write_text(str(n+1)); '
            'sys.exit(0 if n == 0 else 11)')
    import shlex
    preflight = shlex.join([sys.executable, '-c', code])
    env = {**os.environ, 'CUTEAFD_LOCK_DIR': str(lockdir)}
    result = subprocess.run([sys.executable, str(SCRIPT), '--locks', 'gpu0,gpu1',
                             '--preflight', preflight, *(['--gate'] if gate else []),
                             '--', 'touch', str(marker)], env=env, capture_output=True, text=True)
    assert result.returncode == 78
    assert 'hwlock: preflight failed (exit 11)' in result.stderr
    assert count.read_text() == '2' and not marker.exists()
    assert not list((lockdir / 'hwq').glob('*.ticket'))
    held = hwlock._try_flocks(['gpu0', 'gpu1'])
    assert held is not None
    for handle in held:
        handle.close()


@pytest.mark.parametrize('gate', [False, True])
def test_preflight_timeout_at_grant_kills_process_group_and_releases_locks(lockdir, gate):
    import shlex
    count, childpid = lockdir / 'checks', lockdir / 'child.pid'
    child_code = f'import pathlib,time,os; pathlib.Path({str(childpid)!r}).write_text(str(os.getpid())); time.sleep(60)'
    code = (f'import pathlib,subprocess,sys,time; p=pathlib.Path({str(count)!r}); '
            'n=int(p.read_text()) if p.exists() else 0; p.write_text(str(n+1)); '
            'sys.exit(0) if n == 0 else None; '
            f'subprocess.Popen([{sys.executable!r}, "-c", {child_code!r}]); time.sleep(60)')
    result = subprocess.run([sys.executable, str(SCRIPT), '--locks', 'gpu1',
                             '--preflight', shlex.join([sys.executable, '-c', code]),
                             '--preflight-timeout', '0.5', *(['--gate'] if gate else []),
                             '--', 'true'], env={**os.environ, 'CUTEAFD_LOCK_DIR': str(lockdir)},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 78 and 'hwlock: preflight timed out' in result.stderr
    assert count.read_text() == '2'
    assert not list((lockdir / 'hwq').glob('*.ticket'))
    held = hwlock._try_flocks(['gpu1'])
    assert held is not None
    for handle in held:
        handle.close()
    child = Path('/proc') / childpid.read_text()
    # An orphaned child can remain a zombie until the host reaps it, not running.
    if child.exists():
        assert '\nState:\tZ' in (child / 'status').read_text()


@pytest.mark.parametrize('exit_code', [0, 12])
def test_preflight_exit_kills_remaining_helpers_before_releasing_locks(lockdir, exit_code):
    import shlex
    count, childpid = lockdir / 'checks', lockdir / 'child.pid'
    code = (f'import pathlib,subprocess,sys; p=pathlib.Path({str(count)!r}); '
            'n=int(p.read_text()) if p.exists() else 0; p.write_text(str(n+1)); '
            'sys.exit(0) if n == 0 else None; '
            f'child=subprocess.Popen([{sys.executable!r}, "-c", "import time; time.sleep(60)"], '
            'stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); '
            f'pathlib.Path({str(childpid)!r}).write_text(str(child.pid)); sys.exit({exit_code})')
    result = subprocess.run([sys.executable, str(SCRIPT), '--locks', 'gpu1',
                             '--preflight', shlex.join([sys.executable, '-c', code]), '--', 'true'],
                            env={**os.environ, 'CUTEAFD_LOCK_DIR': str(lockdir)},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == (78 if exit_code else 0)
    if exit_code:
        assert 'preflight failed (exit 12)' in result.stderr
    assert not list((lockdir / 'hwq').glob('*.ticket'))
    child = Path('/proc') / childpid.read_text()
    if child.exists():
        assert '\nState:\tZ' in (child / 'status').read_text()
    held = hwlock._try_flocks(['gpu1'])
    assert held is not None
    for handle in held:
        handle.close()


def test_canonical_order_and_unknown_locks():
    assert hwlock.canonical(['gpu1', 'sparks.lock', 'gpu0']) == ['sparks', 'gpu0', 'gpu1']
    with pytest.raises(ValueError):
        hwlock.canonical(['gpu2'])


def test_lan_seed_locks_are_known_and_ordered_after_serving_then_export_turns():
    # The LAN-only seeds' leases sort after every serving lock, and each seed's
    # GPU-export turn after every lease: a build holds its lease when it queues
    # for the export, so the turn must come later in the one canonical order.
    assert hwlock.LAN_SEEDS == ('kakapo', 'cassowary')
    assert hwlock.ORDER[-5:] == ('kakapo', 'cassowary', 'kakapo-export', 'cassowary-export', 'oliver')
    assert hwlock.canonical(['cassowary-export', 'kakapo', 'moa', 'sparks', 'gpu1', 'cassowary', 'rhea', 'gpu0']) == \
        ['sparks', 'gpu0', 'gpu1', 'rhea', 'moa', 'kakapo', 'cassowary', 'cassowary-export']
    assert hwlock.canonical(['oliver', 'gpu0']) == ['gpu0', 'oliver']


def test_hold_stops_waiting_when_cancelled(lockdir, monkeypatch):
    monkeypatch.setattr(hwlock, 'LOCK_DIR', lockdir)
    busy = spawn(lockdir, 'kakapo-export', 5.0, lockdir / 'busy')
    try:
        wait_for(lockdir / 'busy')
        calls = []
        with pytest.raises(hwlock.Cancelled):
            with hwlock.hold(['kakapo-export'], 30, cancelled=lambda: calls.append(1) or len(calls) > 1):
                pytest.fail('a cancelled waiter never holds the lock')
        assert not list((lockdir / 'hwq').glob('*.ticket')) or all(
            'kakapo-export' not in p.read_text() or str(busy.pid) in p.name for p in (lockdir / 'hwq').glob('*.ticket'))
    finally:
        busy.wait(timeout=10)


def test_older_dual_gpu_waiter_is_not_overtaken_by_later_single_gpu_job(lockdir):
    # gpu1 is busy; a dual-GPU job queues; a later gpu1-only job must wait behind it.
    busy = spawn(lockdir, 'gpu1', 1.0, lockdir / 'busy')
    wait_for(lockdir / 'busy')
    dual = spawn(lockdir, 'gpu0,gpu1', 0.3, lockdir / 'dual')
    time.sleep(0.3)
    single = spawn(lockdir, 'gpu1', 0.1, lockdir / 'single')
    for proc in (busy, dual, single):
        assert proc.wait(timeout=30) == 0
    assert wait_for(lockdir / 'dual') < wait_for(lockdir / 'single')


def test_disjoint_locks_run_in_parallel_past_a_waiter(lockdir):
    busy = spawn(lockdir, 'gpu1', 1.0, lockdir / 'busy')
    wait_for(lockdir / 'busy')
    dual = spawn(lockdir, 'gpu0,gpu1', 0.1, lockdir / 'dual')
    time.sleep(0.3)
    sparks = spawn(lockdir, 'sparks', 0.1, lockdir / 'sparks')
    started = wait_for(lockdir / 'sparks')
    assert not (lockdir / 'dual').exists() or started < wait_for(lockdir / 'dual')
    for proc in (busy, dual, sparks):
        assert proc.wait(timeout=30) == 0


def test_stale_ticket_from_dead_process_is_ignored_and_removed(lockdir):
    queue = lockdir / 'hwq'
    queue.mkdir()
    stale = queue / '000000000007.99999.ticket'
    stale.write_text('{"locks": ["gpu1"]}')  # nobody holds its flock
    with hwlock.hold(['gpu1'], wait=5):
        pass
    assert not stale.exists()


def test_timeout_exits_75_without_holding_anything(lockdir):
    busy = spawn(lockdir, 'gpu1', 6.0, lockdir / 'busy')
    wait_for(lockdir / 'busy')
    env = {**os.environ, 'CUTEAFD_LOCK_DIR': str(lockdir)}
    result = subprocess.run([sys.executable, str(SCRIPT), '--locks', 'gpu1', '--wait', '0.3', '--', 'true'],
                            env=env, capture_output=True)
    assert result.returncode == 75
    assert busy.wait(timeout=30) == 0
    assert not list((lockdir / 'hwq').glob('*.ticket'))


def test_hold_mode_exports_held_locks_and_gate_mode_leaves_them_free(lockdir):
    env = {**os.environ, 'CUTEAFD_LOCK_DIR': str(lockdir)}
    shown = subprocess.run([sys.executable, str(SCRIPT), '--locks', 'gpu1,gpu0', '--', 'sh', '-c', 'echo $CUTEAFD_HW_LOCKS_HELD'],
                           env=env, capture_output=True, text=True)
    assert shown.stdout.strip() == 'gpu0,gpu1'
    # --gate: the child can take the flock itself (no deadlock on our own parent).
    child = f'import fcntl; f=open({str(lockdir / "gpu1.lock")!r},"a"); fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)'
    gated = subprocess.run([sys.executable, str(SCRIPT), '--locks', 'gpu1', '--gate', '--', sys.executable, '-c', child],
                           env=env, capture_output=True, text=True)
    assert gated.returncode == 0, gated.stderr


def test_gate_ticket_keeps_later_waiters_behind_the_running_command(lockdir):
    env = {**os.environ, 'CUTEAFD_LOCK_DIR': str(lockdir)}
    first = subprocess.Popen([sys.executable, str(SCRIPT), '--locks', 'gpu1', '--gate', '--', sys.executable, '-c',
                              f'import time,pathlib; pathlib.Path({str(lockdir / "first")!r}).write_text(str(time.time())); time.sleep(1.0)'],
                             env=env, stderr=subprocess.DEVNULL)
    wait_for(lockdir / 'first')
    second = spawn(lockdir, 'gpu1', 0.05, lockdir / 'second')
    assert first.wait(timeout=30) == 0 and second.wait(timeout=30) == 0
    assert wait_for(lockdir / 'second') - wait_for(lockdir / 'first') >= 0.9


def test_command_inside_a_gate_is_not_blocked_by_its_own_gate_ticket(lockdir):
    env = {**os.environ, 'CUTEAFD_LOCK_DIR': str(lockdir)}
    # The gated command queues for the same locks itself (as `cuteafd bench smoke` does).
    inner = f'{sys.executable} {SCRIPT} --locks sparks,gpu1 --wait 5 -- true'
    result = subprocess.run([sys.executable, str(SCRIPT), '--locks', 'sparks,gpu0,gpu1', '--gate', '--wait', '5',
                             '--', 'sh', '-c', inner], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_gate_exemption_covers_only_ancestors(lockdir):
    queue = lockdir / 'hwq'
    queue.mkdir()
    with hwlock.hold(['gpu1'], wait=5, gate=True):
        gate = hwlock.GATE_SEQ[0]
        theirs = [p.name for p in queue.glob('*.ticket')]
        assert hwlock.older_conflicts(queue, ['gpu1'], None) == theirs
        os.environ[hwlock.GATE_ENV] = str(gate)
        try:
            assert hwlock.older_conflicts(queue, ['gpu1'], None) == []
        finally:
            del os.environ[hwlock.GATE_ENV]


def test_gated_child_is_not_deadlocked_by_a_younger_waiter_behind_its_gate(lockdir):
    # 2026-10-12: gate 74 (sparks,gpu0) held the turn; a 3-lock job 76 queued behind it;
    # the gate's smoke child then took ticket 77 (younger than 76) and waited on 76 forever.
    env = {**os.environ, 'CUTEAFD_LOCK_DIR': str(lockdir)}
    marker = lockdir / 'child-ran'
    inner = (f'{sys.executable} -c "import time; time.sleep(0.6)"; '
             f'{sys.executable} {SCRIPT} --locks sparks,gpu0 --wait 10 -- '
             f'{sys.executable} -c "import pathlib; pathlib.Path({str(marker)!r}).write_text(\'1\')"')
    gate = subprocess.Popen([sys.executable, str(SCRIPT), '--locks', 'sparks,gpu0', '--gate', '--wait', '10',
                             '--', 'sh', '-c', inner], env=env, stderr=subprocess.PIPE, text=True)
    time.sleep(0.3)
    blocked = spawn(lockdir, 'sparks,gpu0,gpu1', 0.1, lockdir / 'three')  # queues behind the gate
    assert gate.wait(timeout=30) == 0, gate.stderr.read()
    assert marker.exists()
    assert blocked.wait(timeout=30) == 0


# A command that tears down on SIGTERM (as `cuteafd bench smoke` does): it marks
# the start and end of its teardown, then exits 0.
TEARDOWN_CHILD = '''import pathlib, signal, sys, time
d = pathlib.Path(sys.argv[1])
def stop(*_):
    if (d / "teardown").exists():
        return  # repeats (systemd's cgroup-wide TERM plus hwlock's forward) change nothing
    (d / "teardown").write_text("")
    time.sleep(1.0)
    (d / "torn-down").write_text("")
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
(d / "ready").write_text("")
while True:
    time.sleep(0.05)
'''


def held(lockdir, name):
    import fcntl
    with open(lockdir / f'{name}.lock', 'a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


@pytest.mark.parametrize('cgroup_wide', [False, True])
def test_sigterm_is_forwarded_and_locks_outlive_the_commands_teardown(lockdir, cgroup_wide):
    env = {**os.environ, 'CUTEAFD_LOCK_DIR': str(lockdir)}
    proc = subprocess.Popen([sys.executable, str(SCRIPT), '--locks', 'gpu1', '--', sys.executable, '-c', TEARDOWN_CHILD, str(lockdir)],
                            env=env, stderr=subprocess.PIPE, text=True, start_new_session=True)
    end = time.time() + 10
    while not (lockdir / 'ready').exists():
        assert time.time() < end and proc.poll() is None
        time.sleep(0.02)
    assert held(lockdir, 'gpu1')
    # systemd's default KillMode=control-group signals every process at once;
    # a plain `kill PID` reaches hwlock alone, which must forward it.
    if cgroup_wide:
        os.killpg(proc.pid, 15)
    else:
        proc.send_signal(15)
    end = time.time() + 10
    while not (lockdir / 'teardown').exists():
        assert time.time() < end, 'the command never saw the signal'
        time.sleep(0.02)
    # Mid-teardown: hwlock is still alive and still holds the lock.
    time.sleep(0.3)
    assert proc.poll() is None
    assert not (lockdir / 'torn-down').exists() and held(lockdir, 'gpu1')
    assert proc.wait(timeout=30) == 0
    assert (lockdir / 'torn-down').exists()
    assert not held(lockdir, 'gpu1')
    assert 'forwarded' in proc.stderr.read()


def test_a_command_that_ignores_the_signal_is_killed_after_the_grace(tmp_path):
    import signal
    import threading
    ready = tmp_path / 'ready'
    cmd = [sys.executable, '-c', f'import pathlib,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); '
           f'pathlib.Path({str(ready)!r}).write_text(""); time.sleep(60)']
    def stop():
        end = time.time() + 10
        while not ready.exists() and time.time() < end:
            time.sleep(0.02)
        os.kill(os.getpid(), signal.SIGTERM)
    before = signal.getsignal(signal.SIGTERM)
    threading.Thread(target=stop, daemon=True).start()
    started = time.time()
    assert hwlock.run_forwarding(cmd, dict(os.environ), grace=0.5) == 128 + signal.SIGKILL
    assert time.time() - started < 10
    assert signal.getsignal(signal.SIGTERM) is before


def test_stop_grace_fits_systemd_default_and_exceeds_smokes_teardown_bound():
    rust = (Path(__file__).resolve().parents[2] / 'rust/crates/cuteafd-bench/src/smoke.rs').read_text()
    smoke = int(rust.split('const CANCEL_TEARDOWN_S: u64 = ', 1)[1].split(';', 1)[0])
    spec = importlib.util.spec_from_file_location('wip_cards', SCRIPT.parents[1] / 'bench/wip-cards.py')
    cards = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cards)
    assert smoke < cards.SMOKE_STOP_GRACE_S < hwlock.STOP_GRACE_S < 90

import importlib.util
import os
import re
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
    while True:
        # write_text creates the marker before publishing its timestamp.
        text = path.read_text() if path.exists() else ''
        if text:
            return float(text)
        assert time.time() < end, f'{path} never appeared'
        time.sleep(0.02)


def test_wait_for_retries_an_empty_marker(tmp_path, monkeypatch):
    marker = tmp_path / 'marker'
    marker.write_text('')
    monkeypatch.setattr(time, 'sleep', lambda _: marker.write_text('123.5'))
    assert wait_for(marker) == 123.5


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


def test_maxq_lock_uses_the_standard_flock_path(lockdir):
    assert hwlock.canonical(['maxq.lock']) == ['maxq']
    handles = hwlock._try_flocks(['maxq'])
    assert handles is not None
    try:
        assert (lockdir / 'maxq.lock').is_file()
        assert hwlock._try_flocks(['maxq']) is None
    finally:
        for handle in handles:
            handle.close()


def test_lan_seed_locks_are_known_and_ordered_after_serving():
    # The LAN-only seeds' leases sort after every serving lock.
    assert hwlock.LAN_SEEDS == ('kakapo', 'cassowary')
    assert hwlock.ORDER[-3:] == ('kakapo', 'cassowary', 'maxq')
    assert hwlock.canonical(['kakapo', 'moa', 'sparks', 'gpu1', 'cassowary', 'rhea', 'gpu0']) == \
        ['sparks', 'gpu0', 'gpu1', 'rhea', 'moa', 'kakapo', 'cassowary']
    assert hwlock.canonical(['maxq', 'gpu0']) == ['gpu0', 'maxq']


def test_hold_stops_waiting_when_cancelled(lockdir, monkeypatch):
    monkeypatch.setattr(hwlock, 'LOCK_DIR', lockdir)
    busy = spawn(lockdir, 'kakapo', 5.0, lockdir / 'busy')
    try:
        wait_for(lockdir / 'busy')
        calls = []
        with pytest.raises(hwlock.Cancelled):
            with hwlock.hold(['kakapo'], 30, cancelled=lambda: calls.append(1) or len(calls) > 1):
                pytest.fail('a cancelled waiter never holds the lock')
        assert not list((lockdir / 'hwq').glob('*.ticket')) or all(
            'kakapo' not in p.read_text() or str(busy.pid) in p.name for p in (lockdir / 'hwq').glob('*.ticket'))
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


REAL_LOCK_DIR = Path.home() / '.cache' / 'cuteafd'


def test_session_lock_directory_never_reaches_the_real_cache(tmp_path_factory):
    """#1048: with no per-test patch, taking gpu1 lands in the session's private
    directory, never in ~/.cache/cuteafd."""
    lock_dir = hwlock.lock_dir()
    assert lock_dir == Path(os.environ['CUTEAFD_LOCK_DIR'])
    # pytest's basetemp may itself live under ~/.cache/cuteafd/builds; what matters
    # is that this is not the hardware lock directory or its queue.
    assert lock_dir.is_relative_to(tmp_path_factory.getbasetemp())
    assert lock_dir.parent != REAL_LOCK_DIR
    real_lock = REAL_LOCK_DIR / 'gpu1.lock'
    before = real_lock.stat().st_mtime_ns if real_lock.exists() else None
    with hwlock.hold(['gpu1'], wait=10) as handles:
        assert len(handles) == 1
        assert (lock_dir / 'gpu1.lock').exists()
    after = real_lock.stat().st_mtime_ns if real_lock.exists() else None
    assert after == before
    assert not list((lock_dir / 'hwq').glob('*.ticket'))


def test_the_guard_refuses_the_real_lock_directory():
    """The session fixtures route every lock to tmp; a code path that reaches the
    real directory anyway fails the test instead of taking a cluster lock."""
    with pytest.raises(pytest.fail.Exception):
        open(REAL_LOCK_DIR / 'gpu1.lock')
    with pytest.raises(pytest.fail.Exception):
        os.open(REAL_LOCK_DIR / 'hwq', os.O_RDONLY)


# --------------------------------------------------------------------------- bounded backfill

WHOLE = 'sparks,gpu0,gpu1,rhea,moa'


def spawn_bounded(lockdir, locks, seconds, marker, max_run=None, inner=None):
    """hwlock running a command that stamps `marker` with its start time, sleeps,
    then stamps `marker.end`; `inner` (a shell line) runs first instead of sleeping."""
    env = {**os.environ, 'CUTEAFD_LOCK_DIR': str(lockdir)}
    body = (f'import pathlib,subprocess,time; m=pathlib.Path({str(marker)!r}); m.write_text(str(time.time())); '
            + (f'subprocess.run({inner!r}, shell=True); ' if inner else f'time.sleep({seconds}); ')
            + 'pathlib.Path(str(m)+".end").write_text(str(time.time()))')
    cmd = [sys.executable, str(SCRIPT), '--locks', locks, '--wait', '60',
           *(['--max-run', str(max_run)] if max_run is not None else []), '--', sys.executable, '-c', body]
    return subprocess.Popen(cmd, env=env, stderr=subprocess.PIPE, text=True)


def end_of(marker):
    return wait_for(Path(str(marker) + '.end'), timeout=60)


def queued(lockdir, count, timeout=10):
    """Wait until `count` live tickets are queued (the spawned waiters are in line)."""
    end = time.time() + timeout
    while len(list((lockdir / 'hwq').glob('*.ticket'))) < count:
        assert time.time() < end, 'waiters never queued'
        time.sleep(0.02)


def test_backfill_rule_matrix(lockdir):
    """backfill_check on hand-made queue entries: the holder's recorded bound,
    the candidate's own bound, unbounded and out-of-queue holders."""
    queue = hwlock._queue()
    now = time.time()
    keep = []
    ticket = lambda seq, locks: keep.append(hwlock._publish(queue, seq, 'ticket', dict(locks=locks))[1])
    ticket(5, ['sparks', 'gpu0', 'gpu1', 'rhea', 'moa'])  # B, waiting
    flocks = hwlock._try_flocks(['sparks', 'gpu1'])  # A's locks
    path, handle = hwlock._publish(queue, 1, 'held', dict(seq=1, locks=['sparks', 'gpu1'],
                                                          release_by=now + 100 + hwlock.HARD_STOP_S))
    try:
        assert hwlock.backfill_check(queue, ['gpu0'], 9, 50, now)[0]
        assert hwlock.backfill_check(queue, ['gpu0'], 9, 100, now)[0]
        allowed, reason = hwlock.backfill_check(queue, ['gpu0'], 9, 101, now)
        assert not allowed and 'after ticket 5' in reason
        # Two recorded blockers: the earlier deadline decides.
        other, other_handle = hwlock._publish(queue, 2, 'held', dict(seq=2, locks=['rhea'],
                                                                     release_by=now + 40 + hwlock.HARD_STOP_S))
        rhea = hwlock._try_flocks(['rhea'])
        assert hwlock.backfill_check(queue, ['gpu0'], 9, 40, now)[0]
        assert not hwlock.backfill_check(queue, ['gpu0'], 9, 41, now)[0]
        for item in [other_handle, *rhea]:
            item.close()
        other.unlink()
        assert not hwlock.backfill_check(queue, ['gpu1'], 9, 10, now)[0]  # its own lock is held
        # A ticket older than B that wants only free locks could start now: no jump.
        path3, handle3 = hwlock._publish(queue, 3, 'ticket', dict(locks=['gpu0']))
        allowed, reason = hwlock.backfill_check(queue, ['gpu0'], 9, 10, now)
        assert not allowed and 'ticket 3 could start now' in reason
        handle3.close()
        path3.unlink()
    finally:
        handle.close()
        path.unlink()
    # A granted gate's ticket holds its turn: never jumped.
    path, handle = hwlock._publish(queue, 5, 'held', dict(seq=5, locks=['sparks', 'gpu0', 'gpu1', 'rhea', 'moa'],
                                                          gate=True, release_by=now + 1000))
    try:
        allowed, reason = hwlock.backfill_check(queue, ['kakapo'], 9, 10, now)
        assert allowed  # disjoint from every older ticket
        allowed, reason = hwlock.backfill_check(queue, ['gpu0'], 9, 10, now)
        assert not allowed and 'gpu0 is held' in reason
    finally:
        handle.close()
        path.unlink()
    # A holder without a bound may release any moment.
    path, handle = hwlock._publish(queue, 1, 'held', dict(seq=1, locks=['sparks', 'gpu1'], release_by=None))
    try:
        allowed, reason = hwlock.backfill_check(queue, ['gpu0', 'moa'], 9, 10, now)
        assert not allowed and 'held without a bound' in reason
    finally:
        handle.close()
        path.unlink()
    # No record at all: the flocks are held outside the queue (plain flock, smoke).
    allowed, reason = hwlock.backfill_check(queue, ['moa'], 9, 10, now)
    assert not allowed and 'outside the queue' in reason
    for item in [*flocks, *keep]:
        item.close()


def test_backfill_transitive_convoy(lockdir):
    # The live 2026-11-26 queue: GPU0 blocks the full-cluster head, which
    # blocks a long GPU1+Spark job even though its locks are all free.
    queue = hwlock._queue()
    now = time.time()
    keep = hwlock._try_flocks(['gpu0'])

    def publish(seq, suffix, **data):
        path, handle = hwlock._publish(queue, seq, suffix, data)
        keep.append(handle)
        return path, handle

    publish(1, 'held', locks=['gpu0'], release_by=now + 6900)
    publish(2, 'ticket', locks=WHOLE.split(','), max_run=6375)
    long = ['sparks', 'gpu1', 'rhea', 'moa']
    publish(3, 'ticket', locks=long, max_run=22350)
    publish(4, 'ticket', locks=['gpu1'], max_run=18150)
    try:
        assert not hwlock.backfill_check(queue, long, 3, 22350, now)[0]
        assert not hwlock.backfill_check(queue, ['gpu1'], 4, 18150, now)[0]
        for seq, seconds in enumerate((300, 720, 2880, 5400), 5):
            path, handle = publish(seq, 'ticket', locks=['gpu1'], max_run=seconds)
            assert hwlock.backfill_check(queue, ['gpu1'], seq, seconds, now)[0]
            # Preserve FIFO among eligible jobs. Later short ones can run
            # after the first short ticket leaves, not overtake it.
            allowed, reason = hwlock.backfill_check(queue, ['gpu1'], seq + 1, 100, now)
            assert not allowed and f'ticket {seq} could start now' in reason
            handle.close()
            path.unlink()
        assert hwlock.backfill_check(queue, ['gpu1'], 10, 6900 - hwlock.HARD_STOP_S, now)[0]
        assert not hwlock.backfill_check(queue, ['gpu1'], 10, 6901 - hwlock.HARD_STOP_S, now)[0]
    finally:
        for handle in keep:
            handle.close()


@pytest.mark.parametrize('holder', ['bounded', 'unbounded', 'external'])
def test_backfill_transitive_disjoint_head(lockdir, holder):
    # The candidate does not overlap the head directly. The intermediate
    # ticket carries the head's deadline (or lack thereof) across the chain.
    queue = hwlock._queue()
    now = time.time()
    keep = hwlock._try_flocks(['gpu0'])
    if holder != 'external':
        keep.append(hwlock._publish(queue, 1, 'held', dict(
            locks=['gpu0'], release_by=now + 1000 if holder == 'bounded' else None))[1])
    keep.append(hwlock._publish(queue, 2, 'ticket', dict(locks=['gpu0', 'sparks']))[1])
    keep.append(hwlock._publish(queue, 3, 'ticket', dict(locks=['sparks', 'gpu1'], max_run=2000))[1])
    keep.append(hwlock._publish(queue, 4, 'ticket', dict(locks=['gpu1', 'rhea'], max_run=2000))[1])
    try:
        assert hwlock.backfill_check(queue, ['rhea'], 5, 100, now)[0] == (holder == 'bounded')
        assert not hwlock.backfill_check(queue, ['rhea'], 5, 1000, now)[0]
    finally:
        for handle in keep:
            handle.close()


@pytest.mark.parametrize('metadata,allowed', [({'gate': True, 'max_run': 100}, True), ({}, True),
                                             ({'max_run': 100}, False)])
def test_backfill_older_ticket_eligibility(lockdir, metadata, allowed):
    queue = hwlock._queue()
    now = time.time()
    keep = hwlock._try_flocks(['gpu0'])
    keep.append(hwlock._publish(queue, 1, 'held', dict(locks=['gpu0'], release_by=now + 1000))[1])
    keep.append(hwlock._publish(queue, 2, 'ticket', dict(locks=['gpu0', 'sparks']))[1])
    keep.append(hwlock._publish(queue, 3, 'ticket', dict(locks=['sparks', 'gpu1'], **metadata))[1])
    try:
        assert hwlock.backfill_check(queue, ['gpu1'], 4, 100, now)[0] == allowed
    finally:
        for handle in keep:
            handle.close()


def test_backfill_respects_gate_ancestor_exemption(lockdir, monkeypatch):
    queue = hwlock._queue()
    now = time.time()
    path, handle = hwlock._publish(queue, 1, 'ticket', dict(locks=['gpu1']))
    try:
        assert not hwlock.backfill_check(queue, ['gpu1'], 2, 100, now)[0]
        monkeypatch.setenv(hwlock.GATE_ENV, '1')
        assert hwlock.backfill_check(queue, ['gpu1'], 2, 100, now)[0]
        # Our ancestor exemption must not be used to infer another ticket's
        # eligibility, nor vice versa.
        path2, handle2 = hwlock._publish(queue, 2, 'ticket', dict(
            locks=['gpu1', 'rhea'], gate_ancestors=[1]))
        try:
            assert not hwlock.backfill_check(queue, ['rhea'], 3, 100, now)[0]
        finally:
            handle2.close()
            path2.unlink()
    finally:
        handle.close()
        path.unlink()


def test_bounded_small_job_backfills_ahead_of_a_waiting_whole_cluster_ticket(lockdir):
    # 2026-11-23: A held sparks,gpu1; B (older) wanted the whole cluster; C wanted gpu0 and GPU0 idled.
    a = spawn_bounded(lockdir, 'sparks,gpu1', 4.0, lockdir / 'a', max_run=60)
    wait_for(lockdir / 'a')
    b = spawn_bounded(lockdir, WHOLE, 0.1, lockdir / 'b')
    queued(lockdir, 1)
    c = spawn_bounded(lockdir, 'gpu0', 0.3, lockdir / 'c', max_run=20)
    for proc in (a, b, c):
        assert proc.wait(timeout=60) == 0, proc.stderr.read()
    assert wait_for(lockdir / 'c') < end_of(lockdir / 'a')  # C ran while A held its locks
    assert wait_for(lockdir / 'b') >= max(end_of(lockdir / 'a'), end_of(lockdir / 'c'))
    assert 'backfilled' in c.stderr.read()


@pytest.mark.parametrize('holder_bound,candidate_bound', [(5, 30), (None, 2), (60, None)],
                         ids=['candidate-bound-exceeds-holder', 'holder-unbounded', 'candidate-unbounded'])
def test_no_backfill_unless_both_bounds_allow_it(lockdir, holder_bound, candidate_bound):
    a = spawn_bounded(lockdir, 'sparks,gpu1', 2.0, lockdir / 'a', max_run=holder_bound)
    wait_for(lockdir / 'a')
    b = spawn_bounded(lockdir, WHOLE, 0.5, lockdir / 'b')
    queued(lockdir, 1)
    c = spawn_bounded(lockdir, 'gpu0', 0.1, lockdir / 'c', max_run=candidate_bound)
    for proc in (a, b, c):
        assert proc.wait(timeout=60) == 0, proc.stderr.read()
    # Today's strict order: C waits behind B.
    assert wait_for(lockdir / 'c') >= end_of(lockdir / 'b')
    assert 'backfilled' not in c.stderr.read()


def test_an_overrunning_backfill_is_killed_at_its_bound_and_the_older_ticket_follows(lockdir):
    a = spawn_bounded(lockdir, 'sparks,gpu1', 0.5, lockdir / 'a', max_run=60)
    wait_for(lockdir / 'a')
    b = spawn_bounded(lockdir, WHOLE, 0.1, lockdir / 'b')
    queued(lockdir, 1)
    c = spawn_bounded(lockdir, 'gpu0', 60, lockdir / 'c', max_run=3)
    started = wait_for(lockdir / 'c')
    assert c.wait(timeout=30) == 124
    assert time.time() - started < 3 + 2
    assert '--max-run 3 s reached' in c.stderr.read()
    assert not Path(str(lockdir / 'c') + '.end').exists()
    assert a.wait(timeout=30) == 0 and b.wait(timeout=30) == 0
    # B waited on C's gpu0 past A's release, but no longer than C's bound (+ one 2 s poll).
    began = wait_for(lockdir / 'b')
    assert started + 3 - 0.5 <= began <= started + 3 + hwlock.POLL_S + 2.5
    assert not list((lockdir / 'hwq').glob('*.held'))


def test_nested_acquisition_inside_a_backfilled_job_is_refused(lockdir):
    a = spawn_bounded(lockdir, 'sparks,gpu1', 5.0, lockdir / 'a', max_run=60)
    wait_for(lockdir / 'a')
    b = spawn_bounded(lockdir, WHOLE, 0.1, lockdir / 'b')
    queued(lockdir, 1)
    rcs = lockdir / 'inner-rcs'
    inner = '; '.join(f'{sys.executable} {SCRIPT} --locks {locks} --wait 5 {extra} -- true; echo $? >> {rcs}'
                      for locks, extra in (('gpu0', ''), ('kakapo', ''), ('cassowary', '--gate'),
                                           ('maxq', '--max-run 5')))
    c = spawn_bounded(lockdir, 'gpu0', 0, lockdir / 'c', max_run=20, inner=inner)
    assert c.wait(timeout=60) == 0, c.stderr.read()
    assert wait_for(lockdir / 'c') < end_of(lockdir / 'a')
    assert rcs.read_text().split() == ['77'] * 4
    assert a.wait(timeout=60) == 0 and b.wait(timeout=60) == 0


def test_the_gate_child_path_is_unchanged(lockdir):
    env = {**os.environ, 'CUTEAFD_LOCK_DIR': str(lockdir)}
    # A bounded gate whose child queues for the same locks (as smoke does); the
    # gate's own holder record does not count as an enclosing hold.
    inner = f'{sys.executable} {SCRIPT} --locks sparks,gpu1 --wait 5 -- true'
    gated = subprocess.run([sys.executable, str(SCRIPT), '--locks', 'sparks,gpu0,gpu1', '--gate', '--max-run', '30',
                            '--wait', '5', '--', 'sh', '-c', inner], env=env, capture_output=True, text=True, timeout=30)
    assert gated.returncode == 0, gated.stderr
    # In-process: a gate's record is live while its child holds the flocks without a ticket.
    with hwlock.hold(['gpu1'], wait=5, gate=True):
        assert [data['gate'] for data in hwlock.enclosing_holders(lockdir / 'hwq')] == [True]
        os.environ[hwlock.GATE_ENV] = str(hwlock.GATE_SEQ[0])
        try:
            with hwlock.hold(['gpu1'], wait=5) as handles:
                assert len(handles) == 1
                assert not [p for p in (lockdir / 'hwq').glob('*.ticket') if f'{hwlock.GATE_SEQ[0]:012d}' not in p.name]
        finally:
            del os.environ[hwlock.GATE_ENV]


# 2026-11-23: wip-cards held ticket 13742 (sparks,gpu0,gpu1,moa) as a plain
# holder; a nested launch under it asked hwlock for the same locks, queued a new
# ticket behind its own parent and waited on flocks the parent held. Both GPUs
# sat idle for 13 minutes with three tickets queued behind it.
@pytest.mark.parametrize('bounded', [False, True])
@pytest.mark.parametrize('inner_locks, extra, covered', [
    ('sparks,gpu0,gpu1,moa', '', True),
    ('gpu0', '', True),
    ('gpu1,moa', '--max-run 5', True),
    ('gpu0,rhea', '', False),
    ('kakapo', '--gate', False),
])
def test_nested_acquisition_inside_a_plain_holder_is_refused_at_once(lockdir, bounded, inner_locks, extra, covered):
    env = {**os.environ, 'CUTEAFD_LOCK_DIR': str(lockdir)}
    queue = lockdir / 'hwq'
    report = lockdir / 'inner'
    # The inner call records its exit status, stderr and the tickets it left behind.
    inner = (f'{sys.executable} {SCRIPT} --locks {inner_locks} --wait 30 {extra} -- true 2> {report}.err; '
             f'echo $? > {report}.rc; ls {queue} > {report}.queue')
    started = time.monotonic()
    outer = subprocess.run([sys.executable, str(SCRIPT), '--locks', 'sparks,gpu0,gpu1,moa', '--wait', '5',
                            *(['--max-run', '60'] if bounded else []), '--', 'sh', '-c', inner],
                           env=env, capture_output=True, text=True, timeout=60)
    assert outer.returncode == 0, outer.stderr
    assert time.monotonic() - started < 10, 'the inner request must fail at once, not wait'
    assert Path(f'{report}.rc').read_text().split() == ['77']
    # Only the outer holder's record: the refused request took no ticket.
    left = Path(f'{report}.queue').read_text().split()
    assert not [name for name in left if name.endswith('.ticket')], left
    assert [name for name in left if name.endswith('.held')] and len([n for n in left if n.endswith('.held')]) == 1
    message = Path(f'{report}.err').read_text()
    assert 'hwlock: refusing' in message and 'ancestor ticket' in message
    if covered:
        assert 'already held by ancestor ticket' in message and 'CUTEAFD_HW_LOCKS_HELD' in message
    else:
        assert 'request every lock in the one outer hwlock call' in message
        assert 'already held' not in message
    assert not list(queue.glob('*.ticket')) and not list(queue.glob('*.held'))


def test_in_process_nesting_inside_a_plain_holder_is_refused(lockdir):
    with hwlock.hold(['gpu0', 'gpu1'], wait=5):
        for names, text in ((['gpu1'], 'already held by ancestor ticket'),
                            (['gpu1', 'rhea'], 'request every lock in the one outer hwlock call (--locks gpu0,gpu1,rhea)')):
            with pytest.raises(hwlock.NestedAcquisition, match=re.escape(text)):
                with hwlock.hold(names, wait=5):
                    pytest.fail('a nested hold is never granted')
        assert not list((lockdir / 'hwq').glob('*.ticket'))
    # Released: the same request now proceeds.
    with hwlock.hold(['gpu1'], wait=5) as handles:
        assert len(handles) == 1


def test_many_small_backfills_never_delay_the_older_big_job_past_the_bounds(lockdir):
    a = spawn_bounded(lockdir, 'sparks,gpu1', 3.0, lockdir / 'a', max_run=6)
    granted = wait_for(lockdir / 'a')
    b = spawn_bounded(lockdir, WHOLE, 0.1, lockdir / 'b')
    queued(lockdir, 1)
    smalls = []
    for n in range(12):
        smalls.append(spawn_bounded(lockdir, 'gpu0', 0.2, lockdir / f'c{n}', max_run=1.5))
        time.sleep(0.25)
    for proc in (a, b, *smalls):
        assert proc.wait(timeout=90) == 0, proc.stderr.read()
    began = wait_for(lockdir / 'b')
    # No backfill outlives A's own bound, so B starts by then (plus one poll).
    assert began <= granted + 6 + hwlock.POLL_S + 1.0
    backfilled = [n for n in range(12) if wait_for(lockdir / f'c{n}') < began]
    assert backfilled, 'some small job should have used the idle GPU'
    assert all(end_of(lockdir / f'c{n}') <= began for n in backfilled)
    # The rest waited their turn behind B.
    assert all(wait_for(lockdir / f'c{n}') >= end_of(lockdir / 'b') for n in range(12) if n not in backfilled)


def test_max_run_must_be_finite_and_positive(lockdir):
    env = {**os.environ, 'CUTEAFD_LOCK_DIR': str(lockdir)}
    for value in ('0', '-1', 'inf', 'nan'):
        result = subprocess.run([sys.executable, str(SCRIPT), '--locks', 'gpu1', '--max-run', value, '--', 'true'],
                                env=env, capture_output=True, text=True)
        assert result.returncode == 2 and '--max-run' in result.stderr
    with pytest.raises(ValueError):
        with hwlock.hold(['gpu1'], 5, max_run=0):
            pass

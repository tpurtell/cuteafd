#!/usr/bin/env python3
"""Fair (first come, first served) acquisition of the cluster hardware locks.

  scripts/hw/hwlock.py --locks sparks,gpu0,gpu1 [--wait SECONDS] [--gate] -- CMD ARGS...

Each request takes a numbered ticket in ~/.cache/cuteafd/hwq and holds an
exclusive flock on it while it waits; a ticket whose flock is free belongs to a
dead process and is removed. A request takes its locks only when no older live
ticket wants any of the same locks, so a job waiting for both GPUs is not
overtaken by later single-GPU jobs, while jobs on disjoint locks still run side
by side. The locks themselves are the usual flock files, taken in the canonical
order (sparks, gpu0, gpu1, rhea, moa, kakapo, cassowary, kakapo-export,
cassowary-export, oliver), so callers that still use plain flock stay excluded (they are
just not queued fairly). Exit 75 when the wait expires.

Default: the locks are held while CMD runs, and CUTEAFD_HW_LOCKS_HELD names
them so a `cuteafd bench smoke` child skips taking them again. --gate is for a
CMD that takes the locks itself (an older smoke binary, a driver script): once
it is our turn the locks are left for CMD, and our ticket stays live until CMD
exits, so later ticket holders still wait behind it.

SIGTERM, SIGINT or SIGHUP while CMD runs is forwarded to CMD, and the locks
stay held until CMD exits (at most STOP_GRACE_S = 80 s, then CMD is killed):
`cuteafd bench smoke` tears down its containers on that signal, and the next
lock holder must find the hardware clean. 80 s fits systemd's default
TimeoutStopSec (90 s), so `systemctl --user stop UNIT` on a unit running
hwlock is safe with the default KillMode=control-group; do not pass
KillMode=mixed or process, or a TimeoutStopSec below 90 s.

--preflight runs a bounded shell check before enqueue and after grant (only
after grant with --preflight-at-grant), while
locks are still held even in --gate mode. Failure or timeout exits 78 without
running CMD; --preflight-timeout defaults to 300 seconds per check.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

# kakapo and cassowary are the LAN-only build seeds (10 GbE, never serve): after
# every serving lock, so a serving job's locks are always taken before a seed
# lease's. Each seed's GPU-export turn (SEED-export) comes last: a build takes it
# while its counting lease on that seed is live (scripts/build/build-resource.py).
LAN_SEEDS = ('kakapo', 'cassowary')
# oliver's RTX PRO 6000 Max-Q (SM120, WSL2) runs single-GPU kernel correctness,
# profiles and same-session A/B, never serving; nothing else is held with it.
ORDER = ('sparks', 'gpu0', 'gpu1', 'rhea', 'moa', *LAN_SEEDS, *(f'{seed}-export' for seed in LAN_SEEDS),
         'oliver')
LOCK_DIR = Path(os.environ.get('CUTEAFD_LOCK_DIR', Path.home() / '.cache/cuteafd'))
POLL_S = 2.0
# How long a signalled CMD may take to exit (its teardown) before it is killed;
# under systemd's default TimeoutStopSec (90 s), above smoke's 60 s teardown bound.
STOP_GRACE_S = 80.0
FORWARDED = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
GATE_SEQ = [0]  # this process's ticket while it holds a turn (exported in --gate mode)


def canonical(names) -> list[str]:
    names = [n.removesuffix('.lock') for n in names]
    unknown = sorted(set(names) - set(ORDER))
    if unknown:
        raise ValueError(f'unknown hardware locks: {unknown}')
    return [n for n in ORDER if n in names]


def _queue() -> Path:
    queue = LOCK_DIR / 'hwq'
    queue.mkdir(parents=True, exist_ok=True)
    return queue


def _next_seq(queue: Path) -> int:
    with open(queue / '.seq.lock', 'a+') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        counter = queue / 'seq'
        seq = int(counter.read_text() or 0) + 1 if counter.exists() else 1
        counter.write_text(str(seq))
        return seq


def _take_ticket(queue: Path, locks: list[str]) -> tuple[int, Path, object]:
    seq = _next_seq(queue)
    staged = queue / f'.{seq:012d}.{os.getpid()}.tmp'
    handle = open(staged, 'w')
    # Lock before the ticket becomes visible, so nobody can mistake it for stale.
    fcntl.flock(handle, fcntl.LOCK_EX)
    handle.write(json.dumps(dict(locks=locks, pid=os.getpid(), host=socket.gethostname(),
                                 created=time.time(), argv=sys.argv[1:][:8])))
    handle.flush()
    path = queue / f'{seq:012d}.{os.getpid()}.ticket'
    os.rename(staged, path)
    return seq, path, handle


GATE_ENV = 'CUTEAFD_HW_GATE_TICKETS'


def gate_ancestors() -> set[int]:
    """Tickets of `--gate` ancestors: they hold the turn for this process, so
    a ticket taken inside a gated command must not wait behind them."""
    return {int(s) for s in os.environ.get(GATE_ENV, '').split(':') if s.isdigit()}


def inside_live_gate(queue: Path) -> bool:
    """Whether a `--gate` ancestor's ticket is still live (it holds our turn)."""
    for seq in gate_ancestors():
        for path in queue.glob(f'{seq:012d}.*.ticket'):
            try:
                with open(path) as handle:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            except FileNotFoundError:
                continue
    return False


def older_conflicts(queue: Path, locks: list[str], before: int | None) -> list[str]:
    """Live tickets older than `before` (all, when None) wanting any of `locks`,
    except our own `--gate` ancestors. Removes stale tickets of any age on the
    way (their owner's flock is free)."""
    wanted, found, ancestors = set(locks), [], gate_ancestors()
    for path in sorted(queue.glob('*.ticket')):
        seq = int(path.name.split('.', 1)[0])
        try:
            with open(path) as other:
                try:
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    if before is not None and seq >= before or seq in ancestors:
                        continue  # live and not older (ours or behind us), or our gate
                    data = json.loads(path.read_text() or '{}')
                    if wanted & set(data.get('locks', [])):
                        found.append(path.name)
                    continue
                path.unlink(missing_ok=True)  # its owner is gone
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    return found


def _try_flocks(locks: list[str]) -> list | None:
    held = []
    for name in locks:
        handle = open(LOCK_DIR / f'{name}.lock', 'a')
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            for other in held:
                other.close()
            return None
        held.append(handle)
    return held


class PreflightFailed(Exception):
    pass


class Cancelled(Exception):
    """The waiter's `cancelled` callback asked it to stop waiting."""


def check_preflight(command, timeout):
    if not command:
        return
    proc = subprocess.Popen(command, shell=True, start_new_session=True)
    try:
        code = proc.wait(timeout=timeout)
    except BaseException as error:
        # The shell may have forked a harness; kill the entire check, not just sh.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        if isinstance(error, subprocess.TimeoutExpired):
            raise PreflightFailed('preflight timed out') from error
        raise
    # Preflights must not leave helpers on the hardware, even on success.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if code:
        raise PreflightFailed(f'preflight failed (exit {code})')


@contextlib.contextmanager
def hold(names, wait: float, label: str = '', gate: bool = False, preflight: str = '',
         preflight_timeout: float = 300, preflight_at_grant_only: bool = False, cancelled=None):
    """Hold `names` (canonical order) for the body; FIFO against other tickets.
    With `gate`, release the flocks at our turn and keep the ticket instead.
    `cancelled()`, polled while waiting, raises Cancelled when true."""
    locks = canonical(names)
    # A check on state the queue itself changes (free memory on shared hosts) is
    # meaningful only once our turn comes; such checks run at grant alone.
    if not preflight_at_grant_only:
        check_preflight(preflight, preflight_timeout)
    queue = _queue()
    if not gate and inside_live_gate(queue):
        # The --gate ancestor already won this job's turn; a new, younger
        # ticket would wait behind jobs that themselves wait on the gate.
        held = None
        deadline = time.monotonic() + wait
        while (held := _try_flocks(locks)) is None:
            if cancelled is not None and cancelled():
                raise Cancelled(f'stopped waiting for {",".join(locks)}')
            if time.monotonic() >= deadline:
                raise TimeoutError(f'no flocks for {",".join(locks)} after {wait:.0f} s inside a gate')
            time.sleep(POLL_S)
        try:
            yield held
        finally:
            for handle in held:
                handle.close()
        return
    seq, path, ticket = _take_ticket(queue, locks)
    GATE_SEQ[0] = seq
    deadline = time.monotonic() + wait
    held, announced = None, False
    try:
        while True:
            if not older_conflicts(queue, locks, seq):
                held = _try_flocks(locks)
                if held is not None:
                    break
            if cancelled is not None and cancelled():
                raise Cancelled(f'stopped waiting for {",".join(locks)}')
            if time.monotonic() >= deadline:
                raise TimeoutError(f'no turn for {",".join(locks)} after {wait:.0f} s')
            if not announced:
                print(f'hwlock: ticket {seq} queued for {",".join(locks)} {label}'.rstrip(), file=sys.stderr)
                announced = True
            time.sleep(POLL_S)
    except BaseException:
        path.unlink(missing_ok=True)
        ticket.close()
        raise
    try:
        check_preflight(preflight, preflight_timeout)
    except BaseException:
        for handle in held:
            handle.close()
        path.unlink(missing_ok=True)
        ticket.close()
        raise
    if gate:
        for handle in held:
            handle.close()
        held = []
    else:
        # The flocks now exclude everyone; the ticket only ordered the waiters.
        path.unlink(missing_ok=True)
        ticket.close()
    try:
        yield held
    finally:
        for handle in held:
            handle.close()
        if gate:
            path.unlink(missing_ok=True)
            ticket.close()


def run_forwarding(cmd, env, grace: float = STOP_GRACE_S) -> int:
    """Runs CMD to completion while the caller holds the locks. A SIGTERM,
    SIGINT or SIGHUP is forwarded to CMD, which then gets `grace` seconds to
    exit (its own teardown) before SIGKILL; only then does this return, so
    the locks outlive whatever CMD cleans up. Returns CMD's exit status
    (128 + N when a signal N ended it)."""
    received: list[int] = []
    child = None

    def forward(signum, _frame):
        if not received:
            received.append(signum)
        if child is not None and child.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                child.send_signal(signum)

    previous = {sig: signal.signal(sig, forward) for sig in FORWARDED}
    try:
        child = subprocess.Popen(cmd, env=env)
        if received:  # arrived before CMD existed
            forward(received[0], None)
        deadline = None
        while True:
            try:
                code = child.wait(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                pass
            if received and deadline is None:
                deadline = time.monotonic() + grace
                print(f'hwlock: signal {received[0]} forwarded; holding the locks up to {grace:.0f} s '
                      'while the command stops', file=sys.stderr)
            if deadline is not None and time.monotonic() >= deadline:
                print(f'hwlock: the command did not stop within {grace:.0f} s; killing it', file=sys.stderr)
                child.kill()
                code = child.wait()
                break
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 128 - code if code < 0 else code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--locks', required=True, help='comma list of ' + ', '.join(ORDER))
    parser.add_argument('--wait', type=float, default=float(os.environ.get('CUTEAFD_HW_LOCK_WAIT_S', 21600)))
    parser.add_argument('--preflight', default='', help='shell command checked before enqueue and at grant')
    parser.add_argument('--preflight-timeout', type=float, default=300, help='timeout per preflight in seconds (default 300)')
    parser.add_argument('--preflight-at-grant', action='store_true',
                        help='run --preflight only after the locks are granted (for checks on shared, queue-dependent state)')
    parser.add_argument('--gate', action='store_true', help='CMD takes the locks itself; keep only our turn')
    parser.add_argument('cmd', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cmd = args.cmd[1:] if args.cmd[:1] == ['--'] else args.cmd
    if not cmd:
        parser.error('missing command')
    if not 0 < args.preflight_timeout < float('inf'):
        parser.error('--preflight-timeout must be finite and positive')
    try:
        locks = canonical(args.locks.split(','))
        with hold(locks, args.wait, ' '.join(cmd[:2]), gate=args.gate,
                  preflight=args.preflight, preflight_timeout=args.preflight_timeout,
                  preflight_at_grant_only=args.preflight_at_grant):
            env = dict(os.environ)
            if not args.gate:
                env['CUTEAFD_HW_LOCKS_HELD'] = ','.join(locks)
            else:
                env[GATE_ENV] = ':'.join(filter(None, [os.environ.get(GATE_ENV, ''), str(GATE_SEQ[0])]))
            return run_forwarding(cmd, env)
    except PreflightFailed as error:
        print(f'hwlock: {error}', file=sys.stderr)
        return 78
    except TimeoutError as error:
        print(f'hwlock: {error}', file=sys.stderr)
        return 75


if __name__ == '__main__':
    sys.exit(main())

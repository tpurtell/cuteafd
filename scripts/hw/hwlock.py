#!/usr/bin/env python3
"""Fair (first come, first served) acquisition of the cluster hardware locks.

  scripts/hw/hwlock.py --locks sparks,gpu0,gpu1 [--wait SECONDS] [--gate] -- CMD ARGS...

Each request takes a numbered ticket in ~/.cache/cuteafd/hwq and holds an
exclusive flock on it while it waits; a ticket whose flock is free belongs to a
dead process and is removed. A request takes its locks only when no older live
ticket wants any of the same locks, so a job waiting for both GPUs is not
overtaken by later single-GPU jobs, while jobs on disjoint locks still run side
by side. The locks themselves are the usual flock files, taken in the canonical
order (sparks, gpu0, gpu1, rhea, moa, kakapo, cassowary, maxq), so callers
that still use plain flock stay excluded (they are just not queued fairly).
Exit 75 when the wait expires.

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

Bounded backfill (--max-run SECONDS). The bound is recorded in the ticket and
enforced from the grant on, at-grant preflight included: SIGTERM to CMD (or to
an in-process holder) at the bound, the usual STOP_GRACE_S, then SIGKILL; such
a run exits 124. Every holder publishes `SEQ.PID.held` in the queue (flocked
like a ticket) naming its locks and `release_by`, the latest wall time its
locks are free again (bound + HARD_STOP_S), or null without --max-run. A
request with --max-run, not --gate, not inside a live gate, may start before
older live tickets that want some of its locks only when all of these hold:
  - none of its locks is flocked or named by a live holder record;
  - for every such older ticket T, our own release_by is no later than its
    blocking deadline W(T). Compute W in ticket order, considering only older
    overlapping tickets. If T has held locks, W(T) is the earliest recorded
    blocker deadline, as before. If all T's locks are free, apply this same
    backfill rule to T: if it could start now, W(T) is now; otherwise inherit
    the earliest deadline of the older tickets that prevent its grant. An
    unbounded or gate request cannot itself backfill.
A lock held outside the queue (plain flock, `cuteafd bench smoke`) or by a
holder without a bound counts as "may free any moment", so nobody backfills
past a ticket waiting on it, directly or through an older blocking ticket.
A granted gate's locks are named by its holder record even when not flocked;
its own children retain their gate exemption, not a backfill grant.
A backfilled job is gone by W(T), preserving the existing no-delay guarantee
at the recorded deadlines (a holder may in fact finish earlier). Every
backfill fits under that same deadline; a convoy of waiting tickets does not
make it later. The bound must cover the
at-grant preflight and CMD's own teardown. Inside a live gate the gate's own
bound applies (its children take flocks without tickets or records).
A request without --max-run never backfills and queues exactly as before.

No nesting inside a holder, gates excepted: any acquisition (a new ticket,
--gate or in-process hold) by a live non-gate holder's process or its
descendants (by /proc parent links) is refused at once (exit 77). An inner
ticket queues behind every older ticket, and those wait on the outer holder's
flocks (directly, or through a lock they also want), so it can never be granted
while the outer hold waits for it: a self-deadlock that idles the hardware.
When the outer hold covers the request, the child runs without hwlock
(CUTEAFD_HW_LOCKS_HELD names the held locks for that); otherwise request every
lock in the one outer hwlock call. Inside a backfilled job the refusal names
the ticket the backfill went ahead of. Backfill never applies inside a live
gate: such a request queues as before. The --gate child path is unchanged: a
gate's holder record is exempt, and inside a live gate a hold takes the flocks
without a ticket.
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
import threading
import time
from pathlib import Path

# kakapo and cassowary are the LAN-only build seeds (10 GbE, never serve): after
# every serving lock, so a serving job's locks are always taken before a seed
# lease's. Kernel AOT exports run on CPU and need no GPU export lock.
LAN_SEEDS = ('kakapo', 'cassowary')
# maxq's RTX PRO 6000 Max-Q (SM120, WSL2) handles quants, CPU gates and
# single-GPU correctness, never perf A/B, cards or serving.
ORDER = ('sparks', 'gpu0', 'gpu1', 'rhea', 'moa', *LAN_SEEDS, 'maxq')
_DEFAULT_LOCK_DIR = Path(os.environ.get('CUTEAFD_LOCK_DIR', Path.home() / '.cache/cuteafd'))
# An explicit assignment (a driver pinning a private directory, a test) wins;
# otherwise CUTEAFD_LOCK_DIR is read when the lock is taken, so a process that
# sets it after importing this module still never reaches the real locks.
LOCK_DIR = _DEFAULT_LOCK_DIR


def lock_dir() -> Path:
    """The lock directory for this call: an explicit LOCK_DIR assignment, else CUTEAFD_LOCK_DIR."""
    if LOCK_DIR != _DEFAULT_LOCK_DIR:
        return LOCK_DIR
    return Path(os.environ.get('CUTEAFD_LOCK_DIR', Path.home() / '.cache/cuteafd'))


POLL_S = 2.0
# How long a signalled CMD may take to exit (its teardown) before it is killed;
# under systemd's default TimeoutStopSec (90 s), above smoke's 60 s teardown bound.
STOP_GRACE_S = 80.0
FORWARDED = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
GATE_SEQ = [0]  # this process's ticket while it holds a turn (exported in --gate mode)
# From a bounded run's SIGTERM to its SIGKILL: the stop grace plus a margin for
# run_forwarding's 0.5 s poll, so a CLI holder kills CMD itself first.
HARD_STOP_S = STOP_GRACE_S + 5.0
OVERRAN = [False]  # set when this process's --max-run bound fired
NESTED_EXIT = 77


def canonical(names) -> list[str]:
    names = [n.removesuffix('.lock') for n in names]
    unknown = sorted(set(names) - set(ORDER))
    if unknown:
        raise ValueError(f'unknown hardware locks: {unknown}')
    return [n for n in ORDER if n in names]


def _queue() -> Path:
    queue = lock_dir() / 'hwq'
    queue.mkdir(parents=True, exist_ok=True)
    return queue


def _next_seq(queue: Path) -> int:
    with open(queue / '.seq.lock', 'a+') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        counter = queue / 'seq'
        seq = int(counter.read_text() or 0) + 1 if counter.exists() else 1
        counter.write_text(str(seq))
        return seq


def _publish(queue: Path, seq: int, suffix: str, data: dict) -> tuple[Path, object]:
    staged = queue / f'.{seq:012d}.{os.getpid()}.{suffix}.tmp'
    handle = open(staged, 'w')
    # Lock before the entry becomes visible, so nobody can mistake it for stale.
    fcntl.flock(handle, fcntl.LOCK_EX)
    handle.write(json.dumps(data))
    handle.flush()
    path = queue / f'{seq:012d}.{os.getpid()}.{suffix}'
    os.rename(staged, path)
    return path, handle


def _take_ticket(queue: Path, locks: list[str], max_run: float | None = None,
                 gate: bool = False) -> tuple[int, Path, object]:
    seq = _next_seq(queue)
    data = dict(locks=locks, pid=os.getpid(), host=socket.gethostname(), created=time.time(), argv=sys.argv[1:][:8],
                gate=gate, gate_ancestors=sorted(gate_ancestors()))
    if max_run is not None:
        data['max_run'] = max_run
    path, handle = _publish(queue, seq, 'ticket', data)
    return seq, path, handle


def _live_entries(queue: Path, suffix: str) -> list[tuple[int, str, dict]]:
    """Live `SEQ.PID.SUFFIX` entries, oldest first, as (seq, name, data). Removes
    stale ones on the way (their owner's flock is free)."""
    found = []
    for path in sorted(queue.glob(f'*.{suffix}')):
        try:
            seq = int(path.name.split('.', 1)[0])
        except ValueError:
            continue
        try:
            with open(path) as other:
                try:
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    found.append((seq, path.name, json.loads(path.read_text() or '{}')))
                    continue
                path.unlink(missing_ok=True)  # its owner is gone
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    return found


def _lock_busy(name: str) -> bool:
    """Whether someone holds `name`'s flock now (a brief non-blocking probe)."""
    held = _try_flocks([name])
    if held is None:
        return True
    for handle in held:
        handle.close()
    return False


def _ancestry() -> set[int]:
    """This process and its ancestors (by /proc parent links)."""
    pids, pid = set(), os.getpid()
    while pid > 1 and pid not in pids:
        pids.add(pid)
        try:
            pid = int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            break
    return pids


class NestedAcquisition(ValueError):
    """An acquisition inside a live non-gate holder (refused; module doc)."""


def enclosing_holders(queue: Path) -> list[dict]:
    """Live holder records of this process or its ancestors (we run inside them)."""
    host, mine = socket.gethostname(), _ancestry()
    return [data for _, _, data in _live_entries(queue, 'held') if data.get('host') == host and data.get('pid') in mine]


def backfill_check(queue: Path, locks: list[str], seq: int, max_run: float, now: float | None = None) -> tuple[bool, str]:
    """Whether ticket `seq` (wanting `locks`, bounded by `max_run`) may start now
    ahead of the older live tickets that want some of its locks (module doc).
    Returns (allowed, reason)."""
    now = time.time() if now is None else now
    ours = now + max_run + HARD_STOP_S
    holders = _live_entries(queue, 'held')
    release: dict[str, list] = {}
    for _, _, data in holders:
        for name in data.get('locks', []):
            release.setdefault(name, []).append(data.get('release_by'))
    # Snapshot the physical lock state at most once per name, including locks
    # held without a queue record. A granted gate counts as holding its locks.
    busy: dict[str, bool] = {}

    def lock_busy(name):
        if name not in busy:
            busy[name] = _lock_busy(name)
        return busy[name]

    for name in locks:
        if name in release or lock_busy(name):
            return False, f'{name} is held'

    # Each entry is (locks, blocking deadline, reason). A None deadline means
    # the ticket could start now or its blocker may disappear at any moment.
    # Evaluate only earlier tickets, so the dependency walk cannot cycle.
    waiting = {}
    ancestors = gate_ancestors()
    for other, _, data in _live_entries(queue, 'ticket'):
        if other >= seq:
            break
        theirs = set(data.get('locks', []))
        bounds = []
        reason = ''
        for name in sorted(theirs):
            if name in release:
                if any(bound is None for bound in release[name]):
                    reason = f'ticket {other} waits on {name}, held without a bound'
                    break
                bounds.append(max(release[name]))
            elif lock_busy(name):
                reason = f'ticket {other} waits on {name}, held outside the queue'
                break
        if reason:
            deadline = None
        elif bounds:
            deadline = min(bounds)
        else:
            # Free physical locks alone do not mean a ticket can start. Check
            # the same backfill budget it would have to satisfy at this poll.
            # Legacy tickets lack gate metadata; assuming a bounded one can
            # backfill is conservative (it may instead have to wait).
            budget = data.get('max_run')
            end = now + budget + HARD_STOP_S if budget is not None and not data.get('gate') else None
            exempt = set(data.get('gate_ancestors', []))
            blockers = [(bound, why) for older, (wanted, bound, why) in waiting.items()
                        if older not in exempt and theirs & wanted
                        and (end is None or bound is None or end > bound)]
            if blockers:
                deadline, reason = min(blockers, key=lambda entry: entry[0] if entry[0] is not None else now)
                if deadline is None:
                    reason = f'ticket {other} waits behind {reason}'
            else:
                deadline = None
                reason = f'ticket {other} could start now'
        waiting[other] = (theirs, deadline, reason)

    wanted, passed = set(locks), []
    for other, (theirs, deadline, reason) in waiting.items():
        if other in ancestors or not wanted & theirs:
            continue
        if deadline is None:
            return False, reason
        if ours > deadline:
            return False, f'our bound ends {ours - deadline:.0f} s after ticket {other}\'s first blocker must release'
        passed.append(other)
    return True, 'ahead of ticket(s) ' + ','.join(map(str, passed))


class _Watchdog:
    """Enforces a holder's bound from the grant: SIGTERM to this process at
    `max_run` (run_forwarding passes it to CMD; an in-process holder's own
    handler tears down), SIGKILL HARD_STOP_S later if it still runs."""

    def __init__(self, max_run: float):
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._run, args=(max_run,), daemon=True)
        self.thread.start()

    def _run(self, max_run):
        if self.done.wait(max_run):
            return
        OVERRAN[0] = True
        print(f'hwlock: --max-run {max_run:.0f} s reached; stopping', file=sys.stderr, flush=True)
        os.kill(os.getpid(), signal.SIGTERM)
        if self.done.wait(HARD_STOP_S):
            return
        print(f'hwlock: still running {HARD_STOP_S:.0f} s past --max-run; killing', file=sys.stderr, flush=True)
        os.kill(os.getpid(), signal.SIGKILL)

    def stop(self):
        self.done.set()


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
        handle = open(lock_dir() / f'{name}.lock', 'a')
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
         preflight_timeout: float = 300, preflight_at_grant_only: bool = False, cancelled=None,
         max_run: float | None = None):
    """Hold `names` (canonical order) for the body; FIFO against other tickets.
    With `gate`, release the flocks at our turn and keep the ticket instead.
    `cancelled()`, polled while waiting, raises Cancelled when true. `max_run`
    bounds the hold from its grant (enforced) and lets a non-gate request
    backfill (module doc); inside a live non-gate holder, NestedAcquisition."""
    locks = canonical(names)
    if max_run is not None and not 0 < max_run < float('inf'):
        raise ValueError('max_run must be finite and positive')
    # Read without creating the queue: a failing enqueue preflight leaves no trace.
    enclosing = enclosing_holders(lock_dir() / 'hwq')
    # A gate leaves its flocks to CMD, which may take them (path below); any
    # other enclosing holder keeps them while it waits for us (module doc).
    outer = [data for data in enclosing if not data.get('gate')]
    inner = next((data for data in outer if data.get('backfill')), None)
    if inner is not None:
        raise NestedAcquisition(
            f'refusing {",".join(locks)} inside backfilled ticket {inner.get("seq")} '
            f'(holding {",".join(inner.get("locks", []))}): an inner ticket would queue behind the ticket '
            'it went ahead of; request every lock in the one outer hwlock call')
    if outer:
        tickets = ','.join(str(data.get('seq')) for data in outer)
        held_here = canonical({name for data in outer for name in data.get('locks', [])})
        if set(locks) <= set(held_here):
            raise NestedAcquisition(
                f'refusing {",".join(locks)}: already held by ancestor ticket {tickets} '
                f'({",".join(held_here)}); run this child without hwlock (CUTEAFD_HW_LOCKS_HELD names the held locks)')
        raise NestedAcquisition(
            f'refusing {",".join(locks)} inside ancestor ticket {tickets} (holding {",".join(held_here)}): '
            'an inner ticket would wait on its own ancestor; request every lock in the one outer hwlock call '
            f'(--locks {",".join(canonical(set(held_here) | set(locks)))})')
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
    seq, path, ticket = _take_ticket(queue, locks, max_run, gate)
    GATE_SEQ[0] = seq
    deadline = time.monotonic() + wait
    held, announced, backfill = None, False, False
    try:
        while True:
            if not older_conflicts(queue, locks, seq):
                held = _try_flocks(locks)
                if held is not None:
                    break
            elif max_run is not None and not gate and not enclosing:
                allowed, reason = backfill_check(queue, locks, seq, max_run)
                if allowed and (held := _try_flocks(locks)) is not None:
                    backfill = True
                    print(f'hwlock: ticket {seq} backfilled for {",".join(locks)} with --max-run {max_run:.0f} s, '
                          f'{reason}', file=sys.stderr)
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
    # Published while the flocks are held, before the at-grant preflight: the
    # bound counts from here, and waiters see it before our ticket goes.
    granted, started = time.time(), time.monotonic()
    record, record_handle = _publish(queue, seq, 'held', dict(
        seq=seq, locks=locks, pid=os.getpid(), host=socket.gethostname(), granted=granted, gate=gate,
        backfill=backfill, max_run=max_run, release_by=None if max_run is None else granted + max_run + HARD_STOP_S))
    watchdog = None

    def release_record():
        if watchdog is not None:
            watchdog.stop()
        record.unlink(missing_ok=True)
        record_handle.close()
    try:
        # The at-grant check runs inside the bound; the watchdog takes the rest.
        check_preflight(preflight, preflight_timeout if max_run is None else min(preflight_timeout, max_run))
        if max_run is not None:
            left = max_run - (time.monotonic() - started)
            if left < 1:
                raise PreflightFailed(f'the at-grant preflight used the whole --max-run {max_run:.0f} s')
            watchdog = _Watchdog(left)
    except BaseException:
        for handle in held:
            handle.close()
        release_record()
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
        release_record()
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
    parser.add_argument('--max-run', type=float, metavar='SECONDS',
                        help='hard bound on the hold from its grant (SIGTERM, grace, SIGKILL; exit 124); '
                        'lets this request backfill ahead of older tickets it cannot delay (not with --gate)')
    parser.add_argument('cmd', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cmd = args.cmd[1:] if args.cmd[:1] == ['--'] else args.cmd
    if not cmd:
        parser.error('missing command')
    if not 0 < args.preflight_timeout < float('inf'):
        parser.error('--preflight-timeout must be finite and positive')
    if args.max_run is not None and not 0 < args.max_run < float('inf'):
        parser.error('--max-run must be finite and positive')
    try:
        locks = canonical(args.locks.split(','))
        with hold(locks, args.wait, ' '.join(cmd[:2]), gate=args.gate,
                  preflight=args.preflight, preflight_timeout=args.preflight_timeout,
                  preflight_at_grant_only=args.preflight_at_grant, max_run=args.max_run):
            env = dict(os.environ)
            if not args.gate:
                env['CUTEAFD_HW_LOCKS_HELD'] = ','.join(locks)
            else:
                env[GATE_ENV] = ':'.join(filter(None, [os.environ.get(GATE_ENV, ''), str(GATE_SEQ[0])]))
            code = run_forwarding(cmd, env)
        return 124 if OVERRAN[0] else code
    except NestedAcquisition as error:
        print(f'hwlock: {error}', file=sys.stderr)
        return NESTED_EXIT
    except PreflightFailed as error:
        print(f'hwlock: {error}', file=sys.stderr)
        return 78
    except TimeoutError as error:
        print(f'hwlock: {error}', file=sys.stderr)
        return 75


if __name__ == '__main__':
    sys.exit(main())

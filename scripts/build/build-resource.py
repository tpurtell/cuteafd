#!/usr/bin/env python3
"""Bounded build-only host leases; serving seeds retain their hardware mutex.

A LAN-only build seed (LAN_SEEDS: GB10, 10 GbE, never serves) admits several
builds at once: a shared SEED.lock umbrella plus one exclusive SEED.build.N.lock
per counting slot, min(cores/6, MiB/32768) of them. The slots cover Cargo and the
other CPU phases only; wip.sh takes the seed's exclusive GPU-export turn
(hwlock SEED-export) around the native export inside the lease. With --balance an
automatic seed takes the LAN seed with the fewest held slots (the named --host on
a tie), then --fallback (rhea/moa under their exclusive locks). --pick prints that
choice without waiting or holding anything (wip.sh's early seed placement).
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

# Keep in sync with release_lan_seed_hosts (scripts/lib/release-common.sh) and
# scripts/hw/hwlock.py LAN_SEEDS.
LAN_SEEDS = ('kakapo', 'cassowary')


def _hwlock():
    spec = importlib.util.spec_from_file_location('cuteafd_hwlock', Path(__file__).resolve().parents[1] / 'hw/hwlock.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capacity(cores: int, memory_mib: int) -> int:
    return max(1, min(cores // 6, memory_mib // 32768))


def probe(host: str) -> int | None:
    try:
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', host,
                                 'nproc; awk \'/MemTotal/ {print int($2/1024)}\' /proc/meminfo'],
                                capture_output=True, text=True, timeout=15, check=True)
        return capacity(*map(int, result.stdout.split()))
    except (subprocess.SubprocessError, OSError, ValueError, TypeError):
        return None


def owner_alive(pid: int, start: str) -> bool:
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == start
    except (OSError, IndexError):
        return False


def try_lock(path: Path, shared: bool = False):
    handle = path.open('a')
    try:
        fcntl.flock(handle, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        return handle
    except BlockingIOError:
        handle.close()
        return None


def held_slots(directory: Path, host: str, slots: int) -> int:
    """Builds holding a counting slot on `host` now; all of them when an exclusive
    user (a legacy SEED.lock holder) has the umbrella."""
    umbrella = try_lock(directory / f'{host}.lock', shared=True)
    if umbrella is None:
        return slots
    try:
        held = 0
        for slot in range(slots):
            handle = try_lock(directory / f'{host}.build.{slot}.lock')
            if handle is None:
                held += 1
            else:
                handle.close()
        return held
    finally:
        umbrella.close()


def lan_order(directory: Path, preferred: str, capacities: dict[str, int | None]) -> list[str]:
    """Reachable LAN seeds, least loaded first; `preferred` wins a tie."""
    usable = [host for host in LAN_SEEDS if capacities.get(host)]
    return sorted(usable, key=lambda host: (held_slots(directory, host, capacities[host]), host != preferred,
                                            LAN_SEEDS.index(host)))


def acquire(directory: Path, host: str, slots: int):
    umbrella = try_lock(directory / f'{host}.lock', shared=True)
    if umbrella is None:
        return None
    for slot in range(slots):
        handle = try_lock(directory / f'{host}.build.{slot}.lock')
        if handle is not None:
            return [umbrella, handle]
    umbrella.close()
    return None


def take_export_turn(stack: contextlib.ExitStack, directory: Path, host: str, args, cancelled,
                     handles: list | None = None) -> str:
    """The seed's GPU-export turn for this lease: the fair hwlock SEED-export lock
    on a LAN seed, whose counting lease admits other builds; a seed under an
    exclusive lock (rhea, moa) is already ours. Drop the plain-flocked SEED.lock
    umbrella before queueing in hwlock so a fair-queued job wanting both SEED and
    SEED-export cannot deadlock with this turn; the counting slot stays held.
    Tells wip.sh through export-granted or export-failed; the turn is held in
    `stack` until export-done or teardown."""
    started = time.monotonic()
    if host in LAN_SEEDS:
        while handles and len(handles) > 1:
            handles.pop(0).close()
        hwlock = _hwlock()
        hwlock.LOCK_DIR = directory
        try:
            stack.enter_context(hwlock.hold([f'{host}-export'], args.wait, f'GPU export on {host}', cancelled=cancelled))
        except hwlock.Cancelled:
            return 'cancelled'
        except TimeoutError as error:
            (args.lease_dir / 'export-failed').write_text(str(error))
            return 'failed'
    (args.lease_dir / 'export-granted').write_text(json.dumps({'host': host, 'waited_s': round(time.monotonic() - started)}))
    return 'held'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='kakapo')
    parser.add_argument('--slots', type=int)
    parser.add_argument('--fallback', default='')
    parser.add_argument('--lease-dir', type=Path, required=True)
    parser.add_argument('--wait', type=int, default=1800)
    parser.add_argument('--owner', type=int, default=os.getppid())
    parser.add_argument('--container')
    parser.add_argument('--lock-host')
    parser.add_argument('--balance', action='store_true',
                        help='automatic seed: the least-loaded LAN seed, --host first on a tie')
    parser.add_argument('--pick', action='store_true',
                        help='print the seed --balance/--fallback would try first now, hold nothing, exit 75 if none')
    args = parser.parse_args()
    if args.slots is not None and args.slots < 1 or args.wait < 0:
        parser.error('slots must be positive and wait nonnegative')
    candidates = list(filter(None, args.fallback.split(',')))
    if any(host not in ('rhea', 'moa') for host in candidates):
        parser.error('fallback seeds must be rhea or moa')
    if args.balance and args.host not in LAN_SEEDS:
        parser.error('--balance needs a LAN seed --host')
    directory = Path(os.environ.get('CUTEAFD_LOCK_DIR', str(Path.home() / '.cache/cuteafd')))
    if not directory.is_dir():
        parser.error(f'lock directory does not exist: {directory}')
    lan = LAN_SEEDS if args.balance else (args.host,) if args.host in LAN_SEEDS else ()
    def seed_capacity(host):
        # Balancing drops an unreachable seed even when --slots fixes the count.
        if args.slots is not None and not args.balance:
            return args.slots
        found = probe(host)
        return found and (args.slots or found)
    capacities = {host: seed_capacity(host) for host in lan}
    if args.pick:
        order = lan_order(directory, args.host, capacities) if lan else [args.host]
        # A full LAN set still names its least-loaded seed: the lease waits there.
        for host in order:
            print(host)
            return
        for host in candidates:
            if probe(host) is not None:
                print(host)
                return
        raise SystemExit(75)
    try:
        start = Path(f'/proc/{args.owner}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return  # the owner is already gone: nothing to acquire for

    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGHUP, stop)
    signal.signal(signal.SIGQUIT, stop)
    deadline = time.monotonic() + args.wait
    handles = None
    slots = None
    while not stopping and owner_alive(args.owner, start):
        host = args.host
        handles = None
        if lan:
            # Ordering and taking a slot are one step, so two builds arriving
            # together land on different idle seeds.
            with (directory / 'lan-seeds.pick.lock').open('a') as guard:
                fcntl.flock(guard, fcntl.LOCK_EX)
                for host in lan_order(directory, args.host, capacities):
                    handles = acquire(directory, host, capacities[host])
                    if handles is not None:
                        slots = capacities[host]
                        break
        else:
            lock = try_lock(directory / f'{args.lock_host or host}.lock')
            handles, slots = ([lock], 1) if lock is not None else (None, None)
        if handles is None:
            for candidate in candidates:
                lock = try_lock(directory / f'{candidate}.lock')
                if lock is not None:
                    handles, host = [lock], candidate
                    break
        if handles is not None:
            break
        if time.monotonic() >= deadline:
            raise SystemExit(75)
        time.sleep(1)
    if handles is None:
        return
    fd = None
    export_turn = contextlib.ExitStack()
    try:
        if stopping or not owner_alive(args.owner, start):
            return
        (args.lease_dir / 'granted').write_text(json.dumps({'host': host, 'slots': slots or 1}))
        fd = os.open(args.lease_dir / 'release', os.O_RDONLY | os.O_NONBLOCK)
        # No lease expiry: each phase has its own terminating timeout. Stop
        # the instance's build container before *any* mutex release.
        export_state = 'idle'
        while not stopping and owner_alive(args.owner, start):
            if os.read(fd, 1):
                break
            if export_state == 'idle' and (args.lease_dir / 'export-request').exists():
                export_state = take_export_turn(export_turn, directory, host, args,
                                                lambda: stopping or not owner_alive(args.owner, start), handles)
                if export_state == 'cancelled':
                    break
            elif export_state == 'held' and (args.lease_dir / 'export-done').exists():
                export_turn.close()  # the export's ssh returned: nothing of it runs on the GPU
                export_state = 'done'
            time.sleep(0.2)
        if args.container and (args.lease_dir / 'active').exists():
            while True:
                try:
                    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', host,
                                         f'names=$(docker ps -a --filter name=^/{args.container}$ --format \'{{{{.Names}}}}\') && '
                                         f'{{ test -z "$names" || {{ docker stop -t 10 {args.container} >/dev/null; '
                                         f'test "$(docker inspect -f \'{{{{.State.Running}}}}\' {args.container})" = false; }}; }}'],
                                            timeout=30, check=False)
                except (subprocess.SubprocessError, OSError):
                    time.sleep(60)
                    continue
                if result.returncode == 0:
                    break
                # Fail closed: an unreachable host is not proof that CUDA stopped.
                time.sleep(60)
    finally:
        if fd is not None:
            os.close(fd)
        export_turn.close()
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    main()

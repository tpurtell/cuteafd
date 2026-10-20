#!/usr/bin/env python3
"""Memory admission/watchdog and fair-queued coordinator export placement.

`hold --gpus 1,0 --wait 14400 -- CMD` reuses a caller-held GPU, otherwise
queues for the first configured GPU. Memory admission and watchdog limits apply
inside that turn; a native-cache hit never calls this helper.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

NO_TURN = 75
CONTEXT_MIB = 640  # one exporter's CUDA context (~550 MiB measured) plus slack
SIZING_MIB = 2048  # the largest per-process EXL3 sizing pass measured
MAX_CAP_MIB = 12288  # the measured 8.7 GiB full-scope peak plus margin
HEADROOM_MIB = 4096  # left free on the device beyond the cap
MIN_CAP_MIB = SIZING_MIB + 2 * CONTEXT_MIB


def query(smi: str, fields: str, extra=()) -> list[list[str]]:
    out = subprocess.run([smi, f'--query-{fields}', '--format=csv,noheader,nounits', *extra],
                         check=True, capture_output=True, text=True, timeout=60).stdout
    return [[c.strip() for c in line.split(',')] for line in out.splitlines() if line.strip()]


def gpu_row(rows: list[list[str]], gpu: str) -> list[str]:
    """rows of index, uuid, memory.used, memory.total, memory.free."""
    for row in rows:
        if gpu in (row[0], row[1]):
            return row
    raise SystemExit(f'export-placement: GPU {gpu} not found')


def admit(free_mib: int, max_cap_mib: int = MAX_CAP_MIB, headroom_mib: int = HEADROOM_MIB) -> tuple[int, int] | None:
    """The cap (MiB) and Ninja job count for an export on a device with this much
    free memory, or None when even the smallest useful cap does not fit."""
    cap = min(max_cap_mib, free_mib - headroom_mib)
    if cap < MIN_CAP_MIB:
        return None
    jobs = max(1, (cap - SIZING_MIB) // CONTEXT_MIB)
    return cap, jobs


RAM_PER_WORKER_MIB = 4096
RAM_HEADROOM_MIB = 8192


@contextmanager
def ledger(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(path.read_text()) if path.exists() else {}
        yield state
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state))
        temporary.replace(path)


def available_ram():
    return next(int(line.split()[1]) // 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))


def reserve(path, token, row, ram_mib, owner, max_cap=MAX_CAP_MIB, max_jobs=64):
    with ledger(path) as state:
        if callable(row):
            row = row()
        if callable(ram_mib):
            ram_mib = ram_mib()
        if token in state.get('leases', {}):
            raise ValueError('duplicate export reservation')
        if not state.get('leases'):
            state.clear()
            state.update(baseline=int(row[2]), ram_baseline=ram_mib, total=int(row[3]), leases={})
        leases = state['leases']
        reserved = sum(item['cap'] for item in leases.values())
        ram_reserved = sum(item['ram'] for item in leases.values())
        # Account for reservations not yet allocated, and external allocations
        # since the common baseline. Existing reservations are never reaped on
        # a PID check alone: their owned container could still be running.
        gpu_free = min(int(row[4]), state['total'] - max(state['baseline'], int(row[2])) - reserved)
        ram_free = min(ram_mib, state['ram_baseline']) - ram_reserved
        decision = admit(gpu_free, max_cap)
        if decision is None:
            return None
        cap, _ = decision
        jobs = min(cap // (SIZING_MIB + CONTEXT_MIB), max_jobs, (ram_free - RAM_HEADROOM_MIB) // RAM_PER_WORKER_MIB)
        if jobs < 2:
            return None
        cap = jobs * (SIZING_MIB + CONTEXT_MIB)
        ram = jobs * RAM_PER_WORKER_MIB
        leases[token] = dict(cap=cap, ram=ram, jobs=jobs, owner=owner)
        return cap, jobs, state['baseline'], ram


def release_reservation(path, token):
    with ledger(path) as state:
        state.get('leases', {}).pop(token, None)
        if not state.get('leases'):
            state.clear()


def reservation_budget(path, token):
    with ledger(path) as state:
        if token not in state.get('leases', {}):
            raise ValueError('export reservation disappeared')
        return state['baseline'], sum(item['cap'] for item in state['leases'].values())


def cgroup_pids(container_id: str) -> set[int]:
    pids = set()
    for proc in Path('/proc').iterdir():
        if proc.name.isdigit():
            try:
                if container_id in (proc / 'cgroup').read_text():
                    pids.add(int(proc.name))
            except OSError:
                pass
    return pids


def export_mib(smi: str, gpu: str, cgroup: str | None, baseline: int | None) -> int:
    if cgroup:
        uuid = gpu_row(query(smi, 'gpu=index,uuid,memory.used,memory.total,memory.free'), gpu)[1]
        owned = cgroup_pids(cgroup)
        total = 0
        for app_uuid, pid, used in query(smi, 'compute-apps=gpu_uuid,pid,used_memory'):
            if app_uuid == uuid and pid.isdigit() and int(pid) in owned and used.isdigit():
                total += int(used)
        return total
    used = int(gpu_row(query(smi, 'gpu=index,uuid,memory.used,memory.total,memory.free'), gpu)[2])
    return max(0, used - baseline)


def watch(args) -> int:
    process = subprocess.Popen(args.command, start_new_session=True)
    peak = 0
    while process.poll() is None:
        try:
            if getattr(args, 'ledger', None):
                baseline, cap = reservation_budget(args.ledger, args.reservation)
                used = export_mib(args.smi, args.gpu, None, baseline)
                args.cap_mib = cap
            else:
                used = export_mib(args.smi, args.gpu, args.cgroup, args.baseline_mib)
        except (subprocess.SubprocessError, OSError, ValueError, SystemExit) as error:
            print(f'export-placement: memory budget unreadable: {error}; stopping export', file=sys.stderr)
            used = args.cap_mib + 1
        peak = max(peak, used)
        if used > args.cap_mib:
            print(f'export-placement: export used {used} MiB over its {args.cap_mib} MiB cap; stopping it',
                  file=sys.stderr)
            os.killpg(process.pid, signal.SIGTERM)
            if args.cgroup:
                for pid in cgroup_pids(args.cgroup):
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except OSError:
                        pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return NO_TURN
        time.sleep(args.interval)
    print(f'export-placement: export peak {peak} MiB (cap {args.cap_mib} MiB)', file=sys.stderr)
    return process.returncode


def held_by_caller() -> list[str]:
    return [lock for lock in os.environ.get('CUTEAFD_HW_LOCKS_HELD', '').split(',') if lock]


def run_held(command, env, handle=None):
    # The child retains the mutex even if this supervisor is killed abruptly.
    # Forwarders are installed before the spawn so a signal that arrives while
    # the child starts is delivered to it rather than killing this supervisor;
    # one that arrives before the child exists is forwarded once it does.
    process = None
    pending = []
    previous = {}
    def forward(signum, frame):
        if process is None:
            pending.append(signum)
            return
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass
    try:
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT):
            previous[signum] = signal.signal(signum, forward)
        process = subprocess.Popen(command, env=env, start_new_session=True,
                                   pass_fds=(() if handle is None else (handle.fileno(),)))
        for signum in pending:
            forward(signum, None)
        return process.wait()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def hold(gpus: list[str], command: list[str], lock_dir: Path, wait: float = 14400) -> int:
    if not gpus or any(gpu not in ('0', '1') for gpu in gpus):
        raise ValueError('export GPUs must be a nonempty list of 0 or 1')
    held = held_by_caller()
    for gpu in gpus:
        if f'gpu{gpu}' in held:
            print(f'export-placement: caller already holds gpu{gpu}.lock', file=sys.stderr)
            return run_held(command, {**os.environ, 'CUTEAFD_EXPORT_GPU': gpu})
    # A seed/serving lock held by an outer caller cannot be released here.
    if held:
        raise ValueError('release other hardware locks before queueing an export')
    spec = importlib.util.spec_from_file_location('export_hwlock', Path(__file__).parents[1] / 'hw/hwlock.py')
    hwlock = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hwlock)
    hwlock.LOCK_DIR = lock_dir
    lock_dir.mkdir(parents=True, exist_ok=True)
    gpu = gpus[0]
    try:
        with hwlock.hold([f'gpu{gpu}'], wait, 'coordinator export',
                         preflight='command -v bash >/dev/null') as handles:
            print(f'export-placement: holding gpu{gpu}.lock for the export', file=sys.stderr)
            env = {**os.environ, 'CUTEAFD_EXPORT_GPU': gpu, 'CUTEAFD_HW_LOCKS_HELD': f'gpu{gpu}'}
            return run_held(command, env, handles[0])
    except TimeoutError as error:
        print(f'export-placement: {error}', file=sys.stderr)
        return NO_TURN


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='action', required=True)
    a = sub.add_parser('admit')
    a.add_argument('--smi', default='nvidia-smi')
    a.add_argument('--gpu', required=True)
    a.add_argument('--max-cap-mib', type=int, default=MAX_CAP_MIB)
    a.add_argument('--headroom-mib', type=int, default=HEADROOM_MIB)
    r = sub.add_parser('reserve')
    r.add_argument('--ledger', type=Path, required=True)
    r.add_argument('--reservation', required=True)
    r.add_argument('--owner', type=int, required=True)
    r.add_argument('--smi', default='nvidia-smi')
    r.add_argument('--gpu', required=True)
    r.add_argument('--max-jobs', type=int, default=os.cpu_count() or 1)
    r.add_argument('--max-cap-mib', type=int, default=MAX_CAP_MIB)
    d = sub.add_parser('release')
    d.add_argument('--ledger', type=Path, required=True)
    d.add_argument('--reservation', required=True)
    h = sub.add_parser('hold')
    h.add_argument('--gpus', default='1,0')
    h.add_argument('--wait', type=float, default=14400)
    h.add_argument('command', nargs=argparse.REMAINDER)
    w = sub.add_parser('watch')
    w.add_argument('--smi', default='nvidia-smi')
    w.add_argument('--gpu', required=True)
    w.add_argument('--cap-mib', type=int, required=True)
    group = w.add_mutually_exclusive_group(required=True)
    group.add_argument('--cgroup')
    group.add_argument('--baseline-mib', type=int)
    group.add_argument('--ledger', type=Path)
    w.add_argument('--reservation')
    w.add_argument('--interval', type=float, default=1.0)
    w.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.action == 'release':
        release_reservation(args.ledger, args.reservation)
        return 0
    if args.action == 'reserve':
        row = lambda: gpu_row(query(args.smi, 'gpu=index,uuid,memory.used,memory.total,memory.free'), args.gpu)
        decision = reserve(args.ledger, args.reservation, row, available_ram, args.owner,
                           args.max_cap_mib, args.max_jobs)
        if decision is None:
            return NO_TURN
        print(*decision)
        return 0
    if args.action == 'admit':
        row = gpu_row(query(args.smi, 'gpu=index,uuid,memory.used,memory.total,memory.free'), args.gpu)
        decision = admit(int(row[4]), args.max_cap_mib, args.headroom_mib)
        if decision is None:
            print(f'export-placement: GPU {args.gpu} has {row[4]} MiB free; an export needs '
                  f'{MIN_CAP_MIB + args.headroom_mib} MiB', file=sys.stderr)
            return NO_TURN
        print(decision[0], decision[1], row[2])
        return 0
    if args.command[:1] == ['--']:
        args.command = args.command[1:]
    if not args.command:
        parser.error(f'{args.action} needs a command')
    if args.action == 'hold':
        lock_dir = Path(os.environ.get('CUTEAFD_LOCK_DIR', Path.home() / '.cache/cuteafd'))
        if not 0 <= args.wait < float('inf'):
            parser.error('--wait must be finite and nonnegative')
        return hold([g for g in args.gpus.split(',') if g], args.command, lock_dir, args.wait)
    return watch(args)


if __name__ == '__main__':
    sys.exit(main())

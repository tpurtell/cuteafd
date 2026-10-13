#!/usr/bin/env python3
"""Admission, memory cap and SM check for a coordinator native/AOT export that
runs without a fair GPU turn (wip.sh CUTEAFD_WIP_COORDINATOR_EXPORT).

The coordinator exporters only compile: CuTe DSL and Triton targets come from
device properties, and the live device hosts CUDA contexts (~550 MiB each) and
the EXL3 buffer-sizing pass (<= 2 GiB per process). A cold export of the full
release scope peaked at 8.7 GiB with ~15 exporters in parallel and 3 of 797
one-second samples showed any SM activity (2026-10-13, raptor gpu1). So an
export can share a GPU with serving, or with someone else's job, under a cap.

  export-placement.py admit --smi SMI --gpu INDEX|UUID [--max-cap-mib N] [--headroom-mib N]
      Read the HOST's nvidia-smi (on WSL2 /usr/lib/wsl/lib/nvidia-smi: a
      container's cudaMemGetInfo cannot see the host's other processes) and
      print `cap_mib jobs used_mib` when the export fits, else exit 75.
  export-placement.py hold --gpus 1,0 -- CMD...
      Take the first of those GPUs' hardware locks that is free right now
      (non-blocking flock on ~/.cache/cuteafd/gpuN.lock, never a wait, so no
      queued job is overtaken), run CMD holding it with CUTEAFD_EXPORT_GPU=N,
      and release it after. A GPU named in CUTEAFD_HW_LOCKS_HELD (the caller
      already holds it) is used without probing. Exit 75 when all are held:
      a held lock means the GPU is not ours (TJ's interactive gpu0 use too).
  export-placement.py watch --smi SMI --gpu G --cap-mib N (--cgroup ID | --baseline-mib N) -- CMD...
      Run CMD; kill it and exit 75 as soon as the export's device memory goes
      over the cap. --cgroup sums the processes in that container's cgroup
      (per-process memory is reported natively); --baseline-mib watches
      device-wide growth over the admission reading (WSL2 reports per-process
      memory as N/A).
"""
from __future__ import annotations

import argparse
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
            used = export_mib(args.smi, args.gpu, args.cgroup, args.baseline_mib)
        except (subprocess.SubprocessError, OSError, ValueError):
            used = 0  # a failed read never kills; the next one decides
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


def hold(gpus: list[str], command: list[str], lock_dir: Path) -> int:
    import fcntl

    held = held_by_caller()
    for gpu in gpus:
        if f'gpu{gpu}' in held:
            print(f'export-placement: caller already holds gpu{gpu}.lock', file=sys.stderr)
            return subprocess.run(command, env={**os.environ, 'CUTEAFD_EXPORT_GPU': gpu}).returncode
    for gpu in gpus:
        handle = open(lock_dir / f'gpu{gpu}.lock', 'a')
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            continue
        try:
            print(f'export-placement: holding gpu{gpu}.lock for the export', file=sys.stderr)
            env = {**os.environ, 'CUTEAFD_EXPORT_GPU': gpu,
                   'CUTEAFD_HW_LOCKS_HELD': ','.join([*held, f'gpu{gpu}'])}
            return subprocess.run(command, env=env).returncode
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()
    print(f'export-placement: every candidate GPU lock is held ({",".join("gpu" + g for g in gpus)})', file=sys.stderr)
    return NO_TURN


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='action', required=True)
    a = sub.add_parser('admit')
    a.add_argument('--smi', default='nvidia-smi')
    a.add_argument('--gpu', required=True)
    a.add_argument('--max-cap-mib', type=int, default=MAX_CAP_MIB)
    a.add_argument('--headroom-mib', type=int, default=HEADROOM_MIB)
    h = sub.add_parser('hold')
    h.add_argument('--gpus', default='1,0')
    h.add_argument('command', nargs=argparse.REMAINDER)
    w = sub.add_parser('watch')
    w.add_argument('--smi', default='nvidia-smi')
    w.add_argument('--gpu', required=True)
    w.add_argument('--cap-mib', type=int, required=True)
    group = w.add_mutually_exclusive_group(required=True)
    group.add_argument('--cgroup')
    group.add_argument('--baseline-mib', type=int)
    w.add_argument('--interval', type=float, default=1.0)
    w.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
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
        return hold([g for g in args.gpus.split(',') if g], args.command, lock_dir)
    return watch(args)


if __name__ == '__main__':
    sys.exit(main())

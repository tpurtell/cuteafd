#!/usr/bin/env python3
"""Python client for PeerSwarm's SQLite fair resource & lock scheduler.

Replaces the legacy flock/hwq implementation in scripts/hw/hwlock.py,
scripts/build/build-slot.sh, and scripts/build/build-resource.py. All locking,
FIFO queuing, bounded backfill, nesting refusal, and dead-PID reaping are
delegated to the Rust `peerswarm resource` binary.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

LAN_SEEDS = ('kakapo', 'cassowary')
ORDER = ('sparks', 'gpu0', 'gpu1', 'rhea', 'moa', *LAN_SEEDS, 'maxq', 'build', 'cpu-gate')
STOP_GRACE_S = 80.0
HARD_STOP_S = STOP_GRACE_S + 5.0
NESTED_EXIT = 77
OVERRAN = [False]
# Optional explicit state-dir override (used by test fixtures isolating locks).
LOCK_DIR: Path | None = None


class PreflightFailed(RuntimeError):
    """Raised when a pre-enqueue or at-grant preflight check fails."""


class Cancelled(BaseException):
    """Raised when a waiter or holder is cancelled by signal."""


def repo_root() -> Path:
    env_root = os.environ.get('PEERSWARM_ROOT')
    if env_root and (Path(env_root) / 'peerswarm.toml').is_file():
        return Path(env_root).resolve()
    cur = Path(__file__).resolve().parents[2]
    while True:
        if (cur / 'peerswarm.toml').is_file():
            return cur
        if cur.parent == cur:
            return Path(__file__).resolve().parents[2]
        cur = cur.parent


def cache_root() -> Path:
    for var in ('CUTEAFD_CACHE_ROOT', 'PEERSWARM_CACHES'):
        val = os.environ.get(var)
        if val:
            return Path(val).expanduser()
    root = repo_root()
    if (root / 'peerswarm.toml').is_file():
        return root / 'peerswarm.state' / 'caches'
    return Path.home() / '.cache' / 'cuteafd'


def build_root() -> Path:
    val = os.environ.get('CUTEAFD_BUILD_ROOT')
    if val:
        return Path(val).expanduser()
    return cache_root() / 'builds'


def peerswarm_bin() -> str:
    env_bin = os.environ.get('PEERSWARM_BIN')
    if env_bin and os.access(env_bin, os.X_OK):
        return env_bin
    candidate = repo_root() / 'peerswarm' / 'target' / 'release' / 'peerswarm'
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    found = shutil.which('peerswarm')
    if found:
        return found
    raise FileNotFoundError(
        f'peerswarm binary not found; build with: cargo build --release --manifest-path {repo_root() / "peerswarm/Cargo.toml"}'
    )


def _base_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault('PEERSWARM_ROOT', str(repo_root()))
    # Honor explicit test isolation via LOCK_DIR or CUTEAFD_LOCK_DIR.
    if LOCK_DIR is not None:
        env['PEERSWARM_STATE_DIR'] = str(LOCK_DIR)
    elif 'PEERSWARM_STATE_DIR' not in env and os.environ.get('CUTEAFD_LOCK_DIR'):
        env['PEERSWARM_STATE_DIR'] = os.environ['CUTEAFD_LOCK_DIR']
    if extra:
        env.update(extra)
    return env


def canonical(names) -> list[str]:
    cleaned = [str(n).strip().removesuffix('.lock') for n in names if str(n).strip()]
    unknown = sorted(set(cleaned) - set(ORDER))
    if unknown:
        raise ValueError(f'unknown hardware locks: {unknown}')
    return [n for n in ORDER if n in cleaned]


def check_preflight(command: str, timeout: float = 300.0) -> None:
    if not command:
        return
    try:
        result = subprocess.run(
            ['bash', '-lc', command],
            timeout=max(1.0, float(timeout)),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise PreflightFailed(f'preflight timed out after {timeout:.0f}s') from exc
    if result.returncode != 0:
        raise PreflightFailed(f'preflight failed (exit {result.returncode}): {command}')


def exec_argv(
    locks,
    command: list[str],
    *,
    wait: float = 7200.0,
    max_run: float | None = None,
    stop_grace: float = STOP_GRACE_S,
    exclusive: bool = False,
    gate: bool = False,
    preflight: str = '',
    preflight_at_grant: bool = False,
    preflight_timeout: float = 300.0,
    owner: str | None = None,
    nice: int | None = None,
) -> list[str]:
    ordered = canonical(locks)
    argv = [
        peerswarm_bin(),
        '--root', str(repo_root()),
        'resource', 'exec',
        '--locks', ','.join(ordered),
        '--wait', str(max(1, int(wait))),
        '--stop-grace', str(max(1, int(stop_grace))),
    ]
    if exclusive:
        argv.append('--exclusive')
    if gate:
        argv.append('--gate')
    if max_run is not None:
        argv.extend(['--max-run', str(max(1, int(max_run)))])
    if preflight:
        argv.extend(['--preflight', preflight, '--preflight-timeout', str(max(1, int(preflight_timeout)))])
        if preflight_at_grant:
            argv.append('--preflight-at-grant')
    if owner:
        argv.extend(['--owner', owner])
    if nice is not None:
        argv.extend(['--nice', str(nice)])
    argv.append('--')
    argv.extend(command)
    return argv


def is_free(locks) -> bool:
    """Return True iff every named resource currently has zero held slots in PeerSwarm."""
    ordered = canonical(locks)
    res = subprocess.run(
        [peerswarm_bin(), '--root', str(repo_root()), '--json', 'resource', 'status'],
        env=_base_env(),
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(res.stdout)
    by_name = {entry['name']: entry for entry in data.get('resources', [])}
    return all(by_name.get(name, {}).get('held_slots', 0) == 0 for name in ordered)


def pick(pool: str, *, prefer: str | None = None, fallback: list[str] | None = None, probe_ssh: bool = False) -> str:
    argv = [peerswarm_bin(), '--root', str(repo_root()), 'resource', 'pick', '--pool', pool]
    if prefer:
        argv.extend(['--prefer', prefer])
    if fallback:
        argv.extend(['--fallback', ','.join(fallback)])
    if probe_ssh:
        argv.append('--probe-ssh')
    res = subprocess.run(argv, env=_base_env(), capture_output=True, text=True, check=True)
    return res.stdout.strip()


@contextlib.contextmanager
def hold(
    locks,
    wait: float = 7200.0,
    owner: str = 'python',
    *,
    preflight: str = '',
    preflight_timeout: float = 300.0,
    preflight_at_grant_only: bool = False,
    max_run: float | None = None,
    cancelled=None,
    exclusive: bool = False,
    gate: bool = False,
):
    """Hold PeerSwarm SQLite resource leases for the duration of a Python `with` block."""
    OVERRAN[0] = False
    ordered = canonical(locks)
    if preflight and not preflight_at_grant_only:
        check_preflight(preflight, preflight_timeout)

    with tempfile.TemporaryDirectory(prefix='peerswarm-hold-') as td:
        ready = Path(td) / 'granted.json'
        argv = [
            peerswarm_bin(),
            '--root', str(repo_root()),
            'resource', 'hold',
            '--locks', ','.join(ordered),
            '--wait', str(max(1, int(wait))),
            '--owner', str(owner),
            '--ready-file', str(ready),
        ]
        if exclusive:
            argv.append('--exclusive')
        if gate:
            argv.append('--gate')
        if max_run is not None:
            argv.extend(['--max-run', str(max(1, int(max_run)))])

        proc = subprocess.Popen(argv, env=_base_env())
        watchdog: threading.Timer | None = None
        grant_data: dict | None = None
        try:
            while not ready.is_file():
                if cancelled is not None and cancelled():
                    proc.terminate()
                    proc.wait(timeout=10)
                    raise Cancelled('cancelled while waiting for resource lease')
                ret = proc.poll()
                if ret is not None:
                    if ret == 75:
                        raise TimeoutError(f'timed out after {wait:.0f}s waiting for {",".join(ordered)}')
                    if ret == NESTED_EXIT:
                        raise ValueError(f'nested lock acquisition refused for {",".join(ordered)}')
                    raise RuntimeError(f'peerswarm resource hold failed (exit {ret})')
                time.sleep(0.1)

            grant_data = json.loads(ready.read_text())

            if preflight:
                check_preflight(preflight, preflight_timeout)

            if max_run is not None:
                def _on_overrun():
                    OVERRAN[0] = True
                    os.kill(os.getpid(), signal.SIGTERM)

                watchdog = threading.Timer(float(max_run), _on_overrun)
                watchdog.daemon = True
                watchdog.start()

            prev_held = os.environ.get('CUTEAFD_HW_LOCKS_HELD')
            prev_ps = os.environ.get('PEERSWARM_LOCKS_HELD')
            merged = ','.join(canonical([*(prev_held.split(',') if prev_held else []), *ordered]))
            os.environ['CUTEAFD_HW_LOCKS_HELD'] = merged
            os.environ['PEERSWARM_LOCKS_HELD'] = merged
            try:
                yield grant_data
            finally:
                if prev_held is None:
                    os.environ.pop('CUTEAFD_HW_LOCKS_HELD', None)
                else:
                    os.environ['CUTEAFD_HW_LOCKS_HELD'] = prev_held
                if prev_ps is None:
                    os.environ.pop('PEERSWARM_LOCKS_HELD', None)
                else:
                    os.environ['PEERSWARM_LOCKS_HELD'] = prev_ps
        finally:
            if watchdog is not None:
                watchdog.cancel()
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

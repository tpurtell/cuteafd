"""scripts/dev/cache-gc: idle selection and the mount/process/lock safety rules."""
import fcntl
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/dev/cache-gc'
LOADER = importlib.machinery.SourceFileLoader('cache_gc', str(SCRIPT))
SPEC = importlib.util.spec_from_loader('cache_gc', LOADER)
gc = importlib.util.module_from_spec(SPEC)
sys.modules['cache_gc'] = gc  # dataclasses resolve annotations through sys.modules
LOADER.exec_module(gc)

NOW = time.time()
OLD = NOW - 30 * 86400


def make(path, age=OLD, size=16):
    """A file at path (parents created), with every created entry back-dated to age."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'x' * size)
    return touch(path, age)


def touch(path, age):
    os.utime(path, (age, age), follow_symlinks=False)
    return path


def backdate(root, age=OLD):
    for directory, dirs, files in os.walk(root):
        for name in dirs + files:
            touch(Path(directory) / name, age)
    touch(root, age)


@pytest.fixture
def cache(tmp_path):
    root = tmp_path / 'cuteafd'
    make(root / 'builds/idle/target/debug/big.rlib')
    make(root / 'builds/fresh/target/debug/big.rlib')
    make(root / 'builds/fresh/STATUS.md', age=NOW)  # one fresh file keeps the whole task
    make(root / 'builds/active/STATUS.md', age=NOW)
    make(root / 'builds/active/head/tmp/pytest/popen-gw0/test_x0/blob')
    make(root / 'builds/active/head/target-cpu-gate/debug/deps/a.rlib')
    make(root / 'builds/active/base/tmp/old')
    make(root / 'builds/compiler-cache/kache/x86_64/blob')
    make(root / 'builds/codex-runs/brief.md')
    make(root / 'wip-run/slot/serve-default/bin/cuteafd')
    make(root / 'wip-run/inst/slot/serve-x/lib/libcuteafd_native.so')
    make(root / 'wip-run/slot2.tmp/raw/cuteafd')
    make(root / 'wip-run/live/serve-default/bin/cuteafd', age=NOW)
    for store in ('bench/bench.sqlite', 'fidelity/rows/a.json', 'console/secret', 'instances/default/usage/usage.sqlite',
                  'jit/abc/x86_64/triton/k', 'kache/x86_64/b', 'sccache/x86_64/c', 'cargo-home/x86_64/registry/r',
                  'hwq/q', 'tools/x86_64/kache', 'wip-slots/s.json', 'run-config/a.config'):
        make(root / store)
    make(root / 'gpu1.lock', size=0)
    backdate(root / 'builds/idle')
    for path in (root / 'builds/active/head', root / 'builds/active/base', root / 'builds/active/head/tmp',
                 root / 'builds/active/head/target-cpu-gate'):
        backdate(path)
    for path in (root / 'wip-run/slot', root / 'wip-run/inst', root / 'wip-run/slot2.tmp'):
        backdate(path)
    return root


def chosen(root, env=None, days=7):
    selected, kept = gc.plan(root, days, NOW, env or gc.Environment())
    return {str(c.path.relative_to(root)) for c in selected}, {str(c.path.relative_to(root)): c.reason for c in kept}


def test_selects_only_idle_scratch_and_never_shared_stores(cache):
    selected, kept = chosen(cache)
    assert selected == {'builds/idle', 'builds/active/head/tmp', 'builds/active/head/target-cpu-gate',
                        'builds/active/base/tmp', 'wip-run/slot/serve-default', 'wip-run/inst/slot/serve-x',
                        'wip-run/slot2.tmp'}
    assert kept == {}
    # Fresh anywhere inside keeps the entry; shared stores are never even candidates.
    assert not any(p.startswith(('builds/fresh', 'builds/compiler-cache', 'builds/codex-runs', 'wip-run/live'))
                   for p in selected)
    roots = {str(c.path.relative_to(cache)).split('/')[0] for c in gc.candidates(cache)}
    assert roots == {'builds', 'wip-run'}


def test_days_threshold(cache):
    selected, _ = chosen(cache, days=60)
    assert selected == set()


def test_directory_mtime_counts_too(cache):
    touch(cache / 'builds/idle/target/debug', NOW)  # a file was just deleted there: still active
    selected, _ = chosen(cache)
    assert 'builds/idle' not in selected


@pytest.mark.parametrize('mount, protected', [
    ('builds/idle', 'builds/idle'),                       # is the mount source
    ('builds/idle/target', 'builds/idle'),                # contains the mount source
    ('builds', 'builds/idle'),                            # inside the mount source
    ('builds/active/head/tmp/pytest', 'builds/active/head/tmp'),
])
def test_container_mount_sources_are_never_deleted(cache, mount, protected):
    env = gc.Environment(mounts=[gc.real(cache / mount)])
    selected, kept = chosen(cache, env)
    assert protected not in selected
    assert 'mounted by a Docker container' in kept[protected]


def test_unrelated_mount_does_not_protect(cache):
    env = gc.Environment(mounts=[gc.real(cache / 'builds/idle-not')], busy=[gc.real(cache / 'builds/idler')])
    selected, _ = chosen(cache, env)
    assert 'builds/idle' in selected  # path-component match, not string prefix


def test_process_cwd_or_open_file_protects(cache):
    env = gc.Environment(busy=[gc.real(cache / 'builds/idle/target/debug')])
    selected, kept = chosen(cache, env)
    assert 'builds/idle' not in selected and 'in use by a process' in kept['builds/idle']


def test_held_lock_inside_protects(cache):
    inode = (cache / 'builds/idle/target/debug/big.rlib').stat().st_ino
    selected, kept = chosen(cache, gc.Environment(held_inodes={inode}))
    assert 'builds/idle' not in selected and 'locked file' in kept['builds/idle']


def test_live_process_scan_sees_cwd_and_open_files(cache):
    target = cache / 'builds/idle/target/debug'
    held = cache / 'builds/active/head/tmp/pytest/popen-gw0/test_x0/blob'
    process = subprocess.Popen([sys.executable, '-c', f'f = open({str(held)!r}); import time; time.sleep(30)'],
                               cwd=target)
    try:
        deadline = time.time() + 10
        while time.time() < deadline:
            busy, _ = gc.process_paths(cache)
            if gc.real(held) in busy:
                break
            time.sleep(0.05)
        assert gc.real(target) in busy and gc.real(held) in busy
    finally:
        process.kill()
        process.wait()


def test_held_flock_is_found_in_proc_locks(cache):
    path = cache / 'builds/idle/target/debug/big.rlib'
    with open(path) as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        assert path.stat().st_ino in gc.held_lock_inodes()


def test_unreadable_or_foreign_directories_are_kept(cache):
    locked = cache / 'builds/idle/target'
    locked.chmod(0o000)
    try:
        selected, kept = chosen(cache)
    finally:
        locked.chmod(0o755)
    if os.getuid() == 0:
        pytest.skip('root reads everything')
    assert 'builds/idle' not in selected and 'unreadable' in kept['builds/idle']


def test_remove_handles_read_only_directories(tmp_path):
    tree = tmp_path / 'tree'
    make(tree / 'a/b/c')
    (tree / 'a/b').chmod(0o500)
    (tree / 'a').chmod(0o500)
    gc.remove(tree)
    assert not tree.exists()


def run_script(cache, *args, path=None):
    env = {**os.environ, 'PATH': path if path is not None else os.environ['PATH']}
    return subprocess.run([sys.executable, str(SCRIPT), '--root', str(cache), '--no-docker-prune', '--no-placement-prune', *args],
                          capture_output=True, text=True, env=env, timeout=120)


def fake_docker(tmp_path, mounts, fail=False):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    docker = bin_dir / 'docker'
    lines = '\n'.join(str(m) for m in mounts)
    docker.write_text('#!/bin/sh\n' + ('exit 1\n' if fail else '') +
                      f'case "$1" in ps) echo abc ;; inspect) printf "%s\\n" "{lines}" ;; esac\n')
    docker.chmod(0o755)
    return f'{bin_dir}:{os.environ["PATH"]}'


def test_dry_run_deletes_nothing_and_logs_nothing(cache, tmp_path):
    result = run_script(cache, '--dry-run', path=fake_docker(tmp_path, []))
    assert result.returncode == 0, result.stderr
    assert 'would delete build' in result.stdout and str(cache / 'builds/idle') in result.stdout
    assert (cache / 'builds/idle').exists() and not (cache / 'cache-gc.log').exists()


def test_run_deletes_idle_keeps_mounted_and_logs_one_line(cache, tmp_path):
    result = run_script(cache, path=fake_docker(tmp_path, [cache / 'wip-run/slot/serve-default']))
    assert result.returncode == 0, result.stderr
    assert not (cache / 'builds/idle').exists()
    assert not (cache / 'builds/active/head/tmp').exists()
    assert (cache / 'builds/active/STATUS.md').exists() and (cache / 'builds/fresh').exists()
    assert (cache / 'wip-run/slot/serve-default/bin/cuteafd').exists()  # mounted by a container
    assert not (cache / 'wip-run/inst').exists()  # emptied, then pruned
    for store in ('bench', 'fidelity', 'console', 'instances', 'jit', 'kache', 'sccache', 'cargo-home', 'hwq',
                  'tools', 'wip-slots', 'run-config', 'gpu1.lock', 'builds/compiler-cache', 'builds/codex-runs'):
        assert (cache / store).exists(), store
    log = (cache / 'cache-gc.log').read_text().splitlines()
    assert len(log) == 1 and 'deleted 6/6' in log[0] and 'freed' in log[0] and '(df)' in log[0]
    assert 'kept 1' in log[0]


def test_docker_failure_aborts_without_deleting(cache, tmp_path):
    result = run_script(cache, path=fake_docker(tmp_path, [], fail=True))
    assert result.returncode == 2
    assert 'deleting nothing' in result.stderr
    assert (cache / 'builds/idle').exists()

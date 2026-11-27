"""scripts/dev/prune-placement-inputs: keep every referenced publication, delete only the rest."""
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/dev/prune-placement-inputs'
LOADER = importlib.machinery.SourceFileLoader('prune_placement_inputs', str(SCRIPT))
SPEC = importlib.util.spec_from_loader('prune_placement_inputs', LOADER)
prune = importlib.util.module_from_spec(SPEC)
sys.modules['prune_placement_inputs'] = prune
LOADER.exec_module(prune)

NOW = time.time()


def sha(index):
    return f'{index:040x}'


def publish(store, name, commit, age_days):
    path = store / name
    (path / 'share').mkdir(parents=True)
    (path / 'INPUTS.json').write_text(json.dumps({'schema': 1, 'engine_commit': commit, 'files': {}}, indent=2))
    (path / 'share/PROGRAMS.json').write_text('{}')
    stamp = NOW - age_days * 86400
    for item in (path / 'share/PROGRAMS.json', path / 'share', path / 'INPUTS.json', path):
        os.utime(item, (stamp, stamp), follow_symlinks=False)
    return path


@pytest.fixture
def cache(tmp_path):
    root = tmp_path / 'cuteafd'
    store = root / 'builds/placement-inputs'
    v4 = root / 'builds/v4-placement-inputs'
    for index in range(1, 9):  # 1 newest .. 8 oldest
        publish(store, sha(index), sha(index), age_days=index)
    publish(v4, sha(101), sha(101), age_days=1)
    publish(v4, sha(102), sha(102), age_days=2)
    (store / 'current').symlink_to(sha(6))
    (v4 / 'current').symlink_to(sha(102))
    (store / 'by-key').mkdir()
    (store / 'by-key/abc').symlink_to(f'../{sha(7)}')
    # Not publications: staging, logs, expect files, and a link escaping the store.
    (store / 'staging-x/programs').mkdir(parents=True)
    (store / 'export.log').write_text('log')
    outside = tmp_path / 'outside'
    publish(outside, sha(200), sha(200), age_days=30)
    (store / sha(200)).symlink_to(outside / sha(200))
    # A link inside a doomed publication must not be followed by the delete.
    (store / sha(8) / 'share/escape').symlink_to(outside / sha(200))
    # A sealed WIP slot built from commit 5.
    slot = root / 'builds/wip-x/slots/c/coordinator'
    (slot / 'workspace').mkdir(parents=True)
    (slot / 'META.json').write_text('{}')
    (slot / 'workspace/BUILD_IDENTITY.json').write_text(json.dumps({'commit': sha(5), 'dirty': False}))
    return root


def references(cache, heads=None, mounts=(), busy=()):
    return prune.References(heads=heads or {}, slots=prune.sealed_slot_commits(cache / 'builds'),
                            gc=prune.GC.Environment(mounts=[Path(os.path.realpath(m)) for m in mounts],
                                                    busy=[Path(os.path.realpath(b)) for b in busy]))


def names(publications):
    return {p.path.name for p in publications}


def test_keeps_links_open_prs_sealed_slots_mounts_processes_and_newest(cache):
    store = cache / 'builds/placement-inputs'
    chosen, kept = prune.plan(cache, 2, references(
        cache, heads={sha(4): 2900}, mounts=[store / sha(3) / 'share'], busy=[store / sha(8) / 'share/PROGRAMS.json']))
    reasons = {p.path.name: p.reason for p in kept}
    assert names(chosen) == set()  # everything in this fixture is referenced
    assert reasons[sha(1)] == reasons[sha(2)] == 'among the 2 newest'
    assert 'mounted by a Docker container' in reasons[sha(3)]
    assert reasons[sha(4)] == 'commit on open PR #2900'
    assert reasons[sha(5)] == 'source of sealed WIP slot wip-x/slots/c/coordinator'
    assert reasons[sha(6)] == 'target of current'
    assert reasons[sha(7)] == 'target of by-key/abc'
    assert 'in use by a process' in reasons[sha(8)]
    assert sha(102) in reasons and reasons[sha(101)] == 'among the 2 newest'


def test_unreferenced_publications_are_chosen_with_sizes_and_nothing_else(cache):
    chosen, kept = prune.plan(cache, 2, references(cache))
    assert names(chosen) == {sha(3), sha(4), sha(8)}
    assert all(p.allocated > 0 and p.store == 'placement-inputs' for p in chosen)
    assert sha(200) not in names(chosen) | names(kept)  # a symlinked entry is never a publication


def run(cache, monkeypatch, *args, heads=None, mounts=None):
    monkeypatch.setattr(prune, 'open_pr_commits', lambda repository=prune.REPOSITORY: heads)
    monkeypatch.setattr(prune.GC, 'docker_mounts', lambda: mounts)
    monkeypatch.setattr(prune.GC, 'process_paths', lambda root, ignore_own_cmdline=False: ([], 0))
    return prune.main(['--root', str(cache), '--keep-newest', '2', *args])


def test_dry_run_is_the_default_and_deletes_nothing(cache, monkeypatch, capsys):
    assert run(cache, monkeypatch, heads={}, mounts=[]) == 0
    out = capsys.readouterr().out
    assert sum(line.startswith('would delete ') for line in out.splitlines()) == 3 and 'would delete 3' in out
    assert all((cache / 'builds/placement-inputs' / sha(i)).is_dir() for i in range(1, 9))
    assert not (cache / 'cache-gc.log').exists()


def test_delete_removes_only_unreferenced_and_never_current_or_outside(cache, monkeypatch, tmp_path):
    store = cache / 'builds/placement-inputs'
    assert run(cache, monkeypatch, '--delete', heads={sha(4): 1}, mounts=[]) == 0
    assert not (store / sha(3)).exists() and not (store / sha(8)).exists()
    for index in (1, 2, 4, 5, 6, 7):
        assert (store / sha(index) / 'INPUTS.json').is_file()
    assert (store / 'current').is_symlink() and os.readlink(store / 'current') == sha(6)
    assert (store / 'by-key/abc').is_symlink()
    assert (store / sha(200)).is_symlink() and (tmp_path / 'outside' / sha(200) / 'INPUTS.json').is_file()
    assert (store / 'staging-x/programs').is_dir() and (store / 'export.log').is_file()
    assert (cache / 'builds/v4-placement-inputs' / sha(102)).is_dir()
    log = (cache / 'cache-gc.log').read_text().splitlines()
    assert len(log) == 1 and 'deleted 2/2' in log[0]


@pytest.mark.parametrize('heads,mounts', [(None, []), ({}, None)])
def test_unlistable_prs_or_containers_delete_nothing(cache, monkeypatch, heads, mounts):
    assert run(cache, monkeypatch, '--delete', heads=heads, mounts=mounts) == 2
    assert all((cache / 'builds/placement-inputs' / sha(i)).is_dir() for i in range(1, 9))


def test_cache_gc_runs_the_prune_dry_unless_deleting(cache, tmp_path):
    import subprocess
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    for tool, body in (('docker', 'case "$1" in ps) ;; esac\n'), ('gh', 'echo \'{"data":{"repository":{"pullRequests":{"pageInfo":{"hasNextPage":false},"nodes":[]}}}}\'\n')):
        (bin_dir / tool).write_text('#!/bin/sh\n' + body)
        (bin_dir / tool).chmod(0o755)
    env = {**os.environ, 'PATH': f'{bin_dir}:{os.environ["PATH"]}'}
    for index in range(300, 310):  # ten newer publications push the old ones past --keep-newest 10
        publish(cache / 'builds/placement-inputs', sha(index), sha(index), age_days=0.1)
    command = [sys.executable, str(ROOT / 'scripts/dev/cache-gc'), '--root', str(cache), '--no-docker-prune',
               '--days', '10000']
    dry = subprocess.run([*command, '--dry-run'], capture_output=True, text=True, env=env, timeout=120)
    assert dry.returncode == 0, dry.stderr
    assert 'placement publications: would delete' in dry.stdout
    assert (cache / 'builds/placement-inputs' / sha(8)).is_dir()
    real = subprocess.run(command, capture_output=True, text=True, env=env, timeout=120)
    assert real.returncode == 0, real.stderr
    assert not (cache / 'builds/placement-inputs' / sha(8)).exists()
    assert (cache / 'builds/placement-inputs' / sha(6)).is_dir()  # current
    assert (cache / 'builds/placement-inputs' / sha(5)).is_dir()  # sealed slot


def test_open_pr_rule_covers_branch_commits_and_force_pushed_heads(monkeypatch):
    import subprocess
    pages = [
        {'pageInfo': {'hasNextPage': True, 'endCursor': 'c1'}, 'nodes': [
            {'number': 2877, 'headRefOid': sha(1), 'commits': {'totalCount': 2, 'nodes': [
                {'commit': {'oid': sha(2)}}, {'commit': {'oid': sha(1)}}]},
             'timelineItems': {'nodes': [{'beforeCommit': {'oid': sha(3)}, 'afterCommit': {'oid': sha(1)}}, {}]}}]},
        {'pageInfo': {'hasNextPage': False, 'endCursor': None}, 'nodes': [
            {'number': 2878, 'headRefOid': sha(4), 'commits': {'totalCount': 1, 'nodes': [{'commit': {'oid': sha(4)}}]},
             'timelineItems': {'nodes': []}}]}]
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        page = pages[len(calls) - 1]
        return subprocess.CompletedProcess(command, 0, json.dumps(
            {'data': {'repository': {'pullRequests': page}}}), '')

    monkeypatch.setattr(prune.subprocess, 'run', fake_run)
    assert prune.open_pr_commits() == {sha(1): 2877, sha(2): 2877, sha(3): 2877, sha(4): 2878}
    assert 'after=c1' in calls[1]
    pages[1]['nodes'][0]['commits']['totalCount'] = 400  # truncated history refuses, never under-protects
    calls.clear()
    assert prune.open_pr_commits() is None
    monkeypatch.setattr(prune.subprocess, 'run', lambda command, **kwargs: subprocess.CompletedProcess(command, 1, '', 'x'))
    assert prune.open_pr_commits() is None

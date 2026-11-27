"""scripts/dev/land-pr: verify, merge, then promote current on every host, fail-closed."""
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
LOADER = importlib.machinery.SourceFileLoader('land_pr', str(ROOT / 'scripts/dev/land-pr'))
SPEC = importlib.util.spec_from_loader('land_pr', LOADER)
land = importlib.util.module_from_spec(SPEC)
sys.modules['land_pr'] = land
LOADER.exec_module(land)

FINGERPRINT = {'sparkinfer_revision': 'f' * 40, 'source_tree_sha256': 'e' * 64,
               'engine_sources_sha256': {'python/tools/aot/a.py': '1' * 64}}


def git(cwd, *args):
    return subprocess.run(['git', '-C', str(cwd), *args], check=True, capture_output=True, text=True,
                          env={**os.environ, 'GIT_AUTHOR_NAME': 't', 'GIT_AUTHOR_EMAIL': 't@t',
                               'GIT_COMMITTER_NAME': 't', 'GIT_COMMITTER_EMAIL': 't@t'}).stdout.strip()


def commit(work, name):
    (work / name).write_text(name)
    git(work, 'add', name)
    git(work, 'commit', '-qm', name)
    return git(work, 'rev-parse', 'HEAD')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(store, name, engine_commit, fingerprint=FINGERPRINT, changes=None):
    path = store / name
    (path / 'share').mkdir(parents=True)
    (path / 'share/PROGRAMS.json').write_text('{"programs": []}')
    if changes is not None:
        (path / 'CHANGES.json').write_text(json.dumps(changes))
    files = {'share/PROGRAMS.json': sha256(path / 'share/PROGRAMS.json')}
    (path / 'INPUTS.json').write_text(json.dumps({'schema': 1, 'engine_commit': engine_commit, **fingerprint,
                                                  'files': files}, indent=2))
    return sha256(path / 'INPUTS.json')


class FakeRemote(land.LocalTransport):
    """A second host: its own store directory, reached without ssh."""


@pytest.fixture
def world(tmp_path, monkeypatch):
    origin = tmp_path / 'origin.git'
    subprocess.run(['git', 'init', '-q', '--bare', '-b', 'dev', str(origin)], check=True)
    work = tmp_path / 'work'
    subprocess.run(['git', 'clone', '-q', str(origin), str(work)], check=True, capture_output=True)
    git(work, 'checkout', '-q', '-b', 'dev')
    base = commit(work, 'base')
    git(work, 'push', '-q', 'origin', 'dev')
    git(work, 'checkout', '-q', '-b', 'feature')
    head = commit(work, 'feature')
    git(work, 'push', '-q', 'origin', f'{head}:refs/pull/7/head')
    clone = tmp_path / 'coordinator'
    subprocess.run(['git', 'clone', '-q', str(origin), str(clone)], check=True, capture_output=True)
    monkeypatch.setattr(land, 'REPO', clone)
    local, remote = tmp_path / 'raptor/builds', tmp_path / 'maxq/builds'
    for builds in (local, remote):
        for store in land.STORES.values():
            (builds / store).mkdir(parents=True)
    old = publish(local / 'placement-inputs', 'a' * 40, base, fingerprint={**FINGERPRINT, 'source_tree_sha256': 'd' * 64})
    publish(remote / 'placement-inputs', 'a' * 40, base, fingerprint={**FINGERPRINT, 'source_tree_sha256': 'd' * 64})
    v4 = publish(local / 'v4-placement-inputs', 'b' * 40, base)
    publish(remote / 'v4-placement-inputs', 'b' * 40, base)
    for builds in (local, remote):
        (builds / 'placement-inputs/current').symlink_to('a' * 40)
        (builds / 'v4-placement-inputs/current').symlink_to('b' * 40)
    candidate = publish(local / 'placement-inputs', head, head)
    merged = []

    def merge_pr(number, sha):
        merged.append((number, sha))
        git(work, 'checkout', '-q', 'dev')
        git(work, 'merge', '-q', '--no-ff', '-m', f'Merge #{number}', sha)
        git(work, 'push', '-q', 'origin', 'dev')

    pr = {'state': 'OPEN', 'isDraft': False, 'baseRefName': 'dev', 'headRefOid': head}
    hosts = [land.Host('raptor', local, land.LocalTransport()), land.Host('maxq', remote, FakeRemote())]
    return SimpleNamespace(base=base, head=head, local=local, remote=remote, candidate=candidate, old=old, v4=v4,
                           work=work, origin=origin, merged=merged, merge_pr=merge_pr, pr=pr, hosts=hosts)


def args(world, **overrides):
    values = dict(pr=7, match_head_commit=world.head, publication=world.head, inputs_sha256=world.candidate,
                  v4_publication=None, v4_inputs_sha256=None, dry_run=False)
    return SimpleNamespace(**{**values, **overrides})


def run(world, fingerprints=None, hosts=None, **overrides):
    expected = fingerprints or {'all': FINGERPRINT, 'v4': FINGERPRINT}
    return land.run(args(world, **overrides), hosts or world.hosts, fingerprints=lambda head: expected,
                    view=lambda number: world.pr, merge_pr=world.merge_pr)


def link(builds, store='placement-inputs'):
    return os.readlink(builds / store / 'current')


def test_lands_copies_to_the_remote_and_flips_both_hosts(world, capsys):
    assert run(world) == 0
    assert world.merged == [(7, world.head)]
    out = capsys.readouterr()
    dev = git(world.origin, 'rev-parse', 'dev')
    assert f'dev {dev}' in out.out
    for builds in (world.local, world.remote):
        assert link(builds) == world.head and link(builds, 'v4-placement-inputs') == 'b' * 40
        assert sha256(builds / 'placement-inputs/current/INPUTS.json') == world.candidate
    assert not list(world.remote.glob('placement-inputs/.*tmp*'))
    assert f'maxq:{world.remote}/placement-inputs/current -> {world.head}' in out.out


def test_unchanged_inputs_keep_current_and_still_require_both_hosts_to_agree(world):
    assert run(world, fingerprints={'all': {**FINGERPRINT, 'source_tree_sha256': 'd' * 64}, 'v4': FINGERPRINT},
               publication=None, inputs_sha256=None) == 0
    assert link(world.local) == link(world.remote) == 'a' * 40


def test_changed_inputs_without_a_named_publication_refuse_before_merging(world, capsys):
    assert run(world, publication=None, inputs_sha256=None) == 2
    assert 'changes placement-inputs inputs' in capsys.readouterr().err
    assert world.merged == [] and link(world.local) == 'a' * 40


@pytest.mark.parametrize('mutation,message', [
    ('head', 'head is'), ('draft', 'draft'), ('base', 'targets'), ('stale', 'does not contain origin/dev'),
    ('sha', 'INPUTS.json sha256'), ('commit', 'was published from'), ('pin', 'changes placement-inputs inputs'),
    ('leaks', 'leaks'), ('modified', 'listed file differs'), ('remote-differs', 'differs'), ('current', 'not a real publication directory'),
])
def test_every_failed_check_refuses_before_merging(world, capsys, mutation, message):
    overrides, fingerprints = {}, None
    if mutation == 'head':
        world.pr['headRefOid'] = '0' * 40
    elif mutation == 'draft':
        world.pr['isDraft'] = True
    elif mutation == 'base':
        world.pr['baseRefName'] = 'main'
    elif mutation == 'stale':
        git(world.work, 'checkout', '-q', 'dev')
        commit(world.work, 'concurrent')
        git(world.work, 'push', '-q', 'origin', 'dev')
    elif mutation == 'sha':
        overrides['inputs_sha256'] = '0' * 64
    elif mutation == 'commit':
        other = git(world.work, 'rev-parse', 'dev')
        overrides.update(publication='c' * 40, inputs_sha256=publish(world.local / 'placement-inputs', 'c' * 40, other))
    elif mutation == 'pin':
        fingerprints = {'all': {**FINGERPRINT, 'sparkinfer_revision': '9' * 40}, 'v4': FINGERPRINT}
    elif mutation == 'leaks':
        name = 'd' * 40
        overrides.update(publication=name, inputs_sha256=publish(
            world.local / 'placement-inputs', name, world.head, changes={'leaks': ['x'], 'sm170_mismatches': []}))
    elif mutation == 'modified':
        (world.local / 'placement-inputs' / world.head / 'share/PROGRAMS.json').write_text('changed')
    elif mutation == 'remote-differs':
        publish(world.remote / 'placement-inputs', world.head, world.head)
        (world.remote / 'placement-inputs' / world.head / 'INPUTS.json').write_text('{}')
    elif mutation == 'current':
        overrides['publication'] = 'current'
    assert run(world, fingerprints, **overrides) == 2
    assert message in capsys.readouterr().err
    assert world.merged == []
    assert link(world.local) == link(world.remote) == 'a' * 40


def test_dry_run_checks_and_changes_nothing(world, capsys):
    assert run(world, dry_run=True) == 0
    assert 'would copy' in capsys.readouterr().err
    assert world.merged == [] and not (world.remote / 'placement-inputs' / world.head).exists()
    assert link(world.local) == 'a' * 40


def test_failed_remote_copy_leaves_no_partial_publication_and_does_not_merge(world, capsys):
    class Broken(FakeRemote):
        def copy(self, source, destination):
            super().copy(source, destination)
            (Path(destination) / 'share/PROGRAMS.json').write_text('truncated')

    hosts = [world.hosts[0], land.Host('maxq', world.remote, Broken())]
    assert run(world, hosts=hosts) == 2
    assert 'failed verification' in capsys.readouterr().err
    assert world.merged == [] and not list((world.remote / 'placement-inputs').glob(f'*{world.head}*'))


def test_concurrent_merge_is_detected_and_current_is_not_flipped(world, capsys):
    def racing_merge(number, sha):
        git(world.work, 'checkout', '-q', 'dev')
        commit(world.work, 'raced')
        git(world.work, 'merge', '-q', '--no-ff', '-m', 'merge', sha)
        git(world.work, 'push', '-q', 'origin', 'dev')

    world.merge_pr = racing_merge
    assert run(world) == 2
    assert 'another merge landed concurrently' in capsys.readouterr().err
    assert link(world.local) == link(world.remote) == 'a' * 40


def test_hosts_that_disagree_after_flipping_are_not_success(world, capsys):
    class Lagging(FakeRemote):
        def run(self, argv, stdin):
            if argv[2] == 'flip':
                return subprocess.CompletedProcess(argv, 0, json.dumps({'ok': True}), '')
            return super().run(argv, stdin)

    hosts = [world.hosts[0], land.Host('maxq', world.remote, Lagging())]
    assert run(world, hosts=hosts) == 3
    assert 'hosts disagree' in capsys.readouterr().err
    assert link(world.local) == world.head and link(world.remote) == 'a' * 40


def test_ssh_transport_runs_ops_remotely_and_rsyncs_into_a_temporary(world, tmp_path, monkeypatch, capsys):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    calls = tmp_path / 'calls'
    # ssh -o BatchMode=yes HOST 'COMMAND': run COMMAND here; rsync SRC/ HOST:DST/: strip the host.
    (bin_dir / 'ssh').write_text(f'#!/bin/sh\necho "ssh $3" >> {calls}\nshift 3\nexec sh -c "$1"\n')
    (bin_dir / 'rsync').write_text(f'#!/bin/sh\necho "rsync $*" >> {calls}\n'
                                   'eval last=\\${$#}\nset -- $(echo "$@" | sed "s/ maxq:/ /")\n'
                                   f'exec {subprocess.check_output(["which", "rsync"], text=True).strip()} "$@"\n')
    for tool in ('ssh', 'rsync'):
        (bin_dir / tool).chmod(0o755)
    monkeypatch.setenv('PATH', f'{bin_dir}:{os.environ["PATH"]}')
    hosts = [world.hosts[0], land.Host('maxq', world.remote, land.SshTransport('maxq'))]
    assert run(world, hosts=hosts) == 0, capsys.readouterr().err
    assert link(world.remote) == world.head
    assert sha256(world.remote / 'placement-inputs/current/INPUTS.json') == world.candidate
    log = calls.read_text()
    assert 'ssh maxq' in log and f'maxq:{world.remote}/placement-inputs/.{world.head}.tmp-' in log


def test_exporters_refuse_to_write_a_current_link_or_its_target(tmp_path):
    publisher = land  # noqa: F841  (keep the module import order stable)
    spec = importlib.util.spec_from_file_location('guard_publisher', ROOT / 'scripts/build/export-placement-inputs.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    store = tmp_path / 'placement-inputs'
    (store / ('a' * 40)).mkdir(parents=True)
    (store / 'current').symlink_to('a' * 40)
    for path in (store / 'current', store / 'current/share', store / ('a' * 40), store / ('a' * 40) / 'x'):
        with pytest.raises(ValueError, match='land-pr'):
            module.refuse_current(path)
    module.refuse_current(store / ('b' * 40), store / f'staging-{"b" * 40}')
    for script, argv in (('export-placement-inputs.py', ['--output', str(store / 'current'), '--staging', str(tmp_path / 's')]),
                         ('compare-sm-exports.py', ['--v4-placement-inputs', str(store / 'current'),
                                                    '--output-dir', str(tmp_path / 's')])):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/build' / script), *argv],
                                capture_output=True, text=True, timeout=120)
        assert result.returncode != 0 and 'land-pr' in result.stderr, (script, result.stderr[-400:])
        assert os.readlink(store / 'current') == 'a' * 40

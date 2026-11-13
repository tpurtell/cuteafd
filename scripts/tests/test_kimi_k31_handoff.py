"""Native storage failures must preserve the published quant's recovery links."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'quantization'))
from kimi_k31 import handoff_k2_core_k6 as storage


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    for key, relative in (('BASE', 'task'), ('ROOT', 'task/production'),
                          ('EXPORT', 'export'), ('LOCAL_HUB', 'hub'),
                          ('STAGE_PARENT', 'stage'), ('BUILD_ROOT', 'build'),
                          ('MANAGED', 'managed')):
        directory = tmp_path / relative
        directory.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(storage, key, directory)
    root, export = storage.ROOT, storage.EXPORT
    revision, identity = 'ab' * 20, {'producer': 'frozen'}
    snapshot = storage.LOCAL_HUB / storage.REPO_NAME / 'snapshots' / revision
    snapshot.mkdir(parents=True)
    files = {}
    for name in [f'model-{i:05d}.safetensors' for i in range(16)] + ['README.md', 'errors/index.json']:
        path = export / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b'weights')
        other = snapshot / name
        other.parent.mkdir(exist_ok=True)
        os.link(path, other)
        files[name] = dict(path=path, size=7, sha256=hashlib.sha256(b'weights').hexdigest(),
                           blob_id=None, file_identity=storage.file_identity(path))
    write(root / 'plan.json', dict(root=str(root), export=str(export), identity=identity,
                                   publication_repo=storage.TARGET_REPO))
    write(root / 'publication-complete.json', dict(identity=identity, repo_id=storage.TARGET_REPO,
        revision=revision, manifest_sha256=storage._manifest_digest(files),
        local_snapshot=str(snapshot), files=len(files)))
    ref = storage.MANAGED / storage.REPO_NAME / 'refs/main'
    ref.parent.mkdir(parents=True)
    ref.write_text(revision)
    managed_source = storage.MANAGED / 'models--source--keep'
    managed_source.mkdir()
    (managed_source / 'source.safetensors').write_bytes(b'preserve')
    (root / 'evidence').mkdir()
    (root / 'evidence/receipt.json').write_text('keep')
    for leaf in ('frontier', 'head/stages', 'layers/092/expert-projections'):
        path = root / leaf
        path.mkdir(parents=True)
        (path / 'obsolete').write_bytes(b'recovery')
    behavior = dict(import_error=False, placement_error=False, copied=False, bad_residency=False,
                    busy=False, offgrid=True, offgrid_error=False, uneven=False)
    calls = []
    plan = dict(id=19, feasible=True, blocked=[], goal=dict(goal='place',
        selector='hf:' + storage.TARGET_REPO, hosts=list(storage.TARGETS), factor=1/8,
        by_speed=False), steps=[dict(kind='replicate', host=storage.TARGETS[0],
                                    copies=[dict(group=storage.TARGET_REPO)])])

    def native(*args):
        calls.append(args)
        if args[:2] == ('hf', 'import'):
            assert '--offline' in args and '--no-copy' in args
            return {'job': 11}
        if args[:2] == ('plan', '--place'):
            assert args.count('--host') == 8 and args[args.index('--factor') + 1] == '1/8'
            return plan
        if args[:2] == ('plan', '--offgrid'):
            return dict(id=20, feasible=True, blocked=[], goal=dict(goal='offgrid',
                selector='hf:' + storage.TARGET_REPO, hosts=[]),
                steps=[dict(kind='evict', host='maxq', copies=[dict(group=storage.TARGET_REPO)])])
        if args[:2] == ('plan', 'apply'):
            return {'job': 12 if args[2] == 19 else 13}
        if args[0] == 'jobs':
            error = behavior[{11: 'import_error', 12: 'placement_error', 13: 'offgrid_error'}[args[1]]]
            result = dict(job=dict(finished=True, error='failed' if error else None,
                                   cancelled=False, pending=0, hosts={}))
            if args[1] == 11:
                result['import'] = dict(finished=True, errors=[], copied_bytes=int(behavior['copied']),
                                        files=len(files), adopted=0, skipped=0)
            if args[1] == 13 and not error:
                behavior['offgrid'] = False
            return result
        assert args[0] == 'ls'
        if Path(args[1]).name == 'refs':
            return dict(entries=[dict(name='main', kind='regular', writing_on=None,
                hosts=[storage.TARGETS[0]] + (['maxq'] if behavior['offgrid'] else []))])
        directory = Path(args[1]).relative_to(storage.NAMESPACE / storage.REPO_NAME / 'snapshots' / revision)
        rows = []
        for i, (name, value) in enumerate(files.items()):
            if Path(name).parent != directory:
                continue
            hosts = [storage.TARGETS[i % 8]] + (['maxq'] if behavior['offgrid'] else [])
            if behavior['bad_residency'] and i == 0:
                hosts = ['raptor']
            if behavior['uneven']:
                hosts = [storage.TARGETS[0]]
            rows.append(dict(name=Path(name).name, kind='regular', size=value['size'],
                             sealed=True, writing_on=None, hosts=hosts))
        return dict(entries=rows)

    def unused(paths):
        if behavior['busy']:
            raise RuntimeError('still used by a process')

    monkeypatch.setattr(storage, 'native', native)
    monkeypatch.setattr(storage, 'unused', unused)
    monkeypatch.setattr(storage, '_complete_plan', lambda *a: identity)
    monkeypatch.setattr(storage, '_collect_local', lambda *a: files)
    # Test pipeline state independently of host filesystem clone support.
    def stage(snapshot, files, revision):
        path = storage.STAGE_PARENT / storage.REPO_NAME
        path.mkdir(exist_ok=True)
        return path
    monkeypatch.setattr(storage, 'stage_snapshot', stage)
    return root, export, snapshot, calls, behavior, plan, managed_source


def test_native_nocopy_eight_places_cleanup_and_idempotency(artifact):
    root, export, snapshot, calls, _, _, managed_source = artifact
    result = storage.handoff(root)
    assert result['copied_bytes'] == 0
    assert set(result['distribution_bytes']) == set(storage.TARGETS)
    assert set(result['distribution_bytes'].values()) == {14}
    assert not export.exists() and not snapshot.exists()
    assert not (root / 'frontier').exists()
    assert (root / 'evidence/receipt.json').read_text() == 'keep'
    assert (managed_source / 'source.safetensors').read_bytes() == b'preserve'
    count = len(calls)
    assert storage.handoff(root) == result
    assert len(calls) == count


@pytest.mark.parametrize('failure', ['import_error', 'placement_error', 'copied', 'bad_residency',
                                    'busy', 'offgrid_error', 'uneven'])
def test_failure_preserves_outputs_and_resumes_native_jobs(artifact, failure):
    root, export, snapshot, calls, behavior, _, _ = artifact
    behavior[failure] = True
    with pytest.raises((ValueError, RuntimeError)):
        storage.handoff(root)
    assert export.exists() and snapshot.exists() and (root / 'frontier').exists()
    assert not (root / 'handoff-complete.json').exists()
    imports = sum(c[:2] == ('hf', 'import') for c in calls)
    behavior[failure] = False
    assert storage.handoff(root)['status'] == 'passed'
    assert sum(c[:2] == ('hf', 'import') for c in calls) == imports


@pytest.mark.parametrize('damage', ['identity', 'other_model', 'other_hosts', 'other_disk', 'weighted'])
def test_wrong_context_or_plan_cannot_apply_or_retire(artifact, damage):
    root, export, _, calls, _, plan, _ = artifact
    if damage == 'identity':
        path = root / 'publication-complete.json'
        value = json.loads(path.read_text())
        value['identity'] = {'producer': 'changed'}
        write(path, value)
    elif damage == 'other_model':
        plan['steps'][0]['copies'][0]['group'] = 'another/model'
    elif damage == 'other_hosts':
        plan['goal']['hosts'] = ['raptor']
    elif damage == 'other_disk':
        plan['steps'][0]['host'] = 'raptor'
    else:
        plan['goal']['by_speed'] = True
    with pytest.raises(ValueError):
        storage.handoff(root)
    assert export.exists()
    assert not any(c[:2] == ('plan', 'apply') for c in calls)


def test_no_rehash_and_no_copy_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, 'STAGE_PARENT', tmp_path / 'staging')
    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    source = snapshot / 'weight.safetensors'
    source.write_bytes(b'weights')
    (snapshot / 'second.safetensors').write_bytes(b'weights')
    files = {'weight.safetensors': dict(size=7, sha256='known-receipt',
                                       file_identity=storage.file_identity(source))}
    calls = []

    def clone(command, **kwargs):
        calls.append(command)
        assert '--reflink=always' in command
        Path(command[-1]).write_bytes(Path(command[-2]).read_bytes())

    monkeypatch.setattr(storage.subprocess, 'run', clone)
    stage = storage.stage_snapshot(snapshot, files, 'ab' * 20)
    # Existing receipts permit metadata-only resume; opening weight bytes fails.
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        if path.suffix == '.safetensors':
            raise AssertionError('handoff must not reread model weights')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)
    assert storage.stage_snapshot(snapshot, files, 'ab' * 20) == stage
    assert len(calls) == 1
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr(storage.subprocess, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        storage.stage_snapshot(snapshot, {'second.safetensors': dict(size=7, sha256='receipt',
            file_identity=storage.file_identity(snapshot / 'second.safetensors'))}, 'ab' * 20)


def test_native_handoff_never_opens_weight_payloads(artifact, monkeypatch):
    root, *_ = artifact
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        if path.suffix == '.safetensors':
            raise AssertionError('storage handoff must use receipts and metadata')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)
    assert storage.handoff(root)['status'] == 'passed'


def test_cleanup_rejects_an_open_payload(tmp_path):
    path = tmp_path / 'live.payload'
    path.write_bytes(b'in use')
    with path.open('rb'):
        with pytest.raises(RuntimeError, match='still used by a process'):
            storage.unused([tmp_path])


def test_incomplete_native_host_job_is_not_success(monkeypatch):
    monkeypatch.setattr(storage, 'native', lambda *a: dict(job=dict(finished=True,
        error=None, pending=0, hosts={'moa': dict(finished=True, failed=[],
        done_files=1, total_files=1, done_bytes=3, total_bytes=7)})))
    with pytest.raises(ValueError, match='job failed or is incomplete'):
        storage.wait_job(12)

"""Native import/placement failures cannot authorize local weight retirement."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'quantization'))
from mistral_large4.core_k6 import archive as storage


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    root = tmp_path / 'run'
    root.mkdir()
    export = tmp_path / 'export'
    export.mkdir()
    revision = 'ab' * 20
    repo_name = 'models--' + storage.TARGET_REPO.replace('/', '--')
    snapshot = tmp_path / 'hub' / repo_name / 'snapshots' / revision
    snapshot.mkdir(parents=True)
    for folder in (export, snapshot):
        for i in range(1, storage.SHARD_COUNT + 1):
            (folder / f'model-{i:05d}-of-{storage.SHARD_COUNT:05d}.safetensors').write_bytes(b'weight')
        (folder / 'README.md').write_text('evidence retained')
    identity = {'producer': 'frozen'}
    write(root / 'plan.json', dict(root=str(root), export=str(export), identity=identity,
                                 publication_repo=storage.TARGET_REPO))
    write(root / 'quantization-complete.json', dict(identity=identity, summary=dict(
        shards=storage.SHARD_COUNT, tensors=storage.TARGET_TENSOR_COUNT,
        payload_bytes=storage.TARGET_PAYLOAD_BYTES, quantized_projections=storage.MODULE_COUNT)))
    write(root / 'publication-complete.json', dict(identity=identity, repo_id=storage.TARGET_REPO,
        revision=revision, manifest_sha256='cd' * 32, local_snapshot=str(snapshot)))
    calls = []
    behavior = {'import_error': False, 'placement_error': False, 'bad_residency': False}
    plan = dict(id=19, feasible=True, blocked=[], goal=dict(goal='place',
        selector='hf:' + storage.TARGET_REPO, hosts=list(storage.HOSTS), factor=1/6),
        steps=[dict(copies=[dict(group=storage.TARGET_REPO)])])

    def native(*args):
        calls.append(args)
        if args[:2] == ('hf', 'import'):
            assert '--offline' in args and '--no-copy' not in args
            return {'job': 11}
        if args[:2] == ('plan', '--place'):
            assert args.count('--host') == 6
            return plan
        if args[:2] == ('plan', 'apply'):
            return {'job': 12}
        if args[0] == 'jobs':
            error = behavior['import_error'] if args[1] == 11 else behavior['placement_error']
            result = dict(job=dict(finished=True, error='failed' if error else None,
                                   cancelled=False, pending=0, hosts={}))
            if args[1] == 11:
                # Copying is allowed; verification belongs to SparkNest.
                result['import'] = dict(finished=True, errors=[], copied_bytes=999)
            return result
        assert args[0] == 'ls'
        rows = [dict(name=f'model-{i:05d}-of-{storage.SHARD_COUNT:05d}.safetensors',
            sealed=True, writing_on=None, hosts=[storage.HOSTS[i % 6]])
            for i in range(1, storage.SHARD_COUNT + 1)]
        if behavior['bad_residency']:
            rows[0]['hosts'] = ['raptor']
        return dict(entries=rows)

    monkeypatch.setattr(storage, '_native', native)
    return root, export, snapshot, calls, behavior, plan


def test_native_import_spread_and_idempotent_retirement(artifact):
    root, export, snapshot, calls, _, _ = artifact
    complete = storage.archive(root)
    assert complete['weight_shards'] == 73
    assert set(complete['distribution']) == set(storage.HOSTS)
    assert sum(complete['distribution'].values()) == 73
    assert not list(export.glob('*.safetensors'))
    assert not list(snapshot.glob('*.safetensors'))
    assert (snapshot / 'README.md').is_file() and (export / 'README.md').is_file()
    count = len(calls)
    assert storage.archive(root) == complete
    assert len(calls) == count


@pytest.mark.parametrize('failure', ['import_error', 'placement_error', 'bad_residency'])
def test_native_failure_keeps_sources_and_reuses_recorded_jobs(artifact, failure):
    root, export, _, calls, behavior, _ = artifact
    behavior[failure] = True
    with pytest.raises(ValueError):
        storage.archive(root)
    assert not (root / 'archive-complete.json').exists()
    assert len(list(export.glob('*.safetensors'))) == 73
    imports = sum(args[:2] == ('hf', 'import') for args in calls)
    behavior[failure] = False
    storage.archive(root)
    assert sum(args[:2] == ('hf', 'import') for args in calls) == imports


@pytest.mark.parametrize('damage', ['identity', 'other_model', 'other_hosts'])
def test_wrong_identity_or_placement_cannot_retire_weights(artifact, damage):
    root, export, _, calls, _, plan = artifact
    if damage == 'identity':
        value = json.loads((root / 'publication-complete.json').read_text())
        value['identity'] = {'producer': 'different'}
        write(root / 'publication-complete.json', value)
    elif damage == 'other_model':
        plan['steps'][0]['copies'][0]['group'] = 'other/model'
    else:
        plan['goal']['hosts'] = ['raptor']
    with pytest.raises(ValueError):
        storage.archive(root)
    assert len(list(export.glob('*.safetensors'))) == 73
    assert not any(args[:2] == ('plan', 'apply') for args in calls)


def test_native_job_host_failures_are_not_silenced(monkeypatch):
    monkeypatch.setattr(storage, '_native', lambda *a: dict(job=dict(finished=True,
        error=None, hosts={'moa': dict(failed=['disk error'])})))
    with pytest.raises(ValueError, match='job did not complete'):
        storage._wait(1)

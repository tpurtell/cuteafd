"""Receipt-only publication and native storage lifecycle; no network/GPU fixtures."""
from copy import deepcopy
import hashlib
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2 import publish, handoff
from kimi_k3.uniform_k2.storage import atomic_json, file_identity
from test_inkling2_export import fixture as export_fixture
from inkling2.export import DirectMCWriter

REVISION = '1' * 40


def completed_export(tmp_path, *, attributes=True):
    args, entries, final = export_fixture(tmp_path)
    assets = tmp_path / 'assets'
    assets.mkdir()
    (assets / 'README.md').write_text('test checkpoint\n')
    (assets / '.gitattributes').write_text('*.safetensors filter=lfs diff=lfs merge=lfs -text\n')
    writer = DirectMCWriter(**args)
    writer.write_group('layer-000', **entries)
    writer.finalize(**final, assets={n: assets / n for n in
        (('README.md', '.gitattributes') if attributes else ('README.md',))})
    return args['output']


class Hub:
    """Fake Hub for verified HTTP LFS batch commits; tracks every sent file."""
    token = False

    def __init__(self, files):
        self.files = files
        self.remote = {}
        self.revision = '0' * 40
        self.uploads = 0
        self.sent = []
        self.batches = []
        self.workers = []
        self.fail = False
        self.lose_reply = False
        self.normalized = None
        self.mutate = None

    def model_info(self, repo, *, revision=None, files_metadata=False):
        assert repo == publish.TARGET_REPO
        assert revision is None or revision == self.revision
        return SimpleNamespace(sha=self.revision, private=False,
            siblings=[SimpleNamespace(rfilename=n, size=v['bytes'], blob_id=v['git_blob'],
                lfs=SimpleNamespace(sha256=v['sha256']) if n.endswith('.safetensors') else None)
                for n, v in self.remote.items()])

    def create_commit(self, repo, operations, **kwargs):
        assert repo == publish.TARGET_REPO and kwargs['revision'] == 'main'
        assert kwargs['parent_commit'] == self.revision
        names = [op.path_in_repo for op in operations]
        assert not set(names) & set(self.remote), 'committed file sent again'
        for op in operations:
            # Receipt SHA256 supplied; the SDK owns uploaded/committed state.
            assert op.upload_info.is_hashed
            assert op.upload_info.sha256.hex() == self.files[op.path_in_repo]['sha256']
            assert op._is_uploaded is False and op._is_committed is False
        self.batches.append(names)
        self.workers.append(kwargs['num_threads'])
        if self.mutate is not None and len(self.batches) == self.mutate[0]:
            with Path(self.files[self.mutate[1]]['path']).open('ab') as stream:
                stream.write(b'changed during transfer')
        if self.fail and self.uploads == 1:
            raise OSError('upload interrupted')
        self.sent.extend(names)
        self.uploads += 1
        self.remote.update({name: deepcopy(self.files[name]) for name in names})
        if self.normalized is not None and '.gitattributes' in self.remote:
            raw = self.normalized
            self.remote['.gitattributes'].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                git_blob=hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest())
        self.revision = REVISION if self.uploads == 1 else f'{self.uploads:040x}'
        if self.lose_reply:
            self.lose_reply = False
            raise OSError('lost commit reply')
        return SimpleNamespace(oid=self.revision)


def test_publication_uses_streaming_receipts_and_metadata_only(tmp_path, monkeypatch):
    output = completed_export(tmp_path)
    real_open = Path.open

    def deny_weight_reads(path, *a, **kw):
        if str(path).endswith('.safetensors'):
            pytest.fail('publication attempted a weight reread')
        return real_open(path, *a, **kw)

    monkeypatch.setattr(Path, 'open', deny_weight_reads)
    packet = publish.publication_inventory(output)
    api = Hub(packet['files'])
    root = tmp_path / 'publication'
    receipt = publish.publish(output, root, api=api, anonymous_api=api)
    assert receipt['status'] == 'passed'
    assert receipt['manifest_sha256'] == packet['manifest_sha256']
    assert api.uploads == 1
    assert 'no remote weight reads' in receipt['verification']
    shutil.rmtree(output)
    assert publish.publish(output, root, api=api, anonymous_api=api) == receipt
    assert api.uploads == 1


def test_publish_supplies_missing_protocol_metadata_without_rewriting_sealed_package(tmp_path, monkeypatch):
    # Match Runner.seal: it supplies README/tokenizer assets, not .gitattributes.
    output = completed_export(tmp_path, attributes=False)
    sealed = {name: (output / name).read_bytes() for name in
              ('package-state.json', 'package-validation.json', 'package-provenance.json')}
    assert '.gitattributes' not in publish.load(output / 'package-state.json')['new_files']
    real_open = Path.open

    def no_weight_reads(path, *args, **kwargs):
        if str(path).endswith('.safetensors'):
            pytest.fail('publication metadata recovery reread weights')
        return real_open(path, *args, **kwargs)

    class PreparedHub(Hub):
        def create_commit(self, repo, operations, **kwargs):
            self.files = publish.publication_inventory(output)['files']
            return super().create_commit(repo, operations, **kwargs)

    monkeypatch.setattr(Path, 'open', no_weight_reads)
    api, root = PreparedHub({}), tmp_path / 'publication'
    saved = publish.publish(output, root, api=api, anonymous_api=api)
    assert saved['status'] == 'passed'
    assert (output / '.gitattributes').read_bytes() == publish.LFS_ATTRIBUTES
    assert all((output / name).read_bytes() == raw for name, raw in sealed.items())
    assert publish.load(root / 'publication-metadata.json')['role'] == 'publication-only-protocol-metadata'
    before = file_identity(output / '.gitattributes')
    publish.prepare_publication_metadata(output, root)
    assert file_identity(output / '.gitattributes') == before
    assert not list(output.glob('.gitattributes.*.tmp'))


@pytest.mark.parametrize('kind', ['conflict', 'symlink'])
def test_unsealed_publication_attributes_are_never_overwritten(tmp_path, kind):
    output = completed_export(tmp_path, attributes=False)
    path = output / '.gitattributes'
    if kind == 'conflict':
        path.write_bytes(b'foreign rules\n')
    else:
        path.symlink_to(output / 'README.md')
    with pytest.raises(ValueError, match='attributes|symlink'):
        publish.prepare_publication_metadata(output, tmp_path / 'publication')
    assert path.is_symlink() if kind == 'symlink' else path.read_bytes() == b'foreign rules\n'


def test_partial_upload_resumes_without_resending_committed_files(tmp_path):
    output = completed_export(tmp_path)
    packet = publish.publication_inventory(output)
    api = Hub(packet['files'])
    api.fail = True
    root, options = tmp_path / 'publication', publish.hub_lfs.UploadOptions(workers=3, batch_files=2)
    with pytest.raises(OSError, match='interrupted'):
        publish.publish(output, root, api=api, anonymous_api=api, options=options)
    committed = set(api.remote)
    assert len(committed) == 2
    api.fail = False
    assert publish.publish(output, root, api=api, anonymous_api=api, options=options)['status'] == 'passed'
    assert len(api.sent) == len(set(api.sent)) == len(packet['files'])
    assert not committed & {name for batch in api.batches[2:] for name in batch}
    assert set(api.workers) == {3} and all(len(batch) <= 2 for batch in api.batches)


def http_files(tmp_path, count=34):
    files = {}
    names = ['.gitattributes', 'config.json', *[f'shard-{i:02d}.safetensors' for i in range(count)]]
    for i, name in enumerate(names):
        path = tmp_path / name
        path.write_bytes(name.encode() * (i + 1))
        files[name] = dict(path=str(path), bytes=path.stat().st_size,
                           device=path.stat().st_dev, file_identity=file_identity(path),
                           **publish.metadata_identity(path))
    return files


def run_http_lfs(api, files, tmp_path, options=None):
    return publish._upload_http_lfs(api, api.model_info(publish.TARGET_REPO), files, tmp_path, None,
                                    target=publish.INKLING, options=options or publish.hub_lfs.UploadOptions())


def test_http_lfs_defaults_to_16_workers_and_48_file_batches(tmp_path):
    files = http_files(tmp_path, count=100)
    api = Hub(files)
    run_http_lfs(api, files, tmp_path)
    assert [len(batch) for batch in api.batches] == [48, 48, 6]
    assert set(api.workers) == {publish.hub_lfs.DEFAULT_WORKERS} == {16}
    # Attributes first, then shards largest first, small metadata last.
    assert api.batches[0][0] == '.gitattributes' and api.batches[-1][-1] == 'config.json'
    sizes = [files[n]['bytes'] for batch in api.batches for n in batch if n.endswith('.safetensors')]
    assert sizes == sorted(sizes, reverse=True)
    assert set(api.remote) == set(files)


def test_http_lfs_commits_verified_batches_and_skips_committed_files(tmp_path):
    files = http_files(tmp_path)
    api = Hub(files)
    api.fail = True
    options = publish.hub_lfs.UploadOptions(workers=8, batch_files=16)
    with pytest.raises(OSError, match='interrupted'):
        run_http_lfs(api, files, tmp_path, options)
    assert len(api.remote) == 16
    committed = set(api.remote)
    api.fail = False
    run_http_lfs(api, files, tmp_path, options)
    assert api.uploads == 3 and set(api.remote) == set(files)
    assert set(api.workers) == {8}
    assert len(api.sent) == len(set(api.sent)) == len(files)
    assert not committed & {name for batch in api.batches[2:] for name in batch}


def test_http_lfs_lost_commit_reply_resumes_from_hub_tree(tmp_path):
    files = http_files(tmp_path, count=4)
    api = Hub(files)
    api.lose_reply = True
    options = publish.hub_lfs.UploadOptions(batch_files=3)
    with pytest.raises(OSError, match='lost commit reply'):
        run_http_lfs(api, files, tmp_path, options)
    run_http_lfs(api, files, tmp_path, options)
    assert len(api.sent) == len(set(api.sent)) == len(files)


@pytest.mark.parametrize('change', ['bytes', 'sha256'])
def test_http_lfs_refuses_receipt_mismatch_before_transfer(tmp_path, change):
    files = http_files(tmp_path, count=4)
    shard = 'shard-02.safetensors'
    if change == 'bytes':
        files[shard]['bytes'] += 1
    else:
        files[shard]['sha256'] = 'not-a-sha256'
    api = Hub(files)
    with pytest.raises(ValueError, match='sealed receipt|receipt SHA256'):
        run_http_lfs(api, files, tmp_path, publish.hub_lfs.UploadOptions(batch_files=64))
    assert api.batches == [] and api.remote == {}


def test_http_lfs_refuses_source_changed_mid_batch(tmp_path):
    files = http_files(tmp_path, count=6)
    api = Hub(files)
    options = publish.hub_lfs.UploadOptions(batch_files=3)
    changed = publish.hub_lfs.upload_order(files)[4]
    api.mutate = (2, changed)  # Changes while batch 2 is transferring.
    with pytest.raises(ValueError, match='source changed'):
        run_http_lfs(api, files, tmp_path, options)
    assert len(api.batches) == 2  # Its verify never ran and no later batch started.


def test_publish_resume_refuses_source_changed_mid_batch(tmp_path):
    output = completed_export(tmp_path)
    packet = publish.publication_inventory(output)
    api = Hub(packet['files'])
    options = publish.hub_lfs.UploadOptions(batch_files=2)
    changed = next(n for n in publish.hub_lfs.upload_order(packet['files'])[2:4] if n != '.gitattributes')
    api.mutate = (2, changed)
    root = tmp_path / 'publication'
    with pytest.raises(ValueError, match='source changed'):
        publish.publish(output, root, api=api, anonymous_api=api, options=options)
    with pytest.raises(ValueError, match='committed file changed|source changed'):
        publish.publish(output, root, api=api, anonymous_api=api, options=options)
    assert len(api.batches) == 2 and not (root / 'publication-complete.json').exists()


def test_http_lfs_refuses_remote_not_matching_receipts(tmp_path):
    files = http_files(tmp_path, count=4)
    api = Hub(files)
    run_http_lfs(api, files, tmp_path, publish.hub_lfs.UploadOptions(batch_files=2))
    api.remote['shard-00.safetensors']['sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='Hub file identity differs'):
        publish.verify_hub(api, api.revision, files)


def test_publication_requires_http_lfs_before_any_remote_call(tmp_path, monkeypatch):
    from huggingface_hub import constants
    output = completed_export(tmp_path)
    api = Hub(publish.publication_inventory(output)['files'])
    monkeypatch.setattr(constants, 'HF_HUB_DISABLE_XET', False)
    with pytest.raises(ValueError, match='HF_HUB_DISABLE_XET=1'):
        publish.publish(output, tmp_path / 'publication', api=api, anonymous_api=api)
    assert api.batches == [] and not (tmp_path / 'publication' / 'publication-destination.json').exists()


@pytest.mark.parametrize('change', ['sha256', 'bytes', 'extra', 'nonlfs'])
def test_remote_metadata_mismatch_fails(tmp_path, change):
    output = completed_export(tmp_path)
    files = publish.publication_inventory(output)['files']
    api = Hub(files)
    api.remote, api.revision = deepcopy(files), REVISION
    shard = next(n for n in files if n.endswith('.safetensors'))
    if change == 'extra':
        api.remote['stray'] = deepcopy(api.remote['README.md'])
    elif change == 'nonlfs':
        real_info = api.model_info

        def info(*a, **kw):
            result = real_info(*a, **kw)
            next(v for v in result.siblings if v.rfilename == shard).lfs = None
            return result

        api.model_info = info
    else:
        api.remote[shard][change] = 1 if change == 'bytes' else '0' * 64
    with pytest.raises(ValueError, match='Hub file'):
        publish.verify_hub(api, REVISION, files)


def test_normalized_hub_metadata_download_stays_in_task_cache(tmp_path, monkeypatch):
    import huggingface_hub
    output = completed_export(tmp_path)
    files = publish.publication_inventory(output)['files']
    api, root = Hub(files), tmp_path / 'publication'
    shard = next(name for name in files if name.endswith('.safetensors'))
    api.normalized = (output / '.gitattributes').read_bytes() + (
        shard + ' filter=lfs diff=lfs merge=lfs -text\n').encode()
    downloads = []

    def download(repo, name, **kwargs):
        assert repo == publish.TARGET_REPO and name == '.gitattributes'
        assert kwargs['revision'] == REVISION
        assert kwargs['local_dir'] == root / 'hub-metadata'
        downloads.append(name)
        path = kwargs['local_dir'] / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(api.normalized)
        return str(path)

    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', download)
    assert publish.publish(output, root, api=api, anonymous_api=api)['status'] == 'passed'
    assert downloads == ['.gitattributes']


def test_only_literal_appended_hub_attributes_are_accepted(tmp_path):
    output = completed_export(tmp_path)
    packet = publish.publication_inventory(output)
    api = Hub(packet['files'])
    shard = next(n for n in packet['files'] if n.endswith('.safetensors'))
    api.normalized = (output / '.gitattributes').read_bytes() + (shard + ' filter=lfs diff=lfs merge=lfs -text\n').encode()
    fetched = []

    def fetch(revision, name):
        fetched.append(name)
        return api.normalized

    receipt = publish.publish(output, tmp_path / 'publication', api=api, anonymous_api=api, fetch_metadata=fetch)
    assert fetched == ['.gitattributes']
    assert receipt['files']['.gitattributes']['path'].endswith('published.gitattributes')
    assert receipt['manifest_sha256'] != packet['manifest_sha256']
    api.normalized += b'*.py filter=lfs diff=lfs merge=lfs -text\n'
    with pytest.raises(ValueError, match='normalized attributes'):
        publish._attribute_files(api, REVISION, packet['files'], tmp_path, fetch)


def test_changed_shard_stat_and_unowned_destination_fail(tmp_path):
    output = completed_export(tmp_path)
    packet = publish.publication_inventory(output)
    api = Hub(packet['files'])
    api.remote = deepcopy(packet['files'])
    with pytest.raises(ValueError, match='new, empty'):
        publish.publish(output, tmp_path / 'publication', api=api, anonymous_api=api)
    shard = output / next(n for n in packet['files'] if n.endswith('.safetensors'))
    with shard.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='committed file changed'):
        publish.publication_inventory(output)


def storage_fixture(tmp_path):
    output, root, stage, managed, source = (tmp_path / n for n in ('export', 'root', 'stage', 'managed', 'source'))
    for p in (output, root, source):
        p.mkdir()
    files = {}
    for i in range(8):
        path = output / f'weight-{i:02d}.safetensors'
        raw = b'x' * (100 + i)
        path.write_bytes(raw)
        files[path.name] = dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
            git_blob=None, file_identity=file_identity(path), device=path.stat().st_dev)
    identity = dict(family='inkling2', test='storage')
    atomic_json(output / 'package-state.json', dict(identity=identity))
    receipt = dict(status='passed', repo_id=publish.TARGET_REPO, revision=REVISION,
        output=str(output), source_snapshot=str(source), files=files, identity=identity,
        manifest_sha256=publish.manifest_digest(files))
    atomic_json(root / 'publication-complete.json', receipt)
    ref = managed / handoff.MODEL / 'refs/main'
    ref.parent.mkdir(parents=True)
    ref.write_text(REVISION)
    return root, stage, managed, receipt


class Nest:
    def __init__(self, published):
        self.files = published['files']
        self.calls = []
        self.offgrid = False
        self.plans = {}
        self.changed_show = False

    def __call__(self, *args):
        self.calls.append(tuple(map(str, args)))
        if args[:2] == ('hf', 'import'):
            assert args[-2:] == ('--offline', '--no-copy')
            return {'job': 1}
        if args[0] == 'jobs':
            number = args[1]
            count = len(self.files) + (number == 3)
            result = dict(job=dict(finished=True, error=None, cancelled=False, pending=0, hosts={
                'maxq': dict(finished=number != 3, total_files=count, done_files=count,
                            total_bytes=0, done_bytes=0, failed=[], cancelled=False)}))
            if number == 1:
                result['import'] = dict(finished=True, errors=[], cancelled=False, copied_bytes=0,
                                        files=0, adopted=len(self.files), skipped=0)
            return result
        if args[:2] == ('plan', 'show'):
            plan = deepcopy(self.plans[args[2]])
            if self.changed_show:
                plan['steps'][0]['copies'][0]['size'] += 1
            return plan
        if args[:2] == ('plan', 'apply'):
            self.offgrid = args[2] == 20
            return {'job': 3 if self.offgrid else 2}
        if args[0] == 'plan':
            offgrid = args[1] == '--offgrid'
            number = 20 if offgrid else 10
            steps = []
            for i, (name, row) in enumerate(self.files.items()):
                copy = dict(name=name, path=str(handoff.NAMESPACE / handoff.MODEL / 'snapshots' / REVISION / name),
                            group=publish.TARGET_REPO, size=row['bytes'])
                steps.append(dict(kind='evict' if offgrid else 'replicate', host='maxq' if offgrid else handoff.TARGETS[i], copies=[copy]))
            if offgrid:
                steps[0]['copies'].append(dict(name='main', path=str(handoff.NAMESPACE / handoff.MODEL / 'refs/main'),
                    group=str(handoff.NAMESPACE / handoff.MODEL / 'refs'), size=len(REVISION)))
            goal = dict(goal='offgrid' if offgrid else 'place', selector='hf:' + publish.TARGET_REPO,
                hub=str(handoff.NAMESPACE), hosts=[] if offgrid else
                [args[i + 1] for i, value in enumerate(args) if value == '--host'],
                factor=1 / 8, by_speed=False)
            self.plans[number] = dict(id=number, feasible=True, blocked=[], goal=goal, steps=steps)
            return self.plans[number]
        if args[0] == 'ls':
            if Path(args[1]).name == 'refs':
                return {'entries': [dict(name='main', kind='regular', size=len(REVISION), sealed=False,
                    writing_on=None, hosts=[handoff.TARGETS[0]] + ([] if self.offgrid else ['maxq']))]}
            return {'entries': [dict(name=n, size=row['bytes'], kind='regular', sealed=True, writing_on=None,
                hosts=[handoff.TARGETS[i]] + ([] if self.offgrid else ['maxq']))
                for i, (n, row) in enumerate(self.files.items())]}
        pytest.fail('unexpected native command ' + str(args))


def test_hardlink_import_native_spread_resume_and_guarded_retirement(tmp_path):
    root, stage, managed, published = storage_fixture(tmp_path)
    nest = Nest(published)
    result = handoff.handoff(root, stage, call=nest, managed=managed)
    assert result['status'] == 'passed' and result['one_copy_per_file']
    for name, row in published['files'].items():
        assert (stage / handoff.MODEL / 'snapshots' / REVISION / name).samefile(row['path'])
    commands = nest.calls
    placement = next(c for c in commands if c[:2] == ('plan', '--place'))
    assert placement.count('--host') == 8 and placement[placement.index('--factor') + 1] == '1/8'
    assert commands.index(('plan', 'show', '10')) < commands.index(('plan', 'apply', '10'))
    calls = len(commands)
    assert handoff.handoff(root, stage, call=nest, managed=managed) == result
    assert len(nest.calls) == calls
    frontier = root / 'frontier'
    frontier.mkdir()
    inspected = []
    cleaned = handoff.retire(root, intermediates=[frontier], inspect=lambda paths: inspected.extend(paths))
    assert set(inspected) == {Path(published['output']), stage, frontier}
    assert cleaned['status'] == 'passed' and not stage.exists() and not frontier.exists()
    assert Path(published['source_snapshot']).exists() and managed.exists()
    assert (root / 'publication-complete.json').exists()
    assert handoff.retire(root, inspect=lambda paths: None)['paths'] == []


@pytest.mark.parametrize('replacement', ['raptor', 'raptor:d1', 'kakapo', 'raptor:d3'])
def test_native_goal_disk_aliases_do_not_allow_other_targets(tmp_path, replacement):
    _, _, _, published = storage_fixture(tmp_path)
    nest = Nest(published)
    arguments = ('plan', '--place', 'hf:' + publish.TARGET_REPO, '--factor', '1/8',
                 *(value for host in handoff.TARGET_ARGS for value in ('--host', host)))
    plan = nest(*arguments)
    assert plan['goal']['hosts'] == list(handoff.TARGET_ARGS)
    assert handoff.check_plan(plan, REVISION, published['files']) == plan
    plan['goal']['hosts'][0] = replacement
    with pytest.raises(ValueError, match='authorized even eight-place spread'):
        handoff.check_plan(plan, REVISION, published['files'])


@pytest.mark.parametrize('failure', ['unfinished', 'failed', 'count', 'not-terminal'])
def test_pure_eviction_requires_terminal_job_and_exact_counters(failure):
    host = dict(finished=False, failed=[], cancelled=False, done_files=3, total_files=3, total_bytes=0)
    result = dict(job=dict(finished=True, pending=0, cancelled=False, error=None, hosts={'maxq': host}))
    if failure == 'unfinished':
        with pytest.raises(ValueError, match='host job'):
            handoff.successful(result)
        assert handoff.successful(result, eviction_files=3) == result
    else:
        if failure == 'failed':
            host['failed'] = ['one file']
        elif failure == 'count':
            host['done_files'] = host['total_files'] = 2
        else:
            result['job']['finished'] = False
        with pytest.raises(ValueError):
            handoff.successful(result, eviction_files=3)


def test_plan_change_refuses_apply_and_keeps_payloads(tmp_path):
    root, stage, managed, published = storage_fixture(tmp_path)
    nest = Nest(published)
    nest.changed_show = True
    with pytest.raises(ValueError, match='unreviewed file'):
        handoff.handoff(root, stage, call=nest, managed=managed)
    assert not any(c[:2] == ('plan', 'apply') for c in nest.calls)
    assert Path(published['output']).exists()
    assert not (root / 'handoff-complete.json').exists()
    with pytest.raises(FileNotFoundError):
        handoff.retire(root, inspect=lambda paths: pytest.fail('cleanup before verified handoff'))


def test_cleanup_rejects_busy_or_unrelated_payloads(tmp_path):
    root, stage, managed, published = storage_fixture(tmp_path)
    handoff.handoff(root, stage, call=Nest(published), managed=managed)
    with pytest.raises(ValueError, match='task-owned'):
        handoff.retire(root, intermediates=[tmp_path / 'unrelated'], inspect=lambda _: None)
    with pytest.raises(ValueError, match='build root'):
        handoff.retire(root, build_root=tmp_path / 'builds', inspect=lambda _: None)

    def busy(_):
        raise RuntimeError('process uses task output')

    with pytest.raises(RuntimeError, match='process uses'):
        handoff.retire(root, inspect=busy)
    assert stage.exists() and Path(published['output']).exists()
    assert not (root / 'retired-receipts').exists()


@pytest.mark.parametrize('argv,comm,expected', [
    ([b'sshd-session: tj@notty', b''], 'sshd-session', True),
    ([b'sshd-session: tj@pts/5', b'', b''], 'sshd-session', True),
    ([b'sshd-session: tj@pts/8  ', b''], 'sshd-session', True),
    ([b'sshd-session: tj@pts/5 command', b''], 'sshd-session', False),
    ([b'sshd-session: tj@pts/5', b'command'], 'sshd-session', False),
    ([b'sshd-session: tj@pts/5', b''], 'python', False),
    ([b'sshd-session: other@pts/5', b''], 'sshd-session', False),
    ([b'sshd-session: tj@pts/x', b''], 'sshd-session', False),
    ([b'python', b''], 'python', False),
    ([b'/usr/bin/ssh-agent', b'-D', b''], 'ssh-agent', True),
    ([b'ssh-agent', b'tmux', b'new', b'-s', b'work', b''], 'ssh-agent', True),
    ([b'ssh-agent', b'tmux', b'new', b'-s', b'work', b'other-command', b''], 'ssh-agent', False),
    ([b'/tmp/ssh-agent', b'-D', b''], 'ssh-agent', False),
])
def test_unreadable_ssh_infrastructure_is_exact(argv, comm, expected):
    assert handoff._known_unreadable_process(argv, comm) is expected


def test_cleanup_retains_receipts_before_removing_any_payload(tmp_path, monkeypatch):
    root, stage, managed, published = storage_fixture(tmp_path)
    handoff.handoff(root, stage, call=Nest(published), managed=managed)
    output = Path(published['output'])
    packed = root / 'production/packed'
    packed.mkdir(parents=True)
    activation = root / 'production/activations'
    (activation / 'layer-000').mkdir(parents=True)
    taps = root / 'production/target-taps'
    (taps / 'layer-000').mkdir(parents=True)
    build = tmp_path / '.cache/cuteafd/builds/inkling2-test'
    (build / 'release-staging').mkdir(parents=True)
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: tmp_path))
    kept = {
        'output/package-state.json': output / 'package-state.json',
        'output/events.jsonl': output / 'events.jsonl',
        'stage/.inkling2-owned-stage.json': stage / '.inkling2-owned-stage.json',
        'intermediates/production/packed/projection-000.json': packed / 'projection-000.json',
        'intermediates/production/activations/layer-000/manifest.json': activation / 'layer-000/manifest.json',
        'intermediates/production/target-taps/layer-000/manifest.json': taps / 'layer-000/manifest.json',
        'build/release-staging/build.log': build / 'release-staging/build.log',
        'build/release-staging/errors.csv': build / 'release-staging/errors.csv',
    }
    for name, source in kept.items():
        if not source.exists():
            source.write_text(name)
    original = {name: source.stat().st_ino for name, source in kept.items()}
    (activation / 'layer-000/payload.safetensors').write_bytes(b'payload')
    error_csv = root / 'per-module-errors.csv'
    error_csv.write_text('preserve original')
    real_rmtree = handoff.shutil.rmtree
    archive = root / 'retired-receipts'

    def checked_remove(path):
        assert all((archive / name).stat().st_ino == inode for name, inode in original.items())
        return real_rmtree(path)

    def no_payload_open(path, *args, **kwargs):
        if path.suffix == '.safetensors':
            pytest.fail('cleanup opened payload bytes')
        return real_open(path, *args, **kwargs)

    real_open = Path.open
    monkeypatch.setattr(Path, 'open', no_payload_open)
    monkeypatch.setattr(handoff.shutil, 'rmtree', checked_remove)
    inspected = []
    result = handoff.retire(root, intermediates=[packed, activation, taps], build_root=build,
                            inspect=lambda paths: inspected.extend(paths))
    assert result['retained_metadata_files'] == len(kept)
    assert set(inspected) == {output, stage, packed, activation, taps, build / 'release-staging'}
    assert all(not path.exists() for path in inspected)
    assert error_csv.read_text() == 'preserve original'
    assert not list(archive.rglob('*.safetensors'))
    assert publish.load(archive / '.inkling2-owned-archive.json')['identity'] == result['identity']
    assert handoff.retire(root, inspect=lambda _: None)['paths'] == []


@pytest.mark.parametrize('conflict', ['file', 'symlink', 'ownership'])
def test_cleanup_archive_retry_and_conflicts_preserve_payloads(tmp_path, monkeypatch, conflict):
    root, stage, managed, published = storage_fixture(tmp_path)
    complete = handoff.handoff(root, stage, call=Nest(published), managed=managed)
    output = Path(published['output'])
    archive = root / 'retired-receipts'
    real_link = handoff.os.link
    calls = []

    def interrupted(source, target):
        calls.append(target)
        if len(calls) == 2:
            raise OSError('archive interrupted')
        return real_link(source, target)

    monkeypatch.setattr(handoff.os, 'link', interrupted)
    with pytest.raises(OSError, match='archive interrupted'):
        handoff.retire(root, inspect=lambda _: None)
    assert output.exists() and stage.exists()
    target = archive / 'output/package-state.json'
    assert target.samefile(output / 'package-state.json')
    monkeypatch.setattr(handoff.os, 'link', real_link)
    if conflict == 'ownership':
        atomic_json(archive / '.inkling2-owned-archive.json', dict(identity={'wrong': True}))
    else:
        target.unlink()
        if conflict == 'file':
            target.write_text('foreign')
        else:
            target.symlink_to(output / 'package-state.json')
    with pytest.raises(ValueError, match='archive|symlink'):
        handoff.retire(root, inspect=lambda _: None)
    assert output.exists() and stage.exists()
    if conflict == 'ownership':
        atomic_json(archive / '.inkling2-owned-archive.json',
            dict(schema='cuteafd-inkling2-retired-receipts-v1', identity=complete['identity']))
    else:
        target.unlink()
        real_link(output / 'package-state.json', target)
    assert handoff.retire(root, inspect=lambda _: None)['status'] == 'passed'


def test_staging_rejects_symlink_and_existing_unowned_leaf(tmp_path):
    root, stage, _, published = storage_fixture(tmp_path)
    stage.mkdir()
    (stage / 'unrelated').write_text('keep')
    with pytest.raises(ValueError, match='empty task-owned'):
        handoff.stage_snapshot(published, stage)
    link = tmp_path / 'symlink-stage'
    link.symlink_to(stage)
    with pytest.raises(ValueError, match='symlink'):
        handoff.stage_snapshot(published, link)


def test_staging_extra_files_and_invalid_revision_fail(tmp_path):
    _, stage, _, published = storage_fixture(tmp_path)
    repo = handoff.stage_snapshot(published, stage)
    (repo / 'snapshots' / REVISION / 'unrelated.bin').write_bytes(b'keep')
    with pytest.raises(ValueError, match='unexpected files'):
        handoff.stage_snapshot(published, stage)
    published['revision'] = '../../outside'
    with pytest.raises(ValueError, match='verified publication'):
        handoff.stage_snapshot(published, stage)


def test_cleanup_preserves_managed_source_even_when_called_as_output(tmp_path):
    root, stage, managed, published = storage_fixture(tmp_path)
    handoff.handoff(root, stage, call=Nest(published), managed=managed)
    # Matching metadata cannot turn a managed source into disposable output.
    published['source_snapshot'] = published['output']
    atomic_json(root / 'publication-complete.json', published)
    with pytest.raises(ValueError, match='protected'):
        handoff.retire(root, inspect=lambda _: pytest.fail('protected path reached cleanup scanner'))
    assert Path(published['output']).exists()

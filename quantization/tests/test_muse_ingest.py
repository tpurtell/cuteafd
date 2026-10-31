"""Operational ingest barriers and failure recovery without network/GPU use."""
from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from muse_spark.core_k6 import ingest
from kimi_k3.uniform_k2.storage import atomic_json, file_identity


@pytest.fixture
def case(tmp_path):
    root = tmp_path / 'run';root.mkdir()
    source = tmp_path / 'export';source.mkdir()
    rows = []
    for i in range(6):
        path = source / f'model-{i + 1:05d}-of-00006.safetensors'
        path.write_bytes(bytes([i + 1]) * (100 - i))
        rows.append(dict(relative=path.name, source=str(path), size=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), file_identity=file_identity(path), weight=True))
    asset = source / 'LICENSE.md';asset.write_bytes(b'original license')
    rows.append(dict(relative=asset.name, source=str(asset), size=asset.stat().st_size,
        sha256=hashlib.sha256(asset.read_bytes()).hexdigest(), file_identity=file_identity(asset), weight=False))
    ref = source / 'main';ref.write_bytes(b'a' * 40)
    inputs = dict(root=str(root), export=str(source), revision='a' * 40,
        files=rows, publication_marker_sha256='b' * 64, manifest_sha256='c' * 64,
        identity={'source_revision':'pin'}, ref_source=str(ref))
    plan = ingest.make_plan(inputs)
    return SimpleNamespace(root=root, source=source, inputs=inputs, plan=plan)


class FakeTransport:
    def __init__(self, plan):
        self.plan = plan;self.files = {};self.calls = [];self.fail_upload = None
        self.live = set();self.bad_hash = set();self.offline = True

    def add(self, namespace, data, host, *, owners=None, sealed=True):
        self.files[namespace] = dict(data=data, hosts=owners or [host], sealed=sealed)

    def metadata(self, namespace, **kwargs):
        row = self.files.get(namespace)
        if row is None:return None
        return dict(kind='regular', size=len(row['data']), hosts=list(row['hosts']),
            sealed=row['sealed'], writing_on=row['hosts'][0] if namespace in self.live else None,
            resolved_namespace=namespace)

    def stable(self, namespace, **kwargs):
        if namespace in self.live:raise TimeoutError('live writer')
        return self.metadata(namespace)

    def mkdir(self, host, namespace):
        self.calls.append(('mkdir', host, namespace))

    def upload(self, item):
        self.calls.append(('upload', item['relative']))
        data=Path(item['source']).read_bytes()
        if self.fail_upload == item['relative']:
            self.add(item['temporary'], data[:2], item['host'], sealed=False)
            raise InterruptedError('simulated connection cancellation')
        self.add(item['temporary'], data, item['host'], sealed=False)

    def remote_sha256(self, host, namespace):
        self.calls.append(('hash', host, namespace))
        if namespace in self.bad_hash:return '0'*64
        return hashlib.sha256(self.files[namespace]['data']).hexdigest()

    def seal(self, namespace):
        self.calls.append(('seal', namespace));self.files[namespace]['sealed']=True

    def remove_temporary(self, namespace):
        self.calls.append(('remove', namespace));del self.files[namespace]

    def replicate_asset(self, namespace, host):
        self.calls.append(('replicate', namespace, host));self.files[namespace]['hosts'].append(host)

    def rename(self, temporary, final):
        if final in self.files:raise FileExistsError(final)
        self.calls.append(('rename', temporary, final));self.files[final]=self.files.pop(temporary)

    def read_ref(self, namespace):
        return self.files.get(namespace,{}).get('data')

    def inventory(self, namespace):
        return {n[len(namespace)+1:] for n in self.files if n.startswith(namespace+'/')}

    def offline_snapshot(self, repo_id, revision):
        self.calls.append(('offline',repo_id,revision))
        if not self.offline:raise ValueError('offline lookup failure')
        return '/mnt/sparknest'+self.plan['snapshot_namespace']

    def readiness(self):
        self.calls.append(('readiness',))
        return {host:'ready' for host in ingest.HOSTS}


def test_plan_is_deterministic_balanced_and_all_six_get_weights(case):
    assert ingest.make_plan(case.inputs)==case.plan
    assert {f['host'] for f in case.plan['files'] if f['weight']}==set(ingest.HOSTS)
    assert [f['host'] for f in case.plan['files'] if f['weight']]==list(ingest.HOSTS)
    assert sum(case.plan['host_bytes'].values())==sum(f['size'] for f in case.inputs['files'])
    assert case.plan['ref']['namespace'].endswith('/refs/main')
    assert all(f['temporary'].endswith('.incomplete') and 'muse-ingest-' in f['temporary'] for f in case.plan['files'])


def test_full_run_publishes_ref_only_after_every_snapshot_file(case):
    transport=FakeTransport(case.plan)
    worker=ingest.Ingestor(case.plan,transport)
    result=worker.run()
    uploads=[call[1] for call in transport.calls if call[0]=='upload']
    assert uploads[-1]=='refs/main'
    assert len(uploads)==len(case.plan['files'])+1
    assert result['weight_shards']==result['weight_copies']==6
    assert (case.root/'ingest-complete.json').is_file()
    assert not transport.files[case.plan['ref']['namespace']]['sealed']
    before=len(uploads)
    ingest.Ingestor(case.plan,transport).run()
    assert len([call for call in transport.calls if call[0]=='upload'])==before


def test_cancel_keeps_owned_partial_and_resume_reuses_completed_files(case):
    transport=FakeTransport(case.plan)
    target=case.plan['files'][2]
    transport.fail_upload=target['relative']
    with pytest.raises(InterruptedError):ingest.Ingestor(case.plan,transport).run()
    assert target['temporary'] in transport.files
    assert case.plan['ref']['namespace'] not in transport.files
    assert not (case.root/'ingest-complete.json').exists()
    committed=json.loads((case.root/'ingest-state.json').read_text())['files']
    assert len(committed)==2
    transport.fail_upload=None
    result=ingest.Ingestor(case.plan,transport).run()
    assert result['weight_copies']==6
    assert ('remove',target['temporary']) in transport.calls
    uploads=[call[1] for call in transport.calls if call[0]=='upload']
    assert all(uploads.count(name)==1 for name in committed)
    assert uploads.count(target['relative'])==2


def test_live_partial_is_never_deleted_or_overwritten(case):
    transport=FakeTransport(case.plan);item=case.plan['files'][0]
    worker=ingest.Ingestor(case.plan,transport)
    worker.state['intents'][item['temporary']]=dict(relative=item['relative'],host=item['host']);worker.save()
    transport.add(item['temporary'],b'x',item['host'],sealed=False);transport.live.add(item['temporary'])
    with pytest.raises(TimeoutError):worker.ensure(item)
    assert item['temporary'] in transport.files
    assert not any(call[0] in ('remove','upload') for call in transport.calls)


def test_unrecognized_partial_is_left_untouched(case):
    transport=FakeTransport(case.plan);item=case.plan['files'][0]
    transport.add(item['temporary'],b'partial',item['host'],sealed=False)
    with pytest.raises(ValueError,match='predates'):ingest.Ingestor(case.plan,transport).ensure(item)
    assert not any(call[0] in ('remove','upload') for call in transport.calls)


def test_existing_final_wrong_hash_or_owner_is_not_overwritten(case):
    transport=FakeTransport(case.plan);item=next(f for f in case.plan['files'] if f['weight'])
    transport.add(item['namespace'],b'z'*item['size'],item['host'])
    worker=ingest.Ingestor(case.plan,transport)
    with pytest.raises(ValueError,match='checksum'):worker.ensure(item)
    assert not any(call[0] in ('remove','upload','rename') for call in transport.calls)
    transport.files[item['namespace']]['data']=Path(item['source']).read_bytes()
    transport.files[item['namespace']]['hosts']=[item['host'],'raptor (D2)']
    with pytest.raises(ValueError,match='exactly'):worker.ensure(item)
    assert not any(call[0] in ('remove','upload','rename') for call in transport.calls)


def test_shared_asset_owners_preserved_and_assigned_copy_added(case):
    transport=FakeTransport(case.plan);item=next(f for f in case.plan['files'] if not f['weight'])
    owners=['raptor (D2)','raptor (D3)']
    transport.add(item['namespace'],Path(item['source']).read_bytes(),item['host'],owners=list(owners))
    result=ingest.Ingestor(case.plan,transport).ensure(item)
    assert result['preexisting_owners']==owners
    assert set(result['owners'])==set(owners+[item['host']])
    assert not any(call[0] in ('remove','upload','rename') for call in transport.calls)
    assert ('replicate',item['namespace'],item['host']) in transport.calls


def test_other_existing_ref_blocks_before_any_write(case):
    transport=FakeTransport(case.plan)
    transport.add(case.plan['ref']['namespace'],b'd'*40,'ostrich')
    with pytest.raises(ValueError,match='another revision'):ingest.Ingestor(case.plan,transport).run()
    assert transport.calls==[]
    assert not (case.root/'ingest-state.json').exists()


def test_failure_before_file_barrier_cannot_publish_ref(case):
    transport=FakeTransport(case.plan)
    target=case.plan['files'][3]
    transport.bad_hash.add(target['temporary'])
    with pytest.raises(ValueError,match='checksum'):ingest.Ingestor(case.plan,transport).run()
    assert case.plan['ref']['namespace'] not in transport.files
    assert not (case.root/'ingest-complete.json').exists()


def test_resume_rejects_changed_assignment_or_code_identity(case):
    transport=FakeTransport(case.plan);worker=ingest.Ingestor(case.plan,transport);worker.save()
    changed=copy.deepcopy(case.plan);changed['code_sha256']='different'
    with pytest.raises(ValueError,match='immutable placement'):ingest.Ingestor(changed,transport)


def test_preflight_has_no_writes_and_detects_stale_plan(case):
    transport=FakeTransport(case.plan)
    report=ingest.preflight(case.plan,transport)
    assert report['status']=='passed'
    assert list(case.root.iterdir())==[]
    assert transport.calls==[('readiness',)]
    atomic_json(case.root/'ingest-plan.json',dict(other='plan'))
    with pytest.raises(ValueError,match='immutable ingest plan'):ingest.preflight(case.plan,transport)


class StreamConnection:
    def __init__(self,address,port,*,timeout):
        self.address=address;self.port=port;self.headers={};self.blocks=[];self.closed=False
        self.raise_send=None

    def putrequest(self,method,endpoint):self.method=method;self.endpoint=endpoint
    def putheader(self,name,value):self.headers[name]=value
    def endheaders(self):pass
    def send(self,data):
        if self.raise_send:raise self.raise_send
        self.blocks.append(bytes(data))
    def close(self):self.closed=True
    def getresponse(self):
        result=dict(path=parse_qs(urlsplit(self.endpoint).query)['path'][0],bytes=sum(map(len,self.blocks)))
        return SimpleNamespace(status=200,read=lambda _:json.dumps(result).encode())


def test_native_transport_streams_owned_bytes_with_fixed_address(case,monkeypatch,capsys):
    connections=[]
    def factory(*args,**kwargs):
        connection=StreamConnection(*args,**kwargs);connections.append(connection);return connection
    transport=ingest.NativeTransport(http_factory=factory);transport._token='f'*40
    item=case.plan['files'][0]
    monkeypatch.setattr(ingest,'CHUNK',7)
    result=transport.upload(item)
    connection=connections[-1]
    assert connection.address==ingest.ADDRESSES[item['host']]
    assert connection.port==7401 and connection.closed
    assert all(len(block)<=7 for block in connection.blocks)
    assert b''.join(connection.blocks)==Path(item['source']).read_bytes()
    assert connection.headers['Content-Length']==str(item['size'])
    assert result['bytes']==item['size']
    assert transport._token not in capsys.readouterr().out
    assert transport._token not in repr(transport)
    with pytest.raises(ValueError,match='allowlist'):transport._connection('raptor')


def test_native_stream_cancel_closes_connection(case):
    connection=StreamConnection('ignored',7401,timeout=1);connection.raise_send=InterruptedError('cancel')
    transport=ingest.NativeTransport(http_factory=lambda *a,**kw:connection);transport._token='f'*40
    with pytest.raises(InterruptedError):transport.upload(case.plan['files'][0])
    assert connection.closed


def test_native_stream_detects_same_length_source_corruption(case):
    connection=StreamConnection('ignored',7401,timeout=1)
    transport=ingest.NativeTransport(http_factory=lambda *a,**kw:connection);transport._token='f'*40
    item=dict(case.plan['files'][0]);item['sha256']='0'*64
    with pytest.raises(ValueError,match='source bytes changed'):transport.upload(item)
    assert connection.closed


def test_path_escape_and_oversized_control_rejected():
    for relative in ('../weight','/absolute','a/../b','a//b','a\\b'):
        with pytest.raises(ValueError):ingest.safe_relative(relative)
    for namespace in ('/etc/shadow','/hf-home/hub/../private','/hf-home/hub/x/../../x'):
        with pytest.raises(ValueError):ingest.safe_namespace(namespace)
    response=SimpleNamespace(status=200,read=lambda _:b'x'*(1024*1024+1))
    with pytest.raises(ValueError,match='oversized'):ingest.NativeTransport._response(SimpleNamespace(getresponse=lambda:response))


def prepare_published_case(case, monkeypatch):
    repo='models--'+ingest.TARGET_REPO.replace('/','--')
    local=case.root.parent/'local-hub'/repo
    snapshot=local/'snapshots'/case.plan['revision'];snapshot.mkdir(parents=True)
    refs=local/'refs';refs.mkdir();(refs/'main').hardlink_to(case.inputs['ref_source'])
    shards={f['relative'] for f in case.inputs['files'] if f['weight']}
    monkeypatch.setattr(ingest.publish,'_expected_shards',lambda:shards)
    monkeypatch.setattr(ingest.publish,'_receipt_path',lambda export,name:export/(name+'.receipt.json'))
    monkeypatch.setattr(ingest.publish,'_complete_plan',lambda *args:case.inputs['identity'])
    for item in case.inputs['files']:
        (snapshot/item['relative']).hardlink_to(item['source'])
        if item['weight']:
            atomic_json(case.source/(item['relative']+'.receipt.json'),dict(identity=case.inputs['identity'],
                filename=item['relative'],sha256=item['sha256'],file_identity=item['file_identity']))
    atomic_json(case.root/'plan.json',dict(export=str(case.source)))
    marker=dict(repo_id=ingest.TARGET_REPO,identity=case.inputs['identity'],revision=case.plan['revision'],
        files=len(case.inputs['files']),local_snapshot=str(snapshot),
        manifest_sha256=ingest.digest([[r['relative'],r['size'],r['sha256']]
            for r in sorted(case.inputs['files'],key=lambda r:r['relative'])]))
    atomic_json(case.root/'publication-complete.json',marker)
    return snapshot


def test_loader_binds_manifest_owned_inodes_and_receipts(case,monkeypatch):
    snapshot=prepare_published_case(case,monkeypatch)
    inputs=ingest.load_inputs(case.root)
    assert len(inputs['files'])==7
    assert inputs['revision']==case.plan['revision']
    # A byte-for-byte copy has the right digest but violates the no-second-copy
    # owned-snapshot contract; it must not become an untracked source.
    cached=snapshot/case.inputs['files'][0]['relative']
    data=cached.read_bytes();cached.unlink();cached.write_bytes(data)
    with pytest.raises(ValueError,match='hardlink'):ingest.load_inputs(case.root)


def test_loader_rejects_altered_published_asset(case,monkeypatch):
    snapshot=prepare_published_case(case,monkeypatch)
    (snapshot/'LICENSE.md').write_bytes(b'changed license')
    with pytest.raises(ValueError,match='manifest differs'):ingest.load_inputs(case.root)


def test_loader_rejects_changed_weight_receipt_identity(case,monkeypatch):
    prepare_published_case(case,monkeypatch)
    name=next(f['relative'] for f in case.inputs['files'] if f['weight'])
    path=case.source/(name+'.receipt.json');receipt=json.loads(path.read_text());receipt['identity']={'wrong':'producer'}
    atomic_json(path,receipt)
    with pytest.raises(ValueError,match='receipt differs'):ingest.load_inputs(case.root)


def test_atomic_namespace_rename_never_overwrites_existing_final(tmp_path):
    transport=ingest.NativeTransport(mount=tmp_path)
    namespace='/hf-home/hub/models--toy/snapshots/pin/'
    directory=tmp_path/namespace.lstrip('/');directory.mkdir(parents=True)
    (directory/'temporary').write_bytes(b'correct artifact')
    (directory/'final').write_bytes(b'unrelated final')
    with pytest.raises(FileExistsError):transport.rename(namespace+'temporary',namespace+'final')
    assert (directory/'final').read_bytes()==b'unrelated final'
    assert (directory/'temporary').read_bytes()==b'correct artifact'
    (directory/'final').unlink()
    transport.rename(namespace+'temporary',namespace+'final')
    assert (directory/'final').read_bytes()==b'correct artifact'
    assert not (directory/'temporary').exists()


def test_auth_uses_existing_40_hex_credential_without_printing(monkeypatch,capsys):
    transport=ingest.NativeTransport()
    monkeypatch.setattr(transport,'_local',lambda *args,**kwargs:dict(token='d'*40))
    transport.authenticate()
    assert transport._token=='d'*40
    assert capsys.readouterr().out==''
    monkeypatch.setattr(transport,'_local',lambda *args,**kwargs:dict(token='invalid'))
    with pytest.raises(ValueError,match='valid API credential'):transport.authenticate()


def test_metadata_follows_managed_asset_symlinks_only(monkeypatch):
    transport=ingest.NativeTransport()
    target='/hf-home/hub/blobs/aa/digest'
    def local(method,endpoint):
        path=parse_qs(urlsplit(endpoint).query)['path'][0]
        if path==target:return dict(entries=[dict(kind='regular',name='digest',size=3,hosts=['raptor (D2)'],sealed=True,writing_on=None)])
        return dict(entries=[dict(kind='symlink',name='asset',target='../../../../blobs/aa/digest')])
    monkeypatch.setattr(transport,'_local',local)
    row=transport.metadata('/hf-home/hub/models--toy/snapshots/pin/nested/asset')
    assert row['resolved_namespace']==target
    monkeypatch.setattr(transport,'_local',lambda *args:dict(entries=[dict(kind='symlink',name='asset',target='/etc/shadow')]))
    with pytest.raises(ValueError,match='unsafe SparkNest'):transport.metadata('/hf-home/hub/models--toy/asset')


def test_task_staging_symlink_is_not_followed_for_removal(case,monkeypatch):
    transport=FakeTransport(case.plan);item=case.plan['files'][0]
    worker=ingest.Ingestor(case.plan,transport)
    worker.state['intents'][item['temporary']]=dict(relative=item['relative'],host=item['host']);worker.save()
    transport.add(item['temporary'],b'x',item['host'])
    original=transport.metadata
    def metadata(namespace,**kwargs):
        row=original(namespace,**kwargs)
        if namespace==item['temporary']:
            row['kind']='symlink';row['target']='/hf-home/hub/unrelated'
        return row
    monkeypatch.setattr(transport,'metadata',metadata)
    with pytest.raises(ValueError,match='staging owner'):worker.ensure(item)
    assert not any(call[0] in ('remove','upload','rename') for call in transport.calls)


def test_ref_staging_does_not_trigger_hf_incomplete_rename_sealing(case):
    assert case.plan['ref']['temporary'].endswith('.pending')
    assert all(item['temporary'].endswith('.incomplete') for item in case.plan['files'])

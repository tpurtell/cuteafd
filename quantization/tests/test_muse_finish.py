"""CPU finish sequencing, full verification barriers and crash-safe retirement."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from muse_spark.core_k6 import finish,ingest,publish
from kimi_k3.uniform_k2.storage import atomic_json,file_identity,sha256_file


@pytest.fixture
def case(tmp_path,monkeypatch):
    work=tmp_path/'work';work.mkdir()
    root=work/'production-k128';root.mkdir();export=work/'export-k128';export.mkdir()
    monkeypatch.setattr(finish,'WORK_ROOT',work)
    monkeypatch.setattr(publish,'LOCAL_HUB_CACHE',work/'local-hub')
    monkeypatch.setattr(publish,'SHARD_COUNT',6)
    names={f'model-{i:05d}-of-00006.safetensors' for i in range(1,7)}
    monkeypatch.setattr(publish,'_expected_shards',lambda:names)
    monkeypatch.setattr(publish,'_receipt_path',lambda export,name:export/(name+'.receipt.json'))
    identity=dict(source_revision='pinned',recipe=dict(covariance='K128'))
    monkeypatch.setattr(publish,'_complete_plan',lambda *args:identity)
    revision='a'*40;snapshot=publish._cache_snapshot(finish.TARGET_REPO,revision);snapshot.mkdir(parents=True)
    ref=snapshot.parents[1]/'refs/main';ref.parent.mkdir();ref.write_text(revision)
    rows=[];audit_shards={}
    for i,name in enumerate(sorted(names)):
        path=export/name;path.write_bytes(bytes([i+1])*(100-i));(snapshot/name).hardlink_to(path)
        sha=sha256_file(path);item=dict(relative=name,source=str(path),size=path.stat().st_size,
            sha256=sha,file_identity=file_identity(path),weight=True);rows.append(item)
        atomic_json(export/(name+'.receipt.json'),dict(identity=identity,filename=name,sha256=sha,file_identity=item['file_identity']))
        audit_shards[name]=dict(sha256=sha,payload_bytes=item['size'])
    readme=export/'README.md';readme.write_text('retained model card');(snapshot/readme.name).hardlink_to(readme)
    rows.append(dict(relative=readme.name,source=str(readme),size=readme.stat().st_size,sha256=sha256_file(readme),
        file_identity=file_identity(readme),weight=False))
    (export/'quantization-errors/layer-000').mkdir(parents=True)
    (export/'quantization-errors/layer-000/raw-error.json').write_text('{"error":0.001}')
    (root/'events.jsonl').write_text('retained production events\n')
    payload=sum(r['size'] for r in rows if r['weight'])
    monkeypatch.setattr(finish,'TARGET_PAYLOAD_BYTES',payload);monkeypatch.setattr(finish,'TARGET_TENSOR_COUNT',12)
    monkeypatch.setattr(finish,'MODULE_COUNT',6)
    summary=dict(shards=6,tensors=12,payload_bytes=payload,quantized_projections=6,preserved_tensors=1,preserved_bytes=4)
    atomic_json(root/'quantization-complete.json',dict(identity=identity,summary=summary))
    atomic_json(root/'completed-export-audit.json',dict(status='passed',identity=identity,shards=audit_shards,
        tensors=12,payload_bytes=payload,numerical_records=6,preserved_tensors=1,preserved_bytes=4))
    manifest=ingest.digest([[r['relative'],r['size'],r['sha256']] for r in sorted(rows,key=lambda r:r['relative'])])
    marker=dict(repo_id=finish.TARGET_REPO,identity=identity,revision=revision,local_snapshot=str(snapshot),files=len(rows),manifest_sha256=manifest)
    atomic_json(root/'publication-complete.json',marker)
    inputs=dict(root=str(root),export=str(export),revision=revision,files=rows,
        publication_marker_sha256=sha256_file(root/'publication-complete.json'),manifest_sha256=manifest,
        identity=identity,ref_source=str(ref))
    plan=ingest.make_plan(inputs)
    complete=dict(schema=ingest.SCHEMA,plan_sha256=ingest.digest(plan),repo_id=finish.TARGET_REPO,revision=revision,
        published_manifest_sha256=manifest,file_count=len(rows),weight_shards=6,weight_copies=6,
        managed_snapshot=str(ingest.MOUNT/plan['snapshot_namespace'].lstrip('/')),
        files={r['relative']:dict(sha256=r['sha256'],size=r['size'],host=r['host'],namespace=r['namespace'],weight=r['weight'],owners=[r['host']]) for r in plan['files']},
        ref=dict(sha256=plan['ref']['sha256']))
    calls=[];bad_hash=[None];ingest_error=[False];ref_error=[False]
    def published(export_value,root_value,repo):calls.append(('publish',));return revision
    def run_ingest(root_value,export_value):
        calls.append(('ingest',))
        if ingest_error[0]:raise RuntimeError('interrupted ingest')
        atomic_json(root/'ingest-plan.json',plan);atomic_json(root/'ingest-complete.json',complete)
    class Transport:
        def readiness(self):calls.append(('readiness',));return {h:'ready' for h in ingest.HOSTS}
        def read_ref(self,namespace):return ('b'*40 if ref_error[0] else revision).encode()
        def inventory(self,namespace):return {r['relative'] for r in plan['files']}
        def stable(self,namespace):
            row=plan['ref'] if namespace==plan['ref']['namespace'] else next(r for r in plan['files'] if r['namespace']==namespace)
            return dict(kind='regular',writing_on=None,size=row['size'],sealed=True,hosts=[row['host']])
        def remote_sha256(self,host,namespace):
            calls.append(('remote_hash',namespace))
            if namespace==bad_hash[0]:return '0'*64
            row=plan['ref'] if namespace==plan['ref']['namespace'] else next(r for r in plan['files'] if r['namespace']==namespace)
            return row['sha256']
        def offline_snapshot(self,repo,rev):return complete['managed_snapshot']
    monkeypatch.setattr(ingest,'NativeTransport',Transport)
    monkeypatch.setattr(publish,'publish',published)
    monkeypatch.setattr(finish,'_run_ingest',run_ingest)
    public_verify=finish._verify_public
    monkeypatch.setattr(finish,'_verify_public',lambda marker,plan:calls.append(('public_verified',)))
    original_unlink=finish._unlink_weight
    def unlink(row):calls.append(('unlink',row['path']));original_unlink(row)
    monkeypatch.setattr(finish,'_unlink_weight',unlink)
    return SimpleNamespace(work=work,root=root,export=export,snapshot=snapshot,identity=identity,rows=rows,
        plan=plan,complete=complete,calls=calls,bad_hash=bad_hash,ingest_error=ingest_error,ref_error=ref_error,
        unlink=unlink,run_ingest=run_ingest,public_verify=public_verify,transport_type=Transport)


def test_finish_verifies_all_remote_objects_before_first_unlink_and_preserves_results(case):
    result=finish.finish(case.root,case.export)
    assert result['retired_weight_shards']==6
    assert result['retired_payload_bytes']==sum(r['size'] for r in case.rows if r['weight'])
    first=next(i for i,c in enumerate(case.calls) if c[0]=='unlink')
    assert sum(c[0]=='remote_hash' for c in case.calls[:first])==len(case.rows)+1
    assert case.calls[:2]==[('publish',),('ingest',)]
    for row in case.rows:
        for directory in (case.export,case.snapshot):
            assert (directory/row['relative']).exists() is (not row['weight'])
    assert (case.export/'quantization-errors/layer-000/raw-error.json').exists()
    assert (case.root/'events.jsonl').read_text()=='retained production events\n'
    assert len(list(case.export.glob('*.receipt.json')))==6
    assert (case.root/'local-retirement-complete.json').is_file()


def test_partial_retirement_resume_never_republishes_deleted_weights(case,monkeypatch):
    count=[0]
    def interrupted(row):
        case.unlink(row);count[0]+=1
        if count[0]==1:raise InterruptedError('crash between unlink and journal')
    monkeypatch.setattr(finish,'_unlink_weight',interrupted)
    with pytest.raises(InterruptedError):finish.finish(case.root,case.export)
    assert (case.root/'local-retirement-plan.json').exists()
    assert not (case.root/'local-retirement-complete.json').exists()
    monkeypatch.setattr(finish,'_unlink_weight',case.unlink)
    monkeypatch.setattr(publish,'publish',lambda *a:pytest.fail('must not republish during retirement resume'))
    monkeypatch.setattr(finish,'_run_ingest',lambda *a:pytest.fail('must not rerun ingest during retirement resume'))
    assert finish.finish(case.root,case.export)['retired_weight_shards']==6
    assert not any(case.export.glob('*.safetensors'))
    assert not any(case.snapshot.glob('*.safetensors'))


def test_fully_retired_retry_rechecks_managed_identity_without_local_weights(case,monkeypatch):
    first=finish.finish(case.root,case.export)
    calls=len(case.calls)
    monkeypatch.setattr(publish,'publish',lambda *a:pytest.fail('retired publication replay'))
    assert finish.finish(case.root,case.export)==first
    fresh=case.calls[calls:]
    assert sum(c[0]=='remote_hash' for c in fresh)==len(case.rows)+1
    assert not any(c[0]=='unlink' for c in fresh)


def test_partial_ingest_or_changed_managed_hash_never_retires(case):
    case.ingest_error[0]=True
    with pytest.raises(RuntimeError,match='interrupted ingest'):finish.finish(case.root,case.export)
    assert not any(c[0]=='unlink' for c in case.calls)
    assert all((case.export/r['relative']).exists() for r in case.rows)
    case.ingest_error[0]=False;case.bad_hash[0]=case.plan['files'][-1]['namespace']
    with pytest.raises(ValueError,match='checksum changed'):finish.finish(case.root,case.export)
    assert not any(c[0]=='unlink' for c in case.calls)
    assert not (case.root/'local-retirement-plan.json').exists()


def test_existing_ref_change_blocks_all_retirement(case):
    case.ref_error[0]=True
    with pytest.raises(ValueError,match='HF ref changed'):finish.finish(case.root,case.export)
    assert not any(c[0]=='unlink' for c in case.calls)


def test_local_corruption_rejected_before_any_unlink(case):
    path=case.export/next(r['relative'] for r in case.rows if r['weight'])
    path.write_bytes(b'wrong bytes')
    with pytest.raises(ValueError,match='committed file changed'):finish.finish(case.root,case.export)
    assert not any(c[0]=='unlink' for c in case.calls)
    assert not (case.root/'local-retirement-plan.json').exists()


def test_arbitrary_public_cache_path_is_not_accepted(case):
    path=case.root/'publication-complete.json';marker=json.loads(path.read_text())
    marker['local_snapshot']=str(case.work/'unrelated-cache');atomic_json(path,marker)
    with pytest.raises(ValueError,match='approved publisher cache'):finish.finish(case.root,case.export)
    assert not any(c[0]=='unlink' for c in case.calls)


def test_symlink_weight_is_left_untouched(case):
    name=next(r['relative'] for r in case.rows if r['weight'])
    protected=case.work.parent/'protected-source';protected.write_bytes(b'BF16 source')
    (case.snapshot/name).unlink();(case.snapshot/name).symlink_to(protected)
    with pytest.raises(ValueError,match='symlink'):finish.finish(case.root,case.export)
    assert protected.read_bytes()==b'BF16 source'
    assert not any(c[0]=='unlink' for c in case.calls)


def test_unrecorded_missing_weight_is_not_treated_as_partial_retirement(case):
    (case.snapshot/next(r['relative'] for r in case.rows if r['weight'])).unlink()
    with pytest.raises(ValueError,match='missing before retirement'):finish.finish(case.root,case.export)
    assert not any(c[0]=='unlink' for c in case.calls)


def test_audit_mismatch_and_old_task_paths_are_rejected(case):
    path=case.root/'completed-export-audit.json';audit=json.loads(path.read_text());audit['identity']={'wrong':'producer'};atomic_json(path,audit)
    with pytest.raises(ValueError,match='does not match'):finish.finish(case.root,case.export)
    with pytest.raises(ValueError,match='active clean K128'):finish.finish(case.work/'production',case.work/'export')
    with pytest.raises(ValueError,match='authorized local'):finish._owned('/mnt/sparknest/hf-home/hub')
    assert case.calls==[]


def test_terminal_retired_name_tampering_is_rejected(case):
    finish.finish(case.root,case.export)
    path=case.root/'local-retirement-complete.json';value=json.loads(path.read_text());value['retired_names']=[];atomic_json(path,value)
    with pytest.raises(ValueError,match='terminal retirement identity'):finish.finish(case.root,case.export)


def test_anonymous_public_metadata_checks_every_shard_and_metadata_blob(case,monkeypatch):
    import huggingface_hub
    marker=json.loads((case.root/'publication-complete.json').read_text())
    siblings=[]
    for row in case.plan['files']:
        sibling=dict(rfilename=row['relative'],size=row['size'])
        if row['weight']:sibling['lfs']=dict(sha256=row['sha256'])
        else:sibling['blob_id']=publish._metadata_digests(Path(row['source']))[1]
        siblings.append(sibling)
    class API:
        def __init__(self,*,token):assert token is False
        def model_info(self,repo,*,revision,files_metadata,timeout):
            assert repo==finish.TARGET_REPO and revision==marker['revision'] and files_metadata and timeout==60
            return dict(sha=revision,private=False,siblings=siblings)
    monkeypatch.setattr(huggingface_hub,'HfApi',API)
    case.public_verify(marker,case.plan)
    next(s for s in siblings if s.get('lfs'))['lfs']['sha256']='0'*64
    with pytest.raises(ValueError,match='LFS checksum'):case.public_verify(marker,case.plan)
    assert not any(c[0]=='unlink' for c in case.calls)


def test_shared_metadata_owners_are_preserved_while_weights_stay_single_copy(case,monkeypatch):
    class SharedAssets(case.transport_type):
        def stable(self,namespace):
            result=super().stable(namespace)
            item=next((r for r in case.plan['files'] if r['namespace']==namespace),None)
            if item is not None and not item['weight']:result['hosts']+=['raptor (D2)','raptor (D3)']
            return result
    monkeypatch.setattr(ingest,'NativeTransport',SharedAssets)
    assert finish.finish(case.root,case.export)['retired_weight_shards']==6


def test_publication_cli_runs_full_finish_and_resumes_after_local_retirement(case,monkeypatch,capsys):
    from muse_spark.core_k6 import publish_job
    monkeypatch.setattr(sys,'argv',['publish_job.py','--root',str(case.root),'--export',str(case.export)])
    publish_job.main()
    result=json.loads(capsys.readouterr().out.splitlines()[-1])
    assert result==dict(event='publication_job_complete',result='a'*40,
        managed_ingest_complete=True,local_retirement_complete=True,retired_weight_shards=6)
    assert not any(case.export.glob('*.safetensors'))
    publications=sum(call[0]=='publish' for call in case.calls)
    publish_job.main()
    assert sum(call[0]=='publish' for call in case.calls)==publications
    assert json.loads(capsys.readouterr().out.splitlines()[-1])==result


def test_publication_cli_rejects_unmatched_independent_audit_before_publication(case,monkeypatch):
    from muse_spark.core_k6 import publish_job
    path=case.root/'completed-export-audit.json';audit=json.loads(path.read_text())
    audit['identity']={'different':'producer'};atomic_json(path,audit)
    monkeypatch.setattr(sys,'argv',['publish_job.py','--root',str(case.root),'--export',str(case.export)])
    with pytest.raises(ValueError,match='does not match'):publish_job.main()
    assert case.calls==[]
    assert all((case.export/row['relative']).exists() for row in case.rows)

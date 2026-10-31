"""CPU publication, managed ingest and narrowly scoped local weight retirement."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from kimi_k3.uniform_k2.storage import atomic_json,canonical,file_identity,fsync_directory,sha256_file,verify_receipt
from . import ingest,publish
from .recipe import TARGET_REPO,MODULE_COUNT,TARGET_PAYLOAD_BYTES,TARGET_TENSOR_COUNT

WORK_ROOT=Path.home()/'Developer/cute-worktrees/muse-quant-work'
SCHEMA='cuteafd-muse-local-retirement-v1'
ACTIVE_ROOT='production-k128'
ACTIVE_EXPORT='export-k128'


def _json(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('terminal task metadata is missing or unsafe')
    value=json.loads(path.read_text());canonical(value)
    return value


def _owned(path):
    """Only this task's local trees; never the source/default managed stores."""
    path=Path(path).expanduser().absolute()
    base=WORK_ROOT.expanduser().absolute()
    if path==base or not path.is_relative_to(base):
        raise ValueError('finish path is outside the authorized local Muse task tree')
    cursor=base
    if cursor.is_symlink():raise ValueError('authorized task tree is a symlink')
    for part in path.relative_to(base).parts:
        if part in ('','.', '..'):raise ValueError('unsafe terminal task path')
        cursor=cursor/part
        if cursor.is_symlink():raise ValueError('refuse symlink terminal task path')
    return path


def _admission(root,export):
    root,export=_owned(root),_owned(export)
    if root!=WORK_ROOT/ACTIVE_ROOT or export!=WORK_ROOT/ACTIVE_EXPORT:
        raise ValueError('finish may only retire the active clean K128 production/export')
    if (root==export or root.is_relative_to(export) or export.is_relative_to(root)
            or not root.is_dir() or not export.is_dir()):
        raise ValueError('final production and export must be separate existing task trees')
    identity=publish._complete_plan(export,root,TARGET_REPO)
    complete=_json(root/'quantization-complete.json');audit=_json(root/'completed-export-audit.json')
    summary=complete.get('summary',{})
    expected=dict(shards=publish.SHARD_COUNT,tensors=TARGET_TENSOR_COUNT,payload_bytes=TARGET_PAYLOAD_BYTES,
        quantized_projections=MODULE_COUNT)
    if (audit.get('status')!='passed' or audit.get('identity')!=identity or complete.get('identity')!=identity
            or any(summary.get(k)!=v for k,v in expected.items())
            or audit.get('tensors')!=summary['tensors'] or audit.get('payload_bytes')!=summary['payload_bytes']
            or audit.get('numerical_records')!=summary['quantized_projections']
            or set(audit.get('shards',{}))!=publish._expected_shards()):
        raise ValueError('independent export audit does not match the completed producer/inventory')
    for key in ('preserved_tensors','preserved_bytes'):
        if key in summary and audit.get(key)!=summary[key]:
            raise ValueError('independent retained-tensor audit differs from completion')
    for name,record in audit['shards'].items():
        receipt=_json(publish._receipt_path(export,name))
        if receipt.get('identity')!=identity or receipt.get('filename')!=name or record.get('sha256')!=receipt.get('sha256'):
            raise ValueError('independent export audit differs from a shard receipt')
    return root,export,identity,audit


def _publication(root,export,identity):
    marker=_json(root/'publication-complete.json')
    revision=marker.get('revision','')
    if (marker.get('repo_id')!=TARGET_REPO or marker.get('identity')!=identity
            or not re.fullmatch(r'[0-9a-f]{40}',revision)):
        raise ValueError('public completion marker differs from the approved producer')
    expected=publish._cache_snapshot(TARGET_REPO,revision).absolute()
    snapshot=_owned(expected)
    if Path(marker.get('local_snapshot','')).absolute()!=snapshot or not snapshot.is_dir():
        raise ValueError('public local snapshot differs from the approved publisher cache')
    return marker,snapshot


def _ingest_bindings(root,export,identity,marker):
    plan=_json(root/'ingest-plan.json');complete=_json(root/'ingest-complete.json')
    if (plan.get('schema')!=ingest.SCHEMA or plan.get('repo_id')!=TARGET_REPO
            or plan.get('producer')!=identity or plan.get('revision')!=marker['revision']
            or Path(plan.get('root','')).absolute()!=root or Path(plan.get('export','')).absolute()!=export
            or plan.get('published_manifest_sha256')!=marker.get('manifest_sha256')
            or plan.get('publication_marker_sha256')!=sha256_file(root/'publication-complete.json')
            or complete.get('schema')!=ingest.SCHEMA or complete.get('plan_sha256')!=ingest.digest(plan)
            or complete.get('repo_id')!=TARGET_REPO or complete.get('revision')!=marker['revision']
            or complete.get('published_manifest_sha256')!=marker.get('manifest_sha256')):
        raise ValueError('managed completion does not bind the public producer and placement plan')
    rows=plan.get('files',[]);names=[r['relative'] for r in rows]
    if len(set(names))!=len(names) or len(names)!=marker.get('files') or complete.get('file_count')!=len(names):
        raise ValueError('managed/public file inventory differs')
    manifest=[[r['relative'],r['size'],r['sha256']] for r in sorted(rows,key=lambda r:r['relative'])]
    if ingest.digest(manifest)!=marker.get('manifest_sha256'):
        raise ValueError('saved placement manifest differs from published objects')
    weights={r['relative'] for r in rows if r.get('weight')}
    if (weights!=publish._expected_shards() or complete.get('weight_shards')!=len(weights)
            or complete.get('weight_copies')!=len(weights) or set(complete.get('files',{}))!=set(names)):
        raise ValueError('managed checkpoint weight-copy inventory differs')
    expected_snapshot=f'{ingest.HUB}/models--{TARGET_REPO.replace("/","--")}/snapshots/{marker["revision"]}'
    if plan.get('snapshot_namespace')!=expected_snapshot:
        raise ValueError('managed namespace does not name the published checkpoint')
    for row in rows:
        relative=ingest.safe_relative(row['relative']);expected_source=export/relative
        if (Path(row['source']).absolute()!=expected_source or row.get('host') not in ingest.HOSTS
                or row.get('namespace')!=expected_snapshot+'/'+relative
                or bool(row.get('weight'))!=relative.endswith('.safetensors')
                or not re.fullmatch(r'[0-9a-f]{64}',row.get('sha256',''))
                or type(row.get('size'))is not int or row['size']<0):
            raise ValueError('saved placement source or destination is outside the approved artifact')
        record=complete['files'][relative]
        if any(record.get(k)!=row[k] for k in ('sha256','size','host','namespace','weight')):
            raise ValueError('managed per-file completion evidence differs')
    ref=plan.get('ref',{})
    expected_ref=f'{ingest.HUB}/models--{TARGET_REPO.replace("/","--")}/refs/main'
    if (ref.get('namespace')!=expected_ref or ref.get('host') not in ingest.HOSTS
            or ref.get('sha256')!=__import__('hashlib').sha256(marker['revision'].encode()).hexdigest()
            or ref.get('size')!=40 or complete.get('ref',{}).get('sha256')!=ref['sha256']):
        raise ValueError('managed HF ref completion differs')
    return plan,complete


def _verify_public(marker,plan):
    """Anonymous pinned metadata verification; never re-download weight bytes."""
    from huggingface_hub import HfApi
    info=HfApi(token=False).model_info(TARGET_REPO,revision=marker['revision'],files_metadata=True,timeout=60)
    if publish._field(info,'sha')!=marker['revision'] or publish._field(info,'private') is not False:
        raise ValueError('completed public revision is not anonymously accessible')
    siblings={publish._field(row,'rfilename'):row for row in publish._field(info,'siblings')}
    files={row['relative']:row for row in plan['files']}
    if set(siblings)!=set(files):
        raise ValueError('public pinned revision file inventory differs')
    for name,row in files.items():
        local=dict(size=row['size'],sha256=row['sha256'],blob_id=None)
        if not row['weight']:
            path=_owned(row['source'])
            if not path.is_file():raise ValueError('published scalar metadata was not retained locally')
            sha,blob=publish._metadata_digests(path)
            if sha!=row['sha256']:raise ValueError('retained published metadata changed')
            local['blob_id']=blob
        publish._audit_remote(name,local,siblings[name])


def _verify_managed(plan,complete,transport):
    """Read-only full checksum barrier before the first retirement unlink."""
    transport.readiness()
    if transport.read_ref(plan['ref']['namespace'])!=plan['revision'].encode():
        raise ValueError('managed HF ref changed after ingest')
    expected={row['relative'] for row in plan['files']}
    if transport.inventory(plan['snapshot_namespace'])!=expected:
        raise ValueError('managed checkpoint snapshot inventory differs')
    verified={}
    for row in plan['files']:
        entry=transport.stable(row['namespace'])
        hosts=[] if entry is None else entry.get('hosts',[])
        if (entry is None or entry.get('kind')!='regular' or entry.get('writing_on')
                or entry.get('size')!=row['size'] or not entry.get('sealed')
                or (row['weight'] and (len(hosts)!=1 or hosts!=[row['host']]))
                or (not row['weight'] and row['host'] not in hosts)):
            raise ValueError('managed file ownership, immutability or size changed')
        if transport.remote_sha256(row['host'],row['namespace'])!=row['sha256']:
            raise ValueError('managed file checksum changed; local weights retained')
        verified[row['relative']]=dict(sha256=row['sha256'],size=row['size'],owners=sorted(hosts))
    ref=transport.stable(plan['ref']['namespace'])
    if (ref is None or ref.get('kind')!='regular' or ref.get('writing_on')
            or ref.get('size')!=40 or plan['ref']['host'] not in ref.get('hosts',[])
            or transport.remote_sha256(plan['ref']['host'],plan['ref']['namespace'])!=plan['ref']['sha256']):
        raise ValueError('managed ref ownership or checksum changed')
    managed=transport.offline_snapshot(TARGET_REPO,plan['revision'])
    expected_path=str(ingest.MOUNT/plan['snapshot_namespace'].lstrip('/'))
    if managed!=expected_path or complete.get('managed_snapshot')!=expected_path:
        raise ValueError('managed offline HF resolution differs')
    result=dict(files=verified,ref_sha256=plan['ref']['sha256'],snapshot=managed)
    print(canonical(dict(event='final_managed_checksum_barrier_passed',files=len(verified),
        weight_shards=sum(row['weight'] for row in plan['files']),verification_sha256=ingest.digest(result))),flush=True)
    return result


def _run_ingest(root,export):
    command=[sys.executable,str(Path(ingest.__file__)), '--root',str(root),'--export',str(export)]
    subprocess.run(command+['--preflight'],check=True,timeout=900)
    subprocess.run(command,check=True,timeout=86400+300)


def _retirement_plan(root,export,identity,marker,snapshot,iplan):
    rows=[]
    for item in sorted((r for r in iplan['files'] if r['weight']),key=lambda r:r['relative']):
        receipt=_json(publish._receipt_path(export,item['relative']))
        if (receipt.get('sha256')!=item['sha256'] or receipt.get('identity')!=identity
                or receipt.get('file_identity')!=item['file_identity']):
            raise ValueError('retirement weight receipt differs from ingest source')
        for location,path in (('snapshot',snapshot/item['relative']),('export',export/item['relative'])):
            _owned(path)
            rows.append(dict(location=location,path=str(path),relative=item['relative'],sha256=item['sha256'],
                size=item['size'],file_identity=item['file_identity']))
    return dict(schema=SCHEMA,identity=identity,root=str(root),export=str(export),local_snapshot=str(snapshot),
        repo_id=TARGET_REPO,revision=marker['revision'],published_manifest_sha256=marker['manifest_sha256'],
        publication_marker_sha256=sha256_file(root/'publication-complete.json'),
        ingest_plan_sha256=ingest.digest(iplan),ingest_complete_sha256=sha256_file(root/'ingest-complete.json'),
        completed_audit_sha256=sha256_file(root/'completed-export-audit.json'),
        finish_code_sha256=sha256_file(Path(__file__)),paths=rows)


def _verify_local_paths(retirement,*,allow_missing):
    by_inode={};present=[]
    for row in retirement['paths']:
        path=_owned(row['path'])
        if not path.exists():
            if not allow_missing:raise ValueError('local weight missing before retirement was authorized')
            continue
        if not path.is_file():raise ValueError('local retirement target is not a regular weight file')
        verify_receipt(path,row)
        stat=path.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
        if key not in by_inode:by_inode[key]=sha256_file(path)
        if by_inode[key]!=row['sha256']:
            raise ValueError('local weight checksum differs; retirement refused')
        present.append(row)
    return present


def _unlink_weight(row):
    path=_owned(row['path'])
    verify_receipt(path,row)
    path.unlink()
    fsync_directory(path.parent)


def finish(root,export):
    """Run the authorized CPU continuation; retry safely after partial cleanup."""
    root,export,identity,audit=_admission(root,export)
    lock_path=root/'finish.lock'
    if lock_path.is_symlink():raise ValueError('unsafe task finish lock')
    with lock_path.open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise RuntimeError('another finisher owns this task') from error
        terminal_path=root/'local-retirement-complete.json'
        retirement_path=root/'local-retirement-plan.json'
        ingest_complete=root/'ingest-complete.json'
        if not ingest_complete.exists():
            if terminal_path.exists() or retirement_path.exists():
                raise ValueError('retirement state exists without matching managed completion')
            publish.publish(export,root,TARGET_REPO)
            _run_ingest(root,export)
        marker,snapshot=_publication(root,export,identity)
        iplan,icomplete=_ingest_bindings(root,export,identity,marker)
        retirement=_retirement_plan(root,export,identity,marker,snapshot,iplan)
        existing=retirement_path.exists()
        if existing and _json(retirement_path)!=retirement:
            raise ValueError('immutable local retirement plan differs; nothing unlinked')
        if terminal_path.exists():
            terminal=_json(terminal_path)
            expected_names=sorted({row['relative'] for row in retirement['paths']})
            if (terminal.get('schema')!=SCHEMA or terminal.get('identity')!=identity
                    or terminal.get('retirement_plan_sha256')!=ingest.digest(retirement)
                    or terminal.get('revision')!=marker['revision']
                    or terminal.get('repo_id')!=TARGET_REPO
                    or terminal.get('published_manifest_sha256')!=marker['manifest_sha256']
                    or terminal.get('ingest_plan_sha256')!=ingest.digest(iplan)
                    or terminal.get('ingest_complete_sha256')!=sha256_file(ingest_complete)
                    or terminal.get('retired_names')!=expected_names
                    or terminal.get('retired_paths')!=[row['path'] for row in retirement['paths']]
                    or terminal.get('retired_weight_shards')!=len(expected_names)
                    or terminal.get('retired_payload_bytes')!=sum(row['size'] for row in retirement['paths'] if row['location']=='export')
                    or terminal.get('unlinked_path_bytes')!=sum(row['size'] for row in retirement['paths'])):
                raise ValueError('terminal retirement identity differs')
            _verify_public(marker,iplan)
            _verify_managed(iplan,icomplete,ingest.NativeTransport())
            if any(_owned(row['path']).exists() for row in retirement['paths']):
                raise ValueError('retired local weight unexpectedly reappeared')
            return terminal
        _verify_public(marker,iplan)
        verification=_verify_managed(iplan,icomplete,ingest.NativeTransport())
        present=_verify_local_paths(retirement,allow_missing=existing)
        if not existing:atomic_json(retirement_path,retirement)
        state_path=root/'local-retirement-state.json'
        state=dict(schema=SCHEMA,retirement_plan_sha256=ingest.digest(retirement),removed=[])
        if state_path.exists():
            state=_json(state_path)
            if (state.get('retirement_plan_sha256')!=ingest.digest(retirement)
                    or not set(state.get('removed',[]))<={row['path'] for row in retirement['paths']}):
                raise ValueError('local retirement recovery state differs')
        for row in present:
            _unlink_weight(row)
            if row['path'] not in state['removed']:state['removed'].append(row['path'])
            atomic_json(state_path,state)
        if any(_owned(row['path']).exists() for row in retirement['paths']):
            raise ValueError('local task weight retirement is incomplete')
        names=sorted({row['relative'] for row in retirement['paths']})
        result=dict(schema=SCHEMA,identity=identity,repo_id=TARGET_REPO,revision=marker['revision'],
            retirement_plan_sha256=ingest.digest(retirement),ingest_plan_sha256=ingest.digest(iplan),
            ingest_complete_sha256=sha256_file(ingest_complete),published_manifest_sha256=marker['manifest_sha256'],
            managed_verification_sha256=ingest.digest(verification),retired_names=names,
            retired_paths=[row['path'] for row in retirement['paths']],retired_weight_shards=len(names),
            retired_payload_bytes=sum(row['size'] for row in retirement['paths'] if row['location']=='export'),
            unlinked_path_bytes=sum(row['size'] for row in retirement['paths']),
            finished=datetime.now(timezone.utc).isoformat())
        atomic_json(terminal_path,result)
        print(canonical(dict(event='publication_ingest_retirement_complete',revision=marker['revision'],
            shards=len(names),retired_payload_bytes=result['retired_payload_bytes'])),flush=True)
        return result

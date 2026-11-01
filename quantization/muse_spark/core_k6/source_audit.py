"""Audit immutable Muse source bytes against pinned Hub LFS/Git metadata."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'quantization'))
if __package__ in (None,''):
    __package__='muse_spark.core_k6'
from glm5_flash.core_k6.source_audit import hash_file
from kimi_k3.uniform_k2.storage import atomic_json, sha256_file
from .recipe import BASE_REPO, BASE_REVISION

LOCAL_STORES={'raptor','raptor (D2)','raptor (D3)'}


def require_local_source(snapshot, *, runner=subprocess.run):
    """Accept sealed local files spread across raptor's three managed disks."""
    snapshot=Path(snapshot).expanduser().absolute()
    if snapshot.name!=BASE_REVISION:
        raise ValueError('source revision differs from reviewed Muse release')
    records={}
    for path in sorted(snapshot.iterdir()):
        if not path.is_file():
            raise ValueError('source snapshot contains an unexpected non-file')
        resolved=path.resolve(strict=True)
        if not resolved.is_relative_to('/mnt/sparknest'):
            raise ValueError('Muse source must remain SparkNest-managed')
        result=runner(['nest','--json','ls',str(resolved)],check=True,text=True,capture_output=True,timeout=30)
        entries=json.loads(result.stdout).get('entries',[])
        if len(entries)!=1:
            raise ValueError(f'no unique managed source blob: {path.name}')
        entry=entries[0]
        hosts=set(entry.get('hosts',[]))
        if (entry.get('kind')!='regular' or not entry.get('sealed') or entry.get('writing_on')
                or not hosts & LOCAL_STORES or entry.get('name')!=resolved.name
                or entry.get('size')!=resolved.stat().st_size):
            raise ValueError(f'Muse source not immutable/local-ready: {path.name}')
        records[path.name]=dict(path=str(resolved),bytes=entry['size'],hosts=sorted(hosts & LOCAL_STORES))
    if not records:
        raise ValueError('empty Muse source snapshot')
    return dict(status='ready',files=records,bytes=sum(r['bytes'] for r in records.values()),
        local_stores=sorted(LOCAL_STORES),method='per-resolved-blob sealed, nonwriting, local-copy union')


def hub_manifest():
    """Fetch only official metadata; tensor/tokenizer bytes are read via nest."""
    from huggingface_hub import HfApi
    info=HfApi().model_info(BASE_REPO,revision=BASE_REVISION,files_metadata=True)
    if info.sha!=BASE_REVISION:
        raise ValueError('Hub source revision differs')
    files={}
    for entry in info.siblings:
        name=entry.rfilename
        if Path(name).name!=name:
            raise ValueError('unreviewed nested Muse source asset')
        if entry.lfs:
            sha=entry.lfs.sha256
            if not re.fullmatch(r'[0-9a-f]{64}',sha):
                raise ValueError('invalid Hub LFS SHA256')
            files[name]=dict(bytes=entry.size,sha256=sha)
        else:
            if not re.fullmatch(r'[0-9a-f]{40}',entry.blob_id):
                raise ValueError('invalid Hub Git blob SHA1')
            files[name]=dict(bytes=entry.size,git_blob_sha1=entry.blob_id)
    return dict(repo_id=BASE_REPO,revision=BASE_REVISION,files=files)


def audit(snapshot,manifest,output,*,workers=2):
    snapshot=Path(snapshot).expanduser().absolute()
    manifest_path=Path(manifest)
    if manifest_path.exists():
        plan=json.loads(manifest_path.read_text())
    else:
        plan=hub_manifest();atomic_json(manifest_path,plan)
    if plan.get('repo_id')!=BASE_REPO or plan.get('revision')!=BASE_REVISION:
        raise ValueError('source Hub manifest identity differs')
    readiness=require_local_source(snapshot)
    if set(readiness['files'])!=set(plan['files']):
        raise ValueError('source file inventory differs from pinned Hub metadata')
    files=sorted(plan['files']);records={}
    def verify(name):
        return hash_file(snapshot/name,plan['files'][name])
    if not 1<=workers<=2:
        raise ValueError('source audit supports at most two bounded readers')
    started=datetime.now(timezone.utc).isoformat()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name,record in zip(files,pool.map(verify,files),strict=True):
            records[name]=record
            print(json.dumps(dict(event='source_file_verified',name=name,**record)),flush=True)
    result=dict(status='passed',source=str(snapshot),repo_id=BASE_REPO,revision=BASE_REVISION,
        hub_manifest_sha256=sha256_file(manifest_path),files=records,
        bytes=sum(r['bytes'] for r in records.values()),readiness=readiness,
        started=started,finished=datetime.now(timezone.utc).isoformat())
    atomic_json(output,result)
    print(json.dumps(dict(event='source_audit_passed',files=len(records),bytes=result['bytes'])),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--hub-manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--ready-only',action='store_true')
    args=parser.parse_args()
    if args.ready_only:
        print(json.dumps(require_local_source(args.snapshot)),flush=True)
    else:
        audit(args.snapshot,args.hub_manifest,args.output,workers=args.workers)


if __name__=='__main__':
    main()

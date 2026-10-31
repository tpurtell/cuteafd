"""Audit pinned Hub content hashes and sealed copies across raptor's local stores."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'quantization'))
from kimi_k3.uniform_k2.storage import atomic_json,sha256_file
from glm5_flash.core_k6.source_audit import hash_file
from mistral_large4.core_k6.recipe import BASE_REPO,BASE_REVISION


def source_identity(path):
    resolved=Path(path).resolve(strict=True)
    info=resolved.stat()
    return dict(blob=str(resolved),bytes=info.st_size,mtime_ns=info.st_mtime_ns,inode=info.st_ino)


def local_source(snapshot):
    """D2 and D3 are physically local raptor disks, not missing downloads."""
    result=subprocess.check_output(['nest','ls','--json',str(Path(snapshot).parent.parent/'blobs')],text=True,timeout=60)
    entries={row['name']:row for row in json.loads(result)['entries']}
    for path in Path(snapshot).rglob('*'):
        if not path.is_file():
            continue
        # Native snapshot links point to this repository's managed blob names;
        # their IDs need not equal content hashes. Hashes are audited separately.
        name=Path(os.readlink(path)).name if path.is_symlink() else path.name
        row=entries.get(name)
        if row is not None and row.get('kind') == 'symlink':
            resolved=path.resolve(strict=True)
            found=json.loads(subprocess.check_output(['nest','ls','--json',str(resolved)],text=True,timeout=60))
            row=next((item for item in found['entries'] if item['name']==resolved.name), None)
        if (row is None or not row.get('sealed') or row.get('writing_on') is not None
                or not any(host=='raptor' or host.startswith('raptor (D') for host in row.get('hosts',[]))):
            raise ValueError('source not fully sealed on a physical raptor disk: '+str(path))
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--hub-manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=args.snapshot.resolve(strict=True)
    manifest=json.loads(args.hub_manifest.read_text())
    if source.name!=BASE_REVISION or manifest['revision']!=BASE_REVISION or manifest['repo_id']!=BASE_REPO:
        raise ValueError('source revision differs')
    actual={str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()}
    if actual!=set(manifest['files']):
        raise ValueError(f'source file inventory differs: extra={actual-set(manifest["files"])} missing={set(manifest["files"])-actual}')
    local_source(source)
    files=sorted(manifest['files'])
    def run(name):
        p=Path(name)
        if p.is_absolute() or '..' in p.parts:
            raise ValueError('invalid source manifest path')
        before=source_identity(source/p)
        result=hash_file(source/p,manifest['files'][name])
        if source_identity(source/p)!=before:
            raise ValueError('sealed source identity changed during hashing')
        return dict(result,file_identity=before)
    records={}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for name,record in zip(files,pool.map(run,files),strict=True):
            records[name]=record
            print(json.dumps(dict(event='source_file_verified',name=name,**record)),flush=True)
    atomic_json(args.output,dict(status='passed',source=str(source),repo_id=BASE_REPO,revision=BASE_REVISION,
        hub_manifest_sha256=sha256_file(args.hub_manifest),files=records,bytes=sum(row['bytes'] for row in records.values())))

if __name__=='__main__':
    main()

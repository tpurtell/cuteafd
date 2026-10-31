"""Read-only admission before queueing and again when GPU1 is granted."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import runpy
import socket
import subprocess
import sys
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'quantization'))
from kimi_k3.uniform_k2.storage import sha256_file
from mistral_large4.core_k6.container import IMAGE,GPU_UUID,BUILD
from mistral_large4.core_k6.source_audit import local_source,source_identity
from mistral_large4.core_k6.recipe import BASE_REPO,BASE_REVISION


def verify_source(snapshot, audit_path):
    audit=json.loads(Path(audit_path).read_text())
    if (audit.get('status')!='passed' or audit.get('repo_id')!=BASE_REPO
            or audit.get('revision')!=BASE_REVISION or audit.get('source')!=str(snapshot)
            or len(audit.get('files',{}))!=68):
        raise ValueError('complete pinned source checksum audit required')
    for name in ('config.json','consolidated.safetensors.index.json','tekken.json'):
        if sha256_file(snapshot/name)!=audit['files'][name]['sha256']:
            raise ValueError('source asset changed after checksum audit')
    for name,receipt in audit['files'].items():
        if source_identity(snapshot/name)!=receipt.get('file_identity'):
            raise ValueError('sealed source object changed after checksum audit: '+name)
    local_source(snapshot)


def verify_code():
    if socket.gethostname().lower()!='raptor':
        raise ValueError('Mistral quantization is authorized only on raptor GPU1')
    for directory,paths in ((REPO,['quantization/mistral_large4']), (REPO/'third_party/gptqmodel',[])):
        if subprocess.check_output(['git','-C',str(directory),'status','--porcelain','--',*paths],text=True).strip():
            raise ValueError('uncommitted production source')
    fork=REPO/'third_party/gptqmodel'
    commit=subprocess.check_output(['git','-C',str(fork),'rev-parse','HEAD'],text=True).strip()
    lock=json.loads((REPO/'third_party/gptqmodel.lock.json').read_text())
    digest=runpy.run_path(str(REPO/'scripts/build/verify-sparkinfer-source.py'))['source_tree_sha256'](fork)
    if commit!=lock['revision'] or digest!=lock['source_tree_sha256']:
        raise ValueError('GPTQModel pin or source tree differs')
    return commit


def qualified(path,commit):
    report=json.loads(Path(path).read_text())
    if report.get('status')!='passed' or report.get('identity',{}).get('gptqmodel_commit')!=commit:
        raise ValueError('passing matching bounded pilot required')
    code=report['identity'].get('code',{})
    if not code:
        raise ValueError('pilot lacks numerical source attestation')
    for name,digest in code.items():
        path=REPO/name
        if sha256_file(path)!=digest:
            raise ValueError('numerical source changed after pilot: '+name)
    if report.get('native_boundaries_byte_exact') is not True or report.get('serialized_projection_replay') is not True:
        raise ValueError('pilot lacks native and serialized replay proof')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--publication-only',action='store_true')
    parser.add_argument('--source-audit',type=Path,required=True)
    parser.add_argument('--qualification-report',type=Path)
    parser.add_argument('arguments',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    arguments=args.arguments[1:] if args.arguments[:1]==['--'] else args.arguments
    commit=verify_code()
    if not args.publication_only:
        snapshot=Path(arguments[arguments.index('--source-snapshot')+1]).resolve()
        verify_source(snapshot,args.source_audit)
        if not args.pilot:
            qualified(args.qualification_report,commit)
        device=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid,name,memory.total','--format=csv,noheader,nounits'],text=True,timeout=30)
        if GPU_UUID not in device or 'RTX PRO 6000' not in device or int(device.strip().rsplit(',',1)[1])<96000:
            raise ValueError('GPU1 UUID/card differs from authorized device')
        subprocess.run(['docker','image','inspect',IMAGE],check=True,stdout=subprocess.DEVNULL,timeout=30)
        receipt=json.loads((BUILD/'exllamav3/build-receipt.json').read_text())
        if receipt['fork_commit']!=commit or sha256_file(receipt['binary'])!=receipt['sha256']:
            raise ValueError('precompiled EXL3 extension identity differs')
    runner=Path(__file__).with_name('pilot.py' if args.pilot else 'run.py')
    subprocess.run([str(REPO/'.venv/bin/python'),str(runner),*arguments,'--preflight'],check=True,timeout=300)
    print(json.dumps(dict(admission='passed',host='raptor',gpu=GPU_UUID,pilot=args.pilot)),flush=True)

if __name__=='__main__':
    main()

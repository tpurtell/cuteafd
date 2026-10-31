"""Fail-closed CPU admission before enqueue and again when GPU0 is granted."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import socket
import runpy
import subprocess
import sys
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'quantization'))
if __package__ in (None,''):__package__='muse_spark.core_k6'
from kimi_k3.uniform_k2.storage import sha256_file
from .source_audit import require_local_source
from .recipe import BASE_REPO,BASE_REVISION
from .container import command,IMAGE


def verify_source(snapshot,audit_path):
    audit=json.loads(Path(audit_path).read_text())
    if (audit.get('status')!='passed' or audit.get('repo_id')!=BASE_REPO
            or audit.get('revision')!=BASE_REVISION or len(audit.get('files',{}))!=78):
        raise ValueError('full pinned source Hub checksum audit is required')
    require_local_source(snapshot)
    for name in ('config.json','model.safetensors.index.json'):
        if sha256_file(snapshot/name)!=audit['files'][name]['sha256']:
            raise ValueError('source metadata changed after audit')


def verify_tree():
    for root,paths in ((REPO,['quantization/muse_spark']),(REPO/'third_party/gptqmodel',[])):
        if subprocess.check_output(['git','-C',str(root),'status','--porcelain','--',*paths],text=True).strip():
            raise ValueError('Muse numerical source must be committed before hardware admission')
    fork=REPO/'third_party/gptqmodel'
    lock=json.loads((REPO/'third_party/gptqmodel.lock.json').read_text())
    revision=subprocess.check_output(['git','-C',str(fork),'rev-parse','HEAD'],text=True).strip()
    digest=runpy.run_path(str(REPO/'scripts/build/verify-sparkinfer-source.py'))['source_tree_sha256'](fork)
    if revision!=lock['revision'] or digest!=lock['source_tree_sha256']:
        raise ValueError('GPTQModel pinned source tree differs')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--qualification-report',type=Path)
    parser.add_argument('--source-audit',type=Path,required=True)
    parser.add_argument('arguments',nargs=argparse.REMAINDER)
    args=parser.parse_args();common=args.arguments[1:] if args.arguments[:1]==['--'] else args.arguments
    if socket.gethostname().lower()!='raptor':raise ValueError('Muse GPU quant is authorized on raptor only')
    verify_tree()
    try:snapshot=Path(common[common.index('--source-snapshot')+1])
    except (ValueError,IndexError):parser.error('source snapshot required')
    verify_source(snapshot,args.source_audit)
    if not args.pilot:
        if not args.qualification_report:raise ValueError('short pilot qualification required')
        from .pilot import validate_report
        validate_report(args.qualification_report)
    runner=Path(__file__).with_name('pilot.py' if args.pilot else 'run.py')
    subprocess.run(command([str(runner),*common,'--preflight'],name='muse-quant-preflight',gpu=False),check=True,timeout=240)
    print(json.dumps(dict(admission='passed',hardware='raptor GPU0 only',pilot=args.pilot)),flush=True)

if __name__=='__main__':main()

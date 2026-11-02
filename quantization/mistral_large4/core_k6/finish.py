"""Authorized CPU publication followed by one distributed managed checkpoint."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-snapshot','tokens','root','export'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--publish-only',action='store_true',help='CPU-only publication and storage finalization')
    args=parser.parse_args()
    root=args.root.expanduser().resolve()
    # Archive completion deliberately retires standalone weight paths. Never
    # invoke the publisher against those retired paths on an idempotent retry.
    if (root/'archive-complete.json').exists():
        from mistral_large4.core_k6.archive import archive
        archive(root)
        return
    common=[]
    for name in ('source_snapshot','tokens','root','export'):
        common+=['--'+name.replace('_','-'),str(getattr(args,name).expanduser().resolve())]
    if args.resume:
        common+=['--resume']
    subprocess.run([sys.executable,str(Path(__file__).with_name('run.py')),*common,'--publish-only'],
                   check=True,timeout=7*24*3600)
    from mistral_large4.core_k6.archive import archive
    archive(root)


if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    main()

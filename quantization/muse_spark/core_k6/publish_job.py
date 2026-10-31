"""CPU-only continuation after the owned GPU container and lease are released."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'quantization'))
if __package__ in (None,''):__package__='muse_spark.core_k6'
from kimi_k3.uniform_k2.storage import atomic_json
from .recipe import TARGET_REPO
from .publish import publish


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--export',type=Path,required=True)
    args=parser.parse_args()
    audit=json.loads((args.root/'completed-export-audit.json').read_text())
    complete=json.loads((args.root/'quantization-complete.json').read_text())
    if audit.get('status')!='passed' or audit.get('identity')!=complete.get('identity'):
        raise ValueError('independent completed export audit is required before upload')
    result=publish(args.export,args.root,TARGET_REPO)
    print(json.dumps(dict(event='publication_job_complete',result=result)),flush=True)

if __name__=='__main__':main()

"""CPU-only continuation after the owned GPU container and lease are released."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'quantization'))
if __package__ in (None,''):__package__='muse_spark.core_k6'
from .finish import finish


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--export',type=Path,required=True)
    args=parser.parse_args()
    # The finisher owns the audit gate and publication/ingest/retirement
    # recovery barriers. Repeating the CPU continuation after local weights
    # retire must not replay a publisher that requires those deleted files.
    result=finish(args.root,args.export)
    print(json.dumps(dict(event='publication_job_complete',result=result['revision'],
        managed_ingest_complete=True,local_retirement_complete=True,
        retired_weight_shards=result['retired_weight_shards'])),flush=True)


if __name__=='__main__':main()

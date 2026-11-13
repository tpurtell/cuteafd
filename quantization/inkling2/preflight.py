"""CPU-only maxq admission, run before enqueue and again after lease grant."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import socket
import subprocess

from .source import HeaderSource
from . import recipe

REPO = Path(__file__).resolve().parents[2]
TASK = Path.home()/'.cache/cuteafd/quantization/inkling2-k4-mc'
BUILD = Path.home()/'.cache/cuteafd/builds/inkling2-k4-mc'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--production', action='store_true')
    args = parser.parse_args()
    if socket.gethostname().lower() != 'maxq':
        raise ValueError('only maxq is authorized for this quantization')
    subprocess.run(['python3', str(REPO/'scripts/build/assert-build-filesystem.py'), str(BUILD)],
                   check=True, timeout=30)
    header = HeaderSource(args.source)
    corpus = json.loads((TASK/'prepared-corpus/manifest.json').read_text())
    if corpus.get('status') != 'passed' or corpus['source']['revision'] != recipe.BASE_REVISION:
        raise ValueError('prepared corpus provenance differs')
    if shutil.disk_usage(TASK).free < 700 * 1024**3:
        raise ValueError('less than 700 GiB task NVMe headroom')
    if args.production:
        from .pilot import successful
        report = json.loads((TASK/'pilot.json').read_text())
        if report.get('passed') is not True or not successful(report):
            raise ValueError('the bounded smoke has not passed')
        from .run import plan
        plan(header, TASK/'prepared-corpus')
    print(json.dumps(dict(admission='passed', hardware='maxq only',
        source_tensors=header.tensor_count, source_bytes=header.payload_bytes)), flush=True)


if __name__ == '__main__':
    main()

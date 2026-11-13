#!/usr/bin/env python3
"""Read-only K3.1 admission, shared by queue enrollment and grant checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import shutil
import socket
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'quantization'))
if __package__ in (None, ''):
    __package__ = 'kimi_k31.k2_core_k6'


def verify_fork():
    fork = REPO / 'third_party/gptqmodel'
    for directory, paths in ((REPO, ['quantization/kimi_k31', 'quantization/kimi_k3/prepare_corpus.py',
                                    'python/reference/families/kimi_k3']), (fork, [])):
        if subprocess.check_output(['git', '-C', str(directory), 'status', '--porcelain', '--', *paths],
                                   text=True, timeout=30).strip():
            raise ValueError('uncommitted K3.1 production source')
    actual = subprocess.check_output(['git', '-C', str(fork), 'rev-parse', 'HEAD'], text=True, timeout=30).strip()
    lock = json.loads((REPO / 'third_party/gptqmodel.lock.json').read_text())
    digest = runpy.run_path(str(REPO / 'scripts/build/verify-sparkinfer-source.py'))['source_tree_sha256'](fork)
    if actual != lock['revision'] or digest != lock['source_tree_sha256']:
        raise ValueError('GPTQModel pin or source-tree lock differs')
    return actual


def qualified(path, fork_commit):
    from .pilot import validate_qualification, validate_report_code
    report = validate_qualification(json.loads(Path(path).read_text()))
    if report['identity']['gptqmodel_commit'] != fork_commit:
        raise ValueError('pilot fork commit differs')
    validate_report_code(report)
    return report


def require_local_source():
    report = json.loads(subprocess.check_output(['nest', 'where',
        'hf:moonshotai/Kimi-K3.1-Pro', '--json'], text=True, timeout=60))
    host = next((entry for entry in report['hosts'] if entry['host'] == 'maxq'), None)
    if host is None or host.get('ready') is not True or host.get('missing_files') != 0:
        raise ValueError('the entire pinned source must be local in SparkNest on maxq')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualification-only', action='store_true')
    parser.add_argument('--publication-only', action='store_true')
    parser.add_argument('--qualification-report', type=Path, default=Path.home() /
        '.cache/cuteafd/quantization/kimi-k31-k2-core-k6/pilot/qualification.json')
    parser.add_argument('--python', type=Path, default=Path.home() /
        '.cache/cuteafd/quantization/kimi-k31-k2-core-k6/environment/bin/python')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if socket.gethostname().lower() != 'maxq':
        raise ValueError('K3.1 quantization is authorized only on maxq')
    if args.qualification_only and args.publication_only:
        parser.error('qualification and publication are separate modes')
    commit = verify_fork()
    if not args.qualification_only:
        qualified(args.qualification_report, commit)
    if not args.publication_only:
        require_local_source()
        smi = shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'
        devices = subprocess.check_output([smi, '--query-gpu=name,memory.total',
            '--format=csv,noheader,nounits'], text=True, timeout=30).strip().splitlines()
        if (len(devices) != 1 or 'RTX PRO 6000' not in devices[0]
                or int(devices[0].rsplit(',', 1)[1].strip()) < 96000):
            raise ValueError('expected exactly one local 96GB RTX PRO 6000')
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    runner = Path(__file__).with_name('pilot.py' if args.qualification_only else 'run.py')
    subprocess.run([str(args.python), str(runner), *arguments, '--preflight'], check=True, timeout=540)
    print(json.dumps(dict(admission='passed', host='maxq', fork_commit=commit,
        pilot=args.qualification_only, publication=args.publication_only)), flush=True)


if __name__ == '__main__':
    main()

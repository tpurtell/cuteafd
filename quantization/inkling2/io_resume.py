"""Explicit I/O-only resume entry; the frozen numerical runner stays unchanged."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from kimi_k3.uniform_k2.storage import atomic_json, canonical
from .output_io import OutputNumpyProxy, code_sha256


def assert_producer_lock(root):
    """Require the unchanged runner's flock to belong to this exact process."""
    stat = (Path(root)/'producer.lock').stat()
    for line in Path('/proc/locks').read_text().splitlines():
        fields = line.split()
        if len(fields) < 6 or fields[1] != 'FLOCK' or fields[3] != 'WRITE' or fields[4] != str(os.getpid()):
            continue
        major,minor,inode = fields[5].split(':')
        if (int(major,16),int(minor,16),int(inode)) == (os.major(stat.st_dev),os.minor(stat.st_dev),stat.st_ino):
            return
    raise ValueError('sequential output activation requires the frozen producer lock in this process')


def activation_guard(root, identity):
    assert_producer_lock(root)
    marker = Path(root)/'output-io-wrapper.json'
    expected = dict(schema='inkling2-io-only-wrapper-v1', frozen_numerical_identity=identity,
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        backend_sha256=code_sha256(),
        numerical_entry='unchanged inkling2.run.main with only run.np output-memmap delegation')
    if marker.exists():
        if json.loads(marker.read_text()) != expected:
            raise ValueError('sequential I/O wrapper provenance differs')
    else:
        atomic_json(marker,expected)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--root',type=Path,required=True)
    args,_ = parser.parse_known_args()
    root = args.root.expanduser().absolute()
    if root.resolve() != root:
        raise ValueError('producer root must not traverse symlinks')
    identity = json.loads((root/'identity.json').read_text())
    complete = root/'quantization-complete.json'
    if complete.exists():
        receipt = json.loads(complete.read_text())
        if receipt.get('status') != 'passed' or receipt.get('identity') != identity:
            raise ValueError('completed numerical job identity differs')
        print(json.dumps(dict(status='already_complete',numerical_job_restarted=False)),flush=True)
        return
    from . import run as frozen
    proxy = OutputNumpyProxy(root,identity,activation_guard=lambda:activation_guard(root,identity))
    frozen.np = proxy
    print(json.dumps(dict(output_storage='sequential-expert-files',
        frozen_numerical_identity_sha256=hashlib.sha256(canonical(identity).encode()).hexdigest())),flush=True)
    # Calling imported main preserves the proxy. runpy of the frozen module
    # would instantiate new globals and silently restore the old memmap.
    frozen.main()


if __name__ == '__main__':
    main()

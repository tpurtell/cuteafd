#!/usr/bin/env python3
"""Byte-exact toolchain identity (independent of kernel pins).

Dockerfile frontend/cache syntax changes intentionally require a new tc- tag:
keep this conservative rather than risk normalizing away content-affecting edits.
"""
import hashlib
from pathlib import Path

BASE = 'nvcr.io/nvidia/pytorch:26.05-py3@sha256:'
BASE_DIGESTS = {
    'amd64': 'ca73b4795f0d3ae27e9cd81b1b1f1b7fc6c0a129f7d51a359d2326e95af48a3d',
    'arm64': 'aa400d4373fa71f30e1714664beabcc64c2d198e72d65e9a3641440b07e7cc83',
}


INPUTS = ('docker/Dockerfile.dev', 'docker/entrypoint.sh',
          'scripts/build/install-dev-cache-tools.sh', 'scripts/build/install-nsys.sh',
          'scripts/build/dev-toolchain.py')


def identity(source):
    digest = hashlib.sha256()
    for name in INPUTS:
        digest.update(name.encode() + b'\0' + (source / name).read_bytes() + b'\0')
    return digest.hexdigest()[:16]


if __name__ == '__main__':
    import sys
    print(BASE + BASE_DIGESTS[sys.argv[1]] if len(sys.argv) > 1
          else identity(Path(__file__).resolve().parents[2]))

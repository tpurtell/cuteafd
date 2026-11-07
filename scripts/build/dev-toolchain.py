#!/usr/bin/env python3
"""Byte-exact toolchain identity (independent of kernel pins).

Dockerfile frontend/cache syntax changes intentionally require a new tc- tag:
keep this conservative rather than risk normalizing away content-affecting edits.
"""
import hashlib
from pathlib import Path

BASE = 'nvcr.io/nvidia/pytorch:26.09-py3@sha256:'
BASE_DIGESTS = {
    'amd64': 'f71fd67f6d71f685e0bd08ce6c8df48807624cc5d48886370d3471bc4f89d81d',
    'arm64': '307c449df7dabed6b36e017a2b2f2e45bdec9dc203b81507111315f1e971de6e',
}


def identity(source):
    digest = hashlib.sha256()
    for name in ('docker/Dockerfile.dev', 'docker/entrypoint.sh',
                 'scripts/build/install-dev-cache-tools.sh', 'scripts/build/dev-toolchain.py'):
        digest.update(name.encode() + b'\0' + (source / name).read_bytes() + b'\0')
    return digest.hexdigest()[:16]


if __name__ == '__main__':
    import sys
    print(BASE + BASE_DIGESTS[sys.argv[1]] if len(sys.argv) > 1
          else identity(Path(__file__).resolve().parents[2]))

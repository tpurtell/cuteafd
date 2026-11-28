#!/usr/bin/env python3
"""Select a checkout-matched published toolchain, or build it on this host."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

_spec = importlib.util.spec_from_file_location('dev_toolchain', Path(__file__).with_name('dev-toolchain.py'))
_toolchain = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_toolchain)
REGISTRY = 'ghcr.io/tpurtell/cuteafd-dev'
SHA256 = re.compile(r'sha256:[0-9a-f]{64}')


class ImageError(RuntimeError):
    pass


def inspect(image):
    result = subprocess.run(['docker', 'image', 'inspect', image], capture_output=True, text=True)
    if result.returncode:
        raise ImageError(result.stderr.strip() or f'image inspect failed: {image}')
    try:
        return json.loads(result.stdout)[0]
    except (ValueError, IndexError, TypeError) as error:
        raise ImageError(f'invalid image inspect response: {image}') from error


def verify(obj, arch, toolchain_hash, base):
    labels = obj.get('Config', {}).get('Labels') or {}
    if obj.get('Architecture') != arch:
        raise ImageError(f'architecture mismatch: expected {arch}, got {obj.get("Architecture")}')
    if labels.get('io.cuteafd.toolchain.hash') != toolchain_hash:
        raise ImageError('toolchain label mismatch')
    if labels.get('io.cuteafd.base.digest') != base:
        raise ImageError('base-digest label mismatch')
    if not SHA256.fullmatch(obj.get('Id', '')):
        raise ImageError('invalid immutable image ID')


def registry_digest(obj):
    for value in obj.get('RepoDigests', []):
        repository, separator, digest = value.partition('@')
        if separator and repository == REGISTRY and SHA256.fullmatch(digest):
            return digest
    raise ImageError('published image has no registry digest')


def write_manifest(path, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + '\n')


def local_candidates(arch, toolchain_hash):
    role = 'cuteafd-coordinator-dev' if arch == 'amd64' else 'cuteafd-spark-expert-dev'
    return (f'{role}:tc-{toolchain_hash}', f'cuteafd-dev-local:tc-{toolchain_hash}-{arch}')


def select(source, arch, engine_commit, output, mode='registry'):
    toolchain_hash = _toolchain.identity(source)
    base = _toolchain.BASE + _toolchain.BASE_DIGESTS[arch]
    reference = f'{REGISTRY}:tc-{toolchain_hash}-{arch}'
    obj = None
    pulled_digest = None
    local_reference = None
    rejected_registry_id = None
    reason = 'CUTEAFD_RELEASE_DEV_IMAGE_SOURCE=build'
    if mode == 'registry':
        try:
            try:
                obj = inspect(reference)
            except ImageError:
                print(f'pull {reference} on this host', file=sys.stderr)
                pull = subprocess.run(['docker', 'pull', reference], stdout=sys.stderr, stderr=subprocess.PIPE, text=True)
                if pull.returncode:
                    detail = pull.stderr.strip()
                    absent = any(marker in detail.lower() for marker in ('manifest unknown', 'not found', 'manifest_unknown'))
                    prefix = f'no published image for tc-{toolchain_hash}' if absent else 'pull failed'
                    raise ImageError(f'{prefix}: {detail}')
                obj = inspect(reference)
            verify(obj, arch, toolchain_hash, base)
            pulled_digest = registry_digest(obj)
        except ImageError as error:
            reason = str(error)
            if obj is not None:
                rejected_registry_id = obj.get('Id')
            obj = None
    if obj is None and mode == 'registry':
        # Reuse an already-built, verified local toolchain image (matching the
        # exact tc-hash, base digest and arch) instead of rebuilding a new image
        # ID on every release run, so WIP and release share one export-cache key.
        for candidate in local_candidates(arch, toolchain_hash):
            try:
                found = inspect(candidate)
                if rejected_registry_id and found.get('Id') == rejected_registry_id:
                    continue
                verify(found, arch, toolchain_hash, base)
            except ImageError:
                continue
            obj = found
            local_reference = candidate
            print(f'dev image fallback on {arch}: {reason}; reusing verified local {candidate}', file=sys.stderr)
            break
    if obj is None:
        # A private tag: the shared default (cuteafd-*-dev) moves only through
        # build-dev-images.sh, so a release fallback never retags it under running
        # WIP builds or export hosts. Callers use the returned immutable ID.
        local_tag = f'cuteafd-dev-local:tc-{toolchain_hash}-{arch}'
        print(f'dev image fallback on {arch}: {reason}; building locally from Dockerfile.dev', file=sys.stderr)
        subprocess.run(['docker', 'build', '--build-arg', f'BASE_IMAGE={base}',
                        '--build-arg', f'CUTEAFD_TOOLCHAIN_HASH={toolchain_hash}',
                        '--build-arg', f'CUTEAFD_ENGINE_COMMIT={engine_commit}',
                        '-f', str(source / 'docker/Dockerfile.dev'), '-t', local_tag, str(source)],
                       stdout=sys.stderr, check=True)
        obj = inspect(local_tag)
        verify(obj, arch, toolchain_hash, base)
    manifest = dict(schema='cuteafd.release-dev-image-reuse/2',
                    source='registry' if pulled_digest else ('local' if local_reference else 'build'),
                    architecture=arch,
                    image_id=obj['Id'], registry_reference=reference if pulled_digest else None,
                    registry_digest=pulled_digest, toolchain_hash=toolchain_hash, base_digest=base)
    if local_reference:
        manifest['local_reference'] = local_reference
    if not pulled_digest:
        manifest['fallback_reason'] = reason
    write_manifest(output, manifest)
    # Consumers run the exact inspected image, never its mutable registry/local tag.
    return obj['Id']


def labels(manifest):
    values = {'io.cuteafd.dev-image.reused': manifest['image_id'],
              'io.cuteafd.dev-image.source': manifest['source'],
              'io.cuteafd.dev-image.toolchain.hash': manifest['toolchain_hash']}
    if manifest.get('registry_reference'):
        values['io.cuteafd.dev-image.registry.reference'] = manifest['registry_reference']
        values['io.cuteafd.dev-image.registry.digest'] = manifest['registry_digest']
    for key, value in values.items():
        print('--label')
        print(f'{key}={value}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    selection = sub.add_parser('select')
    selection.add_argument('--source', type=Path, required=True)
    selection.add_argument('--arch', choices=('amd64', 'arm64'), required=True)
    selection.add_argument('--engine-commit', required=True)
    selection.add_argument('--output', type=Path, required=True)
    selection.add_argument('--mode', choices=('registry', 'build'), default='registry')
    for command in ('labels', 'image', 'record-override'):
        child = sub.add_parser(command)
        child.add_argument('--manifest', type=Path, required=True)
        if command == 'record-override':
            child.add_argument('--source', type=Path, required=True)
    merge = sub.add_parser('merge')
    merge.add_argument('--coordinator', type=Path, required=True)
    merge.add_argument('--spark', type=Path, required=True)
    merge.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'select':
        print(select(args.source.resolve(), args.arch, args.engine_commit, args.output, args.mode))
    elif args.command == 'merge':
        write_manifest(args.output, dict(schema='cuteafd.release-dev-image-reuse/2', legs={
            'coordinator': json.loads(args.coordinator.read_text()),
            'spark-expert': json.loads(args.spark.read_text())}))
    else:
        manifest = json.loads(args.manifest.read_text())
        if args.command == 'labels':
            labels(manifest)
        elif args.command == 'image':
            print(manifest['image_id'])
        else:
            # Keep the existing explicit override's live/BuildKit proof intact.
            manifest.update(source='override', architecture='amd64',
                            toolchain_hash=_toolchain.identity(args.source),
                            registry_reference=None, registry_digest=None)
            write_manifest(args.manifest, manifest)


if __name__ == '__main__':
    try:
        main()
    except (ImageError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f'dev image selection failed: {error}', file=sys.stderr)
        raise SystemExit(1)

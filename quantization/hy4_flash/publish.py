"""Publish the sealed Hy4-flash K4 MC package through verified HTTP LFS batches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

from core_export import MUL1
import hub_lfs
from inkling2 import publish as hub

from . import recipe

TARGET_REPO = recipe.TARGET_REPO
FAMILY = 'hy4_flash'
ENGRAM_FILE = re.compile(r'engram-fp8-layer-(?:01|11|21|31)-part-[0-9]{5}\.safetensors')


def pinned_files(files):
    return {name for name in files if ENGRAM_FILE.fullmatch(name)}


def validate_mc(output, state, validation, *, repo, revision):
    identity = state['identity']
    if (identity.get('source_repo') != repo
            or identity.get('source_revision') != revision):
        raise ValueError('Hy4 publication requires the pinned source')
    manifest_path = hub.regular(output / 'cuteafd-mc.json')
    if hub.metadata_identity(manifest_path)['sha256'] != validation['new_file_sha256'].get(manifest_path.name):
        raise ValueError('Hy4 manifest differs from package validation')
    manifest = hub.load(manifest_path)
    if (identity.get('codebook') != 'mul1' or identity.get('codebook_multiplier') != MUL1
            or manifest.get('exl3_codebook') != dict(name='mul1', multiplier=MUL1)):
        raise ValueError('Hy4 publication requires the validated MUL1 package')
    if (manifest.get('source') != dict(repo=repo, revision=revision)
            or manifest.get('original_core_dtype') != 'BF16'):
        raise ValueError('Hy4 publication requires the sealed three-view manifest')
    if {row['name'] for row in manifest['variants']} != {'original', 'k8-compact', 'k6-compact'}:
        raise ValueError('Hy4 publication requires all three core views')
    return manifest


def _validate_package(output, state, validation):
    manifest = validate_mc(output, state, validation, repo=recipe.BASE_REPO, revision=recipe.BASE_REVISION)
    proof = state.get('engram_fp8_proof', {})
    engram = manifest.get('engram_fp8', {})
    layers = engram.get('layers', [])
    if (set(proof) != set(recipe.ENGRAM_TABLES)
            or {row['tensor'] for row in layers} != set(recipe.ENGRAM_TABLES)
            or len(layers) != len(recipe.ENGRAM_TABLES)
            or not engram.get('recipe', '').endswith('; reference=' + recipe.REFERENCE_REVISION)):
        raise ValueError('Hy4 publication requires the pinned four-table Engram proof')
    parts = {}
    for layer in layers:
        tensor = layer['tensor']
        rows = recipe.ENGRAM_TABLES[tensor]
        if (proof[tensor].get('reference_dtype') != 'BF16'
                or proof[tensor].get('rows') != rows or proof[tensor].get('mismatch_count') != 0
                or sum(part['rows'] for part in layer['parts']) != rows):
            raise ValueError('Engram all-row BF16 proof is missing or disagrees')
        for part in layer['parts']:
            name = part['file']
            if name in parts or state['new_shards'][name]['sha256'] != part['sha256']:
                raise ValueError('Engram manifest differs from sealed shard receipt')
            parts[name] = part
    if set(parts) != pinned_files(state['new_shards']) or not parts:
        raise ValueError('Engram alternate/shard inventory differs')


TARGET = hub.PublicationTarget(TARGET_REPO, FAMILY, 'Publish calibrated Hy4-flash K4 MC checkpoint',
                               _validate_package)


def _check_numerical_completion(root, output, target=TARGET, revision=recipe.BASE_REVISION):
    complete = hub.load(hub.regular(root / 'numerical-complete.json'))
    state = hub.load(hub.regular(output / 'package-state.json'))
    identity = dict(complete['identity'], schema='cuteafd-core-package-v1', family=target.family)
    if (complete.get('status') != 'passed' or complete.get('output') != str(output)
            or complete.get('repository') != target.repo_id or complete.get('revision') != revision
            or complete.get('package_validation') != str(output / 'package-validation.json')
            or identity != state['identity']):
        raise ValueError('numerical completion does not match the Hy4 export')


def run_cli(target, revision):
    from huggingface_hub import HfApi

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    hub_lfs.add_arguments(parser)
    args = parser.parse_args()
    # No implicit login or token persistence; the completed receipt is public.
    root, output = args.root.resolve(), args.output.absolute()
    token = None
    if not (root / 'publication-complete.json').is_file():
        _check_numerical_completion(root, output, target, revision)
        if sys.stdin.isatty():
            parser.error('provide the write token through stdin')
        token = sys.stdin.readline(1024).strip()
        if not re.fullmatch(r'hf_[A-Za-z0-9]+', token):
            parser.error('the write token on stdin is missing or malformed')
    result = hub.publish(output, root, api=HfApi(token=token or False), target=target,
                         options=hub_lfs.options_from(args))
    print(json.dumps({key: result[key] for key in ('status', 'repo_id', 'revision', 'url')}), flush=True)


def main():
    run_cli(TARGET, recipe.BASE_REVISION)


if __name__ == '__main__':
    main()

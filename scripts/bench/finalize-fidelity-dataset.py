#!/usr/bin/env python3
"""Finish a sealed fidelity dataset from own-model text and reference logits."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'python/reference'))
from fidelity_windows import validate_public_metadata, validate_public_text


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_media_input(config, entry):
    relative = Path(entry['path'])
    assert not relative.is_absolute() and relative.parts and all(part not in ('.', '..') for part in relative.parts), 'Unsafe media input path'
    path = config / relative
    assert path.resolve().is_relative_to(config.resolve()) and not path.is_symlink(), 'Unsafe media input path'
    assert path.is_file() and path.stat().st_size == entry['bytes'] and digest(path) == entry['sha256'], 'Media input size/checksum differs'
    if path.suffix == '.png':
        import zlib
        data = path.read_bytes()
        assert data[:8] == b'\x89PNG\r\n\x1a\n', 'Invalid PNG signature'
        offset, first = 8, True
        while offset < len(data):
            assert len(data) - offset >= 12, 'Truncated PNG chunk'
            size = int.from_bytes(data[offset:offset + 4], 'big')
            kind = data[offset + 4:offset + 8]
            end = offset + 12 + size
            assert end <= len(data), 'Truncated PNG chunk'
            assert zlib.crc32(data[offset + 4:end - 4]) == int.from_bytes(data[end - 4:end], 'big'), 'PNG chunk checksum differs'
            assert not first or (kind == b'IHDR' and size == 13), 'Invalid PNG header'
            assert kind not in (b'tEXt', b'iTXt', b'zTXt', b'eXIf'), 'PNG privacy metadata chunks are not allowed'
            offset, first = end, False
            if kind == b'IEND':
                assert size == 0 and offset == len(data), 'Invalid PNG end/trailing data'
                return path
        raise AssertionError('PNG lacks IEND')
    assert path.suffix in ('.bf16', '.json'), 'Unsupported declared media input format'
    return path


def validate_output_file(path, config, manifest):
    # Preserve exact upstream licence notices, including public business contacts.
    licences = [entry for entry in manifest.get('licence_files', [])
                if entry['path'] == 'CHECKPOINT_LICENSE'
                and path.resolve() == (config / 'CHECKPOINT_LICENSE').resolve()]
    if licences:
        assert len(licences) == 1 and digest(path) == licences[0]['sha256'], 'Licence checksum differs'
    elif path.suffix == '.json':
        validate_public_metadata(json.loads(path.read_text()))
    elif path.name == 'README.md':
        validate_public_text(path.read_text(), scored_text=True)
    else:
        media = [entry for entry in manifest.get('media_inputs', [])
                 if path.resolve() == (config / entry['path']).resolve()]
        if media:
            assert len(media) == 1, 'Duplicate media input'
            validate_media_input(config, media[0])
            if path.suffix in ('.png', '.bf16'):
                return
        validate_public_text(path.read_text(encoding='utf-8'))



def load_tool(name):
    path = Path(__file__).with_name(name + '.py')
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'out', 'tokenizer', 'license'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--config', required=True)
    parser.add_argument('--seal', action='store_true')
    args = parser.parse_args()
    validator = load_tool('validate-fidelity-dataset')
    result = validator.validate_reference_only(args.source, seal=args.seal)
    (args.source / 'reference-validation.json').write_text(json.dumps(result, indent=2) + '\n')
    packager = load_tool('package-reference-only-fidelity')
    index = packager.package(args.source, args.out, args.config, args.tokenizer, args.license)
    config = args.out / args.config
    manifest = json.loads((config / 'manifest.json').read_text())
    for path in config.rglob('*'):
        if path.is_file() and path.suffix != '.safetensors':
            validate_output_file(path, config, manifest)
    print(json.dumps(index, indent=2))
    print('Finished fidelity dataset; no upload performed.')


if __name__ == '__main__':
    main()

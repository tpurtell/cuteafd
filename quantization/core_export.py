"""Strict, CPU-streamed export of isolated dense trellis core variants."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct

from kimi_k3.uniform_k2.storage import (
    assemble_shard, atomic_json, read_header, sha256_file, verify_receipt,
)

SUFFIXES = ('trellis', 'suh', 'svh', 'mcg')
MCG = 0xCBAC1FED
REPO = Path(__file__).resolve().parents[1]


def packed_specs(module, geometry):
    inputs = geometry['in_features']
    outputs = geometry['stored_out_features']
    bits = geometry['bits']
    if (type(bits) is not int or not 1 <= bits <= 8 or inputs <= 0 or inputs % 128
            or outputs <= 0 or outputs % 128 or not 0 < geometry['out_features'] <= outputs):
        raise ValueError('invalid dense core geometry')
    return {module + '.trellis': ('I16', [inputs // 16, outputs // 16, bits * 16]),
            module + '.suh': ('F16', [inputs]), module + '.svh': ('F16', [outputs]),
            module + '.mcg': ('I32', [])}


def check_projection(path, module, geometry):
    header, base = read_header(path)
    expected = packed_specs(module, geometry)
    if {n: (v['dtype'], v['shape']) for n, v in header.items() if n != '__metadata__'} != expected:
        raise ValueError(f'core projection tensor inventory differs: {module}')
    with Path(path).open('rb') as stream:
        stream.seek(base + header[module + '.mcg']['data_offsets'][0])
        if struct.unpack('<I', stream.read(4))[0] != MCG:
            raise ValueError('core projection MCG multiplier differs')
    return expected


def assemble_core_shard(replacements, selection, export, filename, identity):
    if not selection or replacements.keys() != selection.keys():
        raise ValueError('core shard replacements must exactly cover selection')
    if Path(filename).name != filename or not filename.endswith('.safetensors'):
        raise ValueError('unsafe core shard filename')
    export = Path(export)
    export.mkdir(parents=True, exist_ok=True)
    entries = {}
    expected = {}
    for module, geometry in selection.items():
        specs = check_projection(replacements[module], module, geometry)
        expected.update(specs)
        entries.update({name: (Path(replacements[module]), name) for name in specs})
    path = export / filename
    receipt_path = export / (filename + '.json')
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get('identity') != identity or receipt.get('filename') != filename
                or receipt.get('modules') != sorted(selection)):
            raise ValueError('core shard recovery identity differs')
        verify_receipt(path, receipt, checksum=True)
    else:
        # An interrupted write without a receipt owns no durable export.
        if path.is_symlink():
            raise ValueError('refuse symlink core shard')
        path.unlink(missing_ok=True)
        receipt = dict(assemble_shard(entries, path), identity=identity,
                       filename=filename, modules=sorted(selection))
        atomic_json(receipt_path, receipt)
    header, _ = read_header(path)
    if {n: (v['dtype'], v['shape']) for n, v in header.items() if n != '__metadata__'} != expected:
        raise ValueError('assembled core shard inventory differs')
    return receipt


def finalize_core_export(selection, export, identity):
    export = Path(export)
    expected = {n: spec for module, geometry in selection.items()
                for n, spec in packed_specs(module, geometry).items()}
    mapping = {}
    payload = 0
    files = sorted(export.glob('*.safetensors'))
    if not files:
        raise ValueError('core export has no shards')
    for path in files:
        receipt = json.loads(path.with_name(path.name + '.json').read_text())
        if receipt.get('identity') != identity or receipt.get('filename') != path.name:
            raise ValueError('core export shard identity differs')
        verify_receipt(path, receipt, checksum=True)
        header, _ = read_header(path)
        shard_names = {n for m in receipt['modules'] for n in packed_specs(m, selection[m])}
        if set(header) - {'__metadata__'} != shard_names:
            raise ValueError('core shard receipt inventory differs')
        for name, value in header.items():
            if name == '__metadata__':
                continue
            if name in mapping or expected.get(name) != (value['dtype'], value['shape']):
                raise ValueError(f'extra, duplicate or invalid core tensor: {name}')
            mapping[name] = path.name
            payload += value['data_offsets'][1] - value['data_offsets'][0]
    if mapping.keys() != expected.keys():
        raise ValueError('incomplete core variant inventory')
    storage = {module: dict(quant_format='exl3', bits_per_weight=g['bits'],
                in_features=g['in_features'], out_features=g['out_features'],
                stored_out_features=g['stored_out_features'],
                tensors={suffix: module + '.' + suffix for suffix in SUFFIXES})
               for module, g in selection.items()}
    atomic_json(export / 'model.safetensors.index.json',
                dict(metadata=dict(total_size=payload), weight_map=mapping))
    atomic_json(export / 'quantize_config.json', dict(quant_method='exl3', tensor_storage=storage))
    summary = dict(modules=len(selection), tensors=len(mapping), payload_bytes=payload,
                   file_bytes=sum(p.stat().st_size for p in files), shards=len(files),
                   bits=sorted({g['bits'] for g in selection.values()}))
    atomic_json(export / 'core-export.json', dict(identity=identity, **summary))
    return summary


def sampled_packed_oracle(path, module, geometry, *, device='cuda:0'):
    """Compare real serialized tiles to an independent MSB-first CPU oracle."""
    import numpy as np
    import torch
    from muse_spark.core_k6.codec import _quantizer

    check_projection(path, module, geometry)
    header, base = read_header(path)
    entry = header[module + '.trellis']
    count = math.prod(entry['shape'][:2])
    indices = sorted({0, count - 1, *(i * (count - 1) // 7 for i in range(8))})
    # The pinned raw decoder launches groups of eight output tiles.
    chosen = (indices * 8)[:8]
    tile_bytes = geometry['bits'] * 32
    with Path(path).open('rb') as stream:
        chunks = []
        for index in chosen:
            stream.seek(base + entry['data_offsets'][0] + index * tile_bytes)
            raw = stream.read(tile_bytes)
            if len(raw) != tile_bytes:
                raise ValueError('truncated sampled trellis tile')
            chunks.append(raw)
    raw = b''.join(chunks)
    oracle_path = REPO / 'third_party/sparkinfer/tests/_reference/native_trellis.py'
    spec = importlib.util.spec_from_file_location('_core_trellis_reference', oracle_path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    expected = reference.Reference('mcg').decode_tiles(raw, geometry['bits']).astype(np.float16)
    packed = torch.from_numpy(np.frombuffer(raw, dtype='<i2').copy()).reshape(1, 8, -1).to(device)
    output = torch.empty((16, 128), dtype=torch.float16, device=device)
    _quantizer().ext.reconstruct(output, packed, geometry['bits'], True, False)
    actual = output.cpu().numpy().reshape(16, 8, 16).transpose(1, 0, 2).copy()
    if not np.array_equal(actual.view(np.uint16), expected.view(np.uint16)):
        raise ValueError(f'packed-byte reference decode differs: {module}')
    return dict(schema='cuteafd-core-packed-byte-oracle-v1', passed=True,
                bits=geometry['bits'], tiles=chosen, tile_count=len(chosen),
                sampled_sha256=hashlib.sha256(raw).hexdigest(),
                decoded_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                reference_sha256=sha256_file(oracle_path),
                comparison='pinned-GPTQModel-raw-F16-vs-independent-MSB-first-CPU-exact')

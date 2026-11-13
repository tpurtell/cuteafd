"""Tiny CPU fixtures for independent MC publication and MTP alias proofs."""
from pathlib import Path
import sys

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mc_package as package
from core_export import MCG, packed_specs
from inkling2.export import DirectMCWriter, VIEWS, load_source_inventory


def fixture(tmp_path):
    source, inputs, output = (tmp_path / name for name in ('source', 'inputs', 'output'))
    source.mkdir()
    inputs.mkdir()
    target, mtp = 'model.layers.0.self_attn.q_proj', 'mtp.layers.0.self_attn.q_proj'
    geometry = dict(in_features=128, out_features=128, stored_out_features=128)
    selection = {target: geometry, mtp: geometry}
    save_file({target + '.weight': torch.ones(128, 128, dtype=torch.bfloat16),
               'native.weight': torch.ones(3, dtype=torch.bfloat16)}, source / 'source.safetensors')
    save_file({'layers.0.self_attn.q_proj.weight': torch.full((128, 128), 2., dtype=torch.bfloat16)},
              source / 'mtp.safetensors')
    package.atomic_json(source / 'model.safetensors.index.json', dict(metadata=dict(total_size=999999),
        weight_map={target+'.weight': 'source.safetensors', 'native.weight': 'source.safetensors'}))
    inventory = load_source_inventory(source, reviewed_payload_bytes=65542, reviewed_tensor_count=3,
                                     extra_shards={'mtp.safetensors': 'mtp.'})
    encodes = {}
    for bits in (8, 6):
        tensors = {name: torch.ones(shape, dtype={'I16': torch.int16, 'F16': torch.float16,
                    'I32': torch.int32}[dtype]) for module, g in selection.items()
                   for name, (dtype, shape) in packed_specs(module, dict(g, bits=bits)).items()}
        for module in selection:
            tensors[module + '.mcg'] = torch.tensor(MCG, dtype=torch.uint32).view(torch.int32)
        path = inputs / f'k{bits}.safetensors'
        save_file(tensors, path)
        encodes[bits] = {name: (path, name) for name in tensors}
    # Stand-in expert tensors are shared across all views.
    expert = inputs / 'experts.safetensors'
    save_file({'expert.trellis': torch.ones(128, dtype=torch.int16)}, expert)
    shared = {'native.weight': (source / 'source.safetensors', 'native.weight'),
              'expert.trellis': (expert, 'expert.trellis')}
    original = {name: (source / row['shard'], row['source_name'])
                for name, row in inventory['locations'].items() if name != 'native.weight'}
    configs = {view: dict(model_type='inkling2', quantization_config={'quant_method': 'exl3'}) for view in VIEWS}
    quantizes = {}
    for view, bits in (('original', None), ('k8-compact', 8), ('k6-compact', 6)):
        storage = {'expert': dict(bits_per_weight=4, stored_tensors={'expert.trellis':
                    dict(shape=[128], torch_dtype='int16')})}
        if bits:
            for module, g in selection.items():
                storage[module] = dict(bits_per_weight=bits, stored_tensors={n: dict(
                    shape=shape, torch_dtype=package.DTYPES[dtype]) for n, (dtype, shape)
                    in packed_specs(module, dict(g, bits=bits)).items()})
        quantizes[view] = dict(tensor_storage=storage, retained_native_tensors={
            'native.weight': dict(dtype='BF16', shape=[3])})
    writer_args = dict(output=output, source_snapshot=source, source_repo='test/inkling',
        source_revision='source-sha', identity=dict(calibration='test'), source_inventory=inventory,
        shard_target=20000)
    entries = dict(shared=shared, original=original, k8=encodes[8], k6=encodes[6])
    final_args = dict(core_selection=selection, configs=configs, quantizes=quantizes)
    return writer_args, entries, final_args


def test_direct_views_mtp_byte_proof_and_retired_encode_resume(tmp_path, monkeypatch):
    args, entries, final = fixture(tmp_path)
    # New families do not depend on historical parent/family dispatch.
    monkeypatch.setattr(package, 'family_recipe', lambda _: pytest.fail('parent recipe queried'))
    writer = DirectMCWriter(**args)
    group = writer.write_group('layer-000', **entries)
    assert group['complete']
    assert max(sum(group['sizes'][r['kind']][n] for n in r['names']) for r in group['shards']) <= 32768
    for path in {p for kind in ('k8', 'k6') for p, _ in entries[kind].values()}:
        path.unlink()
    writer = DirectMCWriter(**args)
    assert writer.group_complete('layer-000')
    assert writer.write_group('layer-000', **entries) == group
    receipt = writer.finalize(**final)
    assert receipt['status'] == 'validated-headers-and-byte-exactness'
    assert len(receipt['original_core_tensor_sha256']) == 2
    state = package.load_json(args['output'] / 'package-state.json')
    assert state['source_tensor_locations']['mtp.layers.0.self_attn.q_proj.weight']['source_name'] == 'layers.0.self_attn.q_proj.weight'
    assert state['copies'] == {}
    maps = {v: package.inventory(args['output'], package.mc_manifest.FILES[v][0])['mapping'] for v in VIEWS}
    assert all(m['expert.trellis'] == maps['original']['expert.trellis'] for m in maps.values())
    assert 'mtp.layers.0.self_attn.q_proj.weight' in maps['original']
    assert 'mtp.layers.0.self_attn.q_proj.trellis' in maps['k8-compact']
    assert maps['k8-compact']['mtp.layers.0.self_attn.q_proj.trellis'] != maps['k6-compact']['mtp.layers.0.self_attn.q_proj.trellis']
    assert writer.finalize(**final)['views'] == receipt['views']


def test_nested_tokenizer_asset_preserved_without_escaping_output(tmp_path):
    args, entries, final = fixture(tmp_path)
    tokenizer = tmp_path/'tokenizer.model'
    tokenizer.write_bytes(b'native tokenizer asset')
    writer = DirectMCWriter(**args)
    writer.write_group('layer-000', **entries)
    writer.finalize(**final, assets={'tiktoken/tokenizer.model':tokenizer})
    assert (args['output']/'tiktoken/tokenizer.model').read_bytes() == tokenizer.read_bytes()
    with pytest.raises(ValueError, match='unsafe'):
        writer.finalize(**final, assets={'../escape':tokenizer})


@pytest.mark.parametrize('kind', ['count', 'bytes', 'physical', 'alias'])
def test_source_inventory_guards(tmp_path, kind):
    args, _, _ = fixture(tmp_path)
    source = args['source_snapshot']
    count, extent = 3, 65542
    if kind == 'count':
        count += 1
    elif kind == 'bytes':
        extent += 1
    elif kind == 'physical':
        document = package.load_json(source / 'model.safetensors.index.json')
        del document['weight_map']['native.weight']
        package.atomic_json(source / 'model.safetensors.index.json', document)
        extent -= 6
    elif kind == 'alias':
        save_file({'native.weight': torch.ones(3, dtype=torch.bfloat16)}, source / 'mtp.safetensors')
        extent = 32780
    with pytest.raises(ValueError, match='reviewed|inventory|duplicate'):
        load_source_inventory(source, reviewed_payload_bytes=extent, reviewed_tensor_count=count,
                              extra_shards={'mtp.safetensors': '' if kind == 'alias' else 'mtp.'})


@pytest.mark.parametrize('failure', ['width', 'missing', 'native', 'unknown', 'payload', 'metadata', 'marker'])
def test_sealing_guards(tmp_path, failure):
    args, entries, final = fixture(tmp_path)
    if failure == 'width':
        entries['k6'] = entries['k8']
    elif failure == 'missing':
        entries['k6'] = {}
    elif failure == 'native':
        entries['original'] = {name: (p, 'native.weight') for name, (p, _) in entries['original'].items()
                               if p.name == 'source.safetensors'}
    elif failure == 'metadata':
        final['quantizes']['k6-compact']['tensor_storage'][next(iter(final['core_selection']))]['bits_per_weight'] = 8
    elif failure == 'marker':
        p = next(iter(entries['k6'].values()))[0]
        from safetensors.torch import load_file
        tensors = load_file(p)
        tensors[next(n for n in tensors if n.endswith('.mcg'))] = torch.tensor(1, dtype=torch.int32)
        save_file(tensors, p)
    writer = DirectMCWriter(**args)
    if failure == 'native':
        with pytest.raises(ValueError, match='original tensor'):
            writer.write_group('layer-000', **entries)
        return
    writer.write_group('layer-000', **entries)
    if failure == 'unknown':
        save_file({'stray': torch.ones(1)}, args['output'] / 'unknown.safetensors')
    elif failure == 'payload':
        final['expected_payload_bytes'] = 1
    with pytest.raises(ValueError, match='inventory|physical|width|payload|MCG'):
        writer.finalize(**final)


def test_recovery_identity_input_and_collisions(tmp_path):
    args, entries, _ = fixture(tmp_path)
    writer = DirectMCWriter(**args)
    writer.write_group('layer-000', **entries)
    with pytest.raises(ValueError, match='recovery identity'):
        DirectMCWriter(**dict(args, identity=dict(calibration='changed')))
    with pytest.raises(ValueError, match='duplicate active'):
        writer.write_group('other', shared=entries['shared'])
    with pytest.raises(ValueError, match='recovery entries'):
        writer.write_group('layer-000', shared=entries['shared'])
    with pytest.raises(ValueError, match='label'):
        writer.write_group('../escape', **entries)


def test_interrupted_group_resumes_only_from_same_inputs(tmp_path, monkeypatch):
    args, entries, final = fixture(tmp_path)
    writer = DirectMCWriter(**args)
    real_write = package.write_shard
    calls = 0

    def fail_second(*a, **kw):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError('simulated interruption')
        return real_write(*a, **kw)

    monkeypatch.setattr(package, 'write_shard', fail_second)
    with pytest.raises(OSError, match='interruption'):
        writer.write_group('layer-000', **entries)
    assert not writer.group_complete('layer-000')
    monkeypatch.setattr(package, 'write_shard', real_write)
    writer = DirectMCWriter(**args)
    writer.write_group('layer-000', **entries)
    writer.finalize(**final)


def test_original_alias_payload_proof_catches_source_change(tmp_path):
    args, entries, final = fixture(tmp_path)
    writer = DirectMCWriter(**args)
    writer.write_group('layer-000', **entries)
    save_file({'layers.0.self_attn.q_proj.weight': torch.zeros(128, 128, dtype=torch.bfloat16)},
              args['source_snapshot'] / 'mtp.safetensors')
    with pytest.raises(ValueError, match='original source file changed: mtp'):
        writer.finalize(**final)


def test_default_direct_finalize_reuses_streaming_hashes_without_payload_rereads(tmp_path, monkeypatch):
    args, entries, final = fixture(tmp_path)
    expected = {n: package.tensor_hash(p, source_name) for n, (p, source_name) in entries['original'].items()}
    writer = DirectMCWriter(**args)
    writer.write_group('layer-000', **entries)
    real_sha = package.sha256_file

    def metadata_sha(path):
        assert not str(path).endswith('.safetensors'), 'whole-shard reread during direct validation'
        return real_sha(path)

    monkeypatch.setattr(package, 'sha256_file', metadata_sha)
    monkeypatch.setattr(package, 'tensor_hash', lambda *a: pytest.fail('native tensor reread during direct validation'))
    receipt = writer.finalize(**final)
    assert receipt['original_core_tensor_sha256'] == expected
    assert receipt['verification_mode'] == 'streaming-write-proofs-stat-headers-and-source-aliases'
    state = package.load_json(args['output'] / 'package-state.json')
    original_shard = state['direct_expected_mappings']['original']['mtp.layers.0.self_attn.q_proj.weight']
    proof = state['new_shards'][original_shard]['original_tensor_proofs']['mtp.layers.0.self_attn.q_proj.weight']
    assert proof['source_name'] == 'layers.0.self_attn.q_proj.weight'
    assert proof['sha256'] == expected['mtp.layers.0.self_attn.q_proj.weight']
    assert package.validate(args['output'], verification='streaming-receipts')['views'] == receipt['views']


@pytest.mark.parametrize('failure', ['proof', 'alias', 'inventory'])
def test_streaming_validation_requires_every_source_bound_copy_proof(tmp_path, failure):
    args, entries, final = fixture(tmp_path)
    writer = DirectMCWriter(**args)
    writer.write_group('layer-000', **entries)
    writer.finalize(**final)
    path = args['output'] / 'package-state.json'
    state = package.load_json(path)
    name = 'mtp.layers.0.self_attn.q_proj.weight'
    if failure == 'inventory':
        del state['source_tensor_locations'][name]
    else:
        shard = state['direct_expected_mappings']['original'][name]
        if failure == 'proof':
            del state['new_shards'][shard]['original_tensor_proofs'][name]
        else:
            state['new_shards'][shard]['original_tensor_proofs'][name]['source_name'] = 'unreviewed.weight'
    package.atomic_json(path, state)
    with pytest.raises(ValueError, match='original streaming'):
        package.validate(args['output'], verification='streaming-receipts')


def test_historical_full_validation_default_still_reads_payloads(tmp_path, monkeypatch):
    args, entries, final = fixture(tmp_path)
    writer = DirectMCWriter(**args)
    writer.write_group('layer-000', **entries)
    writer.finalize(**final)
    real_tensor_hash = package.tensor_hash
    real_file_sha = package.sha256_file
    tensor_calls, file_calls = [], []

    def tensor(path, name):
        tensor_calls.append((Path(path), name))
        return real_tensor_hash(path, name)

    def whole_file(path):
        file_calls.append(Path(path))
        return real_file_sha(path)

    monkeypatch.setattr(package, 'tensor_hash', tensor)
    monkeypatch.setattr(package, 'sha256_file', whole_file)
    package.validate(args['output'])
    assert any(p.parent == args['source_snapshot'] for p, _ in tensor_calls)
    assert any(p.suffix == '.safetensors' for p in file_calls)

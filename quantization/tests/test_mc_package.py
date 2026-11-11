"""Small physical-shard fixtures for MC and native-FP8-only publication trees."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mc_package as package
from core_export import packed_specs


def dump(root, name, data):
    (root / name).write_text(json.dumps(data))


def indexed(root, shards):
    mapping, size = {}, 0
    for filename, tensors in shards.items():
        save_file(tensors, root / filename)
        mapping.update({n: filename for n in tensors})
        size += sum(t.numel() * t.element_size() for t in tensors.values())
    dump(root, 'model.safetensors.index.json', dict(metadata=dict(total_size=size), weight_map=mapping))


def tensors(module, geometry, bits):
    return {n: torch.ones(shape, dtype={'I16': torch.int16, 'I32': torch.int32,
            'F16': torch.float16}[dtype])
            for n, (dtype, shape) in packed_specs(module, dict(geometry, bits=bits)).items()}


def fixture(tmp_path, monkeypatch, family='glmf'):
    parent, source, k8, output = [tmp_path / p for p in ('parent-sha', 'source-sha', 'k8', 'output')]
    for path in (parent, source, k8):
        path.mkdir()
    geometry = dict(in_features=128, out_features=128, stored_out_features=128, bits=8)
    selection = {'head': geometry}
    recipe = SimpleNamespace(BASE_REPO='test/source', BASE_REVISION='source-sha',
        source_names=lambda module, g: (module + '.weight',))
    monkeypatch.setattr(package, 'family_recipe', lambda family: (recipe, selection))
    monkeypatch.setitem(package.PARENTS, family, ('test-parent', 'parent-sha'))
    shared = {'expert.weight': torch.ones(128, 128, dtype=torch.bfloat16)}
    indexed(parent, {'mixed.safetensors': tensors('head', geometry, 6) | shared})
    indexed(source, {'source.safetensors': {'head.weight': torch.arange(128*128, dtype=torch.float32).reshape(128,128).bfloat16()}})
    if family == 'ml4':
        dump(source, 'consolidated.safetensors.index.json', package.load_json(source / 'model.safetensors.index.json'))
    indexed(k8, {'core.safetensors': tensors('head', geometry, 8)})
    producer = dict(source_repo=recipe.BASE_REPO, source_revision=recipe.BASE_REVISION)
    dump(k8, 'core-export.json', dict(identity=producer, modules=1, bits=[8]))
    from kimi_k3.uniform_k2.storage import file_identity, sha256_file
    dump(k8, 'core.safetensors.json', dict(identity=producer,
        file_identity=file_identity(k8/'core.safetensors'), sha256=sha256_file(k8/'core.safetensors')))
    storage = {'head': dict(bits_per_weight=6, stored_tensors={n: dict(shape=shape,
        torch_dtype=package.DTYPES[dtype]) for n, (dtype, shape) in packed_specs('head', dict(geometry,bits=6)).items()})}
    q = dict(quant_method='exl3', tensor_storage=storage, core_bits=6, head_bits=6,
        module_include=['^head$', '^expert$'], lm_head=True)
    dump(parent, 'quantize_config.json', q)
    dump(parent, 'config.json', dict(quantization_config=dict(quant_method='exl3', core_bits=6, head_bits=6)))
    return dict(family=family,parent=parent,source=source,k8=k8 if family != 'ml4' else None,output=output)


def test_mc_all_views_byte_exact_and_shared_parent_copy(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    result = package.assemble(**args)
    assert result['copied_files'] == 1
    assert set(result['views']) == {'original','k8-compact','k6-compact'}
    assert len(result['original_core_tensor_sha256']) == 1
    output = args['output']
    assert (output / 'mixed.safetensors').is_symlink()
    assert package.load_json(output/'quantize_config.json')['tensor_storage'] == {}
    assert package.load_json(output/'quantize_config.k8.json')['tensor_storage']['head']['bits_per_weight'] == 8
    assert package.assemble(**args) == result


def test_ml4_repack_has_no_inactive_physical_core(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch, 'ml4')
    result = package.assemble(**args)
    assert result['copied_files'] == 0
    assert not (args['output']/'cuteafd-mc.json').exists()
    assert not (args['output']/'mixed.safetensors').exists()
    assert set(package.inventory(args['output'])['mapping']) == {'expert.weight','head.weight'}
    assert 'head.trellis' not in package.read_header(args['output']/'native-parent-mixed.safetensors')[0]


def test_unexpected_parent_physical_tensor_rejected(tmp_path, monkeypatch):
    args = fixture(tmp_path,monkeypatch)
    index=package.load_json(args['parent']/'model.safetensors.index.json')
    index['weight_map'].pop('head.mcg');index['metadata']['total_size']-=4
    dump(args['parent'],'model.safetensors.index.json',index)
    with pytest.raises(ValueError,match='physical tensor inventory'):
        package.assemble(**args)


def test_restored_payload_tamper_rejected(tmp_path, monkeypatch):
    args=fixture(tmp_path,monkeypatch)
    package.assemble(**args)
    file=next(args['output'].glob('original-core-*.safetensors'))
    with file.open('r+b') as stream:
        stream.seek(-1,2);stream.write(b'X')
    with pytest.raises(ValueError,match='committed file'):
        package.validate(args['output'])


def test_wrong_source_revision_rejected_before_writes(tmp_path, monkeypatch):
    args=fixture(tmp_path,monkeypatch)
    args['source']=tmp_path/'wrong-revision'
    with pytest.raises(ValueError,match='exact reviewed revisions'):
        package.assemble(**args)
    assert not args['output'].exists()


def test_publication_packet_is_local_and_detects_changed_metadata(tmp_path, monkeypatch):
    from mc_publish import publication_inventory
    args=fixture(tmp_path,monkeypatch)
    package.assemble(**args)
    packet=publication_inventory(args['output'])
    assert packet['repo_id']=='wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K4-MC'
    assert packet['copies']=={'mixed.safetensors':'mixed.safetensors'}
    assert 'package-validation.json' in packet['new_files']
    (args['output']/'config.k8.json').write_text('{}')
    with pytest.raises(ValueError,match='changed'):
        publication_inventory(args['output'])


def test_nested_drafter_and_license_assets_are_copied(tmp_path, monkeypatch):
    args=fixture(tmp_path,monkeypatch)
    (args['parent']/'dflash').mkdir()
    save_file({'draft.weight':torch.ones(4)},args['parent']/'dflash/model.safetensors')
    (args['parent']/'LICENSE').write_text('source terms')
    package.assemble(**args)
    state=package.load_json(args['output']/'package-state.json')
    assert state['copies']['dflash/model.safetensors']=='dflash/model.safetensors'
    assert (args['output']/'LICENSE').read_text()=='source terms'


def test_output_overlap_refused(tmp_path, monkeypatch):
    args=fixture(tmp_path,monkeypatch)
    args['output']=args['parent']/'output'
    with pytest.raises(ValueError,match='disjoint'):
        package.assemble(**args)


def test_exact_pattern_excludes_only_requested_modules():
    patterns=package.exact_patterns(['^layers\\.[0-9]+\\.(?:core|expert)$'],
        {'layers.0.core','layers.1.core','layers.0.expert'}, {'layers.0.core'})
    import re
    assert re.search(patterns[0],'layers.1.core')
    assert re.search(patterns[0],'layers.0.expert')
    assert not re.search(patterns[0],'layers.0.core')

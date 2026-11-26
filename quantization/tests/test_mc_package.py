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


def test_tensor_storage_entry_matches_kimi_schema_and_preserves_optional_multiplier():
    from inkling2.run import tensor_storage_entry as inkling_entry

    geometry = dict(bits=6, in_features=256, out_features=192, stored_out_features=256)
    stored = {name: dict(shape=shape, torch_dtype=package.DTYPES[dtype])
              for name, (dtype, shape) in packed_specs('head', geometry).items()}
    kimi = dict(quant_format='exl3', bits_per_weight=6, mcg_multiplier=package.MCG,
                in_features=256, out_features=192, stored_out_features=256,
                stored_tensors=stored)
    assert inkling_entry is package.tensor_storage_entry
    assert inkling_entry(geometry, stored) == kimi
    assert set(kimi) == {'quant_format', 'bits_per_weight', 'mcg_multiplier',
                         'in_features', 'out_features', 'stored_out_features', 'stored_tensors'}
    assert inkling_entry(geometry, stored, mul1_multiplier=0.5) == dict(kimi, mul1_multiplier=0.5)


@pytest.mark.parametrize('view,filename', [('original', 'quantize_config.json'),
    ('k8-compact', 'quantize_config.k8.json'), ('k6-compact', 'quantize_config.k6.json')])
def test_inkling_view_declaration_is_mirrored_in_quantize_config(view, filename):
    from inkling2 import recipe
    from inkling2.run import quantize_config_for_view

    inline = dict(quant_method='exl3', format='mcg',
        per_module_metadata=filename + ':tensor_storage',
        activation_quantization=recipe.ACTIVATION_QUANTIZATION)
    result = quantize_config_for_view(inline, tensor_storage={}, core_view=view)
    assert all(result[key] == value for key, value in inline.items())
    result['activation_quantization']['block_size'] = 64
    assert inline['activation_quantization']['block_size'] == 32


@pytest.mark.parametrize('core_bits', [None, 8, 6])
def test_gptqmodel_mc_schema_preserves_provenance_and_declarations(core_bits):
    from inkling2 import recipe
    pointer = 'quantize_config.json:tensor_storage'
    previous = dict(quant_method='exl3', format='mcg', per_module_metadata=pointer,
                    activation_quantization=recipe.ACTIVATION_QUANTIZATION)
    metadata = dict(tensor_storage={}, retained_native_tensors={}, provenance={'producer': 'test'},
                    core_view='original' if core_bits is None else f'k{core_bits}-compact',
                    upstream_activations='k8-compact', **previous)
    quant, config = package.gptqmodel_mc_view(dict(quantization_config=previous), metadata,
        bits=4, core_bits=core_bits, module_include=[])
    for key in ('quant_method', 'method', 'format', 'checkpoint_format'):
        assert quant[key] == 'exl3'
    assert quant['codebook'] == 'mcg' and quant['group_size'] == -1 and quant['desc_act'] is False
    assert all(quant[k] == v for k, v in config['quantization_config'].items())
    assert quant['meta']['source_quantization_config'] == previous
    for key in ('core_view', 'provenance', 'upstream_activations', 'activation_quantization'):
        assert key not in quant and quant['meta'][key] == metadata[key]
    assert ('core_bits' in quant) == (core_bits is not None)
    assert metadata['format'] == 'mcg'


@pytest.mark.parametrize('family', ['kimi_k31', 'glm5', 'glm5_flash', 'muse_spark', 'mistral_large4'])
def test_shared_storage_builder_preserves_recorded_family_recipes(family):
    import importlib
    from kimi_k3.uniform_k2.storage import canonical

    leaf = 'k2_core_k6' if family == 'kimi_k31' else 'core_k6'
    recipe = importlib.import_module(f'{family}.{leaf}.recipe')
    for module, geometry in recipe.reviewed_modules().items():
        stored = {name: dict(shape=shape, torch_dtype=package.DTYPES[dtype])
                  for name, (dtype, shape) in recipe.packed_specs(module, geometry).items()}
        extra = dict(sharded_side=geometry.get('sharded_side', 1)) if family == 'muse_spark' else {}
        previous = dict(quant_format='exl3', bits_per_weight=geometry['bits'],
                        mcg_multiplier=recipe.MCG_MULTIPLIER,
                        **{key: geometry[key] for key in
                           ('in_features', 'out_features', 'stored_out_features')},
                        **extra, stored_tensors=stored)
        current = package.tensor_storage_entry(geometry, stored,
                    mcg_multiplier=recipe.MCG_MULTIPLIER, **extra)
        assert canonical(current) == canonical(previous)


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


def fixture(tmp_path, monkeypatch, family='glmf', calibration_key='samples', calibration_tokens=1234567):
    parent, source, k8, output = [tmp_path / p for p in ('parent-sha', 'source-sha', 'k8', 'output')]
    for path in (parent, source, k8):
        path.mkdir()
    geometry = dict(in_features=128, out_features=128, stored_out_features=128, bits=8)
    selection = {'head': geometry}
    recipe = SimpleNamespace(BASE_REPO='test/source', BASE_REVISION='source-sha',
        source_names=lambda module, g: (module + '.weight',),
        reviewed_core_modules=lambda **kwargs: selection)
    monkeypatch.setattr(package, 'family_recipe', lambda family: (recipe, selection))
    monkeypatch.setitem(package.PARENTS, family, ('test-parent', 'parent-sha'))
    shared = {'expert.weight': torch.ones(128, 128, dtype=torch.bfloat16)}
    indexed(parent, {'mixed.safetensors': tensors('head', geometry, 6) | shared})
    indexed(source, {'source.safetensors': {'head.weight': torch.arange(128*128, dtype=torch.float32).reshape(128,128).bfloat16()}})
    if family == 'ml4':
        dump(source, 'consolidated.safetensors.index.json', package.load_json(source / 'model.safetensors.index.json'))
    indexed(k8, {'core.safetensors': tensors('head', geometry, 8)})
    producer = dict(source_repo=recipe.BASE_REPO, source_revision=recipe.BASE_REVISION,
                    calibration={calibration_key: 1426, 'tokens': calibration_tokens})
    if family in ('step5', 'glm5'):
        producer.update(gptqmodel_commit='current-encoder',
            encoder_delta=dict(k6_gptqmodel_commit='historical-encoder', change='K8 validation enabled'),
            model_runtime=dict(transformers_version='model-version'), packages=dict(torch='torch-version'))
    (k8 / 'quantization-errors').mkdir()
    dump(k8 / 'quantization-errors', 'head.json', dict(module='head', logical_shape=[128,128],
        evidence=dict(bits=8, identity_rows=0, hessian_rows=calibration_tokens),
        metrics=dict(hessian_sample_count=calibration_tokens, quantizer_path='hessian_ldlq',
            hessian_numerical_contract='signed-block-hadamard-congruence-fp64-v1',
            hessian_weighted_relative_error=1e-6,
            reconstruction=dict(reference_finite=True, error_finite=True)),
        packed_byte_oracle=dict(schema='cuteafd-core-packed-byte-oracle-v1', bits=8,
                                passed=True, tile_count=8)))
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


@pytest.mark.parametrize('disagreement', [None, 'payload', 'inventory'])
def test_source_advisory_size_requires_reviewed_headers(tmp_path, monkeypatch, disagreement):
    args = fixture(tmp_path, monkeypatch)
    recipe, selection = package.family_recipe(args['family'])
    recipe.BASE_PAYLOAD_BYTES = 128 * 128 * 2 + (2 if disagreement == 'payload' else 0)

    def check_inventory(source):
        header, _ = package.read_header(source.snapshot / source.weight_map['head.weight'])
        expected = [128, 64] if disagreement == 'inventory' else [128, 128]
        if header['head.weight']['shape'] != expected:
            raise ValueError('source recipe inventory differs')

    recipe.inventory = check_inventory
    dump(args['source'], 'config.json', dict(text_config={}))
    index = package.load_json(args['source'] / 'model.safetensors.index.json')
    index['metadata']['total_size'] += 1234
    dump(args['source'], 'model.safetensors.index.json', index)
    if disagreement:
        with pytest.raises(ValueError, match='reviewed recipe|recipe inventory'):
            package.assemble(**args)
        assert not args['output'].exists()
        return
    package.assemble(**args)
    provenance = package.load_json(args['output'] / 'package-provenance.json')
    assert provenance['source_payload_bytes'] == dict(declared=34002, measured=32768, reviewed=32768)
    assert package.inventory(args['output'])['index']['metadata']['total_size'] == 65536


@pytest.mark.parametrize('kind', ['parent', 'k8', 'output'])
def test_non_source_index_totals_stay_strict(tmp_path, monkeypatch, kind):
    args = fixture(tmp_path, monkeypatch)
    if kind == 'output':
        package.assemble(**args)
    root = args[kind]
    index = package.load_json(root / 'model.safetensors.index.json')
    index['metadata']['total_size'] += 1
    dump(root, 'model.safetensors.index.json', index)
    with pytest.raises(ValueError, match='total_size differs'):
        package.inventory(root)


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


@pytest.mark.parametrize('bits', [None, 6, 8])
def test_view_metadata_rewrites_nested_quantization_configs(bits):
    from copy import deepcopy

    inline = dict(quant_method='exl3', format='mcg', core_bits=6, head_bits=6, mtp_bits=4,
                  activation_quantization=dict(block_size=32, format='mxfp8_e4m3',
                      rounding='nearest', scale_dtype='ue8m0'),
                  per_module_metadata='quantize_config.json:tensor_storage')
    config = dict(quantization_config=deepcopy(inline),
                  text_config=dict(quantization_config=deepcopy(inline)),
                  components=[dict(text_config=dict(quantization_config=deepcopy(inline)))])
    before = deepcopy(config)
    filename = 'quantize_config.json' if bits is None else f'quantize_config.k{bits}.json'
    quant, rewritten = package.view_metadata(dict(tensor_storage={}, module_include=[], mtp_bits=4),
        config, {}, {}, bits=bits, quantize_file=filename)
    blocks = [rewritten['quantization_config'], rewritten['text_config']['quantization_config'],
              rewritten['components'][0]['text_config']['quantization_config']]
    for block in blocks:
        assert all(quant[key] == value for key, value in block.items())
        assert block['quant_method'] == 'exl3'
        assert block['per_module_metadata'] == quant['per_module_metadata'] == filename + ':tensor_storage'
        for field in ('core_bits', 'head_bits'):
            assert (field not in block) if bits is None else block[field] == bits
        assert ('mtp_bits' not in block) if bits is None else block['mtp_bits'] == 4
    assert ('mtp_bits' not in quant) if bits is None else quant['mtp_bits'] == 4
    assert config == before


def test_mc_assembly_rewrites_multimodal_text_metadata(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    config = package.load_json(args['parent'] / 'config.json')
    config['text_config'] = dict(quantization_config=dict(config['quantization_config'],
        per_module_metadata='quantize_config.json:tensor_storage'))
    dump(args['parent'], 'config.json', config)
    package.assemble(**args)
    for view, (_, filename, quantize, bits) in package.mc_manifest.FILES.items():
        published = package.load_json(args['output'] / filename)
        for block in (published['quantization_config'], published['text_config']['quantization_config']):
            assert block['per_module_metadata'] == quantize + ':tensor_storage'
            for field in ('core_bits', 'head_bits'):
                assert (field not in block) if view == 'original' else block[field] == bits


@pytest.mark.parametrize('family', ['muse', 'kimi-k31'])
def test_restored_native_metadata_uses_safetensors_dtype(tmp_path, monkeypatch, family):
    args = fixture(tmp_path, monkeypatch, family=family)
    quant = package.load_json(args['parent'] / 'quantize_config.json')
    retained = {'expert.weight': dict(dtype='BF16', shape=[128, 128])}
    kept = {'expert.weight': dict(torch_dtype='bfloat16', shape=[128, 128])}
    quant.update(retained_native_tensors=retained, kept_tensors=kept)
    dump(args['parent'], 'quantize_config.json', quant)
    if family == 'kimi-k31':
        revision = 'a' * 40
        parent = args['parent'].with_name(revision)
        args['parent'].rename(parent)
        args.update(parent=parent, parent_revision=revision)
        monkeypatch.delitem(package.PARENTS, family)
    package.assemble(**args)
    original = package.load_json(args['output'] / 'quantize_config.json')
    assert original['retained_native_tensors'] == retained | {
        'head.weight': dict(dtype='BF16', shape=[128, 128])}
    assert original['kept_tensors'] == kept | {
        'head.weight': dict(torch_dtype='bfloat16', shape=[128, 128])}
    for bits in (6, 8):
        compact = package.load_json(args['output'] / f'quantize_config.k{bits}.json')
        assert compact['retained_native_tensors'] == retained
        assert compact['kept_tensors'] == kept


def test_validate_rejects_retained_native_schema_and_dtype(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    package.assemble(**args)
    path = args['output'] / 'quantize_config.json'
    quant = package.load_json(path)
    quant['retained_native_tensors'] = {'head.weight': dict(torch_dtype='bfloat16', shape=[128, 128])}
    dump(args['output'], path.name, quant)
    state = package.load_json(args['output'] / 'package-state.json')
    manifest = package.mc_manifest.build(args['output'], source_repo='test/source',
        source_revision='source-sha', core_tensors=state['core_tensors'])
    dump(args['output'], package.mc_manifest.MANIFEST, manifest)
    with pytest.raises(ValueError, match='retained native metadata'):
        package.validate(args['output'])
    quant['retained_native_tensors']['head.weight'] = dict(dtype='F32', shape=[128, 128])
    dump(args['output'], path.name, quant)
    state = package.load_json(args['output'] / 'package-state.json')
    manifest = package.mc_manifest.build(args['output'], source_repo='test/source',
        source_revision='source-sha', core_tensors=state['core_tensors'])
    dump(args['output'], package.mc_manifest.MANIFEST, manifest)
    with pytest.raises(ValueError, match='retained native metadata'):
        package.validate(args['output'])


@pytest.mark.parametrize('key,tokens', [('samples', 1234567), ('prompts', 1201086)])
def test_k8_native_calibration_key_preserved(tmp_path, monkeypatch, key, tokens):
    family = 'kimi-k31' if key == 'prompts' else 'glmf'
    args = fixture(tmp_path, monkeypatch, family=family, calibration_key=key, calibration_tokens=tokens)
    if family == 'kimi-k31':
        revision = 'a' * 40
        parent = args['parent'].with_name(revision)
        args['parent'].rename(parent)
        args.update(parent=parent, parent_revision=revision)
        monkeypatch.delitem(package.PARENTS, family)
    original = (args['k8'] / 'core-export.json').read_bytes()
    result = package.assemble(**args)
    assert result['k8_calibration_count_key'] == key
    assert package.load_json(args['output'] / 'package-state.json')['k8_calibration_count_key'] == key
    assert package.load_json(args['output'] / 'quantization-provenance.k8.json')['calibration'] == {
        key: 1426, 'tokens': tokens}
    assert (args['k8'] / 'core-export.json').read_bytes() == original


@pytest.mark.parametrize('counts', [{}, {'samples': 1426, 'prompts': 1426}])
def test_k8_calibration_ambiguous_or_missing_refused(tmp_path, monkeypatch, counts):
    args = fixture(tmp_path, monkeypatch)
    identity = package.load_json(args['k8'] / 'core-export.json')['identity']
    identity['calibration'] = dict(counts, tokens=1234567)
    with pytest.raises(ValueError, match='exactly one of samples or prompts'):
        package.k8_diagnostics(args['k8'] / 'quantization-errors', {'head': {}}, identity)


def test_k8_diagnostics_require_full_rows_and_publish_reports(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    package.assemble(**args)
    assert (args['output'] / 'quantization-errors.k8/head.json').exists()
    assert (args['output'] / 'quantization-provenance.k8.json').exists()
    path = args['k8'] / 'quantization-errors/head.json'
    row = package.load_json(path)
    row['evidence']['hessian_rows'] -= 1
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='full-corpus'):
        package.assemble(**args)


def test_ml4_repack_has_no_inactive_physical_core(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch, 'ml4')
    result = package.assemble(**args)
    assert result['copied_files'] == 0
    assert not (args['output']/'cuteafd-mc.json').exists()
    assert not (args['output']/'mixed.safetensors').exists()
    assert set(package.inventory(args['output'])['mapping']) == {'expert.weight','head.weight'}
    assert 'head.trellis' not in package.read_header(args['output']/'native-parent-mixed.safetensors')[0]


def test_ml4_independent_audit_proves_experts_and_rejects_tamper(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch, 'ml4')
    package.assemble(**args)
    report = package.audit_ml4(args['output'])
    assert report['inherited_tensors'] == 1 and report['core_tensors'] == 1
    shard = args['output'] / 'native-parent-mixed.safetensors'
    with shard.open('r+b') as stream:
        stream.seek(-1, 2)
        stream.write(b'X')
    with pytest.raises(ValueError, match='inherited tensor bytes differ'):
        package.audit_ml4(args['output'])


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


@pytest.mark.parametrize('family,repo', [
    ('step5', 'wrldsuksgo2mars/Step-5-Preview-EXL3-K4-MC'),
    ('glm5', 'wrldsuksgo2mars/GLM-5.3-EXL3-K4-MC'),
])
def test_seed_publication_packets_copy_whole_parent_shards(tmp_path, monkeypatch, family, repo):
    from mc_publish import publication_inventory
    args = fixture(tmp_path, monkeypatch, family=family)
    package.assemble(**args)
    packet = publication_inventory(args['output'])
    assert packet['repo_id'] == repo
    assert packet['copies'] == {'mixed.safetensors': 'mixed.safetensors'}
    uploaded_shards = {n for n in packet['new_files'] if n.endswith('.safetensors')}
    assert uploaded_shards == {'original-core-0001.safetensors', 'k8-core.safetensors'}
    assert 'mixed.safetensors' not in packet['new_files']
    for view in ('original', 'k8-compact'):
        index = package.mc_manifest.FILES[view][0]
        mapping = package.load_json(args['output'] / index)['weight_map']
        assert mapping['expert.weight'] == 'mixed.safetensors'
        assert not any(mapping.get('head.' + suffix) == 'mixed.safetensors'
                       for suffix in package.SUFFIXES)
    k6_index = package.mc_manifest.FILES['k6-compact'][0]
    mapping = package.load_json(args['output'] / k6_index)['weight_map']
    assert mapping['head.trellis'] == 'mixed.safetensors'


def test_nested_drafter_and_license_assets_are_copied(tmp_path, monkeypatch):
    args=fixture(tmp_path,monkeypatch)
    (args['parent']/'dflash').mkdir()
    save_file({'draft.weight':torch.ones(4)},args['parent']/'dflash/model.safetensors')
    (args['parent']/'LICENSE').write_text('source terms')
    package.assemble(**args)
    state=package.load_json(args['output']/'package-state.json')
    assert state['copies']['dflash/model.safetensors']=='dflash/model.safetensors'
    assert (args['output']/'LICENSE').read_text()=='source terms'


def test_parent_license_front_matter_is_preserved(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    front = '---\nlicense: other\nlicense_name: source-license\nbase_model: test/source\n---\n'
    (args['parent'] / 'README.md').write_text(front + '# Historical parent\n')
    package.assemble(**args)
    assert (args['output'] / 'README.md').read_text().startswith(front)
    assert (args['output'] / 'README.parent.md').read_text() == front + '# Historical parent\n'


def test_output_overlap_refused(tmp_path, monkeypatch):
    args=fixture(tmp_path,monkeypatch)
    args['output']=args['parent']/'output'
    with pytest.raises(ValueError,match='disjoint'):
        package.assemble(**args)


class FakePublishApi:
    """Hub fake for parent-guarded HTTP LFS batches with server-side copies."""
    token = False

    def __init__(self, packet):
        self.packet = packet
        self.head = 'a' * 40
        self.files = {}
        self.trees = {}
        self.uploaded = []
        self.batches = []
        self.workers = []
        self.commits = 0
        self.fail_after = None
        self.lose_reply = False
        self.parent = {source: dict(bytes=packet['copy_files'][target]['bytes'],
            blob=packet['copy_files'][target]['sha256']) for target, source in packet['copies'].items()}

    def repo_info(self, repo, **kwargs):
        return SimpleNamespace(sha=self.head, private=False)

    def model_info(self, repo, revision):
        assert revision == self.head
        return SimpleNamespace(sha=self.head)

    def tree(self, client, repo, revision):
        return self.parent if repo == self.packet['parent_repo'] else self.trees.get(revision, {})

    def create_commit(self, repo, operations, **kwargs):
        from huggingface_hub import CommitOperationCopy
        assert kwargs['parent_commit'] == self.head
        files = dict(self.files)
        names = []
        for operation in operations:
            if isinstance(operation, CommitOperationCopy):
                files[operation.path_in_repo] = self.parent[operation.src_path_in_repo]
                continue
            name = operation.path_in_repo
            assert name not in self.files, 'committed file sent again'
            planned = self.packet['new_files'][name]
            # Receipt SHA256 supplied; no payload hashing in the SDK.
            assert operation.upload_info.is_hashed
            assert operation.upload_info.sha256.hex() == planned['sha256']
            if self.fail_after is not None and len(self.uploaded) >= self.fail_after:
                raise RuntimeError('transport interrupted')
            self.uploaded.append(name)
            names.append(name)
            files[name] = dict(bytes=operation.upload_info.size, blob=planned['sha256'])
        self.batches.append(names)
        self.workers.append(kwargs['num_threads'])
        self.files = files
        self.commits += 1
        self.head = f'{self.commits + 10:040x}'
        self.trees[self.head] = dict(files)
        if self.lose_reply:
            self.lose_reply = False
            raise RuntimeError('lost commit reply')
        return SimpleNamespace(oid=self.head)


def publisher_fixture(tmp_path, monkeypatch):
    import mc_publish
    import hub_lfs
    import huggingface_hub
    args = fixture(tmp_path, monkeypatch)
    package.assemble(**args)
    packet = mc_publish.publication_inventory(args['output'])
    api = FakePublishApi(packet)
    monkeypatch.setattr(huggingface_hub, 'HfApi', lambda **kwargs: api)
    monkeypatch.setattr(hub_lfs, 'remote_tree', api.tree)
    return mc_publish, args['output'], tmp_path / 'upload-state', packet, api


def test_publish_resumes_batches_and_lost_commit_reply_without_resending(tmp_path, monkeypatch):
    publisher, output, state, packet, api = publisher_fixture(tmp_path, monkeypatch)
    options = publisher.hub_lfs.UploadOptions(workers=5, batch_files=2)
    api.fail_after = 3  # Batch 1 (2 files) commits; batch 2 fails after its first file.
    with pytest.raises(RuntimeError, match='transport interrupted'):
        publisher.publish(output, state, packet, api=api, options=options)
    committed = set(api.files) - set(api.packet['copies'])
    assert api.commits == 1 and len(committed) == 2 and len(api.uploaded) == 3
    assert set(api.packet['copies']) <= set(api.files)  # Copies ride the first batch.
    api.fail_after = None
    api.lose_reply = True
    with pytest.raises(RuntimeError, match='lost commit reply'):
        publisher.publish(output, state, packet, api=api, options=options)
    result = publisher.publish(output, state, packet, api=api, options=options)
    assert result['status'] == 'uploaded'
    sent = [name for batch in api.batches for name in batch]  # Committed transfers.
    assert len(sent) == len(set(sent)) == len(packet['new_files'])
    assert set(api.workers) == {5} and all(len(batch) <= 2 for batch in api.batches)
    assert not committed & {name for batch in api.batches[1:] for name in batch}
    assert set(result['files']) == set(packet['new_files']) | set(packet['copies'])


def test_publish_defaults_to_16_workers_and_48_file_batches(tmp_path, monkeypatch):
    publisher, output, state, packet, api = publisher_fixture(tmp_path, monkeypatch)
    publisher.publish(output, state, packet, api=api)
    assert set(api.workers) == {16}
    assert [len(batch) for batch in api.batches] == [min(48, len(packet['new_files']))]


def test_publish_reuses_writer_shard_sha256_without_rereading_weights(tmp_path, monkeypatch):
    publisher, output, state, packet, api = publisher_fixture(tmp_path, monkeypatch)
    shards = {n for n in packet['new_files'] if n.endswith('.safetensors')}
    assert shards
    real_open = Path.open

    def no_weight_reads(path, *args, **kwargs):
        if path.name in shards:
            pytest.fail('publication reread a shard with a writer SHA256 receipt')
        return real_open(path, *args, **kwargs)

    real_read_bytes = Path.read_bytes

    def no_weight_read_bytes(path):
        if path.name in shards:
            pytest.fail('publication reread a shard with a writer SHA256 receipt')
        return real_read_bytes(path)

    monkeypatch.setattr(Path, 'open', no_weight_reads)
    monkeypatch.setattr(Path, 'read_bytes', no_weight_read_bytes)
    assert publisher.publication_inventory(output) == packet
    publisher.publish(output, state, packet, api=api)
    assert set(api.uploaded) == set(packet['new_files'])


def test_publish_refuses_receipt_mismatch_without_hub_action(tmp_path, monkeypatch):
    publisher, output, state, packet, api = publisher_fixture(tmp_path, monkeypatch)
    shard = next(n for n in packet['new_files'] if n.endswith('.safetensors'))
    receipt = package.load_json(output / (shard + '.receipt.json'))
    state_doc = package.load_json(output / 'package-state.json')
    state_doc['new_shards'][shard]['sha256'] = '0' * 64
    package.atomic_json(output / 'package-state.json', state_doc)
    with pytest.raises(ValueError, match='validated new file changed'):
        publisher.publish(output, state, packet, api=api)
    assert receipt['sha256'] != '0' * 64 and not api.uploaded and api.commits == 0


def test_publish_refuses_source_changed_mid_batch(tmp_path, monkeypatch):
    publisher, output, state, packet, api = publisher_fixture(tmp_path, monkeypatch)
    import hub_lfs
    options = hub_lfs.UploadOptions(batch_files=2)
    victim = hub_lfs.upload_order(packet['new_files'])[1]
    create = api.create_commit

    def mutate_during_transfer(repo, operations, **kwargs):
        result = create(repo, operations, **kwargs)
        if api.commits == 1:
            with (output / victim).open('ab') as stream:
                stream.write(b'changed during transfer')
        return result

    api.create_commit = mutate_during_transfer
    with pytest.raises(ValueError, match='source changed'):
        publisher.publish(output, state, packet, api=api, options=options)
    assert api.commits == 1 and not (state / 'upload-complete.json').exists()
    with pytest.raises(ValueError, match='changed|separately approved'):
        publisher.publish(output, state, packet, api=api, options=options)
    assert api.commits == 1


def test_publish_rejects_approval_mismatch_without_hub_action(tmp_path, monkeypatch):
    publisher, output, state, packet, api = publisher_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='separately approved'):
        publisher.publish(output, state, dict(packet, new_bytes=-1), api=api)
    assert not state.exists()
    assert not api.uploaded and api.commits == 0


def test_publish_rejects_wrong_pinned_parent_bytes(tmp_path, monkeypatch):
    publisher, output, state, packet, api = publisher_fixture(tmp_path, monkeypatch)
    api.parent['mixed.safetensors']['blob'] = 'f' * 64
    with pytest.raises(ValueError, match='pinned parent'):
        publisher.publish(output, state, packet, api=api)
    assert not api.uploaded
    assert not (state / 'destination.json').exists()


def test_ml4_upload_inventory_rejects_hidden_quantized_core_asset(tmp_path, monkeypatch):
    from mc_publish import publication_inventory
    args = fixture(tmp_path, monkeypatch, 'ml4')
    save_file(tensors('head', dict(in_features=128, out_features=128,
        stored_out_features=128), 6), args['parent'] / 'unindexed.safetensors')
    package.assemble(**args)
    with pytest.raises(ValueError, match='physically contains quantized core'):
        publication_inventory(args['output'])


def test_exact_pattern_excludes_only_requested_modules():
    patterns=package.exact_patterns(['^layers\\.[0-9]+\\.(?:core|expert)$'],
        {'layers.0.core','layers.1.core','layers.0.expert'}, {'layers.0.core'})
    import re
    assert re.search(patterns[0],'layers.1.core')
    assert re.search(patterns[0],'layers.0.expert')
    assert not re.search(patterns[0],'layers.0.core')


@pytest.mark.parametrize('family,count,mtp_layer', [('step5',645,92), ('glm5',547,78)])
def test_seed_families_register_target_core_only(family, count, mtp_layer):
    recipe, selection = package.family_recipe(family)
    assert len(selection) == count
    assert selection == recipe.reviewed_core_modules(include_mtp=False)
    assert all(g['bits'] == 6 for g in selection.values())
    assert all('.experts.' not in name for name in selection)
    assert all(not name.startswith(f'model.layers.{layer}.')
               for name in selection for layer in range(mtp_layer, mtp_layer+3))
    assert len(package.PARENTS[family][1]) == 40


@pytest.mark.parametrize('family', ['step5', 'glm5', 'glmf'])
def test_mc_restores_original_mtp_without_new_compact_encodes(tmp_path, monkeypatch, family):
    args = fixture(tmp_path, monkeypatch, family=family)
    geometry = dict(in_features=128,out_features=128,stored_out_features=128,bits=4)
    module = 'model.layers.999.mtp'
    packed_mtp = tensors(module, geometry, 4)
    recipe, selection = package.family_recipe(family)
    recipe.reviewed_core_modules = lambda **kwargs: selection | {module: geometry}
    original_mtp = {module + '.weight': torch.arange(128*128, dtype=torch.float32).reshape(128,128).bfloat16()}
    indexed(args['source'], {'source.safetensors': original_mtp | {'head.weight': original_mtp[module + '.weight'].clone()}})
    parent = args['parent']
    indexed(parent, {'mixed.safetensors': tensors('head',geometry,6) | packed_mtp})
    quant = package.load_json(parent/'quantize_config.json')
    quant['tensor_storage'][module] = dict(bits_per_weight=4,
        stored_tensors={name: dict(shape=shape, torch_dtype=package.DTYPES[dtype])
                       for name, (dtype, shape) in packed_specs(module, geometry).items()})
    quant['module_include'].append('^model\\.layers\\.999\\.mtp$')
    dump(parent,'quantize_config.json',quant)
    result = package.assemble(**args)
    assert len(result['original_core_tensor_sha256']) == 2
    state = package.load_json(args['output'] / 'package-state.json')
    for view in ('original','k8-compact','k6-compact'):
        index_name = package.mc_manifest.FILES[view][0]
        mapping = package.load_json(args['output']/index_name)['weight_map']
        q = package.load_json(args['output'] / package.mc_manifest.FILES[view][2])
        if view == 'original':
            assert not packed_mtp.keys() & mapping.keys()
            assert module + '.weight' in mapping
            assert module not in q['tensor_storage']
            assert module + '.weight' in state['core_tensors'][view]
            import re
            assert not any(re.search(p, module) for p in q['module_include'])
        else:
            assert all(mapping[name]=='mixed.safetensors' for name in packed_mtp)
            assert module + '.weight' not in mapping
            assert q['tensor_storage'][module]['bits_per_weight'] == 4
            assert packed_mtp.keys() <= set(state['core_tensors'][view])
    manifest = package.load_json(args['output'] / 'cuteafd-mc.json')
    package.mc_manifest.validate(manifest, args['output'])
    card = (args['output']/'README.md').read_text()
    assert '1 MTP matrices' in card
    assert 'Compact views preserve the existing K4 MTP' in card


@pytest.mark.parametrize('family,count', [('glmf',8), ('glm5',8), ('step5',27)])
def test_original_selection_includes_all_mtp_but_compact_selection_stays_frozen(family, count):
    recipe, compact = package.family_recipe(family)
    original = package.original_core_selection(family, recipe, compact)
    mtp = original.keys() - compact.keys()
    assert len(mtp) == count
    assert all(original[m]['group'] == 'mtp_core' and original[m]['bits'] == 4 for m in mtp)
    assert all('.experts.' not in m for m in original)


@pytest.mark.parametrize('family', ['step5', 'glm5'])
def test_seed_card_discloses_encoder_delta_and_actual_runtime(tmp_path, monkeypatch, family):
    args = fixture(tmp_path, monkeypatch, family=family)
    package.assemble(**args)
    card = (args['output'] / 'README.md').read_text()
    assert 'current-encoder' in card and 'historical-encoder' in card
    assert 'K8 validation enabled' in card
    assert 'model-version' in card and 'torch-version' in card
    assert 'quantization-provenance.k8.json' in card


@pytest.mark.parametrize('family', ['step5', 'glm5'])
def test_seed_package_rejects_missing_encoder_delta(tmp_path, monkeypatch, family):
    args = fixture(tmp_path, monkeypatch, family=family)
    receipt = package.load_json(args['k8'] / 'core-export.json')
    receipt['identity'].pop('encoder_delta')
    dump(args['k8'], 'core-export.json', receipt)
    with pytest.raises(ValueError, match='encoder delta'):
        package.assemble(**args)


def test_publish_refuses_moved_head_without_first_batch(tmp_path, monkeypatch):
    publisher, output, state, packet, api = publisher_fixture(tmp_path, monkeypatch)
    publisher.publish(output, state, packet, api=api)  # Writes the destination guard.
    # A foreign writer moves an empty destination: only Hub default attributes, no copies.
    api.head, api.files = 'c' * 40, {'.gitattributes': dict(bytes=1, blob='e' * 40)}
    api.trees[api.head] = dict(api.files)
    (state / 'upload-complete.json').unlink()
    (state / 'destination.json').write_text('{"commit": "' + 'd' * 40 + '"}')
    commits = api.commits
    with pytest.raises(ValueError, match='before the first verified batch'):
        publisher.publish(output, state, packet, api=api)
    assert api.commits == commits

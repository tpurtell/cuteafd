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
    producer = dict(source_repo=recipe.BASE_REPO, source_revision=recipe.BASE_REVISION,
                    calibration=dict(samples=1426, tokens=1234567))
    (k8 / 'quantization-errors').mkdir()
    dump(k8 / 'quantization-errors', 'head.json', dict(module='head', logical_shape=[128,128],
        evidence=dict(bits=8, identity_rows=0, hessian_rows=1234567),
        metrics=dict(hessian_sample_count=1234567, quantizer_path='hessian_ldlq',
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
    def __init__(self, packet):
        self.packet = packet
        self.head = 'a' * 40
        self.files = {}
        self.uploaded = []
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

    def preupload_lfs_files(self, repo, operations, **kwargs):
        import hashlib
        for operation in operations:
            if self.fail_after is not None and len(self.uploaded) >= self.fail_after:
                raise RuntimeError('transport interrupted')
            operation._should_ignore = False
            operation._upload_mode = 'lfs'
            with operation.as_file() as stream:
                operation.upload_info.sha256 = hashlib.sha256(stream.read()).digest()
            operation._is_uploaded = True
            self.uploaded.append(operation.path_in_repo)

    def create_commit(self, repo, operations, **kwargs):
        from huggingface_hub import CommitOperationCopy
        assert kwargs['parent_commit'] == self.head
        for operation in operations:
            if isinstance(operation, CommitOperationCopy):
                self.files[operation.path_in_repo] = self.parent[operation.src_path_in_repo]
            else:
                self.files[operation.path_in_repo] = dict(bytes=operation.upload_info.size,
                    blob=operation.upload_info.sha256.hex())
        self.commits += 1
        self.head = 'b' * 40
        if self.lose_reply:
            raise RuntimeError('lost commit reply')
        return SimpleNamespace(oid=self.head)


def publisher_fixture(tmp_path, monkeypatch):
    import mc_publish
    import upload_model
    import huggingface_hub
    from huggingface_hub.utils import _runtime
    args = fixture(tmp_path, monkeypatch)
    package.assemble(**args)
    packet = mc_publish.publication_inventory(args['output'])
    api = FakePublishApi(packet)
    monkeypatch.setattr(mc_publish.importlib.metadata, 'version', lambda name: '1.26.1')
    monkeypatch.setattr(_runtime, 'is_xet_available', lambda: True)
    monkeypatch.setattr(huggingface_hub, 'HfApi', lambda **kwargs: api)
    monkeypatch.setattr(upload_model, '_remote', lambda client, repo, rev:
        api.parent if repo == packet['parent_repo'] else api.files)
    return mc_publish, args['output'], tmp_path / 'upload-state', packet, api


def test_publish_recovers_preupload_and_lost_commit_reply(tmp_path, monkeypatch):
    publisher, output, state, packet, api = publisher_fixture(tmp_path, monkeypatch)
    api.fail_after = 2
    with pytest.raises(RuntimeError, match='transport interrupted'):
        publisher.publish(output, state, packet, api=api)
    assert api.commits == 0
    api.fail_after = None
    api.lose_reply = True
    with pytest.raises(RuntimeError, match='lost commit reply'):
        publisher.publish(output, state, packet, api=api)
    prepared = list(api.uploaded)
    result = publisher.publish(output, state, packet, api=api)
    assert result['status'] == 'uploaded'
    assert api.uploaded == prepared
    assert len(prepared) == len(set(prepared)) == len(packet['new_files'])
    assert api.commits == 1


def test_publish_rejects_approval_mismatch_without_hub_action(tmp_path, monkeypatch):
    publisher, output, state, packet, api = publisher_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='separately approved'):
        publisher.publish(output, state, dict(packet, new_bytes=-1), api=api)
    assert not state.exists()
    assert not api.uploaded


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


@pytest.mark.parametrize('family,mtp_count', [('step5',27), ('glm5',8)])
def test_seed_mc_preserves_shared_k4_mtp_across_views(tmp_path, monkeypatch, family, mtp_count):
    args = fixture(tmp_path, monkeypatch, family=family)
    geometry = dict(in_features=128,out_features=128,stored_out_features=128,bits=4)
    shared_mtp = tensors('model.layers.999.mtp', geometry, 4)
    parent = args['parent']
    indexed(parent, {'mixed.safetensors': tensors('head',geometry,6) | shared_mtp})
    quant = package.load_json(parent/'quantize_config.json')
    quant['tensor_storage']['model.layers.999.mtp'] = dict(bits_per_weight=4,
        stored_tensors={name: dict(shape=shape, torch_dtype=package.DTYPES[dtype])
                       for name, (dtype, shape) in packed_specs('model.layers.999.mtp', geometry).items()})
    dump(parent,'quantize_config.json',quant)
    package.assemble(**args)
    for view in ('original','k8-compact','k6-compact'):
        index_name = package.mc_manifest.FILES[view][0]
        mapping = package.load_json(args['output']/index_name)['weight_map']
        assert all(mapping[name]=='mixed.safetensors' for name in shared_mtp)
    card = (args['output']/'README.md').read_text()
    assert f'({mtp_count} modules)' in card
    assert 'neither restored to BF16 nor encoded at K8' in card

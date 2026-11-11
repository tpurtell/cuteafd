"""Tiny clean-repack fixtures and exact destination/parent registry contract."""
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glmf_k325_mc as mixed
import mc_package as package
import mc_publish
from test_mc_package import dump, fixture, indexed, tensors


def setup(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    module = 'model.language_model.layers.3.mlp.experts.0.gate_proj'
    geometry = dict(in_features=128, out_features=128, stored_out_features=128)
    shared = {'norm.weight': torch.ones(128, dtype=torch.bfloat16),
              'model.language_model.layers.45.norm.weight': torch.ones(128, dtype=torch.bfloat16)}
    indexed(args['parent'], {'mixed.safetensors': tensors('head', geometry, 6) | tensors(module, geometry, 4) | shared})
    q = package.load_json(args['parent'] / 'quantize_config.json')
    q['tensor_storage'][module] = dict(bits_per_weight=4, stored_tensors={n: dict(
        torch_dtype=package.DTYPES[dtype], shape=shape) for n, (dtype, shape) in package.packed_specs(module, dict(geometry, bits=4)).items()})
    q['module_include'].append('^' + module.replace('.', r'\.') + '$')
    dump(args['parent'], 'quantize_config.json', q)
    (args['parent'] / 'README.md').write_text('---\nlicense: mit\n---\n# Parent\n')
    package.assemble(**args)
    parent = args['output']
    # Tiny fixture carries a native tensor alongside K3 experts; it must never
    # be carried into the output, even as an inactive physical payload.
    experts = tmp_path / mixed.EXPERT_REVISION
    experts.mkdir()
    indexed(experts, {'source.safetensors': tensors(module, geometry, 3) | {'forbidden.weight_scale_inv': torch.ones(1)}})
    dump(experts, 'quantize_config.json', dict(bits=3, tensor_storage={module: dict(bits_per_weight=3,
        stored_tensors={n: dict(torch_dtype=package.DTYPES[dtype], shape=shape)
        for n, (dtype, shape) in package.packed_specs(module, dict(geometry, bits=3)).items()})}))
    audit = tmp_path / 'audit.json'
    state = package.load_json(parent / 'package-state.json')
    files = state['new_files'] + list(state['copies']) + ['package-validation.json', 'package-provenance.json']
    dump(tmp_path, 'audit.json', dict(commit=mixed.PARENT_REVISION,
        repo_id='wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K4-MC', files={n: {} for n in files}))
    monkeypatch.setattr(mixed, 'EXPECTED_TARGET_TENSORS', 4)
    return dict(parent=parent, experts=experts, output=tmp_path / 'clean', parent_audit=audit)


def test_clean_experts_core_and_shared_exact_without_inactive_native(tmp_path, monkeypatch):
    args = setup(tmp_path, monkeypatch)
    result = mixed.assemble(**args)
    assert result['counts']['target-expert'] == 4
    state = package.load_json(args['output'] / 'package-state.json')
    assert 'mixed.safetensors' not in state['copies']
    assert any(n.startswith('shared-native-') for n in state['new_files'])
    assert any(n.startswith('k6-core-') for n in state['new_files'])
    assert not any('forbidden' in n for n in state['tensor_origins'])
    assert mixed.assemble(**args) == result
    for view in ('original', 'k8-compact', 'k6-compact'):
        index, config, quant, _ = package.mc_manifest.FILES[view]
        inv = package.inventory(args['output'], index)
        assert 'norm.weight' in inv['mapping']
        assert package.load_json(args['output'] / quant)['bits'] == 3
        assert package.load_json(args['output'] / config)['quantization_config']['bits'] == 3
    packet = mc_publish.publication_inventory(args['output'])
    assert packet['repo_id'] == 'wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-MC'
    assert packet['parent_repo'] == 'wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K4-MC'
    assert packet['parent_revision'] == '91a8dbc9c12e790c65aa1a38c6aede6a3e23a77d'


def test_independent_audit_rejects_expert_tamper(tmp_path, monkeypatch):
    args = setup(tmp_path, monkeypatch)
    mixed.assemble(**args)
    path = next(args['output'].glob('mixed-experts-*.safetensors'))
    with path.open('r+b') as stream:
        stream.seek(-1, 2)
        stream.write(b'X')
    with pytest.raises(ValueError, match='committed file'):
        mixed.validate(args['output'])


def test_wrong_published_parent_refused_before_output(tmp_path, monkeypatch):
    args = setup(tmp_path, monkeypatch)
    audit = package.load_json(args['parent_audit'])
    audit['commit'] = 'a' * 40
    dump(tmp_path, 'audit.json', audit)
    with pytest.raises(ValueError, match='independently verified'):
        mixed.assemble(**args)
    assert not args['output'].exists()


def test_target_expert_classifier_excludes_mtp_and_native():
    assert mixed.target_expert('model.language_model.layers.3.mlp.experts.0.up_proj.trellis')
    assert not mixed.target_expert('model.language_model.layers.45.mlp.experts.0.up_proj.trellis')
    assert not mixed.target_expert('model.language_model.layers.3.mlp.experts.0.up_proj.weight')

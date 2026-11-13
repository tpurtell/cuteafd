"""CPU guards for K3.1 isolated calibrated K8 core production."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimi_k31.k2_core_k6 import core_only as core
from kimi_k3.core_k6.run import ProjectionCheckpoints
from kimi_k3.uniform_k2.storage import atomic_json


class Progress:
    def emit(self, *args, **kwargs):
        pass


def metrics(value=1e-5):
    return dict(hessian_weighted_relative_error=value, quantizer_path='hessian_ldlq',
                hessian_metric_status='ok',
                hessian_numerical_contract='signed-block-hadamard-congruence-fp64-v1')


def test_reviewed_core_only_selection():
    selected = core.selection()
    assert len(selected) == 926
    assert {g['bits'] for g in selected.values()} == {8}
    assert not any('.experts.' in n or '.indexer.' in n for n in selected)
    assert 'language_model.lm_head' in selected
    for bits in (6, 4, True, 8.0):
        with pytest.raises(ValueError):
            core.selection(bits)


@pytest.mark.parametrize('bits', [None, 8])
def test_checkpoint_explicit_bits_preserves_legacy_default(tmp_path, bits):
    calls = []
    class Numerical:
        def quantize_core_projection(self, weight, raw, **kwargs):
            calls.append(kwargs)
            return {'trellis': torch.zeros(1, dtype=torch.int16)}, metrics(), {'bits': kwargs['bits'], 'hessian_rows': 3}
    g = dict(in_features=4, out_features=4)
    if bits is not None:
        g['bits'] = bits
    cp = ProjectionCheckpoints(tmp_path/'packed', tmp_path/'evidence', {}, {'head': g},
                               Progress(), device='cpu', numerical=Numerical())
    cp.ensure('head', torch.ones(4, 4), {'count': 3})
    cp.ensure('head', torch.ones(4, 4))
    assert calls == [dict(bits=6 if bits is None else bits, seed=787, sigma=.025, device='cpu')]


def test_native_expert_replay_never_invokes_quantization():
    engine = object.__new__(core.Engine)
    assert engine._experts(SimpleNamespace(), SimpleNamespace()) is None
    # The inherited phase loads original checkpoint experts; no K2 replacements.
    assert core.Engine._phase_e is core.run.CoreEngine._phase_e


@pytest.mark.parametrize('value', [float('nan'), -1, .01])
def test_projection_rejects_invalid_or_excessive_error_before_oracle(tmp_path, monkeypatch, value):
    cp = core.Checkpoints(tmp_path/'packed', tmp_path/'evidence', {},
                          {'head': dict(bits=8)}, Progress(), device='cpu', numerical=object())
    monkeypatch.setattr(core, 'sampled_packed_oracle', lambda *a, **k: pytest.fail('bad metrics reached oracle'))
    with pytest.raises(ValueError):
        cp._evidence('head', dict(metrics=metrics(value)))


def test_projection_oracle_required_on_fresh_resume(tmp_path, monkeypatch):
    module = 'language_model.lm_head'
    geometry = dict(bits=8, in_features=128, out_features=128, stored_out_features=128)
    calls = []
    monkeypatch.setattr(core, 'sampled_packed_oracle', lambda *a, **k: calls.append(a) or
                        dict(passed=True, bits=8, schema='cuteafd-core-packed-byte-oracle-v1'))
    row = dict(module=module, logical_shape=[128, 128], metrics=metrics(), evidence=dict(bits=8))
    for _ in range(2):
        cp = core.Checkpoints(tmp_path/'packed', tmp_path/'evidence', {}, {module: geometry},
                              Progress(), device='cpu', numerical=object())
        cp._evidence(module, row)
        cp._evidence(module, row)
    assert len(calls) == 2
    public = json.loads((tmp_path/'evidence'/(module+'.json')).read_text())
    assert public['packed_byte_oracle']['passed'] is True


def test_finalization_rejects_missing_or_failed_oracle(tmp_path):
    module = 'language_model.lm_head'
    selected = {module: dict(bits=8, in_features=128, out_features=128)}
    path = tmp_path/'evidence/head'/(module+'.json')
    row = dict(module=module, logical_shape=[128, 128], metrics=metrics(), evidence=dict(bits=8))
    atomic_json(path, row)
    with pytest.raises(ValueError, match='oracle'):
        core.validate_evidence(selected, tmp_path)
    row['packed_byte_oracle'] = dict(passed=True, bits=8, schema='cuteafd-core-packed-byte-oracle-v1')
    atomic_json(path, row)
    core.validate_evidence(selected, tmp_path)
    row['packed_byte_oracle']['passed'] = False
    atomic_json(path, row)
    with pytest.raises(ValueError, match='oracle'):
        core.validate_evidence(selected, tmp_path)


def test_plan_validates_corpus_and_paths_without_cuda(tmp_path, monkeypatch):
    original = dict(identity=dict(recipe=copy.deepcopy(core.run.RECIPE), code={}),
                    temporary_reserve_bytes=100)
    monkeypatch.setattr(core.run, 'make_plan', lambda *a: original)
    monkeypatch.setattr(core.run, 'load_token_records', lambda *a: [{}] * 1426)
    monkeypatch.setattr(torch.cuda, 'set_per_process_memory_fraction', lambda *a: pytest.fail('preflight CUDA'))
    plan = core.make_plan(object(), tmp_path/'tokens', tmp_path/'run', tmp_path/'export', 8)
    assert plan['identity']['recipe']['core_only'] is True
    assert plan['identity']['recipe']['expert_bits'] is None
    assert original['identity']['recipe'] == core.run.RECIPE
    assert plan['inventory']['core_modules'] == 926
    assert plan['required_free_bytes'] > plan['inventory']['core_payload_bytes']
    with pytest.raises(ValueError, match='disjoint'):
        core.make_plan(object(), tmp_path/'tokens', tmp_path/'run', tmp_path/'run/export', 8)
    monkeypatch.setattr(core.run, 'load_token_records', lambda *a: [{}])
    with pytest.raises(ValueError, match='full'):
        core.make_plan(object(), tmp_path/'tokens', tmp_path/'run', tmp_path/'export', 8)


def test_core_adapter_supplies_complete_causal_and_codec_surface(tmp_path):
    engine = core.Engine(object(), [], tmp_path/'run', tmp_path/'export', {}, {}, Progress(), device='cpu')
    from gptqmodel.utils import kimi_k31, kimi_k3_core_stream, kimi_k3_core_format
    assert engine.math.DenseRawHessian is kimi_k31.DenseRawHessian
    for name in ('attention_inputs', 'attention_finish', 'capture_linear_input',
                 'shared_down_input', 'routed_up_input', 'complete_moe', 'final_head_input'):
        assert getattr(engine.math, name) is getattr(kimi_k3_core_stream, name)
    assert engine.math.quantize_core_projection is kimi_k3_core_format.quantize_core_projection
    assert engine.math.reconstruct_core_projection is kimi_k3_core_format.reconstruct_core_projection

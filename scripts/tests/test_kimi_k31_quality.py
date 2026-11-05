"""New and resumed training-error alarms precede projection installation."""
from pathlib import Path
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'quantization'))
from kimi_k31.k2_core_k6 import quality
from kimi_k3.uniform_k2.storage import atomic_json

MODULE = 'language_model.model.layers.0.self_attn.q_proj'


def metrics(value):
    return dict(hessian_weighted_relative_error=value, quantizer_path='hessian_ldlq',
        hessian_metric_status='ok', hessian_numerical_contract=quality.HESSIAN_CONTRACT)


def setup(tmp_path, value):
    import torch
    baseline = tmp_path / quality.BASELINE_REVISION
    atomic_json(baseline / 'quantization-errors/core-k6/layer-000' / (MODULE + '.json'),
                dict(module=MODULE, metrics=metrics(.0002)))
    events, reconstructed = [], []
    def quantize(weight, raw, **kwargs):
        return dict(trellis=weight.clone()), metrics(value), dict(hessian_rows=4)
    def reconstruct(packed, shape, **kwargs):
        reconstructed.append(True)
        return packed['trellis']
    numerical = types.SimpleNamespace(quantize_core_projection=quantize,
                                     reconstruct_core_projection=reconstruct)
    def make():
        return quality.QualityCheckpoints(tmp_path / 'projections', tmp_path / 'evidence',
            dict(producer='test'), {MODULE: dict(out_features=2, in_features=2)},
            types.SimpleNamespace(emit=lambda *a, **kw: events.append((a, kw))),
            numerical=numerical, device='cpu', baseline_snapshot=baseline)
    return make, torch.ones((2, 2)), reconstructed


def test_new_and_resumed_good_checkpoint_retains_labelled_diagnostics(tmp_path):
    import torch
    make, weight, reconstructed = setup(tmp_path, .0003)
    first = make()
    first.ensure(MODULE, weight, dict(H=weight, count=4, finalized=False))
    second = make()
    second.ensure(MODULE, weight)  # Fresh object resumes the actual serialized receipt.
    linear = torch.nn.Linear(2, 2)
    second.install(MODULE, linear)
    assert reconstructed == [True]
    import json
    evidence = json.loads((tmp_path / 'evidence' / (MODULE + '.json')).read_text())
    alarm = evidence['quality']
    assert alarm['heldout'] is False and alarm['domain'] == 'regularized-Hessian-training'
    assert alarm['limit'] == .001 and alarm['ratio_to_k3'] == pytest.approx(1.5)
    assert alarm['k3_baseline']['revision'] == quality.BASELINE_REVISION


def test_new_anomaly_is_not_installed_and_resume_remains_rejected(tmp_path):
    import torch
    make, weight, reconstructed = setup(tmp_path, .00101)
    first = make()
    with pytest.raises(ValueError, match='training-error alarm'):
        first.ensure(MODULE, weight, dict(H=weight, count=4, finalized=False))
    # The bad tensor may remain as forensic state, but a fresh process can
    # neither ensure nor install it. No quantization rerun silently masks it.
    second = make()
    with pytest.raises(ValueError, match='training-error alarm'):
        second.ensure(MODULE, weight)
    with pytest.raises(ValueError, match='training-error alarm'):
        second.install(MODULE, torch.nn.Linear(2, 2))
    assert reconstructed == []


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -.001, True])
def test_invalid_metric_is_never_accepted(value):
    with pytest.raises(ValueError, match='invalid calibrated'):
        quality.training_error(metrics(value))


def test_existing_absolute_limit_is_inclusive(tmp_path):
    make, weight, _ = setup(tmp_path, quality.LIMIT)
    make().ensure(MODULE, weight, dict(H=weight, count=4, finalized=False))

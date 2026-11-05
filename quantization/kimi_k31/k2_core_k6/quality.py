"""K6 training-metric alarm and explicitly labelled K3 diagnostics.

The 1e-3 absolute alarm reuses the existing K6 operational ceiling. This is
not a heldout measurement or full-model qualification: production receipts
contain the quantizer's regularized-Hessian training error. No damping,
bitrate, activations, covariance or quantized weights are changed here.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

from kimi_k3.core_k6.run import ProjectionCheckpoints
from kimi_k3.uniform_k2.storage import atomic_json, sha256_file

LIMIT = 1e-3
BASELINE_REVISION = '1a28050d2954426a43a3db705e53ca120e5f2b47'
DEFAULT_BASELINE = Path('/mnt/sparknest/hf-home/hub') / (
    'models--wrldsuksgo2mars--Kimi-K3-EXL3-K2-Core-K6/snapshots/' + BASELINE_REVISION)
HESSIAN_CONTRACT = 'signed-block-hadamard-congruence-fp64-v1'


def training_error(metrics):
    value = metrics.get('hessian_weighted_relative_error')
    if (isinstance(value, bool) or not isinstance(value, (float, int))
            or not math.isfinite(value) or value < 0
            or metrics.get('quantizer_path') != 'hessian_ldlq'
            or metrics.get('hessian_metric_status') != 'ok'
            or metrics.get('hessian_numerical_contract') != HESSIAN_CONTRACT):
        raise ValueError('invalid calibrated K6 training metric')
    return value


def baseline_error(snapshot, module):
    snapshot = Path(snapshot)
    if snapshot.name != BASELINE_REVISION:
        raise ValueError('K3 diagnostics require the pinned published baseline')
    if module == 'language_model.lm_head':
        directory = 'head'
    else:
        match = re.fullmatch(r'language_model\.model\.layers\.(\d+)\..+', module)
        if match is None or not 0 <= int(match[1]) < 93:
            raise ValueError('K6 module is outside the Kimi baseline inventory')
        directory = f'layer-{int(match[1]):03d}'
    path = snapshot / 'quantization-errors/core-k6' / directory / (module + '.json')
    row = json.loads(path.read_text())
    if row.get('module') != module:
        raise ValueError('K3 baseline module differs')
    return dict(repo_id='wrldsuksgo2mars/Kimi-K3-EXL3-K2-Core-K6',
        revision=BASELINE_REVISION, receipt=str(path.relative_to(snapshot)),
        receipt_sha256=sha256_file(path), metric='hessian_weighted_relative_error',
        value=training_error(row['metrics']),
        comparison='same-tensor-K3-production-training-metric-not-a-shared-heldout-distribution')


def assess(module, saved, *, baseline_snapshot=DEFAULT_BASELINE):
    value = training_error(saved.get('metrics', {}))
    baseline = baseline_error(baseline_snapshot, module)
    result = dict(schema='kimi-k31-core-training-alarm-v1',
        metric='hessian_weighted_relative_error', domain='regularized-Hessian-training',
        heldout=False, limit=LIMIT, value=value, passed=value <= LIMIT,
        k3_baseline=baseline,
        ratio_to_k3=value / baseline['value'] if baseline['value'] > 0 else None)
    if value > LIMIT:
        raise ValueError(f'K6 training-error alarm: {module}: {value:.9g} exceeds {LIMIT}; '
                         f'K3 same-tensor training error {baseline["value"]:.9g}')
    return result


class QualityCheckpoints(ProjectionCheckpoints):
    """Reject new and resumed anomalies before installing weights or replaying."""
    def __init__(self, *args, baseline_snapshot=DEFAULT_BASELINE, **kwargs):
        super().__init__(*args, **kwargs)
        self.baseline_snapshot = baseline_snapshot

    def _evidence(self, module, saved):
        # The reused coordinator calls this hook for BOTH newly saved tensors
        # and existing checkpoint loads, including install/replacements paths.
        quality = assess(module, saved, baseline_snapshot=self.baseline_snapshot)
        public = {key: saved[key] for key in ('module', 'logical_shape', 'metrics', 'evidence')}
        public['quality'] = quality
        atomic_json(self.evidence_directory / (module + '.json'), public)

    def ensure(self, module, weight, raw_hessian=None):
        path = super().ensure(module, weight, raw_hessian)
        # Re-read through the ordinary checksum/identity path. A new or resumed
        # metric must pass the same guard before the caller can install it.
        if self.load(module) is None:
            raise ValueError('K6 quality guard lost the projection commitment')
        return path

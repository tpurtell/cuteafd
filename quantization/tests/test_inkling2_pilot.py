import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from inkling2.pilot import BARS, REQUIRED, Checks, atomic_json, run_bounded, successful


def test_brief_bars_and_head_are_required_without_drafter():
    assert BARS=={8:1e-4,6:1e-3,4:1e-2}
    assert {'wr_du_k8','gate_attn_k6','unembed_k8','unembed_k6','head_fp32_logits','expert_down_k4'}<=set(REQUIRED)
    assert not any('draft' in name or 'mtp' in name for name in REQUIRED)


def test_hard_cap_rejects_expansion_before_launch(tmp_path):
    with pytest.raises(ValueError,match='600'):
        run_bounded([],tmp_path/'receipt.json',601,run=lambda *a,**k:pytest.fail('launched'))


def test_timeout_preserves_named_progress_and_fails_closed(tmp_path):
    output=tmp_path/'receipt.json'
    def timeout(*args,**kwargs):
        atomic_json(output,dict(checks=[dict(name='gpu_mxfp8_bytes',passed=True)]))
        assert kwargs['timeout']==600
        raise subprocess.TimeoutExpired('worker',600)
    result=run_bounded([],output,600,run=timeout)
    assert result['passed'] is False
    assert result['failure']=='wall-clock cap 600s exceeded'
    assert result['checks'][0]['passed'] is True


def test_worker_success_still_requires_all_named_checks(tmp_path):
    output=tmp_path/'receipt.json'
    def incomplete(*args,**kwargs):
        atomic_json(output,dict(passed=True,checks=[]))
        return SimpleNamespace(returncode=0)
    result=run_bounded([],output,60,run=incomplete)
    assert not result['passed']
    assert 'omitted' in result['failure']


def test_completed_pilot_is_not_restarted(tmp_path):
    output=tmp_path/'receipt.json'
    report=dict(passed=True,checks=[dict(name=name,passed=True) for name in REQUIRED])
    atomic_json(output,report)
    assert run_bounded([],output,60,run=lambda *a,**k:pytest.fail('restarted'))==report


def test_failed_receipt_requires_diagnosis_not_implicit_restart(tmp_path):
    output=tmp_path/'receipt.json'
    atomic_json(output,dict(passed=False,checks=[]))
    with pytest.raises(ValueError,match='diagnose'):
        run_bounded([],output,60,run=lambda *a,**k:pytest.fail('restarted'))


def test_check_exception_is_named_in_durable_failure_receipt(tmp_path):
    output=tmp_path/'receipt.json'
    checks=Checks(output,dict(checks=[]))
    def bad():
        raise ValueError('shape mismatch')
    with pytest.raises(ValueError,match='shape mismatch'):
        checks.run('source_global_attention',bad)
    row=json.loads(output.read_text())['checks'][0]
    assert row['name']=='source_global_attention' and row['passed'] is False
    assert row['error']=='ValueError: shape mismatch'

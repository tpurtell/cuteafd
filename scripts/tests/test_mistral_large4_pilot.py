"""Bounded pilot selection, heldout math and receipt-preserving recovery."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'quantization'))
from kimi_k3.uniform_k2.storage import atomic_json, commit_safetensors
from mistral_large4.core_k6 import native, numerical, pilot


def test_short_pilot_uses_complete_disjoint_original_prompts():
    train = [dict(id=f'c{i}', input_ids=list(range(i+3))) for i in range(4)]
    heldout = [dict(id=f'h{i}', input_ids=list(range(i+5))) for i in range(4)]
    actual = pilot.select_records(train, heldout)
    assert actual == dict(calibration=train[:2], heldout=heldout[:2])
    assert sum(len(row['input_ids']) for row in actual['calibration']) == 7
    with pytest.raises(ValueError, match='overlap'):
        pilot.select_records(train, [train[0], heldout[1]])


def test_short_pilot_has_twelve_matrices_and_sixteen_quantizations():
    specs = [pilot.candidate_specs(layer) for layer in pilot.PROOF_LAYERS]
    assert sum(len(spec) for spec in specs) == 12
    assert sum(len(bits) for spec in specs for bits in spec.values()) == 16
    assert not any('.indexer.' in module or '.wkv_b' in module for spec in specs for module in spec)
    assert 'layers.0.attention.wkv_a_with_mqa' in specs[0]
    assert pilot.WITNESS_TOKENS > 2048


@pytest.mark.parametrize('bits,nmse,expected', [(6, .001, True), (6, .001001, False),
    (5, .02, True), (4, .04, True), (6, float('nan'), False), (5, float('inf'), False)])
def test_core_acceptance_absolute_nmse(bits, nmse, expected):
    assert pilot.accepted('layers.0.attention.wq', bits, nmse) is expected


def test_expert_acceptance_and_padding_accounting():
    assert pilot.accepted('layers.3.experts.0.w1', 4, .05)
    assert not pilot.accepted('layers.3.experts.0.w1', 4, .05001)
    expected = 6144 * 640 * 6 // 8 + (6144+640)*2 + 4
    assert pilot.tp2_bytes('layers.0.attention.wkv_a_with_mqa', [576,6144], 6) == [expected]*2


def test_heldout_nmse_matches_direct_quadratic_and_rejects_invalid_energy():
    torch.manual_seed(55)
    weight = torch.randn(7,4)
    candidate = weight + .05*torch.randn_like(weight)
    heldout = torch.randn(31,4)
    H = heldout.T @ heldout
    error = candidate-weight
    expected = torch.trace(error.double()@H.double()@error.double().T)/torch.trace(weight.double()@H.double()@weight.double().T)
    assert pilot.weighted_nmse(weight,candidate,H) == pytest.approx(float(expected), rel=2e-6)
    assert pilot.weighted_nmse(weight,weight,H) == 0
    with pytest.raises(ValueError, match='invalid'):
        pilot.weighted_nmse(weight,candidate,torch.zeros_like(H))


def test_verified_capture_survives_identity_failure_and_rejects_corruption(tmp_path):
    directory = tmp_path/'capture/layer-003'
    directory.mkdir(parents=True)
    path = directory/'calibration-test.safetensors'
    entry = dict(filename=path.name, split='calibration', module='test', count=3,
        **commit_safetensors(path,dict(H=torch.eye(4))))
    manifest = dict(identity={'producer':'frozen'},layer=3,hessians=[entry],proofs=[],expert=2)
    atomic_json(directory/'manifest.json',manifest)
    assert pilot.saved_capture(tmp_path,3,{'producer':'frozen'}) == manifest
    before = path.read_bytes()
    with pytest.raises(ValueError,match='identity'):
        pilot.saved_capture(tmp_path,3,{'producer':'changed'})
    assert path.read_bytes() == before
    with path.open('ab') as stream:
        stream.write(b'corrupt')
    with pytest.raises(ValueError,match='changed'):
        pilot.saved_capture(tmp_path,3,{'producer':'frozen'})
    assert path.exists(), 'a numerical/corruption diagnosis must not delete committed work'


def test_capture_separates_calibration_from_heldout_and_removes_hooks(monkeypatch):
    class Raw:
        def __init__(self,width,device):
            self.H,self.count=torch.zeros(width,width,device=device),0
        def add(self,value):
            rows=value.reshape(-1,value.shape[-1]).float()
            self.H.addmm_(rows.T,rows)
            self.count+=len(rows)
    monkeypatch.setitem(numerical.__dict__,'DenseRawHessian',Raw)
    modules={name:torch.nn.Linear(4,4,bias=False) for name in pilot.candidate_specs(0)}
    monkeypatch.setattr(native,'linear_module',lambda block,name:modules[name])
    shared=dict(hessians={})
    train=torch.randn(1,5,4)
    heldout=torch.randn(1,3,4)
    with pilot.Capture(SimpleNamespace(),0,shared) as capture:
        capture.split='calibration'
        for module in modules.values():module(train)
        capture.split='heldout'
        for module in modules.values():module(heldout)
    for name,module in modules.items():
        assert not module._forward_pre_hooks
        assert shared['hessians']['calibration',name].count==5
        assert shared['hessians']['heldout',name].count==3
        assert torch.equal(shared['hessians']['calibration',name].H,train[0].T@train[0])


def test_pilot_budget_stops_without_mutating_cached_work(tmp_path):
    retained=tmp_path/'retained.txt'
    retained.write_text('committed')
    progress=pilot.Progress(tmp_path,0)
    progress.started-=1
    with pytest.raises(TimeoutError,match='cache retained'):
        progress.check()
    assert retained.read_text()=='committed'


def complete_summary_inputs():
    selected=pilot.select_records([dict(id=f'c{i}',input_ids=[1,2,3]) for i in range(2)],
                                  [dict(id=f'h{i}',input_ids=[4,5]) for i in range(2)])
    identity={'test':'producer'}
    proofs=[dict(layer=layer,split=split,id=row['id'],tokens=len(row['input_ids']),
        hidden='byte-exact',reuse_state='byte-exact')
        for layer in pilot.PROOF_LAYERS for split,records in selected.items() for row in records]
    witness=dict(identity=identity,status='passed',results=[dict(layer=layer,tokens=2205,topk_width=2048,
        attention_output='byte-exact',reuse_state='byte-exact') for layer in pilot.WITNESS_LAYERS])
    results=[]
    for layer in pilot.PROOF_LAYERS:
        for name,arms in pilot.candidate_specs(layer).items():
            name=name.replace('$selected','7')
            for bits in arms:
                results.append(dict(tensor=name,arm=f'mcg-k{bits}',bits=bits,nmse=.0001,passed=True,
                    serialized_replay='passed',calibration_rows=100,heldout_rows=20))
            control='source-bf16' if name.endswith('wkv_a_with_mqa') else 'source-fp8'
            for arm in (control,'dequantized-bf16'):
                results.append(dict(tensor=name,arm=arm,nmse=0.,passed=True))
    return results,proofs,witness,selected,identity


def test_summary_flags_require_complete_declared_inventories():
    args=complete_summary_inputs()
    assert pilot.validate_summary_inputs(*args,72)==dict(native_boundaries_byte_exact=True,
        serialized_projection_replay=True,source_layers=72)
    results,proofs,witness,selected,identity=args
    with pytest.raises(ValueError,match='72'):
        pilot.validate_summary_inputs(*args,71)
    with pytest.raises(ValueError,match='proof inventory'):
        pilot.validate_summary_inputs(results,proofs[:-1],witness,selected,identity,72)
    with pytest.raises(ValueError,match='candidate/control inventory'):
        pilot.validate_summary_inputs(results[:-1],proofs,witness,selected,identity,72)
    witness['results'][0]['tokens']=2048
    with pytest.raises(ValueError,match='sparse'):
        pilot.validate_summary_inputs(*args,72)


def test_summary_rejects_unverified_serialized_projection_or_failed_candidate():
    args=complete_summary_inputs()
    args[0][0]['serialized_replay']='missing'
    with pytest.raises(ValueError,match='serialized replay'):
        pilot.validate_summary_inputs(*args,72)
    args=complete_summary_inputs()
    args[0][0]['nmse']=.1
    with pytest.raises(ValueError,match='acceptance'):
        pilot.validate_summary_inputs(*args,72)
    args=complete_summary_inputs()
    args[0][0]['passed']=False
    with pytest.raises(ValueError,match='failed'):
        pilot.validate_summary_inputs(*args,72)

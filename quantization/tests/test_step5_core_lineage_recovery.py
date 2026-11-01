"""Audited resume keeps old commitments immutable and rejects other producers."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kimi_k3.uniform_k2.storage import atomic_json,commit_safetensors
from step5.core_k6 import storage,run
from step5.core_k6.compat import identity_digest
from test_step5_core_compat import identities,manifest

MODULE='model.layers.0.self_attn.q_proj'


def projection(root,identity,name=MODULE):
    path=root/'layers/000/projections'/(name+'.safetensors')
    path.parent.mkdir(parents=True,exist_ok=True)
    saved=dict(identity=identity,module=name,bits=6,logical_shape=[2,2],metrics={},evidence={},
               **commit_safetensors(path,{'packed':torch.ones(2)}))
    atomic_json(path.with_suffix('.json'),saved)
    evidence=root/'evidence/layer-000';evidence.mkdir(parents=True,exist_ok=True)
    atomic_json(evidence/(name+'.json'),{k:saved[k] for k in ('module','bits','logical_shape','metrics','evidence')})
    return path,saved


def checkpoints(root,identity,name=MODULE):
    return storage.ProjectionStore(root/'layers/000/projections',root/'evidence/layer-000',identity,
        {name:dict(bits=6,out_features=2,in_features=2)},None,None,'cpu')


def test_resume_old_projection_preserves_receipt_and_evidence_bytes(tmp_path):
    old,current=identities();manifest(tmp_path,old,current,MODULE)
    path,saved=projection(tmp_path,old)
    receipt=path.with_suffix('.json');evidence=tmp_path/'evidence/layer-000'/(MODULE+'.json')
    before=[p.read_bytes() for p in (receipt,evidence)]
    assert checkpoints(tmp_path,current).load(MODULE)==saved
    assert [p.read_bytes() for p in (receipt,evidence)]==before


@pytest.mark.parametrize('damage',['unlisted','geometry','other_producer','payload'])
def test_projection_rejects_unlisted_origin_geometry_identity_and_corrupt_bytes(tmp_path,damage):
    old,current=identities();manifest(tmp_path,old,current,MODULE)
    name=MODULE if damage!='unlisted' else 'model.layers.1.self_attn.q_proj'
    path,saved=projection(tmp_path,old,name)
    if damage=='geometry':saved['bits']=4
    elif damage=='other_producer':saved['identity']=deepcopy(old);saved['identity']['source_revision']='x'*40
    elif damage=='payload':path.write_bytes(path.read_bytes()[:-1]+b'x')
    atomic_json(path.with_suffix('.json'),saved)
    with pytest.raises(ValueError):checkpoints(tmp_path,current,name).load(name)


def test_new_evidence_records_active_producer_and_engine_uses_wrapper(tmp_path):
    old,current=identities();manifest(tmp_path,old,current,MODULE)
    name='model.layers.4.self_attn.q_proj'
    engine=object.__new__(run.Engine)
    engine.root=tmp_path;engine.evidence_root=tmp_path/'evidence';engine.identity=current
    engine.selection={name:dict(bits=6,out_features=2,in_features=2)}
    engine.progress=None;engine.math=None;engine.device='cpu'
    store=engine.checkpoints(4)
    assert isinstance(store,storage.ProjectionStore)
    saved=dict(identity=current,module=name,bits=6,logical_shape=[2,2],metrics={},evidence={})
    store.record_evidence(name,saved)
    result=json.loads((store.evidence/(name+'.json')).read_text())
    assert result['producer_identity_sha256']==identity_digest(current)


def test_frontier_and_phase_resume_validate_bytes_and_keep_original_identity(tmp_path):
    old,current=identities();manifest(tmp_path,old,current,MODULE)
    frontier=storage.RollingBoundary(tmp_path/'frontier',old)
    directory=frontier.directory(-1);directory.mkdir()
    entry=dict(filename='000.safetensors',**commit_safetensors(directory/'000.safetensors',{'hidden':torch.ones(2)}))
    frontier.commit(-1,[entry])
    phases=storage.StageStore(tmp_path/'stages',old)
    directory=phases.directory('d');directory.mkdir()
    phase_entry=dict(filename='000.safetensors',**commit_safetensors(directory/'000.safetensors',{'hidden':torch.ones(2)}))
    phases.commit('d',[phase_entry])
    originals={p:p.read_bytes() for p in tmp_path.rglob('*.json')}
    assert storage.RollingBoundary(tmp_path/'frontier',current).current(checksum=True)['identity']==old
    assert storage.StageStore(tmp_path/'stages',current).load('d',checksum=True)['identity']==old
    assert all(p.read_bytes()==raw for p,raw in originals.items())
    (directory/'000.safetensors').write_bytes(b'corrupt')
    with pytest.raises(ValueError):storage.StageStore(tmp_path/'stages',current).load('d',checksum=True)


def test_route_resume_keeps_manifest_and_uses_bounded_accumulator(tmp_path,monkeypatch):
    old,current=identities();manifest(tmp_path,old,current,MODULE)
    values=torch.arange(4,dtype=torch.bfloat16).reshape(2,2)
    progress=SimpleNamespace(emit=lambda *a,**k:None)
    def router(x):
        return torch.arange(8).repeat(len(x),1),None,torch.arange(8,16).repeat(len(x),1)
    spool=storage.RouteSpool.create(tmp_path/'routes',old,2,[(0,{},dict(mlp_input=values))],
        router,progress,3,width=2,experts=16,minimum=0)
    spool.close();path=tmp_path/'routes/manifest.json';before=path.read_bytes()
    calls=[];original=storage.accumulate_hessian_fp32
    def accumulate(H,x):calls.append(len(x));return original(H,x)
    monkeypatch.setattr(storage,'accumulate_hessian_fp32',accumulate)
    spool=storage.RouteSpool(tmp_path/'routes',current,2,width=2,experts=16,minimum=0)
    raw,evidence=spool.hessian(0,device='cpu')
    assert torch.equal(raw['H'],values.float().T@values.float())
    assert calls==[2] and evidence['hessian_accumulation']==storage.HESSIAN_ACCUMULATION_CONTRACT
    assert path.read_bytes()==before
    spool.close()

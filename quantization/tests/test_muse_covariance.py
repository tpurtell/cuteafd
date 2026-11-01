"""Bounded actual-row products against independent FP64 covariance oracles."""
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from muse_spark.core_k6 import codec
from muse_spark.core_k6.storage import RouteSpool


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float32])
def test_actual_rows_and_dtype_bound_every_product(monkeypatch,dtype):
    x=torch.linspace(-2,2,301*16).reshape(301,16).to(dtype)
    x[0,:4]=torch.tensor([0.,-0.,1.,-1.],dtype=dtype)
    x[1,:4]=torch.nextafter(x[0,:4],torch.full((4,),float('inf'),dtype=dtype))
    state=codec.covariance_accumulator(16,'cpu');parts=[];original=torch.mm
    def observed(left,right,*a,**kw):
        assert right.dtype==left.dtype==torch.float32
        assert torch.equal(left,right.T)
        parts.append(right.clone())
        return original(left,right,*a,**kw)
    monkeypatch.setattr(torch,'mm',observed)
    assert state.add(x)==301
    H=state.snapshot()
    assert [len(p) for p in parts]==[128,128,45]
    assert torch.equal(torch.cat(parts),x.float())
    reference=x.double().T@x.double()
    assert float(torch.linalg.norm(H.double()-reference)/torch.linalg.norm(reference))<=1e-6
    assert bool(torch.isfinite(H).all())


def test_dense_counts_and_rejects_invalid_rows_without_changing_state():
    capture=codec.DenseRawHessian(8,device='cpu')
    for x in (torch.empty(0,8),torch.ones(3,7),torch.full((3,8),float('nan'))):
        with pytest.raises(ValueError):capture.add(x)
        assert capture.count==0 and torch.count_nonzero(capture.H)==0
    x=torch.randn(317,8,generator=torch.Generator().manual_seed(7)).bfloat16()
    capture.add(x[:200]);capture.add(x[200:])
    raw=capture.raw_dict(minimum_rows=317)
    assert raw['count']==317 and raw['identity_rows']==0
    assert raw['accumulation_contract']==codec.ACCUMULATION_CONTRACT
    oracle=x.double().T@x.double()
    assert float(torch.linalg.norm(raw['H'].double()-oracle)/torch.linalg.norm(oracle))<1e-6


@pytest.mark.parametrize('down',[False,True])
def test_routed_product_bound_preserves_native_activation_shapes_and_floor(monkeypatch,down):
    spool=RouteSpool.__new__(RouteSpool)
    spool.width=128;spool.minimum=1024
    spool.route_rows=[np.arange(301,dtype=np.int64)]
    spool.recovery_rows=[np.arange(301,311,dtype=np.int64)]
    source=torch.randn(311,128,generator=torch.Generator().manual_seed(19)).bfloat16()
    spool.gather=lambda rows,device:source[rows].to(device)
    fused=torch.randn(256,128,generator=torch.Generator().manual_seed(29)).bfloat16() if down else None
    chunks=[];linear_shapes=[];original=torch.mm;linear=torch.nn.functional.linear
    def observed(left,right,*a,**kw):
        chunks.append(len(right));return original(left,right,*a,**kw)
    def native(x,w,*a,**kw):
        linear_shapes.append(len(x));return linear(x,w,*a,**kw)
    monkeypatch.setattr(torch,'mm',observed)
    monkeypatch.setattr(torch.nn.functional,'linear',native)
    raw,evidence=spool.hessian(0,down=fused,device='cpu')
    assert chunks==[128,128,45,10]
    assert linear_shapes==([301,10] if down else [])
    values=[]
    for x in (source[:301],source[301:]):
        if down:
            gate,up=linear(x,fused).chunk(2,dim=-1)
            x=torch.nn.functional.silu(gate)*up
        values.append(x.double())
    X=torch.cat(values)
    oracle=X.T@X+torch.eye(128,dtype=torch.float64)*713
    assert float(torch.linalg.norm(raw['H'].double()-oracle)/torch.linalg.norm(oracle))<1e-6
    assert raw['count']==1024 and raw['identity_rows']==713
    assert evidence['natural_rows']==301 and evidence['augmented_rows']==10
    assert raw['accumulation_contract']==evidence['accumulation_contract']==codec.ACCUMULATION_CONTRACT


def test_unmarked_legacy_covariance_cannot_enter_new_codec():
    with pytest.raises(ValueError,match='verified K128'):
        codec.quantize_projection(torch.ones(128,128,dtype=torch.bfloat16),
            dict(H=torch.eye(128),count=128,finalized=False))

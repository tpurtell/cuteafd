"""Causal replay and real disk-barrier faults using small native GLM blocks."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from glm5.core_k6 import run
from glm5.core_k6.storage import RouteSpool,StageStore
from gptqmodel.utils import glm5_core as native


class Progress:
    def __init__(self):
        self.events=[]
    def emit(self,event,**fields):
        self.events.append((event,fields))


def fixture(tmp_path):
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaDecoderLayer,GlmMoeDsaRotaryEmbedding
    config=GlmMoeDsaConfig(hidden_size=128,intermediate_size=256,moe_intermediate_size=128,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,q_lora_rank=128,
        kv_lora_rank=128,qk_nope_head_dim=64,qk_rope_head_dim=64,v_head_dim=128,
        n_routed_experts=256,num_experts_per_tok=8,n_group=1,topk_group=1,
        first_k_dense_replace=1,index_n_heads=2,index_head_dim=128,index_topk=3,
        indexer_types=["full"],rope_parameters=dict(rope_type="default",rope_theta=8000000.))
    config._attn_implementation="sdpa"
    torch.manual_seed(787)
    block=GlmMoeDsaDecoderLayer(config,0).to(torch.bfloat16).eval()
    for parameter in block.parameters():
        parameter.data.normal_(0,.03)
    block.self_attn.indexer.weights_proj=block.self_attn.indexer.weights_proj.float()
    embedding=torch.randn(32,128).bfloat16()
    class Source:
        def __init__(self):
            self.config=config.to_dict()
            self.loads=0
        def load_block(self,*args,**kwargs):
            self.loads+=1
            return copy.deepcopy(block)
        def tensor(self,name,device="cpu"):
            assert name=="model.embed_tokens.weight"
            return embedding.clone().to(device)
        def rotary(self,device="cpu"):
            return GlmMoeDsaRotaryEmbedding(config).to(device)
        def close(self):
            pass
    calls=[]
    class Numerical:
        DenseRawHessian=native.DenseRawHessian
        attention_inputs=staticmethod(native.attention_inputs)
        prepare_context=staticmethod(native.prepare_context)
        capture_linear_input=staticmethod(native.capture_linear_input)
        attention_finish=staticmethod(native.attention_finish)
        shared_down_input=staticmethod(native.shared_down_input)
        @staticmethod
        def quantize_projection(weight,raw,**kwargs):
            # A deliberately visible weight perturbation isolates orchestration
            # from trellis accuracy (independently qualified on actual CUDA).
            changed=(weight.float()*.875+.005).bfloat16()
            calls.append(dict(weight=weight.clone(),changed=changed.clone(),H=raw["H"].clone(),count=raw["count"]))
            packed=dict(trellis=changed.view(torch.int16).clone(),suh=torch.ones(weight.shape[1]).half(),
                        svh=torch.ones(weight.shape[0]).half(),mcg=torch.tensor(0xCBAC1FED,dtype=torch.uint32).view(torch.int32))
            return packed,dict(hessian_weighted_relative_error=.01),dict(hessian_rows=raw["count"],identity_rows=0)
        @staticmethod
        def reconstruct_projection(packed,shape,**kwargs):
            return packed["trellis"].view(torch.bfloat16).clone().to(kwargs["device"])
    selection={}
    for part in ("self_attn.q_a_proj","self_attn.q_b_proj","self_attn.kv_a_proj_with_mqa","self_attn.o_proj",
                 "mlp.gate_proj","mlp.up_proj","mlp.down_proj"):
        linear=block.get_submodule(part)
        selection["model.layers.0."+part]=dict(in_features=linear.in_features,out_features=linear.out_features,bits=6)
    records=[dict(id="a",input_ids=[1,2,3,4]),dict(id="b",input_ids=[5,6,7,8,9])]
    source=Source()
    engine=run.Engine(source,records,tmp_path/"run",tmp_path/"export",dict(run="test"),selection,
        Progress(),numerical=Numerical,device="cpu")
    return engine,source,block,embedding,records,calls


def test_causal_phases_match_final_native_block_and_downstream_hessians(tmp_path,monkeypatch):
    from glm5.core_k6 import export
    engine,source,block,embedding,records,calls=fixture(tmp_path)
    initial=engine.initial()
    class StopAfterF(Exception):
        pass
    def stop(*args,**kwargs):
        raise StopAfterF
    monkeypatch.setattr(export,"assemble_layer",stop)
    with torch.inference_mode(),pytest.raises(StopAfterF):
        engine.layer(0,initial)
    assert engine.boundary.current()["layer"]==-1
    store=StageStore(engine.root/"layers/000/stages",engine.identity)
    assert store.load("f",checksum=True) is not None
    assert store.load("a") is store.load("d") is None
    checkpoints=engine.checkpoints(0)
    for name in engine.selection:
        checkpoints.install(name,block.get_submodule(name.removeprefix("model.layers.0.")))
    captured_q=[];captured_down=[]
    handles=[block.self_attn.q_b_proj.register_forward_pre_hook(lambda m,a:captured_q.append(a[0].clone())),
             block.mlp.down_proj.register_forward_pre_hook(lambda m,a:captured_down.append(a[0].clone()))]
    for index,row,state in engine._read(store,"f"):
        ids=torch.tensor(row["input_ids"])
        x=torch.nn.functional.embedding(ids,embedding).unsqueeze(0)
        with torch.inference_mode():
            expected,topk=block(x,**native.prepare_context(x,source.rotary()))
        assert torch.equal(state["hidden"],expected)
        assert torch.equal(state["prev_topk_indices"],topk)
    for handle in handles:
        handle.remove()
    assert len(calls)==7 and all(row["count"]==9 for row in calls)
    # Quantizer invocation order is q_a,kv_a,q_b,o,gate,up,down.
    for call,values in ((calls[2],captured_q),(calls[6],captured_down)):
        x=torch.cat([x.reshape(-1,x.shape[-1]).double() for x in values])
        ref=x.T@x
        assert float(torch.linalg.norm(call["H"].double()-ref)/torch.linalg.norm(ref)) < 1e-6
    assert not torch.equal(calls[0]["weight"],calls[0]["changed"])
    # Recover after F but before assembly without reloading or requantizing.
    monkeypatch.setattr(export,"assemble_layer",lambda *a,**k:dict(payload_bytes=1))
    before=source.loads
    current=engine.layer(0,initial)
    assert current["layer"]==0 and source.loads==before and len(calls)==7
    assert not engine.boundary.directory(-1).exists()
    assert not list((engine.root/"layers/000/projections").glob("*.safetensors"))


def test_resume_after_frontier_commit_finishes_retirement(tmp_path,monkeypatch):
    from glm5.core_k6 import export
    engine,source,_,_,_,_=fixture(tmp_path)
    current=engine.initial()
    monkeypatch.setattr(export,"assemble_layer",lambda *a,**k:dict(payload_bytes=1))
    original=StageStore.clear
    def fail_after_pointer(store):
        raise RuntimeError("power loss after frontier publication")
    monkeypatch.setattr(StageStore,"clear",fail_after_pointer)
    with torch.inference_mode(),pytest.raises(RuntimeError,match="power loss"):
        engine.layer(0,current)
    assert engine.boundary.current(checksum=True)["layer"]==0
    assert list((engine.root/"layers/000/projections").glob("*.safetensors"))
    monkeypatch.setattr(StageStore,"clear",original)
    before=source.loads
    recovered=engine.initial()
    assert recovered["layer"]==0 and source.loads==before
    assert not (engine.root/"layers/000/stages").exists()
    assert not list((engine.root/"layers/000/projections").glob("*.safetensors"))


def test_partial_projection_resume_never_requantizes_committed_predecessor(tmp_path,monkeypatch):
    from glm5.core_k6 import export
    from glm5.core_k6.storage import ProjectionStore
    engine,source,_,_,_,calls=fixture(tmp_path)
    current=engine.initial()
    original=ProjectionStore.ensure
    def stop_after_qa(store,name,*args,**kwargs):
        result=original(store,name,*args,**kwargs)
        if name.endswith("self_attn.q_a_proj"):
            raise RuntimeError("crash after q_a bytes and receipt")
        return result
    monkeypatch.setattr(ProjectionStore,"ensure",stop_after_qa)
    with torch.inference_mode(),pytest.raises(RuntimeError,match="q_a bytes"):
        engine.layer(0,current)
    assert len(calls)==1 and engine.boundary.current()["layer"]==-1
    monkeypatch.setattr(ProjectionStore,"ensure",original)
    monkeypatch.setattr(export,"assemble_layer",lambda *a,**k:dict(payload_bytes=1))
    with torch.inference_mode():
        final=engine.layer(0,current)
    assert final["layer"]==0 and source.loads==2 and len(calls)==7


def test_cold_expert_routes_and_raw_floor_match_independent_fp64(tmp_path):
    torch.manual_seed(787)
    inputs=torch.randn(1,16,128).bfloat16()
    def router(x):
        return (torch.arange(8).expand(len(x),8),torch.ones(len(x),8),torch.arange(8,16).expand(len(x),8))
    spool=RouteSpool.create(tmp_path,dict(test=1),16,iter([(0,dict(id="a"),dict(mlp_input=inputs))]),
        router,Progress(),3,width=128,experts=32)
    try:
        for expert,natural,augmented in ((0,16,0),(8,0,16),(31,0,0)):
            raw,evidence=spool.hessian(expert,device="cpu")
            actual=natural+augmented
            x=inputs.reshape(16,128).double() if actual else torch.zeros(0,128,dtype=torch.float64)
            ref=x.T@x+(1024-actual)*torch.eye(128,dtype=torch.float64)
            assert raw["count"]==1024 and evidence["identity_rows"]==1024-actual
            assert evidence["natural_rows"]==natural and evidence["augmented_rows"]==augmented
            assert float(torch.linalg.norm(raw["H"].double()-ref)/torch.linalg.norm(ref)) < 1e-6
        gate=torch.randn(64,128).bfloat16();up=torch.randn(64,128).bfloat16()
        fused=torch.cat((gate,up),dim=0)
        raw,evidence=spool.hessian(8,down=fused,device="cpu")
        x=inputs.reshape(16,128)
        gate_value,up_value=torch.nn.functional.linear(x,fused).chunk(2,dim=-1)
        y=(torch.nn.functional.silu(gate_value)*up_value).double()
        ref=y.T@y+1008*torch.eye(64,dtype=torch.float64)
        assert float(torch.linalg.norm(raw["H"].double()-ref)/torch.linalg.norm(ref)) < 1e-6
        assert evidence["ranks"]==[9,16]
    finally:
        spool.close()

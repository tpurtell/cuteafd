"""Independent metric and native-source ownership checks for the bounded pilot."""
import json
from pathlib import Path
import sys

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glm5.core_k6.pilot import commit_batch, mcg_candidate, modules, tp2_projection_bytes, weighted_nmse, validate_capture_basis
from glm5.core_k6.source import GlmBF16Source
from kimi_k3.uniform_k2.storage import RollingBoundary


def test_mcg_k6_uses_pinned_default_without_optional_codebook_keyword():
    class Codec:
        def quantize_core_projection(self, weight, raw):
            assert weight == "weight" and raw == "raw"
            return {"mcg":"standard"}, {"evidence":"ok"}
    assert mcg_candidate("weight","raw","mcg-k6",Codec(),None) == {"mcg":"standard"}


def test_capture_recovery_rejects_changed_capture_math_or_other_dependencies():
    import hashlib
    old="def forward(x):\n    return x + 1\n\ndef score():\n    return 1\n"
    new=old.replace("return 1","return 2")
    key="quantization/glm5/core_k6/pilot.py"
    prior=dict(code={key:hashlib.sha256(old.encode()).hexdigest(),"source":"unchanged"},tokens=35229)
    current=dict(code={key:hashlib.sha256(new.encode()).hexdigest(),"source":"unchanged"},tokens=35229)
    validate_capture_basis(prior,current,old,new)
    with pytest.raises(ValueError,match="implementation changed"):
        validate_capture_basis(prior,current,old,new.replace("x + 1","x + 2"))
    current["tokens"]=35230
    with pytest.raises(ValueError,match="inputs or dependencies"):
        validate_capture_basis(prior,current,old,new)


def test_batch_receipts_publish_rolling_successor_and_retire_prior(tmp_path):
    boundary=RollingBoundary(tmp_path,dict(pilot="test"))
    old=commit_batch(boundary.directory(-1)/"00000.safetensors",dict(hidden=torch.ones(1,2,128)))
    boundary.commit(-1,[old])
    new=commit_batch(boundary.directory(0)/"00000.safetensors",dict(hidden=torch.ones(1,2,128)*2,
        prev_topk_indices=torch.zeros(1,2,2,dtype=torch.int32)))
    assert boundary.current()["layer"] == -1
    boundary.commit(0,[new])
    assert boundary.current(checksum=True)["layer"] == 0
    assert not boundary.directory(-1).exists()


def test_quadratic_metric_matches_direct_fp64_outputs():
    gen=torch.Generator().manual_seed(787)
    x=torch.randn(137, 128, generator=gen)
    w=torch.randn(256, 128, generator=gen)
    candidate=w+torch.randn(w.shape, generator=gen)*.025
    y=x.double()@w.double().T
    dy=x.double()@(candidate.double()-w.double()).T
    oracle=float(dy.square().sum()/y.square().sum())
    actual=weighted_nmse(w, candidate, x.T@x)
    assert abs(actual-oracle)/oracle < 1e-6
    assert weighted_nmse(w,w,x.T@x) == 0


def test_quadratic_metric_rejects_wrong_geometry_and_zero_energy():
    with pytest.raises(ValueError, match="geometry"):
        weighted_nmse(torch.ones(2,128),torch.ones(2,128),torch.eye(64))
    with pytest.raises(ValueError, match="zero-energy"):
        weighted_nmse(torch.zeros(2,128),torch.zeros(2,128),torch.eye(128))


def test_storage_accounts_replication_and_per_rank_scales():
    assert tp2_projection_bytes("self_attn.q_a_proj",(2048,6144),"bf16") == [25165824]*2
    shape=(16384,2048)
    full=6*shape[0]*shape[1]//8+2*sum(shape)+4
    half=tp2_projection_bytes("self_attn.q_b_proj",shape,"mcg-k6")[0]
    assert half*2 == full+2*shape[1]+4
    assert len(modules(1)) == len(modules(6)) == 11
    assert "model.layers.1.mlp.down_proj" in modules(1)
    assert "model.layers.6.mlp.shared_experts.down_proj" in modules(6)


def test_reader_returns_owned_buffers_and_closes_one_shard(tmp_path):
    values={"a":torch.arange(32).reshape(4,8).float(), "b":torch.ones(4,8)}
    save_file({"a":values["a"]},tmp_path/"one.safetensors")
    save_file({"b":values["b"]},tmp_path/"two.safetensors")
    (tmp_path/"config.json").write_text('{}')
    (tmp_path/"model.safetensors.index.json").write_text(json.dumps(dict(weight_map={"a":"one.safetensors","b":"two.safetensors"})))
    source=GlmBF16Source(tmp_path)
    owned=source.tensor("a")
    owned[0,0]=999
    assert source.tensor("a")[0,0] == 0
    assert torch.equal(source.tensor("b"), values["b"])
    assert source._shard == "two.safetensors"
    source.close()
    assert source._reader is source._shard is None
    assert owned[0,0] == 999


@pytest.mark.parametrize("layer",[1,3])
def test_native_loader_matches_dense_and_fused_expert_weights(tmp_path,layer):
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaDecoderLayer
    config=GlmMoeDsaConfig(hidden_size=128,intermediate_size=256,moe_intermediate_size=128,
        num_hidden_layers=4,num_attention_heads=2,num_key_value_heads=2,
        q_lora_rank=128,kv_lora_rank=128,qk_nope_head_dim=64,qk_rope_head_dim=64,
        v_head_dim=128,n_routed_experts=4,num_experts_per_tok=2,n_group=1,topk_group=1,
        first_k_dense_replace=3,index_n_heads=2,index_head_dim=128,index_topk=8,
        indexer_types=["full"]*4,rope_parameters=dict(rope_theta=8000000.,rope_type="default"))
    torch.manual_seed(787)
    native=GlmMoeDsaDecoderLayer(config,layer).to(torch.bfloat16)
    for p in native.parameters():
        p.data.normal_(0,.01)
    if layer == 3:
        native.mlp.gate.e_score_correction_bias=native.mlp.gate.e_score_correction_bias.float()
    native.self_attn.indexer.weights_proj=native.self_attn.indexer.weights_proj.float()
    state={}
    prefix=f"model.layers.{layer}."
    for name,value in native.state_dict().items():
        if name == "mlp.experts.gate_up_proj":
            for i in range(4):
                for part,tensor in zip(("gate_proj","up_proj"),value[i].chunk(2,0)):
                    state[f"{prefix}mlp.experts.{i}.{part}.weight"]=tensor.contiguous()
        elif name == "mlp.experts.down_proj":
            for i in range(4):
                state[f"{prefix}mlp.experts.{i}.down_proj.weight"]=value[i].contiguous()
        else:
            # Match checkpoint BF16 storage for the FP32 execution module.
            state[prefix+name]=value.bfloat16() if "indexer.weights_proj" in name else value
    save_file(state,tmp_path/"weights.safetensors")
    (tmp_path/"config.json").write_text(config.to_json_string())
    (tmp_path/"model.safetensors.index.json").write_text(json.dumps(dict(weight_map={k:"weights.safetensors" for k in state})))
    source=GlmBF16Source(tmp_path)
    loaded=source.load_block(layer,"cpu")
    for name,value in native.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name],value,rtol=0,atol=0)
    assert loaded.self_attn.indexer.weights_proj.weight.dtype == torch.float32
    assert source._reader is None

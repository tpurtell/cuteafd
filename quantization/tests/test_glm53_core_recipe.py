"""Header-only GLM inventory, recipe boundaries and padding accounting."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glm5.core_k6 import recipe


def config_fixture():
    return dict(model_type="glm_moe_dsa", num_hidden_layers=78, num_nextn_predict_layers=1,
        hidden_size=6144, vocab_size=154880, intermediate_size=12288, moe_intermediate_size=2048,
        first_k_dense_replace=3, n_routed_experts=256, n_shared_experts=1, num_experts_per_tok=8,
        num_attention_heads=64, q_lora_rank=2048, kv_lora_rank=512, qk_nope_head_dim=192,
        qk_rope_head_dim=64, v_head_dim=256, index_n_heads=32, index_head_dim=128,
        index_topk=2048, scoring_func="sigmoid", topk_method="noaux_tc", routed_scaling_factor=2.5,
        tie_word_embeddings=False, rope_interleave=True, indexer_rope_interleave=True,
        hidden_act="silu", n_group=1, topk_group=1, norm_topk_prob=True,
        moe_router_dtype="float32", rms_norm_eps=1e-5, dtype="bfloat16",
        rope_parameters=dict(rope_theta=8000000, rope_type="default"),
        indexer_types=["full" if n in (0, 1, 2, *range(6, 75, 4)) else "shared" for n in range(78)],
        mlp_layer_types=["dense"] * 3 + ["sparse"] * 75)


@pytest.fixture
def source(monkeypatch, tmp_path):
    headers, weight_map, offsets = {}, {}, {}
    for name, (dtype, shape) in recipe.reviewed_source_specs().items():
        shard = "layer-" + name.split(".")[2] if name.startswith("model.layers.") else "extras"
        h = headers.setdefault(shard, {})
        start = offsets.get(shard, 0)
        size = (4 if dtype == "F32" else 2)
        for n in shape:
            size *= n
        h[name] = dict(dtype=dtype, shape=shape, data_offsets=[start, start+size])
        offsets[shard] = start + size
        weight_map[name] = shard
    monkeypatch.setattr(recipe, "read_header", lambda path: (headers[path.name], 8))
    return SimpleNamespace(config=config_fixture(), snapshot=tmp_path, weight_map=weight_map, headers=headers)


def test_reviewed_tensor_scope_and_storage():
    core, experts = recipe.reviewed_core_modules(), recipe.reviewed_expert_modules()
    assert len(core) == 555 and len(experts) == 58368
    assert len(recipe.reviewed_core_modules(include_mtp=False)) == 547
    assert len(recipe.reviewed_expert_modules(include_mtp=False)) == 57600
    assert core["model.layers.1.mlp.down_proj"]["in_features"] == 12288
    assert core["model.layers.6.mlp.shared_experts.down_proj"]["out_features"] == 6144
    assert experts["model.layers.6.mlp.experts.255.down_proj"]["in_features"] == 2048
    assert all(g["bits"] == 4 for n, g in core.items() if n.startswith("model.layers.78."))
    assert core["lm_head"]["bits"] == 6
    assert "model.layers.6.self_attn.indexer.wq_b" not in core
    assert "model.layers.6.self_attn.kv_b_proj" not in core
    assert not set(core) & set(experts)
    g = core["model.layers.6.self_attn.kv_a_proj_with_mqa"]
    assert (g["out_features"], g["stored_out_features"]) == (576, 640)
    packed = recipe.packed_specs("kv_a", g)
    assert packed["kv_a.trellis"] == ("I16", [384, 40, 96])
    assert packed["kv_a.svh"] == ("F16", [640])
    assert recipe.packed_bytes(g) == 2_962_692
    assert recipe.packed_bytes(g, bits=4) == 1_979_652


def test_complete_inventory_and_exact_payload(source):
    x = recipe.inventory(source)
    assert x["source_tensors"] == 59585
    assert x["source_payload_bytes"] == 1_506_659_919_872
    assert x["selected_modules"] == 58923
    assert x["target_tensors"] == 236354
    assert x["target_payload_bytes"] == 385_362_552_236
    assert x["preserved_tensors"] == 662
    assert x["parts"]["base_core"]["source_bytes"] == 32_377_405_440
    assert sum(p["packed_bytes"] for p in x["parts"].values()) == x["selected_packed_bytes"]
    assert x["preserved_bytes"] + x["selected_packed_bytes"] == x["target_payload_bytes"]
    assert recipe.reviewed_source_specs()["model.layers.78.mlp.gate.e_score_correction_bias"] == ("F32", [256])


@pytest.mark.parametrize("damage", ["missing", "extra", "dtype", "shape", "extent", "orphan_header", "config", "indexer", "mtp"])
def test_inventory_fails_closed(source, damage):
    name = "model.layers.6.self_attn.kv_a_proj_with_mqa.weight"
    spec = source.headers[source.weight_map[name]][name]
    if damage == "missing":
        del source.weight_map[name]
    elif damage == "extra":
        source.weight_map["unexpected.weight"] = "extras"
    elif damage == "dtype":
        spec["dtype"] = "F16"
    elif damage == "shape":
        spec["shape"] = [640, 6144]
    elif damage == "extent":
        spec["data_offsets"][1] -= 2
    elif damage == "orphan_header":
        source.headers[source.weight_map[name]]["unexpected.weight"] = deepcopy(spec)
    elif damage == "config":
        source.config["num_nextn_predict_layers"] = 0
    elif damage == "indexer":
        source.config["indexer_types"][6] = "shared"
    else:
        del source.weight_map["model.layers.78.shared_head.norm.weight"]
    with pytest.raises(ValueError):
        recipe.inventory(source)


@pytest.mark.parametrize("bits", [0, 2, 3, 7, 16])
def test_unreviewed_bits_rejected(bits):
    with pytest.raises(ValueError):
        recipe.packed_specs("head", recipe.reviewed_core_modules()["lm_head"], bits=bits)

"""Independent official-vs-staged native boundaries, including exact KV reuse."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.mistral_large4.modeling_mistral_large4")
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "quantization"))
from mistral_large4.core_k6 import native
from mistral_large4.core_k6 import source


def tiny_config():
    from transformers.models.mistral_large4.configuration_mistral_large4 import MistralLarge4TextConfig
    config = MistralLarge4TextConfig(vocab_size=89, hidden_size=128, intermediate_size=256,
        moe_intermediate_size=128, num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=4,
        kv_lora_rank=32, qk_rope_head_dim=16, qk_nope_head_dim=16, v_head_dim=32, head_dim=16,
        n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1, n_group=1, topk_group=1,
        index_n_heads=4, index_head_dim=32, index_topk=8,
        mlp_layer_types=["dense"] + ["sparse"] * 7,
        indexer_types=["full", "full", "shared", "full", "full", "shared", "shared", "full"],
        kv_source_layers=[0, 1, 2, 3, 4, 5, 6, 3], no_rope_layers=[1, 1, 1, 0, 1, 1, 1, 0],
        layer_types=["indexed_attention"] * 8, final_logit_softcapping=30.0)
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    return config


def checkpoint(tmp_path, monkeypatch):
    from transformers.models.mistral_large4.modeling_mistral_large4 import MistralLarge4TextModel
    torch.manual_seed(417)
    config = tiny_config()
    model = MistralLarge4TextModel(config).eval()
    tensors = {}

    def put(name, value):
        if value.ndim == 2 and (".wq.weight" in name or ".experts." in name):
            rows, cols = value.shape
            scale = torch.full(((rows + 127) // 128, (cols + 127) // 128), 2 ** -8)
            expanded = scale.repeat_interleave(128, 0)[:rows].repeat_interleave(128, 1)[:, :cols]
            tensors[name] = (value.float() / expanded).clamp(-448, 448).to(torch.float8_e4m3fn)
            tensors[name.removesuffix("weight") + "qscale_weight"] = scale
        else:
            tensors[name] = value.detach().bfloat16()

    for index, block in enumerate(model.layers):
        params = dict(block.named_parameters())
        for parameter, names in source.layer_mapping(config, index, block).items():
            value = params[parameter].detach()
            if isinstance(names, str):
                put(names, value)
            elif parameter.endswith("gate_up_proj"):
                for expert, (gate, up) in enumerate(names):
                    put(gate, value[expert, :128])
                    put(up, value[expert, 128:])
            else:
                for expert, name in enumerate(names):
                    put(name, value[expert])
    tensors["tok_embeddings.weight"] = model.embed_tokens.weight.detach().bfloat16()
    tensors["norm.weight"] = model.norm.weight.detach().bfloat16()
    tensors["lerope.alpha"] = (torch.randn(model.rotary_emb.alpha.shape) * .1).bfloat16()
    tensors["output.weight"] = torch.randn((89, 128)).bfloat16()
    save_file(tensors, str(tmp_path / "consolidated-00001-of-00001.safetensors"))
    raw = dict(model_type="mistral_large4", text_config=config.to_dict(),
               quantization_config=dict(quant_method="fp8", weight_block_size=[128, 128], activation_scheme="dynamic"),
               transformers_weights="consolidated.safetensors.index.json")
    (tmp_path / "config.json").write_text(json.dumps(raw))
    (tmp_path / raw["transformers_weights"]).write_text(json.dumps(dict(weight_map={
        name: "consolidated-00001-of-00001.safetensors" for name in tensors})))
    monkeypatch.setattr(source, "validate_config", lambda _: None)
    return source.MistralSource(tmp_path), config


@pytest.mark.parametrize("shape", [(128, 256), (137, 270)])
def test_fp8_exact_scale_multiplication(shape):
    torch.manual_seed(10)
    rows, cols = shape
    weight = torch.randn(shape).to(torch.float8_e4m3fn)
    scale = torch.exp2(torch.randint(-12, 5, ((rows + 127) // 128, (cols + 127) // 128)).float())
    expanded = scale.repeat_interleave(128, 0)[:rows].repeat_interleave(128, 1)[:, :cols]
    expected = (weight.float() * expanded).bfloat16()
    assert torch.equal(source.dequantize_fp8(weight, scale), expected)


@pytest.mark.parametrize("bad", [0., -1., float("nan"), float("inf"), .3])
def test_fp8_rejects_invalid_source_scale(bad):
    with pytest.raises(ValueError, match="qscale"):
        source.dequantize_fp8(torch.ones((128, 128)).to(torch.float8_e4m3fn), torch.tensor([[bad]]))


def test_native_owned_reads_fp32_router_and_exact_kept_bytes(tmp_path, monkeypatch):
    checkpoint_source, _ = checkpoint(tmp_path, monkeypatch)
    name = "layers.0.attention.wq.weight"
    raw = checkpoint_source.tensor_raw(name)
    assert raw.dtype == torch.float8_e4m3fn
    dtype, shape = checkpoint_source.tensor_header(name)
    assert dtype == "F8_E4M3" and shape == [128, 128]
    expected = source.dequantize_fp8(raw, checkpoint_source.tensor_raw(name.removesuffix("weight") + "qscale_weight"))
    checkpoint_source.close()
    assert torch.equal(checkpoint_source.tensor(name), expected)
    first = checkpoint_source.load_block(1, "cpu", include_experts=False)
    assert first.mlp.gate.weight.dtype == torch.float32
    assert first.self_attn.indexer.weights_proj.weight.dtype == torch.float32
    assert first.mlp.experts.gate_up_proj.is_meta
    checkpoint_source.materialize_routed_experts(first, device="cpu")
    assert not first.mlp.experts.gate_up_proj.is_meta
    assert first.mlp.experts.gate_up_proj.shape == (4, 256, 128)
    assert source.linear_module(first, "layers.1.attention.wq") is first.self_attn.q_proj
    assert source.linear_module(first, "layers.1.shared_experts.w3") is first.mlp.shared_experts.up_proj
    preserved = checkpoint_source.tensor_raw("layers.1.attention.wkv_b.weight")
    clone = preserved.clone()
    checkpoint_source.close()
    assert torch.equal(preserved, clone)


def test_staged_matches_official_every_layer_and_exact_reuse(tmp_path, monkeypatch):
    checkpoint_source, config = checkpoint(tmp_path, monkeypatch)
    rotary = checkpoint_source.rotary("cpu")
    torch.manual_seed(999)
    expected = actual = torch.randn((1, 11, 128)).bfloat16()
    official_shared = {}
    expected_topk = None
    state = {}
    for index in range(8):
        block = checkpoint_source.load_block(index, "cpu", include_experts=index > 0)
        expected_context = native.prepare_context(expected, rotary, expected_topk, shared_kv_states=official_shared)
        with torch.inference_mode():
            expected, expected_topk = block(expected, past_key_values=None, use_cache=False, **expected_context)
            normalized = native.attention_inputs(block, actual)
            o_input, next_state = native.attention_capture(block, normalized, rotary, state)
            residual, mlp_input = native.attention_finish(block, actual, block.self_attn.o_proj(o_input))
            actual = (residual + block.mlp(mlp_input)) if index == 0 else native.complete_moe(residual, mlp_input, block.mlp)
        assert torch.equal(actual, expected), f"native staged boundary differs at layer {index}"
        assert torch.equal(next_state["prev_topk_indices"], expected_topk)
        if index >= 3:
            assert torch.equal(next_state["kv_nope_pass"], official_shared["nope"][0])
            assert torch.equal(next_state["kv_nope_rot"], official_shared["nope"][1])
            assert next_state["kv_nope_rot"].shape == (1, 1, 11, 0)
        if index == 7:
            assert block.self_attn.kv_a_proj_with_mqa is None
        state = next_state


def test_capture_removes_hooks_and_propagates_numerical_failures():
    linear = torch.nn.Linear(4, 4)
    before = len(linear._forward_pre_hooks)
    inputs = torch.randn((2, 4))
    assert torch.equal(native.capture_linear_input(linear, linear, inputs), inputs)
    assert len(linear._forward_pre_hooks) == before
    with pytest.raises(ValueError, match="numerical"):
        native.capture_linear_input(linear, lambda _: (_ for _ in ()).throw(ValueError("numerical failure")), inputs)
    assert len(linear._forward_pre_hooks) == before


def test_incomplete_kv_reuse_state_rejected():
    with pytest.raises(ValueError, match="incomplete"):
        native._shared_from_state(dict(kv_nope_pass=torch.ones(1)))

"""Small CPU witnesses against the pinned official Muse decoder."""
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if transformers.__version__ != "5.22.1":
    pytest.skip("Muse numerical witness requires official transformers 5.22.1", allow_module_level=True)

_path = Path(__file__).resolve().parents[1] / "muse_spark/core_k6/numerical.py"
_spec = importlib.util.spec_from_file_location("muse_numerical", _path)
numerical = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(numerical)


def block(layer):
    from transformers.models.muse_spark.configuration_muse_spark import MuseSparkTextConfig
    from transformers.models.muse_spark.modeling_muse_spark import MuseSparkTextDecoderLayer
    config = MuseSparkTextConfig(hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, vocab_size=256, num_local_experts=16,
        num_experts_per_tok=8, num_shared_experts=0, moe_hidden_size=128,
        moe_intermediate_size=128, layer_types=["sliding_attention", "full_attention"],
        layer_rope_theta=[500000.0, 0], mlp_layer_types=["sparse", "sparse"],
        sliding_window=8, eos_token_id=1, bos_token_id=0,
        rope_parameters={"rope_type": "default", "rope_theta": 500000.0})
    config._attn_implementation = "sdpa"
    config._experts_implementation = "eager"
    torch.manual_seed(23)
    value = MuseSparkTextDecoderLayer(config, layer)
    for name, parameter in list(value.named_parameters()) + list(value.named_buffers()):
        dtype = torch.float32 if ".gate." in name or name.endswith("residual_gate.gate") else torch.bfloat16
        tensor = torch.randn(parameter.shape, dtype=torch.float32).mul_(0.025).to(dtype)
        numerical._assign(value, name, tensor)
    value.layer_idx = layer
    return value.eval()


@pytest.mark.parametrize("layer", [0, 1])
def test_official_native_equals_staged(layer):
    numerical.source_identity()
    value = block(layer)
    hidden = torch.randn(1, 13, 128, dtype=torch.bfloat16)
    actual = numerical.native_forward(value, dict(hidden=hidden))
    expected = numerical.staged_forward(value, dict(hidden=hidden))
    assert torch.equal(actual["hidden"], expected["hidden"])
    assert torch.isfinite(actual["hidden"]).all()


@pytest.mark.parametrize("layer", [0, 1])
def test_official_o_operand_includes_output_norm_and_gate(layer):
    value = block(layer)
    hidden = torch.randn(1, 13, 128, dtype=torch.bfloat16)
    normalized = numerical.attention_inputs(value, hidden)
    context = numerical.prepare_context(value, normalized)
    operand = numerical.attention_output_input(value, normalized, context)
    result = numerical.attention_forward(value, normalized, context)
    assert torch.equal(value.self_attn.o_proj(operand), result)
    assert (context["position_embeddings"] is None) == (layer == 1)
    if layer == 0:
        mask = context["attention_mask"][0, 0]
        assert mask.dtype == torch.bool
        assert mask[12, 12]
        assert not mask[0, 1]
        assert not mask[12, 0]
    else:
        # Official full-attention SDPA omits an explicit mask and selects
        # the fused is_causal route.
        assert context["attention_mask"] is None


def test_fused_split_and_official_post_expert_norm():
    value = block(0)
    x = torch.randn(7, 128, dtype=torch.bfloat16)
    e = 3
    gate = numerical.expert_weight(value, e, "gate_proj")
    up = numerical.expert_weight(value, e, "up_proj")
    down = numerical.expert_weight(value, e, "down_proj")
    fused = value.mlp.experts.gate_up_proj[e]
    assert torch.equal(gate, fused[:128])
    assert torch.equal(up, fused[128:])
    observed = numerical.expert_down_input(value, gate, up, x)
    expected_gate, expected_up = torch.nn.functional.linear(x, fused).chunk(2, dim=-1)
    expected = value.mlp.experts.act_fn(expected_gate) * expected_up
    assert torch.equal(observed, expected)
    assert torch.equal(numerical.expert_output(value, observed, down),
        value.mlp.experts.post_expert_norm(torch.nn.functional.linear(expected, down)))


def test_router_observes_native_top8_and_disjoint_near_misses():
    value = block(0)
    hidden = torch.randn(1, 13, 128, dtype=torch.bfloat16)
    selected, weights, near = numerical.router_choices(value.mlp, hidden)
    _, official_weights, official_selected = value.mlp.gate(hidden)
    assert torch.equal(selected, official_selected)
    assert torch.equal(weights, official_weights)
    assert not (selected[:, :, None] == near[:, None, :]).any()
    assert weights.dtype == torch.float32


def test_reject_noncontiguous_prompt_positions():
    value = block(0)
    hidden = torch.randn(1, 13, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="positions"):
        numerical.prepare_context(value, hidden, position_ids=torch.arange(1, 14)[None])

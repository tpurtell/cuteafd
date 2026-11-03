"""ops.py against the official transformers modules on tiny random configs.

Needs transformers with ``models/mistral_large4`` (main >= 66f51e76); skips
otherwise. CPU only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "families" / "mistral_large4"))
import ops  # noqa: E402

ml4 = pytest.importorskip("transformers.models.mistral_large4.modeling_mistral_large4")
cfg_mod = pytest.importorskip("transformers.models.mistral_large4.configuration_mistral_large4")


def tiny(**overrides):
    layers = 8
    kwargs = dict(vocab_size=64, hidden_size=256, intermediate_size=256, moe_intermediate_size=128,
                  num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=4, n_routed_experts=8,
                  num_experts_per_tok=2, kv_lora_rank=512, qk_rope_head_dim=64, qk_nope_head_dim=64,
                  v_head_dim=128, index_n_heads=4, index_head_dim=128, index_topk=4,
                  kv_source_layers=[0, 1, 2, 3, 4, 5, 6, 3],
                  indexer_types=["full", "full", "shared", "full", "full", "shared", "shared", "full"],
                  max_position_embeddings=1 << 20)
    kwargs.update(overrides)
    config = cfg_mod.MistralLarge4TextConfig(**kwargs)
    config._attn_implementation = "eager"
    return config


def bf16(*shape, scale=1.0):
    return (torch.randn(*shape) * scale).bfloat16()


def test_lerope_inv_freq_matches_official_rotary():
    torch.manual_seed(0)
    config = tiny()
    rotary = ml4.MistralLarge4TextRotaryEmbedding(config)
    rotary.alpha.copy_(torch.randn(32) * 0.1)
    positions = torch.tensor([[0, 1, 7, 511, 512, 4095, 1 << 19]])
    cos_ref, sin_ref = rotary(torch.zeros(1, dtype=torch.bfloat16), positions)
    inv = ops.yarn_lerope_inv_freq(alpha=rotary.alpha)
    torch.testing.assert_close(inv, rotary.inv_freq * rotary.alpha.exp(), rtol=0, atol=0)
    cos, sin = ops.cos_sin(positions[0], inv)
    assert torch.equal(cos, cos_ref[0, :, :32]) and torch.equal(sin, sin_ref[0, :, :32])


@pytest.mark.parametrize("rope", [True, False])
def test_mla_query_matches_official_attention(rope):
    torch.manual_seed(1)
    config = tiny()
    layer = 2 if rope else 3
    attn = ml4.MistralLarge4TextAttention(config, layer).bfloat16()
    with torch.no_grad():
        for p in attn.parameters():
            p.copy_(torch.randn_like(p) * 0.05)
        attn.softmax_scaling.scale.copy_(torch.rand(4) * 0.2)
        attn.softmax_scaling.bias.copy_(1 + torch.rand(4) * 0.1)
    rows = 9
    x = bf16(1, rows, 256)
    positions = torch.arange(100, 100 + rows)
    rotary = ml4.MistralLarge4TextRotaryEmbedding(config)
    cos_full, sin_full = rotary(x, positions[None])
    q = attn.q_proj(x).view(1, rows, 4, 128).transpose(1, 2)
    nope = attn.qk_nope_head_dim
    q_pass, q_rot = torch.split(q, [nope, 128 - nope], dim=-1)
    if rope:
        q_rot, _ = ml4.apply_rotary_pos_emb_interleave(q_rot, None, cos_full, sin_full)
    expected = attn.softmax_scaling(torch.cat((q_pass, q_rot), -1), positions[None], head_dim_first=True)
    cos, sin = ops.cos_sin(positions, ops.yarn_lerope_inv_freq())
    actual = ops.mla_query(attn.q_proj(x)[0].view(rows, 4, 128), positions, cos, sin,
                           attn.softmax_scaling.scale, attn.softmax_scaling.bias, 64 if rope else 128, rope)
    assert torch.equal(actual, expected[0].transpose(0, 1))


@pytest.mark.parametrize("layer", [0, 3])
def test_indexer_selection_matches_official(layer):
    torch.manual_seed(2)
    config = tiny(index_topk=6)
    indexer = ml4.MistralLarge4TextIndexer(config, layer).bfloat16()
    with torch.no_grad():
        for p in indexer.parameters():
            p.copy_(torch.randn_like(p) * 0.1)
        indexer.weights_proj.float()
    rows = 12
    x = bf16(1, rows, 256)
    positions = torch.arange(rows)
    rotary = ml4.MistralLarge4TextRotaryEmbedding(config)
    pe = rotary(x, positions[None])
    mask = torch.full((1, rows, rows), float("-inf")).triu(1)
    expected = indexer(x, pe, mask, positions[None])[0]
    cos, sin = ops.cos_sin(positions, ops.yarn_lerope_inv_freq())
    scores = ops.indexer_scores(x[0], positions, cos, sin, indexer.q_input_norm.weight, indexer.q_input_norm.bias,
                                indexer.wq_b.weight, indexer.wk.weight, indexer.k_norm.weight, indexer.k_norm.bias,
                                indexer.weights_proj.weight, indexer.softmax_scaling.scale,
                                indexer.softmax_scaling.bias, heads=4, rope=bool(config.no_rope_layers[layer]))
    actual = scores.topk(6, dim=-1).indices
    assert torch.equal(actual.sort(-1).values, expected.long().sort(-1).values)
    # The engine's stable rule may pick a different key among exact relu-0
    # ties at the cutoff (the official topk order is undefined there), never a
    # different score: the selected score multisets agree.
    stable = ops.index_topk(scores, 6)
    assert torch.equal(scores.gather(-1, stable).sort(-1).values, scores.gather(-1, actual).sort(-1).values)


def test_index_topk_breaks_relu_ties_toward_lower_ids():
    scores = torch.tensor([[0.0, 2.0, 0.0, 0.0, 1.0, 0.0]])
    assert ops.index_topk(scores, 4).tolist() == [[0, 1, 2, 4]]
    assert ops.index_topk(scores[:, :2], 4).tolist() == [[0, 1, 2, 3]]


def test_router_matches_official():
    torch.manual_seed(3)
    config = tiny()
    gate = ml4.MistralLarge4TextTopkRouter(config)
    with torch.no_grad():
        gate.weight.copy_(torch.randn_like(gate.weight))
    x = bf16(5, 256)
    _, w_ref, i_ref = gate(x)
    w, i = ops.router(x, gate.weight, topk=2)
    order_ref, order = i_ref.sort(-1), i.sort(-1)
    assert torch.equal(order.values, order_ref.values)
    assert torch.equal(w.gather(1, order.indices), w_ref.gather(1, order_ref.indices))


def test_softcap_and_rms_norm():
    x = bf16(3, 16, scale=50)
    assert torch.equal(ops.softcap(x.float()), torch.tanh(x.float() / 30) * 30)
    norm = ml4.MistralLarge4RMSNorm(16, eps=1e-5).bfloat16()
    with torch.no_grad():
        norm.weight.copy_(torch.randn(16))
    assert torch.equal(ops.rms_norm(x, norm.weight, 1e-5), norm(x))

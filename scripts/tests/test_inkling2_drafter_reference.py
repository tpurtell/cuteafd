"""Bundled draft math tests; intentionally independent of the serving runtime."""
import json
from pathlib import Path
import sys

import pytest

pytestmark = pytest.mark.container_torch
torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/reference"))
from families.inkling2 import drafter


def release_config():
    return json.loads((ROOT / "rust/crates/cuteafd-loader/src/families/inkling2/fixtures/config.json").read_text())


def tiny_drafter():
    raw = release_config()
    raw["text_config"].update(hidden_size=32, vocab_size=40, unpadded_vocab_size=34, num_hidden_layers=4)
    raw["mtp_config"].update(hidden_size=32, num_hidden_layers=3, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=8, intermediate_size=64,
                             local_layer_ids=[0, 2], sliding_window_size=4,
                             target_layer_ids=[1, 2, 4], markov_rank=8)
    c = drafter.Config.from_dict(raw)
    generator = torch.Generator().manual_seed(1093)

    def normal(*shape):
        return (torch.randn(*shape, generator=generator)*.12).bfloat16()

    def gamma(width):
        return (1+torch.randn(width, generator=generator)*.1).bfloat16()

    weights = {"fuse_weight": normal(c.layers+1, len(c.taps)), "mask_embed": normal(c.hidden),
               "norm.weight": gamma(c.hidden), "markov_head.dflare_norm.weight": gamma(c.hidden),
               "markov_head.markov_w1.weight": normal(c.padded_vocab, c.rank),
               "markov_head.rnn_proj.weight": normal(3*c.rank, 2*c.rank+2*c.hidden),
               "markov_head.rnn_proj.bias": normal(3*c.rank),
               "markov_head.hidden_out.weight": normal(c.hidden, c.rank),
               "markov_head.hidden_out.bias": normal(c.hidden)}
    for layer in range(c.layers):
        p, a = f"layers.{layer}", f"layers.{layer}.self_attn"
        for name in ("input_layernorm", "post_attention_layernorm", "dflare_norm"):
            weights[f"{p}.{name}.weight"] = gamma(c.hidden)
        for name in ("q_norm", "k_norm", "k_norm_ctx"):
            weights[f"{a}.{name}.weight"] = gamma(c.head_dim)
        for name, rows in (("q_proj", c.heads*c.head_dim), ("k_proj", c.kv_heads*c.head_dim),
                           ("v_proj", c.kv_heads*c.head_dim), ("k_proj_ctx", c.kv_heads*c.head_dim),
                           ("v_proj_ctx", c.kv_heads*c.head_dim)):
            weights[f"{a}.{name}.weight"] = normal(rows, c.hidden)
        weights[f"{a}.o_proj.weight"] = normal(c.hidden, c.heads*c.head_dim)
        weights[f"{p}.mlp.gate_proj.weight"] = normal(c.intermediate, c.hidden)
        weights[f"{p}.mlp.up_proj.weight"] = normal(c.intermediate, c.hidden)
        weights[f"{p}.mlp.down_proj.weight"] = normal(c.hidden, c.intermediate)
    model = drafter.Drafter(c, weights, normal(c.padded_vocab, c.hidden), gamma(c.hidden),
                            normal(c.padded_vocab, c.hidden))
    return model, normal(13, len(c.taps), c.hidden)


def test_release_geometry_is_fill_in_seven_prediction_layout():
    c = drafter.Config.from_dict(release_config())
    assert (c.block, c.rank, c.hidden) == (8, 256, 4608)
    assert (c.padded_vocab, c.valid_vocab) == (201024, 200058)
    assert 2*c.rank+2*c.hidden == 9728
    assert c.taps == (5, 11, 17, 23, 29, 35, 41, 48)
    assert c.local == (0, 1, 3, 4)


def test_mix_norms_unrounded_fp32_sum():
    torch.manual_seed(19)
    features = torch.randn(9, 8, 4608).bfloat16()
    alpha = torch.randn(8).softmax(0).bfloat16()
    gamma = (torch.randn(4608)*.1+1).bfloat16()
    mixed = torch.zeros_like(features[:, 0], dtype=torch.float32)
    for tap in range(8):
        mixed += features[:, tap].float()*alpha[tap].float()
    expected = (mixed*torch.rsqrt(mixed.square().mean(-1, keepdim=True)+1e-6)*gamma.float()).bfloat16()
    assert torch.equal(drafter.mix(features, alpha, gamma, 1e-6), expected)
    assert not torch.equal(drafter.norm(mixed.bfloat16(), gamma, 1e-6), expected)


def test_mask_init_preserves_anchor_and_learned_masks_byte_exactly():
    anchors = torch.arange(3*32).reshape(3, 32).bfloat16()
    mask = -torch.arange(32).bfloat16()
    out = drafter.init_block(anchors, mask, 8)
    assert torch.equal(out[:, 0], anchors)
    assert torch.equal(out[:, 1:], mask.expand(3, 7, 32))
    assert out.data_ptr() != anchors.data_ptr()


def test_recurrent_gate_uses_previous_state_and_independent_output_latent():
    torch.manual_seed(23)
    raw = torch.randn(13, 768)*4
    state = torch.randn(13, 256).bfloat16()
    gate, candidate, output = raw.bfloat16().chunk(3, -1)
    expected = gate.sigmoid()*state + (1-gate.sigmoid())*candidate.tanh()
    new, latent = drafter.recurrent_gate(raw, state)
    assert torch.equal(new, expected)
    assert torch.equal(latent, output.tanh())
    changed, same_latent = drafter.recurrent_gate(raw, torch.zeros_like(state))
    assert not torch.equal(changed, new)
    assert torch.equal(same_latent, latent)


def test_fold_keeps_hidden_out_bias_and_single_folded_projection():
    model, _ = tiny_drafter()
    w = model.weights
    assert torch.equal(model.fold_weight,
                       (model.lm_head.double() @ w["markov_head.hidden_out.weight"].double()).bfloat16())
    assert torch.equal(model.fold_bias,
                       (model.lm_head.double() @ w["markov_head.hidden_out.bias"].double()).bfloat16())
    assert bool(model.fold_bias.abs().sum())


def test_block_attention_uses_anchor_window_and_all_draft_positions():
    q = torch.zeros(8, 4, 8, dtype=torch.bfloat16)
    key = torch.zeros(13, 2, 8, dtype=torch.bfloat16)
    value = torch.arange(13).view(13, 1, 1).expand_as(key).bfloat16()
    dk = torch.zeros(8, 2, 8, dtype=torch.bfloat16)
    dv = torch.arange(13, 21).view(8, 1, 1).expand_as(dk).bfloat16()
    local = drafter.attention(q, key, value, dk, dv, local_window=4)
    full = drafter.attention(q, key, value, dk, dv)
    assert torch.equal(local, torch.full_like(q, 14.5))
    assert torch.equal(full, torch.full_like(q, 10.0))
    assert torch.equal(local[0], local[-1])


def test_tiny_full_forward_tp1_tp2_hidden_logits_and_seven_predictions():
    model, features = tiny_drafter()
    whole = model.forward(3, features)
    split = model.forward(3, features, ranks=2)
    assert whole["hidden"].shape == (8, 32)
    assert whole["logits"].shape == (7, 34)
    assert whole["states"].shape == (7, 8)
    assert torch.equal(whole["tokens"], split["tokens"])
    torch.testing.assert_close(whole["hidden"], split["hidden"], atol=.015625, rtol=.02)
    torch.testing.assert_close(whole["logits"], split["logits"], atol=.015625, rtol=.02)
    assert bool((whole["tokens"] < model.config.valid_vocab).all())
    assert all(k.shape == (13, 2, 8) and v.shape == (13, 2, 8) for k, v in whole["contexts"])


def test_context_projection_is_independent_of_draft_projection():
    model, features = tiny_drafter()
    before = model.context(features)
    model.weights["layers.0.self_attn.k_proj.weight"].zero_()
    after = model.context(features)
    assert torch.equal(before[0][0], after[0][0])
    assert bool(before[0][0].abs().sum())


def test_forward_refuses_missing_last_committed_features_and_invalid_tp():
    model, features = tiny_drafter()
    with pytest.raises(ValueError, match="last committed"):
        model.forward(3, features[:0])
    with pytest.raises(ValueError, match="TP split"):
        model.forward(3, features, ranks=3)


def test_mini_dflash_normalizes_anchor_and_mask_embeddings():
    from families.glm5.dflash2.reference import Drafter
    model = Drafter.__new__(Drafter)
    model.inkling, model.block, model.mask, model.device = True, 8, 1, "cpu"
    model.embed = torch.tensor([[1., 2., 3., 4.], [2., 3., 4., 5.]], dtype=torch.bfloat16)
    model.embed_norm = torch.tensor([1., 2., 1., 3.], dtype=torch.bfloat16)
    model.embed_eps = 1e-6
    raw = model.embed[torch.tensor([0] + [1]*7)]
    expected = (raw.float() * torch.rsqrt(raw.float().square().mean(-1, keepdim=True) + model.embed_eps)
                * model.embed_norm.float()).bfloat16()
    assert torch.equal(model.block_embeddings(0), expected)
    assert not torch.equal(expected, raw)

"""Mistral Large 4 mtp_dspark reference vs vLLM bd4fa263's own definitions
(tests/models/mistral/test_ml4_dspark.py, Apache-2.0: the trainer RNN head and
the training window mask), at a tiny geometry with random weights."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from families.mistral_large4 import dspark  # noqa: E402

H, HEADS, NOPE, ROPE, LATENT, V, INTER, VOCAB, RANK, DEPTH, LAYERS = 32, 2, 8, 8, 16, 8, 48, 50, 4, 3, 2


def _weights(seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: (torch.randn(*s, generator=g) * 0.2).bfloat16()  # noqa: E731
    w = {"embed": r(VOCAB, H), "head": r(VOCAB, H), "backbone.mask_embedding": r(DEPTH - 1, H),
         "backbone.target_projection.weight": r(H, 5 * H), "backbone.target_norm.weight": r(H) + 1,
         "backbone.final_norm.weight": r(H) + 1,
         "autoregressive_head.embedding.weight": r(VOCAB, RANK),
         "autoregressive_head.joint_proj.weight": r(3 * RANK, H + 2 * RANK),
         "autoregressive_head.joint_proj.bias": r(3 * RANK),
         "autoregressive_head.output.weight": r(VOCAB, RANK)}
    for i in range(LAYERS):
        a, f = f"backbone.layers.{i}.attention.", f"backbone.layers.{i}.feed_forward."
        w.update({a + "wq.weight": r(HEADS * (NOPE + ROPE), H), a + "wkv_a_with_mqa.weight": r(LATENT + ROPE, H),
                  a + "kv_a_norm.weight": r(LATENT) + 1, a + "wkv_b.weight": r(HEADS * (NOPE + V), LATENT),
                  a + "wo.weight": r(H, HEADS * V), f + "w1.weight": r(INTER, H), f + "w3.weight": r(INTER, H),
                  f + "w2.weight": r(H, INTER), f"backbone.layers.{i}.attention_norm.weight": r(H) + 1,
                  f"backbone.layers.{i}.ffn_norm.weight": r(H) + 1})
    return w


def _drafter(window=6, seed=0):
    w = _weights(seed)
    inv = 1.0 / (1e4 ** (torch.arange(0, ROPE, 2).float() / ROPE))
    return dspark.Drafter(w.__getitem__, layers=LAYERS, depth=DEPTH, window=window, heads=HEADS, nope=NOPE,
                          rope=ROPE, latent=LATENT, v_head=V, inv_freq=inv), w


def _trainer_rnn_logits(base_logits, predecessor_ids, hidden, embedding, joint_proj, output):
    """vLLM's mistral.model.mtp.dspark.autoregressive_head.RNNHead, unrolled over D."""
    state, logits = None, []
    for d in range(base_logits.shape[1]):
        emb = embedding(predecessor_ids[:, d])
        if state is None:
            state = torch.zeros_like(emb)
        joint = joint_proj(torch.cat([state, emb, hidden[:, d]], dim=-1))
        gate_raw, candidate_raw, output_raw = joint.chunk(3, dim=-1)
        gate = torch.sigmoid(gate_raw)
        state = gate * state + (1.0 - gate) * torch.tanh(candidate_raw)
        logits.append(base_logits[:, d] + output(torch.tanh(output_raw)))
    return torch.stack(logits, dim=1)


def test_rnn_head_drafts_equal_the_trainer_head_fed_its_own_picks():
    drafter, w = _drafter()
    hidden = (torch.randn(DEPTH, H) * 2).bfloat16()
    anchor = 7
    drafts = drafter.heads_(anchor, hidden)
    embedding = lambda ids: w["autoregressive_head.embedding.weight"][ids]  # noqa: E731
    joint = lambda x: F.linear(x, w["autoregressive_head.joint_proj.weight"], w["autoregressive_head.joint_proj.bias"])  # noqa: E731
    output = lambda x: F.linear(x, w["autoregressive_head.output.weight"]).float()  # noqa: E731
    base = (torch.tanh(F.linear(hidden, w["head"]).float() / 30) * 30)[None]
    prev = torch.tensor([[anchor, *drafts[:-1]]])
    logits = _trainer_rnn_logits(base, prev, hidden[None], embedding, joint, output)
    assert logits[0].argmax(-1).tolist() == drafts


def test_block_uses_position_specific_masks_and_is_noncausal():
    drafter, w = _drafter()
    taps = torch.randn(10, 5 * H).bfloat16()
    ctx = drafter.context(taps, torch.arange(10))
    h = drafter.body(3, 10, drafter.window_of(ctx, 10))
    assert h.shape == (DEPTH, H)
    # Non-causal: the last mask row changes the anchor row's output.
    w2 = dict(w)
    w2["backbone.mask_embedding"] = w["backbone.mask_embedding"].clone()
    w2["backbone.mask_embedding"][-1] += 1
    other = dspark.Drafter(w2.__getitem__, layers=LAYERS, depth=DEPTH, window=6, heads=HEADS, nope=NOPE,
                           rope=ROPE, latent=LATENT, v_head=V, inv_freq=drafter.inv_freq)
    assert not torch.equal(other.body(3, 10, other.window_of(ctx, 10))[0], h[0])


def test_window_matches_the_training_mask():
    """vLLM's trainer mask: a block anchored after source sees context t with
    source - t < window and t <= source, plus its whole block."""
    drafter, _ = _drafter(window=6)
    ctx = [(torch.arange(20)[:, None].float(), torch.arange(20)[:, None].float())]
    for position in range(1, 20):
        source = position - 1
        seen = drafter.window_of(ctx, position)[0][0][:, 0].long().tolist()
        assert seen == [t for t in range(source + 1) if source - t < 6], position
    # Context rows outside the window do not move the block.
    taps = torch.randn(16, 5 * H).bfloat16()
    full = drafter.context(taps, torch.arange(16))
    clipped = [(l[10:], k[10:]) for l, k in full]
    a = drafter.body(4, 16, drafter.window_of(full, 16))
    b = drafter.body(4, 16, [(l[0:6], k[0:6]) for l, k in clipped])
    assert torch.equal(a, b)

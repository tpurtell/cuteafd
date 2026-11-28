"""CPU numeric oracle tests for QMLA long-context evaluation, dense equivalence, compact masks, and buckets."""
from __future__ import annotations

from functools import partial
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "quantization"))
sys.path.insert(0, str(ROOT / "scripts/lib"))

from hy4_flash.forward import Layer, Model, reference
from hy4_flash.qmla_capture import Hy4FlashCaptureAdapter
from hy4_flash.qmla_run import (
    attach_long_context_eval,
    build_parser,
    calibrate_hy4_flash_from_banks,
    run_hy4_flash_long_eval,
)
import panel_corpus as pc
from qmla import (
    DEFAULT_POSITION_BUCKETS,
    LONG_EVAL_SCHEMA,
    CapturedLayerParams,
    CapturedSequence,
    FitConfig,
    LongContextEvaluator,
    LongContextRecord,
    MlaLayerGeometry,
    apply_content_forward,
    apply_rope_pair_transform,
    causal_mask,
    compensate_kv_b,
    eval_attention_output,
    fit_content_layer,
    fit_rope_layer,
    load_long_context_records,
    load_sidecar_transforms,
    position_bucket_masks,
    qdq_asymmetric_group,
    rms_norm,
    sylvester_hadamard,
)
from qmla.capture import linear
from qmla.measurement import layer_candidates


def _dense_allowed_mask(seq: CapturedSequence, geometry: MlaLayerGeometry) -> torch.Tensor:
    if seq.allowed_mask is not None:
        return seq.allowed_mask
    t = seq.seq_len
    base = causal_mask(t, device=seq.u.device, sliding_window=geometry.sliding_window)
    if seq.sparse_selection is not None:
        qid = torch.arange(t, device=seq.u.device, dtype=torch.int64)
        base = base & seq.sparse_selection.tile_mask(0, t, 0, t, qid, seq_len=t)
    return base


def _eval_attention_output_dense_oracle(
    seq: CapturedSequence,
    params: CapturedLayerParams,
    geometry: MlaLayerGeometry,
    *,
    content_fn=None,
    kv_b_override=None,
    rope_query_fn=None,
    rope_key_fn=None,
    exact_content_fn=None,
    exact_rope_key_fn=None,
    sink_tokens: int = 0,
    recent_tokens: int = 0,
) -> torch.Tensor:
    """Reference dense [H, T, T] FP32 attention oracle for short-sequence equivalence checks."""
    protect = geometry.kind != "swa" and (sink_tokens > 0 or recent_tokens > 0)
    t, h, nope, vdim = seq.seq_len, geometry.heads, geometry.qk_nope_dim, geometry.v_head_dim
    dev = seq.u.device

    c = seq.u.float() * params.kv_norm_weight.float()
    latent = content_fn(c) if content_fn is not None else c
    kv_b = (params.kv_b_weight if kv_b_override is None else kv_b_override).float()
    expanded = linear(latent, kv_b).reshape(t, h, nope + vdim)
    keys, values = expanded.split([nope, vdim], dim=-1)
    exact_keys = exact_values = None
    if protect and exact_content_fn is not None:
        exact_kv = linear(exact_content_fn(c), kv_b).reshape_as(expanded)
        exact_keys, exact_values = exact_kv.split([nope, vdim], dim=-1)
    elif protect and exact_rope_key_fn is not None:
        exact_keys = keys

    query = seq.q_nope.float()
    if geometry.has_rope:
        qp = rope_query_fn(seq.q_rope.float()) if rope_query_fn is not None else seq.q_rope.float()
        kp = rope_key_fn(seq.k_rope.float()) if rope_key_fn is not None else seq.k_rope.float()
        query = torch.cat((query, qp), dim=-1)
        keys = torch.cat((keys, kp[:, None, :].expand(-1, h, -1)), dim=-1)
        if exact_keys is not None:
            ekp = exact_rope_key_fn(seq.k_rope.float()) if exact_rope_key_fn is not None else kp
            exact_keys = torch.cat((exact_keys, ekp[:, None, :].expand(-1, h, -1)), dim=-1)

    q_ht, k_ht, v_ht = query.transpose(0, 1), keys.transpose(0, 1), values.transpose(0, 1)
    scores = torch.matmul(q_ht, k_ht.transpose(-1, -2))
    idx = torch.arange(t, device=dev)
    protected = (idx[None, :] < sink_tokens) | ((idx[:, None] - idx[None, :]) < recent_tokens) if protect else None
    if exact_keys is not None and protected is not None:
        scores = torch.where(protected.unsqueeze(0), torch.matmul(q_ht, exact_keys.transpose(0, 1).transpose(-1, -2)), scores)

    allowed = _dense_allowed_mask(seq, geometry)
    scores = (scores * geometry.effective_scale).masked_fill(~allowed.unsqueeze(0), torch.finfo(scores.dtype).min)
    if params.sink is not None:
        scores = torch.cat((scores, params.sink.float()[:, None, None].expand(-1, t, -1)), dim=-1)
    probs = (scores - scores.max(dim=-1, keepdim=True).values).softmax(dim=-1)
    if params.sink is not None:
        probs = probs[..., :-1]

    if exact_values is not None and protected is not None:
        ev_ht = exact_values.transpose(0, 1)
        attended = torch.matmul(probs.masked_fill(protected.unsqueeze(0), 0.0), v_ht) + torch.matmul(
            probs.masked_fill(~protected.unsqueeze(0), 0.0), ev_ht
        )
    else:
        attended = torch.matmul(probs, v_ht)

    attended_t = attended.transpose(0, 1)
    if seq.gate is not None:
        attended_t = attended_t * seq.gate.float().reshape_as(attended_t)
    out = linear(attended_t.reshape(t, h * vdim), params.o_proj_weight.float())
    if params.post_proj_norm_weight is not None:
        out = rms_norm(out, params.post_proj_norm_weight.float(), geometry.rms_norm_eps)
    return out


def _rel_l2(pred: torch.Tensor, ref: torch.Tensor) -> float:
    return float((pred.float() - ref.float()).norm().item() / float(ref.float().norm().item()))


def _synthetic_sparse_bank(seed: int, *, seq_len: int = 261, candidate_topk: int = 64, d_c: int = 64, d_r: int = 64):
    """Seeded synthetic Hy4-flash sparse layer with real DSA pruning past candidate_topk."""
    g = torch.Generator().manual_seed(seed)
    rand = lambda *shape: torch.randn(*shape, generator=g)

    heads, nope, vdim, q_rank, hidden, ratio = 2, 8, 8, 12, 16, 4
    config = SimpleNamespace(
        attention=lambda _idx: (heads, nope, d_r, vdim),
        layer_types=[reference.SPARSE],
        qk_rope_head_dim=d_r, kv_lora_rank=d_c, rms_norm_eps=1e-5, sliding_window=8,
        rope_parameters={reference.SPARSE: {"rope_theta": 10_000_000, "mrope_section": [12, 10, 10]}},
        compressed_indexer_compress_ratio=ratio,
        compressed_indexer_group_topk=candidate_topk // ratio,
        compressed_indexer_candidate_topk=candidate_topk,
        compressed_indexer_num_attention_heads=2,
        compressed_indexer_head_dim=d_r,
    )
    kv_a = rand(d_c + d_r, hidden)
    kv_a[0] *= 14.0
    kv_a[d_c] *= 12.0
    weights = {
        "self_attn.q_a_proj.weight": rand(q_rank, hidden),
        "self_attn.q_a_layernorm.weight": torch.ones(q_rank),
        "self_attn.q_b_proj.weight": rand(heads * (nope + d_r), q_rank),
        "self_attn.kv_a_proj_with_mqa.weight": kv_a,
        "self_attn.kv_a_layernorm.weight": 1 + 0.1 * rand(d_c),
        "self_attn.kv_b_proj.weight": rand(heads * (nope + vdim), d_c),
        "self_attn.linear_gate.weight": rand(heads * vdim, hidden),
        "self_attn.o_proj.weight": rand(hidden, heads * vdim),
        "self_attn.post_proj_norm.weight": torch.ones(hidden),
        "self_attn.learnable_sink_param": rand(heads),
        "self_attn.compressed_indexer.token_key_projection.weight": rand(d_r, hidden),
        "self_attn.compressed_indexer.token_key_norm.weight": torch.ones(d_r),
        "self_attn.compressed_indexer.query_projection.weight": rand(2 * d_r, q_rank),
        "self_attn.compressed_indexer.key_value_projection.weight": rand(d_r, d_r),
        "self_attn.compressed_indexer.key_gate_projection.weight": rand(d_r, d_r),
        "self_attn.compressed_indexer.key_norm.weight": torch.ones(d_r),
        "self_attn.compressed_indexer.head_weight_projection.weight": rand(2, hidden),
        "self_attn.compressed_indexer.intra_group_ape": rand(ratio, d_r),
    }
    model = Model(SimpleNamespace(config=config))
    layer = Layer(0, "model.layers.0.", weights)
    adapter = Hy4FlashCaptureAdapter(model)
    inputs = [(f"seq_{i:02d}", torch.randn(seq_len + (i % 5), hidden, generator=g), None) for i in range(6)]
    prov = {
        "dataset_repo": pc.V3_PANEL.repo_id, "dataset_revision": pc.V3_PANEL.revision,
        "components": {
            "agentic": {"sequences": 4, "tokens": 600, "input_ids_sha256": "1" * 64},
            "exl3_standard": {"sequences": 2, "tokens": 300, "input_ids_sha256": "2" * 64},
            "long_context": {"sequences": 2, "tokens": 4096, "input_ids_sha256": "3" * 64},
        },
        "input_ids_sha256": "4" * 64,
    }
    bank = adapter.capture_layer(layer, inputs, train_count=4, val_count=1, test_count=1, seed=seed, provenance=prov)
    return layer, adapter, bank, inputs


def test_tiled_and_compact_mask_match_dense_oracle_all_arms_and_tile_edges():
    """Equivalence gate (<1e-6): tiled eval + compact SparseSelection vs dense [H,T,T] for every arm & tile edges."""
    layer, adapter, bank, raw_inputs = _synthetic_sparse_bank(503, seq_len=263, candidate_topk=64)
    geom, params = bank.geometry, bank.params
    sid, x_long, _ = raw_inputs[-1]

    seq_dense, out_dense, _ = adapter.capture_sequence(x_long, layer, seq_id=sid, compact_mask=False)
    seq_compact, out_compact, _ = adapter.capture_sequence(x_long, layer, seq_id=sid, compact_mask=True)
    assert seq_dense.allowed_mask is not None and seq_dense.sparse_selection is None
    assert seq_compact.allowed_mask is None and seq_compact.sparse_selection is not None
    assert not bool(seq_dense.allowed_mask[-1].all())
    assert torch.equal(_dense_allowed_mask(seq_compact, geom), seq_dense.allowed_mask)
    assert _rel_l2(out_compact, out_dense) < 1e-6

    cfg = FitConfig(steps=6, val_every=3, seed=0, protection="sink4_local128", sink_tokens=4, recent_tokens=128)
    fc, fr = fit_content_layer(bank, 4, cfg), fit_rope_layer(bank, 4, cfg)
    cands = layer_candidates(bank, {4: fc}, {4: fr}, cfg)
    sq_c, sq_r = cands.content["smoothquant:c4"], cands.rope["smoothquant:r4"]
    had, ones = sylvester_hadamard(geom.kv_lora_rank), torch.ones(geom.kv_lora_rank)

    qdq = lambda fn, bits, dt: (lambda x: qdq_asymmetric_group(fn(x), bits, scale_dtype=dt, ste=False))
    c_hooks = lambda r, s: {
        "content_fn": qdq(partial(apply_content_forward, r_c=r, s_c=s), 4, "BF16"),
        "kv_b_override": compensate_kv_b(params.kv_b_weight, r, s),
        "exact_content_fn": partial(apply_content_forward, r_c=r, s_c=s),
    }
    r_hooks = lambda th, sp: {
        "rope_query_fn": partial(apply_rope_pair_transform, theta=th, scale=sp, rope_kind=geom.rope_kind, inverse_scale=True),
        "rope_key_fn": qdq(partial(apply_rope_pair_transform, theta=th, scale=sp, rope_kind=geom.rope_kind), 4, "F32"),
        "exact_rope_key_fn": partial(apply_rope_pair_transform, theta=th, scale=sp, rope_kind=geom.rope_kind),
    }
    native_c = lambda c: c
    arms = {
        "teacher": {},
        "rtn": {"content_fn": qdq(native_c, 4, "BF16"), "exact_content_fn": native_c,
                "rope_key_fn": qdq(lambda k: k, 4, "F32"), "exact_rope_key_fn": lambda k: k},
        "hadamard": {**c_hooks(had, ones), "rope_key_fn": qdq(lambda k: k, 4, "F32"), "exact_rope_key_fn": lambda k: k},
        "smoothquant": {**c_hooks(sq_c.r_c, sq_c.s_c), **r_hooks(sq_r.theta, sq_r.s_p)},
        "fitted": {**c_hooks(fc.r_c, fc.s_c), **r_hooks(fr.theta, fr.s_p)},
    }

    for prot in ({}, {"sink_tokens": 4, "recent_tokens": 128}):
        for name, base_hooks in arms.items():
            hooks = {**base_hooks, **prot}
            if not prot:
                hooks.pop("exact_content_fn", None)
                hooks.pop("exact_rope_key_fn", None)
            dense_out = _eval_attention_output_dense_oracle(seq_dense, params, geom, **hooks)
            assert _rel_l2(eval_attention_output(seq_dense, params, geom, **hooks), dense_out) < 1e-6, name
            assert _rel_l2(eval_attention_output(seq_compact, params, geom, **hooks), dense_out) < 1e-6, name

    # Verify per-query protection right across 128-row tile edges and sink+recent cutoff.
    pred_edge = eval_attention_output(seq_compact, params, geom, **arms["fitted"], sink_tokens=4, recent_tokens=128)
    dense_edge = _eval_attention_output_dense_oracle(seq_compact, params, geom, **arms["fitted"], sink_tokens=4, recent_tokens=128)
    for row in (0, 3, 4, 127, 128, 129, 131, 132, 133, 255, 256, seq_compact.seq_len - 1):
        assert _rel_l2(pred_edge[row], dense_edge[row]) < 1e-6, f"edge row {row}"

    # Structural-zero DSA check: changing kv_a_proj_with_mqa / kv_a_layernorm cannot alter Hy4-flash selection.
    mutated = dict(layer.weights)
    mutated["self_attn.kv_a_proj_with_mqa.weight"] = torch.zeros_like(mutated["self_attn.kv_a_proj_with_mqa.weight"])
    mutated["self_attn.kv_a_layernorm.weight"] = mutated["self_attn.kv_a_layernorm.weight"] * 7.0
    seq_mut, _, _ = adapter.capture_sequence(x_long, Layer(0, layer.prefix, mutated), seq_id="mut", compact_mask=True)
    assert torch.equal(seq_compact.sparse_selection.groups, seq_mut.sparse_selection.groups)
    assert adapter.dsa_indexer_depends_on_mla_latents is False


def test_position_buckets_boundaries_and_aggregation():
    """Verify half-open bucket boundaries (0-2k through 32k+), 3D M-RoPE, and weighted aggregation."""
    pos = torch.tensor([0, 2047, 2048, 4095, 4096, 8191, 8192, 16383, 16384, 32767, 32768, 40000])
    masks = position_bucket_masks(pos, DEFAULT_POSITION_BUCKETS)
    assert list(masks) == ["0-2k", "2k-4k", "4k-8k", "8k-16k", "16k-32k", "32k+"]
    for i, name in enumerate(masks):
        assert torch.nonzero(masks[name]).flatten().tolist() == [2 * i, 2 * i + 1]
    assert bool((sum(m.to(torch.int64) for m in masks.values()) == 1).all())

    mrope = torch.stack((pos, pos + 5000, pos + 10000), dim=0)
    for name, m in position_bucket_masks(mrope).items():
        assert torch.equal(m, masks[name])

    layer, adapter, _, raw_inputs = _synthetic_sparse_bank(619, seq_len=24, candidate_topk=12)
    geom, params = adapter.geometry_for(0), adapter.layer_params(layer)
    multi_pos = torch.tensor(
        [0, 100, 2047, 2048, 3000, 4095, 4096, 6000, 8191, 8192, 12000, 16383,
         16384, 24000, 32767, 32768, 34000, 36000, 38000, 40000, 41000, 42000, 43000, 44000],
        dtype=torch.int64,
    )
    seq, _, _ = adapter.capture_sequence(raw_inputs[0][1][:24], layer, seq_id="depth", positions=multi_pos, compact_mask=True)
    ev = LongContextEvaluator(
        modes=((4, 4),),
        content_transforms={0: (torch.eye(geom.kv_lora_rank), torch.ones(geom.kv_lora_rank))},
        rope_transforms={0: (torch.zeros(geom.qk_rope_dim // 2), torch.ones(geom.qk_rope_dim // 2))},
        config=FitConfig(protection="sink4_local128", sink_tokens=2, recent_tokens=4),
    )
    ev.evaluate_layer_sequence(seq, params, geom, split="heldout", device="cpu")
    rep = ev.finalize(corpus_meta={"sequences": 1}, transform_fingerprint="f" * 64, wall_seconds=0.1, device="cpu")
    row = rep["combinations"]["4:4"]
    assert [row["by_position_bucket"][k]["tokens"] for k in row["by_position_bucket"]] == [3, 3, 3, 3, 3, 9]
    weighted = sum(b["attn_rel_err"] * b["tokens"] for b in row["by_position_bucket"].values()) / row["layer_tokens"]
    assert row["attn_rel_err"] == pytest.approx(weighted)


def test_streaming_long_eval_end_to_end_and_cli_flags(tmp_path: Path):
    """End-to-end streaming long-eval over dense+MoE+SWA+DSA layers with real sidecar and report attachment."""
    g = torch.Generator().manual_seed(601)
    rand = lambda *shape: torch.randn(*shape, generator=g)

    hidden, hc, vocab = 16, 4, 64
    q_rank, d_c, d_r, nope, vdim = 12, 64, 64, 8, 8
    inter, moe_inter, moe_lat, n_exp = 24, 12, 10, 4
    layer_types = [reference.SPARSE, reference.SLIDING, reference.SPARSE]
    mlp_types = ["dense", "sparse", "sparse"]
    config = SimpleNamespace(
        attention=lambda idx: (4 if layer_types[idx] == reference.SLIDING else 2, nope, d_r, vdim),
        layer_types=layer_types, mlp_layer_types=mlp_types, engram_layer_ids=[],
        vocab_size=vocab, hidden_size=hidden, hc_mult=hc, hc_eps=1e-6, hc_magnitude=2.0,
        q_lora_rank=q_rank, kv_lora_rank=d_c, qk_rope_head_dim=d_r, rms_norm_eps=1e-5, sliding_window=8,
        intermediate_size=inter, moe_intermediate_size=moe_inter, routed_expert_hidden_size=moe_lat,
        n_routed_experts=n_exp, n_shared_experts=1, n_group=1, topk_group=1, num_experts_per_tok=2,
        norm_topk_prob=True, routed_scaling_factor=1.0, swiglu_limit=10.0,
        rope_parameters={
            reference.SPARSE: {"rope_theta": 10_000_000, "mrope_section": [12, 10, 10]},
            reference.SLIDING: {"rope_theta": 10_000, "mrope_section": [12, 10, 10]},
        },
        compressed_indexer_compress_ratio=4, compressed_indexer_group_topk=4,
        compressed_indexer_candidate_topk=16, compressed_indexer_num_attention_heads=2, compressed_indexer_head_dim=d_r,
    )

    def _make_layer(idx: int) -> Layer:
        h = 4 if layer_types[idx] == reference.SLIDING else 2
        w = {
            "input_layernorm.weight": torch.ones(hidden), "post_attention_layernorm.weight": torch.ones(hidden),
            "hc_attn_layer.hc_pre.hc_fn": rand(2 * hc, hc * hidden) * 0.1,
            "hc_attn_layer.hc_pre.hc_base": torch.zeros(2 * hc), "hc_attn_layer.hc_pre.hc_scale": torch.ones(2) * 0.1,
            "hc_mlp_layer.hc_pre.hc_fn": rand(2 * hc, hc * hidden) * 0.1,
            "hc_mlp_layer.hc_pre.hc_base": torch.zeros(2 * hc), "hc_mlp_layer.hc_pre.hc_scale": torch.ones(2) * 0.1,
            "self_attn.q_a_proj.weight": rand(q_rank, hidden) * 0.2, "self_attn.q_a_layernorm.weight": torch.ones(q_rank),
            "self_attn.q_b_proj.weight": rand(h * (nope + d_r), q_rank) * 0.2,
            "self_attn.kv_a_proj_with_mqa.weight": rand(d_c + d_r, hidden) * 0.2,
            "self_attn.kv_a_layernorm.weight": 1 + 0.1 * rand(d_c), "self_attn.kv_b_proj.weight": rand(h * (nope + vdim), d_c) * 0.2,
            "self_attn.linear_gate.weight": rand(h * vdim, hidden) * 0.2, "self_attn.o_proj.weight": rand(hidden, h * vdim) * 0.2,
            "self_attn.post_proj_norm.weight": torch.ones(hidden), "self_attn.learnable_sink_param": rand(h) * 0.1,
        }
        if layer_types[idx] == reference.SPARSE:
            w.update({
                "self_attn.compressed_indexer.token_key_projection.weight": rand(d_r, hidden) * 0.2,
                "self_attn.compressed_indexer.token_key_norm.weight": torch.ones(d_r),
                "self_attn.compressed_indexer.query_projection.weight": rand(2 * d_r, q_rank) * 0.2,
                "self_attn.compressed_indexer.key_value_projection.weight": rand(d_r, d_r) * 0.2,
                "self_attn.compressed_indexer.key_gate_projection.weight": rand(d_r, d_r) * 0.2,
                "self_attn.compressed_indexer.key_norm.weight": torch.ones(d_r),
                "self_attn.compressed_indexer.head_weight_projection.weight": rand(2, hidden) * 0.2,
                "self_attn.compressed_indexer.intra_group_ape": rand(4, d_r) * 0.1,
            })
        if mlp_types[idx] == "dense":
            w.update({"mlp.gate_proj.weight": rand(inter, hidden) * 0.2, "mlp.up_proj.weight": rand(inter, hidden) * 0.2,
                      "mlp.down_proj.weight": rand(hidden, inter) * 0.2, "mlp.post_fc2_norm.weight": torch.ones(hidden)})
        else:
            w.update({
                "mlp.shared_experts.gate_proj.weight": rand(moe_inter, hidden) * 0.2,
                "mlp.shared_experts.up_proj.weight": rand(moe_inter, hidden) * 0.2,
                "mlp.shared_experts.down_proj.weight": rand(hidden, moe_inter) * 0.2,
                "mlp.gate.weight": rand(n_exp, hidden) * 0.2, "mlp.gate.e_score_correction_bias": torch.zeros(n_exp),
                "mlp.routed_expert_down_proj.weight": rand(moe_lat, hidden) * 0.2,
                "mlp.routed_expert_up_proj.weight": rand(hidden, moe_lat) * 0.2,
                "mlp.experts.gate_up_proj": rand(n_exp, 2 * moe_inter, moe_lat) * 0.2,
                "mlp.experts.down_proj": rand(n_exp, moe_lat, moe_inter) * 0.2,
                "mlp.combine_output_norm.weight": torch.ones(hidden),
            })
        return Layer(idx, f"model.layers.{idx}.", w)

    layers_map = {i: _make_layer(i) for i in range(3)}
    embedding = rand(vocab, hidden) * 0.2
    reader = SimpleNamespace(
        device=torch.device("cpu"), config=config,
        load_layer=lambda idx: layers_map[idx], raw=lambda _name: embedding,
    )
    adapter = Hy4FlashCaptureAdapter(Model(reader), include_swa=False)
    cal_inputs = [(f"cal_{i:02d}", torch.randn(36 + (i % 3), hidden, generator=g), None) for i in range(4)]
    prov = {
        "dataset_repo": pc.V3_PANEL.repo_id, "dataset_revision": pc.V3_PANEL.revision,
        "components": {
            "agentic": {"sequences": 2, "tokens": 72, "input_ids_sha256": "1" * 64},
            "exl3_standard": {"sequences": 2, "tokens": 74, "input_ids_sha256": "2" * 64},
            "long_context": {"sequences": 2, "tokens": 4096, "input_ids_sha256": "3" * 64},
        },
        "input_ids_sha256": "4" * 64,
    }
    banks = {lid: adapter.capture_layer(layers_map[lid], cal_inputs, train_count=2, val_count=1, test_count=1, seed=0, provenance=prov) for lid in (0, 2)}
    source = {
        "repo": "wrldsuksgo2mars/Hy4-flash-EXL3-K4-MC", "revision": "a" * 40,
        "base_repo": "tencent/Hy4-flash", "base_revision": "b" * 40,
        "family": "hy4_flash", "model_type": "hy_v4_flash", "config_sha256": "c" * 64,
        "checkpoint_kind": "mc", "calibrated_view": "original",
    }
    out_dir = tmp_path / "run"
    cal_res = calibrate_hy4_flash_from_banks(
        banks, output_dir=out_dir, source=source, modes=((4, 4), (4, 0), (0, 4)),
        corpus_provenance=prov, fit_config=FitConfig(steps=4, val_every=2, seed=0, sink_tokens=2, recent_tokens=6),
    )
    manifest, c_tf, r_tf = load_sidecar_transforms(cal_res["sidecar_dir"], strict=True)
    assert set(c_tf) == {0, 2} and set(r_tf) == {0, 2}

    # Expose quantized history beyond the deployed four sinks and 128 recent keys.
    long_records = [
        LongContextRecord("lc-hold-01", "heldout", "16k", torch.randint(0, vocab, (192,), generator=g, dtype=torch.int64), "5" * 64),
        LongContextRecord("lc-cal-01", "calibration", "32k", torch.randint(0, vocab, (224,), generator=g, dtype=torch.int64), "6" * 64),
    ]
    long_eval = run_hy4_flash_long_eval(
        adapter, sidecar_path=cal_res["sidecar_dir"], records=long_records,
        corpus_meta={"source_kind": "prepared_panel_corpus", "sequences": 2, "tokens": 416}, device="cpu",
        expected_source={"repo": source["repo"], "revision": source["revision"], "family": "hy4_flash"},
    )
    assert long_eval["schema"] == LONG_EVAL_SCHEMA
    assert long_eval["layers_evaluated"] == [0, 2]
    assert long_eval["dsa_selection_drift"]["structural_zero"] is True
    assert long_eval["dsa_selection_drift"]["sparse_layers"] == [0, 2]
    assert set(long_eval["combinations"]) == {"4:4", "4:0", "0:4"}
    for mkey, row in long_eval["combinations"].items():
        assert row["layer_tokens"] == 832 and set(row["by_split"]) == {"heldout", "calibration"}
        assert set(row["per_layer"]) == {"0", "2"}
        assert row["rtn_attn_rel_err"] > 0
        assert row["ratio_to_rtn"] == pytest.approx(row["attn_rel_err"] / row["rtn_attn_rel_err"])
        if mkey == "0:4":
            assert row["hadamard_attn_rel_err"] == pytest.approx(row["rtn_attn_rel_err"])

    updated = attach_long_context_eval(out_dir, long_eval)
    assert "combinations" in updated and updated["long_context_eval"] == long_eval
    standalone_dir = tmp_path / "standalone"
    attach_long_context_eval(standalone_dir, long_eval, manifest=manifest)
    assert json.loads((standalone_dir / "qmla-validation-report.json").read_text())["long_context_eval"] == long_eval

    parser = build_parser()
    args = parser.parse_args(["--checkpoint", str(tmp_path), "--source-revision", "a" * 40,
                              "--output", str(standalone_dir), "--long-eval", "--sidecar", str(cal_res["sidecar_dir"])])
    assert args.long_eval is True and args.sidecar == Path(cal_res["sidecar_dir"])


HUB = Path.home() / ".cache/huggingface/hub"
HY4F_SNAPSHOT = HUB / "models--tencent--Hy4-flash/snapshots/d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959"


def test_long_eval_records_full_length_from_v3_panel(tmp_path: Path):
    """--long-eval renders full-length v3 long_context records through panel_corpus (no staged fallback)."""
    pytest.importorskip("transformers")
    from hy4_flash import prepare_corpus as hy4_corpus
    from hy4_flash.qmla_run import LONG_EVAL_CHAT, load_long_eval_records
    from qmla import panel_snapshot_dir

    v3 = HUB / f"datasets--wrldsuksgo2mars--agentic-quant-panel-v3/snapshots/{pc.V3_PANEL.revision}"
    if not v3.is_dir() or not HY4F_SNAPSHOT.is_dir():
        pytest.skip("cached v3 panel or Hy4-flash snapshot not available locally")
    assert panel_snapshot_dir().name == pc.V3_PANEL.revision

    records, meta = load_long_eval_records(HY4F_SNAPSHOT, None)
    assert meta["source_kind"] == "panel_corpus_full"
    assert (meta["dataset_repo"], meta["dataset_revision"]) == (pc.V3_PANEL.repo_id, pc.V3_PANEL.revision)
    assert [(k, meta["splits"][k]["sequences"]) for k in ("heldout", "calibration")] == [("heldout", 4), ("calibration", 12)]
    assert {k: v["sequences"] for k, v in meta["length_classes"].items()} == {"16k": 10, "32k": 6}
    assert [r.split for r in records] == ["heldout"] * 4 + ["calibration"] * 12
    for r in records:
        lo, hi = pc.LONG_CONTEXT_LENGTH_BOUNDS[r.length_class]
        assert lo <= r.input_ids.numel() <= hi and r.input_ids_sha256 == pc.ids_sha256(r.input_ids.tolist())

    # Exactly the records a full-mode panel_corpus preparation stores (same renderer and chat kwargs).
    chat = dict(hy4_corpus.CHAT)
    assert {k: chat[k] for k in LONG_EVAL_CHAT} == LONG_EVAL_CHAT
    out = tmp_path / "full"
    pc.prepare_panel_corpus(HY4F_SNAPSHOT, v3, out, source_repo="tencent/Hy4-flash", source_revision=HY4F_SNAPSHOT.name,
                            chat_kwargs=chat, components=("long_context",), full=False, long_context_mode="full")
    for sp in ("heldout", "calibration"):
        rows, _ = pc.load_component_records(out, sp, components=("long_context",), long_context_mode="full")
        mine = [r for r in records if r.split == sp]
        assert [r.seq_id for r in mine] == [w["id"] for w in rows]
        assert all(r.input_ids.tolist() == w["input_ids"] for r, w in zip(mine, rows))

    # A v2 snapshot has no long_context and fails closed.
    v2 = HUB / f"datasets--wrldsuksgo2mars--agentic-quant-panel-v2/snapshots/{pc.V2_PANEL.revision}"
    if v2.is_dir():
        with pytest.raises(ValueError, match="not present in two-component"):
            load_long_eval_records(HY4F_SNAPSHOT, pc.V2_PANEL.revision)

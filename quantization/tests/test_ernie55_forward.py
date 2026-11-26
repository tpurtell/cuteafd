"""Synthetic CPU witnesses against the PR #2808 reference entry points."""
import json
from pathlib import Path
import sys

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ernie55_flash.forward import ErnieForward, reference


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    raw = dict(model_type="ernie5_5_moe", hidden_size=16, num_hidden_layers=3, hc_mult=4,
               hc_sinkhorn_iters=20, hc_eps=1e-6, rms_norm_eps=1e-6, o_groups=8,
               num_attention_heads=8, head_dim=16, o_lora_rank=4, q_lora_rank=8,
               compress_ratios=[128, 128, -2], compress_rope_theta=160000, rope_theta=8000000,
               qk_rope_head_dim=8, sliding_window=4, index_topk=4, index_head_dim=16, index_n_heads=8,
               hybrid_mla_num_attention_heads=8, hybrid_mla_q_lora_rank=8, hybrid_mla_kv_lora_rank=8,
               hybrid_mla_qk_nope_head_dim=8, hybrid_mla_qk_rope_head_dim=8, hybrid_mla_v_head_dim=8,
               rope_scaling=dict(original_max_position_embeddings=64, factor=16., beta_fast=32., beta_slow=1.),
               first_k_dense_replace=1, moe_split_feature_routing=True, latent_moe_use_norm=True,
               scoring_func="sigmoid", topk_method="noaux_tc", norm_topk_prob=True,
               routed_scaling_factor_learnable=True, routed_scaling_factor=3.,
               n_routed_experts=24, num_experts_per_tok=10, hidden_act="situ",
               activation_situ_beta=4., activation_situ_linear_beta=25., n_shared_experts=1,
               moe_latent_size=8, moe_intermediate_size=8, intermediate_size=16,
               vocab_size=64, multimax="lm_head", dspark_markov_rank=8, dspark_target_layer_ids=[0, 1, 2],
               dspark_block_size=3, dspark_noise_token_id=11)
    config = reference.Config.parse(raw)
    specs = reference.tensor_contract(config) | reference.mtp_contract(config)
    for index in range(3):
        specs.update(reference.tensor_contract(config, index))
        specs.update(reference.mtp_contract(config, index))
    g = torch.Generator().manual_seed(224)
    weights = {}
    for name, (shape, dtype) in specs.items():
        w = torch.randn(shape, generator=g) * .15
        if 'norm.weight' in name or name.endswith('_scale'):
            w.add_(1.)
        weights[name] = w.bfloat16() if dtype == "BF16" else w
    snapshot = tmp_path_factory.mktemp("ernie55")
    save_file(weights, str(snapshot / "weights.safetensors"))
    (snapshot / "config.json").write_text(json.dumps(raw))
    (snapshot / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {k: "weights.safetensors" for k in specs}}))
    return ErnieForward.from_snapshot(snapshot)


def reduce_moe(model, state, weights):
    batch = model.route_batch(state.x, weights)
    ids, gamma = reference.moe_routing(state.x, weights, model.config)
    assert torch.equal(batch["ids"], ids)
    assert torch.equal(batch["gamma"], gamma)
    latent = batch["latent"]
    assert torch.equal(latent, reference.linear(state.x, weights["block_sparse_moe.fc1_latent_proj.weight"]))
    summed = torch.zeros_like(latent)
    for i in range(model.config.n_routed_experts):
        rows, slots = torch.where(ids == i)
        if len(rows):
            w1, w3, w2 = weights["experts"].get(i, torch.device("cpu"))
            active = model.activated(latent[rows], w1, w3)
            summed[rows] += reference.linear(active, w2) * gamma[rows, slots, None]
    expected = reference.norm(summed, weights["block_sparse_moe.latent_norm.weight"], model.config.rms_norm_eps)
    assert torch.equal(model.fc2_input(summed, weights), expected)
    return model.finish_moe(state, summed, weights)


@pytest.mark.parametrize(("index", "length"), [(0, 131), (1, 16), (2, 13)])
def test_dense_hca_and_dsa_boundaries_match_reference(tiny, index, length):
    c = tiny.config
    hidden = torch.randn(length, c.hc_mult, c.hidden_size, generator=torch.Generator().manual_seed(7))
    weights = tiny.load_layer(index)
    positions = torch.arange(length)
    state = tiny.mlp_input(hidden, weights, index, positions)
    if c.is_moe_layer(index):
        assert isinstance(weights["experts"], reference.LazyExpertStore)
        actual = reduce_moe(tiny, state, weights)
    else:
        actual = tiny.finish_dense(state, weights)
    assert torch.equal(actual, tiny.native_layer(hidden, weights, index, positions))


def test_draft_context_rope_stage_head_and_markov_order(tiny):
    c = tiny.config
    g = torch.Generator().manual_seed(17)
    taps = [torch.randn(7, c.hidden_size, generator=g) for _ in c.dspark_target_layer_ids]
    stages = [tiny.load_draft_layer(i) for i in range(c.num_mtp_stages)]
    aux = tiny.checkpoint.mtp_aux(c)
    outer = {k: tiny.checkpoint.load(k) for k in reference.OUTER_NAMES}
    assert torch.equal(tiny.draft_features(taps, aux), reference.dspark_project_target_taps(taps, aux, c))
    ctx = torch.randn(4, c.head_dim, generator=g)
    positions = torch.arange(7, 7 + c.dspark_block_size)
    hidden = torch.randn(c.dspark_block_size, c.hc_mult, c.hidden_size, generator=g)
    state = tiny.mlp_input(hidden, stages[0], 0, positions, draft_context_kv=ctx)
    actual = reduce_moe(tiny, state, stages[0])
    assert torch.equal(actual, tiny.native_draft_stage(hidden, stages[0], positions, ctx))
    assert torch.equal(tiny.logits(hidden, outer), reference.lm_head_logits(
        hidden, outer, c, eps_outside_sqrt=False))
    block = tiny.native_proposal(taps, 2, aux, stages, outer)
    expected = reference.dspark_propose(taps, 2, aux, stages, outer, c, draft_rope_mode="plain",
                                        multimax_after_markov=True, eps_outside_sqrt=False)
    assert all(torch.equal(a, b) for a, b in zip(block, expected, strict=True))


def test_hca_observer_exposes_distinct_real_inputs_for_all_eight_groups(tiny):
    # The second tile contains exactly three real rows. Padding them to 128
    # for shape-invariant GEMM must not increase the calibration sample count.
    length, c = 131, tiny.config
    hidden = torch.randn(length, c.hc_mult, c.hidden_size, generator=torch.Generator().manual_seed(3))
    weights = tiny.load_layer(0)
    positions = torch.arange(length)
    captures = [[] for _ in range(c.o_groups)]

    def record(group, start, tile):
        captures[group].append((start, tile.clone()))

    observed = tiny.mlp_input(hidden, weights, 0, positions, observe_hca_group=record)
    baseline = tiny.mlp_input(hidden, weights, 0, positions)
    assert torch.equal(observed.x, baseline.x)
    for group in captures:
        assert [(start, len(tile)) for start, tile in group] == [(0, 128), (128, 3)]
    inputs = [torch.cat([tile for _, tile in chunks]) for chunks in captures]
    assert all(rows.shape == (length, c.head_dim) for rows in inputs)
    assert not torch.equal(inputs[0], inputs[1])

    # Prove the tile actually feeds its named slice: perturb a single group's
    # matrix and use the recorded rows to predict the delta in attention.
    group = 3
    name = "self_attn.linear_o_group_proj"
    delta = torch.zeros_like(weights[name])
    group_delta = delta.view(c.o_groups, c.o_lora_rank, -1)[group]
    group_delta.fill_(.125)
    _, _, x = reference.mhc_pre(hidden, weights, "self_attention_hyper_connection",
                                weights["input_layernorm.weight"], c)
    gate = reference.norm(reference.linear(x, weights["self_attn.linear_q_down_proj.weight"]),
                          weights["self_attn.q_layernorm.weight"], c.rms_norm_eps)
    gate = reference.linear(gate, weights["self_attn.gate_proj.weight"]).sigmoid()
    # Zeroing the other groups isolates this slice without subtractive cancellation.
    isolated = torch.zeros_like(gate)
    isolated[:, group*c.o_lora_rank:(group+1)*c.o_lora_rank] = reference.linear(inputs[group], group_delta)
    expected = reference.linear(isolated * gate, weights["self_attn.o_proj.weight"])
    actual = reference.hca_attention(x, weights | {name: delta}, c, positions)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_staged_layer_major_and_draft_proposal_preflight(tiny):
    from ernie55_flash.forward_preflight import Prompt, run_forward

    prompts = [Prompt(torch.arange(n) % 64, {}) for n in (10, 7)]
    witness = torch.arange(259) % 64
    metrics, records = run_forward(tiny, prompts, [witness], pilot_layers=(0, 1, 2))
    assert all(r["error"] <= r["bound"] for r in metrics.values())
    assert {"draft_tokens", "dspark", "dense", "latent_moe", "long_hca", "long_mla_dsa", "target_token"} <= metrics.keys()
    assert records[0][0]["compressed_chunks"] == 2
    assert records[0][1]["pruned_query_rows"] > 0
    assert records[0][1]["stop"] == len(witness)
    for sample in prompts:
        assert sample.saved["layer_1"].shape == (len(sample.ids), 4, 16)
        assert sample.saved["tap_2"].shape == (len(sample.ids), 16)
        assert sample.saved["draft2_positions"].tolist() == list(range(len(sample.ids), len(sample.ids) + 3))
        assert sample.saved["draft0_context_kv"].shape == (4, 16)
        assert sample.saved["reference_proposal"].shape == (3,)

"""Synthetic contract, arithmetic, prefix-invariance and DSpark tests for ERNIE 5.5 Flash."""
from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/reference"))
from families.ernie55_flash import sources


def test_sources_pins_well_formed():
    assert len(sources.REVISION) == 40
    assert len(sources.VLLM_ERNIE_REVISION) == 40
    assert len(sources.SGLANG_REVISION) == 40
    assert "latent_norm(routed)" in sources.SEMANTICS["latent_moe_and_situ"]
    assert "vLLM-ERNIE" in sources.SEMANTICS["latent_moe_and_situ"]
    assert "SGLang" in sources.SEMANTICS["latent_moe_and_situ"]


try:
    import torch
    from families.ernie55_flash import model
except ImportError:  # pragma: no cover - host gate-venv without torch
    torch = None
    model = None

requires_torch = pytest.mark.skipif(torch is None, reason="torch required")


def _tiny_config(**overrides) -> model.Config:
    raw = {
        "model_type": "ernie5_5_moe",
        "vocab_size": 128,
        "hidden_size": 32,
        "num_hidden_layers": 3,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "q_lora_rank": 16,
        "kv_lora_rank": 16,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 8,
        "o_groups": 8,
        "o_lora_rank": 4,
        "sliding_window": 4,
        "compress_ratios": [128, 128, -2],
        "hybrid_mla_num_attention_heads": 8,
        "hybrid_mla_q_lora_rank": 16,
        "hybrid_mla_kv_lora_rank": 16,
        "hybrid_mla_qk_nope_head_dim": 8,
        "hybrid_mla_qk_rope_head_dim": 8,
        "hybrid_mla_v_head_dim": 8,
        "index_n_heads": 8,
        "index_head_dim": 16,
        "index_topk": 4,
        "max_position_embeddings": 1024,
        "rope_theta": 8000000.0,
        "compress_rope_theta": 160000.0,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 16.0,
            "original_max_position_embeddings": 64,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        },
        "rms_norm_eps": 1e-6,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-6,
        "intermediate_size": 48,
        "moe_intermediate_size": 16,
        "moe_latent_size": 16,
        "n_routed_experts": 6,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "first_k_dense_replace": 1,
        "routed_scaling_factor": 3.0,
        "routed_scaling_factor_learnable": True,
        "norm_topk_prob": True,
        "hidden_act": "situ",
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "multimax": "lm_head",
        "moe_split_feature_routing": True,
        "latent_moe_use_norm": True,
        "n_group": 1,
        "topk_group": 1,
        "tie_word_embeddings": False,
        "dspark_block_size": 4,
        "dspark_markov_rank": 8,
        "dspark_noise_token_id": 7,
        "dspark_target_layer_ids": [0, 1, 2],
    }
    raw.update(overrides)
    return model.Config.parse(raw)


def _rand_from_contract(contract, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    out = {}
    for name, (shape, _dtype) in contract.items():
        if name.endswith((".weight", "linear_o_group_proj", "mapping_proj.weight", "hc_head_fn")):
            t = torch.randn(shape, generator=g, dtype=torch.float32) * 0.1
        elif name.endswith(("norm.weight", "layernorm.weight", "routed_scaling_factor_param", "alpha_pre", "alpha_post", "alpha_res", "hc_head_scale")):
            t = torch.ones(shape, dtype=torch.float32) + torch.randn(shape, generator=g) * 0.05
        else:
            t = torch.randn(shape, generator=g, dtype=torch.float32) * 0.05
        out[name] = t
    return out


def _strip(d: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {k.removeprefix(prefix): v for k, v in d.items()}


@requires_torch
def test_real_config_contract_counts():
    """Real ERNIE 5.5 config contract accounts for all 70,826 indexed + 12 flashindexer tensors."""
    ratios = [128] * 43
    for i in (7, 14, 21, 28, 35, 42):
        ratios[i] = -2
    c = model.Config.parse({
        "model_type": "ernie5_5_moe",
        "vocab_size": 201216,
        "hidden_size": 4096,
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "q_lora_rank": 1024,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 448,
        "qk_rope_head_dim": 64,
        "o_groups": 8,
        "o_lora_rank": 1024,
        "sliding_window": 128,
        "compress_ratios": ratios,
        "hybrid_mla_num_attention_heads": 64,
        "hybrid_mla_q_lora_rank": 1024,
        "hybrid_mla_kv_lora_rank": 512,
        "hybrid_mla_qk_nope_head_dim": 192,
        "hybrid_mla_qk_rope_head_dim": 64,
        "hybrid_mla_v_head_dim": 256,
        "index_n_heads": 64,
        "index_head_dim": 128,
        "index_topk": 2048,
        "max_position_embeddings": 1048576,
        "rope_theta": 8000000.0,
        "compress_rope_theta": 160000.0,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 16.0,
            "original_max_position_embeddings": 65536,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        },
        "rms_norm_eps": 1e-6,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-6,
        "intermediate_size": 14336,
        "moe_intermediate_size": 2048,
        "moe_latent_size": 2048,
        "n_routed_experts": 512,
        "n_shared_experts": 1,
        "num_experts_per_tok": 10,
        "first_k_dense_replace": 1,
        "routed_scaling_factor": 3.0,
        "routed_scaling_factor_learnable": True,
        "norm_topk_prob": True,
        "hidden_act": "situ",
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "multimax": "lm_head",
        "moe_split_feature_routing": True,
        "latent_moe_use_norm": True,
        "n_group": 1,
        "topk_group": 1,
        "tie_word_embeddings": False,
        "dspark_block_size": 7,
        "dspark_markov_rank": 256,
        "dspark_noise_token_id": 1024,
        "dspark_target_layer_ids": [40, 41, 42],
    })
    total = len(model.tensor_contract(c))
    for i in range(43):
        total += len(model.tensor_contract(c, i))
    total += len(model.mtp_contract(c))
    for s in range(3):
        total += len(model.mtp_contract(c, s))
    assert total == 70826
    assert len(model.flashindexer_contract(c)) == 12


@requires_torch
def test_situ_and_multimax_exact():
    g = torch.tensor([-10.0, -1.0, 0.0, 2.5, 12.0])
    u = torch.tensor([-50.0, -3.0, 0.0, 4.0, 60.0])
    got = model.situ(g, u, 4.0, 25.0)
    want = (4.0 * torch.tanh(g / 4.0) * torch.sigmoid(g)) * (25.0 * torch.tanh(u / 25.0))
    torch.testing.assert_close(got, want, atol=0.0, rtol=0.0)

    logits = torch.tensor([[-20.0, 0.0, 9.0, 16.0, 25.0]])
    r = torch.tensor([15.625, 15.9375, 8.75, 0.3671875])
    t = torch.tensor([-0.1337890625, 0.279296875, -0.09765625, -0.0301513671875])
    got_m = model.multimax(logits, r, t)
    want_m = (
        logits
        + t[0] * (r[0] - logits).relu()
        + t[1] * (logits - r[1]).relu()
        + t[2] * (r[2] - logits).relu().square()
        + t[3] * (logits - r[3]).relu().square()
    )
    torch.testing.assert_close(got_m, want_m, atol=0.0, rtol=0.0)


@requires_torch
def test_mhc_pre_post_and_hc_head_eps_modes():
    c = _tiny_config(rms_norm_eps=0.5)  # Large eps separates eps_outside_sqrt vs inside
    w = _strip(_rand_from_contract(model.tensor_contract(c, 0), seed=1), "model.layers.0.")
    residual = torch.randn(5, 4, 32, generator=torch.Generator().manual_seed(2))
    post, comb, layer_in = model.mhc_pre(residual, w, "self_attention_hyper_connection", w["input_layernorm.weight"], c)
    assert post.shape == (5, 4, 1) and comb.shape == (5, 4, 4) and layer_in.shape == (5, 32)
    # Doubly-stochastic check after 20 Sinkhorn iterations
    torch.testing.assert_close(comb.sum(dim=-1), torch.ones(5, 4), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(comb.sum(dim=-2), torch.ones(5, 4), atol=1e-4, rtol=1e-4)
    updated = model.mhc_post(layer_in, residual, post, comb)
    assert updated.shape == residual.shape

    outer = _rand_from_contract(model.tensor_contract(c), seed=3)
    in_sqrt = model.hc_head_pre_norm(
        residual, outer[model.OUTER_NAMES[1]], outer[model.OUTER_NAMES[2]], outer[model.OUTER_NAMES[3]], c,
        eps_outside_sqrt=False,
    )
    out_sqrt = model.hc_head_pre_norm(
        residual, outer[model.OUTER_NAMES[1]], outer[model.OUTER_NAMES[2]], outer[model.OUTER_NAMES[3]], c,
        eps_outside_sqrt=True,
    )
    assert (in_sqrt - out_sqrt).abs().max() > 1e-3


@requires_torch
def test_hca_and_mla_dsa_prefix_invariance_and_flashindexer():
    c = _tiny_config()
    model.configure_f32()
    # Length > 128 exercises HCA compressor; length > window + topk (8) exercises DSA top-k pruning.
    seq_full = 140
    seq_pref = 132
    g = torch.Generator().manual_seed(11)
    hidden_full = torch.randn(seq_full, 4, 32, generator=g)

    # Layer 0: dense + HCA with compressor active (> 128 tokens)
    w0 = _strip(_rand_from_contract(model.tensor_contract(c, 0), seed=10), "model.layers.0.")
    out_full_0 = model.layer_forward(hidden_full, w0, c, 0)
    out_pref_0 = model.layer_forward(hidden_full[:seq_pref], w0, c, 0)
    torch.testing.assert_close(out_full_0[:seq_pref], out_pref_0, atol=1e-5, rtol=1e-5)

    # Layer 1: multi-router latent MoE + HCA
    w1 = _strip(_rand_from_contract(model.tensor_contract(c, 1), seed=20), "model.layers.1.")
    out_full_1 = model.layer_forward(out_full_0, w1, c, 1)
    out_pref_1 = model.layer_forward(out_pref_0, w1, c, 1)
    torch.testing.assert_close(out_full_1[:seq_pref], out_pref_1, atol=1e-5, rtol=1e-5)

    # Layer 2: MLA-DSA + FlashIndexer off vs on
    w2 = _strip(_rand_from_contract(model.tensor_contract(c, 2), seed=30), "model.layers.2.")
    fi_c = model.flashindexer_contract(c)
    fi_w = _strip(_rand_from_contract(fi_c, seed=31), "model.layers.2.")
    w2.update(fi_w)
    out_full_2 = model.layer_forward(out_full_1, w2, c, 2, enable_flashindexer=False)
    out_pref_2 = model.layer_forward(out_pref_1, w2, c, 2, enable_flashindexer=False)
    torch.testing.assert_close(out_full_2[:seq_pref], out_pref_2, atol=1e-5, rtol=1e-5)

    out_fi_2 = model.layer_forward(out_full_1, w2, c, 2, enable_flashindexer=True)
    assert out_fi_2.shape == out_full_2.shape and out_fi_2.isfinite().all()


@requires_torch
def test_dspark_drafter_2x2_modes():
    c = _tiny_config()
    outer = _rand_from_contract(model.tensor_contract(c), seed=40)
    mtp_aux = _strip(_rand_from_contract(model.mtp_contract(c), seed=41), "model.mtp.")
    stages = [
        _strip(_rand_from_contract(model.mtp_contract(c, s), seed=50 + s), f"model.mtp.{s}.")
        for s in range(3)
    ]
    g = torch.Generator().manual_seed(60)
    taps = [torch.randn(10, 32, generator=g) for _ in range(3)]

    results = {}
    for rope_mode in ("plain", "yarn"):
        for mm_after in (True, False):
            toks, logits, conf = model.dspark_propose(
                taps, 5, mtp_aux, stages, outer, c,
                draft_rope_mode=rope_mode, multimax_after_markov=mm_after,
            )
            assert toks.shape == (4,) and logits.shape == (4, 128) and conf.shape == (4,)
            assert logits.isfinite().all() and ((conf >= 0.0) & (conf <= 1.0)).all()
            results[(rope_mode, mm_after)] = logits

    # Distinct arms actually change logits
    assert not torch.allclose(results[("plain", True)], results[("yarn", True)])
    assert not torch.allclose(results[("plain", True)], results[("plain", False)])

"""GLM-specific QMLA capture and numerical replay against the pinned Transformers module."""
from pathlib import Path
import sys

import pytest
import torch
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextAttention

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glm5_flash.qmla_capture import Glm5FlashCaptureAdapter
from qmla.capture import eval_attention_output


def _layer(dtype, backend):
    torch.manual_seed(129)
    cfg = Glm5NextTextConfig(
        hidden_size=24, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2,
        kv_lora_rank=16, q_lora_rank=12, qk_rope_head_dim=0, qk_nope_head_dim=8, v_head_dim=8,
        index_kpool=4, index_topk=8, index_head_dim=8, index_n_heads=2,
    )
    cfg._attn_implementation = backend
    return cfg, Glm5NextTextAttention(cfg, 3).eval().to(dtype)


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_glm_flash_capture_replays_native_rounding_and_sparse_pool_tail(backend, dtype):
    cfg, layer = _layer(dtype, backend)
    adapter = Glm5FlashCaptureAdapter(cfg)
    assert [g.layer_id for g in adapter.mla_geometries()] == [3]
    assert not adapter.geometry_for(3).has_rope
    with pytest.raises(ValueError, match="full-history"):
        adapter.geometry_for(0)
    x = torch.randn(143, cfg.hidden_size).to(dtype)
    with torch.no_grad():
        dense, native = adapter.capture_sequence(x, layer, seq_id="dense")
        compact, tiled = adapter.capture_sequence(x, layer, seq_id="compact", compact_mask=True)
    assert dense.q_rope is None and dense.k_rope is None
    assert dense.allowed_mask[-1].sum().item() == 11
    assert compact.allowed_mask is None
    slots = torch.arange(len(x))
    mask = compact.sparse_selection.tile_mask(0, len(x), 0, len(x), slots)
    assert torch.equal(mask & (slots <= slots[:, None]), dense.allowed_mask)
    # The independently tiled capture must preserve the native selection and post-o_proj output.
    torch.testing.assert_close(tiled, native, rtol=2e-5, atol=2e-5)
    params = adapter.layer_params(layer)
    for seq in (dense, compact):
        recomputed = eval_attention_output(seq, params, adapter.geometry_for(3))
        torch.testing.assert_close(recomputed, native.float(), rtol=2e-5, atol=2e-5)


def test_capture_hook_rejects_padded_records_and_removes_all_hooks():
    cfg, layer = _layer(torch.float32, "eager")
    adapter = Glm5FlashCaptureAdapter(cfg)
    x = torch.randn(1, 32, cfg.hidden_size)
    mask = torch.ones(1, 32, dtype=torch.bool)
    mask[0, 0] = False
    with pytest.raises(ValueError, match="unpadded"):
        with adapter.capture_hook(layer, seq_id="padded"):
            layer(x, attention_mask=mask)
    for module in (layer, layer.indexer, layer.q_b_proj, layer.kv_a_proj_with_mqa):
        assert not module._forward_hooks


def test_reject_unknown_glm_layer_schedule():
    cfg, _ = _layer(torch.float32, "eager")
    cfg.indexer_types[3] = "shared"
    with pytest.raises(ValueError, match="unsupported attention/indexer"):
        Glm5FlashCaptureAdapter(cfg)


def test_glm_native_protection_uses_the_original_bf16_kv():
    cfg, layer = _layer(torch.bfloat16, "sdpa")
    adapter = Glm5FlashCaptureAdapter(cfg)
    # Native weights usually carry learned RMS scales; unit scales hide the BF16 product rounding.
    with torch.no_grad():
        layer.kv_a_layernorm.weight.copy_(torch.randn_like(layer.kv_a_layernorm.weight))
    x = torch.randn(21, cfg.hidden_size).bfloat16()
    seq, native = adapter.capture_sequence(x, layer, seq_id="protected")
    params = adapter.layer_params(layer)
    quantized = eval_attention_output(
        seq, params, adapter.geometry_for(3), content_fn=torch.zeros_like,
        kv_b_override=params.kv_b_weight * params.kv_norm_weight,
        exact_content_fn=lambda u: u,
        recent_tokens=32,
    )
    torch.testing.assert_close(quantized, native.float(), rtol=2e-5, atol=2e-5)

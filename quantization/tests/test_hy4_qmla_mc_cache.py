"""Hy4 MC captures and intervening attention layers see the same BF16 content cache."""
import pytest
import torch

from hy4_flash import forward
from hy4_flash.qmla_capture import Hy4FlashCaptureAdapter
from hy4_flash.qmla_mc import MCModel
from qmla.capture import eval_attention_output
from qmla.fit import check_content_identity
from test_qmla_capture import _synthetic_hy4_layer


@pytest.mark.parametrize("kind", [forward.reference.SPARSE, forward.reference.SLIDING])
def test_mc_attention_and_capture_round_the_weighted_latent(kind):
    _, teacher, layer, rng = _synthetic_hy4_layer(139, kind=kind)
    inputs = torch.randn(19, 10, generator=rng)
    model = MCModel(teacher.reader)
    adapter = Hy4FlashCaptureAdapter(model, include_swa=True)
    weight = layer.weights["self_attn.kv_a_layernorm.weight"]
    weighted = teacher.attention_first(inputs, layer).kv
    expected = weighted.bfloat16().float()
    assert not torch.equal(weighted, expected)

    seen = []

    def project(name, values, projection):
        if name.endswith(".self_attn.kv_b_proj"):
            seen.append(values.detach())
        return forward.native_projection(name, values, projection)

    propagated, _ = model.attention(inputs, layer, project)
    captured, observed, _ = adapter.capture_sequence(inputs, layer, seq_id="mc-cache", project=project)
    assert len(seen) == 2
    for cache in seen:
        assert torch.equal(cache, expected)
    assert torch.equal(captured.u * weight, weighted)
    assert not torch.equal((captured.u.bfloat16().float() * weight).bfloat16().float(), expected)
    torch.testing.assert_close(observed, propagated, rtol=2e-5, atol=2e-5)
    assert torch.equal(observed, captured.ref_output)


@pytest.mark.parametrize("kind", [forward.reference.SPARSE, forward.reference.SLIDING])
@pytest.mark.parametrize("compact", [False, True])
def test_mc_quantization_off_replays_the_cached_operand(kind, compact):
    _, teacher, layer, rng = _synthetic_hy4_layer(143, kind=kind)
    adapter = Hy4FlashCaptureAdapter(MCModel(teacher.reader), include_swa=True)
    seq, _, _ = adapter.capture_sequence(
        torch.randn(145, 10, generator=rng), layer, seq_id="mc-off", compact_mask=compact,
    )
    params = adapter.layer_params(layer)
    assert params.attention_arithmetic == "f32_tiled_bf16_cache"
    replayed = eval_attention_output(seq, params, adapter.geometry_for(layer.index))
    assert torch.equal(replayed, seq.ref_output)
    check_content_identity([seq], params, adapter.geometry_for(layer.index))

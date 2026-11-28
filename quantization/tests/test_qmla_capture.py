"""CPU numeric oracle tests for mode-agnostic QMLA capture and Hy4-flash hooks."""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hy4_flash.forward import Layer, Model, reference
from hy4_flash.qmla_capture import Hy4FlashCaptureAdapter, hy4_flash_mla_geometries
from qmla.capture import (
    MlaLayerGeometry,
    eval_attention_output,
    eval_positional_qk,
    load_layer_capture,
    save_layer_capture,
)


def _synthetic_hy4_layer(seed: int, *, kind: str, layer_id: int = 0):
    g = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=g)

    heads, nope, rope, vdim = (2, 4, 6, 3) if kind == reference.SPARSE else (4, 4, 6, 3)
    kv_rank, q_rank, hidden = 8, 6, 10
    config = SimpleNamespace(
        attention=lambda idx: (heads, nope, rope, vdim),
        layer_types=[kind] * (layer_id + 1),
        qk_rope_head_dim=rope,
        kv_lora_rank=kv_rank,
        rms_norm_eps=1e-5,
        sliding_window=4,
        rope_parameters={
            reference.SPARSE: {"rope_theta": 10_000_000, "mrope_section": [1, 1, 1]},
            reference.SLIDING: {"rope_theta": 10_000, "mrope_section": [1, 1, 1]},
        },
        compressed_indexer_compress_ratio=2,
        compressed_indexer_group_topk=3,
        compressed_indexer_candidate_topk=6,
        compressed_indexer_num_attention_heads=2,
        compressed_indexer_head_dim=6,
    )
    shapes = {
        "q_a_proj": (q_rank, hidden),
        "q_b_proj": (heads * (nope + rope), q_rank),
        "q_a_layernorm": (q_rank,),
        "kv_a_proj_with_mqa": (kv_rank + rope, hidden),
        "kv_b_proj": (heads * (nope + vdim), kv_rank),
        "kv_a_layernorm": (kv_rank,),
        "linear_gate": (heads * vdim, hidden),
        "o_proj": (hidden, heads * vdim),
        "post_proj_norm": (hidden,),
    }
    if kind == reference.SPARSE:
        shapes.update({
            "compressed_indexer.token_key_projection": (6, hidden),
            "compressed_indexer.token_key_norm": (6,),
            "compressed_indexer.query_projection": (12, q_rank),
            "compressed_indexer.key_value_projection": (6, 6),
            "compressed_indexer.key_gate_projection": (6, 6),
            "compressed_indexer.key_norm": (6,),
            "compressed_indexer.head_weight_projection": (2, hidden),
        })
    weights = {f"self_attn.{name}.weight": rand(*shape) for name, shape in shapes.items()}
    weights["self_attn.learnable_sink_param"] = rand(heads)
    if kind == reference.SPARSE:
        weights["self_attn.compressed_indexer.intra_group_ape"] = rand(2, 6)
    model = Model(SimpleNamespace(config=config))
    layer = Layer(layer_id, f"model.layers.{layer_id}.", weights)
    return config, model, layer, g


@pytest.mark.parametrize("kind,seq_len", [(reference.SPARSE, 19), (reference.SLIDING, 15)])
def test_hy4_flash_capture_reproduces_independent_f32_reference(kind: str, seq_len: int):
    config, model, layer, g = _synthetic_hy4_layer(41 if kind == reference.SPARSE else 43, kind=kind)
    inputs = torch.randn(seq_len, 10, generator=g)

    adapter = Hy4FlashCaptureAdapter(model, include_swa=True)
    captured, out, tap = adapter.capture_sequence(
        inputs, layer, seq_id=f"{kind}_0", capture_tap=(kind == reference.SPARSE)
    )
    expected = reference.attention(inputs, layer.weights, config, layer.index)
    torch.testing.assert_close(out, expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(captured.ref_output, expected, rtol=2e-5, atol=2e-5)

    # Pure shared evaluator on captured tensors must reproduce exact reference output.
    geom = adapter.geometry_for(layer.index)
    params = adapter.layer_params(layer)
    recomputed = eval_attention_output(captured, params, geom)
    torch.testing.assert_close(recomputed, expected, rtol=2e-5, atol=2e-5)

    # Positional QK oracle matches direct q_rope @ k_rope^T and preserves mask.
    z_p, mask = eval_positional_qk(captured, geom)
    direct = torch.einsum("thd,sd->hts", captured.q_rope, captured.k_rope)
    torch.testing.assert_close(z_p, direct, rtol=1e-5, atol=1e-5)
    assert torch.equal(mask, captured.allowed_mask)
    if kind == reference.SPARSE:
        assert tap is not None and tap.groups.shape == (seq_len, 3)
        # Confirm sparse top-k actually pruned older keys past candidate_topk=6.
        assert not bool(captured.allowed_mask[-1, :seq_len].all())
    else:
        # SWA-4 mask forbids distance >= 4.
        assert not bool(captured.allowed_mask[6, 0])
        assert bool(captured.allowed_mask[6, 3])


def test_hy4_flash_capture_3d_interleaved_mrope_matches_reference():
    config, model, layer, g = _synthetic_hy4_layer(73, kind=reference.SPARSE)
    seq_len = 11
    inputs = torch.randn(seq_len, 10, generator=g)
    # Non-degenerate 3D M-RoPE coordinates simulating an image span (T != H != W).
    pos_t = torch.tensor([0, 1, 2, 2, 2, 2, 3, 4, 5, 6, 7], dtype=torch.int64)
    pos_h = torch.tensor([0, 1, 2, 2, 3, 3, 3, 4, 5, 6, 7], dtype=torch.int64)
    pos_w = torch.tensor([0, 1, 2, 3, 2, 3, 3, 4, 5, 6, 7], dtype=torch.int64)
    mrope_pos = torch.stack((pos_t, pos_h, pos_w), dim=0)

    adapter = Hy4FlashCaptureAdapter(model)
    captured, out, _ = adapter.capture_sequence(inputs, layer, seq_id="vl_mrope_0", positions=mrope_pos)
    expected = reference.attention(inputs, layer.weights, config, layer.index, rope_positions=mrope_pos)
    torch.testing.assert_close(out, expected, rtol=2e-5, atol=2e-5)

    recomputed = eval_attention_output(captured, adapter.layer_params(layer), adapter.geometry_for(0))
    torch.testing.assert_close(recomputed, expected, rtol=2e-5, atol=2e-5)
    assert captured.positions.shape == (3, seq_len)

    # Sanity check that 3D M-RoPE actually differs from 1D text positions.
    _, out_1d, _ = adapter.capture_sequence(inputs, layer, seq_id="text_1d_0", positions=pos_t)
    assert not torch.allclose(out, out_1d, atol=1e-4)


def test_capture_bank_xxh3_roundtrip_and_corruption_fail_closed(tmp_path: Path):
    _, model, layer, g = _synthetic_hy4_layer(97, kind=reference.SPARSE)
    adapter = Hy4FlashCaptureAdapter(model)
    sequences = [
        (f"seq_{i:02d}", torch.randn(8 + i, 10, generator=g), None)
        for i in range(6)
    ]
    bank = adapter.capture_layer(
        layer,
        sequences,
        train_count=4,
        val_count=2,
        seed=7,
        provenance={"corpus": "synthetic-oracle"},
    )
    assert len(bank.train) == 4 and len(bank.val) == 2
    assert {s.seq_id for s in bank.train}.isdisjoint({s.seq_id for s in bank.val})

    save_layer_capture(tmp_path, bank)
    reloaded = load_layer_capture(tmp_path, layer.index)
    assert reloaded.geometry == bank.geometry
    assert [s.seq_id for s in reloaded.train] == [s.seq_id for s in bank.train]
    assert [s.seq_id for s in reloaded.val] == [s.seq_id for s in bank.val]
    for orig, loaded in zip(bank.train, reloaded.train, strict=True):
        assert torch.equal(orig.u, loaded.u)
        assert torch.equal(orig.q_rope, loaded.q_rope)
        assert torch.equal(orig.k_rope, loaded.k_rope)
        assert torch.equal(orig.ref_output, loaded.ref_output)

    # Corrupting a single payload byte must fail XXH3-128 verification.
    payload_path = tmp_path / "layer-0000.safetensors"
    blob = bytearray(payload_path.read_bytes())
    blob[-1] ^= 0xFF
    payload_path.write_bytes(bytes(blob))
    with pytest.raises(ValueError, match="XXH3-128 mismatch"):
        load_layer_capture(tmp_path, layer.index)


def test_hy4_flash_geometry_filtering_and_nope_validation():
    config = SimpleNamespace(
        attention=lambda idx: (32, 192, 64, 256) if idx in (0, 1) else (64, 192, 64, 256),
        layer_types=[reference.SPARSE, reference.SPARSE, reference.SLIDING, reference.SLIDING],
        kv_lora_rank=512,
        rms_norm_eps=1e-5,
        sliding_window=512,
        rope_parameters={
            reference.SPARSE: {"rope_theta": 10_000_000, "mrope_section": [12, 10, 10]},
            reference.SLIDING: {"rope_theta": 10_000, "mrope_section": [12, 10, 10]},
        },
    )
    dsa_only = hy4_flash_mla_geometries(config, include_swa=False)
    assert [g.layer_id for g in dsa_only] == [0, 1]
    assert all(g.kind == "dsa" and g.mrope_section == (12, 10, 10) for g in dsa_only)

    all_layers = hy4_flash_mla_geometries(config, include_swa=True)
    assert [g.layer_id for g in all_layers] == [0, 1, 2, 3]
    assert [g.sliding_window for g in all_layers] == [None, None, 512, 512]

    # Pure-NoPE geometry (e.g. GLM 5.3 Flash) validates cleanly and rejects qk_rope_dim > 0.
    nope_geom = MlaLayerGeometry(
        layer_id=3, kv_source_layer=3, kind="dsa", heads=64,
        kv_lora_rank=512, qk_nope_dim=256, qk_rope_dim=0, v_head_dim=256, rope_kind="none",
    )
    nope_geom.validate()
    assert not nope_geom.has_rope
    with pytest.raises(ValueError, match="qk_rope_dim == 0"):
        MlaLayerGeometry(
            layer_id=3, kv_source_layer=3, kind="dsa", heads=64,
            kv_lora_rank=512, qk_nope_dim=256, qk_rope_dim=64, v_head_dim=256, rope_kind="none",
        ).validate()

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
        train_count=3,
        val_count=2,
        test_count=1,
        seed=7,
        provenance={"corpus": "synthetic-oracle"},
    )
    assert (len(bank.train), len(bank.val), len(bank.test)) == (3, 2, 1)
    ids = [{s.seq_id for s in bank.split(name)} for name in ("train", "val", "test")]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])

    save_layer_capture(tmp_path, bank)
    reloaded = load_layer_capture(tmp_path, layer.index)
    assert reloaded.geometry == bank.geometry
    assert [s.seq_id for s in reloaded.train] == [s.seq_id for s in bank.train]
    assert [s.seq_id for s in reloaded.val] == [s.seq_id for s in bank.val]
    assert [s.seq_id for s in reloaded.test] == [s.seq_id for s in bank.test]
    assert reloaded.provenance["splits"] == bank.provenance["splits"]
    for orig, loaded in zip(bank.train, reloaded.train, strict=True):
        assert torch.equal(orig.u, loaded.u)
        assert torch.equal(orig.q_rope, loaded.q_rope)
        assert torch.equal(orig.k_rope, loaded.k_rope)
        assert torch.equal(orig.ref_output, loaded.ref_output)

    # Large capture banks must stay file-backed, even after the loader closes its fd.
    payload_path = tmp_path / "layer-0000.safetensors"
    if Path("/proc/self/maps").is_file():
        address = reloaded.train[0].u.data_ptr()
        ranges = [line.split()[0].split("-") for line in Path("/proc/self/maps").read_text().splitlines()
                  if line.endswith(str(payload_path))]
        assert any(int(lo, 16) <= address < int(hi, 16) for lo, hi in ranges)
    reloaded.train[0].u.zero_()
    assert torch.equal(load_layer_capture(tmp_path, layer.index).train[0].u, bank.train[0].u)

    # Corrupting a single payload byte must fail XXH3-128 verification.
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


def test_protected_replay_uses_each_query_prefix_for_content_and_rope():
    _, model, layer, rng = _synthetic_hy4_layer(67, kind=reference.SPARSE)
    adapter = Hy4FlashCaptureAdapter(model)
    seq, _, _ = adapter.capture_sequence(torch.randn(140, 10, generator=rng), layer, seq_id="prefix")
    geometry = adapter.geometry_for(0)
    params = adapter.layer_params(layer)

    def exact(c):
        return c

    def rounded(u):
        return exact(u).round()

    def rope(k):
        return k.round()

    pred = eval_attention_output(
        seq, params, geometry, content_fn=rounded, rope_key_fn=rope, exact_content_fn=exact,
        exact_rope_key_fn=lambda k: k, sink_tokens=4, recent_tokens=8,
    )
    index = torch.arange(seq.seq_len)
    for row in (0, 11, 12, 46, 127, 128, 139):
        keep = ((index < 4) | (row - index < 8))[:, None]
        oracle = eval_attention_output(
            seq, params, geometry, content_fn=lambda u: torch.where(keep, exact(u), rounded(u)),
            rope_key_fn=lambda k: torch.where(keep, k, rope(k)),
        )
        torch.testing.assert_close(pred[row], oracle[row], atol=2e-5, rtol=2e-5)


def test_three_way_split_disjoint_deterministic_salt_versioned_and_refuses_reuse_mismatch(tmp_path: Path):
    import hashlib
    import json

    from hy4_flash.qmla_run import check_reused_captures, corpus_split_description
    from qmla.capture import (
        CAPTURE_SCHEMA,
        SPLIT_SALT,
        check_split_description,
        describe_splits,
        split_train_val_test,
    )

    records = [(f"agentic:{i:05d}", torch.arange(i, i + 5 + i % 4)) for i in range(40)]
    train, val, test = split_train_val_test(records, train_count=16, val_count=4, test_count=4, key_fn=lambda r: r[0])
    ids = [[r[0] for r in part] for part in (train, val, test)]
    assert [len(x) for x in ids] == [16, 4, 4]
    assert not (set(ids[0]) & set(ids[1]) or set(ids[0]) & set(ids[2]) or set(ids[1]) & set(ids[2]))

    # Deterministic and independent of input order; a different seed gives a different partition.
    again = split_train_val_test(list(reversed(records)), train_count=16, val_count=4, test_count=4, key_fn=lambda r: r[0])
    assert [[r[0] for r in part] for part in again] == ids
    other = split_train_val_test(records, train_count=16, val_count=4, test_count=4, key_fn=lambda r: r[0], seed=1)
    assert [[r[0] for r in part] for part in other] != ids

    # The order is the versioned salt's: the v1 two-way salt orders the rows differently.
    assert SPLIT_SALT == "cuteafd-qmla-split-v2"
    v2 = sorted((r[0] for r in records), key=lambda k: hashlib.sha256(f"{SPLIT_SALT}:0:{k}".encode()).hexdigest())
    v1 = sorted((r[0] for r in records), key=lambda k: hashlib.sha256(f"cuteafd-qmla-split-v1:0:{k}".encode()).hexdigest())
    assert v2[:24] == [*ids[0], *ids[1], *ids[2]] and v1[:24] != v2[:24]
    with pytest.raises(ValueError, match="positive"):
        split_train_val_test(records, train_count=16, val_count=4, test_count=0)
    with pytest.raises(ValueError, match="need at least"):
        split_train_val_test(records[:10], train_count=8, val_count=2, test_count=1)

    description = corpus_split_description(records, train_count=16, val_count=4, test_count=4, seed=0)
    assert description["salt"] == SPLIT_SALT and description["test"]["seq_ids"] == ids[2]
    assert description["test"]["sha256"][0] == hashlib.sha256(
        dict(records)[ids[2][0]].contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()
    stale = json.loads(json.dumps(description))
    stale["salt"] = "cuteafd-qmla-split-v1"
    with pytest.raises(ValueError, match="split salt"):
        check_split_description(stale)
    tampered = json.loads(json.dumps(description))
    tampered["val"]["sha256"][0] = "0" * 64
    with pytest.raises(ValueError, match="val split digest mismatch"):
        check_split_description(tampered)
    overlap = describe_splits({"train": [("a", "1" * 64)], "val": [("b", "2" * 64)], "test": [("c", "3" * 64)]}, seed=0)
    overlap["test"] = overlap["val"]
    with pytest.raises(ValueError, match="overlaps"):
        check_split_description(overlap)

    # Capture banks carry the description; reuse refuses a split, seed or count change.
    _, model, layer, g = _synthetic_hy4_layer(131, kind=reference.SPARSE)
    adapter = Hy4FlashCaptureAdapter(model)
    rows = [(f"agentic:{i:05d}", torch.randn(8 + i, 10, generator=g), None) for i in range(7)]
    teacher = {"target_revision": "a" * 40}
    provenance = {"dataset_revision": "b" * 40, "components": {"agentic": {}}, **teacher}
    bank = adapter.capture_layer(layer, rows, train_count=3, val_count=2, test_count=2, provenance=provenance)
    save_layer_capture(tmp_path, bank)
    reloaded = {0: load_layer_capture(tmp_path, 0)}
    resolved = {"panel_revision": "b" * 40, "components": ("agentic",),
                "train_sequences": 3, "val_sequences": 2, "test_sequences": 2}
    check_reused_captures(reloaded, resolved, teacher, seed=0, expected_splits=bank.provenance["splits"])
    with pytest.raises(ValueError, match="test split count"):
        check_reused_captures(reloaded, {**resolved, "test_sequences": 3}, teacher, seed=0)
    with pytest.raises(ValueError, match="split seed"):
        check_reused_captures(reloaded, resolved, teacher, seed=1)
    reordered = adapter.capture_layer(layer, rows, train_count=3, val_count=2, test_count=2, seed=5,
                                      provenance=provenance).provenance["splits"]
    reordered["seed"] = 0
    with pytest.raises(ValueError, match="split membership differs"):
        check_reused_captures(reloaded, resolved, teacher, seed=0, expected_splits=reordered)

    # A two-way capture manifest under the old schema refuses to load.
    manifest = json.loads((tmp_path / "layer-0000.json").read_text())
    manifest["schema"] = "cuteafd-qmla-capture-v1"
    (tmp_path / "layer-0000.json").write_text(json.dumps(manifest))
    assert CAPTURE_SCHEMA == "cuteafd-qmla-capture-v2"
    with pytest.raises(ValueError, match="three-way"):
        load_layer_capture(tmp_path, 0)


def test_content_transform_uses_rounded_cache_and_identity_matches_native_bytes():
    from dataclasses import replace
    import torch.nn.functional as F

    from qmla.capture import CapturedLayerParams, CapturedSequence, cached_content
    from qmla.fit import _check_content_equivalence, apply_content_forward, check_content_identity, compensate_kv_b

    rng = torch.Generator().manual_seed(108)
    rows, rank, heads, dim = 17, 8, 2, 4

    def native(*shape):
        return torch.randn(shape, generator=rng).bfloat16()

    u, gamma = native(rows, rank), native(rank)
    w, o = native(2 * heads * dim, rank), native(rank, heads * dim)
    q = native(rows, heads, dim)
    mask = torch.ones(rows, rows, dtype=torch.bool).tril()
    c = u * gamma
    assert not torch.equal(c.float(), u.float() * gamma.float())
    k, v = F.linear(c, w).reshape(rows, heads, 2 * dim).chunk(2, dim=-1)
    attended = F.scaled_dot_product_attention(
        q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], attn_mask=mask[None, None],
    )[0].transpose(0, 1)
    ref = F.linear(attended.reshape(rows, heads * dim), o).float()
    geom = MlaLayerGeometry(layer_id=3, kv_source_layer=3, kind="dsa", heads=heads, kv_lora_rank=rank,
                            qk_nope_dim=dim, qk_rope_dim=0, v_head_dim=dim, rope_kind="none")
    params = CapturedLayerParams(gamma.float(), w.float(), o.float(), attention_arithmetic="bf16_sdpa")
    seq = CapturedSequence("native", torch.arange(rows), u.float(), q.float(), None, None, mask, None, ref)
    assert torch.equal(cached_content(seq.u, params), c.float())
    rotation = torch.linalg.qr(torch.randn(rank, rank, generator=rng)).Q
    scales = torch.exp(torch.randn(rank, generator=rng))
    weight = compensate_kv_b(w, rotation, scales)
    torch.testing.assert_close(weight, (w.float() @ rotation) / scales)
    torch.testing.assert_close(F.linear(apply_content_forward(c, rotation, scales), weight), F.linear(c.float(), w.float()))
    assert _check_content_equivalence([seq], params, geom, rotation, scales) < 1e-5
    check_content_identity([seq], params, geom)
    perturbed = ref.clone()
    perturbed.view(torch.uint8).flatten()[0] ^= 1
    with pytest.raises(ValueError, match="identity differs from native capture bytes"):
        check_content_identity([replace(seq, ref_output=perturbed)], params, geom)


@pytest.mark.parametrize("kind", [reference.SPARSE, reference.SLIDING])
def test_identity_matches_hy4_capture_bytes(kind):
    from qmla.fit import check_content_identity

    _, model, layer, rng = _synthetic_hy4_layer(531, kind=kind)
    adapter = Hy4FlashCaptureAdapter(model)
    seq, _, _ = adapter.capture_sequence(torch.randn(140, 10, generator=rng), layer, seq_id="identity")
    check_content_identity([seq], adapter.layer_params(layer), adapter.geometry_for(0))

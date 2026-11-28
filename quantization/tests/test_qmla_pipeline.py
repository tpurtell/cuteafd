"""CPU numeric oracle tests for shared QMLA quantizer, fit, sidecar, publish, and Hy4-flash pipeline."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "quantization"))
sys.path.insert(0, str(ROOT / "scripts/lib"))

import mc_manifest
from hy4_flash.forward import Layer, Model, reference
from hy4_flash.qmla_capture import Hy4FlashCaptureAdapter
from hy4_flash.qmla_run import (
    build_parser,
    calibrate_hy4_flash_from_banks,
    load_pinned_qmla_corpus,
    parse_components_arg,
    parse_layers_arg,
    parse_modes_arg,
    resolve_run_settings,
)
from qmla.measurement import _token_relative_squares
from qmla import (
    DEFAULT_PROTECTION_POLICY,
    DEFAULT_QMLA_PANEL_REVISION,
    PROTECTION_SCOPE,
    QMLA_BASE_COMPONENTS,
    QMLA_KNOWN_COMPONENTS,
    QMLA_PANEL_REPO,
    FitConfig,
    MlaLayerGeometry,
    apply_rope_pair_transform,
    attach_qmla_to_mc,
    content_output_loss,
    eval_attention_output,
    fit_config_digest,
    fit_content_layer,
    fit_rope_layer,
    load_fit_checkpoint,
    pack_content_row,
    pack_joint_record,
    pack_rope_row,
    parse_qmla_mode,
    plan_incremental_revision,
    protection_mask,
    publish_incremental_revision,
    qdq_asymmetric_group,
    read_sidecar_manifest,
    record_bytes_for_mode,
    resolve_qmla_panel,
    save_fit_checkpoint,
    sylvester_hadamard,
    unpack_content_row,
    unpack_joint_record,
    unpack_rope_row,
    validate_sidecar,
)


def _synthetic_panel_provenance(*, include_long_context: bool = False, rev: str = DEFAULT_QMLA_PANEL_REVISION) -> dict:
    comps = {
        "agentic": {"sequences": 4, "tokens": 600, "input_ids_sha256": "1" * 64},
        "exl3_standard": {"sequences": 2, "tokens": 300, "input_ids_sha256": "2" * 64},
    }
    if include_long_context:
        comps["long_context"] = {"sequences": 2, "tokens": 32768, "input_ids_sha256": "3" * 64}
    return {
        "dataset_repo": QMLA_PANEL_REPO,
        "dataset_revision": rev,
        "components": comps,
        "input_ids_sha256": "4" * 64,
    }


def _make_skewed_hy4_bank(
    seed: int,
    *,
    layer_id: int = 0,
    d_c: int = 64,
    d_r: int = 64,
    seqs: int = 6,
    base_seq_len: int = 16,
):
    """Build a synthetic Hy4-flash layer with strong channel outliers (using repo generators only)."""
    g = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=g)

    heads, nope, vdim = 2, 8, 8
    q_rank, hidden = 12, 16
    config = SimpleNamespace(
        attention=lambda _idx: (heads, nope, d_r, vdim),
        layer_types=[reference.SPARSE] * (layer_id + 1),
        qk_rope_head_dim=d_r,
        kv_lora_rank=d_c,
        rms_norm_eps=1e-5,
        sliding_window=4,
        rope_parameters={
            reference.SPARSE: {"rope_theta": 10_000_000, "mrope_section": [12, 10, 10]},
        },
        compressed_indexer_compress_ratio=2,
        compressed_indexer_group_topk=3,
        compressed_indexer_candidate_topk=max(32, base_seq_len + 8),
        compressed_indexer_num_attention_heads=2,
        compressed_indexer_head_dim=d_r,
    )
    kv_a = rand(d_c + d_r, hidden)
    kv_a[0] *= 18.0
    kv_a[1] *= 12.0
    kv_a[d_c + 0] *= 15.0
    kv_a[d_c + 1] *= 15.0
    kv_a[d_c + 2] *= 0.08
    kv_a[d_c + 3] *= 0.08

    kv_norm = torch.ones(d_c)
    kv_norm[0] = 4.0
    kv_norm[2] = 0.25

    weights = {
        "self_attn.q_a_proj.weight": rand(q_rank, hidden),
        "self_attn.q_a_layernorm.weight": torch.ones(q_rank),
        "self_attn.q_b_proj.weight": rand(heads * (nope + d_r), q_rank),
        "self_attn.kv_a_proj_with_mqa.weight": kv_a,
        "self_attn.kv_a_layernorm.weight": kv_norm,
        "self_attn.kv_b_proj.weight": rand(heads * (nope + vdim), d_c),
        "self_attn.linear_gate.weight": rand(heads * vdim, hidden),
        "self_attn.o_proj.weight": rand(hidden, heads * vdim),
        "self_attn.post_proj_norm.weight": torch.ones(hidden),
        "self_attn.learnable_sink_param": rand(heads),
    }
    model = Model(SimpleNamespace(config=config))
    layer = Layer(layer_id, f"model.layers.{layer_id}.", weights)
    adapter = Hy4FlashCaptureAdapter(model)
    inputs = [(f"seq_{i:02d}", rand(base_seq_len + (i % 3), hidden), None) for i in range(seqs)]
    bank = adapter.capture_layer(
        layer,
        inputs,
        train_count=seqs - 2,
        val_count=2,
        seed=seed,
        provenance=_synthetic_panel_provenance(),
    )
    return config, model, layer, adapter, bank


def test_panel_resolution_defaults_to_two_components_and_supports_known_long_context(tmp_path: Path):
    """Amended Correction 1: default runs on agentic + exl3_standard; long_context accepted by revision; unknown rejected."""
    from safetensors.torch import save_file

    # Default resolves immediately to the two existing components at 54683731eb01c7c2fa5b801a0888e029a0e1b1c7.
    repo, rev, comps = resolve_qmla_panel()
    assert repo == QMLA_PANEL_REPO
    assert rev == "54683731eb01c7c2fa5b801a0888e029a0e1b1c7"
    assert comps == ("agentic", "exl3_standard") == QMLA_BASE_COMPONENTS

    # Adding known third component `long_context` with a new panel revision succeeds.
    repo3, rev3, comps3 = resolve_qmla_panel(
        panel_revision="e" * 40, components=("agentic", "exl3_standard", "long_context")
    )
    assert repo3 == QMLA_PANEL_REPO and rev3 == "e" * 40 and comps3 == QMLA_KNOWN_COMPONENTS
    assert parse_components_arg("agentic,exl3_standard") == ("agentic", "exl3_standard")
    assert parse_components_arg("agentic,exl3_standard,long_context") == QMLA_KNOWN_COMPONENTS

    # Arbitrary unknown component names are rejected (no pluggable arbitrary list).
    with pytest.raises(ValueError, match="unknown QMLA corpus component"):
        resolve_qmla_panel(components=("agentic", "exl3_standard", "custom_arbitrary"))
    with pytest.raises(ValueError, match="requires base components"):
        resolve_qmla_panel(components=("agentic",))

    # Direct hy4_flash.prepare_corpus output directory (manifest.json + calibration.safetensors) loads directly.
    prep_dir = tmp_path / "prepared_v2"
    prep_dir.mkdir()
    g = torch.Generator().manual_seed(77)
    rows = [torch.randint(10, 1000, (12 + i,), generator=g, dtype=torch.int64) for i in range(4)]
    offsets = [0]
    for r in rows:
        offsets.append(offsets[-1] + int(r.numel()))
    save_file(
        {"input_ids": torch.cat(rows), "offsets": torch.tensor(offsets, dtype=torch.int64)},
        str(prep_dir / "calibration.safetensors"),
    )
    (prep_dir / "manifest.json").write_text(
        json.dumps({
            "dataset": {"repo_id": QMLA_PANEL_REPO, "revision": DEFAULT_QMLA_PANEL_REVISION},
            "components": {"agentic": {"count": 2}, "exl3_standard": {"count": 2}},
        }),
        encoding="utf-8",
    )
    recs, prov = load_pinned_qmla_corpus(prep_dir)
    assert len(recs) == 4
    assert prov["dataset_revision"] == DEFAULT_QMLA_PANEL_REVISION
    assert sorted(prov["components"].keys()) == ["agentic", "exl3_standard"]
    assert prov["components"]["agentic"]["sequences"] == 2
    assert prov["components"]["exl3_standard"]["sequences"] == 2


def test_quantizer_rne_pack_unpack_bit_exact_and_record_sizes():
    g = torch.Generator().manual_seed(11)
    content = torch.randn(512, generator=g) * 3.0
    rope = torch.randn(64, generator=g) * 2.5

    for bits in (2, 4, 8):
        packed = pack_content_row(content, bits)
        unpacked = unpack_content_row(packed, 512, bits)
        expected = qdq_asymmetric_group(content, bits, scale_dtype="BF16", ste=False)
        assert torch.equal(unpacked, expected), f"content INT{bits} pack/unpack mismatch"

    for bits in (4, 8):
        packed = pack_rope_row(rope, bits)
        unpacked = unpack_rope_row(packed, 64, bits)
        expected = qdq_asymmetric_group(rope, bits, scale_dtype="F32", ste=False)
        assert torch.equal(unpacked, expected), f"rope INT{bits} pack/unpack mismatch"

    for c_bits, r_bits, expected_bytes in (
        (4, 4, 320),  # Exact QuantMLA Eq. 40
        (2, 4, 192),
        (4, 8, 352),
        (8, 8, 608),
    ):
        raw = pack_joint_record(content, rope, c_bits, r_bits)
        assert len(raw) == expected_bytes
        uc, ur = unpack_joint_record(raw, 512, 64, c_bits, r_bits)
        assert torch.equal(uc, qdq_asymmetric_group(content, c_bits, scale_dtype="BF16", ste=False))
        assert torch.equal(ur, qdq_asymmetric_group(rope, r_bits, scale_dtype="F32", ste=False))

    assert record_bytes_for_mode(0, 0, base_kv_cache="bf16") == 1152
    assert record_bytes_for_mode(0, 0, base_kv_cache="fp8") == 656
    assert record_bytes_for_mode(4, 0, d_r=64) == 416
    assert record_bytes_for_mode(2, 0, d_r=64) == 288
    assert record_bytes_for_mode(8, 0, d_r=64) == 672
    assert record_bytes_for_mode(0, 4, d_r=64, base_kv_cache="fp8") == 576
    assert record_bytes_for_mode(0, 8, d_r=64, base_kv_cache="fp8") == 608
    assert record_bytes_for_mode(4, 0, d_r=0) == 288
    assert record_bytes_for_mode(2, 0, d_r=0) == 160
    assert record_bytes_for_mode(8, 0, d_r=0) == 544

    assert parse_qmla_mode("4:4") == (4, 4)
    assert parse_qmla_mode("0:0") == (0, 0)
    with pytest.raises(ValueError, match="INT2 RoPE"):
        parse_qmla_mode("4:2")
    with pytest.raises(ValueError):
        parse_qmla_mode("3:4")
    with pytest.raises(ValueError, match="NoPE"):
        record_bytes_for_mode(4, 4, d_r=0)

    h = sylvester_hadamard(64)
    torch.testing.assert_close(h @ h.T, torch.eye(64), atol=1e-6, rtol=1e-6)


def test_rope_pair_transform_commutes_with_3d_interleaved_mrope_and_split_half():
    g = torch.Generator().manual_seed(29)
    t, heads, d_r = 9, 3, 64
    q_raw = torch.randn(t, heads, d_r, generator=g)
    k_raw = torch.randn(t, d_r, generator=g)
    theta = torch.randn(d_r // 2, generator=g) * 0.7
    scale = torch.exp(torch.randn(d_r // 2, generator=g).clamp(-1.5, 1.5))

    pos = torch.stack((
        torch.tensor([0, 1, 2, 2, 2, 2, 3, 4, 5]),
        torch.tensor([0, 1, 2, 2, 3, 3, 3, 4, 5]),
        torch.tensor([0, 1, 2, 3, 2, 3, 3, 4, 5]),
    ))
    cos, sin = reference.mrope_phase(pos, d_r, 10_000_000.0, (12, 10, 10))

    k_pre = reference.rotate(
        apply_rope_pair_transform(k_raw, theta, scale, rope_kind="interleaved", inverse_scale=False).unsqueeze(1),
        cos,
        sin,
    ).squeeze(1)
    k_post = apply_rope_pair_transform(
        reference.rotate(k_raw.unsqueeze(1), cos, sin).squeeze(1),
        theta,
        scale,
        rope_kind="interleaved",
        inverse_scale=False,
    )
    torch.testing.assert_close(k_pre, k_post, rtol=1e-5, atol=1e-5)

    for rope_kind in ("interleaved", "split_half"):
        q_rot = reference.rotate(q_raw, cos, sin) if rope_kind == "interleaved" else q_raw
        k_rot = reference.rotate(k_raw.unsqueeze(1), cos, sin).squeeze(1) if rope_kind == "interleaved" else k_raw
        ref_scores = torch.einsum("thd,sd->hts", q_rot, k_rot)
        q_comp = apply_rope_pair_transform(q_rot, theta, scale, rope_kind=rope_kind, inverse_scale=True)
        k_trans = apply_rope_pair_transform(k_rot, theta, scale, rope_kind=rope_kind, inverse_scale=False)
        comp_scores = torch.einsum("thd,sd->hts", q_comp, k_trans)
        torch.testing.assert_close(comp_scores, ref_scores, rtol=1e-5, atol=1e-5)


def test_default_sink4_local128_protection_fit_and_none_alternative(tmp_path: Path):
    """Correction 2: sink4_local128 is default in calibration, keeps P_T exact, fails closed on 100%-covered short seqs."""
    m = protection_mask(148)
    assert m is not None
    assert int(m.sum().item()) == 132
    assert bool(m[:4].all()) and bool(m[20:].all()) and not bool(m[4:20].any())
    assert protection_mask(148, protection="none") is None
    assert FitConfig().protection == DEFAULT_PROTECTION_POLICY == "sink4_local128"

    _, _, _, _, short_bank = _make_skewed_hy4_bank(99, layer_id=0, d_c=64, d_r=64, seqs=4, base_seq_len=16)
    with pytest.raises(ValueError, match="covers 100%"):
        fit_content_layer(short_bank, bits=4, config=FitConfig(steps=2, val_every=2))

    _, _, _, _, long_bank = _make_skewed_hy4_bank(101, layer_id=0, d_c=64, d_r=64, seqs=4, base_seq_len=148)
    eye = torch.eye(64)
    ones = torch.ones(64)
    loss_prot = float(content_output_loss(long_bank.val, long_bank.params, long_bank.geometry, eye, ones, 2, ste=False).item())
    loss_unprot = float(
        content_output_loss(
            long_bank.val, long_bank.params, long_bank.geometry, eye, ones, 2, ste=False, protection="none"
        ).item()
    )
    assert 0.0 < loss_prot < loss_unprot

    cfg_prot = FitConfig(
        steps=20,
        val_every=10,
        content_rot_lr=5e-2,
        content_scale_lr=5e-2,
        rope_angle_lr=5e-2,
        rope_scale_lr=5e-2,
        seed=3,
        protection="sink4_local128",
        sink_tokens=4,
        recent_tokens=128,
    )
    res_c = fit_content_layer(long_bank, bits=2, config=cfg_prot)
    assert res_c.protection_calibrated == "sink4_local128"
    assert res_c.orth_max_err < 1e-5
    assert res_c.equiv_max_rel_err < 1e-4
    assert res_c.best_val_loss < res_c.baselines["rtn_native_val"]
    assert res_c.best_val_loss <= res_c.baselines["smoothquant_alpha_val"] + 1e-9

    res_r = fit_rope_layer(long_bank, bits=4, config=cfg_prot)
    assert res_r.protection_calibrated == "sink4_local128"
    assert res_r.equiv_max_rel_err < 1e-4
    assert res_r.best_val_loss < res_r.baselines["rtn_val"]

    cfg_none = FitConfig(
        steps=20,
        val_every=10,
        content_rot_lr=5e-2,
        content_scale_lr=5e-2,
        rope_angle_lr=5e-2,
        rope_scale_lr=5e-2,
        seed=3,
        protection="none",
    )
    res_none = fit_content_layer(short_bank, bits=2, config=cfg_none)
    assert res_none.protection_calibrated == "none"
    assert res_none.best_val_loss < res_none.baselines["rtn_native_val"]

    _, _, _, _, bank_consumer = _make_skewed_hy4_bank(103, layer_id=3, d_c=64, d_r=64, seqs=4, base_seq_len=16)
    joint_loss = content_output_loss(
        short_bank.val,
        short_bank.params,
        short_bank.geometry,
        res_none.r_c,
        res_none.s_c,
        bits=4,
        ste=False,
        protection="none",
        extra_consumers=((bank_consumer.params, bank_consumer.geometry, bank_consumer.val),),
    )
    assert torch.isfinite(joint_loss) and float(joint_loss.item()) >= 0.0

    id_key = fit_config_digest(cfg_prot, extra={"layer": 0})
    save_fit_checkpoint(tmp_path, res_c, identity_key=id_key)
    reloaded = load_fit_checkpoint(tmp_path, kind="content", bits=2, layer_id=0, identity_key=id_key)
    assert reloaded is not None and reloaded.protection_calibrated == "sink4_local128"
    assert torch.equal(reloaded.r_c, res_c.r_c) and torch.equal(reloaded.s_c, res_c.s_c)
    with pytest.raises(ValueError, match="identity mismatch"):
        load_fit_checkpoint(tmp_path, kind="content", bits=2, layer_id=0, identity_key="stale-identity")


def test_attention_error_weights_tokens_equally():
    rng = torch.Generator().manual_seed(715)
    ref = torch.randn(3, 12, generator=rng)
    gains = torch.arange(1., 4.).view(-1, 1)
    err = _token_relative_squares(ref * gains, ref)
    torch.testing.assert_close(err, (gains[:, 0] - 1).square())


def test_hy4_flash_two_component_default_single_file_sidecar_mc_section_and_incremental_publish(tmp_path: Path):
    """End-to-end cassowary path: 2-component default (agentic+exl3_standard), sink4_local128, single mla_calibration.safetensors + cuteafd-mc.json['qmla']."""
    _, _, _, _, bank0 = _make_skewed_hy4_bank(201, layer_id=0, d_c=64, d_r=64, seqs=4, base_seq_len=24)
    _, _, _, _, bank5 = _make_skewed_hy4_bank(205, layer_id=5, d_c=64, d_r=64, seqs=4, base_seq_len=24)

    parser = build_parser()
    pilot_args = parser.parse_args([
        "--checkpoint", str(tmp_path),
        "--source-revision", "a" * 40,
        "--output", str(tmp_path / "out"),
        "--pilot",
        "--layers", "0,5",
    ])
    resolved = resolve_run_settings(pilot_args, (0, 1, 5, 9, 21, 41))
    assert resolved["layers"] == (0, 5)
    assert resolved["modes"] == ((4, 4),)
    assert resolved["panel_revision"] == DEFAULT_QMLA_PANEL_REVISION
    assert resolved["components"] == ("agentic", "exl3_standard")
    assert resolved["protection"] == "sink4_local128"
    assert resolved["train_sequences"] == 16 and resolved["val_sequences"] == 4
    assert parse_layers_arg("all-dsa", (0, 1, 5)) == (0, 1, 5)
    assert parse_modes_arg("4:4,2:4,4:0") == ((4, 4), (2, 4), (4, 0))

    source_rev = "b" * 40
    source = {
        "repo": "wrldsuksgo2mars/Hy4-flash-EXL3-K4-MC",
        "revision": source_rev,
        "base_repo": "tencent/Hy4-flash",
        "base_revision": "c" * 40,
        "family": "hy4_flash",
        "model_type": "hy_v4_flash",
        "config_sha256": "d" * 64,
        "checkpoint_kind": "mc",
        "calibrated_view": "original",
    }
    run_out = tmp_path / "run"
    result = calibrate_hy4_flash_from_banks(
        {0: bank0, 5: bank5},
        output_dir=run_out,
        source=source,
        modes=((4, 4), (2, 4)),
        corpus_provenance=_synthetic_panel_provenance(include_long_context=False),
        fit_config=FitConfig(steps=6, val_every=3, seed=0, protection="sink4_local128", sink_tokens=2, recent_tokens=6),
    )
    sidecar_dir = Path(result["sidecar_dir"])
    assert (sidecar_dir / "mla_calibration.safetensors").is_file()
    assert not (sidecar_dir / "mla_calibration.json").exists()
    assert result["manifest"]["artifact_status"] == "calibrated_unqualified"
    assert result["manifest"]["applicable_views"] == ["original"]
    report = result["report"]
    assert report["sidecar_bytes"] == (sidecar_dir / "mla_calibration.safetensors").stat().st_size
    assert set(report["combinations"]) == {"4:4", "2:4"}
    for mode, metrics in report["combinations"].items():
        layer_rows = metrics["per_layer"]
        assert set(layer_rows) == {"0", "5"}
        assert metrics["layer_tokens"] == sum(s.seq_len for bank in [bank0, bank5] for s in bank.val)
        assert metrics["ratio_to_rtn"] == pytest.approx(metrics["attn_rel_err"] / metrics["rtn_attn_rel_err"])
        for name in ("attn_rel_err", "rtn_attn_rel_err", "hadamard_attn_rel_err", "smoothquant_attn_rel_err"):
            expected = sum(row[name] * row["tokens"] for row in layer_rows.values()) / metrics["layer_tokens"]
            assert metrics[name] == pytest.approx(expected)
            assert metrics[name] == result["manifest"]["calibration_provenance"]["combination_measurements"][mode][name]
    assert report["fit_by_layer"]["0"]["content"]["2"]["fit_seconds"] > 0
    assert result["manifest"]["protection_policy"] == "sink4_local128"
    assert result["manifest"]["protection_scope"] == PROTECTION_SCOPE
    assert sorted(result["manifest"]["supported_modes"]) == sorted(["4:0", "2:0", "0:4", "4:4", "2:4"])
    assert result["manifest"]["calibration_provenance"]["dataset_revision"] == DEFAULT_QMLA_PANEL_REVISION
    assert sorted(result["manifest"]["calibration_provenance"]["components"].keys()) == [
        "agentic",
        "exl3_standard",
    ]

    re_read = read_sidecar_manifest(sidecar_dir)
    assert re_read == result["manifest"]

    mc_root = tmp_path / "mc"
    mc_root.mkdir()
    index_orig = {"weight_map": {"shared.w": "model-00001.safetensors", "core.w": "core-orig.safetensors"}}
    index_k8 = {"weight_map": {"shared.w": "model-00001.safetensors", "core.trellis": "core-k8.safetensors"}}
    for fname, obj in (
        ("model.safetensors.index.json", index_orig),
        ("config.json", {"model_type": "hy_v4_flash"}),
        ("quantize_config.json", {"bits": 4}),
        ("model.k8.safetensors.index.json", index_k8),
        ("config.k8.json", {"model_type": "hy_v4_flash"}),
        ("quantize_config.k8.json", {"bits": 4, "core_bits": 8}),
    ):
        (mc_root / fname).write_text(json.dumps(obj), encoding="utf-8")
    mc_doc = mc_manifest.build(
        mc_root,
        source_repo="tencent/Hy4-flash",
        source_revision="c" * 40,
        core_tensors={"original": ["core.w"], "k8-compact": ["core.trellis"]},
    )
    updated_mc = attach_qmla_to_mc(mc_doc, sidecar_dir)
    assert updated_mc.get("required_features", []) == []
    assert updated_mc["shared_inventory_sha256"] == mc_doc["shared_inventory_sha256"]
    assert updated_mc["variants"] == mc_doc["variants"]
    mc_manifest.validate(updated_mc, root=mc_root)
    validate_sidecar(sidecar_dir, mc_manifest=updated_mc, strict=True)

    bad_mc = json.loads(json.dumps(updated_mc))
    bad_mc["qmla"]["artifact_status"] = "runtime_qualified"
    with pytest.raises(ValueError, match="disagrees with mla_calibration.safetensors"):
        validate_sidecar(sidecar_dir, mc_manifest=bad_mc, strict=True)

    remote_tree = {
        "model-00001.safetensors": {"bytes": 4_000_000_000, "blob": "9" * 64},
        "core-orig.safetensors": {"bytes": 2_000_000_000, "blob": "8" * 64},
        "cuteafd-mc.json": {"bytes": 1234, "blob": "7" * 40},
        "config.json": {"bytes": 512, "blob": "6" * 40},
    }
    plan = plan_incremental_revision(
        source["repo"],
        source_rev,
        sidecar_dir,
        remote_tree=remote_tree,
        existing_mc_manifest=mc_doc,
        stage_dir=tmp_path / "stage",
        validation_report_path=run_out / "qmla-validation-report.json",
        expected_source={"repo": source["repo"], "revision": source_rev, "family": "hy4_flash"},
    )
    assert plan["reuploaded_shard_bytes"] == 0
    assert plan["preserved_shards_count"] == 2
    assert set(plan["additions"]) == {
        "mla_calibration.safetensors",
        "cuteafd-mc.json",
        "qmla-validation-report.json",
    }
    dry_receipt = publish_incremental_revision(
        plan, api=None, state_root=tmp_path / "pub-state", dry_run=True
    )
    assert dry_receipt["status"] == "dry_run"
    assert dry_receipt["reuploaded_shard_bytes"] == 0
    assert dry_receipt["preserved_shards_count"] == 2

    st_path = sidecar_dir / "mla_calibration.safetensors"
    corrupted = bytearray(st_path.read_bytes())
    corrupted[-1] ^= 0x01
    st_path.write_bytes(bytes(corrupted))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_sidecar(sidecar_dir, strict=True)


def test_long_context_recalibration_and_offline_fake_api_non_mc_incremental_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """3-component recalibration (new panel revision + long_context) + non-MC single-file publish."""
    from safetensors.torch import save_file
    import hub_lfs
    from qmla import publish as qmla_publish

    corpus_dir = tmp_path / "panel_v3"
    corpus_dir.mkdir()
    g = torch.Generator().manual_seed(313)
    for comp in ("agentic", "exl3_standard", "long_context"):
        s0 = torch.randint(10, 1000, (12,), generator=g, dtype=torch.int64)
        s1 = torch.randint(10, 1000, (15,), generator=g, dtype=torch.int64)
        save_file(
            {"input_ids": torch.cat((s0, s1)), "offsets": torch.tensor([0, 12, 27], dtype=torch.int64)},
            str(corpus_dir / f"{comp}.safetensors"),
        )
    recs, prov = load_pinned_qmla_corpus(
        corpus_dir, panel_revision="e" * 40, components=("agentic", "exl3_standard", "long_context")
    )
    assert len(recs) == 6
    assert prov["dataset_revision"] == "e" * 40
    assert sorted(prov["components"].keys()) == ["agentic", "exl3_standard", "long_context"]

    _, _, _, _, bank0 = _make_skewed_hy4_bank(317, layer_id=0, d_c=64, d_r=64, seqs=4, base_seq_len=20)
    parent_sha = "1" * 40
    new_sha = "2" * 40
    source = {
        "repo": "tencent/Hy4-flash",
        "revision": parent_sha,
        "base_repo": "tencent/Hy4-flash",
        "base_revision": parent_sha,
        "family": "hy4_flash",
        "model_type": "hy_v4_flash",
        "config_sha256": "d" * 64,
        "checkpoint_kind": "plain",
        "calibrated_view": "original",
    }
    res = calibrate_hy4_flash_from_banks(
        {0: bank0},
        output_dir=tmp_path / "cal",
        source=source,
        modes=((4, 4),),
        corpus_provenance=prov,
        fit_config=FitConfig(steps=6, val_every=3, seed=0, protection="sink4_local128", sink_tokens=2, recent_tokens=4),
    )
    sidecar_dir = Path(res["sidecar_dir"])

    initial_tree = {
        "model-00001.safetensors": {"bytes": 5_000_000_000, "blob": "a" * 64},
        "config.json": {"bytes": 400, "blob": "b" * 40},
    }
    plan = plan_incremental_revision(
        source["repo"],
        parent_sha,
        sidecar_dir,
        remote_tree=initial_tree,
        existing_mc_manifest=None,
        expected_source={"repo": source["repo"], "revision": parent_sha, "family": "hy4_flash"},
    )
    assert set(plan["additions"]) == {"mla_calibration.safetensors"}

    class _FakeApi:
        def __init__(self, head_sha: str):
            self.head = head_sha
            self.trees = {parent_sha: dict(initial_tree)}
            self.committed_paths: list[str] = []

        def repo_info(self, repo_id, *, repo_type):
            return SimpleNamespace(sha=self.head, private=False)

        def create_commit(self, *, repo_id, repo_type, operations, commit_message, parent_commit, num_threads):
            assert parent_commit == parent_sha
            updated = dict(self.trees[parent_sha])
            for op in operations:
                self.committed_paths.append(op.path_in_repo)
                row = plan["additions"][op.path_in_repo]
                updated[op.path_in_repo] = {
                    "bytes": row["bytes"],
                    "blob": row["sha256"] if op.path_in_repo.endswith(".safetensors") else row["git_blob"],
                }
            self.trees[new_sha] = updated
            self.head = new_sha
            return SimpleNamespace(oid=new_sha)

    fake_api = _FakeApi(parent_sha)
    monkeypatch.setattr(qmla_publish, "require_qualified_hub", lambda: None)
    monkeypatch.setattr(hub_lfs, "remote_tree", lambda api, repo, rev: api.trees[rev])

    pub = publish_incremental_revision(plan, api=fake_api, state_root=tmp_path / "pub_ok", dry_run=False)
    assert pub["status"] == "published"
    assert pub["new_revision"] == new_sha
    assert pub["reuploaded_shard_bytes"] == 0
    assert fake_api.committed_paths == ["mla_calibration.safetensors"]
    assert "cuteafd-mc.json" not in fake_api.trees[new_sha]
    assert fake_api.trees[new_sha]["model-00001.safetensors"] == initial_tree["model-00001.safetensors"]

    moved_api = _FakeApi("f" * 40)
    with pytest.raises(ValueError, match="head moved"):
        publish_incremental_revision(plan, api=moved_api, state_root=tmp_path / "pub_moved", dry_run=False)


def test_pure_nope_sidecar_disallows_nonzero_rope_modes_and_rejects_unknown_components(tmp_path: Path):
    """GLM 5.3 Flash pattern: pure NoPE (qk_rope_dim=0) emits C:0 only and rejects arbitrary component names."""
    from qmla import CapturedLayerBank, CapturedLayerParams, CapturedSequence, write_sidecar

    g = torch.Generator().manual_seed(409)
    d_c, heads, nope, vdim, hidden = 64, 2, 8, 8, 12
    geom = MlaLayerGeometry(
        layer_id=3, kv_source_layer=3, kind="dsa", heads=heads,
        kv_lora_rank=d_c, qk_nope_dim=nope, qk_rope_dim=0, v_head_dim=vdim, rope_kind="none",
    )
    params = CapturedLayerParams(
        kv_norm_weight=torch.ones(d_c),
        kv_b_weight=torch.randn(heads * (nope + vdim), d_c, generator=g),
        o_proj_weight=torch.randn(hidden, heads * vdim, generator=g),
    )

    def _seq(sid: str, t: int):
        u = torch.randn(t, d_c, generator=g)
        q_nope = torch.randn(t, heads, nope, generator=g)
        stub = CapturedSequence(
            seq_id=sid, positions=torch.arange(t, dtype=torch.int64),
            u=u, q_nope=q_nope, q_rope=None, k_rope=None, allowed_mask=None, gate=None,
            ref_output=torch.zeros(t, hidden),
        )
        ref = eval_attention_output(stub, params, geom)
        return CapturedSequence(
            seq_id=sid, positions=stub.positions, u=u, q_nope=q_nope,
            q_rope=None, k_rope=None, allowed_mask=None, gate=None, ref_output=ref,
        )

    bank = CapturedLayerBank(
        geometry=geom,
        params=params,
        train=tuple(_seq(f"tr_{i}", 20) for i in range(4)),
        val=tuple(_seq(f"va_{i}", 20) for i in range(2)),
        provenance=_synthetic_panel_provenance(),
    )
    res_c4 = fit_content_layer(
        bank,
        bits=4,
        config=FitConfig(steps=6, val_every=3, seed=0, protection="sink4_local128", sink_tokens=2, recent_tokens=4),
    )
    base_prov = {
        **_synthetic_panel_provenance(include_long_context=False),
        "train_sequences": 4,
        "val_sequences": 2,
        "optimizer": "adamw",
        "steps_per_path": 6,
        "seed": 0,
        "code_commit": "test",
    }
    source = {
        "repo": "wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K4-MC",
        "revision": "5" * 40,
        "family": "glm5_flash",
        "model_type": "glm5_next",
        "config_sha256": "6" * 64,
        "checkpoint_kind": "mc",
        "calibrated_view": "original",
    }
    # Reject arbitrary unknown component names (no pluggable arbitrary list).
    bad_prov = json.loads(json.dumps(base_prov))
    bad_prov["components"]["arbitrary_extra"] = {"sequences": 1, "tokens": 100, "input_ids_sha256": "9" * 64}
    with pytest.raises(ValueError, match="unknown entries"):
        write_sidecar(
            tmp_path / "bad",
            source=source,
            geometries=[geom],
            content_profiles={4: [res_c4]},
            rope_profiles={},
            provenance=bad_prov,
        )

    manifest = write_sidecar(
        tmp_path / "ok",
        source=source,
        geometries=[geom],
        content_profiles={4: [res_c4]},
        rope_profiles={},
        provenance=base_prov,
    )
    assert manifest["supported_modes"] == ["4:0"]
    assert "*:4" in manifest["unsupported_modes_reason"]
    assert manifest["protection_policy"] == "sink4_local128"
    assert not (tmp_path / "ok" / "mla_calibration.json").exists()
    validate_sidecar(tmp_path / "ok", expected_geometries=[geom], banks_by_layer={3: bank}, strict=True)

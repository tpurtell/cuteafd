"""CPU numeric oracle tests for shared QMLA quantizer, fit, sidecar, publish, and Hy4-flash pipeline."""
from __future__ import annotations

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

import mc_manifest
from hy4_flash.forward import Layer, Model, reference
from hy4_flash.qmla_capture import Hy4FlashCaptureAdapter
from hy4_flash.qmla_run import (
    build_parser,
    calibrate_hy4_flash_from_banks,
    parse_corpus_component_arg,
    parse_layers_arg,
    parse_modes_arg,
    resolve_run_settings,
)
from qmla import (
    CorpusComponentSpec,
    FitConfig,
    MlaLayerGeometry,
    apply_content_forward,
    apply_rope_pair_transform,
    attach_qmla_advisory_to_mc,
    cayley_orthogonal,
    compensate_kv_b,
    content_output_loss,
    eval_attention_output,
    eval_positional_qk,
    fit_config_digest,
    fit_content_layer,
    fit_rope_layer,
    load_fit_checkpoint,
    pack_content_row,
    pack_joint_record,
    pack_rope_row,
    parse_qmla_mode,
    plan_incremental_revision,
    publish_incremental_revision,
    qdq_asymmetric_group,
    record_bytes_for_mode,
    save_fit_checkpoint,
    sylvester_hadamard,
    unpack_content_row,
    unpack_joint_record,
    unpack_rope_row,
    validate_sidecar,
)


def _make_skewed_hy4_bank(seed: int, *, layer_id: int = 0, d_c: int = 64, d_r: int = 64, seqs: int = 6):
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
        compressed_indexer_candidate_topk=32,
        compressed_indexer_num_attention_heads=2,
        compressed_indexer_head_dim=d_r,
    )
    kv_a = rand(d_c + d_r, hidden)
    # Inject strong outlier channels in content latent and frequency-pair outliers in RoPE key.
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
    inputs = [(f"seq_{i:02d}", rand(10 + (i % 3), hidden), None) for i in range(seqs)]
    bank = adapter.capture_layer(
        layer,
        inputs,
        train_count=seqs - 2,
        val_count=2,
        seed=seed,
        provenance={"synthetic": True},
    )
    return config, model, layer, adapter, bank


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

    # Full record size table verification (16-byte aligned).
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

    # Mode parser fail-closed rules.
    assert parse_qmla_mode("4:4") == (4, 4)
    assert parse_qmla_mode("0:0") == (0, 0)
    with pytest.raises(ValueError, match="INT2 RoPE"):
        parse_qmla_mode("4:2")
    with pytest.raises(ValueError):
        parse_qmla_mode("3:4")
    with pytest.raises(ValueError, match="NoPE"):
        record_bytes_for_mode(4, 4, d_r=0)

    # Hadamard orthogonality oracle.
    h = sylvester_hadamard(64)
    torch.testing.assert_close(h @ h.T, torch.eye(64), atol=1e-6, rtol=1e-6)


def test_rope_pair_transform_commutes_with_3d_interleaved_mrope_and_split_half():
    g = torch.Generator().manual_seed(29)
    t, heads, d_r = 9, 3, 64
    q_raw = torch.randn(t, heads, d_r, generator=g)
    k_raw = torch.randn(t, d_r, generator=g)
    theta = torch.randn(d_r // 2, generator=g) * 0.7
    scale = torch.exp(torch.randn(d_r // 2, generator=g).clamp(-1.5, 1.5))

    # Non-degenerate 3D interleaved M-RoPE positions (T != H != W).
    pos = torch.stack((
        torch.tensor([0, 1, 2, 2, 2, 2, 3, 4, 5]),
        torch.tensor([0, 1, 2, 2, 3, 3, 3, 4, 5]),
        torch.tensor([0, 1, 2, 3, 2, 3, 3, 4, 5]),
    ))
    cos, sin = reference.mrope_phase(pos, d_r, 10_000_000.0, (12, 10, 10))

    # Commutation check: RoPE(T(x)) == T(RoPE(x)) for both K and Q.
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

    # Positional QK equivalence for both interleaved and split-half.
    for rope_kind in ("interleaved", "split_half"):
        q_rot = reference.rotate(q_raw, cos, sin) if rope_kind == "interleaved" else q_raw
        k_rot = reference.rotate(k_raw.unsqueeze(1), cos, sin).squeeze(1) if rope_kind == "interleaved" else k_raw
        ref_scores = torch.einsum("thd,sd->hts", q_rot, k_rot)
        q_comp = apply_rope_pair_transform(q_rot, theta, scale, rope_kind=rope_kind, inverse_scale=True)
        k_trans = apply_rope_pair_transform(k_rot, theta, scale, rope_kind=rope_kind, inverse_scale=False)
        comp_scores = torch.einsum("thd,sd->hts", q_comp, k_trans)
        torch.testing.assert_close(comp_scores, ref_scores, rtol=1e-5, atol=1e-5)


def test_content_and_rope_fit_improve_over_rtn_and_preserve_equivalence(tmp_path: Path):
    _, _, _, _, bank = _make_skewed_hy4_bank(101, layer_id=0, d_c=64, d_r=64, seqs=6)
    cfg = FitConfig(steps=40, val_every=10, content_rot_lr=5e-2, content_scale_lr=5e-2, rope_angle_lr=5e-2, rope_scale_lr=5e-2, seed=3)

    res_c = fit_content_layer(bank, bits=2, config=cfg)
    assert res_c.orth_max_err < 1e-5
    assert res_c.equiv_max_rel_err < 1e-4
    assert res_c.best_val_loss < res_c.baselines["rtn_native_val"]
    assert res_c.best_val_loss <= res_c.baselines["smoothquant_alpha_val"] + 1e-9

    res_r = fit_rope_layer(bank, bits=4, config=cfg)
    assert res_r.equiv_max_rel_err < 1e-4
    assert res_r.best_val_loss < res_r.baselines["rtn_val"]

    # Shared-KV multi-consumer content loss (Mistral Large 4 pattern) evaluates cleanly.
    _, _, _, _, bank_consumer = _make_skewed_hy4_bank(103, layer_id=3, d_c=64, d_r=64, seqs=6)
    joint_loss = content_output_loss(
        bank.val,
        bank.params,
        bank.geometry,
        res_c.r_c,
        res_c.s_c,
        bits=4,
        ste=False,
        extra_consumers=((bank_consumer.params, bank_consumer.geometry, bank_consumer.val),),
    )
    assert torch.isfinite(joint_loss) and float(joint_loss.item()) >= 0.0

    # Fit checkpoint XXH3-128 save/resume and identity fail-closed.
    id_key = fit_config_digest(cfg, extra={"layer": 0})
    save_fit_checkpoint(tmp_path, res_c, identity_key=id_key)
    reloaded = load_fit_checkpoint(tmp_path, kind="content", bits=2, layer_id=0, identity_key=id_key)
    assert reloaded is not None and torch.equal(reloaded.r_c, res_c.r_c) and torch.equal(reloaded.s_c, res_c.s_c)
    with pytest.raises(ValueError, match="identity mismatch"):
        load_fit_checkpoint(tmp_path, kind="content", bits=2, layer_id=0, identity_key="stale-identity")


def test_hy4_flash_end_to_end_pilot_sidecar_mc_advisory_and_incremental_publish(tmp_path: Path):
    _, _, _, _, bank0 = _make_skewed_hy4_bank(201, layer_id=0, d_c=64, d_r=64, seqs=6)
    _, _, _, _, bank5 = _make_skewed_hy4_bank(205, layer_id=5, d_c=64, d_r=64, seqs=6)

    # CLI pilot vs full flag resolution + pluggable corpus component parser.
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
    assert resolved["train_sequences"] == 16 and resolved["val_sequences"] == 4

    comp_desc = parse_corpus_component_arg(
        "long_context_16k=wrldsuksgo2mars/qmla-long-context-v1@1234567890abcdef1234567890abcdef12345678:/tmp/lc16k"
    )
    assert comp_desc["name"] == "long_context_16k"
    assert parse_layers_arg("all-dsa", (0, 1, 5)) == (0, 1, 5)
    assert parse_modes_arg("4:4,2:4,4:0") == ((4, 4), (2, 4), (4, 0))

    # End-to-end calibration with 3 pluggable components (base two + optional long-context).
    source_rev = "b" * 40
    components = [
        CorpusComponentSpec(
            name="agentic",
            repo="wrldsuksgo2mars/agentic-quant-panel-v2",
            revision="54683731eb01c7c2fa5b801a0888e029a0e1b1c7",
            sequences=4,
            tokens=44,
            input_ids_sha256="1" * 64,
        ),
        CorpusComponentSpec(
            name="exl3_standard",
            repo="wrldsuksgo2mars/agentic-quant-panel-v2",
            revision="54683731eb01c7c2fa5b801a0888e029a0e1b1c7",
            sequences=2,
            tokens=22,
            input_ids_sha256="2" * 64,
        ),
        CorpusComponentSpec(
            name="long_context_16k",
            repo="wrldsuksgo2mars/qmla-long-context-v1",
            revision="3" * 40,
            sequences=2,
            tokens=32768,
            input_ids_sha256="4" * 64,
        ),
    ]
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
        corpus_components=components,
        fit_config=FitConfig(steps=10, val_every=5, seed=0),
    )
    sidecar_dir = Path(result["sidecar_dir"])
    assert result["manifest"]["artifact_status"] == "calibrated_unqualified"
    assert sorted(result["manifest"]["supported_modes"]) == sorted(["4:0", "2:0", "0:4", "4:4", "2:4"])
    assert [c["name"] for c in result["manifest"]["calibration_provenance"]["components"]] == [
        "agentic",
        "exl3_standard",
        "long_context_16k",
    ]

    # Attach advisory qmla pointer to a valid cuteafd-mc.json and confirm mc_manifest accepts it
    # without touching required_features or shared_inventory_sha256.
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
    updated_mc = attach_qmla_advisory_to_mc(mc_doc, sidecar_dir)
    assert updated_mc.get("required_features", []) == []
    assert updated_mc["shared_inventory_sha256"] == mc_doc["shared_inventory_sha256"]
    assert updated_mc["variants"] == mc_doc["variants"]
    mc_manifest.validate(updated_mc, root=mc_root)

    # Same-repo incremental publish plan + dry-run: preserves all existing weight shards (0 re-uploaded).
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
        "mla_calibration.json",
        "cuteafd-mc.json",
        "qmla-validation-report.json",
    }
    dry_receipt = publish_incremental_revision(
        plan, api=None, state_root=tmp_path / "pub-state", dry_run=True
    )
    assert dry_receipt["status"] == "dry_run"
    assert dry_receipt["reuploaded_shard_bytes"] == 0
    assert dry_receipt["preserved_shards_count"] == 2

    # Tampering with mla_calibration.safetensors must fail sidecar validation closed.
    st_path = sidecar_dir / "mla_calibration.safetensors"
    corrupted = bytearray(st_path.read_bytes())
    corrupted[-1] ^= 0x01
    st_path.write_bytes(bytes(corrupted))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_sidecar(sidecar_dir, strict=True)


def test_token_stream_component_loader_and_offline_fake_api_incremental_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from safetensors.torch import save_file
    from huggingface_hub.hf_api import RepoFile
    from hy4_flash.qmla_run import load_token_stream_component
    import hub_lfs
    from qmla import publish as qmla_publish

    # 1. Pluggable token-stream component loader (e.g. optional long-context bank).
    comp_dir = tmp_path / "lc16k"
    comp_dir.mkdir()
    g = torch.Generator().manual_seed(313)
    seq0 = torch.randint(10, 1000, (12,), generator=g, dtype=torch.int64)
    seq1 = torch.randint(10, 1000, (15,), generator=g, dtype=torch.int64)
    save_file(
        {"input_ids": torch.cat((seq0, seq1)), "offsets": torch.tensor([0, 12, 27], dtype=torch.int64)},
        str(comp_dir / "calibration.safetensors"),
    )
    recs, spec = load_token_stream_component(
        "long_context_16k", "wrldsuksgo2mars/qmla-long-context-v1", "e" * 40, comp_dir
    )
    assert len(recs) == 2 and spec.tokens == 27 and spec.sequences == 2
    assert recs[0][0] == "long_context_16k:00000" and torch.equal(recs[0][1], seq0)

    # 2. Build a real sidecar from a synthetic Hy4-flash bank.
    _, _, _, _, bank0 = _make_skewed_hy4_bank(317, layer_id=0, d_c=64, d_r=64, seqs=6)
    parent_sha = "1" * 40
    new_sha = "2" * 40
    source = {
        "repo": "wrldsuksgo2mars/Hy4-flash-EXL3-K4-MC",
        "revision": parent_sha,
        "base_repo": "tencent/Hy4-flash",
        "base_revision": "c" * 40,
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
        corpus_components=[spec],
        fit_config=FitConfig(steps=6, val_every=3, seed=0),
    )
    sidecar_dir = Path(res["sidecar_dir"])

    # 3. Offline fake HfApi verifying parent guard, zero shard re-upload, and post-commit tree check.
    initial_tree = {
        "model-00001.safetensors": {"bytes": 5_000_000_000, "blob": "a" * 64},
        "config.json": {"bytes": 400, "blob": "b" * 40},
    }
    plan = plan_incremental_revision(
        source["repo"],
        parent_sha,
        sidecar_dir,
        remote_tree=initial_tree,
        expected_source={"repo": source["repo"], "revision": parent_sha, "family": "hy4_flash"},
    )

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

    pub = publish_incremental_revision(
        plan, api=fake_api, state_root=tmp_path / "pub_ok", dry_run=False
    )
    assert pub["status"] == "published"
    assert pub["new_revision"] == new_sha
    assert pub["reuploaded_shard_bytes"] == 0
    assert set(fake_api.committed_paths) == {"mla_calibration.safetensors", "mla_calibration.json"}
    # Original weight shard preserved untouched.
    assert fake_api.trees[new_sha]["model-00001.safetensors"] == initial_tree["model-00001.safetensors"]

    # Guard against moved remote head.
    moved_api = _FakeApi("f" * 40)
    with pytest.raises(ValueError, match="head moved"):
        publish_incremental_revision(plan, api=moved_api, state_root=tmp_path / "pub_moved", dry_run=False)


def test_pure_nope_sidecar_disallows_nonzero_rope_modes(tmp_path: Path):
    """GLM 5.3 Flash pattern: pure NoPE (qk_rope_dim=0) emits C:0 modes only."""
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
        train=tuple(_seq(f"tr_{i}", 8) for i in range(4)),
        val=tuple(_seq(f"va_{i}", 8) for i in range(2)),
        provenance={"synthetic": True},
    )
    res_c4 = fit_content_layer(bank, bits=4, config=FitConfig(steps=6, val_every=3, seed=0))
    manifest = write_sidecar(
        tmp_path,
        source={
            "repo": "wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K4-MC",
            "revision": "5" * 40,
            "family": "glm5_flash",
            "model_type": "glm5_next",
            "config_sha256": "6" * 64,
            "checkpoint_kind": "mc",
            "calibrated_view": "original",
        },
        geometries=[geom],
        content_profiles={4: [res_c4]},
        rope_profiles={},
        provenance={
            "components": [
                CorpusComponentSpec(
                    name="agentic", repo="wrldsuksgo2mars/agentic-quant-panel-v2",
                    revision="54683731eb01c7c2fa5b801a0888e029a0e1b1c7",
                    sequences=6, tokens=48, input_ids_sha256="7" * 64,
                )
            ],
            "train_sequences": 4,
            "val_sequences": 2,
            "input_ids_sha256": "7" * 64,
            "optimizer": "adamw",
            "steps_per_path": 6,
            "seed": 0,
            "code_commit": "test",
        },
    )
    assert manifest["supported_modes"] == ["4:0"]
    assert "*:4" in manifest["unsupported_modes_reason"]
    validate_sidecar(tmp_path, expected_geometries=[geom], banks_by_layer={3: bank}, strict=True)


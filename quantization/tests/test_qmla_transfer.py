"""K8 transfer must use authentic packed view operands and the frozen test token hashes."""
import os
from pathlib import Path

import pytest
import torch

from hy4_flash import forward, recipe
from inkling2 import numerical
from hy4_flash.qmla_mc import MCModel, MCReader
from hy4_flash.qmla_transfer import summarize_transfer, transfer_records
from qmla.capture import describe_splits, sequence_sha256
from qmla.sidecar import attach_view_transfer, load_sidecar_transforms
from test_qmla_pipeline import _make_skewed_hy4_bank, _synthetic_panel_provenance
from hy4_flash.qmla_run import calibrate_hy4_flash_from_banks
from qmla.fit import FitConfig


def test_transfer_matches_ordered_test_tokens_and_preserves_sidecar(tmp_path):
    rng = torch.Generator().manual_seed(1)
    corpus = [(str(i), torch.randint(100, (17,), generator=rng)) for i in range(5)]
    selected = {"train": corpus[1:3], "val": corpus[3:4], "test": [corpus[4], corpus[0]]}
    split = describe_splits({k: [(i, sequence_sha256(t)) for i, t in v] for k, v in selected.items()}, seed=42)
    records = transfer_records(corpus, {"splits": split})
    assert [r.seq_id for r in records] == ["4", "0"]
    assert all(torch.equal(r.input_ids, tokens) for r, (_, tokens) in zip(records, selected["test"], strict=True))
    bad = [*corpus[:-1], ("4", corpus[-1][1] + 1)]
    with pytest.raises(ValueError, match="test tokens differ"):
        transfer_records(bad, {"splits": split})

    _, _, _, _, bank = _make_skewed_hy4_bank(3)
    source = dict(repo=recipe.TARGET_REPO, revision="a" * 40, base_repo=recipe.BASE_REPO,
                  base_revision=recipe.BASE_REVISION, family="hy4_flash", checkpoint_kind="mc",
                  model_type="hy_v4_flash", config_sha256="b" * 64, calibrated_view="original")
    out = calibrate_hy4_flash_from_banks(
        {0: bank}, source=source, output_dir=tmp_path, modes=((4, 4),),
        corpus_provenance=_synthetic_panel_provenance(), fit_config=FitConfig(steps=1, protection="none"),
    )
    path = out["sidecar_dir"]
    before, content, rope = load_sidecar_transforms(path)
    section = dict(view="k8-compact", reference="k8-compact 0:0", split="test", source_revision=source["revision"],
                   test_digest=before["calibration_provenance"]["splits"]["test"]["digest"],
                   transform_fingerprint=before["transform_fingerprint"],
                   combinations=before["calibration_provenance"]["combination_measurements"])
    with pytest.raises(ValueError, match="differs"):
        attach_view_transfer(path, "k8-compact", {**section, "test_digest": "a" * 64})
    attached = attach_view_transfer(path, "k8-compact", section)
    assert attached["tensors_sha256"] != before["tensors_sha256"]
    assert attached["applicable_views"] == ["original", "k8-compact"]
    assert attached["view_transfer_validation"]["k8-compact"] == section
    assert attached == attach_view_transfer(path, "k8-compact", section)
    after, actual_c, actual_r = load_sidecar_transforms(path)
    assert after["transform_fingerprint"] == before["transform_fingerprint"]
    for old, new in zip((*content[0], *rope[0]), (*actual_c[0], *actual_r[0]), strict=True):
        assert torch.equal(old, new)


@pytest.mark.parametrize("view", ["original", "k8-compact"])
def test_installed_mc_view_dispatches_its_core_and_shared_k4_experts(monkeypatch, view):
    hub = Path(os.environ.get("HF_HUB_CACHE", Path(os.environ.get("HF_HOME", Path.home()/".cache/huggingface"))/"hub"))
    checkpoint = hub / "models--wrldsuksgo2mars--Hy4-flash-EXL3-K4-MC/snapshots/cc72179113573d751ed3245e3e58d3640694378d"
    if not checkpoint.is_dir():
        pytest.skip("pinned Hy4-flash MC snapshot is absent")
    reader = MCReader(checkpoint, "cpu", view=view)
    assert reader.source["revision"] == recipe.BASE_REVISION
    # Exercise real packed scale/marker reads and the same validator used before CUDA decode.
    dispatched = []
    def verify(packed, shape, *, device, dtype, codebook):
        info = numerical.validate_projection(packed, shape, codebook=codebook)
        dispatched.append((info["bits"], device, dtype, codebook))
        return torch.empty(shape, device="meta", dtype=dtype)

    monkeypatch.setattr(numerical, "reconstruct_projection", verify)
    first_projection = "model.layers.1.self_attn.q_a_proj"
    if view == "original":
        name = first_projection + ".weight"
        native = reader.raw(name)
        assert native.dtype == torch.bfloat16 and name in reader.core
        assert torch.equal(native.float(), reader.load(name))
        assert reader.variant["fingerprint"] == reader.original_fingerprint
    else:
        reader.reconstruct(first_projection, (1024, 3072), 8)
        assert reader.variant["fingerprint"] != reader.original_fingerprint
    reader.reconstruct("model.layers.1.mlp.experts.237.gate_proj", (1536, 1536), 4)
    assert [call[0] for call in dispatched] == ([4] if view == "original" else [8, 4])
    assert all(call[1:] == ("cpu", torch.bfloat16, "mul1") for call in dispatched)
    with pytest.raises(ValueError, match="wrong bit width"):
        reader.reconstruct("model.layers.1.mlp.experts.237.gate_proj", (1536, 1536), 8)
    # Check the physical packed operands and their bits without loading a CUDA extension.
    calls = {}
    def inspect(module, shape, bits):
        t = reader.descriptor(module + ".trellis")
        assert t["shape"] == [shape[1] // 16, ((shape[0] + 127) // 128) * 8, 16 * bits]
        if bits == 8:
            assert module + ".trellis" in reader.core
        else:
            assert bits == 4 and module + ".trellis" not in reader.core
        calls[module] = bits
        return torch.empty(shape, device="meta")

    monkeypatch.setattr(reader, "reconstruct", inspect)
    monkeypatch.setattr(reader, "load", lambda name, **kw: torch.empty(reader.descriptor(name)["shape"], device="meta"))
    monkeypatch.setattr(reader, "raw", lambda name: torch.empty(reader.descriptor(name)["shape"], device="meta"))
    model = MCModel(reader)
    assert reader.maps  # The real packed reads above opened these shards.
    layer = model.load_layer(1)
    assert not reader.maps  # Mocked weights did not open any new maps.
    if view == "original":
        assert calls == {}
        assert "self_attn.kv_b_proj.weight" in layer.weights
    else:
        assert calls["model.layers.1.self_attn.kv_b_proj"] == 8
    assert forward.reference.TABLE in layer.weights
    assert all(key not in layer.weights for key in forward.reference.EXPERTS)
    latent = torch.empty((2, reader.config.routed_expert_hidden_size), device="meta")
    weights = model.routed_expert_weights(237, latent, layer)
    assert all(calls[f"model.layers.1.mlp.experts.237.{proj}_proj"] == 4 for proj in ("gate", "up", "down"))
    assert weights[0].shape == (reader.config.moe_intermediate_size, latent.shape[-1])
    assert weights[2].shape == weights[0].T.shape


def test_transfer_summary_requires_the_complete_measured_test_grid():
    fields = dict(attn_rel_err=0.1, rtn_attn_rel_err=0.2, hadamard_attn_rel_err=0.3, ratio_to_rtn=0.5)
    observed = {**fields, "layer_tokens": 10, "per_layer": {"0": {**fields, "tokens": 10}}}
    identity = dict(modes=["4:4"], layers=[0])
    results = dict(combinations={"4:4": observed}, wall_seconds=1.0)
    assert summarize_transfer(results, identity, 10)["combinations"]["4:4"] == observed
    with pytest.raises(ValueError, match="coverage"):
        summarize_transfer(results, identity, 9)
    observed["attn_rel_err"] = 0.2
    with pytest.raises(ValueError, match="invalid K8"):
        summarize_transfer(results, identity, 10)

"""CPU contract checks for GLM Flash's isolated calibrated K8 target core."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glm5_flash.core_k6 import core_only, recipe, run, storage
from kimi_k3.uniform_k2.storage import read_header


def geometry(bits=8):
    return dict(in_features=128, out_features=128, stored_out_features=128,
                bits=bits, group="base_core")


def packed_fixture(path, name, g):
    tensors = {key: torch.ones(shape, dtype={"I16": torch.int16,
        "I32": torch.int32, "F16": torch.float16}[dtype])
        for key, (dtype, shape) in recipe.packed_specs(name, g).items()}
    tensors[name + ".mcg"] = torch.tensor(recipe.MCG_MULTIPLIER,
        dtype=torch.uint32).view(torch.int32)
    save_file(tensors, path)
    return path


def shared_export():
    # Parent owns core_export.py; tests can point PYTHONPATH to its worktree
    # before that shared integration lands in this branch.
    return pytest.importorskip("core_export")


def test_default_recipe_unchanged_and_k8_only_changes_target_bits():
    default = recipe.reviewed_modules()
    assert len(default) == 37476
    assert len([g for g in default.values() if g["bits"] == 6]) == 316
    assert len([g for g in default.values() if g["bits"] == 4]) == 37160
    selection = core_only.core_selection(8)
    assert len(selection) == recipe.BASE_CORE_MODULE_COUNT == 316
    assert selection.keys() == recipe.reviewed_core_modules(include_mtp=False).keys()
    assert all(g["bits"] == 8 and g["group"] == "base_core" for g in selection.values())
    assert all(".experts." not in n and ".layers.45." not in n for n in selection)
    assert selection["lm_head"]["bits"] == 8
    for name, g in selection.items():
        assert {key: value for key, value in g.items() if key != "bits"} == {
            key: value for key, value in default[name].items() if key != "bits"}
        assert recipe.packed_specs(name, g)[name + ".trellis"][1][-1] == 128
    mtp = {n: g for n, g in recipe.reviewed_core_modules(core_bits=8).items()
           if g["group"] == "mtp_core"}
    assert len(mtp) == 8 and all(g["bits"] == 4 for g in mtp.values())


@pytest.mark.parametrize("bits", [True, 4, 7, 9, "8", 8.])
def test_unreviewed_core_bits_rejected(bits):
    with pytest.raises(ValueError, match="core"):
        core_only.core_selection(bits)
    with pytest.raises(ValueError, match="core"):
        recipe.reviewed_core_modules(core_bits=bits)


def test_native_experts_are_loaded_never_quantized_or_spooled():
    calls = []
    source = SimpleNamespace(materialize_routed_experts=lambda *a, **kw: calls.append((a, kw)))
    engine = object.__new__(core_only.CoreOnlyEngine)
    engine.source, engine.device = source, "cpu"
    engine.progress = SimpleNamespace(emit=lambda *a, **kw: None)
    block = SimpleNamespace(mlp=SimpleNamespace(experts=torch.nn.Linear(2, 2)))
    old = block.mlp.experts.weight.clone()
    engine._experts(block, 3, object(), object())
    assert calls == [((block,), dict(layer=3, device="cpu"))]
    assert torch.equal(block.mlp.experts.weight, old)


def test_core_shards_have_only_four_tensors_and_k8_storage(tmp_path):
    helper = shared_export()
    names = [recipe.PREFIX + "layers.0.self_attn.q_proj", "lm_head"]
    selection = {name: geometry() for name in names}
    export = tmp_path / "export"
    identity = dict(test="core-only")
    engine = object.__new__(core_only.CoreOnlyEngine)
    engine.selection, engine.export, engine.identity = selection, export, identity
    path = packed_fixture(tmp_path / "projection.safetensors", names[0], selection[names[0]])
    engine.assemble_layer(0, {names[0]: path})
    head = packed_fixture(tmp_path / "head.safetensors", names[1], selection[names[1]])
    helper.assemble_core_shard({names[1]: head}, {names[1]: selection[names[1]]},
        export, "core-head.safetensors", identity)
    summary = helper.finalize_core_export(selection, export, identity)
    assert summary["modules"] == 2 and summary["tensors"] == 8 and summary["bits"] == [8]
    index = json.loads((export / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == {name + "." + suffix for name in names
                                      for suffix in recipe.SUFFIXES}
    config = json.loads((export / "quantize_config.json").read_text())
    assert all(g["bits_per_weight"] == 8 for g in config["tensor_storage"].values())
    assert not any(name.endswith(".weight") for name in index["weight_map"])
    assert not any("experts" in name or "visual" in name for name in index["weight_map"])
    for path in export.glob("*.safetensors"):
        header, _ = read_header(path)
        assert len(set(header) - {"__metadata__"}) == 4


def test_scalar_reconstruction_and_packed_oracle_durable_on_resume(tmp_path, monkeypatch):
    helper = shared_export()
    name, g = "lm_head", geometry()
    calls = []
    monkeypatch.setattr(helper, "sampled_packed_oracle", lambda path, module, geometry, **kw:
        calls.append((path, module, geometry["bits"])) or
        dict(passed=True, bits=8, sampled_sha256="test", comparison="test-independent-exact"))
    dense = torch.full((128, 128), .5, dtype=torch.bfloat16)

    class Numerical:
        @staticmethod
        def quantize_projection(weight, raw, **kw):
            assert kw["bits"] == 8 and raw["count"] == 1426
            path = packed_fixture(tmp_path / "fake-quantizer.safetensors", name, g)
            packed = {key.removeprefix(name + "."): value for key, value in load_file(path).items()}
            return packed, dict(hessian_weighted_relative_error=.001), dict(hessian_rows=1426, identity_rows=0)

        @staticmethod
        def reconstruct_projection(packed, shape, **kw):
            assert packed["trellis"].shape[-1] == 128 and shape == [128, 128]
            return dense.clone()

    def store():
        return storage.CoreProjectionStore(tmp_path / "projections", tmp_path / "evidence",
            {"core_bits": 8}, {name: g}, SimpleNamespace(emit=lambda *a, **kw: None), Numerical, "cpu")

    first = store()
    first.ensure(name, dense, dict(count=1426))
    assert torch.equal(first.dense(name), dense)
    saved = json.loads((tmp_path / "evidence/lm_head.json").read_text())
    assert saved["evidence"]["packed_byte_oracle"]["passed"] is True
    assert saved["evidence"]["reconstruction"]["max_abs"] == .5
    assert saved["metrics"]["hessian_weighted_relative_error"] == .001
    second = store()
    assert second.load(name)["bits"] == 8
    assert len(calls) == 1
    assert json.loads((tmp_path / "evidence/lm_head.json").read_text()) == saved
    # Missing evidence must run the real oracle again, not trust an old receipt.
    (tmp_path / "evidence/lm_head.json").unlink()
    store().load(name)
    assert len(calls) == 2


def test_failed_packed_byte_oracle_cannot_be_accepted(tmp_path, monkeypatch):
    helper = shared_export()
    monkeypatch.setattr(helper, "sampled_packed_oracle", lambda *a, **kw: (_ for _ in ()).throw(
        ValueError("packed-byte reference decode differs")))
    name, g = "lm_head", geometry()
    packed_fixture(tmp_path / "packed.safetensors", name, g)
    store = storage.CoreProjectionStore(tmp_path / "projections", tmp_path / "evidence",
        {}, {name: g}, None, None, "cpu")
    with pytest.raises(ValueError, match="packed-byte"):
        store.record_evidence(name, dict(sha256="bad", evidence={}, module=name,
            bits=8, logical_shape=[128, 128], metrics={}))
    assert not (tmp_path / "evidence/lm_head.json").exists()


def test_cli_opt_in_does_not_change_default_full_export(monkeypatch, tmp_path):
    observed = []
    monkeypatch.setattr(core_only, "run_core_only", lambda args: observed.append(args))
    argv = [str(Path(run.__file__)), "--source-snapshot", str(tmp_path / "source"),
        "--tokens", str(tmp_path / "tokens"), "--root", str(tmp_path / "run"),
        "--export", str(tmp_path / "export")]
    monkeypatch.setattr(sys, "argv", argv + ["--core-only", "--core-bits", "8", "--preflight"])
    run.main()
    assert observed[0].core_only is True and observed[0].core_bits == 8 and observed[0].preflight
    monkeypatch.setattr(sys, "argv", argv + ["--core-bits", "8"])
    with pytest.raises(SystemExit) as error:
        run.main()
    assert error.value.code == 2
    monkeypatch.setattr(sys, "argv", argv + ["--core-only", "--publish-only"])
    with pytest.raises(SystemExit):
        run.main()
    def full_source(snapshot):
        raise RuntimeError("legacy-full-export-selected")
    monkeypatch.setitem(sys.modules, "gptqmodel.utils.glm5_flash_core",
        SimpleNamespace(GlmFlashBF16Source=full_source))
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="legacy-full-export-selected"):
        run.main()
    assert len(observed) == 1


def test_preflight_does_not_touch_cuda_or_native_tensors(tmp_path, monkeypatch):
    shared_export()
    source = SimpleNamespace(close=lambda: None)
    monkeypatch.setitem(sys.modules, "gptqmodel.utils.glm5_flash_core",
        SimpleNamespace(GlmFlashBF16Source=lambda _: source))
    monkeypatch.setattr(core_only, "make_core_plan", lambda *a, **kw: dict(required_free_bytes=0))
    def forbidden(*a, **kw):
        raise AssertionError("preflight touched CUDA")
    monkeypatch.setattr(torch.cuda, "get_device_properties", forbidden)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", forbidden)
    core_only.run_core_only(SimpleNamespace(source_snapshot=tmp_path / "source",
        tokens=tmp_path / "tokens", root=tmp_path / "run", export=tmp_path / "export",
        core_bits=8, preflight=True, resume=False))
    assert not (tmp_path / "run/plan.json").exists()
    assert not (tmp_path / "export").exists()


def test_finalization_requires_reconstruction_and_oracle_for_every_module(tmp_path):
    name, g = "lm_head", geometry()
    identity = dict(calibration=dict(tokens=1097737))
    evidence = dict(identity=identity, module=name, bits=8, logical_shape=[128, 128],
        evidence=dict(identity_rows=0, hessian_rows=1097737,
            packed_byte_oracle=dict(passed=True, bits=8),
            reconstruction=dict(finite=True, logical_shape=[128, 128])))
    directory = tmp_path / "evidence/head"
    directory.mkdir(parents=True)
    path = directory / "lm_head.json"
    path.write_text(json.dumps(evidence))
    core_only.verify_core_evidence({name: g}, tmp_path, identity)
    for key in ("reconstruction", "packed_byte_oracle"):
        broken = deepcopy(evidence)
        broken["evidence"].pop(key)
        path.write_text(json.dumps(broken))
        with pytest.raises(ValueError, match="incomplete"):
            core_only.verify_core_evidence({name: g}, tmp_path, identity)


def test_core_plan_requires_full_calibration_and_hashes_fresh_recipe(tmp_path, monkeypatch):
    source_dir = tmp_path / recipe.BASE_REVISION
    source_dir.mkdir()
    (source_dir / "config.json").write_text("{}")
    (source_dir / "model.safetensors.index.json").write_text("{}")
    tokens = tmp_path / "tokens"
    tokens.mkdir()
    manifest = dict(splits=dict(calibration=dict(count=1426, tokens=1097737)))
    (tokens / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(recipe, "inventory", lambda _: {})
    monkeypatch.setattr(core_only, "sha256_file", lambda path: "test-" + path.name)
    monkeypatch.setattr(core_only.importlib.metadata, "version", lambda _: "test")
    monkeypatch.setattr(core_only.subprocess, "check_output", lambda *a, **kw: "test-pin")
    source = SimpleNamespace(snapshot=source_dir)
    before = deepcopy(run.RECIPE)
    plan = run.make_plan(source, tokens, tmp_path / "run", tmp_path / "export",
        core_bits=8, core_only=True)
    assert run.RECIPE == before
    assert plan["inventory"]["selected_modules"] == 316
    assert plan["inventory"]["target_tensors"] == 1264
    assert plan["inventory"]["preserved_bytes"] == 0
    assert plan["inventory"]["selected_source_bytes"] == sum(
        g["in_features"] * g["out_features"] * 2 for g in core_only.core_selection(8).values())
    assert set(plan["inventory"]["parts"]) == {"base_core"}
    assert plan["publication_repo"] is None
    assert plan["identity"]["calibration"] == dict(samples=1426, tokens=1097737)
    assert plan["identity"]["recipe"]["core_bits"] == 8
    assert plan["identity"]["recipe"]["expert_bits"] is None
    assert "quantization/core_export.py" in plan["identity"]["code"]
    manifest["splits"]["calibration"]["count"] = 32
    (tokens / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="full"):
        run.make_plan(source, tokens, tmp_path / "run", tmp_path / "export", core_bits=8, core_only=True)
    with pytest.raises(ValueError, match="core-only"):
        run.make_plan(source, tokens, tmp_path / "run", tmp_path / "export", core_bits=8)

"""CPU contracts for calibrated Step-5 K8 target-core production and smoke."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step5.core_k6 import core_only as core, recipe, run
from kimi_k3.uniform_k2.storage import read_header


def test_exact_reviewed_target_modules_and_unchanged_full_recipe():
    original = recipe.reviewed_modules()
    selected = core.core_selection(8)
    assert len(selected) == 645
    assert selected.keys() == recipe.reviewed_core_modules(include_mtp=False).keys()
    assert all(g["bits"] == 8 and g["group"] == "base_core" for g in selected.values())
    assert not any("experts" in name or ".layers.92." in name for name in selected)
    for name, geometry in selected.items():
        assert {k: v for k, v in geometry.items() if k != "bits"} == {
            k: v for k, v in original[name].items() if k != "bits"}
    assert recipe.reviewed_modules() == original
    assert len([g for g in original.values() if g["group"] == "mtp_core"]) == 27
    assert sum(recipe.packed_bytes(g) for g in selected.values()) < 16 * run.GIB


@pytest.mark.parametrize("bits", [True, 4, 7, 9, "8", 8.])
def test_unreviewed_core_bits_rejected(bits):
    with pytest.raises(ValueError, match="core"):
        core.core_selection(bits)


def test_native_routed_experts_are_loaded_without_quantizing_or_spooling():
    engine = object.__new__(core.CoreOnlyEngine)
    block = object()
    engine.source = SimpleNamespace(materialize_routed_experts=Mock())
    engine.device = "cpu"
    engine.progress = SimpleNamespace(emit=Mock())
    engine._experts(block, 3, object(), object())
    engine.source.materialize_routed_experts.assert_called_once_with(block, layer=3, device="cpu")


def test_core_assembly_cannot_include_native_or_expert_payloads(tmp_path):
    name = "model.layers.0.self_attn.q_proj"
    geometry = dict(in_features=128, out_features=128, stored_out_features=128,
                    bits=8, group="base_core")
    packed = {key: torch.ones(shape, dtype={"I16": torch.int16,
        "I32": torch.int32, "F16": torch.float16}[dtype])
        for key, (dtype, shape) in recipe.packed_specs(name, geometry).items()}
    packed[name + ".mcg"] = torch.tensor(recipe.MCG_MULTIPLIER,
        dtype=torch.uint32).view(torch.int32)
    candidate = tmp_path / "candidate.safetensors"
    save_file(packed, candidate)
    engine = object.__new__(core.CoreOnlyEngine)
    engine.selection, engine.identity, engine.export = {name: geometry}, {}, tmp_path / "out"
    engine.assemble_layer(0, {name: candidate})
    header, _ = read_header(engine.export / "core-layer-000.safetensors")
    assert set(header) - {"__metadata__"} == {name + "." + suffix for suffix in recipe.SUFFIXES}
    with pytest.raises(KeyError):
        engine.assemble_layer(0, {"model.layers.3.moe.experts.0.gate_proj": candidate})


def plan_fixture(tmp_path, monkeypatch):
    source_dir = tmp_path / recipe.BASE_REVISION
    source_dir.mkdir()
    tokens = tmp_path / "tokens"
    tokens.mkdir()
    manifest = dict(splits=dict(calibration=dict(count=1426, tokens=1105269)))
    (tokens / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(recipe, "inventory", lambda _: {})
    monkeypatch.setattr(core, "sha256_file", lambda path: "test-" + path.name)
    monkeypatch.setattr(core.importlib.metadata, "version", lambda _: "test")
    monkeypatch.setattr(core, "model_runtime", lambda: {"transformers_version": "model-test"})
    monkeypatch.setattr(core.subprocess, "check_output", lambda *a, **kw: "test-pin")
    source = SimpleNamespace(snapshot=source_dir,
        reference=SimpleNamespace(verify_runtime_source=lambda: {"pin": "transformers"}))
    return source, tokens, manifest


def test_smoke_identity_keeps_full_selection_and_numerics(tmp_path, monkeypatch):
    source, tokens, manifest = plan_fixture(tmp_path, monkeypatch)
    before = deepcopy(run.RECIPE)
    full = core.make_core_plan(source, tokens, tmp_path / "run", tmp_path / "out", core_bits=8)
    smoke = core.make_core_plan(source, tokens, tmp_path / "smoke", tmp_path / "smoke-out",
        core_bits=8, layer_limit=4)
    assert run.RECIPE == before
    assert full["inventory"] == smoke["inventory"]
    assert full["inventory"]["selected_modules"] == 645
    assert full["inventory"]["target_tensors"] == 2580
    assert full["inventory"]["preserved_bytes"] == 0
    assert full["identity"]["recipe"]["smoke_layer_limit"] is None
    assert smoke["identity"]["recipe"]["smoke_layer_limit"] == 4
    assert smoke["identity"]["recipe"]["hessian_accumulation"] == run.RECIPE["hessian_accumulation"]
    assert smoke["identity"]["runtime"] == {"pin": "transformers"}
    assert smoke["identity"]["encoder_delta"]["k6_gptqmodel_commit"] == "25dd11c391ce6bf3ceadf8722e8274c3cde7079a"
    assert "quantization/glm5_flash/core_k6/storage.py" in smoke["identity"]["code"]
    assert "python/reference/families/step5/golden.py" in smoke["identity"]["code"]
    assert full["required_free_bytes"] < 300 * run.GIB
    assert full["publication_repo"] is None
    for key, value in (("count", 2), ("tokens", 10)):
        invalid = deepcopy(manifest)
        invalid["splits"]["calibration"][key] = value
        (tokens / "manifest.json").write_text(json.dumps(invalid))
        with pytest.raises(ValueError, match="full Step-5"):
            core.make_core_plan(source, tokens, tmp_path / "run", tmp_path / "out", core_bits=8)


@pytest.mark.parametrize("limit", [0, -1, 93, True, "4", 1.5])
def test_invalid_layer_limit_rejected_before_inventory(limit):
    with pytest.raises(ValueError, match="layer-limit"):
        core.make_core_plan(None, None, None, None, core_bits=8, layer_limit=limit)


def test_cli_dispatch_is_opt_in_and_legacy_default_unchanged(monkeypatch, tmp_path):
    observed = []
    monkeypatch.setattr(core, "run_core_only", lambda args: observed.append(args))
    argv = [str(Path(run.__file__)), "--source-snapshot", str(tmp_path / "source"),
        "--tokens", str(tmp_path / "tokens"), "--root", str(tmp_path / "run"),
        "--export", str(tmp_path / "out")]
    monkeypatch.setattr(sys, "argv", argv + ["--core-only", "--core-bits", "8", "--layer-limit", "4"])
    run.main()
    assert observed[0].layer_limit == 4 and observed[0].core_bits == 8
    for flags in (["--core-bits", "8"], ["--layer-limit", "4"], ["--core-only", "--publish-only"]):
        monkeypatch.setattr(sys, "argv", argv + flags)
        with pytest.raises(SystemExit):
            run.main()
    def legacy(*a, **kw):
        raise RuntimeError("legacy source selected")
    monkeypatch.setitem(sys.modules, "gptqmodel.utils.step5_core", SimpleNamespace(Step5BF16Source=legacy))
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="legacy source"):
        run.main()


def test_preflight_never_touches_cuda_or_reads_weights(tmp_path, monkeypatch):
    source = SimpleNamespace(close=Mock())
    monkeypatch.setitem(sys.modules, "gptqmodel.utils.step5_core",
        SimpleNamespace(Step5BF16Source=lambda *a, **kw: source))
    monkeypatch.setattr(core, "make_core_plan", lambda *a, **kw: dict(required_free_bytes=0))
    monkeypatch.setattr(torch.cuda, "get_device_properties", Mock(side_effect=AssertionError("CUDA")))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", Mock(side_effect=AssertionError("CUDA")))
    core.run_core_only(SimpleNamespace(source_snapshot=tmp_path / "source", tokens=tmp_path / "tokens",
        root=tmp_path / "run", export=tmp_path / "out", core_bits=8, layer_limit=4,
        preflight=True, resume=False))
    assert not (tmp_path / "run/plan.json").exists()
    assert not (tmp_path / "out").exists()
    source.close.assert_called_once()


def test_smoke_runs_through_native_moe_and_never_head_or_finalize(tmp_path, monkeypatch):
    from contextlib import nullcontext
    import core_export
    from step5 import prepare_corpus
    source = SimpleNamespace(close=Mock())
    monkeypatch.setitem(sys.modules, "gptqmodel.utils.step5_core",
        SimpleNamespace(Step5BF16Source=lambda *a, **kw: source))
    plan = dict(required_free_bytes=0, identity=dict(calibration=dict(samples=1, tokens=2)))
    monkeypatch.setattr(core, "make_core_plan", lambda *a, **kw: plan)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=128 * run.GIB))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda _: None)
    monkeypatch.setattr(core, "exclusive_run", lambda _: nullcontext())
    monkeypatch.setattr(prepare_corpus, "load_token_records", lambda *a: [dict(input_ids=[1, 2])])
    events, layers, evidence = [], [], []
    progress = SimpleNamespace(state={}, emit=lambda event, **kw: events.append((event, kw)))
    monkeypatch.setattr(core, "Progress", lambda *a, **kw: progress)
    selected = core.core_selection(8)
    engine = SimpleNamespace(identity=plan["identity"], selection=selected,
        initial=lambda: {"layer": -1}, layer=lambda layer, current: layers.append(layer) or {"layer": layer},
        head=Mock(side_effect=AssertionError("smoke must not encode head")))
    monkeypatch.setattr(core, "CoreOnlyEngine", lambda *a: engine)
    monkeypatch.setattr(core, "verify_core_evidence", lambda selection, *a: evidence.append(selection))
    monkeypatch.setattr(core_export, "finalize_core_export", Mock(side_effect=AssertionError("smoke cannot finalize")))
    core.run_core_only(SimpleNamespace(source_snapshot=tmp_path / "source", tokens=tmp_path / "tokens",
        root=tmp_path / "run", export=tmp_path / "out", core_bits=8, layer_limit=4,
        preflight=False, resume=False))
    assert layers == [0, 1, 2, 3]
    assert len(evidence[0]) == 28 and "lm_head" not in evidence[0]
    receipt = json.loads((tmp_path / "run/smoke-complete.json").read_text())
    assert receipt["summary"]["native_expert_layers"] == [3]
    assert receipt["summary"]["full_export"] is False
    assert receipt["summary"]["packed_byte_oracle_passed"] is True
    assert not (tmp_path / "run/quantization-complete.json").exists()
    assert events[-1][0] == "core_smoke_complete"


def test_completion_demands_every_calibrated_reconstruction_and_oracle(tmp_path):
    name = "model.layers.3.self_attn.q_proj"
    geometry = dict(in_features=128, out_features=128, bits=8)
    identity = dict(calibration=dict(tokens=1105269))
    saved = dict(identity=identity, module=name, bits=8, logical_shape=[128, 128],
        evidence=dict(identity_rows=0, hessian_rows=1105269,
            packed_byte_oracle=dict(passed=True, bits=8),
            reconstruction=dict(finite=True, logical_shape=[128, 128])))
    path = tmp_path / "evidence/layer-003" / (name + ".json")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(saved))
    core.verify_core_evidence({name: geometry}, tmp_path, identity)
    for key in ("reconstruction", "packed_byte_oracle", "hessian_rows"):
        broken = deepcopy(saved)
        broken["evidence"].pop(key)
        path.write_text(json.dumps(broken))
        with pytest.raises(ValueError, match="incomplete"):
            core.verify_core_evidence({name: geometry}, tmp_path, identity)

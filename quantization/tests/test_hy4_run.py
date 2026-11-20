"""CPU checks for Hy4 causal encodes and bounded sequential recovery."""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hy4_flash import run
from hy4_flash.forward import Layer
from hy4_flash.storage import Routes, Stages
from kimi_k3.uniform_k2.storage import assemble_shard, read_header


def test_sealed_segment_supports_selective_replay_and_direct_assembly(tmp_path):
    store = Stages(tmp_path / "phases", {"test": True})
    packed = torch.arange(24, dtype=torch.int16).reshape(3, 8)
    first = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    second = first + 100
    manifest = store.write_batches("moe", [
        (0, dict(packed=packed, weighted=first, empty=torch.empty((2, 0), dtype=torch.int32))),
        (1, dict(packed=packed + 10, weighted=second)),
    ])
    assert manifest["entries"][0]["filename"] == manifest["entries"][1]["filename"]
    assert manifest["entries"][0]["hash_algorithm"] == "xxh3-128"

    reloaded = Stages(store.root, {"test": True})
    assert torch.equal(reloaded.fields("moe", 1, ("weighted",))["weighted"], second)
    assert reloaded.fields("moe", 0, ("empty",))["empty"].shape == (2, 0)
    refs = {f"p{n}": reloaded.reference("moe", n, "packed") for n in range(2)}
    target = tmp_path / "final.safetensors"
    assemble_shard(refs, target)
    from safetensors.torch import load_file
    observed = load_file(str(target))
    assert torch.equal(observed["p0"], packed)
    assert torch.equal(observed["p1"], packed + 10)
    with pytest.raises(ValueError, match="invalid safetensors record extent"):
        read_header(refs["p1"][0], offset=refs["p1"][2], length=2)


def test_prepared_but_uncommitted_segment_is_replayed(tmp_path):
    store = Stages(tmp_path, {"test": True})
    def interrupted():
        yield 0, {"value": torch.ones(2)}
        raise RuntimeError("preempted")

    with pytest.raises(RuntimeError, match="preempted"):
        store.write_batches("partial", interrupted(), segment_batches=1)
    assert store.load("partial") is None
    landed = store.write_batches("partial", [(0, {"value": torch.zeros(2)})], segment_batches=1)
    assert len(landed["entries"]) == 1
    assert torch.equal(store.read_batch("partial", 0)["value"], torch.zeros(2))


def test_stable_expert_dispatch_preserves_reference_rounding_order():
    batch = dict(latent=torch.tensor([[10.0], [20.0], [30.0]]),
                 ids=torch.tensor([[2, 1, 0], [0, 1, 2], [1, 2, 0]]),
                 gamma=torch.tensor([[.5, .3, .2], [.3, .2, .5], [.2, .5, .3]]))
    routes = Routes([(0, batch)], experts=3, tokens=3)
    for expert in range(3):
        rows, weights = routes.expert(expert)
        assert rows.tolist() == [0, 1, 2]
        assert torch.equal(weights, batch["gamma"][batch["ids"] == expert])
    outputs = [torch.full((3, 1), value) for value in (1e20, -1e20, 3.)]
    assert torch.equal(routes.combined(enumerate(outputs)), outputs[-1])
    with pytest.raises(ValueError, match="out of order"):
        routes.combined([(1, outputs[1])])


def test_both_encodes_receive_same_hessian_and_only_serialized_k8_replays(tmp_path, monkeypatch):
    name = "layer.gate_proj"
    inputs = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    source = torch.zeros(2, 2)
    runner = run.Runner.__new__(run.Runner)
    runner.device = torch.device("cpu")
    runner.core = {name: dict(in_features=2, out_features=2)}
    runner.records = [0, 1]
    runner.calibration_count = 1
    runner.native_modules = set()
    runner.stages = Stages(tmp_path / "stages", {"test": True})
    runner.progress = SimpleNamespace(event=lambda *args, **kwargs: None)
    attempts, reloads = [], []

    def quantize(weight, raw, *, bits, device, codebook):
        assert codebook == "mul1"
        assert weight is source
        attempts.append((bits, raw))
        return {suffix: torch.full((2, 2), bits) for suffix in run.recipe.SUFFIXES}, {"hessian_weighted_relative_error": 0.}, {"bits": bits}

    def reconstruct(packed, shape, *, device, dtype, codebook):
        assert codebook == "mul1"
        reloads.append(dtype)
        return packed["trellis"].to(dtype)

    numerical = SimpleNamespace(quantize_projection=quantize, reconstruct_projection=reconstruct,
                                covariance_accumulator=run.numerical.covariance_accumulator)
    monkeypatch.setattr(run, "numerical", numerical)
    layer = Layer(7, "layer.", {"gate_proj.weight": source})
    overrides = {}
    fit = runner.core_fit("phase", layer, [(name,)], lambda: [(inputs,), (inputs * 100,)], overrides)
    assert [bits for bits, _ in attempts] == [8, 6]
    assert attempts[0][1] is attempts[1][1]
    assert torch.equal(attempts[0][1]["H"], inputs.T @ inputs)
    assert torch.equal(overrides[name], torch.full((2, 2), 8, dtype=torch.float32))
    assert reloads == [torch.bfloat16]

    def should_not_run():
        raise AssertionError("already committed Hessian was replayed")

    runner.core_fit(fit, layer, [(name,)], should_not_run, overrides)
    assert len(attempts) == 2


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_resident_expert_replay_uses_only_sealed_weights_and_preserves_sum(tmp_path, monkeypatch, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is required for the former CPU versus resident GPU reduction witness")

    monkeypatch.setattr(run, "EXPERT_GROUP", 1)
    monkeypatch.setattr(run, "EXPERT_ROWS", 2)
    monkeypatch.setattr(run.recipe, "EXPERTS_PER_LAYER", 3)
    batch = dict(latent=torch.tensor([[1., -2.], [-3., 4.], [5., 6.], [7., -8.]]),
                 ids=torch.tensor([[2, 1], [0, 1], [1, 2], [0, 2]]),
                 gamma=torch.tensor([[.5, .3], [.3, .2], [.2, .5], [.9, .2]]))
    cpu_routes = Routes([(0, batch)], experts=3, tokens=4)
    near = torch.tensor([[0], [2], [0], [1]], dtype=torch.int32)
    device_routes = Routes([(0, batch | dict(candidate_ids=near, candidate_gamma=torch.ones(4, 1)))],
                           experts=3, tokens=4, device=device, recover=True)
    layer = SimpleNamespace(prefix="model.layers.1.", index=1)
    runner = run.Runner.__new__(run.Runner)
    runner.device = torch.device(device)
    runner.stages = Stages(tmp_path / "stages", {"test": True})
    runner.progress = SimpleNamespace(base={}, event=lambda *args, **kwargs: None)
    runner.model = SimpleNamespace(config=SimpleNamespace(swiglu_limit=10., moe_intermediate_size=2))
    weights = {
        expert: tuple(torch.tensor([[expert + 1, p - 2], [3 - p, -expert]], dtype=torch.int16)
                      for p in range(3))
        for expert in range(3)
    }
    runner.expert_witness = None
    runner.reader = SimpleNamespace(expert_weights=lambda _, expert: weights[expert])
    encodes = []

    def packed(name, weight):
        return {f"k4.{name}.trellis": weight.to(torch.int16).cpu(),
                f"k4.{name}.suh": torch.ones(2, dtype=torch.float16),
                f"k4.{name}.svh": torch.ones(2, dtype=torch.float16),
                f"k4.{name}.mul1": torch.tensor(1, dtype=torch.int32)}

    def encode(name, weight, covariance, bits, device, metrics):
        encodes.append(name)
        metrics.append(dict(module=name, bits=bits))
        return packed(name, weight)

    def reconstruct(values, name, bits, shape, device):
        assert bits == 4 and tuple(shape) == (2, 2)
        return values[f"k4.{name}.trellis"].to(device=device, dtype=torch.float32)

    monkeypatch.setattr(run, "encode", encode)
    monkeypatch.setattr(run, "reconstruction", reconstruct)
    run.reference.configure_f32()
    legacy_parts = []
    with torch.inference_mode():
        for expert, matrices in weights.items():
            rows, coefficients = cpu_routes.expert(expert)
            names = runner.expert_names(layer, expert)
            gate, up, down = (w.to(device=device, dtype=torch.float32) for w in matrices)
            chunks = []
            for row_indices, gamma in zip(rows.split(2), coefficients.split(2), strict=True):
                inputs = cpu_routes.latent[row_indices].to(device)
                gated = run.native_projection(names[0], inputs, gate).clamp_max(10.)
                lifted = run.native_projection(names[1], inputs, up).clamp(-10., 10.)
                activated = torch.nn.functional.silu(gated) * lifted
                chunks.append((run.native_projection(names[2], activated, down) * gamma.to(device)[:, None]).cpu())
            legacy_parts.append(torch.cat(chunks))

        saved_names = runner.expert_names(layer, 0)
        legacy_record = {"weighted": torch.full_like(legacy_parts[0], float("nan"))}
        for name, weight in zip(saved_names, weights[0], strict=True):
            legacy_record.update(packed(name, weight))
        legacy = runner.stages.write_batches("mixed-expert-000", [(0, legacy_record)],
            {"metrics": [dict(module=name, bits=4) for name in saved_names]})
        expected = cpu_routes.combined(enumerate(legacy_parts))
        references, errors, stages, observed = runner.experts("mixed", layer, device_routes)
        assert torch.equal(observed.cpu().view(torch.int32), expected.view(torch.int32))
        assert len(references) == 36 and len(errors) == 9 and len(encodes) == 6
        assert runner.stages.load(stages[0]) == legacy
        for stage in stages[1:]:
            assert "weighted" not in runner.stages.load(stage)["entries"][0]["tensors"]

        _, _, _, replayed = runner.experts("mixed", layer, device_routes)
        assert torch.equal(replayed.cpu().view(torch.int32), expected.view(torch.int32))
        assert len(encodes) == 6


def test_resume_requires_same_frozen_identity(tmp_path):
    fresh = {"schema": "cuteafd-hy4-causal-mc-v2", "codebook": "mul1"}
    assert run.checkpoint_identity(tmp_path, fresh) == fresh
    assert run.checkpoint_identity(tmp_path, fresh) == fresh
    with pytest.raises(ValueError, match="numerical identity"):
        run.checkpoint_identity(tmp_path, fresh | {"codebook": "mcg"})


def test_activation_segment_recovery_seeks_producer_and_honors_byte_limit(tmp_path):
    store = Stages(tmp_path / "states", {"panel": "v2"})
    record = {"state": torch.arange(64, dtype=torch.float32)}
    from hy4_flash.storage import record_bytes
    size = record_bytes(record)
    generated = []

    def crash(start):
        assert start == 0
        for i in range(3):
            generated.append(i)
            yield i, record
        raise RuntimeError("interrupted")

    # Two records fit, a third must open a fresh segment before writing.
    with pytest.raises(RuntimeError, match="interrupted"):
        store.write_activations("wave", crash, 5, segment_bytes=2 * size + 1)
    directory = store.directory("wave")
    sealed = directory / "segment-00000.bin"
    first_inode = sealed.stat().st_ino
    assert sealed.stat().st_size == 2 * size
    assert (directory / "segment-00001.bin.partial").stat().st_size == size
    assert store.load("wave") is None

    recovered = Stages(store.root, {"panel": "v2"})
    def finish(start):
        assert start == 2
        for i in range(start, 5):
            generated.append(i)
            yield i, {"state": record["state"] + i}

    manifest = recovered.write_activations("wave", finish, 5, segment_bytes=2 * size + 1)
    assert first_inode == sealed.stat().st_ino
    assert generated == [0, 1, 2, 2, 3, 4]
    assert all(e["file_identity"]["bytes"] <= 2 * size + 1 for e in manifest["entries"])
    assert [i for i, _ in recovered.records("wave", start=3)] == [3, 4]
    for i in range(5):
        assert torch.equal(recovered.read_batch("wave", i)["state"], record["state"] + (i if i >= 2 else 0))
    recovered.load("wave", checksum=True)


def test_activation_resume_refuses_changed_identity_and_oversize_activation(tmp_path):
    store = Stages(tmp_path / "states", {"panel": 1})
    value = {"v": torch.ones(80)}
    def crash(start):
        yield start, value
        raise RuntimeError("interrupted")
    with pytest.raises(RuntimeError, match="interrupted"):
        store.write_activations("wave", crash, 2, segment_batches=1)
    other = Stages(store.root, {"panel": 2})
    with pytest.raises(ValueError, match="provenance differs"):
        other.write_activations("wave", crash, 2)
    with pytest.raises(ValueError, match="limit"):
        store.write_activations("wide", lambda start: [(start, value)], 1, segment_bytes=32)
    # A single packed module is indivisible and owns its own file when oversize.
    packed = store.write_batches("weights", [(0, value), (1, value)], segment_bytes=32)
    assert packed["entries"][0]["filename"] != packed["entries"][1]["filename"]


def test_live_activation_writer_keeps_ownership_of_partial_tail(tmp_path):
    store = Stages(tmp_path, {"test": True})
    other = Stages(tmp_path, store.identity)
    def produce(start):
        yield 0, {"v": torch.ones(2)}
        with pytest.raises(BlockingIOError):
            other.write_activations("wave", produce, 2)
        yield 1, {"v": torch.zeros(2)}
    store.write_activations("wave", produce, 2)
    assert torch.equal(store.read_batch("wave", 0)["v"], torch.ones(2))


def test_source_group_separates_bf16_head_from_compact_views():
    runner = run.Runner.__new__(run.Runner)
    runner.core = {"lm_head": {}}
    runner.header = SimpleNamespace(specs={"lm_head.weight": ("BF16", [4, 2])},
                                    source_entries=set)
    runner.native_modules = set()
    assert runner.source_group("lm_head.") == (set(), {"lm_head.weight"})
    runner.native_modules = {"lm_head"}
    assert runner.source_group("lm_head.") == ({"lm_head.weight"}, set())

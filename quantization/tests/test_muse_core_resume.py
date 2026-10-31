"""Real disk-barrier recovery around tiny official Muse decoder layers.

Only trellis coding is replaced with a visible deterministic perturbation. The
official decoder, causal stage graph, projection receipts and rolling boundaries
are the production implementations, executing on the CPU.
"""
import builtins
import copy
import os
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("psutil")
transformers = pytest.importorskip("transformers")
if transformers.__version__ != "5.22.1":
    pytest.skip("Muse resume witnesses require official transformers 5.22.1", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from muse_spark.core_k6 import run
from muse_spark.core_k6.storage import ProjectionStore, StageStore
from glm5.core_k6.storage import RouteSpool as SharedRouteSpool
from kimi_k3.uniform_k2.storage import commit_safetensors
from safetensors.torch import load_file
from test_muse_core_numerical import block as official_block


class Crash(RuntimeError):
    pass


class Progress:
    def __init__(self):
        self.stop = None
        self.events = []

    def emit(self, event, **fields):
        self.events.append((event, fields))
        if self.stop == (event, fields.get("phase")):
            self.stop = None
            raise Crash(f"after durable {event} {fields.get('phase')}")


def fixture(tmp_path, monkeypatch):
    template = official_block(0)
    records = [dict(id="first", input_ids=[1, 2, 3, 4, 5]),
               dict(id="second", input_ids=[8, 9, 10, 11, 12, 13, 14])]
    hidden = [torch.randn(1, len(row["input_ids"]), 128, dtype=torch.bfloat16) for row in records]

    class Source:
        loads = 0

        def close(self):
            pass

    source = Source()

    def load_block(*args, **kwargs):
        source.loads += 1
        return copy.deepcopy(template)

    from transformers.models.muse_spark.modeling_muse_spark import MuseSparkTextRotaryEmbedding
    monkeypatch.setattr(run.numerical, "load_block", load_block)
    monkeypatch.setattr(run.numerical, "rotary", lambda *a, **k: MuseSparkTextRotaryEmbedding(template.config))
    monkeypatch.setattr(run.numerical, "materialize_routed_experts", lambda *a, **k: None)
    monkeypatch.setattr(run, "load_tensors", lambda path: {k: v.clone() for k, v in load_file(str(path)).items()})
    monkeypatch.setattr(run, "compact", lambda: None)
    monkeypatch.setattr(run, "range", lambda *a: builtins.range(16) if a == (256,) else builtins.range(*a), raising=False)

    selection = {}
    prefix = "model.language_model.layers.0."
    for name, linear in template.named_modules():
        if isinstance(linear, torch.nn.Linear) and name != "self_attn.gate_proj":
            selection[prefix + name] = dict(in_features=linear.in_features, out_features=linear.out_features, bits=6)
    for expert in range(16):
        for part in ("gate_proj", "up_proj", "down_proj"):
            selection[prefix + f"mlp.experts.{expert}.{part}"] = dict(in_features=128, out_features=128, bits=4)
    monkeypatch.setattr(run, "reviewed_modules", lambda: selection)
    calls = []
    active = {}
    real_hessian = run.codec.DenseRawHessian

    class Codec:
        @staticmethod
        def DenseRawHessian(width):
            return real_hessian(128, device="cpu")

        @staticmethod
        def quantize_projection(weight, raw, **kwargs):
            changed = (weight.float() * .875 + .005).bfloat16()
            calls.append(dict(module=active["name"], H=raw["H"].clone(),
                              count=raw["count"], identity_rows=raw.get("identity_rows", 0)))
            return dict(dense=changed), dict(hessian_weighted_relative_error=.01), dict(
                hessian_rows=raw["count"], identity_rows=raw.get("identity_rows", 0))

        @staticmethod
        def reconstruct_projection(packed, shape, **kwargs):
            assert tuple(packed["dense"].shape) == tuple(shape)
            return packed["dense"].clone().to(kwargs["device"])

    monkeypatch.setattr(run, "codec", Codec)

    class Store(ProjectionStore):
        def ensure(self, name, *args, **kwargs):
            active["name"] = name
            return super().ensure(name, *args, **kwargs)

    class Spool:
        @classmethod
        def create(cls, root, identity, tokens, batches, progress, layer):
            latest = {}

            def adapted():
                for index, state in batches:
                    latest.update(state)
                    yield index, None, dict(mlp_input=state["input"])

            def router(_hidden):
                return latest["selected"], None, latest["near"]

            value = SharedRouteSpool.create(root, identity, tokens, adapted(), router,
                progress, layer, width=128, experts=16, minimum=1024)
            original = value.hessian
            value.hessian = lambda expert, down=None: original(expert, down=down, device="cpu")
            return value

    monkeypatch.setattr(run, "RouteSpool", Spool)
    progress = Progress()
    engine = run.Engine(source, records, tmp_path / "run", tmp_path / "export", dict(run="test"), progress)

    def store(layer):
        return Store(engine.root / f"layers/{layer:03d}/projections", engine.root / "evidence" / f"layer-{layer:03d}",
            engine.identity, selection, progress, Codec, device="cpu")

    engine.store = store
    entries = []
    directory = engine.boundary.directory(-1)
    for index, row in enumerate(records):
        filename = f"batch-{index:05d}.safetensors"
        entries.append(dict(filename=filename, id=row["id"], tokens=len(row["input_ids"]),
            **commit_safetensors(directory / filename, dict(hidden=hidden[index]))))
    current = engine.boundary.commit(-1, entries)
    from muse_spark.core_k6 import export
    monkeypatch.setattr(export, "assemble_layer", lambda *a, **k: dict(payload_bytes=1))
    return engine, current, template, hidden, calls, source, progress


def restore_weights(engine, template):
    value = copy.deepcopy(template)
    store = engine.store(0)
    prefix = "model.language_model.layers.0."
    for name in engine.selection:
        relative = name.removeprefix(prefix)
        if ".experts." in relative:
            expert, part = relative.removeprefix("mlp.experts.").split(".")
            run.numerical.expert_weight(value, int(expert), part).copy_(store.dense(name))
        else:
            store.install(name, value.get_submodule(relative))
    return value


def changed_template(engine, template):
    value = copy.deepcopy(template)
    for name in engine.selection:
        relative = name.removeprefix("model.language_model.layers.0.")
        if ".experts." in relative:
            expert, part = relative.removeprefix("mlp.experts.").split(".")
            weight = run.numerical.expert_weight(value, int(expert), part)
            weight.copy_((weight.float() * .875 + .005).bfloat16())
        else:
            linear = value.get_submodule(relative)
            linear.weight = torch.nn.Parameter((linear.weight.float() * .875 + .005).bfloat16(), requires_grad=False)
    return value


def test_causal_stage_inputs_and_final_native_boundary(tmp_path, monkeypatch):
    engine, current, template, hidden, calls, _, _ = fixture(tmp_path, monkeypatch)
    from muse_spark.core_k6 import export
    monkeypatch.setattr(export, "assemble_layer", lambda *a, **k: (_ for _ in ()).throw(Crash("after F")))
    with pytest.raises(Crash, match="after F"):
        engine.layer(0, current)
    assert engine.boundary.current()["layer"] == -1
    stages = StageStore(engine.root / "layers/000/stages", engine.identity)
    assert stages.load("f", checksum=True) is not None
    value = restore_weights(engine, template)
    captured = {key: [] for key in ("self_attn.o_proj", "mlp.pre_expert_proj", "mlp.post_expert_proj")}
    handles = []
    for name in captured:
        handles.append(value.get_submodule(name).register_forward_pre_hook(
            lambda _m, args, key=name: captured[key].append(args[0].clone())))
    try:
        for index, state in engine.phase_batches(stages, "f"):
            expected = run.numerical.native_forward(value, dict(hidden=hidden[index]))["hidden"]
            assert torch.equal(state["hidden"], expected)
    finally:
        for handle in handles:
            handle.remove()
    by_name = {call["module"]: call for call in calls}
    prefix = "model.language_model.layers.0."
    for relative, values in captured.items():
        rows = torch.cat([x.reshape(-1, 128).double() for x in values])
        expected = rows.T @ rows
        actual = by_name[prefix + relative]["H"].double()
        assert torch.linalg.norm(actual - expected) / torch.linalg.norm(expected) < 1e-6
    assert len(calls) == len(engine.selection)
    assert all(c["identity_rows"] == 0 for c in calls if ".experts." not in c["module"])
    assert any(c["identity_rows"] > 0 for c in calls if ".experts." in c["module"])
    store = engine.store(0)
    for name in engine.selection:
        if ".experts." not in name:
            continue
        evidence = store.load(name)["evidence"]
        assert evidence["natural_rows"] + evidence["augmented_rows"] + evidence["identity_rows"] == evidence["hessian_rows"]
        assert evidence["effective_rows"] == evidence["hessian_rows"]
        assert evidence["selection_policy"] == "rank-major-then-original-corpus-row"


@pytest.mark.parametrize("phase", ["a", "b", "c", "e", "f"])
def test_resume_each_durable_phase_preserves_projections_and_matches_native(tmp_path, monkeypatch, phase):
    engine, current, template, hidden, calls, _, progress = fixture(tmp_path, monkeypatch)
    progress.stop = ("phase_committed", phase)
    with pytest.raises(Crash, match="after durable"):
        engine.layer(0, current)
    assert engine.boundary.current(checksum=True)["layer"] == -1
    committed = {c["module"] for c in calls}
    result = engine.layer(0, current)
    assert result["layer"] == 0
    assert len(calls) == len(engine.selection)
    assert len({c["module"] for c in calls}) == len(calls)
    assert committed <= {c["module"] for c in calls}
    assert not engine.boundary.directory(-1).exists()
    assert not list((engine.root / "layers/000/projections").glob("*.safetensors"))
    changed = changed_template(engine, template)
    for index in range(len(hidden)):
        actual = engine.boundary_batch(result, index)["hidden"]
        expected = run.numerical.native_forward(changed, dict(hidden=hidden[index]))["hidden"]
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("part", ["gate_proj", "up_proj", "down_proj"])
def test_resume_between_expert_receipt_and_weight_install(tmp_path, monkeypatch, part):
    engine, current, template, hidden, calls, _, _ = fixture(tmp_path, monkeypatch)
    store_class = type(engine.store(0))
    original = store_class.ensure
    fired = False

    def stop_after_receipt(store, name, *args, **kwargs):
        nonlocal fired
        result = original(store, name, *args, **kwargs)
        if not fired and name.endswith(f"mlp.experts.3.{part}"):
            fired = True
            raise Crash("receipt exists before live weight copy")
        return result

    monkeypatch.setattr(store_class, "ensure", stop_after_receipt)
    with pytest.raises(Crash, match="receipt exists"):
        engine.layer(0, current)
    assert engine.boundary.current(checksum=True)["layer"] == -1
    assert (engine.root / "layers/000/routes/manifest.json").exists()
    result = engine.layer(0, current)
    assert len(calls) == len(engine.selection)
    assert len({c["module"] for c in calls}) == len(calls)
    changed = changed_template(engine, template)
    for index, original_hidden in enumerate(hidden):
        actual = engine.boundary_batch(result, index)["hidden"]
        expected = run.numerical.native_forward(changed, dict(hidden=original_hidden))["hidden"]
        assert torch.equal(actual, expected)


def test_resume_after_frontier_publication_finishes_retirement(tmp_path, monkeypatch):
    engine, current, _, _, calls, source, _ = fixture(tmp_path, monkeypatch)
    original = StageStore.clear
    monkeypatch.setattr(StageStore, "clear", lambda self: (_ for _ in ()).throw(Crash("after pointer")))
    with pytest.raises(Crash, match="after pointer"):
        engine.layer(0, current)
    assert engine.boundary.current(checksum=True)["layer"] == 0
    assert list((engine.root / "layers/000/projections").glob("*.safetensors"))
    monkeypatch.setattr(StageStore, "clear", original)
    before = source.loads, len(calls)
    recovered = engine.initial()
    assert recovered["layer"] == 0
    assert before == (source.loads, len(calls))
    assert not (engine.root / "layers/000/stages").exists()
    assert not list((engine.root / "layers/000/projections").glob("*.safetensors"))


@pytest.mark.parametrize("filename", ["hessian.safetensors", "batch-00000.safetensors"])
def test_resume_rejects_corruption_with_restored_file_identity(tmp_path, monkeypatch, filename):
    engine, current, _, _, calls, source, progress = fixture(tmp_path, monkeypatch)
    progress.stop = ("phase_committed", "c")
    with pytest.raises(Crash):
        engine.layer(0, current)
    path = engine.root / "layers/000/stages/c" / filename
    before = path.stat()
    with path.open("r+b") as stream:
        stream.seek(-1, os.SEEK_END)
        original = stream.read(1)
        stream.seek(-1, os.SEEK_END)
        stream.write(bytes([original[0] ^ 1]))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = path.stat()
    assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
    count, loads = len(calls), source.loads
    with pytest.raises(ValueError, match="checksum"):
        engine.layer(0, current)
    assert len(calls) == count and source.loads == loads
    assert engine.boundary.current(checksum=True)["layer"] == -1

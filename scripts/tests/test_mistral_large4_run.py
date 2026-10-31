"""Causal native replay, exact state persistence and real recovery barriers."""
from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.mistral_large4.modeling_mistral_large4")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "quantization"))
from mistral_large4.core_k6 import native, run, source
from mistral_large4.core_k6.storage import ProjectionStore, RouteSpool, StageStore


class Progress:
    def __init__(self):
        self.events = []

    def emit(self, event, **fields):
        self.events.append((event, fields))


class RawHessian:
    def __init__(self, width, device="cpu"):
        self.H = torch.zeros((width, width), dtype=torch.float32, device=device)
        self.count = 0

    def add(self, value):
        rows = value.reshape(-1, value.shape[-1]).float()
        self.H.addmm_(rows.T, rows)
        self.count += len(rows)

    def raw_dict(self, minimum_rows=0):
        if self.count < minimum_rows:
            raise ValueError("missing actual rows")
        return dict(H=self.H.clone(), count=self.count, finalized=False, identity_rows=0)


def fixture(tmp_path, *, layers=1):
    from transformers.models.mistral_large4.configuration_mistral_large4 import MistralLarge4TextConfig
    from transformers.models.mistral_large4.modeling_mistral_large4 import (
        MistralLarge4TextDecoderLayer, MistralLarge4TextRotaryEmbedding)
    torch.set_num_threads(2)
    config = MistralLarge4TextConfig(vocab_size=32, hidden_size=128, intermediate_size=128,
        moe_intermediate_size=128, num_hidden_layers=layers, num_attention_heads=4,
        num_key_value_heads=4, kv_lora_rank=32, qk_rope_head_dim=16,
        qk_nope_head_dim=16, v_head_dim=32, head_dim=16, n_routed_experts=128,
        num_local_experts=128, num_experts_per_tok=4, n_shared_experts=1,
        n_group=1, topk_group=1, index_n_heads=4, index_head_dim=32, index_topk=3,
        mlp_layer_types=(["dense"] + ["sparse"] * 7)[:layers],
        indexer_types=["full", "full", "shared", "full", "full", "shared", "shared", "full"][:layers],
        kv_source_layers=[0, 1, 2, 3, 4, 5, 6, 3][:layers],
        no_rope_layers=[1, 1, 1, 0, 1, 1, 1, 0][:layers],
        layer_types=["indexed_attention"] * layers)
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    torch.manual_seed(787)
    blocks = [MistralLarge4TextDecoderLayer(config, layer).bfloat16().eval() for layer in range(layers)]
    for layer, block in enumerate(blocks):
        block.layer_idx = layer
        for parameter in block.parameters():
            parameter.data.normal_(0, .03)
        if block.self_attn.indexer is not None:
            block.self_attn.indexer.weights_proj = block.self_attn.indexer.weights_proj.float()
        if layer:
            block.mlp.gate = block.mlp.gate.float()
    embedding = torch.randn((32, 128)).bfloat16()

    class Source:
        text_config = config.to_dict()
        loads = 0

        def load_block(self, layer, device="cpu", *, include_experts=False):
            self.loads += 1
            block = copy.deepcopy(blocks[layer]).to(device)
            if layer and not include_experts:
                for key, value in list(block.mlp.experts._parameters.items()):
                    block.mlp.experts._parameters[key] = torch.nn.Parameter(
                        torch.empty(value.shape, dtype=value.dtype, device="meta"), requires_grad=False)
            return block

        def materialize_routed_experts(self, block, layer=None, device="cpu"):
            layer = block.layer_idx if layer is None else layer
            block.mlp.experts.load_state_dict({name: value.clone().to(device)
                for name, value in blocks[layer].mlp.experts.state_dict().items()}, assign=True)
            return block

        def tensor(self, name, device="cpu"):
            assert name == "tok_embeddings.weight"
            return embedding.clone().to(device)

        def rotary(self, device="cpu"):
            return MistralLarge4TextRotaryEmbedding(config).to(device)

        def close(self):
            pass

    calls = []

    class Numerical:
        DenseRawHessian = RawHessian
        attention_inputs = staticmethod(native.attention_inputs)
        attention_capture = staticmethod(native.attention_capture)
        attention_finish = staticmethod(native.attention_finish)
        shared_down_input = staticmethod(native.shared_down_input)
        router_choices = staticmethod(native.router_choices)
        expert_weight = staticmethod(native.expert_weight)
        complete_moe = staticmethod(native.complete_moe)

        @staticmethod
        def quantize_projection(weight, raw, **kwargs):
            # A visible perturbation makes stale/native downstream calibration
            # detectable independently of the CUDA trellis producer's quality.
            changed = (weight.float() * .875 + .005).bfloat16()
            calls.append(dict(weight=weight.clone(), changed=changed.clone(), H=raw["H"].clone(),
                              count=raw["count"], bits=kwargs["bits"]))
            packed = dict(trellis=changed.view(torch.int16).clone(),
                suh=torch.ones(weight.shape[1]).half(), svh=torch.ones(weight.shape[0]).half(),
                mcg=torch.tensor(0xCBAC1FED, dtype=torch.uint32).view(torch.int32))
            return packed, dict(hessian_weighted_relative_error=.01), dict(hessian_rows=raw["count"])

        @staticmethod
        def reconstruct_projection(packed, shape, **kwargs):
            return packed["trellis"].view(torch.bfloat16).clone().to(kwargs["device"])

    selection = {}
    for layer, block in enumerate(blocks):
        for parameter, checkpoint_name in source.layer_mapping(config, layer, block).items():
            if isinstance(checkpoint_name, str):
                module = checkpoint_name.removesuffix(".weight")
                if (checkpoint_name.endswith(".weight") and (module.endswith((".wq", ".wo", ".wkv_a_with_mqa"))
                        or ".feed_forward." in module or ".shared_experts." in module)):
                    value = block.get_parameter(parameter)
                    selection[module] = dict(in_features=value.shape[1], out_features=value.shape[0], bits=6)
        if layer:
            for expert in range(128):
                for part in ("w1", "w3", "w2"):
                    selection[f"layers.{layer}.experts.{expert}.{part}"] = dict(
                        in_features=128, out_features=128, bits=4)
    records = [dict(id="a", input_ids=[1, 2, 3, 4]), dict(id="b", input_ids=[5, 6, 7, 8, 9])]
    checkpoint_source = Source()
    engine = run.Engine(checkpoint_source, records, tmp_path / "run", tmp_path / "export",
                        dict(test="native-causal"), selection, Progress(), numerical=Numerical, device="cpu")
    return engine, checkpoint_source, blocks, embedding, records, calls


def gram(values):
    rows = torch.cat([value.reshape(-1, value.shape[-1]).double() for value in values])
    return rows.T @ rows


def test_dense_causal_hessians_and_export_failure_resume(tmp_path, monkeypatch):
    from mistral_large4.core_k6 import export
    engine, checkpoint_source, blocks, embedding, records, calls = fixture(tmp_path)
    initial = engine.initial()
    block = blocks[0]

    def fail_export(*args, **kwargs):
        raise RuntimeError("injected export interruption")

    monkeypatch.setattr(export, "assemble_layer", fail_export)
    with pytest.raises(RuntimeError, match="export interruption"):
        engine.layer(0, initial)
    assert engine.boundary.current(checksum=True)["layer"] == -1
    store = StageStore(engine.root / "layers/000/stages", engine.identity)
    assert store.load("f", checksum=True) is not None
    assert store.load("a") is store.load("c") is store.load("d") is store.load("e") is None
    checkpoints = engine.checkpoints(0)
    for name in engine.selection:
        checkpoints.install(name, source.linear_module(block, name))
    captured_o, captured_gate, captured_down = [], [], []
    handles = [linear.register_forward_pre_hook(lambda module, args, values=values: values.append(args[0].clone()))
               for linear, values in ((block.self_attn.o_proj, captured_o),
                   (block.mlp.gate_proj, captured_gate), (block.mlp.down_proj, captured_down))]
    try:
        for index, row, state in engine._read(store, "f"):
            x = torch.nn.functional.embedding(torch.tensor(row["input_ids"]), embedding).unsqueeze(0)
            context = native.prepare_context(x, checkpoint_source.rotary())
            with torch.inference_mode():
                expected, topk = block(x, past_key_values=None, use_cache=False, **context)
            assert torch.equal(state["hidden"], expected)
            assert torch.equal(state["prev_topk_indices"], topk)
    finally:
        for handle in handles:
            handle.remove()
    assert len(calls) == 6 and all(call["count"] == 9 for call in calls)
    for call, values in ((calls[2], captured_o), (calls[3], captured_gate), (calls[5], captured_down)):
        reference = gram(values)
        assert float(torch.linalg.norm(call["H"].double() - reference) / torch.linalg.norm(reference)) < 1e-6
    assert not torch.equal(calls[0]["weight"], calls[0]["changed"])
    monkeypatch.setattr(export, "assemble_layer", lambda *a, **k: dict(payload_bytes=1))
    before = checkpoint_source.loads
    current = engine.layer(0, initial)
    assert current["layer"] == 0 and checkpoint_source.loads == before and len(calls) == 6
    assert not engine.boundary.directory(-1).exists()
    assert not list((engine.root / "layers/000/projections").glob("*.safetensors"))


def test_partial_projection_oom_keeps_predecessor_and_resumes(tmp_path, monkeypatch):
    from mistral_large4.core_k6 import export
    engine, checkpoint_source, _, _, _, calls = fixture(tmp_path)
    initial = engine.initial()
    original = ProjectionStore.ensure

    def fail_after_first(store, name, *args, **kwargs):
        result = original(store, name, *args, **kwargs)
        if name.endswith(".attention.wq"):
            raise torch.OutOfMemoryError("injected OOM after durable projection receipt")
        return result

    monkeypatch.setattr(ProjectionStore, "ensure", fail_after_first)
    with pytest.raises(torch.OutOfMemoryError):
        engine.layer(0, initial)
    assert len(calls) == 1 and engine.boundary.current(checksum=True)["layer"] == -1
    monkeypatch.setattr(ProjectionStore, "ensure", original)
    monkeypatch.setattr(export, "assemble_layer", lambda *a, **k: dict(payload_bytes=1))
    assert engine.layer(0, initial)["layer"] == 0
    assert checkpoint_source.loads == 2 and len(calls) == 6


def test_resume_after_frontier_commit_finishes_retirement(tmp_path, monkeypatch):
    from mistral_large4.core_k6 import export
    engine, checkpoint_source, _, _, _, calls = fixture(tmp_path)
    initial = engine.initial()
    monkeypatch.setattr(export, "assemble_layer", lambda *a, **k: dict(payload_bytes=1))
    original = StageStore.clear

    def fail_cleanup(store):
        raise RuntimeError("injected post-pointer power loss")

    monkeypatch.setattr(StageStore, "clear", fail_cleanup)
    with pytest.raises(RuntimeError, match="power loss"):
        engine.layer(0, initial)
    assert engine.boundary.current(checksum=True)["layer"] == 0
    assert list((engine.root / "layers/000/projections").glob("*.safetensors"))
    monkeypatch.setattr(StageStore, "clear", original)
    before = checkpoint_source.loads
    assert engine.initial()["layer"] == 0
    assert checkpoint_source.loads == before and len(calls) == 6
    assert not (engine.root / "layers/000/stages").exists()
    assert not list((engine.root / "layers/000/projections").glob("*.safetensors"))


def test_exact_shared_kv_and_topk_survive_all_native_frontiers(tmp_path, monkeypatch):
    from mistral_large4.core_k6 import export
    engine, checkpoint_source, blocks, embedding, records, calls = fixture(tmp_path, layers=8)
    monkeypatch.setattr(export, "assemble_layer", lambda *a, **k: dict(payload_bytes=1))
    current = engine.initial()
    expected = [torch.nn.functional.embedding(torch.tensor(row["input_ids"]), embedding).unsqueeze(0)
                for row in records]
    shared, topks = [{} for _ in records], [None for _ in records]
    for layer, block in enumerate(blocks):
        previous_calls = len(calls)
        current = engine.layer(layer, current)
        new_calls = calls[previous_calls:]
        # Install the independently captured producer outputs on the reference
        # block after retirement removed packed payloads. Invocation order is
        # core q/kv/o, shared gate/up/down, then each routed gate/up/down.
        core_names = [f"layers.{layer}.attention.wq"]
        if layer != 7:
            core_names.append(f"layers.{layer}.attention.wkv_a_with_mqa")
        core_names.append(f"layers.{layer}.attention.wo")
        mlp = "feed_forward" if layer == 0 else "shared_experts"
        core_names.extend(f"layers.{layer}.{mlp}.{part}" for part in ("w1", "w3", "w2"))
        with torch.inference_mode():
            for name, call in zip(core_names, new_calls, strict=False):
                source.linear_module(block, name).weight.copy_(call["changed"])
            if layer:
                for position, call in enumerate(new_calls[len(core_names):]):
                    expert, part = divmod(position, 3)
                    native.expert_weight(block, expert, ("w1", "w3", "w2")[part]).copy_(call["changed"])
            for index in range(len(records)):
                context = native.prepare_context(expected[index], checkpoint_source.rotary(), topks[index],
                                                 shared_kv_states=shared[index])
                expected[index], topks[index] = block(expected[index], past_key_values=None, use_cache=False, **context)
                saved = engine.boundary_batch(current, index)
                assert torch.equal(saved["hidden"], expected[index]), f"hidden layer {layer} batch {index}"
                assert torch.equal(saved["prev_topk_indices"], topks[index])
                if layer >= 3:
                    assert torch.equal(saved["kv_nope_pass"], shared[index]["nope"][0])
                    assert torch.equal(saved["kv_nope_rot"], shared[index]["nope"][1])
                    assert saved["kv_nope_rot"].shape[-1] == 0
        # Resume has no source load and validates byte-exact serialized state.
        before = checkpoint_source.loads
        assert engine.initial()["layer"] == layer
        assert checkpoint_source.loads == before
    assert blocks[7].self_attn.kv_a_proj_with_mqa is None


def test_cold_routes_recovery_and_swiglu_hessians_match_fp64(tmp_path):
    torch.manual_seed(787)
    inputs = torch.randn((1, 16, 128)).bfloat16()

    def router(x):
        return (torch.arange(4).expand(len(x), 4), torch.ones(len(x), 4),
                torch.arange(4, 8).expand(len(x), 4))

    spool = RouteSpool.create(tmp_path, dict(test=1), 16,
        iter([(0, dict(id="a"), dict(mlp_input=inputs))]), router, Progress(), 3,
        width=128, experts=16)
    try:
        for expert, natural, augmented in ((0, 16, 0), (4, 0, 16), (15, 0, 0)):
            raw, evidence = spool.hessian(expert, device="cpu")
            actual = natural + augmented
            x = inputs.reshape(16, 128).double() if actual else torch.zeros((0, 128), dtype=torch.float64)
            reference = x.T @ x + (1024 - actual) * torch.eye(128, dtype=torch.float64)
            assert raw["count"] == 1024 and evidence["identity_rows"] == 1024 - actual
            assert evidence["natural_rows"] == natural and evidence["augmented_rows"] == augmented
            assert evidence["ranks"] == [5, 8]
            assert float(torch.linalg.norm(raw["H"].double() - reference) / torch.linalg.norm(reference)) < 1e-6
        gate, up = torch.randn((64, 128)).bfloat16(), torch.randn((64, 128)).bfloat16()
        raw, evidence = spool.hessian(4, down=(gate, up), device="cpu")
        g, u = torch.nn.functional.linear(inputs.reshape(16, 128), torch.cat((gate, up))).chunk(2, -1)
        x = (torch.nn.functional.silu(g) * u).double()
        reference = x.T @ x + 1008 * torch.eye(64, dtype=torch.float64)
        assert float(torch.linalg.norm(raw["H"].double() - reference) / torch.linalg.norm(reference)) < 1e-6
    finally:
        spool.close()


def test_stage_manifest_is_the_commit_barrier(tmp_path):
    from kimi_k3.uniform_k2.storage import commit_safetensors
    store = StageStore(tmp_path, dict(test=1))
    store.directory("a").mkdir()
    receipt = commit_safetensors(store.directory("a") / "H.safetensors", dict(H=torch.eye(4)))
    assert store.load("a") is None
    entries = [dict(filename="H.safetensors", kind="hessian", **receipt)]
    store.commit("a", entries)
    assert store.load("a", checksum=True) is not None
    with pytest.raises(ValueError, match="identity"):
        StageStore(tmp_path, dict(test=2)).load("a")
    path = store.directory("a") / "H.safetensors"
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        store.load("a", checksum=True)

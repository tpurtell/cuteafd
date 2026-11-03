"""Muse Spark / Mistral Large 4 reference adapters against the official modules.

Tiny random configs, written in each checkpoint's own tensor naming and storage
formats, run through (a) the official full text model and (b) the layer-major
adapter path; logits must agree, and fixed-geometry prefixes must be bit-exact.
"""
import contextlib
import json
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")
if not hasattr(transformers, "MistralLarge4TextConfig") or not hasattr(transformers, "MuseSparkTextConfig"):
    pytest.skip("transformers lacks Muse Spark / Mistral Large 4", allow_module_level=True)
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python/reference"))
from families.muse_spark import model as muse
from families.mistral_large4 import model as mistral


@pytest.fixture(autouse=True)
def f32():
    # CPU elementwise kernels split vector loops by length across threads; the
    # golden's prefix proof runs on CUDA, where elementwise ops are per element.
    previous, threads = torch.get_default_dtype(), torch.get_num_threads()
    torch.manual_seed(0)
    torch.set_num_threads(1)
    yield
    torch.set_default_dtype(previous)
    torch.set_num_threads(threads)


def tiny_muse_config():
    from transformers.models.muse_spark.configuration_muse_spark import MuseSparkTextConfig
    layers = 4
    config = MuseSparkTextConfig(
        vocab_size=97, hidden_size=64, intermediate_size=64, moe_hidden_size=32, moe_intermediate_size=48,
        num_hidden_layers=layers, num_attention_heads=8, num_key_value_heads=2, head_dim=8,
        num_local_experts=6, num_experts_per_tok=2, sliding_window=5,
        layer_types=["sliding_attention", "full_attention"] * 2, mlp_layer_types=["sparse"] * layers,
        layer_rope_theta=[500000.0, 0, 500000.0, 0], rope_parameters={"rope_theta": 500000.0, "rope_type": "default"},
        pad_token_id=0)
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    return config


def write_checkpoint(tmp_path, tensors, index_name):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(snapshot / "shard.safetensors"))
    (snapshot / index_name).write_text(json.dumps({"weight_map": {k: "shard.safetensors" for k in tensors}}))
    return snapshot


def randomize(module):
    with torch.no_grad():
        for name, value in list(module.named_parameters()) + list(module.named_buffers()):
            if value.dtype.is_floating_point and "inv_freq" not in name:
                value.copy_(torch.randn_like(value) * (0.3 if value.ndim > 1 else 0.5))


def test_muse_layer_major_matches_official_model(tmp_path):
    from transformers.models.muse_spark import modeling_muse_spark as reference
    config = tiny_muse_config()
    official = reference.MuseSparkTextModel(config).eval()
    randomize(official)
    head = torch.randn(config.vocab_size, config.hidden_size) * 0.3
    tensors = {muse.PREFIX + k: v.detach().clone() for k, v in official.state_dict().items()
               if "rotary" not in k}
    tensors[muse.HEAD] = head.bfloat16().float()
    # The checkpoint ships BF16 weights with the F32 router and bias.
    for key, value in tensors.items():
        if not key.endswith(("mlp.gate.weight", "e_score_correction_bias")):
            tensors[key] = value.bfloat16()
    official.load_state_dict({k[len(muse.PREFIX):]: v.float() for k, v in tensors.items() if k.startswith(muse.PREFIX)},
                             strict=False)
    # As shipped: the (1 + w) norm weights are BF16 parameters in the official module.
    for layer in official.layers:
        for name in muse.NATIVE_DTYPE:
            parent, _, leaf = name.rpartition(".")
            owner = layer.get_submodule(parent)
            owner._parameters[leaf] = torch.nn.Parameter(owner._parameters[leaf].bfloat16(), requires_grad=False)
    snapshot = write_checkpoint(tmp_path, tensors, "model.safetensors.index.json")
    ids = torch.randint(1, config.vocab_size, (1, 23))
    with torch.inference_mode():
        expected = muse.logits(official(input_ids=ids).last_hidden_state, head.bfloat16().float(), config)
        checkpoint = muse.Checkpoint(snapshot)
        outer = muse.load_outer(checkpoint, config, device="cpu", dtype=torch.float32)
        hidden = outer["embed"](ids)
        routes = []
        for index in range(config.num_hidden_layers):
            layer = muse.load_layer(checkpoint, config, index, device="cpu", dtype=torch.float32)
            assert isinstance(layer.mlp.experts.gate_up_proj, muse.LazyExperts)
            with muse.capture_routes(layer) as records:
                hidden = muse.layer_forward(layer, config, index, hidden, outer["rotary"])
            routes.append(records[0][0])
        actual = muse.logits(outer["norm"](hidden), outer["head"], config)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert all(r.shape == (ids.shape[1], config.num_experts_per_tok) for r in routes)


def test_muse_fixed_geometry_prefix_is_bit_exact(tmp_path):
    from transformers.models.muse_spark import modeling_muse_spark as reference
    config = tiny_muse_config()
    layer = reference.MuseSparkTextDecoderLayer(config, 0).eval()
    randomize(layer)
    rotary = reference.MuseSparkTextRotaryEmbedding(config)
    hidden = torch.randn(1, 300, config.hidden_size)
    with torch.inference_mode(), muse.geometry(reference):
        long = muse.layer_forward(layer, config, 0, hidden, rotary)
        short = muse.layer_forward(layer, config, 0, hidden[:, :170], rotary)
    plain = muse.layer_forward(layer, config, 0, hidden, rotary)
    assert torch.equal(long[:, :170], short)
    assert torch.allclose(long, plain, atol=1e-5, rtol=1e-5)


def test_muse_source_pins_fail_closed(monkeypatch):
    monkeypatch.setattr(muse, "SOURCE_PINS", {**muse.SOURCE_PINS, "masking_utils.py": "0" * 64})
    with pytest.raises(ValueError, match="source differs"):
        muse.source_identity()


def tiny_mistral_config():
    from transformers.models.mistral_large4.configuration_mistral_large4 import MistralLarge4TextConfig
    layers = 8
    config = MistralLarge4TextConfig(
        vocab_size=89, hidden_size=128, intermediate_size=256, moe_intermediate_size=128,
        num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=4,
        kv_lora_rank=32, qk_rope_head_dim=16, qk_nope_head_dim=16, v_head_dim=32, head_dim=16,
        n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1, n_group=1, topk_group=1,
        index_n_heads=4, index_head_dim=32, index_topk=128,
        mlp_layer_types=["dense"] + ["sparse"] * (layers - 1),
        indexer_types=["full", "full", "shared", "full", "full", "shared", "shared", "full"],
        kv_source_layers=[0, 1, 2, 3, 4, 5, 6, 3], no_rope_layers=[1, 1, 1, 0, 1, 1, 1, 0],
        layer_types=["indexed_attention"] * layers, final_logit_softcapping=30.0)
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    return config


def power_of_two_fp8(weight):
    rows, cols = weight.shape
    sr, sc = (rows + 127) // 128, (cols + 127) // 128
    scale = torch.exp2(torch.randint(-9, -5, (sr, sc)).float())
    grown = scale.repeat_interleave(128, 0)[:rows].repeat_interleave(128, 1)[:, :cols]
    quantized = (weight / grown).clamp(-448, 448).to(torch.float8_e4m3fn)
    return quantized, scale, quantized.float() * grown


def mistral_checkpoint(official, config, head):
    """Official text state -> consolidated names; FP8 for the converted modules."""
    inverse = {v: k for k, v in mistral._ATTENTION.items()}
    fp8 = ("q_proj", "o_proj", "indexer.wq_b", "experts", "shared_experts", "gate_proj", "up_proj", "down_proj")
    keep = ("kv_b_proj", "indexer.wk", "indexer.weights_proj", "kv_a_proj_with_mqa", "mlp.gate.weight")
    tensors, exact = {}, {}

    def put(name, value, official_name):
        if value.ndim == 2 and any(t in official_name for t in fp8) and not any(t in official_name for t in keep):
            q, s, v = power_of_two_fp8(value)
            tensors[name], tensors[name.removesuffix("weight") + "qscale_weight"] = q, s
            exact[official_name] = v
        else:
            tensors[name] = value.bfloat16()
            exact[official_name] = value.bfloat16().float()

    for index, layer in enumerate(official.layers):
        mapping = mistral.layer_map(config, index, layer)
        params = dict(layer.named_parameters())
        for name, source in mapping.items():
            if isinstance(source, str):
                put(source, params[name].detach(), f"layers.{index}.{name}")
                continue
            stack = params[name].detach()
            values = []
            for e, item in enumerate(source):
                if isinstance(item, str):
                    put(item, stack[e], f"layers.{index}.{name}.{e}")
                    values.append(exact[f"layers.{index}.{name}.{e}"])
                else:
                    half = stack.shape[1] // 2
                    put(item[0], stack[e, :half], f"layers.{index}.{name}.{e}.g")
                    put(item[1], stack[e, half:], f"layers.{index}.{name}.{e}.u")
                    values.append(torch.cat([exact[f"layers.{index}.{name}.{e}.g"],
                                             exact[f"layers.{index}.{name}.{e}.u"]]))
            exact[f"layers.{index}.{name}"] = torch.stack(values)
    tensors["tok_embeddings.weight"] = official.embed_tokens.weight.detach().bfloat16()
    tensors["norm.weight"] = official.norm.weight.detach().bfloat16()
    tensors["lerope.alpha"] = (torch.randn(official.rotary_emb.alpha.shape) * 0.1).bfloat16()
    tensors["output.weight"] = head.bfloat16()
    return tensors, exact


def test_mistral_layer_major_matches_official_model(tmp_path):
    from transformers.models.mistral_large4 import modeling_mistral_large4 as reference
    config = tiny_mistral_config()
    official = reference.MistralLarge4TextModel(config).eval()
    randomize(official)
    head = torch.randn(config.vocab_size, config.hidden_size) * 0.3
    tensors, exact = mistral_checkpoint(official, config, head)
    state = {}
    for index, layer in enumerate(official.layers):
        for name, _ in layer.named_parameters():
            state[f"layers.{index}.{name}"] = exact[f"layers.{index}.{name}"]
    state["embed_tokens.weight"] = tensors["tok_embeddings.weight"].float()
    state["norm.weight"] = tensors["norm.weight"].float()
    official.load_state_dict(state, strict=False)
    official.rotary_emb.alpha.copy_(tensors["lerope.alpha"].float())
    snapshot = write_checkpoint(tmp_path, tensors, "consolidated.safetensors.index.json")
    ids = torch.randint(1, config.vocab_size, (1, 19))
    with torch.inference_mode():
        expected = mistral.logits(official(input_ids=ids).last_hidden_state, head.bfloat16().float(), config)
        checkpoint = mistral.Checkpoint(snapshot)
        outer = mistral.load_outer(checkpoint, config, device="cpu", dtype=torch.float32)
        hidden = torch.nn.functional.embedding(ids, outer["embed"])
        shared, topk = {}, None
        for index in range(config.num_hidden_layers):
            layer = mistral.load_layer(checkpoint, config, index, device="cpu", dtype=torch.float32)
            hidden, topk = mistral.layer_forward(layer, config, hidden, outer["rotary"], shared, topk)
        actual = mistral.logits(outer["norm"](hidden), outer["head"], config)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_mistral_fixed_geometry_prefix_is_bit_exact_and_matches_untiled():
    from transformers.models.mistral_large4 import modeling_mistral_large4 as reference
    config = tiny_mistral_config()
    official = reference.MistralLarge4TextModel(config).eval()
    randomize(official)
    hidden = torch.randn(1, 400, config.hidden_size) * 0.5

    def run(x, tiled):
        shared, topk, out = {}, None, x
        context = mistral.geometry(reference) if tiled else stable_topk(reference)
        with torch.inference_mode(), context:
            for layer in official.layers:
                out, topk = mistral.layer_forward(layer, config, out, official.rotary_emb, shared, topk)
        return out
    long, short = run(hidden, True), run(hidden[:, :290], True)
    assert torch.equal(long[:, :290], short)
    # Untiled official indexer with the same deterministic tie rule (relu makes exact
    # 0.0 ties common, and official topk leaves their order undefined).
    untiled = run(hidden, False)
    assert ((long - untiled).norm() / untiled.norm()).item() < 1e-5


@pytest.mark.parametrize("layer_index", [0, 3])
def test_mistral_sparse_selection_uses_official_full_window_geometry(monkeypatch, layer_index):
    from transformers.models.mistral_large4 import modeling_mistral_large4 as reference
    import torch.nn.functional as functional
    import shape_invariant
    config = tiny_mistral_config()
    indexer = reference.MistralLarge4TextIndexer(config, layer_index).eval()
    randomize(indexer)
    hidden = torch.randn(1, 385, config.hidden_size)
    positions = torch.arange(hidden.shape[1])[None]
    rotary = reference.MistralLarge4TextRotaryEmbedding(config)
    embeddings = rotary(hidden, positions)
    mask = mistral.causal_mask(hidden.shape[1], device="cpu", dtype=hidden.dtype)[:, 0]
    original = reference.MistralLarge4TextIndexer.forward
    linear, layer_norm = functional.linear, torch.nn.LayerNorm.forward
    calls = []

    def observed(self, states, *args, **kwargs):
        calls.append(states.shape[1])
        assert functional.linear is linear
        assert torch.nn.LayerNorm.forward is layer_norm
        return original(self, states, *args, **kwargs)

    with torch.inference_mode(), mistral.stable_ties(reference):
        expected = indexer(hidden, embeddings, mask, positions)
    monkeypatch.setattr(reference.MistralLarge4TextIndexer, "forward", observed)
    with torch.inference_mode(), mistral.geometry(reference):
        actual = indexer(hidden, embeddings, mask, positions)
        assert functional.linear is shape_invariant.linear
        assert torch.nn.LayerNorm.forward is not layer_norm
    assert calls == [385]
    assert torch.equal(actual, expected)
    assert functional.linear is linear
    assert torch.nn.LayerNorm.forward is layer_norm


def stable_topk(reference):
    return mistral.stable_ties(reference)


# ---- layer-major goldens end to end on CPU (prefix proof, official arm, resume) ----

def synthetic_panel(tmp_path, family, snapshot, vocab, lengths):
    import hashlib as _h
    from fidelity_windows import SET_SCHEMA, bucket, set_hash
    generator = torch.Generator().manual_seed(1)
    windows = []
    for i, length in enumerate(lengths):
        tokens = torch.randint(1, vocab, (length,), generator=generator).tolist()
        start = length - 64
        windows.append({"id": f"w{i}", "block": "E", "bucket": bucket(start), "tokens": tokens,
                        "roles": ["ctx"] * (start + 10) + ["gen"] * (length - start - 10), "score_from": start})
    manifest = {"schema": SET_SCHEMA, "family": family, "model": "fixture", "checkpoint": "fixture",
                "version": "test", "quick_windows": ["w0"], "windows": windows,
                "tokenizer_sha256": _h.sha256((snapshot / "tokenizer.json").read_bytes()).hexdigest()}
    manifest["set_sha256"] = set_hash(manifest)
    path = tmp_path / "windows.json"
    path.write_text(json.dumps(manifest))
    return path


def run_golden(tmp_path, monkeypatch, golden, family, snapshot, panel, media_adapter=None, geometry="tiled", official_state=None):
    import layer_major
    import types
    monkeypatch.setattr(layer_major, "DEVICE", "cpu")
    monkeypatch.setattr(layer_major.shutil, "disk_usage", lambda path: types.SimpleNamespace(free=2 ** 41))
    out = tmp_path / "golden"
    args = types.SimpleNamespace(snapshot=snapshot, windows=panel, out=out, layers=None, prefix_only=False,
                                 wait_shards=0, routes=tmp_path / "routes", geometry=geometry,
                                 official_state=official_state)
    args.media = media_adapter is not None
    args.official_tokens = 4 if args.media else 512
    layer_major.run(args, family, extras_root=getattr(args, "routes", None) if family.name == "muse_spark" else None,
                    media_adapter=media_adapter)
    return out


def synthetic_media(panel, hidden):
    import types
    from fidelity_windows import set_hash
    manifest = json.loads(panel.read_text())
    features = {"ab" * 32: torch.randn(6, hidden).bfloat16()}
    for window in manifest["windows"]:
        window["tokens"][1:7] = [5, 5, 6, 5, 5, 7]
        window["media"] = [{"kind": "image", "start": 1, "len": 6, "grid": [1, 4, 4],
            "key": "ab" * 32, "fixture": {"path": "image.png", "sha256": "cd" * 32}}]
    manifest["image_processor"] = {"merge": 2, "row_markers": 1,
        "image_token_id": 5, "row_break_token_id": 6, "end_token_id": 7}
    manifest["set_sha256"] = set_hash(manifest)
    panel.write_text(json.dumps(manifest))
    adapter = types.SimpleNamespace(validate_spans=lambda m, s: None,
        provenance=lambda a, m: {"synthetic_test_adapter": True},
        window_features=lambda a, m: (features, {"fixture": True}))
    return adapter, features


def fixture_snapshot(tmp_path, tensors, index_name):
    snapshot = write_checkpoint(tmp_path, tensors, index_name)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "tokenizer.json").write_text("{}")
    return snapshot


@pytest.mark.parametrize("geometry", ["tiled", "official-full"])
@pytest.mark.parametrize("with_media", [False, True])
def test_muse_golden_qualifies_scores_resumes_and_writes_routes(tmp_path, monkeypatch, with_media, geometry):
    from families.muse_spark import golden
    from transformers.models.muse_spark import modeling_muse_spark as reference
    config = tiny_muse_config()
    official = reference.MuseSparkTextModel(config).eval()
    randomize(official)
    tensors = {muse.PREFIX + k: v.detach().bfloat16() for k, v in official.state_dict().items() if "rotary" not in k}
    for key in list(tensors):
        if key.endswith(("mlp.gate.weight", "e_score_correction_bias", "residual_gate.gate")):
            tensors[key] = tensors[key].float()
    tensors[muse.HEAD] = (torch.randn(config.vocab_size, config.hidden_size) * 0.3).bfloat16()
    snapshot = fixture_snapshot(tmp_path, tensors, "model.safetensors.index.json")
    monkeypatch.setattr(muse, "load_official", lambda s: (reference, config))
    monkeypatch.setattr(muse, "source_identity", lambda: {"fixture": True})
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self, *a, **k: self)
    family = golden.Family.__new__(golden.Family)
    family.reference, family.config = reference, config
    family.checkpoint, family.layers, family.dtype, family.outer = muse.Checkpoint(snapshot), config.num_hidden_layers, torch.float32, None
    family.rotary = reference.MuseSparkTextRotaryEmbedding(config)
    panel = synthetic_panel(tmp_path, "muse_spark", snapshot, config.vocab_size, [700, 300])
    adapter, features = synthetic_media(panel, config.hidden_size) if with_media else (None, None)
    out = run_golden(tmp_path, monkeypatch, golden, family, snapshot, panel, media_adapter=adapter, geometry=geometry)
    meta = json.loads((out / "meta.json").read_text())
    if geometry == "tiled":
        assert meta["prefix_qualification"]["passed"] and meta["official_forward"]["passed"]
    else:
        from fidelity_windows import validate_reference
        assert validate_reference(meta, json.loads(panel.read_text())) == config.vocab_size
        assert not {"prefix_qualification", "fixed_rows", "official_forward"}.intersection(meta)
    assert meta["scored_tokens"] == 128 and len(meta["seconds_per_layer"]) == config.num_hidden_layers
    routes = tmp_path / "routes" / "layer000"
    ids = __import__("numpy").fromfile(routes / "w0.ids.bin", dtype="<i2").reshape(700, 2)
    assert ids.min() >= 0 and ids.max() < config.num_local_experts
    assert json.loads((routes / "index.json").read_text())["files"]
    window = json.loads(panel.read_text())["windows"][1]
    with torch.inference_mode():
        checkpoint = muse.Checkpoint(snapshot)
        outer = muse.load_outer(checkpoint, config, device="cpu", dtype=torch.float32)
        hidden = outer["embed"](torch.tensor([window["tokens"]]))
        if with_media:
            hidden[0, 1:7] = features["ab" * 32].float()
        for index in range(config.num_hidden_layers):
            layer = muse.load_layer(checkpoint, config, index, device="cpu", dtype=torch.float32)
            hidden = muse.layer_forward(layer, config, index, hidden, outer["rotary"])
        head_input = hidden[0] if geometry == "official-full" else hidden[0, window["score_from"] - 1:-1]
        expected = muse.logits(outer["norm"](head_input), outer["head"], config).numpy()
        if geometry == "official-full":
            expected = expected[window["score_from"] - 1:-1]
    import numpy as np
    actual = np.fromfile(out / "windows" / "w1" / "logits.bin", dtype="<f4").reshape(expected.shape)
    assert np.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    # Resume after completion reuses the proof and reproduces identical logits.
    first = (out / "windows" / "w0" / "logits.bin").read_bytes()
    family.outer = None
    run_golden(tmp_path, monkeypatch, golden, family, snapshot, panel, media_adapter=adapter, geometry=geometry)
    assert (out / "windows" / "w0" / "logits.bin").read_bytes() == first
    # An unset diagnostic hook must preserve the complete serialized score outputs.
    monkeypatch.delenv("CUTEAFD_ACTIVATION_TRACE_POSITION", raising=False)
    monkeypatch.delenv("CUTEAFD_ACTIVATION_TRACE_WINDOW", raising=False)
    monkeypatch.delenv("CUTEAFD_ACTIVATION_TRACE_TOKENS", raising=False)
    observed = {str(p.relative_to(out)): p.read_bytes() for p in (out / "windows").rglob("*.bin")}
    plain_root = tmp_path / "plain-run"
    plain_root.mkdir()
    with monkeypatch.context() as plain:
        plain.setattr(golden, "activation_trace", lambda *args: contextlib.nullcontext())
        family.outer = None
        plain_out = run_golden(plain_root, monkeypatch, golden, family, snapshot, panel)
    assert {str(p.relative_to(plain_out)): p.read_bytes() for p in (plain_out / "windows").rglob("*.bin")} == observed

    # The diagnostic entry point never invokes qualification or publishes a golden/state.
    import layer_major
    trace_out = tmp_path / "trace-only"
    monkeypatch.setenv("CUTEAFD_ACTIVATION_TRACE_WINDOW", "w1")
    monkeypatch.setenv("CUTEAFD_ACTIVATION_TRACE_TOKENS", "7")
    monkeypatch.setenv("CUTEAFD_ACTIVATION_TRACE_POSITION", "3")
    monkeypatch.setenv("CUTEAFD_ACTIVATION_TRACE_DIR", str(trace_out))
    monkeypatch.delenv("CUTEAFD_TRACE_LAYER", raising=False)
    monkeypatch.setattr("shape_invariant.qualify", lambda *args: pytest.fail("diagnostic qualified a prefix"))
    args = __import__("types").SimpleNamespace(snapshot=snapshot, windows=panel, out=trace_out,
                                             prefix_only=False, wait_shards=0)
    family.outer = None
    result = layer_major.run(args, family)
    assert result == {"diagnostic": True, "window": "w1", "tokens": 7}
    files = list(trace_out.rglob("*"))
    assert len([p for p in files if p.is_file()]) == config.num_hidden_layers * 9 + 2
    assert all(p.suffix == ".bin" for p in files if p.is_file())
    assert {p.name for p in trace_out.iterdir()} == {"position3-prefill"}
    folder = trace_out / "position3-prefill"
    np = __import__("numpy")
    assert np.fromfile(folder / "final-logits.f32.bin", dtype="<f4").shape == (config.vocab_size,)
    assert (folder / "final-logits.f32.bin").read_bytes() == (folder / "common-head-self-logits.f32.bin").read_bytes()

    # Native BF16 keeps router/gates F32 and reuses the same official common head.
    family = golden.Family(snapshot, trace_dtype="bf16", trace_common_hidden=[
        ("f32", folder / f"layer{config.num_hidden_layers - 1}-hidden.f32.bin")])
    assert family.dtype is None
    module = family.load_layer(0)
    assert module.mlp.gate.weight.dtype == torch.float32
    assert module.post_attention_residual_gate.gate.dtype == torch.float32
    assert module.self_attn.q_proj.weight.dtype == torch.bfloat16
    del module
    bf16_out = tmp_path / "trace-bf16"
    monkeypatch.setenv("CUTEAFD_ACTIVATION_TRACE_DIR", str(bf16_out))
    args.out = bf16_out
    assert layer_major.run(args, family) == result
    bf16_folder = bf16_out / "position3-prefill"
    assert (bf16_folder / "common-head-f32-logits.f32.bin").read_bytes() == (folder / "common-head-self-logits.f32.bin").read_bytes()
    assert np.isfinite(np.fromfile(bf16_folder / "final-logits.f32.bin", dtype="<f4")).all()
    monkeypatch.delenv("CUTEAFD_ACTIVATION_TRACE_POSITION")
    with pytest.raises(ValueError, match="require activation trace"):
        golden.Family(snapshot, trace_dtype="bf16")
    with pytest.raises(ValueError, match="require activation trace"):
        golden.Family(snapshot, trace_common_hidden=[("f32", folder / "layer0-hidden.f32.bin")])


@pytest.mark.parametrize("geometry", ["tiled", "official-full"])
@pytest.mark.parametrize("with_media", [False, True])
def test_mistral_golden_threads_shared_kv_and_topk_across_layers(tmp_path, monkeypatch, with_media, geometry):
    from families.mistral_large4 import golden
    from transformers.models.mistral_large4 import modeling_mistral_large4 as reference
    config = tiny_mistral_config()
    official = reference.MistralLarge4TextModel(config).eval()
    randomize(official)
    head = torch.randn(config.vocab_size, config.hidden_size) * 0.3
    tensors, _ = mistral_checkpoint(official, config, head)
    snapshot = fixture_snapshot(tmp_path, tensors, "consolidated.safetensors.index.json")
    monkeypatch.setattr(mistral, "load_official", lambda s: (reference, config))
    monkeypatch.setattr(mistral, "source_identity", lambda: {"fixture": True})
    family = golden.Family.__new__(golden.Family)
    family.reference, family.config = reference, config
    family.checkpoint, family.layers, family.dtype = mistral.Checkpoint(snapshot), config.num_hidden_layers, torch.float32
    family.outer, family.rotary = None, None
    family.source = {i: s for i, s in enumerate(config.kv_source_layers) if s != i}
    family.stores = set(family.source.values())
    family.last_reader = {s: max(i for i, x in family.source.items() if x == s) for s in family.stores}
    panel = synthetic_panel(tmp_path, "mistral_large4", snapshot, config.vocab_size, [700, 260])
    adapter, features = synthetic_media(panel, config.hidden_size) if with_media else (None, None)
    if geometry == "official-full":
        import shape_invariant
        def forbidden(*args, **kwargs):
            raise AssertionError("official-full used tiled arithmetic")
        monkeypatch.setattr(family, "geometry", forbidden)
        monkeypatch.setattr(shape_invariant, "linear", forbidden)
    out = run_golden(tmp_path, monkeypatch, golden, family, snapshot, panel, media_adapter=adapter, geometry=geometry)
    meta = json.loads((out / "meta.json").read_text())
    if geometry == "tiled":
        assert meta["prefix_qualification"]["passed"] and meta["official_forward"]["passed"]
    else:
        from fidelity_windows import validate_reference
        assert validate_reference(meta, json.loads(panel.read_text())) == config.vocab_size
        assert not {"prefix_qualification", "fixed_rows", "official_forward"}.intersection(meta)
        assert meta["official_full"]["completed_layers"] == config.num_hidden_layers
        assert not list(out.glob("prefix-gate-*"))
    # Layer-major logits equal the official full model run on the same window (stable top-k).
    window = json.loads(panel.read_text())["windows"][1]
    ids = torch.tensor([window["tokens"]])
    state = {}
    with torch.inference_mode(), stable_topk(reference):
        official_full = reference.MistralLarge4TextModel(config).eval()
        checkpoint = mistral.Checkpoint(snapshot)
        outer = mistral.load_outer(checkpoint, config, device="cpu", dtype=torch.float32)
        hidden = torch.nn.functional.embedding(ids, outer["embed"])
        if with_media:
            hidden[0, 1:7] = features["ab" * 32].float()
        shared, topk = {}, None
        for index in range(config.num_hidden_layers):
            layer = mistral.load_layer(checkpoint, config, index, device="cpu", dtype=torch.float32)
            hidden, topk = mistral.layer_forward(layer, config, hidden, outer["rotary"], shared, topk)
        head_input = hidden[0] if geometry == "official-full" else hidden[0, window["score_from"] - 1:-1]
        expected = mistral.logits(outer["norm"](head_input), outer["head"], config).numpy()
        if geometry == "official-full":
            expected = expected[window["score_from"] - 1:-1]
    import numpy as np
    actual = np.fromfile(out / "windows" / "w1" / "logits.bin", dtype="<f4").reshape(expected.shape)
    assert np.abs(actual - expected).max() / np.abs(expected).max() < 1e-5


def test_official_full_geometry_selection_and_equality(tmp_path, monkeypatch):
    import copy
    import hashlib
    import types
    import numpy as np
    import layer_major
    from fidelity_windows import canonical
    family = types.SimpleNamespace(config=types.SimpleNamespace(index_topk=8))
    panel = {"windows": [{"id": "w", "tokens": list(range(9)), "media": [{"kind": "image"}]}]}
    args = types.SimpleNamespace(geometry="auto")
    assert layer_major.reference_geometry(args, family, panel) == "official-full"
    panel["windows"][0]["media"] = []
    assert layer_major.reference_geometry(args, family, panel) == "tiled"
    args.geometry, args.prefix_only = "official-full", True
    with pytest.raises(ValueError, match="does not use"):
        layer_major.reference_geometry(args, family, panel)
    args.prefix_only = False
    monkeypatch.setattr(layer_major.shutil, "disk_usage", lambda p: types.SimpleNamespace(free=2 ** 41))
    snapshot = {"snapshot_revision": "pin"}
    binding = {"state_bytes_per_layer": 1024, "snapshot_identity": snapshot,
               "windows": {"official_short": hashlib.sha256(canonical(list(range(9)))).hexdigest()}}
    old = layer_major.LayerState(tmp_path / "old", binding)
    pending = old.begin(3)
    hidden = torch.arange(18).reshape(1, 9, 2).float()
    old.save(pending, "official_short", {"hidden": hidden})
    old.commit(pending)
    new = layer_major.LayerState(tmp_path / "new", binding)
    pending = new.begin(3)
    new.save(pending, "w", {"hidden": hidden})
    new.commit(pending)
    saved = old.root / "latest.json"
    proof = layer_major.official_state_equality(saved, new, panel, snapshot)
    assert proof["passed"] and proof["shape"] == [1, 9, 2]
    pending = new.begin(4)
    new.save(pending, "w", {"hidden": hidden + 1})
    new.commit(pending)
    old.latest["layer"] = 4
    layer_major.atomic_json(saved, old.latest)
    proof = layer_major.official_state_equality(saved, new, panel, snapshot)
    assert not proof["passed"] and proof["different_values"] == 18
    changed = copy.deepcopy(panel)
    changed["windows"][0]["tokens"][0] = 99
    with pytest.raises(ValueError, match="tokens differ"):
        layer_major.official_state_equality(saved, new, changed, snapshot)
    metrics = layer_major.logit_comparison(np.ones((37, 5)), np.ones((37, 5)))
    assert metrics == {"kl": 0., "max_kl": 0., "top1": 1., "relative_logit_l2": 0., "max_abs_logit_difference": 0.}


def test_muse_centered_norm_weights_keep_the_official_bf16_add(tmp_path):
    from transformers.models.muse_spark import modeling_muse_spark as reference
    config = tiny_muse_config()
    official = reference.MuseSparkTextModel(config).eval()
    randomize(official)
    tensors = {muse.PREFIX + k: v.detach().bfloat16() for k, v in official.state_dict().items() if "rotary" not in k}
    snapshot = write_checkpoint(tmp_path, tensors, "model.safetensors.index.json")
    layer = muse.load_layer(muse.Checkpoint(snapshot), config, 0, device="cpu", dtype=torch.float32)
    for name in muse.NATIVE_DTYPE:
        assert layer.get_parameter(name).dtype == torch.bfloat16
    assert layer.self_attn.q_proj.weight.dtype == torch.float32
    assert layer.post_attention_residual_gate.gate.dtype == torch.float32


def test_muse_ordered_experts_equal_the_official_cpu_accumulation():
    from transformers.models.muse_spark import modeling_muse_spark as reference
    config = tiny_muse_config()
    moe = reference.MuseSparkMoE(config).eval()
    randomize(moe)
    hidden = torch.randn(37, config.moe_hidden_size)
    _, weights, index = moe.gate(torch.randn(37, config.hidden_size))
    with torch.inference_mode():
        official = moe.experts(hidden, index, weights)
        ordered = muse.ordered_experts(moe.experts, hidden, index, weights)
    assert torch.equal(official, ordered)


@pytest.mark.parametrize("filename", [
    "models/mistral_large4/modeling_mistral_large4.py",
    "models/pixtral/modeling_pixtral.py",
    "models/pixtral/configuration_pixtral.py",
    "models/pixtral/image_processing_pil_pixtral.py",
    "models/pixtral/processing_pixtral.py",
])
def test_mistral_vision_source_pins_fail_before_construction(monkeypatch, filename):
    monkeypatch.setattr(mistral, "SOURCE_PINS", {**mistral.SOURCE_PINS, filename: "0" * 64})
    with pytest.raises(ValueError, match="source differs"):
        mistral.load_vision_official({})


def test_mistral_vision_qualifier_preserves_official_rope_and_weights(monkeypatch):
    import importlib.util
    import numpy as np
    path = Path(__file__).resolve().parents[2] / "python/tools/qualify/mistral_large4/qualify-vision.py"
    spec = importlib.util.spec_from_file_location("ml4_vision_qualification", path)
    qualifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qualifier)
    monkeypatch.setattr(qualifier, "DEPTH", 1)
    config = {
        "vision_config": {"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1,
                          "num_attention_heads": 4, "patch_size": 14, "image_size": 56},
        "text_config": {"hidden_size": 48},
        "spatial_merge_size": 2,
    }
    tower, projector = mistral.load_vision_official(config)
    tensors = {}
    def keep(name, value):
        tensors[name] = value.detach().bfloat16().view(torch.uint16).cpu().numpy().copy()
    keep("vision_encoder.patch_conv.weight", tower.patch_conv.weight)
    keep("vision_encoder.ln_pre.weight", tower.ln_pre.weight)
    rename = {"attention.q_proj": "attention.wq", "attention.k_proj": "attention.wk",
              "attention.v_proj": "attention.wv", "attention.o_proj": "attention.wo",
              "feed_forward.gate_proj": "feed_forward.w1", "feed_forward.up_proj": "feed_forward.w3",
              "feed_forward.down_proj": "feed_forward.w2", "attention_norm": "attention_norm",
              "ffn_norm": "ffn_norm"}
    for target, source in rename.items():
        keep(f"vision_encoder.transformer.layers.0.{source}.weight",
             tower.get_parameter(f"transformer.layers.0.{target}.weight"))
    for target, source in {"norm": "pre_mm_projector_norm", "patch_merger.merging_layer": "patch_merger.merging_layer",
                           "linear_1": "vision_language_adapter.w_in", "linear_2": "vision_language_adapter.w_out"}.items():
        keep(source + ".weight", projector.get_parameter(target + ".weight"))
    expected_buffers = dict(tower.named_buffers())
    actual, projected, sources = qualifier.reference(config, tensors, torch.bfloat16, device="cpu")
    assert actual.config._attn_implementation == "sdpa"
    for name, buffer in actual.named_buffers():
        assert buffer.dtype == expected_buffers[name].dtype == torch.float32
        assert torch.equal(buffer, expected_buffers[name])
    assert expected_buffers and all(p.dtype == torch.bfloat16 for p in actual.parameters())
    assert all(p.dtype == torch.bfloat16 for p in projected.parameters())
    assert {p.name for p in sources} >= {"modeling_pixtral.py", "modeling_mistral_large4.py", "model.py"}
    pixel_values = torch.randn(1, 3, 56, 56).bfloat16()
    sizes = torch.tensor([[56, 56]])
    with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        features = actual(pixel_values, image_sizes=sizes).last_hidden_state
        assert projected(features.squeeze(0), sizes).shape == (4, 48)
        assert torch.isfinite(features).all()
    for name, value in actual.state_dict().items():
        expected = tower.state_dict()[name].bfloat16()
        np.testing.assert_array_equal(value.view(torch.uint16).numpy(), expected.view(torch.uint16).numpy())


def test_mistral_ordered_experts_equal_the_official_cpu_accumulation():
    from transformers.models.mistral_large4 import modeling_mistral_large4 as reference
    config = tiny_mistral_config()
    moe = reference.MistralLarge4TextMoE(config).eval()
    randomize(moe)
    hidden = torch.randn(37, config.hidden_size)
    _, weights, index = moe.gate(hidden)
    with torch.inference_mode():
        assert torch.equal(moe.experts(hidden, index, weights), mistral.ordered_experts(moe.experts, hidden, index, weights))


def test_muse_activation_trace_is_opt_in_and_preserves_outputs(tmp_path, monkeypatch):
    from families.muse_spark.golden import activation_trace
    from transformers.models.muse_spark import modeling_muse_spark as reference
    import numpy as np
    config = tiny_muse_config()
    layer = reference.MuseSparkTextDecoderLayer(config, 0).eval()
    randomize(layer)
    rotary = reference.MuseSparkTextRotaryEmbedding(config)
    hidden = torch.randn(1, 7, config.hidden_size)
    monkeypatch.delenv("CUTEAFD_ACTIVATION_TRACE_POSITION", raising=False)
    monkeypatch.setenv("CUTEAFD_ACTIVATION_TRACE_DIR", str(tmp_path))
    with torch.inference_mode():
        expected = muse.layer_forward(layer, config, 0, hidden, rotary)
        with activation_trace(layer, 0, hidden):
            untraced = muse.layer_forward(layer, config, 0, hidden, rotary)
    assert torch.equal(expected, untraced)
    assert not list(tmp_path.iterdir())
    monkeypatch.setenv("CUTEAFD_ACTIVATION_TRACE_POSITION", "3")
    monkeypatch.setenv("CUTEAFD_TRACE_LAYER", "0")
    with torch.inference_mode(), activation_trace(layer, 0, hidden):
        actual = muse.layer_forward(layer, config, 0, hidden, rotary)
    assert torch.equal(expected, actual)
    folder = tmp_path / "position3-prefill"
    assert {p.name for p in folder.iterdir()} == {
        f"layer0-{name}{suffix}.bin" for name in ("hidden", "residual", "combined", "route_weights")
        for suffix in ("", ".f32")
    } | {"layer0-route_ids.bin"}
    np.testing.assert_array_equal(np.fromfile(folder / "layer0-hidden.f32.bin", dtype="<f4"),
                                  expected[0, 3].numpy())
    assert (folder / "layer0-hidden.bin").read_bytes() == expected[0, 3].bfloat16().view(torch.int16).numpy().tobytes()
    assert (folder / "layer0-route_ids.bin").stat().st_size == config.num_experts_per_tok * 4
    assert (folder / "layer0-route_weights.bin").stat().st_size == config.num_experts_per_tok * 2
    assert (folder / "layer0-combined.bin").stat().st_size == config.moe_hidden_size * 2
    assert all(not m._forward_hooks for m in layer.modules())
    monkeypatch.setenv("CUTEAFD_TRACE_LAYER", "1")
    with torch.inference_mode(), activation_trace(layer, 0, hidden):
        assert torch.equal(expected, muse.layer_forward(layer, config, 0, hidden, rotary))


def test_muse_activation_trace_removes_hooks_after_failure(tmp_path, monkeypatch):
    from families.muse_spark.golden import activation_trace
    from transformers.models.muse_spark import modeling_muse_spark as reference
    layer = reference.MuseSparkTextDecoderLayer(tiny_muse_config(), 0).eval()
    monkeypatch.setenv("CUTEAFD_ACTIVATION_TRACE_POSITION", "0")
    monkeypatch.setenv("CUTEAFD_ACTIVATION_TRACE_DIR", str(tmp_path))
    monkeypatch.delenv("CUTEAFD_TRACE_LAYER", raising=False)
    with pytest.raises(RuntimeError, match="injected"), activation_trace(layer, 0, torch.zeros(1, 1, 64)):
        raise RuntimeError("injected")
    assert all(not m._forward_hooks for m in layer.modules())


def test_layer_major_media_embedding_validates_and_preserves_text():
    from types import SimpleNamespace
    from layer_major import embed_window
    hidden = torch.randn(1, 9, 4)
    family = SimpleNamespace(embed=lambda tokens: hidden.clone())
    window = {"tokens": list(range(9)), "media": [{"key": "image", "start": 2, "len": 3}]}
    value = torch.ones(3, 4, dtype=torch.bfloat16)
    result = embed_window(family, window, {"image": value})
    assert torch.equal(result[:, :2], hidden[:, :2])
    assert torch.equal(result[:, 5:], hidden[:, 5:])
    assert torch.equal(result[0, 2:5], value.float())
    assert torch.equal(embed_window(family, {"tokens": list(range(9))}), hidden)
    for features in (None, {}, {"image": torch.ones(2, 4)}, {"image": torch.ones(3, 5)}):
        with pytest.raises(ValueError, match="official image features"):
            embed_window(family, window, features)
    for start in (-1, 8):
        window["media"][0]["start"] = start
        with pytest.raises(ValueError, match="span extent"):
            embed_window(family, window, {"image": value})

"""CPU contracts for the official Qwen media hook (no CUDA or checkpoint weights)."""
import ast
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/reference"))
import qwen_media as media


def window():
    return {"tokens": [17, media.START_ID] + [media.IMAGE_ID] * 6 + [media.END_ID, 18, 19],
            "roles": ["ctx"] * 10 + ["gen"], "score_from": 10,
            "media": [{"kind": "image", "start": 2, "len": 6, "grid": [1, 4, 6],
                       "key": "a" * 64, "fixture": {"path": "image.png", "sha256": "b" * 64}}]}


@pytest.mark.parametrize("mutation", ["mimo", "cache", "start", "end", "unbound", "grid", "order"])
def test_native_spans_fail_closed(mutation):
    w = window()
    if mutation == "mimo":
        w["tokens"][2:8] = [151655] * 6
    elif mutation == "cache":
        w["tokens"][2:8] = [0x80000001] * 6
    elif mutation == "start":
        w["tokens"][1] = 0
    elif mutation == "end":
        w["tokens"][8] = 0
    elif mutation == "unbound":
        w["tokens"][9] = media.IMAGE_ID
    elif mutation == "grid":
        w["media"][0]["grid"] = [1, 4, 4]
    else:
        w["media"].append(w["media"][0].copy())
    with pytest.raises(ValueError):
        media.validate_window(w)


def test_native_spans_and_text():
    w = window()
    assert media.validate_window(w) == w["media"]
    assert media.validate_window({"tokens": [17, 18], "roles": ["ctx", "gen"], "score_from": 1}) == []


def test_snapshot_identity_hashes_actual_modeling(tmp_path, monkeypatch):
    source = tmp_path / "modeling_qwen4_exp.py"
    source.write_bytes(b"actual pinned modeling bytes")
    monkeypatch.setattr(media, "official_reference", lambda: types.SimpleNamespace(__file__=str(source)))
    for file in ("config.json", "tokenizer.json", "preprocessor_config.json"):
        (tmp_path / file).write_bytes(file.encode())
    identity = media.snapshot_identity(tmp_path)
    assert set(identity) == {"snapshot_revision", "transformers_revision", "modeling_sha256",
                            "config_sha256", "tokenizer_sha256", "preprocessor_sha256"}
    assert identity["modeling_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert identity["transformers_revision"] == media.TRANSFORMERS_REVISION
    for key, file in (("config", "config.json"), ("tokenizer", "tokenizer.json"),
                      ("preprocessor", "preprocessor_config.json")):
        assert identity[key + "_sha256"] == hashlib.sha256((tmp_path / file).read_bytes()).hexdigest()


def test_processor_uses_official_pil_size_and_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "official_reference", lambda: None)
    module = "transformers.models.qwen2_vl.image_processing_pil_qwen2_vl"
    monkeypatch.setitem(sys.modules, module, types.SimpleNamespace(Qwen2VLImageProcessorPil=lambda **kw: kw))
    (tmp_path / "preprocessor_config.json").write_text(json.dumps({
        "size": {"shortest_edge": 65536, "longest_edge": 16777216}, "patch_size": 16,
        "merge_size": 2, "temporal_patch_size": 2, "image_mean": [0.5] * 3,
        "image_std": [0.5] * 3, "processor_class": "Qwen3VLProcessor",
        "image_processor_type": "Qwen2VLImageProcessorFast"}))
    config = media.processor(tmp_path)
    assert config["max_pixels"] == 4096 * 32 ** 2
    assert config["size"]["shortest_edge"] == 65536
    assert config["image_mean"] == config["image_std"] == [0.5] * 3
    assert "processor_class" not in config and "image_processor_type" not in config


class CpuArray(np.ndarray):
    """Only the CPU indexing API used by the unchanged official rope methods."""
    @property
    def device(self):
        return "cpu"

    def view(self, *shape):
        return self.reshape(*shape)

    def expand(self, *shape):
        return np.broadcast_to(self, tuple(n if s == -1 else s for n, s in zip(self.shape, shape))).view(CpuArray)

    def long(self):
        return self.astype(np.int64)

    def to(self, *args, **kwargs):
        return self

    def unsqueeze(self, dim):
        return np.expand_dims(self, dim)


def array(value, **kwargs):
    return np.asarray(value, dtype=kwargs.get("dtype")).view(CpuArray)


def cpu_torch():
    return types.SimpleNamespace(long=np.int64, tensor=array, arange=lambda n, **kw: array(np.arange(n)),
        zeros=lambda *shape, **kw: array(np.zeros(shape), dtype=kw["dtype"]), zeros_like=np.zeros_like,
        cat=lambda values, dim=0: array(np.concatenate(values, axis=dim)),
        stack=lambda values, dim=0: array(np.stack(values, axis=dim)),
        meshgrid=lambda *values, **kw: tuple(array(v) for v in np.meshgrid(*values, **kw)))


def official_rope_methods(torch):
    # The checkout may have no initialized submodules; use the repository's pinned source.
    source = ROOT / "third_party/transformers/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py"
    if not source.exists():
        override = os.environ.get("QWEN_REFERENCE_SOURCE")
        if not override:
            pytest.skip("initialize pinned transformers or set QWEN_REFERENCE_SOURCE for official rope CPU gate")
        source = Path(override)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == media.MODELING_SHA256
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen4ExpModel")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in
               ("get_vision_position_ids", "get_rope_index")]
    scope = {"torch": torch, "itertools": itertools}
    code = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + methods,
                      type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), str(source), "exec"), scope)
    return types.SimpleNamespace(Qwen4ExpModel=types.SimpleNamespace(**{n.name: scope[n.name] for n in methods}))


def test_official_rope_real_spans_multiple_images_and_prefix(tmp_path, monkeypatch):
    torch = cpu_torch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(media, "official_reference", lambda: official_rope_methods(torch))
    (tmp_path / "config.json").write_text(json.dumps({"image_token_id": media.IMAGE_ID,
        "vision_start_token_id": media.START_ID, "vision_end_token_id": media.END_ID,
        "vision_config": {"spatial_merge_size": 2}}))
    w = window()
    pos = media.rope_positions(w, tmp_path, device="cpu")
    assert pos.shape == (4, 1, 11)
    assert pos[0, 0].tolist() == list(range(11))
    assert pos[1:, 0, 2:8].tolist() == [[2] * 6, [2, 2, 2, 3, 3, 3], [2, 3, 4, 2, 3, 4]]
    assert pos[1:, 0, 8:].tolist() == [[5, 6, 7]] * 3
    extended = window()
    extended["tokens"] += [23] * 64
    extended["roles"] += ["gen"] * 64
    assert np.array_equal(pos, media.rope_positions(extended, tmp_path, device="cpu")[:, :, :11])
    second = window()
    second["tokens"] += [media.START_ID] + [media.IMAGE_ID] * 6 + [media.END_ID, 24]
    second["roles"] = ["ctx"] * (len(second["tokens"]) - 1) + ["gen"]
    second["score_from"] = len(second["tokens"]) - 1
    second["media"].append({**second["media"][0], "start": 12})
    assert media.rope_positions(second, tmp_path, device="cpu")[1:, 0, 12:18].tolist() == [
        [9] * 6, [9, 9, 9, 10, 10, 10], [9, 10, 11, 9, 10, 11]]
    text = {"tokens": [17] * 640, "roles": ["ctx"] * 640, "score_from": 64}
    text_pos = media.rope_positions(text, tmp_path, device="cpu")
    assert text_pos.tolist() == [[list(range(640))]] * 4


def test_injection_replicates_four_hc_lanes_without_changing_native_ids(monkeypatch):
    class Embedding(CpuArray):
        def copy_(self, value):
            self[:] = value

        def repeat(self, *copies):
            return np.tile(np.asarray(self), copies)

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        bfloat16=np.dtype(np.uint16), isfinite=lambda x: np.isfinite(
            (np.asarray(x).astype(np.uint32) << 16).view(np.float32))))
    w = window()
    original_ids = list(w["tokens"])
    embed = np.full((1, len(original_ids), media.WIDTH), 0x3f80, dtype=np.uint16).view(Embedding)
    rows = np.full((6, media.WIDTH), 0x4000, dtype=np.uint16).view(CpuArray)
    output = media.inject_embeddings(embed, w, {"a" * 64: rows}, 4)
    assert output.shape == (1, 11, 4 * 2560)
    assert np.all(output[0, 2:8] == 0x4000) and np.all(output[0, :2] == 0x3f80)
    assert np.all(output[0, 8:] == 0x3f80) and w["tokens"] == original_ids
    with pytest.raises(ValueError, match="geometry"):
        media.inject_embeddings(embed, w, {"a" * 64: rows}, 1)
    with pytest.raises(ValueError, match="feature shape"):
        media.inject_embeddings(embed, w, {"a" * 64: rows[:, :8]}, 4)


def test_feature_export_abi_index_and_probe_suppression(tmp_path, monkeypatch):
    class Features:
        def contiguous(self):
            return self

        def view(self, dtype):
            return self

        def numpy(self):
            return np.full((6, media.WIDTH), 0x3f80, dtype=np.uint16)

    torch = types.SimpleNamespace(uint16=np.uint16, cuda=types.SimpleNamespace(
        synchronize=lambda: None, empty_cache=lambda: None))
    monkeypatch.setitem(sys.modules, "torch", torch)
    identity = {"snapshot_revision": "pinned", "tokenizer_sha256": "c" * 64,
                "modeling_sha256": media.MODELING_SHA256}
    monkeypatch.setattr(media, "snapshot_identity", lambda snapshot: identity)
    monkeypatch.setattr(media, "load_tower", lambda snapshot: object())
    monkeypatch.setattr(media, "processor", lambda snapshot: object())
    calls = []
    monkeypatch.setattr(media, "encode_span", lambda *args: calls.append(args[-1]) or Features())
    out = tmp_path / "features"
    args = types.SimpleNamespace(media_root=tmp_path, windows=tmp_path / "windows.json",
                                 snapshot=tmp_path, media_features_out=out)
    w = window()
    panel = {"family": "qwen4", "checkpoint": "Qwen/Qwen3.8-Flash-Next", "set_sha256": "d" * 64,
             "windows": [w, window()]}
    features, actual = media.window_features(args, panel)
    assert len(calls) == len(features) == 1 and actual == identity
    key = w["media"][0]["key"]
    data = (out / (key + ".bf16")).read_bytes()
    meta = json.loads((out / (key + ".json")).read_text())
    assert len(data) == 6 * media.WIDTH * 2 and data[:2] == b"\x80\x3f"
    assert meta["shape"] == [6, 2560] and meta["dtype"] == "bf16-le" and meta["tower_dtype"] == "bf16"
    assert meta["snapshot_identity"] == identity and meta["fixture_sha256"] == "b" * 64
    assert meta["sha256"] == hashlib.sha256(data).hexdigest()
    index = json.loads((out / "features.json").read_text())
    assert index == {"schema": "cuteafd.media.features.index/1", "family": "qwen4",
                     "checkpoint": panel["checkpoint"], "tokenizer_sha256": "c" * 64,
                     "set_sha256": "d" * 64, "features": [key]}
    media.window_features(args, panel)
    args._prefix_probe, args.media_features_out = True, tmp_path / "probe"
    media.window_features(args, panel)
    assert not args.media_features_out.exists()
    bad = window()
    bad["media"][0]["fixture"]["sha256"] = "e" * 64
    panel["windows"].append(bad)
    with pytest.raises(ValueError, match="different fixtures/grids"):
        media.window_features(args, panel)


def test_imported_processor_provenance_fails_closed(tmp_path, monkeypatch):
    root = tmp_path / "src/transformers"
    model_path = root / "models/qwen4_exp/modeling_qwen4_exp.py"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"modeling")
    source = ROOT / "third_party/transformers/src/transformers"
    if not source.exists():
        pytest.skip("initialize pinned Transformers for provenance contract")
    modules = {}
    for name in media.PROCESSOR_SOURCES:
        path = root / (name.replace(".", "/") + ".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((source / path.relative_to(root)).read_bytes())
        modules["transformers." + name] = types.SimpleNamespace(__file__=str(path))
    copied = root / "models/qwen2_vl/image_processing_qwen2_vl.py"
    copied.write_bytes((source / copied.relative_to(root)).read_bytes())
    monkeypatch.setattr(media.importlib, "import_module", lambda name: modules[name])
    revision, changes = media.TRANSFORMERS_REVISION, ""
    monkeypatch.setattr(media.subprocess, "check_output", lambda args, **kw:
                        revision if args[-1] == "HEAD" else changes)
    ref = types.SimpleNamespace(__file__=str(model_path))
    media.verify_sources(ref)
    revision = "wrong"
    with pytest.raises(ValueError, match="clean pinned"):
        media.verify_sources(ref)
    revision, changes = media.TRANSFORMERS_REVISION, " M src/transformers/image_utils.py"
    with pytest.raises(ValueError, match="clean pinned"):
        media.verify_sources(ref)
    changes = ""
    path = root / "image_utils.py"
    original = path.read_bytes()
    path.write_bytes(original + b"\n# changed")
    with pytest.raises(ValueError, match="pinned imported image_utils"):
        media.verify_sources(ref)
    path.write_bytes(original)
    shadow = tmp_path / "shadow.py"
    shadow.write_bytes(original)
    modules["transformers.image_utils"].__file__ = str(shadow)
    with pytest.raises(ValueError, match="pinned imported image_utils"):
        media.verify_sources(ref)


@pytest.mark.parametrize("orientation,side", [(2, (64, 32)), (3, (64, 32)), (4, (64, 32)),
                                               (6, (32, 32)), (8, (32, 32))])
def test_exif_pixels_transpose_even_when_grid_is_unchanged(tmp_path, monkeypatch, orientation, side):
    import io
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from PIL import Image
    media.official_reference()
    width, height = side
    pixels = np.arange(width * height * 3, dtype=np.uint32).reshape(height, width, 3).astype(np.uint8)
    image = Image.fromarray(pixels)
    exif = Image.Exif()
    exif[274] = orientation
    encoded = io.BytesIO()
    image.save(encoded, format="PNG", exif=exif)
    path = tmp_path / "exif.png"
    path.write_bytes(encoded.getvalue())
    transforms = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
                  4: Image.Transpose.FLIP_TOP_BOTTOM, 6: Image.Transpose.ROTATE_270,
                  8: Image.Transpose.ROTATE_90}
    expected = np.asarray(image.transpose(transforms[orientation]))
    assert expected.shape == pixels.shape and not np.array_equal(expected, pixels)
    captured = []
    grid = [1, height // 16, width // 16]
    span = {"grid": grid, "len": grid[1] * grid[2] // 4,
            "fixture": {"path": "exif.png", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}

    def processor(**kwargs):
        captured.append(np.asarray(kwargs["images"]).copy())
        assert kwargs["images"].mode == "RGB" and kwargs["images"].getexif().get(274) is None
        return {"pixel_values": torch.zeros(1, 8), "image_grid_thw": torch.tensor([grid])}

    model = lambda *args, **kwargs: types.SimpleNamespace(pooler_output=torch.zeros(
        span["len"], media.WIDTH, dtype=torch.bfloat16))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    media.encode_span(model, processor, tmp_path, span)
    assert np.array_equal(captured[0], expected)


def test_real_cpu_torch_official_positions_and_hc(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers.models.qwen4_exp import modeling_qwen4_exp as ref
    assert hashlib.sha256(Path(ref.__file__).read_bytes()).hexdigest() == media.MODELING_SHA256
    (tmp_path / "config.json").write_text(json.dumps({"image_token_id": media.IMAGE_ID,
        "vision_start_token_id": media.START_ID, "vision_end_token_id": media.END_ID,
        "vision_config": {"spatial_merge_size": 2}}))
    w = window()
    pos = media.rope_positions(w, tmp_path, device="cpu")
    assert pos[1:, 0, 2:8].tolist() == [[2] * 6, [2, 2, 2, 3, 3, 3], [2, 3, 4, 2, 3, 4]]
    original = list(w["tokens"])
    embed = torch.ones(1, len(original), media.WIDTH, dtype=torch.bfloat16)
    rows = torch.full((6, media.WIDTH), 2, dtype=torch.bfloat16)
    out = media.inject_embeddings(embed, w, {"a" * 64: rows}, 4)
    assert torch.equal(out[0, 2:8].reshape(6, 4, media.WIDTH), rows[:, None].expand(-1, 4, -1))
    assert torch.all(out[0, :2] == 1) and w["tokens"] == original


def test_real_cpu_official_tower_output_selects_merged_pooler(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpVisionConfig
    ref = media.official_reference()
    config = Qwen4ExpVisionConfig(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                                  patch_size=2, temporal_patch_size=2, spatial_merge_size=2,
                                  out_hidden_size=media.WIDTH, num_position_embeddings=16)
    config._attn_implementation = "sdpa"
    model = ref.Qwen4ExpVisionModel(config).to(torch.bfloat16).eval()
    # Exercise unchanged official forward/return contract on CPU, not the full weight checkpoint.
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    monkeypatch.setattr(media, "read_fixture", lambda root, span: b"not decoded in this contract")
    from PIL import Image
    monkeypatch.setattr(Image, "open", lambda data: Image.new("RGB", (8, 8)))
    prepared = {"pixel_values": torch.zeros(16, 24), "image_grid_thw": torch.tensor([[1, 4, 4]])}
    image_processor = lambda **kw: prepared
    span = {"grid": [1, 4, 4], "len": 4}
    with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        official = model(prepared["pixel_values"].bfloat16(), prepared["image_grid_thw"], return_dict=True)
    assert official.last_hidden_state.shape == (16, 16) and official.pooler_output.shape == (4, media.WIDTH)
    result = media.encode_span(model, image_processor, tmp_path, span)
    assert torch.equal(result, official.pooler_output)


def test_qwen_media_prefix_adjusted_lengths_and_full_vocab_bits(tmp_path, monkeypatch):
    from fidelity_windows import (SET_SCHEMA, bucket, finish_golden, qualify_prefix,
                                  set_hash, validate_qualification, write_scored_logits)
    # Keep media targets in the toy vocabulary scored by the shared NLL writer.
    for name, token_id in (("START_ID", 20), ("IMAGE_ID", 21), ("END_ID", 22)):
        monkeypatch.setattr(media, name, token_id)
    w = window()
    w["tokens"] = [17] * 100 + w["tokens"] + [18] * 700
    w["roles"] = ["ctx"] * len(w["tokens"])
    w["media"][0]["start"] += 100
    w.update(id="vision00", block="vision", score_from=193, bucket=bucket(193))
    manifest = {"schema": SET_SCHEMA, "family": "qwen4", "checkpoint": "model",
                "windows": [w], "quick_windows": ["vision00"]}
    manifest["set_sha256"] = set_hash(manifest)
    identity = {"snapshot_revision": "pinned", "modeling_sha256": media.MODELING_SHA256}
    a = types.SimpleNamespace(out=tmp_path, windows=tmp_path / "windows.json", layers=None)
    changed = False

    def execute(args):
        panel = json.loads(args.windows.read_text())
        entries = []
        for i, probe in enumerate(panel["windows"]):
            media.validate_window(probe)
            logits = np.zeros((len(probe["tokens"]) - probe["score_from"], 32), dtype=np.float32)
            if changed and i:
                logits[0, -1] = np.float32(1e-8)  # Same top-1 is not enough.
            entries.append(write_scored_logits(args.out, probe, logits))
        finish_golden(args.out, panel, entries, snapshot_identity=identity, seconds=0)

    proof = qualify_prefix(a, manifest, execute)
    assert proof["lengths"] == [108 + 512, 108 + 576] and proof["rows"] == 512
    assert proof["finite"] and proof["passed"] and proof["media"] == w["media"]
    validate_qualification(proof, manifest, identity)
    changed = True
    with pytest.raises(ValueError, match="prefix invariance failed"):
        qualify_prefix(a, manifest, execute)


def test_golden_injects_before_repeat_and_keeps_ple_native_ids():
    source = (ROOT / "python/reference/families/qwen4/golden.py").read_text()
    tree = ast.parse(source)
    loop = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_windows")
    calls = [n for n in ast.walk(loop) if isinstance(n, ast.Call)]
    assert any(isinstance(n.func, ast.Name) and n.func.id == "inject_embeddings" for n in calls)
    assert any(isinstance(n.func, ast.Name) and n.func.id == "rope_positions" for n in calls)
    layer = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "layer")
    assert next(k.value.id for k in layer.keywords if k.arg == "ple_input_ids") == "ids"
    assert "qualify(a, manifest" in source
    helper = ast.parse((ROOT / "python/reference/qwen_media.py").read_text())
    inject = next(n for n in helper.body if isinstance(n, ast.FunctionDef) and n.name == "inject_embeddings")
    assert isinstance(inject.body[-1], ast.Return) and inject.body[-1].value.func.attr == "repeat"

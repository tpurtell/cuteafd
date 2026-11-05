"""Behavioral checks for mixed-rate K3.1 export and exact native retention."""
import json
from pathlib import Path
from types import SimpleNamespace
import struct
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "quantization"))
from kimi_k31.k2_core_k6 import export, recipe
from kimi_k3.uniform_k2.storage import read_header


def write_tensors(path, specs):
    """A tiny independent safetensors writer needs no GPU or torch import."""
    header, chunks, offset = {}, [], 0
    for name, (dtype, shape, payload) in sorted(specs.items()):
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[offset, offset + len(payload)])
        chunks.append(payload)
        offset += len(payload)
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(chunks))


def fixture(tmp_path, *, multiplier=recipe.MCG_MULTIPLIER):
    core, routed = "layer.core", "layer.experts.0.w1"
    selection = {core: dict(in_features=128, out_features=64, stored_out_features=128, bits=6),
                 routed: dict(in_features=128, out_features=128, stored_out_features=128, bits=2)}
    source = tmp_path / "source.safetensors"
    native = b"\x13\x24" * 128
    write_tensors(source, {core + ".weight": ("BF16", [64, 128], bytes(64 * 128 * 2)),
        routed + ".weight_packed": ("U8", [128, 64], bytes(128 * 64)),
        routed + ".weight_scale": ("U8", [128, 4], bytes(128 * 4)),
        "layer.indexer.k_norm.weight": ("BF16", [128], native)})
    replacements = {}
    for index, (module, geometry) in enumerate(selection.items()):
        path = tmp_path / f"packed-{index}.safetensors"
        specs = {}
        for name, (dtype, shape) in recipe.packed_specs(module, geometry).items():
            import math
            data = struct.pack("<I", multiplier) if name.endswith(".mcg") else bytes(math.prod(shape) * 2)
            specs[name] = (dtype, shape, data)
        write_tensors(path, specs)
        replacements[module] = path
    entries = {n: (source, n) for n in read_header(source)[0]}
    return entries, selection, replacements, native


def test_mixed_export_preserves_native_and_padded_shapes(tmp_path):
    entries, selected, replacements, native = fixture(tmp_path)
    target = tmp_path / "output.safetensors"
    receipt = export._assemble(entries, selected, replacements, target,
                               tmp_path / "receipt.json", {"producer": "test"}, layer=3)
    header, base = read_header(target)
    assert not any(n.endswith((".weight_packed", ".weight_scale")) for n in header)
    assert header["layer.core.svh"]["shape"] == [128]
    assert header["layer.core.trellis"]["shape"][-1] == 96
    assert header["layer.experts.0.w1.trellis"]["shape"][-1] == 32
    assert receipt["preserved_audit"]["payload_bytes"] == len(native)
    start, end = header["layer.indexer.k_norm.weight"]["data_offsets"]
    assert target.read_bytes()[base + start:base + end] == native


def test_resume_rejects_corrupted_committed_payload(tmp_path):
    entries, selected, replacements, _ = fixture(tmp_path)
    target, receipt_path = tmp_path / "output.safetensors", tmp_path / "receipt.json"
    identity = {"producer": "test"}
    first = export._assemble(entries, selected, replacements, target, receipt_path, identity)
    assert export._assemble(entries, selected, replacements, target, receipt_path, identity) == first
    data = bytearray(target.read_bytes())
    data[-1] ^= 1
    target.write_bytes(data)
    with pytest.raises(ValueError, match="changed|checksum"):
        export._assemble(entries, selected, replacements, target, receipt_path, identity)


def test_export_rejects_wrong_mcg(tmp_path):
    entries, selected, replacements, _ = fixture(tmp_path, multiplier=1)
    with pytest.raises(ValueError, match="multiplier"):
        export._assemble(entries, selected, replacements, tmp_path / "output.safetensors",
                         tmp_path / "receipt.json", {"producer": "test"})


def test_attention_kind_comes_from_headers_and_crosschecks_config(monkeypatch):
    monkeypatch.setattr(recipe, "DSA_LAYERS", (1,))
    names = {"language_model.model.layers.0.self_attn.q_proj.weight": "a",
             "language_model.model.layers.1.self_attn.indexer.wq_b.weight": "b"}
    config = dict(num_hidden_layers=2, dsa_config=dict(dsa_layers=[2]),
                  linear_attn_config=dict(full_attn_layers=[2], kda_layers=[1]))
    assert recipe.derive_attention_layers(names, config) == dict(kda=[0], dsa=[1])
    config["dsa_config"]["dsa_layers"] = [1]
    with pytest.raises(ValueError, match="disagree"):
        recipe.derive_attention_layers(names, config)


def test_tiny_inventory_rejects_unindexed_and_wrong_geometry(tmp_path, monkeypatch):
    path = tmp_path / "tiny.safetensors"
    name = "language_model.model.layers.0.self_attn.q_proj.weight"
    specs = {name: ("BF16", [128, 128])}
    doc = dict(model_type="kimi_k31", architectures=["KimiK31ForConditionalGeneration"], text_config={})
    source = SimpleNamespace(snapshot=tmp_path, config=doc, text_config={}, weight_map={name: path.name})
    monkeypatch.setattr(recipe, "SOURCE_CONFIG_SHA256", __import__("hashlib").sha256(recipe.canonical(doc).encode()).hexdigest())
    monkeypatch.setattr(recipe, "derive_attention_layers", lambda *args: {})
    monkeypatch.setattr(recipe, "reviewed_source_specs", lambda: specs)
    write_tensors(path, {name: ("BF16", [128, 128], bytes(128 * 128 * 2)),
                         "unindexed.weight": ("BF16", [1], bytes(2))})
    with pytest.raises(ValueError, match="unindexed"):
        recipe.inventory(source)
    write_tensors(path, {name: ("BF16", [64, 256], bytes(128 * 128 * 2))})
    with pytest.raises(ValueError, match="geometry"):
        recipe.inventory(source)


def test_reviewed_recipe_totals_and_native_indexer():
    specs = recipe.reviewed_source_specs()
    selected = recipe.reviewed_modules()
    assert len(specs) == 497340
    assert len(selected) == 248222
    assert recipe.TARGET_TENSOR_COUNT == 994710
    assert recipe.TARGET_PAYLOAD_BYTES == 731632865912
    assert "language_model.model.layers.3.self_attn.indexer.wq_b.weight" in specs
    assert "language_model.model.layers.3.self_attn.indexer.wq_b" not in selected

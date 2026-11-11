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


def test_publication_requires_matching_complete_producer(tmp_path, monkeypatch):
    from kimi_k31.k2_core_k6 import publish
    monkeypatch.setattr(publish, "SHARD_COUNT", 1)
    monkeypatch.setattr(publish, "TARGET_TENSOR_COUNT", 2)
    monkeypatch.setattr(publish, "TARGET_PAYLOAD_BYTES", 256)
    monkeypatch.setattr(publish, "MODULE_COUNT", 1)
    monkeypatch.setattr(publish, "shard_name", lambda i: "tiny.safetensors")
    root, output = tmp_path / "run", tmp_path / "export"
    root.mkdir()
    output.mkdir()
    identity = {"producer": "immutable"}
    plan = dict(identity=identity, root=str(root), export=str(output), publication_repo=publish.TARGET_REPO)
    (root / "plan.json").write_text(json.dumps(plan))
    complete = dict(identity=identity, summary=dict(shards=1, tensors=2, payload_bytes=256, quantized_projections=1))
    (root / "quantization-complete.json").write_text(json.dumps(complete))
    (output / "model.safetensors.index.json").write_text(json.dumps(dict(metadata=dict(total_size=256),
        weight_map={"a": "tiny.safetensors", "b": "tiny.safetensors"})))
    (output / "quantization-provenance.json").write_text(json.dumps(identity))
    assert publish._complete_plan(output, root, publish.TARGET_REPO) == identity
    complete["identity"] = {"producer": "obsolete"}
    (root / "quantization-complete.json").write_text(json.dumps(complete))
    with pytest.raises(ValueError, match="producer"):
        publish._complete_plan(output, root, publish.TARGET_REPO)
    with pytest.raises(ValueError, match="target"):
        publish._complete_plan(output, root, "wrong/repository")


def metadata_fixture(tmp_path):
    source_dir, output = tmp_path / "source", tmp_path / "export"
    source_dir.mkdir()
    output.mkdir()
    (source_dir / "LICENSE").write_text("original license bytes\n")
    (output / "LICENSE").write_bytes((source_dir / "LICENSE").read_bytes())
    identity = {"producer": "immutable"}
    geometry = dict(in_features=128, out_features=128, stored_out_features=128, bits=6)
    selected = {"layer.core": geometry}
    declaration = dict(quant_method="exl3", format="exl3", checkpoint_format="exl3", bits=2., core_bits=6.,
                       per_module_metadata="quantize_config.json:tensor_storage")
    source_doc = dict(model_type="kimi_k31", text_config=dict(hidden_size=128))
    config = dict(source_doc, quantization_config=declaration,
                  text_config=dict(source_doc["text_config"], quantization_config=declaration))
    (output / "config.json").write_text(json.dumps(config))
    weight_map = {n: "tiny.safetensors" for n in recipe.packed_specs("layer.core", geometry)}
    (output / "model.safetensors.index.json").write_text(json.dumps(dict(metadata=dict(total_size=1234), weight_map=weight_map)))
    tensor_storage = {"layer.core": dict(quant_format="exl3", bits_per_weight=6,
        mcg_multiplier=recipe.MCG_MULTIPLIER, **{k: geometry[k] for k in ("in_features", "out_features", "stored_out_features")},
        stored_tensors={n: dict(shape=shape, torch_dtype={"I16": "int16", "F16": "float16", "I32": "int32"}[dtype])
                        for n, (dtype, shape) in recipe.packed_specs("layer.core", geometry).items()})}
    quant = dict(method="exl3", **declaration, codebook="mcg", out_scales="auto", group_size=-1, desc_act=False,
                 meta=dict(quantization_provenance=identity), tensor_storage=tensor_storage,
                 module_include=[r"^language_model\.model\.layers\.[0-9]+\.block_sparse_moe\.experts\.[0-9]+\.w[123]$",
                                 r"^layer\.core$"])
    (output / "quantize_config.json").write_text(json.dumps(quant))
    (output / "quantization-provenance.json").write_text(json.dumps(identity))
    (output / "README.md").write_text(f"---\nlicense: other\nlicense_name: kimi-k3.1-pro\nbase_model: {recipe.BASE_REPO}\n---\n")
    errors = output / "quantization-errors"
    errors.mkdir()
    record = dict(module="layer.core", logical_shape=[128, 128],
                  metrics=dict(quantizer_path="hessian_ldlq", hessian_metric_status="ok", hessian_weighted_relative_error=.001))
    (errors / "layer-000.jsonl").write_text(json.dumps(record) + "\n")
    (errors / "index.json").write_text(json.dumps(dict(records=1, bundles={"layer-000.jsonl": 1})))
    source = SimpleNamespace(snapshot=source_dir, config=source_doc)
    return source, selected, output, identity, weight_map


def test_final_metadata_audit_checks_native_assets_and_one_pass_metadata(tmp_path):
    from kimi_k31.k2_core_k6.finalize import audit_metadata
    source, selected, output, identity, weight_map = metadata_fixture(tmp_path)
    assert audit_metadata(source, selected, output, identity, weight_map, 1234) == dict(
        native_assets=1, diagnostic_records=1, diagnostic_bundles=1)
    (output / "LICENSE").write_text("changed license bytes\n")
    with pytest.raises(ValueError, match="asset"):
        audit_metadata(source, selected, output, identity, weight_map, 1234)
    (output / "LICENSE").write_bytes((source.snapshot / "LICENSE").read_bytes())
    quant_path = output / "quantize_config.json"
    quant = json.loads(quant_path.read_text())
    quant["base_k2_provenance"] = "obsolete"
    quant_path.write_text(json.dumps(quant))
    with pytest.raises(ValueError, match="one-pass"):
        audit_metadata(source, selected, output, identity, weight_map, 1234)


def test_final_metadata_audit_rejects_header_index_mismatch(tmp_path):
    from kimi_k31.k2_core_k6.finalize import audit_metadata
    source, selected, output, identity, weight_map = metadata_fixture(tmp_path)
    weight_map = dict(weight_map, unexpected="tiny.safetensors")
    with pytest.raises(ValueError, match="index"):
        audit_metadata(source, selected, output, identity, weight_map, 1234)


def test_publication_stops_at_independent_audit_before_upload(tmp_path, monkeypatch):
    from kimi_k31.k2_core_k6 import publish
    monkeypatch.setattr(publish, "_complete_plan", lambda *args: {"producer": "test"})
    calls = []
    def audit(root):
        calls.append("audit")
        raise ValueError("independent audit rejected")
    monkeypatch.setattr(publish, "audit_completed_export", audit)
    monkeypatch.setattr(publish, "_collect_local", lambda *args: calls.append("collect"))
    with pytest.raises(ValueError, match="independent audit"):
        publish.publish(tmp_path / "export", tmp_path / "run", publish.TARGET_REPO)
    assert calls == ["audit"]

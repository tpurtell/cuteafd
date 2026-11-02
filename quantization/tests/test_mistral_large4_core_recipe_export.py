"""CPU contracts for native FP8 scale retirement and byte-preserving export."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mistral_large4.core_k6 import export, recipe
from kimi_k3.uniform_k2.storage import canonical, read_header


def source_config():
    text = dict(model_type="mistral_large4_text", hidden_size=6144, num_hidden_layers=72,
        vocab_size=204800, intermediate_size=24576, moe_intermediate_size=6144,
        first_k_dense_replace=1, n_routed_experts=128, n_shared_experts=1, num_experts_per_tok=4,
        num_attention_heads=64, num_key_value_heads=64, head_dim=64, qk_head_dim=128,
        qk_nope_head_dim=64, qk_rope_head_dim=64, v_head_dim=128, kv_lora_rank=512,
        index_n_heads=32, index_head_dim=128, index_topk=2048, no_rope_layer_interval=4,
        final_logit_softcapping=30.0, rms_norm_eps=1e-5, routed_scaling_factor=1.0,
        hidden_act="silu", n_group=1, topk_group=1, tie_word_embeddings=False,
        indexer_types=["full" if i in recipe.FULL_INDEXER_LAYERS else "shared" for i in range(72)],
        no_rope_layers=[0 if i % 4 == 3 else 1 for i in range(72)],
        kv_source_layers=[35 if i >= 39 and i % 4 == 3 else i for i in range(72)],
        mlp_layer_types=["dense"] + ["sparse"] * 71, layer_types=["indexed_attention"] * 72,
        rope_parameters=dict(attention_factor=1.0, beta_fast=32, beta_slow=1, factor=2048,
            original_max_position_embeddings=512, rope_theta=10000.0, rope_type="yarn"))
    return dict(model_type="mistral_large4", text_config=text,
        transformers_weights="consolidated.safetensors.index.json",
        quantization_config=dict(quant_method="fp8", weight_block_size=[128, 128], activation_scheme="dynamic",
            modules_to_not_convert=["model.vision_tower", "model.multi_modal_projector", "lm_head", "kv_b_proj",
                                   "indexer.wk", "indexer.weights_proj", "kv_a_proj_with_mqa"]))


@pytest.fixture
def header_source(tmp_path, monkeypatch):
    headers, weight_map, offset = {}, {}, {}
    widths = dict(BF16=2, F32=4, F8_E4M3=1)
    for index, (name, (dtype, shape)) in enumerate(recipe.reviewed_source_specs().items()):
        shard = f"part-{index % 54:02d}.safetensors"
        size = widths[dtype]
        for dimension in shape:
            size *= dimension
        start = offset.get(shard, 0)
        headers.setdefault(shard, {})[name] = dict(dtype=dtype, shape=list(shape), data_offsets=[start, start + size])
        offset[shard] = start + size
        weight_map[name] = shard
    monkeypatch.setattr(recipe, "read_header", lambda path: (headers[path.name], 8))
    return SimpleNamespace(snapshot=tmp_path, config=source_config(), weight_map=weight_map, headers=headers)


def test_reviewed_source_matches_native_inventory_and_padding(header_source):
    summary = recipe.inventory(header_source)
    assert summary["source_payload_bytes"] == 1_060_548_427_328
    assert summary["source_tensors"] == 56563
    assert summary["selected_modules"] == 27688
    assert summary["preserved_tensors"] == 1251
    assert summary["preserved_bytes"] == 12_415_146_560
    assert summary["target_payload_bytes"] == 540_617_010_400
    assert summary["target_tensors"] == 112003
    assert summary["parts"]["base_core"]["modules"] == 424
    assert summary["parts"]["routed"]["modules"] == 27264
    assert summary["parts"]["routed"]["packed_bytes"] == 515_260_918_272
    core = recipe.reviewed_core_modules()
    assert core["layers.0.attention.wkv_a_with_mqa"]["stored_out_features"] == 640
    assert core["layers.3.attention.wkv_a_with_mqa"]["out_features"] == 512
    assert "layers.39.attention.wkv_a_with_mqa" not in core
    assert "layers.71.attention.wkv_a_with_mqa" not in core
    assert "layers.3.attention.wkv_a" not in core
    assert len(recipe.FULL_INDEXER_LAYERS) == 44
    specs = recipe.reviewed_source_specs()
    assert specs["layers.35.attention.indexer.wq_b.weight"] == ("F8_E4M3", [4096, 6144])
    assert specs["layers.35.attention.indexer.wq_b.qscale_weight"] == ("F32", [32, 48])
    assert specs["mtp_dspark.backbone.target_projection.weight"] == ("BF16", [6144, 30720])


@pytest.mark.parametrize("damage", ["missing_scale", "scale_dtype", "scale_shape", "reuse_tensor", "indexer", "kv_source", "head_dim", "quant", "extra"])
def test_inventory_rejects_unreviewed_native_tensors(header_source, damage):
    name = "layers.0.attention.wq.qscale_weight"
    spec = header_source.headers[header_source.weight_map[name]][name]
    if damage == "missing_scale":
        del header_source.weight_map[name]
    elif damage == "scale_dtype":
        spec["dtype"] = "F8_E8M0"
    elif damage == "scale_shape":
        spec["shape"] = [48, 64]
    elif damage == "reuse_tensor":
        header_source.weight_map["layers.39.attention.wkv_a_with_mqa.weight"] = "part-00.safetensors"
    elif damage == "indexer":
        header_source.config["text_config"]["indexer_types"][17] = "full"
    elif damage == "kv_source":
        header_source.config["text_config"]["kv_source_layers"][39] = 39
    elif damage == "head_dim":
        header_source.config["text_config"]["qk_head_dim"] = 64
    elif damage == "quant":
        header_source.config["quantization_config"]["weight_block_size"] = [64, 128]
    else:
        header_source.headers[header_source.weight_map[name]]["unexpected.weight"] = deepcopy(spec)
    with pytest.raises(ValueError):
        recipe.inventory(header_source)


def raw_safetensors(path, values):
    """Serialize explicit bytes so CPU export tests need no CUDA/PyTorch import."""
    header, payload = {}, bytearray()
    for name, (dtype, shape, data) in sorted(values.items()):
        start = len(payload)
        payload.extend(data)
        header[name] = dict(dtype=dtype, shape=list(shape), data_offsets=[start, len(payload)])
    encoded = canonical(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return path


def packed_fixture(path, module, g):
    values = {}
    for name, (dtype, shape) in recipe.packed_specs(module, g).items():
        size = 4 if dtype == "I32" else 2
        for dimension in shape:
            size *= dimension
        values[name] = dtype, shape, bytes(size)
    values[module + ".mcg"] = "I32", [], struct.pack("<I", recipe.MCG_MULTIPLIER)
    return raw_safetensors(path, values)


@pytest.fixture
def toy_source(tmp_path, monkeypatch):
    directory, output = tmp_path / "source", tmp_path / "export"
    directory.mkdir()
    selection = {
        "layers.0.attention.wq": recipe._geometry(128, 128, 6, "base_core"),
        "layers.3.attention.wkv_a_with_mqa": recipe._geometry(128, 576, 6, "base_core"),
        "layers.3.experts.0.w1": recipe._geometry(128, 128, 4, "routed"),
        "output": recipe._geometry(128, 128, 6, "base_core"),
    }
    tensors, specs = {}, {}
    for module, g in selection.items():
        names = recipe.source_names(module, g)
        fp8 = len(names) == 2
        shape = [g["out_features"], g["in_features"]]
        tensors[names[0]] = "F8_E4M3" if fp8 else "BF16", shape, b"\x38" * (shape[0] * shape[1] * (1 if fp8 else 2))
        if fp8:
            tensors[names[1]] = "F32", [1, 1], struct.pack("<f", 1.)
    for layer in range(72):
        tensors[f"layers.{layer}.attention_norm.weight"] = "BF16", [128], b"\x80\x3f" * 128
    tensors.update({
        "layers.3.attention.indexer.wq_b.weight": ("F8_E4M3", [128, 128], b"\x38" * 16384),
        "layers.3.attention.indexer.wq_b.qscale_weight": ("F32", [1, 1], struct.pack("<f", .5)),
        "layers.3.attention.wkv_b.weight": ("BF16", [128, 128], b"\x00\x3f" * 16384),
        "mtp_dspark.backbone.layers.0.attention.wq.weight": ("BF16", [128, 128], b"\x80\x3e" * 16384),
        "tok_embeddings.weight": ("BF16", [5, 128], b"\x00\x3f" * 640),
    })
    buckets = [{} for _ in range(5)]
    for i, (name, value) in enumerate(tensors.items()):
        buckets[i % 5][name] = value
        specs[name] = value[:2]
    weight_map = {}
    for i, bucket in enumerate(buckets):
        name = f"source-{i}.safetensors"
        raw_safetensors(directory / name, bucket)
        weight_map.update({key: name for key in bucket})
    source = SimpleNamespace(snapshot=directory, weight_map=weight_map, config=source_config())
    source_bytes = sum(len(v[2]) for v in tensors.values())
    removed = {name for module, g in selection.items() for name in recipe.source_names(module, g)}
    preserved = {name: value for name, value in tensors.items() if name not in removed}
    target_bytes = source_bytes - sum(len(tensors[n][2]) for n in removed) + sum(recipe.packed_bytes(g) for g in selection.values())
    target_count = len(tensors) - len(removed) + 4 * len(selection)
    for name, value in dict(BASE_TENSOR_COUNT=len(tensors), BASE_PAYLOAD_BYTES=source_bytes,
                            TARGET_TENSOR_COUNT=target_count, TARGET_PAYLOAD_BYTES=target_bytes,
                            PRESERVED_TENSOR_COUNT=len(preserved),
                            PRESERVED_BYTES=sum(len(v[2]) for v in preserved.values())).items():
        monkeypatch.setattr(export, name, value)
    monkeypatch.setattr(export, "reviewed_modules", lambda: selection)
    monkeypatch.setattr(export, "selected_modules", lambda _: selection)
    monkeypatch.setattr(export, "reviewed_source_specs", lambda: specs)
    candidates = {name: packed_fixture(tmp_path / f"candidate-{i}.safetensors", name, g)
                  for i, (name, g) in enumerate(selection.items())}
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "projection.json").write_text('{"metric_status":"ok"}')
    for name in ("LICENSE", "tokenizer.json", "tokenizer_config.json", "tekken.json", "chat_template.jinja", "params.json", "generation_config.json"):
        (directory / name).write_text("original " + name)
    return SimpleNamespace(source=source, selection=selection, candidates=candidates, output=output,
                           evidence=evidence, target_bytes=target_bytes, target_count=target_count)


def assemble_all(toy):
    for layer in range(72):
        replacements = {name: toy.candidates[name] for name in toy.selection if name.startswith(f"layers.{layer}.")}
        export.assemble_layer(toy.source, layer, replacements, toy.output, {"run": "toy"})


def test_layer_export_retires_selected_fp8_scales_and_preserves_indexer(toy_source):
    toy = toy_source
    names = {name: toy.candidates[name] for name in toy.selection if name.startswith("layers.3.")}
    receipt = export.assemble_layer(toy.source, 3, names, toy.output, {"run": "toy"})
    header, _ = read_header(toy.output / receipt["filename"])
    assert "layers.3.experts.0.w1.weight" not in header
    assert "layers.3.experts.0.w1.qscale_weight" not in header
    assert header["layers.3.attention.indexer.wq_b.weight"]["dtype"] == "F8_E4M3"
    assert header["layers.3.attention.indexer.wq_b.qscale_weight"]["dtype"] == "F32"
    assert header["layers.3.attention.wkv_a_with_mqa.svh"]["shape"] == [640]
    assert receipt["preserved_audit"]["tensors"] == 4
    for path in names.values():
        path.unlink()
    assert export.assemble_layer(toy.source, 3, names, toy.output, {"run": "toy"}) == receipt


def test_final_export_and_independent_audit_preserve_drafter_and_native_config(toy_source):
    toy = toy_source
    assemble_all(toy)
    summary = export.finalize_export(toy.source, toy.selection, {"output": toy.candidates["output"]},
                                     toy.output, {"run": "toy"}, toy.evidence)
    assert summary["shards"] == 73
    assert summary["tensors"] == toy.target_count
    assert summary["payload_bytes"] == toy.target_bytes
    assert (toy.output / "model.safetensors.index.json").read_bytes() == (toy.output / "consolidated.safetensors.index.json").read_bytes()
    q = json.loads((toy.output / "quantize_config.json").read_text())
    assert q["tensor_storage"]["output"]["bits_per_weight"] == 6
    assert q["tensor_storage"]["layers.3.experts.0.w1"]["bits_per_weight"] == 4
    assert q["native_quantization_config"] == toy.source.config["quantization_config"]
    audit = export.audit_completed_export(toy.source, toy.output, {"run": "toy"})
    assert audit["findings"] == []
    assert audit["summary"]["tensors"] == toy.target_count
    assert audit["native_audits"]["model-00073-of-00073.safetensors"]["tensors"] == 2
    assert "license: other" in (toy.output / "README.md").read_text()


@pytest.mark.parametrize("damage", ["mcg", "nan", "padding"])
def test_rejects_invalid_packed_numerics(toy_source, damage):
    toy = toy_source
    module = "layers.3.attention.wkv_a_with_mqa"
    path = toy.candidates[module]
    header, base = read_header(path)
    with path.open("r+b") as f:
        suffix = "mcg" if damage == "mcg" else "suh" if damage == "nan" else "svh"
        offset = header[module + "." + suffix]["data_offsets"][0]
        if damage == "padding":
            offset += 576 * 2
        f.seek(base + offset)
        f.write(struct.pack("<I", 0) if damage == "mcg" else struct.pack("<e", float("nan") if damage == "nan" else 1.))
    with pytest.raises(ValueError):
        export._check_packed(module, toy.selection[module], path)


def test_corrupt_native_bytes_never_commit_assembly_receipt(toy_source, monkeypatch):
    toy = toy_source
    assemble = export.assemble_shard

    def corrupt(entries, path):
        receipt = assemble(entries, path)
        h, base = read_header(path)
        with path.open("r+b") as f:
            f.seek(base + h["layers.0.attention_norm.weight"]["data_offsets"][0])
            f.write(b"\xff\xff")
        return receipt

    monkeypatch.setattr(export, "assemble_shard", corrupt)
    name = "layers.0.attention.wq"
    with pytest.raises(ValueError, match="preserved tensor bytes"):
        export.assemble_layer(toy.source, 0, {name: toy.candidates[name]}, toy.output, {"run": "toy"})
    assert not (toy.output / "layer-000-receipt.json").exists()


@pytest.mark.parametrize("damage", ["native", "mcg", "nan", "config", "metadata"])
def test_final_audit_rejects_damage_even_with_rehashed_receipt(toy_source, damage):
    toy = toy_source
    assemble_all(toy)
    export.finalize_export(toy.source, toy.selection, {"output": toy.candidates["output"]},
                           toy.output, {"run": "toy"}, toy.evidence)
    if damage == "config":
        p = toy.output / "config.json"
        value = json.loads(p.read_text())
        value["text_config"]["head_dim"] = 128
        p.write_text(json.dumps(value))
    elif damage == "metadata":
        p = toy.output / "quantize_config.json"
        value = json.loads(p.read_text())
        value["tensor_storage"]["output"]["bits_per_weight"] = 4
        p.write_text(json.dumps(value))
    else:
        path = toy.output / "model-00001-of-00073.safetensors"
        h, base = read_header(path)
        name = ("layers.0.attention_norm.weight" if damage == "native" else
                "layers.0.attention.wq." + ("mcg" if damage == "mcg" else "suh"))
        with path.open("r+b") as f:
            f.seek(base + h[name]["data_offsets"][0])
            f.write(b"\xff\xff" if damage == "native" else struct.pack("<I", 0) if damage == "mcg"
                    else struct.pack("<e", float("nan")))
        rp = toy.output / "layer-000-receipt.json"
        receipt = json.loads(rp.read_text())
        receipt["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        stat = path.stat()
        receipt["file_identity"] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, inode=stat.st_ino)
        rp.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        export.audit_completed_export(toy.source, toy.output, {"run": "toy"})


def test_module_patterns_exclude_kept_islands():
    patterns = [re.compile(p) for p in export._module_patterns()]
    assert all(any(p.fullmatch(name) for p in patterns) for name in recipe.reviewed_modules())
    for name in ("layers.3.attention.wkv_b", "layers.3.attention.indexer.wq_b", "layers.3.gate",
                 "mtp_dspark.backbone.layers.0.attention.wq", "vision_encoder.transformer.layers.0.attention.wq"):
        assert not any(p.fullmatch(name) for p in patterns)

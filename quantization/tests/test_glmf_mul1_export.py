"""A miniature end-to-end GLM Flash MC with real discovery and producer digest checks."""
import ctypes
from pathlib import Path
import sys

import pytest
from safetensors.torch import save_file
import torch
import xxhash

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glm5_flash.mul1_mc import export
from kimi_k3.uniform_k2.storage import atomic_json


def geometry(width):
    return dict(bits=width, in_features=128, out_features=128, stored_out_features=128,
                group="mtp_core" if width == 8 else "routed", codebook="mul1")


def fixture(tmp_path, monkeypatch):
    core = "model.language_model.layers.45.self_attn.q_b_proj"
    expert = "model.language_model.layers.45.mlp.experts.0.up_proj"
    native = export.source.NATIVE_EH + ".weight"
    src = tmp_path / "source"
    src.mkdir()
    for asset in export.ASSETS:
        (src / asset).write_text("{}")
    config = dict(model_type="glm5_next", text_config=dict(model_type="glm5_next_text"))
    atomic_json(src / "config.json", config)
    bf16 = {n + ".weight": torch.zeros((128, 128), dtype=torch.bfloat16) for n in (core, expert)}
    bf16[native] = torch.ones((128,256), dtype=torch.bfloat16)
    save_file(bf16, src / "input.safetensors")
    total = sum(t.numel() * t.element_size() for t in bf16.values())
    atomic_json(src / "model.safetensors.index.json", dict(metadata=dict(total_size=total),
                weight_map={n: "input.safetensors" for n in bf16}))
    monkeypatch.setattr(export.source, "audit", lambda _: dict(source_payload_bytes=total, source_tensors=3))
    monkeypatch.setattr(export.source, "core_modules", lambda _: {core: geometry(8)})
    monkeypatch.setattr(export.source, "expert_modules", lambda: {expert: geometry(4)})
    components = dict(agentic=dict(count=1, tokens=1024),
                      exl3_standard=dict(count=1, tokens=2048, input_ids_sha256="0" * 64))
    identity = dict(codebook="mul1", codebook_multiplier=export.package.MUL1, seed=787, sigma=0.025,
                    dataset=dict(repo_id="o/dataset", revision="a"*40), components=components,
                    code=dict(fork="b"*40), diagnostics=dict(dsa=dict(selected_groups=512, group_tokens=4,
                    max_tokens_before_first_group_exclusion=2051, sparse_calibration_count=1)))
    output = tmp_path / "output"
    writer = export.DirectMCWriter(output, src, identity)
    hashes, entries, packed_sizes = {}, {}, {}
    for kind, bits, name in (("shared", 4, expert), ("k8-compact", 8, core), ("k6-compact", 6, core)):
        packed = {n: torch.ones(shape, dtype={"I16":torch.int16,"F16":torch.float16,"I32":torch.int32}[dtype])
                  for n, (dtype, shape) in export.packed_specs(name, geometry(bits)).items()}
        packed[name + ".mul1"].fill_(ctypes.c_int32(export.package.MUL1).value)
        path = tmp_path / f"{bits}.safetensors"
        save_file(packed, path)
        entries[kind] = {n: (path, n) for n in packed}
        hashes[kind] = {n: xxhash.xxh3_128_hexdigest(v.numpy().tobytes()) for n, v in packed.items()}
        packed_sizes[kind] = sum(v.numel() * v.element_size() for v in packed.values())
    shared = packed_sizes["shared"] + 128 * 256 * 2
    monkeypatch.setattr(export.source, "payloads", lambda: dict(
        package_payload_bytes=shared + 32768 + packed_sizes["k8-compact"] + packed_sizes["k6-compact"],
        view_payload_bytes={k: shared + packed_sizes.get(k, 32768) for k in export.VIEWS}))
    group = dict(shared={**entries["shared"], native:(src / "input.safetensors", native)},
                 original={core+".weight":(src / "input.safetensors", core+".weight")},
                 k8=entries["k8-compact"], k6=entries["k6-compact"], packed_hashes=hashes)
    return writer, group, core, expert


def test_digest_mismatch_stops_before_group_commit(tmp_path, monkeypatch):
    writer, group, core, _ = fixture(tmp_path, monkeypatch)
    group["packed_hashes"]["k8-compact"][core + ".trellis"] = "0"*32
    with pytest.raises(ValueError, match="packed tensor differs"):
        writer.write_group("mtp", **group)
    assert not writer.group_complete("mtp")


def test_producer_bytes_survive_group_resume_and_default_loader(tmp_path, monkeypatch):
    writer, group, core, expert = fixture(tmp_path, monkeypatch)
    writer.write_group("mtp", **group)
    for path in tmp_path.glob("*.safetensors"):
        path.unlink()
    writer.write_group("mtp", **group)
    csv = tmp_path / "metrics.csv"
    csv.write_text("module,bits\n")
    receipt = writer.finalize(error_csv=csv, card_dir=tmp_path / "card")
    assert receipt["status"] == "validated-headers-and-byte-exactness"
    assert receipt["original_discovery"]["status"] == "passed"
    assert core + ".weight" in receipt["original_core_tensor_sha256"]
    eh_weight = export.source.NATIVE_EH + ".weight"
    assert eh_weight in receipt["original_core_tensor_sha256"]
    eh_shards = set()
    for view in export.VIEWS:
        index, config_name, quant_name, bits = export.package.mc_manifest.FILES[view]
        config = export.package.load_json(writer.output / config_name)
        meta = export.package.load_json(writer.output / quant_name)
        storage = meta["tensor_storage"]
        eh_shards.add(export.package.load_json(writer.output / index)["weight_map"][eh_weight])
        assert meta["retained_native_tensors"][eh_weight] == {"dtype":"BF16","shape":[128,256]}
        assert export.source.NATIVE_EH not in storage
        assert "mtp_bits" not in meta
        assert meta["safetensors_index"] == index
        assert config["text_config"]["quantization_config"] == config["quantization_config"]
        assert meta["meta"]["upstream_activations"] == "k8-compact"
        assert storage[expert]["mul1_multiplier"] == export.package.MUL1
        if view == "original":
            assert core not in storage
        else:
            assert storage[core]["bits_per_weight"] == bits
    assert len(eh_shards) == 1

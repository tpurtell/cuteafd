import json
from pathlib import Path
import struct
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2 import recipe
from inkling2.source import HeaderSource


def shard(path, specs):
    header, offset = {}, 0
    for name, (dtype, shape) in specs.items():
        size = 1
        for dimension in shape:
            size *= dimension
        size *= {"BF16": 2, "U8": 1}[dtype]
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[offset, offset + size])
        offset += size
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(offset))
    return offset


def source_fixture(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(dict(text_config={})))
    (tmp_path / "hf_quant_config.json").write_text("{}")
    name = "model-00001-of-00048.safetensors"
    size = shard(tmp_path / name, {"model.llm.test.weight": ("BF16", [2, 2])})
    shard(tmp_path / "mtp.safetensors", {"norm.weight": ("BF16", [2])})
    index = dict(metadata=dict(total_size=size), weight_map={"model.llm.test.weight": name})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    return index


def test_recipe_counts_geometry_and_scale_inclusive_storage():
    experts, k6, k8 = recipe.reviewed_expert_modules(), recipe.reviewed_core_modules(), recipe.reviewed_core_modules(core_bits=8)
    assert len(experts) == 55296
    assert len(k6) == len(k8) == 334
    assert len(recipe.reviewed_core_modules(include_mtp=False)) == 289
    assert sum(n.startswith("mtp.") for n in k6) == 45
    assert k6["model.llm.unembed"]["stored_out_features"] == 201088
    assert sum(recipe.packed_bytes(g) for g in experts.values()) == recipe.EXPERT_K4_BYTES
    assert sum(recipe.packed_bytes(g) for g in k6.values()) == recipe.CORE_K6_BYTES
    assert sum(recipe.packed_bytes(g) for g in k8.values()) == recipe.CORE_K8_BYTES
    assert recipe.EXPERT_K4_BYTES + recipe.NATIVE_SHARED_BYTES + recipe.CORE_SOURCE_BYTES + recipe.CORE_K8_BYTES + recipe.CORE_K6_BYTES == recipe.TARGET_PAYLOAD_BYTES
    assert recipe.ENCODE_COUNT == len(experts) + 2 * len(k6)
    assert len(recipe.GLOBAL_LAYERS) == 8 and len(recipe.LOCAL_LAYERS) == 40
    with pytest.raises(ValueError, match="K6 or K8"):
        recipe.reviewed_core_modules(core_bits=True)


def test_expert_interleaved_gate_up_rows_and_native_block_scale():
    prefix = "model.llm.layers.47.mlp.experts.383."
    gate, up, down = [recipe.expert_source(prefix + p) for p in ("gate_proj", "up_proj", "down_proj")]
    assert gate["weight"] == up["weight"] == "model.llm.layers.47.mlp.experts.w13_weight"
    assert gate["row_slice"] == [0, 4608, 2]
    assert up["row_slice"] == [1, 4608, 2]
    assert down["row_slice"] == [0, 4608, 1]
    assert down["scale"].endswith("w2_weight_scale")
    assert gate["scale_block_size"] == 32
    for invalid in (prefix + "other", prefix.replace("383", "384") + "gate_proj", prefix.replace("47", "48") + "gate_proj"):
        with pytest.raises(ValueError, match="unreviewed"):
            recipe.expert_source(invalid)


def test_header_only_source_namespaces_drafter_and_preserves_raw_location(tmp_path):
    source_fixture(tmp_path)
    source = HeaderSource(tmp_path, verify=False)
    assert source.specs["mtp.norm.weight"] == ("BF16", [2])
    assert source.locations["mtp.norm.weight"] == dict(shard="mtp.safetensors", source_name="norm.weight")
    assert source.payload_bytes == 12 and source.tensor_count == 2
    assert source.payload_totals == dict(declared=8, measured=12, reviewed=recipe.BASE_PAYLOAD_BYTES)
    assert source.as_inventory()["mapping"] == source.weight_map
    assert source.shard_offsets["mtp.safetensors"] > 8


@pytest.mark.parametrize("failure", ["mapping", "size", "unsafe", "mtp_prefix", "missing_mtp"])
def test_source_rejects_mismatched_or_unreviewed_headers(tmp_path, failure):
    index = source_fixture(tmp_path)
    if failure == "mapping":
        index["weight_map"] = {"missing": "model-00001-of-00048.safetensors"}
    elif failure == "size":
        index["metadata"]["total_size"] = 7
    elif failure == "unsafe":
        index["weight_map"]["model.llm.test.weight"] = "../model-00001-of-00048.safetensors"
    elif failure == "mtp_prefix":
        shard(tmp_path / "mtp.safetensors", {"mtp.norm.weight": ("BF16", [2])})
    else:
        (tmp_path / "mtp.safetensors").unlink()
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises((ValueError, FileNotFoundError)):
        HeaderSource(tmp_path, verify=False)


def test_pinned_mode_rejects_modified_index_before_payload_reads(tmp_path):
    source_fixture(tmp_path)
    with pytest.raises(ValueError, match="source index changed"):
        HeaderSource(tmp_path)

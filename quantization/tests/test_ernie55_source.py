"""Checkpoint-role and pinned revision witnesses for ERNIE 5.5 Flash."""
from collections import Counter
import json
from pathlib import Path
import struct
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checkpoint_cache import local_snapshot
from ernie55_flash import source


def test_reviewed_mc_roles():
    roles = source.reviewed_roles()
    assert Counter(roles.values()) == dict(core=573, routed_expert=69120, native=1133, flashindexer=12)
    assert roles["model.layers.7.self_attn.q_b_proj.weight"] == "core"
    assert roles["model.layers.7.self_attn.kv_b_proj.weight"] == "native"
    assert roles["model.layers.7.self_attn.core_attention.indexer.wq_b.weight"] == "native"
    assert roles["model.layers.7.self_attn.core_attention.indexer.linear_gate.weight"] == "flashindexer"
    assert roles["model.layers.0.self_attn.linear_o_group_proj"] == "core"
    assert roles["model.mtp.feature_projection.weight"] == "core"
    assert roles["model.mtp.2.block_sparse_moe.gate.routed_scaling_factor_param"] == "native"
    assert roles["model.mtp.2.block_sparse_moe.experts.511.w3.weight"] == "routed_expert"
    assert roles["lm_head.weight"] == "native"


def test_real_pinned_source_and_unindexed_flashindexer():
    snapshot = local_snapshot(source.BASE_REPO, source.BASE_REVISION,
                              files=[source.FLASHINDEXER_FILE], indexed=True)
    actual = source.HeaderSource(snapshot)
    assert actual.tensor_count == 70838
    assert actual.audit["indexed_bytes"] == 599573371912
    assert actual.payload_bytes - actual.audit["indexed_bytes"] == 3146496
    for name in actual.audit["flashindexer"]:
        assert actual.source_entries([name])[name] == (snapshot / source.FLASHINDEXER_FILE, name)
        assert actual.specs[name][0] == "BF16"
        assert actual.specs[name][1] == ([64] if name.endswith("_bias") else [64, 4096])


def test_reject_incomplete_snapshot(tmp_path):
    snapshot = tmp_path / source.BASE_REVISION
    snapshot.mkdir()
    with pytest.raises(ValueError, match="source file manifest"):
        source.HeaderSource(snapshot)


def test_reject_unreviewed_revision(tmp_path):
    with pytest.raises(ValueError, match="pinned ERNIE revision"):
        source.HeaderSource(tmp_path)


def synthetic_source(tmp_path, monkeypatch):
    # Preserve the 74-shard plus unindexed sidecar topology while using two-byte payloads.
    snapshot = tmp_path / source.BASE_REVISION
    (snapshot / "flashindexer").mkdir(parents=True)
    mapping, headers = {}, {}
    for shard_id in range(1, 76):
        name = f"w{shard_id}"
        if shard_id == 75:
            shard = source.FLASHINDEXER_FILE
        else:
            shard = f"model-{shard_id:05d}-of-00074.safetensors"
            mapping[name] = shard
        header = {name: dict(dtype="BF16", shape=[1], data_offsets=[0, 2])}
        encoded = json.dumps(header).encode()
        (snapshot / shard).write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(2))
        headers[shard] = header
    index = dict(weight_map=mapping, metadata=dict(total_size=148))
    docs = {"config.json": {}, "model.safetensors.index.json": index}
    monkeypatch.setattr(source, "_pinned_files", lambda *_: ({}, docs))
    roles = {**dict.fromkeys(mapping, "core"), "w75": "flashindexer"}
    monkeypatch.setattr(source, "reviewed_roles", lambda: roles)
    monkeypatch.setattr(source, "HEADERS_SHA256", source._sha(headers))
    return snapshot, index


@pytest.mark.parametrize(("change", "error"), [
    ("moved", "model-00001"), ("missing", "indexed tensor names"),
    ("indexed_sidecar", "indexed tensor names"), ("index_bytes", "index metadata"),
    ("dtype", "dtype, shape or byte range"), ("truncated", "payload length differs"),
])
def test_reject_inconsistent_source(tmp_path, monkeypatch, change, error):
    snapshot, index = synthetic_source(tmp_path, monkeypatch)
    names = index["weight_map"]
    if change == "moved":
        names["w1"], names["w2"] = names["w2"], names["w1"]
    elif change == "missing":
        del names["w4"]
    elif change == "indexed_sidecar":
        names["w75"] = source.FLASHINDEXER_FILE
    elif change == "index_bytes":
        index["metadata"]["total_size"] += 2
    elif change == "dtype":
        path = snapshot / source.FLASHINDEXER_FILE
        path.write_bytes(path.read_bytes().replace(b'"BF16"', b'"F16" '))
    else:
        path = snapshot / names["w3"]
        path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match=error):
        source.HeaderSource(snapshot)

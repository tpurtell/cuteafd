"""Byte-level repack and Hub digest contracts, without model weights."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

import huggingface_hub

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "quantization"))
SPEC = importlib.util.spec_from_file_location("strided_engram", ROOT / "quantization/strided_engram.py")
repack = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repack)


def test_repack_preserves_every_row_and_unmodified_tensor(tmp_path):
    rows = 200003  # Every ~66K-row part crosses the streaming chunk boundary.
    weight = np.arange(rows * 256, dtype=np.uint8).reshape(rows, 256)
    scale = np.arange(rows * 8, dtype=np.uint8).reshape(rows, 8) ^ 127
    extra = b"unchanged projection bytes"
    prefix = "layers.1.engram.embed"
    values = [(prefix + ".weight", "F8_E4M3", [rows, 256], weight.tobytes()),
              (prefix + ".scale", "F8_E8M0", [rows, 8], scale.tobytes()),
              ("layers.1.engram.q_weight", "U8", [len(extra)], extra)]
    header, offset = {}, 0
    for name, dtype, shape, data in values:
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[offset, offset + len(data)])
        offset += len(data)
    encoded = json.dumps(header).encode()
    source = tmp_path / "source.safetensors"
    source.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(v[3] for v in values))
    sha, blob, tensors = repack.inventory_hash(source)
    assert sha == hashlib.sha256(source.read_bytes()).hexdigest()
    assert tensors == {name: hashlib.sha256(data).hexdigest() for name, _, _, data in values}
    small = tmp_path / "config.json"
    small.write_bytes(b"{}\n")
    assert repack.inventory_hash(small)[1] == hashlib.sha1(b"blob 3\0{}\n").hexdigest()
    output = tmp_path / "output.safetensors"
    result = repack.repack_shard(source, output, 1, tensors)
    assert result["verified_all_rows"]
    assert len(result["parts"]) == 3
    first = 0
    for number, part in enumerate(result["parts"]):
        path = tmp_path / f"engram-01-part-{number:05}.safetensors"
        meta, base = repack.header(path)
        count = part["rows"]
        assert meta[part["tensor"]]["shape"] == [count, 264]
        assert meta["__metadata__"]["row_start"] == str(first)
        packed = np.memmap(path, dtype=np.uint8, mode="r", offset=base, shape=(count, 264))
        assert np.array_equal(packed[:, :256], weight[first:first + count])
        assert np.array_equal(packed[:, 256:], scale[first:first + count])
        assert result["files"][path.name]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        first += count
    assert first == rows
    assert prefix + ".weight" not in repack.header(output)[0]
    assert repack.inventory_hash(output)[2]["layers.1.engram.q_weight"] == hashlib.sha256(extra).hexdigest()
    assert not list(tmp_path.glob("*.partial"))
    tensors[prefix + ".weight"] = "bad hash"
    with pytest.raises(ValueError, match="tensor hash mismatch"):
        repack.repack_shard(source, tmp_path / "invalid.safetensors", 1, tensors)


def test_segmented_upload_inventory_and_storage_declaration(tmp_path, monkeypatch):
    import sys
    import types

    source, output = tmp_path / "source", tmp_path / "export"
    source.mkdir()
    output.mkdir()
    index, receipts = {}, {}
    for layer, number in [(1, 47), (14, 48)]:
        prefix = f"layers.{layer}.engram.embed"
        name = f"model-{number:05}-of-00048.safetensors"
        metadata, data = {}, b""
        for suffix, dtype, shape, payload in [
            ("weight", "F8_E4M3", [7, 256], bytes(range(256)) * 7),
            ("scale", "F8_E8M0", [7, 8], bytes(range(56))),
        ]:
            tensor = prefix + "." + suffix
            metadata[tensor] = dict(dtype=dtype, shape=shape, data_offsets=[len(data), len(data) + len(payload)])
            data += payload
            index[tensor] = name
        projection = f"layers.{layer}.engram.q_weight"
        metadata[projection] = dict(dtype="U8", shape=[1], data_offsets=[len(data), len(data) + 1])
        data += b"x"
        index[projection] = name
        encoded = json.dumps(metadata).encode()
        (source / name).write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
        receipts[name] = repack.repack_shard(source / name, output / name, layer,
                                            repack.inventory_hash(source / name)[2])
    (source / "config.json").write_text('{"quantization_config":{"unchanged":true}}')
    (source / "README.md").write_text("original")
    repack.write_json(source / "model.safetensors.index.json", {"weight_map": index, "metadata": {"total_size": 3698}})
    storage, rewritten = repack.storage_and_index(source, receipts)
    assert len(rewritten["weight_map"]) == 8  # Six parts plus two untouched tensors.
    assert rewritten["metadata"] == {"total_size": 3698}
    repack.write_json(output / "config.json", dict(json.loads((source / "config.json").read_text()), engram_storage=storage))
    repack.write_json(output / "model.safetensors.index.json", rewritten)
    (output / "README.md").write_text("repacked")
    siblings, evidence = [], {}
    for path in source.iterdir():
        sha, blob, tensors = repack.inventory_hash(path)
        siblings.append(dict(rfilename=path.name, blobId=blob, size=path.stat().st_size))
        evidence[path.name] = dict(sha256=sha, hub_hash=blob, tensors=tensors)
    repack.write_json(tmp_path / "source-hub.json", dict(sha=repack.REVISION, siblings=siblings))
    repack.write_json(tmp_path / "source-verified.json", evidence)
    repack.write_json(tmp_path / "repack-verified.json", dict(source=repack.SOURCE, target=repack.TARGET,
                      revision=repack.REVISION, source_files=len(siblings), shards=receipts))
    operations_seen, batches, workers = [], [], []
    trees = {"0" * 40: {}}

    class Operation:
        def __init__(self, path_in_repo, path_or_fileobj):
            self.path_in_repo = path_in_repo
            self.path_or_fileobj = path_or_fileobj
            data = Path(path_or_fileobj).read_bytes()
            self.upload_info = types.SimpleNamespace(sha256=hashlib.sha256(data).digest(), size=len(data))

    class Api:
        head = "0" * 40

        def create_repo(self, *args, **kwargs):
            pass

        def repo_info(self, repo, **kwargs):
            return types.SimpleNamespace(sha=self.head, private=False)

        def create_commit(self, repo, operations, **kwargs):
            assert kwargs["parent_commit"] == self.head and kwargs["num_threads"] == 16
            tree = dict(trees[self.head])
            assert not {op.path_in_repo for op in operations} & set(tree), "committed file sent again"
            for op in operations:
                tree[op.path_in_repo] = dict(bytes=op.upload_info.size, blob=op.upload_info.sha256.hex())
            operations_seen.extend(operations)
            batches.append(len(operations))
            Api.head = f"{len(batches):040x}"
            trees[Api.head] = tree
            return types.SimpleNamespace(oid=Api.head)

    import hub_lfs
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "CommitOperationAdd", Operation)
    monkeypatch.setattr(hub_lfs, "remote_tree", lambda api, repo, revision: trees[revision])
    monkeypatch.setattr(hub_lfs, "require_qualified_hub", lambda: None)
    monkeypatch.setattr(repack, "require_qualified_hub", lambda: None)
    args = types.SimpleNamespace(source=source, output=output, remote_output_host=None,
                                 upload_options=hub_lfs.UploadOptions(batch_files=4))
    repack.upload(args)
    assert len(operations_seen) == len(siblings) + 6
    assert all(size <= 4 for size in batches)
    assert {op.path_in_repo for op in operations_seen} >= {
        f"engram-{layer:02}-part-{part:05}.safetensors" for layer in [1, 14] for part in range(3)}
    repack.upload(args)  # A rerun resumes from the Hub tree and sends nothing again.
    assert len(operations_seen) == len(siblings) + 6
    receipt_path = tmp_path / "repack-verified.json"
    original_receipt = receipt_path.read_text()
    mixed = json.loads(original_receipt)
    mixed["shards"]["model-00047-of-00048.safetensors"]["source_tensor_hashes"] = {"wrong-source": "bad"}
    repack.write_json(receipt_path, mixed)
    with pytest.raises(ValueError, match="source tensor attestation mismatch"):
        repack.upload(args)
    receipt_path.write_text(original_receipt)
    # Rehashing the actual operation inputs must reject post-verification changes.
    (output / "engram-01-part-00000.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after verification"):
        repack.upload(args)


def test_ssh_upload_stream_seek_hash_and_bounds(tmp_path, monkeypatch):
    import subprocess

    data = bytes(range(256)) * 32769
    source = tmp_path / "remote shard.safetensors"
    source.write_bytes(data)
    popen = subprocess.Popen

    def local_ssh(command, **kwargs):
        assert command[0] == "ssh"
        assert command[-2] == "test-host"
        return popen(["bash", "-c", command[-1]], **kwargs)

    monkeypatch.setattr(repack.subprocess, "Popen", local_ssh)
    with repack.SshFile("test-host", source, len(data)) as stream:
        assert stream.read(512) == data[:512]
        stream.seek(0)
        digest = hashlib.sha256()
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
        assert digest.digest() == hashlib.sha256(data).digest()
        assert stream.tell() == len(data)
        stream.seek(0)
        assert stream.tell() == 0
        assert stream.read(100) == data[:100]
        assert stream.seek(200) == 200
        assert stream.read(5) == data[200:205]
        assert stream.seek(-10, 2) == len(data) - 10
        assert stream.read(100) == data[-10:]
        assert stream.read(1) == b""
        with pytest.raises(ValueError, match="bounded"):
            stream.read()
        with pytest.raises(ValueError, match="invalid seek"):
            stream.seek(-1)
    assert stream.closed


def test_strided_examples_only_change_checkpoint_and_share_release_matrix():
    matrix = json.loads((ROOT / "scripts/bench/release-smoke-v4.json").read_text())
    entries = {e["name"]: e for e in matrix["entries"]}
    def keys(path):
        return dict(line.split("=", 1) for line in path.read_text().splitlines()
                    if line and line[0].isupper() and "=" in line)
    for size in ["min", "max"]:
        paired = keys(ROOT / f"examples/configs/deepseek-v41-mxfp4-{size}.config")
        strided = keys(ROOT / f"examples/configs/v41-flash-strided-{size}.config")
        assert strided.pop("MODEL_ID") == repack.TARGET
        paired.pop("MODEL_ID")
        paired.pop("MODEL_REVISION")
        revision = strided.pop("MODEL_REVISION")
        assert revision == "51b3391435c83db9c38e87e114abf7bcfe7fc9c3"
        assert paired == strided
        original = entries[f"v41-flash-{size}"]
        candidate = entries[f"v41-flash-strided-{size}"]
        assert original["gpus"] == candidate["gpus"]
        assert original["sparks"] == candidate["sparks"]
        assert candidate["set"]["MODEL_ID"] == repack.TARGET
        assert candidate["set"]["MODEL_REVISION"] == revision

#!/usr/bin/env python3
"""Byte-preserving V4.1 Engram row interleave; verify before publication."""
import argparse
import hashlib
import io
import json
import os
import shlex
import subprocess
from pathlib import Path
import shutil
import struct
import time
import urllib.request

import hub_lfs
from hub_requirements import require_qualified_hub

SOURCE = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
TARGET = "wrldsuksgo2mars/DeepSeek-V4.1-Flash-1017-StridedPLE"
LAYOUT = {"layout": "fp8_row_interleaved_v1", "weight_bytes": 256,
          "scale_bytes": 8, "tensor_suffix": "strided"}
CHUNK = 16 * 1024 * 1024


def header(path):
    with path.open("rb") as file:
        size, = struct.unpack("<Q", file.read(8))
        if size > 64 * 1024 * 1024:
            raise ValueError("oversized safetensors header")
        return json.loads(file.read(size)), size + 8


def inventory_hash(path):
    """Hash the Hub file and each tensor in one sequential pass."""
    sha, blob = hashlib.sha256(), hashlib.sha1()
    blob.update(f"blob {path.stat().st_size}\0".encode())
    tensors = {}
    if path.suffix == ".safetensors":
        meta, base = header(path)
        spans = sorted((base + v["data_offsets"][0], base + v["data_offsets"][1], k)
                       for k, v in meta.items() if k != "__metadata__")
        tensors = {name: hashlib.sha256() for _, _, name in spans}
    else:
        spans = []
    position = index = 0
    with path.open("rb") as file:
        while chunk := file.read(CHUNK):
            sha.update(chunk)
            if not spans:
                blob.update(chunk)
            end = position + len(chunk)
            while index < len(spans) and spans[index][0] < end:
                start, stop, name = spans[index]
                tensors[name].update(memoryview(chunk)[max(start - position, 0):min(stop - position, len(chunk))])
                if stop > end:
                    break
                index += 1
            position = end
    return sha.hexdigest(), blob.hexdigest(), {k: v.hexdigest() for k, v in tensors.items()}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def repack_shard(source, output, layer, expected, part_rows=None):
    import numpy as np
    metadata, base = header(source)
    prefix = f"layers.{layer}.engram.embed"
    weight, scale = metadata[prefix + ".weight"], metadata[prefix + ".scale"]
    rows = weight["shape"][0]
    if weight["dtype"] != "F8_E4M3" or weight["shape"] != [rows, 256] or scale["dtype"] != "F8_E8M0" or scale["shape"] != [rows, 8]:
        raise ValueError("unexpected source Engram format")
    part_rows = part_rows or (rows + 2) // 3
    if part_rows <= 0 or part_rows * 264 > 45_000_000_000:
        raise ValueError("part rows must fit the 45 GB payload cap")
    weights = np.memmap(source, mode="r", dtype=np.uint8, offset=base + weight["data_offsets"][0], shape=(rows, 256))
    scales = np.memmap(source, mode="r", dtype=np.uint8, offset=base + scale["data_offsets"][0], shape=(rows, 8))
    restored_weight, restored_scale = hashlib.sha256(), hashlib.sha256()
    parts, files = [], {}
    provenance = {"engram_layout": LAYOUT["layout"], "source_revision": REVISION}
    for number, start in enumerate(range(0, rows, part_rows)):
        count = min(part_rows, rows - start)
        name = f"{prefix}.strided_{number:05}"
        path = output.parent / f"engram-{layer:02}-part-{number:05}.safetensors"
        temporary = path.with_suffix(".partial")
        meta = {"__metadata__": dict(provenance, row_start=str(start)),
                name: dict(dtype="U8", shape=[count, 264], data_offsets=[0, count * 264])}
        encoded = json.dumps(meta, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        with temporary.open("wb") as file:
            file.write(struct.pack("<Q", len(encoded)) + encoded)
            for first in range(start, start + count, 65536):
                stop = min(first + 65536, start + count)
                packed = np.empty((stop - first, 264), dtype=np.uint8)
                packed[:, :256], packed[:, 256:] = weights[first:stop], scales[first:stop]
                file.write(packed.tobytes())
            file.flush()
            os.fsync(file.fileno())
        _, newbase = header(temporary)
        packed = np.memmap(temporary, mode="r", dtype=np.uint8, offset=newbase, shape=(count, 264))
        for first in range(0, count, 65536):
            stop = min(first + 65536, count)
            w, sc = packed[first:stop, :256].copy(), packed[first:stop, 256:].copy()
            if not np.array_equal(w, weights[start + first:start + stop]) or not np.array_equal(sc, scales[start + first:start + stop]):
                raise ValueError(f"row mismatch at {start + first}")
            restored_weight.update(w.tobytes())
            restored_scale.update(sc.tobytes())
        for row in {0, count // 2, count - 1}:
            if packed[row].tobytes() != weights[start + row].tobytes() + scales[start + row].tobytes():
                raise ValueError(f"sample mismatch at {start + row}")
        del packed
        temporary.replace(path)
        digest, _, _ = inventory_hash(path)
        files[path.name] = dict(sha256=digest, size=path.stat().st_size)
        parts.append(dict(tensor=name, rows=count))
    if restored_weight.hexdigest() != expected[prefix + ".weight"] or restored_scale.hexdigest() != expected[prefix + ".scale"]:
        raise ValueError("deinterleaved source tensor hash mismatch")
    # Retain original small projection tensors in their original shard.
    new, offset = {}, 0
    for name, info in metadata.items():
        if name == "__metadata__" or name in (prefix + ".weight", prefix + ".scale"):
            continue
        size = info["data_offsets"][1] - info["data_offsets"][0]
        new[name] = dict(info, data_offsets=[offset, offset + size])
        offset += size
    encoded = json.dumps(dict(new, __metadata__=provenance), separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    temporary = output.with_suffix(".partial")
    with temporary.open("wb") as file, source.open("rb") as original:
        file.write(struct.pack("<Q", len(encoded)) + encoded)
        for name in new:
            start, stop = metadata[name]["data_offsets"]
            original.seek(base + start)
            remaining = stop - start
            while remaining:
                chunk = original.read(min(CHUNK, remaining))
                if not chunk:
                    raise EOFError(name)
                file.write(chunk)
                remaining -= len(chunk)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(output)
    digest, _, tensors = inventory_hash(output)
    if any(tensors[name] != expected[name] for name in new):
        raise ValueError("untouched tensor mismatch")
    files[output.name] = dict(sha256=digest, size=output.stat().st_size)
    return dict(rows=rows, parts=parts, files=files, source_tensor_hashes=expected, verified_all_rows=True)


def storage_and_index(source, result):
    storage = dict(LAYOUT, parts={str(layer): result[f"model-{shard:05}-of-00048.safetensors"]["parts"]
                                  for layer, shard in [(1, 47), (14, 48)]})
    index = json.loads((source / "model.safetensors.index.json").read_text())
    for layer in (1, 14):
        prefix = f"layers.{layer}.engram.embed"
        shard = index["weight_map"].pop(prefix + ".weight")
        if index["weight_map"].pop(prefix + ".scale") != shard:
            raise ValueError("source Engram planes are in different shards")
        for number, part in enumerate(storage["parts"][str(layer)]):
            index["weight_map"][part["tensor"]] = f"engram-{layer:02}-part-{number:05}.safetensors"
    return storage, index


def repack(args):
    source, output = args.source, args.output
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 220 * 1024**3:
        raise ValueError("repack needs at least 220 GiB free on the output filesystem")
    url = f"https://huggingface.co/api/models/{SOURCE}/revision/{REVISION}?blobs=true"
    remote = json.load(urllib.request.urlopen(url))
    write_json(output.parent / "source-hub.json", remote)
    evidence_path = output.parent / "source-verified.json"
    evidence = json.loads(evidence_path.read_text()) if evidence_path.exists() else {}
    for item in remote["siblings"]:
        name = item["rfilename"]
        path = source / name
        # A prior receipt is not proof that the current files are unchanged.
        sha, blob, tensors = inventory_hash(path)
        expected = item.get("lfs", {}).get("sha256") or item["blobId"]
        actual = sha if item.get("lfs") else blob
        if actual != expected or path.stat().st_size != item["size"]:
            raise ValueError(f"Hub hash/size mismatch: {name}")
        evidence[name] = dict(sha256=sha, hub_hash=expected, tensors=tensors)
        write_json(evidence_path, evidence)
        print(f"source verified {name}", flush=True)
    result = {}
    for layer, shard in [(1, 47), (14, 48)]:
        name = f"model-{shard:05}-of-00048.safetensors"
        result[name] = repack_shard(source / name, output / name, layer, evidence[name]["tensors"])
        print(f"repacked and fully verified {name}", flush=True)
    config = json.loads((source / "config.json").read_text())
    storage, index = storage_and_index(source, result)
    config["engram_storage"] = storage
    write_json(output / "config.json", config)
    write_json(output / "model.safetensors.index.json", index)
    card = """---
license: mit
library_name: transformers
pipeline_tag: image-text-to-text
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
tags:
- deepseek_v41
- text-generation
- image-text-to-text
- fp8
- 8-bit
---
# DeepSeek V4.1 Flash 1017 StridedPLE

Byte-preserving storage repack of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, under the original MIT license (see LICENSE).

Only the two Engram embedding tables (layers 1 and 14) change storage. Each `layers.{layer}.engram.embed.strided_{part:05}` tensor is U8 `[rows, 264]`: bytes 0..256 are the original F8_E4M3 weight row, bytes 256..264 are the original F8_E8M0 scales. These bytes are opaque storage, not quantized integer values. All values are byte-identical; there is no requantization. Every row was compared and the deinterleaved planes matched full source tensor SHA256 hashes. Every untouched tensor retains its source bytes. Shards 1..46 are unchanged; 47/48 retain all other tensors unchanged.

`config.json.engram_storage` declares `layout=fp8_row_interleaved_v1`, `weight_bytes=256`, `scale_bytes=8`, `tensor_suffix=strided`, plus an ordered `parts` map per layer. Each part declares `tensor` and `rows`; cumulative row counts define its global range. Each table has three contiguous row-range parts in separate shards below 45 GB; concatenate their row spaces in declared order. Rewritten safetensors headers also carry `engram_layout=fp8_row_interleaved_v1` in metadata. The original quantization_config is unchanged. A loader supporting this storage layout (CuteAFD) is required; unmodified Transformers does not load the interleaved table. GPU math is unchanged: the CPU gathers one contiguous row then splits existing weight/scale staging. No speed improvement is claimed. `MAPPED_SCALE_PRELOAD_GIB` does not preload interleaved scales; `ENGRAM_WARM` warms the whole table.

The original model implementation, tokenizer, license, and other repository assets are preserved. Refer to the source model card for model capabilities and usage requirements.
"""
    card = card.replace("tags:\n- deepseek_v41\n- text-generation\n- image-text-to-text\n- fp8\n- 8-bit",
                        "tags: " + json.dumps(remote.get("tags", [])))
    (output / "README.md").write_text(card)
    write_json(output.parent / "repack-verified.json", dict(source=SOURCE, revision=REVISION, target=TARGET,
        source_files=len(evidence), shards=result, completed_at=time.time()))


class SshFile(io.BufferedIOBase):
    """Seekable, bounded-memory SSH stream; credentials stay on the publisher."""

    def __init__(self, host, path, size):
        self.host, self.path, self.size = host, path, size
        self.position, self.process = 0, None

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def _stop(self):
        if self.process is not None:
            self.process.stdout.close()
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None

    def seek(self, offset, whence=os.SEEK_SET):
        target = offset if whence == os.SEEK_SET else self.position + offset if whence == os.SEEK_CUR else self.size + offset
        if whence not in (os.SEEK_SET, os.SEEK_CUR, os.SEEK_END) or target < 0:
            raise ValueError("invalid seek")
        if target != self.position:
            self._stop()
            self.position = target
        return target

    def read(self, size=-1):
        # Hub hashing and multipart upload request bounded chunks. Refuse a full
        # 100-GB materialization if a future Hub client changes that contract.
        if size < 0 or size > 64 * 1024 * 1024:
            raise ValueError("SSH upload reads must be bounded to at most 64 MiB")
        size = min(size, max(0, self.size - self.position))
        if not size:
            return b""
        if self.process is None:
            command = f"dd if={shlex.quote(str(self.path))} bs=16M iflag=skip_bytes skip={self.position} status=none"
            self.process = subprocess.Popen(["ssh", "-oBatchMode=yes", "-oServerAliveInterval=30",
                "-oServerAliveCountMax=3", self.host, command], stdout=subprocess.PIPE)
        data = self.process.stdout.read(size)
        if len(data) != size:
            raise EOFError(f"short remote read: {self.path} at {self.position}")
        self.position += len(data)
        return data

    def close(self):
        self._stop()
        super().close()


def upload(args):
    require_qualified_hub()
    from contextlib import ExitStack
    from huggingface_hub import HfApi, CommitOperationAdd
    output, source = args.output, args.source
    result = json.loads((output.parent / "repack-verified.json").read_text())
    expected_shards = {f"model-{shard:05}-of-00048.safetensors" for shard in (47, 48)}
    if (result["source"] != SOURCE or result["target"] != TARGET or result["revision"] != REVISION
            or set(result["shards"]) != expected_shards
            or not all(x["verified_all_rows"] for x in result["shards"].values())):
        raise ValueError("missing verified repack")
    remote = json.loads((output.parent / "source-hub.json").read_text())
    evidence = json.loads((output.parent / "source-verified.json").read_text())
    if remote["sha"] != REVISION or len(remote["siblings"]) != result["source_files"]:
        raise ValueError("source inventory revision/count mismatch")
    for shard, receipt in result["shards"].items():
        if receipt.get("source_tensor_hashes") != evidence[shard]["tensors"]:
            raise ValueError(f"repack source tensor attestation mismatch: {shard}")
    storage, original_index = storage_and_index(source, result["shards"])
    original_config = json.loads((source / "config.json").read_text())
    original_config["engram_storage"] = storage
    if json.loads((output / "config.json").read_text()) != original_config:
        raise ValueError("publication config differs beyond storage declaration")
    if json.loads((output / "model.safetensors.index.json").read_text()) != original_index:
        raise ValueError("publication index differs beyond Engram tensor names")
    generated = {name: info for receipt in result["shards"].values() for name, info in receipt["files"].items()}
    for layer in (1, 14):
        parts = storage["parts"][str(layer)]
        source_shard = f"model-{47 if layer == 1 else 48:05}-of-00048.safetensors"
        if len(parts) != 3 or sum(part["rows"] for part in parts) != result["shards"][source_shard]["rows"]:
            raise ValueError("invalid row-part receipt")
    expected_generated = expected_shards | {f"engram-{layer:02}-part-{part:05}.safetensors"
                                          for layer in (1, 14) for part in range(3)}
    if set(generated) != expected_generated:
        raise ValueError("incomplete generated shard receipt")
    if any(info["size"] > 45_000_001_024 for info in generated.values()):
        raise ValueError("generated shard exceeds publication cap")
    api = HfApi()
    with ExitStack() as stack:
        operations = []
        source_items = {item["rfilename"]: item for item in remote["siblings"]}
        for name in sorted(set(source_items) | set(generated)):
            item = source_items.get(name)
            expected_hub = (item.get("lfs", {}).get("sha256") or item["blobId"]) if item else None
            if item and evidence[name]["hub_hash"] != expected_hub:
                raise ValueError(f"source attestation mismatch: {name}")
            if name in generated and args.remote_output_host:
                # Stream verified remote parts without staging checkpoint bytes locally.
                path = args.remote_output / name
                size = int(subprocess.check_output(["ssh", args.remote_output_host,
                    f"stat -c %s -- {shlex.quote(str(path))}"], text=True))
                value = stack.enter_context(SshFile(args.remote_output_host, path, size))
            else:
                value = str(output / name if (output / name).exists() else source / name)
            operation = CommitOperationAdd(path_in_repo=name, path_or_fileobj=value)
            expected = generated[name]["sha256"] if name in generated else evidence[name]["sha256"]
            if name not in {"README.md", "config.json", "model.safetensors.index.json"} and operation.upload_info.sha256.hex() != expected:
                raise ValueError(f"publication bytes changed after verification: {name}")
            if isinstance(value, SshFile):
                value._stop()
            operations.append(operation)
            print(f"publication bytes checked {name}", flush=True)
        # Upload only after all byte checks, as verified parallel HTTP LFS
        # batches on a guarded parent. Each SSH stream is its own process, so
        # parallel workers stay bounded-memory; a rerun skips committed files.
        api.create_repo(TARGET, repo_type="model", exist_ok=True)
        info = api.repo_info(TARGET, repo_type="model")
        by_name = {operation.path_in_repo: operation for operation in operations}
        rows = {name: dict(bytes=operation.upload_info.size, sha256=operation.upload_info.sha256.hex())
                for name, operation in by_name.items()}
        tree = hub_lfs.remote_tree(api, TARGET, info.sha)
        done = {name for name, row in tree.items() if name in rows and hub_lfs.matches(row, rows[name])}
        if set(tree) - done - {".gitattributes"}:
            raise ValueError("destination holds files outside this publication")

        def verify(commit, batch, batch_operations):
            done.update(batch)
            remote = hub_lfs.remote_tree(api, TARGET, commit)
            if set(remote) - {".gitattributes"} != done or not all(
                    hub_lfs.matches(remote[name], rows[name]) for name in done):
                raise ValueError("published Engram inventory differs")
            for name in batch:
                value = by_name[name].path_or_fileobj
                if isinstance(value, SshFile):
                    value.close()
            print(f"publication objects committed {len(done)}/{len(rows)}", flush=True)

        pending = [name for name in hub_lfs.upload_order(rows) if name not in done]
        revision = hub_lfs.commit_batches(api, TARGET, rows, pending, revision=info.sha,
            commit_message="Publish byte-identical row-interleaved Engram storage", options=args.upload_options,
            check_sources=lambda batch: None,  # Byte checks above hashed every operation input.
            verify=verify, operation=lambda name, row: by_name[name]) if pending else info.sha
    write_json(output.parent / "upload.json", dict(repo=TARGET, commit=revision, revision=revision))
    print(revision, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["repack", "upload"])
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--remote-output-host", help="Stream rewritten shards over SSH during upload")
    parser.add_argument("--remote-output", type=Path, help="Remote export directory (upload only)")
    hub_lfs.add_arguments(parser)
    arguments = parser.parse_args()
    arguments.upload_options = hub_lfs.options_from(arguments)
    if bool(arguments.remote_output_host) != bool(arguments.remote_output):
        parser.error("--remote-output-host and --remote-output must be specified together")
    globals()[arguments.action](arguments)

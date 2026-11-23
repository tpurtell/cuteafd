"""Durable rolling boundaries and bounded, byte-preserving shard assembly."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
from functools import lru_cache


@lru_cache(maxsize=1)
def checkpoint_io():
    """Load the pinned lightweight file helper without model/GPU imports."""
    path = Path(__file__).resolve().parents[3] / "third_party/gptqmodel/gptqmodel/utils/checkpoint_io.py"
    spec = importlib.util.spec_from_file_location("_cuteafd_checkpoint_io", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".json-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as file:
            file.write(canonical(value) + "\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256_file(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def file_identity(path):
    stat = Path(path).stat()
    return dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, inode=stat.st_ino)


def verify_receipt(path, receipt, *, checksum=False):
    if file_identity(path) != receipt["file_identity"]:
        raise ValueError(f"committed file changed: {path}")
    algorithm = receipt.get("hash_algorithm", "sha256")
    if algorithm not in ("sha256", "xxh3-128"):
        raise ValueError(f"unsupported committed checksum: {algorithm}")
    segmented = receipt.get("storage_format") == "safetensors-segment-v1"
    offset, length = (receipt["offset"], receipt["bytes"]) if segmented else (0, None)
    if (type(offset) is not int or offset < 0 or
            (length is not None and (type(length) is not int or length <= 0)) or
            (length is not None and offset + length > receipt["file_identity"]["bytes"])):
        raise ValueError(f"committed file extent differs: {path}")
    if checksum:
        actual = (sha256_file(path) if algorithm == "sha256" else
                  checkpoint_io().hash_file(path, offset=offset, length=length))
        if actual != receipt["sha256" if algorithm == "sha256" else "xxh3_128"]:
            raise ValueError(f"committed file checksum differs: {path}")


def commit_safetensors(path, tensors, *, metadata=None, hash_algorithm="xxh3-128"):
    """Stream internal scratch once; SHA256 is explicit for protocol outputs."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    receipt = checkpoint_io().write_safetensors(path,
        {key: value.detach().cpu().contiguous() for key, value in tensors.items()},
        metadata=metadata, hash_algorithm=hash_algorithm)
    # Preserve the original compact receipt contract. In particular, callers
    # may use `offset` for corpus rows and `tensors` for an inventory count.
    return dict(checksum_fields(receipt), hash_algorithm=receipt["hash_algorithm"],
                file_identity=file_identity(path))


def checksum_identity(receipt):
    algorithm = receipt.get("hash_algorithm", "sha256")
    if algorithm not in ("sha256", "xxh3-128"):
        raise ValueError("unsupported checkpoint checksum")
    return dict(algorithm=algorithm, value=receipt["sha256" if algorithm == "sha256" else "xxh3_128"])


def checksum_fields(receipt):
    """Retain legacy SHA field shape; explicitly tag new scratch checksums."""
    checksum = checksum_identity(receipt)
    if checksum["algorithm"] == "sha256":
        return dict(sha256=checksum["value"])
    return dict(hash_algorithm=checksum["algorithm"], xxh3_128=checksum["value"])


def scratch_receipt(path):
    """One final XXH3 scan for mutable mmap layouts; no serializer readback."""
    return dict(hash_algorithm="xxh3-128", xxh3_128=checkpoint_io().hash_file(path),
                file_identity=file_identity(path))


def read_header(path, *, offset=0, length=None):
    """Validate one whole file or one bounded safetensors record in a segment."""
    with Path(path).open("rb") as file:
        size_on_disk = os.fstat(file.fileno()).st_size
        if (type(offset) is not int or offset < 0
                or (length is None and offset != 0)
                or (length is not None and (type(length) is not int or length < 10))
                or offset + (length or 0) > size_on_disk):
            raise ValueError("invalid safetensors record extent")
        region = size_on_disk if length is None else length
        file.seek(offset)
        raw = file.read(8)
        if len(raw) != 8:
            raise ValueError("truncated safetensors prefix")
        size, = struct.unpack("<Q", raw)
        if not 2 <= size <= min(64 * 1024 * 1024, region - 8):
            raise ValueError("invalid safetensors header size")
        header = json.loads(file.read(size))
        payload_size = region - 8 - size
    cursor = 0
    for start, end in sorted(value["data_offsets"] for key, value in header.items() if key != "__metadata__"):
        if start != cursor or end < start:
            raise ValueError("non-contiguous safetensors payload")
        cursor = end
    if cursor != payload_size:
        raise ValueError("safetensors payload length differs")
    widths = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
              "F8_E8M0": 1, "I16": 2, "U16": 2, "BF16": 2, "F16": 2,
              "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}
    for key, value in header.items():
        if key == "__metadata__":
            continue
        count = 1
        for dimension in value["shape"]:
            if type(dimension) is not int or dimension < 0:
                raise ValueError("invalid safetensors tensor dimension")
            count *= dimension
        if value["dtype"] not in widths or count * widths[value["dtype"]] != value["data_offsets"][1] - value["data_offsets"][0]:
            raise ValueError("safetensors tensor shape/dtype extent differs")
    return header, offset + size + 8


def assemble_shard(entries, path, *, chunk_bytes=16 * 1024 * 1024, tensor_xxh3=None):
    """Copy tensor byte spans once, hashing during writes with bounded RAM."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if not entries or not 1 <= chunk_bytes <= 64 * 1024 * 1024:
        raise ValueError("invalid shard entries/chunk bound")
    if tensor_xxh3 is not None:
        import xxhash
        if tensor_xxh3.keys() != entries.keys():
            raise ValueError("packed tensor digest inventory differs")
    sources, spans, header, offset = {}, [], {}, 0
    for name, entry in sorted(entries.items()):
        if len(entry) == 2:
            source_path, source_name = entry
            start_record, record_length = 0, None
        elif len(entry) == 4:
            source_path, source_name, start_record, record_length = entry
        else:
            raise ValueError("invalid shard source entry")
        source_path = Path(source_path)
        key = (source_path, start_record, record_length)
        if key not in sources:
            sources[key] = read_header(source_path, offset=start_record, length=record_length)
        source_header, base = sources[key]
        value = source_header[source_name]
        start, end = value["data_offsets"]
        header[name] = dict(dtype=value["dtype"], shape=value["shape"], data_offsets=[offset, offset + end - start])
        offset += end - start
        spans.append((name, source_path, base + start, end - start))
    encoded = canonical(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    digest = hashlib.sha256()
    with temporary.open("wb") as output:
        prefix = struct.pack("<Q", len(encoded)) + encoded
        output.write(prefix)
        digest.update(prefix)
        for name, source_path, position, length in spans:
            tensor_hash = xxhash.xxh3_128() if tensor_xxh3 is not None else None
            with source_path.open("rb") as source:
                source.seek(position)
                while length:
                    block = source.read(min(length, chunk_bytes))
                    if not block:
                        raise ValueError("truncated source tensor")
                    output.write(block)
                    digest.update(block)
                    length -= len(block)
                    if tensor_hash is not None:
                        tensor_hash.update(block)
            if tensor_hash is not None and tensor_hash.hexdigest() != tensor_xxh3[name]:
                raise ValueError(f"packed tensor differs from producer receipt: {name}")
        output.flush()
        os.fsync(output.fileno())
    if temporary.stat().st_size != 8 + len(encoded) + offset:
        raise ValueError("assembled shard length differs")
    os.replace(temporary, path)
    fsync_directory(path.parent)
    return dict(sha256=digest.hexdigest(), file_identity=file_identity(path), payload_bytes=offset)


class RollingBoundary:
    """The previous complete boundary survives until the next is committed.

    Publication of current.json is the retirement barrier. A crash before it
    preserves the old bank; a crash afterward permits idempotent old-bank
    cleanup. At most two banks are retained, including partial successor work.
    """
    def __init__(self, root, identity):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = identity

    def current(self, *, checksum=False):
        path = self.root / "current.json"
        if not path.exists():
            return None
        record = json.loads(path.read_text())
        if record["identity"] != self.identity:
            raise ValueError("boundary run identity differs")
        directory = self.root / record["directory"]
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest != record["manifest"]:
            raise ValueError("boundary manifest differs from commitment")
        for entry in manifest["batches"]:
            verify_receipt(directory / entry["filename"], entry, checksum=checksum)
        return record

    def directory(self, layer):
        return self.root / f"layer-{layer:03d}"

    def commit(self, layer, batches):
        if not batches or len({entry["filename"] for entry in batches}) != len(batches):
            raise ValueError("empty/duplicate boundary inventory")
        old = self.current()
        if old is not None and layer != old["layer"] + 1:
            raise ValueError("boundary must advance exactly one layer")
        if old is None and layer != -1:
            raise ValueError("first boundary must contain embeddings")
        directory = self.directory(layer)
        for entry in batches:
            verify_receipt(directory / entry["filename"], entry)
        manifest = dict(layer=layer, identity=self.identity, batches=batches)
        atomic_json(directory / "manifest.json", manifest)
        record = dict(layer=layer, identity=self.identity, directory=directory.name, manifest=manifest)
        atomic_json(self.root / "current.json", record)
        self.retire_old()
        return record

    def retire_old(self):
        current = self.current()
        if current is None:
            return
        for directory in self.root.glob("layer-*"):
            try:
                layer = int(directory.name.removeprefix("layer-"))
            except ValueError:
                continue
            if layer < current["layer"]:
                if directory.is_symlink():
                    raise ValueError("refuse symlink in boundary retirement")
                shutil.rmtree(directory)
        fsync_directory(self.root)

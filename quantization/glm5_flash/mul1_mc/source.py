"""Pinned GLM Flash MUL1 source-header audit and direct MC projection inventory."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

from glm5.core_k6.recipe import packed_bytes
from glm5_flash.core_k6 import recipe as flash
from kimi_k3.uniform_k2.storage import atomic_json, canonical, read_header

TARGET_REPO = "wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K4-MC-MUL1"
BASE_REPO = flash.BASE_REPO
BASE_REVISION = "a5b45eb41df6402735dedc900be14a42e8d5e538"
CONFIG_SHA256 = "33e63ec7fe607658be712bd6dd3c16c6549960d8e7f0483d34b939881b55f943"
# Canonical name -> {bytes, oid}, from the pinned Hub revision's siblings
# with blobs=true. OIDs are LFS SHA256 for weights, Git blob SHA1 for assets.
SOURCE_MANIFEST_SHA256 = "762d842a4d3f3ce3266c034b5f7776f0fde8aeadf9512e5eba2203b9aa61d5f8"
MUL1_MULTIPLIER = 0x83DCD12D
NATIVE_EH = flash.PREFIX + "layers.45.eh_proj"


def core_modules(bits):
    """Keep the MTP input projection source-native after the held-out pilot."""
    return {name: {**geometry, "bits": bits, "codebook": "mul1"}
            for name, geometry in flash.reviewed_core_modules(core_bits=bits).items() if name != NATIVE_EH}


def expert_modules():
    return {name: {**geometry, "codebook": "mul1"}
            for name, geometry in flash.reviewed_expert_modules().items()}


def payloads():
    experts = expert_modules()
    core = core_modules(8)
    original_core = sum(2 * g["in_features"] * g["out_features"] for g in core.values())
    original_expert = sum(2 * g["in_features"] * g["out_features"] for g in experts.values())
    packed_expert = sum(packed_bytes(g) for g in experts.values())
    native = flash.BASE_PAYLOAD_BYTES - original_core - original_expert
    shared = native + packed_expert
    compact = {f"k{bits}-compact": sum(packed_bytes(g, bits=bits) for g in core.values())
               for bits in (8, 6)}
    views = {"original": original_core, **compact}
    return dict(shared_expert_bytes=packed_expert, retained_native_bytes=native,
                view_payload_bytes={view: shared + count for view, count in views.items()},
                package_payload_bytes=shared + sum(views.values()),
                expert_modules=len(experts), core_modules_per_view=len(core))


def cache_manifest(snapshot):
    """Verify the pinned Hub inventory without rereading weight payloads over the LAN."""
    blobs = snapshot.parent.parent / "blobs"
    files = {}
    for path in sorted(snapshot.iterdir()):
        target = Path(os.path.normpath(path.parent / path.readlink()))
        if target.parent != blobs or not target.is_file():
            raise ValueError(f"unexpected Hub cache blob: {path.name}")
        size = target.stat().st_size
        if path.suffix != ".safetensors":
            content = path.read_bytes()
            if len(target.name) == 64:
                digest = hashlib.sha256(content).hexdigest()
            else:
                digest = hashlib.sha1(f"blob {size}\0".encode() + content).hexdigest()
            if digest != target.name:
                raise ValueError(f"source asset content differs: {path.name}")
        files[path.name] = dict(bytes=size, oid=target.name)
    digest = hashlib.sha256(canonical(files).encode()).hexdigest()
    if digest != SOURCE_MANIFEST_SHA256:
        raise ValueError("source file names, sizes or Hub OIDs differ from pinned revision")
    return files


def audit(snapshot):
    snapshot = Path(snapshot).resolve(strict=True)
    if snapshot.name != BASE_REVISION:
        raise ValueError("unreviewed GLM Flash source revision")
    files = cache_manifest(snapshot)
    config = (snapshot / "config.json").read_bytes()
    if hashlib.sha256(config).hexdigest() != CONFIG_SHA256:
        raise ValueError("source config file digest differs")
    flash.validate_config(json.loads(config))
    index = json.loads((snapshot / "model.safetensors.index.json").read_bytes())
    indexed = index["weight_map"]
    expected = flash.reviewed_source_specs()
    if indexed.keys() != expected.keys():
        raise ValueError("source index differs from reviewed module inventory")
    by_shard = defaultdict(set)
    for name, shard in indexed.items():
        by_shard[shard].add(name)
    if set(by_shard) != {name for name in files if name.endswith(".safetensors")}:
        raise ValueError("source index shard set differs from Hub manifest")
    total = 0
    for shard, names in by_shard.items():
        header, _ = read_header(snapshot / shard)
        header.pop("__metadata__", None)
        if header.keys() != names:
            raise ValueError(f"source header names differ: {shard}")
        for name, spec in header.items():
            if (spec["dtype"], spec["shape"]) != expected[name]:
                raise ValueError(f"source dtype or geometry differs: {name}")
            start, end = spec["data_offsets"]
            total += end - start
    if total != flash.BASE_PAYLOAD_BYTES or index["metadata"]["total_size"] != total:
        raise ValueError("source payload total differs")
    return dict(status="passed", audit="hub-cache-inventory-and-safetensors-headers-v1",
                repo_id=BASE_REPO, revision=BASE_REVISION, config_sha256=CONFIG_SHA256,
                manifest_sha256=SOURCE_MANIFEST_SHA256, files=files,
                source=str(snapshot), source_payload_bytes=total, source_tensors=len(indexed),
                shards=len(by_shard), **payloads())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    atomic_json(args.output, audit(args.snapshot))


if __name__ == "__main__":
    main()

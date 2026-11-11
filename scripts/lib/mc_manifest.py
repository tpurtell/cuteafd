"""Versioned MC discovery metadata shared by publication and loader fixtures.

No tensor payload/header reads: only the selected three JSON files are needed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "cuteafd-mc-v1"
MANIFEST = "cuteafd-mc.json"
MAX_JSON_BYTES = 512 * 1024 * 1024
FILES = {
    "original": ("model.safetensors.index.json", "config.json", "quantize_config.json", 16),
    "k8-compact": ("model.k8.safetensors.index.json", "config.k8.json", "quantize_config.k8.json", 8),
    "k6-compact": ("model.k6.safetensors.index.json", "config.k6.json", "quantize_config.k6.json", 6),
}


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _text(value):
    return isinstance(value, str) and bool(value) and not any(c in value for c in "\0\n\r")


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _fields(value, expected):
    _require(isinstance(value, dict) and value.keys() == set(expected), f"expected fields {sorted(expected)}")


def shared_inventory_digest(weight_map, core_tensors):
    """Hash sorted non-core tensor-name/NUL/shard/newline entries."""
    _require(isinstance(weight_map, dict), "weight_map must be an object")
    core = set(core_tensors)
    _require(core <= weight_map.keys(), "core tensor missing from selected index")
    h = hashlib.sha256()
    for name, shard in sorted(weight_map.items()):
        _require(_text(name) and _text(shard) and shard.endswith(".safetensors")
                 and not shard.startswith(".") and not any(c in shard for c in "/\\"), "invalid tensor or shard name")
        if name not in core:
            h.update(f"{name}\0{shard}\n".encode())
    return h.hexdigest()


def view_fingerprint(name, shared_digest, metadata_sha256):
    h = hashlib.sha256(f"cuteafd-mc-view-v1\0{name}\0{shared_digest}\0".encode())
    for filename, digest in sorted(metadata_sha256.items()):
        h.update(f"{filename}\0{digest}\n".encode())
    return h.hexdigest()


def _read(root, name):
    path = Path(root) / name
    _require(path.stat().st_size <= MAX_JSON_BYTES, f"{name} exceeds metadata limit")
    with path.open("rb") as handle:
        data = handle.read(MAX_JSON_BYTES + 1)
    _require(len(data) <= MAX_JSON_BYTES, f"{name} exceeds metadata limit")
    return data


def validate(manifest, root=None, view=None):
    """Validate schema and optionally selected (or all) view metadata on disk."""
    _fields(manifest, ("schema", "source", "original_core_dtype", "shared_inventory_sha256", "variants"))
    _require(manifest["schema"] == SCHEMA, "unsupported MC schema")
    _fields(manifest["source"], ("repo", "revision"))
    _require(all(_text(v) for v in manifest["source"].values()), "MC requires pinned source")
    _require(manifest["original_core_dtype"] in ("BF16", "F32"), "FP8-origin checkpoints must not be MC")
    shared = manifest["shared_inventory_sha256"]
    _require(_digest(shared), "invalid shared inventory digest")
    variants = manifest["variants"]
    _require(isinstance(variants, list) and 2 <= len(variants) <= 3, "MC requires original and compact variants")
    names = set()
    for variant in variants:
        _fields(variant, ("name", "index", "config", "quantize", "core_bits", "core_tensors", "metadata_sha256", "fingerprint"))
        name = variant["name"]
        _require(name in FILES and name not in names, "unknown or duplicate MC variant")
        names.add(name)
        index, config, quantize, bits = FILES[name]
        _require((variant["index"], variant["config"], variant["quantize"]) == (index, config, quantize), "noncanonical MC filenames")
        _require(type(variant["core_bits"]) is int and variant["core_bits"] == bits, "MC core_bits mismatch")
        core = variant["core_tensors"]
        _require(isinstance(core, list) and bool(core) and all(_text(t) for t in core)
                 and core == sorted(set(core)), "core_tensors must be sorted, nonempty and unique")
        if name == "original":
            _require(not any(t.rsplit(".", 1)[-1] in ("trellis", "suh", "svh", "mcg", "mul1", "su", "sv") for t in core),
                     "original core contains a quantized tensor")
        hashes = variant["metadata_sha256"]
        _fields(hashes, (index, config, quantize))
        _require(all(_digest(d) for d in hashes.values()), "invalid metadata digest")
        _require(_digest(variant["fingerprint"]) and variant["fingerprint"] == view_fingerprint(name, shared, hashes), "MC fingerprint mismatch")
        if root is not None and (view is None or name == ("original" if view == "fp8-compact" else view)):
            contents = {filename: _read(root, filename) for filename in hashes}
            _require(all(hashlib.sha256(contents[n]).hexdigest() == d for n, d in hashes.items()), "MC metadata digest mismatch")
            weights = json.loads(contents[index])["weight_map"]
            _require(shared_inventory_digest(weights, core) == shared, "MC shared inventory digest mismatch")
    _require("original" in names, "MC requires original view")
    if view is not None:
        _require(("original" if view == "fp8-compact" else view) in names, f"CORE_VIEW={view} is absent")
    return manifest


def build(root, *, source_repo, source_revision, core_tensors, original_core_dtype="BF16"):
    """Build discovery metadata from sealed view files and their core name lists.

    Does not write or publish anything. `core_tensors` maps variant names to all
    active core tensor names, including trellis rotations and markers.
    """
    variants = []
    shared = None
    for name, core in sorted(core_tensors.items()):
        _require(name in FILES, "unknown MC variant")
        index, config, quantize, bits = FILES[name]
        core = sorted(core)
        weights = json.loads(_read(root, index))["weight_map"]
        digest = shared_inventory_digest(weights, core)
        _require(shared is None or shared == digest, "MC views disagree on shared inventory")
        shared = digest
        hashes = {filename: hashlib.sha256(_read(root, filename)).hexdigest() for filename in (index, config, quantize)}
        variants.append(dict(name=name, index=index, config=config, quantize=quantize, core_bits=bits,
                             core_tensors=core, metadata_sha256=hashes,
                             fingerprint=view_fingerprint(name, digest, hashes)))
    manifest = dict(schema=SCHEMA, source=dict(repo=source_repo, revision=source_revision),
                    original_core_dtype=original_core_dtype, shared_inventory_sha256=shared, variants=variants)
    validate(manifest, root)
    _require(len(json.dumps(manifest).encode()) <= MAX_JSON_BYTES, "MC manifest exceeds metadata limit")
    return manifest

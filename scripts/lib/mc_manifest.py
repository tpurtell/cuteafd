"""Versioned MC discovery metadata shared by publication and loader fixtures.

No tensor payload/header reads: only the selected three JSON files are needed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

SCHEMA = "cuteafd-mc-v1"
MANIFEST = "cuteafd-mc.json"
SUPPORTED_FEATURES: frozenset[str] = frozenset()
MAX_JSON_BYTES = 512 * 1024 * 1024
ENGRAM_LAYOUT = "fp8_e4m3_f32scale_row_interleaved_v1"
ENGRAM_ROW_BYTES = 100
ENGRAM_MAX_PART_BYTES = 45_000_000_000
ENGRAM_RECIPE = ("vendor BF16 per-row E4M3 round trip, BF16 scale held in F32, "
                 "decode = BF16(F32 code*scale)")
ENGRAM_TENSOR = re.compile(r"model\.layers\.(0|[1-9][0-9]{0,3})\.engram_layer\.multi_head_embedding\.embedding\.weight")
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


def _fields(value, expected, *, optional=(), strict=True):
    expected = set(expected)
    _require(isinstance(value, dict) and expected <= value.keys()
             and (not strict or value.keys() <= expected.union(optional)),
             f"expected fields {sorted(expected)}" + (f" with optional fields {sorted(optional)}" if optional else ""))


def engram_layer(tensor):
    match = ENGRAM_TENSOR.fullmatch(tensor) if isinstance(tensor, str) else None
    _require(match is not None, "invalid Engram BF16 tensor name")
    return int(match[1])


def engram_part_names(tensor, layer, number):
    return (tensor.removesuffix(".weight") + f".strided_{number:05d}",
            f"engram-fp8-layer-{layer:02d}-part-{number:05d}.safetensors")


def validate_engram_fp8(engram, *, strict=False):
    """Validate the optional alternate without opening tensor files."""
    _fields(engram, ("layout", "row_bytes", "code_dtype", "scale_dtype", "recipe", "layers"),
            strict=strict)
    _require(engram["layout"] == ENGRAM_LAYOUT
             and type(engram["row_bytes"]) is int and engram["row_bytes"] == ENGRAM_ROW_BYTES
             and engram["code_dtype"] == "F8_E4M3" and engram["scale_dtype"] == "F32",
             "invalid FP8 Engram layout")
    recipe = engram["recipe"]
    _require(isinstance(recipe, str)
             and re.fullmatch(re.escape(ENGRAM_RECIPE) + r"; reference=[0-9a-f]{40}", recipe),
             "invalid FP8 Engram recipe or pinned reference")
    layers = engram["layers"]
    _require(isinstance(layers, list) and bool(layers), "FP8 Engram requires layers")
    previous = -1
    for layer in layers:
        _fields(layer, ("tensor", "sha256", "parts"), strict=strict)
        number = engram_layer(layer["tensor"])
        _require(number > previous and _digest(layer["sha256"]),
                 "FP8 Engram layers must be ordered and carry BF16 tensor SHA256")
        previous = number
        parts = layer["parts"]
        _require(isinstance(parts, list) and 0 < len(parts) <= 100_000,
                 "FP8 Engram requires ordered row parts")
        for i, part in enumerate(parts):
            _fields(part, ("tensor", "file", "rows", "sha256"), strict=strict)
            _require((part["tensor"], part["file"]) == engram_part_names(layer["tensor"], number, i),
                     "invalid FP8 Engram part tensor or filename order")
            _require(type(part["rows"]) is int and 0 < part["rows"] * ENGRAM_ROW_BYTES < ENGRAM_MAX_PART_BYTES,
                     "invalid FP8 Engram part extent")
            _require(_digest(part["sha256"]), "invalid FP8 Engram part SHA256")
    return engram


def _check_engram_index(engram, weights, core):
    if engram is None:
        return
    shards = set(weights.values())
    core = set(core)
    for layer in engram["layers"]:
        name = layer["tensor"]
        filename = f"engram-bf16-layer-{engram_layer(name):02d}.safetensors"
        _require(weights.get(name) == filename and name not in core,
                 "BF16 Engram table missing from shared index")
        _require(all(part["tensor"] not in weights and part["file"] not in shards
                     for part in layer["parts"]), "FP8 Engram part leaked into standard index")


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


QMLA_SCHEMA = "cuteafd-qmla-v3"
QMLA_TENSORS_FILE = "mla_calibration.safetensors"
QMLA_STATUSES = frozenset({"calibrated_unqualified", "runtime_qualified"})
QMLA_PROTECTION_POLICIES = frozenset({"sink4_local128", "none"})
QMLA_PROTECTION_SCOPE = "full_history_mla_only"
QMLA_TRANSFORM_KINDS = frozenset({"identity", "scale_only", "full"})
QMLA_MODE_RE = re.compile(r"^(0|2|4|8):(0|4|8)$")


def _hex40(value):
    return isinstance(value, str) and len(value) == 40 and all(c in "0123456789abcdef" for c in value)


def validate_qmla_advisory(qmla, *, strict=False):
    """Validate optional embedded QMLA manifest section in cuteafd-mc.json (never in required_features)."""
    _fields(
        qmla,
        (
            "schema",
            "artifact_status",
            "protection_policy",
            "protection_scope",
            "source",
            "applicable_views",
            "conventions",
            "quantizer_recipe",
            "geometry",
            "profiles",
            "supported_modes",
            "unsupported_modes_reason",
            "calibration_provenance",
            "tensors_file",
            "tensors_payload_bytes",
            "tensors_payload_sha256",
            "tensors_bytes",
            "tensors_sha256",
            "transform_fingerprint",
        ),
        strict=strict,
    )
    _require(qmla["schema"] == QMLA_SCHEMA, "unsupported MC QMLA schema")
    _require(qmla["artifact_status"] in QMLA_STATUSES, "invalid MC QMLA artifact_status")
    policy = qmla["protection_policy"]
    _require(policy in QMLA_PROTECTION_POLICIES, "invalid MC QMLA protection_policy")
    _require(qmla["protection_scope"] == QMLA_PROTECTION_SCOPE, "invalid MC QMLA protection_scope")
    _require(qmla["tensors_file"] == QMLA_TENSORS_FILE, "noncanonical MC QMLA tensors_file")
    _require(
        type(qmla["tensors_payload_bytes"]) is int and qmla["tensors_payload_bytes"] >= 0
        and type(qmla["tensors_bytes"]) is int and qmla["tensors_bytes"] > 0
        and qmla["tensors_bytes"] >= qmla["tensors_payload_bytes"],
        "invalid MC QMLA tensor byte sizes",
    )
    _require(
        all(_digest(qmla[k]) for k in ("tensors_payload_sha256", "tensors_sha256", "transform_fingerprint")),
        "invalid MC QMLA digest",
    )
    source = qmla["source"]
    _require(isinstance(source, dict) and all(_text(source.get(k)) for k in (
        "repo", "revision", "family", "model_type", "config_sha256", "checkpoint_kind"
    )), "invalid MC QMLA source metadata")
    views = qmla["applicable_views"]
    _require(isinstance(views, list) and bool(views) and all(_text(v) for v in views),
             "MC QMLA requires applicable_views")
    _require(isinstance(qmla["conventions"], dict)
             and isinstance(qmla["quantizer_recipe"], dict)
             and isinstance(qmla["unsupported_modes_reason"], dict),
             "invalid MC QMLA metadata sections")
    _require(qmla["conventions"].get("content_basis_space") == "post_norm"
             and qmla["conventions"].get("content_forward_formula") == "c @ R_C @ diag(s_C)"
             and qmla["conventions"].get("content_inverse_formula") == "diag(1/s_C) @ R_C^T",
             "unsupported MC QMLA content basis or compensation")
    geoms = qmla["geometry"]
    _require(isinstance(geoms, list) and bool(geoms) and all(isinstance(g, dict) for g in geoms),
             "MC QMLA requires non-empty geometry")
    profiles = qmla["profiles"]
    _fields(profiles, ("content", "rope"), strict=strict)
    _require(all(profiles[path] is None or isinstance(profiles[path], dict) for path in ("content", "rope"))
             and (profiles["content"] is not None or profiles["rope"] is not None),
             "MC QMLA requires one shipped content or RoPE profile")
    for section, widths in (("content", {2, 4, 8}), ("rope", {4, 8})):
        prof = profiles[section]
        if prof is None:
            continue
        bits = prof.get("bits")
        width_map = prof.get("widths")
        _require(prof.get("protection_calibrated") == policy
                 and _digest(prof.get("profile_fingerprint"))
                 and isinstance(bits, list) and bool(bits)
                 and all(type(b) is int and b in widths for b in bits)
                 and isinstance(width_map, dict)
                 and set(width_map.keys()) == {str(b) for b in bits},
                 f"invalid MC QMLA {section} profile")
        for bkey, wentry in width_map.items():
            _require(isinstance(wentry, dict)
                     and _digest(wentry.get("width_fingerprint"))
                     and isinstance(wentry.get("kind_counts"), dict)
                     and isinstance(wentry.get("per_layer"), dict)
                     and bool(wentry["per_layer"])
                     and all(isinstance(row, dict) and row.get("kind") in QMLA_TRANSFORM_KINDS
                             and isinstance(row.get("tensors"), dict)
                             for row in wentry["per_layer"].values()),
                     f"invalid MC QMLA {section} width {bkey} selection")
    modes = qmla["supported_modes"]
    _require(isinstance(modes, list) and bool(modes)
             and all(isinstance(m, str) and QMLA_MODE_RE.fullmatch(m) and m != "0:0" for m in modes),
             "invalid MC QMLA supported_modes")
    prov = qmla["calibration_provenance"]
    _require(isinstance(prov, dict)
             and _text(prov.get("dataset_repo"))
             and _hex40(prov.get("dataset_revision"))
             and _digest(prov.get("input_ids_sha256")),
             "invalid MC QMLA calibration_provenance")
    comps = prov.get("components")
    _require(isinstance(comps, dict)
             and {"agentic", "exl3_standard"} <= set(comps.keys()) <= {"agentic", "exl3_standard", "long_context"},
             "MC QMLA calibration_provenance.components must use {agentic, exl3_standard} (+ optional long_context)")
    for row in comps.values():
        _require(isinstance(row, dict)
                 and type(row.get("sequences")) is int and row["sequences"] > 0
                 and type(row.get("tokens")) is int and row["tokens"] > 0
                 and _digest(row.get("input_ids_sha256")),
                 "invalid MC QMLA corpus component entry")
    return qmla


def _check_qmla_index(qmla, weights):
    if qmla is None:
        return
    shards = set(weights.values())
    _require(QMLA_TENSORS_FILE not in shards, "QMLA sidecar safetensors leaked into standard view index")


def validate(manifest, root=None, view=None):
    """Validate schema and optionally selected (or all) view metadata on disk."""
    _fields(manifest, ("schema", "source", "original_core_dtype", "shared_inventory_sha256", "variants"),
            optional=("required_features", "engram_fp8", "exl3_codebook", "qmla"))
    if "exl3_codebook" in manifest:
        declaration = manifest["exl3_codebook"]
        _fields(declaration, ("name", "multiplier"))
        _require(declaration in ({"name": "mcg", "multiplier": 0xCBAC1FED},
                                 {"name": "mul1", "multiplier": 0x83DCD12D}),
                 "invalid MC EXL3 codebook declaration")
    engram = validate_engram_fp8(manifest["engram_fp8"]) if "engram_fp8" in manifest else None
    qmla = validate_qmla_advisory(manifest["qmla"], strict=True) if "qmla" in manifest else None
    _require(manifest["schema"] == SCHEMA, "unsupported MC schema")
    features = manifest.get("required_features", [])
    _require(isinstance(features, list) and all(isinstance(f, str) for f in features),
             "MC required_features must be a list of strings")
    for feature in features:
        _require(feature in SUPPORTED_FEATURES, f"unsupported MC feature {feature!r}")
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
            _check_engram_index(engram, weights, core)
            _check_qmla_index(qmla, weights)
    _require("original" in names, "MC requires original view")
    if view is not None:
        _require(("original" if view == "fp8-compact" else view) in names, f"CORE_VIEW={view} is absent")
    return manifest


def build(root, *, source_repo, source_revision, core_tensors, original_core_dtype="BF16",
          required_features=None, engram_fp8=None, exl3_codebook=None, qmla=None):
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
    if required_features is not None:
        manifest["required_features"] = required_features
    if exl3_codebook is not None:
        manifest["exl3_codebook"] = exl3_codebook
    if engram_fp8 is not None:
        manifest["engram_fp8"] = engram_fp8
    if qmla is not None:
        manifest["qmla"] = qmla
    validate(manifest, root)
    _require(len(json.dumps(manifest).encode()) <= MAX_JSON_BYTES, "MC manifest exceeds metadata limit")
    return manifest

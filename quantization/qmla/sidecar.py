"""Versioned cuteafd-qmla-v3 single-file sidecar writer, fail-closed validator, and loader resolution.

There is NO separate ``mla_calibration.json`` file. The ONLY sidecar file added at a
checkpoint root is ``mla_calibration.safetensors`` (F32 tensors + self-describing
``__metadata__["cuteafd_qmla"]`` manifest):
- Per-width selections: each ``(layer, path, serving_width)`` records an explicit kind
  (``identity`` | ``scale_only`` | ``full``), its exact tensor mapping in the safetensors
  shard, and training-width provenance.
- Deduplication / aliasing: within each ``(path, layer)``, identical non-identity F32
  transforms selected by multiple widths share one canonical tensor slot (``t0``, ``t1``, ...)
  by exact byte equality. ``identity`` emits zero tensors; ``scale_only`` emits only ``.s``;
  ``full`` emits ``.r``+``.s`` (content) or ``.theta``+``.s`` (RoPE).
- On MC checkpoints: the full ``cuteafd-qmla-v3`` manifest lives as the optional advisory
  ``"qmla"`` section inside ``cuteafd-mc.json`` (never in ``required_features``), pinning
  whole-file ``tensors_sha256`` and ``tensors_bytes`` and cross-checked for exact equality
  with ``__metadata__["cuteafd_qmla"]``.
- On non-MC checkpoints: ``mla_calibration.safetensors`` is self-describing via
  ``__metadata__["cuteafd_qmla"]`` with zero extra JSON files.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
from typing import Any, Mapping, Sequence

import torch

import panel_corpus as pc

from .capture import (
    DeviceSplit, split_residency_budget,
    CapturedLayerBank,
    MlaLayerGeometry,
    _canonical_json,
    _fsync_dir,
    check_split_description,
    eval_positional_qk,
)
from .fit import (
    DEFAULT_PROTECTION_POLICY,
    ROTATION_IDENTITY_TOL,
    SCALE_IDENTITY_TOL,
    VALID_PROTECTION_POLICIES,
    VALID_TRANSFORM_KINDS,
    ContentFitResult,
    RopeFitResult,
    _check_content_equivalence,
    _content_native_output_error,
    apply_rope_pair_transform,
    apply_rope_scale_only,
    check_content_identity,
    classify_content_kind,
    classify_rope_kind,
)
from .measurement import SELECTION_OBJECTIVE, parse_candidate_id
from .quantizer import CONTENT_NONZERO_BITS, EPS_SCALE, GROUP_SIZE, ROPE_NONZERO_BITS, parse_qmla_mode

QMLA_SCHEMA = "cuteafd-qmla-v3"
PROFILE_PATHS = ("content", "rope")
TENSOR_ALIASING_POLICY = "exact_per_layer_f32_dedup"
TENSORS_FILE = "mla_calibration.safetensors"
QMLA_METADATA_KEY = "cuteafd_qmla"
VALID_ARTIFACT_STATUSES = frozenset({"calibrated_unqualified", "runtime_qualified"})
PROTECTION_SCOPE = "full_history_mla_only"

QMLA_BASE_COMPONENTS = pc.BASE_COMPONENTS
QMLA_OPTIONAL_COMPONENT = "long_context"
QMLA_KNOWN_COMPONENTS = pc.COMPONENT_ORDER

CANONICAL_CONVENTIONS = {
    "vector_layout": "row",
    "content_basis_space": "post_norm",
    "content_forward_formula": "c @ R_C @ diag(s_C)",
    "content_inverse_formula": "diag(1/s_C) @ R_C^T",
    "rope_angle_sign": "right_row_phi_plus_theta",
    "rope_forward_key_formula": "k_rope @ R_P(theta) @ diag(s_P_pair)",
    "rope_query_compensation_formula": "q_rope @ R_P(theta) @ diag(1/s_P_pair)",
    "storage_dtype": "F32",
}

CANONICAL_KIND_TOLERANCES = {
    "rotation_identity_tol": ROTATION_IDENTITY_TOL,
    "scale_identity_tol": SCALE_IDENTITY_TOL,
}

CANONICAL_QUANTIZER_RECIPE = {
    "scheme": "affine_asymmetric_per_token",
    "group_size": GROUP_SIZE,
    "rounding": "rne",
    "content_scale_dtype": "BF16",
    "content_zp_dtype": "U8",
    "rope_scale_dtype": "F32",
    "rope_zp_dtype": "F32",
    "eps_scale": EPS_SCALE,
    "pack_order": "lsb_first",
}

_EMBEDDED_REQUIRED_FIELDS = frozenset({
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
    "transform_fingerprint",
})
_MC_EXTRA_FIELDS = frozenset({"tensors_bytes", "tensors_sha256"})
_CONTENT_TENSOR_RE = re.compile(r"^content\.layer_([0-9]+)\.t([0-9]+)\.(r|s)$")
_ROPE_TENSOR_RE = re.compile(r"^rope\.layer_([0-9]+)\.t([0-9]+)\.(theta|s)$")


@dataclass(frozen=True)
class ResolvedContentTransform:
    """Loader-resolved content transform for one ``(layer_id, serving_width)``."""

    layer_id: int
    bits: int
    kind: str  # "identity" | "scale_only" | "full"
    r_c: torch.Tensor | None  # F32[d_c, d_c] iff kind == "full"
    s_c: torch.Tensor | None  # F32[d_c] iff kind in ("scale_only", "full")
    source_candidate: str
    source_bits: int | None
    tensor_names: dict[str, str]


@dataclass(frozen=True)
class ResolvedRopeTransform:
    """Loader-resolved RoPE transform for one ``(layer_id, serving_width)``."""

    layer_id: int
    bits: int
    kind: str  # "identity" | "scale_only" | "full"
    theta: torch.Tensor | None  # F32[d_r/2] iff kind == "full"
    s_p: torch.Tensor | None  # F32[d_r/2] iff kind in ("scale_only", "full")
    source_candidate: str
    source_bits: int | None
    tensor_names: dict[str, str]


@dataclass(frozen=True)
class ResolvedModeSelection:
    """Complete active selection resolved by a loader for one serving mode ``c:r``."""

    mode: str
    content_bits: int
    rope_bits: int
    protection_policy: str
    transform_fingerprint: str
    mode_selection_fingerprint: str
    content: dict[int, ResolvedContentTransform]
    rope: dict[int, ResolvedRopeTransform]


def _is_hex(value: Any, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resolve_qmla_panel(
    *,
    panel_revision: str | None = None,
    components: Sequence[str] | None = None,
    full: bool = True,
) -> tuple[str, str, tuple[str, ...]]:
    """Resolve ``(dataset_repo, dataset_revision, components)`` via ``panel_corpus.resolve_panel_policy``."""
    repo, rev, ordered = pc.resolve_panel_policy(panel_revision=panel_revision, components=components, full=full)
    if not set(QMLA_BASE_COMPONENTS) <= set(ordered):
        raise ValueError(f"QMLA calibration requires base components {QMLA_BASE_COMPONENTS}; got {ordered}")
    return repo, rev, ordered


def _validate_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    prov = deepcopy(dict(provenance))
    repo = prov.get("dataset_repo")
    rev = prov.get("dataset_revision")
    if not isinstance(repo, str) or not repo or "/" not in repo:
        raise ValueError(f"calibration_provenance.dataset_repo invalid: {repo!r}")
    if not _is_hex(rev, 40):
        raise ValueError(f"calibration_provenance.dataset_revision must be a 40-char SHA, got {rev!r}")

    components = prov.get("components")
    if not isinstance(components, Mapping):
        raise ValueError(
            "calibration_provenance.components must be a dict over known components "
            f"{QMLA_KNOWN_COMPONENTS}"
        )
    keys = set(components.keys())
    unknown = keys - set(QMLA_KNOWN_COMPONENTS)
    if unknown:
        raise ValueError(
            f"calibration_provenance.components has unknown entries {sorted(unknown)}; "
            f"allowed known components are {QMLA_KNOWN_COMPONENTS}"
        )
    if not set(QMLA_BASE_COMPONENTS) <= keys:
        raise ValueError(
            f"calibration_provenance.components must include base components {QMLA_BASE_COMPONENTS}; "
            f"got {sorted(keys)}"
        )
    validated_components: dict[str, dict[str, Any]] = {}
    for comp_name in sorted(keys):
        row = components[comp_name]
        if not isinstance(row, Mapping):
            raise ValueError(f"corpus component {comp_name!r} entry must be an object")
        seqs = row.get("sequences")
        toks = row.get("tokens")
        sha = row.get("input_ids_sha256")
        if type(seqs) is not int or seqs <= 0 or type(toks) is not int or toks <= 0:
            raise ValueError(f"corpus component {comp_name!r} requires positive integer sequences and tokens")
        if not _is_hex(sha, 64):
            raise ValueError(f"corpus component {comp_name!r} has invalid input_ids_sha256")
        validated_components[comp_name] = {
            "sequences": seqs,
            "tokens": toks,
            "input_ids_sha256": sha,
        }
    prov["components"] = validated_components

    splits = check_split_description(prov.get("splits"))
    for name in ("train", "val", "test"):
        key = f"{name}_sequences"
        val = prov.get(key)
        if type(val) is not int or val <= 0:
            raise ValueError(f"calibration_provenance.{key} must be a positive integer")
        if val != splits[name]["count"]:
            raise ValueError(f"calibration_provenance.{key} disagrees with its split description")
    if not _is_hex(prov.get("input_ids_sha256"), 64):
        raise ValueError("calibration_provenance.input_ids_sha256 must be a 64-char hex digest")
    for key in ("optimizer", "code_commit"):
        if not isinstance(prov.get(key), str) or not prov[key]:
            raise ValueError(f"calibration_provenance.{key} must be a non-empty string")
    if type(prov.get("steps_per_path")) is not int or prov["steps_per_path"] < 0:
        raise ValueError("calibration_provenance.steps_per_path must be a non-negative integer")
    if type(prov.get("seed")) is not int:
        raise ValueError("calibration_provenance.seed must be an integer")
    return prov


def _resolve_sidecar_path(path_or_dir: Path | str) -> Path:
    p = Path(path_or_dir)
    if p.is_dir():
        return p / TENSORS_FILE
    return p


def _prepare_f32_tensors(
    tensors: Mapping[str, torch.Tensor],
) -> tuple[dict[str, Any], list[tuple[str, bytes]], dict[str, str], str, int]:
    """Serialize sorted F32 tensor payloads and compute payload SHA256 before building header."""
    header: dict[str, Any] = {}
    ordered: list[tuple[str, bytes]] = []
    tensor_digests: dict[str, str] = {}
    payload_hash = hashlib.sha256()
    offset = 0
    for name in sorted(tensors):
        tensor = tensors[name].detach().contiguous().cpu().to(torch.float32)
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"non-finite values in sidecar tensor {name}")
        raw = tensor.numpy().tobytes()
        end = offset + len(raw)
        header[name] = {"dtype": "F32", "shape": list(tensor.shape), "data_offsets": [offset, end]}
        tensor_digests[name] = _sha256_bytes(raw)
        payload_hash.update(raw)
        ordered.append((name, raw))
        offset = end
    return header, ordered, tensor_digests, payload_hash.hexdigest(), offset


def _write_f32_safetensors_with_metadata(
    path: Path,
    header_tensors: dict[str, Any],
    ordered_payloads: Sequence[tuple[str, bytes]],
    embedded_manifest: Mapping[str, Any],
) -> tuple[str, int]:
    """Write ``mla_calibration.safetensors`` with embedded ``__metadata__["cuteafd_qmla"]``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    if tmp.is_symlink() or path.is_symlink():
        raise ValueError(f"refuse symlink sidecar path: {path}")

    header = dict(header_tensors)
    header["__metadata__"] = {QMLA_METADATA_KEY: _canonical_json(embedded_manifest)}
    encoded = _canonical_json(header).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    prefix = struct.pack("<Q", len(encoded)) + encoded

    file_hash = hashlib.sha256()
    payload_len = 0
    with tmp.open("wb") as handle:
        handle.write(prefix)
        file_hash.update(prefix)
        for _, raw in ordered_payloads:
            handle.write(raw)
            file_hash.update(raw)
            payload_len += len(raw)
        handle.flush()
        os.fsync(handle.fileno())
    total_bytes = len(prefix) + payload_len
    if tmp.stat().st_size != total_bytes:
        raise ValueError("sidecar safetensors size mismatch after write")
    os.replace(tmp, path)
    _fsync_dir(path.parent)
    return file_hash.hexdigest(), total_bytes


def _read_f32_safetensors_with_metadata(
    path: Path,
) -> tuple[str, int, str, int, dict[str, Any], dict[str, torch.Tensor], dict[str, str]]:
    """Read ``mla_calibration.safetensors`` returning whole-file + payload digests, embedded manifest, tensors."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"sidecar tensors missing or symlink: {path}")
    raw = path.read_bytes()
    file_sha256 = _sha256_bytes(raw)
    file_bytes = len(raw)
    if file_bytes < 8:
        raise ValueError("truncated sidecar safetensors file")
    header_len = struct.unpack("<Q", raw[:8])[0]
    if 8 + header_len > file_bytes:
        raise ValueError("corrupt sidecar safetensors header")
    header = json.loads(raw[8 : 8 + header_len].decode("utf-8"))
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict) or QMLA_METADATA_KEY not in metadata:
        raise ValueError(f"sidecar safetensors missing __metadata__[{QMLA_METADATA_KEY!r}]")
    embedded_manifest = json.loads(metadata[QMLA_METADATA_KEY])

    base = 8 + header_len
    payload_bytes = file_bytes - base
    payload_sha256 = _sha256_bytes(raw[base:])

    tensors: dict[str, torch.Tensor] = {}
    tensor_digests: dict[str, str] = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        if spec["dtype"] != "F32":
            raise ValueError(f"sidecar tensor {name} must be F32, found {spec['dtype']}")
        start, end = spec["data_offsets"]
        if start < 0 or end < start or end > payload_bytes:
            raise ValueError(f"sidecar tensor {name} has out-of-bounds data_offsets {[start, end]}")
        span = raw[base + start : base + end]
        tensor_digests[name] = _sha256_bytes(span)
        buf = bytearray(span)
        tensors[name] = torch.frombuffer(buf, dtype=torch.float32).reshape(spec["shape"]).clone()
    return file_sha256, file_bytes, payload_sha256, payload_bytes, embedded_manifest, tensors, tensor_digests


def _width_fingerprint(
    path: str,
    bits: int,
    protection_calibrated: str,
    layers: Sequence[int],
    per_layer: Mapping[str, Mapping[str, Any]],
    tensor_digests: Mapping[str, str],
) -> str:
    """Bind one ``(path, serving_width)`` selection: kinds, tensor bindings, and referenced tensor bytes."""
    used_tensors: dict[str, str] = {}
    layer_spec: dict[str, Any] = {}
    for lid in layers:
        row = per_layer[str(lid)]
        tmap = dict(row["tensors"])
        for tname in tmap.values():
            used_tensors[tname] = tensor_digests[tname]
        layer_spec[str(lid)] = {
            "kind": row["kind"],
            "tensors": tmap,
            "source_candidate": row["source_candidate"],
            "source_bits": row["source_bits"],
        }
    payload = {
        "schema": QMLA_SCHEMA,
        "path": path,
        "bits": int(bits),
        "protection_calibrated": protection_calibrated,
        "layers": list(layers),
        "per_layer": layer_spec,
        "tensors": {k: used_tensors[k] for k in sorted(used_tensors)},
    }
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _profile_fingerprint(
    path: str,
    bits: Sequence[int],
    protection_calibrated: str,
    layers: Sequence[int],
    width_fingerprints: Mapping[str, str],
) -> str:
    """Bind one path profile across all its served widths."""
    payload = {
        "schema": QMLA_SCHEMA,
        "path": path,
        "bits": list(bits),
        "protection_calibrated": protection_calibrated,
        "layers": list(layers),
        "kind_tolerances": CANONICAL_KIND_TOLERANCES,
        "tensor_aliasing": TENSOR_ALIASING_POLICY,
        "widths": {str(b): width_fingerprints[str(b)] for b in bits},
    }
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def compute_transform_fingerprint(manifest: Mapping[str, Any]) -> str:
    """Whole-sidecar SHA256 binding geometry, conventions, profiles, and raw tensor payload."""
    payload = {
        "schema": manifest["schema"],
        "protection_policy": manifest["protection_policy"],
        "protection_scope": manifest["protection_scope"],
        "source_revision": manifest["source"]["revision"],
        "base_revision": manifest["source"].get("base_revision"),
        "geometry": manifest["geometry"],
        "conventions": manifest["conventions"],
        "quantizer_recipe": manifest["quantizer_recipe"],
        "profiles": {
            path: (None if manifest["profiles"][path] is None else manifest["profiles"][path]["profile_fingerprint"])
            for path in PROFILE_PATHS
        },
        "tensors_payload_sha256": manifest["tensors_payload_sha256"],
    }
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def compute_mode_selection_fingerprint(manifest: Mapping[str, Any], mode: tuple[int, int] | str) -> str:
    """Active-mode cache identity digest binding the exact ``(c_bits, r_bits)`` per-layer selections.

    Prefix cache namespaces and CUDA graph keys include this digest so two calibrations or
    two serving modes with different active layer kinds/tensors can never collide.
    """
    c_bits, r_bits = parse_qmla_mode(mode) if isinstance(mode, str) else (int(mode[0]), int(mode[1]))
    mode_str = f"{c_bits}:{r_bits}"
    if mode_str != "0:0" and mode_str not in manifest["supported_modes"]:
        raise ValueError(f"mode {mode_str!r} is not in supported_modes {manifest['supported_modes']}")
    c_prof = manifest["profiles"].get("content")
    r_prof = manifest["profiles"].get("rope")
    c_fp = c_prof["widths"][str(c_bits)]["width_fingerprint"] if (c_bits > 0 and c_prof is not None) else None
    r_fp = r_prof["widths"][str(r_bits)]["width_fingerprint"] if (r_bits > 0 and r_prof is not None) else None
    payload = {
        "schema": manifest["schema"],
        "mode": mode_str,
        "protection_policy": manifest["protection_policy"],
        "protection_scope": manifest["protection_scope"],
        "source_revision": manifest["source"]["revision"],
        "base_revision": manifest["source"].get("base_revision"),
        "geometry": manifest["geometry"],
        "conventions": manifest["conventions"],
        "quantizer_recipe": manifest["quantizer_recipe"],
        "content_width_fingerprint": c_fp,
        "rope_width_fingerprint": r_fp,
    }
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def derive_supported_modes(
    content_bits_available: Sequence[int],
    rope_bits_available: Sequence[int],
    geometries: Sequence[MlaLayerGeometry],
) -> tuple[list[str], dict[str, str]]:
    """Compute exact supported QUANT_MLA modes and explicit unsupported reasons."""
    c_set = sorted(set(content_bits_available))
    r_set = sorted(set(rope_bits_available))
    any_rope = any(g.has_rope for g in geometries)
    modes: list[str] = []
    reasons: dict[str, str] = {"*:2": "INT2 RoPE is not supported"}
    if not any_rope:
        reasons["*:4"] = "model MLA layers have no RoPE key cache (qk_rope_dim=0)"
        reasons["*:8"] = "model MLA layers have no RoPE key cache (qk_rope_dim=0)"
    for c in c_set:
        modes.append(f"{c}:0")
    if any_rope:
        for r in r_set:
            modes.append(f"0:{r}")
        for c in c_set:
            for r in r_set:
                modes.append(f"{c}:{r}")
    return modes, reasons


def _expected_tensor_keys_for_kind(path: str, kind: str) -> tuple[str, ...]:
    if kind == "identity":
        return ()
    if kind == "scale_only":
        return ("s",)
    if kind == "full":
        return ("r", "s") if path == "content" else ("theta", "s")
    raise ValueError(f"invalid transform kind: {kind!r}")


def _validate_tensor_name_slot(path: str, lid: int, kind: str, tmap: Mapping[str, Any]) -> None:
    expected_keys = set(_expected_tensor_keys_for_kind(path, kind))
    if not isinstance(tmap, Mapping) or set(tmap.keys()) != expected_keys:
        raise ValueError(
            f"{path} layer {lid} kind={kind!r} requires tensor keys {sorted(expected_keys)}, got {sorted(tmap)}"
        )
    if not expected_keys:
        return
    pattern = _CONTENT_TENSOR_RE if path == "content" else _ROPE_TENSOR_RE
    slots: set[int] = set()
    for role, name in tmap.items():
        if not isinstance(name, str):
            raise ValueError(f"{path} layer {lid} tensor name must be str")
        m = pattern.fullmatch(name)
        if m is None or int(m.group(1)) != lid or m.group(3) != role:
            raise ValueError(f"{path} layer {lid} tensor {role!r} has noncanonical name {name!r}")
        slots.add(int(m.group(2)))
    if len(slots) != 1:
        raise ValueError(f"{path} layer {lid} tensors {dict(tmap)} must share one slot index")


def _tensor_bytes_f32(t: torch.Tensor) -> bytes:
    arr = t.detach().contiguous().cpu().to(torch.float32)
    if not bool(torch.isfinite(arr).all()):
        raise ValueError("non-finite values in transform tensor")
    return arr.numpy().tobytes()


def write_sidecar(
    output_dir: Path | str,
    *,
    source: Mapping[str, Any],
    geometries: Sequence[MlaLayerGeometry],
    content_selections: Mapping[int, Mapping[int, ContentFitResult]] | None,
    rope_selections: Mapping[int, Mapping[int, RopeFitResult]] | None,
    selection_meta: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]] | None],
    provenance: Mapping[str, Any],
    applicable_views: Sequence[str] | None = None,
    artifact_status: str = "calibrated_unqualified",
    protection_policy: str = DEFAULT_PROTECTION_POLICY,
    extra_unsupported_reasons: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Write single-file ``mla_calibration.safetensors`` (cuteafd-qmla-v3) and validate it.

    Parameters
    ----------
    content_selections : ``{serving_bits: {layer_id: ContentFitResult}}`` (or None/empty)
    rope_selections : ``{serving_bits: {layer_id: RopeFitResult}}`` (or None/empty)
    selection_meta : ``{"content": {str(bits): {str(lid): {candidate, source_bits, val_score}}}, "rope": ...}``
    """
    if artifact_status not in VALID_ARTIFACT_STATUSES:
        raise ValueError(f"invalid artifact_status: {artifact_status!r}")
    if protection_policy not in VALID_PROTECTION_POLICIES:
        raise ValueError(f"invalid protection_policy: {protection_policy!r}")
    if not geometries:
        raise ValueError("write_sidecar requires at least one MlaLayerGeometry")

    geom_by_layer: dict[int, MlaLayerGeometry] = {}
    for g in geometries:
        g.validate()
        if g.layer_id in geom_by_layer:
            raise ValueError(f"duplicate geometry for layer {g.layer_id}")
        geom_by_layer[g.layer_id] = g

    c_sel_in = {int(b): {int(lid): res for lid, res in layers.items()} for b, layers in (content_selections or {}).items()}
    r_sel_in = {int(b): {int(lid): res for lid, res in layers.items()} for b, layers in (rope_selections or {}).items()}
    if not c_sel_in and not r_sel_in:
        raise ValueError("write_sidecar requires at least one content or RoPE width selection")

    for path, sel_map, allowed in (
        ("content", c_sel_in, CONTENT_NONZERO_BITS),
        ("rope", r_sel_in, ROPE_NONZERO_BITS),
    ):
        for b in sel_map:
            if b not in allowed:
                raise ValueError(f"unsupported {path} serving width: {b}")
        if sel_map:
            layer_sets = [sorted(layers.keys()) for layers in sel_map.values()]
            if not layer_sets[0] or any(ls != layer_sets[0] for ls in layer_sets[1:]):
                raise ValueError(f"{path} widths must cover the exact same non-empty layer set")

    tensors: dict[str, torch.Tensor] = {}
    path_width_rows: dict[str, dict[str, dict[str, Any]]] = {"content": {}, "rope": {}}
    path_layers: dict[str, list[int]] = {"content": [], "rope": []}
    path_bits: dict[str, list[int]] = {"content": sorted(c_sel_in), "rope": sorted(r_sel_in)}
    path_equiv_max: dict[str, float] = {"content": 0.0, "rope": 0.0}
    content_orth_max = 0.0

    for path, sel_map in (("content", c_sel_in), ("rope", r_sel_in)):
        if not sel_map:
            continue
        bits_sorted = sorted(sel_map)
        layers_sorted = sorted(next(iter(sel_map.values())).keys())
        path_layers[path] = layers_sorted
        meta_path = selection_meta.get(path)
        if not isinstance(meta_path, Mapping):
            raise ValueError(f"selection_meta[{path!r}] is required")

        # Per-layer exact F32 deduplication table: (kind, bytes_key) -> tensor name map
        layer_slot_dedup: dict[int, dict[tuple[str, bytes], dict[str, str]]] = {lid: {} for lid in layers_sorted}
        layer_next_slot: dict[int, int] = {lid: 0 for lid in layers_sorted}

        for b in bits_sorted:
            bkey = str(b)
            meta_width = meta_path.get(bkey)
            if not isinstance(meta_width, Mapping):
                raise ValueError(f"selection_meta[{path!r}][{bkey!r}] is required")
            per_layer_out: dict[str, Any] = {}
            kind_counts = {"identity": 0, "scale_only": 0, "full": 0}
            for lid in layers_sorted:
                if lid not in geom_by_layer:
                    raise ValueError(f"{path} layer {lid} absent from geometries")
                g = geom_by_layer[lid]
                res = sel_map[b][lid]
                if res.protection_calibrated != protection_policy:
                    raise ValueError(
                        f"{path} width {b} layer {lid} protection_calibrated={res.protection_calibrated!r} "
                        f"disagrees with sidecar protection_policy={protection_policy!r}"
                    )
                lmeta = meta_width.get(str(lid))
                if not isinstance(lmeta, Mapping):
                    raise ValueError(f"selection_meta[{path!r}][{bkey!r}][{str(lid)!r}] missing")
                src_cand = str(lmeta["candidate"])
                parse_candidate_id(src_cand)
                src_bits = lmeta.get("source_bits")
                if src_bits is not None and type(src_bits) is not int:
                    raise ValueError(f"{path} width {b} layer {lid} source_bits must be int or None")
                val_score = lmeta.get("val_score")
                if not isinstance(val_score, Mapping) or not val_score:
                    raise ValueError(f"{path} width {b} layer {lid} needs non-empty val_score")
                for mode_k, v in val_score.items():
                    parse_qmla_mode(mode_k)
                    if not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0.0:
                        raise ValueError(f"{path} width {b} layer {lid} invalid val_score at {mode_k}")

                if path == "content":
                    assert isinstance(res, ContentFitResult)
                    if res.r_c.shape != (g.kv_lora_rank, g.kv_lora_rank) or res.s_c.shape != (g.kv_lora_rank,):
                        raise ValueError(f"content layer {lid} tensor shape mismatch")
                    kind = classify_content_kind(res.r_c, res.s_c)
                    if kind == "identity":
                        tmap: dict[str, str] = {}
                        equiv_err = 0.0
                        orth_err = 0.0
                    elif kind == "scale_only":
                        s_bytes = _tensor_bytes_f32(res.s_c)
                        dedup_key = ("scale_only", s_bytes)
                        if dedup_key in layer_slot_dedup[lid]:
                            tmap = dict(layer_slot_dedup[lid][dedup_key])
                        else:
                            slot = layer_next_slot[lid]
                            layer_next_slot[lid] += 1
                            tname_s = f"content.layer_{lid}.t{slot}.s"
                            tensors[tname_s] = res.s_c
                            tmap = {"s": tname_s}
                            layer_slot_dedup[lid][dedup_key] = dict(tmap)
                        equiv_err = float(res.equiv_max_rel_err)
                        orth_err = 0.0
                    else:
                        r_bytes = _tensor_bytes_f32(res.r_c)
                        s_bytes = _tensor_bytes_f32(res.s_c)
                        dedup_key = ("full", r_bytes + s_bytes)
                        if dedup_key in layer_slot_dedup[lid]:
                            tmap = dict(layer_slot_dedup[lid][dedup_key])
                        else:
                            slot = layer_next_slot[lid]
                            layer_next_slot[lid] += 1
                            tname_r = f"content.layer_{lid}.t{slot}.r"
                            tname_s = f"content.layer_{lid}.t{slot}.s"
                            tensors[tname_r] = res.r_c
                            tensors[tname_s] = res.s_c
                            tmap = {"r": tname_r, "s": tname_s}
                            layer_slot_dedup[lid][dedup_key] = dict(tmap)
                        equiv_err = float(res.equiv_max_rel_err)
                        orth_err = float(res.orth_max_err)
                        content_orth_max = max(content_orth_max, orth_err)
                    path_equiv_max["content"] = max(path_equiv_max["content"], equiv_err)
                    per_layer_out[str(lid)] = {
                        "kind": kind,
                        "tensors": tmap,
                        "source_candidate": src_cand,
                        "source_bits": src_bits,
                        "val_score": {k: float(v) for k, v in val_score.items()},
                        "equiv_max_rel_err": equiv_err,
                        "orth_max_err": orth_err,
                    }
                else:
                    assert isinstance(res, RopeFitResult)
                    if not g.has_rope:
                        raise ValueError(f"RoPE profile cannot include NoPE layer {lid}")
                    pairs = g.qk_rope_dim // 2
                    if res.theta.shape != (pairs,) or res.s_p.shape != (pairs,):
                        raise ValueError(f"RoPE layer {lid} tensor shape mismatch")
                    kind = classify_rope_kind(res.theta, res.s_p)
                    if kind == "identity":
                        tmap = {}
                        equiv_err = 0.0
                    elif kind == "scale_only":
                        s_bytes = _tensor_bytes_f32(res.s_p)
                        dedup_key = ("scale_only", s_bytes)
                        if dedup_key in layer_slot_dedup[lid]:
                            tmap = dict(layer_slot_dedup[lid][dedup_key])
                        else:
                            slot = layer_next_slot[lid]
                            layer_next_slot[lid] += 1
                            tname_s = f"rope.layer_{lid}.t{slot}.s"
                            tensors[tname_s] = res.s_p
                            tmap = {"s": tname_s}
                            layer_slot_dedup[lid][dedup_key] = dict(tmap)
                        equiv_err = float(res.equiv_max_rel_err)
                    else:
                        t_bytes = _tensor_bytes_f32(res.theta)
                        s_bytes = _tensor_bytes_f32(res.s_p)
                        dedup_key = ("full", t_bytes + s_bytes)
                        if dedup_key in layer_slot_dedup[lid]:
                            tmap = dict(layer_slot_dedup[lid][dedup_key])
                        else:
                            slot = layer_next_slot[lid]
                            layer_next_slot[lid] += 1
                            tname_t = f"rope.layer_{lid}.t{slot}.theta"
                            tname_s = f"rope.layer_{lid}.t{slot}.s"
                            tensors[tname_t] = res.theta
                            tensors[tname_s] = res.s_p
                            tmap = {"theta": tname_t, "s": tname_s}
                            layer_slot_dedup[lid][dedup_key] = dict(tmap)
                        equiv_err = float(res.equiv_max_rel_err)
                    path_equiv_max["rope"] = max(path_equiv_max["rope"], equiv_err)
                    per_layer_out[str(lid)] = {
                        "kind": kind,
                        "tensors": tmap,
                        "source_candidate": src_cand,
                        "source_bits": src_bits,
                        "val_score": {k: float(v) for k, v in val_score.items()},
                        "equiv_max_rel_err": equiv_err,
                    }
                kind_counts[kind] += 1
            path_width_rows[path][bkey] = {
                "kind_counts": kind_counts,
                "per_layer": per_layer_out,
            }

    header_tensors, ordered_payloads, tensor_digests, payload_sha256, payload_bytes = _prepare_f32_tensors(tensors)

    profiles: dict[str, Any] = {}
    for path in PROFILE_PATHS:
        bits_sorted = path_bits[path]
        if not bits_sorted:
            profiles[path] = None
            continue
        layers_sorted = path_layers[path]
        widths_out: dict[str, Any] = {}
        width_fps: dict[str, str] = {}
        for b in bits_sorted:
            bkey = str(b)
            wdata = path_width_rows[path][bkey]
            wfp = _width_fingerprint(path, b, protection_policy, layers_sorted, wdata["per_layer"], tensor_digests)
            width_fps[bkey] = wfp
            widths_out[bkey] = {
                "width_fingerprint": wfp,
                "kind_counts": wdata["kind_counts"],
                "per_layer": wdata["per_layer"],
            }
        validation: dict[str, Any] = {"equiv_max_rel_err": float(path_equiv_max[path])}
        if path == "content":
            validation["equivalence_metric"] = "expanded_kv_rel_l2"
            validation["orth_max_err"] = float(content_orth_max)
        profiles[path] = {
            "bits": bits_sorted,
            "layers": layers_sorted,
            "protection_calibrated": protection_policy,
            "selection_objective": SELECTION_OBJECTIVE,
            "kind_tolerances": dict(CANONICAL_KIND_TOLERANCES),
            "tensor_aliasing": TENSOR_ALIASING_POLICY,
            "widths": widths_out,
            "validation": validation,
            "profile_fingerprint": _profile_fingerprint(path, bits_sorted, protection_policy, layers_sorted, width_fps),
        }

    supported_modes, unsupported_reasons = derive_supported_modes(path_bits["content"], path_bits["rope"], geometries)
    if extra_unsupported_reasons:
        unsupported_reasons.update(extra_unsupported_reasons)

    prov = _validate_provenance(provenance)

    resolved_views = (
        list(applicable_views)
        if applicable_views is not None
        else (["all"] if source.get("checkpoint_kind") == "mc" else [str(source.get("calibrated_view", "original"))])
    )
    embedded_manifest: dict[str, Any] = {
        "schema": QMLA_SCHEMA,
        "artifact_status": artifact_status,
        "protection_policy": protection_policy,
        "protection_scope": PROTECTION_SCOPE,
        "source": dict(source),
        "applicable_views": resolved_views,
        "conventions": dict(CANONICAL_CONVENTIONS),
        "quantizer_recipe": dict(CANONICAL_QUANTIZER_RECIPE),
        "geometry": [
            dict(
                g.to_dict(),
                content_enabled=profiles["content"] is not None and g.layer_id in profiles["content"]["layers"],
                rope_enabled=profiles["rope"] is not None and g.layer_id in profiles["rope"]["layers"],
            )
            for g in sorted(geometries, key=lambda x: x.layer_id)
        ],
        "profiles": profiles,
        "supported_modes": supported_modes,
        "unsupported_modes_reason": unsupported_reasons,
        "calibration_provenance": prov,
        "tensors_file": TENSORS_FILE,
        "tensors_payload_bytes": payload_bytes,
        "tensors_payload_sha256": payload_sha256,
    }
    embedded_manifest["transform_fingerprint"] = compute_transform_fingerprint(embedded_manifest)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensors_path = output_dir / TENSORS_FILE
    file_sha256, file_bytes = _write_f32_safetensors_with_metadata(
        tensors_path, header_tensors, ordered_payloads, embedded_manifest
    )

    full_manifest = dict(embedded_manifest)
    full_manifest["tensors_bytes"] = file_bytes
    full_manifest["tensors_sha256"] = file_sha256

    validate_sidecar(tensors_path, mc_qmla=full_manifest, strict=True)
    return full_manifest


def read_sidecar_manifest(sidecar_path_or_dir: Path | str) -> dict[str, Any]:
    """Read embedded ``cuteafd_qmla`` manifest plus whole-file ``tensors_bytes``/``tensors_sha256``."""
    tensors_path = _resolve_sidecar_path(sidecar_path_or_dir)
    file_sha256, file_bytes, _, _, embedded_manifest, _, _ = _read_f32_safetensors_with_metadata(tensors_path)
    full = dict(embedded_manifest)
    full["tensors_bytes"] = file_bytes
    full["tensors_sha256"] = file_sha256
    return full


def validate_sidecar(
    sidecar_path_or_dir: Path | str,
    *,
    mc_manifest: Mapping[str, Any] | None = None,
    mc_qmla: Mapping[str, Any] | None = None,
    expected_source: Mapping[str, Any] | None = None,
    expected_geometries: Sequence[MlaLayerGeometry] | None = None,
    banks_by_layer: Mapping[int, CapturedLayerBank] | None = None,
    device: torch.device | str = "cpu",
    orth_tol: float = 1e-5,
    equiv_tol: float = 1e-4,
    strict: bool = True,
) -> dict[str, Any]:
    """Fail-closed validation of a ``cuteafd-qmla-v3`` sidecar (and optional ``cuteafd-mc.json["qmla"]``)."""
    tensors_path = _resolve_sidecar_path(sidecar_path_or_dir)
    (
        actual_file_sha256,
        actual_file_bytes,
        actual_payload_sha256,
        actual_payload_bytes,
        manifest,
        tensors,
        tensor_digests,
    ) = _read_f32_safetensors_with_metadata(tensors_path)

    if not isinstance(manifest, dict) or not _EMBEDDED_REQUIRED_FIELDS <= manifest.keys():
        missing = sorted(_EMBEDDED_REQUIRED_FIELDS - set(manifest)) if isinstance(manifest, dict) else ["<not-dict>"]
        raise ValueError(f"QMLA embedded manifest missing required fields: {missing}")
    if strict and manifest.keys() != _EMBEDDED_REQUIRED_FIELDS:
        raise ValueError(f"unexpected QMLA embedded manifest fields: {sorted(set(manifest) - _EMBEDDED_REQUIRED_FIELDS)}")
    if manifest["schema"] != QMLA_SCHEMA:
        raise ValueError(f"unsupported QMLA schema: {manifest['schema']!r} (expected {QMLA_SCHEMA!r})")
    if manifest["artifact_status"] not in VALID_ARTIFACT_STATUSES:
        raise ValueError(f"invalid artifact_status: {manifest['artifact_status']!r}")
    if manifest["protection_policy"] not in VALID_PROTECTION_POLICIES:
        raise ValueError(f"invalid protection_policy: {manifest['protection_policy']!r}")
    if manifest["protection_scope"] != PROTECTION_SCOPE:
        raise ValueError(f"invalid protection_scope: {manifest['protection_scope']!r}")
    if manifest["conventions"] != CANONICAL_CONVENTIONS:
        raise ValueError(f"QMLA manifest conventions differ from the {QMLA_SCHEMA} post_norm contract")
    if manifest["quantizer_recipe"] != CANONICAL_QUANTIZER_RECIPE:
        raise ValueError(f"QMLA manifest quantizer_recipe differs from canonical {QMLA_SCHEMA}")
    if manifest["tensors_file"] != TENSORS_FILE:
        raise ValueError(f"unexpected QMLA tensors_file: {manifest['tensors_file']!r}")
    if type(manifest["tensors_payload_bytes"]) is not int or manifest["tensors_payload_bytes"] != actual_payload_bytes:
        raise ValueError("QMLA tensors payload byte size mismatch")
    if manifest["tensors_payload_sha256"] != actual_payload_sha256:
        raise ValueError(
            f"QMLA tensors payload SHA256 mismatch: {actual_payload_sha256} != {manifest['tensors_payload_sha256']}"
        )

    source = manifest["source"]
    for key in ("repo", "revision", "family", "model_type", "config_sha256", "checkpoint_kind"):
        if not isinstance(source.get(key), str) or not source[key]:
            raise ValueError(f"QMLA manifest source missing {key}")
    if expected_source is not None:
        for key, expected_val in expected_source.items():
            if source.get(key) != expected_val:
                raise ValueError(f"QMLA source.{key} mismatch: {source.get(key)!r} != {expected_val!r}")

    _validate_provenance(manifest["calibration_provenance"])

    geoms = [MlaLayerGeometry.from_dict(g) for g in manifest["geometry"]]
    if not geoms:
        raise ValueError("QMLA manifest geometry list is empty")
    geom_by_layer = {g.layer_id: g for g in geoms}
    if len(geom_by_layer) != len(geoms):
        raise ValueError("duplicate layer_id in QMLA manifest geometry")
    if expected_geometries is not None:
        expected_map = {g.layer_id: g for g in expected_geometries}
        if geom_by_layer != expected_map:
            raise ValueError("QMLA manifest geometry disagrees with checkpoint geometry")

    profiles = manifest["profiles"]
    if not isinstance(profiles, Mapping) or set(profiles) != set(PROFILE_PATHS):
        raise ValueError(f"QMLA profiles must be keyed by path {PROFILE_PATHS}")
    content_prof, rope_prof = profiles["content"], profiles["rope"]
    if content_prof is None and rope_prof is None:
        raise ValueError("QMLA manifest defines no profiles")

    policy = manifest["protection_policy"]
    referenced_tensors: set[str] = set()
    max_orth_err = 0.0
    # Track unique non-identity transforms per (path, layer) for bank equivalence validation.
    unique_content_checks: dict[int, dict[tuple[str, str | None, str], tuple[str, torch.Tensor | None, torch.Tensor]]] = {}
    unique_rope_checks: dict[int, dict[tuple[str, str | None, str], tuple[str, torch.Tensor | None, torch.Tensor]]] = {}

    expected_profile_fields = {
        "bits",
        "layers",
        "protection_calibrated",
        "selection_objective",
        "kind_tolerances",
        "tensor_aliasing",
        "widths",
        "validation",
        "profile_fingerprint",
    }

    for path, prof in profiles.items():
        if prof is None:
            continue
        if not isinstance(prof, Mapping) or set(prof) != expected_profile_fields:
            raise ValueError(f"QMLA {path} profile fields must be {sorted(expected_profile_fields)}")
        allowed = CONTENT_NONZERO_BITS if path == "content" else ROPE_NONZERO_BITS
        bits = prof["bits"]
        if (
            not isinstance(bits, list)
            or not bits
            or bits != sorted(set(bits))
            or any(type(b) is not int or b not in allowed for b in bits)
        ):
            raise ValueError(f"invalid {path} profile served widths: {bits!r}")
        if prof["protection_calibrated"] != policy:
            raise ValueError(f"{path} profile protection_calibrated disagrees with protection_policy")
        if prof["selection_objective"] != SELECTION_OBJECTIVE:
            raise ValueError(f"{path} profile selection_objective must be {SELECTION_OBJECTIVE!r}")
        if prof["kind_tolerances"] != CANONICAL_KIND_TOLERANCES:
            raise ValueError(f"{path} profile kind_tolerances differ from canonical {CANONICAL_KIND_TOLERANCES}")
        if prof["tensor_aliasing"] != TENSOR_ALIASING_POLICY:
            raise ValueError(f"{path} profile tensor_aliasing must be {TENSOR_ALIASING_POLICY!r}")

        layers = [int(x) for x in prof["layers"]]
        if not layers or layers != sorted(set(layers)):
            raise ValueError(f"{path} profile layers must be sorted and unique")
        if any(lid not in geom_by_layer for lid in layers):
            raise ValueError(f"{path} profile references a layer absent from geometry")

        widths = prof["widths"]
        if not isinstance(widths, Mapping) or set(widths.keys()) != {str(b) for b in bits}:
            raise ValueError(f"{path} profile widths keys must match bits {[str(b) for b in bits]}")

        width_fps: dict[str, str] = {}
        rot_tol = float(prof["kind_tolerances"]["rotation_identity_tol"])
        scale_tol = float(prof["kind_tolerances"]["scale_identity_tol"])
        expected_row_fields = (
            {"kind", "tensors", "source_candidate", "source_bits", "val_score", "equiv_max_rel_err", "orth_max_err"}
            if path == "content"
            else {"kind", "tensors", "source_candidate", "source_bits", "val_score", "equiv_max_rel_err"}
        )

        for b in bits:
            bkey = str(b)
            wentry = widths[bkey]
            if not isinstance(wentry, Mapping) or set(wentry) != {"width_fingerprint", "kind_counts", "per_layer"}:
                raise ValueError(f"{path} width {b} entry has unexpected fields")
            per_layer = wentry["per_layer"]
            if not isinstance(per_layer, Mapping) or sorted(int(k) for k in per_layer) != layers:
                raise ValueError(f"{path} width {b} per_layer must cover exact layers {layers}")
            actual_counts = {"identity": 0, "scale_only": 0, "full": 0}
            for lid in layers:
                g = geom_by_layer[lid]
                if path == "rope" and not g.has_rope:
                    raise ValueError(f"RoPE profile references NoPE layer {lid}")
                row = per_layer[str(lid)]
                if not isinstance(row, Mapping) or set(row) != expected_row_fields:
                    raise ValueError(f"{path} width {b} layer {lid} has unexpected fields")
                kind = row["kind"]
                if kind not in VALID_TRANSFORM_KINDS:
                    raise ValueError(f"{path} width {b} layer {lid} invalid kind {kind!r}")
                actual_counts[kind] += 1
                parse_candidate_id(str(row["source_candidate"]))
                if row["source_bits"] is not None and type(row["source_bits"]) is not int:
                    raise ValueError(f"{path} width {b} layer {lid} invalid source_bits")
                if not isinstance(row["val_score"], Mapping) or not row["val_score"]:
                    raise ValueError(f"{path} width {b} layer {lid} requires non-empty val_score")
                for mk, mv in row["val_score"].items():
                    parse_qmla_mode(mk)
                    if not isinstance(mv, (int, float)) or not math.isfinite(mv) or mv < 0.0:
                        raise ValueError(f"{path} width {b} layer {lid} invalid val_score at {mk}")

                tmap = row["tensors"]
                _validate_tensor_name_slot(path, lid, kind, tmap)
                for tname in tmap.values():
                    if tname not in tensors:
                        raise ValueError(f"{path} width {b} layer {lid} references missing tensor {tname!r}")
                    referenced_tensors.add(tname)

                if kind == "identity":
                    continue

                s_vec = tensors[tmap["s"]]
                if not bool(torch.isfinite(s_vec).all()) or not bool((s_vec > 0.0).all()):
                    raise ValueError(f"{path} width {b} layer {lid} scale tensor must be finite and positive")

                if path == "content":
                    if s_vec.shape != (g.kv_lora_rank,):
                        raise ValueError(f"content width {b} layer {lid} scale shape mismatch")
                    if kind == "scale_only":
                        actual_kind = classify_content_kind(None, s_vec, rot_tol=rot_tol, scale_tol=scale_tol)
                        if actual_kind != "scale_only":
                            raise ValueError(
                                f"content width {b} layer {lid} declared scale_only but tensors classify as {actual_kind!r}"
                            )
                        unique_content_checks.setdefault(lid, {})[("scale_only", None, tmap["s"])] = (
                            "scale_only", None, s_vec
                        )
                    else:
                        r_mat = tensors[tmap["r"]]
                        if r_mat.shape != (g.kv_lora_rank, g.kv_lora_rank):
                            raise ValueError(f"content width {b} layer {lid} rotation shape mismatch")
                        if not bool(torch.isfinite(r_mat).all()):
                            raise ValueError(f"content width {b} layer {lid} rotation non-finite")
                        eye = torch.eye(g.kv_lora_rank, dtype=torch.float32)
                        orth = float((r_mat.transpose(0, 1) @ r_mat - eye).abs().max().item())
                        max_orth_err = max(max_orth_err, orth)
                        if orth > orth_tol:
                            raise ValueError(
                                f"content R_C orthogonality error {orth:.3e} > {orth_tol:.3e} at layer {lid}"
                            )
                        actual_kind = classify_content_kind(r_mat, s_vec, rot_tol=rot_tol, scale_tol=scale_tol)
                        if actual_kind != "full":
                            raise ValueError(
                                f"content width {b} layer {lid} declared full but tensors classify as {actual_kind!r}"
                            )
                        unique_content_checks.setdefault(lid, {})[("full", tmap["r"], tmap["s"])] = (
                            "full", r_mat, s_vec
                        )
                else:
                    pairs = g.qk_rope_dim // 2
                    if s_vec.shape != (pairs,):
                        raise ValueError(f"RoPE width {b} layer {lid} scale shape mismatch")
                    if kind == "scale_only":
                        actual_kind = classify_rope_kind(None, s_vec, rot_tol=rot_tol, scale_tol=scale_tol)
                        if actual_kind != "scale_only":
                            raise ValueError(
                                f"RoPE width {b} layer {lid} declared scale_only but tensors classify as {actual_kind!r}"
                            )
                        unique_rope_checks.setdefault(lid, {})[("scale_only", None, tmap["s"])] = (
                            "scale_only", None, s_vec
                        )
                    else:
                        theta = tensors[tmap["theta"]]
                        if theta.shape != (pairs,) or not bool(torch.isfinite(theta).all()):
                            raise ValueError(f"RoPE width {b} layer {lid} theta shape/finiteness mismatch")
                        actual_kind = classify_rope_kind(theta, s_vec, rot_tol=rot_tol, scale_tol=scale_tol)
                        if actual_kind != "full":
                            raise ValueError(
                                f"RoPE width {b} layer {lid} declared full but tensors classify as {actual_kind!r}"
                            )
                        unique_rope_checks.setdefault(lid, {})[("full", tmap["theta"], tmap["s"])] = (
                            "full", theta, s_vec
                        )

            if wentry["kind_counts"] != actual_counts:
                raise ValueError(f"{path} width {b} kind_counts {wentry['kind_counts']} != actual {actual_counts}")
            expected_wfp = _width_fingerprint(path, b, policy, layers, per_layer, tensor_digests)
            if wentry["width_fingerprint"] != expected_wfp:
                raise ValueError(f"{path} width {b} width_fingerprint mismatch")
            width_fps[bkey] = expected_wfp

        expected_pfp = _profile_fingerprint(path, bits, policy, layers, width_fps)
        if prof["profile_fingerprint"] != expected_pfp:
            raise ValueError(f"{path} profile fingerprint mismatch")

    if set(tensors) != referenced_tensors:
        raise ValueError(
            f"sidecar safetensors inventory mismatch: extra={sorted(set(tensors) - referenced_tensors)[:5]} "
            f"missing={sorted(referenced_tensors - set(tensors))[:5]}"
        )

    expected_tfp = compute_transform_fingerprint(manifest)
    if manifest["transform_fingerprint"] != expected_tfp:
        raise ValueError("QMLA transform_fingerprint mismatch")

    expected_modes, _ = derive_supported_modes(
        content_prof["bits"] if content_prof is not None else [],
        rope_prof["bits"] if rope_prof is not None else [],
        geoms,
    )
    for mode in manifest["supported_modes"]:
        parse_qmla_mode(mode)
    if sorted(manifest["supported_modes"]) != sorted(expected_modes):
        raise ValueError(f"supported_modes {manifest['supported_modes']} != expected {expected_modes}")

    if mc_manifest is not None:
        if "qmla" not in mc_manifest:
            raise ValueError("cuteafd-mc.json missing qmla section")
        mc_qmla = mc_manifest["qmla"]
    if mc_qmla is not None:
        if not isinstance(mc_qmla, Mapping):
            raise ValueError("cuteafd-mc.json['qmla'] must be an object")
        expected_mc_keys = _EMBEDDED_REQUIRED_FIELDS | _MC_EXTRA_FIELDS
        if strict and set(mc_qmla.keys()) != expected_mc_keys:
            raise ValueError(f"cuteafd-mc.json['qmla'] keys mismatch: {sorted(set(mc_qmla) ^ expected_mc_keys)}")
        if mc_qmla.get("tensors_sha256") != actual_file_sha256:
            raise ValueError(
                f"cuteafd-mc.json['qmla'].tensors_sha256 mismatch: {actual_file_sha256} != {mc_qmla.get('tensors_sha256')}"
            )
        if int(mc_qmla.get("tensors_bytes", -1)) != actual_file_bytes:
            raise ValueError("cuteafd-mc.json['qmla'].tensors_bytes mismatch")
        stripped = {k: v for k, v in mc_qmla.items() if k not in _MC_EXTRA_FIELDS}
        if stripped != manifest:
            raise ValueError("cuteafd-mc.json['qmla'] disagrees with mla_calibration.safetensors __metadata__")

    max_equiv_err = 0.0
    identity_layers: list[int] = []
    native_output_errors: list[float] = []
    if banks_by_layer is not None:
        with torch.no_grad():
            for lid, bank in banks_by_layer.items():
                g = geom_by_layer[lid]
                if content_prof is not None and lid in content_prof["layers"]:
                    params = bank.params.to(device)
                    val_seqs = DeviceSplit(bank.val, device, budget_bytes=split_residency_budget(device))
                    check_content_identity(val_seqs, params, g)
                    identity_layers.append(lid)
                    eye = torch.eye(g.kv_lora_rank, device=device, dtype=torch.float32)
                    for kind, r_mat, s_vec in unique_content_checks.get(lid, {}).values():
                        r_dev = eye if r_mat is None else r_mat.to(device)
                        s_dev = s_vec.to(device)
                        for seq in val_seqs:
                            err = _check_content_equivalence((seq,), params, g, r_dev, s_dev)
                            if not math.isfinite(err) or err >= equiv_tol:
                                raise ValueError(
                                    f"content quantization-off equivalence failed at layer {lid} ({kind}): "
                                    f"{err:.3e} (requires <{equiv_tol:.3e})"
                                )
                            max_equiv_err = max(max_equiv_err, err)
                            drift = _content_native_output_error((seq,), params, g, r_dev, s_dev)
                            if not math.isfinite(drift):
                                raise ValueError(f"non-finite quantization-off attention drift at layer {lid}")
                            native_output_errors.append(drift)
                if rope_prof is not None and lid in rope_prof["layers"]:
                    for kind, theta, s_vec in unique_rope_checks.get(lid, {}).values():
                        for seq in bank.val:
                            ref_z, mask = eval_positional_qk(seq, g)
                            if kind == "scale_only":
                                q_fn = lambda q: apply_rope_scale_only(q, s_vec, rope_kind=g.rope_kind, inverse_scale=True)
                                k_fn = lambda k: apply_rope_scale_only(k, s_vec, rope_kind=g.rope_kind, inverse_scale=False)
                            else:
                                assert theta is not None
                                q_fn = lambda q: apply_rope_pair_transform(
                                    q, theta, s_vec, rope_kind=g.rope_kind, inverse_scale=True
                                )
                                k_fn = lambda k: apply_rope_pair_transform(
                                    k, theta, s_vec, rope_kind=g.rope_kind, inverse_scale=False
                                )
                            pred_z, _ = eval_positional_qk(seq, g, rope_query_fn=q_fn, rope_key_fn=k_fn)
                            m = mask.unsqueeze(0)
                            denom = float(ref_z.masked_fill(~m, 0.0).norm().item()) or 1.0
                            err = float((pred_z - ref_z).masked_fill(~m, 0.0).norm().item() / denom)
                            max_equiv_err = max(max_equiv_err, err)
                            if err > equiv_tol:
                                raise ValueError(
                                    f"RoPE quantization-off equivalence failed at layer {lid} ({kind}): "
                                    f"{err:.3e} > {equiv_tol:.3e}"
                                )

    return {
        "status": "passed",
        "schema": QMLA_SCHEMA,
        "artifact_status": manifest["artifact_status"],
        "protection_policy": policy,
        "protection_scope": manifest["protection_scope"],
        "tensors_payload_sha256": actual_payload_sha256,
        "tensors_payload_bytes": actual_payload_bytes,
        "tensors_sha256": actual_file_sha256,
        "tensors_bytes": actual_file_bytes,
        "transform_fingerprint": expected_tfp,
        "supported_modes": list(manifest["supported_modes"]),
        "stored_tensors": len(tensors),
        "max_orth_err": max_orth_err,
        "max_equiv_err": max_equiv_err,
        "content_identity_verified_layers": sorted(identity_layers),
        "max_content_native_output_rel_err": max(native_output_errors, default=None),
    }


def load_sidecar_transforms(
    sidecar_path_or_dir: Path | str,
    *,
    expected_source: Mapping[str, Any] | None = None,
    expected_geometries: Sequence[MlaLayerGeometry] | None = None,
    strict: bool = True,
) -> tuple[
    dict[str, Any],
    dict[int, dict[int, ResolvedContentTransform]],
    dict[int, dict[int, ResolvedRopeTransform]],
]:
    """Validate a v3 sidecar and return ``(manifest, content_by_width, rope_by_width)``.

    ``content_by_width[bits][layer_id]`` is a ``ResolvedContentTransform`` and
    ``rope_by_width[bits][layer_id]`` is a ``ResolvedRopeTransform``.
    """
    tensors_path = _resolve_sidecar_path(sidecar_path_or_dir)
    validate_sidecar(
        tensors_path,
        expected_source=expected_source,
        expected_geometries=expected_geometries,
        strict=strict,
    )
    file_sha256, file_bytes, _, _, embedded_manifest, tensors, _ = _read_f32_safetensors_with_metadata(tensors_path)
    manifest = dict(embedded_manifest)
    manifest["tensors_bytes"] = file_bytes
    manifest["tensors_sha256"] = file_sha256

    content_by_width: dict[int, dict[int, ResolvedContentTransform]] = {}
    c_prof = manifest["profiles"].get("content")
    if c_prof is not None:
        for b in c_prof["bits"]:
            wmap: dict[int, ResolvedContentTransform] = {}
            per_layer = c_prof["widths"][str(b)]["per_layer"]
            for lid in c_prof["layers"]:
                row = per_layer[str(lid)]
                tmap = dict(row["tensors"])
                wmap[int(lid)] = ResolvedContentTransform(
                    layer_id=int(lid),
                    bits=int(b),
                    kind=str(row["kind"]),
                    r_c=tensors[tmap["r"]] if "r" in tmap else None,
                    s_c=tensors[tmap["s"]] if "s" in tmap else None,
                    source_candidate=str(row["source_candidate"]),
                    source_bits=row["source_bits"],
                    tensor_names=tmap,
                )
            content_by_width[int(b)] = wmap

    rope_by_width: dict[int, dict[int, ResolvedRopeTransform]] = {}
    r_prof = manifest["profiles"].get("rope")
    if r_prof is not None:
        for b in r_prof["bits"]:
            wmap_r: dict[int, ResolvedRopeTransform] = {}
            per_layer = r_prof["widths"][str(b)]["per_layer"]
            for lid in r_prof["layers"]:
                row = per_layer[str(lid)]
                tmap = dict(row["tensors"])
                wmap_r[int(lid)] = ResolvedRopeTransform(
                    layer_id=int(lid),
                    bits=int(b),
                    kind=str(row["kind"]),
                    theta=tensors[tmap["theta"]] if "theta" in tmap else None,
                    s_p=tensors[tmap["s"]] if "s" in tmap else None,
                    source_candidate=str(row["source_candidate"]),
                    source_bits=row["source_bits"],
                    tensor_names=tmap,
                )
            rope_by_width[int(b)] = wmap_r

    return manifest, content_by_width, rope_by_width


def resolve_mode_transforms(
    sidecar_path_or_dir: Path | str,
    mode: tuple[int, int] | str,
    *,
    expected_source: Mapping[str, Any] | None = None,
    expected_geometries: Sequence[MlaLayerGeometry] | None = None,
    strict: bool = True,
) -> ResolvedModeSelection:
    """Resolve one serving mode ``c:r`` from a v3 sidecar into its exact per-layer transforms.

    Fails closed if the mode is unsupported or if validation fails.
    """
    c_bits, r_bits = parse_qmla_mode(mode) if isinstance(mode, str) else (int(mode[0]), int(mode[1]))
    mode_str = f"{c_bits}:{r_bits}"
    manifest, content_by_width, rope_by_width = load_sidecar_transforms(
        sidecar_path_or_dir,
        expected_source=expected_source,
        expected_geometries=expected_geometries,
        strict=strict,
    )
    if mode_str != "0:0" and mode_str not in manifest["supported_modes"]:
        reason = manifest["unsupported_modes_reason"].get(f"*:{r_bits}") or manifest["unsupported_modes_reason"].get(mode_str)
        detail = f" ({reason})" if reason else ""
        raise ValueError(f"mode {mode_str!r} is not supported by this sidecar{detail}")
    mode_fp = compute_mode_selection_fingerprint(manifest, (c_bits, r_bits))
    return ResolvedModeSelection(
        mode=mode_str,
        content_bits=c_bits,
        rope_bits=r_bits,
        protection_policy=manifest["protection_policy"],
        transform_fingerprint=manifest["transform_fingerprint"],
        mode_selection_fingerprint=mode_fp,
        content=dict(content_by_width[c_bits]) if c_bits > 0 else {},
        rope=dict(rope_by_width[r_bits]) if r_bits > 0 else {},
    )


def build_mc_qmla_section(sidecar_path_or_dir: Path | str) -> dict[str, Any]:
    """Build the full embedded ``qmla`` section for ``cuteafd-mc.json`` (never in required_features)."""
    validate_sidecar(sidecar_path_or_dir, strict=True)
    return read_sidecar_manifest(sidecar_path_or_dir)


def attach_qmla_to_mc(mc_manifest: Mapping[str, Any], sidecar_path_or_dir: Path | str) -> dict[str, Any]:
    """Return a copy of ``cuteafd-mc.json`` with the full advisory ``qmla`` section attached."""
    updated = deepcopy(dict(mc_manifest))
    updated["qmla"] = build_mc_qmla_section(sidecar_path_or_dir)
    return updated

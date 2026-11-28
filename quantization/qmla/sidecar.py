"""Versioned cuteafd-qmla-v1 single-file sidecar writer, fail-closed validator, and MC integration.

There is NO separate ``mla_calibration.json`` file. The ONLY sidecar file added at a
checkpoint root is ``mla_calibration.safetensors`` (F32 per-path/per-width profiles):
- On MC checkpoints (have ``cuteafd-mc.json``): the full ``cuteafd-qmla-v1`` manifest lives
  as the optional advisory ``"qmla"`` section inside ``cuteafd-mc.json`` (never in
  ``required_features``), pinning whole-file ``tensors_sha256`` and ``tensors_bytes``.
- On non-MC checkpoints (official / plain without ``cuteafd-mc.json``): the exact same
  canonical manifest JSON lives in ``mla_calibration.safetensors``'s
  ``__metadata__["cuteafd_qmla"]`` header (written for BOTH MC and non-MC checkpoints so
  every sidecar file is self-describing and cross-checked).

Corpus provenance uses the pinned HF dataset ``wrldsuksgo2mars/agentic-quant-panel-v2``
(default revision ``54683731eb01c7c2fa5b801a0888e029a0e1b1c7``) over the fixed known
component set ``{"agentic", "exl3_standard", "long_context"}`` (no pluggable arbitrary
component list). Pilot and full runs calibrate immediately on ``agentic`` +
``exl3_standard``; a later recalibration adds ``long_context`` by passing a new
``--panel-revision`` and including ``long_context``.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence

import torch

from .capture import (
    CapturedLayerBank,
    MlaLayerGeometry,
    _canonical_json,
    _fsync_dir,
    eval_attention_output,
    eval_positional_qk,
)
from .fit import (
    DEFAULT_PROTECTION_POLICY,
    VALID_PROTECTION_POLICIES,
    ContentFitResult,
    RopeFitResult,
    apply_content_forward,
    apply_rope_pair_transform,
    compensate_kv_b,
)
from .quantizer import CONTENT_NONZERO_BITS, EPS_SCALE, GROUP_SIZE, ROPE_NONZERO_BITS, parse_qmla_mode

QMLA_SCHEMA = "cuteafd-qmla-v1"
TENSORS_FILE = "mla_calibration.safetensors"
QMLA_METADATA_KEY = "cuteafd_qmla"
VALID_ARTIFACT_STATUSES = frozenset({"calibrated_unqualified", "runtime_qualified"})
PROTECTION_SCOPE = "full_history_mla_only"

# Pinned panel contract (matches quantization/hy4_flash/prepare_corpus.py).
# Pilot and full runs default to agentic + exl3_standard at DEFAULT_QMLA_PANEL_REVISION;
# a later recalibration adds "long_context" by updating --panel-revision (no arbitrary component list).
QMLA_PANEL_REPO = "wrldsuksgo2mars/agentic-quant-panel-v2"
DEFAULT_QMLA_PANEL_REVISION = "54683731eb01c7c2fa5b801a0888e029a0e1b1c7"
QMLA_BASE_COMPONENTS = ("agentic", "exl3_standard")
QMLA_OPTIONAL_COMPONENT = "long_context"
QMLA_KNOWN_COMPONENTS = ("agentic", "exl3_standard", "long_context")

CANONICAL_CONVENTIONS = {
    "vector_layout": "row",
    "content_basis_space": "unit_rms",
    "content_forward_formula": "u @ R_C @ diag(s_C)",
    "content_inverse_formula": "diag(1/s_C) @ R_C^T @ diag(gamma)",
    "rope_angle_sign": "right_row_phi_plus_theta",
    "rope_forward_key_formula": "k_rope @ R_P(theta) @ diag(s_P_pair)",
    "rope_query_compensation_formula": "q_rope @ R_P(theta) @ diag(1/s_P_pair)",
    "storage_dtype": "F32",
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
    "view_transfer_validation",
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


def _is_hex(value: Any, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resolve_qmla_panel(
    *,
    panel_revision: str = DEFAULT_QMLA_PANEL_REVISION,
    components: Sequence[str] = QMLA_BASE_COMPONENTS,
) -> tuple[str, str, tuple[str, ...]]:
    """Validate and resolve the pinned QMLA panel revision and fixed known components.

    Always requires the two base components (``agentic``, ``exl3_standard``) and
    optionally allows ``long_context`` (never arbitrary unknown component names).
    Returns ``(dataset_repo, dataset_revision, canonical_components_tuple)``.
    """
    if not _is_hex(panel_revision, 40):
        raise ValueError(f"panel_revision must be a 40-char lowercase commit SHA, got {panel_revision!r}")
    seen: list[str] = []
    for name in components:
        if not isinstance(name, str) or name not in QMLA_KNOWN_COMPONENTS:
            raise ValueError(
                f"unknown QMLA corpus component {name!r}; allowed known components are {QMLA_KNOWN_COMPONENTS}"
            )
        if name not in seen:
            seen.append(name)
    if not set(QMLA_BASE_COMPONENTS) <= set(seen):
        raise ValueError(
            f"QMLA calibration requires base components {QMLA_BASE_COMPONENTS}; got {tuple(seen)}"
        )
    ordered = tuple(c for c in QMLA_KNOWN_COMPONENTS if c in seen)
    return QMLA_PANEL_REPO, panel_revision, ordered


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

    for key in ("train_sequences", "val_sequences"):
        val = prov.get(key)
        if type(val) is not int or val <= 0:
            raise ValueError(f"calibration_provenance.{key} must be a positive integer")
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


def _profile_fingerprint(
    kind: str,
    bits: int,
    transform_kind: str,
    protection_calibrated: str,
    layers: Sequence[int],
    tensor_digests: Mapping[str, str],
) -> str:
    payload = {
        "schema": QMLA_SCHEMA,
        "kind": kind,
        "bits": bits,
        "transform_kind": transform_kind,
        "protection_calibrated": protection_calibrated,
        "layers": list(layers),
        "tensors": {k: tensor_digests[k] for k in sorted(tensor_digests)},
    }
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def compute_transform_fingerprint(manifest: Mapping[str, Any]) -> str:
    """Deterministic SHA256 binding coordinate & protection identity across graphs and prefix caches."""
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
            "content": {
                k: v["profile_fingerprint"]
                for k, v in sorted(manifest["profiles"]["content"].items())
            },
            "rope": {
                k: v["profile_fingerprint"]
                for k, v in sorted(manifest["profiles"]["rope"].items())
            },
        },
        "tensors_payload_sha256": manifest["tensors_payload_sha256"],
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


def write_sidecar(
    output_dir: Path | str,
    *,
    source: Mapping[str, Any],
    geometries: Sequence[MlaLayerGeometry],
    content_profiles: Mapping[int, Sequence[ContentFitResult]],
    rope_profiles: Mapping[int, Sequence[RopeFitResult]],
    provenance: Mapping[str, Any],
    applicable_views: Sequence[str] = ("original",),
    view_transfer_validation: Mapping[str, Any] | None = None,
    artifact_status: str = "calibrated_unqualified",
    protection_policy: str = DEFAULT_PROTECTION_POLICY,
    extra_unsupported_reasons: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Write single-file ``mla_calibration.safetensors`` (with embedded metadata) and validate it.

    Returns the full MC-ready ``qmla`` manifest dict (embedded manifest plus whole-file
    ``tensors_bytes`` and ``tensors_sha256``). Never writes a separate ``mla_calibration.json``.
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

    if not content_profiles and not rope_profiles:
        raise ValueError("write_sidecar requires at least one content or RoPE profile")

    tensors: dict[str, torch.Tensor] = {}
    for bits in sorted(content_profiles):
        if bits not in CONTENT_NONZERO_BITS:
            raise ValueError(f"unsupported content profile bits: {bits}")
        results = sorted(content_profiles[bits], key=lambda r: r.layer_id)
        if not results:
            raise ValueError(f"empty content profile for c{bits}")
        prefix = f"content.c{bits}"
        for res in results:
            if res.protection_calibrated != protection_policy:
                raise ValueError(
                    f"content c{bits} layer {res.layer_id} protection_calibrated={res.protection_calibrated!r} "
                    f"disagrees with sidecar protection_policy={protection_policy!r}"
                )
            if res.layer_id not in geom_by_layer:
                raise ValueError(f"content c{bits} layer {res.layer_id} absent from geometries")
            g = geom_by_layer[res.layer_id]
            if res.r_c.shape != (g.kv_lora_rank, g.kv_lora_rank) or res.s_c.shape != (g.kv_lora_rank,):
                raise ValueError(f"content c{bits} layer {res.layer_id} tensor shape mismatch")
            tensors[f"{prefix}.layer_{res.layer_id}.r"] = res.r_c
            tensors[f"{prefix}.layer_{res.layer_id}.s"] = res.s_c

    for bits in sorted(rope_profiles):
        if bits not in ROPE_NONZERO_BITS:
            raise ValueError(f"unsupported RoPE profile bits: {bits}")
        results = sorted(rope_profiles[bits], key=lambda r: r.layer_id)
        if not results:
            raise ValueError(f"empty RoPE profile for r{bits}")
        prefix = f"rope.r{bits}"
        for res in results:
            if res.protection_calibrated != protection_policy:
                raise ValueError(
                    f"RoPE r{bits} layer {res.layer_id} protection_calibrated={res.protection_calibrated!r} "
                    f"disagrees with sidecar protection_policy={protection_policy!r}"
                )
            if res.layer_id not in geom_by_layer:
                raise ValueError(f"RoPE r{bits} layer {res.layer_id} absent from geometries")
            g = geom_by_layer[res.layer_id]
            if not g.has_rope:
                raise ValueError(f"RoPE r{bits} cannot include NoPE layer {res.layer_id}")
            pairs = g.qk_rope_dim // 2
            if res.theta.shape != (pairs,) or res.s_p.shape != (pairs,):
                raise ValueError(f"RoPE r{bits} layer {res.layer_id} tensor shape mismatch")
            tensors[f"{prefix}.layer_{res.layer_id}.theta"] = res.theta
            tensors[f"{prefix}.layer_{res.layer_id}.s"] = res.s_p

    header_tensors, ordered_payloads, tensor_digests, payload_sha256, payload_bytes = _prepare_f32_tensors(tensors)

    content_meta: dict[str, Any] = {}
    for bits in sorted(content_profiles):
        results = sorted(content_profiles[bits], key=lambda r: r.layer_id)
        prefix = f"content.c{bits}"
        layers = [r.layer_id for r in results]
        prof_digests = {k: v for k, v in tensor_digests.items() if k.startswith(prefix + ".")}
        content_meta[f"c{bits}"] = {
            "bits": bits,
            "transform_kind": "dense_orth",
            "layers": layers,
            "tensors_prefix": prefix,
            "protection_calibrated": protection_policy,
            "validation": {
                "val_loss_mean": float(sum(r.best_val_loss for r in results) / len(results)),
                "equiv_max_rel_err": float(max(r.equiv_max_rel_err for r in results)),
                "orth_max_err": float(max(r.orth_max_err for r in results)),
                "per_layer": {
                    str(r.layer_id): {
                        "best_step": r.best_step,
                        "best_val_loss": r.best_val_loss,
                        "init_alpha": r.init_alpha,
                        "equiv_max_rel_err": r.equiv_max_rel_err,
                        "orth_max_err": r.orth_max_err,
                        "baselines": r.baselines,
                    }
                    for r in results
                },
            },
            "profile_fingerprint": _profile_fingerprint(
                "content", bits, "dense_orth", protection_policy, layers, prof_digests
            ),
        }

    rope_meta: dict[str, Any] = {}
    for bits in sorted(rope_profiles):
        results = sorted(rope_profiles[bits], key=lambda r: r.layer_id)
        prefix = f"rope.r{bits}"
        layers = [r.layer_id for r in results]
        prof_digests = {k: v for k, v in tensor_digests.items() if k.startswith(prefix + ".")}
        rope_meta[f"r{bits}"] = {
            "bits": bits,
            "transform_kind": "pair_so2",
            "layers": layers,
            "tensors_prefix": prefix,
            "protection_calibrated": protection_policy,
            "validation": {
                "val_loss_mean": float(sum(r.best_val_loss for r in results) / len(results)),
                "equiv_max_rel_err": float(max(r.equiv_max_rel_err for r in results)),
                "per_layer": {
                    str(r.layer_id): {
                        "best_step": r.best_step,
                        "best_val_loss": r.best_val_loss,
                        "init_alpha": r.init_alpha,
                        "equiv_max_rel_err": r.equiv_max_rel_err,
                        "baselines": r.baselines,
                    }
                    for r in results
                },
            },
            "profile_fingerprint": _profile_fingerprint(
                "rope", bits, "pair_so2", protection_policy, layers, prof_digests
            ),
        }

    supported_modes, unsupported_reasons = derive_supported_modes(
        sorted(content_profiles), sorted(rope_profiles), geometries
    )
    if extra_unsupported_reasons:
        unsupported_reasons.update(extra_unsupported_reasons)

    prov = _validate_provenance(provenance)

    embedded_manifest: dict[str, Any] = {
        "schema": QMLA_SCHEMA,
        "artifact_status": artifact_status,
        "protection_policy": protection_policy,
        "protection_scope": PROTECTION_SCOPE,
        "source": dict(source),
        "applicable_views": list(applicable_views),
        "view_transfer_validation": dict(view_transfer_validation or {}),
        "conventions": dict(CANONICAL_CONVENTIONS),
        "quantizer_recipe": dict(CANONICAL_QUANTIZER_RECIPE),
        "geometry": [
            dict(
                g.to_dict(),
                content_enabled=any(g.layer_id in p["layers"] for p in content_meta.values()),
                rope_enabled=any(g.layer_id in p["layers"] for p in rope_meta.values()),
            )
            for g in sorted(geometries, key=lambda x: x.layer_id)
        ],
        "profiles": {"content": content_meta, "rope": rope_meta},
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
    orth_tol: float = 1e-5,
    equiv_tol: float = 1e-4,
    strict: bool = True,
) -> dict[str, Any]:
    """Fail-closed validation of ``mla_calibration.safetensors`` (and optional ``cuteafd-mc.json["qmla"]``)."""
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
        raise ValueError(f"unsupported QMLA schema: {manifest['schema']!r}")
    if manifest["artifact_status"] not in VALID_ARTIFACT_STATUSES:
        raise ValueError(f"invalid artifact_status: {manifest['artifact_status']!r}")
    if manifest["protection_policy"] not in VALID_PROTECTION_POLICIES:
        raise ValueError(f"invalid protection_policy: {manifest['protection_policy']!r}")
    if manifest["protection_scope"] != PROTECTION_SCOPE:
        raise ValueError(f"invalid protection_scope: {manifest['protection_scope']!r}")
    if manifest["conventions"] != CANONICAL_CONVENTIONS:
        raise ValueError("QMLA manifest conventions differ from canonical cuteafd-qmla-v1")
    if manifest["quantizer_recipe"] != CANONICAL_QUANTIZER_RECIPE:
        raise ValueError("QMLA manifest quantizer_recipe differs from canonical cuteafd-qmla-v1")
    if manifest["tensors_file"] != TENSORS_FILE:
        raise ValueError(f"unexpected QMLA tensors_file: {manifest['tensors_file']!r}")
    if int(manifest["tensors_payload_bytes"]) != actual_payload_bytes:
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

    # Validate geometries.
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

    expected_tensors: set[str] = set()
    content_profiles = manifest["profiles"]["content"]
    rope_profiles = manifest["profiles"]["rope"]
    if not content_profiles and not rope_profiles:
        raise ValueError("QMLA manifest defines no profiles")

    policy = manifest["protection_policy"]
    max_orth_err = 0.0
    for key, prof in content_profiles.items():
        bits = int(prof["bits"])
        if key != f"c{bits}" or bits not in CONTENT_NONZERO_BITS:
            raise ValueError(f"invalid content profile key/bits: {key}/{bits}")
        if prof.get("protection_calibrated") != policy:
            raise ValueError(f"content profile {key} protection_calibrated disagrees with protection_policy")
        prefix = f"content.c{bits}"
        if prof["tensors_prefix"] != prefix:
            raise ValueError(f"unexpected tensors_prefix for {key}")
        layers = [int(x) for x in prof["layers"]]
        if not layers or layers != sorted(set(layers)):
            raise ValueError(f"content profile {key} layers must be sorted and unique")
        prof_digests: dict[str, str] = {}
        for lid in layers:
            g = geom_by_layer[lid]
            r_name = f"{prefix}.layer_{lid}.r"
            s_name = f"{prefix}.layer_{lid}.s"
            expected_tensors.update((r_name, s_name))
            if r_name not in tensors or s_name not in tensors:
                raise ValueError(f"missing content tensors for layer {lid} profile {key}")
            r_mat, s_vec = tensors[r_name], tensors[s_name]
            if r_mat.shape != (g.kv_lora_rank, g.kv_lora_rank) or s_vec.shape != (g.kv_lora_rank,):
                raise ValueError(f"content tensor shape mismatch at layer {lid} profile {key}")
            if not bool(torch.isfinite(r_mat).all()) or not bool(torch.isfinite(s_vec).all()):
                raise ValueError(f"non-finite content transform at layer {lid} profile {key}")
            if not bool((s_vec > 0.0).all()):
                raise ValueError(f"non-positive content scale at layer {lid} profile {key}")
            eye = torch.eye(g.kv_lora_rank, dtype=torch.float32)
            orth = float((r_mat.transpose(0, 1) @ r_mat - eye).abs().max().item())
            max_orth_err = max(max_orth_err, orth)
            if orth > orth_tol:
                raise ValueError(f"content R_C orthogonality error {orth:.3e} > {orth_tol:.3e} at layer {lid} {key}")
            prof_digests[r_name] = tensor_digests[r_name]
            prof_digests[s_name] = tensor_digests[s_name]
        expected_fp = _profile_fingerprint("content", bits, prof["transform_kind"], policy, layers, prof_digests)
        if prof["profile_fingerprint"] != expected_fp:
            raise ValueError(f"content profile fingerprint mismatch for {key}")

    for key, prof in rope_profiles.items():
        bits = int(prof["bits"])
        if key != f"r{bits}" or bits not in ROPE_NONZERO_BITS:
            raise ValueError(f"invalid RoPE profile key/bits: {key}/{bits}")
        if prof.get("protection_calibrated") != policy:
            raise ValueError(f"RoPE profile {key} protection_calibrated disagrees with protection_policy")
        prefix = f"rope.r{bits}"
        if prof["tensors_prefix"] != prefix:
            raise ValueError(f"unexpected tensors_prefix for {key}")
        layers = [int(x) for x in prof["layers"]]
        if not layers or layers != sorted(set(layers)):
            raise ValueError(f"RoPE profile {key} layers must be sorted and unique")
        prof_digests = {}
        for lid in layers:
            g = geom_by_layer[lid]
            if not g.has_rope:
                raise ValueError(f"RoPE profile {key} references NoPE layer {lid}")
            pairs = g.qk_rope_dim // 2
            t_name = f"{prefix}.layer_{lid}.theta"
            s_name = f"{prefix}.layer_{lid}.s"
            expected_tensors.update((t_name, s_name))
            if t_name not in tensors or s_name not in tensors:
                raise ValueError(f"missing RoPE tensors for layer {lid} profile {key}")
            theta, s_vec = tensors[t_name], tensors[s_name]
            if theta.shape != (pairs,) or s_vec.shape != (pairs,):
                raise ValueError(f"RoPE tensor shape mismatch at layer {lid} profile {key}")
            if not bool(torch.isfinite(theta).all()) or not bool(torch.isfinite(s_vec).all()):
                raise ValueError(f"non-finite RoPE transform at layer {lid} profile {key}")
            if not bool((s_vec > 0.0).all()):
                raise ValueError(f"non-positive RoPE scale at layer {lid} profile {key}")
            prof_digests[t_name] = tensor_digests[t_name]
            prof_digests[s_name] = tensor_digests[s_name]
        expected_fp = _profile_fingerprint("rope", bits, prof["transform_kind"], policy, layers, prof_digests)
        if prof["profile_fingerprint"] != expected_fp:
            raise ValueError(f"RoPE profile fingerprint mismatch for {key}")

    if set(tensors) != expected_tensors:
        raise ValueError(
            f"sidecar safetensors inventory mismatch: extra={sorted(set(tensors) - expected_tensors)[:5]} "
            f"missing={sorted(expected_tensors - set(tensors))[:5]}"
        )

    expected_tfp = compute_transform_fingerprint(manifest)
    if manifest["transform_fingerprint"] != expected_tfp:
        raise ValueError("QMLA transform_fingerprint mismatch")

    expected_modes, _ = derive_supported_modes(
        [int(p["bits"]) for p in content_profiles.values()],
        [int(p["bits"]) for p in rope_profiles.values()],
        geoms,
    )
    for mode in manifest["supported_modes"]:
        parse_qmla_mode(mode)
    if sorted(manifest["supported_modes"]) != sorted(expected_modes):
        raise ValueError(f"supported_modes {manifest['supported_modes']} != expected {expected_modes}")

    # Cross-check cuteafd-mc.json["qmla"] when provided.
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

    # Optional quantization-off algebraic equivalence verification against captured banks.
    max_equiv_err = 0.0
    if banks_by_layer is not None:
        with torch.no_grad():
            for lid, bank in banks_by_layer.items():
                g = geom_by_layer[lid]
                for key, prof in content_profiles.items():
                    if lid not in prof["layers"]:
                        continue
                    r_mat = tensors[f"{prof['tensors_prefix']}.layer_{lid}.r"]
                    s_vec = tensors[f"{prof['tensors_prefix']}.layer_{lid}.s"]
                    w_tilde = compensate_kv_b(bank.params.kv_b_weight, bank.params.kv_norm_weight, r_mat, s_vec)
                    for seq in bank.val:
                        pred = eval_attention_output(
                            seq,
                            bank.params,
                            g,
                            content_fn=lambda u: apply_content_forward(u, r_mat, s_vec),
                            kv_b_override=w_tilde,
                        )
                        denom = float(seq.ref_output.float().norm().item()) or 1.0
                        err = float((pred - seq.ref_output.float()).norm().item() / denom)
                        max_equiv_err = max(max_equiv_err, err)
                        if err > equiv_tol:
                            raise ValueError(
                                f"content quantization-off equivalence failed at layer {lid} {key}: {err:.3e} > {equiv_tol:.3e}"
                            )
                for key, prof in rope_profiles.items():
                    if lid not in prof["layers"]:
                        continue
                    theta = tensors[f"{prof['tensors_prefix']}.layer_{lid}.theta"]
                    s_vec = tensors[f"{prof['tensors_prefix']}.layer_{lid}.s"]
                    for seq in bank.val:
                        ref_z, mask = eval_positional_qk(seq, g)
                        pred_z, _ = eval_positional_qk(
                            seq,
                            g,
                            rope_query_fn=lambda q: apply_rope_pair_transform(
                                q, theta, s_vec, rope_kind=g.rope_kind, inverse_scale=True
                            ),
                            rope_key_fn=lambda k: apply_rope_pair_transform(
                                k, theta, s_vec, rope_kind=g.rope_kind, inverse_scale=False
                            ),
                        )
                        m = mask.unsqueeze(0)
                        denom = float(ref_z.masked_fill(~m, 0.0).norm().item()) or 1.0
                        err = float((pred_z - ref_z).masked_fill(~m, 0.0).norm().item() / denom)
                        max_equiv_err = max(max_equiv_err, err)
                        if err > equiv_tol:
                            raise ValueError(
                                f"RoPE quantization-off equivalence failed at layer {lid} {key}: {err:.3e} > {equiv_tol:.3e}"
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
        "max_orth_err": max_orth_err,
        "max_equiv_err": max_equiv_err,
    }


def build_mc_qmla_section(sidecar_path_or_dir: Path | str) -> dict[str, Any]:
    """Build the full embedded ``qmla`` section for ``cuteafd-mc.json`` (never in required_features)."""
    validate_sidecar(sidecar_path_or_dir, strict=True)
    return read_sidecar_manifest(sidecar_path_or_dir)


def attach_qmla_to_mc(mc_manifest: Mapping[str, Any], sidecar_path_or_dir: Path | str) -> dict[str, Any]:
    """Return a copy of ``cuteafd-mc.json`` with the full advisory ``qmla`` section attached.

    Preserves ``variants``, ``shared_inventory_sha256``, ``required_features``,
    ``source``, ``original_core_dtype``, ``exl3_codebook``, and ``engram_fp8`` byte-for-byte.
    """
    updated = deepcopy(dict(mc_manifest))
    updated["qmla"] = build_mc_qmla_section(sidecar_path_or_dir)
    return updated

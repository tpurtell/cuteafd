"""Versioned cuteafd-qmla-v1 sidecar writer, fail-closed validator, and MC advisory integration.

Writes root ``mla_calibration.safetensors`` (F32 per-path/per-width profiles) and
``mla_calibration.json`` with deterministic ``transform_fingerprint``, pluggable
corpus component provenance, and explicit ``artifact_status`` distinguishing
``calibrated_unqualified`` artifacts from ``runtime_qualified`` ones.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence

import torch

from .capture import (
    CapturedLayerBank,
    MlaLayerGeometry,
    _atomic_json,
    _canonical_json,
    _fsync_dir,
    eval_attention_output,
    eval_positional_qk,
)
from .fit import (
    ContentFitResult,
    RopeFitResult,
    apply_content_forward,
    apply_rope_pair_transform,
    compensate_kv_b,
)
from .quantizer import CONTENT_NONZERO_BITS, EPS_SCALE, GROUP_SIZE, ROPE_NONZERO_BITS, parse_qmla_mode

QMLA_SCHEMA = "cuteafd-qmla-v1"
MANIFEST_FILE = "mla_calibration.json"
TENSORS_FILE = "mla_calibration.safetensors"
VALID_ARTIFACT_STATUSES = frozenset({"calibrated_unqualified", "runtime_qualified"})

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


@dataclass(frozen=True)
class CorpusComponentSpec:
    """Data-driven pinned corpus component (agentic, exl3_standard, or optional long-context)."""

    name: str
    repo: str
    revision: str
    sequences: int
    tokens: int
    input_ids_sha256: str
    renderer_template_sha256: str | None = None
    thinking_mode: str | None = None
    notes: str | None = None

    def validate(self) -> None:
        for field_name in ("name", "repo", "revision", "input_ids_sha256"):
            val = getattr(self, field_name)
            if not isinstance(val, str) or not val or any(c in val for c in "\0\n\r"):
                raise ValueError(f"invalid corpus component {field_name}: {val!r}")
        if self.sequences <= 0 or self.tokens <= 0:
            raise ValueError(f"corpus component {self.name!r} requires positive sequences and tokens")
        if len(self.input_ids_sha256) != 64 or not all(c in "0123456789abcdef" for c in self.input_ids_sha256):
            raise ValueError(f"corpus component {self.name!r} has invalid input_ids_sha256")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        row: dict[str, Any] = {
            "name": self.name,
            "repo": self.repo,
            "revision": self.revision,
            "sequences": self.sequences,
            "tokens": self.tokens,
            "input_ids_sha256": self.input_ids_sha256,
        }
        if self.renderer_template_sha256 is not None:
            row["renderer_template_sha256"] = self.renderer_template_sha256
        if self.thinking_mode is not None:
            row["thinking_mode"] = self.thinking_mode
        if self.notes is not None:
            row["notes"] = self.notes
        return row


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_f32_safetensors_sha256(path: Path, tensors: Mapping[str, torch.Tensor]) -> tuple[str, int, dict[str, str]]:
    """Write F32 safetensors sequentially while streaming file SHA256 and per-tensor SHA256s."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    if tmp.is_symlink() or path.is_symlink():
        raise ValueError(f"refuse symlink sidecar path: {path}")

    header: dict[str, Any] = {}
    ordered: list[tuple[str, bytes]] = []
    tensor_digests: dict[str, str] = {}
    offset = 0
    for name in sorted(tensors):
        tensor = tensors[name].detach().contiguous().cpu().to(torch.float32)
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"non-finite values in sidecar tensor {name}")
        raw = tensor.numpy().tobytes()
        end = offset + len(raw)
        header[name] = {"dtype": "F32", "shape": list(tensor.shape), "data_offsets": [offset, end]}
        tensor_digests[name] = _sha256_bytes(raw)
        ordered.append((name, raw))
        offset = end

    encoded = _canonical_json(header).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    prefix = struct.pack("<Q", len(encoded)) + encoded
    file_hash = hashlib.sha256()
    with tmp.open("wb") as handle:
        handle.write(prefix)
        file_hash.update(prefix)
        for _, raw in ordered:
            handle.write(raw)
            file_hash.update(raw)
        handle.flush()
        os.fsync(handle.fileno())
    total_bytes = len(prefix) + offset
    if tmp.stat().st_size != total_bytes:
        raise ValueError("sidecar safetensors size mismatch after write")
    os.replace(tmp, path)
    _fsync_dir(path.parent)
    return file_hash.hexdigest(), total_bytes, tensor_digests


def _read_f32_safetensors_sha256(path: Path) -> tuple[str, dict[str, torch.Tensor], dict[str, str]]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"sidecar tensors missing or symlink: {path}")
    raw = path.read_bytes()
    file_sha256 = _sha256_bytes(raw)
    if len(raw) < 8:
        raise ValueError("truncated sidecar safetensors file")
    header_len = struct.unpack("<Q", raw[:8])[0]
    if 8 + header_len > len(raw):
        raise ValueError("corrupt sidecar safetensors header")
    header = json.loads(raw[8 : 8 + header_len].decode("utf-8"))
    base = 8 + header_len
    tensors: dict[str, torch.Tensor] = {}
    tensor_digests: dict[str, str] = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        if spec["dtype"] != "F32":
            raise ValueError(f"sidecar tensor {name} must be F32, found {spec['dtype']}")
        start, end = spec["data_offsets"]
        span = raw[base + start : base + end]
        tensor_digests[name] = _sha256_bytes(span)
        buf = bytearray(span)
        tensors[name] = torch.frombuffer(buf, dtype=torch.float32).reshape(spec["shape"]).clone()
    return file_sha256, tensors, tensor_digests


def _profile_fingerprint(
    kind: str,
    bits: int,
    transform_kind: str,
    layers: Sequence[int],
    tensor_digests: Mapping[str, str],
) -> str:
    payload = {
        "schema": QMLA_SCHEMA,
        "kind": kind,
        "bits": bits,
        "transform_kind": transform_kind,
        "layers": list(layers),
        "tensors": {k: tensor_digests[k] for k in sorted(tensor_digests)},
    }
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def compute_transform_fingerprint(manifest: Mapping[str, Any]) -> str:
    """Deterministic SHA256 binding coordinate identity across graphs & prefix caches."""
    payload = {
        "schema": manifest["schema"],
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
        "tensors_sha256": manifest["tensors_sha256"],
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
    extra_unsupported_reasons: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Write ``mla_calibration.safetensors`` and ``mla_calibration.json`` and validate them."""
    if artifact_status not in VALID_ARTIFACT_STATUSES:
        raise ValueError(f"invalid artifact_status: {artifact_status!r}")
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
    content_meta: dict[str, Any] = {}
    rope_meta: dict[str, Any] = {}

    for bits in sorted(content_profiles):
        if bits not in CONTENT_NONZERO_BITS:
            raise ValueError(f"unsupported content profile bits: {bits}")
        results = sorted(content_profiles[bits], key=lambda r: r.layer_id)
        if not results:
            raise ValueError(f"empty content profile for c{bits}")
        prefix = f"content.c{bits}"
        for res in results:
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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensors_sha256, tensors_bytes, tensor_digests = _write_f32_safetensors_sha256(
        output_dir / TENSORS_FILE, tensors
    )

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
            "protection_calibrated": "none",
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
            "profile_fingerprint": _profile_fingerprint("content", bits, "dense_orth", layers, prof_digests),
        }

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
            "protection_calibrated": "none",
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
            "profile_fingerprint": _profile_fingerprint("rope", bits, "pair_so2", layers, prof_digests),
        }

    supported_modes, unsupported_reasons = derive_supported_modes(
        sorted(content_profiles), sorted(rope_profiles), geometries
    )
    if extra_unsupported_reasons:
        unsupported_reasons.update(extra_unsupported_reasons)

    # Validate data-driven provenance components (never hard-coded).
    prov = deepcopy(dict(provenance))
    components = prov.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("calibration_provenance.components must be a non-empty list of component dicts")
    for comp in components:
        if isinstance(comp, CorpusComponentSpec):
            comp.validate()
        elif isinstance(comp, Mapping):
            CorpusComponentSpec(
                name=comp["name"],
                repo=comp["repo"],
                revision=comp["revision"],
                sequences=int(comp["sequences"]),
                tokens=int(comp["tokens"]),
                input_ids_sha256=str(comp["input_ids_sha256"]),
                renderer_template_sha256=comp.get("renderer_template_sha256"),
                thinking_mode=comp.get("thinking_mode"),
                notes=comp.get("notes"),
            ).validate()
        else:
            raise ValueError("each calibration_provenance.components entry must be a dict/CorpusComponentSpec")
    prov["components"] = [
        c.to_dict() if isinstance(c, CorpusComponentSpec) else dict(c) for c in components
    ]

    manifest: dict[str, Any] = {
        "schema": QMLA_SCHEMA,
        "artifact_status": artifact_status,
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
        "tensors_bytes": tensors_bytes,
        "tensors_sha256": tensors_sha256,
    }
    manifest["transform_fingerprint"] = compute_transform_fingerprint(manifest)
    _atomic_json(output_dir / MANIFEST_FILE, manifest)
    validate_sidecar(output_dir, strict=True)
    return manifest


def validate_sidecar(
    root: Path | str,
    *,
    expected_source: Mapping[str, Any] | None = None,
    expected_geometries: Sequence[MlaLayerGeometry] | None = None,
    banks_by_layer: Mapping[int, CapturedLayerBank] | None = None,
    orth_tol: float = 1e-5,
    equiv_tol: float = 1e-4,
    strict: bool = True,
) -> dict[str, Any]:
    """Fail-closed validation of ``mla_calibration.json`` and ``mla_calibration.safetensors``."""
    root = Path(root)
    manifest_path = root / MANIFEST_FILE
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"missing QMLA sidecar manifest: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))

    required_top = {
        "schema",
        "artifact_status",
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
        "tensors_bytes",
        "tensors_sha256",
        "transform_fingerprint",
    }
    if not isinstance(manifest, dict) or not required_top <= manifest.keys():
        raise ValueError(f"QMLA manifest missing required fields: {sorted(required_top - set(manifest))}")
    if strict and manifest.keys() != required_top:
        raise ValueError(f"unexpected QMLA manifest fields: {sorted(set(manifest) - required_top)}")
    if manifest["schema"] != QMLA_SCHEMA:
        raise ValueError(f"unsupported QMLA schema: {manifest['schema']!r}")
    if manifest["artifact_status"] not in VALID_ARTIFACT_STATUSES:
        raise ValueError(f"invalid artifact_status: {manifest['artifact_status']!r}")
    if manifest["conventions"] != CANONICAL_CONVENTIONS:
        raise ValueError("QMLA manifest conventions differ from canonical cuteafd-qmla-v1")
    if manifest["quantizer_recipe"] != CANONICAL_QUANTIZER_RECIPE:
        raise ValueError("QMLA manifest quantizer_recipe differs from canonical cuteafd-qmla-v1")
    if manifest["tensors_file"] != TENSORS_FILE:
        raise ValueError(f"unexpected QMLA tensors_file: {manifest['tensors_file']!r}")

    source = manifest["source"]
    for key in ("repo", "revision", "family", "model_type", "config_sha256", "checkpoint_kind"):
        if not isinstance(source.get(key), str) or not source[key]:
            raise ValueError(f"QMLA manifest source missing {key}")
    if expected_source is not None:
        for key, expected_val in expected_source.items():
            if source.get(key) != expected_val:
                raise ValueError(f"QMLA source.{key} mismatch: {source.get(key)!r} != {expected_val!r}")

    # Validate corpus provenance components.
    components = manifest["calibration_provenance"].get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("QMLA calibration_provenance.components must be non-empty")
    for comp in components:
        CorpusComponentSpec(
            name=comp["name"],
            repo=comp["repo"],
            revision=comp["revision"],
            sequences=int(comp["sequences"]),
            tokens=int(comp["tokens"]),
            input_ids_sha256=str(comp["input_ids_sha256"]),
            renderer_template_sha256=comp.get("renderer_template_sha256"),
            thinking_mode=comp.get("thinking_mode"),
            notes=comp.get("notes"),
        ).validate()

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

    # Verify safetensors digest and tensors.
    tensors_path = root / TENSORS_FILE
    actual_sha256, tensors, tensor_digests = _read_f32_safetensors_sha256(tensors_path)
    if actual_sha256 != manifest["tensors_sha256"]:
        raise ValueError(f"QMLA tensors SHA256 mismatch: {actual_sha256} != {manifest['tensors_sha256']}")
    if tensors_path.stat().st_size != int(manifest["tensors_bytes"]):
        raise ValueError("QMLA tensors byte size mismatch")

    expected_tensors: set[str] = set()
    content_profiles = manifest["profiles"]["content"]
    rope_profiles = manifest["profiles"]["rope"]
    if not content_profiles and not rope_profiles:
        raise ValueError("QMLA manifest defines no profiles")

    max_orth_err = 0.0
    for key, prof in content_profiles.items():
        bits = int(prof["bits"])
        if key != f"c{bits}" or bits not in CONTENT_NONZERO_BITS:
            raise ValueError(f"invalid content profile key/bits: {key}/{bits}")
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
        expected_fp = _profile_fingerprint("content", bits, prof["transform_kind"], layers, prof_digests)
        if prof["profile_fingerprint"] != expected_fp:
            raise ValueError(f"content profile fingerprint mismatch for {key}")

    for key, prof in rope_profiles.items():
        bits = int(prof["bits"])
        if key != f"r{bits}" or bits not in ROPE_NONZERO_BITS:
            raise ValueError(f"invalid RoPE profile key/bits: {key}/{bits}")
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
        expected_fp = _profile_fingerprint("rope", bits, prof["transform_kind"], layers, prof_digests)
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

    # Validate supported_modes list consistency.
    expected_modes, _ = derive_supported_modes(
        [int(p["bits"]) for p in content_profiles.values()],
        [int(p["bits"]) for p in rope_profiles.values()],
        geoms,
    )
    for mode in manifest["supported_modes"]:
        parse_qmla_mode(mode)
    if sorted(manifest["supported_modes"]) != sorted(expected_modes):
        raise ValueError(f"supported_modes {manifest['supported_modes']} != expected {expected_modes}")

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
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "tensors_sha256": actual_sha256,
        "tensors_bytes": int(manifest["tensors_bytes"]),
        "transform_fingerprint": expected_tfp,
        "supported_modes": list(manifest["supported_modes"]),
        "max_orth_err": max_orth_err,
        "max_equiv_err": max_equiv_err,
    }


def build_mc_qmla_advisory(sidecar_dir: Path | str) -> dict[str, Any]:
    """Build the advisory ``qmla`` pointer object for ``cuteafd-mc.json`` (never in required_features)."""
    report = validate_sidecar(sidecar_dir, strict=True)
    return {
        "schema": QMLA_SCHEMA,
        "manifest": MANIFEST_FILE,
        "manifest_sha256": report["manifest_sha256"],
        "tensors": TENSORS_FILE,
        "tensors_sha256": report["tensors_sha256"],
        "transform_fingerprint": report["transform_fingerprint"],
        "artifact_status": report["artifact_status"],
    }


def attach_qmla_advisory_to_mc(mc_manifest: Mapping[str, Any], sidecar_dir: Path | str) -> dict[str, Any]:
    """Return a copy of ``cuteafd-mc.json`` with the advisory ``qmla`` section attached.

    Preserves ``variants``, ``shared_inventory_sha256``, ``required_features``,
    ``source``, ``original_core_dtype``, ``exl3_codebook``, and ``engram_fp8`` byte-for-byte.
    """
    updated = deepcopy(dict(mc_manifest))
    updated["qmla"] = build_mc_qmla_advisory(sidecar_dir)
    return updated

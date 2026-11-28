"""Mode-agnostic MLA capture interface, exact reference evaluators, and XXH3-128 storage.

Every MLA family adapter (starting with Hy4-flash in ``quantization/hy4_flash/``)
produces ``CapturedLayerBank`` instances through ``QmlaCaptureAdapter`` so all
downstream calibration, baseline evaluation, validation, and sidecar assembly
remain completely family-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
from typing import Any, Callable, Mapping, Sequence

import torch
import xxhash
import torch.nn.functional as F

REFERENCE = Path(__file__).resolve().parents[2] / "python" / "reference"
if str(REFERENCE) not in sys.path:
    sys.path.insert(0, str(REFERENCE))
from families.hy4_common import tiled_sink_attention
from shape_invariant import ROWS, linear

CAPTURE_SCHEMA = "cuteafd-qmla-capture-v2"
SPLIT_SALT = "cuteafd-qmla-split-v2"
SPLIT_NAMES = ("train", "val", "test")
VALID_ATTENTION_KINDS = frozenset({"dsa", "swa", "full"})
VALID_ROPE_KINDS = frozenset({"interleaved", "split_half", "unrotated_pe", "none"})

_ST_DTYPE_TO_TORCH = {
    "F32": torch.float32,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "I64": torch.int64,
    "I32": torch.int32,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}
_TORCH_TO_ST_DTYPE = {v: k for k, v in _ST_DTYPE_TO_TORCH.items()}


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.is_symlink():
        raise ValueError(f"refuse symlink temporary path: {tmp}")
    data = (_canonical_json(payload) + "\n").encode("utf-8")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def unit_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Weightless RMSNorm_1(x) in FP32: x / sqrt(mean(x^2) + eps)."""
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + float(eps))


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Weighted RMSNorm in FP32 matching reference models."""
    return unit_rms_norm(x, eps) * weight.to(device=x.device, dtype=torch.float32)


def causal_mask(
    seq_len: int,
    device: torch.device | str = "cpu",
    *,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """Exact [T, T] boolean causal mask (optionally restricted to sliding_window)."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    idx = torch.arange(seq_len, device=device)
    dist = idx[:, None] - idx[None, :]
    allowed = dist >= 0
    if sliding_window is not None:
        if sliding_window <= 0:
            raise ValueError("sliding_window must be positive")
        allowed &= dist < sliding_window
    return allowed


@dataclass(frozen=True)
class MlaLayerGeometry:
    """Per-layer MLA geometry descriptor shared across capture, fitting, and sidecars."""

    layer_id: int
    kv_source_layer: int
    kind: str
    heads: int
    kv_lora_rank: int
    qk_nope_dim: int
    qk_rope_dim: int
    v_head_dim: int
    rope_kind: str
    mrope_section: tuple[int, int, int] | None = None
    rms_norm_eps: float = 1e-5
    attention_scale: float | None = None
    sliding_window: int | None = None
    shared_consumers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.shared_consumers:
            object.__setattr__(self, "shared_consumers", (self.layer_id,))

    @property
    def effective_scale(self) -> float:
        if self.attention_scale is not None:
            return float(self.attention_scale)
        return float((self.qk_nope_dim + self.qk_rope_dim) ** -0.5)

    @property
    def has_rope(self) -> bool:
        return self.qk_rope_dim > 0 and self.rope_kind != "none"

    def validate(self) -> None:
        if self.layer_id < 0 or self.kv_source_layer < 0 or self.kv_source_layer > self.layer_id:
            raise ValueError(f"invalid layer_id/kv_source_layer: {self.layer_id}/{self.kv_source_layer}")
        if self.kind not in VALID_ATTENTION_KINDS:
            raise ValueError(f"unsupported MLA attention kind: {self.kind!r}")
        if self.rope_kind not in VALID_ROPE_KINDS:
            raise ValueError(f"unsupported MLA rope_kind: {self.rope_kind!r}")
        for name in ("heads", "kv_lora_rank", "qk_nope_dim", "v_head_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.qk_rope_dim < 0:
            raise ValueError("qk_rope_dim cannot be negative")
        if self.rope_kind == "none":
            if self.qk_rope_dim != 0:
                raise ValueError("rope_kind='none' requires qk_rope_dim == 0")
            if self.mrope_section is not None:
                raise ValueError("rope_kind='none' cannot carry mrope_section")
        else:
            if self.qk_rope_dim == 0 or self.qk_rope_dim % 2 != 0:
                raise ValueError(f"rope_kind={self.rope_kind!r} requires positive even qk_rope_dim")
            if self.mrope_section is not None:
                if len(self.mrope_section) != 3 or any(x <= 0 for x in self.mrope_section):
                    raise ValueError(f"invalid mrope_section: {self.mrope_section!r}")
                if sum(self.mrope_section) * 2 != self.qk_rope_dim:
                    raise ValueError(
                        f"mrope_section {self.mrope_section} sums to {sum(self.mrope_section)} pairs, "
                        f"expected {self.qk_rope_dim // 2}"
                    )
        if not math.isfinite(self.rms_norm_eps) or self.rms_norm_eps <= 0.0:
            raise ValueError("rms_norm_eps must be finite and positive")
        if self.attention_scale is not None and (
            not math.isfinite(self.attention_scale) or self.attention_scale <= 0.0
        ):
            raise ValueError("attention_scale must be finite and positive")
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError("sliding_window must be positive when set")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        consumers = list(self.shared_consumers) if self.shared_consumers else [self.layer_id]
        return {
            "layer_id": self.layer_id,
            "kv_source_layer": self.kv_source_layer,
            "kind": self.kind,
            "heads": self.heads,
            "kv_lora_rank": self.kv_lora_rank,
            "qk_nope_dim": self.qk_nope_dim,
            "qk_rope_dim": self.qk_rope_dim,
            "v_head_dim": self.v_head_dim,
            "rope_kind": self.rope_kind,
            "mrope_section": list(self.mrope_section) if self.mrope_section is not None else None,
            "rms_norm_eps": self.rms_norm_eps,
            "attention_scale": self.attention_scale,
            "sliding_window": self.sliding_window,
            "shared_consumers": consumers,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MlaLayerGeometry":
        mrope = data.get("mrope_section")
        consumers = data.get("shared_consumers", ())
        geom = cls(
            layer_id=int(data["layer_id"]),
            kv_source_layer=int(data["kv_source_layer"]),
            kind=str(data["kind"]),
            heads=int(data["heads"]),
            kv_lora_rank=int(data["kv_lora_rank"]),
            qk_nope_dim=int(data["qk_nope_dim"]),
            qk_rope_dim=int(data["qk_rope_dim"]),
            v_head_dim=int(data["v_head_dim"]),
            rope_kind=str(data["rope_kind"]),
            mrope_section=tuple(int(x) for x in mrope) if mrope is not None else None,
            rms_norm_eps=float(data.get("rms_norm_eps", 1e-5)),
            attention_scale=None if data.get("attention_scale") is None else float(data["attention_scale"]),
            sliding_window=None if data.get("sliding_window") is None else int(data["sliding_window"]),
            shared_consumers=tuple(int(x) for x in consumers),
        )
        geom.validate()
        return geom


@dataclass(frozen=True)
class CapturedSequence:
    """Mode-agnostic attention capture for one sequence at one MLA layer."""

    seq_id: str
    positions: torch.Tensor
    u: torch.Tensor
    q_nope: torch.Tensor
    q_rope: torch.Tensor | None
    k_rope: torch.Tensor | None
    allowed_mask: torch.Tensor | None
    gate: torch.Tensor | None
    ref_output: torch.Tensor

    @property
    def seq_len(self) -> int:
        return int(self.u.shape[0])

    def to(self, device: torch.device | str) -> "CapturedSequence":
        def _mv(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t.to(device)

        return CapturedSequence(
            seq_id=self.seq_id,
            positions=self.positions.to(device),
            u=self.u.to(device),
            q_nope=self.q_nope.to(device),
            q_rope=_mv(self.q_rope),
            k_rope=_mv(self.k_rope),
            allowed_mask=_mv(self.allowed_mask),
            gate=_mv(self.gate),
            ref_output=self.ref_output.to(device),
        )

    def validate(self, geometry: MlaLayerGeometry) -> None:
        geometry.validate()
        if not self.seq_id or any(c in self.seq_id for c in "\0\n\r/"):
            raise ValueError(f"invalid seq_id: {self.seq_id!r}")
        if self.u.ndim != 2 or self.u.shape[0] <= 0 or self.u.shape[1] != geometry.kv_lora_rank:
            raise ValueError(f"u shape {tuple(self.u.shape)} disagrees with kv_lora_rank={geometry.kv_lora_rank}")
        t = self.seq_len
        if self.positions.ndim == 1:
            if self.positions.shape != (t,):
                raise ValueError(f"1D positions shape {tuple(self.positions.shape)} != ({t},)")
        elif self.positions.ndim == 2:
            if self.positions.shape != (3, t):
                raise ValueError(f"3D M-RoPE positions shape {tuple(self.positions.shape)} != (3, {t})")
        else:
            raise ValueError(f"positions must be [T] or [3, T], got {tuple(self.positions.shape)}")
        if self.q_nope.shape != (t, geometry.heads, geometry.qk_nope_dim):
            raise ValueError(
                f"q_nope shape {tuple(self.q_nope.shape)} != ({t}, {geometry.heads}, {geometry.qk_nope_dim})"
            )
        if geometry.has_rope:
            if self.q_rope is None or self.k_rope is None:
                raise ValueError("RoPE layer capture requires q_rope and k_rope")
            if self.q_rope.shape != (t, geometry.heads, geometry.qk_rope_dim):
                raise ValueError(f"q_rope shape {tuple(self.q_rope.shape)} disagrees with geometry")
            if self.k_rope.shape != (t, geometry.qk_rope_dim):
                raise ValueError(f"k_rope shape {tuple(self.k_rope.shape)} disagrees with geometry")
        else:
            if self.q_rope is not None or self.k_rope is not None:
                raise ValueError("NoPE layer capture must set q_rope=None and k_rope=None")
        if self.allowed_mask is not None:
            if self.allowed_mask.dtype != torch.bool or self.allowed_mask.shape != (t, t):
                raise ValueError(f"allowed_mask must be bool [{t}, {t}], got {self.allowed_mask.dtype} {tuple(self.allowed_mask.shape)}")
            if not bool(self.allowed_mask.any(dim=-1).all()):
                raise ValueError("every query row must have at least one allowed key")
        if self.gate is not None and self.gate.shape != (t, geometry.heads * geometry.v_head_dim):
            raise ValueError(f"gate shape {tuple(self.gate.shape)} disagrees with heads*v_head_dim")
        if self.ref_output.ndim != 2 or self.ref_output.shape[0] != t or self.ref_output.shape[1] <= 0:
            raise ValueError(f"invalid ref_output shape: {tuple(self.ref_output.shape)}")
        for name, tensor in (
            ("u", self.u),
            ("q_nope", self.q_nope),
            ("q_rope", self.q_rope),
            ("k_rope", self.k_rope),
            ("gate", self.gate),
            ("ref_output", self.ref_output),
        ):
            if tensor is not None and not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"non-finite values in captured {name} for seq_id={self.seq_id}")


@dataclass(frozen=True)
class CapturedLayerParams:
    """Frozen layer parameters needed to evaluate attention outputs from captured states."""

    kv_norm_weight: torch.Tensor
    kv_b_weight: torch.Tensor
    o_proj_weight: torch.Tensor
    sink: torch.Tensor | None = None
    post_proj_norm_weight: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "CapturedLayerParams":
        def _mv(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t.to(device)

        return CapturedLayerParams(
            kv_norm_weight=self.kv_norm_weight.to(device),
            kv_b_weight=self.kv_b_weight.to(device),
            o_proj_weight=self.o_proj_weight.to(device),
            sink=_mv(self.sink),
            post_proj_norm_weight=_mv(self.post_proj_norm_weight),
        )

    def validate(self, geometry: MlaLayerGeometry) -> None:
        geometry.validate()
        if self.kv_norm_weight.shape != (geometry.kv_lora_rank,):
            raise ValueError(f"kv_norm_weight shape {tuple(self.kv_norm_weight.shape)} != ({geometry.kv_lora_rank},)")
        expected_kv_b = (geometry.heads * (geometry.qk_nope_dim + geometry.v_head_dim), geometry.kv_lora_rank)
        if self.kv_b_weight.shape != expected_kv_b:
            raise ValueError(f"kv_b_weight shape {tuple(self.kv_b_weight.shape)} != {expected_kv_b}")
        if self.o_proj_weight.ndim != 2 or self.o_proj_weight.shape[1] != geometry.heads * geometry.v_head_dim:
            raise ValueError(
                f"o_proj_weight shape {tuple(self.o_proj_weight.shape)} disagrees with heads*v_head_dim"
            )
        hidden = int(self.o_proj_weight.shape[0])
        if self.sink is not None and self.sink.shape != (geometry.heads,):
            raise ValueError(f"sink shape {tuple(self.sink.shape)} != ({geometry.heads},)")
        if self.post_proj_norm_weight is not None and self.post_proj_norm_weight.shape != (hidden,):
            raise ValueError(f"post_proj_norm_weight shape {tuple(self.post_proj_norm_weight.shape)} != ({hidden},)")
        for name, tensor in (
            ("kv_norm_weight", self.kv_norm_weight),
            ("kv_b_weight", self.kv_b_weight),
            ("o_proj_weight", self.o_proj_weight),
            ("sink", self.sink),
            ("post_proj_norm_weight", self.post_proj_norm_weight),
        ):
            if tensor is not None and not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"non-finite values in layer param {name}")


@dataclass(frozen=True)
class CapturedLayerBank:
    """Disjoint train/val/test captures plus frozen params for one MLA layer.

    ``train`` fits transforms, ``val`` selects fit steps and the shipped transform, and
    ``test`` produces every reported error. ``provenance["splits"]`` describes all three.
    """

    geometry: MlaLayerGeometry
    params: CapturedLayerParams
    train: tuple[CapturedSequence, ...]
    val: tuple[CapturedSequence, ...]
    test: tuple[CapturedSequence, ...]
    provenance: dict[str, Any]

    def split(self, name: str) -> tuple[CapturedSequence, ...]:
        if name not in SPLIT_NAMES:
            raise ValueError(f"unknown split {name!r}")
        return getattr(self, name)

    def validate(self) -> None:
        self.geometry.validate()
        self.params.validate(self.geometry)
        seen: dict[str, str] = {}
        for name in SPLIT_NAMES:
            seqs = self.split(name)
            if not seqs:
                raise ValueError(f"CapturedLayerBank requires at least one {name} sequence")
            ids = [s.seq_id for s in seqs]
            if len(set(ids)) != len(ids):
                raise ValueError(f"duplicate seq_id within {name} split")
            for seq_id in ids:
                if seq_id in seen:
                    raise ValueError(f"{name} and {seen[seq_id]} splits overlap at {seq_id!r}")
                seen[seq_id] = name
        check_split_description(self.provenance.get("splits"), {
            name: [s.seq_id for s in self.split(name)] for name in SPLIT_NAMES
        })
        hidden = int(self.params.o_proj_weight.shape[0])
        for name in SPLIT_NAMES:
            for seq in self.split(name):
                seq.validate(self.geometry)
                if seq.ref_output.shape[1] != hidden:
                    raise ValueError(f"ref_output width {seq.ref_output.shape[1]} != o_proj out {hidden}")


def split_train_val_test(
    items: Sequence[Any],
    *,
    train_count: int,
    val_count: int,
    test_count: int,
    key_fn: Callable[[Any], str] | None = None,
    seed: int = 0,
) -> tuple[list[Any], list[Any], list[Any]]:
    """Deterministically partition items into disjoint train, val and test slices.

    Items are ordered by SHA256 over ``SPLIT_SALT``, the seed and each item's key, then
    sliced. Changing the salt, seed, counts or the key set changes the partition, and
    ``describe_splits`` records all of them so a reused capture can be checked.
    """
    counts = {"train": train_count, "val": val_count, "test": test_count}
    for name, count in counts.items():
        if type(count) is not int or count <= 0:
            raise ValueError(f"{name}_count must be a positive integer")
    total = train_count + val_count + test_count
    if len(items) < total:
        raise ValueError(f"need at least {total} items, got {len(items)}")
    keys = [str(item) if key_fn is None else str(key_fn(item)) for item in items]
    if len(set(keys)) != len(keys):
        raise ValueError("split keys must be unique")

    def _order(index: int) -> str:
        return hashlib.sha256(f"{SPLIT_SALT}:{seed}:{keys[index]}".encode("utf-8")).hexdigest()

    ordered = [items[i] for i in sorted(range(len(items)), key=_order)]
    return (
        ordered[:train_count],
        ordered[train_count : train_count + val_count],
        ordered[train_count + val_count : total],
    )


def sequence_sha256(tensor: torch.Tensor) -> str:
    """SHA256 of a sequence's contiguous CPU bytes (token IDs for corpus rows)."""
    return hashlib.sha256(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def describe_splits(
    splits: Mapping[str, Sequence[tuple[str, str]]],
    *,
    seed: int,
) -> dict[str, Any]:
    """Describe three splits as ``(seq_id, content_sha256)`` rows plus a digest per split.

    The description goes into capture provenance, the sidecar's calibration provenance
    and the fit checkpoint identity, so any split change refuses a stale reuse.
    """
    if set(splits) != set(SPLIT_NAMES):
        raise ValueError(f"splits must be exactly {SPLIT_NAMES}, got {sorted(splits)}")
    out: dict[str, Any] = {"salt": SPLIT_SALT, "seed": int(seed)}
    for name in SPLIT_NAMES:
        rows = [(str(seq_id), str(digest)) for seq_id, digest in splits[name]]
        out[name] = {
            "count": len(rows),
            "seq_ids": [seq_id for seq_id, _ in rows],
            "sha256": [digest for _, digest in rows],
            "digest": hashlib.sha256(_canonical_json(rows).encode("utf-8")).hexdigest(),
        }
    check_split_description(out)
    return out


def check_split_description(
    description: Any,
    seq_ids: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Fail closed unless ``description`` is a current three-way split (matching ``seq_ids``)."""
    if not isinstance(description, Mapping):
        raise ValueError("capture provenance requires a three-way 'splits' description")
    if description.get("salt") != SPLIT_SALT:
        raise ValueError(
            f"split salt {description.get('salt')!r} != {SPLIT_SALT!r}: refuse a capture split under another scheme"
        )
    if type(description.get("seed")) is not int:
        raise ValueError("split description seed must be an integer")
    if set(description) != {"salt", "seed", *SPLIT_NAMES}:
        raise ValueError(f"split description fields must be salt, seed and {SPLIT_NAMES}")
    seen: set[str] = set()
    for name in SPLIT_NAMES:
        row = description[name]
        if not isinstance(row, Mapping) or set(row) != {"count", "seq_ids", "sha256", "digest"}:
            raise ValueError(f"invalid {name} split description")
        ids, digests = list(row["seq_ids"]), list(row["sha256"])
        if type(row["count"]) is not int or row["count"] <= 0 or row["count"] != len(ids) or len(ids) != len(digests):
            raise ValueError(f"{name} split count disagrees with its sequence rows")
        if not all(isinstance(d, str) and len(d) == 64 for d in digests):
            raise ValueError(f"{name} split sequence hashes must be SHA256 hex digests")
        expected = hashlib.sha256(_canonical_json([[i, d] for i, d in zip(ids, digests)]).encode("utf-8")).hexdigest()
        if row["digest"] != expected:
            raise ValueError(f"{name} split digest mismatch")
        if seen & set(ids) or len(set(ids)) != len(ids):
            raise ValueError(f"{name} split overlaps another split or repeats a sequence")
        seen |= set(ids)
        if seq_ids is not None and list(seq_ids[name]) != ids:
            raise ValueError(f"{name} split sequences disagree with the capture's split description")
    return dict(description)


def eval_positional_qk(
    seq: CapturedSequence,
    geometry: MlaLayerGeometry,
    *,
    rope_query_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    rope_key_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate unscaled positional QK scores Z_P [H, T, T] and valid causal mask [T, T]."""
    if not geometry.has_rope or seq.q_rope is None or seq.k_rope is None:
        raise ValueError(f"layer {geometry.layer_id} has no RoPE path (rope_kind={geometry.rope_kind!r})")
    qp = seq.q_rope.float()
    kp = seq.k_rope.float()
    if rope_query_fn is not None:
        qp = rope_query_fn(qp)
    if rope_key_fn is not None:
        kp = rope_key_fn(kp)
    # qp: [T, H, d_r] -> [H, T, d_r]; kp: [T, d_r] -> [1, d_r, T]
    z_p = torch.matmul(qp.transpose(0, 1), kp.transpose(0, 1).unsqueeze(0))
    mask = (
        seq.allowed_mask
        if seq.allowed_mask is not None
        else causal_mask(seq.seq_len, device=qp.device, sliding_window=geometry.sliding_window)
    )
    return z_p, mask


def eval_attention_output(
    seq: CapturedSequence,
    params: CapturedLayerParams,
    geometry: MlaLayerGeometry,
    *,
    content_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    kv_b_override: torch.Tensor | None = None,
    rope_query_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    rope_key_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    exact_content_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    exact_rope_key_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    sink_tokens: int = 0,
    recent_tokens: int = 0,
) -> torch.Tensor:
    """Evaluate exact post-o_proj (and optional post_proj_norm) attention output in FP32.

    With all hook arguments left as None, reproduces the full-precision teacher
    attention output for any MLA family captured via ``CapturedSequence``.
    """
    if sink_tokens < 0 or recent_tokens < 0:
        raise ValueError("sink_tokens and recent_tokens must be nonnegative")
    protect = geometry.kind != "swa" and (sink_tokens or recent_tokens)
    t = seq.seq_len
    h = geometry.heads
    nope = geometry.qk_nope_dim
    vdim = geometry.v_head_dim
    scale = geometry.effective_scale

    u = seq.u.float()
    if content_fn is not None:
        latent = content_fn(u)
    else:
        latent = u * params.kv_norm_weight.float()

    kv_b = (params.kv_b_weight if kv_b_override is None else kv_b_override).float()
    expanded = linear(latent, kv_b).reshape(t, h, nope + vdim)
    keys, values = expanded.split([nope, vdim], dim=-1)
    exact_keys = None
    exact_values = None
    if protect and exact_content_fn is not None:
        exact_kv = linear(exact_content_fn(u), kv_b).reshape_as(expanded)
        exact_keys, exact_values = exact_kv.split([nope, vdim], dim=-1)
    elif protect and exact_rope_key_fn is not None:
        exact_keys = keys

    query = seq.q_nope.float()
    if geometry.has_rope:
        if seq.q_rope is None or seq.k_rope is None:
            raise ValueError("missing q_rope/k_rope for RoPE layer")
        qp = seq.q_rope.float()
        kp = seq.k_rope.float()
        if rope_query_fn is not None:
            qp = rope_query_fn(qp)
        if rope_key_fn is not None:
            kp = rope_key_fn(kp)
        query = torch.cat((query, qp), dim=-1)
        keys = torch.cat((keys, kp[:, None, :].expand(-1, h, -1)), dim=-1)
        if exact_keys is not None:
            exact_kp = exact_rope_key_fn(seq.k_rope.float()) if exact_rope_key_fn is not None else kp
            exact_keys = torch.cat((exact_keys, exact_kp[:, None, :].expand(-1, h, -1)), dim=-1)

    def tile_mask(start, stop, first, width, query_id):
        allowed = seq.allowed_mask[start:stop, first:stop]
        return F.pad(allowed, (0, width - (stop - first), 0, ROWS - (stop - start)))

    gate = seq.gate.float().reshape_as(values) if seq.gate is not None else None
    sink = params.sink.float()[:, None, None] if params.sink is not None else None
    attended = tiled_sink_attention(
        query, keys, values, gate, sink, scale, sliding_window=geometry.sliding_window,
        tile_mask=tile_mask if seq.allowed_mask is not None else None,
        exact_keys=exact_keys, exact_values=exact_values, sink_tokens=sink_tokens, recent_tokens=recent_tokens,
    )
    out = linear(attended, params.o_proj_weight.float())
    if params.post_proj_norm_weight is not None:
        out = rms_norm(out, params.post_proj_norm_weight.float(), geometry.rms_norm_eps)
    return out


def _write_safetensors_streamed(path: Path, tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Write one sequential safetensors file while streaming XXH3-128 in the write pass."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    if tmp.is_symlink() or path.is_symlink():
        raise ValueError(f"refuse symlink capture path: {path}")

    header: dict[str, Any] = {}
    ordered: list[tuple[str, bytes]] = []
    offset = 0
    for name in sorted(tensors):
        tensor = tensors[name].detach().contiguous().cpu()
        st_dtype = _TORCH_TO_ST_DTYPE.get(tensor.dtype)
        if st_dtype is None:
            raise ValueError(f"unsupported capture tensor dtype {tensor.dtype} for {name}")
        raw = tensor.view(torch.uint8).numpy().tobytes()
        end = offset + len(raw)
        header[name] = {"dtype": st_dtype, "shape": list(tensor.shape), "data_offsets": [offset, end]}
        ordered.append((name, raw))
        offset = end

    encoded = _canonical_json(header).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    prefix = struct.pack("<Q", len(encoded)) + encoded
    hasher = xxhash.xxh3_128()
    with tmp.open("wb") as handle:
        handle.write(prefix)
        hasher.update(prefix)
        for _, raw in ordered:
            handle.write(raw)
            hasher.update(raw)
        handle.flush()
        os.fsync(handle.fileno())
    total_bytes = len(prefix) + offset
    if tmp.stat().st_size != total_bytes:
        raise ValueError("capture payload size mismatch after write")
    os.replace(tmp, path)
    _fsync_dir(path.parent)
    return {"file": path.name, "bytes": total_bytes, "xxh3_128": hasher.hexdigest()}


def _read_safetensors_verified(path: Path, expected_bytes: int, expected_xxh3: str) -> dict[str, torch.Tensor]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"capture payload missing or symlink: {path}")
    raw = path.read_bytes()
    if len(raw) != expected_bytes:
        raise ValueError(f"capture payload byte length {len(raw)} != expected {expected_bytes}")
    actual_xxh3 = xxhash.xxh3_128_hexdigest(raw)
    if actual_xxh3 != expected_xxh3:
        raise ValueError(f"capture payload XXH3-128 mismatch: {actual_xxh3} != {expected_xxh3}")
    header_len = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_len].decode("utf-8"))
    base = 8 + header_len
    result: dict[str, torch.Tensor] = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        dtype = _ST_DTYPE_TO_TORCH[spec["dtype"]]
        start, end = spec["data_offsets"]
        buf = bytearray(raw[base + start : base + end])
        if not buf:
            result[name] = torch.empty(spec["shape"], dtype=dtype)
        else:
            result[name] = torch.frombuffer(buf, dtype=dtype).reshape(spec["shape"]).clone()
    return result


def save_layer_capture(directory: Path | str, bank: CapturedLayerBank) -> dict[str, Any]:
    """Persist one CapturedLayerBank sequentially with streamed XXH3-128 receipt."""
    bank.validate()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {
        "params.kv_norm_weight": bank.params.kv_norm_weight.float(),
        "params.kv_b_weight": bank.params.kv_b_weight.float(),
        "params.o_proj_weight": bank.params.o_proj_weight.float(),
    }
    if bank.params.sink is not None:
        tensors["params.sink"] = bank.params.sink.float()
    if bank.params.post_proj_norm_weight is not None:
        tensors["params.post_proj_norm_weight"] = bank.params.post_proj_norm_weight.float()

    split_meta: dict[str, list[dict[str, Any]]] = {}
    for split_name in SPLIT_NAMES:
        seqs = bank.split(split_name)
        rows = []
        for idx, seq in enumerate(seqs):
            prefix = f"{split_name}.{idx:05d}."
            tensors[prefix + "positions"] = seq.positions.to(torch.int64)
            tensors[prefix + "u"] = seq.u.float()
            tensors[prefix + "q_nope"] = seq.q_nope.float()
            tensors[prefix + "ref_output"] = seq.ref_output.float()
            if seq.q_rope is not None:
                tensors[prefix + "q_rope"] = seq.q_rope.float()
            if seq.k_rope is not None:
                tensors[prefix + "k_rope"] = seq.k_rope.float()
            if seq.allowed_mask is not None:
                tensors[prefix + "allowed_mask"] = seq.allowed_mask.to(torch.bool)
            if seq.gate is not None:
                tensors[prefix + "gate"] = seq.gate.float()
            rows.append({
                "index": idx,
                "seq_id": seq.seq_id,
                "seq_len": seq.seq_len,
                "has_mask": seq.allowed_mask is not None,
                "has_gate": seq.gate is not None,
            })
        split_meta[split_name] = rows

    stem = f"layer-{bank.geometry.layer_id:04d}"
    payload_receipt = _write_safetensors_streamed(directory / f"{stem}.safetensors", tensors)
    manifest = {
        "schema": CAPTURE_SCHEMA,
        "geometry": bank.geometry.to_dict(),
        "provenance": bank.provenance,
        "has_sink": bank.params.sink is not None,
        "has_post_proj_norm": bank.params.post_proj_norm_weight is not None,
        "splits": split_meta,
        "payload": payload_receipt,
    }
    _atomic_json(directory / f"{stem}.json", manifest)
    return manifest


def load_layer_capture(directory: Path | str, layer_id: int) -> CapturedLayerBank:
    """Reload and verify one CapturedLayerBank against its XXH3-128 receipt."""
    directory = Path(directory)
    stem = f"layer-{layer_id:04d}"
    manifest = json.loads((directory / f"{stem}.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != CAPTURE_SCHEMA:
        raise ValueError(
            f"unsupported capture schema: {manifest.get('schema')!r} (expected {CAPTURE_SCHEMA!r}); "
            "recapture under the three-way train/val/test split"
        )
    geometry = MlaLayerGeometry.from_dict(manifest["geometry"])
    if geometry.layer_id != layer_id:
        raise ValueError(f"capture layer_id {geometry.layer_id} != requested {layer_id}")
    payload = manifest["payload"]
    tensors = _read_safetensors_verified(
        directory / payload["file"], int(payload["bytes"]), str(payload["xxh3_128"])
    )
    params = CapturedLayerParams(
        kv_norm_weight=tensors["params.kv_norm_weight"],
        kv_b_weight=tensors["params.kv_b_weight"],
        o_proj_weight=tensors["params.o_proj_weight"],
        sink=tensors["params.sink"] if manifest["has_sink"] else None,
        post_proj_norm_weight=tensors["params.post_proj_norm_weight"] if manifest["has_post_proj_norm"] else None,
    )
    loaded: dict[str, tuple[CapturedSequence, ...]] = {}
    for split_name in SPLIT_NAMES:
        seqs = []
        for row in manifest["splits"][split_name]:
            prefix = f"{split_name}.{int(row['index']):05d}."
            seqs.append(
                CapturedSequence(
                    seq_id=str(row["seq_id"]),
                    positions=tensors[prefix + "positions"],
                    u=tensors[prefix + "u"],
                    q_nope=tensors[prefix + "q_nope"],
                    q_rope=tensors.get(prefix + "q_rope"),
                    k_rope=tensors.get(prefix + "k_rope"),
                    allowed_mask=tensors[prefix + "allowed_mask"] if row["has_mask"] else None,
                    gate=tensors[prefix + "gate"] if row["has_gate"] else None,
                    ref_output=tensors[prefix + "ref_output"],
                )
            )
        loaded[split_name] = tuple(seqs)

    bank = CapturedLayerBank(
        geometry=geometry,
        params=params,
        train=loaded["train"],
        val=loaded["val"],
        test=loaded["test"],
        provenance=dict(manifest.get("provenance", {})),
    )
    bank.validate()
    return bank


class QmlaCaptureAdapter(ABC):
    """Family adapter contract: implement only geometry, RoPE, and capture hooks."""

    family_id: str

    @abstractmethod
    def mla_geometries(self, *, include_swa: bool = False) -> tuple[MlaLayerGeometry, ...]:
        """Return ordered MlaLayerGeometry entries for all eligible MLA layers."""

    @abstractmethod
    def layer_params(self, layer: Any) -> CapturedLayerParams:
        """Extract frozen layer parameters for exact reference attention evaluation."""

    @abstractmethod
    def capture_sequence(
        self,
        inputs: torch.Tensor,
        layer: Any,
        *,
        seq_id: str,
        positions: torch.Tensor | None = None,
    ) -> tuple[CapturedSequence, torch.Tensor]:
        """Run exact attention forward on one sequence and return (capture, ref_output)."""

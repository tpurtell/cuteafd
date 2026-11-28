"""Mode-agnostic MLA capture interface, exact reference evaluators, and XXH3-128 storage.

Every MLA family adapter (starting with Hy4-flash in ``quantization/hy4_flash/``)
produces ``CapturedLayerBank`` instances through ``QmlaCaptureAdapter`` so all
downstream calibration, baseline evaluation, validation, and sidecar assembly
remain completely family-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
import hashlib
import json
import math
import mmap
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

CAPTURE_SCHEMA = "cuteafd-qmla-capture-v3"
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


def layer_memory_bytes(device: torch.device | str) -> dict[str, int]:
    """Layer-boundary footprint. System available memory includes reclaimable file cache."""
    keys = {"VmRSS": "rss", "RssAnon": "rss_anon", "RssFile": "rss_file",
            "MemAvailable": "system_available", "MemTotal": "system_total"}
    result = {}
    for path in (Path("/proc/self/status"), Path("/proc/meminfo")):
        if path.is_file():
            for line in path.read_text().splitlines():
                label, _, value = line.partition(":")
                if label in keys:
                    result[keys[label]] = int(value.split()[0]) * 1024
    dev = torch.device(device)
    if dev.type == "cuda":
        result.update(cuda_allocated=torch.cuda.memory_allocated(dev),
                      cuda_reserved=torch.cuda.memory_reserved(dev),
                      cuda_peak_allocated=torch.cuda.max_memory_allocated(dev))
    return result


def available_device_memory_bytes(device: torch.device | str) -> int | None:
    """Unified memory uses reclaimable host memory, discrete CUDA uses free VRAM."""
    dev = torch.device(device)
    mem = layer_memory_bytes("cpu")
    if dev.type != "cuda" or not torch.cuda.is_available():
        return mem.get("system_available")
    props = torch.cuda.get_device_properties(dev)
    if getattr(props, "is_integrated", False) or (props.major, props.minor) == (12, 1):
        return mem.get("system_available")
    return int(torch.cuda.mem_get_info(dev)[0])


def group_token_records(records, device, *, bytes_per_token, reserve_bytes=32 * 1024**3):
    """Token-budgeted consecutive groups; an oversized singleton is never split."""
    available = available_device_memory_bytes(device) or 0
    budget = max(1, (available - reserve_bytes) // (5 * bytes_per_token))
    group, tokens = [], 0
    for record in records:
        count = record[1].numel()
        if group and tokens + count > budget:
            yield group
            group, tokens = [], 0
        group.append(record)
        tokens += count
    if group:
        yield group


class DeviceSplit(Sequence):
    """Reiterable split, either cached as a whole or transferred one row at a time."""

    def __init__(self, seqs, device, *, budget_bytes=0):
        self.device = torch.device(device)
        self.source = seqs
        self.resident = None
        if isinstance(seqs, _MappedSplit):
            size = seqs.device_bytes
            self.tokens = sum(row["seq_len"] for row in seqs.rows)
        else:
            size = sum(t.numel() * t.element_size() for seq in seqs for f in fields(seq)
                       if isinstance(t := getattr(seq, f.name), torch.Tensor))
            self.tokens = sum(seq.seq_len for seq in seqs)
        if size <= budget_bytes:
            self.resident = tuple(seq.to(device) for seq in seqs)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        return self.resident[index] if self.resident is not None else self.source[index].to(self.device)


def split_residency_budget(device):
    # Leave capacity for per-sequence workspaces, temporary compensated weights,
    # training batches and (on GB10) reclaimable capture file cache.
    available = available_device_memory_bytes(device) or 0
    return max(0, (available - 32 * 1024**3) // 2)


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
class SparseSelection:
    """Compact per-query top-k DSA group selection (avoids O(T^2) boolean masks).

    ``groups`` has shape ``[T, group_topk]`` (int32/int64). Each nonnegative group id ``g``
    selects token slots ``[g * compress_ratio, (g + 1) * compress_ratio)`` clamped to the
    sequence end. Queries with slot index ``< candidate_topk`` attend causally to all keys.
    """

    groups: torch.Tensor
    compress_ratio: int
    candidate_topk: int

    @property
    def seq_len(self) -> int:
        return int(self.groups.shape[0])

    @property
    def group_topk(self) -> int:
        return int(self.groups.shape[1])

    def to(self, device: torch.device | str) -> "SparseSelection":
        return SparseSelection(
            groups=self.groups.to(device),
            compress_ratio=self.compress_ratio,
            candidate_topk=self.candidate_topk,
        )

    def validate(self, seq_len: int) -> None:
        if self.groups.ndim != 2 or self.groups.shape[0] != seq_len or self.groups.shape[1] <= 0:
            raise ValueError(
                f"sparse_selection.groups must be [{seq_len}, k] with k > 0, got {tuple(self.groups.shape)}"
            )
        if self.groups.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"sparse_selection.groups must be int32 or int64, got {self.groups.dtype}")
        if self.compress_ratio <= 0 or self.candidate_topk <= 0:
            raise ValueError("sparse_selection compress_ratio and candidate_topk must be positive")

    def tile_mask(
        self,
        start: int,
        stop: int,
        first: int,
        width: int,
        query_id: torch.Tensor,
        *,
        seq_len: int | None = None,
    ) -> torch.Tensor:
        """Return the [ROWS, width] boolean DSA mask expected by ``tiled_sink_attention``."""
        t = self.seq_len if seq_len is None else int(seq_len)
        dev = query_id.device
        rows = int(query_id.shape[0])
        if stop <= self.candidate_topk:
            return torch.ones((rows, width), dtype=torch.bool, device=dev)
        g = self.groups[start:stop].to(device=dev, dtype=torch.int64)
        valid_g = g >= 0
        ratio = int(self.compress_ratio)
        offsets = torch.arange(ratio, device=dev, dtype=torch.int64)
        slots = (g.clamp_min(0)[..., None] * ratio + offsets).flatten(1)
        valid_slots = valid_g.unsqueeze(-1).expand(-1, -1, ratio).flatten(1)
        local_slots = slots.clamp_max(t - 1) - first
        in_range = valid_slots & (local_slots >= 0) & (local_slots < width)
        scatter_idx = torch.where(in_range, local_slots.clamp(0, width), width)
        selected = torch.zeros((rows, width + 1), dtype=torch.bool, device=dev)
        selected[: stop - start].scatter_(1, scatter_idx, True)
        return selected[:, :width] | (query_id[:, None] < self.candidate_topk)


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
    sparse_selection: SparseSelection | None = None

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
            sparse_selection=None if self.sparse_selection is None else self.sparse_selection.to(device),
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
        if self.sparse_selection is not None:
            self.sparse_selection.validate(t)
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
    attention_arithmetic: str = "f32_tiled"

    def to(self, device: torch.device | str) -> "CapturedLayerParams":
        def _mv(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t.to(device)

        return CapturedLayerParams(
            kv_norm_weight=self.kv_norm_weight.to(device),
            kv_b_weight=self.kv_b_weight.to(device),
            o_proj_weight=self.o_proj_weight.to(device),
            sink=_mv(self.sink),
            post_proj_norm_weight=_mv(self.post_proj_norm_weight),
            attention_arithmetic=self.attention_arithmetic,
        )

    def validate(self, geometry: MlaLayerGeometry) -> None:
        geometry.validate()
        if self.attention_arithmetic not in ("f32_tiled", "f32_tiled_bf16_cache", "bf16_eager", "bf16_sdpa"):
            raise ValueError(f"unsupported attention arithmetic {self.attention_arithmetic!r}")
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
    train: Sequence[CapturedSequence]
    val: Sequence[CapturedSequence]
    test: Sequence[CapturedSequence]
    provenance: dict[str, Any]

    def split(self, name: str) -> Sequence[CapturedSequence]:
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


def native_masked_attention(query, keys, values, mask, scale, arithmetic):
    """Unbatched [T,H,D] attention with the native BF16 teacher's rounding boundaries."""
    q, k, v = (x.transpose(0, 1).unsqueeze(0) for x in (query, keys, values))
    if arithmetic == "bf16_sdpa":
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[None, None], scale=scale)
    elif arithmetic == "bf16_eager":
        scores = (q @ k.transpose(-1, -2)) * scale
        probs = scores.masked_fill(~mask, torch.finfo(q.dtype).min).softmax(-1, dtype=torch.float32).to(q.dtype)
        out = probs @ v
    else:
        raise ValueError(f"unsupported native attention arithmetic: {arithmetic}")
    return out[0].transpose(0, 1)


def cached_content(u: torch.Tensor, params: CapturedLayerParams) -> torch.Tensor:
    """Recreate the checkpoint's cached weighted latent from the saved RMSNorm input.

    BF16 adapters save u after its native cast. The product is rounded to BF16
    before any transform, even when a capture serialized u as an FP32 tensor.
    The FP32 reference adapters retain their native weighted FP32 product.
    """
    c = u.float() * params.kv_norm_weight.float()
    return c if params.attention_arithmetic == "f32_tiled" else c.bfloat16().float()


def _native_bf16_output(seq, params, geom, latent, kv_b_override, exact_content_fn,
                        sink_tokens, recent_tokens):
    if geom.has_rope or geom.kind != "dsa" or seq.gate is not None or params.sink is not None:
        raise ValueError("native BF16 arithmetic currently supports ungated NoPE DSA")
    dtype = torch.bfloat16
    out_dim = geom.qk_nope_dim + geom.v_head_dim
    count = seq.seq_len

    def expand(z, weight):
        kv = F.linear(z.to(dtype), weight.to(dtype))
        return kv.reshape(count, geom.heads, out_dim).split([geom.qk_nope_dim, geom.v_head_dim], dim=-1)

    weight = params.kv_b_weight if kv_b_override is None else kv_b_override
    keys, values = expand(latent, weight)
    query = seq.q_nope.to(dtype)
    protect = exact_content_fn is not None and (sink_tokens or recent_tokens)
    if protect:
        exact_k, exact_v = expand(cached_content(seq.u, params), params.kv_b_weight)
    # Match the native prefill's shape and SDPA backend for ordinary calibration sequences.
    # Only long contexts or query-dependent selection of the exact/quantized KV require tiles.
    step = count if seq.sparse_selection is None and not protect else ROWS
    pieces = []
    for start in range(0, count, step):
        stop = min(start + step, count)
        rows = torch.arange(start, stop, device=query.device)
        # Keep the native key reduction extent across query tiles. Shortening it
        # at every causal boundary changes the SDPA accumulation order.
        width = count
        slots = torch.arange(width, device=query.device)
        if seq.allowed_mask is not None:
            allowed = seq.allowed_mask[start:stop, :width]
        else:
            allowed = slots[None, :] <= rows[:, None]
            if seq.sparse_selection is not None:
                allowed &= seq.sparse_selection.tile_mask(start, stop, 0, width, rows)
        key, value = keys[:width], values[:width]
        if protect:
            # Put one exact or one quantized version of each key in the same softmax.
            # Only sink/recent regions can be selected from the exact copy in this tile.
            protected = (slots < sink_tokens) | ((rows[:, None] - slots) < recent_tokens)
            exact_slots = slots[(slots < stop) & ((slots < sink_tokens) | (slots >= start + 1 - recent_tokens))]
            key = torch.cat((key, exact_k[exact_slots]))
            value = torch.cat((value, exact_v[exact_slots]))
            allowed = torch.cat((allowed & ~protected, (allowed & protected)[:, exact_slots]), dim=1)
        pieces.append(native_masked_attention(query[start:stop], key, value, allowed, geom.effective_scale,
                                              params.attention_arithmetic))
    attended = torch.cat(pieces).reshape(count, geom.heads * geom.v_head_dim)
    return F.linear(attended, params.o_proj_weight.to(dtype)).float()


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

    Uses ``tiled_sink_attention`` (128-row query tiles over the full causal KV history,
    with per-query sink/recent protection). Supports both dense ``allowed_mask`` and
    compact ``sparse_selection`` group IDs so long sequences never materialize ``[T, T]``.
    With all hook arguments left as None, reproduces the full-precision teacher output on
    the exact same code path used by every quantized arm.
    """
    if sink_tokens < 0 or recent_tokens < 0:
        raise ValueError("sink_tokens and recent_tokens must be nonnegative")
    protect = geometry.kind != "swa" and (sink_tokens or recent_tokens)
    t = seq.seq_len
    h = geometry.heads
    nope = geometry.qk_nope_dim
    vdim = geometry.v_head_dim
    scale = geometry.effective_scale

    c = cached_content(seq.u, params)
    latent = content_fn(c) if content_fn is not None else c

    if params.attention_arithmetic.startswith("bf16_"):
        return _native_bf16_output(seq, params, geometry, latent, kv_b_override, exact_content_fn,
                                   sink_tokens, recent_tokens)

    kv_b = (params.kv_b_weight if kv_b_override is None else kv_b_override).float()
    expanded = linear(latent, kv_b).reshape(t, h, nope + vdim)
    keys, values = expanded.split([nope, vdim], dim=-1)
    exact_keys = None
    exact_values = None
    if protect and exact_content_fn is not None:
        exact_kv = linear(c, params.kv_b_weight.float()).reshape_as(expanded)
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
        if not torch.is_grad_enabled():
            values = values.contiguous()
            del expanded
        if exact_keys is not None:
            exact_kp = exact_rope_key_fn(seq.k_rope.float()) if exact_rope_key_fn is not None else kp
            exact_keys = torch.cat((exact_keys, exact_kp[:, None, :].expand(-1, h, -1)), dim=-1)
            if not torch.is_grad_enabled() and exact_values is not None:
                exact_values = exact_values.contiguous()
                del exact_kv

    tile_mask = None
    if seq.allowed_mask is not None:
        def tile_mask(start, stop, first, width, query_id):
            allowed = seq.allowed_mask[start:stop, first:stop]
            return F.pad(allowed, (0, width - (stop - first), 0, ROWS - (stop - start)))
    elif seq.sparse_selection is not None:
        sparse_sel = seq.sparse_selection

        def tile_mask(start, stop, first, width, query_id):
            return sparse_sel.tile_mask(start, stop, first, width, query_id, seq_len=t)

    gate = seq.gate.float().reshape_as(values) if seq.gate is not None else None
    sink = params.sink.float()[:, None, None] if params.sink is not None else None
    attended = tiled_sink_attention(
        query, keys, values, gate, sink, scale, sliding_window=geometry.sliding_window,
        tile_mask=tile_mask,
        exact_keys=exact_keys, exact_values=exact_values, sink_tokens=sink_tokens, recent_tokens=recent_tokens,
    )
    del query, keys, values, exact_keys, exact_values
    out = linear(attended, params.o_proj_weight.float())
    del attended
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
    offset = 0
    for name in sorted(tensors):
        tensor = tensors[name]
        st_dtype = _TORCH_TO_ST_DTYPE.get(tensor.dtype)
        if st_dtype is None:
            raise ValueError(f"unsupported capture tensor dtype {tensor.dtype} for {name}")
        end = offset + tensor.numel() * tensor.element_size()
        header[name] = {"dtype": st_dtype, "shape": list(tensor.shape), "data_offsets": [offset, end]}
        offset = end

    encoded = _canonical_json(header).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    prefix = struct.pack("<Q", len(encoded)) + encoded
    hasher = xxhash.xxh3_128()
    with tmp.open("wb") as handle:
        handle.write(prefix)
        hasher.update(prefix)
        for name in header:
            tensor = tensors[name].detach().cpu().contiguous()
            raw = memoryview(tensor.reshape(-1).view(torch.uint8).numpy())
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


def _verified_mapping(path: Path, expected_bytes: int, expected_xxh3: str) -> mmap.mmap:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"capture payload missing or symlink: {path}")
    with path.open("rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        if size != expected_bytes or size < 8:
            raise ValueError(f"capture payload byte length {size} != expected {expected_bytes}")
        # Every tensor owns a reference to this private mapping. Clean pages can be
        # evicted between layers and a tensor write cannot alter a published receipt.
        raw = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_COPY)
    actual_xxh3 = xxhash.xxh3_128_hexdigest(raw)
    if actual_xxh3 != expected_xxh3:
        raise ValueError(f"capture payload XXH3-128 mismatch: {actual_xxh3} != {expected_xxh3}")
    return raw


def _mapped_tensors(raw: mmap.mmap) -> dict[str, torch.Tensor]:
    size = len(raw)
    header_len = struct.unpack_from("<Q", raw)[0]
    base = 8 + header_len
    header = json.loads(raw[8:base])
    result: dict[str, torch.Tensor] = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        dtype = _ST_DTYPE_TO_TORCH[spec["dtype"]]
        start, end = spec["data_offsets"]
        count = math.prod(spec["shape"])
        if start < 0 or base + end > size or end - start != count * dtype.itemsize:
            raise ValueError(f"invalid capture payload offsets for {name}")
        if count:
            result[name] = torch.frombuffer(raw, dtype=dtype, count=count, offset=base + start).reshape(spec["shape"])
        else:
            result[name] = torch.empty(spec["shape"], dtype=dtype)
    return result


def _read_safetensors_verified(path: Path, expected_bytes: int, expected_xxh3: str) -> dict[str, torch.Tensor]:
    return _mapped_tensors(_verified_mapping(path, expected_bytes, expected_xxh3))


@dataclass(frozen=True)
class AttentionInput:
    seq_id: str
    hidden: torch.Tensor
    positions: torch.Tensor
    allowed_mask: torch.Tensor | None

    @property
    def seq_len(self):
        return self.hidden.shape[0]


class LayerCaptureWriter:
    """Persist BF16 hidden inputs and exact DSA mask metadata, one row at a time."""

    def __init__(self, directory, geometry, provenance):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.geometry = geometry
        self.provenance = provenance
        self.stem = f"layer-{geometry.layer_id:04d}"
        self.splits = {name: [] for name in SPLIT_NAMES}
        description = check_split_description(provenance.get("splits"))
        self.expected = [(name, sid) for name in SPLIT_NAMES for sid in description[name]["seq_ids"]]
        self.count = 0

    def append(self, row: AttentionInput):
        if self.count >= len(self.expected) or row.seq_id != self.expected[self.count][1]:
            raise ValueError(f"unexpected capture row {row.seq_id!r} at index {self.count}")
        name, _ = self.expected[self.count]
        tensors = {"hidden": row.hidden.detach().bfloat16(), "positions": row.positions.long()}
        if row.allowed_mask is not None:
            # Store exact selected token indices in packed bit rows (8x smaller than bool).
            mask = row.allowed_mask
            pad = (-mask.shape[-1]) % 8
            bits = F.pad(mask, (0, pad)).reshape(mask.shape[0], -1, 8).to(torch.uint8)
            tensors["mask_bits"] = (bits << torch.arange(8, device=bits.device, dtype=torch.uint8)).sum(-1).to(torch.uint8)
        payload = _write_safetensors_streamed(self.directory / f"{self.stem}-{self.count:05d}.safetensors", tensors)
        self.splits[name].append({"seq_id": row.seq_id, "seq_len": row.seq_len, "payload": payload})
        self.count += 1

    def finish(self):
        if self.count != len(self.expected):
            raise ValueError(f"incomplete layer capture: {self.count} != {len(self.expected)}")
        manifest = {"schema": CAPTURE_SCHEMA, "geometry": self.geometry.to_dict(),
                    "provenance": self.provenance, "splits": self.splits}
        _atomic_json(self.directory / f"{self.stem}.json", manifest)
        return manifest


class AttentionReplay:
    """One checkpoint-backed MLA layer at a time, shared across all capture banks."""

    def __init__(self, load, replay):
        self.load = load
        self.replay_fn = replay
        self.layer_id = None
        self.layer = self.params = None

    def get(self, layer_id):
        if self.layer_id != layer_id:
            self.layer = self.params = None
            self.layer, self.params = self.load(layer_id)
            self.layer_id = layer_id
        return self.layer, self.params

    def replay(self, layer_id, row):
        layer, params = self.get(layer_id)
        return self.replay_fn(layer, params, row)


class _MappedSplit(Sequence[CapturedSequence]):
    def __init__(self, directory, rows, geometry, replay):
        self.directory, self.rows, self.geometry, self.replay = directory, rows, geometry, replay

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        row = self.rows[index]
        payload = row["payload"]
        path = self.directory / payload["file"]
        with path.open("rb") as handle:
            raw = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_COPY)
        t = _mapped_tensors(raw)
        if t["hidden"].dtype != torch.bfloat16 or t["hidden"].shape[0] != row["seq_len"]:
            raise ValueError("invalid BF16 attention input")
        mask = t.get("mask_bits")
        if mask is not None:
            mask = ((mask[..., None] >> torch.arange(8, dtype=torch.uint8)) & 1).bool().flatten(1)[:, :row["seq_len"]]
        return self.replay.replay(self.geometry.layer_id, AttentionInput(row["seq_id"], t["hidden"], t["positions"], mask))

    @property
    def lengths(self):
        return [row["seq_len"] for row in self.rows]

    @property
    def ids(self):
        return [row["seq_id"] for row in self.rows]

    @property
    def device_bytes(self):
        g = self.geometry
        per_token = 4 * (g.kv_lora_rank + g.heads * (g.qk_nope_dim + g.qk_rope_dim + g.v_head_dim) + g.qk_rope_dim + 8192)
        return sum(n * (per_token + n + 8) for n in self.lengths)


class ReplayLayerBank:
    def __init__(self, geometry, replay, splits, provenance):
        self.geometry, self.replay, self.provenance = geometry, replay, provenance
        self.train, self.val, self.test = splits

    @property
    def params(self):
        return self.replay.get(self.geometry.layer_id)[1]

    def split(self, name):
        return getattr(self, name)

    def validate(self):
        self.geometry.validate()
        check_split_description(self.provenance.get("splits"), {name: self.split(name).ids for name in SPLIT_NAMES})


def split_lengths(seqs):
    return seqs.lengths if isinstance(seqs, _MappedSplit) else [s.seq_len for s in seqs]


def load_layer_capture(directory: Path | str, layer_id: int, replay: AttentionReplay):
    """Verify every input receipt before constructing a lazy checkpoint-backed bank."""
    directory = Path(directory)
    manifest = json.loads((directory / f"layer-{layer_id:04d}.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != CAPTURE_SCHEMA:
        raise ValueError(f"unsupported capture schema: {manifest.get('schema')!r}; recapture")
    geometry = MlaLayerGeometry.from_dict(manifest["geometry"])
    if geometry.layer_id != layer_id:
        raise ValueError(f"capture layer_id {geometry.layer_id} != requested {layer_id}")
    for rows in manifest["splits"].values():
        for row in rows:
            payload = row["payload"]
            _verified_mapping(directory / payload["file"], payload["bytes"], payload["xxh3_128"]).close()
    bank = ReplayLayerBank(geometry, replay, tuple(
        _MappedSplit(directory, manifest["splits"][name], geometry, replay) for name in SPLIT_NAMES), manifest["provenance"])
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

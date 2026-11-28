"""Candidate cross-tables, per-width selection, v2 comparison and held-out attention errors.

In QMLA v3, each layer × path has a candidate pool containing:
- ``identity`` (always present; exact native coordinates, zero transform work);
- full fits ``fitted:c{b}`` / ``fitted:r{b}`` (for every width with a full fit, including
  any already-produced 8-bit full fit checkpoints);
- scale-only fits ``scale_fitted:c{b}`` / ``scale_fitted:r{b}`` (the default 8-bit budget);
- step-0 SmoothQuant alpha-grid initializations ``smoothquant:c{b}`` / ``smoothquant:r{b}``.

``measure_candidates`` scores every candidate cell (and every content × RoPE pair on joint
combinations) on one split beside the identity (RTN) and fixed-Hadamard baselines.
``select_per_width`` picks one transform per ``(layer, path, serving_width)`` on ``val``,
classifies each winner's explicit kind (``identity`` | ``scale_only`` | ``full``) from its
actual tensors, and ``select_shared_v2`` computes the old v2 single-transform-per-path
policy on the same ``val`` table for direct comparison on held-out ``test``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
import itertools
import math
from typing import Any, Mapping, Sequence

import torch

from .capture import CapturedLayerBank, DeviceSplit, split_residency_budget, eval_attention_output
from .fit import (
    ContentFitResult,
    FitConfig,
    ROTATION_IDENTITY_TOL,
    RopeFitResult,
    SCALE_IDENTITY_TOL,
    _check_content_equivalence,
    _check_rope_equivalence,
    _init_log_scales,
    content_stats,
    rope_stats,
    apply_content_forward,
    apply_content_scale_only,
    apply_rope_pair_transform,
    apply_rope_scale_only,
    classify_content_kind,
    classify_rope_kind,
    compensate_kv_b,
    compensate_kv_b_scale_only,
)
from .quantizer import qdq_asymmetric_group, sylvester_hadamard

SELECTION_OBJECTIVE = "per_width_mean_val_ratio_to_rtn"
V2_SELECTION_OBJECTIVE = "mean_val_ratio_to_rtn"
HEADLINE_DEPLOYMENT_MODES: tuple[str, ...] = ("4:0", "2:4", "8:0", "8:8", "4:8")
CANDIDATE_KINDS = ("identity", "fitted", "scale_fitted", "smoothquant")
KIND_SIMPLICITY_RANK = {"identity": 0, "scale_only": 1, "full": 2}
NATIVE_ROPE = "native"
_TIE_REL_TOL = 1e-12


def candidate_id(kind: str, path: str | None = None, bits: int | None = None) -> str:
    if kind == "identity":
        return "identity"
    if kind not in CANDIDATE_KINDS or path not in ("content", "rope") or bits is None:
        raise ValueError(f"invalid candidate {kind!r}/{path!r}/{bits!r}")
    return f"{kind}:{'c' if path == 'content' else 'r'}{int(bits)}"


def parse_candidate_id(cid: str) -> tuple[str, str | None, int | None]:
    if cid == "identity":
        return "identity", None, None
    kind, sep, width = cid.partition(":")
    if (
        not sep
        or kind not in ("fitted", "scale_fitted", "smoothquant")
        or len(width) < 2
        or width[0] not in "cr"
        or not width[1:].isdigit()
    ):
        raise ValueError(f"invalid candidate id {cid!r}")
    return kind, ("content" if width[0] == "c" else "rope"), int(width[1:])


def pair_id(content: str | None, rope: str | None) -> str:
    """Key of one table cell: a lone candidate on single-path combinations, else ``c+r``."""
    if content is None and rope is None:
        raise ValueError("a table cell needs at least one candidate")
    if content is None:
        return str(rope)
    if rope is None:
        return str(content)
    return f"{content}+{rope}"


def combo_key(mode: tuple[int, int]) -> str:
    return f"{mode[0]}:{mode[1]}"


@dataclass(frozen=True)
class LayerCandidates:
    """Candidate transforms for one layer, keyed by ``candidate_id`` in deterministic order."""

    content: dict[str, ContentFitResult]
    rope: dict[str, RopeFitResult]


def _content_stats(bank: CapturedLayerBank, device) -> torch.Tensor:
    return content_stats(bank.train, bank.params.to(device), device)


def _rope_stats(bank: CapturedLayerBank, device) -> torch.Tensor:
    return rope_stats(bank.train, bank.geometry, device)


def _init_scale(fit, stats_fn, bank, config) -> torch.Tensor:
    if fit.init_s is not None:
        return fit.init_s.float().cpu()
    return _init_log_scales(stats_fn(bank, "cpu"), fit.init_alpha, config.log_scale_bound).exp()


def _identity_content_result(layer_id: int, d_c: int, protection: str) -> ContentFitResult:
    return ContentFitResult(
        layer_id=layer_id,
        bits=0,
        r_c=torch.eye(d_c, dtype=torch.float32),
        s_c=torch.ones(d_c, dtype=torch.float32),
        best_step=0,
        best_val_loss=0.0,
        init_alpha=0.0,
        equiv_max_rel_err=0.0,
        orth_max_err=0.0,
        baselines={},
        protection_calibrated=protection,
        init_s=torch.ones(d_c, dtype=torch.float32),
        fit_mode="scale_only",
    )


def _identity_rope_result(layer_id: int, pairs: int, protection: str) -> RopeFitResult:
    return RopeFitResult(
        layer_id=layer_id,
        bits=0,
        theta=torch.zeros(pairs, dtype=torch.float32),
        s_p=torch.ones(pairs, dtype=torch.float32),
        best_step=0,
        best_val_loss=0.0,
        init_alpha=0.0,
        equiv_max_rel_err=0.0,
        baselines={},
        protection_calibrated=protection,
        init_s=torch.ones(pairs, dtype=torch.float32),
        fit_mode="scale_only",
    )


def layer_candidates(
    bank: CapturedLayerBank,
    content_fits: Mapping[int, ContentFitResult],
    rope_fits: Mapping[int, RopeFitResult],
    config: FitConfig,
    *,
    content_scale_fits: Mapping[int, ContentFitResult] | None = None,
    rope_scale_fits: Mapping[int, RopeFitResult] | None = None,
) -> LayerCandidates:
    """Build the full candidate pool for one layer.

    Includes:
    - ``identity`` (always present for each active path);
    - full fits ``fitted:c{b}`` / ``fitted:r{b}``;
    - scale-only fits ``scale_fitted:c{b}`` / ``scale_fitted:r{b}``;
    - step-0 SmoothQuant initializations ``smoothquant:c{b}`` / ``smoothquant:r{b}``.
    """
    c_scale = dict(content_scale_fits or {})
    r_scale = dict(rope_scale_fits or {})
    content: dict[str, ContentFitResult] = {}
    rope: dict[str, RopeFitResult] = {}
    d_c = bank.geometry.kv_lora_rank
    lid = bank.geometry.layer_id

    if content_fits or c_scale:
        content["identity"] = _identity_content_result(lid, d_c, config.protection)
        for bits in sorted(content_fits):
            content[candidate_id("fitted", "content", bits)] = content_fits[bits]
        for bits in sorted(c_scale):
            content[candidate_id("scale_fitted", "content", bits)] = c_scale[bits]
        for bits in sorted(set(content_fits) | set(c_scale)):
            fit = content_fits.get(bits) or c_scale[bits]
            content[candidate_id("smoothquant", "content", bits)] = replace(
                fit,
                r_c=torch.eye(d_c, dtype=torch.float32),
                s_c=_init_scale(fit, _content_stats, bank, config),
                best_step=0,
                best_val_loss=fit.baselines["smoothquant_alpha_val"],
                equiv_max_rel_err=math.nan,
                orth_max_err=0.0,
                fit_mode="scale_only",
            )

    if (rope_fits or r_scale) and bank.geometry.has_rope:
        pairs = bank.geometry.qk_rope_dim // 2
        rope["identity"] = _identity_rope_result(lid, pairs, config.protection)
        for bits in sorted(rope_fits):
            rope[candidate_id("fitted", "rope", bits)] = rope_fits[bits]
        for bits in sorted(r_scale):
            rope[candidate_id("scale_fitted", "rope", bits)] = r_scale[bits]
        for bits in sorted(set(rope_fits) | set(r_scale)):
            fit = rope_fits.get(bits) or r_scale[bits]
            scale = _init_scale(fit, _rope_stats, bank, config)
            rope[candidate_id("smoothquant", "rope", bits)] = replace(
                fit,
                theta=torch.zeros_like(scale),
                s_p=scale,
                best_step=0,
                best_val_loss=fit.baselines["smoothquant_alpha_val"],
                equiv_max_rel_err=math.nan,
                fit_mode="scale_only",
            )
    return LayerCandidates(content=content, rope=rope)


def candidate_kind(path: str, cid: str, result: Any) -> str:
    """Classify a candidate's actual transform kind from its tensors (never blindly by name)."""
    if result is None:
        # Lightweight test stub fallback when tensors are absent.
        ckind, _, _ = parse_candidate_id(cid)
        if ckind == "identity":
            return "identity"
        if ckind in ("scale_fitted", "smoothquant"):
            return "scale_only"
        return "full"
    if hasattr(result, "kind"):
        return str(result.kind)
    if path == "content":
        return classify_content_kind(result.r_c, result.s_c, rot_tol=ROTATION_IDENTITY_TOL, scale_tol=SCALE_IDENTITY_TOL)
    return classify_rope_kind(result.theta, result.s_p, rot_tol=ROTATION_IDENTITY_TOL, scale_tol=SCALE_IDENTITY_TOL)


def with_equivalence(result, bank: CapturedLayerBank, device):
    """Measure quantization-off equivalence for a candidate that has not been checked."""
    if math.isfinite(result.equiv_max_rel_err):
        return result
    seqs = DeviceSplit(bank.val, device, budget_bytes=split_residency_budget(device))
    if isinstance(result, ContentFitResult):
        kind = classify_content_kind(result.r_c, result.s_c)
        if kind == "identity":
            return replace(result, equiv_max_rel_err=0.0, orth_max_err=0.0)
        err = _check_content_equivalence(
            seqs, bank.params.to(device), bank.geometry, result.r_c.to(device), result.s_c.to(device)
        )
    else:
        kind = classify_rope_kind(result.theta, result.s_p)
        if kind == "identity":
            return replace(result, equiv_max_rel_err=0.0)
        err = _check_rope_equivalence(seqs, bank.geometry, result.theta.to(device), result.s_p.to(device))
    return replace(result, equiv_max_rel_err=err)


def _quantized(transform, bits: int, scale_dtype: str, config: FitConfig):
    def apply(x):
        return qdq_asymmetric_group(
            transform(x), bits, group_size=config.group_size, scale_dtype=scale_dtype,
            eps=config.eps_scale, ste=False,
        )

    return apply


def _token_relative_squares(pred, ref):
    """Each held-out token has equal weight, irrespective of output norm or sequence length."""
    energy = ref.float().square().sum(dim=-1)
    return (pred.float() - ref.float()).square().sum(dim=-1) / energy


class _Scorer:
    """Score one layer's split under arbitrary content/RoPE arms with kind-specific execution."""

    def __init__(
        self,
        bank: CapturedLayerBank | None,
        split: str | None,
        config: FitConfig,
        device,
        *,
        geom=None,
        params=None,
        seqs=(),
    ):
        if bank is not None:
            geom = bank.geometry
            params = bank.params
            seqs = bank.split(split)
        self.geom = geom
        self.params = params.to(device)
        self.device = device
        self.config = config
        self.seqs = DeviceSplit(seqs, device, budget_bytes=split_residency_budget(device))
        self.tokens = self.seqs.tokens
        self.protection = (
            {"sink_tokens": config.sink_tokens, "recent_tokens": config.recent_tokens}
            if config.protection == "sink4_local128" else {}
        )
        d_c = self.geom.kv_lora_rank
        self.hadamard = sylvester_hadamard(d_c, device=device) if (d_c & (d_c - 1)) == 0 else None
        self._kv_b: dict[str, torch.Tensor] = {}

    def content_hooks(self, bits: int, arm) -> dict[str, Any]:
        """``arm`` is ``"rtn"``, ``"hadamard"``, or ``(cache_key, ContentFitResult | ResolvedContentTransform)``."""
        if not bits:
            return {}
        params = self.params
        hooks: dict[str, Any] = {}
        if arm == "rtn":
            exact = lambda c: c
        elif arm == "hadamard":
            if self.hadamard is None:
                raise ValueError("fixed Hadamard needs a power-of-two kv_lora_rank")
            key, rotation, scale = "hadamard", self.hadamard, torch.ones(self.geom.kv_lora_rank, device=self.device)
            exact = partial(apply_content_forward, r_c=rotation, s_c=scale)
            if key not in self._kv_b:
                self._kv_b[key] = compensate_kv_b(params.kv_b_weight, rotation, scale)
            hooks["kv_b_override"] = self._kv_b[key]
        else:
            key, result = arm
            kind = (
                result.kind
                if hasattr(result, "kind")
                else classify_content_kind(result.r_c, result.s_c)
            )
            if kind == "identity":
                exact = lambda c: c
            elif kind == "scale_only":
                scale = result.s_c.to(self.device)
                exact = partial(apply_content_scale_only, s_c=scale)
                if key not in self._kv_b:
                    self._kv_b[key] = compensate_kv_b_scale_only(params.kv_b_weight, scale)
                hooks["kv_b_override"] = self._kv_b[key]
            else:
                rotation, scale = result.r_c.to(self.device), result.s_c.to(self.device)
                exact = partial(apply_content_forward, r_c=rotation, s_c=scale)
                if key not in self._kv_b:
                    self._kv_b[key] = compensate_kv_b(params.kv_b_weight, rotation, scale)
                hooks["kv_b_override"] = self._kv_b[key]
        hooks["content_fn"] = _quantized(exact, bits, "BF16", self.config)
        if self.protection:
            hooks["exact_content_fn"] = exact
        return hooks

    def rope_hooks(self, bits: int, arm) -> dict[str, Any]:
        """``arm`` is ``NATIVE_ROPE`` or a ``RopeFitResult`` / ``ResolvedRopeTransform``."""
        if not bits or not self.geom.has_rope:
            return {}
        if arm == NATIVE_ROPE:
            kind = "identity"
        else:
            kind = (
                arm.kind
                if hasattr(arm, "kind")
                else classify_rope_kind(arm.theta, arm.s_p)
            )
        if kind == "identity":
            exact = lambda x: x
            hooks: dict[str, Any] = {}
        elif kind == "scale_only":
            scale = arm.s_p.to(self.device)
            exact = partial(apply_rope_scale_only, scale=scale, rope_kind=self.geom.rope_kind)
            hooks = {"rope_query_fn": partial(exact, inverse_scale=True)}
        else:
            theta, scale = arm.theta.to(self.device), arm.s_p.to(self.device)
            exact = partial(apply_rope_pair_transform, theta=theta, scale=scale, rope_kind=self.geom.rope_kind)
            hooks = {"rope_query_fn": partial(exact, inverse_scale=True)}
        hooks["rope_key_fn"] = _quantized(exact, bits, "F32", self.config)
        if self.protection:
            hooks["exact_rope_key_fn"] = exact
        return hooks

    def score(self, mode: tuple[int, int], content_arm, rope_arm) -> float:
        cb, rb = mode
        hooks = {**self.content_hooks(cb, content_arm), **self.rope_hooks(rb, rope_arm), **self.protection}
        sums = []
        for seq in self.seqs:
            pred = eval_attention_output(seq, self.params, self.geom, **hooks)
            sums.append(_token_relative_squares(pred, seq.ref_output).sum(dtype=torch.float64))
        mean = torch.stack(sums).sum().item() / self.tokens
        if not math.isfinite(mean):
            raise ValueError(f"invalid attention error at layer {self.geom.layer_id} for {combo_key(mode)}")
        return mean


def layer_quantizes(geometry, mode: tuple[int, int]) -> bool:
    """Whether a combination quantizes anything at this layer (RoPE needs a RoPE path)."""
    return bool(mode[0]) or (bool(mode[1]) and geometry.has_rope)


def table_cells(
    candidates: LayerCandidates, geometry, mode: tuple[int, int]
) -> list[tuple[str | None, str | None]]:
    """(content id, RoPE id) cells a combination is scored on at this layer."""
    cb, rb = mode
    rope_on = bool(rb) and geometry.has_rope
    content_ids: list[str | None] = list(candidates.content) if cb else [None]
    rope_ids: list[str | None] = list(candidates.rope) if rope_on else [None]
    if cb and not content_ids:
        raise ValueError(f"combination {combo_key(mode)} needs a content candidate")
    if rope_on and not rope_ids:
        raise ValueError(f"combination {combo_key(mode)} needs a RoPE candidate")
    return [(c, r) for c in content_ids for r in rope_ids]


@torch.no_grad()
def measure_candidates(
    bank: CapturedLayerBank,
    candidates: LayerCandidates,
    modes: Sequence[tuple[int, int]],
    config: FitConfig,
    device: torch.device | str,
    *,
    split: str,
) -> dict[str, dict[str, Any]]:
    """Score every candidate cell plus RTN (identity) and fixed Hadamard on ``split``."""
    scorer = _Scorer(bank, split, config, device)
    table: dict[str, dict[str, Any]] = {}
    jobs = []
    for mode in modes:
        if not layer_quantizes(bank.geometry, mode):
            continue
        cb, rb = mode
        rope_on = bool(rb) and bank.geometry.has_rope
        row = {"tokens": scorer.tokens, "cells": {}}
        table[combo_key(mode)] = row
        jobs.append((row, "rtn", mode, "rtn", NATIVE_ROPE))
        if cb:
            jobs.append((row, "hadamard", mode, "hadamard", NATIVE_ROPE))
        for c, r in table_cells(candidates, bank.geometry, mode):
            if c in (None, "identity") and r in (None, "identity"):
                continue
            content_arm = (c, candidates.content[c]) if c is not None else "rtn"
            rope_arm = candidates.rope[r] if r is not None and rope_on else NATIVE_ROPE
            jobs.append((row["cells"], pair_id(c, r), mode, content_arm, rope_arm))
    sums = [[] for _ in jobs]
    for seq in scorer.seqs:
        for acc, (_, _, (cb, rb), c, r) in zip(sums, jobs, strict=True):
            hooks = {**scorer.content_hooks(cb, c), **scorer.rope_hooks(rb, r), **scorer.protection}
            pred = eval_attention_output(seq, scorer.params, scorer.geom, **hooks)
            acc.append(_token_relative_squares(pred, seq.ref_output).sum(dtype=torch.float64))
    for acc, (row, key, *_rest) in zip(sums, jobs, strict=True):
        value = torch.stack(acc).sum().item() / scorer.tokens
        if not math.isfinite(value):
            raise ValueError(f"invalid candidate attention error at layer {bank.geometry.layer_id}")
        row[key] = value
    for mode in modes:
        if combo_key(mode) not in table:
            continue
        row = table[combo_key(mode)]
        row.setdefault("hadamard", row["rtn"])
        for c, r in table_cells(candidates, bank.geometry, mode):
            if c in (None, "identity") and r in (None, "identity"):
                row["cells"][pair_id(c, r)] = row["rtn"]
        # Keep the original canonical candidate enumeration order.
        row["cells"] = {pair_id(c, r): row["cells"][pair_id(c, r)] for c, r in table_cells(candidates, bank.geometry, mode)}
    return table


def _cell(table_row: Mapping[str, Any], c: str | None, r: str | None) -> float:
    return float(table_row["cells"][pair_id(c, r)])


def _mode_of(key: str) -> tuple[int, int]:
    c, r = key.split(":")
    return int(c), int(r)


def _is_strictly_better(val: float, tie: tuple[Any, ...], best_val: float, best_tie: tuple[Any, ...]) -> bool:
    tol = _TIE_REL_TOL * max(1.0, abs(best_val))
    if val < best_val - tol:
        return True
    if abs(val - best_val) <= tol and tie < best_tie:
        return True
    return False


def select_per_width(
    val_table: Mapping[str, Mapping[str, Any]],
    candidates: LayerCandidates,
    geometry,
) -> dict[str, Any]:
    """Select one transform per ``(path, serving_width)`` minimizing mean val ratio to RTN.

    Because content width selections ``c_w`` are conditionally independent once the RoPE
    width selections ``(r_4, r_8)`` are fixed, exact global minimization over all serving
    widths runs in milliseconds while preserving joint content × RoPE interaction checks.

    Ties (within ``1e-12`` relative tolerance) prefer simpler transform kinds
    (``identity < scale_only < full``), then natural training width (``source_bits == width``),
    then lexicographic candidate ID.
    """
    if not val_table:
        raise ValueError(f"layer {geometry.layer_id} has no measured combination to select on")

    mode_rows = [(_mode_of(k), k, row) for k, row in sorted(val_table.items())]
    for (_, key, row) in mode_rows:
        if not row["rtn"] > 0.0:
            raise ValueError(f"layer {geometry.layer_id} {key} RTN val error is not positive")

    content_bits = sorted({cb for (cb, _), _, _ in mode_rows if cb > 0})
    rope_bits = sorted({rb for (_, rb), _, _ in mode_rows if rb > 0}) if geometry.has_rope else []
    content_ids = list(candidates.content) if content_bits else []
    rope_ids = list(candidates.rope) if rope_bits else []
    if content_bits and not content_ids:
        raise ValueError(f"layer {geometry.layer_id} needs content candidates for widths {content_bits}")
    if rope_bits and not rope_ids:
        raise ValueError(f"layer {geometry.layer_id} needs RoPE candidates for widths {rope_bits}")

    c_kinds = {cid: candidate_kind("content", cid, candidates.content[cid]) for cid in content_ids}
    r_kinds = {rid: candidate_kind("rope", rid, candidates.rope[rid]) for rid in rope_ids}
    c_src_bits = {cid: parse_candidate_id(cid)[2] for cid in content_ids}
    r_src_bits = {rid: parse_candidate_id(rid)[2] for rid in rope_ids}

    modes_by_cb: dict[int, list[tuple[tuple[int, int], str, Mapping[str, Any]]]] = {cb: [] for cb in content_bits}
    rope_only_modes: list[tuple[tuple[int, int], str, Mapping[str, Any]]] = []
    for entry in mode_rows:
        (cb, rb), _, _ = entry
        if cb > 0:
            modes_by_cb[cb].append(entry)
        elif rb > 0 and geometry.has_rope:
            rope_only_modes.append(entry)

    rope_combos = list(itertools.product(rope_ids, repeat=len(rope_bits))) if rope_bits else [()]
    best_obj = float("inf")
    best_tie: tuple[Any, ...] = ()
    best_c_map: dict[int, str] = {}
    best_r_map: dict[int, str] = {}
    n_modes = len(mode_rows)

    for r_tuple in rope_combos:
        r_map = dict(zip(rope_bits, r_tuple))
        total_ratio = 0.0
        for (_, rb), _, row in rope_only_modes:
            total_ratio += _cell(row, None, r_map[rb]) / row["rtn"]

        c_map: dict[int, str] = {}
        for cb in content_bits:
            cb_entries = modes_by_cb[cb]
            best_c_sum = float("inf")
            best_c_tie: tuple[Any, ...] = ()
            best_cid = content_ids[0]
            for cid in content_ids:
                s = 0.0
                for (_, rb), _, row in cb_entries:
                    r_pick = r_map[rb] if (rb > 0 and geometry.has_rope) else None
                    s += _cell(row, cid, r_pick) / row["rtn"]
                c_tie = (
                    KIND_SIMPLICITY_RANK[c_kinds[cid]],
                    0 if cid == "identity" else 1,
                    0 if c_src_bits[cid] == cb else 1,
                    cid,
                )
                if not math.isfinite(best_c_sum) or _is_strictly_better(s, c_tie, best_c_sum, best_c_tie):
                    best_c_sum = s
                    best_c_tie = c_tie
                    best_cid = cid
            c_map[cb] = best_cid
            total_ratio += best_c_sum

        obj = total_ratio / n_modes
        total_rank = sum(KIND_SIMPLICITY_RANK[c_kinds[c_map[cb]]] for cb in content_bits) + sum(
            KIND_SIMPLICITY_RANK[r_kinds[r_map[rb]]] for rb in rope_bits
        )
        non_explicit_id = sum(0 if c_map[cb] == "identity" else 1 for cb in content_bits) + sum(
            0 if r_map[rb] == "identity" else 1 for rb in rope_bits
        )
        mismatches = sum(0 if c_src_bits[c_map[cb]] == cb else 1 for cb in content_bits) + sum(
            0 if r_src_bits[r_map[rb]] == rb else 1 for rb in rope_bits
        )
        tie = (
            total_rank,
            non_explicit_id,
            mismatches,
            tuple(c_map[cb] for cb in content_bits),
            tuple(r_map[rb] for rb in rope_bits),
        )
        if not math.isfinite(best_obj) or _is_strictly_better(obj, tie, best_obj, best_tie):
            best_obj = obj
            best_tie = tie
            best_c_map = c_map
            best_r_map = r_map

    content_out: dict[str, Any] | None = None
    if content_bits:
        content_out = {}
        for cb in content_bits:
            cid = best_c_map[cb]
            scores = {}
            ratios = []
            for (_, rb), key, row in modes_by_cb[cb]:
                r_pick = best_r_map[rb] if (rb > 0 and geometry.has_rope) else None
                val_err = _cell(row, cid, r_pick)
                scores[key] = val_err
                ratios.append(val_err / row["rtn"])
            content_out[str(cb)] = {
                "candidate": cid,
                "kind": c_kinds[cid],
                "source_bits": c_src_bits[cid],
                "val_score": scores,
                "width_mean_val_ratio_to_rtn": sum(ratios) / len(ratios),
            }

    rope_out: dict[str, Any] | None = None
    if rope_bits:
        rope_out = {}
        for rb in rope_bits:
            rid = best_r_map[rb]
            scores = {}
            ratios = []
            for (cb, m_rb), key, row in mode_rows:
                if m_rb != rb:
                    continue
                c_pick = best_c_map[cb] if cb > 0 else None
                val_err = _cell(row, c_pick, rid)
                scores[key] = val_err
                ratios.append(val_err / row["rtn"])
            rope_out[str(rb)] = {
                "candidate": rid,
                "kind": r_kinds[rid],
                "source_bits": r_src_bits[rid],
                "val_score": scores,
                "width_mean_val_ratio_to_rtn": sum(ratios) / len(ratios),
            }

    return {
        "objective": SELECTION_OBJECTIVE,
        "overall_objective": best_obj,
        "content": content_out,
        "rope": rope_out,
    }


def select_shared_v2(
    val_table: Mapping[str, Mapping[str, Any]],
    candidates: LayerCandidates,
    geometry,
) -> dict[str, Any]:
    """Legacy v2 single-transform-per-path selector, kept strictly as a report comparison baseline."""
    if not val_table:
        raise ValueError(f"layer {geometry.layer_id} has no measured combination to select on")
    uses_content = any(_mode_of(k)[0] for k in val_table)
    uses_rope = any(_mode_of(k)[1] for k in val_table) and geometry.has_rope
    content_ids: list[str | None] = list(candidates.content) if uses_content else [None]
    rope_ids: list[str | None] = list(candidates.rope) if uses_rope else [None]
    best_obj = float("inf")
    best_tie: tuple[Any, ...] = ()
    c_best: str | None = None
    r_best: str | None = None
    objectives: dict[str, float] = {}
    for c in content_ids:
        for r in rope_ids:
            ratios = []
            for key, row in val_table.items():
                cb, rb = _mode_of(key)
                cell_c = c if cb else None
                cell_r = r if rb and geometry.has_rope else None
                if not row["rtn"] > 0.0:
                    raise ValueError(f"layer {geometry.layer_id} {key} RTN val error is not positive")
                ratios.append(_cell(row, cell_c, cell_r) / row["rtn"])
            objective = sum(ratios) / len(ratios)
            objectives[pair_id(c, r)] = objective
            c_rank = KIND_SIMPLICITY_RANK[candidate_kind("content", c, candidates.content[c])] if c is not None else 0
            r_rank = KIND_SIMPLICITY_RANK[candidate_kind("rope", r, candidates.rope[r])] if r is not None else 0
            tie = (c_rank + r_rank, str(c or ""), str(r or ""))
            if not math.isfinite(best_obj) or _is_strictly_better(objective, tie, best_obj, best_tie):
                best_obj = objective
                best_tie = tie
                c_best, r_best = c, r

    def path_record(path: str, cid: str | None) -> dict[str, Any] | None:
        if cid is None:
            return None
        ckind, _, bits = parse_candidate_id(cid)
        pool = candidates.content if path == "content" else candidates.rope
        val_score = {}
        for key, row in val_table.items():
            cb, rb = _mode_of(key)
            if (path == "content" and cb) or (path == "rope" and rb):
                val_score[key] = _cell(row, c_best if cb else None, r_best if rb and geometry.has_rope else None)
        return {
            "candidate": cid,
            "chosen": ckind,
            "kind": candidate_kind(path, cid, pool[cid]),
            "source_bits": bits,
            "objective": best_obj,
            "val_score": val_score,
        }

    return {
        "objective": V2_SELECTION_OBJECTIVE,
        "content": path_record("content", c_best),
        "rope": path_record("rope", r_best),
        "pair_objectives": objectives,
    }


def layer_rows(
    test_table: Mapping[str, Mapping[str, Any]],
    val_table: Mapping[str, Mapping[str, Any]],
    selection: Mapping[str, Any],
    geometry,
    *,
    v2_selection: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Per-combination held-out test rows for one layer: v3 shipped, v2 shared, identity, Hadamard."""
    rows = {}
    for key, row in test_table.items():
        cb, rb = _mode_of(key)
        rope_on = bool(rb) and geometry.has_rope
        c_sel = selection["content"][str(cb)] if cb else None
        r_sel = selection["rope"][str(rb)] if rope_on else None
        shipped = (c_sel["candidate"] if c_sel else None, r_sel["candidate"] if r_sel else None)
        if v2_selection is not None:
            v2_c = v2_selection["content"]["candidate"] if (cb and v2_selection["content"]) else None
            v2_r = v2_selection["rope"]["candidate"] if (rope_on and v2_selection["rope"]) else None
            v2_cell = (v2_c, v2_r)
            v2_err = _cell(row, *v2_cell)
            v2_cell_id = pair_id(*v2_cell)
        else:
            v2_err = _cell(row, *shipped)
            v2_cell_id = pair_id(*shipped)

        rows[key] = {
            "tokens": row["tokens"],
            "attn_rel_err": _cell(row, *shipped),
            "v2_shared_attn_rel_err": v2_err,
            "rtn_attn_rel_err": row["rtn"],
            "hadamard_attn_rel_err": row["hadamard"],
            "shipped_cell": pair_id(*shipped),
            "shipped_kinds": {
                "content": c_sel["kind"] if c_sel else None,
                "rope": r_sel["kind"] if r_sel else None,
            },
            "v2_shared_cell": v2_cell_id,
            "val_cells": dict(val_table[key]["cells"]),
            "cells": dict(row["cells"]),
        }
    return rows


REPORTED_ERRORS = (
    "attn_rel_err",
    "v2_shared_attn_rel_err",
    "rtn_attn_rel_err",
    "hadamard_attn_rel_err",
)


def summarize_modes(by_layer: Mapping[Any, Mapping[str, Mapping[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Token-weighted test summaries per combination over layers, plus the full candidate cross-table."""
    summaries: dict[str, dict[str, Any]] = {}
    for layer, modes in by_layer.items():
        for key, row in modes.items():
            summary = summaries.setdefault(key, {"per_layer": {}})
            summary["per_layer"][str(layer)] = row
    for summary in summaries.values():
        rows = list(summary["per_layer"].values())
        tokens = sum(row["tokens"] for row in rows)
        summary["layer_tokens"] = tokens
        for field in REPORTED_ERRORS:
            summary[field] = sum(row[field] * row["tokens"] for row in rows) / tokens
        rtn = summary["rtn_attn_rel_err"]
        v2 = summary["v2_shared_attn_rel_err"]
        summary["ratio_to_rtn"] = summary["attn_rel_err"] / rtn if rtn else None
        summary["v2_shared_ratio_to_rtn"] = v2 / rtn if rtn else None
        summary["ratio_to_v2_shared"] = summary["attn_rel_err"] / v2 if v2 else None
        summary["hadamard_ratio_to_rtn"] = summary["hadamard_attn_rel_err"] / rtn if rtn else None
        summary["layers"] = len(rows)

        cross: dict[str, dict[str, Any]] = {}
        for row in rows:
            for cell, err in row["cells"].items():
                entry = cross.setdefault(cell, {"weighted": 0.0, "tokens": 0, "layers": 0})
                entry["weighted"] += err * row["tokens"]
                entry["tokens"] += row["tokens"]
                entry["layers"] += 1
        summary["cross_table"] = {
            cell: {
                "attn_rel_err": e["weighted"] / e["tokens"],
                "ratio_to_rtn": (e["weighted"] / e["tokens"]) / rtn if rtn else None,
                "layers": e["layers"],
            }
            for cell, e in sorted(cross.items())
        }
    return summaries


def shipped_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """The per-combination held-out test numbers carried in the sidecar manifest."""
    return {
        "layer_tokens": summary["layer_tokens"],
        "attn_rel_err": summary["attn_rel_err"],
        "rtn_attn_rel_err": summary["rtn_attn_rel_err"],
        "v2_shared_attn_rel_err": summary["v2_shared_attn_rel_err"],
        "hadamard_attn_rel_err": summary["hadamard_attn_rel_err"],
        "ratio_to_rtn": summary["ratio_to_rtn"],
        "v2_shared_ratio_to_rtn": summary["v2_shared_ratio_to_rtn"],
        "ratio_to_v2_shared": summary["ratio_to_v2_shared"],
    }


def _layer_spread(per_layer: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    """Compute per-layer median, best (lowest ratio), and worst (highest ratio) improvement vs identity."""
    entries: list[tuple[float, int, float, float]] = []
    for lid_str, row in per_layer.items():
        rtn = float(row["rtn_attn_rel_err"])
        err = float(row["attn_rel_err"])
        if rtn > 0.0:
            entries.append((err / rtn, int(lid_str), err, rtn))
    if not entries:
        return None
    entries.sort(key=lambda item: (item[0], item[1]))
    n = len(entries)
    mid = n // 2
    med_ratio = entries[mid][0] if (n % 2 == 1) else 0.5 * (entries[mid - 1][0] + entries[mid][0])
    best_r, best_lid, best_err, best_rtn = entries[0]
    worst_r, worst_lid, worst_err, worst_rtn = entries[-1]
    return {
        "layers": n,
        "median": {
            "ratio_to_rtn": med_ratio,
            "error_reduction_pct": (1.0 - med_ratio) * 100.0,
        },
        "best": {
            "layer_id": best_lid,
            "ratio_to_rtn": best_r,
            "error_reduction_pct": (1.0 - best_r) * 100.0,
            "attn_rel_err": best_err,
            "rtn_attn_rel_err": best_rtn,
        },
        "worst": {
            "layer_id": worst_lid,
            "ratio_to_rtn": worst_r,
            "error_reduction_pct": (1.0 - worst_r) * 100.0,
            "attn_rel_err": worst_err,
            "rtn_attn_rel_err": worst_rtn,
        },
    }


def _unsupported_reason_for_mode(
    mode: str,
    unsupported_reasons: Mapping[str, str] | None,
    profiles: Mapping[str, Mapping[str, Any] | None] | None,
) -> str:
    c_bits, r_bits = _mode_of(mode)
    reasons = unsupported_reasons or {}
    if mode in reasons:
        return str(reasons[mode])
    if r_bits > 0 and f"*:{r_bits}" in reasons:
        return str(reasons[f"*:{r_bits}"])
    if r_bits > 0 and profiles is not None and profiles.get("rope") is None:
        return "model MLA layers have no RoPE key cache (qk_rope_dim=0)"
    return "mode not evaluated in this run"


def _short_or_long_block(entry: Mapping[str, Any]) -> dict[str, Any]:
    ratio_id = entry.get("ratio_to_rtn")
    per_layer = entry.get("per_layer")
    return {
        "layer_tokens": entry.get("layer_tokens", entry.get("tokens")),
        "rtn_attn_rel_err": entry["rtn_attn_rel_err"],
        "attn_rel_err": entry["attn_rel_err"],
        "ratio_to_rtn": ratio_id,
        "error_reduction_pct": None if ratio_id is None else (1.0 - ratio_id) * 100.0,
        "per_layer_spread": _layer_spread(per_layer) if isinstance(per_layer, Mapping) else None,
    }


def build_headline_table(
    combinations: Mapping[str, Mapping[str, Any]],
    profiles: Mapping[str, Mapping[str, Any] | None] | None = None,
    *,
    long_combinations: Mapping[str, Mapping[str, Any]] | None = None,
    modes: Sequence[str] = HEADLINE_DEPLOYMENT_MODES,
    unsupported_reasons: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Build the leading headline table for deployment modes ``4:0, 2:4, 8:0, 8:8, 4:8``.

    Shows short-context (held-out test) and long-context (full-length long-eval overall)
    side by side with the long-to-short improvement ratio when ``long_combinations`` is
    supplied. Every supported entry comes from ONE end-to-end joint post-``o_proj``
    measurement combining each layer's selected content and RoPE transforms (never
    per-path errors summed). Unsupported/NoPE modes are emitted explicitly as ``n/a``.
    """
    rows: list[dict[str, Any]] = []
    c_prof = profiles.get("content") if profiles else None
    r_prof = profiles.get("rope") if profiles else None
    for mode in modes:
        c_bits, r_bits = _mode_of(mode)
        if mode not in combinations:
            rows.append({
                "mode": mode,
                "supported": False,
                "status": "n/a",
                "reason": _unsupported_reason_for_mode(mode, unsupported_reasons, profiles),
                "short": None,
                "long": None,
                "long_to_short_improvement_ratio": None,
                "layer_tokens": None,
                "rtn_attn_rel_err": None,
                "attn_rel_err": None,
                "ratio_to_rtn": None,
                "error_reduction_vs_identity_pct": None,
                "v2_shared_attn_rel_err": None,
                "ratio_to_v2_shared": None,
                "error_reduction_vs_v2_shared_pct": None,
                "per_layer_spread": None,
                "kind_counts": {"content": None, "rope": None},
            })
            continue
        entry = combinations[mode]
        short_block = _short_or_long_block(entry)
        long_block = (
            _short_or_long_block(long_combinations[mode])
            if (long_combinations is not None and mode in long_combinations)
            else None
        )
        short_imp = short_block["error_reduction_pct"]
        long_imp = None if long_block is None else long_block["error_reduction_pct"]
        if short_imp is not None and long_imp is not None and abs(short_imp) > 1e-12:
            l2s_ratio: float | None = long_imp / short_imp
        else:
            l2s_ratio = None
        ratio_id = short_block["ratio_to_rtn"]
        ratio_v2 = entry.get("ratio_to_v2_shared")
        rows.append({
            "mode": mode,
            "supported": True,
            "status": "measured",
            "reason": None,
            "short": short_block,
            "long": long_block,
            "long_to_short_improvement_ratio": l2s_ratio,
            "layer_tokens": short_block["layer_tokens"],
            "rtn_attn_rel_err": short_block["rtn_attn_rel_err"],
            "attn_rel_err": short_block["attn_rel_err"],
            "ratio_to_rtn": ratio_id,
            "error_reduction_vs_identity_pct": short_imp,
            "v2_shared_attn_rel_err": entry.get("v2_shared_attn_rel_err"),
            "ratio_to_v2_shared": ratio_v2,
            "error_reduction_vs_v2_shared_pct": None if ratio_v2 is None else (1.0 - ratio_v2) * 100.0,
            "per_layer_spread": short_block["per_layer_spread"],
            "kind_counts": {
                "content": (
                    dict(c_prof["widths"][str(c_bits)]["kind_counts"])
                    if (c_bits > 0 and c_prof is not None and str(c_bits) in c_prof.get("widths", {})) else None
                ),
                "rope": (
                    dict(r_prof["widths"][str(r_bits)]["kind_counts"])
                    if (r_bits > 0 and r_prof is not None and str(r_bits) in r_prof.get("widths", {})) else None
                ),
            },
        })
    return rows


HEADLINE_TABLE_CAPTION = (
    "*Joint end-to-end post-`o_proj` attention error combining each layer's selected content "
    "transform at the content width with its selected RoPE transform at the RoPE width (never "
    "per-path numbers added together). Short = held-out test split; Long = full-length long-eval "
    "(overall); Long/Short = ratio of long improvement % to short improvement %. `:0` means RoPE "
    "is kept unquantized (native format).*"
)


def _format_spread(spread: Mapping[str, Any] | None) -> str:
    if spread is None:
        return "—"
    med = f"{spread['median']['ratio_to_rtn']:.3f}x ({spread['median']['error_reduction_pct']:+.1f}%)"
    best = (
        f"L{spread['best']['layer_id']} {spread['best']['ratio_to_rtn']:.3f}x "
        f"({spread['best']['error_reduction_pct']:+.1f}%)"
    )
    worst = (
        f"L{spread['worst']['layer_id']} {spread['worst']['ratio_to_rtn']:.3f}x "
        f"({spread['worst']['error_reduction_pct']:+.1f}%)"
    )
    return f"{med} / {best} / {worst}"


def format_headline_table(rows: Sequence[Mapping[str, Any]], *, caption: str = HEADLINE_TABLE_CAPTION) -> str:
    """Render the headline table as GitHub markdown (combined short vs long when long is present)."""
    has_long = any(row.get("long") is not None for row in rows)
    if has_long:
        lines = [
            "| Mode (C:R) | Short Identity | Short Shipped | Short Impr. | Long Identity | Long Shipped | Long Impr. | Long/Short Impr. | Short Spread (med / best / worst) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in rows:
            if not row.get("supported", True):
                reason = row.get("reason") or "not supported"
                lines.append(f"| `{row['mode']}` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a ({reason}) |")
                continue
            s = row.get("short") or row
            l = row.get("long")
            s_imp = s.get("error_reduction_pct", row.get("error_reduction_vs_identity_pct"))
            s_imp_s = "—" if s_imp is None else f"{s_imp:+.1f}%"
            if l is not None:
                l_id_s = f"{l['rtn_attn_rel_err']:.6g}"
                l_sh_s = f"{l['attn_rel_err']:.6g}"
                l_imp = l.get("error_reduction_pct")
                l_imp_s = "—" if l_imp is None else f"{l_imp:+.1f}%"
            else:
                l_id_s = l_sh_s = l_imp_s = "—"
            l2s = row.get("long_to_short_improvement_ratio")
            l2s_s = "—" if l2s is None else f"{l2s:.2f}x"
            lines.append(
                f"| `{row['mode']}` | {s['rtn_attn_rel_err']:.6g} | {s['attn_rel_err']:.6g} | {s_imp_s} | "
                f"{l_id_s} | {l_sh_s} | {l_imp_s} | {l2s_s} | {_format_spread(s.get('per_layer_spread'))} |"
            )
    else:
        lines = [
            "| Mode (C:R) | Identity (RTN) | Shipped (v3) | Ratio | Reduction | Per-layer spread (median / best / worst) |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for row in rows:
            if not row.get("supported", True):
                reason = row.get("reason") or "not supported"
                lines.append(f"| `{row['mode']}` | n/a | n/a | n/a | n/a | n/a ({reason}) |")
                continue
            r_id = "—" if row["ratio_to_rtn"] is None else f"{row['ratio_to_rtn']:.4f}x"
            red = "—" if row["error_reduction_vs_identity_pct"] is None else f"{row['error_reduction_vs_identity_pct']:+.1f}%"
            lines.append(
                f"| `{row['mode']}` | {row['rtn_attn_rel_err']:.6g} | {row['attn_rel_err']:.6g} | "
                f"{r_id} | {red} | {_format_spread(row.get('per_layer_spread'))} |"
            )
    if caption:
        lines.append("")
        lines.append(caption)
    return "\n".join(lines)


@torch.no_grad()
def measure_shipped(
    bank: CapturedLayerBank,
    content_by_width: Mapping[int, Any] | None,
    rope_by_width: Mapping[int, Any] | None,
    modes: Sequence[tuple[int, int]],
    config: FitConfig,
    device: torch.device | str,
) -> dict[str, dict[str, Any]]:
    """Score per-width shipped transforms vs identity (RTN) and Hadamard on ``bank.test``."""
    scorer = _Scorer(bank, "test", config, device)
    c_map = content_by_width or {}
    r_map = rope_by_width or {}
    rows = {}
    for mode in modes:
        if not layer_quantizes(bank.geometry, mode):
            continue
        cb, rb = mode
        rope_on = bool(rb) and bank.geometry.has_rope
        if cb and cb not in c_map:
            raise ValueError(f"combination {combo_key(mode)} missing shipped content transform for width {cb}")
        if rope_on and rb not in r_map:
            raise ValueError(f"combination {combo_key(mode)} missing shipped RoPE transform for width {rb}")
        c_arm = (f"shipped_c{cb}", c_map[cb]) if cb else "rtn"
        r_arm = r_map[rb] if rope_on else NATIVE_ROPE
        fitted = scorer.score(mode, c_arm, r_arm)
        rtn = scorer.score(mode, "rtn", NATIVE_ROPE)
        rows[combo_key(mode)] = {
            "tokens": scorer.tokens,
            "attn_rel_err": fitted,
            "rtn_attn_rel_err": rtn,
            "hadamard_attn_rel_err": scorer.score(mode, "hadamard", NATIVE_ROPE) if cb else rtn,
            "ratio_to_rtn": fitted / rtn if rtn else None,
        }
    return rows


def inverse_max_error(result, geometry) -> float:
    """Maximum elementwise error of forward and inverse coordinate transforms on a basis."""
    if isinstance(result, ContentFitResult):
        kind = classify_content_kind(result.r_c, result.s_c)
        if kind == "identity":
            return 0.0
        s = result.s_c.float()
        if kind == "scale_only":
            return float((s * (1.0 / s) - 1.0).abs().max().item())
        r = result.r_c.float()
        identity = torch.eye(len(s), device=s.device)
        roundtrip = (r * s) @ (r.T / s[:, None])
        return float((roundtrip - identity).abs().max().item())
    kind = classify_rope_kind(result.theta, result.s_p)
    if kind == "identity":
        return 0.0
    if kind == "scale_only":
        s = result.s_p.float()
        return float((s * (1.0 / s) - 1.0).abs().max().item())
    identity = torch.eye(geometry.qk_rope_dim, device=result.theta.device)
    forward = apply_rope_pair_transform(identity, result.theta, result.s_p, rope_kind=geometry.rope_kind)
    roundtrip = apply_rope_pair_transform(
        forward, -result.theta, result.s_p, inverse_scale=True, rope_kind=geometry.rope_kind
    )
    return float((roundtrip - identity).abs().max().item())

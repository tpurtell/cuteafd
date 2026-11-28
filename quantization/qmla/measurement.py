"""Candidate cross-tables, shipped-transform selection and held-out attention errors.

Every width's fit (``fitted:c4``) and its SmoothQuant initialization (``smoothquant:c4``)
is a candidate transform for its path. ``measure_candidates`` scores every candidate
(and every content x RoPE candidate pair on joint combinations) against every
requested combination on one split, beside the RTN and fixed-Hadamard baselines.
``select_shipped`` picks one content and one RoPE transform per layer on ``val``;
reported numbers come from the same table measured on ``test``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
import math
from typing import Any, Mapping, Sequence

import torch

from .capture import CapturedLayerBank, cached_content, eval_attention_output
from .fit import (
    ContentFitResult,
    FitConfig,
    RopeFitResult,
    _check_content_equivalence,
    _check_rope_equivalence,
    _init_log_scales,
    _quantile_999,
    apply_content_forward,
    apply_rope_pair_transform,
    compensate_kv_b,
)
from .quantizer import qdq_asymmetric_group, sylvester_hadamard

SELECTION_OBJECTIVE = "mean_val_ratio_to_rtn"
CANDIDATE_KINDS = ("fitted", "smoothquant")
NATIVE_ROPE = "native"


def candidate_id(kind: str, path: str, bits: int) -> str:
    if kind not in CANDIDATE_KINDS or path not in ("content", "rope"):
        raise ValueError(f"invalid candidate {kind!r}/{path!r}")
    return f"{kind}:{'c' if path == 'content' else 'r'}{int(bits)}"


def parse_candidate_id(cid: str) -> tuple[str, str, int]:
    kind, _, width = cid.partition(":")
    if kind not in CANDIDATE_KINDS or len(width) < 2 or width[0] not in "cr" or not width[1:].isdigit():
        raise ValueError(f"invalid candidate id {cid!r}")
    return kind, "content" if width[0] == "c" else "rope", int(width[1:])


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
    """Candidate transforms for one layer, keyed by ``candidate_id`` in a fixed order."""

    content: dict[str, ContentFitResult]
    rope: dict[str, RopeFitResult]


def _content_stats(bank: CapturedLayerBank, device) -> torch.Tensor:
    return _quantile_999(torch.cat([cached_content(seq.u, bank.params) for seq in bank.train]).to(device))


def _rope_stats(bank: CapturedLayerBank, device) -> torch.Tensor:
    k = torch.cat([seq.k_rope.float() for seq in bank.train])
    if bank.geometry.rope_kind == "split_half":
        pair_values = torch.cat(k.chunk(2, dim=-1))
    else:
        pair_values = torch.cat((k[:, ::2], k[:, 1::2]))
    return _quantile_999(pair_values.to(device))


def _init_scale(fit, stats_fn, bank, config) -> torch.Tensor:
    if fit.init_s is not None:
        return fit.init_s.float().cpu()
    return _init_log_scales(stats_fn(bank, "cpu"), fit.init_alpha, config.log_scale_bound).exp()


def layer_candidates(
    bank: CapturedLayerBank,
    content_fits: Mapping[int, ContentFitResult],
    rope_fits: Mapping[int, RopeFitResult],
    config: FitConfig,
) -> LayerCandidates:
    """Every width's fit plus the SmoothQuant initialization it started from.

    A SmoothQuant candidate is the fit's step 0: identity rotation (or zero angles) and
    the width's best alpha-grid scales, as recorded by the fit (``init_s``). Its
    ``equiv_max_rel_err`` is NaN until ``with_equivalence`` measures it.
    """
    content: dict[str, ContentFitResult] = {}
    rope: dict[str, RopeFitResult] = {}
    d_c = bank.geometry.kv_lora_rank
    for bits in sorted(content_fits):
        content[candidate_id("fitted", "content", bits)] = content_fits[bits]
    for bits in sorted(content_fits):
        fit = content_fits[bits]
        content[candidate_id("smoothquant", "content", bits)] = replace(
            fit, r_c=torch.eye(d_c), s_c=_init_scale(fit, _content_stats, bank, config), best_step=0,
            best_val_loss=fit.baselines["smoothquant_alpha_val"], equiv_max_rel_err=math.nan, orth_max_err=0.0,
        )
    if rope_fits and bank.geometry.has_rope:
        for bits in sorted(rope_fits):
            rope[candidate_id("fitted", "rope", bits)] = rope_fits[bits]
        for bits in sorted(rope_fits):
            fit = rope_fits[bits]
            scale = _init_scale(fit, _rope_stats, bank, config)
            rope[candidate_id("smoothquant", "rope", bits)] = replace(
                fit, theta=torch.zeros_like(scale), s_p=scale, best_step=0,
                best_val_loss=fit.baselines["smoothquant_alpha_val"], equiv_max_rel_err=math.nan,
            )
    return LayerCandidates(content=content, rope=rope)


def with_equivalence(result, bank: CapturedLayerBank, device):
    """Measure quantization-off equivalence for a candidate that has not been checked."""
    if math.isfinite(result.equiv_max_rel_err):
        return result
    seqs = tuple(seq.to(device) for seq in bank.val)
    if isinstance(result, ContentFitResult):
        err = _check_content_equivalence(
            seqs, bank.params.to(device), bank.geometry, result.r_c.to(device), result.s_c.to(device)
        )
    else:
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
    """Score one layer's split under arbitrary content/RoPE arms at a given combination."""

    def __init__(self, bank: CapturedLayerBank | None, split: str | None, config: FitConfig, device,
                 *, geom=None, params=None, seqs=()):
        if bank is not None:
            geom = bank.geometry
            params = bank.params
            seqs = bank.split(split)
        self.geom = geom
        self.params = params.to(device)
        self.device = device
        self.config = config
        self.seqs = tuple(seq.to(device) for seq in seqs)
        self.tokens = sum(seq.seq_len for seq in self.seqs)
        self.protection = (
            {"sink_tokens": config.sink_tokens, "recent_tokens": config.recent_tokens}
            if config.protection == "sink4_local128" else {}
        )
        d_c = self.geom.kv_lora_rank
        self.hadamard = sylvester_hadamard(d_c, device=device) if (d_c & (d_c - 1)) == 0 else None
        self._kv_b: dict[str, torch.Tensor] = {}

    def content_hooks(self, bits: int, arm) -> dict[str, Any]:
        """``arm`` is ``"rtn"``, ``"hadamard"`` or ``(cache_key, ContentFitResult)``."""
        if not bits:
            return {}
        params = self.params
        hooks: dict[str, Any] = {}
        if arm == "rtn":
            exact = lambda c: c
        else:
            if arm == "hadamard":
                if self.hadamard is None:
                    raise ValueError("fixed Hadamard needs a power-of-two kv_lora_rank")
                key, rotation, scale = "hadamard", self.hadamard, torch.ones(self.geom.kv_lora_rank, device=self.device)
            else:
                key, result = arm
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
        """``arm`` is ``NATIVE_ROPE`` (RTN and the Hadamard arm) or a ``RopeFitResult``."""
        if not bits or not self.geom.has_rope:
            return {}
        if arm == NATIVE_ROPE:
            # A full Hadamard would mix positional frequencies, so the Hadamard arm uses native RoPE.
            exact = lambda x: x
            hooks: dict[str, Any] = {}
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
    """Score every candidate cell plus RTN and fixed Hadamard for every combination on ``split``.

    Joint combinations score every content x RoPE pair; single-path combinations score each
    candidate of the quantized path. A combination that quantizes nothing at this layer
    (a RoPE-only combination on a NoPE layer) is omitted.
    """
    scorer = _Scorer(bank, split, config, device)
    table: dict[str, dict[str, Any]] = {}
    for mode in modes:
        if not layer_quantizes(bank.geometry, mode):
            continue
        cb, _ = mode
        rtn = scorer.score(mode, "rtn", NATIVE_ROPE)
        row: dict[str, Any] = {
            "tokens": scorer.tokens,
            "rtn": rtn,
            # Hadamard rotates content only, so on RoPE-only combinations it is RTN.
            "hadamard": scorer.score(mode, "hadamard", NATIVE_ROPE) if cb else rtn,
            "cells": {},
        }
        for c, r in table_cells(candidates, bank.geometry, mode):
            content_arm = (c, candidates.content[c]) if c is not None else "rtn"
            rope_arm = candidates.rope[r] if r is not None else NATIVE_ROPE
            row["cells"][pair_id(c, r)] = scorer.score(mode, content_arm, rope_arm)
        table[combo_key(mode)] = row
    return table


def _cell(table_row: Mapping[str, Any], c: str | None, r: str | None) -> float:
    return float(table_row["cells"][pair_id(c, r)])


def _mode_of(key: str) -> tuple[int, int]:
    c, r = key.split(":")
    return int(c), int(r)


def select_shipped(
    val_table: Mapping[str, Mapping[str, Any]],
    candidates: LayerCandidates,
    geometry,
) -> dict[str, Any]:
    """Pick the (content, RoPE) candidate pair with the lowest mean val ratio to RTN.

    The mean runs over every measured combination at this layer, so each width counts
    equally: a 2-bit combination's larger absolute error does not outvote the 8-bit ones.
    Content-only combinations depend on the content candidate alone and RoPE-only ones on
    the RoPE candidate alone; joint combinations score the pair together, which is what
    the per-path fits cannot see.
    """
    if not val_table:
        raise ValueError(f"layer {geometry.layer_id} has no measured combination to select on")
    uses_content = any(_mode_of(k)[0] for k in val_table)
    uses_rope = any(_mode_of(k)[1] for k in val_table) and geometry.has_rope
    content_ids: list[str | None] = list(candidates.content) if uses_content else [None]
    rope_ids: list[str | None] = list(candidates.rope) if uses_rope else [None]
    best: tuple[float, str | None, str | None] | None = None
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
            if best is None or objective < best[0]:
                best = (objective, c, r)
    assert best is not None
    _, c_best, r_best = best

    def path_record(path: str, cid: str | None) -> dict[str, Any] | None:
        if cid is None:
            return None
        kind, _, bits = parse_candidate_id(cid)
        val_score = {}
        for key, row in val_table.items():
            cb, rb = _mode_of(key)
            if (path == "content" and cb) or (path == "rope" and rb):
                val_score[key] = _cell(row, c_best if cb else None, r_best if rb and geometry.has_rope else None)
        return {
            "candidate": cid,
            "chosen": kind,
            "source_bits": bits,
            "objective": objectives[pair_id(c_best, r_best)],
            "val_score": val_score,
        }

    return {
        "objective": SELECTION_OBJECTIVE,
        "content": path_record("content", c_best),
        "rope": path_record("rope", r_best),
        "pair_objectives": objectives,
    }


def natural_cell(kind: str, mode: tuple[int, int], geometry) -> tuple[str | None, str | None]:
    """The cell a combination's own widths give: e.g. ``fitted:c4+fitted:r4`` at 4:4."""
    cb, rb = mode
    return (
        candidate_id(kind, "content", cb) if cb else None,
        candidate_id(kind, "rope", rb) if rb and geometry.has_rope else None,
    )


def layer_rows(
    test_table: Mapping[str, Mapping[str, Any]],
    val_table: Mapping[str, Mapping[str, Any]],
    selection: Mapping[str, Any],
    geometry,
) -> dict[str, dict[str, Any]]:
    """Per-combination test rows for one layer: shipped, per-width fitted, SmoothQuant, baselines."""
    shipped_c = selection["content"]["candidate"] if selection["content"] else None
    shipped_r = selection["rope"]["candidate"] if selection["rope"] else None
    rows = {}
    for key, row in test_table.items():
        mode = _mode_of(key)
        cb, rb = mode
        rope_on = bool(rb) and geometry.has_rope
        shipped = (shipped_c if cb else None, shipped_r if rope_on else None)
        fitted = natural_cell("fitted", mode, geometry)
        smooth = natural_cell("smoothquant", mode, geometry)
        out = {
            "tokens": row["tokens"],
            "attn_rel_err": _cell(row, *shipped),
            "per_width_fitted_attn_rel_err": _cell(row, *fitted),
            "smoothquant_attn_rel_err": _cell(row, *smooth),
            "rtn_attn_rel_err": row["rtn"],
            "hadamard_attn_rel_err": row["hadamard"],
            "shipped_cell": pair_id(*shipped),
            # The per-combination guard: which of this combination's own fitted/init cells wins on val.
            "guard_val": {
                pair_id(c, r): _cell(val_table[key], c, r)
                for c in ((fitted[0], smooth[0]) if cb else (None,))
                for r in ((fitted[1], smooth[1]) if rope_on else (None,))
            },
            "cells": dict(row["cells"]),
        }
        out["guard_choice"] = min(out["guard_val"], key=out["guard_val"].get)
        out["guard_attn_rel_err"] = _cell(row, *_split_pair(out["guard_choice"]))
        out["fitted_beats_smoothquant"] = out["per_width_fitted_attn_rel_err"] < out["smoothquant_attn_rel_err"]
        rows[key] = out
    return rows


def _split_pair(pid: str) -> tuple[str | None, str | None]:
    if "+" in pid:
        c, r = pid.split("+")
        return c, r
    _, path, _ = parse_candidate_id(pid)
    return (pid, None) if path == "content" else (None, pid)


REPORTED_ERRORS = (
    "attn_rel_err",
    "per_width_fitted_attn_rel_err",
    "smoothquant_attn_rel_err",
    "guard_attn_rel_err",
    "rtn_attn_rel_err",
    "hadamard_attn_rel_err",
)


def summarize_modes(by_layer: Mapping[Any, Mapping[str, Mapping[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Token-weighted test summaries per combination over layers, plus the cross-table."""
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
        for field, ratio in (
            ("attn_rel_err", "ratio_to_rtn"),
            ("per_width_fitted_attn_rel_err", "per_width_fitted_ratio_to_rtn"),
            ("smoothquant_attn_rel_err", "smoothquant_ratio_to_rtn"),
            ("hadamard_attn_rel_err", "hadamard_ratio_to_rtn"),
        ):
            summary[ratio] = summary[field] / rtn if rtn else None
        summary["fitted_beats_smoothquant_layers"] = sum(bool(row["fitted_beats_smoothquant"]) for row in rows)
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
            for cell, e in cross.items()
        }
    return summaries


def shipped_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """The per-combination numbers the sidecar manifest carries (test split, shipped transforms)."""
    return {
        "layer_tokens": summary["layer_tokens"],
        "attn_rel_err": summary["attn_rel_err"],
        "rtn_attn_rel_err": summary["rtn_attn_rel_err"],
        "hadamard_attn_rel_err": summary["hadamard_attn_rel_err"],
        "smoothquant_attn_rel_err": summary["smoothquant_attn_rel_err"],
        "per_width_fitted_attn_rel_err": summary["per_width_fitted_attn_rel_err"],
        "ratio_to_rtn": summary["ratio_to_rtn"],
    }


@torch.no_grad()
def measure_shipped(
    bank: CapturedLayerBank,
    content: ContentFitResult | None,
    rope: RopeFitResult | None,
    modes: Sequence[tuple[int, int]],
    config: FitConfig,
    device: torch.device | str,
) -> dict[str, dict[str, Any]]:
    """Shipped transforms vs RTN and Hadamard on ``bank.test`` (e.g. a K8 transfer bank).

    Errors are against the bank's own 0:0 reference outputs.
    """
    scorer = _Scorer(bank, "test", config, device)
    rows = {}
    for mode in modes:
        if not layer_quantizes(bank.geometry, mode):
            continue
        cb, rb = mode
        if cb and content is None or rb and bank.geometry.has_rope and rope is None:
            raise ValueError(f"combination {combo_key(mode)} needs a shipped transform for each quantized path")
        fitted = scorer.score(mode, ("shipped", content) if cb else "rtn", rope if rb and rope is not None else NATIVE_ROPE)
        rtn = scorer.score(mode, "rtn", NATIVE_ROPE)
        rows[combo_key(mode)] = {
            "tokens": scorer.tokens,
            "attn_rel_err": fitted,
            "rtn_attn_rel_err": rtn,
            "hadamard_attn_rel_err": scorer.score(mode, "hadamard", NATIVE_ROPE) if cb else rtn,
            "ratio_to_rtn": fitted / rtn if rtn else None,
        }
    return rows


def inverse_max_error(result, geometry):
    """Maximum elementwise error of forward and inverse coordinate transforms on a basis."""
    if isinstance(result, ContentFitResult):
        r = result.r_c.float()
        s = result.s_c.float()
        identity = torch.eye(len(s), device=s.device)
        roundtrip = (r * s) @ (r.T / s[:, None])
    else:
        identity = torch.eye(geometry.qk_rope_dim, device=result.theta.device)
        forward = apply_rope_pair_transform(identity, result.theta, result.s_p, rope_kind=geometry.rope_kind)
        roundtrip = apply_rope_pair_transform(
            forward, -result.theta, result.s_p, inverse_scale=True, rope_kind=geometry.rope_kind
        )
    return float((roundtrip - identity).abs().max())

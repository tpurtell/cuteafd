"""Held-out, post-output-projection measurements for pairs of independent cache profiles."""
from __future__ import annotations

from functools import partial
import math
from typing import Any, Mapping, Sequence

import torch

from .capture import CapturedLayerBank, eval_attention_output
from .fit import (
    ContentFitResult,
    FitConfig,
    RopeFitResult,
    _init_log_scales,
    _quantile_999,
    apply_content_forward,
    apply_rope_pair_transform,
    compensate_kv_b,
)
from .quantizer import qdq_asymmetric_group, sylvester_hadamard


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


@torch.no_grad()
def measure_layer_modes(
    bank: CapturedLayerBank,
    training_bank: CapturedLayerBank,
    modes: Sequence[tuple[int, int]],
    content: Mapping[int, ContentFitResult],
    rope: Mapping[int, RopeFitResult],
    config: FitConfig,
    device: torch.device | str,
) -> dict[str, dict[str, Any]]:
    """Compare all arms with the held-out bank's own 0:0 reference.

    SmoothQuant initialization statistics always come from the fitting view's training
    bank, including when evaluating transferred fits on a different validation view.
    """
    geom = bank.geometry
    params = bank.params.to(device)
    eye = torch.eye(geom.kv_lora_rank, device=device)
    ones = torch.ones(geom.kv_lora_rank, device=device)
    hadamard = sylvester_hadamard(geom.kv_lora_rank, device=device) if content else None
    if content:
        u = torch.cat([seq.u.float() for seq in training_bank.train])
        a_content = _quantile_999(u.to(device))
        del u
    if rope:
        pairs = geom.qk_rope_dim // 2
        k = torch.cat([seq.k_rope.float() for seq in training_bank.train])
        if geom.rope_kind == "split_half":
            pair_values = torch.cat(k.chunk(2, dim=-1))
        else:
            pair_values = torch.cat((k[:, ::2], k[:, 1::2]))
        a_rope = _quantile_999(pair_values.to(device))
        del k, pair_values

    val = tuple(seq.to(device) for seq in bank.val)
    tokens = sum(seq.seq_len for seq in val)
    protection = (
        {"sink_tokens": config.sink_tokens, "recent_tokens": config.recent_tokens}
        if config.protection == "sink4_local128" else {}
    )

    def content_hooks(bits, arm):
        if not bits:
            return {}
        result = content[bits]
        hooks = {}
        if arm == "rtn":
            exact = lambda u: u * params.kv_norm_weight.float()
        else:
            if arm == "hadamard":
                rotation, scale = hadamard, ones
            elif arm == "smoothquant":
                rotation = eye
                scale = _init_log_scales(a_content, result.init_alpha, config.log_scale_bound).exp()
            else:
                rotation, scale = result.r_c.to(device), result.s_c.to(device)
            exact = partial(apply_content_forward, r_c=rotation, s_c=scale)
            hooks["kv_b_override"] = compensate_kv_b(
                params.kv_b_weight, params.kv_norm_weight, rotation, scale
            )
        hooks["content_fn"] = _quantized(exact, bits, "BF16", config)
        if protection:
            hooks["exact_content_fn"] = exact
        return hooks

    def rope_hooks(bits, arm):
        if not bits or not geom.has_rope:
            return {}
        result = rope[bits]
        if arm in ("rtn", "hadamard"):
            # A full Hadamard would mix positional frequencies. This arm uses native RoPE.
            exact = lambda x: x
            hooks = {}
        else:
            if arm == "smoothquant":
                theta = torch.zeros_like(a_rope)
                scale = _init_log_scales(a_rope, result.init_alpha, config.log_scale_bound).exp()
            else:
                theta, scale = result.theta.to(device), result.s_p.to(device)
            exact = partial(apply_rope_pair_transform, theta=theta, scale=scale, rope_kind=geom.rope_kind)
            hooks = {"rope_query_fn": partial(exact, inverse_scale=True)}
        hooks["rope_key_fn"] = _quantized(exact, bits, "F32", config)
        if protection:
            hooks["exact_rope_key_fn"] = exact
        return hooks

    results = {}
    for cb, rb in modes:
        row = {"tokens": tokens}
        for arm in ("fitted", "rtn", "hadamard", "smoothquant"):
            hooks = {**content_hooks(cb, arm), **rope_hooks(rb, arm), **protection}
            token_sums = []
            for seq in val:
                pred = eval_attention_output(seq, params, geom, **hooks)
                token_sums.append(_token_relative_squares(pred, seq.ref_output).sum(dtype=torch.float64))
            mean = torch.stack(token_sums).sum().item() / tokens
            if not math.isfinite(mean):
                raise ValueError(f"invalid {arm} attention error at layer {geom.layer_id} for {cb}:{rb}")
            row["attn_rel_err" if arm == "fitted" else f"{arm}_attn_rel_err"] = mean
        row["ratio_to_rtn"] = row["attn_rel_err"] / row["rtn_attn_rel_err"] if row["rtn_attn_rel_err"] else None
        results[f"{cb}:{rb}"] = row
    return results


def summarize_modes(by_layer):
    summaries = {}
    for layer, modes in by_layer.items():
        for key, row in modes.items():
            summary = summaries.setdefault(key, {"per_layer": {}})
            summary["per_layer"][str(layer)] = row
    for summary in summaries.values():
        rows = list(summary["per_layer"].values())
        tokens = sum(row["tokens"] for row in rows)
        summary["layer_tokens"] = tokens
        for field in ("attn_rel_err", "rtn_attn_rel_err", "hadamard_attn_rel_err", "smoothquant_attn_rel_err"):
            summary[field] = sum(row[field] * row["tokens"] for row in rows) / tokens
        rtn = summary["rtn_attn_rel_err"]
        summary["ratio_to_rtn"] = summary["attn_rel_err"] / rtn if rtn else None
    return summaries


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

"""Function-aligned dual-path QMLA transformation optimizer and baseline evaluator.

Reproduces QuantMLA (Su et al., 2026-09, arXiv:2609.36760) Section 3 and Appendix C.4,
with CuteAFD Adaptation A4: recent-token protection (``sink4_local128``) is enabled by
default on full-history MLA layers during both calibration and serving (SWA untouched),
with ``protection="none"`` available as the measured unprotected alternative:
- Content path: Cayley-parameterized orthogonal R_C and log-space positive scale S_C
  on the native cached latent (z_C = c R_C S_C, compensated consumer
  W_kv_b_tilde = W_kv_b @ R_C @ diag(1/S_C)), trained by post-o_proj
  attention-output reconstruction L_C (Eq. 8). Supports shared-KV multi-consumer layers.
- RoPE path: per-frequency-pair SO(2) angles theta_i and pair-tied positive scales s_i
  (Q'_P = Q_P R_P S_P^-1, K'_P = K_P R_P S_P), trained by causal positional-QK
  reconstruction L_P (Eq. 10).
- Step-0 baselines: RTN on the native cached latent, fixed Sylvester Hadamard
  (QuaRot-style / gr8 comparison), and 99.9th-percentile SmoothQuant-style alpha grid.
- Sequential per-layer checkpointing with streamed XXH3-128 receipts for resume.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch
import xxhash

from .capture import (
    CapturedLayerBank,
    CapturedLayerParams,
    CapturedSequence,
    DeviceSplit,
    split_residency_budget,
    split_lengths,
    MlaLayerGeometry,
    ROWS,
    causal_mask,
    cached_content,
    _atomic_json,
    _read_safetensors_verified,
    _write_safetensors_streamed,
    eval_attention_output,
    eval_positional_qk,
)
from .quantizer import (
    CONTENT_NONZERO_BITS,
    EPS_SCALE,
    GROUP_SIZE,
    ROPE_NONZERO_BITS,
    qdq_asymmetric_group,
    sylvester_hadamard,
)

FIT_CHECKPOINT_SCHEMA = "cuteafd-qmla-fit-checkpoint-v3"
ALPHA_INIT_GRID = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
DEFAULT_PROTECTION_POLICY = "sink4_local128"
VALID_PROTECTION_POLICIES = frozenset({"sink4_local128", "none"})
DEFAULT_SINK_TOKENS = 4
DEFAULT_RECENT_TOKENS = 128

# Explicit transform kind classification tolerances (cuteafd-qmla-v3).
# Both sit >1000x below BF16 half-ULP (~3.9e-3) and INT8 step size (>=3.9e-3),
# while absorbing FP32 Cayley/exp(0) roundoff (~few x 1e-7) and staying 10x tighter
# than the orthogonality gate (1e-5).
ROTATION_IDENTITY_TOL = 1e-6
SCALE_IDENTITY_TOL = 1e-6
VALID_TRANSFORM_KINDS = frozenset({"identity", "scale_only", "full"})
VALID_FIT_MODES = frozenset({"full", "scale_only"})
VALID_CHECKPOINT_MODES = frozenset({"full", "scale_only", "smoothquant"})
# Default v3 fitting budget: fit BOTH full and scale_only across every calibrated width
# (plus step-0 SmoothQuant inits). Never blanket-skip 8-bit full fitting: Hy4-flash layer 25
# gained 10.9% beyond scale-only on 8-bit RoPE (vs 0.7% median).
DEFAULT_FIT_BUDGET: dict[str, dict[int, tuple[str, ...]]] = {
    "content": {
        2: ("full", "scale_only"),
        4: ("full", "scale_only"),
        8: ("full", "scale_only"),
    },
    "rope": {
        4: ("full", "scale_only"),
        8: ("full", "scale_only"),
    },
}


@dataclass(frozen=True)
class FitConfig:
    """Hyperparameters matching QuantMLA Appendix C.4 Table 7 (overridable for pilots).

    Defaults ``protection="sink4_local128"`` (4 sink + 128 recent tokens exact on
    full-history MLA layers, CuteAFD Adaptation A4).
    """

    steps: int = 300
    content_rot_lr: float = 3e-3
    content_scale_lr: float = 1e-3
    rope_angle_lr: float = 1e-3
    rope_scale_lr: float = 1e-3
    anchor: float = 1e-4
    warmup_frac: float = 0.1
    grad_clip: float = 1.0
    val_every: int = 20
    log_scale_bound: float = 2.0
    group_size: int = GROUP_SIZE
    eps_scale: float = EPS_SCALE
    seed: int = 0
    alpha_grid: tuple[float, ...] = ALPHA_INIT_GRID
    batch_size: int = 4
    protection: str = DEFAULT_PROTECTION_POLICY
    sink_tokens: int = DEFAULT_SINK_TOKENS
    recent_tokens: int = DEFAULT_RECENT_TOKENS

    def validate(self) -> None:
        if self.steps < 0:
            raise ValueError("steps must be non-negative")
        if self.val_every <= 0 or self.batch_size <= 0 or self.group_size <= 0:
            raise ValueError("val_every, batch_size, and group_size must be positive")
        if not (0.0 <= self.warmup_frac <= 1.0) or self.log_scale_bound <= 0.0:
            raise ValueError("invalid warmup_frac or log_scale_bound")
        if self.protection not in VALID_PROTECTION_POLICIES:
            raise ValueError(
                f"invalid protection policy {self.protection!r}; expected one of {sorted(VALID_PROTECTION_POLICIES)}"
            )
        if self.sink_tokens < 0 or self.recent_tokens < 0:
            raise ValueError("sink_tokens and recent_tokens must be non-negative")


@dataclass(frozen=True)
class ContentFitResult:
    layer_id: int
    bits: int
    r_c: torch.Tensor
    s_c: torch.Tensor
    best_step: int
    best_val_loss: float
    init_alpha: float
    equiv_max_rel_err: float
    orth_max_err: float
    baselines: dict[str, float]
    protection_calibrated: str = DEFAULT_PROTECTION_POLICY
    init_s: torch.Tensor | None = None  # SmoothQuant step-0 scales (identity rotation)
    fit_mode: str = "full"  # "full" | "scale_only"


@dataclass(frozen=True)
class RopeFitResult:
    layer_id: int
    bits: int
    theta: torch.Tensor
    s_p: torch.Tensor
    best_step: int
    best_val_loss: float
    init_alpha: float
    equiv_max_rel_err: float
    baselines: dict[str, float]
    protection_calibrated: str = DEFAULT_PROTECTION_POLICY
    init_s: torch.Tensor | None = None  # SmoothQuant step-0 pair scales (zero angles)
    fit_mode: str = "full"  # "full" | "scale_only"


def classify_content_kind(
    r_c: torch.Tensor | None,
    s_c: torch.Tensor | None,
    *,
    rot_tol: float = ROTATION_IDENTITY_TOL,
    scale_tol: float = SCALE_IDENTITY_TOL,
) -> str:
    """Classify a content transform into ``identity`` | ``scale_only`` | ``full``.

    Any transform whose rotation is within ``rot_tol`` (L_inf) of I and whose scales
    are within ``scale_tol`` (L_inf) of 1 is ``identity``, regardless of which optimizer
    produced it. A non-unit scale with identity rotation is ``scale_only``; any
    non-identity rotation is ``full``.
    """
    if r_c is None:
        rot_is_id = True
    else:
        rf = r_c.detach().float()
        eye = torch.eye(rf.shape[0], device=rf.device, dtype=torch.float32)
        rot_is_id = float((rf - eye).abs().max().item()) <= rot_tol
    if s_c is None:
        scale_is_id = True
    else:
        sf = s_c.detach().float()
        scale_is_id = float((sf - 1.0).abs().max().item()) <= scale_tol
    if rot_is_id and scale_is_id:
        return "identity"
    if rot_is_id:
        return "scale_only"
    return "full"


def classify_rope_kind(
    theta: torch.Tensor | None,
    s_p: torch.Tensor | None,
    *,
    rot_tol: float = ROTATION_IDENTITY_TOL,
    scale_tol: float = SCALE_IDENTITY_TOL,
) -> str:
    """Classify a RoPE pair transform into ``identity`` | ``scale_only`` | ``full``.

    Angles are wrapped to ``(-pi, pi]`` via ``atan2(sin(theta), cos(theta))`` so
    exact 2*pi multiples also classify as zero rotation.
    """
    if theta is None:
        rot_is_id = True
    else:
        tf = theta.detach().float()
        wrapped = torch.atan2(torch.sin(tf), torch.cos(tf))
        rot_is_id = float(wrapped.abs().max().item()) <= rot_tol
    if s_p is None:
        scale_is_id = True
    else:
        sf = s_p.detach().float()
        scale_is_id = float((sf - 1.0).abs().max().item()) <= scale_tol
    if rot_is_id and scale_is_id:
        return "identity"
    if rot_is_id:
        return "scale_only"
    return "full"


def protection_mask(
    seq_len: int,
    *,
    protection: str = DEFAULT_PROTECTION_POLICY,
    sink_tokens: int = DEFAULT_SINK_TOKENS,
    recent_tokens: int = DEFAULT_RECENT_TOKENS,
    device: torch.device | str = "cpu",
) -> torch.Tensor | None:
    """Return boolean mask [T] where True marks protected exact tokens P_T (None for ``none``)."""
    if protection not in VALID_PROTECTION_POLICIES:
        raise ValueError(f"unsupported protection policy: {protection!r}")
    if protection == "none":
        return None
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    if sink_tokens > 0:
        mask[: min(sink_tokens, seq_len)] = True
    if recent_tokens > 0:
        mask[max(0, seq_len - recent_tokens) :] = True
    return mask


def _require_unprotected_tokens(
    seqs: Sequence[CapturedSequence],
    *,
    protection: str,
    sink_tokens: int,
    recent_tokens: int,
    split_name: str,
) -> None:
    if protection == "none":
        return
    unprotected = 0
    for length in split_lengths(seqs):
        mask = protection_mask(
            length,
            protection=protection,
            sink_tokens=sink_tokens,
            recent_tokens=recent_tokens,
            device="cpu",
        )
        assert mask is not None
        unprotected += int((~mask).sum().item())
    if unprotected == 0:
        raise ValueError(
            f"protection={protection!r} (sink_tokens={sink_tokens}, recent_tokens={recent_tokens}) "
            f"covers 100% of {split_name} tokens; sequences must be longer than {sink_tokens + recent_tokens} "
            "tokens (or pass --protection none for an explicit unprotected measurement)"
        )


def cayley_orthogonal(skew_raw: torch.Tensor) -> torch.Tensor:
    """Exact SO(d) matrix via Cayley map: A = 0.5*(B - B^T), R = (I - A)(I + A)^-1."""
    d = skew_raw.shape[0]
    a = 0.5 * (skew_raw - skew_raw.transpose(0, 1))
    eye = torch.eye(d, device=skew_raw.device, dtype=skew_raw.dtype)
    return torch.linalg.solve(eye + a, eye - a)


def apply_content_forward(c: torch.Tensor, r_c: torch.Tensor, s_c: torch.Tensor) -> torch.Tensor:
    """Row-vector content transform z_C = c @ R_C @ diag(s_C)."""
    return torch.matmul(c.float(), r_c.float()) * s_c.float()


def apply_content_scale_only(c: torch.Tensor, s_c: torch.Tensor) -> torch.Tensor:
    """Row-vector scale-only content transform z_C = c * s_C (no dense rotation)."""
    return c.float() * s_c.float()


def compensate_kv_b(
    kv_b_weight: torch.Tensor,
    r_c: torch.Tensor,
    s_c: torch.Tensor,
) -> torch.Tensor:
    """Compensate HF-layout kv_b_proj [out, d_c]: W_tilde = W_kv_b @ R_C @ diag(1/s_C)."""
    return torch.matmul(kv_b_weight.float(), r_c.float()) * (1.0 / s_c.float()).unsqueeze(0)


def compensate_kv_b_scale_only(
    kv_b_weight: torch.Tensor,
    s_c: torch.Tensor,
) -> torch.Tensor:
    """Compensate HF-layout kv_b_proj [out, d_c] for scale_only: W_tilde = W_kv_b @ diag(1/s_C)."""
    return kv_b_weight.float() * (1.0 / s_c.float()).unsqueeze(0)


def apply_rope_scale_only(
    x: torch.Tensor,
    scale: torch.Tensor,
    *,
    rope_kind: str,
    inverse_scale: bool = False,
) -> torch.Tensor:
    """Apply pair-tied RoPE scales without rotation (``scale_only`` kind)."""
    xf = x.float()
    g = (1.0 / scale.float()) if inverse_scale else scale.float()
    if rope_kind in ("interleaved", "unrotated_pe"):
        x0 = xf[..., 0::2] * g
        x1 = xf[..., 1::2] * g
        return torch.stack((x0, x1), dim=-1).flatten(-2)
    if rope_kind == "split_half":
        half = xf.shape[-1] // 2
        return torch.cat((xf[..., :half] * g, xf[..., half:] * g), dim=-1)
    raise ValueError(f"apply_rope_scale_only unsupported for rope_kind={rope_kind!r}")


def apply_rope_pair_transform(
    x: torch.Tensor,
    theta: torch.Tensor,
    scale: torch.Tensor,
    *,
    rope_kind: str,
    inverse_scale: bool = False,
) -> torch.Tensor:
    """Apply per-frequency-pair R_i(theta_i) and pair-tied scale s_i (or 1/s_i).

    Row-vector convention (matches ``qmla-design.md`` §3):
      y0 = (x0 * cos(theta) - x1 * sin(theta)) * g
      y1 = (x0 * sin(theta) + x1 * cos(theta)) * g
    where ``g = 1/scale`` for queries and ``g = scale`` for keys.
    """
    xf = x.float()
    c = torch.cos(theta.float())
    s = torch.sin(theta.float())
    g = (1.0 / scale.float()) if inverse_scale else scale.float()
    cg = c * g
    sg = s * g

    if rope_kind in ("interleaved", "unrotated_pe"):
        x0 = xf[..., 0::2]
        x1 = xf[..., 1::2]
        y0 = x0 * cg - x1 * sg
        y1 = x0 * sg + x1 * cg
        return torch.stack((y0, y1), dim=-1).flatten(-2)
    if rope_kind == "split_half":
        half = xf.shape[-1] // 2
        x0 = xf[..., :half]
        x1 = xf[..., half:]
        y0 = x0 * cg - x1 * sg
        y1 = x0 * sg + x1 * cg
        return torch.cat((y0, y1), dim=-1)
    raise ValueError(f"apply_rope_pair_transform unsupported for rope_kind={rope_kind!r}")


def _normalized_output_mse(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    diff = pred - ref
    num = diff.pow(2).mean()
    denom = ref.pow(2).mean().clamp_min(1e-8)
    return num / denom


def content_output_loss(
    seqs: Sequence[CapturedSequence],
    params: CapturedLayerParams,
    geometry: MlaLayerGeometry,
    r_c: torch.Tensor | None,
    s_c: torch.Tensor,
    bits: int,
    *,
    quantize: bool = True,
    group_size: int = GROUP_SIZE,
    eps_scale: float = EPS_SCALE,
    ste: bool = True,
    protection: str = DEFAULT_PROTECTION_POLICY,
    sink_tokens: int = DEFAULT_SINK_TOKENS,
    recent_tokens: int = DEFAULT_RECENT_TOKENS,
    extra_consumers: Sequence[tuple[CapturedLayerParams, MlaLayerGeometry, Sequence[CapturedSequence]]] = (),
) -> torch.Tensor:
    """Mean normalized attention-output reconstruction loss L_C (QuantMLA Eq. 8 + A4 protection)."""
    if protection not in VALID_PROTECTION_POLICIES:
        raise ValueError(f"unsupported protection policy: {protection!r}")
    all_groups = [(params, geometry, seqs), *extra_consumers]
    losses: list[torch.Tensor] = []
    for c_params, c_geom, c_seqs in all_groups:
        if r_c is None:
            w_tilde = compensate_kv_b_scale_only(c_params.kv_b_weight, s_c)

            def exact_content(c):
                return apply_content_scale_only(c, s_c)
        else:
            w_tilde = compensate_kv_b(c_params.kv_b_weight, r_c, s_c)

            def exact_content(c):
                return apply_content_forward(c, r_c, s_c)

        def quantized_content(c):
            return qdq_asymmetric_group(
                exact_content(c), bits, group_size=group_size, scale_dtype="BF16", eps=eps_scale, ste=ste
            )

        for seq in c_seqs:
            pred = eval_attention_output(
                seq, c_params, c_geom, content_fn=quantized_content if quantize else exact_content,
                kv_b_override=w_tilde, exact_content_fn=exact_content if quantize and protection != "none" else None,
                sink_tokens=sink_tokens, recent_tokens=recent_tokens,
            )
            losses.append(_normalized_output_mse(pred, seq.ref_output.float()))
    return torch.stack(losses).mean()


def rope_qk_loss(
    seqs: Sequence[CapturedSequence],
    geometry: MlaLayerGeometry,
    theta: torch.Tensor | None,
    s_p: torch.Tensor,
    bits: int,
    *,
    quantize: bool = True,
    group_size: int = GROUP_SIZE,
    eps_scale: float = EPS_SCALE,
    ste: bool = True,
    protection: str = DEFAULT_PROTECTION_POLICY,
    sink_tokens: int = DEFAULT_SINK_TOKENS,
    recent_tokens: int = DEFAULT_RECENT_TOKENS,
) -> torch.Tensor:
    """Mean normalized causal positional-QK reconstruction loss L_P (QuantMLA Eq. 10 + A4 protection)."""
    if protection not in VALID_PROTECTION_POLICIES:
        raise ValueError(f"unsupported protection policy: {protection!r}")
    losses: list[torch.Tensor] = []

    def _q_fn(qp: torch.Tensor) -> torch.Tensor:
        if theta is None:
            return apply_rope_scale_only(qp, s_p, rope_kind=geometry.rope_kind, inverse_scale=True)
        return apply_rope_pair_transform(qp, theta, s_p, rope_kind=geometry.rope_kind, inverse_scale=True)

    for seq in seqs:
        if seq.k_rope is None or seq.q_rope is None:
            raise ValueError("missing q_rope/k_rope for RoPE layer")
        q, k = seq.q_rope.float().transpose(0, 1), seq.k_rope.float()
        qt = _q_fn(q)
        if theta is None:
            kt = apply_rope_scale_only(k, s_p, rope_kind=geometry.rope_kind, inverse_scale=False)
        else:
            kt = apply_rope_pair_transform(k, theta, s_p, rope_kind=geometry.rope_kind, inverse_scale=False)
        kq = qdq_asymmetric_group(
            kt, bits, group_size=group_size, scale_dtype="F32", eps=eps_scale, ste=ste
        ) if quantize else kt
        protect = quantize and protection != "none" and geometry.kind != "swa"
        mask = seq.allowed_mask
        if mask is None:
            mask = causal_mask(seq.seq_len, device=k.device, sliding_window=geometry.sliding_window)
        errors, norms = [], []
        indices = torch.arange(seq.seq_len, device=k.device)
        for start in range(0, seq.seq_len, ROWS):
            stop = min(start + ROWS, seq.seq_len)
            ref = q[:, start:stop] @ k[:stop].T
            pred = qt[:, start:stop] @ kq[:stop].T
            if protect:
                distance = indices[start:stop, None] - indices[:stop]
                exact = (indices[:stop] < sink_tokens) | (distance < recent_tokens)
                pred = torch.where(exact, qt[:, start:stop] @ kt[:stop].T, pred)
            allowed = mask[start:stop, :stop]
            errors.append((pred - ref).masked_fill(~allowed, 0.0).square().sum())
            norms.append(ref.masked_fill(~allowed, 0.0).square().sum())
        losses.append(torch.stack(errors).sum() / torch.stack(norms).sum().clamp_min(1e-8))
    return torch.stack(losses).mean()


def _quantile_999(samples_2d: torch.Tensor) -> torch.Tensor:
    """99.9th-percentile absolute magnitude per channel (QuantMLA Appendix C.4 Eq. 37)."""
    abs_x = samples_2d.float().abs()
    return torch.quantile(abs_x, 0.999, dim=0).clamp_min(1e-8)


def _bounded_quantile(samples, device):
    # torch.quantile limits the total element count to 2**24; columns are independent.
    width = max(1, (2**24) // samples.shape[0])
    if samples.shape[0] > 2**24:
        raise ValueError("quantile input exceeds supported row count")
    return torch.cat([_quantile_999(part.to(device).contiguous())
                      for part in samples.split(width, dim=1)])


def content_stats(seqs, params, device):
    # Only the cached latent is needed; the large query/output tensors never enter this bank.
    samples = torch.empty((sum(split_lengths(seqs)), params.kv_norm_weight.numel()),
                          dtype=torch.float32, device="cpu")
    offset = 0
    for seq in seqs:
        values = cached_content(seq.u.to(device), params)
        samples[offset:offset + len(values)].copy_(values)
        offset += len(values)
    return _bounded_quantile(samples, device)


def rope_stats(seqs, geom, device):
    count = sum(split_lengths(seqs))
    pairs = geom.qk_rope_dim // 2
    samples = torch.empty((2 * count, pairs), dtype=torch.float32, device="cpu")
    offset = 0
    for seq in seqs:
        k = seq.k_rope.float()
        a, b = k.chunk(2, dim=-1) if geom.rope_kind == "split_half" else (k[:, ::2], k[:, 1::2])
        samples[offset:offset + len(k)].copy_(a)
        samples[count + offset:count + offset + len(k)].copy_(b)
        offset += len(k)
    return _bounded_quantile(samples, device)


def _init_log_scales(a_j: torch.Tensor, alpha: float, bound: float) -> torch.Tensor:
    log_a = torch.log(a_j.clamp_min(1e-8))
    return (float(alpha) * (log_a.mean() - log_a)).clamp(-bound, bound)


def _cosine_lr_factor(step: int, total_steps: int, warmup_frac: float) -> float:
    if total_steps <= 1:
        return 1.0
    warmup = max(1, int(round(total_steps * warmup_frac)))
    if step < warmup:
        return float(step + 1) / float(warmup)
    progress = float(step - warmup) / float(max(1, total_steps - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _relative_l2(pred: torch.Tensor, ref: torch.Tensor) -> float:
    denom = float(ref.float().norm().item())
    if denom == 0.0:
        return float((pred.float() - ref.float()).norm().item())
    return float((pred.float() - ref.float()).norm().item() / denom)


@torch.no_grad()
def check_content_identity(seqs, params, geometry):
    """Require the unquantized identity write and compensated read to reproduce native bytes."""
    rank = geometry.kv_lora_rank
    r_c = torch.eye(rank, device=params.kv_b_weight.device)
    s_c = torch.ones(rank, device=params.kv_b_weight.device)
    weight = compensate_kv_b(params.kv_b_weight, r_c, s_c)
    for seq in seqs:
        pred = eval_attention_output(
            seq, params, geometry, content_fn=lambda c: apply_content_forward(c, r_c, s_c),
            kv_b_override=weight,
        )
        ref = seq.ref_output
        if pred.dtype != ref.dtype or not torch.equal(
            pred.contiguous().view(torch.uint8), ref.contiguous().view(torch.uint8)
        ):
            raise ValueError(f"layer {geometry.layer_id} sequence {seq.seq_id}: "
                             "quantization-off identity differs from native capture bytes "
                             f"(relative L2 {_relative_l2(pred, ref):.8g})")


@torch.no_grad()
def _check_content_equivalence(seqs, params, geometry, r_c, s_c):
    """Max per-sequence expanded K/V relative L2, before the checkpoint's native GEMM rounding."""
    weight = params.kv_b_weight.float()
    w_tilde = compensate_kv_b(weight, r_c, s_c)
    maximum = 0.0
    for seq in seqs:
        sums = torch.zeros(2, dtype=torch.float64, device=weight.device)
        for tile in seq.u.split(ROWS):
            c = cached_content(tile, params)
            native = torch.nn.functional.linear(c, weight)
            transformed = torch.nn.functional.linear(apply_content_forward(c, r_c, s_c), w_tilde)
            sums += torch.stack(((transformed - native).square().sum(), native.square().sum()))
        error, norm = sums.tolist()
        if not math.isfinite(error) or not math.isfinite(norm):
            raise ValueError(f"nonfinite expanded K/V equivalence at layer {geometry.layer_id}")
        maximum = max(maximum, math.sqrt(error / norm if norm else error))
    return maximum


def _content_native_output_error(
    seqs: Sequence[CapturedSequence],
    params: CapturedLayerParams,
    geometry: MlaLayerGeometry,
    r_c: torch.Tensor,
    s_c: torch.Tensor,
) -> float:
    w_tilde = compensate_kv_b(params.kv_b_weight, r_c, s_c)
    max_err = 0.0
    with torch.no_grad():
        for seq in seqs:
            pred = eval_attention_output(
                seq,
                params,
                geometry,
                content_fn=lambda u: apply_content_forward(u, r_c, s_c),
                kv_b_override=w_tilde,
            )
            err = _relative_l2(pred, seq.ref_output)
            if not math.isfinite(err):
                raise ValueError(f"nonfinite native output error at layer {geometry.layer_id}")
            max_err = max(max_err, err)
    return max_err


def _check_rope_equivalence(
    seqs: Sequence[CapturedSequence],
    geometry: MlaLayerGeometry,
    theta: torch.Tensor,
    s_p: torch.Tensor,
) -> float:
    max_err = 0.0
    with torch.no_grad():
        for seq in seqs:
            ref_z, mask = eval_positional_qk(seq, geometry)
            pred_z, _ = eval_positional_qk(
                seq,
                geometry,
                rope_query_fn=lambda q: apply_rope_pair_transform(
                    q, theta, s_p, rope_kind=geometry.rope_kind, inverse_scale=True
                ),
                rope_key_fn=lambda k: apply_rope_pair_transform(
                    k, theta, s_p, rope_kind=geometry.rope_kind, inverse_scale=False
                ),
            )
            m = mask.unsqueeze(0)
            max_err = max(max_err, _relative_l2(pred_z.masked_fill(~m, 0.0), ref_z.masked_fill(~m, 0.0)))
    return max_err


def fit_content_layer(
    bank: CapturedLayerBank,
    bits: int,
    config: FitConfig = FitConfig(),
    *,
    device: torch.device | str = "cpu",
    extra_consumers: Sequence[CapturedLayerBank] = (),
    fit_mode: str = "full",
) -> ContentFitResult:
    """Fit orthogonal R_C and positive scale S_C (or scale_only S_C with R_C=I) for one MLA layer."""
    if bits not in CONTENT_NONZERO_BITS:
        raise ValueError(f"fit_content_layer requires bits in {sorted(CONTENT_NONZERO_BITS)}, got {bits}")
    if fit_mode not in VALID_FIT_MODES:
        raise ValueError(f"invalid fit_mode {fit_mode!r}; expected one of {sorted(VALID_FIT_MODES)}")
    config.validate()
    bank.validate()
    _require_unprotected_tokens(
        bank.train,
        protection=config.protection,
        sink_tokens=config.sink_tokens,
        recent_tokens=config.recent_tokens,
        split_name="train",
    )
    _require_unprotected_tokens(
        bank.val,
        protection=config.protection,
        sink_tokens=config.sink_tokens,
        recent_tokens=config.recent_tokens,
        split_name="val",
    )
    dev = torch.device(device)

    geom = bank.geometry
    d_c = geom.kv_lora_rank
    params = bank.params.to(dev)
    train_seqs = DeviceSplit(bank.train, dev)
    budget = split_residency_budget(dev) // (1 + len(extra_consumers))
    val_seqs = DeviceSplit(bank.val, dev, budget_bytes=budget)
    extra_train = tuple((b.params.to(dev), b.geometry, DeviceSplit(b.train, dev)) for b in extra_consumers)
    extra_val = tuple((ep, b.geometry, DeviceSplit(b.val, dev, budget_bytes=budget))
                      for (ep, _, _), b in zip(extra_train, extra_consumers, strict=True))

    eye = torch.eye(d_c, device=dev, dtype=torch.float32)
    ones = torch.ones(d_c, device=dev, dtype=torch.float32)
    prot_kwargs = {
        "protection": config.protection,
        "sink_tokens": config.sink_tokens,
        "recent_tokens": config.recent_tokens,
    }

    # All initialization arms share one replay of each validation row. Preserve
    # the original per-arm stack/mean reduction and consumer order.
    baselines: dict[str, float] = {}
    with torch.no_grad():
        a_j = content_stats(bank.train, params, dev)
        base_r = None if fit_mode == "scale_only" else eye
        arms = [("rtn_native_val", base_r, ones)]
        if (d_c & (d_c - 1)) == 0:
            arms.append(("hadamard_val", sylvester_hadamard(d_c, device=dev), ones))
        alpha_scales = [_init_log_scales(a_j, alpha, config.log_scale_bound) for alpha in config.alpha_grid]
        arms += [(f"alpha:{alpha}", base_r, scale.exp()) for alpha, scale in zip(config.alpha_grid, alpha_scales)]
        values = [[] for _ in arms]
        for cp, cg, seqs in [(params, geom, val_seqs), *extra_val]:
            for seq in seqs:
                for acc, (_, rotation, scale) in zip(values, arms, strict=True):
                    acc.append(content_output_loss((seq,), cp, cg, rotation, scale, bits,
                        group_size=config.group_size, eps_scale=config.eps_scale, ste=False, **prot_kwargs))
        scores = {key: torch.stack(v).mean().item() for (key, _, _), v in zip(arms, values, strict=True)}
        baselines = {key: value for key, value in scores.items() if not key.startswith("alpha:")}
        best_alpha = 0.0
        best_alpha_loss = float("inf")
        best_init_log_s = torch.zeros(d_c, device=dev)
        for alpha, scale in zip(config.alpha_grid, alpha_scales):
            cand_loss = scores[f"alpha:{alpha}"]
            if cand_loss < best_alpha_loss - 1e-12 or (abs(cand_loss - best_alpha_loss) <= 1e-12 and alpha == 0.0):
                best_alpha, best_alpha_loss, best_init_log_s = float(alpha), cand_loss, scale
        baselines["smoothquant_alpha_val"] = best_alpha_loss

    log_s_param = best_init_log_s.clone().detach().requires_grad_(True)
    if fit_mode == "full":
        skew_param = torch.zeros((d_c, d_c), device=dev, dtype=torch.float32, requires_grad=True)
        opt_params = [
            {"params": [skew_param], "lr": config.content_rot_lr},
            {"params": [log_s_param], "lr": config.content_scale_lr},
        ]
        clip_params = [skew_param, log_s_param]
    else:
        skew_param = None
        opt_params = [{"params": [log_s_param], "lr": config.content_scale_lr}]
        clip_params = [log_s_param]
    optimizer = torch.optim.AdamW(opt_params, weight_decay=0.0)

    best_step = 0
    best_val_loss = best_alpha_loss
    best_r = eye.detach().clone()
    best_s = torch.exp(best_init_log_s).detach().clone()

    gen = torch.Generator(device="cpu").manual_seed(config.seed + geom.layer_id * 1009 + bits * 37)
    n_train = len(train_seqs)

    for step in range(1, config.steps + 1):
        lr_mult = _cosine_lr_factor(step - 1, config.steps, config.warmup_frac)
        if fit_mode == "full":
            optimizer.param_groups[0]["lr"] = config.content_rot_lr * lr_mult
            optimizer.param_groups[1]["lr"] = config.content_scale_lr * lr_mult
        else:
            optimizer.param_groups[0]["lr"] = config.content_scale_lr * lr_mult

        perm = torch.randperm(n_train, generator=gen)[: min(config.batch_size, n_train)].tolist()
        batch = tuple(train_seqs[i] for i in perm)
        batch_extra = tuple((ep, eg, tuple(es[i % len(es)] for i in perm)) for ep, eg, es in extra_train)

        optimizer.zero_grad(set_to_none=True)
        log_s = log_s_param.clamp(-config.log_scale_bound, config.log_scale_bound)
        s_c = torch.exp(log_s)
        if fit_mode == "full":
            assert skew_param is not None
            r_c = cayley_orthogonal(skew_param)
            loss = content_output_loss(
                batch, params, geom, r_c, s_c, bits,
                group_size=config.group_size, eps_scale=config.eps_scale, ste=True,
                extra_consumers=batch_extra, **prot_kwargs,
            )
            if config.anchor > 0.0:
                reg = config.anchor * ((r_c - eye).pow(2).sum() / d_c + log_s.pow(2).mean())
                loss = loss + reg
        else:
            loss = content_output_loss(
                batch, params, geom, None, s_c, bits,
                group_size=config.group_size, eps_scale=config.eps_scale, ste=True,
                extra_consumers=batch_extra, **prot_kwargs,
            )
            if config.anchor > 0.0:
                loss = loss + config.anchor * log_s.pow(2).mean()
        loss.backward()
        if config.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(clip_params, config.grad_clip)
        optimizer.step()
        with torch.no_grad():
            log_s_param.clamp_(-config.log_scale_bound, config.log_scale_bound)

        if step % config.val_every == 0 or step == config.steps:
            with torch.no_grad():
                s_eval = torch.exp(log_s_param.clamp(-config.log_scale_bound, config.log_scale_bound))
                if fit_mode == "full":
                    assert skew_param is not None
                    # Score the FP32 rotation that will ship. A single-precision solve
                    # at production rank can lose orthogonality before the FP32 cast.
                    r_eval = cayley_orthogonal(skew_param.double()).to(skew_param.dtype)
                    val_loss = float(
                        content_output_loss(
                            val_seqs, params, geom, r_eval, s_eval, bits,
                            group_size=config.group_size, eps_scale=config.eps_scale, ste=False,
                            extra_consumers=extra_val, **prot_kwargs,
                        ).item()
                    )
                else:
                    r_eval = eye
                    val_loss = float(
                        content_output_loss(
                            val_seqs, params, geom, None, s_eval, bits,
                            group_size=config.group_size, eps_scale=config.eps_scale, ste=False,
                            extra_consumers=extra_val, **prot_kwargs,
                        ).item()
                    )
                if math.isfinite(val_loss) and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = step
                    best_r = r_eval.detach().clone()
                    best_s = s_eval.detach().clone()

    orth_err = 0.0 if fit_mode == "scale_only" else float((best_r.transpose(0, 1) @ best_r - eye).abs().max().item())
    equiv_err = _check_content_equivalence(val_seqs, params, geom, best_r, best_s)
    return ContentFitResult(
        layer_id=geom.layer_id,
        bits=bits,
        r_c=best_r.cpu(),
        s_c=best_s.cpu(),
        best_step=best_step,
        best_val_loss=best_val_loss,
        init_alpha=best_alpha,
        equiv_max_rel_err=equiv_err,
        orth_max_err=orth_err,
        baselines=baselines,
        protection_calibrated=config.protection,
        init_s=torch.exp(best_init_log_s).detach().cpu(),
        fit_mode=fit_mode,
    )


def fit_rope_layer(
    bank: CapturedLayerBank,
    bits: int,
    config: FitConfig = FitConfig(),
    *,
    device: torch.device | str = "cpu",
    fit_mode: str = "full",
) -> RopeFitResult:
    """Fit per-frequency-pair SO(2) angles theta_i and pair-tied scales s_i (or scale_only s_i with theta=0)."""
    if bits not in ROPE_NONZERO_BITS:
        raise ValueError(f"fit_rope_layer requires bits in {sorted(ROPE_NONZERO_BITS)}, got {bits}")
    if fit_mode not in VALID_FIT_MODES:
        raise ValueError(f"invalid fit_mode {fit_mode!r}; expected one of {sorted(VALID_FIT_MODES)}")
    config.validate()
    bank.validate()
    geom = bank.geometry
    if not geom.has_rope:
        raise ValueError(f"layer {geom.layer_id} has no RoPE path (rope_kind={geom.rope_kind!r})")
    _require_unprotected_tokens(
        bank.train,
        protection=config.protection,
        sink_tokens=config.sink_tokens,
        recent_tokens=config.recent_tokens,
        split_name="train",
    )
    _require_unprotected_tokens(
        bank.val,
        protection=config.protection,
        sink_tokens=config.sink_tokens,
        recent_tokens=config.recent_tokens,
        split_name="val",
    )

    dev = torch.device(device)
    pairs = geom.qk_rope_dim // 2
    train_seqs = DeviceSplit(bank.train, dev)
    val_seqs = DeviceSplit(bank.val, dev, budget_bytes=split_residency_budget(dev))

    zeros = torch.zeros(pairs, device=dev, dtype=torch.float32)
    ones = torch.ones(pairs, device=dev, dtype=torch.float32)
    prot_kwargs = {
        "protection": config.protection,
        "sink_tokens": config.sink_tokens,
        "recent_tokens": config.recent_tokens,
    }

    baselines: dict[str, float] = {}
    with torch.no_grad():
        a_pair = rope_stats(bank.train, geom, dev)
        base_theta = None if fit_mode == "scale_only" else zeros
        alpha_scales = [_init_log_scales(a_pair, alpha, config.log_scale_bound) for alpha in config.alpha_grid]
        arms = [ones, *(scale.exp() for scale in alpha_scales)]
        values = [[] for _ in arms]
        for seq in val_seqs:
            for acc, scale in zip(values, arms, strict=True):
                acc.append(rope_qk_loss((seq,), geom, base_theta, scale, bits,
                    group_size=config.group_size, eps_scale=config.eps_scale, ste=False, **prot_kwargs))
        scores = [torch.stack(v).mean().item() for v in values]
        baselines["rtn_val"] = scores[0]
        best_alpha = 0.0
        best_alpha_loss = float("inf")
        best_init_log_s = torch.zeros(pairs, device=dev)
        for alpha, scale, cand_loss in zip(config.alpha_grid, alpha_scales, scores[1:], strict=True):
            if cand_loss < best_alpha_loss - 1e-12 or (abs(cand_loss - best_alpha_loss) <= 1e-12 and alpha == 0.0):
                best_alpha, best_alpha_loss, best_init_log_s = float(alpha), cand_loss, scale
        baselines["smoothquant_alpha_val"] = best_alpha_loss

    log_s_param = best_init_log_s.clone().detach().requires_grad_(True)
    if fit_mode == "full":
        theta_param = torch.zeros(pairs, device=dev, dtype=torch.float32, requires_grad=True)
        opt_params = [
            {"params": [theta_param], "lr": config.rope_angle_lr},
            {"params": [log_s_param], "lr": config.rope_scale_lr},
        ]
        clip_params = [theta_param, log_s_param]
    else:
        theta_param = None
        opt_params = [{"params": [log_s_param], "lr": config.rope_scale_lr}]
        clip_params = [log_s_param]
    optimizer = torch.optim.AdamW(opt_params, weight_decay=0.0)

    best_step = 0
    best_val_loss = best_alpha_loss
    best_theta = zeros.detach().clone()
    best_s = torch.exp(best_init_log_s).detach().clone()

    gen = torch.Generator(device="cpu").manual_seed(config.seed + geom.layer_id * 2027 + bits * 53)
    n_train = len(train_seqs)

    for step in range(1, config.steps + 1):
        lr_mult = _cosine_lr_factor(step - 1, config.steps, config.warmup_frac)
        if fit_mode == "full":
            optimizer.param_groups[0]["lr"] = config.rope_angle_lr * lr_mult
            optimizer.param_groups[1]["lr"] = config.rope_scale_lr * lr_mult
        else:
            optimizer.param_groups[0]["lr"] = config.rope_scale_lr * lr_mult

        perm = torch.randperm(n_train, generator=gen)[: min(config.batch_size, n_train)].tolist()
        batch = tuple(train_seqs[i] for i in perm)

        optimizer.zero_grad(set_to_none=True)
        log_s = log_s_param.clamp(-config.log_scale_bound, config.log_scale_bound)
        s_p = torch.exp(log_s)
        if fit_mode == "full":
            assert theta_param is not None
            loss = rope_qk_loss(
                batch, geom, theta_param, s_p, bits,
                group_size=config.group_size, eps_scale=config.eps_scale, ste=True, **prot_kwargs,
            )
            if config.anchor > 0.0:
                reg = config.anchor * (theta_param.pow(2).mean() + log_s.pow(2).mean())
                loss = loss + reg
        else:
            loss = rope_qk_loss(
                batch, geom, None, s_p, bits,
                group_size=config.group_size, eps_scale=config.eps_scale, ste=True, **prot_kwargs,
            )
            if config.anchor > 0.0:
                loss = loss + config.anchor * log_s.pow(2).mean()
        loss.backward()
        if config.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(clip_params, config.grad_clip)
        optimizer.step()
        with torch.no_grad():
            log_s_param.clamp_(-config.log_scale_bound, config.log_scale_bound)

        if step % config.val_every == 0 or step == config.steps:
            with torch.no_grad():
                s_eval = torch.exp(log_s_param.clamp(-config.log_scale_bound, config.log_scale_bound))
                if fit_mode == "full":
                    assert theta_param is not None
                    val_loss = float(
                        rope_qk_loss(
                            val_seqs, geom, theta_param, s_eval, bits,
                            group_size=config.group_size, eps_scale=config.eps_scale, ste=False, **prot_kwargs,
                        ).item()
                    )
                    theta_eval = theta_param
                else:
                    val_loss = float(
                        rope_qk_loss(
                            val_seqs, geom, None, s_eval, bits,
                            group_size=config.group_size, eps_scale=config.eps_scale, ste=False, **prot_kwargs,
                        ).item()
                    )
                    theta_eval = zeros
                if math.isfinite(val_loss) and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = step
                    best_theta = theta_eval.detach().clone()
                    best_s = s_eval.detach().clone()

    equiv_err = _check_rope_equivalence(val_seqs, geom, best_theta, best_s)
    return RopeFitResult(
        layer_id=geom.layer_id,
        bits=bits,
        theta=best_theta.cpu(),
        s_p=best_s.cpu(),
        best_step=best_step,
        best_val_loss=best_val_loss,
        init_alpha=best_alpha,
        equiv_max_rel_err=equiv_err,
        baselines=baselines,
        protection_calibrated=config.protection,
        init_s=torch.exp(best_init_log_s).detach().cpu(),
        fit_mode=fit_mode,
    )


def fit_checkpoint_stem(kind: str, bits: int, layer_id: int, *, fit_mode: str = "full") -> str:
    if fit_mode not in VALID_CHECKPOINT_MODES:
        raise ValueError(f"invalid fit_mode {fit_mode!r}; expected one of {sorted(VALID_CHECKPOINT_MODES)}")
    prefix = "c" if kind == "content" else "r"
    tag = "scale" if fit_mode == "scale_only" else fit_mode
    return f"{kind}-{tag}-{prefix}{bits}-layer-{layer_id:04d}"


def save_fit_checkpoint(
    directory: Path | str,
    result: ContentFitResult | RopeFitResult,
    *,
    identity_key: str,
) -> dict[str, Any]:
    """Save one v3 candidate fit checkpoint (full, scale_only, or smoothquant) with XXH3-128 receipt."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fit_mode = result.fit_mode
    if fit_mode not in VALID_CHECKPOINT_MODES:
        raise ValueError(f"invalid fit_mode {fit_mode!r}; expected one of {sorted(VALID_CHECKPOINT_MODES)}")
    if isinstance(result, ContentFitResult):
        kind = "content"
        stem = fit_checkpoint_stem(kind, result.bits, result.layer_id, fit_mode=fit_mode)
        classified_kind = classify_content_kind(result.r_c, result.s_c)
        tensors = {"r": result.r_c.float(), "s": result.s_c.float()}
        if result.init_s is not None:
            tensors["init_s"] = result.init_s.float()
        meta = {
            "kind": kind,
            "fit_mode": fit_mode,
            "classified_kind": classified_kind,
            "layer_id": result.layer_id,
            "bits": result.bits,
            "best_step": result.best_step,
            "best_val_loss": result.best_val_loss,
            "init_alpha": result.init_alpha,
            "equiv_max_rel_err": result.equiv_max_rel_err,
            "orth_max_err": result.orth_max_err,
            "baselines": result.baselines,
            "protection_calibrated": result.protection_calibrated,
        }
    else:
        kind = "rope"
        stem = fit_checkpoint_stem(kind, result.bits, result.layer_id, fit_mode=fit_mode)
        classified_kind = classify_rope_kind(result.theta, result.s_p)
        tensors = {"theta": result.theta.float(), "s": result.s_p.float()}
        if result.init_s is not None:
            tensors["init_s"] = result.init_s.float()
        meta = {
            "kind": kind,
            "fit_mode": fit_mode,
            "classified_kind": classified_kind,
            "layer_id": result.layer_id,
            "bits": result.bits,
            "best_step": result.best_step,
            "best_val_loss": result.best_val_loss,
            "init_alpha": result.init_alpha,
            "equiv_max_rel_err": result.equiv_max_rel_err,
            "baselines": result.baselines,
            "protection_calibrated": result.protection_calibrated,
        }
    payload = _write_safetensors_streamed(directory / f"{stem}.safetensors", tensors)
    record = {
        "schema": FIT_CHECKPOINT_SCHEMA,
        "identity_key": identity_key,
        "meta": meta,
        "payload": payload,
    }
    _atomic_json(directory / f"{stem}.json", record)
    return record


def load_fit_checkpoint(
    directory: Path | str,
    *,
    kind: str,
    bits: int,
    layer_id: int,
    identity_key: str,
    fit_mode: str = "full",
) -> ContentFitResult | RopeFitResult | None:
    """Reload a verified v3 fit checkpoint (rejecting any legacy schema or identity mismatch)."""
    directory = Path(directory)
    stem = fit_checkpoint_stem(kind, bits, layer_id, fit_mode=fit_mode)
    receipt_path = directory / f"{stem}.json"
    if not receipt_path.exists():
        return None
    record = json.loads(receipt_path.read_text(encoding="utf-8"))
    if record.get("schema") != FIT_CHECKPOINT_SCHEMA:
        raise ValueError(
            f"unsupported fit checkpoint schema: {record.get('schema')!r} (expected {FIT_CHECKPOINT_SCHEMA!r})"
        )
    if record.get("identity_key") != identity_key:
        raise ValueError(f"fit checkpoint identity mismatch for {stem}: refuse stale resume")
    meta = record["meta"]
    if meta.get("fit_mode") != fit_mode:
        raise ValueError(f"fit checkpoint {stem} fit_mode={meta.get('fit_mode')!r} != requested {fit_mode!r}")
    payload = record["payload"]
    tensors = _read_safetensors_verified(
        directory / payload["file"], int(payload["bytes"]), str(payload["xxh3_128"])
    )
    if kind == "content":
        return ContentFitResult(
            layer_id=int(meta["layer_id"]),
            bits=int(meta["bits"]),
            r_c=tensors["r"],
            s_c=tensors["s"],
            best_step=int(meta["best_step"]),
            best_val_loss=float(meta["best_val_loss"]),
            init_alpha=float(meta["init_alpha"]),
            equiv_max_rel_err=float(meta["equiv_max_rel_err"]),
            orth_max_err=float(meta["orth_max_err"]),
            baselines=dict(meta["baselines"]),
            protection_calibrated=str(meta["protection_calibrated"]),
            init_s=tensors.get("init_s"),
            fit_mode=fit_mode,
        )
    return RopeFitResult(
        layer_id=int(meta["layer_id"]),
        bits=int(meta["bits"]),
        theta=tensors["theta"],
        s_p=tensors["s"],
        best_step=int(meta["best_step"]),
        best_val_loss=float(meta["best_val_loss"]),
        init_alpha=float(meta["init_alpha"]),
        equiv_max_rel_err=float(meta["equiv_max_rel_err"]),
        baselines=dict(meta["baselines"]),
        protection_calibrated=str(meta["protection_calibrated"]),
        init_s=tensors.get("init_s"),
        fit_mode=fit_mode,
    )


def fit_config_digest(config: FitConfig, extra: dict[str, Any] | None = None) -> str:
    payload = {"config": asdict(config), "extra": extra or {}}
    return xxhash.xxh3_128_hexdigest(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))

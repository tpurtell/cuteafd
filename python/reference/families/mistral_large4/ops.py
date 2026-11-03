"""Mistral Large 4 op-level references for engine kernel unit numerics.

Not a model: the whole-model reference is the shared ``model.py`` (the
fidelity golden's implementation; top-k selection and FP8 dequantization
delegate to it). These are the individual operations the
coordinator programs fuse differently from the official modeling, written
from transformers ``modeling_mistral_large4.py`` (PR #50121, main 66f51e76,
2026-10-30) and checked against Mistral's vLLM port (vllm#64534,
NickLucche/vllm@bd4fa263 ``vllm/models/mistral``). Each function names the
reference site it reproduces; kernels are compared against these on
synthetic shapes before the checkpoint is available.

Conventions: ``[rows, heads, dim]`` activations, BF16 in and out unless
stated; ``positions`` int64 ``[rows]``.
"""
from __future__ import annotations

import math

import torch


# --- RoPE: interleaved YaRN with LeRoPE ---------------------------------------

def yarn_lerope_inv_freq(dim: int = 64, theta: float = 1e4, factor: float = 2048.0,
                         original_max_position: int = 512, beta_fast: float = 32.0,
                         beta_slow: float = 1.0, alpha: torch.Tensor | None = None) -> torch.Tensor:
    """FP32 ``inv_freq [dim/2]``.

    HF: ``_compute_yarn_parameters`` (truncate=True, attention_factor 1.0),
    then ``MistralLarge4TextRotaryEmbedding.forward`` multiplies by
    ``alpha.exp()`` per call. vLLM ``MistralRotaryEmbedding._compute_inv_freq``
    scales both YaRN branches by ``exp(alpha)`` before the ramp blend, which
    is the same product (the blend is linear). The draft model uses
    ``alpha=None`` (plain YaRN).
    """
    pos_freqs = theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    extrapolation = 1.0 / pos_freqs
    interpolation = 1.0 / (factor * pos_freqs)

    def correction_dim(rotations):
        return (dim * math.log(original_max_position / (rotations * 2 * math.pi))) / (2 * math.log(theta))

    low = max(math.floor(correction_dim(beta_fast)), 0)
    high = min(math.ceil(correction_dim(beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
    extrapolation_factor = 1 - ramp
    inv_freq = interpolation * (1 - extrapolation_factor) + extrapolation * extrapolation_factor
    if alpha is not None:
        inv_freq = inv_freq * alpha.float().exp()
    return inv_freq


def cos_sin(positions: torch.Tensor, inv_freq: torch.Tensor, dtype=torch.bfloat16):
    """HF ``forward``: FP32 angles, cos/sin cast to the activation dtype (BF16)."""
    freqs = positions.float()[:, None] * inv_freq.to(positions.device)[None, :]
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def rope_interleave(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """HF ``apply_rotary_pos_emb_interleave`` on ``x [rows, heads, 2*P]``:
    pairs ``(x[2i], x[2i+1])`` rotate by angle ``i``; the output is
    de-interleaved ``[even' (P) | odd' (P)]`` in the activation dtype."""
    c, s = cos[:, None, :], sin[:, None, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.cat([even * c - odd * s, odd * c + even * s], dim=-1)


# --- Scalable softmax ------------------------------------------------------------

def ssmax_scale(positions: torch.Tensor, s: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``[rows, heads]`` BF16 ``s * log(pos + 1) + b``; HF
    ``MistralLarge4TextSoftmaxScaling``: ``log_pos`` in FP32, then
    ``log_pos, s, b`` each cast to BF16 and the product and sum rounded in BF16
    (vLLM's fused kernel rounds every step the same way)."""
    log_pos = (positions + 1).float().log().to(torch.bfloat16)
    s, b = s.to(torch.bfloat16), b.to(torch.bfloat16)
    return s[None, :] * log_pos[:, None] + b[None, :]


def ssmax(q: torch.Tensor, positions: torch.Tensor, s: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``q [rows, heads, D]`` BF16 times the per-(row, head) BF16 scale."""
    return q * ssmax_scale(positions, s, b)[..., None]


# --- MLA query and latent ----------------------------------------------------------

def mla_query(q: torch.Tensor, positions: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
              s: torch.Tensor, b: torch.Tensor, nope: int, rope: bool) -> torch.Tensor:
    """``wq(x)`` viewed ``[rows, 64, 128]`` -> the scaled query the attention
    consumes, ``[q_nope | rope(q_rot)] * ssmax``.

    HF ``MistralLarge4TextAttention.forward``: split ``[nope, rope]``
    (RoPE layers 64 + 64; NoPE layers 128 + 0), interleaved RoPE on the rope
    part only, concatenate, then SSMax over the whole 128-wide head.
    """
    if not rope:
        return ssmax(q, positions, s, b)
    q_pass, q_rot = q[..., :nope], q[..., nope:]
    return ssmax(torch.cat([q_pass, rope_interleave(q_rot, cos, sin)], dim=-1), positions, s, b)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """HF ``MistralLarge4RMSNorm``: FP32 statistics, BF16 product order
    ``weight * x_normed.to(bf16)``."""
    dtype = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(dtype)


def mla_latent(kv_a: torch.Tensor, kv_a_norm: torch.Tensor, positions: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor, eps: float, rope: bool):
    """``wkv_a_with_mqa(x)`` -> (normed latent ``[rows, 512]``, rotated key
    ``[rows, 64]`` or ``None`` on NoPE layers)."""
    latent = rms_norm(kv_a[..., :512], kv_a_norm, eps)
    if not rope:
        return latent, None
    return latent, rope_interleave(kv_a[..., None, 512:576], cos, sin)[:, 0]


# --- DSA indexer -------------------------------------------------------------------

def indexer_scores(x: torch.Tensor, positions: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                   q_norm_w, q_norm_b, w_q, w_k, k_norm_w, k_norm_b, w_weights, s, b,
                   heads: int = 32, head_dim: int = 128, rope_dim: int = 64,
                   rope: bool = True, eps: float = 1e-6) -> torch.Tensor:
    """FP32 ``[rows, keys]`` index scores, causal (``-inf`` above the diagonal);
    top-k selects from them. HF ``MistralLarge4TextIndexer.forward`` with the
    keys of the same rows (one prompt, no cache):

    q = ssmax(wq_b(LayerNorm(x)))  [rows, H, 128], RoPE (interleaved) on dims
    0:64 when the layer uses RoPE; k = LayerNorm(wk(x)), same RoPE;
    score = sum_h (weights_proj(x) * H^-0.5)_h * relu(q_h . k / sqrt(128)).
    """
    import torch.nn.functional as F
    rows = x.shape[0]
    q = F.layer_norm(x, (x.shape[-1],), q_norm_w, q_norm_b, eps) @ w_q.t()
    q = ssmax(q.view(rows, heads, head_dim), positions, s, b)
    k = F.layer_norm(x @ w_k.t(), (head_dim,), k_norm_w, k_norm_b, eps)[:, None, :]
    if rope:
        q = torch.cat([rope_interleave(q[..., :rope_dim], cos, sin), q[..., rope_dim:]], -1)
        k = torch.cat([rope_interleave(k[..., :rope_dim], cos, sin), k[..., rope_dim:]], -1)
    scores = torch.relu(torch.einsum("qhd,kd->qhk", q.float(), k[:, 0].float()) * head_dim ** -0.5)
    weights = (x.to(w_weights.dtype) @ w_weights.t()).float() * heads ** -0.5
    index = torch.einsum("qh,qhk->qk", weights, scores)
    causal = torch.ones(rows, rows, dtype=torch.bool, device=x.device).tril()
    return index.masked_fill(~causal, float("-inf"))


def index_topk(scores: torch.Tensor, slots: int) -> torch.Tensor:
    """The engine's selection rule, the shared model.py's (``fixed_index_topk``:
    highest score first, exact ties (common after relu) to the lower key id,
    the selected ids ascending, ``-inf`` padding past the causal extent). The
    official ``topk`` leaves tie order undefined."""
    return _model().fixed_index_topk(scores, slots).indices


# --- Router and head ---------------------------------------------------------------

def router(x: torch.Tensor, gate: torch.Tensor, topk: int = 4, scale: float = 1.0):
    """HF ``MistralLarge4TextTopkRouter`` (n_group 1): FP32 logits and
    softmax, top-k of the probabilities, renormalized (``+ 1e-20``), times
    ``routed_scaling_factor``. Returns (weights FP32 ``[rows, k]``, ids)."""
    probs = torch.nn.functional.linear(x.float(), gate.float()).softmax(-1)
    ids = probs.topk(topk, dim=-1, sorted=False).indices
    weights = probs.gather(1, ids)
    return weights / (weights.sum(-1, keepdim=True) + 1e-20) * scale, ids


def softcap(logits: torch.Tensor, cap: float = 30.0) -> torch.Tensor:
    """HF ``MistralLarge4ForConditionalGeneration.forward``: ``tanh(x / c) * c``."""
    return torch.tanh(logits / cap) * cap


def fp8_block_dequant(weight: torch.Tensor, qscale: torch.Tensor) -> torch.Tensor:
    """Official ``qscale_weight`` dequantization, the shared model.py's
    (``Fp8Dequantize`` arithmetic, power-of-two scales verified)."""
    return _model().dequantize(weight, qscale, torch.float32)


def _model():
    """The shared reference (model.py) and its sibling helpers on the path."""
    import sys
    from pathlib import Path
    here = Path(__file__).resolve().parent
    for path in (here, here.parents[1]):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import model
    return model

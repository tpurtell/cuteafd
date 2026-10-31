"""Calibration stages around unchanged pinned official Mistral Large 4 math.

The producer installs serialized/reconstructed weights between these stages.
Exact sparse selections and layer 35's compressed KV travel with each batch;
regenerating them from later hidden states would invalidate causal calibration.
"""
from __future__ import annotations

import torch

from .source import linear_module, layer_mapping

STATE_KEYS = ("prev_topk_indices", "kv_nope_pass", "kv_nope_rot", "position_ids")


def __getattr__(name):
    if name in ("DenseRawHessian", "quantize_projection", "reconstruct_projection", "validate_projection"):
        # Importing source/native for headers or source audits must not import
        # GPTQModel or initialize its CUDA extension machinery.
        from . import numerical
        return getattr(numerical, name)
    raise AttributeError(name)


def layer_names(block, layer=None):
    layer = block.layer_idx if layer is None else layer
    return {source.removesuffix(".weight"): parameter.removesuffix(".weight")
            for parameter, source in layer_mapping(block.self_attn.config, layer, block).items()
            if isinstance(source, str) and source.endswith(".weight")}


@torch.inference_mode()
def prepare_context(hidden, rotary, prev_topk=None, *, position_ids=None,
                    shared_kv_states=None, attention_mask=None):
    if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] < 1:
        raise ValueError("Mistral calibration requires one nonempty unpadded sequence")
    batch, length, _ = hidden.shape
    if position_ids is None:
        position_ids = torch.arange(length, device=hidden.device)[None]
    if (position_ids.dtype not in (torch.int32, torch.int64)
            or tuple(position_ids.shape) != (batch, length)
            or bool((position_ids < 0).any())):
        raise ValueError("invalid Mistral absolute position IDs")
    position_ids = position_ids.to(hidden.device)
    if attention_mask is None:
        indices = torch.arange(length, device=hidden.device)
        future = indices[None, :] > indices[:, None]
        attention_mask = torch.zeros((length, length), device=hidden.device, dtype=hidden.dtype)
        attention_mask.masked_fill_(future, torch.finfo(hidden.dtype).min)
        attention_mask = attention_mask[None, None]
    if (tuple(attention_mask.shape) != (batch, 1, length, length)
            or not attention_mask.is_floating_point()):
        raise ValueError("Mistral eager attention requires an additive causal mask")
    if prev_topk is not None:
        if (prev_topk.ndim != 3 or tuple(prev_topk.shape[:2]) != (batch, length)
                or prev_topk.shape[-1] > length or prev_topk.shape[-1] < 1
                or prev_topk.dtype not in (torch.int32, torch.int64)
                or bool((prev_topk < 0).any()) or bool((prev_topk >= length).any())):
            raise ValueError("invalid exact shared Mistral DSA top-k state")
        prev_topk = prev_topk.to(hidden.device)
    return dict(position_embeddings=rotary(hidden, position_ids), attention_mask=attention_mask.to(hidden.device),
                position_ids=position_ids, shared_kv_states=shared_kv_states if shared_kv_states is not None else {},
                prev_topk_indices=prev_topk)


class _Captured(BaseException):
    def __init__(self, owner):
        self.owner = owner


@torch.inference_mode()
def capture_linear_input(target, forward, *args, **kwargs):
    captured = {}

    def retain(module, positional, named):
        value = positional[0] if positional else named.get("input")
        if not isinstance(value, torch.Tensor):
            raise TypeError("captured native Mistral linear input must be a tensor")
        captured["value"] = value.detach()
        raise _Captured(captured)

    handle = target.register_forward_pre_hook(retain, with_kwargs=True)
    try:
        try:
            forward(*args, **kwargs)
        except _Captured as error:
            if error.owner is not captured:
                raise
        else:
            raise RuntimeError("native Mistral forward did not reach the selected linear")
    finally:
        handle.remove()
    return captured["value"]


@torch.inference_mode()
def attention_inputs(block, hidden):
    return block.input_layernorm(hidden)


attention_entry = attention_inputs


def _shared_from_state(state):
    if ("kv_nope_pass" in state) != ("kv_nope_rot" in state):
        raise ValueError("incomplete exact Mistral KV-source state")
    return {} if "kv_nope_pass" not in state else {"nope": (state["kv_nope_pass"], state["kv_nope_rot"])}


@torch.inference_mode()
def attention_capture(block, normalized, rotary, state=None):
    """Return native o-projection input and complete exact reuse state.

    Capture aborts immediately before o_proj. The official attention has then
    stored the source latent (layer35) and selected its exact DSA top-k. The
    source pair has shape [1,1,T,512]/[1,1,T,0] and must survive every successor
    layer, including layers that neither produce nor consume it.
    """
    state = {} if state is None else state
    context = prepare_context(normalized, rotary, state.get("prev_topk_indices"),
        position_ids=state.get("position_ids"), shared_kv_states=_shared_from_state(state))
    captured = {}
    indexer = block.self_attn.indexer
    handle = None
    if indexer is not None:
        handle = indexer.register_forward_hook(lambda module, args, output: captured.update(topk=output.detach()))
    try:
        value = capture_linear_input(block.self_attn.o_proj, block.self_attn,
            hidden_states=normalized, past_key_values=None, **context)
    finally:
        if handle is not None:
            handle.remove()
    topk = captured.get("topk", state.get("prev_topk_indices"))
    if topk is None:
        raise ValueError("native Mistral attention lost exact top-k selection")
    persistent = dict(prev_topk_indices=topk)
    if "position_ids" in state:
        persistent["position_ids"] = state["position_ids"]
    shared = context["shared_kv_states"]
    if set(shared) - {"nope"}:
        raise ValueError("unreviewed Mistral KV sharing class")
    if "nope" in shared:
        persistent["kv_nope_pass"], persistent["kv_nope_rot"] = shared["nope"]
    if not bool(torch.isfinite(value).all()):
        raise ValueError("nonfinite native Mistral attention output input")
    return value, persistent


@torch.inference_mode()
def attention_forward(block, normalized, context):
    output, _, topk = block.self_attn(hidden_states=normalized, past_key_values=None, **context)
    return output, topk


@torch.inference_mode()
def attention_finish(block, residual, attention_output):
    residual = residual + attention_output
    return residual, block.post_attention_layernorm(residual)


@torch.inference_mode()
def shared_down_input(shared, hidden):
    return shared.act_fn(shared.gate_proj(hidden)) * shared.up_proj(hidden)


@torch.inference_mode()
def router_choices(moe, hidden, *, rank_max=8):
    """Official top-four routes; next four learned ranks recover cold experts."""
    logits, weights, selected = moe.gate(hidden)
    if (selected.dtype != torch.int64 or selected.ndim != 2 or selected.shape != weights.shape
            or selected.shape[1] != 4 or rank_max != 8
            or moe.gate.num_group != 1 or moe.gate.topk_group != 1):
        raise ValueError("Mistral recovery requires the reviewed top-four ungrouped router")
    # Preserve the router's original selected order and normalization. Recovery
    # is separate and never changes a natural route or the successor reduction.
    scores = logits.softmax(-1)
    available = scores.scatter(1, selected, float("-inf"))
    near = torch.topk(available, 4, dim=-1, sorted=True).indices
    return selected, weights, near


@torch.inference_mode()
def complete_moe(residual, normalized, moe):
    logits, weights, selected = moe.gate(normalized)
    shape = normalized.shape
    routed = moe.experts(normalized.reshape(-1, shape[-1]), selected, weights).view(shape)
    return residual + (routed + moe.shared_experts(normalized))


@torch.inference_mode()
def expert_weight(block, expert, part):
    experts = block.mlp.experts
    if part in ("w2", "down_proj"):
        return experts.down_proj[expert]
    inter = experts.gate_up_proj.shape[1] // 2
    if part in ("w1", "gate_proj"):
        return experts.gate_up_proj[expert, :inter]
    if part in ("w3", "up_proj"):
        return experts.gate_up_proj[expert, inter:]
    raise ValueError("unknown native Mistral routed projection")

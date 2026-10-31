"""Checkpoint-native Muse Spark stages for causal EXL3 calibration.

The official Transformers decoder owns every norm, gate, routing decision and
attention operation.  This adapter exposes its linear boundaries and loads one
layer at a time; it does not approximate Muse's per-expert output norm.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

PREFIX = "model.language_model."
TRANSFORMERS_VERSION = "5.22.1"
SOURCE_PINS = {
    "models/muse_spark/modeling_muse_spark.py": "969ef27d0187c6f59dc2c8511449d739c790bc0adf4059d88e1d999f48b155d5",
    "models/muse_spark/configuration_muse_spark.py": "9320c4cb5c89f5b42f2d8cb61d835ad92c3659d045429c84bf5e0a699dbdad1f",
    "masking_utils.py": "931b92cf0deeb7a4925259a30c905bbf4d645fda88c6eebbc9e61770f5abaa89",
}
EXPERT_STACKS = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")


def source_identity():
    """Fail closed on the exact official model release reviewed for this job."""
    import transformers
    root = Path(transformers.__file__).resolve().parent
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise ValueError(f"Muse quant requires transformers {TRANSFORMERS_VERSION}")
    actual = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_PINS}
    if actual != SOURCE_PINS:
        raise ValueError("official Muse Spark modeling source differs from the pinned release")
    return dict(transformers=transformers.__version__, sources=actual,
                adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                attention="official-sdpa", experts="official-eager",
                torch=str(torch.__version__))


def _official():
    from transformers.models.muse_spark import modeling_muse_spark
    return modeling_muse_spark


def native_config(source):
    from transformers.models.muse_spark.configuration_muse_spark import MuseSparkTextConfig
    source_identity()
    document = getattr(source, "document", None)
    if document is None:
        document = json.loads((Path(source.snapshot) / "config.json").read_bytes())
    config = MuseSparkTextConfig(**document["text_config"])
    config._attn_implementation = "sdpa"
    config._experts_implementation = "eager"
    config.use_cache = False
    return config


def module(block, relative_name):
    return block.get_submodule(relative_name)


def _assign(owner, key, value):
    parent, _, leaf = key.rpartition(".")
    target = owner.get_submodule(parent) if parent else owner
    if leaf in target._buffers:
        target._buffers[leaf] = value
    else:
        target._parameters[leaf] = nn.Parameter(value, requires_grad=False)


@torch.inference_mode()
def load_block(source, layer, device="cuda:0", *, include_experts=False):
    config = native_config(source)
    if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < config.num_hidden_layers:
        raise ValueError("Muse layer outside the official decoder schedule")
    with torch.device("meta"):
        block = _official().MuseSparkTextDecoderLayer(config, layer)
    prefix = f"{PREFIX}layers.{layer}."
    try:
        for key, expected in list(block.named_parameters()) + list(block.named_buffers()):
            if key in EXPERT_STACKS:
                continue
            value = source.tensor(prefix + key, device)
            if value.shape != expected.shape or value.dtype not in (torch.bfloat16, torch.float32):
                raise ValueError(f"native Muse shape/dtype differs: {prefix + key}")
            # The checkpoint stores these gate vectors in BF16; official HF
            # loading keeps the residual-gate modules in FP32 for execution.
            if key.endswith("residual_gate.gate"):
                value = value.float()
            if key in ("mlp.gate.weight", "mlp.gate.e_score_correction_bias") and value.dtype != torch.float32:
                raise ValueError("Muse router weight and correction bias must remain FP32")
            _assign(block, key, value)
        block.layer_idx = layer
        if include_experts:
            materialize_routed_experts(source, block, layer, device)
    finally:
        source.close()
    return block.eval()


@torch.inference_mode()
def materialize_routed_experts(source, block, layer=None, device="cuda:0"):
    """Fill the two official stacks one expert at a time, bounding host RAM."""
    layer = block.layer_idx if layer is None else layer
    experts = block.mlp.experts
    for key in ("gate_up_proj", "down_proj"):
        value = getattr(experts, key)
        if value.is_meta:
            experts._parameters[key] = nn.Parameter(torch.empty(value.shape,
                dtype=torch.bfloat16, device=device), requires_grad=False)
    try:
        for expert in range(block.config.num_local_experts):
            for part in ("gate_proj", "up_proj", "down_proj"):
                value = source.expert(layer, expert, part, device=device)
                target = expert_weight(block, expert, part)
                if value.shape != target.shape or value.dtype != torch.bfloat16:
                    raise ValueError("native Muse expert shape/dtype differs")
                target.copy_(value)
                del value
    finally:
        source.close()
    return block


def expert_weight(block, expert, part):
    experts = block.mlp.experts
    if part == "down_proj":
        return experts.down_proj[expert]
    width = block.config.moe_intermediate_size
    if part == "gate_proj":
        return experts.gate_up_proj[expert, :width]
    if part == "up_proj":
        return experts.gate_up_proj[expert, width:]
    raise ValueError("unknown Muse expert projection")


@torch.inference_mode()
def embedding(source, device="cuda:0"):
    config = native_config(source)
    with torch.device("meta"):
        value = _official().MuseSparkTextNormedEmbedding(config.vocab_size, config.hidden_size,
            config.pad_token_id, config.rms_norm_eps)
    try:
        _assign(value, "weight", source.tensor(PREFIX + "embed_tokens.weight", device))
    finally:
        source.close()
    return value.eval()


def rotary(source, device="cuda:0"):
    config = native_config(source)
    with torch.device(device):
        return _official().MuseSparkTextRotaryEmbedding(config).eval()


def prepare_context(block, hidden, rotary=None, *, position_ids=None):
    """One unpadded prompt, exact native causal/window visibility and RoPE."""
    if hidden.ndim != 3 or hidden.shape[0] != 1:
        raise ValueError("Muse calibration requires one unpadded [1,T,H] prompt")
    length = hidden.shape[1]
    if position_ids is None:
        position_ids = torch.arange(length, device=hidden.device)[None]
    expected_positions = torch.arange(length, device=hidden.device)[None]
    if position_ids.shape != expected_positions.shape or not torch.equal(position_ids, expected_positions):
        raise ValueError("Muse calibration positions must start at zero and be contiguous")
    index = block.layer_idx
    config = block.config
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    mask_function = (create_sliding_window_causal_mask
        if config.layer_types[index] == "sliding_attention" else create_causal_mask)
    # Use the model's mask builder, including its SDPA is_causal fast path.
    # A hand-created additive mask would force a different native kernel route
    # for full attention and for sliding prompts shorter than the window.
    mask = mask_function(config=config, inputs_embeds=hidden, attention_mask=None,
                         past_key_values=None, position_ids=position_ids)
    embeddings = None
    if config.layer_rope_theta[index]:
        if rotary is None:
            with torch.device(hidden.device):
                rotary = _official().MuseSparkTextRotaryEmbedding(config)
        embeddings = rotary(hidden, position_ids)
    return dict(position_embeddings=embeddings, attention_mask=mask,
                position_ids=position_ids, past_key_values=None)


@torch.inference_mode()
def attention_inputs(block, hidden):
    return block.input_layernorm(hidden)


@torch.inference_mode()
def attention_forward(block, normalized, context):
    return block.self_attn(normalized, **context)[0]


@torch.inference_mode()
def attention_output_input(block, normalized, context):
    """Observe the exact official per-head norm/output-gate o_proj operand."""
    captured = []
    handle = block.self_attn.o_proj.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach()))
    try:
        attention_forward(block, normalized, context)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError("official Muse attention did not call o_proj exactly once")
    return captured[0]


@torch.inference_mode()
def attention_finish(block, residual, output):
    residual = block.post_attention_residual_gate(residual, block.post_attention_layernorm(output))
    return residual, block.pre_feedforward_layernorm(residual)


@torch.inference_mode()
def expert_input(block, normalized):
    return block.mlp.pre_expert_norm(block.mlp.pre_expert_proj(normalized.reshape(-1, normalized.shape[-1])))


pre_expert_input = expert_input


@torch.inference_mode()
def router_choices(moe, hidden, *, rank_max=16):
    logits, weights, selected = moe.gate(hidden)
    if rank_max != 16 or selected.dtype != torch.int64 or selected.shape != weights.shape or selected.shape[1] != 8:
        raise ValueError("Muse recovery requires the official top8 and corrected ranks 9..16")
    scores = moe.gate.scoring_function(logits) + moe.gate.e_score_correction_bias
    native = torch.topk(scores, k=8, dim=-1, sorted=False).indices
    if not torch.equal(native, selected):
        raise ValueError("Muse recovery scoring differs from official top8 routing")
    # Remove the live selected set before ranking near misses. This also keeps
    # ties from accidentally duplicating an already selected expert.
    choice = scores.clone()
    choice.scatter_(1, selected, -torch.inf)
    near = torch.topk(choice, k=8, dim=-1, sorted=True).indices
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(weights).all()):
        raise ValueError("nonfinite native Muse routing")
    return selected, weights, near


@torch.inference_mode()
def expert_down_input(block, gate_weight, up_weight, hidden):
    return block.mlp.experts.act_fn(torch.nn.functional.linear(hidden, gate_weight)) * torch.nn.functional.linear(hidden, up_weight)


@torch.inference_mode()
def expert_output(block, hidden, down_weight):
    """Apply the official shared per-layer post norm before any routing weight."""
    return block.mlp.experts.post_expert_norm(torch.nn.functional.linear(hidden, down_weight))


@torch.inference_mode()
def mlp_finish(block, residual, output):
    return block.post_feedforward_residual_gate(residual, block.post_feedforward_layernorm(output))


@torch.inference_mode()
def finish_layer(block, residual, normalized):
    return mlp_finish(block, residual, block.mlp(normalized))


@torch.inference_mode()
def final_norm(source, hidden):
    config = native_config(source)
    with torch.device("meta"):
        norm = _official().MuseSparkRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    try:
        _assign(norm, "weight", source.tensor(PREFIX + "norm.weight", hidden.device))
    finally:
        source.close()
    return norm(hidden)


@torch.inference_mode()
def native_forward(block, state, rotary=None):
    context = prepare_context(block, state["hidden"], rotary, position_ids=state.get("position_ids"))
    return dict(hidden=block(state["hidden"], **context))


@torch.inference_mode()
def staged_forward(block, state, rotary=None):
    hidden = state["hidden"]
    context = prepare_context(block, hidden, rotary, position_ids=state.get("position_ids"))
    normalized = attention_inputs(block, hidden)
    output = attention_forward(block, normalized, context)
    residual, normalized = attention_finish(block, hidden, output)
    return dict(hidden=finish_layer(block, residual, normalized))


def configure_math():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _codec():
    # Importing the codec may initialize the fork package; keep header-only
    # inspection and official-module proofs independent of that dependency.
    from . import codec
    return codec


def DenseRawHessian(*args, **kwargs):
    return _codec().DenseRawHessian(*args, **kwargs)


def quantize_projection(*args, **kwargs):
    return _codec().quantize_projection(*args, **kwargs)


def reconstruct_projection(*args, **kwargs):
    return _codec().reconstruct_projection(*args, **kwargs)


def validate_projection(*args, **kwargs):
    return _codec().validate_projection(*args, **kwargs)

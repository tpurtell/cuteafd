"""Independent IEEE-F32 Inkling 2 text oracle, with native MXFP8 expert QDQ.

Arithmetic follows the pinned vLLM and SGLang vendor integrations in sources.py.
No engine imports, vision tower, drafter, or serving-kernel approximations.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct

import torch
import torch.nn.functional as F

from shape_invariant import ROWS, linear

CHECKPOINT = "thinkingmachines/Inkling-2-Small-MXFP8"
REVISION = "3febc62259ee92ea36e2fb6e8303371c4eb3922c"
TEXT = "model.llm."


@dataclass(frozen=True)
class Config:
    text: dict

    @classmethod
    def from_dict(cls, root):
        c = root["text_config"]
        if (root.get("model_type") != "inkling2" or c.get("q_bias", False)
                or c.get("o_bias", False) or c.get("pre_norm_pre_affine", False)
                or c.get("final_norm_pre_affine", False)
                or not c["use_value_norm_no_gamma"] or not c["use_gate_bias"]
                or c["n_shared_experts"] != 0 or not c["attn_res_normalize_values"]):
            raise ValueError("unsupported Inkling 2 text semantics")
        for name in ("hidden_size", "intermediate_size", "num_hidden_layers", "vocab_size",
                     "unpadded_vocab_size", "num_attention_heads", "num_key_value_heads",
                     "head_dim", "swa_num_attention_heads", "swa_num_key_value_heads",
                     "swa_head_dim", "d_rel", "rel_extent", "sliding_window_size",
                     "sconv_kernel_size", "n_layers_per_attn_res_block", "n_routed_experts",
                     "num_experts_per_tok"):
            if type(c[name]) is not int or c[name] <= 0:
                raise ValueError("invalid Inkling 2 dimension: " + name)
        if (c["hidden_size"] % 32 or c["intermediate_size"] % 32
                or c["num_attention_heads"] % c["num_key_value_heads"]
                or c["swa_num_attention_heads"] % c["swa_num_key_value_heads"]
                or c["num_experts_per_tok"] > c["n_routed_experts"]
                or c["unpadded_vocab_size"] > c["vocab_size"]
                or c["n_layers_per_attn_res_block"] > c["num_hidden_layers"]
                or len(set(c["local_layer_ids"])) != len(c["local_layer_ids"])
                or any(type(i) is not int or not 0 <= i < c["num_hidden_layers"]
                       for i in c["local_layer_ids"])):
            raise ValueError("unsupported Inkling 2 geometry")
        for name in ("rms_norm_eps", "qk_l2_clip_threshold", "rel_l2_clip_threshold",
                     "mlp_gate_softcap", "mlp_up_softcap", "route_scale", "final_logit_softcapping"):
            if not 0 < c[name] < float("inf"):
                raise ValueError("invalid Inkling 2 scale: " + name)
        floor = c.get("log_scaling_n_floor")
        if floor is not None and (type(floor) is not int or floor <= 0
                                  or not 0 <= c["log_scaling_alpha"] < float("inf")):
            raise ValueError("invalid Inkling 2 log scaling")
        return cls(c)

    def __getattr__(self, name):
        return self.text[name]

    def geometry(self, index):
        local = index in self.local_layer_ids
        return ((self.swa_num_attention_heads, self.swa_num_key_value_heads, self.swa_head_dim)
                if local else (self.num_attention_heads, self.num_key_value_heads, self.head_dim))


def rms_norm(x, weight, eps):
    value = x.float()
    result = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
    return result if weight is None else result * weight.float()


def l2_clip(x, threshold, eps, scale=None):
    value = x.float() if scale is None else x.float() * scale.float()
    return value * (threshold / (value.norm(dim=-1, keepdim=True) + eps)).clamp(max=1.)


def attn_res(blocks, partial, query, eps, addend=None):
    """Normalize keys AND values; softmax over completed blocks plus partial."""
    values = [b.float() for b in blocks]
    values.append(partial.float() if addend is None else partial.float() + addend.float())
    values = torch.stack([rms_norm(v, None, eps) for v in values])
    queries = query.float().reshape(-1, partial.shape[-1]) * partial.shape[-1] ** -.5
    result = []
    for q in queries:
        scores = (values * q).sum(-1).transpose(0, 1).softmax(-1)
        result.append((values * scores.transpose(0, 1)[..., None]).sum(0))
    return tuple(result)


def short_conv(x, weight, dynamic=None):
    """Static taps are oldest-first; dynamic static_w is lag-first (lag 0=current)."""
    if dynamic is None:
        taps = weight.reshape(x.shape[-1], -1)
        width = taps.shape[-1]
    else:
        width = weight.shape[0]
        if (dynamic.ndim != 3 or dynamic.shape[0] != x.shape[0]
                or dynamic.shape[2] != width or x.shape[1] % dynamic.shape[1]):
            raise ValueError("invalid dynamic convolution geometry")
    padded = F.pad(x.float(), (0, 0, width - 1, 0))
    result = torch.zeros_like(x, dtype=torch.float32)
    for lag in range(width):
        value = padded[width - 1 - lag:width - 1 - lag + len(x)]
        if dynamic is None:
            tap = taps[:, width - 1 - lag].float()
        else:
            tap = weight[lag].float() + dynamic[:, :, lag].float().repeat_interleave(
                x.shape[1] // dynamic.shape[1], dim=1)
        result += value * tap
    return (result + x.float()) * (width * (2 if dynamic is not None else 1) + 1) ** -.5


def dequant_mxfp8(weight, scale):
    if (weight.dtype != torch.float8_e4m3fn or weight.ndim != 2 or weight.shape[-1] % 32
            or scale.dtype != torch.uint8 or tuple(scale.shape) != (weight.shape[0], weight.shape[1] // 32)
            or torch.any(scale == 255)):
        raise ValueError("invalid Inkling MXFP8 E4M3/UE8M0 layout")
    result = torch.ldexp(weight.float().reshape(*scale.shape, 32), scale.int()[..., None] - 127)
    if not torch.isfinite(result).all():
        raise ValueError("non-finite Inkling MXFP8 weight")
    return result.reshape(weight.shape)


def quantize_mxfp8(x):
    """Group32 ceil-power-of-two scale, nearest-even/satfinite E4M3 values.

    `nearest` in the checkpoint describes E4M3 rounding, not nearest exponent.
    Both pinned vendor kernels round the exponent up using float32 bits.
    """
    if x.shape[-1] % 32 or not torch.isfinite(x).all():
        raise ValueError("MXFP8 activations require finite group32 values")
    blocks = x.float().reshape(*x.shape[:-1], -1, 32)
    raw = blocks.abs().amax(-1, keepdim=True) * (1. / 448.)
    exponent = ((raw.contiguous().view(torch.int32) + 0x007FFFFF) & 0x7F800000) >> 23
    exponent = torch.where(raw <= 2. ** -127, 0, exponent.clamp(max=254))
    scale = torch.ldexp(torch.ones_like(raw), exponent - 127)
    return ((blocks / scale).clamp(-448., 448.).to(torch.float8_e4m3fn).float() * scale).reshape(x.shape)


def capped_swiglu(gate_up, gamma, gate_cap, up_cap):
    gate, up = gate_up[..., 0::2], gate_up[..., 1::2]
    # Sigmoid is intentionally evaluated before softcapping the gate.
    return (((gate_cap * torch.tanh(gate / gate_cap)) * gate.sigmoid())
            * (up_cap * torch.tanh(up / up_cap))) * gamma[..., None]


def route(logits, bias, topk, scale):
    ids = (logits.sigmoid() + bias).argsort(dim=-1, descending=True, stable=True)[..., :topk]
    weights = F.logsigmoid(logits.gather(-1, ids)).softmax(-1) * scale
    return ids, weights


def relative_attention(q, k, v, relative, gate, sinks, *, window=None):
    """Relative bias is unscaled; a zero-valued sink adds exp(sink) to Z.

    Fixed query/key geometry per causal M128 tile preserves prefix bytes even
    when a longer suffix changes the last tile or expert-token populations.
    """
    if (q.ndim != 3 or k.ndim != 3 or v.shape != k.shape or not k.shape[1]
            or q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]
            or q.shape[1] % k.shape[1] or gate.shape != q.shape
            or sinks.shape != q.shape[:2] or relative.ndim != 3
            or relative.shape[:2] != q.shape[:2] or not relative.shape[-1]
            or (window is not None and (type(window) is not int or window <= 0))):
        raise ValueError("invalid Inkling attention geometry")
    length, heads, dim = q.shape
    output = torch.empty_like(q)
    for start in range(0, length, ROWS):
        count, end = min(ROWS, length - start), start + ROWS
        first = 0 if window is None else max(0, start - window + 1)
        qpos = torch.arange(start, end, device=q.device)[:, None]
        kpos = torch.arange(first, end, device=q.device)[None, :]
        distance = qpos - kpos
        valid = distance >= 0
        if window is not None:
            valid &= distance < window
        extent = relative.shape[-1]
        for head in range(heads):
            kv_head = head // (heads // k.shape[1])
            query = F.pad(q[start:start + count, head], (0, 0, 0, ROWS - count))
            keys = F.pad(k[first:min(end, length), kv_head], (0, 0, 0, max(0, end - length)))
            values = F.pad(v[first:min(end, length), kv_head], (0, 0, 0, max(0, end - length)))
            rel = F.pad(relative[start:start + count, head], (0, 0, 0, ROWS - count))
            bias = rel.gather(-1, distance.clamp(0, extent - 1))
            bias = bias.masked_fill((distance < 0) | (distance >= extent), 0.)
            scores = ((query @ keys.T) + bias / dim ** -.5) * dim ** -.5
            scores.masked_fill_(~valid, -float("inf"))
            sink = F.pad(sinks[start:start + count, head], (0, ROWS - count))
            probabilities = torch.cat((scores, sink[:, None]), -1).softmax(-1)[:, :-1]
            attended = probabilities @ values
            output[start:start + count, head] = (attended[:count]
                * gate[start:start + count, head].sigmoid())
    return output


def tensor_contract(cfg, index=None):
    result = {}
    def add(name, shape, dtype="BF16"):
        result[TEXT + name] = (list(shape), dtype)
    h = cfg.hidden_size
    if index is None:
        for name in ("embed.weight", "unembed.weight"):
            add(name, (cfg.vocab_size, h))
        for name in ("embed_norm.weight", "norm.weight"):
            add(name, (h,))
        add("attn_res.output_query", (h,), "F32")
        return result
    heads, kv_heads, dim = cfg.geometry(index)
    local = index in cfg.local_layer_ids
    p = f"layers.{index}."
    for name, shape in (("wq_du.weight", (heads * dim, h)), ("wr_du.weight", (heads * cfg.d_rel, h)),
                        ("wk_dv.weight", (kv_heads * dim, h)), ("wv_dv.weight", (kv_heads * dim, h)),
                        ("gate_attn.weight", (heads * dim, h)), ("attn_sink_proj.weight", (heads, h)),
                        ("wo_ud.weight", (h, heads * dim)), ("q_l2_clip.scale", (heads, dim)),
                        ("rel_l2_clip.scale", (heads, cfg.d_rel)),
                        ("rel_logits_proj.proj", (cfg.d_rel, cfg.sliding_window_size if local else cfg.rel_extent))):
        add(p + "attn." + name, shape)
    for name in ("attn", "mlp"):
        add(p + name + "_sconv.weight", (h, 1, cfg.sconv_kernel_size))
    for name in (("attn_norm",) if local else ("q_input_norm", "k_input_norm", "v_input_norm")):
        add(p + name + ".weight", (h,))
    add(p + "mlp_norm.weight", (h,))
    for name in ("k", "v"):
        stem = p + "attn." + name + "_sconv."
        if local:
            add(stem + "weight", (kv_heads * dim, 1, cfg.sconv_kernel_size))
        else:
            add(stem + "static_w", (cfg.sconv_kernel_size, kv_heads * dim))
            add(stem + "weight_proj", (kv_heads * cfg.sconv_kernel_size, h))
    add(p + "mlp.gate.weight", (cfg.n_routed_experts, h))
    add(p + "mlp.gate.bias", (cfg.n_routed_experts,), "F32")
    for name, rows, cols in (("w13", 2 * cfg.intermediate_size, h), ("w2", h, cfg.intermediate_size)):
        add(p + "mlp.experts." + name + "_weight", (cfg.n_routed_experts, rows, cols), "F8_E4M3")
        add(p + "mlp.experts." + name + "_weight_scale", (cfg.n_routed_experts, rows, cols // 32), "U8")
    add(f"attn_res.mlp_queries.{index}", (h,), "F32")
    if index:
        add(f"attn_res.attn_queries.{index - 1}", (h,) if local else (3, h), "F32")
    return result


class Weights:
    """One native packed layer resident; decode only one expert matrix at a time."""
    def __init__(self, snapshot, device="cpu"):
        self.snapshot, self.device = Path(snapshot), torch.device(device)
        self.index = json.loads((self.snapshot / "model.safetensors.index.json").read_bytes())["weight_map"]
        self.headers, self.maps, self.read_bytes = {}, {}, 0

    def descriptor(self, name):
        if name not in self.index:
            raise ValueError("missing native Inkling tensor: " + name)
        shard = self.index[name]
        if shard not in self.headers:
            if not isinstance(shard, str) or Path(shard).name != shard or shard in (".", ".."):
                raise ValueError("unsafe Inkling shard path")
            path = self.snapshot / shard
            with path.open("rb") as stream:
                count = struct.unpack("<Q", stream.read(8))[0]
                if count > 64 * 2**20:
                    raise ValueError("oversized safetensors header")
                self.headers[shard] = json.loads(stream.read(count))
        return self.headers[shard][name]

    def validate(self, cfg):
        quant = json.loads((self.snapshot / "hf_quant_config.json").read_bytes())
        if quant.get("quant_method") != "inkling2":
            raise ValueError("unsupported Inkling quantization method")
        groups = quant["model"]["parameter_groups"]
        contracts = [tensor_contract(cfg), *(tensor_contract(cfg, i) for i in range(cfg.num_hidden_layers))]
        expected = {name: desc for contract in contracts for name, desc in contract.items()}
        group_map = {}
        for group in groups:
            for name in group["tensors"]:
                if name in group_map:
                    raise ValueError("duplicate Inkling quantization tensor")
                group_map[name] = group
        for name, (shape, dtype) in expected.items():
            descriptor = self.descriptor(name)
            if descriptor["shape"] != shape or descriptor["dtype"] != dtype:
                raise ValueError("native Inkling tensor layout differs: " + name)
            if name.endswith("_weight_scale"):
                continue
            g = group_map.get(name, {})
            if dtype == "F8_E4M3":
                if (g.get("compute_dtype") != "mxfp8" or g.get("weight_block_shape") != [32, 32]
                        or g.get("weight_transform") != "mxfp8_nearest"
                        or g.get("required_activation_quantization") != {
                            "block_size": 32, "format": "mxfp8_e4m3", "rounding": "nearest", "scale_dtype": "ue8m0"}):
                    raise ValueError("unsupported Inkling native MXFP8 policy: " + name)
            elif g.get("compute_dtype") != ("float32" if dtype == "F32" else "bfloat16"):
                raise ValueError("unsupported Inkling dense precision: " + name)
        actual = {name for name in self.index if name.startswith(TEXT)}
        if actual != expected.keys():
            raise ValueError("unrecognized Inkling text tensors: " + str(sorted(actual ^ expected.keys())))

    def tensor(self, name):
        from safetensors import safe_open
        descriptor = self.descriptor(name)
        shard = self.index[name]
        if shard not in self.maps:
            self.maps[shard] = safe_open(str(self.snapshot / shard), framework="pt", device="cpu")
        self.read_bytes += descriptor["data_offsets"][1] - descriptor["data_offsets"][0]
        return self.maps[shard].get_tensor(name).to(device=self.device, copy=True)

    def load_layer(self, index, cfg):
        prefix = TEXT + f"layers.{index}."
        values = {}
        for name in tensor_contract(cfg, index):
            value = self.tensor(name)
            key = name.removeprefix(prefix) if name.startswith(prefix) else name.removeprefix(TEXT)
            values[key] = value if ".experts." in name else value.float()
        self.close()
        return values

    def close(self):
        self.maps.clear()


def attention_forward(inputs, layer, cfg, index):
    local = index in cfg.local_layer_ids
    heads, kv_heads, dim = cfg.geometry(index)
    names = ("attn_norm",) if local else ("q_input_norm", "k_input_norm", "v_input_norm")
    sources = [rms_norm(x, layer[name + ".weight"], cfg.rms_norm_eps)
               for x, name in zip(inputs, names, strict=True)]
    xq, xk, xv = (sources * 3 if local else sources)
    def project(x, name):
        return linear(x, layer["attn." + name + ".weight"])
    q = project(xq, "wq_du").reshape(-1, heads, dim)
    r = project(xq, "wr_du").reshape(-1, heads, cfg.d_rel)
    gate = project(xq, "gate_attn").reshape(-1, heads, dim)
    sinks = project(xq, "attn_sink_proj")
    kv = []
    for x, name, proj in ((xk, "k", "wk_dv"), (xv, "v", "wv_dv")):
        stem = "attn." + name + "_sconv."
        dynamic = None if local else linear(x, layer[stem + "weight_proj"]).reshape(
            -1, kv_heads, cfg.sconv_kernel_size)
        value = short_conv(project(x, proj), layer[stem + ("weight" if local else "static_w")], dynamic)
        kv.append(value.reshape(-1, kv_heads, dim))
    k, v = kv
    q = l2_clip(q, cfg.qk_l2_clip_threshold, cfg.rms_norm_eps, layer["attn.q_l2_clip.scale"])
    k = l2_clip(k, cfg.qk_l2_clip_threshold, cfg.rms_norm_eps)
    v = rms_norm(v, None, cfg.rms_norm_eps)
    r = l2_clip(r, cfg.rel_l2_clip_threshold, cfg.rms_norm_eps, layer["attn.rel_l2_clip.scale"])
    if not local and cfg.text.get("log_scaling_n_floor") is not None:
        positions = torch.arange(len(q), device=q.device)
        tau = 1. + cfg.log_scaling_alpha * torch.log(((positions + 1).float() / cfg.log_scaling_n_floor).clamp_min(1.))
        q, r = q * tau[:, None, None], r * tau[:, None, None]
    relative = linear(r, layer["attn.rel_logits_proj.proj"].T)
    out = relative_attention(q, k, v, relative, gate, sinks,
                             window=cfg.sliding_window_size if local else None)
    return linear(out.reshape(len(q), -1), layer["attn.wo_ud.weight"])


def moe_forward(x, layer, cfg):
    logits = linear(x, layer["mlp.gate.weight"])
    ids, gammas = route(logits, layer["mlp.gate.bias"], cfg.num_experts_per_tok, cfg.route_scale)
    inputs = quantize_mxfp8(x)
    slots = torch.empty((len(x), cfg.num_experts_per_tok, cfg.hidden_size), device=x.device, dtype=torch.float32)
    for expert in range(cfg.n_routed_experts):
        rows, choices = torch.where(ids == expert)
        if not len(rows):
            continue
        gate_up_weight = dequant_mxfp8(layer["mlp.experts.w13_weight"][expert],
                                      layer["mlp.experts.w13_weight_scale"][expert])
        gate_up = linear(inputs[rows], gate_up_weight)
        del gate_up_weight
        activation = capped_swiglu(gate_up, gammas[rows, choices], cfg.mlp_gate_softcap, cfg.mlp_up_softcap)
        del gate_up
        down_weight = dequant_mxfp8(layer["mlp.experts.w2_weight"][expert],
                                   layer["mlp.experts.w2_weight_scale"][expert])
        slots[rows, choices] = linear(quantize_mxfp8(activation), down_weight)
        del activation, down_weight
    return slots.sum(1)


def layer_forward(partial, blocks, layer, cfg, index):
    if blocks:
        inputs = attn_res(blocks, partial, layer[f"attn_res.attn_queries.{index - 1}"], cfg.rms_norm_eps)
    else:
        inputs = (partial,) if index in cfg.local_layer_ids else (partial,) * 3
    attention = short_conv(attention_forward(inputs, layer, cfg, index), layer["attn_sconv.weight"])
    first = index % cfg.n_layers_per_attn_res_block == 0
    query = layer[f"attn_res.mlp_queries.{index}"]
    mlp_input = attn_res([*blocks, partial] if first else blocks,
                         attention if first else partial, query, cfg.rms_norm_eps,
                         addend=None if first else attention)[0]
    mlp_input = rms_norm(mlp_input, layer["mlp_norm.weight"], cfg.rms_norm_eps)
    mlp = short_conv(moe_forward(mlp_input, layer, cfg), layer["mlp_sconv.weight"])
    return attention + mlp if first else (partial + attention) + mlp


def output_norm(partial, blocks, query, norm, cfg):
    mixed = attn_res(blocks, partial, query, cfg.rms_norm_eps)[0]
    return rms_norm(mixed, norm, cfg.rms_norm_eps)


def head_forward(hidden, head, cfg):
    # The shipped matrix includes padding; the vendor logits processor removes it.
    logits = linear(hidden.float(), head[:cfg.unpadded_vocab_size].float())
    return cfg.final_logit_softcapping * torch.tanh(logits / cfg.final_logit_softcapping)

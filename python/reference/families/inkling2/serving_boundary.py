"""Independent vendor expert materialization beside the unchanged IEEE-F32 oracle.

Pinned vLLM d664cd1e: ops/trtllm_bmm.py:183 FC1/FC2 BF16;
ops/activation.py:78 weighted SiTU BF16 before A8;
ops/moe.py:1077-1080 FP32 slot-order combine, then moe.py:239-240 BF16.
Attention/dense/residual operations remain the #1915 IEEE-F32 oracle; this
mode changes ONLY the documented expert serving boundaries, not the oracle.
"""
import torch
from . import model as f32
from shape_invariant import linear


def moe_forward(x, layer, cfg):
    ids, gamma = f32.route(linear(x, layer["mlp.gate.weight"]), layer["mlp.gate.bias"],
                          cfg.num_experts_per_tok, cfg.route_scale)
    inputs = f32.quantize_mxfp8(x)
    slots = torch.zeros(len(x), cfg.num_experts_per_tok, cfg.hidden_size, device=x.device)
    for expert in ids.unique().tolist():
        rows, choices = torch.where(ids == expert)
        w13 = f32.dequant_mxfp8(layer["mlp.experts.w13_weight"][expert], layer["mlp.experts.w13_weight_scale"][expert])
        gate_up = linear(inputs[rows], w13).bfloat16().float()
        del w13
        activation = f32.capped_swiglu(gate_up, gamma[rows, choices], cfg.mlp_gate_softcap,
                                      cfg.mlp_up_softcap).bfloat16().float()
        down = f32.dequant_mxfp8(layer["mlp.experts.w2_weight"][expert], layer["mlp.experts.w2_weight_scale"][expert])
        slots[rows, choices] = linear(f32.quantize_mxfp8(activation), down).bfloat16().float()
        del down
    combined = torch.zeros_like(slots[:, 0])
    for choice in range(cfg.num_experts_per_tok):
        combined += slots[:, choice]
    return combined.bfloat16().float()


def layer_forward(partial, blocks, layer, cfg, index):
    inputs = (f32.attn_res(blocks, partial, layer[f"attn_res.attn_queries.{index-1}"], cfg.rms_norm_eps)
              if blocks else (partial,) if index in cfg.local_layer_ids else (partial,)*3)
    attention = f32.short_conv(f32.attention_forward(inputs, layer, cfg, index), layer["attn_sconv.weight"])
    first = index % cfg.n_layers_per_attn_res_block == 0
    mlp_input = f32.attn_res([*blocks, partial] if first else blocks, attention if first else partial,
        layer[f"attn_res.mlp_queries.{index}"], cfg.rms_norm_eps, addend=None if first else attention)[0]
    mlp_input = f32.rms_norm(mlp_input, layer["mlp_norm.weight"], cfg.rms_norm_eps)
    mlp = f32.short_conv(moe_forward(mlp_input, layer, cfg), layer["mlp_sconv.weight"])
    return attention+mlp if first else (partial+attention)+mlp


def is_partial(last_layer, layers):
    return last_layer + 1 != layers

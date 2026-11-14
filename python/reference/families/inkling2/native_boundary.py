"""Independent native-boundary wiring oracle, never imported by serving.

Matches the Rust boundary audit, not a claim of full vendor semantics:
F32 residuals/AttnRes/convs/clips/projection accumulators; BF16 norm egress,
projection inputs, attention Q/K/V/relative/PV probabilities/output, gated wo
input, and final normalized head input. Expert boundaries are pinned vendor
BF16 FC1/SiTU/FC2 plus ordered F32 combine. No engine or kernel imports.
"""
import torch
from . import model as f32
from . import serving_boundary as vendor
from shape_invariant import linear


def narrow(x):
    return x.bfloat16().float()


def attention_forward(inputs, layer, cfg, index):
    local = index in cfg.local_layer_ids
    heads, kv_heads, dim = cfg.geometry(index)
    norms = ('attn_norm',) if local else ('q_input_norm', 'k_input_norm', 'v_input_norm')
    sources = [narrow(f32.rms_norm(x, layer[name+'.weight'], cfg.rms_norm_eps))
               for x, name in zip(inputs, norms, strict=True)]
    xq, xk, xv = sources*3 if local else sources
    def project(x, name):
        return linear(x, layer['attn.'+name+'.weight'])
    q = project(xq, 'wq_du').reshape(-1, heads, dim)
    r = project(xq, 'wr_du').reshape(-1, heads, cfg.d_rel)
    gate = project(xq, 'gate_attn').reshape(-1, heads, dim)
    sinks = project(xq, 'attn_sink_proj')
    kv = []
    for x, name, projection in [(xk, 'k', 'wk_dv'), (xv, 'v', 'wv_dv')]:
        stem = 'attn.'+name+'_sconv.'
        dynamic = None if local else linear(x, layer[stem+'weight_proj']).reshape(-1, kv_heads, cfg.sconv_kernel_size)
        value = f32.short_conv(project(x, projection), layer[stem+('weight' if local else 'static_w')], dynamic)
        kv.append(value.reshape(-1, kv_heads, dim))
    q = f32.l2_clip(q, cfg.qk_l2_clip_threshold, cfg.rms_norm_eps, layer['attn.q_l2_clip.scale'])
    k = f32.l2_clip(kv[0], cfg.qk_l2_clip_threshold, cfg.rms_norm_eps)
    v = f32.rms_norm(kv[1], None, cfg.rms_norm_eps)
    r = f32.l2_clip(r, cfg.rel_l2_clip_threshold, cfg.rms_norm_eps, layer['attn.rel_l2_clip.scale'])
    if not local and cfg.text.get('log_scaling_n_floor') is not None:
        positions = torch.arange(len(q), device=q.device)
        tau = 1.+cfg.log_scaling_alpha*torch.log(((positions+1).float()/cfg.log_scaling_n_floor).clamp_min(1.))
        q, r = q*tau[:, None, None], r*tau[:, None, None]
    relative = narrow(linear(narrow(r), layer['attn.rel_logits_proj.proj'].T))
    q, k, v = narrow(q), narrow(k), narrow(v)
    length = len(q)
    position = torch.arange(length, device=q.device)
    distance = position[:, None]-position[None, :]
    valid = distance>=0
    if local:
        valid &= distance<cfg.sliding_window_size
    out = torch.empty_like(q)
    # Reference-only dense causal attention. The native kernel uses tiled
    # online softmax; both materialize BF16 probabilities before PV.
    for head in range(heads):
        bias = relative[:, head].gather(-1, distance.clamp(0, relative.shape[-1]-1))
        bias = bias.masked_fill((distance<0)|(distance>=relative.shape[-1]), 0.)
        scores = ((q[:, head]@k[:, head//(heads//kv_heads)].T)+bias/(dim**-.5))*(dim**-.5)
        scores.masked_fill_(~valid, -float('inf'))
        # Online attention fixes its PV numerator scale from real keys; the
        # zero-valued sink enters only the final denominator normalization.
        maximum = scores.amax(-1)
        numerator = (scores-maximum[:, None]).exp()
        denominator = numerator.sum(-1)+(sinks[:, head]-maximum).exp()
        out[:, head] = narrow((narrow(numerator)@v[:, head//(heads//kv_heads)])/denominator[:, None])
    gated = narrow(out*gate.sigmoid())
    return linear(gated.reshape(length, -1), layer['attn.wo_ud.weight'])


def layer_forward(partial, blocks, layer, cfg, index):
    inputs = (f32.attn_res(blocks, partial, layer[f'attn_res.attn_queries.{index-1}'], cfg.rms_norm_eps)
              if blocks else (partial,) if index in cfg.local_layer_ids else (partial,)*3)
    attention = f32.short_conv(attention_forward(inputs, layer, cfg, index), layer['attn_sconv.weight'])
    first = index%cfg.n_layers_per_attn_res_block==0
    mixed = f32.attn_res([*blocks, partial] if first else blocks, attention if first else partial,
        layer[f'attn_res.mlp_queries.{index}'], cfg.rms_norm_eps, addend=None if first else attention)[0]
    mlp_input = narrow(f32.rms_norm(mixed, layer['mlp_norm.weight'], cfg.rms_norm_eps))
    mlp = f32.short_conv(vendor.moe_forward(mlp_input, layer, cfg), layer['mlp_sconv.weight'])
    return attention+mlp if first else (partial+attention)+mlp

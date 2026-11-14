#!/usr/bin/env python3
"""Diagnostic ONLY: real Inkling layer0/5 kernel wiring, not a Rust golden.

Torch norm/linear/pack/route/residual are explicitly permitted golden-side
bridges. Attention, native MXFP8 W8A8 experts, static/dynamic convolutions
and clipping use CuTe producers. Never imported by serving. Layer0 and layer5
use real checkpoint embedding rows as their input: layer5 is an isolated
arithmetic check, not the output of layers0..4 or a fidelity measurement.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python/reference")]

import torch
import torch.nn.functional as F

from families.inkling2 import model as oracle
from shape_invariant import linear


class ArenaWeights(oracle.Weights):
    """Diagnostic packed immutable native views, one arena per loaded layer.

    BF16/F32 core tensors and native expert storage all alias this backing.
    The fused w13 checkpoint is copied into separate gate/up views without
    dequantizing; w2 scales retain their native UE8M0 padded-K row layout.
    Transient host safetensors tensors remain outside the device arena.
    """
    def load_layer(self, index, cfg):
        from safetensors import safe_open
        from b12x.integration.cuteafd._mxfp4_down_a8_plan import mxfp4_scale_row_bytes
        contract = oracle.tensor_contract(cfg, index)
        layouts, offset = {}, 0
        for name, (shape, kind) in contract.items():
            itemsize = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1}[kind]
            count = 1
            for dim in shape:
                count *= dim
            offset = (offset + 255) // 256 * 256
            layouts[name] = (offset, count * itemsize, shape)
            offset += count * itemsize
        self.arena = torch.empty(offset, dtype=torch.uint8, device=self.device)
        values = {}
        prefix = oracle.TEXT + f"layers.{index}."
        for name, (at, size, shape) in layouts.items():
            shard = self.index[name]
            with safe_open(str(self.snapshot / shard), framework="pt", device="cpu") as stream:
                value = stream.get_tensor(name)
                view = self.arena[at:at + size].view(value.dtype).view(shape)
                view.copy_(value)
            key = name.removeprefix(prefix) if name.startswith(prefix) else name.removeprefix(oracle.TEXT)
            values[key] = view
        # Splitting fused interleaved gate/up creates one independently retired
        # expert arena, not three tensor allocations. Only copies raw FP8 bytes.
        h, i, e = cfg.hidden_size, cfg.intermediate_size, cfg.n_routed_experts
        w13, s13 = values["mlp.experts.w13_weight"], values["mlp.experts.w13_weight_scale"]
        w2, s2 = values["mlp.experts.w2_weight"], values["mlp.experts.w2_weight_scale"]
        shapes = [(e, i, h), (e, i, h//32), (e, i, h), (e, i, h//32),
                  (e, h, i), (e, h, mxfp4_scale_row_bytes(i))]
        dtypes = [w13.dtype, s13.dtype, w13.dtype, s13.dtype, w2.dtype, s2.dtype]
        total, offsets = 0, []
        for shape, dtype in zip(shapes, dtypes):
            total = (total + 255)//256*256
            size = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
            offsets.append((total, size))
            total += size
        self.expert_arena = torch.empty(total, dtype=torch.uint8, device=self.device)
        expert = [self.expert_arena[at:at+size].view(dtype).view(shape)
                  for (at, size), dtype, shape in zip(offsets, dtypes, shapes)]
        expert[0].copy_(w13[:, 0::2]); expert[1].copy_(s13[:, 0::2])
        expert[2].copy_(w13[:, 1::2]); expert[3].copy_(s13[:, 1::2])
        expert[4].copy_(w2); expert[5].zero_(); expert[5][:, :, :i//32].copy_(s2)
        # Compare one expert at a time: bounded temporary comparisons, no
        # extra full weight representation and no numerical conversion.
        for eindex in range(e):
            for dest, source in [(expert[0][eindex], w13[eindex, 0::2]),
                                 (expert[2][eindex], w13[eindex, 1::2])]:
                assert torch.equal(dest.view(torch.uint8), source.view(torch.uint8))
            assert torch.equal(expert[1][eindex], s13[eindex, 0::2])
            assert torch.equal(expert[3][eindex], s13[eindex, 1::2])
        print("diagnostic w13 rawFP8 byte permutation: exact", flush=True)
        self.expert = expert
        self.owner_bytes = {"inkling2.diagnostic.source_arena": self.arena.numel(),
                            "inkling2.diagnostic.expert_arena": self.expert_arena.numel()}
        return values


def check(label, actual, expected, report, *, rms_bar=.02, cos_bar=.999, enforce=True):
    a, b = actual.float().flatten(), expected.float().flatten()
    finite = bool(torch.isfinite(a).all())
    rms = float((a-b).square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-20))
    cosine = float(F.cosine_similarity(a[None], b[None]))
    result = dict(finite=finite, normalized_rms=rms, cosine=cosine,
                  max_abs=float((a-b).abs().max()))
    print(label, json.dumps(result), flush=True)
    report[label] = result
    if enforce and (not finite or rms > rms_bar or cosine < cos_bar):
        raise AssertionError(f"{label}: diagnostic bar failed")


class Native:
    def __init__(self, cfg, rows):
        from b12x.integration.cuteafd import INKLING2_SMALL
        from b12x.integration.cuteafd.fp8_moe import GEOMETRIES, compile_fp8_moe_aot, fp8_moe_scratch_bytes
        from b12x.integration.cuteafd.inkling2_attention import compile_inkling2_attention_aot
        from b12x.integration.cuteafd.inkling2_convolution import compile_inkling2_convolution_aot
        from b12x.integration.cuteafd.inkling2_clip import compile_inkling2_clip_aot
        self.cfg, self.rows, self.g = cfg, rows, INKLING2_SMALL
        self.attn = {kind: compile_inkling2_attention_aot(kind=kind, route="prefill", max_rows=4096)
                     for kind in ("local", "global")}
        self.conv = {(width, dynamic): compile_inkling2_convolution_aot(width=width, heads=8 if dynamic else 1,
                     dynamic=dynamic, dtype=torch.float32, max_rows=4096)
                     for width, dynamic in [(4608, False), (1024, False), (1024, True)]}
        self.clip = {}
        for local in (True, False):
            for name, heads, dim, threshold, scaled in [("q", 48, 128, cfg.qk_l2_clip_threshold, True),
                    ("k", 8, 128, cfg.qk_l2_clip_threshold, False), ("r", 48, 16, cfg.rel_l2_clip_threshold, True)]:
                floor, alpha = (0, 0.) if local or name=="k" else (cfg.log_scaling_n_floor, cfg.log_scaling_alpha)
                self.clip[local, name] = compile_inkling2_clip_aot(heads=heads, dim=dim, threshold=threshold,
                    eps=cfg.rms_norm_eps, scaled=scaled, log_floor=floor or 0, log_alpha=alpha,
                    dtype=torch.float32, max_rows=4096)
        geometry = GEOMETRIES["inkling2_mxfp8"]
        self.experts = compile_fp8_moe_aot(geometry, route="stream", max_rows=4096, wire=True)
        self.expert_scratch = torch.empty(fp8_moe_scratch_bytes(geometry, "stream", 4096, True),
                                           dtype=torch.uint8, device="cuda")
        self.positions = torch.arange(rows, device="cuda", dtype=torch.int64)
        self.first = torch.zeros(rows, device="cuda", dtype=torch.int32)
        self.slots = torch.zeros(rows, device="cuda", dtype=torch.int64)

    def short_conv(self, x, weight, dynamic=None):
        width = x.shape[-1]
        history = torch.zeros(1, 3, width, device="cuda")
        out = torch.empty_like(x, dtype=torch.float32)
        self.conv[width, dynamic is not None].launch(x.contiguous(), weight, dynamic,
            history, self.first, self.slots, out, scalars=(self.rows,))
        return out

    def clipped(self, x, scale, local, name):
        out = torch.empty_like(x, dtype=torch.float32)
        self.clip[local, name].launch(x.contiguous(), scale, self.positions, out, scalars=(self.rows,))
        return out

    def attention(self, inputs, layer, cfg, index):
        local = index in cfg.local_layer_ids
        heads, kv_heads, dim = cfg.geometry(index)
        names = ("attn_norm",) if local else ("q_input_norm", "k_input_norm", "v_input_norm")
        sources = [oracle.rms_norm(x, layer[name+".weight"], cfg.rms_norm_eps)
                   for x, name in zip(inputs, names, strict=True)]
        xq, xk, xv = sources*3 if local else sources
        def project(x, name):
            return linear(x, layer["attn."+name+".weight"].float())
        q = project(xq, "wq_du").reshape(-1, heads, dim)
        r = project(xq, "wr_du").reshape(-1, heads, cfg.d_rel)
        gate = project(xq, "gate_attn").reshape(-1, heads, dim)
        sinks = project(xq, "attn_sink_proj")
        kv = []
        for x, name, proj in [(xk, "k", "wk_dv"), (xv, "v", "wv_dv")]:
            stem = "attn."+name+"_sconv."
            dynamic = None if local else linear(x, layer[stem+"weight_proj"].float()).reshape(-1, kv_heads, 4)
            kv.append(self.short_conv(project(x, proj), layer[stem+("weight" if local else "static_w")], dynamic)
                      .reshape(-1, kv_heads, dim))
        q = self.clipped(q, layer["attn.q_l2_clip.scale"], local, "q")
        k = self.clipped(kv[0], None, local, "k")
        v = oracle.rms_norm(kv[1], None, cfg.rms_norm_eps)
        r = self.clipped(r, layer["attn.rel_l2_clip.scale"], local, "r")
        relative = linear(r, layer["attn.rel_logits_proj.proj"].float().T).bfloat16()
        records = torch.cat([k.reshape(self.rows, -1), v.reshape(self.rows, -1)], 1).bfloat16()
        output = torch.empty_like(q, dtype=torch.bfloat16)
        scratch = torch.empty(1024, dtype=torch.uint8, device="cuda")
        if local:
            ring = torch.zeros(self.g.ring_rows, self.g.record_elems, device="cuda", dtype=torch.bfloat16)
            self.attn["local"].launch(q.bfloat16(), records, ring, self.positions,
                self.positions%self.g.ring_rows, self.first, relative, sinks, output, scratch, scalars=(self.rows,))
        else:
            pages = (self.rows+63)//64
            cache = F.pad(records, (0, 0, 0, pages*64-self.rows))
            table = torch.arange(pages, device="cuda", dtype=torch.int32)
            self.attn["global"].launch(q.bfloat16(), cache, self.positions, table, relative,
                sinks, output, scratch, scalars=(self.rows, 0))
        return linear((output.float()*gate.sigmoid()).reshape(self.rows, -1), layer["attn.wo_ud.weight"].float())

    def moe(self, x, layer, weights):
        logits = linear(x, layer["mlp.gate.weight"].float())
        ids, gammas = oracle.route(logits, layer["mlp.gate.bias"], self.cfg.num_experts_per_tok, self.cfg.route_scale)
        # Golden-only wire packing: group32 UE8M0 + E4M3, no weight conversion.
        blocks = x.float().reshape(self.rows, -1, 32)
        raw = blocks.abs().amax(-1, keepdim=True) * (1. / 448.)
        exponent = ((raw.contiguous().view(torch.int32) + 0x007FFFFF) & 0x7F800000) >> 23
        exponent = torch.where(raw <= 2. ** -127, 0, exponent.clamp(max=254))
        scale = torch.ldexp(torch.ones_like(raw), exponent - 127)
        codes = (blocks / scale).clamp(-448., 448.).to(torch.float8_e4m3fn)
        wire = torch.cat([codes.view(torch.uint8).reshape(self.rows, 4608),
                          exponent.to(torch.uint8).reshape(self.rows, 144)], 1).contiguous()
        out = torch.empty_like(x, dtype=torch.bfloat16)
        self.experts.launch(wire, ids.int().contiguous(), gammas.contiguous(), *weights.expert,
                            out, self.expert_scratch, scalars=(self.rows,))
        return out.float()

    def layer(self, partial, blocks, layer, cfg, index, weights):
        if blocks:
            inputs = oracle.attn_res(blocks, partial, layer[f"attn_res.attn_queries.{index-1}"], cfg.rms_norm_eps)
        else:
            inputs = (partial,) if index in cfg.local_layer_ids else (partial,)*3
        attention = self.short_conv(self.attention(inputs, layer, cfg, index), layer["attn_sconv.weight"])
        first = index%cfg.n_layers_per_attn_res_block == 0
        mlp_input = oracle.attn_res([*blocks, partial] if first else blocks,
            attention if first else partial, layer[f"attn_res.mlp_queries.{index}"], cfg.rms_norm_eps,
            addend=None if first else attention)[0]
        mlp_input = oracle.rms_norm(mlp_input, layer["mlp_norm.weight"], cfg.rms_norm_eps)
        mlp = self.short_conv(self.moe(mlp_input, layer, weights), layer["mlp_sconv.weight"])
        return attention+mlp if first else (partial+attention)+mlp


def expert_boundary_oracle(x, layer, cfg):
    """Separate counterfactual: native kernel BF16 FC1/SiTU and route-sum boundary.

    #1915 already includes both MXFP8 A8 QDQs. This diagnostic adds the BF16
    projection/SiTU materialization documented by #294, without replacing it.
    """
    logits = linear(x, layer["mlp.gate.weight"].float())
    ids, gamma = oracle.route(logits, layer["mlp.gate.bias"], cfg.num_experts_per_tok, cfg.route_scale)
    inputs = oracle.quantize_mxfp8(x)
    slots = torch.zeros(len(x), cfg.num_experts_per_tok, cfg.hidden_size, device=x.device)
    for expert in ids.unique().tolist():
        rows, choices = torch.where(ids == expert)
        w13 = oracle.dequant_mxfp8(layer["mlp.experts.w13_weight"][expert], layer["mlp.experts.w13_weight_scale"][expert])
        gate_up = linear(inputs[rows], w13).bfloat16().float()
        del w13
        activation = oracle.capped_swiglu(gate_up, gamma[rows, choices], cfg.mlp_gate_softcap, cfg.mlp_up_softcap).bfloat16().float()
        down = oracle.dequant_mxfp8(layer["mlp.experts.w2_weight"][expert], layer["mlp.experts.w2_weight_scale"][expert])
        slots[rows, choices] = linear(oracle.quantize_mxfp8(activation), down)
        del down
    return slots.sum(1).bfloat16().float()


def stage_probe(native, hidden, layer, cfg, index, weights, report):
    """Diagnostic table only; do not substitute rounded numerics for the oracle."""
    inputs = (hidden,) if index in cfg.local_layer_ids else (hidden,)*3
    reference_attention = oracle.short_conv(oracle.attention_forward(inputs, layer, cfg, index),
                                            layer["attn_sconv.weight"])
    native_attention = native.short_conv(native.attention(inputs, layer, cfg, index), layer["attn_sconv.weight"])
    check(f"layer{index}/attention_out", native_attention, reference_attention, report, enforce=False)
    first = index % cfg.n_layers_per_attn_res_block == 0
    def mlp_inputs(attention):
        mixed = oracle.attn_res([hidden] if first else [], attention if first else hidden,
            layer[f"attn_res.mlp_queries.{index}"], cfg.rms_norm_eps,
            addend=None if first else attention)[0]
        return mixed, oracle.rms_norm(mixed, layer["mlp_norm.weight"], cfg.rms_norm_eps)
    reference_mix, reference_input = mlp_inputs(reference_attention)
    native_mix, native_input = mlp_inputs(native_attention)
    check(f"layer{index}/residual_mix", native_mix, reference_mix, report, enforce=False)
    check(f"layer{index}/mlp_in", native_input, reference_input, report, enforce=False)
    logits = [linear(x, layer["mlp.gate.weight"].float()) for x in [reference_input, native_input]]
    routes = [oracle.route(x, layer["mlp.gate.bias"], cfg.num_experts_per_tok, cfg.route_scale) for x in logits]
    changed = (routes[0][0] != routes[1][0]).any(-1)
    set_changed = (routes[0][0].sort(-1).values != routes[1][0].sort(-1).values).any(-1)
    scores = logits[0].sigmoid() + layer["mlp.gate.bias"]
    ordered = scores.sort(-1, descending=True).values
    margins = ordered[:, cfg.num_experts_per_tok-1] - ordered[:, cfg.num_experts_per_tok]
    route_report = dict(changed_order_rows=int(changed.sum()), changed_set_rows=int(set_changed.sum()),
        topk_boundary_margins=margins[set_changed].cpu().tolist(),
        changed_rows=torch.where(set_changed)[0].cpu().tolist(),
        reference_ids=routes[0][0][set_changed].cpu().tolist(), native_ids=routes[1][0][set_changed].cpu().tolist(),
        max_abs_router_logit_diff=float((logits[0]-logits[1]).abs().max()))
    print(f"layer{index}/route_changes", json.dumps(route_report), flush=True)
    report[f"layer{index}/route_changes"] = route_report
    reference_mlp = oracle.moe_forward(reference_input, layer, cfg)
    # Counterfactual isolates upstream amplification from native expert math.
    reference_on_native_input = oracle.moe_forward(native_input, layer, cfg)
    check(f"layer{index}/reference_mlp_upstream_sensitivity", reference_on_native_input, reference_mlp, report, enforce=False)
    native_same_input = native.moe(reference_input, layer, weights)
    check(f"layer{index}/mlp_out_same_reference_input", native_same_input, reference_mlp, report, enforce=False)
    native_mlp = native.moe(native_input, layer, weights)
    check(f"layer{index}/mlp_out", native_mlp, reference_mlp, report, enforce=False)
    check(f"layer{index}/native_expert_same_native_input", native_mlp, reference_on_native_input, report, enforce=False)
    boundaries = expert_boundary_oracle(native_input, layer, cfg)
    check(f"layer{index}/native_vs_bf16_boundary_counterfactual", native_mlp, boundaries, report, enforce=False)
    check(f"layer{index}/bf16_boundary_vs_f32_reference", boundaries, reference_on_native_input, report, enforce=False)
    matched = ~set_changed
    route_report["flip_rate"] = float(set_changed.float().mean())
    route_report["matched_rows"] = int(matched.sum())
    check(f"layer{index}/mlp_out_matched_set", native_mlp[matched], reference_mlp[matched], report)
    reference_tail = oracle.short_conv(reference_mlp, layer["mlp_sconv.weight"])
    native_tail = native.short_conv(native_mlp, layer["mlp_sconv.weight"])
    check(f"layer{index}/mlp_conv", native_tail, reference_tail, report, enforce=False)
    actual = native_attention+native_tail if first else (hidden+native_attention)+native_tail
    expected = reference_attention+reference_tail if first else (hidden+reference_attention)+reference_tail
    clean = matched.clone()
    for lag in range(1, cfg.sconv_kernel_size):
        clean[lag:] &= matched[:-lag]
    route_report["lag_clean_final_rows"] = int(clean.sum())
    check(f"layer{index}/final_lag_clean", actual[clean], expected[clean], report)
    return actual, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", action="store_true", required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", default="0,5")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--stage-taps", action="store_true", help="emit all stage metrics before enforcing unchanged layer bar")
    args = parser.parse_args()
    if not 8 <= args.rows <= 4096:
        parser.error("rows must be8..4096")
    layers = [int(n) for n in args.layers.split(",")]
    if not layers or any(n not in (0, 5) for n in layers):
        parser.error("diagnostic layers are0,5 only")
    # Packed immutable core views stay BF16. The independent F32 oracle widens
    # each projection only during its call, rather than retaining another core.
    oracle.linear = lambda x, weight, bias=None: linear(x.float(), weight.float(),
        None if bias is None else bias.float())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.set_device(0)
    cfg = oracle.Config.from_dict(json.loads((args.snapshot/"config.json").read_bytes()))
    weights = ArenaWeights(args.snapshot, "cuda")
    weights.validate(cfg)
    native = Native(cfg, args.rows)
    report = {"diagnostic_only": True, "torch_glue": ["norm", "linear", "pack", "route", "residual"],
              "rows": args.rows, "layers": layers, "comparison": "isolated real-weight layers, not full-model fidelity"}
    with torch.inference_mode():
        # Read only the selected embedding rows on CPU, not a full F32 table.
        from safetensors import safe_open
        name = oracle.TEXT+"embed.weight"
        with safe_open(str(args.snapshot/weights.index[name]), framework="pt", device="cpu") as stream:
            source = stream.get_slice(name)
            hidden = source[1000:1000+args.rows].cuda().float()
        norm = weights.tensor(oracle.TEXT+"embed_norm.weight")
        hidden = oracle.rms_norm(hidden, norm, cfg.rms_norm_eps)
        weights.close()
        failed = []
        args.out.mkdir(parents=True, exist_ok=True)
        for index in layers:
            layer = weights.load_layer(index, cfg)
            print("diagnostic owners", json.dumps(weights.owner_bytes), flush=True)
            if args.stage_taps:
                actual, expected = stage_probe(native, hidden, layer, cfg, index, weights, report)
            else:
                expected = oracle.layer_forward(hidden, [], layer, cfg, index)
                actual = native.layer(hidden, [], layer, cfg, index, weights)
            check(f"layer{index}", actual, expected, report, enforce=not args.stage_taps)
            metric = report[f"layer{index}"]
            if args.stage_taps:
                metrics = [report[f"layer{index}/mlp_out_matched_set"], report[f"layer{index}/final_lag_clean"]]
            else:
                metrics = [metric]
            if any(not m["finite"] or m["normalized_rms"] > .02 or m["cosine"] < .999 for m in metrics):
                failed.append(index)
            (args.out/"diagnostic.json").write_text(json.dumps(report, indent=2)+"\n")
            del layer, expected, actual, weights.arena, weights.expert_arena, weights.expert
            gc.collect()
            torch.cuda.empty_cache()
    (args.out/"diagnostic.json").write_text(json.dumps(report, indent=2)+"\n")
    if failed:
        raise AssertionError(f"unchanged diagnostic bar failed for layers {failed}; stage table retained")


if __name__ == "__main__":
    main()

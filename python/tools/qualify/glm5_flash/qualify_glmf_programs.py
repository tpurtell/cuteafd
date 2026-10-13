#!/usr/bin/env python3
"""Qualify the GLM 5.3 Flash coordinator programs (b12x ``glmf``) against transformers.

For each requested layer, feeds the golden streams entering that layer
(``runs/glmf-golden``, from python/reference/families/glm5_flash/golden.py) through
the reference Glm5NextTextDecoderLayer pieces and through the AOT programs
launched in process, and prints cosines per stage: mHC pre, the attention
sublayer (KDA or MLA), mHC post+pre, the dense FFN or the MoE router logits
and shared expert. KDA layers also run the prefix as a prefill and the last
``--decode`` rows as single-row steps (recurrent state and conv state carried
in the program's pools).

  PYTHONPATH=third_party/transformers/src USE_HUB_KERNELS=0 \
    qualify_glmf_programs.py --snapshot SNAP --golden runs/glmf-golden --layers 0 3 [--rows T]
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[2] / _d) for _d in ("lib",)]  # sibling tool dirs

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

import _pinned_sparkinfer  # noqa: F401  (verifies and prepends the pinned b12x tree)

ROOT = Path(__file__).resolve().parents[4]


def golden_module():
    spec = importlib.util.spec_from_file_location("glmf_golden", ROOT / "python/reference/families/glm5_flash/golden.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))


def report(name: str, ours: torch.Tensor, ref: torch.Tensor) -> float:
    c = cosine(ours, ref)
    rows = ours.reshape(ours.shape[0], -1).double()
    refs = ref.reshape(ref.shape[0], -1).double()
    per_row = (rows * refs).sum(-1) / (rows.norm(dim=-1) * refs.norm(dim=-1)).clamp_min(1e-30)
    rel = float((ours.double() - ref.double()).norm() / ref.double().norm().clamp_min(1e-30))
    print(f"  {name:28s} cosine {c:.6f} rel_l2 {rel:.2e} worst row {float(per_row.min()):.6f}", flush=True)
    return c


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--golden", type=Path, required=True)
    p.add_argument("--layers", type=int, nargs="+", default=[0, 3])
    p.add_argument("--rows", type=int, help="first T golden tokens (default all)")
    p.add_argument("--decode", type=int, default=8, help="KDA rows re-run as single-row steps")
    a = p.parse_args()

    from transformers import AutoConfig
    from transformers.models.glm5_next import modeling_glm5_next as ref

    from b12x.integration.cuteafd import GLM53_FLASH as g
    from b12x.integration.cuteafd import dsv4_mhc, glm_attention, glm_sparse_mla, glmf

    golden = golden_module()
    torch.backends.cuda.matmul.allow_tf32 = False
    config = AutoConfig.from_pretrained(a.snapshot).text_config
    config._attn_implementation = "eager"
    dense = golden.Weights(a.snapshot)
    tokens = np.fromfile(a.golden / "tokens.bin", dtype=np.int32)
    t = min(a.rows or len(tokens), len(tokens))
    h = config.hidden_size
    mask = torch.ones(1, t, dtype=torch.bool, device="cuda")
    mg = glmf.mhc_geometry(g)
    programs: dict[str, object] = {}

    def program(name, thunk):
        if name not in programs:
            programs[name] = thunk()
        return programs[name]

    def streams_before(layer: int) -> torch.Tensor:
        if layer == 0:
            ids = torch.tensor(tokens[:t].astype(np.int64), device="cuda")
            embed = torch.nn.functional.embedding(ids, dense.get(golden.PREFIX + "embed_tokens.weight"))
            return embed[:, None, :].expand(-1, 4, -1).contiguous()
        raw = np.fromfile(a.golden / f"layer{layer - 1:02d}.bin", dtype=np.int16)
        return torch.from_numpy(raw).view(torch.bfloat16).view(-1, 4, h)[:t].cuda()

    for layer_id in a.layers:
        kind = config.layer_types[layer_id]
        print(f"layer {layer_id} ({kind}, {config.mlp_layer_types[layer_id]}), {t} rows", flush=True)
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("meta"):
            layer = ref.Glm5NextTextDecoderLayer(config, layer_id)
        torch.set_default_dtype(torch.float32)
        layer = layer.to_empty(device="cuda")
        fp32_keys = ("conv1d", "dt_bias", "A_log", "e_score_correction_bias", "hc.base", "hc.scale")
        for name, param in list(layer.named_parameters()) + list(layer.named_buffers()):
            if any(k in name for k in fp32_keys):
                param.data = param.data.float()
        experts = getattr(layer.mlp, "experts", None)
        if experts is not None:
            # Routed experts are not qualified here; skip their 14 GB load.
            layer.mlp.experts = None
        prefix = f"{golden.PREFIX}layers.{layer_id}."
        golden.load_layer(layer, dense, dense, prefix)
        w = lambda name: dense.get(prefix + name)  # noqa: E731

        streams = streams_before(layer_id)
        with torch.inference_mode():
            post_ref, comb_ref, collapsed = layer.attn_hc(streams[None])
            x_ref = layer.input_layernorm(collapsed)[0]
            post_ref, comb_ref = post_ref[0], comb_ref[0]

            # mHC pre
            pre = program("mhc_pre", lambda: dsv4_mhc.compile_dsv4_mhc_pre_aot(mg))
            fn, scale, base = w("hc_attn_fn").float(), w("hc_attn_scale").float(), w("hc_attn_base").float()
            post = torch.empty(t, 4, device="cuda")
            comb = torch.empty(t, 4, 4, device="cuda")
            x = torch.empty(t, h, dtype=torch.bfloat16, device="cuda")
            scratch = torch.empty(dsv4_mhc.mhc_scratch_bytes(mg, t), dtype=torch.uint8, device="cuda")
            pre.launch(streams, fn, scale, base, w("input_layernorm.weight"), post, comb, x, scratch, scalars=[t])
            report("mhc_pre post", post, post_ref)
            report("mhc_pre comb", comb, comb_ref)
            report("mhc_pre y (attn input)", x, x_ref)

            if kind == "linear_attention":
                attn_ref = layer.self_attn(x_ref[None], cache_params=None, attention_mask=mask)[0]
                kda = program("kda", lambda: glmf.compile_glmf_kda_aot(g, max_rows=max(t, 64)))
                d = g.kda_width
                w_in = torch.cat([w(f"self_attn.{n}.weight") for n in
                                  ("q_proj", "k_proj", "v_proj", "f_a_proj", "g_a_proj", "b_proj")], 0).contiguous()
                w_fg = torch.stack([w("self_attn.f_b_proj.weight"), w("self_attn.g_b_proj.weight")], 0).contiguous()
                conv_w = torch.cat([w(f"self_attn.{n}_conv1d.weight") for n in "qkv"], 0)[:, 0, :].float().contiguous()
                args = (w_in, w_fg, conv_w, w("self_attn.A_log").float(), w("self_attn.dt_bias").float(),
                        w("self_attn.o_norm.weight"), w("self_attn.o_proj.weight"))
                scratch = torch.empty(kda.scratch_bytes(max(t, 64))["scratch"], dtype=torch.uint8, device="cuda")

                def run(rows_x, slots, seq_first, conv_state, state):
                    out = torch.empty(rows_x.shape[0], h, dtype=torch.bfloat16, device="cuda")
                    kda.launch(rows_x, *args, conv_state, state, slots, seq_first, out, scratch,
                               scalars=[rows_x.shape[0]])
                    return out

                conv_state = torch.zeros(2, 3, 3 * d, dtype=torch.bfloat16, device="cuda")
                state = torch.zeros(2, g.kda_heads, 128, 128, device="cuda")
                zeros = torch.zeros(t, dtype=torch.int32, device="cuda")
                out = run(x_ref, zeros, zeros, conv_state, state)
                report("kda prefill", out, attn_ref)
                # Prefix as one prefill, then single-row steps (slot 1).
                k = min(a.decode, t - 1)
                n0 = t - k
                ones = torch.ones(n0, dtype=torch.int32, device="cuda")
                steps = [run(x_ref[:n0], ones, torch.zeros(n0, dtype=torch.int32, device="cuda"), conv_state, state)]
                one = torch.ones(1, dtype=torch.int32, device="cuda")
                zero = torch.zeros(1, dtype=torch.int32, device="cuda")
                for r in range(n0, t):
                    steps.append(run(x_ref[r:r + 1], one, zero, conv_state, state))
                report(f"kda prefill {n0} + {k} steps", torch.cat(steps), attn_ref)
                report(f"kda last {k} step rows", torch.cat(steps[1:]), attn_ref[n0:])
                print(f"  kda state slot0 vs slot1 cosine {cosine(state[0], state[1]):.6f}")
            else:
                attn_ref = layer.self_attn(x_ref[None], attention_mask=mask, prev_topk_indices=None)[0][0]
                rows_cap = max(t, 64)
                producer = program("mla_producer", lambda: glmf.compile_glmf_mla_producer_aot(g, max_rows=rows_cap))
                mla = program("sparse_mla", lambda: glm_sparse_mla.compile_glm_sparse_mla_aot(
                    g, route="prefill", max_rows=rows_cap, name="glmf_sparse_mla"))
                o_prog = program("o", lambda: glm_attention.compile_glm_o_aot(g, max_rows=rows_cap))
                n, lat, nope = g.heads, g.kv_lora_rank, g.qk_nope_dim
                kv_b = w("self_attn.kv_b_proj.weight").view(n, nope + g.v_head_dim, lat)
                w_uk = kv_b[:, :nope, :].transpose(1, 2).contiguous()     # [N, 512, 256]
                w_uv = kv_b[:, nope:, :].contiguous()                     # [N, 256, 512]
                w_qkv_a = torch.cat([w("self_attn.q_a_proj.weight"), w("self_attn.kv_a_proj_with_mqa.weight")], 0)
                pages = -(-t // 64)
                cache = torch.zeros(pages, g.kv_page_bytes, dtype=torch.uint8, device="cuda")
                slots = torch.arange(t, dtype=torch.int64, device="cuda")
                query = torch.empty(t, n, lat, dtype=torch.bfloat16, device="cuda")
                q_resid = torch.empty(t, g.q_lora_rank, dtype=torch.bfloat16, device="cuda")
                scratch = torch.empty(glmf.mla_producer_scratch_bytes(g, rows_cap), dtype=torch.uint8, device="cuda")
                producer.launch(x_ref, slots, w_qkv_a.contiguous(), w("self_attn.q_a_layernorm.weight"),
                                w("self_attn.kv_a_layernorm.weight"), w("self_attn.q_b_proj.weight"), w_uk, cache,
                                query, q_resid, scratch, scalars=[t])
                q_resid_ref = layer.self_attn.q_a_layernorm(layer.self_attn.q_a_proj(x_ref))
                report("mla q_resid", q_resid, q_resid_ref)
                k = g.sparse_topk
                pos = torch.arange(t, device="cuda")
                # Indexer: per-token keys, pool keys, pool top-k, expansion to record slots.
                index_producer = program("index_producer",
                                         lambda: glmf.compile_glmf_index_producer_aot(g, max_rows=rows_cap))
                pool_pages = -(-t // (4 * 64))
                topk = program("index_topk", lambda: glmf.compile_glmf_index_topk_aot(
                    g, max_rows=rows_cap, max_pages=max(pool_pages, 1), mode="prefill"))
                expand = program("index_expand", lambda: glmf.compile_glmf_index_expand_aot(g))
                ix = layer.self_attn.indexer
                w_ik = torch.cat([ix.wk.weight, ix.weights_proj.weight, ix.index_kpool_compress_gate], 0).contiguous()
                pool_slots = torch.where(pos % 4 == 3, pos // 4, -1).to(torch.int64)
                token_keys = torch.zeros(pages * 64, 256, dtype=torch.bfloat16, device="cuda")
                index_cache = torch.zeros(pool_pages, 64 * 132, dtype=torch.uint8, device="cuda")
                q_fp8 = torch.empty(t, g.index_heads, 128, dtype=torch.float8_e4m3fn, device="cuda")
                head_w = torch.empty(t, g.index_heads, device="cuda")
                ip_scratch = torch.empty(glmf.index_producer_scratch_bytes(g, rows_cap), dtype=torch.uint8,
                                         device="cuda")
                index_producer.launch(x_ref, q_resid, slots, pool_slots, ix.wq_b.weight.contiguous(), w_ik,
                                      ix.k_norm.weight, ix.k_norm.bias, ix.index_kpool_compress_ape.contiguous(),
                                      token_keys, index_cache, q_fp8, head_w, ip_scratch, scalars=[t])
                pools = torch.full((t, g.index_topk // 4), -1, dtype=torch.int32, device="cuda")
                if t > g.index_topk + 3:
                    from b12x.integration.cuteafd.glm_indexer import glm_index_topk_scratch_bytes
                    pt = glmf._PoolTopK(index_topk=g.index_topk // 4, index_heads=g.index_heads)
                    tk_scratch = torch.zeros(glm_index_topk_scratch_bytes(pt, max_rows=rows_cap,
                                             max_pages=max(pool_pages, 1), mode="prefill"),
                                             dtype=torch.uint8, device="cuda")
                    topk.launch(q_fp8, head_w, index_cache, torch.arange(pool_pages, dtype=torch.int32, device="cuda"),
                                ((pos + 1) // 4).to(torch.int32), pools, tk_scratch, scalars=[t, pool_pages, 0])
                indices = torch.empty(t, k, dtype=torch.int32, device="cuda")
                lengths = torch.empty(t, dtype=torch.int32, device="cuda")
                expand.launch(pos.to(torch.int64), pools, torch.arange(pool_pages, dtype=torch.int32, device="cuda"),
                              torch.arange(pages, dtype=torch.int32, device="cuda"), indices, lengths,
                              scalars=[t, 0])
                ref_idx = ix(hidden_states=x_ref[None], q_resid=q_resid_ref[None], attention_mask=mask,
                             past_key_values=None)[0]
                same, jaccard, long_jaccard = 0, 0.0, 0.0
                ours_l, ref_l = indices.cpu(), ref_idx.cpu()
                for r in range(t):
                    mine = set(ours_l[r, :int(lengths[r])].tolist())
                    theirs = set(v for v in ref_l[r].tolist() if v >= 0)
                    same += mine == theirs
                    overlap = len(mine & theirs) / max(len(mine | theirs), 1)
                    jaccard += overlap
                    long_jaccard += overlap if r >= 2051 else 0.0
                print(f"  {'index selection vs reference':28s} identical rows {same}/{t}, mean Jaccard "
                      f"{jaccard / t:.5f}; rows past 2051: {max(t - 2051, 0)}, their Jaccard "
                      f"{long_jaccard / max(t - 2051, 1):.5f}", flush=True)
                latent = torch.empty(t, n, lat, dtype=torch.bfloat16, device="cuda")
                mla_scratch = torch.empty(glm_sparse_mla.sparse_mla_scratch_bytes(g, route="prefill", rows=rows_cap),
                                          dtype=torch.uint8, device="cuda")
                mla.launch(query, cache, indices, lengths, latent, mla_scratch, scalars=[t])
                out = torch.empty(t, h, dtype=torch.bfloat16, device="cuda")
                o_scratch = torch.empty(glm_attention.o_scratch_bytes(g, rows_cap), dtype=torch.uint8, device="cuda")
                o_prog.launch(latent, w_uv, w("self_attn.o_proj.weight"), out, o_scratch, scalars=[t])
                report("mla attention out", out, attn_ref)
                # Decode route on the last rows against the same cache.
                k_rows = min(a.decode, t)
                dec = program("sparse_mla_decode", lambda: glm_sparse_mla.compile_glm_sparse_mla_aot(
                    g, route="decode", max_rows=64, name="glmf_sparse_mla"))
                dec_scratch = torch.empty(glm_sparse_mla.sparse_mla_scratch_bytes(
                    g, route="decode", rows=64, buckets=glm_sparse_mla.decode_buckets(g, 64)),
                    dtype=torch.uint8, device="cuda")
                dec_latent = torch.empty(k_rows, n, lat, dtype=torch.bfloat16, device="cuda")
                dec.launch(query[t - k_rows:].contiguous(), cache, indices[t - k_rows:].contiguous(),
                           lengths[t - k_rows:].contiguous(), dec_latent, dec_scratch, scalars=[k_rows, 8])
                report(f"mla decode route, {k_rows} rows", dec_latent, latent[t - k_rows:])

            # mHC post + ffn pre
            streams_ref = post_ref.to(torch.bfloat16)[..., None] * attn_ref.reshape(t, 1, h) + torch.matmul(
                comb_ref.to(torch.bfloat16).transpose(-1, -2), streams)
            post2_ref, comb2_ref, collapsed2 = layer.ffn_hc(streams_ref[None])
            x2_ref = layer.post_attention_layernorm(collapsed2)[0]
            pp = program("mhc_post_pre", lambda: dsv4_mhc.compile_dsv4_mhc_post_pre_aot(mg, max_rows=max(t, 96)))
            streams_out = torch.empty_like(streams)
            post2 = torch.empty(t, 4, device="cuda")
            comb2 = torch.empty(t, 4, 4, device="cuda")
            x2 = torch.empty(t, h, dtype=torch.bfloat16, device="cuda")
            pp.launch(attn_ref.contiguous(), streams, post_ref.contiguous(), comb_ref.contiguous(),
                      w("hc_ffn_fn").float(), w("hc_ffn_scale").float(), w("hc_ffn_base").float(),
                      w("post_attention_layernorm.weight"), streams_out, post2, comb2, x2, scratch_mhc(mg, t),
                      scalars=[t])
            report("mhc_post_pre streams", streams_out, streams_ref)
            report("mhc_post_pre y (ffn input)", x2, x2_ref)

            if config.mlp_layer_types[layer_id] == "dense":
                ffn_ref = layer.mlp(x2_ref[None])[0]
                inter = config.intermediate_size
                ffn = program(f"ffn{inter}", lambda: glmf.compile_glmf_ffn_aot(g, inter=inter, max_rows=max(t, 64)))
                w_gu = torch.cat([w("mlp.gate_proj.weight"), w("mlp.up_proj.weight")], 0).contiguous()
                out = torch.empty(t, h, dtype=torch.bfloat16, device="cuda")
                from b12x.integration.cuteafd.glm_ffn import ffn_scratch_bytes
                fs = torch.empty(ffn_scratch_bytes(inter, max(t, 64)), dtype=torch.uint8, device="cuda")
                ffn.launch(x2_ref, w_gu, w("mlp.down_proj.weight"), out, fs, scalars=[t])
                report("dense ffn (clamped SwiGLU)", out, ffn_ref)
            else:
                router = program("router", lambda: glmf.compile_glmf_router_scores_aot(g))
                logits = torch.empty(t, g.routed_experts, device="cuda")
                router.launch(x2_ref, w("mlp.gate.weight"), logits, scalars=[t])
                logits_ref = layer.mlp.gate(x2_ref)[0]
                report("router logits (FP32)", logits, logits_ref)
                shared_ref = layer.mlp.shared_experts(x2_ref[None])[0]
                inter = config.moe_intermediate_size
                ffn = program(f"ffn{inter}", lambda: glmf.compile_glmf_ffn_aot(g, inter=inter, max_rows=max(t, 64)))
                w_gu = torch.cat([w("mlp.shared_experts.gate_proj.weight"), w("mlp.shared_experts.up_proj.weight")],
                                 0).contiguous()
                out = torch.empty(t, h, dtype=torch.bfloat16, device="cuda")
                from b12x.integration.cuteafd.glm_ffn import ffn_scratch_bytes
                fs = torch.empty(ffn_scratch_bytes(inter, max(t, 64)), dtype=torch.uint8, device="cuda")
                ffn.launch(x2_ref, w_gu, w("mlp.shared_experts.down_proj.weight"), out, fs, scalars=[t])
                report("shared expert", out, shared_ref)
        del layer
        torch.cuda.empty_cache()

    if a.layers and max(a.layers) >= 0:
        # Head: mean of the last golden streams + model.norm.
        last = config.num_hidden_layers - 1
        path = a.golden / f"layer{last:02d}.bin"
        if path.exists():
            raw = np.fromfile(path, dtype=np.int16)
            streams = torch.from_numpy(raw).view(torch.bfloat16).view(-1, 4, h)[:t].cuda()
            norm = ref.Glm5NextTextRMSNorm(h, config.rms_norm_eps).cuda().to(torch.bfloat16)
            with torch.inference_mode():
                norm.weight.copy_(dense.get(golden.PREFIX + "norm.weight"))
                head_ref = norm(streams.mean(dim=1))
                head = program("head", lambda: glmf.compile_glmf_head_aot(g))
                out = torch.empty(t, h, dtype=torch.bfloat16, device="cuda")
                head.launch(streams, dense.get(golden.PREFIX + "norm.weight"), out, scalars=[t])
            print("head")
            report("mean + norm", out, head_ref)


def scratch_mhc(mg, t: int) -> torch.Tensor:
    from b12x.integration.cuteafd import dsv4_mhc

    return torch.empty(dsv4_mhc.mhc_scratch_bytes(mg, max(t, 96)), dtype=torch.uint8, device="cuda")


if __name__ == "__main__":
    sys.exit(main())

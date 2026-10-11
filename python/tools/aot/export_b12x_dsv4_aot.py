#!/usr/bin/env python3
"""Export the coordinator programs (b12x.integration.cuteafd): DeepSeek V4
(``--geometry flash,pro``, families ``dsv4f``/``dsv4p``), GLM 5.x
(``--geometry glm``, family ``glm``), MiMo V2 Flash (``--geometry mimo``,
family ``mimo``) and GLM 5.3 Flash (``--geometry glmf``, family ``glmf``), in
any combination, into one table.

One object and header per program, a manifest with every program's pointer
ABI and scratch sizes at its capacity, and ``dsv4_programs.h``: the table the
generic native shim (native/shared/src/dsv4_programs.cc) launches from. Capacities are
compile-time: decode programs cover ``--decode-rows``, prefill programs
``--prefill-rows``, and cache extents follow ``--max-context``. GLM 5.3 Flash also
exports its decode programs at ``--glmf-wide-decode-rows`` (wider verify steps), after
every other program.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1] / _d) for _d in ("lib",)]  # sibling tool dirs

import argparse
import dataclasses
import hashlib
import json
import os
import re
from pathlib import Path

os.environ["B12X_COMPILE_DISK_CACHE"] = "0"
os.environ["B12X_COMPILE_MEMORY_CACHE"] = "0"
import _pinned_sparkinfer

SCALAR_KINDS = {"int32": "i", "int64": "l", "float32": "f"}
PAGE_ROWS = 64
DEFAULT_MAX_CONTEXT = 1048576
# max_position_embeddings from these checkpoints' config.json. Head splits
# retain the same context geometry. MiMo has no compiled index extent.
CHECKPOINT_CONTEXTS = {
    "flash": ("deepseek-ai/DeepSeek-V4-Flash-0731", 1048576),
    "pro": ("deepseek-ai/DeepSeek-V4-Pro-0813", 1048576),
    "glm": ("zai-org/GLM-5.3", 1048576),
    "glmf": ("zai-org/GLM-5.3-Flash", 1048576),
    "qwen4": ("Qwen/Qwen3.8-Flash-Next", 262144),
}


def geometry_context(name: str, requested: int) -> int:
    if requested <= 0:
        raise ValueError("max_context must be positive")
    base = name[:-1] if name.endswith("2") else name
    checkpoint = CHECKPOINT_CONTEXTS.get(base)
    return min(requested, checkpoint[1]) if checkpoint else requested


def validate_table_residency(stem: str, geometry: dict, *, diagnostic: bool = False) -> None:
    # Fused top-k has fixed CTA-group spin barriers. The native table has no
    # loaded-kernel occupancy gate/fallback; even a new decode bucket must not
    # silently introduce this route into serving.
    if geometry.get("route") == "paged_fused" and not diagnostic:
        raise ValueError(f"{stem}: co-resident paged_fused program requires a live-kernel "
                         "residency gate and grid-agnostic fallback before entering the serving table")


def programs(g, decode_rows: int, prefill_rows: int, max_context: int):
    """(stem suffix, op, params, compile thunk) for every exported program."""
    from b12x.integration.cuteafd import dsv4_compressor as comp
    from b12x.integration.cuteafd import dsv4_ffn as ffn
    from b12x.integration.cuteafd import weights
    from b12x.integration.cuteafd import dsv4_indexer as idx
    from b12x.integration.cuteafd import dsv4_mhc as mhc
    from b12x.integration.cuteafd import dsv4_producer as prod
    from b12x.integration.cuteafd import dsv4_sparse_mla as mla
    from b12x.integration.cuteafd import dsv4_wo as wo

    # Index cache: one row per four tokens, 64 rows per page.
    index_pages = -(-max_context // 4 // PAGE_ROWS)
    # C128 layers attend every completed compressed entry.
    c128_width = -(-max_context // 128 // 64) * 64
    out = [
        ("mhc_pre", "mhc_pre", {}, lambda: mhc.compile_dsv4_mhc_pre_aot(g)),
        ("mhc_post", "mhc_post", {}, lambda: mhc.compile_dsv4_mhc_post_aot(g)),
        ("mhc_head", "mhc_head", {}, lambda: mhc.compile_dsv4_mhc_head_aot(g)),
        ("router_scores", "router_scores", {}, lambda: ffn.compile_dsv4_router_scores_aot(g)),
        ("expert_input_quant", "expert_input_quant", {},
         lambda: ffn.compile_dsv4_expert_input_quant_aot(g)),
        # Load-time weight preparation (the rest of every weight is raw bytes).
        ("block_fp8_scale_prep", "block_fp8_scale_prep", {},
         lambda: weights.compile_dsv4_block_fp8_scale_prep_aot()),
        ("i64_to_i32", "i64_to_i32", {}, lambda: weights.compile_dsv4_i64_to_i32_aot()),
    ]
    for rows in (decode_rows, prefill_rows):
        out += [
            (f"mhc_post_pre_m{rows}", "mhc_post_pre", {"max_rows": rows},
             lambda r=rows: mhc.compile_dsv4_mhc_post_pre_aot(g, max_rows=r)),
            (f"producer_m{rows}", "producer", {"max_rows": rows},
             lambda r=rows: prod.compile_dsv4_producer_aot(g, max_rows=r)),
            (f"index_producer_m{rows}", "index_producer", {"max_rows": rows},
             lambda r=rows: prod.compile_dsv4_index_producer_aot(g, max_rows=r)),
            (f"wo_m{rows}", "wo", {"max_rows": rows},
             lambda r=rows: wo.compile_dsv4_wo_projection_aot(g, max_rows=r)),
            (f"shared_ffn_m{rows}", "shared_ffn", {"max_rows": rows},
             lambda r=rows: ffn.compile_dsv4_shared_ffn_aot(g, max_rows=r)),
        ]
    for ratio in (4, 128):
        for mode in ("decode", "prefill", "continuation"):
            fn = getattr(comp, f"compile_dsv4_compressor_{mode}_aot")
            out.append((f"compressor_{mode}_c{ratio}", f"compressor_{mode}", {"ratio": ratio},
                        lambda fn=fn, ratio=ratio: fn(g, ratio=ratio)))
    for mode, rows in (("decode", decode_rows), ("prefill", prefill_rows)):
        out.append((f"index_topk_{mode}_m{rows}", "index_topk",
                    {"mode": mode, "max_rows": rows, "max_pages": index_pages},
                    lambda mode=mode, rows=rows: idx.compile_dsv4_index_topk_aot(
                        g, max_rows=rows, max_pages=index_pages, mode=mode)))
    attention = (("win", 0, 64), ("c4", g.index_topk, 64), ("c128", c128_width, 2))
    for route, rows in (("decode", decode_rows), ("prefill", prefill_rows)):
        for kind, width, page_rows in attention:
            out.append((f"sparse_mla_{route}_{kind}_m{rows}", "sparse_mla",
                        {"route": route, "max_rows": rows, "indexed_width": width,
                         "indexed_page_rows": page_rows},
                        lambda route=route, rows=rows, width=width, page_rows=page_rows:
                            mla.compile_dsv4_sparse_mla_aot(
                                g, route=route, max_rows=rows, indexed_width=width,
                                indexed_page_rows=page_rows)))
    return out


def head_split_programs(g, decode_rows: int, prefill_rows: int, max_context: int):
    """One GPU's share of a two-GPU DeepSeek V4 head split (``g`` the half geometry: half the
    heads and output-projection groups, half the shared expert's intermediate): the producer
    (the replicated q_a / kv projection, its heads' queries), sparse MLA, wo (a partial over
    its groups) and the shared expert (a partial); mHC, the compressors, the indexer, router
    and expert input stay the whole model's programs."""
    keep = ("producer_m", "wo_m", "shared_ffn_m", "sparse_mla_")
    return [item for item in programs(g, decode_rows, prefill_rows, max_context) if item[0].startswith(keep)]


def glm_programs(g, decode_rows: int, prefill_rows: int, max_context: int):
    """GLM 5.x programs, same (stem suffix, op, params, thunk) shape as ``programs``."""
    from b12x.integration.cuteafd import glm_attention as attn
    from b12x.integration.cuteafd import glm_ffn as ffn
    from b12x.integration.cuteafd import glm_indexer as idx
    from b12x.integration.cuteafd import glm_sparse_mla as mla

    # Index and latent caches: one row per token, 64 rows per page.
    index_pages = -(-max_context // PAGE_ROWS)
    out = [
        ("norm", "norm", {}, lambda: ffn.compile_glm_norm_aot(g)),
        ("router_scores", "router_scores", {}, lambda: ffn.compile_glm_router_scores_aot(g)),
        ("expert_input_quant", "expert_input_quant", {}, lambda: ffn.compile_glm_expert_input_quant_aot(g)),
    ]
    # The checkpoint's FP8 weights (E4M3 + FP32 128x128 scales; kv_b per head with per-row
    # x 64-K scales) are the only copies: decode programs run the GEMV up to 16 rows and
    # W8A16 GEMMs above, prefill programs W8A8 (``fp8_rows`` nonzero) or W8A16.
    for rows in (decode_rows, prefill_rows):
        w8 = "decode" if rows == decode_rows else "prefill"
        out += [
            (f"producer_m{rows}", "producer", {"max_rows": rows, "fp8_only": w8},
             lambda r=rows, m=w8: attn.compile_glm_producer_aot(g, max_rows=r, fp8_only=m)),
            (f"index_producer_m{rows}", "index_producer", {"max_rows": rows, "fp8_only": w8},
             lambda r=rows, m=w8: attn.compile_glm_index_producer_aot(g, max_rows=r, fp8_only=m)),
            (f"o_m{rows}", "o", {"max_rows": rows, "fp8_only": w8},
             lambda r=rows, m=w8: attn.compile_glm_o_aot(g, max_rows=r, fp8_only=m)),
        ]
        for inter in (g.moe_inter, g.dense_inter):
            out.append((f"ffn_i{inter}_m{rows}", "ffn", {"max_rows": rows, "inter": inter, "fp8_only": w8},
                        lambda r=rows, i=inter, m=w8: ffn.compile_glm_ffn_aot(g, inter=i, max_rows=r, fp8_only=m)))
    # A checkpoint's BF16 attention, indexer and shared experts (nvidia/GLM-5.3-NVFP4) as-is: the
    # BF16 programs (skinny GEMV for few decode rows, BF16 tensor-core GEMMs above).
    for rows in (decode_rows, prefill_rows):
        out += [
            (f"producer_bf16_m{rows}", "producer", {"max_rows": rows, "weights": "bf16"},
             lambda r=rows: attn.compile_glm_producer_aot(g, max_rows=r)),
            (f"index_producer_bf16_m{rows}", "index_producer", {"max_rows": rows, "weights": "bf16"},
             lambda r=rows: attn.compile_glm_index_producer_aot(g, max_rows=r)),
            (f"o_bf16_m{rows}", "o", {"max_rows": rows, "weights": "bf16"},
             lambda r=rows: attn.compile_glm_o_aot(g, max_rows=r)),
            (f"ffn_i{g.moe_inter}_bf16_m{rows}", "ffn", {"max_rows": rows, "inter": g.moe_inter, "weights": "bf16"},
             lambda r=rows: ffn.compile_glm_ffn_aot(g, inter=g.moe_inter, max_rows=r)),
        ]
    # ModelOpt per-tensor FP8 dense MLPs (nvidia/GLM-5.3-NVFP4): static W8A8 prefill with the
    # checkpoint's input_scale and weight_scale (decode rows take the same bytes on the GEMV).
    out.append((f"ffn_i{g.dense_inter}_pt_m{prefill_rows}", "ffn",
                {"max_rows": prefill_rows, "inter": g.dense_inter, "fp8_only": "prefill", "tensor_scales": True},
                lambda: ffn.compile_glm_ffn_aot(g, inter=g.dense_inter, max_rows=prefill_rows, fp8_only="prefill",
                                                tensor_scales=True)))
    for mode, rows in (("decode", decode_rows), ("prefill", prefill_rows)):
        out.append((f"index_topk_{mode}_m{rows}", "index_topk",
                    {"mode": mode, "max_rows": rows, "max_pages": index_pages},
                    lambda mode=mode, rows=rows: idx.compile_glm_index_topk_aot(
                        g, max_rows=rows, max_pages=index_pages, mode=mode)))
        out.append((f"sparse_mla_{mode}_m{rows}", "sparse_mla", {"route": mode, "max_rows": rows},
                    lambda mode=mode, rows=rows: mla.compile_glm_sparse_mla_aot(g, route=mode, max_rows=rows)))
    return out


def glm_head_split_programs(g, decode_rows: int, prefill_rows: int, max_context: int):
    """One GPU's share of a two-GPU GLM 5.x head split (``g`` the half geometry: half the
    heads, half the dense and shared-expert intermediates): the MLA producer (the replicated
    latent record, its heads' queries), sparse MLA, W_UV + o_proj (a partial over its heads)
    and the dense / shared-expert MLPs (partials over their intermediate slices); the indexer,
    norms, router and expert input stay the whole model's programs."""
    keep = ("producer_m", "producer_bf16_m", "o_m", "o_bf16_m", "ffn_i", "sparse_mla_")
    return [item for item in glm_programs(g, decode_rows, prefill_rows, max_context) if item[0].startswith(keep)]


def mimo_programs(g, decode_rows: int, prefill_rows: int, max_context: int):
    """MiMo V2 programs, same (stem suffix, op, params, thunk) shape as ``programs``.
    Full-attention producers and attention come in both KV record formats (BF16, int8 ``_kvint8``). qkv and the dense FFN take only the checkpoint's E4M3 weights (per-row x 128-K FP32
    scales): decode rows up to ``fp8_rows`` on the GEMVs, W8A16 above; prefill W8A8
    (``fp8_rows`` nonzero) or W8A16. ``o_w8`` takes one E4M3 output weight for
    every row count, with no BF16 operand; its prefill activation precision is
    selected by ``fp8_rows``. ``o`` retains the older ABI for BF16 consumers.
    ``head_fp8`` takes the E4M3 head for up to 16 rows per launch."""
    from b12x.integration.cuteafd import mimo_attention as attn
    from b12x.integration.cuteafd import mimo_ffn as ffn

    out = [
        ("norm", "norm", {}, lambda: ffn.compile_mimo_norm_aot(g)),
        ("router_scores", "router_scores", {}, lambda: ffn.compile_mimo_router_scores_aot(g)),
        ("expert_input_quant", "expert_input_quant", {}, lambda: ffn.compile_mimo_expert_input_quant_aot(g)),
        ("head_fp8", "head_fp8", {}, lambda: attn.compile_mimo_head_fp8_aot(g)),
    ]
    for mode, rows in (("decode", decode_rows), ("prefill", prefill_rows)):
        f8 = mode == "decode"
        out += [
            (f"o_m{rows}", "o", {"max_rows": rows, "fp8": f8},
             lambda r=rows, f=f8: attn.compile_mimo_o_aot(g, max_rows=r, fp8=f)),
            (f"o_w8_m{rows}", "o", {"max_rows": rows, "fp8_only": mode},
             lambda r=rows, m=mode: attn.compile_mimo_o_aot(g, max_rows=r, fp8_only=m)),
            (f"ffn_m{rows}", "ffn", {"max_rows": rows, "inter": g.dense_inter, "fp8_only": mode},
             lambda r=rows, m=mode: ffn.compile_mimo_ffn_aot(g, max_rows=r, fp8_only=m)),
        ]
        # KV records: BF16 (no tag) and, for the full-attention page pool, int8 (``_kvint8``: FP32
        # scales per 32 dims). SWA rings stay BF16 (bounded per sequence; widening their 8-bit
        # records costs a decode step ~1.3%). The fork's E4M3 records are not exported (golden KL
        # +0.235 Flash / +0.012 V2.6 Pro over BF16).
        for kind in ("full", "swa"):
            for kv, tag in (("bf16", ""), ("int8", "_kvint8"))[:2 if kind == "full" else 1]:
                out += [
                    (f"{kind}_producer{tag}_m{rows}", "producer",
                     {"kind": kind, "max_rows": rows, "fp8_only": mode, "kv": kv},
                     lambda k=kind, r=rows, m=mode, c=kv: attn.compile_mimo_producer_aot(
                         g, kind=k, max_rows=r, fp8_only=m, kv=c)),
                    (f"{kind}_attention{tag}_{mode}_m{rows}", "attention",
                     {"kind": kind, "route": mode, "max_rows": rows, "kv": kv},
                     lambda k=kind, m=mode, r=rows, c=kv: attn.compile_mimo_attention_aot(
                         g, kind=k, route=m, max_rows=r, kv=c)),
                ]
    return out


def mimo_head_split_programs(g, decode_rows: int, prefill_rows: int, max_context: int):
    """One GPU's share of a two-GPU head split (``g`` the half geometry: half the query and
    KV heads, half the dense intermediate): the qkv producers, attention, o_proj (a partial
    over its heads) and dense FFN (a partial over its intermediate); norms, router, expert
    input and the LM head stay the whole model's programs."""
    keep = ("o_m", "o_w8_m", "ffn_m", "full_producer_", "swa_producer_", "full_attention_", "swa_attention_")
    return [item for item in mimo_programs(g, decode_rows, prefill_rows, max_context) if item[0].startswith(keep)]


def glmf_programs(g, decode_rows: int, prefill_rows: int, max_context: int):
    """GLM 5.3 Flash programs, same (stem suffix, op, params, thunk) shape as ``programs``.
    mHC is the DeepSeek V4 program set at this model's width and epsilons."""
    from b12x.integration.cuteafd import dsv4_mhc as mhc
    from b12x.integration.cuteafd import glm_sparse_mla as mla
    from b12x.integration.cuteafd import glmf

    mg = glmf.mhc_geometry(g)
    # Pool index cache: 64 pools (256 tokens) per page.
    pool_pages = -(-max_context // (g.index_kpool * PAGE_ROWS))
    out = [
        ("mhc_pre", "mhc_pre", {}, lambda: mhc.compile_dsv4_mhc_pre_aot(mg)),
        ("mhc_post", "mhc_post", {}, lambda: mhc.compile_dsv4_mhc_post_aot(mg)),
        ("head", "head", {}, lambda: glmf.compile_glmf_head_aot(g)),
        ("head_fp8", "head_fp8", {}, lambda: glmf.compile_glmf_head_fp8_aot(g)),
        ("add", "add", {}, lambda: glmf.compile_glmf_add_aot(g)),
        ("router_scores", "router_scores", {}, lambda: glmf.compile_glmf_router_scores_aot(g)),
        ("expert_input_quant", "expert_input_quant", {}, lambda: glmf.compile_glmf_expert_input_quant_aot(g)),
        ("index_expand", "index_expand", {}, lambda: glmf.compile_glmf_index_expand_aot(g)),
        ("kda_commit", "kda_commit", {}, lambda: glmf.compile_glmf_kda_commit_aot(g)),
        # The compact index cache (--index-cache compact): the commit that also rebuilds the
        # sequences' index tails, and below the producers without per-token keys.
        ("kda_commit_c", "kda_commit", {"index_cache": "compact"}, lambda: glmf.compile_glmf_kda_commit_c_aot(g)),
    ]
    # MLA, dense and shared-expert projections take only E4M3 weights with 128x128 scales
    # (the official FP8 release's): decode rows up to ``fp8_rows`` on the GEMV, W8A16
    # above; prefill W8A8 when ``fp8_rows`` is nonzero, else W8A16. KDA projections (BF16
    # in the release) run ``kda_m*`` over BF16 (the FP8 operands of those programs are the
    # retired dual-copy ABI), or ``kda_w8_m*`` over E4M3 only (per-row x 128-K scales,
    # K-block major, quantized at load): decode rows up to ``fp8_rows`` on the GEMV, W8A16
    # above; prefill W8A8 on ``fp8_rows`` bits (1 in-projection, 2 o_proj), else W8A16.
    for mode, rows in (("decode", decode_rows), ("prefill", prefill_rows)):
        f8 = True if mode == "decode" else "prefill"
        # Prefill mHC mixes run on TF32 tensor cores (split FP32 fn) from 384 rows.
        route = None if mode == "decode" else "tf32"
        out += [
            (f"index_producer_m{rows}", "index_producer", {"max_rows": rows},
             lambda r=rows: glmf.compile_glmf_index_producer_aot(g, max_rows=r)),
            (f"index_producer_c_m{rows}", "index_producer", {"max_rows": rows, "index_cache": "compact"},
             lambda r=rows: glmf.compile_glmf_index_producer_c_aot(g, max_rows=r)),
            (f"index_topk_{mode}_m{rows}", "index_topk", {"mode": mode, "max_rows": rows, "max_pages": pool_pages},
             lambda m=mode, r=rows: glmf.compile_glmf_index_topk_aot(g, max_rows=r, max_pages=pool_pages, mode=m)),
            (f"mhc_post_pre_m{rows}", "mhc_post_pre", {"max_rows": rows, "route": route},
             lambda r=rows, rt=route: mhc.compile_dsv4_mhc_post_pre_aot(mg, max_rows=r, route=rt)),
            (f"kda_m{rows}", "kda", {"max_rows": rows, "fp8": f8},
             lambda r=rows, f=f8: glmf.compile_glmf_kda_aot(g, max_rows=r, fp8=f)),
            (f"kda_w8_m{rows}", "kda", {"max_rows": rows, "fp8_only": mode},
             lambda r=rows, m=mode: glmf.compile_glmf_kda_aot(g, max_rows=r, fp8_only=m)),
            (f"mla_producer_m{rows}", "mla_producer", {"max_rows": rows, "fp8_only": mode},
             lambda r=rows, m=mode: glmf.compile_glmf_mla_producer_aot(g, max_rows=r, fp8_only=m)),
            (f"o_m{rows}", "o", {"max_rows": rows, "fp8_only": mode},
             lambda r=rows, m=mode: glmf.compile_glmf_o_aot(g, max_rows=r, fp8_only=m)),
            (f"sparse_mla_{mode}_m{rows}", "sparse_mla", {"route": mode, "max_rows": rows},
             lambda m=mode, r=rows: mla.compile_glm_sparse_mla_aot(g, route=m, max_rows=r,
                                      name="glmf_sparse_mla", fp32_partials=m == "decode")),
        ]
        for inter in (g.moe_inter, g.dense_inter):
            out.append((f"ffn_i{inter}_m{rows}", "ffn", {"max_rows": rows, "inter": inter, "fp8_only": mode},
                        lambda r=rows, i=inter, m=mode: glmf.compile_glmf_ffn_aot(g, inter=i, max_rows=r,
                                                                                  fp8_only=m)))
    # BF16 KDA recurrent state (serve --kda-state bf16): the BF16-projection KDA programs and the
    # replay commit over a BF16 state, computed in FP32 and rounded after every decode, verify and
    # commit row and where each chunked-prefill window stores it (``s16t``: after every 16-row
    # tile). New stems: the FP32 programs above, and their objects, are unchanged.
    for rows, f8, tag, rounding in ((decode_rows, True, "s16", "window"), (prefill_rows, "prefill", "s16", "window"),
                                    (prefill_rows, "prefill", "s16t", "tile")):
        out.append((f"kda_{tag}_m{rows}", "kda",
                    {"max_rows": rows, "fp8": f8, "state_dtype": "bfloat16", "state_rounding": rounding},
                    lambda r=rows, f=f8, sr=rounding: glmf.compile_glmf_kda_aot(
                        g, max_rows=r, fp8=f, state_dtype="bfloat16", state_rounding=sr)))
    out.append(("kda_commit_s16", "kda_commit", {"state_dtype": "bfloat16"},
                lambda: glmf.compile_glmf_kda_commit_aot(g, state_dtype="bfloat16")))
    # Both: the compact index cache's commit (state and index tails in one launch) over a BF16 state.
    out.append(("kda_commit_c_s16", "kda_commit", {"index_cache": "compact", "state_dtype": "bfloat16"},
                lambda: glmf.compile_glmf_kda_commit_c_aot(g, state_dtype="bfloat16")))
    return out


def glmf_wide_decode_programs(g, decode_rows: int, wide_rows: int, max_context: int):
    """GLM 5.3 Flash decode and verify steps of ``decode_rows`` < rows <= ``wide_rows`` (serve
    ``--decode-rows``): every decode program a step launches again at ``wide_rows`` rows, as new
    stems ``*_m{wide_rows}``, and the replay commits over records of that many rows
    (``kda_commit*_m{wide_rows}``). The KDA programs keep the decode structures at that capacity
    (the token-sequential recurrence, which records every verify row) and the mHC mix its decode
    route. None when ``wide_rows`` does not exceed ``decode_rows``. Steps of up to ``decode_rows``
    rows keep the ``glmf_programs`` above, whose stems and objects this list leaves alone (the
    exporter compiles it after every other program)."""
    if wide_rows <= decode_rows:
        return []
    from b12x.integration.cuteafd import dsv4_mhc as mhc
    from b12x.integration.cuteafd import glm_sparse_mla as mla
    from b12x.integration.cuteafd import glmf

    mg = glmf.mhc_geometry(g)
    pool_pages = -(-max_context // (g.index_kpool * PAGE_ROWS))
    r = int(wide_rows)
    out = [
        (f"index_producer_m{r}", "index_producer", {"max_rows": r},
         lambda: glmf.compile_glmf_index_producer_aot(g, max_rows=r)),
        (f"index_producer_c_m{r}", "index_producer", {"max_rows": r, "index_cache": "compact", "replay_rows": r},
         lambda: glmf.compile_glmf_index_producer_c_aot(g, max_rows=r, verify_rows=r)),
        (f"index_topk_decode_m{r}", "index_topk", {"mode": "decode", "max_rows": r, "max_pages": pool_pages},
         lambda: glmf.compile_glmf_index_topk_aot(g, max_rows=r, max_pages=pool_pages, mode="decode")),
        (f"mhc_post_pre_m{r}", "mhc_post_pre", {"max_rows": r, "route": "decode"},
         lambda: mhc.compile_dsv4_mhc_post_pre_aot(mg, max_rows=r, route="decode")),
        (f"kda_m{r}", "kda", {"max_rows": r, "fp8": True},
         lambda: glmf.compile_glmf_kda_aot(g, max_rows=r, fp8=True)),
        (f"kda_w8_m{r}", "kda", {"max_rows": r, "fp8_only": "decode"},
         lambda: glmf.compile_glmf_kda_aot(g, max_rows=r, fp8_only="decode")),
        (f"kda_s16_m{r}", "kda", {"max_rows": r, "fp8": True, "state_dtype": "bfloat16", "state_rounding": "window"},
         lambda: glmf.compile_glmf_kda_aot(g, max_rows=r, fp8=True, state_dtype="bfloat16", state_rounding="window")),
        (f"mla_producer_m{r}", "mla_producer", {"max_rows": r, "fp8_only": "decode"},
         lambda: glmf.compile_glmf_mla_producer_aot(g, max_rows=r, fp8_only="decode")),
        (f"o_m{r}", "o", {"max_rows": r, "fp8_only": "decode"},
         lambda: glmf.compile_glmf_o_aot(g, max_rows=r, fp8_only="decode")),
        # One split for the 128-row bucket, as the 64-row program's 64-row bucket: the split
        # planner finds no split count within its waves once the unsplit launch exceeds them
        # (128 rows x 4 head blocks, 512 CTAs on 170 SMs) and would split it 33 ways, FP32
        # partials of 33 chunks per row and head in a 554,729,472-byte scratch. On 188 SMs the
        # planner's own plan is one split, so the option changes no object there.
        (f"sparse_mla_decode_m{r}", "sparse_mla", {"route": "decode", "max_rows": r, "full_launch_splits": 1},
         lambda: mla.compile_glm_sparse_mla_aot(g, route="decode", max_rows=r, name="glmf_sparse_mla",
                                                fp32_partials=True, full_launch_splits=1)),
    ]
    for inter in (g.moe_inter, g.dense_inter):
        out.append((f"ffn_i{inter}_m{r}", "ffn", {"max_rows": r, "inter": inter, "fp8_only": "decode"},
                    lambda i=inter: glmf.compile_glmf_ffn_aot(g, inter=i, max_rows=r, fp8_only="decode")))
    # The commits read the speculative step's records: r rows per layer (glmf_kda_commit's 64).
    for stem, params, state in (("kda_commit", {}, "float32"), ("kda_commit_s16", {"state_dtype": "bfloat16"}, "bfloat16")):
        out.append((f"{stem}_m{r}", "kda_commit", {**params, "replay_rows": r},
                    lambda s=state: glmf.compile_glmf_kda_commit_aot(g, state_dtype=s, replay_rows=r)))
    for stem, params, state in (("kda_commit_c", {}, "float32"),
                                ("kda_commit_c_s16", {"state_dtype": "bfloat16"}, "bfloat16")):
        out.append((f"{stem}_m{r}", "kda_commit", {"index_cache": "compact", **params, "replay_rows": r},
                    lambda s=state: glmf.compile_glmf_kda_commit_c_aot(g, state_dtype=s, replay_rows=r)))
    return out


def glmf_head_split_programs(g, decode_rows: int, prefill_rows: int, max_context: int):
    """One GPU's share of a two-GPU GLM 5.3 Flash head split (``g`` the half geometry: half the
    MLA and KDA heads, half the dense and shared-expert intermediates): KDA (its heads'
    in-projection rows, conv, recurrence and state, a partial o_proj) and its verify-by-replay
    commit, the MLA producer (the replicated latent record, its heads' queries), sparse MLA,
    W_UV + o_proj (a partial over its heads) and the dense / shared-expert MLPs (partials over
    their intermediate slices); mHC, the DSA indexer, router, expert input and head stay the
    whole model's programs."""
    keep = ("kda_m", "kda_w8_m", "kda_commit", "mla_producer_m", "o_m", "sparse_mla_", "ffn_i")
    from b12x.integration.cuteafd import glmf

    # The head split keeps the per-token index keys and an FP32 KDA state (no compact index cache
    # or BF16-state ``_s16`` programs on two GPUs yet).
    programs = [item for item in glmf_programs(g, decode_rows, prefill_rows, max_context)
                if item[0].startswith(keep) and not item[0].startswith("kda_commit_c") and "_s16" not in item[0]]
    programs.append(("add_fp32", "add_fp32", {}, lambda: glmf.compile_glmf_add_fp32_aot(g)))
    programs.append(("join_heads", "join", {"half_width": g.kda_width},
                     lambda: glmf.compile_glmf_join_aot(g.kda_width)))
    programs.append(("join_rows", "join_rows", {"width": g.hidden},
                     lambda: glmf.compile_glmf_join_rows_aot(g.hidden)))
    for mode, rows in (("decode", decode_rows), ("prefill", prefill_rows)):
        programs.append((f"kda_w8_norm_m{rows}", "kda", {"max_rows": rows, "fp8_only": mode, "output_kind": "norm"},
                         lambda r=rows, m=mode: glmf.compile_glmf_kda_aot(
                             g, max_rows=r, fp8_only=m, output_kind="norm")))
        programs.append((f"kda_output_rows_m{rows}", "kda_output_rows", {"max_rows": rows, "fp8_only": mode},
                         lambda r=rows, m=mode: glmf.compile_glmf_kda_output_rows_aot(
                             g, max_rows=r, fp8_only=m)))
        for suffix, dtype in (("f32", "float32"),):
            programs.append((f"kda_w8_{suffix}_m{rows}", "kda", {"max_rows": rows, "fp8_only": mode, "output_dtype": dtype},
                             lambda r=rows, m=mode, d=dtype: glmf.compile_glmf_kda_aot(
                                 g, max_rows=r, fp8_only=m, output_dtype=d)))
    for suffix, dtype in (("", "bfloat16"), ("_f32", "float32")):
        programs.append((f"kda_w8{suffix}_expanded_m{prefill_rows}", "kda",
                         {"max_rows": prefill_rows, "fp8_only": "prefill", "output_dtype": dtype, "prefill_expanded": True},
                         lambda d=dtype: glmf.compile_glmf_kda_aot(
                             g, max_rows=prefill_rows, fp8_only="prefill", output_dtype=d, prefill_expanded=True)))
    programs.extend([
        (f"kda_w8_norm_expanded_m{prefill_rows}", "kda", {"max_rows": prefill_rows, "output_kind": "norm", "prefill_expanded": True},
         lambda: glmf.compile_glmf_kda_aot(g, max_rows=prefill_rows, fp8_only="prefill", output_kind="norm", prefill_expanded=True)),
        (f"kda_output_rows_expanded_m{prefill_rows}", "kda_output_rows", {"max_rows": prefill_rows, "prefill_expanded": True},
         lambda: glmf.compile_glmf_kda_output_rows_aot(g, max_rows=prefill_rows, fp8_only="prefill", prefill_expanded=True)),
    ])
    return programs


def qwen4_programs(g, decode_rows: int, prefill_rows: int, max_context: int):
    """Qwen 3.8 Flash Next programs, same (stem suffix, op, params, thunk) shape as ``programs``.
    Hyper-connection, head, MoE-front and PLE programs serve any row count (their
    scratch is sized from ``rows``); GDN and attention have decode/prefill capacities."""
    from b12x.integration.cuteafd import qwen4, qwen4_gdn

    out = [
        ("hc_pre", "hc_pre", {}, lambda: qwen4.compile_qwen4_hc_pre_aot(g)),
        ("hc_post_pre", "hc_post_pre", {}, lambda: qwen4.compile_qwen4_hc_post_pre_aot(g)),
        ("hc_post", "hc_post", {}, lambda: qwen4.compile_qwen4_hc_post_aot(g)),
        ("head", "head", {}, lambda: qwen4.compile_qwen4_head_aot(g)),
        ("router_scores", "router_scores", {}, lambda: qwen4.compile_qwen4_router_scores_aot(g)),
        ("shared", "shared", {}, lambda: qwen4.compile_qwen4_shared_aot(g)),
        ("add", "add", {}, lambda: qwen4.compile_qwen4_add_aot(g)),
        ("expert_input_quant", "expert_input_quant", {}, lambda: qwen4.compile_qwen4_expert_input_quant_aot(g)),
        ("ple_bf16", "ple", {"fp8": False}, lambda: qwen4.compile_qwen4_ple_aot(g, fp8=False)),
        ("ple_fp8", "ple", {"fp8": True}, lambda: qwen4.compile_qwen4_ple_aot(g, fp8=True)),
        # Speculation: MTP input feedback and the verify-by-replay commits.
        ("mtp_feedback", "mtp_feedback", {}, lambda: qwen4.compile_qwen4_mtp_feedback_aot(g)),
        ("head_fp8", "head_fp8", {}, lambda: qwen4.compile_qwen4_head_fp8_aot(g)),
        ("ple_commit", "ple_commit", {}, lambda: qwen4.compile_qwen4_ple_commit_aot(g)),
        ("gdn_commit", "gdn_commit", {}, lambda: qwen4_gdn.compile_qwen4_gdn_commit_aot(g)),
    ]
    from b12x.integration.cuteafd import qwen4_attention as attn

    out.append(("index_expand", "index_expand", {}, lambda: attn.compile_qwen4_index_expand_aot(g)))
    for rows in (decode_rows, prefill_rows):
        out += [
            (f"gdn_m{rows}", "gdn", {"max_rows": rows}, lambda r=rows: qwen4_gdn.compile_qwen4_gdn_aot(g, max_rows=r)),
            (f"attn_producer_m{rows}", "attn_producer", {"max_rows": rows},
             lambda r=rows: attn.compile_qwen4_attn_producer_aot(g, max_rows=r)),
            (f"index_topk_m{rows}", "index_topk", {"max_rows": rows, "max_context": max_context},
             lambda r=rows: attn.compile_qwen4_index_topk_aot(g, max_rows=r, max_context=max_context)),
            (f"sparse_gqa_m{rows}", "sparse_gqa", {"max_rows": rows},
             lambda r=rows: attn.compile_qwen4_sparse_gqa_aot(g, max_rows=r)),
            (f"attn_o_m{rows}", "attn_o", {"max_rows": rows}, lambda r=rows: attn.compile_qwen4_attn_o_aot(g, max_rows=r)),
        ]
    # Decode programs over E4M3 copies of the large projections (<= 16 live rows).
    rows = decode_rows
    out += [
        (f"gdn_fp8_m{rows}", "gdn", {"max_rows": rows, "fp8": True},
         lambda: qwen4_gdn.compile_qwen4_gdn_aot(g, max_rows=rows, fp8=True)),
        (f"attn_producer_fp8_m{rows}", "attn_producer", {"max_rows": rows, "fp8": True},
         lambda: attn.compile_qwen4_attn_producer_aot(g, max_rows=rows, fp8=True)),
        (f"attn_o_fp8_m{rows}", "attn_o", {"max_rows": rows, "fp8": True},
         lambda: attn.compile_qwen4_attn_o_aot(g, max_rows=rows, fp8=True)),
    ]
    # Single-copy FP8 projections (serve-qwen4 --fp8-decode): the GDN in/out and attention
    # in/o weights as E4M3 + FP32 128x128 block scales only. Decode: 16-row GEMV, W8A16 above;
    # prefill: ``fp8_rows`` 0 W8A16, nonzero W8A8 (the GDN in-projection stays W8A16).
    # (``cap``, not ``rows``: the lambdas above read ``rows`` when they run.)
    for mode, cap in (("decode", decode_rows), ("prefill", prefill_rows)):
        out += [
            (f"gdn_w8_m{cap}", "gdn", {"max_rows": cap, "fp8_only": mode},
             lambda r=cap, m=mode: qwen4_gdn.compile_qwen4_gdn_aot(g, max_rows=r, fp8_only=m)),
            (f"attn_producer_w8_m{cap}", "attn_producer", {"max_rows": cap, "fp8_only": mode},
             lambda r=cap, m=mode: attn.compile_qwen4_attn_producer_aot(g, max_rows=r, fp8_only=m)),
            (f"attn_o_w8_m{cap}", "attn_o", {"max_rows": cap, "fp8_only": mode},
             lambda r=cap, m=mode: attn.compile_qwen4_attn_o_aot(g, max_rows=r, fp8_only=m)),
        ]
    # KV format is independent of projection-weight format; preserve the BF16 exports.
    for cap in (decode_rows, prefill_rows):
        out += [
            (f"attn_producer_kv_fp8_m{cap}", "attn_producer", {"max_rows": cap, "kv_format": "fp8"},
             lambda r=cap: attn.compile_qwen4_attn_producer_aot(g, max_rows=r, kv_format="fp8")),
            (f"sparse_gqa_kv_fp8_m{cap}", "sparse_gqa", {"max_rows": cap, "kv_format": "fp8"},
             lambda r=cap: attn.compile_qwen4_sparse_gqa_aot(g, max_rows=r, kv_format="fp8")),
        ]
    out.append((f"attn_producer_fp8_kv_fp8_m{decode_rows}", "attn_producer",
                {"max_rows": decode_rows, "fp8": True, "kv_format": "fp8"},
                lambda: attn.compile_qwen4_attn_producer_aot(g, max_rows=decode_rows, fp8=True, kv_format="fp8")))
    for mode, cap in (("decode", decode_rows), ("prefill", prefill_rows)):
        out.append((f"attn_producer_w8_kv_fp8_m{cap}", "attn_producer",
                    {"max_rows": cap, "fp8_only": mode, "kv_format": "fp8"},
                    lambda r=cap, m=mode: attn.compile_qwen4_attn_producer_aot(g, max_rows=r, fp8_only=m, kv_format="fp8")))
    return out


def context_split_programs(g, decode_rows: int, max_context: int):
    """Opt-in K1 primitives only; never change a serving table by default."""
    from b12x.integration.cuteafd import context_split as split

    v4 = hasattr(g, "o_groups")
    pooled = v4 or hasattr(g, "index_kpool")
    k = g.index_topk // getattr(g, "index_kpool", 1)
    pages = -(-max_context // (4 if pooled else 1) // PAGE_ROWS)
    out = [
        (f"scored_index_topk_decode_m{decode_rows}", "scored_index_topk",
         {"max_rows": decode_rows, "max_pages": pages, "mode": "decode"},
         lambda: split.compile_scored_index_topk_aot(g, max_rows=decode_rows, max_pages=pages)),
        ("dsa_candidate_merge", "dsa_candidate_merge", {"topk": k},
         lambda: split.compile_dsa_candidate_merge_aot(topk=k)),
        ("lse_combine2", "lse_combine2", {"heads": g.heads // 2, "has_sink": v4},
         lambda: split.compile_lse_combine2_aot(heads=g.heads // 2, has_sink=v4)),
    ]
    # Both SM120 products share the same table; select its explicit split
    # variant once at load, not whichever GPU happened to run the exporter.
    device_plans = {sms: split.sparse_mla_partial_split_plan(
        g, max_rows=decode_rows, head_count=g.heads // 2, sm_count=sms) for sms in (170, 188)}
    for begin in (0, g.heads // 2):
        for splits in sorted(set(device_plans.values())):
            out.append((f"sparse_mla_partial_h{begin}_s{splits}_m{decode_rows}", "sparse_mla_partial",
                        {"max_rows": decode_rows, "head_begin": begin, "head_count": g.heads // 2,
                         "num_splits": splits, "device_sm_counts": [sm for sm, s in device_plans.items() if s == splits]},
                        lambda begin=begin, splits=splits: split.compile_sparse_mla_partial_aot(
                            g, max_rows=decode_rows, head_begin=begin, head_count=g.heads // 2, num_splits=splits)))
    layouts = [("index", 128, 64, 8448)]
    layouts += ([("swa", 584, 256, 149760), ("c4", 584, 64, 37440)] if v4 else
                [("kv", g.record_bytes, g.page_rows, g.kv_page_bytes)])
    for kind, row_bytes, page_rows, page_bytes in layouts:
        out.append((f"paged_staging_gather_{kind}", "paged_staging_gather",
                    {"row_bytes": row_bytes, "page_rows": page_rows, "page_bytes": page_bytes},
                    lambda rb=row_bytes, pr=page_rows, pb=page_bytes:
                        split.compile_paged_staging_gather_aot(row_bytes=rb, page_rows=pr, page_bytes=pb)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--geometry", default="flash",
                        help="comma-separated geometries in one table: flash, pro (DeepSeek V4; flash2 / pro2: one GPU of "
                             "its two-GPU head split), glm (GLM 5.x), "
                             "glm2 (GLM 5.x, one GPU of a two-GPU head split), mimo (MiMo V2 Flash), mimop (MiMo V2.6 Pro), "
                             "mimop2 (V2.6 Pro, one GPU of a two-GPU head split), mimof/mimof2 (V2.6 Flash MOPD), "
                             "glmf (GLM 5.3 Flash), glmf2 (GLM 5.3 Flash, one GPU of a two-GPU head split), "
                             "qwen4 (Qwen 3.8 Flash Next)")
    parser.add_argument("--decode-rows", type=int, default=64)
    parser.add_argument("--glmf-wide-decode-rows", type=int, default=128,
                        help="with glmf: also the decode and verify programs at this many rows (serve "
                             "--decode-rows 128), new stems after every other program; 0 exports none")
    parser.add_argument("--prefill-rows", type=int, default=4096)
    parser.add_argument("--max-context", type=int, default=DEFAULT_MAX_CONTEXT)
    parser.add_argument("--only", help="comma-separated stem suffixes (diagnostics)")
    parser.add_argument("--context-split-only", action="store_true",
                        help="export K1 primitives for flash/pro/glm/glmf, no engine wiring")
    args = parser.parse_args()

    import torch
    from b12x.integration.cuteafd import (
        FLASH, GLM53, GLM53_FLASH, MIMO_V2_FLASH, MIMO_V26_FLASH, MIMO_V26_PRO, PRO, QWEN38_FLASH_NEXT, exportable_compilation,
        validate_exported_header,
    )

    geometries = [name.strip() for name in args.geometry.split(",") if name.strip()]
    if not geometries or any(name not in ("flash", "flash2", "pro", "pro2", "glm", "glm2", "mimo", "mimo2", "mimop",
                                          "mimop2", "mimof", "mimof2", "glmf", "glmf2", "qwen4") for name in geometries):
        raise SystemExit("--geometry takes flash, flash2, pro, pro2, glm, glm2, mimo, mimo2, mimop, mimop2, mimof, mimof2, "
                         "glmf, glmf2 and/or qwen4")
    if args.context_split_only and any(name not in ("flash", "pro", "glm", "glmf") for name in geometries):
        raise SystemExit("--context-split-only takes flash, pro, glm and/or glmf")
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) not in ((12, 0), (12, 1)) or (props.minor != 0 and not args.context_split_only):
        raise SystemExit("coordinator programs export on SM120 (K1 primitives also on SM121)")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    selected = set(args.only.split(",")) if args.only else None
    manifest = {
        "schema": 1,
        "families": {},
        "family_capacities": {},
        "capacities": {"decode_rows": args.decode_rows, "prefill_rows": args.prefill_rows,
                       "max_context": args.max_context},
        "sparkinfer_revision": _pinned_sparkinfer.REVISION,
        "capability": [props.major, props.minor],
        "programs": [],
    }
    entries, includes = [], []
    work = []
    for name in geometries:
        # mimop2: V2.6 Pro split over two GPUs by heads (64 query / 4 KV heads each, the
        # dense MLP by intermediate).
        mimo2 = dataclasses.replace(MIMO_V2_FLASH, name="mimo_v2_flash_tp2", heads=MIMO_V2_FLASH.heads // 2,
                                    full_kv_heads=MIMO_V2_FLASH.full_kv_heads // 2,
                                    swa_kv_heads=MIMO_V2_FLASH.swa_kv_heads // 2,
                                    dense_inter=MIMO_V2_FLASH.dense_inter // 2)
        mimop2 = dataclasses.replace(MIMO_V26_PRO, name="mimo_v2_pro_tp2", heads=MIMO_V26_PRO.heads // 2,
                                     full_kv_heads=MIMO_V26_PRO.full_kv_heads // 2,
                                     swa_kv_heads=MIMO_V26_PRO.swa_kv_heads // 2,
                                     dense_inter=MIMO_V26_PRO.dense_inter // 2)
        mimof2 = dataclasses.replace(MIMO_V26_FLASH, name="mimo_v26_flash_tp2", heads=MIMO_V26_FLASH.heads // 2,
                                     full_kv_heads=MIMO_V26_FLASH.full_kv_heads // 2,
                                     swa_kv_heads=MIMO_V26_FLASH.swa_kv_heads // 2,
                                     dense_inter=MIMO_V26_FLASH.dense_inter // 2)
        # glm2: GLM 5.x split over two GPUs by heads (32 each; dense and shared MLPs by
        # intermediate).
        glm2 = dataclasses.replace(GLM53, name="glm53_tp2", heads=GLM53.heads // 2, dense_inter=GLM53.dense_inter // 2,
                                   moe_inter=GLM53.moe_inter // 2)
        # glmf2: GLM 5.3 Flash split over two GPUs by heads (32 MLA and 32 KDA heads each; dense
        # and shared MLPs by intermediate).
        glmf2 = dataclasses.replace(GLM53_FLASH, name="glm53_flash_tp2", heads=GLM53_FLASH.heads // 2,
                                    kda_heads=GLM53_FLASH.kda_heads // 2, dense_inter=GLM53_FLASH.dense_inter // 2,
                                    moe_inter=GLM53_FLASH.moe_inter // 2)
        # flash2 / pro2: DeepSeek V4 split over two GPUs by heads (and wo groups; the shared
        # expert by intermediate).
        dsv4_half = {n: dataclasses.replace(base, name=f"{base.name}_tp2", heads=base.heads // 2,
                                            o_groups=base.o_groups // 2, moe_inter=base.moe_inter // 2)
                     for n, base in (("flash2", FLASH), ("pro2", PRO))}
        g = {"flash": FLASH, "flash2": dsv4_half["flash2"], "pro": PRO, "pro2": dsv4_half["pro2"], "glm": GLM53,
             "glm2": glm2, "mimo": MIMO_V2_FLASH, "mimop": MIMO_V26_PRO,
             "mimo2": mimo2, "mimop2": mimop2, "mimof": MIMO_V26_FLASH, "mimof2": mimof2, "glmf": GLM53_FLASH, "glmf2": glmf2, "qwen4": QWEN38_FLASH_NEXT}[name]
        family = {"flash": "dsv4f", "flash2": "dsv4f2", "pro": "dsv4p", "pro2": "dsv4p2", "glm": "glm", "glm2": "glm2", "mimo": "mimo", "mimop": "mimop",
                  "mimo2": "mimo2", "mimop2": "mimop2", "mimof": "mimof", "mimof2": "mimof2", "glmf": "glmf", "glmf2": "glmf2", "qwen4": "qwen4"}[name]
        manifest["families"][family] = {k: v for k, v in vars(g).items()}
        extent = geometry_context(name, args.max_context)
        manifest["family_capacities"][family] = {"max_context": extent}
        make = {"flash2": head_split_programs, "pro2": head_split_programs, "glm": glm_programs,
                "glm2": glm_head_split_programs, "mimo": mimo_programs, "mimop": mimo_programs,
                "mimo2": mimo_head_split_programs, "mimop2": mimo_head_split_programs,
                "mimof": mimo_programs, "mimof2": mimo_head_split_programs,
                "glmf": glmf_programs, "glmf2": glmf_head_split_programs,
                "qwen4": qwen4_programs}.get(name, programs)
        family_work = (context_split_programs(g, args.decode_rows, extent) if args.context_split_only else
                       make(g, args.decode_rows, args.prefill_rows, extent))
        work += [(family, *item) for item in family_work]
    if "glmf" in geometries and not args.context_split_only:
        # Last, so every program above compiles exactly as before.
        work += [("glmf", *item) for item in glmf_wide_decode_programs(GLM53_FLASH, args.decode_rows,
                                                                       args.glmf_wide_decode_rows, geometry_context("glmf", args.max_context))]
    for family, suffix, op, params, thunk in work:
        if selected is not None and suffix not in selected:
            continue
        stem = f"{family}_{suffix}"
        with exportable_compilation():
            program = thunk()
        validate_table_residency(stem, program.geometry, diagnostic=selected is not None)
        program.export_to_c(str(output), stem, "cuteafd_" + stem)
        header = output / f"{stem}.h"
        checked = validate_exported_header(program, header, "cuteafd_" + stem)
        abi = program.abi
        kinds = "".join(SCALAR_KINDS[kind] for _, kind in abi["scalars"])
        if checked["argument_count"] != len(abi["pointers"]) + len(kinds) + 1:
            raise ValueError(f"{stem}: header argument count disagrees with the ABI")
        capacity = int(params.get("max_rows", args.prefill_rows))
        manifest["programs"].append({
            "name": stem,
            "family": family,
            "op": op,
            "params": params,
            "geometry": dict(program.geometry),
            "pointers": [dict(zip(("name", "dtype", "shape", "role"), p)) for p in abi["pointers"]],
            "scalars": [dict(zip(("name", "type"), s)) for s in abi["scalars"]],
            "scratch_bytes_at_capacity": program.scratch_bytes(capacity),
            "capacity_rows": capacity,
            "entry": checked["symbol"],
            "object_sha256": hashlib.sha256((output / f"{stem}.o").read_bytes()).hexdigest(),
        })
        includes.append(f'#include "{stem}.h"')
        entries.append(
            f'{{"{stem}", _mlir_cuteafd_{stem}_cuda_init, _mlir_cuteafd_{stem}_cuda_load_to_device, '
            f'{checked["symbol"]}, {len(abi["pointers"])}, "{kinds}"}}'
        )
        print(f"exported {stem}: {len(abi['pointers'])} pointers, scalars '{kinds}'", flush=True)
    if not entries:
        raise SystemExit("no programs selected")
    (output / "dsv4_programs.h").write_text("\n".join([
        "#pragma once",
        *includes,
        f"#define CUTEAFD_DSV4_CC_MINOR {props.minor}",
        "#define CUTEAFD_DSV4_PROGRAMS " + ", ".join(entries),
        "",
    ]))
    (output / "dsv4_programs.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

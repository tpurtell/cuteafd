#!/usr/bin/env python3
"""Export native Hy4-flash coordinator AOT programs from the verified SparkInfer fork."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import _pinned_sparkinfer


GROUPS = ("ihc", "engram", "attention", "indexer", "pointwise", "projections", "draft")


def builders(tp: int, group: str, *, decode_rows: int = 64, max_rows: int = 4096, max_context: int = 262144):
    """Shared compiler definitions for Hy4-flash TP1 (hy4f) and TP2 (hy4f2)."""
    if tp not in (1, 2):
        raise ValueError(f"Hy4-flash supports TP1 or TP2 coordinator shards, got tp={tp}")
    dsa_heads = 32 // tp
    swa_heads = 64 // tp
    dense_inter = 13824 // tp
    shared_inter = 1536 // tp

    if group == "ihc":
        from b12x.integration.cuteafd import hy4_ihc as ihc

        return {
            "ihc_pre": lambda: ihc.compile_hy4_ihc_pre_aot(ihc.HY4_FLASH_IHC),
            "ihc_post": lambda: ihc.compile_hy4_ihc_post_aot(ihc.HY4_FLASH_IHC),
            "ihc_post_pre": lambda: ihc.compile_hy4_ihc_post_pre_aot(ihc.HY4_FLASH_IHC),
            "hc_head": lambda: ihc.compile_hy4_hc_head_aot(ihc.HY4_FLASH_IHC),
        }

    if group == "engram":
        from b12x.integration.cuteafd import hy4_engram as eg

        return {
            "engram_head_norm": lambda: eg.compile_hy4_engram_head_norm_aot(eg.HY4_FLASH_ENGRAM),
            "engram_gate_add": lambda: eg.compile_hy4_engram_gate_add_aot(eg.HY4_FLASH_ENGRAM),
        }

    if group == "attention":
        from b12x.integration.cuteafd import hy4_mla as mla

        dsa_bf16 = mla.Hy4MlaGeometry(heads=dsa_heads, window_size=None, kv_cache="bf16")
        dsa_fp8 = mla.Hy4MlaGeometry(heads=dsa_heads, window_size=None, kv_cache="fp8")
        swa_bf16 = mla.Hy4MlaGeometry(heads=swa_heads, window_size=512, kv_cache="bf16")
        swa_fp8 = mla.Hy4MlaGeometry(heads=swa_heads, window_size=512, kv_cache="fp8")
        entries = {
            "mla_rank_norm_pack_bf16": lambda: mla.compile_hy4_mla_rank_norm_pack_aot(dsa_bf16),
            "mla_rank_norm_pack_fp8": lambda: mla.compile_hy4_mla_rank_norm_pack_aot(dsa_fp8),
            "mla_q_absorb_dsa": lambda: mla.compile_hy4_mla_q_absorb_aot(dsa_bf16),
            "mla_q_absorb_swa": lambda: mla.compile_hy4_mla_q_absorb_aot(swa_bf16),
            "mla_sink_gate_uv_dsa": lambda: mla.compile_hy4_mla_sink_gate_uv_aot(dsa_bf16),
            "mla_sink_gate_uv_swa": lambda: mla.compile_hy4_mla_sink_gate_uv_aot(swa_bf16),
            "mla_dsa_decode_bf16": lambda: mla.compile_hy4_mla_attention_aot(
                dsa_bf16, max_rows=max_rows, max_context=2048, mode="decode", page_size=1
            ),
            "mla_dsa_decode_fp8": lambda: mla.compile_hy4_mla_attention_aot(
                dsa_fp8, max_rows=max_rows, max_context=2048, mode="decode", page_size=1
            ),
            "mla_swa_decode_bf16": lambda: mla.compile_hy4_mla_attention_aot(
                swa_bf16, max_rows=decode_rows, max_context=512, mode="decode", page_size=64
            ),
            "mla_swa_decode_fp8": lambda: mla.compile_hy4_mla_attention_aot(
                swa_fp8, max_rows=decode_rows, max_context=512, mode="decode", page_size=64
            ),
            "mla_swa_extend_bf16": lambda: mla.compile_hy4_mla_attention_aot(
                swa_bf16, max_rows=max_rows, max_context=max_context, mode="extend", page_size=64
            ),
            "mla_swa_extend_fp8": lambda: mla.compile_hy4_mla_attention_aot(
                swa_fp8, max_rows=max_rows, max_context=max_context, mode="extend", page_size=64
            ),
        }
        if tp == 2:
            from b12x.integration.cuteafd import context_split as cs

            dsa32_bf16 = mla.Hy4MlaGeometry(heads=32, window_size=None, kv_cache="bf16")
            dsa32_fp8 = mla.Hy4MlaGeometry(heads=32, window_size=None, kv_cache="fp8")
            entries.update({
                "mla_dsa_partial_bf16_h0": lambda: mla.compile_hy4_mla_dsa_partial_aot(
                    dsa32_bf16, max_rows=max_rows, head_begin=0, head_count=16, num_splits=16, row_chunk_size=64
                ),
                "mla_dsa_partial_bf16_h16": lambda: mla.compile_hy4_mla_dsa_partial_aot(
                    dsa32_bf16, max_rows=max_rows, head_begin=16, head_count=16, num_splits=16, row_chunk_size=64
                ),
                "mla_dsa_partial_fp8_h0": lambda: mla.compile_hy4_mla_dsa_partial_aot(
                    dsa32_fp8, max_rows=max_rows, head_begin=0, head_count=16, num_splits=16, row_chunk_size=64
                ),
                "mla_dsa_partial_fp8_h16": lambda: mla.compile_hy4_mla_dsa_partial_aot(
                    dsa32_fp8, max_rows=max_rows, head_begin=16, head_count=16, num_splits=16, row_chunk_size=64
                ),
                "lse_combine2": lambda: cs.compile_lse_combine2_aot(heads=16, has_sink=True),
                "mla_gate_uv_dsa": lambda: mla.compile_hy4_mla_gate_uv_aot(dsa_bf16),
            })
        return entries

    if group == "indexer":
        from b12x.integration.cuteafd import hy4_indexer as idx

        entries = {
            "indexer_query_rope": lambda: idx.compile_hy4_indexer_query_rope_aot(idx.HY4_FLASH_INDEXER),
            "indexer_pool_group": lambda: idx.compile_hy4_indexer_pool_group_aot(idx.HY4_FLASH_INDEXER),
            "indexer_score_groups": lambda: idx.compile_hy4_indexer_score_groups_aot(idx.HY4_FLASH_INDEXER),
            "indexer_topk_expand": lambda: idx.compile_hy4_indexer_topk_expand_aot(idx.HY4_FLASH_INDEXER),
        }
        if tp == 2:
            entries.update({
                "indexer_scored_topk": lambda: idx.compile_hy4_indexer_scored_topk_aot(idx.HY4_FLASH_INDEXER),
                "indexer_merge_expand_owned": lambda: idx.compile_hy4_indexer_merge_expand_owned_aot(idx.HY4_FLASH_INDEXER),
            })
        return entries

    import torch
    from b12x.integration.cuteafd import pointwise as p

    if group == "pointwise":
        from b12x.integration.cuteafd.dsv4_ffn import compile_dsv4_expert_input_quant_aot

        return {
            "rms_norm_3072": lambda: p.compile_affine_norm_aot(
                width=3072, eps=1e-5, input_dtype=torch.bfloat16, parameter_dtype=torch.bfloat16
            ),
            "rms_norm_128": lambda: p.compile_affine_norm_aot(
                width=128, eps=1e-5, input_dtype=torch.bfloat16, parameter_dtype=torch.bfloat16
            ),
            "route_weights": lambda: p.compile_route_weights_aot(
                experts=512, topk=16, epsilon=1e-20, factor=3.998
            ),
            "expert_input_quant": lambda: compile_dsv4_expert_input_quant_aot(SimpleNamespace(hidden=1536)),
            "post_silu_split_dense": lambda: p.compile_post_silu_split_aot(width=dense_inter, limit=1e30),
            "post_silu_split_shared": lambda: p.compile_post_silu_split_aot(width=shared_inter, limit=1e30),
        }

    if group == "projections":
        reduce_dtype = torch.bfloat16 if tp == 1 else torch.float32
        shapes = {
            "q_a": (3072, 1024, torch.bfloat16),
            "kv_a": (3072, 576, torch.bfloat16),
            "q_b_dsa": (1024, dsa_heads * 256, torch.bfloat16),
            "q_b_swa": (1024, swa_heads * 256, torch.bfloat16),
            "linear_gate_dsa": (3072, dsa_heads * 256, torch.bfloat16),
            "linear_gate_swa": (3072, swa_heads * 256, torch.bfloat16),
            "o_proj_dsa": (dsa_heads * 256, 3072, reduce_dtype),
            "o_proj_swa": (swa_heads * 256, 3072, reduce_dtype),
            "indexer_wq_b": (1024, 2048, torch.bfloat16),
            "indexer_wk": (3072, 128, torch.bfloat16),
            "indexer_weights": (3072, 16, torch.bfloat16),
            "engram_value": (1536, 3072, torch.bfloat16),
            "engram_key": (1536, 12288, torch.bfloat16),
            "router": (3072, 512, torch.float32),
            "latent_down": (3072, 1536, torch.bfloat16),
            "latent_up": (1536, 3072, torch.bfloat16),
            "dense_gate_up": (3072, dense_inter, torch.bfloat16),
            "dense_down": (dense_inter, 3072, reduce_dtype),
            "shared_gate_up": (3072, shared_inter, torch.bfloat16),
            "shared_down": (shared_inter, 3072, reduce_dtype),
        }
        return {
            name: (
                lambda width=width, outputs=outputs, dtype=dtype: p.compile_bf16_projection_aot(
                    width=width, outputs=outputs, output_dtype=dtype
                )
            )
            for name, (width, outputs, dtype) in shapes.items()
        }

    if group == "draft" and tp == 1:
        from b12x.integration.cuteafd import hy4_mla as mla
        from b12x.integration.cuteafd.context_split import compile_paged_staging_gather_aot

        draft = {
            "draft_gated_residual": lambda: p.compile_gated_residual_bf16_aot(width=3072),
            "draft_silu": lambda: p.compile_post_silu_split_aot(width=9216, limit=1e30),
            "draft_correction_silu": lambda: p.compile_post_silu_split_aot(width=3072, limit=1e30),
        }
        for kv in ("bf16", "fp8"):
            g = mla.Hy4MlaGeometry(kv_cache=kv)
            draft[f"draft_gather_{kv}"] = lambda g=g: compile_paged_staging_gather_aot(
                row_bytes=g.record_bytes, page_rows=1)
            draft[f"draft_attention_{kv}"] = lambda g=g: mla.compile_hy4_mla_attention_aot(
                g, max_rows=6 * 64, max_context=2048 + 6, mode="decode", page_size=1)
        for name, width, outputs in (
            ("gate_up", 3072, 9216), ("down", 9216, 3072),
            ("join", 6144, 3072), ("correction_down", 3072, 3072),
        ):
            draft[f"draft_proj_{name}"] = lambda width=width, outputs=outputs: p.compile_bf16_projection_aot(
                width=width, outputs=outputs)
        return draft
    if group == "draft":
        return {}  # TP2 shares the rank-0 TP1 drafter and sharded target head.

    raise ValueError(f"unknown Hy4-flash export group: {group}")


def programs(tp: int, decode_rows: int, max_rows: int, max_context: int):
    """Entries for the shared Programs dispatcher in export_b12x_dsv4_aot.py."""
    def capacity(name):
        if name.startswith("draft_"):
            return 6 * 64
        return decode_rows if "swa_decode" in name else max_rows

    return [
        (name, name, {"max_rows": capacity(name)}, build)
        for group in GROUPS
        for name, build in builders(
            tp, group, decode_rows=decode_rows, max_rows=max_rows, max_context=max_context
        ).items()
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tp", type=int, choices=(1, 2), default=1)
    parser.add_argument("--group", choices=GROUPS, default="ihc")
    parser.add_argument("--decode-rows", type=int, default=64)
    parser.add_argument("--max-rows", type=int, default=4096)
    parser.add_argument("--max-context", type=int, default=262144)
    args = parser.parse_args()

    from b12x._lib.aot_target import describe_target
    import torch

    describe_target("sm120")
    from b12x.integration.cuteafd import exportable_compilation, validate_exported_header

    destination = args.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    tag = "hy4f" if args.tp == 1 else "hy4f2"
    manifest = destination / f"{tag}_{args.group}.json"
    manifest.unlink(missing_ok=True)
    records = []
    for name, build in builders(
        args.tp, args.group, decode_rows=args.decode_rows, max_rows=args.max_rows, max_context=args.max_context
    ).items():
        label = f"{tag}_{name}"
        with exportable_compilation():
            program = build()
        program.export_to_c(str(destination), label, "cuteafd_" + label)
        validate_exported_header(program, destination / (label + ".h"), "cuteafd_" + label)
        artifacts = {
            label + ext: hashlib.sha256((destination / (label + ext)).read_bytes()).hexdigest()
            for ext in (".o", ".h")
        }
        records.append(
            dict(
                label=label,
                geometry=dict(program.geometry),
                abi=program.abi,
                scratch_bytes_per_row=program.scratch_bytes(1),
                artifacts=artifacts,
            )
        )
        print(f"exported {label}", flush=True)
    manifest.write_text(
        json.dumps(dict(schema=1, sparkinfer=_pinned_sparkinfer.LOCK_DATA, programs=records), indent=2) + "\n"
    )
    if torch.cuda.is_initialized():
        raise RuntimeError("offline Hy4 export initialized CUDA")


if __name__ == "__main__":
    main()

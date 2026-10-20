#!/usr/bin/env python3
"""Export native Step-5 pointwise and staged attention ABIs from the verified fork."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import _pinned_sparkinfer


def builders(heads, group, summary_capacity=131072, max_staged_regions=513):
    """Shared compiler definitions for diagnostics and the generic native table."""
    if heads not in (32, 64) or min(summary_capacity, max_staged_regions) <= 0:
        raise ValueError("Step-5 requires heads32/64 and positive summary capacities")
    groups = heads // 16
    if group in ("attention", "region-scores"):
        from b12x.integration.cuteafd import region_attention as a
        if group == "region-scores":
            return {"region_scores": lambda: a.compile_region_scores_aot(
                groups=groups, capacity=summary_capacity)}
        return {
            "region_compress": a.compile_region_compress_aot,
            "region_scores": lambda: a.compile_region_scores_aot(groups=groups, capacity=summary_capacity),
            "swa_attention": lambda: a.compile_selected_gqa_aot(
                query_heads=heads, kv_heads=groups, selection_width=512, selection_groups=1),
            "csa_attention": lambda: a.compile_selected_gqa_aot(query_heads=heads, kv_heads=groups),
        }
    if group == "metadata":
        from b12x.integration.cuteafd import cache_transactions as c, region_selection as s
        return {
            "sliding_positions": s.compile_sliding_positions_aot,
            "region_selection": lambda: s.compile_region_selection_aot(groups=groups, capacity=summary_capacity),
            "sliding_commit": lambda: c.compile_sliding_commit_aot(kv_heads=groups),
            "region_commit": lambda: c.compile_region_commit_aot(
                max_staged_regions=max_staged_regions, capacity=summary_capacity),
        }
    if group not in ("pointwise", "projections"):
        raise ValueError(f"unknown Step-5 export group: {group}")
    import torch
    from b12x.integration.cuteafd import pointwise as p
    if group == "projections":
        shapes = {
            "attn_q": (4096, heads * 192, torch.bfloat16),
            "attn_kv": (4096, groups * 192, torch.bfloat16),
            "attn_gate": (4096, heads, torch.bfloat16),
            "attn_out": (heads * 192, 4096, torch.bfloat16),
            "attn_out_partial": (heads * 192, 4096, torch.float32),
            "index_q": (4096, groups * 4 * 256, torch.bfloat16),
            "index_kz": (4096, 256, torch.bfloat16),
            "router": (4096, 352, torch.float32),
            "dense_gate_up": (4096, 2 * 13824, torch.bfloat16),
            "dense_down": (13824, 4096, torch.bfloat16),
            "shared_gate_up": (4096, 2 * 1536, torch.bfloat16),
            "shared_down": (1536, 4096, torch.bfloat16),
            "mtp_eh": (8192, 4096, torch.bfloat16),
        }
        return {name: (lambda width=width, outputs=outputs, dtype=dtype:
                p.compile_bf16_projection_aot(width=width, outputs=outputs, output_dtype=dtype))
                for name, (width, outputs, dtype) in shapes.items()}
    result = {
        "residual_bf16": lambda: p.compile_residual_add_aot(width=4096),
        "residual_f32": lambda: p.compile_residual_add_aot(width=4096, delta_dtype=torch.float32),
        "hidden_norm": lambda: p.compile_affine_norm_aot(width=4096, zero_centered=True, eps=None),
        "final_norm": lambda: p.compile_affine_norm_aot(width=4096, zero_centered=True,
            eps=None, input_dtype=torch.bfloat16),
        "index_q_norm": lambda: p.compile_affine_norm_aot(width=256, heads=groups * 4,
            eps=None, zero_centered=True, rotary=32,
            input_dtype=torch.bfloat16, parameter_dtype=torch.bfloat16),
        "index_k_norm": lambda: p.compile_affine_norm_aot(width=256, eps=None,
            subtract_mean=True, bias=True, rotary=32,
            input_dtype=torch.bfloat16, parameter_dtype=torch.bfloat16),
        "index_weight": lambda: p.compile_bf16_scaled_projection_aot(width=4096, outputs=groups * 4, prescale=0.5),
        "head_gate": lambda: p.compile_head_gate_aot(heads=heads, dim=192),
        "route_weights": lambda: p.compile_route_weights_aot(experts=352, topk=8),
    }
    for span in (32, 64, 192):
        for scaling in (False, True):
            def table(span=span, scaling=scaling):
                from b12x.integration.cuteafd.rope import compile_rope_tables_aot
                return compile_rope_tables_aot(rotary_span=span, llama3=scaling)
            result[f"rope_s{span}_{'llama3' if scaling else 'plain'}"] = table
    for kind, rotary in (("swa", 192), ("csa", 64)):
        for name, count in (("q", heads), ("k", groups)):
            result[f"{kind}_{name}_norm_rope"] = lambda count=count, rotary=rotary: p.compile_affine_norm_aot(
                width=192, heads=count, rotary=rotary, zero_centered=True, eps=None, input_dtype=torch.bfloat16)
    for width in (1536, 13824):
        for limit in (7.0, 10.0):
            result[f"post_silu_w{width}_c{int(limit)}"] = lambda width=width, limit=limit: p.compile_post_silu_aot(
                width=width, limit=limit)
    return result


def programs(heads, max_rows, max_context):
    """Entries for the existing Programs dispatcher; no separate native ABI."""
    capacity = (max_context + 7) // 8
    maximum = (max_rows + 14) // 8
    return [(name, name, {"max_rows": max_rows}, build)
            for group in ("pointwise", "projections", "attention", "metadata")
            for name, build in builders(heads, group, capacity, maximum).items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--heads", type=int, choices=(32, 64), default=64)
    parser.add_argument("--group", choices=("pointwise", "projections", "attention", "metadata", "region-scores"), default="pointwise")
    parser.add_argument("--summary-capacity", type=int, default=131072)
    parser.add_argument("--max-staged-regions", type=int, default=513)
    args = parser.parse_args()
    if min(args.summary_capacity, args.max_staged_regions) <= 0:
        parser.error("summary and staged-region capacities must be positive")
    from b12x.integration.cuteafd import exportable_compilation, validate_exported_header

    destination = args.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / f"step5_h{args.heads}_{args.group}.json"
    manifest.unlink(missing_ok=True)
    records = []
    for name, build in builders(args.heads, args.group, args.summary_capacity, args.max_staged_regions).items():
        label = f"step5_h{args.heads}_{name}"
        with exportable_compilation():
            program = build()
        program.export_to_c(str(destination), label, "cuteafd_" + label)
        validate_exported_header(program, destination / (label + ".h"), "cuteafd_" + label)
        artifacts = {label + ext: hashlib.sha256((destination / (label + ext)).read_bytes()).hexdigest()
                     for ext in (".o", ".h")}
        records.append(dict(label=label, geometry=dict(program.geometry), abi=program.abi,
                            scratch_bytes_per_row=program.scratch_bytes(1), artifacts=artifacts))
        print(f"exported {label}", flush=True)
    manifest.write_text(json.dumps(dict(schema=1, sparkinfer=_pinned_sparkinfer.LOCK_DATA,
                                        programs=records), indent=2) + "\n")


if __name__ == "__main__":
    main()

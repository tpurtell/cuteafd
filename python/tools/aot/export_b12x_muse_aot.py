#!/usr/bin/env python3
"""Export Muse Spark coordinator programs through the shared native dispatcher."""
from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import _pinned_sparkinfer


EXPERT_WIDTH = 4096
WINDOW = 2048
# Ring rows beyond the window: the widest rollback step (decode/verify). The
# cache (`cache.rs` swa_ring) uses the same formula; a Rust test pins it.
SWA_RING_STEP = 64


def compile_expert_input_quant_aot():
    """Expert-width BF16 rows -> the shared FP8 K32 Spark wire (E 4096, row 4224 bytes)."""
    from b12x.integration.cuteafd._common import AotProgram, Operand, Scalar
    from b12x._lib.quant.mxfp8_rows import compile_mxfp8_rows_quant_aot
    import torch

    h = EXPERT_WIDTH
    compiled = compile_mxfp8_rows_quant_aot(size_k=h, expected_m=80, amax_floor=1e-4, wire_rows=True)
    return AotProgram(
        name="muse_expert_input_quant", compiled=compiled,
        operands=(Operand("source_ptr", torch.bfloat16, f"[rows,{h}]"),
                  Operand("values_ptr", torch.uint32, f"[rows,{(h + h // 32) // 4}] (u8 [rows,{h + h // 32}])", "out"),
                  Operand("scale_rows_ptr", torch.uint8, "values_ptr + H", "out"),
                  Operand("scale_mma_ptr", torch.uint8, "unused", "in")),
        scalars=(Scalar("m"), Scalar("grid_x")),
        geometry={"hidden": h, "row_bytes": h + h // 32, "expected_m": 80, "amax_floor": 1e-4,
                  "exact_row_cap": 2**31 - 1},
        doc=compile_expert_input_quant_aot.__doc__,
    )


def builders(heads=128, group="pointwise", max_rows=4096):
    if heads not in (64, 128) or not 1 <= max_rows <= 4096:
        raise ValueError("Muse requires heads64/128 and max_rows in 1..4096")
    from b12x.integration.cuteafd import muse as m
    if group == "pointwise":
        result = {op: partial(m.compile_norm_aot, op=op) for op in m.NORMS}
        result.update(residual_gate=m.compile_residual_gate_aot,
                      route_select=m.compile_route_select_aot,
                      softcap=m.compile_softcap_aot,
                      qk_norm_rope_swa=partial(m.compile_qk_norm_rope_aot, heads=heads, kind="swa"),
                      qk_norm_nope_full=partial(m.compile_qk_norm_rope_aot, heads=heads, kind="full"),
                      attention_output=partial(m.compile_attention_output_aot, heads=heads),
                      expert_input_quant=compile_expert_input_quant_aot)
        return result
    if group == "tp-combine":
        from b12x.integration.cuteafd.muse_tp import compile_tp_combine_aot
        return {f"tp_combine_{format}": partial(compile_tp_combine_aot, format=format)
                for format in ("bf16", "nvfp4", "fp8")}
    if group == "projections":
        from b12x.integration.cuteafd.pointwise import compile_bf16_projection_aot
        return {"router": m.compile_router_aot,
                "gate_proj": partial(compile_bf16_projection_aot,
                                     width=8192, outputs=heads * 64)}
    if group == "attention":
        from b12x.integration.cuteafd import _muse_attention as a
        ring_rows = ((WINDOW + SWA_RING_STEP + 64 - 1) // 64) * 64
        result = {f"attention_{kind}_{route}": partial(a.compile_attention_aot,
                      heads=heads, kind=kind, route=route, max_rows=max_rows, ring_rows=ring_rows)
                  for kind in ("full", "swa") for route in ("prefill", "decode")}
        result["swa_commit"] = partial(a.compile_swa_commit_aot, heads=heads,
                                      max_rows=max_rows, ring_rows=ring_rows)
        return result
    raise ValueError(f"unknown Muse export group: {group}")


def programs(heads=128, max_rows=4096):
    return [(name, name, {"max_rows": max_rows}, build)
            for group in ("pointwise", "projections", "attention", "tp-combine")
            for name, build in builders(heads, group, max_rows).items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--heads", type=int, choices=(64, 128), default=128)
    parser.add_argument("--group", choices=("pointwise", "projections", "attention", "tp-combine"), default="pointwise")
    args = parser.parse_args()
    from export_b12x_dsv4_aot import main as export
    names = ",".join(builders(args.heads, args.group))
    sys.argv = [sys.argv[0], "--geometry", "muse" if args.heads == 128 else "muse2",
                "--only", names, "--output-dir", str(args.output_dir)]
    export()


if __name__ == "__main__":
    main()

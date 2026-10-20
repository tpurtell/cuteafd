#!/usr/bin/env python3
"""Kimi K3 native K6 core projections (shared trellis ABI v1, opt-in).

Dimensions are local [K,N] after TP2 placement, not the full checkpoint sizes.
Whole/replicated MLA and latent operands retain their full dimensions. kv_a
requires a stored N640 tensor and a caller-side N576 trim; unpadded N576 is
not a supported native trellis geometry. Independently calibrated projections
are never concatenated. One fixed arithmetic covers rows 1 through 4096.
"""
from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import _pinned_sparkinfer

# (name, K, N, shard side): side 1 = output units, side 2 = input units.
# Whole/replicated tensors use output units, as their EXL3 payload is unsliced.
TRELLIS_SHAPES = (
    ("kda_qkvg_tp2", 7168, 6144, 1),
    ("kda_o_tp2", 6144, 7168, 2),
    ("shared_gate_up_tp2", 7168, 3072, 1),
    ("shared_down_tp2", 3072, 7168, 2),
    ("latent_down", 7168, 3584, 1),
    ("latent_up", 3584, 7168, 1),
    ("mla_q_a", 7168, 1536, 1),
    ("mla_kv_a_padded", 7168, 640, 1),
    ("mla_q_b_owned", 1536, 18432, 1),
    ("mla_q_b_tp2", 1536, 9216, 1),
    ("mla_g_owned", 7168, 12288, 1),
    ("mla_o_owned", 12288, 7168, 1),
    ("lm_head", 7168, 163840, 1),
)


CHUNK_WEIGHT_LIMIT_BYTES = 256 * 1024 * 1024


def chunk_columns(k, n):
    columns = min(n, CHUNK_WEIGHT_LIMIT_BYTES // (2 * k) // 128 * 128)
    while columns and n % columns:
        columns -= 128
    if not columns:
        raise ValueError("no H128 chunk fits the weight scratch limit")
    return columns


def row_policies():
    result = []
    for name, k, n, side in TRELLIS_SHAPES:
        columns = chunk_columns(k, n)
        for book in ("mcg", "mul1"):
            result.append(dict(shape=name, input_features=k, output_features=n,
                sharded_side=side, codebook=book, threshold_rows=64,
                below="shared_native", at_or_above="chunk_dequant_bf16",
                native_program=f"trellis_{name}_{book}",
                native_input=f"trellis_input_k{k}_bf16",
                wide_input=f"trellis_input_k{k}_bf16_rotated_bf16",
                dequant_program=f"trellis_dequant_{name}_{book}",
                projection_program=f"trellis_chunk_bf16_{name}_f32",
                output_program=f"trellis_output_n{n}_{'f32' if n == 163840 else 'bf16'}",
                chunk_columns=columns, weight_scratch_bytes=columns*k*2,
                chunk_output_dtype="f32", chunk_output_scratch_bytes_per_row=columns*4,
                stitch_before_output_transform=True))
    return result


def programs():
    return programs_for_shapes(TRELLIS_SHAPES)


def programs_for_shapes(shapes, *, max_rows=4096, include_chunk=True, f32_outputs=(163840,)):
    """Explicit-shape hook for small native fixture packages; production uses the table."""
    from b12x.integration.cuteafd.trellis_linear import compile_trellis_linear_aot
    from b12x.integration.cuteafd.trellis_transform import (
        compile_trellis_input_aot, compile_trellis_output_aot,
    )
    import torch
    from b12x.integration.cuteafd.trellis_dequant import compile_trellis_dequant_aot
    from b12x.integration.cuteafd.pointwise import compile_bf16_projection_aot
    result = []
    for name, k, n, side in shapes:
        columns = chunk_columns(k, n)
        if include_chunk:
            projection = dict(width=k, outputs=columns, output_dtype="f32")
            result.append((f"trellis_chunk_bf16_{name}_f32", "bf16_projection", projection,
                           partial(compile_bf16_projection_aot, width=k, outputs=columns, output_dtype=torch.float32)))
        for codebook in ("mcg", "mul1"):
            params = dict(input_features=k, output_features=n, bits=6,
                          codebook=codebook, sharded_side=side,
                          rotation_count=1, max_rows=max_rows)
            result.append((f"trellis_{name}_{codebook}", "trellis_linear", params,
                           partial(compile_trellis_linear_aot, **params)))
            if include_chunk:
                dequant = dict(input_features=k, output_features=n, chunk_columns=columns,
                               codebook=codebook, sharded_side=side)
                result.append((f"trellis_dequant_{name}_{codebook}", "trellis_dequant", dequant,
                               partial(compile_trellis_dequant_aot, **dequant)))
    for k in sorted({k for _, k, _, _ in shapes}):
        params = dict(features=k, input_dtype="bf16", max_rows=max_rows)
        result.append((f"trellis_input_k{k}_bf16", "trellis_input", params,
                       partial(compile_trellis_input_aot, **params)))
        if include_chunk:
            wide_params = dict(params, output_dtype="bf16")
            result.append((f"trellis_input_k{k}_bf16_rotated_bf16", "trellis_input", wide_params,
                           partial(compile_trellis_input_aot, **wide_params)))
    for n in sorted({n for _, _, n, _ in shapes}):
        dtype = "f32" if n in f32_outputs else "bf16"
        params = dict(features=n, output_dtype=dtype, max_rows=max_rows)
        result.append((f"trellis_output_n{n}_{dtype}", "trellis_output", params,
                       partial(compile_trellis_output_aot, **params)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--only", help="comma-separated program names")
    args = parser.parse_args()
    from export_b12x_dsv4_aot import main as export
    sys.argv = [sys.argv[0], "--geometry", "kimi_k6_dense", "--output-dir", str(args.output_dir)]
    if args.only:
        sys.argv += ["--only", args.only]
    export()


if __name__ == "__main__":
    main()

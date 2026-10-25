#!/usr/bin/env python3
"""Shared native dense operands used by EXLR8 checkpoints (opt-in table)."""
from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import _pinned_sparkinfer

# (name, K, N, sharded side): the GLM 5.3 checkpoint's unique trellis shapes.
TRELLIS_SHAPES = (
    ("q_b", 2048, 16384, 1),
    ("o", 16384, 6144, 2),
    ("gate_up", 6144, 12288, 1),
    ("down", 12288, 6144, 2),
    ("q_b_tp2", 2048, 8192, 1),
    ("o_tp2", 8192, 6144, 2),
    ("gate_up_tp2", 6144, 6144, 1),
    ("down_tp2", 6144, 6144, 2),
)


SCALED_SHAPES = (
    ("mxfp8_q_a", 6144, 2048, "mxfp8", 1),
    ("mxfp8_kv_a", 6144, 576, "mxfp8", 1),
    ("mxfp8_index_q", 2048, 4096, "mxfp8", 1),
    ("mxfp8_index_k", 6144, 128, "mxfp8", 1),
    ("int4_q_b", 2048, 16384, "int4_g32", 1),
    ("int4_o", 16384, 6144, "int4_g32", 2),
    ("int4_q_b_tp2", 2048, 8192, "int4_g32", 1),
    ("int4_o_tp2", 8192, 6144, "int4_g32", 2),
    ("fp8_row_lm_head", 6144, 154880, "fp8_row", 1),
)


def programs():
    from b12x.integration.cuteafd.trellis_linear import compile_trellis_linear_aot
    from b12x.integration.cuteafd.scaled_linear import compile_scaled_linear_aot
    from b12x.integration.cuteafd.fp8_row_embedding import compile_fp8_row_embedding_aot
    from b12x.integration.cuteafd.trellis_unpack import compile_trellis_unpack_e4m3_aot
    from b12x.integration.cuteafd.trellis_transform import (
        compile_trellis_input_aot, compile_trellis_output_aot)
    from b12x.integration.cuteafd.native_dense import (
        compile_trellis_fp8_prefill_aot, compile_swiglu_split_aot,
        compile_f32_narrow_pack_aot)
    from b12x.integration.cuteafd.pointwise import compile_bf16_projection_aot
    import torch
    result = []
    for name, k, n, side in TRELLIS_SHAPES:
        params = dict(input_features=k, output_features=n, bits=4,
                      codebook="e4m3", sharded_side=side, rotation_count=4,
                      max_rows=4096)
        result.append((f"trellis_{name}", "trellis_linear", params,
                       partial(compile_trellis_linear_aot, **params)))
    for name, k, n, side in TRELLIS_SHAPES:
        params = dict(input_features=k, output_features=n, bits=4,
                      codebook="e4m3", sharded_side=side, rotation_count=4,
                      max_rows=4096, input_dtype="bf16")
        result.append((f"trellis_{name}_bf16", "trellis_linear", params,
                       partial(compile_trellis_linear_aot, **params)))
    for name, k, n, encoding, side in SCALED_SHAPES:
        params = dict(input_features=k, output_features=n, encoding=encoding,
                      sharded_side=side, max_rows=4096)
        result.append((name, "scaled_linear", params,
                       partial(compile_scaled_linear_aot, **params)))
    params = dict(hidden=6144, vocabulary=154880, max_rows=4096)
    result.append(("fp8_row_embedding", "fp8_row_embedding", params,
                   partial(compile_fp8_row_embedding_aot, **params)))
    for name, k, n, side in TRELLIS_SHAPES:
        params = dict(input_features=k, output_features=n, sharded_side=side)
        result.append((f"unpack_{name}", "trellis_unpack_e4m3", params,
                       partial(compile_trellis_unpack_e4m3_aot, **params)))
        params = dict(input_features=k, output_features=n, max_rows=4096)
        result.append((f"prefill_{name}", "trellis_fp8_prefill", params,
                       partial(compile_trellis_fp8_prefill_aot, **params)))
    for k in (2048, 6144, 8192, 12288, 16384):
        for dtype in ("f16", "bf16"):
            params = dict(features=k, max_rows=4096, output_dtype=dtype)
            result.append((f"trellis_input_k{k}_{dtype}", "trellis_input", params,
                           partial(compile_trellis_input_aot, **params)))
    for n in (6144, 8192, 12288, 16384):
        params = dict(features=n, max_rows=4096)
        result.append((f"trellis_output_n{n}", "trellis_output", params,
                       partial(compile_trellis_output_aot, **params)))
    for inter in (12288, 6144):
        params = dict(inter=inter)
        result.append((f"swiglu_split_i{inter}", "swiglu_split", params,
                       partial(compile_swiglu_split_aot, **params)))
    for width, pitch in ((2048, 2624), (576, 2624), (128, 160), (32, 160)):
        params = dict(width=width, pitch=pitch)
        result.append((f"f32_narrow_pack_n{width}_p{pitch}", "f32_narrow_pack", params,
                       partial(compile_f32_narrow_pack_aot, **params)))
    # This small indexer operand is BF16 in the checkpoint, not a conversion.
    params = dict(width=6144, outputs=32, output_dtype=torch.float32)
    result.append(("bf16_index_weights", "bf16_projection",
                   dict(width=6144, outputs=32, output_dtype="torch.float32"),
                   partial(compile_bf16_projection_aot, **params)))
    from b12x.integration.cuteafd.dense_mxfp8 import (
        compile_dense_mxfp8_quant_aot, compile_dense_mxfp8_aot)
    for name, k, n, side in TRELLIS_SHAPES:
        params = dict(input_features=k, output_features=n, encoding="trellis",
                      sharded_side=side, input_dtype="f16", rotation_count=4,
                      max_rows=4096)
        result.append((f"trellis_{name}_a8", "dense_mxfp8", params,
                       partial(compile_dense_mxfp8_aot, **params)))
    for k in sorted({shape[1] for shape in TRELLIS_SHAPES}):
        params = dict(input_features=k, input_dtype="f16", rotation_count=4,
                      max_rows=4096)
        result.append((f"quant_k{k}_a8", "dense_mxfp8_quant", params,
                       partial(compile_dense_mxfp8_quant_aot, **params)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    from export_b12x_dsv4_aot import main as export
    sys.argv = [sys.argv[0], "--geometry", "exlr8_dense",
                "--output-dir", str(args.output_dir)]
    export()


if __name__ == "__main__":
    main()

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


# Geometry requests from native checkpoint headers, including TP2 execution slices.
# A full KDA o and a TP2 MLA o share (8192, 4096, 2).
NATIVE_TRELLIS_GEOMETRIES = (
    (1536, 16384, 1), (4096, 12288, 1), (8192, 4096, 2),
    (12288, 4096, 2), (16384, 4096, 2),
    (1536, 8192, 1), (4096, 6144, 1), (4096, 4096, 2), (6144, 4096, 2),
)
NATIVE_MXFP8_GEOMETRIES = (
    (128, 8192, 1), (1536, 4096, 1), (4096, 64, 1),
    (4096, 128, 1), (4096, 512, 1), (4096, 1536, 1), (4096, 8192, 1),
    (128, 4096, 1), (4096, 32, 1), (4096, 4096, 1),
)


def additional_programs():
    from b12x.integration.cuteafd.trellis_linear import compile_trellis_linear_aot
    from b12x.integration.cuteafd.scaled_linear import compile_scaled_linear_aot
    from b12x.integration.cuteafd.fp8_row_embedding import compile_fp8_row_embedding_aot
    from b12x.integration.cuteafd.trellis_transform import (
        compile_trellis_input_aot, compile_trellis_output_aot)
    from b12x.integration.cuteafd.native_dense import (
        compile_f32_narrow_pack_aot)
    from b12x.integration.cuteafd.pointwise import compile_bf16_projection_aot
    import torch

    result = []
    legacy_trellis = {shape[1:] for shape in TRELLIS_SHAPES}
    shapes = sorted(set(NATIVE_TRELLIS_GEOMETRIES) - legacy_trellis)
    for k, n, side in shapes:
        stem = f"k{k}_n{n}_s{side}"
        for dtype in ("f16", "bf16"):
            params = dict(input_features=k, output_features=n, bits=4,
                          codebook="e4m3", sharded_side=side, rotation_count=4,
                          max_rows=4096)
            if dtype == "bf16":
                params["input_dtype"] = dtype
            suffix = "_bf16" if dtype == "bf16" else ""
            result.append((f"trellis_{stem}{suffix}", "trellis_linear", params,
                           partial(compile_trellis_linear_aot, **params)))
    legacy_mxfp8 = {(k, n, side) for _, k, n, encoding, side in SCALED_SHAPES
                    if encoding == "mxfp8"}
    for k, n, side in sorted(set(NATIVE_MXFP8_GEOMETRIES) - legacy_mxfp8):
        params = dict(input_features=k, output_features=n, encoding="mxfp8",
                      sharded_side=side, max_rows=4096)
        result.append((f"mxfp8_k{k}_n{n}_s{side}", "scaled_linear", params,
                       partial(compile_scaled_linear_aot, **params)))
    params = dict(input_features=4096, output_features=154880, encoding="fp8_row",
                  sharded_side=1, max_rows=4096)
    result.append(("fp8_row_k4096_n154880_s1", "scaled_linear", params,
                   partial(compile_scaled_linear_aot, **params)))
    params = dict(hidden=4096, vocabulary=154880, max_rows=4096)
    result.append(("fp8_row_embedding_k4096_n154880", "fp8_row_embedding", params,
                   partial(compile_fp8_row_embedding_aot, **params)))
    for k in sorted({k for k, _, _ in shapes} - {shape[1] for shape in TRELLIS_SHAPES}):
        for dtype in ("f16", "bf16"):
            params = dict(features=k, max_rows=4096, output_dtype=dtype)
            result.append((f"trellis_input_k{k}_{dtype}", "trellis_input", params,
                           partial(compile_trellis_input_aot, **params)))
    for n in sorted({n for _, n, _ in shapes} - {shape[2] for shape in TRELLIS_SHAPES}):
        params = dict(features=n, max_rows=4096)
        result.append((f"trellis_output_n{n}", "trellis_output", params,
                       partial(compile_trellis_output_aot, **params)))
    for width, pitch in ((1536, 2048), (512, 2048), (128, 288), (32, 288)):
        params = dict(width=width, pitch=pitch)
        result.append((f"f32_narrow_pack_n{width}_p{pitch}", "f32_narrow_pack", params,
                       partial(compile_f32_narrow_pack_aot, **params)))
    for outputs in (32, 128, 288):
        params = dict(width=4096, outputs=outputs, output_dtype=torch.float32)
        result.append((f"bf16_projection_k4096_n{outputs}", "bf16_projection",
                       dict(width=4096, outputs=outputs, output_dtype="torch.float32"),
                       partial(compile_bf16_projection_aot, **params)))
    return result


def legacy_programs():
    from b12x.integration.cuteafd.trellis_linear import compile_trellis_linear_aot
    from b12x.integration.cuteafd.scaled_linear import compile_scaled_linear_aot
    from b12x.integration.cuteafd.fp8_row_embedding import compile_fp8_row_embedding_aot
    from b12x.integration.cuteafd.trellis_transform import (
        compile_trellis_input_aot, compile_trellis_output_aot)
    from b12x.integration.cuteafd.native_dense import (
        compile_swiglu_split_aot,
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


def programs(families=("glm",)):
    requested = set(families)
    if not requested or requested - {"glm", "glmf"}:
        raise ValueError("EXLR8 dense families must contain glm and/or glmf")
    legacy = legacy_programs()
    if "glmf" not in requested:
        return legacy
    if "glm" in requested:
        return legacy + additional_programs()
    # Reuse legacy companions by geometry; never emit aliases for another family.
    inputs = {k for k, _, _ in NATIVE_TRELLIS_GEOMETRIES}
    outputs = {n for _, n, _ in NATIVE_TRELLIS_GEOMETRIES}
    def needed(entry):
        _, op, params, _ = entry
        if op == "trellis_input":
            return params["features"] in inputs
        if op == "trellis_output":
            return params["features"] in outputs
        if op == "swiglu_split":
            return params["inter"] in outputs
        if op == "f32_narrow_pack":
            return (params["width"], params["pitch"]) in ((128, 160), (32, 160))
        if op == "trellis_linear":
            side = params.get("sharded_side")
            return any((params["input_features"], params["output_features"]) == (k, n)
                       and (side is None or side == s) for k, n, s in NATIVE_TRELLIS_GEOMETRIES)
        if op == "scaled_linear" and params["encoding"] == "mxfp8":
            return (params["input_features"], params["output_features"],
                    params["sharded_side"]) in NATIVE_MXFP8_GEOMETRIES
        return False
    return [entry for entry in legacy if needed(entry)] + additional_programs()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dense-trellis-families", default="glm")
    args = parser.parse_args()
    from export_b12x_dsv4_aot import main as export
    sys.argv = [sys.argv[0], "--geometry", "exlr8_dense",
                "--dense-trellis-families", args.dense_trellis_families,
                "--output-dir", str(args.output_dir)]
    export()


if __name__ == "__main__":
    main()

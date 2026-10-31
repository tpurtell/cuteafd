#!/usr/bin/env python3
"""Shared native K6 core projections (trellis ABI v1).

Dimensions are local [K,N] after TP2 placement, not the full checkpoint sizes.
Whole/replicated MLA and latent operands retain their full dimensions. GLM
and Kimi kv_a use stored N640 with a caller-side N576 trim; Flash is NoPE
and uses native N512 without padding. Unpadded N576 is unsupported.
Independently calibrated projections are never concatenated. One fixed
arithmetic covers rows 1 through 4096.

KDA q/k/v/g share one activation and run as one grouped launch per H128
stage. Each projection retains its own payload, scales and rotation. Native
prefill decodes exact FP16 weights in shared memory, never into global scratch.
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
KIMI_SHAPES = (
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


# GLM 5.3 architecture, with whole and TP2 attention/FFN ownership. Builds do
# not read checkpoints; runtime headers must match one of these exact shapes.
GLM_SHAPES = (
    ("glm_q_a", 6144, 2048, 1),
    ("glm_kv_a_padded", 6144, 640, 1),
    ("glm_q_b", 2048, 16384, 1),
    ("glm_q_b_tp2", 2048, 8192, 1),
    ("glm_o", 16384, 6144, 1),
    ("glm_o_tp2", 8192, 6144, 2),
    ("glm_dense_gate_up", 6144, 12288, 1),
    ("glm_dense_gate_up_tp2", 6144, 6144, 1),
    ("glm_dense_down", 12288, 6144, 1),
    ("glm_dense_down_tp2", 6144, 6144, 2),
    ("glm_shared_gate_up", 6144, 2048, 1),
    ("glm_shared_gate_up_tp2", 6144, 1024, 1),
    ("glm_shared_down", 2048, 6144, 1),
    ("glm_shared_down_tp2", 1024, 6144, 2),
    ("glm_lm_head", 6144, 154880, 1),
)
# GLM Flash is NoPE: kv_a is N512, not the padded N640 GLM/Kimi operand.
# Gate/up and q/k/v remain independently calibrated projections.
GLMF_SHAPES = (
    ("glmf_kda_qkv", 4096, 8192, 1),
    ("glmf_kda_qkv_tp2", 4096, 4096, 1),
    ("glmf_kda_o", 8192, 4096, 1),
    ("glmf_kda_o_tp2", 4096, 4096, 2),
    ("glmf_q_a", 4096, 1536, 1),
    ("glmf_kv_a", 4096, 512, 1),
    ("glmf_q_b", 1536, 16384, 1),
    ("glmf_q_b_tp2", 1536, 8192, 1),
    ("glmf_o", 16384, 4096, 1),
    ("glmf_o_tp2", 8192, 4096, 2),
    ("glmf_dense_gate_up", 4096, 12288, 1),
    ("glmf_dense_gate_up_tp2", 4096, 6144, 1),
    ("glmf_dense_down", 12288, 4096, 1),
    ("glmf_dense_down_tp2", 6144, 4096, 2),
    ("glmf_shared_gate_up", 4096, 2048, 1),
    ("glmf_shared_gate_up_tp2", 4096, 1024, 1),
    ("glmf_shared_down", 2048, 4096, 1),
    ("glmf_shared_down_tp2", 1024, 4096, 2),
    ("glmf_lm_head", 4096, 154880, 1),
)
# Mistral Large 4: MLA without q-LoRA (wq straight from the hidden state),
# kv_a N576 padded to N640 on RoPE layers and native N512 on NoPE layers,
# 64 heads x 128 (v 128), dense layer 0 (I 24576), one shared expert (I 6144).
ML4_SHAPES = (
    ("ml4_q", 6144, 8192, 1),
    # Also the indexer's wq_b [4096, 6144] when the recipe moves it into the core.
    ("ml4_q_tp2", 6144, 4096, 1),
    ("ml4_kv_a_padded", 6144, 640, 1),
    ("ml4_kv_a_nope", 6144, 512, 1),
    ("ml4_o", 8192, 6144, 1),
    ("ml4_o_tp2", 4096, 6144, 2),
    ("ml4_dense_gate_up", 6144, 24576, 1),
    ("ml4_dense_gate_up_tp2", 6144, 12288, 1),
    ("ml4_dense_down", 24576, 6144, 1),
    ("ml4_dense_down_tp2", 12288, 6144, 2),
    ("ml4_shared_gate_up", 6144, 6144, 1),
    ("ml4_shared_gate_up_tp2", 6144, 3072, 1),
    ("ml4_shared_down", 6144, 6144, 1),
    ("ml4_shared_down_tp2", 3072, 6144, 2),
    ("ml4_lm_head", 6144, 204800, 1),
)
FAMILY_SHAPES = {"kimik3": KIMI_SHAPES, "glm": GLM_SHAPES, "glmf": GLMF_SHAPES, "ml4": ML4_SHAPES}
F32_OUTPUTS = (163840, 154880, 204800)


def shapes_for(families):
    """Canonicalize equal geometries so runtime lookup is never ambiguous."""
    result, seen = [], set()
    for family in families:
        for item in FAMILY_SHAPES[family]:
            if item[1:] not in seen:
                result.append(item)
                seen.add(item[1:])
    return tuple(result)


# Projections sharing one activation and geometry, launched as one group of four.
GROUPED_SHAPES = {"kda_qkvg_tp2": 4}
GROUPED_MAX_ROWS = 4096


def row_policies(families=("kimik3",)):
    result = []
    for name, k, n, side in shapes_for(families):
        for book in ("mcg", "mul1"):
            result.append(dict(shape=name, input_features=k, output_features=n,
                sharded_side=side, codebook=book,
                native_program=f"trellis_{name}_{book}",
                native_input=f"trellis_input_k{k}_bf16",
                output_program=f"trellis_output_n{n}_{'f32' if n in F32_OUTPUTS else 'bf16'}",
                **({"grouped": grouped_policy(name, k, n, book)} if name in GROUPED_SHAPES else {})))
    return result


def grouped_policy(name, k, n, book):
    groups = GROUPED_SHAPES[name]
    return dict(groups=groups, max_rows=GROUPED_MAX_ROWS,
                native_program=f"trellis_{name}_{book}_g{groups}",
                native_input=f"trellis_input_k{k}_bf16_g{groups}",
                output_program=f"trellis_output_n{n}_bf16_g{groups}")


def programs(*, families=("kimik3",), input_dtype="f16"):
    # Keep all Kimi compile calls in their original order. Other families append
    # only new geometry/transform entries, including in a multi-family build.
    result, seen_names, seen_shapes = [], set(), set()
    for family in families:
        shapes = tuple(shape for shape in shapes_for((family,)) if shape[1:] not in seen_shapes)
        seen_shapes.update(shape[1:] for shape in shapes)
        for item in programs_for_shapes(shapes, input_dtype=input_dtype):
            if item[0] not in seen_names:
                result.append(item)
                seen_names.add(item[0])
    return result


def programs_for_shapes(shapes, *, max_rows=4096, f32_outputs=F32_OUTPUTS, input_dtype="f16"):
    """Explicit-shape hook for small native fixture packages; production uses the table."""
    from b12x.integration.cuteafd.trellis_linear import compile_trellis_linear_aot
    from b12x.integration.cuteafd.trellis_transform import (
        compile_trellis_input_aot, compile_trellis_output_aot,
    )
    if input_dtype not in ("f16", "bf16"):
        raise ValueError("trellis input dtype must be f16 or bf16")
    core_dtype = {"input_dtype": "bf16"} if input_dtype == "bf16" else {}
    ingress_dtype = {"output_dtype": "bf16"} if input_dtype == "bf16" else {}
    result = []
    for name, k, n, side in shapes:
        for codebook in ("mcg", "mul1"):
            params = dict(input_features=k, output_features=n, bits=6,
                          codebook=codebook, sharded_side=side,
                          rotation_count=1, max_rows=max_rows, **core_dtype)
            result.append((f"trellis_{name}_{codebook}", "trellis_linear", params,
                           partial(compile_trellis_linear_aot, **params)))
    for k in sorted({k for _, k, _, _ in shapes}):
        params = dict(features=k, input_dtype="bf16", max_rows=max_rows, **ingress_dtype)
        result.append((f"trellis_input_k{k}_bf16", "trellis_input", params,
                       partial(compile_trellis_input_aot, **params)))
    for n in sorted({n for _, _, n, _ in shapes}):
        dtype = "f32" if n in f32_outputs else "bf16"
        params = dict(features=n, output_dtype=dtype, max_rows=max_rows)
        result.append((f"trellis_output_n{n}_{dtype}", "trellis_output", params,
                       partial(compile_trellis_output_aot, **params)))
    for name, k, n, side in shapes:
        groups = GROUPED_SHAPES.get(name)
        if not groups:
            continue
        rows = min(max_rows, GROUPED_MAX_ROWS)
        for codebook in ("mcg", "mul1"):
            params = dict(input_features=k, output_features=n, bits=6, codebook=codebook,
                          sharded_side=side, rotation_count=groups, max_rows=rows, groups=groups, **core_dtype)
            result.append((f"trellis_{name}_{codebook}_g{groups}", "trellis_linear", params,
                           partial(compile_trellis_linear_aot, **params)))
        params = dict(features=k, input_dtype="bf16", max_rows=rows, groups=groups, **ingress_dtype)
        result.append((f"trellis_input_k{k}_bf16_g{groups}", "trellis_input", params,
                       partial(compile_trellis_input_aot, **params)))
        params = dict(features=n, output_dtype="bf16", max_rows=rows, groups=groups)
        result.append((f"trellis_output_n{n}_bf16_g{groups}", "trellis_output", params,
                       partial(compile_trellis_output_aot, **params)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--only", help="comma-separated program names")
    args = parser.parse_args()
    from export_b12x_dsv4_aot import main as export
    sys.argv = [sys.argv[0], "--geometry", "dense_trellis_k6", "--output-dir", str(args.output_dir)]
    if args.only:
        sys.argv += ["--only", args.only]
    export()


if __name__ == "__main__":
    main()

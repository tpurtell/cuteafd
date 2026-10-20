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


def programs():
    from b12x.integration.cuteafd.trellis_linear import compile_trellis_linear_aot
    result = []
    for name, k, n, side in TRELLIS_SHAPES:
        for codebook in ("mcg", "mul1"):
            params = dict(input_features=k, output_features=n, bits=6,
                          codebook=codebook, sharded_side=side,
                          rotation_count=1, max_rows=4096)
            result.append((f"trellis_{name}_{codebook}", "trellis_linear", params,
                           partial(compile_trellis_linear_aot, **params)))
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

#!/usr/bin/env python3
"""Export official V4.1 expert kernels and planner-owned native scratch layouts."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1] / _d) for _d in ("lib",)]  # sibling tool dirs

import argparse
import hashlib
from pathlib import Path

import _pinned_sparkinfer


def export_input_quantizer(output_dir: Path, manifest: dict) -> None:
    from b12x._lib.quant.mxfp8_rows import (
        compile_mxfp8_rows_quant_aot, mxfp8_rows_quant_aot_grid,
    )
    from export_b12x_v41_fp8_aot import validate_abi
    label = "v41_expert_input_quant"
    compiled = compile_mxfp8_rows_quant_aot(
        size_k=5120, expected_m=80, amax_floor=1e-4, wire_rows=True,
    )
    compiled.export_to_c(str(output_dir), label, "cuteafd_" + label)
    abi = validate_abi(output_dir / (label + ".h"), label, "quant")
    grids = [mxfp8_rows_quant_aot_grid(size_k=5120, rows=m, expected_m=80,
             sm_count=48 if manifest['capability'][1] == 1 else 188) for m in range(1, 4097)]
    header = ["#pragma once", f'#include "{label}.h"',
              f"#define CUTEAFD_V41_INPUT_QUANT_ENTRY {abi['symbol']}",
              "static const uint32_t cuteafd_v41_input_quant_grids[] = {" +
              ",".join(map(str, grids)) + "};"]
    (output_dir / "v41_input_quant_dispatch.h").write_text("\n".join(header) + "\n")
    manifest["input_quantizer"] = {"format": "row E4M3 payload then UE8M0 K32 scales",
        "row_bytes": 5280, "max_rows": 4096, "expected_m": 80, "amax_floor": 1e-4,
        "abi": abi, "object_sha256": hashlib.sha256((output_dir / (label + ".o")).read_bytes()).hexdigest()}


# Roles served by the SM121 Spark expert family. `spark` is the historical TP4
# shard; `spark_tp2`/`spark_tp3`/`spark_tp6` are the replicated-group TP shards.
# The degree is a plan-time role property, never derived from live rows.
SPARK_ROLES = ("spark", "spark_tp2", "spark_tp3", "spark_tp6", "spark_tp8")
SPARK_TP_DEGREES = {"spark": 4, "spark_tp2": 2, "spark_tp3": 3, "spark_tp6": 6, "spark_tp8": 8}


def export(output_dir: Path, role: str, rows: tuple[int, ...], input_format: str | None = None, compact_live_rows: int | None = None) -> None:
    if input_format is None:
        input_format = "bf16" if role == "coordinator" else "fp8_k32"
    if input_format not in ("bf16", "fp8_k32") or (role not in SPARK_ROLES and input_format != "bf16"):
        raise ValueError("FP8 K32 input is supported only for Spark backbone experts")
    if role in ("spark_tp2", "spark_tp3", "spark_tp6", "spark_tp8") and input_format != "fp8_k32":
        raise ValueError("Spark TP2/TP3/TP6 use the native FP8 K32 slice export only")
    if compact_live_rows is not None and (role not in SPARK_ROLES or input_format != "fp8_k32"):
        raise ValueError("compact dispatch requires native FP8 Spark experts")
    if role == "coordinator":
        from b12x.moe._shared.kernels.v41_slice_pipeline import V41DraftSlicePipeline
        from export_b12x_slices_aot import export as export_slices

        export_slices(output_dir, rows, V41DraftSlicePipeline.DEFAULT_WIDTH,
                      role=role, standard_names=True)
        return
    if role in SPARK_ROLES and input_format == "fp8_k32":
        from export_b12x_slices_aot import export as export_slices

        # Keep width-192 arithmetic and ordered route output at every capacity.
        # Compact specialization stays limited to the historical TP4 role.
        widths = {capacity: 192 for capacity in rows}
        compact = role == "spark" and compact_live_rows is not None
        export_slices(output_dir, rows, widths,
                      role=role, standard_names=True,
                      compact_max_capacity=16 if compact else None,
                      compact_live_rows=compact_live_rows if compact else None)
        return
    raise ValueError("Spark experts require FP8 K32 input")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--role", choices=("spark", "spark_tp2", "spark_tp3", "spark_tp6", "spark_tp8", "coordinator"), required=True)
    parser.add_argument("--rows", default="1,16,80,256,1024,4096")
    parser.add_argument("--input-format", choices=("bf16", "fp8_k32"))
    parser.add_argument("--compact-live-rows", type=int, help="Experimental Spark live-row compact cutoff")
    args = parser.parse_args()
    rows = tuple(int(value) for value in args.rows.split(","))
    if (
        not rows
        or len(set(rows)) != len(rows)
        or any(value < 1 or value > 4096 for value in rows)
    ):
        parser.error("--rows must contain distinct positive capacities up to 4096")
    export(args.output_dir, args.role, rows, args.input_format, args.compact_live_rows)


if __name__ == "__main__":
    main()

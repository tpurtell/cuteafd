#!/usr/bin/env python3
"""Export native V4.1 slice pipelines through the existing expert ABI.

Explicit widths support comparisons; the ordinary exporter selects qualified coordinator and Spark recipes
and uses standard role-specific artifact names.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1] / _d) for _d in ("lib",)]  # sibling tool dirs

import argparse
import hashlib
import os
import json
import re
from pathlib import Path

from v41_spark_tp3_launch_geometry import launch_geometry
from expert_grouping import native_group_macros, slice_group_rows
from mxfp4_layouts import stored_width, native_slice_width, native_fc1_k_splits, native_prefill, native_prefill_scratch
from spark_packages import native_spark_artifacts, native_width_role

os.environ["B12X_COMPILE_DISK_CACHE"] = "0"
os.environ["B12X_COMPILE_MEMORY_CACHE"] = "0"
import _pinned_sparkinfer

# Plan-time expert placement roles. Geometry, SM guard and native role id are
# properties of the role, never of the live row count. `spark` is the historical
# TP4 shard (576 exact storage); `spark_tp2`/`spark_tp3`/`spark_tp6` are
# the replicated-group shards and are already 128-aligned. These pure tables are
# the authoritative source for the export geometry and are covered by a CPU-only
# test that needs neither torch nor CUDA.
ROLE_GEOMETRY = {
    "coordinator": (128, 2304, 2304, 3),
    "dspark_tp2": (128, 1152, 1152, 3),
    "rtx_tp2": (384, 1152, 1152, 6),
    "rtx_backbone": (384, 2304, 2304, 6),
}
ROLE_GEOMETRY.update({artifact.name: (384, artifact.logical, artifact.stored, 6)
                      for artifacts in native_spark_artifacts().values() for artifact in artifacts})
ROLE_SM = {role: (12, 0) for role in ("coordinator", "dspark_tp2", "rtx_tp2", "rtx_backbone")}
ROLE_SM.update({artifact.name: (12, 1) for artifacts in native_spark_artifacts().values()
                for artifact in artifacts})
# Native `cuteafd_expert_info_t.role` values (see native/shared/include/cuteafd_experts.h).
ROLE_NATIVE_ID = {"coordinator": 0, "rtx_backbone": 2, "rtx_tp2": 3, "dspark_tp2": 4}
ROLE_NATIVE_ID.update({artifact.name: artifact.role for artifacts in native_spark_artifacts().values()
                       for artifact in artifacts})
SPARK_TP_DEGREE = {artifact.name: world for world, artifacts in native_spark_artifacts().items()
                   for artifact in artifacts if not artifact.name.startswith("spark_w")}

# Routed-expert shapes that share the V4.1 storage (packed FP4, E8M0 K32 scales)
# and slice pipeline. The family names the exported symbols (`cuteafd_{family}_*`)
# and must match `ExpertGeometry::family` in cuteafd-core. Other families derive
# each role's slice from the TP degree; draft and coordinator roles are V4.1-only.
FAMILY_GEOMETRY = {
    "v41": {"hidden": 5120, "experts": 384, "topk": 6, "intermediate": 2304},
    "dsv4f": {"hidden": 4096, "experts": 256, "topk": 6, "intermediate": 2048},
    "dsv4p": {"hidden": 7168, "experts": 384, "topk": 6, "intermediate": 3072},
}
ROLE_TP_DEGREE = {"rtx_backbone": 1, "rtx_tp2": 2}


def role_geometry(family, role):
    """(experts, intermediate, kernel_intermediate, topk) for one family role."""
    shape = FAMILY_GEOMETRY[family]
    if role.startswith("spark"):
        artifact = next((artifact for artifacts in native_spark_artifacts(family).values()
                         for artifact in artifacts if artifact.name == role), None)
        if artifact is None:
            raise ValueError(f"{family} has no native Spark artifact {role}")
        return shape["experts"], artifact.logical, artifact.stored, shape["topk"]
    if family == "v41":
        return ROLE_GEOMETRY[role]
    tp = ROLE_TP_DEGREE.get(role)
    if tp is None or shape["intermediate"] % tp or (shape["intermediate"] // tp) % 32:
        raise ValueError(f"{family} has no {role} expert slice")
    intermediate = shape["intermediate"] // tp
    return shape["experts"], intermediate, stored_width(shape["hidden"], intermediate), shape["topk"]


def expert_plane_bytes(family: str, hidden: int, intermediate: int, stored: int) -> tuple[int, ...]:
    """Package-fixed payload/scale extents; coded FC2 uses logical, not padded K."""
    values = hidden * stored
    if family == "v41":
        return (values, 2 * stored * (2 * ((hidden // 32 + 3) // 4) + 1),
                values // 2, hidden * (2 * ((intermediate // 32 + 3) // 4) + 1))
    return values, values // 16, values // 2, values // 32


def numerical_recipe(width: int, intermediate: int, *, atomic: bool, compact: bool) -> dict:
    """Pinned slice products and fold order, independent of launch capacity."""
    if width not in (32, 64, 128, 192) or intermediate <= 0 or intermediate % 32:
        raise ValueError("unsupported native slice reduction geometry")
    return dict(
        schema=1,
        slice_width=width,
        fc1_mma_mnk=[16, 8, 32],
        fc1_k_stage=128,
        fc2_k_steps=width // 32,
        block_threads=128,
        plane_count=(intermediate + width - 1) // width,
        plane_fold="hybrid_live_rows" if compact else "atomic_tokens" if atomic else "ascending_fp32",
        implementation="hybrid" if compact else "slices",
        output_kind="fp32_tokens" if atomic else "fp32_routes",
    )


def fc2_n_splits(role: str, capacity: int, width: int, hidden: int) -> int:
    """Static launch partition; canonical K steps and plane folds stay unchanged."""
    return 2 if role in ("spark", "rtx_tp2") and capacity == 1 and width == 128 and hidden % 256 == 0 else 1


def export(output, capacities, width, atomic_min_capacity=None, role="spark", *, standard_names=False, compact_max_capacity=None, compact_live_rows=None, family="v41"):
    import torch
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_tensor
    from b12x._lib.utils import current_cuda_stream
    from b12x.moe._shared.kernels.v41_slice_pipeline import V41SlicePipeline, V41DraftSlicePipeline
    from export_b12x_v41_experts_aot import export_input_quantizer

    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) not in ((12, 0), (12, 1)):
        raise ValueError("native Blackwell device required")
    width_role = role.startswith("spark_w")
    if role not in ROLE_GEOMETRY and not width_role:
        raise ValueError(f"unsupported expert placement role {role!r}")
    coordinator = role == "coordinator"
    tp2 = role == "rtx_tp2"
    draft_tp2 = role == "dspark_tp2"
    expected_sm = (12, 1) if width_role else ROLE_SM[role]
    if (props.major, props.minor) != expected_sm:
        raise ValueError(
            f"{role} expert slices require SM{expected_sm[0]}{expected_sm[1]}, "
            f"got SM{props.major}{props.minor}"
        )
    if draft_tp2 and atomic_min_capacity is not None:
        raise ValueError("dSpark TP2 uses ordered route output before rank reduction")
    if coordinator and atomic_min_capacity is not None:
        raise ValueError("coordinator slices use ordered route output only")
    # The role fixes the Spark TP degree at plan time and it is baked into this
    # AOT artifact; live rows are never part of the compile key.
    spark_tp_degree = SPARK_TP_DEGREE.get(role)
    experts, intermediate, kernel_intermediate, topk = role_geometry(family, role)
    hidden = FAMILY_GEOMETRY[family]["hidden"]
    wire = hidden + hidden // 32  # FP8 E4M3 row, then one UE8M0 scale per 32 values
    if family != "v41" and compact_max_capacity is not None:
        raise ValueError("compact launch-geometry exports are V4.1-only")
    if compact_max_capacity is not None:
        if role not in ("spark", "rtx_tp2") or compact_max_capacity < 1:
            raise ValueError("compact specialization requires Spark/TP2 and positive capacity")
        if compact_live_rows is None:
            compact_live_rows = 8 if tp2 else 2
        if compact_live_rows < 1 or compact_live_rows > compact_max_capacity:
            raise ValueError("compact live-row cutoff must fit the selected capacities")
    elif compact_live_rows is not None:
        raise ValueError("compact live-row cutoff requires compact capacity selection")
    workspace_sms = 48 if props.minor == 1 else 188
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        schema=1,
        experimental=not standard_names,
        role=role,
        family=family,
        spark_tp_degree=spark_tp_degree,
        input_format="bf16" if coordinator else "fp8_k32",
        weight_scale_layout="ue8m0_offset4_n256_k128" if family == "v41" else "ue8m0_n256_k128",
        expert_plane_bytes=expert_plane_bytes(family, hidden, intermediate, kernel_intermediate),
        sparkinfer_revision=_pinned_sparkinfer.REVISION,
        device=_pinned_sparkinfer.export_target(props),
        capability=[props.major, props.minor],
        physical_sms=props.multi_processor_count,
        width=width,
        compact_max_capacity=compact_max_capacity,
        compact_live_rows=compact_live_rows,
        geometry=dict(experts=experts, hidden=hidden, intermediate=intermediate,
                      kernel_intermediate=kernel_intermediate, topk=topk),
        variants=[],
    )
    includes, entries, reductions = [], [], []
    group_rows = slice_group_rows(_pinned_sparkinfer.SOURCE)
    for capacity in capacities:
        stream_prefill = (standard_names and role not in ("coordinator", "dspark_tp2")
                          and native_prefill(hidden, intermediate, props.major * 10 + props.minor, capacity))
        compact = compact_max_capacity is not None and capacity <= compact_max_capacity
        atomic = not stream_prefill and not compact and atomic_min_capacity is not None and capacity >= atomic_min_capacity
        selected_width = width[capacity] if isinstance(width, dict) else width
        if role.startswith("spark"):
            selected_width = native_slice_width(hidden, intermediate, props.major * 10 + props.minor,
                                                capacity, selected_width)
        routes = capacity * topk
        planes = (intermediate + selected_width - 1) // selected_width
        # The native Spark TP3 launch geometry is recorded from the reviewed
        # identity contract (pinned kernel constants + this variant's compiled
        # width). It is keyed on static capacity and model geometry only: no live
        # row, token, group or expert count enters the export, the compile key,
        # or this audit block. Other roles omit this TP3-specific identity.
        recorded_geometry = (
            launch_geometry(
                capacity,
                selected_width,
                intermediate=intermediate,
                kernel_intermediate=kernel_intermediate,
                output_kind="fp32_tokens" if atomic else "fp32_routes",
                revision=_pinned_sparkinfer.REVISION,
            )
            if family == "v41" and role == "spark_tp3" and not stream_prefill
            else None
        )
        # Ordinary FC1 writes FP8 words plus K32 scale words for raw and coded weights.
        # The hybrid path also reuses this arena for hidden-wide BF16 projections.
        partial_cols = hidden if compact else selected_width // 4 + selected_width // 32
        specs = [
            (cutlass.Uint32, (capacity, hidden // 4), (wire // 4, 1)),
            (cutlass.Uint8, (capacity, hidden // 32), (wire, 1)),
            # Per-expert packed W13, S13, W2 and S2 in 32-bit words.
            *[
                (cutlass.Uint32, (experts * plane_bytes // 4,), (1,))
                for plane_bytes in expert_plane_bytes(family, hidden, intermediate, kernel_intermediate)
            ],
            (cutlass.Int32, (routes,), (1,)),
            (cutlass.Float32, (routes,), (1,)),
            (cutlass.Int32, (1,), (1,)),
            (cutlass.Int32, (experts * routes,), (1,)),
            (cutlass.Int32, (experts,), (1,)),
            (cutlass.Int32, (experts, 2), (2, 1)),
            (cutlass.Int32, (routes, 19), (19, 1)),
            (cutlass.Float32, (routes,), (1,)),
            (cutlass.Int32, (routes,), (1,)),
            (cutlass.Float32, (1,), (1,)) if atomic else
                (cutlass.Float32, (planes, routes, partial_cols), (routes * partial_cols, partial_cols, 1)),
            (cutlass.Float32, (capacity * hidden,), (1,)) if atomic else
                (cutlass.Float32, (routes, hidden), (hidden, 1)),
        ]
        args = [
            make_fake_tensor(dtype, shape, stride, assumed_align=16)
            for dtype, shape, stride in specs
        ]
        label = (f"{family}_{role}_m{capacity}" if standard_names
                 else f"v41_slices_m{capacity}_w{selected_width}")
        pipeline = (V41DraftSlicePipeline(capacity, selected_width, scale_codes=family == "v41")
                    if coordinator else V41SlicePipeline(capacity, selected_width, atomic_tokens=atomic,
                                               experts=experts, topk=topk, intermediate=intermediate,
                                               hidden=hidden, scale_codes=family == "v41",
                                               exact_storage=kernel_intermediate == intermediate and intermediate % 128 != 0,
                                               fc2_n_splits=fc2_n_splits(role, capacity, selected_width, hidden),
                                               fc1_k_splits=(native_fc1_k_splits(
                                                   hidden, intermediate, props.major * 10 + props.minor, capacity)
                                                   if role.startswith("spark") else 1)))
        if stream_prefill:
            from b12x.moe._shared.kernels.v4_stream_pipeline import V4StreamPipeline, stream_scratch_layout
            from b12x._lib.utils import make_ptr
            pipeline = V4StreamPipeline(capacity, hidden=hidden, intermediate=intermediate,
                                        experts=experts, topk=topk, scale_codes=family == "v41")
            stream_layout, stream_size = native_prefill_scratch(capacity, hidden, intermediate, experts, topk)
            pinned_layout, pinned_size = stream_scratch_layout(capacity, hidden, intermediate, experts, topk)
            if tuple(stream_layout) != pinned_layout or stream_size != pinned_size:
                raise ValueError("native STREAM planner scratch differs from pinned pipeline")
            scale_type = cutlass.Uint8 if family == "v41" else cutlass.Uint32
            pointer_types = [cutlass.Uint8, cutlass.Uint32, scale_type, cutlass.Uint32,
                             scale_type, cutlass.Int32, cutlass.Float32,
                             cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Uint8,
                             cutlass.Float32, cutlass.Int32, cutlass.Float32]
            args = [make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16) for dtype in pointer_types]

        if compact:
            from b12x.moe._shared.kernels.v41_compact_pipeline import V41HybridPipeline
            pipeline = V41HybridPipeline(capacity, selected_width,
                intermediate=intermediate, cutoff=min(capacity, compact_live_rows), scale_codes=True)
        if coordinator:
            args.insert(0, make_fake_tensor(cutlass.BFloat16, (capacity, 5120), (5120, 1), assumed_align=16))
        compiled = cute.compile(
            pipeline,
            *args,
            cutlass.Int32(capacity),
            current_cuda_stream(),
        )
        compiled.export_to_c(str(output), label, "cuteafd_" + label)
        header = (output / f"{label}.h").read_text()
        symbol = re.findall(
            r"void (_mlir_\w+)\(void \*\*args, int32_t num_args\);", header
        )
        if len(symbol) != 1:
            raise ValueError("unexpected slice export entry")
        names = (
            "x",
            "xs",
            "w13",
            "s13",
            "w2",
            "s2",
            "ids",
            "routing",
            "live",
            "packed",
            "counts",
            "prefixes",
            "metadata",
            "grouped",
            "inverse",
            "partial",
            "output",
        )
        if stream_prefill:
            names = ("x", "w13", "s13", "w2", "s2", "ids", "routing", "meta",
                     "pair_row", "pair_pos", "act", "grouped_routing", "grouped_route", "output")
        if coordinator:
            names = ("source", *names)
        argument_count = len(names) + 3
        signature = re.search(
            r"static inline int32_t cute_dsl_\w+_wrapper\(([^)]*)\)", header
        )
        prefix = "cuteafd_" + label
        declarations = (
            [f"{prefix}_Kernel_Module_t*module"]
            + [f"void*{name}" if stream_prefill else f"{prefix}_Tensor_{name}_t*{name}" for name in names]
            + ["int32_trows", "cudaStream_tstream"]
        )
        if (
            signature is None
            or [re.sub(r"\s+", "", x) for x in signature[1].split(",")] != declarations
        ):
            raise ValueError("unexpected slice parameter ABI")
        arguments = re.search(rf"void \*args\[{argument_count}\] = \{{([^}}]+)\}}", header)
        if arguments is None or re.sub(r"\s+", "", arguments[1]) != ",".join(
            name for name in (*(tuple("&" + name for name in names) if stream_prefill else names), "&rows", "&stream", "&ret")
        ):
            raise ValueError("unexpected slice argument order")
        # Static tensor layouts must expose exactly one data pointer each.
        bodies = re.findall(r"typedef struct\s*\{([^}]+)\}\s*\w+_Tensor_\w+_t;", header)
        if (len(bodies) != (0 if stream_prefill else len(names))) or any(
            re.sub(r"\s+", "", b) != "void*data;" for b in bodies
        ):
            raise ValueError("unexpected slice tensor ABI")
        offset, scratch, offsets = 0, [], [None] * 44
        # Existing weight binder owns slots 22..33, 38/39. Other unused slots
        # point at valid dummy scratch to preserve public nonnull validation.
        scratch_slots = [3, 4, 5, 6, 7, 8, 9, 10, 41]
        for index, slot in ([] if stream_prefill else zip(range(8, 17), scratch_slots)):
            dtype, shape, _ = specs[index]
            size = dtype.width // 8
            for dim in shape:
                size *= dim
            offset = (offset + 15) // 16 * 16
            offsets[slot] = offset
            scratch.append(dict(slot=slot, offset=offset, nbytes=size, shape=shape))
            offset += size
        if stream_prefill:
            scratch_slots = [3, 4, 5, 6, 7, 8, 41]
            for plane, slot in zip(stream_layout, scratch_slots, strict=True):
                offsets[slot] = plane["offset"]
                scratch.append(dict(slot=slot, **plane))
            offset = stream_size
        for slot in (37, 40):
            offset = (offset + 15) // 16 * 16
            offsets[slot] = offset
            offset += experts * 4
        if coordinator:
            offset = (offset + 15) // 16 * 16
            offsets[11] = offset
            scratch.append(dict(slot=11, offset=offset, nbytes=capacity*5280, shape=(capacity, 5280)))
            offset += capacity * 5280
        for slot in list(range(3, 22)) + [34, 35, 36, 42, 43]:
            if offsets[slot] is None:
                offsets[slot] = 0
        includes.append(f'#include "{label}.h"')
        # The old entry gets 44 pointer addresses, seven scalar addresses,
        # stream and status. Repackage them into the generated static ABI.
        mapping = [0, None, 30, 31, 32, 33, 1, 2, *scratch_slots]
        if stream_prefill:
            mapping = [0, 30, 31, 32, 33, 1, 2, *scratch_slots]
        input_slot = 11 if coordinator else 0
        mapping[0] = input_slot
        if coordinator:
            mapping.insert(0, 0)
        mapped = [
            f"args[{slot}]" if slot is not None else "&scales" for slot in mapping
        ]
        includes.extend(
            [
                f"static void {label}_bridge(void** args, int32_t count) {{",
                "  if (count != 53) { *static_cast<int32_t*>(args[52]) = 1; return; }",
                f"  void* scales = static_cast<char*>(*static_cast<void**>(args[{input_slot}])) + {hidden};",
                f"  void* mapped[] = {{{', '.join(mapped)}, args[44], args[51], args[52]}};",
                f"  {symbol[0]}(mapped, {argument_count});",
                "}",
            ]
        )
        info = [
            3 if atomic else 2,
            native_width_role(intermediate) if width_role else ROLE_NATIVE_ID[role],
            experts,
            hidden,
            intermediate,
            kernel_intermediate,
            topk,
            capacity,
            offset,
            capacity,
            capacity,
            routes,
            routes,
            workspace_sms,
            1 if coordinator else 7,
        ]
        prefix = "_mlir_cuteafd_" + label
        entries.append(
            "{{"
            + ",".join(map(str, info))
            + "},"
            + prefix
            + "_cuda_init,"
            + prefix
            + "_cuda_load_to_device,"
            + label
            + "_bridge,{"
            + ",".join("UINT64_MAX" if x is None else str(x) for x in offsets)
            + "}}"
        )
        recipe = numerical_recipe(selected_width, intermediate, atomic=atomic, compact=compact)
        if stream_prefill:
            recipe = dict(schema=2, slice_width=128, fc1_mma_mnk=[16, 8, 32],
                          fc1_k_stage=128, fc2_k_steps=(intermediate + 31) // 32,
                          block_threads=256, plane_count=1, plane_fold="original_routes",
                          implementation="native_stream", output_kind="fp32_routes")
            reductions.append(f"case {capacity}: *out = {{2,128,{recipe['fc2_k_steps']},1,4,3,32,256,128}}; return 0;")
        else:
            reductions.append(
                f"case {capacity}: *out = {{1,{selected_width},{recipe['fc2_k_steps']},"
                f"{planes},{3 if compact else 2 if atomic else 1},{2 if compact else 1},32,128,128}}; return 0;")
        manifest["variants"].append(
            dict(
                name=label,
                implementation="native_stream" if stream_prefill else "hybrid" if compact else "slices",
                group_rows=128 if stream_prefill else group_rows,
                compact_group_rows=1 if compact else None,
                compact_live_rows=min(capacity, compact_live_rows) if compact else None,
                width=selected_width,
                **({"fc1_k_splits": pipeline.fc1.k_splits}
                   if not compact and not atomic and not coordinator and not stream_prefill else {}),
                numerical_recipe=recipe,
                output_kind="fp32_tokens" if atomic else "fp32_routes",
                native_abi_version=3 if atomic else 2,
                capacity_rows=capacity,
                core_scratch_nbytes=offset,
                scratch=scratch,
                scratch_pointer_offsets=offsets,
                native_entry=symbol[0],
                **({"launch_geometry": recorded_geometry}
                   if recorded_geometry is not None else {}),
            )
        )
    (output / "v41_expert_variants.h").write_text(
        "\n".join(
            [
                "#pragma once",
                *includes,
                f"#define CUTEAFD_V41_CC_MINOR {props.minor}",
                f"#define CUTEAFD_V41_SMS {workspace_sms}",
                "#define CUTEAFD_V41_VARIANTS " + ",".join(entries),
                *native_group_macros(manifest["variants"]),
                "#define CUTEAFD_EXPERT_REDUCTION_CASES " + " ".join(reductions),
                "#define CUTEAFD_V41_OUTPUT_KIND(capacity) (" +
                (" || ".join(f"((capacity)=={v['capacity_rows']})" for v in manifest['variants']
                             if v['output_kind'] == 'fp32_tokens') or "0") + " ? 1u : 0u)",
                "",
            ]
        )
    )
    if family == "v41":
        # Other families receive rows the coordinator already quantized.
        export_input_quantizer(output, manifest)
    # Sorted: directory order is filesystem-specific and made two hosts'
    # otherwise identical manifests differ (#127).
    manifest["artifact_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.iterdir())
        if path.suffix in (".h", ".o")
    }
    (output / "v41_experts.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", default="spark", help="placement role or geometry-derived spark_wWIDTH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--geometry", choices=sorted(FAMILY_GEOMETRY), default="v41",
                        help="routed-expert family; names the exported symbols")
    parser.add_argument("--rows", default="1,16,80")
    parser.add_argument(
        "--width",
        required=True,
        help="32/64/128/192 or explicit capacity:width pairs, e.g. 1:64,16:192,80:192",
    )
    parser.add_argument("--atomic-min-capacity", type=int, choices=[256, 1024, 4096],
                        help="Use ABI 3 direct FP32 token accumulation at these larger capacities")
    parser.add_argument("--standard-names", action="store_true")
    parser.add_argument("--compact-max-capacity", type=int, help="Experimental Spark/TP2 hybrid specialization up to this capacity")
    parser.add_argument("--compact-live-rows", type=int, help="Live-row cutoff within hybrid capacity; defaults to Spark 2 / RTX 8")
    args = parser.parse_args()
    capacities = tuple(int(x) for x in args.rows.split(","))
    if (
        not capacities
        or len(set(capacities)) != len(capacities)
        or any(x < 1 or x > 4096 for x in capacities)
    ):
        parser.error("rows must be unique capacities in 1..4096")
    try:
        if ":" in args.width:
            pairs = [tuple(map(int, x.split(":"))) for x in args.width.split(",")]
            width = dict(pairs)
            if len(width) != len(pairs) or set(width) != set(capacities):
                raise ValueError("width map must cover every capacity exactly once")
            widths = width.values()
        else:
            width = int(args.width)
            widths = [width]
        if any(w not in (32, 64, 128, 192) for w in widths):
            raise ValueError("width must be 32, 64, 128 or 192")
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    export(args.output_dir, capacities, width, args.atomic_min_capacity, args.role,
           standard_names=args.standard_names, compact_max_capacity=args.compact_max_capacity, compact_live_rows=args.compact_live_rows,
           family=args.geometry)

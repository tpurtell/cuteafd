#!/usr/bin/env python3
"""Build or verify an exact FP8 routed-expert package (``fp8-<geometry>``).

The checkpoint's E4M3 experts with FP32 128x128 block scales (or MXFP4
experts, ``mimop``: packed E2M1 + UE8M0 per 32; or NVIDIA ModelOpt NVFP4
experts, ``<family>_nvfp4`` into ``fp8-<family>-nvfp4``: packed E2M1 + E4M3
per 16 + FP32 per-expert weight and activation calibration, W4A4 from original
BF16 inputs at every row count) run as
``b12x.integration.cuteafd.fp8_moe`` programs (route ``auto``: grouped GEMV
up to 1024 live rows (MXFP4 on GB10: 640) and the expert-stationary streaming
GEMMs above on GB10 (spark packages, wire input; MXFP4 gate/up by
block-scaled E4M3 x E2M1 MMAs); GEMV up to 2048 rows and the grouped TMA GEMM
above for SM120 / BF16 input). A package holds one directory
per layout, each with ``libcuteafd_fp8moe.so`` carrying one program per
capacity, and a verified ``manifest.json``:

  fp8-mimo/
    manifest.json
    tp1/libcuteafd_fp8moe.so     full intermediate (RTX local / MTP layers)
    tp2/libcuteafd_fp8moe.so     half (Spark TP2)
    tp4/libcuteafd_fp8moe.so     quarter (Spark TP4: every rank runs the same slice width)
    tp6/libcuteafd_fp8moe.so     sixth in whole 128-blocks (2048: 3/2 blocks, stored 384 zero-padded)

Library ABI (``native/shared/include/cuteafd_fp8_moe.h``)::

    int32_t cuteafd_fp8moe_info(uint32_t* words, uint32_t count);
        words: [0] bytes [version=5, storage, activation, projection]. Storage:
        1 FP8 block128, 2 MXFP4, 4 NVFP4 W4A4, 5 BF16 (no scales).
        Activation: 1 pre-SiLU clamp/BF16 intermediate, 2 post-SiLU clamp/FP32
        product. Projection: 1 BF16 gate/up and down. All bytes are mandatory.
        [1] hidden, [2] slice, [3] experts, [4] top-k,
        [5] intermediate, [6] tp, [7] input dtype (7 = FP8 K32 wire rows, 1 = BF16 rows),
        [8] SwiGLU limit (FP32 bits, 0 = none), [9] capacities n, [10..] capacities
    int32_t cuteafd_fp8moe_scratch_bytes(uint32_t capacity, uint64_t* bytes);
    int32_t cuteafd_fp8moe_create(void** context);           // loads every program on the current device
    int32_t cuteafd_fp8moe_launch(void* context, uint32_t capacity, void* const* pointers,
                                  int32_t rows, void* stream);  // pointers: the fp8_moe ABI (11)
    void    cuteafd_fp8moe_destroy(void* context);
    int32_t cuteafd_fp8moe_set_options(void* context, uint32_t options);
        // the large-row form of FP8 programs: 0 the package default ("auto"),
        // 1 W8A16 (the former programs), 2 W8A8 (BF16 rows quantized to wire rows)

FP8 packages run wire-input large row counts W8A8 (b12x fp8_moe: block-scaled
E4M3 x E4M3 gate/up over the wire rows) and BF16 rows W8A16 by default. Each
capacity whose program differs in another form also carries that form's
program (compile_fp8_moe_aot(prefill="w8a16" | "w8a8")), launched while the
matching option is set.

  package_fp8_moe_aot.py build --role spark --geometry mimo --output DIR --build-dir DIR \\
      [--cross-sm121] --cxx c++ --cuda-include I --cuda-libdir L --runtime libcute_dsl_runtime.so
  package_fp8_moe_aot.py verify --package DIR
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1] / _d) for _d in ("lib",)]  # sibling tool dirs

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("B12X_COMPILE_DISK_CACHE", "0")
os.environ.setdefault("B12X_COMPILE_MEMORY_CACHE", "0")

SCHEMA = "cuteafd.fp8moe-package.v1"
ROLE_LAYOUTS = {"spark": ("tp4", "tp2", "tp6"), "coordinator": ("tp1", "tp2")}
MXFP4_ROLE_LAYOUTS = {"spark": ("tp6", "tp2"), "coordinator": ("tp1", "tp2")}
# NVFP4 slices own whole 16-value blocks: every Spark world the FP8 worker runs.
NVFP4_ROLE_LAYOUTS = {"spark": ("tp4", "tp2", "tp3", "tp6"), "coordinator": ("tp1", "tp2")}
ABI_VERSION = 5
STORAGE = {"fp8": 1, "mxfp4": 2, "nvfp4a4": 4, "bf16": 5}
ACTIVATION = {"silu_bf16_product": 1, "silu_post_clamp_fp32_product": 2}
PROJECTION = {"bf16": 1}
ROLE_COMPUTE = {"spark": (12, 1), "coordinator": (12, 0)}
# Spark ranks take the FP8 K32 wire rows of the expert protocol; coordinator
# experts take the BF16 activations directly (no input quantization at all).
ROLE_INPUT = {"spark": "wire", "coordinator": "bf16"}


def input_kind(role: str, weights: str, requested: str | None = None) -> str:
    """NVFP4 quantizes the original BF16 row once with each expert's scale."""
    if weights in ("nvfp4", "bf16"):
        if requested not in (None, "bf16"):
            raise ValueError(f"{weights.upper()} requires BF16 input; FP8 K32 wire loses checkpoint numerics")
        return "bf16"
    return requested or ROLE_INPUT[role]

LIBRARY = "libcuteafd_fp8moe.so"
POINTERS = 11
INFO_WORDS = 16
# Large-row program forms, in cuteafd_fp8moe_set_options order.
FORMS = ("auto", "w8a16", "w8a8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capacity_scratch(programs: list[dict]) -> list[dict]:
    """Match the static ABI reservation across every retained prefill form."""
    scratch = {}
    for program in programs:
        capacity = program["capacity"]
        scratch[capacity] = max(scratch.get(capacity, 0), program["scratch"])
    return [{"capacity": capacity, "scratch_bytes": scratch[capacity]}
            for capacity in sorted(scratch)]


def bridge_source(programs: list[dict], info: list[int]) -> str:
    includes = "\n".join(f'#include "{p["stem"]}.h"' for p in programs)
    rows = ",\n".join(
        f'  {{{p["capacity"]}u, {p["scratch"]}ull, {FORMS.index(p["form"])}u, _mlir_cuteafd_{p["stem"]}_cuda_init, '
        f'_mlir_cuteafd_{p["stem"]}_cuda_load_to_device, {p["entry"]}}}' for p in programs)
    grouping_rows = []
    for p in programs:
        grouping = p["weight_read_grouping"]
        contract = {"exact_grouping": 1, "conservative_route_upper_bound": 2}[grouping["contract"]]
        previous = 0
        for segment in grouping["segments"]:
            end, group = segment["end_rows"], segment["group_rows"]
            if not previous < end <= p["capacity"] or group < 1:
                raise ValueError("invalid installed weight-read grouping segment")
            if contract == 2 and group != 1:
                raise ValueError("conservative route bound must charge each routed occurrence")
            grouping_rows.append(f'  {{{p["capacity"]}u, {FORMS.index(p["form"])}u, {end}u, {group}u, {contract}u}}')
            previous = end
        if previous != p["capacity"]:
            raise ValueError("installed weight-read grouping does not cover capacity")
    grouping_table = ",\n".join(grouping_rows)
    words = ", ".join(f"{w}u" for w in info)
    return f"""// Generated by package_fp8_moe_aot.py: exact FP8 routed-expert programs.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <mutex>
{includes}

namespace {{
using ModuleFn = void (*)(void**);
using LaunchFn = void (*)(void**, int32_t);
struct Program {{
  uint32_t capacity;
  uint64_t scratch;
  uint32_t form;  // 0 the default program, 1 its W8A16 form, 2 its W8A8 form
  ModuleFn initialize;
  ModuleFn load;
  LaunchFn launch;
}};
const Program kPrograms[] = {{
{rows}
}};
constexpr uint32_t kCount = sizeof(kPrograms) / sizeof(kPrograms[0]);
const uint32_t kInfo[{INFO_WORDS}] = {{{words}}};
struct WeightReadGroup {{ uint32_t capacity, form, end_rows, group_rows, contract; }};
const WeightReadGroup kWeightReadGroups[] = {{
{grouping_table}
}};
struct Context {{
  uint32_t form;
}};
// AOT launchers keep package-global kernel handles. Reuse the same libraries
// across ranks so per-device loading never replaces another rank's handles.
std::mutex library_mutex;
cudaLibrary_t libraries[kCount] = {{}};
uint32_t library_references = 0;
void unload_libraries() {{
  for (uint32_t i = 0; i < kCount; ++i) {{
    if (libraries[i]) cudaLibraryUnload(libraries[i]);
    libraries[i] = nullptr;
  }}
}};
// The capacity's program in the requested form; the default program when the
// capacity has no such form (the forms are the same program there).
const Program* find(uint32_t capacity, uint32_t form) {{
  const Program* fallback = nullptr;
  for (uint32_t i = 0; i < kCount; ++i) {{
    if (kPrograms[i].capacity != capacity) continue;
    if (kPrograms[i].form == form) return &kPrograms[i];
    if (!kPrograms[i].form) fallback = &kPrograms[i];
  }}
  return fallback;
}}
}}  // namespace

extern "C" int32_t cuteafd_fp8moe_info(uint32_t* words, uint32_t count) {{
  if (!words || count < {INFO_WORDS}) return cudaErrorInvalidValue;
  std::memcpy(words, kInfo, sizeof(kInfo));
  return cudaSuccess;
}}

// Static query: zero rows names the selected program's final segment.
extern "C" int32_t cuteafd_fp8moe_weight_read_group(uint32_t capacity, uint32_t form, uint32_t rows,
                                                    uint32_t* group_rows, uint32_t* contract) {{
  if (form > 2u || rows > capacity || !group_rows || !contract) return cudaErrorInvalidValue;
  const Program* program = find(capacity, form);
  if (!program) return cudaErrorInvalidValue;
  const uint32_t required = rows ? rows : capacity;
  for (const auto& group : kWeightReadGroups) {{
    if (group.capacity == capacity && group.form == program->form && required <= group.end_rows) {{
      *group_rows = group.group_rows;
      *contract = group.contract;
      return cudaSuccess;
    }}
  }}
  return cudaErrorInvalidValue;
}}

// Scratch of the capacity's largest form (callers size scratch before choosing).
extern "C" int32_t cuteafd_fp8moe_scratch_bytes(uint32_t capacity, uint64_t* bytes) {{
  if (!find(capacity, 0u) || !bytes) return cudaErrorInvalidValue;
  *bytes = 0;
  for (uint32_t i = 0; i < kCount; ++i)
    if (kPrograms[i].capacity == capacity && kPrograms[i].scratch > *bytes) *bytes = kPrograms[i].scratch;
  return cudaSuccess;
}}

extern "C" int32_t cuteafd_fp8moe_set_options(void* context, uint32_t options) {{
  auto* ctx = static_cast<Context*>(context);
  if (!ctx || options > 2u) return cudaErrorInvalidValue;
  ctx->form = options;
  return cudaSuccess;
}}

extern "C" void cuteafd_fp8moe_destroy(void* context) {{
  auto* ctx = static_cast<Context*>(context);
  if (!ctx) return;
  std::lock_guard<std::mutex> guard(library_mutex);
  if (--library_references == 0) unload_libraries();
  delete ctx;
}}

extern "C" int32_t cuteafd_fp8moe_create(void** context) {{
  if (!context) return cudaErrorInvalidValue;
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  std::lock_guard<std::mutex> guard(library_mutex);
  for (uint32_t i = 0; i < kCount; ++i) {{
    cudaLibrary_t* library = &libraries[i];
    if (!library_references) {{
      void* init_args[] = {{&library, &status}};
      kPrograms[i].initialize(init_args);
      if (status != cudaSuccess) {{ unload_libraries(); return status; }}
    }}
    // The AOT loader sets dynamic-SMEM opt-in for this device on the shared
    // kernels, including programs needing more than the default 48 KiB.
    int32_t device_id = device;
    void* load_args[] = {{&library, &device_id, &status}};
    kPrograms[i].load(load_args);
    if (status != cudaSuccess) {{
      if (!library_references) unload_libraries();
      return status;
    }}
  }}
  auto* ctx = new Context{{}};
  ++library_references;
  *context = ctx;
  return cudaSuccess;
}}

extern "C" int32_t cuteafd_fp8moe_launch(void* context, uint32_t capacity, void* const* pointers, int32_t rows,
                                         void* stream) {{
  if (!context) return cudaErrorInvalidValue;
  const Program* program = find(capacity, static_cast<Context*>(context)->form);
  if (!program || !pointers || rows < 1 || uint32_t(rows) > capacity) return cudaErrorInvalidValue;
  void* values[{POINTERS}];
  void* arguments[{POINTERS} + 3];
  for (uint32_t i = 0; i < {POINTERS}; ++i) {{
    const bool unused_scale = ((kInfo[0] >> 8) & 0xffu) == 5u && (i == 4u || i == 6u || i == 8u);
    if (!pointers[i] && !unused_scale) return cudaErrorInvalidValue;
    values[i] = pointers[i];
    arguments[i] = &values[i];
  }}
  int32_t count = rows;
  int32_t result = 0;
  arguments[{POINTERS}] = &count;
  arguments[{POINTERS} + 1] = &stream;
  arguments[{POINTERS} + 2] = &result;
  program->launch(arguments, {POINTERS} + 3);
  return result;
}}
"""


def exact_widths(intermediate: int, tp: int, unit: int = 128) -> list[int]:
    """Stored widths of TP ranks owning whole 128-row blocks exactly (TP6 of
    2048: 384, 384, 384, 384, 256, 256), as the worker's exact layouts expect."""
    if unit not in (32, 128):
        return []
    blocks = intermediate // unit
    if not 1 <= tp <= 6 or intermediate % unit or blocks < tp:
        return []
    return [(blocks // tp + (rank < blocks % tp)) * unit for rank in range(tp)]


def spark_layouts(intermediate: int, *, exact: bool = True) -> list[str]:
    """Every meaningful TP1..6 layout, with unequal whole-H128 rank extents."""
    layouts = []
    for tp in range(1, 7):
        widths = exact_widths(intermediate, tp)
        if not widths:
            continue
        layout = f"tp{tp}"
        layouts.append(layout)
        if exact and len(set(widths)) > 1:
            layouts.extend(f"{layout}-w{width}" for width in sorted(set(widths), reverse=True))
    return layouts


def with_width(g, width: int):
    """``g`` compiled for a stored slice of ``width`` rows (an exact layout):
    the programs read only ``g.slice``, so the subclass overrides it."""
    from dataclasses import dataclass, fields

    @dataclass(frozen=True)
    class ExactSlice(type(g)):
        width: int = 0

        @property
        def slice(self) -> int:  # noqa: D401 - overrides the padded width
            return self.width

    values = {f.name: getattr(g, f.name) for f in fields(g)}
    return ExactSlice(**values, width=int(width))


def layout_geometry(base, layout: str):
    """``tp<n>`` (padded) or ``tp<n>-w<width>`` (exact) -> (tp, geometry)."""
    tp_part, _, width = layout.partition("-w")
    tp = int(tp_part.removeprefix("tp"))
    if tp not in range(1, 7):
        raise SystemExit(f"{layout}: Spark TP must be in 1..6")
    if tp in (1, 5) and not exact_widths(base.intermediate, tp):
        raise SystemExit(f"{layout}: intermediate {base.intermediate} needs a nonempty whole-H128 slice per rank")
    g = base.with_tp(tp)
    if width:
        widths = exact_widths(base.intermediate, tp)
        if base.weights == "mxfp4" and tp in (3, 6) and base.intermediate // 128 % tp:
            widths += exact_widths(base.intermediate, tp, 32)
        if int(width) not in widths:
            raise SystemExit(f"{layout}: no rank of TP{tp} owns {width} rows of {base.intermediate} exactly")
        g = with_width(g, int(width))
    return tp, g


def info_words(g, capacities: list[int], wire: bool) -> list[int]:
    limit = struct.unpack("<I", struct.pack("<f", float(g.swiglu_limit)))[0]
    import math

    if not math.isfinite(g.swiglu_limit) or g.swiglu_limit < 0:
        raise ValueError("SwiGLU limit must be finite and nonnegative")
    if g.kind not in STORAGE or g.activation not in ACTIVATION or g.projection not in PROJECTION:
        raise ValueError("unknown expert storage/activation/projection contract")
    if g.kind == "mxfp4" and g.activation != "silu_bf16_product":
        raise ValueError("MXFP4 supports only the pre-SiLU activation contract")
    if wire and g.weights in ("nvfp4", "bf16"):
        raise ValueError("NVFP4/BF16 packages require BF16 input")
    if not capacities or any(c <= 0 or c > 0x7fffffff for c in capacities) or capacities != sorted(set(capacities)):
        raise ValueError("capacities must be positive, strictly increasing int32 values")
    contract = ABI_VERSION | STORAGE[g.kind] << 8 | ACTIVATION[g.activation] << 16 | PROJECTION[g.projection] << 24
    words = [contract, g.hidden, g.slice, g.experts, g.top_k, g.intermediate, g.tp, 7 if wire else 1, limit,
             len(capacities), *capacities]
    if len(words) > INFO_WORDS:
        raise ValueError(f"at most {INFO_WORDS - 10} capacities per package")
    return words + [0] * (INFO_WORDS - len(words))


def build(args: argparse.Namespace) -> None:
    if args.cross_sm121:
        from exl3_cross_sm121 import describe_gb10

        describe_gb10()
    import _pinned_sparkinfer
    import torch
    from b12x.integration.cuteafd import exportable_compilation, validate_exported_header
    from b12x.integration.cuteafd.fp8_moe import (
        GEOMETRIES, compile_fp8_moe_aot, fp8_moe_scratch_bytes, prefill_forms,
    )

    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) != ROLE_COMPUTE[args.role]:
        raise SystemExit(f"{args.role} packages need compute {ROLE_COMPUTE[args.role]} "
                         f"(use --cross-sm121 on an SM120 host)")
    base = GEOMETRIES[args.geometry]
    capacities = sorted({int(v) for v in args.capacities.split(",")})
    layouts = args.layouts.split(",") if args.layouts else list(ROLE_LAYOUTS[args.role])
    if base.weights == "mxfp4" and not args.layouts:
        # MXFP4 slices pad to 128 (MiMo V2.6 Pro: TP6 over all six Sparks, TP2 x EP3 shards).
        layouts = list(MXFP4_ROLE_LAYOUTS[args.role])
        if args.geometry == "mimof" and args.role == "spark":
            layouts = ["tp2", "tp3", "tp4"]
    elif base.weights == "nvfp4" and not args.layouts:
        layouts = [l for l in NVFP4_ROLE_LAYOUTS[args.role]
                   if base.intermediate // 16 >= int(l.removeprefix("tp"))]
    # Default Spark layouts keep the TP degrees that split the intermediate into
    # whole 128-row blocks (Qwen 3.8 Flash Next's 640 splits into none), and
    # TP6 of any intermediate of at least six blocks: uneven whole blocks
    # zero-padded to the widest (2048: 3, 3, 3, 3, 2, 2 blocks stored as 384).
    elif args.role == "spark" and not args.layouts and not getattr(args, "all_spark_counts", False):
        def servable(tp: int) -> bool:
            blocks = base.intermediate // 128
            return base.intermediate % (128 * tp) == 0 or (tp == 6 and blocks >= 6)

        layouts = [l for l in layouts if servable(int(l.removeprefix("tp")))]
        if not layouts:
            raise SystemExit(f"{args.geometry}: intermediate {base.intermediate} has no default {args.role} "
                             "TP layout of whole 128-row blocks; pass --layouts")
    if getattr(args, "all_spark_counts", False):
        if args.role != "spark":
            raise SystemExit("--all-spark-counts requires Spark role")
        if args.layouts:
            raise SystemExit("--all-spark-counts and --layouts are mutually exclusive")
        layouts = spark_layouts(base.intermediate)
    if args.geometry.startswith("glmfdense") and not args.layouts:
        layouts = ["tp1"]  # One always-selected dense expert is not an RTX TP2 backend.
    if args.exact_slices:
        # Exact layouts beside each padded TP layout whose ranks would store padding.
        for layout in list(layouts):
            tp = int(layout.partition("-w")[0].removeprefix("tp"))
            widths = sorted(set(exact_widths(base.intermediate, tp)), reverse=True)
            if tp > 1 and widths and base.with_tp(tp).slice * tp > base.intermediate:
                layouts += [f"tp{tp}-w{w}" for w in widths if f"tp{tp}-w{w}" not in layouts]
    # Fine-unit MXFP4 programs coexist with the qualified H128 layouts while
    # balanced placement is gated. One width is selected for each rank/layer.
    if args.exact_slices and base.weights == "mxfp4" and args.role == "spark":
        for tp in (3, 6):
            if f"tp{tp}" in layouts and base.intermediate // 128 % tp:
                layouts += [f"tp{tp}-w{width}" for width in sorted(set(exact_widths(base.intermediate, tp, 32)), reverse=True)
                            if f"tp{tp}-w{width}" not in layouts]
    wire = input_kind(args.role, base.weights, args.input) == "wire"
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; remove it or choose another --output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": SCHEMA, "role": args.role, "geometry": args.geometry,
                "sparkinfer_revision": _pinned_sparkinfer.REVISION, "compute": [props.major, props.minor],
                "sms": props.multi_processor_count, "layouts": {}, "files": {}}
    with tempfile.TemporaryDirectory(prefix=".fp8moe-package-", dir=args.output.parent) as temporary:
        stage = Path(temporary)
        for layout in layouts:
            tp, g = layout_geometry(base, layout)
            raw = args.build_dir / args.geometry / layout
            if raw.exists():
                shutil.rmtree(raw)
            raw.mkdir(parents=True)
            programs = []
            # Each capacity's default program, plus its other large-row forms where they differ.
            forms = [(c, "auto") for c in capacities]
            forms += [(c, form) for c in capacities for form in prefill_forms(g, c, wire)]
            for capacity, form in forms:
                stem = (f"fp8moe_{args.geometry}_{layout.replace('-', '_')}_m{capacity}"
                        f"{'' if form == 'auto' else '_' + form}")
                with exportable_compilation():
                    program = compile_fp8_moe_aot(g, route="auto", max_rows=capacity, wire=wire, prefill=form)
                program.export_to_c(str(raw), stem, "cuteafd_" + stem)
                checked = validate_exported_header(program, raw / f"{stem}.h", "cuteafd_" + stem)
                if checked["argument_count"] != POINTERS + 2:
                    raise ValueError(f"{stem}: unexpected ABI arity {checked['argument_count']}")
                programs.append({"stem": stem, "capacity": capacity, "entry": checked["symbol"], "form": form,
                                 "scratch": fp8_moe_scratch_bytes(g, "auto", capacity, wire, form),
                                 "weight_read_grouping": program.geometry["weight_read_grouping"]})
                print(f"exported {stem} (scratch {programs[-1]['scratch']} B)", flush=True)
            (raw / "fp8moe_bridge.cc").write_text(bridge_source(programs, info_words(g, capacities, wire)))
            target = stage / layout / LIBRARY
            target.parent.mkdir(parents=True)
            subprocess.run([args.cxx, "-shared", "-fPIC", "-std=c++17", f"-I{args.cuda_include}", f"-I{raw}",
                            str(raw / "fp8moe_bridge.cc"), *[str(raw / f"{p['stem']}.o") for p in programs],
                            f"-L{args.cuda_libdir}", "-lcudart", f"-L{args.runtime.parent}", "-lcute_dsl_runtime",
                            "-Wl,-z,defs", "-o", str(target)], check=True)
            manifest["layouts"][layout] = {
                "tp": tp, "hidden": g.hidden, "slice": g.slice, "experts": g.experts, "top_k": g.top_k,
                "intermediate": g.intermediate, "swiglu_limit": g.swiglu_limit, "input": "wire" if wire else "bf16",
                "weights": g.weights, "activation": g.activation, "projection": g.projection, "abi_version": ABI_VERSION,
                "weight_read_grouping": [{"capacity": p["capacity"], "form": p["form"],
                                          **p["weight_read_grouping"]} for p in programs],
                "capacities": capacity_scratch(programs),
                "prefill_forms": {form: [p["capacity"] for p in programs if p["form"] == form]
                                  for form in FORMS[1:]}}
        manifest["files"] = {str(p.relative_to(stage)): {"bytes": p.stat().st_size, "sha256": digest(p)}
                             for p in sorted(stage.rglob("*")) if p.is_file()}
        manifest["runtime"] = {"library": "libcute_dsl_runtime.so", "sha256": digest(args.runtime)}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        verify(stage)
        shutil.copytree(stage, args.output)
        for path in [args.output, *args.output.rglob("*")]:
            path.chmod(0o755 if path.is_dir() or path.suffix == ".so" else 0o644)
    print(json.dumps({"package": str(args.output), "role": args.role, "layouts": layouts,
                      "capacities": capacities}), flush=True)


def verify(package: Path, *, revision: str | None = None, role: str | None = None,
           input_kind: str | None = None, layout: str | None = None,
           min_capacity: int | None = None) -> dict:
    manifest = json.loads((package / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"{package}: not an FP8 expert package")
    if revision is not None and manifest.get("sparkinfer_revision") != revision:
        raise ValueError(f"{package}: SparkInfer revision {manifest.get('sparkinfer_revision')} differs from {revision}")
    expected_role = "spark" if role == "expert" else role
    if expected_role is not None and manifest.get("role") != expected_role:
        raise ValueError(f"{package}: role {manifest.get('role')} differs from {expected_role}")
    layouts = manifest["layouts"]
    if min_capacity is not None and min_capacity < 1:
        raise ValueError("minimum capacity must be positive")
    if layout is not None and layout not in layouts:
        raise ValueError(f"{package}: no {layout} layout (built: {sorted(layouts)})")
    for name in ([layout] if layout is not None else layouts):
        info = layouts[name]
        if info.get("abi_version") != ABI_VERSION or info.get("activation") not in ACTIVATION or info.get("projection") not in PROJECTION:
            raise ValueError(f"{package}/{name}: missing or unknown expert activation/projection ABI")
        if info.get("weights") == "mxfp4" and info["activation"] != "silu_bf16_product":
            raise ValueError(f"{package}/{name}: MXFP4 supports only the pre-SiLU activation contract")
        if info.get("weights") in ("nvfp4", "bf16") and info.get("input") != "bf16":
            raise ValueError(f"{package}/{name}: {info['weights'].upper()} requires BF16 input")
        if input_kind is not None and info.get("input") != input_kind:
            raise ValueError(f"{package}/{name}: input {info.get('input')} differs from {input_kind}")
        if min_capacity is not None and not any(c["capacity"] >= min_capacity for c in info["capacities"]):
            raise ValueError(f"{package}/{name}: no capacity for {min_capacity} rows")
    files = {str(p.relative_to(package)) for p in package.rglob("*") if p.is_file()} - {"manifest.json"}
    if files != set(manifest["files"]):
        raise ValueError(f"{package}: file set differs from the manifest")
    for name, entry in manifest["files"].items():
        path = package / name
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError(f"{package}/{name}: size or digest mismatch")
    for layout in manifest["layouts"]:
        if f"{layout}/{LIBRARY}" not in files:
            raise ValueError(f"{package}: layout {layout} has no library")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("build")
    create.add_argument("--role", choices=sorted(ROLE_LAYOUTS), required=True)
    create.add_argument("--geometry", choices=("mimo", "mimop", "mimof", "glm", "glmf", "qwen4", "glm_nvfp4", "glmf_nvfp4",
                                               "qwen4_nvfp4", "glm_nvfp4a4", "glmf_nvfp4a4", "qwen4_nvfp4a4",
                                               "glmfdense_nvfp4", "glmfdense_nvfp4a4", "step5", "step5_bf16", "step5_nvfp4"),
                        required=True)
    create.add_argument("--layouts", help="comma list (default: tp4,tp2,tp6 where they split for spark, "
                        "tp1 for coordinator)")
    create.add_argument("--all-spark-counts", action="store_true",
                        help="build TP1..6 with at least one H128 block per rank, including unequal exact slices")
    create.add_argument("--capacities", default="1,16,80,256,1024,4096")
    create.add_argument("--exact-slices", action="store_true",
                        help="also build tp<n>-w<width> layouts: ranks own whole 128-row blocks, each stored at its "
                        "own width (no zero padding; the worker prefers them when every width is present)")
    create.add_argument("--input", choices=("wire", "bf16"),
                        help="expert input rows (NVFP4: bf16; otherwise wire for spark, bf16 for coordinator)")
    create.add_argument("--cross-sm121", action="store_true", help="build a Spark package on an SM120 host")
    create.add_argument("--build-dir", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--cxx", default="c++")
    create.add_argument("--cuda-include", type=Path, required=True)
    create.add_argument("--cuda-libdir", type=Path, required=True)
    create.add_argument("--runtime", type=Path, required=True)
    check = commands.add_parser("verify")
    check.add_argument("--package", type=Path, required=True)
    check.add_argument("--sparkinfer-revision")
    check.add_argument("--role", choices=("coordinator", "spark", "expert"))
    check.add_argument("--input", choices=("wire", "bf16"))
    check.add_argument("--layout")
    check.add_argument("--min-capacity", type=int)
    args = parser.parse_args()
    if args.command == "build":
        build(args)
    else:
        manifest = verify(args.package, revision=args.sparkinfer_revision, role=args.role,
                          input_kind=args.input, layout=args.layout, min_capacity=args.min_capacity)
        print(json.dumps({"verified": True, "role": manifest["role"], "geometry": manifest["geometry"],
                          "layouts": sorted(manifest["layouts"])}))


if __name__ == "__main__":
    main()

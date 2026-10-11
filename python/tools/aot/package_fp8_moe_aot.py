#!/usr/bin/env python3
"""Build or verify an exact FP8 routed-expert package (``fp8-<geometry>``).

The checkpoint's E4M3 experts with FP32 128x128 block scales (or MXFP4
experts, ``mimop``: packed E2M1 + UE8M0 per 32; or NVIDIA ModelOpt NVFP4
experts, ``<family>_nvfp4`` into ``fp8-<family>-nvfp4``: packed E2M1 + E4M3
per 16 + an FP32 alpha per expert, W4A16 on the grouped GEMV at every row
count) run as
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
        words: [0] ABI 1 (E4M3 + FP32 128x128 scales), 2 (MXFP4: packed E2M1 +
        UE8M0 per 32) or 3 (NVFP4: packed E2M1 + E4M3 per 16, each scale operand
        followed by the experts' FP32 weight_scale_2 and input_scale) or 4 (NVFP4,
        W4A4 stream route above the GEMV: the checkpoint's static input_scale
        quantizes the activations, block-scaled FP4 MMAs); packed FP4 slices are
        zero-padded to a 128-aligned width, [1] hidden, [2] slice, [3] experts, [4] top-k,
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
ABI = {"fp8": 1, "mxfp4": 2, "nvfp4": 3, "nvfp4a4": 4}
ROLE_COMPUTE = {"spark": (12, 1), "coordinator": (12, 0)}
# Spark ranks take the FP8 K32 wire rows of the expert protocol; coordinator
# experts take the BF16 activations directly (no input quantization at all).
ROLE_INPUT = {"spark": "wire", "coordinator": "bf16"}
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
    words = ", ".join(f"{w}u" for w in info)
    return f"""// Generated by package_fp8_moe_aot.py: exact FP8 routed-expert programs.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
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
struct Context {{
  cudaLibrary_t libraries[kCount];
  uint32_t form;
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
  for (uint32_t i = 0; i < kCount; ++i)
    if (ctx->libraries[i]) cudaLibraryUnload(ctx->libraries[i]);
  delete ctx;
}}

extern "C" int32_t cuteafd_fp8moe_create(void** context) {{
  if (!context) return cudaErrorInvalidValue;
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  auto* ctx = new Context{{}};
  for (uint32_t i = 0; i < kCount; ++i) {{
    cudaLibrary_t* library = &ctx->libraries[i];
    void* init_args[] = {{&library, &status}};
    kPrograms[i].initialize(init_args);
    if (status != cudaSuccess) {{ cuteafd_fp8moe_destroy(ctx); return status; }}
    int32_t device_id = device;
    void* load_args[] = {{&library, &device_id, &status}};
    kPrograms[i].load(load_args);
    if (status != cudaSuccess) {{ cuteafd_fp8moe_destroy(ctx); return status; }}
  }}
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
    if (!pointers[i]) return cudaErrorInvalidValue;
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


def exact_widths(intermediate: int, tp: int) -> list[int]:
    """Stored widths of TP ranks owning whole 128-row blocks exactly (TP6 of
    2048: 384, 384, 384, 384, 256, 256), as the worker's exact layouts expect."""
    blocks = intermediate // 128
    if intermediate % 128 or blocks < tp:
        return []
    return [(blocks // tp + (rank < blocks % tp)) * 128 for rank in range(tp)]


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
    g = base.with_tp(tp)
    if width:
        if int(width) not in exact_widths(base.intermediate, tp):
            raise SystemExit(f"{layout}: no rank of TP{tp} owns {width} rows of {base.intermediate} exactly")
        g = with_width(g, int(width))
    return tp, g


def info_words(g, capacities: list[int], wire: bool) -> list[int]:
    limit = struct.unpack("<I", struct.pack("<f", float(g.swiglu_limit)))[0]
    words = [ABI[g.kind], g.hidden, g.slice, g.experts, g.top_k, g.intermediate, g.tp, 7 if wire else 1, limit,
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
            layouts = ["tp2", "tp4"]
    elif base.weights == "nvfp4" and not args.layouts:
        layouts = [l for l in NVFP4_ROLE_LAYOUTS[args.role]
                   if base.intermediate // 16 >= int(l.removeprefix("tp"))]
    # Default Spark layouts keep the TP degrees that split the intermediate into
    # whole 128-row blocks (Qwen 3.8 Flash Next's 640 splits into none), and
    # TP6 of any intermediate of at least six blocks: uneven whole blocks
    # zero-padded to the widest (2048: 3, 3, 3, 3, 2, 2 blocks stored as 384).
    elif not args.layouts:
        def servable(tp: int) -> bool:
            blocks = base.intermediate // 128
            return base.intermediate % (128 * tp) == 0 or (tp == 6 and blocks >= 6)

        layouts = [l for l in layouts if servable(int(l.removeprefix("tp")))]
        if not layouts:
            raise SystemExit(f"{args.geometry}: intermediate {base.intermediate} has no default {args.role} "
                             "TP layout of whole 128-row blocks; pass --layouts")
    if args.geometry.startswith("glmfdense") and not args.layouts:
        layouts = ["tp1"]  # One always-selected dense expert is not an RTX TP2 backend.
    if args.exact_slices:
        # Exact layouts beside each padded TP layout whose ranks would store padding.
        for layout in list(layouts):
            tp = int(layout.partition("-w")[0].removeprefix("tp"))
            widths = sorted(set(exact_widths(base.intermediate, tp)), reverse=True)
            if tp > 1 and widths and base.with_tp(tp).slice * tp > base.intermediate:
                layouts += [f"tp{tp}-w{w}" for w in widths if f"tp{tp}-w{w}" not in layouts]
    wire = (args.input or ROLE_INPUT[args.role]) == "wire"
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
                                 "scratch": fp8_moe_scratch_bytes(g, "auto", capacity, wire, form)})
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
                "weights": g.weights,
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
                                               "glmfdense_nvfp4", "glmfdense_nvfp4a4"),
                        required=True)
    create.add_argument("--layouts", help="comma list (default: tp4,tp2,tp6 where they split for spark, "
                        "tp1 for coordinator)")
    create.add_argument("--capacities", default="1,16,80,256,1024,4096")
    create.add_argument("--exact-slices", action="store_true",
                        help="also build tp<n>-w<width> layouts: ranks own whole 128-row blocks, each stored at its "
                        "own width (no zero padding; the worker prefers them when every width is present)")
    create.add_argument("--input", choices=("wire", "bf16"),
                        help="expert input rows (default: wire for spark, bf16 for coordinator)")
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

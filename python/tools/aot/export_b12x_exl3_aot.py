#!/usr/bin/env python3
"""Export mixed EXL3 compute/epilogue objects and exact native buffer metadata.

Packed exports include native route preparation. This export does not claim
that an end-to-end native serving backend is available.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1] / _d) for _d in ("lib",)]  # sibling tool dirs

import argparse
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import re

import _pinned_sparkinfer


def write_bridge(output: Path, manifest: dict) -> None:
    """Generate a small owned-device C bridge; reject unfamiliar exported types."""
    if (manifest['blocks_per_sm'] not in (1, 2)
            or not 1 <= manifest['sms'] <= (2**31 - 1) // manifest['blocks_per_sm']):
        raise ValueError('invalid EXL3 cooperative grid capacity')
    # Mixed Trellis owns a complete K reduction per MN tile; a smaller grid
    # changes CTA assignment, not reduction order. Barriers count actual grid_x.
    # Barrier offsets and scratch use the declared workspace capacity, not the
    # export card. A smaller SM120 build can therefore use every PRO 6000 SM.
    # CuTe AOT stores kernel handles in globals inside each loaded DSO. Every
    # context must retain the same CUDA libraries; recreating them per lane or
    # device replaces those globals and loses another device's launch attributes.
    lines = ['#include <new>', '#include <mutex>', '#include "v41_exl3_core.h"', '#include "v41_exl3_sum.h"',
        'struct Context { int device; int32_t grid_cap; };',
        'struct Modules { std::mutex mutex; unsigned users = 0; cuteafd_v41_exl3_core_Kernel_Module_t core{}; cuteafd_v41_exl3_sum_Kernel_Module_t sum{}; };',
        'static Modules modules;',
        'static void unload_modules() { if (modules.sum.module) cudaLibraryUnload(modules.sum.module); if (modules.core.module) cudaLibraryUnload(modules.core.module); modules.sum.module = nullptr; modules.core.module = nullptr; }']
    for entry in manifest['objects']:
        label = entry['label']; role = label.rsplit('_', 1)[1]
        lines += [f'static int load_{role}(int device) {{',
            f'cudaLibrary_t* library = &modules.{role}.module; cudaError_t status = cudaSuccess;',
            'if (!*library) {',
            'struct { cudaLibrary_t** library; cudaError_t* status; } init{&library, &status};',
            f'_mlir_cuteafd_{label}_cuda_init(reinterpret_cast<void**>(&init));',
            'if (status != cudaSuccess) return int(status); }',
            'struct { cudaLibrary_t** library; int32_t* device; cudaError_t* status; } load{&library, &device, &status};',
            f'_mlir_cuteafd_{label}_cuda_load_to_device(reinterpret_cast<void**>(&load));',
            'return int(status); }']
    lines += ['extern "C" int cuteafd_exl3_create(void** out) {',
        'if (!out) return int(cudaErrorInvalidValue); *out = nullptr;',
        'Context* ctx = new(std::nothrow) Context; if (!ctx) return int(cudaErrorMemoryAllocation);',
        'cudaError_t status = cudaGetDevice(&ctx->device); int major = 0, minor = 0, sms = 0;',
        'if (status == cudaSuccess) status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, ctx->device);',
        'if (status == cudaSuccess) status = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, ctx->device);',
        'if (status == cudaSuccess) status = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, ctx->device);',
        'if (status != cudaSuccess) { delete ctx; return int(status); }',
        f'if (major != {manifest["compute"][0]} || minor != {manifest["compute"][1]} || sms < 1 || sms > {manifest["sms"]}) {{ delete ctx; return int(cudaErrorInvalidDevice); }}',
        f'ctx->grid_cap = sms * {manifest["blocks_per_sm"]};',
        'std::lock_guard<std::mutex> lock(modules.mutex);',
        'int error = load_core(ctx->device); if (!error) error = load_sum(ctx->device);',
        'if (error) { if (!modules.users) unload_modules(); delete ctx; return error; }',
        '++modules.users; *out = ctx; return 0; }',
        'extern "C" void cuteafd_exl3_destroy(void* opaque) { auto* ctx = static_cast<Context*>(opaque); if (!ctx) return; std::lock_guard<std::mutex> lock(modules.mutex); if (--modules.users == 0) unload_modules(); delete ctx; }']
    for entry in manifest['objects']:
        role = entry['label'].rsplit('_', 1)[1]
        args = [f'&modules.{role}']; declarations = []; checks = []; pointers = []; scalars = []
        for parameter in entry['parameters'][1:]:
            match = re.fullmatch(r'(.+?)(\w+)', parameter)
            if match is None: raise ValueError(f'unrecognized parameter: {parameter}')
            type_name, name = match[1].strip(), match[2]
            if type_name == 'cudaStream_t':
                args.append('static_cast<cudaStream_t>(stream)')
            elif type_name == 'int32_t':
                index = len(scalars); scalars.append(name); args.append(f's[{index}]')
                if name == 'active_m': checks.append(f'if (s[{index}] < 1 || s[{index}] > {manifest["capacity"]}) return int(cudaErrorInvalidValue);')
                if name == 'grid_x':
                    checks.append(f'if (s[{index}] < 1) return int(cudaErrorInvalidValue);')
                    # Clamp a local argument, never mutate caller-owned launch tables.
                    args[-1] = f'(s[{index}] < ctx->grid_cap ? s[{index}] : ctx->grid_cap)'
            elif type_name == 'void *' or re.fullmatch(r'cuteafd_v41_exl3_\w+_Tensor_\w+_t \*', type_name):
                index = len(pointers); pointers.append(name)
                checks.append(f'if (!p[{index}]) return int(cudaErrorInvalidValue);')
                if type_name == 'void *': args.append(f'p[{index}]')
                else:
                    tensor_type = type_name[:-1].strip()
                    header = (output / (entry['label'] + '.h')).read_text()
                    if not re.search(r'typedef struct\s*\{\s*void\s*\*data;\s*\}\s*' + re.escape(tensor_type) + r';', header):
                        raise ValueError(f'unsupported tensor descriptor {tensor_type}')
                    declarations.append(f'{tensor_type} {name}{{p[{index}]}};'); args.append('&' + name)
            else: raise ValueError(f'unsupported native parameter {parameter}')
        entry['pointer_slots'] = pointers; entry['scalar_slots'] = scalars
        lines += [f'extern "C" int cuteafd_exl3_{role}(void* opaque, void* const* p, const int32_t* s, void* stream) {{',
            'if (!opaque || !p || !s) return int(cudaErrorInvalidValue); auto* ctx = static_cast<Context*>(opaque);',
            'int device = -1; if (cudaGetDevice(&device) != cudaSuccess || device != ctx->device) return int(cudaErrorInvalidDevice);',
            *checks, *declarations, f'return {entry["wrapper"]}({", ".join(args)});', '}']
    core, epilogue = manifest['objects']
    info = [2, manifest['hidden'], manifest['intermediate'], manifest['experts'],
        manifest['capacity'], manifest['top_k'], len(manifest['bits']),
        len(core['pointer_slots']), len(core['scalar_slots']),
        len(epilogue['pointer_slots']), len(epilogue['scalar_slots']),
        *manifest['bits'], *([0] * (4 - len(manifest['bits']))),
        2 if manifest['output_dtype'] == 'bf16' else 4]
    boundary = manifest.get('paired_boundary')
    if boundary is None:
        lines += ['extern "C" int cuteafd_exl3_info(uint32_t* out, uint32_t words) {',
            'if (!out || words != 16) return int(cudaErrorInvalidValue);',
            'const uint32_t info[16] = {' + ','.join(map(str, info)) + '};',
            'for (int i = 0; i < 16; ++i) out[i] = info[i]; return 0; }']
    else:
        # Old consumers must fail instead of interpreting four-row descriptors
        # as the original three-row/disjoint contract.
        lines += ['extern "C" int cuteafd_exl3_info(uint32_t*, uint32_t) { return int(cudaErrorInvalidValue); }']
        paired_info = [3, *info[1:], 1 if boundary == 'first' else 2, 4]
        lines += ['extern "C" int cuteafd_exl3_paired_info(uint32_t* out, uint32_t words) {',
            'if (!out || words != 18) return int(cudaErrorInvalidValue);',
            'const uint32_t info[18] = {' + ','.join(map(str, paired_info)) + '};',
            'for (int i = 0; i < 18; ++i) out[i] = info[i]; return 0; }']
    (output / 'v41_exl3_bridge.cc').write_text('\n'.join(lines) + '\n')


def export(output: Path, intermediate: int, experts: int, capacity: int,
           bits: tuple[int, ...], routing: str, topk: int = 6, output_dtype: str = "bf16",
           blocks_per_sm: int | None = None, paired_boundary: str | None = None,
           tile: tuple[int, ...] | None = None, hidden: int = 5120, route_block: int = 8,
           token_major_rotation: bool = False, swiglu_limit: float | None = 10.0,
           fused_input_rotation: bool = False, warp_specialized: bool = False,
           wire_input: bool = False, ws_input_stages: int | None = None,
           ws_dynamic_tiles: bool = False, decode_schedule: str | None = None,
           compile_only: bool = False) -> dict:
    if paired_boundary not in (None, "first", "last"):
        raise ValueError("paired boundary must be first, last, or None")
    if paired_boundary is not None and (intermediate != 640 or len(bits) != 2 or topk != 6
                                        or hidden != 5120):
        raise ValueError("paired export requires width 640, two tiers and top-k 6")
    if paired_boundary is not None and tile is not None:
        raise ValueError("paired exports keep the qualified tile policy; the offline "
                         "tile override is for disjoint layouts only")
    # The (fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n) tuple is b12x's own
    # geometry vocabulary: _projection_mixed_tile_config()/the compile path reject a
    # thread-count mismatch or a tile that does not fit the problem, so the pinned
    # planner stays the single authority on what a legal tile is.
    # Packed routes are grouped into M blocks of route_block rows per expert; each
    # block decodes its expert's Trellis tiles once, so larger blocks amortize the
    # decode over more rows once many rows share an expert (large prefill).
    if route_block not in (8, 16, 32, 64):
        raise ValueError("EXL3 route block must be 8, 16, 32 or 64 rows")
    if route_block != 8 and paired_boundary is not None:
        raise ValueError("paired exports keep the qualified 8-row route block")
    # Token-major input rotation reads each token's input block once for all of
    # its routes (bit-identical); it applies to two-tier packed-route exports.
    if token_major_rotation and (paired_boundary is not None or len(bits) != 2):
        raise ValueError("token-major rotation requires a disjoint two-tier export")
    # FC1 rotates the staged token rows in shared memory instead of reading
    # materialized per-route copies (bit-identical); wide packed blocks only.
    if fused_input_rotation and (paired_boundary is not None or len(bits) != 2 or route_block < 16):
        raise ValueError("fused input rotation requires a disjoint two-tier export with 16+ row blocks")
    # Warp-specialized prefill: producer warps stream weights and input rows and
    # rotate FC1 inputs; consumer warps decode and multiply (bit-identical to the
    # cooperative kernel). FC1, SwiGLU and FC2 become three launches of the core.
    if warp_specialized and (paired_boundary is not None or len(bits) != 2 or route_block < 16
                             or fused_input_rotation or token_major_rotation):
        raise ValueError("warp-specialized prefill requires a disjoint two-tier export with 16+ "
                         "row blocks and brings its own input rotation")
    # Wire input: the warp-specialized FC1 producers read the E4M3 + UE8M0 K32
    # wire rows ([H] E4M3 then [H/32] UE8M0 bytes per row) and widen them in
    # shared memory to the FP16 values the BF16 path gets from the decoded rows
    # (bit-identical), so the worker skips its BF16 decode pass.
    if wire_input and (not warp_specialized or hidden % 512):
        raise ValueError("wire input requires the warp-specialized kernel and hidden % 512 == 0")
    # Warp-specialized input-row ring depth (bit-identical; latency hiding only).
    if ws_input_stages is not None and (not warp_specialized or ws_input_stages not in range(2, 7)):
        raise ValueError("input stages (2..6) apply to the warp-specialized kernel only")
    # Dynamic tile claims (bit-identical): CTAs take tiles from a counter in the
    # zero-on-create workspace, so an expert's route blocks start together and
    # share its weight stream through L2.
    if ws_dynamic_tiles and not warp_specialized:
        raise ValueError("dynamic tile claims apply to the warp-specialized kernel only")
    if output_dtype not in ("bf16", "fp32"):
        raise ValueError("EXL3 output must be bf16 or fp32")
    # Decode schedule (b12x parse_decode_schedule: a preset such as gb10, or
    # l2/pf1/pf2/pdl options): when and with which L2 policy the cooperative
    # kernel fetches weight words. Bit-identical to the default schedule; the
    # manifest records the canonical options, and only when one is set.
    if decode_schedule is not None and (paired_boundary is not None or warp_specialized
                                        or len(bits) != 2):
        raise ValueError("a decode schedule applies to cooperative disjoint two-tier exports")
    # Disk-loaded B12x executors omit the compiler IR required by export_to_c.
    os.environ["B12X_COMPILE_DISK_CACHE"] = "0"
    import torch
    from b12x.moe._shared.kernels.w4a16.host import route_pack_capacity
    from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
        EXL3_AOT_MAX_SMS, compile_mixed_trellis, compile_mixed_trellis3, make_mixed_trellis_buffers,
    )
    from b12x.moe._shared.kernels.w4a16.mixed_trellis4 import compile_mixed_trellis4
    from b12x.moe.fused_moe._impl import (
        _projection_mixed_direct_topk_routes, _projection_mixed_tile_config,
    )

    if len(bits) not in (2, 3, 4) or len(set(bits)) != len(bits) or any(b not in range(2, 6) for b in bits):
        raise ValueError("export requires two to four distinct K2..K5 decoder tiers")
    if blocks_per_sm is not None and len(bits) != 2:
        raise ValueError("explicit residency is currently supported only for two-tier exports")
    # swiglu_limit None is b12x's unclamped SwiGLU (GLM); DeepSeek clamps at 10.
    if swiglu_limit is not None and not swiglu_limit > 0:
        raise ValueError("SwiGLU limit must be positive or None (no clamp)")
    if not 1 <= capacity <= 4096 or not 1 <= topk <= 16 or experts < topk or experts > 512:
        raise ValueError("invalid EXL3 capacity, top-k or expert count")
    # Whole H128 rotation blocks on both projection axes.
    if hidden < 128 or hidden % 128 or intermediate < 128 or intermediate % 128:
        raise ValueError("EXL3 hidden and intermediate must be positive multiples of 128")
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) not in ((12, 0), (12, 1)):
        raise ValueError("V4.1 export requires native SM120 or SM121")
    workspace_sms = EXL3_AOT_MAX_SMS[(props.major, props.minor)]
    if props.multi_processor_count > workspace_sms:
        raise ValueError("EXL3 device exceeds the declared AOT workspace SM capacity")
    direct = _projection_mixed_direct_topk_routes(capacity, topk, direct_exl3=len(bits) == 2)
    if routing != "auto":
        direct = routing == "direct"
    if direct and len(bits) != 2:
        raise ValueError("three/four-tier export requires packed routing")
    block_m = 8 if direct else route_block
    route_slots = capacity * topk if direct else route_pack_capacity(capacity * topk, block_m, experts, topk=topk)[1]
    route_blocks = route_slots if direct else (route_slots + block_m - 1) // block_m
    options = dict(size_m=capacity, hidden_size=hidden, intermediate_size=intermediate,
        tier0_num_experts=experts, tier1_num_experts=experts, top_k=topk,
        route_num_experts=experts, max_m_blocks=route_blocks,
        sms=props.multi_processor_count, workspace_sms=workspace_sms,
        max_shared_mem=props.shared_memory_per_block_optin,
        force_tile_config=_projection_mixed_tile_config(tile, hidden_size=hidden,
            intermediate_size=intermediate, token_count=capacity, direct_topk_routes=direct),
        tier0_bits=bits[0], tier1_bits=bits[1], trellis_codebook="mcg", swiglu_limit=swiglu_limit,
        moe_block_size=block_m, rotation_input_dtype="bf16", full_rotation_output_dtype=output_dtype,
        route_ids_dtype=torch.int32)
    if paired_boundary is not None:
        options['paired_boundary'] = paired_boundary
    if len(bits) == 2:
        rotation = {"token_major_rotation": True} if token_major_rotation and not direct else {}
        if fused_input_rotation and not direct:
            rotation = {"fused_input_rotation": True}
        if warp_specialized and not direct:
            rotation = {"warp_specialized": True}
            if wire_input:
                rotation["input_format"] = "e4m3_k32"
            if ws_input_stages is not None:
                rotation["ws_input_stages"] = ws_input_stages
            if ws_dynamic_tiles:
                rotation["ws_dynamic_tiles"] = True
        schedule = {} if decode_schedule is None else {"decode_schedule": decode_schedule}
        launch = compile_mixed_trellis(**options, direct_topk_routes=direct,
                                      force_blocks_per_sm=blocks_per_sm, **rotation, **schedule)
    elif len(bits) == 3:
        launch = compile_mixed_trellis3(**options, tier2_num_experts=experts, tier2_bits=bits[2])
    else:
        launch = compile_mixed_trellis4(**options, tier2_num_experts=experts, tier2_bits=bits[2],
            tier3_num_experts=experts, tier3_bits=bits[3])
    output.mkdir(parents=True, exist_ok=True)
    objects = []
    for label, compiled in [("v41_exl3_core", launch.compiled), ("v41_exl3_sum", launch.topk_sum.compiled)]:
        compiled.export_to_c(str(output), label, "cuteafd_" + label)
        header = (output / (label + ".h")).read_text()
        wrapper = re.search(r"static inline int32_t (cute_dsl_\w+_wrapper)\((.*?)\) \{", header, re.S)
        if wrapper is None:
            raise ValueError(f"missing native wrapper for {label}")
        objects.append({"label": label, "wrapper": wrapper[1],
            "parameters": [p.strip() for p in wrapper[2].split(',')],
            "object_sha256": hashlib.sha256((output / (label + '.o')).read_bytes()).hexdigest(),
            "header_sha256": hashlib.sha256(header.encode()).hexdigest()})
    # Compile-only qualification needs the same shapes/aliasing, not CUDA storage.
    buffers = make_mixed_trellis_buffers(launch,
        device=torch.device("meta") if compile_only else torch.device("cuda", 0),
        sms=props.multi_processor_count)
    lut_bytes = launch.trellis_lut.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    (output / 'trellis_lut.bin').write_bytes(lut_bytes)
    layouts = {}
    owners = {}
    for field in fields(buffers):
        value = getattr(buffers, field.name)
        address = id(value) if compile_only else value.untyped_storage().data_ptr()
        owner = owners.setdefault(address, field.name)
        layouts[field.name] = {"shape": list(value.shape), "dtype": str(value.dtype),
            "bytes": value.numel() * value.element_size(), "allocation": owner,
            "zero_on_create": field.name == "workspace"}
    # The mixed executor's Python buffers cover exact live-capacity routes,
    # while the precompiled route packer rounds token capacity to its bucket.
    # Its initialization kernels write that entire bucket, including padding.
    # Publish the canonical packer's larger metadata allocations; compute data
    # buffers still cover only the actual token capacity.
    if not direct:
        for name, count in (("packed_route_indices", route_slots), ("block_expert_ids", route_blocks)):
            spec = layouts[name]
            if spec['allocation'] != name or spec['dtype'] != 'torch.int32' or len(spec['shape']) != 1:
                raise ValueError(f'unexpected route metadata layout: {name}')
            if spec['bytes'] > count * 4:
                raise ValueError(f'route metadata exceeds canonical capacity: {name}')
            spec.update(shape=[count], bytes=count * 4)
    manifest = {"schema": "cuteafd.v41-exl3-aot.v1", "sparkinfer_revision": _pinned_sparkinfer.REVISION,
        "gpu": props.name, "compute": [props.major, props.minor], "sms": workspace_sms,
        "physical_sms": props.multi_processor_count,
        "hidden": hidden, "intermediate": intermediate, "experts": experts, "top_k": topk,
        "capacity": capacity, "output_dtype": output_dtype, "bits": list(bits), "swiglu_limit": swiglu_limit,
        "direct": direct, "route_slots": route_slots, "route_blocks": route_blocks,
        "tile": list(options['force_tile_config']), "blocks_per_sm": launch.blocks_per_sm,
        "shared_memory_bytes": launch.shared_memory_bytes, "buffers": layouts, "objects": objects,
        "unique_execution_buffer_bytes": sum(v['bytes'] for k,v in layouts.items() if v['allocation'] == k),
        "requires_route_preparation": not direct,
        "required_link_libraries": ["cudart", "cute_dsl_runtime"],
        "trellis_lut": {"file": "trellis_lut.bin", "bytes": len(lut_bytes),
            "sha256": hashlib.sha256(lut_bytes).hexdigest()},
        "native_execution_verified": False}
    if block_m != 8:
        # Recorded only when it differs, so 8-row manifests stay byte-identical.
        manifest["route_block"] = block_m
    if warp_specialized and not direct:
        manifest["warp_specialized"] = True
        if wire_input:
            manifest["input_format"] = "e4m3_k32"
        if ws_input_stages is not None:
            manifest["ws_input_stages"] = ws_input_stages
        if ws_dynamic_tiles:
            manifest["ws_dynamic_tiles"] = True
    elif fused_input_rotation and not direct:
        manifest["fused_input_rotation"] = True
    elif token_major_rotation and not direct:
        manifest["token_major_rotation"] = True
    if paired_boundary is not None:
        manifest.update(paired_boundary=paired_boundary, descriptor_rows=4, native_info_version=3)
    if getattr(launch, "decode_schedule", None) is not None:
        manifest["decode_schedule"] = launch.decode_schedule
    write_bridge(output, manifest)
    if not direct:
        from export_b12x_exl3_routes_aot import export as export_routes
        route_manifest = export_routes(output / 'routes', capacity, experts, topk, block_m)
        for name in ('packed_route_indices', 'block_expert_ids', 'packed_route_count', 'expert_offsets', 'expert_counts'):
            if route_manifest['buffers'][name]['bytes'] > layouts[name]['bytes']:
                raise ValueError(f'mixed execution buffer {name} cannot hold route preparation')
        manifest['route_preparation'] = {
            'manifest': 'routes/v41_exl3_routes.json',
            'sha256': hashlib.sha256((output / 'routes/v41_exl3_routes.json').read_bytes()).hexdigest(),
        }
    (output / "v41_exl3.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ['capacity','intermediate','bits','direct','unique_execution_buffer_bytes']}), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--intermediate", type=int, required=True)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--routing", choices=("auto", "direct", "packed"), default="auto")
    parser.add_argument("--topk", type=int, choices=range(1, 17), metavar="1..16", default=6)
    parser.add_argument("--swiglu-limit", default="10",
                        help="SwiGLU clamp limit, or none for an unclamped SwiGLU (GLM)")
    parser.add_argument("--output-dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--blocks-per-sm", type=int, choices=(1, 2), help="Offline residency override; default uses B12x policy")
    parser.add_argument("--paired-boundary", choices=("first", "last"), help="Candidate TP4 ownership-aware layout")
    parser.add_argument("--route-block", type=int, choices=(8, 16, 32, 64), default=8,
                        help="Packed-route M block (rows per expert block); direct routes ignore it")
    parser.add_argument("--fused-input-rotation", action="store_true",
                        help="FC1 rotates staged token rows itself (packed 16+ row blocks)")
    parser.add_argument("--warp-specialized", action="store_true",
                        help="Warp-specialized prefill kernels (packed 16+ row blocks)")
    parser.add_argument("--wire-input", action="store_true",
                        help="Warp-specialized FC1 reads E4M3 + UE8M0 K32 wire rows (no BF16 decode)")
    parser.add_argument("--ws-input-stages", type=int, choices=range(2, 7),
                        help="Warp-specialized input-row ring depth (default: kernel policy)")
    parser.add_argument("--ws-dynamic-tiles", action="store_true",
                        help="Warp-specialized CTAs claim tiles from a workspace counter")
    parser.add_argument("--token-major-rotation", action="store_true",
                        help="Rotate each token's input once for all of its routes (packed routes)")
    parser.add_argument("--decode-schedule",
                        help="Cooperative-kernel decode schedule: a b12x preset (gb10) or "
                             "l2=,pf1=,pf2=,pdl= options; bit-identical to the default")
    parser.add_argument("--tile", help="Offline disjoint-layout tile override fc1_k,fc1_n,fc2_k,fc2_n "
                                       "(for example 64,256,64,256 or 128,128,128,128); default is the "
                                       "B12x per-capacity policy")
    args = parser.parse_args()
    export(args.output, args.intermediate, args.experts, args.capacity, tuple(args.bits), args.routing, args.topk, args.output_dtype, args.blocks_per_sm, args.paired_boundary,
           tile=None if args.tile is None else tuple(args.tile.split(",")), hidden=args.hidden,
           route_block=args.route_block, token_major_rotation=args.token_major_rotation,
           fused_input_rotation=args.fused_input_rotation,
           warp_specialized=args.warp_specialized, wire_input=args.wire_input,
           ws_input_stages=args.ws_input_stages, ws_dynamic_tiles=args.ws_dynamic_tiles,
           swiglu_limit=None if args.swiglu_limit.lower() == "none" else float(args.swiglu_limit),
           decode_schedule=args.decode_schedule)


if __name__ == "__main__":
    main()

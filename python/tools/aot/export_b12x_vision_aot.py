#!/usr/bin/env python3
"""Export shared BF16 vision operations by tower geometry, offline."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import sys

CAPACITY = 16384
HEADS = 16
MUSE_GEOMETRY = dict(hidden=1536, intermediate=8960, patch_width=1176,
                     merge=2, adapter_width=4096, output_width=8192)


def muse_transformer_names():
    """CMake and the exporter share one geometry/program inventory."""
    g = MUSE_GEOMETRY
    h, i, a = g["hidden"], g["intermediate"], g["adapter_width"]
    names = [f"layer_norm_h{h}", f"residual_h{h}", f"row_gather_h{h}",
             f"spatial_merge_h{h}_r{g['merge']}"]
    names += [f"biased_projection_h{w}_o{o}" for w, o in
              ((h, 3 * h), (h, h), (h, i), (i, h))]
    names += [f"projection_h{w}_o{o}" for w, o in
              ((g["patch_width"], h), (h * g["merge"] ** 2, a),
               (a, a), (a, g["output_width"]))]
    names += [f"gelu_h{w}" for w in dict.fromkeys((i, a))]
    return [f"vision_{name}" for name in names]


def perception_names(kind):
    inkling = kind in ("inkling2", "inkling2_mini")
    h,i,p,o = (1024,4096,1200,2304 if kind == "inkling2_mini" else 4608) if inkling else (1536,8960,588,4096)
    names = [f"pe_norm_h{h}",f"pe_scale_h{h}",f"pe_position_h{h}",f"pe_rotary_d{h//16}"]
    names += [f"biased_projection_h{w}_o{n}" for w,n in ((h,3*h),(h,h),(h,i),(i,h))]
    names += [f"projection_h{p}_o{h}"]
    if inkling:
        names += ["pe_rgb_p20","pe_merge_h1024","pe_patch_projection_h4096_o4096",f"projection_h4096_o{o}","gelu_h4096"]
    else:
        names += ["rgb_normalize","resize_coefficients_bilinear","resize_horizontal","resize_vertical",
            "pe_patches_p14","pe_conv_lower_h1536","pe_conv_lower_h3072","pe_quick_gelu_h8960",
            "biased_projection_h13824_o3072","biased_projection_h27648_o6144","pe_patch_projection_h6144_o4096"]
    return ["vision_"+name for name in names]


def tile_for_head_dim(head_dim):
    if head_dim <= 0 or head_dim > 128 or head_dim % 8:
        raise ValueError("vision head dimension must be an aligned value in 8..128")
    return (128, 128 if head_dim <= 64 else 64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--capability", type=int, choices=(120, 121))
    parser.add_argument("--group", choices=("legacy", "pointwise", "attention", "transformer", "perception", "tower"), default="legacy")
    parser.add_argument("--list-transformer-programs", action="store_true")
    parser.add_argument("--perception", choices=("inkling2","inkling2_mini","step5"))
    parser.add_argument("--list-perception-programs", action="store_true")
    parser.add_argument("--tower", choices=("kimi","minimax","hy4f","hy4"))
    parser.add_argument("--list-tower-programs", action="store_true")
    parser.add_argument("--head-dim", type=int, default=96)
    parser.add_argument("--hidden", type=int, default=1536)
    parser.add_argument("--bf16-products", action="store_true")
    parser.add_argument("--learned-position", action="store_true")
    parser.add_argument("--with-transformer", action="store_true")
    parser.add_argument("--bf16-before-gamma", action="store_true")
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--omit-program", action="append", default=[])
    args = parser.parse_args()
    if args.list_transformer_programs:
        print(";".join(muse_transformer_names()))
        return
    if args.list_perception_programs:
        if args.perception is None:parser.error("--list-perception-programs needs --perception")
        print(";".join(perception_names(args.perception)))
        return
    if args.list_tower_programs:
        if args.tower is None:parser.error("--list-tower-programs needs --tower")
        from vision_tower_programs import names
        print(";".join(names(args.tower)))
        return
    if args.output_dir is None or args.capability is None:
        parser.error("--output-dir and --capability are required for export")
    # No driver/runtime access or initialized device context is permitted here.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["B12X_COMPILE_DISK_CACHE"] = "0"
    os.environ["B12X_COMPILE_MEMORY_CACHE"] = "0"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
    import _pinned_sparkinfer
    from b12x._lib.aot_target import describe_target
    capability = (args.capability // 10, args.capability % 10)
    describe_target(capability)
    import torch
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr
    from cuda.bindings import driver as cuda
    from b12x._lib.compiler import compile as b12x_compile
    from b12x.attention._shared.contiguous.api import _VarlenAttentionForwardLaunch

    class VisionAttentionLaunch:
        def __init__(self, dim, tile):
            shape = (CAPACITY, HEADS, dim)
            self.launch = _VarlenAttentionForwardLaunch(q_shape=shape, k_shape=shape, v_shape=shape,
                cu_seqlens_q_shape=(2,), cu_seqlens_k_shape=(2,), dtype=torch.bfloat16,
                causal=False, window_size_left=-1, window_size_right=-1,
                has_attention_sink_bias=False, max_seqlen_q=CAPACITY, max_seqlen_k=CAPACITY,
                tile_m=tile[0], tile_n=tile[1])

        @cute.jit
        def __call__(self, q_ptr: cute.Pointer, k_ptr: cute.Pointer,
                     v_ptr: cute.Pointer, o_ptr: cute.Pointer, lse_ptr: cute.Pointer,
                     cu_seqlens_q_ptr: cute.Pointer, cu_seqlens_k_ptr: cute.Pointer,
                     attention_sink_bias_ptr: cute.Pointer, softmax_scale: cutlass.Float32,
                     l2_cache_size: cutlass.Int32, current_stream: cuda.CUstream):
            # The library launch annotates scale as Python float, unsupported by
            # export_to_c. This ABI-only adapter preserves its kernel arithmetic.
            # l2_cache_size is the launcher's runtime cudaDevAttrL2CacheSize.
            self.launch(q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, cu_seqlens_q_ptr,
                        cu_seqlens_k_ptr, attention_sink_bias_ptr, softmax_scale,
                        l2_cache_size, current_stream)

    destination = args.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    if args.group != "legacy":
        if args.group == "tower":
            if args.tower is None:parser.error("--group tower needs --tower")
            from vision_tower_programs import builders as tower_builders
            declarations = tower_builders(args.tower)
        elif args.group == "perception":
            if args.perception is None:parser.error("--group perception needs --perception")
            declarations = perception_builders(args.perception)
        else:
            declarations = builders(args.group, head_dim=args.head_dim, hidden=args.hidden,
                bf16_products=args.bf16_products, learned_position=args.learned_position,
                bf16_before_gamma=args.bf16_before_gamma, windowed=args.windowed)
        if args.with_transformer:
            if args.group != "pointwise":
                parser.error("--with-transformer requires --group pointwise")
            declarations.update(builders("transformer"))
        omitted = set(args.omit_program)
        unknown = omitted - {f"vision_{name}" for name in declarations}
        if unknown:parser.error(f"unknown omitted programs: {sorted(unknown)}")
        declarations = {name:build for name,build in declarations.items() if f"vision_{name}" not in omitted}
        export_programs(destination, "vision", declarations)
        if torch.cuda.is_initialized():
            raise RuntimeError("offline vision export initialized CUDA")
        return
    for dim in (64, 72):
        stem = f"vision_attention_d{dim}"
        symbol = "cuteafd_" + stem
        manifest_path = destination / (stem + ".json")
        manifest_path.unlink(missing_ok=True)
        tile = tile_for_head_dim(dim)
        shape = (CAPACITY, HEADS, dim)
        pointers = [make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16) for _ in range(4)]
        pointers += [make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=4)]
        pointers += [make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=4) for _ in range(2)]
        pointers += [make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=4)]
        program = b12x_compile(VisionAttentionLaunch(dim, tile), *pointers,
                               cutlass.Float32(1.0), cutlass.Int32(0), cuda.CUstream(0))
        program.export_to_c(str(destination), stem, symbol)
        header = (destination / (stem + ".h")).read_text()
        signature = re.search(r"static inline int32_t cute_dsl_" + symbol + r"_wrapper\(([^)]*)\)", header)
        if signature is None:
            raise ValueError("missing generated vision attention wrapper")
        # Every pointer is opaque; cumulative lengths delimit the live image rows.
        expected = [symbol + "_Kernel_Module_t *module"]
        pointers = ["q_ptr", "k_ptr", "v_ptr", "o_ptr", "lse_ptr", "cu_seqlens_q_ptr", "cu_seqlens_k_ptr", "attention_sink_bias_ptr"]
        expected += ["void *" + name for name in pointers]
        expected += ["float softmax_scale", "int32_t l2_cache_size", "cudaStream_t current_stream"]
        if re.sub(r"\s+", "", signature[1]) != re.sub(r"\s+", "", ",".join(expected)):
            raise ValueError("unexpected generated vision attention ABI: " + signature[1])
        artifacts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(destination.glob(stem + ".*")) if p.is_file() and p != manifest_path}
        manifest = dict(schema=1, sparkinfer_revision=_pinned_sparkinfer.REVISION,
            source_tree_sha256=_pinned_sparkinfer.LOCK_DATA["source_tree_sha256"],
            capability=list(capability), geometry=dict(heads=HEADS, head_dim=dim,
            capacity=CAPACITY, tile_m=tile[0], tile_n=tile[1], causal=False),
            dtype="BF16", live_rows="cu_seqlens [0,n]; host validates n <= capacity",
            lse_bytes=CAPACITY * HEADS * 4,
            abi=dict(pointers=pointers, f32=["softmax_scale"], i32=["l2_cache_size"],
                     stream="current_stream"),
            artifacts=artifacts)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(manifest_path, flush=True)
    if torch.cuda.is_initialized():
        raise RuntimeError("offline vision export initialized CUDA")


def perception_builders(kind):
    from b12x.integration.cuteafd import media,perception,perception_forward,image_resize
    from b12x.integration.cuteafd import perception_pixels
    inkling = kind in ("inkling2", "inkling2_mini")
    h,i,p,o = (1024,4096,1200,2304 if kind == "inkling2_mini" else 4608) if inkling else (1536,8960,588,4096)
    result = {
        f"pe_norm_h{h}":lambda:perception_forward.compile_layer_norm_aot(hidden=h,vendor_rsqrt=inkling),
        f"pe_scale_h{h}":lambda:perception.compile_scaled_residual_aot(width=h),
        f"pe_position_h{h}":lambda:perception_forward.compile_learned_position_aot(hidden=h,grid=32 if inkling else 52,class_token=inkling,torch_bilinear=not inkling),
        f"pe_rotary_d{h//16}":lambda:perception.compile_rotary_2d_aot(heads=16,dim=h//16,class_token=inkling),
    }
    for w,n in ((h,3*h),(h,h),(h,i),(i,h)):
        result[f"biased_projection_h{w}_o{n}"] = lambda w=w,n=n:media.compile_biased_projection_aot(width=w,outputs=n)
    result[f"projection_h{p}_o{h}"] = lambda:media.compile_patch_projection_aot(width=p,outputs=h)
    if inkling:
        result["pe_rgb_p20"] = lambda:perception_pixels.compile_pixel_patches_aot(patch=20)
        result["pe_merge_h1024"] = lambda:perception.compile_merge2_aot(width=1024,class_token=True)
        for w,n in ((4096,4096),(4096,o)):
            name = f"pe_patch_projection_h{w}_o{n}" if n == 4096 else f"projection_h{w}_o{n}"
            result[name] = lambda w=w,n=n:media.compile_patch_projection_aot(width=w,outputs=n)
        result["gelu_h4096"] = lambda:media.compile_gelu_aot(width=4096)
    else:
        result.update({"rgb_normalize":image_resize.compile_rgb_normalize_aot,
            "resize_coefficients_bilinear":lambda:image_resize.compile_resize_coefficients_aot(filter="bilinear"),
            "resize_horizontal":lambda:image_resize.compile_resize_pass_aot(vertical=False),
            "resize_vertical":lambda:image_resize.compile_resize_pass_aot(vertical=True),
            "pe_patches_p14":lambda:perception_forward.compile_normalized_patches_aot(patch=14),
            "pe_conv_lower_h1536":lambda:perception.compile_conv3_stride2_lower_aot(channels=1536),
            "pe_conv_lower_h3072":lambda:perception.compile_conv3_stride2_lower_aot(channels=3072),
            "pe_quick_gelu_h8960":lambda:perception.compile_quick_gelu_aot(width=8960)})
        for w,n in ((13824,3072),(27648,6144)):
            result[f"biased_projection_h{w}_o{n}"] = lambda w=w,n=n:media.compile_biased_projection_aot(width=w,outputs=n)
        result["pe_patch_projection_h6144_o4096"] = lambda:media.compile_patch_projection_aot(width=6144,outputs=4096)
    if ["vision_"+name for name in result] != perception_names(kind):
        raise ValueError("PE declaration inventory differs from CMake")
    return result


def builders(group, *, head_dim=96, hidden=1536, bf16_products=False,
             learned_position=False, bf16_before_gamma=False, windowed=False):
    """Static tower geometry only; image dimensions and rows remain launch data."""
    from b12x.integration.cuteafd import media
    if group == "transformer":
        result = media.vision_transformer_programs(**MUSE_GEOMETRY)
        if [f"vision_{name}" for name in result] != muse_transformer_names():
            raise ValueError("Muse transformer declaration inventory differs from CMake")
        return result
    if group == "attention":
        return {f"attention_d{head_dim}_{'windows' if windowed else 'full'}":
                lambda: media.compile_vision_attention_aot(dim=head_dim, windowed=windowed)}
    if group != "pointwise":
        raise ValueError(f"unknown vision group: {group}")
    return media.vision_programs(dim=head_dim, hidden=hidden, bf16_products=bf16_products,
        learned_position=learned_position, bf16_before_gamma=bf16_before_gamma)


def export_programs(destination, prefix, declarations):
    """Emit the shared raw-pointer ABI and validate every generated signature."""
    from b12x.integration.cuteafd import exportable_compilation, validate_exported_header
    records = []
    for name, build in declarations.items():
        stem = f"{prefix}_{name}"
        with exportable_compilation():
            program = build()
        symbol = "cuteafd_" + stem
        program.export_to_c(str(destination), stem, symbol)
        validate_exported_header(program, destination / (stem + ".h"), symbol)
        records.append(dict(name=stem, geometry=dict(program.geometry), abi=program.abi,
            artifacts={stem + ext: hashlib.sha256((destination / (stem + ext)).read_bytes()).hexdigest()
                       for ext in (".o", ".h")}))
        print(stem, flush=True)
    (destination / (prefix + ".json")).write_text(json.dumps(dict(schema=1, programs=records), indent=2) + "\n")



if __name__ == "__main__":
    main()

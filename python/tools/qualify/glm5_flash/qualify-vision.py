#!/usr/bin/env python3
"""GLM Flash G2/G3, inside the matching architecture's CUDA container.

Only model.visual.* payloads are read. Compare pinned official FP32 stages and
an official BF16 floor; never hide strict verdicts. G3 repeats interleaved shapes
and uses the native allocation ledger, not GB10 free-memory fluctuations.
"""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

_common_path = Path(__file__).resolve().parents[1] / "mimo_v2/qualify-vision.py"
_common_spec = importlib.util.spec_from_file_location("_vision_metrics", _common_path)
common = importlib.util.module_from_spec(_common_spec)
_common_spec.loader.exec_module(common)
Block, Ledger, OBSERVER = common.Block, common.Ledger, common.OBSERVER
NO_OFFSET = (1 << 64) - 1


Spec = common.Spec


def read_visual(snapshot):
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    tensors = {}
    for shard in sorted({v for k, v in index.items() if k.startswith("model.visual.")}):
        if Path(shard).is_absolute() or any(p in ("", ".", "..") for p in shard.split("/")):
            raise ValueError("tower shard escapes snapshot")
        with (snapshot / shard).open("rb") as source:
            size = int.from_bytes(source.read(8), "little")
            header = json.loads(source.read(size))
            for name, entry in header.items():
                if not name.startswith("model.visual."):
                    continue
                if entry["dtype"] != "BF16" or name in tensors:
                    raise ValueError(f"unsupported or duplicate tower tensor: {name}")
                start, end = entry["data_offsets"]
                source.seek(8 + size + start)
                raw = source.read(end - start)
                if len(raw) != end - start:
                    raise ValueError(f"truncated tower tensor: {name}")
                tensors[name.removeprefix("model.visual.")] = np.frombuffer(raw, "<u2").reshape(entry["shape"]).copy()
    return tensors


def pack_spec(config, tensors):
    vision = config["vision_config"]
    expected = {"depth": 24, "hidden_size": 1024, "intermediate_size": 4096,
        "num_heads": 16, "out_hidden_size": 4096, "patch_size": 14,
        "temporal_patch_size": 2, "spatial_merge_size": 2,
        "projection_intermediate_size": 10240, "in_channels": 3}
    if (config["model_type"] != "glm5_next" or config["text_config"]["hidden_size"] != 4096
            or any(vision[k] != v for k, v in expected.items())
            or vision["rms_norm_eps"] != 1e-5 or vision["swiglu_limit"] != 10
            or not vision["attention_bias"] or vision["hidden_act"] != "silu"):
        raise ValueError("unsupported official GLM vision geometry")
    spec = Spec(abi_version=2, reserved=3, max_tokens=4096, output_width=4096,
        hidden=1024, depth=24, heads=16, kv_heads=16, head_dim=64, intermediate=4096,
        patch_size=14, merger_width=4096, norm_eps=1e-5)
    for name in ("patch_bias", "pos_embed", "merger_norm_bias", "merger_fc1_bias", "merger_fc2_bias"):
        setattr(spec, name, NO_OFFSET)
    for name, count in (("merger_extra", 8), ("norm1_bias", spec.depth), ("norm2_bias", spec.depth), ("q_norm", spec.depth), ("k_norm", spec.depth)):
        for i in range(count):
            getattr(spec, name)[i] = NO_OFFSET
    chunks, mapped, size = [], set(), 0

    def put(value):
        nonlocal size
        gap = -size % 256
        chunks.append(bytes(gap))
        size += gap
        offset = size
        data = value.tobytes()
        chunks.append(data)
        size += len(data)
        return offset

    def get(name, shape, vector=False):
        value = tensors[name]
        if list(value.shape) != list(shape) or value.dtype != np.uint16:
            raise ValueError(f"{name}: expected BF16 {shape}, got {value.dtype} {value.shape}")
        mapped.add(name)
        return (value.astype(np.uint32) << 16).view(np.float32) if vector else value

    spec.patch = put(get("patch_embed.proj.weight", [1024, 3, 2, 14, 14]))
    spec.patch_bias = put(get("patch_embed.proj.bias", [1024], True))
    for i in range(24):
        block, prefix = spec.blocks[i], f"blocks.{i}."
        for field, name, shape, vector in (
            ("qkv", "attn.qkv.weight", [3072, 1024], False),
            ("qkv_bias", "attn.qkv.bias", [3072], True),
            ("proj", "attn.proj.weight", [1024, 1024], False),
            ("proj_bias", "attn.proj.bias", [1024], True),
            ("down", "mlp.down_proj.weight", [1024, 4096], False),
            ("down_bias", "mlp.down_proj.bias", [1024], True),
            ("norm1", "norm1.weight", [1024], True), ("norm2", "norm2.weight", [1024], True)):
            setattr(block, field, put(get(prefix + name, shape, vector)))
        block.gate_up = put(np.concatenate([get(prefix + f"mlp.{name}_proj.weight", [4096, 1024]) for name in ("gate", "up")]))
        block.gate_up_bias = put(np.concatenate([get(prefix + f"mlp.{name}_proj.bias", [4096], True) for name in ("gate", "up")]))
        block.key0_bias = NO_OFFSET
        spec.q_norm[i] = put(get(prefix + "attn.q_norm.weight", [64], True))
        spec.k_norm[i] = put(get(prefix + "attn.k_norm.weight", [64], True))
    spec.merger_norm = put(get("post_layernorm.weight", [1024], True))
    for i, name, shape, vector in (
        (0, "downsample.weight", [4096, 1024, 2, 2], False),
        (1, "downsample.bias", [4096], True),
        (2, "merger.proj.weight", [4096, 4096], False),
        (3, "merger.post_projection_norm.weight", [4096], True),
        (6, "merger.down_proj.weight", [4096, 10240], False)):
        spec.merger_extra[i] = put(get(name, shape, vector))
    spec.merger_norm_bias = put(get("merger.post_projection_norm.bias", [4096], True))
    spec.merger_extra[4] = put(np.concatenate([get(f"merger.{name}_proj.weight", [10240, 4096]) for name in ("gate", "up")]))
    if mapped != set(tensors):
        raise ValueError(f"unmapped official tower tensors: {sorted(set(tensors) - mapped)}")
    spec.inv_freq = put((1 / np.power(np.float32(10000), np.arange(16, dtype=np.float32) / 16)).astype(np.float32))
    spec.weight_bytes = size
    return spec, b"".join(chunks)


def normalization_lut(snapshot):
    document = json.loads((snapshot / "processor_config.json").read_text())
    processor = document["image_processor"]
    if (processor["patch_size"] != 14 or processor["temporal_patch_size"] != 2
            or processor["merge_size"] != 2 or not processor.get("do_rescale", True)):
        raise ValueError("unsupported GLM processor geometry or rescaling")
    mean = np.asarray(processor["image_mean"], np.float32)
    std = np.asarray(processor["image_std"], np.float32)
    if (mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or not (std > 0).all()):
        raise ValueError("invalid GLM normalization parameters")
    return np.ascontiguousarray((np.arange(256, dtype=np.float32)[None, :]
        * np.float32(processor.get("rescale_factor", 1 / 255)) - mean[:, None]) / std[:, None])


def fixture(tokens):
    gh, gw = {256: (16, 64), 1024: (32, 128), 4096: (64, 256)}[tokens]
    yy, xx = np.indices((gh * 14, gw * 14), dtype=np.uint32)
    rgb = np.stack([(xx * 7 + yy * 3) % 256, (xx // 8 + yy * 11) % 256,
        ((xx // 32) ^ (yy // 16)) * 31 % 256], axis=-1).astype(np.uint8)
    return gh, gw, np.ascontiguousarray(rgb)


def patches(rgb, lut):
    h, w, _ = rgb.shape
    gh, gw = h // 14, w // 14
    values = np.stack([lut[c, rgb[:, :, c]] for c in range(3)])
    values = values.reshape(3, gh // 2, 2, 14, gw // 2, 2, 14).transpose(1, 4, 2, 5, 0, 3, 6)
    return np.ascontiguousarray(np.repeat(values[:, :, :, :, :, None], 2, axis=5).reshape(gh * gw, 1176))


def reference_model(config, tensors):
    import torch
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
    from transformers.models.glm5_next import modeling_glm5_next as ref
    vision = Glm5NextVisionConfig(**config["vision_config"])
    vision._attn_implementation = "sdpa"
    model = ref.Glm5NextVisionModel(vision)
    state = {key: torch.from_numpy((value.astype(np.uint32) << 16).view(np.float32)) for key, value in tensors.items()}
    model.load_state_dict(state, strict=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return model.cuda().float().eval(), Path(ref.__file__)


def calibrated_metrics(measured, floor, stage):
    result = dict(measured, strict_pass=measured["pass"])
    # GLM's official BF16 output misses the MiMo absolute bar too; G4 gates LM effect.
    minimum_mean = floor["mean_cosine"] - 0.00005
    result["pass"] = (measured["relative_l2"] <= floor["relative_l2"] + 0.002
        and measured["mean_cosine"] >= minimum_mean
        and measured["worst_cosine"] >= floor["worst_cosine"] - 0.001)
    return result


def run(args):
    import torch
    import torch.nn.functional as F
    config = json.loads((args.snapshot / "config.json").read_text())
    tensors = read_visual(args.snapshot)
    spec, blob = pack_spec(config, tensors)
    library, cudart = C.CDLL(str(args.library)), C.CDLL("libcudart.so")
    library.cuteafd_vision_required.argtypes = [C.POINTER(Spec), C.POINTER(Ledger)]
    library.cuteafd_vision_create.argtypes = [C.POINTER(Spec), C.c_int32, C.c_uint64, C.POINTER(C.c_void_p)]
    library.cuteafd_vision_upload.argtypes = [C.c_void_p, C.c_uint64, C.c_void_p, C.c_uint64]
    library.cuteafd_vision_encode.argtypes = [C.c_void_p, C.c_void_p, C.c_uint64, C.c_void_p, C.c_int32, C.c_int32, C.c_void_p, C.c_uint64, OBSERVER, C.c_void_p]
    library.cuteafd_vision_get_ledger.argtypes = [C.c_void_p, C.POINTER(Ledger)]
    library.cuteafd_vision_destroy.argtypes = [C.c_void_p]
    cudart.cudaMemcpy.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int]
    cudart.cudaMemGetInfo.argtypes = [C.POINTER(C.c_size_t), C.POINTER(C.c_size_t)]

    def check(status):
        if status:
            raise RuntimeError(f"native GLM vision status {status}")

    lut = normalization_lut(args.snapshot)
    owner, required = C.c_void_p(), Ledger()
    check(library.cuteafd_vision_required(C.byref(spec), C.byref(required)))
    check(library.cuteafd_vision_create(C.byref(spec), 0, required.weights + required.scratch + required.blas_workspace, C.byref(owner)))
    original = F.scaled_dot_product_attention
    # Split independent query rows to bound official FP32 math scratch at 16K.
    def bounded(q, k, v, attn_mask=None, **kwargs):
        if attn_mask is not None or kwargs.get("is_causal", False):
            raise ValueError("GLM tower reference expects full noncausal attention")
        return torch.cat([original(q[:, :, start:start + 128], k, v, **kwargs)
            for start in range(0, q.shape[-2], 128)], dim=-2)
    F.scaled_dot_product_attention = bounded
    result = dict(checkpoint=str(args.snapshot), device=torch.cuda.get_device_name(0),
        sm=torch.cuda.get_device_capability(0), numerics=1, ledger=required.as_dict(),
        calibration_policy="glm-bf16-relative-all-stages/2",
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
        processor_sha256=hashlib.sha256((args.snapshot / "processor_config.json").read_bytes()).hexdigest(),
        g2={}, bf16_yardstick={}, g3={}, encode_ms={})
    try:
        check(library.cuteafd_vision_upload(owner, 0, blob, len(blob)))
        del blob
        model, source = reference_model(config, tensors) if not args.native_only else (None, None)
        if source:
            result["modeling_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        del tensors

        def encode(tokens, observe=False):
            gh, gw, rgb = fixture(tokens)
            output, stages, errors = np.empty((tokens, 4096), np.uint16), {}, []
            def callback(_ctx, stage, pointer, rows, width, column):
                try:
                    if stage == 26:
                        return 0
                    if column:
                        raise ValueError("GLM stage unexpectedly reordered columns")
                    value = np.empty((rows, width), np.float32)
                    check(cudart.cudaMemcpy(value.ctypes.data, pointer, value.nbytes, 2))
                    stages[stage] = value
                    return 0
                except BaseException as error:
                    errors.append(repr(error))
                    return 1
            observer = OBSERVER(callback) if observe else OBSERVER()
            started = time.perf_counter()
            check(library.cuteafd_vision_encode(owner, rgb.ctypes.data, rgb.nbytes, lut.ctypes.data,
                gh, gw, output.ctypes.data, output.nbytes, observer, None))
            if errors:
                raise RuntimeError(errors)
            return output, stages, (time.perf_counter() - started) * 1000

        saved = {}
        for tokens in args.tokens:
            encode(tokens)
            native, stages, _ = encode(tokens, model is not None)
            saved[tokens] = native
            result.setdefault("output_sha256", {})[str(tokens)] = hashlib.sha256(native.tobytes()).hexdigest()
            if model is not None:
                gh, gw, rgb = fixture(tokens)
                grid = torch.tensor([[1, gh, gw]], device="cuda")
                refs, hooks = {}, []
                def hook(stage):
                    def capture(_module, _inputs, value):
                        refs[stage] = value.detach().float().cpu().numpy()
                    return capture
                hooks.append(model.patch_embed.register_forward_hook(hook(0)))
                hooks.append(model.post_layernorm.register_forward_hook(hook(25)))
                for i, block in enumerate(model.blocks):
                    if (i + 1) % 4 == 0:
                        hooks.append(block.register_forward_hook(hook(i + 1)))
                try:
                    with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                        ref = model(torch.from_numpy(patches(rgb, lut)).cuda(), grid).pooler_output.float().cpu().numpy()
                    measured = {str(stage): common.metrics(value, refs[stage]) for stage, value in stages.items()}
                    measured["26"] = common.metrics((native.astype(np.uint32) << 16).view(np.float32), ref)
                    fp32_refs, refs = refs, {}
                    inv_freq = model.rotary_pos_emb.inv_freq.detach().clone()
                    model.bfloat16()
                    # Rotary angles are computed in FP32 by the official tower.
                    model.rotary_pos_emb.inv_freq = inv_freq
                    with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                        floor = model(torch.from_numpy(patches(rgb, lut)).cuda(), grid).pooler_output.float().cpu().numpy()
                    yardstick = {str(stage): common.metrics(value, fp32_refs[stage]) for stage, value in refs.items()}
                    yardstick["26"] = common.metrics(floor, ref)
                    result["bf16_yardstick"][str(tokens)] = yardstick
                    result["g2"][str(tokens)] = {stage: calibrated_metrics(value, yardstick[stage], stage) for stage, value in measured.items()}
                    model.float()
                    model.rotary_pos_emb.inv_freq.copy_(inv_freq)
                finally:
                    for handle in hooks:
                        handle.remove()
                print(json.dumps(dict(event="G2", tokens=tokens, stages=result["g2"][str(tokens)])), flush=True)
        initial, final = Ledger(), Ledger()
        check(library.cuteafd_vision_get_ledger(owner, C.byref(initial)))
        def memory():
            free, total = C.c_size_t(), C.c_size_t()
            check(cudart.cudaMemGetInfo(C.byref(free), C.byref(total)))
            return dict(free=free.value, total=total.value,
                        cuda_free_scope="raw cudaMemGetInfo; GB10 under-reports reclaimable page cache")
        before = memory()
        for repeat in range(3):
            for tokens in reversed(args.tokens) if repeat % 2 else args.tokens:
                value, _, elapsed = encode(tokens)
                result["encode_ms"].setdefault(str(tokens), []).append(elapsed)
                result["g3"][str(tokens)] = result["g3"].get(str(tokens), True) and bool(np.array_equal(value, saved[tokens]))
        check(library.cuteafd_vision_get_ledger(owner, C.byref(final)))
        result.update(common.allocation_metrics(initial, final, before, memory(), result["sm"]))
        result.update(initial_ledger=initial.as_dict(), final_ledger=final.as_dict(),
            g2_pass=bool(result["g2"]) and all(m["pass"] for stages in result["g2"].values() for m in stages.values()),
            g3_pass=bool(result["g3"]) and all(result["g3"].values()))
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return 0 if result["no_encode_device_allocation"] and result["g3_pass"] and (result["g2_pass"] or args.native_only) else 1
    finally:
        F.scaled_dot_product_attention = original
        check(library.cuteafd_vision_destroy(owner))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[256, 1024, 4096], choices=[256, 1024, 4096])
    parser.add_argument("--native-only", action="store_true")
    raise SystemExit(run(parser.parse_args()))

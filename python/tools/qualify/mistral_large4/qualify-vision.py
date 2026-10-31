#!/usr/bin/env python3
"""Mistral Large 4 Pixtral tower G2/G3, inside the matching architecture's CUDA container.

Only the tower, projector and the two marker embedding rows are read (official
consolidated tensors). The reference is the official transformers modules
(PixtralVisionModel + MistralLarge4MultiModalProjector) in BF16, the
checkpoint's dtype; an FP32 run of the same modules gives the BF16 floor, and
the native tower passes when it is within that floor (the shared calibrated
metrics). Marker rows of the span must equal tok_embeddings byte for byte.
G3 repeats interleaved shapes on the native allocation ledger.

Two halves, so a Spark container needs no Transformers: `--native-only
--save-native DIR` encodes on the device (ledger, markers, G3) and saves each
span; `--compare-native DIR` runs the official modules (CPU or CUDA) over the
same fixtures and scores the saved spans.
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
Spec, Ledger, OBSERVER = common.Spec, common.Ledger, common.OBSERVER
NO_OFFSET = (1 << 64) - 1
H, I, LM, DEPTH = 2048, 6144, 6144, 30
TOWER = ("vision_encoder.", "patch_merger.", "pre_mm_projector_norm.", "vision_language_adapter.")


def read_tower(snapshot):
    index = json.loads((snapshot / "consolidated.safetensors.index.json").read_text())["weight_map"]
    wanted = {k: v for k, v in index.items() if k.startswith(TOWER)}
    tensors = {}
    for shard in sorted(set(wanted.values())):
        if Path(shard).is_absolute() or any(p in ("", ".", "..") for p in shard.split("/")):
            raise ValueError("tower shard escapes snapshot")
        with (snapshot / shard).open("rb") as source:
            size = int.from_bytes(source.read(8), "little")
            header = json.loads(source.read(size))
            for name, entry in header.items():
                if name == "__metadata__" or not name.startswith(TOWER):
                    continue
                if entry["dtype"] != "BF16" or name in tensors:
                    raise ValueError(f"unsupported or duplicate tower tensor: {name}")
                start, end = entry["data_offsets"]
                source.seek(8 + size + start)
                raw = source.read(end - start)
                if len(raw) != end - start:
                    raise ValueError(f"truncated tower tensor: {name}")
                tensors[name] = np.frombuffer(raw, "<u2").reshape(entry["shape"]).copy()
    return tensors


def marker_rows(snapshot, ids):
    """The [IMG_BREAK]/[IMG_END] rows of the BF16 embedding table, read directly."""
    index = json.loads((snapshot / "consolidated.safetensors.index.json").read_text())["weight_map"]
    with (snapshot / index["tok_embeddings.weight"]).open("rb") as source:
        size = int.from_bytes(source.read(8), "little")
        entry = json.loads(source.read(size))["tok_embeddings.weight"]
        if entry["dtype"] != "BF16" or entry["shape"][1] != LM:
            raise ValueError("tok_embeddings must be BF16 [vocab, 6144]")
        rows = []
        for token in ids:
            source.seek(8 + size + entry["data_offsets"][0] + token * LM * 2)
            rows.append(np.frombuffer(source.read(LM * 2), "<u2").copy())
    return rows


def markers(snapshot):
    vision = json.loads((snapshot / "params.json").read_text())["vision_encoder"]
    return vision["image_token_id"], vision["image_break_token_id"], vision["image_end_token_id"]


def pack_spec(config, tensors, snapshot, merged=55 * 55):
    v = config["vision_config"]
    if (config["model_type"] != "mistral_large4" or v["model_type"] != "pixtral" or v["hidden_size"] != H
            or v["num_hidden_layers"] != DEPTH or v["num_attention_heads"] != 16 or v["head_dim"] != 128
            or v["intermediate_size"] != I or v["patch_size"] != 14 or v["hidden_act"] != "silu"
            or config["spatial_merge_size"] != 2 or config["projector_hidden_act"] != "gelu"
            or config["multimodal_projector_bias"] or config["text_config"]["hidden_size"] != LM):
        raise ValueError("unsupported official Mistral Large 4 vision geometry")
    spec = Spec(abi_version=2, reserved=4, max_tokens=merged, output_width=LM, hidden=H, depth=DEPTH,
        heads=16, kv_heads=16, head_dim=128, intermediate=I, patch_size=14, merger_width=4 * H, norm_eps=1e-5)
    for name in ("patch_bias", "pos_embed", "merger_norm_bias", "merger_fc1_bias", "merger_fc2_bias"):
        setattr(spec, name, NO_OFFSET)
    for name, count in (("merger_extra", 8), ("norm1_bias", DEPTH), ("norm2_bias", DEPTH), ("q_norm", DEPTH), ("k_norm", DEPTH)):
        for i in range(count):
            getattr(spec, name)[i] = NO_OFFSET
    chunks, mapped, size = [], set(), 0

    def put(value):
        nonlocal size
        gap = -size % 256
        chunks.append(bytes(gap))
        size += gap
        offset = size
        chunks.append(value.tobytes())
        size += value.nbytes
        return offset

    def get(name, shape):
        value = tensors[name]
        if list(value.shape) != list(shape) or value.dtype != np.uint16:
            raise ValueError(f"{name}: expected BF16 {shape}, got {value.dtype} {value.shape}")
        mapped.add(name)
        return value

    spec.patch = put(get("vision_encoder.patch_conv.weight", [H, 3, 14, 14]))
    spec.merger_extra[0] = put(get("vision_encoder.ln_pre.weight", [H]))
    for i in range(DEPTH):
        block, p = spec.blocks[i], f"vision_encoder.transformer.layers.{i}."
        block.qkv = put(np.concatenate([get(p + f"attention.{n}.weight", [H, H]) for n in ("wq", "wk", "wv")]))
        block.proj = put(get(p + "attention.wo.weight", [H, H]))
        block.gate_up = put(np.concatenate([get(p + f"feed_forward.{n}.weight", [I, H]) for n in ("w1", "w3")]))
        block.down = put(get(p + "feed_forward.w2.weight", [H, I]))
        block.norm1 = put(get(p + "attention_norm.weight", [H]))
        block.norm2 = put(get(p + "ffn_norm.weight", [H]))
        for field in ("qkv_bias", "proj_bias", "gate_up_bias", "down_bias", "key0_bias"):
            setattr(block, field, NO_OFFSET)
    spec.merger_norm = put(get("pre_mm_projector_norm.weight", [H]))
    spec.merger_extra[1] = put(get("patch_merger.merging_layer.weight", [H, 4 * H]))
    spec.merger_fc1 = put(get("vision_language_adapter.w_in.weight", [LM, H]))
    spec.merger_fc2 = put(get("vision_language_adapter.w_out.weight", [LM, LM]))
    if mapped != set(tensors):
        raise ValueError(f"unmapped official tower tensors: {sorted(set(tensors) - mapped)}")
    _, row_break, end = markers(snapshot)
    rows = marker_rows(snapshot, (row_break, end))
    spec.merger_extra[4], spec.merger_extra[5] = put(rows[0]), put(rows[1])
    base = 1 / np.power(np.float32(10000), np.arange(0, 128, 2, dtype=np.float32) / np.float32(128))
    spec.inv_freq = put(np.concatenate([base[0::2], base[1::2]]).astype(np.float32))
    spec.weight_bytes = size
    return spec, b"".join(chunks), rows


def planner_scratch_bytes(merged=55 * 55):
    """cuteafd-loader `plan::layout::mistral_scratch_bytes`: the planner's tower charge."""
    n, span, mib = merged * 4, merged + 55, 1 << 20
    parts = [n*2048*4, n*2048*4, n*2048*2, n*588*2, n*2048*2, n*2048*2, n*2048*2, n*2048*2, span*6144*2,
             4096*6144*4, 4096*2048*4, 4096*12288*4, 4096*6144*2, 4096*6144*4, 4096*6144*2, 4096*6144*4,
             n*588, 3*256*4, n*2*4, 16384*16*4, 2*4, 4*mib]
    return sum(-(-b // 256) * 256 for b in parts)


def normalization_lut(snapshot):
    processor = json.loads((snapshot / "processor_config.json").read_text())["image_processor"]
    if processor["patch_size"] != 14 or processor["image_processor_type"] != "PixtralImageProcessor":
        raise ValueError("unsupported Pixtral processor")
    mean = np.asarray(processor["image_mean"], np.float32)
    std = np.asarray(processor["image_std"], np.float32)
    return np.ascontiguousarray((np.arange(256, dtype=np.float32)[None, :]
        * np.float32(processor["rescale_factor"]) - mean[:, None]) / std[:, None])


GRIDS = {"square": (16, 16), "wide": (24, 80), "tall": (110, 30), "max": (110, 110)}


def fixture(name):
    gh, gw = GRIDS[name]
    yy, xx = np.indices((gh * 14, gw * 14), dtype=np.uint32)
    rgb = np.stack([(xx * 7 + yy * 3) % 256, (xx // 8 + yy * 11) % 256,
        ((xx // 32) ^ (yy // 16)) * 31 % 256], axis=-1).astype(np.uint8)
    return gh, gw, np.ascontiguousarray(rgb)


def reference(config, tensors, dtype, device="cuda"):
    """Official modules, loaded from consolidated names (HF's converter mapping)."""
    import torch
    from transformers.models.pixtral.configuration_pixtral import PixtralVisionConfig
    from transformers.models.pixtral.modeling_pixtral import PixtralVisionModel
    from transformers.models.mistral_large4 import modeling_mistral_large4 as ml4
    from transformers.models.mistral_large4.configuration_mistral_large4 import MistralLarge4Config
    vision_config = PixtralVisionConfig(**config["vision_config"])
    vision_config._attn_implementation = "sdpa"
    tower = PixtralVisionModel(vision_config)
    full = MistralLarge4Config(**{k: v for k, v in config.items() if k not in ("quantization_config",)})
    projector = ml4.MistralLarge4MultiModalProjector(full)
    t = lambda name: torch.from_numpy((tensors[name].astype(np.uint32) << 16).view(np.float32))
    state = {"patch_conv.weight": t("vision_encoder.patch_conv.weight"), "ln_pre.weight": t("vision_encoder.ln_pre.weight")}
    rename = {"attention.wq": "attention.q_proj", "attention.wk": "attention.k_proj", "attention.wv": "attention.v_proj",
        "attention.wo": "attention.o_proj", "feed_forward.w1": "feed_forward.gate_proj",
        "feed_forward.w3": "feed_forward.up_proj", "feed_forward.w2": "feed_forward.down_proj",
        "attention_norm": "attention_norm", "ffn_norm": "ffn_norm"}
    for i in range(DEPTH):
        for source, target in rename.items():
            state[f"transformer.layers.{i}.{target}.weight"] = t(f"vision_encoder.transformer.layers.{i}.{source}.weight")
    tower.load_state_dict(state, strict=True)
    projector.load_state_dict({"norm.weight": t("pre_mm_projector_norm.weight"),
        "patch_merger.merging_layer.weight": t("patch_merger.merging_layer.weight"),
        "linear_1.weight": t("vision_language_adapter.w_in.weight"),
        "linear_2.weight": t("vision_language_adapter.w_out.weight")}, strict=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return tower.to(device, dtype).eval(), projector.to(device, dtype).eval(), [Path(ml4.__file__)]


def official(models, name, dtype, lut, device):
    import torch
    tower, projector, _ = models[dtype]
    gh, gw, rgb = fixture(name)
    pixels = np.stack([lut[c, rgb[:, :, c]] for c in range(3)])[None]
    sizes = torch.tensor([[gh * 14, gw * 14]], device=device)
    with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        hidden = tower(torch.from_numpy(pixels).to(device, dtype), image_sizes=sizes).last_hidden_state
        return projector(hidden.squeeze(0), sizes).float().cpu().numpy()


def stage_states(models, name, dtype, lut, device):
    """Official hidden states after ln_pre (0) and after every fifth layer, as F32."""
    import torch
    tower, _, _ = models[dtype]
    states, hooks = {}, []
    def keep(stage):
        def hook(_module, _inputs, value):
            value = value[0] if isinstance(value, tuple) else value
            states[stage] = value.detach().float().reshape(-1, H).cpu().numpy()
        return hook
    hooks.append(tower.ln_pre.register_forward_hook(keep(0)))
    for i, layer in enumerate(tower.transformer.layers):
        if (i + 1) % 5 == 0:
            hooks.append(layer.register_forward_hook(keep(i + 1)))
    try:
        official(models, name, dtype, lut, device)
    finally:
        for handle in hooks:
            handle.remove()
    return states


def score(native, models, name, lut, device):
    import torch
    gh, gw, _ = fixture(name)
    uh, uw = gh // 2, gw // 2
    images = (native.reshape(uh, uw + 1, LM)[:, :uw].reshape(uh * uw, LM).astype(np.uint32) << 16).view(np.float32)
    bf16, fp32 = official(models, name, torch.bfloat16, lut, device), official(models, name, torch.float32, lut, device)
    floor = common.metrics(bf16, fp32)
    measured = common.calibrated_metrics(common.metrics(images, fp32), floor)
    measured["vs_official_bf16"] = common.metrics(images, bf16)
    return floor, measured


def compare(args):
    """Reference half: score spans saved by a device run (no native library)."""
    import torch
    import torch.nn.functional as F
    config = json.loads((args.snapshot / "config.json").read_text())
    tensors = read_tower(args.snapshot)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {d: reference(config, tensors, d, device) for d in (torch.bfloat16, torch.float32)}
    original = F.scaled_dot_product_attention
    # Independent query blocks bound the official math path's score matrix.
    def bounded(q, k, v, attn_mask=None, **kwargs):
        if attn_mask is not None and not bool((attn_mask == 0).all()):
            raise ValueError("one image per call: the block mask must be all-visible")
        return torch.cat([original(q[:, :, start:start + 512], k, v, **kwargs)
            for start in range(0, q.shape[-2], 512)], dim=-2)
    F.scaled_dot_product_attention = bounded
    lut = normalization_lut(args.snapshot)
    saved = json.loads((args.compare_native / "native.json").read_text())
    result = dict(saved, reference_device=device, g2={}, bf16_floor={},
        modeling_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in models[torch.float32][2]})
    for name in args.grids:
        native = np.load(args.compare_native / f"{name}.npy")
        result["bf16_floor"][name], result["g2"][name] = score(native, models, name, lut, device)
        print(json.dumps(dict(event="G2", grid=name, stages=result["g2"][name])), flush=True)
    result["stages"] = {}
    for name in args.stages:
        refs = {d: stage_states(models, name, d, lut, device) for d in (torch.bfloat16, torch.float32)}
        rows = {}
        for stage in sorted(refs[torch.float32]):
            path = args.compare_native / f"{name}.stage{stage}.npy"
            if path.exists():
                value = np.load(path)
                rows[stage] = dict(native_vs_fp32=common.metrics(value, refs[torch.float32][stage]),
                    native_vs_bf16=common.metrics(value, refs[torch.bfloat16][stage]),
                    bf16_vs_fp32=common.metrics(refs[torch.bfloat16][stage], refs[torch.float32][stage]))
        result["stages"][name] = rows
        for stage, row in rows.items():
            print(json.dumps(dict(event="stage", grid=name, stage=stage, **{k: round(v["relative_l2"], 5)
                for k, v in row.items()})), flush=True)
    result["g2_pass"] = bool(result["g2"]) and all(m["pass"] for m in result["g2"].values())
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["g2_pass"] else 1


def run(args):
    import torch
    import torch.nn.functional as F
    if args.compare_native:
        return compare(args)
    config = json.loads((args.snapshot / "config.json").read_text())
    tensors = read_tower(args.snapshot)
    spec, blob, rows = pack_spec(config, tensors, args.snapshot)
    library, cudart = C.CDLL(str(args.library)), C.CDLL("libcudart.so")
    library.cuteafd_vision_required.argtypes = [C.POINTER(Spec), C.POINTER(Ledger)]
    library.cuteafd_vision_create.argtypes = [C.POINTER(Spec), C.c_int32, C.c_uint64, C.POINTER(C.c_void_p)]
    library.cuteafd_vision_upload.argtypes = [C.c_void_p, C.c_uint64, C.c_void_p, C.c_uint64]
    library.cuteafd_vision_encode.argtypes = [C.c_void_p, C.c_void_p, C.c_uint64, C.c_void_p, C.c_int32, C.c_int32, C.c_void_p, C.c_uint64, OBSERVER, C.c_void_p]
    library.cuteafd_vision_get_ledger.argtypes = [C.c_void_p, C.POINTER(Ledger)]
    library.cuteafd_vision_destroy.argtypes = [C.c_void_p]
    cudart.cudaMemGetInfo.argtypes = [C.POINTER(C.c_size_t), C.POINTER(C.c_size_t)]

    def check(status):
        if status:
            raise RuntimeError(f"native Mistral vision status {status}")

    lut = normalization_lut(args.snapshot)
    owner, required = C.c_void_p(), Ledger()
    check(library.cuteafd_vision_required(C.byref(spec), C.byref(required)))
    # The planner charges exactly what the native owner allocates.
    if required.scratch + required.blas_workspace != planner_scratch_bytes() or required.weights != spec.weight_bytes:
        raise RuntimeError(f"native ledger {required.as_dict()} differs from the planner charge {planner_scratch_bytes()}")
    check(library.cuteafd_vision_create(C.byref(spec), 0, required.weights + required.scratch + required.blas_workspace, C.byref(owner)))
    original = F.scaled_dot_product_attention
    # Split independent query rows to bound the reference's math scratch at 12K patches.
    def bounded(q, k, v, attn_mask=None, **kwargs):
        if attn_mask is not None and not bool((attn_mask == 0).all()):
            raise ValueError("one image per call: the block mask must be all-visible")
        return torch.cat([original(q[:, :, start:start + 512], k, v, **kwargs)
            for start in range(0, q.shape[-2], 512)], dim=-2)
    F.scaled_dot_product_attention = bounded
    result = dict(checkpoint=str(args.snapshot), device=torch.cuda.get_device_name(0),
        sm=torch.cuda.get_device_capability(0), numerics=2, reserved=4, ledger=required.as_dict(),
        planner_scratch_bytes=planner_scratch_bytes(),
        calibration_policy="mistral-bf16-official-floor/1",
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
        processor_sha256=hashlib.sha256((args.snapshot / "processor_config.json").read_bytes()).hexdigest(),
        g2={}, bf16_floor={}, markers_exact={}, g3={}, encode_ms={})
    try:
        check(library.cuteafd_vision_upload(owner, 0, blob, len(blob)))
        del blob
        models = {} if args.native_only else {d: reference(config, tensors, d) for d in (torch.bfloat16, torch.float32)}
        if models:
            result["modeling_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in models[torch.float32][2]}

        cudart.cudaMemcpy.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int]

        def encode(name, stages=None):
            gh, gw, rgb = fixture(name)
            uh, uw = gh // 2, gw // 2
            output, errors = np.empty((uh * (uw + 1), LM), np.uint16), []
            def callback(_ctx, stage, pointer, rows, width, _column):
                try:
                    if stage <= DEPTH:  # F32 hidden states; stage 31 is the BF16 span.
                        value = np.empty((rows, width), np.float32)
                        check(cudart.cudaMemcpy(value.ctypes.data, pointer, value.nbytes, 2))
                        stages[stage] = value
                    return 0
                except BaseException as error:
                    errors.append(repr(error))
                    return 1
            observer = OBSERVER(callback) if stages is not None else OBSERVER()
            started = time.perf_counter()
            check(library.cuteafd_vision_encode(owner, rgb.ctypes.data, rgb.nbytes, lut.ctypes.data,
                gh, gw, output.ctypes.data, output.nbytes, observer, None))
            if errors:
                raise RuntimeError(errors)
            return output, (time.perf_counter() - started) * 1000

        saved = {}
        for name in args.grids:
            encode(name)
            native, _ = encode(name)
            saved[name] = native
            gh, gw, _ = fixture(name)
            uh, uw = gh // 2, gw // 2
            span = native.reshape(uh, uw + 1, LM)
            result["markers_exact"][name] = bool(np.array_equal(span[:-1, uw], np.broadcast_to(rows[0], (uh - 1, LM)))
                and np.array_equal(span[-1, uw], rows[1]))
            if models:
                result["bf16_floor"][name], result["g2"][name] = score(native, models, name, lut, "cuda")
            if args.save_native:
                args.save_native.mkdir(parents=True, exist_ok=True)
                np.save(args.save_native / f"{name}.npy", native)
                if name in args.stages:
                    stages = {}
                    encode(name, stages)
                    for stage, value in stages.items():
                        np.save(args.save_native / f"{name}.stage{stage}.npy", value)
            print(json.dumps(dict(event="G2", grid=name, markers=result["markers_exact"][name],
                stages=result["g2"].get(name))), flush=True)
        initial, final = Ledger(), Ledger()
        check(library.cuteafd_vision_get_ledger(owner, C.byref(initial)))
        def memory():
            free, total = C.c_size_t(), C.c_size_t()
            check(cudart.cudaMemGetInfo(C.byref(free), C.byref(total)))
            return dict(free=free.value, total=total.value,
                        cuda_free_scope="raw cudaMemGetInfo; GB10 under-reports reclaimable page cache")
        before = memory()
        for repeat in range(3):
            for name in reversed(args.grids) if repeat % 2 else args.grids:
                value, elapsed = encode(name)
                result["encode_ms"].setdefault(name, []).append(elapsed)
                result["g3"][name] = result["g3"].get(name, True) and bool(np.array_equal(value, saved[name]))
        check(library.cuteafd_vision_get_ledger(owner, C.byref(final)))
        result.update(common.allocation_metrics(initial, final, before, memory(), result["sm"]))
        result.update(initial_ledger=initial.as_dict(), final_ledger=final.as_dict(),
            markers_pass=all(result["markers_exact"].values()),
            g2_pass=bool(result["g2"]) and all(m["pass"] for m in result["g2"].values()),
            g3_pass=bool(result["g3"]) and all(result["g3"].values()))
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if args.save_native:
            (args.save_native / "native.json").write_text(json.dumps({k: v for k, v in result.items()
                if k not in ("g2", "bf16_floor")}, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return 0 if (result["no_encode_device_allocation"] and result["g3_pass"] and result["markers_pass"]
                     and (result["g2_pass"] or args.native_only)) else 1
    finally:
        F.scaled_dot_product_attention = original
        check(library.cuteafd_vision_destroy(owner))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grids", nargs="+", default=list(GRIDS), choices=list(GRIDS))
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--save-native", type=Path, help="device half: keep each encoded span here")
    parser.add_argument("--compare-native", type=Path, help="reference half: score spans saved by --save-native")
    parser.add_argument("--stages", nargs="*", default=[], choices=list(GRIDS),
        help="also keep (device) / score (reference) the tower's hidden states of these grids")
    raise SystemExit(run(parser.parse_args()))

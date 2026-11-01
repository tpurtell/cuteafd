#!/usr/bin/env python3
"""Muse BF16 tower qualification against its officially bundled Transformers tower.

Run in the architecture's dev container. Only vision tensor extents are loaded.
The reference uses official modules and utilities, while explicit layout stages
make temporal/channel order, swapped position axes and pixel shuffle observable.
"""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

COMMON = Path(__file__).resolve().parents[1] / "mimo_v2/qualify-vision.py"
_spec = importlib.util.spec_from_file_location("vision_common", COMMON)
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)
Spec, Ledger, OBSERVER = common.Spec, common.Ledger, common.OBSERVER
PREFIXES = ("model.vision_tower.", "model.vision_adapter.", "model.vision_projection.")


def read_visual(snapshot):
    mapping = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    tensors = {}
    for name in sorted({v for k, v in mapping.items() if k.startswith(PREFIXES)}):
        with (snapshot / name).open("rb") as file:
            length = int.from_bytes(file.read(8), "little")
            header = json.loads(file.read(length))
            for key, entry in header.items():
                if not key.startswith(PREFIXES):
                    continue
                if entry["dtype"] != "BF16":
                    raise ValueError(f"{key}: expected native BF16")
                start, end = entry["data_offsets"]
                file.seek(8 + length + start)
                raw = file.read(end - start)
                if len(raw) != end - start:
                    raise ValueError(f"truncated {key}")
                tensors[key] = np.frombuffer(raw, np.uint16).reshape(entry["shape"]).copy()
    return tensors


def pack_spec(cfg, tensors, max_tokens=4096):
    vision = cfg["vision_config"]
    for name, want in dict(hidden_size=1536, intermediate_size=8960, num_attention_heads=16,
                           num_hidden_layers=50, patch_size=14, patch_temporal=2,
                           merge_size=2, pos_emb_height=32, pos_emb_width=32).items():
        if vision[name] != want:
            raise ValueError(f"unsupported Muse {name}={vision[name]}")
    if cfg["text_config"]["hidden_size"] != 8192 or not 1 <= max_tokens <= 4096:
        raise ValueError("unsupported Muse LM width/capacity")
    spec = Spec(abi_version=2, reserved=5, max_tokens=max_tokens, output_width=8192,
                hidden=1536, depth=50, heads=16, kv_heads=16, head_dim=96,
                intermediate=8960, patch_size=14, merger_width=6144, norm_eps=1e-5)
    for name in ("patch_bias", "merger_fc1_bias", "merger_fc2_bias"):
        setattr(spec, name, common.NO_OFFSET)
    spec.merger_extra[:] = [common.NO_OFFSET] * 8
    chunks, used, size = [], set(), 0

    def put(name, shape):
        nonlocal size
        value = tensors[name]
        if list(value.shape) != list(shape):
            raise ValueError(f"{name}: {value.shape}, expected {shape}")
        used.add(name)
        gap = (-size) % 256
        chunks.append(bytes(gap))
        size += gap
        offset = size
        chunks.append(value.tobytes())
        size += value.nbytes
        return offset

    base = "model.vision_tower."
    spec.patch = put(base + "patch_embedder.patch_embedding.weight", [1536, 1176])
    spec.pos_embed = put(base + "patch_embedder.position_embedding_table.weight", [1024, 1536])
    spec.merger_extra[0] = put(base + "ln_pre.weight", [1536])
    spec.merger_extra[1] = put(base + "ln_pre.bias", [1536])
    for i, kind in enumerate(vision["layer_types"]):
        block = spec.blocks[i]
        prefix = base + f"layers.{i}."
        qkv = [put(prefix + f"attn.{p}_proj.weight", [1536, 1536]) for p in "qkv"]
        biases = [put(prefix + f"attn.{p}_proj.bias", [1536]) for p in "qkv"]
        assert qkv[1] == qkv[0] + 1536 * 1536 * 2 and qkv[2] == qkv[1] + 1536 * 1536 * 2
        assert biases[1] == biases[0] + 3072 and biases[2] == biases[1] + 3072
        block.qkv, block.qkv_bias = qkv[0], biases[0]
        for field, suffix, shape in [
            ("proj", "attn.proj.weight", [1536, 1536]), ("proj_bias", "attn.proj.bias", [1536]),
            ("gate_up", "mlp.fc1.weight", [8960, 1536]), ("gate_up_bias", "mlp.fc1.bias", [8960]),
            ("down", "mlp.fc2.weight", [1536, 8960]), ("down_bias", "mlp.fc2.bias", [1536]),
            ("norm1", "norm1.weight", [1536]), ("norm2", "norm2.weight", [1536])]:
            setattr(block, field, put(prefix + suffix, shape))
        spec.norm1_bias[i] = put(prefix + "norm1.bias", [1536])
        spec.norm2_bias[i] = put(prefix + "norm2.bias", [1536])
        spec.q_norm[i] = spec.k_norm[i] = block.key0_bias = common.NO_OFFSET
        if kind not in ("window_attention", "full_attention"):
            raise ValueError(f"unknown Muse attention {kind}")
        block.window = 32 if kind == "window_attention" else 0
    spec.merger_norm = put(base + "ln_post.weight", [1536])
    spec.merger_norm_bias = put(base + "ln_post.bias", [1536])
    spec.merger_fc1 = put("model.vision_adapter.fc1.weight", [4096, 6144])
    spec.merger_fc2 = put("model.vision_adapter.fc2.weight", [4096, 4096])
    spec.merger_extra[2] = put("model.vision_projection.weight", [8192, 4096])
    if used != tensors.keys():
        raise ValueError(f"unaccounted Muse vision tensors {tensors.keys() - used}")
    chunks.append(bytes((-size) % 256))
    size += (-size) % 256
    spec.inv_freq = size
    frequencies = 1.0 / np.power(np.float32(10000), np.arange(24, dtype=np.float32) / np.float32(24))
    chunks.append(frequencies.astype(np.float32).tobytes())
    spec.weight_bytes = size + 96
    return spec, b"".join(chunks)


def patches(rgb, patch=14):
    """Official raster rows, each flattened as temporal/channel/y/x."""
    height, width, _ = rgb.shape
    normalized = (rgb.astype(np.float64) / 255.0).astype(np.float32)
    normalized = (normalized - np.float32(0.5)) / np.float32(0.5)
    raster = normalized.reshape(height // patch, patch, width // patch, patch, 3)
    raster = raster.transpose(0, 2, 4, 1, 3)
    return np.broadcast_to(raster[:, :, None], (height // patch, width // patch, 2, 3, patch, patch)).reshape(-1, 1176).copy()


def official_modules(snapshot, tensors, device):
    import torch
    from transformers.models.muse_spark.configuration_muse_spark import MuseSparkConfig
    from transformers.models.muse_spark.modeling_muse_spark import MuseSparkVisionModel, MuseSparkVisionAdapter
    cfg = MuseSparkConfig.from_pretrained(snapshot, local_files_only=True)
    cfg.vision_config._attn_implementation = "sdpa"
    # Allocate directly in native BF16 rather than briefly constructing FP32 weights.
    with torch.device("meta"):
        tower = MuseSparkVisionModel(cfg.vision_config)
        adapter = MuseSparkVisionAdapter(cfg)
        projection = torch.nn.Linear(cfg.projector_hidden_size, cfg.text_config.hidden_size, bias=False)
    modules = [(tower, "model.vision_tower."), (adapter, "model.vision_adapter."), (projection, "model.vision_projection.")]
    for model, prefix in modules:
        state = {name[len(prefix):]: torch.from_numpy(value).view(torch.bfloat16).to(device)
                 for name, value in tensors.items() if name.startswith(prefix)}
        model.load_state_dict(state, strict=True, assign=True)
        model.eval()
    inv, _ = tower.rotary_emb.compute_axial_rope_parameters(cfg.vision_config, device=device)
    tower.rotary_emb.inv_freq = inv
    tower.rotary_emb.original_inv_freq = inv.clone()
    return tower, adapter, projection


def reference_stages(modules, rgb):
    """Official implementation with separately captured layout-sensitive stages."""
    import torch
    from transformers.models.muse_spark import modeling_muse_spark as official
    tower, adapter, projection = modules
    device = next(tower.parameters()).device
    gh, gw = rgb.shape[0] // 14, rgb.shape[1] // 14
    grid = torch.tensor([[1, gh, gw]], dtype=torch.long, device=device)
    x = torch.from_numpy(patches(rgb)).to(device, dtype=torch.bfloat16)
    result = {"patch_input": x}
    with torch.no_grad():
        cu = official.get_vision_cu_seqlens(grid)
        permutation, windows = official.get_vision_window_index(grid, spatial_merge_size=1,
            window_size=448, patch_size=14)
        result["permutation"], result["window_lengths"] = permutation, windows
        positions = official.get_vision_position_ids(grid, spatial_merge_size=1).flip(-1) + 1
        positions = positions[permutation]
        result["positions"] = positions
        x = tower.patch_embedder(x, grid)
        result["0"] = x
        x = tower.ln_pre(x)[permutation]
        result["51"] = x
        rotary = tower.rotary_emb(x, positions)
        result["cos"], result["sin"] = rotary
        for i, block in enumerate(tower.layers):
            kind = tower.config.layer_types[i]
            x = block(x, cu_seqlens=windows if kind == "window_attention" else cu,
                      position_embeddings=rotary)
            result[str(i + 1)] = x
        x = x[torch.argsort(permutation)]
        result["52"] = x
        x = tower.ln_post(x)
        result["53"] = x
        x = tower.pixel_shuffle(x, grid)
        result["54"] = x
        x = adapter(x)
        result["55"] = x
        result["56"] = projection(x)
        # Confirm explicit stages still equal the official public tower path.
        public = projection(adapter(tower(torch.from_numpy(patches(rgb)).to(device, dtype=torch.bfloat16), grid).last_hidden_state))
        if not torch.equal(public, result["56"]):
            raise AssertionError("explicit official layout stages differ from public forward")
    return {name: value.detach().float().cpu().numpy() for name, value in result.items()}


def run(args):
    import torch
    from PIL import Image
    cfg = json.loads((args.snapshot / "config.json").read_text())
    tensors = read_visual(args.snapshot)
    spec, blob = pack_spec(cfg, tensors)
    modules = official_modules(args.snapshot, tensors, "cuda")
    del tensors
    lib = C.CDLL(str(args.library))
    lib.cuteafd_vision_required.argtypes = [C.POINTER(Spec), C.POINTER(Ledger)]
    lib.cuteafd_vision_create.argtypes = [C.POINTER(Spec), C.c_int32, C.c_uint64, C.POINTER(C.c_void_p)]
    lib.cuteafd_vision_upload.argtypes = [C.c_void_p, C.c_uint64, C.c_void_p, C.c_uint64]
    lib.cuteafd_vision_encode.argtypes = [C.c_void_p, C.c_void_p, C.c_uint64, C.c_void_p, C.c_int32, C.c_int32, C.c_void_p, C.c_uint64, OBSERVER, C.c_void_p]
    lib.cuteafd_vision_get_ledger.argtypes = [C.c_void_p, C.POINTER(Ledger)]
    lib.cuteafd_vision_destroy.argtypes = [C.c_void_p]
    cudart = C.CDLL("libcudart.so")
    cudart.cudaMemcpy.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int]
    def check(status):
        if status:
            raise RuntimeError(f"native vision status {status}")
    required, owner = Ledger(), C.c_void_p()
    check(lib.cuteafd_vision_required(C.byref(spec), C.byref(required)))
    check(lib.cuteafd_vision_create(C.byref(spec), 0, required.weights + required.scratch + required.blas_workspace, C.byref(owner)))
    result = dict(device=torch.cuda.get_device_name(0), ledger=required.as_dict(), images={})
    try:
        check(lib.cuteafd_vision_upload(owner, 0, blob, len(blob)))
        del blob
        from transformers.models.muse_glimmer.image_processing_muse_glimmer import smart_resize
        for path in args.images:
            with Image.open(path) as image:
                width, height = image.size
                rh, rw = smart_resize(height, width, 28, 4096)
                if (rh // 28) * (rw // 28) > 4096:
                    raise ValueError("official resize output exceeds admitted capacity")
                rgb = np.asarray(image.convert("RGB").resize((rw, rh), Image.Resampling.LANCZOS)).copy()
            refs = reference_stages(modules, rgb)
            gh, gw = rh // 14, rw // 14
            tokens = gh * gw // 4
            lut = ((np.arange(256, dtype=np.float64) / 255.0).astype(np.float32) - np.float32(.5)) / np.float32(.5)
            lut = np.tile(lut, (3, 1)).copy()
            stages, errors = {}, []
            def observe(_ctx, stage, ptr, rows, width, _order):
                try:
                    value = np.empty((rows, width), np.uint16)
                    check(cudart.cudaMemcpy(value.ctypes.data, ptr, value.nbytes, 2))
                    stages[str(stage)] = (value.astype(np.uint32) << 16).view(np.float32)
                    return 0
                except BaseException as error:
                    errors.append(repr(error))
                    return 1
            output = np.empty((tokens, 8192), np.uint16)
            callback = OBSERVER(observe)
            def encode(cb):
                check(lib.cuteafd_vision_encode(owner, rgb.ctypes.data, rgb.nbytes, lut.ctypes.data,
                    gh, gw, output.ctypes.data, output.nbytes, cb, None))
            encode(OBSERVER())
            start = time.perf_counter()
            encode(callback)
            elapsed = (time.perf_counter() - start) * 1000
            if errors:
                raise RuntimeError(errors)
            stages["56"] = (output.astype(np.uint32) << 16).view(np.float32)
            metadata = {native: bool(np.array_equal(stages[native], refs[official]))
                        for native, official in [("-1", "patch_input"), ("-2", "cos"), ("-3", "sin")]}
            comparisons = {key: common.metrics(value, refs[key]) for key, value in stages.items() if int(key) >= 0}
            replay = {}
            if args.replay_blocks:
                tower = modules[0]
                device = next(tower.parameters()).device
                rotary = tuple(torch.from_numpy(refs[name]).to(device, dtype=torch.bfloat16) for name in ("cos", "sin"))
                cu = torch.tensor([0, gh * gw], device=device, dtype=torch.int32)
                windows = torch.from_numpy(refs["window_lengths"]).to(device, dtype=torch.int32)
                with torch.no_grad():
                    for i, block in enumerate(tower.layers):
                        previous = stages["51" if i == 0 else str(i)]
                        actual = torch.from_numpy(previous).to(device, dtype=torch.bfloat16)
                        expected = block(actual, position_embeddings=rotary,
                            cu_seqlens=windows if tower.config.layer_types[i] == "window_attention" else cu)
                        expected = expected.float().cpu().numpy()
                        row = common.metrics(stages[str(i+1)], expected)
                        per_row = np.linalg.norm(stages[str(i+1)]-expected, axis=1) / np.maximum(np.linalg.norm(expected, axis=1), 1e-20)
                        worst = int(np.argmax(per_row))
                        raster = int(refs["permutation"][worst])
                        row.update(worst_relative_row=float(per_row[worst]), window_row=worst,
                            raster_hw=[raster // gw, raster % gw], kind=tower.config.layer_types[i])
                        replay[str(i+1)] = row
            if not all(str(i) in stages for i in [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 50, 51, 52, 53, 54, 55, 56]):
                raise AssertionError("missing native stage observations")
            saved = output.copy()
            encode(OBSERVER())
            repeat = bool(np.array_equal(saved, output))
            result["images"][str(path)] = dict(grid=[gh, gw], tokens=tokens, encode_ms=elapsed,
                stages=comparisons, metadata_exact=metadata, block_replay=replay,
                byte_repeatable=repeat, output_sha256=hashlib.sha256(output.tobytes()).hexdigest())
            if args.dump_stages:
                args.dump_stages.mkdir(parents=True, exist_ok=True)
                np.savez(args.dump_stages / (path.stem + ".npz"), **{"native_"+k:v for k,v in stages.items()}, **{"ref_"+k:v for k,v in refs.items()})
        final = Ledger()
        check(lib.cuteafd_vision_get_ledger(owner, C.byref(final)))
        result["final_ledger"] = final.as_dict()
        result["pass"] = final.device_allocations == 2 and all(row["byte_repeatable"] and all(row["metadata_exact"].values()) and all(stage["pass"] for stage in row["stages"].values()) for row in result["images"].values())
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        return 0 if result["pass"] else 1
    finally:
        check(lib.cuteafd_vision_destroy(owner))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "library", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--dump-stages", type=Path)
    parser.add_argument("--replay-blocks", action="store_true")
    raise SystemExit(run(parser.parse_args()))

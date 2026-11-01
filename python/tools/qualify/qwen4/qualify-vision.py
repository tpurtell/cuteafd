#!/usr/bin/env python3
"""Qwen G2/G3: pinned official FP32 tower, BF16 yardstick and native byte identity."""
import argparse
import ctypes as C
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
_spec = importlib.util.spec_from_file_location("mimo_tower_gate", ROOT / "python/tools/qualify/mimo_v2/qualify-vision.py")
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)
Block, Ledger, OBSERVER = common.Block, common.Ledger, common.OBSERVER


Spec = common.Spec


def read_visual(snapshot):
    weight_map = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    tensors = {}
    for shard in sorted({f for name, f in weight_map.items() if name.startswith("model.visual.")}):
        with (snapshot / shard).open("rb") as f:
            n = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(n))
            for name, entry in header.items():
                if not name.startswith("model.visual."):
                    continue
                if entry["dtype"] != "BF16":
                    raise ValueError(f"{name}: need BF16, got {entry['dtype']}")
                a, b = entry["data_offsets"]
                f.seek(8 + n + a)
                raw = f.read(b-a)
                if len(raw) != b-a:
                    raise ValueError(f"truncated {name}")
                tensors[name[len("model.visual."):]] = np.frombuffer(raw, np.uint16).reshape(entry["shape"]).copy()
    return tensors


def pack_spec(cfg, tensors):
    expected = dict(depth=27, hidden_size=1152, intermediate_size=4304, num_heads=16,
        patch_size=16, temporal_patch_size=2, spatial_merge_size=2, out_hidden_size=2560,
        num_position_embeddings=2304)
    for name, want in expected.items():
        if cfg["vision_config"][name] != want:
            raise ValueError(f"vision_config.{name}: need {want}")
    spec = Spec(abi_version=2, reserved=2, max_tokens=4096, output_width=2560,
        hidden=1152, depth=27, heads=16, kv_heads=16, head_dim=72,
        intermediate=4304, patch_size=16, merger_width=4608, norm_eps=1e-6)
    chunks, size = [], 0
    def put(name, shape, vector=False):
        nonlocal size
        a = tensors[name]
        if list(a.shape) != shape:
            raise ValueError(f"{name}: {a.shape}, need {shape}")
        if vector:
            a = (a.astype(np.uint32) << 16).view(np.float32)
        gap = (-size) % 256
        chunks.append(bytes(gap)); size += gap
        offset = size
        chunks.append(a.tobytes()); size += a.nbytes
        return offset
    spec.patch = put("patch_embed.proj.weight", [1152,3,2,16,16])
    spec.patch_bias = put("patch_embed.proj.bias", [1152], True)
    spec.pos_embed = put("pos_embed.weight", [2304,1152])
    for i in range(27):
        b = spec.blocks[i]
        for field, suffix, shape, vector in [
            ("qkv","attn.qkv.weight",[3456,1152],False), ("qkv_bias","attn.qkv.bias",[3456],True),
            ("proj","attn.proj.weight",[1152,1152],False), ("proj_bias","attn.proj.bias",[1152],True),
            ("gate_up","mlp.linear_fc1.weight",[4304,1152],False), ("gate_up_bias","mlp.linear_fc1.bias",[4304],True),
            ("down","mlp.linear_fc2.weight",[1152,4304],False), ("down_bias","mlp.linear_fc2.bias",[1152],True),
            ("norm1","norm1.weight",[1152],True), ("norm2","norm2.weight",[1152],True),
        ]:
            setattr(b, field, put(f"blocks.{i}.{suffix}", shape, vector))
        spec.norm1_bias[i] = put(f"blocks.{i}.norm1.bias", [1152], True)
        spec.norm2_bias[i] = put(f"blocks.{i}.norm2.bias", [1152], True)
        b.key0_bias = common.NO_OFFSET
        spec.q_norm[i] = spec.k_norm[i] = common.NO_OFFSET
    for field, name, shape, vector in [
        ("merger_norm","merger.norm.weight",[1152],True), ("merger_norm_bias","merger.norm.bias",[1152],True),
        ("merger_fc1","merger.linear_fc1.weight",[4608,4608],False), ("merger_fc1_bias","merger.linear_fc1.bias",[4608],True),
        ("merger_fc2","merger.linear_fc2.weight",[2560,4608],False), ("merger_fc2_bias","merger.linear_fc2.bias",[2560],True),
    ]:
        setattr(spec, field, put(name, shape, vector))
    for i in range(8): spec.merger_extra[i] = common.NO_OFFSET
    gap = (-size) % 256; chunks.append(bytes(gap)); size += gap
    spec.inv_freq = size
    inv = (1 / np.power(np.float32(10000), np.arange(18, dtype=np.float32) / 18)).astype(np.float32)
    chunks.append(inv.tobytes()); size += inv.nbytes
    spec.weight_bytes = size
    return spec, b"".join(chunks)


def fixture(tokens):
    if tokens != 2048:
        return common.fixture(tokens)
    gh, gw = 32, 256
    yy, xx = np.indices((gh*16, gw*16), dtype=np.uint32)
    rgb = np.stack([(xx*7+yy*3)%256, (xx//8+yy*11)%256,
        ((xx//32)^(yy//16))*31%256], axis=-1).astype(np.uint8)
    return gh, gw, np.ascontiguousarray(rgb)


def normalization_lut():
    # Host processor: f64 rescale then f32 subtraction/division, not f32 rescale.
    values = (np.arange(256, dtype=np.float64) * (1.0 / 255.0)).astype(np.float32)
    return np.ascontiguousarray(np.broadcast_to((values - np.float32(.5)) / np.float32(.5), (3,256)))


def reference_model(cfg, tensors):
    sys.path.insert(0, str(ROOT / "third_party/transformers/src"))
    import torch
    import torch.nn.functional as F
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpVisionConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel
    config = Qwen4ExpVisionConfig(**cfg["vision_config"])
    config._attn_implementation = "sdpa"
    model = Qwen4ExpVisionModel(config)
    state = {name: torch.from_numpy((bits.astype(np.uint32) << 16).view(np.float32)) for name, bits in tensors.items()}
    model.load_state_dict(state, strict=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    original = F.scaled_dot_product_attention
    def bounded(q, k, v, attn_mask=None, **kwargs):
        return torch.cat([original(q[:,:,a:a+128], k, v,
            attn_mask=None if attn_mask is None else attn_mask[...,a:a+128,:], **kwargs)
            for a in range(0, q.shape[-2], 128)], dim=-2)
    F.scaled_dot_product_attention = bounded
    return model.cuda().float().eval()


def calibrated_metrics(measured, floor, *, final_output):
    # Only merged feature rows reach the LM; intermediate drift uses the BF16 floor.
    result = dict(measured, strict_pass=measured["pass"])
    result["pass"] = (measured["relative_l2"] <= floor["relative_l2"] + 0.002
        and measured["mean_cosine"] >= floor["mean_cosine"] - 0.00005
        and measured["worst_cosine"] >= floor["worst_cosine"] - 0.001
        and (not final_output or measured["mean_cosine"] >= 0.9995))
    return result


def run(args):
    import torch
    cfg = json.loads((args.snapshot / "config.json").read_text())
    tensors = read_visual(args.snapshot)
    spec, blob = pack_spec(cfg, tensors)
    lib = C.CDLL(str(args.library))
    lib.cuteafd_vision_required.argtypes = [C.POINTER(Spec), C.POINTER(Ledger)]
    lib.cuteafd_vision_create.argtypes = [C.POINTER(Spec), C.c_int32, C.c_uint64, C.POINTER(C.c_void_p)]
    lib.cuteafd_vision_upload.argtypes = [C.c_void_p, C.c_uint64, C.c_void_p, C.c_uint64]
    lib.cuteafd_vision_encode.argtypes = [C.c_void_p, C.c_void_p, C.c_uint64, C.c_void_p, C.c_int32, C.c_int32, C.c_void_p, C.c_uint64, OBSERVER, C.c_void_p]
    lib.cuteafd_vision_get_ledger.argtypes = [C.c_void_p, C.POINTER(Ledger)]
    lib.cuteafd_vision_destroy.argtypes = [C.c_void_p]
    cudart = C.CDLL("libcudart.so")
    cudart.cudaMemcpy.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int]
    def check(rc):
        if rc: raise RuntimeError(f"native vision status {rc}")
    required, owner = Ledger(), C.c_void_p()
    check(lib.cuteafd_vision_required(C.byref(spec), C.byref(required)))
    check(lib.cuteafd_vision_create(C.byref(spec), 0, required.weights+required.scratch+required.blas_workspace, C.byref(owner)))
    result = dict(checkpoint=str(args.snapshot), sm=torch.cuda.get_device_capability(0), numerics=1,
        ledger=required.as_dict(), g2={}, bf16_yardstick={}, g3={}, encode_ms={},
        calibration_policy="qwen-bf16-relative-intermediates-final-mean-0.9995-v1")
    lut = normalization_lut()
    try:
        check(lib.cuteafd_vision_upload(owner, 0, blob, len(blob)))
        del blob
        model = None if args.native_only else reference_model(cfg, tensors)
        del tensors
        def encode(tokens, observing=False):
            gh, gw, rgb = fixture(tokens)
            out = np.empty((tokens,2560), np.uint16); stages, errors = {}, []
            def observe(_ctx, stage, ptr, rows, width, _col):
                try:
                    if stage != 30:
                        a = np.empty((rows,width), np.float32)
                        check(cudart.cudaMemcpy(a.ctypes.data, ptr, a.nbytes, 2)); stages[stage] = a
                    return 0
                except BaseException as error:
                    errors.append(repr(error)); return 1
            callback = OBSERVER(observe) if observing else OBSERVER()
            started = time.perf_counter()
            check(lib.cuteafd_vision_encode(owner, rgb.ctypes.data, rgb.nbytes, lut.ctypes.data,
                gh, gw, out.ctypes.data, out.nbytes, callback, None))
            if errors: raise RuntimeError(errors)
            return out, stages, (time.perf_counter()-started)*1000
        saved = {}
        for tokens in args.tokens:
            encode(tokens)
            native, stages, _ = encode(tokens, model is not None); saved[tokens] = native
            result.setdefault("output_sha256", {})[str(tokens)] = hashlib.sha256(native.tobytes()).hexdigest()
            if model is None: continue
            gh, gw, rgb = fixture(tokens)
            grid = torch.tensor([[1,gh,gw]], device="cuda")
            refs, hooks = {}, []
            def hook(stage):
                def capture(_module, _input, output):
                    refs[stage] = output.detach().float().cpu().numpy()
                return capture
            def pre_hook(_module, inputs):
                refs[0] = inputs[0].detach().float().cpu().numpy()
            hooks.append(model.blocks[0].register_forward_pre_hook(pre_hook))
            hooks.append(model.merger.norm.register_forward_hook(hook(29)))
            for i, block in enumerate(model.blocks):
                if (i+1)%4==0 or i==26: hooks.append(block.register_forward_hook(hook(i+1)))
            with torch.no_grad():
                ref = model(torch.from_numpy(common.patches(rgb,lut)).cuda(), grid).pooler_output.float().cpu().numpy()
            fp32_refs, refs = refs, {}
            measured = {str(stage): common.metrics(value,fp32_refs[stage]) for stage,value in stages.items()}
            measured["30"] = common.metrics((native.astype(np.uint32)<<16).view(np.float32), ref)
            inv = model.rotary_pos_emb.inv_freq.detach().clone()
            model.bfloat16(); model.rotary_pos_emb.inv_freq = inv
            with torch.no_grad():
                floor = model(torch.from_numpy(common.patches(rgb,lut)).cuda(),grid).pooler_output.float().cpu().numpy()
            yardstick = {str(stage): common.metrics(value,fp32_refs[stage]) for stage,value in refs.items()}
            yardstick["30"] = common.metrics(floor,ref)
            if args.dump_stages is not None:
                args.dump_stages.mkdir(parents=True, exist_ok=True)
                np.savez(args.dump_stages / f"stages-{tokens}.npz",
                    **{f"native_{stage}": value for stage, value in stages.items()},
                    **{f"fp32_{stage}": value for stage, value in fp32_refs.items()},
                    **{f"bf16_{stage}": value for stage, value in refs.items()},
                    native_30=(native.astype(np.uint32)<<16).view(np.float32), fp32_30=ref, bf16_30=floor)
            result["bf16_yardstick"][str(tokens)] = yardstick
            result["g2"][str(tokens)] = {stage: calibrated_metrics(value,yardstick[stage],final_output=stage=="30") for stage,value in measured.items()}
            model.float(); model.rotary_pos_emb.inv_freq.copy_(inv)
            for h in hooks: h.remove()
            print(json.dumps(dict(event="G2",tokens=tokens,stages=result["g2"][str(tokens)])),flush=True)
        initial = Ledger(); check(lib.cuteafd_vision_get_ledger(owner,C.byref(initial)))
        def cuda_memory():
            free,total = C.c_size_t(),C.c_size_t()
            check(cudart.cudaMemGetInfo(C.byref(free),C.byref(total)))
            return dict(free=free.value,total=total.value,
                        cuda_free_scope="raw cudaMemGetInfo; GB10 under-reports reclaimable page cache")
        before = cuda_memory()
        for repeat in range(3):
            for tokens in reversed(args.tokens) if repeat%2 else args.tokens:
                native, _, elapsed = encode(tokens)
                result["encode_ms"].setdefault(str(tokens),[]).append(elapsed)
                result["g3"][str(tokens)] = result["g3"].get(str(tokens),True) and bool(np.array_equal(native,saved[tokens]))
        final = Ledger(); check(lib.cuteafd_vision_get_ledger(owner,C.byref(final)))
        result["initial_ledger"],result["final_ledger"] = initial.as_dict(),final.as_dict()
        result.update(common.allocation_metrics(initial,final,before,cuda_memory(),result["sm"]))
        result["g2_pass"] = bool(result["g2"]) and all(m["pass"] for stages in result["g2"].values() for m in stages.values())
        result["g3_pass"] = all(result["g3"].values())
        args.output.write_text(json.dumps(result,indent=2)+"\n")
        return 0 if result["no_encode_device_allocation"] and result["g3_pass"] and (args.native_only or result["g2_pass"]) else 1
    finally:
        check(lib.cuteafd_vision_destroy(owner))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot","library","output"): p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--tokens",type=int,nargs="+",choices=(256,1024,2048,4096),default=[256,1024,4096])
    p.add_argument("--dump-stages",type=Path,help="diagnostic-only native/FP32/BF16 observer arrays")
    p.add_argument("--native-only",action="store_true")
    raise SystemExit(run(p.parse_args()))

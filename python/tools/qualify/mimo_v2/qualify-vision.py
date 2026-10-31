#!/usr/bin/env python3
"""G2/G3 for the resident MiMo tower. Run inside the architecture's container.

Loads only visual.* extents; executes the snapshot's official vision classes
verbatim in FP32. Query-chunking the official SDPA bounds reference temporary
memory but does not change its mask, weights, arithmetic or model definition.
"""
from __future__ import annotations
import argparse
import ctypes as C
import json
import math
from pathlib import Path
import time
import types

import numpy as np

NO_OFFSET = (1 << 64) - 1
FIELDS = "qkv qkv_bias proj proj_bias gate_up gate_up_bias down down_bias norm1 norm2 key0_bias".split()

class Block(C.Structure):
    _fields_ = [(k, C.c_uint64) for k in FIELDS] + [("window", C.c_int32), ("column_order", C.c_int32)]

class Spec(C.Structure):
    _fields_ = [(k, C.c_uint32) for k in ("storage_abi", "abi_version", "max_tokens", "output_width", "reserved")]
    _fields_ += [(k, C.c_uint64) for k in ("weight_bytes", "patch", "merger_norm", "merger_fc1", "merger_fc2", "inv_freq")]
    _fields_ += [("blocks", C.POINTER(Block))]
    _fields_ += [(k, C.c_uint32) for k in (
        "hidden", "depth", "heads", "kv_heads", "head_dim", "intermediate", "patch_size", "merger_width")]
    _fields_ += [("norm_eps", C.c_float), ("geometry_reserved", C.c_uint32)]
    _fields_ += [(k, C.c_uint64) for k in (
        "patch_bias", "pos_embed", "merger_norm_bias", "merger_fc1_bias", "merger_fc2_bias")]
    _fields_ += [("merger_extra", C.c_uint64 * 8)]
    _fields_ += [(k, C.POINTER(C.c_uint64)) for k in ("norm1_bias", "norm2_bias", "q_norm", "k_norm")]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.storage_abi = 3
        # Keep every backing allocation alive through required/create. Native
        # registration copies these arrays before the synchronous call returns.
        self._layer_storage = {"blocks": (Block * self.depth)()}
        self._layer_storage.update({k: (C.c_uint64 * self.depth)()
            for k in ("norm1_bias", "norm2_bias", "q_norm", "k_norm")})
        for name, backing in self._layer_storage.items():
            setattr(self, name, backing)

class Ledger(C.Structure):
    _fields_ = [(k, C.c_uint64) for k in ("weights", "scratch", "blas_workspace", "device_allocations", "encodes")]
    def as_dict(self):
        return {k: getattr(self, k) for k, _ in self._fields_}

OBSERVER = C.CFUNCTYPE(C.c_int32, C.c_void_p, C.c_int32, C.c_void_p, C.c_int32, C.c_int32, C.c_int32)


def read_visual(snapshot):
    index = snapshot / "model.safetensors.index.json"
    files = sorted({v for k, v in json.loads(index.read_text())["weight_map"].items() if k.startswith("visual.")}) if index.exists() else sorted(p.name for p in snapshot.glob("*.safetensors"))
    tensors = {}
    for name in files:
        with (snapshot / name).open("rb") as f:
            n = int.from_bytes(f.read(8), "little")
            hdr = json.loads(f.read(n))
            for key, entry in hdr.items():
                if not key.startswith("visual."):
                    continue
                if entry["dtype"] != "BF16":
                    raise ValueError(f"{key}: expected BF16, got {entry['dtype']}")
                a, b = entry["data_offsets"]
                f.seek(8 + n + a)
                data = f.read(b-a)
                if len(data) != b-a:
                    raise ValueError(f"truncated {key}")
                tensors[key[7:]] = np.frombuffer(data, np.uint16).reshape(entry["shape"]).copy()
    return tensors


def pack_spec(cfg, tensors):
    vc = cfg["vision_config"]
    for k, expected in {"depth":28,"hidden_size":1280,"intermediate_size":4608,"num_heads":32,"num_key_value_heads":8,"patch_size":16,"temporal_patch_size":2,"spatial_merge_size":2}.items():
        assert vc[k] == expected, (k, vc[k])
    assert vc.get("qk_channels", 64) == 64
    assert vc["out_hidden_size"] in (4096, 6144)
    spec = Spec(abi_version=1, max_tokens=4096, output_width=vc["out_hidden_size"], depth=vc["depth"])
    chunks = []
    size = 0
    def put(a):
        nonlocal size
        gap = (-size) % 256
        chunks.append(bytes(gap)); size += gap
        offset = size
        data = a.tobytes(); chunks.append(data); size += len(data)
        return offset
    def get(name, shape, vector=False):
        a = tensors[name]
        assert list(a.shape) == list(shape), (name, a.shape, shape)
        return (a.astype(np.uint32) << 16).view(np.float32) if vector else a
    spec.patch = put(get("patch_embed.proj.weight", [1280,3,2,16,16]))
    for i in range(spec.depth):
        b = spec.blocks[i]
        p = f"blocks.{i}."
        for field, name, shape, vector in [
            ("qkv", "attn.qkv.weight", [3072,1280], False),
            ("qkv_bias", "attn.qkv.bias", [3072], True),
            ("proj", "attn.proj.weight", [1280,2048], False),
            ("proj_bias", "attn.proj.bias", [1280], True),
            ("down", "mlp.down_proj.weight", [1280,4608], False),
            ("down_bias", "mlp.down_proj.bias", [1280], True),
            ("norm1", "norm1.weight", [1280], True),
            ("norm2", "norm2.weight", [1280], True),
        ]:
            setattr(b, field, put(get(p+name,shape,vector)))
        b.gate_up = put(np.concatenate([get(p+f"mlp.{t}_proj.weight",[4608,1280]) for t in ("gate","up")]))
        b.gate_up_bias = put(np.concatenate([get(p+f"mlp.{t}_proj.bias",[4608],True) for t in ("gate","up")]))
        full = i in vc["fullatt_block_indexes"]
        b.key0_bias = put(get(p+"attn.sinks",[32],True)) if vc["use_sink"] and not full else NO_OFFSET
        b.window = 0 if full else vc["visual_token_window_size"]
        b.column_order = int(vc["vit_window_attn_types"][i] == 1)
    spec.merger_norm = put(get("merger.ln_q.weight",[1280],True))
    spec.merger_fc1 = put(get("merger.mlp.0.weight",[5120,5120]))
    spec.merger_fc2 = put(get("merger.mlp.2.weight",[spec.output_width,5120]))
    spec.inv_freq = put((1 / np.power(np.float32(10000), np.arange(0,32,2,dtype=np.float32)/32)).astype(np.float32))
    spec.weight_bytes = size
    return spec, b"".join(chunks)


def normalization_lut():
    # Same float32 operation order as the pinned PIL processor, supplied by WP1.
    mean = np.array([0.48145466,0.4578275,0.40821073],np.float32)
    std = np.array([0.26862954,0.26130258,0.27577711],np.float32)
    return np.ascontiguousarray((np.arange(256,dtype=np.float32)[None,:] * np.float32(1/255) - mean[:,None]) / std[:,None])


def fixture(tokens):
    # Rectangular grids exercise the unit permutation, not just a square image.
    shapes = {256:(16,64),1024:(32,128),4096:(64,256)}
    gh,gw = shapes[tokens]
    yy,xx = np.indices((gh*16,gw*16),dtype=np.uint32)
    rgb = np.stack([(xx*7+yy*3)%256,(xx//8+yy*11)%256,((xx//32)^(yy//16))*31%256],axis=-1).astype(np.uint8)
    return gh,gw,np.ascontiguousarray(rgb)


def patches(rgb, lut):
    h,w,_ = rgb.shape;gh,gw=h//16,w//16
    x = np.stack([lut[c,rgb[:,:,c]] for c in range(3)])
    x = x.reshape(3,gh//2,2,16,gw//2,2,16).transpose(1,4,2,5,0,3,6)
    return np.ascontiguousarray(np.repeat(x[:,:,:,:,:,None],2,axis=5).reshape(gh*gw,1536))


def reference_model(snapshot,cfg,tensors,device):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    source = (snapshot / "modeling_mimo_v2.py").read_text()
    source = source[source.index("def _rotate_half_vision"):source.index("# Audio encoder")]
    ns = dict(torch=torch,nn=nn,F=F,math=math,ACT2FN={"silu":nn.SiLU()})
    exec(compile(source,str(snapshot/"modeling_mimo_v2.py")+" [official vision]","exec"),ns)
    model = ns["MiMoVisionTransformer"](types.SimpleNamespace(**cfg["vision_config"]))
    state = {k:torch.from_numpy((v.astype(np.uint32)<<16).view(np.float32)) for k,v in tensors.items()}
    missing,extra = model.load_state_dict(state,strict=False)
    expected = {"merger.ln_q.bias","merger.mlp.0.bias","merger.mlp.2.bias"}
    assert set(missing)==expected and not extra,(missing,extra)
    with torch.no_grad():
        for key in missing:
            model.get_parameter(key).zero_()
            assert torch.count_nonzero(model.get_parameter(key)).item()==0
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    # FP32 SDPA math's quadratic scratch would exceed GB10/RTX memory with a
    # 16K-patch image. Splitting independent query rows preserves the reference.
    original = F.scaled_dot_product_attention
    def bounded(q,k,v,attn_mask=None,**kw):
        return torch.cat([original(q[:,:,a:a+128],k,v,attn_mask=None if attn_mask is None else attn_mask[...,a:a+128,:],**kw)
                          for a in range(0,q.shape[-2],128)],dim=-2)
    F.scaled_dot_product_attention=bounded
    return model.to(device,dtype=torch.float32).eval(),missing


def metrics(native,ref):
    a=native.astype(np.float64); b=ref.astype(np.float64)
    an=np.linalg.norm(a,axis=-1);bn=np.linalg.norm(b,axis=-1)
    cos=np.sum(a*b,axis=-1)/np.maximum(an*bn,1e-30)
    result=dict(relative_l2=float(np.linalg.norm(a-b)/np.linalg.norm(b)),mean_cosine=float(cos.mean()),worst_cosine=float(cos.min()))
    result["pass"]=result["relative_l2"]<=0.03 and result["mean_cosine"]>=0.9995 and result["worst_cosine"]>=0.99
    return result


def calibrated_metrics(measured,floor):
    result=dict(measured,strict_pass=measured["pass"])
    result["pass"]=(measured["relative_l2"]<=floor["relative_l2"]+0.002
                    and measured["mean_cosine"]>=max(0.9995,floor["mean_cosine"]-0.00005)
                    and measured["worst_cosine"]>=floor["worst_cosine"]-0.001)
    return result


def allocation_metrics(initial,final,before,after,sm):
    # GB10 free memory also reflects CPU/page-cache activity. Only the native
    # ledger decides whether the resident owner allocated during encode.
    return dict(no_encode_device_allocation=initial.device_allocations==final.device_allocations==2,
                steady_cuda_memory_growth_bytes=before["free"]-after["free"],
                unified_memory_observation=tuple(sm)==(12,1))


def run(args):
    import torch
    cfg=json.loads((args.snapshot/"config.json").read_text())
    tensors=read_visual(args.snapshot)
    spec,blob=pack_spec(cfg,tensors)
    lib=C.CDLL(str(args.library))
    lib.cuteafd_vision_required.argtypes=[C.POINTER(Spec),C.POINTER(Ledger)]
    lib.cuteafd_vision_create.argtypes=[C.POINTER(Spec),C.c_int32,C.c_uint64,C.POINTER(C.c_void_p)]
    lib.cuteafd_vision_upload.argtypes=[C.c_void_p,C.c_uint64,C.c_void_p,C.c_uint64]
    lib.cuteafd_vision_encode.argtypes=[C.c_void_p,C.c_void_p,C.c_uint64,C.c_void_p,C.c_int32,C.c_int32,C.c_void_p,C.c_uint64,OBSERVER,C.c_void_p]
    lib.cuteafd_vision_get_ledger.argtypes=[C.c_void_p,C.POINTER(Ledger)]
    lib.cuteafd_vision_destroy.argtypes=[C.c_void_p]
    cudart=C.CDLL("libcudart.so")
    cudart.cudaMemcpy.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.c_int]
    def check(rc):
        if rc:raise RuntimeError(f"native vision status {rc}")
    owner=C.c_void_p();required=Ledger()
    check(lib.cuteafd_vision_required(C.byref(spec),C.byref(required)))
    check(lib.cuteafd_vision_create(C.byref(spec),0,required.weights+required.scratch+required.blas_workspace,C.byref(owner)))
    lut=normalization_lut()
    result=dict(checkpoint=str(args.snapshot),device=torch.cuda.get_device_name(0),sm=torch.cuda.get_device_capability(0),numerics=1,
                ledger=required.as_dict(),g2={},bf16_yardstick={},g3={},encode_ms={})
    try:
        check(lib.cuteafd_vision_upload(owner,0,blob,len(blob)))
        del blob
        model,missing=reference_model(args.snapshot,cfg,tensors,"cuda") if not args.native_only else (None,None)
        result["missing_keys"]=missing
        del tensors
        def encode(tokens,observe=False):
            gh,gw,rgb=fixture(tokens); out=np.empty((tokens,spec.output_width),np.uint16)
            stages={}; errors=[]
            def on_stage(ctx,stage,ptr,rows,width,col):
                try:
                    if stage==30:return 0
                    a=np.empty((rows,width),np.float32);check(cudart.cudaMemcpy(a.ctypes.data,ptr,a.nbytes,2))
                    if col:
                        index=np.arange(tokens).reshape(gh//2,gw//2).T.reshape(-1)
                        a=a.reshape(tokens,4,width)[np.argsort(index)].reshape(rows,width)
                    stages[stage]=a
                    return 0
                except BaseException as exc:
                    errors.append(repr(exc));return 1
            callback=OBSERVER(on_stage) if observe else OBSERVER()
            start=time.perf_counter()
            check(lib.cuteafd_vision_encode(owner,rgb.ctypes.data,rgb.nbytes,lut.ctypes.data,gh,gw,out.ctypes.data,out.nbytes,callback,None))
            elapsed=(time.perf_counter()-start)*1000
            if errors:raise RuntimeError(errors)
            return out,stages,elapsed
        saved={}
        for tokens in args.tokens:
            encode(tokens)  # one shape warm-up; exclude from timing
            native,stages,_=encode(tokens,observe=model is not None)
            saved[tokens]=native
            if model is not None:
                gh,gw,rgb=fixture(tokens);grid=torch.tensor([[1,gh,gw]],device="cuda")
                refs={};hooks=[]
                index=np.arange(tokens).reshape(gh//2,gw//2).T.reshape(-1)
                def hook(stage,col=False):
                    def capture(_mod,_inputs,output):
                        a=output.detach().float().cpu().numpy()
                        if col:a=a.reshape(tokens,4,1280)[np.argsort(index)].reshape(-1,1280)
                        refs[stage]=a
                    return capture
                hooks.append(model.patch_embed.register_forward_hook(hook(0)))
                hooks.append(model.merger.ln_q.register_forward_hook(hook(29)))
                for i,b in enumerate(model.blocks):
                    if (i+1)%4==0:hooks.append(b.register_forward_hook(hook(i+1,cfg["vision_config"]["vit_window_attn_types"][i]==1)))
                with torch.no_grad():
                    ref=model(torch.from_numpy(patches(rgb,lut)).to("cuda"),grid).float().cpu().numpy()
                compared={str(stage):metrics(a,refs[stage]) for stage,a in stages.items()}
                compared["30"]=metrics((native.astype(np.uint32)<<16).view(np.float32),ref)
                if args.bf16_yardstick:
                    fp32_refs=refs;refs={}
                    inv_freq=model.rotary_pos_emb.inv_freq.detach().clone()
                    model.bfloat16()
                    if args.bf16_rotary_fp32:
                        model.rotary_pos_emb.inv_freq=inv_freq
                    with torch.no_grad():
                        bf16_ref=model(torch.from_numpy(patches(rgb,lut)).to("cuda"),grid).float().cpu().numpy()
                    yardstick={str(stage):metrics(a,fp32_refs[stage]) for stage,a in refs.items()}
                    yardstick["30"]=metrics(bf16_ref,ref)
                    result["bf16_yardstick"][str(tokens)]=yardstick
                    # Orchestrator-calibrated tower floor; keep all original strict
                    # verdicts so this does not conceal a literal design-bar miss.
                    compared={stage:calibrated_metrics(m,yardstick[stage]) for stage,m in compared.items()}
                    model.float()
                    model.rotary_pos_emb.inv_freq.copy_(inv_freq)
                for h in hooks:h.remove()
                result["g2"][str(tokens)]=compared
                print(json.dumps(dict(event="G2",tokens=tokens,stages=compared)),flush=True)
            del stages
        # Three encodes of each image, interleaved with different sizes. CUDA's
        # free memory is informational; the ledger gates native allocations.
        initial_ledger=Ledger();check(lib.cuteafd_vision_get_ledger(owner,C.byref(initial_ledger)))
        result["initial_ledger"]=initial_ledger.as_dict()
        def cuda_memory():
            free=C.c_size_t();total=C.c_size_t()
            check(cudart.cudaMemGetInfo(C.byref(free),C.byref(total)))
            return dict(free=free.value,total=total.value,
                        cuda_free_scope="raw cudaMemGetInfo; GB10 under-reports reclaimable page cache")
        result["steady_memory_before"]=cuda_memory()
        for repeat in range(3):
            for tokens in reversed(args.tokens) if repeat%2 else args.tokens:
                native,_,elapsed=encode(tokens)
                result["encode_ms"].setdefault(str(tokens),[]).append(elapsed)
                result["g3"][str(tokens)]=result["g3"].get(str(tokens),True) and bool(np.array_equal(native,saved[tokens]))
        ledger=Ledger();check(lib.cuteafd_vision_get_ledger(owner,C.byref(ledger)))
        result["final_ledger"]=ledger.as_dict()
        result["steady_memory_after"]=cuda_memory()
        result.update(allocation_metrics(initial_ledger,ledger,result["steady_memory_before"],
                                         result["steady_memory_after"],result["sm"]))
        result["bf16_rotary_fp32"]=bool(args.bf16_rotary_fp32)
        result["g2_pass"]=bool(result["g2"]) and all(m["pass"] for stages in result["g2"].values() for m in stages.values())
        result["g3_pass"]=all(result["g3"].values())
        args.output.write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result),flush=True)
        return 0 if result["no_encode_device_allocation"] and result["g3_pass"] and (result["g2_pass"] or args.native_only) else 1
    finally:
        check(lib.cuteafd_vision_destroy(owner))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot",type=Path,required=True)
    p.add_argument("--library",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--tokens",type=int,nargs="+",default=[256,1024,4096],choices=[256,1024,4096])
    p.add_argument("--native-only",action="store_true")
    p.add_argument("--bf16-rotary-fp32",action="store_true",help="retain FP32 RoPE buffer in the official BF16 yardstick")
    p.add_argument("--bf16-yardstick",action="store_true",help="score official BF16 against FP32 and apply the calibrated per-stage floor")
    raise SystemExit(run(p.parse_args()))

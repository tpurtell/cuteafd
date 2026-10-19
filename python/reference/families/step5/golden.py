#!/usr/bin/env python3
"""Pinned Step-5 BF16 text reference, layer-major and target-only (layers 0-91).

An explicit native-FP8 diagnostic isolates engine error; BF16 remains the fidelity root.
Uses the release's decoder, FP32 norms/residual/router, clamped SwiGLU, CSA,
E4M3 indexer, region selector and sparse GQA kernels. Only execution geometry
changes: fixed-M128 linears and bounded query blocks. SSMax is inactive in the
release. MTP and the vision encoder are not evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python/reference"))
from fidelity_windows import (CheckpointStorage, finish_golden, load_set,
                              release_checkpoint, set_hash, verify_snapshot, write_scored_logits)

CHECKPOINT = "stepfun-ai/Step-5-Preview-BF16"
REVISION = "7734b7ffac118d70d8398bfa3eb0fab007eb45b1"
FP8_CHECKPOINT = "stepfun-ai/Step-5-Preview"
FP8_REVISION = "364b3b03fed2477aa0953fe855a2c4c8c48ee3cc"
SOURCE_PINS = {
    "model.py": "330a166878910b21e37f65669033a84fcaada3b58d7903fc16ef55c496cb1ea5",
    "kernel.py": "2cd01d57d2d135397d2fd36c203684af7564b5e6c6f5d3853b3d35bedeb548be",
    "vision_encoder.py": "8943ee555ffd2a32213f6f9f0a7146d1f49566c65c8b7691c37055208cdb6b53",
}
SPARSE_LAUNCH = {
    "official_launch_configuration": False,
    "kernel": "_sparse_attn_prefill_kernel",
    "num_stages": 1,
    "block_regions": 16,
    "arithmetic": "official tiles and region-loop order unchanged; pipeline stages overridden",
}


class SparsePrefillLaunch:
    """Override pipelining only; retain the official wrapper and kernel arithmetic."""
    def __init__(self, kernel):
        self.kernel = kernel

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def invoke(*args, **kwargs):
            if "num_stages" in kwargs:
                raise ValueError("official sparse prefill now sets pipeline stages explicitly")
            return launch(*args, **kwargs, num_stages=SPARSE_LAUNCH["num_stages"])

        return invoke


def verify_source(snapshot, fp8_diagnostic=False):
    revision = FP8_REVISION if fp8_diagnostic else REVISION
    if snapshot.name != revision:
        raise ValueError("Step-5 reference requires the pinned official revision for its mode")
    pins = {name: hashlib.sha256((snapshot / "inference" / name).read_bytes()).hexdigest()
            for name in SOURCE_PINS}
    if pins != SOURCE_PINS:
        raise ValueError("Step-5 official inference source differs from the pinned release")
    return {"revision": revision, **{name.removesuffix(".py") + "_sha256": pin for name, pin in pins.items()}}


def reference_manifest(args):
    """Rebind identical BF16-root tokens to the pinned FP8 arm, diagnostic only."""
    import copy
    manifest = load_set(args.windows, "step5")
    if not args.fp8_diagnostic:
        return manifest
    original = manifest.get("diagnostic_bf16_set_sha256", manifest["set_sha256"])
    if "diagnostic_bf16_set_sha256" not in manifest:
        if manifest["checkpoint"] != CHECKPOINT:
            raise ValueError("FP8 diagnostic input must originate from the BF16 fidelity root")
        arm = manifest.get("reference_snapshot", manifest.get("generation_arm", {}))
        if arm.get("snapshot_revision") != REVISION:
            raise ValueError("FP8 diagnostic requires pinned BF16 input provenance")
    tokenizer = hashlib.sha256((args.snapshot / "tokenizer.json").read_bytes()).hexdigest()
    if manifest.get("tokenizer_sha256") not in (None, tokenizer):
        raise ValueError("FP8 diagnostic tokenizer differs from the BF16 input tokens")
    manifest = copy.deepcopy(manifest)
    manifest["checkpoint"] = FP8_CHECKPOINT
    manifest["diagnostic_bf16_set_sha256"] = original
    manifest["reference_snapshot"] = {
        "checkpoint": FP8_CHECKPOINT, "precision": "fp8",
        "snapshot_revision": FP8_REVISION,
        "tokenizer_sha256": hashlib.sha256((args.snapshot / "tokenizer.json").read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256((args.snapshot / "config.json").read_bytes()).hexdigest(),
    }
    manifest["set_sha256"] = set_hash(manifest)
    return manifest


def fixed_fp8_rows(function):
    """Keep official FP8 GEMM M fixed across routed-count changes, no math substitution."""
    def linear(x, weight, scale):
        if scale is None:
            return function(x, weight, scale)
        import torch
        from shape_invariant import ROWS
        flat = x.reshape(-1, x.shape[-1])
        if not len(flat):
            return function(x, weight, scale)
        result = []
        for start in range(0, len(flat), ROWS):
            count = min(ROWS, len(flat) - start)
            padded = flat.new_zeros((ROWS, flat.shape[-1]))
            padded[:count].copy_(flat[start:start + count])
            result.append(function(padded, weight, scale)[:count])
        return torch.cat(result).reshape(*x.shape[:-1], weight.shape[0])
    return linear


def decoder_config(payload):
    """The release stores expert quantization beside text_config, not inside it."""
    text = dict(payload["text_config"])
    if "quantization_config" in payload:
        quantization = payload["quantization_config"]
        if "quantization_config" in text and text["quantization_config"] != quantization:
            raise ValueError("conflicting Step-5 expert quantization metadata")
        text["quantization_config"] = quantization
    return text


def validate_config(config, fp8_diagnostic=False):
    expected = list(range(3, 88)) if fp8_diagnostic else []
    quantized = [i for i in config.moe_layer_list if config.is_fp8_expert_layer(i)]
    if config.num_hidden_layers != 92 or config.tp != 1 or quantized != expected:
        raise ValueError("Step-5 reference target geometry or native expert format differs")


def verify_runtime_source():
    """Bind the imported config/tokenizer runtime to the verified source tree."""
    import transformers
    source = ROOT / "third_party/transformers"
    if Path(transformers.__file__).resolve() != (source / "src/transformers/__init__.py").resolve():
        raise ValueError("Step-5 reference requires the pinned Transformers source")
    path = ROOT / "scripts/build/verify-transformers-source.py"
    spec = importlib.util.spec_from_file_location("step5_transformers_lock", path)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    lock = verifier.verify(source, ROOT / "third_party/transformers.lock.json")
    return {"transformers_revision": lock["revision"],
            "transformers_source_tree_sha256": lock["source_tree_sha256"]}


def load_release(snapshot, fp8_diagnostic=False):
    """Import only hash-checked release modules, never an arbitrary remote package."""
    verify_source(snapshot, fp8_diagnostic)
    previous = {name: sys.modules.get(name) for name in ("kernel", "vision_encoder")}
    modules = {}
    try:
        for name in ("kernel", "vision_encoder", "model"):
            path = snapshot / "inference" / (name + ".py")
            module_name = "step5_release_model" if name == "model" else name
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            modules[name] = module
        modules["kernel"]._sparse_attn_prefill_kernel = SparsePrefillLaunch(
            modules["kernel"]._sparse_attn_prefill_kernel)
        if fp8_diagnostic:
            modules["model"].linear_fp8_or_bf16 = fixed_fp8_rows(
                modules["kernel"].linear_fp8_or_bf16)
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return modules["model"], modules["kernel"]


def norm_accuracy_fp64(x, weight, epsilon, *, zero_centered=True, bias=None,
                       subtract_mean=False, cos=None, sin=None):
    """Same-input unrounded oracle for component qualification, not model execution.

    Preserve the declared operand dtypes and the BF16 normalized producer before
    optional RoPE; only the reference reductions and affine arithmetic use FP64.
    """
    import torch
    value = x.double()
    mean = value.mean(-1, keepdim=True) if subtract_mean else 0.
    variance = value.square().mean(-1, keepdim=True) - mean * mean
    epsilon = float(torch.tensor(epsilon, dtype=torch.float32))
    value = (value - mean) * torch.rsqrt(variance + epsilon)
    value *= weight.double() + float(zero_centered)
    if bias is not None:
        value += bias.double()
    if cos is not None:
        if sin is None or x.ndim != 3:
            raise ValueError("rotary norm oracle requires paired tables and token/head/dim input")
        half = cos.shape[-1]
        rounded = value.to(torch.bfloat16).double()
        real, imaginary = rounded[..., :half], rounded[..., half:2 * half]
        c, s = cos.double().unsqueeze(1), sin.double().unsqueeze(1)
        value[..., :half] = real * c - imaginary * s
        value[..., half:2 * half] = real * s + imaginary * c
    return value


def sliding_attention(query, key, value, window):
    """Fixed 128-query, (window+127)-key SDPA, without quadratic masks."""
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from shape_invariant import ROWS

    length, heads, width = query.shape
    key_extent = window + ROWS - 1
    outputs = []
    for start in range(0, length, ROWS):
        count = min(ROWS, length - start)
        first = start - window + 1
        source_start, source_end = max(first, 0), min(start + ROWS, length)
        left, right = source_start - first, key_extent - (source_end - first)
        q = F.pad(query[start:start + ROWS], (0, 0, 0, 0, 0, ROWS - count))
        k = F.pad(key[source_start:source_end], (0, 0, 0, 0, left, right))
        v = F.pad(value[source_start:source_end], (0, 0, 0, 0, left, right))
        qpos = torch.arange(start, start + ROWS, device=query.device)[:, None]
        kpos = torch.arange(first, first + key_extent, device=query.device)[None, :]
        keep = (kpos >= 0) & (kpos <= qpos) & (kpos > qpos - window) & (kpos < length)
        keep[count:] = False
        keep[count:, left] = True
        with sdpa_kernel(SDPBackend.MATH):
            output = F.scaled_dot_product_attention(
                q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
                attn_mask=keep[None, None], scale=width ** -0.5, enable_gqa=True)
        outputs.append(output[0].transpose(0, 1)[:count])
    return torch.cat(outputs)


def sparse_attention(attention, kernel, query, key, value, hidden, positions, extent):
    """Call official CSA/scoring/selection/attention with bounded query metadata."""
    import torch
    from shape_invariant import ROWS

    indexer = attention.indexer
    geometry = indexer.geometry
    length, groups = len(query), attention.num_kv_groups
    cache = kernel.DSALayerCache(batch=1, max_tokens=extent, num_kv_groups=groups,
                                head_dim=attention.head_dim, geometry=geometry,
                                device=query.device, dtype=query.dtype)
    cache.write_kv(0, 0, key, value)
    iq, ik, iz, weights = indexer.project(hidden, positions)
    kernel.update_summaries_prefill(cache, ik, iz, torch.tensor([length], device=query.device, dtype=torch.int32))
    # Fixed region width avoids sequence-dependent reduction/GEMM geometry.
    regions = (extent + geometry.region_size - 1) // geometry.region_size
    summaries = cache.summary[:regions].to(iq.dtype)
    outputs = []
    for start in range(0, length, ROWS):
        stop = min(start + ROWS, length)
        count = stop - start
        # indexer_logits already tiles queries by 64 and keys by 128. Padding the
        # final tile explicitly also fixes the group-major metadata row stride.
        q = torch.zeros((ROWS, *iq.shape[1:]), device=iq.device, dtype=iq.dtype)
        w = torch.zeros((ROWS, *weights.shape[1:]), device=weights.device, dtype=weights.dtype)
        q[:count], w[:count] = iq[start:stop], weights[start:stop]
        scores = kernel.indexer_logits(q, w, summaries)
        pos = torch.arange(start, start + ROWS, device=query.device, dtype=torch.int32)
        packed, counts = kernel.prefill_sparse_meta(
            scores, pos.repeat(groups).contiguous(), cache.block_table,
            torch.zeros(ROWS * groups, device=query.device, dtype=torch.int32),
            topk=geometry.topk, region_size=geometry.region_size,
            regions_per_page=geometry.regions_per_page)
        # Padded rows must not select future or out-of-capacity K/V.
        counts = counts.view(groups, ROWS)
        counts[:, count:] = 0
        q_main = torch.zeros((ROWS, *query.shape[1:]), device=query.device, dtype=query.dtype)
        q_main[:count] = query[start:stop]
        result, _ = kernel.sparse_attention_prefill(
            q_main, cache.key, cache.value, packed, counts.flatten(), num_kv_groups=groups,
            region_size=geometry.region_size, softmax_scale=attention.scaling)
        outputs.append(result[:count])
    return torch.cat(outputs)


def layer_forward(layer, kernel, residual, extent):
    """Retain the official decoder arithmetic; only bound attention's queries."""
    import torch
    attention = layer.self_attn
    hidden = layer.input_layernorm(residual).to(attention.q_proj.weight.dtype)
    positions = torch.arange(len(hidden), device=hidden.device, dtype=torch.int32)
    query, key, value = attention._project(hidden, positions)
    if attention.is_sparse:
        output = sparse_attention(attention, kernel, query, key, value, hidden, positions, extent)
    else:
        output = sliding_attention(query, key, value, attention.sliding_window)
    output = attention._gate_and_project(output, hidden)
    residual = residual + output.float()
    hidden = layer.post_attention_layernorm(residual).to(output.dtype)
    return residual + layer._feed_forward(hidden).float()


class Weights:
    """Read each layer once in shard-offset order; retire mappings before compute."""
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
        self.files = {}
        self.read_bytes = 0

    def get(self, name):
        from safetensors import safe_open
        shard = self.index[name]
        if shard not in self.files:
            self.files[shard] = safe_open(str(self.snapshot / shard), framework="pt", device="cpu")
        value = self.files[shard].get_tensor(name).clone()
        self.read_bytes += value.numel() * value.element_size()
        return value

    def load_layer(self, layer, layer_id):
        import torch
        prefix = f"model.layers.{layer_id}."
        parameters = {**dict(layer.named_parameters()), **dict(layer.named_buffers())}
        names = {name for name in self.index if name.startswith(prefix)}
        persistent = set(layer.state_dict())
        if names != {prefix + name for name in persistent}:
            raise ValueError(f"layer {layer_id} checkpoint inventory differs from official decoder")
        for shard in sorted({self.index[name] for name in names}):
            with (self.snapshot / shard).open("rb") as stream:
                header = json.loads(stream.read(struct.unpack("<Q", stream.read(8))[0]))
            ordered = sorted((name for name in names if self.index[name] == shard),
                             key=lambda name: header[name]["data_offsets"][0])
            for name in ordered:
                value = self.get(name)
                target = parameters[name.removeprefix(prefix)]
                if value.dtype != target.dtype or value.shape != target.shape:
                    raise ValueError(f"{name}: shape/dtype differs from the native reference")
                target.copy_(value)
                del value
        release_checkpoint(torch.cuda, self)
        layer.self_attn.pack_qkv_()


def run_windows(args, ref, kernel, config, weights, extent):
    import numpy as np
    import torch
    import tokenizers
    import transformers
    import platform
    from shape_invariant import qualify

    manifest = reference_manifest(args)
    identity = verify_snapshot(manifest, args.snapshot)
    args.out.mkdir(parents=True, exist_ok=True)
    proof = qualify(args, manifest, lambda probe: run_windows(probe, ref, kernel, config, weights, extent))
    if args.prefix_only and not getattr(args, "_prefix_probe", False):
        return
    started, times, states, entries = time.monotonic(), [], [], []
    loss, total = 0.0, 0
    with torch.inference_mode():
        embed = weights.get("model.embed_tokens.weight").cuda()
        for window in manifest["windows"]:
            ids = torch.tensor(window["tokens"], device="cuda")
            states.append(torch.nn.functional.embedding(ids, embed).float().cpu())
        del embed, ids
        memory = CheckpointStorage(torch.cuda, weights, max_growth_gib=16)
        rope = {}
        for layer_id in range(config.num_hidden_layers):
            before, read_before = time.monotonic(), weights.read_bytes
            with torch.device("meta"):
                layer = ref.Step5DecoderLayer(config, layer_id, dtype=torch.bfloat16)
            layer = layer.to_empty(device="cuda").eval()
            weights.load_layer(layer, layer_id)
            read_seconds = time.monotonic() - before
            layer.self_attn.build_rope(max_position=extent, device=torch.device("cuda"), shared_cache=rope)
            for i, window in enumerate(manifest["windows"]):
                hidden = layer_forward(layer, kernel, states[i].cuda(), extent)
                if args.layers and layer_id in args.layers:
                    folder = args.out / "windows" / window["id"]
                    folder.mkdir(parents=True, exist_ok=True)
                    hidden.cpu().numpy().astype("<f4").tofile(folder / f"layer{layer_id:02d}.bin")
                states[i] = hidden.cpu()
                if not torch.isfinite(states[i]).all():
                    raise ValueError(f"non-finite Step-5 hidden state at layer {layer_id}")
                del hidden
            if getattr(args, "_prefix_probe", False):
                short, extended = states
                if not torch.equal(short.view(torch.int32), extended[:len(short)].view(torch.int32)):
                    raise ValueError(f"Step-5 prefix hidden-state mismatch at layer {layer_id}")
            del layer
            memory.release()
            memory.check(f"layer {layer_id}")
            times.append(dict(layer=layer_id, seconds=time.monotonic() - before,
                              load_seconds=read_seconds, read_bytes=weights.read_bytes - read_before))
            print(f"layer {layer_id}: {times[-1]['seconds']:.3f}s load={read_seconds:.3f}s "
                  f"bytes={times[-1]['read_bytes']} windows={len(states)}", flush=True)
        norm = ref.Step5RMSNorm(config.hidden_size, eps=config.rms_norm_eps).cuda()
        norm.weight.copy_(weights.get("model.norm.weight").cuda())
        head = weights.get("lm_head.weight").cuda()
        for i, window in enumerate(manifest["windows"]):
            start = window["score_from"]
            # Official final narrowing precedes zero-centered RMSNorm.
            hidden = states[i][start - 1:len(window["tokens"]) - 1].cuda().bfloat16()
            logits = torch.nn.functional.linear(norm(hidden), head).float()
            if not torch.isfinite(logits).all():
                raise ValueError("non-finite Step-5 golden logits")
            targets = torch.tensor(window["tokens"][start:], device="cuda")
            nll = sum(torch.nn.functional.cross_entropy(logits[j:j + 32].double(), targets[j:j + 32],
                                                       reduction="sum").item()
                      for j in range(0, len(logits), 32))
            entry = write_scored_logits(args.out, window, logits.cpu().numpy().astype(np.float32))
            entry["nll"] = nll / len(logits)
            (args.out / entry["path"] / "meta.json").write_text(json.dumps(entry, sort_keys=True) + "\n")
            entries.append(entry)
            loss, total = loss + nll, total + len(logits)
            states[i] = None
            del logits, hidden, targets
    finish_golden(args.out, manifest, entries, snapshot_identity=identity, prefix_qualification=proof,
                  reference=("DIAGNOSTIC official Step-5 native FP8 target decoder; act_quant and block-scaled fp8_gemm"
                             if args.fp8_diagnostic else
                             "pinned official Step-5 BF16 target decoder; fixed-M128; bounded eager SWA and official sparse GQA"),
                  diagnostic_only=args.fp8_diagnostic, fidelity_root=CHECKPOINT,
                  diagnostic_bf16_set_sha256=manifest.get("diagnostic_bf16_set_sha256"),
                  expert_arithmetic="official act_quant + block-scaled fp8_gemm" if args.fp8_diagnostic else "BF16",
                  source_pin={**verify_source(args.snapshot, args.fp8_diagnostic), **args.runtime_source}, seconds=time.monotonic() - started,
                  seconds_per_layer=times, nll=loss / total, scored_tokens=total, mtp=False,
                  use_cache=False, attention="eager", ssmax="checkpoint-only; inactive",
                  dtype="native_fp8_experts_bf16_tails" if args.fp8_diagnostic else "bf16",
                  native_fp32_parameters="norms, indexer weights, router bias and residual stream",
                  gates={"head": "sigmoid", "ffn": "post-SiLU clamp 7; FP32 product"},
                  sparse_attention_launch=SPARSE_LAUNCH,
                  golden_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  torch=torch.__version__, tokenizers=tokenizers.__version__, transformers=transformers.__version__,
                  platform=dict(machine=platform.machine(), gpu=torch.cuda.get_device_name(0),
                                capability=list(torch.cuda.get_device_capability(0))))


def diagnostic_source(args):
    """Bind a tail/head probe to the completed native FP8 reference, not a new golden."""
    meta_bytes = (args.out / "meta.json").read_bytes()
    meta = json.loads(meta_bytes)
    manifest = reference_manifest(args)
    identity = verify_snapshot(manifest, args.snapshot)
    proof = meta.get("prefix_qualification") or {}
    if not (args.fp8_diagnostic and meta.get("diagnostic_only") is True
            and meta.get("checkpoint") == FP8_CHECKPOINT
            and (meta.get("source_pin") or {}).get("revision") == FP8_REVISION
            and meta.get("set_sha256") == manifest["set_sha256"]
            and meta.get("snapshot_identity") == identity
            and proof.get("passed") is True and proof.get("finite") is True
            and proof.get("snapshot_identity") == identity
            and proof.get("set_sha256") == manifest["set_sha256"]):
        raise ValueError("tail/head diagnostic needs the completed pinned native FP8 reference and prefix proof")
    for window, entry in zip(manifest["windows"], meta["windows"], strict=True):
        if entry["id"] != window["id"] or entry["path"] != f"windows/{window['id']}":
            raise ValueError("tail/head reference window inventory mismatch")
        expected = b"".join(struct.pack("<I", token) for token in window["tokens"])
        if (args.out / entry["path"] / "tokens.bin").read_bytes() != expected:
            raise ValueError("tail/head reference tokens differ")
    return manifest, meta, hashlib.sha256(meta_bytes).hexdigest()


def tail_diagnostic(args, ref, kernel, config, weights, extent):
    """Continue saved layer 87 through four official layers, preserving source proof."""
    import numpy as np
    import torch
    manifest, meta, source_sha = diagnostic_source(args)
    states, sources = [], []
    for window, entry in zip(manifest["windows"], meta["windows"], strict=True):
        path = args.out / entry["path"] / "layer87.bin"
        values = np.fromfile(path, dtype="<f4")
        if values.size != len(window["tokens"]) * config.hidden_size or not np.isfinite(values).all():
            raise ValueError("tail diagnostic layer-87 input extent or finite mismatch")
        states.append(torch.from_numpy(values.reshape(-1, config.hidden_size)))
        sources.append(dict(id=window["id"], layer87_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    with torch.inference_mode():
        rope = {}
        for layer_id in range(88, 92):
            with torch.device("meta"):
                layer = ref.Step5DecoderLayer(config, layer_id, dtype=torch.bfloat16)
            layer = layer.to_empty(device="cuda").eval()
            weights.load_layer(layer, layer_id)
            layer.self_attn.build_rope(max_position=extent, device=torch.device("cuda"), shared_cache=rope)
            for i, entry in enumerate(meta["windows"]):
                hidden = layer_forward(layer, kernel, states[i].cuda(), extent)
                if not torch.isfinite(hidden).all():
                    raise ValueError(f"nonfinite tail diagnostic at layer {layer_id}")
                states[i] = hidden.cpu()
                path = args.out / entry["path"] / f"layer{layer_id:02d}.bin"
                payload = states[i].numpy().astype("<f4").tobytes()
                if layer_id == 91 and path.read_bytes() != payload:
                    raise ValueError("tail-only continuation differs from original layer-91 state")
                if layer_id != 91:
                    path.write_bytes(payload)
            del layer, hidden
            release_checkpoint(torch.cuda, weights)
            print(f"tail diagnostic layer {layer_id}: finite; original layer91 byte check enforced", flush=True)
        norm = ref.Step5RMSNorm(config.hidden_size, eps=config.rms_norm_eps).cuda()
        norm.weight.copy_(weights.get("model.norm.weight").cuda())
        for state, entry in zip(states, meta["windows"], strict=True):
            norm(state.cuda().bfloat16()).float().cpu().numpy().astype("<f4").tofile(
                args.out / entry["path"] / "final_norm.bin")
    files = {f"{entry['path']}/{name}": hashlib.sha256((args.out / entry["path"] / name).read_bytes()).hexdigest()
             for entry in meta["windows"] for name in ["layer88.bin", "layer89.bin", "layer90.bin", "layer91.bin", "final_norm.bin"]}
    (args.out / "tail-diagnostic.json").write_text(json.dumps(dict(
        schema="cuteafd.step5.tail-diagnostic/1", diagnostic_only=True, source_meta_sha256=source_sha,
        sources=sources, files=files, original_layer91_byte_exact=True,
        source_pin=verify_source(args.snapshot, True)), indent=2) + "\n")


def cpu_head_check(args):
    """BF16 products and FP32 sums, as in the shared head; no CUDA or target execution."""
    import numpy as np
    import torch
    manifest, meta, source_sha = diagnostic_source(args)
    torch.set_num_threads(16)
    weights = Weights(args.snapshot)
    head = weights.get("lm_head.weight")
    scale = weights.get("model.norm.weight")
    if head.dtype != torch.bfloat16 or head.shape != (128896, 4096) or scale.dtype != torch.float32:
        raise ValueError("CPU head source operands disagree with engine BF16/F32 layout")
    epsilon = decoder_config(json.loads((args.snapshot / "config.json").read_text()))["rms_norm_eps"]
    result = dict(schema="cuteafd.step5.cpu-head/1", diagnostic_only=True, source_meta_sha256=source_sha, windows=[])
    with torch.inference_mode():
        head = head.float()
        for window, entry in zip(manifest["windows"], meta["windows"], strict=True):
            folder = args.out / entry["path"]
            residual = torch.from_numpy(np.fromfile(folder / "layer91.bin", dtype="<f4").reshape(-1, 4096))
            if len(residual) != len(window["tokens"]) or not torch.isfinite(residual).all():
                raise ValueError("CPU head final hidden extent or finite mismatch")
            narrowed = residual.bfloat16().float()
            normalized = (narrowed * torch.rsqrt(narrowed.square().mean(-1, keepdim=True) + epsilon)
                          * (scale + 1)).bfloat16()
            positions = entry["positions"]
            selected = normalized[torch.tensor(positions) - 1].float()
            reference = np.memmap(folder / "logits.bin", dtype="<f4", mode="r", shape=(len(positions), 128896))
            kls, rounded_kls, nlls, rounded_nlls = [], [], [], []
            max_error, rounded_max_error, equal = 0., 0., 0
            for start in range(0, len(positions), 16):
                stop = min(start + 16, len(positions))
                actual = (selected[start:stop] @ head.T).double()
                expected = torch.from_numpy(np.array(reference[start:stop])).double()
                rounded = actual.bfloat16().double()
                lp = expected - torch.logsumexp(expected, -1, keepdim=True)
                targets = torch.tensor([window["tokens"][p] for p in positions[start:stop]])
                for values, kl, nll in [(actual, kls, nlls), (rounded, rounded_kls, rounded_nlls)]:
                    lq = values - torch.logsumexp(values, -1, keepdim=True)
                    kl.extend((lp.exp() * (lp - lq)).sum(-1).tolist())
                    nll.extend((-lq[torch.arange(stop - start), targets]).tolist())
                max_error = max(max_error, (actual - expected).abs().max().item())
                rounded_max_error = max(rounded_max_error, (rounded - expected).abs().max().item())
                equal += (rounded == expected).sum().item()
            result["windows"].append(dict(id=entry["id"], rows=len(positions), reference_nll=entry["nll"],
                fp32_head_nll=sum(nlls) / len(nlls), fp32_head_mean_kl=sum(kls) / len(kls), fp32_head_max_kl=max(kls),
                bf16_rounded_nll=sum(rounded_nlls) / len(rounded_nlls),
                bf16_rounded_mean_kl=sum(rounded_kls) / len(rounded_kls), bf16_rounded_max_kl=max(rounded_kls),
                fp32_head_max_absolute_error=max_error, bf16_rounded_max_absolute_error=rounded_max_error,
                bf16_rounded_equal_fraction=equal / reference.size))
    (args.out / "cpu-head-check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="*")
    parser.add_argument("--fp8-diagnostic", action="store_true",
                        help="Pinned native FP8 comparison only; never replaces the BF16 fidelity root")
    parser.add_argument("--cuda-memory-cap-gib", type=float, default=32,
                        help="Torch allocator cap, GiB (0 < cap <= 64)")
    diagnostic = parser.add_mutually_exclusive_group()
    diagnostic.add_argument("--tail-diagnostic", action="store_true",
                            help="Extend an existing native FP8 reference from saved layer 87")
    diagnostic.add_argument("--cpu-head-check", action="store_true",
                            help="Replay final norm and BF16-products/FP32-sums head without CUDA")
    parser.add_argument("--prefix-only", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    verify_source(args.snapshot, args.fp8_diagnostic)
    if not 0 < args.cuda_memory_cap_gib <= 64:
        parser.error("CUDA allocator cap must be greater than zero and at most 64 GiB")
    manifest = reference_manifest(args)
    identity = verify_snapshot(manifest, args.snapshot)
    expected = FP8_CHECKPOINT if args.fp8_diagnostic else CHECKPOINT
    if manifest["checkpoint"] != expected:
        parser.error("reference manifest checkpoint differs from selected mode")
    args.runtime_source = verify_runtime_source()
    if args.cpu_head_check:
        cpu_head_check(args)
        return
    ref, kernel = load_release(args.snapshot, args.fp8_diagnostic)
    payload = json.loads((args.snapshot / "config.json").read_text())
    config = ref.Step5Config(**decoder_config(payload))
    validate_config(config, args.fp8_diagnostic)
    if args.check_only:
        print(json.dumps(dict(snapshot_identity=identity,
                              source_pin={**verify_source(args.snapshot, args.fp8_diagnostic), **args.runtime_source},
                              sparse_attention_launch=SPARSE_LAUNCH)))
        return
    import torch
    from shape_invariant import install
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., args.cuda_memory_cap_gib * 2**30 / total))
    print(f"reference mode={'native-fp8-diagnostic' if args.fp8_diagnostic else 'bf16-fidelity-root'} "
          f"torch_allocator_cap_gib={args.cuda_memory_cap_gib}", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    install()
    extent = max(len(w["tokens"]) for w in manifest["windows"])
    extent = ((extent + 127) // 128) * 128
    if args.tail_diagnostic:
        tail_diagnostic(args, ref, kernel, config, Weights(args.snapshot), extent)
    else:
        run_windows(args, ref, kernel, config, Weights(args.snapshot), extent)


if __name__ == "__main__":
    main()

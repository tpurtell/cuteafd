#!/usr/bin/env python3
"""Pinned Step-5 BF16 text reference, layer-major and target-only (layers 0-91).

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
                              release_checkpoint, verify_snapshot, write_scored_logits)

CHECKPOINT = "stepfun-ai/Step-5-Preview-BF16"
REVISION = "7734b7ffac118d70d8398bfa3eb0fab007eb45b1"
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


def verify_source(snapshot):
    if snapshot.name != REVISION:
        raise ValueError("Step-5 reference requires the pinned official BF16 revision")
    pins = {name: hashlib.sha256((snapshot / "inference" / name).read_bytes()).hexdigest()
            for name in SOURCE_PINS}
    if pins != SOURCE_PINS:
        raise ValueError("Step-5 official inference source differs from the pinned release")
    return {"revision": REVISION, **{name.removesuffix(".py") + "_sha256": pin for name, pin in pins.items()}}


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


def load_release(snapshot):
    """Import only hash-checked release modules, never an arbitrary remote package."""
    verify_source(snapshot)
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
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return modules["model"], modules["kernel"]


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
                    raise ValueError(f"{name}: shape/dtype differs from the native BF16 reference")
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

    manifest = load_set(args.windows, "step5")
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
                  reference="pinned official Step-5 BF16 target decoder; fixed-M128; bounded eager SWA and official sparse GQA",
                  source_pin={**verify_source(args.snapshot), **args.runtime_source}, seconds=time.monotonic() - started,
                  seconds_per_layer=times, nll=loss / total, scored_tokens=total, mtp=False,
                  use_cache=False, attention="eager", ssmax="checkpoint-only; inactive", dtype="bf16",
                  native_fp32_parameters="norms, indexer weights, router bias and residual stream",
                  gates={"head": "sigmoid", "ffn": "post-SiLU clamp 7; FP32 product"},
                  sparse_attention_launch=SPARSE_LAUNCH,
                  golden_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  torch=torch.__version__, tokenizers=tokenizers.__version__, transformers=transformers.__version__,
                  platform=dict(machine=platform.machine(), gpu=torch.cuda.get_device_name(0),
                                capability=list(torch.cuda.get_device_capability(0))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="*")
    parser.add_argument("--prefix-only", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    verify_source(args.snapshot)
    manifest = load_set(args.windows, "step5")
    identity = verify_snapshot(manifest, args.snapshot)
    if identity.get("checkpoint") != CHECKPOINT:
        parser.error("reference manifest must name the official BF16 root")
    args.runtime_source = verify_runtime_source()
    ref, kernel = load_release(args.snapshot)
    payload = json.loads((args.snapshot / "config.json").read_text())["text_config"]
    config = ref.Step5Config(**payload)
    if (config.num_hidden_layers != 92 or config.tp != 1
            or any(config.is_fp8_expert_layer(i) for i in config.moe_layer_list)):
        parser.error("reference requires all 92 target layers and native BF16 expert weights")
    if args.check_only:
        print(json.dumps(dict(snapshot_identity=identity,
                              source_pin={**verify_source(args.snapshot), **args.runtime_source},
                              sparse_attention_launch=SPARSE_LAUNCH)))
        return
    import torch
    from shape_invariant import install
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    install()
    extent = max(len(w["tokens"]) for w in manifest["windows"])
    extent = ((extent + 127) // 128) * 128
    run_windows(args, ref, kernel, config, Weights(args.snapshot), extent)


if __name__ == "__main__":
    main()

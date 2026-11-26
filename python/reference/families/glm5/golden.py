#!/usr/bin/env python3
"""Golden activations for GLM 5.x (glm_moe_dsa) from transformers' reference.

Runs one GlmMoeDsaDecoderLayer at a time on one GPU (the model does not fit),
eager attention with the DSA top-k mask, FP8 weights dequantized to BF16 with
their 128x128 FP32 block scales. Writes the raw files dsv4-golden style
harnesses read:

  tokens.bin      u32 [T]
  layerNN.bin     bf16 [T, hidden]   output of layer NN
  logits.bin      f32  [T, vocab]
  meta.json

  golden.py --snapshot SNAP (--text "..." | --text-file F | --tokens-file tokens.bin) --out DIR
            [--layers 0 1 2 ...] [--device 0]

``--tokens-file`` takes the u32 ids exactly (an existing fixture's
``tokens.bin``), so the reference is keyed to those ids with no
decode/re-tokenize round trip. The prompt runs as one causal window through
the same DSA adaptation, mask and BF16 final norm as ``--windows``.
"""
from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

# Purge freed CPU staging pages immediately on the ARM Torch allocator.
import os
os.environ.setdefault("MIMALLOC_PURGE_DELAY", "0")

import torch
from safetensors import safe_open

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shape_invariant import install, qualify
from fidelity_windows import (CheckpointStorage, LayerCheckpoints, release_checkpoint, load_set,
                              verify_snapshot, write_scored_logits, finish_golden, log_checkpoint_reads,
                              read_tokens_file, write_tokens_file)


def index_scores(q, k, weights, scaling, extent):
    """Keep index GEMMs fixed, bounding the head-expanded score allocation."""
    import torch.nn.functional as F
    from shape_invariant import ROWS
    length = q.shape[1]
    if k.shape[1] > extent:
        raise ValueError("DSA keys exceed pinned panel extent")
    k = F.pad(k, (0, 0, 0, extent - k.shape[1])).float().transpose(-1, -2).unsqueeze(1)
    output = []
    for start in range(0, length, ROWS):
        count = min(ROWS, length - start)
        query = F.pad(q[:, start:start + ROWS].float(), (0, 0, 0, 0, 0, ROWS - count))
        weight = F.pad(weights[:, start:start + ROWS], (0, 0, 0, ROWS - count))
        scores = F.relu(torch.matmul(query, k) * scaling)
        output.append(torch.matmul(weight.unsqueeze(-2), scores).squeeze(-2)[:, :count])
    return torch.cat(output, dim=1)[..., :length]


def install_dsa(ref, extent):
    """Adapt reference shape/order only; retain vendor projections and score math."""
    import ast
    import inspect
    import textwrap
    from shape_invariant import fixed_index_topk, ROWS
    original = getattr(ref.GlmMoeDsaIndexer.forward, "_reference_original", ref.GlmMoeDsaIndexer.forward)
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    function = tree.body[0]
    # The head-expanded temporary is bounded before it can allocate S*S*H.
    begin = next(i for i, n in enumerate(function.body) if isinstance(n, ast.Assign)
                 and ast.unparse(n.targets[0]) == "scores")
    end = next(i for i, n in enumerate(function.body) if isinstance(n, ast.Assign)
               and ast.unparse(n.targets[0]) == "index_scores")
    if "F.relu(scores)" not in ast.unparse(ast.Module(body=function.body[begin:end], type_ignores=[])):
        raise ValueError("unsupported official DSA score site")
    function.body[begin:end + 1] = ast.parse(
        "weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float() * (self.n_heads**-0.5)\n"
        "index_scores = _reference_glm_scores(q, k, weights, self.softmax_scale, _reference_glm_extent)").body
    if ast.unparse(function.body[-1]) != "return index_scores.topk(topk, dim=-1).indices.to(torch.int32)":
        raise ValueError("unsupported official DSA selection site")
    function.body[-1] = ast.parse("return _reference_glm_topk(index_scores, self.index_topk)").body[0]

    def selection(scores, slots):
        ids = fixed_index_topk(scores, slots).indices
        # Eager scatters into real keys: invalid/masked slots alias causal key0.
        # Duplicate scatter entries cannot add support or change reduction order.
        valid = ids < scores.shape[-1]
        values = scores.gather(-1, ids.clamp_max(scores.shape[-1] - 1))
        ids = torch.where(valid & torch.isfinite(values), ids, 0)
        return ids.sort(dim=-1).values.to(torch.int32)

    ref.__dict__.update(_reference_glm_scores=index_scores, _reference_glm_extent=extent,
                        _reference_glm_topk=selection)
    scope = {}
    exec(compile(ast.fix_missing_locations(tree), inspect.getsourcefile(original), "exec"), ref.__dict__, scope)
    scope["forward"]._reference_original = original
    ref.GlmMoeDsaIndexer.forward = scope["forward"]
    eager = getattr(ref.eager_attention_forward, "_reference_original", ref.eager_attention_forward)

    def attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        import torch.nn.functional as F
        if module.training or dropout or attention_mask is None:
            raise ValueError("GLM reference requires masked inference attention")
        length, keys = query.shape[-2], key.shape[-2]
        if keys > extent:
            raise ValueError("attention keys exceed pinned panel extent")
        key = F.pad(key, (0, 0, 0, extent - keys))
        value = F.pad(value, (0, 0, 0, extent - keys))
        outputs = []
        for start in range(0, length, ROWS):
            count = min(ROWS, length - start)
            q = F.pad(query[..., start:start + ROWS, :], (0, 0, 0, ROWS - count))
            mask = F.pad(attention_mask[..., start:start + ROWS, :],
                         (0, extent - keys, 0, ROWS - count), value=-float("inf"))
            # Padded query rows must not feed all-masked NaNs into the GEMM.
            if count < ROWS:
                mask[..., count:, 0] = 0
            result, weights = eager(module, q, key, value, mask, scaling=scaling, dropout=0.0, **kwargs)
            outputs.append(result[:, :count])
            del weights
        return torch.cat(outputs, dim=1), None

    attention._reference_original = eager
    ref.eager_attention_forward = attention


def checkpoint_states(states):
    arrays = []
    for hidden, topk in states:
        if topk is None or topk.dtype != torch.int32:
            raise ValueError("GLM checkpoint requires the per-window DSA int32 selection")
        arrays.extend((hidden.view(torch.uint16).numpy(),
                       topk.contiguous().view(torch.uint16).reshape(*topk.shape, 2).numpy()))
    return arrays


def restore_states(arrays):
    if len(arrays) % 2:
        raise ValueError("incomplete GLM hidden/index checkpoint pairs")
    return [(torch.from_numpy(arrays[i]).view(torch.bfloat16),
             torch.from_numpy(arrays[i + 1]).view(torch.int32).squeeze(-1))
            for i in range(0, len(arrays), 2)]


class Weights:
    # Re-quantize FP8 blocks to power-of-two (UE8M0) scales, as the b12x FP8
    # linears require, instead of using the checkpoint's FP32 scales.
    requant_ue8m0 = False

    def __init__(self, snapshot: Path):
        self.snapshot = snapshot
        self.index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
        self.files: dict[str, object] = {}
        self.read_bytes = 0
        self.read_seconds = 0.0
        self.staged = {}
        self.layer_read_bytes = 0
        self.layer_read_seconds = 0.0
        self.slow_layers = 0

    def check_layer_read_rate(self, prefix, payload_bytes, elapsed, *, minimum_mbps=200,
                              slow_mbps=100, consecutive_slow_layers=2):
        # Prefix probes reuse this reader; each layer-0 pass gets its own history.
        if prefix == "model.layers.0.":
            self.layer_read_bytes = 0
            self.layer_read_seconds = 0.0
            self.slow_layers = 0
        self.layer_read_bytes += payload_bytes
        self.layer_read_seconds += elapsed
        rate = payload_bytes / 1e6 / max(elapsed, 1e-9)
        average = self.layer_read_bytes / 1e6 / max(self.layer_read_seconds, 1e-9)
        self.slow_layers = self.slow_layers + 1 if rate < slow_mbps else 0
        print(f"checkpoint sequential reads {prefix}: bytes={payload_bytes} "
              f"elapsed={elapsed:.3f}s MB/s={rate:.3f} cumulative_MB/s={average:.3f} "
              f"consecutive_below_{slow_mbps}={self.slow_layers}", flush=True)
        if average < minimum_mbps or self.slow_layers >= consecutive_slow_layers:
            self.staged.clear()
            release_checkpoint(torch.cuda, self)
            raise RuntimeError(f"GLM archive read guard: cumulative {average:.3f}MB/s "
                               f"(minimum {minimum_mbps}), consecutive below {slow_mbps}MB/s "
                               f"={self.slow_layers} (limit {consecutive_slow_layers}); stop, do not crawl")

    def stage_layer(self, prefix: str):
        """Clone complete tensors in physical shard-offset order, never table rows."""
        if self.staged:
            raise ValueError("previous GLM layer staging has not been retired")
        names = [name for name in self.index if name.startswith(prefix)]
        if not names:
            raise ValueError(f"no tensors for {prefix}")
        started, before = time.monotonic(), self.read_bytes
        for shard in sorted({self.index[name] for name in names}):
            # Headers are small; tensor payloads are touched once in offset order.
            with (self.snapshot / shard).open("rb") as stream:
                length = struct.unpack("<Q", stream.read(8))[0]
                header = json.loads(stream.read(length))
            ordered = sorted((name for name in names if self.index[name] == shard),
                             key=lambda name: header[name]["data_offsets"][0])
            with safe_open(str(self.snapshot / shard), framework="pt", device="cpu") as handle:
                for name in ordered:
                    tensor_start = time.monotonic()
                    value = handle.get_tensor(name).clone()
                    self.staged[name] = value
                    self.read_bytes += value.numel() * value.element_size()
                    self.read_seconds += time.monotonic() - tensor_start
        elapsed = time.monotonic() - started
        self.check_layer_read_rate(prefix, self.read_bytes - before, elapsed)

    def raw(self, name: str) -> torch.Tensor:
        if name in self.staged:
            return self.staged.pop(name)
        started = time.monotonic()
        shard = self.index[name]
        if shard not in self.files:
            self.files[shard] = safe_open(str(self.snapshot / shard), framework="pt", device="cpu")
        value = self.files[shard].get_tensor(name).clone()
        self.read_bytes += value.numel() * value.element_size()
        self.read_seconds += time.monotonic() - started
        return value

    def get(self, name: str, device: str = "cuda") -> torch.Tensor:
        """BF16/FP32 tensors as stored; FP8 weights times their block scales."""
        value = self.raw(name).to(device)
        if value.dtype != torch.float8_e4m3fn:
            return value
        scale = self.raw(name.removesuffix("weight") + "weight_scale_inv").to(device).float()
        rows, cols = value.shape
        grow = lambda t: t.repeat_interleave(128, 0)[:rows].repeat_interleave(128, 1)[:, :cols]
        values = value.float()
        # Routed experts are served as EXL3; only coordinator weights are re-quantized.
        if self.requant_ue8m0 and ".mlp.experts." not in name:
            # w * s = w' * 2^ceil(log2 s): w' = fp8(w * s / 2^ceil(log2 s)), never larger than w.
            power = torch.exp2(torch.ceil(torch.log2(scale)))
            values = (values * grow(scale / power)).to(torch.float8_e4m3fn).float()
            scale = power
        return (values * grow(scale)).bfloat16()


def load_layer(layer: torch.nn.Module, weights: Weights, prefix: str) -> None:
    params = dict(layer.named_parameters())
    loaded = set()
    experts = getattr(getattr(layer, "mlp", None), "experts", None)
    if experts is not None:
        count = experts.gate_up_proj.shape[0]
        for e in range(count):
            base = f"{prefix}mlp.experts.{e}."
            with torch.no_grad():
                experts.gate_up_proj[e].copy_(torch.cat([weights.get(base + "gate_proj.weight"),
                                                         weights.get(base + "up_proj.weight")], 0))
                experts.down_proj[e].copy_(weights.get(base + "down_proj.weight"))
        loaded |= {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"}
    for key, param in params.items():
        if key in loaded:
            continue
        name = prefix + key
        if name not in weights.index:
            raise KeyError(f"{name}: no checkpoint tensor")
        with torch.no_grad():
            param.copy_(weights.get(name).to(param.dtype))
        loaded.add(key)
    for key, buffer in layer.named_buffers():
        name = prefix + key
        if name in weights.index:
            with torch.no_grad():
                buffer.copy_(weights.get(name).to(buffer.dtype))
    release_checkpoint(torch.cuda, weights)


def run_windows(a, config, ref, weights):
    manifest = load_set(a.windows, "glm5")
    identity = verify_snapshot(manifest, a.snapshot)
    if not getattr(a, "_prefix_probe", False):
        a._dsa_extent = max(len(w["tokens"]) for w in manifest["windows"])
    install_dsa(ref, a._dsa_extent)
    diagnostic_stop = getattr(a, "diagnostic_layer", None)
    proof = None if diagnostic_stop is not None else qualify(a, manifest, lambda probe: run_windows(probe, config, ref, weights))
    if getattr(a, "prefix_only", False) and not getattr(a, "_prefix_probe", False):
        return
    started, rows, times, states = time.time(), [], [], []
    checkpoints = None
    if getattr(a, "checkpoint_layers", False) and not getattr(a, "_prefix_probe", False):
        shapes = {}
        for w in manifest["windows"]:
            shapes[w["id"] + "-hidden"] = [1, len(w["tokens"]), config.hidden_size]
            shapes[w["id"] + "-index"] = [1, len(w["tokens"]), config.index_topk, 2]
        binding = {"set_sha256": manifest["set_sha256"], "snapshot_identity": identity,
                   "source_seal_sha256": a.source_seal_sha256, "dsa_extent": a._dsa_extent}
        checkpoints = LayerCheckpoints(a.out / "layer-checkpoints", binding, shapes,
                                       resume=getattr(a, "resume_layers", None))
    first_layer = 0
    with torch.inference_mode():
        if checkpoints is not None and checkpoints.resumed is not None:
            last, arrays, times = checkpoints.resumed
            if not 0 <= last < config.num_hidden_layers or len(times) != last + 1:
                raise ValueError("invalid resumed layer extent")
            states = restore_states(arrays)
            checkpoints.resumed = None
            del arrays
            first_layer = last + 1
        else:
            embed = weights.get("model.embed_tokens.weight")
            for w in manifest["windows"]:
                ids = torch.tensor([w["tokens"]], device="cuda")
                h = torch.nn.functional.embedding(ids, embed)
                states.append((h.cpu(), None))
            del embed, ids, h
        memory = CheckpointStorage(torch.cuda, weights)
        rotary = ref.GlmMoeDsaRotaryEmbedding(config=config).cuda()
        for layer_id in range(first_layer, config.num_hidden_layers):
            start = time.time()
            read_started, read_before = time.monotonic(), weights.read_bytes
            weights.stage_layer(f"model.layers.{layer_id}.")
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("meta"):
                layer = ref.GlmMoeDsaDecoderLayer(config, layer_id)
            torch.set_default_dtype(torch.float32)
            layer = layer.to_empty(device="cuda")
            load_layer(layer, weights, f"model.layers.{layer_id}.")
            if weights.staged:
                raise ValueError(f"unconsumed layer staging: {sorted(weights.staged)[:8]}")
            log_checkpoint_reads(f"layer {layer_id} load", (weights,), read_before, read_started)
            layer.eval()
            for i, w in enumerate(manifest["windows"]):
                host_h, host_topk = states[i]
                h = host_h.cuda()
                topk = host_topk.cuda() if host_topk is not None else None
                positions = torch.arange(len(w["tokens"]), device="cuda")[None]
                cos_sin = rotary(h, position_ids=positions)
                # Reused DSA indices belong to this window, never the previous visit.
                mask = torch.full((len(w["tokens"]), len(w["tokens"])), -float("inf"),
                                  device="cuda", dtype=h.dtype).triu(1)[None, None]
                h, topk = layer(h, attention_mask=mask, position_ids=positions,
                                position_embeddings=cos_sin, prev_topk_indices=topk)
                if ((a.layers is not None and layer_id in a.layers)
                        or (getattr(a, "prefix_trace", False) and getattr(a, "_prefix_probe", False))):
                    folder = a.out / "windows" / w["id"]
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / f"layer{layer_id:02d}.bin").write_bytes(h[0].contiguous().view(torch.int16).cpu().numpy().tobytes())
                states[i] = (h.cpu(), topk.cpu() if topk is not None else None)
                del h, topk, positions, cos_sin, mask
            del layer
            memory.release()
            memory.check(f"layer {layer_id}")
            times.append(time.time() - start)
            if checkpoints is not None:
                checkpoints.commit(layer_id, checkpoint_states(states), times)
                memory.release()
                memory.check(f"layer {layer_id} checkpoint")
            print(f"layer {layer_id} {times[-1]:.1f}s ({len(states)} windows)", flush=True)
            if diagnostic_stop == layer_id:
                (a.out / "diagnostic.json").write_text(json.dumps({
                    "qualification": False, "kind": "layer-timing-only",
                    "set_sha256": manifest["set_sha256"], "snapshot_identity": identity,
                    "seconds_per_layer": times, "windows": [w["id"] for w in manifest["windows"]],
                }, sort_keys=True) + "\n")
                return
        norm = ref.GlmMoeDsaRMSNorm(config.hidden_size, config.rms_norm_eps).cuda().to(torch.bfloat16)
        norm.weight.copy_(weights.get("model.norm.weight"))
        head = weights.get("lm_head.weight").float()
        for i, w in enumerate(manifest["windows"]):
            h = states[i][0][:, w["score_from"] - 1:len(w["tokens"]) - 1].cuda()
            logits = torch.nn.functional.linear(norm(h).float()[0], head)
            rows.append(write_scored_logits(a.out, w, logits.cpu().numpy()))
            states[i] = None
            del h, logits
    finish_golden(a.out, manifest, rows, snapshot=str(a.snapshot),
        reference="transformers glm_moe_dsa (eager, fixed-M128 linears/index/attention, fixed ascending-id2048 DSA slots and panel-fixed masked key extent)",
        seconds=time.time() - started, seconds_per_layer=times, snapshot_identity=identity,
        prefix_qualification=proof)


def run_tokens(a, config, ref, weights, tokens):
    """One causal prompt, layer by layer: tokens.bin, layerNN.bin, logits.bin, meta.json."""
    if not tokens:
        raise ValueError("the prompt is empty")
    install_dsa(ref, len(tokens))
    write_tokens_file(a.out / "tokens.bin", tokens)
    layers = config.num_hidden_layers
    save = set(range(layers)) if not a.layers else set(a.layers)
    started, times = time.time(), []
    ids = torch.tensor([tokens], device="cuda")
    positions = torch.arange(len(tokens), device="cuda")[None]
    with torch.inference_mode():
        h = torch.nn.functional.embedding(ids, weights.get("model.embed_tokens.weight"))
        rotary = ref.GlmMoeDsaRotaryEmbedding(config=config).cuda()
        cos_sin = rotary(h, position_ids=positions)
        # The official model's causal mask; DSA top-k narrows it per query row.
        mask = torch.full((len(tokens), len(tokens)), -float("inf"),
                          device="cuda", dtype=h.dtype).triu(1)[None, None]
        topk = None
        memory = CheckpointStorage(torch.cuda, weights)
        for layer_id in range(layers):
            start = time.time()
            read_started, read_before = time.monotonic(), weights.read_bytes
            weights.stage_layer(f"model.layers.{layer_id}.")
            # BF16 parameters; the router keeps its explicit FP32 bias buffer.
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("meta"):
                layer = ref.GlmMoeDsaDecoderLayer(config, layer_id)
            torch.set_default_dtype(torch.float32)
            layer = layer.to_empty(device="cuda")
            load_layer(layer, weights, f"model.layers.{layer_id}.")
            if weights.staged:
                raise ValueError(f"unconsumed layer staging: {sorted(weights.staged)[:8]}")
            log_checkpoint_reads(f"layer {layer_id} load", (weights,), read_before, read_started)
            layer.eval()
            h, topk = layer(h, attention_mask=mask, position_ids=positions, position_embeddings=cos_sin,
                            prev_topk_indices=topk)
            if layer_id in save:
                (a.out / f"layer{layer_id:02d}.bin").write_bytes(h[0].contiguous().view(torch.int16).cpu().numpy().tobytes())
            del layer
            memory.release()
            memory.check(f"layer {layer_id}")
            times.append(time.time() - start)
            print(f"layer {layer_id} {times[-1]:.1f}s", flush=True)
        norm = ref.GlmMoeDsaRMSNorm(config.hidden_size, config.rms_norm_eps).cuda().to(torch.bfloat16)
        norm.weight.copy_(weights.get("model.norm.weight"))
        logits = torch.nn.functional.linear(norm(h).float()[0], weights.get("lm_head.weight").float())
        (a.out / "logits.bin").write_bytes(logits.contiguous().cpu().numpy().tobytes())
    argmax = logits.argmax(-1)
    next_ok = (argmax[:-1] == ids[0, 1:]).float().mean().item()
    nll = -torch.log_softmax(logits[:-1].double(), -1).gather(1, ids[0, 1:, None]).mean().item()
    (a.out / "meta.json").write_text(json.dumps({
        "family": "glm5", "tokens": len(tokens), "snapshot": str(a.snapshot),
        "tokens_file": str(a.tokens_file) if a.tokens_file else None,
        "reference": "transformers glm_moe_dsa (eager, causal mask, fixed-M128 linears/index/attention, "
                     "fixed ascending-id2048 DSA slots); BF16 final norm, FP32 head",
        "requant_ue8m0": a.requant_ue8m0, "seconds": time.time() - started, "seconds_per_layer": times,
        "argmax_last": int(argmax[-1]), "next_token_accuracy": next_ok, "mean_nll": nll,
    }, indent=1))
    print(f"argmax of last position: {int(argmax[-1])}; next-token accuracy {next_ok:.3f}; mean NLL {nll:.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--text", help="prompt text (tokenized with the snapshot tokenizer)")
    p.add_argument("--text-file", type=Path, help="prompt text from a file")
    p.add_argument("--tokens-file", type=Path, help="u32 tokens.bin: the prompt's exact token ids")
    p.add_argument("--layers", type=int, nargs="*", help="layers whose outputs to save (default all)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--requant-ue8m0", action="store_true",
                   help="re-quantize FP8 blocks to power-of-two scales (the b12x linear format)")
    p.add_argument("--windows", type=Path, help="pinned fidelity set; layer-major scored logits")
    p.add_argument("--prefix-only", action="store_true")
    p.add_argument("--diagnostic-layer", type=int, help="stop after layer timing; never qualifies")
    p.add_argument("--prefix-trace", action="store_true", help="save all layers for actual prefix diagnostic")
    p.add_argument("--checkpoint-layers", action="store_true", help="rolling sealed hidden+DSA states")
    p.add_argument("--resume-layers", type=Path, help="resume into fresh output from complete layer states")
    p.add_argument("--source-seal-sha256", help="verified immutable checkpoint source seal")
    a = p.parse_args()
    if a.diagnostic_layer is not None and (not a.windows or a.prefix_only or a.diagnostic_layer < 0):
        p.error("--diagnostic-layer requires full --windows and a nonnegative layer")
    if a.checkpoint_layers and (not a.windows or a.prefix_only or not a.source_seal_sha256
            or len(a.source_seal_sha256) != 64 or any(c not in "0123456789abcdef" for c in a.source_seal_sha256)):
        p.error("--checkpoint-layers requires full --windows and a verified source seal SHA256")
    if a.resume_layers and not a.checkpoint_layers:
        p.error("--resume-layers requires --checkpoint-layers")
    if a.prefix_only and not a.windows:
        p.error("--prefix-only requires --windows")
    if a.windows and (a.text is not None or a.text_file is not None or a.tokens_file is not None
                      or a.requant_ue8m0):
        p.error("--windows cannot be combined with legacy text/tokens or serving requantization")
    if not a.windows and sum(x is not None for x in (a.text, a.text_file, a.tokens_file)) != 1:
        p.error("give exactly one of --text, --text-file, --tokens-file or --windows")

    from tokenizers import Tokenizer
    from transformers import AutoConfig
    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as ref

    torch.cuda.set_device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    install()
    config = AutoConfig.from_pretrained(a.snapshot)
    config._attn_implementation = "eager"
    if a.diagnostic_layer is not None and a.diagnostic_layer >= config.num_hidden_layers:
        p.error("diagnostic layer exceeds model extent")
    if a.windows:
        a.out.mkdir(parents=True, exist_ok=True)
        run_windows(a, config, ref, Weights(a.snapshot))
        return
    if a.tokens_file:
        tokens = read_tokens_file(a.tokens_file, config.vocab_size)
    else:
        text = a.text_file.read_text() if a.text_file else a.text
        tokens = Tokenizer.from_file(str(a.snapshot / "tokenizer.json")).encode(text, add_special_tokens=False).ids
    weights = Weights(a.snapshot)
    weights.requant_ue8m0 = a.requant_ue8m0
    a.out.mkdir(parents=True, exist_ok=True)
    run_tokens(a, config, ref, weights, tokens)


if __name__ == "__main__":
    main()

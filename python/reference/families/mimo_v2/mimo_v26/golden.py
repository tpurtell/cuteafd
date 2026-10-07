#!/usr/bin/env python3
"""Golden activations for MiMo V2.6 Pro (``mimo_v2``, MiMoV2ForCausalLM) from the
snapshot's own modeling code (``modeling_mimo_v2.py``, trust_remote_code).

Runs one ``MiMoV2DecoderLayer`` at a time on one GPU (the model is 535 GB),
eager attention with explicit additive masks (causal for the 10 full layers,
causal and 128-token sliding window for the 60 SWA layers: key k is visible
to query q when ``q - 128 < k <= q``), exactly as ``MiMoV2Model.forward``
chains them. Only the checkpoint layout is decoded here; the arithmetic is the
reference module's:

* ``qkv_proj`` (FP8 E4M3, FP32 ``weight_scale_inv``) is stored TP-interleaved
  for the checkpoint's ``tp_size`` (8, index metadata): each of the 8 row
  shards is ``[q (16 heads x 192) | k (1 x 192) | v (1 x 128)]`` with its own
  128x128 block grid over the whole shard (27 row blocks). Grid row 25
  spans k rows 128-191 and v rows 0-63: scales do not restart at v.
  The shards are dequantized before splitting, then de-interleaved into
  the ``[q; k; v]`` the modeling code splits. Hugh Madden identified the
  former per-segment scale bug in issue #3. Flash TP4 full layers have the
  same shard geometry; their SWA shards (3072/384/256) are aligned.
* Other FP8 tensors: 128x128 blocks. ``o_proj`` is BF16.
* Routed experts are MXFP4: ``weight`` U8 ``[N, K/2]`` (two E2M1 codes per
  byte, the even element in the low nibble) and ``weight_scale`` U8 ``[N,
  K/32]`` (UE8M0, value ``2^(s - 127)``, SGLang's reading). They are widened
  exactly to BF16 (every E2M1 value times a power of two is a BF16 number).
* The router weight is BF16 and ``e_score_correction_bias`` FP32, as stored
  (the gate promotes both to FP32). The MoE sums routes in FP32 and rounds
  once (the module's own ``moe``).

The NVIDIA torch image defaults FP32 matmuls to TF32: turned off here (the
router's FP32 logits pick the routes). Writes the files the golden commands
read:

  tokens.bin      i32  [T]
  layerNN.bin     bf16 [T, hidden]   output of layer NN
  logits.bin      f32  [T, vocab]
  meta.json

  golden.py --snapshot SNAP --text-file prompt.txt --out DIR [--layers 0 1 ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Purge freed CPU staging pages immediately on the ARM Torch allocator.
import os
os.environ.setdefault("MIMALLOC_PURGE_DELAY", "0")

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mimo_v2.golden import masks  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from fidelity_windows import CheckpointStorage, release_checkpoint, load_set, write_scored_logits, finish_golden, verify_snapshot


def run_windows(a, config, Layer, Rotary, Norm, weights):
    manifest = load_set(a.windows, "mimo_v2")
    from fidelity_media import require_media_flag
    media = require_media_flag(manifest, getattr(a, "media", False), "mimo_v2")
    identity = verify_snapshot(manifest, a.snapshot)
    if media:
        from mimo_media import snapshot_identity
        identity.update(snapshot_identity(a.snapshot))
    from shape_invariant import qualify
    proof = qualify(a, manifest, lambda probe: run_windows(probe, config, Layer, Rotary, Norm, weights))
    if getattr(a, "prefix_only", False) and not getattr(a, "_prefix_probe", False):
        return
    started, rows, times = time.time(), [], []
    states = []
    features = {}
    if media:
        from mimo_media import window_features
        features, _ = window_features(a, manifest)
    with torch.inference_mode():
        embed = weights.get("model.embed_tokens.weight")
        for w in manifest["windows"]:
            ids = torch.tensor([w["tokens"]], device="cuda")
            state = torch.nn.functional.embedding(ids, embed).cpu()
            for span in w.get("media", []):
                if features[span["key"]].shape != (span["len"], config.hidden_size):
                    raise ValueError("official tower and LM hidden widths differ")
                state[0, span["start"]:span["start"] + span["len"]].copy_(features[span["key"]])
            states.append(state)
        del embed, ids, features
        memory = CheckpointStorage(torch.cuda, weights)
        for layer_id in range(config.num_hidden_layers):
            start = time.time()
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("meta"):
                layer = Layer(config, layer_id, attention_projection_layout=config.attention_projection_layout)
            torch.set_default_dtype(torch.float32)
            layer = layer.to_empty(device="cuda").eval()
            gate = getattr(layer.mlp, "gate", None)
            if gate is not None:
                gate.e_score_correction_bias.data = gate.e_score_correction_bias.data.float()
            load_layer(layer, weights, f"model.layers.{layer_id}.", layer_id)
            kind = layer.attention_type
            rotary = Rotary(config=config, is_swa=kind == "sliding_window_attention").cuda()
            for i, w in enumerate(manifest["windows"]):
                h = states[i].cuda()
                positions = torch.arange(len(w["tokens"]), device="cuda")[None]
                mask = masks(len(w["tokens"]), config.sliding_window)
                selected_mask = mask["sliding_attention" if kind == "sliding_window_attention" else "full_attention"]
                h = layer(h, attention_mask=selected_mask, position_ids=positions,
                          position_embeddings=rotary(h, positions))
                if a.layers is not None and layer_id in a.layers:
                    folder = a.out / "windows" / w["id"]
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / f"layer{layer_id:02d}.bin").write_bytes(h[0].contiguous().view(torch.int16).cpu().numpy().tobytes())
                states[i] = h.cpu()
                del h, mask, selected_mask, positions
            del layer, rotary
            memory.release()
            memory.check(f"layer {layer_id}")
            times.append(time.time() - start)
            print(f"layer {layer_id} ({kind}) {times[-1]:.1f}s ({len(states)} windows)", flush=True)
        norm = Norm(config.hidden_size, eps=config.layernorm_epsilon).cuda().to(torch.bfloat16)
        norm.weight.copy_(weights.get("model.norm.weight"))
        head = weights.get("lm_head.weight").float()
        for i, w in enumerate(manifest["windows"]):
            h = states[i][:, w["score_from"] - 1:len(w["tokens"]) - 1].cuda()
            logits = torch.nn.functional.linear(norm(h).float()[0], head)
            rows.append(write_scored_logits(a.out, w, logits.cpu().numpy()))
            states[i] = None
            del h, logits
    finish_golden(a.out, manifest, rows, snapshot=str(a.snapshot),
        reference="snapshot modeling_mimo_v2.py (trust_remote_code, eager); qkv de-interleaved from TP8 shards; MXFP4 experts widened exactly",
        seconds=time.time() - started, seconds_per_layer=times, snapshot_identity=identity,
        prefix_qualification=proof)

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def mxfp4_to_bf16(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """U8 ``[N, K/2]`` E2M1 pairs (even element low) and U8 ``[N, K/32]`` UE8M0 -> BF16 ``[N, K]``."""
    table = torch.tensor(E2M1 + tuple(-v for v in E2M1), dtype=torch.float32, device=packed.device)
    codes = torch.stack([packed & 0xF, packed >> 4], -1).reshape(packed.shape[0], -1).long()
    values = table[codes]
    power = torch.exp2(scale.float() - 127.0).repeat_interleave(32, 1)
    return (values * power).bfloat16()


def fp8_blocks_to_bf16(value: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """E4M3 ``[R, C]`` times FP32 128x128 block scales ``[ceil(R/128), ceil(C/128)]``."""
    rows, cols = value.shape
    assert scale.shape == (-(-rows // 128), -(-cols // 128)), (value.shape, scale.shape)
    grown = scale.repeat_interleave(128, 0)[:rows].repeat_interleave(128, 1)[:, :cols]
    return (value.float() * grown).bfloat16()


class Weights:
    def __init__(self, snapshot: Path, config):
        self.snapshot = snapshot
        index = json.loads((snapshot / "model.safetensors.index.json").read_text())
        self.index = index["weight_map"]
        self.ckpt_tp = int(index.get("metadata", {}).get("tp_size", 1))
        self.config = config
        self.files: dict[str, object] = {}

    def raw(self, name: str, device: str = "cuda") -> torch.Tensor:
        shard = self.index[name]
        if shard not in self.files:
            self.files[shard] = safe_open(str(self.snapshot / shard), framework="pt", device="cpu")
        return self.files[shard].get_tensor(name).clone().to(device)

    def qkv_shards(self, layer: int) -> tuple[int, int, int]:
        """(q, k, v) rows of one checkpoint TP shard of ``qkv_proj``."""
        c = self.config
        swa = layer >= 0 and c.hybrid_layer_pattern[layer] == 1
        nh = c.swa_num_attention_heads if swa else c.num_attention_heads
        nkv = c.swa_num_key_value_heads if swa else c.num_key_value_heads
        hd = c.swa_head_dim if swa else c.head_dim
        vhd = c.swa_v_head_dim if swa else c.v_head_dim
        tp = self.ckpt_tp
        return nh // tp * hd, max(1, nkv // tp) * hd, max(1, nkv // tp) * vhd

    def get(self, name: str, layer: int = -1) -> torch.Tensor:
        """BF16/FP32 tensors as stored; FP8 and MXFP4 weights widened to BF16."""
        value = self.raw(name)
        if value.dtype == torch.uint8 and ".mlp.experts." in name:
            return mxfp4_to_bf16(value, self.raw(name + "_scale"))
        if value.dtype != torch.float8_e4m3fn:
            return value
        scale = self.raw(name + "_scale_inv").float()
        if not name.endswith("qkv_proj.weight"):
            return fp8_blocks_to_bf16(value, scale)
        q, k, v = self.qkv_shards(layer)
        rows = q + k + v
        blocks = -(-rows // 128)
        assert value.shape[0] == rows * self.ckpt_tp and scale.shape[0] == blocks * self.ckpt_tp, \
            (name, value.shape, scale.shape)
        parts: list[list[torch.Tensor]] = [[], [], []]
        for s in range(self.ckpt_tp):
            w = value[s * rows:(s + 1) * rows]
            g = scale[s * blocks:(s + 1) * blocks]
            # Hugh Madden, issue #3: dequantize across the k/v boundary first.
            shard = fp8_blocks_to_bf16(w, g)
            first = 0
            for i, size in enumerate((q, k, v)):
                parts[i].append(shard[first:first + size])
                first += size
        return torch.cat([torch.cat(p, 0) for p in parts], 0)


def load_layer(layer: torch.nn.Module, weights: Weights, prefix: str, layer_id: int) -> None:
    for key, param in layer.named_parameters():
        name = prefix + key
        if name not in weights.index:
            raise KeyError(f"{name}: no checkpoint tensor")
        with torch.no_grad():
            param.copy_(weights.get(name, layer_id).to(param.dtype))
    release_checkpoint(torch.cuda, weights)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--text", help="prompt text (tokenized with the snapshot tokenizer)")
    p.add_argument("--windows", type=Path, help="pinned fidelity set; scored-row logits; streams saved only with --layers")
    p.add_argument("--text-file", type=Path, help="prompt text from a file")
    p.add_argument("--max-tokens", type=int, help="keep the first N tokens")
    p.add_argument("--layers", type=int, nargs="*", help="layers whose outputs to save (default all)")
    p.add_argument("--stop-after", type=int, help="run only layers 0..N (no logits)")
    p.add_argument("--prefix-only", action="store_true", help="qualify reference prefix arithmetic without running the full panel")
    p.add_argument("--media", action="store_true", help="inject the official BF16 tower for pinned media windows")
    p.add_argument("--media-root", type=Path, help="fixture root (default: windows manifest directory)")
    p.add_argument("--media-features-out", type=Path, help="write immutable probe feature files")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", type=int, default=0)
    a = p.parse_args()
    if a.prefix_only and not a.windows:
        p.error("--prefix-only requires --windows")
    if (a.media or a.media_root or a.media_features_out) and not (a.media and a.windows):
        p.error("media options require --media --windows")

    from tokenizers import Tokenizer
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    torch.cuda.set_device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    from shape_invariant import install
    install()
    config = AutoConfig.from_pretrained(a.snapshot, trust_remote_code=True)
    config._attn_implementation = "eager"
    cls = lambda name: get_class_from_dynamic_module(f"modeling_mimo_v2.{name}", str(a.snapshot))  # noqa: E731
    Layer, Rotary, Norm = cls("MiMoV2DecoderLayer"), cls("MiMoV2RotaryEmbedding"), cls("MiMoV2RMSNorm")
    from shape_invariant import install_eager
    install_eager(sys.modules[Layer.__module__])
    if a.windows:
        if a.text or a.text_file or a.max_tokens or a.stop_after is not None:
            p.error("--windows cannot be combined with legacy text/truncation/stop options")
        weights = Weights(a.snapshot, config)
        a.out.mkdir(parents=True, exist_ok=True)
        run_windows(a, config, Layer, Rotary, Norm, weights)
        return
    text = a.text_file.read_text() if a.text_file else a.text
    tokens = Tokenizer.from_file(str(a.snapshot / "tokenizer.json")).encode(text, add_special_tokens=False).ids
    tokens = tokens[:a.max_tokens] if a.max_tokens else tokens
    weights = Weights(a.snapshot, config)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "tokens.bin").write_bytes(torch.tensor(tokens, dtype=torch.int32).numpy().tobytes())
    n = config.num_hidden_layers
    layers = n if a.stop_after is None else min(a.stop_after + 1, n)
    save = set(range(layers)) if not a.layers else set(a.layers)
    ids = torch.tensor([tokens], device="cuda")
    positions = torch.arange(len(tokens), device="cuda")[None]
    mask = masks(len(tokens), config.sliding_window)
    mask = {"full_attention": mask["full_attention"], "sliding_window_attention": mask["sliding_attention"]}
    started = time.time()
    with torch.inference_mode():
        h = torch.nn.functional.embedding(ids, weights.get("model.embed_tokens.weight"))
        cos_sin = {"full_attention": Rotary(config=config, is_swa=False).cuda()(h, positions),
                   "sliding_window_attention": Rotary(config=config, is_swa=True).cuda()(h, positions)}
        memory = CheckpointStorage(torch.cuda, weights)
        for layer_id in range(layers):
            start = time.time()
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("meta"):
                layer = Layer(config, layer_id, attention_projection_layout=config.attention_projection_layout)
            torch.set_default_dtype(torch.float32)
            layer = layer.to_empty(device="cuda").eval()
            gate = getattr(layer.mlp, "gate", None)
            if gate is not None:
                # e_score_correction_bias is FP32 in the checkpoint (the weight is BF16).
                gate.e_score_correction_bias.data = gate.e_score_correction_bias.data.float()
            load_layer(layer, weights, f"model.layers.{layer_id}.", layer_id)
            kind = layer.attention_type
            h = layer(h, attention_mask=mask[kind], position_ids=positions, position_embeddings=cos_sin[kind])
            if layer_id in save:
                (a.out / f"layer{layer_id:02d}.bin").write_bytes(
                    h[0].contiguous().view(torch.int16).cpu().numpy().tobytes())
            del layer
            memory.release()
            memory.check(f"layer {layer_id}")
            print(f"layer {layer_id} ({kind}) {time.time() - start:.1f}s", flush=True)
        if layers < n:
            return
        norm = Norm(config.hidden_size, eps=config.layernorm_epsilon).cuda().to(torch.bfloat16)
        norm.weight.copy_(weights.get("model.norm.weight"))
        logits = torch.nn.functional.linear(norm(h).float(), weights.get("lm_head.weight").float())
        (a.out / "logits.bin").write_bytes(logits[0].contiguous().cpu().numpy().tobytes())
    argmax = logits[0].argmax(-1)
    next_ok = (argmax[:-1] == ids[0, 1:]).float().mean().item()
    nll = -torch.log_softmax(logits[0, :-1].double(), -1).gather(1, ids[0, 1:, None]).mean().item()
    (a.out / "meta.json").write_text(json.dumps({
        "tokens": len(tokens), "snapshot": str(a.snapshot),
        "reference": f"snapshot modeling_mimo_v2.py (trust_remote_code, eager); qkv de-interleaved from TP{weights.ckpt_tp} shards; "
                     "MXFP4 experts widened exactly",
        "argmax_last": int(argmax[-1]), "next_token_accuracy": next_ok, "mean_nll": nll,
        "seconds": time.time() - started,
    }, indent=1))
    print(f"argmax of last position: {int(argmax[-1])}; next-token accuracy {next_ok:.3f}; mean NLL {nll:.4f}")


if __name__ == "__main__":
    main()

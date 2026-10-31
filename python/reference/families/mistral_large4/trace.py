#!/usr/bin/env python3
"""Mistral Large 4 engine-golden trace: per-layer outputs and logits of one prompt.

Writes what ``cuteafd glm-golden --golden DIR`` reads (the GLM engine serves
Mistral Large 4 as its ``ml4`` variant):

  tokens.bin      u32  [T]
  layerNN.bin     bf16 [T, hidden]   output of layer NN (the residual stream)
  logits.bin      f32  [T, vocab]    softcapped head (only when every layer ran)
  meta.json

One official decoder layer at a time through the shared reference
(``model.py``: exact FP8 dequantization, official transformers 5.22.1 layer),
threading layer 35's NoPE latent to layers 39, 43, ... and the last full
indexer's top-k to shared layers, with the stable lower-key-id top-k tie
rule the engine uses. ``--layers N`` runs the first N layers only (a
CPU-feasible slice with ``--device cpu``); its layer files still compare.

  trace.py --snapshot SNAP --tokens 1,2,3 --out DIR [--layers N] [--device cuda|cpu] [--dtype bfloat16|float32]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import UserDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python/reference"))
from families.mistral_large4 import model  # noqa: E402


def read_tokens(args):
    if args.tokens_file:
        data = args.tokens_file.read_bytes()
        return [int.from_bytes(data[i:i + 4], "little") for i in range(0, len(data), 4)]
    if args.tokens:
        return [int(t) for t in args.tokens.split(",") if t]
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(args.snapshot).encode(args.text)


def carry_shared(config, index, carry):
    """The official ``shared_kv_states`` a layer reads (layer 35's latent for its readers)."""
    shared = UserDict()
    source = config.kv_source_layers[index]
    if source != index:
        k_pass = carry["kv"][source]
        shared["rope" if config.no_rope_layers[source] else "nope"] = (k_pass, k_pass.new_empty((*k_pass.shape[:-1], 0)))
    return shared


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--tokens", help="comma-separated token ids")
    src.add_argument("--tokens-file", type=Path, help="u32 tokens.bin")
    src.add_argument("--text", help="prompt text (the snapshot's own tokenizer)")
    p.add_argument("--layers", type=int, help="run only the first N layers (no logits)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    args = p.parse_args()
    import torch
    dtype = getattr(torch, args.dtype)
    if dtype is torch.float32:
        model.configure_f32()
    reference, config = model.load_official(args.snapshot)
    checkpoint = model.Checkpoint(args.snapshot)
    tokens = read_tokens(args)
    layers = config.num_hidden_layers if args.layers is None else args.layers
    if not 0 < layers <= config.num_hidden_layers or not tokens:
        raise ValueError("layers must be 1..72 and the prompt nonempty")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tokens.bin").write_bytes(b"".join(t.to_bytes(4, "little") for t in tokens))
    outer = model.load_outer(checkpoint, config, device=args.device, dtype=dtype, head=layers == config.num_hidden_layers)
    hidden = torch.nn.functional.embedding(torch.tensor([tokens], device=args.device), outer["embed"])
    carry = {"kv": {}, "topk": None}
    started = time.time()
    with torch.no_grad(), model.stable_ties(reference):
        for index in range(layers):
            layer = model.load_layer(checkpoint, config, index, device=args.device, dtype=dtype)
            shared = carry_shared(config, index, carry)
            hidden, topk = model.layer_forward(layer, config, hidden, outer["rotary"], shared, carry["topk"])
            if config.indexer_types[index] == "full":
                carry["topk"] = topk
            if index in config.kv_source_layers[index + 1:]:
                kind = "rope" if config.no_rope_layers[index] else "nope"
                carry["kv"][index] = shared[kind][0]
            (args.out / f"layer{index:02d}.bin").write_bytes(hidden[0].bfloat16().contiguous().view(torch.uint16)
                                                             .cpu().numpy().tobytes())
            del layer
            print(f"layer {index:2d}: {time.time() - started:.1f} s", flush=True)
        if layers == config.num_hidden_layers:
            logits = model.logits(outer["norm"](hidden), outer["head"], config)
            (args.out / "logits.bin").write_bytes(logits[0].float().contiguous().cpu().numpy().tobytes())
    (args.out / "meta.json").write_text(json.dumps({
        "family": "mistral_large4", "tokens": len(tokens), "layers": layers, "dtype": args.dtype,
        "reference": "official transformers MistralLarge4TextDecoderLayer via families/mistral_large4/model.py; "
                     "untiled, stable lower-key-id index top-k ties", **model.source_identity()}, indent=1))


if __name__ == "__main__":
    main()

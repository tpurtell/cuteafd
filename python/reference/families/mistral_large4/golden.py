#!/usr/bin/env python3
"""Mistral Large 4 FP8 text golden, layer-major, strict F32 (layers 0-71).

The official transformers 5.22.1 decoder layer (families/mistral_large4/model.py)
runs every window per layer visit. FP8 E4M3 block-128 weights are dequantized
exactly (power-of-two qscale) to F32; BF16 tensors are upcast exactly; one F32
expert matrix exists only during its official call. Cross-layer state the
official model threads through its forward (layer 35's NoPE latent for layers
39, 43, ...; the last full indexer's top-k for shared layers) is persisted per
window with the residual stream. Prefix qualification and an untiled
official-geometry arm precede the panel.
"""
from __future__ import annotations

import argparse
from collections import UserDict
from contextlib import contextmanager, nullcontext
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python/reference"))
import layer_major
from families.mistral_large4 import model


class Family:
    name = "mistral_large4"

    def __init__(self, snapshot):
        import torch
        self.reference, self.config = model.load_official(snapshot)
        self.checkpoint = model.Checkpoint(snapshot)
        self.layers = self.config.num_hidden_layers
        self.dtype = torch.float32
        self.outer = None
        self.rotary = None
        self.source = {i: s for i, s in enumerate(self.config.kv_source_layers) if s != i}
        self.stores = {s for s in self.source.values()}
        self.last_reader = {s: max(i for i, x in self.source.items() if x == s) for s in self.stores}

    def configure(self):
        return model.configure_f32()

    def provenance(self):
        return {"checkpoint": model.CHECKPOINT, "revision": model.REVISION, **model.source_identity(),
                "layer_major_sha256": layer_major.digest(layer_major.__file__),
                "golden_sha256": layer_major.digest(__file__)}

    def check_layer(self, index):
        model.check_layer(self.checkpoint, self.config, index)

    def tensors(self, index):
        return [k for k in self.checkpoint.index if k.startswith(f"layers.{index}.")]

    def missing(self, index):
        return self.checkpoint.missing(self.tensors(index))

    def layer_bytes(self, index):
        total = 0
        for name in self.tensors(index):
            _, descriptor = self.checkpoint.descriptor(name)
            size = descriptor["data_offsets"][1] - descriptor["data_offsets"][0]
            routed = ".experts." in name
            # Routed experts stay FP8 (one F32 matrix per call); dense becomes F32.
            total += size if routed else size * {"F8_E4M3": 4, "BF16": 2}.get(descriptor["dtype"], 1)
        return total

    def work_bytes(self, tokens):
        h, heads = self.config.hidden_size, self.config.num_attention_heads
        return (tokens * h * 4 * 16 + 128 * heads * tokens * 4 * 3 + tokens * tokens * (4 * 3 + 1)
                + tokens * self.config.index_topk * 8 + 2 * self.config.moe_intermediate_size * h * 4)

    def state_bytes(self, tokens):
        return tokens * (self.config.hidden_size * 4 + self.config.index_topk * 4 + self.config.kv_lora_rank * 4)

    def ensure_outer(self):
        if self.outer is None:
            self.outer = model.load_outer(self.checkpoint, self.config, device=layer_major.DEVICE, dtype=self.dtype)
            self.rotary = self.outer["rotary"]

    def embed(self, tokens):
        import torch
        self.ensure_outer()
        ids = torch.tensor([tokens], device=layer_major.DEVICE, dtype=torch.long)
        return torch.nn.functional.embedding(ids, self.outer["embed"])

    def release_outer(self):
        # The rotary table (LeRoPE alpha, F32) is needed by every layer.
        self.rotary = self.outer["rotary"]
        self.outer = None

    def load_layer(self, index):
        if self.rotary is None:
            self.ensure_outer()
            self.release_outer()
        return model.load_layer(self.checkpoint, self.config, index, device=layer_major.DEVICE, dtype=self.dtype)

    def release_layer(self, module):
        self.checkpoint.close()

    def geometry(self):
        return model.geometry(self.reference)

    def official_geometry(self):
        return model.stable_ties(self.reference)

    def observe(self, module, index):
        return nullcontext(None)

    def forward(self, module, index, hidden, carry):
        import torch
        shared = UserDict()
        reuse = []
        if index in self.source:
            k_pass = carry["kv_nope"]
            shared["rope" if self.config.no_rope_layers[self.source[index]] else "nope"] = (
                k_pass, k_pass.new_empty((*k_pass.shape[:-1], 0)))
        prev = carry.get("topk")
        hidden, topk = model.layer_forward(module, self.config, hidden, self.rotary, shared, prev)
        out = {}
        following = index + 1 < self.layers and self.config.indexer_types[index + 1] == "shared"
        if following:
            out["topk"] = topk
            if self.config.indexer_types[index] == "shared":
                reuse.append("topk")
        # The NoPE latent lives from its source layer to its last reader.
        for source in self.stores:
            if source <= index < self.last_reader[source]:
                if index == source:
                    kind = "rope" if self.config.no_rope_layers[source] else "nope"
                    k_pass, k_rot = shared[kind]
                    if k_rot.shape[-1]:
                        raise ValueError("shared KV source unexpectedly carries RoPE keys")
                    out["kv_nope"] = k_pass
                else:
                    out["kv_nope"] = carry["kv_nope"]
                    reuse.append("kv_nope")
        return hidden, out, tuple(reuse)

    def load_final(self):
        self.ensure_outer()

    def final(self, hidden):
        import shape_invariant
        return model.logits(self.outer["norm"](hidden), self.outer["head"], self.config,
                            linear=shape_invariant.linear)

    def release_final(self):
        self.outer = None
        self.checkpoint.close()

    def write_extras(self, root, index, extras, windows):
        raise AssertionError("no Mistral sidecar")

    def golden_meta(self):
        return {"reference": "official transformers 5.22.1 MistralLarge4TextDecoderLayer; fixed-M128 linears; "
                             "128-row causal eager tiles; DSA indexer on 128-row tiles with stable lower-id ties",
                "expert_storage": "FP8 bytes + qscale resident; one F32 expert matrix per official F.linear call",
                "index_topk_ties": "relu scores tie at 0.0; selection by score, ties to lower key id "
                                   "(official topk tie order is unspecified)",
                "root_arithmetic": {
                    "source": "official transformers 5.22.1 mistral_large4 modeling (pinned SHA256), consolidated "
                              "checkpoint tensors",
                    "arithmetic": "Reference computed in F32 from the official FP8 checkpoint: E4M3 block-128 weights "
                                  "times their power-of-two qscale_weight (exact in F32; Transformers Fp8Dequantize "
                                  "arithmetic), BF16 tensors upcast exactly; F32 activations (no activation "
                                  "re-quantization), norms, SSMax, LeRoPE/YaRN, router, DSA indexer, MLA attention and "
                                  "softcapped head; TF32 disabled, highest matmul precision; fixed-M128 linears and "
                                  "128-row causal tiles; MTP drafter and cache off."}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "windows", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prefix-only", action="store_true", help="qualify the prefix and official arm, then stop")
    parser.add_argument("--wait-shards", type=float, default=0,
                        help="seconds to wait for a layer's shards to finish downloading")
    parser.add_argument("--layers", type=int, default=None)
    args = parser.parse_args()
    if args.layers is not None:
        raise ValueError("partial layers cannot qualify a reference")
    layer_major.run(args, Family(args.snapshot))


if __name__ == "__main__":
    main()

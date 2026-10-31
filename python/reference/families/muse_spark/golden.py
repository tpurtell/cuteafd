#!/usr/bin/env python3
"""Muse Spark Open BF16 text golden, layer-major, strict F32 (layers 0-61).

The official transformers 5.22.1 decoder layer (families/muse_spark/model.py)
runs every window per layer visit; BF16 checkpoint tensors are upcast exactly to
F32; one F32 expert matrix exists only during its official call. Prefix
qualification and an untiled official-geometry arm precede the panel. Router
selections (bias-chosen top-8 ids, int16, and renormalized weights, F32) are
written per layer beside the golden; they never feed back into the logits.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python/reference"))
import layer_major
from families.muse_spark import model


class Family:
    name = "muse_spark"

    def __init__(self, snapshot):
        import torch
        self.reference, self.config = model.load_official(snapshot)
        self.checkpoint = model.Checkpoint(snapshot)
        self.layers = self.config.num_hidden_layers
        self.dtype = torch.float32
        self.outer = None
        with torch.device(layer_major.DEVICE):
            self.rotary = self.reference.MuseSparkTextRotaryEmbedding(self.config)

    def configure(self):
        return model.configure_f32()

    def provenance(self):
        return {"checkpoint": model.CHECKPOINT, "revision": model.REVISION, **model.source_identity(),
                "layer_major_sha256": layer_major.digest(layer_major.__file__),
                "golden_sha256": layer_major.digest(__file__)}

    def check_layer(self, index):
        model.check_layer(self.checkpoint, self.config, index)

    def tensors(self, index):
        return [k for k in self.checkpoint.index if k.startswith(f"{model.PREFIX}layers.{index}.")]

    def missing(self, index):
        return self.checkpoint.missing(self.tensors(index))

    def layer_bytes(self, index):
        total = 0
        for name in self.tensors(index):
            _, descriptor = self.checkpoint.descriptor(name)
            size = descriptor["data_offsets"][1] - descriptor["data_offsets"][0]
            # Experts stay BF16 (one F32 matrix per call); dense tensors become F32.
            total += size if ".experts." in name and "norm" not in name else size * 2
        return total

    def work_bytes(self, tokens):
        h = self.config.hidden_size
        # Residual, q/gate/o activations, routed copies, one 128-row tile of [heads, keys]
        # scores, the additive [T, T] mask and one F32 expert gate_up matrix.
        return (tokens * h * 4 * 12 + 128 * self.config.num_attention_heads * tokens * 4 * 3
                + tokens * tokens * 4 + 2 * self.config.moe_intermediate_size * self.config.moe_hidden_size * 4)

    def state_bytes(self, tokens):
        return tokens * self.config.hidden_size * 4

    def embed(self, tokens):
        import torch
        if self.outer is None:
            self.outer = model.load_outer(self.checkpoint, self.config, device=layer_major.DEVICE, dtype=self.dtype)
        ids = torch.tensor([tokens], device=layer_major.DEVICE, dtype=torch.long)
        return self.outer["embed"](ids)

    def release_outer(self):
        self.outer = None

    def load_layer(self, index):
        return model.load_layer(self.checkpoint, self.config, index, device=layer_major.DEVICE, dtype=self.dtype)

    def release_layer(self, module):
        self.checkpoint.close()

    def geometry(self):
        return model.geometry(self.reference)

    def official_geometry(self):
        return nullcontext()

    @contextmanager
    def observe(self, module, index):
        with model.capture_routes(module) as records:
            box = {}
            yield box
            if records:
                indices, weights = records[-1]
                box["ids"] = indices.to("cpu").to(dtype=__import__("torch").int16).numpy()
                box["weights"] = weights.to("cpu").float().numpy()

    def forward(self, module, index, hidden, carry):
        hidden = model.layer_forward(module, self.config, index, hidden, self.rotary)
        return hidden, {}, ()

    def write_extras(self, root, index, extras, windows):
        """Router sidecar: routes/layerNNN/{wid}.ids.bin int16 [T,8], .weights.bin F32 [T,8]."""
        import numpy as np
        folder = Path(root) / ("layer%03d" % index)
        folder.mkdir(parents=True, exist_ok=True)
        lengths = {w["id"]: len(w["tokens"]) for w in windows}
        entries = {}
        for wid, box in extras.items():
            ids, weights = box["ids"], box["weights"]
            if ids.shape != (lengths[wid], self.config.num_experts_per_tok) or weights.shape != ids.shape:
                raise ValueError("router sidecar shape differs from window tokens")
            for suffix, array in (("ids", ids.astype("<i2")), ("weights", weights.astype("<f4"))):
                path = folder / f"{wid}.{suffix}.bin"
                array.tofile(path)
                entries[f"{wid}.{suffix}"] = hashlib.sha256(path.read_bytes()).hexdigest()
        layer_major.atomic_json(folder / "index.json", {
            "schema": "cuteafd.muse.routes/1", "layer": index, "top_k": self.config.num_experts_per_tok,
            "experts": self.config.num_local_experts, "ids_dtype": "<i2", "weights_dtype": "<f4",
            "selection": "sigmoid(router F32 x output_multiplier) + e_score_correction_bias, top-k unsorted",
            "weights": "renormalized sigmoid scores (eps 1e-15) x routed_scaling_factor", "files": entries})

    def load_final(self):
        if self.outer is None:
            self.outer = model.load_outer(self.checkpoint, self.config, device=layer_major.DEVICE, dtype=self.dtype)

    def final(self, hidden):
        import shape_invariant
        return model.logits(self.outer["norm"](hidden), self.outer["head"], self.config,
                            linear=shape_invariant.linear)

    def release_final(self):
        self.outer = None
        self.checkpoint.close()

    def golden_meta(self):
        return {"reference": "official transformers 5.22.1 MuseSparkTextDecoderLayer; fixed-M128 linears; "
                             "128-row causal (sliding 2048 / NoPE full) eager tiles",
                "expert_storage": "BF16 stack resident; one F32 expert matrix per official F.linear call",
                "root_arithmetic": {
                    "source": "official transformers 5.22.1 muse_spark modeling (pinned SHA256), checkpoint BF16/F32 tensors",
                    "arithmetic": "Reference computed in F32 from the native BF16 checkpoint: every BF16 tensor upcast "
                                  "exactly to F32; native F32 router weight and e_score_correction_bias; F32 activations, "
                                  "norms, residual gates, router, attention and softcapped head; TF32 disabled, highest "
                                  "matmul precision; fixed-M128 linears and 128-row causal attention tiles; MTP/cache off."}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "windows", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--routes", type=Path, help="router sidecar directory (full panel only)")
    parser.add_argument("--prefix-only", action="store_true", help="qualify the prefix and official arm, then stop")
    parser.add_argument("--wait-shards", type=float, default=0,
                        help="seconds to wait for a layer's shards to finish downloading")
    parser.add_argument("--layers", type=int, default=None)
    args = parser.parse_args()
    if args.layers is not None:
        raise ValueError("partial layers cannot qualify a reference")
    family = Family(args.snapshot)
    layer_major.run(args, family, extras_root=args.routes)


if __name__ == "__main__":
    main()

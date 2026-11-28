"""Hy4-flash MC forward for QuantMLA calibration and compact-view transfer."""
from __future__ import annotations

import json

import torch

from inkling2 import numerical
from . import forward, recipe
from .engram import mc_manifest


class MCReader(forward.reference.Checkpoint):
    """Read the selected MC core and shared K4 experts from that view's verified index."""

    def __init__(self, checkpoint, device, *, view):
        if view not in ("original", "k8-compact"):
            raise ValueError(f"unsupported Hy4-flash QMLA view: {view}")
        super().__init__(checkpoint, device=device)
        mc = json.loads((self.snapshot / "cuteafd-mc.json").read_bytes())
        mc_manifest.validate(mc, self.snapshot, view=view)
        variants = {v["name"]: v for v in mc["variants"]}
        self.variant = variants[view]
        self.original_fingerprint = variants["original"]["fingerprint"]
        self.source = mc["source"]
        document = json.loads((self.snapshot / self.variant["config"]).read_bytes())
        storage = document.pop("quantization_config")
        if (storage["checkpoint_format"], storage["bits"], storage["codebook"]) != ("exl3", 4, "mul1") or (
            storage["safetensors_index"] != self.variant["index"]
            or mc["exl3_codebook"] != {"name": "mul1", "multiplier": recipe.MUL1_MULTIPLIER}
        ):
            raise ValueError("Hy4-flash MC requires shared K4 MUL1 experts in the selected view index")
        expected_bits = 16 if view == "original" else 8
        if self.variant["core_bits"] != expected_bits or storage.get("core_bits", 16) != expected_bits:
            raise ValueError(f"Hy4-flash MC {view} core width differs from {expected_bits}")
        if view == "original" and mc["original_core_dtype"] != "BF16":
            raise ValueError("Hy4-flash original view requires a BF16 core")

        self.config = forward.reference.Config.parse(document)
        self.index = json.loads((self.snapshot / self.variant["index"]).read_bytes())["weight_map"]
        self.core = frozenset(self.variant["core_tensors"])

    def close(self):
        super().close()
        # Each layer's expert shards bring thousands of parsed tensor descriptors.
        # Retire those alongside their mappings; raw() reloads both on demand.
        self.headers.clear()
        self.header_offsets.clear()

    def reconstruct(self, module, shape, bits):
        packed = {suffix: self.raw(f"{module}.{suffix}") for suffix in recipe.SUFFIXES}
        if packed["trellis"].shape[-1] != bits * 16:
            raise ValueError(f"wrong bit width for {module}: expected K{bits}")
        return numerical.reconstruct_projection(
            packed, shape, device=str(self.device), dtype=torch.bfloat16, codebook="mul1"
        ).float()

    def load_layer(self, index):
        # Release the previous layer's shard mappings. Any still-live tensor
        # (e.g. the global embedding) retains its own mapped storage.
        self.close()
        prefix = f"model.layers.{index}."
        weights = {}
        for name, (shape, _dtype) in forward.reference.tensor_contract(self.config, index).items():
            leaf = name.removeprefix(prefix)
            if leaf in forward.reference.EXPERTS:
                continue  # The MC stores separate, per-expert K4 modules.
            module = name.removesuffix(".weight")
            if name in self.core:
                if self.descriptor(name)["dtype"] != "BF16":
                    raise ValueError(f"expected native BF16 MC core tensor {name}")
                weights[leaf] = self.load(name)
            elif name.endswith(".weight") and module + ".trellis" in self.core:
                weights[leaf] = self.reconstruct(module, shape, 8)
            elif leaf == forward.reference.TABLE:
                weights[leaf] = self.raw(name)  # Map large native Engram tables and gather only used rows.
            else:
                weights[leaf] = self.load(name, native=leaf == "engram_layer.lookup_table")
        return forward.Layer(index, prefix, weights)


class MCModel(forward.Model):
    def routed_expert_weights(self, index, latent, layer):
        prefix = layer.prefix + f"mlp.experts.{index}."
        width = latent.shape[-1]
        inner = self.config.moe_intermediate_size
        return (
            self.reader.reconstruct(prefix + "gate_proj", (inner, width), 4),
            self.reader.reconstruct(prefix + "up_proj", (inner, width), 4),
            self.reader.reconstruct(prefix + "down_proj", (width, inner), 4),
        )

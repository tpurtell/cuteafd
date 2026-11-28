"""Layer-major GLM reference weights from one immutable MC view (native or compact core)."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

from core_export import MUL1
from mistral_large4.core_k6.numerical import fork_module

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/lib"))
import mc_manifest

native = fork_module("utils.glm5_flash_core")
SUFFIXES = ("trellis", "suh", "svh", "mul1")


class GlmFlashMCSource(native.GlmFlashBF16Source):
    """Expose only the selected view's operands to the pinned native layer loader.

    The native loader allocates the fused routed weights once per layer and fills
    one projection at a time. Reconstruct directly on that device so each decoded
    expert matrix is copied device-to-device into its slice and then released.
    """
    def __init__(self, checkpoint, view="original"):
        self.snapshot = Path(checkpoint).resolve()
        self._reader = self._reader_context = self._reader_shard = None
        if view not in ("original", "k8-compact"):
            raise ValueError(f"unsupported calibration view {view!r}")
        manifest = json.loads((self.snapshot / "cuteafd-mc.json").read_bytes())
        mc_manifest.validate(manifest, self.snapshot, view=view)
        if manifest["exl3_codebook"] != {"name": "mul1", "multiplier": MUL1} or manifest["original_core_dtype"] != "BF16":
            raise ValueError("expected BF16 original core and EXL3 MUL1 declaration")
        self.variant = next(v for v in manifest["variants"] if v["name"] == view)
        self.original_fingerprint = next(v["fingerprint"] for v in manifest["variants"] if v["name"] == "original")
        self.source = manifest["source"]
        self.document = json.loads((self.snapshot / self.variant["config"]).read_bytes())
        storage = self.document.pop("quantization_config")
        self.config = self.document["text_config"]
        if (self.config.pop("quantization_config") != storage or
                self.document["model_type"] != "glm5_next" or self.config["model_type"] != "glm5_next_text" or
                self.config["qk_rope_head_dim"] != 0 or self.config["hc_mult"] != 4 or
                (storage["checkpoint_format"], storage["bits"], storage["codebook"], storage["codebook_multiplier"],
                 storage.get("core_bits"), storage["safetensors_index"]) !=
                ("exl3", 4, "mul1", MUL1, 8 if view == "k8-compact" else None, self.variant["index"])):
            raise ValueError("GLM Flash configuration or selected view storage differs")
        self.index = json.loads((self.snapshot / self.variant["index"]).read_bytes())
        weights = self.index["weight_map"]
        metadata = json.loads((self.snapshot / self.variant["quantize"]).read_bytes())["tensor_storage"]
        core = set(self.variant["core_tensors"])
        modules = {name.removesuffix(".trellis") for name in weights if name.endswith(".trellis")}
        if modules != set(metadata):
            raise ValueError("compact tensor storage and indexed EXL3 modules differ")
        self.packed = {}
        self.weight_map = dict(weights)
        for module, details in metadata.items():
            name = module + ".weight"
            trellis = module + ".trellis"
            expert = ".mlp.experts." in module
            bits = 4 if expert else 8
            if (name in weights or details["bits_per_weight"] != bits or details["mul1_multiplier"] != MUL1
                    or (not expert and (view != "k8-compact" or trellis not in core))
                    or (expert and trellis in core)
                    or any(module + "." + suffix not in weights for suffix in SUFFIXES)):
                raise ValueError(f"invalid GLM {view} packed operand: {module}")
            self.packed[name] = (module, (details["out_features"], details["in_features"]), bits)
            # The native layer loader schedules logical weights by their physical shard.
            self.weight_map[name] = weights[trellis]
        if any(".mlp.experts." in name and name.endswith(".weight") for name in weights):
            raise ValueError("native routed expert weights found in the selected MC index")
        self.decode_device = None

    def load_block(self, layer, device="cuda:0", *, include_experts=False):
        self.decode_device = device
        try:
            return super().load_block(layer, device, include_experts=include_experts)
        finally:
            self.decode_device = None

    def tensor(self, name, device=None):
        packed = self.packed.get(name)
        if packed is None:
            return super().tensor(name, device or "cpu")
        module, shape, bits = packed
        read = super().tensor
        pieces = {suffix: read(f"{module}.{suffix}") for suffix in SUFFIXES}
        if pieces["trellis"].shape[-1] != bits * 16:
            raise ValueError(f"physical trellis width differs for {module}: expected K{bits}")
        # reconstruct_core_projection validates native shape, dtypes, scales and marker.
        return fork_module("utils.kimi_k3_core_format").reconstruct_core_projection(
            pieces, shape, device=str(device or self.decode_device or "cpu"), dtype=torch.bfloat16, codebook="mul1")

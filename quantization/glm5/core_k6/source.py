"""Owned, one-shard-at-a-time BF16 source reads and native GLM blocks."""
from __future__ import annotations

import json
from pathlib import Path


class GlmBF16Source:
    def __init__(self, snapshot):
        self.snapshot = Path(snapshot).resolve()
        self.config = json.loads((self.snapshot / "config.json").read_text())
        self.index = json.loads((self.snapshot / "model.safetensors.index.json").read_text())
        self.weight_map = self.index["weight_map"]
        if self.config.get("quantization_config"):
            raise ValueError("pilot requires the original BF16 checkpoint")
        self._reader = self._shard = None

    def close(self):
        if self._reader is not None:
            self._reader.__exit__(None, None, None)
        self._reader = self._shard = None

    def tensor(self, name, device="cpu"):
        from safetensors import safe_open
        shard = self.weight_map[name]
        if shard != self._shard:
            self.close()
            self._reader = safe_open(str(self.snapshot / shard), framework="pt", device="cpu")
            self._reader.__enter__()
            self._shard = shard
        # Never hand a file mapping directly to a CUDA transfer. Only one
        # matrix's owned host buffer lives through the synchronous copy.
        owned = self._reader.get_tensor(name).clone()
        return owned.to(device=device)

    def native_config(self):
        from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
        config = GlmMoeDsaConfig.from_dict(self.config)
        config._attn_implementation = "sdpa"
        config.use_cache = False
        return config

    def load_block(self, layer, device="cuda:0"):
        import torch
        from torch import nn
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaDecoderLayer
        config = self.native_config()
        with torch.device("meta"):
            block = GlmMoeDsaDecoderLayer(config, layer)
        prefix = f"model.layers.{layer}."
        state = block.state_dict()
        fused = {key for key in state if key.startswith("mlp.experts.")}
        assignments = []
        for key, value in state.items():
            if key in fused:
                continue
            name = prefix + key
            assignments.append((self.weight_map[name], name, key))
        for _, name, key in sorted(assignments):
            tensor = self.tensor(name, device)
            if tensor.shape != state[key].shape:
                raise ValueError(f"native state shape differs: {name}")
            # Match HF's declared keep-in-FP32 execution module. Its source
            # BF16 bytes remain unchanged in any eventual checkpoint.
            if key == "self_attn.indexer.weights_proj.weight":
                tensor = tensor.float()
            parent, _, leaf = key.rpartition(".")
            module = block.get_submodule(parent) if parent else block
            if leaf in module._buffers:
                module._buffers[leaf] = tensor
            else:
                module._parameters[leaf] = nn.Parameter(tensor, requires_grad=False)
        if fused:
            if fused != {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"}:
                raise ValueError("native GLM expert storage differs")
            experts = block.mlp.experts
            experts.gate_up_proj = nn.Parameter(torch.empty(state["mlp.experts.gate_up_proj"].shape,
                dtype=torch.bfloat16, device=device), requires_grad=False)
            experts.down_proj = nn.Parameter(torch.empty(state["mlp.experts.down_proj"].shape,
                dtype=torch.bfloat16, device=device), requires_grad=False)
            inter = config.moe_intermediate_size
            assignments = []
            for expert in range(config.n_routed_experts):
                for part in ("gate_proj", "up_proj", "down_proj"):
                    name = f"{prefix}mlp.experts.{expert}.{part}.weight"
                    assignments.append((self.weight_map[name], name, expert, part))
            for _, name, expert, part in sorted(assignments):
                target = (experts.down_proj[expert] if part == "down_proj" else
                          experts.gate_up_proj[expert, :inter if part == "gate_proj" else 2*inter])
                if part == "up_proj":
                    target = experts.gate_up_proj[expert, inter:]
                tensor = self.tensor(name)
                if tensor.dtype != torch.bfloat16 or tensor.shape != target.shape:
                    raise ValueError(f"expert geometry or dtype differs: {name}")
                target.copy_(tensor)
        for key, value in (*block.named_parameters(), *block.named_buffers()):
            if value.is_meta:
                raise ValueError(f"native GLM state remains meta: {key}")
        self.close()
        return block.eval()

    def rotary(self, device="cuda:0"):
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaRotaryEmbedding
        return GlmMoeDsaRotaryEmbedding(self.native_config()).to(device)

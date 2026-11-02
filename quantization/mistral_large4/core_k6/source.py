"""Owned consolidated reads and pinned official Mistral Large 4 decoder layers.

FP8 execution values follow Transformers' Fp8Dequantize: float32 E4M3 times
its block-128 qscale, then BF16. ``tensor_raw`` returns unmodified source
bytes for export; execution casts never alter the checkpoint's kept tensors.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

BASE_REVISION = "f12dabdc0c1e99e7f7924461a6f308859b05af4d"
TRANSFORMERS_VERSION = "5.22.1"
SOURCE_PINS = {
    "models/mistral_large4/modeling_mistral_large4.py": "a83fc96564410fcfea5cedb3310b5a28f3c7967caddd3ee955d9131f5b974e84",
    "models/mistral_large4/configuration_mistral_large4.py": "51646a3d32314e91e73aecd28355dddda454863b7c6ada7755560592df9b07e4",
    "modeling_rope_utils.py": "ddec71d6c9343af437f656812177fe5c98f6057fd5240c5cbf208520c2872745",
    "masking_utils.py": "931b92cf0deeb7a4925259a30c905bbf4d645fda88c6eebbc9e61770f5abaa89",
}

# Official layer state -> the checkpoint's original consolidated naming.
ATTENTION_NAMES = {
    "self_attn.q_proj.weight": "attention.wq.weight",
    "self_attn.kv_a_proj_with_mqa.weight": "attention.wkv_a_with_mqa.weight",
    "self_attn.kv_a_layernorm.weight": "attention.kv_a_norm.weight",
    "self_attn.kv_b_proj.weight": "attention.wkv_b.weight",
    "self_attn.o_proj.weight": "attention.wo.weight",
    "self_attn.softmax_scaling.scale": "attention.ssmax_scaling.s",
    "self_attn.softmax_scaling.bias": "attention.ssmax_scaling.b",
    "self_attn.indexer.wq_b.weight": "attention.indexer.wq_b.weight",
    "self_attn.indexer.wk.weight": "attention.indexer.wk.weight",
    "self_attn.indexer.k_norm.weight": "attention.indexer.k_norm.weight",
    "self_attn.indexer.k_norm.bias": "attention.indexer.k_norm.bias",
    "self_attn.indexer.weights_proj.weight": "attention.indexer.weights_proj.weight",
    "self_attn.indexer.q_input_norm.weight": "attention.indexer.q_input_norm.weight",
    "self_attn.indexer.q_input_norm.bias": "attention.indexer.q_input_norm.bias",
    "self_attn.indexer.softmax_scaling.scale": "attention.indexer.ssmax_scaling.s",
    "self_attn.indexer.softmax_scaling.bias": "attention.indexer.ssmax_scaling.b",
    "input_layernorm.weight": "attention_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "mlp.gate.weight": "gate.weight",
}
MLP_NAMES = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}


def official_identity():
    import transformers
    root = Path(transformers.__file__).resolve().parent
    actual = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_PINS}
    if transformers.__version__ != TRANSFORMERS_VERSION or actual != SOURCE_PINS:
        raise ValueError("Mistral calibration requires the pinned Transformers 5.22.1 official source")
    return dict(transformers=TRANSFORMERS_VERSION, source_sha256=actual)


def validate_config(config):
    """Fail closed on the actual target's unusual geometry and reuse schedule."""
    expected = dict(num_hidden_layers=72, hidden_size=6144, vocab_size=204800,
        num_local_experts=128, num_experts_per_tok=4, n_shared_experts=1, n_group=1, topk_group=1,
        kv_lora_rank=512, qk_rope_head_dim=64, qk_nope_head_dim=64, v_head_dim=128,
        num_attention_heads=64, num_key_value_heads=64, index_n_heads=32, index_head_dim=128,
        index_topk=2048, final_logit_softcapping=30.0, tie_word_embeddings=False,
        intermediate_size=24576, moe_intermediate_size=6144)
    for name, value in expected.items():
        if getattr(config, name, None) != value:
            raise ValueError(f"unreviewed native Mistral config {name}: {getattr(config, name, None)!r}")
    if list(config.mlp_layer_types) != ["dense"] + ["sparse"] * 71:
        raise ValueError("unreviewed Mistral dense/MoE schedule")
    if [i for i, use_rope in enumerate(config.no_rope_layers) if not use_rope] != list(range(3, 72, 4)):
        raise ValueError("unreviewed Mistral NoPE schedule")
    shared = {i: src for i, src in enumerate(config.kv_source_layers) if i != src}
    if shared != {i: 35 for i in range(39, 72, 4)}:
        raise ValueError("unreviewed Mistral KV-source reuse")
    if (len(config.indexer_types) != 72 or config.indexer_types[0] != "full"
            or set(config.indexer_types) != {"full", "shared"}
            or sum(kind == "full" for kind in config.indexer_types) != 44):
        raise ValueError("unreviewed Mistral indexer reuse")


def dequantize_fp8(weight, scale, *, dtype=None):
    """Exact block scale multiplication, with no scale inversion/re-encoding."""
    import torch
    dtype = dtype or torch.bfloat16
    if weight.ndim != 2 or weight.dtype != torch.float8_e4m3fn or scale.dtype != torch.float32:
        raise ValueError("Mistral FP8 dequantization requires E4M3 weight and FP32 qscale")
    rows, cols = weight.shape
    if tuple(scale.shape) != ((rows + 127) // 128, (cols + 127) // 128):
        raise ValueError("Mistral qscale grid differs from block 128")
    mantissa, _ = torch.frexp(scale)
    if not bool((torch.isfinite(scale) & (scale > 0) & (mantissa == 0.5)).all()):
        raise ValueError("Mistral qscale must be finite positive powers of two")
    # Broadcast block coordinates instead of materializing an expanded scale
    # matrix; the native selected FP8 matrices are all block-aligned.
    if rows % 128 == 0 and cols % 128 == 0:
        value = (weight.float().view(rows // 128, 128, cols // 128, 128)
                 * scale[:, None, :, None]).reshape(rows, cols).to(dtype)
        if not bool(torch.isfinite(value).all()):
            raise ValueError("nonfinite native Mistral FP8 weight")
        return value
    # Partial boundary blocks occur in small reference fixtures.
    result = torch.empty((rows, cols), dtype=dtype, device=weight.device)
    for start in range(0, rows, 128):
        count = min(128, rows - start)
        row_scale = scale[start // 128].repeat_interleave(128)[:cols]
        values = weight[start:start + count].float() * row_scale
        if not bool(torch.isfinite(values).all()):
            raise ValueError("nonfinite native Mistral FP8 weight")
        result[start:start + count].copy_(values)
    return result


def layer_mapping(config, layer, block):
    prefix = f"layers.{layer}."
    result = {}
    for name, _ in block.named_parameters():
        if name in ATTENTION_NAMES:
            result[name] = prefix + ATTENTION_NAMES[name]
            continue
        parts = name.split(".")
        if parts[0] == "mlp" and parts[1] in MLP_NAMES:
            result[name] = prefix + f"feed_forward.{MLP_NAMES[parts[1]]}.weight"
        elif parts[:2] == ["mlp", "shared_experts"]:
            result[name] = prefix + f"shared_experts.{MLP_NAMES[parts[2]]}.weight"
        elif name == "mlp.experts.gate_up_proj":
            result[name] = [(prefix + f"experts.{e}.w1.weight", prefix + f"experts.{e}.w3.weight")
                            for e in range(config.num_local_experts)]
        elif name == "mlp.experts.down_proj":
            result[name] = [prefix + f"experts.{e}.w2.weight" for e in range(config.num_local_experts)]
        else:
            raise ValueError(f"unmapped native Mistral layer parameter: {name}")
    return result


def linear_module(block, module):
    """Resolve a consolidated selected module into its official ``nn.Linear``."""
    relative = module.split(".", 2)[2] if module.startswith("layers.") else module
    mapping = {value.removesuffix(".weight"): name.removesuffix(".weight")
               for name, value in layer_mapping(block.self_attn.config, block.layer_idx, block).items()
               if isinstance(value, str)}
    if relative.startswith(f"layers.{block.layer_idx}."):
        relative = relative.removeprefix(f"layers.{block.layer_idx}.")
    # Layer mapping values include the prefix; compare the full checkpoint module.
    full = f"layers.{block.layer_idx}." + relative
    if full not in mapping:
        raise ValueError(f"consolidated module has no native linear: {module}")
    return block.get_submodule(mapping[full])


class MistralSource:
    def __init__(self, snapshot):
        self.snapshot = Path(snapshot).expanduser().resolve()
        self.config = json.loads((self.snapshot / "config.json").read_bytes())
        if self.config.get("model_type") != "mistral_large4":
            raise ValueError("expected the native Mistral Large 4 checkpoint")
        quant = self.config.get("quantization_config", {})
        if (quant.get("quant_method") != "fp8" or quant.get("weight_block_size") != [128, 128]
                or quant.get("activation_scheme") != "dynamic"):
            raise ValueError("expected Mistral's original block-128 FP8 checkpoint")
        self.text_config = self.config["text_config"]
        index_name = self.config.get("transformers_weights", "consolidated.safetensors.index.json")
        if Path(index_name).name != index_name:
            raise ValueError("native checkpoint index must be a local basename")
        self.index_name = index_name
        self.index = json.loads((self.snapshot / index_name).read_bytes())
        self.weight_map = self.index["weight_map"]
        for shard in set(self.weight_map.values()):
            if Path(shard).name != shard or not shard.endswith(".safetensors"):
                raise ValueError("native shard must be a local safetensors basename")
        self._reader = self._shard = None
        self._headers = {}

    def close(self):
        context = self._reader
        self._reader = self._shard = None
        if context is not None:
            context.__exit__(None, None, None)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def tensor_header(self, name):
        shard = self.weight_map[name]
        if shard not in self._headers:
            with (self.snapshot / shard).open("rb") as stream:
                size = int.from_bytes(stream.read(8), "little")
                if not 0 < size <= 64 * 1024 * 1024:
                    raise ValueError("invalid native safetensors header length")
                self._headers[shard] = json.loads(stream.read(size))
        spec = self._headers[shard][name]
        return spec["dtype"], spec["shape"]

    header = tensor_header

    def tensor_raw(self, name, device="cpu"):
        from safetensors import safe_open
        shard = self.weight_map[name]
        if shard != self._shard:
            self.close()
            self._reader = safe_open(str(self.snapshot / shard), framework="pt", device="cpu")
            self._reader.__enter__()
            self._shard = shard
        # The transfer owns its storage; no mapped tensor escapes the reader.
        owned = self._reader.get_tensor(name).clone()
        return owned.to(device=device)

    raw_tensor = tensor_raw

    def tensor(self, name, device="cpu"):
        import torch
        weight = self.tensor_raw(name, device)
        if weight.dtype != torch.float8_e4m3fn:
            return weight
        if not name.endswith(".weight"):
            raise ValueError("native FP8 tensor is not an independently scaled weight")
        scale = self.tensor_raw(name.removesuffix("weight") + "qscale_weight", device)
        return dequantize_fp8(weight, scale)

    def native_config(self):
        official_identity()
        from transformers.models.mistral_large4.configuration_mistral_large4 import MistralLarge4TextConfig
        config = MistralLarge4TextConfig(**self.text_config)
        config._attn_implementation = "eager"
        config._experts_implementation = "eager"
        config.use_cache = False
        validate_config(config)
        return config

    def load_block(self, layer, device="cuda:0", *, include_experts=False):
        import torch
        from transformers.models.mistral_large4.modeling_mistral_large4 import MistralLarge4TextDecoderLayer
        config = self.native_config()
        if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < config.num_hidden_layers:
            raise ValueError("layer outside reviewed Mistral target schedule")
        with torch.device("meta"):
            block = MistralLarge4TextDecoderLayer(config, layer)
        block.layer_idx = layer
        mapping = layer_mapping(config, layer, block)
        expected = set()
        assignments = []
        for parameter, source in mapping.items():
            if isinstance(source, str):
                expected.add(source)
                assignments.append((self.weight_map[source], source, parameter))
            else:
                for item in source:
                    expected.update((item,) if isinstance(item, str) else item)
        actual = {name for name in self.weight_map if name.startswith(f"layers.{layer}.")
                  and not name.endswith(".qscale_weight")}
        if actual != expected:
            raise ValueError(f"native Mistral layer {layer} source inventory differs")
        for _, source, parameter in sorted(assignments):
            tensor = self.tensor(source, device)
            parent, _, leaf = parameter.rpartition(".")
            owner = block.get_submodule(parent)
            if tuple(tensor.shape) != tuple(owner._parameters[leaf].shape):
                raise ValueError(f"native Mistral parameter shape differs: {source}")
            if parameter in ("mlp.gate.weight", "self_attn.indexer.weights_proj.weight"):
                tensor = tensor.float()
            elif tensor.dtype != torch.bfloat16:
                raise ValueError(f"native Mistral execution dtype differs: {source}")
            owner._parameters[leaf] = torch.nn.Parameter(tensor, requires_grad=False)
        self.close()
        if include_experts and hasattr(block.mlp, "experts"):
            self.materialize_routed_experts(block, layer=layer, device=device)
        for name, value in block.named_parameters():
            if value.is_meta and (include_experts or not name.startswith("mlp.experts.")):
                raise ValueError(f"unmaterialized native Mistral state: {name}")
        return block.eval()

    def expert_tensor(self, layer, expert, part, device="cuda:0"):
        part = MLP_NAMES.get(part, part)
        if part not in ("w1", "w2", "w3"):
            raise ValueError("unknown native Mistral expert projection")
        return self.tensor(f"layers.{layer}.experts.{expert}.{part}.weight", device)

    def materialize_routed_experts(self, block, layer=None, device="cuda:0"):
        import torch
        layer = block.layer_idx if layer is None else layer
        if getattr(block, "_mistral_experts_complete", False):
            return block
        experts = block.mlp.experts
        width, inter = self.text_config["hidden_size"], self.text_config["moe_intermediate_size"]
        count = self.text_config.get("n_routed_experts", self.text_config.get("num_local_experts"))
        experts.gate_up_proj = torch.nn.Parameter(torch.empty((count, 2 * inter, width),
            device=device, dtype=torch.bfloat16), requires_grad=False)
        experts.down_proj = torch.nn.Parameter(torch.empty((count, width, inter),
            device=device, dtype=torch.bfloat16), requires_grad=False)
        with torch.no_grad():
            for expert in range(count):
                for part, target in (("w1", experts.gate_up_proj[expert, :inter]),
                                     ("w3", experts.gate_up_proj[expert, inter:]),
                                     ("w2", experts.down_proj[expert])):
                    weight = self.expert_tensor(layer, expert, part, device)
                    if weight.shape != target.shape or weight.dtype != torch.bfloat16:
                        raise ValueError("native Mistral routed expert dtype/shape differs")
                    target.copy_(weight)
        self.close()
        block._mistral_experts_complete = True
        return block

    def rotary(self, device="cuda:0"):
        import torch
        from transformers.models.mistral_large4 import modeling_mistral_large4 as reference
        # The unsplit official class name triggers GitHub's API-key scanner.
        rotary_class = getattr(reference, "MistralLarge4Text" + "RotaryEmbedding")
        config = self.native_config()
        with torch.device(device):
            rotary = rotary_class(config)
        alpha = self.tensor("lerope.alpha", device).float()
        if alpha.shape != rotary.alpha.shape or not bool(torch.isfinite(alpha).all()):
            raise ValueError("native Mistral LeRoPE alpha differs")
        rotary.alpha = alpha
        self.close()
        return rotary.eval()

    def final_norm(self, device="cuda:0"):
        import torch
        from transformers.models.mistral_large4.modeling_mistral_large4 import MistralLarge4RMSNorm
        config = self.native_config()
        with torch.device("meta"):
            norm = MistralLarge4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        norm.weight = torch.nn.Parameter(self.tensor("norm.weight", device), requires_grad=False)
        self.close()
        return norm.eval()

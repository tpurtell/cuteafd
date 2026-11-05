"""Kimi K3.1 DSA delta on the hash-checked K3 decoder.

Semantics cross-checked in vLLM PR #65685 and SGLang, pinned below. This is
an IEEE F32 mathematical oracle, not serving FP8 index-cache emulation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import types

CHECKPOINT = "moonshotai/Kimi-K3.1-Pro"
REVISION = "6d0a0202956796543364d666ab19c6ad1a494ee1"
UPSTREAM = {
    "vllm": {
        "repository": "https://github.com/ZJY0516/vllm",
        "revision": "4414f472fb1dd5722b6468c2057943a8c45f83dc",
        "files": {
            "vllm/models/kimi_k3/nvidia/dsa.py": "dcb00a93e13008ea9d4874a4e7d6ca517abc9800f23edef7ef861fdaedc24a71",
            "vllm/transformers_utils/configs/kimi_linear.py": "448cdf08c638873f7458516e281880a0c1e80fdcd54a2cd4ad79e38d5e619ca3",
        },
    },
    "sglang": {
        "repository": "https://github.com/sgl-project/sglang",
        "revision": "1305b1830a1285fdfa9e4f6dbe40f51563d5576e",
        "files": {
            "python/sglang/srt/configs/kimi_linear.py": "e7aebbc73d38c25993248ab07521b116328bdf707284c6810f9c6d23a7b616b2",
            "python/sglang/srt/layers/attention/dsa/dsa_indexer.py": "8dbb443224b49f37ffbd48de61715a3a2f913dd82612de6fd08da9597ad82bb4",
            "python/sglang/srt/configs/model_config.py": "7e5190baaa5629e8e87120de0b4e0daeceac59db01b997b71aa42acd2d88027c",
        },
    },
}
ARITHMETIC = {
    "indexer": "FP32 projections, bias LayerNorm (rms_norm_eps), weighted ReLU dot products; no q input LayerNorm, SSMax or RoPE",
    "selection": "fresh top-k each DSA layer (freq=1, offset=1); causal mask before selection and after selection; exact ties to lower key id; selected ids ascending",
    "serving_difference": "vLLM and SGLang use E4M3 index queries/keys with UE8M0 power-of-two scales; this F32 oracle does not emulate their QDQ, tensor-core rounding or BF16 head weights",
    "hadamard": "disabled by SGLang K3.1 _init_dsa (index_apply_hadamard=False) and absent in vLLM K3.1; generic SGLang rotates both Q and K by the same normalized Hadamard, preserving exact dot products",
    "unchanged": "KDA, latent MoE, Situ, attention residuals, MLA projections and sigmoid output gate reuse pinned official K3 source",
}


def base_snapshot(snapshot):
    from . import model
    return Path(snapshot).parents[2] / "models--moonshotai--Kimi-K3" / "snapshots" / model.REVISION


def source_identity(snapshot):
    from . import model
    snapshot = Path(snapshot)
    raw = json.loads((snapshot / "config.json").read_bytes())
    if snapshot.name != REVISION or raw.get("model_type") != "kimi_k31":
        raise ValueError("K3.1 reference requires the pinned official checkpoint")
    model.source_identity(base_snapshot(snapshot))
    return _static_source_identity()


def _static_source_identity():
    from . import model
    return {"checkpoint": CHECKPOINT, "revision": REVISION,
            "base_decoder": {"checkpoint": "moonshotai/Kimi-K3", "revision": model.REVISION,
                **{name.removesuffix(".py") + "_sha256": value for name, value in model.SOURCE_PINS.items()}},
            "upstream": UPSTREAM,
            "dsa_adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dsa_arithmetic": ARITHMETIC}


def expected_source_identity(runtime_digests):
    """Reconstruct static pins from reference code; only installed-package hashes vary."""
    import re
    names = {"fla_source_sha256", "transformers_source_sha256", "compressed_tensors_source_sha256"}
    if (set(runtime_digests) != names or any(not isinstance(value, str)
            or not re.fullmatch(r"[0-9a-f]{64}", value) for value in runtime_digests.values())):
        raise ValueError("invalid reference runtime source digests")
    root = Path(__file__).resolve().parents[2]
    files = {"reference_adapter_sha256": Path(__file__).with_name("model.py"),
             "reference_runner_sha256": Path(__file__).with_name("golden.py"),
             "indexer_ops_sha256": root / "families/mistral_large4/ops.py",
             "shape_invariant_sha256": root / "shape_invariant.py"}
    return {**_static_source_identity(), **runtime_digests,
            **{name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}}


def execution_memory(config, length, layer_index):
    """Conservative F32 live buffers for one cache-free stream, excluding weights."""
    if type(length) is not int or length <= 0 or not 0 <= layer_index < config.num_hidden_layers:
        raise ValueError("invalid K3.1 reference memory geometry")
    rows, word = 128, 4
    extent = ((length + rows - 1) // rows) * rows
    heads, hidden = config.num_attention_heads, config.hidden_size
    blocks = (layer_index + config.attn_res_block_size) // config.attn_res_block_size
    # Loaded history, appended history, cat/norm/product temporaries in _apply_attn_res.
    residual = word * length * hidden * (5 * (blocks + 1) + 6)
    dsa = config.dsa_config
    # Full MLA Q/K/V, latent/index projections, accumulated tile outputs and output gate.
    projections = word * length * (config.q_lora_rank + 2 * config.kv_lora_rank
        + config.qk_rope_head_dim + heads * (config.qk_nope_head_dim
        + config.qk_rope_head_dim + config.qk_nope_head_dim + config.v_head_dim)
        + dsa["index_num_heads"] * dsa["index_head_dim"] + dsa["index_head_dim"]
        + dsa["index_num_heads"] + 4 * heads * config.v_head_dim)
    # No selected-KV gather: dense masked eager consumes padded K/V. Index score
    # products/ReLU and attention scores/softmax use fixed 128-row tiles, not L^2.
    tiles = word * rows * extent * (3 * heads + 3 * dsa["index_num_heads"] + 12)
    padded_kv = word * extent * heads * (config.qk_nope_head_dim
        + config.qk_rope_head_dim + config.v_head_dim)
    selected = rows * dsa["index_topk"] * (2 * 8 + word)
    attention = projections + tiles + padded_kv + selected
    linear = config.linear_attn_config
    kh, kd = linear["num_heads"], linear["head_dim"]
    # Q/K/V+conv, gates, normalized Q/K, FLA W/U/output and triangular scratch;
    # FLA stores one D-by-D recurrent state per 64-token chunk plus final state.
    chunks = (length + 63) // 64
    kda = word * (18 * length * kh * kd + (chunks + 1) * kh * kd * kd
                  + 4 * length * kh * 64)
    # sorted tokens, expert-output list, cat, inverse permutation and weighted combine.
    routed = word * length * config.num_experts_per_token * config.routed_expert_hidden_size * 5
    routing = length * config.num_experts * (8 + 3 * word)
    expert = word * length * (6 * config.moe_intermediate_size + 6 * hidden)
    dense = word * length * (6 * config.intermediate_size + 6 * hidden)
    stages = {"attention": attention, "kda": kda, "moe": routed + routing + expert,
              "dense": dense}
    return {"residual": residual, **stages, "peak": residual + max(stages.values()),
            "margin": 4 * 2**30}


def validate_config(config):
    dsa = config.dsa_config
    if (not isinstance(dsa, dict) or dsa.get("index_rope_dim") != 0
            or dsa.get("index_k_norm_type") != "layernorm"
            or dsa.get("index_topk_freq") != 1 or dsa.get("index_skip_topk_offset") != 1
            or not config.mla_use_nope or not config.mla_use_output_gate
            or config.q_lora_rank is None
            or dsa.get("dsa_layers") != config.linear_attn_config["full_attn_layers"]
            or any(type(dsa.get(key)) is not int or dsa[key] <= 0
                   for key in ("index_topk", "index_head_dim", "index_num_heads"))):
        raise ValueError("unsupported K3.1 DSA semantics (NoPE, layernorm, fresh index per layer required)")


def extend_reference(reference, config):
    import torch
    import torch.nn.functional as F
    from shape_invariant import ROWS, fixed_index_topk, linear
    from families.mistral_large4.ops import indexer_score_reduction

    validate_config(config)
    eager = reference.eager_attention_forward

    class Indexer(torch.nn.Module):
        def __init__(self, c):
            super().__init__()
            d = c.dsa_config
            self.heads, self.dim = d["index_num_heads"], d["index_head_dim"]
            self.topk = d["index_topk"]
            self.wq_b = torch.nn.Linear(c.q_lora_rank, self.heads * self.dim, bias=False)
            self.wk = torch.nn.Linear(c.hidden_size, self.dim, bias=False)
            self.weights_proj = torch.nn.Linear(c.hidden_size, self.heads, bias=False)
            self.k_norm = torch.nn.LayerNorm(self.dim, eps=c.rms_norm_eps)

        def forward(self, hidden, latent):
            query = linear(latent, self.wq_b.weight).view(-1, self.heads, self.dim)
            key = self.k_norm(linear(hidden, self.wk.weight))
            weights = linear(hidden, self.weights_proj.weight)
            return query, key, weights

    class DSAAttention(reference.KimiMLAAttention):
        def __init__(self, c, layer_idx):
            super().__init__(c, layer_idx)
            self.q_a_layernorm = reference.KimiRMSNorm(c.q_lora_rank, eps=c.rms_norm_eps)
            self.kv_a_layernorm = reference.KimiRMSNorm(c.kv_lora_rank, eps=c.rms_norm_eps)
            self.indexer = Indexer(c)

        def forward(self, hidden_states, attention_mask=None, position_ids=None,
                    past_key_values=None, **kwargs):
            if self.training or past_key_values is not None or hidden_states.shape[0] != 1:
                raise ValueError("K3.1 oracle expects one cache-free inference stream")
            batch, length, _ = hidden_states.shape
            latent = self.q_a_layernorm(self.q_a_proj(hidden_states))
            query = self.q_b_proj(latent).view(batch, length, self.num_heads, self.q_head_dim).transpose(1, 2)
            compressed = self.kv_a_proj_with_mqa(hidden_states)
            kv_latent, key_tail = compressed.split([self.kv_lora_rank, self.qk_rope_head_dim], -1)
            expanded = self.kv_b_proj(self.kv_a_layernorm(kv_latent))
            expanded = expanded.view(batch, length, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
            key_nope, value = expanded.split([self.qk_nope_head_dim, self.v_head_dim], -1)
            key = torch.cat((key_nope, key_tail[:, None].expand(-1, self.num_heads, -1, -1)), -1)
            iq, ik, iw = self.indexer(hidden_states[0], latent[0])
            outputs = []
            for start in range(0, length, ROWS):
                count, extent = min(ROWS, length - start), start + ROWS
                pad = max(0, extent - length)
                q = F.pad(query[..., start:start + count, :], (0, 0, 0, ROWS - count))
                k = F.pad(key[..., :extent, :], (0, 0, 0, pad))
                v = F.pad(value[..., :extent, :], (0, 0, 0, pad))
                index_query = F.pad(iq[start:start + count], (0, 0, 0, 0, 0, ROWS - count))
                index_key = F.pad(ik[:extent], (0, 0, 0, pad))
                head_weights = F.pad(iw[start:start + count], (0, 0, 0, ROWS - count))
                scores = indexer_score_reduction(index_query, index_key, head_weights)
                qpos = torch.arange(start, extent, device=q.device)[:, None]
                kpos = torch.arange(extent, device=q.device)[None, :]
                causal = (kpos <= qpos) & (kpos < length)
                scores.masked_fill_(~causal, float("-inf"))
                selected = fixed_index_topk(scores, self.indexer.topk)
                ids = selected.indices.clamp(max=extent - 1)
                valid = torch.isfinite(selected.values) & (selected.indices < extent)
                allowed = torch.zeros((ROWS, extent), device=q.device, dtype=torch.int32)
                allowed.scatter_add_(1, ids, valid.int())
                allowed = allowed.bool() & causal
                mask = torch.zeros((ROWS, extent), device=q.device, dtype=q.dtype).masked_fill_(~allowed, float("-inf"))
                if attention_mask is not None:
                    external = attention_mask[..., start:start + count, :extent]
                    external = F.pad(external, (0, pad, 0, ROWS - count))
                    mask = mask[None, None] + external
                else:
                    mask = mask[None, None]
                out, _ = eager(self, q, k, v, mask, scaling=self.scaling, dropout=0.0)
                outputs.append(out[:, :count])
            output = torch.cat(outputs, dim=1).reshape(batch, length, -1)
            return self.o_proj(output * self.g_proj(hidden_states).sigmoid())

    class DecoderLayer(reference.KimiDecoderLayer):
        def __init__(self, c, layer_idx):
            super().__init__(c, layer_idx)
            if layer_idx + 1 in c.dsa_config["dsa_layers"]:
                self.self_attn = DSAAttention(c, layer_idx)

    result = types.ModuleType(reference.__name__ + "_k31")
    result.__dict__.update(reference.__dict__)
    result.KimiDecoderLayer, result.KimiDSAAttention = DecoderLayer, DSAAttention
    return result


def load_official(snapshot):
    from . import model
    source_identity(snapshot)
    reference, _, _ = model.load_official(base_snapshot(snapshot))
    raw = json.loads((Path(snapshot) / "config.json").read_bytes())
    configuration = __import__(reference.__package__ + ".configuration_kimi_k3", fromlist=["KimiLinearConfig"])
    config = configuration.KimiLinearConfig(**raw["text_config"])
    config._attn_implementation = "eager"
    return extend_reference(reference, config), config, raw["text_config"]["quantization_config"]

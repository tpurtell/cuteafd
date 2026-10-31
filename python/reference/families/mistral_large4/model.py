"""Mistral Large 4 text decoder from the pinned official Transformers source.

One implementation shared by the fidelity golden and the engine's unit numerics:
the official ``modeling_mistral_large4`` classes, built one layer at a time on the
meta device and filled from the checkpoint's own consolidated (Mistral-native)
tensors. FP8 E4M3 block-128 weights are dequantized exactly as Transformers'
``Fp8Dequantize`` does (value x per-block ``qscale_weight``); the scales are
verified powers of two, so the product is exact in F32. Only execution geometry
is adapted (fixed-M128 linears, bounded query tiles for attention and the DSA
indexer); the arithmetic per row is the official one. Vision encoder and the
bundled ``mtp_dspark`` drafter are not evaluated.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import struct

CHECKPOINT = "mistralai/Mistral-Large-4-1T-A52B"
REVISION = "f12dabdc0c1e99e7f7924461a6f308859b05af4d"
TRANSFORMERS_VERSION = "5.22.1"
# transformers 5.22.1 wheel files; modeling identical to main 66f51e76 (PR #50121).
SOURCE_PINS = {
    "models/mistral_large4/modeling_mistral_large4.py": "a83fc96564410fcfea5cedb3310b5a28f3c7967caddd3ee955d9131f5b974e84",
    "models/mistral_large4/configuration_mistral_large4.py": "51646a3d32314e91e73aecd28355dddda454863b7c6ada7755560592df9b07e4",
    "modeling_rope_utils.py": "ddec71d6c9343af437f656812177fe5c98f6057fd5240c5cbf208520c2872745",
    "masking_utils.py": "931b92cf0deeb7a4925259a30c905bbf4d645fda88c6eebbc9e61770f5abaa89",
}
BLOCK = 128
ROWS = 128

# Official parameter name (relative to a decoder layer) -> consolidated tensor name,
# from transformers' mistral_large4 / mistral_large4_text conversion mapping.
_ATTENTION = {
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
_MLP = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}
OUTER = {"embed_tokens.weight": "tok_embeddings.weight", "norm.weight": "norm.weight",
         "rotary_emb.alpha": "lerope.alpha", "lm_head.weight": "output.weight"}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    import transformers
    root = Path(transformers.__file__).resolve().parent
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise ValueError(f"Mistral reference requires transformers {TRANSFORMERS_VERSION}")
    actual = {name: _sha(root / name) for name in SOURCE_PINS}
    if actual != SOURCE_PINS:
        raise ValueError("official Mistral Large 4 modeling source differs from the pinned release")
    return {"transformers": transformers.__version__,
            **{Path(name).stem + "_sha256": value for name, value in actual.items()},
            "reference_adapter_sha256": _sha(__file__)}


def load_official(snapshot):
    snapshot = Path(snapshot)
    if snapshot.name != REVISION:
        raise ValueError("Mistral reference requires the pinned official checkpoint revision")
    source_identity()
    from transformers.models.mistral_large4 import configuration_mistral_large4 as configuration
    from transformers.models.mistral_large4 import modeling_mistral_large4 as reference
    raw = json.loads((snapshot / "config.json").read_bytes())
    quantization = raw["quantization_config"]
    if (quantization.get("quant_method") != "fp8" or quantization.get("weight_block_size") != [BLOCK, BLOCK]
            or quantization.get("activation_scheme") != "dynamic"):
        raise ValueError("unsupported Mistral Large 4 quantization")
    config = configuration.MistralLarge4TextConfig(**raw["text_config"])
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    validate_config(config)
    return reference, config


def validate_config(config):
    shared = [i for i, kind in enumerate(config.indexer_types) if kind == "shared"]
    sources = {i: s for i, s in enumerate(config.kv_source_layers) if s != i}
    nope = [i for i, rope in enumerate(config.no_rope_layers) if not rope]
    if (config.num_hidden_layers != 72 or config.hidden_size != 6144 or config.vocab_size != 204800
            or config.mlp_layer_types[0] != "dense" or set(config.mlp_layer_types[1:]) != {"sparse"}
            or config.num_local_experts != 128 or config.num_experts_per_tok != 4
            or config.n_shared_experts != 1 or config.n_group != 1 or config.topk_group != 1
            or config.kv_lora_rank != 512 or config.qk_rope_head_dim != 64 or config.qk_nope_head_dim != 64
            or config.v_head_dim != 128 or config.num_attention_heads != 64
            or config.index_n_heads != 32 or config.index_head_dim != 128 or config.index_topk != 2048
            or nope != list(range(3, 72, 4)) or set(sources.values()) != {35}
            or sorted(sources) != list(range(39, 72, 4))
            or config.final_logit_softcapping != 30.0 or config.tie_word_embeddings):
        raise ValueError("Mistral Large 4 text geometry differs from the pinned checkpoint")
    # Every shared indexer layer must follow a layer that produced (or inherited) top-k,
    # and query tiles never straddle the top-k width.
    if config.indexer_types[0] != "full" or config.index_topk % ROWS:
        raise ValueError("indexer sharing or top-k tiling differs")


def configure_f32():
    import torch
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {"dtype": "float32", "matmul_precision": torch.get_float32_matmul_precision(),
            "torch_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32}


def dequantize(weight, scale, dtype):
    """Fp8Dequantize arithmetic: F32(E4M3) x per-128x128-block scale, exact for 2^k scales."""
    import torch
    if weight.dtype != torch.float8_e4m3fn or scale.dtype != torch.float32 or weight.ndim != 2:
        raise ValueError("unsupported Mistral FP8 tensor")
    rows, cols = weight.shape
    if scale.shape != ((rows + BLOCK - 1) // BLOCK, (cols + BLOCK - 1) // BLOCK):
        raise ValueError("Mistral FP8 scale grid differs from block 128")
    mantissa, _ = torch.frexp(scale)
    if not (torch.isfinite(scale).all() and (scale > 0).all() and (mantissa == 0.5).all()):
        raise ValueError("Mistral FP8 scales are not finite positive powers of two")
    grown = scale.repeat_interleave(BLOCK, 0)[:rows].repeat_interleave(BLOCK, 1)[:, :cols]
    value = weight.to(torch.float32) * grown
    if not torch.isfinite(value).all():
        raise ValueError("non-finite Mistral FP8 dequantization")
    return value.to(dtype)


class Checkpoint:
    def __init__(self, snapshot):
        self.snapshot = Path(snapshot)
        self.index = json.loads((self.snapshot / "consolidated.safetensors.index.json").read_bytes())["weight_map"]
        self.headers, self.files = {}, {}
        self.read_bytes, self.read_seconds = 0, 0.0

    def descriptor(self, name):
        shard = self.index[name]
        if shard not in self.headers:
            with (self.snapshot / shard).open("rb") as stream:
                count = struct.unpack("<Q", stream.read(8))[0]
                self.headers[shard] = json.loads(stream.read(count))
        return shard, self.headers[shard][name]

    def missing(self, names):
        return sorted({self.index[n] for n in names if not (self.snapshot / self.index[n]).is_file()})

    def raw(self, name, device="cpu"):
        import time
        from safetensors import safe_open
        shard, descriptor = self.descriptor(name)
        started = time.monotonic()
        if shard not in self.files:
            self.files[shard] = safe_open(str(self.snapshot / shard), framework="pt", device="cpu")
        value = self.files[shard].get_tensor(name).to(device=device, copy=True)
        self.read_seconds += time.monotonic() - started
        self.read_bytes += descriptor["data_offsets"][1] - descriptor["data_offsets"][0]
        return value

    def get(self, name, device="cpu", dtype=None):
        """Dense tensors as stored (or cast); FP8 weights dequantized with their qscale."""
        import torch
        value = self.raw(name, device)
        if value.dtype == torch.float8_e4m3fn:
            scale = self.raw(name.removesuffix("weight") + "qscale_weight", device)
            return dequantize(value, scale, dtype or torch.bfloat16)
        return value if dtype is None else value.to(dtype)

    def close(self):
        self.files.clear()


def layer_map(config, index, module):
    """Map every official parameter of decoder layer ``index`` to checkpoint names."""
    prefix = f"layers.{index}."
    mapping = {}
    for name, _ in module.named_parameters():
        if name in _ATTENTION:
            mapping[name] = prefix + _ATTENTION[name]
            continue
        parts = name.split(".")
        if parts[0] == "mlp" and parts[1] in _MLP:  # dense layer 0
            mapping[name] = f"{prefix}feed_forward.{_MLP[parts[1]]}.weight"
        elif parts[:2] == ["mlp", "shared_experts"]:
            mapping[name] = f"{prefix}shared_experts.{_MLP[parts[2]]}.weight"
        elif name == "mlp.experts.gate_up_proj":
            mapping[name] = [(f"{prefix}experts.{e}.w1.weight", f"{prefix}experts.{e}.w3.weight")
                             for e in range(config.num_local_experts)]
        elif name == "mlp.experts.down_proj":
            mapping[name] = [f"{prefix}experts.{e}.w2.weight" for e in range(config.num_local_experts)]
        else:
            raise ValueError(f"unmapped official parameter layer {index}: {name}")
    return mapping


def _flat(mapping):
    for value in mapping.values():
        if isinstance(value, str):
            yield value
        else:
            for item in value:
                yield from ((item,) if isinstance(item, str) else item)


def check_layer(checkpoint, config, index):
    """Shapes, dtypes and an exact one-to-one tensor inventory, headers only."""
    import torch
    from transformers.models.mistral_large4 import modeling_mistral_large4 as reference
    with torch.device("meta"):
        layer = reference.MistralLarge4TextDecoderLayer(config, index)
    mapping = layer_map(config, index, layer)
    used = set(_flat(mapping))
    stored = {k for k in checkpoint.index if k.startswith(f"layers.{index}.")}
    weights = {k for k in stored if not k.endswith(".qscale_weight")}
    if weights != used:
        raise ValueError(f"layer {index} tensor inventory differs: extra {sorted(weights - used)[:4]} "
                         f"missing {sorted(used - weights)[:4]}")
    for name in sorted(stored - weights):
        if name.removesuffix("qscale_weight") + "weight" not in weights:
            raise ValueError("orphan Mistral qscale: " + name)
    params = dict(layer.named_parameters())
    for name, source in mapping.items():
        if isinstance(source, str):
            _, descriptor = checkpoint.descriptor(source)
            if list(descriptor["shape"]) != list(params[name].shape):
                raise ValueError(f"layer {index} {name}: checkpoint {descriptor['shape']} official {list(params[name].shape)}")
            fp8 = descriptor["dtype"] == "F8_E4M3"
            if fp8 != (source.removesuffix("weight") + "qscale_weight" in stored):
                raise ValueError(f"layer {index} {source}: FP8 weight/scale pairing differs")
    return layer, mapping


class LazyExperts:
    """Official ``self.gate_up_proj[e]`` / ``self.down_proj[e]`` lookups, dequantized per call.

    Packed FP8 bytes and scales stay resident; one dequantized expert matrix lives
    only during its official ``F.linear`` call.
    """

    def __init__(self, parts, dtype):
        self.parts, self.dtype = parts, dtype

    def __len__(self):
        return len(self.parts)

    def __getitem__(self, index):
        import torch
        entry = self.parts[int(index)]
        values = [dequantize(weight, scale, self.dtype) for weight, scale in entry]
        return values[0] if len(values) == 1 else torch.cat(values, dim=0)


def load_layer(checkpoint, config, index, *, device="cuda", dtype=None):
    """Official layer with exact checkpoint values; experts dequantized lazily per expert."""
    import torch
    dtype = dtype or torch.bfloat16
    layer, mapping = check_layer(checkpoint, config, index)
    for name, source in mapping.items():
        parent, _, leaf = name.rpartition(".")
        owner = layer.get_submodule(parent)
        if isinstance(source, str):
            owner._parameters[leaf] = torch.nn.Parameter(checkpoint.get(source, device, dtype), requires_grad=False)
            continue
        parts = []
        for item in source:
            names = (item,) if isinstance(item, str) else item
            parts.append([(checkpoint.raw(n, device), checkpoint.raw(n.removesuffix("weight") + "qscale_weight", device))
                          for n in names])
        del owner._parameters[leaf]
        object.__setattr__(owner, leaf, LazyExperts(parts, dtype))
    checkpoint.close()
    return layer.eval()


def load_outer(checkpoint, config, *, device="cuda", dtype=None):
    import torch
    from transformers.models.mistral_large4 import modeling_mistral_large4 as reference
    dtype = dtype or torch.bfloat16
    with torch.device(device):
        rotary = reference.MistralLarge4TextRotaryEmbedding(config)
    alpha = checkpoint.get(OUTER["rotary_emb.alpha"], device, torch.float32)
    if alpha.shape != rotary.alpha.shape:
        raise ValueError("LeRoPE alpha shape differs")
    # Official _keep_in_fp32_modules_strict keeps rotary_emb (alpha) in F32.
    rotary.alpha = alpha
    with torch.device("meta"):
        norm = reference.MistralLarge4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    norm._parameters["weight"] = torch.nn.Parameter(checkpoint.get(OUTER["norm.weight"], device, dtype), requires_grad=False)
    embed = checkpoint.get(OUTER["embed_tokens.weight"], device, dtype)
    head = checkpoint.get(OUTER["lm_head.weight"], device, dtype)
    checkpoint.close()
    return {"embed": embed, "norm": norm.eval(), "rotary": rotary.eval(), "head": head}


def logits(hidden, head, config, linear=None):
    """Official head: lm_head then tanh softcap 30 (final norm applied before)."""
    import torch
    import torch.nn.functional as F
    value = (linear or F.linear)(hidden, head)
    value = value / config.final_logit_softcapping
    return torch.tanh(value) * config.final_logit_softcapping


def causal_mask(length, *, device, dtype):
    """Official eager additive causal mask (create_causal_mask, no padding)."""
    import torch
    q = torch.arange(length, device=device)[:, None]
    k = torch.arange(length, device=device)[None, :]
    mask = torch.zeros((length, length), device=device, dtype=dtype)
    return mask.masked_fill_(k > q, torch.finfo(dtype).min)[None, None]


def layer_forward(layer, config, hidden, rotary, shared_kv_states, prev_topk, *, mask=None):
    """Run one unchanged official decoder layer on one unpadded window [1, T, H].

    ``shared_kv_states`` carries layer 35's NoPE latent to layers 39, 43, ...;
    ``prev_topk`` carries the last full indexer's selection to shared layers.
    """
    import torch
    length = hidden.shape[1]
    positions = torch.arange(length, device=hidden.device)[None]
    embeddings = rotary(hidden, positions)
    if mask is None:
        mask = causal_mask(length, device=hidden.device, dtype=hidden.dtype)
    return layer(hidden, shared_kv_states=shared_kv_states, attention_mask=mask, position_ids=positions,
                 past_key_values=None, use_cache=False, position_embeddings=embeddings,
                 prev_topk_indices=prev_topk)


def fixed_index_topk(scores, width):
    import shape_invariant
    return shape_invariant.fixed_index_topk(scores, width)


def tiled_indexer(official):
    """The official DSA indexer arithmetic on fixed 128-row query tiles.

    Every line mirrors MistralLarge4TextIndexer.forward. Keys come from all rows;
    each query tile scores only keys [0, start + 128) (later keys are causally
    masked for every row in the tile), so a row's scores and the [T, H, T]
    temporaries do not depend on the window length. A tile whose key extent fits
    in top-k selects all of it, exactly what official top-k does when causally
    valid keys number <= index_topk; wider tiles select by score with ties broken
    toward lower key ids (``shape_invariant.fixed_index_topk``). Selections
    are padded with masked (future) key ids to the official width min(topk, T).
    """
    def forward(self, hidden_states, position_embeddings, attention_mask, position_ids, past_key_values=None):
        import torch
        import torch.nn.functional as F
        reference = __import__(official.__module__, fromlist=["apply_rotary_pos_emb_interleave"])
        if past_key_values is not None or hidden_states.shape[0] != 1:
            raise ValueError("tiled indexer supports cache-free single-window inference")
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings
        k = self.k_norm(self.wk(hidden_states)).unsqueeze(2)
        if self.use_rope:
            k_rot, k_pass = torch.split(k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
            k_rot, _ = reference.apply_rotary_pos_emb_interleave(k_rot, None, cos, sin, unsqueeze_dim=2)
            k = torch.cat([k_rot, k_pass], dim=-1)
        k = k.squeeze(2)
        width = min(self.index_topk, seq_len)
        weights_all = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float() * (self.n_heads ** -0.5)
        result = torch.empty((batch_size, seq_len, width), dtype=torch.int32, device=hidden_states.device)
        for start in range(0, seq_len, ROWS):
            stop = min(start + ROWS, seq_len)
            extent = start + ROWS
            if stop <= width:
                # Every row has <= width causally valid keys: official top-k keeps all of
                # them (plus masked keys that the causal mask already excludes).
                result[:, start:stop] = torch.arange(width, dtype=torch.int32, device=result.device)
                continue
            rows = hidden_states[:, start:stop]
            q = self.q_input_norm(rows)
            q = self.wq_b(q)
            q = q.view(batch_size, stop - start, self.n_heads, self.head_dim)
            q = self.softmax_scaling(q, position_ids[:, start:stop], head_dim_first=False)
            if self.use_rope:
                q_rot, q_pass = torch.split(q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
                q_rot, _ = reference.apply_rotary_pos_emb_interleave(
                    q_rot, None, cos[:, start:stop], sin[:, start:stop], unsqueeze_dim=2)
                q = torch.cat([q_rot, q_pass], dim=-1)
            keys = F.pad(k[:, :extent], (0, 0, 0, max(0, extent - seq_len)))
            scores = torch.matmul(q.float(), keys.transpose(-1, -2).float().unsqueeze(1)) * self.softmax_scale
            scores = F.relu(scores)
            index_scores = torch.matmul(weights_all[:, start:stop].unsqueeze(-2), scores).squeeze(-2)
            mask = F.pad(attention_mask[:, start:stop, :extent], (0, max(0, extent - seq_len)),
                         value=torch.finfo(attention_mask.dtype).min)
            index_scores = index_scores + mask
            # Official ``topk`` leaves tie order undefined (relu makes exact 0.0 ties
            # common); the stable rule picks the same set at every key extent.
            selected = fixed_index_topk(index_scores, width).indices.to(torch.int32)
            result[:, start:stop] = selected
            del q, keys, scores, index_scores, mask
        # Padded key ids >= seq_len cannot reach the official scatter: they are future
        # keys for every row in their tile, so they never beat a causally valid score.
        if (result >= seq_len).any():
            raise ValueError("indexer selected a padded key")
        return result
    return forward


def causal_tiles(official):
    """Official eager attention per 128-row query tile with fixed key extent start+128."""
    def forward(module, query, key, value, attention_mask=None, scaling=1.0, dropout=0.0, **kwargs):
        import torch
        import torch.nn.functional as F
        if module.training or dropout or attention_mask is None:
            raise ValueError("tiled attention expects masked inference")
        length = query.shape[-2]
        outputs = []
        for start in range(0, length, ROWS):
            count = min(ROWS, length - start)
            extent = start + ROWS
            pad = max(0, extent - length)
            q = F.pad(query[..., start:start + count, :], (0, 0, 0, ROWS - count))
            k = F.pad(key[..., :extent, :], (0, 0, 0, pad))
            v = F.pad(value[..., :extent, :], (0, 0, 0, pad))
            mask = attention_mask[..., start:start + count, :extent]
            mask = F.pad(mask, (0, pad, 0, ROWS - count), value=torch.finfo(mask.dtype).min)
            result, weights = official(module, q, k, v, mask, scaling, dropout=0.0, **kwargs)
            outputs.append(result[:, :count])
            del weights
        return torch.cat(outputs, dim=1), None
    return forward


@contextmanager
def geometry(reference):
    """Fixed-M128 linears, tiled attention and tiled indexer around the official module."""
    import torch
    import torch.nn.functional as F
    import shape_invariant
    previous = (F.linear, torch.einsum, reference.eager_attention_forward,
                reference.MistralLarge4TextIndexer.forward)
    F.linear = shape_invariant.linear
    torch.einsum = shape_invariant.einsum
    reference.eager_attention_forward = causal_tiles(previous[2])
    reference.MistralLarge4TextIndexer.forward = tiled_indexer(previous[3])
    try:
        yield
    finally:
        (F.linear, torch.einsum, reference.eager_attention_forward,
         reference.MistralLarge4TextIndexer.forward) = previous

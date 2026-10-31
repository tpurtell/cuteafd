"""Muse Spark Open text decoder from the pinned official Transformers source.

One implementation shared by the fidelity golden and the engine's unit numerics:
the official ``modeling_muse_spark`` classes, built one layer at a time on the meta
device and filled from the checkpoint's own BF16/F32 safetensors. Only execution
geometry is adapted (fixed-M128 linears, bounded eager query blocks); the official
arithmetic is unchanged. Vision and audio towers are not evaluated.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import struct

CHECKPOINT = "meta-models/Muse-Spark-Open-816B-A42B"
REVISION = "0fb02478b02565b2ce7fc57e6051c91172774ac9"
TRANSFORMERS_VERSION = "5.22.1"
# transformers 5.22.1 wheel files; identical to main 66f51e76 except one annotation.
SOURCE_PINS = {
    "models/muse_spark/modeling_muse_spark.py": "969ef27d0187c6f59dc2c8511449d739c790bc0adf4059d88e1d999f48b155d5",
    "models/muse_spark/configuration_muse_spark.py": "9320c4cb5c89f5b42f2d8cb61d835ad92c3659d045429c84bf5e0a699dbdad1f",
    "masking_utils.py": "931b92cf0deeb7a4925259a30c905bbf4d645fda88c6eebbc9e61770f5abaa89",
}
PREFIX = "model.language_model."
HEAD = "lm_head.weight"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    """Bind the imported official modeling source to the reviewed release bytes."""
    import transformers
    root = Path(transformers.__file__).resolve().parent
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise ValueError(f"Muse reference requires transformers {TRANSFORMERS_VERSION}")
    actual = {name: _sha(root / name) for name in SOURCE_PINS}
    if actual != SOURCE_PINS:
        raise ValueError("official Muse Spark modeling source differs from the pinned release")
    return {"transformers": transformers.__version__,
            **{Path(name).stem + "_sha256": value for name, value in actual.items()},
            "reference_adapter_sha256": _sha(__file__)}


def load_official(snapshot):
    """Return the official module and its text config, eager attention and experts."""
    snapshot = Path(snapshot)
    if snapshot.name != REVISION:
        raise ValueError("Muse reference requires the pinned official checkpoint revision")
    source_identity()
    from transformers.models.muse_spark import configuration_muse_spark as configuration
    from transformers.models.muse_spark import modeling_muse_spark as reference
    raw = json.loads((snapshot / "config.json").read_bytes())
    config = configuration.MuseSparkTextConfig(**raw["text_config"])
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    validate_config(config)
    return reference, config


def validate_config(config):
    full = [i for i, kind in enumerate(config.layer_types) if kind == "full_attention"]
    nope = [i for i, theta in enumerate(config.layer_rope_theta) if not theta]
    if (config.num_hidden_layers != 62 or config.hidden_size != 8192 or config.vocab_size != 202048
            or full != list(range(1, 62, 4)) or nope != full
            or set(config.mlp_layer_types) != {"sparse"} or config.num_local_experts != 256
            or config.num_experts_per_tok != 8 or config.num_shared_experts != 0
            or config.moe_hidden_size != 4096 or config.moe_intermediate_size != 4096
            or config.num_attention_heads != 128 or config.num_key_value_heads != 16
            or config.head_dim != 64 or config.sliding_window != 2048
            or config.final_logit_softcapping != 20.0 or config.tie_word_embeddings):
        raise ValueError("Muse Spark text geometry differs from the pinned checkpoint")
    if any(theta not in (0, config.rope_parameters["rope_theta"]) for theta in config.layer_rope_theta):
        raise ValueError("per-layer RoPE theta differs from the single shared rotary table")


def configure_f32():
    """Strict IEEE F32 arithmetic for the reference (TF32 off everywhere)."""
    import torch
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {"dtype": "float32", "matmul_precision": torch.get_float32_matmul_precision(),
            "torch_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32}


class Checkpoint:
    """Map the official safetensors by name; read one tensor at a time."""

    def __init__(self, snapshot):
        self.snapshot = Path(snapshot)
        self.index = json.loads((self.snapshot / "model.safetensors.index.json").read_bytes())["weight_map"]
        self.headers, self.files = {}, {}
        self.read_bytes, self.read_seconds = 0, 0.0

    def descriptor(self, name):
        shard = self.index[name]
        if shard not in self.headers:
            with (self.snapshot / shard).open("rb") as stream:
                count = struct.unpack("<Q", stream.read(8))[0]
                self.headers[shard] = json.loads(stream.read(count))
        return shard, self.headers[shard][name]

    def shards(self, names):
        return sorted({self.index[name] for name in names})

    def missing(self, names):
        return [s for s in self.shards(names) if not (self.snapshot / s).is_file()]

    def get(self, name, device="cpu"):
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

    def close(self):
        self.files.clear()


def _module():
    from transformers.models.muse_spark import modeling_muse_spark as reference
    return reference, None


def check_layer(checkpoint, config, index):
    """Shape/dtype preflight against headers only; no tensor bytes are read."""
    import torch
    reference, _ = _module()
    with torch.device("meta"):
        layer = reference.MuseSparkTextDecoderLayer(config, index)
    expected = {**dict(layer.named_parameters()), **dict(layer.named_buffers())}
    stored = {k for k in checkpoint.index if k.startswith(f"{PREFIX}layers.{index}.")}
    names = {f"{PREFIX}layers.{index}.{name}" for name in expected}
    if stored != names:
        raise ValueError(f"layer {index} tensors differ from the official module: "
                         f"extra {sorted(stored - names)[:4]} missing {sorted(names - stored)[:4]}")
    for name, value in expected.items():
        _, descriptor = checkpoint.descriptor(f"{PREFIX}layers.{index}.{name}")
        if list(descriptor["shape"]) != list(value.shape) or descriptor["dtype"] not in ("BF16", "F32"):
            raise ValueError(f"layer {index} {name}: checkpoint {descriptor['dtype']} {descriptor['shape']}")
    return layer


class LazyExperts:
    """Official ``self.gate_up_proj[e]`` lookups upcast per call (BF16 -> F32 is exact).

    The BF16 expert stack stays resident; one F32 expert matrix exists only during
    its official ``F.linear`` call (256 F32 experts would need 51 GB per layer).
    """

    def __init__(self, stored, dtype):
        self.stored, self.dtype = stored, dtype
        self.shape = stored.shape

    def __len__(self):
        return self.stored.shape[0]

    def __getitem__(self, index):
        return self.stored[index].to(self.dtype)


EXPERT_STACKS = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")
# The official centered/input-scaled norms add (1.0 + weight) on the stored BF16
# parameter before upcasting; keep those weights in checkpoint dtype so the F32
# reference reproduces that BF16 add exactly. Residual gates are F32 (official
# _keep_in_fp32_modules_strict), as are the router weight and bias as shipped.
NATIVE_DTYPE = ("input_layernorm.weight", "pre_feedforward_layernorm.weight",
                "post_feedforward_layernorm.weight", "mlp.pre_expert_norm.weight",
                "mlp.experts.post_expert_norm.weight")


def load_layer(checkpoint, config, index, *, device="cuda", dtype=None):
    """Build the official layer on meta and fill it from the checkpoint.

    ``dtype=torch.float32`` upcasts stored tensors exactly (BF16 -> F32 is lossless;
    expert stacks per expert call) except the (1 + w) norm weights, which keep the
    official BF16 add; ``None`` keeps checkpoint dtypes (BF16, F32 router/bias).
    """
    import torch
    layer = check_layer(checkpoint, config, index)
    for name, _ in list(layer.named_parameters()) + list(layer.named_buffers()):
        value = checkpoint.get(f"{PREFIX}layers.{index}.{name}", device)
        parent, _, leaf = name.rpartition(".")
        owner = layer.get_submodule(parent) if parent else layer
        if dtype is not None and name in EXPERT_STACKS:
            del owner._parameters[leaf]
            object.__setattr__(owner, leaf, LazyExperts(value, dtype))
            continue
        if dtype is not None and name not in NATIVE_DTYPE:
            value = value.to(dtype)
        if leaf in owner._parameters:
            owner._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)
        else:
            owner._buffers[leaf] = value
    checkpoint.close()
    return layer.eval()


def load_outer(checkpoint, config, *, device="cuda", dtype=None, head=True):
    """Embedding (with its weightless norm), final norm and untied head."""
    import torch
    reference, _ = _module()
    with torch.device("meta"):
        embed = reference.MuseSparkTextNormedEmbedding(config.vocab_size, config.hidden_size,
                                                       config.pad_token_id, config.rms_norm_eps)
        norm = reference.MuseSparkRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    with torch.device(device):
        rotary = reference.MuseSparkTextRotaryEmbedding(config)
    modules = {}
    for module, name, key in ((embed, "weight", PREFIX + "embed_tokens.weight"),
                              *(((norm, "weight", PREFIX + "norm.weight"),) if head else ())):
        value = checkpoint.get(key, device)
        if dtype is not None:
            value = value.to(dtype)
        module._parameters[name] = torch.nn.Parameter(value, requires_grad=False)
    modules.update(embed=embed.eval(), rotary=rotary.eval())
    if head:
        weight = checkpoint.get(HEAD, device)
        modules.update(norm=norm.eval(), head=weight.to(dtype) if dtype is not None else weight)
    checkpoint.close()
    return modules


def logits(hidden, head, config, linear=None):
    """Official head: lm_head, x output_multiplier, tanh softcap 20 (final norm applied before)."""
    import torch
    import torch.nn.functional as F
    value = (linear or F.linear)(hidden, head)
    value = value * config.output_multiplier
    value = value / config.final_logit_softcapping
    value = torch.tanh(value)
    return value * config.final_logit_softcapping


def layer_mask(config, index, length, *, device, dtype, rows=None):
    """The official eager additive mask (causal, plus kv > q - window on sliding layers)."""
    import torch
    rows = rows if rows is not None else (0, length)
    q = torch.arange(*rows, device=device)[:, None]
    k = torch.arange(length, device=device)[None, :]
    allowed = k <= q
    if config.layer_types[index] == "sliding_attention":
        allowed &= k > q - config.sliding_window
    mask = torch.zeros(allowed.shape, device=device, dtype=dtype)
    return mask.masked_fill_(~allowed, torch.finfo(dtype).min)[None, None]


def layer_forward(layer, config, index, hidden, rotary, *, mask=None):
    """Run the unchanged official decoder layer on one unpadded window [1, T, H]."""
    import torch
    length = hidden.shape[1]
    positions = torch.arange(length, device=hidden.device)[None]
    embeddings = rotary(hidden, positions) if config.layer_rope_theta[index] else None
    if mask is None:
        mask = layer_mask(config, index, length, device=hidden.device, dtype=hidden.dtype)
    return layer(hidden, position_embeddings=embeddings, attention_mask=mask,
                 position_ids=positions, past_key_values=None)


ROWS = 128


def causal_tiles(official):
    """Official eager attention per 128-row query tile with fixed key extent start+128.

    The tile's rows see the official additive mask (causal, plus the sliding window)
    sliced and padded with its own fill value; a row's matmul and softmax shapes
    depend only on its position, never on the window length.
    """
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
    """Fixed-M128 linears and bounded eager attention around the official module."""
    import torch.nn.functional as F
    import torch
    import shape_invariant
    previous = F.linear, torch.einsum, reference.eager_attention_forward
    F.linear = shape_invariant.linear
    torch.einsum = shape_invariant.einsum
    reference.eager_attention_forward = causal_tiles(previous[2])
    try:
        yield
    finally:
        F.linear, torch.einsum, reference.eager_attention_forward = previous


@contextmanager
def capture_routes(layer):
    """Record the router's bias-selected top-8 ids and renormalized weights.

    Observes MuseSparkTopkRouter.forward's return value only; logits are unchanged.
    """
    records = []

    def hook(module, args, output):
        _, weights, indices = output
        records.append((indices.detach(), weights.detach()))
    handle = layer.mlp.gate.register_forward_hook(hook)
    try:
        yield records
    finally:
        handle.remove()

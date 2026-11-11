"""Reviewed, fail-closed GLM 5.3 Flash BF16 -> EXL3 K4/Core-K6 inventory."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math

from kimi_k3.uniform_k2.storage import canonical, read_header
from glm5.core_k6.recipe import packed_specs, packed_bytes, _geometry

BASE_REPO = "zai-org/GLM-5.3-Flash-BF16"
BASE_REVISION = "f9fed3bbc82291073373d67452e9f25317ef7577"
TARGET_REPO = "wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K4-Core-K6"
BASE_PAYLOAD_BYTES = 642_646_653_816
BASE_TENSOR_COUNT = 38_770
BASE_CORE_MODULE_COUNT = 316
MTP_CORE_MODULE_COUNT = 8
EXPERT_MODULE_COUNT = 37_152
MODULE_COUNT = BASE_CORE_MODULE_COUNT + MTP_CORE_MODULE_COUNT + EXPERT_MODULE_COUNT
TARGET_TENSOR_COUNT = BASE_TENSOR_COUNT + MODULE_COUNT * 3
TARGET_PAYLOAD_BYTES = 165_621_841_672
MCG_MULTIPLIER = 0xCBAC1FED
SUFFIXES = ("trellis", "suh", "svh", "mcg")
PREFIX = "model.language_model."
LAYER_COUNT = 46
TARGET_LAYER_COUNT = 45
MTP_LAYER = 45
MLA_LAYERS = (*range(3, 44, 4), MTP_LAYER)
KDA_LAYERS = tuple(i for i in range(TARGET_LAYER_COUNT) if i not in MLA_LAYERS)
SOURCE_CONFIG_SHA256 = "cef231c35fa99c8d2ae06b0d76dfd090499fbac9f5f51b854b1f32760e3a8e18"


def reviewed_core_modules(*, include_mtp=True, core_bits=6):
    if type(core_bits) is not int or core_bits not in (6, 8):
        raise ValueError("unreviewed GLM Flash core bitrate")
    selected = {}
    for layer in range(LAYER_COUNT if include_mtp else TARGET_LAYER_COUNT):
        prefix = f"{PREFIX}layers.{layer}."
        bits = 4 if layer == MTP_LAYER else core_bits
        group = "mtp_core" if layer == MTP_LAYER else "base_core"
        attention = (("q_proj", 4096, 8192), ("k_proj", 4096, 8192),
                     ("v_proj", 4096, 8192), ("o_proj", 8192, 4096)) if layer in KDA_LAYERS else (
                     ("q_a_proj", 4096, 1536), ("q_b_proj", 1536, 16384),
                     ("kv_a_proj_with_mqa", 4096, 512), ("o_proj", 16384, 4096))
        for name, inputs, outputs in attention:
            selected[prefix + "self_attn." + name] = _geometry(inputs, outputs, bits, group)
        mlp = "mlp." if layer < 3 else "mlp.shared_experts."
        intermediate = 12288 if layer < 3 else 2048
        for name, inputs, outputs in (("gate_proj", 4096, intermediate),
                ("up_proj", 4096, intermediate), ("down_proj", intermediate, 4096)):
            selected[prefix + mlp + name] = _geometry(inputs, outputs, bits, group)
        if layer == MTP_LAYER:
            selected[prefix + "eh_proj"] = _geometry(8192, 4096, 4, group)
    selected["lm_head"] = _geometry(4096, 154880, core_bits, "base_core")
    return selected


def reviewed_expert_modules(*, include_mtp=True):
    selected = {}
    for layer in range(3, LAYER_COUNT if include_mtp else TARGET_LAYER_COUNT):
        for expert in range(288):
            prefix = f"{PREFIX}layers.{layer}.mlp.experts.{expert}."
            for name, inputs, outputs in (("gate_proj", 4096, 2048),
                    ("up_proj", 4096, 2048), ("down_proj", 2048, 4096)):
                selected[prefix + name] = _geometry(inputs, outputs, 4, "routed")
    return selected


def reviewed_modules():
    return reviewed_core_modules() | reviewed_expert_modules()


def reviewed_source_specs():
    """Every source tensor, including the unchanged bundled vision tower."""
    specs = {module + ".weight": ("BF16", [g["out_features"], g["in_features"]])
             for module, g in reviewed_modules().items()}
    specs[PREFIX + "embed_tokens.weight"] = ("BF16", [154880, 4096])
    specs[PREFIX + "norm.weight"] = ("BF16", [4096])
    for layer in range(LAYER_COUNT):
        p = f"{PREFIX}layers.{layer}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            specs[p + name + ".weight"] = ("BF16", [4096])
        if layer < TARGET_LAYER_COUNT:
            for side in ("attn", "ffn"):
                specs[p + f"hc_{side}_base"] = ("F32", [24])
                specs[p + f"hc_{side}_scale"] = ("F32", [3])
                specs[p + f"hc_{side}_fn"] = ("BF16", [24, 16384])
        if layer >= 3:
            specs[p + "mlp.gate.weight"] = ("BF16", [288, 4096])
            specs[p + "mlp.gate.e_score_correction_bias"] = ("F32", [288])
        if layer in KDA_LAYERS:
            for name, shape in (("b_proj.weight", [64, 4096]),
                    ("f_a_proj.weight", [128, 4096]), ("f_b_proj.weight", [8192, 128]),
                    ("g_a_proj.weight", [128, 4096]), ("g_b_proj.weight", [8192, 128]),
                    ("o_norm.weight", [128]),
                    *((f"{kind}_conv1d.weight", [8192, 1, 4]) for kind in "qkv")):
                specs[p + "self_attn." + name] = ("BF16", shape)
            specs[p + "self_attn.A_log"] = ("F32", [64])
            specs[p + "self_attn.dt_bias"] = ("F32", [8192])
        else:
            for name, shape in (("q_a_layernorm.weight", [1536]),
                    ("kv_a_layernorm.weight", [512]), ("kv_b_proj.weight", [32768, 512]),
                    ("indexer.wq_b.weight", [4096, 1536]), ("indexer.wk.weight", [128, 4096]),
                    ("indexer.weights_proj.weight", [32, 4096]), ("indexer.k_norm.weight", [128]),
                    ("indexer.k_norm.bias", [128]), ("indexer.index_kpool_compress_ape", [4, 128]),
                    ("indexer.index_kpool_compress_gate", [128, 4096])):
                specs[p + "self_attn." + name] = ("BF16", shape)
        if layer == MTP_LAYER:
            for name in ("enorm", "hnorm", "shared_head.norm"):
                specs[p + name + ".weight"] = ("BF16", [4096])
    vision = {
        "attn.k_norm.weight": [64], "attn.q_norm.weight": [64],
        "attn.proj.bias": [1024], "attn.proj.weight": [1024, 1024],
        "attn.qkv.bias": [3072], "attn.qkv.weight": [3072, 1024],
        "mlp.down_proj.bias": [1024], "mlp.down_proj.weight": [1024, 4096],
        "mlp.gate_proj.bias": [4096], "mlp.gate_proj.weight": [4096, 1024],
        "mlp.up_proj.bias": [4096], "mlp.up_proj.weight": [4096, 1024],
        "norm1.weight": [1024], "norm2.weight": [1024],
    }
    for layer in range(24):
        for name, shape in vision.items():
            specs[f"model.visual.blocks.{layer}." + name] = ("BF16", shape)
    for name, shape in {
        "downsample.bias": [4096], "downsample.weight": [4096, 1024, 2, 2],
        "merger.down_proj.weight": [4096, 10240], "merger.gate_proj.weight": [10240, 4096],
        "merger.post_projection_norm.bias": [4096], "merger.post_projection_norm.weight": [4096],
        "merger.proj.weight": [4096, 4096], "merger.up_proj.weight": [10240, 4096],
        "patch_embed.proj.bias": [1024], "patch_embed.proj.weight": [1024, 3, 2, 14, 14],
        "post_layernorm.weight": [1024],
    }.items():
        specs["model.visual." + name] = ("BF16", shape)
    return specs


def validate_config(document):
    # Pin the entire reviewed source configuration, including vision, layer
    # schedules, KDA recurrent dimensions and mHC. Architecture drift must
    # receive a new review instead of silently reusing this recipe.
    if hashlib.sha256(canonical(document).encode()).hexdigest() != SOURCE_CONFIG_SHA256:
        raise ValueError("unreviewed GLM Flash source configuration")
    if document.get("quantization_config") or document["text_config"].get("quantization_config"):
        raise ValueError("source is already quantized")


def inventory(source):
    document = getattr(source, "document", None)
    if document is None:
        import json
        document = json.loads((source.snapshot / "config.json").read_text())
    validate_config(document)
    if source.config != document["text_config"]:
        raise ValueError("GLM Flash source text config differs from source document")
    specs, selected = reviewed_source_specs(), reviewed_modules()
    if set(source.weight_map) != set(specs):
        raise ValueError(f"source tensor inventory differs: missing={sorted(set(specs)-set(source.weight_map))[:5]} "
                         f"extra={sorted(set(source.weight_map)-set(specs))[:5]}")
    by_shard = defaultdict(set)
    for name, shard in source.weight_map.items():
        if not isinstance(shard, str) or not shard.endswith(".safetensors") or "/" in shard or "\\" in shard or shard in (".", ".."):
            raise ValueError("source shard must be a local safetensors basename")
        by_shard[shard].add(name)
    total = 0
    for shard, names in by_shard.items():
        header, _ = read_header(source.snapshot / shard)
        if set(header) - {"__metadata__"} != names:
            raise ValueError(f"header/index inventory differs: {shard}")
        for name in names:
            actual, (dtype, shape) = header[name], specs[name]
            size = math.prod(shape) * {"BF16": 2, "F32": 4}[dtype]
            if (actual["dtype"] != dtype or actual["shape"] != shape
                    or actual["data_offsets"][1] - actual["data_offsets"][0] != size):
                raise ValueError(f"unreviewed source dtype/geometry: {name}")
            total += size
    groups = Counter(g["group"] for g in selected.values())
    if (len(specs) != BASE_TENSOR_COUNT or total != BASE_PAYLOAD_BYTES
            or groups != dict(base_core=BASE_CORE_MODULE_COUNT, mtp_core=MTP_CORE_MODULE_COUNT, routed=EXPERT_MODULE_COUNT)):
        raise ValueError("reviewed GLM Flash source totals differ")
    selected_source = sum(g["in_features"] * g["out_features"] * 2 for g in selected.values())
    packed = sum(packed_bytes(g) for g in selected.values())
    if total - selected_source + packed != TARGET_PAYLOAD_BYTES:
        raise ValueError("reviewed GLM Flash target payload differs")
    parts = {group: dict(modules=groups[group],
                        source_bytes=sum(g["in_features"]*g["out_features"]*2 for g in selected.values() if g["group"] == group),
                        packed_bytes=sum(packed_bytes(g) for g in selected.values() if g["group"] == group))
             for group in groups}
    return dict(source_payload_bytes=total, source_tensors=len(specs), selected_modules=len(selected),
                selected_source_bytes=selected_source, selected_packed_bytes=packed,
                preserved_bytes=total-selected_source, preserved_tensors=len(specs)-len(selected),
                target_payload_bytes=total-selected_source+packed,
                target_tensors=TARGET_TENSOR_COUNT, shards=len(by_shard), parts=parts)


def selected_modules(source):
    inventory(source)
    return reviewed_modules()

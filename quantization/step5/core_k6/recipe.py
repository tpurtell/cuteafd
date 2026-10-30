"""Reviewed Step-5 BF16 -> mixed K4/K6 EXL3 inventory, including split experts."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
import re

from kimi_k3.uniform_k2.storage import canonical, read_header
from glm5.core_k6.recipe import packed_specs, packed_bytes, _geometry

BASE_REPO = "stepfun-ai/Step-5-Preview-BF16"
BASE_REVISION = "7734b7ffac118d70d8398bfa3eb0fab007eb45b1"
TARGET_REPO = "wrldsuksgo2mars/Step-5-Preview-EXL3-K4-Core-K6"
BASE_PAYLOAD_BYTES = 1_205_950_680_832
BASE_TENSOR_COUNT = 2_449
BASE_CORE_MODULE_COUNT = 645
MTP_CORE_MODULE_COUNT = 27
EXPERT_MODULE_COUNT = 92_928
MODULE_COUNT = BASE_CORE_MODULE_COUNT + MTP_CORE_MODULE_COUNT + EXPERT_MODULE_COUNT
SOURCE_SELECTED_TENSOR_COUNT = BASE_CORE_MODULE_COUNT + MTP_CORE_MODULE_COUNT + 88 * 3
TARGET_TENSOR_COUNT = BASE_TENSOR_COUNT - SOURCE_SELECTED_TENSOR_COUNT + MODULE_COUNT * 4
MCG_MULTIPLIER = 0xCBAC1FED
SUFFIXES = ("trellis", "suh", "svh", "mcg")
PREFIX = "model."
TARGET_LAYER_COUNT = 92
MTP_LAYERS = (92, 93, 94)
LAYER_COUNT = 95
FULL_ATTENTION_LAYERS = tuple(range(3, 92, 4))
ROUTED_LAYERS = tuple(range(3, 91))
K6_ROUTED_LAYERS = (88, 89, 90)
SOURCE_CONFIG_SHA256 = "968b76ac10de53839530d1ae30f0dfa91afb9a264a1a4b955352dd2a5a94f726"
_FUSED = re.compile(r"^(model\.layers\.(?:[0-9]+)\.moe)\.experts\.([0-9]+)\.(gate_proj|up_proj|down_proj)$")


def reviewed_core_modules(*, include_mtp=True):
    selected = {}
    for layer in range(LAYER_COUNT if include_mtp else TARGET_LAYER_COUNT):
        prefix = f"model.layers.{layer}."
        bits, group = (4, "mtp_core") if layer in MTP_LAYERS else (6, "base_core")
        for name, inputs, outputs in (("q_proj", 4096, 12288), ("k_proj", 4096, 768),
                                      ("v_proj", 4096, 768), ("o_proj", 12288, 4096)):
            selected[prefix + "self_attn." + name] = _geometry(inputs, outputs, bits, group)
        mlp, intermediate = ("share_expert.", 1536) if layer in ROUTED_LAYERS else ("mlp.", 13824)
        for name, inputs, outputs in (("gate_proj", 4096, intermediate),
                ("up_proj", 4096, intermediate), ("down_proj", intermediate, 4096)):
            selected[prefix + mlp + name] = _geometry(inputs, outputs, bits, group)
        if layer in MTP_LAYERS:
            selected[prefix + "eh_proj"] = _geometry(8192, 4096, 4, group)
            # These are actual stage-local tensors, not loader aliases. MTP K4
            # throughout includes each of its independently calibrated heads.
            selected[prefix + "transformer.shared_head.output"] = _geometry(4096, 128896, 4, group)
    selected["lm_head"] = _geometry(4096, 128896, 6, "base_core")
    return selected


def reviewed_expert_modules(*, include_mtp=True):
    selected = {}
    for layer in ROUTED_LAYERS:
        bits = 6 if layer in K6_ROUTED_LAYERS else 4
        for expert in range(352):
            prefix = f"model.layers.{layer}.moe.experts.{expert}."
            for name, inputs, outputs in (("gate_proj", 4096, 1536),
                    ("up_proj", 4096, 1536), ("down_proj", 1536, 4096)):
                selected[prefix + name] = _geometry(inputs, outputs, bits, "routed")
    return selected


def reviewed_modules():
    return reviewed_core_modules() | reviewed_expert_modules()


def source_tensor(module):
    """Return (source name, first-axis expert index or None); never load data."""
    m = _FUSED.fullmatch(module)
    if m:
        expert = int(m[2])
        layer = int(m[1].split('.')[2])
        if layer not in ROUTED_LAYERS or not 0 <= expert < 352:
            raise ValueError("expert outside reviewed source")
        return f"{m[1]}.{m[3]}.weight", expert
    return module + ".weight", None


def selected_source_names(selection=None):
    return {source_tensor(module)[0] for module in (reviewed_modules() if selection is None else selection)}


def reviewed_source_specs():
    specs = {module + ".weight": ("BF16", [g["out_features"], g["in_features"]])
             for module, g in reviewed_core_modules().items()}
    specs["model.embed_tokens.weight"] = ("BF16", [128896, 4096])
    specs["model.norm.weight"] = ("F32", [4096])
    for layer in range(LAYER_COUNT):
        p = f"model.layers.{layer}."
        for name, shape in (("input_layernorm.weight", [4096]),
                ("post_attention_layernorm.weight", [4096]),
                ("self_attn.q_norm.weight", [192]), ("self_attn.k_norm.weight", [192])):
            specs[p + name] = ("F32", shape)
        specs[p + "self_attn.g_proj.weight"] = ("BF16", [64, 4096])
        if layer in ROUTED_LAYERS:
            for name, shape in (("gate_proj", [352, 1536, 4096]), ("up_proj", [352, 1536, 4096]),
                                ("down_proj", [352, 4096, 1536])):
                specs[p + "moe." + name + ".weight"] = ("BF16", shape)
            specs[p + "moe.gate.weight"] = ("BF16", [352, 4096])
            specs[p + "moe.router_bias"] = ("F32", [352])
        if layer in FULL_ATTENTION_LAYERS:
            for name, dtype, shape in (("sparse_indexer_q.weight", "BF16", [4096, 4096]),
                    ("sparse_indexer_k.weight", "BF16", [256, 4096]),
                    ("sparse_indexer_z.weight", "BF16", [256, 4096]),
                    ("sparse_indexer_w.weight", "F32", [16, 4096]),
                    ("sparse_indexer_q_norm.weight", "F32", [256]),
                    ("sparse_indexer_k_norm.weight", "F32", [256]),
                    ("sparse_indexer_k_norm.bias", "F32", [256]), ("ssmax_s", "F32", [64])):
                specs[p + "self_attn." + name] = (dtype, shape)
        if layer in MTP_LAYERS:
            for name in ("enorm", "hnorm", "transformer.shared_head.norm"):
                specs[p + name + ".weight"] = ("F32", [4096])
    vision = {"attn.in_proj_bias": [4608], "attn.in_proj_weight": [4608, 1536],
        "attn.out_proj.bias": [1536], "attn.out_proj.weight": [1536, 1536],
        "ln_1.bias": [1536], "ln_1.weight": [1536], "ln_2.bias": [1536], "ln_2.weight": [1536],
        "ls_1.gamma": [1536], "ls_2.gamma": [1536], "mlp.c_fc.bias": [8960],
        "mlp.c_fc.weight": [8960, 1536], "mlp.c_proj.bias": [1536], "mlp.c_proj.weight": [1536, 8960]}
    for layer in range(47):
        for name, shape in vision.items():
            specs[f"vision_model.transformer.resblocks.{layer}." + name] = ("BF16", shape)
    for name, shape in {"vision_model.conv1.weight": [1536, 3, 14, 14],
        "vision_model.ln_pre.bias": [1536], "vision_model.ln_pre.weight": [1536],
        "vision_model.positional_embedding": [2704, 1536],
        "vision_model.vit_downsampler1.bias": [3072], "vision_model.vit_downsampler1.weight": [3072, 1536, 3, 3],
        "vision_model.vit_downsampler2.bias": [6144], "vision_model.vit_downsampler2.weight": [6144, 3072, 3, 3],
        "vit_large_projector.weight": [4096, 6144]}.items():
        specs[name] = ("BF16", shape)
    return specs


def reviewed_header_aliases():
    """Explicit redundant source payloads ignored by the authoritative HF index."""
    return {"model-00001.safetensors": {"model.embed_tokens.weight"},
            "model-00024.safetensors": {n for n in reviewed_source_specs() if n.startswith("model.layers.92.")}}


def validate_header_inventory(shard, header, indexed_names):
    aliases = reviewed_header_aliases().get(shard, set())
    if set(header) - {"__metadata__"} != set(indexed_names) | aliases or aliases & set(indexed_names):
        raise ValueError(f"header/index inventory differs: {shard}")
    specs = reviewed_source_specs()
    for name in aliases:
        dtype, shape = specs[name]
        v = header[name]
        if (v["dtype"] != dtype or v["shape"] != shape
                or v["data_offsets"][1] - v["data_offsets"][0] != math.prod(shape) * {"BF16":2,"F32":4}[dtype]):
            raise ValueError(f"redundant source tensor geometry differs: {name}")


def validate_config(document):
    if hashlib.sha256(canonical(document).encode()).hexdigest() != SOURCE_CONFIG_SHA256:
        raise ValueError("unreviewed Step-5 source configuration")
    if document.get("quantization_config") or document["text_config"].get("quantization_config"):
        raise ValueError("source is already quantized")


def inventory(source):
    document = source.document
    validate_config(document)
    if source.config != document["text_config"]:
        raise ValueError("Step-5 text config differs from source document")
    specs, selected = reviewed_source_specs(), reviewed_modules()
    if set(source.weight_map) != set(specs):
        raise ValueError(f"source tensor inventory differs: missing={sorted(set(specs)-set(source.weight_map))[:5]} "
                         f"extra={sorted(set(source.weight_map)-set(specs))[:5]}")
    by_shard = defaultdict(set)
    for name, shard in source.weight_map.items():
        if not isinstance(shard, str) or not shard.endswith('.safetensors') or '/' in shard or '\\' in shard:
            raise ValueError("source shard must be a local safetensors basename")
        by_shard[shard].add(name)
    total = 0
    for shard, names in by_shard.items():
        header, _ = read_header(source.snapshot / shard)
        validate_header_inventory(shard, header, names)
        for name in names:
            actual, (dtype, shape) = header[name], specs[name]
            size = math.prod(shape) * {"BF16": 2, "F32": 4}[dtype]
            if (actual["dtype"] != dtype or actual["shape"] != shape
                    or actual["data_offsets"][1] - actual["data_offsets"][0] != size):
                raise ValueError(f"unreviewed source dtype/geometry: {name}")
            total += size
    groups = Counter(g['group'] for g in selected.values())
    if (len(specs) != BASE_TENSOR_COUNT or total != BASE_PAYLOAD_BYTES
            or groups != dict(base_core=BASE_CORE_MODULE_COUNT,mtp_core=MTP_CORE_MODULE_COUNT,routed=EXPERT_MODULE_COUNT)):
        raise ValueError("reviewed Step-5 source totals differ")
    names = selected_source_names(selected)
    selected_source = sum(math.prod(specs[n][1])*2 for n in names)
    packed = sum(packed_bytes(g) for g in selected.values())
    if total - selected_source + packed != TARGET_PAYLOAD_BYTES:
        raise ValueError("reviewed Step-5 target payload differs")
    parts = {}
    for group, bits in (("base_core",6),("mtp_core",4),("routed",4),("routed",6)):
        subset = [g for g in selected.values() if g['group']==group and g['bits']==bits]
        parts[f"{group}_k{bits}"] = dict(modules=len(subset),source_bytes=sum(g['in_features']*g['out_features']*2 for g in subset),packed_bytes=sum(packed_bytes(g) for g in subset))
    return dict(source_payload_bytes=total,source_tensors=len(specs),selected_modules=len(selected),
        selected_source_tensors=len(names),selected_source_bytes=selected_source,selected_packed_bytes=packed,
        preserved_bytes=total-selected_source,preserved_tensors=len(specs)-len(names),
        target_payload_bytes=total-selected_source+packed,target_tensors=TARGET_TENSOR_COUNT,shards=len(by_shard),parts=parts)


def selected_modules(source):
    inventory(source)
    return reviewed_modules()


# Scale vectors and one int32 MCG scalar per independently stored projection.
TARGET_PAYLOAD_BYTES = 315_357_147_520

"""Fail-closed K3.1 MXFP4 -> interleaved K2 experts / K6 core inventory."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import re
from types import SimpleNamespace
from pathlib import Path
import json

from kimi_k3.core_k6.recipe import reviewed_modules as _k3_core, packed_specs as _packed_specs
from kimi_k3.uniform_k2.storage import canonical, read_header


BASE_REPO = "moonshotai/Kimi-K3.1-Pro"
BASE_REVISION = "6d0a0202956796543364d666ab19c6ad1a494ee1"
TARGET_REPO = "wrldsuksgo2mars/Kimi-K3.1-Pro-EXL3-K2-Core-K6"
BASE_TENSOR_COUNT = 497_340
BASE_PAYLOAD_BYTES = 1_561_217_377_280
CORE_MODULE_COUNT = 926
EXPERT_MODULE_COUNT = 247_296
MODULE_COUNT = CORE_MODULE_COUNT + EXPERT_MODULE_COUNT
LAYER_COUNT = 93
SHARD_COUNT = 94
MCG_MULTIPLIER = 0xCBAC1FED
SUFFIXES = ("trellis", "suh", "svh", "mcg")
DSA_LAYERS = tuple(range(3, 92, 4)) + (92,)
_LAYER = re.compile(r"language_model\.model\.layers\.([0-9]+)\.")


def packed_specs(module, geometry, *, bits=None):
    return _packed_specs(module, geometry, bits=geometry["bits"] if bits is None else bits)


def packed_bytes(geometry):
    return sum(math.prod(shape) * {"I16": 2, "F16": 2, "I32": 4}[dtype]
               for dtype, shape in packed_specs("projection", geometry).values())


def reviewed_core_modules():
    return {name: dict(g, bits=6, group="core") for name, g in _k3_core().items()}


def reviewed_expert_modules():
    return {f"language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}.{name}":
            dict(in_features=inputs, out_features=outputs, stored_out_features=outputs,
                 bits=2, group="routed")
            for layer in range(1, LAYER_COUNT) for expert in range(896)
            for name, inputs, outputs in (("w1", 3584, 3072), ("w3", 3584, 3072),
                                          ("w2", 3072, 3584))}


def reviewed_modules():
    return reviewed_core_modules() | reviewed_expert_modules()


def source_names(module, geometry):
    return ((module + ".weight_packed", module + ".weight_scale")
            if geometry["bits"] == 2 else (module + ".weight",))


def reviewed_source_specs():
    """Every tensor has a reviewed dtype and shape, including preserved tensors."""
    specs = {n + ".weight": ("BF16", [g["out_features"], g["in_features"]])
             for n, g in reviewed_core_modules().items()}
    for n, g in reviewed_expert_modules().items():
        specs[n + ".weight_packed"] = ("U8", [g["out_features"], g["in_features"] // 2])
        specs[n + ".weight_scale"] = ("U8", [g["out_features"], g["in_features"] // 32])
    for layer in range(LAYER_COUNT):
        p = f"language_model.model.layers.{layer}."
        for name in ("input_layernorm", "post_attention_layernorm", "self_attention_res_norm", "mlp_res_norm"):
            specs[p + name + ".weight"] = ("BF16", [7168])
        for name in ("self_attention_res_proj", "mlp_res_proj"):
            specs[p + name + ".weight"] = ("BF16", [1, 7168])
        if layer == 0:
            for name, shape in (("gate_proj", [33792, 7168]), ("up_proj", [33792, 7168]),
                                ("down_proj", [7168, 33792])):
                specs[p + "mlp." + name + ".weight"] = ("BF16", shape)
        else:
            specs[p + "block_sparse_moe.gate.weight"] = ("BF16", [896, 7168])
            specs[p + "block_sparse_moe.gate.e_score_correction_bias"] = ("F32", [896])
            specs[p + "block_sparse_moe.routed_expert_norm.weight"] = ("BF16", [3584])
        if layer in DSA_LAYERS:
            for name, shape in (("q_a_layernorm.weight", [1536]), ("kv_a_layernorm.weight", [512]),
                ("kv_b_proj.weight", [24576, 512]), ("indexer.wq_b.weight", [4096, 1536]),
                ("indexer.wk.weight", [128, 7168]), ("indexer.weights_proj.weight", [32, 7168]),
                ("indexer.k_norm.weight", [128]), ("indexer.k_norm.bias", [128])):
                specs[p + "self_attn." + name] = ("BF16", shape)
        else:
            for name, shape in (("A_log", [128]), ("dt_bias", [12288]),
                ("q_conv1d.weight", [12288, 1, 4]), ("k_conv1d.weight", [12288, 1, 4]),
                ("v_conv1d.weight", [12288, 1, 4]), ("o_norm.weight", [128])):
                specs[p + "self_attn." + name] = ("F32", shape)
            for name, shape in (("b_proj.weight", [96, 7168]), ("f_a_proj.weight", [128, 7168]),
                                ("f_b_proj.weight", [12288, 128])):
                specs[p + "self_attn." + name] = ("BF16", shape)
    for name, shape in (("language_model.model.embed_tokens.weight", [163840, 7168]),
        ("language_model.model.norm.weight", [7168]), ("language_model.model.output_attn_res_norm.weight", [7168]),
        ("language_model.model.output_attn_res_proj.weight", [1, 7168]),
        ("mm_projector.post_norm.weight", [7168]), ("mm_projector.proj.0.weight", [4096, 4096]),
        ("mm_projector.proj.2.weight", [7168, 4096]),
        ("vision_tower.encoder.final_layernorm.weight", [1024]),
        ("vision_tower.patch_embed.pos_emb.weight", [64, 64, 1024]),
        ("vision_tower.patch_embed.proj.weight", [1024, 3, 14, 14])):
        specs[name] = ("BF16", shape)
    for layer in range(27):
        for name, shape in (("mlp.fc0", [4096, 1024]), ("mlp.fc1", [1024, 4096]),
                ("norm0", [1024]), ("norm1", [1024]), ("wo", [1024, 1536]), ("wqkv", [4608, 1024])):
            specs[f"vision_tower.encoder.blocks.{layer}.{name}.weight"] = ("BF16", shape)
    return specs


def header_source(snapshot):
    snapshot = Path(snapshot).resolve(strict=True)
    config = json.loads((snapshot / "config.json").read_text())
    weight_map = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    return SimpleNamespace(snapshot=snapshot, config=config, text_config=config["text_config"], weight_map=weight_map)


def derive_attention_layers(weight_map, config):
    """Infer layer kind from actual projection names, then cross-check 1-based config."""
    kinds = defaultdict(set)
    for name in weight_map:
        match = _LAYER.match(name)
        if match and ".self_attn." in name:
            if name.endswith(".self_attn.q_proj.weight"):
                kinds["kda"].add(int(match[1]))
            if name.endswith(".self_attn.indexer.wq_b.weight"):
                kinds["dsa"].add(int(match[1]))
    kda, dsa = kinds["kda"], kinds["dsa"]
    if kda & dsa or kda | dsa != set(range(config["num_hidden_layers"])):
        raise ValueError("attention headers do not partition every layer")
    linear, sparse = config["linear_attn_config"], config["dsa_config"]
    if (dsa != {x - 1 for x in sparse["dsa_layers"]}
            or dsa != {x - 1 for x in linear["full_attn_layers"]}
            or kda != {x - 1 for x in linear["kda_layers"]}):
        raise ValueError("header-derived attention kinds disagree with 1-based config")
    if dsa != set(DSA_LAYERS):
        raise ValueError("unreviewed DSA layer placement")
    return dict(kda=sorted(kda), dsa=sorted(dsa))


def inventory(source):
    document = source.config
    if document.get("model_type") != "kimi_k31" or document.get("architectures") != ["KimiK31ForConditionalGeneration"]:
        raise ValueError("expected official Kimi K3.1 Pro checkpoint")
    if source.text_config != document["text_config"]:
        raise ValueError("text config differs from source document")
    # Hash pins all numerical, routing, sparse-index and compression settings.
    if hashlib.sha256(canonical(document).encode()).hexdigest() != SOURCE_CONFIG_SHA256:
        raise ValueError("unreviewed Kimi K3.1 configuration")
    attention = derive_attention_layers(source.weight_map, source.text_config)
    specs = reviewed_source_specs()
    if set(source.weight_map) != set(specs):
        raise ValueError("source tensor names differ from reviewed complete inventory")
    shards = defaultdict(set)
    for name, shard in source.weight_map.items():
        if not isinstance(shard, str) or Path(shard).name != shard or not shard.endswith(".safetensors"):
            raise ValueError("source shard must be a local safetensors basename")
        shards[shard].add(name)
    total = 0
    for shard, names in shards.items():
        header, _ = read_header(source.snapshot / shard)
        if set(header) - {"__metadata__"} != names:
            raise ValueError("source shard contains missing, duplicate or unindexed tensors")
        for name in names:
            value = header[name]
            if (value["dtype"], value["shape"]) != specs[name]:
                raise ValueError(f"unreviewed source tensor geometry: {name}")
            total += value["data_offsets"][1] - value["data_offsets"][0]
    if len(shards) != 96 or total != BASE_PAYLOAD_BYTES or len(specs) != BASE_TENSOR_COUNT:
        raise ValueError("source totals differ from reviewed release")
    modules = reviewed_modules()
    selected_names = {n for module, g in modules.items() for n in source_names(module, g)}
    selected_source_bytes = sum(math.prod(specs[n][1]) * {"BF16": 2, "U8": 1}[specs[n][0]] for n in selected_names)
    packed = sum(packed_bytes(g) for g in modules.values())
    native = {n: spec for n, spec in specs.items() if n not in selected_names}
    native_bytes = sum(math.prod(shape) * {"BF16": 2, "F32": 4}[dtype] for dtype, shape in native.values())
    core_packed = sum(packed_bytes(g) for g in reviewed_core_modules().values())
    # Conservative TP2 weights: split selected dense matrices; all small native
    # non-vision pieces replicated, embedding and kv_b included at full size.
    native_text = sum(math.prod(shape) * {"BF16": 2, "F32": 4}[dtype] for n, (dtype, shape) in native.items()
                      if n.startswith("language_model."))
    return dict(source_tensors=len(specs), source_payload_bytes=total, source_shards=len(shards),
        selected_modules=len(modules), core_projections=CORE_MODULE_COUNT, routed_projections=EXPERT_MODULE_COUNT,
        selected_source_tensors=len(selected_names), selected_source_bytes=selected_source_bytes,
        selected_packed_bytes=packed, preserved_tensors=len(native), preserved_bytes=native_bytes,
        target_tensors=len(native) + len(modules) * 4, target_payload_bytes=native_bytes + packed,
        core_k6_bytes=core_packed, routed_k2_bytes=packed-core_packed,
        tp2_core_per_gpu_upper_bytes=core_packed // 2 + native_text, attention=attention)


def selected_modules(source):
    inventory(source)
    return reviewed_modules()


SOURCE_CONFIG_SHA256 = "e41fce80465638582755873fde372bc17364650f083d88d3709bcd97e7cd9cd2"
TARGET_TENSOR_COUNT = BASE_TENSOR_COUNT - CORE_MODULE_COUNT - EXPERT_MODULE_COUNT * 2 + MODULE_COUNT * 4
TARGET_PAYLOAD_BYTES = 731_632_865_912

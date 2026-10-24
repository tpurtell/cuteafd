"""Reviewed GLM 5.3 BF16 -> interleaved EXL3 K4 experts / K6 core inventory."""
from __future__ import annotations

from collections import Counter, defaultdict
import math

from kimi_k3.uniform_k2.storage import read_header

BASE_REPO = "zai-org/GLM-5.3-BF16"
BASE_REVISION = "9d2398f478cab2de883137db3a36ad2c96205e24"
TARGET_REPO = "wrldsuksgo2mars/GLM-5.3-EXL3-K4-Core-K6"
BASE_PAYLOAD_BYTES = 1_506_659_919_872
BASE_TENSOR_COUNT = 59_585
TARGET_PAYLOAD_BYTES = 385_362_552_236
BASE_CORE_MODULE_COUNT = 547
MTP_CORE_MODULE_COUNT = 8
EXPERT_MODULE_COUNT = 58_368
MODULE_COUNT = BASE_CORE_MODULE_COUNT + MTP_CORE_MODULE_COUNT + EXPERT_MODULE_COUNT
TARGET_TENSOR_COUNT = BASE_TENSOR_COUNT + MODULE_COUNT * 3
MCG_MULTIPLIER = 0xCBAC1FED
SUFFIXES = ("trellis", "suh", "svh", "mcg")
FULL_INDEXER_LAYERS = (0, 1, 2, *range(6, 75, 4))


def _geometry(inputs, outputs, bits, group):
    return dict(in_features=inputs, out_features=outputs,
                stored_out_features=(outputs + 127) // 128 * 128,
                bits=bits, group=group)


def reviewed_core_modules(*, include_mtp=True):
    """Only reviewed non-expert matrices; all MTP selections are K4."""
    selected = {}
    for layer in range(79 if include_mtp else 78):
        prefix = f"model.layers.{layer}."
        bits = 4 if layer == 78 else 6
        group = "mtp_core" if layer == 78 else "base_core"
        for name, inputs, outputs in (
            ("self_attn.q_a_proj", 6144, 2048),
            ("self_attn.q_b_proj", 2048, 16384),
            ("self_attn.kv_a_proj_with_mqa", 6144, 576),
            ("self_attn.o_proj", 16384, 6144),
        ):
            selected[prefix + name] = _geometry(inputs, outputs, bits, group)
        mlp = "mlp." if layer < 3 else "mlp.shared_experts."
        intermediate = 12288 if layer < 3 else 2048
        for name, inputs, outputs in (
            ("gate_proj", 6144, intermediate),
            ("up_proj", 6144, intermediate),
            ("down_proj", intermediate, 6144),
        ):
            selected[prefix + mlp + name] = _geometry(inputs, outputs, bits, group)
        if layer == 78:
            selected[prefix + "eh_proj"] = _geometry(12288, 6144, 4, group)
    selected["lm_head"] = _geometry(6144, 154880, 6, "base_core")
    return selected


def reviewed_expert_modules(*, include_mtp=True):
    selected = {}
    for layer in range(3, 79 if include_mtp else 78):
        for expert in range(256):
            prefix = f"model.layers.{layer}.mlp.experts.{expert}."
            for name, inputs, outputs in (
                ("gate_proj", 6144, 2048), ("up_proj", 6144, 2048),
                ("down_proj", 2048, 6144),
            ):
                selected[prefix + name] = _geometry(inputs, outputs, 4, "routed")
    return selected


def reviewed_modules():
    return reviewed_core_modules() | reviewed_expert_modules()


def packed_specs(module, geometry, *, bits=None):
    """Standard MCG buffers; logical output 576 occupies stored width 640."""
    inputs, outputs = geometry["in_features"], geometry["stored_out_features"]
    bits = geometry["bits"] if bits is None else bits
    if inputs <= 0 or inputs % 128 or outputs <= 0 or outputs % 128 or bits not in (4, 5, 6, 8):
        raise ValueError(f"unreviewed EXL3 geometry/bitrate: {module}")
    return {module + ".trellis": ("I16", [inputs // 16, outputs // 16, bits * 16]),
            module + ".suh": ("F16", [inputs]),
            module + ".svh": ("F16", [outputs]), module + ".mcg": ("I32", [])}


def packed_bytes(geometry, *, bits=None):
    return sum(math.prod(shape) * {"I16": 2, "F16": 2, "I32": 4}[dtype]
               for dtype, shape in packed_specs("projection", geometry, bits=bits).values())


def reviewed_source_specs():
    """Complete source inventory: unknown, missing or transformed tensors fail."""
    specs = {module + ".weight": ("BF16", [g["out_features"], g["in_features"]])
             for module, g in reviewed_modules().items()}
    specs["model.embed_tokens.weight"] = ("BF16", [154880, 6144])
    specs["model.norm.weight"] = ("BF16", [6144])
    for layer in range(79):
        prefix = f"model.layers.{layer}."
        for name, width in (("input_layernorm", 6144), ("post_attention_layernorm", 6144),
                            ("self_attn.q_a_layernorm", 2048), ("self_attn.kv_a_layernorm", 512)):
            specs[prefix + name + ".weight"] = ("BF16", [width])
        specs[prefix + "self_attn.kv_b_proj.weight"] = ("BF16", [28672, 512])
        if layer >= 3:
            specs[prefix + "mlp.gate.weight"] = ("BF16", [256, 6144])
            specs[prefix + "mlp.gate.e_score_correction_bias"] = ("F32", [256])
        if layer in FULL_INDEXER_LAYERS or layer == 78:
            for name, shape in (("wq_b.weight", [4096, 2048]), ("wk.weight", [128, 6144]),
                                ("weights_proj.weight", [32, 6144]),
                                ("k_norm.weight", [128]), ("k_norm.bias", [128])):
                specs[prefix + "self_attn.indexer." + name] = ("BF16", shape)
        if layer == 78:
            for name in ("enorm", "hnorm", "shared_head.norm"):
                specs[prefix + name + ".weight"] = ("BF16", [6144])
    return specs


def validate_config(config):
    expected = dict(model_type="glm_moe_dsa", num_hidden_layers=78, num_nextn_predict_layers=1,
        hidden_size=6144, vocab_size=154880, intermediate_size=12288, moe_intermediate_size=2048,
        first_k_dense_replace=3, n_routed_experts=256, n_shared_experts=1, num_experts_per_tok=8,
        num_attention_heads=64, q_lora_rank=2048, kv_lora_rank=512, qk_nope_head_dim=192,
        qk_rope_head_dim=64, v_head_dim=256, index_n_heads=32, index_head_dim=128,
        index_topk=2048, scoring_func="sigmoid", topk_method="noaux_tc", routed_scaling_factor=2.5,
        tie_word_embeddings=False, rope_interleave=True, indexer_rope_interleave=True,
        hidden_act="silu", n_group=1, topk_group=1, norm_topk_prob=True,
        moe_router_dtype="float32", rms_norm_eps=1e-5, dtype="bfloat16",
        rope_parameters=dict(rope_theta=8000000, rope_type="default"))
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"unreviewed GLM config {key}: {config.get(key)!r}")
    if config.get("quantization_config"):
        raise ValueError("source is already quantized")
    if config.get("indexer_types") != ["full" if layer in FULL_INDEXER_LAYERS else "shared" for layer in range(78)]:
        raise ValueError("unreviewed DSA indexer schedule")
    if config.get("mlp_layer_types") != ["dense"] * 3 + ["sparse"] * 75:
        raise ValueError("unreviewed GLM MLP schedule")


def inventory(source):
    """Validate one header at a time without reading tensor payloads."""
    validate_config(source.config)
    specs, selected = reviewed_source_specs(), reviewed_modules()
    if set(source.weight_map) != set(specs):
        raise ValueError(f"source tensor inventory differs: missing={sorted(set(specs)-set(source.weight_map))[:5]} "
                         f"extra={sorted(set(source.weight_map)-set(specs))[:5]}")
    by_shard = defaultdict(set)
    for name, shard in source.weight_map.items():
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
        raise ValueError("reviewed GLM source totals differ")
    selected_source = sum(g["in_features"] * g["out_features"] * 2 for g in selected.values())
    packed = sum(packed_bytes(g) for g in selected.values())
    if total - selected_source + packed != TARGET_PAYLOAD_BYTES:
        raise ValueError("reviewed GLM target payload differs")
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

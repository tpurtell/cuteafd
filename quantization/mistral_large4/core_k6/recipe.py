"""Pinned native Mistral Large 4 FP8 -> EXL3 K4 experts / K6 core recipe.

Every source tensor is accounted for before payload reads. Selected FP8
weights and their block scales are replaced together; retained weights,
including the FP8 indexer, keep their original payload bytes.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path

from kimi_k3.uniform_k2.storage import read_header

BASE_REPO = "mistralai/Mistral-Large-4-1T-A52B"
BASE_REVISION = "f12dabdc0c1e99e7f7924461a6f308859b05af4d"
TARGET_REPO = "wrldsuksgo2mars/Mistral-Large-4-EXL3-K4-Core-K6"
BASE_PAYLOAD_BYTES = 1_060_548_427_328
BASE_TENSOR_COUNT = 56_563
TARGET_PAYLOAD_BYTES = 540_617_010_400
TARGET_TENSOR_COUNT = 112_003
BASE_CORE_MODULE_COUNT = 424
EXPERT_MODULE_COUNT = 27_264
MODULE_COUNT = BASE_CORE_MODULE_COUNT + EXPERT_MODULE_COUNT
PRESERVED_TENSOR_COUNT = 1_251
PRESERVED_BYTES = 12_415_146_560
CORE_PACKED_BYTES = 12_940_945_568
EXPERT_PACKED_BYTES = 515_260_918_272
MCG_MULTIPLIER = 0xCBAC1FED
SUFFIXES = ("trellis", "suh", "svh", "mcg")
LAYER_COUNT = 72
NOPE_LAYERS = tuple(range(3, 72, 4))
KV_REUSE_LAYERS = tuple(range(39, 72, 4))
FULL_INDEXER_LAYERS = tuple(range(17)) + tuple(x for x in range(17, 72) if x % 4 in (0, 3))
_WIDTHS = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "I16": 2, "F16": 2, "I32": 4}


def _geometry(inputs, outputs, bits, group):
    return dict(in_features=inputs, out_features=outputs,
                stored_out_features=(outputs + 127) // 128 * 128,
                bits=bits, group=group)


def reviewed_core_modules():
    selected = {}
    for layer in range(LAYER_COUNT):
        prefix = f"layers.{layer}."
        for name, inputs, outputs in (("attention.wq", 6144, 8192),
                                      ("attention.wo", 8192, 6144)):
            selected[prefix + name] = _geometry(inputs, outputs, 6, "base_core")
        if layer not in KV_REUSE_LAYERS:
            selected[prefix + "attention.wkv_a_with_mqa"] = _geometry(
                6144, 512 if layer in NOPE_LAYERS else 576, 6, "base_core")
        mlp, intermediate = ("feed_forward", 24576) if layer == 0 else ("shared_experts", 6144)
        for name, inputs, outputs in (("w1", 6144, intermediate),
                                      ("w3", 6144, intermediate),
                                      ("w2", intermediate, 6144)):
            selected[prefix + mlp + "." + name] = _geometry(inputs, outputs, 6, "base_core")
    selected["output"] = _geometry(6144, 204800, 6, "base_core")
    return selected


def reviewed_expert_modules():
    return {f"layers.{layer}.experts.{expert}.{part}": _geometry(6144, 6144, 4, "routed")
            for layer in range(1, LAYER_COUNT) for expert in range(128)
            for part in ("w1", "w3", "w2")}


def reviewed_modules():
    return reviewed_core_modules() | reviewed_expert_modules()


def packed_specs(module, geometry, *, bits=None):
    inputs, outputs = geometry["in_features"], geometry["stored_out_features"]
    bits = geometry["bits"] if bits is None else bits
    if (inputs <= 0 or inputs % 128 or outputs <= 0 or outputs % 128
            or outputs < geometry["out_features"] or bits not in (4, 5, 6, 8)):
        raise ValueError(f"unreviewed EXL3 geometry/bitrate: {module}")
    return {module + ".trellis": ("I16", [inputs // 16, outputs // 16, bits * 16]),
            module + ".suh": ("F16", [inputs]), module + ".svh": ("F16", [outputs]),
            module + ".mcg": ("I32", [])}


def packed_bytes(geometry, *, bits=None):
    return sum(math.prod(shape) * _WIDTHS[dtype]
               for dtype, shape in packed_specs("projection", geometry, bits=bits).values())


def source_names(module, geometry=None):
    """The exact original tensors retired by one reviewed replacement."""
    if geometry is None:
        geometry = reviewed_modules()[module]
    names = [module + ".weight"]
    if not (module == "output" or module.endswith(".attention.wkv_a_with_mqa")):
        names.append(module + ".qscale_weight")
    return tuple(names)


def reviewed_source_specs():
    specs = {}

    def add(name, shape, dtype="BF16"):
        if name in specs:
            raise ValueError(f"duplicate reviewed source tensor: {name}")
        specs[name] = (dtype, list(shape))

    for module, g in reviewed_modules().items():
        names = source_names(module, g)
        add(names[0], [g["out_features"], g["in_features"]], "F8_E4M3" if len(names) == 2 else "BF16")
        if len(names) == 2:
            add(names[1], [math.ceil(g["out_features"] / 128), g["in_features"] // 128], "F32")
    add("tok_embeddings.weight", [204800, 6144])
    add("norm.weight", [6144])
    add("lerope.alpha", [32])
    for layer in range(LAYER_COUNT):
        prefix = f"layers.{layer}."
        for name in ("attention_norm", "ffn_norm"):
            add(prefix + name + ".weight", [6144])
        for name in ("s", "b"):
            add(prefix + "attention.ssmax_scaling." + name, [64])
        add(prefix + "attention.wkv_b.weight", [16384 if layer in NOPE_LAYERS else 12288, 512])
        if layer not in KV_REUSE_LAYERS:
            add(prefix + "attention.kv_a_norm.weight", [512])
        if layer:
            add(prefix + "gate.weight", [128, 6144])
        if layer in FULL_INDEXER_LAYERS:
            index = prefix + "attention.indexer."
            add(index + "wq_b.weight", [4096, 6144], "F8_E4M3")
            add(index + "wq_b.qscale_weight", [32, 48], "F32")
            add(index + "wk.weight", [128, 6144])
            add(index + "weights_proj.weight", [32, 6144])
            for name, width in (("k_norm", 128), ("q_input_norm", 6144)):
                for suffix in ("weight", "bias"):
                    add(index + name + "." + suffix, [width])
            for suffix in ("s", "b"):
                add(index + "ssmax_scaling." + suffix, [32])
    # Bundled dSpark is an immutable BF16 island, including its vocabulary.
    for layer in range(3):
        prefix = f"mtp_dspark.backbone.layers.{layer}."
        for name, shape in (("attention.kv_a_norm.weight", [512]),
                            ("attention.wkv_a_with_mqa.weight", [576, 6144]),
                            ("attention.wkv_b.weight", [12288, 512]),
                            ("attention.wq.weight", [8192, 6144]),
                            ("attention.wo.weight", [6144, 8192]),
                            ("attention_norm.weight", [6144]), ("ffn_norm.weight", [6144]),
                            ("feed_forward.w1.weight", [24576, 6144]),
                            ("feed_forward.w3.weight", [24576, 6144]),
                            ("feed_forward.w2.weight", [6144, 24576])):
            add(prefix + name, shape)
    for name, shape in (("backbone.final_norm.weight", [6144]),
                        ("backbone.target_norm.weight", [6144]),
                        ("backbone.mask_embedding", [2, 6144]),
                        ("backbone.target_projection.weight", [6144, 30720]),
                        ("autoregressive_head.embedding.weight", [204800, 512]),
                        ("autoregressive_head.output.weight", [204800, 512]),
                        ("autoregressive_head.joint_proj.weight", [1536, 7168]),
                        ("autoregressive_head.joint_proj.bias", [1536])):
        add("mtp_dspark." + name, shape)
    # Official Pixtral tower and projector, retained without reinterpretation.
    add("vision_encoder.patch_conv.weight", [2048, 3, 14, 14])
    add("vision_encoder.ln_pre.weight", [2048])
    for layer in range(30):
        prefix = f"vision_encoder.transformer.layers.{layer}."
        for part in ("wq", "wk", "wv", "wo"):
            add(prefix + "attention." + part + ".weight", [2048, 2048])
        for part in ("attention_norm", "ffn_norm"):
            add(prefix + part + ".weight", [2048])
        for part in ("w1", "w3"):
            add(prefix + "feed_forward." + part + ".weight", [6144, 2048])
        add(prefix + "feed_forward.w2.weight", [2048, 6144])
    for name, shape in (("patch_merger.merging_layer.weight", [2048, 8192]),
                        ("pre_mm_projector_norm.weight", [2048]),
                        ("vision_language_adapter.w_in.weight", [6144, 2048]),
                        ("vision_language_adapter.w_out.weight", [6144, 6144])):
        add(name, shape)
    return specs


def validate_config(config):
    if config.get("model_type") != "mistral_large4":
        raise ValueError("source model_type must remain mistral_large4")
    if config.get("transformers_weights") != "consolidated.safetensors.index.json":
        raise ValueError("unreviewed Mistral source index convention")
    quant = config.get("quantization_config", {})
    if (quant.get("quant_method") != "fp8" or quant.get("weight_block_size") != [128, 128]
            or quant.get("activation_scheme") != "dynamic"):
        raise ValueError("source must be the reviewed block-128 vendor FP8 checkpoint")
    expected_excluded = {"model.vision_tower", "model.multi_modal_projector", "lm_head", "kv_b_proj",
                         "indexer.wk", "indexer.weights_proj", "kv_a_proj_with_mqa"}
    if set(quant.get("modules_to_not_convert", [])) != expected_excluded:
        raise ValueError("unreviewed vendor FP8 exceptions")
    text = config.get("text_config", {})
    expected = dict(model_type="mistral_large4_text", hidden_size=6144, num_hidden_layers=72,
        vocab_size=204800, intermediate_size=24576, moe_intermediate_size=6144,
        first_k_dense_replace=1, n_routed_experts=128, n_shared_experts=1, num_experts_per_tok=4,
        num_attention_heads=64, num_key_value_heads=64, head_dim=64, qk_head_dim=128,
        qk_nope_head_dim=64, qk_rope_head_dim=64, v_head_dim=128, kv_lora_rank=512,
        index_n_heads=32, index_head_dim=128, index_topk=2048, no_rope_layer_interval=4,
        final_logit_softcapping=30.0, rms_norm_eps=1e-5, routed_scaling_factor=1.0,
        hidden_act="silu", n_group=1, topk_group=1, tie_word_embeddings=False)
    for key, value in expected.items():
        if text.get(key) != value:
            raise ValueError(f"unreviewed Mistral text config {key}: {text.get(key)!r}")
    schedules = dict(indexer_types=["full" if i in FULL_INDEXER_LAYERS else "shared" for i in range(72)],
                     no_rope_layers=[0 if i in NOPE_LAYERS else 1 for i in range(72)],
                     kv_source_layers=[35 if i in KV_REUSE_LAYERS else i for i in range(72)],
                     mlp_layer_types=["dense"] + ["sparse"] * 71,
                     layer_types=["indexed_attention"] * 72)
    for key, value in schedules.items():
        if text.get(key) != value:
            raise ValueError(f"unreviewed Mistral schedule {key}")
    rope = dict(attention_factor=1.0, beta_fast=32, beta_slow=1, factor=2048,
                original_max_position_embeddings=512, rope_theta=10000.0, rope_type="yarn")
    if text.get("rope_parameters") != rope:
        raise ValueError("unreviewed Mistral RoPE parameters")


def inventory(source):
    """Independently validate every source header and all format byte totals."""
    validate_config(source.config)
    specs, selected = reviewed_source_specs(), reviewed_modules()
    if set(source.weight_map) != set(specs):
        raise ValueError(f"source tensor inventory differs: missing={sorted(set(specs)-set(source.weight_map))[:5]} "
                         f"extra={sorted(set(source.weight_map)-set(specs))[:5]}")
    by_shard = defaultdict(set)
    for name, shard in source.weight_map.items():
        if not isinstance(shard, str) or not shard or Path(shard).name != shard:
            raise ValueError("source shard names must be simple filenames")
        by_shard[shard].add(name)
    total = 0
    for shard, names in by_shard.items():
        header, _ = read_header(source.snapshot / shard)
        if set(header) - {"__metadata__"} != names:
            raise ValueError(f"header/index inventory differs: {shard}")
        for name in names:
            actual, (dtype, shape) = header[name], specs[name]
            size = math.prod(shape) * _WIDTHS[dtype]
            if (actual["dtype"] != dtype or actual["shape"] != shape
                    or actual["data_offsets"][1] - actual["data_offsets"][0] != size):
                raise ValueError(f"unreviewed source dtype/geometry: {name}")
            total += size
    groups = Counter(g["group"] for g in selected.values())
    removed = {name for module, g in selected.items() for name in source_names(module, g)}
    selected_bytes = sum(math.prod(specs[name][1]) * _WIDTHS[specs[name][0]] for name in removed)
    packed = sum(packed_bytes(g) for g in selected.values())
    if (len(specs) != BASE_TENSOR_COUNT or total != BASE_PAYLOAD_BYTES or len(by_shard) != 54
            or groups != dict(base_core=BASE_CORE_MODULE_COUNT, routed=EXPERT_MODULE_COUNT)
            or total - selected_bytes != PRESERVED_BYTES
            or len(specs) - len(removed) != PRESERVED_TENSOR_COUNT
            or total - selected_bytes + packed != TARGET_PAYLOAD_BYTES
            or len(specs) - len(removed) + 4 * len(selected) != TARGET_TENSOR_COUNT):
        raise ValueError("reviewed Mistral source/target totals differ")
    parts = {}
    for group in groups:
        modules = {name: g for name, g in selected.items() if g["group"] == group}
        names = {name for module, g in modules.items() for name in source_names(module, g)}
        parts[group] = dict(modules=len(modules), source_bytes=sum(
            math.prod(specs[name][1]) * _WIDTHS[specs[name][0]] for name in names),
            packed_bytes=sum(packed_bytes(g) for g in modules.values()))
    return dict(source_payload_bytes=total, source_tensors=len(specs), selected_modules=len(selected),
        selected_source_bytes=selected_bytes, selected_packed_bytes=packed, removed_source_tensors=len(removed),
        preserved_bytes=PRESERVED_BYTES, preserved_tensors=PRESERVED_TENSOR_COUNT,
        target_payload_bytes=TARGET_PAYLOAD_BYTES, target_tensors=TARGET_TENSOR_COUNT,
        shards=len(by_shard), parts=parts)


def selected_modules(source):
    inventory(source)
    return reviewed_modules()

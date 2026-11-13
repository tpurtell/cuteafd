"""Reviewed Inkling 2 MXFP8 source and direct K4 / BF16+K8+K6 MC inventory."""
from __future__ import annotations

from collections import Counter
import hashlib
import re

from glm5.core_k6.recipe import packed_bytes, packed_specs
from kimi_k3.uniform_k2.storage import canonical

BASE_REPO = "thinkingmachines/Inkling-2-Small-MXFP8"
BASE_REVISION = "3febc62259ee92ea36e2fb6e8303371c4eb3922c"
TARGET_REPO = "wrldsuksgo2mars/Inkling-2-Small-EXL3-K4-MC"
LAYER_COUNT, EXPERTS_PER_LAYER, MTP_LAYER_COUNT = 48, 384, 5
EXPERT_MODULE_COUNT, BASE_CORE_MODULE_COUNT, MTP_CORE_MODULE_COUNT = 55_296, 289, 45
CORE_MODULE_COUNT = BASE_CORE_MODULE_COUNT + MTP_CORE_MODULE_COUNT
MODULE_COUNT = EXPERT_MODULE_COUNT + CORE_MODULE_COUNT
ENCODE_COUNT = EXPERT_MODULE_COUNT + 2 * CORE_MODULE_COUNT
BASE_TENSOR_COUNT, BASE_PAYLOAD_BYTES = 1694, 622_803_317_088
BASE_MODEL_TENSOR_COUNT, BASE_MODEL_PAYLOAD_BYTES = 1610, 619_379_896_320
MTP_TENSOR_COUNT, MTP_PAYLOAD_BYTES = 84, 3_423_420_768
EXPERT_SOURCE_BYTES, EXPERT_K4_BYTES = 605_414_227_968, 294_298_804_224
NATIVE_SHARED_BYTES = 2_834_002_272
CORE_SOURCE_BYTES, CORE_K8_BYTES, CORE_K6_BYTES = 14_555_086_848, 7_284_076_600, 5_464_617_016
TARGET_PAYLOAD_BYTES = 324_436_586_960
TARGET_TENSOR_COUNT = 225_358
MCG_MULTIPLIER = 0xCBAC1FED
SUFFIXES = ("trellis", "suh", "svh", "mcg")
GLOBAL_LAYERS = tuple(range(5, 48, 6))
LOCAL_LAYERS = tuple(n for n in range(48) if n not in GLOBAL_LAYERS)
SOURCE_CONFIG_SHA256 = "9b74c039e5d0a9aeaad899c147f7aa12609eae0719659fe42646fa7da7d27070"
SOURCE_QUANT_CONFIG_SHA256 = "2fe102d245a7f9d1660e52707d888af82d092052ba298836ac123ec704deb950"
SOURCE_SPECS_SHA256 = "fb902234676fa9d33ec2f339ad34fde680a9034b2b6439ae90f2941005ace577"
SOURCE_INDEX_SHA256 = "51f87de7e81264fb2d8ea0b8f8b0b542da662204f26e2c83d06ae1c04f0addc9"
ACTIVATION_QUANTIZATION = dict(block_size=32, format="mxfp8_e4m3", rounding="nearest", scale_dtype="ue8m0")
_EXPERT = re.compile(r"model\.llm\.layers\.([0-9]+)\.mlp\.experts\.([0-9]+)\.(gate_proj|up_proj|down_proj)")


def _digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _geometry(inputs, outputs, bits, group):
    return dict(in_features=inputs, out_features=outputs,
                stored_out_features=(outputs + 127) // 128 * 128, bits=bits, group=group)


def reviewed_core_modules(*, core_bits=6, include_head=True, include_mtp=True):
    if type(core_bits) is not int or core_bits not in (6, 8):
        raise ValueError("Inkling compact core bits must be K6 or K8")
    selected = {}
    for layer in range(LAYER_COUNT):
        for name, inputs, outputs in (("wq_du", 4608, 6144), ("wk_dv", 4608, 1024),
                ("wv_dv", 4608, 1024), ("wo_ud", 6144, 4608),
                ("wr_du", 4608, 768), ("gate_attn", 4608, 6144)):
            selected[f"model.llm.layers.{layer}.attn.{name}"] = _geometry(inputs, outputs, core_bits, "base_core")
    if include_head:
        selected["model.llm.unembed"] = _geometry(4608, 201024, core_bits, "base_core")
    if include_mtp:
        for layer in range(MTP_LAYER_COUNT):
            for name, inputs, outputs in (("q_proj", 4608, 6144), ("k_proj", 4608, 1024),
                    ("v_proj", 4608, 1024), ("k_proj_ctx", 4608, 1024),
                    ("v_proj_ctx", 4608, 1024), ("o_proj", 6144, 4608)):
                selected[f"mtp.layers.{layer}.self_attn.{name}"] = _geometry(inputs, outputs, core_bits, "mtp_core")
            for name, inputs, outputs in (("gate_proj", 4608, 18432),
                    ("up_proj", 4608, 18432), ("down_proj", 18432, 4608)):
                selected[f"mtp.layers.{layer}.mlp.{name}"] = _geometry(inputs, outputs, core_bits, "mtp_core")
    return selected


def reviewed_expert_modules():
    return {f"model.llm.layers.{layer}.mlp.experts.{expert}.{name}": _geometry(inputs, outputs, 4, "routed")
            for layer in range(LAYER_COUNT) for expert in range(EXPERTS_PER_LAYER)
            for name, inputs, outputs in (("gate_proj", 4608, 2304),
                ("up_proj", 4608, 2304), ("down_proj", 2304, 4608))}


def reviewed_modules(*, core_bits=6):
    return reviewed_expert_modules() | reviewed_core_modules(core_bits=core_bits)


def expert_source(module):
    """Describe interleaved fused source rows; no tensor payload is read."""
    match = _EXPERT.fullmatch(module)
    if not match or int(match[1]) >= LAYER_COUNT or int(match[2]) >= EXPERTS_PER_LAYER:
        raise ValueError("unreviewed Inkling expert module")
    layer, expert, projection = int(match[1]), int(match[2]), match[3]
    fused = "w2_weight" if projection == "down_proj" else "w13_weight"
    prefix = f"model.llm.layers.{layer}.mlp.experts."
    # Official SGLang _swiglu uses even gate rows, odd up rows.
    row_slice = [0, 4608, 1] if projection == "down_proj" else [int(projection == "up_proj"), 4608, 2]
    return dict(weight=prefix + fused, scale=prefix + fused + "_scale", expert=expert,
                row_slice=row_slice, scale_block_size=32, scale_layout="output-row-input-block")


def validate_config(document, quantization_config):
    if _digest(document) != SOURCE_CONFIG_SHA256:
        raise ValueError("unreviewed Inkling source configuration")
    if _digest(quantization_config) != SOURCE_QUANT_CONFIG_SHA256:
        raise ValueError("unreviewed Inkling MXFP8 parameter groups")
    expert_groups = [g for g in quantization_config["model"]["parameter_groups"]
                     if g["compute_dtype"] == "mxfp8"]
    if len(expert_groups) != 1 or expert_groups[0]["required_activation_quantization"] != ACTIVATION_QUANTIZATION:
        raise ValueError("native MXFP8 activation contract differs")


def inventory(source):
    """Check the complete pinned header inventory and calculate all MC views."""
    validate_config(source.document, source.quantization_config)
    if source.snapshot.name != BASE_REVISION or source.config != source.document["text_config"]:
        raise ValueError("source is not the pinned Inkling MXFP8 snapshot")
    specs = source.specs
    if _digest(specs) != SOURCE_SPECS_SHA256 or len(specs) != BASE_TENSOR_COUNT:
        raise ValueError("reviewed Inkling tensor dtype/shape inventory differs")
    if sum(source.sizes.values()) != BASE_PAYLOAD_BYTES:
        raise ValueError("reviewed Inkling source payload differs")
    core = reviewed_core_modules()
    experts = reviewed_expert_modules()
    for module, geometry in core.items():
        if specs.get(module + ".weight") != ("BF16", [geometry["out_features"], geometry["in_features"]]):
            raise ValueError("reviewed core source differs: " + module)
    for layer in range(LAYER_COUNT):
        prefix = f"model.llm.layers.{layer}.mlp.experts."
        for leaf, dtype, shape in (("w13_weight", "F8_E4M3", [384, 4608, 4608]),
                ("w2_weight", "F8_E4M3", [384, 4608, 2304]),
                ("w13_weight_scale", "U8", [384, 4608, 144]),
                ("w2_weight_scale", "U8", [384, 4608, 72])):
            if specs.get(prefix + leaf) != (dtype, shape):
                raise ValueError("native MXFP8 expert block layout differs")
    groups = Counter(g["group"] for g in core.values())
    expert_bytes = sum(packed_bytes(g) for g in experts.values())
    k8_bytes = sum(packed_bytes(g, bits=8) for g in core.values())
    k6_bytes = sum(packed_bytes(g, bits=6) for g in core.values())
    core_bytes = sum(source.sizes[n + ".weight"] for n in core)
    native_names = set(specs) - {n + ".weight" for n in core} - {n for n in specs if ".mlp.experts." in n}
    native_bytes = sum(source.sizes[n] for n in native_names)
    if (len(experts) != EXPERT_MODULE_COUNT or groups != dict(base_core=289, mtp_core=45)
            or (expert_bytes, k8_bytes, k6_bytes, core_bytes, native_bytes) !=
            (EXPERT_K4_BYTES, CORE_K8_BYTES, CORE_K6_BYTES, CORE_SOURCE_BYTES, NATIVE_SHARED_BYTES)):
        raise ValueError("reviewed Inkling MC storage accounting differs")
    shared_tensors = len(experts) * 4 + len(native_names)
    views = {name: dict(payload_bytes=expert_bytes + native_bytes + core_size,
                       tensors=shared_tensors + len(core) * (1 if bits == 16 else 4), core_bits=bits)
             for name, bits, core_size in (("original", 16, core_bytes),
                 ("k8-compact", 8, k8_bytes), ("k6-compact", 6, k6_bytes))}
    return dict(source_tensors=len(specs), source_payload_bytes=BASE_PAYLOAD_BYTES, source_shards=49,
        routed_projections=len(experts), core_projections=len(core), quantized_projections=MODULE_COUNT,
        encode_operations=ENCODE_COUNT, routed_k4_bytes=expert_bytes, core_k8_bytes=k8_bytes,
        core_k6_bytes=k6_bytes, preserved_bytes=native_bytes, preserved_tensors=len(native_names),
        original_core_bytes=core_bytes, target_payload_bytes=expert_bytes + native_bytes + core_bytes + k8_bytes + k6_bytes,
        target_tensors=shared_tensors + len(core) * 9, views=views,
        expert_scale_inclusive_bpw=expert_bytes * 8 / sum(g["in_features"] * g["out_features"] for g in experts.values()),
        provenance=dict(causal_upstream_core_bits=8, alternate_core_bits=6,
            alternate_hessians="captured-under-K8-upstream-activations", drafter_bits=[8, 6], drafter_pilot=False))

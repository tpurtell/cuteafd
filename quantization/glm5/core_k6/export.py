"""Bounded GLM K4/K6 assembly with independent native-byte preservation."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct

from kimi_k3.uniform_k2.storage import (assemble_shard, atomic_json, canonical,
                                       fsync_directory, read_header, verify_receipt)
from .recipe import (BASE_REPO, BASE_REVISION, BASE_PAYLOAD_BYTES, BASE_TENSOR_COUNT,
                     MCG_MULTIPLIER, TARGET_PAYLOAD_BYTES, TARGET_TENSOR_COUNT,
                     packed_specs, packed_bytes, reviewed_modules, selected_modules)


_WIDTHS = dict(I16=2, F16=2, I32=4)
_DTYPES = dict(I16="int16", F16="float16", I32="int32")
_CHUNK = 16 * 1024 * 1024
LAYER_COUNT = 79
SHARD_COUNT = LAYER_COUNT + 1


def _spec(header, name):
    value = header[name]
    return value["dtype"], value["shape"]


def _layer_prefix(layer):
    return f"model.layers.{layer}."


def _preserved_audit(source_entries, target, *, chunk_bytes=_CHUNK):
    """Read both sides independently after assembly, with bounded open handles.

    The digest includes each sorted tensor name, length and original payload;
    every target byte is compared against that original. Header caches and
    descriptors are scoped to a single layer and released before its successor.
    """
    if not 1 <= chunk_bytes <= 64 * 1024 * 1024:
        raise ValueError("invalid preserved audit chunk bound")
    target_header, target_base = read_header(target)
    digest = hashlib.sha256()
    total = 0
    with ExitStack() as stack:
        output = stack.enter_context(Path(target).open("rb"))
        sources = {}
        for name, (path, original_name) in sorted(source_entries.items()):
            path = Path(path)
            if path not in sources:
                header, base = read_header(path)
                sources[path] = (stack.enter_context(path.open("rb")), header, base)
            original, header, base = sources[path]
            if name not in target_header or _spec(header, original_name) != _spec(target_header, name):
                raise ValueError(f"preserved tensor geometry differs: {name}")
            start, end = header[original_name]["data_offsets"]
            target_start, target_end = target_header[name]["data_offsets"]
            if end - start != target_end - target_start:
                raise ValueError(f"preserved tensor length differs: {name}")
            length = end - start
            digest.update(canonical([name, length]).encode() + b"\n")
            original.seek(base + start)
            output.seek(target_base + target_start)
            while length:
                count = min(length, chunk_bytes)
                left, right = original.read(count), output.read(count)
                if len(left) != count or left != right:
                    raise ValueError(f"preserved tensor bytes differ: {name}")
                digest.update(left)
                length -= count
            total += end - start
    return dict(tensors=len(source_entries), payload_bytes=total,
                names_and_payload_sha256=digest.hexdigest(),
                method="sorted JSON [tensor name, byte length] plus newline and payload; independent byte comparison")


def _check_packed(module, geometry, path):
    expected = packed_specs(module, geometry)
    header, base = read_header(path)
    if set(header) - {"__metadata__"} != set(expected):
        raise ValueError(f"replacement tensor inventory differs: {module}")
    for name, spec in expected.items():
        if _spec(header, name) != spec:
            raise ValueError(f"replacement EXL3 geometry differs: {name}")
    # A structurally valid MCG scalar with the wrong multiplier is unusable.
    with Path(path).open("rb") as file:
        file.seek(base + header[module + ".mcg"]["data_offsets"][0])
        if struct.unpack("<I", file.read(4))[0] != MCG_MULTIPLIER:
            raise ValueError(f"replacement EXL3 MCG multiplier differs: {module}")
        # Scale vectors are small; validate them without materializing trellis
        # storage or reconstructing the dense matrix.
        import numpy as np
        for suffix in ("suh", "svh"):
            start, end = header[module + "." + suffix]["data_offsets"]
            file.seek(base + start)
            payload = file.read(end - start)
            if len(payload) != end - start:
                raise ValueError(f"truncated EXL3 scales: {module}.{suffix}")
            scale = np.frombuffer(payload, dtype="<f2")
            if not np.isfinite(scale).all():
                raise ValueError(f"nonfinite EXL3 scales: {module}.{suffix}")
            if suffix == "svh" and np.any(scale[geometry["out_features"]:] != 0):
                raise ValueError(f"nonzero EXL3 padded output scales: {module}")
    return expected


def _assemble(source, source_entries, selection, replacements, shard, receipt_path, identity, **labels):
    if shard.is_symlink() or receipt_path.is_symlink():
        raise ValueError("refuse symlink export shard or receipt")
    if set(replacements) != set(selection):
        raise ValueError("replacement modules do not exactly cover this shard's reviewed selection")
    entries = dict(source_entries)
    expected_specs = {}
    source_payload = 0
    headers = {}
    for name, (path, original_name) in source_entries.items():
        if path not in headers:
            headers[path] = read_header(path)[0]
        value = headers[path][original_name]
        source_payload += value["data_offsets"][1] - value["data_offsets"][0]
        expected_specs[name] = _spec(headers[path], original_name)
    removed_bytes, added_bytes = 0, 0
    for module, geometry in selection.items():
        name = module + ".weight"
        if name not in entries or expected_specs[name] != ("BF16", [geometry["out_features"], geometry["in_features"]]):
            raise ValueError(f"source selected tensor geometry differs: {module}")
        del entries[name]
        del expected_specs[name]
        removed_bytes += geometry["in_features"] * geometry["out_features"] * 2
        for packed_name, (dtype, shape) in packed_specs(module, geometry).items():
            if packed_name in entries:
                raise ValueError(f"replacement collides with original tensor: {packed_name}")
            entries[packed_name] = (Path(replacements[module]), packed_name)
            expected_specs[packed_name] = (dtype, shape)
            count = 1
            for dimension in shape:
                count *= dimension
            added_bytes += count * _WIDTHS[dtype]
    preserved = {name: entry for name, entry in source_entries.items() if name in entries}
    expected_payload = source_payload - removed_bytes + added_bytes
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["identity"] != identity or receipt["filename"] != shard.name or any(receipt.get(k) != v for k, v in labels.items()):
            raise ValueError("export shard run identity differs")
        if receipt.get("payload_bytes") != expected_payload or receipt.get("tensors") != len(entries):
            raise ValueError("export shard receipt inventory differs")
        audit = receipt.get("preserved_audit", {})
        if audit.get("tensors") != len(preserved) or audit.get("payload_bytes") != source_payload - removed_bytes:
            raise ValueError("export shard missing preserved-byte audit")
        verify_receipt(shard, receipt, checksum=True)
        header, _ = read_header(shard)
        if {name: _spec(header, name) for name in header if name != "__metadata__"} != expected_specs:
            raise ValueError("committed export geometry differs")
        return receipt
    for module, geometry in selection.items():
        _check_packed(module, geometry, replacements[module])
    if shard.exists():
        shard.unlink()  # A shard without its durable receipt is uncommitted.
    receipt = assemble_shard(entries, shard)
    if receipt["payload_bytes"] != expected_payload:
        raise ValueError("assembled payload size differs")
    audit = _preserved_audit(preserved, shard)
    receipt.update(identity=identity, filename=shard.name, tensors=len(entries),
                   preserved_audit=audit, **labels)
    atomic_json(receipt_path, receipt)
    return receipt


def _shard_name(index):
    return f"model-{index:05d}-of-{SHARD_COUNT:05d}.safetensors"


def _target_shard(name):
    match = re.match(r"^model\.layers\.([0-9]+)\.", name)
    if match:
        layer = int(match[1])
        if not 0 <= layer < LAYER_COUNT:
            raise ValueError(f"tensor outside reviewed layer namespace: {name}")
        return _shard_name(layer + 1)
    return _shard_name(SHARD_COUNT)


def assemble_layer(source, layer, replacements, export, identity):
    """Replace the layer's core/expert selection and audit every native byte."""
    if type(layer) is not int or not 0 <= layer < LAYER_COUNT:
        raise ValueError("invalid GLM layer")
    export = Path(export)
    export.mkdir(parents=True, exist_ok=True)
    prefix = _layer_prefix(layer)
    selection = {name: g for name, g in reviewed_modules().items() if name.startswith(prefix)}
    originals = {name: (source.snapshot / shard, name) for name, shard in source.weight_map.items()
                 if name.startswith(prefix)}
    return _assemble(source, originals, selection, replacements,
        export / _shard_name(layer + 1), export / f"layer-{layer:03d}-receipt.json", identity, layer=layer)


def _copy_assets(source, export):
    excluded = {"config.json", "quantize_config.json", "model.safetensors.index.json",
                "README.md", ".gitattributes", "quantization-provenance.json"}
    for path in source.snapshot.iterdir():
        if (not path.is_file() or path.name in excluded
                or path.name.endswith((".safetensors", ".partial", "-receipt.json"))):
            continue
        shutil.copyfile(path, export / path.name)


def _fsync_tree(root):
    """Flush files and directories before the producer retires recovery state."""
    root = Path(root)
    if root.is_symlink():
        raise ValueError("refuse symlink export directory during durability pass")
    with os.scandir(root) as entries:
        for entry in entries:
            if entry.is_symlink():
                raise ValueError(f"refuse symlink in export durability pass: {entry.path}")
            if entry.is_dir(follow_symlinks=False):
                _fsync_tree(entry.path)
            elif entry.is_file(follow_symlinks=False):
                descriptor = os.open(entry.path, os.O_RDONLY | os.O_NOFOLLOW)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            else:
                raise ValueError(f"unexpected special file in export: {entry.path}")
    fsync_directory(root)


def _source_native_specs(source, selection):
    """Read headers sequentially, retaining only the 662 native specifications."""
    by_shard = {}
    for name, shard in source.weight_map.items():
        by_shard.setdefault(shard, set()).add(name)
    native, total, count = {}, 0, 0
    selected_names = {name + ".weight" for name in selection}
    for shard, names in by_shard.items():
        header, _ = read_header(source.snapshot / shard)
        if set(header) - {"__metadata__"} != names:
            raise ValueError(f"source header/index inventory differs: {shard}")
        for name in names:
            value = header[name]
            total += value["data_offsets"][1] - value["data_offsets"][0]
            count += 1
            if name not in selected_names:
                native[name] = _spec(header, name)
    if count != BASE_TENSOR_COUNT or total != BASE_PAYLOAD_BYTES:
        raise ValueError("source BF16 checkpoint inventory differs")
    return native


def _native_bytes(spec):
    dtype, shape = spec
    count = 1
    for dimension in shape:
        count *= dimension
    return count * {"BF16": 2, "F32": 4}[dtype]


def _module_patterns():
    # Per-module metadata carries exact bits; compact positive patterns avoid
    # tens of thousands of expert-specific regular expressions.
    layers = r"(?:[0-9]|[1-6][0-9]|7[0-8])"
    routed = r"(?:[3-9]|[1-6][0-9]|7[0-8])"
    return [r"^lm_head$",
        rf"^model\.layers\.{layers}\.self_attn\.(?:q_a_proj|q_b_proj|kv_a_proj_with_mqa|o_proj)$",
        rf"^model\.layers\.{routed}\.mlp\.(?:shared_experts|experts\.[0-9]+)\.(?:gate_proj|up_proj|down_proj)$",
        r"^model\.layers\.[0-2]\.mlp\.(?:gate_proj|up_proj|down_proj)$",
        r"^model\.layers\.78\.eh_proj$"]


def finalize_export(source, selection, head_replacements, export, identity, evidence_root):
    """Audit the standard-HF inventory and commit mixed K4/K6 metadata."""
    export = Path(export)
    if selection != selected_modules(source):
        raise ValueError("final quantization selection differs from reviewed source")
    extras = {name: (source.snapshot / shard, name) for name, shard in source.weight_map.items()
              if not name.startswith("model.layers.")}
    head_selection = {name: g for name, g in selection.items() if name == "lm_head"}
    _assemble(source, extras, head_selection, head_replacements, export / _shard_name(SHARD_COUNT),
              export / "extras-receipt.json", identity)
    expected_shards = {_shard_name(i) for i in range(1, SHARD_COUNT + 1)}
    if {path.name for path in export.glob("*.safetensors")} != expected_shards:
        raise ValueError("final export shard inventory differs")
    native_specs = _source_native_specs(source, selection)
    expected_map = {name: _target_shard(name) for name in native_specs}
    for module, geometry in selection.items():
        expected_map.update({name: _target_shard(name) for name in packed_specs(module, geometry)})
    modules_by_shard, native_by_shard = {}, {}
    for module, geometry in selection.items():
        modules_by_shard.setdefault(_target_shard(module), {})[module] = geometry
    for name, spec in native_specs.items():
        native_by_shard.setdefault(_target_shard(name), {})[name] = spec
    weight_map, total, preserved_total, preserved_count = {}, 0, 0, 0
    for shard in sorted(expected_shards):
        path = export / shard
        if path.is_symlink():
            raise ValueError("refuse symlink final shard")
        header, _ = read_header(path)
        ordinal = int(shard.split("-")[1])
        receipt_path = export / ("extras-receipt.json" if ordinal == SHARD_COUNT else
                                  f"layer-{ordinal - 1:03d}-receipt.json")
        if receipt_path.is_symlink():
            raise ValueError("refuse symlink final receipt")
        receipt = json.loads(receipt_path.read_text())
        if (receipt["identity"] != identity or receipt.get("filename") != shard
                or (ordinal != SHARD_COUNT and receipt.get("layer") != ordinal - 1)):
            raise ValueError("final export shard identity differs")
        # Rehash the owned target, not the BF16 source payload. This verifies
        # the exact file that passed assembly's independent native-byte audit.
        verify_receipt(path, receipt, checksum=True)
        native = native_by_shard.get(shard, {})
        expected_audit_bytes = sum(_native_bytes(spec) for spec in native.values())
        audit = receipt.get("preserved_audit", {})
        if (audit.get("tensors") != len(native) or audit.get("payload_bytes") != expected_audit_bytes
                or not re.fullmatch(r"[0-9a-f]{64}", audit.get("names_and_payload_sha256", ""))):
            raise ValueError("final shard missing preserved-byte audit")
        preserved_total += expected_audit_bytes
        preserved_count += len(native)
        replacements = {name: spec for module, g in modules_by_shard.get(shard, {}).items()
                        for name, spec in packed_specs(module, g).items()}
        expected_specs = native | replacements
        if {name: _spec(header, name) for name in header if name != "__metadata__"} != expected_specs:
            raise ValueError(f"final exported tensor inventory or geometry differs: {shard}")
        shard_total = 0
        for name, value in header.items():
            if name == "__metadata__":
                continue
            if name in weight_map or expected_map.get(name) != shard:
                raise ValueError(f"unexpected or duplicate exported tensor: {name}")
            weight_map[name] = shard
            shard_total += value["data_offsets"][1] - value["data_offsets"][0]
        if receipt.get("payload_bytes") != shard_total or receipt.get("tensors") != len(expected_specs):
            raise ValueError("final shard receipt inventory differs")
        total += shard_total
    if weight_map != expected_map or len(weight_map) != TARGET_TENSOR_COUNT or total != TARGET_PAYLOAD_BYTES:
        raise ValueError("final export tensor/payload inventory differs")
    tensor_storage = {}
    for module, geometry in selection.items():
        tensor_storage[module] = dict(quant_format="exl3", bits_per_weight=geometry["bits"],
            mcg_multiplier=MCG_MULTIPLIER, in_features=geometry["in_features"],
            out_features=geometry["out_features"], stored_out_features=geometry["stored_out_features"],
            stored_tensors={name: dict(shape=shape, torch_dtype=_DTYPES[dtype])
                            for name, (dtype, shape) in packed_specs(module, geometry).items()})
    declaration = dict(quant_method="exl3", format="exl3", checkpoint_format="exl3",
        bits=4., core_bits=6., mtp_bits=4., head_bits=6.,
        per_module_metadata="quantize_config.json:tensor_storage")
    config = json.loads(canonical(source.config))
    config["quantization_config"] = declaration
    quant_config = dict(declaration, method="exl3", group_size=-1, desc_act=False, sym=True,
        lm_head=True, out_scales="auto", codebook="mcg", tensor_storage=tensor_storage,
        module_include=_module_patterns(), calibration=dict(samples=1426, tokens=1097737),
        meta=dict(quantization_provenance=identity))
    atomic_json(export / "model.safetensors.index.json", dict(metadata=dict(total_size=total), weight_map=weight_map))
    atomic_json(export / "config.json", config)
    atomic_json(export / "quantize_config.json", quant_config)
    _copy_assets(source, export)
    evidence_root = Path(evidence_root)
    if not evidence_root.is_dir() or evidence_root.is_symlink():
        raise ValueError("numerical evidence root must be a regular directory")
    for path in evidence_root.rglob("*.json"):
        if path.is_symlink():
            raise ValueError("refuse symlink numerical evidence")
        target = export / "quantization-errors" / path.relative_to(evidence_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    atomic_json(export / "quantization-provenance.json", identity)
    (export / "README.md").write_text(_model_card(identity))
    # The caller can publish completion and retire final recovery state only
    # after this recursive durability barrier returns successfully.
    _fsync_tree(export)
    groups = {group: sum(g.get("group") == group for g in selection.values())
              for group in ("base_core", "mtp_core", "routed")}
    return dict(tensors=len(weight_map), payload_bytes=total, shards=SHARD_COUNT,
        quantized_projections=len(selection), core_projections=groups["base_core"] + groups["mtp_core"],
        base_core_projections=groups["base_core"], mtp_core_projections=groups["mtp_core"],
        routed_projections=groups["routed"], preserved_tensors=preserved_count, preserved_bytes=preserved_total,
        replaced_bf16_bytes=sum(g["in_features"] * g["out_features"] * 2 for g in selection.values()),
        core_k6_bytes=sum(packed_bytes(g) for g in selection.values() if g["bits"] == 6),
        mtp_core_k4_bytes=sum(packed_bytes(g) for g in selection.values() if g.get("group") == "mtp_core"),
        routed_k4_bytes=sum(packed_bytes(g) for g in selection.values() if g.get("group") == "routed"))


def _model_card(identity):
    return f'''---
license: other
license_name: glm-5.3
license_link: https://huggingface.co/{BASE_REPO}/blob/{BASE_REVISION}/LICENSE
base_model: {BASE_REPO}
datasets:
- wrldsuksgo2mars/agentic-quant-panel-v1
tags:
- exl3
- quantized
- glm_moe_dsa
---
# GLM 5.3 EXL3 K4 + Core K6

Quantized from the full **BF16** release [{BASE_REPO}](https://huggingface.co/{BASE_REPO}),
pinned at `{BASE_REVISION}`. The source is not the block-FP8 checkpoint used
by the earlier K4-v1 quant. Native architecture, tokenizer, generation assets
and license are preserved.

| Part | Matrices | Format |
| --- | ---: | --- |
| Base attention q_a/q_b/kv_a/o, shared/dense FFNs and head | 547 | EXL3 MCG K6 |
| Routed expert gate/up/down, including native MTP | 58,368 | EXL3 MCG K4 |
| MTP attention/shared FFN/eh_proj | 8 | EXL3 MCG K4 |
| kv_b, DSA indexer, embeddings, routers and norms | 662 retained tensors | native BF16/F32 |

MCG multiplier is `0xCBAC1FED`, seed 787, sigma 0.025 and automatic output
scales; no E4M3 codebook rounding is applied. MLA kv_a logical output remains
576, stored at 640 for complete 128-channel transforms before logical cropping.
Tensor-specific `quantize_config.json:tensor_storage` and safetensors headers
describe every module's own bit width and layout. Tensor payload totals
385,362,552,236 bytes (358.90 GiB), 80 shards and 236,354 tensors; headers and
metadata add a small overhead.

Calibration uses all 1,426 prompts (1,097,737 native GLM tokens) from
`wrldsuksgo2mars/agentic-quant-panel-v1` at
`875ff1d3786bf4346ee7ff52ea5e3f1f698f5902`. Native GLM rendering enables thinking
and maximum reasoning effort. Prompts retain original order, batch one,
without padding, packing or truncation. Provenance records panel, rendered-token
and producer hashes. Dense Hessians contain actual observations; expert
coverage/recovery policy and evidence are recorded separately.

One interleaved causal pass propagates serialized/reconstructed core and
expert predecessors before each successor's calibration. Gate/up precedes
nonlinear/down-input capture, and the rolling frontier retains exact shared
DSA top-k state. Native MTP shares the vocabulary assets; its shifted
token/target-state alignment and native tap contract are recorded in provenance
and numerical evidence.

Quantization ran on Oliver's single RTX PRO 6000 Blackwell Max-Q. Every retained
tensor was independently compared byte-for-byte with the BF16 source; packed
layouts and the complete inventory were checked. Evidence is in
`quantization-errors/`, and the recipe/input hashes are in
`quantization-provenance.json`. Full-model generation, long-context quality and
serving performance qualification have not been performed by this job. A
runtime must support this mixed standard MCG K4/K6 core and retained BF16
components. No EXLR8 sidecar is required.
'''

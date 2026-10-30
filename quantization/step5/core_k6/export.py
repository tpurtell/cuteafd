"""Bounded Step-5 K4/K6 assembly with independent native-byte preservation."""
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
                     packed_specs, packed_bytes, reviewed_modules, selected_modules,
                     source_tensor, selected_source_names, validate_header_inventory)


_WIDTHS = dict(I16=2, F16=2, I32=4)
_DTYPES = dict(I16="int16", F16="float16", I32="int32")
_CHUNK = 16 * 1024 * 1024
LAYER_COUNT = 95
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
    removal, fused = {}, {}
    for module, geometry in selection.items():
        name, expert = source_tensor(module)
        matrix = [geometry["out_features"], geometry["in_features"]]
        shape = matrix if expert is None else [352, *matrix]
        if name not in entries or expected_specs[name] != ("BF16", shape):
            raise ValueError(f"source selected tensor geometry differs: {module}")
        removal[name] = geometry["in_features"] * geometry["out_features"] * 2 * (1 if expert is None else 352)
        if expert is not None:
            fused.setdefault(name, set()).add(expert)
    if any(experts != set(range(352)) for experts in fused.values()):
        raise ValueError("fused source replacement does not cover every expert exactly")
    for name, size in removal.items():
        del entries[name]
        del expected_specs[name]
        removed_bytes += size
    for module, geometry in selection.items():
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
        raise ValueError("invalid Step-5 layer")
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
    """Read headers sequentially, retaining only the 1,513 native specifications."""
    by_shard = {}
    for name, shard in source.weight_map.items():
        by_shard.setdefault(shard, set()).add(name)
    native, total, count = {}, 0, 0
    selected_names = selected_source_names(selection)
    for shard, names in by_shard.items():
        header, _ = read_header(source.snapshot / shard)
        validate_header_inventory(shard, header, names)
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
    # Exact positive ranges, with individual bit widths in tensor_storage.
    layers = r"(?:[0-9]|[1-8][0-9]|9[0-4])"
    routed = r"(?:[3-9]|[1-8][0-9]|90)"
    experts = r"(?:[0-9]|[1-9][0-9]|[12][0-9]{2}|3[0-4][0-9]|35[01])"
    prefix = r"model\.layers"
    return [r"^lm_head$",
        rf"^{prefix}\.{layers}\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)$",
        rf"^{prefix}\.{routed}\.share_expert\.(?:gate_proj|up_proj|down_proj)$",
        rf"^{prefix}\.{routed}\.moe\.experts\.{experts}\.(?:gate_proj|up_proj|down_proj)$",
        rf"^{prefix}\.(?:[0-2]|9[1-4])\.mlp\.(?:gate_proj|up_proj|down_proj)$",
        rf"^{prefix}\.9[2-4]\.eh_proj$",
        rf"^{prefix}\.9[2-4]\.transformer\.shared_head\.output$"]


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
        codebook="mcg", mcg_multiplier=MCG_MULTIPLIER, out_scales="auto",
        expert_bits=4., expert_bits_by_layer={str(layer): 6. for layer in (88,89,90)},
        per_module_metadata="quantize_config.json:tensor_storage")
    config = json.loads(canonical(source.document))
    config["quantization_config"] = declaration
    quant_config = dict(declaration, method="exl3", group_size=-1, desc_act=False, sym=True,
        lm_head=True, out_scales="auto", codebook="mcg", tensor_storage=tensor_storage,
        module_include=_module_patterns(), calibration=identity.get("calibration", {}),
        kept_tensors={name: dict(torch_dtype={"BF16":"bfloat16","F32":"float32"}[dtype],shape=shape)
                      for name,(dtype,shape) in native_specs.items()},
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
        core_k6_bytes=sum(packed_bytes(g) for g in selection.values() if g.get("group") == "base_core" and g["bits"] == 6),
        mtp_core_k4_bytes=sum(packed_bytes(g) for g in selection.values() if g.get("group") == "mtp_core"),
        routed_k4_bytes=sum(packed_bytes(g) for g in selection.values() if g.get("group") == "routed" and g["bits"] == 4),
        routed_k6_bytes=sum(packed_bytes(g) for g in selection.values() if g.get("group") == "routed" and g["bits"] == 6))


def _model_card(identity):
    return f'''---
license: other
license_name: stepfun-model-license-1.0
license_link: https://huggingface.co/{BASE_REPO}/blob/{BASE_REVISION}/LICENSE.txt
base_model: {BASE_REPO}
datasets:
- wrldsuksgo2mars/agentic-quant-panel-v1
tags:
- exl3
- quantized
- step5_vl
- image-text-to-text
---
# Step-5 Preview EXL3 K4 + Core K6

Quantized from the original **BF16** release [{BASE_REPO}](https://huggingface.co/{BASE_REPO}),
pinned at `{BASE_REVISION}`. The checkpoint retains its native sliding-window
and sparse full-attention schedule, FP32 residual/norm numerics, headwise
attention gates, three MTP stages, bundled vision tower, tokenizer, image
processing assets and the source license (copied verbatim).

| Part | Matrices | Format |
| --- | ---: | --- |
| Target attention q/k/v/o, shared/dense FFNs and lm_head | 645 | EXL3 MCG K6 |
| Routed expert gate/up/down, layers 3–87 | 89,760 | EXL3 MCG K4 |
| Routed expert gate/up/down, layers 88–90 | 3,168 | EXL3 MCG K6 |
| MTP attention, dense FFNs, eh_proj and three stage-local output heads | 27 | EXL3 MCG K4 |
| Vision, attention gates, sparse indexers, embeddings, routers and norms | 1,513 retained tensors | native BF16/F32 |

MCG multiplier is `0xCBAC1FED`, seed 787, sigma 0.025 and automatic output
scales. Each of the 352 fused source experts is exported as a separately
named standard EXL3 projection under `model.layers.N.moe.experts.E` in original
expert-index order. Each projection contains trellis int16, suh/svh float16
and one int32 MCG multiplier. Per-module metadata in
`quantize_config.json:tensor_storage` and safetensors headers describe each
projection's bit width and layout. Target payload is 315,357,147,520 bytes
(293.699 GiB), 96 shards and 375,913 tensors. Headers and metadata add overhead.
Vision is preserved byte-for-byte and no `vision_bits` is declared.

Calibration uses the pinned public `wrldsuksgo2mars/agentic-quant-panel-v1`
with the checkpoint's native chat/thinking rendering. Exact sample/token
counts and rendered-token/input/producer hashes are recorded in provenance.
Prompts retain original order, batch one, without padding, packing or
truncation. Dense Hessians contain actual observations; expert coverage and
recovery evidence are recorded separately.

One interleaved causal pass propagates serialized/reconstructed core and
expert predecessors before calibrating each successor layer. Gate/up precedes
the native clamped SwiGLU down-input capture. Sliding-window attention uses
the native 512-token causal window, with the native sparse indexer on the 23
full-attention layers. All three MTP stages use their native shifted-token
alignment and retain their own calibrated K4 output head.

Quantization ran on maxq's single RTX PRO 6000 Blackwell Max-Q. Every retained
tensor was independently compared byte-for-byte with the BF16 source; packed
layouts and the complete inventory were checked. Evidence is in
`quantization-errors/`, with producer/input hashes in
`quantization-provenance.json`. Full-model generation, long-context quality
and serving performance qualification are separate from this job. A runtime
must support Step-5's mixed standard MCG K4/K6 expert and core projections,
three MTP stages and retained native components. No EXLR8 sidecar is required.
'''

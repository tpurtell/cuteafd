"""Bounded native Mistral K4/K6 assembly and independent byte preservation."""
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
                     packed_specs, packed_bytes, reviewed_modules, selected_modules, source_names,
                     reviewed_source_specs, BASE_CORE_MODULE_COUNT, EXPERT_MODULE_COUNT,
                     PRESERVED_BYTES, PRESERVED_TENSOR_COUNT, CORE_PACKED_BYTES, EXPERT_PACKED_BYTES)


_WIDTHS = dict(I16=2, F16=2, I32=4)
_DTYPES = dict(I16="int16", F16="float16", I32="int32")
_CHUNK = 16 * 1024 * 1024
LAYER_COUNT = 72
SHARD_COUNT = LAYER_COUNT + 1


def _spec(header, name):
    value = header[name]
    return value["dtype"], value["shape"]


def _layer_prefix(layer):
    return f"layers.{layer}."


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
    reviewed = reviewed_source_specs()
    for module, geometry in selection.items():
        for name in source_names(module, geometry):
            if name not in entries or expected_specs[name] != reviewed[name]:
                raise ValueError(f"source selected tensor geometry differs: {name}")
            dtype, shape = expected_specs.pop(name)
            removed_bytes += _native_bytes((dtype, shape))
            del entries[name]
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
    match = re.match(r"^layers\.([0-9]+)\.", name)
    if match:
        layer = int(match[1])
        if not 0 <= layer < LAYER_COUNT:
            raise ValueError(f"tensor outside reviewed layer namespace: {name}")
        return _shard_name(layer + 1)
    return _shard_name(SHARD_COUNT)


def assemble_layer(source, layer, replacements, export, identity):
    """Replace the layer's core/expert selection and audit every native byte."""
    if type(layer) is not int or not 0 <= layer < LAYER_COUNT:
        raise ValueError("invalid Mistral layer")
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
                "consolidated.safetensors.index.json", "README.md", ".gitattributes", "quantization-provenance.json"}
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
    selected_names = {name for module, g in selection.items() for name in source_names(module, g)}
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
        raise ValueError("source FP8 checkpoint inventory differs")
    return native


def _native_bytes(spec):
    dtype, shape = spec
    count = 1
    for dimension in shape:
        count *= dimension
    return count * {"BF16": 2, "F32": 4, "F8_E4M3": 1}[dtype]


def _module_patterns():
    return [r"^output$",
        r"^layers\.(?:[0-9]|[1-6][0-9]|7[01])\.attention\.(?:wq|wo|wkv_a_with_mqa)$",
        r"^layers\.(?:[1-9]|[1-6][0-9]|7[01])\.(?:shared_experts|experts\.[0-9]+)\.w[123]$",
        r"^layers\.0\.feed_forward\.w[123]$"]


def audit_completed_export(source, export, identity):
    """Reread retained source bytes and independently validate the final format.

    This audit does not trust the assembler's preservation digest. It compares
    every retained tensor again and checks each packed MCG scalar and scale in
    its final shard, keeping at most one shard header and small vectors live.
    """
    export = Path(export)
    selection = selected_modules(source)
    source_specs = reviewed_source_specs()
    removed = {name for module, g in selection.items() for name in source_names(module, g)}
    native = {name: spec for name, spec in source_specs.items() if name not in removed}
    expected_specs = native | {name: spec for module, g in selection.items()
                              for name, spec in packed_specs(module, g).items()}
    expected_map = {name: _target_shard(name) for name in expected_specs}
    index = json.loads((export / "model.safetensors.index.json").read_text())
    if (index != json.loads((export / "consolidated.safetensors.index.json").read_text())
            or index.get("weight_map") != expected_map
            or index.get("metadata", {}).get("total_size") != TARGET_PAYLOAD_BYTES):
        raise ValueError("completed export index differs from the independent inventory")
    if json.loads((export / "quantization-provenance.json").read_text()) != identity:
        raise ValueError("completed export producer provenance differs")
    config = json.loads((export / "config.json").read_text())
    if ({k: v for k, v in config.items() if k != "quantization_config"}
            != {k: v for k, v in source.config.items() if k != "quantization_config"}):
        raise ValueError("completed export native configuration differs")
    qconfig = json.loads((export / "quantize_config.json").read_text())
    if (qconfig.get("core_bits") != 6 or qconfig.get("head_bits") != 6
            or qconfig.get("out_scales") != "auto" or qconfig.get("codebook") != "mcg"
            or qconfig.get("meta", {}).get("quantization_provenance") != identity
            or qconfig.get("native_quantization_config") != source.config["quantization_config"]
            or set(qconfig.get("tensor_storage", {})) != set(selection)):
        raise ValueError("completed export quantization metadata differs")
    if config.get("quantization_config") != {k: qconfig[k] for k in
            ("quant_method", "format", "checkpoint_format", "bits", "core_bits", "head_bits",
             "native_quantization_config", "per_module_metadata")}:
        raise ValueError("completed export quantization declarations disagree")
    for module, g in selection.items():
        expected = dict(quant_format="exl3", bits_per_weight=g["bits"], mcg_multiplier=MCG_MULTIPLIER,
            in_features=g["in_features"], out_features=g["out_features"],
            stored_out_features=g["stored_out_features"], stored_tensors={name:
                dict(shape=shape, torch_dtype=_DTYPES[dtype]) for name, (dtype, shape)
                in packed_specs(module, g).items()})
        if qconfig["tensor_storage"][module] != expected:
            raise ValueError(f"completed export tensor_storage differs: {module}")
    shard_names = {_shard_name(i) for i in range(1, SHARD_COUNT + 1)}
    if {path.name for path in export.glob("*.safetensors")} != shard_names:
        raise ValueError("completed export shard inventory differs")
    total, count, preserved_count, preserved_bytes = 0, 0, 0, 0
    preserved_audits = {}
    import numpy as np
    for ordinal in range(1, SHARD_COUNT + 1):
        shard = _shard_name(ordinal)
        path = export / shard
        receipt_path = export / ("extras-receipt.json" if ordinal == SHARD_COUNT else
                                f"layer-{ordinal - 1:03d}-receipt.json")
        if path.is_symlink() or receipt_path.is_symlink():
            raise ValueError("completed export contains symlink shard/receipt")
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("identity") != identity or receipt.get("filename") != shard
                or (ordinal != SHARD_COUNT and receipt.get("layer") != ordinal - 1)):
            raise ValueError("completed export shard identity differs")
        verify_receipt(path, receipt, checksum=True)
        header, base = read_header(path)
        specs = {name: spec for name, spec in expected_specs.items() if expected_map[name] == shard}
        if {name: _spec(header, name) for name in header if name != "__metadata__"} != specs:
            raise ValueError(f"completed export header differs: {shard}")
        payload = sum(h["data_offsets"][1] - h["data_offsets"][0]
                      for name, h in header.items() if name != "__metadata__")
        if receipt.get("tensors") != len(specs) or receipt.get("payload_bytes") != payload:
            raise ValueError("completed export receipt totals differ")
        total += payload
        count += len(specs)
        originals = {name: (source.snapshot / source.weight_map[name], name)
                     for name in native if _target_shard(name) == shard}
        audit = _preserved_audit(originals, path)
        if receipt.get("preserved_audit") != audit:
            raise ValueError("independent native audit differs from assembly")
        preserved_audits[shard] = audit
        preserved_count += audit["tensors"]
        preserved_bytes += audit["payload_bytes"]
        with path.open("rb") as file:
            for module, g in selection.items():
                if _target_shard(module) != shard:
                    continue
                file.seek(base + header[module + ".mcg"]["data_offsets"][0])
                if struct.unpack("<I", file.read(4))[0] != MCG_MULTIPLIER:
                    raise ValueError(f"completed EXL3 MCG multiplier differs: {module}")
                for suffix in ("suh", "svh"):
                    start, end = header[module + "." + suffix]["data_offsets"]
                    file.seek(base + start)
                    scale = np.frombuffer(file.read(end - start), dtype="<f2")
                    if not np.isfinite(scale).all():
                        raise ValueError(f"completed EXL3 scale is nonfinite: {module}.{suffix}")
                    if suffix == "svh" and np.any(scale[g["out_features"]:] != 0):
                        raise ValueError(f"completed padded EXL3 scales differ: {module}")
    if (total != TARGET_PAYLOAD_BYTES or count != TARGET_TENSOR_COUNT
            or preserved_count != PRESERVED_TENSOR_COUNT or preserved_bytes != PRESERVED_BYTES):
        raise ValueError("completed export final inventory differs")
    # Assets are small and read independently of the original copy step.
    excluded = {"config.json", "quantize_config.json", "model.safetensors.index.json",
                "consolidated.safetensors.index.json", "README.md", ".gitattributes",
                "quantization-provenance.json"}
    assets = []
    for path in source.snapshot.iterdir():
        if (not path.is_file() or path.name in excluded
                or path.name.endswith((".safetensors", ".partial", "-receipt.json"))):
            continue
        target = export / path.name
        if target.is_symlink() or target.read_bytes() != path.read_bytes():
            raise ValueError(f"completed export asset bytes differ: {path.name}")
        assets.append(path.name)
    summary = dict(shards=SHARD_COUNT, tensors=count, payload_bytes=total,
                   quantized_projections=len(selection), preserved_tensors=preserved_count,
                   preserved_bytes=preserved_bytes)
    return dict(schema="mistral-large4-completed-export-audit-v1", identity=identity,
                summary=summary, native_audits=preserved_audits, assets=sorted(assets), findings=[])


def finalize_export(source, selection, head_replacements, export, identity, evidence_root):
    """Audit the standard-HF inventory and commit mixed K4/K6 metadata."""
    export = Path(export)
    if selection != selected_modules(source):
        raise ValueError("final quantization selection differs from reviewed source")
    extras = {name: (source.snapshot / shard, name) for name, shard in source.weight_map.items()
              if not name.startswith("layers.")}
    head_selection = {name: g for name, g in selection.items() if name == "output"}
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
        # Rehash the owned target. This verifies
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
        bits=4., core_bits=6., head_bits=6.,
        native_quantization_config=source.config["quantization_config"],
        per_module_metadata="quantize_config.json:tensor_storage")
    config = json.loads(canonical(source.config))
    config["quantization_config"] = declaration
    quant_config = dict(declaration, method="exl3", group_size=-1, desc_act=False, sym=True,
        lm_head=True, out_scales="auto", codebook="mcg", tensor_storage=tensor_storage,
        module_include=_module_patterns(), calibration=identity.get("calibration", {}),
        meta=dict(quantization_provenance=identity))
    index = dict(metadata=dict(total_size=total), weight_map=weight_map)
    atomic_json(export / "model.safetensors.index.json", index)
    atomic_json(export / "consolidated.safetensors.index.json", index)
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
              for group in ("base_core", "routed")}
    return dict(tensors=len(weight_map), payload_bytes=total, shards=SHARD_COUNT,
        quantized_projections=len(selection), core_projections=groups["base_core"],
        base_core_projections=groups["base_core"], routed_projections=groups["routed"],
        preserved_tensors=preserved_count, preserved_bytes=preserved_total,
        core_k6_bytes=CORE_PACKED_BYTES, routed_k4_bytes=EXPERT_PACKED_BYTES)


def _model_card(identity):
    calibration = identity.get("calibration", {})
    samples = calibration.get("samples", "recorded in provenance")
    tokens = calibration.get("tokens", "recorded in provenance")
    return f'''---
license: other
license_name: mistral-ecosystem-open-weight
license_link: https://huggingface.co/{BASE_REPO}/blob/{BASE_REVISION}/LICENSE
base_model: {BASE_REPO}
datasets:
- wrldsuksgo2mars/agentic-quant-panel-v1
tags:
- exl3
- quantized
- mistral_large4
---
# Mistral Large 4 EXL3 K4 + Core K6

Quantized from the official **block-128 FP8** checkpoint
[{BASE_REPO}](https://huggingface.co/{BASE_REPO}), pinned at
`{BASE_REVISION}`. The dequantized vendor FP8 weights are the quantization
reference; this release does not claim BF16-source fidelity. Native consolidated
tensor names, architecture, tokenizer, vision tower and bundled dSpark drafter
are preserved. The untouched source license is included.

| Part | Matrices / tensors | Format | Payload GiB |
| --- | ---: | --- | ---: |
| Attention wq/kv_a/wo, shared/dense FFNs and output head | {BASE_CORE_MODULE_COUNT:,} matrices | EXL3 MCG K6 | {CORE_PACKED_BYTES / 1024**3:.3f} |
| Routed experts w1/w3/w2 | {EXPERT_MODULE_COUNT:,} matrices | EXL3 MCG K4 | {EXPERT_PACKED_BYTES / 1024**3:.3f} |
| kv_b, indexer, routers, norms, embeddings, vision and dSpark | {PRESERVED_TENSOR_COUNT:,} tensors | Original BF16/F32/FP8 bytes | {PRESERVED_BYTES / 1024**3:.3f} |

All original tensors outside the quantized selection are independently compared
byte-for-byte with the source. Retained FP8 indexer weights and their original
`qscale_weight` blocks are preserved together. The complete bundled dSpark
drafter remains BF16, exactly as shipped.

MCG multiplier is `0xCBAC1FED`, seed 787, sigma 0.025, with automatic output
scales. Each module's `quantize_config.json:tensor_storage` entry and safetensors
headers specify its own bits, logical dimensions and stored buffers. RoPE kv_a
has logical output 576 stored at 640; all transforms are completed before
cropping. Self-owned NoPE kv_a is 512 wide; KV-reuse layers 39,43,...,71 have
no independent kv_a matrix. The 72 layer shards plus one extras shard contain
{TARGET_TENSOR_COUNT:,} tensors and {TARGET_PAYLOAD_BYTES:,} payload bytes
({TARGET_PAYLOAD_BYTES / 1024**3:.3f} GiB), excluding headers and metadata.

Calibration uses the pinned public agentic panel, with the checkpoint's native
Tekken tokenizer and chat formatting. Sample count: {samples}; token count:
{tokens}. Exact panel, token, producer and source identities are recorded in
`quantization-provenance.json`. One interleaved causal pass propagates serialized,
reconstructed predecessors before successor calibration. Expert gate/up
precedes nonlinear down-input capture. Rolling boundaries preserve the shared
indexer's top-k state and the layer-35 KV latent used by later NoPE layers.
Per-matrix calibrated errors are retained in `quantization-errors/` JSONL bundles.

This job ran on raptor GPU 1 (RTX PRO 6000 Blackwell). It validates the
checkpoint inventory, serialized format and native byte preservation.
Full-model generation, long-context quality and serving performance are separate
qualification work. A runtime must support this Mistral-native mixed K4/K6
format, retained vendor FP8 indexer, BF16 MLA kv_b, official vision and BF16
dSpark components. No EXLR8 sidecar is required.
'''

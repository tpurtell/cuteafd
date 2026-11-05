"""One-pass K3.1 assembly using K3 byte-span copying and native-byte audits."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import shutil
import struct

from kimi_k3.core_k6.export import _preserved_audit, _fsync_tree
from kimi_k3.uniform_k2.storage import assemble_shard, atomic_json, canonical, read_header, verify_receipt
from .recipe import (BASE_REPO, BASE_REVISION, TARGET_REPO, MODULE_COUNT, CORE_MODULE_COUNT,
    EXPERT_MODULE_COUNT, MCG_MULTIPLIER, SHARD_COUNT, TARGET_TENSOR_COUNT, TARGET_PAYLOAD_BYTES,
    packed_specs, reviewed_modules, reviewed_source_specs, selected_modules, source_names)

_DTYPES = dict(I16="int16", F16="float16", I32="int32")


def shard_name(ordinal):
    return f"model-{ordinal:05d}-of-{SHARD_COUNT:05d}.safetensors"


def target_shard(name):
    match = re.match(r"language_model\.model\.layers\.([0-9]+)\.", name)
    return shard_name(int(match[1]) + 1 if match else SHARD_COUNT)


def _check_packed(module, geometry, path):
    expected = packed_specs(module, geometry)
    header, base = read_header(path)
    if set(header) - {"__metadata__"} != set(expected):
        raise ValueError(f"replacement tensor names differ: {module}")
    for name, spec in expected.items():
        if (header[name]["dtype"], header[name]["shape"]) != spec:
            raise ValueError(f"replacement tensor geometry differs: {name}")
    with Path(path).open("rb") as stream:
        stream.seek(base + header[module + ".mcg"]["data_offsets"][0])
        if struct.unpack("<I", stream.read(4))[0] != MCG_MULTIPLIER:
            raise ValueError(f"replacement codebook multiplier differs: {module}")
    return expected


def _assemble(source_entries, selection, replacements, shard, receipt_path, identity, **labels):
    """Commit one shard only after exact replacement and retained-byte checks."""
    if set(replacements) != set(selection):
        raise ValueError("replacement modules do not cover the reviewed shard")
    entries = dict(source_entries)
    expected = {}
    headers = {}
    for name, (path, original) in entries.items():
        if path not in headers:
            headers[path] = read_header(path)[0]
        spec = headers[path][original]
        expected[name] = (spec["dtype"], spec["shape"])
    for module, geometry in selection.items():
        for name in source_names(module, geometry):
            if name not in entries:
                raise ValueError(f"selected source tensor missing: {name}")
            del entries[name], expected[name]
    preserved = dict(entries)
    for module, geometry in selection.items():
        for name, spec in _check_packed(module, geometry, replacements[module]).items():
            if name in entries:
                raise ValueError(f"replacement collision: {name}")
            entries[name] = (Path(replacements[module]), name)
            expected[name] = spec
    expected_payload = sum(math.prod(shape) * {"U8": 1, "BF16": 2, "F16": 2, "I16": 2,
                                               "I32": 4, "F32": 4}[dtype] for dtype, shape in expected.values())
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("identity") != identity or receipt.get("filename") != shard.name
                or any(receipt.get(k) != v for k, v in labels.items())
                or receipt.get("payload_bytes") != expected_payload or receipt.get("tensors") != len(entries)
                or receipt.get("preserved_audit", {}).get("tensors") != len(preserved)):
            raise ValueError("committed export identity or inventory differs")
        verify_receipt(shard, receipt, checksum=True)
    else:
        if shard.exists():
            shard.unlink()  # No durable commitment owns an orphaned shard.
        receipt = assemble_shard(entries, shard)
        receipt.update(identity=identity, filename=shard.name, tensors=len(entries),
                       preserved_audit=_preserved_audit(preserved, shard), **labels)
        atomic_json(receipt_path, receipt)
    header, _ = read_header(shard)
    if {n: (v["dtype"], v["shape"]) for n, v in header.items() if n != "__metadata__"} != expected:
        raise ValueError("assembled shard inventory differs")
    return receipt


def assemble_layer(source, layer, replacements, export, identity):
    if not 0 <= layer < 93:
        raise ValueError("invalid Kimi layer")
    export = Path(export)
    export.mkdir(parents=True, exist_ok=True)
    prefix = f"language_model.model.layers.{layer}."
    selection = {m: g for m, g in reviewed_modules().items() if m.startswith(prefix)}
    entries = {n: (source.snapshot / shard, n) for n, shard in source.weight_map.items() if n.startswith(prefix)}
    return _assemble(entries, selection, replacements, export / shard_name(layer + 1),
                     export / f"layer-{layer:03d}-receipt.json", identity, layer=layer)


def receipt_path(export, shard):
    return Path(export) / ("extras-receipt.json" if shard == shard_name(SHARD_COUNT)
                          else f"layer-{int(shard.split('-')[1]) - 1:03d}-receipt.json")


def audit_export(source, selection, export, identity):
    """Independently reload final headers and compare all tensor names and specs."""
    export = Path(export)
    expected = {n: spec for n, spec in reviewed_source_specs().items()
                if not any(n.endswith(suffix) and n.removesuffix(suffix) in selection
                           for suffix in (".weight", ".weight_packed", ".weight_scale"))}
    for module, geometry in selection.items():
        expected.update(packed_specs(module, geometry))
    actual_map, total = {}, 0
    expected_shards = {shard_name(n) for n in range(1, SHARD_COUNT + 1)}
    if {p.name for p in export.glob("*.safetensors")} != expected_shards:
        raise ValueError("assembled checkpoint shard inventory differs")
    for shard in sorted(expected_shards):
        path = export / shard
        receipt = json.loads(receipt_path(export, shard).read_text())
        if receipt.get("identity") != identity or receipt.get("filename") != shard:
            raise ValueError("final shard producer identity differs")
        verify_receipt(path, receipt)
        if "preserved_audit" not in receipt:
            raise ValueError("final shard lacks its native-byte audit")
        header, _ = read_header(path)
        for name, value in header.items():
            if name == "__metadata__":
                continue
            if (name in actual_map or target_shard(name) != shard
                    or expected.get(name) != (value["dtype"], value["shape"])):
                raise ValueError(f"final checkpoint tensor differs: {name}")
            actual_map[name] = shard
            total += value["data_offsets"][1] - value["data_offsets"][0]
    if (set(actual_map) != set(expected) or len(actual_map) != TARGET_TENSOR_COUNT
            or total != TARGET_PAYLOAD_BYTES):
        raise ValueError("final checkpoint tensor/payload totals differ")
    return actual_map, total


def finalize_export(source, selection, head_replacements, export, identity, evidence_root):
    if selection != selected_modules(source):
        raise ValueError("final selection differs from the reviewed source")
    export = Path(export)
    extras = {n: (source.snapshot / shard, n) for n, shard in source.weight_map.items()
              if not n.startswith("language_model.model.layers.")}
    head = {n: g for n, g in selection.items() if n == "language_model.lm_head"}
    _assemble(extras, head, head_replacements, export / shard_name(SHARD_COUNT),
              export / "extras-receipt.json", identity)
    weight_map, total = audit_export(source, selection, export, identity)
    tensor_storage = {}
    for module, geometry in selection.items():
        tensor_storage[module] = dict(quant_format="exl3", bits_per_weight=geometry["bits"],
            mcg_multiplier=MCG_MULTIPLIER,
            **{k: geometry[k] for k in ("in_features", "out_features", "stored_out_features")},
            stored_tensors={n: dict(shape=shape, torch_dtype=_DTYPES[dtype])
                            for n, (dtype, shape) in packed_specs(module, geometry).items()})
    declaration = dict(quant_method="exl3", format="exl3", checkpoint_format="exl3",
                       bits=2., core_bits=6., per_module_metadata="quantize_config.json:tensor_storage")
    config = json.loads(canonical(source.config))
    config["quantization_config"] = declaration
    config["text_config"]["quantization_config"] = declaration
    atomic_json(export / "config.json", config)
    atomic_json(export / "model.safetensors.index.json", dict(metadata=dict(total_size=total), weight_map=weight_map))
    atomic_json(export / "quantize_config.json", dict(method="exl3", **declaration, codebook="mcg",
        out_scales="auto", group_size=-1, desc_act=False, tensor_storage=tensor_storage,
        module_include=[r"^language_model\.model\.layers\.[0-9]+\.block_sparse_moe\.experts\.[0-9]+\.w[123]$"] +
            ["^" + re.escape(n) + "$" for n, g in selection.items() if g["bits"] == 6],
        meta=dict(quantization_provenance=identity)))
    excluded = {"config.json", "quantize_config.json", "model.safetensors.index.json", "README.md",
                ".gitattributes", "quantization-provenance.json"}
    for path in source.snapshot.iterdir():
        if path.is_file() and path.name not in excluded and not path.name.endswith(".safetensors"):
            shutil.copyfile(path, export / path.name)
        elif path.is_dir() and path.name == "assets":
            shutil.copytree(path, export / path.name, dirs_exist_ok=True)
    # Aggregate complete per-projection records so upload never has 248k jobs.
    evidence_root = Path(evidence_root)
    errors = export / "quantization-errors"
    errors.mkdir(exist_ok=True)
    bundles = {}
    seen = set()
    for directory in sorted(evidence_root.iterdir()):
        if not directory.is_dir():
            continue
        records = []
        for path in sorted(directory.glob("*.json")):
            record = json.loads(path.read_text())
            module = record.get("module")
            if module is None:
                continue
            if module not in selection or module in seen or record.get("identity", identity) != identity:
                raise ValueError("per-projection evidence inventory/identity differs")
            geometry, metrics = selection[module], record.get("metrics", {})
            if (record.get("logical_shape") != [geometry["out_features"], geometry["in_features"]]
                    or metrics.get("quantizer_path") != "hessian_ldlq"
                    or metrics.get("hessian_metric_status") != "ok"
                    or not isinstance(metrics.get("hessian_weighted_relative_error"), (int, float))
                    or not math.isfinite(metrics["hessian_weighted_relative_error"])
                    or metrics["hessian_weighted_relative_error"] < 0):
                raise ValueError("per-projection calibrated numerical evidence differs")
            seen.add(module)
            records.append(canonical(record) + "\n")
        if records:
            destination = errors / (directory.name + ".jsonl")
            destination.write_text("".join(records))
            bundles[destination.name] = len(records)
    if seen != set(selection):
        raise ValueError("per-projection evidence is incomplete")
    atomic_json(errors / "index.json", dict(schema="kimi-k31-exl3-diagnostics-jsonl-v1",
                                            records=len(seen), bundles=bundles))
    atomic_json(export / "quantization-provenance.json", identity)
    (export / "README.md").write_text(_model_card(identity))
    _fsync_tree(export)
    return dict(tensors=len(weight_map), payload_bytes=total, shards=SHARD_COUNT,
                quantized_projections=MODULE_COUNT, core_projections=CORE_MODULE_COUNT,
                routed_projections=EXPERT_MODULE_COUNT)


def _model_card(identity):
    return f'''---
license: other
license_name: kimi-k3.1-pro
license_link: https://huggingface.co/{BASE_REPO}/blob/{BASE_REVISION}/LICENSE
base_model: {BASE_REPO}
datasets:
- wrldsuksgo2mars/agentic-quant-panel-v1
tags:
- exl3
- quantized
- kimi_k31
---
# Kimi K3.1 Pro EXL3 K2 + Core K6

One interleaved causal quantization of [{BASE_REPO}](https://huggingface.co/{BASE_REPO})
at `{BASE_REVISION}`: 247,296 routed expert projections use K2 and 926 selected
attention, shared-expert, latent MoE and language-head projections use K6.
The standard EXL3 MCG codebook uses multiplier 0xCBAC1FED, seed 787,
sigma regularization 0.025 and automatic output scales.

Core projections are quantized before the corresponding routed experts. Every
successor calibrates on serialized and reconstructed earlier EXL3 weights.
The official MXFP4 expert weights are dequantized exactly to BF16. Calibration
uses the pinned agentic panel with the checkpoint's own tokenizer and chat
template, thinking enabled. Dense Hessians contain actual observations only;
the recorded K3 cold-expert floor applies to routed experts.

All other tensors retain source bytes: embeddings, dense layer-0 FFN, routers,
norms, recurrent KDA components, kv_b_proj, attention-residual mixers, DSA
indexer, vision and multimodal projector. Per-module dtype, logical/padded
geometry and bitrates are authoritative in `quantize_config.json:tensor_storage`.
The DSA kv_a logical width 576 is stored at width 640.

Quantization ran on one RTX PRO 6000 Blackwell Max-Q with a disk-backed rolling
frontier. Export independently compares preserved tensor bytes and reloads the
complete tensor inventory. Source license and native tokenizer assets are
included verbatim; provenance and per-projection diagnostics accompany the model.
Full-model generation, KL/top-1 qualification and serving benchmarks are not
part of this quantization run.
'''

"""CPU-only final checkpoint reload and producer completion audit."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

from kimi_k3.uniform_k2.storage import atomic_json, canonical
from .export import audit_export
from .recipe import header_source, reviewed_modules, packed_specs, BASE_REPO, MCG_MULTIPLIER


def _same_bytes(source, target):
    """Compare only native assets; no weight payload enters this pass."""
    if target.is_symlink() or not target.is_file() or source.stat().st_size != target.stat().st_size:
        raise ValueError(f"native publication asset differs: {target.name}")
    with source.open("rb") as left, target.open("rb") as right:
        while data := left.read(4 * 1024 * 1024):
            if data != right.read(len(data)):
                raise ValueError(f"native publication asset bytes differ: {target.name}")


def audit_metadata(source, selection, export, identity, weight_map, total):
    """Verify standard loader metadata, native assets, and complete error evidence."""
    index = json.loads((export / "model.safetensors.index.json").read_text())
    if index != dict(metadata=dict(total_size=total), weight_map=weight_map):
        raise ValueError("published index differs from independently reloaded tensor headers")
    declaration = dict(quant_method="exl3", format="exl3", checkpoint_format="exl3",
                       bits=2., core_bits=6., per_module_metadata="quantize_config.json:tensor_storage")
    expected = json.loads(canonical(source.config))
    expected["quantization_config"] = declaration
    expected["text_config"]["quantization_config"] = declaration
    if json.loads((export / "config.json").read_text()) != expected:
        raise ValueError("published model config changes native architecture or quantization declaration")
    if json.loads((export / "quantization-provenance.json").read_text()) != identity:
        raise ValueError("published provenance differs from completed producer")
    quant = json.loads((export / "quantize_config.json").read_text())
    if (any(quant.get(k) != v for k, v in declaration.items())
            or set(quant) != set(declaration) | {"method", "meta", "codebook", "out_scales", "group_size",
                                                "desc_act", "tensor_storage", "module_include"}
            or quant.get("method") != "exl3" or quant.get("group_size") != -1 or quant.get("desc_act") is not False
            or quant.get("meta") != dict(quantization_provenance=identity)
            or quant.get("codebook") != "mcg" or quant.get("out_scales") != "auto"
            or set(quant.get("tensor_storage", {})) != set(selection)):
        raise ValueError("published quantizer metadata differs from the one-pass recipe")
    patterns = [r"^language_model\.model\.layers\.[0-9]+\.block_sparse_moe\.experts\.[0-9]+\.w[123]$"] + [
        "^" + re.escape(n) + "$" for n, g in selection.items() if g["bits"] == 6]
    if quant.get("module_include") != patterns:
        raise ValueError("published module selection patterns differ from the one-pass recipe")
    widths = dict(I16="int16", I32="int32", F16="float16")
    for module, geometry in selection.items():
        spec = dict(quant_format="exl3", bits_per_weight=geometry["bits"], mcg_multiplier=MCG_MULTIPLIER,
            **{k: geometry[k] for k in ("in_features", "out_features", "stored_out_features")},
            stored_tensors={n: dict(shape=shape, torch_dtype=widths[dtype])
                            for n, (dtype, shape) in packed_specs(module, geometry).items()})
        if quant["tensor_storage"][module] != spec:
            raise ValueError(f"published per-module quantization metadata differs: {module}")
    excluded = {"config.json", "quantize_config.json", "model.safetensors.index.json", "README.md",
                ".gitattributes", "quantization-provenance.json"}
    assets = {}
    for path in source.snapshot.iterdir():
        if path.is_file() and path.name not in excluded and not path.name.endswith(".safetensors"):
            assets[path.name] = path
        elif path.is_dir() and path.name == "assets":
            assets.update({p.relative_to(source.snapshot).as_posix(): p for p in path.rglob("*") if p.is_file()})
    if "LICENSE" not in assets:
        raise ValueError("native source license asset missing")
    for name, path in assets.items():
        _same_bytes(path, export / name)
    card = (export / "README.md").read_text()
    if (not card.startswith("---\n") or f"\nbase_model: {BASE_REPO}\n" not in card
            or "\nlicense_name: kimi-k3.1-pro\n" not in card or "\nlicense: other\n" not in card):
        raise ValueError("model card source or native license metadata differs")
    errors = export / "quantization-errors"
    diagnostics = json.loads((errors / "index.json").read_text())
    bundles = diagnostics.get("bundles", {})
    if diagnostics.get("records") != len(selection) or sum(bundles.values()) != len(selection):
        raise ValueError("final diagnostic bundle inventory differs")
    seen = set()
    for name, count in bundles.items():
        if Path(name).name != name or not name.endswith(".jsonl"):
            raise ValueError("invalid diagnostic bundle filename")
        measured = 0
        with (errors / name).open() as stream:
            for line in stream:
                record = json.loads(line)
                module = record.get("module")
                if module not in selection or module in seen or record.get("identity", identity) != identity:
                    raise ValueError("diagnostic bundle module identity differs")
                geometry, metrics = selection[module], record.get("metrics", {})
                if (record.get("logical_shape") != [geometry["out_features"], geometry["in_features"]]
                        or metrics.get("quantizer_path") != "hessian_ldlq" or metrics.get("hessian_metric_status") != "ok"
                        or not isinstance(metrics.get("hessian_weighted_relative_error"), (int, float))
                        or not math.isfinite(metrics["hessian_weighted_relative_error"])
                        or metrics["hessian_weighted_relative_error"] < 0):
                    raise ValueError("diagnostic bundle calibrated numerical evidence differs")
                seen.add(module)
                measured += 1
        if measured != count:
            raise ValueError("diagnostic bundle record count differs")
    if seen != set(selection):
        raise ValueError("diagnostic evidence omits selected modules")
    # Refuse stale two-pass/base quantizer metadata, extra tensors, and unknown
    # publication files. Internal transaction receipts and Hub cache stay local.
    allowed = (set(assets) | set(weight_map.values()) | excluded | {"quantization-errors/index.json"}
               | {"quantization-errors/" + n for n in bundles})
    for path in export.rglob("*"):
        relative = path.relative_to(export).as_posix()
        if ".cache" in path.relative_to(export).parts:
            continue
        if path.is_symlink():
            raise ValueError("publication payload cannot contain symlinks")
        if path.is_file() and not path.name.endswith("-receipt.json") and relative not in allowed:
            raise ValueError(f"unreviewed publication file: {relative}")
    return dict(native_assets=len(assets), diagnostic_records=len(seen), diagnostic_bundles=len(bundles))


def audit_completed_export(root, source=None):
    root = Path(root).resolve()
    plan = json.loads((root / "plan.json").read_text())
    complete = json.loads((root / "quantization-complete.json").read_text())
    identity = plan["identity"]
    if complete.get("identity") != identity:
        raise ValueError("completion marker belongs to a different producer")
    if source is None:
        source = header_source(plan.get("source", identity.get("source")))
    export = Path(plan["export"])
    selection = reviewed_modules()
    weight_map, total = audit_export(source, selection, export, identity)
    metadata = audit_metadata(source, selection, export, identity, weight_map, total)
    result = dict(status="passed", identity=identity, tensors=len(weight_map), payload_bytes=total,
                  **metadata, method="independent CPU header/metadata reload, native asset byte comparison, "
                                    "complete diagnostics, and committed native-weight byte audit receipts")
    atomic_json(root / "finalization-audit.json", result)
    return result

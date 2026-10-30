#!/usr/bin/env python3
"""Audit completed Step-5 pilot receipts, then retire only recoverable payloads."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "quantization"))
if __package__ in (None, ""):
    __package__ = "step5.core_k6"

from .pilot import (ARMS, CHECKS, LAYERS, LATE_LAYER, modules,
    validate_qualification, validate_report_code)
from kimi_k3.uniform_k2.storage import atomic_json, canonical, sha256_file, verify_receipt


def audit_evidence(root, report=None):
    """CPU-only independent checks; no quantization, replay or policy migration."""
    from safetensors import safe_open
    root = Path(root).resolve()
    report = json.loads((root / "qualification.json").read_text()) if report is None else report
    validate_report_code(validate_qualification(report))
    identity = report["identity"]
    if json.loads((root / "plan.json").read_text()) != identity:
        raise ValueError("pilot qualification differs from captured plan")
    result = json.loads((root / "results.json").read_text())
    if result != dict(identity=identity, results=report["results"]):
        raise ValueError("pilot result observations differ from qualification")
    source, tokens = Path(identity["source"]), Path(identity["tokens"])
    for filename, key in (("config.json", "source_config_sha256"),
                          ("model.safetensors.index.json", "source_index_sha256")):
        if sha256_file(source / filename) != identity[key]:
            raise ValueError("pilot source metadata changed after capture")
    if sha256_file(tokens / "manifest.json") != identity["token_manifest_sha256"]:
        raise ValueError("pilot corpus manifest changed after capture")
    from step5.prepare_corpus import load_token_records
    selected = {}
    for split in ("calibration", "heldout"):
        records = {r["id"]:r for r in load_token_records(tokens, split)}
        selected[split] = [records[name] for name in identity[split + "_ids"]]
        if (hashlib.sha256(canonical(selected[split]).encode()).hexdigest() != identity[split + "_sha256"]
                or sum(len(r["input_ids"]) for r in selected[split]) != identity[split + "_tokens"]):
            raise ValueError("pilot selected token observations changed after capture")
    verified = {name: sha256_file(root / name) for name in ("plan.json", "results.json", "mtp-proof.json")}
    def receipt(path, value):
        path = Path(path)
        verify_receipt(path, value, checksum=True)
        verified[str(path.relative_to(root))] = value["sha256"]
    def events(directory):
        path = directory / "events.jsonl"
        verified[str(path.relative_to(root))] = sha256_file(path)
        return [json.loads(line) for line in path.read_text().splitlines()]
    def one(values, kind, **fields):
        selected = [r for r in values if r.get("event") == kind and all(r.get(k) == v for k,v in fields.items())]
        if len(selected) != 1:
            raise ValueError(f"expected one measured {kind} event: {fields}")
        return selected[0]
    def sparse_pruning(event):
        meta = event.get("sparse_metadata")
        if (not isinstance(meta, dict) or meta.get("tokens") != event["tokens"]
                or meta.get("topk") != 512 or meta.get("region_size") != 8
                or not 0 < meta.get("maximum_selected_regions", 0) <= 513
                or meta.get("maximum_available_regions", 0) <= meta["maximum_selected_regions"]
                or meta.get("pruned_query_rows", 0) <= 0 or meta.get("sparse_pruning_exercised") is not True
                or any(len(meta.get(name, "")) != 64 for name in ("metadata_sha256", "region_counts_sha256", "query_positions_sha256"))):
            raise ValueError("Step-5 sparse pruning proof is missing or inconsistent")
    measured = events(root)
    candidate_count, hessian_count = 0, 0
    for row in report["results"]:
        event = one(measured, "pilot_result", tensor=row["tensor"], arm=row["arm"])
        if any(event.get(key) != value for key,value in row.items()):
            raise ValueError("pilot score and measured event differ")
        if row["arm"] == "bf16":
            continue
        path = root / "candidates" / (row["tensor"] + "." + row["arm"] + ".safetensors")
        item = json.loads(path.with_suffix(".json").read_text())
        if item["identity"] != identity:
            raise ValueError("pilot candidate identity differs")
        receipt(path, item["receipt"])
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            import math
            widths = {"I16":2, "I32":4, "F16":2, "BF16":2, "F32":4, "U8":1}
            size = sum(math.prod(reader.get_slice(key).get_shape()) * widths[reader.get_slice(key).get_dtype()]
                       for key in reader.keys())
            if size != row["payload_bytes"]:
                raise ValueError("pilot candidate byte accounting differs")
        candidate_count += 1
    for split in ("calibration", "heldout"):
        ev = events(root / split)
        for layer in range(8):
            one(ev, "layer_committed", layer=layer)
            if not any(r.get("event") == "native_boundary_exact" and r.get("layer") == layer for r in ev):
                raise ValueError("pilot native boundary event missing")
            if split == "calibration" and layer in (3, 7):
                sparse_pruning(one(ev, "native_boundary_exact", layer=layer, row_id=identity["long_context_witness"]))
        one(ev, "late_expert_witness", source_layer=88, input_layer=7, expert=0,
            tokens=identity[split + "_tokens"], actual_layer88_calibration=False)
        for layer in (*LAYERS, LATE_LAYER):
            path = root / split / "hessians" / f"layer-{layer}.json"
            manifest = json.loads(path.read_text())
            verified[str(path.relative_to(root))] = sha256_file(path)
            if manifest["identity"] != identity or set(manifest["files"]) != set(modules(layer)):
                raise ValueError("pilot Hessian inventory/identity differs")
            for name,value in manifest["files"].items():
                path = root / split / "hessians" / (name + ".safetensors")
                receipt(path, value)
                with safe_open(str(path), framework="pt", device="cpu") as reader:
                    if (set(reader.keys()) != {"H", "count"}
                            or int(reader.get_tensor("count")) != identity[split + "_tokens"]
                            or reader.get_slice("H").get_dtype() != "F32"):
                        raise ValueError("pilot Hessian observations differ")
                hessian_count += 1
            if layer == LATE_LAYER:
                continue
            manifest = json.loads((root / split / "proof-inputs" / f"layer-{layer}.json").read_text())
            if manifest["identity"] != identity or not manifest["entries"]:
                raise ValueError("pilot native proof inputs missing")
            records = selected[split]
            expected_rows = {records[0]["id"]: records[0]}
            expected_rows.update({r["id"]:r for r in records if r["id"] == identity["long_context_witness"]})
            if {item["row_id"] for item in manifest["entries"]} != set(expected_rows):
                raise ValueError("pilot native proof row inventory differs")
            for item in manifest["entries"]:
                if item["tokens"] != len(expected_rows[item["row_id"]]["input_ids"]):
                    raise ValueError("pilot native proof row length differs")
                receipt(root / split / item["file"], item["receipt"])
                one(ev, "native_boundary_exact", layer=layer, row_id=item["row_id"], tokens=item["tokens"])
                if split == "calibration":
                    event = one(measured, "serialized_boundary_exact", layer=layer,
                        row_id=item["row_id"], tokens=item["tokens"])
                    if layer == 7 and item["row_id"] == identity["long_context_witness"]:
                        sparse_pruning(event)
    mtp = json.loads((root / "mtp-proof.json").read_text())
    if (mtp != report["mtp"] or mtp.get("prefix_layers") != 8
            or mtp.get("final_target_frontier") is not False or len(mtp.get("stages", [])) != 3
            or mtp.get("source_row") != selected["calibration"][0]["id"]
            or mtp.get("tokens") != len(selected["calibration"][0]["input_ids"])):
        raise ValueError("pilot MTP proof identity/scope differs")
    for stage, item in enumerate(mtp["stages"]):
        rows = mtp["tokens"] - stage - 1
        if (item.get("stage") != stage or item.get("layer") != stage + 92 or item.get("origin") != stage + 1
                or item.get("shifted_pairs") != rows or item.get("byte_exact") is not True
                or item.get("serialized_k4_byte_exact") is not True or item.get("final_target_frontier") is not False):
            raise ValueError("pilot MTP teacher-stage contract differs")
        evidence, metrics = item["projection_evidence"], item["projection_metrics"]
        if (evidence.get("bits") != 4 or evidence.get("identity_rows") != 0
                or evidence.get("hessian_rows") != rows or metrics.get("hessian_sample_count") != rows
                or metrics.get("quantizer_path") != "hessian_ldlq" or metrics.get("hessian_metric_status") != "ok"):
            raise ValueError("pilot MTP calibrated projection evidence differs")
        one(measured, "mtp_boundary_exact", stage=stage, layer=92 + stage, origin=stage + 1, shifted_pairs=rows)
        receipt(root / "candidates" / f"mtp-stage{stage}-eh-k4.safetensors", item["projection_receipt"])
        candidate_count += 1
    audit = dict(status="passed", identity=identity, native_checks=dict.fromkeys(CHECKS, True),
        verified_files=verified, candidate_payloads=candidate_count, hessians=hessian_count,
        metric_rows=len(report["results"]), finalization="CPU-only-receipt-and-event-audit-no-numerical-rerun",
        time=datetime.now(timezone.utc).isoformat())
    atomic_json(root / "finalization-audit.json", audit)
    return audit


def retire_intermediates(root):
    root = Path(root).resolve()
    audit = json.loads((root / "finalization-audit.json").read_text())
    report = validate_report_code(validate_qualification(json.loads((root / "qualification.json").read_text())))
    if audit.get("status") != "passed" or audit["identity"] != report["identity"]:
        raise ValueError("pilot recovery retirement requires the final audited identity")
    removed = []
    for split in ("calibration", "heldout"):
        for name in ("frontier", "proof-inputs"):
            path = root / split / name
            if path.is_symlink():
                raise ValueError("refuse symlink pilot retirement")
            if path.exists():
                shutil.rmtree(path)
                removed.append(str(path.relative_to(root)))
    for path in (root / "candidates").glob("*.safetensors"):
        if path.is_symlink():
            raise ValueError("refuse symlink candidate retirement")
        path.unlink()
        removed.append(str(path.relative_to(root)))
    atomic_json(root / "recovery-retired.json", dict(status="passed", identity=report["identity"],
        removed=removed, finalization_audit_sha256=sha256_file(root / "finalization-audit.json"),
        time=datetime.now(timezone.utc).isoformat()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--retire", action="store_true")
    args = parser.parse_args(argv)
    audit = audit_evidence(args.root)
    if args.retire:
        retire_intermediates(args.root)
    print(json.dumps({key:value for key,value in audit.items() if key not in ("identity", "verified_files")}))


if __name__ == "__main__":
    main()

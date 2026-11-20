"""Validate and seal a Hy4 package once all numerical groups are committed."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from kimi_k3.uniform_k2.storage import atomic_json, file_identity, verify_receipt

from . import recipe
from .export import DirectMCWriter, verify_original_discovery
from .source import HeaderSource

REPO = Path(__file__).resolve().parents[2]
HEAD = "lm_head.weight"


def read_json(path):
    return json.loads(path.read_text())


def seal_package(root, writer, identity, document, selection, retained, engram, drafter_bits,
                 error_csv, progress):
    output = writer.output
    result = writer.finalize(core_selection=selection, engram=engram, document=document,
                             error_csv=error_csv, card_path=root / "README.md",
                             k6_drafter_bits=drafter_bits, retained_by_gate=retained)
    state = read_json(output / "package-state.json")
    discovery = verify_original_discovery(output, state["direct_expected_mappings"]["original"])
    complete = dict(status="passed", identity=identity, output=str(output),
                    revision=recipe.BASE_REVISION, repository=recipe.TARGET_REPO,
                    completed_target_layers=recipe.LAYER_COUNT,
                    completed_drafter_layers=recipe.MTP_LAYER_COUNT,
                    completed_encodes=progress["committed_encodes"], error_csv=str(error_csv),
                    task_intermediates=[str(root / "stages")],
                    package_validation=str(output / "package-validation.json"),
                    package_status=result["status"], original_discovery=discovery)
    atomic_json(root / "numerical-complete.json", complete)
    atomic_json(root / "status.json", dict(progress, status="passed", phase="numerical_complete",
                                         updated_at=datetime.now(timezone.utc).isoformat()))
    return complete


def repair_head_mapping(root, writer):
    """Reclassify the sealed single-tensor BF16 shard for the original index."""
    group = writer.state["groups"]["head"]
    rows = [row for row in group["shards"] if HEAD in row["names"]]
    if not writer.group_complete("head") or len(rows) != 1 or rows[0]["names"] != [HEAD]:
        raise ValueError("Hy4 head recovery requires a sealed single-tensor source shard")
    row = rows[0]
    location = writer.source_inventory["locations"][HEAD]
    source = (writer.source / location["shard"]).resolve(strict=True)
    proof = row["receipt"]["original_tensor_proofs"][HEAD]
    if (proof["source_name"] != location["source_name"] or proof["source_path"] != str(source)
            or proof["source_file_identity"] != file_identity(source)):
        raise ValueError("Hy4 BF16 head source proof differs")
    recovery_path = root / "head-package-recovery.json"
    recovery = dict(schema="hy4-head-package-recovery-v1", identity=writer.identity,
                    tensor=HEAD, shard=row["filename"], file_identity=row["receipt"]["file_identity"],
                    sha256=row["receipt"]["sha256"], from_view="shared", to_view="original")
    if recovery_path.exists():
        if read_json(recovery_path) != recovery:
            raise ValueError("Hy4 BF16 head recovery receipt differs")
    elif row["kind"] != "shared":
        raise ValueError("Hy4 recovery requires the original shared head error")
    kind = row["kind"]
    other_kind = "original" if kind == "shared" else "shared"
    if (kind not in ("shared", "original")
            or any(set(group[field][kind]) != {HEAD} or group[field][other_kind]
                   for field in ("entries", "sizes", "specs"))
            or group["entries"][kind][HEAD] != [str(source), location["source_name"]]
            or group["specs"][kind][HEAD] != writer.source_inventory["specs"][HEAD]
            or group["sizes"][kind][HEAD] != writer.source_inventory["sizes"][HEAD]):
        raise ValueError("Hy4 BF16 head mapping differs from the reviewed one-tensor repair")
    if not recovery_path.exists():
        atomic_json(recovery_path, recovery)
    if kind == "shared":
        for field in ("entries", "sizes", "specs"):
            group[field]["original"][HEAD] = group[field]["shared"].pop(HEAD)
        row["kind"] = "original"
        atomic_json(writer.state_path, writer.state)


def recover_completed_encodes(root, output, source, engram_receipt):
    root, output = root.resolve(), output.resolve()
    if (root / "numerical-complete.json").exists():
        raise ValueError("Hy4 package has already completed")
    identity = read_json(root / "identity.json")
    frontier = read_json(root / "frontier.json")
    reviewed = recipe.reviewed_core_modules(include_head=True, include_kv_b=True)
    total = len(reviewed) * 2 + recipe.EXPERT_MODULE_COUNT
    if (frontier["identity"] != identity or frontier["target"] != recipe.LAYER_COUNT
            or frontier["head"] is not True or frontier["drafter"] != recipe.MTP_LAYER_COUNT
            or frontier["encodes"] != total):
        raise ValueError("Hy4 recovery requires completed numerical groups under the frozen identity")
    old_head = subprocess.check_output(
        ["git", "show", "2d9b493df0:quantization/hy4_flash/run.py"], cwd=REPO)
    for path, digest in identity["code"]["files"].items():
        data = old_head if path == "quantization/hy4_flash/run.py" else (REPO / path).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Hy4 frozen producer file differs: " + path)
    header = HeaderSource(source)
    if header.source_fingerprints != identity["source_fingerprints"]:
        raise ValueError("Hy4 frozen source fingerprints changed")
    writer = DirectMCWriter(output, header.snapshot, identity, header.as_inventory())
    stage = read_json(root / "stages" / frontier["stage"] / "manifest.json")
    if stage["identity"] != identity:
        raise ValueError("Hy4 final activation stage differs from the numerical identity")
    gate_names = {"head", *(f"target-{index:03d}" for index in (0, 1, 2, recipe.LAYER_COUNT - 1)),
                  *(f"draft-{index:03d}" for index in range(recipe.MTP_LAYER_COUNT))}
    gate_receipts = {p.stem: read_json(p) for p in (root / "gates").glob("*.json")}
    if set(gate_receipts) != gate_names or any(x["status"] != "passed" for x in gate_receipts.values()):
        raise ValueError("Hy4 recovery requires all production gates")
    retained = {r["module"] for g in gate_receipts.values() for r in g["projections"] if r.get("kept_native")}
    selection = {name: spec for name, spec in reviewed.items() if name not in retained}
    draft_bits = gate_receipts[f"draft-{recipe.MTP_LAYER_COUNT-1:03d}"]["published_k6_drafter_bits"]
    engram = read_json(engram_receipt)
    for name, receipt in engram["files"].items():
        verify_receipt(output / name, receipt)
    error_csv = root / "per-module-errors.csv"
    if not error_csv.is_file():
        raise ValueError("Hy4 sealed error CSV is missing")
    if "lm_head" not in selection:
        raise ValueError("Hy4 recovery requires a head selected for compact views")
    repair_head_mapping(root, writer)
    progress = dict(target_layers=recipe.LAYER_COUNT, drafter_layers=recipe.MTP_LAYER_COUNT,
                    completed_target_layers=frontier["target"], completed_drafter_layers=frontier["drafter"],
                    committed_encodes=total, total_encode_operations=total)
    atomic_json(root / "status.json", dict(progress, phase="package_audit", status="running",
                                         updated_at=datetime.now(timezone.utc).isoformat()))
    return seal_package(root, writer, identity, header.document, selection, retained, engram,
                        draft_bits, error_csv, progress)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "output", "source", "engram-receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        result = recover_completed_encodes(args.root, args.output, args.source, args.engram_receipt)
    except Exception as error:
        if not (args.root / "numerical-complete.json").exists():
            path = args.root / "status.json"
            status = read_json(path)
            atomic_json(path, status | dict(status="blocked", phase="package_audit",
                                            error=f"{type(error).__name__}: {error}",
                                            updated_at=datetime.now(timezone.utc).isoformat()))
        raise
    print(json.dumps({k: result[k] for k in ("status", "completed_encodes", "package_validation")}), flush=True)


if __name__ == "__main__":
    main()

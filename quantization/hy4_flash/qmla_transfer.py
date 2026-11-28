"""Score the frozen Hy4-flash QMLA transforms on the MC K8 core and shared K4 experts.

Run after MC original-view calibration with --checkpoint MC_SNAPSHOT --corpus-dir PREPARED
--output CALIBRATION_OUTPUT. Only the recorded test sequences run through the model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from qmla.capture import _atomic_json, check_split_description, sequence_sha256
from qmla.long_eval import LongContextRecord
from qmla.quantizer import parse_qmla_mode
from qmla.sidecar import attach_view_transfer, read_sidecar_manifest, validate_sidecar

from . import recipe
from .qmla_mc import MCModel, MCReader
from .qmla_capture import Hy4FlashCaptureAdapter
from .qmla_run import load_pinned_qmla_corpus, run_hy4_flash_long_eval

VIEW = "k8-compact"


def transfer_records(records, provenance):
    """Bind each input to both its sequence ID and its bytes before any forward."""
    test = check_split_description(provenance["splits"])["test"]
    rows = dict(records)
    if len(rows) != len(records):
        raise ValueError("duplicate corpus record IDs")
    result = []
    for sid, digest in zip(test["seq_ids"], test["sha256"], strict=True):
        if sid not in rows or sequence_sha256(rows[sid]) != digest:
            raise ValueError(f"K8 test tokens differ at {sid}")
        result.append(LongContextRecord(sid, "test", "window", rows[sid], digest))
    return result


def summarize_transfer(measured, identity, test_tokens):
    result = {}
    fields = ("attn_rel_err", "rtn_attn_rel_err", "hadamard_attn_rel_err", "ratio_to_rtn")
    if set(measured["combinations"]) != set(identity["modes"]):
        raise ValueError("incomplete K8 mode grid")
    for key, entry in measured["combinations"].items():
        if (set(entry["per_layer"]) != {str(layer) for layer in identity["layers"]}
                or any(row["tokens"] != test_tokens for row in entry["per_layer"].values())):
            raise ValueError(f"incomplete K8 test coverage for {key}")
        expected = test_tokens * len(identity["layers"])
        if entry["layer_tokens"] != expected:
            raise ValueError(f"K8 token count differs for {key}")
        for field in fields[:-1]:
            by_layer = [row[field] for row in entry["per_layer"].values()]
            if not all(math.isfinite(x) and x >= 0 for x in by_layer) or not math.isclose(
                sum(by_layer) / len(by_layer), entry[field], rel_tol=1e-12, abs_tol=1e-16
            ):
                raise ValueError(f"invalid K8 test measurement {key} {field}")
        result[key] = {field: entry[field] for field in (*fields, "layer_tokens")}
        result[key]["per_layer"] = {
            layer: {f: row[f] for f in (*fields, "tokens")} for layer, row in entry["per_layer"].items()
        }
    return {**identity, "split": "test", "reference": f"{VIEW} 0:0",
            "combinations": result, "wall_seconds": measured["wall_seconds"]}


def run(checkpoint, corpus_dir, output, device):
    sidecar = output / "sidecar"
    report_path = output / "qmla-validation-report.json"
    report = json.loads(report_path.read_bytes())
    manifest = read_sidecar_manifest(sidecar)
    source = manifest["source"]
    expected = dict(repo=recipe.TARGET_REPO, revision=checkpoint.name, calibrated_view="original",
                    config_sha256=hashlib.sha256((checkpoint / "config.json").read_bytes()).hexdigest())
    validate_sidecar(sidecar, expected_source=expected)
    if (report["source"] != source or report["validation"]["transform_fingerprint"] != manifest["transform_fingerprint"]
            or set(report["combinations"]) != set(manifest["calibration_provenance"]["combination_measurements"])):
        raise ValueError("MC original-view report and sidecar disagree")

    reader = MCReader(checkpoint, device, view=VIEW)
    if reader.source != {"repo": source["base_repo"], "revision": source["base_revision"]}:
        raise ValueError("MC upstream source identity differs from calibration provenance")
    provenance = manifest["calibration_provenance"]
    if provenance.get("forward_view_fingerprint") != reader.original_fingerprint:
        raise ValueError("transfer requires transforms fitted on this MC original view")
    records, panel = load_pinned_qmla_corpus(corpus_dir, panel_revision=provenance["dataset_revision"],
                                             components=tuple(provenance["components"]), full=False)
    for key in ("dataset_repo", "dataset_revision", "input_ids_sha256"):
        if panel[key] != provenance[key]:
            raise ValueError(f"K8 corpus {key} differs from calibration")
    if set(panel["components"]) != set(provenance["components"]) or any(
        panel["components"][name].get(field) != value for name, info in provenance["components"].items()
        for field, value in info.items()
    ):
        raise ValueError("K8 corpus component hashes or counts differ from calibration")
    test = transfer_records(records, provenance)
    identity = dict(source_revision=checkpoint.name, view=VIEW, view_fingerprint=reader.variant["fingerprint"],
                    transform_fingerprint=manifest["transform_fingerprint"], test_digest=provenance["splits"]["test"]["digest"],
                    layers=report["layers_calibrated"], modes=sorted(report["combinations"]))
    receipt = output / "qmla-k8-transfer.json"
    test_tokens = sum(len(r.input_ids) for r in test)
    if receipt.is_file():
        section = json.loads(receipt.read_bytes())
        if any(section.get(key) != value for key, value in identity.items()) or section != summarize_transfer(section, identity, test_tokens):
            raise ValueError("existing K8 transfer receipt differs from requested sidecar, test set or view")
    else:
        measured = run_hy4_flash_long_eval(
            Hy4FlashCaptureAdapter(MCModel(reader)), sidecar_path=sidecar, records=test, corpus_meta=panel,
            layers=identity["layers"], modes=[parse_qmla_mode(mode) for mode in identity["modes"]],
            expected_source=expected, device=device, phase="k8_transfer",
        )
        section = summarize_transfer(measured, identity, test_tokens)
        _atomic_json(receipt, section)

    # Publish compact summaries in sidecar metadata. The local report retains individual layer rows.
    summary = {key: value for key, value in section.items() if key not in ("combinations", "wall_seconds")}
    summary["combinations"] = {
        key: {k: v for k, v in row.items() if k != "per_layer"} for key, row in section["combinations"].items()
    }
    updated = attach_view_transfer(sidecar, VIEW, summary)
    report.setdefault("view_transfer_validation", {})[VIEW] = section
    report["sidecar_bytes"] = updated["tensors_bytes"]
    for field in ("tensors_sha256", "tensors_bytes"):
        report["validation"][field] = updated[field]
    _atomic_json(report_path, report)
    return section


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--corpus-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = run(args.checkpoint.resolve(), args.corpus_dir, args.output, args.device)
    print(json.dumps({"phase": "k8_transfer", "status": "passed", "modes": result["modes"],
                      "transform_fingerprint": result["transform_fingerprint"]}), flush=True)


if __name__ == "__main__":
    main()

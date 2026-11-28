"""Publish the full GLM Flash MUL1 calibration onto its verified parent revision."""
from __future__ import annotations

import json

from qmla.publish_cli import build_parser as shared_parser, publish_main
from qmla.sidecar import QMLA_SCHEMA, read_sidecar_manifest
from .qmla_run import BASE_REPO, BASE_REVISION, MODES, TARGET_REPO, TARGET_REVISION


def build_parser():
    return shared_parser(__doc__, TARGET_REPO, mc_only=True)


def verify_release(args):
    if (args.repo_id, args.source_revision) != (TARGET_REPO, TARGET_REVISION):
        raise ValueError("GLM Flash publication must target the original MUL1 repository and pinned parent")
    manifest = read_sidecar_manifest(args.sidecar_dir)
    if (manifest["source"].get("base_repo"), manifest["source"].get("base_revision")) != (BASE_REPO, BASE_REVISION):
        raise ValueError("unexpected MC base checkpoint provenance")
    forward = manifest["source"].get("forward", {})
    if (forward.get("repo"), forward.get("revision"), forward.get("view"), forward.get("routed_experts")) != (
            TARGET_REPO, TARGET_REVISION, "original", "exl3-k4-mul1") or not all(
            isinstance(forward.get(k), str) and len(forward[k]) == 64
            for k in ("index_sha256", "view_fingerprint", "shared_inventory_sha256")):
        raise ValueError("sidecar must identify the original-view MC forward, index and shared K4 experts")
    layers = list(range(3, 44, 4))
    prov = manifest["calibration_provenance"]
    if (manifest["schema"] != QMLA_SCHEMA
            or [g["layer_id"] for g in manifest["geometry"]] != layers
            or set(manifest["supported_modes"]) != set(MODES)
            or manifest["artifact_status"] != "calibrated_unqualified" or manifest["profiles"]["rope"] is not None
            or manifest["protection_policy"] != "sink4_local128" or prov["steps_per_path"] != 300
            or [prov[f"{s}_sequences"] for s in ("train", "val", "test")] != [128, 16, 16]):
        raise ValueError("publication requires a cuteafd-qmla-v3 sidecar with all 11 MLA layers, all content modes and the full protected calibration")
    report = json.loads((args.sidecar_dir.parent / "qmla-validation-report.json").read_text())
    if (report["validation"]["status"] != "passed"
            or report["validation"]["tensors_sha256"] != manifest["tensors_sha256"]
            or report["long_context_eval"]["transform_fingerprint"] != manifest["transform_fingerprint"]
            or report["long_context_eval"]["layers_evaluated"] != layers
            or set(report["long_context_eval"]["combinations"]) != set(MODES)):
        raise ValueError("full validated cuteafd-qmla-v3 sidecar and matching long evaluation are required")


def main(argv=None):
    return publish_main(build_parser(), "glm5_flash", argv, verify=verify_release)


if __name__ == "__main__":
    raise SystemExit(main())

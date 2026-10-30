"""Read-only admission for the bounded Step-5 pilot, quantizer and publisher."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import shutil
import socket
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "quantization"))
if __package__ in (None, ""):
    __package__ = "step5.core_k6"
from kimi_k3.uniform_k2.storage import sha256_file
from kimi_k3.core_k6.container import IMAGE

CHECKS = ("sliding_boundary_byte_exact", "sparse_boundary_byte_exact",
          "serialized_projection_replay", "mtp_three_stage_alignment_byte_exact",
          "mtp_serialized_k4_byte_exact", "production_layer_transaction_byte_exact")


def require_local_source():
    if socket.gethostname().lower() != "maxq":
        raise ValueError("Step-5 quantization is authorized only on maxq")
    result = subprocess.check_output(["nest", "where", "hf:stepfun-ai/Step-5-Preview-BF16"],
                                     text=True, timeout=60)
    if not any(line.split() == ["maxq", "ready"] for line in result.splitlines()):
        raise ValueError("the entire BF16 source must be local in Sparknest on maxq")


def verify_fork():
    fork = REPO / "third_party/gptqmodel"
    for directory, paths in ((REPO, ["quantization/step5"]), (fork, [])):
        if subprocess.check_output(["git", "-C", str(directory), "status", "--porcelain", "--", *paths],
                                   text=True).strip():
            raise ValueError("uncommitted Step-5 production source")
    actual = subprocess.check_output(["git", "-C", str(fork), "rev-parse", "HEAD"], text=True).strip()
    lock = json.loads((REPO / "third_party/gptqmodel.lock.json").read_text())
    digest = runpy.run_path(str(REPO / "scripts/build/verify-sparkinfer-source.py"))["source_tree_sha256"](fork)
    if actual != lock["revision"] or digest != lock["source_tree_sha256"]:
        raise ValueError("GPTQModel pin or source-tree lock differs")
    return actual


def qualified(report_path, fork_commit):
    from .pilot import validate_qualification, validate_report_code
    report = validate_qualification(json.loads(Path(report_path).read_text()))
    identity = report["identity"]
    if (report.get("status") != "passed" or identity.get("gptqmodel_commit") != fork_commit
            or any(report.get("native_checks", {}).get(name) is not True for name in CHECKS)
            or identity.get("layers") != [1, 4, 7]):
        raise ValueError("Step-5 native/reconstructed pilot qualification is incomplete")
    code = identity.get("code", {})
    if not code:
        raise ValueError("pilot has no immutable numerical source identity")
    validate_report_code(report, REPO)
    root=Path(report_path).parent
    for name in ("finalization-audit.json", "recovery-retired.json"):
        if not (root/name).is_file():
            raise ValueError("pilot evidence audit/retirement marker missing")
        marker=json.loads((root/name).read_text())
        if marker.get("status")!="passed" or marker.get("identity")!=identity:
            raise ValueError("pilot audit/retirement identity differs")
    retired=json.loads((root/"recovery-retired.json").read_text())
    if retired.get("finalization_audit_sha256")!=sha256_file(root/"finalization-audit.json"):
        raise ValueError("retired pilot audit seal differs")
    return report


def audited_source(path):
    from .recipe import BASE_REPO, BASE_REVISION
    report = json.loads(Path(path).read_text())
    if (report.get("status") != "passed" or report.get("repo_id") != BASE_REPO
            or report.get("revision") != BASE_REVISION or len(report.get("files", {})) != 48):
        raise ValueError("complete pinned source checksum audit is required")
    for name in ("config.json", "model.safetensors.index.json"):
        if sha256_file(Path(report["source"]) / name) != report["files"][name]["sha256"]:
            raise ValueError("source metadata changed after checksum audit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-only", action="store_true")
    parser.add_argument("--publication-only", action="store_true")
    parser.add_argument("--qualification-report", type=Path, default=Path.home() /
        ".cache/cuteafd/quantization/step5-k4-core-k6/pilot/qualification.json")
    parser.add_argument("--source-audit", type=Path, default=Path.home() /
        ".cache/cuteafd/builds/step5-core-k6/source-audit.json")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    if args.publication_only and args.qualification_only:
        parser.error("publication and pilot are separate modes")
    if socket.gethostname().lower() != "maxq":
        raise ValueError("Step-5 production placement is maxq only")
    commit = verify_fork()
    if not args.publication_only:
        require_local_source()
        if not args.qualification_only:
            qualified(args.qualification_report, commit)
            audited_source(args.source_audit)
        smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
        devices = subprocess.check_output([smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                                          text=True, timeout=30).strip().splitlines()
        if (len(devices) != 1 or "RTX PRO 6000" not in devices[0]
                or int(devices[0].rsplit(",", 1)[1].strip()) < 96000):
            raise ValueError("expected exactly one local 96GB RTX PRO 6000")
        subprocess.run(["docker", "image", "inspect", IMAGE], check=True,
                       stdout=subprocess.DEVNULL, timeout=30)
    runner = Path(__file__).with_name("pilot.py" if args.qualification_only else "run.py")
    subprocess.run([str(REPO / ".venv/bin/python"), str(runner), *arguments, "--preflight"],
                   check=True, timeout=540)
    print(json.dumps(dict(admission="passed", host="maxq", fork_commit=commit,
                          pilot=args.qualification_only, publication=args.publication_only)), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read-only repository, numerical, local-source and hardware admission."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import socket
import subprocess
import sys

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/"quantization"))
from kimi_k3.uniform_k2.storage import atomic_json,sha256_file,verify_receipt
from kimi_k3.core_k6.container import IMAGE


def qualified(fork_commit,*,audit=False):
    root=Path.home()/".cache/cuteafd/builds/glm53-core-k6"
    report=json.loads((root/"native-probe.json").read_text())
    identity=report["identity"]
    if (report.get("status")!="passed" or identity.get("fork_commit")!=fork_commit
            or report.get("quantized_target_layers")!=[0,1,2,3]
            or report.get("native_target_boundaries_byte_exact") is not True
            or report.get("mtp_native_boundary_byte_exact") is not True
            or report.get("mtp_shared_head_norm_byte_exact") is not True
            or report.get("natural_sparse_witness_tokens")!=2205 or report.get("mtp_valid_rows")!=2204):
        raise ValueError("current GLM producer has not passed target GPU qualification")
    for name,digest in identity["code"].items():
        # The report retains the exact test harness that produced it. A later
        # reporting-only harness fix is not a changed production computation.
        if name=="probe.py":
            continue
        if sha256_file(Path(__file__).with_name(name))!=digest:
            raise ValueError(f"producer changed after native qualification: {name}")
    marker=root/"native-probe-audit.json"
    report_sha=sha256_file(root/"native-probe.json")
    if audit:
        probe=root/"native-probe"
        for layer in range(4):
            receipt=json.loads((probe/"partial-export"/f"layer-{layer:03d}-receipt.json").read_text())
            if receipt["identity"]!=identity or receipt["layer"]!=layer:
                raise ValueError("native probe export identity differs")
            verify_receipt(probe/"partial-export"/receipt["filename"],receipt,checksum=True)
        evidence=list((probe/"evidence").rglob("*.json"))
        if len(evidence)!=798:
            raise ValueError("native probe projection evidence is incomplete")
        for path in evidence:
            row=json.loads(path.read_text())
            if row["metrics"].get("quantizer_path")!="hessian_ldlq" or row["metrics"].get("hessian_metric_status")!="ok":
                raise ValueError("native probe used an invalid quantizer path")
        observed=set()
        for line in (probe/"events.jsonl").read_text().splitlines():
            event=json.loads(line)
            if event["event"]=="native_quantized_boundary_exact" and event.get("tokens")==2205:
                observed.add(event["layer"])
        if observed!={0,1,2,3}:
            raise ValueError("missing actual byte-exact native boundary observations")
        atomic_json(marker,dict(schema="glm-native-completed-proof-audit-v1",report_sha256=report_sha,
            fork_commit=fork_commit,projection_evidence=798,layers=[0,1,2,3],status="passed",
            note="independent completed native proof, export and evidence audit; original report identity retained"))
    else:
        completed=json.loads(marker.read_text())
        if (completed.get("status")!="passed" or completed.get("report_sha256")!=report_sha
                or completed.get("fork_commit")!=fork_commit):
            raise ValueError("native proof completion audit differs")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication-only",action="store_true")
    parser.add_argument("--audit-native",action="store_true")
    parser.add_argument("arguments",nargs=argparse.REMAINDER)
    args=parser.parse_args()
    arguments=args.arguments[1:] if args.arguments[:1]==["--"] else args.arguments
    if socket.gethostname().lower()!="oliver":
        raise ValueError("GLM production placement is Oliver only")
    for directory,paths in ((REPO,["quantization/glm5","quantization/kimi_k3/prepare_corpus.py"]),
                            (REPO/"third_party/gptqmodel",[])):
        if subprocess.check_output(["git","-C",str(directory),"status","--porcelain","--",*paths],text=True).strip():
            raise ValueError("uncommitted production code")
    fork=REPO/"third_party/gptqmodel"
    actual=subprocess.check_output(["git","-C",str(fork),"rev-parse","HEAD"],text=True).strip()
    lock=json.loads((REPO/"third_party/gptqmodel.lock.json").read_text())
    digest=runpy.run_path(str(REPO/"scripts/build/verify-sparkinfer-source.py"))["source_tree_sha256"](fork)
    if actual!=lock["revision"] or digest!=lock["source_tree_sha256"]:
        raise ValueError("GPTQModel pin or source lock differs")
    qualified(actual,audit=args.audit_native)
    if args.audit_native:
        print(json.dumps(dict(native_audit="passed",fork_commit=actual)),flush=True)
        return
    if not args.publication_only:
        where=subprocess.check_output(["nest","where","hf:zai-org/GLM-5.3-BF16"],text=True,timeout=60)
        if not any(line.split()==["oliver","ready"] for line in where.splitlines()):
            raise ValueError("the complete BF16 source must be sealed locally on Oliver")
        devices=subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits"],text=True,timeout=30).strip().splitlines()
        if len(devices)!=1 or "RTX PRO 6000" not in devices[0] or int(devices[0].rsplit(",",1)[1].strip())<96000:
            raise ValueError("expected one local 96GB RTX PRO 6000")
        subprocess.run(["docker","image","inspect",IMAGE],check=True,stdout=subprocess.DEVNULL,timeout=30)
    subprocess.run([str(REPO/".venv/bin/python"),str(Path(__file__).with_name("run.py")),
                    *arguments,"--preflight"],check=True,timeout=240)
    print(json.dumps(dict(admission="passed",host="oliver",fork_commit=actual,image=IMAGE)),flush=True)


if __name__=="__main__":
    main()

"""Publish the sealed MUL1 MC, adopt its HF cache, place it and retire task payloads."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from hub_requirements import require_qualified_hub
from hy4_flash import handoff
from hy4_flash.publish import _check_numerical_completion, validate_mc
from inkling2 import publish as hub
from kimi_k3.uniform_k2.storage import atomic_json
from .corpus import SOURCE_REPO, SOURCE_REVISION

TARGET_REPO = "wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K4-MC-MUL1"


def validate_package(output, state, validation):
    validate_mc(output, state, validation, repo=SOURCE_REPO, revision=SOURCE_REVISION)
    if not {".gitattributes", "LICENSE"} <= set(state["new_files"]):
        raise ValueError("seal the regular .gitattributes and LICENSE before publication")


TARGET = hub.PublicationTarget(TARGET_REPO, "glm5_flash", "Publish calibrated GLM Flash K4 MUL1 MC", validate_package)
LAYOUT = handoff.Placement(TARGET, "glmf-mul1", adoption_host="cassowary", keep_ref="keep/glmf-k4-mc-mul1")


def finish(root, output, token_stream):
    from huggingface_hub import HfApi

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("the publication unit requires CUDA_VISIBLE_DEVICES empty")
    root, output = root.resolve(), output.absolute()
    require_qualified_hub()
    token = None
    complete = hub.load(hub.regular(root / "numerical-complete.json"))
    if not (root / "publication-complete.json").is_file():
        _check_numerical_completion(root, output, TARGET, SOURCE_REVISION)
        if complete.get("original_discovery", {}).get("status") != "passed":
            raise ValueError("missing plain original-view loader receipt")
        if token_stream.isatty():
            raise ValueError("provide the write token on stdin")
        token = token_stream.readline(1024).strip()
        if not re.fullmatch(r"hf_[A-Za-z0-9]+", token):
            raise ValueError("missing or malformed Hub write token on stdin")
    atomic_json(root / "publication-runtime.json", {
        "hub": importlib.metadata.version("huggingface-hub"), "xet": importlib.metadata.version("hf-xet")})
    published = hub.publish(output, root, api=HfApi(token=token or False), target=TARGET)
    placed = handoff.handoff(root, output, root / "handoff-stage", layout=LAYOUT)
    retired = handoff.retire(root, output, intermediates=complete["task_intermediates"], layout=LAYOUT)
    return dict(repo_id=published["repo_id"], revision=published["revision"], status=retired["status"],
                targets=placed["targets"], receipt_archive=retired["receipt_archive"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, choices=[TARGET_REPO])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(finish(args.root, args.output, sys.stdin)), flush=True)
        return
    # The transient service owns the continuation, while the caller provides the
    # token only on its inherited stdin. All progress is durably receipted.
    repo = Path(__file__).resolve().parents[3]
    subprocess.run(["systemd-run", "--user", "--pipe", "--wait", "--collect",
                    "--expand-environment=no", "--setenv=PATH", "--setenv=HOME",
                    "--unit=glmf-mul1-publish", "--working-directory=" + str(repo),
                    "--setenv=CUDA_VISIBLE_DEVICES=", "--setenv=HF_XET_FIXED_UPLOAD_CONCURRENCY=32",
                    "--setenv=PYTHONPATH=" + str(repo / "quantization"),
                    sys.executable, "-m", "glm5_flash.mul1_mc.publish", "--worker", "--repo", args.repo,
                    "--root", str(args.root.resolve()), "--output", str(args.output.absolute())], check=True)


if __name__ == "__main__":
    main()

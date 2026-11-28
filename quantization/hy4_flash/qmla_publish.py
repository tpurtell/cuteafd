"""Publish a validated Hy4-flash QMLA sidecar onto a new revision of the same HF repo.

Thin family CLI wrapper over ``quantization/qmla/publish.py``:
- No new repository.
- Zero weight shard re-uploads (preserves every existing shard/index/view/asset).
- HTTP LFS only (``HF_HUB_DISABLE_XET=1``).
- Reads HF write token from stdin only when not ``--dry-run``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hub_lfs
from qmla.publish import plan_incremental_revision, publish_incremental_revision

from . import recipe as hy4_recipe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=hy4_recipe.TARGET_REPO, help="Existing HF repo to add a new revision to")
    parser.add_argument("--source-revision", required=True, help="Immutable 40-char parent commit SHA")
    parser.add_argument("--sidecar-dir", type=Path, required=True, help="Directory holding mla_calibration.{json,safetensors}")
    parser.add_argument("--state-root", type=Path, required=True, help="Task state directory for upload lock/receipts")
    parser.add_argument("--validation-report", type=Path, default=None)
    parser.add_argument("--readme", type=Path, default=None)
    parser.add_argument("--update-mc-manifest", action="store_true", help="Attach advisory qmla pointer in cuteafd-mc.json")
    parser.add_argument("--dry-run", action="store_true", help="Validate plan locally without committing to HF")
    hub_lfs.add_arguments(parser, batches=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from huggingface_hub import HfApi, hf_hub_download

    parser = build_parser()
    args = parser.parse_args(argv)
    token = None
    if not args.dry_run:
        if sys.stdin.isatty():
            parser.error("provide the HF write token through stdin (or pass --dry-run)")
        token = sys.stdin.readline(1024).strip()
        if not re.fullmatch(r"hf_[A-Za-z0-9]+", token):
            parser.error("HF write token on stdin is missing or malformed")

    api = HfApi(token=token or False)
    remote = hub_lfs.remote_tree(api, args.repo_id, args.source_revision)
    existing_mc = None
    if args.update_mc_manifest and "cuteafd-mc.json" in remote:
        mc_path = Path(
            hf_hub_download(
                args.repo_id, "cuteafd-mc.json", revision=args.source_revision, token=token or False
            )
        )
        existing_mc = json.loads(mc_path.read_text(encoding="utf-8"))

    stage_dir = args.state_root.resolve() / "stage"
    stage_dir.mkdir(parents=True, exist_ok=True)
    plan = plan_incremental_revision(
        args.repo_id,
        args.source_revision,
        args.sidecar_dir,
        remote_tree=remote,
        existing_mc_manifest=existing_mc,
        stage_dir=stage_dir,
        validation_report_path=args.validation_report,
        readme_path=args.readme,
        expected_source={"repo": args.repo_id, "revision": args.source_revision, "family": "hy4_flash"},
    )
    receipt = publish_incremental_revision(
        plan,
        api=api,
        state_root=args.state_root,
        dry_run=args.dry_run,
        options=hub_lfs.options_from(args),
    )
    print(json.dumps(receipt), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

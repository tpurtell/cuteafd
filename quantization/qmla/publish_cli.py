"""Command-line interface for a parent-guarded incremental QMLA commit."""
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


def build_parser(description: str, repo: str, *, mc_only: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--repo-id", default=repo, help="Existing HF repo to add a new revision to")
    parser.add_argument("--source-revision", required=True, help="Immutable 40-char parent commit SHA")
    parser.add_argument("--sidecar-dir", type=Path, required=True, help="Directory holding mla_calibration.safetensors")
    parser.add_argument("--state-root", type=Path, required=True, help="Task state directory for upload lock/receipts")
    if not mc_only:
        parser.add_argument("--validation-report", type=Path, default=None)
        parser.add_argument("--readme", type=Path, default=None)
    parser.set_defaults(readme=None, validation_report=None)
    parser.add_argument("--update-mc-manifest", default=mc_only, action="store_true", help="Embed full qmla section in cuteafd-mc.json (MC repos)")
    parser.add_argument("--dry-run", action="store_true", help="Validate plan locally without committing to HF")
    hub_lfs.add_arguments(parser, batches=False)
    return parser


def publish_main(parser: argparse.ArgumentParser, family: str, argv: Sequence[str] | None = None, *,
                 verify=None) -> int:
    from huggingface_hub import HfApi, hf_hub_download

    args = parser.parse_args(argv)
    if verify is not None:
        verify(args)
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
    if args.update_mc_manifest:
        if "cuteafd-mc.json" not in remote:
            parser.error("--update-mc-manifest requested, but remote repo has no cuteafd-mc.json")
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
        expected_source={"repo": args.repo_id, "revision": args.source_revision, "family": family},
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

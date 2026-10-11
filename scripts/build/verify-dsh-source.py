#!/usr/bin/env python3
"""Verify the pinned DSH fork, including metadata-free build copies."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

# Share the tree digest and Git checks rather than introduce a second algorithm.
_spec = importlib.util.spec_from_file_location(
    "source_lock", Path(__file__).with_name("verify-sparkinfer-source.py")
)
_lock = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lock)


def verify(source: Path, lock_path: Path) -> dict:
    source = source.resolve()
    value = _lock._load_lock(lock_path)
    for name in ("LICENSE", "package.json", "pnpm-lock.yaml"):
        if not (source / name).is_file():
            raise _lock.VerificationError(f"DSH source is incomplete: missing {name}")
    if json.loads((source / "package.json").read_text())["name"] != "@deepseek-ai/dsh-root":
        raise _lock.VerificationError("not a DSH source tree")
    if _lock.source_tree_sha256(source) != value["source_tree_sha256"]:
        raise _lock.VerificationError("DSH source digest differs from the lock")
    if (source / ".git").exists():
        if _lock._run_git(source, "rev-parse", "HEAD") != value["revision"]:
            raise _lock.VerificationError("DSH revision differs from the lock")
        if _lock._run_git(source, "status", "--porcelain", "--untracked-files=all"):
            raise _lock.VerificationError("DSH checkout has local source changes")
        if _lock._normalized_repository(_lock._run_git(source, "remote", "get-url", "origin")) != _lock._normalized_repository(value["repository"]):
            raise _lock.VerificationError("DSH origin differs from the lock")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--print-tree-sha256", action="store_true")
    args = parser.parse_args()
    if args.print_tree_sha256:
        print(_lock.source_tree_sha256(args.source))
    else:
        if args.lock is None:
            parser.error("--lock is required for verification")
        value = verify(args.source, args.lock)
        print(f"verified DSH revision={value['revision']} source_tree_sha256={value['source_tree_sha256']}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, _lock.VerificationError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)

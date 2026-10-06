#!/usr/bin/env python3
"""Verify pinned Transformers sources, including metadata-free build freezes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

IGNORED_NAMES = {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__", "build", "dist"}
LOCK_FIELDS = {"schema", "repository", "revision", "source_tree_sha256"}
REQUIRED = (
    "LICENSE", "pyproject.toml", "src/transformers/__init__.py",
    "src/transformers/models/glm5_next/modeling_glm5_next.py",
    "src/transformers/models/glm5_next/image_processing_pil_glm5_next.py",
)


class VerificationError(RuntimeError):
    pass


def source_tree_sha256(source: Path) -> str:
    source = source.resolve()
    digest = hashlib.sha256()
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part in IGNORED_NAMES for part in relative.parts) or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            target = os.readlink(path)
            resolved = (path.parent / target).resolve()
            if not resolved.is_relative_to(source) or not resolved.is_file():
                raise VerificationError(f"Transformers source symlink escapes or is broken: {relative}")
            mode, content = "120000", target.encode()
        elif path.is_file():
            mode = "100755" if path.stat().st_mode & 0o111 else "100644"
            content = path.read_bytes()
        else:
            continue
        digest.update(mode.encode() + b" " + relative.as_posix().encode() + b"\0" + content + b"\0")
    return digest.hexdigest()


def run_git(source: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={source}", "-C", str(source), *args],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise VerificationError(f"Transformers git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def normalized_repository(value: str) -> str:
    match = re.fullmatch(r"git@([^:]+):(.+)", value)
    if match:
        value = f"https://{match[1]}/{match[2]}"
    return value.removesuffix(".git").rstrip("/")


def verify(source: Path, lock_path: Path) -> dict:
    source = source.resolve()
    try:
        lock = json.loads(lock_path.read_text())
    except (OSError, ValueError) as error:
        raise VerificationError(f"cannot read Transformers lock {lock_path}: {error}") from error
    if not isinstance(lock, dict) or set(lock) != LOCK_FIELDS or lock["schema"] != 1:
        raise VerificationError("invalid Transformers source lock fields/schema")
    for field, pattern in (("revision", r"[0-9a-f]{40}"), ("source_tree_sha256", r"[0-9a-f]{64}")):
        if not isinstance(lock[field], str) or not re.fullmatch(pattern, lock[field]):
            raise VerificationError(f"invalid Transformers {field}")
    if not isinstance(lock["repository"], str) or not lock["repository"].startswith("https://"):
        raise VerificationError("Transformers repository must be an HTTPS URL")
    missing = [name for name in REQUIRED if not (source / name).is_file()]
    if missing:
        raise VerificationError("Transformers source is incomplete; initialize the pinned submodule: " + ", ".join(missing))
    actual = source_tree_sha256(source)
    if actual != lock["source_tree_sha256"]:
        raise VerificationError(f"Transformers source content does not match lock: expected {lock['source_tree_sha256']}, found {actual}")
    if (source / ".git").exists():
        if run_git(source, "rev-parse", "HEAD") != lock["revision"]:
            raise VerificationError("Transformers checkout revision does not match lock")
        if run_git(source, "status", "--porcelain", "--untracked-files=all"):
            raise VerificationError("Transformers checkout has tracked or untracked changes")
        if normalized_repository(run_git(source, "remote", "get-url", "origin")) != normalized_repository(lock["repository"]):
            raise VerificationError("Transformers checkout repository does not match lock")
    return lock


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--print-source-digest", action="store_true")
    parser.add_argument("--print-revision", action="store_true")
    args = parser.parse_args()
    lock = verify(args.source, args.lock)
    if args.print_source_digest:
        print(lock["source_tree_sha256"])
    elif args.print_revision:
        print(lock["revision"])
    else:
        print(f"verified Transformers revision={lock['revision']} source_tree_sha256={lock['source_tree_sha256']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (VerificationError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)

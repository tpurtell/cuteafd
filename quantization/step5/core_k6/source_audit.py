"""Check every pinned source file against Hub hashes, using bounded reads."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "quantization"))
from kimi_k3.uniform_k2.storage import atomic_json, sha256_file
if __package__ in (None, ""):
    __package__="step5.core_k6"
from .recipe import BASE_REPO, BASE_REVISION


def hash_file(path, expected):
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if before.st_size != expected["bytes"]:
            raise ValueError(f"source size differs: {path.name}")
        sha256 = hashlib.sha256()
        git_sha1 = hashlib.sha1(f"blob {before.st_size}\0".encode())
        while data := stream.read(8 * 1024 * 1024):
            sha256.update(data)
            if not expected.get("sha256"):
                git_sha1.update(data)
        after = os.fstat(stream.fileno())
    fields = ("st_ino", "st_dev", "st_size", "st_mtime_ns")
    if any(getattr(before, f) != getattr(after, f) for f in fields):
        raise ValueError(f"source changed during audit: {path.name}")
    if expected.get("sha256"):
        matched = sha256.hexdigest() == expected["sha256"]
    else:
        matched = git_sha1.hexdigest() == expected["git_blob_sha1"]
    if not matched:
        raise ValueError(f"source Hub checksum differs: {path.name}")
    return dict(bytes=before.st_size, sha256=sha256.hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--hub-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.snapshot.resolve(strict=True)
    manifest = json.loads(args.hub_manifest.read_text())
    if (source.name != BASE_REVISION or manifest["revision"] != BASE_REVISION
            or manifest["repo_id"] != BASE_REPO):
        raise ValueError("source revision differs from reviewed Step-5 release")
    if set(p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()) != set(manifest["files"]):
        raise ValueError("source file inventory differs from pinned Hub manifest")
    files = sorted(manifest["files"])
    def run(name):
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("unsafe source path")
        return hash_file(source / name, manifest["files"][name])
    records = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for name, record in zip(files, pool.map(run, files), strict=True):
            records[name] = record
            print(json.dumps(dict(event="source_file_verified", name=name, **record)), flush=True)
    atomic_json(args.output, dict(status="passed", source=str(source), repo_id=BASE_REPO,
        revision=BASE_REVISION, hub_manifest_sha256=sha256_file(args.hub_manifest), files=records,
        bytes=sum(r["bytes"] for r in records.values())))
    print(json.dumps(dict(event="source_audit_passed", files=len(records))), flush=True)


if __name__ == "__main__":
    if __package__ in (None, ""):
        __package__ = "step5.core_k6"
    main()

"""Header inventory and pinned Hub metadata checks without rereading 1.5 TiB."""
from __future__ import annotations

from pathlib import Path
import json

from kimi_k3.uniform_k2.storage import atomic_json, file_identity
from .recipe import BASE_REPO, BASE_REVISION, header_source, inventory


def audit_source(snapshot, output, hub_manifest=None):
    source = header_source(snapshot)
    if source.snapshot.name != BASE_REVISION:
        raise ValueError("source revision differs from the reviewed K3.1 release")
    summary = inventory(source)
    files = {}
    if hub_manifest is not None:
        manifest = json.loads(Path(hub_manifest).read_text())
        if manifest.get("repo_id") != BASE_REPO or manifest.get("revision") != BASE_REVISION:
            raise ValueError("Hub metadata manifest names another source")
        expected = manifest["files"]
        actual = {p.relative_to(source.snapshot).as_posix() for p in source.snapshot.rglob("*") if p.is_file()}
        if actual != set(expected):
            raise ValueError("source files differ from pinned Hub metadata manifest")
        for name, metadata in expected.items():
            path = source.snapshot / name
            if Path(name).is_absolute() or ".." in Path(name).parts or path.stat().st_size != metadata["bytes"]:
                raise ValueError("source path or sealed file size differs")
            # SparkNest sealed checksums remain the source byte attestation;
            # this audit does not substitute an unmeasured local hash.
            files[name] = dict(file_identity=file_identity(path), **metadata)
    result = dict(status="passed", repo_id=BASE_REPO, revision=BASE_REVISION,
                  source=str(source.snapshot), inventory=summary, files=files,
                  method="complete reviewed headers plus sealed source metadata; no weight rehash")
    atomic_json(output, result)
    return result

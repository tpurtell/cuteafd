"""Same-repo incremental Hugging Face revision publisher for QMLA sidecars.

Publishes ``mla_calibration.safetensors`` and ``mla_calibration.json`` (plus an
optional advisory ``qmla`` section in ``cuteafd-mc.json`` and optional validation
report / model-card update) onto a **new revision of the same HF repository** at
pinned ``source_revision``.

Guarantees:
- Zero new repositories.
- Zero weight shard re-uploads (``reuploaded_shard_bytes == 0``).
- Every existing shard, index, config, view, tokenizer, and asset at
  ``source_revision`` is preserved byte-for-byte on the new commit.
- HTTP LFS only (``HF_HUB_DISABLE_XET=1``) via ``quantization/hub_lfs.py`` and
  ``quantization/hub_requirements.py``.
- Full offline dry-run support for tests and pre-publish inspection.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import hub_lfs
from hub_requirements import require_qualified_hub

from .capture import _atomic_json
from .sidecar import (
    MANIFEST_FILE,
    TENSORS_FILE,
    attach_qmla_advisory_to_mc,
    validate_sidecar,
)

INCREMENTAL_PLAN_SCHEMA = "cuteafd-qmla-incremental-upload-plan-v1"
ALLOWED_INCREMENTAL_FILES = frozenset({
    MANIFEST_FILE,
    TENSORS_FILE,
    "cuteafd-mc.json",
    "README.md",
    "qmla-validation-report.json",
})


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _file_row(path: Path, name: str) -> dict[str, Any]:
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"incremental upload file missing or symlink: {path}")
    local = hub_lfs.fingerprint(path)
    sha256 = _sha256_file(path)
    return {
        "path": str(path),
        "bytes": int(local["bytes"]),
        "sha256": sha256,
        "git_blob": None if name.endswith(".safetensors") else hub_lfs.git_blob(path),
        "local": local,
    }


def plan_incremental_revision(
    repo_id: str,
    source_revision: str,
    sidecar_dir: Path | str,
    *,
    remote_tree: Mapping[str, Mapping[str, Any]],
    existing_mc_manifest: Mapping[str, Any] | None = None,
    stage_dir: Path | str | None = None,
    validation_report_path: Path | str | None = None,
    readme_path: Path | str | None = None,
    expected_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed same-repo upload plan without modifying any existing shard."""
    if not repo_id or "/" not in repo_id:
        raise ValueError(f"invalid HF repo_id: {repo_id!r}")
    if not source_revision or len(source_revision) != 40 or not all(c in "0123456789abcdef" for c in source_revision):
        raise ValueError(f"source_revision must be a 40-char immutable git commit SHA, got {source_revision!r}")

    sidecar_dir = Path(sidecar_dir).resolve()
    validation = validate_sidecar(sidecar_dir, expected_source=expected_source, strict=True)

    # Identify all preserved weight shards and metadata at source_revision.
    preserved_shards = sorted(
        name for name in remote_tree if name.endswith(".safetensors") and name != TENSORS_FILE
    )
    if not preserved_shards:
        raise ValueError(f"remote tree at {repo_id}@{source_revision} contains no model weight shards")

    additions: dict[str, dict[str, Any]] = {
        TENSORS_FILE: _file_row(sidecar_dir / TENSORS_FILE, TENSORS_FILE),
        MANIFEST_FILE: _file_row(sidecar_dir / MANIFEST_FILE, MANIFEST_FILE),
    }
    if additions[TENSORS_FILE]["sha256"] != validation["tensors_sha256"]:
        raise ValueError("sidecar tensors changed after validation")
    if additions[MANIFEST_FILE]["sha256"] != validation["manifest_sha256"]:
        raise ValueError("sidecar manifest changed after validation")

    if existing_mc_manifest is not None:
        if "cuteafd-mc.json" not in remote_tree:
            raise ValueError("existing_mc_manifest provided, but remote repo has no cuteafd-mc.json")
        if stage_dir is None:
            raise ValueError("stage_dir is required when updating cuteafd-mc.json")
        updated_mc = attach_qmla_advisory_to_mc(existing_mc_manifest, sidecar_dir)
        # Verify invariant MC fields did not move.
        for invariant in (
            "schema",
            "source",
            "original_core_dtype",
            "shared_inventory_sha256",
            "variants",
            "required_features",
            "exl3_codebook",
            "engram_fp8",
        ):
            if existing_mc_manifest.get(invariant) != updated_mc.get(invariant):
                raise ValueError(f"incremental QMLA update must not alter MC field {invariant!r}")
        staged_mc = Path(stage_dir).resolve() / "cuteafd-mc.json"
        _atomic_json(staged_mc, updated_mc)
        additions["cuteafd-mc.json"] = _file_row(staged_mc, "cuteafd-mc.json")

    if validation_report_path is not None:
        additions["qmla-validation-report.json"] = _file_row(
            Path(validation_report_path), "qmla-validation-report.json"
        )
    if readme_path is not None:
        additions["README.md"] = _file_row(Path(readme_path), "README.md")

    # Hard guard: never touch any existing weight shard or disallowed path.
    disallowed = set(additions) - ALLOWED_INCREMENTAL_FILES
    if disallowed:
        raise ValueError(f"incremental QMLA plan touches disallowed files: {sorted(disallowed)}")
    if set(additions) & set(preserved_shards):
        raise ValueError("incremental QMLA plan must never overwrite existing weight shards")

    preserved_tree = {
        name: {"bytes": int(row["bytes"]), "blob": str(row["blob"])}
        for name, row in remote_tree.items()
        if name not in additions
    }

    return {
        "schema": INCREMENTAL_PLAN_SCHEMA,
        "repo_id": repo_id,
        "source_revision": source_revision,
        "artifact_status": validation["artifact_status"],
        "transform_fingerprint": validation["transform_fingerprint"],
        "supported_modes": validation["supported_modes"],
        "additions": additions,
        "upload_bytes": sum(row["bytes"] for row in additions.values()),
        "preserved_files_count": len(preserved_tree),
        "preserved_shards_count": len(preserved_shards),
        "preserved_shard_bytes": sum(int(remote_tree[s]["bytes"]) for s in preserved_shards),
        "reuploaded_shard_bytes": 0,
        "preserved_tree": preserved_tree,
    }


def publish_incremental_revision(
    plan: Mapping[str, Any],
    *,
    api: Any,
    state_root: Path | str,
    dry_run: bool = False,
    options: hub_lfs.UploadOptions | None = None,
    commit_message: str | None = None,
) -> dict[str, Any]:
    """Commit the QMLA sidecar onto a new revision of ``plan['repo_id']``.

    With ``dry_run=True``, validates source fingerprints and returns the complete
    audit receipt without touching Hugging Face.
    """
    if plan.get("schema") != INCREMENTAL_PLAN_SCHEMA:
        raise ValueError(f"unsupported incremental upload plan schema: {plan.get('schema')!r}")
    if int(plan.get("reuploaded_shard_bytes", -1)) != 0:
        raise ValueError("refuse plan with non-zero reuploaded_shard_bytes")

    additions = plan["additions"]
    hub_lfs.check_fingerprints(additions, sorted(additions))

    state_root = Path(state_root).resolve()
    state_root.mkdir(parents=True, exist_ok=True)
    if dry_run:
        receipt = {
            "status": "dry_run",
            "repo_id": plan["repo_id"],
            "source_revision": plan["source_revision"],
            "new_revision": None,
            "artifact_status": plan["artifact_status"],
            "transform_fingerprint": plan["transform_fingerprint"],
            "uploaded_files": sorted(additions),
            "upload_bytes": int(plan["upload_bytes"]),
            "preserved_shards_count": int(plan["preserved_shards_count"]),
            "reuploaded_shard_bytes": 0,
        }
        _atomic_json(state_root / "incremental-publish-dry-run.json", receipt)
        return receipt

    require_qualified_hub()
    hub_lfs.use_http_lfs()
    options = options or hub_lfs.UploadOptions()

    with (state_root / "upload.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        repo_id = str(plan["repo_id"])
        parent = str(plan["source_revision"])
        info = api.repo_info(repo_id, repo_type="model")
        if info.sha != parent:
            raise ValueError(
                f"repo {repo_id} head moved from pinned {parent} to {info.sha}; refuse unguarded incremental commit"
            )
        remote_before = hub_lfs.remote_tree(api, repo_id, parent)
        for name, expected_row in plan["preserved_tree"].items():
            if remote_before.get(name) != expected_row:
                raise ValueError(f"remote preserved file {name} differs from planned tree")

        ordered = hub_lfs.upload_order(additions)
        operations = [hub_lfs.receipt_operation(name, additions[name]) for name in ordered]
        msg = commit_message or (
            f"Add QMLA calibration sidecar ({plan['artifact_status']}, "
            f"modes={','.join(plan['supported_modes'])})"
        )
        commit_info = api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            operations=operations,
            commit_message=msg,
            parent_commit=parent,
            num_threads=options.workers,
        )
        new_sha = getattr(commit_info, "oid", None) or api.repo_info(repo_id, repo_type="model").sha
        hub_lfs.check_fingerprints(additions, ordered)

        remote_after = hub_lfs.remote_tree(api, repo_id, new_sha)
        # Verify every preserved file kept identical bytes & blob, and every added file matches.
        for name, expected_row in plan["preserved_tree"].items():
            if name == ".gitattributes":
                continue
            if remote_after.get(name) != expected_row:
                raise ValueError(f"preserved file {name} changed in published revision {new_sha}")
        for name, row in additions.items():
            if name not in remote_after:
                raise ValueError(f"added QMLA file {name} missing from published revision {new_sha}")
            hub_lfs.resolved(remote_after[name], row, name)

        receipt = {
            "status": "published",
            "repo_id": repo_id,
            "source_revision": parent,
            "new_revision": new_sha,
            "artifact_status": plan["artifact_status"],
            "transform_fingerprint": plan["transform_fingerprint"],
            "supported_modes": list(plan["supported_modes"]),
            "uploaded_files": ordered,
            "upload_bytes": int(plan["upload_bytes"]),
            "preserved_shards_count": int(plan["preserved_shards_count"]),
            "reuploaded_shard_bytes": 0,
        }
        _atomic_json(state_root / "incremental-publish-complete.json", receipt)
        return receipt

"""Resumable verified HTTP LFS batches and a durable local/remote receipt.

Call only after structural export validation. Failures propagate for explicit
recovery; this module adds no retry loop. SDK transport retries remain SDK-owned.
"""
import fcntl
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import RepositoryNotFoundError
from huggingface_hub.hf_api import RepoFile

import hub_lfs
from hub_requirements import require_qualified_hub
from export_assets import asset_target
from write_export import _fingerprint, _publish_json


def _read(path):
    return json.loads(path.read_text()) if path.exists() else None


def _remote(api, repo_id, commit):
    return {entry.path: dict(bytes=entry.size, blob=entry.lfs.sha256 if entry.lfs else entry.blob_id)
            for entry in api.list_repo_tree(repo_id, revision=commit, recursive=True, repo_type="model")
            if isinstance(entry, RepoFile)}


def _restored_operation(name, path, record):
    # Version-pinned SDK: construction reads only the first 512 bytes. Restoring
    # its documented mutated preupload state prevents retransmission on resume.
    operation = CommitOperationAdd(name, path)
    operation._upload_mode = record["mode"]
    operation._should_ignore = False
    if record["mode"] == "lfs":
        operation.upload_info.sha256 = bytes.fromhex(record["blob"])
        operation._is_uploaded = True
    return operation


def upload_artifact(output, state_root, repo_id, *, resume=False, api=None, progress=None, options=None):
    """Verified HTTP LFS batch commits on a guarded parent; resume skips committed files."""
    require_qualified_hub()
    from huggingface_hub.utils import validate_repo_id
    validate_repo_id(repo_id)
    output, state_root = Path(output).resolve(strict=True), Path(state_root).resolve()
    if output == state_root or output.is_relative_to(state_root) or state_root.is_relative_to(output):
        raise ValueError("upload state must be separate from artifact")
    progress = progress or (lambda event: None)
    options = options or hub_lfs.UploadOptions()
    api = api or HfApi()
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() or path.is_symlink():
            name = str(path.relative_to(output))
            asset_target(output, name)
            files[name] = _fingerprint(path)
    if not {"README.md", ".gitattributes", "config.json", "model.safetensors.index.json"}.issubset(files):
        raise ValueError("upload requires complete model metadata and attributes")
    state_root.mkdir(parents=True, exist_ok=True)
    with (state_root / "upload.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = state_root / "upload-plan.json"
        if plan_path.exists() and not resume:
            raise ValueError("existing upload requires explicit resume")
        _publish_json(plan_path, dict(schema="cuteafd-upload-plan-v1", repo_id=repo_id,
                                     output=str(output), files=files))
        receipt_path = state_root / "upload-complete.json"
        receipt = _read(receipt_path)
        if receipt is not None:
            expected = {name: {key: record[key] for key in ("bytes", "blob")}
                        for name, record in receipt["files"].items()}
            if (receipt["repo_id"] != repo_id or set(receipt["files"]) != set(files)
                    or any(receipt["files"][name]["local"] != local for name, local in files.items())
                    or _remote(api, repo_id, receipt["commit"]) != expected):
                raise ValueError("completed upload receipt differs")
            return receipt
        parent_path = state_root / "upload-parent.json"
        parent = _read(parent_path)
        if parent is None:
            try:
                info = api.repo_info(repo_id, repo_type="model")
            except RepositoryNotFoundError:
                api.create_repo(repo_id, repo_type="model", private=False, exist_ok=False)
                info = api.repo_info(repo_id, repo_type="model")
            if set(_remote(api, repo_id, info.sha)) - {".gitattributes"}:
                raise ValueError("refusing to overwrite an existing populated repository")
            parent = dict(commit=info.sha)
            _publish_json(parent_path, parent)
        # The export writer records no payload SHA256 and HTTP LFS needs one per
        # file: hash once, cached by stat identity so a resume reads nothing again.
        rows = hub_lfs.hashed_rows({name: output / name for name in files},
                                   state_root / "upload-hashes.json", workers=options.workers)
        if any(rows[name]["local"] != local for name, local in files.items()):
            raise ValueError("export changed during upload")
        head = api.repo_info(repo_id, repo_type="model").sha
        done = set()
        if head != parent["commit"]:
            # Earlier verified batches, or a lost commit reply. Never overwrite another writer.
            done = hub_lfs.committed_names(_remote(api, repo_id, head), rows)

        def verify(commit, batch, operations):
            done.update(batch)
            remote = _remote(api, repo_id, commit)
            if set(remote) != done:
                raise ValueError("committed remote inventory differs from uploaded files")
            for name in done:
                hub_lfs.resolved(remote[name], rows[name], name)

        pending = [name for name in hub_lfs.upload_order(rows) if name not in done]
        commit = head
        if pending:
            progress(dict(event="upload_started", files=len(pending),
                          bytes=sum(rows[name]["bytes"] for name in pending), workers=options.workers,
                          batch_files=options.batch_files))
            commit = hub_lfs.commit_batches(api, repo_id, rows, pending, revision=head,
                commit_message="Publish routed-only V4.1 EXL3 K3.25 model", options=options,
                check_sources=lambda batch: hub_lfs.check_fingerprints(rows, batch), verify=verify,
                log=lambda message: progress(dict(event="upload_progress", message=message)))
        remote = _remote(api, repo_id, commit)
        if set(remote) != set(files):
            raise ValueError("committed remote inventory differs from uploaded files")
        prepared = {name: dict(local=local, **hub_lfs.resolved(remote[name], rows[name], name))
                    for name, local in files.items()}
        if any(_fingerprint(output / name) != local for name, local in files.items()):
            raise ValueError("export changed before upload receipt")
        receipt = dict(schema="cuteafd-upload-receipt-v1", status="uploaded", repo_id=repo_id,
                       commit=commit, files=prepared)
        _publish_json(receipt_path, receipt)
        progress(dict(event="model_uploaded", repo_id=repo_id, commit=commit, files=len(prepared)))
        return receipt

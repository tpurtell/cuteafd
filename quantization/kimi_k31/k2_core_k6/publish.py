"""Publish a completed K3.1 run, compare Hub checksums, and link a local snapshot."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path

from kimi_k3.uniform_k2.publish import install_upload_compat
from kimi_k3.uniform_k2.storage import atomic_json, file_identity, verify_receipt
from step5.core_k6.publish import (_audit_remote, _cache_snapshot, _field, _hardlink_snapshot,
    _manifest_digest, _metadata_digests, LOCAL_HUB_CACHE)
from .export import receipt_path, shard_name
from .recipe import MODULE_COUNT, SHARD_COUNT, TARGET_PAYLOAD_BYTES, TARGET_REPO, TARGET_TENSOR_COUNT

IGNORE_PATTERNS = ["*-receipt.json", ".cache/**", "**/.cache/**", "*.partial", ".json-*"]


def _complete_plan(export, root, repo_id):
    if repo_id != TARGET_REPO:
        raise ValueError("publication target differs from the reviewed K3.1 repository")
    plan = json.loads((root / "plan.json").read_text())
    complete = json.loads((root / "quantization-complete.json").read_text())
    identity = plan.get("identity")
    if (plan.get("publication_repo") != repo_id or Path(plan["root"]).resolve() != root
            or Path(plan["export"]).resolve() != export or not isinstance(identity, dict) or not identity
            or complete.get("identity") != identity):
        raise ValueError("publication requires the matching immutable completed producer")
    for key, value in dict(shards=SHARD_COUNT, tensors=TARGET_TENSOR_COUNT,
                           payload_bytes=TARGET_PAYLOAD_BYTES, quantized_projections=MODULE_COUNT).items():
        if complete.get("summary", {}).get(key) != value:
            raise ValueError("quantization completion inventory differs")
    index = json.loads((export / "model.safetensors.index.json").read_text())
    if (index.get("metadata", {}).get("total_size") != TARGET_PAYLOAD_BYTES
            or len(index.get("weight_map", {})) != TARGET_TENSOR_COUNT
            or set(index["weight_map"].values()) != {shard_name(i) for i in range(1, SHARD_COUNT + 1)}):
        raise ValueError("publication checkpoint index differs")
    if json.loads((export / "quantization-provenance.json").read_text()) != identity:
        raise ValueError("publication provenance differs")
    return identity


def _collect_local(export, identity):
    files = {}
    shards = {shard_name(i) for i in range(1, SHARD_COUNT + 1)}
    for directory, names, leaves in os.walk(export, followlinks=False):
        if any((Path(directory) / n).is_symlink() for n in names):
            raise ValueError("publication directory cannot be a symlink")
        names[:] = [n for n in names if n != ".cache"]
        for name in leaves:
            if name.endswith(("-receipt.json", ".partial")) or name.startswith(".json-"):
                continue
            path = Path(directory) / name
            relative = path.relative_to(export).as_posix()
            if path.is_symlink() or not path.is_file():
                raise ValueError("publication files must be regular")
            if path.suffix == ".safetensors":
                if relative not in shards:
                    raise ValueError("unexpected publication tensor file")
                receipt = json.loads(receipt_path(export, relative).read_text())
                if receipt.get("identity") != identity or receipt.get("filename") != relative:
                    raise ValueError("publication shard identity differs")
                verify_receipt(path, receipt)
                sha256, blob = receipt["sha256"], None
            else:
                sha256, blob = _metadata_digests(path)
            files[relative] = dict(path=path, size=path.stat().st_size, sha256=sha256,
                                   blob_id=blob, file_identity=file_identity(path))
    required = {"config.json", "quantize_config.json", "model.safetensors.index.json", "README.md",
                "quantization-provenance.json", "tokenizer_config.json", "tokenization_kimi.py", "encoding_k3.py",
                "quantization-errors/index.json"}
    if not required <= files.keys() or {n for n in files if n.endswith(".safetensors")} != shards:
        raise ValueError("publication assets or shard inventory are incomplete")
    if not any(Path(n).name.upper().startswith("LICENSE") for n in files):
        raise ValueError("publication source license is missing")
    return files


def publish(export, root, repo_id):
    export, root = Path(export).expanduser(), Path(root).expanduser()
    if export.is_symlink() or root.is_symlink():
        raise ValueError("publication roots must be regular directories")
    export, root = export.resolve(), root.resolve()
    identity = _complete_plan(export, root, repo_id)
    files = _collect_local(export, identity)
    LOCAL_HUB_CACHE.mkdir(parents=True, exist_ok=True)
    if LOCAL_HUB_CACHE.is_symlink() or export.stat().st_dev != LOCAL_HUB_CACHE.stat().st_dev:
        raise ValueError("publication cache must share the export filesystem")
    marker = root / "publication-complete.json"
    if marker.exists():
        record = json.loads(marker.read_text())
        snapshot = _cache_snapshot(repo_id, record.get("revision"))
        if (record.get("identity") != identity or record.get("repo_id") != repo_id
                or record.get("manifest_sha256") != _manifest_digest(files)
                or record.get("local_snapshot") != str(snapshot)
                or not all((snapshot / n).is_file() and not (snapshot / n).is_symlink()
                           and os.path.samefile(v["path"], snapshot / n) for n, v in files.items())):
            raise ValueError("completed publication identity or cache differs")
        return record["revision"]
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    install_upload_compat()
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
    api.upload_large_folder(repo_id=repo_id, repo_type="model", folder_path=str(export),
        ignore_patterns=IGNORE_PATTERNS, num_workers=4, print_report=True, print_report_every=1800)
    info = api.model_info(repo_id, files_metadata=True)
    revision = _field(info, "sha")
    remote = {_field(item, "rfilename"): item for item in _field(info, "siblings")}
    if _field(info, "private") is not False or set(remote) - files.keys() - {".gitattributes"}:
        raise ValueError("published repository contains unexpected or private content")
    for name, value in files.items():
        if name not in remote:
            raise ValueError(f"published file absent: {name}")
        _audit_remote(name, value, remote[name])
    if ".gitattributes" in remote and ".gitattributes" not in files:
        path = Path(hf_hub_download(repo_id, ".gitattributes", repo_type="model", revision=revision,
                    local_dir=export, cache_dir=LOCAL_HUB_CACHE))
        if path.is_symlink() or path.resolve() != export / ".gitattributes":
            raise ValueError("Hub attributes were not saved to the export")
        sha256, blob = _metadata_digests(path)
        files[".gitattributes"] = dict(path=path, size=path.stat().st_size, sha256=sha256,
                                       blob_id=blob, file_identity=file_identity(path))
        _audit_remote(".gitattributes", files[".gitattributes"], remote[".gitattributes"])
    anonymous = HfApi(token=False).model_info(repo_id, revision=revision)
    if _field(anonymous, "sha") != revision or _field(anonymous, "private") is not False:
        raise ValueError("published revision is not anonymously accessible")
    snapshot = _cache_snapshot(repo_id, revision)
    _hardlink_snapshot(snapshot, files)
    cached = snapshot_download(repo_id, repo_type="model", revision=revision,
                               cache_dir=LOCAL_HUB_CACHE, local_files_only=True)
    if Path(cached).resolve() != snapshot:
        raise ValueError("offline HF resolution differs from the published cache")
    atomic_json(marker, dict(repo_id=repo_id, revision=revision, identity=identity,
        manifest_sha256=_manifest_digest(files), local_snapshot=str(snapshot), files=len(files),
        time=datetime.now(timezone.utc).isoformat()))
    return revision

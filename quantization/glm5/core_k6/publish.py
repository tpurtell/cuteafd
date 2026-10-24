"""Publish only a completed GLM run, audit Hub bytes, and hardlink its cache."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

from kimi_k3.uniform_k2.publish import install_upload_compat
from kimi_k3.uniform_k2.storage import atomic_json, canonical, file_identity, fsync_directory, verify_receipt
from .export import SHARD_COUNT
from .recipe import MODULE_COUNT, TARGET_PAYLOAD_BYTES, TARGET_REPO, TARGET_TENSOR_COUNT

LOCAL_HUB_CACHE = Path.home() / ".cache/huggingface/cuteafd-local-hub"
IGNORE_PATTERNS = ["*-receipt.json", ".cache/**", "**/.cache/**", "*.partial", ".json-*"]
REQUIRED_ASSETS = {"config.json", "quantize_config.json", "model.safetensors.index.json",
    "quantization-provenance.json", "README.md", "LICENSE", "tokenizer.json",
    "tokenizer_config.json", "chat_template.jinja", "generation_config.json"}


def _field(value, name):
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def _expected_shards():
    return {f"model-{i:05d}-of-{SHARD_COUNT:05d}.safetensors" for i in range(1, SHARD_COUNT + 1)}


def _receipt_path(export, name):
    match = re.fullmatch(r"model-([0-9]{5})-of-([0-9]{5})\.safetensors", name)
    if (match is None or int(match[2]) != SHARD_COUNT
            or not 1 <= int(match[1]) <= SHARD_COUNT):
        raise ValueError(f"unreviewed publication shard: {name}")
    ordinal = int(match[1])
    return export / ("extras-receipt.json" if ordinal == SHARD_COUNT else
                     f"layer-{ordinal - 1:03d}-receipt.json")


def _complete_plan(export, root, repo_id):
    if repo_id != TARGET_REPO:
        raise ValueError("publication target differs from the reviewed GLM repository")
    plan = json.loads((root / "plan.json").read_text())
    complete = json.loads((root / "quantization-complete.json").read_text())
    if (plan.get("publication_repo") != repo_id or Path(plan["root"]).resolve() != root
            or Path(plan["export"]).resolve() != export):
        raise ValueError("publication run paths or target differ from the immutable plan")
    identity = plan.get("identity")
    if (not isinstance(identity, dict) or not identity
            or complete.get("identity") != identity):
        raise ValueError("publication requires a matching completed quantization")
    expected = dict(shards=SHARD_COUNT, tensors=TARGET_TENSOR_COUNT,
                    payload_bytes=TARGET_PAYLOAD_BYTES, quantized_projections=MODULE_COUNT)
    if any(complete.get("summary", {}).get(key) != value for key, value in expected.items()):
        raise ValueError("quantization completion inventory differs from the reviewed target")
    index = json.loads((export / "model.safetensors.index.json").read_text())
    if (index.get("metadata", {}).get("total_size") != TARGET_PAYLOAD_BYTES
            or len(index.get("weight_map", {})) != TARGET_TENSOR_COUNT
            or set(index["weight_map"].values()) != _expected_shards()):
        raise ValueError("publication checkpoint index inventory differs")
    if json.loads((export / "quantization-provenance.json").read_text()) != identity:
        raise ValueError("export provenance differs from the completed producer")
    return identity


def _metadata_digests(path):
    sha256, blob = hashlib.sha256(), hashlib.sha1()
    blob.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as file:
        while chunk := file.read(4 * 1024 * 1024):
            sha256.update(chunk)
            blob.update(chunk)
    return sha256.hexdigest(), blob.hexdigest()


def _collect_local(export, identity):
    files = {}
    # Prune upload caches before walking, and never follow a symlink into
    # another tree. Internal receipts remain local numerical evidence.
    for directory, names, leaves in os.walk(export, followlinks=False):
        for name in names:
            if (Path(directory) / name).is_symlink():
                raise ValueError("refuse symlink publication directory")
        names[:] = [name for name in names if name != ".cache"]
        for name in leaves:
            if name.endswith(("-receipt.json", ".partial")) or name.startswith(".json-"):
                continue
            path = Path(directory) / name
            if path.is_symlink() or not path.is_file():
                raise ValueError("publication payload must contain only regular files")
            relative = path.relative_to(export).as_posix()
            if path.suffix == ".safetensors":
                if relative not in _expected_shards():
                    raise ValueError(f"unexpected publication tensor file: {relative}")
                receipt_path = _receipt_path(export, relative)
                if receipt_path.is_symlink():
                    raise ValueError("refuse symlink publication receipt")
                receipt = json.loads(receipt_path.read_text())
                ordinal = int(name.split("-")[1])
                if (receipt.get("identity") != identity or receipt.get("filename") != name
                        or (ordinal != SHARD_COUNT and receipt.get("layer") != ordinal - 1)
                        or not re.fullmatch(r"[0-9a-f]{64}", receipt.get("sha256", ""))):
                    raise ValueError(f"publication shard receipt differs: {relative}")
                verify_receipt(path, receipt, checksum=True)
                digest, blob = receipt["sha256"], None
            else:
                digest, blob = _metadata_digests(path)
            files[relative] = dict(path=path, size=path.stat().st_size, sha256=digest, blob_id=blob,
                                   file_identity=file_identity(path))
    if not REQUIRED_ASSETS <= files.keys():
        raise ValueError(f"publication assets missing: {sorted(REQUIRED_ASSETS-files.keys())}")
    if {name for name in files if name.endswith(".safetensors")} != _expected_shards():
        raise ValueError("publication shard inventory differs")
    return files


def _manifest_digest(files):
    entries = [[name, value["size"], value["sha256"]] for name, value in sorted(files.items())]
    return hashlib.sha256(canonical(entries).encode()).hexdigest()


def _audit_remote(name, local, remote):
    if _field(remote, "size") != local["size"]:
        raise ValueError(f"published file size differs: {name}")
    lfs = _field(remote, "lfs")
    if name.endswith(".safetensors") and lfs is None:
        raise ValueError(f"published shard is not an LFS object: {name}")
    if lfs is not None:
        if _field(lfs, "sha256") != local["sha256"]:
            raise ValueError(f"published LFS checksum differs: {name}")
    elif _field(remote, "blob_id") != local["blob_id"]:
        raise ValueError(f"published Git blob differs: {name}")


def _cache_snapshot(repo_id, revision):
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        raise ValueError("Hub returned an invalid immutable revision")
    return LOCAL_HUB_CACHE / ("models--" + repo_id.replace("/", "--")) / "snapshots" / revision


def _hardlink_snapshot(snapshot, files):
    directories = set()
    for name, value in files.items():
        source, target = value["path"], snapshot / name
        verify_receipt(source, value)
        parent = target.parent
        while parent != LOCAL_HUB_CACHE.parent:
            if parent.is_symlink():
                raise ValueError(f"existing cache directory is a symlink: {name}")
            directories.add(parent)
            parent = parent.parent
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            raise ValueError(f"existing cache entry is a symlink: {name}")
        if target.exists():
            if not os.path.samefile(source, target):
                raise ValueError(f"existing snapshot file differs: {name}")
        else:
            os.link(source, target)
        # Make each newly created directory and link reachable durably without
        # reading or copying the model payload again.
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        fsync_directory(directory)
    refs = snapshot.parents[1] / "refs"
    if refs.is_symlink():
        raise ValueError("existing cache refs directory is a symlink")
    refs.mkdir(exist_ok=True)
    temporary = refs / ".main.partial"
    if temporary.is_symlink() or (refs / "main").is_symlink():
        raise ValueError("existing cache ref is a symlink")
    with temporary.open("w") as file:
        file.write(snapshot.name)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, refs / "main")
    fsync_directory(refs)


def publish(export, root, repo_id):
    """Upload and verify an approved complete run; this function never quantizes."""
    export, root = Path(export).expanduser(), Path(root).expanduser()
    if export.is_symlink() or root.is_symlink():
        raise ValueError("publication roots must be regular directories")
    export, root = export.resolve(), root.resolve()
    identity = _complete_plan(export, root, repo_id)
    files = _collect_local(export, identity)
    LOCAL_HUB_CACHE.mkdir(parents=True, exist_ok=True)
    if LOCAL_HUB_CACHE.is_symlink() or export.stat().st_dev != LOCAL_HUB_CACHE.stat().st_dev:
        raise ValueError("local HF snapshot requires a regular cache on the export filesystem")
    marker = root / "publication-complete.json"
    if marker.exists():
        complete = json.loads(marker.read_text())
        snapshot = _cache_snapshot(repo_id, complete.get("revision"))
        if (complete.get("repo_id") != repo_id or complete.get("identity") != identity
                or complete.get("manifest_sha256") != _manifest_digest(files)
                or Path(complete["local_snapshot"]) != snapshot):
            raise ValueError("publication completion identity or payload differs")
        if not all((snapshot / name).exists() and not (snapshot / name).is_symlink()
                   and os.path.samefile(value["path"], snapshot / name)
                   for name, value in files.items()):
            raise ValueError("published local snapshot no longer matches its export")
        return complete["revision"]

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    install_upload_compat()
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
    api.upload_large_folder(repo_id=repo_id, repo_type="model", folder_path=str(export),
        ignore_patterns=IGNORE_PATTERNS, num_workers=4, print_report=True, print_report_every=1800)
    info = api.model_info(repo_id, files_metadata=True)
    revision = _field(info, "sha")
    snapshot = _cache_snapshot(repo_id, revision)
    if _field(info, "private") is not False:
        raise ValueError("published model repository is not public")
    siblings = _field(info, "siblings")
    remote = {_field(item, "rfilename"): item for item in siblings}
    if set(remote) - files.keys() - {".gitattributes"}:
        raise ValueError("remote repository contains unexpected files")
    for name, local in files.items():
        if name not in remote:
            raise ValueError(f"published file absent: {name}")
        _audit_remote(name, local, remote[name])
    if ".gitattributes" in remote and ".gitattributes" not in files:
        path = Path(hf_hub_download(repo_id, ".gitattributes", repo_type="model", revision=revision,
                    local_dir=export, cache_dir=LOCAL_HUB_CACHE))
        if path.resolve() != export / ".gitattributes" or path.is_symlink():
            raise ValueError("generated Hub attributes were not saved into the export")
        sha256, blob = _metadata_digests(path)
        local = dict(path=path, size=path.stat().st_size, sha256=sha256, blob_id=blob,
                     file_identity=file_identity(path))
        _audit_remote(".gitattributes", local, remote[".gitattributes"])
        with path.open("rb") as file:
            os.fsync(file.fileno())
        fsync_directory(export)
        files[".gitattributes"] = local
    anonymous = HfApi(token=False).model_info(repo_id, revision=revision)
    if _field(anonymous, "sha") != revision or _field(anonymous, "private") is not False:
        raise ValueError("published revision is not anonymously accessible")
    _hardlink_snapshot(snapshot, files)
    cached = snapshot_download(repo_id, repo_type="model", revision=revision,
                               cache_dir=LOCAL_HUB_CACHE, local_files_only=True)
    if Path(cached).resolve() != snapshot.resolve():
        raise ValueError("offline HF cache does not resolve the published revision")
    atomic_json(marker, dict(repo_id=repo_id, revision=revision, identity=identity,
        manifest_sha256=_manifest_digest(files), local_snapshot=str(snapshot), files=len(files),
        time=datetime.now(timezone.utc).isoformat()))
    return revision

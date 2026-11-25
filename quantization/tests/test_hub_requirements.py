"""Exercise the qualified SDK itself, mocking only remote/transport boundaries."""
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from huggingface_hub import CommitOperationAdd, CommitOperationCopy, HfApi
from huggingface_hub import _commit_api
from huggingface_hub.hf_api import RepoFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hub_requirements
from upload_model import _restored_operation


@pytest.mark.parametrize("version,http_lfs", [("1.26.1", True), ("2.2.0", True), ("1.33.0", False)])
def test_unqualified_sdk_or_xet_transport_is_rejected(monkeypatch, version, http_lfs):
    from huggingface_hub import constants
    monkeypatch.setattr(hub_requirements.importlib.metadata, "version", lambda _: version)
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_XET", http_lfs)
    with pytest.raises(ValueError, match="qualified huggingface_hub 1.33.0 over HTTP LFS"):
        hub_requirements.require_qualified_hub()


def test_real_preupload_uses_http_lfs_with_receipt_sha_and_parallel_files(tmp_path, monkeypatch):
    """The pinned SDK's own LFS path: receipt SHA256, no rehash, N files at once."""
    import huggingface_hub.lfs as lfs
    import huggingface_hub.utils as hub_utils
    import hub_lfs
    hub_requirements.require_qualified_hub()
    rows = {}
    for i in range(3):
        path = tmp_path / f"model-{i}.safetensors"
        path.write_bytes(b"synthetic weights %d" % i)
        rows[path.name] = dict(path=str(path), bytes=path.stat().st_size,
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    ops = [hub_lfs.receipt_operation(name, row) for name, row in rows.items()]
    monkeypatch.setattr(lfs, "sha_fileobj", Mock(side_effect=AssertionError("payload rehash")))

    def classify(**kwargs):
        for addition in kwargs["additions"]:
            addition._upload_mode = "lfs"
            addition._should_ignore = False

    def batch_info(**kwargs):
        infos = kwargs["upload_infos"]
        assert [info.sha256.hex() for info in infos] == [row["sha256"] for row in rows.values()]
        return [dict(oid=info.sha256.hex(), actions=dict(upload=dict(href="x"))) for info in infos], [], None

    uploaded, workers = [], []

    def thread_map(function, items, **kwargs):
        workers.append(kwargs["max_workers"])
        return [function(item) for item in items]

    monkeypatch.setattr(_commit_api, "_fetch_upload_modes", Mock(side_effect=classify))
    monkeypatch.setattr(_commit_api, "_upload_xet_files", Mock(side_effect=AssertionError("Xet")))
    monkeypatch.setattr(_commit_api, "post_lfs_batch_info", Mock(side_effect=batch_info))
    monkeypatch.setattr(_commit_api, "lfs_upload", lambda operation, **kwargs: uploaded.append(operation.path_in_repo))
    monkeypatch.setattr(hub_utils, "hf_thread_map", thread_map)
    HfApi(token=False).preupload_lfs_files("test/model", ops, repo_type="model", revision="main",
                                           num_threads=16, free_memory=False, gitignore_content="")
    assert workers == [16] and uploaded == list(rows)
    assert all(op._is_uploaded for op in ops)


def test_restored_receipt_avoids_reupload(tmp_path, monkeypatch):
    hub_requirements.require_qualified_hub()
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"synthetic weights")
    digest = hashlib.sha256(path.read_bytes()).digest()
    monkeypatch.setattr(_commit_api, "_upload_files", Mock(side_effect=AssertionError("reupload")))
    restored = _restored_operation(path.name, path, dict(mode="lfs", blob=digest.hex()))
    HfApi(token=False).preupload_lfs_files("test/model", [restored], free_memory=False, gitignore_content="")
    assert restored._is_uploaded and restored.upload_info.sha256 == digest


def test_real_commit_copies_lfs_server_side_without_reading_source(monkeypatch):
    hub_requirements.require_qualified_hub()
    revision = "a" * 40
    op = CommitOperationCopy("source.safetensors", "model.safetensors", src_revision=revision,
                             src_repo_id="test/source", src_repo_type="model")
    source = RepoFile(path="source.safetensors", size=123, oid="b" * 40,
                      lfs=dict(oid="c" * 64, size=123, pointerSize=130), xetHash="d" * 64)
    files = {_commit_api._CopySource("test/source", "model", "source.safetensors", revision): source}
    fetch = Mock(return_value=files)
    monkeypatch.setattr(_commit_api, "_fetch_files_to_copy", fetch)
    response = Mock(status_code=200)
    response.json.return_value = {"failed": []}
    session = Mock()
    session.post.return_value = response
    monkeypatch.setattr("huggingface_hub.hf_api.get_session", lambda: session)
    monkeypatch.setattr("huggingface_hub.hf_api.hf_raise_for_status", lambda _: None)
    send = Mock(return_value=SimpleNamespace(oid="e" * 40))
    monkeypatch.setattr(_commit_api, "_send_commit", send)
    result = HfApi(token=False).create_commit("test/dest", [op], repo_type="model", revision="main",
                                           parent_commit=revision, commit_message="Mock copy")
    assert result.oid == "e" * 40
    assert op._is_duplicated
    assert fetch.call_args.kwargs["copies"] == [op]
    assert session.post.call_args.args == ("https://huggingface.co/api/models/test/source/lfs-files/duplicate",)
    assert session.post.call_args.kwargs["json"] == dict(
        target=dict(type="model", name="test/dest"),
        files=[dict(xetHash="d" * 64, sha256="c" * 64, filename="source.safetensors")])
    assert send.call_args.kwargs["operations"] == [op]
    assert send.call_args.kwargs["parent_commit"] == revision


def test_publish_units_require_http_lfs_and_never_tune_xet():
    units = sorted((Path(__file__).resolve().parents[1]).glob("**/systemd/*publish*.service"))
    assert units
    for unit in units:
        text = unit.read_text()
        assert "Environment=HF_HUB_DISABLE_XET=1" in text, unit
        assert "HF_XET_" not in text, unit


def test_publisher_cli_options_pin_http_lfs(monkeypatch):
    import argparse
    import os
    from huggingface_hub import constants
    import hub_lfs
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_XET", False)
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    parser = argparse.ArgumentParser()
    hub_lfs.add_arguments(parser)
    options = hub_lfs.options_from(parser.parse_args([]))
    assert (options.workers, options.batch_files) == (16, 48)
    assert constants.HF_HUB_DISABLE_XET is True and os.environ["HF_HUB_DISABLE_XET"] == "1"
    hub_requirements.require_qualified_hub()
    options = hub_lfs.options_from(parser.parse_args(["--upload-workers", "8", "--upload-batch-files", "64"]))
    assert (options.workers, options.batch_files) == (8, 64)
    with pytest.raises(SystemExit):
        parser.parse_args(["--upload-workers", "0"])

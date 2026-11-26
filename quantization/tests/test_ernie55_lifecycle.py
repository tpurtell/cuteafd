"""Local receipts and native/LFS contracts for ERNIE handoff."""
from datetime import datetime
from pathlib import Path

import pytest

from ernie55_flash import progress, publish
from ernie55_flash.handoff import LAYOUT
from ernie55_flash.source import BASE_REPO, BASE_REVISION
from inkling2 import handoff as nest
from inkling2 import publish as hub
from kimi_k3.uniform_k2.storage import atomic_json


def test_published_manifest_binds_ernie_source_and_all_views(tmp_path):
    path = tmp_path / "cuteafd-mc.json"
    mul1 = dict(name="mul1", multiplier=0x83DCD12D)
    identity = dict(source_repo=BASE_REPO, source_revision=BASE_REVISION,
                    codebook="mul1", codebook_multiplier=mul1["multiplier"])
    manifest = dict(source=dict(repo=BASE_REPO, revision=BASE_REVISION), original_core_dtype="BF16",
                    variants=[dict(name=v) for v in ("original", "k8-compact", "k6-compact")], exl3_codebook=mul1)
    atomic_json(path, manifest)
    validation = dict(new_file_sha256={path.name: hub.metadata_identity(path)["sha256"]})
    assert publish.TARGET.validate(tmp_path, dict(identity=identity), validation) == manifest
    with pytest.raises(ValueError, match="pinned source"):
        publish.TARGET.validate(tmp_path, dict(identity=dict(identity, source_repo="other")), validation)


def test_retire_ernie_preserves_source_and_archives_sibling_receipts(tmp_path):
    root, source, output, stage = (tmp_path / name for name in ("production", "source", "output", "stage"))
    for p in (root, source, output):
        p.mkdir()
    (source / "source.safetensors").touch()
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    (bootstrap / "input.bin").touch()
    (bootstrap / "complete.json").write_text("{}")
    identity = dict(repo_id=LAYOUT.target.repo_id, revision="a" * 40, manifest_sha256="b" * 64)
    atomic_json(root / "publication-complete.json", dict(identity, identity={"model": "ERNIE"},
                output=str(output), status="passed", source_snapshot=str(source)))
    atomic_json(output / "package-state.json", dict(identity={"model": "ERNIE"}))
    atomic_json(root / "handoff-complete.json", dict(identity=identity, status="passed", stage_root=str(stage),
                                                      one_fabric_copy_per_file=True))
    kwargs = dict(repo_id=LAYOUT.target.repo_id, family=LAYOUT.target.family,
                  placement_check="one_fabric_copy_per_file", inspect=lambda paths: None)
    for unsafe in ("../elsewhere", "source", "output"):
        with pytest.raises(ValueError):
            nest.retire(root, task_intermediates=[unsafe], **kwargs)
    receipt = nest.retire(root, task_intermediates=LAYOUT.task_intermediates, **kwargs)
    assert (Path(receipt["receipt_archive"]) / "intermediates/task/bootstrap/complete.json").exists()
    assert not bootstrap.exists() and not output.exists()
    assert (source / "source.safetensors").exists()
    assert receipt["status"] == "passed"


def test_unit_chain_requires_same_boot_and_reports_errors(tmp_path):
    pkg = Path(publish.__file__).parent
    for name, next_unit in (("publish", "handoff"), ("handoff", "cleanup"), ("cleanup", "progress")):
        unit = (pkg / f"systemd/cuteafd-ernie55-{name}.service").read_text()
        assert f"OnSuccess=cuteafd-ernie55-{next_unit}.service" in unit
        assert "OnFailure=cuteafd-ernie55-progress.service" in unit
        assert "Restart=no" in unit and "WantedBy=" not in unit
        assert "ExecCondition=/usr/bin/cmp --silent /proc/sys/kernel/random/boot_id" in unit
    unit = (pkg / "systemd/cuteafd-ernie55-publish.service").read_text()
    assert "HF_HUB_DISABLE_XET=1" in unit and "StandardInput=file:" in unit
    assert "--upload-workers 16" in unit
    state = progress.production_state(tmp_path, datetime.now(progress.TAIPEI), set(), {"publish"})
    assert state["phase"] == "publish failed" and "publish.service" in state["blocker"]

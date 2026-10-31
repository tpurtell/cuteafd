"""No-network tests for partial HF imports, one-copy placement and retirement."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "quantization"))
from mistral_large4.core_k6 import archive as storage


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    root, export = tmp_path / "run", tmp_path / "export"
    root.mkdir(); export.mkdir()
    revision = "ab" * 20
    repo_name = "models--" + storage.TARGET_REPO.replace("/", "--")
    snapshot = tmp_path / "local-hub" / repo_name / "snapshots" / revision
    snapshot.mkdir(parents=True)
    names = sorted(storage._shards()) + ["README.md", "config.json", "quantize_config.json",
        "model.safetensors.index.json", "consolidated.safetensors.index.json", "LICENSE",
        "tekken.json", "quantization-errors/layer-000.jsonl"]
    files = {}
    for name in names:
        source, target = export / name, snapshot / name
        source.parent.mkdir(parents=True, exist_ok=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(("published content " + name).encode())
        os.link(source, target)
        files[name] = dict(size=source.stat().st_size, sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    producer = {"schema": "toy-complete-producer"}
    write_json(root / "plan.json", dict(root=str(root), export=str(export), identity=producer,
                                       publication_repo=storage.TARGET_REPO))
    write_json(root / "quantization-complete.json", dict(identity=producer, summary=dict(
        shards=storage.SHARD_COUNT, tensors=storage.TARGET_TENSOR_COUNT,
        payload_bytes=storage.TARGET_PAYLOAD_BYTES, quantized_projections=storage.MODULE_COUNT)))
    write_json(root / "publication-complete.json", dict(repo_id=storage.TARGET_REPO, identity=producer,
        revision=revision, local_snapshot=str(snapshot), manifest_sha256=storage._manifest_digest(files), files=len(files)))
    managed_hub = tmp_path / "managed-hub"
    monkeypatch.setattr(storage, "MANAGED_HUB", managed_hub)
    managed = managed_hub / repo_name / "snapshots" / revision
    return SimpleNamespace(root=root, export=export, snapshot=snapshot, files=files,
                           managed=managed, revision=revision, repo_name=repo_name)


@pytest.fixture
def network(artifact, monkeypatch):
    events, transfers, imports = [], [], []
    behavior = dict(bad_hash=None, copy_bytes=0, failed_host=None, unexpected_host=None, symlink_weights=False)
    jobs = {}

    def remote(host, operation, plan):
        events.append((operation, host))
        if operation == "prepare":
            return dict(same_filesystem=True)
        if operation == "verify":
            files = storage._host_files(plan, host)
            return dict(verified=host != behavior["bad_hash"], identity=plan["identity"],
                        files=len(files), bytes=sum(row["size"] for row in files.values()))
        assert operation == "retire"
        assert (artifact.root / "archive-complete.json").exists(), "retirement precedes durable global proof"
        return dict(retired=True)

    def transfer(host, plan, directory):
        transfers.append((host, set(storage._host_files(plan, host))))
        return Path("/fake-stage") / plan["repo_name"]

    def ssh(host, command, **kwargs):
        if "import" in command:
            assert "--offline" in command and "--no-copy" in command and "--wait" not in command
            assert "--spread" not in command and "--archive" not in command
            imports.append(host)
            plan = json.loads((artifact.root / "archive-plan.json").read_text())
            for name in storage._host_files(plan, host):
                target = artifact.managed / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if name.endswith(".safetensors") and behavior["symlink_weights"]:
                    blob = storage.MANAGED_HUB / "blobs" / name
                    blob.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(artifact.snapshot / name, blob)
                    target.symlink_to(blob)
                else:
                    shutil.copyfile(artifact.snapshot / name, target)
            number = len(jobs) + 1
            jobs[number] = host
            return json.dumps(dict(job=number))
        assert command[:3] == ["nest", "--json", "jobs"]
        host = jobs[int(command[3])]
        error = "transient native import failure" if behavior["failed_host"] == host else None
        return json.dumps(dict(job=dict(finished=True, error=error, cancelled=False),
            import_=None)).replace('"import_": null', '"import": ' + json.dumps(dict(
                finished=True, errors=[], copied_bytes=behavior["copy_bytes"])))

    def run(command, **kwargs):
        assert command[:3] == ["nest", "--json", "ls"]
        plan = json.loads((artifact.root / "archive-plan.json").read_text())
        if Path(command[3]) != artifact.managed:
            name = Path(command[3]).name
            return json.dumps(dict(entries=[dict(name=name, kind="file", hosts=[plan["assignments"][name]],
                sealed=True, writing_on=None, size=plan["files"][name]["size"])]))
        rows = [dict(name=name, hosts=[host] if not behavior["unexpected_host"] else [host, "raptor"],
                     sealed=True, writing_on=None, size=plan["files"][name]["size"])
                for name, host in plan["assignments"].items()]
        if behavior["symlink_weights"]:
            for row in rows:
                row.update(kind="symlink",hosts=[],sealed=False,size=52)
        return json.dumps(dict(entries=rows))

    monkeypatch.setattr(storage, "_remote", remote)
    monkeypatch.setattr(storage, "_transfer", transfer)
    monkeypatch.setattr(storage, "_ssh", ssh)
    monkeypatch.setattr(storage, "_run", run)
    return SimpleNamespace(events=events, transfers=transfers, imports=imports, behavior=behavior)


def test_one_total_weight_copy_is_balanced_over_exactly_six_hosts(artifact):
    assignments = storage.balance(artifact.files)
    assert len(assignments) == 73
    assert set(assignments.values()) == set(storage.HOSTS)
    counts = [sum(owner == host for owner in assignments.values()) for host in storage.HOSTS]
    assert max(counts) - min(counts) <= 1
    assert set(assignments) == storage._shards()


def test_partial_import_union_verified_then_only_weights_retired(artifact, network):
    complete = storage.archive(artifact.root)
    assert complete["status"] == "complete" and complete["weight_shards"] == 73
    assert network.imports == list(storage.HOSTS)
    all_weights = [name for _, names in network.transfers for name in names if name.endswith(".safetensors")]
    assert len(all_weights) == len(set(all_weights)) == 73
    for _, names in network.transfers:
        assert "README.md" in names and "quantization-errors/layer-000.jsonl" in names
    assert not list(artifact.export.glob("*.safetensors"))
    assert not list(artifact.snapshot.glob("*.safetensors"))
    assert (artifact.export / "README.md").is_file()
    assert (artifact.export / "quantization-errors/layer-000.jsonl").is_file()
    assert (artifact.root / "plan.json").is_file()
    assert (artifact.root / "quantization-complete.json").is_file()
    assert all((artifact.managed / name).is_file() for name in artifact.files)
    assert json.loads((artifact.root / "archive-retirement.json").read_text())["local_weights_retired"] is True


def test_completed_archive_repeats_without_accessing_retired_sources_or_publisher(artifact, network):
    first = storage.archive(artifact.root)
    calls = len(network.transfers), len(network.imports), len(network.events)
    assert storage.archive(artifact.root) == first
    assert (len(network.transfers), len(network.imports), len(network.events)) == calls


def test_managed_blob_symlinks_use_resolved_object_residency_and_sealing(artifact, network):
    network.behavior["symlink_weights"] = True
    result = storage.archive(artifact.root)
    assert result["weight_shards"] == 73
    assert all((artifact.managed / name).is_symlink() for name in storage._shards())
    assert not list(artifact.export.glob("*.safetensors"))


@pytest.mark.parametrize("damage", ["weight", "metadata", "identity", "detached_cache"])
def test_bad_published_inputs_fail_before_network_or_retirement(artifact, network, damage):
    if damage in ("weight", "metadata"):
        target = artifact.snapshot / (sorted(storage._shards())[0] if damage == "weight" else "config.json")
        target.write_bytes(b"modified after publication")
    elif damage == "identity":
        path = artifact.root / "publication-complete.json"
        value = json.loads(path.read_text()); value["identity"] = {"schema": "other-producer"}; write_json(path, value)
    else:
        path = artifact.snapshot / "config.json"
        data = path.read_bytes(); path.unlink(); path.write_bytes(data)
    with pytest.raises(ValueError):
        storage.archive(artifact.root)
    assert network.events == [] and network.transfers == []
    assert not (artifact.root / "archive-complete.json").exists()
    assert all((artifact.export / name).exists() for name in storage._shards())


def test_remote_hash_failure_keeps_all_original_weights(artifact, network):
    network.behavior["bad_hash"] = "dodo"
    with pytest.raises(ValueError, match="checksum proof"):
        storage.archive(artifact.root)
    assert network.imports == ["ostrich"]
    assert not any(event[0] == "retire" for event in network.events)
    assert all((artifact.export / name).exists() for name in storage._shards())
    network.behavior["bad_hash"] = None
    storage.archive(artifact.root)
    assert [host for host, _ in network.transfers].count("ostrich") == 1


def test_retirement_failure_resumes_after_durable_marker_without_retransfer(artifact, network, monkeypatch):
    remote = storage._remote
    failed = False
    def retire_once(host, operation, plan):
        nonlocal failed
        if host == "dodo" and operation == "retire" and not failed:
            failed = True
            raise ConnectionError("temporary SSH loss during cleanup")
        return remote(host, operation, plan)
    monkeypatch.setattr(storage, "_remote", retire_once)
    with pytest.raises(ConnectionError):
        storage.archive(artifact.root)
    assert (artifact.root / "archive-complete.json").exists()
    assert all((artifact.export / name).exists() for name in storage._shards())
    count = len(network.transfers)
    storage.archive(artifact.root)
    assert len(network.transfers) == count
    assert not list(artifact.export.glob("*.safetensors"))


def test_failed_native_import_resumes_with_new_job_and_existing_managed_work(artifact, network):
    network.behavior["failed_host"] = "dodo"
    with pytest.raises(ValueError, match="native HF import"):
        storage.archive(artifact.root)
    assert not (artifact.root / "archive-complete.json").exists()
    network.behavior["failed_host"] = None
    storage.archive(artifact.root)
    assert network.imports.count("ostrich") == 1
    assert network.imports.count("dodo") == 2


def test_native_copy_or_bad_placement_never_authorizes_retirement(artifact, network):
    network.behavior["copy_bytes"] = 1
    with pytest.raises(ValueError, match="copied weight bytes"):
        storage.archive(artifact.root)
    assert all((artifact.export / name).exists() for name in storage._shards())
    network.behavior["copy_bytes"] = 0
    network.behavior["unexpected_host"] = True
    with pytest.raises(ValueError, match="exactly its assigned"):
        storage.archive(artifact.root)
    assert not (artifact.root / "archive-complete.json").exists()


def test_final_managed_corruption_preserves_export_and_staging(artifact, network, monkeypatch):
    sha = storage.sha256_file
    def corrupt(path):
        if Path(path).is_relative_to(artifact.managed):
            return "0" * 64
        return sha(path)
    monkeypatch.setattr(storage, "sha256_file", corrupt)
    with pytest.raises(ValueError, match="managed published checksum"):
        storage.archive(artifact.root)
    assert not any(event[0] == "retire" for event in network.events)
    assert all((artifact.export / name).exists() for name in storage._shards())


def test_rsync_uses_only_assigned_shards_and_safe_partial_snapshot_paths(artifact, monkeypatch):
    production, published = storage._context(artifact.root)
    plan = storage._new_plan(artifact.root, production, published)
    calls = []
    monkeypatch.setattr(storage, "_run", lambda command, **kwargs: calls.append((command, kwargs)) or "")
    listing = artifact.root / "lists"; listing.mkdir()
    destination = storage._transfer("ostrich", plan, listing)
    command, kwargs = calls[0]
    assert command[0] == "rsync" and "--sparse" in command and "--from0" in command and "--fsync" in command
    assert "--protect-args" in command and "--delete" not in command
    assert kwargs["timeout"] == 21600
    sent = set((listing / "ostrich.files0").read_bytes().decode().strip("\0").split("\0"))
    assert sent == set(storage._host_files(plan, "ostrich"))
    assert command[-1].startswith("ostrich:") and plan["revision"] in command[-1]
    assert destination.name == artifact.repo_name


def test_ssh_quotes_static_command_and_passes_paths_as_json_stdin(monkeypatch):
    calls = []
    monkeypatch.setattr(storage, "_run", lambda command, **kwargs: calls.append((command, kwargs)) or "{}")
    storage._ssh("ostrich", ["python3", "-c", "print('literal $() `data`')"], input="{}")
    assert calls[0][0][-1] == 'python3 -c \'print(\'"\'"\'literal $() `data`\'"\'"\')\''
    assert calls[0][1]["input"] == "{}"
    with pytest.raises(ValueError):
        storage._ssh("maxq", ["true"])


def test_remote_helper_refuses_foreign_stage_and_symlinks(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    stage_name = "mistral-published-import-fixture"
    stage = tmp_path / ".cache/cuteafd/builds" / stage_name
    stage.mkdir(parents=True)
    payload = dict(operation="prepare", stage_name=stage_name, identity={"publication": "ours"},
                   repo_name="models--org--name", revision="ab" * 20, files={})
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    with pytest.raises(ValueError, match="not owned"):
        exec(storage._REMOTE_HELPER, {})
    write_json(stage / "owner.json", payload["identity"])
    (stage / "hub").symlink_to(tmp_path)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    with pytest.raises(ValueError, match="symlink"):
        exec(storage._REMOTE_HELPER, {})


def test_remote_helper_checks_partial_snapshot_hashes_and_retires_owned_stage(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    original_stat = Path.stat
    device = tmp_path.stat().st_dev
    monkeypatch.setattr(Path, "stat", lambda p, *a, **kw:
        SimpleNamespace(st_dev=device) if str(p) == "/srv/sparknest" else original_stat(p, *a, **kw))
    payload = dict(operation="prepare", stage_name="mistral-published-import-fixture",
        identity={"publication": "ours"}, repo_name="models--org--name", revision="ab" * 20,
        files={"config.json": dict(size=2, sha256=hashlib.sha256(b"{}").hexdigest())})
    def invoke():
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
        exec(storage._REMOTE_HELPER, {})
        return json.loads(capsys.readouterr().out)
    ready = invoke()
    assert ready["same_filesystem"] is True
    snapshot = Path(ready["snapshot"])
    (snapshot / "config.json").write_bytes(b"{}")
    payload["operation"] = "verify"
    assert invoke()["verified"] is True
    (snapshot / "config.json").write_bytes(b"[]")
    with pytest.raises(ValueError, match="checksum"):
        invoke()
    (snapshot / "config.json").write_bytes(b"{}")
    payload["operation"] = "retire"
    assert invoke()["retired"] is True
    assert not snapshot.exists()

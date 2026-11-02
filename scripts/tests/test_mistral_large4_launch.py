"""CPU-only lifecycle contracts for the authorized GPU1 quant and publication."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "quantization"))
from mistral_large4.core_k6 import build_extension, container, launch, preflight, source_audit
from mistral_large4.core_k6.recipe import BASE_REPO, BASE_REVISION


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def environment(command):
    return {command[i + 1].split("=", 1)[0]: command[i + 1].split("=", 1)[1]
            for i, arg in enumerate(command[:-1]) if arg == "-e"}


def test_gpu_container_exposes_only_authorized_uuid_and_caps_memory():
    command = container.command(["run.py", "--resume"], name="owned-mistral")
    assert command[:2] == ["docker", "run"]
    assert command[command.index("--name") + 1] == "owned-mistral"
    assert command[command.index("--memory") + 1] == "78g"
    assert command[command.index("--memory-swap") + 1] == "78g"
    assert "all" not in command
    assert not any("gpu0" in value for value in command)
    devices = [command[i + 1] for i, value in enumerate(command[:-1]) if value in ("--gpus", "--device")]
    assert devices and all(container.GPU_UUID in value for value in devices)
    env = environment(command)
    assert env["CUDA_VISIBLE_DEVICES"] == container.GPU_UUID
    assert env["TORCH_CUDA_ARCH_LIST"] == "12.0"
    assert env["GPTQMODEL_EXLLAMAV3_BUILD_ROOT"] == str(container.BUILD / "exllamav3")
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert command[command.index("--network") + 1] == "none"
    assert command[-3:] == [container.IMAGE, "run.py", "--resume"]


def test_cpu_build_container_has_no_device_access():
    command = container.command(["build_extension.py"], name="compile-mistral", cpu=True)
    assert "--gpus" not in command
    assert "--device" not in command
    assert environment(command)["CUDA_VISIBLE_DEVICES"] == ""
    assert command[-2:] == [container.IMAGE, "build_extension.py"]


@pytest.fixture
def launch_arguments(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "REPO", tmp_path / "checkout")
    root = tmp_path / "run"
    arguments = ["launch.py", "--source-snapshot", str(tmp_path / "source"), "--tokens", str(tmp_path / "tokens"),
                 "--root", str(root), "--export", str(tmp_path / "export"),
                 "--source-audit", str(tmp_path / "audit.json"),
                 "--qualification-report", str(tmp_path / "qualification.json")]
    calls = []
    monkeypatch.setattr(launch.subprocess, "run", lambda cmd, **kwargs:
                        calls.append((list(cmd), kwargs)) or SimpleNamespace(returncode=0))
    return SimpleNamespace(root=root, argv=arguments, calls=calls, checkout=launch.REPO)


def invoke_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", argv)
    launch.main()


def load_attempt(root):
    latest = json.loads((root / "latest-attempt.json").read_text())
    return json.loads(Path(latest["launch"]).read_text())


def test_production_preflights_before_durable_systemd_launch_and_grant(launch_arguments, monkeypatch):
    fixture = launch_arguments
    invoke_main(monkeypatch, [*fixture.argv, "--publish"])
    assert len(fixture.calls) == 2
    admission, admission_kwargs = fixture.calls[0]
    systemd, _ = fixture.calls[1]
    assert admission[1].endswith("preflight.py")
    assert admission_kwargs["check"] is True
    assert "--qualification-report" in admission
    assert systemd[:2] == ["systemd-run", "--user"]
    assert "--collect" in systemd
    assert "--expand-environment=no" in systemd
    assert "--supervise" in systemd
    record = load_attempt(fixture.root)
    command = record["command"]
    assert command[1].endswith("scripts/hw/hwlock.py")
    assert command[command.index("--locks") + 1] == "gpu1"
    assert command[command.index("--wait") + 1] == "21600"
    assert shlex.split(command[command.index("--preflight") + 1]) == admission
    assert "30d" in command
    guard = next(value for value in command if value.endswith("run_guard.py"))
    runner = next(value for value in command if value.endswith("container.py"))
    assert command.index(guard) < command.index(runner)
    assert record["publication_command"][-1] == "--publish-only"
    assert not any(value.endswith("hwlock.py") for value in record["publication_command"])
    assert record["environment"]["CUDA_VISIBLE_DEVICES"] == ""


def test_default_production_does_not_implicitly_upload(launch_arguments, monkeypatch):
    invoke_main(monkeypatch, launch_arguments.argv)
    assert load_attempt(launch_arguments.root)["publication_command"] is None


def test_explicit_resume_reaches_execution_and_enqueue_grant_preflight(launch_arguments, monkeypatch):
    invoke_main(monkeypatch, [*launch_arguments.argv, "--resume", "--publish"])
    record = load_attempt(launch_arguments.root)
    assert "--resume" in record["command"]
    assert "--resume" in shlex.split(record["command"][record["command"].index("--preflight") + 1])
    assert "--resume" in record["publication_command"]


def test_publication_only_never_queues_hardware(launch_arguments, monkeypatch):
    argv = launch_arguments.argv
    invoke_main(monkeypatch, [*argv, "--publish-only"])
    record = load_attempt(launch_arguments.root)
    assert "--publication-only" in launch_arguments.calls[0][0]
    assert not any(value.endswith("hwlock.py") or value.endswith("container.py") for value in record["command"])
    assert record["command"][-1] == "--publish-only"
    assert record["publication_command"] is None


def test_short_pilot_bounded_one_hour_and_cannot_publish(launch_arguments, monkeypatch):
    argv = launch_arguments.argv
    index = argv.index("--export")
    argv[index] = "--output"
    invoke_main(monkeypatch, [*argv, "--pilot"])
    record = load_attempt(launch_arguments.root)
    assert "1h" in record["command"]
    assert any(value.endswith("pilot.py") for value in record["command"])
    assert record["publication_command"] is None
    with pytest.raises(SystemExit):
        invoke_main(monkeypatch, [*argv, "--pilot", "--publish"])


def test_failed_preflight_cannot_create_background_job(launch_arguments, monkeypatch):
    def reject(*args, **kwargs):
        raise launch.subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr(launch.subprocess, "run", reject)
    with pytest.raises(launch.subprocess.CalledProcessError):
        invoke_main(monkeypatch, launch_arguments.argv)
    assert not (launch_arguments.root / "latest-attempt.json").exists()


def test_live_supervisor_rejected_before_second_attempt(launch_arguments, monkeypatch):
    write_json(launch_arguments.root / "attempts/old-process.json", dict(supervisor_pid=os.getpid()))
    with pytest.raises(ValueError, match="remains active"):
        invoke_main(monkeypatch, launch_arguments.argv)
    assert not launch_arguments.calls


@pytest.fixture
def supervision(tmp_path, monkeypatch):
    events, handlers, outcomes, state = [], {}, [], {"last": {}}
    path = write_json(tmp_path / "attempt.json", dict(command=["hardware-command"],
        publication_command=["cpu-upload"], container="owned-container", environment={"PATH": "/bin"},
        log=str(tmp_path / "attempt.log"), root=str(tmp_path), pilot=False, publication_only=False))
    monkeypatch.setattr(launch.signal, "signal", lambda signum, function: handlers.__setitem__(signum, function))

    class Process:
        def __init__(self, command, **kwargs):
            self.command = list(command)
            self.outcome = outcomes.pop(0)
            state["last"] = self.outcome
            if self.outcome.get("raise"):
                raise self.outcome["raise"]
            self.returncode = None
            events.append(("start", self.command, kwargs))
            kwargs["stdout"].write(self.outcome.get("log", ""))
            kwargs["stdout"].flush()

        def wait(self):
            if self.outcome.get("signal"):
                handlers[self.outcome["signal"]](self.outcome["signal"], None)
            if self.outcome.get("category"):
                write_json(tmp_path / "failure.json", dict(category=self.outcome["category"],
                    time=self.outcome.get("failure_time", time.time_ns()),
                    attempt=self.outcome.get("attempt", "owned-container")))
            self.returncode = self.outcome["returncode"]
            events.append(("waited", self.command, self.returncode))
            return self.returncode

        def poll(self):
            return self.returncode

        def send_signal(self, sig):
            events.append(("signal", sig))

    monkeypatch.setattr(launch.subprocess, "Popen", Process)
    def run(command, **kwargs):
        if command[1] == "inspect":
            events.append(("inspect", list(command)))
            return SimpleNamespace(returncode=0, stdout=json.dumps({"OOMKilled": bool(state["last"].get("oom_killed"))}))
        events.append(("cleanup", list(command)))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(launch.subprocess, "run", run)
    return SimpleNamespace(path=path, events=events, outcomes=outcomes)


def test_upload_starts_only_after_successful_hardware_exit_and_cleanup(supervision):
    fixture = supervision
    fixture.outcomes.extend([dict(returncode=0), dict(returncode=0)])
    assert launch.supervise(fixture.path) == 0
    events = [entry[0] for entry in fixture.events]
    assert events == ["start", "waited", "cleanup", "inspect", "cleanup", "start", "waited"]
    upload = [entry for entry in fixture.events if entry[0] == "start"][1]
    assert upload[1] == ["cpu-upload"]
    assert upload[2]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert json.loads(fixture.path.with_name("attempt-exit.json").read_text())["returncode"] == 0


def test_standard_cuda_oom_resumes_same_command_without_discarding_cache(supervision):
    fixture = supervision
    fixture.outcomes.extend([dict(returncode=1, log="torch.OutOfMemoryError: CUDA out of memory\n", category="oom"),
                             dict(returncode=0), dict(returncode=0)])
    assert launch.supervise(fixture.path) == 0
    starts = [event[1] for event in fixture.events if event[0] == "start"]
    assert starts == [["hardware-command"], ["hardware-command", "--resume"], ["cpu-upload"]]
    assert "immutable producer unchanged" in Path(json.loads(fixture.path.read_text())["log"]).read_text()


@pytest.mark.parametrize("message", ["ValueError: nonfinite Hessian", "RuntimeError: not positive-definite", "ValueError: numerical contract violated", "ValueError: hessian_metric_status invalid"])
def test_numerical_failure_is_never_retried_or_published(supervision, message):
    fixture = supervision
    fixture.outcomes.append(dict(returncode=1, log=message + "\n", category="numerical"))
    assert launch.supervise(fixture.path) == 1
    assert len([event for event in fixture.events if event[0] == "start"]) == 1


def test_ambiguous_numerical_failure_after_oom_does_not_resume(supervision):
    fixture = supervision
    fixture.outcomes.append(dict(returncode=1,
        category="numerical", log="Recovered earlier CUDA out of memory\nValueError: non-finite quantized weights\n"))
    assert launch.supervise(fixture.path) == 1
    assert len([event for event in fixture.events if event[0] == "start"]) == 1


def test_cancellation_cannot_start_publication_even_if_child_exits_zero(supervision):
    fixture = supervision
    fixture.outcomes.append(dict(returncode=0, signal=signal.SIGTERM))
    assert launch.supervise(fixture.path) != 0
    assert len([event for event in fixture.events if event[0] == "start"]) == 1


def test_publication_exec_failure_records_unsuccessful_supervisor_exit(supervision):
    fixture = supervision
    fixture.outcomes.extend([dict(returncode=0), {"raise": FileNotFoundError("publisher executable missing")}])
    with pytest.raises(FileNotFoundError):
        launch.supervise(fixture.path)
    record = json.loads(fixture.path.with_name("attempt-exit.json").read_text())
    assert record["returncode"] != 0
    assert "FileNotFoundError" in record["error"]


def test_cleanup_error_after_quantization_cannot_record_success(supervision, monkeypatch):
    fixture = supervision
    fixture.outcomes.append(dict(returncode=0))
    def fail(*args, **kwargs):
        raise TimeoutError("container cleanup did not finish")
    monkeypatch.setattr(launch.subprocess, "run", fail)
    with pytest.raises(TimeoutError):
        launch.supervise(fixture.path)
    record = json.loads(fixture.path.with_name("attempt-exit.json").read_text())
    assert record["returncode"] != 0
    assert "TimeoutError" in record["error"]
    assert len([event for event in fixture.events if event[0] == "start"]) == 1


def test_oom_resume_is_bounded(supervision):
    fixture = supervision
    fixture.outcomes.extend([dict(returncode=1, log="MemoryError: host allocation\n", category="oom") for _ in range(3)])
    assert launch.supervise(fixture.path) == 1
    starts = [event[1] for event in fixture.events if event[0] == "start"]
    assert starts == [["hardware-command"], ["hardware-command", "--resume"], ["hardware-command", "--resume"]]


def test_cgroup_oom_kill_resumes_without_requiring_python_traceback(supervision):
    fixture = supervision
    fixture.outcomes.extend([dict(returncode=137, oom_killed=True), dict(returncode=0), dict(returncode=0)])
    assert launch.supervise(fixture.path) == 0
    starts = [event[1] for event in fixture.events if event[0] == "start"]
    assert starts == [["hardware-command"], ["hardware-command", "--resume"], ["cpu-upload"]]
    state = json.loads(fixture.path.with_name("attempt-container-0.json").read_text())
    assert state["OOMKilled"] is True


@pytest.mark.parametrize("damage", ["stale", "other_attempt", "unclassified_sigkill"])
def test_stale_or_ambiguous_oom_cannot_autoretry(supervision, damage):
    fixture = supervision
    outcome = dict(returncode=137)
    if damage == "stale":
        outcome.update(category="oom", failure_time=1)
    elif damage == "other_attempt":
        outcome.update(category="oom", attempt="some-other-container")
    fixture.outcomes.append(outcome)
    assert launch.supervise(fixture.path) == 137
    assert len([event for event in fixture.events if event[0] == "start"]) == 1


def test_numerical_failure_prevents_retry_even_if_container_reports_oom(supervision):
    fixture = supervision
    fixture.outcomes.append(dict(returncode=1, category="numerical", oom_killed=True))
    assert launch.supervise(fixture.path) == 1
    assert len([event for event in fixture.events if event[0] == "start"]) == 1


@pytest.mark.parametrize("host", ["raptor", "raptor (D2)", "raptor (D3)"])
def test_local_source_accepts_union_of_sealed_physical_raptor_disks(tmp_path, monkeypatch, host):
    snapshot = tmp_path / "snapshots" / BASE_REVISION
    snapshot.mkdir(parents=True)
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "blob-name").write_bytes(b"source")
    (snapshot / "config.json").symlink_to(blobs / "blob-name")
    row = dict(name="blob-name", kind="file", sealed=True, writing_on=None, hosts=[host])
    calls = []
    monkeypatch.setattr(source_audit.subprocess, "check_output", lambda command, **kwargs:
                        calls.append(command) or json.dumps(dict(entries=[row])))
    assert source_audit.local_source(snapshot)
    assert calls[0] == ["nest", "ls", "--json", str(blobs)]


@pytest.mark.parametrize("damage", ["remote", "unsealed", "writing", "missing"])
def test_local_source_rejects_unready_or_only_remote_data(tmp_path, monkeypatch, damage):
    snapshot = tmp_path / "snapshots" / BASE_REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_bytes(b"source")
    row = dict(name="config.json", kind="file", sealed=True, writing_on=None, hosts=["raptor"])
    if damage == "remote":
        row["hosts"] = ["maxq"]
    elif damage == "unsealed":
        row["sealed"] = False
    elif damage == "writing":
        row["writing_on"] = "raptor"
    monkeypatch.setattr(source_audit.subprocess, "check_output", lambda *args, **kwargs:
                        json.dumps(dict(entries=[] if damage == "missing" else [row])))
    with pytest.raises(ValueError, match="not fully sealed"):
        source_audit.local_source(snapshot)


def test_public_source_hash_audit_checks_all_68_files(tmp_path, monkeypatch):
    snapshot = tmp_path / BASE_REVISION
    snapshot.mkdir()
    files = {}
    for index in range(68):
        name = f"part-{index:02d}.bin"
        payload = ("source-" + str(index)).encode()
        (snapshot / name).write_bytes(payload)
        if index % 2:
            files[name] = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        else:
            files[name] = dict(bytes=len(payload), git_blob_sha1=hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest())
    manifest = write_json(tmp_path / "manifest.json", dict(repo_id=BASE_REPO, revision=BASE_REVISION, files=files))
    output = tmp_path / "audit.json"
    monkeypatch.setattr(source_audit, "local_source", lambda path: True)
    monkeypatch.setattr(sys, "argv", ["source_audit.py", "--snapshot", str(snapshot), "--hub-manifest", str(manifest), "--output", str(output)])
    source_audit.main()
    audit = json.loads(output.read_text())
    assert audit["status"] == "passed" and len(audit["files"]) == 68
    assert audit["revision"] == BASE_REVISION
    assert all(len(record["sha256"]) == 64 for record in audit["files"].values())
    (snapshot / "part-00.bin").write_bytes(b"same-len")
    output.unlink()
    with pytest.raises(ValueError, match="checksum|size"):
        source_audit.main()
    assert not output.exists()


def test_pilot_attestation_rejects_code_change_or_missing_replay_proofs(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "REPO", tmp_path)
    numerical = tmp_path / "quantization/mistral_large4/core_k6/native.py"
    numerical.parent.mkdir(parents=True)
    numerical.write_bytes(b"frozen numerical producer")
    report = dict(status="passed", identity=dict(gptqmodel_commit="fork-pin", code={
        str(numerical.relative_to(tmp_path)): hashlib.sha256(numerical.read_bytes()).hexdigest()}),
        native_boundaries_byte_exact=True, serialized_projection_replay=True)
    path = write_json(tmp_path / "pilot.json", report)
    preflight.qualified(path, "fork-pin")
    numerical.write_bytes(b"modified numerical producer")
    with pytest.raises(ValueError, match="source changed"):
        preflight.qualified(path, "fork-pin")


@pytest.mark.parametrize("damage", ["payload", "retargeted_blob", "asset"])
def test_preflight_binds_all_68_hashed_source_objects(tmp_path, monkeypatch, damage):
    snapshot = tmp_path / BASE_REVISION
    snapshot.mkdir()
    names = ["config.json", "consolidated.safetensors.index.json", "tekken.json", "tokenizer_config.json"]
    names += [f"part-{i:02d}.safetensors" for i in range(64)]
    records = {}
    for name in names:
        path = snapshot / name
        path.write_bytes((name + " original").encode())
        records[name] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             file_identity=source_audit.source_identity(path))
    audit = write_json(tmp_path / "audit.json", dict(status="passed", source=str(snapshot),
        repo_id=BASE_REPO, revision=BASE_REVISION, files=records))
    checked = []
    monkeypatch.setattr(preflight, "local_source", lambda path: checked.append(path) or True)
    preflight.verify_source(snapshot, audit)
    assert checked == [snapshot]
    target = snapshot / ("tokenizer_config.json" if damage == "asset" else "part-63.safetensors")
    if damage == "retargeted_blob":
        replacement = tmp_path / "new-sealed-blob"
        replacement.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(replacement)
    else:
        target.write_bytes(b"modified after source hash audit")
    with pytest.raises(ValueError, match="source.*changed"):
        preflight.verify_source(snapshot, audit)


@pytest.mark.parametrize("damage", ["host", "dirty", "fork_pin", "tree"])
def test_preflight_rejects_wrong_host_uncommitted_code_or_fork_identity(tmp_path, monkeypatch, damage):
    monkeypatch.setattr(preflight, "REPO", tmp_path)
    monkeypatch.setattr(preflight.socket, "gethostname", lambda: "maxq" if damage == "host" else "raptor")
    write_json(tmp_path / "third_party/gptqmodel.lock.json", dict(revision="fork-pin", source_tree_sha256="tree-hash"))
    monkeypatch.setattr(preflight.runpy, "run_path", lambda path:
                        dict(source_tree_sha256=lambda root: "changed" if damage == "tree" else "tree-hash"))
    def git(command, **kwargs):
        if "status" in command:
            return " M producer.py\n" if damage == "dirty" else ""
        return "different-pin\n" if damage == "fork_pin" else "fork-pin\n"
    monkeypatch.setattr(preflight.subprocess, "check_output", git)
    with pytest.raises(ValueError):
        preflight.verify_code()


def test_extension_compiler_records_exact_binary_identity_without_gpu_lease(tmp_path, monkeypatch):
    root = tmp_path / "exllamav3/fingerprint"
    binary = root / "ops.so"
    calls = []
    extension = SimpleNamespace(name="ops", build_root=lambda: root, sources=["native.cpp", "native.cu"],
        extra_ldflags=["-lcuda"], _resolve_sequence=list,
        _resolved_extra_cflags=lambda: ["-O3"], _resolved_extra_cuda_cflags=lambda: ["-O3"],
        _resolved_extra_include_paths=lambda: ["include"])
    monkeypatch.setattr(build_extension, "fork_module", lambda name:
                        SimpleNamespace(_EXLLAMAV3_TORCH_OPS_EXTENSION=extension))
    fake = ModuleType("torch.utils.cpp_extension")
    def compile(**kwargs):
        calls.append(kwargs)
        binary.write_bytes(b"compiled exact fork")
    fake.load = compile
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", fake)
    monkeypatch.setattr(build_extension.subprocess, "check_output", lambda *args, **kwargs: "fork-pin\n")
    build_extension.main()
    assert calls[0]["sources"] == ["native.cpp", "native.cu"]
    assert calls[0]["is_python_module"] is False
    receipt = json.loads((root.parent / "build-receipt.json").read_text())
    assert receipt["fork_commit"] == "fork-pin"
    assert receipt["sha256"] == hashlib.sha256(binary.read_bytes()).hexdigest()
    assert receipt["binary"] == str(binary)


def test_any_fresh_non_oom_failure_overrides_ambiguous_container_oom_state(supervision):
    fixture=supervision
    fixture.outcomes.append(dict(returncode=1,category='other',oom_killed=True))
    assert launch.supervise(fixture.path)==1
    assert len([event for event in fixture.events if event[0]=='start'])==1

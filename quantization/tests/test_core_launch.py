"""CPU-only contract tests: no SSH, container, hardware queue or producer runs."""
import hashlib
import importlib
import json
from pathlib import Path
import shlex
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import core_launch as launch


def record(tmp_path, family="glmf", host="rhea"):
    build = tmp_path / ("mc-" + family)
    return dict(family=family, host=host, commit="committed", unit="mc-test", container="mc-test",
                uid=1234, gid=2345, user="remote-user",
                remote_source_root=str(tmp_path / "source"), build_root=str(build),
                runtime_python="python3", runtime_root=str(tmp_path / "source"),
                source_snapshot="/mnt/sparknest/hf-home/hub/model/snapshots/revision",
                tokens=str(build / "tokens.json"), root=str(build / "state"),
                export=str(build / "export"), resume=True, image_id="sha256:pinned")


@pytest.mark.parametrize("host,locks", [("rhea", "rhea"), ("moa", "moa"),
                                         ("cassowary", "cassowary,cassowary-export")])
def test_canonical_host_locks(host, locks):
    assert launch.locks_for(host) == locks


@pytest.mark.parametrize("host", sorted(launch.POOL))
def test_pool_refused_without_sparks_lock(host):
    with pytest.raises(ValueError, match=r"requires sparks.lock"):
        launch.locks_for(host)


@pytest.mark.parametrize("family", launch.FAMILIES)
@pytest.mark.parametrize("preflight", [False, True])
def test_container_is_bounded_user_sm121_readonly_source(tmp_path, family, preflight):
    row = record(tmp_path, family)
    command = launch.container_command(row, preflight=preflight)
    assert command[:3] == ["docker", "run", "--rm"]
    cap = "110g" if family in ("kimi-k31", "muse") else "96g"
    assert command[command.index("--memory") + 1] == cap
    assert command[command.index("--memory-swap") + 1] == cap
    assert "--user" in command and "--privileged" not in command
    assert "CUTEAFD_DEV_MEMORY=" + cap in command and "TORCH_CUDA_ARCH_LIST=12.1" in command
    assert "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0" in command
    assert "NVIDIA_TF32_OVERRIDE=0" in command
    assert "--network" in command and command[command.index("--network") + 1] == "none"
    assert command[command.index("--entrypoint") + 1] == "python3"
    assert command[command.index("--entrypoint") + 2] == row["image_id"]
    assert bool("--gpus" in command) is not preflight
    assert bool("--preflight" in command) is preflight
    assert "CUDA_VISIBLE_DEVICES=" + ("" if preflight else "0") in command
    assert command[command.index("--core-bits") + 1] == "8"
    assert "--core-only" in command and "--resume" in command
    assert "--publish" not in command and "--publish-only" not in command
    assert any("/" + launch.FAMILIES[family] + "/run.py" in arg for arg in command)
    mounts = [command[i + 1] for i, arg in enumerate(command) if arg == "--mount"]
    assert "type=bind,src=/mnt/sparknest,dst=/mnt/sparknest,readonly" in mounts
    assert f'type=bind,src={row["remote_source_root"]},dst={row["remote_source_root"]},readonly' in mounts
    assert not any("src=/home/tj,dst=/home/tj" in value for value in mounts)


def test_runtime_overlay_is_explicit_readonly(tmp_path):
    row = record(tmp_path)
    row.update(runtime_python="/overlay/bin/python", runtime_root="/overlay")
    command = launch.container_command(row)
    assert command[command.index("--entrypoint") + 1] == "/overlay/bin/python"
    assert "type=bind,src=/overlay,dst=/overlay,readonly" in command


@pytest.mark.parametrize("family", launch.FAMILIES)
def test_family_launch_early_dispatch_only(tmp_path, monkeypatch, family):
    module = importlib.import_module(launch.FAMILIES[family].replace("/", ".") + ".launch")
    calls = []
    monkeypatch.setattr(launch, "dispatch", lambda name: calls.append(name) or True)
    # Reaching old argument parsing, or any subprocess, is an error.
    monkeypatch.setattr(module.argparse.ArgumentParser, "parse_args", lambda *a, **k: pytest.fail("legacy parser"))
    module.main()
    assert calls == [family]


@pytest.mark.parametrize("family,digest", [
    ("kimi-k31", "a9ab3069888a0b8e95177f368e10f69a59f62bc78d56f0447ac9c360ae46b2e6"),
    ("glmf", "19bddc8c5ff3a165bad4f4191a4775569cf8a3a2cd60d66820185dfa1de8f7e9"),
    ("muse", "696c2b88d6a1902cdd78c8796d4029d9f593a1b6b05256765d163e896572206d"),
])
def test_legacy_launch_path_byte_identical(family, digest):
    path = launch.REPO / "quantization" / launch.FAMILIES[family] / "launch.py"
    text = path.read_text()
    addition = f"    from core_launch import dispatch\n    if dispatch({family!r}):\n        return\n"
    assert text.count(addition) == 1
    assert hashlib.sha256(text.replace(addition, "", 1).encode()).hexdigest() == digest


def test_legacy_selectors_do_not_dispatch(monkeypatch):
    monkeypatch.setattr(launch, "main", lambda *a, **k: pytest.fail("unexpected core dispatch"))
    for arguments in ([], ["--source-snapshot", "/source", "--tokens", "/tokens", "--root", "/root",
                           "--export", "/export", "--resume"], ["--pilot"], ["--supervise", "/attempt"]):
        assert launch.dispatch("glmf", arguments) is False


@pytest.mark.parametrize("flag", ["--core-only", "--core-bits=8", "--host=rhea",
                                   "--remote-source-root=/source", "--runtime-python=python3"])
def test_new_selectors_dispatch(monkeypatch, flag):
    calls = []
    monkeypatch.setattr(launch, "main", lambda argv, **kwargs: calls.append((argv, kwargs)))
    assert launch.dispatch("glmf", [flag]) is True
    assert calls == [([flag], dict(family="glmf"))]


def test_missing_core_only_and_non_k8_rejected_before_commands(monkeypatch):
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **k: pytest.fail("must reject before commands"))
    with pytest.raises(SystemExit):
        launch.main(["--host", "rhea"], family="muse")
    with pytest.raises(SystemExit):
        launch.main(["--core-only", "--core-bits", "6"], family="muse")


def test_supervisor_preflights_before_enqueue_and_grant_and_marks_success(tmp_path, monkeypatch):
    path = tmp_path / "attempt.json"
    path.write_text(json.dumps(record(tmp_path, host="cassowary")))
    calls = []
    monkeypatch.setattr(launch, "forwarding", lambda cmd, **kw: calls.append(cmd) or 0)
    assert launch.supervise(path) == 0
    command = calls[0]
    assert command[command.index("--locks") + 1] == "cassowary,cassowary-export"
    assert command[command.index("--wait") + 1] == "21600"
    assert "--preflight" in command and "--preflight-at-grant" not in command
    check = shlex.split(command[command.index("--preflight") + 1])
    assert "--remote-check" in check and "--execute" in command
    assert path.with_suffix(".done.json").is_file()
    assert json.loads(path.with_suffix(".exit.json").read_text())["returncode"] == 0


def test_failure_never_writes_done_marker(tmp_path, monkeypatch):
    path = tmp_path / "attempt.json"
    path.write_text(json.dumps(record(tmp_path)))
    monkeypatch.setattr(launch, "forwarding", lambda *a, **kw: 78)
    assert launch.supervise(path) == 78
    assert not path.with_suffix(".done.json").exists()


def test_execute_guard_and_remote_cleanup_precede_return(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "require_lease", lambda *a: None)
    events = []
    monkeypatch.setattr(launch, "forwarding", lambda command, **kw: events.append(("run", command)) or 0)
    monkeypatch.setattr(launch.subprocess, "run", lambda command, **kw: events.append(("cleanup", command)))
    monkeypatch.setattr(launch.subprocess, "check_output", lambda command, **kw: events.append(("probe", command)) or "")
    row = record(tmp_path)
    assert launch.execute(row) == 0
    assert [event[0] for event in events] == ["run", "cleanup", "probe"]
    remote = shlex.split(events[0][1][-1])
    assert remote[:2] == ["timeout", "--foreground"] and str(launch.RUN_SECONDS) in remote
    assert any(value.endswith("run_guard.py") for value in remote)
    assert remote[remote.index("--container") + 1] == row["container"]
    assert "docker rm --force mc-test" == events[1][1][-1]


@pytest.mark.parametrize("problem", ["head", "dirty", "uninitialized", "wrong-pin", "nested-dirty"])
def test_source_refuses_uncommitted_or_unpinned_checkout(tmp_path, monkeypatch, problem):
    answers = ["other" if problem == "head" else "commit",
               " M run.py" if problem == "dirty" else "",
               " M nested.py" if problem == "nested-dirty" else ""]
    monkeypatch.setattr(launch, "output", lambda *a, **k: answers.pop(0))
    prefix = "-" if problem == "uninitialized" else "+" if problem == "wrong-pin" else " "
    monkeypatch.setattr(launch.subprocess, "check_output", lambda *a, **k: prefix + "pin submodule\n")
    with pytest.raises(ValueError):
        launch.checked_source(tmp_path, "commit")


def test_launch_uses_transient_unit_without_daemon_reload(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(launch.socket, "gethostname", lambda: "raptor")
    monkeypatch.setattr(launch, "checked_source", lambda *a: "clean-commit")
    monkeypatch.setattr(launch.Path, "home", lambda: tmp_path)
    monkeypatch.delenv("CUTEAFD_DEV_MEMORY", raising=False)
    commands = []
    def fake_output(command, **kw):
        assert command[0] == "ssh"
        if "image inspect" in command[-1]:
            return "sha256:pinned"
        return json.dumps(dict(home=str(tmp_path), uid=1234, gid=2345, user="tj"))
    monkeypatch.setattr(launch, "output", fake_output)
    monkeypatch.setattr(launch.subprocess, "run", lambda command, **kw: commands.append(command))
    build = tmp_path / ".cache/cuteafd/builds/mc-glmf"
    arguments = ["--core-only", "--core-bits", "8", "--host", "moa", "--source-snapshot",
                 "/mnt/sparknest/source", "--tokens", str(build / "tokens"), "--root", str(build / "state"),
                 "--export", str(build / "export"), "--resume", "--remote-source-root", "/staged/source"]
    assert launch.main(arguments, family="glmf") == 0
    receipt = json.loads(capsys.readouterr().out)
    row = json.loads(Path(receipt["launch"]).read_text())
    assert row["commit"] == "clean-commit" and row["uid"] == 1234
    assert row["resume"] is True and row["host"] == "moa"
    assert commands[0][0] == "ssh" and "--validate-remote" in commands[0][-1]
    assert commands[-1][:2] == ["systemd-run", "--user"]
    assert "--collect" in commands[-1] and "--property=KillMode=mixed" in commands[-1]
    assert any(value.startswith("--property=RuntimeMaxSec=") for value in commands[-1])
    assert not any("daemon-reload" in value for command in commands for value in command)
    assert not Path(receipt["done"]).exists()


@pytest.mark.parametrize("cap", ["128g", "121g", "0", "100g"])
def test_oversized_or_uncapped_memory_override_refused(tmp_path, monkeypatch, cap):
    monkeypatch.setattr(launch.socket, "gethostname", lambda: "raptor")
    monkeypatch.setenv("CUTEAFD_DEV_MEMORY", cap)
    monkeypatch.setattr(launch, "checked_source", lambda *a: pytest.fail("must reject before source validation"))
    with pytest.raises(ValueError, match="110g"):
        launch.main(["--core-only", "--host", "rhea", "--source-snapshot", "/mnt/sparknest/source",
                     "--tokens", "/tokens", "--root", "/root", "--export", "/export"], family="muse")


def test_validate_remote_refuses_output_escape_and_non_sparknest_source(tmp_path, monkeypatch):
    build = tmp_path / ".cache/cuteafd/builds/mc-glmf"
    source = tmp_path / "source"
    (source / ".git").mkdir(parents=True)
    monkeypatch.setattr(launch.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(launch, "REPO", source)
    monkeypatch.setattr(launch.socket, "gethostname", lambda: "rhea")
    monkeypatch.setattr(launch, "checked_source", lambda *a: "committed")
    monkeypatch.setattr(launch.os, "getuid", lambda: 1234)
    monkeypatch.setattr(launch.os, "getgid", lambda: 2345)
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **k: pytest.fail("refuse before subprocess"))
    row = record(tmp_path)
    row.update(build_root=str(build), root=str(tmp_path / "escape"), export=str(build / "export"))
    with pytest.raises(ValueError, match="root must stay"):
        launch.validate_remote(row)
    row["root"] = str(build / "state")
    row["source_snapshot"] = str(tmp_path / "source-model")
    with pytest.raises(ValueError, match="stream from /mnt/sparknest"):
        launch.validate_remote(row)


def test_execute_requires_real_enclosing_flock(tmp_path, monkeypatch):
    directory = tmp_path / '.cache/cuteafd'
    directory.mkdir(parents=True)
    monkeypatch.setattr(launch.Path, 'home', lambda: tmp_path)
    monkeypatch.delenv('CUTEAFD_HW_LOCKS_HELD', raising=False)
    with pytest.raises(ValueError, match='enclosing raptor hwlock'):
        launch.require_lease('rhea')
    monkeypatch.setenv('CUTEAFD_HW_LOCKS_HELD', 'rhea')
    with pytest.raises(ValueError, match='not held'):
        launch.require_lease('rhea')
    with (directory / 'rhea.lock').open('a') as lock:
        launch.fcntl.flock(lock, launch.fcntl.LOCK_EX | launch.fcntl.LOCK_NB)
        launch.require_lease('rhea')


def test_cleanup_failure_cannot_report_success(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, 'require_lease', lambda *a: None)
    monkeypatch.setattr(launch, 'forwarding', lambda *a, **kw: 0)
    monkeypatch.setattr(launch.subprocess, 'run', lambda *a, **kw: None)
    monkeypatch.setattr(launch.subprocess, 'check_output', lambda *a, **kw: 'still-live-container')
    with pytest.raises(RuntimeError, match='remains after cleanup'):
        launch.execute(record(tmp_path))


@pytest.mark.parametrize('family', launch.FAMILIES)
def test_transformers_source_only_for_hf5_families(tmp_path, family):
    row = record(tmp_path, family)
    command = launch.container_command(row)
    pythonpath = next(arg for arg in command if arg.startswith('PYTHONPATH='))
    assert ('third_party/transformers/src' in pythonpath) is (family != 'kimi-k31')


def test_preflight_accepts_corpus_directory_not_file(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    (source / '.git').mkdir(parents=True)
    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    build = tmp_path / '.cache/cuteafd/builds/mc-glmf'
    tokens = build / 'tokens'
    tokens.mkdir(parents=True)
    monkeypatch.setattr(launch.Path, 'home', lambda: tmp_path)
    monkeypatch.setattr(launch, 'REPO', source)
    monkeypatch.setattr(launch.socket, 'gethostname', lambda: 'rhea')
    monkeypatch.setattr(launch, 'checked_source', lambda *a: 'committed')
    monkeypatch.setattr(launch.os, 'getuid', lambda: 1234)
    monkeypatch.setattr(launch.os, 'getgid', lambda: 2345)
    original_beneath = launch.beneath
    monkeypatch.setattr(launch, 'beneath', lambda path, parent: (
        path == snapshot and parent == Path('/mnt/sparknest')) or original_beneath(path, parent))
    monkeypatch.setattr(launch, 'output', lambda command: 'sha256:pinned' if command[0] == 'docker' else '0:1')
    commands = []
    monkeypatch.setattr(launch.subprocess, 'run', lambda command, **kw: commands.append(command))
    row = record(tmp_path)
    row.update(build_root=str(build), root=str(build / 'state'), export=str(build / 'export'),
               tokens=str(tokens), source_snapshot=str(snapshot))
    launch.validate_remote(row)
    assert any('--preflight' in command for command in commands)
    tokens.rmdir()
    tokens.write_text('not a corpus directory')
    with pytest.raises(ValueError, match='corpus directory'):
        launch.validate_remote(row)


def test_explicit_runtime_keeps_its_transformers_release(tmp_path):
    row = record(tmp_path, 'muse')
    row.update(runtime_python='/overlay/bin/python', runtime_root='/overlay')
    command = launch.container_command(row)
    assert not any('third_party/transformers/src' in arg for arg in command)


def test_remote_start_is_bounded_detached_and_idempotent(tmp_path, monkeypatch):
    row = record(tmp_path)
    monkeypatch.setattr(launch.socket, 'gethostname', lambda: row['host'])
    monkeypatch.setattr(launch, 'checked_source', lambda *a: row['commit'])
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return type('Result', (), {'returncode': 0})()
    monkeypatch.setattr(launch.subprocess, 'run', run)
    launch.remote_start(row)
    command = calls[0]
    assert command[:2] == ['systemd-run', '--user']
    assert '--remote-worker' in command
    assert '--property=KillMode=mixed' in command
    assert '--property=RuntimeMaxSec=' + str(launch.RUN_SECONDS + 300) in command
    launch.remote_start(row)
    assert len([c for c in calls if c[0]=='systemd-run']) == 1
    changed = dict(row, resume=False)
    with pytest.raises(ValueError, match='identity differs'):
        launch.remote_start(changed)


def test_remote_wait_returns_receipt_without_conflating_exit_255(tmp_path, capsys):
    row = record(tmp_path)
    _, terminal = launch.remote_paths(row)
    terminal.parent.mkdir(parents=True)
    terminal.write_text(json.dumps(dict(unit=row['unit'], returncode=255)))
    assert launch.remote_wait(row) == 0
    assert json.loads(capsys.readouterr().out)['returncode'] == 255


def test_remote_worker_writes_terminal_after_container_cleanup(tmp_path, monkeypatch):
    row = record(tmp_path)
    path, terminal = launch.remote_paths(row)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(row))
    events = []
    monkeypatch.setattr(launch, 'forwarding', lambda *a, **kw: events.append('run') or 42)
    monkeypatch.setattr(launch.subprocess, 'run', lambda *a, **kw: events.append('cleanup'))
    monkeypatch.setattr(launch, 'output', lambda *a: '')
    assert launch.remote_worker(path) == 42
    assert events == ['run', 'cleanup']
    assert json.loads(terminal.read_text())['returncode'] == 42


@pytest.mark.parametrize('codes', [[0], [255, 0]])
def test_detached_execute_reconnects_without_removing_container(tmp_path, monkeypatch, codes):
    import io
    row = dict(record(tmp_path), remote_supervisor=True)
    monkeypatch.setattr(launch, 'require_lease', lambda *a: None)
    calls = []
    monkeypatch.setattr(launch.subprocess, 'run', lambda command, **kw: calls.append(command) or type('R', (), {'returncode':0})())
    pending = iter(codes)
    class Process:
        def __init__(self, *a, **kw):
            self.returncode = next(pending)
            self.stdout = io.StringIO(json.dumps(dict(unit=row['unit'],returncode=17)))
        def poll(self): return self.returncode
    monkeypatch.setattr(launch.subprocess, 'Popen', Process)
    monkeypatch.setattr(launch.time, 'sleep', lambda *a: None)
    assert launch.execute(row) == 17
    assert len(calls) == 1
    assert '--remote-start' in calls[0][-1]
    assert 'docker rm' not in calls[0][-1]


def test_detached_cancel_stops_only_owned_unit_and_container(tmp_path, monkeypatch):
    import signal
    row = dict(record(tmp_path), remote_supervisor=True)
    monkeypatch.setattr(launch, 'require_lease', lambda *a: None)
    handlers = {}
    monkeypatch.setattr(launch.signal, 'signal', lambda sig, fn: handlers.setdefault(sig,fn))
    calls=[]
    def run(command, **kw):
        calls.append(command)
        if '--remote-start' in command[-1]: handlers[signal.SIGTERM](signal.SIGTERM,None)
        return type('R', (), {'returncode':0})()
    monkeypatch.setattr(launch.subprocess, 'run', run)
    monkeypatch.setattr(launch, 'output', lambda *a: '')
    assert launch.execute(row) == 128+signal.SIGTERM
    assert 'systemctl --user stop mc-test' == calls[1][-1]
    assert 'docker rm --force mc-test' == calls[2][-1]


def test_detached_launch_is_selected_for_new_records(tmp_path, monkeypatch):
    row = dict(record(tmp_path), remote_supervisor=True)
    monkeypatch.setattr(launch, 'execute_detached', lambda value: 23 if value is row else 1)
    assert launch.execute(row) == 23


def test_lost_remote_start_reply_retries_same_unit_without_cleanup(tmp_path, monkeypatch):
    import io
    row = dict(record(tmp_path), remote_supervisor=True)
    monkeypatch.setattr(launch, 'require_lease', lambda *a: None)
    starts = iter([255, 0])
    calls=[]
    def run(command, **kw):
        calls.append(command)
        return type('R', (), {'returncode':next(starts)})()
    monkeypatch.setattr(launch.subprocess, 'run', run)
    class Process:
        returncode=0
        stdout=io.StringIO(json.dumps(dict(unit=row['unit'],returncode=0)))
        def __init__(self,*a,**kw):pass
        def poll(self):return 0
    monkeypatch.setattr(launch.subprocess, 'Popen', Process)
    monkeypatch.setattr(launch.time, 'sleep', lambda *a: None)
    assert launch.execute(row)==0
    assert len(calls)==2 and calls[0]==calls[1]

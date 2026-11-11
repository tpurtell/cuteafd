"""Opt-in, bounded SM121 core-only jobs; legacy family launch paths stay separate.

Source and runtime provisioning are deliberately external. --remote-source-root
must already contain this exact clean commit with recursively pinned submodules.
No image build, package installation, source replication or publication occurs.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
IMAGE = "cuteafd-spark-expert-dev:tc-ba6d37fa779d93e8"
FAMILIES = {
    "kimi-k31": "kimi_k31/k2_core_k6",
    "glmf": "glm5_flash/core_k6",
    "muse": "muse_spark/core_k6",
}
HOST_LOCKS = {"rhea": "rhea", "moa": "moa", "cassowary": "cassowary,cassowary-export"}
POOL = {"ostrich", "dodo", "emu", "kiwi"}
WAIT_SECONDS = 21600
RUN_SECONDS = 72 * 3600
SSH_OPTIONS = ["-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
               "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3"]


def ssh(host, command):
    return ["ssh", *SSH_OPTIONS, host, shlex.join(list(map(str, command)))]


def output(command, **kwargs):
    return subprocess.check_output(list(map(str, command)), text=True, timeout=60, **kwargs).strip()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def checked_source(root, expected=None):
    head = output(["git", "-C", root, "rev-parse", "HEAD"])
    if expected is not None and head != expected:
        raise ValueError("pre-staged remote source HEAD differs from launch commit")
    if output(["git", "-C", root, "status", "--porcelain", "--untracked-files=normal"]):
        raise ValueError("source checkout must be clean and committed")
    modules = subprocess.check_output(["git", "-C", str(root), "submodule", "status", "--recursive"],
                                      text=True, timeout=120)
    if not modules or any(not line.startswith(" ") for line in modules.splitlines()):
        raise ValueError("initialize all recursively pinned submodules before launch")
    dirty = output(["git", "-C", root, "submodule", "foreach", "--quiet", "--recursive",
                    "git status --porcelain --untracked-files=normal"])
    if dirty:
        raise ValueError("submodule source is dirty")
    return head


def beneath(path, parent):
    return path == parent or parent in path.parents


def locks_for(host):
    if host in POOL:
        raise ValueError("pool Spark requires sparks.lock; this launcher does not acquire it")
    if host not in HOST_LOCKS:
        raise ValueError("core-only host must be rhea, moa or cassowary")
    return HOST_LOCKS[host]


def common(record):
    result = ["--core-only", "--core-bits", "8"]
    for name in ("source_snapshot", "tokens", "root", "export"):
        result += ["--" + name.replace("_", "-"), record[name]]
    if record["resume"]:
        result.append("--resume")
    return result


def container_command(record, *, preflight=False):
    source = Path(record["remote_source_root"])
    build = Path(record["build_root"])
    memory = "110g" if record["family"] in ("kimi-k31", "muse") else "96g"
    result = ["docker", "run", "--rm", "--name", record["container"] + ("-check" if preflight else ""),
              "--network", "none", "--user", f"{record['uid']}:{record['gid']}",
              "--memory", memory, "--memory-swap", memory, "--stop-timeout", "30",
              "--ulimit", "nofile=65536:65536", "--workdir", str(source),
              "--mount", f"type=bind,src={source},dst={source},readonly",
              "--mount", "type=bind,src=/mnt/sparknest,dst=/mnt/sparknest,readonly",
              "--mount", f"type=bind,src={build},dst={build}"]
    runtime = record["runtime_python"]
    # An optional HF4 overlay is provisioned separately, never pip-installed here.
    if runtime != "python3":
        environment_root = Path(record["runtime_root"])
        if not beneath(environment_root, source) and not beneath(environment_root, build):
            result += ["--mount", f"type=bind,src={environment_root},dst={environment_root},readonly"]
    if not preflight:
        result += ["--gpus", "all"]
    environment = dict(HOME=str(build / "home"), USER=record["user"], LOGNAME=record["user"],
        CUTEAFD_DEV_MEMORY=memory, CUDA_VISIBLE_DEVICES="" if preflight else "0",
        NVIDIA_VISIBLE_DEVICES="void" if preflight else "all",
        # NGC enables TF32 through an environment override even with Python flags off.
        TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0", NVIDIA_TF32_OVERRIDE="0",
        TORCH_CUDA_ARCH_LIST="12.1", CUDA_HOME="/usr/local/cuda", MAX_JOBS="8",
        OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8",
        USE_HUB_KERNELS="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        HF_HOME="/mnt/sparknest/hf-home", HF_MODULES_CACHE=str(build / "hf-modules"),
        PYTHONDONTWRITEBYTECODE="1", PYTORCH_ALLOC_CONF="expandable_segments:True",
        PYTHONPATH=str(source / "third_party/gptqmodel") + ":" + str(source / "quantization"),
        GLMF_GPTQMODEL_SOURCE=str(source / "third_party/gptqmodel"),
        GPTQMODEL_EXLLAMAV3_BUILD_ROOT=str(build / "exllamav3"),
        TORCH_EXTENSIONS_DIR=str(build / "torch-extensions"),
        TRITON_CACHE_DIR=str(build / "triton"), TMPDIR=str(build / "tmp"))
    if record["family"] != "kimi-k31" and runtime == "python3":
        environment["PYTHONPATH"] += ":" + str(source / "third_party/transformers/src")
    for key, value in environment.items():
        result += ["-e", f"{key}={value}"]
    runner = source / "quantization" / FAMILIES[record["family"]] / "run.py"
    return result + ["--entrypoint", runtime, record["image_id"], str(runner), *common(record),
                     *(["--preflight"] if preflight else [])]


def validate_remote(record):
    host = socket.gethostname().split(".")[0]
    if host != record["host"]:
        raise ValueError("remote preflight reached the wrong host")
    locks_for(host)
    source = Path(record["remote_source_root"])
    if REPO != source:
        raise ValueError("helper is not running from the pre-staged source root")
    checked_source(source, record["commit"])
    if not (source / ".git").is_dir():
        raise ValueError("stage a standalone clone, not a worktree with external Git metadata")
    if (os.getuid(), os.getgid()) != (record["uid"], record["gid"]):
        raise ValueError("remote container identity changed since launch")
    build = Path(record["build_root"])
    expected_build = Path.home() / ".cache/cuteafd/builds" / ("mc-" + record["family"])
    if build != expected_build.resolve():
        raise ValueError("output must be under remote NVMe builds/mc-<family>")
    for name in ("root", "export"):
        if not beneath(Path(record[name]).resolve(), build):
            raise ValueError(name + " must stay under builds/mc-<family>")
    source_snapshot = Path(record["source_snapshot"]).resolve()
    if not beneath(source_snapshot, Path("/mnt/sparknest")) or not source_snapshot.is_dir():
        raise ValueError("source snapshot must stream from /mnt/sparknest")
    tokens = Path(record["tokens"])
    if not tokens.is_dir() or not any(beneath(tokens.resolve(), parent) for parent in
                                        (Path("/mnt/sparknest"), source, build)):
        raise ValueError("tokens must be a corpus directory inside a mounted source, build or SparkNest path")
    subprocess.run(["python3", str(source / "scripts/build/assert-build-filesystem.py"), str(build)],
                   check=True, timeout=60)
    if output(["findmnt", "-n", "-o", "MAJ:MIN", "-T", build.parent]) != output(
            ["findmnt", "-n", "-o", "MAJ:MIN", "-T", "/"]):
        raise ValueError("builds must reside on root NVMe, not a separate archive mount")
    subprocess.run(["df", "-h", "/"], check=True, timeout=30)
    if output(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"] ) != record["image_id"]:
        raise ValueError("remote SM121 image changed since launch")
    for suffix in ("home", "tmp"):
        (build / suffix).mkdir(parents=True, exist_ok=True)
    guard = source / "quantization/kimi_k3/core_k6/run_guard.py"
    name = record["container"] + "-check"
    try:
        subprocess.run(["timeout", "--foreground", "--signal=TERM", "--kill-after=120", "600",
                        "python3", str(guard), "--container", name, "--",
                        *container_command(record, preflight=True)], check=True, timeout=750)
    finally:
        subprocess.run(["docker", "rm", "--force", name], timeout=60, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def forwarding(command, *, stop_seconds=None, **kwargs):
    process = subprocess.Popen(command, **kwargs)
    stopped = []
    def forward(signum, _frame):
        if not stopped:
            stopped.append(time.monotonic())
        if process.poll() is None:
            process.send_signal(signum)
    previous = {sig: signal.signal(sig, forward) for sig in
                (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        while True:
            try:
                return process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                if stopped and stop_seconds is not None and time.monotonic() - stopped[0] >= stop_seconds:
                    process.kill()
                    return process.wait()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def require_lease(host):
    expected = set(locks_for(host).split(","))
    if not expected.issubset(os.environ.get("CUTEAFD_HW_LOCKS_HELD", "").split(",")):
        raise ValueError("execute requires enclosing raptor hwlock lease")
    directory = Path.home() / ".cache/cuteafd"
    for name in expected:
        with (directory / (name + ".lock")).open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            raise ValueError("hardware lock is not held: " + name)


def remote_paths(record):
    root = Path(record["build_root"]) / "launches"
    return root / (record["unit"] + ".json"), root / (record["unit"] + ".exit.json")


def remote_worker(path):
    record = json.loads(Path(path).read_text())
    source = Path(record["remote_source_root"])
    guard = source / "quantization/kimi_k3/core_k6/run_guard.py"
    command = ["timeout", "--foreground", "--signal=TERM", "--kill-after=120", str(RUN_SECONDS),
               "python3", str(guard), "--container", record["container"], "--", *container_command(record)]
    code = 1
    try:
        with Path(path).with_suffix(".log").open("a") as log:
            code = forwarding(command, stop_seconds=35, stdout=log, stderr=subprocess.STDOUT)
    finally:
        subprocess.run(["docker", "rm", "--force", record["container"]], timeout=30, check=False)
        if output(["docker", "ps", "-aq", "--filter", "name=^/" + record["container"] + "$"]):
            raise RuntimeError("owned container remains after remote cleanup")
        _, terminal = remote_paths(record)
        atomic_json(terminal, dict(returncode=code, finished_ns=time.time_ns(), unit=record["unit"]))
    return code


def remote_start(record):
    if socket.gethostname().split(".")[0] != record["host"]:
        raise ValueError("remote supervisor reached the wrong host")
    checked_source(record["remote_source_root"], record["commit"])
    path, terminal = remote_paths(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != record:
            raise ValueError("remote launch identity differs")
        # Lost start replies must not create a second producer.
        if terminal.exists() or subprocess.run(
                ["systemctl", "--user", "is-active", "--quiet", record["unit"]], timeout=30).returncode == 0:
            return
        raise RuntimeError("remote launch exists without an active unit or terminal receipt")
    atomic_json(path, record)
    subprocess.run(["systemd-run", "--user", "--unit", record["unit"], "--collect",
        "--property=Type=exec", "--property=KillMode=mixed", "--property=TimeoutStopSec=180",
        "--property=RuntimeMaxSec=" + str(RUN_SECONDS + 300),
        "--working-directory", record["remote_source_root"], sys.executable,
        str(Path(__file__).resolve()), "--remote-worker", str(path)], check=True, timeout=60)


def remote_wait(record):
    _, terminal = remote_paths(record)
    deadline = time.monotonic() + RUN_SECONDS + 600
    while time.monotonic() < deadline:
        if terminal.exists():
            result = json.loads(terminal.read_text())
            if result["unit"] != record["unit"]:
                raise ValueError("remote terminal identity differs")
            print(json.dumps(result), flush=True)
            return 0
        time.sleep(60)
    raise TimeoutError("remote producer did not write a terminal receipt")


def execute_detached(record):
    require_lease(record["host"])
    remote = str(Path(record["remote_source_root"]) / "quantization/core_launch.py")
    cancelled = []
    previous = {sig: signal.signal(sig, lambda signum, _frame: cancelled.append(signum))
                for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    deadline = time.monotonic() + RUN_SECONDS + 900
    try:
        # A remote unit owns container lifetime, not this SSH connection.
        started = 255
        for attempt in range(3):
            try:
                started = subprocess.run(ssh(record["host"], ["python3", remote, "--remote-start", json.dumps(record)]),
                                         timeout=90).returncode
            except subprocess.TimeoutExpired:
                started = 255
            if started != 255 or cancelled:
                break
            for _ in range(60):
                if cancelled:
                    break
                time.sleep(1)
        if started:
            cancelled.append(0)
        while time.monotonic() < deadline and not cancelled:
            process = subprocess.Popen(ssh(record["host"],
                ["python3", remote, "--remote-wait", json.dumps(record)]), stdout=subprocess.PIPE, text=True)
            while process.poll() is None and not cancelled and time.monotonic() < deadline:
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            if cancelled or time.monotonic() >= deadline:
                process.terminate()
                process.wait(timeout=30)
                break
            if process.returncode == 0:
                try:
                    result = json.loads(process.stdout.read())
                    if result['unit'] != record['unit'] or not isinstance(result['returncode'], int):
                        raise ValueError('remote terminal identity differs')
                    return result['returncode']
                except (ValueError, KeyError):
                    cancelled.append(0)
                    break
            if process.returncode != 255:
                # Nontransport errors can come from the waiter itself; stop the
                # owned unit before releasing the hardware lease in that case.
                cancelled.append(0)
                break
            # Reconnect without killing healthy remote work on a transport failure.
            for _ in range(60):
                if cancelled:
                    break
                time.sleep(1)
        subprocess.run(ssh(record["host"], ["systemctl", "--user", "stop", record["unit"]]),
                       timeout=210, check=False)
        subprocess.run(ssh(record["host"], ["docker", "rm", "--force", record["container"]]),
                       timeout=30, check=False)
        if output(ssh(record["host"], ["docker", "ps", "-aq", "--filter", "name=^/" + record["container"] + "$"])):
            raise RuntimeError("owned container remains after cancellation")
        return 128 + cancelled[0] if cancelled else 124
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def execute(record):
    if record.get("remote_supervisor"):
        return execute_detached(record)
    require_lease(record["host"])
    source = Path(record["remote_source_root"])
    guard = source / "quantization/kimi_k3/core_k6/run_guard.py"
    command = ["timeout", "--foreground", "--signal=TERM", "--kill-after=120", str(RUN_SECONDS),
               "python3", str(guard), "--container", record["container"], "--", *container_command(record)]
    try:
        return forwarding(ssh(record["host"], command), stop_seconds=35)
    finally:
        # Leave room for cleanup inside hwlock's 80-second signal grace. A
        # disconnected SSH client is not proof that its remote container died.
        cleanup = ["docker", "rm", "--force", record["container"]]
        subprocess.run(ssh(record["host"], cleanup), timeout=15, check=False)
        probe = ssh(record["host"], ["docker", "ps", "-aq", "--filter", "name=^/" + record["container"] + "$"])
        remaining = subprocess.check_output(probe, text=True, timeout=15).strip()
        if remaining:
            raise RuntimeError("owned container remains after cleanup")


def supervise(path):
    record = json.loads(path.read_text())
    check = [sys.executable, str(Path(__file__).resolve()), "--remote-check", json.dumps(record)]
    execute_command = [sys.executable, str(Path(__file__).resolve()), "--execute", json.dumps(record)]
    command = [sys.executable, str(REPO / "scripts/hw/hwlock.py"), "--locks", locks_for(record["host"]),
               "--wait", str(WAIT_SECONDS), "--preflight", shlex.join(check),
               "--preflight-timeout", "900", "--", *execute_command]
    with path.with_suffix(".log").open("a") as log:
        code = forwarding(command, stdout=log, stderr=subprocess.STDOUT)
    result = dict(returncode=code, finished_ns=time.time_ns(), unit=record["unit"])
    atomic_json(path.with_suffix(".exit.json"), result)
    if code == 0:
        atomic_json(path.with_suffix(".done.json"), result)
    return code


def dispatch(family, argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # New selectors always use the opt-in parser, including the error for a
    # missing --core-only; no legacy default or command construction changes.
    selectors = ("--core-only", "--core-bits", "--host", "--remote-source-root", "--runtime-python", "--runtime-root")
    if not any(arg.split("=", 1)[0] in selectors for arg in argv):
        return False
    code = main(argv, family=family)
    if code:
        raise SystemExit(code)
    return True


def main(argv=None, *, family=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for private in ("remote-check", "validate-remote", "execute", "supervise", "remote-start", "remote-worker", "remote-wait"):
        parser.add_argument("--" + private, help=argparse.SUPPRESS)
    parser.add_argument("--family", choices=FAMILIES, default=family)
    parser.add_argument("--host", required=False, help="rhea, moa or cassowary (core-only only)")
    parser.add_argument("--core-only", action="store_true")
    parser.add_argument("--core-bits", type=int, choices=[8], default=8)
    for name in ("source-snapshot", "tokens", "root", "export"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--remote-source-root", type=Path, default=REPO,
                        help="pre-staged clean checkout of the same commit, recursively initialized")
    parser.add_argument("--runtime-python", default="python3", help="container Python; optional pre-staged HF4 overlay")
    parser.add_argument("--runtime-root", type=Path, help="read-only overlay root containing --runtime-python")
    args = parser.parse_args(argv)
    if args.remote_start:
        remote_start(json.loads(args.remote_start))
        return 0
    if args.remote_worker:
        return remote_worker(Path(args.remote_worker))
    if args.remote_wait:
        return remote_wait(json.loads(args.remote_wait))
    if args.supervise:
        return supervise(Path(args.supervise))
    if args.execute:
        return execute(json.loads(args.execute))
    if args.validate_remote:
        validate_remote(json.loads(args.validate_remote))
        return 0
    if args.remote_check:
        record = json.loads(args.remote_check)
        remote = Path(record["remote_source_root"]) / "quantization/core_launch.py"
        subprocess.run(ssh(record["host"], ["python3", str(remote), "--validate-remote", json.dumps(record)]),
                       check=True, timeout=850)
        return 0
    if not args.core_only or not args.family or not args.host or any(
            getattr(args, name) is None for name in ("source_snapshot", "tokens", "root", "export")):
        parser.error("core-only requires family, host, source-snapshot, tokens, root and export")
    locks_for(args.host)
    if socket.gethostname().split(".")[0] != "raptor":
        raise ValueError("launch from raptor, where the canonical hardware queue lives")
    expected_cap = "110g" if args.family in ("kimi-k31", "muse") else "96g"
    cap = os.environ.get("CUTEAFD_DEV_MEMORY", expected_cap).lower()
    if cap != expected_cap:
        raise ValueError(f"this family requires CUTEAFD_DEV_MEMORY={expected_cap} (no uncapped override)")
    if args.runtime_python != "python3":
        if not Path(args.runtime_python).is_absolute() or not args.runtime_root or not args.runtime_root.is_absolute():
            parser.error("runtime-python override requires absolute runtime-python and runtime-root")
        if not beneath(Path(args.runtime_python), args.runtime_root):
            parser.error("runtime-python must be inside runtime-root")
    commit = checked_source(REPO)
    stamp = str(time.time_ns())
    unit = "mc-" + args.family + "-" + stamp
    remote_source = args.remote_source_root.expanduser()
    if not remote_source.is_absolute():
        parser.error("remote-source-root must be absolute")
    identity = json.loads(output(ssh(args.host, ["python3", "-c",
        "import os,pwd,json; from pathlib import Path; print(json.dumps(dict(home=str(Path.home()), "
        "uid=os.getuid(), gid=os.getgid(), user=pwd.getpwuid(os.getuid()).pw_name)))"])))
    build = Path(identity["home"]) / ".cache/cuteafd/builds" / ("mc-" + args.family)
    record = dict(family=args.family, host=args.host, commit=commit, unit=unit, container=unit,
                  uid=identity["uid"], gid=identity["gid"], user=identity["user"],
                  build_root=str(build), remote_source_root=str(remote_source), container_memory=expected_cap,
                  runtime_python=args.runtime_python, runtime_root=str(args.runtime_root or remote_source),
                  resume=args.resume, remote_supervisor=True, image_id=output(ssh(args.host,
                      ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"])))
    for name in ("source_snapshot", "tokens", "root", "export"):
        value = getattr(args, name)
        if not value.is_absolute():
            parser.error(name.replace("_", "-") + " must be an absolute REMOTE path")
        record[name] = str(value)
    # Fail early without taking a ticket. hwlock repeats this before enqueue
    # and at grant; neither check exposes a GPU to the producer preflight.
    main(["--remote-check", json.dumps(record)])
    local = Path.home() / ".cache/cuteafd/builds" / ("mc-" + args.family) / "launches"
    subprocess.run(["python3", str(REPO / "scripts/build/assert-build-filesystem.py"), str(local)],
                   check=True, timeout=60)
    local.mkdir(parents=True, exist_ok=True)
    path = local / (stamp + ".json")
    atomic_json(path, record)
    command = ["systemd-run", "--user", "--unit", unit, "--collect",
               "--property=Type=exec", "--property=KillMode=mixed", "--property=TimeoutStopSec=240",
               "--property=RuntimeMaxSec=" + str(WAIT_SECONDS + RUN_SECONDS + 1800),
               "--working-directory", str(REPO), sys.executable, str(Path(__file__).resolve()),
               "--supervise", str(path)]
    subprocess.run(command, check=True, timeout=60)
    print(json.dumps(dict(unit=unit, launch=str(path), log=str(path.with_suffix(".log")),
                          done=str(path.with_suffix(".done.json")), remote_build=str(build))))
    return 0


if __name__ == "__main__":
    sys.exit(main())

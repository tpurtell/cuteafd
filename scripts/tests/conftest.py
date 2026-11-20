"""Fixtures shared by every script test."""
from __future__ import annotations

import builtins
import fcntl
import os
import sys
from pathlib import Path

import pytest

# The real hardware lock directory: flock files next to it (gpu1.lock, build.lock*)
# and its fair-queue tickets (hwq/) exclude the cluster's jobs. No test may touch
# them (#1048: a script-test run held the real gpu1.lock while a hardware job
# waited). Resolved at import, before any test can point HOME elsewhere.
REAL_LOCK_DIR = Path.home() / ".cache" / "cuteafd"
LOCK_DIR_ATTRIBUTES = ("LOCK_DIR", "LOCKS_DIR", "LOCK_ROOT")

# Startup files a shell runs before its first command: every non-interactive bash runs
# $BASH_ENV, an interactive POSIX shell $ENV. NGC-based images export
# BASH_ENV=/etc/bash.bashrc and ENV=/etc/shinit_v2, and that startup code calls
# `nvidia-smi -q -d COMPUTE`. The launcher tests put a bash-script `nvidia-smi` stub first
# on PATH, so inside such an image every stub call ran the startup code again and called
# itself, without end. No script test needs a startup file, so none gets one.
SHELL_STARTUP_VARIABLES = ("BASH_ENV", "ENV")


def _is_real_lock_path(path) -> bool:
    """Whether `path` is a real hardware lock file or fair-queue entry."""
    try:
        name = os.fspath(path)
    except TypeError:
        return False  # an integer fd or some other object `open` accepts
    if isinstance(name, bytes):
        name = os.fsdecode(name)
    real = os.path.realpath(name)
    root = str(REAL_LOCK_DIR)
    queue = os.path.join(root, "hwq")
    if real == queue or real.startswith(queue + os.sep):
        return True
    # gpu1.lock, gpu1.lock.1, build.lock.3, ...
    return os.path.dirname(real) == root and ".lock" in os.path.basename(real)


def _fd_path(fd) -> str | None:
    """The path an open descriptor refers to, when the OS exposes it."""
    try:
        return os.readlink(f"/proc/self/fd/{fd}")
    except (OSError, TypeError, ValueError):
        return None


def _refuse(action: str, path) -> None:
    # pytest.fail raises BaseException, so code under test cannot swallow it.
    pytest.fail(f"script test {action} the real hardware lock path {os.fsdecode(os.fspath(path))!r}; "
                f"hardware locks are private to the run (CUTEAFD_LOCK_DIR="
                f"{os.environ.get('CUTEAFD_LOCK_DIR')!r})")


@pytest.fixture(scope="session", autouse=True)
def isolated_lock_dir(tmp_path_factory):
    """Point every lock user at a private directory for the whole session.

    `CUTEAFD_LOCK_DIR` reaches code that resolves the lock directory at call time
    (hwlock.lock_dir, build-resource, export-placement) and every subprocess that
    inherits the environment; modules already imported that cached a lock-dir
    attribute get it overwritten too. xdist gives each worker its own directory."""
    lock_dir = tmp_path_factory.mktemp("hardware-locks")
    patch = pytest.MonkeyPatch()
    patch.setenv("CUTEAFD_LOCK_DIR", str(lock_dir))
    for module in list(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if not source or not any(hasattr(module, name) for name in LOCK_DIR_ATTRIBUTES):
            continue
        try:
            Path(source).resolve().relative_to(Path(__file__).resolve().parents[1])
        except (OSError, ValueError):
            continue  # not a script module
        for name in LOCK_DIR_ATTRIBUTES:
            if hasattr(module, name):
                patch.setattr(module, name, lock_dir)
    yield lock_dir
    patch.undo()


@pytest.fixture(autouse=True)
def no_real_hardware_locks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail a test that opens or flocks the real hardware locks anyway.

    The session fixture isolates the paths every script resolves; this catches
    code that reaches the real directory by another route (a hard-coded HOME, an
    import that asked for the default before the session fixture ran)."""
    real_open, real_os_open, real_flock = builtins.open, os.open, fcntl.flock

    def guarded_open(file, *args, **kwargs):
        if _is_real_lock_path(file):
            _refuse("opened", file)
        return real_open(file, *args, **kwargs)

    def guarded_os_open(path, *args, **kwargs):
        if _is_real_lock_path(path):
            _refuse("opened", path)
        return real_os_open(path, *args, **kwargs)

    def guarded_flock(fd, operation):
        path = _fd_path(fd)
        if path is not None and _is_real_lock_path(path.removesuffix(" (deleted)")):
            _refuse("flocked", path)
        return real_flock(fd, operation)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(os, "open", guarded_os_open)
    monkeypatch.setattr(fcntl, "flock", guarded_flock)


@pytest.fixture(autouse=True)
def no_shell_startup_files(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove the shell startup variables for the duration of every test.

    Helpers that pass `{**os.environ, ...}`, `dict(os.environ)`, `os.environ.copy()` or
    no `env` at all read the environment when they launch, so their shells start without
    them; helpers that build their environment from scratch never carried them."""
    for name in SHELL_STARTUP_VARIABLES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def no_real_cdi_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    """Launcher tests mock docker and name fake GPUs; the machine's CDI spec is not theirs to check."""
    monkeypatch.setenv("CUTEAFD_CDI_CHECK", "off")


@pytest.fixture(scope="session", autouse=True)
def hermetic_launcher_host(tmp_path_factory):
    """Fix the two host probes a launcher makes, so a launch line is the same on every host.

    run-family.sh mounts a host power governor's state directory when it exists,
    and the CDI helpers read the host's spec to decide whether a single per-GPU
    request may collapse to `all` (WSL2's all-only spec). Script tests mock
    Docker and name fake GPUs, so both inputs are fixtures here: a seeded
    governor directory and a per-GPU spec shaped like raptor's."""
    import launcher_sandbox
    powerd = tmp_path_factory.mktemp("gpu-powerd")
    spec = tmp_path_factory.mktemp("cdi") / "nvidia.yaml"
    names = (*launcher_sandbox.FAKE_GPU_UUIDS, "0", "1", "all")
    spec.write_text("devices:\n" + "".join(f"  - name: {name}\n" for name in names))
    patch = pytest.MonkeyPatch()
    patch.setenv("CUTEAFD_GPU_POWERD_DIR", str(powerd))
    patch.setenv("CUTEAFD_CDI_SPECS", str(spec))
    fuse = tmp_path_factory.mktemp("fuse-connections")
    nest = tmp_path_factory.mktemp("sparknest-run")
    patch.setenv("CUTEAFD_FUSE_SYSFS_ROOT", str(fuse))
    patch.setenv("CUTEAFD_SPARKNEST_RUN_DIR", str(nest))
    tools = tmp_path_factory.mktemp("host-tools")
    hostname = tools / "hostname"
    hostname.write_text("#!/bin/sh\nprintf 'coordinator-test\\n'\n")
    hostname.chmod(0o755)
    patch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
    yield
    patch.undo()


@pytest.fixture(scope="session", autouse=True)
def launcher_inputs(tmp_path_factory):
    """One private template per xdist worker; sandbox directory entries stay test-local."""
    import launcher_sandbox
    launcher_sandbox.configure(tmp_path_factory.mktemp("launcher-inputs"))
    yield
    launcher_sandbox.configure(None)


def pytest_addoption(parser):
    parser.addoption("--container-torch", action="store_true",
                     help="collect container-only reference and offline DSL tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "container_torch: needs the matching dev container's reference/DSL dependencies")
    config.addinivalue_line("markers", "installed_checkpoint: explicit real-checkpoint qualification")


def pytest_ignore_collect(collection_path, config):
    return (not config.getoption("--container-torch") and collection_path.name in {
        "test_muse_mistral_reference.py", "test_inkling2_reference.py", "test_minimax_m3_reference.py"})


def pytest_collection_modifyitems(config, items):
    if config.getoption("--container-torch"):
        return
    removed = [item for item in items if item.get_closest_marker("container_torch")]
    if removed:
        config.hook.pytest_deselected(items=removed)
        items[:] = [item for item in items if not item.get_closest_marker("container_torch")]


@pytest.fixture(scope="session", autouse=True)
def no_live_cache(tmp_path_factory):
    """One worker-private HOME for all script subprocesses and cache defaults."""
    root = tmp_path_factory.mktemp("simulated-home")
    patch = pytest.MonkeyPatch()
    patch.setenv("HOME", str(root))
    patch.setenv("XDG_CACHE_HOME", str(root / ".cache"))
    patch.setenv("CUTEAFD_BUILD_ROOT", str(root / "builds"))
    patch.setenv("CUTEAFD_TEST_BUILD_ROOT", str(root / "builds"))
    from launcher_sandbox import SimCluster, seed_checkpoint_metadata
    patch.setenv("CUTEAFD_TEST_HOST", SimCluster().env()["CUTEAFD_TEST_HOST"])
    hub = root / ".cache/huggingface"
    seed_checkpoint_metadata(hub)
    patch.setenv("HF_HOME", str(hub))
    patch.setenv("HF_HUB_OFFLINE", "1")
    yield root
    patch.undo()


@pytest.fixture
def tmp_build_root(tmp_path, monkeypatch):
    root = tmp_path / "builds"
    root.mkdir()
    monkeypatch.setenv("CUTEAFD_BUILD_ROOT", str(root))
    monkeypatch.setenv("CUTEAFD_TEST_BUILD_ROOT", str(root))
    return root


@pytest.fixture
def fake_repo(tmp_path):
    """Mutable Git history for ref-pinning and release tests, never the checkout."""
    import subprocess
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for key, value in [("user.name", "test"), ("user.email", "test@example.invalid")]:
        subprocess.run(["git", "-C", str(root), "config", key, value], check=True)
    for name in ["run.sh", "wip.sh"]:
        path = root / name
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)
    return root


@pytest.fixture
def sim_cluster():
    from launcher_sandbox import SimCluster
    return SimCluster()


@pytest.fixture(scope="session")
def host_tools():
    import shutil
    return {name: shutil.which(name) for name in ("c++", "cc", "cmake", "tmux", "node", "curl", "jq")}


@pytest.fixture(scope="session")
def launcher_capture():
    """Memoize immutable subprocess results by complete caller-supplied config."""
    import hashlib
    import json
    cache = {}
    def capture(config, run):
        key = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()
        if key not in cache:
            result = run()
            # Captures outlive a function's tmp_path on xdist workers. Normalize
            # their private roots so later assertions never reference deleted dirs.
            import re
            root = re.search(r"(/[^\s]+/test_[^/\s]+[0-9]+(?:/example)?)/hf/", result.stderr)
            if root:
                result = type(result)(result.args, result.returncode,
                    result.stdout.replace(root[1], "<root>"), result.stderr.replace(root[1], "<root>"))
            cache[key] = result
        return cache[key]
    return capture

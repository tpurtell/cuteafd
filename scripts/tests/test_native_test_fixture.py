"""Native fake DSOs compile with Cargo, never under a timed test or nested slot."""
from pathlib import Path
import os
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
FFI = ROOT / "rust/crates/cuteafd-ffi"


@pytest.fixture(scope="module")
def build_script(tmp_path_factory):
    binary = tmp_path_factory.mktemp("native-fixture-build") / "build-script"
    subprocess.run(["rustc", "--edition=2024", str(FFI / "build.rs"), "-o", str(binary)],
                   check=True, timeout=60)
    return binary


def test_production_fixture_build_needs_no_tools(build_script, tmp_path):
    env = {**os.environ, "PATH": str(tmp_path)}
    env.pop("CARGO_FEATURE_TEST_SUPPORT", None)
    result = subprocess.run([str(build_script)], env=env, check=True,
                            capture_output=True, text=True, timeout=5)
    assert result.stdout == "cargo:rerun-if-changed=build.rs\n"


def test_fixture_build_tracks_all_inputs_without_nested_slots(build_script, tmp_path):
    tools = tmp_path / "tools"
    tools.mkdir()
    calls = tmp_path / "calls"
    # Stub subprocesses: the real Cargo smoke verifies the resulting DSO. Here
    # capture the complete build recipe and dependency contract without a build.
    for tool in ("python3", "nice"):
        path = tools / tool
        path.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALLS"\n')
        path.chmod(0o755)
    result = subprocess.run([str(build_script)], check=True, capture_output=True, text=True,
                            timeout=5, env={**os.environ, "PATH": str(tools), "CALLS": str(calls),
                                            "CARGO_FEATURE_TEST_SUPPORT": "1",
                                            "CARGO_MANIFEST_DIR": str(FFI), "OUT_DIR": str(tmp_path)})
    watched = {Path(line.split("=", 1)[1]).resolve() for line in result.stdout.splitlines()}
    sources = [FFI / "tests/fixtures/native_library_lifetime.c",
               FFI / "tests/fixtures/native_library_kernels.cc"]
    inputs = [FFI / "build.rs", *sources,
              ROOT / "python/tools/lib/mxfp4_layouts.py", ROOT / "scripts/build/mxfp4-layouts.json",
              ROOT / "scripts/build/assert-build-filesystem.py"]
    # Include every quoted (project-owned) header, including nested includes.
    pending = sources.copy()
    while pending:
        source = pending.pop()
        for line in source.read_text().splitlines():
            if line.startswith('#include "'):
                name = line.split('"')[1]
                if name == "cuteafd_mxfp4_layouts.h":
                    continue  # generated from the watched Python + JSON inputs
                header = (source.parent / name).resolve()
                if header not in inputs:
                    inputs.append(header)
                    pending.append(header)
    # build.rs's relative directive is relative to the crate, not pytest's cwd.
    watched.discard(Path("build.rs").resolve())
    watched.add(FFI / "build.rs")
    assert set(inputs) <= watched
    commands = calls.read_text().splitlines()
    assert len(commands) == 5
    assert sum("-n 19 cc " in command for command in commands) == 1
    assert sum("-n 19 c++ " in command for command in commands) == 2
    assert all("build-slot" not in command for command in commands)


def test_fixture_runtime_only_materializes_embedded_bytes():
    source = (FFI / "src/native_library_lifetime_fixture.rs").read_text()
    assert "include_bytes!" in source
    assert "Command::" not in source
    assert "build-slot.sh" not in source
    assert "[dev-dependencies]\ncuteafd-ffi = " in (FFI / "Cargo.toml").read_text()

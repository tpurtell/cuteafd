import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "build/verify-dsh-source.py"
SPEC = importlib.util.spec_from_file_location("verify_dsh_source", SCRIPT)
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def fixture(tmp_path):
    source = tmp_path / "dsh"
    source.mkdir()
    (source / "LICENSE").write_text("MIT fixture\n")
    (source / "package.json").write_text('{"name":"@deepseek-ai/dsh-root"}\n')
    (source / "pnpm-lock.yaml").write_text("lockfileVersion: '9.0'\n")
    lock = tmp_path / "dsh.lock.json"
    lock.write_text(json.dumps({
        "schema": 1,
        "repository": "https://github.com/tpurtell/deepseek-harness",
        "revision": "a" * 40,
        "source_tree_sha256": VERIFIER._lock.source_tree_sha256(source),
    }))
    return source, lock


def test_metadata_free_source_and_mutation(tmp_path):
    source, lock = fixture(tmp_path)
    assert VERIFIER.verify(source, lock)["revision"] == "a" * 40
    (source / "pnpm-lock.yaml").write_text("changed\n")
    with pytest.raises(VERIFIER._lock.VerificationError, match="digest"):
        VERIFIER.verify(source, lock)


def test_git_revision_origin_and_dirty_are_checked(tmp_path, monkeypatch):
    source, lock = fixture(tmp_path)
    (source / ".git").write_text("gitdir: fixture\n")
    results = {
        ("rev-parse", "HEAD"): "a" * 40,
        ("status", "--porcelain", "--untracked-files=all"): "",
        ("remote", "get-url", "origin"): "git@github.com:tpurtell/deepseek-harness.git",
    }
    monkeypatch.setattr(VERIFIER._lock, "_run_git", lambda _, *args: results[args])
    assert VERIFIER.verify(source, lock)
    for command, replacement, expected in [
        (("rev-parse", "HEAD"), "b" * 40, "revision"),
        (("status", "--porcelain", "--untracked-files=all"), " M package.json", "local source"),
        (("remote", "get-url", "origin"), "https://example.invalid/other", "origin"),
    ]:
        original = results[command]
        results[command] = replacement
        with pytest.raises(VERIFIER._lock.VerificationError, match=expected):
            VERIFIER.verify(source, lock)
        results[command] = original


def test_required_manifest_and_lock_schema(tmp_path):
    source, lock = fixture(tmp_path)
    (source / "LICENSE").unlink()
    with pytest.raises(VERIFIER._lock.VerificationError, match="missing LICENSE"):
        VERIFIER.verify(source, lock)
    (source / "LICENSE").write_text("MIT fixture\n")
    value = json.loads(lock.read_text())
    value["unexpected"] = True
    lock.write_text(json.dumps(value))
    with pytest.raises(VERIFIER._lock.VerificationError, match="fields mismatch"):
        VERIFIER.verify(source, lock)


def test_entrypoint_keeps_launcher_flags_before_app_flags(tmp_path):
    script = Path(__file__).parents[2] / "docker/agent-entrypoint.sh"
    binary = tmp_path / "bin"
    binary.mkdir()
    (binary / "node").write_text(
        '#!/bin/sh\n'
        'if [ "$1" = "--input-type=module" ]; then printf 192.0.2.10; '
        'else printf "%s\\n" "$@"; fi\n'
    )
    (binary / "node").chmod(0o755)
    environment = dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}",
                       HOME=str(tmp_path / "home"), DSH_HOME=str(tmp_path / "dsh"),
                       DSH_AGENTS_HOME=str(tmp_path / "agents"))
    result = subprocess.run(["sh", str(script), "--profile", "cuteafd", "--port", "3010"],
                            env=environment, text=True, capture_output=True, check=True)
    assert result.stdout.splitlines() == [
        "/opt/dsh/node_modules/@deepseek-ai/dsh/lib/bin.js",
        "--profile", "cuteafd", "--port", "3010", "--host", "192.0.2.10",
    ]

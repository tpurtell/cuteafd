"""CPU contracts for optional keyed API and benchmark launcher wiring."""
from pathlib import Path
import os
import json
import stat
import subprocess

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / "scripts/lib/release-common.sh"


def test_api_policy_keys_are_recognized_and_defaults_stay_open():
    script = f'''set -euo pipefail
source "{COMMON}"
release_known_key API_KEY_FILE
release_known_key ENABLE_BENCH
release_load_config "{ROOT / 'cuteafd.config'}" stop
printf 'KEY=%s BENCH=%s' "$API_KEY_FILE" "$ENABLE_BENCH"
'''
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.endswith("KEY= BENCH=off")


def prepare_key(home, enabled="off", instance="test", supplied=""):
    script = f'''set -euo pipefail
source "{COMMON}"
API_KEY_FILE="$3"
release_prepare_api_key "$1" "$2"
printf '%s' "${{API_KEY_FILE:-}}"
'''
    return subprocess.run(
        ["bash", "-c", script, "test", enabled, instance, str(supplied)],
        env={**os.environ, "HOME": str(home)}, capture_output=True, text=True)


def test_disabled_benchmark_does_not_create_key_or_cache(tmp_path):
    result = prepare_key(tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout == result.stderr == ""
    assert not (tmp_path / ".cache").exists()


def test_supplied_key_is_preserved_without_provisioning(tmp_path):
    key = tmp_path / "supplied"
    key.write_text("my-own-key\n")
    key.chmod(0o640)
    result = prepare_key(tmp_path, "on", supplied=key)
    assert result.returncode == 0, result.stderr
    assert result.stdout == str(key)
    assert result.stderr == ""
    assert key.read_text() == "my-own-key\n"
    assert stat.S_IMODE(key.stat().st_mode) == 0o640
    assert not (tmp_path / ".cache").exists()


def test_generated_key_is_private_atomic_reused_and_path_only(tmp_path):
    result = prepare_key(tmp_path, "on")
    assert result.returncode == 0, result.stderr
    path = tmp_path / ".cache/cuteafd/test/api-key"
    assert result.stdout == str(path)
    assert result.stderr == f"Benchmark API key file: {path}\n"
    key = path.read_text()
    assert len(key.strip()) == 64
    assert key.strip() not in result.stdout + result.stderr
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    named = path.with_name("api-keys")
    assert set(path.parent.iterdir()) == {path, named}
    assert stat.S_IMODE(named.stat().st_mode) == 0o600
    keys = json.loads(named.read_text())
    assert keys["default"] == path.read_text().strip()
    assert keys["agent"] != keys["default"]
    again = prepare_key(tmp_path, "on")
    assert again.returncode == 0, again.stderr
    assert path.read_text() == key
    assert again.stdout == str(path)
    named = path.with_name("api-keys")
    assert set(path.parent.iterdir()) == {path, named}
    assert stat.S_IMODE(named.stat().st_mode) == 0o600
    keys = json.loads(named.read_text())
    assert keys["default"] == path.read_text().strip()
    assert keys["agent"] != keys["default"]


def test_concurrent_key_provisioning_preserves_one_key(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: prepare_key(tmp_path, "on"), range(8)))
    assert all(result.returncode == 0 for result in results), results
    assert len({result.stdout for result in results}) == 1
    path = Path(results[0].stdout)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    named = path.with_name("api-keys")
    assert set(path.parent.iterdir()) == {path, named}
    assert stat.S_IMODE(named.stat().st_mode) == 0o600
    keys = json.loads(named.read_text())
    assert keys["default"] == path.read_text().strip()
    assert keys["agent"] != keys["default"]


def test_generated_key_refuses_unsafe_existing_file(tmp_path):
    path = tmp_path / ".cache/cuteafd/test/api-key"
    path.parent.mkdir(parents=True)
    path.parent.chmod(0o700)
    path.write_text("existing-key\n")
    path.chmod(0o644)
    result = prepare_key(tmp_path, "on")
    assert result.returncode != 0
    assert "0600" in result.stderr
    assert path.read_text() == "existing-key\n"


def test_probe_credentials_use_stdin_not_curl_argv(tmp_path):
    key = tmp_path / "key"
    key.write_text("test-private-key\n")
    script = f'''set -euo pipefail
source "{COMMON}"
API_KEY_FILE="{key}"
curl() {{ printf 'ARGV=%s\\n' "$*"; if [[ $1 == --header ]]; then while IFS= read -r line; do printf 'STDIN=%s\\n' "$line"; done; fi; }}
release_api_curl -fsS http://localhost/v1/models
'''
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == "ARGV=--header @- -fsS http://localhost/v1/models"
    assert lines[1] == "STDIN=Authorization: Bearer test-private-key"


def test_console_secret_reuse_rotation_and_bind_mount_inode(tmp_path):
    script = f'''set -euo pipefail
source "{COMMON}"
release_prepare_console test
printf '%s' "$CONSOLE_SECRET_FILE"
'''
    env = {**os.environ, "HOME": str(tmp_path)}
    first = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    path = Path(first.stdout)
    old = path.read_text()
    inode = path.stat().st_ino
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    second = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    assert path.read_text() == old
    rotate = subprocess.run(["bash", str(ROOT / "scripts/launch/console-secret.sh"), "rotate"],
                            env=env, capture_output=True, text=True)
    assert rotate.returncode == 0, rotate.stderr
    assert path.stat().st_ino == inode
    assert path.read_text() != old
    assert old.strip() not in first.stdout + first.stderr + rotate.stdout + rotate.stderr
    for launcher in [ROOT / "run.sh", ROOT / "scripts/launch/run-family.sh"]:
        source = launcher.read_text()
        assert source.count("release_print_console_link") == 1
        assert "--console-secret-file /run/cuteafd-console-secret" in source
        assert "dst=/run/cuteafd-console-secret,readonly" in source
        assert "/root/.cache/cuteafd/usage" in source


def test_console_capability_label_and_wip_probe(tmp_path):
    script = f'''set -euo pipefail
source "{COMMON}"
docker() {{ if [[ "$*" == *console-gate* ]]; then printf '%s' "$1" >/dev/null; printf '%s' "$LABEL"; else printf '%s' "$HELP"; fi; }}
if release_console_supported test-image "${{BINARY:-}}"; then printf supported; else printf unsupported; fi
'''
    for label, help_text, binary, expected in [
        ("", "", "", "unsupported"), ("1", "", "", "supported"),
        ("1", "old help", "/wip/cuteafd", "unsupported"),
        ("", "--console-secret-file", "/wip/cuteafd", "supported"),
    ]:
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                                env={**os.environ, "HOME": str(tmp_path), "LABEL": label,
                                     "HELP": help_text, "BINARY": binary})
        assert result.returncode == 0, result.stderr
        assert result.stdout == expected
    for path in [ROOT / "run.sh", ROOT / "scripts/launch/run-family.sh"]:
        source = path.read_text()
        assert 'if release_console_supported' in source
        assert 'if ((console_supported)); then release_print_console_link' in source
    assert 'LABEL org.cuteafd.console-gate="1"' in (ROOT / "docker/Dockerfile.release").read_text()


def test_every_serve_path_uses_explicit_keyed_policy():
    families = ROOT / "rust/crates/cuteafd-daemon/src/families"
    paths = [families / family / "serve.rs" for family in
             ("glm5", "glm5_flash", "mimo_v2", "qwen4", "deepseek_v4")]
    paths.append(families / "deepseek_v41/v41_native_serve.rs")
    for path in paths:
        source = path.read_text()
        assert "args.api.load()?" in source, path
        assert "api.app(router," in source, path
        assert "api.serve(profile" in source, path  # shared health + gateway mount
        assert "cuteafd_bench::app(" not in source, path
    for path in [ROOT / "run.sh", ROOT / "scripts/launch/run-family.sh"]:
        source = path.read_text()
        assert "--api-key-file /run/cuteafd-api-key" in source
        assert "--enable-bench" in source
        assert "dst=/run/cuteafd-api-key,readonly" in source
        assert "release_prepare_api_key" in source
    native_launcher = (ROOT / "run.sh").read_text()
    assert native_launcher.index("if ((dry_run)); then") < native_launcher.index("release_prepare_api_key")
    assert native_launcher.index("release_prepare_api_key") < native_launcher.index("start_coordinator()")

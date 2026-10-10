"""CPU-only release-leg supervision; no real Docker, SSH or hardware calls."""
from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build.sh"


def block(name: str) -> str:
    text = BUILD.read_text()
    return text.split(f"# {name}:start", 1)[1].split(f"# {name}:end", 1)[0]


@pytest.mark.parametrize("sequential", ["0", "1"])
def test_dry_run_reports_concurrent_legs_or_sequential_opt_out(tmp_path, sequential):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("docker", "ssh", "rsync", "nvidia-smi"):
        stub = bin_dir / name
        stub.write_text("#!/bin/sh\nexit 91\n")
        stub.chmod(0o755)
    result = subprocess.run(
        [str(BUILD), "--dry-run"], text=True, capture_output=True, timeout=10,
        env={**os.environ, "PATH": f"{bin_dir}:" + os.environ["PATH"],
             "CUTEAFD_RELEASE_SEQUENTIAL": sequential},
    )
    assert result.returncode == 0, result.stderr
    if sequential == "0":
        assert "coordinator and Spark legs build concurrently" in result.stdout
        assert "shared export/distribution waits for both" in result.stdout
    else:
        assert "sequential: coordinator then Spark" in result.stdout
        assert "CUTEAFD_RELEASE_SEQUENTIAL=1" in result.stdout


def run_legs(tmp_path: Path, bodies: str, sequential: int = 0, tail: str = ""):
    script = f"""set -euo pipefail
source {shlex.quote(str(ROOT / 'scripts/lib/release-common.sh'))}
repo_root={shlex.quote(str(ROOT))}
release_leg_log_dir={shlex.quote(str(tmp_path))}
release_sequential={sequential}
release_leg_plan=fixture
seed_host=stub
export_container=fixture-export
release_cancel_coordinator_build() {{ :; }}
release_cancel_remote_build() {{ echo remote-cancel >>"$release_leg_log_dir/cleanup"; }}
{block('release-build-supervision')}
{bodies}
release_build_legs
printf 'tail\n' >"$release_leg_log_dir/tail"
{tail}
"""
    return subprocess.run(["bash", "-c", script], text=True, capture_output=True, timeout=15)


@pytest.mark.parametrize("sequential", [0, 1])
def test_both_legs_finish_before_shared_tail(tmp_path, sequential):
    coord_body = ': >"$release_leg_log_dir/coord-start"'
    if not sequential:
        coord_body += '\nwhile [[ ! -e "$release_leg_log_dir/spark-start" ]]; do sleep 0.01; done'
    bodies = f"""
build_coordinator_release() (
  {coord_body}
  echo '== coordinator banner =='
  : >"$release_leg_log_dir/coord-done"
)
build_spark_release() (
  : >"$release_leg_log_dir/spark-start"
  {'[[ -e "$release_leg_log_dir/coord-done" ]]' if sequential else ':'}
  while [[ ! -e "$release_leg_log_dir/coord-start" ]]; do sleep 0.01; done
  echo '== Spark banner =='
  : >"$release_leg_log_dir/spark-done"
)
"""
    result = run_legs(tmp_path, bodies, sequential,
                      '[[ -e "$release_leg_log_dir/coord-done" && -e "$release_leg_log_dir/spark-done" ]]')
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "tail").exists()
    assert "coordinator banner" in (tmp_path / "coordinator.log").read_text()
    assert "Spark banner" in (tmp_path / "spark.log").read_text()
    assert not (tmp_path / "cleanup").exists()
    assert not (tmp_path / "completions").exists()


def process_running(pid: int) -> bool:
    try:
        # Reparented children can remain zombies briefly; they are stopped.
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0] != "Z"
    except FileNotFoundError:
        return False


def process_stops(pid: int, within: float = 2.0) -> bool:
    """True if `pid` is still running after `within` seconds. A KILLed child
    can outlive its group leader's return for a few milliseconds before the
    kernel tears it down."""
    deadline = time.monotonic() + within
    while process_running(pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    return process_running(pid)


@pytest.mark.parametrize("failing", ["coord", "spark"])
@pytest.mark.parametrize("immediate", [False, True])
def test_either_failure_names_leg_stops_other_group_and_skips_tail(tmp_path, failing, immediate):
    failing_body = 'exit 17'
    if not immediate:
        failing_body = 'while [[ ! -s "$release_leg_log_dir/child.pid" ]]; do sleep 0.01; done\n' + failing_body
    waiting_body = """
  # This descendant ignores TERM, proving group KILL catches more than leaders.
  # Match the timed Docker/SSH launch without creating an escaping process group.
  timeout 60 --foreground bash -c 'trap "" TERM; printf "%s\\n" "$$" >"$1"; exec sleep 30' \
    -- "$release_leg_log_dir/child.pid" &
  child=$!
  wait "$child"
"""
    bodies = f"""
build_coordinator_release() (
{failing_body if failing == 'coord' else waiting_body}
)
build_spark_release() (
{failing_body if failing == 'spark' else waiting_body}
)
"""
    result = run_legs(tmp_path, bodies)
    assert result.returncode != 0
    assert f"[{failing}] release build leg failed (exit 17)" in result.stderr
    assert not (tmp_path / "tail").exists()
    assert (tmp_path / "cleanup").read_text().strip() == "remote-cancel"
    if (tmp_path / "child.pid").exists():
        assert not process_stops(int((tmp_path / "child.pid").read_text()))
    assert not (tmp_path / "completions").exists()


def test_sequential_coordinator_failure_never_starts_spark(tmp_path):
    result = run_legs(tmp_path, """
build_coordinator_release() ( exit 23 )
build_spark_release() ( : >"$release_leg_log_dir/spark-start" )
""", sequential=1)
    assert result.returncode != 0
    assert "[coord] release build leg failed (exit 23)" in result.stderr
    assert not (tmp_path / "spark-start").exists()
    assert not (tmp_path / "cleanup").exists()


def test_term_cancels_both_groups(tmp_path):
    result = run_legs(tmp_path, """
supervisor=$BASHPID
build_coordinator_release() (
  sleep 30 &
  child=$!
  printf '%s\n' "$child" >"$release_leg_log_dir/coord.pid"
  wait "$child"
)
build_spark_release() (
  sleep 30 &
  child=$!
  printf '%s\n' "$child" >"$release_leg_log_dir/spark.pid"
  while [[ ! -s "$release_leg_log_dir/coord.pid" ]]; do sleep 0.01; done
  kill -TERM "$supervisor"
  wait "$child"
)
""")
    assert result.returncode == 143, result.stderr
    assert "stopping both legs" in result.stderr
    for leg in ("coord", "spark"):
        assert not process_running(int((tmp_path / f"{leg}.pid").read_text()))
    assert not (tmp_path / "tail").exists()


@pytest.mark.parametrize('failing', ['coord', 'spark'])
def test_killed_wrapper_without_fifo_completion_cancels_peer(tmp_path, failing):
    bodies = f'''
build_coordinator_release() {{
  {'kill -KILL "$BASHPID"' if failing == 'coord' else 'sleep 30'}
}}
build_spark_release() {{
  {'kill -KILL "$BASHPID"' if failing == 'spark' else 'sleep 30'}
}}
'''
    result = run_legs(tmp_path, bodies)
    assert result.returncode != 0
    assert f'[{failing}] release build leg failed (exit 137)' in result.stderr
    assert 'killed without exit status' in result.stderr
    assert (tmp_path / 'cleanup').exists()
    assert not (tmp_path / 'tail').exists()
    assert not (tmp_path / 'completions').exists()


def test_split_state_and_tail_wiring():
    text = BUILD.read_text()
    invocation = text.index("\nrelease_build_legs\n")
    assert invocation < text.index('echo "== exporting release binaries =="')
    assert invocation < text.index('echo "== distributing fresh Spark inference image =="')
    assert text.index('release_dev_reuse_manifest="$release_leg_log_dir/coordinator-dev-image.json"') < text.index('build_coordinator_release() (')
    assert 'build_coordinator_release() (' in text and 'build_spark_release() (' in text
    assert 'setsid --wait bash -s --' in text
    assert 'trap cleanup_spark_phase EXIT' in text
    assert 'kill -TERM -- "-$$"' in text
    assert '$export_container-probe' in text
    # Manifest generation stays entirely in preparation; workers only verify it.
    assert text.index('--write "$source_manifest"') < text.index('build_coordinator_release() (')
    assert '"${native_build_jobs:-__legacy__}"' in text
    assert 'CMAKE_BUILD_PARALLEL_LEVEL=$native_build_jobs' in text

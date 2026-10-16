#!/usr/bin/env python3
"""CPU-only tests for `push-containers.sh --config`.

The publisher used to hardcode `cuteafd.config`, so the v10 release could only be
published by first mutating the runtime default. `--config FILE` selects the
configuration that names the local image pair while the tag argument and the two
fixed GHCR repositories are unchanged, and the shared SSH transport is untouched.

Nothing here reaches Docker, SSH or a host: `docker` and `ssh` are recording
stubs on PATH. The remote `ssh` stub executes the heredoc body locally, so the
label reads and retags that normally happen on SPARK_0_HOST exercise the same
docker stub and their arguments can be asserted exactly. One test pins the
failure path: a bad `CUTEAFD_RELEASE_SSH_CONFIG` must be refused before the
daemon is queried or any host is contacted.
"""
from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PUSH = REPO / "push-containers.sh"
DEFAULT_CONFIG = REPO / "cuteafd.config"
V10_CONFIG = REPO / "scripts" / "fixtures" / "cuteafd.build-v10.config"

COORDINATOR = "ghcr.io/tpurtell/cuteafd-coordinator"
SPARK = "ghcr.io/tpurtell/cuteafd-spark-expert"

DOCKER_STUB = textwrap.dedent(
    r"""#!/usr/bin/env bash
    set -euo pipefail
    log="${CUTEAFD_TEST_DOCKER_LOG:?CUTEAFD_TEST_DOCKER_LOG must be set}"
    line=""
    for token in "$@"; do
      if [[ -z "$line" ]]; then line="$token"; else line="$line"$'\t'"$token"; fi
    done
    printf '%s\n' "$line" >>"$log"
    case "$1" in
      info) exit 0 ;;
      image)
        [[ "${2:-}" == inspect ]] || exit 1
        shift 2
        format=""
        if [[ "${1:-}" == "-f" ]]; then
          format="$2"
          shift 2
        fi
        [[ -n "${1:-}" ]] || exit 1
        if [[ -n "$format" ]]; then
          case "$format" in
            *org.opencontainers.image.revision*) printf '%s\n' engine-revision ;;
            *org.opencontainers.image.source*) printf '%s\n' https://github.com/tpurtell/cuteafd ;;
            *io.cuteafd.v41.spark_tp_roles*) printf '%s\n' 'tp2;tp3;tp6' ;;
            *) exit 1 ;;
          esac
        fi
        exit 0
        ;;
      tag|push) exit 0 ;;
      *) exit 1 ;;
    esac
    """
)

SSH_STUB = textwrap.dedent(
    r"""#!/usr/bin/env bash
    set -euo pipefail
    log="${CUTEAFD_TEST_SSH_LOG:?CUTEAFD_TEST_SSH_LOG must be set}"
    line=""
    for token in "$@"; do
      if [[ -z "$line" ]]; then line="$token"; else line="$line"$'\t'"$token"; fi
    done
    printf '%s\n' "$line" >>"$log"
    # Drop the option set and the host exactly as OpenSSH would, then run the
    # remote command locally so its heredoc reaches the docker stub.
    while [[ $# -gt 0 ]]; do
      case "$1" in
        -o|-F) shift 2 ;;
        -*) shift ;;
        *) shift; break ;;
      esac
    done
    [[ $# -gt 0 ]] || exit 0
    exec "$@"
    """
)


@pytest.fixture
def harness(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("docker", DOCKER_STUB), ("ssh", SSH_STUB)):
        path = bin_dir / name
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)
    docker_log = tmp_path / "docker.log"
    ssh_log = tmp_path / "ssh.log"
    environment = {
        "PATH": f"{bin_dir}:/usr/bin:/bin:/usr/local/bin",
        "HOME": str(tmp_path),
        "CUTEAFD_TEST_DOCKER_LOG": str(docker_log),
        "CUTEAFD_TEST_SSH_LOG": str(ssh_log),
    }
    for name in ("CUTEAFD_RELEASE_SSH_CONFIG", "CUTEAFD_RELEASE_SPARK_TP_ROLES"):
        environment.pop(name, None)

    def run(*args, env=None, config=None):
        command = [str(PUSH)]
        if config is not None:
            command += ["--config", str(config)]
        command += list(args)
        merged = dict(environment)
        merged.update(env or {})
        return subprocess.run(
            command, cwd=REPO, capture_output=True, text=True, timeout=120,
            env=merged, check=False,
        )

    def docker_calls():
        if not docker_log.exists():
            return []
        return [line.split("\t") for line in docker_log.read_text().splitlines()]

    def ssh_calls():
        if not ssh_log.exists():
            return []
        return [line.split("\t") for line in ssh_log.read_text().splitlines()]

    return run, docker_calls, ssh_calls, tmp_path


def _pushed_refs(calls):
    return [call[1] for call in calls if call and call[0] == "push"]


def _tagged_destinations(calls):
    return [call[2] for call in calls if call and call[0] == "tag"]


def _inspected_refs(calls):
    refs = []
    for call in calls:
        if call[:2] == ["image", "inspect"]:
            refs.append(call[-1])
    return refs


def _ssh_host(call):
    """The first non-option ssh argv element, skipping option values."""
    index = 0
    while index < len(call):
        token = call[index]
        if token in ("-o", "-F", "-i", "-l", "-p"):
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        return token
    return None








def test_explicit_ssh_config_is_honored_without_reaching_a_host(harness):
    run, _, ssh_calls, tmp_path = harness
    ssh_config = tmp_path / "release.config"
    ssh_config.write_text("Include ~/.ssh/config\n", encoding="utf-8")
    result = run("v10", config=V10_CONFIG,
                 env={"CUTEAFD_RELEASE_SSH_CONFIG": str(ssh_config)})
    assert result.returncode == 0, result.stdout + result.stderr
    for call in ssh_calls():
        assert call[0:4] == ["-o", "BatchMode=yes", "-F", str(ssh_config)], call
        assert _ssh_host(call) == "ostrich", call


def test_a_bad_transport_is_refused_before_docker_or_any_host(harness):
    run, docker_calls, ssh_calls, _ = harness
    result = run("v10", config=V10_CONFIG,
                 env={"CUTEAFD_RELEASE_SSH_CONFIG": "/tmp/bad config"})
    assert result.returncode == 2, result.stdout + result.stderr
    assert "canonical absolute path" in result.stderr
    assert docker_calls() == [], "a bad transport must not query the daemon"
    assert ssh_calls() == [], "a bad transport must not contact a host"


def test_missing_or_unknown_config_fails_without_contact(harness):
    run, docker_calls, ssh_calls, tmp_path = harness
    result = run("v10", config=tmp_path / "absent.config")
    assert result.returncode == 2
    assert "configuration file not found" in result.stderr
    assert docker_calls() == []
    assert ssh_calls() == []


@pytest.mark.parametrize("args", [
    (),
    ("--config",),
    ("v10", "v11"),
    ("--config", str(V10_CONFIG)),
    ("--bogus", "v10"),
    ("latest",),
])
def test_usage_errors_exit_two_without_contact(harness, args):
    run, docker_calls, ssh_calls, _ = harness
    result = run(*args)
    assert result.returncode == 2, (args, result.stdout, result.stderr)
    assert docker_calls() == []
    assert ssh_calls() == []




@pytest.mark.parametrize("tag,option,latest", [
    ("v3.0.0-rc1", None, False),
    ("v3.0.0-rc1", "--no-latest", False),
    ("v3.0.0-RC1", None, False),
    ("v3.0.0rc1", None, False),
    ("v3.0.0", None, True),
    ("v3.0.0", "--no-latest", False),
])
def test_latest_is_opt_in_for_release_candidates(harness, tag, option, latest):
    run, docker_calls, _, _ = harness
    result = run(tag, *([option] if option else []), config=V10_CONFIG)
    assert result.returncode == 0, result.stdout + result.stderr
    expected = {f"{COORDINATOR}:{tag}", f"{SPARK}:{tag}"}
    if latest:
        expected |= {f"{COORDINATOR}:latest", f"{SPARK}:latest"}
    assert set(_pushed_refs(docker_calls())) == expected
    assert set(_tagged_destinations(docker_calls())) == expected


@pytest.mark.parametrize("tag", ["v3.0.0-rc1", "v3.0.0-RC1", "v3.0.0rc1"])
def test_release_candidates_refuse_latest_before_contact(harness, tag):
    run, docker_calls, ssh_calls, _ = harness
    result = run(tag, "--latest", config=V10_CONFIG)
    assert result.returncode == 2
    assert "release-candidate tags cannot update latest" in result.stderr
    assert docker_calls() == []
    assert ssh_calls() == []


@pytest.mark.parametrize("options", [("--latest", "--no-latest"), ("--no-latest", "--latest")])
def test_conflicting_latest_options_fail_before_contact(harness, options):
    run, docker_calls, ssh_calls, _ = harness
    result = run(*options, "v3.0.0-rc1", config=V10_CONFIG)
    assert result.returncode == 2
    assert "conflict" in result.stderr
    assert docker_calls() == []
    assert ssh_calls() == []


def test_options_may_precede_or_follow_the_tag(harness):
    run, docker_calls, _, _ = harness
    result = run("v10", "--config", str(V10_CONFIG))
    assert result.returncode == 0, result.stdout + result.stderr
    assert _inspected_refs(docker_calls())[0] == f"{COORDINATOR}:v10"



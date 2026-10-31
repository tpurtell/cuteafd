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
import json
import hashlib
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from launcher_sandbox import copy_launcher_shared_inputs

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
            *org.opencontainers.image.revision*) printf '%s\n' "$CUTEAFD_TEST_REVISION" ;;
            *org.opencontainers.image.source*) printf '%s\n' https://github.com/tpurtell/cuteafd ;;
            *io.cuteafd.release.matrix.sha256*)
              if [[ "$1" == *spark* ]]; then
                printf '%s\n' "${CUTEAFD_TEST_SPARK_MATRIX_SHA256-${CUTEAFD_TEST_MATRIX_SHA256-}}"
              else
                printf '%s\n' "${CUTEAFD_TEST_MATRIX_SHA256-}"
              fi
              ;;
            *io.cuteafd.release.scope*) printf '%s\n' "${CUTEAFD_TEST_SCOPE:-all}" ;;
            *io.cuteafd.v41.spark_tp_roles*) printf '%s\n' 'tp2;tp3;tp6' ;;
            *) exit 1 ;;
          esac
        fi
        exit 0
        ;;
      create) printf '%s\n' "fake-${2##*/}" ;;
      cp)
        source="${2#*:/opt/cuteafd/}"
        cp -a "$CUTEAFD_TEST_INVENTORY/$source" "$3"
        if [[ "$source" == lib ]]; then
          role=coordinator
          [[ "$2" != *spark-expert* ]] || role=expert
          cp -a "$CUTEAFD_TEST_INVENTORY/$role/lib/." "$3/"
        fi
        ;;
      rm|tag|push) exit 0 ;;
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
    exec bash -c "$*"
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
    inventory = tmp_path / "inventory"
    (inventory / "share/V41_EXPERT_WORKSPACES").mkdir(parents=True)
    (inventory / "lib").mkdir()
    for name in ("V41_EXPERT_AOT.json", "V41_FP8_AOT.json", "V41_EXPERT_WORKSPACES/mxfp4_rtx_backbone.json"):
        (inventory / "share" / name).write_text(json.dumps({"variants": [{}]}))
    (inventory / "lib/libcuteafd_native.so").write_bytes(b"fake")
    from test_release_derived_scope import SCOPE, write_floor_packages
    # One fake extraction serves both roles; choose package role via docker image.
    for role in ('coordinator', 'expert'):
        write_floor_packages(inventory / role, role, SCOPE.previous_release_packages()[role])
    (inventory / 'share/PROGRAMS.json').write_text(json.dumps({'programs': [{'family': name} for name in SCOPE.PREVIOUS_RELEASE_FLOOR['MIMO_GEOMETRIES']]}))
    for name in ('V41_EXPERT_WORKSPACES/mxfp4_rtx_tp2.json', 'V41_EXPERT_WORKSPACES/nvfp4_rtx_backbone.json',
                 'V41_EXPERT_WORKSPACES/nvfp4_rtx_tp2.json'):
        (inventory / 'share' / name).write_text(json.dumps({'variants': [{}]}))
    (inventory / 'share/V41_EXPERT_TP_AOT.json').write_text(json.dumps({'spark_tp_roles': ['tp2', 'tp3', 'tp6']}))
    nm = bin_dir / "nm"
    nm.write_text("#!/bin/sh\nprintf '%s\\n' " + ' '.join(
        f'cuteafd_v41{kind}{suffix}_expert_launch' for kind in ('', '_nvfp4')
        for suffix in ('', '_spark_tp2', '_spark_tp3', '_spark_tp6')) + "\n")
    nm.chmod(0o755)
    matrix = tmp_path / "matrix.json"
    config = tmp_path / "matrix.config"
    config.write_text("FAMILY=deepseek_v41\nSPARK_COUNT=0\nRTX_GPUS=1\n")
    matrix.write_text(json.dumps({"entries": [{"name": "fixture-v41", "family": "deepseek_v41", "config": str(config)}]}))
    revision_repo = tmp_path / "revision"
    (revision_repo / "scripts/bench").mkdir(parents=True)
    shutil.copytree(REPO / "scripts/lib", revision_repo / "scripts/lib", ignore=shutil.ignore_patterns('__pycache__'))
    copy_launcher_shared_inputs(REPO, revision_repo)
    (revision_repo / "examples/configs").mkdir(parents=True)
    shutil.copyfile(config, revision_repo / "examples/configs/fixture.config")
    shutil.copyfile(config, revision_repo / "cuteafd.config")
    shutil.copyfile(matrix, revision_repo / "scripts/bench/release-smoke-v4.json")
    subprocess.run(["git", "init", "-q", str(revision_repo)], check=True)
    subprocess.run(["git", "-C", str(revision_repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(revision_repo), "-c", "user.name=Fixture", "-c",
                    "user.email=fixture@example.invalid", "commit", "-qm", "Fixture"], check=True)
    revision = subprocess.check_output(["git", "-C", str(revision_repo), "rev-parse", "HEAD"], text=True).strip()
    git = bin_dir / "git"
    git.write_text('#!/bin/sh\nshift 2\nexec /usr/bin/git -C "$CUTEAFD_TEST_REVISION_REPO" "$@"\n')
    git.chmod(0o755)
    docker_log = tmp_path / "docker.log"
    ssh_log = tmp_path / "ssh.log"
    environment = {
        "PATH": f"{bin_dir}:/usr/bin:/bin:/usr/local/bin",
        "HOME": str(tmp_path),
        "CUTEAFD_TEST_DOCKER_LOG": str(docker_log),
        "CUTEAFD_TEST_INVENTORY": str(inventory),
        "CUTEAFD_RELEASE_MATRIX": str(matrix),
        "CUTEAFD_TEST_MATRIX_SHA256": hashlib.sha256(matrix.read_bytes()).hexdigest(),
        "CUTEAFD_TEST_REVISION": revision,
        "CUTEAFD_TEST_REVISION_REPO": str(revision_repo),
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




def test_push_refuses_incomplete_inventory_before_tagging(harness):
    run, calls, _, root = harness
    (root / "inventory/share/V41_FP8_AOT.json").unlink()
    result = run("v3.0.0", config=V10_CONFIG)
    assert result.returncode != 0
    assert "fixture-v41" in result.stderr and "V41_FP8_AOT" in result.stderr
    assert not _pushed_refs(calls()) and not _tagged_destinations(calls())


@pytest.mark.parametrize("tag", ["v3.0.0", "v3.0.0-rc2", "nightly", "3.0.0", "v3.0.1-hotfix", "v3.0.0-rc.1", "dev-fixture"])
def test_release_tags_refuse_narrow_build_scope(harness, tag):
    run, calls, _, _ = harness
    result = run(tag, config=V10_CONFIG, env={"CUTEAFD_TEST_SCOPE": "deepseek_v41"})
    assert result.returncode == 2
    assert "release tags require scope all" in result.stderr
    assert not _pushed_refs(calls()) and not _tagged_destinations(calls())


def test_dev_tag_accepts_selected_scope_after_inventory_check(harness):
    run, calls, _, _ = harness
    result = run("dev-fixture", "--no-latest", config=V10_CONFIG, env={"CUTEAFD_TEST_SCOPE": "deepseek_v41"})
    assert result.returncode == 0, result.stderr
    assert _pushed_refs(calls())


@pytest.mark.parametrize('tag', ['nightly', '3.0.0', 'v3.0.1-hotfix', 'v3.0.0-rc.1', 'dev', 'wip', 'DEV-fixture'])
def test_non_dev_tags_require_full_scope_even_without_latest(harness, tag):
    run, calls, _, _ = harness
    result = run(tag, '--no-latest', config=V10_CONFIG, env={'CUTEAFD_TEST_SCOPE': 'deepseek_v41'})
    assert result.returncode == 2
    assert 'scope all' in result.stderr
    assert not _pushed_refs(calls()) and not _tagged_destinations(calls())


@pytest.mark.parametrize('digest', ['', '0' * 64, 'not-a-digest'])
@pytest.mark.parametrize('tag', ['v3.0.0-rc1', 'nightly', 'dev-fixture'])
def test_release_and_latest_reject_missing_or_swapped_matrix(harness, digest, tag):
    run, calls, _, _ = harness
    result = run(tag, config=V10_CONFIG, env={'CUTEAFD_TEST_MATRIX_SHA256': digest})
    assert result.returncode == 2
    assert 'matrix' in result.stderr and ('SHA256' in result.stderr or 'sha256' in result.stderr)
    assert not _pushed_refs(calls()) and not _tagged_destinations(calls())


def test_release_ignores_publisher_matrix_override(harness):
    run, calls, _, root = harness
    override = root / 'swapped.json'
    override.write_text('{"entries": []}')
    result = run('v3.0.0-rc1', config=V10_CONFIG, env={'CUTEAFD_RELEASE_MATRIX': str(override)})
    assert result.returncode == 0, result.stderr
    assert _pushed_refs(calls())


def test_dev_matrix_override_cannot_skip_recorded_inventory(harness):
    run, calls, _, root = harness
    matrix = root / 'matrix.json'
    matrix.write_text('{"entries": []}')
    result = run('wip-fixture', '--no-latest', config=V10_CONFIG,
                 env={'CUTEAFD_TEST_SCOPE': 'deepseek_v41'})
    assert result.returncode == 2
    assert 'matrix SHA256' in result.stderr
    assert not _pushed_refs(calls()) and not _tagged_destinations(calls())


def test_image_pair_requires_matching_matrix_digests(harness):
    run, calls, _, _ = harness
    result = run('v3.0.0-rc1', config=V10_CONFIG, env={'CUTEAFD_TEST_SPARK_MATRIX_SHA256': '0' * 64})
    assert result.returncode == 2
    assert 'SHA256 labels are missing or disagree' in result.stderr
    assert not _pushed_refs(calls()) and not _tagged_destinations(calls())

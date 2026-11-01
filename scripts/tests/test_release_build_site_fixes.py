"""Execute the release script's site-specific checks without builds or hardware."""
from pathlib import Path
import os
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[2]


def block(path, start, end):
    source = (REPO / path).read_text()
    return source[source.index(start):source.index(end, source.index(start))]


def shell(code, *args, env=None):
    return subprocess.run(["bash", "-euo", "pipefail", "-c", code, "test", *map(str, args)],
                          env={**os.environ, **(env or {})}, text=True, capture_output=True, timeout=10)


@pytest.mark.parametrize("value", ["0", "-1", "31.5", "on", "=on", "1+2", "", "99999999999999999999"])
def test_remote_free_space_floor_rejects_invalid_values(value):
    code = block("build.sh", 'remote_min_free_gib=', 'native_build_env_args=()')
    result = shell('release_die() { echo "$*" >&2; exit 2; };\n' + code,
                   env={"CUTEAFD_RELEASE_REMOTE_MIN_FREE_GIB": value})
    assert result.returncode == 2
    assert "CUTEAFD_RELEASE_REMOTE_MIN_FREE_GIB" in result.stderr
    assert "arithmetic" not in result.stderr


@pytest.mark.parametrize("value", ["1", "32", "60", "120"])
def test_remote_free_space_floor_accepts_positive_integers(value):
    code = block("build.sh", 'remote_min_free_gib=', 'native_build_env_args=()')
    result = shell('release_die() { exit 2; };\n' + code + '\nprintf "%s" "$remote_min_free_gib"',
                   env={"CUTEAFD_RELEASE_REMOTE_MIN_FREE_GIB": value})
    assert result.returncode == 0, result.stderr
    assert result.stdout == value


def test_remote_free_space_measures_dangling_staging_symlink_target(tmp_path):
    target = tmp_path / "other-filesystem"
    target.mkdir()
    staging = tmp_path / "staging"
    staging.symlink_to(target / "not-created" / "build")
    code = block("build.sh", 'p="$(readlink -m "$1")"', '\nREMOTE\n')
    # Stand in for df so the test observes the chosen filesystem ancestor,
    # without requiring a second mount or enough free space on the test host.
    result = shell('df() { printf "Filesystem Blocks Used Available Capacity Mounted\\nfs 1 0 123 0 %s\\n" "$2"; echo "$2" >&2; };\n' + code, staging)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "123"
    assert result.stderr.strip() == str(target)
    assert not (target / "not-created").exists()


@pytest.mark.parametrize("hosts,rdma,expected", [
    ("seed", False, "single"), ("seed dst", True, "rdma"), ("seed dst", False, "ssh"),
])
def test_image_distribution_uses_rdma_first_with_single_host_and_ssh_fallback(tmp_path, hosts, rdma, expected):
    log = tmp_path / "ssh.log"
    code = block("build.sh", 'rdmapipe_ready=1', '\ncoordinator_revision=')
    setup = '''release_die() { echo "$*" >&2; exit 2; }
release_ssh() {
  printf '%s\\n' "$*" >> "$TEST_LOG"
  if [[ "$*" == *'command -v rdmapipe'* ]]; then [[ "$TEST_RDMA" == yes ]]; return; fi
  cat >/dev/null
}
read -ra RELEASE_BUILD_HOSTS <<< "$TEST_HOSTS"
seed_host=seed
SPARK_EXPERT_DOCKER_INFERENCE=example/image:tag
'''
    result = shell(setup + code, env={"TEST_HOSTS": hosts, "TEST_RDMA": "yes" if rdma else "no", "TEST_LOG": str(log)})
    assert result.returncode == 0, result.stderr
    commands = log.read_text() if log.exists() else ""
    if expected == "single":
        assert "nothing to distribute" in result.stdout
        assert not commands
    else:
        assert "docker image save example/image:tag" in commands
        assert "docker image load" in commands
        assert ("rdmapipe --send" in commands) == (expected == "rdma")
        assert ("rdmapipe --recv" in commands) == (expected == "rdma")


def test_artifacts_script_creates_exl3_directory(tmp_path):
    code = block("scripts/build/build-release-artifacts.sh", 'install -d "$output_dir/exl3"', '\n',)
    result = shell('output_dir="$1"; ' + code, tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "exl3").is_dir()


def test_dockerfile_empty_exl3_glob_does_not_run_verifier(tmp_path):
    code = block("docker/Dockerfile.release", 'for exl3_family in /opt/cuteafd/lib/exl3/exl3-*/; do', 'done &&') + 'done'
    code = code.replace('/opt/cuteafd/lib/exl3', str(tmp_path))
    result = shell('python3() { echo "unexpected verifier invocation" >&2; return 99; };\n' + code)
    assert result.returncode == 0, result.stderr
    assert not result.stderr


def test_empty_dist_exl3_does_not_run_verifier(tmp_path):
    code = block("build.sh", 'if compgen -G "$repo_root/dist/$role/exl3/*"', '    fi') + '    fi'
    result = shell('repo_root="$1"; role=spark; python3() { return 99; };\n' + code, tmp_path)
    assert result.returncode == 0, result.stderr

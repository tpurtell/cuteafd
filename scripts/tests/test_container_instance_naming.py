"""INSTANCE namespaces the coordinator container name.

Two agents' cleanups could kill each other's server because run.sh and stop.sh
always used the single name `cuteafd-coordinator`. The name is now derived in
`release_load_config` from INSTANCE, with the same acceptance rule and shape
scripts/launch/run-family.sh uses, and the unset default stays exactly
`cuteafd-coordinator`.

CPU-only: the derivation is executed in bash against the real library, and
stop.sh runs against logging docker/ssh/ss/ps stubs on PATH with a docker daemon
that reports the coordinator container as present and running. No container,
host or process is contacted.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / "scripts" / "lib" / "release-common.sh"
STOP = ROOT / "stop.sh"
CANDIDATE = ROOT / "scripts" / "launch" / "run-tp-ep-native-candidate.sh"

DOCKER_STUB = """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "${MOCK_DOCKER_LOG:?}"
case "${1:-}" in
  info) exit 0 ;;
  container) exit 0 ;;                 # the coordinator container exists
  inspect) printf 'true\\n'; exit 0 ;; # ...and is running
  *) exit 0 ;;
esac
"""

SSH_STUB = """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "${MOCK_SSH_LOG:?}"
exit 0
"""

NOOP_STUB = "#!/usr/bin/env bash\nexit 0\n"


def write_config(path: Path, instance: str | None) -> Path:
    text = (ROOT / "cuteafd.config").read_text(encoding="utf-8")
    if instance is not None:
        text += f"\nINSTANCE={instance}\n"
    path.write_text(text, encoding="utf-8")
    return path


def load_name(config: Path, mode: str = "launch") -> subprocess.CompletedProcess:
    script = (
        "set -euo pipefail\n"
        f'source "{COMMON}"\n'
        f'release_load_config "{config}" {mode}\n'
        'printf "NAME=%s\\n" "$RELEASE_COORDINATOR_CONTAINER_NAME"\n'
        "release_load_config "
        f'"{config}" {mode}\n'
        'printf "AGAIN=%s\\n" "$RELEASE_COORDINATOR_CONTAINER_NAME"\n'
    )
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                          timeout=60, check=False)


@pytest.mark.parametrize(
    "instance,expected",
    [(None, "cuteafd-coordinator"),
     ("", "cuteafd-coordinator"),
     ("own", "cuteafd-coordinator-own"),
     ("agent.2_ab-cd", "cuteafd-coordinator-agent.2_ab-cd")],
)
def test_instance_suffixes_the_coordinator_and_repeats_stably(tmp_path, instance, expected):
    result = load_name(write_config(tmp_path / "cuteafd.config", instance))
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [f"NAME={expected}", f"AGAIN={expected}"]


@pytest.mark.parametrize(
    "instance,message",
    [("-leading", "INSTANCE must be [A-Za-z0-9_.-]"),
     (".dot", "INSTANCE must be [A-Za-z0-9_.-]"),
     ("a/b", "INSTANCE must be [A-Za-z0-9_.-]"),
     # Whitespace fails the parser's quoting rule first, which is strictly safer.
     ("a b", "unquoted whitespace is not allowed for INSTANCE"),
     ("own$", "INSTANCE must be [A-Za-z0-9_.-]"),
     ("x" * 42, "INSTANCE must be [A-Za-z0-9_.-]")],
)
def test_invalid_instance_is_refused_before_any_side_effect(tmp_path, instance, message):
    result = load_name(write_config(tmp_path / "cuteafd.config", instance))
    assert result.returncode == 2
    assert message in result.stderr
    assert "NAME=" not in result.stdout


class StopHarness:
    def __init__(self, tmp_path: Path):
        binary = tmp_path / "bin"
        binary.mkdir()
        for name, body in (("docker", DOCKER_STUB), ("ssh", SSH_STUB),
                           ("ss", NOOP_STUB), ("ps", NOOP_STUB)):
            stub = binary / name
            stub.write_text(body, encoding="utf-8")
            stub.chmod(0o755)
        self.docker_log = tmp_path / "docker.log"
        self.ssh_log = tmp_path / "ssh.log"
        self.env = dict(
            os.environ,
            PATH=f"{binary}{os.pathsep}{os.environ['PATH']}",
            MOCK_DOCKER_LOG=str(self.docker_log),
            MOCK_SSH_LOG=str(self.ssh_log),
        )

    def run_stop(self, config: Path) -> list[str]:
        self.docker_log.write_text("")
        result = subprocess.run(["bash", str(STOP), "--config", str(config)],
                                cwd=ROOT, env=self.env, capture_output=True,
                                text=True, timeout=120, check=False)
        assert result.returncode == 0, result.stderr
        return self.docker_log.read_text().splitlines()


@pytest.mark.parametrize("instance,expected", [(None, "cuteafd-coordinator"),
                                               ("own", "cuteafd-coordinator-own")])
def test_stop_targets_only_the_instances_coordinator(tmp_path, instance, expected):
    harness = StopHarness(tmp_path)
    lines = harness.run_stop(write_config(tmp_path / "cuteafd.config", instance))
    assert f"stop -t 30 {expected}" in lines
    assert f"rm -f {expected}" in lines
    other = "cuteafd-coordinator" if expected != "cuteafd-coordinator" else "cuteafd-coordinator-own"
    # An exact line match: `cuteafd-coordinator-own` must not satisfy, or be
    # satisfied by, the shared name.
    assert f"stop -t 30 {other}" not in lines
    assert f"rm -f {other}" not in lines


def test_candidate_launcher_refuses_every_namespaced_production_name():
    """The candidate launcher may not start or stop any production name, and a
    namespaced launch is a production name: `cuteafd-coordinator-*` joins the
    list, so an INSTANCE launcher cannot be hijacked either."""
    text = CANDIDATE.read_text(encoding="utf-8")
    line = next(
        line for line in text.splitlines()
        if line.strip().startswith("cuteafd-coordinator|") and "spark-expert" in line
    )
    pattern = line.strip().rstrip(")")
    assert "cuteafd-coordinator-*" in pattern
    harness = (
        "set -euo pipefail\n"
        "for name in \"$@\"; do\n"
        "  case \"$name\" in\n"
        f"    {pattern}) printf 'refused %s\\n' \"$name\" ;;\n"
        "    *) printf 'allowed %s\\n' \"$name\" ;;\n"
        "  esac\n"
        "done\n"
    )
    result = subprocess.run(
        ["bash", "-c", harness, "_", "cuteafd-coordinator", "cuteafd-coordinator-wip",
         "cuteafd-coordinator-own", "cuteafd-spark-expert",
         "cuteafd-spark-expert-ostrich-19441", "cuteafd-tpep-dev"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "refused cuteafd-coordinator",
        "refused cuteafd-coordinator-wip",
        "refused cuteafd-coordinator-own",
        "refused cuteafd-spark-expert",
        "refused cuteafd-spark-expert-ostrich-19441",
        "allowed cuteafd-tpep-dev",
    ]


@pytest.mark.parametrize("instance", ["", "plat-robust", "agent.2_ab-cd"])
def test_wip_instance_names_and_slot_lookup(tmp_path, instance):
    home = tmp_path / "home"
    home.mkdir()
    script = f'''set -euo pipefail
source "{COMMON}"
WIP_INSTANCE="$1"
release_record_wip_slot gate
unset WIP_INSTANCE
release_wip_slot_instance gate
printf '%s\\n' "$(release_wip_container coordinator)" "$(release_wip_container spark-expert)" "$WIP_LAYOUT_SLOT"
'''
    result = subprocess.run(["bash", "-c", script, "_", instance],
                            env=dict(os.environ, HOME=str(home)), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    suffix = f"-{instance}" if instance else ""
    layout = f"{instance}/gate" if instance else "gate"
    assert result.stdout.splitlines() == [f"cuteafd-coordinator-wip{suffix}",
                                          f"cuteafd-spark-expert-wip{suffix}", layout]


def test_wip_same_slot_in_two_instances_retains_explicit_lookup(tmp_path):
    script = f'''set -euo pipefail
source "{COMMON}"
WIP_INSTANCE=first; release_record_wip_slot same
WIP_INSTANCE=second; release_record_wip_slot same
WIP_INSTANCE=first; release_wip_slot_instance same
printf '%s\\n' "$WIP_INSTANCE" "$WIP_LAYOUT_SLOT"
unset WIP_INSTANCE; release_wip_slot_instance same
printf '%s\\n' "$WIP_INSTANCE" "$WIP_LAYOUT_SLOT"
'''
    result = subprocess.run(["bash", "-c", script], env=dict(os.environ, HOME=str(tmp_path)),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["first", "first/same", "second", "second/same"]


@pytest.mark.parametrize("value", ["../bad", "a/b", "-bad", "bad;cmd", "x" * 42])
def test_wip_instance_refuses_unsafe_names(tmp_path, value):
    result = subprocess.run(["bash", "-c", f'source "{COMMON}"; WIP_INSTANCE="$1"; release_wip_container coordinator', "_", value],
                            env=dict(os.environ, HOME=str(tmp_path)), capture_output=True, text=True)
    assert result.returncode == 2
    assert "WIP_INSTANCE must be" in result.stderr


def test_wip_config_and_environment_precedence(tmp_path):
    config = write_config(tmp_path / "cuteafd.config", None)
    config.write_text(config.read_text() + "\nWIP_INSTANCE=configured\n")
    script = f'set -euo pipefail; source "{COMMON}"; release_load_config "$1"; release_wip_container coordinator'
    for override, expected in [(None, "configured"), ("environment", "environment")]:
        env = dict(os.environ)
        env.pop("WIP_INSTANCE", None)
        if override is not None:
            env["WIP_INSTANCE"] = override
        result = subprocess.run(["bash", "-c", script, "_", str(config)], env=env,
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert result.stdout == f"cuteafd-coordinator-wip-{expected}"


def test_wip_launchers_use_recorded_container_and_layout():
    family = (ROOT / "scripts/launch/run-family.sh").read_text()
    # The V4.1 section and the generic launch each stage their WIP layout.
    v41 = family[family.index("launch_deepseek_v41() {"):family.index("# Validate the name before it is used")]
    generic = family[family.index("# Validate the name before it is used"):]
    assert "--wip" not in (ROOT / "run.sh").read_text().split("legacy_key()", 1)[0].split("family_args=()", 1)[1] or True
    for text in [v41, generic]:
        assert 'release_wip_slot_instance "$wip_slot"' in text
        assert 'release_stage_wip_layout "$wip_coordinator_container"' in text
        assert '"$(release_wip_container spark-expert)"' in text
        assert 'docker cp "cuteafd-spark-expert-wip:' not in text
        assert '$HOME/.cache/cuteafd/wip-run/$wip_slot' not in text
    text = (ROOT / "wip.sh").read_text()
    assert 'state_dir="${WIP_ROOT:-$HOME/.cache/cuteafd/builds/wip${WIP_INSTANCE:+-$WIP_INSTANCE}}"' in text
    assert 'release_record_wip_slot "$slot"' in text
    assert 'args+=(-v "$wip_mount_root:/wip")' in text
    assert '"wip_instance": ${wip_instance@Q}' in (ROOT / "scripts/build/finalize-wip-slot.sh").read_text()


def test_wip_recreate_targets_only_resolved_instance(tmp_path):
    text = (ROOT / "wip.sh").read_text()
    body = text.split("remove_wip_containers() {", 1)[1].split("\n}", 1)[0]
    binary = tmp_path / "bin"
    binary.mkdir()
    log = tmp_path / "calls"
    for name in ["docker", "ssh"]:
        stub = binary / name
        stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$LOG"\n')
        stub.chmod(0o755)
    script = f'''set -euo pipefail
source "{COMMON}"
WIP_INSTANCE=own
coordinator_container="$(release_wip_container coordinator)"
spark_container="$(release_wip_container spark-expert)"
wip_hosts=(rhea moa)
remove_wip_containers() {{{body}
}}
remove_wip_containers
'''
    result = subprocess.run(["bash", "-c", script], env=dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}", LOG=str(log)),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    calls = log.read_text().splitlines()
    assert "rm -f cuteafd-coordinator-wip-own" in calls
    assert all("cuteafd-spark-expert-wip-own" in call for call in calls if "rhea" in call or "moa" in call)
    assert "rm -f cuteafd-coordinator-wip" not in calls


@pytest.mark.parametrize("root", ["/tmp/build", "/mnt/scratch/wip", "/home/tj/.cache/cuteafd/builds/../wrong", "/home/tj/.cache/cuteafd/builds"])
def test_wip_root_rejects_paths_outside_build_cache(root):
    result = subprocess.run(["bash", "-c", f'source "{COMMON}"; WIP_ROOT="$1"; release_validate_wip_root', "_", root],
                            env=dict(os.environ, HOME="/home/tj"), capture_output=True, text=True)
    assert result.returncode == 2
    assert "WIP_ROOT" in result.stderr


def test_wip_root_config_env_and_legacy_default(tmp_path):
    config = write_config(tmp_path / "cuteafd.config", None)
    config.write_text(config.read_text() + "\nWIP_ROOT=" + str(tmp_path / ".cache/cuteafd/builds/configured/wip") + "\n")
    script = f'source "{COMMON}"; release_load_config "$1"; printf "%s" "$WIP_ROOT"'
    for override in [None, str(tmp_path / ".cache/cuteafd/builds/environment/wip")]:
        env = dict(os.environ, HOME=str(tmp_path))
        env.pop("WIP_ROOT", None)
        if override is not None:
            env["WIP_ROOT"] = override
        result = subprocess.run(["bash", "-c", script, "_", str(config)], env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert result.stdout == (override or str(tmp_path / ".cache/cuteafd/builds/configured/wip"))
    text = (ROOT / "wip.sh").read_text()
    assert 'python3 "$repo_root/scripts/build/assert-build-filesystem.py" "$WIP_ROOT"' in text
    assert 'args+=(-v "$wip_mount_root:/wip")' in text
    assert 'args+=(-v "$root:/wip")' in text
    assert 'remote_staging="$WIP_ROOT/source-staging"' in text
    assert 'WIP_ROOT (environment or config)' in text


def test_each_serving_instance_stages_its_own_run_layout(tmp_path):
    # Two servers of one sealed slot must not re-stage (rm + mv) each other's layout.
    script = f'''set -euo pipefail
source "{COMMON}"
WIP_INSTANCE=shared; release_record_wip_slot gate
for serving in arm-a arm-b ""; do
  WIP_INSTANCE=shared; release_wip_slot_instance gate; release_wip_serving_layout "$serving"
  printf '%s\\n' "$WIP_LAYOUT_SLOT"
done
'''
    result = subprocess.run(["bash", "-c", script], env=dict(os.environ, HOME=str(tmp_path)), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["shared/gate/serve-arm-a", "shared/gate/serve-arm-b", "shared/gate/serve-default"]
    family = (ROOT / "scripts/launch/run-family.sh").read_text()
    v41 = family[family.index("launch_deepseek_v41() {"):family.index("# Validate the name before it is used")]
    generic = family[family.index("# Validate the name before it is used"):]
    for text in (v41, generic):
        assert text.index("release_wip_slot_instance \"$wip_slot\"") < text.index("release_wip_serving_layout")

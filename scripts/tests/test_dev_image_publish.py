"""Toolchain-only publication gates and registry pull behavior (stub Docker)."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('dev_toolchain_test', ROOT / 'scripts/build/dev-toolchain.py')
TOOLCHAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOLCHAIN)


def fixture_repo(tmp_path):
    repo = tmp_path / 'repo'
    for name in ('scripts/build/build-dev-images.sh', 'scripts/build/dev-toolchain.py',
                 'scripts/build/install-dev-cache-tools.sh', 'scripts/build/assert-build-filesystem.py',
                 'scripts/lib/release-common.sh', 'scripts/lib/build-supervision.sh', 'docker/Dockerfile.dev', 'docker/entrypoint.sh', '.dockerignore'):
        dest = repo / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / name, dest)
    (repo / 'cuteafd.config').write_text('SPARK_COUNT=0\nRTX_EXPERT_LAYERS=40\nSPARK_0_HOST=rhea\n')
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'fixture'], check=True)
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    log = tmp_path / 'commands'
    docker = bin_dir / 'docker'
    docker.write_text('''#!/bin/bash
printf '%s\n' "docker $*" >> "$LOG"
if [[ "$1 $2" == 'image inspect' ]]; then
  case "$*" in
    *toolchain.hash*) printf '%s\n' "${LABEL_HASH:-$HASH}" ;;
    *Architecture*) echo amd64 ;;
    *) [[ "${MISSING:-0}" == 0 ]] ;;
  esac
fi
''')
    docker.chmod(0o755)
    ssh = bin_dir / 'ssh'
    ssh.write_text('''#!/bin/bash
printf '%s\n' "ssh $*" >> "$LOG"
case "$*" in
  *'uname -m'*) echo aarch64 ;;
  *'printf %s'*|*'printf %s'*) printf '%s' "$HOME" ;;
  *'bash -s'*) code="$(</dev/stdin)"; if [[ "$code" == *'toolchain label mismatch'* ]]; then
     [[ "${LABEL_HASH:-$HASH}" == "$HASH" ]] || exit 1
   fi ;;
esac
''')
    ssh.chmod(0o755)
    for name in ('rsync',):
        dest = bin_dir / name
        dest.write_text('#!/bin/bash\nprintf "%s\\n" "rsync $*" >> "$LOG"\n')
        dest.chmod(0o755)
    env = {**os.environ, 'PATH': str(bin_dir) + ':' + os.environ['PATH'], 'LOG': str(log),
           'HASH': TOOLCHAIN.identity(repo), 'HOME': str(tmp_path),
           'CUTEAFD_DEV_IMAGE_REMOTE_DIR': str(tmp_path / 'remote')}
    return repo, env, log


def parallel_fixture(tmp_path):
    repo, env, log = fixture_repo(tmp_path)
    bin_dir = Path(env['PATH'].split(':')[0])
    events = tmp_path / 'events'
    events.mkdir()
    (bin_dir / 'docker').write_text('''#!/bin/bash
set -euo pipefail
printf '%s\\n' "docker $*" >>"$LOG"
leg="${STUB_LEG:-coordinator}"
if [[ "$1" == build ]]; then
  printf '%s\\n' "$BASHPID" >"$EVENTS/$leg.pid"
  : >"$EVENTS/$leg.start"
  other=coordinator; [[ "$leg" != coordinator ]] || other=expert
  if [[ "$MODE" == parallel ]]; then
    while [[ ! -e "$EVENTS/$other.start" ]]; do sleep 0.01; done
  elif [[ "$MODE" == sequential && "$leg" == expert ]]; then
    [[ -e "$EVENTS/coordinator.done" ]]
  fi
  if [[ "${FAIL_LEG:-}" == "$leg" ]]; then
    if [[ "$MODE" != sequential ]]; then
      while [[ ! -e "$EVENTS/$other.start" ]]; do sleep 0.01; done
    fi
    exit 17
  elif [[ -n "${FAIL_LEG:-}" ]]; then
    trap '' TERM
    exec sleep 30
  fi
  : >"$EVENTS/$leg.done"
elif [[ "$1" == tag || "$1" == push ]]; then
  if [[ "$MODE" != single ]]; then
    [[ -e "$EVENTS/coordinator.done" && -e "$EVENTS/expert.done" ]]
  fi
elif [[ "$1 $2" == 'image inspect' ]]; then
  case "$*" in
    *toolchain.hash*) [[ "${BAD_CHECK:-}" != "$leg" ]] && echo "$HASH" || echo wrong ;;
    *Architecture*) [[ "$leg" == coordinator ]] && echo amd64 || echo arm64 ;;
    *) exit 1 ;;
  esac
fi
''')
    (bin_dir / 'ssh').write_text('''#!/bin/bash
set -euo pipefail
printf '%s\\n' "ssh $*" >>"$LOG"
while [[ "$1" == -o ]]; do shift 2; done
shift # host
export HOME="$REMOTE_HOME" STUB_LEG=expert
case "$1" in
  uname) echo aarch64 ;;
  python3) : ;; # filesystem probe's input need not run remotely
  setsid) shift; exec setsid "$@" ;;
  bash)
    code="$(</dev/stdin)"
    if [[ "$code" == *'.cancel'* ]]; then : >"$EVENTS/remote-cancel"; fi
    shift; exec bash "$@" <<<"$code" ;;
  *) : ;;
esac
''')
    remote_dir = tmp_path / 'remote'
    remote_dir.mkdir()
    env.update(EVENTS=str(events), MODE='parallel', REMOTE_HOME=str(tmp_path / 'remote-home'))
    return repo, env, log, events


def launch(repo, env, *args):
    return subprocess.run(['bash', str(repo / 'scripts/build/build-dev-images.sh'), '--spark-hosts', 'rhea', *args],
                          env=env, capture_output=True, text=True, timeout=20)


@pytest.mark.parametrize('sequential', [False, True])
def test_dev_legs_overlap_or_respect_sequential_opt_out(tmp_path, sequential):
    repo, env, log, events = parallel_fixture(tmp_path)
    env.update(MODE='sequential' if sequential else 'parallel',
               CUTEAFD_DEV_IMAGE_SEQUENTIAL=str(int(sequential)))
    result = launch(repo, env)
    assert result.returncode == 0, result.stderr
    assert (events / 'coordinator.done').exists() and (events / 'expert.done').exists()
    assert not (events / 'remote-cancel').exists()
    assert 'docker tag' in log.read_text()


@pytest.mark.parametrize('failing', ['coordinator', 'expert'])
def test_dev_failure_names_leg_cancels_other_and_skips_publish(tmp_path, failing):
    repo, env, log, events = parallel_fixture(tmp_path)
    result = launch(repo, {**env, 'FAIL_LEG': failing}, '--publish')
    assert result.returncode != 0
    assert f'[{failing}] dev image build leg failed (exit 17)' in result.stderr
    assert (events / 'remote-cancel').exists()
    from test_release_build_parallel import process_stops
    for leg in ('coordinator', 'expert'):
        # A cancelled leg may still be exiting when the launcher returns.
        assert not process_stops(int((events / f'{leg}.pid').read_text()))
    assert 'docker push' not in log.read_text()
    assert 'imagetools create' not in log.read_text()
    assert 'docker tag' not in log.read_text()


def test_sequential_coordinator_failure_never_starts_or_cancels_expert(tmp_path):
    repo, env, log, events = parallel_fixture(tmp_path)
    result = launch(repo, {**env, 'MODE': 'sequential', 'FAIL_LEG': 'coordinator',
                          'CUTEAFD_DEV_IMAGE_SEQUENTIAL': '1'})
    assert result.returncode != 0
    assert '[coordinator] dev image build leg failed (exit 17)' in result.stderr
    assert not (events / 'expert.start').exists()
    assert not (events / 'remote-cancel').exists()
    assert 'docker tag' not in log.read_text()


@pytest.mark.parametrize('bad_check', ['coordinator', 'expert', ''])
def test_publish_waits_for_both_checks(tmp_path, bad_check):
    repo, env, log, events = parallel_fixture(tmp_path)
    result = launch(repo, {**env, 'BAD_CHECK': bad_check}, '--publish')
    assert (result.returncode == 0) == (not bad_check), result.stderr
    assert ('docker push' in log.read_text()) == (not bad_check)
    if not bad_check:
        assert (events / 'coordinator.done').exists() and (events / 'expert.done').exists()


@pytest.mark.parametrize('role', ['coordinator', 'expert'])
def test_single_role_builds_only_selected_leg(tmp_path, role):
    repo, env, log, events = parallel_fixture(tmp_path)
    result = launch(repo, {**env, 'MODE': 'single'}, '--role', role)
    assert result.returncode == 0, result.stderr
    other = 'expert' if role == 'coordinator' else 'coordinator'
    assert (events / f'{role}.done').exists()
    assert not (events / f'{other}.start').exists()


@pytest.mark.parametrize('sequential', ['0', '1'])
def test_dev_dry_run_reports_plan_without_external_calls(tmp_path, sequential):
    repo, env, log = fixture_repo(tmp_path)
    result = launch(repo, {**env, 'CUTEAFD_DEV_IMAGE_SEQUENTIAL': sequential}, '--dry-run')
    assert result.returncode == 0, result.stderr
    assert ('legs build concurrently' if sequential == '0' else 'sequential: coordinator then expert') in result.stdout
    assert not log.exists()


def test_dirty_publish_refuses_before_build(tmp_path):
    repo, env, log = fixture_repo(tmp_path)
    (repo / 'untracked').write_text('dirty')
    result = launch(repo, env, '--publish')
    assert result.returncode != 0 and 'dirty tree' in result.stderr
    assert not log.exists()


def test_publish_dry_run_names_only_approved_tags(tmp_path):
    repo, env, log = fixture_repo(tmp_path)
    result = launch(repo, env, '--publish', '--dry-run')
    assert result.returncode == 0, result.stderr
    for suffix in ('-amd64', '-arm64', ''):
        assert f'ghcr.io/tpurtell/cuteafd-dev:tc-{env["HASH"]}{suffix}' in result.stdout
    assert 'ghcr.io/tpurtell/cuteafd-dev:latest' in result.stdout
    assert not log.exists()


def test_label_mismatch_never_pushes(tmp_path):
    repo, env, log = fixture_repo(tmp_path)
    # Sequential: the coordinator leg reports the mismatch before the expert leg can
    # win the failure race with its generic leg-failed message.
    result = launch(repo, {**env, 'LABEL_HASH': 'wrong', 'CUTEAFD_DEV_IMAGE_SEQUENTIAL': '1'}, '--publish')
    assert result.returncode != 0 and 'toolchain label mismatch' in result.stderr
    assert 'docker push' not in log.read_text()


def test_publish_preserves_local_defaults_and_scopes_tags(tmp_path):
    repo, env, log = fixture_repo(tmp_path)
    result = launch(repo, env, '--publish')
    assert result.returncode == 0, result.stderr
    commands = log.read_text()
    assert 'docker tag' not in commands
    assert 'cuteafd-coordinator-dev' not in commands
    assert 'cuteafd-spark-expert-dev' not in commands
    for line in commands.splitlines():
        if 'docker push' in line or 'imagetools create' in line:
            assert 'ghcr.io/tpurtell/cuteafd-dev:' in line
            assert ':si-' not in line


@pytest.mark.parametrize('image,missing,pull,ok', [
    ('ghcr.io/tpurtell/cuteafd-dev:tc-test', True, True, True),
    ('ghcr.io/tpurtell/cuteafd-dev@sha256:' + 'a' * 64, False, False, True),
    ('cuteafd-coordinator-dev:si-fcb6706', False, False, True),
    ('cuteafd-coordinator-dev', True, False, False),
])
def test_pull_only_missing_registry_refs(tmp_path, image, missing, pull, ok):
    repo, env, log = fixture_repo(tmp_path)
    result = subprocess.run(['bash', '-c', 'source "$1"; release_ensure_dev_image "$2"',
                             '_', str(repo / 'scripts/lib/release-common.sh'), image],
                            env={**env, 'MISSING': str(int(missing))}, capture_output=True, text=True)
    assert (result.returncode == 0) == ok
    assert ('docker pull' in log.read_text()) == pull


def test_identity_excludes_kernel_pin_and_includes_tool_versions(tmp_path):
    repo, _, _ = fixture_repo(tmp_path)
    before = TOOLCHAIN.identity(repo)
    (repo / 'third_party').mkdir()
    (repo / 'third_party/sparkinfer.lock.json').write_text('new pin')
    assert TOOLCHAIN.identity(repo) == before
    dockerfile = repo / 'docker/Dockerfile.dev'
    dockerfile.write_text(dockerfile.read_text().replace('1.98.1', '1.98.2'))
    assert TOOLCHAIN.identity(repo) != before

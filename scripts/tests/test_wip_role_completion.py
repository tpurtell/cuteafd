"""Fresh single-role builds succeed; paired and cloned slots stay strict."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('role,arch', [('coordinator', '120'), ('expert', '121')])
@pytest.mark.parametrize('value', [None, 'ON', 'OFF', 'on', 'ON; touch /bad'])
def test_wip_audio_aot_reaches_both_native_architectures(role, arch, value):
    script = (ROOT / 'scripts/build/build-wip-artifacts.sh').read_text()
    options = 'exl3_aot=' + script.split('exl3_aot=', 1)[1].split('[[ "$cuda_arch"', 1)[0]
    cmake = 'cmake \\\n' + script.split('\ncmake \\\n', 1)[1].split('\ncmake --build', 1)[0]
    env = {key: val for key, val in os.environ.items() if not key.startswith('CUTEAFD_WIP_')}
    if value is not None:
        env['CUTEAFD_WIP_AUDIO_AOT'] = value
    command = ('set -euo pipefail\nrole=$1; cuda_arch=$2; source_dir=fixture; build_dir=fixture\n'
               'coordinator_aot=OFF; xgrammar=OFF; spark_tp_roles=; expert_families=; bf16_families=\n'
               'compiler_cache_cmake_args=()\n'
               'cmake() { printf "%s\\n" "$@"; }\n' + options + '\n' + cmake)
    result = subprocess.run(['bash', '-c', command, 'test', role, arch], env=env,
                            capture_output=True, text=True, timeout=10)
    if value not in (None, 'ON', 'OFF'):
        assert result.returncode == 2
        assert 'CUTEAFD_WIP_AUDIO_AOT must be ON or OFF' in result.stderr
        assert not result.stdout
        return
    assert result.returncode == 0, result.stderr
    assert f'-DCUTEAFD_ENABLE_AUDIO_AOT={value or "OFF"}' in result.stdout.splitlines()
    assert f'-DCUTEAFD_CUDA_ARCHITECTURES={arch}' in result.stdout.splitlines()


@pytest.mark.parametrize('value', [None, 'ON', 'OFF', 'on', 'ON; touch /bad'])
def test_wip_audio_opt_in_is_validated_before_operations_and_forwarded(value):
    script = (ROOT / 'wip.sh').read_text()
    validation = 'audio_aot=' + script.split('audio_aot=', 1)[1].split('\nbf16_families=', 1)[0]
    env = {key: val for key, val in os.environ.items() if not key.startswith('CUTEAFD_WIP_')}
    if value is not None:
        env['CUTEAFD_WIP_AUDIO_AOT'] = value
    coordinator = 'build_coordinator() {' + script.split('build_coordinator() {', 1)[1].split('\nbuild_expert()', 1)[0]
    expert = 'build_expert() (' + script.split('build_expert() (', 1)[1].split('\ncase "$role"', 1)[0]
    command = ('set -euo pipefail\nrelease_die() { printf "%s\\n" "$*" >&2; exit 2; }\n' + validation + '\n'
               'slot=test; coordinator_container=coordinator; spark_container=spark; seed_host=fixture\n'
               'COORDINATOR_DOCKER_DEV=fixture; SPARK_EXPERT_DOCKER_DEV=fixture\n'
               'wip_spark_tp_roles=; bf16_families=\n'
               'sync_local_source() { :; }; sync_seed_source() { :; }\n'
               'docker() { printf "docker %s\\n" "$*"; }\n'
               'ssh() { printf "ssh %s\\n" "$*"; }\n'
               'timeout() { shift 2; "$@"; }\n' + coordinator + '\n' + expert + '\n'
               'build_coordinator\nbuild_expert\n')
    result = subprocess.run(['bash', '-c', command], env=env, capture_output=True, text=True, timeout=10)
    if value not in (None, 'ON', 'OFF'):
        assert result.returncode == 2
        assert 'CUTEAFD_WIP_AUDIO_AOT must be ON or OFF' in result.stderr
        assert 'docker' not in result.stdout and 'ssh' not in result.stdout
        return
    assert result.returncode == 0, result.stderr
    builds = [line for line in result.stdout.splitlines() if '/build-wip-artifacts.sh' in line]
    assert len(builds) == 3
    assert all(f'CUTEAFD_WIP_AUDIO_AOT={value or "OFF"}' in line for line in builds)
    assert 'coordinator 120' in builds[0]
    assert 'expert 121' in builds[1] and builds[1].endswith(' rust')
    assert 'expert 121' in builds[2] and builds[2].endswith(' native')
    assert script.index(validation) < script.index('release_load_config "$config"')


def check_readiness(tmp_path, role, present, from_slot=''):
    text = (ROOT / 'wip.sh').read_text()
    block = text.split('# wip-slot-readiness:start', 1)[1].split('# wip-slot-readiness:end', 1)[0]
    slots = tmp_path / 'slots'
    for part in present:
        workspace = slots / 'test' / part / 'workspace'
        workspace.mkdir(parents=True)
        (workspace / 'cuteafd.config').write_text('config')
        (workspace.parent / 'FINGERPRINT').write_text('fingerprint')
    bins = tmp_path / 'bin'
    bins.mkdir()
    docker = bins / 'docker'
    docker.write_text('''#!/usr/bin/env python3
import os, subprocess, sys
script = sys.stdin.read().replace('/wip/slots', os.environ['MOCK_SLOTS'])
sys.exit(subprocess.run(['bash', '-s', '--', 'test'], input=script, text=True).returncode)
''')
    docker.chmod(0o755)
    ssh = bins / 'ssh'
    ssh.write_text('#!/usr/bin/env bash\nshift 3\nexec "$@"\n')
    ssh.chmod(0o755)
    return subprocess.run(['bash', '-c', 'set -euo pipefail\nrole=$1; from_slot=$2; slot=test; '
                           'coordinator_container=fixture; spark_container=fixture; seed_host=fixture\n' + block,
                           'test', role, from_slot], capture_output=True, text=True, timeout=10,
                          env={**os.environ, 'MOCK_SLOTS': str(slots), 'PATH': str(bins)+':'+os.environ['PATH']})


@pytest.mark.parametrize('role', ['coordinator', 'expert'])
def test_fresh_role_only_build_needs_only_selected_artifacts(tmp_path, role):
    present = [role if role == 'coordinator' else 'spark-expert']
    result = check_readiness(tmp_path, role, present)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('role', ['coordinator', 'expert', 'both'])
@pytest.mark.parametrize('present', [[], ['coordinator'], ['spark-expert']])
def test_cloned_and_both_role_builds_require_both_artifacts(tmp_path, role, present):
    result = check_readiness(tmp_path, role, present, 'baseline' if role != 'both' else '')
    assert result.returncode != 0


@pytest.mark.parametrize('role', ['coordinator', 'expert', 'both'])
def test_complete_pair_passes(tmp_path, role):
    result = check_readiness(tmp_path, role, ['coordinator', 'spark-expert'], 'baseline')
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('role', ['coordinator', 'expert'])
def test_selected_role_still_requires_its_artifacts(tmp_path, role):
    result = check_readiness(tmp_path, role, [])
    assert result.returncode != 0


def test_wip_state_symlink_is_excluded_from_source_staging(tmp_path):
    source = tmp_path / 'repo'
    source.mkdir()
    cache = tmp_path / 'cache'
    cache.mkdir()
    (source / '.cuteafd-wip').symlink_to(cache, target_is_directory=True)
    text = (ROOT / 'wip.sh').read_text()
    block = text.split('snapshot_args=(\n', 1)[1].split('\n)', 1)[0]
    stage = tmp_path / 'stage'
    result = subprocess.run(['bash', '-c', 'snapshot_args=(\n'+block+'\n)\n'
                             'rsync "${snapshot_args[@]}" "$1/" "$2/"', 'test', str(source), str(stage)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert not (stage / '.cuteafd-wip').is_symlink()
    release_stage = tmp_path / 'release-stage'
    result = subprocess.run([str(ROOT / 'scripts/build/stage-release-source.sh'), str(source), str(release_stage)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert not (release_stage / '.cuteafd-wip').is_symlink()


def transformers_verifier():
    spec = importlib.util.spec_from_file_location(
        'transformers_source', ROOT / 'scripts/build/verify-transformers-source.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def transformers_fixture(tmp_path):
    module = transformers_verifier()
    source = tmp_path / 'transformers'
    for name in module.REQUIRED:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('pinned fixture\n')
    lock = tmp_path / 'transformers.lock.json'
    lock.write_text(json.dumps({
        'schema': 1, 'repository': 'https://github.com/malaiwah/transformers.git',
        'revision': 'a' * 40, 'source_tree_sha256': module.source_tree_sha256(source),
    }))
    return module, source, lock


def test_transformers_lock_matches_checkout_gitlink():
    module = transformers_verifier()
    lock = module.verify(ROOT / 'third_party/transformers', ROOT / 'third_party/transformers.lock.json')
    gitlink = subprocess.check_output(
        ['git', '-C', str(ROOT), 'ls-tree', 'HEAD', 'third_party/transformers'], text=True)
    assert lock['revision'] == gitlink.split()[2]


def test_metadata_free_transformers_freeze_verifies_and_rejects_modified_dependency(tmp_path):
    module, source, lock = transformers_fixture(tmp_path)
    frozen = tmp_path / 'frozen'
    shutil.copytree(source, frozen)
    assert module.verify(frozen, lock)['revision'] == 'a' * 40
    (frozen / 'src/transformers/dependency.py').write_text('changed dependency')
    with pytest.raises(module.VerificationError, match='content does not match'):
        module.verify(frozen, lock)


def test_transformers_requires_include_bytes_inputs_and_confined_links(tmp_path):
    module, source, lock = transformers_fixture(tmp_path)
    (source / module.REQUIRED[-1]).unlink()
    with pytest.raises(module.VerificationError, match='incomplete'):
        module.verify(source, lock)
    (source / module.REQUIRED[-1]).write_text('pinned fixture\n')
    (source / 'escape.py').symlink_to(lock)
    with pytest.raises(module.VerificationError, match='symlink escapes'):
        module.verify(source, lock)


@pytest.mark.parametrize('role', ['coordinator', 'expert'])
@pytest.mark.parametrize('enabled', [False, True])
def test_wip_programs_stage_current_table_or_empty_even_after_previous_opt_in(tmp_path, role, enabled):
    script = (ROOT / 'scripts/build/build-wip-artifacts.sh').read_text()
    block = '# Generic-family tables' + script.split('# Generic-family tables', 1)[1]
    build = tmp_path / 'build'
    output = tmp_path / 'output'
    generated = build / 'native/dsv4_programs/dsv4_programs.json'
    generated.parent.mkdir(parents=True)
    generated.write_text('{"schema":1,"programs":["current"]}\n')
    output.mkdir()
    for name in ('cuteafd', 'libcuteafd_native.so', 'V41_EXPERT_AOT.json',
                 'V41_EXPERT_TP_AOT.json', 'V41_FP8_AOT.json'):
        (output / name).write_text('artifact')
    (output / 'PROGRAMS.json').write_text('stale table')
    env = {key: value for key, value in os.environ.items() if not key.startswith('CUTEAFD_WIP_')}
    env.update(role=role, build_dir=str(build), output_dir=str(output),
               CUTEAFD_WIP_GLMF_AOT='ON' if enabled else 'OFF')
    result = subprocess.run(['bash', '-c', 'set -euo pipefail\n' + block], env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    expected = ['current'] if role == 'coordinator' and enabled else []
    assert json.loads((output / 'PROGRAMS.json').read_text())['programs'] == expected
    sums = (output / 'ARTIFACT_SHA256SUMS').read_text()
    assert hashlib.sha256((output / 'PROGRAMS.json').read_bytes()).hexdigest() + '  PROGRAMS.json' in sums
    if role == 'coordinator' and enabled:
        generated.unlink()
        result = subprocess.run(['bash', '-c', 'set -euo pipefail\n' + block], env=env,
                                capture_output=True, text=True, timeout=10)
        assert result.returncode != 0


def test_pinned_transformers_digest_invalidates_rust_with_old_source_mtimes(tmp_path):
    script = (ROOT / 'scripts/build/build-wip-artifacts.sh').read_text()
    block = 'wip_tree_fingerprint() {' + script.split('wip_tree_fingerprint() {', 1)[1].split('\nrust_phase_marker=', 1)[0]
    source = tmp_path / 'source'
    build = tmp_path / 'build'
    build.mkdir()
    for name in ('rust/main.rs', 'native/a.cc', 'python/a.py', 'third_party/transformers/src/a.py'):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('same source')
    target = source / 'rust/main.rs'
    env = {**os.environ, 'source_dir': str(source), 'build_dir': str(build),
           'transformers_source_digest': 'a' * 64}
    command = ['bash', '-c', 'set -euo pipefail\n' + block +
               '\nprintf "%s" "$wip_current_fingerprint" >"$wip_fingerprint_marker"']
    assert subprocess.run(command, env=env, capture_output=True).returncode == 0
    os.utime(target, (1, 1))
    assert subprocess.run(command, env=env, capture_output=True).returncode == 0
    assert target.stat().st_mtime == 1
    env['transformers_source_digest'] = 'b' * 64
    assert subprocess.run(command, env=env, capture_output=True).returncode == 0
    assert target.stat().st_mtime > 1
    assert (source / 'third_party/transformers/src/a.py').stat().st_mtime > 1


@pytest.mark.parametrize('role', ['coordinator', 'spark-expert'])
def test_finalizer_retains_checked_programs_and_frozen_transformers_identity(tmp_path, role):
    module, transformers, lock = transformers_fixture(tmp_path)
    source = tmp_path / 'source'
    (source / 'third_party').mkdir(parents=True)
    shutil.copytree(transformers, source / 'third_party/transformers')
    shutil.copyfile(lock, source / 'third_party/transformers.lock.json')
    scripts = source / 'scripts/build'
    scripts.mkdir(parents=True)
    for name in ('verify-transformers-source.py', 'verify-release-source-manifest.py'):
        shutil.copyfile(ROOT / 'scripts/build' / name, scripts / name)
    (scripts / 'verify-sparkinfer-source.py').write_text('print("fixture-sparkinfer")\n')
    output = tmp_path / 'output'
    output.mkdir()
    for name in ('cuteafd', 'libcuteafd_native.so', 'V41_EXPERT_AOT.json', 'V41_FP8_AOT.json'):
        (output / name).write_text('artifact')
    (output / 'cuteafd').chmod(0o755)
    (output / 'V41_EXPERT_TP_AOT.json').write_text('{"schema":1,"spark_tp_roles":[]}')
    (output / 'PROGRAMS.json').write_text('{"schema":1,"programs":[]}')
    sums = ''.join(hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + path.name + '\n'
                   for path in sorted(output.iterdir()))
    (output / 'ARTIFACT_SHA256SUMS').write_text(sums)
    script = (ROOT / 'scripts/build/finalize-wip-slot.sh').read_text().replace('/wip/', str(tmp_path / 'wip') + '/')
    command = ['bash', '-c', script, 'finalize', str(source), role, 'WP9', str(output), 'base', 'sha256:base']
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    frozen = tmp_path / 'wip/slots/WP9' / role
    assert (frozen / 'workspace/.cuteafd-wip/PROGRAMS.json').read_bytes() == (output / 'PROGRAMS.json').read_bytes()
    meta = json.loads((frozen / 'META.json').read_text())
    assert meta['transformers_revision'] == 'a' * 40
    assert meta['transformers_source_sha256'] == module.source_tree_sha256(transformers)
    (output / 'PROGRAMS.json').write_text('corrupted')
    assert subprocess.run(command, capture_output=True, text=True).returncode != 0


def test_wip_verifies_transformers_before_both_role_builds_and_frozen_source(tmp_path):
    script = (ROOT / 'scripts/build/build-wip-artifacts.sh').read_text()
    assert script.index('verify-transformers-source.py') < script.index('cargo build')
    assert script.index('verify-transformers-source.py') > script.index('case "$role" in')
    assert '--print-source-digest' in script
    freeze = (ROOT / 'wip.sh').read_text()
    assert freeze.index('verify-transformers-source.py') > freeze.index('rsync "${snapshot_args[@]}"')
    assert freeze.index('verify-transformers-source.py') < freeze.index('ensure_local_image()')
    block = freeze.split('snapshot_args=(\n', 1)[1].split('\n)', 1)[0]
    _, transformers, lock = transformers_fixture(tmp_path)
    repo = tmp_path / 'repo'
    (repo / 'third_party').mkdir(parents=True)
    shutil.copytree(transformers, repo / 'third_party/transformers')
    shutil.copyfile(lock, repo / 'third_party/transformers.lock.json')
    stage = tmp_path / 'stage'
    result = subprocess.run(['bash', '-c', 'snapshot_args=(\n' + block + '\n)\n'
                             'rsync "${snapshot_args[@]}" "$1/" "$2/"', 'test', str(repo), str(stage)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    transformers_verifier().verify(stage / 'third_party/transformers', stage / 'third_party/transformers.lock.json')

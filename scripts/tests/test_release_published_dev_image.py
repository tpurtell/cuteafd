"""CPU-only published toolchain admission, release SSH transport and provenance."""
import importlib.util
import json
import base64
import os
from pathlib import Path
import shlex
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
SELECTOR = ROOT / 'scripts/build/select-dev-image.py'
SPEC = importlib.util.spec_from_file_location('select_dev_image', SELECTOR)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)
IMAGE = 'sha256:' + '1' * 64
DIGEST = 'sha256:' + '2' * 64


def fixture(tmp_path, arch='amd64', failure='', present=False):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    docker = bin_dir / 'docker'
    docker.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys
args = sys.argv[1:]
root = pathlib.Path(os.environ['STUB_ROOT'])
with (root/'calls').open('a') as out: out.write(json.dumps(args)+'\\n')
built = (root/'built').exists()
if args[0] == 'pull':
    if os.environ['FAILURE'] in ('pull', 'missing'):
        print('manifest unknown' if os.environ['FAILURE'] == 'missing' else 'network unavailable', file=sys.stderr); sys.exit(1)
    (root/'pulled').touch()
elif args[0] == 'build':
    (root/'built').touch()
elif args[:2] == ['image', 'inspect']:
    if not (built or os.environ['PRESENT'] == '1' or (root/'pulled').exists()): sys.exit(1)
    obj = json.loads(os.environ['OBJECT'])
    if not built:
        fail = os.environ['FAILURE']
        if fail == 'hash': obj['Config']['Labels']['io.cuteafd.toolchain.hash'] = 'wrong'
        if fail == 'base': obj['Config']['Labels']['io.cuteafd.base.digest'] = 'wrong'
        if fail == 'arch': obj['Architecture'] = 'wrong'
        if fail == 'digest': obj['RepoDigests'] = []
    print(json.dumps([obj]))
''')
    docker.chmod(0o755)
    obj = dict(Id=IMAGE, Architecture=arch, RepoDigests=[MOD.REGISTRY + '@' + DIGEST],
               Config=dict(Labels={'io.cuteafd.toolchain.hash': MOD._toolchain.identity(ROOT),
                                  'io.cuteafd.base.digest': MOD._toolchain.BASE + MOD._toolchain.BASE_DIGESTS[arch]}))
    return dict(os.environ, PATH=str(bin_dir) + ':' + os.environ['PATH'], STUB_ROOT=str(tmp_path),
                OBJECT=json.dumps(obj), FAILURE=failure, PRESENT=str(int(present)))


def command(tmp_path, arch='amd64', mode='registry'):
    return ['python3', str(SELECTOR), 'select', '--source', str(ROOT), '--arch', arch,
            '--local-tag', 'local-dev', '--engine-commit', 'fixture', '--mode', mode,
            '--output', str(tmp_path/'manifest.json')]


def calls(tmp_path):
    return [json.loads(line) for line in (tmp_path/'calls').read_text().splitlines()]


@pytest.mark.parametrize('arch', ['amd64', 'arm64'])
@pytest.mark.parametrize('present', [False, True])
def test_matching_published_image_used_without_build(tmp_path, arch, present):
    result = subprocess.run(command(tmp_path, arch), env=fixture(tmp_path, arch, present=present), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == IMAGE
    assert not any(call[0] == 'build' for call in calls(tmp_path))
    assert any(call[0] == 'pull' for call in calls(tmp_path)) == (not present)
    manifest = json.loads((tmp_path/'manifest.json').read_text())
    assert manifest['registry_reference'].endswith('-' + arch)
    assert manifest['registry_digest'] == DIGEST
    assert manifest['toolchain_hash'] == MOD._toolchain.identity(ROOT)


@pytest.mark.parametrize('arch', ['amd64', 'arm64'])
@pytest.mark.parametrize('failure,reason', [('hash', 'toolchain label mismatch'), ('base', 'base-digest label mismatch'),
                                          ('arch', 'architecture mismatch'), ('pull', 'pull failed'),
                                          ('missing', 'no published image for tc-'), ('digest', 'no registry digest')])
def test_failure_builds_locally_with_reason(tmp_path, arch, failure, reason):
    result = subprocess.run(command(tmp_path, arch), env=fixture(tmp_path, arch, failure), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert reason in result.stderr
    build = next(call for call in calls(tmp_path) if call[0] == 'build')
    assert str(ROOT/'docker/Dockerfile.dev') in build
    assert 'BASE_IMAGE=' + MOD._toolchain.BASE + MOD._toolchain.BASE_DIGESTS[arch] in build
    assert json.loads((tmp_path/'manifest.json').read_text())['source'] == 'build'


@pytest.mark.parametrize('arch', ['amd64', 'arm64'])
def test_build_mode_never_inspects_or_pulls_registry(tmp_path, arch):
    result = subprocess.run(command(tmp_path, arch, 'build'), env=fixture(tmp_path, arch, present=True), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert calls(tmp_path)[0][0] == 'build'
    assert not any(call[0] == 'pull' for call in calls(tmp_path))


def test_remote_leg_selects_and_reuses_exact_image_across_ssh_phases(tmp_path):
    env = fixture(tmp_path, 'arm64')
    # Execute the actual build.sh SSH heredoc using stubs; never invoke an export.
    text = (ROOT/'build.sh').read_text()
    remote = text.split('build_spark_release_leg() {', 1)[1].split("<<'REMOTE'\n", 1)[1].split('\nREMOTE', 1)[0]
    staging = tmp_path/'source'
    (staging/'scripts/build').mkdir(parents=True)
    for name in ('select-dev-image.py', 'dev-toolchain.py', 'install-dev-cache-tools.sh', 'install-nsys.sh', 'build-caches.sh'):
        (staging/'scripts/build'/name).write_bytes((ROOT/'scripts/build'/name).read_bytes())
    (staging/'scripts/build/verify-sparkinfer-source.py').write_text('')
    (staging/'scripts/lib').mkdir(parents=True); (staging/'scripts/lib/cdi-gpus.sh').write_text('')
    (staging/'docker').mkdir()
    for name in ('Dockerfile.dev', 'entrypoint.sh'):
        (staging/'docker'/name).write_bytes((ROOT/'docker'/name).read_bytes())
    ssh = tmp_path/'bin/ssh'
    ssh.write_text('#!/bin/bash\nshift\nexec bash -s -- "$@"\n')
    ssh.chmod(0o755)
    args = [str(staging), 'local-dev', 'release-image', 'fixture', 'pin', 'version'] + ['__legacy__'] * 5
    args += ['dev', 'fixture-export', '__legacy__', '__legacy__', '__legacy__', '__legacy__', '0', 'registry', 'ON', 'on', '90', '123', '7', base64.b64encode(b'export CUTEAFD_RELEASE_SCOPE=all; export CUTEAFD_RELEASE_MATRIX_SHA256=fixture').decode()]
    result = subprocess.run(['ssh', 'seed', *args], input=remote, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    manifest_path = staging/'.cuteafd-release/fixture-export.dev-image.json'
    assert json.loads(manifest_path.read_text())['registry_digest'] == DIGEST
    assert not any(call[0] == 'build' for call in calls(tmp_path))
    args[11] = 'image'
    result = subprocess.run(['ssh', 'seed', *args], input=remote, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    image_build = next(call for call in calls(tmp_path) if call[0] == 'build')
    assert 'io.cuteafd.dev-image.reused=' + IMAGE in image_build
    assert 'io.cuteafd.dev-image.registry.digest=' + DIGEST in image_build


def test_coordinator_release_branch_uses_pulled_id_and_provenance(tmp_path):
    text = (ROOT/'build.sh').read_text()
    branch = text[text.index('release_dev_reuse_label_args=()'):text.index('echo "== compiling coordinator')]
    script = 'set -euo pipefail\n'
    for key, value in dict(repo_root=str(ROOT), COORDINATOR_DOCKER_DEV='local-dev',
                           CUTEAFD_RELEASE_DEV_IMAGE='', release_dev_image_source='registry',
                           engine_commit='fixture', release_leg_log_dir=str(tmp_path),
                           release_dev_reuse_manifest=str(tmp_path/'manifest.json')).items():
        script += f'{key}={shlex.quote(value)}\n'
    script += branch + '\nprintf "%s\\n" "$COORDINATOR_DOCKER_DEV" "${release_dev_reuse_label_args[@]}"\n'
    result = subprocess.run(['bash', '-c', script], env=fixture(tmp_path), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == 'io.cuteafd.dev-image.registry.digest=' + DIGEST
    assert IMAGE in result.stdout.splitlines()
    assert not any(call[0] == 'build' for call in calls(tmp_path))


def test_provenance_merge_and_labels(tmp_path):
    manifest = dict(image_id=IMAGE, source='registry', toolchain_hash='hash', registry_reference='ref', registry_digest=DIGEST)
    coord = tmp_path/'coordinator.json'
    spark = tmp_path/'spark.json'
    coord.write_text(json.dumps(manifest))
    spark.write_text(json.dumps({**manifest, 'architecture': 'arm64'}))
    output = tmp_path/'DEV_IMAGE_REUSE.json'
    subprocess.run(['python3', str(SELECTOR), 'merge', '--coordinator', str(coord), '--spark', str(spark), '--output', str(output)], check=True)
    data = json.loads(output.read_text())
    assert set(data['legs']) == {'coordinator', 'spark-expert'}
    assert all(leg['registry_digest'] == DIGEST for leg in data['legs'].values())
    result = subprocess.run(['python3', str(SELECTOR), 'labels', '--manifest', str(coord)], check=True, text=True, capture_output=True)
    assert 'io.cuteafd.dev-image.registry.digest=' + DIGEST in result.stdout


def test_dry_run_prints_registry_plan_without_docker_or_ssh(tmp_path):
    env = fixture(tmp_path)
    for name in ('ssh', 'rsync', 'nvidia-smi'):
        stub = tmp_path/'bin'/name
        stub.write_text('#!/bin/sh\nexit 91\n')
        stub.chmod(0o755)
    result = subprocess.run([str(ROOT/'build.sh'), '--dry-run'], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert f'pull {MOD.REGISTRY}:tc-{MOD._toolchain.identity(ROOT)}-arm64 on ostrich' in result.stdout
    assert not (tmp_path/'calls').exists()
    default = subprocess.check_output(['python3', str(ROOT/'scripts/build/release-scope.py'),
                                       'default-matrix'], text=True).strip()
    assert f'release matrix: {ROOT/default}' in result.stdout
    assert 'CUTEAFD_EXPORT_IMAGE_ID=${COORDINATOR_DOCKER_DEV}' in result.stdout
    assert 'CUTEAFD_EXPORT_IMAGE_ID=${dev_image}' in result.stdout
    assert result.stdout.count('CUTEAFD_EXPORT_CACHE_STATS=/output/EXPORT_CACHE_STATS.jsonl') == 2


@pytest.mark.parametrize('mode', ['registry', 'build'])
def test_wip_missing_default_images_select_on_both_hosts(tmp_path, mode):
    local = tmp_path/'local'
    remote = tmp_path/'remote'
    local.mkdir(); remote.mkdir()
    local_env = fixture(local, 'amd64')
    remote_env = fixture(remote, 'arm64')
    staging = tmp_path/'staging'
    (staging/'scripts/build').mkdir(parents=True)
    (staging/'docker').mkdir()
    for name in ('select-dev-image.py', 'dev-toolchain.py', 'install-dev-cache-tools.sh', 'install-nsys.sh', 'build-caches.sh'):
        (staging/'scripts/build'/name).write_bytes((ROOT/'scripts/build'/name).read_bytes())
    # This fixture writes under pytest's isolated temporary directory, not NVMe.
    (staging/'scripts/build/assert-build-filesystem.py').write_text('')
    (staging/'scripts/lib').mkdir(parents=True); (staging/'scripts/lib/cdi-gpus.sh').write_text('')
    for name in ('Dockerfile.dev', 'entrypoint.sh'):
        (staging/'docker'/name).write_bytes((ROOT/'docker'/name).read_bytes())
    (staging/'.dockerignore').write_text('')
    home = remote/'home'
    home.mkdir()
    ssh = local/'bin/ssh'
    assignments = ' '.join(f'{key}={shlex.quote(remote_env[key])}' for key in ('STUB_ROOT', 'OBJECT', 'FAILURE', 'PRESENT'))
    ssh.write_text(f'#!/bin/bash\nshift 3\nexport {assignments} HOME={shlex.quote(str(home))}\nexec bash -c "$*"\n')
    ssh.chmod(0o755)
    text = (ROOT/'wip.sh').read_text()
    functions = text.split('ensure_local_image() {', 1)[1].split('source "$repo_root/scripts/lib/build-supervision.sh"', 1)[0]
    script = f'\nset -euo pipefail\nsource {shlex.quote(str(ROOT/"scripts/lib/release-common.sh"))}\n'
    script += 'ensure_local_image() {' + functions
    for key, value in dict(COORDINATOR_DOCKER_DEV='local-dev', SPARK_EXPERT_DOCKER_DEV='local-dev',
                           seed_host='seed', dev_image_run_id='fixture-wip', source_revision='fixture dirty', staging_dir=str(staging),
                           dev_image_logs=str(local), CUTEAFD_RELEASE_DEV_IMAGE_SOURCE=mode).items():
        script += f'\n{key}={shlex.quote(value)}'
    script += '\nensure_local_image\nensure_seed_image\n'
    result = subprocess.run(['bash', '-c', script], env=local_env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    for host in (local, remote):
        host_calls = calls(host)
        assert any(call[0] == 'tag' and call[1] == IMAGE for call in host_calls)
        assert any(call[0] == 'build' for call in host_calls) == (mode == 'build')
        assert any(call[0] == 'pull' for call in host_calls) == (mode == 'registry')
    assert not list((home/'.cache/cuteafd/builds/wip-dev-image').glob('source.*'))


def test_digest_config_uses_safe_local_fallback_tag(tmp_path):
    argv = command(tmp_path, mode='build')
    argv[argv.index('--local-tag') + 1] = MOD.REGISTRY + '@' + DIGEST
    result = subprocess.run(argv, env=fixture(tmp_path), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    build = next(call for call in calls(tmp_path) if call[0] == 'build')
    assert build[build.index('-t') + 1] == f'cuteafd-dev-local:tc-{MOD._toolchain.identity(ROOT)}-amd64'


def test_wip_cancel_stops_remote_session_that_survives_ssh_client(tmp_path):
    import signal

    text = (ROOT/'wip.sh').read_text()
    callback = 'cancel_seed_dev_image() {' + text.split('cancel_seed_dev_image() {', 1)[1].split('\nrelease_configure_ssh_transport', 1)[0]
    # A remote session is independent of the client's process group, as SSH is.
    # Ignore TERM so the callback must use its bounded process-group KILL path.
    root = tmp_path/'home/.cache/cuteafd/builds/wip-dev-image/processes'
    root.mkdir(parents=True)
    run_id = 'fixture-cancel'
    process = subprocess.Popen(['bash', '-c', 'trap "" TERM; sleep 60 & wait', run_id], start_new_session=True)
    (root/(run_id + '.pid')).write_text(str(process.pid))
    bin_dir = tmp_path/'bin'
    bin_dir.mkdir()
    ssh = bin_dir/'ssh'
    ssh.write_text('#!/bin/bash\nshift 3\nexec "$@"\n')
    ssh.chmod(0o755)
    script = 'set -euo pipefail\nrelease_ssh_opts=()\nseed_host=seed\ndev_image_run_id=' + run_id + '\n' + callback + '\ncancel_seed_dev_image\n'
    try:
        result = subprocess.run(['bash', '-c', script], env=dict(os.environ, HOME=str(tmp_path/'home'), PATH=str(bin_dir)+':'+os.environ['PATH']), capture_output=True, text=True, timeout=12)
        assert result.returncode == 0, result.stderr
        process.wait(timeout=2)
        assert process.returncode == -signal.SIGKILL
        assert (root/(run_id + '.cancel')).exists()
        assert not (root/(run_id + '.pid')).exists()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


@pytest.mark.parametrize('role,image_variable', [('coordinator', 'COORDINATOR_DOCKER_DEV'),
                                               ('expert', 'dev_image')])
def test_release_container_uses_admitted_image_for_cache(tmp_path, role, image_variable):
    text = (ROOT/'build.sh').read_text()
    command = text.split(f'/source/scripts/build/build-release-artifacts.sh /source {role}', 1)[0]
    command = command[command.rfind('mapfile -t export_cache_args'):]
    assert f'cuteafd_export_cache_docker_args "${image_variable}" /output/EXPORT_CACHE_STATS.jsonl' in command
    assert '"${export_cache_args[@]}"' in command
    assert f'"${image_variable}"' in command.split('docker run', 1)[1]
    result = subprocess.run(['bash', '-c', f'source "{ROOT}/scripts/build/build-caches.sh"; '
                             'cuteafd_export_cache_docker_args "$1" /output/EXPORT_CACHE_STATS.jsonl',
                             'test', IMAGE], capture_output=True, text=True, check=True, timeout=10)
    assert result.stdout.splitlines() == ['-e', f'CUTEAFD_EXPORT_IMAGE_ID={IMAGE}',
                                          '-e', 'CUTEAFD_EXPORT_CACHE_STATS=/output/EXPORT_CACHE_STATS.jsonl']


def test_release_artifacts_refuse_missing_image_before_compilation(tmp_path):
    output = tmp_path/'output'
    result = subprocess.run(['bash', str(ROOT/'scripts/build/build-release-artifacts.sh'),
                             str(ROOT), 'coordinator', '120', str(output)],
                            env={**os.environ, 'CUTEAFD_EXPORT_IMAGE_ID': '', 'CUTEAFD_EXPORT_CACHE': str(tmp_path/'store'),
                                 'CUTEAFD_EXPORT_CACHE_STATS': str(output/'EXPORT_CACHE_STATS.jsonl')},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert 'requires the admitted compiler image ID' in result.stderr
    assert json.loads((output/'EXPORT_CACHE_STATS.jsonl').read_text())['event'] == 'bypassed'


def test_release_cache_receipts_survive_staging_and_are_checksummed():
    text = (ROOT/'build.sh').read_text()
    assert '"$artifact_dir/EXPORT_CACHE_STATS.jsonl" "$repo_root/dist/EXPORT_CACHE_STATS.coordinator.jsonl"' in text
    assert '$seed_host:$remote_dir/.cuteafd-release-image/EXPORT_CACHE_STATS.jsonl' in text
    for leg in ('coordinator', 'spark'):
        assert f'dist_source_manifest+=(EXPORT_CACHE_STATS.{leg}.jsonl)' in text
    assert 'stats "$repo_root/dist/EXPORT_CACHE_STATS.$leg.jsonl"' in text


@pytest.mark.parametrize('store', ['', 'off'])
def test_release_cache_disabled_does_not_require_image(tmp_path, store):
    text = (ROOT/'scripts/build/build-release-artifacts.sh').read_text()
    admission = text[text.index('mkdir -p "$output_dir"'):text.index('# Where the writable build root')]
    output = tmp_path/'output'
    result = subprocess.run(['bash', '-euc', 'output_dir="$1"; ' + admission, 'test', str(output)],
                            env={**os.environ, 'CUTEAFD_EXPORT_CACHE': store, 'CUTEAFD_EXPORT_IMAGE_ID': '',
                                 'CUTEAFD_EXPORT_CACHE_STATS': str(output/'EXPORT_CACHE_STATS.jsonl')},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads((output/'EXPORT_CACHE_STATS.jsonl').read_text())['reason'] == 'cache disabled'


def test_missing_release_cache_receipts_warn_without_failing(tmp_path):
    text = (ROOT/'build.sh').read_text()
    collection = text[text.index('# Receipts are informational:'):text.index('if [[ -n "$source_manifest" ]]; then',
                                                                          text.index('# Receipts are informational:'))]
    (tmp_path/'dist').mkdir()
    for leg in ('coordinator', 'spark'):
        (tmp_path/f'dist/EXPORT_CACHE_STATS.{leg}.jsonl').write_text('stale')
    result = subprocess.run(['bash', '-euc', 'repo_root="$1"; artifact_dir="$1/missing"; '
                             'seed_host=seed; remote_dir=/missing; dist_source_manifest=(); '
                             'release_sync() { return 1; }; ' + collection +
                             '\n[[ ${#dist_source_manifest[@]} == 0 ]]', 'test', str(tmp_path)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert 'WARNING: coordinator export cache receipt unavailable' in result.stderr
    assert 'WARNING: Spark export cache receipt unavailable' in result.stderr
    assert not list((tmp_path/'dist').iterdir())

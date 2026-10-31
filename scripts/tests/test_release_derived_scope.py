"""CPU-only matrix derivation and fail-closed image coverage contracts."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from launcher_sandbox import copy_launcher_shared_inputs

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('release_scope', REPO / 'scripts/build/release-scope.py')
SCOPE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCOPE)
P = 'CUTEAFD_RELEASE_'
GLM_PROGRAMS = ['dense_trellis_k6', 'glm', 'glm2', 'glm2_stage', 'glm_stage']


@pytest.fixture
def matrix(tmp_path):
    (tmp_path / 'glm.config').write_text('FAMILY=glm5\nMODEL_ID=fixture/glm\nSPARK_COUNT=4\nRTX_GPUS=1\n')
    (tmp_path / 'flash.config').write_text('FAMILY=glm5_flash\nMODEL_ID=fixture/flash-nvfp4\nSPARK_COUNT=4\n')
    path = tmp_path / 'matrix.json'
    path.write_text(json.dumps(dict(defaults=dict(set=dict(AUDIO='off')), entries=[
        dict(name='glm-fp8', family='glm5', config='glm.config', sparks=['rank-a', 'rank-b'], gpus=[0, 1],
             set=dict(SPARK_TP='2', RTX_GPUS='1')),
        dict(name='flash-nvfp4', family='glm5_flash', config='flash.config', sparks=[], gpus=[0]),
    ])))
    # The helper imports WIP derivation from the source repo, not a fixture repo.
    return tmp_path, path


@pytest.fixture
def fixture_repo(matrix, monkeypatch):
    repo, path = matrix
    real = SCOPE.scopes_module(REPO)
    monkeypatch.setattr(SCOPE, 'scopes_module', lambda _: real)
    return repo, path


def test_union_equals_per_entry_derivation(fixture_repo):
    repo, path = fixture_repo
    union, entries = SCOPE.derive(repo, path, environ={})
    per_entry = [SCOPE.derive(repo, path, entry['name'], {})[0] for entry in entries]
    for key, value in union.items():
        if key.endswith('_AOT'):
            assert value == ('ON' if any(e[key] == 'ON' for e in per_entry) or key.removeprefix(P) in SCOPE.PREVIOUS_RELEASE_FLOOR else 'OFF')
        else:
            expected = set().union(*(SCOPE.tokens(e[key]) for e in per_entry))
            expected |= SCOPE.PREVIOUS_RELEASE_FLOOR.get(key.removeprefix(P), set())
            if key == P + 'GENERIC_SPARK_COUNTS':
                assert expected <= SCOPE.tokens(value)
            elif key == P + 'SPARK_TP_ROLES':
                assert SCOPE.tokens(value) == expected | {'tp8'}
            else:
                assert SCOPE.tokens(value) == expected
    assert SCOPE.PREVIOUS_RELEASE_FLOOR['GENERIC_SPARK_COUNTS'] <= SCOPE.tokens(union[P + 'GENERIC_SPARK_COUNTS'])
    assert SCOPE.PREVIOUS_RELEASE_FLOOR['EXPERT_FAMILIES'] <= SCOPE.tokens(union[P + 'EXPERT_FAMILIES'])
    assert SCOPE.requirements(repo, path, 'glm-fp8')[0]['spark_count'] == 4
    assert not SCOPE.requirements(repo, path, 'glm-fp8')[0]['dual']  # set wins over gpus


def test_full_release_expands_width_aliases_without_widening_card_scopes(fixture_repo):
    repo, path = fixture_repo
    full, _ = SCOPE.derive(repo, path, 'all', {})
    counts = SCOPE.tokens(full[P + 'GENERIC_SPARK_COUNTS'])
    assert {f'glm:exl3={world}' for world in (5, 7, 8)} <= counts
    assert {f'glmf:{format}={world}' for format in ('nvfp4', 'nvfp4a4')
            for world in (2, 3, 4, 5, 6)} <= counts
    assert not {f'glmf:{format}={world}' for format in ('nvfp4', 'nvfp4a4')
                for world in (7, 8)} & counts
    card, _ = SCOPE.derive(repo, path, 'glm-fp8', {})
    assert SCOPE.tokens(card[P + 'GENERIC_SPARK_COUNTS']) == {'glm=2'}
    explicit, _ = SCOPE.derive(repo, path, 'explicit', {
        P + 'EXPERT_FAMILIES': 'glm:exl3-k45', P + 'GENERIC_SPARK_COUNTS': 'glm:exl3=4'})
    assert explicit[P + 'GENERIC_SPARK_COUNTS'] == 'glm:exl3=4'


def test_full_release_expands_native_roles_without_widening_explicit_scope(fixture_repo):
    repo, path = fixture_repo
    selected = 'dsv4f:spark;dsv4p:spark'
    full, _ = SCOPE.derive(repo, path, 'all', {P + 'EXPERT_FAMILIES': selected})
    experts = SCOPE.tokens(full[P + 'EXPERT_FAMILIES'])
    assert {'dsv4f:spark_tp8', 'dsv4p:spark_tp8', 'dsv4p:spark_tp3'} <= experts
    assert not {'dsv4f:spark_tp3', 'dsv4f:spark_tp6', 'dsv4p:spark_tp5',
                'dsv4p:spark_tp7'} & experts
    explicit, _ = SCOPE.derive(repo, path, 'explicit', {P + 'EXPERT_FAMILIES': selected})
    assert explicit[P + 'EXPERT_FAMILIES'] == selected


def test_raw_environment_adds_but_does_not_shrink(fixture_repo):
    repo, path = fixture_repo
    added, _ = SCOPE.derive(repo, path, environ={P + 'EXPERT_FAMILIES': 'qwen4:nvfp4', P + 'GLM_AOT': 'OFF',
                                               P + 'MIMO_GEOMETRIES': 'mimof,mimof2'})
    assert added[P + 'GLM_AOT'] == 'ON'
    assert SCOPE.PREVIOUS_RELEASE_FLOOR['EXPERT_FAMILIES'] <= SCOPE.tokens(added[P + 'EXPERT_FAMILIES'])
    assert SCOPE.PREVIOUS_RELEASE_FLOOR['MIMO_GEOMETRIES'] <= SCOPE.tokens(added[P + 'MIMO_GEOMETRIES'])
    empty, _ = SCOPE.derive(repo, path, environ={P + 'EXPERT_FAMILIES': ''})
    assert SCOPE.PREVIOUS_RELEASE_FLOOR['EXPERT_FAMILIES'] <= SCOPE.tokens(empty[P + 'EXPERT_FAMILIES'])


def test_selectors_and_explicit(fixture_repo):
    repo, path = fixture_repo
    family, entries = SCOPE.derive(repo, path, 'glm5_flash', {})
    assert [e['name'] for e in entries] == ['flash-nvfp4']
    assert family[P + 'EXPERT_FAMILIES'] == 'glmf:nvfp4;glmfdense:nvfp4'
    assert family[P + 'GLM_AOT'] == 'OFF'
    assert family[P + 'MIMO_AOT'] == 'OFF'
    assert family[P + 'SPARK_TP_ROLES'] == ''
    both, _ = SCOPE.derive(repo, path, 'glm-fp8,flash-nvfp4', {})
    all_scope = SCOPE.derive(repo, path, 'all', {})[0]
    assert all_scope[P + 'SPARK_TP_ROLES'] == 'tp2;tp3;tp6;tp8'
    for key in SCOPE.PREVIOUS_RELEASE_FLOOR:
        all_scope[P + key] = both[P + key]
    assert both == all_scope
    explicit, entries = SCOPE.derive(repo, path, 'explicit', {P + 'EXPERT_FAMILIES': 'glm:fp8', P + 'GLM_AOT': 'OFF'})
    assert not entries
    assert explicit[P + 'EXPERT_FAMILIES'] == 'glm:fp8'
    assert explicit[P + 'GLM_AOT'] == 'OFF'
    assert SCOPE.requirements(repo, path, 'explicit') == []
    with pytest.raises(ValueError, match='unknown release scope'):
        SCOPE.derive(repo, path, 'missing', {})


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fake_inventory(root, family='glm'):
    import hashlib
    programs = [dict(family=name, name=f'{name}_norm')
                for name in GLM_PROGRAMS if family == 'glm'] or [dict(family=family, name='norm')]
    write_json(root / 'share/PROGRAMS.json', dict(programs=programs))
    for role in ('coordinator', 'expert'):
        target = root / role
        write_json(target / 'share/PROGRAMS.json', dict(programs=programs))
        package = target / 'lib/fp8/fp8-glm'
        layouts = ['tp1', 'tp2']
        files = {}
        for layout in layouts:
            name = f'{layout}/libcuteafd_fp8_moe.so'
            file = package / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b'fake compiled package')
            files[name] = dict(bytes=file.stat().st_size, sha256=hashlib.sha256(file.read_bytes()).hexdigest())
        write_json(package / 'manifest.json', dict(role='spark' if role == 'expert' else role,
                                                  layouts={layout: {} for layout in layouts}, files=files))
    return root


def test_inventory_requires_actual_program_family_and_package(fixture_repo, tmp_path):
    repo, matrix = fixture_repo
    wanted = SCOPE.requirements(repo, matrix, 'glm-fp8')
    root = fake_inventory(tmp_path / 'images')
    SCOPE.check_inventory(root / 'coordinator', 'coordinator', wanted, symbols=set())
    SCOPE.check_inventory(root / 'expert', 'expert', wanted, symbols=set())
    write_json(root / 'coordinator/share/PROGRAMS.json', dict(programs=[dict(family='dsv4f')]))
    with pytest.raises(ValueError, match='entry glm-fp8.*missing program families.*glm'):
        SCOPE.check_inventory(root / 'coordinator', 'coordinator', wanted, symbols=set())
    write_json(root / 'coordinator/share/PROGRAMS.json', dict(programs=[dict(family=family) for family in GLM_PROGRAMS]))
    (root / 'coordinator/lib/fp8/fp8-glm/manifest.json').unlink()
    with pytest.raises(ValueError, match='entry glm-fp8.*manifest.json'):
        SCOPE.check_inventory(root / 'coordinator', 'coordinator', wanted, symbols=set())


@pytest.mark.parametrize('dual', [False, True])
@pytest.mark.parametrize('missing', GLM_PROGRAMS)
def test_glm_inventory_requires_full_static_namespaces(fixture_repo, tmp_path, dual, missing):
    repo, matrix = fixture_repo
    data = json.loads(matrix.read_text())
    data['entries'][0]['set']['RTX_GPUS'] = '2' if dual else '1'
    write_json(matrix, data)
    wanted = SCOPE.requirements(repo, matrix, 'glm-fp8')
    required = GLM_PROGRAMS if dual else [p for p in GLM_PROGRAMS if p not in ('glm2', 'glm2_stage')]
    assert wanted[0]['programs'] == required
    root = fake_inventory(tmp_path / 'images') / 'coordinator'
    manifest = root / 'share/PROGRAMS.json'
    data = json.loads(manifest.read_text())
    if dual:
        data['programs'].append(dict(family='glm', name='glm_context', op='scored_index_topk'))
    write_json(manifest, data)
    SCOPE.check_inventory(root, 'coordinator', wanted, symbols=set())
    data['programs'] = [p for p in data['programs'] if p['family'] != missing]
    write_json(manifest, data)
    if missing not in required:
        SCOPE.check_inventory(root, 'coordinator', wanted, symbols=set())
        return
    with pytest.raises(ValueError, match=f'missing program families.*{missing}'):
        SCOPE.check_inventory(root, 'coordinator', wanted, symbols=set())


def test_glm_context_program_must_not_be_satisfied_by_neutral_namespace(fixture_repo, tmp_path):
    repo, matrix = fixture_repo
    data = json.loads(matrix.read_text())
    data['entries'][0]['set']['RTX_GPUS'] = '2'
    write_json(matrix, data)
    wanted = SCOPE.requirements(repo, matrix, 'glm-fp8')
    root = fake_inventory(tmp_path / 'images') / 'coordinator'
    manifest = root / 'share/PROGRAMS.json'
    data = json.loads(manifest.read_text())
    data['programs'].append(dict(family='dense_trellis_k6', op='scored_index_topk'))
    write_json(manifest, data)
    with pytest.raises(ValueError, match='missing context-split programs'):
        SCOPE.check_inventory(root, 'coordinator', wanted, symbols=set())


def test_inventory_rejects_missing_spark_layout_and_corruption(fixture_repo, tmp_path):
    repo, matrix = fixture_repo
    wanted = SCOPE.requirements(repo, matrix, 'glm-fp8')
    root = fake_inventory(tmp_path / 'images') / 'expert'
    manifest = root / 'lib/fp8/fp8-glm/manifest.json'
    data = json.loads(manifest.read_text())
    del data['layouts']['tp2']
    write_json(manifest, data)
    with pytest.raises(ValueError, match='entry glm-fp8.*missing layouts.*tp2'):
        SCOPE.check_inventory(root, 'expert', wanted, symbols=set())
    data['layouts']['tp2'] = {}
    write_json(manifest, data)
    (root / 'lib/fp8/fp8-glm/tp2/libcuteafd_fp8_moe.so').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='hash mismatch'):
        SCOPE.check_inventory(root, 'expert', wanted, symbols=set())


def test_exact_spark_slices_are_not_satisfied_by_padded_package(tmp_path):
    root = fake_inventory(tmp_path / 'images') / 'expert'
    path = root / 'lib/fp8/fp8-glm/manifest.json'
    manifest = json.loads(path.read_text())
    manifest['layouts']['tp6'] = {'intermediate': 2048}
    write_json(path, manifest)
    with pytest.raises(ValueError, match='missing layouts.*tp6-w256.*tp6-w384'):
        SCOPE.check_package(root, 'fp8/fp8-glm', ['tp6'], 'expert')


def test_resources_do_not_override_config_topology(fixture_repo):
    repo, matrix = fixture_repo
    entry = SCOPE.requirements(repo, matrix, 'flash-nvfp4')[0]
    assert entry['spark_count'] == 4  # resource list is empty; config still wins
    assert not entry['dual']


def test_coordinator_requires_qwen_exact_width_siblings(tmp_path):
    root = fake_inventory(tmp_path / 'images') / 'coordinator'
    path = root / 'lib/fp8/fp8-glm/manifest.json'
    manifest = json.loads(path.read_text())
    manifest['layouts']['tp2'] = {'intermediate': 640}
    write_json(path, manifest)
    with pytest.raises(ValueError, match='missing layouts.*tp2-w256.*tp2-w384'):
        SCOPE.check_package(root, 'fp8/fp8-glm', ['tp1', 'tp2'], 'coordinator')


def test_stdin_checker_has_no_repository_or_working_directory_dependency(tmp_path):
    import subprocess
    helper = REPO / 'scripts/build/release-scope.py'
    result = subprocess.run(['python3', '-', 'check-files', '--role', 'coordinator', '--root', str(tmp_path),
                             '--requirements', '[]'], input=helper.read_text(), cwd='/', text=True,
                            capture_output=True)
    # It gets as far as the native inventory read, not __file__ resolution.
    assert 'libcuteafd_native.so' in result.stderr
    assert 'IndexError' not in result.stderr


def test_qwen_dual_coordinator_requires_exact_layouts_without_geometry_metadata(tmp_path):
    wanted = [dict(name='qwen-dual', family='qwen4', programs=['qwen4'], experts=['qwen4:nvfp4'],
                   spark_experts=[], spark_count=0, dual=True, audio=False, context_split=False)]
    root = fake_inventory(tmp_path / 'images', family='qwen4') / 'coordinator'
    import shutil
    base = root / 'lib/fp8/fp8-glm'
    shutil.copytree(base, root / 'lib/fp8/fp8-qwen4-nvfp4')
    with pytest.raises(ValueError, match='qwen-dual.*missing layouts.*tp2-w256.*tp2-w384'):
        SCOPE.check_inventory(root, 'coordinator', wanted, symbols=set())


def test_pending_tags_cleanup_runs_after_success_and_failure(tmp_path):
    import subprocess
    build = (REPO / 'build.sh').read_text()
    cleanup = 'release_pending_cleanup() {' + build.split('release_pending_cleanup() {', 1)[1].split("trap 'exit 130' INT", 1)[0]
    # The extraction retains the parent EXIT trap; image promotion never clears it.
    assert 'trap - EXIT' not in build.split('coordinator_container="$(docker create', 1)[1].split('release_ssh "$seed_host" bash', 1)[0]
    for status in (0, 2):
        log = tmp_path / f'cleanup-{status}.log'
        script = tmp_path / 'ssh'
        script.write_text('#!/bin/sh\nprintf "%s\n" "$*" >> "$CLEANUP_LOG"\n')
        script.chmod(0o755)
        result = subprocess.run(['bash', '-c', f"""set -euo pipefail
export PATH={tmp_path}:/usr/bin:/bin CLEANUP_LOG={log}
COORDINATOR_DOCKER_INFERENCE=coordinator-pending-123
SPARK_EXPERT_DOCKER_INFERENCE=spark-pending-123
RELEASE_BUILD_HOSTS=(fixture-a fixture-b)
release_ssh_opts=(-o BatchMode=yes)
docker() {{ printf '%s\n' "$*" >> "$CLEANUP_LOG"; }}
{cleanup}
exit {status}
"""], capture_output=True, text=True)
        assert result.returncode == status, result.stderr
        lines = log.read_text().splitlines()
        assert lines == ['image rm coordinator-pending-123',
                         '-o BatchMode=yes fixture-a docker image rm spark-pending-123',
                         '-o BatchMode=yes fixture-b docker image rm spark-pending-123']


def test_release_requirements_use_matrix_and_configs_at_build_revision(tmp_path):
    import shutil
    import subprocess
    repo = tmp_path / 'revision-repo'
    shutil.copytree(REPO / 'scripts/lib', repo / 'scripts/lib', ignore=shutil.ignore_patterns('__pycache__'))
    copy_launcher_shared_inputs(REPO, repo)
    config = repo / 'examples/configs/fixture.config'
    config.parent.mkdir(parents=True)
    config.write_text('FAMILY=glm5\nMODEL_ID=fixture/glm\nSPARK_COUNT=4\nRTX_GPUS=1\n')
    matrix = repo / SCOPE.DEFAULT_MATRIX
    write_json(matrix, dict(entries=[dict(name='glm-fixture', family='glm5',
                                         config='examples/configs/fixture.config')]))
    (repo / 'cuteafd.config').write_text('FAMILY=glm5\n')
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Fixture', '-c',
                    'user.email=fixture@example.invalid', 'commit', '-qm', 'Fixture'], check=True)
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    # Import only the freshly archived revision, in an isolated interpreter:
    # neither sys.path nor an already cached helper may hide missing inputs.
    import io
    import sys
    import tarfile
    policy = json.loads((repo / 'scripts/build/spark-package-ranges.json').read_text())
    archive = subprocess.check_output(['git', '-C', str(repo), 'archive', revision, '--',
                                       *policy['shared_inputs']])
    archived = tmp_path / 'archived'
    archived.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
        tree.extractall(archived, filter='data')
    subprocess.run([sys.executable, '-I', '-c',
        "import importlib.util, sys; "
        "spec = importlib.util.spec_from_file_location('archived_scopes', sys.argv[1]); "
        "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
        "assert 8 in module.native_spark_roles('v41')",
        str(archived / 'scripts/lib/wip_aot_scopes.py')],
        check=True, capture_output=True, text=True, timeout=30)
    publisher_helper = repo / 'python/tools/lib/spark_packages.py'
    publisher_helper.unlink()  # Fixture inputs can be linked to the session cache.
    publisher_helper.write_text("raise AssertionError('publisher checkout used')\n")
    config.write_text('FAMILY=glm5\nMODEL_ID=fixture/glm\nSPARK_COUNT=0\nRTX_GPUS=2\n')
    matrix.write_text('{"entries": []}')
    wanted = SCOPE.revision_requirements(repo, revision)
    assert wanted[0]['spark_count'] == 4
    assert not wanted[0]['dual']


def write_floor_packages(root, role, packages):
    import hashlib
    for package, layouts in packages.items():
        path = root / 'lib' / package
        files = {}
        for layout in layouts:
            name = layout + '/libcuteafd_package.so'
            file = path / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b'fake package')
            files[name] = dict(bytes=file.stat().st_size, sha256=hashlib.sha256(file.read_bytes()).hexdigest())
        manifest = dict(role='spark' if role == 'expert' else role, files=files)
        if package.startswith('exl3/'):
            manifest.update(schema='cuteafd.exl3-package.v1', variants=[dict(directory=layout + '/m1') for layout in layouts])
        else:
            manifest['layouts'] = {layout: {} for layout in layouts}
        write_json(path / 'manifest.json', manifest)


def test_floor_inventory_requires_qwen_tp6_and_mimo_bf16(fixture_repo, tmp_path):
    repo, matrix = fixture_repo
    floor = next(e for e in SCOPE.requirements(repo, matrix, 'all') if e['family'] == 'release_floor')
    root = tmp_path / 'inventory'
    packages = floor['packages']['expert']
    assert 'tp6' in packages['fp8/fp8-qwen4-nvfp4']
    assert 'tp6' in packages['fp8/fp8-qwen4-nvfp4a4']
    assert not any(l.startswith('tp6') for l in packages['exl3/exl3-qwen4-k45'])
    write_floor_packages(root, 'expert', packages)
    SCOPE.check_inventory(root, 'expert', [floor], symbols=set())
    path = root / 'lib/fp8/fp8-qwen4-nvfp4/manifest.json'
    manifest = json.loads(path.read_text())
    del manifest['layouts']['tp6']
    write_json(path, manifest)
    with pytest.raises(ValueError, match='v2.0.0-export-floor.*missing layouts.*tp6'):
        SCOPE.check_inventory(root, 'expert', [floor], symbols=set())
    write_floor_packages(root, 'expert', packages)
    (root / 'lib/fp8/fp8-mimo-bf16/manifest.json').unlink()
    with pytest.raises(ValueError, match='v2.0.0-export-floor.*manifest.json'):
        SCOPE.check_inventory(root, 'expert', [floor], symbols=set())
    env = SCOPE.derive(repo, matrix, 'all', {P + 'FP8_MOE_BF16_FAMILIES': ''})[0]
    assert env[P + 'FP8_MOE_BF16_FAMILIES'] == 'mimo'
    assert 'qwen4:nvfp4=6' in SCOPE.tokens(env[P + 'GENERIC_SPARK_COUNTS'])
    assert 'qwen4=6' not in SCOPE.tokens(env[P + 'GENERIC_SPARK_COUNTS'])


def test_format_counts_reach_nvfp4_but_not_exl3(fixture_repo, tmp_path):
    import subprocess
    env = SCOPE.derive(*fixture_repo, 'all', {})[0]
    counts = env[P + 'GENERIC_SPARK_COUNTS']
    for filename, start, end, extra, output in [
        ('fp8_moe.cmake', '    foreach(count IN LISTS CUTEAFD_GENERIC_SPARK_COUNTS)',
         '    if(requested_layouts)', 'set(count_family qwen4)\nset(entry qwen4:nvfp4)', 'requested_layouts'),
        ('exl3.cmake', '  foreach(count IN LISTS CUTEAFD_GENERIC_SPARK_COUNTS)',
         '\n  if(CUTEAFD_EXL3_ROLE STREQUAL "spark" AND', 'set(CUTEAFD_EXL3_GEOMETRY qwen4)', 'requested_counts'),
    ]:
        source = (REPO / 'native/cmake/shared' / filename).read_text()
        block = start + source.split(start, 1)[1].split(end, 1)[0]
        script = tmp_path / (filename + '.test')
        script.write_text(f'set(CUTEAFD_GENERIC_SPARK_COUNTS "{counts}")\n{extra}\n{block}\nmessage("COUNTS=${{{output}}}")\n')
        result = subprocess.run(['cmake', '-P', str(script)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        resolved = result.stderr.strip().removeprefix('COUNTS=').split(';')
        assert ('tp6' in resolved) if filename == 'fp8_moe.cmake' else ('6' not in resolved)


def test_v2_v41_role_floor_requires_native_roles_not_nvfp4(tmp_path):
    # v2.0.0 shipped native V4.1 Spark roles tp2/tp3/tp4/tp6 and no NVFP4 Spark
    # roles; NVFP4 Spark roles are new in v3 and come from matrix entries.
    repo = Path(__file__).resolve().parents[2]
    matrix = repo / 'scripts/bench/release-smoke-v3.json'
    floor = [e for e in SCOPE.requirements(repo, matrix, 'all') if e['name'].startswith('v2.0.0-v41-tp')]
    assert sorted(e['spark_tp'] for e in floor) == [2, 3, 4, 6]
    assert not any(e['nvfp4'] for e in floor)


@pytest.mark.parametrize('quant,count,gpus', [('nvfp4', 4, 1), ('fp8', 6, 2)])
def test_step5_release_derives_native_mixed_packages(tmp_path, monkeypatch, quant, count, gpus):
    real = SCOPE.scopes_module(REPO)
    monkeypatch.setattr(SCOPE, 'scopes_module', lambda _: real)
    (tmp_path / 'step.config').write_text(
        f'FAMILY=step5\nMODEL_ID=fixture/step5-{quant}\nSPARK_COUNT={count}\nRTX_GPUS={gpus}\n')
    matrix = tmp_path / 'matrix.json'
    matrix.write_text(json.dumps(dict(entries=[dict(name='step5', family='step5', config='step.config')])))
    derived, _ = SCOPE.derive(tmp_path, matrix, 'step5', {})
    assert derived[P + 'STEP5_AOT'] == 'ON'
    assert SCOPE.tokens(derived[P + 'EXPERT_FAMILIES']) == {f'step5:{quant}', 'step5:bf16'}
    requirement = SCOPE.requirements(tmp_path, matrix, 'step5')[0]
    assert requirement['programs'] == (['step5', 'step52'] if gpus == 2 else ['step5'])
    checked = []
    monkeypatch.setattr(SCOPE, 'check_package', lambda root, package, layouts, role, verified: checked.append((package, layouts)))
    root = tmp_path / 'image'
    write_json(root / 'share/PROGRAMS.json', dict(programs=[dict(family=family) for family in requirement['programs']]))
    SCOPE.check_inventory(root, 'coordinator', [requirement], symbols=set())
    suffix = '' if quant == 'fp8' else '-nvfp4a4'
    assert sorted(checked) == sorted([(f'fp8/fp8-step5{suffix}', ['tp1', 'tp2'] if gpus == 2 else ['tp1']),
                                      ('fp8/fp8-step5-bf16-weights', ['tp1', 'tp2'] if gpus == 2 else ['tp1'])])
    checked.clear()
    SCOPE.check_inventory(root, 'expert', [requirement], symbols=set())
    assert sorted(checked) == sorted([(f'fp8/fp8-step5{suffix}', [f'tp{count}']),
                                      ('fp8/fp8-step5-bf16-weights', [f'tp{count}'])])


def test_kimi_release_derives_coordinator_programs(tmp_path, monkeypatch):
    real = SCOPE.scopes_module(REPO)
    monkeypatch.setattr(SCOPE, 'scopes_module', lambda _: real)
    (tmp_path / 'kimi.config').write_text(
        'FAMILY=kimi_k3\nMODEL_ID=fixture/Kimi-K3-EXL3-K2\nSPARK_COUNT=6\nRTX_GPUS=1\n')
    matrix = tmp_path / 'matrix.json'
    matrix.write_text(json.dumps(dict(entries=[dict(name='kimi', family='kimi_k3', config='kimi.config')])))
    derived, _ = SCOPE.derive(tmp_path, matrix, 'kimi', {})
    assert derived[P + 'KIMIK3_AOT'] == 'ON'
    assert derived[P + 'STEP5_AOT'] == 'OFF'
    assert SCOPE.requirements(tmp_path, matrix, 'kimi')[0]['programs'] == ['dense_trellis_k6', 'kimi_split', 'kimik3']


def test_unrelated_release_keeps_kimi_disabled(tmp_path, monkeypatch):
    real = SCOPE.scopes_module(REPO)
    monkeypatch.setattr(SCOPE, 'scopes_module', lambda _: real)
    (tmp_path / 'glm.config').write_text(
        'FAMILY=glm5\nMODEL_ID=fixture/GLM-5.3-EXL3-K4\nSPARK_COUNT=4\nRTX_GPUS=1\n')
    matrix = tmp_path / 'matrix.json'
    matrix.write_text(json.dumps(dict(entries=[dict(name='glm', family='glm5', config='glm.config')])))
    derived, _ = SCOPE.derive(tmp_path, matrix, 'glm', {})
    assert derived[P + 'KIMIK3_AOT'] == 'OFF'
    assert derived[P + 'GLM_AOT'] == 'ON'
    assert SCOPE.requirements(tmp_path, matrix, 'glm')[0]['programs'] == [
        p for p in GLM_PROGRAMS if p not in ('glm2', 'glm2_stage')
    ]


def test_build_sh_forwards_every_aot_switch_to_the_coordinator_export():
    """Each derived AOT switch must reach the coordinator build container, or the
    family's programs silently drop out of the release image (Kimi K3, v4.0.0-rc1)."""
    build = (REPO / 'build.sh').read_text()
    export = build.split('/source/scripts/build/build-release-artifacts.sh /source coordinator', 1)[0]
    export = export.rsplit('docker run', 1)[1]
    missing = [s for s in SCOPE.SWITCHES if f'-e "{P}{s}_AOT=' not in export]
    assert not missing, missing


def test_release_image_stages_the_shared_scope_inputs(tmp_path):
    import subprocess
    import sys
    dockerfile = (REPO / 'docker/Dockerfile.release').read_text()
    stage = next(line for line in dockerfile.splitlines()
                 if line.startswith('RUN --mount=type=bind,target=/cuteafd-source '))
    import shlex
    command = shlex.split(stage.split(' python3 ', 1)[1])
    source = tmp_path / 'source'
    copy_launcher_shared_inputs(REPO, source)
    destination = tmp_path / 'image'
    subprocess.run([sys.executable, '-I', *command[:-2], str(source), str(destination)],
                   check=True, capture_output=True, text=True, timeout=30)
    for relative in json.loads((source / 'scripts/build/spark-package-ranges.json').read_text())['shared_inputs']:
        assert (destination / relative).read_bytes() == (REPO / relative).read_bytes()


def test_shared_input_staging_can_precede_explicit_copy(tmp_path):
    import launcher_sandbox
    root = tmp_path / 'repo'
    target = root / 'scripts/lib/release-common.sh'
    target.parent.mkdir(parents=True)
    launcher_sandbox.copy_input(REPO / 'scripts/lib/release-common.sh', target)
    path = root / 'scripts/lib/checkpoint-family.py'
    other = tmp_path / 'other.py'
    other.hardlink_to(path)
    launcher_sandbox.copy_input(REPO / 'scripts/lib/checkpoint-family.py', path)
    assert path.read_bytes() == other.read_bytes() == (REPO / 'scripts/lib/checkpoint-family.py').read_bytes()
    assert path.stat().st_ino == other.stat().st_ino

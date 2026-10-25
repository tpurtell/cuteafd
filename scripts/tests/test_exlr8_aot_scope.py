"""EXLR8_AOT build scope: derivation, pass-through, cache keys and inventories (CPU only)."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/lib'))
import wip_aot_scopes as aot  # noqa: E402
from test_wip_family_aot import config_file, dry_run  # noqa: E402

SPEC = importlib.util.spec_from_file_location('release_scope_exlr8', ROOT / 'scripts/build/release-scope.py')
SCOPE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCOPE)
EXLR8 = dict(FAMILY='glm5', MODEL_ID='kindlingai/glm-5.3-exlr8-k2-k3-k4', EXLR8_MANIFEST='up1',
             SPARK_COUNT='4', RTX_GPUS='1')


def test_manifest_derives_exlr8_beside_glm_without_routed_packages():
    family, scopes = aot.config_scopes(dict(EXLR8))
    assert family == 'glm5'
    # GLM attention/dense programs and context split stay; EXLR8 joins both roles.
    assert scopes['CUTEAFD_WIP_GLM_AOT'] == 'ON'
    assert scopes['CUTEAFD_WIP_EXLR8_AOT'] == 'ON'
    assert scopes['CUTEAFD_WIP_CONTEXT_SPLIT_AOT'] == 'ON'
    # Routed experts are EXLR8 units: no FP8/NVFP4/EXL3 packages or TP counts.
    for key in ('CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES', 'CUTEAFD_WIP_EXPERT_FAMILIES',
                'CUTEAFD_WIP_GENERIC_SPARK_COUNTS', 'CUTEAFD_WIP_SPARK_TP_ROLES'):
        assert scopes[key] == ''


def test_without_manifest_glm_scope_is_unchanged():
    values = {k: v for k, v in EXLR8.items() if k != 'EXLR8_MANIFEST'}
    _, scopes = aot.config_scopes(values)
    assert 'CUTEAFD_WIP_EXLR8_AOT' not in scopes
    assert scopes['CUTEAFD_WIP_EXPERT_FAMILIES'] == 'glm:fp8'
    assert scopes['CUTEAFD_WIP_GENERIC_SPARK_COUNTS'] == 'glm=4'


@pytest.mark.parametrize('family', ['glm5_flash', 'deepseek_v4', 'mimo_v2', 'qwen4', 'step5'])
def test_manifest_on_other_family_fails_closed(family):
    with pytest.raises(ValueError, match='EXLR8_MANIFEST is supported only for glm5'):
        aot.family_scopes([dict(family=family, name='x', config=None)],
                          lambda _: dict(EXLR8_MANIFEST='up1', SPARK_COUNT='4'))


def test_card_union_keeps_other_glm_cards_packages():
    cards = [dict(family='glm5', name='glm-exlr8', config='a'), dict(family='glm5', name='glm-fp8', config='b')]
    configs = {'a': dict(EXLR8), 'b': dict(MODEL_ID='zai-org/GLM-5.3-FP8', SPARK_COUNT='4', RTX_GPUS='1')}
    scopes = aot.family_scopes(cards, configs.__getitem__)
    assert scopes['CUTEAFD_WIP_EXLR8_AOT'] == 'ON'
    assert scopes['CUTEAFD_WIP_EXPERT_FAMILIES'] == 'glm:fp8'


def test_wip_dry_run_derives_and_explicit_override_wins(tmp_path):
    path = config_file(tmp_path, 'glm5', 'glm-5.3-exlr8-k2-k3-k4', 4)
    path.write_text(path.read_text() + 'EXLR8_MANIFEST=up1\n')
    output = dry_run(tmp_path, path)
    assert 'CUTEAFD_WIP_EXLR8_AOT=ON' in output and 'EXLR8 AOT: ON' in output
    off = dry_run(tmp_path, path, {'CUTEAFD_WIP_EXLR8_AOT': 'OFF'})
    assert 'CUTEAFD_WIP_EXLR8_AOT=OFF' in off and 'EXLR8 AOT: OFF' in off
    # An explicit ON builds EXLR8 for a config without a manifest.
    plain = config_file(tmp_path, 'glm5', 'GLM-5.3-FP8', 4)
    assert 'EXLR8 AOT: ON' in dry_run(tmp_path, plain, {'CUTEAFD_WIP_EXLR8_AOT': 'ON'})
    assert 'EXLR8 AOT: OFF' in dry_run(tmp_path, plain)
    with pytest.raises(AssertionError, match='CUTEAFD_WIP_EXLR8_AOT must be ON or OFF'):
        dry_run(tmp_path, plain, {'CUTEAFD_WIP_EXLR8_AOT': 'yes'})


def test_wip_passes_the_switch_to_both_legs_and_seals_only_when_on():
    wip = (ROOT / 'wip.sh').read_text()
    coordinator = wip.split('build_coordinator() {', 1)[1].split('build_expert() (', 1)[0]
    expert = wip.split('build_expert() (', 1)[1].split('\n)\n', 1)[0]
    assert coordinator.count('${wip_exlr8_env[@]+"${wip_exlr8_env[@]}"}') == 2  # build + finalize
    assert "${wip_exlr8_env[1]:+ -e '${wip_exlr8_env[1]}'}" in expert  # quoted remote build command
    assert '${wip_exlr8_env[@]+"${wip_exlr8_env[@]}"} "$spark_container"' in expert  # finalize
    assert "wip_exlr8_env=()\n" in wip and 'wip_exlr8_env=(-e CUTEAFD_WIP_EXLR8_AOT=ON)' in wip
    # The remote-export env file carries every CUTEAFD_* -e of the coordinator build.
    assert "grep -E '^CUTEAFD_[A-Z0-9_]+='" in wip
    finalize = (ROOT / 'scripts/build/finalize-wip-slot.sh').read_text()
    assert 'if [[ "${CUTEAFD_WIP_EXLR8_AOT:-OFF}" == ON ]]; then' in finalize
    assert 'programs-check --native-lib' in finalize and '--exlr8-role' in finalize
    # The check runs after the bounded source read and before the slot is staged.
    assert finalize.index('sha256sum -c ARTIFACT_SHA256SUMS') < finalize.index('programs-check') < finalize.index('mkdir -p "$incoming')


def build_key_lines(env):
    """wip.sh's build-key scope lines for an environment (the loop and the EXLR8 line)."""
    wip = (ROOT / 'wip.sh').read_text()
    block = wip.split('      for key in EXL3_AOT', 1)[1].split('    } | sha256sum', 1)[0]
    script = 'for key in EXL3_AOT' + block
    result = subprocess.run(['bash', '-c', script], env=env, capture_output=True, text=True, check=True)
    return result.stdout


def test_build_key_changes_only_when_exlr8_is_on():
    base = {'PATH': '/usr/bin:/bin', 'CUTEAFD_WIP_GLM_AOT': 'ON'}
    unset = build_key_lines(base)
    assert 'EXLR8' not in unset
    assert build_key_lines({**base, 'CUTEAFD_WIP_EXLR8_AOT': 'OFF'}) == unset
    on = build_key_lines({**base, 'CUTEAFD_WIP_EXLR8_AOT': 'ON'})
    assert on != unset and on.endswith('CUTEAFD_WIP_EXLR8_AOT=ON\n')


def test_native_key_includes_the_switch_through_the_environment():
    build = (ROOT / 'scripts/build/build-wip-artifacts.sh').read_text()
    block = build[build.index('wip_native_key() {'):build.index('\n}\n', build.index('wip_native_key() {'))]
    # Every CUTEAFD_WIP_* in the build environment is keyed; EXLR8 is not excluded.
    assert "grep -E '^CUTEAFD_WIP_[A-Z0-9_]+='" in block
    excluded = re.search(r"grep -vE '\^CUTEAFD_WIP_\(([^)]*)\)='", block).group(1).split('|')
    assert 'EXLR8_AOT' not in excluded
    # Always explicit: a reused build tree's CMake cache must not keep an earlier ON.
    assert '-DCUTEAFD_ENABLE_EXLR8_AOT="$exlr8_aot"' in build and 'exlr8_cmake_args' not in build
    assert 'CUTEAFD_WIP_EXLR8_AOT must be ON or OFF' in build


def test_every_aot_switch_is_explicit_on_cmake():
    # CMake caches -D values in a reused build tree: a switch passed only when ON
    # survives into a later OFF build (an EXLR8 table appeared in a plain GLM slot).
    for script in ('build-wip-artifacts.sh', 'build-release-artifacts.sh'):
        text = (ROOT / 'scripts/build' / script).read_text()
        cmake = text[text.index('\ncmake \\\n'):]
        cmake = cmake[:cmake.index('\n', cmake.index('-DCUTEAFD_CUDA_ARCHITECTURES'))]
        assert '-DCUTEAFD_ENABLE_EXLR8_AOT=' in cmake
        mode = 'WIP' if script == 'build-wip-artifacts.sh' else 'RELEASE'
        assert ('-DCUTEAFD_ENABLE_KIMIK3_AOT="$( [[ "$role" == coordinator ]] && echo '
                f'"${{CUTEAFD_{mode}_KIMIK3_AOT:-OFF}}" || echo OFF)"') in cmake
        assert '[@]+' not in cmake, f'{script}: conditional CMake argument arrays can leave cached values'


def test_wip_artifacts_rejects_invalid_override(tmp_path):
    result = subprocess.run(['bash', str(ROOT / 'scripts/build/build-wip-artifacts.sh'), str(ROOT), 'expert', '121',
                             str(tmp_path / 'b'), str(tmp_path / 'o'), 'rust'],
                            env={'PATH': '/usr/bin:/bin', 'CUTEAFD_WIP_EXLR8_AOT': 'maybe', 'TMPDIR': str(tmp_path),
                                 'CARGO_HOME': str(tmp_path / 'cargo')},
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 2 and 'CUTEAFD_WIP_EXLR8_AOT must be ON or OFF' in result.stderr


def test_spark_programs_manifest_is_installed_on_both_roles():
    build = (ROOT / 'scripts/build/build-wip-artifacts.sh').read_text()
    final = build[build.index('# Generic-family tables must come from this build'):]
    assert '|| "${CUTEAFD_WIP_EXLR8_AOT:-OFF}" == ON ]]; then' in final.split('\n')[4]


def test_cmake_appends_exlr8_last_with_role_by_architecture():
    cmake = (ROOT / 'native/cmake/shared/dsv4_programs.cmake').read_text()
    assert cmake.index('if(CUTEAFD_ENABLE_STEP5_AOT)') < cmake.index('if(CUTEAFD_ENABLE_KIMIK3_AOT)') < cmake.index('if(CUTEAFD_ENABLE_EXLR8_AOT)\n  # Last')
    assert 'set(CUTEAFD_EXLR8_ROLE coordinator)' in cmake and 'set(CUTEAFD_EXLR8_ROLE spark)' in cmake
    # SM121 builds only EXLR8 (never a family table) and only when asked.
    assert 'MATCHES "^121" AND CUTEAFD_ENABLE_EXLR8_AOT AND NOT CUTEAFD_FAMILY_PROGRAMS' in cmake
    assert 'list(APPEND CUTEAFD_DSV4_EXPORT_ARGS --exlr8-role "${CUTEAFD_EXLR8_ROLE}")' in cmake
    top = (ROOT / 'native/CMakeLists.txt').read_text()
    assert top.count('CUTEAFD_ENABLE_STEP5_AOT OR CUTEAFD_ENABLE_KIMIK3_AOT OR CUTEAFD_ENABLE_EXLR8_AOT)') == 2  # include + link
    assert 'CUTEAFD_ENABLE_EXLR8_AOT OR CUTEAFD_FP8_EXPERT_FAMILIES)' in top  # SparkInfer AOT runtime


def test_exporter_forwards_role_to_the_fork_and_appends_last():
    text = (ROOT / 'python/tools/aot/export_b12x_dsv4_aot.py').read_text()
    assert '"--exlr8-role", choices=("all", "coordinator", "spark")' in text
    assert 'exlr8_programs(role=args.exlr8_role)' in text
    # After glmf wide decode and context split, immediately before compilation.
    assert text.index('if args.context_split == "on":') < text.index('if "exlr8" in geometries:') \
        < text.index('for family, suffix, op, params, thunk in work:')
    wrapper = ast.parse((ROOT / 'python/tools/aot/export_b12x_exlr8_aot.py').read_text())
    programs = next(n for n in wrapper.body if isinstance(n, ast.FunctionDef) and n.name == 'programs')
    assert any(isinstance(n, ast.ImportFrom) and n.module == 'b12x.integration.cuteafd.exlr8_glue'
               for n in ast.walk(programs))


def test_release_derives_exlr8_and_passes_it_to_both_legs(tmp_path, monkeypatch):
    (tmp_path / 'exlr8.config').write_text('FAMILY=glm5\nMODEL_ID=kindlingai/glm-5.3-exlr8-k2-k3-k4\n'
                                           'EXLR8_MANIFEST=up1\nSPARK_COUNT=4\nRTX_GPUS=1\n')
    matrix = tmp_path / 'matrix.json'
    matrix.write_text(json.dumps(dict(entries=[dict(name='glm-exlr8', family='glm5', config='exlr8.config',
                                                     sparks=['a', 'b', 'c', 'd'], gpus=[0], set={})])))
    real = SCOPE.scopes_module(ROOT)
    monkeypatch.setattr(SCOPE, 'scopes_module', lambda _: real)
    env, _ = SCOPE.derive(tmp_path, matrix, 'glm-exlr8', {})
    assert env['CUTEAFD_RELEASE_EXLR8_AOT'] == 'ON' and env['CUTEAFD_RELEASE_GLM_AOT'] == 'ON'
    assert env['CUTEAFD_RELEASE_EXPERT_FAMILIES'] == ''
    with pytest.raises(ValueError, match='must be ON or OFF'):
        SCOPE.derive(tmp_path, matrix, 'glm-exlr8', {'CUTEAFD_RELEASE_EXLR8_AOT': 'yes'})
    requirement = SCOPE.requirements(tmp_path, matrix, 'glm-exlr8')[0]
    assert requirement['exlr8'] and requirement['programs'] == ['glm']
    build = (ROOT / 'build.sh').read_text()
    assert build.count('-e "CUTEAFD_RELEASE_EXLR8_AOT=${CUTEAFD_RELEASE_EXLR8_AOT:-OFF}"') == 2
    artifacts = (ROOT / 'scripts/build/build-release-artifacts.sh').read_text()
    assert '-DCUTEAFD_ENABLE_EXLR8_AOT="${CUTEAFD_RELEASE_EXLR8_AOT:-OFF}"' in artifacts


def write_programs(root, family_programs, exlr8_role=None, exlr8=(), recorded=None):
    programs = [dict(family=f, name=f + '_norm') for f in family_programs]
    programs += [dict(family='exlr8', name=n) for n in exlr8]
    manifest = dict(programs=programs, families={})
    if exlr8_role:
        manifest['families']['exlr8'] = dict(role=exlr8_role, programs=list(exlr8 if recorded is None else recorded))
    path = root / 'share/PROGRAMS.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest))


def test_release_inventory_requires_each_roles_exlr8_set(tmp_path):
    wanted = [dict(name='glm-exlr8', family='glm5', programs=['glm'], experts=[], spark_experts=[],
                   spark_count=4, spark_tp=4, dual=False, roles=[], nvfp4=False, audio=False,
                   context_split=False, exlr8=True)]
    coordinator = ['exlr8_shared_a16_serving', 'exlr8_shared_group']
    spark = ['exlr8_a16_serving', 'exlr8_group', 'exlr8_a16_grouped', 'exlr8_a8_grouped',
             'exlr8_grouped_group', 'exlr8_a16_grouped64']
    write_programs(tmp_path / 'c', ['glm'], 'coordinator', coordinator)
    write_programs(tmp_path / 'e', [], 'spark', spark)
    SCOPE.check_inventory(tmp_path / 'c', 'coordinator', wanted, symbols=set())
    SCOPE.check_inventory(tmp_path / 'e', 'expert', wanted, symbols=set())
    write_programs(tmp_path / 'c', ['glm'], 'coordinator', coordinator[:-1], recorded=coordinator)
    with pytest.raises(ValueError, match='differ from the recorded set'):
        SCOPE.check_inventory(tmp_path / 'c', 'coordinator', wanted, symbols=set())
    write_programs(tmp_path / 'e', [], None, ())
    with pytest.raises(ValueError, match='no EXLR8 spark programs'):
        SCOPE.check_inventory(tmp_path / 'e', 'expert', wanted, symbols=set())
    write_programs(tmp_path / 'e', [], 'coordinator', coordinator)  # the other role's table
    with pytest.raises(ValueError, match='no EXLR8 spark programs'):
        SCOPE.check_inventory(tmp_path / 'e', 'expert', wanted, symbols=set())


@pytest.mark.parametrize('key,known', [
    ('EXLR8_MANIFEST', True),
    ('EXLR8_PREFILL_ACTIVATIONS', True),
    ('EXLR8_UNKNOWN', False),
])
def test_launch_key_is_known(key, known):
    result = subprocess.run(['bash', '-c', 'source "$1"; release_known_key "$2"',
                             'exlr8-keys', str(ROOT / 'scripts/lib/release-common.sh'), key],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == (0 if known else 1), result.stderr


def test_kimi_build_key_changes_only_when_enabled():
    base = {'PATH': '/usr/bin:/bin', 'CUTEAFD_WIP_GLM_AOT': 'ON'}
    unset = build_key_lines(base)
    assert 'KIMIK3' not in unset
    assert build_key_lines({**base, 'CUTEAFD_WIP_KIMIK3_AOT': 'OFF'}) == unset
    on = build_key_lines({**base, 'CUTEAFD_WIP_KIMIK3_AOT': 'ON'})
    assert on != unset and 'CUTEAFD_WIP_KIMIK3_AOT=ON\n' in on


def test_kimi_switch_is_forwarded_to_coordinator_and_native_dispatch():
    wip = (ROOT / 'wip.sh').read_text()
    assert '-e "CUTEAFD_WIP_KIMIK3_AOT=${CUTEAFD_WIP_KIMIK3_AOT:-OFF}"' in wip
    build = (ROOT / 'scripts/build/build-wip-artifacts.sh').read_text()
    dispatch = next(line for line in build.splitlines() if line.startswith('elif [[ ( "$role" == coordinator'))
    assert '${CUTEAFD_WIP_KIMIK3_AOT:-OFF}' in dispatch
    cmake = (ROOT / 'native/cmake/shared/dsv4_programs.cmake').read_text()
    block = cmake.split('if(CUTEAFD_ENABLE_KIMIK3_AOT)', 1)[1].split('if(CUTEAFD_ENABLE_EXLR8_AOT)', 1)[0]
    assert 'set(CUTEAFD_PROGRAM_GEOMETRY "kimik3,kimi_k6_dense,kimi_split")' in block
    assert 'set(CUTEAFD_PROGRAM_GEOMETRY "${CUTEAFD_PROGRAM_GEOMETRY},kimik3,kimi_k6_dense,kimi_split")' in block


def test_kimi_native_key_preserves_off_and_keys_on():
    build = (ROOT / 'scripts/build/build-wip-artifacts.sh').read_text()
    block = build[build.index('    env | grep -E'):build.index('    printf', build.index('    env | grep -E'))]
    # Run the actual native-key environment filter, not a copy of its rules.
    base = {'PATH': '/usr/bin:/bin', 'CUTEAFD_WIP_GLM_AOT': 'ON'}
    def keyed(extra):
        return subprocess.check_output(['bash', '-c', block], env={**base, **extra}, text=True)
    unset = keyed({})
    assert keyed({'CUTEAFD_WIP_KIMIK3_AOT': 'OFF'}) == unset
    assert keyed({'CUTEAFD_WIP_KIMIK3_AOT': 'ON'}) != unset

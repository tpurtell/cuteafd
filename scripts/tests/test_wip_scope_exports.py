"""WIP builds export only what their scope's cards run: V4.1 and vision follow the scope (CPU only)."""
from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/lib'))
import wip_aot_scopes as aot  # noqa: E402

V41_COORDINATOR = ('V41_LOCAL_EXPERT', 'V41_TP2_EXPERT', 'V41_FP8', 'V41_ATTENTION', 'V41_HC_LAGGED', 'V41_NARROW')
SCOPE_KEYS = ('V41_AOT', 'VISION_ATTENTION_AOT', 'KIMIK3_AOT', 'GLMF_AOT', 'MUSE_AOT')


def scopes(family, model, count, **extra):
    _, derived = aot.config_scopes(dict(FAMILY=family, MODEL_ID='test/' + model, SPARK_COUNT=str(count),
                                        RTX_GPUS='1', **extra))
    return derived


def test_kimi_scope_requests_no_v41_or_vision_exports():
    derived = scopes('kimi_k3', 'Kimi-K3-EXL3-K2', 6)
    assert derived['CUTEAFD_WIP_KIMIK3_AOT'] == 'ON'
    assert 'CUTEAFD_WIP_V41_AOT' not in derived
    assert 'CUTEAFD_WIP_VISION_ATTENTION_AOT' not in derived
    assert derived['CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES'] == 'kimik3:exl3-k23'


@pytest.mark.parametrize('count,roles', [(4, ''), (6, 'tp6'), (0, '')])
def test_v41_scope_requests_v41_exports_only(count, roles):
    derived = scopes('deepseek_v41', 'DeepSeek-V4.1-Flash', count)
    assert derived['CUTEAFD_WIP_V41_AOT'] == 'ON'
    assert derived['CUTEAFD_WIP_SPARK_TP_ROLES'] == roles
    # V4.1 serves its tower through its own kernels, not the shared vision exports.
    assert 'CUTEAFD_WIP_VISION_ATTENTION_AOT' not in derived


@pytest.mark.parametrize('family,model', [('glm5_flash', 'GLM-5.3-Flash-NVFP4'), ('qwen4', 'Qwen3.8-Flash-Next-EXL3'),
                                          ('muse_spark', 'Muse-Spark-Open-EXL3-K4-Core-K6'),
                                          ('mistral_large4', 'Mistral-Large-4-EXL3-K4')])
def test_vision_families_request_vision_unless_off(family, model):
    assert scopes(family, model, 4)['CUTEAFD_WIP_VISION_ATTENTION_AOT'] == 'ON'
    assert 'CUTEAFD_WIP_VISION_ATTENTION_AOT' not in scopes(family, model, 4, VISION='off')
    assert 'CUTEAFD_WIP_V41_AOT' not in scopes(family, model, 4)


@pytest.mark.parametrize('family,model', [('mimo_v2', 'MiMo-V2.6-Pro'), ('glm5', 'GLM-5.3-EXL3-K4'),
                                          ('step5', 'Step-5-Preview'), ('deepseek_v4', 'DeepSeek-V4-Flash')])
def test_towers_without_shared_vision_attention_request_none(family, model):
    derived = scopes(family, model, 4)
    assert 'CUTEAFD_WIP_VISION_ATTENTION_AOT' not in derived and 'CUTEAFD_WIP_V41_AOT' not in derived


def test_card_union_carries_each_cards_switches():
    cards = [dict(family='kimi_k3', name='kimi', config='k'), dict(family='deepseek_v41', name='v41', config='v'),
             dict(family='glm5_flash', name='glmf', config='g')]
    configs = dict(k=dict(MODEL_ID='test/Kimi-K3-EXL3-K2', SPARK_COUNT='6'),
                   v=dict(MODEL_ID='test/DeepSeek-V4.1-Flash', SPARK_COUNT='6'),
                   g=dict(MODEL_ID='test/GLM-5.3-Flash-NVFP4', SPARK_COUNT='4'))
    derived = aot.family_scopes(cards, configs.__getitem__)
    assert {key: derived.get('CUTEAFD_WIP_' + key) for key in SCOPE_KEYS} == dict(
        V41_AOT='ON', VISION_ATTENTION_AOT='ON', KIMIK3_AOT='ON', GLMF_AOT='ON', MUSE_AOT=None)


def test_release_all_scope_is_unchanged_by_wip_only_switches():
    import importlib.util
    spec = importlib.util.spec_from_file_location('release_scope_wip_only', ROOT / 'scripts/build/release-scope.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    derived, _ = module.derive(ROOT, ROOT / module.DEFAULT_MATRIX, 'all', {})
    assert not any(key.endswith(('_V41_AOT', '_VISION_ATTENTION_AOT')) for key in derived)
    # The release builder keeps both always on.
    release = (ROOT / 'scripts/build/build-release-artifacts.sh').read_text()
    assert '-DCUTEAFD_ENABLE_V41_EXPERT_AOT=ON \\' in release
    assert '-DCUTEAFD_ENABLE_VISION_ATTENTION_AOT="${CUTEAFD_RELEASE_VISION_ATTENTION_AOT:-ON}"' in release
    for flag in V41_COORDINATOR:
        assert f'-DCUTEAFD_ENABLE_{flag}_AOT="$coordinator_aot"' in release


def cmake_flags(role, env):
    """The CMake -D flags build-wip-artifacts.sh passes for ROLE under ENV (no build)."""
    script = (ROOT / 'scripts/build/build-wip-artifacts.sh').read_text()
    roles = script[script.index('case "$role" in'):script.index('[[ "$cuda_arch" =~')]
    cmake = 'cmake \\\n' + script.split('\ncmake \\\n', 1)[1].split('\n# One process budget', 1)[0]
    command = ('set -euo pipefail\nrole=$1; cuda_arch=$2; source_dir=fixture; build_dir=fixture\n'
               'spark_tp_roles="${CUTEAFD_WIP_SPARK_TP_ROLES:-}"; compiler_cache_cmake_args=()\n'
               'cmake() { printf "%s\\n" "$@"; }\n'
               + roles[roles.index('esac') + len('esac'):] + '\n' + cmake)
    clean = {key: value for key, value in os.environ.items() if not key.startswith('CUTEAFD_WIP_')}
    result = subprocess.run(['bash', '-c', command, 'test', role, '120' if role == 'coordinator' else '121'],
                            env={**clean, **env, 'coordinator_aot': 'ON' if role == 'coordinator' else 'OFF',
                                 'xgrammar': 'OFF'},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return dict(line[2:].split('=', 1) for line in result.stdout.splitlines() if line.startswith('-D'))


def flags_for(family, model, count, role='coordinator'):
    derived = scopes(family, model, count)
    env = {key: value for key, value in derived.items()}
    if role == 'expert':
        env['CUTEAFD_WIP_EXPERT_FAMILIES'] = derived['CUTEAFD_WIP_EXPERT_FAMILIES']
    else:
        env['CUTEAFD_WIP_EXPERT_FAMILIES'] = derived['CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES']
        env.pop('CUTEAFD_WIP_SPARK_TP_ROLES', None)
    return cmake_flags(role, env)


def test_kimi_coordinator_flags_turn_v41_and_vision_off():
    flags = flags_for('kimi_k3', 'Kimi-K3-EXL3-K2', 6)
    for flag in V41_COORDINATOR + ('V41_EXPERT', 'V41_NVFP4', 'VISION_ATTENTION'):
        assert flags[f'CUTEAFD_ENABLE_{flag}_AOT'] == 'OFF', flag
    assert flags['CUTEAFD_ENABLE_KIMIK3_AOT'] == 'ON'
    assert flags['CUTEAFD_EXPERT_FAMILIES'] == 'kimik3:exl3-k23'


@pytest.mark.parametrize('role', ['coordinator', 'expert'])
def test_v41_flags_keep_todays_artifact_set(role):
    flags = flags_for('deepseek_v41', 'DeepSeek-V4.1-Flash', 6, role)
    on = 'ON' if role == 'coordinator' else 'OFF'
    for flag in V41_COORDINATOR:
        assert flags[f'CUTEAFD_ENABLE_{flag}_AOT'] == on, flag
    assert flags['CUTEAFD_ENABLE_V41_EXPERT_AOT'] == 'ON'
    assert flags['CUTEAFD_ENABLE_V41_NVFP4_AOT'] == 'ON'
    # V4.1 never needed the shared vision exports; it now omits them.
    assert flags['CUTEAFD_ENABLE_VISION_ATTENTION_AOT'] == 'OFF'


@pytest.mark.parametrize('families,roles', [('dsv4f:rtx_backbone;dsv4f:spark_tp2', ''), ('', 'tp6')])
def test_native_expert_base_stays_on_for_v4_native_families_and_spark_roles(families, roles):
    flags = cmake_flags('expert', {'CUTEAFD_WIP_EXPERT_FAMILIES': families, 'CUTEAFD_WIP_SPARK_TP_ROLES': roles})
    assert flags['CUTEAFD_ENABLE_V41_EXPERT_AOT'] == 'ON'
    assert flags['CUTEAFD_ENABLE_V41_NVFP4_AOT'] == 'OFF'


def test_vision_family_flags_enable_vision_only():
    flags = flags_for('glm5_flash', 'GLM-5.3-Flash-NVFP4', 4)
    assert flags['CUTEAFD_ENABLE_VISION_ATTENTION_AOT'] == 'ON'
    assert all(flags[f'CUTEAFD_ENABLE_{flag}_AOT'] == 'OFF' for flag in V41_COORDINATOR + ('V41_EXPERT',))


@pytest.mark.parametrize('key', ['CUTEAFD_WIP_V41_AOT', 'CUTEAFD_WIP_VISION_ATTENTION_AOT'])
def test_invalid_switch_values_are_refused(tmp_path, key):
    result = subprocess.run(['bash', str(ROOT / 'scripts/build/build-wip-artifacts.sh'), str(ROOT), 'expert', '121',
                             str(tmp_path / 'b'), str(tmp_path / 'o'), 'rust'],
                            env={'PATH': '/usr/bin:/bin', key: 'yes', 'TMPDIR': str(tmp_path),
                                 'CARGO_HOME': str(tmp_path / 'cargo')},
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 2 and f'{key} must be ON or OFF' in result.stderr


def test_out_of_scope_v41_manifests_are_explicit_and_empty():
    script = (ROOT / 'scripts/build/build-wip-artifacts.sh').read_text()
    assert '"enabled":false,"variants":[]' in script and 'V41_EXPERT_AOT.json' in script
    assert '{"schema":1,"role":"%s","enabled":false}' in script


def test_wip_forwards_both_switches_to_both_legs_and_the_build_key():
    wip = (ROOT / 'wip.sh').read_text()
    coordinator = wip.split('local -a coordinator_build=(docker exec', 1)[1].split('"$coordinator_container"', 1)[0]
    expert = wip.split('build_command="docker exec', 1)[1].split('\n', 1)[0]
    for key in ('V41_AOT', 'VISION_ATTENTION_AOT'):
        assert f'CUTEAFD_WIP_{key}=' in coordinator and f'CUTEAFD_WIP_{key}=' in expert
    assert 'VISION_ATTENTION_AOT V41_AOT CARGO_FEATURES; do' in wip


def native_key(source, tmp_path, *, families='', bf16='', tp_roles='', exl3='ON', nvfp4='ON'):
    """build-wip-artifacts.sh's wip_native_key over SOURCE (environment held fixed)."""
    script = (ROOT / 'scripts/build/build-wip-artifacts.sh').read_text()
    start = script.index('wip_native_key() {')
    function = script[start:script.index('\n}\n', start) + 3]
    clean = {key: value for key, value in os.environ.items() if not key.startswith(('CUTEAFD_WIP_', 'CUTEAFD_CUTE'))}
    # As in a real build, the environment carries keyed scope switches.
    env = {**clean, 'CUTEAFD_WIP_DEV_IMAGE_ID': 'sha256:fixture', 'CUTEAFD_WIP_HOST_ISA': 'x86-64-v4',
           'CUTEAFD_WIP_KIMIK3_AOT': 'ON', 'role': 'coordinator', 'cuda_arch': '120',
           'source_dir': str(source), 'audio_aot': 'OFF', 'exl3_aot': exl3, 'nvfp4_aot': nvfp4,
           'expert_families': families, 'bf16_families': bf16, 'spark_tp_roles': tp_roles,
           'coordinator_aot': 'ON', 'xgrammar': 'ON', 'v41_expert_aot': 'OFF', 'v41_coordinator_aot': 'OFF',
           'v41_nvfp4_aot': 'OFF', 'vision_attention_aot': 'OFF'}
    command = ('set -euo pipefail\nsource "$source_dir/scripts/build/wip-native-store.sh"\n'
               'host_isas=(x86-64-v4)\n' + function + 'wip_native_key\n')
    result = subprocess.run(['bash', '-c', command],
                            env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.fixture
def source_copy(tmp_path):
    source = tmp_path / 'source'
    files = ['scripts/build/kernel-input-rules.json', 'scripts/build/native-key-files.py',
             'scripts/build/wip-native-store.sh',
             'scripts/build/kernel-records.py', 'scripts/build/compare-sm-exports.py',
             'scripts/build/export-placement-inputs.py', 'scripts/build/build-wip-artifacts.sh',
             'python/tools/aot/export_b12x_dsv4_aot.py', 'python/tools/lib/spark_packages.py',
             'native/CMakeLists.txt', 'third_party/sparkinfer.lock.json',
             'rust/crates/cuteafd-core/src/concurrency.rs', 'rust/crates/cuteafd-core/src/dspark_verify.rs',
             'rust/crates/cuteafd-core/src/serving_capacity.rs', 'rust/crates/cuteafd-core/src/lib.rs']
    for name in files:
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        (source / name).write_bytes((ROOT / name).read_bytes())
    (source / 'scripts/tests').mkdir(parents=True)
    (source / 'python/tests').mkdir(parents=True)
    (source / 'python/tests/test_fixture.py').write_text('x = 1\n')
    return source


@pytest.mark.parametrize('name', ['scripts/build/kernel-records.py', 'scripts/build/compare-sm-exports.py',
                                  'scripts/build/export-placement-inputs.py', 'python/tests/test_fixture.py',
                                  'rust/crates/cuteafd-core/src/lib.rs'])
def test_non_input_tooling_leaves_the_native_key_unchanged(source_copy, tmp_path, name):
    before = native_key(source_copy, tmp_path)
    with (source_copy / name).open('a') as handle:
        handle.write('\n# edit\n')
    assert native_key(source_copy, tmp_path) == before


@pytest.mark.parametrize('name', ['python/tools/aot/export_b12x_dsv4_aot.py', 'python/tools/lib/spark_packages.py',
                                  'native/CMakeLists.txt', 'scripts/build/build-wip-artifacts.sh',
                                  'scripts/build/kernel-input-rules.json',
                                  'rust/crates/cuteafd-core/src/concurrency.rs',
                                  'rust/crates/cuteafd-core/src/dspark_verify.rs',
                                  'rust/crates/cuteafd-core/src/serving_capacity.rs'])
def test_exporter_and_build_inputs_change_the_native_key(source_copy, tmp_path, name):
    before = native_key(source_copy, tmp_path)
    with (source_copy / name).open('a') as handle:
        handle.write('\n')
    assert native_key(source_copy, tmp_path) != before


@pytest.mark.parametrize('scope', [dict(families='qwen4:exl3-k4'), dict(families='qwen4:fp8'),
                                  dict(families='qwen4:fp8', bf16='qwen4'), dict(tp_roles='tp6'),
                                  dict(exl3='OFF'), dict(nvfp4='OFF')])
def test_native_key_binds_resolved_export_scope(source_copy, tmp_path, scope):
    # These are resolved shell variables, not CUTEAFD_WIP_* environment values.
    assert native_key(source_copy, tmp_path, **scope) != native_key(source_copy, tmp_path)


def test_native_key_covers_every_rust_source_the_native_build_reads():
    # A Rust file CMake or an exporter reads for shared constants is an export
    # input: missing from the key, editing it would restore stale exports.
    spec = importlib.util.spec_from_file_location('native_key_files', ROOT / 'scripts/build/native-key-files.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    referenced = set()
    pattern = re.compile(r'rust/crates/[A-Za-z0-9_./-]+\.rs')
    for tree in ('native', 'python/tools/aot', 'python/tools/lib'):
        for path in (ROOT / tree).rglob('*'):
            if path.suffix in ('.txt', '.cmake', '.py') and '/tests/' not in path.as_posix():
                referenced.update(pattern.findall(path.read_text()))
    assert referenced, 'expected the native build to read Rust constants'
    assert referenced <= set(module.EXTERNAL_INPUTS), sorted(referenced - set(module.EXTERNAL_INPUTS))


def test_shipped_v4_pro_k2_config_scopes_the_k23_package_the_launcher_loads():
    # The uniform K2 checkpoint decodes with the k23 tier pair (v41_catalog
    # exl3_catalog_or_read(.., &[2, 3]); the planner's deployment names the same).
    values = aot.read_config(ROOT / 'examples/configs/deepseek-v4-pro-exl3-max.config')
    assert 'K2' in values['MODEL_ID']
    family, derived = aot.config_scopes({**values, 'FAMILY': 'deepseek_v4'})
    assert family == 'deepseek_v4'
    assert 'dsv4p:exl3-k23' in derived['CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES'].split(';')
    assert 'dsv4p:exl3-k23' in derived['CUTEAFD_WIP_EXPERT_FAMILIES'].split(';')
    assert derived['CUTEAFD_WIP_DSV4_AOT'] == 'ON' and 'CUTEAFD_WIP_V41_AOT' not in derived


def test_minimax_scope_uses_only_native_w4a4_expert_package():
    derived = scopes('minimax_m3', 'MiniMax-M3.1-Flash', 4)
    assert derived['CUTEAFD_WIP_MINIMAX_AOT'] == 'ON'
    assert derived['CUTEAFD_WIP_EXPERT_FAMILIES'] == 'minimax:nvfp4a4'
    assert derived['CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES'] == 'minimax:nvfp4a4'
    assert 'minimax=4' in derived['CUTEAFD_WIP_GENERIC_SPARK_COUNTS']


def test_inkling2_scope_exports_attention_and_the_coordinator_mxfp8_package():
    for model, experts in (('Inkling-2-Small-MXFP8', 'inkling2:mxfp8'),
                           ('Inkling-2-Small', 'inkling2:mxfp8;inkling2:nvfp4a4')):
        derived = scopes('inkling2', model, 4)
        assert derived['CUTEAFD_WIP_INKLING2_AOT'] == 'ON'
        assert derived['CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES'] == experts
        # Spark MXFP8 packages follow the engine's MXFP8 expert format; none ships yet.
        assert derived['CUTEAFD_WIP_EXPERT_FAMILIES'] == ''
        assert derived['CUTEAFD_WIP_GENERIC_SPARK_COUNTS'] == ''
        assert cmake_flags('coordinator', derived)['CUTEAFD_ENABLE_INKLING2_AOT'] == 'ON'
        assert cmake_flags('expert', derived)['CUTEAFD_ENABLE_INKLING2_AOT'] == 'OFF'
    exl3 = scopes('inkling2', 'Inkling-2-Small-EXL3-K4-MC', 4)
    assert exl3['CUTEAFD_WIP_INKLING2_AOT'] == 'ON'
    assert exl3['CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES'] == ''

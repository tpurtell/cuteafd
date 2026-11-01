"""Site build defaults are configuration values, not serving placement."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('configured,override,expected', [
    ('', None, ''), ('CUTEAFD_EXPORT_HOST=other\n', None, 'other'),
    ('CUTEAFD_EXPORT_HOST=other\n', 'explicit', 'explicit'),
    ('CUTEAFD_EXPORT_HOST=other\n', '', ''),
])
def test_export_host_config_and_environment_precedence(tmp_path, configured, override, expected):
    config = tmp_path / 'site.config'
    config.write_text('MODEL_ID=Qwen/Qwen3.8-Flash-Next\nSPARK_COUNT=0\nRTX_GPUS=1\n' + configured)
    text = (ROOT / 'wip.sh').read_text()
    start = text.index('wip_export_host_env=')
    end = text.index('# Resolve once', start)
    script = f'source {str(ROOT / "scripts/lib/release-common.sh")!r}\nconfig={str(config)!r}\n'
    script += text[start:end] + '\nprintf "host=<%s> seed=<%s>\\n" "$CUTEAFD_EXPORT_HOST" "$WIP_BUILD_SEED"\n'
    env = dict(os.environ)
    env.pop('CUTEAFD_EXPORT_HOST', None)
    if override is not None:
        env['CUTEAFD_EXPORT_HOST'] = override
    result = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=15, env=env)
    assert result.returncode == 0, result.stderr
    assert f'host=<{expected}> seed=<kakapo>' in result.stdout


def test_remote_exports_have_unique_paths_and_cleanup_owned_containers():
    text = (ROOT / 'scripts/build/export-remote.sh').read_text()
    assert 'export/$key-$run_id' in text
    assert 'mktemp -d "$HOME/.cache/cuteafd/builds/export-$key.XXXXXXXX"' in text
    assert 'docker rm -f "$container"' in text
    assert 'export.build.$slot.lock' in text
    assert 'flock -n "$slot_fd"' in text


def test_seed_admission_after_reuse_and_release_before_gpu_queue():
    text = (ROOT / 'wip.sh').read_text()
    flow = text.split('wip_build_selected_roles() {', 1)[1].split('\n}\n', 1)[0]
    assert flow.index('build_coordinator rust') < flow.index('wip_build_resource_acquire')
    assert flow.index('build_expert') < flow.index('wip_build_resource_release')
    assert flow.index('wip_build_resource_release') < flow.index('wip_distribute_sealed_expert') < flow.index('build_coordinator native')
    assert text.index('wip_build_selected_roles\n') > text.index('wip_reused_expert=1')


def test_every_fingerprinted_build_input_reaches_both_build_containers():
    # A key in the slot fingerprint that never reaches the build would seal a
    # slot whose name claims inputs it was not built with (EXL3_BITS did).
    text = (ROOT / 'wip.sh').read_text()
    keys = text.split('for key in EXL3_AOT', 1)[1].split('; do', 1)[0].replace('\\', ' ').split()
    keys = ['EXL3_AOT', *keys]
    coordinator = text.split('local -a coordinator_build=(docker exec', 1)[1].split('"$coordinator_container"', 1)[0]
    spark = text.split('build_command="docker exec', 1)[1].split('\n', 1)[0]
    consumed = (ROOT / 'scripts/build/build-wip-artifacts.sh').read_text()
    for key in keys:
        name = 'CUTEAFD_WIP_' + key
        if name not in consumed:
            continue
        assert name in coordinator, f'{name} is fingerprinted but not passed to the coordinator build'
    assert 'CUTEAFD_WIP_EXL3_BITS' not in spark


def test_dense_qwen_aot_opt_in_reaches_native_and_fingerprints():
    wip = (ROOT / 'wip.sh').read_text()
    keys = wip.split('for key in EXL3_AOT', 1)[1].split('; do', 1)[0].split()
    assert 'QWEN_DENSE_AOT' in keys
    cmake = (ROOT / 'native/CMakeLists.txt').read_text()
    assert 'option(CUTEAFD_ENABLE_QWEN_DENSE_AOT ' in cmake
    assert cmake.split('option(CUTEAFD_ENABLE_QWEN_DENSE_AOT ', 1)[1].splitlines()[0].endswith(' OFF)')
    assert cmake.count('OR CUTEAFD_ENABLE_QWEN_DENSE_AOT') == 3
    programs = (ROOT / 'native/cmake/shared/dsv4_programs.cmake').read_text()
    dense = programs.split('if(CUTEAFD_ENABLE_QWEN_DENSE_AOT)', 1)[1].split('set(CUTEAFD_DSV4_DIR', 1)[0]
    assert '"qwen35,qwen38"' in dense
    assert '"${CUTEAFD_PROGRAM_GEOMETRY},qwen35,qwen38"' in dense
    for mode, launcher in [('wip', wip), ('release', (ROOT / 'build.sh').read_text())]:
        variable = f'CUTEAFD_{mode.upper()}_QWEN_DENSE_AOT'
        assert f'-e "{variable}=${{{variable}:-OFF}}"' in launcher
        build = (ROOT / f'scripts/build/build-{mode}-artifacts.sh').read_text()
        assert f'-DCUTEAFD_ENABLE_QWEN_DENSE_AOT="$( [[ "$role" == coordinator ]] && echo "${{{variable}:-OFF}}" || echo OFF)"' in build


def test_exl3_release_exports_named_families_and_compact_companions():
    cmake = (ROOT / 'native/cmake/shared/exl3.cmake').read_text()
    release = (ROOT / 'scripts/build/build-release-artifacts.sh').read_text()
    assert 'CUTEAFD_V41_EXL3_BIT_FAMILIES' not in cmake
    assert 'CUTEAFD_RELEASE_EXL3_BIT_FAMILIES' not in release
    assert 'release_family_list' in release
    assert '--compact-companions' in cmake


def test_spark_exec_forwards_context_admission_inputs():
    wip = (ROOT / 'wip.sh').read_text()
    remote = wip.split('ensure_remote_container() {', 1)[1].split('\nREMOTE\n}', 1)[0]
    assert 'run -d --init --name "$container"' in remote
    assert '--user "$(id -u):$(id -g)"' in remote
    spark = wip.split('build_command="docker exec', 1)[1].split('\n', 1)[0]
    assert 'CUTEAFD_EXPORT_HOST_CONTEXTS=' in spark
    assert 'CUTEAFD_EXPORT_WORKER_RAM_MIB=' in spark
    cards = (ROOT / 'scripts/bench/wip-cards.py').read_text()
    scopes = cards.split('def build_scopes(cards):', 1)[1].split('def package_dir', 1)[0]
    assert 'CMAKE_BUILD_PARALLEL_LEVEL' not in scopes


@pytest.mark.parametrize('mode', ['release', 'wip'])
@pytest.mark.parametrize('features', [None, '', 'jemalloc'])
def test_coordinator_allocator_features_reach_cargo(mode, features):
    """Execute launcher argv and the artifact script's feature selection, not grep.

    Docker is replaced by a shell function that imports its actual -e arguments
    into a clean container environment before running the real selection block.
    No hardware, Docker daemon or compiler is contacted.
    """
    launcher = (ROOT / ('build.sh' if mode == 'release' else 'wip.sh')).read_text()
    artifact = (ROOT / f'scripts/build/build-{mode}-artifacts.sh').read_text()
    if mode == 'release':
        start = launcher.index('timeout --foreground "$export_timeout" docker run')
        end = launcher.index(' ||\n  coordinator_export_status=$?', start)
        launch = launcher[start:end]
        start = artifact.index('cargo_feature_args=()')
        selection = artifact[start:artifact.index('\nhost_isa_jobs=()', start)]
        setup = 'export_timeout=30\ntimeout() { shift 2; "$@"; }\n'
        variable = 'CUTEAFD_RELEASE_COORDINATOR_FEATURES'
    else:
        start = launcher.index('wip_cargo_features="')
        validation = launcher[start:launcher.index('\nwip_build_key=', start)]
        start = launcher.index('  local -a coordinator_build=(docker exec')
        end = launcher.index('\n  # Cargo first', start)
        launch = launcher[start:end].replace('local -a', 'declare -a')
        launch += '\n"${coordinator_build[@]}" rust\n'
        start = artifact.index('cargo_features="')
        selection = artifact[start:artifact.index('\nwip_rust_fingerprint=', start)]
        setup = validation + '\n'
        variable = 'CUTEAFD_WIP_CARGO_FEATURES'
    script = '''set -eo pipefail
release_die() { echo "$*" >&2; exit 2; }
docker() (
    # No host environment inheritance: the feature must cross the -e boundary.
    unset CUTEAFD_RELEASE_COORDINATOR_FEATURES CUTEAFD_WIP_CARGO_FEATURES
    while (($#)); do
        if [[ "$1" == -e ]]; then export "$2"; shift 2; else shift; fi
    done
    role=coordinator
''' + selection + '''
    printf '<%s>\\n' "${cargo_feature_args[@]}"
)
''' + setup + launch
    env = {k: v for k, v in os.environ.items()
           if k not in ('CUTEAFD_RELEASE_COORDINATOR_FEATURES', 'CUTEAFD_WIP_CARGO_FEATURES')}
    if features is not None:
        env[variable] = features
    result = subprocess.run(['bash', '-c', script], env=env, capture_output=True,
                            text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == (['<--features>', '<jemalloc>'] if features else ['<>'])


@pytest.mark.parametrize('features,expected', [('jemalloc,dev-probe', 0), ('unknown', 2)])
def test_wip_allocator_features_keep_the_validation_allowlist(features, expected):
    text = (ROOT / 'wip.sh').read_text()
    start = text.index('wip_cargo_features="')
    block = text[start:text.index('\nwip_build_key=', start)]
    result = subprocess.run(['bash', '-c', 'release_die() { echo "$*"; exit 2; }\n' + block],
                            env={**os.environ, 'CUTEAFD_WIP_CARGO_FEATURES': features},
                            text=True, capture_output=True, timeout=15)
    assert result.returncode == expected, result.stdout + result.stderr

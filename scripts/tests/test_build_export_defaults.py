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
    assert 'CUTEAFD_WIP_EXL3_BITS' in spark


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


def test_v41_release_exports_all_decoder_families_and_compact_companions():
    cmake = (ROOT / 'native/cmake/shared/exl3.cmake').read_text()
    release = (ROOT / 'scripts/build/build-release-artifacts.sh').read_text()
    assert 'set(CUTEAFD_V41_EXL3_BIT_FAMILIES "2,3;3,4;4,5" CACHE STRING' in cmake
    assert '${CUTEAFD_RELEASE_EXL3_BIT_FAMILIES:-2,3;3,4;4,5}' in release
    assert '--compact-companions' in cmake

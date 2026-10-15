"""Site build defaults are configuration values, not serving placement."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('configured,override,expected', [
    ('', None, 'oliver'), ('CUTEAFD_EXPORT_HOST=other\n', None, 'other'),
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


def test_seed_admission_after_reuse_and_no_gpu_queue():
    text = (ROOT / 'wip.sh').read_text()
    assert text.index('((wip_build_expert == 0)) || wip_build_resource_acquire') > text.index('wip_reused_expert=1')
    capped = text[text.index('wip_capped_export() {'):text.index('wip_capped_export_on() (')]
    assert capped.index('wip_build_resource_dir/granted') < capped.index('export-placement.py" hold')
    dispatch = text[text.index('    local mode exported=0'):text.index('    ((exported)) || release_die')]
    assert 'wip_hwlock' not in dispatch and 'wait-remote)' in dispatch
    assert '[[ "$seed_host" != kakapo ]]' not in capped
    assert text.index('distribute_expert_slot\nwip_resource_complete=1') < text.rindex('wip_build_resource_release')


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

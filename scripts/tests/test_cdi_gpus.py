import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('cdi_gpus', ROOT / 'scripts/lib/cdi_gpus.py')
cdi = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cdi)
SH = ROOT / 'scripts/lib/cdi-gpus.sh'
YAML = 'devices:\n    - name: "0"\n    - name: "1"\n    - name: GPU-abc-123\n    - name: all\n'


@pytest.fixture(autouse=True)
def real_check(monkeypatch):
    monkeypatch.setenv('CUTEAFD_CDI_CHECK', 'on')


@pytest.fixture
def spec(tmp_path):
    path = tmp_path / 'nvidia.yaml'
    path.write_text(YAML)
    return path


def test_python_args(spec):
    assert cdi.cdi_gpu_args('all', [spec]) == ['--device', 'nvidia.com/gpu=all']
    assert cdi.cdi_gpu_args([1, 0], [spec]) == ['--device', 'nvidia.com/gpu=1', '--device', 'nvidia.com/gpu=0']
    assert cdi.cdi_gpu_args('GPU-abc-123', [spec]) == ['--device', 'nvidia.com/gpu=GPU-abc-123']


def test_python_missing_name_or_spec_fails(spec, tmp_path):
    with pytest.raises(RuntimeError, match='nvidia-ctk cdi generate'):
        cdi.cdi_gpu_args('2', [spec])
    with pytest.raises(RuntimeError, match='nvidia-ctk cdi generate'):
        cdi.cdi_gpu_args('0', [tmp_path / 'absent.yaml'])
    with pytest.raises(ValueError):
        cdi.cdi_gpu_args('', [spec])


def run_sh(spec, request):
    script = f'source "{SH}"; cuteafd_cdi_gpu_args "{request}" && printf "%s\\n" "${{CUTEAFD_CDI_ARGS[@]}}"'
    return subprocess.run(['bash', '-c', script], text=True, capture_output=True, env={**os.environ, 'CUTEAFD_CDI_SPECS': str(spec)})


def test_shell_args_and_failure(spec):
    ok = run_sh(spec, '0,GPU-abc-123')
    assert ok.stdout.split() == ['--device', 'nvidia.com/gpu=0', '--device', 'nvidia.com/gpu=GPU-abc-123']
    bad = run_sh(spec, '7')
    assert bad.returncode != 0 and 'nvidia-ctk cdi generate' in bad.stderr


WSL = 'devices:\n    - name: all\n'


def test_wsl_all_only_spec_maps_single_gpu_host(tmp_path, monkeypatch):
    spec = tmp_path / 'nvidia.yaml'
    spec.write_text(WSL)
    monkeypatch.setenv('CUTEAFD_CDI_GPU_COUNT', '1')
    assert cdi.cdi_gpu_args('GPU-abc-123', [spec]) == ['--device', 'nvidia.com/gpu=all']
    assert run_sh(spec, 'GPU-abc-123').stdout.split() == ['--device', 'nvidia.com/gpu=all']
    monkeypatch.setenv('CUTEAFD_CDI_GPU_COUNT', '2')
    with pytest.raises(RuntimeError, match='nvidia-ctk cdi generate'):
        cdi.cdi_gpu_args('0', [spec])
    assert run_sh(spec, '0').returncode != 0
    monkeypatch.setenv('CUTEAFD_CDI_GPU_COUNT', '1')
    with pytest.raises(RuntimeError):
        cdi.cdi_gpu_args('0,1', [spec])


def test_per_gpu_spec_keeps_exact_names(spec, monkeypatch):
    monkeypatch.setenv('CUTEAFD_CDI_GPU_COUNT', '1')
    assert cdi.cdi_gpu_args('0', [spec]) == ['--device', 'nvidia.com/gpu=0']
    assert run_sh(spec, '0').stdout.split() == ['--device', 'nvidia.com/gpu=0']


def test_v41_launch_uses_cdi_for_coordinator_and_exports_helpers_to_workers():
    text = (ROOT / 'scripts/launch/run-family.sh').read_text()
    v41 = text.split('launch_deepseek_v41() {', 1)[1].split('# The family:', 1)[0]
    assert 'release_coordinator_device_args "${release_gpu_uuids[@]}"' in v41
    assert 'cuteafd_cdi_gpu_args all || exit 2' in v41
    assert '--name "$coordinator" --restart no "${RELEASE_COORDINATOR_DEVICE_ARGS[@]}"' in v41
    helper = text.split('release_coordinator_device_args() {', 1)[1].split('\n}', 1)[0]
    assert 'cuteafd_cdi_gpu_args "$csv" || exit 2' in helper
    assert '--name "$name" --restart no "${CUTEAFD_CDI_ARGS[@]}"' in v41
    assert 'export -f release_die release_hub_mounts release_serving_docker_args release_require_writable_state cuteafd_cdi_require cuteafd_cdi_gpu_args;' in v41


def test_no_legacy_gpus_flag_in_launchers():
    for name in ('build.sh', 'run.sh', 'wip.sh', 'scripts/launch/run-family.sh', 'scripts/build/cuteafd-dev.sh',
                 'scripts/bench/wip-cards.py', 'scripts/bench/golden-run.py', 'scripts/bench/ab-spark-intake.sh',
                 'scripts/bench/glm5/ab-glm-weighted.sh', 'scripts/build/export-remote.sh', 'scripts/build/verify-nvfp4-modelopt-container.sh',
                 'scripts/build/configure-docker-nvidia-runtime.sh', 'quantization/deploy_workers.py',
                 'quantization/launch_quantization.py'):
        text = (ROOT / name).read_text()
        assert not any((' --gpus ' in line and 'hold --gpus' not in line) or "'--gpus'" in line and 'add_argument' not in line
                       for line in text.splitlines() if not line.lstrip().startswith('#')), name

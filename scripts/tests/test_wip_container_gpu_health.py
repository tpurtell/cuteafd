"""wip.sh recreates a reused coordinator container whose GPU stopped answering
NVML (a host daemon-reload drops it in running containers) instead of failing
later in the export."""
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WIP = (ROOT / 'wip.sh').read_text()


def test_build_pins_images_across_shared_default_retag(tmp_path):
    start = WIP.index('wip_pin_dev_images() {')
    end = WIP.index('# stream_between_hosts SRC', start)
    old, new, remote = ('sha256:' + char * 64 for char in ('a', 'b', 'c'))
    script = f'''set -euo pipefail
release_die() {{ echo "$*" >&2; exit 9; }}
default={old}
docker() {{ printf '%s\\n' "$default"; }}
ssh() {{
  case "$*" in
    *"docker tag"*) return 0 ;;
    *"docker image inspect"*)
      if [[ "$*" == *"pinned-"* || "$*" == *"{old}"* ]]; then echo {remote}; else echo "$default"; fi ;;
  esac
}}
spark_image_layers() {{ echo '["layer-a"]'; }}
COORDINATOR_DOCKER_DEV=coordinator-default
SPARK_EXPERT_DOCKER_DEV=spark-default
seed_host=seed wip_spark_leg=1
wip_target_hosts=(receiver)
{WIP[start:end]}
wip_pin_dev_images
default={new}
wip_pin_spark_host seed
wip_pin_spark_host receiver
[[ "$COORDINATOR_DOCKER_DEV" == {old} ]]
[[ "$SPARK_EXPERT_DOCKER_DEV" == cuteafd-spark-expert-dev:pinned-a* ]]
[[ "${{wip_spark_image_ids[seed]}}" == {remote} ]]
[[ "${{wip_spark_image_ids[receiver]}}" == {remote} ]]
spark_image_layers() {{ echo '["wrong-layer"]'; }}
wip_pin_spark_host replacement
'''
    result = subprocess.run(['bash', '-c', script], capture_output=True, text=True)
    assert result.returncode == 9, result.stderr
    assert 'development image changed on replacement during this build' in result.stderr
    assert WIP.index('wip_pin_dev_images\npreflight_existing_container_images') < WIP.index('# Each role\'s key')
    assert '"$spark_container" "${wip_spark_image_ids[$host]}"' in WIP
    assert 'image_id="${wip_spark_image_ids[$seed_host]}"' in WIP
    assert 'docker image rm "$wip_transport_image"' in WIP


def _functions():
    start = WIP.index('# wip_container_gpu_ok CONTAINER')
    end = WIP.index('\n}\n', WIP.index('ensure_local_container() {')) + 3
    return WIP[start:end]


@pytest.mark.parametrize('gpu_ok', [True, False])
def test_reused_container_is_recreated_only_when_nvml_fails(tmp_path, gpu_ok):
    trace = tmp_path / 'trace'
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    # docker: the container exists, matches image/user/labels/mount, is running;
    # `exec ... nvidia-smi -L` succeeds or fails per the case.
    (bindir / 'docker').write_text(f'''#!/usr/bin/env bash
echo "docker $*" >> {trace}
case "$1" in
  image) echo img ;;
  container) exit 0 ;;
  inspect)
    case "$3" in
      '{{{{.Config.User}}}}') echo "$(id -u):$(id -g)" ;;
      *build-caches*) echo "on:hash" ;;
      '{{{{.Image}}}}') echo img ;;
      *Mounts*) echo {tmp_path} ;;
      '{{{{.State.Running}}}}') echo true ;;
    esac ;;
  exec) [[ "$3" == nvidia-smi ]] && exit {0 if gpu_ok else 1}; exit 0 ;;
  run) echo new-container ;;
  rm) exit 0 ;;
esac
''')
    (bindir / 'docker').chmod(0o755)
    script = 'CUTEAFD_CDI_CHECK=off\n' + (ROOT / 'scripts/lib/cdi-gpus.sh').read_text() + _functions() + f'''
release_die() {{ echo "DIE $*" >&2; exit 9; }}
cuteafd_build_cache_docker_args() {{ :; }}
COORDINATOR_DOCKER_DEV=img coordinator_container=c wip_toolchain_hash=hash wip_mount_root={tmp_path}
RELEASE_COORDINATOR_GPU_UUID=GPU-x hf_home={tmp_path} CUTEAFD_BUILD_CACHES=on
ensure_local_container
'''
    result = subprocess.run(['bash', '-c', 'set -euo pipefail\n' + script], capture_output=True, text=True,
                            timeout=60, env={**os.environ, 'PATH': f'{bindir}:{os.environ["PATH"]}'})
    assert result.returncode == 0, result.stderr
    calls = trace.read_text()
    assert 'docker exec c nvidia-smi -L' in calls
    if gpu_ok:
        assert 'docker rm' not in calls and 'docker run' not in calls
    else:
        assert 'lost GPU access' in result.stderr
        assert calls.index('docker rm -f c') < calls.index('docker run -d --name c')


@pytest.mark.parametrize('recorded,bound', [('GPU-x', True), ('nvidia.com/gpu=GPU-x', True),
                                            ('nvidia.com/gpu=GPU-y', False), ('GPU-y', False)])
def test_gpu_binding_preflight_accepts_cdi_and_legacy_requests(tmp_path, recorded, bound):
    start = WIP.index('preflight_existing_container_images() {')
    end = WIP.index('\n}\n', start) + 3
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    (bindir / 'docker').write_text(f'''#!/usr/bin/env bash
case "$*" in
  *"image inspect"*) echo img ;;
  *"container inspect"*) exit 0 ;;
  *DeviceRequests*) echo "{recorded}" ;;
  *) echo img ;;
esac
''')
    (bindir / 'docker').chmod(0o755)
    script = (f'release_die() {{ echo "DIE $*" >&2; exit 9; }}\n'
              'COORDINATOR_DOCKER_DEV=img coordinator_container=c RELEASE_COORDINATOR_GPU_UUID=GPU-x wip_spark_leg=0\n'
              + WIP[start:end] + 'preflight_existing_container_images\n')
    result = subprocess.run(['bash', '-c', script], capture_output=True, text=True,
                            env={**os.environ, 'PATH': f'{bindir}:{os.environ["PATH"]}'})
    assert (result.returncode == 0) == bound, result.stderr
    if not bound:
        assert 'not bound to the configured physical GPU' in result.stderr

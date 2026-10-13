"""wip.sh recreates a reused coordinator container whose GPU stopped answering
NVML (a host daemon-reload drops it in running containers) instead of failing
later in the export."""
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WIP = (ROOT / 'wip.sh').read_text()


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
    script = _functions() + f'''
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

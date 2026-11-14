"""Python install caches are builder-only, architecture-scoped storage."""
import importlib.util
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('name', ['dev', 'release'])
def test_python_installs_use_cache_mounts(name):
    text = (ROOT / f'docker/Dockerfile.{name}').read_text()
    assert text.startswith('# syntax=docker/dockerfile:1.7\n')
    assert re.search(r'^FROM .*\nARG TARGETARCH\n', text, re.M)
    assert '--no-cache' not in text
    installs = 0
    for instruction in re.split(r'\n(?=[A-Z]+\s)', text):
        if re.search(r'\buv pip install\b', instruction):
            installs += 1
            assert instruction.startswith('RUN ')
            assert '--mount=type=cache,target=/root/.cache/uv,id=cuteafd-uv-${TARGETARCH},sharing=locked' in instruction
        if re.search(r'\bpython3 -m pip install\b', instruction):
            assert '--mount=type=cache,target=/root/.cache/pip,id=cuteafd-pip-${TARGETARCH},sharing=locked' in instruction
    assert installs == (1 if name == 'dev' else 4)


@pytest.mark.parametrize('name', ['build.sh', 'scripts/build/build-dev-images.sh'])
def test_local_and_remote_builds_enable_buildkit(name):
    text = (ROOT / name).read_text()
    commands = re.findall(r'^\s*(.*\bdocker build\s.*)$', text, re.M)
    assert len(commands) == 2
    assert all(command.strip().startswith('DOCKER_BUILDKIT=1 docker build ') for command in commands)


def test_toolchain_hash_intentionally_covers_cache_syntax(tmp_path):
    spec = importlib.util.spec_from_file_location('cache_toolchain', ROOT / 'scripts/build/dev-toolchain.py')
    toolchain = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(toolchain)
    for name in ('docker/Dockerfile.dev', 'docker/entrypoint.sh',
                 'scripts/build/install-dev-cache-tools.sh', 'scripts/build/install-nsys.sh',
                 'scripts/build/dev-toolchain.py'):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    dockerfile = tmp_path / 'docker/Dockerfile.dev'
    mounted = dockerfile.read_text()
    mounted_hash = toolchain.identity(tmp_path)
    dockerfile.write_text(mounted.replace('# syntax=docker/dockerfile:1.7\n', ''))
    assert toolchain.identity(tmp_path) != mounted_hash
    dockerfile.write_text(re.sub(r'--mount=type=cache,\S+\s*\\\n\s*', '', mounted))
    assert toolchain.identity(tmp_path) != mounted_hash
    dockerfile.write_text(mounted.replace('uv==0.12.23', 'uv==0.12.24'))
    assert toolchain.identity(tmp_path) != mounted_hash

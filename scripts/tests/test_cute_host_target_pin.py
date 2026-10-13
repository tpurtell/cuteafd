"""#127: AOT exports pin CuTe DSL's host target to a portable baseline instead
of the build host's CPU (python/tools/lib/_pinned_sparkinfer.py)."""
import ast
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / 'python/tools/lib/_pinned_sparkinfer.py').read_text()


def _pin_module():
    """The host-target part of _pinned_sparkinfer, without verifying SparkInfer."""
    tree = ast.parse(SOURCE)
    keep = [node for node in tree.body if (
        isinstance(node, ast.Assign) and any(getattr(t, 'id', '') == 'HOST_TARGETS' for t in node.targets)
    ) or (isinstance(node, ast.FunctionDef) and node.name in ('host_target', '_pin_cute_host_target'))]
    module = types.ModuleType('pin')
    module.os = __import__('os')
    exec(compile(ast.Module(body=keep, type_ignores=[]), 'pin', 'exec'), module.__dict__)
    return module


class FakeHostTarget:
    """The CuTe DSL HostTarget surface the pin touches (parsed triple/cpu/features)."""
    def __init__(self, val=''):
        self._triple, self._cpu, self._features = _parse(val)
        self.value = val

    @property
    def triple(self):
        return self._triple

    @property
    def cpu(self):
        return self._cpu

    @property
    def features(self):
        return self._features


def _parse(spec):
    if not spec:
        return '', '', ''
    fields = dict(token.split('=', 1) for token in spec.split()[1:])
    return fields.get('-mtriple', ''), fields.get('-mcpu', ''), fields.get('-mattr', '')


@pytest.fixture
def fake_cutlass(monkeypatch):
    compiler = types.SimpleNamespace(HostTarget=type('HostTarget', (FakeHostTarget,), {}), _parse_host_target=_parse)
    base = types.ModuleType('cutlass.base_dsl')
    base.compiler = compiler
    monkeypatch.setitem(sys.modules, 'cutlass', types.ModuleType('cutlass'))
    monkeypatch.setitem(sys.modules, 'cutlass.base_dsl', base)
    return compiler


@pytest.mark.parametrize('machine,cpu', [('x86_64', 'x86-64-v3'), ('aarch64', 'cortex-x925')])
def test_default_is_a_portable_baseline_per_architecture(monkeypatch, machine, cpu):
    monkeypatch.delenv('CUTEAFD_CUTE_HOST_TARGET', raising=False)
    monkeypatch.setattr('platform.machine', lambda: machine)
    assert f'-mcpu={cpu}' in _pin_module().host_target()


def test_override_and_native_escape_hatch(monkeypatch):
    pin = _pin_module()
    monkeypatch.setenv('CUTEAFD_CUTE_HOST_TARGET', 'llvm -mtriple=x86_64-unknown-linux-gnu -mcpu=znver3')
    assert pin.host_target().endswith('-mcpu=znver3')
    monkeypatch.setenv('CUTEAFD_CUTE_HOST_TARGET', 'native')
    assert pin.host_target() == ''


def test_pin_fills_only_empty_targets_including_ones_built_before_it(monkeypatch, fake_cutlass):
    monkeypatch.delenv('CUTEAFD_CUTE_HOST_TARGET', raising=False)
    monkeypatch.setattr('platform.machine', lambda: 'x86_64')
    HostTarget = fake_cutlass.HostTarget
    early = HostTarget('')  # e.g. the DSL singleton's options, built at `import cutlass`
    explicit = HostTarget('llvm -mtriple=aarch64-unknown-linux-gnu -mcpu=neoverse-n1')
    _pin_module()._pin_cute_host_target()
    assert (early.triple, early.cpu) == ('x86_64-unknown-linux-gnu', 'x86-64-v3')
    later = HostTarget('')
    later.value = ''  # the string parser assigns "" for an absent --host-target
    assert later.cpu == 'x86-64-v3'
    assert (explicit.triple, explicit.cpu) == ('aarch64-unknown-linux-gnu', 'neoverse-n1')


def test_native_override_leaves_cute_untouched(monkeypatch, fake_cutlass):
    monkeypatch.setenv('CUTEAFD_CUTE_HOST_TARGET', 'native')
    _pin_module()._pin_cute_host_target()
    assert fake_cutlass.HostTarget('').cpu == ''


def test_every_exporter_imports_the_pin_and_the_native_key_carries_it():
    for path in sorted((ROOT / 'python/tools/aot').glob('*.py')):
        text = path.read_text()
        if 'cute.compile' in text or 'b12x' in text:
            assert '_pinned_sparkinfer' in text, path.name
    # The pin runs at import, as a module-level statement every exporter executes.
    assert '\n_pin_cute_host_target()\nHOST_TARGET = host_target()\n' in SOURCE
    build = (ROOT / 'scripts/build/build-wip-artifacts.sh').read_text()
    key = build[build.index('wip_native_key() {'):build.index('\n}\n', build.index('wip_native_key() {'))]
    assert 'CUTEAFD_CUTE_HOST_TARGET' in key and '/python/' not in key and 'find native python' in key


def test_manifests_record_the_compiled_target_not_the_export_hosts_gpu_name():
    pin = _pin_module()
    tree = ast.parse(SOURCE)
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'export_target')
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'pin', 'exec'), pin.__dict__)
    props = types.SimpleNamespace(major=12, minor=0, name='NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition')
    assert pin.export_target(props) == 'sm120'
    for name in ('export_b12x_v41_experts_aot.py', 'export_b12x_v41_nvfp4_aot.py', 'export_b12x_exl3_aot.py',
                 'export_b12x_v41_fp8_aot.py'):
        text = (ROOT / 'python/tools/aot' / name).read_text()
        assert 'props.name' not in text and 'properties.name' not in text, name
        assert '_pinned_sparkinfer.export_target(' in text, name


def test_cuda_objects_are_seeded_per_translation_unit():
    # nvcc embedded tmpxft_<pid> names and random per-TU IDs: two builds of the
    # same source never matched. Every .cu compile gets -objtemp and its own seed.
    cmake = (ROOT / 'native/CMakeLists.txt').read_text()
    block = cmake[cmake.index('function(cuteafd_reproducible_cuda_objects)'):]
    assert '"-objtemp" "--frandom-seed=${target}/${relative}"' in block
    assert 'file(RELATIVE_PATH relative "${CMAKE_CURRENT_SOURCE_DIR}"' in block
    assert 'cmake_language(DEFER CALL cuteafd_reproducible_cuda_objects)' in cmake


def test_export_manifests_never_iterate_directories_unsorted():
    # Directory order is filesystem-specific: an unsorted iterdir/glob into a
    # manifest dict made raptor's and oliver's identical exports differ (#127).
    import re
    for path in sorted((ROOT / 'python/tools/aot').glob('export_*.py')):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r'for \w+ in [\w.]+\.(iterdir|glob|rglob)\(', line):
                assert 'sorted(' in line, f'{path.name}:{number}: {line.strip()}'

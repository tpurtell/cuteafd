"""UMA device admission uses MemAvailable; cgroup limits constrain host bytes."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('allocatable_memory', ROOT / 'python/tools/lib/allocatable_memory.py')
memory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory)


@pytest.fixture
def host_paths(tmp_path):
    meminfo, membership, root = tmp_path / 'meminfo', tmp_path / 'cgroup', tmp_path / 'groups'
    meminfo.write_text('MemFree: 1 kB\nMemAvailable: 8192 kB\nCached: 2048 kB\n')
    membership.write_text('0::/\n')
    root.mkdir()
    return dict(meminfo=meminfo, membership=membership, cgroup_root=root)


def device(integrated, total=16 * 1024**2):
    def raw(*args):
        assert not integrated, 'integrated admission must never query CUDA free'
        return 1234, total
    return SimpleNamespace(current_device=lambda: 0, get_device_properties=lambda _: SimpleNamespace(
        is_integrated=integrated, total_memory=total), mem_get_info=raw)


def group(root, relative, limit, current, *, v1=False):
    path = root / relative
    path.mkdir(parents=True, exist_ok=True)
    (path / ('memory.limit_in_bytes' if v1 else 'memory.max')).write_text(str(limit))
    (path / ('memory.usage_in_bytes' if v1 else 'memory.current')).write_text(str(current))


def test_integrated_reclaims_cache_without_double_counting(host_paths):
    assert memory.cuda_allocatable_memory_info(device(True), **host_paths) == (
        8 * 1024**2, 16 * 1024**2, 'MemAvailable')


def test_integrated_headroom_bounded_by_device_total(host_paths):
    assert memory.cuda_allocatable_memory_info(device(True, total=1024), **host_paths)[0] == 1024


def test_discrete_never_reads_host_files(tmp_path):
    assert memory.cuda_allocatable_memory_info(device(False), meminfo=tmp_path / 'missing') == (
        1234, 16 * 1024**2, 'cudaMemGetInfo')


@pytest.mark.parametrize('v1', [False, True])
@pytest.mark.parametrize('limiting', ['parent/child', 'parent', ''])
def test_every_visible_ancestor_bounds_admission(host_paths, v1, limiting):
    host_paths['membership'].write_text('7:cpu,memory:/parent/child\n' if v1 else '0::/parent/child\n')
    root = host_paths['cgroup_root'] / 'memory' if v1 else host_paths['cgroup_root']
    for relative in ['parent/child', 'parent', '']:
        group(root, relative, 3000 if relative == limiting else 'max', 1000, v1=v1)
    assert memory.host_allocatable_memory(**host_paths) == 2000
    assert memory.cuda_allocatable_memory_info(device(True), **host_paths)[0] == 8 * 1024**2


@pytest.mark.parametrize('relative', ['/invisible/child', '/../../outside'])
def test_namespaced_or_unsafe_membership_still_checks_mount_root(host_paths, relative):
    host_paths['membership'].write_text(f'0::{relative}\n')
    group(host_paths['cgroup_root'], '', 1000, 300)
    assert memory.host_allocatable_memory(**host_paths) == 700


@pytest.mark.parametrize('limit,current,expected', [('max', 'invalid', 8 * 1024**2), (10, 20, 0),
                                                   (2**63 - 4096, 100, 8 * 1024**2)])
def test_unlimited_and_exhausted_cgroup(host_paths, limit, current, expected):
    group(host_paths['cgroup_root'], '', limit, current)
    assert memory.host_allocatable_memory(**host_paths) == expected


@pytest.mark.parametrize('text', ['', 'MemFree: 123 kB\n', 'MemAvailable: invalid kB\n',
                                  'MemAvailable: -1 kB\n', 'MemAvailable: 42 MB\n',
                                  'MemAvailable: 42 kB extra\n',
                                  'MemAvailable: 1 kB\nMemAvailable: 2 kB\n',
                                  f'MemAvailable: {2**100} kB\n'])
def test_malformed_meminfo_fails_closed_without_cuda_query(host_paths, text):
    host_paths['meminfo'].write_text(text)
    with pytest.raises(ValueError, match='MemAvailable'):
        memory.cuda_allocatable_memory_info(device(True), **host_paths)


@pytest.mark.parametrize('limit,current', [('oops', '10'), ('100', '-1'), ('100', 'oops')])
def test_malformed_cgroup_fails_closed(host_paths, limit, current):
    group(host_paths['cgroup_root'], '', limit, current)
    with pytest.raises(ValueError, match='cgroup'):
        memory.host_allocatable_memory(**host_paths)


def test_missing_usage_under_finite_limit_fails_closed(host_paths):
    (host_paths['cgroup_root'] / 'memory.max').write_text('100')
    with pytest.raises(FileNotFoundError):
        memory.host_allocatable_memory(**host_paths)


def test_release_probe_uses_same_helper_and_reports_measure():
    text = (ROOT / 'build.sh').read_text()
    probe = text[text.index('    spark_memory_probe='):text.index('    spark_wait_reason="allocatable memory')]
    assert 'python/tools/lib/allocatable_memory.py:/allocatable_memory.py:ro' in probe
    assert '"$dev_image" /allocatable_memory.py' in probe
    assert 'mem_get_info' not in probe
    assert '($spark_memory_measure)' in probe
    assert 'spark_allocatable_free >= spark_export_min_free_bytes' in probe


@pytest.mark.parametrize('integrated', [False, True])
def test_torch_without_integrated_property_queries_runtime_attribute(monkeypatch, integrated):
    import ctypes
    queried = []

    def attribute(output, name, device):
        queried.append((name, device))
        output._obj.value = int(integrated)
        return 0

    runtime = SimpleNamespace(cudaDeviceGetAttribute=attribute)
    monkeypatch.setattr(memory.ctypes, 'CDLL', lambda _: runtime)
    cuda = SimpleNamespace(get_device_properties=lambda _: SimpleNamespace(total_memory=1024))
    assert memory.device_is_integrated(cuda, 3) is integrated
    assert queried == [(18, 3)]
    assert attribute.argtypes == [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]


def test_runtime_attribute_failure_does_not_fall_back_to_raw_cuda(monkeypatch):
    runtime = SimpleNamespace(cudaDeviceGetAttribute=lambda *args: 1)
    monkeypatch.setattr(memory.ctypes, 'CDLL', lambda _: runtime)
    cuda = SimpleNamespace(get_device_properties=lambda _: SimpleNamespace(total_memory=1024))
    with pytest.raises(RuntimeError, match='integrated'):
        memory.cuda_allocatable_memory_info(cuda, 0)


def test_current_device_is_used_when_unspecified():
    cuda = device(False)
    seen = []
    cuda.current_device = lambda: 2
    cuda.get_device_properties = lambda index: (seen.append(index) or SimpleNamespace(is_integrated=False))
    cuda.mem_get_info = lambda index: (seen.append(index) or (123, 456))
    assert memory.cuda_allocatable_memory_info(cuda) == (123, 456, 'cudaMemGetInfo')
    assert seen == [2, 2]


def test_rust_logical_admission_keeps_allocatable_sampling():
    text = (ROOT / 'rust/crates/cuteafd-ffi/src/lib.rs').read_text()
    method = text.split('pub fn cuda_memory_info(&self)', 1)[1].split('fn admit_gpu_allocation', 1)[0]
    assert 'self.cuda_allocatable_memory_info()?' in method
    assert 'budget.apply(sample)?' in method

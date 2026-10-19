"""Allocatable CUDA headroom: integrated devices reclaim Linux file cache."""
from __future__ import annotations

import ctypes
import ctypes.util
from pathlib import Path
import re
import sys


def mem_available_bytes(text: str) -> int:
    values = [line.removeprefix('MemAvailable:') for line in text.splitlines()
              if line.startswith('MemAvailable:')]
    if len(values) != 1 or not re.fullmatch(r'\s*[0-9]+\s+kB\s*', values[0]):
        raise ValueError('integrated-device admission requires valid MemAvailable')
    available = int(values[0].split()[0]) * 1024
    if available > sys.maxsize * 2 + 1:
        raise ValueError('MemAvailable exceeds the addressable byte count')
    return available


def _unsigned(text: str) -> int:
    if not re.fullmatch(r'[0-9]+', text.strip()):
        raise ValueError('invalid cgroup memory value')
    return int(text.strip())


def cgroup_allocatable_bytes(available: int, membership: str,
                             root: Path = Path('/sys/fs/cgroup')) -> int:
    """Bound by every visible v2/v1 ancestor, including a namespaced mount root."""
    for line in membership.splitlines():
        fields = line.split(':', 2)
        if len(fields) != 3:
            continue
        _, controllers, relative = fields
        if not controllers:
            mount, limit_name, current_name = root, 'memory.max', 'memory.current'
        elif 'memory' in controllers.split(','):
            mount = root / 'memory'
            limit_name, current_name = 'memory.limit_in_bytes', 'memory.usage_in_bytes'
        else:
            continue
        relative = Path(relative.lstrip('/'))
        member = mount if '..' in relative.parts else mount / relative
        for path in (member, *member.parents):
            if path != mount and mount not in path.parents:
                break
            try:
                limit = (path / limit_name).read_text().strip()
            except FileNotFoundError:
                continue
            if limit == 'max':
                continue
            limit = _unsigned(limit)
            current = _unsigned((path / current_name).read_text())
            available = min(available, max(0, limit - current))
    return available


def host_allocatable_memory(*, meminfo: Path = Path('/proc/meminfo'),
                            membership: Path = Path('/proc/self/cgroup'),
                            cgroup_root: Path = Path('/sys/fs/cgroup')) -> int:
    """MemAvailable already includes reclaimable cache; never add cache twice."""
    return cgroup_allocatable_bytes(mem_available_bytes(meminfo.read_text()),
                                    membership.read_text(), cgroup_root)


def device_is_integrated(cuda, device: int = 0) -> bool:
    properties = cuda.get_device_properties(device)
    integrated = getattr(properties, 'is_integrated', None)
    if integrated is not None:
        return bool(integrated)
    # Some PyTorch builds omit cudaDeviceProp.integrated. Query that same
    # property through cudart, never infer UMA from the compute capability.
    runtime = ctypes.CDLL(ctypes.util.find_library('cudart') or 'libcudart.so.13')
    query = runtime.cudaDeviceGetAttribute
    query.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    query.restype = ctypes.c_int
    value = ctypes.c_int()
    status = query(ctypes.byref(value), 18, device)  # cudaDevAttrIntegrated
    if status:
        raise RuntimeError(f'cudaDeviceGetAttribute(integrated) failed: {status}')
    return bool(value.value)


def cuda_allocatable_memory_info(cuda, device: int | None = None, **host_paths) -> tuple[int, int, str]:
    """Return device headroom; GB10 CUDA pages bypass the host-only memcg cap."""
    device = cuda.current_device() if device is None else device
    if not device_is_integrated(cuda, device):
        free, total = cuda.mem_get_info(device)
        return free, total, 'cudaMemGetInfo'
    total = int(cuda.get_device_properties(device).total_memory)
    if total <= 0:
        raise ValueError('invalid integrated-device total memory')
    # Keep cgroup bounds for host_allocatable_memory, not CUDA device admission.
    meminfo = host_paths.get('meminfo', Path('/proc/meminfo'))
    return min(mem_available_bytes(meminfo.read_text()), total), total, 'MemAvailable'


if __name__ == '__main__':
    import torch
    free, _, measure = cuda_allocatable_memory_info(torch.cuda)
    print(f'{free} {measure}')

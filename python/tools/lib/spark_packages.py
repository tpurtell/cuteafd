"""Shared, checkpoint-provenanced Spark package coverage (no CUDA imports)."""
from __future__ import annotations

import json
from pathlib import Path

RANGES_PATH = Path(__file__).resolve().parents[3] / 'scripts/build/spark-package-ranges.json'


def package_entry(family: str, format: str) -> dict:
    format = 'exl3' if format.startswith('exl3') else format.removesuffix('a4')
    return json.loads(RANGES_PATH.read_text())['entries'][f'{family}:{format}']


def width_keys(intermediate: int, unit: int, world: int, *, exl3: bool = False,
               native: bool = False, nvfp4: bool = False) -> set[tuple[bool, int]]:
    """Descriptor ABI and stored columns; world/rank do not determine code."""
    if not 1 <= world <= 8 or intermediate % unit or intermediate // unit < world:
        return set()
    units = intermediate // unit
    if exl3 and units % world:
        return {(True, -(-units // world) * unit)}
    if native or (nvfp4 and intermediate % (128 * world)):
        widest = -(-units // world) * unit
        return {(False, -(-widest // 128) * 128)}
    return {(False, units // world * unit), (False, -(-units // world) * unit)}


def package_worlds(family: str, format: str) -> list[int]:
    entry = package_entry(family, format)
    if entry.get('deprecated'):
        return entry['worlds']
    exl3 = format.startswith('exl3')
    native = family in ('v41', 'dsv4f', 'dsv4p') and not exl3
    unit = 32 if native or family in ('mimof', 'mimop') else 128
    intermediate = entry['intermediate']
    options = dict(exl3=exl3, native=native)
    available = set().union(*(width_keys(intermediate, unit, n, **options) for n in entry['worlds']))
    # Emit every metadata alias the planner can admit using these same objects.
    return [n for n in range(1, 9)
            if (not native or intermediate % (n * 32) == 0)
            and (needed := width_keys(intermediate, unit, n, **options))
            and needed <= available]


def geometry_worlds(geometry: str) -> list[int]:
    family, _, format = geometry.partition('_')
    return package_worlds(family, format or 'fp8')


def ownership_topology(tp: int, ep: int) -> bool:
    """Wire qualification; the checkpoint still decides its own native recipe."""
    return (ep == 1 and any(tp in package_worlds(family, 'mxfp4')
                           for family in ('v41', 'dsv4f', 'dsv4p'))) or (tp, ep) in ((2, 2), (3, 2), (2, 3))

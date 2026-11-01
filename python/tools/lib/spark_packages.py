"""Shared, checkpoint-provenanced Spark package coverage (no CUDA imports)."""
from __future__ import annotations

import importlib.util
import json
from typing import NamedTuple
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
    if nvfp4 and intermediate % (128 * world):
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
            if (needed := width_keys(intermediate, unit, n, **options))
            and needed <= available]


def geometry_worlds(geometry: str) -> list[int]:
    family, _, format = geometry.partition('_')
    return package_worlds(family, format or 'fp8')


def ownership_topology(tp: int, ep: int) -> bool:
    """Wire qualification; the checkpoint still decides its own native recipe."""
    return (ep == 1 and any(tp in package_worlds(family, 'mxfp4')
                           for family in ('v41', 'dsv4f', 'dsv4p'))) or (tp, ep) in ((2, 2), (3, 2), (2, 3))


# Stable legacy ABI ids. Unequal source ownership selects width-keyed artifacts.
NATIVE_SPARK_ROLE_IDS = {2: 5, 3: 6, 4: 1, 6: 7, 8: 8}
NATIVE_WIDTH_ROLE_BASE = 65536


def native_widths(intermediate: int, world: int) -> list[int]:
    if not 1 <= world <= 8 or intermediate <= 0 or intermediate % 32 or intermediate // 32 < world:
        raise ValueError("invalid native K32 shard geometry")
    units = intermediate // 32
    return sorted({32 * (units // world), 32 * -(-units // world)}, reverse=True)


def native_width_role(width: int) -> int:
    if width <= 0 or width % 32 or width > 8192:
        raise ValueError("invalid native artifact width")
    return NATIVE_WIDTH_ROLE_BASE + width


def native_width_inventory(intermediate: int, worlds: list[int]) -> list[int]:
    """Only unequal worlds need additive width-keyed native entry points."""
    return sorted({width for world in worlds if (intermediate // 32) % world
                   for width in native_widths(intermediate, world)}, reverse=True)


class NativeArtifact(NamedTuple):
    role: int
    logical: int
    stored: int
    name: str

    def prefix(self, family: str = "v41") -> str:
        suffix = "" if self.name == "spark" else "_" + self.name
        return f"cuteafd_{family}{suffix}_expert_"


def native_spark_artifacts(family: str = "v41") -> dict[int, tuple[NativeArtifact, ...]]:
    """World -> distinct adjacent executable artifacts, including equal worlds.

    Legacy equal-world symbol identities are retained. Source ownership is not
    an executable key: unequal ranks rotate between these same two artifacts.
    Load the adjacent policy file explicitly so isolated release archives use
    their own inputs rather than a publisher checkout on sys.path.
    """
    spec = importlib.util.spec_from_file_location(
        "spark_mxfp4_layouts", Path(__file__).with_name("mxfp4_layouts.py"))
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    hidden = {"v41": 5120, "dsv4f": 4096, "dsv4p": 7168}[family]
    intermediate = package_entry(family, "mxfp4")["intermediate"]
    result = {}
    for world in package_worlds(family, "mxfp4"):
        widths = native_widths(intermediate, world)
        unequal = len(widths) > 1
        result[world] = tuple(NativeArtifact(
            native_width_role(width) if unequal else NATIVE_SPARK_ROLE_IDS[world],
            width, policy.stored_width(hidden, width),
            f"spark_w{width}" if unequal else "spark" if world == 4 else f"spark_tp{world}")
            for width in widths)
    return result


def native_shard(world: int, rank: int, layer: int, family: str = "v41") -> tuple[int, NativeArtifact]:
    """Actual checkpoint source offset and artifact for one absolute layer."""
    artifacts = native_spark_artifacts(family)[world]
    intermediate = package_entry(family, "mxfp4")["intermediate"]
    if not 0 <= rank < world or layer < 0:
        raise ValueError("invalid native shard rank/layer")
    units = intermediate // 32
    base, extra = divmod(units, world)
    canonical = (rank - layer) % world if extra else rank
    width = 32 * (base + (canonical < extra))
    first = 32 * (canonical * base + min(canonical, extra))
    return first, next(artifact for artifact in artifacts if artifact.logical == width)

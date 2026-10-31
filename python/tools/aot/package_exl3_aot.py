#!/usr/bin/env python3
"""Build relocatable native EXL3 packages; verify them without importing CUDA."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1] / _d) for _d in ("lib",)]  # sibling tool dirs

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import gc
import hashlib
import json
import os
import signal
import struct
from pathlib import Path
import shutil
import subprocess
import tempfile


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_object_target(path, compute):
    """Check embedded CUDA ELF flags, not merely the exporter's target label."""
    data = path.read_bytes()
    target = compute[0] * 10 + compute[1]
    offset = 0
    found = False
    while (offset := data.find(b'\x7fELF', offset)) >= 0:
        start = offset
        offset += 4
        if len(data) - start < 64 or data[start:start + 6] != b'\x7fELF\x02\x01':
            continue
        if struct.unpack_from('<H', data, start + 18)[0] != 190:
            continue
        found = True
        if (struct.unpack_from('<I', data, start + 48)[0] >> 8) & 0xff != target:
            raise ValueError(f'EXL3 CUDA object target differs from package: {path}')
    if not found:
        raise ValueError(f'EXL3 object has no CUDA image: {path}')


def verify(package: Path, revision: str | None = None, runtime: Path | None = None, role: str | None = None) -> dict:
    manifest = json.loads((package / 'manifest.json').read_text())
    if manifest['schema'] != 'cuteafd.exl3-package.v1':
        raise ValueError('unsupported EXL3 package schema')
    if revision is not None and manifest['sparkinfer_revision'] != revision:
        raise ValueError('EXL3 package/source revision mismatch')
    if role is not None and manifest['role'] != ('spark' if role == 'expert' else role):
        raise ValueError('EXL3 package/serving role mismatch')
    expected = manifest['files']
    actual = {str(p.relative_to(package)) for p in package.rglob('*') if p.is_file()}
    if actual != set(expected) | {'manifest.json'}:
        raise ValueError('EXL3 package contains missing or unlisted files')
    for name, spec in expected.items():
        path = package / name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink():
            raise ValueError(f'unsafe EXL3 package path: {name}')
        if path.stat().st_size != spec['bytes'] or digest(path) != spec['sha256']:
            raise ValueError(f'EXL3 package file mismatch: {name}')
        if path.suffix == '.so':
            verify_object_target(path, manifest['compute'])
    if runtime is not None and digest(runtime) != manifest['runtime']['sha256']:
        raise ValueError('EXL3 CuTe runtime mismatch')
    required = set()
    seen = set()
    for variant in manifest['variants']:
        directory = variant['directory']
        if Path(directory).is_absolute() or '..' in Path(directory).parts:
            raise ValueError('unsafe EXL3 variant directory')
        if directory in seen:
            raise ValueError('duplicate EXL3 package variant')
        seen.add(directory)
        meta = json.loads((package / directory / 'v41_exl3.json').read_text())
        if len(meta['bits']) != 2:
            raise ValueError(f'EXL3 packages require two decoder tiers: {directory}')
        if meta.get('compute') != manifest['compute']:
            raise ValueError(f'EXL3 variant target differs from package target: {directory}')
        if manifest.get('geometry') not in GEOMETRIES:
            raise ValueError('unsupported EXL3 package geometry')
        if meta.get('hidden') != GEOMETRIES[manifest['geometry']][0]:
            raise ValueError(f'EXL3 variant hidden size does not match the package geometry: {directory}')
        for key in ('capacity', 'intermediate', 'experts', 'top_k', 'output_dtype', 'bits'):
            if meta[key] != variant[key]:
                raise ValueError(f'EXL3 variant metadata mismatch: {directory}/{key}')
        compact = '-compact-' in directory.split('/')[0]
        contract = (7, 3) if compact else (3, 2)
        for record in (variant, meta):
            if (record.get('descriptor_rows', 3), record.get('native_info_version', 2)) != contract:
                raise ValueError(f'EXL3 variant descriptor contract mismatch: {directory}')
        if compact:
            geometry = manifest['geometry']
            companions = compact_profiles(geometry, profiles_for_role('spark', geometry))
            widths = {dest: width for _, width, _, _, _, destinations in companions
                      for dest in destinations}
            ws = meta.get('warp_specialized', False)
            cw_tile = (ws_tile(geometry, manifest['role'], meta['intermediate'], meta['capacity'],
                               compact=True) if ws else None)
            if (manifest['role'] != 'spark'
                    or widths.get(directory.split('/')[0]) != meta['intermediate']
                    or len(meta['bits']) != 2
                    or (meta.get('tile') != (list(cw_tile) if cw_tile is not None else None) if ws
                        else meta.get('tile') != [64, 128, 64, 128])
                    or (ws and (not compact_ws_export(geometry, manifest['role'],
                                                     meta['intermediate'], meta['capacity'],
                                                     meta['bits'])
                                or meta.get('route_block', 8) != compact_ws_route_block(
                                    geometry, manifest['role'], meta['intermediate'], meta['capacity'])
                                or meta.get('ws_dynamic_tiles', False) != ws_dynamic_tiles(
                                    geometry, manifest['role'], meta['intermediate'], meta['capacity'],
                                    compact=True)
                                or meta.get('ws_input_stages') is not None
                                or meta.get('token_major_rotation', False)
                                or meta.get('fused_input_rotation', False)))
                    or meta.get('decode_schedule') not in (None, 'l2=2,pf1=0,pf2=0,pdl=1')
                    or meta.get('input_format', 'bf16') != 'bf16'):
                raise ValueError(f'EXL3 invalid compact geometry or execution contract: {directory}')
        if 'blocks_per_sm' in variant and variant['blocks_per_sm'] != meta.get('blocks_per_sm'):
            raise ValueError('EXL3 variant residency metadata mismatch')
        # A variant's tile is a claim about the geometry that was compiled, not a
        # build-argument echo, so it is checked against the export's own record.
        # Conditional on the variant carrying the key: v9 manifests predate it and
        # must keep verifying byte-for-byte unchanged.
        for field in ('tile', 'tile_requested'):
            value = variant.get(field)
            if field in variant and (not isinstance(value, list) or len(value) != 4
                    # bool is an int subclass; a True/False tile is a broken record.
                    or not all(isinstance(n, int) and not isinstance(n, bool) and n > 0
                               for n in value)):
                raise ValueError(f'EXL3 variant {field} must be four positive integers: {directory}')
        if 'tile' in variant and variant['tile'] != meta.get('tile'):
            raise ValueError(f'EXL3 variant tile mismatch: {directory}')
        if 'tile_requested' in variant and variant['tile_requested'] != meta.get('tile'):
            raise ValueError(f'EXL3 tile override was not applied as requested: {directory}')
        # Absent means the original 8-row packed-route block.
        if variant.get('route_block', 8) != meta.get('route_block', 8):
            raise ValueError(f'EXL3 variant route block mismatch: {directory}')
        if variant.get('warp_specialized', False) != meta.get('warp_specialized', False):
            raise ValueError(f'EXL3 variant warp specialization mismatch: {directory}')
        if variant.get('ws_dynamic_tiles', False) != meta.get('ws_dynamic_tiles', False):
            raise ValueError(f'EXL3 variant dynamic tile claim mismatch: {directory}')
        if variant.get('ws_input_stages') != meta.get('ws_input_stages'):
            raise ValueError(f'EXL3 variant input stage mismatch: {directory}')
        if variant.get('input_format', 'bf16') != meta.get('input_format', 'bf16'):
            raise ValueError(f'EXL3 variant input format mismatch: {directory}')
        # The variant records its exact compiled schedule, including defaults.
        # Explicit schedule siblings must still carry a schedule and suffix.
        schedule = variant.get('schedule')
        if (variant.get('decode_schedule') != meta.get('decode_schedule')
                or (schedule is not None and meta.get('decode_schedule') is None)
                or directory.rsplit('/', 1)[-1] != f"m{meta['capacity']}"
                + ('' if schedule is None else f'-{schedule}')):
            raise ValueError(f'EXL3 variant decode schedule mismatch: {directory}')
        if variant.get('token_major_rotation', False) != meta.get('token_major_rotation', False):
            raise ValueError(f'EXL3 variant input rotation mismatch: {directory}')
        if meta['sparkinfer_revision'] != manifest['sparkinfer_revision']:
            raise ValueError('EXL3 variant/source revision mismatch')
        required.update(f'{directory}/{name}' for name in
                        ('v41_exl3.json', 'trellis_lut.bin', 'libcuteafd_exl3.so'))
        if meta['requires_route_preparation']:
            required.update(f'{directory}/routes/{name}' for name in
                            ('v41_exl3_routes.json', 'libv41_exl3_routes.so'))
            if digest(package / directory / meta['route_preparation']['manifest']) != meta['route_preparation']['sha256']:
                raise ValueError('EXL3 route manifest mismatch')
    if not seen or not required.issubset(expected):
        raise ValueError('incomplete EXL3 package')
    # `requested_layouts` records what the build was asked to contain, so a partial
    # Spark export cannot be published as a full contract. Packages built before
    # this existed (v9 and earlier) carry no key and verify exactly as before. The
    # check is a minimum, not an exact set: a package may legitimately carry more
    # layouts than one consumer asked about.
    # The capacity set comes from the contract, not from the variants that happen to
    # exist: otherwise dropping one capacity for every rank would still verify.
    # Older packages record no capacities, so they keep the previous behaviour.
    requested_capacities = manifest.get('requested_capacities') or sorted(
        {v['capacity'] for v in manifest['variants']})
    for layout in manifest.get('requested_layouts') or []:
        for capacity in requested_capacities:
            if not any(v['directory'] == f'{layout}/m{capacity}' for v in manifest['variants']):
                raise ValueError(f'EXL3 package is missing requested layout {layout}/m{capacity}')
    return manifest


def verify_root(root: Path, revision: str | None = None, runtime: Path | None = None,
                role: str | None = None) -> list[dict]:
    """Verify an EXL3 root holding either one flat package or family packages.

    Release images keep a single well-known ``exl3/`` root. A multi-family build
    nests ``exl3-kXX`` packages beneath it; a direct ``manifest.json`` in the
    root is the legacy single-family layout. Both are accepted, and this is
    strict by design:

    * a family directory without ``manifest.json`` is an error, not a silently
      skipped entry, so one valid package can never mask a broken sibling;
    * a root carrying both a flat manifest and family packages is rejected as
      ambiguous rather than verifying only one of them;
    * every discovered package goes through the unchanged strict single-package
      ``verify``, and a root yielding no package at all is an error.
    """
    flat_manifest = root / 'manifest.json'
    children = sorted(child for child in root.iterdir() if child.is_dir()) if root.is_dir() else []
    families = [child for child in children if child.name.startswith('exl3-')]
    stray = [
        child for child in children
        if not child.name.startswith('exl3-') and (child / 'manifest.json').is_file()
    ]
    if flat_manifest.is_file() and families:
        raise ValueError(
            f'ambiguous EXL3 root has both a flat manifest and family packages: {root}'
        )
    if stray:
        raise ValueError(f'unexpected EXL3 package directory (expected exl3-<bits>): {stray[0]}')
    if flat_manifest.is_file():
        return [verify(root, revision, runtime, role)]
    if not families:
        raise ValueError(f'no EXL3 package or family manifest under {root}')
    missing = [family for family in families if not (family / 'manifest.json').is_file()]
    if missing:
        raise ValueError(f'EXL3 family is missing its manifest.json: {missing[0]}')
    return [verify(family, revision, runtime, role) for family in families]


def validate_destination(output: Path) -> None:
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError('EXL3 output must be a package directory')
    # Ninja creates the parent directories of declared BYPRODUCTS before the
    # command runs. An empty directory tree is still a fresh destination.
    # Symlinks are never treated as empty scaffolding, including dangling ones.
    if output.exists() and any(p.is_symlink() or not p.is_dir() for p in output.rglob('*')):
        marker = output / 'manifest.json'
        if not marker.is_file() or json.loads(marker.read_text()).get('schema') != 'cuteafd.exl3-package.v1':
            raise ValueError('refusing to replace a non-package directory')


def install_package(source: Path, output: Path) -> None:
    validate_destination(output)
    src, dst = source.resolve(), output.resolve()
    if src == dst or src.is_relative_to(dst) or dst.is_relative_to(src):
        raise ValueError('EXL3 source and destination must not overlap')
    verify(source)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    for path in sorted(source.iterdir()):
        if path.name == 'manifest.json':
            continue
        if path.is_dir():
            shutil.copytree(path, output / path.name)
        else:
            shutil.copy2(path, output / path.name)
    # Completion marker is installed only after all payload files.
    shutil.copy2(source / 'manifest.json', output / 'manifest.json')
    verify(output)


# Routed-expert geometries the EXL3 packages can target: (hidden, intermediate,
# experts, top-k). The names match cuteafd_core::ExpertGeometry::family.
GEOMETRIES = {
    'dsv4p': (7168, 3072, 384, 6),
    'glm': (6144, 2048, 256, 8),
    # GLM 5.3 Flash (glm5_next): 288 experts, SwiGLU clamped at 10.
    'glmf': (4096, 2048, 288, 8),
    # Qwen 3.8 Flash Next (qwen4_exp): 512 experts, softmax top-10, unclamped SiLU.
    'qwen4': (2560, 640, 512, 10),
    'kimik3': (3584, 3072, 896, 16),
    # Muse Spark: TP intermediate slices at the 4096-wide expert stream.
    'muse': (4096, 4096, 256, 8),
}
# SwiGLU clamp per geometry; None is the unclamped SwiGLU (b12x const-expr
# elides the clamp). DeepSeek clamps at 10.
SWIGLU_LIMITS = {'glm': None, 'qwen4': None, 'kimik3': None, 'muse': None}


def activation(geometry: str) -> str:
    return 'situ' if geometry == 'kimik3' else 'silu'


def swiglu_limit(geometry: str) -> float | None:
    return SWIGLU_LIMITS.get(geometry, 10.0)


def route_block(geometry: str, capacity: int, role: str) -> int:
    """Cooperative decode and the remaining V4 Pro/Kimi prefill recipes."""
    override = dict(item.split('=') for item in os.environ.get('CUTEAFD_EXL3_ROUTE_BLOCKS', '').split(',') if item)
    if str(capacity) in override:
        return int(override[str(capacity)])
    if role == 'coordinator':
        if geometry == 'dsv4p' and capacity == 2048:
            return 64
    return 8


def token_major_rotation(geometry: str, capacity: int, role: str) -> bool:
    """Token rotation only for surviving cooperative coordinator prefill."""
    return role == 'coordinator' and geometry == 'dsv4p' and capacity == 2048


# Warp-specialized tile with 128-wide FC1 N tiles and a 256-wide FC2: every
# coordinator rank width and hidden size is a whole 128 tiles, so FC1 never
# straddles a live edge while FC2 halves the FC2 N tile count.
TILE_RTX_MIXED = (64, 128, 64, 256)

# The S1/S4 arm as it exports on SM120: 128-wide FC1/256-wide FC2 tiles with
# dynamic tile claims, 32-row route blocks at m256 and 64 rows above it.
_WS_S14_M256 = dict(route_block=32, dynamic=True, tile=TILE_RTX_MIXED)
_WS_S14 = dict(route_block=64, dynamic=True, tile=TILE_RTX_MIXED)
# Prerotated 256-wide FC1 arm: the WS kernel hoists the token-major rotation
# ahead of the block loop and keeps dynamic claims with 64-row route blocks.
# The hoist is the SparkInfer default on SM120 only; SM121 (GB10) exports rotate
# in tile, which measures 1.05-1.53x faster on every Spark WS cell (#912).
_WS_K3_PREROTATE = dict(route_block=64, dynamic=True, tile=(64, 256, 64, 256))

# Coordinator (RTX SM120) warp-specialized prefill cells, keyed by
# (geometry, width, capacity): the measured options each cell carries, against
# the export the coordinator shipped before it. See `warp_specialized`.
COORDINATOR_WS_CELLS: dict[tuple[str, int, int], dict] = {
    ('glm', 1024, 256): dict(route_block=64, wire=True, stages=3, dynamic=True),
    ('glm', 1024, 1024): dict(route_block=64, dynamic=True),
    ('glm', 1024, 2752): dict(route_block=64),
    ('glm', 1024, 2048): dict(route_block=64),
    ('glm', 1024, 4096): dict(route_block=64, dynamic=True),
    ('glm', 2048, 256): dict(route_block=64, dynamic=True),
    ('glm', 2048, 4096): dict(route_block=64, dynamic=True),
    ('glmf', 1024, 256): dict(route_block=64, dynamic=True),
    ('glmf', 1024, 1024): dict(route_block=64, dynamic=True),
    ('glmf', 1024, 2752): dict(route_block=64, dynamic=True),
    ('glmf', 1024, 4096): dict(route_block=64, dynamic=True),
    ('glmf', 1024, 2048): dict(route_block=64),
    ('glmf', 2048, 256): dict(route_block=64),
    ('dsv4p', 1536, 4096): dict(route_block=64, dynamic=True),
    # Qwen 3.8 Flash Next wins at every served capacity, both RTX splits.
    ('qwen4', 640, 256): _WS_S14_M256,
    ('qwen4', 384, 256): _WS_S14_M256,
    ('qwen4', 256, 256): _WS_S14_M256,
    ('qwen4', 640, 1024): _WS_S14,
    ('qwen4', 384, 1024): _WS_S14,
    ('qwen4', 256, 1024): _WS_S14,
    ('qwen4', 640, 2752): _WS_S14,
    ('qwen4', 640, 2048): _WS_S14,
    # TP2 rank0: N192 wins both exact prefill units on random and recorded routes.
    ('qwen4', 384, 2752): dict(route_block=64, dynamic=True, tile=(64, 192, 64, 256)),
    ('qwen4', 384, 2048): dict(route_block=64, dynamic=True, tile=(64, 192, 64, 256)),
    ('qwen4', 256, 2752): dict(route_block=64, dynamic=True, tile=(64, 256, 64, 256)),
    ('qwen4', 256, 2048): dict(route_block=64, dynamic=True, tile=(64, 256, 64, 256)),
    ('qwen4', 640, 4096): _WS_S14,
    ('qwen4', 384, 4096): _WS_S14,
    ('qwen4', 256, 4096): _WS_S14,
    # V4.1 EXL3 wins from m1024 up; m256 stays cooperative (0.77-0.84x).
    # Kimi K3 ships the warp-specialized kernel on every coordinator prefill
    # export (TJ 2026-10-29, kernel-tie rule): the prerotated 256-wide FC1 arm
    # ties the cooperative 64-row token-major rotation at m1024-m2752 (geomean
    # 1.005, inside round spread) and wins 1.25-1.40x above 2752, where wave
    # merging hands one 4096/5504/8256-row packed extent to the kernel.
    # Interleaved duel session, GPU0 pinned 450 W, 32 exact fixtures with
    # every poisoned prefix and graph replay: rtx-tp1 1.297x m4096, rtx-tp2
    # 1.392x; m5504 1.32/1.25x and m8256 1.32/1.26x. The arm hoists FC1's
    # token-major rotation ahead of the block loop and keeps dynamic claims
    # with 64-row route blocks.
    #
    # Every prefill capacity takes it, m256 included, so the kernel sees one
    # recipe from a short prompt to a merged three-wave extent. The two decode
    # capacities (m1, m80) stay cooperative: `Exl3ReductionPlan` freezes their
    # manifest tile, route block and kernel class as the reduction geometry of
    # every decode row, so moving them would hand C1 decode the prefill kernel
    # with 256-wide FC1 tiles. The duel's floor is m1024, so that assignment is
    # unmeasured, and no other family routes a decode capacity through WS.
    ('kimik3', 3072, 256): _WS_K3_PREROTATE,
    ('kimik3', 1536, 256): _WS_K3_PREROTATE,
    ('kimik3', 3072, 1024): _WS_K3_PREROTATE,
    ('kimik3', 1536, 1024): _WS_K3_PREROTATE,
    ('kimik3', 3072, 2048): _WS_K3_PREROTATE,
    ('kimik3', 1536, 2048): _WS_K3_PREROTATE,
    ('kimik3', 3072, 2752): _WS_K3_PREROTATE,
    ('kimik3', 1536, 2752): _WS_K3_PREROTATE,
    ('kimik3', 3072, 4096): _WS_K3_PREROTATE,
    ('kimik3', 1536, 4096): _WS_K3_PREROTATE,
    ('kimik3', 3072, 5504): _WS_K3_PREROTATE,
    ('kimik3', 1536, 5504): _WS_K3_PREROTATE,
    ('kimik3', 3072, 8256): _WS_K3_PREROTATE,
    ('kimik3', 1536, 8256): _WS_K3_PREROTATE,
}


# ABI8 RTX screen (#862), GPU0 at 450 W: TP1 m256 0.8335-0.8968x,
# m2048 0.9494-0.9495x. Main retains only these cooperative prefill cells;
# other noisy cells inherit the WS decision from their measured neighbours.
COOPERATIVE_PREFILL_CELLS = {('dsv4p', 3072, 256), ('dsv4p', 3072, 2048)}


def coordinator_ws_cell(geometry: str, role: str, width: int, capacity: int) -> dict | None:
    """Measured coordinator warp-specialized options for one export cell."""
    if role != 'coordinator':
        return None
    if capacity <= 80 or (geometry, width, capacity) in COOPERATIVE_PREFILL_CELLS:
        return None
    measured = COORDINATOR_WS_CELLS.get((geometry, width, capacity))
    # Kimi RTX is owned by its separately measured kernel integration.
    if geometry == 'kimik3':
        return measured
    if measured is not None:
        return measured
    # Preserve measured cells; extend the nearest same-width WS recipe first.
    candidates = [(abs(c - capacity), options) for (g, w, c), options
                  in COORDINATOR_WS_CELLS.items() if g == geometry and w == width]
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]
    return _WS_S14_M256 if capacity <= 256 else _WS_S14


# Ordinary GB10 prefill cells qualified against shipped family controls (#762).
SPARK_WS_CELLS = {
    ('qwen4', 640): dict(dynamic=False, tile=(64, 128, 64, 128)),
    ('qwen4', 384): dict(dynamic=True, tile=(64, 192, 64, 256)),
    ('qwen4', 256): dict(dynamic=False, tile=(64, 256, 64, 256)),
    ('qwen4', 128): dict(dynamic=True, tile=(64, 128, 64, 256)),
}


def spark_ws_cell(geometry: str, role: str, width: int, capacity: int) -> dict | None:
    if role != 'spark' or capacity <= 80:
        return None
    return SPARK_WS_CELLS.get((geometry, width))


def ws_route_block(geometry: str, role: str, width: int, capacity: int) -> int:
    """Packed-route M block of a warp-specialized prefill export (see
    `warp_specialized`); every cell whose measured arm split the block by
    capacity carries its own row count here."""
    cell = coordinator_ws_cell(geometry, role, width, capacity)
    return cell['route_block'] if cell else 64


def warp_specialized(geometry: str, role: str, width: int, capacity: int) -> bool:
    """WS prefill except two measured V4 Pro losses; cooperative decode."""
    if role == 'coordinator' and (geometry, width, capacity) in COOPERATIVE_PREFILL_CELLS:
        return False
    if geometry == 'kimik3' and role == 'coordinator':
        return coordinator_ws_cell(geometry, role, width, capacity) is not None
    return role in ('coordinator', 'spark') and capacity > 80


# Measured compact warp-specialized cells on GB10 (SM121), keyed by
# (geometry, width): each qualified capacity maps to its measured
# (route block, FC1 N tile). The packaged compact profile carries 256, 1024
# and 4096, and 2752 is the planner's intermediate prefill unit. Unlisted
# capacities inherit the nearest recipe; unlisted widths use an N128 fallback.
COMPACT_WS_CELLS = {
    # V4 Pro TP5 has its own N128 recipe: the retired V4.1 N192 masked-tail
    # measurement does not qualify this geometry.
    ('dsv4p', 640): {256: (32, 128), 1024: (64, 128), 2048: (64, 128), 2752: (64, 128), 4096: (64, 128)},
    # GLM 5.3 K4/K5 TP6 width 384. Round 2 (same-session A/B against the
    # cooperative oracle and the entrant's N192 arm) found the 192-wide FC1
    # tile wins from m1024 up: m4096 23.019 -> 13.023 ms N128 (1.768x) /
    # 12.538 N192 (1.836x), m2752 16.646 -> 9.342 / 9.040, m1024 7.836 ->
    # 4.627 / 4.601; at m256 the 32-row N128 arm stays best (4.351 -> 3.408,
    # 1.277x, vs 3.413 for N192).
    ('glm', 384): {256: (32, 128), 1024: (64, 192), 2048: (64, 192), 2752: (64, 192), 4096: (64, 192)},
    # GLM 5.3 TP3 width 768: N192 wins from m1024 up (m4096 31.624 -> 22.012 /
    # 21.206, m2752 23.474 -> 15.701 / 15.361, m1024 11.384 -> 8.297 / 8.237);
    # m256 keeps the 64-row N128 arm (7.492 -> 6.527 vs 6.541 for N192).
    ('glm', 768): {256: (64, 128), 1024: (64, 192), 2048: (64, 192), 2752: (64, 192), 4096: (64, 192)},
    # V4.1 K3/K4 TP4 width 640 (the only uneven V4.1 world): the 32-row block
    # wins m256/m1024 and the 64-row block m2752/m4096. Round 2 of the Spark
    # duel measured a non-divisible 192-wide FC1 tile here (640 = 2 x 320-row
    # halves -> 2 x 192-wide tiles with a 128-column masked tail, uniform
    # top-6) faster than N128 at m256/m1024/m2752 (5.594 vs 5.630, 6.630 vs
    # 6.766, 9.942 vs 10.093 ms) and slower at m4096 (13.868 vs 13.831), so
    # N192 takes the three smaller capacities and m4096 keeps N128.
    # Qwen4 K4/K5 TP2 compact width 384: N192 wins m1024/m2752 (3.141 vs
    # 3.159, 5.549 vs 5.596) and ties m4096 (7.570 vs 7.559), so the large
    # capacities consolidate onto the same N192/64 arm as the GLM cells; only
    # m256 keeps N128 with 32 rows (1.933 vs 1.938 for N192, and 1.978 for
    # the 64-row arm).
    ('qwen4', 384): {256: (32, 128), 1024: (64, 192), 2048: (64, 192), 2752: (64, 192), 4096: (64, 192)},
    # TP3 and TP4 compact share width 256; one table entry serves both and
    # takes 64 everywhere (TP3 m256 1.279x vs its 32-row arm 1.282x, TP4
    # m256 1.334x vs 1.327x). Width 256 also rejects a 192-wide tile.
    ('qwen4', 256): {256: (64, 128), 1024: (64, 128), 2048: (64, 128), 2752: (64, 128), 4096: (64, 128)},
    # GLM Flash K3/K4 TP6 width 384: N192 from m1024 up (m4096 14.390 ->
    # 8.526 / 8.307, m2752 10.108 -> 6.438 / 6.241, m1024 4.779 -> 3.175 /
    # 3.062); m256 keeps 32-row N128 (2.814 -> 2.171 vs 2.177).
    ('glmf', 384): {256: (32, 128), 1024: (64, 192), 2048: (64, 192), 2752: (64, 192), 4096: (64, 192)},
    # GLM Flash TP3 width 768: N192 from m1024 up (m4096 20.085 -> 14.542 /
    # 13.928, m2752 14.319 -> 11.093 / 10.848, m1024 6.792 -> 5.645 / 5.405);
    # m256 keeps 32-row N128 (4.804 -> 4.140 vs 4.136).
    ('glmf', 768): {256: (32, 128), 1024: (64, 192), 2048: (64, 192), 2752: (64, 192), 4096: (64, 192)},

}
def compact_warp_specialized(geometry: str, role: str, width: int, capacity: int) -> bool:
    """Compact two-tier prefill shares the ordinary WS execution class."""
    return role == 'spark' and capacity > 80


def compact_ws_export(geometry: str, role: str, width: int, capacity: int,
                      bits: list[int]) -> bool:
    return len(bits) == 2 and compact_warp_specialized(geometry, role, width, capacity)


def compact_ws_cell(geometry: str, width: int, capacity: int) -> tuple[int, int]:
    cells = COMPACT_WS_CELLS.get((geometry, width))
    if cells:
        return cells[min(cells, key=lambda c: abs(c - capacity))]
    # A same-width measured recipe is preferable to a geometry-specific guess.
    peers = [cells for (g, w), cells in COMPACT_WS_CELLS.items() if w == width]
    if peers:
        cells = peers[0]
        return cells[min(cells, key=lambda c: abs(c - capacity))]
    return (32 if capacity <= 256 else 64, 128)


def compact_ws_route_block(geometry: str, role: str, width: int, capacity: int) -> int | None:
    return (compact_ws_cell(geometry, width, capacity)[0]
            if compact_warp_specialized(geometry, role, width, capacity) else None)


def compact_ws_tile(geometry: str, role: str, width: int, capacity: int) -> tuple[int, ...] | None:
    if not compact_warp_specialized(geometry, role, width, capacity):
        return None
    n = compact_ws_cell(geometry, width, capacity)[1]
    return (64, n, 64, 256)


# Warp-specialized tile (fc1_k, fc1_n, fc2_k, fc2_n) of widths that are odd
# multiples of 128 and multiples of 192 (GLM 5.3 TP6 width 384).
TILE_384 = (64, 192, 64, 256)
# Compact widths are live at 256/384 H128 blocks, so FC1 keeps 128-wide N
# tiles (a 192 tile straddles the live edge); FC2 widens over hidden size.
TILE_384_COMPACT = (64, 128, 64, 256)
# The measured 192-wide compact FC1 tile: only the cells whose width is a
# whole multiple of 192 and whose large capacities were measured with it.
TILE_384_COMPACT_N192 = (64, 192, 64, 256)


def wire_input(geometry: str, role: str, width: int, capacity: int, *,
               compact: bool = False) -> bool:
    """Whether a warp-specialized export reads the FP8 E4M3 + UE8M0 K32 wire
    rows itself (the worker then skips its BF16 decode pass and FC1 gathers
    half the bytes). Bit-identical. One of the GLM 5.3 GB10 changes below.
    GB10 (dodo), GLM 5.3 K4/K5 tiers, layer 40 slices, random top-8 routes,
    kernel + top-k sum, median of 3 interleaved runs, ms (m2752 / m4096), all
    bit-identical to the static cooperative-order exports, run-to-run exact:
    width 512 14.50 / 15.42 -> 11.07 / 14.33; 384 11.88 / 14.89 -> 9.70 / 13.11
    (with the 192-wide tile); 256 9.04 / 10.15 -> 7.34 / 9.36. Four input
    stages instead of three lose 0.2-0.8 ms at every width on K4/K5. Measured on
    GLM 5.3 only: GLM 5.3 Flash and V4 Pro keep the former exports until measured.

    SM120 RTX PRO 6000, 2026-10-24, width1024 m256: cooperative 1737.66 us
    -> wire/stage3/dynamic 1516.48 us (1.146x random, 1.204x median paired
    across 72 histograms), conditions in `warp_specialized`. At m1024 wire
    loses on random (0.927x); wider coordinator profiles keep BF16 input.
    """
    if compact:
        return False
    if role == 'coordinator':
        cell = coordinator_ws_cell(geometry, role, width, capacity)
        return bool(cell and cell.get('wire'))
    if role == 'spark' and width in (384, 640) and capacity in (2048, 2752):
        return False
    return geometry == 'glm' and warp_specialized(geometry, role, width, capacity)


def ws_input_stages(geometry: str, role: str, width: int, capacity: int, *,
                    compact: bool = False) -> int | None:
    """Input-row ring depth of the Spark warp-specialized exports: GB10 gathers
    the routed rows from LPDDR5X at high latency, and four stages (the ring the
    weight ring of 256-wide 4-bit tiles leaves room for) hide it; with K4/K5
    tiers three stages measure faster than four (see `wire_input`). Bit-identical.
    Compact widths keep the resolved default: wire input loses at width 384
    (see `compact_warp_specialized`), so its stages stay off too.
    """
    if compact:
        return None
    if role == 'coordinator':
        cell = coordinator_ws_cell(geometry, role, width, capacity)
        return cell.get('stages') if cell else None
    return 3 if wire_input(geometry, role, width, capacity) else None


def ws_dynamic_tiles(geometry: str, role: str, width: int, capacity: int, *,
                     compact: bool = False) -> bool:
    """Whether Spark warp-specialized exports claim tiles dynamically. With the
    static round-robin an expert's route blocks start on different CTAs at
    different times and each streams the expert's weights from LPDDR5X (ncu,
    width 512 m2752: FC1 fills 1761 MB of LPDDR5X for 805 MB of weights).
    Bit-identical; the largest of the GLM 5.3 GB10 gains (see `wire_input`).
    Compact width 384 takes them at every qualified capacity: m4096
    13.89 -> 12.94 ms, m2752 10.07 -> 9.53, m1024 5.16 -> 4.99, and m256 ties
    its static arm at 3.81 (see `compact_warp_specialized`).
    """
    if compact:
        return compact_warp_specialized(geometry, role, width, capacity)
    if role == 'coordinator':
        cell = coordinator_ws_cell(geometry, role, width, capacity)
        return bool(cell and cell.get('dynamic'))
    if (cell := spark_ws_cell(geometry, role, width, capacity)) is not None:
        return cell['dynamic']
    if (role == 'spark' and capacity in (2048, 2752)
            and ((geometry == 'glm' and width in (384, 640))
                 or (geometry == 'glmf' and width in (256, 384, 512, 640, 768, 1024)))):
        return True
    return wire_input(geometry, role, width, capacity)


def ws_tile(geometry: str, role: str, width: int, capacity: int, *,
            compact: bool = False) -> tuple[int, ...] | None:
    """Warp-specialized tile for widths that are odd multiples of 128, which
    the B12x policy gives 128-wide tiles on both stages (four consumer warps).
    Bit-identical (the N tiling does not change any accumulation). Width 384
    (GLM 5.3 TP6 ranks 0-3) takes 192-wide FC1 tiles: m4096 14.89 -> 12.8 ms.
    Compact ownership keeps FC1 at 128 except in the measured cells where the
    192-wide tile wins from m1024 up (`compact_ws_tile`), and widens FC2 over
    the unchanged hidden size.
    """
    if compact:
        return compact_ws_tile(geometry, role, width, capacity)
    if role == 'coordinator':
        cell = coordinator_ws_cell(geometry, role, width, capacity)
        return cell.get('tile') if cell else None
    if (cell := spark_ws_cell(geometry, role, width, capacity)) is not None:
        return cell['tile']
    if ((geometry, role, width) == ('glmf', 'spark', 384)
            and capacity in (2048, 2752)):
        return TILE_384
    if ((geometry, role, width) == ('glmf', 'spark', 640)
            and capacity in (2048, 2752)):
        return TILE_RTX_MIXED
    # GB10 w640: FC2 N256 wins 2.2-5.0% at m1024..m4096; FC1 stays N128.
    if ((geometry, role, width) == ('glm', 'spark', 640)
            and capacity in (1024, 2048, 2752, 4096)):
        return TILE_RTX_MIXED
    if (geometry != 'glm' or not warp_specialized(geometry, role, width, capacity)
            or width % 256 != 128 or width % 192):
        return None
    return TILE_384


# Decode-schedule variants: a capacity directory m<capacity>-<name> beside the
# default m<capacity>, exported with per-capacity options (bit-identical; the
# worker's --exl3-schedule NAME selects it). gb10 is the DGX Spark schedule of
# the GLM 5.3 Flash TP4 decode capacities (FR-G.7(b)), which the worker runs as
# m1 for one row and m80 for 2-80 rows: the b12x gb10 decode schedule (weight
# words staged L2 evict-first), and at m80 64x128 tiles at two CTAs per SM (the
# same K partition, so the same bits; it hides one CTA's tile start behind the
# other's stream). Measured on GB10, one TP4 rank slice, uniform top-8 routes,
# fastest call against the default export (b12x
# benchmarks/benchmark_glmf_decode_schedule.py, 5 rounds): 1 row (m1) 1.071x;
# 2 / 4 / 8 / 16 / 32 / 64 / 80 rows (m80) 1.112x / 1.057x / 1.032x / 1.031x /
# 1.026x / 1.020x / 1.014x. At m1 the narrow tile measured 1.053x, below 1.071x.
DECODE_SCHEDULES = {
    'gb10': {'geometry': 'glmf', 'profiles': ('tp4-',),
             'capacities': {1: {'decode_schedule': 'gb10'},
                            80: {'decode_schedule': 'gb10', 'tile': (64, 128, 64, 128),
                                 'blocks_per_sm': 2}}},
}


PURE_EVICT_FIRST = 'l2=2,pf1=0,pf2=0,pdl=1'


def default_decode_schedule(geometry: str, role: str, profile: str,
                            capacity: int, bits: list[int]) -> str | None:
    """Only measured cooperative profiles use the pure streaming hint.

    GB10, default/control/hint, three alternating rounds, rows 1..80: GLM
    compact TP6 1.054..1.123x (control .997..1.002); Flash TP4 1.027..1.082x
    (.997..1.001); Qwen compact TP4 1.003..1.094x (1.000..1.004); Kimi TP6
    1.020..1.107x (.991..1.004). Compact m16 rows 2/4/8/16: GLM
    1.067..1.119x (.995..1.003), Qwen 1.019..1.083x (.989..1.002).
    RTX TP2 m1/m80: Flash median 1.015..1.144x (control .995..1.016),
    GLM median 1.014..1.037x (.999..1.005), finite normalized fixtures.
    Tiles, occupancy and arithmetic unchanged.
    """
    measured = {
        ('glm', 'tp6-compact-width384', (4, 5)),
        ('glmf', 'tp4-width512', (3, 4)),
        ('qwen4', 'tp4-compact-width256', (4, 5)),
        ('kimik3', 'tp6-width512', (2, 3)),
    }
    identity = (geometry, profile, tuple(bits))
    measured_m16 = {('glm', 'tp6-compact-width384', (4, 5)),
                    ('qwen4', 'tp4-compact-width256', (4, 5))}
    if role == 'spark' and ((capacity in (1, 80) and identity in measured)
                            or (capacity == 16 and identity in measured_m16)):
        return PURE_EVICT_FIRST
    measured_rtx = {('glm', 'rtx-tp2', (4, 5)), ('glmf', 'rtx-tp2', (3, 4))}
    if role == 'coordinator' and capacity in (1, 80) and identity in measured_rtx:
        return PURE_EVICT_FIRST
    return None


def decode_schedule_variants(geometry: str, role: str, profile: str,
                             capacity: int) -> list[tuple[str, dict]]:
    """(name, export options) of every decode-schedule variant of one export."""
    if role != 'spark':
        return []
    return [(name, dict(spec['capacities'][capacity]))
            for name, spec in sorted(DECODE_SCHEDULES.items())
            if spec['geometry'] == geometry and profile.startswith(spec['profiles'])
            and capacity in spec['capacities']]


def package_name(geometry: str, bits: list[int]) -> str:
    """Package directory for one tier family; mirrors the daemon's resolver."""
    tag = ''.join(map(str, bits))
    return f'exl3-{geometry}-k{tag}'


def shard_profiles(geometry: str, role: str) -> list[tuple]:
    """Profiles of a non-V4.1 geometry: whole H128 blocks of the intermediate per
    rank (the first ranks own any extra block, as the Rust loader partitions),
    one export per distinct width shared by the ranks that carry it. TP6 of
    2048 (GLM 5.3, GLM 5.3 Flash): 3, 3, 3, 3, 2, 2 blocks -> tp6-width384
    for ranks 0-3 and tp6-width256 for ranks 4-5 (no padding: each rank runs
    the export of its own width)."""
    _hidden, intermediate, experts, topk = GEOMETRIES[geometry]
    blocks = intermediate // 128
    if role != 'spark':
        profiles = [('rtx-tp1', intermediate, experts, topk, 'fp32', ['rtx-tp1'])]
        if blocks % 2 == 0:
            profiles.append(('rtx-tp2', intermediate // 2, experts, topk, 'fp32', ['rtx-tp2']))
        else:
            # Unequal ranks preserve complete native H128 rotations, with no padding.
            for rank in range(2):
                width = (blocks // 2 + (rank < blocks % 2)) * 128
                name = f'rtx-tp2-rank{rank}'
                profiles.append((name, width, experts, topk, 'fp32', [name]))
        return profiles
    profiles = []
    # Every transport count with at least one whole rotation block per rank.
    # Qwen's five H128 blocks intentionally exclude TP6.
    worlds = range(1, min(8, blocks) + 1)
    for world in worlds:
        widths: dict[int, list[str]] = {}
        for rank in range(world):
            width = (blocks // world + (rank < blocks % world)) * 128
            widths.setdefault(width, []).append(f'tp{world}-rank{rank}')
        for width, destinations in sorted(widths.items(), reverse=True):
            profiles.append((f'tp{world}-width{width}', width, experts, topk, 'bf16', destinations))
    return profiles


def profiles_for_role(role: str, geometry: str = 'dsv4p') -> list[tuple]:
    """(profile, width, experts, top-k, dtype, [layout destinations]) per role.

    TP4 keeps the padded 640/512 pair-split because I=2304/4 is not a whole number
    of 128-wide Trellis blocks; TP2 (9 blocks/rank), TP3 (6) and TP6 (3)
    split exactly, each compiling one export per capacity shared by its ranks.
    Other geometries derive the same H128 split from their own intermediate.
    """
    return shard_profiles(geometry, role)


def build_profiles(role: str, geometry: str, *, all_spark_counts: bool = False) -> list[tuple]:
    profiles = profiles_for_role(role, geometry)
    if role != 'spark' or all_spark_counts:
        return profiles
    from spark_packages import package_worlds
    worlds = set(package_worlds(geometry, 'exl3'))
    return [entry for entry in profiles if int(entry[0].split('-')[0][2:]) in worlds]


def compact_profiles(geometry: str, profiles: list[tuple]) -> list[tuple]:
    """Own-maximum H128 pitch, shared by ranks of selected uneven worlds only."""
    _hidden, intermediate, experts, topk = GEOMETRIES[geometry]
    units = intermediate // 128
    worlds = sorted({int(entry[0].split('-')[0][2:]) for entry in profiles})
    return [(f'tp{world}-compact-width{((units + world - 1) // world) * 128}',
             ((units + world - 1) // world) * 128, experts, topk, 'bf16',
             [f'tp{world}-compact-rank{rank}' for rank in range(world)])
            for world in worlds if units % world]


def parse_requested_layouts(values: list[str], role: str, geometry: str = 'dsv4p',
                            companions: list[tuple] = ()) -> list[str]:
    """`--require-layout tp3-rank0,tp3-rank1` (repeatable): layouts the package needs.

    Scoped to the role being built: a coordinator cannot produce Spark ranks, so
    asking for one must fail during argument validation instead of after a full
    GPU compile.
    """
    requested: list[str] = []
    known = {layout for _, _, _, _, _, destinations in
             [*profiles_for_role(role, geometry), *companions] for layout in destinations}
    for value in values:
        for name in (part.strip() for part in value.split(',')):
            if not name:
                raise ValueError('EXL3 requested layout list has an empty entry')
            if name not in known:
                raise ValueError(f'unknown EXL3 layout requested: {name}')
            if name in requested:
                raise ValueError(f'duplicate EXL3 requested layout: {name}')
            requested.append(name)
    return sorted(requested)


def tile_overrides(values: list[str], capacities: list[int], role: str,
                   geometry: str = 'dsv4p') -> dict[str, dict[int, tuple]]:
    """Parse repeatable `--tile PROFILE=CAPACITIES:FC1_K,FC1_N,FC2_K,FC2_N`.

    Capacity scoping is the point: the pinned B12x policy already picks a wider
    tile only at m16 (verified against `_projection_mixed_tile_config`), so an A/B
    must be able to retarget one capacity without silently flattening the others.
    `CAPACITIES` is `all` or a `+`-joined list of already-selected capacities. The
    tile itself is b12x's own `(fc1_k, fc1_n, fc2_k, fc2_n)` vocabulary and is only
    shape-checked (four integers); the pinned planner decides whether a geometry is
    legal, so its rules are not restated here.
    """
    # Role-scoped for the same reason as --require-layout: an override naming a
    # profile this role does not build would otherwise be accepted and silently
    # ignored, which is a knob that ships looking effective.
    widths = {profile: width for profile, width, *_ in profiles_for_role(role, geometry)}
    result: dict[str, dict[int, tuple]] = {}
    for value in values:
        profile, sep, rest = value.partition('=')
        targets, sep2, tiles = rest.partition(':')
        if not sep or not sep2 or profile not in widths:
            raise ValueError(
                f'EXL3 tile override must name a {role} profile as '
                f'PROFILE=CAPACITIES:FC1_K,FC1_N,FC2_K,FC2_N, got: {value} '
                f'(available: {sorted(widths)})')
        if targets == 'all':
            selected = list(capacities)
        else:
            try:
                selected = sorted({int(part) for part in targets.split('+')})
            except ValueError:
                raise ValueError(f'EXL3 tile override capacities must be integers or all: {targets}') from None
            unknown = [c for c in selected if c not in capacities]
            if unknown:
                raise ValueError(
                    f'EXL3 tile override targets capacities that are not packaged: {unknown} '
                    f'(selected: {capacities})')
        try:
            tile = tuple(int(part) for part in tiles.split(','))
        except ValueError:
            raise ValueError(f'EXL3 tile override needs four integers: {tiles}') from None
        if len(tile) != 4 or any(part < 1 for part in tile):
            raise ValueError(f'EXL3 tile override needs four positive tiles: {tiles}')
        overlap = set(result.get(profile, {})) & set(selected)
        if overlap:
            raise ValueError(f'duplicate EXL3 tile override for {profile} at {sorted(overlap)}')
        per_capacity = result.setdefault(profile, {})
        per_capacity.update({capacity: tile for capacity in selected})
    return result


def init_variant_worker():
    """Once per long-lived worker: its own session (so a failure can kill the
    compilers it started) and the reservation its one CUDA context takes."""
    os.setsid()
    os.environ['CUTEAFD_EXPORT_WORKER_RAM_MIB'] = '4096'
    os.environ['CUTEAFD_EXPORT_WORKER_GPU_MIB'] = '2688'


def compile_variant(payload):
    args, target_compute, raw, width, experts, capacity, topk, dtype, run_options = payload
    import _pinned_sparkinfer
    from export_b12x_exl3_aot import export
    from _export_jobs import export_slot
    import torch
    with export_slot():
        real = torch.cuda.get_device_capability(0)
        if tuple(target_compute) == (12, 1) and real == (12, 0):
            from exl3_cross_sm121 import describe_gb10
            describe_gb10()
        if torch.cuda.get_device_capability(0) != tuple(target_compute):
            raise ValueError(f'EXL3 worker cannot target {target_compute}')
        meta = export(raw, width, experts, capacity, tuple(args.bits), 'auto', topk, dtype, **run_options)
        if meta['compute'] != list(target_compute):
            raise ValueError('EXL3 worker exported the wrong compute target')
        for label in ('v41_exl3_core', 'v41_exl3_sum'):
            verify_object_target(raw / f'{label}.o', target_compute)
        core = raw / 'libcuteafd_exl3.so'
        subprocess.run([args.cxx, '-shared', '-fPIC', '-std=c++17',
            f'-I{args.cuda_include}', str(raw / 'v41_exl3_bridge.cc'),
            str(raw / 'v41_exl3_core.o'), str(raw / 'v41_exl3_sum.o'),
            f'-L{args.cuda_libdir}', '-lcudart', f'-L{args.runtime.parent}',
            '-lcute_dsl_runtime', '-Wl,-z,defs',
            '-o', str(core)], check=True)
        runtime_files = ['v41_exl3.json', 'trellis_lut.bin', 'libcuteafd_exl3.so']
        if meta['requires_route_preparation']:
            routes = raw / 'routes'
            subprocess.run([args.cxx, '-shared', '-fPIC', '-std=c++17',
                f'-I{args.cuda_include}', str(routes / 'v41_exl3_routes.cc'),
                str(args.cuda_driver), '-Wl,-z,defs',
                '-o', str(routes / 'libv41_exl3_routes.so')], check=True)
            runtime_files += ['routes/v41_exl3_routes.json', 'routes/libv41_exl3_routes.so']
        for name in runtime_files:
            if name.endswith('.so'):
                verify_object_target(raw / name, target_compute)
        gc.collect()
        torch.cuda.empty_cache()
    return meta, runtime_files


def compile_variants(payloads, jobs):
    """Observe failures in completion order, assemble successes in input order."""
    # Workers live for the whole package and reuse one CUDA context across
    # variants: a context per variant meant 24-30 create/destroy cycles per
    # package, and GB10 context creation fails (NV_ERR_NO_MEMORY) under the
    # build's own load. Each variant still frees its tensors and cache.
    pool = ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context('spawn'),
                               initializer=init_variant_worker)
    try:
        futures = {pool.submit(compile_variant, payload): index for index, payload in enumerate(payloads)}
        results = [None] * len(payloads)
        for future in as_completed(futures):
            results[futures[future]] = future.result()
        return results
    except BaseException:
        # Python 3.12 has no terminate_workers(). Join before releasing outputs
        # and reservations; cancelling futures does not stop running compilers.
        processes = list((pool._processes or {}).values())
        groups = []
        for process in processes:
            try:
                if os.getpgid(process.pid) == process.pid:
                    groups.append(process.pid)
                    os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            process.terminate()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        # A compiler can outlive a terminated worker. Kill its owned session
        # even when the group's original leader has already exited.
        for group in groups:
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def package_capacities(geometry: str, requested: str) -> list[int]:
    capacities = sorted(set(int(v) for v in requested.split(',')))
    # Kimi K3 pads a <=2752-row prefill unit into merged waves of up to three
    # units, so its packed extents reach 8256; every other family stops at 4096.
    limit = 8256 if geometry == 'kimik3' else 4096
    if not capacities or any(v < 1 or v > limit for v in capacities):
        raise ValueError(f'EXL3 capacities must be in 1..{limit}')
    if geometry == 'kimik3':
        # m16 takes K128 tiles that drift against M1 decode; rows 2..16 run on
        # m80 (like GLM Flash). Every other capacity is allowed.
        capacities = [v for v in capacities if v != 16]
        if not capacities:
            raise ValueError('Kimi EXL3 needs a capacity other than 16')
    return capacities


def build(args: argparse.Namespace) -> None:
    if len(args.bits) != 2:
        raise ValueError('EXL3 packages require two decoder tiers')
    validate_destination(args.output)
    geometry = getattr(args, 'geometry', 'dsv4p')
    hidden = GEOMETRIES[geometry][0]
    capacities = package_capacities(geometry, args.capacities)
    all_counts = getattr(args, 'all_spark_counts', False)
    if all_counts and args.role != 'spark':
        raise ValueError('--all-spark-counts requires a Spark role')
    only = set(getattr(args, 'profile', None) or [])
    profiles = build_profiles(args.role, geometry, all_spark_counts=all_counts or bool(only))
    if only:
        unknown = only - {profile for profile, *_ in profiles}
        if unknown:
            raise ValueError(f'unknown EXL3 profiles for {geometry} {args.role}: {sorted(unknown)}')
        profiles = [entry for entry in profiles if entry[0] in only]
    companions = []
    if getattr(args, 'compact_companions', False):
        if args.role != 'spark' or len(args.bits) != 2:
            raise ValueError('--compact-companions requires two-tier Spark exports')
        companions = compact_profiles(geometry, profiles)
    requested_layouts = parse_requested_layouts(
        getattr(args, 'require_layout', []), args.role, geometry, companions)
    requested_layouts = sorted(set(requested_layouts) | {
        destination for *_, destinations in companions for destination in destinations})
    profiles += companions
    tiles = tile_overrides(getattr(args, 'tile', []), capacities, args.role, geometry)
    # Import the source-pinned compiler only for builds, never package checks.
    import _pinned_sparkinfer
    import torch

    # The planner needs only a context, not a worker's sizing arena.
    os.environ['CUTEAFD_EXPORT_WORKER_RAM_MIB'] = '1024'
    props = torch.cuda.get_device_properties(0)
    # --loopback: Spark layouts compiled for this coordinator GPU, for worker
    # tests on raptor; the manifest records compute 12.0, so no Spark takes it.
    expected_compute = (12, 1) if args.role == 'spark' and not getattr(args, 'loopback', False) else (12, 0)
    if (props.major, props.minor) != expected_compute:
        raise ValueError(f'{args.role} package requires GPU {expected_compute}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.exl3-package-', dir=args.output.parent) as temporary:
        stage = Path(temporary)
        variants = []
        pending = []
        for profile, width, experts, topk, dtype, destinations in profiles:
            compact = '-compact-' in profile
            for capacity in capacities:
                requested = tiles.get(profile, {}).get(capacity)
                tile = (64, 128, 64, 128) if compact else requested
                # Content-keyed: an override changes the compiled geometry, so it
                # must never land in (or be read from) the policy-keyed export, and
                # neither may a pinned capacity: a build tree that predates the pin
                # holds the widened export under the policy key. Without either the
                # path is what every existing build used.
                profile_dir = (args.build_dir / profile if tile is None else
                               args.build_dir / f"{profile}+tile{'-'.join(map(str, tile))}")
                compact_ws = compact and len(args.bits) == 2 and compact_ws_export(
                    geometry, args.role, width, capacity, args.bits)
                compact_block = (compact_ws_route_block(geometry, args.role, width, capacity)
                                 if compact_ws else None)
                if compact_block is not None:
                    # N128 alone also names the cooperative export; separate its cache.
                    dynamic = ws_dynamic_tiles(geometry, args.role, width, capacity, compact=True)
                    profile_dir = profile_dir.with_name(
                        f'{profile_dir.name}+ws-m{compact_block}-{"dynamic" if dynamic else "static"}')
                raw = profile_dir / f'm{capacity}'  # one capacity, one directory
                options = {'compact_weights': True} if compact else {}
                if activation(geometry) != 'silu':
                    options['activation'] = activation(geometry)
                if tile is not None:
                    options['tile'] = tile
                options['hidden'] = hidden
                block = route_block(geometry, capacity, args.role)
                # Coordinator Kimi retains its independently qualified K2/K3 recipe.
                ws = compact_ws or (not compact and len(args.bits) == 2
                                    and warp_specialized(geometry, args.role, width, capacity)
                                    and (geometry != 'kimik3' or args.role != 'coordinator'
                                         or tuple(args.bits) == (2, 3)))
                if ws:
                    block = (compact_block if compact_block is not None
                             else ws_route_block(geometry, args.role, width, capacity))
                    if not compact:
                        # Existing policy-keyed directories contain cooperative exports.
                        # The input format and claim mode are part of the compiled
                        # program, so they name its cache directory.
                        if wire_input(geometry, args.role, width, capacity):
                            suffix = '-wire-s3-dynamic'
                        elif args.role == 'coordinator':
                            dynamic = ws_dynamic_tiles(geometry, args.role, width, capacity)
                            suffix = '-bf16-dynamic' if dynamic else '-bf16-static'
                        else:
                            suffix = ''
                        profile_dir = profile_dir.with_name(f'{profile_dir.name}+ws-m{block}{suffix}')
                        raw = profile_dir / f'm{capacity}'
                if block != 8:
                    options['route_block'] = block
                if ws:
                    options['warp_specialized'] = True
                    if wire_input(geometry, args.role, width, capacity, compact=compact_ws):
                        options['wire_input'] = True
                    if (stages := ws_input_stages(geometry, args.role, width, capacity,
                                                 compact=compact_ws)) is not None:
                        options['ws_input_stages'] = stages
                    if ws_dynamic_tiles(geometry, args.role, width, capacity, compact=compact_ws):
                        options['ws_dynamic_tiles'] = True
                    if (policy_tile := ws_tile(geometry, args.role, width, capacity,
                                              compact=compact_ws)) is not None and requested is None:
                        options['tile'] = policy_tile
                        if compact_ws and policy_tile[1] == 192:
                            profile_dir = profile_dir.with_name(f'{profile_dir.name}-n192')
                            raw = profile_dir / f'm{capacity}'
                elif len(args.bits) == 2 and token_major_rotation(geometry, capacity, args.role):
                    options['token_major_rotation'] = True
                if (limit := swiglu_limit(geometry)) != 10.0:
                    options['swiglu_limit'] = limit
                default_options = options
                if (schedule_hint := default_decode_schedule(
                        geometry, args.role, profile, capacity, args.bits)) is not None:
                    raw = profile_dir.with_name(f'{profile_dir.name}+{schedule_hint}') / raw.name
                    default_options = {**options, 'decode_schedule': schedule_hint}
                # Explicit schedule siblings keep their original options and suffix.
                exports = [(None, raw, default_options)]
                for schedule, schedule_options in ([] if compact else decode_schedule_variants(
                        geometry, args.role, profile, capacity)):
                    exports.append((schedule, profile_dir.with_name(f'{profile_dir.name}+{schedule}')
                                    / raw.name, {**options, **schedule_options}))
                for schedule, raw, run_options in exports:
                    pending.append((raw, width, experts, capacity, topk, dtype, run_options,
                                    destinations, schedule, requested))
        from _export_jobs import export_jobs, gpu_workers, park_context_lease
        jobs = min(gpu_workers(export_jobs()), len(pending))
        # The planner's context idles while its workers run; free its host lease.
        park_context_lease()
        print(f'EXL3 variants: {len(pending)} independent exports, {jobs} workers', flush=True)
        # World and rank destinations do not determine generated code. Keep all
        # arithmetic/schedule options in the key, including compact descriptor ABI.
        unique = {}
        keys = []
        for item in pending:
            key = (*item[1:6], json.dumps(item[6], sort_keys=True))
            keys.append(key)
            unique.setdefault(key, item)
        # Stable width/options cache identity, independent of the first world
        # that happened to request this variant. ABI and schedule are in options.
        for key, item in list(unique.items()):
            identity = json.dumps([geometry, args.role, args.bits, expected_compute, key], sort_keys=True)
            raw = args.build_dir / f'width{item[1]}-{hashlib.sha256(identity.encode()).hexdigest()[:20]}' / f'm{item[3]}'
            unique[key] = (raw, *item[1:])
        payloads = [(args, expected_compute, *item[:7]) for item in unique.values()]
        compiled = dict(zip(unique, compile_variants(payloads, jobs)))
        for item, key in zip(pending, keys):
            meta, runtime_files = compiled[key]
            source = unique[key][0]
            raw, width, experts, capacity, topk, dtype, run_options, destinations, schedule, requested = item
            for destination in destinations:
                directory = f'{destination}/m{capacity}' + ('' if schedule is None else f'-{schedule}')
                for name in runtime_files:
                    target = stage / directory / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source / name, target)
                variant = {'directory': directory, **{key: meta[key] for key in
                    ('capacity', 'intermediate', 'experts', 'top_k', 'output_dtype', 'bits', 'blocks_per_sm')}}
                for key in ('descriptor_rows', 'native_info_version'):
                    if key in meta:
                        variant[key] = meta[key]
                if 'tile' in meta:
                    # What the compiler actually resolved, always: the pinned
                    # policy varies per capacity (m16 is the known special case),
                    # so a build without an override still has a real tile.
                    variant['tile'] = meta['tile']
                if requested is not None and schedule is None:
                    # An A/B tile override applies to the default export; a
                    # schedule variant brings its own options. A capacity the
                    # packaging pins itself is not a request: the field records
                    # what an operator asked for, so that a package can be told
                    # apart from one whose policy resolved the tile.
                    variant['tile_requested'] = list(requested)
                if 'route_block' in meta:
                    variant['route_block'] = meta['route_block']
                if meta.get('token_major_rotation'):
                    variant['token_major_rotation'] = True
                if meta.get('warp_specialized'):
                    variant['warp_specialized'] = True
                if meta.get('ws_dynamic_tiles'):
                    variant['ws_dynamic_tiles'] = True
                if meta.get('ws_input_stages') is not None:
                    variant['ws_input_stages'] = meta['ws_input_stages']
                if 'input_format' in meta:
                    variant['input_format'] = meta['input_format']
                if schedule is not None:
                    variant['schedule'] = schedule
                if meta.get('decode_schedule') is not None:
                    variant['decode_schedule'] = meta['decode_schedule']
                variants.append(variant)
        files = {str(p.relative_to(stage)): {'bytes': p.stat().st_size, 'sha256': digest(p)}
                 for p in sorted(stage.rglob('*')) if p.is_file()}
        manifest = {'schema': 'cuteafd.exl3-package.v1', 'role': args.role,
                    'sparkinfer_revision': _pinned_sparkinfer.REVISION,
                    'compute': [props.major, props.minor], 'sms': props.multi_processor_count,
                    'variants': variants, 'files': files,
                    'runtime': {'library': 'libcute_dsl_runtime.so', 'sha256': digest(args.runtime),
                                'provider': 'installed nvidia-cutlass-dsl CUDA runtime; release entrypoint sets its library path'}}
        manifest['geometry'] = geometry
        if requested_layouts:
            present = {v['directory'].split('/')[0] for v in variants}
            missing = sorted(set(requested_layouts) - present)
            if missing:
                raise ValueError(
                    f'EXL3 package did not produce requested layouts: {missing}')
            manifest['requested_layouts'] = requested_layouts
            manifest['requested_capacities'] = capacities
        # Per-variant `tile` already records what each capacity compiled with; the
        # B12x policy legitimately picks a wider tile only at m16, so a package may
        # mix tile values across capacities and that is not an error.
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        verify(stage, _pinned_sparkinfer.REVISION)
        install_package(stage, args.output)
    print(json.dumps({'package': str(args.output), 'role': args.role, 'variants': len(variants)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('build')
    create.add_argument('--role', choices=('spark', 'coordinator'), required=True)
    create.add_argument('--geometry', choices=sorted(GEOMETRIES), default='dsv4p',
                        help='Routed-expert geometry (cuteafd_core::ExpertGeometry::family)')
    create.add_argument('--all-spark-counts', action='store_true',
                        help='Opt in to all nonempty H128 Spark TP profiles (1..8)')
    create.add_argument('--compact-companions', action='store_true',
                        help='Add compact two-tier exports for selected uneven Spark worlds only')
    create.add_argument('--profile', action='append', default=[],
                        help='Build only these profiles (repeatable), for bring-up; '
                             'without selection the family role profiles remain the default')
    create.add_argument('--require-layout', action='append', default=[],
                        help='Layouts this package must contain (comma list, repeatable). '
                             'Recorded in the manifest and re-checked on verify, so a '
                             'partial build cannot be published as a full contract.')
    create.add_argument('--tile', action='append', default=[],
                        help='Opt-in tile override PROFILE=CAPACITIES:FC1_K,FC1_N,FC2_K,FC2_N '
                             '(CAPACITIES is all or 16+80) for a controlled A/B, for example '
                             'tp3-width768=16:64,256,64,256; default is the B12x per-capacity policy')
    create.add_argument('--loopback', action='store_true',
                        help='Build Spark-role layouts for the SM120 coordinator GPU (loopback worker tests)')
    create.add_argument('--capacities', default='1,16,80,256,1024,2048,2752,4096')
    create.add_argument('--bits', type=int, nargs='+', default=[3, 4])
    create.add_argument('--build-dir', type=Path, required=True)
    create.add_argument('--output', type=Path, required=True)
    create.add_argument('--cxx', required=True)
    create.add_argument('--cuda-include', type=Path, required=True)
    create.add_argument('--cuda-libdir', type=Path, required=True)
    create.add_argument('--cuda-driver', type=Path, required=True)
    create.add_argument('--runtime', type=Path, required=True)
    install = commands.add_parser('install')
    install.add_argument('--package', type=Path, required=True)
    install.add_argument('--output', type=Path, required=True)
    check = commands.add_parser('verify')
    check.add_argument('--package', type=Path, required=True)
    check.add_argument('--sparkinfer-revision')
    check.add_argument('--runtime', type=Path)
    check.add_argument('--role', choices=('spark', 'expert', 'coordinator'))
    args = parser.parse_args()
    if args.command == 'build':
        build(args)
    elif args.command == 'install':
        install_package(args.package, args.output)
    else:
        manifests = verify_root(args.package, args.sparkinfer_revision, args.runtime, args.role)
        print(json.dumps({
            'verified': True,
            'packages': [
                {'role': manifest['role'], 'variants': len(manifest['variants'])}
                for manifest in manifests
            ],
        }))


if __name__ == '__main__':
    main()

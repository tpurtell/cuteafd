#!/usr/bin/env python3
"""Build relocatable native EXL3 packages; verify them without importing CUDA."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1] / _d) for _d in ("lib",)]  # sibling tool dirs

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def residency_overrides(values: list[str], capacities: list[int], paired: bool) -> dict[int, int]:
    """Explicit offline build choices; B12X validates kernel resource limits."""
    if values and not paired:
        raise ValueError('residency overrides require a paired TP4 package')
    result = {}
    for value in values:
        try:
            capacity, blocks = map(int, value.split('='))
        except (ValueError, AttributeError):
            raise ValueError('residency override must be CAPACITY=BLOCKS') from None
        if capacity not in capacities or blocks not in (1, 2):
            raise ValueError('residency override requires a selected capacity and one or two blocks/SM')
        if capacity in result:
            raise ValueError('duplicate residency override capacity')
        result[capacity] = blocks
    return result


def verify(package: Path, revision: str | None = None, runtime: Path | None = None, role: str | None = None) -> dict:
    manifest = json.loads((package / 'manifest.json').read_text())
    if manifest['schema'] != 'cuteafd.exl3-package.v1':
        raise ValueError('unsupported EXL3 package schema')
    if revision is not None and manifest['sparkinfer_revision'] != revision:
        raise ValueError('EXL3 package/source revision mismatch')
    if role is not None and manifest['role'] != ('spark' if role == 'expert' else role):
        raise ValueError('EXL3 package/serving role mismatch')
    paired = manifest.get('paired_tp4', False)
    if not isinstance(paired, bool) or (paired and manifest['role'] != 'spark'):
        raise ValueError('paired EXL3 package requires Spark role')
    overrides = residency_overrides(manifest.get('residency_overrides', []),
                                    [v['capacity'] for v in manifest['variants']], paired)
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
        # Records without a hidden size predate other geometries: they are V4.1.
        if meta.get('hidden', GEOMETRIES['v41'][0]) != GEOMETRIES[manifest.get('geometry', 'v41')][0]:
            raise ValueError(f'EXL3 variant hidden size does not match the package geometry: {directory}')
        for key in ('capacity', 'intermediate', 'experts', 'top_k', 'output_dtype', 'bits'):
            if meta[key] != variant[key]:
                raise ValueError(f'EXL3 variant metadata mismatch: {directory}/{key}')
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
        if variant.get('fused_input_rotation', False) != meta.get('fused_input_rotation', False):
            raise ValueError(f'EXL3 variant fused input rotation mismatch: {directory}')
        if variant.get('warp_specialized', False) != meta.get('warp_specialized', False):
            raise ValueError(f'EXL3 variant warp specialization mismatch: {directory}')
        if variant.get('input_format', 'bf16') != meta.get('input_format', 'bf16'):
            raise ValueError(f'EXL3 variant input format mismatch: {directory}')
        # A schedule variant lives in m<capacity>-<schedule> and its export
        # records the options; a default directory records none.
        schedule = variant.get('schedule')
        if (variant.get('decode_schedule') != meta.get('decode_schedule')
                or (schedule is None) != (meta.get('decode_schedule') is None)
                or directory.rsplit('/', 1)[-1] != f"m{meta['capacity']}"
                + ('' if schedule is None else f'-{schedule}')):
            raise ValueError(f'EXL3 variant decode schedule mismatch: {directory}')
        if variant.get('token_major_rotation', False) != meta.get('token_major_rotation', False):
            raise ValueError(f'EXL3 variant input rotation mismatch: {directory}')
        if meta['capacity'] in overrides and meta.get('blocks_per_sm') != overrides[meta['capacity']]:
            raise ValueError('EXL3 compiled residency differs from requested override')
        boundary = meta.get('paired_boundary')
        if paired:
            rank_name = directory.split('/')[0]
            boundaries = {'tp4-rank0': 'last', 'tp4-rank1': 'first', 'tp4-rank2': 'last', 'tp4-rank3': 'first'}
            if (boundary != boundaries.get(rank_name) or boundary is None
                    or variant.get('paired_boundary') != boundary
                    or meta.get('descriptor_rows') != 4 or meta.get('native_info_version') != 3
                    or meta['intermediate'] != 640 or len(meta['bits']) != 2 or meta['top_k'] != 6):
                raise ValueError('paired EXL3 package boundary/contract mismatch')
        elif boundary is not None or variant.get('paired_boundary') is not None:
            raise ValueError('paired artifact in disjoint EXL3 package')
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
    'v41': (5120, 2304, 384, 6),
    'dsv4f': (4096, 2048, 256, 6),
    'dsv4p': (7168, 3072, 384, 6),
    'glm': (6144, 2048, 256, 8),
    # GLM 5.3 Flash (glm5_next): 288 experts, SwiGLU clamped at 10.
    'glmf': (4096, 2048, 288, 8),
    # Qwen 3.8 Flash Next (qwen4_exp): 512 experts, softmax top-10, unclamped SiLU.
    'qwen4': (2560, 640, 512, 10),
}
# SwiGLU clamp per geometry; None is the unclamped SwiGLU (b12x const-expr
# elides the clamp). DeepSeek clamps at 10.
SWIGLU_LIMITS = {'glm': None, 'qwen4': None}


def swiglu_limit(geometry: str) -> float | None:
    return SWIGLU_LIMITS.get(geometry, 10.0)


def route_block(geometry: str, capacity: int) -> int:
    """Packed-route M block per capacity: rows of one expert that share a Trellis decode.

    V4.1 keeps its qualified 8-row blocks everywhere. DeepSeek V4 Pro Spark prefill
    (H7168, 384 experts, top-6: ~64 rows per expert at 4096 rows) re-decodes every
    weight tile once per 8 rows, so wide prefill capacities use wider blocks
    (GB10, TP4 width 768, random top-6 routes: m4096 53 -> 25 ms, m1024 15 -> 11 ms;
    m81..256 stays fastest at 8). At m4096, 64-row blocks skip their empty M16
    fragments (b12x 4d7cb455), so they beat 32 from about 2048 rows up. The coordinator's whole-intermediate rtx-tp1 package
    gains the same way (RTX PRO 6000, width 3072: m4096 74 -> 35 ms, m1024 23 -> 16 ms).
    """
    # Build-time A/B knob: CUTEAFD_EXL3_ROUTE_BLOCKS=CAPACITY=BLOCK[,...].
    override = dict(item.split('=') for item in os.environ.get('CUTEAFD_EXL3_ROUTE_BLOCKS', '').split(',') if item)
    if str(capacity) in override:
        return int(override[str(capacity)])
    if geometry == 'qwen4':
        return qwen4_route_block(capacity)
    if geometry in ('glm', 'glmf'):
        return glm_route_block(capacity)
    if geometry != 'dsv4p' or capacity <= 256:
        return 8
    return 16 if capacity <= 1024 else 64


def glm_route_block(capacity: int) -> int:
    """GLM (256 experts, top-8): 8 * capacity routes, capacity / 32 rows per
    expert on average. Measured on the coordinator rtx-tp1 package (RTX PRO
    6000, width 2048, random top-8 routes, us): m80 8/16/32 = 2885/2905/2980;
    m256 8/16/32/64 = 3265/3205/3260/3325; m1024 16/32/64 = 5650/4530/3910;
    m4096 32/64 = 13820/11790 (64 with token-major rotation 11470). m256
    keeps 8 rows (16 is 1.9% faster on the RTX, but its Spark TP3 width-640
    kernel resolves a residency the native bridge does not take).
    """
    return 8 if capacity <= 256 else 64


def qwen4_route_block(capacity: int) -> int:
    """Qwen 3.8 Flash Next (512 experts, top-10): 10 * capacity routes, capacity / 51
    rows per expert on average. Coordinator rtx-tp1 package (RTX PRO 6000, width
    640, random top-10 routes, layer 23, median us): 8-row blocks m1024 2315,
    m4096 11626; 64: m1024 1401, m4096 3489; 32: m256 1002 (8: 1023), m1024 1168,
    m4096 3273 (min 3176 vs 3451). m256 with 16 fails the cooperative grid check.
    """
    return 8 if capacity <= 80 else 32


def token_major_rotation(geometry: str, capacity: int) -> bool:
    """Whether a capacity rotates each token's input once for all of its routes.

    Bit-identical to per-route rotation. It pays where the rotation phase is
    large (V4 Pro m2048..4096 prefill); at m1024 and below it is neutral.
    """
    if geometry in ('glm', 'glmf', 'qwen4'):
        # GLM top-8 (rtx-tp1): m1024 3910 -> 3830 us, m4096 11790 -> 11470 us.
        return capacity > 256
    return geometry == 'dsv4p' and capacity > 1024


def fused_input_rotation(geometry: str, role: str, width: int, capacity: int) -> bool:
    """Whether FC1 rotates staged token rows in shared memory (no per-route
    rotated copies). Bit-identical; every FC1 N tile of a block rotates it
    again, so it pays only while few 256-wide tiles share a projection half.
    GB10, TP4 random routes, per layer (b12x ccf9e3cd): V4 Pro width 512
    m4096 19.1 -> 16.5 ms, 768 22.7 -> 21.9, 1024 neutral, 1536 +10%;
    GLM width 512 m1024 8.9 -> 7.3, m4096 22.6 -> 18.3, 768 -11%, 1024 -6%,
    640 (128-wide tiles) +16%. V4 Pro m1024 (16-row blocks) is neutral.
    GLM 5.3 Flash (b12x be78fa4c, m1024 / m2048 / m4096 ms): width 512
    5.41 -> 4.49, 7.93 -> 6.60, 13.77 -> 11.22; 768 7.01 -> 6.60,
    9.93 -> 8.92, 16.42 -> 15.71; 1024 8.47 -> 8.27, 11.74 -> 11.41,
    19.36 -> 19.64, so width 1024 keeps its m4096 package on token-major.
    The coordinator's full-width packages are not measured and stay off.
    """
    if role != 'spark' or width % 256:
        return False
    if geometry == 'dsv4p':
        return width <= 768 and capacity > 1024
    if geometry == 'glm':
        return width <= 1024 and capacity > 256
    if geometry == 'glmf':
        return capacity > 256 and (width <= 768 or (width == 1024 and capacity <= 1024))
    return False


def warp_specialized(geometry: str, role: str, width: int, capacity: int) -> bool:
    """Whether a capacity exports the warp-specialized prefill kernels (b12x
    mixed_trellis_ws, 64-row route blocks): producer warps stream the Trellis
    weights and input rows and rotate FC1 inputs one block ahead, consumer
    warps decode and multiply. Bit-identical to the cooperative kernel.
    Faster than the cooperative exports at every Spark width and prefill
    capacity measured (GB10, random routes, kernel + top-k sum, median of
    three interleaved runs, ms): V4 Pro TP4 768 m1024 11.16 -> 8.36, m4096
    21.98 -> 17.27; GLM 5.3 TP4 512 m1024 7.23 -> 6.67, m4096 18.24 ->
    14.60; GLM 5.3 Flash TP4 512 m4096 11.03 -> 9.07, TP2 1024 m4096 19.32
    -> 15.47; single runs 3-25% faster for TP2/TP3/TP6 widths (tile-128
    GLM TP3 640 included) and at m128/m256. Decode capacities (<= 80 rows)
    keep the cooperative kernel.
    """
    if role != 'spark' or geometry not in ('dsv4p', 'glm', 'glmf'):
        return False
    return capacity >= 256


# Warp-specialized tile (fc1_k, fc1_n, fc2_k, fc2_n) of widths that are odd
# multiples of 128 and multiples of 192 (GLM 5.3 TP6 width 384).
TILE_384 = (64, 192, 64, 256)


def wire_input(geometry: str, role: str, width: int, capacity: int) -> bool:
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
    """
    return geometry == 'glm' and warp_specialized(geometry, role, width, capacity)


def ws_input_stages(geometry: str, role: str, width: int, capacity: int) -> int | None:
    """Input-row ring depth of the Spark warp-specialized exports: GB10 gathers
    the routed rows from LPDDR5X at high latency, and four stages (the ring the
    weight ring of 256-wide 4-bit tiles leaves room for) hide it; with K4/K5
    tiers three stages measure faster than four (see `wire_input`). Bit-identical.
    """
    return 3 if geometry == 'glm' and warp_specialized(geometry, role, width, capacity) else None


def ws_dynamic_tiles(geometry: str, role: str, width: int, capacity: int) -> bool:
    """Whether Spark warp-specialized exports claim tiles dynamically. With the
    static round-robin an expert's route blocks start on different CTAs at
    different times and each streams the expert's weights from LPDDR5X (ncu,
    width 512 m2752: FC1 fills 1761 MB of LPDDR5X for 805 MB of weights).
    Bit-identical; the largest of the GLM 5.3 GB10 gains (see `wire_input`).
    """
    return geometry == 'glm' and warp_specialized(geometry, role, width, capacity)


def ws_tile(geometry: str, role: str, width: int, capacity: int) -> tuple[int, ...] | None:
    """Warp-specialized tile for widths that are odd multiples of 128, which
    the B12x policy gives 128-wide tiles on both stages (four consumer warps).
    Bit-identical (the N tiling does not change any accumulation). Width 384
    (GLM 5.3 TP6 ranks 0-3) takes 192-wide FC1 tiles: m4096 14.89 -> 12.8 ms.
    """
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
    return f'exl3-k{tag}' if geometry == 'v41' else f'exl3-{geometry}-k{tag}'


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
        # Dual-RTX halves only where they are whole H128 blocks (not Qwen's 640).
        if blocks % 2 == 0:
            profiles.append(('rtx-tp2', intermediate // 2, experts, topk, 'fp32', ['rtx-tp2']))
        return profiles
    profiles = []
    # Every transport count with at least one whole rotation block per rank.
    # Qwen's five H128 blocks intentionally exclude TP6/7/8.
    worlds = range(1, min(8, blocks) + 1)
    for world in worlds:
        widths: dict[int, list[str]] = {}
        for rank in range(world):
            width = (blocks // world + (rank < blocks % world)) * 128
            widths.setdefault(width, []).append(f'tp{world}-rank{rank}')
        for width, destinations in sorted(widths.items(), reverse=True):
            profiles.append((f'tp{world}-width{width}', width, experts, topk, 'bf16', destinations))
    return profiles


def profiles_for_role(role: str, geometry: str = 'v41') -> list[tuple]:
    """(profile, width, experts, top-k, dtype, [layout destinations]) per role.

    TP4 keeps the padded 640/512 pair-split because I=2304/4 is not a whole number
    of 128-wide Trellis blocks; TP2 (9 blocks/rank), TP3 (6) and TP6 (3)
    split exactly, each compiling one export per capacity shared by its ranks.
    Other geometries derive the same H128 split from their own intermediate.
    """
    if geometry != 'v41':
        return shard_profiles(geometry, role)
    if role == 'spark':
        return [('tp4-width640', 640, 384, 6, 'bf16', ['tp4-rank0', 'tp4-rank1']),
                ('tp4-width512', 512, 384, 6, 'bf16', ['tp4-rank2', 'tp4-rank3']),
                ('tp2-width1152', 1152, 384, 6, 'bf16', ['tp2-rank0', 'tp2-rank1']),
                ('tp3-width768', 768, 384, 6, 'bf16',
                 ['tp3-rank0', 'tp3-rank1', 'tp3-rank2']),
                ('tp6-width384', 384, 384, 6, 'bf16',
                 [f'tp6-rank{rank}' for rank in range(6)])]
    return [('rtx-tp1', 2304, 384, 6, 'fp32', ['rtx-tp1']),
            ('rtx-tp2', 1152, 384, 6, 'fp32', ['rtx-tp2']),
            ('dspark', 2304, 128, 3, 'bf16', ['dspark'])]


def build_profiles(role: str, geometry: str, *, all_spark_counts: bool = False) -> list[tuple]:
    profiles = profiles_for_role(role, geometry)
    if all_spark_counts or role != 'spark' or geometry == 'v41':
        return profiles
    worlds = {2, 3, 4, 6} if GEOMETRIES[geometry][1] // 128 >= 6 else {2, 3, 4}
    if geometry == 'qwen4':
        worlds.add(1)
    return [profile for profile in profiles if int(profile[0].split('-')[0][2:]) in worlds]


def parse_requested_layouts(values: list[str], role: str, geometry: str = 'v41') -> list[str]:
    """`--require-layout tp3-rank0,tp3-rank1` (repeatable): layouts the package needs.

    Scoped to the role being built: a coordinator cannot produce Spark ranks, so
    asking for one must fail during argument validation instead of after a full
    GPU compile.
    """
    requested: list[str] = []
    known = {layout for _, _, _, _, _, destinations in profiles_for_role(role, geometry)
             for layout in destinations}
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


def tile_overrides(values: list[str], capacities: list[int], role: str, paired: bool,
                   geometry: str = 'v41') -> dict[str, dict[int, tuple]]:
    """Parse repeatable `--tile PROFILE=CAPACITIES:FC1_K,FC1_N,FC2_K,FC2_N`.

    Capacity scoping is the point: the pinned B12x policy already picks a wider
    tile only at m16 (verified against `_projection_mixed_tile_config`), so an A/B
    must be able to retarget one capacity without silently flattening the others.
    `CAPACITIES` is `all` or a `+`-joined list of already-selected capacities. The
    tile itself is b12x's own `(fc1_k, fc1_n, fc2_k, fc2_n)` vocabulary and is only
    shape-checked (four integers); the pinned planner decides whether a geometry is
    legal, so its rules are not restated here.
    """
    if values and paired:
        raise ValueError('EXL3 tile overrides are not supported for paired TP4 packages')
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


def build(args: argparse.Namespace) -> None:
    validate_destination(args.output)
    paired = getattr(args, 'paired_tp4', False)
    geometry = getattr(args, 'geometry', 'v41')
    hidden = GEOMETRIES[geometry][0]
    if paired and (args.role != 'spark' or len(args.bits) != 2 or geometry != 'v41'):
        raise ValueError('paired TP4 package requires V4.1 Spark role and two tiers')
    capacities = sorted(set(int(v) for v in args.capacities.split(',')))
    if not capacities or any(v < 1 or v > 4096 for v in capacities):
        raise ValueError('EXL3 capacities must be in 1..4096')
    overrides = residency_overrides(getattr(args, 'residency', []), capacities, paired)
    all_counts = getattr(args, 'all_spark_counts', False)
    if all_counts and (args.role != 'spark' or paired):
        raise ValueError('--all-spark-counts requires an unpaired Spark role')
    only = set(getattr(args, 'profile', None) or [])
    profiles = build_profiles(args.role, geometry, all_spark_counts=all_counts or bool(only))
    if only:
        unknown = only - {profile for profile, *_ in profiles}
        if unknown:
            raise ValueError(f'unknown EXL3 profiles for {geometry} {args.role}: {sorted(unknown)}')
        profiles = [entry for entry in profiles if entry[0] in only]
    requested_layouts = parse_requested_layouts(getattr(args, 'require_layout', []), args.role, geometry)
    tiles = tile_overrides(getattr(args, 'tile', []), capacities, args.role, paired, geometry)
    # Import the source-pinned compiler only for builds, never package checks.
    import _pinned_sparkinfer
    from export_b12x_exl3_aot import export
    import torch

    props = torch.cuda.get_device_properties(0)
    # --loopback: Spark layouts compiled for this coordinator GPU, for worker
    # tests on raptor; the manifest records compute 12.0, so no Spark takes it.
    expected_compute = (12, 1) if args.role == 'spark' and not getattr(args, 'loopback', False) else (12, 0)
    if (props.major, props.minor) != expected_compute:
        raise ValueError(f'{args.role} package requires GPU {expected_compute}')
    if paired:
        profiles = [('paired-last', 640, 384, 6, 'bf16', ['tp4-rank0', 'tp4-rank2']),
                    ('paired-first', 640, 384, 6, 'bf16', ['tp4-rank1', 'tp4-rank3'])]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.exl3-package-', dir=args.output.parent) as temporary:
        stage = Path(temporary)
        variants = []
        for profile, width, experts, topk, dtype, destinations in profiles:
            for capacity in capacities:
                tile = tiles.get(profile, {}).get(capacity)
                # Content-keyed: an override changes the compiled geometry, so it
                # must never land in (or be read from) the policy-keyed export.
                # Without an override the path is what every existing build used.
                profile_dir = (args.build_dir / profile if tile is None else
                               args.build_dir / f"{profile}+tile{'-'.join(map(str, tile))}")
                raw = profile_dir / f'm{capacity}'  # one capacity, one directory
                options = {'paired_boundary': profile.removeprefix('paired-')} if paired else {}
                if capacity in overrides:
                    options['blocks_per_sm'] = overrides[capacity]
                if tile is not None:
                    options['tile'] = tile
                if geometry != 'v41':
                    options['hidden'] = hidden
                block = route_block(geometry, capacity)
                ws = tile is None and warp_specialized(geometry, args.role, width, capacity)
                if ws:
                    block = 64
                if block != 8:
                    options['route_block'] = block
                if ws:
                    options['warp_specialized'] = True
                    if wire_input(geometry, args.role, width, capacity):
                        options['wire_input'] = True
                    if (stages := ws_input_stages(geometry, args.role, width, capacity)) is not None:
                        options['ws_input_stages'] = stages
                    if ws_dynamic_tiles(geometry, args.role, width, capacity):
                        options['ws_dynamic_tiles'] = True
                    if (policy_tile := ws_tile(geometry, args.role, width, capacity)) is not None:
                        options['tile'] = policy_tile
                elif tile is None and fused_input_rotation(geometry, args.role, width, capacity):
                    options['fused_input_rotation'] = True
                elif token_major_rotation(geometry, capacity):
                    options['token_major_rotation'] = True
                if (limit := swiglu_limit(geometry)) != 10.0:
                    options['swiglu_limit'] = limit
                # The default export, then any decode-schedule variant of it: the
                # same options plus a schedule, content-keyed in the build tree and
                # installed as m<capacity>-<schedule> beside m<capacity>.
                exports = [(None, raw, options)]
                for schedule, schedule_options in decode_schedule_variants(
                        geometry, args.role, profile, capacity):
                    exports.append((schedule, profile_dir.with_name(f'{profile_dir.name}+{schedule}')
                                    / raw.name, {**options, **schedule_options}))
                for schedule, raw, run_options in exports:
                    meta = export(raw, width, experts, capacity, tuple(args.bits), 'auto', topk, dtype, **run_options)
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
                    for destination in destinations:
                        directory = f'{destination}/m{capacity}' + ('' if schedule is None else f'-{schedule}')
                        for name in runtime_files:
                            target = stage / directory / name
                            target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(raw / name, target)
                        variant = {'directory': directory, **{key: meta[key] for key in
                            ('capacity', 'intermediate', 'experts', 'top_k', 'output_dtype', 'bits', 'blocks_per_sm')}}
                        if 'tile' in meta:
                            # What the compiler actually resolved, always: the pinned
                            # policy varies per capacity (m16 is the known special case),
                            # so a build without an override still has a real tile.
                            variant['tile'] = meta['tile']
                        if tile is not None and schedule is None:
                            # An A/B tile override applies to the default export; a
                            # schedule variant brings its own options.
                            variant['tile_requested'] = list(tile)
                        if 'route_block' in meta:
                            variant['route_block'] = meta['route_block']
                        if meta.get('token_major_rotation'):
                            variant['token_major_rotation'] = True
                        if meta.get('fused_input_rotation'):
                            variant['fused_input_rotation'] = True
                        if meta.get('warp_specialized'):
                            variant['warp_specialized'] = True
                        if 'input_format' in meta:
                            variant['input_format'] = meta['input_format']
                        if schedule is not None:
                            variant['schedule'] = schedule
                            variant['decode_schedule'] = meta['decode_schedule']
                        variants.append(variant)
                        if paired:
                            variants[-1]['paired_boundary'] = meta['paired_boundary']
                    # Large prefill exports must not retain another capacity's arenas.
                    gc.collect()
                    torch.cuda.empty_cache()
        files = {str(p.relative_to(stage)): {'bytes': p.stat().st_size, 'sha256': digest(p)}
                 for p in sorted(stage.rglob('*')) if p.is_file()}
        manifest = {'schema': 'cuteafd.exl3-package.v1', 'role': args.role,
                    'sparkinfer_revision': _pinned_sparkinfer.REVISION,
                    'compute': [props.major, props.minor], 'sms': props.multi_processor_count,
                    'variants': variants, 'files': files,
                    'runtime': {'library': 'libcute_dsl_runtime.so', 'sha256': digest(args.runtime),
                                'provider': 'installed nvidia-cutlass-dsl CUDA runtime; release entrypoint sets its library path'}}
        if paired:
            manifest['paired_tp4'] = True
        if geometry != 'v41':
            manifest['geometry'] = geometry
        if overrides:
            manifest['residency_overrides'] = [f'{capacity}={blocks}' for capacity, blocks in sorted(overrides.items())]
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
    create.add_argument('--geometry', choices=sorted(GEOMETRIES), default='v41',
                        help='Routed-expert geometry (cuteafd_core::ExpertGeometry::family)')
    create.add_argument('--all-spark-counts', action='store_true',
                        help='Opt in to all nonempty H128 Spark TP profiles (1..8)')
    create.add_argument('--profile', action='append', default=[],
                        help='Build only these profiles (repeatable), for bring-up; '
                             'without selection the legacy role profiles remain the default')
    create.add_argument('--paired-tp4', action='store_true', help='Export explicit paired H128 ownership kernels for all four Spark ranks')
    create.add_argument('--residency', action='append', default=[], metavar='CAPACITY=BLOCKS',
                        help='Explicit paired-package blocks/SM override; repeat per capacity (for example 80=2). B12X validates resources.')
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
    create.add_argument('--capacities', default='1,16,80,256,1024,4096')
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

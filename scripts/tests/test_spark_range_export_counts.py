"""CPU enumeration of EXL3 CUDA object jobs, not compiled byte-size claims.

FP8 composed programs have a separate exporter and are deliberately excluded
from this exact object count. No object sharing across padded/compact ABIs is
assumed. Set CUTEAFD_SPARK_RANGE_MATRIX for the complete live range receipt.
"""
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


pack = module('spark_count_exl3_pack', ROOT / 'python/tools/aot/package_exl3_aot.py')
scope = module('spark_count_release_scope', ROOT / 'scripts/build/release-scope.py')
CAPACITIES = (1, 16, 80, 256, 1024, 4096)


def test_every_planner_width_covered_world_has_a_package_alias():
    packages = module('spark_package_coverage', ROOT / 'python/tools/lib/spark_packages.py')
    entries = json.loads(packages.RANGES_PATH.read_text())['entries']
    for key, entry in entries.items():
        family, format = key.split(':')
        if entry.get('deprecated'):
            assert packages.package_worlds(family, format) == entry['worlds']
            assert "admitted_worlds" not in entry, key
            continue
        assert 'admitted_worlds' not in entry, key
        exl3 = format == 'exl3'
        native = family in ('v41', 'dsv4f', 'dsv4p') and not exl3
        nvfp4 = format in ('nvfp4', 'nvfp4a4')
        unit = 16 if nvfp4 else 32 if native or family in ('mimof', 'mimop') else 128
        intermediate = entry['intermediate']
        options = dict(exl3=exl3, native=native, nvfp4=nvfp4)
        available = set().union(*(packages.width_keys(intermediate, unit, n, **options)
                                  for n in entry['worlds']))
        emitted = packages.package_worlds(family, format)
        for world in range(1, 9):
            needed = packages.width_keys(intermediate, unit, world, **options)
            admitted = bool(needed) and needed <= available
            assert (world in emitted) == admitted, (key, world, needed, available)
    # TP5 needs no new CUDA class: its H128 program is already TP4's narrow
    # extent. The old default exporter omitted this metadata alias.
    assert 5 not in entries['qwen4:fp8']['worlds']
    assert 5 in packages.geometry_worlds('qwen4')



def test_inkling_native_and_two_tier_exl3_profiles_cover_tp3_tp6():
    packages = module('inkling_spark_package_coverage', ROOT / 'python/tools/lib/spark_packages.py')
    assert pack.GEOMETRIES['inkling2'] == (4608, 2304, 384, 8)
    assert pack.activation('inkling2') == 'situ'
    assert pack.swiglu_limit('inkling2') is None
    for kind in ['mxfp8', 'nvfp4', 'exl3']:
        assert {3, 6} <= set(packages.package_worlds('inkling2', kind))
        entry = packages.package_entry('inkling2', kind)
        assert entry['basis'] == 'lower_bound'
        assert entry['checkpoints'][0]['routed_expert_bytes'] > 0
    profiles = pack.build_profiles('spark', 'inkling2')
    for world in [3, 6]:
        widths = {destination: width for _, width, _, _, _, destinations in profiles
                  for destination in destinations if destination.startswith(f'tp{world}-rank')}
        assert len(widths) == world and sum(widths.values()) == 2304
    assert packages.geometry_worlds('inkling2_mxfp8') == packages.package_worlds('inkling2', 'mxfp8')
    native = module('inkling_native_width_metadata', ROOT / 'python/tools/aot/package_fp8_moe_aot.py')
    for world in range(2, 9):
        widths = native.exact_widths(2304, world)
        assert sum(widths) == 2304
        assert packages.width_keys(2304, 128, world) == {(False, width) for width in widths}
        first = 0
        for width in widths:
            assert first % 128 == width % 128 == 0
            assert first % 32 == width % 32 == 0  # No K32 scale group straddles ranks.
            first += width


def test_native_layout_exact_worlds_preserve_other_source_and_storage_units():
    packages = module('native_layout_policy', ROOT / 'python/tools/lib/spark_packages.py')
    for kind, native, legacy in [('mxfp8', 32, 128), ('nvfp4', 64, 16)]:
        policy = packages.native_layout('inkling2', kind)
        assert policy['exact_worlds'] == [4]
        assert packages.ownership_unit('inkling2', kind, 4) == native
        assert all(packages.ownership_unit('inkling2', kind, world) == legacy
                   for world in [1, 2, 3, 5, 6, 7, 8])
        native_export = module('exact_world_native_export', ROOT / 'python/tools/aot/package_fp8_moe_aot.py')
        assert native_export.exact_widths(2304, 4, native) == [576] * 4
    for key, entry in json.loads(packages.RANGES_PATH.read_text())['entries'].items():
        if not key.startswith('inkling2:'):
            assert entry['native_layout']['exact_worlds'] == [], key


def test_minimax_native_nvfp4_worlds_cover_the_padded_vendor_slices():
    packages = module('minimax_spark_package_coverage', ROOT / 'python/tools/lib/spark_packages.py')
    assert packages.geometry_worlds('minimax_nvfp4a4') == list(range(2, 9))
    assert packages.width_keys(3072, 16, 5, nvfp4=True) == {(False, 640)}
    assert packages.width_keys(3072, 16, 6, nvfp4=True) == {(False, 512)}
    entry = packages.package_entry('minimax', 'nvfp4a4')
    assert entry['basis'] == 'lower_bound'
    assert entry['checkpoints'][0]['routed_expert_bytes'] == 57 * 128 * (3 * 3072 * 6144 * 9 // 16 + 3 * 8)


def test_glmf_package_witness_is_the_live_mc_checkpoint():
    entries = json.loads((ROOT / 'scripts/build/spark-package-ranges.json').read_text())['entries']
    expected = {
        'model': 'wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K4-MC',
        'revision': '558931fe149f59b49d19b2376a692bd7a12fb8b7',
        # Measured 42 layers, 288 experts, three K4 projections plus native auxiliaries.
        'routed_expert_bytes': 42 * 288 * 3 * (4096 * 2048 // 2 + 2 * (4096 + 2048) + 4),
    }
    for key in ('glmf:exl3', 'glmf:fp8'):
        assert entries[key]['checkpoints'][1] == expected


def test_hy4f_exl3_spark_range_and_profiles_cover_tp1_to_tp6():
    packages = module('hy4f_spark_package_coverage', ROOT / 'python/tools/lib/spark_packages.py')
    assert pack.GEOMETRIES['hy4f'] == (1536, 1536, 512, 16)
    assert pack.activation('hy4f') == 'silu'
    assert pack.swiglu_limit('hy4f') == 10.0
    assert packages.package_worlds('hy4f', 'exl3') == [1, 2, 3, 4, 5, 6]
    entry = packages.package_entry('hy4f', 'exl3')
    assert entry['basis'] == 'lower_bound'
    assert entry['checkpoints'][0] == {
        'model': 'wrldsuksgo2mars/Hy4-flash-EXL3-K4-MC',
        'revision': 'cc72179113573d751ed3245e3e58d3640694378d',
        'routed_expert_bytes': 41 * 512 * 3 * (96 * 96 * 64 * 2 + 1536 * 2 * 2 + 4),
    }
    profiles = pack.build_profiles('spark', 'hy4f')
    for world in range(1, 7):
        widths = {destination: width for _, width, _, _, _, destinations in profiles
                  for destination in destinations if destination.startswith(f'tp{world}-rank')}
        assert len(widths) == world and sum(widths.values()) == 1536


def test_hy4_exl3_spark_range_and_profiles_cover_tp3_to_tp8():
    packages = module('hy4_spark_package_coverage', ROOT / 'python/tools/lib/spark_packages.py')
    assert pack.GEOMETRIES['hy4'] == (6144, 2048, 256, 8)
    assert pack.activation('hy4') == 'silu'
    assert pack.swiglu_limit('hy4') == 10.0
    assert packages.package_worlds('hy4', 'exl3') == [3, 4, 5, 6, 7, 8]
    entry = packages.package_entry('hy4', 'exl3')
    assert entry['basis'] == 'lower_bound'
    assert entry['checkpoints'][0] == {
        'model': 'wrldsuksgo2mars/Hy4-EXL3-K4-MC',
        'revision': '67346c4bdee2d8694606b4b4db9a96442e8667b7',
        'routed_expert_bytes': 77 * 256 * 3 * (384 * 128 * 64 * 2 + (6144 + 2048) * 2 + 4),
    }
    profiles = pack.build_profiles('spark', 'hy4')
    for world in range(3, 9):
        widths = {destination: width for _, width, _, _, _, destinations in profiles
                  for destination in destinations if destination.startswith(f'tp{world}-rank')}
        assert len(widths) == world and sum(widths.values()) == 2048


def count_objects(geometry, worlds, intermediate, unit=128):
    """Today's tp-keyed jobs versus proposed compact width-class jobs.

    EXL3 exports core + sum per profile/capacity. GB10 decode schedule variants
    stay distinct; migration retains their width-qualified equivalent.
    """
    capacities = pack.package_capacities(geometry, ','.join(map(str, CAPACITIES)))
    profiles = [p for p in pack.profiles_for_role('spark', geometry)
                if int(p[0].split('-')[0][2:]) in worlds]
    before = 2 * sum(1 + len(pack.decode_schedule_variants(geometry, 'spark', p[0], capacity))
                     for p in profiles for capacity in capacities)
    classes = sorted({width for world in worlds
                      for width in (intermediate // unit // world * unit,
                                    ((intermediate // unit + world - 1) // world) * unit)})
    assert all(width > 0 for width in classes)
    # Schedule variants are keyed by width, not by the retired tp4 prefix.
    schedules = {(p[1], capacity, name) for p in profiles for capacity in capacities
                 for name, _ in pack.decode_schedule_variants(geometry, 'spark', p[0], capacity)}
    after = 2 * (len(classes) * len(capacities) + len(schedules))
    per_world = {world: {intermediate // unit // world * unit,
                        ((intermediate // unit + world - 1) // world) * unit}
                 for world in worlds}
    full_schedules = {(world, width, capacity, name)
                      for world, widths in per_world.items() for width in widths for capacity in capacities
                      for name, _ in pack.decode_schedule_variants(
                          geometry, 'spark', f'tp{world}-width{width}', capacity)}
    shared_schedules = {(width, capacity, name) for _, width, capacity, name in full_schedules}
    full_before = 2 * (sum(map(len, per_world.values())) * len(capacities) + len(full_schedules))
    full_after = 2 * (len(classes) * len(capacities) + len(shared_schedules))
    return dict(before_objects=before, after_objects=after, width_columns=classes,
                full_range_count_keyed_objects=full_before,
                full_range_width_keyed_objects=full_after,
                current_covered_worlds=sorted({int(p[0].split('-')[0][2:]) for p in profiles}),
                proposed_worlds=sorted(worlds), capacities=capacities)


def count_layout_keys(counts):
    """Exact ABI/unit/width key counts; not mixed-format CUDA object counts."""
    per_world = {row['sparks']: {('padded' if row['padded'] else 'compact',
                                row['unit_columns'], width * row['unit_columns'])
                               for width in row['units_per_expert']}
                 for row in counts if row['sparks']}
    union = set().union(*per_world.values()) if per_world else set()
    baseline = set().union(*(keys for n, keys in per_world.items() if n not in (5, 7, 8)))
    return dict(worlds=sorted(per_world), count_keyed_jobs=sum(map(len, per_world.values())),
                width_keyed_jobs=len(union), keys=sorted(union),
                added_keys_5_7_8=len(union - baseline),
                keys_for_5_7_8={str(n): sorted(keys - baseline) for n, keys in per_world.items()
                                 if n in (5, 7, 8)})


def test_width_class_count_is_not_rank_count():
    # Existing TP6 jobs reuse identical-width ranks already; don't claim a
    # six-to-two compile win that today's packager has already achieved.
    result = count_objects('glm', {2, 3, 4, 6}, 2048)
    assert result['before_objects'] == 72
    assert result['after_objects'] == 72
    full = count_objects('glm', set(range(1, 9)), 2048)
    assert full['width_columns'] == [256, 384, 512, 640, 768, 1024, 2048]
    assert full['after_objects'] == 84
    assert full['before_objects'] == 144  # all geometric count-keyed jobs
    assert full['full_range_count_keyed_objects'] == 144
    assert full['full_range_width_keyed_objects'] == 84
    for world in (5, 7, 8):
        smaller = count_objects('glm', set(range(1, 9)) - {world}, 2048)
        assert smaller['full_range_width_keyed_objects'] == 84


def test_layout_key_counts_keep_padded_abi_and_units_distinct():
    counts = [dict(sparks=n, padded=False, unit_columns=128, units_per_expert=widths)
              for n, widths in [(1, [16]), (2, [8]), (3, [5, 6]), (4, [4]),
                                (5, [3, 4]), (6, [2, 3]), (7, [2, 3]), (8, [2])]]
    result = count_layout_keys(counts)
    assert (result['count_keyed_jobs'], result['width_keyed_jobs'], result['added_keys_5_7_8']) == (12, 7, 0)
    counts.append(dict(sparks=9, padded=True, unit_columns=128, units_per_expert=[2]))
    assert count_layout_keys(counts)['width_keyed_jobs'] == 8


def test_previous_release_exl3_count_enumeration(record_property):
    floor = scope.previous_release_packages()['expert']
    results = {}
    for path, layouts in floor.items():
        if not path.startswith('exl3/'):
            continue
        name = path.split('/')[-1].removeprefix('exl3-')
        geometry = name.rsplit('-k', 1)[0] if '-k' in name else 'v41'
        worlds = {int(layout.split('-')[0][2:]) for layout in layouts}
        results[path] = count_objects(geometry, worlds, pack.GEOMETRIES[geometry][1])
    assert results
    record_property('previous_release_exl3_jobs', json.dumps(results))


@pytest.mark.skipif(not os.environ.get('CUTEAFD_SPARK_RANGE_MATRIX'), reason='explicit live header matrix')
def test_whole_matrix_exl3_count_enumeration(record_property):
    entries = json.loads(Path(os.environ['CUTEAFD_SPARK_RANGE_MATRIX']).read_text())
    results = {}
    keys = []
    for entry in entries:
        for column in entry.get('columns', []):
            package = column['package'].split()[0]
            family, format = package.split(':')
            packages = module('spark_matrix_package_coverage', ROOT / 'python/tools/lib/spark_packages.py')
            emitted = packages.package_worlds(family, format)
            for row in column['counts']:
                # Resolved widths are supplied by Rust spark_recipe, while
                # aliases here come from the production Python packager policy.
                if row['sparks'] and row.get('resolved_width_columns') is not None:
                    assert row['sparks'] in emitted, (package, row['sparks'], emitted)
            keys.append(dict(model=entry['model'], package=package, rtx_gpus=column['rtx_gpus'],
                             geometric_minimum=(column.get('range') or {}).get('minimum'),
                             policy_only=column.get('range') is None, **count_layout_keys(column['counts'])))
            if ':exl3' not in package:
                continue
            geometry = package.split(':')[0]
            worlds = {row['sparks'] for row in column['counts'] if row['sparks']}
            key = (package, tuple(sorted(worlds)))
            results[str(key)] = count_objects(geometry, worlds, column['intermediate'], column['unit_columns'])
    assert results
    record_property('matrix_exl3_export_objects', json.dumps(results))
    record_property('matrix_export_keys', json.dumps(keys))
    output = os.environ.get('CUTEAFD_SPARK_RANGE_EXPORT_RESULTS')
    if output:
        Path(output).write_text(json.dumps(dict(exl3_objects=results, layout_keys=keys), indent=2) + '\n')


def test_computed_nvfp4_units_minimize_slowest_exact_rank():
    packages = module('computed_nvfp4_units', ROOT / 'python/tools/lib/spark_packages.py')
    native = module('computed_nvfp4_widths', ROOT / 'python/tools/aot/package_fp8_moe_aot.py')
    for intermediate, world, unit, widths in [
        (3072, 5, 128, [640, 640, 640, 640, 512]),
        (2304, 4, 64, [576] * 4), (640, 4, 32, [160] * 4),
    ]:
        assert packages.nvfp4_ownership_unit(intermediate, world) == unit
        assert native.exact_widths(intermediate, world, unit) == widths
        assert sum(widths) == intermediate
    # Existing V4.1 S576 ownership and byte offsets remain unchanged (#928).
    assert packages.layout_policy().ownership_unit(2304, 4) == 32
    shards = [packages.native_shard(4, rank, layer) for layer in range(4) for rank in range(4)]
    assert [(first, artifact.logical, artifact.stored) for first, artifact in shards] == [
        (rank * 576, 576, 576) for layer in range(4) for rank in range(4)]

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
            # A frozen format admits only the worlds it shipped on, never new ones.
            assert entry['admitted_worlds'] and set(entry['admitted_worlds']) <= set(entry['worlds']), key
            continue
        assert 'admitted_worlds' not in entry, key
        exl3 = format == 'exl3'
        native = family in ('v41', 'dsv4f', 'dsv4p') and not exl3
        unit = 32 if native or family in ('mimof', 'mimop') else 128
        intermediate = entry['intermediate']
        options = dict(exl3=exl3, native=native)
        available = set().union(*(packages.width_keys(intermediate, unit, n, **options)
                                  for n in entry['worlds']))
        emitted = packages.package_worlds(family, format)
        for world in range(1, 9):
            needed = packages.width_keys(intermediate, unit, world, **options)
            admitted = bool(needed) and needed <= available and (not native or intermediate % (32 * world) == 0)
            assert (world in emitted) == admitted, (key, world, needed, available)
    # TP5 needs no new CUDA class: its H128 program is already TP4's narrow
    # extent. The old default exporter omitted this metadata alias.
    assert 5 not in entries['qwen4:fp8']['worlds']
    assert 5 in packages.geometry_worlds('qwen4')



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


def test_variable_world_harness_tables_and_fixed_tile_refusals_without_cuda():
    import argparse
    import ast
    from spark_packages import native_spark_geometry, package_worlds
    expected = native_spark_geometry()
    for relative, symbol in [("python/tools/bench/bench_tp_ep_kernel.py", "TP_GEOMETRIES"),
                             ("python/tools/bench/benchmark_v41_ep_groups.py", "TP_DEGREES")]:
        path = ROOT / relative
        tree = ast.parse(path.read_text())
        nodes = [node for node in tree.body if
                 (isinstance(node, ast.ImportFrom) and node.module == "spark_packages") or
                 (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                  and target.id in (symbol, "SUPPORTED_SLICE_WIDTHS") for target in node.targets)) or
                 (isinstance(node, ast.FunctionDef) and node.name == "parse_args")]
        namespace = {"argparse": argparse, "Path": Path, "__doc__": "CPU parser proof"}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        if symbol == "TP_DEGREES":
            assert namespace[symbol] == (1, *package_worlds("v41", "mxfp4"))
        else:
            assert namespace[symbol] == tuple((tp, row[1], f"tp{tp}") for tp, row in expected.items())
            for world in range(1, 10):
                argv = ["--topologies", f"tp{world}", "--widths", "192", "--rows", "1", "--output", "/unused.json"]
                if world in expected and expected[world][1] % 192 == 0:
                    assert namespace["parse_args"](argv).topologies == f"tp{world}"
                else:
                    # TP8 is packaged but its logical 288-column shard does not
                    # tile this exact-tile-only diagnostic's widths.
                    with pytest.raises(SystemExit):
                        namespace["parse_args"](argv)

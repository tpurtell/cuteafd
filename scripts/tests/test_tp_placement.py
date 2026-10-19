"""Balanced TP expert placement prototype (v4): rules, unit maps, scorer, fit, numerics."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('tp_placement', ROOT / 'scripts/bench/tp-placement.py')
tp = importlib.util.module_from_spec(spec)
sys.modules['tp_placement'] = tp
spec.loader.exec_module(tp)

GLM = tp.Geometry(256)
MIMO = tp.Geometry(384)


@pytest.mark.parametrize('rule', tp.STATIC_RULES)
@pytest.mark.parametrize('world', [2, 3, 4, 5, 6])
def test_every_unit_has_exactly_one_owner(rule, world):
    for geometry in (GLM, MIMO, tp.Geometry(256, 2048, 32), tp.Geometry(512, 640, 128)):
        if world > geometry.units:
            continue
        for layer in (0, 3, 40, 77):
            counts = tp.count_matrix(rule, geometry, layer, world)
            assert (counts.sum(1) == geometry.units).all()
            # Every rule gives one expert floor or ceil of U/N units per rank.
            assert counts.max() - counts.min() <= 1


def test_rotated_is_the_exlr8_owner():
    # loader plan/exlr8.rs owner(): (units * expert + unit + layer) % ranks, and its test vector.
    for e, u, layer, n in ((0, 0, 3, 6), (7, 15, 40, 6), (255, 9, 77, 4), (256, 5, 40, 6)):
        assert tp.unit_owner('rotated', 16, e, u, layer, n) == (16 * e + u + layer) % n
    assert [u for u in range(16) if tp.unit_owner('rotated', 16, 256, u, 40, 6) == 1] == [5, 11]


def test_contiguous_matches_todays_exact_slices():
    # package_fp8_moe_aot.exact_widths / plan::experts widths: TP6 of 2048 is 384x4 + 256x2.
    counts = tp.count_matrix('contiguous', GLM, 40, 6)
    assert (counts * 128 == [384, 384, 384, 384, 256, 256]).all()
    k32 = tp.count_matrix('contiguous', tp.Geometry(384, 2048, 32), 40, 6)
    assert (k32 * 32 == [352, 352, 352, 352, 320, 320]).all()
    ranks = [tp.rank_units('contiguous', GLM, 40, 6, r) for r in range(6)]
    assert ranks[0].column_ranges(9, 128) == [(0, 384)]
    assert ranks[5].column_ranges(9, 128) == [(1792, 2048)]
    assert [r.width_classes() for r in ranks][4] == {2: list(range(256))}


def test_layer_rotation_moves_heavy_ranks_but_keeps_one_width_per_rank_layer():
    for layer in range(3, 9):
        counts = tp.count_matrix('layer-rotation', GLM, layer, 6)
        assert (counts == counts[0]).all()
        assert sorted(np.nonzero(counts[0] == 3)[0].tolist()) == sorted(((np.arange(4) + layer) % 6).tolist())


def test_rotated_balances_every_layers_memory_and_splits_two_width_classes():
    for geometry in (GLM, MIMO):
        for layer in (3, 4, 40, 77):
            ranks = [tp.rank_units('rotated', geometry, layer, 6, r) for r in range(6)]
            resident = [r.resident_units() for r in ranks]
            assert sum(resident) == geometry.experts * 16 and max(resident) - min(resident) <= 1
            for r in ranks:
                classes = r.width_classes()
                assert set(classes) == {2, 3}
                assert sum(len(v) for v in classes.values()) == geometry.experts
    # Units of one expert on one rank stride by N: non-contiguous column ranges.
    rank0 = tp.rank_units('rotated', GLM, 3, 6, 0)
    assert rank0.units[0] == (3, 9, 15)
    assert rank0.column_ranges(0, 128) == [(384, 512), (1152, 1280), (1920, 2048)]


def test_memory_balance():
    out = tp.memory(GLM, 6, 6144, 4.0, range(3, 78))
    assert out['contiguous']['max_over_mean'] == pytest.approx(1.125)
    assert out['rotated']['max_over_mean'] == pytest.approx(1.0, abs=1e-4)  # 4096 units / 6: +-1 per layer
    assert out['layer-rotation']['max_over_mean'] < 1.01


def test_synthetic_routes_reproduce_408():
    result = tp.synthetic(GLM, 6, [8, 43], waves=3000)
    assert result['8']['contiguous']['mean'] == pytest.approx(0.1875)
    assert result['8']['layer-rotation']['mean'] == pytest.approx(0.1875)
    # #408 scorer: rotated 0.177 / 0.171 at U = 8 / 43.
    assert result['8']['rotated']['mean'] == pytest.approx(0.177, abs=0.0015)
    assert result['43']['rotated']['mean'] == pytest.approx(0.171, abs=0.0015)
    for row in result.values():
        assert row['granularity bound']['mean'] <= row['rotated']['mean'] <= row['contiguous']['mean']


def _log(tmp_path, records):
    path = tmp_path / 'expert-routes-h.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in records) + '\n{"event": "expert_rou')
    return path


def _random_records(n, experts=256, seed=1, requests=True):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        reqs = [[1 + i % 4, 2], [8 + i % 3, 1]]
        topk = [rng.choice(experts, 8, replace=False).tolist() for _ in range(3)]
        record = {'event': 'expert_routes', 'kind': 'decode', 'layer': 3 + i % 3, 'rows': 3, 'unix_ns': i, 'topk': topk}
        if requests:
            record['requests'] = reqs
        out.append(record)
    out.append({'event': 'expert_routes', 'kind': 'prefill', 'layer': 3, 'rows': 900, 'expert_rows': [[0, 900]]})
    return out


def test_parse_splits_rows_by_request_parity(tmp_path):
    waves = tp.parse_log([_log(tmp_path, _random_records(4))])
    assert len(waves.layer) == 4 and waves.attributable.all()
    record = _random_records(1)[0]
    even = 0 if record['requests'][0][0] % 2 == 0 else 1
    first_two = sorted({x for row in record['topk'][:2] for x in row})
    assert waves.experts(f'half{even}', 0).tolist() == first_two
    assert waves.experts('all', 0).tolist() == sorted({x for row in record['topk'] for x in row})
    cache = tmp_path / 'w.npz'
    tp.save_waves(waves, cache)
    again = tp.load_waves(cache)
    assert (again.ids['half1'] == waves.ids['half1']).all()


def test_score_and_held_out_fit(tmp_path):
    waves = tp.parse_log([_log(tmp_path, _random_records(60))])
    result = tp.score(waves, GLM, 6, fit=True, passes=2)
    c1 = result['in_sample']['C2-C11']
    assert c1['contiguous']['mean'] == pytest.approx(0.1875)
    assert c1['rotated']['mean'] < 0.1875
    for split in ('request parity', 'time halves'):
        row = result['held_out'][split]['all decode']
        assert row['granularity bound']['mean'] <= row['fitted']['mean'] + 1e-12
    assert 'C1 rows=1' in result['in_sample']
    fine = tp.score(waves, GLM, 6, fit=False, fine_unit=32)['in_sample']['all decode']
    # Contiguous K32 slices: 11 of 64 units on the heavy ranks, every wave.
    assert fine['contiguous u32']['mean'] == pytest.approx(11 / 64)
    assert fine['layer-rotation u32']['mean'] == pytest.approx(11 / 64)


def test_score_falls_back_to_row_buckets_without_request_ids(tmp_path):
    waves = tp.parse_log([_log(tmp_path, _random_records(10, requests=False))])
    assert not waves.attributable.any()
    groups = tp.wave_groups(waves)
    assert groups['rows 2-8'].all() and not groups['rows=1'].any()


def test_fit_is_never_worse_in_sample_and_keeps_valid_patterns():
    rng = np.random.default_rng(3)
    ind = np.zeros((300, 256), np.int32)
    for row in ind:
        row[rng.choice(256, 12, replace=False, p=np.linspace(1, 3, 256) / np.linspace(1, 3, 256).sum())] = 1
    start = tp.count_matrix('rotated', GLM, 5, 6)

    def mean_share(counts):
        loads = ind @ counts
        return (loads.max(1) / loads.sum(1)).mean()

    fitted = tp.fit_layer(ind, start, 16, 6, passes=2)
    assert (fitted.sum(1) == 16).all() and set(np.unique(fitted)) <= {2, 3}
    assert mean_share(fitted) <= mean_share(start) + 1e-12
    assert len(tp.heavy_patterns(16, 6)) == 15


def test_bf16_rounding_and_reduction_model():
    x = np.array([1.0, 1 + 2 ** -8, 1 + 2 ** -7 + 2 ** -8 + 2 ** -10, -3.0e-3], np.float32)
    out = tp.bf16(x)
    assert out[0] == 1.0 and out[1] == 1.0  # ties to even
    assert out[2] == np.float32(1 + 2 ** -6)
    assert (out.view(np.uint32) & 0xFFFF == 0).all()
    small = tp.numerics(trials=2, hidden=64)
    assert small['contiguous']['trials_bitwise_different_from_contiguous'] == 0
    for name, row in small.items():
        assert row['mean_rel_abs_error'] < 0.02, name


def test_cli(tmp_path, capsys):
    log = _log(tmp_path, _random_records(12))
    out = tmp_path / 's.json'
    assert tp.main(['score', str(log), '--experts', '256', '--no-fit', '--json', str(out)]) == 0
    assert json.loads(out.read_text())['world'] == 6
    assert tp.main(['synthetic', '--experts', '384', '--unique', '8', '--waves', '50']) == 0
    assert 'rotated=' in capsys.readouterr().out
    empty = _log(tmp_path, [{'event': 'expert_timing'}])
    assert tp.main(['score', str(empty), '--experts', '256']) == 2
    with pytest.raises(ValueError):
        tp.Geometry(256, 2000, 128).units

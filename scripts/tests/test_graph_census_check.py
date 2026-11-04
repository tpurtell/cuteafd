"""Family-neutral, fail-closed timed CUDA graph census assertion."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'bench/check-graph-census.py'


def run_check(tmp_path, capture_s=None, anchor=True, evidence=True, final_marker=False, **changes):
    lines = ['2026-11-04T12:00:00.000000Z INFO starting']
    if anchor:
        lines.append('2026-11-04T12:00:01.000000Z INFO cuteafd::thermal: current={"queried_s":1.0}')
    if evidence:
        lines.append('2026-11-04T12:00:02.000000Z INFO graph census executable site exec=0x1 site=startup.rs:10:1')
    if capture_s is not None:
        lines.append(f'2026-11-04T12:00:{capture_s:09.6f}Z INFO graph census executable site exec=0x2 site=verify.rs:20:1')
    lines.append('2026-11-04T12:00:30.000000Z INFO stopped')
    if final_marker:
        lines.append('2026-11-04T12:00:32.000000Z INFO cuteafd::bench_census: final server thermal sample '
                     'server_now_s=32.0 source=census-final-thermal.json')
    log = tmp_path / 'coordinator.log'
    log.write_text('\n'.join(lines) + '\n')
    point = {'c': 16, 'phase': 'levels', 'thermal': {'from_s': 10.0, 'to_s': 20.0}}
    point.update(changes)
    sweep = tmp_path / 'sweep.json'
    sweep.write_text(json.dumps({'points': [point]}))
    result = subprocess.run([sys.executable, '-I', str(SCRIPT), '--log', str(log), '--sweep', str(sweep)],
                            capture_output=True, text=True)
    return result.returncode, json.loads(result.stdout)


def test_clean_run_excludes_startup_and_warmup(tmp_path):
    code, report = run_check(tmp_path, capture_s=9.0)
    assert code == 0
    assert report['status'] == 'pass'
    assert report['total_logged_captures'] == 2
    assert report['windows'][0]['captures'] == 0


def test_capture_in_timed_window_fails(tmp_path):
    code, report = run_check(tmp_path, capture_s=15.0)
    assert code == 1
    assert report['status'] == 'fail'
    assert report['windows'][0]['sites'] == {'verify.rs:20:1': 1}


def test_missing_clock_anchor_fails_closed(tmp_path):
    code, report = run_check(tmp_path, anchor=False)
    assert code == 2
    assert 'clock anchors' in report['error']


def test_missing_census_evidence_fails_closed(tmp_path):
    code, report = run_check(tmp_path, evidence=False)
    assert code == 2
    assert 'PROFILE=census evidence' in report['error']


@pytest.mark.parametrize('stamp', [9.999, 10.0, 20.0, 20.001])
def test_rounded_window_edges_fail_conservatively(tmp_path, stamp):
    code, report = run_check(tmp_path, capture_s=stamp)
    assert code == 1
    assert report['windows'][0]['captures'] == 1


@pytest.mark.parametrize('thermal', [None, {}, {'from_s': 20, 'to_s': 10},
                                     {'from_s': float('nan'), 'to_s': 20}])
def test_missing_or_invalid_point_window_fails_closed(tmp_path, thermal):
    code, report = run_check(tmp_path, thermal=thermal)
    assert code == 2
    assert report['status'] == 'error'


def test_final_server_marker_closes_last_window(tmp_path):
    code, report = run_check(tmp_path, final_marker=True, thermal={'from_s': 10, 'to_s': 31})
    assert code == 0
    assert report['windows'][0]['captures'] == 0


def test_log_must_bracket_whole_window(tmp_path):
    code, report = run_check(tmp_path, thermal={'from_s': 10, 'to_s': 31})
    assert code == 2
    assert 'bracket' in report['error']

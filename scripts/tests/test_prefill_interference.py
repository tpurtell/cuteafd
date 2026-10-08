import importlib.util
from pathlib import Path


PATH = Path(__file__).resolve().parents[1] / 'bench/deepseek_v41/bench-prefill-interference.py'
SPEC = importlib.util.spec_from_file_location('prefill_interference', PATH)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def test_encode_window_excludes_gap_straddling_prefill():
    stamps = [0.9, 1.01, 1.03, 1.31, 1.33]
    assert BENCH.gaps(stamps, 1.0, 1.05, contained=True) == [(1.03 - 1.01) * 1000]
    assert max(BENCH.gaps(stamps, 1.05, 1.32)) == (1.31 - 1.03) * 1000


def test_encode_window_ends_before_any_lm_prefill():
    logs = '\n'.join([
        'tokens=994 started_unix_ms=1000 finished_unix_ms=1300 owner_ms=40 roundtrip_ms=300 V4.1 asynchronous image encoder roundtrip',
        'started_unix_ms=1100 V4.1 prefill wave starts',
        'started_unix_ms=1400 V4.1 prefill wave starts',
    ])
    windows = BENCH.encoder_windows(logs, 0.9, 1.5)
    assert len(windows) == 1
    assert windows[0]['start'] == 1.0
    assert windows[0]['end'] == 1.1


def test_empty_encode_window_is_not_reported_as_zero_latency():
    assert BENCH.summarize([]) == {'max_ms': None, 'p99_ms': None, 'count': 0}
    assert BENCH.percentile(list(range(100))) == 98


def test_stream_auth_is_optional_and_not_in_evidence(monkeypatch):
    import io
    import json
    seen = []
    event = {'choices': [{'delta': {'content': 'ok'}}], 'usage': {'completion_tokens': 1}}
    def open_request(request, timeout):
        seen.append(request)
        return io.BytesIO(b'data: ' + json.dumps(event).encode() + b'\n\ndata: [DONE]\n\n')
    monkeypatch.setattr(BENCH.urllib.request, 'urlopen', open_request)
    record = BENCH.stream('http://127.0.0.1:8000', {'messages': []}, api_key='fixture-key')
    assert seen[-1].get_header('Authorization') == 'Bearer fixture-key'
    assert 'fixture-key' not in json.dumps(record)
    BENCH.stream('http://127.0.0.1:8000', {'messages': []})
    assert seen[-1].get_header('Authorization') is None

"""CPU contracts for the V4.1 HTTP stage-2 driver; no server required."""
import argparse
import copy
from pathlib import Path
import runpy

import pytest

GATE = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                         "qualify/deepseek_v41/qualify-prefix-engine.py"))


def report(command="exact", passed=True, metrics=None):
    return {"command": command, "pass": passed,
            "cases": [{"name": "one", "pass": passed, "metrics": metrics or {"cached_tokens": 100}}]}


def test_compare_metrics_and_failures():
    old = report(metrics={"cached_tokens": 100, "byte_exact": True})
    new = report(metrics={"cached_tokens": 120, "byte_exact": True})
    result = GATE["compare_reports"](new, old)
    assert result["pass"]
    assert {"case": "one", "metric": "cached_tokens", "before": 100, "after": 120} in result["rows"]
    assert not GATE["compare_reports"](report(passed=False), old)["pass"]
    assert not GATE["compare_reports"](report(command="host"), old)["pass"]
    new["cases"] = []
    assert not GATE["compare_reports"](new, old)["pass"]


def test_compare_admission_raw_values_and_missing_host_metrics():
    old = report("admission", metrics={"admitted_max": 8, "admitted_mean": 5.5,
        "pool_utilization_max": .6, "pool_utilization_mean": .4, "admission.deferred": 4})
    new = report("admission", metrics={"admitted_max": 16, "admitted_mean": 10,
        "pool_utilization_max": .9, "pool_utilization_mean": .7, "admission.deferred": 2})
    assert GATE["compare_reports"](new, old)["pass"]
    assert GATE["compare_reports"](report("host"), report("host"))["pass"]


@pytest.mark.parametrize("parity, padding", [(0, 0), (1, 1)])
def test_odd_even_padding_observes_server_tokens(parity, padding):
    calls = []
    def send(prompt):
        calls.append(prompt)
        return {"usage": {"prompt_tokens": 100 + prompt.count(" x")}}
    text, response, trials = GATE["parity_prompt"]("prompt", parity, send)
    assert text == "prompt" + " x" * padding
    assert response["usage"]["prompt_tokens"] % 2 == parity
    assert trials[-1]["padding"] == padding
    assert len(calls) == padding + 1


def test_padding_bounded_and_never_guesses_bpe_parity():
    def send(prompt):
        return {"usage": {"prompt_tokens": 100}}
    with pytest.raises(ValueError, match="cannot arrange parity"):
        GATE["parity_prompt"]("prompt", 1, send, attempts=3)


def test_cached_aliases_and_disagreement():
    assert GATE["cached"]({"usage": {"prompt_tokens": 10, "prompt_cache_hit_tokens": 8}}) == 8
    assert GATE["cached"]({"usage": {"prompt_tokens": 10,
        "prompt_tokens_details": {"cached_tokens": 10}}}) == 10
    with pytest.raises(ValueError):
        GATE["cached"]({"usage": {"prompt_tokens": 10, "prompt_cache_hit_tokens": 8,
                                  "prompt_tokens_details": {"cached_tokens": 9}}})


def test_admission_totals_and_kv_gauges():
    result = GATE["sample_metrics"]({"totals": {"requests_admitted": 40, "requests_retired": 24}},
                                   {"g": {"active": 3, "kv": {"pages": 100, "free": 25, "active": 60}}})
    assert result["admitted"] == 16
    assert result["pool_utilization"] == .75
    assert result["kv_pages_active"] == 60
    assert GATE["sample_metrics"]({}, {})["pool_utilization"] is None
    assert GATE["sample_metrics"]({"prefix": {"pages": 10, "pages_free": 4}}, {})["pool_utilization"] == .6


def test_counter_deltas():
    result = GATE["delta"]({"host_cache": {"restores": 1}, "prefix": None},
                           {"host_cache": {"restores": 3}, "prefix": {"hits": 4}})
    assert result == {"host_cache.restores": 2}


def test_exact_checks_full_turn_frontier_and_short_suffix(tmp_path):
    args = argparse.Namespace(command="exact", base_url="http://unused", nonce="test", model="default",
                              output=tmp_path / "report.json", reference_url=None, api_key=None, timeout=2)
    driver = GATE["Driver"](args)
    def response(prompt, hit, completion=4):
        return dict(content="OK", reasoning="", usage={"prompt_tokens": prompt,
                    "completion_tokens": completion, "prompt_cache_hit_tokens": hit})
    initial = response(100, 0)
    responses = iter([response(100, 100), response(115, 103)])
    driver.send = lambda body: next(responses)
    case = {"metrics": {}}
    driver.exact_case(case, driver.body("prompt"), initial)
    assert case["pass"]
    assert case["metrics"]["turn_frontier"] == 103
    assert case["metrics"]["suffix_tokens"] == 12
    responses = iter([response(100, 100), response(115, 100)])
    driver.exact_case(case, driver.body("prompt"), initial)
    assert not case["pass"]


def test_case_exception_cannot_leave_passing_case(tmp_path):
    args = argparse.Namespace(command="host", base_url="http://unused", nonce="test", model="default",
                              output=tmp_path / "report.json", reference_url=None, api_key=None, timeout=2)
    driver = GATE["Driver"](args)
    def fail(case):
        case["pass"] = True
        raise ValueError("later request failed")
    driver.case("failed", fail)
    assert not driver.report["cases"][0]["pass"]


@pytest.mark.parametrize("restores, passed, verified", [(0, False, False), (1, True, True), (None, True, None)])
def test_host_requires_restore_when_telemetry_exposed(tmp_path, restores, passed, verified):
    args = argparse.Namespace(command="host", base_url="http://unused", nonce="test", model="default",
                              output=tmp_path / "report.json", reference_url=None, api_key=None, timeout=2, entries=1)
    driver = GATE["Driver"](args)
    calls = 0
    def send(body):
        nonlocal calls
        calls += 1
        return dict(content="OK", reasoning="", usage={"prompt_tokens": 300,
            "prompt_cache_hit_tokens": 300 if calls == 6 else 0, "completion_tokens": 1})
    snapshots = iter([{"host_cache": {"restores": 0}}, {"host_cache": {"restores": 0}},
                      {"host_cache": {"restores": restores}}] if restores is not None else [{}, {}, {}])
    driver.send = send
    driver.get = lambda path: next(snapshots)
    driver.host()
    case = driver.report["cases"][0]
    assert case["pass"] is passed
    assert case["host_path_verified"] is verified


def test_compare_cli_writes_json(tmp_path, capsys):
    import json
    baseline, candidate, output = [tmp_path / name for name in ("baseline.json", "candidate.json", "compare.json")]
    baseline.write_text(json.dumps(report("host")))
    candidate.write_text(json.dumps(report("host")))
    assert GATE["main"](["compare", "--baseline", str(baseline), "--candidate", str(candidate),
                          "--output", str(output)]) == 0
    assert "100 -> 100" in capsys.readouterr().out
    assert json.loads(output.read_text())["pass"]

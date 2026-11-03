"""Offline loop-abort evidence must use the shipped rules and fail closed."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "rust/crates/cuteafd-daemon/tests"
spec = importlib.util.spec_from_file_location("loop_harness", TESTS / "loop-abort-fidelity-harness.py")
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


@pytest.mark.parametrize("period,length,position,rule", [
    (1, 2000, 1025, "Fast"),
    (8, 2000, 1032, "Fast"),
    (256, 3000, 2048, "Fast"),
    (257, 3000, 2342, "Repeat"),
    (300, 3000, 2643, "Repeat"),
])
def test_shipped_boundary_matches_rust(period, length, position, rule):
    ids = [i % period for i in range(length)]
    assert harness.feed(harness.shipped_guard(), ids) == (position, (rule, period))
    assert harness.feed(harness.RepeatGuard(0, 8, 0, 8), ids) is None


def test_fixture_is_the_full_rust_fixture():
    fixture = ROOT / "rust/crates/cuteafd-daemon/src/shared/repeat_guard_fixture.json"
    assert harness.run_fixture(fixture) == 0


def write_windows(tmp_path, tokens, roles):
    path = tmp_path / "windows.json"
    path.write_text(json.dumps({"windows": [{"id": "test", "tokens": tokens, "roles": roles}]}))
    return path


def test_eligibility_uses_period_gate_and_ignores_context(tmp_path):
    assert harness.rule_r_min_tokens() == 2342
    assert harness.rule_f_min_tokens() == 1025
    path = write_windows(tmp_path, [0] * 3000 + list(range(1024)), ["ctx"] * 3000 + ["gen"] * 1024)
    metrics = harness.fidelity_metrics(path)
    assert metrics["tokens"] == 1024
    assert metrics["fires"] == metrics["rule_r_eligible"] == metrics["rule_f_eligible"] == 0
    path = write_windows(tmp_path, list(range(2342)), ["gen"] * 2342)
    metrics = harness.fidelity_metrics(path)
    assert metrics["rule_r_eligible"] == metrics["rule_f_eligible"] == 1


def test_replay_stops_on_first_false_positive(tmp_path):
    path = write_windows(tmp_path, [0] * 2000, ["gen"] * 2000)
    with pytest.raises(ValueError, match=r"FIRE .*#test at token 1025 rule=Fast period=1"):
        harness.fidelity_metrics(path)


@pytest.mark.parametrize("tokens,roles", [([1], []), ([1], ["unknown"]), ([-1], ["gen"]), (["1"], ["gen"])])
def test_malformed_windows_are_not_silently_skipped(tmp_path, tokens, roles):
    with pytest.raises(ValueError, match="malformed"):
        harness.fidelity_metrics(write_windows(tmp_path, tokens, roles))


def test_config_inventory_is_pinned_and_excludes_media_and_retired():
    registry = ROOT / "rust/crates/cuteafd-bench/src/fidelity_dataset.rs"
    configs = list(harness.registered_configs(registry))
    assert len(configs) == 15
    assert len({name for name, _ in configs}) == len(configs)
    assert all(len(revision) == 40 for _, revision in configs)
    assert all(not name.endswith("_media") for name, _ in configs)

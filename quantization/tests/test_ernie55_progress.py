from datetime import datetime
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ernie55_flash.progress import state, TAIPEI


def test_eta_uses_current_completed_routed_layers(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    events = [(0, 900), (1, 100), (0, None), (0, 4), (1, 10), (2, 20), (3, None)]
    (logs / "preflight.log").write_text("\n".join(json.dumps(dict(
        phase="reference-forward", layer=i, event="loading" if s is None else "passed", seconds=s))
        for i, s in events))
    now = datetime.now(TAIPEI)
    snapshot = state(tmp_path, now, {"preflight"})
    assert snapshot["layer"] == 3 and snapshot["completed_target_layers"] == 3
    assert snapshot["preflight_target_eta"]["remaining_seconds"] == 600
    assert snapshot["preflight_target_eta"]["range_seconds"] == [400, 800]
    assert snapshot["pilot_completion_taipei"] is None
    assert state(tmp_path, now, set())["preflight_target_eta"] is None


def test_incomplete_pilot_is_terminal_with_no_completion_forecast(tmp_path):
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    (pilot / "pilot.json").write_text(json.dumps(dict(
        status="incomplete", active={"module":"q_b_proj","bits":6}, projections=[{}], completed=[],
        reason="deadline", elapsed_seconds=1800)))
    snapshot = state(tmp_path, datetime.now(TAIPEI), set())
    assert snapshot["terminal"] and snapshot["blocker"] == "deadline"
    assert snapshot["pilot_completion_taipei"] is None

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


def test_production_excludes_interrupted_and_dense_and_tracks_handoff(tmp_path):
    from ernie55_flash.progress import production_state
    prod = tmp_path / "production"
    layers = prod / "layers"
    layers.mkdir(parents=True)
    for i, seconds, interrupted in ((0, 10, False), (1, 100, False), (2, 500, True), (3, 90, False)):
        (layers / f"layers-{i:03d}.json").write_text(json.dumps(dict(
            family="layers", index=i, seconds=seconds, interrupted=interrupted)))
    (prod / "status.json").write_text(json.dumps(dict(phase="expert_group_complete", layer=4,
        family="layers", expert_projections=96, group_seconds=10)))
    snap = lambda: production_state(tmp_path, datetime.now(TAIPEI), {"production"})
    state = snap()
    assert state["target_eta"]["remaining_seconds"] == 39 * 95
    assert state["target_eta"]["range_seconds"] == [39 * 90, 39 * 100]
    assert state["completed_target_layers"] == 4 and state["expert_projections"] == 96
    for name in ("numerical-complete", "publication-complete"):
        (prod / (name+".json")).write_text('{"status":"passed"}')
        assert not snap()["terminal"]
    (prod / "handoff-complete.json").write_text('{"status":"passed"}')
    assert snap()["terminal"]


def test_production_does_not_report_finished_pilot_as_finished_quant(tmp_path):
    from ernie55_flash.progress import production_state
    (tmp_path / "pilot").mkdir()
    (tmp_path / "pilot/pilot.json").write_text('{"status":"passed"}')
    packed = tmp_path / "bootstrap-packed"
    packed.mkdir()
    (packed / "complete.json").write_text('{"status":"passed","projections":4}')
    state = production_state(tmp_path, datetime.now(TAIPEI), set())
    assert not state["terminal"] and state["target_eta"] is None
    assert state["layer0_bootstrap_projections"] == 4

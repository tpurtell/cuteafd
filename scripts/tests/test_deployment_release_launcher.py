"""S5 preserves final serving commands for every release-smoke configuration."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from v41_launch_capture import ROOT, capture, parse_run

MATRIX = json.loads((ROOT / "scripts/bench/release-smoke.example.json").read_text())
ENTRIES = MATRIX["entries"]
FIXTURES = Path(__file__).parent / "fixtures/deployment-launch"


def release_entry(entry: dict) -> dict:
    """Apply all smoke overrides while keeping process fixtures portable."""
    entry = {**entry, "set": {**MATRIX["defaults"]["set"], **entry.get("set", {})}}
    for key in entry["set"]:
        if key.startswith("SPARK_") and key.endswith("_HOST"):
            entry["set"][key] = f"worker-{key.split('_')[1]}"
        elif key.startswith("SPARK_") and "_LANE_" in key:
            rank = int(key.split("_")[1])
            entry["set"][key] = f"192.0.{2 if key.endswith('_A') else 3}.{rank + 1}"
    return entry


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e["name"])
def test_deployment_preserves_release_container_commands(entry: dict) -> None:
    golden = json.loads((FIXTURES / f"{entry['name']}.json").read_text())
    current = capture(ROOT, entry["name"], release_entry=release_entry(entry))
    assert [run["host"] for run in current["containers"]] == [run["host"] for run in golden]
    for old, new in zip(golden, current["containers"]):
        before, after = parse_run(old["argv"]), parse_run(new["argv"])
        if old["host"] == "coordinator":
            # capture() bounds readiness to 30s; admission now shares that deadline.
            before["env"]["CUTEAFD_SPARK_ADMISSION_TIMEOUT_SECONDS"] = "30"
        # Includes the byte-identical V4.1 fingerprint. Other families have no
        # resolved-config fingerprint today; adding a fabricated one is a failure.
        if entry["family"] == "deepseek_v41" and "serve-native" in before["command"]:
            position = before["command"].index("--max-output-tokens") + 2
            before["command"][position:position] = ["--mapped-scale-preload-gib", "0"]
        assert after == before, (entry["name"], old["host"])
    assert all("--deployment" in plan for plan in current["plans"])


@pytest.mark.parametrize("entry", [e for e in ENTRIES if e["family"] != "deepseek_v41"], ids=lambda e: e["name"])
def test_all_family_deployment_dry_run_starts_no_serving_containers(entry: dict) -> None:
    result = capture(ROOT, entry["name"], release_entry=release_entry(entry), dry_run=True)
    assert result["containers"] == []
    assert len(result["plans"]) == 1
    assert "--deployment" in result["plans"][0]


def exlr8_entry(manifest: str = "up1", **overrides: str) -> dict:
    entry = release_entry(next(e for e in ENTRIES if e["name"] == "glm53-exl3-max"))
    entry["set"].update(SPARK_COUNT="4", SPECULATOR="off", RTX_EXPERT_LAYERS="0",
        ATTENTION_PLACEMENT="context", EXLR8_MANIFEST=manifest,
        EXLR8_DENSE_FROM="deepseek-ai/DeepSeek-V4.1-Flash", **overrides)
    return entry


@pytest.mark.parametrize("manifest", ["up1", "bit325"])
def test_exlr8_forwards_native_selection_without_dense_worker_flags(manifest: str) -> None:
    result = capture(ROOT, f"exlr8-{manifest}", release_entry=exlr8_entry(manifest))
    assert len(result["plans"]) == 1
    plan = result["plans"][0]
    assert plan[plan.index("--exlr8-manifest") + 1] == manifest
    dense = plan[plan.index("--exlr8-dense-from") + 1]
    assert dense.startswith("/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/")
    assert "CUTEAFD_GLM_PREFILL_LANES=1" in plan
    assert len(result["containers"]) == 5
    for run in result["containers"]:
        parsed = parse_run(run["argv"])
        command = parsed["command"]
        assert command[command.index("--exlr8-manifest") + 1] == manifest
        if "serve-glm" in command:
            assert command[command.index("--exlr8-dense-from") + 1] == dense
            assert parsed["env"]["CUTEAFD_GLM_PREFILL_LANES"] == "1"
        else:
            assert "expertd-native" in command
            assert "--exlr8-dense-from" not in command


@pytest.mark.parametrize("overrides,reason", [
    ({"EXLR8_MANIFEST": "other"}, "EXLR8_MANIFEST"),
    ({"EXLR8_DENSE_FROM": ""}, "EXLR8_DENSE_FROM"),
    ({"RTX_GPUS": "1"}, "two RTX"),
    ({"SPARK_COUNT": "6"}, "canonical TP4"),
    ({"SPECULATOR": "dflash2"}, "SPECULATOR=off"),
    ({"RTX_EXPERT_LAYERS": "1"}, "cannot onboard"),
    ({"ATTENTION_PLACEMENT": "heads"}, "context attention"),
])
def test_exlr8_launcher_refuses_unsupported_route(overrides: dict, reason: str) -> None:
    entry = exlr8_entry()
    entry["set"].update(overrides)
    with pytest.raises(RuntimeError, match=reason):
        capture(ROOT, "exlr8-refusal", release_entry=entry, dry_run=True)


def test_deployment_goldens_cover_the_complete_release_matrix() -> None:
    assert {p.stem for p in FIXTURES.glob("*.json")} == {e["name"] for e in ENTRIES}
    assert {e["family"] for e in ENTRIES} == {
        "deepseek_v41", "deepseek_v4", "glm5", "glm5_flash", "mimo_v2", "qwen4"}


@pytest.mark.parametrize("quota", ["0", "auto", "16GiB"])
@pytest.mark.parametrize("full", ["off", "on"])
def test_host_cache_and_full_logits_reach_plan_and_serve(quota: str, full: str) -> None:
    entry = release_entry(next(e for e in ENTRIES if e["name"] == "glm53-exl3-max"))
    entry["set"].update(HOST_CACHE_BYTES=quota, FULL_PREFILL_LOGITS=full)
    result = capture(ROOT, f"host-budget-{quota}-{full}", release_entry=entry)
    plan = result["plans"][0]
    serve = parse_run(result["containers"][0]["argv"])["command"]
    assert plan[plan.index("--host-cache-bytes") + 1] == quota
    actual = serve[serve.index("--host-cache-bytes") + 1] if "--host-cache-bytes" in serve else "0"
    assert actual == quota
    assert ("--full-prefill-logits" in plan) == (full == "on")
    assert ("--full-prefill-logits" in serve) == (full == "on")


@pytest.mark.parametrize("quota", [None, "0", "2GiB"])
@pytest.mark.parametrize("vision", ["off", "rtx"])
def test_effective_media_cache_quota_reaches_plan(quota: str | None, vision: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import v41_launch_capture

    original_checkpoint = v41_launch_capture.release_checkpoint
    monkeypatch.setattr(v41_launch_capture, "release_checkpoint",
        lambda family, model: {**original_checkpoint(family, model), "vision_config": {"hidden_size": 1024}})
    entry = release_entry(next(e for e in ENTRIES if e["name"] == "mimo26-flash-min"))
    entry["set"].update(VISION=vision)
    entry["set"].pop("MEDIA_CACHE_BYTES", None)
    if quota is not None:
        entry["set"]["MEDIA_CACHE_BYTES"] = quota
    result = capture(ROOT, f"media-budget-{quota}-{vision}", release_entry=entry)
    plan = result["plans"][0]
    serve = parse_run(result["containers"][0]["argv"])["command"]
    expected = vision != "off" and quota is not None
    for command in (plan, serve):
        assert ("--media-cache-bytes" in command) == expected
        if expected:
            assert command[command.index("--media-cache-bytes") + 1] == quota


@pytest.mark.parametrize("enabled", ["off", "on"])
def test_implicit_mimo_host_cache_matches_effective_serve_argv(enabled: str) -> None:
    entry = release_entry(next(e for e in ENTRIES if e["name"] == "mimo26-flash-min"))
    entry["set"].pop("HOST_CACHE_BYTES", None)
    entry["set"]["MIMO_HOST_CACHE"] = enabled
    result = capture(ROOT, f"implicit-host-cache-{enabled}", release_entry=entry)
    plan = result["plans"][0]
    serve = parse_run(result["containers"][0]["argv"])["command"]
    actual = serve[serve.index("--host-cache-bytes") + 1] if "--host-cache-bytes" in serve else "0"
    assert actual == ("auto" if enabled == "on" else "0")
    assert plan[plan.index("--host-cache-bytes") + 1] == actual

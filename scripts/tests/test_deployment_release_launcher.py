"""S5 preserves final serving commands for every release-smoke configuration."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from v41_launch_capture import ROOT, capture, parse_run, namespace_launch_contract

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


@pytest.mark.parametrize("entry,swa_kv", [
    pytest.param(entry, swa_kv, id=entry["name"] + (f"-{swa_kv}" if swa_kv else ""))
    for entry in ENTRIES
    for swa_kv in ([None, "bf16", "int8", "fp8"] if entry["family"] == "mimo_v2" else [None])
])
@pytest.mark.parametrize("draft_kv", ["bf16", "fp8"])
def test_deployment_preserves_release_container_commands(entry: dict, swa_kv: str | None, draft_kv: str) -> None:
    golden = json.loads((FIXTURES / f"{entry['name']}.json").read_text())
    selected = release_entry(entry)
    compact_draft = entry["family"] in {"glm5", "glm5_flash", "mimo_v2"}
    if compact_draft:
        selected["set"]["DRAFT_KV_CACHE"] = draft_kv
    if swa_kv is not None:
        selected["set"]["SWA_KV_CACHE"] = swa_kv
    current = capture(ROOT, entry["name"], release_entry=selected)
    assert [run["host"] for run in current["containers"]] == [run["host"] for run in golden]
    for old, new in zip(golden, current["containers"]):
        before, after = namespace_launch_contract(parse_run(old["argv"])), parse_run(new["argv"])
        if old["host"] == "coordinator":
            # capture() bounds readiness to 30s; admission now shares that deadline.
            before["env"]["CUTEAFD_SPARK_ADMISSION_TIMEOUT_SECONDS"] = "30"
            if entry["family"] == "mimo_v2":
                anchor = "--kv-cache" if "--kv-cache" in before["command"] else "--prefix-partial"
                position = before["command"].index(anchor) + 2
                before["command"][position:position] = ["--swa-kv-cache", swa_kv or "int8"]
            if entry["family"] == "qwen4":
                # Local/dual still execute their whole-step default, now explicit in the plan.
                position = before["command"].index("--console-secret-file")
                before["command"][position:position] = ["--prefill-rows", "4096", "--prefill-waves", "1"]
            if entry["family"] == "deepseek_v4":
                # V4 forwards the planner's resolved unit and wave defaults.
                anchor = "--split-device" if "--split-device" in before["command"] else "--pool-tokens"
                position = before["command"].index(anchor) + 2
                before["command"][position:position] = ["--prefill-rows", "4096", "--prefill-waves", "2"]
            if compact_draft:
                anchor = {"glm5": "--pool-tokens", "glm5_flash": "--host-cache-bytes",
                          "mimo_v2": "--swa-kv-cache"}[entry["family"]]
                position = before["command"].index(anchor) + 2
                before["command"][position:position] = ["--draft-kv-cache", draft_kv]
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
        ATTENTION_PLACEMENT="context", EXLR8_MANIFEST=manifest, **overrides)
    return entry


@pytest.mark.parametrize("speculator", ["dflash2", "off", "unset"])
def test_exlr8_uses_glm_drafter_default_and_explicit_opt_out(speculator: str, tmp_path: Path) -> None:
    entry = exlr8_entry()
    launcher = None
    if speculator == "unset":
        launcher = tmp_path / "run-family.sh"
        source = ROOT / "scripts/launch/run-family.sh"
        # The release fixture has explicit settings; remove both before resolution.
        launcher.write_text(source.read_text().replace(
            'exlr8_manifest="$(get EXLR8_MANIFEST)"',
            'unset \'cfg[SPECULATOR]\' \'cfg[SPECULATOR_MODEL_ID]\'\n'
            'exlr8_manifest="$(get EXLR8_MANIFEST)"'))
    else:
        entry["set"]["SPECULATOR"] = speculator
    result = capture(ROOT, f"exlr8-{speculator}", release_entry=entry, launcher_source=launcher)
    plan = result["plans"][0]
    selected = "off" if speculator == "off" else "dflash2"
    assert plan[plan.index("--speculator") + 1] == selected
    for run in result["containers"]:
        command = parse_run(run["argv"])["command"]
        if "serve-glm" in command:
            assert ("--draft" in command) == (selected == "dflash2")
            if selected == "dflash2":
                assert "models--incoai--GLM-5.3-DFlash2/snapshots/" in command[command.index("--draft") + 1]
        else:
            assert "expertd-native" in command
            assert "--draft" not in command


@pytest.mark.parametrize("manifest", ["up1", "bit325"])
def test_exlr8_forwards_checkpoint_native_selection(manifest: str) -> None:
    result = capture(ROOT, f"exlr8-{manifest}", release_entry=exlr8_entry(manifest))
    assert len(result["plans"]) == 1
    plan = result["plans"][0]
    assert plan[plan.index("--exlr8-manifest") + 1] == manifest
    assert "--exlr8-dense-from" not in plan
    assert "--prefill-lanes" not in plan and "CUTEAFD_GLM_PREFILL_LANES=2" not in plan
    assert len(result["containers"]) == 5
    for run in result["containers"]:
        parsed = parse_run(run["argv"])
        command = parsed["command"]
        assert command[command.index("--exlr8-manifest") + 1] == manifest
        if "serve-glm" in command:
            assert "--exlr8-dense-from" not in command
            assert command[command.index("--prefill-waves") + 1] == "2"
            assert "CUTEAFD_GLM_PREFILL_LANES" not in parsed["env"]
        else:
            assert "expertd-native" in command


@pytest.mark.parametrize("waves", ["1", "2", "3", "4"])
def test_exlr8_launcher_preserves_explicit_prefill_waves(waves: str) -> None:
    result = capture(ROOT, f"exlr8-waves-{waves}", release_entry=exlr8_entry(PREFILL_WAVES=waves))
    plan = result["plans"][0]
    assert plan[plan.index("--prefill-waves") + 1] == waves
    coordinator = parse_run(result["containers"][0]["argv"])
    assert coordinator["command"][coordinator["command"].index("--prefill-waves") + 1] == waves
    assert "CUTEAFD_GLM_PREFILL_LANES" not in coordinator["env"]


@pytest.mark.parametrize("gpus,attention", [("1", "auto"), ("1", "heads"), ("2", "auto"), ("2", "context")])
@pytest.mark.parametrize("ranks", range(1, 7))
def test_exlr8_launcher_accepts_canonical_spark_worlds(ranks: int, gpus: str, attention: str) -> None:
    entry = exlr8_entry()
    entry["set"].update(SPARK_COUNT=str(ranks), RTX_GPUS=gpus, ATTENTION_PLACEMENT=attention)
    result = capture(ROOT, f"exlr8-world-{ranks}-{gpus}-{attention}", release_entry=entry)
    coordinator = parse_run(result["containers"][0]["argv"])["command"]
    assert ("--split-device" in coordinator) == (gpus == "2")
    assert len(result["containers"]) == ranks + 1
    workers = [parse_run(run["argv"])["command"] for run in result["containers"][1:]]
    for rank, command in enumerate(workers):
        assert command[command.index("--rank") + 1] == str(rank)
        assert command[command.index("--world") + 1] == str(ranks)


@pytest.mark.parametrize("overrides,reason", [
    ({"EXLR8_MANIFEST": "other"}, "EXLR8_MANIFEST"),
    ({"RTX_GPUS": "0"}, "RTX GPUs"),
    ({"RTX_GPUS": "3"}, "RTX GPUs"),
    ({"RTX_GPUS": "1"}, "heads attention"),
    ({"RTX_GPUS": "1", "ATTENTION_PLACEMENT": "layers"}, "heads attention"),
    ({"SPARK_COUNT": "9"}, "canonical Spark ranks"),
    ({"SPECULATOR": "mtp"}, "SPECULATOR=off or dflash2"),
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


@pytest.mark.parametrize("family", ["deepseek_v4", "glm5", "glm5_flash", "mimo_v2", "qwen4"])
@pytest.mark.parametrize("quota", [None, "0", "16GiB"])
def test_generic_host_default_matches_plan_and_serve(family: str, quota: str | None) -> None:
    entry = release_entry(next(e for e in ENTRIES if e["family"] == family))
    entry["set"].pop("HOST_CACHE_BYTES", None)
    if quota is not None:
        entry["set"]["HOST_CACHE_BYTES"] = quota
    result = capture(ROOT, f"host-default-{family}-{quota}", release_entry=entry)
    plan = result["plans"][0]
    serve = parse_run(result["containers"][0]["argv"])["command"]
    expected = quota if quota is not None else "auto"
    assert plan[plan.index("--host-cache-bytes") + 1] == expected
    assert serve.count("--host-cache-bytes") == int(expected != "0")
    if expected != "0":
        assert serve[serve.index("--host-cache-bytes") + 1] == expected


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
    assert serve.count("--host-cache-bytes") == 1
    assert actual == "auto"
    assert plan[plan.index("--host-cache-bytes") + 1] == actual


@pytest.mark.parametrize("card,lanes,concurrency,resolved", [
    ("v41-flash-max", "2", "16", "2"),
    ("mimo-pro-max", "1", "16", "1"),
    ("mimo-pro-max", "2", "16", "2"),
    ("mimo-pro-max", "2", "1", "1"),
    ("glm53-exl3-max", "1", "16", None),
])
def test_decode_lanes_forward_planner_resolution(card, lanes, concurrency, resolved):
    entry = release_entry(next(e for e in ENTRIES if e["name"] == card))
    entry["set"].update(DECODE_LANES=lanes, CONCURRENCY=concurrency)
    result = capture(ROOT, "decode-" + card + "-" + lanes + "-" + concurrency, release_entry=entry)
    plan = result["plans"][0]
    assert plan[plan.index("--decode-lanes") + 1] == lanes
    command = parse_run(result["containers"][0]["argv"])["command"]
    if resolved is None:
        assert "--decode-lanes" not in command
    else:
        assert command[command.index("--decode-lanes") + 1] == resolved


@pytest.mark.parametrize("card,lanes", [("v41-flash-max", "1"), ("glm53-exl3-max", "2"),
    ("mimo-pro-max", "0"), ("mimo-pro-max", "3")])
def test_decode_lanes_refuse_invalid_or_unsupported_before_serving(card, lanes):
    entry = release_entry(next(e for e in ENTRIES if e["name"] == card))
    entry["set"]["DECODE_LANES"] = lanes
    with pytest.raises(RuntimeError, match="unsupported --decode-lanes|DECODE_LANES must be"):
        capture(ROOT, "decode-refuse-" + card + "-" + lanes, release_entry=entry)

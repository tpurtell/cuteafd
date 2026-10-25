import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("memory_audit", Path(__file__).parents[1] / "bench" / "memory-audit.py")
memory_audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(memory_audit)

MIB = 1 << 20


def report(used, tracked, scopes):
    value = {"stage": "coordinator", "devices": [{"device": 0, "tracked": tracked, "peak": tracked, "scopes": scopes,
                                                  "used": used, "total": 96 << 30, "untracked": used - tracked}],
             "pinned": {"tracked": 0, "peak": 0, "scopes": {}}, "weights": [], "dual_formats": [], "host": {}}
    return "2026-10-10T00:00:00Z INFO cuteafd::memory: memory ledger report=" + json.dumps(value) + "\n"


def write(tmp_path, items):
    log = tmp_path / "coordinator.log"
    log.write_text(
        report(1100 * MIB, 1000 * MIB, {"weights": 1000 * MIB})
        + "2026-10-10T00:00:01Z INFO serve: DeepSeek V4 API is ready listen=0.0.0.0:1\n"
        + report(1700 * MIB, 1500 * MIB, {"weights": 1000 * MIB, "kv": 500 * MIB})
        # Lazily captured graphs grow the untracked bytes after ready; workspaces arrive with traffic.
        + report(2900 * MIB, 1800 * MIB, {"weights": 1000 * MIB, "kv": 500 * MIB, "workspace": 300 * MIB}))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"memory_layout": {"devices": [{"kind": "rtx", "index": 0, "items": items}]}}))
    return log, plan


def item(category, group, mib, lifetime="ready_resident"):
    return {"category": category, "group": group, "bytes": mib * MIB, "lifetime": lifetime}


def run(log, plan, strict=False, mapping="rtx0=0", tolerance=64.0):
    class Args:
        logs = [str(log)]
        compare = str(plan)
        device_map = mapping
        tolerance_mib = tolerance
        strict_ready = strict
    return memory_audit.ready_gate(Args)


def test_ready_ledger_takes_settled_tracked_bytes_and_untracked_at_ready(tmp_path):
    log, _ = write(tmp_path, [])
    ledger = memory_audit.ready_ledger(str(log))["0"]
    assert ledger["runtime"] == 200 * MIB
    assert ledger["workspace"] == 300 * MIB and ledger["kv"] == 500 * MIB


def test_ready_gate_ignores_growth_items_and_enforces_tolerance(tmp_path):
    fit = [item("weights", "w", 1000), item("kv", "records", 500), item("workspace", "steps", 300),
           item("runtime", "context+modules", 200), item("runtime", "graph growth", 900, "post_ready_reserve")]
    log, plan = write(tmp_path, fit)
    assert run(log, plan) == 0
    log, plan = write(tmp_path, fit[:-1] + [item("kv", "state", 100)])
    assert run(log, plan) == 1


def test_the_tagged_ready_report_wins_over_a_later_periodic_one(tmp_path):
    def tagged(used, tracked, scopes, stage):
        line = report(used, tracked, scopes)
        return line.replace('"stage": "coordinator"', '"stage": "%s"' % stage)
    log = tmp_path / "coordinator.log"
    log.write_text(
        report(1100 * MIB, 1000 * MIB, {"weights": 1000 * MIB})
        # Serve logs the ready ledger before the API line; a periodic report lands after traffic.
        + tagged(1600 * MIB, 1500 * MIB, {"weights": 1000 * MIB, "kv": 500 * MIB}, "ready")
        + "2026-10-10T00:00:01Z INFO serve: DeepSeek V4 API is ready listen=0.0.0.0:1\n"
        + report(2000 * MIB, 1500 * MIB, {"weights": 1000 * MIB, "kv": 500 * MIB}))
    first, last = memory_audit.ready_reports(str(log))
    assert first["stage"] == "ready" and first["devices"][0]["untracked"] == 100 * MIB
    assert last["devices"][0]["untracked"] == 500 * MIB


def tagged_ready():
    return report(1600 * MIB, 1500 * MIB, {"weights": 1000 * MIB, "kv": 500 * MIB}).replace(
        '"stage": "coordinator"', '"stage": "ready"')


def test_strict_ready_uses_one_snapshot_not_later_tracked_allocations(tmp_path):
    fit = [item("weights", "w", 1000), item("kv", "records", 500),
           item("runtime", "context+modules+graphs", 100), item("runtime", "graph growth", 900, "post_ready_reserve")]
    log, plan = write(tmp_path, fit)
    log.write_text(tagged_ready() + report(2900 * MIB, 1800 * MIB,
                                         {"weights": 1000 * MIB, "kv": 500 * MIB, "workspace": 300 * MIB}))
    assert run(log, plan, strict=True) == 0
    assert memory_audit.strict_ready_ledger(log)["0"] == {"weights": 1000 * MIB, "kv": 500 * MIB,
                                                        "runtime": 100 * MIB}
    assert run(log, plan) == 1  # Legacy reconstruction includes the later workspace.
    plan.write_text(json.dumps({"memory_layout": {"devices": [{"kind": "rtx", "index": 0,
                           "items": fit[:-1] + [item("kv", "extra", 65)]}]}}))
    assert run(log, plan, strict=True) == 1


def test_strict_ready_compares_only_structurally_resident_items(tmp_path, capsys):
    fit = [item("runtime", "graph growth", 900, "post_ready_reserve"),
           item("reserved", "runtime growth", 300, "post_ready_reserve"),
           item("reserved", "memory sample margin", 32, "post_ready_reserve"),
           item("runtime", "resident context", 100)]
    log, plan = write(tmp_path, fit)
    log.write_text(report(100 * MIB, 0, {}).replace('"stage": "coordinator"', '"stage": "ready"'))
    assert run(log, plan, strict=True, tolerance=0) == 0
    assert "planned 100 MiB, ledger 100 MiB" in capsys.readouterr().out
    # Labels and categories never determine residency.
    fit[-1]["group"] = "graph growth"
    fit[0]["group"] = "ordinary runtime"
    plan.write_text(json.dumps({"memory_layout": {"devices": [{"kind": "rtx", "index": 0, "items": fit}]}}))
    assert run(log, plan, strict=True, tolerance=0) == 0
    fit[-1]["bytes"] += MIB
    plan.write_text(json.dumps({"memory_layout": {"devices": [{"kind": "rtx", "index": 0, "items": fit}]}}))
    assert run(log, plan, strict=True, tolerance=0) == 1


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("lifetime", [None, "growth", "", 1])
def test_ready_gate_rejects_missing_or_invalid_lifetime(tmp_path, strict, lifetime):
    resident = item("runtime", "context", 100)
    if lifetime is None:
        del resident["lifetime"]
    else:
        resident["lifetime"] = lifetime
    log, plan = write(tmp_path, [resident])
    log.write_text(tagged_ready())
    assert run(log, plan, strict=strict) == 2


def test_startup_plan_uses_serving_admission_and_rejects_missing_duplicate(tmp_path):
    log = tmp_path / "serve.log"
    line = "INFO placement: GLM Flash startup plan plan=" + json.dumps({"memory_layout": {
        "devices": [{"kind": "rtx", "index": 0, "items": [item("runtime", "graphs", 100)]}]},
        "speculation": True, "graph_shapes": 280, "graphs": [12880, 12600]}) + "\n"
    log.write_text(line + tagged_ready())
    assert memory_audit.startup_plan(log)["devices"][0]["items"][0]["bytes"] == 100 * MIB
    log.write_text(tagged_ready())
    with pytest.raises(ValueError, match="exactly one"):
        memory_audit.startup_plan(log)
    log.write_text(line + line + tagged_ready())
    with pytest.raises(ValueError, match="exactly one"):
        memory_audit.startup_plan(log)


def test_strict_ready_rejects_missing_tag_or_requested_device(tmp_path):
    log, plan = write(tmp_path, [])
    assert run(log, plan, strict=True) == 1
    log.write_text(tagged_ready())
    assert run(log, plan, strict=True, mapping="rtx0=1") == 1
    assert run(log, plan, strict=True, mapping="rtx1=0") == 2
    assert run(log, plan, strict=True, mapping="rtx0=0,rtx0=1") == 2
    assert run(log, plan, strict=True, mapping="rtx0") == 2


@pytest.mark.parametrize("change", [
    "missing_used", "missing_untracked", "missing_scopes", "negative_untracked", "tracked_mismatch",
    "used_mismatch", "total_mismatch", "duplicate_device", "empty_devices", "boolean_bytes",
])
def test_strict_ready_rejects_incomplete_or_inconsistent_query_bytes(tmp_path, change):
    log, plan = write(tmp_path, [])
    value = json.loads(tagged_ready().split("report=", 1)[1])
    device = value["devices"][0]
    if change.startswith("missing_"):
        del device[change.removeprefix("missing_")]
    elif change == "negative_untracked":
        device["untracked"] = -1
    elif change == "tracked_mismatch":
        device["tracked"] += 1
    elif change == "used_mismatch":
        device["used"] += 1
    elif change == "total_mismatch":
        device["total"] = 1
    elif change == "duplicate_device":
        value["devices"].append(dict(device))
    elif change == "empty_devices":
        value["devices"] = []
    elif change == "boolean_bytes":
        device["untracked"] = True
    log.write_text("INFO memory ledger report=" + json.dumps(value) + "\n")
    assert run(log, plan, strict=True) == 1


@pytest.mark.parametrize("tolerance", [-1.0, float("nan"), float("inf")])
def test_strict_ready_rejects_invalid_tolerance(tmp_path, tolerance):
    log, plan = write(tmp_path, [])
    log.write_text(tagged_ready())
    assert run(log, plan, strict=True, tolerance=tolerance) == 2


def test_pool_following_v4_page_tables_use_the_planners_kv_records_scope(tmp_path):
    engine = (Path(__file__).parents[2] / "rust/crates/cuteafd-daemon/src/families/deepseek_v4/engine.rs").read_text()
    allocation = engine.split("c4_page_table: {", 1)[1].split("},", 1)[0]
    assert 'scope("kv/records")' in allocation
    assert "ints(page_lists * self.shape.units)?" in allocation
    assert "self.step_buffers(t, if shared_page_list { 1 } else { t })?" in engine
    table_bytes = (2_097_152 // 256) * (2 + 64) * 4
    log = tmp_path / "coordinator.log"
    log.write_text(report(1000 * MIB + table_bytes, 900 * MIB + table_bytes,
                          {"weights": 900 * MIB, "kv/records": table_bytes}))
    categories = memory_audit.summarize(next(memory_audit.reports(str(log))))["devices"][0]["categories"]
    assert categories["kv"] == table_bytes
    assert categories.get("workspace", 0) == 0


def test_v4_admits_tables_and_frozen_prefix_quota_before_first_reservation():
    root = Path(__file__).parents[2] / "rust/crates/cuteafd-daemon/src/families/deepseek_v4"
    startup = (root / "mod.rs").read_text().split("pub(crate) fn with_engine", 1)[1]
    assert startup.index("engine.attach_peer(") < startup.index("admission::host_memory(")
    assert startup.index("admission::host_memory(") < startup.index("engine.attach_context(")
    assert startup.index("admission::host_memory(") < startup.index("engine.prepare_workspaces()")
    assert "body(&engine, &mut transports, &runtime, host_config)" in startup
    admission = (root / "admission.rs").read_text().split("pub(crate) fn host_memory(", 1)[1]
    admission = admission.split("pub(crate) fn request(", 1)[0]
    assert admission.count("resolve_host_config(") == 1
    assert admission.count("admit_host_memory(") == 1
    assert "config.as_ref().map_or(0, |c| c.bytes)" in admission
    assert "&tables, logits" in admission
    serve = (root / "serve.rs").read_text()
    cache = serve.split("fn prefix_cache", 1)[1].split("/// Gives a finished", 1)[0]
    assert "host_config.map(|config|" in cache
    assert ".host_config(" not in cache
    assert ".host_tier(" not in cache
    assert "resolve_host_config(" not in cache
    assert "prefix_cache(engine, prefix, host_config)?" in serve


def test_glmf_kda_only_pool_backing_uses_the_planners_prefix_scope():
    source = (Path(__file__).parents[2] / "rust/crates/cuteafd-daemon/src/families/glm5_flash/engine.rs").read_text()
    allocation = source.split("let pool_mark_backing = if pool_mark_unit_bytes > 0 {", 1)[1].split("} else { None };", 1)[0]
    assert 'scope("prefix/KDA-only pool marks")' in allocation.split("Some(zeroed(", 1)[0]
    assert 'pool_pages.checked_mul(pool_mark_unit_bytes)' in allocation
    assert memory_audit.category("prefix/KDA-only pool marks") == "prefix"


def test_tp2_expert_loaders_scope_both_parent_and_parallel_rank_allocations():
    root = Path(__file__).parents[2] / "rust/crates/cuteafd-daemon/src/shared/experts/rtx"
    for name in ["native.rs", "exl3.rs"]:
        source = (root / name).read_text()
        parent = source.split("pub(crate) fn load_pair_weights(", 1)[1]
        assert '"experts/TP2 expert layer halves {layers:?}"' in parent
        assert "layers.start, layers.end" not in parent
        assert 'scope_owned(&label)' in parent.split("synchronized_load::load_pair", 1)[0]
        parallel = parent.split("synchronized_load::load_pair", 1)[1]
        assert 'scope_owned(&label)' in parallel.split("Ok(vec![", 1)[0]
    assert memory_audit.category("experts/TP2 expert layer halves 0..36") == "experts"


def test_fp8_tp2_pair_scopes_its_bf16_output_with_the_expert_arena():
    root = Path(__file__).parents[2] / "rust/crates/cuteafd-daemon/src/shared/experts/rtx"
    source = (root / "fp8moe.rs").read_text()
    pair = source.split("pub(crate) fn load_pair(", 1)[1].split("fn log_pair_load", 1)[0]
    scope = 'scope("experts/TP2 FP8 expert layer halves")'
    assert scope in pair.split("Fp8Experts::load_pair", 1)[0]
    assert pair.index(scope) < pair.index("Allocation::new(devices[rank], plans[rank].output_bytes)")
    assert memory_audit.category("experts/TP2 FP8 expert layer halves") == "experts"


def test_raw_cuda_audit_labels_gb10_reclaimable_cache(capsys):
    scope = "raw cudaMemGetInfo; GB10 under-reports reclaimable page cache; host MemAvailable is allocatable headroom"
    report = {"stage": "ready", "devices": [{"device": 0, "tracked": 0, "peak": 0,
        "cuda_memory_scope": scope, "scopes": {}}], "host": {"MemAvailable": 45 * 2**30}}
    memory_audit.render("spark.log", memory_audit.summarize(report))
    output = capsys.readouterr().out
    assert scope in output
    assert "MemAvailable 45.00" in output

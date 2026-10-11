import importlib.util
import json
from pathlib import Path

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


def item(category, group, mib):
    return {"category": category, "group": group, "bytes": mib * MIB}


def run(log, plan):
    class Args:
        logs = [str(log)]
        compare = str(plan)
        device_map = "rtx0=0"
        tolerance_mib = 64.0
    return memory_audit.ready_gate(Args)


def test_ready_ledger_takes_settled_tracked_bytes_and_untracked_at_ready(tmp_path):
    log, _ = write(tmp_path, [])
    ledger = memory_audit.ready_ledger(str(log))["0"]
    assert ledger["runtime"] == 200 * MIB
    assert ledger["workspace"] == 300 * MIB and ledger["kv"] == 500 * MIB


def test_ready_gate_ignores_growth_items_and_enforces_tolerance(tmp_path):
    fit = [item("weights", "w", 1000), item("kv", "records", 500), item("workspace", "steps", 300),
           item("runtime", "context+modules", 200), item("runtime", "graph growth", 900)]
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


def test_pool_following_v4_page_tables_use_the_planners_kv_records_scope(tmp_path):
    engine = (Path(__file__).parents[2] / "rust/crates/cuteafd-daemon/src/families/deepseek_v4/engine.rs").read_text()
    allocation = engine.split("c4_page_table: {", 1)[1].split("},", 1)[0]
    assert 'scope("kv/records")' in allocation
    assert "ints(rows.max(1) * self.shape.units)?" in allocation
    table_bytes = (2_097_152 // 256) * (2 * 4096 + 64) * 4
    log = tmp_path / "coordinator.log"
    log.write_text(report(1000 * MIB + table_bytes, 900 * MIB + table_bytes,
                          {"weights": 900 * MIB, "kv/records": table_bytes}))
    categories = memory_audit.summarize(next(memory_audit.reports(str(log))))["devices"][0]["categories"]
    assert categories["kv"] == table_bytes
    assert categories.get("workspace", 0) == 0


def test_tp2_expert_loaders_scope_both_parent_and_parallel_rank_allocations():
    root = Path(__file__).parents[2] / "rust/crates/cuteafd-daemon/src/shared/experts/rtx"
    for name in ["native.rs", "exl3.rs"]:
        source = (root / name).read_text()
        assert 'format!("experts/TP2 expert layer halves {}..{}", layers.start, layers.end)' in source
        parent = source.split("pub(crate) fn load_pair(", 1)[1]
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

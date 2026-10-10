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

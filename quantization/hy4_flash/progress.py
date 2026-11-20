"""Deliver the Hy4 quantization status to the originating Codex task."""
from datetime import datetime, timedelta
import json
from pathlib import Path
import re
from statistics import mean
import sys
import subprocess
from zoneinfo import ZoneInfo

import os

ROOT = Path(os.environ.get("CUTEAFD_HY4_TASK", Path.home() / ".cache/cuteafd/quantization/hy4-flash-k4-mc"))
PRODUCTION = ROOT / "production"


def read(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def eta(now, scope, remaining, rates):
    if not rates:
        return None
    seconds = remaining * mean(rates)
    return dict(scope=scope, sample_count=len(rates), remaining_seconds=seconds,
                completion_time=(now + timedelta(seconds=seconds)).isoformat(),
                remaining_range_seconds=[remaining * min(rates), remaining * max(rates)])


def engram_progress(now):
    receipt_path = ROOT / "engram-export-complete.json"
    receipt = read(receipt_path)
    if receipt:
        if not receipt.get("proof") or any(r["mismatch_count"] for r in receipt["proof"].values()):
            return dict(status="failed")
        rows = sum(row["rows"] for row in receipt["proof"].values())
        completed = datetime.fromtimestamp(receipt_path.stat().st_mtime, now.tzinfo)
        return dict(status="passed", verified_rows=rows, total_rows=rows, remaining_seconds=0,
                    completion_time=completed.isoformat())
    log = ROOT / "engram-export.log"
    if not log.exists():
        return None
    matches = re.findall(r"^(\S+) model\.layers\.(\d+)\..*verified ([\d,]+)/([\d,]+) rows",
                         log.read_text(), re.M)
    if not matches:
        return None
    total_by_layer = {1: 164_994_380, 11: 164_999_008, 21: 165_004_086, 31: 165_008_942}
    total = sum(total_by_layer.values())
    samples = []
    for when, layer, done, _ in matches:
        layer = int(layer)
        checked = int(done.replace(",", "")) + sum(
            rows for index, rows in total_by_layer.items() if index < layer)
        samples.append((datetime.fromisoformat(when), checked, layer))
    latest, done, layer = samples[-1]
    window = samples[-9:]
    rates = [(b[0] - a[0]).total_seconds() / (b[1] - a[1])
             for a, b in zip(window, window[1:], strict=False)
             if 0 < b[1] - a[1] and b[0] > a[0]]
    return dict(status="running", layer=layer, verified_rows=done, total_rows=total,
                observed_at=latest.isoformat(), forecast=eta(now, "Engram export only", total - done, rates))


def layer_progress(now, status, events):
    complete = [r for r in events if r.get("phase") == "target_layer_complete"]
    interrupted = {r["current_layer"] for r in events if r.get("phase") == "expert_group_replayed"}
    started = [r["current_layer"] for r in events if r.get("phase") == "target_layer"]
    interrupted.update(i for i in set(started) if started.count(i) > 1)
    measured = [r["layer_seconds"] for r in complete if r.get("current_layer", 0) > 0
                and r["current_layer"] not in interrupted and r.get("layer_seconds", 0) > 0][-6:]
    left = status.get("target_layers", 42) - status.get("completed_target_layers", 0)
    if left <= 0 or not status.get("phase", "").startswith(("target_layer", "expert_", "core_")):
        return None, len(complete)
    partial = status.get("expert_projections", 0)
    expert_layer_total = status.get("total_expert_projections", 1536)
    if status.get("current_layer", 0) < status.get("completed_target_layers", 0):
        partial = 0
    if measured:
        fraction = partial / expert_layer_total if expert_layer_total else 0
        remaining = max(0, left - fraction)
        return eta(now, "target layers only", remaining, measured), len(complete)
    groups = [r for r in events[-100:] if r.get("phase") == "expert_group_complete"
              and (r.get("expert_group_seconds") or 0) > 0 and r.get("new_expert_projections", 96) > 0]
    rates = [r["expert_group_seconds"] / r.get("new_expert_projections", 96) for r in groups[-8:]]
    remaining = left * expert_layer_total - partial
    estimate = eta(now, "provisional target expert encodes only", remaining, rates)
    return estimate, len(complete)


def read_events():
    path = PRODUCTION / "events.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def main():
    now = datetime.now(ZoneInfo("Asia/Taipei"))
    status = read(PRODUCTION / "status.json")
    events = read_events()
    forecast, completed = layer_progress(now, status, events)
    if not events and status.get("status") != "blocked":
        parity = read(ROOT / "reference-forward-passed.json")
        if parity.get("status") == "passed":
            status.update(phase="production preparation")
        status.update(preflight_layers=len(parity.get("layers", [])), preflight_status=parity.get("status"))
    failed = subprocess.run(["systemctl", "--user", "list-units", "--failed", "--plain", "--no-legend",
                             "cuteafd-hy4-flash-production.service", "cuteafd-hy4-flash-publish.service",
                             "cuteafd-hy4-flash-handoff.service", "cuteafd-hy4-flash-cleanup.service"],
                             capture_output=True, text=True, timeout=5)
    if failed.stdout.strip():
        status.update(status="blocked", failed_units=[line.split()[0] for line in failed.stdout.splitlines()])
    if forecast and status.get("status") == "blocked":
        forecast["assumes_resume_now"] = True
    cleanup = read(PRODUCTION / "handoff-cleanup-complete.json")
    handoff = read(PRODUCTION / "handoff-complete.json")
    if cleanup.get("status") == handoff.get("status") == "passed":
        forecast = dict(remaining_seconds=0, completion_time=cleanup.get("time", now.isoformat()))
    publication = read(PRODUCTION / "publication-complete.json")
    handoff_hold = (PRODUCTION / "sparknest-handoff.hold").is_file()
    waiting_for_handoff = (handoff_hold and publication.get("status") == "passed"
                           and handoff.get("status") != "passed")
    if status.get("phase") == "numerical_complete":
        for receipt, label in ((cleanup, "complete"), (handoff, "cleanup"),
                               (publication, "placement")):
            if receipt.get("status") == "passed":
                status.update(phase=label, status="passed" if label == "complete" else "running")
                break
        else:
            status.update(phase="publication", status="running")
    if waiting_for_handoff:
        status.update(phase="sparknest_handoff", status="paused",
                      pause_reason="TJ requested a hold after the verified Hugging Face upload")
    snapshot = dict(as_of=now.isoformat(), status=status, completed_layer_receipts=completed,
                    forecast=forecast, engram=engram_progress(now),
                    publication={k: publication[k] for k in ("status", "url", "revision") if k in publication},
                    handoff=handoff.get("status"), handoff_hold=handoff_hold, cleanup=cleanup.get("status"))
    path = ROOT / "progress-updates/latest.json"
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(snapshot, indent=2) + "\n")
    temporary.replace(path)
    if "--snapshot-only" in sys.argv:
        return
    message = (
        f"Scheduled 30-minute Hy4-flash K4 MC update requested by TJ. Read {path} and task receipts. "
        "Report phase, current layer/projection progress, time remaining and completion date/time in "
        "Asia/Taipei. Use measured throughput and uncertainty, or explicitly say both estimates are "
        "unavailable if there is no measured basis. Keep head, DFly2, Engram export, upload and "
        "placement separate from layer estimates. Continue the already authorized process from "
        "/home/tj/Developer/cuteafd/hy4-flash-k4-mc-quant.md with full v2 preflight and production gates. "
        "No separate pilot. After a reboot, report status only until TJ explicitly asks to resume. Do not duplicate live work or restart completed jobs."
    )
    if handoff_hold:
        message += (" TJ has paused SparkNest import, placement, and cleanup until explicitly resumed."
                    " Finish and verify the Hugging Face upload, then report its receipt and hold.")
    subprocess.run([os.environ.get("CODEX_CLI", "codex"), "queue", "--thread",
                    os.environ["CODEX_THREAD_ID"], "--message", message],
                   check=True, timeout=45)
    if cleanup.get("status") == handoff.get("status") == "passed":
        subprocess.run(["systemctl", "--user", "disable", "--now",
                        "cuteafd-hy4-flash-progress.timer"], check=True, timeout=20)
    elif waiting_for_handoff:
        subprocess.run(["systemctl", "--user", "stop", "cuteafd-hy4-flash-progress.timer"],
                       check=True, timeout=20)


if __name__ == "__main__":
    main()

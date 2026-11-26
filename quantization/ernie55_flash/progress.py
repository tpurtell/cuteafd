"""Report the maxq forward gate and pilot from receipts. Uses no model or GPU."""
import argparse
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import statistics
import subprocess
from zoneinfo import ZoneInfo

TASK = Path.home() / "Developer/quantization-runs/ernie55-flash-k4-mc"
TAIPEI = ZoneInfo("Asia/Taipei")
TARGET_LAYERS = 43
TIMER = "cuteafd-ernie55-progress.timer"


def forward_log(path):
    current, completed = None, {}
    if path.exists():
        with path.open(errors="replace") as stream:
            for line in stream:
                if not line.startswith("{"):
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("phase") != "reference-forward":
                    continue
                current = event["layer"]
                if event.get("event") == "loading":
                    if current == 0:
                        completed.clear()
                    completed.pop(current, None)
                elif event.get("event") == "passed":
                    completed[current] = event["seconds"]
    return current, completed


def latest(paths):
    return max(paths, key=lambda path: path.stat().st_mtime, default=None)


def state(run, now, active):
    current, layers = forward_log(run / "logs/preflight.log")
    pilot = latest(run.glob("pilot*/pilot.json"))
    snapshot = dict(as_of=now.isoformat(), phase="preparing reference-forward preflight",
                    layer=current, completed_target_layers=len(layers), target_layers=TARGET_LAYERS,
                    projection=None, pilot_remaining_seconds=None, pilot_completion_taipei=None,
                    preflight_target_eta=None, terminal=False)
    local = None
    for path in (run / "preflight").glob("*.json"):
        receipt = json.loads(path.read_text())
        if receipt.get("schema") == "ernie55-quant-forward-parity-v1" and receipt.get("status") == "passed":
            local = path
    if local:
        snapshot.update(phase="forward gate passed, awaiting pilot", receipt=str(local),
                        completed_target_layers=TARGET_LAYERS)
    elif current is not None or "preflight" in active:
        snapshot["phase"] = "reference-forward preflight" if "preflight" in active else "preflight stopped"
        # Layer 0 is dense. Use only finished routed layers from this attempt.
        rates = [seconds for i, seconds in layers.items() if i > 0]
        remaining = TARGET_LAYERS - len(layers)
        if len(rates) >= 2 and "preflight" in active and remaining:
            estimate = round(statistics.median(rates) * remaining)
            snapshot["preflight_target_eta"] = dict(
                remaining_seconds=estimate, completion_taipei=(now + timedelta(seconds=estimate)).isoformat(),
                range_seconds=[round(min(rates) * remaining), round(max(rates) * remaining)],
                completed_routed_layers=len(rates), basis="median seconds per completed routed layer",
                excludes=["head", "dSpark", "fixture export", "pilot"])
    if "pilot" in active:
        snapshot["phase"] = "pilot startup"
    if pilot:
        report = json.loads(pilot.read_text())
        status = report["status"]
        snapshot.update(phase=f"pilot {status}", receipt=str(pilot), projection=report.get("active"),
                        pilot_modules_started=len(report["projections"]), completed_groups=report["completed"],
                        pilot_elapsed_seconds=report["elapsed_seconds"], blocker=report.get("reason"),
                        terminal=status in ("passed", "failed", "incomplete"))
        if status == "running" and "pilot" not in active:
            snapshot["phase"] = "pilot stopped with a partial receipt"
        if status == "passed":
            snapshot.update(pilot_remaining_seconds=0,
                            pilot_completion_taipei=datetime.fromtimestamp(pilot.stat().st_mtime, TAIPEI).isoformat())
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-only", action="store_true")
    args = parser.parse_args()
    active = {phase for phase in ("preflight", "pilot") if subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", f"cuteafd-ernie55-{phase}.service"],
        timeout=10).returncode == 0}
    snapshot = state(TASK, datetime.now(TAIPEI), active)
    path = TASK / "progress-updates/latest.json"
    path.parent.mkdir(exist_ok=True)
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(snapshot, indent=2) + "\n")
    temp.replace(path)
    if args.snapshot_only:
        return
    message = (f"Scheduled 30-minute ERNIE 5.5 Flash K4 MC report for TJ. Read {path} and the task receipts. "
               "Report the current maxq §3 preflight or §11 pilot phase, layer/projection progress, blocker, "
               "measured time remaining and Asia/Taipei finish time with uncertainty. Say unavailable for "
               "unmeasured estimates. Keep dSpark, head, fixture export and pilot separate from target layer ETA. "
               "Report the terminal pilot receipt, NMSE, bpw and recovery distribution when available. "
               "This timer requests reporting only, including after reboot.")
    subprocess.run([os.environ["CODEX_CLI"], "queue", "--thread", os.environ["CODEX_THREAD_ID"],
                    "--message", message], check=True, timeout=45)
    if snapshot["terminal"]:
        subprocess.run(["systemctl", "--user", "disable", "--now", TIMER], check=True, timeout=20)


if __name__ == "__main__":
    main()

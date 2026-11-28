"""Report ERNIE preflight, pilot, or production progress from durable receipts."""
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



def receipt(path):
    try:
        return json.loads(path.read_bytes())
    except FileNotFoundError:
        return {}


def forecast(now, remaining, rates, scope):
    if not rates:
        return None
    point = round(remaining * statistics.median(rates))
    bounds = [round(remaining * min(rates)), round(remaining * max(rates))] if len(rates) >= 2 else None
    return dict(scope=scope, remaining_seconds=point, completion_taipei=(now + timedelta(seconds=point)).isoformat(),
                range_seconds=bounds, sample_count=len(rates), basis="median of recent completed work",
                excludes=["head", "dSpark", "export", "upload", "placement"])


def production_state(run, now, active, failed=()):
    prod = run / "production"
    status = receipt(prod / "status.json")
    attempt = receipt(prod / "attempt.json")
    bootstrap = receipt(run / "bootstrap/complete.json")
    packed = receipt(run / "bootstrap-packed/complete.json")
    target, draft = {}, {}
    for path in sorted((prod / "layers").glob("*.json")):
        layer = receipt(path)
        if layer.get("family") in ("layers", "mtp"):
            rows = target if layer["family"] == "layers" else draft
            rows[layer["index"]] = layer
    current = status.get("layer", 0)
    completed_routed = [r for i, r in sorted(target.items()) if i > 0 and not r.get("interrupted")
                        and (r.get("seconds") or 0) > 0]
    remaining = TARGET_LAYERS - len(target)
    eta = forecast(now, remaining, [r["seconds"] for r in completed_routed[-6:]],
                   "target layers only, current incomplete layer counted in full")
    if not eta and status.get("phase") == "expert_group_complete" and status.get("family") == "layers":
        seconds = status.get("group_seconds")
        if seconds and seconds > 0:
            done = status.get("expert_projections", 0) if current not in target else 0
            outstanding = (42 - sum(i > 0 for i in target)) * 512 * 3 - done
            eta = forecast(now, outstanding, [seconds / (32 * 3)],
                           "provisional target expert work only; excludes attention, core and layer commit")
    numerical = receipt(prod / "numerical-complete.json")
    publication = receipt(prod / "publication-complete.json") or receipt(run / "publication-complete.json")
    handoff = receipt(prod / "handoff-complete.json") or receipt(run / "handoff-complete.json")
    terminal = handoff.get("status") == "passed"
    phase = status.get("phase", "production preparation")
    if status.get("family") == "mtp":
        phase = "dSpark / " + phase
    if not status:
        if packed.get("status") == "passed":
            phase = "layer 0 input projections sealed"
        elif bootstrap.get("status") == "passed":
            phase = "layer 0 input Hessians sealed"
    if numerical.get("status") == "passed":
        phase = "export / publication"
    if publication.get("status") == "passed":
        phase = "SparkNest import / placement"
    if terminal:
        phase = "handoff complete"
    blocker = status.get("blocker") or status.get("error")
    if not terminal and not numerical and (attempt.get("status") == "failed" or status.get("status") == "failed"):
        blocker = attempt.get("detail") or status.get("error") or "numerical worker failed"
    elif not terminal and not numerical and status and "production" not in active:
        blocker = blocker or "production unit inactive; check latest receipt and boot before resuming"
    for stage in ("postproduction", "publish", "handoff", "cleanup"):
        if stage in failed:
            phase = f"{stage} failed"
            blocker = f"cuteafd-ernie55-{stage}.service failed; inspect its log and receipts"
            break
    if eta and blocker:
        eta["assumes_resume_now"] = True
    if not remaining and target:
        eta = dict(scope="target layers only", remaining_seconds=0,
                   completion_taipei=target[max(target)].get("completed_at"), range_seconds=[0, 0])
    return dict(as_of=now.isoformat(), phase=phase, worker_status=status, active_units=sorted(active),
                family=status.get("family", "layers"), layer=current, completed_target_layers=len(target),
                target_layers=TARGET_LAYERS, completed_draft_stages=len(draft), draft_stages=3,
                projection=status.get("projection") or status.get("module"),
                expert_projections=status.get("expert_projections"), bootstrap_prompts=bootstrap.get("prompts"),
                layer0_bootstrap_projections=packed.get("projections", 0), target_eta=eta,
                head_eta=None, dspark_eta=None, export_eta=None, upload_eta=None, placement_eta=None,
                pilot_receipt=str(run / "pilot/pilot.json"), blocker=blocker, terminal=terminal,
                numerical=numerical.get("status"), publication=publication.get("status"),
                handoff=handoff.get("status"))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-only", action="store_true")
    parser.add_argument("--production", action="store_true")
    args = parser.parse_args()
    active = {phase for phase in ("preflight", "pilot", "production", "publish", "handoff") if subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", f"cuteafd-ernie55-{phase}.service"],
        timeout=10).returncode == 0}
    now = datetime.now(TAIPEI)
    if args.production:
        result = subprocess.run(["systemctl", "--user", "list-units", "--state=failed", "--plain",
                                 "--no-legend", "cuteafd-ernie55-*.service"],
                                capture_output=True, text=True, check=True, timeout=10)
        failed = {line.split()[0].removeprefix("cuteafd-ernie55-").removesuffix(".service")
                  for line in result.stdout.splitlines() if line.strip()}
        snapshot = production_state(TASK, now, active, failed)
    else:
        snapshot = state(TASK, now, active)
    path = TASK / "progress-updates/latest.json"
    path.parent.mkdir(exist_ok=True)
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(snapshot, indent=2) + "\n")
    temp.replace(path)
    if args.snapshot_only:
        return
    phase = ("Report target layer and projection progress, blocker, measured time left, Asia/Taipei completion time "
             "and uncertainty. Give head, dSpark, export, upload and placement separately. "
             "Continue the already-authorized production workflow. After a reboot, report status only until TJ "
             "explicitly asks to resume. Never duplicate live or completed numerical work."
             if args.production else
             "Report the §3 or §11 status, layer/projection progress, blocker, measured remaining time and "
             "Asia/Taipei finish with uncertainty. Separate head, dSpark, fixture export and pilot from layer ETA. "
             "Report terminal pilot NMSE, bpw and recovery when available. Report only after a reboot.")
    message = (f"Scheduled 30-minute ERNIE 5.5 Flash K4 MC report for TJ. Read {path} and the task receipts. "
               f"{phase} Say unavailable for unmeasured estimates.")
    subprocess.run([os.environ["CODEX_CLI"], "queue", "--thread", os.environ["CODEX_THREAD_ID"],
                    "--message", message], check=True, timeout=45)
    if snapshot["terminal"]:
        subprocess.run(["systemctl", "--user", "disable", "--now", TIMER], check=True, timeout=20)


if __name__ == "__main__":
    main()

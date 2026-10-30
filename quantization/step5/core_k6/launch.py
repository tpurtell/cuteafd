"""Durable single-MaxQ quantization followed by CPU-only publication.

The canonical maxq hardware lock lives on raptor. An unchanged copy of the
repository's fair queue helper runs there; source, GPU work and output stay
on maxq. The GPU lease is released before the publication subprocess starts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "quantization"))
from kimi_k3.uniform_k2.storage import atomic_json

SSH_OPTIONS = ["-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
               "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3"]


def ssh(host, command):
    return ["ssh", *SSH_OPTIONS, host, shlex.join(list(map(str, command)))]


def stage_lock_helper():
    source = (REPO / "scripts/hw/hwlock.py").read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    path = Path.home() / ".cache/cuteafd/quantization/step5-k4-core-k6" / f"hwlock-{digest}.py"
    # Stage the unmodified shared helper, rather than depending on raptor's
    # currently checked out source tree or changing its shared infrastructure.
    setup = "from pathlib import Path; import sys,hashlib; p=Path(sys.argv[1]); " \
        "b=sys.stdin.buffer.read(); assert hashlib.sha256(b).hexdigest()==sys.argv[2]; " \
        "p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b)"
    subprocess.run(ssh("raptor", ["python3", "-c", setup, path, digest]), input=source,
                   check=True, timeout=60)
    return str(path)


def gpu_command(common, *, container, lock_helper, pilot=False, qualification_report=None):
    python = str(REPO / ".venv/bin/python")
    here = Path(__file__).resolve().parent
    admission = [python, str(here / "preflight.py")]
    if pilot:
        admission += ["--qualification-only"]
    elif qualification_report:
        admission += ["--qualification-report", str(qualification_report)]
    admission += ["--", *common]
    runner = here / ("pilot.py" if pilot else "run.py")
    numerical = [python, str(here / "container.py"), "--name", container, "--",
                 str(runner), *common]
    guarded = [python, str(REPO / "quantization/kimi_k3/core_k6/run_guard.py"),
               "--container", container, "--", *numerical]
    return ssh("raptor", ["python3", lock_helper, "--locks", "maxq", "--wait", "21600",
        "--preflight", shlex.join(ssh("maxq", admission)), "--preflight-timeout", "600", "--",
        "timeout", "--foreground", "--signal=TERM", "--kill-after=70",
        "25m" if pilot else "30d", *ssh("maxq", guarded)])


def supervise(path):
    record = json.loads(path.read_text())
    root=Path(record["root"])
    if (root/"handoff-complete.json").exists():
        done=json.loads((root/"handoff-complete.json").read_text())
        if done.get("status")!="passed":raise ValueError("invalid final handoff marker")
        return 0
    path.with_name(path.stem+"-exit.json").unlink(missing_ok=True)
    result = 1
    process = None
    def forward(signum, _frame):
        if process is not None and process.poll() is None:
            process.send_signal(signum)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, forward)
    with Path(record["log"]).open("a") as output:
        try:
            handoff_only=(root/"sparknest-import-complete.json").exists()
            selected=record["handoff_command"] if handoff_only else record["command"]
            if not handoff_only and record.get("resume_command") and (root/"plan.json").exists():
                selected=record["resume_command"]
            process = subprocess.Popen(selected, cwd=REPO, env=record["environment"],
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
            atomic_json(path.with_name(path.stem + "-process.json"),
                dict(pid=process.pid, supervisor_pid=os.getpid(), started=datetime.now(timezone.utc).isoformat()))
            result = process.wait()
            # An unreachable SSH hop must not leave our GPU container running.
            subprocess.run(["docker", "stop", "--time", "30", record["container"]],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
            if not handoff_only and result == 0 and record.get("publication_command"):
                process = subprocess.Popen(record["publication_command"], cwd=REPO,
                    env=dict(record["environment"], CUDA_VISIBLE_DEVICES=""), stdin=subprocess.DEVNULL,
                    stdout=output, stderr=subprocess.STDOUT)
                result = process.wait()
            if not handoff_only and result == 0 and record.get("handoff_command"):
                process = subprocess.Popen(record["handoff_command"], cwd=REPO,
                    env=dict(record["environment"],CUDA_VISIBLE_DEVICES=""),stdin=subprocess.DEVNULL,
                    stdout=output,stderr=subprocess.STDOUT)
                result=process.wait()
        finally:
            atomic_json(path.with_name(path.stem + "-exit.json"),
                dict(returncode=result, finished=datetime.now(timezone.utc).isoformat()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-snapshot", "tokens", "root"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--output", type=Path, help="pilot report")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--publish-only", action="store_true")
    parser.add_argument("--qualification-report", type=Path)
    parser.add_argument("--supervise", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.supervise:
        sys.exit(supervise(args.supervise))
    if args.pilot and (args.publish or args.publish_only or args.export):
        parser.error("the pilot cannot export or publish")
    required = ("source_snapshot", "tokens", "root", "output" if args.pilot else "export")
    if any(getattr(args, name) is None for name in required):
        parser.error("source-snapshot, tokens, root and output/export are required")
    root = args.root.expanduser().resolve()
    attempts = root / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    for old in attempts.glob("*-process.json"):
        if old.with_name(old.name.replace("-process.json", "-exit.json")).exists():
            continue
        try:
            os.kill(json.loads(old.read_text())["supervisor_pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError("a prior supervisor remains active")
    common = []
    for name in required:
        common += ["--" + name.replace("_", "-"), str(getattr(args, name).expanduser().resolve())]
    if args.resume:
        common.append("--resume")
    python = str(REPO / ".venv/bin/python")
    runner = str(Path(__file__).with_name("run.py"))
    stamp = str(time.time_ns())
    container = "step5-core-k6-" + stamp
    publication = ["timeout", "--foreground", "--signal=TERM", "--kill-after=120", "7d",
        python, runner, *common, "--publish-only"]
    if args.publish_only:
        admission = [python, str(Path(__file__).with_name("preflight.py")), "--publication-only", "--", *common]
        subprocess.run(admission, cwd=REPO, check=True, timeout=600)
        command = publication
    else:
        command = gpu_command(common, container=container, lock_helper=stage_lock_helper(),
            pilot=args.pilot, qualification_report=args.qualification_report)
    environment = {key: os.environ[key] for key in
        ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "HF_HOME") if key in os.environ}
    environment["HF_HUB_CACHE"] = str(Path.home() / ".cache/huggingface/cuteafd-local-hub")
    if args.publish_only:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    path = attempts / (stamp + ".json")
    unit = "step5-core-k6-" + stamp
    resume_command=None if args.pilot or args.publish_only else gpu_command([*common,"--resume"],
        container=container,lock_helper=stage_lock_helper(),qualification_report=args.qualification_report)
    handoff=["timeout","--foreground","--signal=TERM","--kill-after=120","12h",python,
        str(Path(__file__).with_name("handoff.py")),"--root",str(root)]
    atomic_json(path, dict(root=str(root),command=command,resume_command=resume_command,
        publication_command=publication if args.publish else None,handoff_command=handoff if args.publish or args.publish_only else None,
        environment=environment, container=container, log=str(attempts / (stamp + ".log")), unit=unit,
        pilot=args.pilot, publish=args.publish, publication_only=args.publish_only))
    supervisor = [python, str(Path(__file__).resolve()), "--supervise", str(path)]
    # Install an enabled user service so a reboot can resume from the latest
    # durable frontier. Each supervisor attempt remains one blocking run.
    units=Path.home()/".config/systemd/user";units.mkdir(parents=True,exist_ok=True)
    unit_path=units/(unit+".service")
    content="\n".join(["[Unit]","Description=Step-5 quantization and publication",
        "After=basic.target","StartLimitIntervalSec=600","StartLimitBurst=3","[Service]",
        "Type=simple","WorkingDirectory="+str(REPO),
        "ExecStart="+shlex.join(supervisor),"Restart=on-failure","RestartSec=30",
        "TimeoutStopSec=180","[Install]","WantedBy=default.target",""])
    unit_path.write_text(content)
    subprocess.run(["systemctl","--user","daemon-reload"],check=True)
    subprocess.run(["systemctl","--user","enable","--now",unit+".service"],check=True)
    atomic_json(root / "latest-attempt.json", dict(attempt=stamp, launch=str(path),
        unit=unit, log=str(attempts / (stamp + ".log"))))
    print(json.dumps(dict(unit=unit, launch=str(path), log=str(attempts / (stamp + ".log")))))


if __name__ == "__main__":
    main()

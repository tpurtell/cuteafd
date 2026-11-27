#!/usr/bin/env python3
"""Detach a bounded local quantize/validate/publish job with durable attempts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "quantization"))
from kimi_k3.uniform_k2.storage import atomic_json


def now():
    return datetime.now(timezone.utc).isoformat()


def supervise(path):
    launch = json.loads(path.read_text())
    result = 1
    with Path(launch["log"]).open("a") as output:
        try:
            process = subprocess.Popen(launch["command"], cwd=REPO, env=launch["environment"],
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
            atomic_json(path.with_name(path.stem + "-process.json"),
                dict(pid=process.pid, supervisor_pid=os.getpid(), started=now()))
            result = process.wait()
            # A timeout/disconnect must not release the GPU with our container
            # still running. Stopping a nonexistent container is harmless.
            subprocess.run(["docker", "stop", "--time", "30", launch["container"]],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
            if result == 0 and launch.get("publication_command"):
                result = subprocess.call(launch["publication_command"], cwd=REPO,
                    env=dict(launch["environment"], CUDA_VISIBLE_DEVICES=""),
                    stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
        except BaseException:
            result = 1
            raise
        finally:
            atomic_json(path.with_name(path.stem + "-exit.json"), dict(returncode=result, finished=now()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-snapshot", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--export", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--supervise", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.supervise:
        sys.exit(supervise(args.supervise))
    root = args.root.expanduser().resolve()
    attempts = root / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    for path in attempts.glob("*-process.json"):
        if path.with_name(path.name.replace("-process.json", "-exit.json")).exists():
            continue
        prior = json.loads(path.read_text())
        try:
            os.kill(prior["supervisor_pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError("the prior supervisor remains active")
    python = str(REPO / ".venv/bin/python")
    runner = str(Path(__file__).with_name("run.py"))
    common = ["--source-snapshot", str(args.source_snapshot.expanduser().resolve()),
        "--tokens", str(args.tokens.expanduser().resolve()), "--root", str(root),
        "--export", str(args.export.expanduser().resolve())]
    if args.resume:
        common.append("--resume")
    admission = [python, str(Path(__file__).with_name("preflight.py")), "--", *common]
    preflight = shlex.join(admission)
    subprocess.run(admission, cwd=REPO, check=True)
    attempt = str(time.time_ns())
    container = f"kimi-core-k6-{attempt}"
    gpu_command = [python, str(Path(__file__).with_name("container.py")), "--name", container,
        "--", runner, *common]
    # The hardware lock remains inside a supervisor that stops its container
    # on every exit; uploading starts only after this wrapper releases it.
    guarded = [python, str(Path(__file__).with_name("run_guard.py")),
        "--container", container, "--", *gpu_command]
    command = [str(REPO / "peerswarm/target/release/peerswarm"), "--root", str(REPO), "resource", "exec", "--locks", "gpu1", "--wait", "60",
        "--preflight", preflight, "--preflight-timeout", "300", "--",
        "timeout", "--foreground", "--signal=TERM", "--kill-after=120", "30d", *guarded]
    environment = {key: os.environ[key] for key in
        ("PATH", "LD_LIBRARY_PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "HF_HOME", "HF_HUB_CACHE")
        if key in os.environ}
    # The normal hub is a sparknest symlink. Publication's local snapshot must
    # hardlink our NVMe export, without making a second 681GiB checkpoint or
    # trying unsupported cross-filesystem links into that network mount.
    environment["HF_HUB_CACHE"] = str(Path.home() / ".cache/huggingface/cuteafd-local-hub")
    publication = (["timeout", "--foreground", "--signal=TERM", "--kill-after=120", "7d",
                    python, runner, *common, "--publish-only"] if args.publish else None)
    path = attempts / f"{attempt}.json"
    atomic_json(path, dict(schema="cuteafd-kimi-core-local-launch-v1", command=command,
        publication_command=publication, environment=environment, container=container,
        log=str(attempts / f"{attempt}.log"), created=now(), resume=args.resume, publish=args.publish))
    supervisor = [python, str(Path(__file__).resolve()), *common, "--supervise", str(path)]
    with (attempts / f"{attempt}-supervisor.log").open("a") as output:
        process = subprocess.Popen(supervisor, cwd=REPO, stdin=subprocess.DEVNULL,
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    atomic_json(root / "latest-attempt.json", dict(attempt=attempt, pid=process.pid, launch=str(path)))
    print(json.dumps(dict(supervisor_pid=process.pid, launch=str(path), log=str(attempts/f"{attempt}.log"))))


if __name__ == "__main__":
    main()

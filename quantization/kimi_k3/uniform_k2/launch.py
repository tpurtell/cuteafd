#!/usr/bin/env python3
"""Detach one complete quantization/export/publication run on this host."""
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
        launch = json.loads(args.supervise.read_text())
        with Path(launch["log"]).open("a") as output:
            process = subprocess.Popen(launch["command"], cwd=REPO, env=launch["environment"],
                                       stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
            atomic_json(args.supervise.with_name(args.supervise.stem + "-process.json"),
                        dict(pid=process.pid, supervisor_pid=os.getpid(), started=now()))
            result = process.wait()
            if result == 0 and launch.get("publication_command"):
                publication_environment = dict(launch["environment"], CUDA_VISIBLE_DEVICES="")
                result = subprocess.call(launch["publication_command"], cwd=REPO, env=publication_environment,
                                         stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
        atomic_json(args.supervise.with_name(args.supervise.stem + "-exit.json"),
                    dict(returncode=result, finished=now()))
        sys.exit(result)
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    attempts = root / "attempts"
    attempts.mkdir(exist_ok=True)
    for old in attempts.glob("*-process.json"):
        exit_path = old.with_name(old.name.replace("-process.json", "-exit.json"))
        if not exit_path.exists():
            process = json.loads(old.read_text())
            try:
                os.kill(process["supervisor_pid"], 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError("a prior supervisor remains active; inspect its existing attempt")
    # Admission is read-only. Refuse dirty numerical code or unqualified forks
    # before writing the immutable production plan in the child process.
    command = [str(REPO / ".venv/bin/python"), str(Path(__file__).with_name("run.py")),
               "--source-snapshot", str(args.source_snapshot.expanduser().resolve()),
               "--tokens", str(args.tokens.expanduser().resolve()), "--root", str(root),
               "--export", str(args.export.expanduser().resolve())]
    if args.resume:
        command.append("--resume")
    publication_command = (["timeout", "--foreground", "--signal=TERM", "7d", command[0],
                            str(Path(__file__).with_name("publish.py")), *command[2:], "--publish-only"]
                           if args.publish else None)
    # GPU ownership ends when the quantizer process exits. Uploading runs in a
    # separate process with no CUDA device visible and no hardware lock held.
    # The hardware lock is taken through the fair queue (scripts/hw/hwlock.py), never
    # a direct flock that would overtake older gpu1 tickets (#2596). The read-only
    # plan admission runs once the lock is granted, before the quantizer starts.
    admission = shlex.join([*command, "--plan-only"])
    command = [str(REPO / "peerswarm/target/release/peerswarm"), "--root", str(REPO), "resource", "exec", "--locks", "gpu1", "--wait", "60",
               "--preflight", admission, "--preflight-at-grant", "--preflight-timeout", "300", "--",
               "timeout", "--foreground", "--signal=TERM", "30d", *command]
    # Persist only execution settings; credentials stay in the existing HF
    # credential store and never appear in launch data or commands.
    environment = {name: os.environ[name] for name in
                   ("PATH", "LD_LIBRARY_PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "HF_HOME", "HF_HUB_CACHE")
                   if name in os.environ}
    environment.update(CUDA_VISIBLE_DEVICES="0", CUDA_HOME="/usr/local/cuda-13.4",
                       TORCH_CUDA_ARCH_LIST="12.0", CC="/usr/bin/gcc", CXX="/usr/bin/g++", MAX_JOBS="4",
                       FLA_DISABLE_TENSOR_CACHE="1", PYTORCH_ALLOC_CONF="expandable_segments:True",
                       GPTQMODEL_EXLLAMAV3_BUILD_ROOT=str(Path.home() / ".cache/cuteafd/builds/kimi-k3/exllamav3"),
                       OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
    attempt = str(time.time_ns())
    path = attempts / f"{attempt}.json"
    atomic_json(path, dict(schema="cuteafd-local-quant-launch-v1", command=command, publication_command=publication_command,
                          environment=environment,
                          log=str(attempts / f"{attempt}.log"), created=now(), resume=args.resume, publish=args.publish))
    supervisor_command = [str(REPO / ".venv/bin/python"), str(Path(__file__).resolve()),
                          "--source-snapshot", str(args.source_snapshot), "--tokens", str(args.tokens),
                          "--root", str(root), "--export", str(args.export), "--supervise", str(path)]
    with (attempts / f"{attempt}-supervisor.log").open("a") as output:
        process = subprocess.Popen(supervisor_command, cwd=REPO, stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    atomic_json(root / "latest-attempt.json", dict(attempt=attempt, pid=process.pid, launch=str(path)))
    print(json.dumps(dict(supervisor_pid=process.pid, launch=str(path), log=str(attempts / f"{attempt}.log"))))


if __name__ == "__main__":
    main()

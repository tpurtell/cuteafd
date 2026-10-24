#!/usr/bin/env python3
"""Detach a bounded local GLM quantization or explicitly requested publication."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/"quantization"))
from kimi_k3.uniform_k2.storage import atomic_json
from kimi_k3.core_k6.launch import supervise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("source-snapshot","tokens","root","export"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--publish-only",action="store_true")
    parser.add_argument("--supervise",type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.supervise:
        sys.exit(supervise(args.supervise))
    root=args.root.expanduser().resolve();attempts=root/"attempts";attempts.mkdir(parents=True,exist_ok=True)
    for path in attempts.glob("*-process.json"):
        if path.with_name(path.name.replace("-process.json","-exit.json")).exists():
            continue
        previous=json.loads(path.read_text())
        try:
            os.kill(previous["supervisor_pid"],0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError("the previous GLM supervisor remains active")
    python=str(REPO/".venv/bin/python")
    common=[]
    for name in ("source-snapshot","tokens","root","export"):
        common += ["--"+name,str(getattr(args,name.replace("-","_")).expanduser().resolve())]
    if args.resume:
        common.append("--resume")
    preflight=[python,str(Path(__file__).with_name("preflight.py"))]
    if args.publish_only:
        preflight.append("--publication-only")
    preflight += ["--",*common]
    subprocess.run(preflight,cwd=REPO,check=True)
    stamp=str(time.time_ns());container="glm53-core-k6-"+stamp
    runner=str(Path(__file__).with_name("run.py"))
    if args.publish_only:
        # This branch is only reached by the user's separate publication
        # request. No GPU admission or CUDA container is used for uploading.
        command=["timeout","--foreground","--signal=TERM","--kill-after=120","7d",
                 python,runner,*common,"--publish-only"]
    else:
        numerical=[python,str(REPO/"quantization/kimi_k3/core_k6/container.py"),"--name",container,
            "--memory-gib","60","--",runner,*common]
        guarded=[python,str(REPO/"quantization/kimi_k3/core_k6/run_guard.py"),"--container",container,"--",*numerical]
        command=[python,str(REPO/"scripts/hw/hwlock.py"),"--locks","gpu1","--wait","60",
            "--preflight",shlex.join(preflight),"--preflight-timeout","300","--",
            "timeout","--foreground","--signal=TERM","--kill-after=120","30d",*guarded]
    environment={key:os.environ[key] for key in
        ("PATH","HOME","USER","LOGNAME","LANG","LC_ALL","LD_LIBRARY_PATH","HF_HOME") if key in os.environ}
    environment["HF_HUB_CACHE"]=str(Path.home()/".cache/huggingface/cuteafd-local-hub")
    if args.publish_only:
        environment["CUDA_VISIBLE_DEVICES"]=""
    path=attempts/(stamp+".json")
    atomic_json(path,dict(command=command,environment=environment,container=container,
        log=str(attempts/(stamp+".log")),publication_only=args.publish_only,resume=args.resume))
    supervisor=[python,str(Path(__file__).resolve()),*common,"--supervise",str(path)]
    with (attempts/(stamp+"-supervisor.log")).open("a") as output:
        process=subprocess.Popen(supervisor,cwd=REPO,stdin=subprocess.DEVNULL,stdout=output,
            stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(root/"latest-attempt.json",dict(pid=process.pid,attempt=stamp,launch=str(path),log=str(attempts/(stamp+".log"))))
    print(json.dumps(dict(supervisor_pid=process.pid,launch=str(path))),flush=True)


if __name__=="__main__":
    main()

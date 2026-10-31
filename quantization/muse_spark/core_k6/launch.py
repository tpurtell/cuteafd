"""Durable GPU0 lease, bounded container cleanup, then authorized CPU publication."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'quantization'))
from kimi_k3.uniform_k2.storage import atomic_json


def supervise(path):
    record=json.loads(path.read_text());result=1;process=None
    def forward(signum,_frame):
        if process is not None and process.poll() is None:process.send_signal(signum)
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,forward)
    with Path(record['log']).open('a') as output:
        try:
            process=subprocess.Popen(record['command'],cwd=REPO,env=record['environment'],stdin=subprocess.DEVNULL,
                stdout=output,stderr=subprocess.STDOUT)
            atomic_json(path.with_name(path.stem+'-process.json'),dict(pid=process.pid,supervisor_pid=os.getpid(),
                started=datetime.now(timezone.utc).isoformat()))
            result=process.wait()
            # The inner guard stops this owned container before hwlock exits.
            # This second stop also covers a failed process/connection boundary.
            subprocess.run(['docker','stop','--time','30',record['container']],
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=60)
            if result==0 and record.get('publication_command'):
                process=subprocess.Popen(record['publication_command'],cwd=REPO,
                    env=dict(record['environment'],CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,
                    stdout=output,stderr=subprocess.STDOUT)
                result=process.wait()
        finally:
            atomic_json(path.with_name(path.stem+'-exit.json'),dict(returncode=result,
                finished=datetime.now(timezone.utc).isoformat()))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-snapshot','tokens','root','export','output','qualification-report','source-audit','native-reference-root'):
        parser.add_argument('--'+name,type=Path)
    for name in ('pilot','resume','publish','publish-only'):parser.add_argument('--'+name,action='store_true')
    parser.add_argument('--supervise',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.supervise:sys.exit(supervise(args.supervise))
    required=('source_snapshot','tokens','root','output' if args.pilot else 'export')
    if any(getattr(args,k)is None for k in required):parser.error('source/tokens/root and output/export required')
    if args.pilot and (args.publish or args.publish_only):parser.error('pilot cannot publish')
    root=args.root.expanduser().resolve();attempts=root/'attempts';attempts.mkdir(parents=True,exist_ok=True)
    for old in attempts.glob('*.json'):
        if old.stem.endswith(('-exit','-process')):continue
        prior=json.loads(old.read_text())
        if old.with_name(old.stem+'-exit.json').exists():continue
        unit=prior.get('unit')
        if unit and subprocess.run(['systemctl','--user','is-active','--quiet',unit]).returncode==0:
            raise ValueError('prior durable supervisor remains active')
    common=[]
    for name in required:common+=['--'+name.replace('_','-'),str(getattr(args,name).expanduser().resolve())]
    if args.resume:common+=['--resume']
    if args.native_reference_root:
        if not args.pilot:parser.error('native reference reuse is pilot-only')
        common+=['--native-reference-root',str(args.native_reference_root.expanduser().resolve())]
    python=str(REPO/'.venv/bin/python');here=Path(__file__).resolve().parent
    stamp=str(time.time_ns());container='muse-quant-'+stamp;unit=container
    audit=args.source_audit or root.parent/'source-audit.json'
    admission=[python,str(here/'preflight.py'),'--source-audit',str(audit)]
    if args.pilot:admission+=['--pilot']
    else:admission+=['--qualification-report',str(args.qualification_report or root.parent/'pilot/qualification.json')]
    admission+=['--',*common]
    numerical=[python,str(here/'container.py'),'--name',container,'--',str(here/('pilot.py' if args.pilot else 'run.py')),*common]
    guarded=[python,str(REPO/'quantization/kimi_k3/core_k6/run_guard.py'),'--container',container,'--',*numerical]
    command=[python,str(REPO/'scripts/hw/hwlock.py'),'--locks','gpu0','--wait','21600',
        '--preflight',shlex.join(admission),'--preflight-timeout','600','--',
        'timeout','--foreground','--signal=TERM','--kill-after=90','1h' if args.pilot else '30d',*guarded]
    # Publication validates CPU-side receipts/provenance directly; it never
    # recomputes a GPU runtime plan under the host's CPU Torch installation.
    publication=['timeout','--foreground','--signal=TERM','--kill-after=120','7d',python,
        str(here/'publish_job.py'),'--root',str(root),'--export',str(args.export)] if not args.pilot else None
    if args.publish_only:command=publication
    environment={k:os.environ[k] for k in ('PATH','HOME','USER','LOGNAME','LANG','LC_ALL','HF_HOME') if k in os.environ}
    environment['HF_HUB_CACHE']=str(root.parent/'local-hub')
    if args.publish_only:environment['CUDA_VISIBLE_DEVICES']=''
    path=attempts/(stamp+'.json');log=attempts/(stamp+'.log')
    atomic_json(path,dict(command=command,publication_command=publication if args.publish and not args.publish_only else None,
        environment=environment,container=container,unit=unit,log=str(log),pilot=args.pilot,resume=args.resume,publish=args.publish))
    subprocess.run(['systemd-run','--user','--collect','--unit='+unit,'--expand-environment=no',
        '--working-directory='+str(REPO),*[f'--setenv={k}={v}' for k,v in environment.items()],
        python,str(Path(__file__).resolve()),'--supervise',str(path)],check=True)
    atomic_json(root/'latest-attempt.json',dict(attempt=stamp,launch=str(path),unit=unit,log=str(log)))
    print(json.dumps(dict(unit=unit,launch=str(path),log=str(log))),flush=True)

if __name__=='__main__':main()

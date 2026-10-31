"""Durable GPU1 quantization, bounded OOM recovery, and authorized CPU upload."""
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
    record=json.loads(path.read_text())
    result=1
    process=None
    interrupted=False
    failure_message=None
    def forward(signum,frame):
        nonlocal interrupted
        interrupted=True
        if process is not None and process.poll() is None:
            process.send_signal(signum)
    for sig in (signal.SIGHUP,signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,forward)
    with Path(record['log']).open('a') as output:
        try:
            command=record['command']
            for recovery in range(3):
                atomic_json(path.with_name(path.stem+'-process.json'),dict(supervisor_pid=os.getpid(),
                    started=datetime.now(timezone.utc).isoformat(),oom_recovery=recovery))
                started_ns=time.time_ns()
                process=subprocess.Popen(command,cwd=REPO,env=record['environment'],stdin=subprocess.DEVNULL,
                    stdout=output,stderr=subprocess.STDOUT)
                result=process.wait()
                # The inner guard has already torn down this task's container
                # before hwlock releases GPU1; this is idempotent final cleanup.
                subprocess.run(['docker','stop','--time','30',record['container']],stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,timeout=60)
                state_result=subprocess.run(['docker','inspect','--format','{{json .State}}',record['container']],
                    capture_output=True,text=True,timeout=30)
                container_state=json.loads(state_result.stdout) if state_result.returncode==0 else {}
                if container_state:
                    atomic_json(path.with_name(path.stem+f'-container-{recovery}.json'),container_state)
                    subprocess.run(['docker','rm',record['container']],stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,timeout=30)
                if interrupted:
                    result=result or 130
                    break
                if result==0 or record['pilot'] or record['publication_only'] or recovery==2:
                    break
                failure_path=Path(record['root'])/'failure.json'
                failure=json.loads(failure_path.read_text()) if failure_path.exists() else {}
                fresh=(failure.get('time',0)>=started_ns and failure.get('attempt')==record['container'])
                category=failure.get('category') if fresh else None
                if category=='numerical' or not (category=='oom' or container_state.get('OOMKilled') is True):
                    break
                if '--resume' not in command:
                    command=[*command,'--resume']
                output.write(json.dumps(dict(event='oom_resume',attempt=recovery+1,
                    note='immutable producer unchanged; resume only committed receipt-verified work'))+'\n')
                output.flush()
            if result==0 and not interrupted and record.get('publication_command'):
                process=subprocess.Popen(record['publication_command'],cwd=REPO,
                    env=dict(record['environment'],CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,
                    stdout=output,stderr=subprocess.STDOUT)
                result=process.wait()
        except BaseException as error:
            result=result or 1
            failure_message=f'{type(error).__name__}: {error}'
            raise
        finally:
            atomic_json(path.with_name(path.stem+'-exit.json'),dict(returncode=result,
                error=failure_message,
                finished=datetime.now(timezone.utc).isoformat()))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-snapshot','tokens','root','export','output','source-audit','qualification-report'):
        parser.add_argument('--'+name,type=Path)
    for name in ('pilot','resume','publish','publish-only'):
        parser.add_argument('--'+name,action='store_true')
    parser.add_argument('--supervise',type=Path)
    args=parser.parse_args()
    if args.supervise:
        sys.exit(supervise(args.supervise))
    required=('source_snapshot','tokens','root','output' if args.pilot else 'export','source_audit')
    if any(getattr(args,key) is None for key in required):
        parser.error('source-snapshot, tokens, root, source-audit and output/export required')
    if args.pilot and (args.publish or args.publish_only):
        parser.error('pilot does not publish')
    if not args.pilot and not args.publish_only and args.qualification_report is None:
        parser.error('production requires qualification-report')
    root=args.root.expanduser().resolve()
    attempts=root/'attempts'
    attempts.mkdir(parents=True,exist_ok=True)
    for path in attempts.glob('*-process.json'):
        if path.with_name(path.name.replace('-process.json','-exit.json')).exists():
            continue
        try:
            os.kill(json.loads(path.read_text())['supervisor_pid'],0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError('previous supervisor remains active')
    common=[]
    for key in ('source_snapshot','tokens','root','output' if args.pilot else 'export'):
        common += ['--'+key.replace('_','-'),str(getattr(args,key).expanduser().resolve())]
    if args.resume:
        common += ['--resume']
    python=str(REPO/'.venv/bin/python')
    here=Path(__file__).resolve().parent
    admission=[python,str(here/'preflight.py'),'--source-audit',str(args.source_audit.resolve())]
    if args.pilot:
        admission+=['--pilot']
    elif args.publish_only:
        admission+=['--publication-only']
    else:
        admission+=['--qualification-report',str(args.qualification_report.resolve())]
    admission+=['--',*common]
    subprocess.run(admission,cwd=REPO,check=True,timeout=600)
    stamp=str(time.time_ns())
    container='mistral-core-k6-'+stamp
    publication=['timeout','--foreground','--signal=TERM','--kill-after=120','7d',python,
                 str(here/'run.py'),*common,'--publish-only']
    if args.publish_only:
        command=publication
    else:
        numerical=[python,str(here/'container.py'),'--name',container,'--',
                   str(here/('pilot.py' if args.pilot else 'run.py')),*common]
        guarded=[python,str(REPO/'quantization/kimi_k3/core_k6/run_guard.py'),'--container',container,'--',*numerical]
        command=[python,str(REPO/'scripts/hw/hwlock.py'),'--locks','gpu1','--wait','21600',
                 '--preflight',shlex.join(admission),'--preflight-timeout','600','--',
                 'timeout','--foreground','--signal=TERM','--kill-after=120','1h' if args.pilot else '30d',*guarded]
    environment={key:os.environ[key] for key in ('PATH','HOME','USER','LOGNAME','LANG','LC_ALL','HF_HOME') if key in os.environ}
    environment['CUDA_VISIBLE_DEVICES']=''
    environment['TMPDIR']=str(Path.home()/'.cache/cuteafd/builds/mistral-quant/tmp')
    path=attempts/(stamp+'.json')
    unit=container
    atomic_json(path,dict(command=command,publication_command=publication if args.publish else None,
        container=container,root=str(root),environment=environment,log=str(attempts/(stamp+'.log')),unit=unit,
        pilot=args.pilot,publication_only=args.publish_only,publish=args.publish))
    supervisor=[python,str(Path(__file__).resolve()),'--supervise',str(path)]
    subprocess.run(['systemd-run','--user','--collect','--unit='+unit,'--expand-environment=no',
        '--working-directory='+str(REPO),*[f'--setenv={key}={value}' for key,value in environment.items()],
        *supervisor],check=True)
    atomic_json(root/'latest-attempt.json',dict(attempt=stamp,unit=unit,launch=str(path),log=str(attempts/(stamp+'.log'))))
    print(json.dumps(dict(unit=unit,launch=str(path),log=str(attempts/(stamp+'.log')))),flush=True)

if __name__=='__main__':
    main()

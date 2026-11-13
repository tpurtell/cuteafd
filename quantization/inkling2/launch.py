"""Durable task worker with fair remote hardware lease and owned-container teardown."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

from .container import TASK, BUILD, REPO
from .recipe import BASE_REVISION

PYTHON = str(TASK/'environment/bin/python')
SOURCE = Path('/mnt/sparknest/hf-home/hub/models--thinkingmachines--Inkling-2-Small-MXFP8/snapshots')/BASE_REVISION


def maxq_command(arguments):
    return ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'maxq',
        shlex.join(['env', 'PYTHONPATH='+str(REPO/'quantization'), *map(str, arguments)])]


def worker(mode, output_io='legacy'):
    from .container import command
    name = 'cuteafd-inkling2-'+mode
    arguments = ['-m', 'inkling2.pilot', '--source', str(SOURCE), '--output', str(TASK/'pilot.json')]
    if mode == 'production':
        module = 'inkling2.io_resume' if output_io == 'sequential' else 'inkling2.run'
        arguments = ['-m', module, '--source', str(SOURCE), '--corpus', str(TASK/'prepared-corpus'),
            '--root', str(TASK/'production'), '--output', str(TASK/'output')]
    elif mode == 'build':
        code = 'from inkling2.numerical import fork_module; fork_module("exllamav3.ext").prewarm_exllamav3_extension()'
        # Build-slot owns only compilation; pilot/runtime do not retain it.
        arguments = ['-c', 'import subprocess,sys; subprocess.run(['+
            repr(str(REPO/'scripts/build/build-slot.sh'))+',sys.executable,"-c",'+repr(code)+'],check=True)']
    elif mode != 'pilot':
        raise ValueError('unknown worker mode')
    if mode == 'production':
        prewarm = 'from inkling2.numerical import fork_module; fork_module("exllamav3.ext").prewarm_exllamav3_extension()'
        run_arguments = repr(arguments)
        arguments = ['-c', 'import subprocess,sys,runpy; subprocess.run(['+
            repr(str(REPO/'scripts/build/build-slot.sh'))+',sys.executable,"-c",'+repr(prewarm)+
            '],check=True);sys.argv='+run_arguments+'[1:];runpy.run_module('+repr(module)+',run_name="__main__")']
    argv = command(arguments, name=name, fork=Path.home()/'Developer/GPTQModel' if mode != 'production' else None)
    child = None
    def terminate(signum, frame):
        raise InterruptedError('task worker received signal ' + str(signum))
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, terminate)
    try:
        child = subprocess.Popen(argv)
        return child.wait()
    finally:
        # Normal SSH login has fresh Docker supplementary groups. Never stop an
        # unrelated container or release the maxq lease with ours still alive.
        stopped = subprocess.run(['docker', 'container', 'inspect', '--format', '{{.State.Running}}', name],
                                 capture_output=True, text=True, timeout=20)
        if stopped.returncode == 0 and stopped.stdout.strip() == 'true':
            subprocess.run(['docker', 'stop', '--time', '30', name], check=True, timeout=45,
                           stdout=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['build', 'pilot', 'production'], required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--output-io',choices=['legacy','sequential'],default='legacy')
    args = parser.parse_args()
    if args.worker:
        return worker(args.mode,args.output_io)
    root = TASK/'production'
    if args.mode == 'production' and (root/'quantization-complete.json').exists():
        report = json.loads((root/'quantization-complete.json').read_text())
        if report.get('status') != 'passed':
            raise ValueError('unrecognized completion receipt; diagnose before reuse')
        return 0
    preflight = maxq_command([PYTHON, '-m', 'inkling2.preflight', '--source', SOURCE,
                              *(['--production'] if args.mode == 'production' else [])])
    wait = 1800 if args.mode == 'production' else 300
    runtime = 30*24*3600 if args.mode == 'production' else (900 if args.mode == 'build' else 660)
    remote = ['timeout', '--signal=TERM', '--kill-after=90', str(wait+runtime+900), 'python3',
        '/home/tj/Developer/cuteafd/scripts/hw/hwlock.py', '--locks', 'maxq', '--wait', str(wait),
        '--preflight', shlex.join(preflight), '--preflight-timeout', '180', '--',
        *maxq_command(['timeout', '--signal=TERM', '--kill-after=60', str(runtime),
            PYTHON, '-m', 'inkling2.launch', '--worker', '--mode', args.mode,
            '--output-io',args.output_io])]
    return subprocess.call(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                            'raptor', shlex.join(remote)])


if __name__ == '__main__':
    raise SystemExit(main())

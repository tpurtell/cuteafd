"""Durable maxq-only job using the existing hardware queue and supervisor."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'quantization'))
from kimi_k3.uniform_k2.storage import atomic_json
from step5.core_k6.launch import ssh, supervise
from kimi_k31.k2_core_k6.container import RUNTIME


def stage_lock_helper():
    source = (REPO / 'scripts/hw/hwlock.py').read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    path = Path.home() / '.cache/cuteafd/quantization/kimi-k31-k2-core-k6' / f'hwlock-{digest}.py'
    setup = ('from pathlib import Path; import sys,hashlib; p=Path(sys.argv[1]); '
             'b=sys.stdin.buffer.read(); assert hashlib.sha256(b).hexdigest()==sys.argv[2]; '
             'p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b)')
    subprocess.run(ssh('raptor', ['python3', '-c', setup, path, digest]), input=source,
                   check=True, timeout=60)
    return str(path)


def gpu_command(common, *, container, helper, pilot=False, qualification=None):
    here = Path(__file__).resolve().parent
    admission = [str(RUNTIME), str(here / 'preflight.py')]
    if pilot:
        admission += ['--qualification-only']
    elif qualification:
        admission += ['--qualification-report', str(qualification)]
    admission += ['--', *common]
    runner = here / ('pilot.py' if pilot else 'run.py')
    command = [str(RUNTIME), str(here / 'container.py'), '--name', container, '--', str(runner), *common]
    guarded = [str(RUNTIME), str(REPO / 'quantization/kimi_k3/core_k6/run_guard.py'),
               '--container', container, '--', *command]
    return ssh('raptor', ['python3', helper, '--locks', 'maxq', '--wait', '21600',
        '--preflight', shlex.join(ssh('maxq', admission)), '--preflight-timeout', '900', '--',
        'timeout', '--foreground', '--signal=TERM', '--kill-after=90',
        '25m' if pilot else '30d', *ssh('maxq', guarded)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-snapshot', 'tokens', 'root', 'export', 'qualification-report'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--publish', action='store_true')
    p.add_argument('--publish-only', action='store_true')
    p.add_argument('--supervise', type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.supervise:
        sys.exit(supervise(args.supervise))
    required = ('source_snapshot', 'tokens', 'root') + (() if args.pilot else ('export',))
    if any(getattr(args, key) is None for key in required):
        p.error('source-snapshot, tokens, root and production export required')
    if args.pilot and (args.publish or args.publish_only or args.resume):
        p.error('pilot cannot publish or resume a production run')
    if not args.pilot and not args.publish_only and not args.qualification_report:
        p.error('production requires the passing short qualification report')
    root = args.root.expanduser().resolve()
    attempts = root / 'attempts'
    attempts.mkdir(parents=True, exist_ok=True)
    for old in attempts.glob('*-process.json'):
        if old.with_name(old.name.replace('-process.json', '-exit.json')).exists():
            continue
        try:
            os.kill(json.loads(old.read_text())['supervisor_pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('a prior supervisor remains active')
    common = []
    for key in required:
        common += ['--' + key.replace('_', '-'), str(getattr(args, key).expanduser().resolve())]
    if args.resume:
        common.append('--resume')
    stamp = str(time.time_ns())
    unit = 'kimi-k31-quant-' + stamp
    container = unit
    here = Path(__file__).resolve().parent
    publication = ['timeout', '--foreground', '--signal=TERM', '--kill-after=120', '7d',
        str(RUNTIME), str(here / 'run.py'), *common, '--publish-only']
    helper = None if args.publish_only else stage_lock_helper()
    command = publication if args.publish_only else gpu_command(common, container=container,
        helper=helper, pilot=args.pilot, qualification=args.qualification_report)
    resume_command = None if args.pilot or args.publish_only else gpu_command(
        [*common, '--resume'], container=container, helper=helper, qualification=args.qualification_report)
    environment = {key: os.environ[key] for key in
        ('PATH', 'HOME', 'USER', 'LOGNAME', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH') if key in os.environ}
    environment['HF_HOME'] = '/mnt/sparknest/hf-home'
    environment['HF_MODULES_CACHE'] = str(root.parent / 'kernel-cache/hf-modules')
    path = attempts / (stamp + '.json')
    atomic_json(path, dict(root=str(root), command=command, resume_command=resume_command,
        publication_command=publication if args.publish else None, handoff_command=None,
        environment=environment, container=container, log=str(attempts / (stamp + '.log')),
        unit=unit, pilot=args.pilot, publish=args.publish, publication_only=args.publish_only))
    supervisor = [str(RUNTIME), str(Path(__file__).resolve()), '--supervise', str(path)]
    units = Path.home() / '.config/systemd/user'
    units.mkdir(parents=True, exist_ok=True)
    (units / (unit + '.service')).write_text('\n'.join([
        '[Unit]', 'Description=Kimi K3.1 quantization and publication',
        'After=basic.target', 'StartLimitIntervalSec=600', 'StartLimitBurst=3',
        '[Service]', 'Type=simple', 'WorkingDirectory=' + str(REPO),
        'ExecStart=' + shlex.join(supervisor), 'Restart=' + ('no' if args.pilot else 'on-failure'),
        'RestartSec=120', 'TimeoutStopSec=180', '[Install]', 'WantedBy=default.target', '']))
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
    subprocess.run(['systemctl', '--user', 'enable', '--now', unit + '.service'], check=True)
    atomic_json(root / 'latest-attempt.json', dict(attempt=stamp, unit=unit, launch=str(path),
                                                 log=str(attempts / (stamp + '.log'))))
    print(json.dumps(dict(unit=unit, launch=str(path), log=str(attempts / (stamp + '.log')))))


if __name__ == '__main__':
    main()

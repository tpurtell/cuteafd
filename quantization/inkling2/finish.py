"""CPU-only publication and terminal native storage/cleanup service steps."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

from .container import TASK, BUILD
from .publish import publish, load
from .handoff import handoff, retire


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=['publish', 'handoff'])
    args = parser.parse_args()
    root, output = TASK/'production', TASK/'output'
    numerical = load(root/'quantization-complete.json')
    if numerical.get('status') != 'passed':
        raise ValueError('numerical production has not completed')
    if args.step == 'publish':
        if not (root/'publication-complete.json').exists():
            packet = load(output/'package-state.json')
            if (packet['identity'].get('schema') != 'cuteafd-core-package-v1'
                    or packet['identity'].get('family') != 'inkling2'
                    or any(packet['identity'].get(k) != v for k,v in numerical['identity'].items() if k != 'schema')):
                raise ValueError('publication package differs from numerical completion')
        published = publish(output, root)
        print(json.dumps(dict(status='published', url=published['url'], revision=published['revision'])), flush=True)
        return
    complete = handoff(root, TASK/'hf-stage')
    cleanup = retire(root, intermediates=[root/name for name in ('packed','activations','target-taps','routes')],
                     build_root=BUILD)
    # Deliver the terminal receipts to the same progress conversation before
    # retiring its timer; otherwise a fast handoff between ticks goes unreported.
    subprocess.run([sys.executable, str(TASK/'progress-updates/notify.py')], check=True, timeout=90)
    # Disable only completed task units. The active handoff oneshot exits
    # naturally; stopping it here would interrupt its final receipt/log write.
    subprocess.run(['systemctl','--user','disable','cuteafd-inkling2-quant.service',
        'cuteafd-inkling2-publish.service','cuteafd-inkling2-handoff.service'], check=True, timeout=30)
    subprocess.run(['systemctl','--user','disable','--now','cuteafd-inkling2-progress.timer'], check=True, timeout=30)
    published = load(root/'publication-complete.json')
    print(json.dumps(dict(status='complete', url=published['url'],
        placement=complete.get('one_copy_per_file'), cleanup=cleanup['status'])), flush=True)


if __name__ == '__main__':
    main()

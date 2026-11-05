"""Use the existing SM120 quant image with a private HF4 runtime overlay."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'quantization'))
from kimi_k3.core_k6.container import IMAGE, command as shared_command

RUNTIME = Path.home() / '.cache/cuteafd/quantization/kimi-k31-k2-core-k6/environment/bin/python'


def command(arguments, *, name, memory_gib=88):
    result = shared_command(arguments, name=name, memory_gib=memory_gib,
        build_root=Path.home() / '.cache/cuteafd/builds/kimi-k31-quant/exllamav3')
    result[result.index('--entrypoint') + 1] = str(RUNTIME)
    cache = Path.home() / '.cache/cuteafd/quantization/kimi-k31-k2-core-k6/kernel-cache'
    cache.mkdir(parents=True, exist_ok=True)
    environment = dict(USER=os.environ.get('USER', 'tj'), LOGNAME=os.environ.get('LOGNAME', 'tj'),
        USE_HUB_KERNELS='0', TRANSFORMERS_OFFLINE='1', HF_HUB_OFFLINE='1',
        HF_HOME='/mnt/sparknest/hf-home',
        HF_MODULES_CACHE=str(cache / 'hf-modules'),
        LD_LIBRARY_PATH=str(Path.home() / 'Developer/cuteafd/.venv/lib/python3.12/site-packages/torch/lib') + ':/usr/local/cuda-13.4/lib64',
        TRITON_CACHE_DIR=str(cache / 'triton'), TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'))
    at = result.index(IMAGE)
    for key, value in environment.items():
        result[at:at] = ['-e', f'{key}={value}']
        at += 2
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', required=True)
    p.add_argument('arguments', nargs=argparse.REMAINDER)
    args = p.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    if not arguments:
        p.error('a Python script or -m invocation is required')
    sys.exit(subprocess.call(command(arguments, name=args.name)))


if __name__ == '__main__':
    main()

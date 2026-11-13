"""Task-owned SM120 runtime, bounded RAM and isolated compiler outputs."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from kimi_k3.core_k6.container import command as base_command

REPO = Path(__file__).resolve().parents[2]
TASK = Path.home() / '.cache/cuteafd/quantization/inkling2-k4-mc'
BUILD = Path.home() / '.cache/cuteafd/builds/inkling2-k4-mc'


def command(arguments, *, name, fork=None):
    result = base_command(arguments, name=name, memory_gib=80, build_root=BUILD/'exllamav3')
    result[result.index('--workdir') + 1] = str(REPO)
    result[result.index('--entrypoint') + 1] = str(TASK/'environment/bin/python')
    # The task UV environment borrows these packages read-only; no installation
    # or modification of the user's base environment is needed.
    for index, value in enumerate(result):
        if value.startswith('LD_LIBRARY_PATH='):
            result[index] = ('LD_LIBRARY_PATH=' + str(Path.home() /
                'Developer/cuteafd/.venv/lib/python3.12/site-packages/torch/lib') +
                ':/usr/local/cuda-13.4/lib64')
    image_index = next(i for i, value in enumerate(result) if value.startswith('cuteafd-coordinator-dev:'))
    extra = ['-e', 'PYTHONPATH=' + str(REPO/'quantization'),
             '-e', 'TORCH_EXTENSIONS_DIR=' + str(BUILD/'torch-extensions'),
             '-e', 'CUDA_CACHE_PATH=' + str(BUILD/'cuda-cache')]
    if fork is not None:
        extra += ['-e', 'INKLING_GPTQMODEL_ROOT=' + str(Path(fork).resolve())]
    result[image_index:image_index] = extra
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--fork', type=Path)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    if not arguments:
        parser.error('a Python module or script is required')
    return subprocess.call(command(arguments, name=args.name, fork=args.fork))


if __name__ == '__main__':
    raise SystemExit(main())

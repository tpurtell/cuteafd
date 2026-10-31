"""Bounded architecture-matched execution exposing only the authorized GPU."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
IMAGE = 'cuteafd-coordinator-dev:tc-aa74e55ca2130070'
GPU_UUID = 'GPU-95f8f212-9131-df99-fd53-7535965197d7'
BUILD = Path.home()/'.cache/cuteafd/builds/mistral-quant'


def command(arguments, *, name, cpu=False):
    environment = dict(HOME=str(Path.home()), USER='tj', LOGNAME='tj',
        CUDA_VISIBLE_DEVICES='' if cpu else GPU_UUID,
        CUDA_HOME='/usr/local/cuda-13.3', TORCH_CUDA_ARCH_LIST='12.0',
        TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0', CC='/usr/bin/gcc', CXX='/usr/bin/g++',
        MAX_JOBS='4', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
        USE_HUB_KERNELS='0', PYTORCH_ALLOC_CONF='expandable_segments:True',
        GPTQMODEL_EXLLAMAV3_BUILD_ROOT=str(BUILD/'exllamav3'),
        TRITON_CACHE_DIR=str(BUILD/'triton'), TORCHINDUCTOR_CACHE_DIR=str(BUILD/'inductor'),
        TMPDIR=str(BUILD/'tmp'), PYTHONPATH=str(REPO/'quantization'),
        MISTRAL_QUANT_ATTEMPT=name,
        PATH=str(BUILD/'python/bin')+':/usr/local/cuda-13.3/bin:'+os.environ['PATH'],
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    result = ['docker','run','--init','--name',name,'--network','none',
        '--user',f'{os.getuid()}:{os.getgid()}', '--memory','78g','--memory-swap','78g',
        '--ulimit','nofile=65536:65536','--stop-timeout','30','--workdir',str(REPO),
        '--mount',f'type=bind,src={Path.home()},dst={Path.home()}',
        '--mount','type=bind,src=/mnt/sparknest,dst=/mnt/sparknest,readonly',
        '--mount','type=bind,src=/usr/local/cuda-13.3,dst=/usr/local/cuda-13.3,readonly',
        '--entrypoint',str(REPO/'.venv/bin/python')]
    if not cpu:
        result += ['--device','nvidia.com/gpu='+GPU_UUID]
    else:
        result += ['--rm']
    for key,value in environment.items():
        result += ['-e',key+'='+value]
    return result+[IMAGE,*map(str,arguments)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True)
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('arguments',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    arguments=args.arguments[1:] if args.arguments[:1]==['--'] else args.arguments
    if not arguments:
        parser.error('Python arguments required')
    sys.exit(subprocess.call(command(arguments,name=args.name,cpu=args.cpu)))

if __name__=='__main__':
    main()

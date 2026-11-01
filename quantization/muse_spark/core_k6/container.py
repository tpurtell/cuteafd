"""Bounded quantization in the existing SM120 toolchain, exposing GPU0 only."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
REPO=Path(__file__).resolve().parents[3]
IMAGE='cuteafd-coordinator-dev:tc-aa74e55ca2130070'


def gpu_uuid():
    value=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],
        text=True,timeout=20).strip()
    if not value.startswith('GPU-') or '\n' in value:raise ValueError('cannot identify authorized GPU0')
    return value


def command(arguments,*,name,gpu=True,memory_gib=78):
    home=Path.home();build=home/'.cache/cuteafd/builds/muse-quant'
    cache=home/'Developer/cute-worktrees/muse-quant-work/kernel-cache'
    build.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
    uuid=gpu_uuid()
    image_id=subprocess.check_output(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],text=True,timeout=20).strip()
    result=['docker','run','--rm','--name',name,'--network','none',
        '--user',f'{os.getuid()}:{os.getgid()}','--memory',f'{memory_gib}g','--memory-swap',f'{memory_gib}g',
        '--ulimit','nofile=65536:65536','--stop-timeout','30','--workdir',str(REPO),
        '--mount',f'type=bind,src={home},dst={home}',
        '--mount','type=bind,src=/mnt/sparknest,dst=/mnt/sparknest,readonly',
        '--entrypoint',str(REPO/'.venv/bin/python')]
    if gpu:result+=['--gpus',f'device={uuid}']
    env=dict(HOME=str(home),USER='tj',LOGNAME='tj',CUDA_VISIBLE_DEVICES=uuid if gpu else '',NVIDIA_VISIBLE_DEVICES=uuid if gpu else '',
        MUSE_AUTHORIZED_GPU=uuid,MUSE_QUANT_IMAGE_ID=image_id,
        CUDA_HOME='/usr/local/cuda',TORCH_CUDA_ARCH_LIST='12.0',
        TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0',CC='/usr/bin/gcc',CXX='/usr/bin/g++',MAX_JOBS='8',
        OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',
        USE_HUB_KERNELS='0',PYTORCH_ALLOC_CONF='expandable_segments:True',
        GPTQMODEL_EXLLAMAV3_BUILD_ROOT=str(build/'exllamav3'),TORCH_EXTENSIONS_DIR=str(build/'torch-extensions'),
        TMPDIR=str(build/'tmp'),PYTHONDONTWRITEBYTECODE='1',
        TRITON_CACHE_DIR=str(cache/'triton'),TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),
        PYTHONPATH=str(REPO/'.venv/lib/python3.12/site-packages')+':/usr/local/lib/python3.12/dist-packages',
        LD_LIBRARY_PATH='/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/cuda/lib64')
    (build/'tmp').mkdir(exist_ok=True)
    for key,value in env.items():result+=['-e',f'{key}={value}']
    return result+[IMAGE,*map(str,arguments)]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--name',required=True)
    parser.add_argument('--cpu',action='store_true');parser.add_argument('arguments',nargs=argparse.REMAINDER)
    args=parser.parse_args();rest=args.arguments[1:] if args.arguments[:1]==['--'] else args.arguments
    if not rest:parser.error('Python invocation required')
    sys.exit(subprocess.call(command(rest,name=args.name,gpu=not args.cpu)))

if __name__=='__main__':main()

"""Compile the pinned EXL3 extension with no GPU lease or CUDA device access."""
from __future__ import annotations
from pathlib import Path
import subprocess
import sys
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'quantization'))
from mistral_large4.core_k6.numerical import fork_module
from kimi_k3.uniform_k2.storage import atomic_json,sha256_file


def main():
    from torch.utils.cpp_extension import load
    extension=fork_module('exllamav3.ext')._EXLLAMAV3_TORCH_OPS_EXTENSION
    root=extension.build_root()
    root.mkdir(parents=True,exist_ok=True)
    load(name=extension.name,sources=extension._resolve_sequence(extension.sources),
        build_directory=str(root),is_python_module=False,verbose=True,
        extra_cflags=extension._resolved_extra_cflags(),
        extra_cuda_cflags=extension._resolved_extra_cuda_cflags(),
        extra_include_paths=extension._resolved_extra_include_paths(),
        extra_ldflags=extension._resolve_sequence(extension.extra_ldflags))
    binary=root/(extension.name+'.so')
    if not binary.is_file():
        raise RuntimeError('extension compiler produced no shared object')
    atomic_json(root.parent/'build-receipt.json',dict(binary=str(binary),sha256=sha256_file(binary),
        fork_commit=subprocess.check_output(['git','-C',str(REPO/'third_party/gptqmodel'),'rev-parse','HEAD'],text=True).strip()))

if __name__=='__main__':
    main()

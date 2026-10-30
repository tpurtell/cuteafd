"""Use the existing SM120 quantization image and bounded local environment."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "quantization"))
from kimi_k3.core_k6.container import IMAGE, command as shared_command


def command(arguments, *, name, memory_gib=78):
    # The old wrapper's source/venv path is this same checkout. Keep its CUDA
    # toolchain and JIT directory, including already compiled trellis kernels.
    result = shared_command(arguments, name=name, memory_gib=memory_gib)
    cache = Path.home() / ".cache/cuteafd/quantization/step5-k4-core-k6/kernel-cache"
    cache.mkdir(parents=True, exist_ok=True)
    env = dict(USER=os.environ.get("USER", "tj"), LOGNAME=os.environ.get("LOGNAME", "tj"),
        STEP5_GPTQMODEL_SOURCE=str(REPO / "third_party/gptqmodel"),
        USE_HUB_KERNELS="0", LD_LIBRARY_PATH=str(Path.home()/"Developer/cuteafd/.venv/lib/python3.12/site-packages/torch/lib")+":/usr/local/cuda-13.4/lib64", TRITON_CACHE_DIR=str(cache / "triton"),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"))
    at = result.index(IMAGE)
    for key, value in env.items():
        result[at:at] = ["-e", f"{key}={value}"]
        at += 2
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--memory-gib", type=int, default=78)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    if not arguments:
        parser.error("a Python invocation is required")
    sys.exit(subprocess.call(command(arguments, name=args.name, memory_gib=args.memory_gib)))


if __name__ == "__main__":
    main()

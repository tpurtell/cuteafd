"""Isolated maxq worker for ERNIE reference parity and the EXL3 pilot."""
from pathlib import Path

from hy4_flash.container import command as sm120_command, main

TASK = Path.home() / "Developer/quantization-runs/ernie55-flash-k4-mc"
BUILD = Path.home() / ".cache/cuteafd/builds/ernie55-flash-k4-mc"
FAMILY = "ernie55-flash"


def command(arguments, phase, *, family=FAMILY, task=TASK, build=BUILD):
    argv = sm120_command(arguments, phase, family=family, task=task, build=build)
    home = Path.home()
    mount = f"type=bind,src={home},dst={home}"
    argv[argv.index(mount)] = mount + ",readonly"
    image = next(i for i, part in enumerate(argv) if part.startswith("cuteafd-coordinator-dev:"))
    argv[image:image] = [
        "--hostname", "maxq",
        "--mount", f"type=bind,src={task},dst={task}",
        "--mount", f"type=bind,src={build},dst={build}",
        "--env", f"XDG_CACHE_HOME={build}/xdg",
        "--env", f"HF_MODULES_CACHE={build}/hf-modules",
        "--env", "PYTHONDONTWRITEBYTECODE=1",
        "--env", "GIT_OPTIONAL_LOCKS=0",
        "--env", f"PATH={task}/environment/bin:/usr/local/cuda-13.4/bin:/usr/local/bin:/usr/bin:/bin",
    ]
    if phase == "build":
        device = argv.index("--device")
        del argv[device:device + 2]
        argv[argv.index("CUDA_VISIBLE_DEVICES=0")] = "CUDA_VISIBLE_DEVICES="
        argv.insert(3, "--env=NVIDIA_VISIBLE_DEVICES=void")
    return argv


if __name__ == "__main__":
    raise SystemExit(main(family=FAMILY, task=TASK, build=BUILD, builder=command))

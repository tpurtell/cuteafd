from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ernie55_flash.container import command


@pytest.mark.parametrize("phase", ["build", "preflight", "pilot"])
def test_device_and_memory_policy(monkeypatch, tmp_path, phase):
    monkeypatch.setattr("hy4_flash.container.socket.gethostname", lambda: "maxq")
    task, build = tmp_path / "task", tmp_path / "build"
    argv = command(["-m", "example"], phase, task=task, build=build)
    assert argv[-2:] == ["-m", "example"]
    assert argv[argv.index("--memory") + 1] == argv[argv.index("--memory-swap") + 1] == "80g"
    assert argv[argv.index("--entrypoint") + 1] == str(task / "environment/bin/python")
    for path in (task, build):
        assert f"type=bind,src={path},dst={path}" in argv
    home = Path.home()
    assert f"type=bind,src={home},dst={home},readonly" in argv
    if phase == "build":
        assert "--device" not in argv and "CUDA_VISIBLE_DEVICES=" in argv
    else:
        assert argv[argv.index("--device") + 1] == "nvidia.com/gpu=all"

"""The Kimi K3 uniform K2 launcher takes gpu1 through the hwlock fair queue (#2596)."""
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimi_k3.uniform_k2 import launch


def test_launch_takes_gpu1_through_hwlock_not_flock(tmp_path, monkeypatch):
    root = tmp_path / "state"
    monkeypatch.setattr(sys, "argv", ["launch.py", "--source-snapshot", str(tmp_path / "source"),
                                      "--tokens", str(tmp_path / "tokens"), "--root", str(root),
                                      "--export", str(tmp_path / "export"), "--publish"])
    spawns = []
    monkeypatch.setattr(launch.subprocess, "Popen",
                        lambda command, **kwargs: spawns.append((command, kwargs)) or SimpleNamespace(pid=123456789))
    launch.main()
    latest = json.loads((root / "latest-attempt.json").read_text())
    record = json.loads(Path(latest["launch"]).read_text())
    command = record["command"]
    assert len(spawns) == 1 and spawns[0][1]["start_new_session"] is True
    assert not any("flock" in arg for arg in command)
    assert not any(arg.endswith(".lock") for arg in command)
    assert command[1] == str(launch.REPO / "scripts/hw/hwlock.py")
    separator = command.index("--")
    options = command[:separator]
    assert options[options.index("--locks") + 1] == "gpu1"
    assert options[options.index("--wait") + 1] == "60"
    assert "--preflight-at-grant" in options
    # The granted turn first runs the read-only plan admission of the same run.
    quantizer = command[separator + 1:]
    assert quantizer[:4] == ["timeout", "--foreground", "--signal=TERM", "30d"]
    assert any(arg.endswith("uniform_k2/run.py") for arg in quantizer)
    assert shlex.split(options[options.index("--preflight") + 1]) == [*quantizer[4:], "--plan-only"]
    # Publication runs afterwards, outside the lock, with no GPU visible.
    publication = record["publication_command"]
    assert "--publish-only" in publication and not any("hwlock" in arg or "flock" in arg for arg in publication)

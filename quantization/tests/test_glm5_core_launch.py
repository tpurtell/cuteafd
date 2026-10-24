"""GPU ownership and publication separation in detached production launches."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from glm5.core_k6 import launch


@pytest.mark.parametrize("publication",[False,True])
def test_launch_records_bounded_guarded_quant_or_cpu_only_publish(tmp_path,monkeypatch,publication):
    root=tmp_path/"state"
    argv=["launch.py","--source-snapshot",str(tmp_path/"source"),"--tokens",str(tmp_path/"tokens"),
          "--root",str(root),"--export",str(tmp_path/"export")]
    if publication:
        argv.append("--publish-only")
    checks=[];spawns=[]
    monkeypatch.setattr(sys,"argv",argv)
    monkeypatch.setattr(launch.subprocess,"run",lambda command,**kwargs:checks.append(command))
    def spawn(command,**kwargs):
        spawns.append((command,kwargs));return SimpleNamespace(pid=123456789)
    monkeypatch.setattr(launch.subprocess,"Popen",spawn)
    launch.main()
    latest=json.loads((root/"latest-attempt.json").read_text())
    record=json.loads(Path(latest["launch"]).read_text())
    assert len(checks)==len(spawns)==1 and spawns[0][1]["start_new_session"] is True
    assert str(root) in record["command"]
    assert record["environment"]["HF_HUB_CACHE"].endswith("huggingface/cuteafd-local-hub")
    if publication:
        assert "--publish-only" in record["command"]
        assert record["environment"]["CUDA_VISIBLE_DEVICES"]==""
        assert "--publication-only" in checks[0]
        assert not any("hwlock.py" in arg or "container.py" in arg for arg in record["command"])
    else:
        command=record["command"]
        assert any(arg.endswith("hwlock.py") for arg in command)
        assert command[command.index("--wait")+1]=="60"
        assert command[command.index("--locks")+1]=="gpu1"
        assert "--preflight" in command and "30d" in command
        assert any(arg.endswith("run_guard.py") for arg in command)
        assert command.count(record["container"])==2
        assert command[command.index("--memory-gib")+1]=="60"
        assert "--publish-only" not in command


def test_live_prior_supervisor_prevents_duplicate_launch(tmp_path,monkeypatch):
    root=tmp_path/"state";(root/"attempts").mkdir(parents=True)
    (root/"attempts/old-process.json").write_text(json.dumps(dict(supervisor_pid=777,pid=778)))
    monkeypatch.setattr(sys,"argv",["launch.py","--source-snapshot",str(tmp_path/"source"),
        "--tokens",str(tmp_path/"tokens"),"--root",str(root),"--export",str(tmp_path/"export")])
    monkeypatch.setattr(launch.os,"kill",lambda *a:None)
    monkeypatch.setattr(launch.subprocess,"run",lambda *a,**k:pytest.fail("must not preflight a duplicate"))
    with pytest.raises(ValueError,match="remains active"):
        launch.main()

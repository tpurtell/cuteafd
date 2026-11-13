"""Do not reinterpret sequential checkpoints as legacy scatter output."""
import json
import sys

import pytest

from inkling2 import launch


def fixture(tmp_path,monkeypatch):
    root=tmp_path/'production';root.mkdir()
    (root/'output-io-code.json').write_text('{}')
    monkeypatch.setattr(launch,'TASK',tmp_path)
    return root


def test_active_sequential_backend_refuses_legacy_resume(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(sys,'argv',['launch','--mode','production','--worker'])
    monkeypatch.setattr(launch,'worker',lambda *a:pytest.fail('legacy worker started'))
    with pytest.raises(ValueError,match='refuse legacy'):
        launch.main()


def test_explicit_sequential_worker_keeps_backend(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(sys,'argv',['launch','--mode','production','--worker','--output-io','sequential'])
    seen=[]
    monkeypatch.setattr(launch,'worker',lambda *a:seen.append(a) or 0)
    assert launch.main()==0
    assert seen==[('production','sequential')]


def test_completed_worker_does_not_start_container(tmp_path,monkeypatch):
    root=fixture(tmp_path,monkeypatch)
    (root/'quantization-complete.json').write_text(json.dumps(dict(status='passed')))
    monkeypatch.setattr(sys,'argv',['launch','--mode','production','--worker'])
    monkeypatch.setattr(launch,'worker',lambda *a:pytest.fail('completed job restarted'))
    assert launch.main()==0

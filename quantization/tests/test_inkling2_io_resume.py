"""The operational wrapper respects the frozen lock and completion guards."""
import fcntl
import json
import sys

import pytest

from inkling2 import io_resume


def test_activation_requires_this_process_to_hold_frozen_producer_lock(tmp_path):
    with (tmp_path/'producer.lock').open('a') as lock:
        with pytest.raises(ValueError,match='lock'):
            io_resume.assert_producer_lock(tmp_path)
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        io_resume.assert_producer_lock(tmp_path)
        io_resume.activation_guard(tmp_path,{'frozen':'identity'})
    marker=json.loads((tmp_path/'output-io-wrapper.json').read_text())
    assert marker['frozen_numerical_identity']=={'frozen':'identity'}


def test_completed_job_does_not_activate_adapter(tmp_path,monkeypatch,capsys):
    identity={'frozen':'identity'}
    (tmp_path/'identity.json').write_text(json.dumps(identity))
    (tmp_path/'quantization-complete.json').write_text(json.dumps(dict(status='passed',identity=identity)))
    monkeypatch.setattr(sys,'argv',['io_resume','--root',str(tmp_path)])
    monkeypatch.setattr(io_resume,'OutputNumpyProxy',lambda *a,**k:pytest.fail('adapter activated'))
    io_resume.main()
    assert 'already_complete' in capsys.readouterr().out
    assert not (tmp_path/'output-io-wrapper.json').exists()


def test_wrapper_keeps_original_run_module_and_calls_unchanged_main(tmp_path,monkeypatch):
    from inkling2 import run
    identity={'frozen':'identity'}
    (tmp_path/'identity.json').write_text(json.dumps(identity))
    monkeypatch.setattr(sys,'argv',['io_resume','--root',str(tmp_path)])
    sentinel=object()
    monkeypatch.setattr(io_resume,'OutputNumpyProxy',lambda *a,**k:sentinel)
    monkeypatch.setattr(run,'np',run.np)
    seen=[]
    monkeypatch.setattr(run,'main',lambda:seen.append(run.np))
    io_resume.main()
    assert seen==[sentinel]

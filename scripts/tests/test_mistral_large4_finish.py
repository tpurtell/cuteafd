"""CPU upload and storage retirement preserve their ordered durability barriers."""
import sys
from pathlib import Path
from types import ModuleType

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'quantization'))
from mistral_large4.core_k6 import finish


def fixture(tmp_path,monkeypatch,*,archived=False):
    root=tmp_path/'run'
    root.mkdir()
    if archived:
        (root/'archive-complete.json').write_text('{}')
    args=['finish.py','--source-snapshot',str(tmp_path/'source'),'--tokens',str(tmp_path/'tokens'),
          '--root',str(root),'--export',str(tmp_path/'export'),'--publish-only','--resume']
    monkeypatch.setattr(sys,'argv',args)
    calls=[]
    module=ModuleType('mistral_large4.core_k6.archive')
    module.archive=lambda r:calls.append(('archive',r))
    monkeypatch.setitem(sys.modules,module.__name__,module)
    monkeypatch.setattr(finish.subprocess,'run',lambda command,**kwargs:calls.append(('publish',command,kwargs)))
    return root,calls


def test_upload_succeeds_before_managed_storage_work(tmp_path,monkeypatch):
    root,calls=fixture(tmp_path,monkeypatch)
    finish.main()
    assert [r[0] for r in calls]==['publish','archive']
    command=calls[0][1]
    assert command[1].endswith('/run.py') and command[-1]=='--publish-only'
    assert '--resume' in command
    assert calls[0][2]['check'] is True
    assert calls[1][1]==root


def test_upload_failure_cannot_begin_retirement(tmp_path,monkeypatch):
    root,calls=fixture(tmp_path,monkeypatch)
    def fail(*args,**kwargs):
        raise RuntimeError('upload failed')
    monkeypatch.setattr(finish.subprocess,'run',fail)
    with pytest.raises(RuntimeError,match='upload failed'):
        finish.main()
    assert calls==[]


def test_completed_archive_revalidates_without_retired_publisher_paths(tmp_path,monkeypatch):
    root,calls=fixture(tmp_path,monkeypatch,archived=True)
    finish.main()
    assert calls==[('archive',root)]


def test_published_retry_skips_publisher_and_resumes_native_storage(tmp_path,monkeypatch):
    root,calls=fixture(tmp_path,monkeypatch)
    (root/'publication-complete.json').write_text('{}')
    finish.main()
    assert calls==[('archive',root)]

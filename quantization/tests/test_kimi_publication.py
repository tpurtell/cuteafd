"""Exercise qualified Kimi publication entry points without network/GPU."""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimi_k3.uniform_k2.publish import main


def test_uniform_producer_publication_qualifies_before_remote_mutation(tmp_path, monkeypatch):
    from kimi_k3.uniform_k2 import run
    from huggingface_hub import constants
    import hub_lfs
    import hub_requirements

    events = []

    class UploadReached(Exception):
        pass

    class Api:
        def __init__(self):
            events.append("api")

        def create_repo(self, *args, **kwargs):
            events.append("create")

    def upload(api, repo_id, folder, **kwargs):
        assert folder == tmp_path / "export" and kwargs["state"] == tmp_path / "state"
        assert kwargs["options"] is None and kwargs["digests"] == {}
        events.append("upload")
        raise UploadReached

    real_check = hub_requirements.require_qualified_hub

    def check():
        events.append("qualify")
        real_check()

    (tmp_path / "export").mkdir()
    monkeypatch.setattr(hub_requirements, "require_qualified_hub", check)
    monkeypatch.setattr(hub_lfs, "upload_folder", upload)
    monkeypatch.setattr("huggingface_hub.HfApi", Api)
    with pytest.raises(UploadReached):
        run.publish(tmp_path / "export", tmp_path / "state", "test/model")
    assert events == ["qualify", "api", "create", "upload"]

    events.clear()
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_XET", False)
    with pytest.raises(ValueError, match="HF_HUB_DISABLE_XET=1"):
        run.publish(tmp_path / "export", tmp_path / "state", "test/model")
    assert events == ["qualify"]

    events.clear()
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_XET", True)
    monkeypatch.setattr(hub_requirements.importlib.metadata, "version", lambda _: "2.2.0")
    with pytest.raises(ValueError, match="qualified huggingface_hub 1.33.0"):
        run.publish(tmp_path / "export", tmp_path / "state", "test/model")
    assert events == ["qualify"]


def test_publication_cli_pins_http_lfs_before_producer(monkeypatch):
    import os
    from huggingface_hub import constants
    import runpy
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_XET", False)
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    seen = []
    monkeypatch.setattr(runpy, "run_path", lambda *a, **k: seen.append(
        (constants.HF_HUB_DISABLE_XET, os.environ["HF_HUB_DISABLE_XET"])))
    with patch.object(sys, "argv", ["publish.py", "--publish-only"]):
        main()
    assert seen == [(True, "1")]


def test_publication_entry_point_cannot_start_quantization():
    with patch.object(sys, "argv", ["publish.py"]):
        with pytest.raises(SystemExit, match="requires --publish-only"):
            main()

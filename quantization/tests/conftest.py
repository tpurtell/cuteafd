"""Quant tests run publishers as the publish units do: HTTP LFS, never Xet."""
import pytest


@pytest.fixture(autouse=True)
def http_lfs_publication(monkeypatch):
    from huggingface_hub import constants
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_XET", True)

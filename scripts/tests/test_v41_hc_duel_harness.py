"""CPU contracts for the immutable native V4.1 HC duel reference."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def harness():
    path = Path(__file__).resolve().parents[2] / "python/tools/qualify/deepseek_v41/bench_v41_hc_duel.py"
    spec = importlib.util.spec_from_file_location("bench_v41_hc_duel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_oracle_identity(harness):
    harness.check_oracle()
    assert harness.ROOT == Path(__file__).resolve().parents[2]
    assert harness.ARMS == ("current", "current_control", "candidate")


def test_modified_oracle_rejected(harness, tmp_path):
    modified = tmp_path / "oracle.cu"
    modified.write_bytes(harness.FROZEN.read_bytes() + b"\n")
    harness.FROZEN = modified
    with pytest.raises(RuntimeError, match="frozen oracle changed"):
        harness.check_oracle()

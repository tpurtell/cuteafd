"""Shared official CPU frequency bytes guard both the Rust upload and qualifier."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_muse_upload_frequencies_match_official_cpu_fixture():
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("muse_vision_frequencies",
        root / "python/tools/qualify/muse_spark/qualify-vision.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = [int(line, 16) for line in
        (root / "scripts/tests/fixtures/muse-vision-inv-freq.txt").read_text().splitlines()]
    assert module.axial_inv_freq().view(np.uint32).tolist() == expected
    assert not torch.cuda.is_initialized()

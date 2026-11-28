import pytest
import torch
from safetensors.torch import save_file

from ernie55_flash.pilot_proposals import compare_blocks, install_reconstruction


def test_heldout_proposal_threshold():
    ref = list(range(20))
    assert compare_blocks([("a", ref, ref[:-1] + [42])])["passed"]
    assert not compare_blocks([("a", ref, ref[:-2] + [42, 42])])["passed"]
    with pytest.raises(ValueError):
        compare_blocks([])


def test_only_complete_reconstruction_is_installed(tmp_path):
    path = tmp_path / "core.safetensors"
    save_file({"mtp.x": torch.ones(4, 3)}, path)
    weights = {"x": torch.zeros(4, 3), "y": torch.zeros(1)}
    with pytest.raises(ValueError, match="missing"):
        install_reconstruction(weights, path, "mtp.", {"mtp.x", "mtp.y"})
    assert not weights["x"].any()
    install_reconstruction(weights, path, "mtp.", {"mtp.x"})
    assert weights["x"].all()

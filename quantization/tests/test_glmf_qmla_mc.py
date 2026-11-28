import os
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glm5_flash.qmla_mc import GlmFlashMCSource, fork_module


@pytest.mark.parametrize("view,core_bits", [("original", None), ("k8-compact", 8)])
def test_installed_mul1_core_and_expert_operands(view, core_bits, monkeypatch):
    hub = Path(os.environ.get("HF_HUB_CACHE", Path(os.environ.get("HF_HOME", Path.home()/".cache/huggingface"))/"hub"))
    snap = hub / "models--wrldsuksgo2mars--GLM-5.3-Flash-EXL3-K4-MC-MUL1/snapshots/1bca67feffab2747e0e489e571185bc06336ee26"
    if not snap.is_dir():
        pytest.skip("pinned GLM Flash MUL1 MC is absent")
    reader = GlmFlashMCSource(snap, view)
    core = "model.language_model.layers.0.self_attn.q_proj.weight"
    expert = "model.language_model.layers.3.mlp.experts.0.gate_proj.weight"
    assert reader.native_config().model_type == "glm5_next_text"
    assert reader.variant["name"] == view
    observed = []
    codec = fork_module("utils.kimi_k3_core_format")

    def inspect(pieces, shape, *, device, dtype, codebook):
        details = codec.validate_core_tensors(pieces, shape, codebook=codebook)
        observed.append((shape, details["bits"], codebook, dtype))
        return torch.empty(shape, device="meta", dtype=dtype)

    monkeypatch.setattr(codec, "reconstruct_core_projection", inspect)
    if core_bits is None:
        assert core not in reader.packed
        assert reader.tensor_header(core) == ("BF16", [8192, 4096])
    else:
        assert reader.tensor(core).shape == (8192, 4096)
    assert reader.tensor(expert).shape == (2048, 4096)
    assert observed == ([((8192, 4096), 8, "mul1", torch.bfloat16)] if core_bits else []) + [
        ((2048, 4096), 4, "mul1", torch.bfloat16)]
    reader.close()

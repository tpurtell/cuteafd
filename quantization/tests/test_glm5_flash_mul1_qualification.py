"""Exercise the bank and replay witness on CPU, substituting only the CUDA encoder."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glm5_flash.mul1_mc import numerics, qualification


def test_witness_checks_live_and_sealed_outputs(tmp_path, monkeypatch):
    def encode(name, weight, raw, device):
        quantized = (weight.float() * 16).round().to(torch.int16).cpu()
        values = {"test_codes": quantized}
        return numerics.ProjectionEncoding({4: values}, {4: {}}, reconstruct(values, weight.shape, device))

    def reconstruct(values, shape, device):
        return (values["test_codes"].float() / 16).bfloat16().to(device)

    monkeypatch.setattr(numerics, "encode_expert", encode)
    monkeypatch.setattr(numerics, "reconstruct", reconstruct)
    assert qualification.qualify("cpu", tmp_path)["sealed_replay_equal"]


def test_exactness_checks_signed_zero():
    with pytest.raises(ValueError, match="signed zero"):
        qualification.exact(torch.tensor(-0.0), torch.tensor(0.0), "signed zero")

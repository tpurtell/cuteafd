"""Exercise corpus covariance and the shared-Hessian K8 causal contract on CPU."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glm5_flash.mul1_mc import numerics


def test_raw_hessian_handles_identity_before_normalization():
    inputs = torch.randn(256, 5, generator=torch.Generator().manual_seed(7))
    acc = numerics.covariance_accumulator(5, "cpu")
    acc.add(inputs)
    actual = numerics.raw_hessian(acc, 1024 - len(inputs))
    expected = sum((x.T @ x).double() for x in inputs.split(128)).float()
    torch.testing.assert_close(acc.snapshot(), expected, rtol=0, atol=0)
    expected.diagonal().add_(768)
    torch.testing.assert_close(actual["H"], expected, rtol=0, atol=0)
    assert actual["count"] == 1024 and actual["identity_rows"] == 768


def fake_encoder(monkeypatch, errors):
    calls = []
    def encode(weight, raw, *, bits, seed, sigma, device, codebook):
        assert (seed, sigma, device, codebook) == (787, .025, "cpu", "mul1")
        calls.append((bits, id(raw)))
        return ({"svh": torch.tensor([float(bits)], dtype=torch.float16)},
                {"hessian_weighted_relative_error": errors[bits]}, {"bits": bits})

    def restore(packed, shape, device):
        return torch.full(shape, float(packed["svh"].item()), dtype=torch.bfloat16)

    monkeypatch.setattr(numerics.fork, "quantize_projection", encode)
    monkeypatch.setattr(numerics, "reconstruct", restore)
    return calls


def test_core_encodes_one_capture_and_returns_serialized_k8(monkeypatch):
    calls = fake_encoder(monkeypatch, {8: .00001, 6: .0001})
    raw = {"identity_rows": 0}
    result = numerics.encode_core("proj", torch.ones(2, 2), raw, "cpu",
                                  evaluate=lambda bits, restored: 0.0 if restored[0, 0] == bits else 1.0)
    assert calls == [(8, id(raw)), (6, id(raw))]
    assert torch.all(result.reconstruction == 8)
    assert result.metrics[6]["heldout_nmse"] == 0.0
    assert result.metrics[6]["side"] == 1


def test_rejects_bad_second_encode_before_publication(monkeypatch):
    fake_encoder(monkeypatch, {8: 0., 6: .00101})
    with pytest.raises(ValueError, match="K6 training"):
        numerics.encode_core("proj", torch.ones(2, 2), {}, "cpu")


def test_expert_enforces_recovery_and_heldout_contract(monkeypatch):
    fake_encoder(monkeypatch, {4: .001})
    with pytest.raises(ValueError, match="1024"):
        numerics.encode_expert("expert", torch.ones(2, 2), {"count": 511}, "cpu")
    with pytest.raises(ValueError, match="heldout NMSE"):
        numerics.encode_expert("expert", torch.ones(2, 2), {"count": 1024}, "cpu",
                               evaluate=lambda bits, reconstruction: float("nan"))

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2.storage import (ACCUMULATION_CONTRACT, RouteSpool,
    serialize_attnres, unserialize_attnres)


def numerical():
    path = Path(__file__).resolve().parents[2] / "third_party/gptqmodel/gptqmodel/utils/exl3_hessian.py"
    spec = importlib.util.spec_from_file_location("inkling_test_hessian", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return SimpleNamespace(ACCUMULATION_CONTRACT=module.HESSIAN_ACCUMULATION_CONTRACT,
        covariance_accumulator=lambda width, device: module.BoundedHessianAccumulator(width, device=device, max_rows=128),
        mxfp8_qdq=qdq)


def qdq(x):
    # Deliberately visible rounding makes omission/double-use observable.
    return (x.float().mul(2).round() / 2).to(torch.bfloat16)


def observations():
    x = torch.tensor([[1.125, 2.375], [3.125, 4.875], [5.875, -6.125], [-7.375, 8.125]], dtype=torch.bfloat16)
    ids = torch.arange(8).expand(4, -1).clone()
    candidates = torch.tensor([[8, 9, 10, 11, 12, 13, 14, 15],
                               [9, 8, 10, 11, 12, 13, 14, 15],
                               [10, 8, 9, 11, 12, 13, 14, 15],
                               [11, 8, 9, 10, 12, 13, 14, 15]])
    gamma = torch.arange(32).reshape(4, 8).float() / 32
    candidate_gamma = torch.arange(32, 64).reshape(4, 8).float() / 64
    return x, ids, gamma, candidates, candidate_gamma


def create(tmp_path, minimum=3, router_override=None):
    x, ids, gamma, candidates, candidate_gamma = observations()
    def router(hidden):
        if router_override:
            return router_override(hidden)
        start = 0 if len(hidden) == 1 else 1
        stop = start + len(hidden)
        return ids[start:stop], gamma[start:stop], candidates[start:stop], candidate_gamma[start:stop]
    batches = [(0, {"id": "first"}, {"mlp_input": x[:1]}),
               (1, {"id": "next"}, {"mlp_input": x[1:]})]
    return RouteSpool.create(tmp_path, {"run": "test"}, 4, batches, router, None, 5,
                            width=2, experts=24, minimum=minimum)


def test_spool_preserves_bf16_gamma_and_rank_major_recovery(tmp_path):
    spool = create(tmp_path)
    x, ids, gamma, candidates, candidate_gamma = observations()
    assert torch.equal(spool.gather(np.arange(4), "cpu"), x)
    assert np.array_equal(spool.rows(0)[0], [0, 1, 2, 3])
    assert not len(spool.rows(0)[1])
    assert np.array_equal(spool.rows(9)[1], [1, 0, 2])
    assert not len(spool.rows(9)[0])
    assert torch.equal(spool.gamma(0, np.arange(4), "cpu"), gamma[:, 0])
    assert torch.equal(spool.gamma(9, np.array([1, 0, 2]), "cpu"), candidate_gamma[[1, 0, 2], [0, 1, 2]])
    with pytest.raises(ValueError, match="no unique"):
        spool.gamma(23, np.array([0]), "cpu")
    with pytest.raises(ValueError, match="row outside"):
        spool.gather(np.array([-1]), "cpu")
    spool.close()


def test_hessian_uses_qdq_and_candidate_gamma_before_down_capture(tmp_path):
    spool = create(tmp_path)
    numerical_module = numerical()
    raw, evidence = spool.hessian(9, device="cpu", chunk=2, numerical=numerical_module)
    rows = np.array([1, 0, 2])
    expected_x = qdq(observations()[0][rows]).float()
    expected = (expected_x.T @ expected_x).double().float()
    assert torch.equal(raw["H"], expected)
    assert raw["count"] == 3 and raw["finalized"] is False
    assert evidence["ranks"] == [9, 16]
    called_gamma = []
    def down(x, gamma):
        called_gamma.extend(gamma.tolist())
        return qdq((x[:, :1].float() * gamma[:, None] + 2 * x[:, 1:].float()).to(torch.bfloat16))
    raw, evidence = spool.hessian(9, down, "cpu", 2, numerical=numerical_module, down_width=1)
    gamma = spool.gamma(9, rows, "cpu")
    expected_x = qdq((expected_x[:, :1] * gamma[:, None] + 2 * expected_x[:, 1:]).to(torch.bfloat16)).float()
    assert torch.equal(raw["H"], expected_x.T @ expected_x)
    assert called_gamma == gamma.tolist()
    assert evidence["down_inputs"].endswith("then-mxfp8-qdq")
    assert evidence["accumulation_contract"] == ACCUMULATION_CONTRACT
    spool.close()


def test_cold_recovery_declares_remaining_identity_without_changing_damping(tmp_path):
    spool = create(tmp_path, minimum=6)
    raw, evidence = spool.hessian(9, device="cpu", numerical=numerical())
    x = qdq(observations()[0][[1, 0, 2, 3]]).float()
    assert torch.equal(raw["H"], x.T @ x + 2 * torch.eye(2))
    assert (raw["count"], raw["identity_rows"]) == (6, 2)
    assert (evidence["natural_rows"], evidence["augmented_rows"], evidence["identity_rows"]) == (0, 4, 2)
    cold, evidence = spool.hessian(23, device="cpu", numerical=numerical())
    assert torch.equal(cold["H"], torch.eye(2) * 6)
    assert evidence["identity_rows"] == 6
    spool.close()


def test_resume_checks_receipts_and_does_not_replay_router(tmp_path):
    spool = create(tmp_path)
    spool.close()
    def forbidden(*_):
        raise AssertionError("resume must not recapture")
    resumed = RouteSpool.create(tmp_path,
        {"run": "test"}, 4, None, forbidden, None, 5, width=2, experts=24, minimum=3)
    resumed.close()
    with pytest.raises(ValueError, match="identity/geometry"):
        RouteSpool(tmp_path, {"run": "other"}, 4, 2, 24, 3)
    with pytest.raises(ValueError, match="identity/geometry"):
        RouteSpool(tmp_path, {"run": "test"}, 4, 2, 24, 4)
    path = tmp_path / "gamma.f32"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="committed file"):
        RouteSpool(tmp_path, {"run": "test"}, 4, 2, 24, 3)


@pytest.mark.parametrize("case", ["duplicate", "overlap", "negative_gamma", "nan_gamma", "wrong_dtype"])
def test_invalid_router_contract_never_commits(tmp_path, case):
    x, ids, gamma, near, near_gamma = observations()
    if case == "duplicate":
        ids[:, 1] = ids[:, 0]
    elif case == "overlap":
        near[:, 0] = ids[:, 0]
    elif case == "negative_gamma":
        near_gamma[0, 0] = -1
    elif case == "nan_gamma":
        gamma[0, 0] = float("nan")
    else:
        gamma = gamma.to(torch.float16)
    with pytest.raises(ValueError):
        RouteSpool.create(tmp_path, {}, 4, [(0, {}, {"mlp_input": x})],
            lambda _: (ids, gamma, near, near_gamma), None, 0, width=2, experts=24, minimum=3)
    assert not (tmp_path / "manifest.json").exists()


def test_attnres_roundtrip_preserves_fp32_blocks_without_stacking_or_copying():
    partial = torch.randn(4, 2, dtype=torch.float32)
    blocks = [torch.randn_like(partial), torch.randn(4, 2, dtype=torch.bfloat16)]
    tensors, metadata = serialize_attnres(SimpleNamespace(partial=partial, blocks=blocks), metadata={"phase": "attention"})
    restored = unserialize_attnres(tensors, metadata, state_factory=lambda p, b: SimpleNamespace(partial=p, blocks=b))
    assert restored.partial.data_ptr() == partial.data_ptr()
    assert [v.data_ptr() for v in restored.blocks] == [v.data_ptr() for v in blocks]
    assert restored.partial.dtype == torch.float32
    assert metadata["metadata"] == {"phase": "attention"}
    metadata = json.loads(json.dumps(metadata))
    metadata["tensors"]["partial"]["dtype"] = "torch.bfloat16"
    with pytest.raises(ValueError, match="receipt differs"):
        unserialize_attnres(tensors, metadata)

"""CPU decision, integrity and storage contracts for the bounded Muse pilot."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from muse_spark.core_k6 import pilot


def report():
    rows = []
    for layer in pilot.LAYERS:
        for relative in pilot.CORE:
            name = pilot.full_name(layer, relative)
            shape = ([8192, 4096] if relative.endswith("post_expert_proj") else
                [4096, 8192] if relative.endswith("pre_expert_proj") else
                [1024, 8192] if relative.endswith(("k_proj", "v_proj")) else [8192, 8192])
            for bits in (*pilot.ARMS, 16):
                rows.append(observation(name, shape, bits))
    rows.append(observation("lm_head", [202048, 8192], 6))
    experts = []
    for layer in pilot.LAYERS:
        expert_rows = [dict(name=pilot.full_name(layer, f"mlp.experts.7.{part}"),
            bits=bits, heldout_nmse=.001, calibration_basis="serialized-K4-gate-up")
            for bits in ((4, 5) if layer == 1 else (4,))
            for part in ("gate_proj", "up_proj", "down_proj")]
        experts.append(dict(layer=layer, expert=7, calibration_rows=16, heldout_rows=8,
                            observations=expert_rows))
    proofs = {str(layer): dict(status="passed", byte_exact=True, tokens=800) for layer in pilot.LAYERS}
    proofs["sliding_window"] = dict(status="passed", byte_exact=True, tokens=2345,
        visible_tail_tokens=2048, hessian_rows=0)
    return dict(schema=pilot.SCHEMA, status="passed", identity=dict(schema=pilot.SCHEMA,
        source_repo=pilot.BASE_REPO, source_revision=pilot.BASE_REVISION),
        observations=rows, native_proofs=proofs, experts=experts,
        stored_shapes=pilot.stored_shapes(), attention_gate_decision="keep-BF16")


def observation(name, shape, bits):
    return dict(name=name, shape=shape, bits=bits, heldout_nmse=.0001,
        scale_inclusive_bytes=pilot.payload_bytes(shape, bits),
        tp2_bytes=pilot.tp2_bytes(name, shape, bits), calibration_rows=4764, heldout_rows=1812,
        codec=dict(bits=bits, native_shape=shape,
                   stored_shape=[(shape[0] + 127) // 128 * 128, shape[1]]),
        metrics=dict(quantizer_path="hessian_ldlq"))


def test_weighted_trace_metric_uses_heldout_covariance_and_tiling():
    weight = torch.tensor([[1., 2.], [3., 1.]])
    candidate = weight + torch.tensor([[.1, -.2], [.3, .1]])
    H = torch.tensor([[4., 1.], [1., 2.]])
    error = candidate - weight
    expected = torch.trace(error @ H @ error.T) / torch.trace(weight @ H @ weight.T)
    assert pilot.weighted_nmse(weight, candidate, H, device="cpu", rows=1) == pytest.approx(float(expected))
    with pytest.raises(ValueError, match="zero-energy"):
        pilot.weighted_nmse(weight, candidate, torch.zeros_like(H), device="cpu")


def test_k6_head_has_extra64_storage_rows_and_same_transform_extent():
    geometry = pilot.stored_shapes()["lm_head"]
    assert geometry["out_features"] == 202048
    assert geometry["stored_out_features"] == 202112
    assert geometry["sharded_side"] == 1
    assert geometry["stored_tensors"]["lm_head.trellis"] == ["I16", [512, 12632, 96]]
    assert geometry["stored_tensors"]["lm_head.svh"] == ["F16", [202112]]
    assert pilot.payload_bytes([4096, 4096], 4) == 8388608 + 16384 + 4


def test_complete_report_roundtrips_json(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report()))
    assert pilot.validate_report(path)["status"] == "passed"


@pytest.mark.parametrize("failure", ["missing-native", "short-window", "witness-in-H",
    "nonfinite", "large-k6", "missing-comparison", "head-unpadded", "expert-missing",
    "expert-catastrophic", "expert-down-native", "identity"])
def test_report_rejects_incomplete_or_unsafe_evidence(failure):
    value = deepcopy(report())
    if failure == "missing-native":
        value["native_proofs"].pop("61")
    elif failure == "short-window":
        value["native_proofs"]["sliding_window"]["tokens"] = 2048
    elif failure == "witness-in-H":
        value["native_proofs"]["sliding_window"]["hessian_rows"] = 1
    elif failure == "nonfinite":
        value["observations"][0]["heldout_nmse"] = float("nan")
    elif failure == "large-k6":
        value["observations"][0]["heldout_nmse"] = .00101
    elif failure == "missing-comparison":
        value["observations"].pop()
    elif failure == "head-unpadded":
        value["observations"][-1]["codec"]["stored_shape"][0] = 202048
    elif failure == "expert-missing":
        value["experts"].pop()
    elif failure == "expert-catastrophic":
        value["experts"][0]["observations"][0]["heldout_nmse"] = .011
    elif failure == "expert-down-native":
        value["experts"][0]["observations"][-1]["calibration_basis"] = "native-BF16"
    elif failure == "identity":
        value["identity"]["source_revision"] = "other"
    with pytest.raises(ValueError):
        pilot.validate_report(value)


def test_receipt_rejects_corrupt_cache_and_preserves_incomplete(tmp_path):
    path = tmp_path / "data.safetensors"
    pilot.save_tensor(path, dict(H=torch.eye(2)), {"producer": "fixed"})
    assert torch.equal(pilot.read_tensor(path, {"producer": "fixed"})["H"], torch.eye(2))
    with pytest.raises(ValueError, match="producer"):
        pilot.read_tensor(path, {"producer": "changed"})
    with path.open("r+b") as file:
        file.seek(-1, 2)
        file.write(b"x")
    with pytest.raises(ValueError, match="changed|checksum"):
        pilot.read_tensor(path, {"producer": "fixed"})
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "evidence").write_bytes(b"keep me")
    pilot.preserve_partial(partial)
    old = list(tmp_path.glob("partial.incomplete-*"))
    assert len(old) == 1 and (old[0] / "evidence").read_bytes() == b"keep me"
    assert not list(partial.iterdir())

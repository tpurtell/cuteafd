"""CPU gates for fixed-K7 selector capture and keyed logistic priors."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts/bench/calibrate-draft-confidence.py"
SPEC = importlib.util.spec_from_file_location("draft_confidence", TOOL)
confidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(confidence)
KEY = "glm5_flash/dflash2/fp8-w8a8-r1"


def record(accepted=2, observed=3, proposed=7, request=0):
    return dict(type="draft_confidence", key=KEY, request=request, history=[0.75] * 7,
                features=[[1.0, 0.5, 1.0, 2.0]] * 7,
                proposed=proposed, accepted=accepted, observed=observed)


def trace(tmp_path, rows):
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def test_feature_transform_matches_rust_fixture():
    assert confidence.transform(0.75, 7, [1.0, 0.5, 1.0, 2.0]) == pytest.approx(
        [np.log(3), np.log(2), 0, 1, np.log(3), 1], abs=1e-12)
    for position, features in [(8, [1, .5, 1, 2]), (1, [1, 0, 1, 2]),
                               (1, [1, .5, 1, 2.5]), (1, [1, .5, 3, 2])]:
        with pytest.raises(ValueError, match="invalid selector"):
            confidence.transform(.75, position, features)


def test_fixed_k7_observations_censor_terminal_and_rejected_suffixes(tmp_path):
    path = trace(tmp_path, [record(), record(accepted=2, observed=2, request=1),
                            record(accepted=1, observed=2, proposed=3)])
    x, y, requests = confidence.observations([path], KEY)
    assert x.shape == (5, 6)
    assert y.tolist() == [1, 1, 0, 1, 1]
    assert len(set(requests)) == 2
    for bad in [record(observed=4), record(observed=1), record(observed=2.5),
                record(accepted=True), record(proposed=7.0)]:
        with pytest.raises(ValueError, match="uncensored"):
            confidence.observations([trace(tmp_path, [bad])], KEY)
    with pytest.raises(ValueError, match="differs"):
        confidence.observations([trace(tmp_path, [record()])], "glm5/dflash2/bf16-r1")


def test_synthetic_logistic_fit_converges_and_improves_logloss():
    rng = np.random.default_rng(41)
    x = rng.normal(size=(4000, 6))
    z = .4 + x @ np.asarray([.2, -.5, 1.3, -.8, .1, -.3])
    y = (rng.random(len(x)) < 1 / (1 + np.exp(-z))).astype(float)
    model, loss, iterations = confidence.fit(x, y)
    assert iterations < 100
    assert loss < confidence.logloss(np.full(len(y), y.mean()), y) * .8
    assert len(model["coefficients"]) == len(model["mean"]) == len(model["scale"]) == 6
    assert np.isfinite(list(model.values())[2]).all()
    assert min(model["scale"]) > 0
    with pytest.raises(ValueError, match="regularization"):
        confidence.fit(x, y, regularization=0)


def test_cli_writes_only_numerics_keyed_fit(tmp_path):
    path = trace(tmp_path, [record(request=i) for i in range(32)])
    output = tmp_path / "drafter"
    result = subprocess.run([sys.executable, str(TOOL), "--key", KEY, "--trace", str(path),
                             "--output-dir", str(output)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    fit = json.loads((output / "draft-confidence.fp8-w8a8-r1.json").read_text())
    assert fit["key"] == KEY
    assert fit["schema"] == "cuteafd-draft-confidence-v1"
    assert fit["feature_order"] == confidence.FEATURE_ORDER
    assert fit["training"]["requests"] == 32
    assert fit["training"]["observed_positions"] == 96
    assert len(fit["training"]["traces"][0]["sha256"]) == 64

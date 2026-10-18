"""CPU contracts for the ungrouped Kimi gate and native wide dispatch."""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("router_selftest", ROOT / "native/tests/expert_router_selftest.py")
ROUTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROUTER)


def test_kimi_gate_normalizes_uncorrected_scores_and_orders_ties():
    logits = np.zeros((2, 896), dtype=np.float32)
    logits[1] = np.linspace(-4, 4, 896, dtype=np.float32)
    correction = np.zeros(896, dtype=np.float32)
    correction[880:] = 3
    ids, weights = ROUTER.oracle(logits, correction)
    np.testing.assert_array_equal(ids[0], np.arange(880, 896))
    np.testing.assert_array_equal(ids[1], np.arange(895, 879, -1))
    np.testing.assert_array_equal(weights[0], np.full(16, 1 / 16, dtype=np.float32))
    scores = 1 / (1 + np.exp(-logits[1, ids[1]]))
    np.testing.assert_allclose(weights[1], scores / scores.sum(), rtol=2e-7)
    np.testing.assert_allclose(weights.sum(-1), 1, rtol=2e-7)


def test_kimi_gate_zero_scores_and_low_id_ties():
    ids, weights = ROUTER.oracle(np.full((1, 896), -1000, dtype=np.float32),
                                 np.zeros(896, dtype=np.float32))
    np.testing.assert_array_equal(ids[0], np.arange(16))
    np.testing.assert_array_equal(weights, 0)


def test_existing_kernel_definitions_are_unchanged():
    # Pin the original kernel bodies without depending on Git history in a source archive.
    source = (ROOT / "native/shared/cuda/expert_router.cu").read_text()
    existing = source[source.index("__global__ void score_kernel"):
                      source.index("// Wide sigmoid/sqrtsoftplus routing")]
    assert hashlib.sha256(existing.encode()).hexdigest() == (
        "bcc1c81b94e131be8b640c248d1e31fb5dda48869ca7a9fc7d343994efb9b4f1")


def test_native_wide_dispatch_and_reducer_width():
    source = (ROOT / "native/shared/cuda/expert_router.cu").read_text()
    assert "experts>1024 || topk<1 || topk>16" in source
    assert "tid2eid && (sigmoid || experts>512 || topk>8)" in source
    assert "else if(experts>512 || topk>8)" in source
    assert "select_wide_kernel<true><<<rows,1024,0,s>>>" in source
    assert "select_wide_kernel<false><<<rows,1024,0,s>>>" in source
    assert "CUTEAFD_HIDDEN_CASE(3584, __VA_ARGS__)" in (
        ROOT / "native/shared/cuda/expert_hidden.cuh").read_text()

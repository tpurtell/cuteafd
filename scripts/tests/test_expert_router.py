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


def test_existing_selector_definitions_are_unchanged():
    # Pin retained selectors; V4.1 score projection now belongs to the AOT router.
    source = (ROOT / "native/shared/cuda/expert_router.cu").read_text()
    existing = source[source.index("// Top-k with warp-shuffle argmax rounds."):
                      source.index("// Wide sigmoid/sqrtsoftplus routing")]
    assert hashlib.sha256(existing.encode()).hexdigest() == (
        "14669d34f6d106950fe3f16036faee295e7f33b23c2ad49d138f888e2e1d3a74")


def test_v41_projection_uses_aot_without_native_score_fallback():
    source = (ROOT / "native/shared/cuda/expert_router.cu").read_text()
    router = source[source.index('extern "C" int32_t cuteafd_v41_router('):
                    source.index('extern "C" int32_t cuteafd_v41_router_select_logits(')]
    assert "cuteafd_v41_router_scores_aot(hidden,weight,scores,rows,experts,stream)" in router
    assert "return cudaErrorNotSupported;" in router
    for obsolete in ("score_kernel", "score_rows_kernel", "kScoreRowsMax", "MIN_ROWS"):
        assert obsolete not in source


def _parse_fork_v41_router_skinny_max_rows() -> int:
    import ast

    path = ROOT / "third_party/sparkinfer/b12x/moe/_shared/v41_router.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "V41_ROUTER_SKINNY_MAX_ROWS":
                    value = ast.literal_eval(node.value)
                    assert isinstance(value, int)
                    return value
    raise AssertionError(f"V41_ROUTER_SKINNY_MAX_ROWS not found in {path}")


def _parse_engine_v41_router_skinny_cap(source: str) -> int:
    import re

    router = source[source.index('extern "C" int32_t cuteafd_v41_router('):
                    source.index('extern "C" int32_t cuteafd_v41_router_select_logits(')]
    matches = re.findall(r"if\s*\(\s*rows\s*<=\s*(\d+)\s*\)", router)
    assert len(matches) == 1, f"expected one skinny threshold in cuteafd_v41_router, found {matches}"
    return int(matches[0])


def test_v41_skinny_activation_is_fused_before_selection():
    source = (ROOT / "native/shared/cuda/expert_router.cu").read_text()
    router = source[source.index('extern "C" int32_t cuteafd_v41_router('):
                    source.index('extern "C" int32_t cuteafd_v41_router_select_logits(')]
    assert "if(rows<=256)" in router
    assert "select_fast_kernel<false>" in router
    assert "select_fast_kernel<true>" in router
    exporter = (ROOT / "python/tools/aot/export_b12x_v41_router_aot.py").read_text()
    assert "compile_v41_router_scores_aot(experts=experts, activated=True)" in exporter
    assert "V41_ROUTER_SKINNY_MAX_ROWS" in exporter


def test_v41_router_skinny_cap_matches_fork_constant():
    """Drift guard: engine activation threshold in native/shared/cuda/expert_router.cu
    must match V41_ROUTER_SKINNY_MAX_ROWS in third_party/sparkinfer/b12x/moe/_shared/v41_router.py.
    If they diverge while export_b12x_v41_router_aot.py exports activated=True, rows between
    the two caps either double-apply sqrtsoftplus or skip activation entirely."""
    fork_cap = _parse_fork_v41_router_skinny_max_rows()
    engine_source = (ROOT / "native/shared/cuda/expert_router.cu").read_text()
    engine_cap = _parse_engine_v41_router_skinny_cap(engine_source)
    assert engine_cap == fork_cap == 256, (
        f"V4.1 router skinny cap drift: native/shared/cuda/expert_router.cu has rows<={engine_cap}, "
        f"but third_party/sparkinfer/b12x/moe/_shared/v41_router.py defines V41_ROUTER_SKINNY_MAX_ROWS={fork_cap}"
    )


def test_v41_hc_lagged_cap_matches_decode_ceiling():
    """Drift guard: hc_lagged max rows in v41_fp8.cc, v41_hc.rs and export_b12x_v41_fp8_aot.py
    must cover the full V4.1 decode/verify ceiling (256) so verify waves 81..256 never fall
    back to non-row-invariant hc_mixes_workspace."""
    fp8_cc = (ROOT / "native/families/deepseek_v41/src/v41_fp8.cc").read_text()
    hc_rs = (ROOT / "rust/crates/cuteafd-ffi/src/families/deepseek_v41/v41_hc.rs").read_text()
    exporter = (ROOT / "python/tools/aot/export_b12x_v41_fp8_aot.py").read_text()
    assert "if(rows<1 || rows>256 || scratch_bytes<uint64_t(rows)*8000)" in fp8_cc
    assert "if rows > 256 {" in hc_rs
    assert "'max_rows': 256" in exporter


def test_native_wide_dispatch_and_reducer_width():
    source = (ROOT / "native/shared/cuda/expert_router.cu").read_text()
    assert "experts>1024 || topk<1 || topk>16" in source
    assert "tid2eid && (sigmoid || experts>512 || topk>8)" in source
    assert "else if(experts>512 || topk>8)" in source
    assert "select_wide_kernel<true><<<rows,1024,0,s>>>" in source
    assert "select_wide_kernel<false><<<rows,1024,0,s>>>" in source
    hidden = (ROOT / "native/shared/cuda/expert_hidden.cuh").read_text()
    for width in (3584, 4608):
        assert f"CUTEAFD_HIDDEN_CASE({width}, __VA_ARGS__)" in hidden


def test_v41_reference_qualifier_parses_optional_library():
    path = ROOT / "python/tools/qualify/deepseek_v41/qualify_v41_router_native_abi.py"
    spec = importlib.util.spec_from_file_location("v41_native_abi", path)
    qualifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qualifier)
    base = ["--native-lib", "/candidate.so", "--output", "/result.json"]
    assert qualifier.parse_args(base).reference_native_lib is None
    args = qualifier.parse_args(base + ["--reference-native-lib", "/reference.so"])
    assert args.native_lib == Path("/candidate.so")
    assert args.reference_native_lib == Path("/reference.so")


def test_v41_reference_qualifier_compares_all_fields_bitwise(monkeypatch):
    import sys
    from types import SimpleNamespace
    import pytest

    path = ROOT / "python/tools/qualify/deepseek_v41/qualify_v41_router_native_abi.py"
    spec = importlib.util.spec_from_file_location("v41_native_abi_compare", path)
    qualifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qualifier)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(int32=np.int32, equal=np.array_equal))
    reference = [np.zeros((2, 4), dtype=np.float32), np.zeros((2, 3), dtype=np.int32),
                 np.ones((2, 3), dtype=np.float32)]
    actual = [value[:1].copy() for value in reference]
    qualifier.assert_outputs_bitwise(actual, reference, 128, 1)
    for index, name in enumerate(("scores", "ids", "routing")):
        changed = [value.copy() for value in actual]
        changed[index].view(np.int32)[0, 0] ^= 1
        with pytest.raises(AssertionError) as error:
            qualifier.assert_outputs_bitwise(changed, reference, 128, 1)
        assert error.value.args[0] == (128, 1, name)
    signed_zero = [value.copy() for value in actual]
    signed_zero[0][0, 0] = -0.0
    with pytest.raises(AssertionError):
        qualifier.assert_outputs_bitwise(signed_zero, reference, 128, 1)

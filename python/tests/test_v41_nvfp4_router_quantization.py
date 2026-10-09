"""Keep the unused FP8 producer out of NVFP4 router graphs."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
from cuteafd_reference.quant_ref import (
    decode_packed_nvfp4_values, nvfp4_quantize_static_bytes,
)

ROOT = Path(__file__).resolve().parents[2]


def test_nvfp4_router_skips_fp8_producer():
    source = (ROOT / "rust/crates/cuteafd-daemon/src/families/deepseek_v41/v41_backbone_router.rs").read_text()
    enqueue = source.split("unsafe fn enqueue(", 1)[1].split("pub unsafe fn execute(", 1)[0]
    assert "if !self.weights.nvfp4 {\n                self.input_quantizer" in enqueue
    assert enqueue.count("self.input_quantizer") == 1
    assert "self.b(if self.weights.nvfp4 { 0 } else { 5 })" in enqueue


def test_router_rebind_rejects_quantization_format_change():
    source = (ROOT / "rust/crates/cuteafd-daemon/src/families/deepseek_v41/v41_backbone_router.rs").read_text()
    assert "self.weights.nvfp4 == weights.nvfp4" in source


@pytest.mark.parametrize("input_scale", [0.5, 1.0, 4.0])
def test_per_expert_static_nvfp4_round_trip(input_scale):
    # Known E2M1 values with E4M3 block scale 1: exactly representable
    # under each expert's own global calibration scale.
    codes = [0, 1, 2, 3, 4, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 15]
    packed = bytes(codes[i] | codes[i + 1] << 4 for i in range(0, 16, 2))
    values = decode_packed_nvfp4_values(packed, bytes([0x38]), input_scale, 16)
    payload, scales = nvfp4_quantize_static_bytes(values, input_scale)
    assert payload == packed
    assert scales == bytes([0x38])
    assert decode_packed_nvfp4_values(payload, scales, input_scale, 16) == values


def test_static_nvfp4_nearest_even_midpoints():
    values = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0] * 2
    payload, scales = nvfp4_quantize_static_bytes(values, 1.0)
    assert scales == bytes([0x38])
    assert payload == bytes([0x20, 0x42, 0x64, 0x76] * 2)


def test_static_nvfp4_shared_max_is_not_expert_exact():
    values = [0.0039, 0.017, -0.029, 0.045, 0.083, -0.095, 0.123, 0.2] * 2
    exact = nvfp4_quantize_static_bytes(values, 0.0001)
    shared = nvfp4_quantize_static_bytes(values, 0.0006)
    assert decode_packed_nvfp4_values(*exact, 0.0001, 16) != decode_packed_nvfp4_values(*shared, 0.0006, 16)


@pytest.mark.parametrize("input_scale", [0.0002630324, 0.0004795619, 0.02715774])
def test_static_nvfp4_checkpoint_scale_fixture(input_scale):
    # Recorded FC1 scales from layers 0 and 19 of snapshot 3431dde3.
    packed = bytes([0x10, 0x32, 0x54, 0x76, 0x90, 0xBA, 0xDC, 0xFE])
    values = decode_packed_nvfp4_values(packed, bytes([0x40]), input_scale, 16)
    payload, scales = nvfp4_quantize_static_bytes(values, input_scale)
    assert payload == packed
    assert scales == bytes([0x40])
    assert decode_packed_nvfp4_values(payload, scales, input_scale, 16) == values


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_static_nvfp4_rejects_invalid_scale(scale):
    with pytest.raises(ValueError):
        nvfp4_quantize_static_bytes([1.0] * 16, scale)


def test_nvfp4_scale_option_reaches_both_roles():
    source = (ROOT / "run.sh").read_text()
    assert 'V41_NVFP4_INPUT_SCALE=${V41_NVFP4_INPUT_SCALE:-shared_max}' in source
    assert 'V41_NVFP4_INPUT_SCALE=${23:-shared_max}' in source
    assert '"${V41_NVFP4_INPUT_SCALE:-shared_max}" "${dspark_draft_limit:-auto}"' in source
    common = (ROOT / "scripts/lib/release-common.sh").read_text()
    assert "PROBE_DUMP_ROOT|V41_NVFP4_INPUT_SCALE) return 0" in common

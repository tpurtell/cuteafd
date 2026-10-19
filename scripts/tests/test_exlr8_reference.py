"""CPU contracts for the independently implemented EXLR8 FORMAT.md oracle."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "exlr8_reference", Path(__file__).resolve().parents[2] / "python/reference/exlr8.py")
REF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REF)


@pytest.mark.parametrize("data,expected", [
    (b"", 0xEF46DB3751D8E999), (b"a", 0xD24EC4F1A98C6E5B),
    (b"abc", 0x44BC2CF5AD770999),
])
def test_xxhash_known_vectors(data, expected):
    assert REF.xxhash64(data) == expected


def test_e4m3_rounding_ties_subnormal_saturation():
    values = np.array([0, 2**-10, 3*2**-10, 1.0625, 1.1875, 449, -500], dtype=np.float32)
    np.testing.assert_array_equal(REF.e4m3_round(values), [0, 0, 2**-8, 1, 1.25, 448, -448])


def test_hadamard_is_orthogonal_and_block_local():
    x = np.random.default_rng(4).standard_normal((256, 128), dtype=np.float32)
    np.testing.assert_allclose(REF.hadamard128(REF.hadamard128(x)), x, atol=1e-6)
    impulse = np.zeros(256, dtype=np.float32)
    impulse[0] = 1
    np.testing.assert_array_equal(REF.hadamard128(impulse)[128:], 0)


@pytest.mark.parametrize("k", [2, 3, 4])
def test_zero_tile_has_constant_state_and_fp16_then_e4m3_codebook(k):
    tile = REF.decode_tiles(bytes(32*k), k)
    low = np.array([0x3B60], dtype=np.uint16).view(np.float16)[0]
    expected = REF.e4m3_round(np.float32(np.float16(low + low)))
    assert tile.shape == (1, 16, 16)
    np.testing.assert_array_equal(tile, expected)


@pytest.mark.parametrize("k", [2, 3, 4])
def test_scalar_ring_window_matches_vectorized_tiles(k):
    data = np.random.default_rng(k).integers(0, 256, 32*k, dtype=np.uint8).tobytes()
    words = np.frombuffer(data, dtype="<u2").reshape(-1, 2)[:, ::-1].reshape(-1)
    bits = "".join(f"{word:016b}" for word in words)
    reference = np.empty((16, 16), dtype=np.float32)
    for j in range(256):
        end = (j+1)*k
        state = int("".join(bits[b % len(bits)] for b in range(end-16, end)), 2)
        x = ((state*0xCBAC1FED) & 0x8FFF8FFF) ^ 0x3B603B60
        halves = np.array([x & 65535, x >> 16], dtype=np.uint16).view(np.float16)
        value = REF.e4m3_round(np.float32(np.float16(halves[0] + halves[1])))
        t, i = divmod(j, 8)
        r = 2*(t % 4) + (i % 2) + (8 if i % 4 >= 2 else 0)
        c = t//4 + (8 if i >= 4 else 0)
        reference[r, c] = value
    np.testing.assert_array_equal(REF.decode_tiles(data, k)[0], reference)


def test_mxfp8_preserves_zero_and_exact_powers():
    x = np.zeros((2, 128), dtype=np.float32)
    x[1] = np.tile([1, 2, 4, 8], 32)
    np.testing.assert_array_equal(REF.mxfp8(x), x)


def test_fp8_wire_layout_rounding_and_signed_zero():
    pattern = [1.0625, 1.1875, 2**-10, 3*2**-10, -0.0, 448, -448, 1.0]
    hidden = np.array([pattern * 4 + [1] * 32], dtype=np.float32)
    wire = REF.fp8_k32_wire(hidden)
    assert wire.shape == (1, 66) and wire.dtype == np.uint8
    np.testing.assert_array_equal(wire[0, :32], [0x38, 0x3A, 0, 2, 0x80, 0x7E, 0xFE, 0x38] * 4)
    np.testing.assert_array_equal(wire[0, 32:64], 0x78)
    np.testing.assert_array_equal(wire[0, 64:], [127, 119])
    decoded = REF.fp8_k32_decode(wire)
    np.testing.assert_array_equal(decoded[0, :32], [1, 1.25, 0, 2**-8, -0.0, 448, -448, 1] * 4)
    np.testing.assert_array_equal(decoded[0, 32:], 1)
    assert np.signbit(decoded[0, 4])


def test_fp8_wire_floor_and_scale_boundary():
    hidden = np.zeros((5, 32), dtype=np.float32)
    hidden[1] = np.nextafter(np.float32(0), np.float32(1))
    hidden[2] = np.float32(1e-8)
    hidden[3] = np.float32(448)
    hidden[4] = np.nextafter(np.float32(448), np.float32(np.inf))
    wire = REF.fp8_k32_wire(hidden)
    np.testing.assert_array_equal(wire[:, 32], [105, 105, 105, 127, 128])
    np.testing.assert_array_equal(wire[:2, :32], 0)


def test_fp8_wire_quantization_is_row_independent():
    hidden = np.random.default_rng(408).standard_normal((128, 128), dtype=np.float32)
    wire = REF.fp8_k32_wire(hidden)
    for rows in (1, 8, 64, 65, 128):
        np.testing.assert_array_equal(REF.fp8_k32_wire(hidden[:rows]), wire[:rows])
        np.testing.assert_array_equal(REF.fp8_k32_decode(wire[:rows]), REF.fp8_k32_decode(wire)[:rows])
    for first in (1, 63, 64, 65):
        np.testing.assert_array_equal(REF.fp8_k32_wire(hidden[first:]), wire[first:])
        np.testing.assert_array_equal(REF.fp8_k32_decode(wire[first:]), REF.fp8_k32_decode(wire)[first:])


@pytest.mark.parametrize("hidden", [np.zeros(32), np.zeros((0, 32)), np.zeros((1, 0)),
                                    np.zeros((1, 31)), np.full((1, 32), np.nan),
                                    np.full((1, 32), np.inf)])
def test_fp8_wire_rejects_invalid_activations(hidden):
    with pytest.raises(ValueError, match="FP8 K32 wire requires"):
        REF.fp8_k32_wire(hidden)


@pytest.mark.parametrize("wire", [np.zeros(33, dtype=np.uint8), np.zeros((1, 0), dtype=np.uint8),
                                  np.zeros((1, 32), dtype=np.uint8), np.zeros((1, 33))])
def test_fp8_wire_rejects_invalid_extents(wire):
    with pytest.raises(ValueError, match="FP8 K32 wire requires"):
        REF.fp8_k32_decode(wire)


def test_incomplete_tile_rejected():
    with pytest.raises(ValueError, match="complete"):
        REF.decode_tiles(b"bad", 3)

"""CPU contracts for the bounded, non-B12x EXL3 qualification reference."""
import importlib.util
from pathlib import Path
import struct

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    'exl3_independent_reference',
    Path(__file__).resolve().parents[2] / 'python/tools/lib/exl3_reference.py')
REF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REF)


@pytest.mark.parametrize('bits', [2, 3, 4, 5])
def test_tail_biting_unpack_matches_scalar_bitstream(bits):
    symbols = np.random.default_rng(270 + bits).integers(0, 1 << bits, 256)
    words = []
    for lane in range(16):
        stream = ''.join(f'{value:0{bits}b}' for value in symbols[lane * 16:(lane + 1) * 16])
        words.extend(int(stream[start:start + 16], 2) for start in range(0, len(stream), 16))
    packed = np.asarray(words, dtype=np.uint16).reshape(-1, 2)[:, ::-1].reshape(1, 1, -1)
    actual = REF.decode_inner(packed)
    ordered = []
    for position in range(256):
        history = 0
        for offset in range((16 + bits - 1) // bits - 1, -1, -1):
            history = ((history << bits) | int(symbols[(position - offset) % 256])) & 0xffff
        raw = 0x3b603b60 ^ ((history * 0xcbac1fed) & 0x8fff8fff)
        value = sum(struct.unpack('<ee', struct.pack('<I', raw)))
        lane, component = divmod(position, 8)
        row = (lane % 4) * 2 + (component % 4 // 2) * 8 + component % 2
        column = lane // 4 + (component // 4) * 8
        ordered.append((row, column, value))
    expected = np.empty((16, 16))
    for row, column, value in ordered:
        expected[row, column] = value
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('bits', [2, 3, 4, 5])
def test_effective_weight_includes_both_normalized_rotations(bits):
    packed = np.zeros((8, 8, 16 * bits), dtype=np.uint16)
    suh, svh = np.linspace(0.5, 1.5, 128), np.linspace(0.25, 1.25, 128)
    result = REF.dequantize(packed, suh, svh)
    # Constant inner matrix transforms to one 128-scaled coefficient, not
    # 128*128 or 1; nonconstant scales check their original-coordinate axes.
    expected = np.zeros((128, 128))
    scalar = 2 * struct.unpack('<e', struct.pack('<H', 0x3b60))[0]
    expected[0, 0] = 128 * scalar * suh[0] * svh[0]
    np.testing.assert_allclose(result, expected, atol=1e-12)
    x = np.random.default_rng(bits).normal(size=(3, 256))
    np.testing.assert_allclose(REF.hadamard128(REF.hadamard128(x)), x, atol=2e-15)


@pytest.mark.parametrize('layout', ['contiguous', 'transposed', 'strided'])
def test_hadamard_multiblock_matches_dense_transform(layout):
    values = np.random.default_rng(270).normal(size=(384, 256))
    if layout == 'transposed':
        values = values.T
    elif layout == 'strided':
        values = values[::2]
    hadamard = np.ones((1, 1))
    while len(hadamard) < 128:
        hadamard = np.block([[hadamard, hadamard], [hadamard, -hadamard]])
    expected = (values.reshape(-1, 128) @ (hadamard / np.sqrt(128))).reshape(values.shape)
    np.testing.assert_allclose(REF.hadamard128(values), expected, atol=3e-15)


def test_comparison_rejects_numerical_bug_not_just_bad_cosine():
    values = np.arange(1, 513, dtype=float).reshape(2, 256)
    assert REF.comparison(values * 1.001, values)['passed']
    assert not REF.comparison(values * 1.03, values)['passed']
    assert not REF.comparison(-values, values)['passed']
    with pytest.raises(ValueError, match='zero output'):
        REF.comparison(np.zeros_like(values), values)
    with pytest.raises(ValueError, match='nonfinite'):
        REF.comparison(values * np.nan, values)
    with pytest.raises(ValueError, match='K3/K4'):
        REF.decode_inner(np.zeros((8, 8, 16), dtype=np.uint16))


def test_situ_reference_matches_trained_unclamped_formula(monkeypatch):
    monkeypatch.setattr(REF, 'dequantize', lambda *args: np.eye(128))
    tensors = {(p, s): None for p in ('w1', 'w3', 'w2') for s in ('trellis', 'suh', 'svh')}
    x = np.linspace(-30, 30, 128)[None, :]
    expected = 4 * np.tanh(x / 4) / (1 + np.exp(-x)) * 25 * np.tanh(x / 25)
    np.testing.assert_allclose(REF.expert_output(tensors, x, None, activation='situ'), expected)
    with pytest.raises(ValueError, match='unclamped'):
        REF.expert_output(tensors, x, activation='situ')

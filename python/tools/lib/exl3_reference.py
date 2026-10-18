"""Independent NumPy EXL3 MCG decoder for bounded real-tensor qualification.

Clean port of the Apache-2.0 GPTQModel torch reference at
third_party/gptqmodel/gptqmodel/nn_modules/exllamav3_torch.py (5340775d),
which credits the public ExLlamaV3 tensor format. No B12x decoder, lookup
table, preparation routine or GPU kernel participates in the reference.
"""
from __future__ import annotations

import numpy as np


def decode_inner(trellis):
    """Decode checkpoint [K/16,N/16,16*bits] words to float64 [K,N]."""
    words = np.asarray(trellis).astype(np.uint16)
    if words.ndim != 3 or words.shape[-1] not in (32, 48, 64):
        raise ValueError('reference supports native K2/K3/K4 trellis tensors only')
    bits = words.shape[-1] // 16
    words = words.reshape(*words.shape[:-1], -1, 2)[..., ::-1].reshape(words.shape)
    words = words.reshape(*words.shape[:-1], 16, bits).astype(np.uint32)
    symbols = np.empty((*words.shape[:-2], 256), dtype=np.uint32)
    for position in range(16):
        offset = position * bits
        word, shift = divmod(offset, 16)
        if shift + bits <= 16:
            value = words[..., word] >> (16 - shift - bits)
        else:
            second = shift + bits - 16
            value = (words[..., word] << second) | (words[..., word + 1] >> (16 - second))
        symbols[..., position::16] = value & ((1 << bits) - 1)
    state = np.zeros(symbols.shape[:-1], dtype=np.uint32)
    # Tail-biting history: feed the final symbols before decoding the first.
    for position in range(256 - ((16 + bits - 1) // bits - 1), 256):
        state = ((state << bits) | symbols[..., position]) & 0xffff
    codes = np.empty_like(symbols)
    for position in range(256):
        state = ((state << bits) | symbols[..., position]) & 0xffff
        codes[..., position] = state
    raw = (codes.astype(np.uint64) * 0xcbac1fed).astype(np.uint32)
    raw = np.uint32(0x3b603b60) ^ (raw & np.uint32(0x8fff8fff))
    low = (raw & 0xffff).astype(np.uint16).view(np.float16).astype(np.float64)
    high = (raw >> 16).astype(np.uint16).view(np.float16).astype(np.float64)
    decoded = low + high
    permutation = []
    for lane in range(32):
        row, column = (lane % 4) * 2, lane // 4
        permutation.extend((row * 16 + column, (row + 1) * 16 + column,
                            (row + 8) * 16 + column, (row + 9) * 16 + column,
                            row * 16 + column + 8, (row + 1) * 16 + column + 8,
                            (row + 8) * 16 + column + 8, (row + 9) * 16 + column + 8))
    decoded = decoded[..., np.argsort(permutation)]
    k, n = decoded.shape[:2]
    return decoded.reshape(k, n, 16, 16).transpose(0, 2, 1, 3).reshape(k * 16, n * 16)


def hadamard128(values):
    """Normalized blockwise H128 along the final dimension, in float64."""
    # Reshaping a Fortran-ordered transpose can copy; butterflies must update
    # the storage returned below, including multi-block hidden axes.
    values = np.array(values, dtype=np.float64, copy=True, order='C')
    if values.shape[-1] % 128:
        raise ValueError('EXL3 reference needs H128-aligned dimensions')
    shape = values.shape
    blocks = values.reshape(-1, 128)
    step = 1
    while step < 128:
        pairs = blocks.reshape(-1, 128 // (2 * step), 2, step)
        left, right = pairs[:, :, 0].copy(), pairs[:, :, 1].copy()
        pairs[:, :, 0], pairs[:, :, 1] = left + right, left - right
        step *= 2
    return values.reshape(shape) / np.sqrt(128.0)


def dequantize(trellis, suh, svh):
    """Materialize the checkpoint's independent unrounded effective weight."""
    inner = decode_inner(trellis)
    if inner.shape != (len(suh), len(svh)):
        raise ValueError('trellis and rotation geometry disagree')
    weight = hadamard128(inner.T).T * np.asarray(suh, dtype=np.float64)[:, None]
    return hadamard128(weight) * np.asarray(svh, dtype=np.float64)[None, :]


def expert_output(tensors, inputs, limit=10.0, *, activation='silu'):
    """Dense FP64 dequant/matmul/activation reference for one real expert slice."""
    if activation not in ('silu', 'situ') or (activation == 'situ' and limit is not None):
        raise ValueError('expected silu or unclamped situ activation')
    x = np.asarray(inputs, dtype=np.float64)
    projections = {}
    for projection in ('w1', 'w3'):
        weight = dequantize(*(tensors[projection, suffix] for suffix in ('trellis', 'suh', 'svh')))
        projections[projection] = x @ weight
    gate, up = projections['w1'], projections['w3']
    if limit is not None:
        gate = np.minimum(gate, limit)
        up = np.clip(up, -limit, limit)
    sigmoid = 1 / (1 + np.exp(-np.clip(gate, -700, 700)))
    if activation == 'situ':
        hidden = 4.0 * np.tanh(gate / 4.0) * sigmoid * (25.0 * np.tanh(up / 25.0))
    else:
        hidden = gate * sigmoid * up
    down = dequantize(*(tensors['w2', suffix] for suffix in ('trellis', 'suh', 'svh')))
    return hidden @ down


def comparison(actual, expected):
    """Rounding allowance for FP16 rotations/GEMVs versus unrounded FP64."""
    actual, expected = (np.asarray(value, dtype=np.float64) for value in (actual, expected))
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('nonfinite or mismatched independent-reference outputs')
    norm = np.linalg.norm(expected)
    if norm == 0 or np.linalg.norm(actual) == 0:
        raise ValueError('zero output cannot qualify real expert math')
    relative_l2 = float(np.linalg.norm(actual - expected) / norm)
    cosine = float(np.vdot(actual, expected) / (np.linalg.norm(actual) * norm))
    return dict(relative_l2=relative_l2, cosine=cosine,
                max_abs=float(np.max(np.abs(actual - expected))),
                passed=relative_l2 <= 0.02 and cosine >= 0.9995)
